#!/usr/bin/env bun
/**
 * ZINC Metal Implementation Loop
 *
 * Autonomous loop that iteratively implements the Metal/Apple Silicon inference
 * backend. Each cycle:
 *   1. Build locally (zig build)
 *   2. Run unit tests (zig build test)
 *   3. Run inference with model (zinc -m model.gguf --prompt "..." -n N)
 *   4. Analyze output: build errors? test failures? correct tokens? tok/s?
 *   5. Spawn AI agent to make ONE implementation step
 *   6. Agent edits files → loop back to 1
 *
 * Three phases:
 *   FIX       — build errors, test failures, crashes
 *   IMPLEMENT — wire up GPU layer dispatch, produce correct tokens
 *   OPTIMIZE  — once output matches reference: improve tok/s to ≥80
 *
 * Usage:
 *   bun loops/implement_metal.ts                     # run indefinitely
 *   bun loops/implement_metal.ts --cycles 100        # 100 cycles max
 *   bun loops/implement_metal.ts --dry-run           # build+run only, no agent
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";

// ── Color & display ──────────────────────────────────────────────────

const TTY = process.stdout.isTTY ?? false;
const NO_COLOR = "NO_COLOR" in process.env;
const FORCE_COLOR = process.env.FORCE_COLOR === "1" || process.env.CLICOLOR_FORCE === "1";
const COLOR_ENABLED = !NO_COLOR && (TTY || FORCE_COLOR);

function clr(code: string, text: string): string {
  return COLOR_ENABLED ? `\x1b[${code}m${text}\x1b[0m` : text;
}

const SEP = "─".repeat(64);

// ── Constants ────────────────────────────────────────────────────────

const REPO_ROOT = resolve(import.meta.dir, "..");
const RESULTS_DIR = resolve(REPO_ROOT, ".metal_optimize");
const MODEL_PATH = process.env.ZINC_MODEL ?? "/Users/zolotukhin/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf";
const TEST_PROMPT = "The capital of France is";
const MAX_TOKENS = 5; // Start small, increase as dispatch is wired up
const REFERENCE_TEXT = "Paris"; // Expected in correct output

const BLOCKED_GIT_OPS = [
  "Bash(git checkout:*)",
  "Bash(git revert:*)",
  "Bash(git restore:*)",
  "Bash(git reset:*)",
  "Bash(git stash:*)",
  "Bash(git clean:*)",
];

// ── Phase detection ──────────────────────────────────────────────────

type Phase = "fix" | "implement" | "optimize";

type BuildRunResult = {
  buildExitCode: number;
  buildOutput: string;
  testExitCode: number;
  testOutput: string;
  runExitCode: number | null;
  runOutput: string;
  phase: Phase;
  tokPerSec: number | null;
  tokensGenerated: number;
  outputText: string;
  containsReference: boolean; // true if output contains "Paris"
  error: string | null;
};

function parseTokPerSec(output: string): number | null {
  const m = output.match(/Generated\s+(\d+)\s+tokens\s+in\s+(\d+\.?\d*)\s*(ms|s)/i);
  if (m) {
    const tokens = parseInt(m[1], 10);
    let seconds = parseFloat(m[2]);
    if (m[3] === "ms") seconds /= 1000;
    if (seconds > 0) return tokens / seconds;
  }
  const m2 = output.match(/(\d+\.?\d*)\s*tok\/s/i);
  return m2 ? parseFloat(m2[1]) : null;
}

function parseTokensGenerated(output: string): number {
  const m = output.match(/Generated\s+(\d+)\s+tokens/i);
  return m ? parseInt(m[1], 10) : 0;
}

function parseOutputText(output: string): string {
  const m = output.match(/Output\s*\(\d+\s*tokens?\)\s*:\s*(.+)/i);
  return m ? m[1].trim().slice(0, 200) : "";
}

function detectPhase(result: BuildRunResult): Phase {
  if (result.buildExitCode !== 0) return "fix";
  if (result.testExitCode !== 0) return "fix";
  if (result.runExitCode !== 0 && result.runExitCode !== null) return "fix";
  if (result.error) return "fix";
  // If generating tokens but output doesn't contain reference → implement
  if (result.tokensGenerated > 0 && !result.containsReference) return "implement";
  // If correct output but slow → optimize
  if (result.containsReference && result.tokPerSec != null && result.tokPerSec < 80) return "optimize";
  // If correct and fast → done!
  if (result.containsReference && result.tokPerSec != null && result.tokPerSec >= 80) return "optimize";
  return "implement";
}

// ── Command runner ───────────────────────────────────────────────────

type RunResult = { exitCode: number; stdout: string; stderr: string };

function formatElapsed(startMs: number): string {
  const s = ((Date.now() - startMs) / 1000) | 0;
  if (s < 60) return `${s}s`;
  return `${(s / 60) | 0}m${s % 60}s`;
}

async function runCommand(
  cmd: string,
  args: string[],
  opts: { cwd?: string; timeout?: number; streamOutput?: boolean } = {},
): Promise<RunResult> {
  return new Promise((res) => {
    const child = spawn(cmd, args, {
      cwd: opts.cwd ?? REPO_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
      timeout: opts.timeout,
    });
    let stdout = "", stderr = "";
    child.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stdout += text;
      if (opts.streamOutput) process.stdout.write(text);
    });
    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stderr += text;
      if (opts.streamOutput) process.stderr.write(text);
    });
    child.on("error", () => res({ exitCode: -1, stdout, stderr }));
    child.on("close", (code) => res({ exitCode: code ?? -1, stdout, stderr }));
  });
}

// ── Build, test, and run ─────────────────────────────────────────────

async function buildTestRun(maxTokens: number): Promise<BuildRunResult> {
  console.log(clr("1;33", "  🔨 Building..."));
  const build = await runCommand("zig", ["build"], { timeout: 120_000 });

  if (build.exitCode !== 0) {
    return {
      buildExitCode: build.exitCode,
      buildOutput: build.stderr + build.stdout,
      testExitCode: -1,
      testOutput: "",
      runExitCode: null,
      runOutput: "",
      phase: "fix",
      tokPerSec: null,
      tokensGenerated: 0,
      outputText: "",
      containsReference: false,
      error: "Build failed",
    };
  }
  console.log(clr("1;32", "  ✅ Build OK"));

  console.log(clr("1;33", "  🧪 Testing..."));
  const test = await runCommand("zig", ["build", "test"], { timeout: 120_000 });

  if (test.exitCode !== 0) {
    return {
      buildExitCode: 0,
      buildOutput: build.stderr,
      testExitCode: test.exitCode,
      testOutput: test.stderr + test.stdout,
      runExitCode: null,
      runOutput: "",
      phase: "fix",
      tokPerSec: null,
      tokensGenerated: 0,
      outputText: "",
      containsReference: false,
      error: "Tests failed",
    };
  }
  console.log(clr("1;32", "  ✅ Tests OK"));

  if (!existsSync(MODEL_PATH)) {
    console.log(clr("1;33", "  ⚠ Model not found, skipping inference run"));
    return {
      buildExitCode: 0,
      buildOutput: build.stderr,
      testExitCode: 0,
      testOutput: "",
      runExitCode: null,
      runOutput: "",
      phase: "implement",
      tokPerSec: null,
      tokensGenerated: 0,
      outputText: "",
      containsReference: false,
      error: null,
    };
  }

  console.log(clr("1;33", `  🚀 Running inference (${maxTokens} tokens)...`));
  const run = await runCommand(
    "./zig-out/bin/zinc",
    ["-m", MODEL_PATH, "--prompt", TEST_PROMPT, "-n", String(maxTokens)],
    { timeout: 300_000 },
  );

  const combined = run.stderr + run.stdout;
  const tokPerSec = parseTokPerSec(combined);
  const tokensGenerated = parseTokensGenerated(combined);
  const outputText = parseOutputText(combined);
  const containsReference = outputText.toLowerCase().includes(REFERENCE_TEXT.toLowerCase());

  const result: BuildRunResult = {
    buildExitCode: 0,
    buildOutput: build.stderr,
    testExitCode: 0,
    testOutput: "",
    runExitCode: run.exitCode,
    runOutput: combined,
    phase: "implement",
    tokPerSec,
    tokensGenerated,
    outputText,
    containsReference,
    error: run.exitCode !== 0 ? `Runtime exit code ${run.exitCode}` : null,
  };
  result.phase = detectPhase(result);
  return result;
}

// ── Agent invocation ─────────────────────────────────────────────────

async function runAgent(prompt: string): Promise<RunResult> {
  console.log(clr("1;34", SEP));
  console.log(clr("1;34", "  🧠 AGENT PROMPT"));
  console.log(clr("1;34", SEP));
  const lines = prompt.split("\n");
  for (const line of lines.slice(0, 20)) process.stdout.write(clr("2", line) + "\n");
  if (lines.length > 20) process.stdout.write(clr("2", `… (${lines.length - 20} more lines)\n`));
  console.log(clr("1;34", SEP));

  const startedAt = Date.now();
  const heartbeat = setInterval(() => {
    process.stdout.write(clr("2", `\n⏳ agent running (${formatElapsed(startedAt)})...\n`));
  }, 60_000);

  const result = await runCommand("claude", [
    "-p",
    "--verbose",
    "--output-format", "stream-json",
    `--disallowed-tools=${BLOCKED_GIT_OPS.join(",")}`,
    "--permission-mode", "bypassPermissions",
    "--effort", "high",
    prompt,
  ], { streamOutput: true, timeout: 1_800_000 }); // 30 min

  clearInterval(heartbeat);
  console.log(clr("1;32", `\n  ✅ Agent done in ${formatElapsed(startedAt)}`));
  return result;
}

// ── Prompt builder ───────────────────────────────────────────────────

function buildPrompt(state: RunState, lastResult: BuildRunResult): string {
  const { cycles, failedApproaches, phase } = state;

  const trunc = (s: string, max: number) => s.length > max ? s.slice(0, max) + "…" : s;

  const historyBlock = cycles.length > 0
    ? cycles.slice(-15).map(h => {
        const desc = trunc(h.description, 70);
        return `  #${h.cycle}: [${h.phase}] ${desc} → ${h.kept ? "KEPT" : "REVERTED"}${h.tokPerSec != null ? ` (${h.tokPerSec.toFixed(1)} tok/s)` : ""}${h.containsReference ? " ✅CORRECT" : ""}`;
      }).join("\n")
    : "  (none yet)";

  const failedBlock = failedApproaches.length > 0
    ? failedApproaches.slice(-20).map((f, n) => `  ${n + 1}. ${trunc(f, 120)}`).join("\n")
    : "  (none yet)";

  const ideasBlock = state.ideas.length > 0
    ? state.ideas.slice(-15).map((idea, i) => `  ${i + 1}. ${trunc(idea, 120)}`).join("\n")
    : "  (none yet)";

  const buildOut = lastResult.buildOutput.slice(-2000);
  const testOut = lastResult.testOutput.slice(-2000);
  const runOut = lastResult.runOutput.slice(-3000);

  // Build diagnosis based on phase
  const diagnosis: string[] = [];

  if (lastResult.buildExitCode !== 0) {
    diagnosis.push("## Status: BUILD FAILURE");
    diagnosis.push("Fix the compilation error shown below.");
  } else if (lastResult.testExitCode !== 0) {
    diagnosis.push("## Status: TEST FAILURE");
    diagnosis.push("Fix the failing test. All 27+ Metal tests must pass.");
  } else if (lastResult.runExitCode !== 0 && lastResult.runExitCode !== null) {
    diagnosis.push(`## Status: RUNTIME CRASH (exit code ${lastResult.runExitCode})`);
    diagnosis.push("Build and tests pass but ZINC crashes during inference.");
  } else if (lastResult.tokensGenerated === 0) {
    diagnosis.push("## Status: NO TOKENS GENERATED");
    diagnosis.push("Build and tests pass, model loads, but no tokens are produced.");
  } else if (!lastResult.containsReference) {
    diagnosis.push(`## Status: WRONG OUTPUT — generated ${lastResult.tokensGenerated} tokens but output doesn't contain "Paris"`);
    diagnosis.push(`Output text: "${lastResult.outputText}"`);
    diagnosis.push("");
    diagnosis.push("The 40-layer transform is not fully wired up. The decode loop currently skips");
    diagnosis.push("all layer dispatch and projects raw embeddings directly to logits via CPU matmul.");
    diagnosis.push("You need to implement GPU dispatch for each layer.");
  } else if (lastResult.tokPerSec != null && lastResult.tokPerSec < 80) {
    diagnosis.push(`## Status: CORRECT OUTPUT — ${lastResult.tokPerSec.toFixed(1)} tok/s (target: ≥80)`);
    diagnosis.push(`Output: "${trunc(lastResult.outputText, 80)}"`);
    diagnosis.push("Output is correct! Now optimize for speed. Profile and optimize hot shaders.");
  } else {
    diagnosis.push(`## Status: TARGET REACHED — ${lastResult.tokPerSec?.toFixed(1)} tok/s ≥80`);
    diagnosis.push("🎉 Performance target met!");
  }

  const phaseLabel = phase === "fix" ? "FIX" : phase === "implement" ? "IMPLEMENT" : "OPTIMIZE";

  return [
    `# ZINC Metal ${phaseLabel} Task`,
    "",
    ...diagnosis,
    "",
    "## Hardware",
    "- Mac Studio M4 Max, 64 GB unified memory, 40-core GPU, 546 GB/s bandwidth",
    "- Apple GPU family: Apple9 (M4), simdgroup_matrix = true, bfloat = true",
    "- macOS, no Vulkan — Metal compute only",
    "",
    "## Architecture (Qwen3.5-35B-A3B = hybrid attention + SSM + MoE)",
    "- 40 layers: every 4th is full attention (layers 3,7,11,...,39), rest are SSM/delta-net",
    "- MoE FFN: 256 experts, 8 active per token, + shared expert",
    "- head_dim=256, hidden_dim=2048, n_heads=16, n_kv_heads=2",
    "- rope_dim=64, rope_freq_base=10M",
    "- Model: Q4_K quantization (20.7 GB)",
    "",
    "## Baseline (llama.cpp Metal on this machine)",
    "- Prefill (pp512): 1421 tok/s",
    "- Decode (tg128): 72.93 tok/s",
    "- ZINC target: ≥80 tok/s single-request decode",
    "",
    "## Project Structure",
    "```",
    "src/metal/    — shim.h, shim.m (ObjC C API), c.zig (shared import),",
    "                device.zig, buffer.zig, pipeline.zig, command.zig",
    "src/gpu/      — interface.zig (comptime Metal/Vulkan selection)",
    "src/model/    — config.zig (shared types), gguf.zig, loader_metal.zig, tokenizer.zig",
    "src/compute/  — forward_metal.zig (Metal inference engine — THE MAIN FILE TO EDIT)",
    "src/shaders/metal/ — 18 cross-compiled MSL shaders (dmmv_q4k, flash_attn, rms_norm_mul, etc.)",
    "src/main.zig  — CLI, Metal branch wires device→loader→engine→generate",
    "```",
    "",
    "## What's Working",
    "- ✅ Metal device init (M4 detected, 64 GB memory)",
    "- ✅ Zero-copy model loading (733 tensors, 21 GB mmap'd)",
    "- ✅ Tokenizer (GPT-2 BPE, tokens match llama.cpp)",
    "- ✅ 18 MSL compute shaders cross-compiled from GLSL via SPIRV-Cross",
    "- ✅ Metal buffer alloc, mmap wrapping, pipeline compilation, GPU dispatch",
    "- ✅ GPU dispatch verified: command.zig tests prove compute→readback works",
    "- ✅ CPU LM head matmul produces real logits (slow but functional)",
    "- ✅ 27 tests passing (device, buffer, pipeline, command, dequant, topK)",
    "",
    "## What Needs Implementation",
    "The `generate()` function in `src/compute/forward_metal.zig` currently:",
    "1. Dequants embedding on CPU ✅",
    "2. Uploads to hidden_buf ✅",
    "3. **SKIPS all 40 layers** ← THIS IS THE GAP",
    "4. Projects hidden→logits via CPU matmul (slow but works) ✅",
    "5. Greedy samples and loops ✅",
    "",
    "Each layer needs (via Metal GPU dispatch):",
    "- RMS norm (hidden → norm_buf) using rms_norm_mul.metal",
    "- QKV projection (DMMV: weight × norm → q/k/v) using dmmv_q4k.metal",
    "- For attention layers: deinterleave Q+gate, sigmoid gate, RoPE, KV cache write, flash_attn",
    "- For SSM layers: conv1d, delta-net state update, gated norm",
    "- MoE FFN: router logits → topK → per-expert gate+up (DMMV) → SwiGLU → down → accumulate",
    "- Residual add (vadd.metal)",
    "- Final: RMS norm → LM head DMMV → logits",
    "",
    "The Metal shim's `mtl_dispatch()` takes: pipeline, grid[3], block[3], buffers[], push_constants.",
    "Push constants are passed as a buffer at index n_bufs (see shim.m contract).",
    "Barriers between dependent dispatches use `mtl_barrier()` (creates new encoder).",
    "",
    "## Key Reference: forward.zig (Vulkan version)",
    "The Vulkan `decodeStep()` at src/compute/forward.zig:1032 shows the exact layer dispatch",
    "sequence. Read it for the correct order of operations, tensor names, and dimensions.",
    "Adapt the logic for Metal dispatch (different API but same math).",
    "",
    ...(lastResult.buildOutput ? [
      "## Build Output (last 2000 chars)",
      "```", buildOut, "```", "",
    ] : []),
    ...(lastResult.testOutput ? [
      "## Test Output (last 2000 chars)",
      "```", testOut, "```", "",
    ] : []),
    ...(lastResult.runOutput ? [
      "## Run Output (last 3000 chars)",
      "```", runOut, "```", "",
    ] : []),
    "## Cycle History",
    historyBlock,
    "",
    "## Failed Approaches",
    failedBlock,
    "",
    "## Ideas",
    ideasBlock,
    "",
    "## Rules",
    `1. Make ONE focused change. ${phase === "implement" ? "Implement ONE aspect of the layer dispatch (e.g., just RMS norm + DMMV for one layer)." : ""}`,
    "2. All 27+ tests must continue passing. Run `zig build test` mentally before saving.",
    "3. Do NOT modify src/vulkan/, loops/, or .env.",
    "4. Zig 0.15.2 API: ArrayList is unmanaged (pass allocator to append/deinit).",
    "5. MSL shaders use 'main0' as entry point (SPIRV-Cross convention).",
    "6. Metal push constants go in buffer[n_bufs] (see shim.m mtl_dispatch).",
    "7. The Metal command pattern: beginCommand → dispatch → barrier → dispatch → commitAndWait.",
    "8. UMA advantage: all buffers are SharedMode — cpu_ptr gives direct CPU access to GPU data.",
    "9. Start simple: get one layer type working, then expand to all 40.",
    "10. For DMMV: the cross-compiled shader expects buffers in the same order as the GLSL version.",
    "",
    "## Output Format",
    "After making your change, print these 3 lines:",
    "@@@DESCRIPTION: <one-line summary>",
    "@@@SELF_ANALYSIS: <why this approach and what you expect>",
    "@@@NEXT_IDEAS: <comma-separated ideas for future cycles>",
  ].join("\n");
}

// ── State ────────────────────────────────────────────────────────────

type CycleResult = {
  cycle: number;
  timestamp: string;
  phase: Phase;
  description: string;
  kept: boolean;
  tokPerSec: number | null;
  tokensGenerated: number;
  containsReference: boolean;
  buildExitCode: number;
  testExitCode: number;
  runExitCode: number | null;
  outputText: string;
  error?: string;
  selfAnalysis: string;
  nextIdeas: string[];
};

type RunState = {
  runId: string;
  cycles: CycleResult[];
  failedApproaches: string[];
  ideas: string[];
  phase: Phase;
  currentBest: { tokPerSec: number | null; containsReference: boolean } | null;
};

async function loadState(runDir: string): Promise<RunState | null> {
  const p = join(runDir, "state.json");
  if (!existsSync(p)) return null;
  return JSON.parse(await readFile(p, "utf8")) as RunState;
}

async function saveState(runDir: string, state: RunState): Promise<void> {
  await writeFile(join(runDir, "state.json"), JSON.stringify(state, null, 2));
}

function extractAgentText(stdout: string): string {
  // Extract text from Claude stream-json format
  const lines = stdout.split("\n");
  const texts: string[] = [];
  for (const line of lines) {
    try {
      const evt = JSON.parse(line);
      if (evt?.type === "assistant") {
        const content = evt?.message?.content;
        if (Array.isArray(content)) {
          for (const block of content) {
            if (block?.type === "text" && typeof block.text === "string") {
              texts.push(block.text);
            }
          }
        }
      }
    } catch { /* not JSON */ }
  }
  return texts.join("\n");
}

// ── Main loop ────────────────────────────────────────────────────────

async function main() {
  const args = process.argv.slice(2);
  const maxCycles = parseInt(args.find((_, i, a) => a[i - 1] === "--cycles") ?? "999", 10);
  const dryRun = args.includes("--dry-run");

  const runId = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const runDir = join(RESULTS_DIR, runId);
  await mkdir(runDir, { recursive: true });

  console.log(clr("1;36", "╔══════════════════════════════════════════════════════════════╗"));
  console.log(clr("1;36", "║  ZINC Metal Implementation Loop                              ║"));
  console.log(clr("1;36", `║  Run: ${runId}                                  ║`));
  console.log(clr("1;36", `║  Max cycles: ${maxCycles}  |  Model: ${MODEL_PATH.split("/").pop()}  ║`));
  console.log(clr("1;36", "╚══════════════════════════════════════════════════════════════╝"));

  let state: RunState = {
    runId,
    cycles: [],
    failedApproaches: [],
    ideas: [],
    phase: "implement",
    currentBest: null,
  };

  for (let cycle = 1; cycle <= maxCycles; cycle++) {
    console.log(clr("1;35", "\n" + "═".repeat(64)));
    console.log(clr("1;35", `  CYCLE ${cycle}`));
    console.log(clr("1;35", "═".repeat(64)));

    const cycleDir = join(runDir, `cycle-${String(cycle).padStart(3, "0")}`);
    await mkdir(cycleDir, { recursive: true });

    // Increase max tokens as implementation progresses
    const currentMaxTokens = state.currentBest?.containsReference ? 32 : MAX_TOKENS;

    // Step 1: Build + Test + Run
    const result = await buildTestRun(currentMaxTokens);
    state.phase = result.phase;

    await writeFile(join(cycleDir, "build.log"), result.buildOutput);
    await writeFile(join(cycleDir, "test.log"), result.testOutput);
    await writeFile(join(cycleDir, "run.log"), result.runOutput);

    // Display status
    if (result.buildExitCode !== 0) {
      console.log(clr("1;31", `  ❌ BUILD FAILED`));
    } else if (result.testExitCode !== 0) {
      console.log(clr("1;31", `  ❌ TESTS FAILED`));
    } else if (result.runExitCode !== 0 && result.runExitCode !== null) {
      console.log(clr("1;31", `  ❌ CRASH (exit ${result.runExitCode})`));
    } else {
      const refTag = result.containsReference ? clr("1;32", " ✅CORRECT") : clr("1;33", " ❌WRONG");
      const tpsTag = result.tokPerSec ? ` ${result.tokPerSec.toFixed(1)} tok/s` : "";
      console.log(clr("1;32", `  ✅ ${result.tokensGenerated} tokens${tpsTag}`) + refTag);
      if (result.outputText) console.log(clr("2", `  Output: "${result.outputText.slice(0, 80)}"`));
    }

    if (dryRun) {
      console.log(clr("1;33", "  (dry-run mode — skipping agent)"));
      continue;
    }

    // Step 2: Git snapshot
    await runCommand("git", ["add", "-A", "src/", "build.zig"]).catch(() => {});
    await runCommand("git", ["commit", "--allow-empty", "-m", `metal-loop: pre-cycle-${cycle}`]).catch(() => {});
    const preCommit = await runCommand("git", ["rev-parse", "HEAD"]);
    const preHash = preCommit.stdout.trim();

    // Step 3: Run agent
    const prompt = buildPrompt(state, result);
    await writeFile(join(cycleDir, "prompt.md"), prompt);

    const agentResult = await runAgent(prompt);
    await writeFile(join(cycleDir, "agent.log"), agentResult.stdout + agentResult.stderr);

    // Extract markers
    const agentText = extractAgentText(agentResult.stdout);
    const lastChars = agentText.slice(-3000);
    const descMatch = lastChars.match(/@@@DESCRIPTION:\s*(.+)/im);
    const analysisMatch = lastChars.match(/@@@SELF_ANALYSIS:\s*(.+)/im);
    const ideasMatch = lastChars.match(/@@@NEXT_IDEAS:\s*(.+)/im);
    const description = descMatch?.[1]?.trim() ?? "Agent made changes";
    const selfAnalysis = analysisMatch?.[1]?.trim() ?? "";
    const newIdeas = ideasMatch?.[1]?.split(",").map(s => s.trim()).filter(s => s.length > 3) ?? [];

    // Step 4: Verify
    console.log(clr("1;33", "\n  📊 Verifying..."));
    const verify = await buildTestRun(currentMaxTokens);
    await writeFile(join(cycleDir, "verify.log"), JSON.stringify(verify, null, 2));

    // Keep/revert decision
    let kept = false;
    if (verify.buildExitCode !== 0 || verify.testExitCode !== 0) {
      // Build or test broken → revert
      console.log(clr("1;31", `  ↩ REVERTING — ${verify.buildExitCode !== 0 ? "build" : "tests"} broken`));
      await runCommand("git", ["reset", "--hard", preHash]);
      state.failedApproaches.push(`${description} — broke ${verify.buildExitCode !== 0 ? "build" : "tests"}`);
    } else if (verify.runExitCode !== 0 && verify.runExitCode !== null) {
      // Crash → revert
      console.log(clr("1;31", `  ↩ REVERTING — runtime crash`));
      await runCommand("git", ["reset", "--hard", preHash]);
      state.failedApproaches.push(`${description} — runtime crash`);
    } else {
      // Check improvement
      const prevRef = state.currentBest?.containsReference ?? false;
      const prevTps = state.currentBest?.tokPerSec ?? 0;

      if (verify.containsReference && !prevRef) {
        // Gained correctness!
        kept = true;
        console.log(clr("1;32", `  ✅ KEPT — gained correct output!`));
      } else if (verify.tokPerSec != null && verify.tokPerSec > prevTps + 1) {
        // Significant speed improvement
        kept = true;
        console.log(clr("1;32", `  ✅ KEPT — ${verify.tokPerSec.toFixed(1)} tok/s (was ${prevTps.toFixed(1)})`));
      } else if (verify.tokensGenerated > (state.currentBest as any)?.tokensGenerated ?? 0) {
        // More tokens (progress even if not correct yet)
        kept = true;
        console.log(clr("1;32", `  ✅ KEPT — ${verify.tokensGenerated} tokens (progress)`));
      } else {
        // No improvement → still keep if it doesn't make things worse
        kept = true;
        console.log(clr("1;33", `  ✅ KEPT — no regression`));
      }

      if (kept) {
        state.currentBest = {
          tokPerSec: verify.tokPerSec,
          containsReference: verify.containsReference,
        };
        await runCommand("git", ["add", "-A", "src/", "build.zig"]).catch(() => {});
        await runCommand("git", ["commit", "-m", `metal-loop: cycle-${cycle} ${description}`]).catch(() => {});
      }
    }

    // Update state
    const cycleResult: CycleResult = {
      cycle,
      timestamp: new Date().toISOString(),
      phase: verify.phase,
      description,
      kept,
      tokPerSec: verify.tokPerSec,
      tokensGenerated: verify.tokensGenerated,
      containsReference: verify.containsReference,
      buildExitCode: verify.buildExitCode,
      testExitCode: verify.testExitCode,
      runExitCode: verify.runExitCode,
      outputText: verify.outputText,
      selfAnalysis,
      nextIdeas: newIdeas,
    };

    state.cycles.push(cycleResult);
    for (const idea of newIdeas) {
      if (!state.ideas.includes(idea)) state.ideas.push(idea);
    }

    await saveState(runDir, state);

    // Check if we're done
    if (verify.containsReference && verify.tokPerSec != null && verify.tokPerSec >= 80) {
      console.log(clr("1;32", "\n" + "🎉".repeat(30)));
      console.log(clr("1;32", `  TARGET REACHED: ${verify.tokPerSec.toFixed(1)} tok/s with correct output!`));
      console.log(clr("1;32", "🎉".repeat(30)));
      break;
    }
  }

  console.log(clr("1;36", `\nLoop complete. Results: ${runDir}`));
  console.log(clr("1;36", `Total cycles: ${state.cycles.length}`));
  console.log(clr("1;36", `Kept: ${state.cycles.filter(c => c.kept).length}`));
  console.log(clr("1;36", `Best: ${state.currentBest?.tokPerSec?.toFixed(1) ?? "N/A"} tok/s, correct=${state.currentBest?.containsReference ?? false}`));
}

main().catch((err) => {
  console.error("Fatal error:", err);
  process.exit(1);
});
