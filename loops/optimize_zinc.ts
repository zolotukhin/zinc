#!/usr/bin/env bun
/**
 * ZINC Self-Improving Optimization Loop
 *
 * Iteratively builds, deploys, and improves the ZINC inference engine on an
 * RDNA4 test node. Each cycle:
 *   1. rsync source to remote node
 *   2. Build (zig build)
 *   3. Run (zinc --prompt ...)
 *   4. Analyze output (build errors? runtime crash? tok/s?)
 *   5. Spawn AI agent to make ONE fix or optimization
 *   6. Agent edits LOCAL files → loop back to 1
 *
 * Two phases:
 *   FIX   — build errors, shader compilation, Vulkan errors, crashes
 *   OPTIMIZE — once running: improve tok/s, bandwidth utilization
 *
 * Usage:
 *   bun loops/optimize_zinc.ts                           # run indefinitely
 *   bun loops/optimize_zinc.ts --cycles 50               # 50 cycles max
 *   bun loops/optimize_zinc.ts --dry-run                 # build+run only, no agent
 *   bun loops/optimize_zinc.ts --model-path /root/m.gguf # custom model path
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import {
  isGarbageString,
  formatElapsed,
  extractTextFromStreamJson,
} from "./optimize_llm_tps";

// ── Color & display ──────────────────────────────────────────────────

const TTY = process.stdout.isTTY ?? false;
const NO_COLOR = "NO_COLOR" in process.env;
const FORCE_COLOR =
  process.env.FORCE_COLOR === "1" || process.env.CLICOLOR_FORCE === "1";
const COLOR_ENABLED = !NO_COLOR && (TTY || FORCE_COLOR);

function clr(code: string, text: string): string {
  return COLOR_ENABLED ? `\x1b[${code}m${text}\x1b[0m` : text;
}

const SEP = "─".repeat(64);

// ── Constants ────────────────────────────────────────────────────────

const PROJECT_ROOT = resolve(import.meta.dir, "..");
const RESULTS_DIR = resolve(PROJECT_ROOT, ".zinc_optimize");

// Load .env
function loadEnv(): Record<string, string> {
  const envPath = join(PROJECT_ROOT, ".env");
  const vars: Record<string, string> = {};
  if (existsSync(envPath)) {
    const content = require("fs").readFileSync(envPath, "utf8") as string;
    for (const line of content.split("\n")) {
      const m = line.match(/^\s*([A-Z_]+)\s*=\s*(.+?)\s*$/);
      if (m) vars[m[1]] = m[2];
    }
  }
  return vars;
}

const ENV = loadEnv();
const ZINC_HOST = process.env.ZINC_HOST ?? ENV.ZINC_HOST ?? "127.0.0.1";
const ZINC_PORT = Number(process.env.ZINC_PORT ?? ENV.ZINC_PORT ?? "22");
const ZINC_USER = process.env.ZINC_USER ?? ENV.ZINC_USER ?? "root";
const REMOTE_ZINC_DIR = "/root/zinc";
const DEFAULT_MODEL = "/root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf";

const BLOCKED_GIT_OPS = [
  "Bash(git checkout:*)",
  "Bash(git revert:*)",
  "Bash(git restore:*)",
  "Bash(git reset:*)",
  "Bash(git stash:*)",
  "Bash(git clean:*)",
];

type AgentKind = "claude" | "codex";

// ── Phase detection ──────────────────────────────────────────────────

export type Phase = "fix" | "optimize";

export type BuildRunResult = {
  buildExitCode: number;
  buildOutput: string;
  runExitCode: number | null;
  runOutput: string;
  phase: Phase;
  tokPerSec: number | null;
  tokensGenerated: number;
  error: string | null;
};

/** Parse tok/s from ZINC stderr output. */
export function parseTokPerSec(output: string): number | null {
  // Look for patterns like "Generated N tokens" with timing info
  // or "110.5 tok/s" style output
  const m = output.match(/(\d+\.?\d*)\s*tok\/s/i);
  if (m) return parseFloat(m[1]);
  // Fallback: look for timing in logs
  const genMatch = output.match(/Generated\s+(\d+)\s+tokens/i);
  const timeMatch = output.match(/in\s+(\d+\.?\d*)\s*(?:ms|s)/i);
  if (genMatch && timeMatch) {
    const tokens = parseInt(genMatch[1], 10);
    let seconds = parseFloat(timeMatch[1]);
    if (timeMatch[0].includes("ms")) seconds /= 1000;
    if (seconds > 0) return tokens / seconds;
  }
  return null;
}

/** Parse number of tokens generated from ZINC output. */
export function parseTokensGenerated(output: string): number {
  const m = output.match(/Generated\s+(\d+)\s+tokens/i);
  return m ? parseInt(m[1], 10) : 0;
}

/** Determine if we're in fix or optimize phase. */
export function detectPhase(result: BuildRunResult): Phase {
  if (result.buildExitCode !== 0) return "fix";
  if (result.runExitCode !== 0) return "fix";
  if (result.error) return "fix";
  if (result.tokPerSec != null && result.tokPerSec > 0) return "optimize";
  return "fix";
}

// ── Command runner ───────────────────────────────────────────────────

type RunResult = { exitCode: number; stdout: string; stderr: string };

async function runCommand(
  cmd: string,
  args: string[],
  opts: {
    cwd?: string;
    env?: NodeJS.ProcessEnv;
    streamOutput?: boolean;
    timeout?: number;
  } = {},
): Promise<RunResult> {
  const streamOutput = opts.streamOutput ?? false;
  return new Promise((res, rej) => {
    const child = spawn(cmd, args, {
      cwd: opts.cwd ?? PROJECT_ROOT,
      env: opts.env ?? process.env,
      stdio: ["ignore", "pipe", "pipe"],
      timeout: opts.timeout,
    });
    let stdout = "",
      stderr = "";
    child.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stdout += text;
      if (streamOutput) process.stdout.write(text);
    });
    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stderr += text;
      if (streamOutput) process.stderr.write(text);
    });
    child.on("error", rej);
    child.on("close", (code) => {
      res({ exitCode: code ?? 1, stdout, stderr });
    });
  });
}

// ── SSH & rsync ──────────────────────────────────────────────────────

async function ssh(command: string, timeout = 120_000): Promise<string> {
  const { stdout, stderr, exitCode } = await runCommand(
    "ssh",
    [
      "-o", "StrictHostKeyChecking=no",
      "-o", "ConnectTimeout=10",
      "-p", String(ZINC_PORT),
      `${ZINC_USER}@${ZINC_HOST}`,
      command,
    ],
    { streamOutput: false, timeout },
  );
  if (exitCode !== 0 && !stderr.includes("Warning")) {
    throw new Error(`SSH failed (${exitCode}): ${stderr.slice(0, 500)}`);
  }
  return stdout.trim();
}

async function rsyncToRemote(): Promise<void> {
  console.log(clr("2", "  Syncing source to remote node..."));
  const { exitCode, stderr } = await runCommand(
    "rsync",
    [
      "-avz", "--delete",
      "-e", `ssh -p ${ZINC_PORT} -o StrictHostKeyChecking=no`,
      "--exclude", ".zig-cache",
      "--exclude", "zig-out",
      "--exclude", "zig-cache",
      "--exclude", "node_modules",
      "--exclude", ".git",
      "--exclude", ".zinc_optimize",
      "--exclude", ".llm_optimize",
      "--exclude", ".DS_Store",
      "--exclude", "site",
      "--exclude", "research/turboquant-pytorch-master",
      `${PROJECT_ROOT}/`,
      `${ZINC_USER}@${ZINC_HOST}:${REMOTE_ZINC_DIR}/`,
    ],
    { timeout: 120_000 },
  );
  if (exitCode !== 0) {
    throw new Error(`rsync failed: ${stderr.slice(0, 500)}`);
  }
}

// ── Remote build & run ───────────────────────────────────────────────

async function remoteBuild(): Promise<{ exitCode: number; output: string }> {
  console.log(clr("2", "  Building on remote node..."));
  const { stdout, stderr, exitCode } = await runCommand(
    "ssh",
    [
      "-p", String(ZINC_PORT),
      "-o", "StrictHostKeyChecking=no",
      `${ZINC_USER}@${ZINC_HOST}`,
      `cd ${REMOTE_ZINC_DIR} && zig build 2>&1`,
    ],
    { streamOutput: true, timeout: 300_000 },
  );
  return { exitCode, output: stdout + "\n" + stderr };
}

async function remoteRun(
  modelPath: string,
  prompt: string,
): Promise<{ exitCode: number; output: string }> {
  console.log(clr("2", "  Running ZINC on remote node..."));
  const { stdout, stderr, exitCode } = await runCommand(
    "ssh",
    [
      "-p", String(ZINC_PORT),
      "-o", "StrictHostKeyChecking=no",
      `${ZINC_USER}@${ZINC_HOST}`,
      `cd ${REMOTE_ZINC_DIR} && RADV_PERFTEST=coop_matrix timeout 60 ./zig-out/bin/zinc -m ${modelPath} --prompt "${prompt}" 2>&1`,
    ],
    { streamOutput: true, timeout: 120_000 },
  );
  return { exitCode, output: stdout + "\n" + stderr };
}

async function buildAndRun(modelPath: string): Promise<BuildRunResult> {
  // Build
  const build = await remoteBuild();
  if (build.exitCode !== 0) {
    return {
      buildExitCode: build.exitCode,
      buildOutput: build.output,
      runExitCode: null,
      runOutput: "",
      phase: "fix",
      tokPerSec: null,
      tokensGenerated: 0,
      error: "Build failed",
    };
  }

  // Run
  const run = await remoteRun(modelPath, "The capital of France is");
  const tps = parseTokPerSec(run.output);
  const tokensGenerated = parseTokensGenerated(run.output);

  const result: BuildRunResult = {
    buildExitCode: 0,
    buildOutput: build.output,
    runExitCode: run.exitCode,
    runOutput: run.output,
    phase: "fix",
    tokPerSec: tps,
    tokensGenerated,
    error: run.exitCode !== 0 ? `Runtime exit code ${run.exitCode}` : null,
  };
  result.phase = detectPhase(result);
  return result;
}

// ── Agent invocation ─────────────────────────────────────────────────

function buildClaudeArgs(prompt: string): string[] {
  return [
    "-p",
    "--verbose",
    "--output-format", "stream-json",
    "--include-partial-messages",
    `--disallowed-tools=${BLOCKED_GIT_OPS.join(",")}`,
    "--permission-mode", "bypassPermissions",
    "--effort", "high",
    prompt,
  ];
}

async function runAgent(
  agent: AgentKind,
  prompt: string,
): Promise<RunResult> {
  const label = agent === "codex" ? "Codex" : "Claude";
  console.log(clr("1;34", SEP));
  console.log(clr("1;34", `  🧠 PROMPT (${label})`));
  console.log(clr("1;34", SEP));
  const promptLines = prompt.split("\n");
  for (const line of promptLines.slice(0, 15))
    process.stdout.write(clr("2", line) + "\n");
  if (promptLines.length > 15)
    process.stdout.write(clr("2", `… (${promptLines.length - 15} more lines)\n`));
  console.log(clr("1;34", SEP));

  console.log(clr("1;36", SEP));
  console.log(clr("1;36", `  💬 RESPONSE (${label})`));
  console.log(clr("1;36", SEP));

  const startedAt = Date.now();
  const heartbeat = setInterval(() => {
    process.stdout.write(
      clr("2", `\n⏳ still running (${formatElapsed(startedAt)} elapsed)...\n`),
    );
  }, 30_000);

  const result = await runCommand("claude", buildClaudeArgs(prompt), {
    streamOutput: true,
    timeout: 900_000, // 15 min max per agent call
  });

  clearInterval(heartbeat);
  console.log(clr("1;36", SEP));
  console.log(clr("1;32", `  ✅ ${label} done in ${formatElapsed(startedAt)}`));
  console.log(clr("1;36", SEP));

  return result;
}

// ── Prompt builder ───────────────────────────────────────────────────

function buildPrompt(state: RunState, lastResult: BuildRunResult): string {
  const { cycles, failedApproaches, phase } = state;

  const trunc = (s: string, max: number) =>
    s && s.length > max ? s.slice(0, max) + "…" : s;

  const historyBlock =
    cycles.length > 0
      ? cycles
          .slice(-20)
          .map((h) => {
            const desc = trunc(h.description, 80);
            return `  #${h.cycle}: [${h.phase}] ${desc} → ${h.kept ? "✅ KEPT" : "❌ REVERTED"}${h.tokPerSec != null ? ` (${h.tokPerSec.toFixed(1)} tok/s)` : ""}`;
          })
          .join("\n")
      : "  (none yet)";

  const failedBlock =
    failedApproaches.length > 0
      ? failedApproaches
          .slice(-20)
          .map((f, n) => `  ${n + 1}. ${trunc(f, 120)}`)
          .join("\n")
      : "  (none yet)";

  // Truncate build/run output to avoid blowing up prompt
  const buildOut = lastResult.buildOutput.slice(-3000);
  const runOut = lastResult.runOutput.slice(-3000);

  const phaseLabel = phase === "fix" ? "FIX" : "OPTIMIZE";
  const phaseInstructions =
    phase === "fix"
      ? [
          "## Phase: FIX",
          "The build or runtime is failing. Your job is to fix ONE error.",
          "",
          "Priority order:",
          "1. If build fails: fix the Zig compilation error or GLSL shader error",
          "2. If runtime crashes: fix the Vulkan error, segfault, or assertion",
          "3. If shader compilation fails on RADV: the shader may use features not supported by the driver",
          "",
          "Common RADV/RDNA4 issues:",
          "- RADV may not support all GL_KHR_cooperative_matrix features — stub out or use fallbacks",
          "- Shared memory limits on RDNA4: 64KB max per workgroup",
          "- glslc from Ubuntu 24.04 (shaderc 2023.8) is required — newer versions break RADV",
          "- Some Vulkan 1.3 features may need explicit feature enablement on device creation",
        ]
      : [
          "## Phase: OPTIMIZE",
          `ZINC is running! Current: ${state.currentBest?.tokPerSec?.toFixed(1) ?? "?"} tok/s.`,
          "",
          "Your job is to improve throughput. Focus on:",
          "1. DMMV shader bandwidth utilization (target: 90%+ on large matmuls)",
          "2. Reducing dispatch overhead (pre-record command buffers)",
          "3. Memory access patterns in shaders (coalesced reads, wave64 optimization)",
          "4. Fusing operations to reduce dispatch count",
          "5. Correct buffer bindings and descriptor set wiring in the forward pass",
        ];

  return [
    `# ZINC ${phaseLabel} Task`,
    "",
    ...phaseInstructions,
    "",
    "## Hardware (remote RDNA4 node)",
    "- GPU: AMD Radeon AI PRO R9700 (RDNA4, gfx1201, 32GB VRAM, 576 GB/s)",
    "- 64 CUs, wave64 optimal, 32KB L1/CU, 6MB L2",
    "- VK_KHR_cooperative_matrix 16x16x16",
    "- RADV driver (Mesa), RADV_PERFTEST=coop_matrix",
    "- System glslc: shaderc 2023.8 (Ubuntu 24.04) — do NOT use newer versions",
    "- Zig 0.15.2",
    "",
    "## Current Build Output (last 3000 chars)",
    "```",
    buildOut,
    "```",
    "",
    ...(lastResult.runOutput
      ? [
          "## Current Run Output (last 3000 chars)",
          "```",
          runOut,
          "```",
          "",
        ]
      : []),
    "## Project Structure",
    "```",
    "src/vulkan/   — vk.zig, instance.zig, buffer.zig, pipeline.zig, command.zig, gpu_detect.zig",
    "src/model/    — gguf.zig, loader.zig, architecture.zig, tokenizer.zig",
    "src/compute/  — graph.zig, dmmv.zig, elementwise.zig, attention.zig, forward.zig",
    "src/shaders/  — dmmv_q4k.comp, dmmv_q8_0.comp, dmmv_f16.comp, rms_norm_mul.comp,",
    "                swiglu.comp, rope_fused.comp, flash_attn.comp, coop_matmul.comp,",
    "                sigmoid_mul.comp, softmax_topk.comp, tq_*.comp",
    "src/main.zig  — CLI, Vulkan init, model load, inference engine, forward pass",
    "build.zig     — build system with conditional shader compilation",
    "```",
    "",
    "## Optimization History (last 20 cycles)",
    historyBlock,
    "",
    "## FAILED APPROACHES — DO NOT REPEAT",
    failedBlock,
    "",
    "## Rules",
    "1. Make ONE focused change. Do not try multiple things.",
    "2. Edit LOCAL source files only (this machine's working copy).",
    "3. The loop will rsync your changes and rebuild on the remote node.",
    "4. Do NOT modify .env, loops/, or files outside src/ and build.zig.",
    "5. Zig 0.15.2 API: ArrayList is unmanaged (pass allocator to append/deinit),",
    "   StringHashMap → use StringHashMapUnmanaged, File.stdout() not io.getStdOut(),",
    "   writer() takes a buffer arg, process.Child StdIo uses .Pipe not .pipe.",
    "6. GLSL shaders must compile with `glslc --target-env=vulkan1.3 -O`.",
    "",
    "## Output Format",
    "After making your change, print these 3 lines at the very end:",
    "@@@DESCRIPTION: <one-line summary of what you changed>",
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
  buildExitCode: number;
  runExitCode: number | null;
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
  currentBest: { tokPerSec: number | null } | null;
};

async function loadState(runDir: string): Promise<RunState | null> {
  const p = join(runDir, "state.json");
  if (!existsSync(p)) return null;
  return JSON.parse(await readFile(p, "utf8")) as RunState;
}

async function saveState(runDir: string, state: RunState): Promise<void> {
  await writeFile(join(runDir, "state.json"), JSON.stringify(state, null, 2));
}

// ── Cycle runner ─────────────────────────────────────────────────────

async function runCycle(
  runDir: string,
  state: RunState,
  agent: AgentKind,
  modelPath: string,
): Promise<CycleResult> {
  const cycleNum = state.cycles.length + 1;
  const cycleDir = join(runDir, `cycle-${String(cycleNum).padStart(3, "0")}`);
  await mkdir(cycleDir, { recursive: true });

  console.log(clr("1;35", "\n" + "═".repeat(64)));
  console.log(clr("1;35", `  CYCLE ${cycleNum} — phase: ${state.phase.toUpperCase()}`));
  console.log(clr("1;35", "═".repeat(64)));

  // Step 1: rsync + build + run
  try {
    await rsyncToRemote();
  } catch (e) {
    console.log(clr("1;31", `  ❌ rsync failed: ${e}`));
    const result: CycleResult = {
      cycle: cycleNum,
      timestamp: new Date().toISOString(),
      phase: "fix",
      description: "rsync failed",
      kept: false,
      tokPerSec: null,
      buildExitCode: -1,
      runExitCode: null,
      error: String(e),
      selfAnalysis: "",
      nextIdeas: [],
    };
    await writeFile(join(cycleDir, "result.json"), JSON.stringify(result, null, 2));
    return result;
  }

  const buildRun = await buildAndRun(modelPath);
  await writeFile(join(cycleDir, "build.log"), buildRun.buildOutput);
  if (buildRun.runOutput) {
    await writeFile(join(cycleDir, "run.log"), buildRun.runOutput);
  }

  state.phase = buildRun.phase;

  // Print current status
  if (buildRun.buildExitCode !== 0) {
    console.log(clr("1;31", `  ❌ Build failed (exit ${buildRun.buildExitCode})`));
  } else if (buildRun.runExitCode !== 0) {
    console.log(clr("1;31", `  ❌ Runtime failed (exit ${buildRun.runExitCode})`));
  } else {
    console.log(clr("1;32", `  ✅ Build + run succeeded (${buildRun.tokensGenerated} tokens generated)`));
    if (buildRun.tokPerSec != null) {
      console.log(clr("1;33", `  📊 ${buildRun.tokPerSec.toFixed(1)} tok/s`));
    }
  }

  // Step 2: Git snapshot before agent changes
  const preCommit = await runCommand("git", ["rev-parse", "HEAD"], { cwd: PROJECT_ROOT });
  const preHash = preCommit.stdout.trim();

  // Step 3: Build prompt and run agent
  const prompt = buildPrompt(state, buildRun);
  await writeFile(join(cycleDir, "prompt.md"), prompt);

  const agentResult = await runAgent(agent, prompt);
  await writeFile(join(cycleDir, "agent_stdout.txt"), agentResult.stdout);
  await writeFile(join(cycleDir, "agent_stderr.txt"), agentResult.stderr);

  // Extract agent output markers
  const assembledText = extractTextFromStreamJson(agentResult.stdout);
  const lastChars = assembledText.slice(-3000);

  const descMatch =
    lastChars.match(/^@@@DESCRIPTION:\s*(.+)/im) ??
    lastChars.match(/^DESCRIPTION:\s*(.+)/im);
  const analysisMatch =
    lastChars.match(/^@@@SELF_ANALYSIS:\s*(.+)/im) ??
    lastChars.match(/^SELF_ANALYSIS:\s*(.+)/im);
  const ideasMatch =
    lastChars.match(/^@@@NEXT_IDEAS:\s*(.+)/im) ??
    lastChars.match(/^NEXT_IDEAS:\s*(.+)/im);

  const rawDesc = descMatch?.[1]?.trim() ?? "";
  const description = rawDesc && !isGarbageString(rawDesc) ? rawDesc : "Agent made changes";
  const selfAnalysis = analysisMatch?.[1]?.trim() ?? "";
  const newIdeas =
    ideasMatch?.[1]
      ?.split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 3 && s.length < 120 && !isGarbageString(s)) ?? [];

  // Step 4: Verify — rsync + build + run again
  console.log(clr("1;33", "\n  📊 Verifying agent's changes..."));
  let verifyResult: BuildRunResult;
  try {
    await rsyncToRemote();
    verifyResult = await buildAndRun(modelPath);
  } catch (e) {
    console.log(clr("1;31", `  ❌ Verification failed: ${e}`));
    verifyResult = {
      buildExitCode: -1,
      buildOutput: String(e),
      runExitCode: null,
      runOutput: "",
      phase: "fix",
      tokPerSec: null,
      tokensGenerated: 0,
      error: String(e),
    };
  }

  // Step 5: Decide keep or revert
  let keep = false;

  if (buildRun.phase === "fix") {
    // In fix mode: keep if we made progress (fewer errors, or moved to optimize phase)
    if (verifyResult.phase === "optimize") {
      keep = true; // We fixed it!
      console.log(clr("1;32", "  🎉 FIXED! Moving to optimize phase."));
    } else if (
      verifyResult.buildExitCode === 0 &&
      buildRun.buildExitCode !== 0
    ) {
      keep = true; // Build now succeeds
      console.log(clr("1;32", "  ✅ Build now passes."));
    } else if (
      verifyResult.buildExitCode === 0 &&
      verifyResult.runExitCode === 0 &&
      (buildRun.runExitCode !== 0 || buildRun.runExitCode === null)
    ) {
      keep = true; // Runtime now succeeds
      console.log(clr("1;32", "  ✅ Runtime now passes."));
    } else if (
      verifyResult.buildExitCode === 0 &&
      verifyResult.runExitCode === 0 &&
      buildRun.buildExitCode === 0 &&
      buildRun.runExitCode === 0 &&
      verifyResult.tokensGenerated > buildRun.tokensGenerated
    ) {
      keep = true; // More tokens generated — forward progress
      console.log(clr("1;32", `  📈 More tokens: ${buildRun.tokensGenerated} → ${verifyResult.tokensGenerated}`));
    } else if (
      verifyResult.buildExitCode === 0 &&
      verifyResult.runExitCode === 0 &&
      buildRun.buildExitCode === 0 &&
      buildRun.runExitCode === 0
    ) {
      // Both succeed with same token count — keep if no regression (agent may have improved code quality)
      if (verifyResult.tokensGenerated >= buildRun.tokensGenerated) {
        keep = true;
        console.log(clr("1;33", `  ↔ No regression (${verifyResult.tokensGenerated} tokens). Keeping.`));
      }
    } else if (verifyResult.buildExitCode !== 0 && buildRun.buildExitCode !== 0) {
      // Both fail — check if error changed (might be progress)
      const beforeErrors = (buildRun.buildOutput.match(/error:/g) ?? []).length;
      const afterErrors = (verifyResult.buildOutput.match(/error:/g) ?? []).length;
      if (afterErrors < beforeErrors) {
        keep = true;
        console.log(clr("1;33", `  📉 Fewer errors: ${beforeErrors} → ${afterErrors}`));
      }
    }
  } else {
    // In optimize mode: keep if tok/s improved and didn't break
    if (
      verifyResult.phase === "optimize" &&
      verifyResult.tokPerSec != null &&
      buildRun.tokPerSec != null &&
      verifyResult.tokPerSec > buildRun.tokPerSec * 1.005 // 0.5% improvement threshold
    ) {
      keep = true;
      console.log(
        clr(
          "1;32",
          `  📈 Improved: ${buildRun.tokPerSec.toFixed(1)} → ${verifyResult.tokPerSec.toFixed(1)} tok/s`,
        ),
      );
    }
  }

  console.log(
    clr(keep ? "1;32" : "1;31", `  → ${keep ? "✅ KEEPING" : "❌ REVERTING"}`),
  );

  if (!keep) {
    // Revert local changes
    console.log(clr("2", "  Reverting local changes..."));
    await runCommand("git", ["checkout", "."], { cwd: PROJECT_ROOT }).catch(() => {});
    // Also reset any new files the agent may have created
    await runCommand("git", ["clean", "-fd", "src/", "build.zig"], { cwd: PROJECT_ROOT }).catch(() => {});
  } else {
    // Commit successful change
    await runCommand("git", ["add", "src/", "build.zig", "build.zig.zon"], { cwd: PROJECT_ROOT }).catch(() => {});
    await runCommand(
      "git",
      ["commit", "-m", `zinc-loop: ${description}`],
      { cwd: PROJECT_ROOT },
    ).catch(() => {});

    // Update best
    if (verifyResult.tokPerSec != null) {
      state.currentBest = { tokPerSec: verifyResult.tokPerSec };
    }
  }

  const cycleResult: CycleResult = {
    cycle: cycleNum,
    timestamp: new Date().toISOString(),
    phase: buildRun.phase,
    description,
    kept: keep,
    tokPerSec: verifyResult.tokPerSec,
    buildExitCode: verifyResult.buildExitCode,
    runExitCode: verifyResult.runExitCode,
    selfAnalysis,
    nextIdeas: newIdeas,
    error: verifyResult.error ?? undefined,
  };
  await writeFile(join(cycleDir, "result.json"), JSON.stringify(cycleResult, null, 2));
  return cycleResult;
}

// ── Main ─────────────────────────────────────────────────────────────

async function main() {
  const rawArgs = process.argv.slice(2);
  let maxCycles = Infinity;
  let agent: AgentKind = "claude";
  let modelPath = DEFAULT_MODEL;
  let dryRun = false;
  let resumeDir: string | undefined;

  for (let i = 0; i < rawArgs.length; i++) {
    switch (rawArgs[i]) {
      case "--agent":
        agent = rawArgs[++i] as AgentKind;
        break;
      case "--cycles":
        maxCycles = parseInt(rawArgs[++i], 10);
        break;
      case "--model-path":
        modelPath = rawArgs[++i];
        break;
      case "--dry-run":
        dryRun = true;
        break;
      case "--resume":
        resumeDir = rawArgs[++i];
        break;
      case "--help":
        console.log(
          [
            "Usage: bun loops/optimize_zinc.ts [options]",
            "",
            "Options:",
            "  --agent <claude|codex>   Agent to use (default: claude)",
            "  --cycles N               Max cycles (default: infinite)",
            "  --model-path <path>      GGUF model path on remote node",
            "  --dry-run                Build+run only, no agent",
            "  --resume <dir>           Resume from a previous run directory",
          ].join("\n"),
        );
        process.exit(0);
    }
  }

  const runId = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
  const runDir = resumeDir ?? join(RESULTS_DIR, runId);
  await mkdir(runDir, { recursive: true });

  console.log(clr("1;35", "═".repeat(64)));
  console.log(clr("1;35", "  ZINC SELF-IMPROVING LOOP"));
  console.log(clr("1;35", "═".repeat(64)));
  console.log(`  Remote:   ${clr("1", `${ZINC_USER}@${ZINC_HOST}:${ZINC_PORT}`)}`);
  console.log(`  Model:    ${clr("1", modelPath)}`);
  console.log(`  Agent:    ${clr("1", agent)}`);
  console.log(`  Cycles:   ${maxCycles === Infinity ? "infinite" : String(maxCycles)}`);
  console.log(`  Results:  ${runDir}`);
  console.log(clr("1;35", "═".repeat(64)));

  // Pre-check SSH
  console.log(clr("2", "\n  Checking SSH connectivity..."));
  try {
    const osInfo = await ssh("uname -a", 15_000);
    console.log(clr("1;32", `  SSH OK: ${osInfo.slice(0, 80)}`));
  } catch (e) {
    console.error(clr("31", `\n  ❌ Cannot reach remote node: ${e}`));
    process.exit(1);
  }

  // Check for zig on remote
  try {
    const zigVer = await ssh("zig version", 10_000);
    console.log(clr("1;32", `  Zig: ${zigVer}`));
  } catch {
    console.error(clr("31", "  ❌ Zig not found on remote node. Install zig 0.15.2+."));
    process.exit(1);
  }

  // Ensure remote zinc dir exists
  await ssh(`mkdir -p ${REMOTE_ZINC_DIR}`, 10_000).catch(() => {});

  // Dry run: just build+run once
  if (dryRun) {
    console.log(clr("1;33", "\n  DRY RUN: rsync + build + run"));
    await rsyncToRemote();
    const result = await buildAndRun(modelPath);
    console.log(clr("1;33", `\n  Phase: ${result.phase}`));
    if (result.tokPerSec != null) {
      console.log(clr("1;33", `  tok/s: ${result.tokPerSec.toFixed(1)}`));
    }
    if (result.error) {
      console.log(clr("1;31", `  Error: ${result.error}`));
    }
    return;
  }

  // Load or create state
  let state = await loadState(runDir);
  if (!state) {
    state = {
      runId,
      cycles: [],
      failedApproaches: [],
      ideas: [],
      phase: "fix",
      currentBest: null,
    };
    await saveState(runDir, state);
  } else {
    console.log(clr("1;33", `\n  Resuming from cycle ${state.cycles.length}`));
  }

  // Main loop
  let cyclesDone = 0;
  let consecutiveSSHFailures = 0;

  while (cyclesDone < maxCycles) {
    // SSH health check
    try {
      await ssh("echo ok", 15_000);
      consecutiveSSHFailures = 0;
    } catch {
      consecutiveSSHFailures++;
      if (consecutiveSSHFailures >= 3) {
        console.error(
          clr("31", `\n  ❌ SSH unreachable ${consecutiveSSHFailures}x. Waiting 5 min...`),
        );
        await new Promise((r) => setTimeout(r, 300_000));
        continue;
      }
      console.log(clr("33", `  SSH failed (${consecutiveSSHFailures}/3), retry in 60s...`));
      await new Promise((r) => setTimeout(r, 60_000));
      continue;
    }

    const cycleResult = await runCycle(runDir, state, agent, modelPath);
    state.cycles.push(cycleResult);

    // Update failed approaches
    if (!cycleResult.kept && cycleResult.description !== "Agent made changes" && !cycleResult.description.includes("rsync")) {
      const desc = cycleResult.description.slice(0, 100);
      if (!isGarbageString(desc)) {
        state.failedApproaches.push(desc);
      }
      if (state.failedApproaches.length > 30) {
        state.failedApproaches = state.failedApproaches.slice(-30);
      }
    }

    // Merge new ideas
    const existingLower = new Set(state.ideas.map((i) => i.toLowerCase()));
    for (const idea of cycleResult.nextIdeas) {
      if (!existingLower.has(idea.toLowerCase()) && !isGarbageString(idea)) {
        state.ideas.push(idea);
        existingLower.add(idea.toLowerCase());
      }
    }
    if (state.ideas.length > 50) state.ideas = state.ideas.slice(0, 50);

    // Cap cycle history
    if (state.cycles.length > 60) {
      const kept = state.cycles.filter((c) => c.kept);
      const recent = state.cycles.slice(-50);
      const seen = new Set<number>();
      const merged: CycleResult[] = [];
      for (const c of [...kept, ...recent]) {
        if (!seen.has(c.cycle)) {
          seen.add(c.cycle);
          merged.push(c);
        }
      }
      state.cycles = merged.sort((a, b) => a.cycle - b.cycle);
    }

    await saveState(runDir, state);

    // Print summary
    const keptCount = state.cycles.filter((c) => c.kept).length;
    console.log(clr("1;35", "\n" + "═".repeat(64)));
    console.log(clr("1;35", `  AFTER ${state.cycles.length} CYCLES:`));
    console.log(clr("1;35", `  Phase: ${state.phase.toUpperCase()}`));
    if (state.currentBest?.tokPerSec != null) {
      console.log(clr("1;35", `  Best: ${state.currentBest.tokPerSec.toFixed(1)} tok/s`));
    }
    console.log(clr("1;35", `  Kept: ${keptCount}/${state.cycles.length}`));
    console.log(clr("1;35", "═".repeat(64)));

    cyclesDone++;
    await new Promise((r) => setTimeout(r, 3_000));
  }

  console.log(clr("1;32", "\n" + "═".repeat(64)));
  console.log(clr("1;32", "  ZINC OPTIMIZATION COMPLETE"));
  console.log(clr("1;32", `  Cycles: ${state.cycles.length} | Kept: ${state.cycles.filter((c) => c.kept).length}`));
  console.log(clr("1;32", `  Results: ${runDir}`));
  console.log(clr("1;32", "═".repeat(64)));
}

if (import.meta.main) {
  main().catch((err) => {
    console.error(clr("31", `\nFatal error: ${err.message ?? err}`));
    process.exit(1);
  });
}
