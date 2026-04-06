#!/usr/bin/env bun
/**
 * ZINC Performance Optimization Loop
 *
 * Implements multi-hour optimization efforts defined in MULTI_HOUR_EFFORT_*.md
 * documents. Each cycle:
 *   1. Read the optimization plan document
 *   2. Build & benchmark baseline on remote RDNA4 node
 *   3. Spawn AI agent to implement ONE concrete step from the plan
 *   4. Build, run tests, benchmark
 *   5. If tok/s improved AND output correct -> commit, update plan
 *   6. If regressed or broken -> revert, log what went wrong
 *   7. Loop back to 3
 *
 * Usage:
 *   bun loops/optimize_perf.ts --effort 1                        # Push descriptors
 *   bun loops/optimize_perf.ts --effort 2 --model qwen35b       # Fused gate+up on Qwen 35B
 *   bun loops/optimize_perf.ts --effort 3 --agent codex         # Batch prefill with Codex
 *   bun loops/optimize_perf.ts --effort 1 --resume               # Resume previous run
 *   bun loops/optimize_perf.ts --effort 1 --cycles 10 --dry-run  # Baseline only
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join, resolve } from "node:path";
import {
  parseTokPerSec,
  parseBandwidthUtil,
} from "./optimize_zinc";
import { formatElapsed } from "./optimize_llm_tps";

// -- Config ------------------------------------------------------------------

const REPO_ROOT = resolve(import.meta.dir, "..");
const RESULTS_DIR = resolve(REPO_ROOT, ".perf_optimize");

function loadEnv(): Record<string, string> {
  const envPath = join(REPO_ROOT, ".env");
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
const REMOTE_DIR = "/root/zinc";

const MODELS: Record<string, string> = {
  qwen35b: "/root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
  qwen2b: "/root/models/Qwen3.5-2B-Q4_K_M.gguf",
  gemma3: "/root/models/gemma-3-12b-it-Q4_K_M.gguf",
};

const EFFORT_DOCS: Record<number, string> = {
  1: "MULTI_HOUR_EFFORT_1_PUSH_DESCRIPTORS.md",
  2: "MULTI_HOUR_EFFORT_2_FUSED_GATE_UP.md",
  3: "MULTI_HOUR_EFFORT_3_BATCH_PREFILL.md",
};

// Multiple prompts to catch different failure modes:
// - Short factual: catches total corruption
// - Arithmetic: catches subtle numeric drift (wrong MoE routing, bad dequant)
// - Listing: catches mid-sequence divergence (broken RoPE, bad KV cache)
const COHERENCE_CHECKS: { prompt: string; expect: string[] }[] = [
  { prompt: "The capital of France is", expect: ["Paris"] },
  { prompt: "What is 2+2?", expect: ["4"] },
  { prompt: "List the first 4 planets: Mercury,", expect: ["Venus", "Earth", "Mars"] },
];

// All models that must produce coherent output after every change.
// The primary model (--model flag) is benchmarked; these are correctness-only.
const COHERENCE_MODELS: { name: string; path: string }[] = [
  { name: "Qwen3.5-35B", path: "/root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf" },
  { name: "Qwen3.5-2B", path: "/root/models/Qwen3.5-2B-Q4_K_M.gguf" },
  { name: "Gemma3-12B", path: "/root/models/gemma-3-12b-it-Q4_K_M.gguf" },
];

const BLOCKED_FILE_OPS = [
  "Edit(loops/*)", "Write(loops/*)", "Edit(site/*)", "Write(site/*)",
  "Edit(docs/*)", "Write(docs/*)", "Edit(.env)", "Write(.env)",
  "Edit(AGENTS.md)", "Write(AGENTS.md)", "Edit(CLAUDE.md)", "Write(CLAUDE.md)",
  "Edit(MULTI_HOUR_EFFORT_*)", "Write(MULTI_HOUR_EFFORT_*)",
];

const BLOCKED_GIT_OPS = [
  "Bash(git checkout:*)", "Bash(git revert:*)", "Bash(git restore:*)",
  "Bash(git reset:*)", "Bash(git stash:*)", "Bash(git clean:*)",
  "Bash(git push:*)", "Bash(git commit:*)",
];

// Directories the agent may change (used for selective revert)
const REVERTABLE_PATHS = ["src/"];

// -- CLI parsing -------------------------------------------------------------

type AgentType = "claude" | "codex";

function parseArgs() {
  const args = process.argv.slice(2);
  let effort = 0;
  let cycles = 20;
  let dryRun = false;
  let model = "qwen35b";
  let resume = false;
  let agent: AgentType = "claude";

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--effort" && args[i + 1]) effort = parseInt(args[++i], 10);
    else if (args[i] === "--cycles" && args[i + 1]) cycles = parseInt(args[++i], 10);
    else if (args[i] === "--dry-run") dryRun = true;
    else if (args[i] === "--resume") resume = true;
    else if (args[i] === "--model" && args[i + 1]) model = args[++i];
    else if (args[i] === "--agent" && args[i + 1]) agent = args[++i] as AgentType;
  }
  if (!effort || !EFFORT_DOCS[effort]) {
    console.error("Usage: bun loops/optimize_perf.ts --effort <1|2|3> [options]");
    console.error("");
    console.error("Options:");
    console.error("  --effort <1|2|3>         Optimization to run (required)");
    console.error("  --cycles N               Max cycles (default: 20)");
    console.error("  --model NAME             Model: qwen35b, qwen2b, gemma3 (default: qwen35b)");
    console.error("  --agent claude|codex     AI agent to use (default: claude)");
    console.error("  --resume                 Resume from previous run (read history from log)");
    console.error("  --dry-run                Build+bench baseline only, skip agent");
    console.error("");
    console.error("Efforts:");
    console.error("  1 = Push descriptors (~2.5% decode speedup)");
    console.error("  2 = Fused gate+up DMMV (~1-2% decode speedup)");
    console.error("  3 = Batch prefill (~4-8x prefill speedup)");
    process.exit(1);
  }
  if (agent !== "claude" && agent !== "codex") {
    console.error(`Unknown agent: ${agent}. Use 'claude' or 'codex'.`);
    process.exit(1);
  }
  return { effort, cycles, dryRun, model, resume, agent };
}

// -- Display helpers ---------------------------------------------------------

const CLR = process.stdout.isTTY && !("NO_COLOR" in process.env);
const c = (code: string, t: string) => CLR ? `\x1b[${code}m${t}\x1b[0m` : t;
const SEP = "\u2500".repeat(64);

// -- Command runner with streaming -------------------------------------------

type RunResult = { exitCode: number; stdout: string; stderr: string };

async function runCommand(
  cmd: string,
  args: string[],
  opts: {
    cwd?: string;
    timeout?: number;
    streamOutput?: boolean;
    stdoutLineFormatter?: (line: string) => string | null;
  } = {},
): Promise<RunResult> {
  const streamOutput = opts.streamOutput ?? false;
  return new Promise((res, rej) => {
    const child = spawn(cmd, args, {
      cwd: opts.cwd ?? REPO_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
      timeout: opts.timeout ?? 120_000,
    });
    let stdout = "", stderr = "", lineBuffer = "";
    child.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stdout += text;
      if (!streamOutput) return;
      if (opts.stdoutLineFormatter) {
        lineBuffer += text;
        const lines = lineBuffer.split("\n");
        lineBuffer = lines.pop() ?? "";
        for (const line of lines) {
          const f = opts.stdoutLineFormatter(line);
          if (f !== null) process.stdout.write(f);
        }
      } else {
        process.stdout.write(text);
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stderr += text;
      if (streamOutput) process.stderr.write(text);
    });
    child.on("error", rej);
    child.on("close", (code) => {
      if (streamOutput && opts.stdoutLineFormatter && lineBuffer.trim()) {
        const f = opts.stdoutLineFormatter(lineBuffer);
        if (f !== null) process.stdout.write(f);
      }
      res({ exitCode: code ?? 1, stdout, stderr });
    });
  });
}

// -- SSH & rsync -------------------------------------------------------------

async function ssh(command: string, timeout = 120_000): Promise<string> {
  const { stdout, stderr, exitCode } = await runCommand("ssh", [
    "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
    "-p", String(ZINC_PORT), `${ZINC_USER}@${ZINC_HOST}`, command,
  ], { timeout });
  if (exitCode !== 0 && !stderr.includes("Warning"))
    throw new Error(`SSH failed (${exitCode}): ${stderr.slice(0, 500)}`);
  return stdout.trim();
}

async function rsyncToRemote(): Promise<void> {
  const { exitCode, stderr } = await runCommand("rsync", [
    "-avz", "--checksum", "--delete",
    "-e", `ssh -p ${ZINC_PORT} -o StrictHostKeyChecking=no`,
    "--exclude", ".zig-cache", "--exclude", "zig-out", "--exclude", "node_modules",
    "--exclude", ".git", "--exclude", ".perf_optimize", "--exclude", ".zinc_optimize",
    "--exclude", "site", "--exclude", ".DS_Store",
    `${REPO_ROOT}/`, `${ZINC_USER}@${ZINC_HOST}:${REMOTE_DIR}/`,
  ], { timeout: 120_000 });
  if (exitCode !== 0) throw new Error(`rsync failed: ${stderr.slice(0, 300)}`);
}

// -- Build & benchmark -------------------------------------------------------

type BenchResult = {
  buildOk: boolean;
  buildOutput: string;
  tokPerSec: number | null;
  correct: boolean;
  outputText: string;
  bandwidthUtil: number | null;
  error: string | null;
};

async function buildAndBench(modelPath: string): Promise<BenchResult> {
  console.log(c("2", "  Compiling shaders..."));
  try {
    await ssh(`cd ${REMOTE_DIR}/src/shaders && for f in *.comp; do glslc --target-env=vulkan1.3 -fshader-stage=compute $f -o \${f%.comp}.spv 2>&1; done`, 60_000);
  } catch (e) {
    return { buildOk: false, buildOutput: String(e), tokPerSec: null, correct: false, outputText: "", bandwidthUtil: null, error: "shader compile failed" };
  }

  console.log(c("2", "  Building..."));
  let buildOutput: string;
  try {
    buildOutput = await ssh(`cd ${REMOTE_DIR} && zig build 2>&1`, 300_000);
  } catch (e) {
    return { buildOk: false, buildOutput: String(e), tokPerSec: null, correct: false, outputText: "", bandwidthUtil: null, error: "zig build failed" };
  }
  if (buildOutput.includes("error:")) {
    return { buildOk: false, buildOutput, tokPerSec: null, correct: false, outputText: "", bandwidthUtil: null, error: "build errors" };
  }

  console.log(c("2", "  Running correctness test..."));
  let runOutput: string;
  const firstCheck = COHERENCE_CHECKS[0];
  try {
    runOutput = await ssh(
      `cd ${REMOTE_DIR} && ./zig-out/bin/zinc -m ${modelPath} --prompt '${firstCheck.prompt}' -n 32 2>&1`,
      180_000,
    );
  } catch (e) {
    return { buildOk: true, buildOutput, tokPerSec: null, correct: false, outputText: "", bandwidthUtil: null, error: `run failed: ${e}` };
  }

  const tokPerSec = parseTokPerSec(runOutput);
  const textMatch = runOutput.match(/Output text:\s*(.+)/i);
  const outputText = textMatch ? textMatch[1].trim() : "";
  const correct = firstCheck.expect.every(e => outputText.toLowerCase().includes(e.toLowerCase()));
  const bandwidthUtil = parseBandwidthUtil(runOutput);

  return { buildOk: true, buildOutput, tokPerSec, correct, outputText, bandwidthUtil, error: null };
}

/// Run ALL coherence prompts on ALL models.
/// Returns null if all pass, or an error string describing which failed.
async function checkAllModelsCoherent(): Promise<string | null> {
  const failures: string[] = [];
  for (const { name, path } of COHERENCE_MODELS) {
    for (const check of COHERENCE_CHECKS) {
      try {
        const out = await ssh(
          `cd ${REMOTE_DIR} && ./zig-out/bin/zinc -m ${path} --prompt '${check.prompt}' -n 30 2>&1`,
          120_000,
        );
        const textMatch = out.match(/Output text:\s*(.+)/i);
        const outputText = textMatch ? textMatch[1].trim() : "";
        const pass = check.expect.every(e => outputText.toLowerCase().includes(e.toLowerCase()));
        if (!pass) {
          failures.push(`${name} [${check.prompt.slice(0, 25)}]: "${outputText.slice(0, 50)}"`);
        }
      } catch (e) {
        failures.push(`${name} [${check.prompt.slice(0, 25)}]: crashed`);
      }
    }
    if (!failures.some(f => f.startsWith(name))) {
      console.log(c("2", `    ${name}: all ${COHERENCE_CHECKS.length} prompts OK`));
    }
  }
  return failures.length > 0 ? `Coherence failures: ${failures.join("; ")}` : null;
}

// -- Codex stream formatter --------------------------------------------------

function formatCodexStreamLine(rawLine: string): string | null {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try { event = JSON.parse(rawLine) as Record<string, unknown>; } catch { return null; }

  const type = event.type as string | undefined;

  // Agent message with text
  if (type === "message" || type === "agent") {
    const content = (event.content ?? event.message) as string | undefined;
    if (content && typeof content === "string" && content.trim()) {
      return c("96", content.trim()) + "\n";
    }
    return null;
  }

  // Tool/function call
  if (type === "function_call" || type === "tool_use" || type === "action") {
    const name = (event.name ?? event.tool ?? event.function) as string | undefined;
    const cmdOrInput = (event.command ?? event.input ?? event.arguments) as string | Record<string, unknown> | undefined;
    if (name === "shell" || name === "bash" || name === "terminal") {
      const cmd = typeof cmdOrInput === "string" ? cmdOrInput : (cmdOrInput as Record<string, unknown>)?.command as string ?? "";
      return `\n${c("33", "\uD83D\uDD27 shell")}${c("2", `   $ ${cmd.length > 120 ? cmd.slice(0, 120) + "\u2026" : cmd}`)}\n`;
    }
    if (name === "write" || name === "create_file" || name === "patch" || name === "apply_diff") {
      const fp = typeof cmdOrInput === "string" ? cmdOrInput : (cmdOrInput as Record<string, unknown>)?.path as string ?? (cmdOrInput as Record<string, unknown>)?.file_path as string ?? "";
      const short = fp.split("/").slice(-3).join("/");
      return `\n${c("33", `\uD83D\uDD27 ${name}`)}${c("2", ` \u2192 ${short}`)}\n`;
    }
    if (name === "read" || name === "read_file") {
      const fp = typeof cmdOrInput === "string" ? cmdOrInput : (cmdOrInput as Record<string, unknown>)?.path as string ?? (cmdOrInput as Record<string, unknown>)?.file_path as string ?? "";
      const short = fp.split("/").slice(-3).join("/");
      return `${c("33", `\uD83D\uDD27 ${name}`)}${c("2", ` \u2192 ${short}`)}\n`;
    }
    if (name) {
      return `\n${c("33", `\uD83D\uDD27 ${name}`)}\n`;
    }
    return null;
  }

  // Function call output / result — skip (too verbose)
  if (type === "function_call_output" || type === "action_output" || type === "tool_result") {
    return null;
  }

  // Thinking / reasoning — show brief indicator
  if (type === "thinking" || type === "reasoning") {
    return c("2", "  \u2026 thinking\n");
  }

  return null;
}

// -- Claude stream formatter -------------------------------------------------

type ClaudeStreamState = {
  currentToolName: string | null;
  currentBlockIsToolUse: boolean;
  inputJsonBuffer: string;
  inTextBlock: boolean;
  sawTextDeltaInCurrentMessage: boolean;
};

function formatToolInput(name: string, rawJson: string): string {
  let input: Record<string, unknown> = {};
  try { input = JSON.parse(rawJson) as Record<string, unknown>; } catch { /* empty */ }

  const out: string[] = [];
  const shortPath = (fp: string) => fp.split("/").slice(-3).join("/");
  if (name === "edit") {
    out.push(c("2", ` \u2192 ${shortPath((input.file_path as string) ?? "?")}`));
  } else if (name === "write") {
    const lineCount = ((input.content as string) ?? "").split("\n").length;
    out.push(c("2", ` \u2192 ${shortPath((input.file_path as string) ?? "?")} (${lineCount} lines)`));
  } else if (name === "bash") {
    const cmd = (input.command as string) ?? "?";
    out.push(c("2", `   $ ${cmd.length > 120 ? cmd.slice(0, 120) + "\u2026" : cmd}`));
  } else if (name === "read") {
    out.push(c("2", ` \u2192 ${shortPath((input.file_path as string) ?? "?")}`));
  } else if (name === "grep") {
    out.push(c("2", ` \u2192 /${(input.pattern as string) ?? "?"}/`));
  } else if (name === "glob") {
    out.push(c("2", ` \u2192 ${(input.pattern as string) ?? "?"}`));
  }
  return out.length > 0 ? out.join("\n") + "\n" : "";
}

function formatClaudeStreamLine(rawLine: string, state: ClaudeStreamState): string | null {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try { event = JSON.parse(rawLine) as Record<string, unknown>; } catch { return rawLine + "\n"; }

  if (event.type === "stream_event") {
    const e = event.event as Record<string, unknown> | undefined;
    if (!e) return null;
    if (e.type === "content_block_start") {
      const block = e.content_block as Record<string, unknown> | undefined;
      if (block?.type === "tool_use") {
        state.currentToolName = (block.name as string) ?? "tool";
        state.currentBlockIsToolUse = true;
        state.inputJsonBuffer = "";
        state.inTextBlock = false;
        return `\n${c("33", `\uD83D\uDD27 ${state.currentToolName}`)}`;
      }
      if (block?.type === "text") {
        state.inTextBlock = true;
        state.currentBlockIsToolUse = false;
        return CLR ? "\n\x1b[96m" : "\n";
      }
      state.inTextBlock = false;
      state.currentBlockIsToolUse = false;
      return null;
    }
    if (e.type === "content_block_delta") {
      const delta = e.delta as Record<string, unknown> | undefined;
      if (delta?.type === "input_json_delta") {
        state.inputJsonBuffer += (delta.partial_json as string) ?? "";
        return null;
      }
      if (delta?.type === "text_delta" && state.inTextBlock) {
        state.sawTextDeltaInCurrentMessage = true;
        return delta.text as string;
      }
      return null;
    }
    if (e.type === "content_block_stop") {
      if (state.currentBlockIsToolUse) {
        state.currentBlockIsToolUse = false;
        const detail = formatToolInput(state.currentToolName ?? "", state.inputJsonBuffer);
        state.inputJsonBuffer = "";
        return detail || null;
      }
      if (state.inTextBlock) {
        state.inTextBlock = false;
        return CLR ? "\x1b[0m\n" : "\n";
      }
      return null;
    }
    return null;
  }
  if (event.type === "assistant") {
    const msg = event.message as Record<string, unknown> | undefined;
    if (!msg) return null;
    const content = msg.content;
    if (Array.isArray(content)) {
      const parts: string[] = [];
      for (const block of content) {
        const b = block as Record<string, unknown>;
        if (b?.type === "text" && typeof b.text === "string" && b.text.trim())
          parts.push(b.text);
      }
      const text = parts.join("\n");
      if (!text.trim()) return null;
      if (state.sawTextDeltaInCurrentMessage) {
        state.sawTextDeltaInCurrentMessage = false;
        return null;
      }
      return c("96", text) + "\n";
    }
    return null;
  }
  return null;
}

// -- Agent spawn -------------------------------------------------------------

async function spawnAgent(
  _effortDoc: string,
  plan: string,
  baseline: BenchResult,
  cycleNum: number,
  history: string,
  model: string,
  agent: AgentType = "claude",
): Promise<RunResult> {
  const modelPath = MODELS[model] ?? MODELS.qwen35b;
  const prompt = `You are implementing a performance optimization for the ZINC Vulkan inference engine.

## Optimization Plan
${plan}

## Current Baseline
- tok/s: ${baseline.tokPerSec?.toFixed(2) ?? "unknown"}
- bandwidth utilization: ${baseline.bandwidthUtil?.toFixed(1) ?? "unknown"}%
- output: "${baseline.outputText}" (coherence tested with 3 prompts on 3 models after every change)

## Previous Attempts
${history || "None yet."}

## Your Task (Cycle ${cycleNum})
Implement ONE concrete step from the optimization plan above. Pick the next unfinished step.

## CRITICAL RULES — READ CAREFULLY

1. **BUILD MUST PASS.** Before you declare yourself done, you MUST:
   a. rsync your changes to the remote node
   b. Compile shaders: ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR}/src/shaders && for f in *.comp; do glslc --target-env=vulkan1.3 -fshader-stage=compute \\$f -o \\$\{f%.comp}.spv 2>&1; done"
   c. Build: ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR} && zig build 2>&1"
   d. If the build fails, FIX THE ERRORS before finishing. Do NOT leave broken code.

2. **Incremental steps.** If the optimization requires changing many call sites (e.g. 60+ descriptor set conversions), break it into compilable stages:
   - Add new infrastructure (new functions, new fields) FIRST — the old code can coexist.
   - Convert call sites in batches, building after each batch to catch errors.
   - Remove old infrastructure LAST.
   - The code MUST compile at every stage.

3. **ONE focused change per cycle.** Don't try to convert the entire codebase in one shot.

4. **Test on remote node:**
   rsync -avz --checksum --delete -e "ssh -p ${ZINC_PORT} -o StrictHostKeyChecking=no" --exclude .zig-cache --exclude zig-out --exclude node_modules --exclude .git --exclude .perf_optimize --exclude .zinc_optimize --exclude site --exclude .DS_Store ${REPO_ROOT}/ ${ZINC_USER}@${ZINC_HOST}:${REMOTE_DIR}/
   ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR} && zig build && ./zig-out/bin/zinc -m ${modelPath} --prompt '${COHERENCE_CHECKS[0].prompt}' -n 16"

5. **Shader compilation:** glslc --target-env=vulkan1.3 -fshader-stage=compute file.comp -o file.spv

Files you may edit:
- src/compute/*.zig (forward.zig, dmmv.zig, elementwise.zig, attention.zig, argmax.zig)
- src/vulkan/*.zig (pipeline.zig, command.zig, buffer.zig, instance.zig)
- src/model/*.zig (tokenizer.zig, loader.zig, config.zig, architecture.zig)
- src/server/*.zig (routes.zig, runtime.zig)
- src/server/chat.html
- src/shaders/*.comp (GLSL compute shaders)
- src/main.zig`;

  console.log(c("1;34", SEP));
  console.log(c("1;34", `  \uD83E\uDDE0 Agent cycle ${cycleNum} (${agent})`));
  console.log(c("1;34", SEP));

  const startedAt = Date.now();
  const heartbeat = setInterval(() => {
    process.stdout.write(
      c("2", `\n\u23F3 still running (${formatElapsed(startedAt)} elapsed)...\n`),
    );
  }, 30_000);

  let result: RunResult;

  if (agent === "codex") {
    // Codex: uses `codex exec` with --full-auto and --json for JSONL streaming
    result = await runCommand("codex", [
      "exec",
      "--full-auto",
      "--json",
      prompt,
    ], {
      cwd: REPO_ROOT,
      timeout: 1_800_000,
      streamOutput: true,
      stdoutLineFormatter: (line) => formatCodexStreamLine(line),
    });
  } else {
    // Claude: uses stream-json for rich tool-use display
    const claudeState: ClaudeStreamState = {
      currentToolName: null,
      currentBlockIsToolUse: false,
      inputJsonBuffer: "",
      inTextBlock: false,
      sawTextDeltaInCurrentMessage: false,
    };

    result = await runCommand("claude", [
      "-p",
      "--verbose",
      "--output-format", "stream-json",
      "--include-partial-messages",
      `--disallowed-tools=${[...BLOCKED_GIT_OPS, ...BLOCKED_FILE_OPS].join(",")}`,
      "--permission-mode", "bypassPermissions",
      "--effort", "high",
      prompt,
    ], {
      cwd: REPO_ROOT,
      timeout: 1_800_000,
      streamOutput: true,
      stdoutLineFormatter: (line) => formatClaudeStreamLine(line, claudeState),
    });
  }

  clearInterval(heartbeat);
  console.log(c("1;36", SEP));
  console.log(c("1;32", `  \u2705 Agent done in ${formatElapsed(startedAt)}`));
  console.log(c("1;36", SEP));

  if (result.exitCode !== 0) {
    console.log(c("1;31", `  Agent exited with code ${result.exitCode}`));
  }

  return result;
}

// -- Resume from previous run ------------------------------------------------

type LogEntry = {
  cycle: number;
  effort: number;
  tokPerSec: number | null;
  bandwidthUtil: number | null;
  correct: boolean;
  improved: boolean;
  broken: boolean;
  outputText: string;
  timestamp: string;
};

async function loadPreviousRun(effort: number): Promise<{ history: string; bestTokPerSec: number; lastCycle: number }> {
  const logPath = join(RESULTS_DIR, `effort_${effort}_log.jsonl`);
  let history = "";
  let bestTokPerSec = 0;
  let lastCycle = 0;

  try {
    const content = await readFile(logPath, "utf8");
    for (const line of content.split("\n").filter(Boolean)) {
      try {
        const entry = JSON.parse(line) as LogEntry;
        if (entry.effort !== effort) continue;
        lastCycle = Math.max(lastCycle, entry.cycle);
        if (entry.broken) {
          history += `\nCycle ${entry.cycle}: REVERTED \u2014 broken (${entry.outputText?.slice(0, 60)})`;
        } else if (entry.improved) {
          history += `\nCycle ${entry.cycle}: KEPT \u2014 ${entry.tokPerSec?.toFixed(2)} tok/s`;
          if (entry.tokPerSec != null && entry.tokPerSec > bestTokPerSec) {
            bestTokPerSec = entry.tokPerSec;
          }
        } else {
          history += `\nCycle ${entry.cycle}: REVERTED \u2014 no improvement (${entry.tokPerSec?.toFixed(2)} tok/s)`;
        }
      } catch { /* skip malformed lines */ }
    }
  } catch { /* no log file yet */ }

  return { history, bestTokPerSec, lastCycle };
}

// -- Selective revert (only src/, not loops/ or config) ----------------------

async function revertAgentChanges(): Promise<void> {
  for (const path of REVERTABLE_PATHS) {
    await runCommand("git", ["checkout", "--", path], { cwd: REPO_ROOT });
  }
  // Also clean any new untracked files the agent may have created in src/
  const { stdout: untracked } = await runCommand("git", ["ls-files", "--others", "--exclude-standard", "src/"], { cwd: REPO_ROOT });
  for (const f of untracked.split("\n").filter(Boolean)) {
    await runCommand("rm", ["-f", f], { cwd: REPO_ROOT });
  }
  console.log(c("2", "  Reverted agent changes (src/ only)."));
}

// -- Main loop ---------------------------------------------------------------

async function main() {
  const { effort, cycles, dryRun, model, resume, agent } = parseArgs();
  const modelPath = MODELS[model] ?? MODELS.qwen35b;
  const effortFile = EFFORT_DOCS[effort];
  const plan = await readFile(join(REPO_ROOT, effortFile), "utf8");

  await mkdir(RESULTS_DIR, { recursive: true });

  console.log(c("1;37", `\n\u2554${"═".repeat(58)}\u2557`));
  console.log(c("1;37", `\u2551  ZINC Performance Optimization Loop \u2014 Effort ${effort}          \u2551`));
  console.log(c("1;37", `\u2551  ${effortFile.padEnd(54)}\u2551`));
  console.log(c("1;37", `\u2551  Model: ${model.padEnd(49)}\u2551`));
  console.log(c("1;37", `\u2551  Agent: ${agent.padEnd(49)}\u2551`));
  if (resume) console.log(c("1;37", `\u2551  Resuming from previous run                        \u2551`));
  console.log(c("1;37", `\u255A${"═".repeat(58)}\u255D\n`));

  // Step 1: Sync and get baseline
  console.log(c("1;33", "\u2500\u2500 Baseline " + "\u2500".repeat(54)));
  await rsyncToRemote();
  const baseline = await buildAndBench(modelPath);

  if (!baseline.buildOk) {
    console.error(c("1;31", "Baseline build failed! Fix build errors first."));
    process.exit(1);
  }
  if (!baseline.correct) {
    console.error(c("1;31", `Baseline output incorrect: "${baseline.outputText}". Fix correctness first.`));
    process.exit(1);
  }

  console.log(c("1;32", `  Baseline: ${baseline.tokPerSec?.toFixed(2)} tok/s, BW: ${baseline.bandwidthUtil?.toFixed(1)}%`));
  console.log(c("1;32", `  Output: "${baseline.outputText.slice(0, 80)}"`));

  let bestTokPerSec = baseline.tokPerSec ?? 0;
  let history = "";
  let startCycle = 1;

  // Resume: load history from previous run
  if (resume) {
    const prev = await loadPreviousRun(effort);
    if (prev.lastCycle > 0) {
      history = prev.history;
      startCycle = prev.lastCycle + 1;
      if (prev.bestTokPerSec > bestTokPerSec) bestTokPerSec = prev.bestTokPerSec;
      console.log(c("1;36", `  Resumed: ${prev.lastCycle} previous cycles, best ${prev.bestTokPerSec.toFixed(2)} tok/s`));
      console.log(c("2", `  History:${prev.history}`));
    } else {
      console.log(c("2", "  No previous run found, starting fresh."));
    }
  }

  // Step 2: Optimization cycles
  for (let cycle = startCycle; cycle < startCycle + cycles; cycle++) {
    console.log(c("1;33", `\n\u2500\u2500 Cycle ${cycle} ` + "\u2500".repeat(54)));

    if (dryRun) {
      console.log(c("2", "  Dry run \u2014 skipping agent."));
      break;
    }

    // Spawn agent
    await spawnAgent(effortFile, plan, baseline, cycle, history, model, agent);

    // Sync and benchmark
    console.log(c("2", "  Syncing changes..."));
    await rsyncToRemote();
    const result = await buildAndBench(modelPath);

    // Check ALL models for coherent output (not just the benchmark model)
    let coherenceError: string | null = null;
    if (result.buildOk && result.correct) {
      console.log(c("2", "  Checking all models for coherence..."));
      coherenceError = await checkAllModelsCoherent();
      if (coherenceError) {
        console.log(c("1;31", `  ${coherenceError}`));
      }
    }

    const improved = result.tokPerSec != null && result.tokPerSec > bestTokPerSec * 1.001;
    const correct = result.correct && coherenceError == null;
    const broken = !result.buildOk || !correct;

    const delta = result.tokPerSec != null && bestTokPerSec > 0
      ? ((result.tokPerSec - bestTokPerSec) / bestTokPerSec * 100).toFixed(2)
      : "?";

    if (broken) {
      console.log(c("1;31", `  \u274C BROKEN: ${result.error ?? "incorrect output"}`));
      console.log(c("1;31", `     Output: "${result.outputText?.slice(0, 80)}"`));
      history += `\nCycle ${cycle}: REVERTED \u2014 ${result.error ?? "incorrect output"}`;
      await revertAgentChanges();
    } else if (improved) {
      console.log(c("1;32", `  \u2705 IMPROVED: ${result.tokPerSec?.toFixed(2)} tok/s (+${delta}%)`));
      bestTokPerSec = result.tokPerSec!;
      history += `\nCycle ${cycle}: KEPT \u2014 ${result.tokPerSec?.toFixed(2)} tok/s (+${delta}%)`;

      // Commit
      await runCommand("git", ["add", "src/"], { cwd: REPO_ROOT });
      await runCommand("git", ["commit", "-m", `perf(effort-${effort}): cycle ${cycle} \u2014 ${result.tokPerSec?.toFixed(2)} tok/s (+${delta}%)`], { cwd: REPO_ROOT });
      console.log(c("2", "  Committed."));
    } else {
      console.log(c("1;33", `  \u26A0 NO IMPROVEMENT: ${result.tokPerSec?.toFixed(2)} tok/s (${delta}%)`));
      history += `\nCycle ${cycle}: REVERTED \u2014 no improvement (${result.tokPerSec?.toFixed(2)} tok/s, ${delta}%)`;
      await revertAgentChanges();
    }

    // Log cycle result
    const logEntry = {
      cycle, effort,
      tokPerSec: result.tokPerSec,
      bandwidthUtil: result.bandwidthUtil,
      correct: result.correct,
      improved, broken,
      outputText: result.outputText?.slice(0, 200),
      timestamp: new Date().toISOString(),
    };
    const logPath = join(RESULTS_DIR, `effort_${effort}_log.jsonl`);
    await writeFile(logPath, JSON.stringify(logEntry) + "\n", { flag: "a" });
  }

  // Summary
  console.log(c("1;37", `\n${"═".repeat(58)}`));
  console.log(c("1;37", `  Effort ${effort} complete.`));
  console.log(c("1;37", `  Baseline: ${baseline.tokPerSec?.toFixed(2)} tok/s`));
  console.log(c("1;37", `  Best:     ${bestTokPerSec.toFixed(2)} tok/s`));
  if (bestTokPerSec > (baseline.tokPerSec ?? 0)) {
    const gain = ((bestTokPerSec - (baseline.tokPerSec ?? 0)) / (baseline.tokPerSec ?? 1) * 100).toFixed(1);
    console.log(c("1;32", `  Gain:     +${gain}%`));
  }
  console.log(c("1;37", `${"═".repeat(58)}\n`));
}

main().catch((e) => {
  console.error(c("1;31", `Fatal: ${e}`));
  process.exit(1);
});
