#!/usr/bin/env bun
/**
 * ZINC Self-Improving Optimization Loop
 *
 * Iteratively builds, deploys, and improves the ZINC inference engine on an
 * RDNA4 test node. Each cycle:
 *   1. rsync source to remote node
 *   2. Build (zig build -Doptimize=ReleaseFast)
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

const REPO_ROOT = resolve(import.meta.dir, "..");
let PROJECT_ROOT = REPO_ROOT;
let RESULTS_DIR = resolve(REPO_ROOT, ".zinc_optimize");

// Load .env
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
let REMOTE_ZINC_DIR = "/root/zinc";
const DEFAULT_MODEL = "/root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf";

const BLOCKED_GIT_OPS = [
  "Bash(git checkout:*)",
  "Bash(git revert:*)",
  "Bash(git restore:*)",
  "Bash(git reset:*)",
  "Bash(git stash:*)",
  "Bash(git clean:*)",
];

// Prevent the agent from editing files outside the engine codebase
const BLOCKED_FILE_OPS = [
  "Edit(site/*)",
  "Write(site/*)",
  "Edit(loops/*)",
  "Write(loops/*)",
  "Edit(docs/*)",
  "Write(docs/*)",
  "Edit(specs/*)",
  "Write(specs/*)",
  "Edit(writing/*)",
  "Write(writing/*)",
  "Edit(.github/*)",
  "Write(.github/*)",
  "Edit(.env)",
  "Write(.env)",
  "Edit(AGENTS.md)",
  "Write(AGENTS.md)",
  "Edit(CLAUDE.md)",
  "Write(CLAUDE.md)",
  "Edit(README.md)",
  "Write(README.md)",
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
  garbageOutput: boolean;
  coherentText: boolean; // true if decoded text contains real words
  bandwidthUtil: number | null; // % of theoretical bandwidth
  effectiveBW: number | null; // GB/s
  error: string | null;
};

/** Parse decode tok/s from ZINC output (the "Generated N tokens in T ms — X tok/s" line). */
export function parseTokPerSec(output: string): number | null {
  // Match the decode-specific line: "Generated 256 tokens in 5635.3 ms — 45.43 tok/s"
  const decodeMatch = output.match(/Generated\s+\d+\s+tokens\s+in\s+[\d.]+\s*(?:ms|s)\s*[—–-]\s*(\d+\.?\d*)\s*tok\/s/i);
  if (decodeMatch) return parseFloat(decodeMatch[1]);
  // Fallback: compute from "Generated N tokens in T ms/s"
  const genMatch = output.match(/Generated\s+(\d+)\s+tokens\s+in\s+(\d+\.?\d*)\s*(ms|s)/i);
  if (genMatch) {
    const tokens = parseInt(genMatch[1], 10);
    let seconds = parseFloat(genMatch[2]);
    if (genMatch[3] === "ms") seconds /= 1000;
    if (seconds > 0) return tokens / seconds;
  }
  // Last resort: any tok/s (will catch prefill-only output)
  const m = output.match(/(\d+\.?\d*)\s*tok\/s/i);
  if (m) return parseFloat(m[1]);
  return null;
}

/** Parse number of tokens generated from ZINC output. */
export function parseTokensGenerated(output: string): number {
  const m = output.match(/Generated\s+(\d+)\s+tokens/i);
  return m ? parseInt(m[1], 10) : 0;
}

/** Check if output tokens look like garbage (e.g. same token repeated). */
export function isGarbageOutput(output: string): boolean {
  // Match "Output tokens (N): { t1, t2, t3, ... }"
  const m = output.match(/Output tokens\s*\(\d+\)\s*:\s*\{([^}]+)\}/i);
  if (!m) return false; // can't tell, assume OK
  const tokens = m[1].split(",").map((s) => s.trim()).filter((s) => s.length > 0);
  if (tokens.length < 4) return false;
  // Check diversity: if >80% of tokens are the same value, it's garbage
  const counts = new Map<string, number>();
  for (const t of tokens) counts.set(t, (counts.get(t) ?? 0) + 1);
  const maxCount = Math.max(...counts.values());
  if (maxCount / tokens.length > 0.8) return true;
  // Check for short repeating patterns (e.g. ABCABC...)
  if (tokens.length >= 12) {
    for (const plen of [2, 3, 4, 5, 6]) {
      const pattern = tokens.slice(2, 2 + plen).join(",");
      let repeats = 0;
      for (let i = 2; i + plen <= tokens.length; i += plen) {
        if (tokens.slice(i, i + plen).join(",") === pattern) repeats++;
      }
      if (repeats >= 3 && (repeats * plen) / (tokens.length - 2) > 0.7) return true;
    }
  }
  // Also check via decoded text: if output is just numbers, punctuation, or BPE fragments
  const textMatch = output.match(/Output text:\s*(.+)/i);
  if (textMatch) {
    const text = textMatch[1].trim().slice(0, 200);
    // If >60% of characters are digits or single-char punctuation, it's garbage
    const alphaCount = (text.match(/[a-zA-Z]{2,}/g) ?? []).join("").length;
    if (text.length > 20 && alphaCount / text.length < 0.3) return true;
  }
  return false;
}

/** Check if decoded output text looks like coherent, CORRECT language (not just echoing the prompt). */
export function isCoherentText(output: string): boolean {
  const m = output.match(/Output text:\s*(.+)/i);
  if (!m) return false;
  const text = m[1].trim();
  if (text.length < 10) return false;

  // Reject: output that just repeats the prompt or a short pattern
  // (e.g., "The capital of France is is not. The capital of France is is not.")
  const deduped = text.slice(0, 200).replace(/[\u0120\u010A]/g, ' ').trim(); // Ġ→space, Ċ→space
  const words = deduped.split(/\s+/).filter(w => w.length > 0);
  if (words.length >= 8) {
    // Check if the first 4 words repeat in a cycle
    const prefix = words.slice(0, Math.min(8, words.length)).join(' ');
    const rest = words.slice(8).join(' ');
    if (rest.length > 0 && rest.includes(prefix.slice(0, Math.min(20, prefix.length)))) {
      return false; // repeating pattern
    }
  }

  // Must contain varied content words (not just function words)
  const sample = text.slice(0, 200).toLowerCase();
  const commonWords = ["the", "is", "of", "and", "to", "in", "a", "that", "it", "for", "was", "on", "are", "as", "with", "his", "they", "be", "at", "one", "have", "this", "from", "or", "an", "but", "not", "what", "all", "were", "we", "when", "your", "can", "said", "there", "each", "which", "she", "do", "how", "if", "will", "up", "about", "out", "many", "then", "them", "would", "like", "so", "these", "her", "think", "paris", "france", "capital", "city"];
  let found = 0;
  for (const w of commonWords) {
    // Match as whole word or surrounded by non-alpha
    if (new RegExp(`(^|[^a-z])${w}([^a-z]|$)`).test(sample)) found++;
  }
  return found >= 3;
}

/** Parse modeled bandwidth utilization percentage from ZINC output. */
export function parseBandwidthUtil(output: string): number | null {
  const m = output.match(/(\d+\.?\d*)%\s*utilization/i);
  return m ? parseFloat(m[1]) : null;
}

/** Parse modeled effective bandwidth in GB/s from ZINC output. */
export function parseEffectiveBW(output: string): number | null {
  const m = output.match(/(\d+\.?\d*)\s*GB\/s\s*effective/i);
  return m ? parseFloat(m[1]) : null;
}

/** Determine if we're in fix or optimize phase. */
export function detectPhase(result: BuildRunResult): Phase {
  if (result.buildExitCode !== 0) return "fix";
  if (result.runExitCode !== 0) return "fix";
  if (result.error) return "fix";
  // Stay in fix phase if generating tokens but output is garbage/incoherent
  if (result.tokPerSec != null && result.tokPerSec > 0 && !result.coherentText) return "fix";
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
    stdoutLineFormatter?: (line: string) => string | null;
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
      stderr = "",
      lineBuffer = "";
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
      `cd ${REMOTE_ZINC_DIR} && zig build -Doptimize=ReleaseFast 2>&1`,
    ],
    { streamOutput: true, timeout: 300_000 },
  );
  return { exitCode, output: stdout + "\n" + stderr };
}

/** Run `zig build test` on the remote node. Returns true if all tests pass. */
async function remoteTest(): Promise<{ passed: boolean; output: string }> {
  console.log(clr("2", "  Running tests..."));
  const { stdout, stderr, exitCode } = await runCommand(
    "ssh",
    [
      "-p", String(ZINC_PORT),
      "-o", "StrictHostKeyChecking=no",
      `${ZINC_USER}@${ZINC_HOST}`,
      `cd ${REMOTE_ZINC_DIR} && zig build test --summary all 2>&1`,
    ],
    { streamOutput: false, timeout: 120_000 },
  );
  const testPassed = stdout.match(/(\d+)\/\d+ tests passed/);
  if (testPassed) {
    console.log(clr("2", `  ✅ ${testPassed[0]}`));
  }
  if (exitCode !== 0) {
    console.log(clr("1;31", "  ❌ Tests failed!"));
  }
  return { passed: exitCode === 0, output: stdout + "\n" + stderr };
}

async function remoteRun(
  modelPath: string,
  prompt: string,
): Promise<{ exitCode: number; output: string }> {
  console.log(clr("2", "  Running ZINC on remote node (acquiring GPU lock)..."));
  const runCmd = `cd ${REMOTE_ZINC_DIR} && RADV_PERFTEST=coop_matrix timeout 90 ./zig-out/bin/zinc -m ${modelPath} --prompt "${prompt}" 2>&1`;
  const { stdout, stderr, exitCode } = await runCommand(
    "ssh",
    [
      "-p", String(ZINC_PORT),
      "-o", "StrictHostKeyChecking=no",
      `${ZINC_USER}@${ZINC_HOST}`,
      // flock serializes GPU access — only one inference run at a time
      `flock /tmp/zinc-gpu.lock -c '${runCmd.replace(/'/g, "'\\''")}'`,
    ],
    { streamOutput: true, timeout: 180_000 }, // extra time to wait for lock
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
      garbageOutput: false,
      coherentText: false,
      bandwidthUtil: null,
      effectiveBW: null,
      error: "Build failed",
    };
  }

  // Run
  const run = await remoteRun(modelPath, "The capital of France is");
  const tps = parseTokPerSec(run.output);
  const tokensGenerated = parseTokensGenerated(run.output);
  const garbage = isGarbageOutput(run.output);
  const coherent = isCoherentText(run.output);
  const bwUtil = parseBandwidthUtil(run.output);
  const effBW = parseEffectiveBW(run.output);

  const result: BuildRunResult = {
    buildExitCode: 0,
    buildOutput: build.output,
    runExitCode: run.exitCode,
    runOutput: run.output,
    phase: "fix",
    tokPerSec: tps,
    tokensGenerated,
    garbageOutput: garbage || (!coherent && tokensGenerated > 10),
    coherentText: coherent,
    bandwidthUtil: bwUtil,
    effectiveBW: effBW,
    error: run.exitCode !== 0 ? `Runtime exit code ${run.exitCode}` : null,
  };
  result.phase = detectPhase(result);
  return result;
}

// ── Claude stream formatter ──────────────────────────────────────────

type ClaudeStreamState = {
  currentToolName: string | null;
  currentBlockIsToolUse: boolean;
  inputJsonBuffer: string;
  inTextBlock: boolean;
  sawTextDeltaInCurrentMessage: boolean;
};

const MAX_DIFF_LINES = 8;

function formatToolInput(toolName: string, inputJson: string): string {
  let input: Record<string, unknown> = {};
  try { input = JSON.parse(inputJson) as Record<string, unknown>; } catch { /* partial */ }
  const name = toolName.toLowerCase();
  const out: string[] = [];

  if (name === "edit") {
    const fp = (input.file_path as string | undefined) ?? "?";
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")}`));
    const oldLines = ((input.old_string as string | undefined) ?? "").split("\n");
    const newLines = ((input.new_string as string | undefined) ?? "").split("\n");
    for (const l of oldLines.slice(0, MAX_DIFF_LINES)) out.push(clr("31", `   - ${l}`));
    if (oldLines.length > MAX_DIFF_LINES) out.push(clr("2", `   - … (${oldLines.length - MAX_DIFF_LINES} more)`));
    for (const l of newLines.slice(0, MAX_DIFF_LINES)) out.push(clr("32", `   + ${l}`));
    if (newLines.length > MAX_DIFF_LINES) out.push(clr("2", `   + … (${newLines.length - MAX_DIFF_LINES} more)`));
  } else if (name === "write") {
    const fp = (input.file_path as string | undefined) ?? "?";
    const lineCount = ((input.content as string | undefined) ?? "").split("\n").length;
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")} (${lineCount} lines)`));
  } else if (name === "bash") {
    const cmd = (input.command as string | undefined) ?? "?";
    out.push(clr("2", `   $ ${cmd.length > 120 ? cmd.slice(0, 120) + "…" : cmd}`));
  } else if (name === "read") {
    const fp = (input.file_path as string | undefined) ?? "?";
    const offset = input.offset != null ? ` @line ${input.offset}` : "";
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")}${offset}`));
  } else if (name === "grep") {
    const pattern = (input.pattern as string | undefined) ?? "?";
    const path = (input.path as string | undefined) ?? "";
    out.push(clr("2", ` → /${pattern}/${path ? ` in ${path.split("/").slice(-2).join("/")}` : ""}`));
  } else if (name === "glob") {
    out.push(clr("2", ` → ${(input.pattern as string | undefined) ?? "?"}`));
  }
  return out.length > 0 ? out.join("\n") + "\n" : "";
}

function formatToolResult(result: Record<string, unknown>): string {
  const file = result.file as Record<string, unknown> | undefined;
  if (file) return clr("32", `   ☑ accepted`) + clr("2", `  (${file.numLines ?? "?"} lines)`) + "\n";
  const content = coerceDisplayText(result.content);
  if (!content.trim()) return clr("32", "   ☑ accepted") + "\n";
  const lines = content.split("\n").filter((l) => l.trim());
  const tail = lines.slice(-3);
  const ellipsis = lines.length > 3 ? clr("2", "   …\n") : "";
  const body = tail.map((l) => clr("2", `   ${l.trim()}`)).join("\n");
  return clr("32", "   ☑ accepted") + "\n" + ellipsis + body + "\n";
}

function coerceDisplayText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    const parts = value.map((e) => coerceDisplayText(e)).filter((e) => e.trim());
    if (parts.length > 0) return parts.join("\n");
    try { return JSON.stringify(value, null, 2); } catch { return ""; }
  }
  if (typeof value === "object") {
    const r = value as Record<string, unknown>;
    const parts = [r.text, r.message, r.output, r.stdout, r.stderr, r.content, r.result]
      .map((e) => coerceDisplayText(e)).filter((e) => e.trim());
    if (parts.length > 0) return parts.join("\n");
    try { return JSON.stringify(r, null, 2); } catch { return ""; }
  }
  return "";
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
        return `\n${clr("33", `🔧 ${state.currentToolName}`)}`;
      }
      if (block?.type === "text") {
        state.inTextBlock = true;
        state.currentBlockIsToolUse = false;
        return COLOR_ENABLED ? "\n\x1b[96m" : "\n";
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
        return COLOR_ENABLED ? "\x1b[0m\n" : "\n";
      }
      return null;
    }
    return null;
  }
  if (event.type === "user") {
    const result = event.tool_use_result as Record<string, unknown> | undefined;
    if (result) return formatToolResult(result);
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
      return clr("96", text) + "\n";
    }
    return null;
  }
  return null;
}

// ── Agent invocation ─────────────────────────────────────────────────

function buildClaudeArgs(prompt: string): string[] {
  return [
    "-p",
    "--verbose",
    "--output-format", "stream-json",
    "--include-partial-messages",
    `--disallowed-tools=${[...BLOCKED_GIT_OPS, ...BLOCKED_FILE_OPS].join(",")}`,
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

  const claudeState: ClaudeStreamState = {
    currentToolName: null,
    currentBlockIsToolUse: false,
    inputJsonBuffer: "",
    inTextBlock: false,
    sawTextDeltaInCurrentMessage: false,
  };

  const result = await runCommand("claude", buildClaudeArgs(prompt), {
    streamOutput: true,
    timeout: 900_000, // 15 min max per agent call
    stdoutLineFormatter: (line) => formatClaudeStreamLine(line, claudeState),
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
        .slice(-15)
        .map((h) => {
          const desc = trunc(h.description, 70);
          const bw = h.bandwidthUtil != null ? `, ${h.bandwidthUtil.toFixed(0)}% BW` : "";
          const snippet = (h as any).outputSnippet ? ` out="${trunc((h as any).outputSnippet, 30)}"` : "";
          const coherent = (h as any).coherentText ? " ✅COHERENT" : "";
          return `  #${h.cycle}: [${h.phase}] ${desc} → ${h.kept ? "KEPT" : "REVERTED"}${h.tokPerSec != null ? ` (${h.tokPerSec.toFixed(1)} tok/s${bw})` : ""}${snippet}${coherent}`;
        })
        .join("\n")
      : "  (none yet)";

  // Include the last cycle's self-analysis so the agent can build on its own reasoning
  const lastCycle = cycles.length > 0 ? cycles[cycles.length - 1] : null;
  const lastAnalysisBlock = lastCycle?.selfAnalysis
    ? `## Last Cycle's Analysis (cycle #${lastCycle.cycle}, ${lastCycle.kept ? "KEPT" : "REVERTED"})\n${lastCycle.selfAnalysis}\n\nNext ideas from last cycle: ${lastCycle.nextIdeas.join(", ") || "(none)"}`
    : "";

  // Detect stall: count consecutive cycles where output text didn't change
  let stall_count = 0;
  if (cycles.length >= 3) {
    const currentSnippet = lastResult.runOutput.match(/Output text:\s*(.{0,80})/)?.[1]?.trim() ?? "";
    // Count recent cycles with the same output snippet
    for (let i = cycles.length - 1; i >= Math.max(0, cycles.length - 10); i--) {
      const prev = (cycles[i] as any).outputSnippet ?? "";
      if (prev === currentSnippet || !cycles[i].kept) {
        stall_count++;
      } else break;
    }
  }

  // Include accumulated ideas from all cycles
  const ideasBlock = state.ideas.length > 0
    ? state.ideas.slice(-15).map((idea, i) => `  ${i + 1}. ${trunc(idea, 120)}`).join("\n")
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

  // Build precise diagnosis from the actual result
  const diagnosis: string[] = [];

  if (lastResult.buildExitCode !== 0) {
    diagnosis.push("## Status: BUILD FAILURE");
    diagnosis.push("The Zig or GLSL shader build is failing. Fix the compilation error shown below.");
  } else if (lastResult.runExitCode !== 0 && lastResult.runExitCode !== null) {
    diagnosis.push(`## Status: RUNTIME CRASH (exit code ${lastResult.runExitCode})`);
    diagnosis.push("Build succeeds but ZINC crashes at runtime. Fix the crash shown below.");
  } else if (lastResult.tokensGenerated <= 1) {
    diagnosis.push(`## Status: ENGINE NOT GENERATING (${lastResult.tokensGenerated} tokens)`);
    diagnosis.push("Build and run succeed, model loads, but the forward pass doesn't produce real output.");
    diagnosis.push("");
    diagnosis.push("The most likely issue is that the forward pass `decodeStep` in `src/compute/forward.zig`");
    diagnosis.push("records barriers but doesn't actually dispatch compute shaders with real buffer bindings.");
    diagnosis.push("The shaders need descriptor sets bound to actual model weight buffers.");
    diagnosis.push("");
    diagnosis.push("Key things to wire up:");
    diagnosis.push("- Descriptor sets binding model weight buffers to shader pipelines");
    diagnosis.push("- Push constants with correct M/K dimensions for each layer");
    diagnosis.push("- Token embedding lookup (read from token_embd.weight tensor)");
    diagnosis.push("- Actual DMMV dispatches for QKV projections, FFN layers");
  } else if (lastResult.tokPerSec != null && lastResult.tokPerSec > 0) {
    const bwInfo = lastResult.bandwidthUtil != null
      ? ` | modeled ${lastResult.effectiveBW?.toFixed(0) ?? "?"} GB/s (${lastResult.bandwidthUtil.toFixed(0)}% of 576 GB/s)`
      : "";
    diagnosis.push(`## Status: RUNNING — ${lastResult.tokPerSec.toFixed(1)} tok/s (DECODE), ${lastResult.tokensGenerated} tokens${bwInfo}`);
    if (lastResult.garbageOutput && !lastResult.coherentText) {
      diagnosis.push("");
      diagnosis.push("⚠ CRITICAL: Output is GARBAGE — decoded text contains no recognizable words.");
      diagnosis.push("  The forward pass is executing but producing incorrect values.");
      diagnosis.push("  Common causes for this Qwen3.5 hybrid (attention+SSM+MoE) model:");
      diagnosis.push("  - Q/K/V split from fused attn_q.weight is wrong (Q includes gate, needs proper stride extraction)");
      diagnosis.push("  - RoPE freq_base or section dimensions incorrect");
      diagnosis.push("  - Flash attention shader bugs (scaling, causal mask, GQA head mapping)");
      diagnosis.push("  - SSM delta-net state dimensions mismatched (head_k_dim vs head_v_dim)");
      diagnosis.push("  - MoE expert weight offset calculation wrong for stacked tensors");
      diagnosis.push("  - Missing shared expert path (ffn_*_shexp tensors not dispatched)");
      diagnosis.push("  - Post-attention norm applied to wrong buffer");
      diagnosis.push("  FIXING OUTPUT CORRECTNESS is THE #1 PRIORITY. Do NOT optimize tok/s until output is coherent.");
      diagnosis.push("  Compare against llama.cpp's qwen35moe.cpp for the correct computation flow.");
    } else if (lastResult.garbageOutput) {
      diagnosis.push("");
      diagnosis.push("⚠ WARNING: Output tokens are repetitive but text partially recognizable.");
      diagnosis.push("  Fixing output quality is HIGHER PRIORITY than optimizing tok/s.");
    }
    diagnosis.push("");
    if (lastResult.coherentText) {
      diagnosis.push("✅ Output is COHERENT — decoded text contains recognizable language.");
      diagnosis.push("");
      if (lastResult.bandwidthUtil != null && lastResult.bandwidthUtil < 15) {
        diagnosis.push("## BOTTLENECK: LOW GPU UTILIZATION, NOT A HARD BANDWIDTH WALL");
        diagnosis.push(`Modeled decode bandwidth utilization is ${lastResult.bandwidthUtil.toFixed(1)}%, so single-stream decode is not close to saturating DRAM.`);
        diagnosis.push(`Current latency is ${(1000 / (lastResult.tokPerSec ?? 1)).toFixed(0)}ms/tok, which points at dispatch/setup/sync overhead plus fragmented decode work.`);
        diagnosis.push("");
        diagnosis.push("## OPTIMIZATION PRIORITY:");
        diagnosis.push("1. **Benchmark the clean path** — do not force `--debug` in throughput runs");
        diagnosis.push("2. **Use low-perturbation `--profile` output** — find full-token GPU time before changing kernels");
        diagnosis.push("3. **Reduce tail and sync overhead** — GPU argmax / less logits readback for greedy sampling");
        diagnosis.push("4. **Reduce Vulkan setup overhead** — descriptor reuse, fewer barriers, less per-token recording churn");
        diagnosis.push("5. **Fuse same-input decode work** — especially MoE gate/up and related projection fans");
        diagnosis.push("");
        diagnosis.push("## IMPORTANT: Make ONE meaningful architectural change per cycle.");
        diagnosis.push("Multi-file changes are OK if they're part of the same optimization.");
      } else if (lastResult.bandwidthUtil != null && lastResult.bandwidthUtil < 70) {
        diagnosis.push(`Modeled decode bandwidth utilization is ${lastResult.bandwidthUtil.toFixed(0)}% — still well below saturation.`);
        diagnosis.push("Focus on reducing Vulkan overhead, cutting host-visible sync, and improving decode kernel occupancy.");
      } else if (lastResult.bandwidthUtil != null) {
        diagnosis.push(`Modeled decode bandwidth utilization is ${lastResult.bandwidthUtil.toFixed(0)}% — approaching the point where algorithmic byte reduction matters.`);
        diagnosis.push("Further gains require reducing bytes read or algorithmic changes.");
      }
    } else {
      diagnosis.push("ZINC is generating tokens but output is NOT coherent. Fix correctness first.");
      diagnosis.push("Focus on:");
      diagnosis.push("1. Check DMMV shader correctness for each quant type");
      diagnosis.push("2. Verify buffer bindings and descriptor set wiring");
      diagnosis.push("3. Compare against llama.cpp's qwen35moe.cpp for the correct computation flow");
    }
  } else {
    diagnosis.push(`## Status: RUNNING BUT NO METRICS (${lastResult.tokensGenerated} tokens generated)`);
    diagnosis.push("The engine runs and generates tokens but isn't reporting tok/s metrics.");
    diagnosis.push("Add timing measurement to the decode loop and report tok/s.");
  }

  diagnosis.push("");
  diagnosis.push("## Forward Pass Architecture (Qwen3.5-35B-A3B = hybrid attention+SSM+MoE)");
  diagnosis.push("- 40 layers: every 4th (3,7,11,...,39) is full attention, rest are SSM/delta-net");
  diagnosis.push("- Full attention layers: separate attn_q/k/v + Q/K norm + IMRoPE (64/256 dims) + flash attention + sigmoid gate + output proj");
  diagnosis.push("- SSM layers: TWO PATHS available (selected at runtime by pipeline availability):");
  diagnosis.push("  - GPU path (runSsmLayerGpu): conv1d_silu.comp → delta_net.comp → gated_norm.comp → ssm_out DMMV — NO readback");
  diagnosis.push("  - CPU fallback (runSsmLayerCpu): GPU projections → readback → CPU conv1d+delta-net → upload → GPU ssm_out");
  diagnosis.push("  GPU path is gated by: if (self.elementwise.pipeline_ssm_conv1d != null)");
  diagnosis.push("- MoE FFN (all layers): router → softmax_topk (GPU or CPU) → top-8 experts → gate/up/SwiGLU/down → weighted accumulate + shared expert");
  diagnosis.push("  GPU router gated by: if (self.elementwise.pipeline_softmax_topk != null)");
  diagnosis.push("  Shared expert gate: GPU sigmoid_scale_acc (gated by pipeline availability) or CPU readback fallback");
  diagnosis.push("- Command buffer batching: single cmd buffer for all 40 layers, only submit for MoE expert ID readback");
  diagnosis.push("- head_dim=256 (from attention.key_length), hidden_dim=2048, n_heads=16, n_kv_heads=2");
  diagnosis.push("- rope_dim=64, rope_freq_base=10000000.0 (10M)");
  diagnosis.push("- Profiling: --profile flag enables Vulkan timestamp queries + per-token GPU timing summary");
  diagnosis.push("");
  diagnosis.push("## What's ALREADY WORKING (do not break)");
  diagnosis.push("- ✅ Forward pass: coherent output at 33.6 tok/s clean CLI on an idle RDNA4 node, all 40 layers correct");
  diagnosis.push("- ✅ GPU SSM path: conv1d + delta-net + gated_norm shaders WORKING after swiglu_buf overflow fix");
  diagnosis.push("- ✅ GPU softmax_topk.comp: rewritten with shared memory (no subgroupBallot), RADV-compatible");
  diagnosis.push("- ✅ GPU sigmoid_scale_acc.comp: shared expert gate on GPU, no readback");
  diagnosis.push("- ✅ GPU argmax fast path: greedy decode reads back one token id instead of full logits on the fast path");
  diagnosis.push("- ✅ Command buffer batching: fast path stays in one submission through final norm + LM head + argmax");
  diagnosis.push("- ✅ Persistent GPU SSM state buffers (conv: 3.75MB, recurrent: 80MB)");
  diagnosis.push("- ✅ OpenAI-compatible API server with SSE streaming, chat UI at GET /, raw /v1/completions at ~33.5 tok/s");
  diagnosis.push("- ✅ zig build test passes. Build clean on macOS and Linux.");
  diagnosis.push("");
  diagnosis.push("## CURRENT STATE (2026-03-30, updated)");
  diagnosis.push("Output: CORRECT at 33.6 tok/s clean CLI and ~33.5 tok/s raw API on a clean ReleaseFast build.");
  diagnosis.push("Reasoning chat is still slower: one longer chat sample produced 257 completion tokens at ~28.4 tok/s.");
  diagnosis.push("Modeled decode bandwidth at 33.58 tok/s is ~112.5 GB/s (~19.5% of 576 GB/s). Single-stream decode will not saturate DRAM.");
  diagnosis.push("Remaining bottlenecks are medium/small decode kernels, chat template/stop-path overhead, and still-intrusive profiling.");
  diagnosis.push("All changes MUST pass `zig build test`.");
  diagnosis.push("");
  diagnosis.push("## OPTIMIZATION PRIORITY (in order):");
  diagnosis.push("The path from today's >30 tok/s raw decode to >30 tok/s reasoning chat and higher aggregate GPU utilization:");
  diagnosis.push("");
  diagnosis.push("### 1. Benchmark and close the reasoning chat gap");
  diagnosis.push("- Compare `/v1/chat/completions` against raw `/v1/completions` with longer reasoning prompts");
  diagnosis.push("- Measure where chat loses throughput: template application, stop handling, or extra server-side work");
  diagnosis.push("- Prefer changes that preserve raw decode throughput while lifting the templated chat path above 30 tok/s");
  diagnosis.push("");
  diagnosis.push("### 2. Make profiling representative");
  diagnosis.push("- Use `--profile`, but first reduce its own overhead so it does not cut ReleaseFast throughput in half");
  diagnosis.push("- Profile full-token GPU time, not partial decode work");
  diagnosis.push("- Use the result to separate DMMV cost from elementwise and control overhead");
  diagnosis.push("");
  diagnosis.push("### 3. Reduce hot-path Vulkan setup and binding churn");
  diagnosis.push("- Reuse descriptor sets and static bindings where possible");
  diagnosis.push("- Trim per-token descriptor updates and other host work that does not contribute useful math");
  diagnosis.push("");
  diagnosis.push("### 4. Tune the real hot decode shapes");
  diagnosis.push("- Focus on the medium/small decode projections that dominate single-token latency");
  diagnosis.push("- LM head matters, but it is not the only or even the main remaining limiter");
  diagnosis.push("");
  diagnosis.push("### 5. Use batching for higher aggregate bandwidth");
  diagnosis.push("- Single-stream decode above 30 tok/s is already achieved");
  diagnosis.push("- If the goal is materially higher memory-bandwidth utilization, add concurrent decode / batching");
  diagnosis.push("- Profile actual occupancy and tune tile sizes");
  diagnosis.push("");
  diagnosis.push("## APPROACHES THAT FAILED (do not retry):");
  diagnosis.push("- Spin-wait on fences (vkGetFenceStatus) — no improvement");
  diagnosis.push("- Simple submit elimination within existing architecture — too small an effect");
  diagnosis.push("- Pre-allocating SSM buffers — saves <1ms, irrelevant");
  diagnosis.push("- subgroupBallot in softmax_topk — crashes RADV at wave64 (fixed: use shared memory)");
  diagnosis.push("- Enabling GPU SSM without swiglu_buf size fix — buffer overflow, wrong output (fixed)");
  diagnosis.push("");
  diagnosis.push("## REFERENCE: llama.cpp implementation");
  diagnosis.push("On the remote node, the full Qwen3.5-MoE implementation is at:");
  diagnosis.push("  /root/llama.cpp/src/models/qwen35moe.cpp — build_layer_attn, build_layer_attn_linear, build_layer_ffn");
  diagnosis.push("  /root/llama.cpp/src/models/delta-net-base.cpp — build_delta_net_autoregressive");
  diagnosis.push("You can read these files via SSH to understand the correct computation flow.");
  diagnosis.push("Use `ssh -p $ZINC_PORT $ZINC_USER@$ZINC_HOST 'cat /root/llama.cpp/src/models/qwen35moe.cpp'`");
  diagnosis.push("");
  diagnosis.push("## RDNA4 Constraints");
  diagnosis.push("- RADV may not support all GL_KHR_cooperative_matrix features — stub out or use fallbacks");
  diagnosis.push("- Shared memory: 64KB max per workgroup");
  diagnosis.push("- glslc from Ubuntu 24.04 (shaderc 2023.8) — newer versions break RADV");
  diagnosis.push("- wave64 optimal, workgroup size 64");

  const phaseLabel = phase === "fix" ? "FIX" : "OPTIMIZE";

  return [
    `# ZINC ${phaseLabel} Task`,
    "",
    ...diagnosis,
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
    "src/shaders/  — dmmv_{q4k,q5k,q6k,q8_0,f16,f32}.comp, rms_norm_mul.comp,",
    "                swiglu.comp, rope_fused.comp, flash_attn.comp, sigmoid_mul.comp,",
    "                vadd.comp, scale_accumulate.comp, deinterleave.comp, coop_matmul.comp,",
    "                ssm_conv1d.comp, ssm_delta_net.comp, ssm_gated_norm.comp (NEW — GPU SSM),",
    "                softmax_topk.comp (NEW — GPU MoE router),",
    "                sigmoid_scale_acc.comp, scale_acc_sigmoid.comp (shared expert gate),",
    "                argmax.comp, embed_dequant_q4k.comp",
    "src/server/   — http.zig (HTTP parser+SSE), routes.zig (endpoint handlers), session.zig, chat.html",
    "src/scheduler/ — scheduler.zig, request.zig, kv_cache.zig (paged KV pool)",
    "src/main.zig  — CLI (--profile) + server mode (no --prompt → HTTP server)",
    "build.zig     — build system with conditional shader compilation (27 shaders)",
    "```",
    "",
    "## Optimization History (last 20 cycles)",
    historyBlock,
    "",
    "## FAILED APPROACHES — DO NOT REPEAT",
    failedBlock,
    "",
    lastAnalysisBlock,
    "",
    ...(stall_count >= 3 ? [
      "## ⚠ STALL DETECTED — " + stall_count + " cycles with no meaningful output change",
      "Your recent changes had NO EFFECT on the output. You must try a FUNDAMENTALLY DIFFERENT approach.",
      "Do NOT make incremental tweaks. Instead:",
      "- SSH into the remote node and READ llama.cpp's qwen35moe.cpp to understand the correct flow",
      "- Add a CPU reference computation for one specific operation and compare against GPU",
      "- Verify that GPU buffer contents match what you expect by adding readback + logging",
      "- Check if the Q4_K DMMV shader's sub-block decoding matches GGML's reference implementation",
      "- Try disabling entire subsystems (skip attention, skip MoE, skip SSM) to isolate which part is wrong",
      "Previous approaches that were tried and had no effect should NOT be repeated.",
      "",
    ] : []),
    "## Accumulated Ideas (from all previous cycles)",
    ideasBlock,
    "",
    "## Rules",
    `1. Make ONE focused ${phase === "optimize" ? "architectural" : ""} change.${phase === "optimize" ? " You MAY edit multiple files if they're all part of the same optimization (e.g., new shader + dispatch changes)." : " Do not try multiple things."}`,
    "2. Edit LOCAL source files only (this machine's working copy).",
    "3. The loop will rsync your changes and rebuild on the remote node.",
    "4. Do NOT modify .env, loops/, or files outside src/ and build.zig.",
    "5. Zig 0.15.2 API: ArrayList is unmanaged (pass allocator to append/deinit),",
    "   StringHashMap → use StringHashMapUnmanaged, File.stdout() not io.getStdOut(),",
    "   writer() takes a buffer arg, process.Child StdIo uses .Pipe not .pipe.",
    "6. GLSL shaders must compile with `glslc --target-env=vulkan1.3 -O`.",
    "7. For correctness issues: READ llama.cpp's qwen35moe.cpp on the remote node",
    "   to understand the correct computation. Use the SSH command above.",
    "8. When fixing the forward pass, focus on ONE layer type at a time (attention OR SSM).",
    "9. Add log.info debug output to verify intermediate values when debugging correctness.",
    "10. The Q8_0 DMMV shader dispatches (M+1)/2 workgroups (2 rows/WG), NOT (M+63)/64.",
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
  bandwidthUtil: number | null;
  effectiveBW: number | null;
  buildExitCode: number;
  runExitCode: number | null;
  garbageOutput: boolean;
  coherentText: boolean;
  outputSnippet: string; // first 80 chars of output text for change detection
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
  currentBest: { tokPerSec: number | null; tokensGenerated?: number } | null;
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
  worktreeName?: string,
): Promise<CycleResult> {
  const cycleNum = state.cycles.length + 1;
  const cycleDir = join(runDir, `cycle-${String(cycleNum).padStart(3, "0")}`);
  await mkdir(cycleDir, { recursive: true });

  console.log(clr("1;35", "\n" + "═".repeat(64)));
  console.log(clr("1;35", `  CYCLE ${cycleNum}${worktreeName ? ` [${worktreeName}]` : ""}`));
  console.log(clr("1;35", "═".repeat(64)));

  // Step 0 (worktree only): Rebase on main to pick up changes from the other loop
  if (worktreeName) {
    const rebase = await runCommand(
      "git", ["rebase", "main"],
      { cwd: PROJECT_ROOT },
    ).catch(() => null);
    if (rebase && rebase.exitCode !== 0) {
      await runCommand("git", ["rebase", "--abort"], { cwd: PROJECT_ROOT }).catch(() => { });
      console.log(clr("1;33", "  ⚠ Rebase on main had conflicts — continuing with current state"));
    }
  }

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

  // Print current status with actual phase
  if (buildRun.buildExitCode !== 0) {
    console.log(clr("1;31", `  ❌ BUILD FAILED (exit ${buildRun.buildExitCode})`));
  } else if (buildRun.runExitCode !== 0 && buildRun.runExitCode !== null) {
    console.log(clr("1;31", `  ❌ RUNTIME CRASH (exit ${buildRun.runExitCode})`));
  } else if (buildRun.tokPerSec != null && buildRun.tokPerSec > 0) {
    const qualityTag = buildRun.garbageOutput
      ? (buildRun.coherentText ? clr("1;33", " [REPETITIVE]") : clr("1;31", " [GARBAGE - NOT COHERENT]"))
      : (buildRun.coherentText ? clr("1;32", " [COHERENT ✓]") : "");
    const bwTag = buildRun.bandwidthUtil != null ? `, ${buildRun.effectiveBW?.toFixed(0) ?? "?"} GB/s (${buildRun.bandwidthUtil.toFixed(0)}% BW)` : "";
    console.log(clr("1;32", `  ✅ RUNNING — ${buildRun.tokPerSec.toFixed(1)} tok/s, ${buildRun.tokensGenerated} tokens${bwTag}`) + qualityTag);
  } else {
    console.log(clr("1;33", `  ⚠ ENGINE NOT GENERATING — build OK, ${buildRun.tokensGenerated} tokens produced`));
  }

  // Step 2: Git snapshot before agent changes — commit current state so we can revert cleanly
  await runCommand("git", ["add", "-A", "src/", "build.zig", "build.zig.zon", "benchmarks/"], { cwd: PROJECT_ROOT }).catch(() => { });
  await runCommand("git", ["commit", "--allow-empty", "-m", `zinc-loop: pre-cycle-${cycleNum} checkpoint`], { cwd: PROJECT_ROOT }).catch(() => { });
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
  let description = rawDesc && !isGarbageString(rawDesc) ? rawDesc : "";
  // Fallback: summarize from git diff if agent didn't emit @@@DESCRIPTION
  if (!description) {
    try {
      const diff = await runCommand("git", ["diff", "--stat", preHash, "HEAD"], { cwd: PROJECT_ROOT });
      const files = diff.stdout.split("\n")
        .filter((l) => l.includes("|"))
        .map((l) => l.trim().split("|")[0].trim().split("/").pop())
        .filter(Boolean)
        .slice(0, 3);
      description = files.length > 0
        ? `Modified ${files.join(", ")}`
        : "Agent made changes";
    } catch {
      description = "Agent made changes";
    }
  }
  const selfAnalysis = analysisMatch?.[1]?.trim() ?? "";
  const newIdeas =
    ideasMatch?.[1]
      ?.split(",")
      .map((s) => s.trim())
      .filter((s) => s.length > 3 && s.length < 120 && !isGarbageString(s)) ?? [];

  // Step 4: Verify — rsync + build + test + run
  console.log(clr("1;33", "\n  📊 Verifying agent's changes..."));
  let verifyResult: BuildRunResult;
  try {
    await rsyncToRemote();
    // Run tests first — if tests break, treat as build failure
    const testResult = await remoteTest();
    if (!testResult.passed) {
      console.log(clr("1;31", "  ❌ Tests broken by agent's changes — reverting"));
      verifyResult = {
        buildExitCode: 1,
        buildOutput: "Tests failed:\n" + testResult.output,
        runExitCode: null,
        runOutput: "",
        phase: "fix",
        tokPerSec: null,
        tokensGenerated: 0,
        garbageOutput: false,
        coherentText: false,
        bandwidthUtil: null,
        effectiveBW: null,
        error: "Tests failed after agent changes",
      };
      // Skip buildAndRun — go straight to keep/revert decision
    } else {
      verifyResult = await buildAndRun(modelPath);
    }
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
      garbageOutput: false,
      coherentText: false,
      bandwidthUtil: null,
      effectiveBW: null,
      error: String(e),
    };
  }

  // Step 5: Decide keep or revert
  let keep = false;

  if (buildRun.phase === "fix") {
    // In fix mode: keep if we made progress (fewer errors, coherent output, or moved to optimize phase)
    if (verifyResult.phase === "optimize") {
      keep = true; // We fixed it!
      console.log(clr("1;32", "  🎉 FIXED! Output is coherent. Moving to optimize phase."));
    } else if (
      verifyResult.coherentText && !buildRun.coherentText
    ) {
      keep = true; // Output became coherent!
      console.log(clr("1;32", "  🎉 Output is now COHERENT! Major progress."));
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
      buildRun.garbageOutput && !verifyResult.garbageOutput &&
      verifyResult.buildExitCode === 0 && verifyResult.runExitCode === 0
    ) {
      keep = true; // Output is no longer garbage
      console.log(clr("1;32", "  📈 Output is no longer garbage! Progress."));
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
      // Both succeed with same token count — only keep if output actually CHANGED
      // (prevents accepting no-op cycles that waste time)
      const oldOut = buildRun.runOutput.match(/Output text:\s*(.+)/)?.[1]?.trim() ?? "";
      const newOut = verifyResult.runOutput.match(/Output text:\s*(.+)/)?.[1]?.trim() ?? "";
      if (newOut !== oldOut && newOut.length > 0) {
        keep = true;
        console.log(clr("1;33", `  ↔ Output changed (${verifyResult.tokensGenerated} tokens). Keeping.`));
      } else {
        console.log(clr("2", `  ↔ Output unchanged — not keeping no-op change.`));
      }
    } else if (
      verifyResult.buildExitCode === 0 &&
      buildRun.buildExitCode === 0 &&
      buildRun.runExitCode !== 0 &&
      verifyResult.runExitCode !== 0 &&
      verifyResult.tokensGenerated > buildRun.tokensGenerated
    ) {
      // Both crash at runtime but fix produces more tokens — forward progress
      keep = true;
      console.log(clr("1;32", `  📈 Still crashing but more tokens: ${buildRun.tokensGenerated} → ${verifyResult.tokensGenerated}`));
    } else if (
      verifyResult.buildExitCode === 0 &&
      buildRun.buildExitCode === 0 &&
      buildRun.runExitCode !== 0 &&
      verifyResult.runExitCode !== 0 &&
      verifyResult.tokPerSec != null &&
      buildRun.tokPerSec == null
    ) {
      // Baseline had no metrics, fix now reports tok/s — forward progress
      keep = true;
      console.log(clr("1;32", `  📈 Now reporting ${verifyResult.tokPerSec.toFixed(1)} tok/s (was: no metrics)`));
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
    // In optimize mode: keep if tok/s improved meaningfully
    // When below 10 tok/s, accept any improvement ≥ 0.5 tok/s (noise floor is ~0.2)
    // When above 10 tok/s, require +2% or +3 tok/s (whichever is larger)
    const globalBest = state.currentBest?.tokPerSec ?? 0;
    const minThreshold = (buildRun.tokPerSec ?? 0) < 10
      ? 0.5  // large architectural changes should produce at least +0.5 tok/s
      : Math.max(3.0, buildRun.tokPerSec! * 0.02);
    if (
      verifyResult.phase === "optimize" &&
      verifyResult.tokPerSec != null &&
      buildRun.tokPerSec != null &&
      verifyResult.tokPerSec >= buildRun.tokPerSec + minThreshold
    ) {
      if (verifyResult.garbageOutput) {
        console.log(clr("1;33", `  ⚠ Output is garbage (repeated tokens) — not keeping despite tok/s improvement`));
      } else {
        keep = true;
        console.log(
          clr(
            "1;32",
            `  📈 Improved: ${buildRun.tokPerSec.toFixed(1)} → ${verifyResult.tokPerSec.toFixed(1)} tok/s (threshold: +${minThreshold.toFixed(1)}, global best: ${globalBest.toFixed(1)})`,
          ),
        );
      }
    } else if (
      verifyResult.phase === "optimize" &&
      verifyResult.tokPerSec != null &&
      buildRun.tokPerSec != null
    ) {
      const delta = verifyResult.tokPerSec - buildRun.tokPerSec;
      const reason = verifyResult.tokPerSec < globalBest
        ? `below global best (${globalBest.toFixed(1)})`
        : `below threshold (+${minThreshold.toFixed(1)})`;
      console.log(clr("2", `  ℹ ${delta >= 0 ? "+" : ""}${delta.toFixed(1)} tok/s — ${reason}`));
    }
  }

  // Quality gate: block keeping changes that degrade output quality
  // But don't penalize garbage output when baseline was crashing — garbage > crash
  if (keep && verifyResult.garbageOutput && !buildRun.garbageOutput && buildRun.runExitCode === 0) {
    console.log(clr("1;33", "  ⚠ Agent introduced garbage output — reverting"));
    keep = false;
  }
  if (keep && buildRun.coherentText && !verifyResult.coherentText) {
    console.log(clr("1;33", "  ⚠ Agent broke text coherence — reverting"));
    keep = false;
  }

  console.log(
    clr(keep ? "1;32" : "1;31", `  → ${keep ? "✅ KEEPING" : "❌ REVERTING"}`),
  );

  if (!keep) {
    // Revert only the agent's changes — reset to pre-cycle checkpoint
    console.log(clr("2", `  Reverting to pre-cycle checkpoint (${preHash.slice(0, 8)})...`));
    await runCommand("git", ["reset", "--hard", preHash], { cwd: PROJECT_ROOT }).catch(() => { });
  } else {
    // Commit successful change on top of checkpoint
    await runCommand("git", ["add", "-A", "src/", "build.zig", "build.zig.zon", "benchmarks/"], { cwd: PROJECT_ROOT }).catch(() => { });
    await runCommand(
      "git",
      ["commit", "--allow-empty", "-m", `zinc-loop: ${description}`],
      { cwd: PROJECT_ROOT },
    ).catch(() => { });

    // Cherry-pick to main so the other loop can pick up the fix
    if (worktreeName) {
      const commitHash = await runCommand("git", ["rev-parse", "HEAD"], { cwd: PROJECT_ROOT });
      const hash = commitHash.stdout.trim();
      if (hash) {
        const cp = await runCommand(
          "git", ["cherry-pick", "--no-commit", hash],
          { cwd: REPO_ROOT },
        ).catch(() => null);
        if (cp && cp.exitCode === 0) {
          await runCommand(
            "git", ["commit", "-m", `zinc-loop(${worktreeName}): ${description}`],
            { cwd: REPO_ROOT },
          ).catch(() => { });
          console.log(clr("1;36", `  ↗ Cherry-picked to main: ${description.slice(0, 60)}`));
        } else {
          // Conflict — abort and skip merge-back, worktree keeps the change
          await runCommand("git", ["cherry-pick", "--abort"], { cwd: REPO_ROOT }).catch(() => { });
          await runCommand("git", ["reset", "--hard"], { cwd: REPO_ROOT }).catch(() => { });
          console.log(clr("1;33", `  ↗ Cherry-pick to main had conflicts — skipped`));
        }
      }
    }

    // Update best metrics
    if (verifyResult.tokPerSec != null) {
      state.currentBest = { tokPerSec: verifyResult.tokPerSec };
    }
    if (verifyResult.tokensGenerated > (state.currentBest?.tokensGenerated ?? 0)) {
      state.currentBest = {
        ...(state.currentBest ?? {}),
        tokPerSec: verifyResult.tokPerSec,
        tokensGenerated: verifyResult.tokensGenerated,
      };
    }
  }

  const outputSnippet = (verifyResult.runOutput.match(/Output text:\s*(.{0,80})/)?.[1] ?? "").trim();
  const cycleResult: CycleResult = {
    cycle: cycleNum,
    timestamp: new Date().toISOString(),
    phase: buildRun.phase,
    description,
    kept: keep,
    tokPerSec: verifyResult.tokPerSec,
    bandwidthUtil: verifyResult.bandwidthUtil,
    effectiveBW: verifyResult.effectiveBW,
    buildExitCode: verifyResult.buildExitCode,
    runExitCode: verifyResult.runExitCode,
    garbageOutput: verifyResult.garbageOutput,
    coherentText: verifyResult.coherentText,
    outputSnippet,
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
  let worktreeName: string | undefined;

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
      case "--worktree":
        worktreeName = rawArgs[++i];
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
            "  --worktree <name>        Run in a git worktree (enables parallel loops)",
          ].join("\n"),
        );
        process.exit(0);
    }
  }

  // ── Worktree setup ───────────────────────────────────────────────
  if (worktreeName) {
    const branchName = `zinc-loop-${worktreeName}`;
    const worktreePath = join(REPO_ROOT, ".worktrees", `zinc-${worktreeName}`);

    // Create branch from current HEAD if it doesn't exist
    await runCommand("git", ["branch", branchName], { cwd: REPO_ROOT }).catch(() => { });

    // Create worktree if it doesn't exist
    if (!existsSync(worktreePath)) {
      await mkdir(join(REPO_ROOT, ".worktrees"), { recursive: true });
      const wt = await runCommand("git", ["worktree", "add", worktreePath, branchName], { cwd: REPO_ROOT });
      if (wt.exitCode !== 0) {
        console.error(clr("31", `\n  ❌ Failed to create worktree: ${wt.stderr}`));
        process.exit(1);
      }
      console.log(clr("1;32", `  Created worktree: ${worktreePath} (branch: ${branchName})`));
    } else {
      console.log(clr("1;33", `  Using existing worktree: ${worktreePath}`));
    }

    // Point all operations at the worktree
    PROJECT_ROOT = worktreePath;
    REMOTE_ZINC_DIR = `/root/zinc-${worktreeName}`;
  }

  // Results always in main repo so all sessions are visible in one place
  RESULTS_DIR = resolve(REPO_ROOT, ".zinc_optimize");

  const runId = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19)
    + (worktreeName ? `-${worktreeName}` : "");
  const runDir = resumeDir ?? join(RESULTS_DIR, runId);
  await mkdir(runDir, { recursive: true });

  console.log(clr("1;35", "═".repeat(64)));
  console.log(clr("1;35", "  ZINC SELF-IMPROVING LOOP"));
  console.log(clr("1;35", "═".repeat(64)));
  if (worktreeName) {
    console.log(`  Worktree: ${clr("1", `${worktreeName} (${PROJECT_ROOT})`)}`);
    console.log(`  Branch:   ${clr("1", `zinc-loop-${worktreeName}`)}`);
  }
  console.log(`  Remote:   ${clr("1", `${ZINC_USER}@${ZINC_HOST}:${ZINC_PORT}`)}`);
  console.log(`  RemDir:   ${clr("1", REMOTE_ZINC_DIR)}`);
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
  await ssh(`mkdir -p ${REMOTE_ZINC_DIR}`, 10_000).catch(() => { });

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

    const cycleResult = await runCycle(runDir, state, agent, modelPath, worktreeName);
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

    // Merge new ideas (with fuzzy dedup — skip if >60% word overlap with existing)
    for (const idea of cycleResult.nextIdeas) {
      if (isGarbageString(idea)) continue;
      const words = new Set(idea.toLowerCase().split(/\s+/).filter((w) => w.length > 3));
      const isDupe = state.ideas.some((existing) => {
        const existWords = new Set(existing.toLowerCase().split(/\s+/).filter((w) => w.length > 3));
        if (existWords.size === 0 || words.size === 0) return false;
        let overlap = 0;
        for (const w of words) if (existWords.has(w)) overlap++;
        return overlap / Math.min(words.size, existWords.size) > 0.6;
      });
      if (!isDupe) state.ideas.push(idea);
    }
    if (state.ideas.length > 30) state.ideas = state.ideas.slice(-30);

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
