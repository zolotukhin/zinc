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
 *   OPTIMIZE  — once output matches reference: improve tok/s to ≥TARGET_TOK_PER_SEC
 *
 * Usage:
 *   bun loops/implement_metal.ts                     # run indefinitely
 *   bun loops/implement_metal.ts --cycles 100        # 100 cycles max
 *   bun loops/implement_metal.ts --dry-run           # build+run only, no agent
 */

import { spawn } from "node:child_process";
import { existsSync, readdirSync, rmSync } from "node:fs";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import { basename, join, resolve } from "node:path";

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
const MODEL_ID = process.env.ZINC_MODEL_ID ?? "qwen35-35b-a3b-q4k-xl";
const MODEL_PATH = process.env.ZINC_MODEL ?? null;
const TEST_PROMPT = "The capital of France is";
const MAX_TOKENS = 64; // Enough tokens for stable decode throughput measurement
const REFERENCE_TEXT = "Paris"; // Expected in correct output
const TARGET_TOK_PER_SEC = 50;
const BENCHMARK_RUNS = 3; // Median of N inference runs for noise reduction
const PROFILE_EVERY = 5; // Run with --profile every N cycles
const STALL_THRESHOLD = 5; // Cycles without tok/s improvement before studying references

const BLOCKED_GIT_OPS = [
  "Bash(git checkout:*)",
  "Bash(git revert:*)",
  "Bash(git restore:*)",
  "Bash(git reset:*)",
  "Bash(git stash:*)",
  "Bash(git clean:*)",
];

type AgentKind = "claude" | "codex";

function zincModelArgs(): string[] {
  return MODEL_PATH ? ["-m", MODEL_PATH] : ["--model-id", MODEL_ID];
}

function displayModelLabel(): string {
  return MODEL_PATH ? basename(MODEL_PATH) : MODEL_ID;
}

// ── Phase detection ──────────────────────────────────────────────────

export type Phase = "fix" | "implement" | "optimize";

export type OutputEvaluation = {
  normalizedText: string;
  containsReference: boolean;
  strongAnswer: boolean;
  outputQualityScore: number;
  offTopic: boolean;
  evaluationNotes: string[];
};

export type BuildRunResult = {
  buildExitCode: number;
  buildOutput: string;
  testExitCode: number;
  testOutput: string;
  runExitCode: number | null;
  runOutput: string;
  phase: Phase;
  tokPerSec: number | null;
  tokPerSecSamples: number[];
  tokensGenerated: number;
  outputText: string;
  containsReference: boolean;
  strongAnswer: boolean;
  outputQualityScore: number;
  offTopic: boolean;
  evaluationNotes: string[];
  error: string | null;
};

export type ResultSnapshot = {
  cycle: number;
  phase: Phase;
  tokPerSec: number | null;
  tokPerSecSamples: number[];
  tokensGenerated: number;
  outputText: string;
  containsReference: boolean;
  strongAnswer: boolean;
  outputQualityScore: number;
  offTopic: boolean;
  evaluationNotes: string[];
};

export type ControllerState = {
  lastAccepted: ResultSnapshot | null;
  bestSoFar: ResultSnapshot | null;
  bestCorrect: ResultSnapshot | null;
};

type KeepDecision = {
  keep: boolean;
  improvedBestCorrect: boolean;
  reason: string;
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

function normalizeOutputText(text: string): string {
  return text
    .replaceAll("Ġ", " ")
    .replaceAll("Ċ", "\n")
    .replace(/\s+/g, " ")
    .trim();
}

export function evaluateOutputText(text: string): OutputEvaluation {
  const normalizedText = normalizeOutputText(text);
  const lower = normalizedText.toLowerCase();
  const containsReference = lower.includes(REFERENCE_TEXT.toLowerCase());
  const contradictoryCapitalTerms = [
    "capital of germany",
    "capital of italy",
    "capital of spain",
    "capital of portugal",
    "berlin",
    "rome",
    "madrid",
    "lisbon",
  ].some((pattern) => lower.includes(pattern));
  const offTopic = containsReference && contradictoryCapitalTerms;
  const strongAnswer = containsReference && !offTopic &&
    (/^paris\b/i.test(normalizedText) || /^paris[.!?,\s]/i.test(normalizedText));
  const evaluationNotes: string[] = [];
  if (offTopic) evaluationNotes.push("contains contradictory capital/country terms");
  if (containsReference && normalizedText.toLowerCase().startsWith("paris")) {
    evaluationNotes.push("starts with Paris");
  }
  const outputQualityScore = strongAnswer ? 4 : containsReference ? 1 : normalizedText ? 0 : 0;
  return {
    normalizedText,
    containsReference,
    strongAnswer,
    outputQualityScore,
    offTopic,
    evaluationNotes,
  };
}

export function detectPhase(result: BuildRunResult): Phase {
  if (result.buildExitCode !== 0) return "fix";
  if (result.testExitCode !== 0) return "fix";
  if (result.runExitCode !== 0 && result.runExitCode !== null) return "fix";
  if (result.error) return "fix";
  if (result.strongAnswer) return "optimize";
  if (result.tokensGenerated > 0) return "implement";
  return "implement";
}

function canonicalizeMemoryEntry(text: string): string {
  return text
    .toLowerCase()
    .replace(/[`"'()[\],.:;!?-]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function mergeUniqueEntries(existing: string[], incoming: string[], maxEntries: number): string[] {
  const merged: string[] = [];
  const seen = new Set<string>();
  for (const entry of [...existing, ...incoming]) {
    const trimmed = entry.trim();
    if (!trimmed) continue;
    const key = canonicalizeMemoryEntry(trimmed)
      .split(" ")
      .filter(Boolean)
      .sort()
      .join(" ");
    if (!key || seen.has(key)) continue;
    seen.add(key);
    merged.push(trimmed);
    if (merged.length >= maxEntries) break;
  }
  return merged;
}

export function snapshotFromResult(cycle: number, result: BuildRunResult): ResultSnapshot {
  return {
    cycle,
    phase: result.phase,
    tokPerSec: result.tokPerSec,
    tokPerSecSamples: result.tokPerSecSamples,
    tokensGenerated: result.tokensGenerated,
    outputText: result.outputText,
    containsReference: result.containsReference,
    strongAnswer: result.strongAnswer,
    outputQualityScore: result.outputQualityScore,
    offTopic: result.offTopic,
    evaluationNotes: result.evaluationNotes,
  };
}

export function decideKeep(
  verify: BuildRunResult,
  baseline: ResultSnapshot,
  state: ControllerState,
): KeepDecision {
  const bestCorrect = state.bestCorrect;
  const baselineTokens = baseline.tokensGenerated ?? 0;

  if (bestCorrect && !verify.strongAnswer) {
    return {
      keep: false,
      improvedBestCorrect: false,
      reason: "lost short-benchmark correctness relative to accepted baseline",
    };
  }

  if (verify.strongAnswer) {
    if (!bestCorrect) {
      return {
        keep: true,
        improvedBestCorrect: true,
        reason: "first strong correct output",
      };
    }
    const bestTokPerSec = bestCorrect.tokPerSec ?? 0;
    const verifyTokPerSec = verify.tokPerSec ?? 0;
    const improvementThreshold = Math.max(0.5, bestTokPerSec * 0.02);
    if (verifyTokPerSec > bestTokPerSec + improvementThreshold) {
      return {
        keep: true,
        improvedBestCorrect: true,
        reason: "significant correct-throughput improvement",
      };
    }
    return {
      keep: false,
      improvedBestCorrect: false,
      reason: "did not beat best correct throughput",
    };
  }

  if (!bestCorrect && verify.tokensGenerated >= baselineTokens + 2) {
    return {
      keep: true,
      improvedBestCorrect: false,
      reason: "pre-correctness token-progress improvement",
    };
  }

  return {
    keep: false,
    improvedBestCorrect: false,
    reason: "no material progress",
  };
}

export function buildReflectionSummary(state: {
  cycles: Array<{
    cycle: number;
    outputText?: string;
    shortOutputText?: string;
    longOutputText?: string;
    offTopic?: boolean;
    evaluationNotes?: string[];
    decisionReason?: string;
    description?: string;
    kept?: boolean;
  }>;
}): string {
  const recentCycles = state.cycles.slice(-20);
  const total = recentCycles.length;
  const germanyDriftCount = recentCycles.filter((cycle) => {
    const text = normalizeOutputText(
      cycle.longOutputText ?? cycle.outputText ?? cycle.shortOutputText ?? "",
    ).toLowerCase();
    return text.includes("paris") && (text.includes("germany") || text.includes("berlin"));
  }).length;
  const failedCount = recentCycles.filter((cycle) => cycle.kept === false).length;

  const lines = [
    `Last 20 cycles: reviewed ${total} cycle${total === 1 ? "" : "s"}, ${failedCount} rejected.`,
  ];

  if (germanyDriftCount > 0) {
    lines.push(`Repeated failure basin: Paris->Germany list drift (${germanyDriftCount}/${total} recent cycles).`);
  }

  const paritySignals = recentCycles.filter((cycle) =>
    (cycle.evaluationNotes ?? []).some((note) => canonicalizeMemoryEntry(note).includes("contradictory capital country terms"))
  ).length;
  if (paritySignals > 0 || germanyDriftCount > 0) {
    lines.push("Prioritize parity tests around the first wrong layer or expert-down path before more speculative speed work.");
  }

  return lines.join("\n");
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
  opts: {
    cwd?: string;
    timeout?: number;
    streamOutput?: boolean;
    stdoutLineFormatter?: (line: string) => string | null;
    stderrLineFormatter?: (line: string) => string | null;
  } = {},
): Promise<RunResult> {
  return new Promise((res) => {
    const child = spawn(cmd, args, {
      cwd: opts.cwd ?? REPO_ROOT,
      stdio: ["ignore", "pipe", "pipe"],
      timeout: opts.timeout,
    });
    let stdout = "", stderr = "", lineBuffer = "", stderrLineBuffer = "";
    child.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stdout += text;
      if (!opts.streamOutput) return;
      if (opts.stdoutLineFormatter) {
        lineBuffer += text;
        const lines = lineBuffer.split("\n");
        lineBuffer = lines.pop() ?? "";
        for (const line of lines) {
          const formatted = opts.stdoutLineFormatter(line);
          if (formatted != null) process.stdout.write(formatted);
        }
      } else {
        process.stdout.write(text);
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stderr += text;
      if (!opts.streamOutput) return;
      if (opts.stderrLineFormatter) {
        stderrLineBuffer += text;
        const lines = stderrLineBuffer.split("\n");
        stderrLineBuffer = lines.pop() ?? "";
        for (const line of lines) {
          const formatted = opts.stderrLineFormatter(line);
          if (formatted != null) process.stderr.write(formatted);
        }
      } else {
        process.stderr.write(text);
      }
    });
    child.on("error", () => res({ exitCode: -1, stdout, stderr }));
    child.on("close", (code) => {
      if (opts.streamOutput && opts.stdoutLineFormatter && lineBuffer.trim()) {
        const formatted = opts.stdoutLineFormatter(lineBuffer);
        if (formatted != null) process.stdout.write(formatted);
      }
      if (opts.streamOutput && opts.stderrLineFormatter && stderrLineBuffer.trim()) {
        const formatted = opts.stderrLineFormatter(stderrLineBuffer);
        if (formatted != null) process.stderr.write(formatted);
      }
      res({ exitCode: code ?? -1, stdout, stderr });
    });
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
      tokPerSecSamples: [],
      tokensGenerated: 0,
      outputText: "",
      containsReference: false,
      strongAnswer: false,
      outputQualityScore: 0,
      offTopic: false,
      evaluationNotes: [],
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
      tokPerSecSamples: [],
      tokensGenerated: 0,
      outputText: "",
      containsReference: false,
      strongAnswer: false,
      outputQualityScore: 0,
      offTopic: false,
      evaluationNotes: [],
      error: "Tests failed",
    };
  }
  console.log(clr("1;32", "  ✅ Tests OK"));

  if (MODEL_PATH && !existsSync(MODEL_PATH)) {
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
      tokPerSecSamples: [],
      tokensGenerated: 0,
      outputText: "",
      containsReference: false,
      strongAnswer: false,
      outputQualityScore: 0,
      offTopic: false,
      evaluationNotes: [],
      error: null,
    };
  }

  console.log(clr("1;33", `  🚀 Running inference (${maxTokens} tokens, ${BENCHMARK_RUNS} samples)...`));
  const tokPerSecSamples: number[] = [];
  let lastRun: RunResult = { exitCode: -1, stdout: "", stderr: "" };
  let lastCombined = "";

  for (let sample = 0; sample < BENCHMARK_RUNS; sample++) {
    const run = await runCommand(
      "./zig-out/bin/zinc",
      [...zincModelArgs(), "--prompt", TEST_PROMPT, "-n", String(maxTokens)],
      { timeout: 300_000 },
    );
    lastRun = run;
    lastCombined = run.stderr + run.stdout;

    if (run.exitCode !== 0) break; // crash — no point running more samples

    const tps = parseTokPerSec(lastCombined);
    if (tps != null) {
      tokPerSecSamples.push(tps);
      console.log(clr("2", `    sample ${sample + 1}/${BENCHMARK_RUNS}: ${tps.toFixed(2)} tok/s`));
    }
  }

  // Use median of samples for noise-resistant measurement
  const sorted = [...tokPerSecSamples].sort((a, b) => a - b);
  const tokPerSec = sorted.length > 0 ? sorted[Math.floor(sorted.length / 2)] : null;
  const tokensGenerated = parseTokensGenerated(lastCombined);
  const outputText = parseOutputText(lastCombined);
  const evaluation = evaluateOutputText(outputText);

  if (tokPerSec != null && sorted.length > 1) {
    const range = sorted[sorted.length - 1] - sorted[0];
    console.log(clr("1;36", `    median: ${tokPerSec.toFixed(2)} tok/s [${tokPerSecSamples.map(s => s.toFixed(1)).join(", ")}] range=${range.toFixed(1)}`));
  }

  const result: BuildRunResult = {
    buildExitCode: 0,
    buildOutput: build.stderr,
    testExitCode: 0,
    testOutput: "",
    runExitCode: lastRun.exitCode,
    runOutput: lastCombined,
    phase: "implement",
    tokPerSec,
    tokPerSecSamples,
    tokensGenerated,
    outputText,
    containsReference: evaluation.containsReference,
    strongAnswer: evaluation.strongAnswer,
    outputQualityScore: evaluation.outputQualityScore,
    offTopic: evaluation.offTopic,
    evaluationNotes: evaluation.evaluationNotes,
    error: lastRun.exitCode !== 0 ? `Runtime exit code ${lastRun.exitCode}` : null,
  };
  result.phase = detectPhase(result);
  return result;
}

// ── Agent stream formatters ──────────────────────────────────────────

type ClaudeStreamState = {
  currentToolName: string | null;
  currentBlockIsToolUse: boolean;
  inputJsonBuffer: string;
  inTextBlock: boolean;
  sawTextDelta: boolean;
};

type CodexStreamState = {
  startedCommandIds: Set<string>;
};

function coerceDisplayText(value: unknown): string {
  if (typeof value === "string") return value;
  if (value === null || value === undefined) return "";
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  if (Array.isArray(value)) {
    const parts = value.map((entry) => coerceDisplayText(entry)).filter((entry) => entry.trim());
    if (parts.length > 0) return parts.join("\n");
    try { return JSON.stringify(value, null, 2); } catch { return ""; }
  }
  if (typeof value === "object") {
    const record = value as Record<string, unknown>;
    const parts = [
      record.text,
      record.message,
      record.output,
      record.stdout,
      record.stderr,
      record.content,
      record.result,
      record.summary,
      record.output_text,
    ].map((entry) => coerceDisplayText(entry)).filter((entry) => entry.trim());
    if (parts.length > 0) return parts.join("\n");
    try { return JSON.stringify(record, null, 2); } catch { return ""; }
  }
  return "";
}

function formatToolInput(name: string, jsonBuf: string): string {
  let input: Record<string, unknown> = {};
  try { input = JSON.parse(jsonBuf); } catch { return ""; }
  const out: string[] = [];
  const MAX_DIFF = 5;

  if (name === "edit") {
    const fp = (input.file_path as string | undefined) ?? "?";
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")}`));
    const oldLines = ((input.old_string as string | undefined) ?? "").split("\n");
    const newLines = ((input.new_string as string | undefined) ?? "").split("\n");
    for (const l of oldLines.slice(0, MAX_DIFF)) out.push(clr("31", `   - ${l}`));
    if (oldLines.length > MAX_DIFF) out.push(clr("2", `   - … (${oldLines.length - MAX_DIFF} more)`));
    for (const l of newLines.slice(0, MAX_DIFF)) out.push(clr("32", `   + ${l}`));
    if (newLines.length > MAX_DIFF) out.push(clr("2", `   + … (${newLines.length - MAX_DIFF} more)`));
  } else if (name === "write") {
    const fp = (input.file_path as string | undefined) ?? "?";
    const lineCount = ((input.content as string | undefined) ?? "").split("\n").length;
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")} (${lineCount} lines)`));
  } else if (name === "bash") {
    const cmd = (input.command as string | undefined) ?? "?";
    out.push(clr("2", `   $ ${cmd.length > 120 ? cmd.slice(0, 120) + "…" : cmd}`));
  } else if (name === "read") {
    const fp = (input.file_path as string | undefined) ?? "?";
    out.push(clr("2", ` → ${fp.split("/").slice(-3).join("/")}`));
  } else if (name === "grep") {
    const pattern = (input.pattern as string | undefined) ?? "?";
    out.push(clr("2", ` → /${pattern}/`));
  } else if (name === "glob") {
    out.push(clr("2", ` → ${(input.pattern as string | undefined) ?? "?"}`));
  }
  return out.length > 0 ? out.join("\n") + "\n" : "";
}

function formatClaudeStreamLine(rawLine: string, state: ClaudeStreamState): string | null {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try { event = JSON.parse(rawLine) as Record<string, unknown>; } catch { return null; }

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
        state.sawTextDelta = true;
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
    if (result) return clr("32", "   ☑ accepted") + "\n";
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
      if (!text.trim() || state.sawTextDelta) {
        state.sawTextDelta = false;
        return null;
      }
      return clr("96", text) + "\n";
    }
    return null;
  }
  return null;
}

function formatCodexJsonEvent(
  rawLine: string,
  state: CodexStreamState,
): string | null | undefined {
  if (!rawLine.trim()) return null;
  let event: Record<string, unknown>;
  try { event = JSON.parse(rawLine) as Record<string, unknown>; } catch { return undefined; }

  const eventType = typeof event.type === "string" ? event.type : "";
  if (eventType === "thread.started" || eventType === "turn.started" || eventType === "turn.completed")
    return null;
  if (eventType === "error") {
    const message = coerceDisplayText(event.message);
    return message ? `${clr("31", message)}\n` : null;
  }
  if (!eventType.startsWith("item.")) return null;

  const item = event.item as Record<string, unknown> | undefined;
  if (!item) return null;

  const itemType = typeof item.type === "string" ? item.type : "";
  const itemId = typeof item.id === "string" ? item.id : "";
  const phase = eventType.slice("item.".length);

  if (itemType === "reasoning" && phase === "completed") {
    const text = coerceDisplayText(item.summary ?? item.text ?? item.message ?? item.content);
    return text ? `${clr("2", `thinking: ${text}`)}\n` : null;
  }

  if (itemType === "command_execution") {
    const input = item.input as Record<string, unknown> | undefined;
    const cmd = coerceDisplayText(item.command ?? input?.command ?? "").trim();
    const output = coerceDisplayText(item.aggregated_output ?? item.output ?? item.stdout ?? "");
    const exitCode = typeof item.exit_code === "number" ? item.exit_code : null;
    const startedAlready = itemId ? state.startedCommandIds.has(itemId) : false;

    if (phase === "started") {
      if (itemId) state.startedCommandIds.add(itemId);
      return cmd
        ? `\n${clr("33", "🔧 bash")}\n${clr("2", `   $ ${cmd}`)}\n`
        : `\n${clr("33", "🔧 bash")}\n`;
    }

    let out = "";
    if (!startedAlready) {
      out += `\n${clr("33", "🔧 bash")}\n`;
      if (cmd) out += clr("2", `   $ ${cmd}`) + "\n";
    }
    if (phase === "completed") {
      const lines = output.split("\n").filter((line) => line.trim());
      const tail = lines.slice(-3);
      const statusColor = exitCode === 0 ? "32" : exitCode == null ? "33" : "31";
      const statusText = exitCode === 0
        ? "   ☑ accepted"
        : exitCode == null
          ? "   ⚠ completed"
          : `   ✖ exit ${exitCode}`;
      const body = tail.length > 0
        ? (lines.length > 3 ? clr("2", "   …\n") : "") +
          tail.map((line) => clr("2", `   ${line.trim()}`)).join("\n") +
          "\n"
        : "";
      out += `${clr(statusColor, statusText)}\n${body}`;
    }
    if (itemId) state.startedCommandIds.delete(itemId);
    return out || null;
  }

  if (itemType === "file_change" && phase === "completed") {
    const changesSource = [item.changes, item.file_changes, item.files];
    const changes = changesSource.flatMap((value) => {
      if (!Array.isArray(value)) return [];
      return value.map((entry) => {
        const change = entry as Record<string, unknown>;
        return {
          path: coerceDisplayText(change.path ?? change.file_path ?? "?"),
          action: coerceDisplayText(change.change_type ?? change.kind ?? ""),
        };
      });
    });
    if (changes.length === 0) return null;
    let out = `\n${clr("35", "📝 file change")}\n`;
    for (const change of changes.slice(0, 6)) {
      out += clr("2", `   ${change.action ? `${change.action}: ` : ""}${change.path}`) + "\n";
    }
    return out;
  }

  if (itemType === "agent_message" && phase === "completed") {
    const text = coerceDisplayText(item.text ?? item.message ?? item.output_text ?? item.content);
    return text ? `${clr("96", text)}\n` : null;
  }

  if (itemType === "error" && phase === "completed") {
    const text = coerceDisplayText(item.text ?? item.message ?? item.content);
    return text ? `${clr("33", text)}\n` : null;
  }

  return null;
}

function formatCodexStreamLine(rawLine: string, state: CodexStreamState): string | null {
  const jsonFormatted = formatCodexJsonEvent(rawLine, state);
  if (jsonFormatted !== undefined) return jsonFormatted;
  if (!rawLine.trim()) return "\n";
  if (rawLine === "thinking") return `${clr("2", rawLine)}\n`;
  if (rawLine === "codex") return `${clr("1;35", rawLine)}\n`;
  if (rawLine.includes("still running")) return `${clr("2", rawLine)}\n`;
  return `${clr("2", `[codex] ${rawLine}`)}\n`;
}

function formatCodexStderrLine(rawLine: string): string | null {
  if (!rawLine.trim()) return "\n";
  return `${clr("2", `[codex] ${rawLine}`)}\n`;
}

// ── Agent invocation ─────────────────────────────────────────────────

function buildClaudeArgs(prompt: string, model?: string): string[] {
  const args = [
    "-p",
    "--verbose",
    "--output-format", "stream-json",
    "--include-partial-messages",
    `--disallowed-tools=${BLOCKED_GIT_OPS.join(",")}`,
    "--permission-mode", "bypassPermissions",
    "--effort", "high",
  ];
  if (model) args.push("--model", model);
  args.push(prompt);
  return args;
}

function buildCodexArgs(prompt: string, model?: string): string[] {
  const args = [
    "exec",
    "--skip-git-repo-check",
    "--json",
    "--color", "never",
    "--sandbox", "workspace-write",
    "--cd", REPO_ROOT,
  ];
  if (model) args.push("--model", model);
  args.push(prompt);
  return args;
}

async function runAgent(agent: AgentKind, prompt: string, model?: string): Promise<RunResult> {
  const label = agent === "codex" ? "Codex" : "Claude";
  console.log(clr("1;34", SEP));
  console.log(clr("1;34", `  🧠 AGENT PROMPT (${label})`));
  console.log(clr("1;34", SEP));
  const lines = prompt.split("\n");
  for (const line of lines.slice(0, 20)) process.stdout.write(clr("2", line) + "\n");
  if (lines.length > 20) process.stdout.write(clr("2", `… (${lines.length - 20} more lines)\n`));
  console.log(clr("1;34", SEP));

  const startedAt = Date.now();
  const heartbeat = setInterval(() => {
    process.stdout.write(clr("2", `\n⏳ agent running (${formatElapsed(startedAt)})...\n`));
  }, 30_000);

  console.log(clr("1;36", SEP));
  console.log(clr("1;36", `  💬 AGENT RESPONSE (${label})`));
  console.log(clr("1;36", SEP));

  let result: RunResult;
  if (agent === "codex") {
    const streamState: CodexStreamState = {
      startedCommandIds: new Set(),
    };
    result = await runCommand("codex", buildCodexArgs(prompt, model), {
      streamOutput: true,
      timeout: 1_800_000, // 30 min
      stdoutLineFormatter: (line) => formatCodexStreamLine(line, streamState),
      stderrLineFormatter: formatCodexStderrLine,
    });
  } else {
    const streamState: ClaudeStreamState = {
      currentToolName: null,
      currentBlockIsToolUse: false,
      inputJsonBuffer: "",
      inTextBlock: false,
      sawTextDelta: false,
    };
    result = await runCommand("claude", buildClaudeArgs(prompt, model), {
      streamOutput: true,
      timeout: 1_800_000, // 30 min
      stdoutLineFormatter: (line) => formatClaudeStreamLine(line, streamState),
    });
  }

  clearInterval(heartbeat);
  console.log(clr("1;36", SEP));
  console.log(clr("1;32", `  ✅ ${label} done in ${formatElapsed(startedAt)}`));
  return result;
}

// ── Prompt builder ───────────────────────────────────────────────────

export function buildPrompt(state: RunState, lastResult: BuildRunResult): string {
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
    diagnosis.push("Fix the compilation error shown below. Do NOT attempt performance work until it compiles.");
  } else if (lastResult.testExitCode !== 0) {
    diagnosis.push("## Status: TEST FAILURE");
    diagnosis.push("Fix the failing test. All 27+ Metal tests must pass before any perf work.");
  } else if (lastResult.runExitCode !== 0 && lastResult.runExitCode !== null) {
    diagnosis.push(`## Status: RUNTIME CRASH (exit code ${lastResult.runExitCode})`);
    diagnosis.push("Build and tests pass but ZINC crashes during inference. Fix the crash first.");
  } else if (!lastResult.containsReference) {
    diagnosis.push(`## Status: CORRECTNESS REGRESSION — output doesn't contain "Paris"`);
    diagnosis.push(`Output text: "${lastResult.outputText}"`);
    diagnosis.push("The previous optimization broke correctness. You MUST restore correct output first.");
    diagnosis.push("Read the git diff to see what changed and revert the problematic part.");
  } else if (lastResult.tokPerSec != null && lastResult.tokPerSec < TARGET_TOK_PER_SEC) {
    const current = lastResult.tokPerSec;
    const gap = TARGET_TOK_PER_SEC - current;
    const pctNeeded = ((gap / current) * 100).toFixed(0);
    diagnosis.push(`## Status: CORRECT OUTPUT — ${current.toFixed(2)} tok/s → target ≥${TARGET_TOK_PER_SEC}`);
    diagnosis.push(`Gap: ${gap.toFixed(1)} tok/s (need ${pctNeeded}% improvement)`);
    diagnosis.push(`Output: "${trunc(lastResult.outputText, 80)}"`);
    if (lastResult.tokPerSecSamples.length > 1) {
      diagnosis.push(`Benchmark samples: [${lastResult.tokPerSecSamples.map(s => s.toFixed(1)).join(", ")}] tok/s`);
    }
  } else {
    diagnosis.push(`## Status: TARGET REACHED — ${lastResult.tokPerSec?.toFixed(1)} tok/s ≥${TARGET_TOK_PER_SEC}`);
    diagnosis.push("Performance target met!");
  }

  // Stall warning
  if (state.stalledCycles >= STALL_THRESHOLD) {
    diagnosis.push("");
    diagnosis.push(`## ⚠ STALL — ${state.stalledCycles} cycles without meaningful improvement. STUDY THE REFERENCES.`);
    diagnosis.push("");
    diagnosis.push("Guessing is not working. Before making ANY more changes, you MUST study how");
    diagnosis.push("production Metal inference engines solve this exact problem:");
    diagnosis.push("");
    diagnosis.push("### Step 1: Clone and read llama.cpp Metal backend");
    diagnosis.push("```bash");
    diagnosis.push("git clone --depth 1 https://github.com/ggerganov/llama.cpp /tmp/llama.cpp");
    diagnosis.push("```");
    diagnosis.push("Read these files:");
    diagnosis.push("- `/tmp/llama.cpp/ggml/src/ggml-metal/ggml-metal.m` — the core Metal dispatch loop");
    diagnosis.push("- Look at how they batch command buffers, manage encoders, handle MoE expert dispatch");
    diagnosis.push("- Look at their threadgroup sizes for Q4_K DMMV and how they tile matmuls");
    diagnosis.push("- Note how many commitAndWait calls happen per token (likely 1)");
    diagnosis.push("");
    diagnosis.push("### Step 2: Clone and read vLLM Metal / MPS backend");
    diagnosis.push("```bash");
    diagnosis.push("git clone --depth 1 https://github.com/vllm-project/vllm /tmp/vllm");
    diagnosis.push("```");
    diagnosis.push("Read these files:");
    diagnosis.push("- `/tmp/vllm/vllm/attention/backends/` — attention dispatch strategies");
    diagnosis.push("- Look at how they handle MoE routing and expert parallelism");
    diagnosis.push("- Check their memory management and buffer pooling approach");
    diagnosis.push("");
    diagnosis.push("### Step 3: Apply what you learned");
    diagnosis.push("Identify the SPECIFIC technique from llama.cpp or vLLM that addresses our bottleneck,");
    diagnosis.push("then implement it. Cite which file/function you're adapting from in @@@DESCRIPTION.");
    diagnosis.push("Do NOT repeat variations of previously failed approaches.");
  } else if (state.stalledCycles >= 3) {
    diagnosis.push("");
    diagnosis.push(`## Note: ${state.stalledCycles}/${STALL_THRESHOLD} cycles without improvement — will switch to reference study soon`);
  }

  // Reflection summary from recent cycles
  const reflectionSummary = cycles.length >= 5
    ? buildReflectionSummary({ cycles: cycles as any })
    : null;

  const phaseLabel = phase === "fix" ? "FIX" : phase === "implement" ? "IMPLEMENT" : "OPTIMIZE";

  // For optimize phase, use a focused prompt
  const isOptimize = phase === "optimize" || (lastResult.strongAnswer && lastResult.tokPerSec != null);

  const sections: string[] = [
    `# ZINC Metal ${phaseLabel} Task`,
    "",
    ...diagnosis,
    "",
    "## Hardware",
    "- Mac Studio M4 Max, 64 GB unified memory, 40-core GPU, 546 GB/s bandwidth",
    "- Apple GPU family: Apple9 (M4), simdgroup_matrix = true, bfloat = true",
    "- macOS, Metal compute only",
    "",
    "## Model (Qwen3.5-35B-A3B, Q4_K, 20.7 GB)",
    "- 40 layers: every 4th is full attention (layers 3,7,11,...,39), rest are SSM/delta-net",
    "- MoE FFN: 256 experts, 8 active per token, + shared expert",
    "- head_dim=256, hidden_dim=2048, n_heads=16, n_kv_heads=2",
    "- Active parameters per token: ~3B (due to MoE sparsity)",
    "- Effective working set per decode step: ~1.7 GB at Q4_K",
    "",
  ];

  if (isOptimize) {
    // Optimization-specific sections
    sections.push(
      "## Bandwidth Analysis",
      "- Memory BW: 546 GB/s theoretical, ~480 GB/s achievable",
      "- Working set per token: ~1.7 GB (only active experts + attention layers)",
      "- Theoretical BW-limited decode: ~280 tok/s (480 / 1.7)",
      `- Current: ${lastResult.tokPerSec?.toFixed(1)} tok/s → ${((lastResult.tokPerSec ?? 0) / 280 * 100).toFixed(0)}% of theoretical BW limit`,
      "- This means MOST time is lost to dispatch overhead, sync, or compute bottlenecks — NOT bandwidth",
      "",
      "## Baseline Reference",
      "- llama.cpp Metal on this machine: 72.93 tok/s decode (tg128)",
      `- ZINC target: ≥${TARGET_TOK_PER_SEC} tok/s`,
      `- ZINC current: ${lastResult.tokPerSec?.toFixed(1)} tok/s`,
      "",
      "## Optimization Targets (pick ONE per cycle)",
      "",
      "### 1. Reduce command buffer submissions",
      "Each commitAndWait() is a CPU-GPU sync point (~50-100μs overhead).",
      "Ideal: ONE command buffer submit per decode step. Batch all 40 layers into",
      "a single command buffer with barriers between dependent dispatches.",
      "Check how many commits happen per token in forward_metal.zig's decode loop.",
      "",
      "### 2. Minimize Metal encoder recreation",
      "mtl_barrier() creates a new compute command encoder. Each encoder switch costs ~10-30μs.",
      "Only barrier when there is a true data dependency. Adjacent dispatches to different",
      "buffers do NOT need a barrier.",
      "",
      "### 3. MoE expert dispatch batching",
      "With 8 active experts per token, if each expert is a separate dispatch, that's 8 small",
      "dispatches per layer × 30 MoE layers = 240 small dispatches. Each has launch overhead.",
      "Consider: batch multiple experts into one dispatch with offset indexing, or fuse gate+up.",
      "",
      "### 4. Threadgroup size tuning",
      "M4 Max: max 1024 threads per threadgroup, 32 SIMD width.",
      "DMMV shaders for Q4_K: check if threadgroup size matches the row count.",
      "Undersized threadgroups → low occupancy. Oversized → register pressure.",
      "",
      "### 5. Use half/bfloat for intermediates",
      "M4 has 2x throughput for bfloat16 vs float32 in compute.",
      "If intermediate buffers (hidden state, norm output) can use half precision,",
      "this halves bandwidth and doubles ALU throughput for those stages.",
      "",
      "### 6. Fused kernels",
      "RMSNorm + first DMMV could be fused to avoid writing norm_buf to memory.",
      "SwiGLU (gate * silu(up)) is another fusion candidate.",
      "Each fused kernel saves one global memory round-trip.",
      "",
      "### 7. Pipeline state object caching",
      "If getPipeline() does dictionary lookup per dispatch, cache the PSO pointers",
      "for hot paths (called 40× per token).",
      "",
    );
  } else {
    // Fix/implement sections (legacy path, kept for correctness regressions)
    sections.push(
      "## Project Structure",
      "```",
      "src/compute/forward_metal.zig — Metal inference engine (THE MAIN FILE)",
      "src/metal/   — shim.h, shim.m (ObjC C API), device.zig, buffer.zig, command.zig",
      "src/shaders/metal/ — MSL compute shaders (dmmv_q4k, flash_attn, rms_norm_mul, etc.)",
      "```",
      "",
      "## Key Reference: forward.zig (Vulkan version)",
      "The Vulkan `decodeStep()` at src/compute/forward.zig shows the exact layer dispatch",
      "sequence. Read it for the correct order of operations, tensor names, and dimensions.",
      "",
    );
  }

  sections.push(
    "## Key Files to Edit",
    "- src/compute/forward_metal.zig — decode loop, dispatch sequence, buffer management",
    "- src/metal/command.zig — command buffer management, barrier implementation",
    "- src/metal/shim.m — ObjC shim: mtl_dispatch, mtl_barrier, mtl_commit",
    "- src/shaders/metal/*.metal — shader source (threadgroup sizes, occupancy)",
    "",
  );

  // Profile output if available
  if (state.lastProfileOutput) {
    sections.push(
      `## Profile Output (cycle ${state.lastProfileCycle})`,
      "Use this to identify the actual hotspots. Focus optimization on the slowest phases.",
      "```",
      state.lastProfileOutput.slice(-3000),
      "```",
      "",
    );
  }

  // Build/test/run output
  if (lastResult.buildOutput) {
    sections.push("## Build Output (last 2000 chars)", "```", buildOut, "```", "");
  }
  if (lastResult.testOutput) {
    sections.push("## Test Output (last 2000 chars)", "```", testOut, "```", "");
  }
  if (lastResult.runOutput) {
    sections.push("## Run Output (last 3000 chars)", "```", runOut, "```", "");
  }

  // Reflection
  if (reflectionSummary) {
    sections.push("## Reflection (auto-analysis of recent cycles)", reflectionSummary, "");
  }

  // Self-review summaries from periodic reviews
  if (state.reviewSummaries && state.reviewSummaries.length > 0) {
    // Include the latest review
    sections.push(state.reviewSummaries[state.reviewSummaries.length - 1], "");
  }

  sections.push(
    "## Cycle History",
    historyBlock,
    "",
    "## Failed Approaches (DO NOT repeat these)",
    failedBlock,
    "",
    "## Ideas",
    ideasBlock,
    "",
    "## Rules",
    "1. Make ONE focused change per cycle. Measure, don't guess.",
    "2. CORRECTNESS IS SACRED. Output MUST contain 'Paris'. Speed without correctness = instant revert.",
    "3. All 27+ tests must continue passing.",
    "4. Do NOT modify src/vulkan/, loops/, or .env.",
    "5. Zig 0.15.2 API: ArrayList is unmanaged (pass allocator to append/deinit).",
    "6. MSL shaders use 'main0' as entry point (SPIRV-Cross convention).",
    "7. Metal push constants go in buffer[n_bufs] (see shim.m mtl_dispatch).",
    "8. The Metal command pattern: beginCommand → dispatch → barrier → dispatch → commitAndWait.",
    "9. UMA advantage: all buffers are SharedMode — cpu_ptr gives direct CPU access to GPU data.",
    "10. Read the profile output and run output BEFORE deciding what to optimize.",
    "11. Prefer changes to forward_metal.zig and shaders. Avoid refactoring infrastructure.",
    "",
    "## Output Format",
    "After making your change, print these 3 lines:",
    "@@@DESCRIPTION: <one-line summary>",
    "@@@SELF_ANALYSIS: <why this approach and what you expect, with estimated tok/s impact>",
    "@@@NEXT_IDEAS: <comma-separated ideas for future cycles>",
  );

  return sections.join("\n");
}

// ── State ────────────────────────────────────────────────────────────

export type CycleResult = {
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

export type RunState = {
  runId: string;
  cycles: CycleResult[];
  failedApproaches: string[];
  ideas: string[];
  phase: Phase;
  currentBest: { tokPerSec: number | null; containsReference: boolean } | null;
  stalledCycles: number;
  bestTokPerSec: number;
  lastProfileOutput: string | null;
  lastProfileCycle: number | null;
  reviewSummaries: string[];
};

async function loadState(runDir: string): Promise<RunState | null> {
  const p = join(runDir, "state.json");
  if (!existsSync(p)) return null;
  return JSON.parse(await readFile(p, "utf8")) as RunState;
}

async function saveState(runDir: string, state: RunState): Promise<void> {
  await writeFile(join(runDir, "state.json"), JSON.stringify(state, null, 2));
}

async function runProfileBenchmark(): Promise<string> {
  console.log(clr("1;33", "  📊 Profiling run (--profile)..."));
  const run = await runCommand(
    "./zig-out/bin/zinc",
    [...zincModelArgs(), "--prompt", TEST_PROMPT, "-n", "32", "--profile"],
    { timeout: 300_000 },
  );
  const combined = (run.stderr + run.stdout).slice(-4000);
  console.log(clr("2", "    profile captured"));
  return combined;
}

const REVIEW_EVERY = 10;

export function buildSelfReview(state: RunState): string {
  const recent = state.cycles.slice(-REVIEW_EVERY);
  if (recent.length === 0) return "";

  const kept = recent.filter(c => c.kept);
  const reverted = recent.filter(c => !c.kept);
  const tpsValues = kept.filter(c => c.tokPerSec != null).map(c => c.tokPerSec!);
  const tpsStart = recent[0].tokPerSec ?? 0;
  const tpsEnd = recent[recent.length - 1].tokPerSec ?? tpsStart;
  const delta = tpsEnd - tpsStart;

  // Categorize approaches by keywords
  const categories: Record<string, { kept: number; reverted: number }> = {};
  for (const c of recent) {
    const desc = c.description.toLowerCase();
    const tags: string[] = [];
    if (desc.match(/shader|kernel|threadgroup|simd|metal/)) tags.push("shader");
    if (desc.match(/dispatch|command|encoder|barrier|commit|batch/)) tags.push("dispatch");
    if (desc.match(/buffer|alloc|pool|reuse|memory/)) tags.push("memory");
    if (desc.match(/fuse|fusion|merged|combined/)) tags.push("fusion");
    if (desc.match(/moe|expert|router|topk/)) tags.push("moe");
    if (desc.match(/attention|flash|kv.?cache|rope/)) tags.push("attention");
    if (desc.match(/half|float16|bfloat|bf16|f16/)) tags.push("precision");
    if (tags.length === 0) tags.push("other");
    for (const tag of tags) {
      if (!categories[tag]) categories[tag] = { kept: 0, reverted: 0 };
      if (c.kept) categories[tag].kept++;
      else categories[tag].reverted++;
    }
  }

  const lines: string[] = [
    `## Self-Review (last ${recent.length} cycles)`,
    "",
    `- Kept: ${kept.length}/${recent.length} changes`,
    `- tok/s movement: ${tpsStart.toFixed(1)} → ${tpsEnd.toFixed(1)} (${delta >= 0 ? "+" : ""}${delta.toFixed(1)})`,
    `- Best tok/s in window: ${tpsValues.length > 0 ? Math.max(...tpsValues).toFixed(1) : "N/A"}`,
    "",
    "### What's working vs not:",
  ];

  for (const [cat, stats] of Object.entries(categories).sort((a, b) => b[1].kept - a[1].kept)) {
    const total = stats.kept + stats.reverted;
    const rate = ((stats.kept / total) * 100).toFixed(0);
    const indicator = stats.kept > stats.reverted ? "✅" : stats.kept === 0 ? "❌" : "⚠";
    lines.push(`  ${indicator} ${cat}: ${stats.kept}/${total} kept (${rate}% success)`);
  }

  lines.push("");
  if (delta < 1) {
    lines.push("### ⚠ Low progress — strategic pivot recommended:");
    lines.push("- STOP trying small variations of what already failed");
    lines.push("- Focus on categories with >50% success rate above");
    lines.push("- If no category is working, the bottleneck is elsewhere — profile first");
  } else {
    lines.push(`### Progress is positive (+${delta.toFixed(1)} tok/s). Double down on what's working.`);
  }

  // Most impactful kept changes
  const impactful = kept
    .filter(c => c.tokPerSec != null)
    .sort((a, b) => (b.tokPerSec ?? 0) - (a.tokPerSec ?? 0))
    .slice(0, 3);
  if (impactful.length > 0) {
    lines.push("");
    lines.push("### Top performing changes:");
    for (const c of impactful) {
      lines.push(`  - ${c.description} → ${c.tokPerSec?.toFixed(1)} tok/s`);
    }
  }

  return lines.join("\n");
}

function extractAgentText(stdout: string): string {
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
      if (evt?.type === "item.completed" && evt?.item?.type === "agent_message") {
        const text = coerceDisplayText(
          evt.item.text ?? evt.item.message ?? evt.item.output_text ?? evt.item.content,
        );
        if (text.trim()) texts.push(text);
      }
    } catch { /* not JSON */ }
  }
  return texts.join("\n");
}

// ── Main loop ────────────────────────────────────────────────────────

function findLatestRunDir(): string | null {
  if (!existsSync(RESULTS_DIR)) return null;
  const entries = readdirSync(RESULTS_DIR, { withFileTypes: true })
    .filter(d => d.isDirectory())
    .map(d => d.name)
    .sort()
    .reverse();
  for (const entry of entries) {
    const stateFile = join(RESULTS_DIR, entry, "state.json");
    if (existsSync(stateFile)) return join(RESULTS_DIR, entry);
  }
  return null;
}

function cleanupOldRuns(): void {
  if (!existsSync(RESULTS_DIR)) return;
  rmSync(RESULTS_DIR, { recursive: true, force: true });
  console.log(clr("2", `  Cleaned up old runs: ${RESULTS_DIR}`));
}

async function main() {
  const args = process.argv.slice(2);
  let maxCycles = 999;
  let dryRun = false;
  let agent: AgentKind = "claude";
  let model: string | undefined;
  let resume = false;

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case "--agent": {
        const value = args[++i];
        if (value !== "claude" && value !== "codex") {
          console.error(`Invalid --agent: ${value}. Use claude or codex.`);
          process.exit(1);
        }
        agent = value;
        break;
      }
      case "--model":
        model = args[++i];
        break;
      case "--cycles":
        maxCycles = parseInt(args[++i] ?? "999", 10);
        break;
      case "--dry-run":
        dryRun = true;
        break;
      case "--resume":
        resume = true;
        break;
      case "--help":
        console.log([
          "Usage: bun loops/implement_metal.ts [options]",
          "",
          "Options:",
          "  --agent <claude|codex>  Agent to use (default: claude)",
          "  --model <name>          Model override for selected agent",
          "  --cycles N              Max cycles (default: 999)",
          "  --dry-run               Build+run only, no agent",
          "  --resume                Resume the most recent run",
        ].join("\n"));
        process.exit(0);
    }
  }

  const agentLabel = agent === "codex" ? "Codex" : "Claude";

  // Resume or fresh start
  let runId: string;
  let runDir: string;
  let state: RunState;
  let startCycle: number;

  if (resume) {
    const latestDir = findLatestRunDir();
    if (!latestDir) {
      console.error("No previous run found to resume.");
      process.exit(1);
    }
    const loaded = await loadState(latestDir);
    if (!loaded) {
      console.error(`No state.json in ${latestDir}`);
      process.exit(1);
    }
    state = loaded;
    // Backfill fields that may not exist in older state files
    state.reviewSummaries ??= [];
    state.stalledCycles ??= 0;
    state.bestTokPerSec ??= 0;
    state.lastProfileOutput ??= null;
    state.lastProfileCycle ??= null;
    runId = state.runId;
    runDir = latestDir;
    startCycle = state.cycles.length + 1;
    console.log(clr("1;36", "╔══════════════════════════════════════════════════════════════╗"));
    console.log(clr("1;36", "║  ZINC Metal Optimization Loop — RESUMING                     ║"));
    console.log(clr("1;36", `║  Target: ≥${TARGET_TOK_PER_SEC} tok/s  |  Model: ${displayModelLabel().slice(0, 35)}  ║`));
    console.log(clr("1;36", `║  Run: ${runId}  |  Resuming from cycle ${startCycle}            ║`));
    console.log(clr("1;36", "╚══════════════════════════════════════════════════════════════╝"));
    console.log(`  Agent: ${clr("1", agentLabel)}${model ? ` (${model})` : ""}`);
    console.log(`  Previous cycles: ${state.cycles.length}, best: ${state.bestTokPerSec.toFixed(2)} tok/s`);
    console.log(`  Results: ${clr("2", runDir)}`);
  } else {
    // Fresh start — clean up old data
    cleanupOldRuns();
    runId = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
    runDir = join(RESULTS_DIR, runId);
    await mkdir(runDir, { recursive: true });
    startCycle = 1;
    state = {
      runId,
      cycles: [],
      failedApproaches: [],
      ideas: [],
      phase: "optimize",
      currentBest: null,
      stalledCycles: 0,
      bestTokPerSec: 0,
      lastProfileOutput: null,
      lastProfileCycle: null,
      reviewSummaries: [],
    };
    console.log(clr("1;36", "╔══════════════════════════════════════════════════════════════╗"));
    console.log(clr("1;36", "║  ZINC Metal Optimization Loop                                ║"));
    console.log(clr("1;36", `║  Target: ≥${TARGET_TOK_PER_SEC} tok/s  |  Model: ${displayModelLabel().slice(0, 35)}  ║`));
    console.log(clr("1;36", `║  Run: ${runId}  |  Max cycles: ${maxCycles}               ║`));
    console.log(clr("1;36", "╚══════════════════════════════════════════════════════════════╝"));
    console.log(`  Agent: ${clr("1", agentLabel)}${model ? ` (${model})` : ""}`);
    console.log(`  Results: ${clr("2", runDir)}`);
  }

  for (let cycle = startCycle; cycle <= maxCycles; cycle++) {
    console.log(clr("1;35", "\n" + "═".repeat(64)));
    console.log(clr("1;35", `  CYCLE ${cycle}`));
    console.log(clr("1;35", "═".repeat(64)));

    const cycleDir = join(runDir, `cycle-${String(cycle).padStart(3, "0")}`);
    await mkdir(cycleDir, { recursive: true });

    // Always use full token count for stable benchmarking
    const currentMaxTokens = MAX_TOKENS;

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

    const agentResult = await runAgent(agent, prompt, model);
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

    // Keep/revert decision — tight for optimization
    let kept = false;
    const prevTps = state.bestTokPerSec;
    const verifyTps = verify.tokPerSec ?? 0;

    if (verify.buildExitCode !== 0 || verify.testExitCode !== 0) {
      // Build or test broken → revert
      console.log(clr("1;31", `  ↩ REVERTING — ${verify.buildExitCode !== 0 ? "build" : "tests"} broken`));
      await runCommand("git", ["reset", "--hard", preHash]);
      state.failedApproaches.push(`${description} — broke ${verify.buildExitCode !== 0 ? "build" : "tests"}`);
      state.stalledCycles++;
    } else if (verify.runExitCode !== 0 && verify.runExitCode !== null) {
      // Crash → revert
      console.log(clr("1;31", `  ↩ REVERTING — runtime crash`));
      await runCommand("git", ["reset", "--hard", preHash]);
      state.failedApproaches.push(`${description} — runtime crash`);
      state.stalledCycles++;
    } else if (!verify.containsReference && state.currentBest?.containsReference) {
      // Lost correctness → always revert
      console.log(clr("1;31", `  ↩ REVERTING — lost correctness (output: "${verify.outputText.slice(0, 60)}")`));
      await runCommand("git", ["reset", "--hard", preHash]);
      state.failedApproaches.push(`${description} — broke correctness`);
      state.stalledCycles++;
    } else if (verify.containsReference && verifyTps > prevTps + 0.5) {
      // Meaningful speed improvement with correct output
      kept = true;
      state.bestTokPerSec = verifyTps;
      state.stalledCycles = 0;
      console.log(clr("1;32", `  ✅ KEPT — ${verifyTps.toFixed(2)} tok/s (was ${prevTps.toFixed(2)}, +${(verifyTps - prevTps).toFixed(2)})`));
    } else if (verify.containsReference && verifyTps >= prevTps - 0.3) {
      // Within noise band, correct output — keep (might enable future gains)
      kept = true;
      if (verifyTps > prevTps) state.bestTokPerSec = verifyTps;
      state.stalledCycles++;
      console.log(clr("1;33", `  ≈ KEPT — ${verifyTps.toFixed(2)} tok/s (within noise of ${prevTps.toFixed(2)})`));
    } else if (verify.containsReference && !state.currentBest?.containsReference) {
      // Gained correctness for the first time
      kept = true;
      state.bestTokPerSec = verifyTps;
      state.stalledCycles = 0;
      console.log(clr("1;32", `  ✅ KEPT — gained correct output! ${verifyTps.toFixed(2)} tok/s`));
    } else {
      // Regressed speed or no correctness
      console.log(clr("1;31", `  ↩ REVERTING — ${verifyTps.toFixed(2)} tok/s < ${prevTps.toFixed(2)} (regressed ${(prevTps - verifyTps).toFixed(2)} tok/s)`));
      await runCommand("git", ["reset", "--hard", preHash]);
      state.failedApproaches.push(`${description} — regressed from ${prevTps.toFixed(1)} to ${verifyTps.toFixed(1)} tok/s`);
      state.stalledCycles++;
    }

    if (kept) {
      state.currentBest = {
        tokPerSec: verify.tokPerSec,
        containsReference: verify.containsReference,
      };
      await runCommand("git", ["add", "-A", "src/", "build.zig"]).catch(() => {});
      await runCommand("git", ["commit", "-m", `metal-loop: cycle-${cycle} ${description} (${verifyTps.toFixed(1)} tok/s)`]).catch(() => {});
    }

    // Periodic profiling run (after verify, so we profile the current accepted state)
    // Also profile on cycle 1 so the agent has data from the start
    if ((cycle === 1 || cycle % PROFILE_EVERY === 0) && kept && verify.containsReference) {
      try {
        state.lastProfileOutput = await runProfileBenchmark();
        state.lastProfileCycle = cycle;
        await writeFile(join(cycleDir, "profile.log"), state.lastProfileOutput);
      } catch {
        console.log(clr("1;33", "  ⚠ Profile run failed, continuing"));
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

    // Self-review every REVIEW_EVERY cycles
    if (state.cycles.length > 0 && state.cycles.length % REVIEW_EVERY === 0) {
      console.log(clr("1;35", `\n  🔍 Self-review (${state.cycles.length} cycles completed)...`));
      const review = buildSelfReview(state);
      state.reviewSummaries.push(review);
      console.log(clr("2", review));
    }

    await saveState(runDir, state);

    // Status summary
    console.log(clr("2", `  stall=${state.stalledCycles} best=${state.bestTokPerSec.toFixed(2)} target=${TARGET_TOK_PER_SEC}`));

    // Check if we're done
    if (verify.containsReference && verify.tokPerSec != null && verify.tokPerSec >= TARGET_TOK_PER_SEC) {
      console.log(clr("1;32", "\n" + "=".repeat(64)));
      console.log(clr("1;32", `  TARGET REACHED: ${verify.tokPerSec.toFixed(1)} tok/s >= ${TARGET_TOK_PER_SEC} with correct output!`));
      console.log(clr("1;32", "=".repeat(64)));
      break;
    }
  }

  console.log(clr("1;36", `\nLoop complete. Results: ${runDir}`));
  console.log(clr("1;36", `Total cycles: ${state.cycles.length}`));
  console.log(clr("1;36", `Kept: ${state.cycles.filter(c => c.kept).length}`));
  console.log(clr("1;36", `Best: ${state.bestTokPerSec.toFixed(2)} tok/s (target: ${TARGET_TOK_PER_SEC}), correct=${state.currentBest?.containsReference ?? false}`));
}

if (import.meta.main) {
  main().catch((err) => {
    console.error("Fatal error:", err);
    process.exit(1);
  });
}
