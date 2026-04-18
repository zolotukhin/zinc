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
 *   bun loops/optimize_perf.ts --effort 6 --model qwen35b       # RDNA prefill recovery on Qwen 35B
 *   bun loops/optimize_perf.ts --effort 1 --resume               # Resume previous run
 *   bun loops/optimize_perf.ts --effort 1 --cycles 10 --dry-run  # Baseline only
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { readFile, writeFile, mkdir, rm } from "node:fs/promises";
import { join, resolve } from "node:path";
import {
  parseTokPerSec,
  parsePrefillTokPerSec,
  parseBandwidthUtil,
} from "./optimize_zinc";
import { formatElapsed } from "./optimize_llm_tps";

// -- Config ------------------------------------------------------------------

const REPO_ROOT = resolve(import.meta.dir, "..");
const RESULTS_DIR = resolve(REPO_ROOT, ".perf_optimize");
const CLAUDE_EFFORT = "max";
const CODEX_REASONING_EFFORT = "xhigh";

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

type PromptMode = "raw" | "chat";

type ModelTarget = {
  key: string;
  name: string;
  path: string;
  promptMode: PromptMode;
  coherencePromptMode?: PromptMode;
  envVar: string;
  coherenceMaxTokens?: number;
};

function envOrDefault(name: string, fallback: string): string {
  return process.env[name] ?? ENV[name] ?? fallback;
}

const MODELS: Record<string, ModelTarget> = {
  qwen35b: {
    key: "qwen35b",
    name: "Qwen3.5-35B",
    path: envOrDefault("ZINC_RDNA_QWEN35_35B_MODEL", "/root/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"),
    promptMode: "raw",
    // Keep throughput benchmarking on the raw decode path, but run coherence
    // prompts through ChatML so Qwen gets the expected closed-think scaffold.
    coherencePromptMode: "chat",
    envVar: "ZINC_RDNA_QWEN35_35B_MODEL",
  },
  qwen36b: {
    key: "qwen36b",
    name: "Qwen3.6-35B",
    path: envOrDefault("ZINC_RDNA_QWEN36_35B_MODEL", "/root/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
    promptMode: "raw",
    // Qwen3.6 inherits the same closed-think chat prompt behavior as Qwen3.5.
    coherencePromptMode: "chat",
    envVar: "ZINC_RDNA_QWEN36_35B_MODEL",
  },
  qwen8b: {
    key: "qwen8b",
    name: "Qwen3-8B",
    path: envOrDefault("ZINC_RDNA_QWEN3_8B_MODEL", "/root/models/Qwen3-8B-Q4_K_M.gguf"),
    promptMode: "raw",
    envVar: "ZINC_RDNA_QWEN3_8B_MODEL",
  },
  gemma431b: {
    key: "gemma431b",
    name: "Gemma4-31B",
    path: envOrDefault("ZINC_RDNA_GEMMA4_31B_MODEL", "/root/models/gemma-4-31B-it-Q4_K_M.gguf"),
    promptMode: "chat",
    envVar: "ZINC_RDNA_GEMMA4_31B_MODEL",
  },
  gemma412b: {
    key: "gemma412b",
    name: "Gemma4-12B",
    path: envOrDefault("ZINC_RDNA_GEMMA4_12B_MODEL", "/root/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
    promptMode: "chat",
    envVar: "ZINC_RDNA_GEMMA4_12B_MODEL",
  },
  gptoss20b: {
    key: "gptoss20b",
    name: "GPT-OSS-20B",
    path: envOrDefault("ZINC_RDNA_GPT_OSS_20B_MODEL", "/root/models/openai_gpt-oss-20b-Q4_K_M.gguf"),
    promptMode: "chat",
    envVar: "ZINC_RDNA_GPT_OSS_20B_MODEL",
    // GPT-OSS emits analysis/final channel scaffolding before the concise answer.
    coherenceMaxTokens: 96,
  },
};

const MODEL_KEYS = Object.keys(MODELS).join(", ");

const REMOTE_ZINC_ENV = "RADV_PERFTEST=coop_matrix";
const LONG_CONTEXT_BENCH_SENTENCE =
  "Benchmark context only. alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.";

function repeatContext(lines: number): string {
  return Array.from({ length: lines }, () => LONG_CONTEXT_BENCH_SENTENCE).join(" ");
}

function shellQuote(value: string): string {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

const PREFILL_BENCHMARK_PROMPT = [
  "Long reference packet for benchmark purposes only:",
  "",
  repeatContext(6),
  "",
  "Important fact near the end: Paris is the capital of France.",
  "",
  "Ignore unrelated filler and answer from the reference packet.",
  "",
  "Based on the reference above, the capital of France is",
].join("\n");

type MetricMode = "decode" | "prefill";

type EffortSpec = {
  doc: string;
  summary: string;
  metricMode: MetricMode;
  primaryMetricLabel: string;
  benchmarkPrompt: string;
  benchmarkMaxTokens: number;
  benchmarkMethod: string;
};

const EFFORT_SPECS: Record<number, EffortSpec> = {
  1: {
    doc: "MULTI_HOUR_EFFORT_1_PUSH_DESCRIPTORS.md",
    summary: "Push descriptors (~2.5% decode speedup)",
    metricMode: "decode",
    primaryMetricLabel: "decode tok/s",
    benchmarkPrompt: "Write a detailed essay about the history of computing, from mechanical calculators to modern artificial intelligence.",
    benchmarkMaxTokens: 200,
    benchmarkMethod: "200-token decode benchmark on the primary model",
  },
  2: {
    doc: "MULTI_HOUR_EFFORT_2_FUSED_GATE_UP.md",
    summary: "Fused gate+up DMMV (~1-2% decode speedup)",
    metricMode: "decode",
    primaryMetricLabel: "decode tok/s",
    benchmarkPrompt: "Write a detailed essay about the history of computing, from mechanical calculators to modern artificial intelligence.",
    benchmarkMaxTokens: 200,
    benchmarkMethod: "200-token decode benchmark on the primary model",
  },
  3: {
    doc: "MULTI_HOUR_EFFORT_3_BATCH_PREFILL.md",
    summary: "Batch prefill (~4-8x prefill speedup)",
    metricMode: "prefill",
    primaryMetricLabel: "prefill tok/s",
    benchmarkPrompt: PREFILL_BENCHMARK_PROMPT,
    benchmarkMaxTokens: 8,
    benchmarkMethod: "long-context prefill benchmark aligned with the site report",
  },
  4: {
    doc: "MULTI_HOUR_EFFORT_4_PREFILL_RECOVERY.md",
    summary: "Prefill recovery (close the long-context gap vs llama.cpp)",
    metricMode: "prefill",
    primaryMetricLabel: "prefill tok/s",
    benchmarkPrompt: PREFILL_BENCHMARK_PROMPT,
    benchmarkMaxTokens: 8,
    benchmarkMethod: "long-context prefill benchmark aligned with the site report",
  },
  6: {
    doc: "MULTI_HOUR_EFFORT_6_RDNA_QWEN35_PREFILL.md",
    summary: "RDNA Qwen35 prefill recovery (restore flagship TTFT and prefill telemetry)",
    metricMode: "prefill",
    primaryMetricLabel: "prefill tok/s",
    benchmarkPrompt: PREFILL_BENCHMARK_PROMPT,
    benchmarkMaxTokens: 8,
    benchmarkMethod: "long-context prefill benchmark on RDNA for the Qwen3.5-35B flagship workload",
  },
};

export function getEffortSpec(effort: number): EffortSpec | null {
  return EFFORT_SPECS[effort] ?? null;
}

const BENCHMARK_SAMPLES = 3;
const MIN_IMPROVEMENT_ABS_TPS = 0.5;
const MIN_IMPROVEMENT_PCT = 0.01;
const HISTORY_LINES_IN_PROMPT = 20;
const RECENT_CYCLES_IN_PROMPT = 12;
const FAILED_APPROACH_LIMIT = 30;
const IDEA_LIMIT = 24;
const REVIEW_SUMMARY_LIMIT = 6;
const SELF_REVIEW_EVERY = 10;
const STALL_WARNING_THRESHOLD = 4;
const FOUNDATION_KEEP_MAX_DROP_TPS = 0.25;
const MAX_FOUNDATION_KEEPS_IN_A_ROW = 2;
const MAX_CHANGED_FILES_IN_PROMPT = 10;

// Multiple prompts to catch different failure modes:
// - Short factual: catches total corruption
// - Arithmetic: catches subtle numeric drift (wrong MoE routing, bad dequant)
// - Listing: catches mid-sequence divergence (broken RoPE, bad KV cache)
type CoherenceCheck = {
  rawPrompt: string;
  chatPrompt: string;
  expect: string[];
};

const COHERENCE_CHECKS: CoherenceCheck[] = [
  {
    rawPrompt: "The capital of France is",
    chatPrompt: "What is the capital of France? Answer in one word.",
    expect: ["Paris"],
  },
  {
    rawPrompt: "2+2 =",
    chatPrompt: "What is 2+2? Answer using one number.",
    expect: ["4"],
  },
  {
    rawPrompt: "Name the first four planets in order:",
    chatPrompt: "Name the first four planets in order. Answer with only the names separated by commas.",
    expect: ["Mercury", "Venus", "Earth", "Mars"],
  },
];

// All models that must produce coherent output after every change.
// The primary model (--model flag) is benchmarked; these are correctness-only.
const COHERENCE_MODELS: ModelTarget[] = [
  MODELS.qwen35b,
  MODELS.qwen36b,
  MODELS.qwen8b,
  MODELS.gemma431b,
  MODELS.gemma412b,
  MODELS.gptoss20b,
];

type CoherenceFailure = {
  id: string;
  label: string;
  model: string;
  prompt: string;
  outputText: string;
  kind: "mismatch" | "crash";
};

type CoherenceSweep = {
  failures: CoherenceFailure[];
  failureIds: string[];
};

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
  let analyze = false;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--effort" && args[i + 1]) effort = parseInt(args[++i], 10);
    else if (args[i] === "--cycles" && args[i + 1]) cycles = parseInt(args[++i], 10);
    else if (args[i] === "--dry-run") dryRun = true;
    else if (args[i] === "--resume") resume = true;
    else if (args[i] === "--analyze") analyze = true;
    else if (args[i] === "--model" && args[i + 1]) model = args[++i];
    else if (args[i] === "--agent" && args[i + 1]) agent = args[++i] as AgentType;
  }
  if (!effort || !getEffortSpec(effort)) {
    const effortKeys = Object.keys(EFFORT_SPECS).join("|");
    console.error(`Usage: bun loops/optimize_perf.ts --effort <${effortKeys}> [options]`);
    console.error("");
    console.error("Options:");
    console.error(`  --effort <${effortKeys}>         Optimization to run (required)`);
    console.error("  --cycles N               Max cycles (default: 20)");
    console.error(`  --model NAME             Model: ${MODEL_KEYS} (default: qwen35b)`);
    console.error("  --agent claude|codex     AI agent to use (default: claude)");
    console.error("  --resume                 Resume from previous run (read history from log)");
    console.error("  --analyze                Print controller analysis from saved run state");
    console.error("  --dry-run                Build+bench baseline only, skip agent");
    console.error("");
    console.error("Efforts:");
    for (const [id, spec] of Object.entries(EFFORT_SPECS)) {
      console.error(`  ${id} = ${spec.summary}`);
    }
    process.exit(1);
  }
  if (agent !== "claude" && agent !== "codex") {
    console.error(`Unknown agent: ${agent}. Use 'claude' or 'codex'.`);
    process.exit(1);
  }
  if (!(model in MODELS)) {
    console.error(`Unknown model: ${model}. Use one of: ${MODEL_KEYS}.`);
    process.exit(1);
  }
  return { effort, cycles, dryRun, model, resume, agent, analyze };
}

// -- Display helpers ---------------------------------------------------------

const CLR = process.stdout.isTTY && !("NO_COLOR" in process.env);
const c = (code: string, t: string) => CLR ? `\x1b[${code}m${t}\x1b[0m` : t;
const SEP = "\u2500".repeat(64);
const BOX_INNER_WIDTH = 58;

function boxLine(text: string): string {
  const content = text.slice(0, BOX_INNER_WIDTH - 1);
  return `\u2551 ${content.padEnd(BOX_INNER_WIDTH - 1)}\u2551`;
}

// -- Command runner with streaming -------------------------------------------

type RunResult = { exitCode: number; signal: NodeJS.Signals | null; stdout: string; stderr: string };

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
    child.on("close", (code, signal) => {
      if (streamOutput && opts.stdoutLineFormatter && lineBuffer.trim()) {
        const f = opts.stdoutLineFormatter(lineBuffer);
        if (f !== null) process.stdout.write(f);
      }
      res({ exitCode: code ?? 1, signal, stdout, stderr });
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

export type BenchResult = {
  buildOk: boolean;
  buildOutput: string;
  tokPerSec: number | null;
  tokPerSecSamples: number[];
  correct: boolean;
  outputText: string;
  bandwidthUtil: number | null;
  bandwidthSamples: number[];
  error: string | null;
};

export type StepKind = "optimization" | "enablement" | "analysis" | "fix" | "rollback" | "unknown";

export type AgentReport = {
  description: string;
  selfAnalysis: string;
  nextIdeas: string[];
  stepKind: StepKind;
  rawText: string;
};

export type CycleRecord = {
  cycle: number;
  timestamp: string;
  description: string;
  selfAnalysis: string;
  nextIdeas: string[];
  stepKind: StepKind;
  changedFiles: string[];
  categoryTags: string[];
  tokPerSec: number | null;
  tokPerSecSamples: number[];
  bandwidthUtil: number | null;
  bandwidthSamples: number[];
  correct: boolean;
  improved: boolean;
  broken: boolean;
  kept: boolean;
  foundationKeep: boolean;
  decisionReason: string;
  outputText: string;
  commitHash: string | null;
};

export type BenchCheckpoint = {
  cycle: number;
  tokPerSec: number | null;
  tokPerSecSamples: number[];
  bandwidthUtil: number | null;
  bandwidthSamples: number[];
  outputText: string;
  commitHash: string | null;
};

export type LoopState = {
  effort: number;
  planDoc: string;
  benchmarkSignature?: string;
  runStartedAt: string;
  lastUpdatedAt: string;
  lastCycle: number;
  bestTokPerSec: number;
  bestCycle: number | null;
  bestCommitHash: string | null;
  bestResult: BenchCheckpoint | null;
  stalledCycles: number;
  consecutiveFoundationKeeps: number;
  cycles: CycleRecord[];
  failedApproaches: string[];
  ideas: string[];
  reviewSummaries: string[];
};

export type PromptContext = {
  cycles: CycleRecord[];
  failedApproaches: string[];
  ideas: string[];
  stalledCycles: number;
  consecutiveFoundationKeeps: number;
  reviewSummary: string | null;
  bestPerf: BenchCheckpoint | null;
};

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

function trunc(text: string, max: number): string {
  return text.length > max ? `${text.slice(0, max - 1)}…` : text;
}

function benchResultToCheckpoint(result: BenchResult, cycle: number, commitHash: string | null): BenchCheckpoint {
  return {
    cycle,
    tokPerSec: result.tokPerSec,
    tokPerSecSamples: [...result.tokPerSecSamples],
    bandwidthUtil: result.bandwidthUtil,
    bandwidthSamples: [...result.bandwidthSamples],
    outputText: result.outputText,
    commitHash,
  };
}

function checkpointToBenchResult(checkpoint: BenchCheckpoint): BenchResult {
  return {
    buildOk: true,
    buildOutput: "",
    tokPerSec: checkpoint.tokPerSec,
    tokPerSecSamples: [...checkpoint.tokPerSecSamples],
    correct: true,
    outputText: checkpoint.outputText,
    bandwidthUtil: checkpoint.bandwidthUtil,
    bandwidthSamples: [...checkpoint.bandwidthSamples],
    error: null,
  };
}

export function median(samples: number[]): number | null {
  if (samples.length === 0) return null;
  const sorted = [...samples].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)] ?? null;
}

function formatSampleList(samples: number[], digits = 2): string {
  if (samples.length === 0) return "";
  return ` [${samples.map((s) => s.toFixed(digits)).join(", ")}]`;
}

function summarizeBenchMetric(value: number | null, samples: number[], unit: string, digits = 2): string {
  if (value == null) return "unknown";
  return `${value.toFixed(digits)} ${unit}${formatSampleList(samples, digits)}`;
}

function tailHistory(history: string, maxLines = HISTORY_LINES_IN_PROMPT): string {
  const lines = history.split("\n").map((line) => line.trim()).filter(Boolean);
  return lines.slice(-maxLines).join("\n");
}

function inferStepKind(description: string, selfAnalysis: string): StepKind {
  const haystack = `${description}\n${selfAnalysis}`.toLowerCase();
  if (!haystack.trim()) return "unknown";
  if (/\b(fix|compile|build|correctness|crash|error)\b/.test(haystack)) return "fix";
  if (/\b(rollback|revert|undo|back out|step back)\b/.test(haystack)) return "rollback";
  if (/\b(measure|instrument|benchmark|profile|analy[sz]e|study|inspect)\b/.test(haystack)) return "analysis";
  if (/\b(enablement|plumbing|infrastructure|helper|wrapper|scaffold|layout|pipeline plumbing|descriptor plumbing)\b/.test(haystack)) {
    return "enablement";
  }
  if (/\b(push descriptor|descriptor|dispatch helper|call site conversion|pipeline layout)\b/.test(haystack)) {
    return "enablement";
  }
  return "optimization";
}

export function classifyApproachTags(description: string, changedFiles: string[]): string[] {
  const haystack = `${description}\n${changedFiles.join("\n")}`.toLowerCase();
  const tags: string[] = [];
  if (/\bdmmv\b|dmmv\.zig|matmul|q4_k|q5_k|q6_k|q8_0/.test(haystack)) tags.push("dmmv");
  if (/\b(attention|flash_attn|kv cache|kv_cache|rope)\b|attention\.zig/.test(haystack)) tags.push("attention");
  if (/\b(ssm|delta|conv1d|mamba)\b/.test(haystack)) tags.push("ssm");
  if (/\b(elementwise|swiglu|rms norm|sigmoid|softmax topk|scale acc)\b|elementwise\.zig/.test(haystack)) tags.push("elementwise");
  if (/\b(descriptor|push descriptor|pipeline layout)\b|pipeline\.zig|instance\.zig|command\.zig/.test(haystack)) tags.push("descriptor");
  if (/\b(shader|glsl|\.comp\b)\b|src\/shaders\//.test(haystack)) tags.push("shader");
  if (/\b(buffer|pool|alloc|memory|reuse)\b|buffer\.zig/.test(haystack)) tags.push("memory");
  if (/\b(check|test|correctness|coherence|output)\b/.test(haystack)) tags.push("correctness");
  if (/\b(bench|benchmark|measure|profile|instrument)\b/.test(haystack)) tags.push("measurement");
  if (tags.length === 0) tags.push("other");
  return [...new Set(tags)];
}

function isEnablementLike(report: AgentReport, changedFiles: string[]): boolean {
  if (report.stepKind === "enablement") return true;
  const text = `${report.description}\n${report.selfAnalysis}\n${changedFiles.join("\n")}`.toLowerCase();
  return /\b(enablement|plumbing|infrastructure|helper|wrapper|layout|pipeline|descriptor|call site conversion|scaffold)\b/.test(text);
}

function buildCycleHistoryEntry(cycle: CycleRecord): string {
  const outcome = cycle.improved
    ? "KEPT"
    : cycle.foundationKeep
      ? "KEPT-FOUNDATION"
      : cycle.broken
        ? "REVERTED-BROKEN"
        : "REVERTED";
  const metric = cycle.tokPerSec != null ? ` (${cycle.tokPerSec.toFixed(2)} tok/s)` : "";
  const tags = cycle.categoryTags.length > 0 ? ` [${cycle.categoryTags.join(", ")}]` : "";
  return `#${cycle.cycle}: ${outcome}${metric}${tags} ${trunc(cycle.description || cycle.decisionReason, 96)}`;
}

function buildHistoryFromCycles(cycles: CycleRecord[]): string {
  if (cycles.length === 0) return "";
  return cycles.slice(-HISTORY_LINES_IN_PROMPT).map(buildCycleHistoryEntry).join("\n");
}

function buildRecentCycleBlock(cycles: CycleRecord[]): string {
  if (cycles.length === 0) return "  (none yet)";
  return cycles.slice(-RECENT_CYCLES_IN_PROMPT).map((cycle) => `  ${buildCycleHistoryEntry(cycle)}`).join("\n");
}

export function buildSelfReview(state: Pick<LoopState, "cycles" | "stalledCycles" | "consecutiveFoundationKeeps">): string {
  const recent = state.cycles.slice(-SELF_REVIEW_EVERY);
  if (recent.length === 0) return "";

  const improved = recent.filter((cycle) => cycle.improved).length;
  const foundation = recent.filter((cycle) => cycle.foundationKeep).length;
  const broken = recent.filter((cycle) => cycle.broken).length;
  const reverted = recent.filter((cycle) => !cycle.kept).length;
  const tagStats = new Map<string, { kept: number; reverted: number }>();

  for (const cycle of recent) {
    for (const tag of cycle.categoryTags) {
      const entry = tagStats.get(tag) ?? { kept: 0, reverted: 0 };
      if (cycle.kept) entry.kept++;
      else entry.reverted++;
      tagStats.set(tag, entry);
    }
  }

  const deadEnds = [...tagStats.entries()]
    .filter(([, stats]) => stats.reverted > 0 && stats.kept === 0)
    .sort((a, b) => b[1].reverted - a[1].reverted)
    .slice(0, 3)
    .map(([tag, stats]) => `${tag}(${stats.reverted})`);

  const productive = [...tagStats.entries()]
    .filter(([, stats]) => stats.kept > 0)
    .sort((a, b) => (b[1].kept - a[1].kept) || (a[1].reverted - b[1].reverted))
    .slice(0, 3)
    .map(([tag, stats]) => `${tag}(${stats.kept} kept/${stats.reverted} reverted)`);

  const lines = [
    `Last ${recent.length} cycles: ${improved} perf keep, ${foundation} foundation keep, ${reverted} reverted, ${broken} broken.`,
  ];

  if (productive.length > 0) lines.push(`Productive directions: ${productive.join(", ")}.`);
  if (deadEnds.length > 0) lines.push(`Repeated dead ends: ${deadEnds.join(", ")}.`);
  if (state.consecutiveFoundationKeeps > 0) {
    lines.push(`Foundation debt: ${state.consecutiveFoundationKeeps} neutral keep(s) in a row; next cycles should either harvest a speed win or step back.`);
  }
  if (state.stalledCycles >= STALL_WARNING_THRESHOLD) {
    lines.push(`Stall warning: ${state.stalledCycles} cycles without a best-perf win. Stop repeating the last rejected category; pick a different hotspot or a smaller prerequisite.`);
  }

  return lines.join("\n");
}

export function buildAnalysisReport(state: LoopState): string {
  const total = state.cycles.length;
  const improved = state.cycles.filter((cycle) => cycle.improved).length;
  const foundation = state.cycles.filter((cycle) => cycle.foundationKeep).length;
  const broken = state.cycles.filter((cycle) => cycle.broken).length;
  const reverted = state.cycles.filter((cycle) => !cycle.kept).length;

  const tagStats = new Map<string, { kept: number; improved: number; reverted: number }>();
  for (const cycle of state.cycles) {
    for (const tag of cycle.categoryTags) {
      const entry = tagStats.get(tag) ?? { kept: 0, improved: 0, reverted: 0 };
      if (cycle.kept) entry.kept++;
      if (cycle.improved) entry.improved++;
      if (!cycle.kept) entry.reverted++;
      tagStats.set(tag, entry);
    }
  }

  const tagLines = [...tagStats.entries()]
    .sort((a, b) => (b[1].improved - a[1].improved) || (b[1].kept - a[1].kept) || (b[1].reverted - a[1].reverted))
    .slice(0, 8)
    .map(([tag, stats]) => `- ${tag}: ${stats.improved} perf keeps, ${stats.kept} total keeps, ${stats.reverted} reverts`);

  const recent = buildRecentCycleBlock(state.cycles);
  const failed = state.failedApproaches.length > 0
    ? state.failedApproaches.slice(-10).map((entry) => `- ${entry}`).join("\n")
    : "- none";
  const ideas = state.ideas.length > 0
    ? state.ideas.slice(-10).map((entry) => `- ${entry}`).join("\n")
    : "- none";
  const review = state.reviewSummaries.at(-1) ?? buildSelfReview(state);
  const metricLabel = getEffortSpec(state.effort)?.primaryMetricLabel ?? "tok/s";

  return [
    `Run started: ${state.runStartedAt}`,
    `Cycles: ${total} total, ${improved} perf keeps, ${foundation} foundation keeps, ${reverted} reverted, ${broken} broken`,
    `Best checkpoint (${metricLabel}): ${state.bestTokPerSec.toFixed(2)} tok/s (cycle ${state.bestCycle ?? "?"}${state.bestCommitHash ? `, ${state.bestCommitHash.slice(0, 8)}` : ""})`,
    `Current stall count: ${state.stalledCycles}`,
    "",
    "Recent review:",
    review || "No review yet.",
    "",
    "Category stats:",
    tagLines.length > 0 ? tagLines.join("\n") : "- none",
    "",
    "Recent cycles:",
    recent,
    "",
    "Failed approaches:",
    failed,
    "",
    "Idea bank:",
    ideas,
  ].join("\n");
}

export function improvementThreshold(currentTokPerSec: number | null): number {
  if (currentTokPerSec == null || currentTokPerSec <= 0) return MIN_IMPROVEMENT_ABS_TPS;
  return Math.max(MIN_IMPROVEMENT_ABS_TPS, currentTokPerSec * MIN_IMPROVEMENT_PCT);
}

export function isMaterialImprovement(candidate: BenchResult, currentBest: BenchResult): boolean {
  if (candidate.tokPerSec == null) return false;
  const threshold = improvementThreshold(currentBest.tokPerSec);
  const current = currentBest.tokPerSec ?? 0;
  return candidate.tokPerSec > current + threshold;
}

export function buildAgentPrompt(
  plan: string,
  originalBaseline: BenchResult,
  currentBest: BenchResult,
  cycleNum: number,
  history: string,
  model: string,
  context: PromptContext | null = null,
  options: { primaryMetricLabel?: string; benchmarkMethod?: string } = {},
): string {
  const modelTarget = MODELS[model] ?? MODELS.qwen35b;
  const sanityCheckPrompt = coherencePromptForMode(
    COHERENCE_CHECKS[0],
    coherencePromptModeForModel(modelTarget),
  );
  const primaryMetricLabel = options.primaryMetricLabel ?? "decode tok/s";
  const benchmarkMethod = options.benchmarkMethod ?? "200-token decode benchmark on the primary model";
  const historySummary = tailHistory(history);
  const failedBlock = context?.failedApproaches?.length
    ? context.failedApproaches.slice(-12).map((entry, i) => `${i + 1}. ${trunc(entry, 140)}`).join("\n")
    : "None yet.";
  const ideasBlock = context?.ideas?.length
    ? context.ideas.slice(-10).map((entry, i) => `${i + 1}. ${trunc(entry, 140)}`).join("\n")
    : "None yet.";
  const recentCyclesBlock = context ? buildRecentCycleBlock(context.cycles) : "  (state unavailable)";
  const reviewBlock = context?.reviewSummary || "No self-review yet.";
  const bestPerf = context?.bestPerf ?? benchResultToCheckpoint(currentBest, 0, null);
  const currentVsBestNote = bestPerf.tokPerSec != null && currentBest.tokPerSec != null && bestPerf.tokPerSec > currentBest.tokPerSec + 0.05
    ? `- Note: the current checked-out code is ${currentBest.tokPerSec.toFixed(2)} tok/s, below the best checkpoint ${bestPerf.tokPerSec.toFixed(2)} tok/s. You are editing the current code, but real wins are still judged against the best checkpoint.`
    : "- Note: current code and best checkpoint are effectively the same right now.";
  const controllerMode = context && context.stalledCycles >= STALL_WARNING_THRESHOLD
    ? "STEP_BACK"
    : context && context.consecutiveFoundationKeeps > 0
      ? "HARVEST"
      : "ADVANCE";
  return `You are implementing a performance optimization for the ZINC Vulkan inference engine.

## Optimization Plan
${plan}

## Benchmark Focus
- primary metric: ${primaryMetricLabel}
- benchmark method: ${benchmarkMethod}
- success is judged on the primary metric above, not on one lucky decode sample from a different workload.

## Current Checked-Out Code (build on this code)
- primary metric (${primaryMetricLabel}): ${summarizeBenchMetric(currentBest.tokPerSec, currentBest.tokPerSecSamples, "tok/s")}
- bandwidth utilization: ${summarizeBenchMetric(currentBest.bandwidthUtil, currentBest.bandwidthSamples, "%", 1)}
- output: "${currentBest.outputText}" (coherence tested with 3 prompts on 7 models after every change)
- This is the performance of the code currently checked out in the worktree.

## Best Accepted Performance Checkpoint
- primary metric (${primaryMetricLabel}): ${summarizeBenchMetric(bestPerf.tokPerSec, bestPerf.tokPerSecSamples, "tok/s")}
- bandwidth utilization: ${summarizeBenchMetric(bestPerf.bandwidthUtil, bestPerf.bandwidthSamples, "%", 1)}
- output: "${bestPerf.outputText}"
- cycle: ${bestPerf.cycle}${bestPerf.commitHash ? `, commit ${bestPerf.commitHash.slice(0, 8)}` : ""}
${currentVsBestNote}

## Original Run Baseline (for total gain only)
- primary metric (${primaryMetricLabel}): ${summarizeBenchMetric(originalBaseline.tokPerSec, originalBaseline.tokPerSecSamples, "tok/s")}
- bandwidth utilization: ${summarizeBenchMetric(originalBaseline.bandwidthUtil, originalBaseline.bandwidthSamples, "%", 1)}
- output: "${originalBaseline.outputText}"

## Controller State
- mode: ${controllerMode}
- stalled cycles without a new best checkpoint: ${context?.stalledCycles ?? 0}
- consecutive neutral foundation keeps: ${context?.consecutiveFoundationKeeps ?? 0}

## Recent Cycle Ledger
${recentCyclesBlock}

## Reflection (auto-analysis of recent cycles)
${reviewBlock}

## Previous Attempts
${historySummary || "None yet."}

## Failed Approaches (do not repeat)
${failedBlock}

## Idea Bank
${ideasBlock}

## Your Task (Cycle ${cycleNum})
Implement ONE concrete step from the optimization plan above. Pick the next unfinished step.
Your change must beat the best accepted performance checkpoint above, not the original run baseline.
If controller mode is STEP_BACK, do not repeat the same hotspot as the last rejected cycles. Either choose a smaller prerequisite, finish a kept enablement step, or switch to a different bottleneck category.
If you intentionally do a plumbing/enabling step that may be performance-neutral this cycle, mark it as enablement and explain exactly which next step it unlocks.
Do not use sub-agents, delegation, spawn_agent, or wait_agent. Work directly in this repo.
Before editing any file, re-read the exact current contents from disk. Do not rely on stale context, guessed line numbers, or cached snippets.

## CRITICAL RULES — READ CAREFULLY

1. **BUILD MUST PASS.** Before you declare yourself done, you MUST:
   a. rsync your changes to the remote node
   b. Compile shaders: ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR}/src/shaders && for f in *.comp; do glslc --target-env=vulkan1.3 -fshader-stage=compute \\$f -o \\$\{f%.comp}.spv 2>&1; done"
   c. Build: ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR} && zig build -Doptimize=ReleaseFast 2>&1"
   d. If the build fails, FIX THE ERRORS before finishing. Do NOT leave broken code.

2. **Incremental steps.** If the optimization requires changing many call sites (e.g. 60+ descriptor set conversions), break it into compilable stages:
   - Add new infrastructure (new functions, new fields) FIRST — the old code can coexist.
   - Convert call sites in batches, building after each batch to catch errors.
   - Remove old infrastructure LAST.
   - The code MUST compile at every stage.

3. **ONE focused change per cycle.** Don't try to convert the entire codebase in one shot.

4. **Avoid repeated dead ends.**
   - Read the recent cycle ledger and failed approaches first.
   - If the last few rejected cycles hit the same subsystem, do NOT do another cosmetic variation of that same idea.
   - If you are uncertain, add a tiny enabling or measurement step instead of another large speculative refactor.

5. **Test on remote node:**
   rsync -avz --checksum --delete -e "ssh -p ${ZINC_PORT} -o StrictHostKeyChecking=no" --exclude .zig-cache --exclude zig-out --exclude node_modules --exclude .git --exclude .perf_optimize --exclude .zinc_optimize --exclude site --exclude .DS_Store ${REPO_ROOT}/ ${ZINC_USER}@${ZINC_HOST}:${REMOTE_DIR}/
   ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR} && zig build -Doptimize=ReleaseFast && ${REMOTE_ZINC_ENV} ./zig-out/bin/zinc ${zincCliArgs(modelTarget, sanityCheckPrompt, 16)}"

6. **Shader compilation:** glslc --target-env=vulkan1.3 -fshader-stage=compute file.comp -o file.spv

Files you may edit:
- src/compute/*.zig (forward.zig, dmmv.zig, elementwise.zig, attention.zig, argmax.zig)
- src/vulkan/*.zig (pipeline.zig, command.zig, buffer.zig, instance.zig)
- src/model/*.zig (tokenizer.zig, loader.zig, config.zig, architecture.zig)
- src/server/*.zig (routes.zig, runtime.zig)
- src/server/chat.html
- src/shaders/*.comp (GLSL compute shaders)
- src/main.zig

## Output Format
After making your change, print these lines:
@@@DESCRIPTION: <one-line summary of the change>
@@@STEP_KIND: <optimization|enablement|analysis|fix|rollback>
@@@SELF_ANALYSIS: <why this direction, expected effect, and what should happen next>
@@@NEXT_IDEAS: <semicolon-separated follow-up ideas>`;
}

function metricParserForSpec(spec: EffortSpec): (output: string) => number | null {
  return spec.metricMode === "prefill" ? parsePrefillTokPerSec : parseTokPerSec;
}

function coherencePromptForMode(check: CoherenceCheck, promptMode: PromptMode): string {
  return promptMode === "chat" ? check.chatPrompt : check.rawPrompt;
}

function coherencePromptModeForModel(modelTarget: ModelTarget): PromptMode {
  return modelTarget.coherencePromptMode ?? modelTarget.promptMode;
}

function coherenceMaxTokensForModel(modelTarget: ModelTarget): number {
  return modelTarget.coherenceMaxTokens ?? 30;
}

function zincCliArgs(modelTarget: ModelTarget, prompt: string, maxTokens: number, promptMode = modelTarget.promptMode): string {
  const chatFlag = promptMode === "chat" ? " --chat" : "";
  return `-m ${shellQuote(modelTarget.path)}${chatFlag} --prompt ${shellQuote(prompt)} -n ${maxTokens}`;
}

function zincRemoteCommand(modelTarget: ModelTarget, prompt: string, maxTokens: number, promptMode = modelTarget.promptMode): string {
  return `cd ${REMOTE_DIR} && ${REMOTE_ZINC_ENV} ./zig-out/bin/zinc ${zincCliArgs(modelTarget, prompt, maxTokens, promptMode)} 2>&1`;
}

async function buildAndBench(modelTarget: ModelTarget, effortSpec: EffortSpec): Promise<BenchResult> {
  console.log(c("2", "  Compiling shaders..."));
  try {
    await ssh(`cd ${REMOTE_DIR}/src/shaders && for f in *.comp; do glslc --target-env=vulkan1.3 -fshader-stage=compute $f -o \${f%.comp}.spv 2>&1; done`, 60_000);
  } catch (e) {
    return {
      buildOk: false,
      buildOutput: String(e),
      tokPerSec: null,
      tokPerSecSamples: [],
      correct: false,
      outputText: "",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: "shader compile failed",
    };
  }

  console.log(c("2", "  Building..."));
  let buildOutput: string;
  try {
    buildOutput = await ssh(`cd ${REMOTE_DIR} && zig build -Doptimize=ReleaseFast 2>&1`, 300_000);
  } catch (e) {
    return {
      buildOk: false,
      buildOutput: String(e),
      tokPerSec: null,
      tokPerSecSamples: [],
      correct: false,
      outputText: "",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: "zig build failed",
    };
  }
  if (buildOutput.includes("error:")) {
    return {
      buildOk: false,
      buildOutput,
      tokPerSec: null,
      tokPerSecSamples: [],
      correct: false,
      outputText: "",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: "build errors",
    };
  }

  // Quick correctness check (short prompt, few tokens)
  console.log(c("2", "  Running correctness test..."));
  let correctnessOutput: string;
  const firstCheck = COHERENCE_CHECKS[0];
  const correctnessPromptMode = coherencePromptModeForModel(modelTarget);
  const correctnessPrompt = coherencePromptForMode(firstCheck, correctnessPromptMode);
  const correctnessMaxTokens = coherenceMaxTokensForModel(modelTarget);
  try {
    correctnessOutput = await ssh(
      zincRemoteCommand(modelTarget, correctnessPrompt, correctnessMaxTokens, correctnessPromptMode),
      180_000,
    );
  } catch (e) {
    return {
      buildOk: true,
      buildOutput,
      tokPerSec: null,
      tokPerSecSamples: [],
      correct: false,
      outputText: "",
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: `run failed: ${e}`,
    };
  }

  const textMatch = correctnessOutput.match(/Output text:\s*(.+)/i);
  const outputText = textMatch ? textMatch[1].trim() : "";
  const correct = firstCheck.expect.every(e => outputText.toLowerCase().includes(e.toLowerCase()));

  if (!correct) {
    return {
      buildOk: true,
      buildOutput,
      tokPerSec: null,
      tokPerSecSamples: [],
      correct: false,
      outputText,
      bandwidthUtil: null,
      bandwidthSamples: [],
      error: "incorrect output",
    };
  }

  const parseMetric = metricParserForSpec(effortSpec);
  console.log(c(
    "2",
    `  Benchmarking (${BENCHMARK_SAMPLES} x ${effortSpec.benchmarkMethod}, primary metric: ${effortSpec.primaryMetricLabel})...`,
  ));
  const tokPerSecSamples: number[] = [];
  const bandwidthSamples: number[] = [];
  for (let sample = 0; sample < BENCHMARK_SAMPLES; sample++) {
    let benchOutput: string;
    try {
      benchOutput = await ssh(
        zincRemoteCommand(modelTarget, effortSpec.benchmarkPrompt, effortSpec.benchmarkMaxTokens),
        300_000,
      );
    } catch (e) {
      return {
        buildOk: true,
        buildOutput,
        tokPerSec: null,
        tokPerSecSamples,
        correct: true,
        outputText,
        bandwidthUtil: null,
        bandwidthSamples,
        error: `bench failed: ${e}`,
      };
    }

    const tps = parseMetric(benchOutput);
    const bw = effortSpec.metricMode === "decode" ? parseBandwidthUtil(benchOutput) : null;
    if (tps != null) tokPerSecSamples.push(tps);
    if (bw != null) bandwidthSamples.push(bw);
    console.log(c(
      "2",
      `    sample ${sample + 1}/${BENCHMARK_SAMPLES}: ${tps?.toFixed(2) ?? "?"} tok/s (${effortSpec.primaryMetricLabel})${bw != null ? `, BW ${bw.toFixed(1)}%` : ""}`,
    ));
  }

  const tokPerSec = median(tokPerSecSamples);
  const bandwidthUtil = median(bandwidthSamples);

  return {
    buildOk: true,
    buildOutput,
    tokPerSec,
    tokPerSecSamples,
    correct,
    outputText,
    bandwidthUtil,
    bandwidthSamples,
    error: tokPerSec == null ? `${effortSpec.primaryMetricLabel} parse failed` : null,
  };
}

/// Run ALL coherence prompts on ALL models.
/// Returns the full sweep so the controller can enforce non-regression against
/// the accepted baseline instead of demanding global cross-model cleanliness.
function coherenceCaseId(model: string, prompt: string): string {
  return `${model}::${prompt}`;
}

function coherenceCaseLabel(model: string, prompt: string): string {
  return `${model} [${prompt.slice(0, 25)}]`;
}

function formatCoherenceFailure(failure: CoherenceFailure): string {
  return failure.kind === "crash"
    ? `${failure.label}: crashed`
    : `${failure.label}: "${failure.outputText.slice(0, 50)}"`;
}

export function formatCoherenceFailureList(failures: CoherenceFailure[]): string {
  return failures.map((failure) => formatCoherenceFailure(failure)).join("; ");
}

export function summarizeCoherenceRegression(
  candidate: CoherenceSweep,
  acceptedFailureIds: string[],
): string | null {
  const accepted = new Set(acceptedFailureIds);
  const regressions = candidate.failures.filter((failure) => !accepted.has(failure.id));
  if (regressions.length === 0) return null;
  return `New coherence failures vs accepted baseline: ${formatCoherenceFailureList(regressions)}`;
}

async function runCoherenceSweep(): Promise<CoherenceSweep> {
  const failures: CoherenceFailure[] = [];
  for (const modelTarget of COHERENCE_MODELS) {
    const promptMode = coherencePromptModeForModel(modelTarget);
    const maxTokens = coherenceMaxTokensForModel(modelTarget);
    for (const check of COHERENCE_CHECKS) {
      const prompt = coherencePromptForMode(check, promptMode);
      const label = coherenceCaseLabel(modelTarget.name, prompt);
      try {
        const out = await ssh(
          zincRemoteCommand(modelTarget, prompt, maxTokens, promptMode),
          120_000,
        );
        const textMatch = out.match(/Output text:\s*(.+)/i);
        const outputText = textMatch ? textMatch[1].trim() : "";
        const pass = check.expect.every(e => outputText.toLowerCase().includes(e.toLowerCase()));
        if (!pass) {
          failures.push({
            id: coherenceCaseId(modelTarget.name, prompt),
            label,
            model: modelTarget.name,
            prompt,
            outputText,
            kind: "mismatch",
          });
        }
      } catch (e) {
        failures.push({
          id: coherenceCaseId(modelTarget.name, prompt),
          label,
          model: modelTarget.name,
          prompt,
          outputText: "",
          kind: "crash",
        });
      }
    }
    if (!failures.some((failure) => failure.model === modelTarget.name)) {
      console.log(c("2", `    ${modelTarget.name}: all ${COHERENCE_CHECKS.length} prompts OK`));
    }
  }
  return {
    failures,
    failureIds: failures.map((failure) => failure.id),
  };
}

// -- Codex stream formatter --------------------------------------------------

export function formatCodexStreamLine(rawLine: string): string | null {
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

export type ClaudeStreamState = {
  currentToolName: string | null;
  currentBlockIsToolUse: boolean;
  inputJsonBuffer: string;
  inTextBlock: boolean;
  sawTextDeltaInCurrentMessage: boolean;
};

export function formatToolInput(name: string, rawJson: string): string {
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

export function formatClaudeStreamLine(rawLine: string, state: ClaudeStreamState): string | null {
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

function extractAgentText(stdout: string): string {
  const texts: string[] = [];
  for (const line of stdout.split("\n")) {
    if (!line.trim()) continue;
    try {
      const evt = JSON.parse(line) as Record<string, unknown>;
      const type = evt.type;
      if (type === "assistant") {
        const content = (evt.message as Record<string, unknown> | undefined)?.content;
        if (Array.isArray(content)) {
          for (const block of content) {
            const text = (block as Record<string, unknown>)?.text;
            if (typeof text === "string" && text.trim()) texts.push(text);
          }
        }
      } else if (type === "message" || type === "agent") {
        const text = evt.content ?? evt.message;
        if (typeof text === "string" && text.trim()) texts.push(text);
      } else if (type === "item.completed") {
        const item = evt.item as Record<string, unknown> | undefined;
        if (item?.type === "agent_message") {
          const text = item.text ?? item.message ?? item.output_text ?? item.content;
          if (typeof text === "string" && text.trim()) texts.push(text);
        }
      }
    } catch {
      if (line.trim().startsWith("@@@")) texts.push(line.trim());
    }
  }
  return texts.join("\n");
}

export function parseAgentReport(stdout: string): AgentReport {
  const rawText = extractAgentText(stdout).trim();
  const window = rawText.slice(-4000);
  const description = window.match(/@@@DESCRIPTION:\s*(.+)/im)?.[1]?.trim()
    ?? rawText.split("\n").map((line) => line.trim()).find(Boolean)
    ?? "Agent made changes";
  const selfAnalysis = window.match(/@@@SELF_ANALYSIS:\s*(.+)/im)?.[1]?.trim() ?? "";
  const stepKindRaw = window.match(/@@@STEP_KIND:\s*(.+)/im)?.[1]?.trim().toLowerCase() ?? "";
  const stepKind = ["optimization", "enablement", "analysis", "fix", "rollback"].includes(stepKindRaw)
    ? stepKindRaw as StepKind
    : inferStepKind(description, selfAnalysis);
  const ideasRaw = window.match(/@@@NEXT_IDEAS:\s*(.+)/im)?.[1]?.trim() ?? "";
  const nextIdeas = ideasRaw
    .split(/[;,]/)
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 3);

  return {
    description,
    selfAnalysis,
    nextIdeas,
    stepKind,
    rawText,
  };
}

async function listChangedFiles(): Promise<string[]> {
  const tracked = await runCommand("git", ["diff", "--name-only", "--", "src/"], { cwd: REPO_ROOT });
  const untracked = await runCommand("git", ["ls-files", "--others", "--exclude-standard", "src/"], { cwd: REPO_ROOT });
  const files = [
    ...tracked.stdout.split("\n"),
    ...untracked.stdout.split("\n"),
  ].map((entry) => entry.trim()).filter(Boolean);
  return [...new Set(files)].sort();
}

// -- Agent spawn -------------------------------------------------------------

async function spawnAgent(
  _effortDoc: string,
  plan: string,
  originalBaseline: BenchResult,
  currentBest: BenchResult,
  cycleNum: number,
  history: string,
  model: string,
  agent: AgentType = "claude",
  context: PromptContext | null = null,
  effortSpec: EffortSpec | null = null,
): Promise<RunResult> {
  const prompt = buildAgentPrompt(plan, originalBaseline, currentBest, cycleNum, history, model, context, {
    primaryMetricLabel: effortSpec?.primaryMetricLabel,
    benchmarkMethod: effortSpec?.benchmarkMethod,
  });

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
    // Codex: uses `codex exec` with bypass sandbox (needs SSH/rsync to RDNA node)
    result = await runCommand("codex", codexExecArgs(prompt), {
      cwd: REPO_ROOT,
      timeout: 7_200_000,
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
      "--effort", CLAUDE_EFFORT,
      prompt,
    ], {
      cwd: REPO_ROOT,
      timeout: 7_200_000,
      streamOutput: true,
      stdoutLineFormatter: (line) => formatClaudeStreamLine(line, claudeState),
    });
  }

  clearInterval(heartbeat);
  console.log(c("1;36", SEP));
  console.log(c("1;32", `  \u2705 Agent done in ${formatElapsed(startedAt)}`));
  console.log(c("1;36", SEP));

  if (result.exitCode !== 0 || result.signal) {
    const how = result.signal ? `killed by ${result.signal}` : `exited with code ${result.exitCode}`;
    console.log(c("1;31", `  Agent ${how}`));
    const tailStderr = result.stderr.slice(-2000).trimEnd();
    const tailStdout = result.stdout.slice(-2000).trimEnd();
    if (tailStderr) console.log(c("1;31", `  stderr tail:\n${tailStderr}`));
    if (tailStdout) console.log(c("2", `  stdout tail:\n${tailStdout}`));
  }

  return result;
}

// -- Resume from previous run ------------------------------------------------

type LogEntry = {
  cycle: number;
  effort: number;
  tokPerSec: number | null;
  tokPerSecSamples?: number[];
  bandwidthUtil: number | null;
  bandwidthSamples?: number[];
  correct: boolean;
  improved: boolean;
  broken: boolean;
  kept?: boolean;
  foundationKeep?: boolean;
  decisionReason?: string;
  description?: string;
  stepKind?: StepKind;
  changedFiles?: string[];
  outputText: string;
  commitHash?: string | null;
  timestamp: string;
};

export function codexExecArgs(prompt: string): string[] {
  return [
    "exec",
    "-c",
    `model_reasoning_effort="${CODEX_REASONING_EFFORT}"`,
    "--dangerously-bypass-approvals-and-sandbox",
    "--json",
    prompt,
  ];
}

function statePathForEffort(effort: number): string {
  return join(RESULTS_DIR, `effort_${effort}_state.json`);
}

function logPathForEffort(effort: number): string {
  return join(RESULTS_DIR, `effort_${effort}_log.jsonl`);
}

export function effortArtifactPaths(effort: number): string[] {
  return [
    statePathForEffort(effort),
    logPathForEffort(effort),
  ];
}

export async function cleanupPreviousRunArtifacts(effort: number): Promise<string[]> {
  const removed: string[] = [];
  for (const path of effortArtifactPaths(effort)) {
    if (!existsSync(path)) continue;
    await rm(path, { force: true });
    removed.push(path);
  }
  return removed;
}

async function loadLoopState(effort: number): Promise<LoopState | null> {
  const statePath = statePathForEffort(effort);
  if (!existsSync(statePath)) return null;
  return JSON.parse(await readFile(statePath, "utf8")) as LoopState;
}

async function saveLoopState(state: LoopState): Promise<void> {
  state.lastUpdatedAt = new Date().toISOString();
  await writeFile(statePathForEffort(state.effort), JSON.stringify(state, null, 2));
}

export function benchmarkSignatureForSpec(spec: EffortSpec): string {
  return JSON.stringify({
    doc: spec.doc,
    metricMode: spec.metricMode,
    primaryMetricLabel: spec.primaryMetricLabel,
    benchmarkPrompt: spec.benchmarkPrompt,
    benchmarkMaxTokens: spec.benchmarkMaxTokens,
    benchmarkMethod: spec.benchmarkMethod,
  });
}

export function isResumeStateCompatible(saved: LoopState, spec: EffortSpec): boolean {
  return saved.benchmarkSignature === benchmarkSignatureForSpec(spec);
}

function createInitialState(
  effort: number,
  planDoc: string,
  baseline: BenchResult,
  headCommit: string | null,
  benchmarkSignature: string,
): LoopState {
  const now = new Date().toISOString();
  return {
    effort,
    planDoc,
    benchmarkSignature,
    runStartedAt: now,
    lastUpdatedAt: now,
    lastCycle: 0,
    bestTokPerSec: baseline.tokPerSec ?? 0,
    bestCycle: 0,
    bestCommitHash: headCommit,
    bestResult: benchResultToCheckpoint(baseline, 0, headCommit),
    stalledCycles: 0,
    consecutiveFoundationKeeps: 0,
    cycles: [],
    failedApproaches: [],
    ideas: [],
    reviewSummaries: [],
  };
}

export function shouldKeepFoundationStep(
  candidate: BenchResult,
  bestPerf: BenchResult,
  stalledCycles: number,
  consecutiveFoundationKeeps: number,
  report: AgentReport,
  changedFiles: string[],
): boolean {
  if (!candidate.buildOk || !candidate.correct || candidate.tokPerSec == null) return false;
  if (!isEnablementLike(report, changedFiles)) return false;
  if (consecutiveFoundationKeeps >= MAX_FOUNDATION_KEEPS_IN_A_ROW) return false;
  if (changedFiles.length === 0) return false;

  const bestTokPerSec = bestPerf.tokPerSec ?? 0;
  if (candidate.tokPerSec > bestTokPerSec + improvementThreshold(bestTokPerSec)) return false;
  if (candidate.tokPerSec < bestTokPerSec - FOUNDATION_KEEP_MAX_DROP_TPS) return false;

  return stalledCycles >= 2 || report.stepKind === "enablement";
}

export async function loadPreviousRun(effort: number): Promise<{
  history: string;
  bestTokPerSec: number;
  lastCycle: number;
  bestCycle: number | null;
  bestCommitHash: string | null;
}> {
  const state = await loadLoopState(effort);
  if (state) {
    return {
      history: buildHistoryFromCycles(state.cycles),
      bestTokPerSec: state.bestTokPerSec,
      lastCycle: state.lastCycle,
      bestCycle: state.bestCycle,
      bestCommitHash: state.bestCommitHash,
    };
  }

  const logPath = logPathForEffort(effort);
  let history = "";
  let bestTokPerSec = 0;
  let lastCycle = 0;
  let bestCycle: number | null = null;
  let bestCommitHash: string | null = null;

  try {
    const content = await readFile(logPath, "utf8");
    for (const line of content.split("\n").filter(Boolean)) {
      try {
        const entry = JSON.parse(line) as LogEntry;
        if (entry.effort !== effort) continue;
        lastCycle = Math.max(lastCycle, entry.cycle);
        if (entry.broken) {
          history += `\nCycle ${entry.cycle}: REVERTED \u2014 ${entry.decisionReason ?? `broken (${entry.outputText?.slice(0, 60)})`}`;
        } else if (entry.improved) {
          history += `\nCycle ${entry.cycle}: KEPT \u2014 ${entry.tokPerSec?.toFixed(2)} tok/s${entry.tokPerSecSamples?.length ? ` ${formatSampleList(entry.tokPerSecSamples)}` : ""}`;
          if (entry.tokPerSec != null && entry.tokPerSec > bestTokPerSec) {
            bestTokPerSec = entry.tokPerSec;
            bestCycle = entry.cycle;
            bestCommitHash = entry.commitHash ?? null;
          }
        } else if (entry.foundationKeep) {
          history += `\nCycle ${entry.cycle}: KEPT-FOUNDATION \u2014 ${entry.description ?? entry.decisionReason ?? "enablement step"}`;
        } else {
          history += `\nCycle ${entry.cycle}: REVERTED \u2014 ${entry.decisionReason ?? `no improvement (${entry.tokPerSec?.toFixed(2)} tok/s${entry.tokPerSecSamples?.length ? ` ${formatSampleList(entry.tokPerSecSamples)}` : ""})`}`;
        }
      } catch { /* skip malformed lines */ }
    }
  } catch { /* no log file yet */ }

  return { history, bestTokPerSec, lastCycle, bestCycle, bestCommitHash };
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
  const { effort, cycles, dryRun, model, resume, agent, analyze } = parseArgs();
  const modelTarget = MODELS[model] ?? MODELS.qwen35b;
  const effortSpec = getEffortSpec(effort);
  if (!effortSpec) {
    throw new Error(`Unknown effort: ${effort}`);
  }
  const effortFile = effortSpec.doc;
  const plan = await readFile(join(REPO_ROOT, effortFile), "utf8");

  await mkdir(RESULTS_DIR, { recursive: true });

  if (analyze) {
    const saved = await loadLoopState(effort);
    if (!saved) {
      console.error(c("1;31", `No saved state found for effort ${effort}.`));
      process.exit(1);
    }
    console.log(buildAnalysisReport(saved));
    return;
  }

  console.log(c("1;37", `\n\u2554${"═".repeat(BOX_INNER_WIDTH)}\u2557`));
  console.log(c("1;37", boxLine(`ZINC Performance Optimization Loop — Effort ${effort}`)));
  console.log(c("1;37", boxLine(effortFile)));
  console.log(c("1;37", boxLine(`Model: ${model}`)));
  console.log(c("1;37", boxLine(`Agent: ${agent}`)));
  console.log(c("1;37", boxLine(`Cycles this run: ${cycles}`)));
  if (resume) console.log(c("1;37", boxLine("Resuming from previous run")));
  console.log(c("1;37", `\u255A${"═".repeat(BOX_INNER_WIDTH)}\u255D\n`));

  if (!resume) {
    const removedArtifacts = await cleanupPreviousRunArtifacts(effort);
    if (removedArtifacts.length > 0) {
      console.log(c("2", `  Cleaned ${removedArtifacts.length} saved artifact(s) from previous effort-${effort} runs.`));
    }
  }

  // Step 1: Sync and get baseline
  console.log(c("1;33", "\u2500\u2500 Baseline " + "\u2500".repeat(54)));
  await rsyncToRemote();
  const originalBaseline = await buildAndBench(modelTarget, effortSpec);

  if (!originalBaseline.buildOk) {
    console.error(c("1;31", "Baseline build failed! Fix build errors first."));
    process.exit(1);
  }
  if (!originalBaseline.correct) {
    console.error(c("1;31", `Baseline output incorrect: "${originalBaseline.outputText}". Fix correctness first.`));
    process.exit(1);
  }

  console.log(c("1;32", `  Baseline (${effortSpec.primaryMetricLabel}): ${summarizeBenchMetric(originalBaseline.tokPerSec, originalBaseline.tokPerSecSamples, "tok/s")}, BW: ${summarizeBenchMetric(originalBaseline.bandwidthUtil, originalBaseline.bandwidthSamples, "%", 1)}`));
  console.log(c("1;32", `  Output: "${originalBaseline.outputText.slice(0, 80)}"`));

  const benchmarkSignature = benchmarkSignatureForSpec(effortSpec);
  let currentCode = originalBaseline;
  let bestPerf = originalBaseline;
  let bestTokPerSec = bestPerf.tokPerSec ?? 0;
  let startCycle = 1;
  const headCommit = (await runCommand("git", ["rev-parse", "HEAD"], { cwd: REPO_ROOT })).stdout.trim() || null;
  let state = createInitialState(effort, effortFile, originalBaseline, headCommit, benchmarkSignature);

  if (resume) {
    const saved = await loadLoopState(effort);
    if (saved) {
      if (!isResumeStateCompatible(saved, effortSpec)) {
        console.log(c(
          "1;33",
          "  Resume note: saved state uses an older or different benchmark signature. Ignoring it and starting fresh for this effort.",
        ));
      } else {
        state = saved;
        startCycle = saved.lastCycle + 1;
        if (saved.bestResult) {
          bestPerf = checkpointToBenchResult(saved.bestResult);
          bestTokPerSec = saved.bestTokPerSec;
        }
        console.log(c("1;36", `  Resumed: ${saved.lastCycle} previous cycles, recorded best ${saved.bestTokPerSec.toFixed(2)} tok/s (${effortSpec.primaryMetricLabel})`));
        if (saved.cycles.length > 0) {
          console.log(c("2", `  Recent cycles:\n${buildRecentCycleBlock(saved.cycles)}`));
        }
        if (saved.bestTokPerSec > (currentCode.tokPerSec ?? 0) + improvementThreshold(currentCode.tokPerSec)) {
          const bestCommitNote = saved.bestCommitHash ? ` on commit ${saved.bestCommitHash.slice(0, 8)}` : "";
          console.log(c(
            "1;33",
            `  Resume note: recorded best cycle ${saved.bestCycle ?? "?"}${bestCommitNote} was faster than the current HEAD benchmark. The loop will branch from the code you currently have checked out, not from that historical metric.`,
          ));
        }
        if (saved.reviewSummaries.length > 0) {
          console.log(c("2", `  Latest review:\n${saved.reviewSummaries.at(-1)}`));
        }
      }
    } else {
      console.log(c("2", "  No previous run found, starting fresh."));
    }
  }

  console.log(c("2", "  Capturing accepted coherence baseline..."));
  let acceptedCoherence = await runCoherenceSweep();
  if (acceptedCoherence.failures.length > 0) {
    console.log(c("1;33", "  Accepted baseline already has cross-model failures; enforcing non-regression only."));
    console.log(c("2", `    ${formatCoherenceFailureList(acceptedCoherence.failures)}`));
  }

  let history = buildHistoryFromCycles(state.cycles);

  // Step 2: Optimization cycles
  for (let cycle = startCycle; cycle < startCycle + cycles; cycle++) {
    console.log(c("1;33", `\n\u2500\u2500 Cycle ${cycle} ` + "\u2500".repeat(54)));

    if (dryRun) {
      console.log(c("2", "  Dry run \u2014 skipping agent."));
      break;
    }

    const promptContext: PromptContext = {
      cycles: state.cycles,
      failedApproaches: state.failedApproaches,
      ideas: state.ideas,
      stalledCycles: state.stalledCycles,
      consecutiveFoundationKeeps: state.consecutiveFoundationKeeps,
      reviewSummary: state.reviewSummaries.at(-1) ?? null,
      bestPerf: state.bestResult ?? benchResultToCheckpoint(bestPerf, 0, state.bestCommitHash),
    };

    const agentRun = await spawnAgent(effortFile, plan, originalBaseline, currentCode, cycle, history, model, agent, promptContext, effortSpec);
    const agentReport = parseAgentReport(agentRun.stdout);

    let changedFiles = await listChangedFiles();
    if (changedFiles.length === 0) {
      const decisionReason = "no source changes; skipped sync and benchmark";
      console.log(c("1;33", `  \u26A0 NO-OP: ${decisionReason}`));

      state.failedApproaches = mergeUniqueEntries(
        state.failedApproaches,
        [`${agentReport.description} — ${decisionReason}`],
        FAILED_APPROACH_LIMIT,
      );
      state.ideas = mergeUniqueEntries(state.ideas, agentReport.nextIdeas, IDEA_LIMIT);
      state.stalledCycles++;
      state.consecutiveFoundationKeeps = 0;
      state.lastCycle = cycle;

      const cycleRecord: CycleRecord = {
        cycle,
        timestamp: new Date().toISOString(),
        description: agentReport.description,
        selfAnalysis: agentReport.selfAnalysis,
        nextIdeas: agentReport.nextIdeas,
        stepKind: agentReport.stepKind,
        changedFiles: [],
        categoryTags: classifyApproachTags(agentReport.description, []),
        tokPerSec: null,
        tokPerSecSamples: [],
        bandwidthUtil: null,
        bandwidthSamples: [],
        correct: false,
        improved: false,
        broken: false,
        kept: false,
        foundationKeep: false,
        decisionReason,
        outputText: "",
        commitHash: null,
      };
      state.cycles.push(cycleRecord);

      if (state.cycles.length % SELF_REVIEW_EVERY === 0) {
        const review = buildSelfReview(state);
        if (review) state.reviewSummaries = [...state.reviewSummaries.slice(-(REVIEW_SUMMARY_LIMIT - 1)), review];
      }

      await saveLoopState(state);
      history = buildHistoryFromCycles(state.cycles);

      const logEntry: LogEntry = {
        cycle,
        effort,
        tokPerSec: null,
        tokPerSecSamples: [],
        bandwidthUtil: null,
        bandwidthSamples: [],
        correct: false,
        improved: false,
        broken: false,
        kept: false,
        foundationKeep: false,
        decisionReason,
        description: agentReport.description,
        stepKind: agentReport.stepKind,
        changedFiles: [],
        outputText: "",
        commitHash: null,
        timestamp: new Date().toISOString(),
      };
      const logPath = logPathForEffort(effort);
      await writeFile(logPath, JSON.stringify(logEntry) + "\n", { flag: "a" });
      console.log(c("2", `  stall=${state.stalledCycles} best=${bestTokPerSec.toFixed(2)} current=${currentCode.tokPerSec?.toFixed(2) ?? "?"}`));
      continue;
    }

    // Sync and benchmark — with up to 2 fix-up retries if build fails
    console.log(c("2", "  Syncing changes..."));
    await rsyncToRemote();
    let result = await buildAndBench(modelTarget, effortSpec);

    const MAX_FIX_RETRIES = 2;
    for (let fix = 0; fix < MAX_FIX_RETRIES && !result.buildOk; fix++) {
      console.log(c("1;33", `  \u26A0 Build failed — sending errors to agent for fix (retry ${fix + 1}/${MAX_FIX_RETRIES})`));
      const fixPrompt = `The build FAILED after your changes. Fix the errors and make it compile.

## Build errors:
\`\`\`
${result.buildOutput.slice(-2000)}
\`\`\`

## Rules:
- Fix ONLY the build errors. Do not add new features.
- The code must compile: zig build -Doptimize=ReleaseFast must succeed on the remote node.
- Do not use sub-agents, delegation, spawn_agent, or wait_agent.
- Re-read the file right before patching it; do not patch against stale context.
- rsync to remote: rsync -avz --checksum --delete -e "ssh -p ${ZINC_PORT} -o StrictHostKeyChecking=no" --exclude .zig-cache --exclude zig-out --exclude node_modules --exclude .git --exclude .perf_optimize --exclude .zinc_optimize --exclude site --exclude .DS_Store ${REPO_ROOT}/ ${ZINC_USER}@${ZINC_HOST}:${REMOTE_DIR}/
- Build on remote: ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR} && zig build -Doptimize=ReleaseFast 2>&1"
- Shader compilation: ssh -p ${ZINC_PORT} ${ZINC_USER}@${ZINC_HOST} "cd ${REMOTE_DIR}/src/shaders && for f in *.comp; do glslc --target-env=vulkan1.3 -fshader-stage=compute \\$f -o \\$\{f%.comp}.spv 2>&1; done"`;

      if (agent === "codex") {
        await runCommand("codex", codexExecArgs(fixPrompt), {
          cwd: REPO_ROOT, timeout: 600_000, streamOutput: true,
          stdoutLineFormatter: (line) => formatCodexStreamLine(line),
        });
      } else {
        const fixState: ClaudeStreamState = {
          currentToolName: null, currentBlockIsToolUse: false,
          inputJsonBuffer: "", inTextBlock: false, sawTextDeltaInCurrentMessage: false,
        };
        await runCommand("claude", [
          "-p", "--verbose", "--output-format", "stream-json", "--include-partial-messages",
          `--disallowed-tools=${[...BLOCKED_GIT_OPS, ...BLOCKED_FILE_OPS].join(",")}`,
          "--permission-mode", "bypassPermissions",
          "--effort", CLAUDE_EFFORT,
          fixPrompt,
        ], {
          cwd: REPO_ROOT, timeout: 600_000, streamOutput: true,
          stdoutLineFormatter: (line) => formatClaudeStreamLine(line, fixState),
        });
      }

      console.log(c("2", "  Re-syncing after fix..."));
      await rsyncToRemote();
      result = await buildAndBench(modelTarget, effortSpec);
    }

    changedFiles = await listChangedFiles();
    const categoryTags = classifyApproachTags(agentReport.description, changedFiles);
    const improved = isMaterialImprovement(result, bestPerf);
    const foundationCandidate = shouldKeepFoundationStep(
      result,
      bestPerf,
      state.stalledCycles,
      state.consecutiveFoundationKeeps,
      agentReport,
      changedFiles,
    );

    let coherenceError: string | null = null;
    let coherenceSweep: CoherenceSweep | null = null;
    if (result.buildOk && result.correct && (improved || foundationCandidate)) {
      console.log(c("2", "  Checking all models for coherence..."));
      coherenceSweep = await runCoherenceSweep();
      coherenceError = summarizeCoherenceRegression(coherenceSweep, acceptedCoherence.failureIds);
      if (coherenceError) {
        console.log(c("1;31", `  ${coherenceError}`));
      } else if (coherenceSweep.failures.length > 0) {
        console.log(c("2", `  Coherence unchanged vs accepted baseline (${coherenceSweep.failures.length} known failing case(s)).`));
      } else if (acceptedCoherence.failures.length > 0) {
        console.log(c("1;36", "  Coherence improved: all accepted-baseline failures cleared."));
      }
    }

    const correct = result.correct && coherenceError == null;
    const broken = !result.buildOk || !correct;
    const threshold = improvementThreshold(bestPerf.tokPerSec);

    const deltaVsBest = result.tokPerSec != null && (bestPerf.tokPerSec ?? 0) > 0
      ? ((result.tokPerSec - (bestPerf.tokPerSec ?? 0)) / (bestPerf.tokPerSec ?? 1) * 100).toFixed(2)
      : "?";

    let kept = false;
    let foundationKeep = false;
    let decisionReason = "";
    let commitHash: string | null = null;

    if (broken) {
      const failureReason = coherenceError ?? result.error ?? "incorrect output";
      console.log(c("1;31", `  \u274C BROKEN: ${failureReason}`));
      console.log(c("1;31", `     Output: "${result.outputText?.slice(0, 80)}"`));
      decisionReason = failureReason;
      state.failedApproaches = mergeUniqueEntries(
        state.failedApproaches,
        [`${agentReport.description} — ${decisionReason}`],
        FAILED_APPROACH_LIMIT,
      );
      state.stalledCycles++;
      state.consecutiveFoundationKeeps = 0;
      await revertAgentChanges();
    } else if (improved) {
      kept = true;
      console.log(c("1;32", `  \u2705 IMPROVED: ${summarizeBenchMetric(result.tokPerSec, result.tokPerSecSamples, "tok/s")} (${effortSpec.primaryMetricLabel}, +${deltaVsBest}%, threshold +${threshold.toFixed(2)} tok/s vs best checkpoint)`));
      currentCode = result;
      bestPerf = result;
      bestTokPerSec = result.tokPerSec!;
      if (coherenceSweep) acceptedCoherence = coherenceSweep;
      state.stalledCycles = 0;
      state.consecutiveFoundationKeeps = 0;
      decisionReason = `improved by ${deltaVsBest}% vs best checkpoint`;

      await runCommand("git", ["add", "src/"], { cwd: REPO_ROOT });
      await runCommand("git", ["commit", "-m", `perf(effort-${effort}): cycle ${cycle} \u2014 ${result.tokPerSec?.toFixed(2)} ${effortSpec.primaryMetricLabel} (+${deltaVsBest}%)`], { cwd: REPO_ROOT });
      commitHash = (await runCommand("git", ["rev-parse", "HEAD"], { cwd: REPO_ROOT })).stdout.trim() || headCommit;
      state.bestTokPerSec = bestTokPerSec;
      state.bestCycle = cycle;
      state.bestCommitHash = commitHash;
      state.bestResult = benchResultToCheckpoint(result, cycle, commitHash);
      console.log(c("2", "  Committed."));
    } else if (foundationCandidate) {
      kept = true;
      foundationKeep = true;
      currentCode = result;
      if (coherenceSweep) acceptedCoherence = coherenceSweep;
      state.stalledCycles++;
      state.consecutiveFoundationKeeps++;
      decisionReason = `kept enablement step within ${FOUNDATION_KEEP_MAX_DROP_TPS.toFixed(2)} tok/s of best checkpoint`;
      console.log(c("1;36", `  \u2248 FOUNDATION KEEP: ${summarizeBenchMetric(result.tokPerSec, result.tokPerSecSamples, "tok/s")} (${effortSpec.primaryMetricLabel}, ${deltaVsBest}% vs best checkpoint)`));
      await runCommand("git", ["add", "src/"], { cwd: REPO_ROOT });
      await runCommand("git", ["commit", "-m", `perf(effort-${effort}): cycle ${cycle} foundation \u2014 ${trunc(agentReport.description, 72)}`], { cwd: REPO_ROOT });
      commitHash = (await runCommand("git", ["rev-parse", "HEAD"], { cwd: REPO_ROOT })).stdout.trim() || headCommit;
      console.log(c("2", "  Committed foundation step."));
    } else {
      decisionReason = `no improvement (needed +${threshold.toFixed(2)} tok/s vs best checkpoint)`;
      console.log(c("1;33", `  \u26A0 NO IMPROVEMENT: ${summarizeBenchMetric(result.tokPerSec, result.tokPerSecSamples, "tok/s")} (${effortSpec.primaryMetricLabel}, ${deltaVsBest}%, needed +${threshold.toFixed(2)} tok/s vs best checkpoint)`));
      state.failedApproaches = mergeUniqueEntries(
        state.failedApproaches,
        [`${agentReport.description} — ${decisionReason}`],
        FAILED_APPROACH_LIMIT,
      );
      state.stalledCycles++;
      state.consecutiveFoundationKeeps = 0;
      await revertAgentChanges();
    }

    state.ideas = mergeUniqueEntries(state.ideas, agentReport.nextIdeas, IDEA_LIMIT);
    state.lastCycle = cycle;

    const cycleRecord: CycleRecord = {
      cycle,
      timestamp: new Date().toISOString(),
      description: agentReport.description,
      selfAnalysis: agentReport.selfAnalysis,
      nextIdeas: agentReport.nextIdeas,
      stepKind: agentReport.stepKind,
      changedFiles,
      categoryTags,
      tokPerSec: result.tokPerSec,
      tokPerSecSamples: result.tokPerSecSamples,
      bandwidthUtil: result.bandwidthUtil,
      bandwidthSamples: result.bandwidthSamples,
      correct,
      improved,
      broken,
      kept,
      foundationKeep,
      decisionReason,
      outputText: result.outputText?.slice(0, 200),
      commitHash,
    };
    state.cycles.push(cycleRecord);

    if (state.cycles.length % SELF_REVIEW_EVERY === 0) {
      const review = buildSelfReview(state);
      if (review) {
        state.reviewSummaries = [...state.reviewSummaries.slice(-(REVIEW_SUMMARY_LIMIT - 1)), review];
        console.log(c("1;35", `  \uD83D\uDD0D Self-review (${state.cycles.length} cycles)`));
        console.log(c("2", review));
      }
    }

    await saveLoopState(state);
    history = buildHistoryFromCycles(state.cycles);

    // Log cycle result
    const logEntry: LogEntry = {
      cycle,
      effort,
      tokPerSec: result.tokPerSec,
      tokPerSecSamples: result.tokPerSecSamples,
      bandwidthUtil: result.bandwidthUtil,
      bandwidthSamples: result.bandwidthSamples,
      correct,
      improved,
      broken,
      kept,
      foundationKeep,
      decisionReason,
      description: agentReport.description,
      stepKind: agentReport.stepKind,
      changedFiles: changedFiles.slice(0, MAX_CHANGED_FILES_IN_PROMPT),
      outputText: result.outputText?.slice(0, 200),
      commitHash,
      timestamp: new Date().toISOString(),
    };
    const logPath = logPathForEffort(effort);
    await writeFile(logPath, JSON.stringify(logEntry) + "\n", { flag: "a" });
    console.log(c("2", `  stall=${state.stalledCycles} best=${bestTokPerSec.toFixed(2)} current=${currentCode.tokPerSec?.toFixed(2) ?? "?"}`));
  }

  // Summary
  console.log(c("1;37", `\n${"═".repeat(58)}`));
  console.log(c("1;37", `  Effort ${effort} complete.`));
  console.log(c("1;37", `  Baseline (${effortSpec.primaryMetricLabel}): ${originalBaseline.tokPerSec?.toFixed(2)} tok/s`));
  console.log(c("1;37", `  Best (${effortSpec.primaryMetricLabel}):     ${bestTokPerSec.toFixed(2)} tok/s`));
  console.log(c("1;37", `  Current (${effortSpec.primaryMetricLabel}):  ${currentCode.tokPerSec?.toFixed(2) ?? "?"} tok/s`));
  if (bestTokPerSec > (originalBaseline.tokPerSec ?? 0)) {
    const gain = ((bestTokPerSec - (originalBaseline.tokPerSec ?? 0)) / (originalBaseline.tokPerSec ?? 1) * 100).toFixed(1);
    console.log(c("1;32", `  Gain:     +${gain}%`));
  }
  console.log(c("1;37", `  Stall:    ${state.stalledCycles} cycles`));
  console.log(c("1;37", `  State:    ${statePathForEffort(effort)}`));
  console.log(c("1;37", `${"═".repeat(58)}\n`));
}

// Only run main when executed directly, not when imported by tests
const isMainModule = typeof Bun !== "undefined"
  ? Bun.main === import.meta.path
  : !process.argv[1]?.includes(".test.");

if (isMainModule) {
  main().catch((e) => {
    console.error(c("1;31", `Fatal: ${e}`));
    process.exit(1);
  });
}
