#!/usr/bin/env bun
/**
 * ZINC remote GPU optimization loop.
 *
 * This is the vendor-neutral harness for bringing up and tuning a new GPU
 * target. It syncs the local tree to a remote node, verifies the basic ZINC
 * commands, benchmarks ZINC and an optional llama.cpp Vulkan baseline on the
 * same GGUF, then lets one agent cycle make a bounded local source change.
 *
 * Examples:
 *   bun loops/optimize_gpu.ts --agent codex --cycles 20
 *   bun loops/optimize_gpu.ts --model qwen35-9b-q4k-m --agent codex --cycles 20
 *   bun loops/optimize_gpu.ts --model-id qwen38-27b-q4k-m --metric prefill --resume
 *   bun loops/optimize_gpu.ts --model-path /models/foo.gguf --skip-llama --dry-run
 */

import { spawn } from "node:child_process";
import { existsSync, readdirSync, statSync } from "node:fs";
import { mkdir, readFile, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import {
  codexExecArgs,
  formatClaudeStreamLine,
  formatCodexStreamLine,
  type ClaudeStreamState,
} from "./optimize_perf";
import { formatElapsed } from "./optimize_llm_tps";
import { parsePrefillTokPerSec, parsePrefillTokenCount } from "./optimize_zinc";

const REPO_ROOT = resolve(import.meta.dir, "..");
const RESULTS_DIR = join(REPO_ROOT, ".gpu_optimize");
const CLAUDE_MODEL = process.env.ZINC_CLAUDE_MODEL ?? "claude-opus-4-7[1m]";
const CLAUDE_EFFORT = process.env.ZINC_CLAUDE_EFFORT ?? "max";
const BLOCKED_GIT_OPS = [
  "Bash(git checkout:*)",
  "Bash(git revert:*)",
  "Bash(git restore:*)",
  "Bash(git reset:*)",
  "Bash(git stash:*)",
  "Bash(git clean:*)",
  "Bash(git push:*)",
  "Bash(git commit:*)",
];
const BLOCKED_FILE_OPS = [
  "Edit(.env)",
  "Write(.env)",
  "Edit(AGENTS.md)",
  "Write(AGENTS.md)",
  "Edit(CLAUDE.md)",
  "Write(CLAUDE.md)",
];

type AgentKind = "codex" | "claude";
type MetricKind = "decode" | "prefill";
type PromptMode = "raw" | "chat";

type ModelPreset = {
  key: string;
  label: string;
  modelId: string;
  promptMode: PromptMode;
  prompt: string;
  maxTokens: number;
  expect: string[];
};

const QWEN3_8B_COMPARISON_PROMPT = [
  "Benchmark context only. alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.",
  "Benchmark context only. alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.",
  "Benchmark context only. alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.",
  "Benchmark context only. alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.",
  "Important fact: Paris is the capital of France.",
  "Based on the reference above, the capital of France is",
].join(" ");

const MODEL_PRESETS: Record<string, ModelPreset> = {
  "qwen35-9b-q4k-m": {
    key: "qwen35-9b-q4k-m",
    label: "Qwen3.5 9B Q4_K_M",
    modelId: "qwen35-9b-q4k-m",
    promptMode: "raw",
    prompt: QWEN3_8B_COMPARISON_PROMPT,
    maxTokens: 128,
    expect: ["Paris"],
  },
  "qwen38-27b-q4k-m": {
    key: "qwen38-27b-q4k-m",
    label: "Qwen3.8 27B Dense Q4_K_M",
    modelId: "qwen38-27b-q4k-m",
    promptMode: "chat",
    prompt: "What is the capital of France? Answer in one word.",
    maxTokens: 32,
    expect: ["Paris"],
  },
  "qwen36-35b-a3b-q4k-xl": {
    key: "qwen36-35b-a3b-q4k-xl",
    label: "Qwen3.6 35B A3B Q4_K_XL",
    modelId: "qwen36-35b-a3b-q4k-xl",
    promptMode: "chat",
    prompt: "What is the capital of France? Answer in one word.",
    maxTokens: 32,
    expect: ["Paris"],
  },
  "gemma4-26b-a4b-q4k-m": {
    key: "gemma4-26b-a4b-q4k-m",
    label: "Gemma 4 26B-A4B MoE Q4_K_M",
    modelId: "gemma4-26b-a4b-q4k-m",
    promptMode: "chat",
    prompt: "What is the capital of France? Answer in one word.",
    maxTokens: 48,
    expect: ["Paris"],
  },
  "gemma4-31b-q4k-m": {
    key: "gemma4-31b-q4k-m",
    label: "Gemma 4 31B Q4_K_M",
    modelId: "gemma4-31b-q4k-m",
    promptMode: "chat",
    prompt: "What is the capital of France? Answer in one word.",
    maxTokens: 48,
    expect: ["Paris"],
  },
};

const DEFAULT_MODEL = "gemma4-26b-a4b-q4k-m";

type ModelTarget = {
  key: string;
  label: string;
  modelId: string | null;
  modelPath: string;
  promptMode: PromptMode;
  prompt: string;
  maxTokens: number;
  expect: string[];
};

export type LoopOptions = {
  agent: AgentKind;
  cycles: number;
  dryRun: boolean;
  resume: boolean;
  resumeDir: string | null;
  runId: string | null;
  host: string;
  port: number;
  user: string;
  sshPasswordEnvVar: string | null;
  sshPasswordFile: string | null;
  remoteDir: string;
  remoteHome: string;
  xdgCacheHome: string;
  remoteEnv: string;
  remoteLibcConf: string | null;
  model: string | null;
  modelId: string | null;
  modelPath: string | null;
  prompt: string | null;
  promptMode: PromptMode | null;
  maxTokens: number | null;
  contextTokens: number | null;
  metric: MetricKind;
  targetTps: number | null;
  samples: number;
  skipLlama: boolean;
  continueAfterLlama: boolean;
  llamaDir: string;
  llamaBench: string | null;
  llamaVulkanDevice: number;
  llamaPromptTokens: number;
  llamaDecodeTokens: number;
  allowDirty: boolean;
  autoRevert: boolean;
  maxStallCycles: number;
};

export type CommandResult = {
  exitCode: number;
  signal: NodeJS.Signals | null;
  stdout: string;
  stderr: string;
};

type BenchmarkMetrics = {
  decodeTokPerSec: number | null;
  prefillTokPerSec: number | null;
  promptTokens: number | null;
  outputText: string;
  coherent: boolean;
};

type BenchmarkSummary = {
  metric: MetricKind;
  value: number | null;
  samples: number[];
  decodeSamples: number[];
  prefillSamples: number[];
  promptTokenSamples: number[];
  outputText: string;
  coherent: boolean;
};

type LlamaSummary = {
  decodeTokPerSec: number | null;
  prefillTokPerSec: number | null;
  promptTokens: number | null;
  raw: string;
  source?: "llama-cli" | "llama-completion" | "llama-bench";
};

type CycleRecord = {
  cycle: number;
  timestamp: string;
  changedFiles: string[];
  before: BenchmarkSummary;
  after: BenchmarkSummary | null;
  llama: LlamaSummary | null;
  kept: boolean;
  improved: boolean;
  reason: string;
};

type RunState = {
  startedAt: string;
  updatedAt: string;
  runId: string;
  options: {
    modelKey: string;
    modelPath: string;
    metric: MetricKind;
    host: string;
    user: string;
    port: number;
    remoteDir: string;
  };
  best: BenchmarkSummary | null;
  llamaBaseline: LlamaSummary | null;
  cycles: CycleRecord[];
  failedApproaches: string[];
};

export function stateTargetMismatchReason(
  state: Pick<RunState, "options"> | null,
  target: Pick<ModelTarget, "key">,
  opts: Pick<LoopOptions, "metric">,
): string | null {
  if (!state) return null;
  if (state.options.modelKey !== target.key) {
    return `run state target mismatch: run was created for ${state.options.modelKey}, but selected target is ${target.key}. Use --model ${state.options.modelKey} to resume that run, or choose a new --run-id for ${target.key}.`;
  }
  if (state.options.metric !== opts.metric) {
    return `run state metric mismatch: run was created for ${state.options.metric}, but selected metric is ${opts.metric}. Use --metric ${state.options.metric} to resume that run, or choose a new --run-id.`;
  }
  return null;
}

function loadEnv(): Record<string, string> {
  const envPath = join(REPO_ROOT, ".env");
  const vars: Record<string, string> = {};
  if (!existsSync(envPath)) return vars;
  const content = require("fs").readFileSync(envPath, "utf8") as string;
  for (const line of content.split("\n")) {
    const m = line.match(/^\s*(?:export\s+)?([A-Z0-9_]+)\s*=\s*(.+?)\s*$/);
    if (!m) continue;
    vars[m[1]] = m[2].replace(/^['"]|['"]$/g, "");
  }
  return vars;
}

const ENV = loadEnv();

function defaultHomeForUser(user: string): string {
  return user === "root" ? "/root" : `/home/${user}`;
}

function isValueToken(value: string | undefined): value is string {
  return value != null && !value.startsWith("--");
}

function envValue(envMap: Record<string, string>, fileEnv: Record<string, string>, ...keys: string[]): string | undefined {
  for (const key of keys) {
    const value = envMap[key] ?? fileEnv[key];
    if (value != null && value !== "") return value;
  }
  return undefined;
}

function validatePasswordEnvName(name: string): void {
  if (!/^[A-Za-z_][A-Za-z0-9_]*$/.test(name)) {
    throw new Error(`Invalid SSH password environment variable name: ${name}`);
  }
}

function hasSshPasswordAuth(opts: Pick<LoopOptions, "sshPasswordEnvVar" | "sshPasswordFile">): boolean {
  return Boolean(opts.sshPasswordEnvVar || opts.sshPasswordFile);
}

export function buildSshOptions(opts: Pick<LoopOptions, "sshPasswordEnvVar" | "sshPasswordFile">): string[] {
  if (!hasSshPasswordAuth(opts)) {
    return ["-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=no"];
  }
  return [
    "-o", "BatchMode=no",
    "-o", "NumberOfPasswordPrompts=1",
    "-o", "PreferredAuthentications=publickey,password,keyboard-interactive",
    "-o", "ConnectTimeout=15",
    "-o", "StrictHostKeyChecking=no",
  ];
}

function sshAskpassScript(opts: Pick<LoopOptions, "sshPasswordEnvVar" | "sshPasswordFile">): string | null {
  if (opts.sshPasswordFile) return `#!/bin/sh\ncat ${q(opts.sshPasswordFile)}\n`;
  if (!opts.sshPasswordEnvVar) return null;
  validatePasswordEnvName(opts.sshPasswordEnvVar);
  return `#!/bin/sh\nif [ -z "\${${opts.sshPasswordEnvVar}+set}" ]; then exit 1; fi\nprintf '%s\\n' "\${${opts.sshPasswordEnvVar}}"\n`;
}

function resolveSshPasswordAuth(envMap: Record<string, string>, fileEnv: Record<string, string>): Pick<LoopOptions, "sshPasswordEnvVar" | "sshPasswordFile"> {
  for (const key of ["ZINC_GPU_SSH_PASSWORD", "ZINC_INTEL_SSH_PASSWORD", "ZINC_SSH_PASSWORD"]) {
    if (envMap[key]) return { sshPasswordEnvVar: key, sshPasswordFile: null };
  }
  for (const key of ["ZINC_GPU_SSH_PASSWORD", "ZINC_INTEL_SSH_PASSWORD", "ZINC_SSH_PASSWORD"]) {
    if (fileEnv[key]) {
      process.env[key] = fileEnv[key];
      return { sshPasswordEnvVar: key, sshPasswordFile: null };
    }
  }
  const sshPasswordEnvVar = envValue(
    envMap,
    fileEnv,
    "ZINC_GPU_SSH_PASSWORD_ENV",
    "ZINC_INTEL_SSH_PASSWORD_ENV",
    "ZINC_SSH_PASSWORD_ENV",
  );
  if (sshPasswordEnvVar) {
    validatePasswordEnvName(sshPasswordEnvVar);
    return { sshPasswordEnvVar, sshPasswordFile: null };
  }
  const sshPasswordFile = envValue(
    envMap,
    fileEnv,
    "ZINC_GPU_SSH_PASSWORD_FILE",
    "ZINC_INTEL_SSH_PASSWORD_FILE",
    "ZINC_SSH_PASSWORD_FILE",
  );
  return { sshPasswordEnvVar: null, sshPasswordFile: sshPasswordFile ?? null };
}

export function parseArgsFrom(argv: string[], envMap: Record<string, string> = process.env): LoopOptions {
  const fileEnv = envMap === process.env ? ENV : {};
  const sshPasswordAuth = resolveSshPasswordAuth(envMap, fileEnv);
  const user = envMap.ZINC_GPU_USER ?? fileEnv.ZINC_GPU_USER ?? envMap.ZINC_INTEL_USER ?? fileEnv.ZINC_INTEL_USER ?? envMap.ZINC_USER ?? fileEnv.ZINC_USER ?? "root";
  const envRemoteHome = envMap.ZINC_GPU_REMOTE_HOME ?? fileEnv.ZINC_GPU_REMOTE_HOME ?? envMap.ZINC_INTEL_REMOTE_HOME ?? fileEnv.ZINC_INTEL_REMOTE_HOME ?? envMap.ZINC_REMOTE_HOME ?? fileEnv.ZINC_REMOTE_HOME;
  const envRemoteDir = envMap.ZINC_GPU_REMOTE_DIR ?? fileEnv.ZINC_GPU_REMOTE_DIR ?? envMap.ZINC_INTEL_WORKDIR ?? fileEnv.ZINC_INTEL_WORKDIR ?? envMap.ZINC_INTEL_REMOTE_DIR ?? fileEnv.ZINC_INTEL_REMOTE_DIR ?? envMap.ZINC_REMOTE_DIR ?? fileEnv.ZINC_REMOTE_DIR;
  const envXdgCacheHome = envMap.XDG_CACHE_HOME ?? envMap.ZINC_GPU_XDG_CACHE_HOME ?? fileEnv.ZINC_GPU_XDG_CACHE_HOME ?? envMap.ZINC_INTEL_XDG_CACHE_HOME ?? fileEnv.ZINC_INTEL_XDG_CACHE_HOME ?? envMap.ZINC_REMOTE_XDG_CACHE_HOME ?? fileEnv.ZINC_REMOTE_XDG_CACHE_HOME;
  const envLlamaDir = envMap.ZINC_GPU_LLAMA_CPP_DIR ?? fileEnv.ZINC_GPU_LLAMA_CPP_DIR ?? envMap.ZINC_INTEL_LLAMA_CPP_DIR ?? fileEnv.ZINC_INTEL_LLAMA_CPP_DIR ?? envMap.ZINC_LLAMA_CPP_DIR ?? fileEnv.ZINC_LLAMA_CPP_DIR;
  const remoteHome = envRemoteHome ?? defaultHomeForUser(user);
  let remoteHomeExplicit = envRemoteHome != null;
  let remoteDirExplicit = envRemoteDir != null;
  let xdgCacheHomeExplicit = envXdgCacheHome != null;
  let llamaDirExplicit = envLlamaDir != null;
  const opts: LoopOptions = {
    agent: "codex",
    cycles: 20,
    dryRun: false,
    resume: false,
    resumeDir: null,
    runId: null,
    host: envMap.ZINC_GPU_HOST ?? fileEnv.ZINC_GPU_HOST ?? envMap.ZINC_INTEL_HOST ?? fileEnv.ZINC_INTEL_HOST ?? envMap.ZINC_HOST ?? fileEnv.ZINC_HOST ?? "127.0.0.1",
    port: Number(envMap.ZINC_GPU_PORT ?? fileEnv.ZINC_GPU_PORT ?? envMap.ZINC_INTEL_PORT ?? fileEnv.ZINC_INTEL_PORT ?? envMap.ZINC_PORT ?? fileEnv.ZINC_PORT ?? "22"),
    user,
    ...sshPasswordAuth,
    remoteDir: envRemoteDir ?? `${remoteHome}/zinc-gpu-loop`,
    remoteHome,
    xdgCacheHome: envXdgCacheHome ?? `${remoteHome}/.cache`,
    remoteEnv: envMap.ZINC_REMOTE_ENV ?? fileEnv.ZINC_REMOTE_ENV ?? "",
    remoteLibcConf: envMap.ZINC_GPU_REMOTE_LIBC_CONF ?? fileEnv.ZINC_GPU_REMOTE_LIBC_CONF ?? envMap.ZINC_INTEL_REMOTE_LIBC_CONF ?? fileEnv.ZINC_INTEL_REMOTE_LIBC_CONF ?? envMap.ZINC_REMOTE_LIBC_CONF ?? fileEnv.ZINC_REMOTE_LIBC_CONF ?? null,
    model: envMap.ZINC_GPU_MODEL ?? fileEnv.ZINC_GPU_MODEL ?? envMap.ZINC_INTEL_MODEL ?? fileEnv.ZINC_INTEL_MODEL ?? envMap.ZINC_MODEL ?? fileEnv.ZINC_MODEL ?? DEFAULT_MODEL,
    modelId: null,
    modelPath: null,
    prompt: null,
    promptMode: null,
    maxTokens: null,
    contextTokens: Number(envMap.ZINC_GPU_CONTEXT ?? fileEnv.ZINC_GPU_CONTEXT ?? envMap.ZINC_INTEL_CONTEXT ?? fileEnv.ZINC_INTEL_CONTEXT ?? envMap.ZINC_CONTEXT ?? fileEnv.ZINC_CONTEXT ?? "0") || null,
    metric: "decode",
    targetTps: null,
    samples: 3,
    skipLlama: false,
    continueAfterLlama: false,
    llamaDir: envLlamaDir ?? `${remoteHome}/llama.cpp`,
    llamaBench: envMap.ZINC_LLAMA_BENCH ?? fileEnv.ZINC_LLAMA_BENCH ?? null,
    llamaVulkanDevice: Number(
      envMap.ZINC_LLAMA_VULKAN_DEVICE_INDEX ??
      fileEnv.ZINC_LLAMA_VULKAN_DEVICE_INDEX ??
      envMap.ZINC_VULKAN_DEVICE_INDEX ??
      fileEnv.ZINC_VULKAN_DEVICE_INDEX ??
      envMap.ZINC_RDNA_DEVICE_INDEX ??
      fileEnv.ZINC_RDNA_DEVICE_INDEX ??
      "1",
    ),
    llamaPromptTokens: 128,
    llamaDecodeTokens: 128,
    allowDirty: false,
    autoRevert: true,
    maxStallCycles: Number(envMap.ZINC_GPU_MAX_STALL_CYCLES ?? fileEnv.ZINC_GPU_MAX_STALL_CYCLES ?? envMap.ZINC_INTEL_MAX_STALL_CYCLES ?? fileEnv.ZINC_INTEL_MAX_STALL_CYCLES ?? envMap.ZINC_MAX_STALL_CYCLES ?? fileEnv.ZINC_MAX_STALL_CYCLES ?? "50"),
  };

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === "--agent" && argv[i + 1]) opts.agent = argv[++i] as AgentKind;
    else if (arg === "--cycles" && argv[i + 1]) opts.cycles = Number(argv[++i]);
    else if (arg === "--dry-run") opts.dryRun = true;
    else if (arg === "--resume") {
      opts.resume = true;
      if (isValueToken(argv[i + 1])) opts.resumeDir = argv[++i];
    } else if (arg === "--run-id" && argv[i + 1]) opts.runId = argv[++i];
    else if (arg === "--host" && argv[i + 1]) opts.host = argv[++i];
    else if (arg === "--port" && argv[i + 1]) opts.port = Number(argv[++i]);
    else if (arg === "--user" && argv[i + 1]) opts.user = argv[++i];
    else if (arg === "--remote-dir" && argv[i + 1]) {
      opts.remoteDir = argv[++i];
      remoteDirExplicit = true;
    } else if (arg === "--remote-home" && argv[i + 1]) {
      opts.remoteHome = argv[++i];
      remoteHomeExplicit = true;
    } else if (arg === "--xdg-cache-home" && argv[i + 1]) {
      opts.xdgCacheHome = argv[++i];
      xdgCacheHomeExplicit = true;
    } else if (arg === "--remote-env" && argv[i + 1]) opts.remoteEnv = argv[++i];
    else if (arg === "--remote-libc-conf" && argv[i + 1]) opts.remoteLibcConf = argv[++i];
    else if (arg === "--model" && argv[i + 1]) opts.model = argv[++i];
    else if (arg === "--model-id" && argv[i + 1]) opts.modelId = argv[++i];
    else if (arg === "--model-path" && argv[i + 1]) opts.modelPath = argv[++i];
    else if (arg === "--prompt" && argv[i + 1]) opts.prompt = argv[++i];
    else if (arg === "--chat") opts.promptMode = "chat";
    else if (arg === "--raw") opts.promptMode = "raw";
    else if (arg === "--max-tokens" && argv[i + 1]) opts.maxTokens = Number(argv[++i]);
    else if (arg === "--context" && argv[i + 1]) opts.contextTokens = Number(argv[++i]);
    else if (arg === "--metric" && argv[i + 1]) opts.metric = argv[++i] as MetricKind;
    else if (arg === "--target" && argv[i + 1]) opts.targetTps = Number(argv[++i]);
    else if (arg === "--samples" && argv[i + 1]) opts.samples = Number(argv[++i]);
    else if (arg === "--skip-llama") opts.skipLlama = true;
    else if (arg === "--continue-after-llama") opts.continueAfterLlama = true;
    else if (arg === "--llama-dir" && argv[i + 1]) {
      opts.llamaDir = argv[++i];
      llamaDirExplicit = true;
    } else if (arg === "--llama-bench" && argv[i + 1]) opts.llamaBench = argv[++i];
    else if (arg === "--llama-prompt-tokens" && argv[i + 1]) opts.llamaPromptTokens = Number(argv[++i]);
    else if (arg === "--llama-decode-tokens" && argv[i + 1]) opts.llamaDecodeTokens = Number(argv[++i]);
    else if (arg === "--allow-dirty") opts.allowDirty = true;
    else if (arg === "--no-revert") opts.autoRevert = false;
    else if (arg === "--max-stall-cycles" && argv[i + 1]) opts.maxStallCycles = Number(argv[++i]);
    else if (arg === "--help" || arg === "-h") {
      printUsage();
      process.exit(0);
    } else {
      throw new Error(`Unknown argument: ${arg}`);
    }
  }

  if (opts.agent !== "codex" && opts.agent !== "claude") throw new Error("--agent must be codex or claude");
  if (opts.metric !== "decode" && opts.metric !== "prefill") throw new Error("--metric must be decode or prefill");
  if (!Number.isFinite(opts.cycles) || opts.cycles < 0) throw new Error("--cycles must be >= 0");
  if (opts.contextTokens != null && (!Number.isFinite(opts.contextTokens) || opts.contextTokens < 1)) throw new Error("--context must be >= 1");
  if (!Number.isFinite(opts.maxStallCycles) || opts.maxStallCycles < 0) throw new Error("--max-stall-cycles must be >= 0");
  if (!Number.isFinite(opts.samples) || opts.samples < 1) throw new Error("--samples must be >= 1");
  if (!remoteHomeExplicit) opts.remoteHome = defaultHomeForUser(opts.user);
  if (!remoteDirExplicit) opts.remoteDir = `${opts.remoteHome}/zinc-gpu-loop`;
  if (!xdgCacheHomeExplicit) opts.xdgCacheHome = `${opts.remoteHome}/.cache`;
  if (!llamaDirExplicit) opts.llamaDir = `${opts.remoteHome}/llama.cpp`;
  return opts;
}

function printUsage(): void {
  const models = Object.keys(MODEL_PRESETS).join(", ");
  const defaultModel = MODEL_PRESETS[DEFAULT_MODEL];
  console.log(`Usage: bun loops/optimize_gpu.ts [options]\n`);
  console.log(`  --model <preset>          Managed preset (${models})`);
  console.log(`                            Defaults to ${DEFAULT_MODEL} (${defaultModel?.label ?? "selected model"})`);
  console.log("  --model-id <id>           Managed model id from ZINC catalog");
  console.log("  --model-path <path>       Remote GGUF path");
  console.log("  --context <tokens>        Context length passed to ZINC and llama.cpp baselines");
  console.log("  --agent codex|claude      Agent for optimization cycles");
  console.log("  --resume [run-dir]        Resume latest or specified .gpu_optimize run");
  console.log("  --host/--port/--user      Remote SSH target (defaults from ZINC_GPU_*, then ZINC_INTEL_*, then ZINC_*)");
  console.log("  --remote-dir <path>       Remote checkout directory");
  console.log("  --metric decode|prefill   Primary keep metric");
  console.log("                            With llama.cpp available, keep decisions attack the largest remaining decode/prefill gap");
  console.log("  --continue-after-llama    Keep optimizing after ZINC beats llama.cpp on decode and prefill");
  console.log("  --max-stall-cycles <n>    Stop after n cycles without a kept improvement (default 50, 0 disables)");
  console.log("  --skip-llama              Only benchmark ZINC");
  console.log("  --dry-run                 Preflight + baseline only");
}

export function managedCachePath(xdgCacheHome: string, modelId: string): string {
  return `${xdgCacheHome}/zinc/models/models/${modelId}/model.gguf`;
}

export function resolveModelTarget(opts: LoopOptions): ModelTarget {
  const preset = opts.model ? MODEL_PRESETS[opts.model] : undefined;
  const modelId = opts.modelId ?? preset?.modelId ?? null;
  const modelPath = opts.modelPath ?? (modelId ? managedCachePath(opts.xdgCacheHome, modelId) : null);
  if (!modelPath) throw new Error("Provide --model, --model-id, or --model-path");
  return {
    key: opts.modelPath ? "custom-path" : (opts.modelId ?? preset?.key ?? "custom-model-id"),
    label: preset?.label ?? opts.modelId ?? opts.modelPath ?? "custom",
    modelId,
    modelPath,
    promptMode: opts.promptMode ?? preset?.promptMode ?? "raw",
    prompt: opts.prompt ?? preset?.prompt ?? "The capital of France is",
    maxTokens: opts.maxTokens ?? preset?.maxTokens ?? 64,
    expect: preset?.expect ?? ["Paris"],
  };
}

function q(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

function remoteEnvPrefix(opts: LoopOptions): string {
  const parts = [`XDG_CACHE_HOME=${q(opts.xdgCacheHome)}`];
  if (opts.remoteEnv.trim()) parts.push(opts.remoteEnv.trim());
  return `env ${parts.join(" ")}`;
}

export function buildRemoteZincCommand(opts: LoopOptions, target: ModelTarget): string {
  const modelArg = target.modelId && !opts.modelPath
    ? `--model-id ${q(target.modelId)}`
    : `--model ${q(target.modelPath)}`;
  const modeArg = target.promptMode === "chat" ? "--chat" : "--raw";
  const contextArg = opts.contextTokens != null ? ` -c ${opts.contextTokens}` : "";
  return [
    `cd ${q(opts.remoteDir)}`,
    `${remoteEnvPrefix(opts)} ./zig-out/bin/zinc ${modelArg} --prompt ${q(target.prompt)} ${modeArg} -n ${target.maxTokens}${contextArg}`,
  ].join(" && ");
}

export function buildRemoteLlamaBenchCommand(opts: LoopOptions, target: ModelTarget): string {
  const bench = opts.llamaBench ?? `${opts.llamaDir}/build/bin/llama-bench`;
  return [
    `test -x ${q(bench)}`,
    `${q(bench)} -m ${q(target.modelPath)} -ngl 99 -p ${opts.llamaPromptTokens} -n ${opts.llamaDecodeTokens} -r ${Math.max(1, opts.samples)} -fa 1 -o json`,
  ].join(" && ");
}

export function buildRemoteLlamaCliCommand(opts: LoopOptions, target: ModelTarget): string {
  const cli = `${opts.llamaDir}/build/bin/llama-cli`;
  const completion = `${opts.llamaDir}/build/bin/llama-completion`;
  const contextTokens = opts.contextTokens ?? Math.max(1024, target.maxTokens + 256);
  return [
    `llama_completion=${q(completion)}`,
    `llama_cli=${q(cli)}`,
    `llama_bin=""`,
    `llama_kind=""`,
    `llama_mode=""`,
    `if [ -x "$llama_completion" ]; then llama_bin="$llama_completion"; llama_kind="llama-completion"; llama_mode="--no-conversation"; fi`,
    `if [ -z "$llama_bin" ]; then found_completion=$(command -v llama-completion || true); if [ -n "$found_completion" ]; then llama_bin="$found_completion"; llama_kind="llama-completion"; llama_mode="--no-conversation"; fi; fi`,
    `if [ -z "$llama_bin" ] && [ -x "$llama_cli" ]; then llama_bin="$llama_cli"; llama_kind="llama-cli"; llama_mode="-st"; fi`,
    `if [ ! -x "$llama_cli" ]; then llama_cli=$(command -v llama-cli || true); fi`,
    `if [ -z "$llama_bin" ] && [ -n "$llama_cli" ]; then llama_bin="$llama_cli"; llama_kind="llama-cli"; llama_mode="-st"; fi`,
    `test -n "$llama_bin"`,
    `echo "ZINC_LLAMA_SOURCE=$llama_kind" >&2`,
    `timeout 300s "$llama_bin" $llama_mode -m ${q(target.modelPath)} --device Vulkan${opts.llamaVulkanDevice} -ngl 99 -p ${q(target.prompt)} -n ${target.maxTokens} --temp 0 -fa 1 -c ${contextTokens} -b 256 -ub 128`,
  ].join(" && ");
}

function parseDecodeTokPerSec(output: string): number | null {
  const generated = output.match(/Generated\s+\d+\s+tokens\s+in\s+\d+(?:\.\d*)?\s*(?:ms|s)\s+(?:[-–—]\s+)?(\d+(?:\.\d*)?)\s*tok\/s/i);
  if (generated) return Number(generated[1]);
  const simple = output.match(/(?:decode|generation)\s*(?:tok\/s|tokens\/s|throughput)?\s*[:=]\s*(\d+(?:\.\d*)?)\s*tok\/s/i);
  if (simple) return Number(simple[1]);
  return null;
}

export function parseZincMetrics(output: string, expect: string[] = []): BenchmarkMetrics {
  const decodeTokPerSec = parseDecodeTokPerSec(output);
  const prefillTokPerSec = parsePrefillTokPerSec(output);
  const promptTokens = parsePrefillTokenCount(output);
  const outputText =
    output.match(/Output text:\s*"([^"]*)"/)?.[1] ??
    output.match(/Output text:\s*([^\n]*)/)?.[1]?.trim() ??
    output.match(/Output:\s*"([^"]*)"/)?.[1] ??
    output.match(/Output:\s*([\s\S]*?)(?:\n(?:info|warn|error)\(|$)/)?.[1]?.trim() ??
    "";
  const coherent = expect.length === 0
    ? outputText.trim().length > 0
    : expect.some((needle) => outputText.toLowerCase().includes(needle.toLowerCase()));
  return { decodeTokPerSec, prefillTokPerSec, promptTokens, outputText, coherent };
}

function numberFromAny(obj: Record<string, unknown>, keys: string[]): number | null {
  for (const key of keys) {
    const value = obj[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string" && value.trim() && Number.isFinite(Number(value))) return Number(value);
  }
  return null;
}

export function parseLlamaBenchMetrics(output: string): LlamaSummary {
  let decodeTokPerSec: number | null = null;
  let prefillTokPerSec: number | null = null;
  let promptTokens: number | null = null;
  try {
    const jsonStart = output.indexOf("[");
    const data = JSON.parse(jsonStart >= 0 ? output.slice(jsonStart) : output);
    const rows = Array.isArray(data) ? data : [data];
    for (const row of rows) {
      if (!row || typeof row !== "object") continue;
      const rec = row as Record<string, unknown>;
      const tps = numberFromAny(rec, ["avg_ts", "median_ts", "tps", "tokens_per_second"]);
      const test = String(rec.test ?? rec.name ?? "").toLowerCase();
      const nGen = Number(rec.n_gen ?? rec.gen ?? 0);
      const nPrompt = Number(rec.n_prompt ?? rec.prompt ?? 0);
      if (tps != null && (test.includes("tg") || nGen > 0)) decodeTokPerSec = tps;
      if (tps != null && (test.includes("pp") || (nPrompt > 0 && nGen === 0))) {
        prefillTokPerSec = tps;
        promptTokens = nPrompt > 0 ? nPrompt : promptTokens;
      }
    }
  } catch {
    const tg = output.match(/(?:tg|decode)[^|\n]*\|\s*(\d+(?:\.\d+)?)\s*(?:±|\+\/-)?/i);
    const pp = output.match(/(?:pp|prefill)[^|\n]*\|\s*(\d+(?:\.\d+)?)\s*(?:±|\+\/-)?/i);
    if (tg) decodeTokPerSec = Number(tg[1]);
    if (pp) prefillTokPerSec = Number(pp[1]);
  }
  return { decodeTokPerSec, prefillTokPerSec, promptTokens, raw: output, source: "llama-bench" };
}

function parseLlamaTimingLine(line: string): number | null {
  const match = line.match(/,\s*(\d+(?:\.\d+)?)\s+tokens per second\s*\)/i);
  return match ? Number(match[1]) : null;
}

export function parseLlamaCliMetrics(output: string): LlamaSummary {
  let decodeTokPerSec: number | null = null;
  let prefillTokPerSec: number | null = null;
  let promptTokens: number | null = null;
  const source = output.includes("ZINC_LLAMA_SOURCE=llama-completion") ? "llama-completion" : "llama-cli";
  for (const line of output.split(/\r?\n/)) {
    if (/prompt eval time\s*=/.test(line)) {
      prefillTokPerSec = parseLlamaTimingLine(line) ?? prefillTokPerSec;
      const tokenMatch = line.match(/\/\s*(\d+)\s+tokens/i);
      if (tokenMatch) promptTokens = Number(tokenMatch[1]);
    } else if (/\beval time\s*=/.test(line)) {
      decodeTokPerSec = parseLlamaTimingLine(line) ?? decodeTokPerSec;
    }
  }
  return { decodeTokPerSec, prefillTokPerSec, promptTokens, raw: output, source };
}

export function combinedCommandOutput(result: Pick<CommandResult, "stdout" | "stderr">): string {
  return [result.stdout, result.stderr].filter((part) => part.length > 0).join("\n");
}

function sanitizeSshDiagnostic(line: string): string {
  return line
    .replace(/\b[A-Za-z_][A-Za-z0-9_.-]*@(?:\d{1,3}\.){3}\d{1,3}\b/g, "<user>@<host>")
    .replace(/\b[A-Za-z_][A-Za-z0-9_.-]*@[A-Za-z0-9_.-]+\b/g, "<user>@<host>")
    .replace(/(connect to host )\S+( port )\d+/i, "$1<host>$2<port>")
    .replace(/(Could not resolve hostname )\S+/i, "$1<host>")
    .replace(/(Connection (?:closed|reset) by )\S+/i, "$1<host>");
}

export function sshFailureSummary(result: Pick<CommandResult, "stdout" | "stderr">): string {
  const lines = combinedCommandOutput(result)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const sshDiagnostic = lines.find((line) => /^(ssh:|Permission denied|Host key verification failed|Could not resolve hostname|Connection (?:closed|refused|reset|timed out)|No route to host|kex_exchange_identification:)/i.test(line));
  return sanitizeSshDiagnostic(sshDiagnostic ?? lines[0] ?? "no remote command output");
}

export function isAgentAuthFailure(result: Pick<CommandResult, "exitCode" | "stdout" | "stderr">): boolean {
  if (result.exitCode === 0) return false;
  return /(?:Failed to authenticate|Invalid authentication credentials|API Error:\s*401|401 Unauthorized|authentication credentials)/i.test(
    combinedCommandOutput(result),
  );
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.floor(sorted.length / 2)];
}

async function runCommand(cmd: string, args: string[], opts: { cwd?: string; timeout?: number; stream?: boolean; formatter?: (line: string) => string | null; env?: NodeJS.ProcessEnv } = {}): Promise<CommandResult> {
  return new Promise((resolvePromise, reject) => {
    const child = spawn(cmd, args, {
      cwd: opts.cwd ?? REPO_ROOT,
      env: opts.env ?? process.env,
      stdio: ["ignore", "pipe", "pipe"],
      timeout: opts.timeout ?? 120_000,
    });
    let stdout = "";
    let stderr = "";
    let lineBuffer = "";
    child.stdout.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stdout += text;
      if (!opts.stream) return;
      if (!opts.formatter) {
        process.stdout.write(text);
        return;
      }
      lineBuffer += text;
      const lines = lineBuffer.split("\n");
      lineBuffer = lines.pop() ?? "";
      for (const line of lines) {
        const formatted = opts.formatter(line);
        if (formatted !== null) process.stdout.write(formatted);
      }
    });
    child.stderr.on("data", (chunk: Buffer) => {
      const text = chunk.toString("utf8");
      stderr += text;
      if (opts.stream) process.stderr.write(text);
    });
    child.on("error", reject);
    child.on("close", (code, signal) => {
      if (opts.stream && opts.formatter && lineBuffer.trim()) {
        const formatted = opts.formatter(lineBuffer);
        if (formatted !== null) process.stdout.write(formatted);
      }
      resolvePromise({ exitCode: code ?? 1, signal, stdout, stderr });
    });
  });
}

async function runWithOptionalAskpass(
  opts: LoopOptions,
  cmd: string,
  args: string[],
  commandOpts: { cwd?: string; timeout?: number; stream?: boolean; formatter?: (line: string) => string | null } = {},
): Promise<CommandResult> {
  const script = sshAskpassScript(opts);
  if (!script) return runCommand(cmd, args, commandOpts);

  const askpassPath = `/tmp/zinc-askpass-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
  await writeFile(askpassPath, script, { mode: 0o700 });
  try {
    return await runCommand(cmd, args, {
      ...commandOpts,
      env: {
        ...process.env,
        DISPLAY: process.env.DISPLAY ?? "zinc-askpass",
        SSH_ASKPASS: askpassPath,
        SSH_ASKPASS_REQUIRE: "force",
      },
    });
  } finally {
    await rm(askpassPath, { force: true });
  }
}

async function ssh(opts: LoopOptions, command: string, timeout = 120_000): Promise<CommandResult> {
  return runWithOptionalAskpass(opts, "ssh", [
    ...buildSshOptions(opts),
    "-p", String(opts.port),
    `${opts.user}@${opts.host}`,
    command,
  ], { timeout });
}

async function checkedSsh(opts: LoopOptions, command: string, timeout = 120_000): Promise<string> {
  const res = await ssh(opts, command, timeout);
  if (res.exitCode !== 0) {
    throw new Error(`remote command failed (${res.exitCode}): ${sshFailureSummary(res)}`);
  }
  return combinedCommandOutput(res);
}

async function rsyncToRemote(opts: LoopOptions): Promise<void> {
  const res = await runWithOptionalAskpass(opts, "rsync", [
    "-az",
    "--delete",
    "--exclude", ".git",
    "--exclude", ".zig-cache",
    "--exclude", ".zig-cache-local",
    "--exclude", ".zig-global-cache",
    "--exclude", ".zig-global-cache-local",
    "--exclude", ".zig-api-cache",
    "--exclude", "src/.zig-api-cache",
    "--exclude", "zig-out",
    "--exclude", "node_modules",
    "--exclude", "site/node_modules",
    "--exclude", "site/dist",
    "--exclude", ".gpu_optimize",
    "--exclude", ".perf_optimize",
    "--exclude", ".zinc_optimize",
    "--exclude", ".DS_Store",
    "-e", `ssh -p ${opts.port} ${buildSshOptions(opts).join(" ")}`,
    `${REPO_ROOT}/`,
    `${opts.user}@${opts.host}:${opts.remoteDir}/`,
  ], { timeout: 180_000 });
  if (res.exitCode !== 0) throw new Error(`rsync failed: ${res.stderr || res.stdout}`);
}

async function prepareRemote(opts: LoopOptions, target: ModelTarget): Promise<void> {
  await checkedSsh(opts, `mkdir -p ${q(opts.remoteDir)}`, 30_000);
  await rsyncToRemote(opts);
  if (opts.remoteLibcConf) {
    await checkedSsh(opts, `cd ${q(opts.remoteDir)} && mkdir -p .build-support && cp ${q(opts.remoteLibcConf)} .build-support/libc.conf`, 30_000);
  }
  await checkedSsh(opts, `cd ${q(opts.remoteDir)} && zig build -Doptimize=ReleaseFast --summary all`, 240_000);
  await checkedSsh(opts, `cd ${q(opts.remoteDir)} && ./zig-out/bin/zinc --help >/dev/null && ./zig-out/bin/zinc model list --all >/dev/null && ./zig-out/bin/zinc --check`, 120_000);
  if (target.modelId && !opts.modelPath) {
    const exists = await ssh(opts, `test -f ${q(target.modelPath)}`, 30_000);
    if (exists.exitCode !== 0) {
      await checkedSsh(opts, `cd ${q(opts.remoteDir)} && ${remoteEnvPrefix(opts)} ./zig-out/bin/zinc model pull ${q(target.modelId)}`, 1_800_000);
    }
  }
}

async function benchmarkZinc(opts: LoopOptions, target: ModelTarget): Promise<BenchmarkSummary> {
  const decodeSamples: number[] = [];
  const prefillSamples: number[] = [];
  const promptTokenSamples: number[] = [];
  let outputText = "";
  let coherent = true;
  for (let i = 0; i < opts.samples; i++) {
    const res = await checkedSsh(opts, buildRemoteZincCommand(opts, target), 900_000);
    const metrics = parseZincMetrics(res, target.expect);
    if (metrics.decodeTokPerSec != null) decodeSamples.push(metrics.decodeTokPerSec);
    if (metrics.prefillTokPerSec != null) prefillSamples.push(metrics.prefillTokPerSec);
    if (metrics.promptTokens != null) promptTokenSamples.push(metrics.promptTokens);
    outputText = metrics.outputText;
    coherent = coherent && metrics.coherent;
    console.log(`    ZINC sample ${i + 1}/${opts.samples}: decode=${metrics.decodeTokPerSec?.toFixed(2) ?? "?"} prefill=${metrics.prefillTokPerSec?.toFixed(2) ?? "?"}${metrics.promptTokens != null ? ` prompt=${metrics.promptTokens}tok` : ""} coherent=${metrics.coherent ? "yes" : "no"}`);
  }
  const samples = opts.metric === "decode" ? decodeSamples : prefillSamples;
  return {
    metric: opts.metric,
    value: median(samples),
    samples,
    decodeSamples,
    prefillSamples,
    promptTokenSamples,
    outputText,
    coherent,
  };
}

async function benchmarkLlama(opts: LoopOptions, target: ModelTarget): Promise<LlamaSummary | null> {
  if (opts.skipLlama) return null;
  const command = opts.llamaBench
    ? buildRemoteLlamaBenchCommand(opts, target)
    : buildRemoteLlamaCliCommand(opts, target);
  const res = await ssh(opts, command, 900_000);
  if (res.exitCode !== 0) {
    console.log(`    llama.cpp skipped: ${res.stderr || res.stdout}`.slice(0, 500));
    return null;
  }
  const output = combinedCommandOutput(res);
  return opts.llamaBench ? parseLlamaBenchMetrics(output) : parseLlamaCliMetrics(output);
}

function valueForMetric(summary: BenchmarkSummary | null, metric: MetricKind): number | null {
  if (!summary) return null;
  if (summary.metric === metric) return summary.value;
  return median(metric === "decode" ? summary.decodeSamples : summary.prefillSamples);
}

function llamaValueForMetric(llama: LlamaSummary | null, metric: MetricKind): number | null {
  if (!llama) return null;
  return metric === "decode" ? llama.decodeTokPerSec : llama.prefillTokPerSec;
}

function metricRatio(summary: BenchmarkSummary | null, llama: LlamaSummary | null, metric: MetricKind): number | null {
  const zinc = valueForMetric(summary, metric);
  const baseline = llamaValueForMetric(llama, metric);
  if (zinc == null || baseline == null || baseline <= 0) return null;
  return zinc / baseline;
}

function weakestLlamaGapMetric(summary: BenchmarkSummary | null, llama: LlamaSummary | null): MetricKind | null {
  const candidates = (["decode", "prefill"] as const)
    .map((metric) => ({ metric, ratio: metricRatio(summary, llama, metric) }))
    .filter((entry): entry is { metric: MetricKind; ratio: number } => entry.ratio != null && entry.ratio < 1);
  candidates.sort((a, b) => a.ratio - b.ratio);
  return candidates[0]?.metric ?? null;
}

function beatsLlamaOnBoth(summary: BenchmarkSummary | null, llama: LlamaSummary | null): boolean {
  const decodeRatio = metricRatio(summary, llama, "decode");
  const prefillRatio = metricRatio(summary, llama, "prefill");
  return decodeRatio != null && prefillRatio != null && decodeRatio >= 1 && prefillRatio >= 1;
}

function materiallyImprovedMetric(before: BenchmarkSummary, after: BenchmarkSummary, metric: MetricKind): boolean {
  const prev = valueForMetric(before, metric);
  const next = valueForMetric(after, metric);
  if (prev == null || next == null) return false;
  const threshold = Math.max(0.25, prev * 0.01);
  return next >= prev + threshold;
}

function doesNotLoseBeatenMetric(before: BenchmarkSummary, after: BenchmarkSummary, llama: LlamaSummary | null, metric: MetricKind): boolean {
  const beforeRatio = metricRatio(before, llama, metric);
  const afterRatio = metricRatio(after, llama, metric);
  if (beforeRatio == null || afterRatio == null) return true;
  if (beforeRatio < 1) return true;
  return afterRatio >= 0.98;
}

function isImproved(before: BenchmarkSummary, after: BenchmarkSummary, llama: LlamaSummary | null): boolean {
  if (!after.coherent) return false;
  const focus = weakestLlamaGapMetric(before, llama);
  if (focus) {
    if (!materiallyImprovedMetric(before, after, focus)) return false;
    return doesNotLoseBeatenMetric(before, after, llama, "decode")
      && doesNotLoseBeatenMetric(before, after, llama, "prefill");
  }
  return materiallyImprovedMetric(before, after, before.metric);
}

function formatMetricComparison(summary: BenchmarkSummary | null, llama: LlamaSummary | null, metric: MetricKind): string {
  const zinc = valueForMetric(summary, metric);
  const baseline = llamaValueForMetric(llama, metric);
  if (zinc == null || baseline == null || baseline <= 0) return `${metric}: unavailable`;
  const ratio = (zinc / baseline) * 100;
  const delta = zinc - baseline;
  const status = zinc >= baseline ? "beating" : "behind";
  return `${metric}: ZINC ${zinc.toFixed(2)} vs llama.cpp ${baseline.toFixed(2)} tok/s (${ratio.toFixed(1)}%, ${status}, delta ${delta >= 0 ? "+" : ""}${delta.toFixed(2)})`;
}

function formatLlamaGoal(summary: BenchmarkSummary | null, llama: LlamaSummary | null, targetLabel = "selected model"): string {
  if (!llama) return "llama.cpp comparison unavailable; optimize the selected ZINC metric without regressing correctness.";
  const focus = weakestLlamaGapMetric(summary, llama);
  const lines = [
    `Objective: beat llama.cpp on both sustained decode and prompt prefill for ${targetLabel}.`,
    formatMetricComparison(summary, llama, "decode"),
    formatMetricComparison(summary, llama, "prefill"),
  ];
  if (focus) {
    lines.push(`Current attack metric: ${focus} (largest remaining llama.cpp gap).`);
  } else if (beatsLlamaOnBoth(summary, llama)) {
    lines.push("Current attack metric: both decode and prefill already beat llama.cpp; preserve both and look for additional margin.");
  }
  return lines.join("\n");
}

async function currentChangedFiles(): Promise<string[]> {
  const res = await runCommand("git", ["diff", "--name-only"], { timeout: 30_000 });
  const untracked = await runCommand("git", ["ls-files", "--others", "--exclude-standard"], { timeout: 30_000 });
  return [...new Set([
    ...res.stdout.split("\n"),
    ...untracked.stdout.split("\n"),
  ].map((s) => s.trim()).filter(Boolean))].sort();
}

export function changedSince(before: string[], after: string[]): string[] {
  const beforeSet = new Set(before);
  return after.filter((file) => !beforeSet.has(file));
}

async function requireCleanWorktree(): Promise<void> {
  const res = await runCommand("git", ["status", "--porcelain"], { timeout: 30_000 });
  if (res.stdout.trim()) {
    throw new Error("worktree is dirty; commit/stash first or pass --allow-dirty --no-revert");
  }
}

async function revertFiles(files: string[]): Promise<void> {
  if (files.length === 0) return;
  const tracked = await runCommand("git", ["ls-files", "--", ...files], { timeout: 30_000 });
  const trackedFiles = tracked.stdout.split("\n").map((s) => s.trim()).filter(Boolean);
  const trackedSet = new Set(trackedFiles);
  if (trackedFiles.length > 0) {
    await runCommand("git", ["restore", "--", ...trackedFiles], { timeout: 30_000 });
  }
  for (const file of files) {
    if (trackedSet.has(file)) continue;
    await rm(join(REPO_ROOT, file), { force: true, recursive: true });
  }
}

function summarizeError(error: unknown): string {
  const text = error instanceof Error ? error.message : String(error);
  const lines = text.split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  const interesting = lines.find((line) =>
    line.includes("FenceWaitFailed") ||
    line.includes("ShaderFileNotFound") ||
    line.startsWith("error:") ||
    line.startsWith("err(")
  );
  return (interesting ?? lines[0] ?? "unknown error").slice(0, 500);
}

function buildRunId(target: ModelTarget, opts: LoopOptions): string {
  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  return opts.runId ?? `${target.key}-${opts.metric}-${stamp}`;
}

function findLatestRunDir(): string | null {
  if (!existsSync(RESULTS_DIR)) return null;
  const dirs = readdirSync(RESULTS_DIR)
    .map((name) => join(RESULTS_DIR, name))
    .filter((p) => existsSync(join(p, "state.json")))
    .sort((a, b) => statSync(b).mtimeMs - statSync(a).mtimeMs);
  return dirs[0] ?? null;
}

async function loadState(runDir: string): Promise<RunState | null> {
  const path = join(runDir, "state.json");
  if (!existsSync(path)) return null;
  return JSON.parse(await readFile(path, "utf8")) as RunState;
}

async function saveState(runDir: string, state: RunState): Promise<void> {
  await mkdir(runDir, { recursive: true });
  state.updatedAt = new Date().toISOString();
  await writeFile(join(runDir, "state.json"), JSON.stringify(state, null, 2));
}

async function saveCyclePatch(runDir: string, cycle: number, changedFiles: string[]): Promise<void> {
  if (changedFiles.length === 0) return;
  const diff = await runCommand("git", ["diff", "--binary", "--", ...changedFiles], { cwd: REPO_ROOT });
  const body = combinedCommandOutput(diff).trimEnd();
  if (body.length === 0) return;
  await mkdir(runDir, { recursive: true });
  await writeFile(join(runDir, `cycle-${String(cycle).padStart(4, "0")}.diff`), `${body}\n`);
}

function summarizeBench(summary: BenchmarkSummary | null): string {
  if (!summary) return "not measured";
  const selected = summary.value == null ? "?" : `${summary.value.toFixed(2)} [${summary.samples.map((v) => v.toFixed(2)).join(", ")}]`;
  const decode = summarizeMetricSamples(summary.decodeSamples);
  const prefill = summarizeMetricSamples(summary.prefillSamples);
  const promptTokens = summarizePromptTokenSamples(summary.promptTokenSamples);
  return `${summary.metric}=${selected} tok/s; decode=${decode}; prefill=${prefill}; prompt=${promptTokens}; coherent=${summary.coherent ? "yes" : "no"}`;
}

function summarizeMetricSamples(samples: number[]): string {
  const value = median(samples);
  if (value == null) return "?";
  return `${value.toFixed(2)} [${samples.map((v) => v.toFixed(2)).join(", ")}]`;
}

function summarizePromptTokenSamples(samples: number[]): string {
  const value = median(samples);
  if (value == null) return "?";
  return `${value}tok [${samples.map((v) => String(v)).join(", ")}]`;
}

function cycleMetricValue(cycle: CycleRecord, metric: MetricKind): number | null {
  const after = cycle.after;
  if (!after) return null;
  return valueForMetric(after, metric);
}

function classifyCycle(cycle: CycleRecord): string {
  const files = (cycle.changedFiles ?? []).join(" ");
  const reason = cycle.reason.toLowerCase();
  if (reason.includes("authentication") || reason.includes("401")) return "agent auth";
  if (reason.includes("no source changes")) return "no source";
  if (reason.includes("remote build") || reason.includes("benchmark failed")) return "build/bench fail";
  if (cycle.after && !cycle.after.coherent) return "coherence";
  if (files.includes("dmmv_q6k_batch")) return "q6k batch";
  if (files.includes("dmmv_q4k_batch")) return "q4k batch";
  if (files.includes("mul_mm_q4k")) return "mul_mm q4k";
  if (files.includes("src/compute/dmmv.zig")) return "dmmv host";
  if (files.includes("src/compute/forward.zig")) return "forward dispatch";
  if (files.includes("dmmv_q6k.comp")) return "q6k decode";
  if (files.includes("dmmv_q4k.comp")) return "q4k decode";
  return "other";
}

function formatCycleBrief(cycle: CycleRecord): string {
  const decode = cycleMetricValue(cycle, "decode");
  const prefill = cycleMetricValue(cycle, "prefill");
  const files = cycle.changedFiles.length > 0 ? cycle.changedFiles.join(",") : "none";
  return `#${cycle.cycle} ${cycle.kept ? "kept" : "reverted"} d=${decode?.toFixed(2) ?? "?"} p=${prefill?.toFixed(2) ?? "?"} files=${files}`;
}

function buildCycleMemory(state: RunState): string {
  const cycles = state.cycles;
  if (cycles.length === 0) return "(no prior cycles)";

  const measured = cycles.filter((c) => c.after != null);
  const kept = cycles.filter((c) => c.kept);
  const noSource = cycles.filter((c) => c.changedFiles.length === 0).length;
  const bestDecode = measured
    .map((c) => cycleMetricValue(c, "decode"))
    .filter((v): v is number => v != null)
    .sort((a, b) => b - a)[0];
  const bestPrefill = measured
    .map((c) => cycleMetricValue(c, "prefill"))
    .filter((v): v is number => v != null)
    .sort((a, b) => b - a)[0];

  const categories = new Map<string, { total: number; kept: number; bestDecode: number | null; bestPrefill: number | null }>();
  for (const cycle of cycles) {
    const key = classifyCycle(cycle);
    const entry = categories.get(key) ?? { total: 0, kept: 0, bestDecode: null, bestPrefill: null };
    entry.total += 1;
    if (cycle.kept) entry.kept += 1;
    const decode = cycleMetricValue(cycle, "decode");
    const prefill = cycleMetricValue(cycle, "prefill");
    if (decode != null) entry.bestDecode = entry.bestDecode == null ? decode : Math.max(entry.bestDecode, decode);
    if (prefill != null) entry.bestPrefill = entry.bestPrefill == null ? prefill : Math.max(entry.bestPrefill, prefill);
    categories.set(key, entry);
  }

  const categoryLines = [...categories.entries()]
    .sort((a, b) => b[1].total - a[1].total)
    .slice(0, 8)
    .map(([key, entry]) =>
      `${key}: ${entry.kept}/${entry.total} kept, best d=${entry.bestDecode?.toFixed(2) ?? "?"}, p=${entry.bestPrefill?.toFixed(2) ?? "?"}`
    );
  const keptLines = kept.slice(-8).map(formatCycleBrief);
  const rejectedMeasured = cycles.filter((c) => !c.kept && c.after != null && c.after.coherent);
  const topRejectedPrefill = [...rejectedMeasured]
    .sort((a, b) => (cycleMetricValue(b, "prefill") ?? -Infinity) - (cycleMetricValue(a, "prefill") ?? -Infinity))
    .slice(0, 5)
    .map(formatCycleBrief);
  const topRejectedDecode = [...rejectedMeasured]
    .sort((a, b) => (cycleMetricValue(b, "decode") ?? -Infinity) - (cycleMetricValue(a, "decode") ?? -Infinity))
    .slice(0, 5)
    .map(formatCycleBrief);

  return [
    `cycles=${cycles.length}; measured=${measured.length}; kept=${kept.length}; no-source=${noSource}`,
    `best observed decode=${bestDecode?.toFixed(2) ?? "?"} tok/s; best observed prefill=${bestPrefill?.toFixed(2) ?? "?"} tok/s`,
    `accepted baseline: ${summarizeBench(state.best)}`,
    `categories: ${categoryLines.join(" | ")}`,
    `kept changes: ${keptLines.join(" | ") || "none"}`,
    `top rejected prefill: ${topRejectedPrefill.join(" | ") || "none"}`,
    `top rejected decode: ${topRejectedDecode.join(" | ") || "none"}`,
  ].join("\n");
}

function cyclesSinceLastKeep(state: RunState): number {
  for (let i = state.cycles.length - 1; i >= 0; i--) {
    if (state.cycles[i].kept) return state.cycles.length - 1 - i;
  }
  return state.cycles.length;
}

export function buildAgentPrompt(state: RunState, opts: LoopOptions, target: ModelTarget, baseline: BenchmarkSummary, llama: LlamaSummary | null): string {
  const recent = state.cycles.slice(-8).map((c) =>
    `- cycle ${c.cycle}: ${c.kept ? "kept" : "reverted"} ${c.reason}; changed=${c.changedFiles.join(", ") || "none"}; after=${summarizeBench(c.after)}`
  ).join("\n") || "(no cycles yet)";
  const failed = state.failedApproaches.slice(-10).map((f) => `- ${f}`).join("\n") || "(none)";
  const llamaSource = llama?.source ? `${llama.source} ` : "";
  const llamaLine = llama
    ? `llama.cpp ${llamaSource}same-prompt baseline: decode=${llama.decodeTokPerSec?.toFixed(2) ?? "?"} tok/s, prefill=${llama.prefillTokPerSec?.toFixed(2) ?? "?"} tok/s${llama.promptTokens != null ? `, prompt=${llama.promptTokens}tok` : ""}`
    : "llama.cpp baseline unavailable or skipped";
  const goal = formatLlamaGoal(baseline, llama, target.label);
  return `
You are optimizing ZINC on a remote consumer GPU target. Make exactly one bounded source change, then stop.

Target:
- model: ${target.label}
- model path: ${target.modelPath}
- harness metric: ${opts.metric} tok/s; keep decisions prefer the largest remaining llama.cpp gap when a llama.cpp baseline exists
- measured prompt shape: ${summarizePromptTokenSamples(baseline.promptTokenSamples)}
- current ZINC baseline: ${summarizeBench(baseline)}
- ${llamaLine}
- remote: ${opts.user}@${opts.host}:${opts.port} ${opts.remoteDir}

Llama.cpp gap:
${goal}

All-cycle memory:
${buildCycleMemory(state)}

Required workflow:
1. Inspect the relevant local code before editing.
2. Make one small, testable change under src/, build.zig, loops/, or diagnostics/site support only if needed for this target.
3. Do not commit, push, reset, stash, or edit secrets.
4. Run at least a local compile or focused test if feasible.
5. End with:
   STEP_KIND: optimization|fix|analysis|rollback
   DESCRIPTION: one sentence
   SELF_ANALYSIS: what changed, why it should help, and what to measure next
   NEXT_IDEAS:
   - idea 1
   - idea 2

Recent cycles:
${recent}

Known failed approaches:
${failed}
`.trim();
}

async function runAgent(prompt: string, agent: AgentKind): Promise<CommandResult> {
  const startedAt = Date.now();
  const heartbeat = setInterval(() => {
    process.stdout.write(`\nstill running (${formatElapsed(startedAt)} elapsed)...\n`);
  }, 30_000);

  try {
    if (agent === "codex") {
      return await runCommand("codex", codexExecArgs(prompt), {
        timeout: 7_200_000,
        stream: true,
        formatter: (line) => formatCodexStreamLine(line),
      });
    }
    const claudeState: ClaudeStreamState = {
      currentToolName: null,
      currentBlockIsToolUse: false,
      inputJsonBuffer: "",
      inTextBlock: false,
      sawTextDeltaInCurrentMessage: false,
    };
    return await runCommand("claude", [
      "-p",
      "--verbose",
      "--output-format", "stream-json",
      "--include-partial-messages",
      `--disallowed-tools=${[...BLOCKED_GIT_OPS, ...BLOCKED_FILE_OPS].join(",")}`,
      "--permission-mode", "bypassPermissions",
      "--model", CLAUDE_MODEL,
      "--effort", CLAUDE_EFFORT,
      prompt,
    ], {
      timeout: 7_200_000,
      stream: true,
      formatter: (line) => formatClaudeStreamLine(line, claudeState),
    });
  } finally {
    clearInterval(heartbeat);
  }
}

async function main(): Promise<void> {
  const opts = parseArgsFrom(process.argv.slice(2));
  const target = resolveModelTarget(opts);
  const runDir = opts.resume
    ? (opts.resumeDir ?? findLatestRunDir())
    : join(RESULTS_DIR, buildRunId(target, opts));
  if (!runDir) throw new Error("No previous .gpu_optimize run found for --resume");
  if (!opts.allowDirty && !opts.dryRun) await requireCleanWorktree();

  console.log(`ZINC remote GPU optimization loop`);
  console.log(`  agent: ${opts.agent}`);
  console.log(`  model: ${target.label}`);
  console.log(`  metric: ${opts.metric}`);
  console.log(`  run: ${runDir}`);

  let state = await loadState(runDir);
  const mismatch = stateTargetMismatchReason(state, target, opts);
  if (mismatch) throw new Error(mismatch);

  console.log("\nPreparing remote...");
  await prepareRemote(opts, target);

  if (!state) {
    console.log("\nBaseline...");
    const baseline = await benchmarkZinc(opts, target);
    const llama = await benchmarkLlama(opts, target);
    state = {
      startedAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      runId: runDir.split("/").pop() ?? "run",
      options: {
        modelKey: target.key,
        modelPath: target.modelPath,
        metric: opts.metric,
        host: opts.host,
        user: opts.user,
        port: opts.port,
        remoteDir: opts.remoteDir,
      },
      best: baseline,
      llamaBaseline: llama,
      cycles: [],
      failedApproaches: [],
    };
    await saveState(runDir, state);
  } else {
    console.log(`\nResumed with best: ${summarizeBench(state.best)}`);
  }

  if (opts.dryRun || opts.cycles === 0) {
    console.log(`\nBaseline: ${summarizeBench(state.best)}`);
    if (state.llamaBaseline) {
      const source = state.llamaBaseline.source ? ` ${state.llamaBaseline.source}` : "";
      console.log(`llama.cpp${source}: decode=${state.llamaBaseline.decodeTokPerSec?.toFixed(2) ?? "?"} prefill=${state.llamaBaseline.prefillTokPerSec?.toFixed(2) ?? "?"}`);
      console.log(formatLlamaGoal(state.best, state.llamaBaseline, target.label));
    }
    return;
  }

  for (let i = 0; i < opts.cycles; i++) {
    const stalledCycles = cyclesSinceLastKeep(state);
    if (opts.maxStallCycles > 0 && stalledCycles >= opts.maxStallCycles) {
      console.log(`\nPlateau stop: ${stalledCycles} cycles since the last kept improvement (limit ${opts.maxStallCycles}).`);
      console.log(`Accepted best remains: ${summarizeBench(state.best)}`);
      console.log("Use --max-stall-cycles 0 to disable this guard after choosing a structurally different direction.");
      break;
    }

    const cycle = state.cycles.length + 1;
    const before = state.best ?? await benchmarkZinc(opts, target);
    const prompt = buildAgentPrompt(state, opts, target, before, state.llamaBaseline);
    console.log(`\nCycle ${cycle}`);
    const dirtyBeforeAgent = await currentChangedFiles();
    const agentResult = await runAgent(prompt, opts.agent);
    if (isAgentAuthFailure(agentResult)) {
      throw new Error(
        `Agent authentication failed for ${opts.agent}; stopping the loop before burning more cycles. Refresh the ${opts.agent} CLI credentials, then resume this run.`,
      );
    }
    const changedFiles = opts.allowDirty
      ? changedSince(dirtyBeforeAgent, await currentChangedFiles())
      : await currentChangedFiles();
    await saveCyclePatch(runDir, cycle, changedFiles);

    let after: BenchmarkSummary | null = null;
    let kept = false;
    let improved = false;
    let reason = "agent failed";
    if (agentResult.exitCode === 0 && changedFiles.length > 0) {
      try {
        await prepareRemote(opts, target);
        after = await benchmarkZinc(opts, target);
        improved = isImproved(before, after, state.llamaBaseline);
        kept = improved;
        const focus = weakestLlamaGapMetric(before, state.llamaBaseline) ?? opts.metric;
        reason = improved
          ? `improved ${focus} against llama.cpp gap; ${formatMetricComparison(after, state.llamaBaseline, focus)}`
          : `no improvement on llama.cpp focus metric or coherence failure (${summarizeBench(after)}; ${formatLlamaGoal(after, state.llamaBaseline, target.label).replace(/\n/g, " | ")})`;
      } catch (error) {
        reason = `remote build or benchmark failed: ${summarizeError(error)}`;
      }
    } else if (changedFiles.length === 0) {
      reason = "no source changes";
    }

    if (!kept && opts.autoRevert && !opts.allowDirty) {
      await revertFiles(changedFiles);
      if (changedFiles.length > 0) {
        try {
          await prepareRemote(opts, target);
        } catch (error) {
          reason = `${reason}; remote restore failed after revert: ${summarizeError(error)}`;
        }
      }
    }
    if (!kept) state.failedApproaches.push(reason);
    if (kept && after) state.best = after;

    state.cycles.push({
      cycle,
      timestamp: new Date().toISOString(),
      changedFiles,
      before,
      after,
      llama: state.llamaBaseline,
      kept,
      improved,
      reason,
    });
    await saveState(runDir, state);
    console.log(`  ${kept ? "kept" : "reverted"}: ${reason}`);
    if (!kept && changedFiles.length > 0 && opts.autoRevert && !opts.allowDirty) {
      console.log(`  restored accepted best: ${summarizeBench(state.best)}`);
    }

    if (opts.targetTps != null && state.best?.value != null && state.best.value >= opts.targetTps) {
      console.log(`Target reached: ${state.best.value.toFixed(2)} >= ${opts.targetTps}`);
      break;
    }
    if (opts.targetTps == null && !opts.continueAfterLlama && beatsLlamaOnBoth(state.best, state.llamaBaseline)) {
      console.log("Target reached: ZINC beats llama.cpp on both decode and prefill.");
      break;
    }
  }
}

if (import.meta.main) {
  main().catch((err) => {
    console.error(err instanceof Error ? err.message : String(err));
    process.exit(1);
  });
}
