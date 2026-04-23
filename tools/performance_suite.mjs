import { spawn, spawnSync } from "node:child_process";
import { promises as fs } from "node:fs";
import net from "node:net";
import os from "node:os";
import path from "node:path";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

const TOOL_PATH = fileURLToPath(import.meta.url);
const ROOT = path.resolve(path.dirname(TOOL_PATH), "..");
const DEFAULT_SITE_DATA = path.join(ROOT, "site", "src", "data", "zinc-performance.json");
const DEFAULT_OUTPUT = `/tmp/zinc-performance-${Date.now()}.json`;
const DEFAULT_TIMEOUT_MS = 15 * 60 * 1000;
const DEFAULT_LOCAL_CACHE = path.join(ROOT, ".zig-global-cache");
export const DEFAULT_LOCAL_MODEL_ROOT = path.join(os.homedir(), "Library", "Caches", "zinc", "models", "models");
const DEFAULT_DOCKER_LLAMA_SERVER = path.join(os.homedir(), ".docker", "bin", "inference", "llama-server");
const DEFAULT_RDNA_WORKDIR = "/root/zinc";
const DEFAULT_RDNA_MODEL_ROOT = "/root/models";
const TARGET_ORDER = ["rdna", "metal"];
const MAX_CAPTURE_CHARS = 256_000;
const PUBLIC_BENCHMARK_EXCLUDED_MODEL_IDS = new Set([
  "qwen35-2b-q4k-m",
]);

function modelPath(root, dir) {
  return path.join(root, dir, "model.gguf");
}

function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
}

function shouldIncludeInPublishedBenchmarks(modelId, requestedModels = null) {
  return requestedModels?.has(modelId) || !PUBLIC_BENCHMARK_EXCLUDED_MODEL_IDS.has(modelId);
}

function parseInteger(arg, flag) {
  const value = Number.parseInt(arg ?? "", 10);
  if (!Number.isFinite(value) || value < 0) {
    throw new Error(`Invalid ${flag} value '${arg}'`);
  }
  return value;
}

function stripOptionalQuotes(value) {
  if (value.length >= 2 && ((value.startsWith('"') && value.endsWith('"')) || (value.startsWith("'") && value.endsWith("'")))) {
    return value.slice(1, -1);
  }
  return value;
}

export function parseDotEnv(text) {
  const env = {};
  for (const rawLine of text.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (!line || line.startsWith("#")) continue;
    const exportPrefix = line.startsWith("export ") ? "export ".length : 0;
    const eq = line.indexOf("=", exportPrefix);
    if (eq === -1) continue;
    const key = line.slice(exportPrefix, eq).trim();
    const value = stripOptionalQuotes(line.slice(eq + 1).trim());
    if (key) env[key] = value;
  }
  return env;
}

async function readDotEnv(dotEnvPath) {
  try {
    return parseDotEnv(await fs.readFile(dotEnvPath, "utf8"));
  } catch {
    return {};
  }
}

function whichLocalBinary(name) {
  const result = spawnSync("which", [name], { encoding: "utf8" });
  if (result.status === 0) {
    return result.stdout.trim() || null;
  }
  return null;
}

export function resolveLocalLlamaServer(args, detectedPath = whichLocalBinary("llama-server"), dockerPath = DEFAULT_DOCKER_LLAMA_SERVER) {
  if (args.llamaServer) return args.llamaServer;
  if (detectedPath) return detectedPath;
  if (path.isAbsolute(dockerPath)) return dockerPath;
  return null;
}

export function parseArgs(argv) {
  const args = {
    target: "both",
    output: DEFAULT_OUTPUT,
    siteData: DEFAULT_SITE_DATA,
    writeSiteData: true,
    runs: 3,
    warmupRuns: 1,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    models: null,
    buildLocal: true,
    rdnaSync: false,
    rdnaBuild: false,
    rdnaStartLlama: false,
    metalPullMissing: false,
    localModelRoot: DEFAULT_LOCAL_MODEL_ROOT,
    rdnaModelRoot: DEFAULT_RDNA_MODEL_ROOT,
    rdnaWorkdir: DEFAULT_RDNA_WORKDIR,
    localCacheDir: DEFAULT_LOCAL_CACHE,
    llamaCli: process.env.ZINC_LLAMA_CLI ?? null,
    llamaServer: process.env.ZINC_LLAMA_SERVER ?? null,
    discoverModels: false,
    help: false,
  };

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    switch (arg) {
      case "--target": {
        const value = argv[++i] ?? "";
        if (!["metal", "rdna", "both"].includes(value)) {
          throw new Error(`Invalid --target '${value}'. Expected metal, rdna, or both.`);
        }
        args.target = value;
        break;
      }
      case "--output":
        args.output = argv[++i] ?? args.output;
        break;
      case "--site-data":
        args.siteData = argv[++i] ?? args.siteData;
        break;
      case "--no-site-write":
        args.writeSiteData = false;
        break;
      case "--runs":
        args.runs = parseInteger(argv[++i], "--runs");
        break;
      case "--warmup":
        args.warmupRuns = parseInteger(argv[++i], "--warmup");
        break;
      case "--timeout-ms":
        args.timeoutMs = parseInteger(argv[++i], "--timeout-ms");
        break;
      case "--models":
        args.models = new Set((argv[++i] ?? "").split(",").map((value) => value.trim()).filter(Boolean));
        break;
      case "--discover-models":
      case "--all-models":
        args.discoverModels = true;
        break;
      case "--llama-cli":
        args.llamaCli = argv[++i] ?? args.llamaCli;
        break;
      case "--llama-server":
        args.llamaServer = argv[++i] ?? args.llamaServer;
        break;
      case "--skip-local-build":
        args.buildLocal = false;
        break;
      case "--rdna-sync":
        args.rdnaSync = true;
        break;
      case "--rdna-build":
        args.rdnaBuild = true;
        break;
      case "--rdna-start-llama":
        args.rdnaStartLlama = true;
        break;
      case "--metal-pull-missing":
        args.metalPullMissing = true;
        break;
      case "--local-model-root":
        args.localModelRoot = argv[++i] ?? args.localModelRoot;
        break;
      case "--rdna-model-root":
        args.rdnaModelRoot = argv[++i] ?? args.rdnaModelRoot;
        break;
      case "--rdna-workdir":
        args.rdnaWorkdir = argv[++i] ?? args.rdnaWorkdir;
        break;
      case "--local-cache-dir":
        args.localCacheDir = argv[++i] ?? args.localCacheDir;
        break;
      case "-h":
      case "--help":
        args.help = true;
        break;
      default:
        throw new Error(`Unknown argument '${arg}'`);
    }
  }

  if (args.runs === 0) throw new Error("--runs must be at least 1");
  return args;
}

function usage() {
  return `Usage: bun tools/performance_suite.mjs [options]
  --target <metal|rdna|both>  Which benchmark target(s) to run
  --output <path>             JSON artifact output path
  --site-data <path>          Site JSON data path to update
  --no-site-write             Do not update the site JSON artifact
  --runs <n>                  Measured runs per model (default: 3)
  --warmup <n>                Warmup runs per model (default: 1)
  --timeout-ms <ms>           Per-command timeout (default: 900000)
  --models <ids>              Comma-separated model ids to benchmark
  --discover-models           Benchmark every discovered local model cache entry
  --all-models                Alias for --discover-models
  --llama-cli <path>          Optional local llama-cli path for Metal comparisons
  --llama-server <path>       Optional local llama-server path for Metal OpenAI baselines
  --skip-local-build          Skip local ReleaseFast build before Metal benchmarks
  --metal-pull-missing        Pull missing managed Metal models into the default local cache before running
  --rdna-sync                 Rsync current repo to the RDNA node before running
  --rdna-build                Build ReleaseFast on the RDNA node before running
  --rdna-start-llama          Start llama-server on the RDNA node before baseline runs
  --local-model-root <path>   Override local GGUF cache root
  --rdna-model-root <path>    Override remote model root
  --rdna-workdir <path>       Override remote ZINC checkout path
  --local-cache-dir <path>    Override local Zig cache directory
  -h, --help                  Show this help
`;
}

export function parseZincCliOutput(text) {
  const promptTokens = text.match(/Prompt tokens \((\d+)\):/);
  const prefill = text.match(/Prefill(?:\s+complete)?\s*:\s*(\d+)\s+tokens\s+in\s+([\d.]+)\s*(ms|s)\s*\(([\d.]+)\s+tok\/s\)/i);
  const prefillTimingOnly = text.match(/Prefill(?:\s+complete)?\s*:\s*(\d+)\s+tokens\s+in\s+([\d.]+)\s*(ms|s)/i);
  const generated = text.match(/Generated\s+(\d+)\s+tokens\s+in\s+([\d.]+)\s+ms\s+[—-]\s+([\d.]+)\s+tok\/s\s+\(([\d.]+)\s+ms\/tok\)/);
  const output = text.match(/Output \((\d+) tokens\):\s*(.+)$/m);
  const outputText = text.match(/info\(zinc\): Output text:\s*([\s\S]*?)(?:\ninfo\(zinc\): Output tokens|\s*$)/);

  if (!generated) {
    throw new Error("Could not parse ZINC CLI output");
  }

  return {
    promptTokens: promptTokens ? Number(promptTokens[1]) : null,
    prefillTokens: prefill ? Number(prefill[1]) : (prefillTimingOnly ? Number(prefillTimingOnly[1]) : (promptTokens ? Number(promptTokens[1]) : null)),
    prefillMs: prefill
      ? (prefill[3] === "s" ? Number(prefill[2]) * 1000 : Number(prefill[2]))
      : (prefillTimingOnly ? (prefillTimingOnly[3] === "s" ? Number(prefillTimingOnly[2]) * 1000 : Number(prefillTimingOnly[2])) : null),
    prefillTps: prefill
      ? Number(prefill[4])
      : (prefillTimingOnly
        ? (() => {
            const tokens = Number(prefillTimingOnly[1]);
            const seconds = prefillTimingOnly[3] === "s" ? Number(prefillTimingOnly[2]) : Number(prefillTimingOnly[2]) / 1000;
            return seconds > 0 ? tokens / seconds : null;
          })()
        : null),
    generatedTokens: Number(generated[1]),
    decodeMs: Number(generated[2]),
    decodeTps: Number(generated[3]),
    msPerToken: Number(generated[4]),
    outputPreview: output ? output[2].trim() : (outputText ? outputText[1].trim() : ""),
  };
}

export function parseLlamaCliOutput(text) {
  const prompt = text.match(/^\s*llama_print_timings:\s+prompt eval time =\s*([\d.]+)\s*ms\s*\/\s*(\d+)\s+tokens.*?,\s*([\d.]+)\s+tokens per second\)/m);
  const decode = text.match(/^\s*llama_print_timings:\s+eval time =\s*([\d.]+)\s*ms\s*\/\s*(\d+)\s+(?:runs|tokens).*?,\s*([\d.]+)\s+tokens per second\)/m);

  if (!decode) {
    throw new Error("Could not parse llama.cpp CLI timings");
  }

  const decodeMs = Number(decode[1]);
  const generatedTokens = Number(decode[2]);
  return {
    promptTokens: prompt ? Number(prompt[2]) : null,
    prefillTokens: prompt ? Number(prompt[2]) : null,
    prefillMs: prompt ? Number(prompt[1]) : null,
    prefillTps: prompt ? Number(prompt[3]) : null,
    generatedTokens,
    decodeMs,
    decodeTps: Number(decode[3]),
    msPerToken: generatedTokens > 0 ? decodeMs / generatedTokens : null,
    outputPreview: "",
  };
}

export function parseLlamaCppVersionOutput(text) {
  const versionLine = text
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => /^version:/i.test(line));

  if (!versionLine) return null;

  const match = versionLine.match(/^version:\s*([^\s]+)\s+\(([0-9a-f]+)\)/i);
  if (!match) return null;

  return {
    version: match[1],
    commit: match[2],
  };
}

export function parseOpenAiCompletionOutput(text) {
  const body = JSON.parse(text);
  const content = body.choices?.[0]?.text ?? body.choices?.[0]?.message?.content ?? "";
  const promptTokens = body.usage?.prompt_tokens ?? null;
  const generatedTokens = body.usage?.completion_tokens ?? 0;
  const promptTps = body.timings?.prompt_per_second ?? null;
  const decodeTps = body.timings?.predicted_per_second ?? null;
  return {
    promptTokens,
    prefillTokens: promptTokens,
    prefillMs: promptTokens != null && promptTps ? (promptTokens / promptTps) * 1000 : null,
    prefillTps: promptTps,
    generatedTokens,
    decodeMs: generatedTokens > 0 && decodeTps ? (generatedTokens / decodeTps) * 1000 : null,
    decodeTps,
    msPerToken: decodeTps ? 1000 / decodeTps : null,
    outputPreview: typeof content === "string" ? content.trim() : "",
  };
}

export function summarizeValues(values) {
  if (!values.length) return null;
  const xs = [...values].sort((a, b) => a - b);
  const sum = xs.reduce((total, value) => total + value, 0);
  const mid = Math.floor(xs.length / 2);
  const median = xs.length % 2 === 1 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2;
  const avg = sum / xs.length;
  const variance = xs.reduce((total, value) => total + ((value - avg) ** 2), 0) / xs.length;
  const percentile = (p) => {
    if (xs.length === 1) return xs[0];
    const rank = (xs.length - 1) * p;
    const lo = Math.floor(rank);
    const hi = Math.ceil(rank);
    if (lo === hi) return xs[lo];
    const frac = rank - lo;
    return xs[lo] * (1 - frac) + xs[hi] * frac;
  };
  return {
    min: xs[0],
    max: xs[xs.length - 1],
    avg,
    median,
    p95: percentile(0.95),
    stddev: Math.sqrt(variance),
    samples: values,
  };
}

export function buildComparison(zincSummary, baselineSummary) {
  if (!zincSummary || !baselineSummary) return null;
  const zincDecode = zincSummary.decode_tps?.median ?? zincSummary.decode_tps?.avg ?? null;
  const baselineDecode = baselineSummary.decode_tps?.median ?? baselineSummary.decode_tps?.avg ?? null;
  const zincPrefill = zincSummary.prefill_tps?.median ?? zincSummary.prefill_tps?.avg ?? null;
  const baselinePrefill = baselineSummary.prefill_tps?.median ?? baselineSummary.prefill_tps?.avg ?? null;
  const zincLatency = zincSummary.total_latency_ms?.median ?? zincSummary.total_latency_ms?.avg ?? null;
  const baselineLatency = baselineSummary.total_latency_ms?.median ?? baselineSummary.total_latency_ms?.avg ?? null;
  const zincEndToEnd = zincSummary.end_to_end_tps?.median ?? zincSummary.end_to_end_tps?.avg ?? null;
  const baselineEndToEnd = baselineSummary.end_to_end_tps?.median ?? baselineSummary.end_to_end_tps?.avg ?? null;
  if (!zincDecode || !baselineDecode) return null;

  const pctOfBaseline = (zincDecode / baselineDecode) * 100;
  const gapTps = zincDecode - baselineDecode;
  return {
    baseline_name: baselineSummary.name,
    zinc_prompt_tps: zincPrefill,
    baseline_prompt_tps: baselinePrefill,
    zinc_decode_tps: zincDecode,
    baseline_decode_tps: baselineDecode,
    zinc_total_latency_ms: zincLatency,
    baseline_total_latency_ms: baselineLatency,
    zinc_end_to_end_tps: zincEndToEnd,
    baseline_end_to_end_tps: baselineEndToEnd,
    pct_of_baseline: pctOfBaseline,
    delta_tps: gapTps,
    delta_pct: pctOfBaseline - 100,
    leader: gapTps >= 0 ? "zinc" : "baseline",
    prompt_pct_of_baseline: zincPrefill && baselinePrefill ? (zincPrefill / baselinePrefill) * 100 : null,
    latency_pct_of_baseline: zincLatency && baselineLatency ? (zincLatency / baselineLatency) * 100 : null,
    latency_delta_ms: zincLatency != null && baselineLatency != null ? zincLatency - baselineLatency : null,
    end_to_end_pct_of_baseline: zincEndToEnd && baselineEndToEnd ? (zincEndToEnd / baselineEndToEnd) * 100 : null,
    end_to_end_delta_tps: zincEndToEnd != null && baselineEndToEnd != null ? zincEndToEnd - baselineEndToEnd : null,
  };
}

export function mergeArtifacts(existing, incomingTargets) {
  const byId = new Map();
  const normalizeTarget = (target) => {
    const models = [...(target?.models ?? [])].sort((a, b) => a.label.localeCompare(b.label));
    return {
      ...target,
      models,
      summary: targetSummary(models),
    };
  };

  for (const target of existing?.targets ?? []) {
    byId.set(target.id, normalizeTarget(target));
  }
  for (const target of incomingTargets) {
    byId.set(target.id, normalizeTarget(target));
  }
  const targets = [...byId.values()].sort((a, b) => {
    const ia = TARGET_ORDER.indexOf(a.id);
    const ib = TARGET_ORDER.indexOf(b.id);
    return (ia === -1 ? 999 : ia) - (ib === -1 ? 999 : ib);
  });
  return {
    schema_version: 1,
    generated_at: new Date().toISOString(),
    targets,
  };
}

export function buildArtifact(targets) {
  return mergeArtifacts({ schema_version: 1, generated_at: new Date().toISOString(), targets: [] }, targets);
}

function mean(values) {
  if (!values.length) return null;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function compactMean(values) {
  return mean(values.filter((value) => value != null));
}

function targetSummary(models) {
  if (!models.length) {
    return {
      fastest_model_id: null,
      fastest_model_label: null,
      fastest_decode_tps: null,
      average_decode_tps: null,
      average_prompt_tps: null,
      average_end_to_end_tps: null,
      average_total_latency_ms: null,
      benchmarked_models: 0,
      successful_models: 0,
      compared_models: 0,
      average_pct_of_llama: null,
    };
  }

  const successful = models.filter((model) => model?.zinc?.decode_tps);
  const fastest = successful.length > 0
    ? [...successful].sort((a, b) => (b.zinc.decode_tps.median ?? b.zinc.decode_tps.avg) - (a.zinc.decode_tps.median ?? a.zinc.decode_tps.avg))[0]
    : null;
  const compared = models.filter((model) => model.comparison?.pct_of_baseline != null);

  return {
    fastest_model_id: fastest?.id ?? null,
    fastest_model_label: fastest?.label ?? null,
    fastest_decode_tps: fastest ? (fastest.zinc.decode_tps.median ?? fastest.zinc.decode_tps.avg) : null,
    benchmarked_models: models.length,
    successful_models: successful.length,
    average_decode_tps: compactMean(successful.map((model) => model.zinc.decode_tps.median ?? model.zinc.decode_tps.avg)),
    average_prompt_tps: compactMean(successful.map((model) => model.zinc.prefill_tps?.median ?? model.zinc.prefill_tps?.avg ?? null)),
    average_end_to_end_tps: compactMean(successful.map((model) => model.zinc.end_to_end_tps?.median ?? model.zinc.end_to_end_tps?.avg ?? null)),
    average_total_latency_ms: compactMean(successful.map((model) => model.zinc.total_latency_ms?.median ?? model.zinc.total_latency_ms?.avg ?? null)),
    compared_models: compared.length,
    average_pct_of_llama: mean(compared.map((model) => model.comparison.pct_of_baseline)),
  };
}

function createStats(name, rows) {
  const prefillValues = rows.map((row) => row.prefillTps).filter((value) => value != null);
  const decodeValues = rows.map((row) => row.decodeTps).filter((value) => value != null);
  const msValues = rows.map((row) => row.msPerToken).filter((value) => value != null);
  const prefillMsValues = rows.map((row) => row.prefillMs).filter((value) => value != null);
  const decodeMsValues = rows.map((row) => row.decodeMs).filter((value) => value != null);
  const totalLatencyValues = rows.map((row) => row.totalLatencyMs).filter((value) => value != null);
  const totalTpsValues = rows.map((row) => row.totalTps).filter((value) => value != null);
  return {
    name,
    prompt_tokens: rows[0]?.promptTokens ?? null,
    generated_tokens: rows[0]?.generatedTokens ?? null,
    output_preview: rows.find((row) => row.outputPreview)?.outputPreview ?? "",
    prefill_ms: summarizeValues(prefillMsValues),
    prefill_tps: summarizeValues(prefillValues),
    decode_ms: summarizeValues(decodeMsValues),
    decode_tps: summarizeValues(decodeValues),
    ms_per_token: summarizeValues(msValues),
    total_latency_ms: summarizeValues(totalLatencyValues),
    end_to_end_tps: summarizeValues(totalTpsValues),
  };
}

async function pathExists(filepath) {
  try {
    await fs.access(filepath);
    return true;
  } catch {
    return false;
  }
}

async function runShell(command, { cwd = ROOT, env = process.env, timeoutMs = DEFAULT_TIMEOUT_MS } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn("/bin/zsh", ["-lc", command], {
      cwd,
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    const timeout = setTimeout(() => {
      child.kill("SIGTERM");
    }, timeoutMs);

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString();
      if (stdout.length > MAX_CAPTURE_CHARS) stdout = stdout.slice(-MAX_CAPTURE_CHARS);
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString();
      if (stderr.length > MAX_CAPTURE_CHARS) stderr = stderr.slice(-MAX_CAPTURE_CHARS);
    });
    child.on("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    child.on("close", (code, signal) => {
      clearTimeout(timeout);
      if (code === 0) {
        resolve({ stdout, stderr, combined: `${stdout}${stderr}` });
        return;
      }
      const details = `${stdout}${stderr}`.trim();
      const error = new Error(`Command failed (${signal ?? code}): ${command}${details ? `\n${details}` : ""}`);
      error.stdout = stdout;
      error.stderr = stderr;
      reject(error);
    });
  });
}

function llamaBinaryEnv(binaryPath) {
  const libDir = path.join(path.dirname(path.dirname(binaryPath)), "lib");
  return {
    ...process.env,
    DYLD_LIBRARY_PATH: process.env.DYLD_LIBRARY_PATH ? `${libDir}:${process.env.DYLD_LIBRARY_PATH}` : libDir,
  };
}

async function captureGitProvenance(cwd = ROOT, timeoutMs = 10_000) {
  try {
    const version = await runShell("git describe --tags --always --dirty --abbrev=12", { cwd, timeoutMs });
    const commit = await runShell("git rev-parse HEAD", { cwd, timeoutMs });
    return {
      version: version.stdout.trim() || null,
      commit: commit.stdout.trim() || null,
    };
  } catch {
    return {
      version: null,
      commit: null,
    };
  }
}

async function captureRemoteGitProvenance(creds, timeoutMs = 120_000) {
  try {
    const command = rdnaRemoteCommand(
      `cd ${shellQuote(creds.workdir)} && git describe --tags --always --dirty --abbrev=12 && git rev-parse HEAD`,
      creds,
    );
    const result = await runShell(command, { cwd: ROOT, timeoutMs });
    const [version = null, commit = null] = result.stdout
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter(Boolean);
    return {
      version,
      commit,
    };
  } catch {
    return {
      version: null,
      commit: null,
    };
  }
}

async function captureLlamaCppProvenance(binaryPath, { cwd = ROOT, env = process.env, timeoutMs = 10_000 } = {}) {
  if (!binaryPath) {
    return {
      binary: null,
      version: null,
      commit: null,
    };
  }

  try {
    const result = await runShell(`${shellQuote(binaryPath)} --version`, { cwd, env, timeoutMs });
    const parsed = parseLlamaCppVersionOutput(result.combined);
    return {
      binary: path.basename(binaryPath),
      version: parsed?.version ?? null,
      commit: parsed?.commit ?? null,
    };
  } catch {
    return {
      binary: path.basename(binaryPath),
      version: null,
      commit: null,
    };
  }
}

async function captureRemoteLlamaCppProvenance(binaryPath, creds, timeoutMs = 120_000) {
  if (!binaryPath) {
    return {
      binary: null,
      version: null,
      commit: null,
    };
  }

  try {
    const command = rdnaRemoteCommand(`${shellQuote(binaryPath)} --version`, creds);
    const result = await runShell(command, { cwd: ROOT, timeoutMs });
    const parsed = parseLlamaCppVersionOutput(result.combined);
    return {
      binary: path.basename(binaryPath),
      version: parsed?.version ?? null,
      commit: parsed?.commit ?? null,
    };
  } catch {
    return {
      binary: path.basename(binaryPath),
      version: null,
      commit: null,
    };
  }
}

function shouldUseManagedModelId(caseDef) {
  if (!caseDef.model_id) return false;
  if (!caseDef.model_path) return true;
  const normalizedRoot = path.normalize(DEFAULT_LOCAL_MODEL_ROOT + path.sep);
  const normalizedPath = path.normalize(caseDef.model_path);
  return normalizedPath.startsWith(normalizedRoot);
}

function managedModelPullCommand(modelId) {
  return `./zig-out/bin/zinc model pull ${shellQuote(modelId)}`;
}

export function localZincCommand(caseDef) {
  const parts = ["./zig-out/bin/zinc"];
  if (caseDef.prompt_mode === "chat") parts.push("--chat");
  parts.push("-n", String(caseDef.max_tokens));
  if (shouldUseManagedModelId(caseDef)) {
    parts.push("--model-id", caseDef.model_id);
  } else {
    parts.push("-m", shellQuote(caseDef.model_path));
  }
  parts.push("--prompt", shellQuote(caseDef.prompt));
  return parts.join(" ");
}

function localLlamaCliCommand(caseDef, llamaCli) {
  const parts = [shellQuote(llamaCli), "-m", shellQuote(caseDef.model_path), "-n", String(caseDef.max_tokens), "-p", shellQuote(caseDef.prompt), "--temp", "0", "-ngl", "999"];
  if (caseDef.prompt_mode === "chat") parts.push("-cnv");
  return parts.join(" ");
}

async function pickOpenPort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      if (!address || typeof address === "string") {
        server.close(() => reject(new Error("Could not determine ephemeral port")));
        return;
      }
      const { port } = address;
      server.close((error) => {
        if (error) reject(error);
        else resolve(port);
      });
    });
  });
}

function withTimeout(promise, timeoutMs, label) {
  return new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`${label} timed out after ${timeoutMs} ms`)), timeoutMs);
    promise.then(
      (value) => {
        clearTimeout(timer);
        resolve(value);
      },
      (error) => {
        clearTimeout(timer);
        reject(error);
      },
    );
  });
}

async function httpJson(url, payload, timeoutMs) {
  const response = await withTimeout(fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  }), timeoutMs, `POST ${url}`);
  const text = await response.text();
  if (!response.ok) {
    throw new Error(`HTTP ${response.status} from ${url}: ${text.slice(0, 400)}`);
  }
  return JSON.parse(text);
}

function buildOpenAiPayload(caseDef) {
  if (caseDef.prompt_mode === "chat") {
    return {
      model: "q",
      messages: [{ role: "user", content: caseDef.prompt }],
      max_tokens: caseDef.max_tokens,
      temperature: 0,
      stream: false,
    };
  }
  return {
    model: "q",
    prompt: caseDef.prompt,
    max_tokens: caseDef.max_tokens,
    temperature: 0,
    stream: false,
  };
}

async function waitForHealthyServer(baseUrl, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    try {
      const response = await fetch(`${baseUrl}/health`);
      if (response.ok) return;
    } catch {
      // Server still starting.
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  throw new Error(`Server did not become healthy within ${timeoutMs} ms`);
}

async function stopProcess(child, graceMs = 5000) {
  if (!child || child.exitCode != null) return;
  child.kill("SIGINT");
  try {
    await withTimeout(new Promise((resolve) => child.once("close", resolve)), graceMs, "process shutdown");
  } catch {
    child.kill("SIGKILL");
  }
}

function detectLocalLlamaServer(args) {
  return resolveLocalLlamaServer(args);
}

async function launchLocalLlamaServer(caseDef, serverPath, timeoutMs) {
  const port = await pickOpenPort();
  const baseUrl = `http://127.0.0.1:${port}/v1`;
  const serverEnv = llamaBinaryEnv(serverPath);
  const args = [
    "--host", "127.0.0.1",
    "--port", String(port),
    "-m", caseDef.model_path,
    "-ngl", "999",
    "--metrics",
    "--ctx-size", "4096",
    "--parallel", "1",
  ];

  const child = spawn(serverPath, args, {
    cwd: ROOT,
    env: serverEnv,
    stdio: ["ignore", "pipe", "pipe"],
  });

  let combinedLog = "";
  const appendLog = (chunk) => {
    combinedLog += chunk.toString();
    if (combinedLog.length > 16000) {
      combinedLog = combinedLog.slice(-16000);
    }
  };
  child.stdout.on("data", appendLog);
  child.stderr.on("data", appendLog);

  const exitPromise = new Promise((_, reject) => {
    child.once("exit", (code, signal) => {
      reject(new Error(`llama-server exited early (${signal ?? code})\n${combinedLog}`));
    });
  });

  try {
    await Promise.race([
      waitForHealthyServer(baseUrl.replace(/\/v1$/, ""), Math.min(timeoutMs, 300000)),
      exitPromise,
    ]);
  } catch (error) {
    await stopProcess(child, 1000);
    throw error;
  }

  return {
    baseUrl,
    child,
    logTail: () => combinedLog,
  };
}

export function detectRdnaServerStartupFailure(logText) {
  if (!logText) return null;

  const patterns = [
    /unknown model architecture:\s*'[^']+'/i,
    /error loading model architecture:/i,
    /srv load_model:\s*failed to load model/i,
    /main:\s*exiting due to model loading error/i,
    /error loading model:/i,
  ];

  for (const pattern of patterns) {
    const match = logText.match(pattern);
    if (match) return match[0];
  }

  return null;
}

async function tailRdnaLog(creds, logPath, lines = 80) {
  if (!logPath) return "";
  const command = rdnaRemoteCommand(
    `tail -n ${Math.max(1, lines)} ${shellQuote(logPath)} 2>/dev/null || true`,
    creds,
  );
  const result = await runShell(command, { cwd: ROOT, timeoutMs: 10000 });
  return result.stdout.trim();
}

async function waitForHealthyRdnaServer(creds, port, logPath, timeoutMs) {
  const deadline = Date.now() + timeoutMs;
  let lastLogTail = "";
  while (Date.now() < deadline) {
    try {
      await runShell(
        rdnaRemoteCommand(`curl -fsS http://127.0.0.1:${port}/health >/dev/null`, creds),
        { cwd: ROOT, timeoutMs: 10000 },
      );
      return;
    } catch {
      // Remote server may still be starting, or it may already have failed.
      lastLogTail = await tailRdnaLog(creds, logPath);
      const startupFailure = detectRdnaServerStartupFailure(lastLogTail);
      if (startupFailure) {
        throw new Error(`Remote llama-server failed to start: ${startupFailure}`);
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  const detail = lastLogTail ? `\n${lastLogTail}` : "";
  throw new Error(`Remote server on port ${port} did not become healthy within ${timeoutMs} ms${detail}`);
}

async function stopRdnaLlamaServer(creds, port) {
  if (!port) return;
  try {
    await runShell(
      rdnaRemoteCommand(`pkill -f 'llama-server --host 127.0.0.1 --port ${port}' || true`, creds),
      { cwd: ROOT, timeoutMs: 30000 },
    );
  } catch {
    // Best effort cleanup.
  }
}

async function launchRdnaLlamaServer(caseDef, creds, serverPath, timeoutMs) {
  const port = await pickOpenPort();
  const logPath = `/tmp/zinc-rdna-llama-${port}.log`;
  const cmd = [
    serverPath,
    "--host", "127.0.0.1",
    "--port", String(port),
    "-m", caseDef.model_path,
    "--device", "Vulkan0",
    "-ngl", "999",
    "--metrics",
    "--ctx-size", "4096",
    "--parallel", "1",
    "-ctk", "q8_0",
    "-ctv", "q8_0",
    "-b", "4096",
    "-ub", "1024",
    "--flash-attn", "on",
  ];
  const launchScript = [
    "import os, subprocess",
    `cmd = ${JSON.stringify(cmd)}`,
    `env = dict(os.environ, RADV_PERFTEST="coop_matrix")`,
    `log = open(${JSON.stringify(logPath)}, "ab", buffering=0)`,
    `subprocess.Popen(cmd, cwd=${JSON.stringify(creds.workdir)}, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)`,
    'print("started")',
  ].join("; ");
  const remote = `python3 -c ${shellQuote(launchScript)}`;

  await runShell(rdnaRemoteCommand(remote, creds), { cwd: ROOT, timeoutMs: 120000 });

  try {
    await waitForHealthyRdnaServer(creds, port, logPath, Math.min(timeoutMs, 300000));
  } catch (error) {
    await stopRdnaLlamaServer(creds, port);
    throw error;
  }

  return { port, logPath };
}

async function runOpenAiSeries({ label, warmupRuns, runs, baseUrl, caseDef, timeoutMs }) {
  const endpoint = caseDef.prompt_mode === "chat" ? `${baseUrl}/chat/completions` : `${baseUrl}/completions`;
  const payload = buildOpenAiPayload(caseDef);
  const measured = [];

  for (let i = 0; i < warmupRuns; i += 1) {
    console.log(`  warmup ${i + 1}/${warmupRuns}: ${label}`);
    const body = await httpJson(endpoint, payload, timeoutMs);
    parseOpenAiCompletionOutput(JSON.stringify(body));
  }

  for (let i = 0; i < runs; i += 1) {
    console.log(`  run ${i + 1}/${runs}: ${label}`);
    const started = performance.now();
    const body = await httpJson(endpoint, payload, timeoutMs);
    const ended = performance.now();
    const parsed = parseOpenAiCompletionOutput(JSON.stringify(body));
    parsed.totalLatencyMs = ended - started;
    const totalTokens = (parsed.promptTokens ?? 0) + (parsed.generatedTokens ?? 0);
    parsed.totalTps = totalTokens > 0 ? totalTokens / Math.max((ended - started) / 1000, 1e-9) : null;
    measured.push(parsed);
  }

  return measured;
}

async function runRdnaOpenAiSeries({ label, warmupRuns, runs, creds, port, caseDef, timeoutMs }) {
  const endpoint = caseDef.prompt_mode === "chat" ? `/v1/chat/completions` : `/v1/completions`;
  const payload = buildOpenAiPayload(caseDef);
  const command = rdnaRemoteCommand(
    `curl -sS http://127.0.0.1:${port}${endpoint} -H 'Content-Type: application/json' -d ${shellQuote(JSON.stringify(payload))}`,
    creds,
  );
  const measured = [];

  for (let i = 0; i < warmupRuns; i += 1) {
    console.log(`  warmup ${i + 1}/${warmupRuns}: ${label}`);
    const result = await runShell(command, { cwd: ROOT, timeoutMs });
    parseOpenAiCompletionOutput(result.stdout);
  }

  for (let i = 0; i < runs; i += 1) {
    console.log(`  run ${i + 1}/${runs}: ${label}`);
    const started = performance.now();
    const result = await runShell(command, { cwd: ROOT, timeoutMs });
    const ended = performance.now();
    const parsed = parseOpenAiCompletionOutput(result.stdout);
    parsed.totalLatencyMs = ended - started;
    const totalTokens = (parsed.promptTokens ?? 0) + (parsed.generatedTokens ?? 0);
    parsed.totalTps = totalTokens > 0 ? totalTokens / Math.max((ended - started) / 1000, 1e-9) : null;
    measured.push(parsed);
  }

  return measured;
}

function rdnaRemoteCommand(remoteCommand, creds) {
  const sshArgs = [];
  if (process.env.ZINC_SSH_STRICT === "no") {
    sshArgs.push("-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null");
  }
  sshArgs.push("-p", creds.port, `${creds.user}@${creds.host}`, shellQuote(remoteCommand));
  return `ssh ${sshArgs.join(" ")}`;
}

function rdnaDetachedCommand(remoteCommand, creds) {
  const sshArgs = ["-f"];
  if (process.env.ZINC_SSH_STRICT === "no") {
    sshArgs.push("-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null");
  }
  sshArgs.push("-p", creds.port, `${creds.user}@${creds.host}`, shellQuote(remoteCommand));
  return `ssh ${sshArgs.join(" ")}`;
}

function rdnaSshTransport(creds) {
  const parts = ["ssh"];
  if (process.env.ZINC_SSH_STRICT === "no") {
    parts.push("-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null");
  }
  parts.push("-p", creds.port);
  return parts.join(" ");
}

export function rdnaZincCommand(caseDef, creds) {
  const parts = [
    `cd ${shellQuote(creds.workdir)}`,
    "&&",
    "RADV_PERFTEST=coop_matrix",
    "./zig-out/bin/zinc",
    "-n",
    String(caseDef.max_tokens),
    "-m",
    shellQuote(caseDef.model_path),
  ];
  if (caseDef.prompt_mode === "chat") parts.push("--chat");
  parts.push("--prompt", shellQuote(caseDef.prompt));
  return rdnaRemoteCommand(parts.join(" "), creds);
}

function rdnaLlamaCommand(caseDef, creds, llamaCliPath) {
  const parts = [
    "RADV_PERFTEST=coop_matrix",
    shellQuote(llamaCliPath),
    "-m",
    shellQuote(caseDef.model_path),
    "-n",
    String(caseDef.max_tokens),
    "-p",
    shellQuote(caseDef.prompt),
    "--temp",
    "0",
    "-ngl",
    "999",
    "--device",
    "Vulkan0",
    "--flash-attn",
    "on",
    "-ctk",
    "q8_0",
    "-ctv",
    "q8_0",
    "-b",
    "4096",
    "-ub",
    "1024",
  ];
  if (caseDef.prompt_mode === "chat") parts.push("-cnv");
  return rdnaRemoteCommand(parts.join(" "), creds);
}

async function detectRdnaLlamaCliPath(creds) {
  if (process.env.ZINC_LLAMA_CLI_REMOTE) return process.env.ZINC_LLAMA_CLI_REMOTE;
  const command = rdnaRemoteCommand(
    "which llama-cli || command -v llama-cli || find /root/llama.cpp /root/workspace/llama.cpp -name llama-cli -type f 2>/dev/null | head -n 1",
    creds,
  );
  const result = await runShell(command, { cwd: ROOT, timeoutMs: 120000 });
  const candidate = result.stdout.split(/\r?\n/).map((line) => line.trim()).find(Boolean);
  return candidate || null;
}

async function detectRdnaLlamaServerPath(creds) {
  if (process.env.ZINC_LLAMA_SERVER_REMOTE) return process.env.ZINC_LLAMA_SERVER_REMOTE;
  const command = rdnaRemoteCommand(
    "which llama-server || command -v llama-server || find /root/llama.cpp /root/workspace/llama.cpp -name llama-server -type f 2>/dev/null | head -n 1",
    creds,
  );
  const result = await runShell(command, { cwd: ROOT, timeoutMs: 120000 });
  const candidate = result.stdout.split(/\r?\n/).map((line) => line.trim()).find(Boolean);
  return candidate || null;
}

async function runSeries({ label, warmupRuns, runs, command, parser, cwd, timeoutMs }) {
  const measured = [];
  for (let i = 0; i < warmupRuns; i += 1) {
    console.log(`  warmup ${i + 1}/${warmupRuns}: ${label}`);
    const result = await runShell(command, { cwd, timeoutMs });
    parser(result.combined);
  }
  for (let i = 0; i < runs; i += 1) {
    console.log(`  run ${i + 1}/${runs}: ${label}`);
    const started = performance.now();
    const result = await runShell(command, { cwd, timeoutMs });
    const ended = performance.now();
    const parsed = parser(result.combined);
    parsed.totalLatencyMs = ended - started;
    const totalTokens = (parsed.promptTokens ?? 0) + (parsed.generatedTokens ?? 0);
    parsed.totalTps = totalTokens > 0 ? totalTokens / Math.max((ended - started) / 1000, 1e-9) : null;
    measured.push(parsed);
  }
  return measured;
}

async function detectMetalMachine() {
  try {
    const result = await runShell("system_profiler SPHardwareDataType SPDisplaysDataType -json", { timeoutMs: 120000 });
    const parsed = JSON.parse(result.stdout);
    const hardware = parsed.SPHardwareDataType?.[0] ?? {};
    const displays = parsed.SPDisplaysDataType?.[0]?.spdisplays_ndrvs?.[0] ?? {};
    const chip = hardware.chip_type ?? null;
    return {
      label: hardware.machine_name ?? hardware.model_name ?? "Apple Silicon Mac",
      machine_model: hardware.machine_model ?? null,
      chip,
      memory: hardware.physical_memory ?? null,
      gpu: chip ? `${chip} GPU` : displays.sppci_model ?? displays._name ?? "Apple GPU",
      os: `${os.type()} ${os.release()}`,
    };
  } catch {
    return {
      label: "Apple Silicon Mac",
      machine_model: null,
      chip: null,
      memory: null,
      gpu: "Apple GPU",
      os: `${os.type()} ${os.release()}`,
    };
  }
}

export function defaultMetalCases(modelRoot) {
  return [
    {
      id: "gpt-oss-20b-q4k-m",
      model_id: "gpt-oss-20b-q4k-m",
      label: "OpenAI GPT-OSS 20B Q4_K_M",
      family: "GPT-OSS",
      quant: "Q4_K_M",
      model_path: modelPath(modelRoot, "gpt-oss-20b-q4k-m"),
      prompt_mode: "chat",
      prompt: "What is the capital of France? Answer in one word.",
      max_tokens: defaultMaxTokensForModelId("gpt-oss-20b-q4k-m"),
      notes: ["Harmony chat-template path on local Metal", "Uses a larger decode budget so the final answer channel is reached"],
    },
    {
      id: "gemma4-12b-q4k-m",
      model_id: "gemma4-12b-q4k-m",
      label: "Gemma 4 12B Q4_K_M",
      family: "Gemma 4",
      quant: "Q4_K_M",
      model_path: modelPath(modelRoot, "gemma4-12b-q4k-m"),
      prompt_mode: "chat",
      prompt: "What is the capital of France? Answer in one word.",
      max_tokens: defaultMaxTokensForModelId("gemma4-12b-q4k-m"),
      notes: ["ISWA / MoE Metal path"],
    },
    {
      id: "gemma4-31b-q4k-m",
      model_id: "gemma4-31b-q4k-m",
      label: "Gemma 4 31B Q4_K_M",
      family: "Gemma 4",
      quant: "Q4_K_M",
      model_path: modelPath(modelRoot, "gemma4-31b-q4k-m"),
      prompt_mode: "chat",
      prompt: "What is the capital of France? Answer in one word.",
      max_tokens: defaultMaxTokensForModelId("gemma4-31b-q4k-m"),
      notes: ["Large-model Metal path"],
    },
    {
      id: "qwen3-8b-q4k-m",
      model_id: "qwen3-8b-q4k-m",
      label: "Qwen 3 8B Q4_K_M",
      family: "Qwen 3",
      quant: "Q4_K_M",
      model_path: modelPath(modelRoot, "qwen3-8b-q4k-m"),
      prompt_mode: "raw",
      prompt: "The capital of France is",
      max_tokens: defaultMaxTokensForModelId("qwen3-8b-q4k-m"),
      notes: ["Raw decode path to avoid visible think blocks in CLI output"],
    },
    {
      id: "qwen35-35b-a3b-q4k-xl",
      model_id: "qwen35-35b-a3b-q4k-xl",
      label: "Qwen 3.5 35B A3B UD Q4_K_XL",
      family: "Qwen 3.5",
      quant: "Q4_K_XL",
      model_path: modelPath(modelRoot, "qwen35-35b-a3b-q4k-xl"),
      prompt_mode: "raw",
      prompt: defaultPromptForModelId("qwen35-35b-a3b-q4k-xl"),
      max_tokens: defaultMaxTokensForModelId("qwen35-35b-a3b-q4k-xl"),
      notes: ["Managed-cache local Qwen 3.5 case on Apple Silicon"],
    },
    {
      id: "qwen36-35b-a3b-q4k-xl",
      model_id: "qwen36-35b-a3b-q4k-xl",
      label: "Qwen 3.6 35B A3B UD Q4_K_XL",
      family: "Qwen 3.6",
      quant: "Q4_K_XL",
      model_path: modelPath(modelRoot, "qwen36-35b-a3b-q4k-xl"),
      prompt_mode: "raw",
      prompt: defaultPromptForModelId("qwen36-35b-a3b-q4k-xl"),
      max_tokens: defaultMaxTokensForModelId("qwen36-35b-a3b-q4k-xl"),
      notes: ["Managed-cache local Qwen 3.6 case on Apple Silicon"],
    },
  ];
}

function titleFromId(id) {
  return id
    .split("-")
    .map((chunk) => (chunk.length <= 3 ? chunk.toUpperCase() : `${chunk[0].toUpperCase()}${chunk.slice(1)}`))
    .join(" ");
}

export function canonicalModelIdFromPath(modelFile) {
  const base = path.basename(modelFile);
  if (base === "model.gguf") {
    const parent = path.basename(path.dirname(modelFile));
    if (parent && parent !== "models") return parent.toLowerCase();
  }

  return path.basename(modelFile, path.extname(modelFile))
    .toLowerCase()
    .replace(/^openai[_-]/, "")
    .replace(/^bartowski[_-]/, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .replace(/^qwen3-6-/, "qwen36-")
    .replace(/^qwen-3-6-/, "qwen36-")
    .replace(/^qwen3-5-/, "qwen35-")
    .replace(/^qwen-3-5-/, "qwen35-")
    .replace(/^qwen-3-/, "qwen3-")
    .replace(/-ud-(q[0-9].*)$/, "-$1")
    .replace(/-q([0-9])-k-([a-z0-9]+)/g, "-q$1k-$2")
    .replace(/-q([0-9])-k$/g, "-q$1k");
}

export function guessFamily(id) {
  if (id.startsWith("gemma4")) return "Gemma 4";
  if (id.startsWith("gemma")) return "Gemma";
  if (id.startsWith("gpt-oss")) return "GPT-OSS";
  if (id.startsWith("qwen36")) return "Qwen 3.6";
  if (id.startsWith("qwen35")) return "Qwen 3.5";
  if (id.startsWith("qwen3")) return "Qwen 3";
  if (id.startsWith("qwen")) return "Qwen";
  return titleFromId(id.split("-q")[0] ?? id);
}

function guessQuant(id) {
  const match = id.match(/-(q[0-9a-z_]+(?:-[0-9a-z_]+)?)/i);
  return match ? match[1].toUpperCase().replace(/-/g, "_") : "GGUF";
}

export function prefersChatPrompt(id) {
  return id.startsWith("gemma") || id.startsWith("gpt-oss");
}

export function defaultPromptForModelId(id) {
  return prefersChatPrompt(id)
    ? "What is the capital of France? Answer in one word."
    : "The capital of France is";
}

export function defaultMaxTokensForModelId(id) {
  return id.startsWith("gpt-oss") ? 48 : 8;
}

const BENCHMARK_CONTEXT_SENTENCE =
  "Benchmark context only. alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu.";

function repeatContext(lines) {
  return Array.from({ length: lines }, () => BENCHMARK_CONTEXT_SENTENCE).join(" ");
}

function contextualizePrompt(promptMode, basePrompt, variant) {
  if (variant === "core") return basePrompt;

  if (promptMode === "chat" && variant === "decode-extended") {
    return "Write six short bullet points explaining why local LLM benchmark reports should separate prefill throughput from decode throughput.";
  }

  if (promptMode === "raw" && variant === "decode-extended") {
    return "LLM benchmark reports should separate prompt processing from token generation because";
  }

  const context = variant === "context-medium"
    ? [
      "Reference notes:",
      "Paris is the capital of France.",
      "Rome is the capital of Italy.",
      "Madrid is the capital of Spain.",
      "Use only the notes above.",
    ].join("\n")
    : [
      "Long reference packet for benchmark purposes only:",
      repeatContext(6),
      "Important fact near the end: Paris is the capital of France.",
      "Ignore unrelated filler and answer from the reference packet.",
    ].join("\n\n");

  if (promptMode === "chat") {
    return `Read the context below and answer the final question briefly.\n\n${context}\n\nQuestion: What is the capital of France? Answer in one word.`;
  }
  return `${context}\n\nBased on the reference above, the capital of France is`;
}

function scenarioMaxTokens(modelId, scenarioId) {
  const base = defaultMaxTokensForModelId(modelId);
  if (scenarioId === "decode-extended") {
    if (modelId.startsWith("gpt-oss")) return 96;
    return Math.max(base, 32);
  }
  return base;
}

export function defaultScenarioDefsForModel(modelId, promptMode, basePrompt) {
  return [
    {
      id: "core",
      label: "Core Prompt",
      short_label: "Core",
      prompt_mode: promptMode,
      prompt: contextualizePrompt(promptMode, basePrompt, "core"),
      max_tokens: scenarioMaxTokens(modelId, "core"),
      notes: ["Single-turn reference prompt"],
    },
    {
      id: "context-medium",
      label: "Context Retrieval",
      short_label: "Context",
      prompt_mode: promptMode,
      prompt: contextualizePrompt(promptMode, basePrompt, "context-medium"),
      max_tokens: scenarioMaxTokens(modelId, "context-medium"),
      notes: ["Short retrieval workload with explicit context"],
    },
    {
      id: "context-long",
      label: "Long Context Retrieval",
      short_label: "Long ctx",
      prompt_mode: promptMode,
      prompt: contextualizePrompt(promptMode, basePrompt, "context-long"),
      max_tokens: scenarioMaxTokens(modelId, "context-long"),
      notes: ["Long prompt workload with the answer embedded near the end"],
    },
    {
      id: "decode-extended",
      label: "Long Decode",
      short_label: "Long decode",
      prompt_mode: promptMode,
      prompt: contextualizePrompt(promptMode, basePrompt, "decode-extended"),
      max_tokens: scenarioMaxTokens(modelId, "decode-extended"),
      notes: ["Longer generation target instead of a short factual completion"],
    },
  ];
}

export function buildMeasurementPhases(modelId, promptMode, basePrompt) {
  const scenarios = defaultScenarioDefsForModel(modelId, promptMode, basePrompt);
  return [
    ...scenarios.map((scenarioDef) => ({ phase: "zinc", scenarioDef })),
    ...scenarios.map((scenarioDef) => ({ phase: "baseline", scenarioDef })),
  ];
}

async function discoverMetalCases(modelRoot) {
  let entries;
  try {
    entries = await fs.readdir(modelRoot, { withFileTypes: true });
  } catch {
    return [];
  }

  const discovered = [];
  for (const entry of entries) {
    if (!entry.isDirectory()) continue;
    const id = entry.name;
    const modelFile = modelPath(modelRoot, id);
    if (!(await pathExists(modelFile))) continue;
    const chatPrompt = prefersChatPrompt(id);
    discovered.push({
      id,
      model_id: id,
      label: titleFromId(id),
      family: guessFamily(id),
      quant: guessQuant(id),
      model_path: modelFile,
      prompt_mode: chatPrompt ? "chat" : "raw",
      prompt: defaultPromptForModelId(id),
      max_tokens: defaultMaxTokensForModelId(id),
      notes: ["Auto-discovered from the local model cache"],
    });
  }

  return discovered;
}

function defaultRdnaCases(modelRoot) {
  return [
    {
      id: "qwen35-35b-a3b-q4k-xl",
      label: "Qwen 3.5 35B A3B UD Q4_K_XL",
      family: "Qwen 3.5",
      quant: "Q4_K_XL",
      model_path: path.join(modelRoot, "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf"),
      prompt_mode: "raw",
      prompt: "The capital of France is",
      max_tokens: 128,
      notes: ["RDNA4 flagship comparison against llama.cpp server"],
    },
  ];
}

async function discoverRdnaCases(modelRoot, creds) {
  const command = rdnaRemoteCommand(`find ${shellQuote(modelRoot)} -maxdepth 2 \\( -name '*.gguf' -o -name 'model.gguf' \\) | sort`, creds);
  let result;
  try {
    result = await runShell(command, { cwd: ROOT, timeoutMs: 120000 });
  } catch {
    return [];
  }

  return result.stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((modelFile) => {
      const id = canonicalModelIdFromPath(modelFile);
      const chatPrompt = prefersChatPrompt(id);
      return {
        id,
        label: titleFromId(id),
        family: guessFamily(id),
        quant: guessQuant(id),
        model_path: modelFile,
        prompt_mode: chatPrompt ? "chat" : "raw",
        prompt: defaultPromptForModelId(id),
        max_tokens: defaultMaxTokensForModelId(id),
        notes: ["Auto-discovered on the RDNA benchmark node"],
      };
    });
}

async function prepareLocalBuild(args) {
  if (!args.buildLocal) return;
  console.log("Preparing local ReleaseFast build...");
  const buildCmd = `env ZIG_GLOBAL_CACHE_DIR=${shellQuote(args.localCacheDir)} zig build -Doptimize=ReleaseFast`;
  await runShell(buildCmd, { cwd: ROOT, timeoutMs: 30 * 60 * 1000 });
}

async function ensureManagedMetalModels(args, cases) {
  if (!args.metalPullMissing) return;
  for (const entry of cases) {
    if (args.models && !args.models.has(entry.id)) continue;
    if (!entry.model_id) continue;
    if (await pathExists(entry.model_path)) continue;
    console.log(`[metal] pulling missing managed model ${entry.model_id} into ${DEFAULT_LOCAL_MODEL_ROOT}...`);
    await runShell(managedModelPullCommand(entry.model_id), {
      cwd: ROOT,
      timeoutMs: 6 * 60 * 60 * 1000,
    });
  }
}

async function prepareRdna(args, creds) {
  if (args.rdnaSync) {
    console.log("Syncing current repo to RDNA node...");
    const rsyncCmd = [
      "rsync -az --delete",
      "--exclude '.zig-cache'",
      "--exclude 'zig-out'",
      "--exclude 'node_modules'",
      "--exclude '.DS_Store'",
      "--exclude 'site'",
      `-e ${shellQuote(rdnaSshTransport(creds))}`,
      `${shellQuote(`${ROOT}/`)}`,
      `${creds.user}@${creds.host}:${shellQuote(`${creds.workdir}/`)}`,
    ].join(" ");
    await runShell(rsyncCmd, { cwd: ROOT, timeoutMs: 60 * 60 * 1000 });
  }

  if (args.rdnaBuild) {
    console.log("Building ReleaseFast on RDNA node...");
    const remote = `cd ${shellQuote(creds.workdir)} && zig build -Doptimize=ReleaseFast`;
    await runShell(rdnaRemoteCommand(remote, creds), { cwd: ROOT, timeoutMs: 60 * 60 * 1000 });
  }

  if (args.rdnaStartLlama) {
    console.log("Ensuring llama-server is running on RDNA node...");
    const remote = "systemctl start llama-server && sleep 10";
    await runShell(rdnaRemoteCommand(remote, creds), { cwd: ROOT, timeoutMs: 120000 });
  }
}

function buildScenarioCase(entry, scenarioDef) {
  return {
    ...entry,
    prompt_mode: scenarioDef.prompt_mode ?? entry.prompt_mode,
    prompt: scenarioDef.prompt,
    max_tokens: scenarioDef.max_tokens,
  };
}

function scenarioResultPayload(entry, scenarioDef, zinc, baseline) {
  return {
    id: scenarioDef.id,
    label: scenarioDef.label,
    short_label: scenarioDef.short_label ?? scenarioDef.label,
    prompt_mode: scenarioDef.prompt_mode ?? entry.prompt_mode,
    prompt: scenarioDef.prompt,
    max_tokens: scenarioDef.max_tokens,
    notes: scenarioDef.notes ?? [],
    zinc,
    baseline: baseline?.decode_tps ? baseline : baseline,
    comparison: baseline?.decode_tps ? buildComparison(zinc, baseline) : null,
  };
}

function primaryScenarioSummary(entry, scenarios) {
  const primary = scenarios.find((scenario) => scenario.id === "core") ?? scenarios[0];
  return {
    id: entry.id,
    label: entry.label,
    family: entry.family,
    quant: entry.quant,
    prompt_mode: primary?.prompt_mode ?? entry.prompt_mode,
    prompt: primary?.prompt ?? entry.prompt,
    max_tokens: primary?.max_tokens ?? entry.max_tokens,
    notes: entry.notes,
    zinc: primary?.zinc ?? null,
    baseline: primary?.baseline ?? null,
    comparison: primary?.comparison ?? null,
    scenarios,
  };
}

async function runMetalTarget(args) {
  await prepareLocalBuild(args);
  const machine = await detectMetalMachine();
  const knownCases = defaultMetalCases(args.localModelRoot);
  const discoveredCases = args.discoverModels ? await discoverMetalCases(args.localModelRoot) : [];
  const mergedCases = new Map();
  for (const entry of [...discoveredCases, ...knownCases]) {
    mergedCases.set(entry.id, entry);
  }
  const cases = [...mergedCases.values()];
  await ensureManagedMetalModels(args, cases);
  const filtered = [];
  for (const entry of cases) {
    if (args.models && !args.models.has(entry.id)) continue;
    if (!shouldIncludeInPublishedBenchmarks(entry.id, args.models)) continue;
    if (await pathExists(entry.model_path)) {
      filtered.push(entry);
      continue;
    }
    if (entry.model_id) {
      console.log(
        `[metal] skipping ${entry.id}: missing from managed cache (${entry.model_path}). Install with ${managedModelPullCommand(entry.model_id)}`,
      );
    }
  }

  const llamaCli = args.llamaCli || whichLocalBinary("llama-cli");
  const resolvedLlamaServer = detectLocalLlamaServer(args);
  const llamaServer = resolvedLlamaServer && await pathExists(resolvedLlamaServer) ? resolvedLlamaServer : null;
  const baselineBinary = llamaServer || llamaCli;
  const zincProvenance = await captureGitProvenance(ROOT);
  const llamaCppProvenance = await captureLlamaCppProvenance(baselineBinary, {
    cwd: ROOT,
    env: baselineBinary ? llamaBinaryEnv(baselineBinary) : process.env,
  });
  const models = [];

  for (const entry of filtered) {
    console.log(`[metal] ${entry.id}`);
    const phases = buildMeasurementPhases(entry.id, entry.prompt_mode, entry.prompt);
    const scenarioDefs = phases.filter((phase) => phase.phase === "zinc").map((phase) => phase.scenarioDef);
    const zincByScenario = new Map();
    const baselineByScenario = new Map();
    let launchedServer = null;
    let serverLaunchError = null;
    let triedLaunchingServer = false;
    try {
      for (const phase of phases) {
        const scenarioDef = phase.scenarioDef;
        const caseDef = buildScenarioCase(entry, scenarioDef);
        if (phase.phase === "zinc") {
          let zinc = null;
          try {
            const zincRows = await runSeries({
              label: `zinc ${entry.id} ${scenarioDef.id}`,
              warmupRuns: args.warmupRuns,
              runs: args.runs,
              command: localZincCommand(caseDef),
              parser: parseZincCliOutput,
              cwd: ROOT,
              timeoutMs: args.timeoutMs,
            });
            zinc = createStats("ZINC", zincRows);
          } catch (error) {
            zinc = {
              name: "ZINC",
              unavailable_reason: `ZINC run failed: ${error.message.split("\n")[0]}`,
            };
          }
          zincByScenario.set(scenarioDef.id, zinc);
          continue;
        }

        if (!triedLaunchingServer && llamaServer) {
          triedLaunchingServer = true;
          try {
            launchedServer = await launchLocalLlamaServer(entry, llamaServer, args.timeoutMs);
          } catch (error) {
            serverLaunchError = error;
          }
        }

        let baseline = null;
        if (launchedServer) {
          try {
            const baselineRows = await runOpenAiSeries({
              label: `llama.cpp ${entry.id} ${scenarioDef.id}`,
              warmupRuns: args.warmupRuns,
              runs: args.runs,
              baseUrl: launchedServer.baseUrl,
              caseDef,
              timeoutMs: args.timeoutMs,
            });
            baseline = createStats("llama.cpp", baselineRows);
          } catch (error) {
            baseline = {
              name: "llama.cpp",
              unavailable_reason: `Local baseline failed: ${error.message.split("\n")[0]}`,
            };
          }
        } else if (llamaCli) {
          try {
            const baselineRows = await runSeries({
              label: `llama.cpp ${entry.id} ${scenarioDef.id}`,
              warmupRuns: args.warmupRuns,
              runs: args.runs,
              command: localLlamaCliCommand(caseDef, llamaCli),
              parser: parseLlamaCliOutput,
              cwd: ROOT,
              timeoutMs: args.timeoutMs,
            });
            baseline = createStats("llama.cpp", baselineRows);
          } catch (error) {
            baseline = {
              name: "llama.cpp",
              unavailable_reason: `Local baseline failed: ${error.message.split("\n")[0]}`,
            };
          }
        } else if (serverLaunchError) {
          baseline = {
            name: "llama.cpp",
            unavailable_reason: `Local baseline failed: ${serverLaunchError.message.split("\n")[0]}`,
          };
        } else {
          baseline = {
            name: "llama.cpp",
            unavailable_reason: "No local llama-server or llama-cli binary found. Pass --llama-server or --llama-cli to enable Metal comparisons.",
          };
        }

        baselineByScenario.set(scenarioDef.id, baseline);
      }
    } finally {
      if (launchedServer) {
        await stopProcess(launchedServer.child);
      }
    }

    const scenarios = scenarioDefs.map((scenarioDef) => scenarioResultPayload(
      entry,
      scenarioDef,
      zincByScenario.get(scenarioDef.id) ?? {
        name: "ZINC",
        unavailable_reason: "No ZINC benchmark result was recorded for this scenario.",
      },
      baselineByScenario.get(scenarioDef.id) ?? {
        name: "llama.cpp",
        unavailable_reason: "No llama.cpp baseline result was recorded for this scenario.",
      },
    ));
    models.push(primaryScenarioSummary(entry, scenarios));
  }

  return {
    id: "metal",
    label: "Metal",
    description: "Apple Silicon local benchmark suite rendered from the current workspace build.",
    generated_at: new Date().toISOString(),
    source: "tools/performance_suite.mjs",
    machine,
    provenance: {
      zinc: zincProvenance,
      llama_cpp: llamaCppProvenance,
    },
    methodology: {
      runner: "zinc cli + local llama.cpp baseline",
      benchmark_style: "multi-scenario repeated CLI runs with same-model local llama.cpp comparison",
      notes: [
        "ZINC is measured from the local ReleaseFast build in the current workspace.",
        "Each model runs a scenario matrix: core prompt, medium context, long context, and extended decode.",
        "Gemma and GPT-OSS cases use chat-template mode; Qwen cases use raw completion mode to avoid visible thinking scaffolds in CLI output.",
        "Default Metal reference cases resolve from the managed model cache in ~/Library/Caches/zinc/models/models.",
        "Use --metal-pull-missing to install missing managed models through `zinc model pull` before the suite runs.",
        "When available, llama.cpp is measured from one local llama-server launch per model so the same loaded model can serve the whole scenario matrix.",
        "If llama-server is unavailable, the suite falls back to local llama-cli for same-model CLI comparison.",
      ],
      runs: args.runs,
      warmup_runs: args.warmupRuns,
    },
    summary: targetSummary(models),
    models,
  };
}

async function buildRdnaCreds(args) {
  const dotEnv = await readDotEnv(path.join(ROOT, ".env"));
  const host = process.env.ZINC_HOST ?? dotEnv.ZINC_HOST;
  const user = process.env.ZINC_USER ?? dotEnv.ZINC_USER;
  const port = process.env.ZINC_PORT ?? dotEnv.ZINC_PORT ?? "22";
  if (!host || !user) {
    throw new Error("RDNA benchmarking needs ZINC_HOST and ZINC_USER in the environment or .env");
  }
  return {
    host,
    user,
    port,
    workdir: args.rdnaWorkdir,
  };
}

async function runRdnaTarget(args) {
  const creds = await buildRdnaCreds(args);
  await prepareRdna(args, creds);
  const rdnaLlamaCli = await detectRdnaLlamaCliPath(creds);
  const rdnaLlamaServer = await detectRdnaLlamaServerPath(creds);
  const baselineBinary = rdnaLlamaServer || rdnaLlamaCli;
  const zincProvenance = await captureRemoteGitProvenance(creds);
  const llamaCppProvenance = await captureRemoteLlamaCppProvenance(baselineBinary, creds);

  const knownCases = defaultRdnaCases(args.rdnaModelRoot);
  const discoveredCases = args.discoverModels ? await discoverRdnaCases(args.rdnaModelRoot, creds) : [];
  const mergedCases = new Map();
  for (const entry of [...discoveredCases, ...knownCases]) {
    mergedCases.set(entry.id, entry);
  }
  const cases = [...mergedCases.values()].filter((entry) => {
    if (args.models && !args.models.has(entry.id)) return false;
    return shouldIncludeInPublishedBenchmarks(entry.id, args.models);
  });
  const models = [];

  for (const entry of cases) {
    console.log(`[rdna] ${entry.id}`);
    const phases = buildMeasurementPhases(entry.id, entry.prompt_mode, entry.prompt);
    const scenarioDefs = phases.filter((phase) => phase.phase === "zinc").map((phase) => phase.scenarioDef);
    const zincByScenario = new Map();
    const baselineByScenario = new Map();
    let launchedServer = null;
    let triedLaunchingServer = false;
    let serverLaunchFailure = null;
    try {
      for (const phase of phases) {
        const scenarioDef = phase.scenarioDef;
        const caseDef = buildScenarioCase(entry, scenarioDef);
        if (phase.phase === "zinc") {
          let zinc = null;
          try {
            const zincRows = await runSeries({
              label: `zinc ${entry.id} ${scenarioDef.id}`,
              warmupRuns: args.warmupRuns,
              runs: args.runs,
              command: rdnaZincCommand(caseDef, creds),
              parser: parseZincCliOutput,
              cwd: ROOT,
              timeoutMs: args.timeoutMs,
            });
            zinc = createStats("ZINC", zincRows);
          } catch (error) {
            zinc = {
              name: "ZINC",
              unavailable_reason: `ZINC run failed: ${error.message.split("\n")[0]}`,
            };
          }
          zincByScenario.set(scenarioDef.id, zinc);
          continue;
        }

        if (!triedLaunchingServer && rdnaLlamaServer) {
          triedLaunchingServer = true;
          try {
            launchedServer = await launchRdnaLlamaServer(entry, creds, rdnaLlamaServer, args.timeoutMs);
          } catch (error) {
            launchedServer = null;
            serverLaunchFailure = error;
          }
        }

        let baseline = null;
        if (entry.baseline_enabled !== false && launchedServer) {
          try {
            const baselineRows = await runRdnaOpenAiSeries({
              label: `llama.cpp ${entry.id} ${scenarioDef.id}`,
              warmupRuns: args.warmupRuns,
              runs: args.runs,
              creds,
              port: launchedServer.port,
              caseDef,
              timeoutMs: args.timeoutMs,
            });
            baseline = createStats("llama.cpp", baselineRows);
          } catch (error) {
            baseline = {
              name: "llama.cpp",
              unavailable_reason: `RDNA baseline failed: ${error.message.split("\n")[0]}`,
            };
          }
        } else if (entry.baseline_enabled !== false && serverLaunchFailure) {
          baseline = {
            name: "llama.cpp",
            unavailable_reason: `RDNA baseline failed: ${serverLaunchFailure.message.split("\n")[0]}`,
          };
        } else if (entry.baseline_enabled !== false && rdnaLlamaCli) {
          try {
            const baselineRows = await runSeries({
              label: `llama.cpp ${entry.id} ${scenarioDef.id}`,
              warmupRuns: args.warmupRuns,
              runs: args.runs,
              command: rdnaLlamaCommand(caseDef, creds, rdnaLlamaCli),
              parser: parseLlamaCliOutput,
              cwd: ROOT,
              timeoutMs: args.timeoutMs,
            });
            baseline = createStats("llama.cpp", baselineRows);
          } catch (error) {
            baseline = {
              name: "llama.cpp",
              unavailable_reason: `RDNA baseline failed: ${error.message.split("\n")[0]}`,
            };
          }
        } else {
          baseline = {
            name: "llama.cpp",
            unavailable_reason: rdnaLlamaServer || rdnaLlamaCli
              ? "No RDNA llama.cpp baseline is defined for this case yet."
              : "Could not locate a remote llama.cpp baseline binary on the RDNA node.",
          };
        }

        baselineByScenario.set(scenarioDef.id, baseline);
      }
    } finally {
      if (launchedServer) {
        await stopRdnaLlamaServer(creds, launchedServer.port);
      }
    }

    const scenarios = scenarioDefs.map((scenarioDef) => scenarioResultPayload(
      entry,
      scenarioDef,
      zincByScenario.get(scenarioDef.id) ?? {
        name: "ZINC",
        unavailable_reason: "No ZINC benchmark result was recorded for this scenario.",
      },
      baselineByScenario.get(scenarioDef.id) ?? {
        name: "llama.cpp",
        unavailable_reason: "No llama.cpp baseline result was recorded for this scenario.",
      },
    ));
    models.push(primaryScenarioSummary(entry, scenarios));
  }

  return {
    id: "rdna",
    label: "RDNA",
    description: "AMD RDNA benchmark node results collected over SSH and compared against llama.cpp on the same hardware.",
    generated_at: new Date().toISOString(),
    source: "tools/performance_suite.mjs",
    machine: {
      label: "AMD RDNA bench node",
      machine_model: "Remote Linux host",
      chip: "AMD Radeon AI PRO R9700",
      memory: "32 GB VRAM · 576 GB/s",
      gpu: "Radeon AI PRO R9700",
      os: "Ubuntu Linux",
    },
    provenance: {
      zinc: zincProvenance,
      llama_cpp: llamaCppProvenance,
    },
    methodology: {
      runner: "zinc cli + remote llama.cpp baseline",
      benchmark_style: "remote repeated multi-scenario runs over SSH with per-model same-machine llama.cpp baselines",
      notes: [
        "ZINC runs use the remote ReleaseFast build with RADV_PERFTEST=coop_matrix.",
        "llama.cpp comparison runs on the RDNA node against the same model file, preferably through one per-model llama-server launch reused across the scenario matrix.",
        "Each model runs a scenario matrix: core prompt, medium context, long context, and extended decode.",
        "Use --all-models to auto-discover GGUFs on the RDNA node instead of the default reference set.",
        "If remote llama-server is unavailable, the suite falls back to remote llama-cli for baseline collection.",
        "The suite assumes the remote node is already in a clean benchmark state unless --rdna-sync / --rdna-build / --rdna-start-llama are provided.",
      ],
      runs: args.runs,
      warmup_runs: args.warmupRuns,
    },
    summary: targetSummary(models),
    models,
  };
}

async function loadExistingArtifact(siteDataPath) {
  try {
    return JSON.parse(await fs.readFile(siteDataPath, "utf8"));
  } catch {
    return { schema_version: 1, generated_at: new Date().toISOString(), targets: [] };
  }
}

async function writeJson(filepath, data) {
  await fs.mkdir(path.dirname(filepath), { recursive: true });
  await fs.writeFile(filepath, `${JSON.stringify(data, null, 2)}\n`, "utf8");
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(usage());
    return;
  }

  const incoming = [];
  if (args.target === "metal" || args.target === "both") {
    incoming.push(await runMetalTarget(args));
  }
  if (args.target === "rdna" || args.target === "both") {
    incoming.push(await runRdnaTarget(args));
  }

  const outputArtifact = buildArtifact(incoming);
  await writeJson(args.output, outputArtifact);
  if (args.writeSiteData) {
    const existing = await loadExistingArtifact(args.siteData);
    const merged = mergeArtifacts(existing, incoming);
    await writeJson(args.siteData, merged);
  }

  console.log(`\nWrote benchmark artifact: ${args.output}`);
  if (args.writeSiteData) {
    console.log(`Updated site data: ${args.siteData}`);
  }
  for (const target of incoming) {
    console.log(`${target.label}: ${target.models.length} model(s), fastest ${(target.summary.fastest_decode_tps ?? 0).toFixed(2)} tok/s`);
  }
}

if (process.argv[1] === TOOL_PATH) {
  main().catch((error) => {
    console.error(error.stack || error.message);
    process.exit(1);
  });
}
