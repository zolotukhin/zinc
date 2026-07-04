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
const DEFAULT_RDNA_WORKDIR = "/root/zinc-bench";
const DEFAULT_RDNA_MODEL_ROOT = "/root/models";
const TARGET_ORDER = ["rdna", "cuda", "intel", "metal"];
const MAX_CAPTURE_CHARS = 256_000;

function modelPath(root, dir) {
  return path.join(root, dir, "model.gguf");
}

function shellQuote(value) {
  return `'${String(value).replace(/'/g, `'\\''`)}'`;
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
    phase: "all",
    scenarios: null,
    timeoutMs: DEFAULT_TIMEOUT_MS,
    models: null,
    buildLocal: true,
    rdnaSync: false,
    rdnaBuild: false,
    rdnaStartLlama: false,
    rdnaNode: process.env.ZINC_RDNA_NODE ?? process.env.ZINC_NODE ?? null,
    rdnaBackend: process.env.ZINC_RDNA_BACKEND ?? "vulkan",
    rdnaVkDevice: process.env.ZINC_RDNA_VK_DEVICE != null
      ? parseInteger(process.env.ZINC_RDNA_VK_DEVICE, "ZINC_RDNA_VK_DEVICE")
      : 0,
    requireRdnaDeviceSubstring: process.env.ZINC_RDNA_REQUIRE_DEVICE_SUBSTRING ?? null,
    cudaSync: false,
    cudaBuild: false,
    cudaStartLlama: false,
    cudaNode: process.env.ZINC_CUDA_NODE ?? null,
    cudaGpuUuid: process.env.ZINC_CUDA_GPU_UUID ?? process.env.ZINC_CUDA_GPU ?? null,
    cudaWorkdir: process.env.ZINC_CUDA_WORKDIR ?? null,
    cudaModelRoot: process.env.ZINC_CUDA_MODEL_ROOT ?? null,
    cudaLlamaServer: process.env.ZINC_CUDA_LLAMA_SERVER_REMOTE ?? null,
    intelSync: false,
    intelBuild: false,
    intelStartLlama: false,
    metalPullMissing: false,
    localModelRoot: DEFAULT_LOCAL_MODEL_ROOT,
    rdnaModelRoot: DEFAULT_RDNA_MODEL_ROOT,
    rdnaWorkdir: DEFAULT_RDNA_WORKDIR,
    intelModelRoot: null,
    intelWorkdir: null,
    intelXdgCacheHome: null,
    intelRemoteLibcConf: null,
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
        if (!["metal", "rdna", "cuda", "intel", "both", "all"].includes(value)) {
          throw new Error(`Invalid --target '${value}'. Expected metal, rdna, cuda, intel, both, or all.`);
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
      case "--phase": {
        const value = argv[++i] ?? "";
        if (!["all", "zinc", "baseline"].includes(value)) {
          throw new Error(`Invalid --phase '${value}'. Expected all, zinc, or baseline.`);
        }
        args.phase = value;
        break;
      }
      case "--scenarios":
        args.scenarios = new Set((argv[++i] ?? "").split(",").map((value) => value.trim()).filter(Boolean));
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
      case "--rdna-node":
        args.rdnaNode = argv[++i] ?? args.rdnaNode;
        break;
      case "--rdna-backend":
        args.rdnaBackend = argv[++i] ?? args.rdnaBackend;
        break;
      case "--intel-sync":
        args.intelSync = true;
        break;
      case "--intel-build":
        args.intelBuild = true;
        break;
      case "--intel-start-llama":
        args.intelStartLlama = true;
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
      case "--rdna-vk-device":
        args.rdnaVkDevice = parseInteger(argv[++i], "--rdna-vk-device");
        break;
      case "--require-rdna-device-substring":
        args.requireRdnaDeviceSubstring = argv[++i] ?? args.requireRdnaDeviceSubstring;
        break;
      case "--cuda-sync":
        args.cudaSync = true;
        break;
      case "--cuda-build":
        args.cudaBuild = true;
        break;
      case "--cuda-start-llama":
        args.cudaStartLlama = true;
        break;
      case "--cuda-node":
        args.cudaNode = argv[++i] ?? args.cudaNode;
        break;
      case "--cuda-gpu":
        args.cudaGpuUuid = argv[++i] ?? args.cudaGpuUuid;
        break;
      case "--cuda-workdir":
        args.cudaWorkdir = argv[++i] ?? args.cudaWorkdir;
        break;
      case "--cuda-model-root":
        args.cudaModelRoot = argv[++i] ?? args.cudaModelRoot;
        break;
      case "--intel-model-root":
        args.intelModelRoot = argv[++i] ?? args.intelModelRoot;
        break;
      case "--intel-workdir":
        args.intelWorkdir = argv[++i] ?? args.intelWorkdir;
        break;
      case "--intel-xdg-cache-home":
        args.intelXdgCacheHome = argv[++i] ?? args.intelXdgCacheHome;
        break;
      case "--intel-remote-libc-conf":
        args.intelRemoteLibcConf = argv[++i] ?? args.intelRemoteLibcConf;
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
  if (!["auto", "vulkan", "zinc_rt"].includes(args.rdnaBackend)) {
    throw new Error(`Invalid --rdna-backend '${args.rdnaBackend}'. Expected auto, vulkan, or zinc_rt.`);
  }
  return args;
}

function usage() {
  return `Usage: bun tools/performance_suite.mjs [options]
  --target <metal|rdna|intel|both|all>
                              Which benchmark target(s) to run; both means rdna+metal, all includes Intel
  --output <path>             JSON artifact output path
  --site-data <path>          Site JSON data path to update
  --no-site-write             Do not update the site JSON artifact
  --runs <n>                  Measured runs per model (default: 3)
  --warmup <n>                Warmup runs per model (default: 1)
  --phase <all|zinc|baseline> Measure both phases, only ZINC, or only llama.cpp (default: all)
  --scenarios <ids>           Comma-separated scenario ids: core,context-medium,context-long,decode-extended
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
  --rdna-node <name>          Select node-specific env keys, e.g. rdna2 -> ZINC_RDNA2_HOST/PORT/USER
  --rdna-backend <backend>    ZINC backend to build on RDNA: auto, vulkan, zinc_rt (default: vulkan)
  --intel-sync                Rsync current repo to the Intel node before running
  --intel-build               Build ReleaseFast on the Intel node before running
  --intel-start-llama         Stop stale llama-server processes on the Intel node before baseline runs
  --local-model-root <path>   Override local GGUF cache root
  --rdna-model-root <path>    Override remote model root
  --rdna-workdir <path>       Override remote ZINC checkout path
  --rdna-vk-device <n>        Vulkan device index on the RDNA node (default 0; env: ZINC_RDNA_VK_DEVICE).
                              Use 'vulkaninfo --summary' on the node to pick the discrete GPU.
  --require-rdna-device-substring <s>
                              Refuse to run if the selected Vulkan device name does not contain <s>
                              (env: ZINC_RDNA_REQUIRE_DEVICE_SUBSTRING). E.g. 'GFX1201' for R9700.
  --intel-model-root <path>   Override Intel remote model root
  --intel-workdir <path>      Override Intel remote ZINC checkout path
  --intel-xdg-cache-home <p>  Override Intel remote XDG cache home for managed models
  --intel-remote-libc-conf <p>
                              Copy an existing remote libc.conf into the Intel checkout before building
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

export function parseZincServerOutput(text) {
  const marker = "\n__ZINC_TIMING__\n";
  const markerIndex = text.lastIndexOf(marker);
  if (markerIndex === -1) {
    throw new Error("Could not parse ZINC server timing output");
  }

  const body = JSON.parse(text.slice(0, markerIndex).trim());
  if (body.error) {
    throw new Error(body.error.message ?? "ZINC server returned an error");
  }

  const parsed = parseZincCliOutput(text.slice(markerIndex + marker.length));
  const content = body.choices?.[0]?.text ?? body.choices?.[0]?.message?.content ?? "";
  const promptTokens = body.usage?.prompt_tokens ?? parsed.promptTokens;
  const generatedTokens = body.usage?.completion_tokens ?? parsed.generatedTokens;
  return {
    ...parsed,
    promptTokens,
    prefillTokens: promptTokens ?? parsed.prefillTokens,
    generatedTokens,
    outputPreview: typeof content === "string" && content.trim() ? content.trim() : parsed.outputPreview,
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

export function parseZincVersionOutput(text) {
  const lines = String(text ?? "")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const versionLine = lines.find((line) => /^zinc\s+/i.test(line));
  const field = (name) => {
    const prefix = `${name}:`;
    return lines.find((line) => line.toLowerCase().startsWith(prefix))?.slice(prefix.length).trim() ?? null;
  };
  return {
    version: versionLine ? versionLine.replace(/^zinc\s+/i, "").trim() : null,
    commit: field("commit"),
    target: field("target"),
    optimize: field("optimize"),
    backend: field("backends") ?? field("backend"),
  };
}

export function validateZincBackend(versionText, expectedBackend) {
  if (!expectedBackend || expectedBackend === "auto") return parseZincVersionOutput(versionText);
  const parsed = parseZincVersionOutput(versionText);
  if (parsed.backend !== expectedBackend) {
    const observed = parsed.backend ?? "unknown";
    throw new Error(`RDNA ZINC binary backend mismatch: expected ${expectedBackend}, observed ${observed}. Rebuild or stop the process that overwrote zig-out/bin/zinc.`);
  }
  return parsed;
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

function finiteMetric(value) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function finiteTokenCount(value) {
  return typeof value === "number" && Number.isFinite(value) && value >= 0 ? value : null;
}

function backendPhaseSeconds(prefillTps, decodeTps, promptTokens, generatedTokens) {
  const prompt = finiteTokenCount(promptTokens) ?? 0;
  const generated = finiteTokenCount(generatedTokens) ?? 0;
  if (prompt <= 0 && generated <= 0) return null;

  let seconds = 0;
  if (prompt > 0) {
    if (!prefillTps || prefillTps <= 0) return null;
    seconds += prompt / prefillTps;
  }
  if (generated > 0) {
    if (!decodeTps || decodeTps <= 0) return null;
    seconds += generated / decodeTps;
  }
  return seconds > 0 ? seconds : null;
}

function matchingGeneratedBudget(zincGenerated, baselineGenerated, expectedGeneratedTokens) {
  const zinc = finiteTokenCount(zincGenerated);
  const baseline = finiteTokenCount(baselineGenerated);
  if (zinc == null || baseline == null) return false;
  if (zinc !== baseline) return false;
  const expected = finiteTokenCount(expectedGeneratedTokens);
  if (expected != null && zinc < expected) return false;
  return true;
}

export function outputQualityStatus(preview, generatedTokens = null) {
  const text = `${preview ?? ""}`.trim();
  if (!text) {
    return {
      tone: "neutral",
      label: "No Preview",
      note: "No output preview was captured for this workload.",
    };
  }

  if (/^<\|(?:im_end|endoftext|end)\|>$/.test(text) || (generatedTokens != null && generatedTokens <= 4 && text.includes("<|im_end|>"))) {
    return {
      tone: "caution",
      label: "Preview flagged",
      note: "ZINC completed numerically, but the preview is only a stop token.",
    };
  }

  if (/<\|[^>\n]{1,64}\|>/.test(text)) {
    return {
      tone: "caution",
      label: "Preview flagged",
      note: "ZINC completed numerically, but the preview contains raw chat control tokens.",
    };
  }

  if ((text.match(/<think>/g) ?? []).length > 1 || /<\/?parameter>/.test(text)) {
    return {
      tone: "caution",
      label: "Preview flagged",
      note: "ZINC completed numerically, but the preview contains repeated reasoning/control tags.",
    };
  }

  if (
    /(?:^|[\s:])(?:\d+\.){12,}/.test(text) ||
    /(?:^|\n)\s*(\d+[.)])\s*(?:\n\s*\1\s*){11,}/.test(text) ||
    hasRepeatedNumberedItemText(text) ||
    /([#*_=-])\1{31,}/.test(text) ||
    /\b([A-Za-z]{3,})\1{8,}\b/.test(text)
  ) {
    return {
      tone: "caution",
      label: "Preview flagged",
      note: "ZINC completed numerically, but the preview shows a repeated pattern.",
    };
  }

  return {
    tone: "positive",
    label: "Preview OK",
    note: "The captured preview does not show an obvious stop-token, control-token, or repetition failure.",
  };
}

function hasRepeatedNumberedItemText(text) {
  const items = [...text.matchAll(/(?:^|\s)(\d{1,3})[.)]\s+([\s\S]*?)(?=(?:\s+\d{1,3}[.)]\s+)|$)/g)]
    .map((match) => `${match[2] ?? ""}`
      .toLowerCase()
      .replace(/\s+/g, " ")
      .trim())
    .filter((item) => item.length >= 24);
  let last = "";
  let run = 0;
  for (const item of items) {
    const key = item.slice(0, 160);
    if (key === last) {
      run += 1;
    } else {
      last = key;
      run = 1;
    }
    if (run >= 4) return true;
  }
  return false;
}

function scenarioNeedsOutputReview(scenario) {
  return outputQualityStatus(
    scenario?.zinc?.output_preview,
    scenario?.zinc?.generated_tokens,
  ).tone === "caution";
}

export function buildComparison(zincSummary, baselineSummary, options = {}) {
  if (!zincSummary || !baselineSummary) return null;
  const zincDecode = finiteMetric(zincSummary.decode_tps?.median ?? zincSummary.decode_tps?.avg ?? null);
  const baselineDecode = finiteMetric(baselineSummary.decode_tps?.median ?? baselineSummary.decode_tps?.avg ?? null);
  const zincPrefill = finiteMetric(zincSummary.prefill_tps?.median ?? zincSummary.prefill_tps?.avg ?? null);
  const baselinePrefill = finiteMetric(baselineSummary.prefill_tps?.median ?? baselineSummary.prefill_tps?.avg ?? null);
  const zincPromptTokens = finiteTokenCount(zincSummary.prompt_tokens);
  const baselinePromptTokens = finiteTokenCount(baselineSummary.prompt_tokens);
  const zincGeneratedTokens = finiteTokenCount(zincSummary.generated_tokens);
  const baselineGeneratedTokens = finiteTokenCount(baselineSummary.generated_tokens);
  const zincLatency = finiteMetric(zincSummary.total_latency_ms?.median ?? zincSummary.total_latency_ms?.avg ?? null);
  const baselineLatency = finiteMetric(baselineSummary.total_latency_ms?.median ?? baselineSummary.total_latency_ms?.avg ?? null);
  const zincEndToEnd = finiteMetric(zincSummary.end_to_end_tps?.median ?? zincSummary.end_to_end_tps?.avg ?? null);
  const baselineEndToEnd = finiteMetric(baselineSummary.end_to_end_tps?.median ?? baselineSummary.end_to_end_tps?.avg ?? null);
  if (!zincDecode || !baselineDecode) return null;

  // Overall rating is a wall-clock comparison, not an average of prefill and
  // decode percentages. Each backend uses its own tokenizer counts; early-stop
  // rows do not get an aggregate percentage.
  const zincOverallSeconds = backendPhaseSeconds(zincPrefill, zincDecode, zincPromptTokens, zincGeneratedTokens);
  const baselineOverallSeconds = backendPhaseSeconds(baselinePrefill, baselineDecode, baselinePromptTokens, baselineGeneratedTokens);
  const zincTotalPhaseTokens = (zincPromptTokens ?? 0) + (zincGeneratedTokens ?? 0);
  const baselineTotalPhaseTokens = (baselinePromptTokens ?? 0) + (baselineGeneratedTokens ?? 0);
  const zincOverall = zincOverallSeconds && zincTotalPhaseTokens > 0 ? zincTotalPhaseTokens / zincOverallSeconds : null;
  const baselineOverall = baselineOverallSeconds && baselineTotalPhaseTokens > 0 ? baselineTotalPhaseTokens / baselineOverallSeconds : null;
  const overallComparable = matchingGeneratedBudget(zincGeneratedTokens, baselineGeneratedTokens, options.expectedGeneratedTokens);
  const overallPct = overallComparable && zincOverallSeconds && baselineOverallSeconds
    ? (baselineOverallSeconds / zincOverallSeconds) * 100
    : null;

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
    zinc_overall_tps: zincOverall,
    baseline_overall_tps: baselineOverall,
    zinc_overall_seconds: zincOverallSeconds,
    baseline_overall_seconds: baselineOverallSeconds,
    overall_basis: "phase_wall_time_v2",
    overall_comparable: overallPct != null,
    overall_unavailable_reason: overallPct == null
      ? "Generated token counts did not match the scenario budget, or phase telemetry was incomplete."
      : null,
    pct_of_baseline: pctOfBaseline,
    delta_tps: gapTps,
    delta_pct: pctOfBaseline - 100,
    leader: gapTps >= 0 ? "zinc" : "baseline",
    prompt_pct_of_baseline: zincPrefill && baselinePrefill ? (zincPrefill / baselinePrefill) * 100 : null,
    latency_pct_of_baseline: zincLatency && baselineLatency ? (zincLatency / baselineLatency) * 100 : null,
    latency_delta_ms: zincLatency != null && baselineLatency != null ? zincLatency - baselineLatency : null,
    end_to_end_pct_of_baseline: zincEndToEnd && baselineEndToEnd ? (zincEndToEnd / baselineEndToEnd) * 100 : null,
    end_to_end_delta_tps: zincEndToEnd != null && baselineEndToEnd != null ? zincEndToEnd - baselineEndToEnd : null,
    overall_pct_of_baseline: overallPct,
    overall_delta_tps: overallPct != null && zincOverall != null && baselineOverall != null ? zincOverall - baselineOverall : null,
  };
}

function mergeScenarioForPartialRun(existingScenario, incomingScenario) {
  if (!existingScenario) return incomingScenario;
  if (!incomingScenario) return existingScenario;

  const zinc = incomingScenario.zinc ?? existingScenario.zinc ?? null;
  const baseline = incomingScenario.baseline ?? existingScenario.baseline ?? null;
  return {
    ...existingScenario,
    ...incomingScenario,
    zinc,
    baseline,
    comparison: buildComparison(zinc, baseline, { expectedGeneratedTokens: incomingScenario.max_tokens ?? existingScenario.max_tokens }),
  };
}

function mergeModelForPartialRun(existingModel, incomingModel) {
  if (!existingModel) return incomingModel;
  if (!incomingModel) return existingModel;

  const scenarioIds = new Set([
    ...(existingModel.scenarios ?? []).map((scenario) => scenario.id),
    ...(incomingModel.scenarios ?? []).map((scenario) => scenario.id),
  ]);
  const existingScenarios = new Map((existingModel.scenarios ?? []).map((scenario) => [scenario.id, scenario]));
  const incomingScenarios = new Map((incomingModel.scenarios ?? []).map((scenario) => [scenario.id, scenario]));
  const scenarios = [...scenarioIds]
    .map((id) => mergeScenarioForPartialRun(existingScenarios.get(id), incomingScenarios.get(id)))
    .filter(Boolean);

  const primary = scenarios.find((scenario) => scenario.id === "core") ?? scenarios[0];
  const zinc = primary?.zinc ?? incomingModel.zinc ?? existingModel.zinc ?? null;
  const baseline = primary?.baseline ?? incomingModel.baseline ?? existingModel.baseline ?? null;

  return {
    ...existingModel,
    ...incomingModel,
    zinc,
    baseline,
    comparison: buildComparison(zinc, baseline, { expectedGeneratedTokens: primary?.max_tokens ?? incomingModel.max_tokens ?? existingModel.max_tokens }),
    scenarios,
  };
}

function mergeTargetForPartialRun(existingTarget, incomingTarget) {
  if (!existingTarget) return incomingTarget;
  if (!incomingTarget) return existingTarget;

  const modelIds = new Set([
    ...(existingTarget.models ?? []).map((model) => model.id),
    ...(incomingTarget.models ?? []).map((model) => model.id),
  ]);
  const existingModels = new Map((existingTarget.models ?? []).map((model) => [model.id, model]));
  const incomingModels = new Map((incomingTarget.models ?? []).map((model) => [model.id, model]));
  const models = [...modelIds]
    .map((id) => mergeModelForPartialRun(existingModels.get(id), incomingModels.get(id)))
    .filter(Boolean);

  return {
    ...existingTarget,
    ...incomingTarget,
    models,
  };
}

function recalculateScenarioComparison(scenario) {
  const comparison = scenario?.baseline?.decode_tps
    ? buildComparison(scenario.zinc, scenario.baseline, { expectedGeneratedTokens: scenario.max_tokens })
    : null;
  return {
    ...scenario,
    comparison,
    output_quality: outputQualityStatus(
      scenario?.zinc?.output_preview,
      scenario?.zinc?.generated_tokens,
    ),
  };
}

function recalculateModelComparisons(model) {
  const scenarios = Array.isArray(model.scenarios)
    ? model.scenarios.map(recalculateScenarioComparison)
    : null;
  const primary = selectPrimaryScenario(scenarios);
  const comparison = primary?.comparison
    ?? (model.baseline?.decode_tps
      ? buildComparison(model.zinc, model.baseline, { expectedGeneratedTokens: model.max_tokens })
      : null);
  return {
    ...model,
    ...(primary ? {
      prompt_mode: primary.prompt_mode ?? model.prompt_mode,
      prompt: primary.prompt ?? model.prompt,
      max_tokens: primary.max_tokens ?? model.max_tokens,
      zinc: primary.zinc ?? model.zinc,
      baseline: primary.baseline ?? model.baseline,
    } : {}),
    comparison,
    ...(scenarios ? { scenarios } : {}),
  };
}

export function normalizedModelName(model) {
  return (model?.label ?? model?.id ?? "")
    .toLowerCase()
    .replace(/\bgemma4\b/g, "gemma 4")
    .replace(/\bqwen36\b/g, "qwen 3.6")
    .replace(/\bqwen3\b/g, "qwen 3")
    .replace(/q4_k/g, "q4k")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

export function compareModelsByName(a, b) {
  return normalizedModelName(a).localeCompare(normalizedModelName(b), undefined, {
    numeric: true,
    sensitivity: "base",
  });
}

export function mergeArtifacts(existing, incomingTargets, options = {}) {
  const byId = new Map();
  const preserveMissingPhases = options.preserveMissingPhases ?? false;
  const normalizeTarget = (target) => {
    const models = [...(target?.models ?? [])]
      .map(recalculateModelComparisons)
      .sort(compareModelsByName);
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
    const normalizedIncoming = normalizeTarget(target);
    const previous = byId.get(target.id);
    byId.set(
      target.id,
      normalizeTarget(preserveMissingPhases && previous
        ? mergeTargetForPartialRun(previous, normalizedIncoming)
        : normalizedIncoming),
    );
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

function modelScenariosForSummary(model) {
  if (Array.isArray(model.scenarios) && model.scenarios.length > 0) return model.scenarios;
  return [{
    id: "core",
    zinc: model.zinc,
    baseline: model.baseline,
    comparison: model.comparison,
  }];
}

function aggregateOverallPct(scenarios) {
  let zincSeconds = 0;
  let baselineSeconds = 0;
  let count = 0;
  for (const scenario of scenarios) {
    const comparison = scenario?.comparison;
    if (comparison?.overall_pct_of_baseline == null) continue;
    if (!comparison?.overall_comparable) continue;
    const zinc = finiteMetric(comparison.zinc_overall_seconds);
    const baseline = finiteMetric(comparison.baseline_overall_seconds);
    if (zinc == null || baseline == null || zinc <= 0 || baseline <= 0) continue;
    zincSeconds += zinc;
    baselineSeconds += baseline;
    count += 1;
  }
  return count > 0 && zincSeconds > 0 ? (baselineSeconds / zincSeconds) * 100 : null;
}

function scenarioComparableForPrimary(scenario) {
  const comparison = scenario?.comparison;
  return !comparison || comparison.overall_comparable !== false;
}

function selectPrimaryScenario(scenarios) {
  if (!Array.isArray(scenarios) || scenarios.length === 0) return null;
  const core = scenarios.find((scenario) => scenario.id === "core") ?? null;
  if (core && scenarioComparableForPrimary(core)) return core;
  return scenarios.find(scenarioComparableForPrimary) ?? core ?? scenarios[0];
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

  const validPrimaryRows = models
    .map((model) => {
      const scenario = selectPrimaryScenario(modelScenariosForSummary(model).filter((row) => !scenarioNeedsOutputReview(row)));
      return scenario ? { model, scenario } : null;
    })
    .filter(Boolean);
  const successful = validPrimaryRows.filter((row) => row.scenario?.zinc?.decode_tps);
  const fastest = successful.length > 0
    ? [...successful].sort((a, b) => {
        const bDecode = b.scenario.zinc.decode_tps.median ?? b.scenario.zinc.decode_tps.avg;
        const aDecode = a.scenario.zinc.decode_tps.median ?? a.scenario.zinc.decode_tps.avg;
        return bDecode - aDecode;
      })[0]
    : null;
  const scenarioRows = models.flatMap(modelScenariosForSummary);
  const compared = models.filter((model) => modelScenariosForSummary(model).some((scenario) => !scenarioNeedsOutputReview(scenario) && scenario.comparison?.overall_pct_of_baseline != null));

  return {
    fastest_model_id: fastest?.model?.id ?? null,
    fastest_model_label: fastest?.model?.label ?? null,
    fastest_decode_tps: fastest ? (fastest.scenario.zinc.decode_tps.median ?? fastest.scenario.zinc.decode_tps.avg) : null,
    benchmarked_models: models.length,
    successful_models: new Set(successful.map((row) => row.model.id)).size,
    average_decode_tps: compactMean(successful.map((row) => row.scenario.zinc.decode_tps.median ?? row.scenario.zinc.decode_tps.avg)),
    average_prompt_tps: compactMean(successful.map((row) => row.scenario.zinc.prefill_tps?.median ?? row.scenario.zinc.prefill_tps?.avg ?? null)),
    average_end_to_end_tps: compactMean(successful.map((row) => row.scenario.zinc.end_to_end_tps?.median ?? row.scenario.zinc.end_to_end_tps?.avg ?? null)),
    average_total_latency_ms: compactMean(successful.map((row) => row.scenario.zinc.total_latency_ms?.median ?? row.scenario.zinc.total_latency_ms?.avg ?? null)),
    compared_models: compared.length,
    average_pct_of_llama: aggregateOverallPct(scenarioRows.filter((scenario) => !scenarioNeedsOutputReview(scenario))),
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

export function benchmarkFailureReason(prefix, error) {
  const lines = String(error?.message ?? error ?? "unknown error")
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
  const firstLine = lines[0] ?? "unknown error";
  const commandFailure = firstLine.match(/^Command failed \(([^)]+)\):/);
  if (commandFailure) {
    const diagnostic = lines
      .slice(1)
      .find((line) => /^(err\(|error:|.*\b(?:ENOMEM|QueueSubmitFailed|ShaderFileNotFound|FileNotFound)\b)/i.test(line));
    if (diagnostic) return `${prefix}: ${diagnostic}`;
    return `${prefix}: command exited unsuccessfully (${commandFailure[1]}).`;
  }
  return `${prefix}: ${firstLine}`;
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

function remoteEnvPrefix(creds) {
  const entries = Object.entries(creds.env ?? {});
  if (!entries.length) return "";
  return entries.map(([key, value]) => `${key}=${shellQuote(value)}`).join(" ");
}

const REMOTE_ZINC_TUNING_ENV_KEYS = [
  "ZINC_BATCHED_PREFILL",
  "ZINC_INTEL_BATCHED_PREFILL",
  "ZINC_INTEL_BATCHED_PREFILL_CHUNK",
  "ZINC_FUSED_QK_KV",
  "ZINC_FA_SPLIT_K",
  "ZINC_MUL_MM_PROJ",
  "ZINC_Q4K_BATCH_KPAR",
  "ZINC_FUSED_RMS_ROUTER",
  "ZINC_FUSED_SSM_AB",
  "ZINC_SSM_DELTA_COLS8",
  "ZINC_SSM_DELTA_NORMED_QK",
  "ZINC_FUSED_SSM_QKV_Z",
  "ZINC_MOE_KPAR",
  "ZINC_MOE_Q5K_KPAR",
  "ZINC_MOE_Q8_1_GATE_UP_COLS",
  "ZINC_MOE_Q8_1_DOWN_COLS",
  "ZINC_MOE_Q5K_Q8_1_DOWN_ACC",
  "ZINC_MOE_FUSED_GATE_UP",
  "ZINC_MOE_FUSED_GATE_UP_SWIGLU",
  "ZINC_FUSE_MOE_DOWN_ACC",
  "ZINC_QWEN36_MOE_TOPK",
  "ZINC_QWEN36_MOE_PREFILL_TOPK",
  "ZINC_QWEN36_MOE_PREFILL_TOPK_GUARD",
  "ZINC_QWEN36_MOE_GROUPED_EXACT",
  "ZINC_INTEL_A3B_PRODUCTION",
  "ZINC_QWEN36_Q8_WIDE4_SSM_OUT",
  "ZINC_Q8_1_LM_HEAD",
  "ZINC_Q8_1_SSM_QKV_Z",
  "ZINC_GEMMA_MOE_TOPK",
  "ZINC_GEMMA_MOE_PREFILL_TOPK",
  "ZINC_PREFILL_PROFILE",
  "ZINC_PREFILL_Q8",
  "ZINC_PREFILL_DP4A",
  "ZINC_PREFILL_F16",
  "ZINC_PREFILL_LOWSMEM",
  "ZINC_PREFILL_TC",
  "ZINC_PREFILL_Q8_TC",
  "ZINC_PREFILL_GATE_UP_SWIGLU",
  "ZINC_QWEN35_9B_BM64_DOWN",
  "ZINC_QWEN35_9B_K12288_BK2",
  "ZINC_QWEN36_27B_DENSE_PREFILL",
  "ZINC_QWEN36_27B_DENSE_PREFILL_LAYERS",
  "ZINC_QWEN36_27B_DENSE_PREFILL_SEGMENT",
  "ZINC_QWEN36_27B_PREFIX_TAIL_PIPELINE",
  "ZINC_QWEN36_27B_SSM_BATCHED_DELTA",
  "ZINC_QWEN36_27B_SSM_PREFILL_PROJ",
  "ZINC_QWEN36_27B_FULL_ATTN_BATCHED",
  "ZINC_SSM_CHUNKED",
  "ZINC_SSM_WARP",
  "ZINC_CUDA_GRAPH",
  "ZINC_BATCH_GRAPH",
  "ZINC_BATCH_B1_MATVEC",
  "ZINC_BATCH_MROW",
  "ZINC_BATCH_MOE_SHARED",
  "ZINC_SERVE_CUBLAS",
  "ZINC_SERVE_CUBLAS_MINB",
  "ZINC_SERVE_WCACHE",
  "ZINC_SERVE_WCACHE_MB",
];

export function collectRemoteZincTuningEnv(dotEnv) {
  const env = {};
  for (const key of REMOTE_ZINC_TUNING_ENV_KEYS) {
    const value = process.env[key] ?? dotEnv[key];
    if (value != null && value !== "") env[key] = value;
  }
  return env;
}

export function llamaDeviceArgs(device) {
  return !device || device === "none" ? [] : ["--device", device];
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
  if (caseDef.context_tokens != null) parts.push("--context", String(caseDef.context_tokens));
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

export function zincServerTimingWaitSeconds(timeoutMs) {
  const commandBudgetSeconds = Math.max(10, Math.ceil(timeoutMs / 1000) - 5);
  return Math.min(60, commandBudgetSeconds);
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

export function buildZincOpenAiPayload(caseDef) {
  const payload = buildOpenAiPayload(caseDef);
  // ZINC's server benchmark measures the GGUF loaded at process start; a
  // request-level model id can activate managed-model routing instead.
  delete payload.model;
  return payload;
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

export function rdnaDpmHighScript() {
  return "for c in /sys/class/drm/card*/device; do if [ -f \"$c/pp_dpm_mclk\" ] && grep -q amdgpu \"$c/uevent\" 2>/dev/null; then levels=$(awk 'END { print NR }' \"$c/pp_dpm_mclk\" 2>/dev/null || printf 0); if [ \"${levels:-0}\" -ge 5 ]; then echo high > \"$c/power_dpm_force_performance_level\" 2>/dev/null || true; fi; fi; done; true";
}

async function lockRdnaDpmHigh(creds, timeoutMs = 30000) {
  await runShell(rdnaRemoteCommand(rdnaDpmHighScript(), creds), { cwd: ROOT, timeoutMs }).catch(() => {});
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
        rdnaRemoteCommand(`curl -fsS http://${creds.loopbackHost ?? "127.0.0.1"}:${port}/health >/dev/null`, creds),
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
      rdnaRemoteCommand(`pkill -f '[l]lama-server --host ${creds.loopbackHost ?? "127.0.0.1"} --port ${port}' || true`, creds),
      { cwd: ROOT, timeoutMs: 30000 },
    );
  } catch {
    // Best effort cleanup.
  }
}

async function stopRdnaZincServer(creds, port) {
  if (!port) return;
  try {
    await runShell(
      rdnaRemoteCommand(`pkill -f '[z]ig-out/bin/zinc .*--port ${port}' || true`, creds),
      { cwd: ROOT, timeoutMs: 30000 },
    );
  } catch {
    // Best effort cleanup.
  }
}

async function launchRdnaLlamaServer(caseDef, creds, serverPath, timeoutMs, targetId = "rdna", lockDpm = true) {
  const port = await pickOpenPort();
  const logPath = `/tmp/zinc-${targetId}-llama-${port}.log`;
  // Force the discrete RDNA4 GPU's DPM state to "high" before launch.
  // Without this, an idle llama-server holds the GPU but doesn't pin
  // memclk to a high level — the DPM heuristic sees "barely active"
  // and clocks down to level 0 (96 MHz mclk on gfx1201), which makes
  // every subsequent benchmark run at ~15% of peak memory bandwidth.
  // Empirically reproduced on this node: idle llama-server pinned
  // tg128 at 28-30 tok/s on Qwen 3.6 35B-A3B vs 108 tok/s right after
  // reboot; locking DPM to high recovers ~92 tok/s (level 4 of 5).
  // Card discovery: the discrete GPU is whichever card1/card2 has
  // amdgpu driver + a 6-level pp_dpm_mclk; we just try both.
  if (lockDpm) {
    await lockRdnaDpmHigh(creds);
  }

  const cmd = [
    serverPath,
    "--host", (creds.loopbackHost ?? "127.0.0.1"),
    "--port", String(port),
    "-m", caseDef.model_path,
    ...(creds.serverDeviceArgs ?? ["--device", `Vulkan${creds.vkDevice ?? 1}`]),
    "-ngl", "999",
    "--metrics",
    "--ctx-size", "4096",
    "--parallel", "1",
    // Default f16 KV cache. The previous -ctk q8_0 / -ctv q8_0 added
    // dequant-per-attention-read overhead that artificially lowered
    // llama.cpp's decode tok/s vs ZINC (which uses f32 KV) — apples-to-
    // not-apples. Drop to defaults so the published baseline reflects
    // peak llama.cpp throughput on this model + hardware.
    "-b", "4096",
    "-ub", "1024",
    "--flash-attn", "on",
  ];
  const launchScript = [
    "import os, subprocess",
    `cmd = ${JSON.stringify(cmd)}`,
    "env = dict(os.environ)",
    `env.update(${JSON.stringify(creds.env ?? {})})`,
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

async function launchRdnaZincServer(caseDef, creds, timeoutMs) {
  const port = await pickOpenPort();
  const logPath = `/tmp/zinc-rdna-zinc-${port}.log`;
  await lockRdnaDpmHigh(creds);

  const cmd = [
    "./zig-out/bin/zinc",
    "-m", caseDef.model_path,
    ...(caseDef.context_tokens != null ? ["--context", String(caseDef.context_tokens)] : []),
    ...(creds.vkDevice != null ? ["-d", String(creds.vkDevice)] : []),
    "--port", String(port),
  ];
  const launchScript = [
    "import os, subprocess",
    `cmd = ${JSON.stringify(cmd)}`,
    "env = dict(os.environ)",
    `env.update(${JSON.stringify({ RADV_PERFTEST: "coop_matrix", ...(creds.env ?? {}) })})`,
    `log = open(${JSON.stringify(logPath)}, "ab", buffering=0)`,
    `subprocess.Popen(cmd, cwd=${JSON.stringify(creds.workdir)}, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)`,
    'print("started")',
  ].join("; ");
  const remote = `python3 -c ${shellQuote(launchScript)}`;

  await runShell(rdnaRemoteCommand(remote, creds), { cwd: ROOT, timeoutMs: 120000 });

  try {
    await waitForHealthyRdnaServer(creds, port, logPath, Math.min(timeoutMs, 300000));
  } catch (error) {
    await stopRdnaZincServer(creds, port);
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
    `curl -sS http://${creds.loopbackHost ?? "127.0.0.1"}:${port}${endpoint} -H 'Content-Type: application/json' -d ${shellQuote(JSON.stringify(payload))}`,
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

async function runRdnaZincOpenAiSeries({ label, warmupRuns, runs, creds, port, logPath, caseDef, timeoutMs }) {
  const endpoint = caseDef.prompt_mode === "chat" ? `/v1/chat/completions` : `/v1/completions`;
  const payload = buildZincOpenAiPayload(caseDef);
  const host = creds.loopbackHost ?? "127.0.0.1";
  const deadlineSeconds = zincServerTimingWaitSeconds(timeoutMs);
  const remoteScript = [
    "set -euo pipefail",
    `LOG=${shellQuote(logPath)}`,
    `count_generated() { awk '/info\\(forward\\): Generated / { c++ } END { print c + 0 }' "$LOG" 2>/dev/null || printf '0\\n'; }`,
    "before=$(count_generated)",
    `response=$(curl -sS http://${host}:${port}${endpoint} -H 'Content-Type: application/json' -d ${shellQuote(JSON.stringify(payload))})`,
    "if printf '%s' \"$response\" | grep -q '\"error\"'; then msg=$(printf '%s' \"$response\" | python3 -c 'import json,sys; body=json.load(sys.stdin); print(body.get(\"error\",{}).get(\"message\", \"\"))' 2>/dev/null || true); printf '%s\\n' \"$response\"; printf '\\n__ZINC_TIMING__\\n'; tail -n 40 \"$LOG\" 2>/dev/null || true; if [ -n \"$msg\" ]; then echo \"err(zinc-bench): ZINC server returned error: $msg\" >&2; else echo 'err(zinc-bench): ZINC server returned an error response' >&2; fi; exit 125; fi",
    `deadline=$((SECONDS + ${deadlineSeconds}))`,
    "while [ \"$(count_generated)\" -le \"$before\" ]; do if [ \"$SECONDS\" -ge \"$deadline\" ]; then printf '%s\\n' \"$response\"; printf '\\n__ZINC_TIMING__\\n'; tail -n 40 \"$LOG\" 2>/dev/null || true; echo 'err(zinc-bench): no ZINC Generated log line after API response' >&2; exit 124; fi; sleep 0.05; done",
    "printf '%s\\n' \"$response\"",
    "printf '\\n__ZINC_TIMING__\\n'",
    "awk '/info\\(forward\\): Prefill:|info\\(forward\\): Generated / { lines[++n] = $0 } END { start = n > 1 ? n - 1 : 1; for (i = start; i <= n; i++) print lines[i] }' \"$LOG\"",
  ].join("\n");
  const command = rdnaRemoteCommand(remoteScript, creds);
  const measured = [];

  for (let i = 0; i < warmupRuns; i += 1) {
    console.log(`  warmup ${i + 1}/${warmupRuns}: ${label}`);
    const result = await runShell(command, { cwd: ROOT, timeoutMs });
    parseZincServerOutput(result.stdout);
  }

  for (let i = 0; i < runs; i += 1) {
    console.log(`  run ${i + 1}/${runs}: ${label}`);
    const started = performance.now();
    const result = await runShell(command, { cwd: ROOT, timeoutMs });
    const ended = performance.now();
    const parsed = parseZincServerOutput(result.stdout);
    parsed.totalLatencyMs = ended - started;
    const totalTokens = (parsed.promptTokens ?? 0) + (parsed.generatedTokens ?? 0);
    parsed.totalTps = totalTokens > 0 ? totalTokens / Math.max((ended - started) / 1000, 1e-9) : null;
    measured.push(parsed);
  }

  return measured;
}

function remoteCommand(commandText, creds) {
  const sshArgs = [];
  if (creds.sshKey) {
    sshArgs.push("-i", shellQuote(creds.sshKey), "-o", "IdentitiesOnly=yes");
  }
  if (process.env.ZINC_SSH_STRICT === "no") {
    sshArgs.push("-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null");
  }
  sshArgs.push("-p", creds.port, `${creds.user}@${creds.host}`, shellQuote(commandText));
  return `ssh ${sshArgs.join(" ")}`;
}

function rdnaRemoteCommand(commandText, creds) {
  return remoteCommand(commandText, creds);
}

function remoteDetachedCommand(commandText, creds) {
  const sshArgs = ["-f"];
  if (creds.sshKey) {
    sshArgs.push("-i", shellQuote(creds.sshKey), "-o", "IdentitiesOnly=yes");
  }
  if (process.env.ZINC_SSH_STRICT === "no") {
    sshArgs.push("-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null");
  }
  sshArgs.push("-p", creds.port, `${creds.user}@${creds.host}`, shellQuote(commandText));
  return `ssh ${sshArgs.join(" ")}`;
}

function rdnaDetachedCommand(commandText, creds) {
  return remoteDetachedCommand(commandText, creds);
}

function remoteSshTransport(creds) {
  const parts = ["ssh"];
  if (creds.sshKey) {
    parts.push("-i", shellQuote(creds.sshKey), "-o", "IdentitiesOnly=yes");
  }
  if (process.env.ZINC_SSH_STRICT === "no") {
    parts.push("-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null");
  }
  parts.push("-p", creds.port);
  return parts.join(" ");
}

function rdnaSshTransport(creds) {
  return remoteSshTransport(creds);
}

export function remoteZincCommand(caseDef, creds) {
  const parts = [];
  if (creds.commandPrelude) parts.push(creds.commandPrelude, "&&");
  parts.push(`cd ${shellQuote(creds.workdir)}`, "&&");
  const envPrefix = remoteEnvPrefix(creds);
  if (envPrefix) parts.push(envPrefix);
  parts.push(
    "./zig-out/bin/zinc",
    "-n",
    String(caseDef.max_tokens),
  );
  if (caseDef.context_tokens != null) parts.push("--context", String(caseDef.context_tokens));
  parts.push("-m", shellQuote(caseDef.model_path));
  if (creds.vkDevice != null) parts.push("-d", String(creds.vkDevice));
  if (caseDef.prompt_mode === "chat") parts.push("--chat");
  parts.push("--prompt", shellQuote(caseDef.prompt));
  return remoteCommand(parts.join(" "), creds);
}

export function rdnaZincCommand(caseDef, creds) {
  return remoteZincCommand(caseDef, {
    ...creds,
    commandPrelude: rdnaDpmHighScript(),
    env: { RADV_PERFTEST: "coop_matrix", ...(creds.env ?? {}) },
  });
}

export function intelZincCommand(caseDef, creds) {
  return remoteZincCommand(caseDef, creds);
}

function remoteLlamaCommand(caseDef, creds, llamaCliPath) {
  const parts = [
  ];
  const envPrefix = remoteEnvPrefix(creds);
  if (envPrefix) parts.push(envPrefix);
  parts.push(
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
    ...(creds.cliDeviceArgs ?? ["--device", `Vulkan${creds.vkDevice ?? 1}`]),
    "--flash-attn",
    "on",
    "-b",
    "4096",
    "-ub",
    "1024",
  );
  if (caseDef.prompt_mode === "chat") parts.push("-cnv");
  return remoteCommand(parts.join(" "), creds);
}

function rdnaLlamaCommand(caseDef, creds, llamaCliPath) {
  return remoteLlamaCommand(caseDef, { ...creds, env: { RADV_PERFTEST: "coop_matrix", ...(creds.env ?? {}) } }, llamaCliPath);
}

async function detectRemoteLlamaCliPath(creds, searchRoots, override = null) {
  if (override) return override;
  const quotedRoots = searchRoots.map((root) => shellQuote(root)).join(" ");
  const command = rdnaRemoteCommand(
    `which llama-cli || command -v llama-cli || find ${quotedRoots} -name llama-cli -type f 2>/dev/null | head -n 1`,
    creds,
  );
  const result = await runShell(command, { cwd: ROOT, timeoutMs: 120000 });
  const candidate = result.stdout.split(/\r?\n/).map((line) => line.trim()).find(Boolean);
  return candidate || null;
}

async function detectRemoteLlamaServerPath(creds, searchRoots, override = null) {
  if (override) return override;
  const quotedRoots = searchRoots.map((root) => shellQuote(root)).join(" ");
  const command = rdnaRemoteCommand(
    `which llama-server || command -v llama-server || find ${quotedRoots} -name llama-server -type f 2>/dev/null | head -n 1`,
    creds,
  );
  const result = await runShell(command, { cwd: ROOT, timeoutMs: 120000 });
  const candidate = result.stdout.split(/\r?\n/).map((line) => line.trim()).find(Boolean);
  return candidate || null;
}

async function detectRdnaLlamaCliPath(creds) {
  return detectRemoteLlamaCliPath(creds, ["/root/llama.cpp", "/root/workspace/llama.cpp"], process.env.ZINC_LLAMA_CLI_REMOTE ?? null);
}

async function detectRdnaLlamaServerPath(creds) {
  return detectRemoteLlamaServerPath(creds, ["/root/llama.cpp", "/root/workspace/llama.cpp"], process.env.ZINC_LLAMA_SERVER_REMOTE ?? null);
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
      id: "gemma4-26b-a4b-q4k-m",
      model_id: "gemma4-26b-a4b-q4k-m",
      label: "Gemma 4 26B-A4B MoE Q4_K_M",
      family: "Gemma 4",
      quant: "Q4_K_M",
      model_path: modelPath(modelRoot, "gemma4-26b-a4b-q4k-m"),
      prompt_mode: "chat",
      prompt: defaultPromptForModelId("gemma4-26b-a4b-q4k-m"),
      max_tokens: defaultMaxTokensForModelId("gemma4-26b-a4b-q4k-m"),
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
      prompt: defaultPromptForModelId("gemma4-31b-q4k-m"),
      max_tokens: defaultMaxTokensForModelId("gemma4-31b-q4k-m"),
      notes: ["Large-model Metal path"],
    },
    {
      id: "qwen35-9b-q4k-m",
      model_id: "qwen35-9b-q4k-m",
      label: "Qwen 3.5 9B Q4_K_M",
      family: "Qwen 3.5",
      quant: "Q4_K_M",
      model_path: modelPath(modelRoot, "qwen35-9b-q4k-m"),
      prompt_mode: "raw",
      prompt: defaultPromptForModelId("qwen35-9b-q4k-m"),
      max_tokens: defaultMaxTokensForModelId("qwen35-9b-q4k-m"),
      notes: ["Raw decode path to avoid visible think blocks in CLI output"],
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
    {
      id: "qwen36-27b-q4k-m",
      model_id: "qwen36-27b-q4k-m",
      label: "Qwen 3.6 27B Dense Q4_K_M",
      family: "Qwen 3.6",
      quant: "Q4_K_M",
      model_path: modelPath(modelRoot, "qwen36-27b-q4k-m"),
      prompt_mode: "raw",
      prompt: defaultPromptForModelId("qwen36-27b-q4k-m"),
      max_tokens: defaultMaxTokensForModelId("qwen36-27b-q4k-m"),
      notes: ["Dense hybrid Qwen 3.6 case on Apple Silicon"],
    },
  ];
}

function titleFromId(id) {
  if (id === "gemma4-26b-a4b-q4k-m") return "Gemma 4 26B-A4B MoE Q4_K_M";
  if (id === "gemma4-31b-q4k-m") return "Gemma 4 31B Q4_K_M";
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
    .replace(/^qwen-qwen3-6-/, "qwen36-")
    .replace(/^qwen-qwen-3-6-/, "qwen36-")
    .replace(/^qwen3-6-/, "qwen36-")
    .replace(/^qwen-3-6-/, "qwen36-")
    .replace(/^qwen-qwen3-5-/, "qwen35-")
    .replace(/^qwen-qwen-3-5-/, "qwen35-")
    .replace(/^qwen3-5-/, "qwen35-")
    .replace(/^qwen-3-5-/, "qwen35-")
    .replace(/^qwen-3-/, "qwen3-")
    .replace(/-ud-(q[0-9].*)$/, "-$1")
    .replace(/-q([0-9])-k-([a-z0-9]+)/g, "-q$1k-$2")
    .replace(/-q([0-9])-k$/g, "-q$1k")
    .replace(/^gemma-4-/, "gemma4-")
    .replace(/-it-(q[0-9].*)$/, "-$1");
}

export function guessFamily(id) {
  if (id.startsWith("gemma4")) return "Gemma 4";
  if (id.startsWith("gemma")) return "Gemma";
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
  return id.startsWith("gemma");
}

export function defaultPromptForModelId(id) {
  return prefersChatPrompt(id)
    ? "A teammate sent two local LLM benchmark screenshots with different tok/s numbers for the same model. Explain the likely causes in two short paragraphs, then give one concrete measurement rule."
    : "Developer question: two local LLM benchmark screenshots show different tok/s values for the same model. A useful answer explains likely causes and gives one fair measurement rule.\n\nAnswer:";
}

export function defaultMaxTokensForModelId(id) {
  return 96;
}

const CODING_REVIEW_SNIPPET = [
  "File: src/cache.ts",
  "```ts",
  "const cache = new Map<string, string>();",
  "const pending = new Map<string, Promise<string>>();",
  "",
  "export async function getValue(key: string, load: () => Promise<string>) {",
  "  if (cache.has(key)) return cache.get(key)!;",
  "  if (pending.has(key)) return cache.get(key)!;",
  "",
  "  const task = load().then((value) => {",
  "    cache.set(key, value);",
  "    pending.delete(key);",
  "    return value;",
  "  });",
  "  pending.set(key, task);",
  "  return task;",
  "}",
  "```",
].join("\n");

const INCIDENT_CONTEXT = [
  "Incident notes from a local inference service:",
  "- 09:14: A developer compared two screenshots and saw 49 tok/s in one run and 37 tok/s in another.",
  "- 09:18: The faster run generated one token after a short chat prompt. The slower run generated 160 tokens after a pasted support ticket.",
  "- 09:22: The CLI output reports separate lines for prefill throughput and generated-token throughput.",
  "- 09:27: The team sometimes runs with --profile, which adds per-dispatch accounting and changes CPU-side overhead.",
  "- 09:33: The llama.cpp baseline is measured through a persistent server, while some ZINC checks were one-off CLI runs.",
  "- 09:41: A background video export was active during one Apple Silicon run.",
  "- 09:48: The benchmark dashboard now computes an overall prompt+decode score from prompt tokens, generated tokens, prefill speed, and decode speed.",
  "",
  "Relevant policy:",
  "- Publish medians from repeated warm runs, not screenshots.",
  "- Keep the same model file, prompt mode, output cap, and backend residency for both engines.",
  "- Treat one-token completions as coherence smoke tests, not sustained throughput measurements.",
  "- Report prefill, decode, total latency, and prompt+decode throughput together.",
  "",
  "Question: summarize the measurement mistake and propose the benchmark protocol the team should use next.",
].join("\n");

const LONG_CODING_PLAN_PROMPT =
  "Write an implementation plan for adding a stable benchmark preset to a local LLM CLI. Include the command shape, warmup policy, metrics to collect, failure handling, llama.cpp comparison, and how the site should display prefill, decode, latency, and overall prompt+decode throughput.";

function contextualizePrompt(promptMode, basePrompt, variant) {
  if (variant === "core") return basePrompt;

  if (variant === "context-medium") {
    if (promptMode === "chat") {
      return [
        "You are reviewing a small TypeScript helper in a production service.",
        "Identify the bug, explain why it appears under concurrent requests, and provide a corrected version.",
        "",
        CODING_REVIEW_SNIPPET,
      ].join("\n");
    }
    return [
      "Code review request: identify the bug, explain why it appears under concurrent requests, and provide a corrected version.",
      "",
      CODING_REVIEW_SNIPPET,
      "",
      "Review:",
    ].join("\n");
  }

  if (variant === "context-long") {
    if (promptMode === "chat") {
      return `Read the incident context and answer the final question with concrete engineering guidance.\n\n${INCIDENT_CONTEXT}`;
    }
    return `${INCIDENT_CONTEXT}\n\nEngineering guidance:`;
  }

  if (variant === "decode-extended") {
    if (promptMode === "chat") return LONG_CODING_PLAN_PROMPT;
    return `${LONG_CODING_PLAN_PROMPT}\n\nPlan:\n1.`;
  }

  return basePrompt;
}

function scenarioMaxTokens(modelId, scenarioId) {
  const caps = {
    core: defaultMaxTokensForModelId(modelId),
    "context-medium": 160,
    "context-long": 128,
    "decode-extended": 256,
  };
  return caps[scenarioId] ?? defaultMaxTokensForModelId(modelId);
}

export function defaultScenarioDefsForModel(modelId, promptMode, basePrompt) {
  return [
    {
      id: "core",
      label: "Quick Chat",
      short_label: "Core",
      prompt_mode: promptMode,
      prompt: contextualizePrompt(promptMode, basePrompt, "core"),
      max_tokens: scenarioMaxTokens(modelId, "core"),
      notes: ["Short real-world assistant turn with enough decode budget to avoid one-token rates"],
    },
    {
      id: "context-medium",
      label: "Coding Review",
      short_label: "Coding",
      prompt_mode: promptMode,
      prompt: contextualizePrompt(promptMode, basePrompt, "context-medium"),
      max_tokens: scenarioMaxTokens(modelId, "context-medium"),
      notes: ["Small code-review workload with a realistic TypeScript snippet"],
    },
    {
      id: "context-long",
      label: "Incident Context",
      short_label: "Context",
      prompt_mode: promptMode,
      prompt: contextualizePrompt(promptMode, basePrompt, "context-long"),
      max_tokens: scenarioMaxTokens(modelId, "context-long"),
      notes: ["Support-style incident context that makes prefill visible"],
    },
    {
      id: "decode-extended",
      label: "Long Coding Draft",
      short_label: "Long draft",
      prompt_mode: promptMode,
      prompt: contextualizePrompt(promptMode, basePrompt, "decode-extended"),
      max_tokens: scenarioMaxTokens(modelId, "decode-extended"),
      notes: ["Longer coding-plan generation to measure sustained decode"],
    },
  ];
}

export function filterScenarioDefs(scenarios, selectedIds = null) {
  if (!selectedIds || selectedIds.size === 0) return scenarios;
  const selected = scenarios.filter((scenario) => selectedIds.has(scenario.id));
  const known = new Set(scenarios.map((scenario) => scenario.id));
  const unknown = [...selectedIds].filter((id) => !known.has(id));
  if (unknown.length > 0) {
    throw new Error(`Unknown scenario id(s): ${unknown.join(", ")}`);
  }
  if (selected.length === 0) {
    throw new Error("No benchmark scenarios selected");
  }
  return selected;
}

function phaseEnabled(requestedPhase, phase) {
  return requestedPhase === "all" || requestedPhase === phase;
}

export function buildMeasurementPhases(modelId, promptMode, basePrompt, requestedPhase = "all", selectedScenarios = null) {
  const scenarios = filterScenarioDefs(defaultScenarioDefsForModel(modelId, promptMode, basePrompt), selectedScenarios);
  const phases = [
    ...scenarios.map((scenarioDef) => ({ phase: "zinc", scenarioDef })),
    ...scenarios.map((scenarioDef) => ({ phase: "baseline", scenarioDef })),
  ];
  return phases.filter((phase) => phaseEnabled(requestedPhase, phase.phase));
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

export function defaultRdnaCases(modelRoot) {
  return [
    {
      id: "gemma4-26b-a4b-q4k-m",
      label: "Gemma 4 26B-A4B MoE Q4_K_M",
      family: "Gemma 4",
      quant: "Q4_K_M",
      model_path: path.join(modelRoot, "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
      prompt_mode: "chat",
      prompt: defaultPromptForModelId("gemma4-26b-a4b-q4k-m"),
      max_tokens: defaultMaxTokensForModelId("gemma4-26b-a4b-q4k-m"),
      notes: ["RDNA4 Gemma MoE comparison against llama.cpp server"],
    },
    {
      id: "gemma4-31b-q4k-m",
      label: "Gemma 4 31B Q4_K_M",
      family: "Gemma 4",
      quant: "Q4_K_M",
      model_path: path.join(modelRoot, "gemma-4-31B-it-Q4_K_M.gguf"),
      prompt_mode: "chat",
      prompt: defaultPromptForModelId("gemma4-31b-q4k-m"),
      max_tokens: defaultMaxTokensForModelId("gemma4-31b-q4k-m"),
      notes: ["RDNA4 dense Gemma comparison against llama.cpp server"],
    },
    {
      id: "qwen36-35b-a3b-q4k-xl",
      label: "Qwen 3.6 35B A3B UD Q4_K_XL",
      family: "Qwen 3.6",
      quant: "Q4_K_XL",
      model_path: path.join(modelRoot, "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
      prompt_mode: "raw",
      prompt: defaultPromptForModelId("qwen36-35b-a3b-q4k-xl"),
      max_tokens: defaultMaxTokensForModelId("qwen36-35b-a3b-q4k-xl"),
      notes: ["RDNA4 flagship comparison against llama.cpp server"],
    },
    {
      id: "qwen36-27b-q4k-m",
      label: "Qwen 3.6 27B Dense Q4_K_M",
      family: "Qwen 3.6",
      quant: "Q4_K_M",
      model_path: path.join(modelRoot, "Qwen3.6-27B-Q4_K_M.gguf"),
      prompt_mode: "raw",
      prompt: defaultPromptForModelId("qwen36-27b-q4k-m"),
      max_tokens: defaultMaxTokensForModelId("qwen36-27b-q4k-m"),
      notes: ["RDNA4 dense Qwen 3.6 comparison against llama.cpp server"],
    },
    {
      id: "qwen35-9b-q4k-m",
      label: "Qwen 3.5 9B Q4_K_M",
      family: "Qwen 3.5",
      quant: "Q4_K_M",
      model_path: path.join(modelRoot, "Qwen3.5-9B-Q4_K_M.gguf"),
      prompt_mode: "raw",
      prompt: defaultPromptForModelId("qwen35-9b-q4k-m"),
      max_tokens: defaultMaxTokensForModelId("qwen35-9b-q4k-m"),
      notes: ["RDNA4 small Qwen comparison against llama.cpp server"],
    },
  ];
}

export function defaultIntelCases(modelRoot) {
  return defaultMetalCases(modelRoot).map((entry) => ({
    ...entry,
    model_path: modelPath(modelRoot, entry.id),
    context_tokens: defaultIntelContextTokensForModel(entry.id),
    notes: ["Intel Arc Vulkan comparison against llama.cpp on the same host"],
  }));
}

function defaultIntelContextTokensForModel(_id) {
  return 512;
}

async function discoverRemoteCases(modelRoot, creds, note) {
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
        notes: [note],
      };
    });
}

async function discoverRdnaCases(modelRoot, creds) {
  return discoverRemoteCases(modelRoot, creds, "Auto-discovered on the RDNA benchmark node");
}

async function discoverIntelCases(modelRoot, creds) {
  const cases = await discoverRemoteCases(modelRoot, creds, "Auto-discovered on the Intel benchmark node");
  return cases.map((entry) => ({
    ...entry,
    context_tokens: entry.context_tokens ?? defaultIntelContextTokensForModel(entry.id),
  }));
}

async function rdnaPathExists(modelFile, creds) {
  try {
    await runShell(rdnaRemoteCommand(`test -f ${shellQuote(modelFile)}`, creds), { cwd: ROOT, timeoutMs: 30000 });
    return true;
  } catch {
    return false;
  }
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
      "--exclude '.git'",
      "--exclude '.env'",
      "--exclude '.env.*'",
      "--exclude '.zinc_rt_autopilot'",
      "--exclude '.zinc_optimize'",
      "--exclude '.llm_optimize'",
      "--exclude '.perf_optimize'",
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
    const buildArgs = ["zig", "build", "-Doptimize=ReleaseFast"];
    if (args.rdnaBackend !== "auto") {
      buildArgs.push(`-Dbackend=${args.rdnaBackend}`);
    }
    if (args.rdnaSync) {
      const syncedProvenance = await captureGitProvenance(ROOT);
      if (syncedProvenance.version) buildArgs.push(`-Dversion=${shellQuote(syncedProvenance.version)}`);
      if (syncedProvenance.commit) buildArgs.push(`-Dcommit=${shellQuote(syncedProvenance.commit)}`);
    }
    const remote = `cd ${shellQuote(creds.workdir)} && ${buildArgs.join(" ")}`;
    await runShell(rdnaRemoteCommand(remote, creds), { cwd: ROOT, timeoutMs: 60 * 60 * 1000 });
  }

  if (args.rdnaStartLlama) {
    // Pre-flight cleanup: stop any pre-existing llama-server processes
    // (systemd unit + leftover dynamic-port servers from prior crashed
    // runs). The per-case launchRdnaLlamaServer handles fresh launches
    // with the right flags + DPM lock; a co-resident systemd server
    // running an unrelated model just steals VRAM and tanks everything.
    // The historical regression on Qwen 3.6 35B (108 → 29 tok/s) was
    // partly caused by this co-residency.
    console.log("Stopping pre-existing llama-server processes on RDNA node...");
    const remote = [
      "systemctl stop llama-server 2>/dev/null || true",
      "pkill -9 -x llama-server 2>/dev/null || true",
      "sleep 2",
      "echo 'cleared'",
    ].join(" && ");
    await runShell(rdnaRemoteCommand(remote, creds), { cwd: ROOT, timeoutMs: 60000 });
  }
}

async function verifyRdnaZincBackend(args, creds, { quiet = false } = {}) {
  if (args.rdnaBackend === "auto") return null;
  const remote = `cd ${shellQuote(creds.workdir)} && ./zig-out/bin/zinc --version`;
  const result = await runShell(rdnaRemoteCommand(remote, creds), { cwd: ROOT, timeoutMs: 120000 });
  const parsed = validateZincBackend(result.stdout, args.rdnaBackend);
  if (!quiet) console.log(`[rdna] verified ZINC binary backend: ${parsed.backend}`);
  return parsed;
}

async function prepareIntel(args, creds, remoteLibcConf) {
  if (args.intelSync) {
    console.log("Syncing current repo to Intel node...");
    const rsyncCmd = [
      "rsync -az --delete",
      "--exclude '.zig-cache'",
      "--exclude 'zig-out'",
      "--exclude 'node_modules'",
      "--exclude '.git'",
      "--exclude '.env'",
      "--exclude '.env.*'",
      "--exclude '.zinc_rt_autopilot'",
      "--exclude '.zinc_optimize'",
      "--exclude '.llm_optimize'",
      "--exclude '.perf_optimize'",
      "--exclude '.DS_Store'",
      "--exclude 'site'",
      `-e ${shellQuote(remoteSshTransport(creds))}`,
      `${shellQuote(`${ROOT}/`)}`,
      `${creds.user}@${creds.host}:${shellQuote(`${creds.workdir}/`)}`,
    ].join(" ");
    await runShell(rsyncCmd, { cwd: ROOT, timeoutMs: 60 * 60 * 1000 });
  }

  if (remoteLibcConf) {
    const remote = `cd ${shellQuote(creds.workdir)} && mkdir -p .build-support && cp ${shellQuote(remoteLibcConf)} .build-support/libc.conf`;
    await runShell(rdnaRemoteCommand(remote, creds), { cwd: ROOT, timeoutMs: 120000 });
  }

  if (args.intelBuild) {
    console.log("Building ReleaseFast on Intel node...");
    const envPrefix = remoteEnvPrefix(creds);
    const remote = `cd ${shellQuote(creds.workdir)} && ${envPrefix ? `${envPrefix} ` : ""}zig build -Doptimize=ReleaseFast`;
    await runShell(rdnaRemoteCommand(remote, creds), { cwd: ROOT, timeoutMs: 60 * 60 * 1000 });
  }

  if (args.intelStartLlama) {
    console.log("Stopping pre-existing llama-server processes on Intel node...");
    const remote = [
      "pkill -9 -x llama-server 2>/dev/null || true",
      "sleep 2",
      "echo 'cleared'",
    ].join(" && ");
    await runShell(rdnaRemoteCommand(remote, creds), { cwd: ROOT, timeoutMs: 60000 });
  }
}

function buildScenarioCase(entry, scenarioDef) {
  return {
    ...entry,
    prompt_mode: scenarioDef.prompt_mode ?? entry.prompt_mode,
    prompt: scenarioDef.prompt,
    max_tokens: scenarioDef.max_tokens,
    context_tokens: scenarioDef.context_tokens ?? entry.context_tokens,
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
    comparison: baseline?.decode_tps ? buildComparison(zinc, baseline, { expectedGeneratedTokens: scenarioDef.max_tokens }) : null,
    output_quality: outputQualityStatus(zinc?.output_preview, zinc?.generated_tokens),
  };
}

function primaryScenarioSummary(entry, scenarios) {
  const primary = selectPrimaryScenario(scenarios);
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
    const scenarioDefs = filterScenarioDefs(defaultScenarioDefsForModel(entry.id, entry.prompt_mode, entry.prompt), args.scenarios);
    const phases = buildMeasurementPhases(entry.id, entry.prompt_mode, entry.prompt, args.phase, args.scenarios);
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
              unavailable_reason: benchmarkFailureReason("ZINC run failed", error),
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
              unavailable_reason: benchmarkFailureReason("Local baseline failed", error),
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
              unavailable_reason: benchmarkFailureReason("Local baseline failed", error),
            };
          }
        } else if (serverLaunchError) {
          baseline = {
            name: "llama.cpp",
            unavailable_reason: benchmarkFailureReason("Local baseline failed", serverLaunchError),
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

    const includeZinc = phaseEnabled(args.phase, "zinc");
    const includeBaseline = phaseEnabled(args.phase, "baseline");
    const scenarios = scenarioDefs.map((scenarioDef) => scenarioResultPayload(
      entry,
      scenarioDef,
      zincByScenario.get(scenarioDef.id) ?? (includeZinc ? {
        name: "ZINC",
        unavailable_reason: "No ZINC benchmark result was recorded for this scenario.",
      } : undefined),
      baselineByScenario.get(scenarioDef.id) ?? (includeBaseline ? {
        name: "llama.cpp",
        unavailable_reason: "No llama.cpp baseline result was recorded for this scenario.",
      } : undefined),
    ));
    const summary = primaryScenarioSummary(entry, scenarios);
    models.push(summary);
    logModelSummary("metal", summary);
    await writePartialSnapshot(args, "metal", "Metal", models);
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
      runner: "ZINC CLI vs llama.cpp on the same Apple Silicon machine",
      benchmark_style: "Four-scenario matrix with same-file baselines",
      notes: [
        "Hardware: the local Apple Silicon target shown above. ZINC and llama.cpp run on the same machine and use the same GGUF file for each model.",
        "ZINC path: build the current workspace with zig build -Doptimize=ReleaseFast, then measure generation through the ZINC CLI.",
        "Baseline path: launch llama.cpp against the same model file, preferring one reusable llama-server per model across the full scenario matrix and falling back to llama-cli when the server path is unavailable.",
        "Scenarios: Quick Chat, Coding Review, Incident Context, and Long Coding Draft. The prompts are real-world chat, code-review, support-context, and coding-plan workloads instead of synthetic factual completions.",
        "Statistics: one warmup pass is discarded, then three measured runs are collected. Published prefill, decode, end-to-end throughput, and latency values are medians.",
        "Model resolution: managed catalog models are loaded from the ZINC model cache when available; explicit GGUF paths are preserved when provided.",
        "Run hygiene: published local runs assume no competing workload is saturating CPU, memory bandwidth, or GPU.",
      ],
      runs: args.runs,
      warmup_runs: args.warmupRuns,
    },
    summary: targetSummary(models),
    models,
  };
}

function envValueFrom(envMap, dotEnv, ...keys) {
  for (const key of keys) {
    const value = envMap[key] ?? dotEnv[key];
    if (value != null && value !== "") return value;
  }
  return null;
}

function envValue(dotEnv, ...keys) {
  return envValueFrom(process.env, dotEnv, ...keys);
}

export function rdnaNodeEnvKey(node, suffix) {
  const normalized = String(node ?? "")
    .trim()
    .toUpperCase()
    .replace(/[^A-Z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  if (!normalized) return null;
  return `ZINC_${normalized}_${suffix}`;
}

export function rdnaEnvValue(dotEnv, args, suffix, ...fallbackKeys) {
  const envMap = args?.envMap ?? process.env;
  const node = args?.rdnaNode ?? envValueFrom(envMap, dotEnv, "ZINC_RDNA_NODE", "ZINC_NODE");
  const nodeKey = node ? rdnaNodeEnvKey(node, suffix) : null;
  return envValueFrom(envMap, dotEnv, ...(nodeKey ? [nodeKey] : []), ...fallbackKeys);
}

function remoteHomeForUser(user) {
  return user === "root" ? "/root" : `/home/${user}`;
}

async function buildRdnaCreds(args) {
  const dotEnv = await readDotEnv(path.join(ROOT, ".env"));
  const host = rdnaEnvValue(dotEnv, args, "HOST", "ZINC_RDNA_HOST", "ZINC_HOST");
  const user = rdnaEnvValue(dotEnv, args, "USER", "ZINC_RDNA_USER", "ZINC_USER");
  const port = rdnaEnvValue(dotEnv, args, "PORT", "ZINC_RDNA_PORT", "ZINC_PORT") ?? "22";
  const sshKey = rdnaEnvValue(dotEnv, args, "SSH_KEY", "ZINC_RDNA_SSH_KEY", "ZINC_SSH_KEY");
  if (!host || !user) {
    throw new Error("RDNA benchmarking needs node-specific ZINC_<NODE>_HOST/ZINC_<NODE>_USER, ZINC_RDNA_HOST/ZINC_RDNA_USER, or ZINC_HOST/ZINC_USER in the environment or .env");
  }
  return {
    host,
    user,
    port,
    sshKey,
    workdir: args.rdnaWorkdir,
    env: { RADV_PERFTEST: "coop_matrix" },
    vkDevice: args.rdnaVkDevice ?? 1,
  };
}

async function buildIntelConfig(args) {
  const dotEnv = await readDotEnv(path.join(ROOT, ".env"));
  const host = envValue(dotEnv, "ZINC_INTEL_HOST");
  const user = envValue(dotEnv, "ZINC_INTEL_USER") ?? "tempuser";
  const port = envValue(dotEnv, "ZINC_INTEL_PORT") ?? "22";
  const sshKey = envValue(dotEnv, "ZINC_INTEL_SSH_KEY", "ZINC_SSH_KEY");
  if (!host || !user) {
    throw new Error("Intel benchmarking needs ZINC_INTEL_HOST and ZINC_INTEL_USER in the environment or .env");
  }
  const vkDeviceRaw = envValue(dotEnv, "ZINC_INTEL_VK_DEVICE") ?? "0";
  if (!/^\d+$/.test(vkDeviceRaw)) {
    throw new Error(`Invalid ZINC_INTEL_VK_DEVICE '${vkDeviceRaw}'. Expected a Vulkan GPU index.`);
  }
  const vkDevice = Number.parseInt(vkDeviceRaw, 10);
  const remoteHome = envValue(dotEnv, "ZINC_INTEL_REMOTE_HOME") ?? remoteHomeForUser(user);
  const xdgCacheHome = args.intelXdgCacheHome
    ?? envValue(dotEnv, "ZINC_INTEL_XDG_CACHE_HOME", "ZINC_INTEL_REMOTE_XDG_CACHE_HOME")
    ?? `${remoteHome}/.cache`;
  const workdir = args.intelWorkdir ?? envValue(dotEnv, "ZINC_INTEL_WORKDIR", "ZINC_INTEL_REMOTE_DIR") ?? `${remoteHome}/zinc-gpu-loop`;
  const modelRoot = args.intelModelRoot ?? envValue(dotEnv, "ZINC_INTEL_MODEL_ROOT") ?? `${xdgCacheHome}/zinc/models/models`;
  const remoteLibcConf = args.intelRemoteLibcConf ?? envValue(dotEnv, "ZINC_INTEL_REMOTE_LIBC_CONF");
  const remoteZig = envValue(dotEnv, "ZINC_INTEL_ZIG_REMOTE");
  const remoteZigDir = envValue(dotEnv, "ZINC_INTEL_ZIG_DIR", "ZINC_INTEL_REMOTE_ZIG_DIR")
    ?? (remoteZig ? path.posix.dirname(remoteZig) : `${remoteHome}/.local/zig/0.15.2`);
  const llamaDevice = envValue(dotEnv, "ZINC_INTEL_LLAMA_DEVICE") ?? "Vulkan0";
  const intelLlamaDeviceArgs = llamaDeviceArgs(llamaDevice);
  const remoteEnv = {
    PATH: `${remoteZigDir}:/usr/local/sbin:/usr/local/bin:/usr/bin:/bin`,
    ...collectRemoteZincTuningEnv(dotEnv),
  };
  const llamaSearchRoots = [
    `${remoteHome}/llama.cpp`,
    "/workspace/llama.cpp",
    "/workspace",
  ];
  return {
    creds: {
      host,
      user,
      port,
      sshKey,
      workdir,
      env: remoteEnv,
      vkDevice,
      serverDeviceArgs: intelLlamaDeviceArgs,
      cliDeviceArgs: intelLlamaDeviceArgs,
    },
    remoteHome,
    xdgCacheHome,
    modelRoot,
    remoteLibcConf,
    requireDeviceSubstring: envValue(dotEnv, "ZINC_INTEL_REQUIRE_DEVICE_SUBSTRING") ?? "Intel",
    llamaCliOverride: envValue(dotEnv, "ZINC_INTEL_LLAMA_CLI_REMOTE"),
    llamaServerOverride: envValue(dotEnv, "ZINC_INTEL_LLAMA_SERVER_REMOTE"),
    llamaSearchRoots,
  };
}

async function verifyRemoteVulkanDevice(creds, requireSubstring, options = {}) {
  const targetId = options.targetId ?? "rdna";
  const nodeLabel = options.nodeLabel ?? `${targetId.toUpperCase()} node`;
  const selectorHelp = options.selectorHelp ?? "Pick the discrete GPU with --rdna-vk-device <n> or ZINC_RDNA_VK_DEVICE=<n>.";
  const envPrefix = remoteEnvPrefix(creds);
  const cmd = remoteCommand(`${envPrefix ? `${envPrefix} ` : ""}vulkaninfo --summary 2>/dev/null || true`, creds);
  const { stdout } = await runShell(cmd, { cwd: ROOT, timeoutMs: 30000 });
  const blocks = stdout.split(/^GPU(\d+):/m);
  const devices = [];
  for (let i = 1; i < blocks.length; i += 2) {
    const idx = Number(blocks[i]);
    const body = blocks[i + 1] ?? "";
    const name = (body.match(/deviceName\s*=\s*(.+)/) ?? [])[1]?.trim() ?? "";
    const type = (body.match(/deviceType\s*=\s*(\S+)/) ?? [])[1]?.trim() ?? "";
    devices.push({ idx, name, type });
  }
  if (devices.length === 0) {
    throw new Error(`Could not parse vulkaninfo --summary on ${nodeLabel}; cannot verify selected device.`);
  }
  const selected = devices.find((d) => d.idx === creds.vkDevice);
  const summary = devices.map((d) => `  GPU${d.idx}: ${d.type} ${d.name}`).join("\n");
  if (!selected) {
    throw new Error(`Selected Vulkan device index ${creds.vkDevice} not present on ${nodeLabel}.\nDetected:\n${summary}`);
  }
  if (selected.type !== "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU") {
    throw new Error(
      `Refusing to run: Vulkan device ${creds.vkDevice} is ${selected.type} (${selected.name}), not a discrete GPU.\n` +
      `Detected:\n${summary}\n` +
      selectorHelp,
    );
  }
  if (requireSubstring && !selected.name.includes(requireSubstring)) {
    throw new Error(
      `Refusing to run: Vulkan device ${creds.vkDevice} name '${selected.name}' does not include required substring '${requireSubstring}'.\n` +
      `Detected:\n${summary}`,
    );
  }
  process.stdout.write(`[${targetId}] Vulkan device ${creds.vkDevice}: ${selected.name} (${selected.type})\n`);
}

async function runRdnaTarget(args) {
  const creds = await buildRdnaCreds(args);
  await prepareRdna(args, creds);
  await verifyRdnaZincBackend(args, creds);
  await verifyRemoteVulkanDevice(creds, args.requireRdnaDeviceSubstring);
  const rdnaLlamaCli = await detectRdnaLlamaCliPath(creds);
  const rdnaLlamaServer = await detectRdnaLlamaServerPath(creds);
  const baselineBinary = rdnaLlamaServer || rdnaLlamaCli;
  const zincProvenance = args.rdnaSync
    ? await captureGitProvenance(ROOT)
    : await captureRemoteGitProvenance(creds);
  const llamaCppProvenance = await captureRemoteLlamaCppProvenance(baselineBinary, creds);

  const knownCases = defaultRdnaCases(args.rdnaModelRoot);
  const discoveredCases = args.discoverModels ? await discoverRdnaCases(args.rdnaModelRoot, creds) : [];
  const mergedCases = new Map();
  for (const entry of [...discoveredCases, ...knownCases]) {
    mergedCases.set(entry.id, entry);
  }
  const cases = [];
  for (const entry of mergedCases.values()) {
    if (args.models && !args.models.has(entry.id)) continue;
    if (!(await rdnaPathExists(entry.model_path, creds))) {
      console.log(`[rdna] skipping ${entry.id}: missing GGUF at ${entry.model_path}`);
      continue;
    }
    cases.push(entry);
  }
  const models = [];

  for (const entry of cases) {
    console.log(`[rdna] ${entry.id}`);
    const scenarioDefs = filterScenarioDefs(defaultScenarioDefsForModel(entry.id, entry.prompt_mode, entry.prompt), args.scenarios);
    const phases = buildMeasurementPhases(entry.id, entry.prompt_mode, entry.prompt, args.phase, args.scenarios);
    const zincByScenario = new Map();
    const baselineByScenario = new Map();
    let launchedZincServer = null;
    let triedLaunchingZincServer = false;
    let zincServerLaunchFailure = null;
    let launchedServer = null;
    let triedLaunchingServer = false;
    let serverLaunchFailure = null;
    try {
      for (const phase of phases) {
        const scenarioDef = phase.scenarioDef;
        const caseDef = buildScenarioCase(entry, scenarioDef);
        if (phase.phase === "zinc") {
          let zinc = null;
          if (!triedLaunchingZincServer) {
            triedLaunchingZincServer = true;
            try {
              await verifyRdnaZincBackend(args, creds, { quiet: true });
              launchedZincServer = await launchRdnaZincServer(entry, creds, args.timeoutMs);
            } catch (error) {
              launchedZincServer = null;
              zincServerLaunchFailure = error;
            }
          }
          try {
            if (!launchedZincServer) {
              throw zincServerLaunchFailure ?? new Error("ZINC server was not launched");
            }
            const zincRows = await runRdnaZincOpenAiSeries({
              label: `zinc-server ${entry.id} ${scenarioDef.id}`,
              warmupRuns: args.warmupRuns,
              runs: args.runs,
              creds,
              port: launchedZincServer.port,
              logPath: launchedZincServer.logPath,
              caseDef,
              timeoutMs: args.timeoutMs,
            });
            zinc = createStats("ZINC", zincRows);
          } catch (error) {
            zinc = {
              name: "ZINC",
              unavailable_reason: benchmarkFailureReason("ZINC run failed", error),
            };
          }
          zincByScenario.set(scenarioDef.id, zinc);
          continue;
        }

        if (launchedZincServer) {
          await stopRdnaZincServer(creds, launchedZincServer.port);
          launchedZincServer = null;
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
              unavailable_reason: benchmarkFailureReason("RDNA baseline failed", error),
            };
          }
        } else if (entry.baseline_enabled !== false && serverLaunchFailure) {
          baseline = {
            name: "llama.cpp",
            unavailable_reason: benchmarkFailureReason("RDNA baseline failed", serverLaunchFailure),
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
              unavailable_reason: benchmarkFailureReason("RDNA baseline failed", error),
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
      if (launchedZincServer) {
        await stopRdnaZincServer(creds, launchedZincServer.port);
      }
      if (launchedServer) {
        await stopRdnaLlamaServer(creds, launchedServer.port);
      }
    }

    const includeZinc = phaseEnabled(args.phase, "zinc");
    const includeBaseline = phaseEnabled(args.phase, "baseline");
    const scenarios = scenarioDefs.map((scenarioDef) => scenarioResultPayload(
      entry,
      scenarioDef,
      zincByScenario.get(scenarioDef.id) ?? (includeZinc ? {
        name: "ZINC",
        unavailable_reason: "No ZINC benchmark result was recorded for this scenario.",
      } : undefined),
      baselineByScenario.get(scenarioDef.id) ?? (includeBaseline ? {
        name: "llama.cpp",
        unavailable_reason: "No llama.cpp baseline result was recorded for this scenario.",
      } : undefined),
    ));
    const summary = primaryScenarioSummary(entry, scenarios);
    models.push(summary);
    logModelSummary("rdna", summary);
    await writePartialSnapshot(args, "rdna", "RDNA", models);
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
      runner: "ZINC server vs llama.cpp server on the same RDNA node",
      benchmark_style: "Four-scenario matrix with same-file baselines",
      notes: [
        "Hardware: one AMD Radeon AI PRO R9700 benchmark node with 32 GB VRAM and 576 GB/s memory bandwidth. ZINC and llama.cpp run on the same Ubuntu host and use the same GGUF file for each model.",
        "ZINC path: sync the current source tree to the RDNA node, build with zig build -Doptimize=ReleaseFast, then measure generation through one reusable ZINC server per model with RADV cooperative matrix support enabled.",
        `ZINC RDNA backend: ${args.rdnaBackend}. Published RDNA runs default to Vulkan; zinc_rt is opt-in because it is a separate bring-up runtime.`,
        "Baseline path: launch llama.cpp against the same model file, preferring one reusable llama-server per model across the full scenario matrix and falling back to llama-cli when the server path is unavailable.",
        "Scenarios: Quick Chat, Coding Review, Incident Context, and Long Coding Draft. The prompts are real-world chat, code-review, support-context, and coding-plan workloads instead of synthetic factual completions.",
        "Statistics: one warmup pass is discarded, then three measured runs are collected. Published prefill, decode, end-to-end throughput, and latency values are medians.",
        "Execution order: the harness records the full ZINC scenario matrix before starting the llama.cpp baseline phase, so only one inference engine owns the GPU during measurement.",
        "Run hygiene: published RDNA runs assume a clean node with no competing ZINC, llama.cpp, or other GPU workloads.",
      ],
      runs: args.runs,
      warmup_runs: args.warmupRuns,
    },
    summary: targetSummary(models),
    models,
  };
}

// ---------------------------------------------------------------------------
// CUDA target (NVIDIA, e.g. RTX 5090). Mirrors the rdna path: SSH to the node,
// sync the tree, build -Dbackend=cuda, then run the ZINC CLI vs llama.cpp across
// the four-scenario matrix. The GPU is pinned through CUDA_VISIBLE_DEVICES, so
// no Vulkan device flag is used. Host/port/user/GPU selector come from env
// (ZINC_CUDA_* / ZINC_HOST), never committed — see docs/BENCHMARKING.md.
// ---------------------------------------------------------------------------
function cudaEnvValue(dotEnv, args, suffix, ...fallbackKeys) {
  const envMap = args?.envMap ?? process.env;
  const node = args?.cudaNode ?? envValueFrom(envMap, dotEnv, "ZINC_CUDA_NODE");
  const nodeKey = node ? rdnaNodeEnvKey(node, suffix) : null;
  return envValueFrom(envMap, dotEnv, ...(nodeKey ? [nodeKey] : []), ...fallbackKeys);
}

async function buildCudaCreds(args) {
  const dotEnv = await readDotEnv(path.join(ROOT, ".env"));
  const host = cudaEnvValue(dotEnv, args, "HOST", "ZINC_CUDA_HOST", "ZINC_HOST");
  const user = cudaEnvValue(dotEnv, args, "USER", "ZINC_CUDA_USER", "ZINC_USER");
  const port = cudaEnvValue(dotEnv, args, "PORT", "ZINC_CUDA_PORT", "ZINC_PORT") ?? "22";
  const sshKey = cudaEnvValue(dotEnv, args, "SSH_KEY", "ZINC_CUDA_SSH_KEY", "ZINC_SSH_KEY");
  if (!host || !user) {
    throw new Error("CUDA benchmarking needs ZINC_CUDA_HOST/ZINC_CUDA_USER (or ZINC_HOST/ZINC_USER) in the environment or .env");
  }
  const gpuUuid = args.cudaGpuUuid
    ?? cudaEnvValue(dotEnv, args, "GPU_UUID", "ZINC_CUDA_GPU_UUID", "ZINC_CUDA_GPU");
  if (!gpuUuid) {
    throw new Error("CUDA benchmarking needs --cuda-gpu or ZINC_CUDA_GPU_UUID/ZINC_CUDA_GPU in the environment or .env");
  }
  const remoteHome = remoteHomeForUser(user);
  const workdir = args.cudaWorkdir ?? cudaEnvValue(dotEnv, args, "WORKDIR", "ZINC_CUDA_WORKDIR") ?? `${remoteHome}/zinc-bench-cuda`;
  const modelRoot = args.cudaModelRoot ?? cudaEnvValue(dotEnv, args, "MODEL_ROOT", "ZINC_CUDA_MODEL_ROOT") ?? `${remoteHome}/workspace/models`;
  const llamaServerOverride = args.cudaLlamaServer ?? cudaEnvValue(dotEnv, args, "LLAMA_SERVER_REMOTE", "ZINC_CUDA_LLAMA_SERVER_REMOTE") ?? null;
  return {
    host,
    user,
    port,
    sshKey,
    workdir,
    // CUDA_VISIBLE_DEVICES pins the target GPU by UUID for both the ZINC CLI and
    // the llama-server baseline (remoteEnvPrefix / the launch script inject it).
    env: { CUDA_VISIBLE_DEVICES: gpuUuid, ...collectRemoteZincTuningEnv(dotEnv) },
    // No Vulkan device flag on CUDA; the env var above already isolates the GPU.
    serverDeviceArgs: [],
    cliDeviceArgs: [],
    // This WSL2 box is net-fenced to 127.99/16; plain 127.0.0.1 loopback is
    // blackholed (SYN dropped), so the llama-server bind + its health/API curls
    // must use a 127.99.x.x address. Overridable via ZINC_CUDA_LOOPBACK_HOST.
    loopbackHost: args.cudaLoopbackHost ?? process.env.ZINC_CUDA_LOOPBACK_HOST ?? "127.99.0.1",
    // The box's zig lives at ~/zig-0.15.2/zig and is not on the non-interactive
    // SSH PATH; use the absolute path for the build. Overridable via ZINC_CUDA_ZIG.
    zig: process.env.ZINC_CUDA_ZIG ?? `${remoteHome}/zig-0.15.2/zig`,
    gpuUuid,
    modelRoot,
    llamaServerOverride,
    llamaSearchRoots: [`${remoteHome}/workspace/llama.cpp`, `${remoteHome}/llama.cpp`, "/workspace/llama.cpp"],
  };
}

export function defaultCudaCases(modelRoot) {
  return defaultRdnaCases(modelRoot).map((entry) => ({
    ...entry,
    notes: ["RTX 5090 (CUDA) comparison against llama.cpp server on the same host"],
  }));
}

async function prepareCuda(args, creds) {
  if (args.cudaSync) {
    console.log("Syncing current repo to CUDA node...");
    const rsyncCmd = [
      "rsync -az --delete",
      "--exclude '.zig-cache'",
      "--exclude 'zig-out'",
      "--exclude 'node_modules'",
      "--exclude '.git'",
      "--exclude '.env'",
      "--exclude '.env.*'",
      "--exclude '.zinc_rt_autopilot'",
      "--exclude '.zinc_optimize'",
      "--exclude '.llm_optimize'",
      "--exclude '.perf_optimize'",
      "--exclude '.DS_Store'",
      "--exclude 'site'",
      `-e ${shellQuote(remoteSshTransport(creds))}`,
      `${shellQuote(`${ROOT}/`)}`,
      `${creds.user}@${creds.host}:${shellQuote(`${creds.workdir}/`)}`,
    ].join(" ");
    await runShell(rsyncCmd, { cwd: ROOT, timeoutMs: 60 * 60 * 1000 });
  }

  if (args.cudaBuild) {
    console.log("Building ReleaseFast (cuda) on CUDA node...");
    // CUDA kernels are NVRTC-compiled at runtime for the visible GPU's compute
    // capability, so shader precompilation is off (-Dshaders=false) and one
    // build runs on any visible NVIDIA GPU.
    const buildArgs = [creds.zig ?? "zig", "build", "-Doptimize=ReleaseFast", "-Dbackend=cuda", "-Dshaders=false"];
    if (args.cudaSync) {
      const syncedProvenance = await captureGitProvenance(ROOT);
      if (syncedProvenance.version) buildArgs.push(`-Dversion=${shellQuote(syncedProvenance.version)}`);
      if (syncedProvenance.commit) buildArgs.push(`-Dcommit=${shellQuote(syncedProvenance.commit)}`);
    }
    const remote = `cd ${shellQuote(creds.workdir)} && ${buildArgs.join(" ")}`;
    await runShell(rdnaRemoteCommand(remote, creds), { cwd: ROOT, timeoutMs: 60 * 60 * 1000 });
  }

  if (args.cudaStartLlama) {
    console.log("Stopping pre-existing llama-server processes on CUDA node...");
    const remote = [
      "pkill -9 -x llama-server 2>/dev/null || true",
      "sleep 2",
      "echo 'cleared'",
    ].join(" && ");
    await runShell(rdnaRemoteCommand(remote, creds), { cwd: ROOT, timeoutMs: 60000 });
  }
}

async function verifyCudaZincBackend(creds, { quiet = false } = {}) {
  const remote = `cd ${shellQuote(creds.workdir)} && ./zig-out/bin/zinc --version`;
  const result = await runShell(rdnaRemoteCommand(remote, creds), { cwd: ROOT, timeoutMs: 120000 });
  const parsed = validateZincBackend(result.stdout, "cuda");
  if (!quiet) console.log(`[cuda] verified ZINC binary backend: ${parsed.backend}`);
  return parsed;
}

async function runCudaTarget(args) {
  const creds = await buildCudaCreds(args);
  await prepareCuda(args, creds);
  await verifyCudaZincBackend(creds);
  const cudaLlamaServer = await detectRemoteLlamaServerPath(creds, creds.llamaSearchRoots, creds.llamaServerOverride);
  const cudaLlamaCli = await detectRemoteLlamaCliPath(creds, creds.llamaSearchRoots, process.env.ZINC_CUDA_LLAMA_CLI_REMOTE ?? null);
  const baselineBinary = cudaLlamaServer || cudaLlamaCli;
  const zincProvenance = args.cudaSync
    ? await captureGitProvenance(ROOT)
    : await captureRemoteGitProvenance(creds);
  const llamaCppProvenance = await captureRemoteLlamaCppProvenance(baselineBinary, creds);

  const knownCases = defaultCudaCases(creds.modelRoot);
  const discoveredCases = args.discoverModels ? await discoverRemoteCases(creds.modelRoot, creds, "Auto-discovered on the CUDA benchmark node") : [];
  const mergedCases = new Map();
  for (const entry of [...discoveredCases, ...knownCases]) {
    mergedCases.set(entry.id, entry);
  }
  const cases = [];
  for (const entry of mergedCases.values()) {
    if (args.models && !args.models.has(entry.id)) continue;
    if (!(await rdnaPathExists(entry.model_path, creds))) {
      console.log(`[cuda] skipping ${entry.id}: missing GGUF at ${entry.model_path}`);
      continue;
    }
    cases.push(entry);
  }
  const models = [];

  for (const entry of cases) {
    console.log(`[cuda] ${entry.id}`);
    const scenarioDefs = filterScenarioDefs(defaultScenarioDefsForModel(entry.id, entry.prompt_mode, entry.prompt), args.scenarios);
    const phases = buildMeasurementPhases(entry.id, entry.prompt_mode, entry.prompt, args.phase, args.scenarios);
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
            await verifyCudaZincBackend(creds, { quiet: true });
            const zincRows = await runSeries({
              label: `zinc ${entry.id} ${scenarioDef.id}`,
              warmupRuns: args.warmupRuns,
              runs: args.runs,
              command: remoteZincCommand(caseDef, creds),
              parser: parseZincCliOutput,
              cwd: ROOT,
              timeoutMs: args.timeoutMs,
            });
            zinc = createStats("ZINC", zincRows);
          } catch (error) {
            zinc = {
              name: "ZINC",
              unavailable_reason: benchmarkFailureReason("ZINC run failed", error),
            };
          }
          zincByScenario.set(scenarioDef.id, zinc);
          continue;
        }

        if (!triedLaunchingServer && cudaLlamaServer) {
          triedLaunchingServer = true;
          try {
            launchedServer = await launchRdnaLlamaServer(entry, creds, cudaLlamaServer, args.timeoutMs, "cuda", false);
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
              unavailable_reason: benchmarkFailureReason("CUDA baseline failed", error),
            };
          }
        } else if (entry.baseline_enabled !== false && serverLaunchFailure) {
          baseline = {
            name: "llama.cpp",
            unavailable_reason: benchmarkFailureReason("CUDA baseline failed", serverLaunchFailure),
          };
        } else if (entry.baseline_enabled !== false && cudaLlamaCli) {
          try {
            const baselineRows = await runSeries({
              label: `llama.cpp ${entry.id} ${scenarioDef.id}`,
              warmupRuns: args.warmupRuns,
              runs: args.runs,
              command: remoteLlamaCommand(caseDef, creds, cudaLlamaCli),
              parser: parseLlamaCliOutput,
              cwd: ROOT,
              timeoutMs: args.timeoutMs,
            });
            baseline = createStats("llama.cpp", baselineRows);
          } catch (error) {
            baseline = {
              name: "llama.cpp",
              unavailable_reason: benchmarkFailureReason("CUDA baseline failed", error),
            };
          }
        } else {
          baseline = {
            name: "llama.cpp",
            unavailable_reason: cudaLlamaServer
              ? "No CUDA llama.cpp baseline is defined for this case yet."
              : "Could not locate a remote llama.cpp baseline binary on the CUDA node.",
          };
        }

        baselineByScenario.set(scenarioDef.id, baseline);
      }
    } finally {
      if (launchedServer) {
        await stopRdnaLlamaServer(creds, launchedServer.port);
      }
    }

    const includeZinc = phaseEnabled(args.phase, "zinc");
    const includeBaseline = phaseEnabled(args.phase, "baseline");
    const scenarios = scenarioDefs.map((scenarioDef) => scenarioResultPayload(
      entry,
      scenarioDef,
      zincByScenario.get(scenarioDef.id) ?? (includeZinc ? {
        name: "ZINC",
        unavailable_reason: "No ZINC benchmark result was recorded for this scenario.",
      } : undefined),
      baselineByScenario.get(scenarioDef.id) ?? (includeBaseline ? {
        name: "llama.cpp",
        unavailable_reason: "No llama.cpp baseline result was recorded for this scenario.",
      } : undefined),
    ));
    const summary = primaryScenarioSummary(entry, scenarios);
    models.push(summary);
    logModelSummary("cuda", summary);
    await writePartialSnapshot(args, "cuda", "RTX 5090", models);
  }

  return {
    id: "cuda",
    label: "RTX 5090",
    description: "NVIDIA RTX 5090 (CUDA) benchmark node results collected over SSH and compared against llama.cpp on the same hardware.",
    generated_at: new Date().toISOString(),
    source: "tools/performance_suite.mjs",
    machine: {
      label: "NVIDIA CUDA bench node",
      machine_model: "Remote Linux host (WSL2)",
      chip: "NVIDIA GeForce RTX 5090",
      memory: "32 GB VRAM · 1792 GB/s",
      gpu: "GeForce RTX 5090",
      os: "Ubuntu (WSL2)",
    },
    provenance: {
      zinc: zincProvenance,
      llama_cpp: llamaCppProvenance,
    },
    methodology: {
      runner: "ZINC CLI vs llama.cpp on the same CUDA node",
      benchmark_style: "Four-scenario matrix with same-file baselines",
      notes: [
        "Hardware: one NVIDIA GeForce RTX 5090 (32 GB VRAM, 1792 GB/s) benchmark node. ZINC and llama.cpp run on the same host and use the same GGUF file for each model.",
        "ZINC path: sync the current source tree to the CUDA node, build with zig build -Doptimize=ReleaseFast -Dbackend=cuda, then measure generation through the ZINC CLI. CUDA kernels are NVRTC-compiled at runtime for the visible GPU.",
        "GPU selection: the target GPU is pinned by UUID via CUDA_VISIBLE_DEVICES for both engines, because nvidia-smi indices are unreliable on the WSL2 passthrough.",
        "Baseline path: launch llama.cpp against the same model file, preferring one reusable llama-server per model across the full scenario matrix and falling back to llama-cli when the server path is unavailable.",
        "Scenarios: Quick Chat, Coding Review, Incident Context, and Long Coding Draft. The prompts are real-world chat, code-review, support-context, and coding-plan workloads instead of synthetic factual completions.",
        "Statistics: one warmup pass is discarded, then three measured runs are collected. Published prefill, decode, end-to-end throughput, and latency values are medians.",
        "Execution order: the harness records the full ZINC scenario matrix before starting the llama.cpp baseline phase, so only one inference engine owns the GPU during measurement.",
        "Run hygiene: published CUDA runs assume a clean node with no competing ZINC, llama.cpp, or other GPU workloads.",
      ],
      runs: args.runs,
      warmup_runs: args.warmupRuns,
    },
    summary: targetSummary(models),
    models,
  };
}

async function runIntelTarget(args) {
  const config = await buildIntelConfig(args);
  const creds = config.creds;
  await verifyRemoteVulkanDevice(creds, config.requireDeviceSubstring, {
    targetId: "intel",
    nodeLabel: "Intel node",
    selectorHelp: "Set ZINC_INTEL_VK_DEVICE=<n> to the Intel discrete GPU index.",
  });
  await prepareIntel(args, creds, config.remoteLibcConf);
  const intelLlamaCli = await detectRemoteLlamaCliPath(creds, config.llamaSearchRoots, config.llamaCliOverride);
  const intelLlamaServer = await detectRemoteLlamaServerPath(creds, config.llamaSearchRoots, config.llamaServerOverride);
  const baselineBinary = intelLlamaServer || intelLlamaCli;
  const zincProvenance = args.intelSync
    ? await captureGitProvenance(ROOT)
    : await captureRemoteGitProvenance(creds);
  const llamaCppProvenance = await captureRemoteLlamaCppProvenance(baselineBinary, creds);

  const knownCases = defaultIntelCases(config.modelRoot);
  const discoveredCases = args.discoverModels ? await discoverIntelCases(config.modelRoot, creds) : [];
  const mergedCases = new Map();
  for (const entry of [...discoveredCases, ...knownCases]) {
    mergedCases.set(entry.id, entry);
  }
  const cases = [];
  for (const entry of mergedCases.values()) {
    if (args.models && !args.models.has(entry.id)) continue;
    if (!(await rdnaPathExists(entry.model_path, creds))) {
      console.log(`[intel] skipping ${entry.id}: missing GGUF at ${entry.model_path}`);
      continue;
    }
    cases.push(entry);
  }
  const models = [];

  for (const entry of cases) {
    console.log(`[intel] ${entry.id}`);
    const scenarioDefs = filterScenarioDefs(defaultScenarioDefsForModel(entry.id, entry.prompt_mode, entry.prompt), args.scenarios);
    const phases = buildMeasurementPhases(entry.id, entry.prompt_mode, entry.prompt, args.phase, args.scenarios);
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
              command: intelZincCommand(caseDef, creds),
              parser: parseZincCliOutput,
              cwd: ROOT,
              timeoutMs: args.timeoutMs,
            });
            zinc = createStats("ZINC", zincRows);
          } catch (error) {
            zinc = {
              name: "ZINC",
              unavailable_reason: benchmarkFailureReason("ZINC run failed", error),
            };
          }
          zincByScenario.set(scenarioDef.id, zinc);
          continue;
        }

        if (!triedLaunchingServer && intelLlamaServer) {
          triedLaunchingServer = true;
          try {
            launchedServer = await launchRdnaLlamaServer(entry, creds, intelLlamaServer, args.timeoutMs, "intel", false);
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
              unavailable_reason: benchmarkFailureReason("Intel baseline failed", error),
            };
          }
        } else if (entry.baseline_enabled !== false && intelLlamaCli) {
          try {
            const baselineRows = await runSeries({
              label: `llama.cpp ${entry.id} ${scenarioDef.id}`,
              warmupRuns: args.warmupRuns,
              runs: args.runs,
              command: remoteLlamaCommand(caseDef, creds, intelLlamaCli),
              parser: parseLlamaCliOutput,
              cwd: ROOT,
              timeoutMs: args.timeoutMs,
            });
            baseline = createStats("llama.cpp", baselineRows);
          } catch (error) {
            baseline = {
              name: "llama.cpp",
              unavailable_reason: benchmarkFailureReason("Intel baseline failed", error),
            };
          }
        } else if (entry.baseline_enabled !== false && serverLaunchFailure) {
          baseline = {
            name: "llama.cpp",
            unavailable_reason: benchmarkFailureReason("Intel baseline failed", serverLaunchFailure),
          };
        } else {
          baseline = {
            name: "llama.cpp",
            unavailable_reason: intelLlamaServer || intelLlamaCli
              ? "No Intel llama.cpp baseline is defined for this case yet."
              : "Could not locate a remote llama.cpp baseline binary on the Intel node.",
          };
        }

        baselineByScenario.set(scenarioDef.id, baseline);
      }
    } finally {
      if (launchedServer) {
        await stopRdnaLlamaServer(creds, launchedServer.port);
      }
    }

    const includeZinc = phaseEnabled(args.phase, "zinc");
    const includeBaseline = phaseEnabled(args.phase, "baseline");
    const scenarios = scenarioDefs.map((scenarioDef) => scenarioResultPayload(
      entry,
      scenarioDef,
      zincByScenario.get(scenarioDef.id) ?? (includeZinc ? {
        name: "ZINC",
        unavailable_reason: "No ZINC benchmark result was recorded for this scenario.",
      } : undefined),
      baselineByScenario.get(scenarioDef.id) ?? (includeBaseline ? {
        name: "llama.cpp",
        unavailable_reason: "No llama.cpp baseline result was recorded for this scenario.",
      } : undefined),
    ));
    const summary = primaryScenarioSummary(entry, scenarios);
    models.push(summary);
    logModelSummary("intel", summary);
    await writePartialSnapshot(args, "intel", "Intel Arc", models);
  }

  return {
    id: "intel",
    label: "Intel Arc",
    description: "Intel Arc benchmark node results collected over SSH and compared against llama.cpp on the same hardware.",
    generated_at: new Date().toISOString(),
    source: "tools/performance_suite.mjs",
    machine: {
      label: "Intel Arc bench node",
      machine_model: "Remote Linux host",
      chip: "Intel Arc Pro B70 / BMG G31",
      memory: null,
      gpu: "Intel(R) Graphics (BMG G31)",
      os: "Ubuntu Linux",
    },
    provenance: {
      zinc: zincProvenance,
      llama_cpp: llamaCppProvenance,
    },
    methodology: {
      runner: "ZINC CLI vs llama.cpp on the same Intel Arc node",
      benchmark_style: "Four-scenario matrix with same-file baselines",
      notes: [
        "Hardware: one Intel Arc Vulkan benchmark node. ZINC and llama.cpp run on the same Ubuntu host and use the same GGUF file for each model.",
        "ZINC path: optionally sync the current source tree to the Intel node, build with zig build -Doptimize=ReleaseFast, then measure generation through the ZINC CLI.",
        "Baseline path: launch llama.cpp against the same model file, preferring one reusable llama-server per model across the full scenario matrix and falling back to llama-cli when the server path is unavailable or fails to start.",
        "Scenarios: Quick Chat, Coding Review, Incident Context, and Long Coding Draft. The prompts are real-world chat, code-review, support-context, and coding-plan workloads instead of synthetic factual completions.",
        "Statistics: one warmup pass is discarded, then three measured runs are collected. Published prefill, decode, end-to-end throughput, and latency values are medians.",
        "Execution order: the harness records the full ZINC scenario matrix before starting the llama.cpp baseline phase, so only one inference engine owns the GPU during measurement.",
        "Run hygiene: published Intel runs assume a clean node with no competing ZINC, llama.cpp, or other GPU workloads.",
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

function partialSnapshotPath(outputPath) {
  return `${outputPath}.partial.json`;
}

async function writePartialSnapshot(args, targetId, label, models) {
  if (!args?.output) return;
  const snapshot = {
    schema_version: 1,
    generated_at: new Date().toISOString(),
    in_progress: { target: targetId, label, completed_models: models.length },
    models,
  };
  try {
    await writeJson(partialSnapshotPath(args.output), snapshot);
  } catch (error) {
    process.stderr.write(`  (partial snapshot write failed: ${error.message})\n`);
  }
}

function fmtTps(value) {
  return value == null ? "  n/a " : value.toFixed(1).padStart(6);
}

function logModelSummary(targetId, modelSummary) {
  const primary = modelSummary.scenarios?.find((s) => s.id === "core") ?? modelSummary.scenarios?.[0];
  const cmp = primary?.comparison ?? modelSummary.comparison ?? {};
  const zincPrefill = cmp.zinc_prompt_tps ?? primary?.zinc?.prefill_tps?.median ?? primary?.zinc?.prefill_tps?.avg ?? null;
  const zincDecode = cmp.zinc_decode_tps ?? primary?.zinc?.decode_tps?.median ?? primary?.zinc?.decode_tps?.avg ?? null;
  const blPrefill = cmp.baseline_prompt_tps ?? primary?.baseline?.prefill_tps?.median ?? primary?.baseline?.prefill_tps?.avg ?? null;
  const blDecode = cmp.baseline_decode_tps ?? primary?.baseline?.decode_tps?.median ?? primary?.baseline?.decode_tps?.avg ?? null;
  const overallPct = cmp.overall_pct_of_baseline ?? cmp.pct_of_baseline ?? null;
  const overall = overallPct == null ? "  n/a" : `${overallPct.toFixed(1)}%`;
  process.stdout.write(
    `  [${targetId}] DONE ${modelSummary.id}: ` +
    `zinc prefill ${fmtTps(zincPrefill)} / decode ${fmtTps(zincDecode)} | ` +
    `llama prefill ${fmtTps(blPrefill)} / decode ${fmtTps(blDecode)} | ` +
    `overall ${overall}\n`,
  );
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help) {
    process.stdout.write(usage());
    return;
  }

  const incoming = [];
  if (args.target === "metal" || args.target === "both" || args.target === "all") {
    incoming.push(await runMetalTarget(args));
  }
  if (args.target === "rdna" || args.target === "both" || args.target === "all") {
    incoming.push(await runRdnaTarget(args));
  }
  if (args.target === "intel" || args.target === "all") {
    incoming.push(await runIntelTarget(args));
  }
  if (args.target === "cuda") {
    incoming.push(await runCudaTarget(args));
  }

  const outputArtifact = buildArtifact(incoming);
  await writeJson(args.output, outputArtifact);
  if (args.writeSiteData) {
    const existing = await loadExistingArtifact(args.siteData);
    const merged = mergeArtifacts(existing, incoming, {
      preserveMissingPhases: args.phase !== "all",
    });
    // Guard: never publish host/network info into the public site data. Scan for
    // ACTUAL host indicators (hostnames, tailscale IPs, tailnet domains) — not a
    // bare :port, which false-positives on numeric metric values (e.g. a decode_ms
    // of 2222.06 serializes as ":2222" in compact JSON).
    const leak = JSON.stringify(merged).match(/puffalo|tailbe7bef|agent-zinc|127\.99\.\d+\.\d+|\b100\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|\.ts\.net/g);
    if (leak) {
      throw new Error(`Refusing to write site data: host/network info present (${[...new Set(leak)].join(", ")}). Sanitize the artifact before publishing.`);
    }
    await writeJson(args.siteData, merged);
  }

  console.log(`\nWrote benchmark artifact: ${args.output}`);
  if (args.writeSiteData) {
    console.log(`Updated site data: ${args.siteData}`);
  }
  for (const target of incoming) {
    console.log(`${target.label}: ${target.models.length} model(s), fastest ${(target.summary.fastest_decode_tps ?? 0).toFixed(2)} tok/s`);
  }

  await fs.rm(partialSnapshotPath(args.output), { force: true }).catch(() => {});
}

if (process.argv[1] === TOOL_PATH) {
  main().catch((error) => {
    console.error(error.stack || error.message);
    process.exit(1);
  });
}
