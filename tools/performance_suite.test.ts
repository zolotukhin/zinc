import { expect, test } from "bun:test";
import { readFileSync } from "node:fs";

import {
  buildArtifact,
  buildComparison,
  buildMeasurementPhases,
  buildZincOpenAiPayload,
  benchmarkFailureReason,
  canonicalModelIdFromPath,
  collectRemoteZincTuningEnv,
  compareModelsByName,
  detectRdnaServerStartupFailure,
  DEFAULT_LOCAL_MODEL_ROOT,
  defaultIntelCases,
  defaultMetalCases,
  defaultMaxTokensForModelId,
  defaultPromptForModelId,
  defaultRdnaCases,
  defaultScenarioDefsForModel,
  guessFamily,
  llamaDeviceArgs,
  localZincCommand,
  mergeArtifacts,
  outputQualityStatus,
  parseArgs,
  parseDotEnv,
  parseLlamaCliOutput,
  parseLlamaCppVersionOutput,
  parseOpenAiCompletionOutput,
  parseZincCliOutput,
  parseZincServerOutput,
  parseZincVersionOutput,
  prefersChatPrompt,
  rdnaEnvValue,
  rdnaNodeEnvKey,
  intelZincCommand,
  rdnaZincCommand,
  resolveLocalLlamaServer,
  rdnaDpmHighScript,
  summarizeValues,
  validateZincBackend,
  zincServerTimingWaitSeconds,
} from "./performance_suite.mjs";

test("parseArgs reads suite options", () => {
  const args = parseArgs([
    "--target",
    "metal",
    "--runs",
    "5",
    "--warmup",
    "2",
    "--models",
    "gemma4-26b-a4b-q4k-m,qwen35-9b-q4k-m",
    "--llama-cli",
    "/tmp/llama-cli",
    "--llama-server",
    "/tmp/llama-server",
    "--phase",
    "zinc",
    "--scenarios",
    "core,decode-extended",
    "--no-site-write",
  ]);

  expect(args.target).toBe("metal");
  expect(args.runs).toBe(5);
  expect(args.warmupRuns).toBe(2);
  expect(args.llamaCli).toBe("/tmp/llama-cli");
  expect(args.llamaServer).toBe("/tmp/llama-server");
  expect(args.phase).toBe("zinc");
  expect(args.writeSiteData).toBe(false);
  expect(args.scenarios && [...args.scenarios]).toEqual(["core", "decode-extended"]);
  expect(args.models && [...args.models]).toEqual(["gemma4-26b-a4b-q4k-m", "qwen35-9b-q4k-m"]);
});

test("parseArgs reads Intel suite options", () => {
  const args = parseArgs([
    "--target",
    "intel",
    "--intel-sync",
    "--intel-build",
    "--intel-start-llama",
    "--intel-model-root",
    "/home/tempuser/.cache/zinc/models/models",
    "--intel-workdir",
    "/home/tempuser/zinc-intel-loop",
    "--intel-xdg-cache-home",
    "/home/tempuser/.cache",
    "--intel-remote-libc-conf",
    "/workspace/zinc/.build-support/libc.conf",
  ]);

  expect(args.target).toBe("intel");
  expect(args.intelSync).toBe(true);
  expect(args.intelBuild).toBe(true);
  expect(args.intelStartLlama).toBe(true);
  expect(args.intelModelRoot).toBe("/home/tempuser/.cache/zinc/models/models");
  expect(args.intelWorkdir).toBe("/home/tempuser/zinc-intel-loop");
  expect(args.intelXdgCacheHome).toBe("/home/tempuser/.cache");
  expect(args.intelRemoteLibcConf).toBe("/workspace/zinc/.build-support/libc.conf");
});

test("remote tuning env forwards tuning toggles", () => {
  const env = collectRemoteZincTuningEnv({
    ZINC_Q8_1_LM_HEAD: "1",
    ZINC_Q8_1_SSM_QKV_Z: "1",
    ZINC_MOE_Q5K_Q8_1_DOWN_ACC: "1",
    ZINC_INTEL_A3B_PRODUCTION: "0",
    ZINC_QWEN35_9B_BM64_DOWN: "0",
    ZINC_QWEN35_9B_K12288_BK2: "0",
    ZINC_QWEN36_27B_DENSE_PREFILL_LAYERS: "4",
    ZINC_QWEN36_27B_DENSE_PREFILL_SEGMENT: "0",
    ZINC_QWEN36_27B_PREFIX_TAIL_PIPELINE: "0",
    ZINC_QWEN36_27B_SSM_BATCHED_DELTA: "0",
    ZINC_QWEN36_27B_SSM_PREFILL_PROJ: "both",
    ZINC_QWEN36_27B_FULL_ATTN_BATCHED: "0",
  });
  expect(env.ZINC_Q8_1_LM_HEAD).toBe("1");
  expect(env.ZINC_Q8_1_SSM_QKV_Z).toBe("1");
  expect(env.ZINC_MOE_Q5K_Q8_1_DOWN_ACC).toBe("1");
  expect(env.ZINC_INTEL_A3B_PRODUCTION).toBe("0");
  expect(env.ZINC_QWEN35_9B_BM64_DOWN).toBe("0");
  expect(env.ZINC_QWEN35_9B_K12288_BK2).toBe("0");
  expect(env.ZINC_QWEN36_27B_DENSE_PREFILL_LAYERS).toBe("4");
  expect(env.ZINC_QWEN36_27B_DENSE_PREFILL_SEGMENT).toBe("0");
  expect(env.ZINC_QWEN36_27B_PREFIX_TAIL_PIPELINE).toBe("0");
  expect(env.ZINC_QWEN36_27B_SSM_BATCHED_DELTA).toBe("0");
  expect(env.ZINC_QWEN36_27B_SSM_PREFILL_PROJ).toBe("both");
  expect(env.ZINC_QWEN36_27B_FULL_ATTN_BATCHED).toBe("0");
});

test("parseArgs reads RDNA backend and device options", () => {
  const args = parseArgs([
    "--target",
    "rdna",
    "--rdna-node",
    "rdna1",
    "--rdna-backend",
    "vulkan",
    "--rdna-vk-device",
    "1",
    "--require-rdna-device-substring",
    "GFX1201",
  ]);

  expect(args.target).toBe("rdna");
  expect(args.rdnaNode).toBe("rdna1");
  expect(args.rdnaBackend).toBe("vulkan");
  expect(args.rdnaVkDevice).toBe(1);
  expect(args.requireRdnaDeviceSubstring).toBe("GFX1201");
  expect(args.rdnaWorkdir).toBe("/root/zinc-bench");
});

test("RDNA DPM high script targets AMD memory-clock controls safely", () => {
  const script = rdnaDpmHighScript();
  expect(script).toContain("/sys/class/drm/card*/device");
  expect(script).toContain("pp_dpm_mclk");
  expect(script).toContain("power_dpm_force_performance_level");
  expect(script).toContain("echo high");
  expect(script).toContain("2>/dev/null || true");
  expect(script).not.toContain("do;");
  expect(script).not.toContain("then;");
});

test("parseArgs rejects invalid RDNA backend", () => {
  expect(() => parseArgs(["--target", "rdna", "--rdna-backend", "metal"])).toThrow(
    "Invalid --rdna-backend 'metal'",
  );
});

test("parseZincVersionOutput extracts the compiled backend", () => {
  const parsed = parseZincVersionOutput(`
zinc 67cc418bf8f8
commit: 67cc418bf8f8ce30ec82a6d9a599c3e90186904f
target: x86_64-linux-gnu
optimize: ReleaseFast
backends: vulkan
`);

  expect(parsed.version).toBe("67cc418bf8f8");
  expect(parsed.commit).toBe("67cc418bf8f8ce30ec82a6d9a599c3e90186904f");
  expect(parsed.target).toBe("x86_64-linux-gnu");
  expect(parsed.optimize).toBe("ReleaseFast");
  expect(parsed.backend).toBe("vulkan");
});

test("validateZincBackend rejects stale or overwritten RDNA binaries", () => {
  expect(() => validateZincBackend("info(zinc_rt): M0 runtime initialized", "vulkan")).toThrow(
    "RDNA ZINC binary backend mismatch: expected vulkan, observed unknown",
  );
  expect(() => validateZincBackend("zinc dev\nbackends: zinc_rt\n", "vulkan")).toThrow(
    "expected vulkan, observed zinc_rt",
  );
  expect(validateZincBackend("zinc dev\nbackends: vulkan\n", "vulkan").backend).toBe("vulkan");
});

test("parseArgs enables discovery mode", () => {
  const args = parseArgs(["--target", "metal", "--discover-models"]);
  expect(args.discoverModels).toBe(true);
});

test("parseArgs enables managed Metal pulls", () => {
  const args = parseArgs(["--target", "metal", "--metal-pull-missing"]);
  expect(args.metalPullMissing).toBe(true);
});

test("resolveLocalLlamaServer prefers explicit path, then PATH, then docker fallback", () => {
  expect(resolveLocalLlamaServer({ llamaServer: "/tmp/explicit" }, "/tmp/path", "/tmp/docker")).toBe("/tmp/explicit");
  expect(resolveLocalLlamaServer({ llamaServer: null }, "/tmp/path", "/tmp/docker")).toBe("/tmp/path");
  expect(resolveLocalLlamaServer({ llamaServer: null }, null, "/tmp/docker")).toBe("/tmp/docker");
});

test("Gemma uses the chat prompt path in the performance suite", () => {
  expect(prefersChatPrompt("gemma4-26b-a4b-q4k-m")).toBe(true);
  expect(defaultPromptForModelId("gemma4-26b-a4b-q4k-m")).toContain("benchmark screenshots");
  expect(defaultMaxTokensForModelId("gemma4-26b-a4b-q4k-m")).toBe(96);
  expect(prefersChatPrompt("qwen35-9b-q4k-m")).toBe(false);
  expect(defaultPromptForModelId("qwen35-9b-q4k-m")).toContain("Developer question");
  expect(defaultMaxTokensForModelId("qwen35-9b-q4k-m")).toBe(96);
});

test("default Metal cases use managed cache ids and include Qwen 3.6", () => {
  const cases = defaultMetalCases("/tmp/models");

  const qwen36 = cases.find((entry) => entry.id === "qwen36-35b-a3b-q4k-xl");
  expect(qwen36?.model_id).toBe("qwen36-35b-a3b-q4k-xl");
  expect(qwen36?.model_path).toBe("/tmp/models/qwen36-35b-a3b-q4k-xl/model.gguf");

  const qwen36Dense = cases.find((entry) => entry.id === "qwen36-27b-q4k-m");
  expect(qwen36Dense?.model_id).toBe("qwen36-27b-q4k-m");
  expect(qwen36Dense?.model_path).toBe("/tmp/models/qwen36-27b-q4k-m/model.gguf");
});

test("default RDNA cases include Gemma and current Qwen rows", () => {
  const cases = defaultRdnaCases("/root/models");
  const gemma26 = cases.find((entry) => entry.id === "gemma4-26b-a4b-q4k-m");
  const gemma31 = cases.find((entry) => entry.id === "gemma4-31b-q4k-m");
  const qwen36Dense = cases.find((entry) => entry.id === "qwen36-27b-q4k-m");
  const qwen35 = cases.find((entry) => entry.id === "qwen35-9b-q4k-m");

  expect(gemma26?.model_path).toBe("/root/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf");
  expect(gemma26?.prompt_mode).toBe("chat");
  expect(gemma26?.prompt).toContain("benchmark screenshots");
  expect(gemma26?.max_tokens).toBe(96);

  expect(gemma31?.model_path).toBe("/root/models/gemma-4-31B-it-Q4_K_M.gguf");
  expect(gemma31?.prompt_mode).toBe("chat");
  expect(gemma31?.prompt).toContain("benchmark screenshots");
  expect(gemma31?.max_tokens).toBe(96);

  expect(qwen36Dense?.model_path).toBe("/root/models/Qwen3.6-27B-Q4_K_M.gguf");
  expect(qwen36Dense?.prompt_mode).toBe("raw");
  expect(qwen36Dense?.prompt).toContain("Developer question");
  expect(qwen36Dense?.max_tokens).toBe(96);

  expect(qwen35?.model_path).toBe("/root/models/Qwen3.5-9B-Q4_K_M.gguf");
  expect(qwen35?.prompt_mode).toBe("raw");
  expect(qwen35?.prompt).toContain("Developer question");
  expect(qwen35?.max_tokens).toBe(96);
});

test("default Intel cases use the remote managed cache layout", () => {
  const cases = defaultIntelCases("/remote/cache");
  expect(cases.map((entry) => entry.id)).toEqual([
    "gemma4-26b-a4b-q4k-m",
    "gemma4-31b-q4k-m",
    "qwen35-9b-q4k-m",
    "qwen36-35b-a3b-q4k-xl",
    "qwen36-27b-q4k-m",
  ]);
  const qwen = cases.find((entry) => entry.id === "qwen35-9b-q4k-m");
  const gemma = cases.find((entry) => entry.id === "gemma4-26b-a4b-q4k-m");

  expect(qwen?.model_path).toBe("/remote/cache/qwen35-9b-q4k-m/model.gguf");
  expect(qwen?.prompt_mode).toBe("raw");
  expect(qwen?.context_tokens).toBe(512);
  expect(gemma?.model_path).toBe("/remote/cache/gemma4-26b-a4b-q4k-m/model.gguf");
  expect(gemma?.prompt_mode).toBe("chat");
  expect(gemma?.notes).toEqual(["Intel Arc Vulkan comparison against llama.cpp on the same host"]);
});

test("llama device args support Intel Vulkan0 and no-device modes", () => {
  expect(llamaDeviceArgs("Vulkan0")).toEqual(["--device", "Vulkan0"]);
  expect(llamaDeviceArgs("none")).toEqual([]);
  expect(llamaDeviceArgs(null)).toEqual([]);
});

test("performance suite canonicalizes and labels Qwen 3.6 GGUFs", () => {
  expect(canonicalModelIdFromPath("/tmp/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf")).toBe("qwen36-35b-a3b-q4k-xl");
  expect(canonicalModelIdFromPath("/tmp/Qwen3.6-27B-Q4_K_M.gguf")).toBe("qwen36-27b-q4k-m");
  expect(canonicalModelIdFromPath("/tmp/Qwen_Qwen3.6-27B-Q4_K_M.gguf")).toBe("qwen36-27b-q4k-m");
  expect(canonicalModelIdFromPath("/tmp/models/qwen36-35b-a3b-q4k-xl/model.gguf")).toBe("qwen36-35b-a3b-q4k-xl");
  expect(guessFamily("qwen36-35b-a3b-q4k-xl")).toBe("Qwen 3.6");
  expect(guessFamily("qwen36-27b-q4k-m")).toBe("Qwen 3.6");
});

test("performance suite canonicalizes RDNA Gemma GGUF filenames to published ids", () => {
  expect(canonicalModelIdFromPath("/root/models/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf")).toBe("gemma4-26b-a4b-q4k-m");
  expect(canonicalModelIdFromPath("/root/models/gemma-4-31B-it-Q4_K_M.gguf")).toBe("gemma4-31b-q4k-m");
  expect(guessFamily("gemma4-26b-a4b-q4k-m")).toBe("Gemma 4");
});

test("local ZINC command prefers managed model ids when using the default cache", () => {
  const cmd = localZincCommand({
    model_id: "qwen35-9b-q4k-m",
    model_path: `${DEFAULT_LOCAL_MODEL_ROOT}/qwen35-9b-q4k-m/model.gguf`,
    prompt_mode: "raw",
    prompt: "The capital of France is",
    max_tokens: 8,
    context_tokens: 512,
  });
  expect(cmd).toContain("--model-id qwen35-9b-q4k-m");
  expect(cmd).toContain("--context 512");
  expect(cmd).not.toContain(" -m ");
});

test("RDNA ZINC command preserves chat prompt mode", () => {
  const cmd = rdnaZincCommand({
    model_path: "/root/models/gpt.gguf",
    prompt_mode: "chat",
    prompt: "What is the capital of France?",
    max_tokens: 48,
  }, {
    host: "bench.local",
    user: "root",
    port: "2222",
    workdir: "/root/zinc",
  });

  expect(cmd).toContain("./zig-out/bin/zinc");
  expect(cmd).toContain("--chat");
  expect(cmd).toContain("--prompt");
  expect(cmd).toContain("pp_dpm_mclk");
  expect(cmd).toContain("power_dpm_force_performance_level");
  expect(cmd).toContain("done; true && cd");
  expect(cmd).not.toContain("do;");
  expect(cmd).not.toContain("then;");
});

test("Intel ZINC command does not inject RDNA-specific environment", () => {
  const cmd = intelZincCommand({
    model_path: "/home/tempuser/.cache/zinc/models/models/qwen35-9b-q4k-m/model.gguf",
    prompt_mode: "raw",
    prompt: "The capital of France is",
    max_tokens: 8,
    context_tokens: 512,
  }, {
    host: "intel.local",
    user: "tempuser",
    port: "8888",
    workdir: "/home/tempuser/zinc",
    env: {},
  });

  expect(cmd).toContain("tempuser@intel.local");
  expect(cmd).toContain("./zig-out/bin/zinc");
  expect(cmd).toContain("--context 512");
  expect(cmd).not.toContain("RADV_PERFTEST");
  expect(cmd).not.toContain("--chat");
});

test("Intel ZINC command can pin a temporary SSH key", () => {
  const cmd = intelZincCommand({
    model_path: "/home/tempuser/.cache/zinc/models/models/qwen35-9b-q4k-m/model.gguf",
    prompt_mode: "raw",
    prompt: "The capital of France is",
    max_tokens: 8,
    context_tokens: 512,
  }, {
    host: "intel.local",
    user: "tempuser",
    port: "8888",
    sshKey: "/tmp/zinc_key",
    workdir: "/home/tempuser/zinc",
    env: {},
  });

  expect(cmd).toContain("ssh -i '/tmp/zinc_key' -o IdentitiesOnly=yes -p 8888 tempuser@intel.local");
});

test("RDNA startup failure detection spots unsupported model architecture logs", () => {
  const failure = detectRdnaServerStartupFailure(`
llama_model_load: error loading model: error loading model architecture: unknown model architecture: 'gemma4'
srv load_model: failed to load model
main: exiting due to model loading error
`);

  expect(failure).toBe("unknown model architecture: 'gemma4'");
  expect(detectRdnaServerStartupFailure("server ready")).toBeNull();
});

test("benchmark failure reasons do not publish shell commands", () => {
  const error = new Error("Command failed (1): remote benchmark command with private args\nprivate details");
  expect(benchmarkFailureReason("ZINC run failed", error)).toBe("ZINC run failed: command exited unsuccessfully (1).");
  const diagnostic = new Error("Command failed (1): remote benchmark command with private args\nerr(zinc): Failed to init inference engine: QueueSubmitFailed");
  expect(benchmarkFailureReason("ZINC run failed", diagnostic)).toBe("ZINC run failed: err(zinc): Failed to init inference engine: QueueSubmitFailed");
  expect(benchmarkFailureReason("Intel baseline failed", new Error("Remote server failed to start"))).toBe("Intel baseline failed: Remote server failed to start");
});

test("performance suite rsync excludes local secrets and run state", () => {
  const source = readFileSync(new URL("./performance_suite.mjs", import.meta.url), "utf8");
  expect(source).toContain("--exclude '.git'");
  expect(source).toContain("--exclude '.env'");
  expect(source).toContain("--exclude '.env.*'");
  expect(source).toContain("--exclude '.zinc_rt_autopilot'");
});

test("benchmark suite uses a multi-scenario matrix instead of a single prompt", () => {
  const qwen = defaultScenarioDefsForModel("qwen35-9b-q4k-m", "raw", defaultPromptForModelId("qwen35-9b-q4k-m"));
  expect(qwen.map((scenario) => scenario.id)).toEqual(["core", "context-medium", "context-long", "decode-extended"]);
  expect(qwen.map((scenario) => scenario.label)).toEqual(["Quick Chat", "Coding Review", "Incident Context", "Long Coding Draft"]);
  expect(qwen[1]?.prompt).not.toBe(qwen[0]?.prompt);
  expect(qwen[1]?.prompt).toContain("src/cache.ts");
  expect(qwen[1]?.max_tokens).toBe(160);
  expect(qwen[2]?.prompt).toContain("Incident notes");
  expect(qwen[2]?.max_tokens).toBe(128);
  expect(qwen[3]?.prompt).toContain("stable benchmark preset");
  expect(qwen[3]?.max_tokens).toBe(256);
});

test("benchmark suite measures all ZINC scenarios before starting baselines", () => {
  const phases = buildMeasurementPhases("qwen36-35b-a3b-q4k-xl", "raw", "The capital of France is");
  expect(phases.map((phase) => phase.phase)).toEqual([
    "zinc",
    "zinc",
    "zinc",
    "zinc",
    "baseline",
    "baseline",
    "baseline",
    "baseline",
  ]);
  expect(phases.slice(0, 4).map((phase) => phase.scenarioDef.id)).toEqual(["core", "context-medium", "context-long", "decode-extended"]);
  expect(phases.slice(4).map((phase) => phase.scenarioDef.id)).toEqual(["core", "context-medium", "context-long", "decode-extended"]);
});

test("benchmark suite can split ZINC and baseline phases for clean reboot runs", () => {
  const zincOnly = buildMeasurementPhases("qwen36-35b-a3b-q4k-xl", "raw", "The capital of France is", "zinc");
  expect(zincOnly.map((phase) => phase.phase)).toEqual(["zinc", "zinc", "zinc", "zinc"]);

  const baselineOnly = buildMeasurementPhases("qwen36-35b-a3b-q4k-xl", "raw", "The capital of France is", "baseline");
  expect(baselineOnly.map((phase) => phase.phase)).toEqual(["baseline", "baseline", "baseline", "baseline"]);
});

test("benchmark suite can filter scenarios for targeted optimization runs", () => {
  const phases = buildMeasurementPhases(
    "qwen35-9b-q4k-m",
    "raw",
    "The capital of France is",
    "all",
    new Set(["core", "decode-extended"]),
  );
  expect(phases.map((phase) => `${phase.phase}:${phase.scenarioDef.id}`)).toEqual([
    "zinc:core",
    "zinc:decode-extended",
    "baseline:core",
    "baseline:decode-extended",
  ]);
  expect(() => buildMeasurementPhases(
    "qwen35-9b-q4k-m",
    "raw",
    "The capital of France is",
    "all",
    new Set(["unknown"]),
  )).toThrow(/Unknown scenario/);
});

test("parseDotEnv handles export lines and quotes", () => {
  const env = parseDotEnv(`
    export ZINC_HOST=bench.local
    ZINC_USER="root"
    ZINC_PORT='2222'
  `);
  expect(env.ZINC_HOST).toBe("bench.local");
  expect(env.ZINC_USER).toBe("root");
  expect(env.ZINC_PORT).toBe("2222");
});

test("RDNA node selection prefers node-specific environment keys", () => {
  const env = parseDotEnv(`
    ZINC_RDNA_NODE=rdna2
    ZINC_RDNA_HOST=rdna-default.local
    ZINC_RDNA_USER=root
    ZINC_RDNA_PORT=22
    ZINC_RDNA2_HOST=rdna-two.local
    ZINC_RDNA2_USER=bench
    ZINC_RDNA2_PORT=2222
  `);

  expect(rdnaNodeEnvKey("rdna-2", "HOST")).toBe("ZINC_RDNA_2_HOST");
  expect(rdnaEnvValue(env, { envMap: {} }, "HOST", "ZINC_RDNA_HOST", "ZINC_HOST")).toBe("rdna-two.local");
  expect(rdnaEnvValue(env, { envMap: {}, rdnaNode: "rdna1" }, "HOST", "ZINC_RDNA_HOST", "ZINC_HOST")).toBe("rdna-default.local");
  expect(rdnaEnvValue(env, { envMap: {} }, "USER", "ZINC_RDNA_USER", "ZINC_USER")).toBe("bench");
  expect(rdnaEnvValue(env, { envMap: {} }, "PORT", "ZINC_RDNA_PORT", "ZINC_PORT")).toBe("2222");
});

test("parseZincCliOutput extracts prompt, prefill, decode, and output preview", () => {
  const parsed = parseZincCliOutput(`
info(zinc): Prompt tokens (25): { 1, 2, 3 }
info(forward): Prefill: 25 tokens in 40716.9 ms (0.6 tok/s)
info(forward): Generated 8 tokens in 1269.5 ms — 0.79 tok/s (1269.5 ms/tok)
info(zinc): Output (1 tokens): Paris
`);

  expect(parsed.promptTokens).toBe(25);
  expect(parsed.prefillMs).toBe(40716.9);
  expect(parsed.prefillTps).toBe(0.6);
  expect(parsed.decodeTps).toBe(0.79);
  expect(parsed.msPerToken).toBe(1269.5);
  expect(parsed.outputPreview).toBe("Paris");
});

test("parseZincCliOutput accepts Vulkan-style prefill complete lines", () => {
  const parsed = parseZincCliOutput(`
info(zinc): Prompt tokens (10): { 1, 2, 3 }
info(forward): Prefill complete: 10 tokens in 1.3 ms (7459.27 tok/s)
info(forward): Generated 8 tokens in 61.4 ms — 130.32 tok/s (7.7 ms/tok)
info(zinc): Output text: Paris.
`);

  expect(parsed.promptTokens).toBe(10);
  expect(parsed.prefillMs).toBe(1.3);
  expect(parsed.prefillTps).toBeCloseTo(7459.27, 2);
  expect(parsed.decodeTps).toBe(130.32);
  expect(parsed.outputPreview).toContain("Paris.");
});

test("parseZincCliOutput tolerates missing prefill lines and parses output text", () => {
  const parsed = parseZincCliOutput(`
info(zinc): Prompt tokens (5): { 1, 2, 3 }
info(forward): Generated 8 tokens in 61.4 ms — 130.32 tok/s (7.7 ms/tok)
info(zinc): Output text:  Paris.
A. True
info(zinc): Output tokens (8): first20={ 1, 2, 3 }
`);

  expect(parsed.promptTokens).toBe(5);
  expect(parsed.prefillTokens).toBe(5);
  expect(parsed.prefillMs).toBeNull();
  expect(parsed.prefillTps).toBeNull();
  expect(parsed.decodeTps).toBe(130.32);
  expect(parsed.outputPreview).toContain("Paris.");
});

test("parseZincServerOutput combines OpenAI response usage with server log timings", () => {
  const parsed = parseZincServerOutput(`{"id":"cmpl-1","object":"text_completion","choices":[{"index":0,"text":" Command shape","finish_reason":"length"}],"usage":{"prompt_tokens":64,"completion_tokens":32,"total_tokens":96}}
__ZINC_TIMING__
info(forward): Prefill: 64 tokens in 183.1 ms (349.62 tok/s)
info(forward): Generated 32 tokens in 977.9 ms — 32.72 tok/s (30.6 ms/tok)
`);

  expect(parsed.promptTokens).toBe(64);
  expect(parsed.prefillTokens).toBe(64);
  expect(parsed.prefillMs).toBe(183.1);
  expect(parsed.prefillTps).toBe(349.62);
  expect(parsed.generatedTokens).toBe(32);
  expect(parsed.decodeTps).toBe(32.72);
  expect(parsed.outputPreview).toBe("Command shape");
});

test("RDNA ZINC server payload keeps the preloaded GGUF active", () => {
  const raw = buildZincOpenAiPayload({
    prompt_mode: "raw",
    prompt: "The capital of France is",
    max_tokens: 32,
  });
  expect(raw).toEqual({
    prompt: "The capital of France is",
    max_tokens: 32,
    temperature: 0,
    stream: false,
  });
  expect(raw).not.toHaveProperty("model");

  const chat = buildZincOpenAiPayload({
    prompt_mode: "chat",
    prompt: "Tell me about C++",
    max_tokens: 48,
  });
  expect(chat).toEqual({
    messages: [{ role: "user", content: "Tell me about C++" }],
    max_tokens: 48,
    temperature: 0,
    stream: false,
  });
  expect(chat).not.toHaveProperty("model");
});

test("RDNA ZINC timing wait is bounded after the API response", () => {
  expect(zincServerTimingWaitSeconds(12_000)).toBe(10);
  expect(zincServerTimingWaitSeconds(65_000)).toBe(60);
  expect(zincServerTimingWaitSeconds(1_800_000)).toBe(60);
});

test("parseLlamaCliOutput extracts prompt and decode timings", () => {
  const parsed = parseLlamaCliOutput(`
llama_print_timings: prompt eval time =   152.58 ms /    16 tokens (    9.54 ms per token,   104.87 tokens per second)
llama_print_timings:        eval time =   474.72 ms /    15 runs   (   31.65 ms per token,    31.60 tokens per second)
`);

  expect(parsed.promptTokens).toBe(16);
  expect(parsed.prefillTps).toBe(104.87);
  expect(parsed.generatedTokens).toBe(15);
  expect(parsed.decodeTps).toBe(31.6);
  expect(parsed.msPerToken).toBeCloseTo(31.648, 3);
});

test("parseLlamaCppVersionOutput extracts version and commit from llama.cpp --version output", () => {
  const parsed = parseLlamaCppVersionOutput(`
load_backend: loaded BLAS backend from /opt/homebrew/Cellar/ggml/0.9.11/libexec/libggml-blas.so
version: 8610 (2b86e5cae)
built with AppleClang 17.0.0.17000604 for Darwin arm64
`);

  expect(parsed?.version).toBe("8610");
  expect(parsed?.commit).toBe("2b86e5cae");
});

test("parseOpenAiCompletionOutput extracts throughput from server JSON", () => {
  const parsed = parseOpenAiCompletionOutput(JSON.stringify({
    usage: {
      prompt_tokens: 24,
      completion_tokens: 128,
    },
    timings: {
      prompt_per_second: 220.5,
      predicted_per_second: 107.2,
    },
    choices: [{ text: " Paris" }],
  }));

  expect(parsed.promptTokens).toBe(24);
  expect(parsed.prefillMs).toBeCloseTo((24 / 220.5) * 1000, 6);
  expect(parsed.prefillTps).toBe(220.5);
  expect(parsed.decodeMs).toBeCloseTo((128 / 107.2) * 1000, 6);
  expect(parsed.decodeTps).toBe(107.2);
  expect(parsed.msPerToken).toBeCloseTo(1000 / 107.2, 6);
  expect(parsed.outputPreview).toBe("Paris");
});

test("summarizeValues includes median, p95, and stddev", () => {
  const summary = summarizeValues([10, 20, 30, 40]);
  expect(summary?.avg).toBe(25);
  expect(summary?.median).toBe(25);
  expect(summary?.p95).toBe(38.5);
  expect(summary?.stddev).toBeGreaterThan(11);
});

test("buildComparison adds prompt and latency deltas", () => {
  const comparison = buildComparison(
    {
      name: "ZINC",
      prompt_tokens: 10,
      generated_tokens: 20,
      prefill_tps: { median: 50, avg: 50 },
      decode_tps: { median: 40, avg: 40 },
      total_latency_ms: { median: 2500, avg: 2500 },
      end_to_end_tps: { median: 30, avg: 30 },
    },
    {
      name: "llama.cpp",
      prompt_tokens: 10,
      generated_tokens: 20,
      prefill_tps: { median: 100, avg: 100 },
      decode_tps: { median: 80, avg: 80 },
      total_latency_ms: { median: 2000, avg: 2000 },
      end_to_end_tps: { median: 60, avg: 60 },
    },
  );

  expect(comparison?.pct_of_baseline).toBe(50);
  expect(comparison?.prompt_pct_of_baseline).toBe(50);
  expect(comparison?.latency_pct_of_baseline).toBe(125);
  expect(comparison?.latency_delta_ms).toBe(500);
  expect(comparison?.end_to_end_pct_of_baseline).toBe(50);
  expect(comparison?.end_to_end_delta_tps).toBe(-30);
  expect(comparison?.overall_pct_of_baseline).toBe(50);
  expect(comparison?.zinc_overall_tps).toBeCloseTo(42.857, 3);
  expect(comparison?.baseline_overall_tps).toBeCloseTo(85.714, 3);
  expect(comparison?.overall_delta_tps).toBeCloseTo(-42.857, 3);
});

test("buildComparison rates overall by backend wall time, not mixed throughput averages", () => {
  const comparison = buildComparison(
    {
      name: "ZINC",
      prompt_tokens: 36,
      generated_tokens: 96,
      prefill_tps: { median: 45.4, avg: 45.4 },
      decode_tps: { median: 50.46, avg: 50.46 },
    },
    {
      name: "llama.cpp",
      prompt_tokens: 35,
      generated_tokens: 96,
      prefill_tps: { median: 2820, avg: 2820 },
      decode_tps: { median: 55.64533518199503, avg: 55.64533518199503 },
    },
    { expectedGeneratedTokens: 96 },
  );

  expect(comparison?.prompt_pct_of_baseline).toBeCloseTo(1.61, 2);
  expect(comparison?.pct_of_baseline).toBeCloseTo(90.68, 2);
  expect(comparison?.zinc_overall_tps).toBeCloseTo(48.971, 3);
  expect(comparison?.baseline_overall_tps).toBeCloseTo(75.390, 3);
  expect(comparison?.overall_pct_of_baseline).toBeCloseTo(64.465, 3);
  expect(comparison?.overall_delta_tps).toBeCloseTo(-26.419, 3);
});

test("buildComparison withholds overall percentage for early-stop rows", () => {
  const comparison = buildComparison(
    {
      name: "ZINC",
      prompt_tokens: 36,
      generated_tokens: 2,
      prefill_tps: { median: 85, avg: 85 },
      decode_tps: { median: 140, avg: 140 },
    },
    {
      name: "llama.cpp",
      prompt_tokens: 35,
      generated_tokens: 96,
      prefill_tps: { median: 160, avg: 160 },
      decode_tps: { median: 100, avg: 100 },
    },
    { expectedGeneratedTokens: 96 },
  );

  expect(comparison?.pct_of_baseline).toBe(140);
  expect(comparison?.overall_pct_of_baseline).toBeNull();
  expect(comparison?.overall_comparable).toBe(false);
  expect(comparison?.overall_delta_tps).toBeNull();
});

test("buildArtifact skips non-comparable core rows for model headlines", () => {
  const artifact = buildArtifact([
    {
      id: "intel",
      label: "Intel",
      models: [
        {
          id: "m",
          label: "Model",
          scenarios: [
            {
              id: "core",
              max_tokens: 64,
              zinc: { name: "ZINC", prompt_tokens: 0, generated_tokens: 2, decode_tps: { median: 200, avg: 200 } },
              baseline: { name: "llama.cpp", prompt_tokens: 0, generated_tokens: 64, decode_tps: { median: 50, avg: 50 } },
            },
            {
              id: "context-long",
              max_tokens: 64,
              zinc: { name: "ZINC", prompt_tokens: 0, generated_tokens: 64, decode_tps: { median: 40, avg: 40 } },
              baseline: { name: "llama.cpp", prompt_tokens: 0, generated_tokens: 64, decode_tps: { median: 50, avg: 50 } },
            },
          ],
        },
      ],
    },
  ]);

  const model = artifact.targets[0]?.models[0];
  expect(model?.scenarios[0]?.comparison?.overall_comparable).toBe(false);
  expect(model?.max_tokens).toBe(64);
  expect(model?.zinc?.generated_tokens).toBe(64);
  expect(model?.comparison?.overall_pct_of_baseline).toBeCloseTo(80, 3);
  expect(artifact.targets[0]?.summary.average_pct_of_llama).toBeCloseTo(80, 3);
});

test("buildArtifact recomputes stale overall ratings and aggregates by total wall time", () => {
  const artifact = buildArtifact([
    {
      id: "cuda",
      label: "CUDA",
      models: [
        {
          id: "m",
          label: "Model",
          zinc: { name: "ZINC", prompt_tokens: 0, generated_tokens: 100, decode_tps: { median: 10, avg: 10 } },
          baseline: { name: "llama.cpp", prompt_tokens: 0, generated_tokens: 100, decode_tps: { median: 20, avg: 20 } },
          scenarios: [
            {
              id: "long",
              max_tokens: 100,
              zinc: { name: "ZINC", prompt_tokens: 0, generated_tokens: 100, decode_tps: { median: 10, avg: 10 } },
              baseline: { name: "llama.cpp", prompt_tokens: 0, generated_tokens: 100, decode_tps: { median: 20, avg: 20 } },
              comparison: { overall_pct_of_baseline: 999 },
            },
            {
              id: "short",
              max_tokens: 10,
              zinc: { name: "ZINC", prompt_tokens: 0, generated_tokens: 10, decode_tps: { median: 10, avg: 10 } },
              baseline: { name: "llama.cpp", prompt_tokens: 0, generated_tokens: 10, decode_tps: { median: 5, avg: 5 } },
            },
          ],
        },
      ],
    },
  ]);

  const model = artifact.targets[0]?.models[0];
  expect(model?.scenarios[0]?.comparison?.overall_pct_of_baseline).toBe(50);
  expect(model?.summary).toBeUndefined();
  expect(artifact.targets[0]?.summary.compared_models).toBe(1);
  expect(artifact.targets[0]?.summary.average_pct_of_llama).toBeCloseTo(63.636, 3);
});

test("mergeArtifacts replaces matching targets and preserves others", () => {
  const merged = mergeArtifacts(
    {
      schema_version: 1,
      generated_at: "old",
      targets: [
        {
          id: "rdna",
          label: "RDNA",
          provenance: {
            zinc: { version: "old-rdna", commit: "old-rdna-commit" },
            llama_cpp: { binary: "llama-server", version: "10", commit: "abc" },
          },
        },
        {
          id: "metal",
          label: "Metal old",
          provenance: {
            zinc: { version: "old-metal", commit: "old-metal-commit" },
            llama_cpp: { binary: "llama-cli", version: "20", commit: "def" },
          },
        },
      ],
    },
    [{
      id: "metal",
      label: "Metal new",
      provenance: {
        zinc: { version: "new-metal", commit: "new-metal-commit" },
        llama_cpp: { binary: "llama-server", version: "30", commit: "ghi" },
      },
    }],
  );

  expect(merged.targets[0]?.id).toBe("rdna");
  expect(merged.targets[0]?.label).toBe("RDNA");
  expect(merged.targets[0]?.provenance).toEqual({
    zinc: { version: "old-rdna", commit: "old-rdna-commit" },
    llama_cpp: { binary: "llama-server", version: "10", commit: "abc" },
  });
  expect(merged.targets[0]?.summary.fastest_model_id).toBeNull();
  expect(merged.targets[1].id).toBe("metal");
  expect(merged.targets[1].label).toBe("Metal new");
  expect(merged.targets[1].provenance).toEqual({
    zinc: { version: "new-metal", commit: "new-metal-commit" },
    llama_cpp: { binary: "llama-server", version: "30", commit: "ghi" },
  });
  expect(merged.targets[1].models).toEqual([]);
  expect(merged.targets[1].summary.fastest_model_id).toBeNull();
});

test("compareModelsByName normalizes published model label variants", () => {
  const models = [
    { id: "qwen36-35b-a3b-q4k-xl", label: "Qwen36 35B A3B Q4K XL" },
    { id: "qwen36-27b-q4k-m", label: "Qwen 3.6 27B Dense Q4_K_M" },
    { id: "qwen35-9b-q4k-m", label: "Qwen3.5 9B Q4K M" },
    { id: "gemma4-31b-q4k-m", label: "Gemma 4 31B Q4_K_M" },
    { id: "gemma4-26b-a4b-q4k-m", label: "Gemma 4 26B-A4B MoE Q4_K_M" },
  ];

  expect(models.sort(compareModelsByName).map((model) => model.id)).toEqual([
    "gemma4-26b-a4b-q4k-m",
    "gemma4-31b-q4k-m",
    "qwen35-9b-q4k-m",
    "qwen36-27b-q4k-m",
    "qwen36-35b-a3b-q4k-xl",
  ]);
});

test("mergeArtifacts replaces an existing target to avoid stale model rows", () => {
  const merged = mergeArtifacts(
    {
      schema_version: 1,
      generated_at: "old",
      targets: [
        {
          id: "metal",
          label: "Metal",
          models: [
            { id: "a", label: "A", zinc: { decode_tps: { median: 10, avg: 10 } } },
            { id: "b", label: "B", zinc: { decode_tps: { median: 20, avg: 20 } } },
          ],
          summary: {},
        },
      ],
    },
    [
      {
        id: "metal",
        label: "Metal",
        provenance: {
          zinc: { version: "zinc", commit: "zinc-commit" },
          llama_cpp: { binary: "llama-server", version: "42", commit: "xyz" },
        },
        models: [
          { id: "b", label: "B", zinc: { decode_tps: { median: 30, avg: 30 } } },
          { id: "c", label: "C", zinc: { decode_tps: { median: 40, avg: 40 } } },
        ],
        summary: {},
      },
    ],
  );

  const metal = merged.targets.find((target) => target.id === "metal");
  expect(metal?.models.map((model) => model.id)).toEqual(["b", "c"]);
  expect(metal?.summary.fastest_model_id).toBe("c");
  expect(metal?.provenance).toEqual({
    zinc: { version: "zinc", commit: "zinc-commit" },
    llama_cpp: { binary: "llama-server", version: "42", commit: "xyz" },
  });
});

test("mergeArtifacts can preserve missing phase data for split benchmark runs", () => {
  const merged = mergeArtifacts(
    {
      schema_version: 1,
      generated_at: "old",
      targets: [
        {
          id: "rdna",
          label: "RDNA",
          models: [
            {
              id: "qwen",
              label: "Qwen",
              zinc: { name: "ZINC", decode_tps: { median: 120, avg: 120 } },
              baseline: null,
              scenarios: [
                {
                  id: "core",
                  label: "Core Prompt",
                  zinc: { name: "ZINC", decode_tps: { median: 120, avg: 120 } },
                },
              ],
            },
          ],
        },
      ],
    },
    [
      {
        id: "rdna",
        label: "RDNA",
        models: [
          {
            id: "qwen",
            label: "Qwen",
            baseline: { name: "llama.cpp", decode_tps: { median: 100, avg: 100 } },
            scenarios: [
              {
                id: "core",
                label: "Core Prompt",
                baseline: { name: "llama.cpp", decode_tps: { median: 100, avg: 100 } },
              },
            ],
          },
        ],
      },
    ],
    { preserveMissingPhases: true },
  );

  const model = merged.targets[0]?.models[0];
  expect(model?.zinc?.decode_tps.median).toBe(120);
  expect(model?.baseline?.decode_tps.median).toBe(100);
  expect(model?.comparison?.pct_of_baseline).toBe(120);
  expect(model?.scenarios[0]?.comparison?.pct_of_baseline).toBe(120);
});

test("buildArtifact writes only the incoming targets", () => {
  const artifact = buildArtifact([
    {
      id: "metal",
      label: "Metal",
      provenance: {
        zinc: { version: "zinc", commit: "zinc-commit" },
        llama_cpp: { binary: "llama-server", version: "42", commit: "xyz" },
      },
      models: [{ id: "m", label: "Model M", zinc: { decode_tps: { median: 12, avg: 12 } } }],
    },
  ]);

  expect(artifact.targets.map((target) => target.id)).toEqual(["metal"]);
  expect(artifact.targets[0]?.summary.fastest_model_id).toBe("m");
  expect(artifact.targets[0]?.provenance).toEqual({
    zinc: { version: "zinc", commit: "zinc-commit" },
    llama_cpp: { binary: "llama-server", version: "42", commit: "xyz" },
  });
});

test("output quality status flags malformed benchmark previews", () => {
  expect(outputQualityStatus("<|im_end|>", 2).tone).toBe("caution");
  expect(outputQualityStatus("2\n</think>\n<|im_start|>0.\n<|im_end|>", 96).tone).toBe("caution");
  expect(outputQualityStatus("##\n<think>first</think>\n<think>second", 128).tone).toBe("caution");
  expect(outputQualityStatus("1.\n1.\n1.\n1.\n1.\n1.\n1.\n1.\n1.\n1.\n1.\n1.", 128).tone).toBe("caution");
  expect(outputQualityStatus(
    "1. What is the most important thing to remember about the relationship between the brain and the body? " +
    "2. What is the most important thing to remember about the relationship between the brain and the body? " +
    "3. What is the most important thing to remember about the relationship between the brain and the body? " +
    "4. What is the most important thing to remember about the relationship between the brain and the body?",
    96,
  ).tone).toBe("caution");
  expect(outputQualityStatus("This implementation plan explains the command shape and warmup policy.", 96).tone).toBe("positive");
});

test("artifact target summary excludes preview-flagged rows from headline stats", () => {
  const artifact = buildArtifact([
    {
      id: "intel",
      label: "Intel",
      models: [
        {
          id: "bad-fast",
          label: "Bad Fast",
          scenarios: [{
            id: "core",
            label: "Core",
            max_tokens: 96,
            zinc: {
              prompt_tokens: 10,
              generated_tokens: 96,
              output_preview: "2\n</think>\n<|im_start|>0.\n<|im_end|>",
              prefill_tps: { median: 100, avg: 100 },
              decode_tps: { median: 200, avg: 200 },
            },
            baseline: {
              prompt_tokens: 10,
              generated_tokens: 96,
              prefill_tps: { median: 100, avg: 100 },
              decode_tps: { median: 100, avg: 100 },
            },
          }],
        },
        {
          id: "good-slower",
          label: "Good Slower",
          scenarios: [{
            id: "core",
            label: "Core",
            max_tokens: 96,
            zinc: {
              prompt_tokens: 10,
              generated_tokens: 96,
              output_preview: "This implementation plan explains the command shape and warmup policy.",
              prefill_tps: { median: 90, avg: 90 },
              decode_tps: { median: 50, avg: 50 },
            },
            baseline: {
              prompt_tokens: 10,
              generated_tokens: 96,
              prefill_tps: { median: 90, avg: 90 },
              decode_tps: { median: 45, avg: 45 },
            },
          }],
        },
      ],
    },
  ]);

  const intel = artifact.targets[0];
  const badFast = intel?.models.find((model) => model.id === "bad-fast");
  expect(badFast?.zinc?.decode_tps.median).toBe(200);
  expect(badFast?.scenarios?.[0]?.output_quality?.tone).toBe("caution");
  expect(intel?.summary.fastest_model_id).toBe("good-slower");
  expect(intel?.summary.fastest_decode_tps).toBe(50);
  expect(intel?.summary.successful_models).toBe(1);
  expect(intel?.summary.compared_models).toBe(1);
});
