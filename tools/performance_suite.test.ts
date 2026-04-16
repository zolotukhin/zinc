import { expect, test } from "bun:test";

import {
  buildArtifact,
  buildComparison,
  buildMeasurementPhases,
  defaultMetalCases,
  defaultMaxTokensForModelId,
  defaultPromptForModelId,
  defaultScenarioDefsForModel,
  localZincCommand,
  mergeArtifacts,
  parseArgs,
  parseDotEnv,
  parseLlamaCliOutput,
  parseOpenAiCompletionOutput,
  parseZincCliOutput,
  prefersChatPrompt,
  resolveLocalLlamaServer,
  summarizeValues,
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
    "gemma4-12b-q4k-m,qwen3-8b-q4k-m",
    "--llama-cli",
    "/tmp/llama-cli",
    "--llama-server",
    "/tmp/llama-server",
    "--no-site-write",
  ]);

  expect(args.target).toBe("metal");
  expect(args.runs).toBe(5);
  expect(args.warmupRuns).toBe(2);
  expect(args.llamaCli).toBe("/tmp/llama-cli");
  expect(args.llamaServer).toBe("/tmp/llama-server");
  expect(args.writeSiteData).toBe(false);
  expect(args.models && [...args.models]).toEqual(["gemma4-12b-q4k-m", "qwen3-8b-q4k-m"]);
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

test("GPT-OSS uses the chat prompt path in the performance suite", () => {
  expect(prefersChatPrompt("gpt-oss-20b-q4k-m")).toBe(true);
  expect(defaultPromptForModelId("gpt-oss-20b-q4k-m")).toBe("What is the capital of France? Answer in one word.");
  expect(defaultMaxTokensForModelId("gpt-oss-20b-q4k-m")).toBe(48);
  expect(prefersChatPrompt("qwen3-8b-q4k-m")).toBe(false);
  expect(defaultPromptForModelId("qwen3-8b-q4k-m")).toBe("The capital of France is");
  expect(defaultMaxTokensForModelId("qwen3-8b-q4k-m")).toBe(8);
});

test("default Metal cases use managed cache ids and include Qwen 3.5", () => {
  const cases = defaultMetalCases("/tmp/models");
  const qwen35 = cases.find((entry) => entry.id === "qwen35-35b-a3b-q4k-xl");
  expect(qwen35?.model_id).toBe("qwen35-35b-a3b-q4k-xl");
  expect(qwen35?.model_path).toBe("/tmp/models/qwen35-35b-a3b-q4k-xl/model.gguf");
});

test("local ZINC command prefers managed model ids when using the default cache", () => {
  const cmd = localZincCommand({
    model_id: "qwen3-8b-q4k-m",
    model_path: "/Users/zolotukhin/Library/Caches/zinc/models/models/qwen3-8b-q4k-m/model.gguf",
    prompt_mode: "raw",
    prompt: "The capital of France is",
    max_tokens: 8,
  });
  expect(cmd).toContain("--model-id qwen3-8b-q4k-m");
  expect(cmd).not.toContain(" -m ");
});

test("benchmark suite uses a multi-scenario matrix instead of a single prompt", () => {
  const qwen = defaultScenarioDefsForModel("qwen3-8b-q4k-m", "raw", "The capital of France is");
  expect(qwen.map((scenario) => scenario.id)).toEqual(["core", "context-medium", "context-long", "decode-extended"]);
  expect(qwen[1]?.prompt).not.toBe(qwen[0]?.prompt);
  expect(qwen[3]?.max_tokens).toBe(32);

  const gptoss = defaultScenarioDefsForModel("gpt-oss-20b-q4k-m", "chat", "What is the capital of France? Answer in one word.");
  expect(gptoss[3]?.max_tokens).toBe(96);
});

test("benchmark suite measures all ZINC scenarios before starting baselines", () => {
  const phases = buildMeasurementPhases("qwen35-35b-a3b-q4k-xl", "raw", "The capital of France is");
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
      prefill_tps: { median: 50, avg: 50 },
      decode_tps: { median: 40, avg: 40 },
      total_latency_ms: { median: 2500, avg: 2500 },
      end_to_end_tps: { median: 30, avg: 30 },
    },
    {
      name: "llama.cpp",
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
});

test("mergeArtifacts replaces matching targets and preserves others", () => {
  const merged = mergeArtifacts(
    {
      schema_version: 1,
      generated_at: "old",
      targets: [
        { id: "rdna", label: "RDNA" },
        { id: "metal", label: "Metal old" },
      ],
    },
    [{ id: "metal", label: "Metal new" }],
  );

  expect(merged.targets[0]?.id).toBe("rdna");
  expect(merged.targets[0]?.label).toBe("RDNA");
  expect(merged.targets[0]?.summary.fastest_model_id).toBeNull();
  expect(merged.targets[1].id).toBe("metal");
  expect(merged.targets[1].label).toBe("Metal new");
  expect(merged.targets[1].models).toEqual([]);
  expect(merged.targets[1].summary.fastest_model_id).toBeNull();
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
});

test("buildArtifact writes only the incoming targets", () => {
  const artifact = buildArtifact([
    { id: "metal", label: "Metal", models: [{ id: "m", label: "Model M", zinc: { decode_tps: { median: 12, avg: 12 } } }] },
  ]);

  expect(artifact.targets.map((target) => target.id)).toEqual(["metal"]);
  expect(artifact.targets[0]?.summary.fastest_model_id).toBe("m");
});
