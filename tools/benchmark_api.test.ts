import { expect, test } from "bun:test";

import { defaultScenarios, parseArgs, percentile, summarizeResult } from "./benchmark_api.mjs";

test("percentile interpolates between sorted values", () => {
  expect(percentile([10, 20, 30, 40], 0.5)).toBe(25);
  expect(percentile([10, 20, 30, 40], 0.95)).toBe(38.5);
});

test("defaultScenarios splits chat and raw modes", () => {
  expect(defaultScenarios("chat").every((scenario) => scenario.kind === "chat")).toBe(true);
  expect(defaultScenarios("raw").every((scenario) => scenario.kind === "raw")).toBe(true);
  expect(defaultScenarios("both").some((scenario) => scenario.kind === "chat")).toBe(true);
  expect(defaultScenarios("both").some((scenario) => scenario.kind === "raw")).toBe(true);
});

test("parseArgs reads explicit overrides", () => {
  const args = parseArgs([
    "--base",
    "http://example.test/v1",
    "--mode",
    "raw",
    "--output",
    "/tmp/out.json",
    "--timeout-ms",
    "1234",
  ]);
  expect(args.base).toBe("http://example.test/v1");
  expect(args.mode).toBe("raw");
  expect(args.output).toBe("/tmp/out.json");
  expect(args.timeoutMs).toBe(1234);
});

test("summarizeResult formats non-streaming results", () => {
  const summary = summarizeResult({
    name: "raw_c1_t256",
    kind: "raw",
    prompt_chars: 24,
    max_tokens: 256,
    concurrency: 1,
    requests: 1,
    makespan_s: 15,
    latency_avg_s: 15,
    latency_p50_s: 15,
    latency_p95_s: 15,
    prompt_tokens_avg: 6,
    completion_tokens_avg: 256,
    completion_tps_avg: 16.7,
    aggregate_completion_tps: 16.7,
    aggregate_total_tps: 17.1,
  });
  expect(summary).toContain("raw_c1_t256");
  expect(summary).toContain("agg_completion_tps=16.70");
});

test("summarizeResult formats streaming results", () => {
  const summary = summarizeResult({
    name: "short_stream_c1_t64",
    kind: "chat",
    prompt_chars: 24,
    max_tokens: 64,
    concurrency: 1,
    stream: true,
    requests: 1,
    makespan_s: 5.5,
    latency_avg_s: 5.5,
    latency_p50_s: 5.5,
    latency_p95_s: 5.5,
    ttft_avg_s: 3.4,
    ttft_p50_s: 3.4,
    ttft_p95_s: 3.4,
    chunks_avg: 1,
  });
  expect(summary).toContain("short_stream_c1_t64");
  expect(summary).toContain("ttft_avg=3.40s");
});
