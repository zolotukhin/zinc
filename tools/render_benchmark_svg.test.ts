import { expect, test } from "bun:test";

import { collectRows, renderSvg } from "./render_benchmark_svg.mjs";

function metric(median: number) {
  return { min: median, max: median, avg: median, median, p95: median, stddev: 0, samples: [median] };
}

function scenario(id: string, zincPrefill: number, zincDecode: number, basePrefill: number, baseDecode: number, speculative = false) {
  return {
    id,
    zinc: {
      prefill_tps: metric(zincPrefill),
      decode_tps: metric(zincDecode),
      ...(speculative ? { speculative_decoding: { enabled: true } } : {}),
    },
    baseline: { prefill_tps: metric(basePrefill), decode_tps: metric(baseDecode) },
  };
}

function artifact() {
  return {
    targets: [
      {
        id: "rdna-rocm",
        generated_at: "2026-09-21T01:23:40.356Z",
        machine: { gpu: "Radeon AI PRO R9700" },
        provenance: { zinc: { version: "abc123" }, llama_cpp: { commit: "def456" } },
        models: [
          { id: "slow", label: "Slow Model Q4_K_M", scenarios: [scenario("core", 110, 105, 100, 100)] },
          { id: "fast", label: "Fast Model Q4_K_M", scenarios: [scenario("core", 200, 180, 100, 100, true)] },
          { id: "incomplete", label: "No Baseline", scenarios: [{ id: "core", zinc: { decode_tps: metric(50) } }] },
        ],
      },
    ],
  };
}

test("collectRows keeps complete core rows, strongest first", () => {
  const { rows } = collectRows(artifact(), "rdna-rocm");
  expect(rows.map((r) => r.label)).toEqual(["Fast Model", "Slow Model"]);
  expect(rows[0].prefillPct).toBe(200);
  expect(rows[0].speculative).toBe(true);
  expect(rows[1].decodePct).toBe(105);
  // The row without a baseline cannot be a comparison, so it is dropped.
  expect(rows.some((r) => r.label === "No Baseline")).toBe(false);
});

test("renderSvg draws one labelled pair per model and cites provenance", () => {
  const svg = renderSvg(artifact(), "rdna-rocm");
  expect(svg.startsWith("<svg")).toBe(true);
  expect(svg).toContain("Radeon AI PRO R9700");
  expect(svg).toContain(">200%<");
  expect(svg).toContain(">105%<");
  expect(svg).toContain("ZINC abc123");
  expect(svg).toContain("llama.cpp def456");
  // The speculative row is marked and explained.
  expect(svg).toContain("Fast Model *");
  expect(svg).toContain("NextN block to draft tokens");
});

test("renderSvg keeps every bar and label inside the card", () => {
  const svg = renderSvg(artifact(), "rdna-rocm");
  const cardRight = 1000;
  for (const [, x, w] of [...svg.matchAll(/<rect x="([\d.]+)" y="\d+" width="([\d.]+)" height="17"/g)]) {
    expect(Number(x) + Number(w)).toBeLessThanOrEqual(cardRight);
  }
  for (const [, x] of [...svg.matchAll(/<text x="([\d.]+)"[^>]*font-size="13.5"/g)]) {
    expect(Number(x)).toBeLessThan(cardRight);
  }
});

test("renderSvg rejects a target that is not in the artifact", () => {
  expect(() => renderSvg(artifact(), "metal")).toThrow(/No target 'metal'/);
});
