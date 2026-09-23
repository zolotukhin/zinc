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
  // No off-variant was measured, so the bar cannot be like-for-like and says so.
  expect(rows[0].comparable).toBe(false);
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
  // A speculative row with no off-variant is marked as NOT like-for-like.
  expect(svg).toContain("Fast Model *");
  expect(svg).toContain("NOT like-for-like");
});

test("a measured off-variant becomes the bar, and the speculative figure is footnoted", () => {
  const data = artifact();
  const core = data.targets[0].models[1].scenarios[0] as any;
  core.variants = { mtp_off: { zinc: { prefill_tps: metric(150), decode_tps: metric(120) } } };
  const { rows } = collectRows(data, "rdna-rocm");
  const fast = rows.find((r) => r.label === "Fast Model")!;
  // Bar uses the non-speculative 120 vs baseline 100, not the speculative 180.
  expect(fast.decodePct).toBe(120);
  expect(fast.comparable).toBe(true);
  expect(fast.speculativePct).toBe(180);
  // Prefill comes from the same non-speculative run as the decode bar.
  expect(fast.prefillPct).toBe(150);

  const svg = renderSvg(data, "rdna-rocm");
  expect(svg).toContain(">120%<");
  expect(svg).not.toContain(">180%<");
  expect(svg).toContain("both engines shown without speculative decoding");
  expect(svg).toContain("separate draft model");
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

test("renderSvg names every ZINC build when rows come from more than one run", () => {
  const data = artifact() as any;
  data.targets[0].models[0].provenance = { zinc: { version: "aaa111" } };
  data.targets[0].models[1].provenance = { zinc: { version: "bbb222" } };
  const svg = renderSvg(data, "rdna-rocm");
  expect(svg).toContain("ZINC aaa111 + bbb222");
});
