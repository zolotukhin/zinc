import { expect, test } from "bun:test";

import { mergeMtpVariant } from "./merge_mtp_variant.mjs";

function metric(median: number) {
  return { min: median, max: median, avg: median, median, p95: median, stddev: 0, samples: [median] };
}

function siteFixture() {
  return {
    targets: [
      {
        id: "rdna",
        models: [
          {
            id: "qwen38-27b-q4k-m",
            scenarios: [
              {
                id: "core",
                zinc: { name: "ZINC", decode_tps: metric(55), prefill_tps: metric(260), speculative_decoding: { enabled: true } },
                baseline: { name: "llama.cpp", decode_tps: metric(31), prefill_tps: metric(184) },
                comparison: { pct_of_baseline: 177 },
              },
            ],
          },
        ],
      },
    ],
  };
}

function offFixture(overrides: Record<string, unknown> = {}) {
  return {
    targets: [
      {
        id: "rdna",
        generated_at: "2026-09-19T20:42:00.000Z",
        provenance: { zinc: { version: "03a1d912b793" } },
        models: [
          {
            id: "qwen38-27b-q4k-m",
            scenarios: [
              {
                id: "core",
                zinc: { name: "ZINC", decode_tps: metric(32.8), prefill_tps: metric(249), ...overrides },
              },
            ],
          },
        ],
      },
    ],
  };
}

test("mergeMtpVariant attaches the off-variant and compares it to the published baseline", () => {
  const site = siteFixture();
  const { merged, skipped } = mergeMtpVariant(site, offFixture(), "rdna");

  expect(merged).toEqual(["qwen38-27b-q4k-m/core"]);
  expect(skipped).toEqual([]);

  const variant = site.targets[0].models[0].scenarios[0].variants.mtp_off;
  expect(variant.zinc.decode_tps.median).toBe(32.8);
  expect(variant.provenance.version).toBe("03a1d912b793");
  // 32.8 / 31 of the same published llama.cpp baseline.
  expect(Math.round(variant.comparison.pct_of_baseline)).toBe(106);

  // The MTP-on row itself is untouched.
  expect(site.targets[0].models[0].scenarios[0].zinc.decode_tps.median).toBe(55);
});

test("mergeMtpVariant refuses an artifact that still had speculation enabled", () => {
  expect(() => mergeMtpVariant(siteFixture(), offFixture({ speculative_decoding: { enabled: true } }), "rdna"))
    .toThrow(/still reports speculative decoding/);
});

test("mergeMtpVariant reports rows that have no published counterpart", () => {
  const site = siteFixture();
  site.targets[0].models[0].id = "some-other-model";
  const { merged, skipped } = mergeMtpVariant(site, offFixture(), "rdna");
  expect(merged).toEqual([]);
  expect(skipped).toEqual(["qwen38-27b-q4k-m (no published row)"]);
});

test("mergeMtpVariant rejects a target that is not published", () => {
  expect(() => mergeMtpVariant(siteFixture(), offFixture(), "metal")).toThrow(/no target 'metal'/);
});
