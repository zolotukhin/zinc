import { expect, test } from 'bun:test';

import { collectRows, renderSvg } from './render_benchmark_svg.mjs';

function metric(median: number) {
  return {
    min: median,
    max: median,
    avg: median,
    median,
    p95: median,
    stddev: 0,
    samples: [median],
  };
}

function scenario(
  id: string,
  zincPrefill: number,
  zincDecode: number,
  basePrefill: number,
  baseDecode: number,
  speculative = false,
) {
  return {
    id,
    zinc: {
      prefill_tps: metric(zincPrefill),
      decode_tps: metric(zincDecode),
      speculative_decoding: { enabled: speculative },
    },
    baseline: {
      speculative_decoding: { enabled: speculative },
      prefill_tps: metric(basePrefill),
      decode_tps: metric(baseDecode),
    },
  };
}

function artifact() {
  return {
    targets: [
      {
        id: 'rdna-rocm',
        generated_at: '2026-09-21T01:23:40.356Z',
        machine: { gpu: 'Radeon AI PRO R9700' },
        provenance: { zinc: { version: 'abc123' }, llama_cpp: { commit: 'def456' } },
        models: [
          {
            id: 'slow',
            label: 'Slow Model Q4_K_M',
            scenarios: [scenario('core', 110, 105, 100, 100)],
          },
          {
            id: 'fast',
            label: 'Fast Model Q4_K_M',
            scenarios: [
              {
                ...scenario('core', 200, 180, 100, 100, true),
                variants: { mtp_off: scenario('core', 150, 120, 100, 100) },
              },
            ],
          },
          {
            id: 'incomplete',
            label: 'No Baseline',
            scenarios: [{ id: 'core', zinc: { decode_tps: metric(50) } }],
          },
        ],
      },
    ],
  };
}

test('collectRows keeps complete core rows, strongest first', () => {
  const { rows } = collectRows(artifact(), 'rdna-rocm');
  expect(rows.map((r) => r.label)).toEqual(['Fast Model', 'Slow Model']);
  expect(rows[0].prefillPct).toBe(150);
  expect(rows[0].comparable).toBe(true);
  expect(rows[1].decodePct).toBe(105);
  // The row without a baseline cannot be a comparison, so it is dropped.
  expect(rows.some((r) => r.label === 'No Baseline')).toBe(false);
});

test('renderSvg draws one labelled pair per model and cites provenance', () => {
  const svg = renderSvg(artifact(), 'rdna-rocm');
  expect(svg.startsWith('<svg')).toBe(true);
  expect(svg).toContain('Radeon AI PRO R9700');
  expect(svg).toContain('>1.50×<');
  expect(svg).toContain('>1.05×<');
  expect(svg).toContain('>150.0<');
  expect(svg).toContain('>100.0<');
  expect(svg).toContain('ZINC abc123');
  expect(svg).toContain('llama.cpp def456');
  expect(svg).toContain('Fast Model');
  expect(svg).toContain('MTP off for both engines');
});

test('a complete off-variant selects its own baseline as well as ZINC', () => {
  const data = artifact();
  const core = data.targets[0].models[1].scenarios[0] as any;
  core.variants = { mtp_off: scenario('core', 150, 120, 50, 60) };
  const { rows } = collectRows(data, 'rdna-rocm');
  const fast = rows.find((r) => r.label === 'Fast Model')!;
  // Bar uses the non-speculative 120 vs baseline 100, not the speculative 180.
  expect(fast.decodePct).toBe(200);
  expect(fast.comparable).toBe(true);
  // Prefill comes from the same non-speculative run as the decode bar.
  expect(fast.prefillPct).toBe(300);

  const svg = renderSvg(data, 'rdna-rocm');
  expect(svg).toContain('>2.00×<');
  expect(svg).not.toContain('>1.80×<');
  expect(svg).toContain('MTP off for both engines');
});

test('renderSvg keeps every bar and label inside the card', () => {
  for (const layout of ['wide', 'compact']) {
    for (const theme of ['light', 'dark']) {
      const svg = renderSvg(artifact(), 'rdna-rocm', { layout, theme });
      const [, width, height] = svg.match(/viewBox="0 0 (\d+) (\d+)"/)!;
      const bars = [
        ...svg.matchAll(/<rect x="([\d.]+)" y="([\d.]+)" width="([\d.]+)" height="([\d.]+)"/g),
      ];
      expect(bars.length).toBeGreaterThan(16);
      for (const [, x, y, w, h] of bars) {
        expect(Number(x) + Number(w)).toBeLessThanOrEqual(Number(width));
        expect(Number(y) + Number(h)).toBeLessThanOrEqual(Number(height));
      }
      for (const [, x, y] of svg.matchAll(/<text x="([\d.]+)" y="([\d.]+)"/g)) {
        expect(Number(x)).toBeLessThan(Number(width));
        expect(Number(y)).toBeLessThan(Number(height));
      }
    }
  }
});

test('renderSvg rejects a target that is not in the artifact', () => {
  expect(() => renderSvg(artifact(), 'metal')).toThrow(/No target 'metal'/);
});

test('renderSvg names every ZINC build when rows come from more than one run', () => {
  const data = artifact() as any;
  data.targets[0].models[0].provenance = { zinc: { version: 'aaa111' } };
  data.targets[0].models[1].provenance = { zinc: { version: 'bbb222' } };
  const svg = renderSvg(data, 'rdna-rocm');
  expect(svg).toContain('ZINC aaa111');
  expect(svg).toContain('ZINC bbb222');
  expect(svg).toContain('Mixed measurement dates and revisions');
});

test('unknown modes, mismatched modes, missing off variants and correctness cautions cannot become bars', () => {
  for (const change of [
    (core: any) => {
      delete core.variants;
    },
    (core: any) => {
      delete core.variants.mtp_off.baseline.speculative_decoding;
    },
    (core: any) => {
      core.variants.mtp_off.baseline.speculative_decoding.enabled = true;
    },
    (core: any) => {
      core.variants.mtp_off.correctness_review = { tone: 'caution' };
    },
  ]) {
    const data = artifact();
    change(data.targets[0].models[1].scenarios[0]);
    const { rows } = collectRows(data, 'rdna-rocm');
    expect(rows.map((row) => row.label)).toEqual(['Slow Model']);
    expect(renderSvg(data, 'rdna-rocm')).not.toContain('Fast Model');
  }
});
