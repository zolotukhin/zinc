import { expect, test } from 'bun:test';
import data from '../data/zinc-performance.json';
import { buildComparison } from '../../../tools/performance_suite.mjs';
import {
  aggregate,
  formatDate,
  hasMtpVariants,
  metric,
  overallRatio,
  phaseSeconds,
  promptParity,
  qualityStatus,
  rateRatio,
  rowNotes,
  rowsFor,
  scenarioOrder,
} from './benchmark-view';

test('MTP off selects the entire measured variant without changing the artifact', () => {
  const target = data.targets.find((target) => target.id === 'rdna-rocm')!;
  const original = target.models.find((model) => model.id === 'qwen38-27b-q4k-m')!.scenarios[0];
  const snapshot = JSON.stringify(target);
  const off = rowsFor(target, 'core', 'off').find((row) => row.model.id === 'qwen38-27b-q4k-m')!;
  const on = rowsFor(target, 'core', 'on').find((row) => row.model.id === 'qwen38-27b-q4k-m')!;
  expect(off.zinc).toEqual(original.variants!.mtp_off.zinc);
  expect(off.measured_at).toBe(original.variants!.mtp_off.measured_at);
  expect(off.provenance.zinc).toEqual(original.variants!.mtp_off.provenance);
  expect(off.baseline).toEqual(on.baseline);
  expect(metric(off.zinc.decode_tps)).toBe(32.79);
  expect(metric(on.zinc.decode_tps)).toBe(63.76);
  expect(metric(off.zinc.prefill_tps)).not.toBe(metric(on.zinc.prefill_tps));
  expect(aggregate(rowsFor(target, 'core', 'off')).decode).not.toBe(
    aggregate(rowsFor(target, 'core', 'on')).decode,
  );
  expect(JSON.stringify(target)).toBe(snapshot);
});

test('displayed phase times agree with the harness across every published target and MTP mode', () => {
  let compared = 0;
  for (const target of data.targets) {
    for (const mode of hasMtpVariants(target) ? (['off', 'on'] as const) : (['off'] as const)) {
      for (const scenarioId of scenarioOrder) {
        for (const row of rowsFor(target, scenarioId, mode)) {
          const reference = buildComparison(row.zinc, row.baseline, {
            expectedGeneratedTokens: row.max_tokens,
          });
          if (reference?.zinc_overall_seconds != null) {
            expect(phaseSeconds(row.zinc)).toBeCloseTo(reference.zinc_overall_seconds, 10);
          }
          const ratio = overallRatio(row);
          if (ratio != null) {
            expect(ratio).toBeCloseTo(reference.overall_pct_of_baseline / 100, 10);
            expect(promptParity(row)).toBe(true);
            compared++;
          }
        }
      }
    }
  }
  expect(compared).toBeGreaterThan(50);
});

const engine = (decode: number, generated = 10) => ({
  speculative_decoding: { enabled: false },
  prompt_tokens: 10,
  prefill_tokens: 10,
  generated_tokens: generated,
  prefill_tps: { median: 10 },
  decode_tps: { median: decode },
});
const row = (zinc = engine(10), baseline = engine(20)) => ({ zinc, baseline, max_tokens: 10 });

test('overall summary compares summed times, not the average of speed ratios', () => {
  const rows = [row(engine(10), engine(20)), row(engine(20), engine(10))];
  expect(aggregate(rows).overall).toBe(1);
  expect(overallRatio(rows[0])).toBe(0.75);
  expect(overallRatio(rows[1])).toBeCloseTo(4 / 3);
});

test('missing data, early stops, mismatched prompts, and flagged output cannot win the overall comparison', () => {
  const missing = row();
  delete (missing.zinc as any).decode_tps;
  const mismatch = row();
  mismatch.zinc.prompt_tokens = 15;
  const flagged = { ...row(), output_quality: { tone: 'caution', note: 'Repeated output.' } };
  const rows = [missing, mismatch, row(engine(10, 4)), flagged];
  for (const item of rows) expect(overallRatio(item)).toBeNull();
  expect(aggregate(rows).overall).toBeNull();
  expect(aggregate(rows).comparable).toBe(0);
  expect(metric({ median: NaN })).toBeNull();
  expect(metric({ median: Infinity })).toBeNull();
  expect(phaseSeconds({})).toBeNull();
});

test('a later correctness finding excludes every variant while preserving recorded rates', () => {
  const reviewed = {
    ...row(engine(1000)),
    model: { correctness_review: { tone: 'caution', note: 'Known incorrect prefill.' } },
    output_quality: { tone: 'positive', note: 'Fluent preview.' },
  };
  expect(qualityStatus(reviewed).tone).toBe('caution');
  expect(metric(reviewed.zinc.decode_tps)).toBe(1000);
  expect(overallRatio(reviewed)).toBeNull();
  expect(rateRatio(reviewed, 'prefill_tps')).toBeNull();
  expect(rateRatio(reviewed, 'decode_tps')).toBeNull();
  expect(aggregate([reviewed, row()])).toMatchObject({ completed: 1, comparable: 1, decode: 10 });

  const before = JSON.stringify(data);
  let reviewedModels = 0;
  for (const target of data.targets) {
    for (const model of target.models) {
      if (!(model as any).correctness_review) continue;
      reviewedModels++;
      for (const mode of ['off', 'on'] as const) {
        for (const scenario of scenarioOrder) {
          const selected = rowsFor(target, scenario, mode).find((r) => r.model.id === model.id);
          if (!selected) continue;
          expect(qualityStatus(selected).tone).toBe('caution');
          expect(qualityStatus(selected).source_url).toMatch(/^\/zinc\/docs\//);
          expect(overallRatio(selected)).toBeNull();
          expect(aggregate([selected]).completed).toBe(0);
        }
      }
    }
  }
  // Restoring comparisons needs an explicit review of replacement measurements.
  expect(reviewedModels).toBe(5);
  expect(JSON.stringify(data)).toBe(before);
});

test('unavailable measurements retain their correctness note and investigation link', () => {
  const reviewed = {
    ...row(),
    zinc: {
      speculative_decoding: { enabled: false },
      unavailable_reason: 'Earlier run could not finish.',
    },
    model: {
      correctness_review: {
        tone: 'caution',
        note: 'Known incorrect prefill.',
        source_url: '/zinc/docs/vulkan-gemma-sliding-window/',
      },
    },
  };
  expect(rowNotes(reviewed)).toHaveLength(2);
  expect(rowNotes(reviewed)[0].text).toContain('unavailable');
  expect(rowNotes(reviewed)[1]).toEqual({
    text: 'Known incorrect prefill. Excluded from comparison summaries.',
    href: '/zinc/docs/vulkan-gemma-sliding-window/',
  });
  // A positive model review must not suppress a separately flagged output.
  expect(
    qualityStatus({
      ...row(),
      model: { correctness_review: { tone: 'positive' } },
      output_quality: { tone: 'caution', note: 'Repetitive output.' },
    }).tone,
  ).toBe('caution');
});

test('MTP off never silently displays an unpaired speculative measurement', () => {
  const target = {
    models: [
      {
        label: 'Unpaired',
        scenarios: [
          {
            ...row(),
            id: 'core',
            zinc: { ...engine(100), speculative_decoding: { enabled: true } },
          },
        ],
      },
    ],
  };
  expect(hasMtpVariants(target)).toBe(false);
  const off = rowsFor(target, 'core', 'off')[0];
  expect(metric(off.zinc.decode_tps)).toBeNull();
  expect(off.zinc.unavailable_reason).toContain('MTP off');
  expect(metric(rowsFor(target, 'core', 'on')[0].zinc.decode_tps)).toBe(100);
});

test('each MTP mode uses its own output quality and preserves baseline provenance', () => {
  const target = {
    provenance: { llama_cpp: { commit: 'baseline' } },
    models: [
      {
        label: 'Variant',
        provenance: { zinc: { commit: 'zinc' } },
        scenarios: [
          {
            ...row(),
            id: 'core',
            zinc: { ...engine(100), speculative_decoding: { enabled: true } },
            output_quality: { tone: 'caution', note: 'Flagged original run.' },
            variants: { mtp_off: { zinc: { ...engine(10), output_preview: 'A useful answer.' } } },
          },
        ],
      },
    ],
  };
  const off = rowsFor(target, 'core', 'off')[0];
  expect(qualityStatus(off).tone).toBe('positive');
  expect(overallRatio(off)).not.toBeNull();
  expect(off.provenance.llama_cpp.commit).toBe('baseline');
  expect(qualityStatus(rowsFor(target, 'core', 'on')[0]).tone).toBe('caution');
  off.zinc = { ...off.zinc, output_preview: 'Repeat this. '.repeat(20) };
  expect(qualityStatus(off).tone).toBe('caution');
  expect(overallRatio(off)).toBeNull();
});

test('dates use a fixed UTC calendar regardless of the build machine timezone', () => {
  expect(formatDate('2026-09-25T00:05:00Z')).toBe('Sep 25, 2026');
  expect(formatDate('invalid')).toBe('Not recorded');
});

test('historical flat model records normalize to a usable core workload', () => {
  const target = { models: [{ id: 'legacy', label: 'Legacy model', ...row() }] };
  const legacy = rowsFor(target, 'core', 'off');
  expect(legacy).toHaveLength(1);
  expect(legacy[0].id).toBe('core');
  expect(metric(legacy[0].zinc.decode_tps)).toBe(10);
  expect(overallRatio(legacy[0])).toBe(0.75);
});

test('MTP mismatch and missing telemetry hide ratios without altering raw rates', () => {
  const on = { ...engine(200), speculative_decoding: { enabled: true } };
  for (const baseline of [engine(10), { ...engine(10), speculative_decoding: undefined }]) {
    const value = row(on, baseline);
    expect(overallRatio(value)).toBeNull();
    expect(rateRatio(value, 'decode_tps')).toBeNull();
    expect(aggregate([value]).comparable).toBe(0);
    expect(metric(value.zinc.decode_tps)).toBe(200);
    expect(metric(value.baseline.decode_tps)).toBe(10);
    expect(
      rowNotes(value)
        .map((note) => note.text)
        .join(' '),
    ).toContain('Excluded from comparisons');
  }
});
