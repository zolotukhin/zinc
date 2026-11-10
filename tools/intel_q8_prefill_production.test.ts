import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read(
  '../site/public/zinc/validation/intel-q8-prefill-production-ab-2026-11-10.json',
);
const support = read(
  '../site/public/zinc/validation/intel-q8-prefill-package-support-2026-11-10.json',
);
const medianOfThree = (values: number[]) => {
  expect(values).toHaveLength(3);
  return [...values].sort((a, b) => a - b)[1];
};

test('production prefill A/B uses the exact qualified archive and explicit selected pipelines', () => {
  expect(report.source_report_sha256).toBe(
    'f3ed13fa564964ca799f45e873a88596fbb016cd3a2a9117e8859c6507b77a11',
  );
  expect(report.archive).toEqual(support.archive);
  expect(report.binary_sha256).toBe(support.binary_sha256);
  expect(report.shader_sha256).toEqual(support.shader_sha256);
  expect(Object.keys(report.shader_sha256)).toHaveLength(215);
  expect(report.model).toBe('qwen3-0.6b-q8-0');
  expect(report.model_sha256).toBe(
    support.models.find((model) => model.id === report.model).sha256,
  );
  expect(report.control).toEqual({ baseline: { ZINC_INTEL_Q8_PREFILL: '0' }, candidate: {} });
  for (const process of report.processes) {
    expect(process.selected_shader).toBe(
      process.treatment === 'candidate' ? 'mul_mm_q8_0_intel.spv' : 'mul_mm_q8_0.spv',
    );
  }
  expect(report.hardware).toEqual(support.hardware_after);
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware_unchanged).toBe(true);
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('three balanced prefill rounds preserve every warmup and identical complete output', () => {
  expect(report.completed && report.passed).toBe(true);
  expect(report.settings).toEqual({
    kv_cache: 'f16',
    mtp: 'off',
    prompt_cache: false,
    slots: 1,
    rounds: 3,
    warmup: 1,
    runs: 3,
  });
  expect(report.samples).toHaveLength(96);
  expect(report.processes.map((process) => [process.round, process.treatment])).toEqual([
    [1, 'baseline'],
    [1, 'candidate'],
    [2, 'candidate'],
    [2, 'baseline'],
    [3, 'baseline'],
    [3, 'candidate'],
  ]);
  for (const process of report.processes) {
    expect(process.native_before && process.native_after && process.healthy_after).toBe(true);
  }
  for (const scenario of ['core', 'context-medium', 'context-long', 'decode-extended']) {
    const samples = report.samples.filter((sample) => sample.scenario === scenario);
    expect(samples).toHaveLength(24);
    expect(
      new Set(
        samples.map((sample) =>
          JSON.stringify([
            sample.content_sha256,
            sample.promptTokens,
            sample.generatedTokens,
            sample.finish_reason,
          ]),
        ),
      ).size,
    ).toBe(1);
    for (const round of [1, 2, 3]) {
      for (const treatment of ['baseline', 'candidate']) {
        const group = samples.filter(
          (sample) => sample.round === round && sample.treatment === treatment,
        );
        expect(group.map((sample) => sample.warmup)).toEqual([true, false, false, false]);
      }
    }
  }
  expect(report.samples.some((sample) => !sample.output_budget_complete)).toBe(true);
  expect(report.cleanup).toBe('no GPU process remains');
});

test('the measured prefill gain keeps decode stable and makes no broader performance claim', () => {
  expect(report.comparisons).toHaveLength(4);
  for (const comparison of report.comparisons) {
    expect(comparison.rounds).toHaveLength(3);
    for (const round of comparison.rounds) {
      for (const treatment of ['baseline', 'candidate']) {
        const samples = report.samples.filter(
          (sample) =>
            !sample.warmup &&
            sample.scenario === comparison.scenario &&
            sample.round === round.round &&
            sample.treatment === treatment,
        );
        for (const metric of ['prefillTps', 'decodeTps', 'totalLatencyMs']) {
          expect(round[treatment][metric]).toBe(
            medianOfThree(samples.map((sample) => sample[metric])),
          );
        }
      }
      expect(round.prefill_ratio).toBe(round.candidate.prefillTps / round.baseline.prefillTps);
      expect(round.prefill_ratio).toBeGreaterThan(1.1);
      expect(round.decode_ratio).toBe(round.candidate.decodeTps / round.baseline.decodeTps);
      expect(round.decode_ratio).toBeGreaterThan(0.99);
      expect(round.decode_ratio).toBeLessThan(1.01);
    }
  }
  expect(report.release_qualification).toBe(false);
  expect(report.complete_model_matrix).toBe(false);
  expect(report.all_models_all_dimensions_better).toBe(false);
});
