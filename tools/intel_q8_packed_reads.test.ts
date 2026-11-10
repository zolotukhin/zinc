import { expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read('../site/public/zinc/validation/intel-q8-packed-reads-2026-11-09.json');
const support = read(
  '../site/public/zinc/validation/intel-native-sycl-fp32-baseline-2026-11-09.json',
);
const policy = read('../config/release/targets.json');
const median = (values: number[]) =>
  [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

test('packed Q8 evidence binds the shader source and bitwise proof to the qualified binary', () => {
  const source = readFileSync(new URL('../src/shaders/dmmv_q8_0_intel_rows.comp', import.meta.url));
  expect(createHash('sha256').update(source).digest('hex')).toBe(report.candidate_source_sha256);
  expect(report.numerical_proof.source_sha256['dmmv_q8_0_intel_rows.comp']).toBe(
    report.candidate_source_sha256,
  );
  expect(report.numerical_proof).toEqual(
    expect.objectContaining({ passed: true, cases: 228, bitwise_pairs: 114 }),
  );
  expect(report.binary_sha256).toBe(support.zinc.sha256);
  expect(report.baseline_shader_sha256).toBe(support.zinc.shader_sha256[report.changed_shader]);
  expect(report.candidate_shader_sha256).not.toBe(report.baseline_shader_sha256);
  expect(report.changed_shader).toBe('share/zinc/shaders/dmmv_q8_0_intel_rows.spv');
  expect(report.model).toBe('qwen3-0.6b-q8-0');
  expect(report.model_sha256).toBe(policy.models.find((model) => model.id === report.model).sha256);
  expect(report.hardware).toEqual(support.hardware_after);
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware_unchanged).toBe(true);
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('three balanced server rounds retain identical full output and all warmups', () => {
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
  expect(report.processes).toHaveLength(6);
  expect(report.processes.map((process) => [process.round, process.treatment])).toEqual([
    [1, 'baseline'],
    [1, 'candidate'],
    [2, 'candidate'],
    [2, 'baseline'],
    [3, 'baseline'],
    [3, 'candidate'],
  ]);
  for (const process of report.processes)
    expect(process.native_before && process.native_after && process.healthy_after).toBe(true);
  expect(report.samples).toHaveLength(96);
  const scenarios = ['core', 'context-medium', 'context-long', 'decode-extended'];
  expect([...new Set(report.samples.map((sample) => sample.scenario))].sort()).toEqual(
    [...scenarios].sort(),
  );
  for (const scenario of scenarios) {
    const samples = report.samples.filter((sample) => sample.scenario === scenario);
    // Equal throughput numbers alone cannot prove the same generated work.
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
    for (const round of [1, 2, 3])
      for (const treatment of ['baseline', 'candidate']) {
        const group = samples.filter(
          (sample) => sample.round === round && sample.treatment === treatment,
        );
        expect(group).toHaveLength(4);
        expect(group.map((sample) => sample.warmup)).toEqual([true, false, false, false]);
      }
  }
  expect(report.samples.some((sample) => !sample.output_budget_complete)).toBe(true);
  expect(report.cleanup).toBe('no GPU process remains');
});

test('every decode improvement comes from all three measured requests in each balanced round', () => {
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
        expect(samples).toHaveLength(3);
        for (const metric of ['prefillTps', 'decodeTps', 'totalLatencyMs'])
          expect(round[treatment][metric]).toBe(median(samples.map((sample) => sample[metric])));
      }
      expect(round.decode_ratio).toBe(round.candidate.decodeTps / round.baseline.decodeTps);
      expect(round.decode_ratio).toBeGreaterThan(1.15);
      expect(round.latency_ratio).toBeGreaterThan(1.12);
    }
  }
  // This is an internal shader comparison with EOS retained. It cannot make
  // claims about a competitor, unrelated models or unmeasured dimensions.
  expect(report.release_qualification).toBe(false);
  expect(report.all_models_all_dimensions_better).toBe(false);
  expect(report.limitations.some((text) => text.includes('Q4 candidate remains untested'))).toBe(
    true,
  );
});
