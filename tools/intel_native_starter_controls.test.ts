import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read('../site/public/zinc/validation/intel-native-starter-controls-2026-11-10.json');
const campaign = read(
  '../site/public/zinc/validation/intel-native-sycl-fp32-packed-q8-2026-11-10.json',
);

test('later starter controls retain the same frozen reference, model and official GPU stack', () => {
  expect(report.source_report_sha256).toBe(
    '44e612afa6ec2157f8c7c42b570b5c40a942183ef9e805674450a197b570192e',
  );
  expect(report.completed).toBe(true);
  expect(report.reference_commit).toBe(campaign.reference.commit);
  expect(report.harness_commit).toBe(campaign.harness_commit);
  expect(report.model).toBe('qwen3-0.6b-q8-0');
  expect(report.model_sha256).toBe(campaign.models[0].sha256);
  expect(report.hardware).toEqual(campaign.hardware_after);
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware_unchanged).toBe(true);
  expect(report.native_stack_sha256).toBe(campaign.native_stack.sha256);
  expect(report.native_stack_after_sha256).toBe(report.native_stack_sha256);
  expect(report.native_package_count).toBe(41);
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('all four attempts retain every repeated output and short JSON/SSE control', () => {
  expect(report.rows.map((row) => row.id)).toEqual([
    'fp32-default-reproduction',
    'fp32-async-copy-off',
    'fp32-fusion-off',
    'fp32-q8-wide-off',
  ]);
  const defaults = campaign.settings.native_runtime;
  expect(report.rows.map((row) => row.settings)).toEqual([
    defaults,
    { ...defaults, GGML_SYCL_USE_ASYNC_MEM_OP: '0' },
    { ...defaults, GGML_SYCL_ENABLE_FUSION: '0' },
    { ...defaults, GGML_SYCL_MMVQ_WIDE: '0' },
  ]);
  for (const row of report.rows) {
    expect(row.passed).toBe(true);
    expect(row.build).toEqual(campaign.native_build);
    expect(row.native_reference_validation.passed).toBe(true);
    expect(row.native_reference_validation.cases).toHaveLength(3);
    expect(row.samples).toHaveLength(24);
    expect(row.streams).toHaveLength(8);
    expect(row.output_drift).toEqual([]);
    for (const scenario of ['core', 'context-medium', 'context-long', 'decode-extended']) {
      const samples = row.samples.filter((sample) => sample.scenario === scenario);
      expect(new Set(samples.map((sample) => sample.content_sha256)).size).toBe(1);
      for (const lifetime of [0, 1]) {
        const group = samples.filter((sample) => sample.lifetime === lifetime);
        expect(group.map((sample) => sample.repeat)).toEqual([0, 1, 2]);
        expect(group.map((sample) => sample.warmup)).toEqual([true, false, false]);
      }
    }
    expect(row.streams.every((stream) => stream.matches_json && stream.output_cap === 16)).toBe(
      true,
    );
    expect(row.processes).toHaveLength(2);
    for (const process of row.processes) {
      expect(process.native_runtime.runtime).toBe('level_zero');
      expect(process.native_runtime_error).toBeUndefined();
      expect(process.gpu_offload.offloaded).toBe(process.gpu_offload.total);
      expect(process.gpu_offload.total).toBeGreaterThan(0);
    }
  }
});

test('a later passing default cannot erase the earlier drift or qualify all-model performance', () => {
  expect(campaign.models[0].passed).toBe(false);
  expect(campaign.models[0].error).toBe('llama context-medium repeated complete outputs differ');
  expect(report.rows[0].settings).toEqual(campaign.settings.native_runtime);
  expect(report.settings.server_lifetimes_per_treatment).toBe(2);
  expect(report.reference_is_latest_claim).toBe(false);
  expect(report.release_qualification).toBe(false);
  expect(report.performance_qualified).toBe(false);
  expect(report.complete_model_matrix).toBe(false);
  expect(report.all_models_all_dimensions_better).toBe(false);
  expect(report.cleanup).toBe('no GPU process remains');
});
