import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read('../site/public/zinc/validation/intel-native-35b-controls-2026-11-09.json');
const support = read(
  '../site/public/zinc/validation/intel-native-sycl-fp32-baseline-2026-11-09.json',
);

test('35B controls preserve the exact failing reference and official GPU runtime', () => {
  expect(report.completed).toBe(true);
  expect(report.reference_commit).toBe(support.reference.commit);
  expect(report.source_report_sha256).toBe(
    'c6abc21d64803656d2681e237496b9a446a14207d477481890e467ea6508c50d',
  );
  expect(report.harness_commit).toBe('17409ad1374a974d4ec89ddadecfb6924dda4aa6');
  expect(report.model).toBe('qwen36-35b-a3b-q4k-xl');
  expect(report.model_sha256).toBe(
    support.models.find((model) => model.id === report.model).sha256,
  );
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware).toEqual(support.hardware_after);
  expect(report.hardware_unchanged).toBe(true);
  expect(report.native_stack_sha256).toBe(support.native_stack.sha256);
  expect(report.native_stack_after_sha256).toBe(report.native_stack_sha256);
  expect(report.native_package_count).toBe(41);
  for (const row of report.rows) {
    expect(row.build.settings.GGML_SYCL_F16).toBe('OFF');
    expect(row.build.settings.GGML_SYCL_HOST_MEM_FALLBACK).toBe('OFF');
    expect(row.native_reference_validation.passed).toBe(true);
    expect(row.native_reference_validation.cases).toHaveLength(3);
    expect(row.processes).toHaveLength(2);
    for (const process of row.processes) {
      expect(process.gpu_offload).toEqual({ backend: 'sycl', offloaded: 41, total: 41 });
      expect(process.native_runtime.runtime).toBe('level_zero');
      expect(Object.keys(process.native_runtime.library_sha256).length).toBeGreaterThan(0);
      expect(process.native_runtime_error).toBeUndefined();
    }
  }
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('both treatments retain every request and their actual full-output drift', () => {
  expect(report.rows.map((row) => row.id)).toEqual(['fp32-native-attention', 'fp32-no-dnn']);
  expect(report.rows.map((row) => row.settings)).toEqual([
    { GGML_SYCL_ENABLE_GRAPH: '0', GGML_SYCL_ENABLE_DNN: '1', GGML_SYCL_FA_ONEDNN: '0' },
    { GGML_SYCL_ENABLE_GRAPH: '0', GGML_SYCL_ENABLE_DNN: '0', GGML_SYCL_FA_ONEDNN: '0' },
  ]);
  expect(report.rows.map((row) => row.content_hash_drift_count)).toEqual([10, 4]);
  expect(report.rows.map((row) => row.json_sse_matches)).toEqual([6, 8]);
  for (const row of report.rows) {
    expect(row.passed).toBe(false);
    expect(row.samples).toHaveLength(24);
    expect(row.streams).toHaveLength(8);
    const first = new Map<string, string>();
    let drift = 0;
    for (const sample of row.samples) {
      expect(sample.content_sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(sample.latency_ms).toBeGreaterThan(0);
      if (!first.has(sample.scenario)) first.set(sample.scenario, sample.content_sha256);
      else if (first.get(sample.scenario) !== sample.content_sha256) drift++;
    }
    expect(drift).toBe(row.content_hash_drift_count);
    expect(row.output_drift).toHaveLength(drift);
    expect(row.streams.filter((stream) => stream.matches_json)).toHaveLength(row.json_sse_matches);
    for (const lifetime of [0, 1])
      for (const scenario of ['core', 'context-medium', 'context-long', 'decode-extended']) {
        const samples = row.samples.filter(
          (sample) => sample.lifetime === lifetime && sample.scenario === scenario,
        );
        expect(samples.map((sample) => sample.repeat)).toEqual([0, 1, 2]);
        expect(samples.map((sample) => sample.warmup)).toEqual([true, false, false]);
      }
  }
});

test('neither a passing short continuation nor reduced drift qualifies a competitor baseline', () => {
  expect(report.settings).toEqual({
    kv_cache: 'f16',
    mtp: 'off',
    slots: 1,
    prompt_cache: false,
    ram_cache_bytes: 0,
    server_lifetimes_per_treatment: 2,
    workloads: 4,
    requests_per_workload_per_lifetime: 3,
    warmup_per_workload_per_lifetime: 1,
    measured_per_workload_per_lifetime: 2,
  });
  expect(report.reference_is_latest_claim).toBe(false);
  expect(report.release_qualification).toBe(false);
  expect(report.performance_qualified).toBe(false);
  expect(report.complete_model_matrix).toBe(false);
  expect(report.all_models_all_dimensions_better).toBe(false);
  expect(report.cleanup).toBe('no GPU process remains');
});
