import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const evidence = JSON.parse(
  readFileSync(
    new URL(
      '../site/public/zinc/validation/intel-native-runtime-controls-2026-11-09.json',
      import.meta.url,
    ),
    'utf8',
  ),
);

test('native runtime diagnostics retain all precisions, controls and complete GPU requests', () => {
  expect(evidence.reference_commit).toBe('5f490d7bfe78844e6390c869454c149fd7e003a8');
  expect(evidence.rows.map((row) => row.id)).toEqual([
    'fp32-default',
    'fp32-native-attention',
    'fp32-no-dnn',
    'fp16-default',
    'fp16-native-attention',
    'fp16-no-dnn',
  ]);
  for (const row of evidence.rows) {
    expect(row.requests).toBe(24);
    expect(row.samples).toHaveLength(24);
    expect(row.streams).toHaveLength(8);
    expect(row.processes).toHaveLength(2);
    expect(row.build.settings.GGML_SYCL_F16).toBe(row.id.startsWith('fp16') ? 'ON' : 'OFF');
    expect(row.settings.GGML_SYCL_ENABLE_GRAPH).toBe('0');
    expect(row.settings.GGML_SYCL_ENABLE_DNN).toBe(row.id.endsWith('no-dnn') ? '0' : '1');
    expect(row.settings.GGML_SYCL_FA_ONEDNN).toBe(row.id.endsWith('default') ? '1' : '0');
    expect(row.samples.filter((sample) => sample.warmup)).toHaveLength(8);
    expect(new Set(row.samples.map((sample) => sample.scenario)).size).toBe(4);
    for (const sample of row.samples) {
      expect(sample.usage.prompt_tokens_details.cached_tokens).toBe(0);
      expect(sample.usage.completion_tokens).toBeGreaterThan(0);
      expect(sample.prefill_tps).toBeGreaterThan(0);
      expect(sample.decode_tps).toBeGreaterThan(0);
    }
    for (const process of row.processes) {
      expect(process.gpu_offload.offloaded).toBe(process.gpu_offload.total);
      expect(process.gpu_offload.offloaded).toBeGreaterThan(0);
      expect(process.gpu_offload.backend).toBe('sycl');
      expect(process.native_runtime.runtime).toBe('level_zero');
      const libraries = Object.keys(process.native_runtime.library_sha256);
      expect(libraries.some((name) => name.startsWith('libsycl.so'))).toBe(true);
      expect(libraries.some((name) => name.startsWith('libur_adapter_level_zero'))).toBe(true);
    }
  }
  expect(() => assertPublicArtifact(evidence)).not.toThrow();
});

test('native repeatability verdicts derive from full responses across both server lifetimes', () => {
  for (const row of evidence.rows) {
    // One first-seen identity per scenario spans BOTH lifetimes, including
    // warmups. Resetting at restart would hide four FP16 changes (9 -> 5).
    const first = new Map();
    const expectedDrift = [];
    let differences = 0;
    for (const sample of row.samples) {
      const identity = JSON.stringify([
        sample.content_sha256,
        sample.usage.prompt_tokens,
        sample.usage.completion_tokens,
        sample.usage.total_tokens,
        sample.usage.prompt_tokens_details.cached_tokens,
        sample.finish_reason,
      ]);
      if (!first.has(sample.scenario)) first.set(sample.scenario, identity);
      else if (first.get(sample.scenario) !== identity) {
        differences++;
        expectedDrift.push([sample.lifetime, sample.scenario, sample.repeat]);
      }
    }
    expect(row.output_drift).toHaveLength(differences);
    expect(row.drift_count).toBe(differences);
    expect(
      row.output_drift.map((sample) => [sample.lifetime, sample.scenario, sample.repeat]),
    ).toEqual(expectedDrift);
    expect(row.passed).toBe(differences === 0 && row.streams.every((probe) => probe.matches_json));
  }
  expect(evidence.rows.filter((row) => row.passed).map((row) => row.id)).toEqual([
    'fp32-default',
    'fp32-no-dnn',
    'fp16-native-attention',
    'fp16-no-dnn',
  ]);
  expect(evidence.rows.find((row) => row.id === 'fp32-native-attention').drift_count).toBe(2);
  expect(evidence.rows.find((row) => row.id === 'fp16-default').drift_count).toBe(9);
});

test('starter controls preserve the separate all-model and release gates', () => {
  expect(evidence.model).toBe('qwen3-0.6b-q8-0');
  expect(evidence.complete_model_matrix).toBe(false);
  expect(evidence.release_qualification).toBe(false);
  expect(evidence.all_models_all_dimensions_better).toBe(false);
  expect(evidence.cleanup).toBe('no GPU process remains');
  expect(evidence.limitations.join(' ')).toContain('every catalog model');
});
