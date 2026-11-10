import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read('../site/public/zinc/validation/intel-native-exact-cadence-2026-11-10.json');
const previous = read(
  '../site/public/zinc/validation/intel-native-sycl-fp32-packed-q8-2026-11-10.json',
);
const policy = read('../config/release/targets.json');
const identity = (sample) => ({
  content_sha256: sample.content_sha256,
  promptTokens: sample.promptTokens,
  generatedTokens: sample.generatedTokens,
  finish_reason: sample.finish_reason,
});

test('the native replay retains its actual pre-build selection, environment and unchanged GPU stack', () => {
  expect(report.source_report_sha256).toBe(
    '9325efa600ee0ac27cf00ed62d9d92988c9bf918cbd308f6c45a93e12c650a39',
  );
  expect(report.reference.commit).toBe('3b4d6ab3f79651935fba1dd93732f29022a7f5f5');
  expect(report.reference_selection.commit).toBe(report.reference.commit);
  const times = [
    report.reference_selection.checked_at_utc,
    report.build_started_at_utc,
    report.build_completed_at_utc,
    report.started_at_utc,
    report.completed_at_utc,
  ].map(Date.parse);
  expect(times.every(Number.isFinite)).toBe(true);
  expect(times).toEqual([...times].sort((a, b) => a - b));
  expect(report.native_build.settings).toEqual(previous.native_build.settings);
  expect(report.native_build.settings.GGML_SYCL_F16).toBe('OFF');
  expect(report.native_build.settings.GGML_SYCL_HOST_MEM_FALLBACK).toBe('OFF');
  expect(report.native_stack).toEqual(previous.native_stack);
  expect(report.native_stack.packages).toHaveLength(41);
  expect(report.hardware).toEqual(previous.hardware_after);
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware_unchanged).toBe(true);
  expect(report.environment_source_commit).toBe('c1c3bd5f4427137676051f1cfb19e3ff20aaf4fe');
  expect(report.environment_source_sha256).toBe(
    'bc8524c10c43a6d69f1ea95d0c7dc78f870d9100c731c47a9c510acc9957713e',
  );
  expect(report.inherited_onednn_keys).toEqual([]);
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('all 64 full responses and 16 short streams retain the original native cadence and model contexts', () => {
  expect(report.models.map((m) => m.id)).toEqual(['qwen3-0.6b-q8-0', 'qwen36-35b-a3b-q4k-xl']);
  expect(report.settings).toEqual({
    reference_backend: 'sycl',
    runtime: 'level_zero',
    kv_cache: 'f16',
    mtp: 'off',
    prompt_cache: false,
    ram_cache_bytes: 0,
    server_lifetimes_per_model: 2,
    workloads: 4,
    warmup_per_workload_per_lifetime: 1,
    measured_per_workload_per_lifetime: 3,
    extra_json_control_before_sse: false,
    startup_poll_interval_ms: 10,
    native_runtime: {
      GGML_SYCL_ENABLE_GRAPH: '0',
      GGML_SYCL_ENABLE_DNN: '1',
      GGML_SYCL_FA_ONEDNN: '1',
    },
  });
  for (const model of report.models) {
    const pin = policy.models.find((m) => m.id === model.id);
    expect(model.context).toBe(pin.context);
    expect(model.model_sha256).toBe(pin.sha256);
    expect(model.native_reference_validation.passed).toBe(true);
    expect(model.native_reference_validation.performance_configuration_checked).toBe(true);
    expect(model.native_reference_validation.cases).toHaveLength(3);
    expect(model.samples).toHaveLength(32);
    expect(model.streams).toHaveLength(8);
    expect(model.processes).toHaveLength(2);
    for (const scenario of ['core', 'context-medium', 'context-long', 'decode-extended']) {
      for (const lifetime of [0, 1]) {
        const rows = model.samples.filter(
          (s) => s.scenario === scenario && s.lifetime === lifetime,
        );
        expect(rows.map((s) => s.repeat)).toEqual([0, 1, 2, 3]);
        expect(rows.map((s) => s.warmup)).toEqual([true, false, false, false]);
        expect(
          model.streams.filter((s) => s.scenario === scenario && s.lifetime === lifetime),
        ).toHaveLength(1);
        for (const sample of rows) {
          expect(sample.content_sha256).toMatch(/^[a-f0-9]{64}$/);
          expect(sample.usage.prompt_tokens).toBe(sample.promptTokens);
          expect(sample.usage.completion_tokens).toBe(sample.generatedTokens);
          expect(sample.usage.prompt_tokens_details.cached_tokens).toBe(0);
          expect(sample.server_timings.cache_n).toBe(0);
          expect(sample.server_timings.prompt_n).toBe(sample.promptTokens);
          expect(sample.server_timings.predicted_n).toBe(sample.generatedTokens);
          expect(sample.server_timings.prompt_ms).toBeGreaterThan(0);
          expect(sample.server_timings.predicted_ms).toBeGreaterThan(0);
        }
      }
    }
    for (const process of model.processes) {
      expect(process.gpu_offload.backend).toBe('sycl');
      expect(process.gpu_offload.offloaded).toBe(process.gpu_offload.total);
      expect(process.gpu_offload.total).toBeGreaterThan(0);
      expect(process.native_runtime.runtime).toBe('level_zero');
      expect(process.native_runtime_error).toBeUndefined();
      const libraries = Object.keys(process.native_runtime.library_sha256);
      expect(libraries.some((n) => n.startsWith('libsycl.so'))).toBe(true);
      expect(libraries.some((n) => n.startsWith('libur_adapter_level_zero'))).toBe(true);
    }
  }
});

test('drift is reconstructed from every response, including warmups and server restarts', () => {
  for (const model of report.models) {
    const first = new Map();
    const drift = [];
    for (const sample of model.samples) {
      const actual = identity(sample);
      if (!first.has(sample.scenario)) first.set(sample.scenario, actual);
      const expected = first.get(sample.scenario);
      if (JSON.stringify(actual) !== JSON.stringify(expected))
        drift.push({
          lifetime: sample.lifetime,
          scenario: sample.scenario,
          repeat: sample.repeat,
          expected,
          actual,
        });
    }
    expect(model.output_drift).toEqual(drift);
    const streams = new Map();
    const streamDrift = [];
    for (const sample of model.streams) {
      if (!streams.has(sample.scenario)) streams.set(sample.scenario, sample.content_sha256);
      if (streams.get(sample.scenario) !== sample.content_sha256)
        streamDrift.push({
          lifetime: sample.lifetime,
          scenario: sample.scenario,
          expected: streams.get(sample.scenario),
          actual: sample.content_sha256,
        });
    }
    expect(model.stream_drift).toEqual(streamDrift);
    expect(model.passed).toBe(drift.length === 0 && streamDrift.length === 0);
  }
  expect(
    report.models.map((m) => [m.passed, m.output_drift.length, m.stream_drift.length]),
  ).toEqual([
    [true, 0, 0],
    [false, 14, 1],
  ]);
});

test('a limited passing replay cannot erase prior failures or qualify native ZINC and all-model performance', () => {
  expect(previous.models[0].passed).toBe(false);
  expect(report.prior_starter_control_differences).toEqual({
    context: 4096,
    extra_json_control_before_sse: true,
    requests_per_workload_per_lifetime: 3,
  });
  expect(report.completed).toBe(true);
  for (const field of [
    'passed',
    'release_qualification',
    'performance_qualified',
    'complete_model_matrix',
    'all_models_all_dimensions_better',
    'native_zinc_backend',
    'reference_is_latest_claim',
  ]) {
    expect(report[field]).toBe(false);
  }
  expect(report.official_driver_check.matches_installed).toBe(true);
  expect(report.official_driver_check.compute_runtime.tag).toBe('26.40.40196.8');
  expect(report.official_driver_check.igc.tag).toBe('v2.42.6');
  expect(report.cleanup).toBe('no GPU process remains');
});
