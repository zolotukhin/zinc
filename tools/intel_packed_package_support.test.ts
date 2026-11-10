import { expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read(
  '../site/public/zinc/validation/intel-native-sycl-fp32-packed-q8-2026-11-10.json',
);
const previous = read(
  '../site/public/zinc/validation/intel-native-sycl-fp32-baseline-2026-11-09.json',
);
const policy = read('../config/release/targets.json');
const hash = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');

test('new Intel support evidence binds the actual CI archive and its single shader change', () => {
  expect(report.zinc.commit).toBe('cc1a01dcd55c992cd62f109ebefd3b72686dd13a');
  expect(report.zinc.source_tree).toBe('b35a9f10a0f46a82be0fec385993ffa390e32265');
  expect(report.zinc.archive_sha256).toBe(
    '07785ffdf160f36a27e889dbbcd4802ba19a5e6e00bfd2be60807d8be8dd31a6',
  );
  expect(report.zinc.q8_shader_source_sha256).toBe(
    hash(readFileSync(new URL('../src/shaders/dmmv_q8_0_intel_rows.comp', import.meta.url))),
  );
  const names = Object.keys(report.zinc.shader_sha256);
  expect(names).toHaveLength(214);
  expect(
    names.filter((name) => report.zinc.shader_sha256[name] !== previous.zinc.shader_sha256[name]),
  ).toEqual(['share/zinc/shaders/dmmv_q8_0_intel_rows.spv']);
  expect(report.q8_decode_reference).toEqual(
    expect.objectContaining({ passed: true, cases: 228, bitwise_pairs: 114 }),
  );
  expect(report.q8_decode_reference.shader_sha256.dmmv_q8_0_intel_rows).toBe(
    report.zinc.shader_sha256['share/zinc/shaders/dmmv_q8_0_intel_rows.spv'],
  );
  expect(report.kernel_reference).toEqual({
    passed: true,
    cases: 376,
    suites: { attention_window: 256, gemma_router: 36, shared_cache: 84 },
  });
  expect(report.harness_commit).toBe('c73ed59457c1943f5cbbf148515968056a194104');
  expect(report.source_report_sha256).toBe(
    'ef6351889b2dc09febdf96440b612231595470c4a08d92609c96fee46afc0135',
  );
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('the complete seven-model catalog retains full native answers, streams and recovery', () => {
  expect(report.zinc_support_complete).toBe(true);
  expect(report.models.map((model) => model.id).sort()).toEqual(
    policy.models.map((model) => model.id).sort(),
  );
  expect(report.selected_models).toEqual(report.required_models);
  let cases = 0,
    httpContinuations = 0;
  for (const model of report.models) {
    expect(model.sha256).toBe(policy.models.find((pin) => pin.id === model.id).sha256);
    expect(model.correctness.passed).toBe(true);
    expect(model.correctness.http_full_continuations).toBe(model.correctness.cases.length * 3);
    httpContinuations += model.correctness.http_full_continuations;
    expect(model.correctness.http_after_cancellation).toBe(true);
    expect(model.correctness.http_after_context_error).toBe(true);
    expect(model.raw_streaming.passed).toBe(true);
    expect(model.raw_streaming.complete_native_replay_after_cancellation).toBe(true);
    expect(model.raw_streaming.cases).toHaveLength(model.correctness.cases.length);
    for (const [index, reference] of model.correctness.cases.entries()) {
      expect(reference.reference_ids).toHaveLength(16);
      expect(reference.zinc_ids).toHaveLength(16);
      const accepted = [hash(reference.reference_text)];
      if (reference.validated_text != null) {
        expect(reference.comparison.near_ties.length).toBeGreaterThan(0);
        accepted.push(hash(reference.validated_text));
      }
      expect(accepted).toContain(model.raw_streaming.cases[index].content_sha256);
      cases++;
    }
    expect(model.native_reference_validation.passed).toBe(true);
  }
  expect(cases).toBe(27);
  expect(httpContinuations).toBe(81);
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware_unchanged).toBe(true);
  expect(report.native_stack_after).toEqual(report.native_stack);
  expect(report.native_build_after).toEqual(report.native_build);
  expect(report.native_probe_after).toEqual(report.native_probe);
  expect(report.native_opencl_after).toEqual(report.native_opencl);
  expect(report.cleanup).toBe('no GPU process remains');
});

test('fresh native failures keep their rejected samples and final process observations', () => {
  expect(report.reference.commit).toBe('cf6463ace735a570d5734da4957d54b720cac6f8');
  expect(report.latest_reference_check).toEqual(
    expect.objectContaining({
      commit: report.reference.commit,
      observed_head: report.reference.commit,
      checked_at: '2026-11-10T00:13:13.834Z',
      scope: 'campaign-before-build',
    }),
  );
  expect(report.settings.native_runtime).toEqual({
    GGML_SYCL_ENABLE_GRAPH: '0',
    GGML_SYCL_ENABLE_DNN: '1',
    GGML_SYCL_FA_ONEDNN: '1',
  });
  const failures = report.models.filter(
    (model) =>
      model.error?.startsWith('llama ') && model.error.includes('repeated complete outputs differ'),
  );
  expect(failures).toHaveLength(7);
  for (const model of failures) {
    expect(model.passed).toBe(false);
    const rejected = model.samples.at(-1);
    expect(rejected.engine).toBe('llama');
    const same = model.samples.filter(
      (sample) => sample.engine === 'llama' && sample.scenario === rejected.scenario,
    );
    expect(new Set(same.map((sample) => sample.content_sha256)).size).toBeGreaterThan(1);
    for (const process of model.processes.filter((process) => process.engine === 'llama')) {
      expect(process.native_runtime?.runtime).toBe('level_zero');
      expect(process.memory?.observed_peak.host_peak_rss_bytes).toBeGreaterThan(0);
      expect(process.memory_error).toBeUndefined();
      expect(process.native_runtime_error).toBeUndefined();
    }
  }
  expect(report.release_qualification).toBe(false);
  expect(report.all_models_all_dimensions_better).toBe(false);
  expect(report.complete_performance_matrix).toBe(false);
  expect(report.text_previews_omitted).toBe(true);
});
