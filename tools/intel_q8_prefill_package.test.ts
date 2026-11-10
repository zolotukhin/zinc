import { expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read(
  '../site/public/zinc/validation/intel-q8-prefill-package-support-2026-11-10.json',
);
const previous = read(
  '../site/public/zinc/validation/intel-native-sycl-fp32-packed-q8-2026-11-10.json',
);
const policy = read('../config/release/targets.json');
const hash = (value: string | Buffer) => createHash('sha256').update(value).digest('hex');

function completeResponses(row) {
  expect(row.correctness.passed).toBe(true);
  expect(row.correctness.http_after_cancellation).toBe(true);
  expect(row.correctness.http_after_context_error).toBe(true);
  expect(row.correctness.http_full_continuations).toBe(row.correctness.cases.length * 3);
  expect(row.raw_streaming.passed).toBe(true);
  expect(row.raw_streaming.complete_native_replay_after_cancellation).toBe(true);
  expect(row.raw_streaming.cases).toHaveLength(row.correctness.cases.length);
  for (const [index, reference] of row.correctness.cases.entries()) {
    expect(reference.reference_ids).toHaveLength(16);
    expect(reference.zinc_ids).toHaveLength(16);
    const accepted = [hash(reference.reference_text)];
    if (reference.validated_text != null) {
      expect(reference.comparison.near_ties.length).toBeGreaterThan(0);
      accepted.push(hash(reference.validated_text));
    }
    expect(accepted).toContain(row.raw_streaming.cases[index].content_sha256);
  }
}

test('production prefill proof binds the exact CI archive and immutable source/shader identities', () => {
  expect(report.passed).toBe(true);
  expect(report.archive).toEqual({
    version: '0.1.0-dev.42739579552',
    commit: 'ac891db07707d12daa2fe1103078da90cfcaa644',
    archive_sha256: '3e16bc5ab6ce3ea6cc080fb8febba210e930b3bb95926330260ba7cb8a0faa34',
    source_tree: '0c1197c3f93d8977c039159a07e71865ce958140',
    run_id: 42739579552,
    selected_head: '9a4c0e81694f6339ed349fefd68228cf14a4cb7d',
  });
  expect(report.source_report_sha256).toBe(
    '2ae32fb12e5b4fbb24bcc2be376779f3e9aa0db168678f96d0ec31956ff76afd',
  );
  expect(Object.keys(report.shader_sha256)).toHaveLength(215);
  for (const [name, digest] of Object.entries(previous.zinc.shader_sha256)) {
    expect(report.shader_sha256[name]).toBe(digest);
  }
  expect(
    Object.keys(report.shader_sha256).filter((name) => !(name in previous.zinc.shader_sha256)),
  ).toEqual(['share/zinc/shaders/mul_mm_q8_0_intel.spv']);
  // This inventory belongs to the recorded archive. Host policy edits have
  // CPU regressions; the current kernel stays bound to its numerical proof.
  expect(report.production_source_sha256).toEqual({
    'build.zig': '1ad726527f93a0c8983923e4ed48d2abddf0c68817b10bfe1ef7afaa30b03a07',
    'src/compute/dmmv.zig': '5d273fe12a03e88110aa2269002e341f0f273a4852a7c82517c242c7e01418c6',
    'src/compute/forward.zig': '04108ecc93ee370632b2ef873a5722925fc38bafc1c39bc71a7f8ed3930a0d7e',
    'src/gpu/prefill_policy.zig':
      'bbdb9ab58174ffca718421d9bf757edb3453dda34a808dd3b061c683bf68cfa8',
    'src/shaders/mul_mm_q8_0_intel.comp':
      'cc50ff8e9be46b449e26876401681aebd3cd1b9d9cb82faee4f78753cb012eea',
    'tools/validate_q8_prefill.py':
      'aadbeed14a90882e2daf102259387b0a964465be0b58fb2f11b99f513c190449',
  });
  expect(
    hash(readFileSync(new URL('../src/shaders/mul_mm_q8_0_intel.comp', import.meta.url))),
  ).toBe(report.production_source_sha256['src/shaders/mul_mm_q8_0_intel.comp']);
  expect(report.kernel_reference.cases).toBe(376);
  expect(report.kernel_reference.passed).toBe(true);
  expect(report.q8_decode_reference).toEqual(
    expect.objectContaining({ passed: true, cases: 228, bitwise_pairs: 114 }),
  );
  expect(report.numerical_harness_commit).toBe(report.archive.commit);
  expect(report.numerical_prefill.passed).toBe(true);
  expect(report.numerical_prefill.binary_sha256).toBe(report.binary_sha256);
  expect(report.numerical_prefill.kv_cache).toBe('f32');
  expect(report.numerical_prefill.selected_q8_gemm_shader).toBe('mul_mm_q8_0_intel.spv');
  expect(report.numerical_prefill.selected_q8_gemm_shader_sha256).toBe(
    report.shader_sha256['share/zinc/shaders/mul_mm_q8_0_intel.spv'],
  );
  expect(report.numerical_prefill.cases.map((row) => row.prompt_tokens)).toEqual([
    1, 2, 15, 16, 31, 32, 33, 95, 96, 97, 257, 383, 384, 385,
  ]);
  expect(report.numerical_default_pipeline_loads).toBe(14);
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('the entire catalog passes complete native/API/SSE checks and only the starter selects packed prefill', () => {
  expect(report.models.map((row) => row.id).sort()).toEqual(
    policy.models.map((row) => row.id).sort(),
  );
  expect(report.required_models).toEqual(report.models.map((row) => row.id));
  let cases = 0,
    http = 0;
  for (const row of report.models) {
    const pin = policy.models.find((model) => model.id === row.id);
    expect(row.passed && row.tokenizer_parity).toBe(true);
    expect(row.sha256).toBe(pin.sha256);
    expect(row.context).toBe(pin.context);
    completeResponses(row);
    cases += row.correctness.cases.length;
    http += row.correctness.http_full_continuations;
    expect(row.zinc_processes_checked).toBe(row.correctness.cases.length + 2);
    expect(row.packed_prefill_pipeline_loads).toBe(
      row.id === 'qwen3-0.6b-q8-0' ? row.zinc_processes_checked : 0,
    );
  }
  expect(cases).toBe(27);
  expect(http).toBe(81);
});

test('opt-out and one-missing-asset fallback preserve complete continuations and recovery', () => {
  expect(report.controls.map((row) => row.id)).toEqual([
    'generic-opt-out',
    'missing-asset-fallback',
  ]);
  expect(report.controls.map((row) => row.settings)).toEqual([{ ZINC_INTEL_Q8_PREFILL: '0' }, {}]);
  for (const row of report.controls) {
    expect(row.passed).toBe(true);
    expect(row.binary_sha256).toBe(report.binary_sha256);
    const expected = Object.fromEntries(
      Object.entries(report.shader_sha256).filter(
        ([name]) =>
          row.id !== 'missing-asset-fallback' ||
          name !== 'share/zinc/shaders/mul_mm_q8_0_intel.spv',
      ),
    );
    expect(row.shader_sha256).toEqual(expected);
    completeResponses(row);
    expect(row.correctness.cases).toHaveLength(3);
    expect(row.zinc_processes_checked).toBe(5);
  }
});

test('package support keeps hardware/oracle identities and makes no release or competitor qualification claim', () => {
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware_unchanged).toBe(true);
  expect(report.oracle.commit).toBe(policy.reference.commit);
  expect(report.settings).toEqual({
    backend: 'vulkan',
    kv_cache: 'f16',
    mtp: 'off',
    prompt_cache: false,
    slots: 1,
  });
  expect(report.release_qualification).toBe(false);
  expect(report.performance_qualified).toBe(false);
  expect(report.all_models_all_dimensions_better).toBe(false);
  expect(report.cleanup).toBe('no GPU process remains');
  expect(report.error_type).toBeUndefined();
});
