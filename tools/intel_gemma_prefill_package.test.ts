import { expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';
const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read(
  '../site/public/zinc/validation/intel-gemma-prefill-package-support-2026-11-10.json',
);
const prototype = read('../site/public/zinc/validation/intel-gemma-prefill-tiling-2026-11-10.json');
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

test('the qualified Gemma archive adds only two shader assets and passes 808 numerical cases', () => {
  expect(report.archive).toEqual({
    archive_sha256: '59acd92fc2c5b3c3ba56c3301a36600f29eb04899d4fa515b17e02786462caa5',
    commit: 'a9673e01aac5f603754ebe394318e86d67f7176d',
    run_id: 42762262671,
    selected_head: 'b1cbee893a571d5d2d5d7c01d893ba9997c6f7ae',
    source_tree: '9935516c653ff63e335d9b26401f767f82e613ee',
    version: '0.1.0-dev.42762262671',
  });
  expect(report.source_report_sha256).toBe(
    'f3f7b7bbd79d042c8528923eb86935e3764eb3211f5ea4ed30d372810badf759',
  );
  expect(report.gemma_numerical_harness_commit).toBe(report.archive.commit);
  expect(Object.keys(report.shader_sha256)).toHaveLength(217);
  for (const [name, digest] of Object.entries(prototype.baseline_shader_sha256))
    expect(report.shader_sha256[name]).toBe(digest);
  expect(
    Object.keys(report.shader_sha256).filter((name) => !(name in prototype.baseline_shader_sha256)),
  ).toEqual([
    'share/zinc/shaders/mul_mm_q4k_gate_up_geglu_intel.spv',
    'share/zinc/shaders/mul_mm_q4k_gate_up_geglu_intel_full.spv',
  ]);
  expect(report.measured_kernel_bytes_identical).toBe(true);
  for (const suffix of ['', '_full']) {
    const measured =
      prototype.numerical_proof.shader_sha256[`share/zinc/shaders/gemma_geglu_tm2tn8${suffix}.spv`];
    expect(
      report.shader_sha256[`share/zinc/shaders/mul_mm_q4k_gate_up_geglu_intel${suffix}.spv`],
    ).toBe(measured);
    expect(report.measured_kernel_sha256[suffix || 'ragged']).toBe(measured);
  }
  expect(report.kernel_reference).toEqual(expect.objectContaining({ passed: true, cases: 376 }));
  expect(report.q8_decode_reference).toEqual(
    expect.objectContaining({ passed: true, cases: 228, bitwise_pairs: 114 }),
  );
  expect(report.gemma_geglu_reference).toEqual(
    expect.objectContaining({ passed: true, cases: 204, bitwise_pairs: 102 }),
  );
  expect(report.gemma_geglu_reference.maximum_absolute_error).toBeLessThan(0.001);
  for (const [name, digest] of Object.entries(report.gemma_geglu_reference.shader_sha256))
    expect(report.shader_sha256[`share/zinc/shaders/${name}.spv`]).toBe(digest);
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware_unchanged).toBe(true);
  expect(report.cleanup).toBe('no GPU process remains');
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('every managed Intel model passes complete CLI, API, JSON/SSE and state recovery checks', () => {
  expect(report.models.map((row) => row.id).sort()).toEqual(
    policy.models.map((row) => row.id).sort(),
  );
  expect(report.models.reduce((n, row) => n + row.correctness.cases.length, 0)).toBe(27);
  expect(report.models.reduce((n, row) => n + row.correctness.http_full_continuations, 0)).toBe(81);
  for (const row of report.models) {
    const pin = policy.models.find((model) => model.id === row.id);
    expect(row.passed).toBe(true);
    expect(row.tokenizer_parity).toBe(true);
    expect(row.sha256).toBe(pin.sha256);
    expect(row.context).toBe(pin.context);
    completeResponses(row);
    expect(row.zinc_processes_checked).toBe(row.correctness.cases.length + 2);
    expect(row.gemma_prefill_pipeline_loads).toEqual(
      row.id === 'gemma4-31b-q4k-m' ? { ragged: 8, full: 8 } : { ragged: 0, full: 0 },
    );
    expect(row.packed_prefill_pipeline_loads).toBe(row.id === 'qwen3-0.6b-q8-0' ? 5 : 0);
    if (pin.sliding_window) {
      expect(row.correctness.cases).toHaveLength(6);
      expect(
        row.correctness.cases.slice(3).every((c) => c.prompt_tokens > pin.sliding_window),
      ).toBe(true);
    }
  }
  expect(report.settings).toEqual({
    backend: 'vulkan',
    kv_cache: 'f16',
    mtp: 'off',
    prompt_cache: false,
    slots: 1,
  });
  expect(report.zinc_support_complete).toBe(true);
  expect(report.passed).toBe(true);
  expect(report.performance_qualified).toBe(false);
  expect(report.release_qualification).toBe(false);
  expect(report.all_models_all_dimensions_better).toBe(false);
});

test('opt-out and each missing Gemma asset preserve complete short and window-crossing answers', () => {
  expect(report.controls.map((row) => row.id)).toEqual([
    'generic-opt-out',
    'missing-ragged-asset',
    'missing-full-asset',
  ]);
  for (const [index, row] of report.controls.entries()) {
    expect(row.passed).toBe(true);
    expect(row.binary_sha256).toBe(report.binary_sha256);
    const missing =
      index === 0
        ? null
        : `share/zinc/shaders/mul_mm_q4k_gate_up_geglu_intel${index === 2 ? '_full' : ''}.spv`;
    expect(row.missing_asset).toBe(missing);
    expect(Object.keys(row.shader_sha256)).toHaveLength(missing ? 216 : 217);
    for (const [name, digest] of Object.entries(report.shader_sha256)) {
      if (name === missing) expect(row.shader_sha256[name]).toBeUndefined();
      else expect(row.shader_sha256[name]).toBe(digest);
    }
    expect(row.cli_cases).toHaveLength(2);
    expect(row.cli_cases[0].prompt_tokens).toBeLessThan(1024);
    expect(row.cli_cases[1].prompt_tokens).toBeGreaterThan(1024);
    for (const sample of row.cli_cases) {
      expect(sample.generated_ids).toHaveLength(16);
      expect(sample.complete_output_identical).toBe(true);
    }
    expect(row.http_full_continuations).toBe(6);
    expect(row.http_after_cancellation).toBe(true);
    expect(row.http_after_context_error).toBe(true);
    expect(row.raw_streaming.passed).toBe(true);
    expect(row.raw_streaming.cases).toHaveLength(2);
    expect(row.raw_streaming.complete_native_replay_after_cancellation).toBe(true);
    expect(row.zinc_processes_checked).toBe(4);
  }
});
