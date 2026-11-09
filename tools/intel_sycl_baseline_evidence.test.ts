import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read(
  '../site/public/zinc/validation/intel-native-sycl-fp32-baseline-2026-11-09.json',
);
const policy = read('../config/release/targets.json');

test('the final CI package retains all seven native, streaming and recovery support checks', () => {
  const required = policy.gpu_profiles.find((profile) => profile.id === 'vulkan-intel-xe2').models;
  expect(report.models.map((model) => model.id).sort()).toEqual([...required].sort());
  expect(new Set(report.models.map((model) => model.id)).size).toBe(7);
  expect(report.zinc.commit).toBe('2d0e1369d361671edc5ddf2a676efb12625bbb66');
  expect(report.zinc.sha256).toBe(
    '3bea62b84a25665bf8fe899d838ed92ef3460b5e1c446178ac8a753c29c3778a',
  );
  expect(report.zinc.archive_sha256).toBe(
    '0bebe4850b999110c4811769b37927f0731a0b7a4a6befd1164926b76a2ce52e',
  );
  expect(Object.keys(report.zinc.shader_sha256)).toHaveLength(214);
  for (const [name, digest] of Object.entries(report.q8_decode_reference.shader_sha256)) {
    expect(report.zinc.shader_sha256[`share/zinc/shaders/${name}.spv`]).toBe(digest);
  }
  expect(report.kernel_reference).toEqual({
    passed: true,
    cases: 376,
    suites: { attention_window: 256, gemma_router: 36, shared_cache: 84 },
  });
  expect(report.q8_decode_reference.passed).toBe(true);
  expect(report.q8_decode_reference.cases).toBe(228);
  expect(report.q8_decode_reference.bitwise_pairs).toBe(114);
  let cases = 0,
    exact = 0,
    ties = 0,
    replays = 0;
  for (const model of report.models) {
    const pin = policy.models.find((entry) => entry.id === model.id);
    const count = pin.sliding_window ? 6 : 3;
    expect(model.sha256).toBe(pin.sha256);
    expect(model.context).toBe(pin.context);
    expect(model.correctness.passed).toBe(true);
    expect(model.correctness.cases).toHaveLength(count);
    expect(model.raw_streaming.passed).toBe(true);
    expect(model.raw_streaming.cases).toHaveLength(count);
    expect(model.raw_streaming.complete_native_replay_after_cancellation).toBe(true);
    expect(model.correctness.http_after_cancellation).toBe(true);
    expect(model.correctness.http_after_context_error).toBe(true);
    expect(model.correctness.http_full_continuations).toBe(count * 3);
    for (const [index, current] of model.correctness.cases.entries()) {
      expect(current.reference_ids).toHaveLength(16);
      expect(current.zinc_ids).toHaveLength(16);
      if (pin.sliding_window && index >= 3)
        expect(current.prompt_tokens).toBeGreaterThan(pin.sliding_window);
      expect(model.raw_streaming.cases[index].usage.completion_tokens).toBe(16);
    }
    cases += count;
    exact += model.correctness.reference_agreement.exact_tokens;
    ties += model.correctness.reference_agreement.near_tie_tokens;
    replays += model.correctness.http_full_continuations;
  }
  expect({ cases, exact, ties, replays }).toEqual({ cases: 27, exact: 431, ties: 1, replays: 81 });
  expect(report.zinc_support_complete).toBe(true);
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware_unchanged).toBe(true);
  expect(report.native_stack_after).toEqual(report.native_stack);
  expect(report.cleanup).toBe('no GPU process remains');
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('the official native baseline preserves rejected outputs and cannot claim an all-model win', () => {
  expect(report.reference.commit).toBe('0a94a3f289687883617b9a2abbc7e0ade2d6b6c4');
  expect(report.harness_commit).toBe('4043accee57a391aede645f53430159905add880');
  expect(report.latest_reference_check.scope).toBe('campaign-before-build');
  expect(report.latest_reference_check.observed_head).toBe(report.reference.commit);
  expect(report.settings.native_runtime).toEqual({
    GGML_SYCL_ENABLE_GRAPH: '0',
    GGML_SYCL_ENABLE_DNN: '1',
    GGML_SYCL_FA_ONEDNN: '1',
  });
  expect(report.native_build.settings.GGML_SYCL_F16).toBe('OFF');
  expect(report.native_build.settings.GGML_SYCL_HOST_MEM_FALLBACK).toBe('OFF');
  expect(report.settings.native_stack_verified).toBe(true);
  expect(report.settings).toEqual(
    expect.objectContaining({
      backend: 'vulkan',
      reference_backend: 'sycl',
      protocol: 'raw',
      kv_cache: 'f16',
      mtp: 'off',
      slots: 1,
      prompt_cache: false,
      rounds: 3,
      runs: 3,
      warmup: 1,
      temperature: 0,
      top_p: 1,
      seed: 0,
    }),
  );
  expect(report.passed).toBe(false);
  expect(report.complete_model_matrix).toBe(false);
  expect(report.complete_performance_matrix).toBe(false);
  expect(report.all_models_all_dimensions_better).toBe(false);
  expect(report.release_qualification).toBe(false);
  expect(report.models.filter((model) => model.passed).map((model) => model.id)).toEqual([
    'qwen3-0.6b-q8-0',
  ]);
  for (const model of report.models) {
    expect(model.native_reference_validation.passed).toBe(true);
    for (const process of model.processes.filter((entry) => entry.engine === 'llama')) {
      expect(process.gpu_offload.backend).toBe('sycl');
      expect(process.gpu_offload.offloaded).toBe(process.gpu_offload.total);
      expect(process.gpu_offload.offloaded).toBeGreaterThan(0);
    }
    if (model.passed) continue;
    expect(model.error).toMatch(/^llama .+ repeated complete outputs differ$/);
    const identities = new Map<string, string>();
    let drift = false;
    for (const sample of model.samples.filter((entry) => entry.engine === 'llama')) {
      if (!identities.has(sample.scenario)) identities.set(sample.scenario, sample.content_sha256);
      else if (identities.get(sample.scenario) !== sample.content_sha256) drift = true;
    }
    expect(drift).toBe(true);
    expect(model.samples.at(-1).engine).toBe('llama');
    // Earlier failed timed processes lack their final observations. Keep that
    // gap visible instead of borrowing evidence from another server lifetime.
    expect(
      model.processes.some((process) => process.engine === 'llama' && !process.native_runtime),
    ).toBe(true);
  }
});
