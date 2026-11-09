import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const evidence = read('../site/public/zinc/validation/intel-seven-model-support-2026-11-09.json');
const policy = read('../config/release/targets.json');

test('the updated Intel support record covers every qualified model and immutable package', () => {
  const required = policy.gpu_profiles.find((row) => row.id === 'vulkan-intel-xe2').models;
  expect([...evidence.required_models].sort()).toEqual([...required].sort());
  expect(evidence.models.map((row) => row.id).sort()).toEqual([...required].sort());
  expect(new Set(evidence.models.map((row) => row.id)).size).toBe(required.length);
  for (const model of evidence.models) {
    const pin = policy.models.find((row) => row.id === model.id);
    expect(model.sha256).toBe(pin.sha256);
    expect(model.size_bytes).toBe(pin.size_bytes);
    expect(model.context).toBe(pin.context);
    expect(model.passed).toBe(true);
  }
  expect(evidence.zinc.commit).toBe('56bec2287ed7c2bdd5e2b67bac0d12bfff79768a');
  expect(evidence.zinc.sha256).toBe(
    'd43ef924c170c8b9df5ed4cebad21a83d343b1d05065351518927a12c27cd7c7',
  );
  expect(evidence.zinc.archive_sha256).toBe(
    '4e7c12b807cd7d01d89aeb33edb00cc5e4dec7279cdda8acb9a80fe105e3ec9b',
  );
  expect(Object.keys(evidence.zinc.shader_sha256)).toHaveLength(214);
  for (const digest of Object.values(evidence.zinc.shader_sha256))
    expect(digest).toMatch(/^[a-f0-9]{64}$/);
  expect(evidence.hardware_unchanged).toBe(true);
  expect(evidence.cleanup).toBe('no GPU process remains');
  expect(() => assertPublicArtifact(evidence)).not.toThrow();
});

test('every Intel model retains complete native, JSON, streaming and recovery controls', () => {
  let cases = 0,
    exact = 0,
    ties = 0,
    replays = 0;
  for (const model of evidence.models) {
    const count = model.sliding_window ? 6 : 3;
    expect(model.tokenizer_cases).toBeGreaterThan(0);
    expect(model.native_cases).toHaveLength(count);
    expect(model.raw_streaming.cases).toHaveLength(count);
    expect(model.full_http_continuations).toBe(count * 3);
    expect(model.http_after_cancellation).toBe(true);
    expect(model.http_after_context_error).toBe(true);
    expect(model.raw_streaming.complete_native_replay_after_cancellation).toBe(true);
    for (const [index, current] of model.native_cases.entries()) {
      expect(current.reference_ids).toHaveLength(16);
      expect(current.zinc_ids).toHaveLength(16);
      expect(current.comparison.tokens).toBe(16);
      if (model.sliding_window && index >= 3) {
        expect(current.prompt_tokens).toBeGreaterThan(model.sliding_window);
        expect(current.prompt_tokens + 16).toBeLessThanOrEqual(model.context);
      }
      const stream = model.raw_streaming.cases[index];
      expect([current.reference_text_sha256, current.validated_text_sha256]).toContain(
        stream.content_sha256,
      );
      expect(stream.usage.prompt_tokens).toBe(current.prompt_tokens);
      expect(stream.usage.completion_tokens).toBe(16);
      expect(stream.usage.total_tokens).toBe(current.prompt_tokens + 16);
    }
    cases += count;
    exact += model.reference_agreement.exact_tokens;
    ties += model.reference_agreement.near_tie_tokens;
    replays += model.full_http_continuations;
  }
  expect({ cases, exact, ties, replays }).toEqual({ cases: 27, exact: 431, ties: 1, replays: 81 });
  const moe = evidence.models.find((row) => row.id === 'qwen36-35b-a3b-q4k-xl');
  expect(moe.prefill_cancellation.passed).toBe(true);
});

test('failed native FP16 controls cannot become a baseline or all-dimensions win', () => {
  expect(evidence.passed).toBe(true);
  expect(evidence.complete_model_matrix).toBe(true);
  expect(evidence.release_qualification).toBe(false);
  expect(evidence.complete_performance_matrix).toBe(false);
  expect(evidence.all_models_all_dimensions_better).toBe(false);
  const native = evidence.native_fp16_attempt;
  expect(native.qualified).toBe(false);
  expect(native.build_settings.GGML_SYCL_F16).toBe('ON');
  expect(native.reference_commit).toBe('a4d4f2fd4ec85ec6a0a9ece4870416c1b6cc24f5');
  expect(native.latest_reference_check.observed_head).toBe(native.reference_commit);
  expect(native.findings.map((row) => row.id).sort()).toEqual([...evidence.required_models].sort());
  expect(native.findings.every((row) => !row.qualified && typeof row.error === 'string')).toBe(
    true,
  );
  const fp32 = evidence.native_fp32_starter_diagnostic;
  expect(fp32.qualified).toBe(false);
  expect(fp32.diagnostic_only).toBe(true);
  expect(fp32.reference_commit).toBe(native.reference_commit);
  expect(fp32.build.settings.GGML_SYCL_F16).toBe('OFF');
  expect(fp32.requests).toBe(24);
  expect(fp32.streams).toBe(8);
  expect(fp32.server_lifetimes).toBe(2);
  expect(fp32.streams_match_json).toBe(true);
  expect(fp32.output_drift).toHaveLength(1);
  const drift = fp32.output_drift[0];
  expect(drift.scenario).toBe('context-medium');
  expect(drift.actual.content_sha256).not.toBe(drift.expected.content_sha256);
  expect(drift.actual.usage).toEqual(drift.expected.usage);
  expect(evidence.limitations.join(' ')).toContain('OpenClaw');
});
