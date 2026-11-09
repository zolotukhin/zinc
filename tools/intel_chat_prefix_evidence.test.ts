import { expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read('../site/public/zinc/validation/intel-gemma-chat-prefix-2026-11-09.json');
const support = read(
  '../site/public/zinc/validation/intel-native-sycl-fp32-baseline-2026-11-09.json',
);
const policy = read('../config/release/targets.json');
const hash = (text: string) => createHash('sha256').update(text).digest('hex');
const fixtures = ['rate-limiter-single', 'cart-multi-file', 'python-log-cli'];
const controls = [
  'edited-system',
  'early-system-edit',
  'shortened-system',
  'identical-retry',
  'shortened-history',
  'context-growth',
  'intervening-request',
  'prefill-disconnect',
];

test('Gemma evidence binds the tested CI package to the seven-model inference replay', () => {
  expect(report.passed).toBe(true);
  expect(report.source_report_sha256).toBe(
    'a583e661a0bab5eda6409d9274a0a451025821c94c90546b5a1372bf9f5d1b76',
  );
  expect(report.zinc.commit).toBe('2d0e1369d361671edc5ddf2a676efb12625bbb66');
  expect(report.zinc.binary_sha256).toBe(support.zinc.sha256);
  expect(report.zinc.archive_sha256).toBe(support.zinc.archive_sha256);
  expect(Object.keys(report.zinc.shader_sha256)).toHaveLength(214);
  for (const [name, digest] of Object.entries(report.zinc.shader_sha256)) {
    expect(support.zinc.shader_sha256[`share/zinc/shaders/${name}`]).toBe(digest);
  }
  expect(report.numerical_reference).toEqual({
    passed: true,
    cases: 376,
    suites: { attention_window: 256, gemma_router: 36, shared_cache: 84 },
  });
  expect(report.q8_decode_reference).toEqual(
    expect.objectContaining({ passed: true, cases: 228, bitwise_pairs: 114 }),
  );
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware_unchanged).toBe(true);
  expect(report.cleanup).toBe('no GPU process remains');
  expect(report.settings).toEqual({
    kv_cache: 'f16',
    mtp: 'off',
    slots: 1,
    ZINC_CHAT_PREFIX_ROLLBACK: 'unset',
  });
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('both Gemmas retain complete native continuations and independent actual-client acceptance', () => {
  expect(report.models.map((model) => model.id)).toEqual([
    'gemma4-26b-a4b-q4k-m',
    'gemma4-31b-q4k-m',
  ]);
  expect(report.opencode_fixture_ids).toEqual(fixtures);
  expect(report.opencode_cli.version).toBe('2.2.6');
  expect(report.openclaw_version).toBe('2026.11.1');
  let nativeReplays = 0;
  for (const model of report.models) {
    const pin = policy.models.find((entry) => entry.id === model.id);
    const reference = support.models.find((entry) => entry.id === model.id).correctness.cases;
    expect(model.model_sha256).toBe(pin.sha256);
    expect(model.context).toBe(model.id.includes('26b') ? 16384 : 12288);
    expect(model.passed).toBe(true);
    expect(model.healthy_after).toBe(true);
    expect(model.tokenizer_cases).toBe(12);
    const replays = [model.native_before, model.native_after];
    if (model.disabled_override) replays.push(model.disabled_override.native_after);
    for (const replay of replays) {
      expect(replay).toHaveLength(6);
      for (const [index, response] of replay.entries()) {
        expect(response.case).toBe(index);
        expect(response.content_sha256).toBe(hash(reference[index].reference_text));
        expect(response.response_sha256).toMatch(/^[a-f0-9]{64}$/);
        expect(response.completion_tokens).toBe(16);
        nativeReplays++;
      }
    }
    expect(model.openclaw.passed).toBe(true);
    expect(model.openclaw.tasks.map((task) => task.name)).toEqual([
      'text',
      'workspace',
      'recovery',
    ]);
    expect(model.openclaw.tasks.every((task) => task.completed && task.passed)).toBe(true);
    expect(model.openclaw.tasks[1].checks).toEqual({
      'contract.json': true,
      'feature.test.mjs': true,
      receipt: true,
      tests: true,
    });
    expect(model.openclaw.tasks[1].toolSummary.calls).toBeGreaterThan(0);
    expect(model.opencode.passed).toBe(true);
    expect(model.opencode.fixtures.map((fixture) => fixture.fixture)).toEqual(fixtures);
    for (const fixture of model.opencode.fixtures) {
      expect(fixture.passed).toBe(true);
      expect(fixture.initial_test_exit).not.toBe(0);
      expect(fixture.final_test_exit).toBe(0);
      expect(fixture.final_tests.pass).toBeGreaterThan(0);
      expect(fixture.final_tests.fail).toBe(0);
      expect(fixture.client_exit).toBe(0);
      expect(fixture.timed_out).toBe(false);
      expect(fixture.tool_calls).toBeGreaterThan(0);
      expect(fixture.client_errors).toBe(0);
      expect(fixture.protected_file_violations).toBe(0);
      expect(fixture.unexpected_files).toBe(0);
      expect(fixture.shortcuts).toBe(0);
    }
  }
  expect(nativeReplays).toBe(30);
});

test('default rewind, early edits, invalidation and the explicit disable control remain distinct', () => {
  for (const model of report.models) {
    expect(model.controls.map((control) => control.case)).toEqual(controls);
    expect(model.controls.every((control) => control.passed)).toBe(true);
    for (const control of model.controls.filter((control) => control.rewind)) {
      expect(control.cached_response_sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(control.full_response_sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(control.content_sha256).toMatch(/^[a-f0-9]{64}$/);
      expect(control.cached_ms).toBeGreaterThan(0);
      expect(control.full_prefill_ms).toBeGreaterThan(0);
      if (model.id.includes('31b')) {
        expect(control.rewind).toEqual([]);
      } else {
        const reused = Math.max(...control.rewind.map((event) => event.reused));
        expect(reused).toBeGreaterThan(0);
        if (['early-system-edit', 'shortened-system'].includes(control.case))
          expect(reused).toBeLessThan(1024);
        else expect(reused).toBeGreaterThan(1024);
      }
    }
    expect(
      model.controls.find((control) => control.case === 'intervening-request')
        .canonical_reuse_events,
    ).toBe(0);
  }
  const moe = report.models[0];
  expect(moe.actual_openclaw_rewinds.length).toBeGreaterThan(0);
  expect(report.models[1].actual_openclaw_rewinds).toEqual([]);
  expect(moe.disabled_override.passed).toBe(true);
  expect(moe.disabled_override.controls).toHaveLength(1);
  expect(moe.disabled_override.controls[0].case).toBe('early-system-edit');
  expect(moe.disabled_override.controls[0].rewind).toEqual([]);
});

test('earlier failures remain visible and this two-model record cannot certify a release or all-model speed win', () => {
  expect(report.release_qualification).toBe(false);
  expect(report.all_models_all_dimensions_better).not.toBe(true);
  expect(report.earlier_attempts).toHaveLength(3);
  expect(report.earlier_attempts.every((attempt) => attempt.passed === false)).toBe(true);
  expect(report.earlier_attempts[2].completed).toBe(false);
  const extra = report.earlier_attempts[2].completed_extra_fixture_results;
  expect(extra).toHaveLength(1);
  expect(extra[0]).toEqual(
    expect.objectContaining({
      fixture: 'zinc-cli-validation',
      passed: false,
      client_exit: 0,
      timed_out: false,
      final_tests: { pass: 2, fail: 4 },
      protected_file_violations: 0,
    }),
  );
  expect(report.all_seven_inference_record).toBe(
    '/zinc/validation/intel-native-sycl-fp32-baseline-2026-11-09.json',
  );
});
