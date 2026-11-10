import { expect, test } from 'bun:test';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read('../site/public/zinc/validation/intel-client-acceptance-2026-11-09.json');
const support = read(
  '../site/public/zinc/validation/intel-native-sycl-fp32-baseline-2026-11-09.json',
);
const gemma = read('../site/public/zinc/validation/intel-gemma-chat-prefix-2026-11-09.json');
const policy = read('../config/release/targets.json');
const fixtures = ['rate-limiter-single', 'cart-multi-file', 'python-log-cli'];
const hash = (text: string) => createHash('sha256').update(text).digest('hex');

test('fresh Intel client evidence covers the full pinned catalog on one immutable CI package', () => {
  expect(report.completed).toBe(true);
  expect(report.zinc).toEqual(gemma.zinc);
  expect(report.zinc.binary_sha256).toBe(support.zinc.sha256);
  expect(report.source_report_sha256.gemma).toBe(gemma.source_report_sha256);
  expect(report.source_report_sha256.inference).toBe(support.source_report_sha256);
  expect(report.source_report_sha256.remaining_models).toMatch(/^[a-f0-9]{64}$/);
  expect(report.harness_commit).toBe('a6db470e981d8b991b914cc6b2b1763274f05676');
  expect(report.clients.opencode).toEqual({
    ...gemma.opencode_cli,
    protocol_record: '/zinc/validation/opencode-protocol-2.2.6-2026-11-09.json',
    protocol_record_sha256: gemma.opencode_cli.protocol_sha256,
  });
  expect(report.models.map((model) => model.id).sort()).toEqual(
    policy.models.map((model) => model.id).sort(),
  );
  expect(Object.keys(report.zinc.shader_sha256)).toHaveLength(214);
  expect(report.zinc.shader_sha256).toEqual(
    Object.fromEntries(
      Object.entries(support.zinc.shader_sha256).map(([name, digest]) => [
        name.replace(/^share\/zinc\/shaders\//, ''),
        digest,
      ]),
    ),
  );
  expect(report.hardware).toEqual(gemma.hardware_after);
  expect(report.hardware_unchanged).toBe(true);
  expect(report.cleanup).toBe('no GPU process remains');
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('every model retains complete native continuations before and after actual coding tasks', () => {
  expect(report.all_models_inference_stable).toBe(true);
  let responses = 0;
  for (const model of report.models) {
    const pin = policy.models.find((entry) => entry.id === model.id);
    const cases = support.models.find((entry) => entry.id === model.id).correctness.cases;
    expect(model.model_sha256).toBe(pin.sha256);
    expect(model.inference_stable).toBe(true);
    expect(model.native_replay_before_after).toBe(true);
    expect(model.context).toBe(model.id === 'gemma4-31b-q4k-m' ? 12288 : 16384);
    for (const replay of [model.native_before, model.native_after]) {
      expect(replay).toHaveLength(cases.length);
      for (const [index, response] of replay.entries()) {
        const reference = cases[index];
        const accepted = [hash(reference.reference_text)];
        if (reference.validated_text != null) {
          // Only the independently teacher-forced native near tie may add an
          // alternate complete continuation; client output cannot add one.
          expect(reference.comparison.near_ties).not.toHaveLength(0);
          accepted.push(hash(reference.validated_text));
        }
        expect(response.case).toBe(index);
        expect(accepted).toContain(response.content_sha256);
        expect(response.completion_tokens).toBe(16);
        expect(response.response_sha256).toMatch(/^[a-f0-9]{64}$/);
        expect(response.comparison).toBe(
          response.content_sha256 === accepted[0] ? 'exact' : 'previously-qualified-near-tie',
        );
        responses++;
      }
    }
  }
  expect(responses).toBe(54);
});

test('client acceptance requires independent tests and protected files, separate from inference support', () => {
  expect(report.core_fixtures).toEqual(fixtures);
  expect(report.settings).toEqual({
    kv_cache: 'f16',
    mtp: 'off',
    slots: 1,
    thinking: false,
    ZINC_CHAT_PREFIX_ROLLBACK: 'unset',
    opencode_steps: 16,
    opencode_output_tokens: 2048,
    opencode_fixture_deadline_seconds: 900,
    openclaw_task_deadline_seconds: 300,
  });
  const accepted = [];
  for (const model of report.models) {
    expect(model.opencode.core_fixtures).toEqual(fixtures);
    expect(model.opencode.results.map((result) => result.fixture)).toEqual(fixtures);
    expect(model.opencode.core_passed).toBe(
      model.opencode.process_exit === 0 && model.opencode.results.every((result) => result.passed),
    );
    for (const result of model.opencode.results) {
      expect(result.initial_test_exit).not.toBe(0);
      expect(result.trace).toEqual({ files: 0, repairedEvents: 0, suppressedContentEvents: 0 });
      if (result.passed) {
        expect(result.protected_file_violations).toBe(0);
        expect(result.unexpected_files).toBe(0);
        expect(result.shortcuts).toBe(0);
        expect(result.client_exit).toBe(0);
        expect(result.timed_out).toBe(false);
        expect(result.client_error_count).toBe(0);
        expect(result.tools.toolCalls).toBeGreaterThan(0);
        expect(result.final_test_exit).toBe(0);
        expect(result.final_tests.pass).toBeGreaterThan(0);
        expect(result.final_tests.fail).toBe(0);
      }
    }
    if (model.opencode.core_passed) accepted.push(model.id);
  }
  expect(report.core_coding_accepted_models).toEqual(accepted.sort());
  expect(report.all_models_core_coding_accepted).toBe(accepted.length === 7);
  expect(accepted).toEqual(
    report.models
      .filter((model) => model.id !== 'qwen3-0.6b-q8-0')
      .map((model) => model.id)
      .sort(),
  );
  const starter = report.models.find((model) => model.id === 'qwen3-0.6b-q8-0');
  expect(starter.opencode.results.every((result) => result.passed === false)).toBe(true);
  expect(report.all_models_core_coding_accepted).toBe(false);
});

test('fresh OpenClaw scope and earlier failures remain explicit without release or performance promotion', () => {
  expect(report.clients.openclaw_version).toBe('OpenClaw 2026.11.1 (4ffdd20)');
  const claw = report.models.filter((model) => model.openclaw);
  expect(claw.map((model) => model.id).sort()).toEqual(
    gemma.models.map((model) => model.id).sort(),
  );
  for (const model of claw) {
    expect(model.openclaw.passed).toBe(true);
    expect(model.openclaw.tasks.every((task) => task.completed && task.passed)).toBe(true);
    expect(model.openclaw.tasks[1].checks).toEqual({
      'contract.json': true,
      'feature.test.mjs': true,
      receipt: true,
      tests: true,
    });
    expect(model.cache_controls).toHaveLength(8);
  }
  expect(report.earlier_attempts).toHaveLength(3);
  expect(
    report.earlier_attempts.every((attempt) => attempt.source_controller_passed === false),
  ).toBe(true);
  const extra = report.earlier_attempts[2].completed_extra_fixture_results;
  expect(extra).toHaveLength(1);
  expect(extra[0]).toEqual(
    expect.objectContaining({
      fixture: 'zinc-cli-validation',
      passed: false,
      client_exit: 0,
      final_tests: { pass: 2, fail: 4 },
      protected_file_violations: 0,
    }),
  );
  expect(report.release_qualification).toBe(false);
  expect(report.all_models_all_dimensions_better).not.toBe(true);
});
