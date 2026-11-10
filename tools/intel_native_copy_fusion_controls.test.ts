import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read(
  '../site/public/zinc/validation/intel-native-35b-copy-fusion-controls-2026-11-10.json',
);
const previous = read('../site/public/zinc/validation/intel-native-exact-cadence-2026-11-10.json');

test('35B feature controls preserve the frozen native build and distinguish completed and rejected attempts', () => {
  for (const key of [
    'reference',
    'reference_selection',
    'native_build',
    'native_stack',
    'environment_source_commit',
    'environment_source_sha256',
  ]) {
    expect(report[key]).toEqual(previous[key]);
  }
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware).toEqual(previous.hardware);
  expect(report.treatments.map((row) => row.id)).toEqual(['synchronous-copy', 'unfused']);
  expect(report.treatments.map((row) => row.protocol_completed)).toEqual([true, false]);
  expect(report.treatments.map((row) => row.requested_environment)).toEqual([
    { GGML_SYCL_USE_ASYNC_MEM_OP: '0' },
    { GGML_SYCL_ENABLE_FUSION: '0' },
  ]);
  for (const row of report.treatments) {
    expect(row.passed).toBe(false);
    expect(row.source_report_sha256).toMatch(/^[a-f0-9]{64}$/);
    expect(Date.parse(row.completed_at_utc)).toBeGreaterThan(Date.parse(row.started_at_utc));
    expect(row.models).toHaveLength(1);
    expect(row.models[0].id).toBe('qwen36-35b-a3b-q4k-xl');
    expect(row.models[0].context).toBe(4096);
    expect(row.models[0].model_sha256).toBe(previous.models[1].model_sha256);
    expect(row.cleanup).toBe('no GPU process remains');
  }
  expect(report.passed).toBe(false);
  expect(report.performance_qualified).toBe(false);
  expect(report.release_qualification).toBe(false);
  expect(report.complete_model_matrix).toBe(false);
  expect(report.all_models_all_dimensions_better).toBe(false);
  expect(report.native_zinc_backend).toBe(false);
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('synchronous-copy failures retain every warmup, measured response and short stream', () => {
  const row = report.treatments[0].models[0];
  expect(row.native_reference_validation.passed).toBe(true);
  expect(row.samples).toHaveLength(32);
  expect(row.streams).toHaveLength(8);
  expect(row.processes).toHaveLength(2);
  const expected = new Map();
  const changes = [];
  for (const sample of row.samples) {
    const identity = Object.fromEntries(
      ['content_sha256', 'promptTokens', 'generatedTokens', 'finish_reason'].map((key) => [
        key,
        sample[key],
      ]),
    );
    const first = expected.get(sample.scenario) ?? identity;
    expected.set(sample.scenario, first);
    expect(sample.warmup).toBe(sample.repeat === 0);
    expect(sample.usage.prompt_tokens).toBe(sample.promptTokens);
    expect(sample.usage.completion_tokens).toBe(sample.generatedTokens);
    expect(sample.usage.prompt_tokens_details.cached_tokens).toBe(0);
    expect(sample.server_timings.cache_n).toBe(0);
    if (JSON.stringify(first) !== JSON.stringify(identity))
      changes.push({
        lifetime: sample.lifetime,
        scenario: sample.scenario,
        repeat: sample.repeat,
        expected: first,
        actual: identity,
      });
  }
  expect(changes).toEqual(row.output_drift);
  expect(changes).toHaveLength(19);
  const streams = new Map();
  const streamChanges = [];
  for (const sample of row.streams) {
    const first = streams.get(sample.scenario) ?? sample.content_sha256;
    streams.set(sample.scenario, first);
    if (first !== sample.content_sha256)
      streamChanges.push({
        lifetime: sample.lifetime,
        scenario: sample.scenario,
        expected: first,
        actual: sample.content_sha256,
      });
  }
  expect(streamChanges).toEqual(row.stream_drift);
  expect(streamChanges).toHaveLength(1);
  for (const process of row.processes) {
    expect(process.gpu_offload.offloaded).toBe(process.gpu_offload.total);
    expect(process.native_runtime.runtime).toBe('level_zero');
  }
});

test('the unfused attempt fails actual frozen-oracle agreement before any timing can be qualified', () => {
  const treatment = report.treatments[1];
  const row = treatment.models[0];
  expect(row.samples).toEqual([]);
  expect(row.streams).toEqual([]);
  expect(row.processes).toEqual([]);
  expect(row.native_reference_validation.passed).toBe(false);
  expect(row.native_reference_validation.failed_gate).toBe('native_frozen_oracle');
  expect(row.native_reference_validation.tail_qualification_completed).toBe(false);
  expect(row.native_reference_validation.cases).toHaveLength(3);
  for (const sample of row.native_reference_validation.cases) {
    expect(sample.reference_ids).toHaveLength(16);
    expect(sample.native_ids).toHaveLength(16);
    expect(sample.source_case_sha256).toMatch(/^[a-f0-9]{64}$/);
    if (sample.comparison.near_ties.length) expect(sample.tail_qualified).toBe(false);
  }
  const check = spawnSync(
    'python3',
    [
      '-c',
      `import json,sys
from release_lib import reference_agreement
try: reference_agreement(json.loads(sys.argv[1]), expected_cases=3)
except ValueError as error: print(str(error))
else: sys.exit(1)
`,
      JSON.stringify(row.native_reference_validation.cases.map((sample) => sample.comparison)),
    ],
    {
      cwd: new URL('.', import.meta.url),
      encoding: 'utf8',
      timeout: 30_000,
    },
  );
  expect(check.status, check.stdout + check.stderr).toBe(0);
  expect(check.stdout.trim()).toBe('reference exact-choice agreement is too low');
  expect(treatment.postflight).toEqual(
    expect.objectContaining({
      reference_unchanged: true,
      native_stack_unchanged: true,
      hardware_unchanged: true,
      cleanup: 'no GPU process remains',
    }),
  );
});
