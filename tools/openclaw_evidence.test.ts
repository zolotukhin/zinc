import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';

const evidence = JSON.parse(
  readFileSync(
    new URL('../site/public/zinc/validation/openclaw-amd-2026-11-05.json', import.meta.url),
    'utf8',
  ),
);
const policy = JSON.parse(
  readFileSync(new URL('../config/release/targets.json', import.meta.url), 'utf8'),
);

test('AMD OpenClaw evidence covers every catalog/backend attempt and immutable model bytes', () => {
  const models = new Map(policy.models.map((model) => [model.id, model]));
  expect(evidence.engine.version).toBe('0.1.0');
  expect(evidence.engine.commit).toBe('c3d4c0a23defa92b12394af09750554615b3cf41');
  expect(evidence.engine.kv_cache).toEqual({ vulkan: 'f16', rocm: 'f32' });
  const rows = evidence.matrix;
  expect(rows).toHaveLength(models.size * 2);
  expect(new Set(rows.map((row) => `${row.backend}/${row.model}`)).size).toBe(rows.length);
  for (const backend of ['vulkan', 'rocm']) {
    expect(
      rows
        .filter((row) => row.backend === backend)
        .map((row) => row.model)
        .sort(),
    ).toEqual([...models.keys()].sort());
  }
  for (const row of rows) expect(row.model_sha256).toBe(models.get(row.model).sha256);
  for (const model of evidence.models) {
    expect(model.sha256).toBe(models.get(model.id).sha256);
    expect(model.size_bytes).toBe(models.get(model.id).size_bytes);
  }
});

test('agent passes require independent artifacts and recovery; unsupported starter is never counted', () => {
  expect(evidence.counts).toEqual({
    attempted_backend_model_rows: 14,
    served_rows: 13,
    accepted_rows: 12,
    unsupported_rows: 1,
    completed_tasks: 39,
    passed_tasks: 36,
  });
  for (const row of evidence.matrix.filter((row) => row.acceptance_passed)) {
    expect(row.availability).toBe('served');
    expect(row.server_healthy_after).toBe(true);
    expect(row.tasks.map((task) => task.name)).toEqual(['text', 'workspace', 'recovery']);
    expect(row.tasks.every((task) => task.completed && task.passed)).toBe(true);
    const workspace = row.tasks.find((task) => task.name === 'workspace');
    expect(workspace.checks).toEqual({
      'contract.json': true,
      'feature.test.mjs': true,
      receipt: true,
      tests: true,
    });
    expect(workspace.toolSummary.calls).toBeGreaterThanOrEqual(3);
    expect(workspace.toolSummary.tools).toContain('read');
    expect(workspace.toolSummary.tools).toContain('exec');
  }
  const starter = evidence.matrix.filter((row) => row.model === 'qwen3-0.6b-q8-0');
  expect(starter.every((row) => !row.acceptance_passed)).toBe(true);
  expect(starter.find((row) => row.backend === 'rocm').availability).toBe(
    'unsupported_architecture',
  );
  expect(evidence.starter_native_control.full_gpu_offload).toBe(true);
  expect(
    evidence.starter_native_control.summary.results.find((task) => task.name === 'workspace')
      .passed,
  ).toBe(false);
});

test('retains the memory/context failures and the clean bounded-scratch confirmation', () => {
  expect(evidence.retained_context_failure.summary.passed).toBe(false);
  expect(evidence.retained_context_failure.summary.healthyAfter).toBe(true);
  const initial = evidence.matrix.find(
    (row) => row.backend === 'rocm' && row.model === 'gemma4-31b-q4k-m',
  );
  expect(initial.allocation_failure_messages).toBeGreaterThan(0);
  const bounded = evidence.bounded_gemma31_rocm;
  expect(bounded.context).toBe(initial.context);
  expect(bounded.prefill_chunk).toBe(128);
  expect(bounded.allocation_failure_messages).toBe(0);
  expect(bounded.summary.passed).toBe(true);
  expect(evidence.cpu_protocol.kind).toBe('cpu-scripted-protocol');
  expect(evidence.cpu_protocol.passed).toBe(true);
  expect(Object.values(evidence.cpu_protocol.failureChecks).every(Boolean)).toBe(true);
  expect(Object.values(evidence.final_gpu_process_counts).every((count) => count === 0)).toBe(true);
});
