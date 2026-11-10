import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const report = JSON.parse(
  readFileSync(
    new URL(
      '../site/public/zinc/validation/intel-gemma-prefill-tiling-2026-11-10.json',
      import.meta.url,
    ),
    'utf8',
  ),
);
const scenarios = ['core', 'context-medium', 'context-long', 'decode-extended'];
const median = (values: number[]) =>
  [...values].sort((a, b) => a - b)[Math.floor(values.length / 2)];

test('Gemma tiling proof retains the actual prototype archive and both independent references', () => {
  expect(report.archive).toEqual({
    archive_sha256: 'a0006af0e70ad1acbba7764dd9974209b9a93b1ae60e50828f7cd21c51bac68f',
    commit: 'cbbd10bb4e872f7748df04937abb2988bf64f01f',
    run_id: 42750914233,
    selected_head: '939b251872b7d3d35bc77237485a88aa4002930a',
    source_tree: '4c4de11c54e48c22029dabeac631869c214ced8f',
    version: '0.1.0-dev.42750914233',
  });
  expect(report.binary_sha256).toBe(
    '5e5742a9d7559e3ba848e67f477688c19503e8cb3be86ddb352f65c659d4e76a',
  );
  expect(report.numerical_proof.archive).toEqual(report.archive);
  expect(report.numerical_proof.binary_sha256).toBe(report.binary_sha256);
  expect(report.numerical_proof).toEqual(
    expect.objectContaining({ passed: true, completed: true, cases: 204, bitwise_pairs: 102 }),
  );
  expect(report.numerical_proof.maximum_absolute_error).toBeGreaterThanOrEqual(0);
  expect(report.numerical_proof.maximum_absolute_error).toBeLessThan(0.001);
  expect(report.candidate_source_sha256).toEqual(report.numerical_proof.source_sha256);
  expect(Object.keys(report.baseline_shader_sha256)).toHaveLength(215);
  expect(Object.keys(report.candidate_shader_sha256)).toHaveLength(215);
  expect(Object.keys(report.numerical_proof.shader_sha256)).toHaveLength(217);
  const changed = Object.keys(report.baseline_shader_sha256).filter(
    (name) => report.baseline_shader_sha256[name] !== report.candidate_shader_sha256[name],
  );
  expect(changed).toEqual([
    'share/zinc/shaders/mul_mm_q4k_gate_up_geglu.spv',
    'share/zinc/shaders/mul_mm_q4k_gate_up_geglu_full.spv',
  ]);
  for (const suffix of ['', '_full']) {
    expect(
      report.candidate_shader_sha256[`share/zinc/shaders/mul_mm_q4k_gate_up_geglu${suffix}.spv`],
    ).toBe(
      report.numerical_proof.shader_sha256[`share/zinc/shaders/gemma_geglu_tm2tn8${suffix}.spv`],
    );
  }
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('all 96 Gemma requests retain their complete outputs and all three balanced rounds', () => {
  expect(report.settings).toEqual({
    backend: 'vulkan',
    kv_cache: 'f16',
    mtp: 'off',
    prompt_cache: false,
    slots: 1,
    rounds: 3,
    runs: 3,
    warmup: 1,
  });
  expect(report.samples).toHaveLength(96);
  expect(report.summary.map((row) => row.scenario)).toEqual(scenarios);
  for (const scenario of scenarios) {
    const rows = report.samples.filter((row) => row.scenario === scenario);
    expect(rows).toHaveLength(24);
    expect(
      new Set(
        rows.map((row) =>
          JSON.stringify([
            row.content_sha256,
            row.promptTokens,
            row.generatedTokens,
            row.finish_reason,
          ]),
        ),
      ).size,
    ).toBe(1);
    const summary = report.summary.find((row) => row.scenario === scenario);
    expect(summary.content_sha256).toBe(rows[0].content_sha256);
    expect(summary.prompt_tokens).toBe(rows[0].promptTokens);
    expect(summary.generated_tokens).toBe(rows[0].generatedTokens);
    expect(summary.finish_reason).toBe(rows[0].finish_reason);
    expect(summary.rounds.map((row) => row.round)).toEqual([1, 2, 3]);
    for (const number of [1, 2, 3]) {
      const recorded = summary.rounds[number - 1];
      for (const treatment of ['baseline', 'candidate']) {
        const all = rows.filter((row) => row.round === number && row.treatment === treatment);
        expect(all).toHaveLength(4);
        expect(all.filter((row) => row.warmup)).toHaveLength(1);
        const measured = all.filter((row) => !row.warmup);
        expect(measured).toHaveLength(3);
        for (const field of ['prefillTps', 'decodeTps', 'totalTps', 'totalLatencyMs']) {
          expect(measured.every((row) => Number.isFinite(row[field]) && row[field] > 0)).toBe(true);
          expect(recorded.medians[treatment][field]).toBe(
            median(measured.map((row) => row[field])),
          );
        }
      }
      expect(recorded.prefill_ratio).toBe(
        recorded.medians.candidate.prefillTps / recorded.medians.baseline.prefillTps,
      );
      expect(recorded.decode_ratio).toBe(
        recorded.medians.candidate.decodeTps / recorded.medians.baseline.decodeTps,
      );
      expect(recorded.prefill_ratio).toBeGreaterThan(1);
      expect(Math.abs(recorded.decode_ratio - 1)).toBeLessThan(0.01);
    }
  }
});

test('six Gemma server lives retain 72 full native answers including window boundaries', () => {
  expect(report.processes.map((row) => [row.round, row.treatment])).toEqual([
    [1, 'baseline'],
    [1, 'candidate'],
    [2, 'candidate'],
    [2, 'baseline'],
    [3, 'baseline'],
    [3, 'candidate'],
  ]);
  const identities = new Map<number, string>();
  for (const server of report.processes) {
    expect(server.native_before).toBe(true);
    expect(server.native_after).toBe(true);
    expect(server.healthy_after).toBe(true);
    for (const phase of ['native_before_responses', 'native_after_responses']) {
      expect(server[phase]).toHaveLength(6);
      for (const [index, response] of server[phase].entries()) {
        expect(response.case).toBe(index);
        expect(response.completion_tokens).toBe(16);
        expect(response.response_sha256).toMatch(/^[a-f0-9]{64}$/);
        expect(response.content_sha256).toMatch(/^[a-f0-9]{64}$/);
        if (index >= 3) expect(response.prompt_tokens).toBeGreaterThan(1024);
        const previous = identities.get(index);
        if (previous) expect(response.content_sha256).toBe(previous);
        identities.set(index, response.content_sha256);
      }
    }
  }
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware_unchanged).toBe(true);
  expect(report.cleanup).toBe('no GPU process remains');
  expect(report.passed).toBe(true);
  expect(report.release_qualification).toBe(false);
  expect(report.performance_qualified).toBe(false);
  expect(report.all_models_all_dimensions_better).toBe(false);
});
