import { expect, test } from 'bun:test';
import { execFileSync } from 'node:child_process';
import reference from '../config/benchmark-reference.json';
import policy from '../config/release/targets.json';
import { mtpModelId, mtpWorkloads } from './mtp_workloads.mjs';
import {
  mergeMatchedReports,
  requestDigest,
  validateHarnessSources,
  validateMatchedReport,
} from './merge_matched_mtp.mjs';
import { overallRatio, rateRatio, rowNotes, rowsFor } from '../site/src/lib/benchmark-view';

function fixture(): any {
  const model = policy.models.find((model) => model.id === mtpModelId)!;
  const report = {
    schema_version: 'zinc-matched-mtp-v1',
    backend: 'vulkan',
    protocol_passed: true,
    passed: true,
    completed_at: '2026-11-02T03:00:00Z',
    output_drift: [],
    samples: [],
    model: { id: model.id, sha256: model.sha256, size_bytes: model.size_bytes },
    reference: {
      commit: reference.commit,
      version: 'version: 0.8.0-dev (build 12033, commit ' + reference.commit.slice(0, 9) + ')',
      server_sha256: '1'.repeat(64),
      tokenizer_sha256: '2'.repeat(64),
      library_sha256: { 'libggml.so': '3'.repeat(64) },
    },
    zinc: { commit: '4'.repeat(40), sha256: '5'.repeat(64) },
    harness_source_commit: '7'.repeat(40),
    harness_sha256: Object.fromEntries(
      [
        'tools/benchmark_matched_mtp.py',
        'tools/mtp_workloads.mjs',
        'tools/parse_mtp_sample.mjs',
        'tools/performance_suite.mjs',
        'tools/benchmark_speculation.mjs',
        'tools/release_gpu.py',
        'tools/release_lib.py',
        'config/benchmark-reference.json',
      ].map((name) => [name, '6'.repeat(64)]),
    ),
    settings: {
      context: 4096,
      slots: 1,
      kv_cache: 'f32',
      temperature: 0,
      seed: 0,
      top_p: 1,
      prompt_cache: false,
      rounds: 3,
      runs: 3,
      warmup: 1,
      drafts: 3,
    },
    scenarios: mtpWorkloads.map((scenario) => ({
      id: scenario.id,
      request_sha256: requestDigest(scenario.body),
    })),
  } as any;
  for (const engine of ['zinc', 'llama'])
    for (const mtp of ['off', 'on'])
      for (const scenario of mtpWorkloads)
        for (let round = 1; round <= 3; round++)
          for (let iteration = 0; iteration < 4; iteration++) {
            const rate = engine === 'zinc' ? (mtp === 'on' ? 100 : 40) : mtp === 'on' ? 80 : 30;
            report.samples.push({
              engine,
              mtp,
              scenario: scenario.id,
              round,
              warmup: iteration === 0,
              promptTokens: 58,
              prefillTokens: 58,
              generatedTokens: scenario.max_tokens,
              finish_reason: 'length',
              prefillTps: 100,
              decodeTps: rate,
              msPerToken: 1000 / rate,
              totalLatencyMs: 2000,
              content_sha256: 'a'.repeat(64),
              content_bytes: 100,
              speculative:
                mtp === 'on'
                  ? { kind: 'nextn', drafted: 50, accepted: 30, acceptancePct: 60 }
                  : null,
            });
          }
  return report;
}

function published(): any {
  return {
    targets: [
      {
        id: 'rdna',
        generated_at: '2026-09-01T00:00:00Z',
        provenance: { zinc: { commit: 'old-zinc' }, llama_cpp: { commit: 'old-llama' } },
        models: [
          { id: mtpModelId, label: 'Qwen 3.8 27B', scenarios: [] },
          {
            id: 'retained',
            label: 'Retained',
            generated_at: '2026-09-01T00:00:00Z',
            scenarios: [],
          },
        ],
      },
    ],
  };
}

test('canonical payload hashes agree with the frozen Python serializer', () => {
  for (const scenario of mtpWorkloads) {
    const expected = execFileSync(
      'python3',
      [
        '-c',
        'import sys,json,hashlib; print(hashlib.sha256(json.dumps(json.load(sys.stdin),sort_keys=True).encode()).hexdigest())',
      ],
      { input: JSON.stringify(scenario.body), encoding: 'utf8' },
    ).trim();
    expect(requestDigest(scenario.body)).toBe(expected);
  }
});

test('matched publication switches both engines and preserves other models sources and dates', () => {
  const result = mergeMatchedReports(published(), [fixture()]);
  const on = rowsFor(result.targets[0], 'core', 'on').find((row) => row.model.id === mtpModelId)!;
  const off = rowsFor(result.targets[0], 'core', 'off').find((row) => row.model.id === mtpModelId)!;
  expect(on.baseline.speculative_decoding.enabled).toBe(true);
  expect(off.baseline.speculative_decoding.enabled).toBe(false);
  expect(on.baseline.decode_tps.median).toBe(80);
  expect(off.baseline.decode_tps.median).toBe(30);
  expect(on.zinc.decode_tps.samples).toHaveLength(9);
  expect(on.baseline.speculative_decoding.drafted_tokens).toBe(450);
  expect(rateRatio(on, 'decode_tps')).toBe(1.25);
  expect(rateRatio(on, 'prefill_tps')).toBeNull();
  expect(overallRatio(on)).toBeNull();
  expect(on.comparison.latency_pct_of_baseline).toBe(100);
  expect(on.comparison.prompt_pct_of_baseline).toBeNull();
  expect(rateRatio(off, 'decode_tps')).toBeCloseTo(4 / 3);
  const retained = result.targets[0].models.find((model) => model.id === 'retained');
  expect(retained.generated_at).toBe('2026-09-01T00:00:00Z');
  expect(retained.provenance.llama_cpp.commit).toBe('old-llama');
  const old = rowsFor(result.targets[0], 'core', 'on').find((row) => row.model.id === 'retained')!;
  expect(old.provenance.harness).toBeNull();
  expect(on.provenance.harness.path).toBe('tools/benchmark_matched_mtp.py');
  expect(on.model.measurement_history[0].provenance.llama_cpp.commit).toBe('old-llama');
  const again = mergeMatchedReports(result, [fixture()]);
  expect(
    again.targets[0].models.find((model) => model.id === mtpModelId).measurement_history,
  ).toHaveLength(1);
});

test('incomplete matrices, stale references, disabled drafts and changed payloads cannot publish', () => {
  for (const alter of [
    (report: any) => report.samples.pop(),
    (report: any) => {
      report.reference.commit = '5f436dddb';
    },
    (report: any) => {
      report.samples.find((sample) => sample.mtp === 'on').speculative = null;
    },
    (report: any) => {
      report.scenarios[0].request_sha256 = 'f'.repeat(64);
    },
    (report: any) => {
      report.samples[0].prefillTokens = 1;
    },
    (report: any) => {
      report.samples[0].content_sha256 = 'b'.repeat(64);
    },
    (report: any) => {
      report.reference.library_sha256 = {};
    },
  ]) {
    const report = fixture();
    alter(report);
    expect(() => validateMatchedReport(report, { includeDrift: true })).toThrow();
  }
});

test('a failed exact-output control remains excluded only in its affected mode and workload', () => {
  const report = fixture();
  report.passed = false;
  report.output_drift = [
    {
      engine: 'llama',
      scenario: 'context-long',
      off_sha256: 'a'.repeat(64),
      on_sha256: 'b'.repeat(64),
    },
  ];
  for (const sample of report.samples)
    if (sample.engine === 'llama' && sample.scenario === 'context-long' && sample.mtp === 'on')
      sample.content_sha256 = 'b'.repeat(64);
  expect(() => mergeMatchedReports(published(), [report])).toThrow('explicit diagnostic');
  const result = mergeMatchedReports(published(), [report], {
    includeDrift: true,
    evidenceUrls: ['/zinc/benchmarks/control.json'],
  });
  const target = result.targets[0];
  const on = rowsFor(target, 'context-long', 'on').find((row) => row.model.id === mtpModelId)!;
  const off = rowsFor(target, 'context-long', 'off').find((row) => row.model.id === mtpModelId)!;
  const core = rowsFor(target, 'core', 'on').find((row) => row.model.id === mtpModelId)!;
  expect(overallRatio(on)).toBeNull();
  expect(rateRatio(on, 'decode_tps')).toBeNull();
  expect(on.comparison).toBeNull();
  expect(off.correctness_review).toBeUndefined();
  expect(overallRatio(off)).not.toBeNull();
  expect(rateRatio(core, 'decode_tps')).not.toBeNull();
  expect(overallRatio(core)).toBeNull();
  expect(core.comparison.overall_unavailable_reason).toContain('prefill timers');
  expect(on.zinc.decode_tps.median).toBe(100);
  expect(rowNotes(on).some((note) => note.href === '/zinc/benchmarks/control.json')).toBe(true);
});

test('unrecorded output drift and mismatched frozen harness files are rejected', () => {
  const report = fixture();
  report.output_drift = [
    { engine: 'llama', scenario: 'core', off_sha256: 'a'.repeat(64), on_sha256: 'b'.repeat(64) },
  ];
  expect(() => validateMatchedReport(report, { includeDrift: true })).toThrow('accurately');
  const commit = execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  expect(() => validateHarnessSources(fixture(), commit)).toThrow('Frozen harness source differs');
  expect(() => validateHarnessSources(fixture(), 'HEAD')).toThrow('full frozen');
});
