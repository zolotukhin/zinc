import { expect, test } from 'bun:test';
import policy from '../config/release/targets.json';
import reference from '../config/benchmark-reference.json';
import { gemmaModelId, gemmaWorkloads, gemmaWorkloadsForModel } from './gemma_rocm_workloads.mjs';
import { requestDigest } from './merge_matched_mtp.mjs';
import { mergeGemmaReport, validateGemmaReport } from './merge_gemma_rocm.mjs';
import { overallRatio, qualityStatus, rowsFor } from '../site/src/lib/benchmark-view';

function fixture(): any {
  const model = policy.models.find((model) => model.id === gemmaModelId)!;
  const identity = {
    commit: reference.commit,
    server_sha256: 'a'.repeat(64),
    tokenizer_sha256: 'b'.repeat(64),
    library_sha256: { 'libllama.so': 'c'.repeat(64) },
  };
  const cases = Array.from({ length: 6 }, (_, i) => ({
    prompt_tokens: i < 3 ? 30 : 1136,
    reference_ids: Array(16).fill(7),
    zinc_ids: Array(16).fill(7),
    reference_text: 'Full reference continuation',
    comparison: { tokens: 16, near_ties: [] },
  }));
  return {
    schema_version: 'zinc-gemma-rocm-v1',
    passed: true,
    release_qualification: false,
    cleanup: 'no GPU process remains',
    started_at: '2026-11-02T05:00:00Z',
    completed_at: '2026-11-02T06:00:00Z',
    model: {
      id: model.id,
      sha256: model.sha256,
      size_bytes: model.size_bytes,
      sliding_window: model.sliding_window,
    },
    reference: identity,
    oracle: structuredClone(identity),
    zinc: { commit: 'd'.repeat(40), sha256: 'e'.repeat(64) },
    harness_source_commit: 'f'.repeat(40),
    settings: {
      backend: 'rocm',
      context: 4096,
      slots: 1,
      kv_cache: 'f32',
      mtp: 'off',
      prefill: 'default-serial',
      prompt_cache: false,
      rounds: 3,
      runs: 3,
      warmup: 1,
      temperature: 0,
      top_p: 1,
      seed: 0,
    },
    correctness: {
      passed: true,
      oracle_backend: 'Vulkan',
      tokenizer_cases: 12,
      http_full_continuations: 18,
      http_after_cancellation: true,
      http_after_context_error: true,
      generation: structuredClone(policy.reference.generation),
      agreement_policy: structuredClone(policy.reference.agreement),
      cases,
      reference_agreement: { total_tokens: 96, exact_tokens: 96, near_tie_tokens: 0 },
    },
    scenarios: gemmaWorkloads.map((scenario) => ({
      id: scenario.id,
      request_sha256: requestDigest(scenario.body),
    })),
    samples: ['zinc', 'llama'].flatMap((engine) =>
      gemmaWorkloads.flatMap((scenario) =>
        Array.from({ length: 12 }, (_, i) => ({
          engine,
          scenario: scenario.id,
          mtp: 'off',
          round: Math.floor(i / 4) + 1,
          warmup: i % 4 === 0,
          promptTokens: 81,
          prefillTokens: 81,
          generatedTokens: scenario.max_tokens,
          finish_reason: 'length',
          prefillTps: 90,
          decodeTps: engine === 'zinc' ? 100 : 80,
          totalLatencyMs: 3000,
          content_sha256: '1'.repeat(64),
          content_bytes: 100,
          outputPreview: 'Here are several reasons the two benchmark results differ.',
        })),
      ),
    ),
  };
}

function vulkanFixture(modelId = gemmaModelId, target = 'rdna'): any {
  const report = fixture();
  const model = policy.models.find((model) => model.id === modelId)!;
  report.schema_version = 'zinc-gemma-vulkan-v1';
  report.target = target;
  report.model = {
    id: model.id,
    sha256: model.sha256,
    size_bytes: model.size_bytes,
    sliding_window: model.sliding_window,
  };
  report.settings = {
    ...report.settings,
    backend: 'vulkan',
    kv_cache: 'f16',
    prefill: 'backend-default',
  };
  report.hardware = {
    gpu: target === 'rdna' ? 'AMD Radeon AI PRO R9700' : 'Intel Arc Pro',
    os: 'Linux',
    driver: 'Mesa',
  };
  report.kernel_reference = { passed: true, cases: 376 };
  report.latest_reference_check = {
    commit: reference.commit,
    checked_at: '2026-11-02T05:00:01Z',
    scope: 'run-start',
    observed_head_at_run: reference.commit,
  };
  report.scenarios = gemmaWorkloadsForModel(modelId).map((scenario) => ({
    id: scenario.id,
    request_sha256: requestDigest(scenario.body),
  }));
  return report;
}

test('Gemma publication requires full reference, recovery and repeated benchmark gates', () => {
  expect(() => validateGemmaReport(fixture())).not.toThrow();
});

test('Python JSON key ordering does not change the reference policy', () => {
  const report = fixture();
  for (const key of ['generation', 'agreement_policy']) {
    report.correctness[key] = Object.fromEntries(Object.entries(report.correctness[key]).sort());
  }
  expect(() => validateGemmaReport(report)).not.toThrow();
});

for (const target of ['rdna', 'intel'])
  for (const modelId of [gemmaModelId, 'gemma4-31b-q4k-m']) {
    test(`Vulkan ${target} ${modelId} needs the full current-reference matrix`, () => {
      expect(() => validateGemmaReport(vulkanFixture(modelId, target))).not.toThrow();
    });
  }

test('a campaign freezes the reference verified before its build while recording later upstream changes', () => {
  const report = vulkanFixture();
  const selected = Date.parse(reference.checked_at_utc);
  report.started_at = new Date(selected + 5 * 60 * 1000).toISOString();
  report.completed_at = new Date(selected + 30 * 60 * 1000).toISOString();
  report.latest_reference_check = {
    commit: reference.commit,
    checked_at: reference.checked_at_utc,
    scope: 'campaign-before-build',
    observed_head_at_run: '9'.repeat(40),
  };
  expect(() => validateGemmaReport(report)).not.toThrow();
  report.started_at = new Date(selected + 3 * 60 * 60 * 1000).toISOString();
  report.completed_at = new Date(selected + 4 * 60 * 60 * 1000).toISOString();
  expect(() => validateGemmaReport(report)).toThrow('current upstream');
});

test('a changed upstream HEAD cannot be silently labeled as a current run-start reference', () => {
  const report = vulkanFixture();
  report.latest_reference_check.observed_head_at_run = '9'.repeat(40);
  expect(() => validateGemmaReport(report)).toThrow('current upstream');
});

for (const [name, mutate] of [
  [
    'missing shader checks',
    (r: any) => {
      delete r.kernel_reference;
    },
  ],
  [
    'incomplete shader checks',
    (r: any) => {
      r.kernel_reference.cases = 256;
    },
  ],
  [
    'wrong physical GPU',
    (r: any) => {
      r.hardware.gpu = 'Intel Arc Pro';
    },
  ],
  [
    'unverified latest reference',
    (r: any) => {
      delete r.latest_reference_check;
    },
  ],
  [
    'stale latest-reference check',
    (r: any) => {
      r.latest_reference_check.commit = '0'.repeat(40);
    },
  ],
  [
    'different KV storage',
    (r: any) => {
      r.settings.kv_cache = 'f32';
    },
  ],
] as const)
  test(`Vulkan publication rejects ${name}`, () => {
    const report = vulkanFixture();
    mutate(report);
    expect(() => validateGemmaReport(report)).toThrow();
  });

test('Vulkan replacements retire only their own model and hardware cautions', () => {
  const existing: any = {
    correctness_review: { affected_model_rows: 4 },
    targets: ['rdna', 'intel'].map((id) => ({
      id,
      methodology: { notes: [] },
      models: [gemmaModelId, 'gemma4-31b-q4k-m'].map((modelId) => ({
        id: modelId,
        label: modelId,
        generated_at: '2026-09-24T00:00:00Z',
        correctness_review: { tone: 'caution', note: 'Old sliding-window implementation.' },
        scenarios: [],
      })),
    })),
  };
  const result = mergeGemmaReport(existing, vulkanFixture('gemma4-31b-q4k-m'), '/vulkan.json');
  expect(result.correctness_review.affected_model_rows).toBe(3);
  expect(
    result.targets
      .find((t: any) => t.id === 'intel')
      .models.every((m: any) => m.correctness_review.tone === 'caution'),
  ).toBe(true);
  const target = result.targets.find((t: any) => t.id === 'rdna');
  const dense = target.models.find((m: any) => m.id === 'gemma4-31b-q4k-m');
  expect(dense.correctness_review).toBeNull();
  expect(dense.server_kv_cache).toEqual({ zinc: 'f16', llama_cpp: 'f16' });
  expect(dense.measurement_history[0].correctness_review.tone).toBe('caution');
  expect(dense.revalidation_evidence.prefill).toBe('backend-default');
  expect(
    overallRatio(rowsFor(target, 'core', 'off').find((row: any) => row.model.id === dense.id)!),
  ).not.toBeNull();
});

const failures: [string, (report: any) => void][] = [
  [
    'old baseline',
    (r) => {
      r.reference.commit = '0'.repeat(40);
    },
  ],
  [
    'different weights',
    (r) => {
      r.model.sha256 = '0'.repeat(64);
    },
  ],
  [
    'batched prefill',
    (r) => {
      r.settings.prefill = 'batched';
    },
  ],
  [
    'weakened reference',
    (r) => {
      r.correctness.agreement_policy.max_logprob_gap = 3;
    },
  ],
  [
    'short extended case',
    (r) => {
      r.correctness.cases[5].prompt_tokens = 1024;
    },
  ],
  [
    'wrong full continuation',
    (r) => {
      r.correctness.cases[5].zinc_ids[15] = 8;
    },
  ],
  [
    'truncated continuation',
    (r) => {
      r.correctness.cases[5].zinc_ids.pop();
    },
  ],
  [
    'missing recovery',
    (r) => {
      r.correctness.http_after_context_error = false;
    },
  ],
  [
    'missing sample',
    (r) => {
      r.samples.pop();
    },
  ],
  [
    'early stop',
    (r) => {
      r.samples[0].generatedTokens--;
    },
  ],
  [
    'prompt cache',
    (r) => {
      r.samples[0].prefillTokens = 2;
    },
  ],
  [
    'different prompts',
    (r) => {
      r.samples[0].promptTokens += 5;
      r.samples[0].prefillTokens += 5;
    },
  ],
  [
    'different repeated output',
    (r) => {
      r.samples[1].content_sha256 = '2'.repeat(64);
    },
  ],
  [
    'active MTP',
    (r) => {
      r.samples[0].speculative = { drafted: 30, accepted: 20 };
    },
  ],
  [
    'different payload',
    (r) => {
      r.scenarios[0].request_sha256 = '0'.repeat(64);
    },
  ],
];
for (const [name, mutate] of failures)
  test(`Gemma publication rejects ${name}`, () => {
    const report = fixture();
    mutate(report);
    expect(() => validateGemmaReport(report)).toThrow();
  });

test('only a fresh passing Gemma measurement clears its warning; the old record stays intact', () => {
  const caution = {
    tone: 'caution',
    label: 'Revalidation required',
    note: 'Historical batched prefill is unqualified.',
  };
  const old = {
    id: gemmaModelId,
    label: 'Gemma 4 26B',
    generated_at: '2026-09-24T00:00:00Z',
    correctness_review: caution,
    scenarios: [],
  };
  const retained = {
    id: 'other',
    label: 'Other model',
    generated_at: '2026-09-25T00:00:00Z',
    scenarios: [],
    correctness_review: caution,
  };
  const existing: any = {
    targets: [
      {
        id: 'rdna-rocm',
        generated_at: '2026-09-25T00:00:00Z',
        models: [old, retained],
        methodology: { notes: [] },
      },
    ],
  };
  const report = fixture();
  const first = mergeGemmaReport(existing, report, '/evidence.json');
  const target = first.targets[0];
  const model = target.models.find((model: any) => model.id === gemmaModelId)!;
  expect(model.correctness_review).toBeNull();
  expect(model.measurement_history).toEqual([old]);
  expect(target.models.find((model: any) => model.id === 'other').correctness_review).toEqual(
    caution,
  );
  expect(old.correctness_review).toEqual(caution);
  for (const scenario of gemmaWorkloads) {
    const row = rowsFor(target, scenario.id, 'off')[0];
    expect(qualityStatus(row).tone).not.toBe('caution');
    expect(overallRatio(row)).toBeGreaterThan(1);
    expect(row.zinc.decode_tps.samples).toHaveLength(9);
  }
  const again = mergeGemmaReport(first, report, '/evidence.json');
  expect(
    again.targets[0].models.find((model: any) => model.id === gemmaModelId).measurement_history,
  ).toEqual([old]);
});
