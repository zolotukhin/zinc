#!/usr/bin/env bun
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import referencePin from '../config/benchmark-reference.json' with { type: 'json' };
import policy from '../config/release/targets.json' with { type: 'json' };
import { gemmaModelId, gemmaWorkloads } from './gemma_rocm_workloads.mjs';
import { requestDigest } from './merge_matched_mtp.mjs';
import { createStats, mergeArtifacts, requireObservedMtp } from './performance_suite.mjs';
import { requireBenchmarkReference } from './benchmark_speculation.mjs';

const modelPin = policy.models.find((model) => model.id === gemmaModelId);
const digest = (value) => /^[a-f0-9]{64}$/.test(value ?? '');
const requireValue = (condition, message) => {
  if (!condition) throw new Error(message);
};
const harnessFiles = [
  'tools/benchmark_gemma_rocm.py',
  'tools/gemma_rocm_workloads.mjs',
  'tools/benchmark_matched_mtp.py',
  'tools/parse_mtp_sample.mjs',
  'tools/performance_suite.mjs',
  'tools/benchmark_speculation.mjs',
  'tools/release_gpu.py',
  'tools/release_lib.py',
  'config/benchmark-reference.json',
  'config/release/targets.json',
];

export function validateGemmaHarness(report, commit) {
  requireValue(/^[a-f0-9]{40}$/.test(commit ?? ''), 'Supply the full frozen harness commit');
  requireValue(
    Object.keys(report.harness_sha256 ?? {})
      .sort()
      .join() === [...harnessFiles].sort().join(),
    'Harness sources are incomplete',
  );
  for (const name of harnessFiles) {
    const source = execFileSync('git', ['show', `${commit}:${name}`]);
    requireValue(
      createHash('sha256').update(source).digest('hex') === report.harness_sha256[name],
      `Harness source differs: ${name}`,
    );
  }
}

export function validateGemmaReport(report) {
  requireValue(
    report.schema_version === 'zinc-gemma-rocm-v1' &&
      report.passed === true &&
      !report.error &&
      report.cleanup === 'no GPU process remains',
    'Gemma campaign did not pass',
  );
  requireValue(
    Number.isFinite(Date.parse(report.completed_at)) &&
      Date.parse(report.completed_at) >= Date.parse(report.started_at),
    'Campaign dates are invalid',
  );
  requireValue(report.release_qualification === false, 'A benchmark cannot qualify a release');
  requireValue(
    report.model?.id === gemmaModelId &&
      report.model.sha256 === modelPin.sha256 &&
      report.model.size_bytes === modelPin.size_bytes &&
      report.model.sliding_window === modelPin.sliding_window,
    'Model identity differs',
  );
  for (const ref of [report.reference, report.oracle]) {
    requireBenchmarkReference(
      { commit: ref?.commit, sha256: ref?.server_sha256 },
      referencePin.commit,
    );
    requireValue(
      digest(ref.tokenizer_sha256) &&
        Object.values(ref.library_sha256 ?? {}).length > 0 &&
        Object.values(ref.library_sha256).every(digest),
      'Reference artifacts are incomplete',
    );
  }
  requireValue(
    /^[a-f0-9]{40}$/.test(report.zinc?.commit ?? '') && digest(report.zinc?.sha256),
    'ZINC identity is missing',
  );
  const settings = report.settings;
  requireValue(
    settings?.backend === 'rocm' &&
      settings.context === 4096 &&
      settings.slots === 1 &&
      settings.kv_cache === 'f32' &&
      settings.mtp === 'off' &&
      settings.prefill === 'default-serial' &&
      settings.prompt_cache === false &&
      settings.temperature === 0 &&
      settings.top_p === 1 &&
      settings.seed === 0,
    'Controlled settings differ',
  );
  requireValue(
    [settings.rounds, settings.runs, settings.warmup].every(Number.isSafeInteger) &&
      settings.rounds >= 3 &&
      settings.runs >= 3 &&
      settings.warmup >= 1,
    'Full repeated matrix is required',
  );
  const check = report.correctness;
  requireValue(
    check?.passed === true &&
      check.oracle_backend === 'Vulkan' &&
      check.tokenizer_cases === 12 &&
      check.http_full_continuations === 18 &&
      check.http_after_cancellation === true &&
      check.http_after_context_error === true,
    'Complete correctness and recovery gates are required',
  );
  requireValue(
    JSON.stringify(check.generation) === JSON.stringify(policy.reference.generation) &&
      JSON.stringify(check.agreement_policy) === JSON.stringify(policy.reference.agreement),
    'Reference policy changed',
  );
  requireValue(check.cases?.length === 6, 'Six full reference cases are required');
  for (const [index, sample] of check.cases.entries()) {
    requireValue(
      sample.reference_ids?.length === 16 &&
        sample.zinc_ids?.length === 16 &&
        sample.zinc_ids.every(
          (id, i) => Number.isSafeInteger(id) && id >= 0 && id === sample.reference_ids[i],
        ) &&
        sample.comparison?.tokens === 16 &&
        sample.comparison.near_ties?.length === 0 &&
        sample.reference_text?.length > 0,
      'Every reference continuation must match exactly',
    );
    requireValue(
      Number.isSafeInteger(sample.prompt_tokens) &&
        sample.prompt_tokens + 16 <= settings.context &&
        (index < 3 || sample.prompt_tokens > modelPin.sliding_window),
      'Extended reference did not cross the sliding window',
    );
  }
  requireValue(
    check.reference_agreement?.total_tokens === 96 &&
      check.reference_agreement.exact_tokens === 96 &&
      check.reference_agreement.near_tie_tokens === 0,
    'Reference totals differ',
  );
  requireValue(
    report.scenarios?.length === gemmaWorkloads.length &&
      gemmaWorkloads.every(
        (scenario) =>
          report.scenarios.find((row) => row.id === scenario.id)?.request_sha256 ===
          requestDigest(scenario.body),
      ),
    'Canonical workload payloads differ',
  );
  requireValue(
    report.samples?.length ===
      2 * gemmaWorkloads.length * settings.rounds * (settings.runs + settings.warmup),
    'Sample matrix is incomplete',
  );
  for (const scenario of gemmaWorkloads) {
    const counts = [];
    for (const engine of ['zinc', 'llama']) {
      const rows = report.samples.filter(
        (sample) => sample.engine === engine && sample.scenario === scenario.id,
      );
      requireValue(
        rows.length === settings.rounds * (settings.runs + settings.warmup),
        'A treatment is incomplete',
      );
      for (let round = 1; round <= settings.rounds; round++) {
        const group = rows.filter((sample) => sample.round === round);
        requireValue(
          group.filter((sample) => sample.warmup === true).length === settings.warmup &&
            group.filter((sample) => sample.warmup === false).length === settings.runs,
          'A round is incomplete',
        );
      }
      for (const sample of rows) {
        requireValue(sample.mtp === 'off', 'MTP modes differ');
        requireObservedMtp(sample, 'off');
        requireValue(
          sample.generatedTokens === scenario.max_tokens && sample.finish_reason === 'length',
          'Output budgets differ',
        );
        requireValue(
          Number.isSafeInteger(sample.promptTokens) &&
            sample.promptTokens > 0 &&
            sample.prefillTokens >= sample.promptTokens - 1,
          'Prompt cache reduced work',
        );
        requireValue(
          [sample.prefillTps, sample.decodeTps, sample.totalLatencyMs].every(
            (value) => typeof value === 'number' && Number.isFinite(value) && value > 0,
          ) &&
            digest(sample.content_sha256) &&
            sample.content_bytes > 0,
          'Sample telemetry is incomplete',
        );
        counts.push(sample.promptTokens);
      }
      requireValue(
        new Set(rows.map((sample) => sample.content_sha256)).size === 1,
        'Repeated outputs differ',
      );
    }
    requireValue(Math.max(...counts) - Math.min(...counts) <= 1, 'Prompt counts differ');
  }
}

export function mergeGemmaReport(existing, report, evidenceUrl) {
  validateGemmaReport(report);
  const target = existing.targets.find((target) => target.id === 'rdna-rocm');
  const previous = target?.models.find((model) => model.id === gemmaModelId);
  requireValue(previous, 'Published Gemma model is missing');
  const { measurement_history: history = [], ...old } = previous;
  const sameCampaign =
    previous.generated_at === report.completed_at &&
    previous.provenance?.zinc?.sha256 === report.zinc.sha256;
  const provenance = {
    harness: { commit: report.harness_source_commit, path: 'tools/benchmark_gemma_rocm.py' },
    zinc: {
      commit: report.zinc.commit,
      version: report.zinc.commit.slice(0, 12),
      sha256: report.zinc.sha256,
    },
    llama_cpp: {
      commit: report.reference.commit,
      sha256: report.reference.server_sha256,
      library_sha256: report.reference.library_sha256,
    },
  };
  const scenarios = gemmaWorkloads.map(({ body, ...scenario }) => ({
    ...scenario,
    measured_at: report.completed_at,
    zinc: createStats(
      'ZINC',
      report.samples.filter(
        (sample) => !sample.warmup && sample.engine === 'zinc' && sample.scenario === scenario.id,
      ),
    ),
    baseline: createStats(
      'llama.cpp',
      report.samples.filter(
        (sample) => !sample.warmup && sample.engine === 'llama' && sample.scenario === scenario.id,
      ),
    ),
  }));
  const result = mergeArtifacts(
    existing,
    [
      {
        ...target,
        generated_at: report.completed_at,
        provenance,
        source: 'tools/benchmark_gemma_rocm.py',
        methodology: {
          ...target.methodology,
          runner: 'Persistent server vs persistent server; model-specific recorded runners',
          notes: [
            ...new Set([
              ...target.methodology.notes,
              'Gemma 26B-A4B was revalidated with default serial prefill, six complete CLI reference continuations and eighteen HTTP/recovery continuations. Its new off/off measurements use nine measured requests per workload and engine.',
            ]),
          ],
        },
        models: [
          {
            ...previous,
            scenarios,
            provenance,
            generated_at: report.completed_at,
            correctness_review: null,
            measurement_history: sameCampaign ? history : [...history, old],
            server_context_tokens: 4096,
            server_kv_cache: { zinc: 'f32', llama_cpp: 'f32' },
            revalidation_evidence: {
              url: evidenceUrl,
              prefill: 'serial',
              exact_cli_tokens: 96,
              http_full_continuations: 18,
              release_qualification: false,
            },
          },
        ],
      },
    ],
    { preserveMissingPhases: true },
  );
  if (result.correctness_review) {
    result.correctness_review = {
      ...result.correctness_review,
      reviewed_at: report.completed_at,
      initial_affected_model_rows:
        result.correctness_review.initial_affected_model_rows ??
        result.correctness_review.affected_model_rows,
      affected_model_rows: result.targets
        .flatMap((target) => target.models)
        .filter((model) => model.correctness_review?.tone === 'caution').length,
    };
  }
  return result;
}

if (import.meta.main) {
  const [file, commit] = process.argv.slice(2);
  requireValue(
    file && commit,
    'Usage: bun tools/merge_gemma_rocm.mjs <report.json> <harness-commit>',
  );
  const report = JSON.parse(await readFile(file, 'utf8'));
  validateGemmaHarness(report, commit);
  report.harness_source_commit = commit;
  const name = report.completed_at.slice(0, 10) + '-gemma-rocm-revalidated.json';
  const url = '/zinc/benchmarks/' + name;
  const siteFile = 'site/src/data/zinc-performance.json';
  const result = mergeGemmaReport(JSON.parse(await readFile(siteFile, 'utf8')), report, url);
  requireValue(
    !/\/(?:Users|root|home)\/|192\.168\.\d+\.\d+|\.ts\.net\b/.test(
      JSON.stringify({ report, result }),
    ),
    'Public evidence contains private host data',
  );
  await mkdir('site/public/zinc/benchmarks', { recursive: true });
  await writeFile(
    path.join('site/public/zinc/benchmarks', name),
    JSON.stringify(report, null, 2) + '\n',
  );
  await writeFile(siteFile, JSON.stringify(result, null, 2) + '\n');
  console.log(
    'Published revalidated ROCm Gemma measurements; original caution remains in measurement history.',
  );
}
