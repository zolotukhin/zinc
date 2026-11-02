#!/usr/bin/env bun
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import referencePin from '../config/benchmark-reference.json' with { type: 'json' };
import releasePolicy from '../config/release/targets.json' with { type: 'json' };
import { mtpModelId, mtpWorkloads } from './mtp_workloads.mjs';
import {
  createStats,
  mergeArtifacts,
  parseLlamaCppVersionOutput,
  requireObservedMtp,
} from './performance_suite.mjs';
import { requireBenchmarkReference } from './benchmark_speculation.mjs';

const requiredHarness = [
  'tools/benchmark_matched_mtp.py',
  'tools/parse_mtp_sample.mjs',
  'tools/mtp_workloads.mjs',
  'tools/performance_suite.mjs',
  'tools/benchmark_speculation.mjs',
  'tools/release_gpu.py',
  'tools/release_lib.py',
  'config/benchmark-reference.json',
];
const modelPin = releasePolicy.models.find((model) => model.id === mtpModelId);
const digest = (value) => /^[a-f0-9]{64}$/.test(value ?? '');
const requireValue = (condition, message) => {
  if (!condition) throw new Error(message);
};

// Match the frozen Python runner's json.dumps(body, sort_keys=True).
export function requestDigest(body) {
  const canonical = (value) => {
    if (Array.isArray(value)) return '[' + value.map(canonical).join(', ') + ']';
    if (value && typeof value === 'object')
      return (
        '{' +
        Object.keys(value)
          .sort()
          .map((key) => JSON.stringify(key) + ': ' + canonical(value[key]))
          .join(', ') +
        '}'
      );
    return JSON.stringify(value).replace(
      /[\u007f-\uffff]/g,
      (character) => '\\u' + character.charCodeAt(0).toString(16).padStart(4, '0'),
    );
  };
  return createHash('sha256').update(canonical(body)).digest('hex');
}

export function validateHarnessSources(report, commit) {
  requireValue(/^[a-f0-9]{40}$/.test(commit ?? ''), 'Supply the full frozen --harness-commit');
  const root = fileURLToPath(new URL('../', import.meta.url));
  for (const [name, expected] of Object.entries(report.harness_sha256 ?? {})) {
    requireValue(
      /^(?:tools\/[a-z_]+\.(?:mjs|py)|config\/benchmark-reference\.json)$/.test(name),
      'Unknown harness source path',
    );
    const source = execFileSync('git', ['show', commit + ':' + name], { cwd: root });
    requireValue(
      createHash('sha256').update(source).digest('hex') === expected,
      'Frozen harness source differs: ' + name,
    );
  }
}

export function validateMatchedReport(report, { includeDrift = false } = {}) {
  requireValue(report.schema_version === 'zinc-matched-mtp-v1', 'Unknown campaign schema');
  requireValue(
    report.protocol_passed === true && !report.error,
    'The complete protocol did not pass',
  );
  requireValue(
    report.passed === true || (includeDrift && report.output_drift?.length > 0),
    'A failed output control needs explicit diagnostic inclusion',
  );
  requireValue(['vulkan', 'rocm'].includes(report.backend), 'Unknown native backend');
  requireValue(
    report.model?.id === mtpModelId &&
      report.model.sha256 === modelPin.sha256 &&
      report.model.size_bytes === modelPin.size_bytes,
    'Model identity differs',
  );
  requireBenchmarkReference(
    { commit: report.reference?.commit, sha256: report.reference?.server_sha256 },
    referencePin.commit,
  );
  requireValue(
    digest(report.reference?.tokenizer_sha256) &&
      Object.keys(report.reference.library_sha256 ?? {}).some((key) => key.startsWith('libggml')) &&
      Object.values(report.reference.library_sha256).every(digest),
    'Reference libraries were not identified',
  );
  requireValue(
    /^[a-f0-9]{40}$/.test(report.zinc?.commit ?? '') && digest(report.zinc?.sha256),
    'ZINC was not identified',
  );
  requireValue(
    requiredHarness.every((name) => digest(report.harness_sha256?.[name])) &&
      Object.keys(report.harness_sha256 ?? {}).length === requiredHarness.length &&
      /^[a-f0-9]{40}$/.test(report.harness_source_commit ?? '') &&
      Object.values(report.harness_sha256).every(digest),
    'Harness was not identified',
  );
  requireValue(
    Number.isFinite(Date.parse(report.completed_at)),
    'Campaign completion date is missing',
  );
  const settings = report.settings;
  requireValue(
    settings?.context === 4096 &&
      settings.slots === 1 &&
      settings.kv_cache === 'f32' &&
      settings.temperature === 0 &&
      settings.seed === 0 &&
      settings.top_p === 1 &&
      settings.prompt_cache === false,
    'Campaign settings differ from the controlled protocol',
  );
  requireValue(
    [settings.rounds, settings.runs, settings.warmup, settings.drafts].every(
      Number.isSafeInteger,
    ) &&
      settings.rounds >= 3 &&
      settings.runs >= 3 &&
      settings.warmup >= 1 &&
      settings.drafts === 3,
    'The campaign lacks the full repeated matrix',
  );
  requireValue(
    report.scenarios?.length === mtpWorkloads.length,
    'A diagnostic subset cannot be published as a full campaign',
  );
  for (const scenario of mtpWorkloads)
    requireValue(
      report.scenarios.find((row) => row.id === scenario.id)?.request_sha256 ===
        requestDigest(scenario.body),
      'A workload request differs from the canonical payload',
    );
  const expectedCount =
    4 * mtpWorkloads.length * settings.rounds * (settings.warmup + settings.runs);
  requireValue(report.samples?.length === expectedCount, 'The sample matrix is incomplete');
  const identities = new Map();
  const observedDrift = [];
  for (const scenario of mtpWorkloads) {
    const promptCounts = [];
    for (const engine of ['zinc', 'llama']) {
      const hashes = {};
      for (const mode of ['off', 'on']) {
        const rows = report.samples.filter(
          (sample) =>
            sample.engine === engine && sample.mtp === mode && sample.scenario === scenario.id,
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
          requireObservedMtp(sample, mode);
          requireValue(
            sample.generatedTokens === scenario.max_tokens && sample.finish_reason === 'length',
            'Output budgets differ',
          );
          requireValue(
            Number.isSafeInteger(sample.promptTokens) &&
              sample.promptTokens > 0 &&
              sample.prefillTokens >= sample.promptTokens - 1,
            'Prompt prefill was reduced or unrecorded',
          );
          requireValue(
            [sample.prefillTps, sample.decodeTps, sample.totalLatencyMs].every(
              (value) => typeof value === 'number' && Number.isFinite(value) && value > 0,
            ),
            'Timing telemetry is incomplete',
          );
          requireValue(
            digest(sample.content_sha256) && sample.content_bytes > 0,
            'Output was not captured',
          );
          if (mode === 'on')
            requireValue(
              sample.speculative.accepted >= 0 &&
                sample.speculative.accepted <= sample.speculative.drafted,
              'Invalid acceptance counts',
            );
          promptCounts.push(sample.promptTokens);
        }
        requireValue(
          new Set(rows.map((sample) => sample.content_sha256)).size === 1,
          'Repeated outputs were not deterministic',
        );
        hashes[mode] = rows[0].content_sha256;
      }
      identities.set(engine + '/' + scenario.id, hashes);
      if (hashes.off !== hashes.on)
        observedDrift.push({
          engine,
          scenario: scenario.id,
          off_sha256: hashes.off,
          on_sha256: hashes.on,
        });
    }
    requireValue(
      Math.max(...promptCounts) - Math.min(...promptCounts) <= 1,
      'Prompt counts differ',
    );
  }
  const sorted = (values) =>
    values
      .map((item) => JSON.stringify([item.engine, item.scenario, item.off_sha256, item.on_sha256]))
      .sort()
      .join('\n');
  requireValue(
    sorted(observedDrift) === sorted(report.output_drift ?? []),
    'Output-control failures were not recorded accurately',
  );
  requireValue(
    report.passed === (observedDrift.length === 0),
    'Campaign pass flag contradicts its output controls',
  );
  return identities;
}

export function mergeMatchedReports(
  existing,
  reports,
  { includeDrift = false, evidenceUrls = [] } = {},
) {
  requireValue(
    new Set(reports.map((report) => report.backend)).size === reports.length,
    'Duplicate backend report',
  );
  let result = existing;
  for (const [index, report] of reports.entries()) {
    const identities = validateMatchedReport(report, { includeDrift });
    const targetId = report.backend === 'rocm' ? 'rdna-rocm' : 'rdna';
    const originalTarget = result.targets.find((target) => target.id === targetId);
    const originalModel = originalTarget?.models.find((model) => model.id === mtpModelId);
    requireValue(originalModel, 'No published target/model matches this campaign');
    const { measurement_history: history = [], ...previousMeasurement } = originalModel;
    const sameCampaign =
      originalModel.generated_at === report.completed_at &&
      originalModel.provenance?.llama_cpp?.commit === report.reference.commit &&
      originalModel.provenance?.zinc?.sha256 === report.zinc.sha256;
    const measurementHistory = sameCampaign
      ? history
      : [
          ...history,
          {
            ...previousMeasurement,
            generated_at: previousMeasurement.generated_at ?? originalTarget.generated_at,
            provenance: previousMeasurement.provenance ?? originalTarget.provenance,
          },
        ];
    const provenance = {
      harness: { commit: report.harness_source_commit, path: 'tools/benchmark_matched_mtp.py' },
      zinc: {
        commit: report.zinc.commit,
        version: report.zinc.commit.slice(0, 12),
        sha256: report.zinc.sha256,
      },
      llama_cpp: {
        commit: report.reference.commit,
        version: parseLlamaCppVersionOutput(report.reference.version)?.version ?? null,
        sha256: report.reference.server_sha256,
        library_sha256: report.reference.library_sha256,
      },
    };
    const scenarios = mtpWorkloads.map(({ body, ...scenario }) => {
      const measurement = (mode) => {
        const rawStats = (engine) =>
          createStats(
            engine === 'zinc' ? 'ZINC' : 'llama.cpp',
            report.samples.filter(
              (sample) =>
                !sample.warmup &&
                sample.engine === engine &&
                sample.mtp === mode &&
                sample.scenario === scenario.id,
            ),
          );
        const stats = (engine) => ({
          ...rawStats(engine),
          prefill_timing_scope:
            mode === 'on' && engine === 'llama' ? 'target-and-draft-prefill' : 'target-prefill',
        });
        const drift =
          mode === 'on' && report.output_drift.some((entry) => entry.scenario === scenario.id);
        return {
          zinc: stats('zinc'),
          baseline: stats('llama'),
          measured_at: report.completed_at,
          provenance,
          correctness_review: drift
            ? {
                tone: 'caution',
                label: 'MTP output differs',
                note: 'At least one engine changed its complete greedy response between MTP modes. Raw timings are diagnostic; the exact-output control failed.',
                source_url: evidenceUrls[index],
              }
            : undefined,
          correctness_controls: {
            repeated_outputs_identical: true,
            mtp_output_identical: !drift,
            cross_engine_output_identical:
              identities.get('zinc/' + scenario.id)[mode] ===
              identities.get('llama/' + scenario.id)[mode],
          },
        };
      };
      return { ...scenario, ...measurement('on'), variants: { mtp_off: measurement('off') } };
    });
    const incoming = {
      ...originalTarget,
      generated_at: report.completed_at,
      source: 'tools/benchmark_matched_mtp.py',
      provenance,
      methodology: {
        ...originalTarget.methodology,
        runner: 'Persistent server vs persistent server; no CLI fallback (matched 27B MTP refresh)',
        zinc_backend: report.backend,
        llama_backend: report.backend === 'rocm' ? 'ROCm (HIP)' : 'Vulkan',
        runs: report.settings.rounds * report.settings.runs,
        warmup_runs: report.settings.warmup,
        notes: [
          'The Qwen 3.8 27B refresh uses matched MTP on/on and off/off, F32 K/V, one slot, 4K context and nine measured requests per treatment. Other models retain their own original dates and provenance.',
        ],
      },
      models: [
        {
          ...originalModel,
          measurement_history: measurementHistory,
          scenarios,
          provenance,
          generated_at: report.completed_at,
          server_context_tokens: 4096,
          server_kv_cache: { zinc: 'f32', llama_cpp: 'f32' },
          matched_mtp_evidence: {
            url: evidenceUrls[index],
            protocol_passed: true,
            output_controls_passed: report.passed,
            release_qualification: false,
            drafts: report.settings.drafts,
          },
        },
      ],
    };
    result = mergeArtifacts(result, [incoming], { preserveMissingPhases: true });
  }
  return result;
}

if (import.meta.main) {
  const args = {
    reports: [],
    site: 'site/src/data/zinc-performance.json',
    evidence: 'site/public/zinc/benchmarks',
    includeDrift: false,
    harnessCommit: null,
  };
  for (let i = 2; i < process.argv.length; i++) {
    const flag = process.argv[i];
    if (flag === '--report') args.reports.push(process.argv[++i]);
    else if (flag === '--site') args.site = process.argv[++i];
    else if (flag === '--evidence-dir') args.evidence = process.argv[++i];
    else if (flag === '--include-diagnostics') args.includeDrift = true;
    else if (flag === '--harness-commit') args.harnessCommit = process.argv[++i];
    else throw new Error('Unknown argument: ' + flag);
  }
  requireValue(args.reports.length > 0, 'Supply --report for each completed campaign');
  const reports = await Promise.all(
    args.reports.map(async (file) => JSON.parse(await readFile(file, 'utf8'))),
  );
  for (const report of reports) {
    validateHarnessSources(report, args.harnessCommit);
    report.harness_source_commit = args.harnessCommit;
  }
  const names = reports.map(
    (report) => report.completed_at.slice(0, 10) + '-matched-mtp-' + report.backend + '.json',
  );
  const evidenceUrls = names.map((name) => '/zinc/benchmarks/' + name);
  const existing = JSON.parse(await readFile(args.site, 'utf8'));
  const merged = mergeMatchedReports(existing, reports, {
    includeDrift: args.includeDrift,
    evidenceUrls,
  });
  // Logs, argv, hosts and source paths never enter this public report schema.
  const serialized = JSON.stringify({ reports, merged });
  requireValue(
    !/\/(?:Users|home|root)\/|192\.168\.\d+\.\d+|\.ts\.net\b/.test(serialized),
    'Public evidence contains host-specific data',
  );
  await mkdir(args.evidence, { recursive: true });
  for (const [index, report] of reports.entries())
    await writeFile(path.join(args.evidence, names[index]), JSON.stringify(report, null, 2) + '\n');
  await writeFile(args.site, JSON.stringify(merged, null, 2) + '\n');
  console.log(
    'Published ' +
      reports.length +
      ' matched MTP campaigns; failed output controls remain excluded.',
  );
}
