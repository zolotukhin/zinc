import { expect, test } from 'bun:test';
import { readFile } from 'node:fs/promises';
import { resolve } from 'node:path';
import { renderLlmsTxt, renderLlmsFullTxt } from '../src/lib/agent-docs';
import {
  docsReleaseStatus,
  STABLE_RELEASE,
  STABLE_VALIDATION_PATH,
} from '../src/lib/docs-release-status';
import type { ZigApiIndex } from '../src/lib/zig-api-loader';

const publishedRelease = await Bun.file(
  resolve(import.meta.dir, '../public' + STABLE_VALIDATION_PATH),
).json();

const index = {
  sections: [],
  moduleCount: 1,
  codeLineCount: 10,
  exportCount: 1,
  memberCount: 1,
  lastmod: '2026-11-15',
} as unknown as ZigApiIndex;

test('stable client guidance follows the published maintenance certificate', () => {
  expect(publishedRelease.published).toBe(true);
  expect(publishedRelease.status).toBe('published');
  expect(STABLE_RELEASE).toBe(publishedRelease.version);
  expect(publishedRelease.publication.status).toBe('passed');
  expect(publishedRelease.publication.public_install_jobs).toHaveLength(2);
});

test('active OpenCode install recipes match the final-archive client cohort', async () => {
  const pins = [
    ...new Set(publishedRelease.supplementary.actual_clients.map((row: any) => row.opencode)),
  ];
  expect(pins).toHaveLength(1);
  for (const name of ['README.md', 'docs/OPENCODE.md', 'site/src/pages/zinc/index.astro']) {
    const source = await readFile(resolve(import.meta.dir, '../..', name), 'utf8');
    const recipes = [...source.matchAll(/@opencode\/cli@(\d+\.\d+\.\d+)/g)].map(
      (match) => match[1],
    );
    expect(recipes.length, name).toBeGreaterThan(0);
    expect([...new Set(recipes)], name).toEqual(pins);
  }
});

test('both agent indexes put installed client setup before implementation references', () => {
  for (const render of [renderLlmsTxt, renderLlmsFullTxt]) {
    const text = render(index, [], 'https://example.test/');
    expect(text).toContain('https://example.test/zinc/docs/running-zinc/');
    expect(text).not.toContain('/zinc/docs/agent-setup/');
    expect(text).not.toContain('/zinc/docs/binary-releases/');
    expect(text.indexOf('## Run ZINC or connect a client')).toBeLessThan(
      text.indexOf('ZINC Zig API:') < 0
        ? text.indexOf('Zig API root:')
        : text.indexOf('ZINC Zig API:'),
    );
    expect(text).toContain(
      'zinc serve --model-id qwen35-9b-q4k-m -c 16384 --parallel 1 --port 9090',
    );
    expect(text).toContain('http://localhost:9090/v1/models');
    expect(text).toContain('standalone helper');
    expect(text).toContain(`Stable Linux release: ${STABLE_RELEASE}`);
    expect(text).toContain('AMD RDNA: prefer HIP / ROCm');
    expect(text).toContain('Intel Arc uses ZINC_BACKEND=vulkan');
    expect(text).toContain('https://example.test' + STABLE_VALIDATION_PATH);
    expect(text).toContain('Structured discovery (0.1.1+)');
    expect(text).toContain('are released features');
    expect(text).not.toContain('Development additions');
    expect(text).toContain('absent capabilities are unknown');
    expect(text).toContain('AGENTS.md');
    expect(text).not.toContain('prefer the generated Zig API reference first');
  }
});

test('release notices distinguish stable installation, mixed guides and current source', () => {
  for (const slug of ['getting-started', 'openclaw', 'opencode']) {
    expect(docsReleaseStatus(slug).channel).toBe('stable');
  }
  for (const slug of ['api', 'running-zinc', 'hardware-requirements', 'rocm']) {
    expect(docsReleaseStatus(slug).channel).toBe('mixed');
  }
  expect(docsReleaseStatus('zig-api/compute-forward').channel).toBe('development');
});

test('both agent indexes scope native lifecycle controls by release and live discovery', () => {
  for (const render of [renderLlmsTxt, renderLlmsFullTxt]) {
    const text = render(index, [], 'https://example.test/');
    expect(text).toContain(
      'Stable 0.1.1 HIP provides CLI/HTTP chat without browser UI or model switching',
    );
    expect(text).toContain('Current HIP source adds browser chat');
    expect(text).toContain(
      'current HIP/CUDA source advertises installed-model switching and unload',
    );
    expect(text).toContain('Enable controls through live capability discovery');
    expect(text).not.toContain('native HIP/CUDA do not offer hot model management');
  }
});

const candidatePath = '/zinc/validation/release-candidate-0.1.2-amd-2026-11-18.json';
const candidate = await Bun.file(resolve(import.meta.dir, '../public' + candidatePath)).json();
// These pin the coverage and observations of one immutable, dated snapshot.
// A new run gets a separate record; arithmetic and doc rows are derived below.
const candidateCoverage = {
  source: '588db7f4c928fa89756aaefcfafaadb2f418a621',
  lifecycle: { operations: 52, references: 156, models: 13 },
  clients: { cohorts: 12, api: 322, code: 36, claw: 36 },
  gemmaLoop: { vulkanRejected: 10, vulkanSteps: 16, hipRejected: 3 },
};

const latestCandidatePath = '/zinc/validation/release-candidate-0.1.2-amd-2026-11-19.json';
const latestCandidate = await Bun.file(
  resolve(import.meta.dir, '../public' + latestCandidatePath),
).json();
const latestCoverage = {
  source: 'd5d339fac52e3de1e2c214bea2721f56fa8d1cb9',
  lifecycle: { operations: 52, references: 156, models: 13 },
  clients: { cohorts: 12, api: 322, code: 36, claw: 36 },
};
const candidateSnapshots = [
  ['2026-11-18', candidate, candidatePath, candidateCoverage],
  ['2026-11-19', latestCandidate, latestCandidatePath, latestCoverage],
] as const;

test.each(candidateSnapshots)(
  'AMD preparation remains a partial snapshot, separate from the stable certificate (%s)',
  (_label, candidate, candidatePath, candidateCoverage) => {
    // Two passing AMD profiles cannot stand in for the required Intel profile.
    // Keep this guard after publication too: the dated partial record is historical.
    expect(candidate.kind).toBe('amd-release-preparation');
    expect(candidate.certified).toBe(false);
    expect(candidate.published).toBe(false);
    expect(candidate.version).toBe('0.1.2');
    expect(candidate.source).toBe(candidateCoverage.source);
    expect(candidate.manifest.commit).toBe(candidate.source);
    expect(candidate.manifest.version).toBe(candidate.version);
    expect(String(candidate.manifest.candidate_run)).toBe(candidate.candidate_run);
    const present = new Set(candidate.hardware_reports.map((row: any) => row.profile));
    const missing = candidate.manifest.required_profiles
      .map((row: any) => row.id)
      .filter((id: string) => !present.has(id));
    expect([...present].sort()).toEqual(['rocm-rdna4', 'vulkan-rdna4']);
    expect(missing).toEqual(['vulkan-intel-xe2']);
    expect(candidate.missing_profiles).toEqual(missing);
    expect(STABLE_VALIDATION_PATH).not.toBe(candidatePath);
  },
);

test.each(candidateSnapshots)(
  'candidate AMD lifecycle totals, memory growth and archive identities agree (%s)',
  (_label, candidate, candidatePath, candidateCoverage) => {
    let operations = 0;
    let references = 0;
    let models = 0;
    for (const report of candidate.hardware_reports) {
      const artifact = candidate.manifest.artifacts.find(
        (row: any) => row.target === report.target,
      );
      expect(report.commit).toBe(candidate.source);
      expect(report.policy_sha256).toBe(candidate.manifest.policy_sha256);
      expect(report.archive_sha256).toBe(artifact.sha256);
      expect(report.binary_sha256).toBe(artifact.binary_sha256);
      expect(Object.values(report.checks).every((value) => value === true)).toBe(true);
      expect(report.models.every((model: any) => model.passed)).toBe(true);
      const soak = report.lifecycle_soak;
      const rounds = soak.options.warmup + soak.options.cycles;
      expect(soak.status).toBe('passed');
      expect(soak.context).toBe(4096);
      expect(soak.options.warmup).toBe(1);
      expect(soak.completed_cycles).toBe(3);
      expect(soak.samples).toHaveLength(4);
      expect(soak.operations).toBe(rounds * report.models.length);
      expect(soak.direct_switches).toBe(rounds * (report.models.length - 1));
      expect(soak.reference_cases).toBe(soak.operations * 3);
      expect(soak.binary_sha256).toBe(artifact.binary_sha256);
      expect(soak.cleanup).toBe(true);
      for (const key of ['rss_bytes', 'vram_bytes', 'gpu_host_bytes', 'fds']) {
        const growth = Math.max(
          0,
          ...soak.samples.slice(1).map((sample: any) => sample[key] - soak.samples[0][key]),
        );
        expect(soak.max_growth[key]).toBe(growth);
        expect(growth).toBeLessThanOrEqual(soak.options.max_growth[key]);
      }
      operations += soak.operations;
      references += soak.reference_cases;
      models += report.models.length;
    }
    expect({ operations, references, models }).toEqual(candidateCoverage.lifecycle);
    const native = candidate.hardware_reports.find((row: any) => row.profile === 'rocm-rdna4');
    expect(native.allocation_failures).toHaveLength(2);
    for (const probe of native.allocation_failures) {
      expect(probe.status).toBe('passed');
      expect(probe.response_status).toBe(500);
      expect(probe.restored && probe.continuation_matches && probe.cleanup).toBe(true);
      expect(probe.binary_sha256).toBe(native.binary_sha256);
    }
  },
);

test.each(candidateSnapshots)(
  'candidate client scores are recomputed from independent checks on the frozen bytes (%s)',
  (_label, candidate, candidatePath, candidateCoverage) => {
    const { campaigns, summary } = candidate.actual_clients;
    const pairs = new Set();
    let api = 0;
    let code = 0;
    let claw = 0;
    for (const row of campaigns) {
      const artifact = candidate.manifest.artifacts.find(
        (item: any) => item.target === 'linux-x86_64-' + row.backend,
      );
      expect(row.commit).toBe(candidate.source);
      expect(row.trial).toBe('');
      expect(String(row.candidate_run)).toBe(candidate.candidate_run);
      expect(row.archive_sha256).toBe(artifact.sha256);
      expect(row.binary_sha256).toBe(artifact.binary_sha256);
      const profile = candidate.hardware_reports.find(
        (report: any) => report.target === artifact.target,
      );
      const model = profile.models.find((entry: any) => entry.id === row.model);
      expect(row.model_sha256).toBe(model.sha256);
      expect(row.hardware.selection_verified).toBe(true);
      expect(row.model).not.toBe('qwen3-0.6b-q8-0');
      expect(row.context).toBe(row.model === 'gemma4-31b-q4k-m' ? 8192 : 16384);
      expect(row.parallel).toBe(1);
      expect(row.gpu_idle_after && row.passed).toBe(true);
      pairs.add(row.backend + ':' + row.model);
      for (const [name, version] of Object.entries(summary.client_versions)) {
        expect(row.client_versions[name].version).toBe(version);
        expect(row.client_versions[name].executable_sha256).toBe(
          summary.client_executable_sha256[name],
        );
      }
      expect(row.harness_sha256).toEqual(summary.helpers_sha256);
      expect(row.api.complete).toBe(true);
      expect(row.api.checks.every((check: any) => check.passed)).toBe(true);
      expect(row.api.total).toBe(row.api.checks.length);
      expect(row.api.passed).toBe(row.api.total);
      api += row.api.total;
      const openclaw = row.clients.find((client: any) => client.client === 'OpenClaw');
      expect(openclaw.cases.map((task: any) => task.name).sort()).toEqual([
        'recovery',
        'text',
        'workspace',
      ]);
      expect(openclaw.healthy_after && openclaw.passed).toBe(true);
      for (const task of openclaw.cases) {
        expect(task.passed && task.completed).toBe(true);
        if (task.name === 'workspace')
          expect(Object.values(task.checks).every((v) => v === true)).toBe(true);
        claw++;
      }
      const opencode = row.clients.filter((client: any) => client.client === 'OpenCode');
      expect(opencode.map((client: any) => client.fixture).sort()).toEqual([
        'cart-multi-file',
        'python-log-cli',
        'rate-limiter-single',
      ]);
      for (const client of opencode) {
        expect(client.healthy_after && client.passed).toBe(true);
        expect(client.generation_recovery.passed).toBe(true);
        expect(client.cases).toHaveLength(1);
        const task = client.cases[0];
        // A zero-test exit must never turn a model delivery into a pass.
        const expectedTests = client.fixture === 'python-log-cli' ? 4 : 3;
        expect(task.baseline.exitCode).toBeGreaterThan(0);
        expect(task.baseline.tests.fail).toBeGreaterThan(0);
        expect(task.final_tests.exitCode).toBe(0);
        expect(task.final_tests.tests).toEqual({ pass: expectedTests, fail: 0 });
        expect(task.client_exit).toBe(0);
        expect(client.generation_recovery.deadline_seconds).toBe(15);
        expect(client.generation_recovery.seconds).toBeLessThanOrEqual(15);
        expect(task.protected_files_unchanged).toBe(true);
        expect(task.unexpected_files + task.proxy_repaired_events).toBe(0);
        expect(task.timeout).toBe(false);
        code++;
      }
    }
    expect(pairs.size).toBe(candidateCoverage.clients.cohorts);
    expect({ api, code, claw }).toEqual({
      api: candidateCoverage.clients.api,
      code: candidateCoverage.clients.code,
      claw: candidateCoverage.clients.claw,
    });
    expect(summary.api).toEqual({ passed: api, total: api });
    expect(summary.OpenCode).toEqual({ passed: code, total: code });
    expect(summary.OpenClaw).toEqual({ passed: claw, total: claw });
    expect(summary.complete && summary.all_passed).toBe(true);
  },
);

test('superseded failures and matched helper follow-ups do not become current-archive passes', () => {
  const old = candidate.superseded_attempt;
  expect(old.source).not.toBe(candidate.source);
  expect(old.summary.OpenCode).toEqual({ passed: 35, total: 36 });
  expect(old.summary.all_passed).toBe(false);
  const failed = old.failed_gemma31_hip.clients.find(
    (row: any) => row.fixture === 'python-log-cli',
  );
  expect(failed.passed).toBe(false);
  expect(failed.cases[0].final_tests.tests).toEqual({ pass: 0, fail: 4 });
  const followups = candidate.matched_gemma_follow_up.campaigns;
  expect(followups).toHaveLength(2);
  for (const row of candidate.actual_clients.campaigns) expect(row.trial).toBe('');
  for (const row of followups) {
    expect(row.commit).toBe(old.source);
    expect(row.harness_commit).not.toBe(row.commit);
    expect(row.trial).toBe('compact-budget');
    expect(row.passed).toBe(true);
  }
});

test('passing Gemma acceptance retains redundant edits and missing agent verification', () => {
  // The independent evaluator passed the code even when the agent used its
  // remaining steps on identical edits. Do not silently turn that into a claim
  // that every agent completed verification or followed an efficient workflow.
  const trace = candidate.actual_clients.delivery_trace_audit;
  expect(trace.complete).toBe(true);
  expect(trace.fixture_traces).toBe(candidateCoverage.clients.code);
  expect(trace.observed_delivery_limits).toHaveLength(2);
  const vk = trace.observed_delivery_limits.find((row: any) => row.backend === 'vulkan');
  expect(vk.model).toBe('gemma4-26b-a4b-q4k-m');
  expect(vk.acceptance_passed).toBe(true);
  expect(vk.rejected_identical_edits).toBe(candidateCoverage.gemmaLoop.vulkanRejected);
  expect(vk.steps_started).toBe(candidateCoverage.gemmaLoop.vulkanSteps);
  expect(vk.shell_calls_after_final_file_mutation).toBe(0);
  expect(vk.independent_tests).toEqual({ pass: 3, fail: 0 });
  const hip = trace.observed_delivery_limits.find((row: any) => row.backend === 'rocm');
  expect(hip.rejected_identical_edits).toBe(candidateCoverage.gemmaLoop.hipRejected);
  expect(hip.recognized_test_commands_after_final_file_mutation).toBe(1);
});

test.each(candidateSnapshots)(
  'every coding trace agrees with acceptance and retains delivery limits (%s)',
  (_label, candidate, _candidatePath, candidateCoverage) => {
    const audit = candidate.actual_clients.delivery_trace_audit;
    expect(audit.source).toBe(candidate.source);
    expect(audit.candidate_run).toBe(candidate.candidate_run);
    expect(audit.complete).toBe(true);
    expect(audit.completed_cohorts).toBe(candidateCoverage.clients.cohorts);
    expect(audit.fixture_traces).toBe(candidateCoverage.clients.code);
    expect(audit.fixtures).toHaveLength(audit.fixture_traces);
    const keys = new Set();
    for (const trace of audit.fixtures) {
      const key = [trace.backend, trace.model, trace.fixture].join(':');
      expect(keys.has(key), key).toBe(false);
      keys.add(key);
      const cohort = candidate.actual_clients.campaigns.find(
        (row: any) => row.backend === trace.backend && row.model === trace.model,
      );
      const delivery = cohort.clients.find((row: any) => row.fixture === trace.fixture);
      const task = delivery.cases[0];
      expect(trace.acceptance_passed).toBe(delivery.passed);
      expect(trace.independent_tests).toEqual(task.final_tests.tests);
      expect(trace.tool_calls).toBe(task.tool_calls);
      for (const name of [
        'timeout',
        'client_exit',
        'protected_files_unchanged',
        'unexpected_files',
        'proxy_repaired_events',
      ]) {
        expect(trace[name], key + ':' + name).toBe(task[name]);
      }
      // Acceptance can pass despite agent inefficiency or omitted shell checks.
      // Those observations must remain visible in the summarized limitations.
      if (
        trace.rejected_identical_edits >= 3 ||
        (trace.successful_file_mutations > 0 && trace.shell_calls_after_final_file_mutation === 0)
      ) {
        expect(audit.observed_delivery_limits).toContainEqual(
          expect.objectContaining({
            backend: trace.backend,
            model: trace.model,
            fixture: trace.fixture,
            rejected_identical_edits: trace.rejected_identical_edits,
            shell_calls_after_final_file_mutation: trace.shell_calls_after_final_file_mutation,
          }),
        );
      }
    }
    for (const name of ['tool_errors', 'rejected_identical_edits']) {
      expect(audit[name]).toBe(
        audit.fixtures.reduce((total: number, row: any) => total + row[name], 0),
      );
    }
  },
);

test.each(candidateSnapshots)(
  'both installer backends retain distinct upgrade, rollback and repair controls (%s)',
  (_label, candidate, candidatePath, candidateCoverage) => {
    const install = candidate.installation;
    expect(install.source).toBe(candidate.source);
    expect(String(install.candidate_run)).toBe(candidate.candidate_run);
    expect(install.passed).toBe(true);
    expect(install.checks).toHaveLength(10);
    expect(new Set(install.checks.map((row: any) => row.backend + ':' + row.case)).size).toBe(10);
    for (const row of install.checks) {
      expect(row.passed && row.cache_preserved && row.command_link_matches_current).toBe(true);
      expect(row.version_command_exit_status).toBe(0);
      if (row.case === 'rollback') {
        const baseline = install.baseline.find((entry: any) => entry.backend === row.backend);
        expect(row.version).toBe('0.1.1');
        expect(row.source).toBe(baseline.source);
        expect(row.binary_sha256).toBe(baseline.binary_sha256);
      } else {
        const artifact = candidate.manifest.artifacts.find(
          (entry: any) => entry.target === 'linux-x86_64-' + row.backend,
        );
        expect(row.version).toBe(candidate.version);
        expect(row.source).toBe(candidate.source);
        expect(row.binary_sha256).toBe(artifact.binary_sha256);
      }
    }
  },
);

test('browser evidence scopes 64K to the earlier tested 27B preview', () => {
  const browser = candidate.browser_smoke;
  expect(browser.source).toBe(candidate.source);
  expect(browser.passed).toBe(true);
  expect(browser.health).toMatchObject({
    backend: 'rocm',
    context_length: 65536,
    model: 'qwen38-27b-q4k-m',
    mtp_request_control: true,
  });
  expect(browser.page_errors).toEqual([]);
  expect(
    browser.views.map((view: any) => view.width).sort((a: number, b: number) => a - b),
  ).toEqual([390, 1280]);
  for (const view of browser.views) {
    expect(view.status).toBe('Connected');
    expect(view.activeModel).toBe(browser.health.model);
    expect(view.composerEnabled && view.mtpControlEnabled).toBe(true);
    expect(view.hasHorizontalOverflow).toBe(false);
  }
});

test.each(candidateSnapshots)(
  'static chat UI inspection binds both exact candidate archives (%s)',
  (_label, candidate) => {
    const ui = candidate.embedded_ui;
    expect(ui.source).toBe(candidate.source);
    expect(ui.chat_ui_sha256).toMatch(/^[0-9a-f]{64}$/);
    expect(ui.archives.map((row: any) => row.archive).sort()).toEqual(
      candidate.manifest.artifacts.map((row: any) => row.filename).sort(),
    );
    for (const row of ui.archives) {
      const artifact = candidate.manifest.artifacts.find(
        (entry: any) => entry.filename === row.archive,
      );
      expect(row.binary_sha256).toBe(artifact.binary_sha256);
      expect(row.contains_exact_chat_ui).toBe(true);
    }
  },
);

test('candidate preparation tables follow the frozen evidence', async () => {
  const markdown = await readFile(
    resolve(import.meta.dir, '../../docs/releases/0.1.2-plan.md'),
    'utf8',
  );
  const lines = markdown.split(/\r?\n/);
  const cells = (line: string) =>
    line
      .split('|')
      .slice(1, -1)
      .map((cell) => cell.trim().replace(/`/g, ''));
  // These authored GFM tables contain plain labels and numeric cells. Compare
  // data cells rather than whitespace or formatter-dependent table padding.
  function rows(header: string) {
    const matches = lines.flatMap((line, index) =>
      line.trim().startsWith('|') && cells(line)[0] === header ? [index] : [],
    );
    expect(matches).toHaveLength(1);
    const first = matches[0];
    const result: string[][] = [];
    for (const line of lines.slice(first + 2)) {
      if (!line.trim().startsWith('|')) break;
      result.push(cells(line));
    }
    return result;
  }
  const label = (backend: string) => (backend === 'rocm' ? 'HIP / ROCm' : 'Vulkan');
  const growth = (bytes: number, decimals: number) =>
    bytes === 0 ? '0' : (bytes / 1048576).toFixed(decimals) + ' MiB';
  const profiles = [...latestCandidate.hardware_reports].sort((a: any, b: any) =>
    a.profile.localeCompare(b.profile),
  );
  expect(rows('R9700 backend')).toEqual(
    profiles.map((report: any) => {
      const s = report.lifecycle_soak;
      return [
        label(s.backend),
        String(report.models.length),
        String(s.operations),
        String(s.direct_switches),
        String(s.reference_cases),
        growth(s.max_growth.rss_bytes, 1),
        growth(s.max_growth.vram_bytes, 2),
      ];
    }),
  );
  expect(rows('Backend')).toEqual(
    profiles.map((report: any) => [label(report.lifecycle_soak.backend), report.archive_sha256]),
  );
  expect(markdown).toContain(latestCandidate.manifest_sha256);
  expect(markdown).toContain(latestCandidate.source);
  // Release plans are read on GitHub as well as through links from the site.
  // A site-root-relative download link would target github.com/zinc/... there.
  expect(markdown).toContain(`https://zolotukhin.ai${latestCandidatePath}`);
  const install = latestCandidate.installation.checks;
  expect(markdown).toContain(
    `**${install.filter((row: any) => row.passed).length}/${install.length}** isolated`,
  );
  const browserContext = candidate.browser_smoke.health.context_length.toLocaleString('en-US');
  expect(markdown.replace(/\s+/g, ' ')).toContain(
    `check on Qwen3.8 27B at ${browserContext}-token allocated capacity`,
  );

  const modelLabels = [
    ['qwen35-9b-q4k-m', 'Qwen3.5 9B'],
    ['qwen36-35b-a3b-q4k-xl', 'Qwen3.6 35B-A3B'],
    ['qwen36-27b-q4k-m', 'Qwen3.6 27B'],
    ['qwen38-27b-q4k-m', 'Qwen3.8 27B'],
    ['gemma4-26b-a4b-q4k-m', 'Gemma 4 26B-A4B'],
    ['gemma4-31b-q4k-m', 'Gemma 4 31B'],
  ];
  expect(rows('Model')).toEqual(
    modelLabels.map(([id, name]) => {
      const campaigns = ['rocm', 'vulkan'].map((backend) =>
        latestCandidate.actual_clients.campaigns.find(
          (row: any) => row.backend === backend && row.model === id,
        ),
      );
      expect(campaigns[0].context).toBe(campaigns[1].context);
      const scores = campaigns.map((row: any) => {
        const code = row.clients.filter((c: any) => c.client === 'OpenCode');
        const claw = row.clients.find((c: any) => c.client === 'OpenClaw').cases;
        return `${row.api.passed}/${row.api.total} · ${code.filter((c: any) => c.passed).length}/${code.length} · ${claw.filter((c: any) => c.passed).length}/${claw.length}`;
      });
      return [name, campaigns[0].context.toLocaleString('en-US'), ...scores];
    }),
  );
});

test('the failed RSS peak remains a failure even though its final sample is below budget', () => {
  const failed = latestCandidate.previous_failed_cut;
  expect(failed.source).not.toBe(latestCandidate.source);
  expect(failed.manifest.commit).toBe(failed.source);
  expect(String(failed.manifest.candidate_run)).toBe(failed.candidate_run);
  const report = failed.hardware_report;
  const artifact = failed.manifest.artifacts.find((row: any) => row.target === report.target);
  expect(report.commit).toBe(failed.source);
  expect(report.archive_sha256).toBe(artifact.sha256);
  expect(report.binary_sha256).toBe(artifact.binary_sha256);
  const soak = report.lifecycle_soak;
  expect(soak.status).toBe('failed');
  expect(soak.failure_stage).toBe('memory_plateau');
  const peak = Math.max(
    0,
    ...soak.samples.slice(1).map((row: any) => row.rss_bytes - soak.samples[0].rss_bytes),
  );
  const budget = soak.options.max_growth.rss_bytes;
  expect(budget).toBe(128 * 1024 * 1024);
  expect(peak).toBe(140365824);
  expect(soak.max_growth.rss_bytes).toBe(peak);
  expect(peak).toBeGreaterThan(budget);
  expect(soak.samples.at(-1).rss_bytes - soak.samples[0].rss_bytes).toBeLessThan(budget);
  for (const row of latestCandidate.hardware_reports) {
    expect(row.lifecycle_soak.options.max_growth).toEqual(soak.options.max_growth);
  }
});

test('longer development runs and CPU contracts do not inflate final-archive coverage', () => {
  const development = latestCandidate.longer_development;
  expect(development.final_archive_certification).toBe(false);
  expect(development.source).not.toBe(latestCandidate.source);
  expect(development.results.completed && development.independent_review.passed).toBe(true);
  for (const report of development.results.reports) {
    expect(report.source).toBe(development.source);
    expect(report.target).toBe('x86_64-linux-gnu.2.39');
    const soak = report.lifecycle;
    expect(soak.status).toBe('passed');
    expect(soak.cleanup).toBe(true);
    expect(soak.options).toMatchObject({ cycles: 8, warmup: 1 });
    expect(soak.completed_cycles).toBe(8);
    expect(soak.samples).toHaveLength(9);
    expect(soak.operations).toBe(9 * soak.models.length);
    expect(soak.reference_cases).toBe(soak.operations * 3);
    const final = latestCandidate.hardware_reports.find(
      (row: any) => row.profile === report.backend + '-rdna4',
    );
    expect(soak.binary_sha256).not.toBe(final.binary_sha256);
    expect(final.lifecycle_soak.operations).toBe(4 * final.models.length);
  }
  const contract = development.linux_cpu_contract;
  expect(contract.passed).toBe(true);
  const actual = contract.rows.find((row: any) => row.variant === 'actual');
  const disabled = contract.rows.find((row: any) => row.variant === 'disabled');
  expect(actual.exit_status).toBe(0);
  expect(actual.rss_returned_bytes).toBeGreaterThanOrEqual(32 * 1024 * 1024);
  expect(disabled.exit_status).toBeGreaterThan(0);
});

test('source CI is complete without turning a missing GPU profile into certification', () => {
  const ci = latestCandidate.post_merge_source_ci;
  expect(ci.source).toBe(latestCandidate.source);
  expect(ci.all_passed).toBe(true);
  expect(ci.workflows.map((row: any) => row.name).sort()).toEqual([
    'Release packaging',
    'Repository privacy',
    'Site Build Check',
    'Socket Security Check',
    'Tests',
  ]);
  for (const workflow of ci.workflows) {
    expect(workflow.status).toBe('completed');
    expect(workflow.conclusion).toBe('success');
    expect(workflow.url).toBe(
      `https://github.com/zolotukhin/zinc/actions/runs/${workflow.databaseId}`,
    );
  }
  expect(latestCandidate.recorded_workflow_conclusion).toBe('failure');
  expect(latestCandidate.missing_profiles).toEqual(['vulkan-intel-xe2']);
  expect(latestCandidate.certified || latestCandidate.published).toBe(false);
  // The live browser check belongs to the earlier archive, not this cut.
  expect(latestCandidate.browser_smoke).toBeUndefined();
  expect(latestCandidate.prior_records.map((row: any) => row.url)).toContain(
    'https://zolotukhin.ai' + candidatePath,
  );
});

test('refreshed Gemma acceptance retains repeated edits and actual test-command observations', async () => {
  const audit = latestCandidate.actual_clients.delivery_trace_audit;
  expect(audit.tool_errors).toBe(12);
  expect(audit.rejected_identical_edits).toBe(12);
  expect(audit.observed_delivery_limits).toHaveLength(2);
  for (const [backend, count] of [
    ['rocm', 4],
    ['vulkan', 8],
  ] as const) {
    const row = audit.observed_delivery_limits.find((item: any) => item.backend === backend);
    expect(row.model).toBe('gemma4-26b-a4b-q4k-m');
    expect(row.fixture).toBe('rate-limiter-single');
    expect(row.rejected_identical_edits).toBe(count);
    expect(row.acceptance_passed).toBe(true);
  }
  expect(
    audit.fixtures.filter((row: any) => row.recognized_test_commands_after_final_file_mutation > 0),
  ).toHaveLength(36);
  const plan = await readFile(
    resolve(import.meta.dir, '../../docs/releases/0.1.2-plan.md'),
    'utf8',
  );
  expect(plan.replace(/\s+/g, ' ')).toContain(
    'task rejected four identical edits on HIP and eight on Vulkan',
  );
});
