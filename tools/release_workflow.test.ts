import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

function workflow(name: string): any {
  return Bun.YAML.parse(
    readFileSync(join(import.meta.dir, '..', '.github', 'workflows', name), 'utf8'),
  );
}

test('PR packaging has no trusted GPU, signing or publication permissions', () => {
  const preview = workflow('release-preview.yml');
  expect(preview.on.pull_request).toBeDefined();
  expect(preview.on.pull_request_target).toBeUndefined();
  expect(preview.permissions).toEqual({ contents: 'read' });
  expect(
    preview.jobs.package.steps.some((step: any) => step.run?.includes('release_build.py')),
  ).toBe(true);
  expect(JSON.stringify(preview)).not.toContain('secrets.');
  expect(JSON.stringify(preview)).not.toContain('release_gpu.py');
});

test('candidate hardware consumes frozen archives after signing and cannot run from a PR event', () => {
  const candidate = workflow('release-candidate.yml');
  expect(candidate.on.pull_request).toBeUndefined();
  expect(candidate.on.pull_request_target).toBeUndefined();
  expect(candidate.jobs.prepare.steps[0].run).toContain('refs/heads/main');
  expect(candidate.jobs.sign.environment).toBe('release-signing');
  expect(candidate.jobs.sign.if).toBe("needs.prepare.outputs.signing_required == 'true'");
  expect(candidate.jobs.prepare.outputs.signing_required).toBe(
    '${{ steps.resolve.outputs.signing_required }}',
  );
  expect(candidate.jobs.manifest.needs).toContain('sign');
  expect(candidate.jobs.hardware.needs).toContain('manifest');
  expect(candidate.jobs.hardware.environment).toBe('release-workers');
  expect(
    candidate.jobs.hardware.steps.some((step: any) => step.with?.name === 'candidate-inputs'),
  ).toBe(true);
  expect(
    candidate.jobs.hardware.steps.some((step: any) => step.run?.includes('release_gpu_group.py')),
  ).toBe(true);
  expect(JSON.stringify(candidate.jobs.hardware)).not.toContain('zig.sh build');
  expect(candidate.jobs.hardware.concurrency['cancel-in-progress']).toBe(false);
  expect(candidate.jobs.certify.needs).toContain('hardware');
  expect(candidate.jobs.certify.permissions['attestations']).toBe('write');
});

test('manifest permits an unnecessary skipped signer and blocks failed or required signing', () => {
  const candidate = workflow('release-candidate.yml');
  const expression = candidate.jobs.manifest.if.replace(/^\s*\$\{\{|\}\}\s*$/g, '');
  const scenarios = [
    {
      required: false,
      sign: 'skipped',
      build: 'success',
      prepare: 'success',
      cancelled: false,
      runs: true,
    },
    {
      required: true,
      sign: 'success',
      build: 'success',
      prepare: 'success',
      cancelled: false,
      runs: true,
    },
    {
      required: true,
      sign: 'skipped',
      build: 'success',
      prepare: 'success',
      cancelled: false,
      runs: false,
    },
    {
      required: false,
      sign: 'failure',
      build: 'success',
      prepare: 'success',
      cancelled: false,
      runs: false,
    },
    {
      required: true,
      sign: 'failure',
      build: 'success',
      prepare: 'success',
      cancelled: false,
      runs: false,
    },
    {
      required: false,
      sign: 'skipped',
      build: 'failure',
      prepare: 'success',
      cancelled: false,
      runs: false,
    },
    {
      required: false,
      sign: 'skipped',
      build: 'success',
      prepare: 'failure',
      cancelled: false,
      runs: false,
    },
    {
      required: false,
      sign: 'skipped',
      build: 'success',
      prepare: 'success',
      cancelled: true,
      runs: false,
    },
  ];
  for (const row of scenarios) {
    const resolved = expression
      .replaceAll('cancelled()', JSON.stringify(row.cancelled))
      .replaceAll('needs.prepare.result', JSON.stringify(row.prepare))
      .replaceAll('needs.build.result', JSON.stringify(row.build))
      .replaceAll('needs.sign.result', JSON.stringify(row.sign))
      .replaceAll('needs.prepare.outputs.signing_required', JSON.stringify(String(row.required)));
    expect(new Function(`return (${resolved});`)()).toBe(row.runs);
  }
  const signedDownload = candidate.jobs.manifest.steps.find(
    (step: any) => step.with?.name === 'candidate-metal',
  );
  expect(signedDownload.if).toBe("needs.prepare.outputs.signing_required == 'true'");
});

test('publication requires review of a candidate before the protected job and verifies public installs afterwards', () => {
  const publisher = workflow('release.yml');
  expect(Object.keys(publisher.on)).toEqual(['workflow_dispatch']);
  expect(publisher.jobs.publish.needs).toBe('preflight');
  expect(publisher.jobs.publish.environment.name).toBe('release');
  expect(publisher.jobs.publish.permissions.contents).toBe('write');
  expect(
    publisher.jobs.preflight.steps.some((step: any) => step.run?.includes('release_publish.py')),
  ).toBe(true);
  expect(publisher.jobs['post-publish'].needs).toContain('publish');
  expect(JSON.stringify(publisher)).not.toContain('release_build.py');
  expect(JSON.stringify(publisher)).toContain('--public-download');
  expect(JSON.stringify(publisher)).toContain('install-smoke');
});

test('release actions use immutable commit references and no saved checkout credentials', () => {
  for (const name of [
    'release-preview.yml',
    'release-candidate.yml',
    'release-worker-check.yml',
    'release.yml',
  ]) {
    const data = workflow(name);
    for (const job of Object.values(data.jobs) as any[]) {
      for (const step of job.steps ?? []) {
        if (step.uses) expect(step.uses).toMatch(/@[0-9a-f]{40}$/);
        if (step.uses?.startsWith('actions/checkout@'))
          expect(step.with['persist-credentials']).toBe(false);
      }
    }
  }
});

test('worker readiness is main-only, protected, and separate from candidate certification', () => {
  const check = workflow('release-worker-check.yml');
  expect(Object.keys(check.on)).toEqual(['workflow_dispatch']);
  expect(check.jobs.prepare.if).toContain("github.ref == 'refs/heads/main'");
  expect(check.jobs.check.environment).toBe('release-workers');
  expect(check.jobs.check.concurrency.group).toContain('runner_variable');
  expect(JSON.stringify(check)).not.toContain('secrets.');
  expect(JSON.stringify(check)).not.toContain('release.py aggregate');
  expect(check.permissions).toEqual({ contents: 'read', actions: 'read' });
  expect(check.jobs.prepare.outputs.validation_commit).toBe(
    '${{ steps.resolve.outputs.validation_commit }}',
  );
  expect(check.jobs.prepare.steps[0].with.ref).toBe('${{ github.sha }}');
  const checkout = check.jobs.check.steps.find((step: any) =>
    step.uses?.startsWith('actions/checkout@'),
  );
  expect(checkout.with.ref).toBe('${{ needs.prepare.outputs.validation_commit }}');
  const validation = check.jobs.check.steps.find((step: any) => step.env?.PREVIEW_COMMIT);
  expect(validation.env.PREVIEW_COMMIT).toBe('${{ needs.prepare.outputs.commit }}');
  expect(validation.env.VALIDATION_COMMIT).toBe('${{ needs.prepare.outputs.validation_commit }}');
  expect(validation.run).toContain(
    '--commit "$PREVIEW_COMMIT" --validation-commit "$VALIDATION_COMMIT"',
  );
  expect(validation.run).toContain('--directory "$RUNNER_TEMP/zinc-worker-preview"');
  expect(validation.run).toContain(
    '--output "$RUNNER_TEMP/zinc-worker-preview/worker-readiness.json"',
  );
  const report = check.jobs.check.steps.find((step: any) =>
    step.uses?.startsWith('actions/upload-artifact@'),
  );
  expect(report.with.path).toBe('${{ runner.temp }}/zinc-worker-preview/worker-readiness.json');
  expect(
    check.jobs.check.steps.find((step: any) => step.uses?.startsWith('actions/download-artifact@'))
      .with.path,
  ).toBe('${{ runner.temp }}/zinc-worker-preview');
});
