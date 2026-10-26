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
  expect(candidate.jobs.manifest.needs).toContain('sign');
  expect(candidate.jobs.hardware.needs).toContain('manifest');
  expect(
    candidate.jobs.hardware.steps.some((step: any) => step.with?.name === 'candidate-inputs'),
  ).toBe(true);
  expect(
    candidate.jobs.hardware.steps.some((step: any) => step.run?.includes('release_gpu.py')),
  ).toBe(true);
  expect(JSON.stringify(candidate.jobs.hardware)).not.toContain('zig.sh build');
  expect(candidate.jobs.hardware.concurrency['cancel-in-progress']).toBe(false);
  expect(candidate.jobs.certify.needs).toContain('hardware');
  expect(candidate.jobs.certify.permissions['attestations']).toBe('write');
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
  for (const name of ['release-preview.yml', 'release-candidate.yml', 'release.yml']) {
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
