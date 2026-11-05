import { expect, test } from 'bun:test';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createFixture, evaluate, verifyFixture } from './openclaw_eval.mjs';
import { INTERRUPTED_CODE } from './setup_openclaw.mjs';

test('acceptance requires real code, exact readback and unchanged independent tests', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'zinc-openclaw-verification-'));
  const project = path.join(root, 'project');
  try {
    const fixture = await createFixture(project);
    expect((await verifyFixture(project, fixture)).passed).toBe(false);
    await writeFile(
      path.join(project, 'feature.mjs'),
      "export function normalizeTag(value) {return value.trim().toLowerCase().replace(/\\s+/g,'-');}\n",
    );
    expect((await verifyFixture(project, fixture)).passed).toBe(false);
    const { receipt } = JSON.parse(await readFile(path.join(project, 'contract.json'), 'utf8'));
    await writeFile(path.join(project, 'receipt.txt'), receipt + '\n');
    expect((await verifyFixture(project, fixture)).passed).toBe(true);
    await writeFile(path.join(project, 'feature.test.mjs'), '// fake success\n');
    const forged = await verifyFixture(project, fixture);
    expect(forged.passed).toBe(false);
    expect(forged.checks['feature.test.mjs']).toBe(false);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test('cancelling a task cannot start a subsequent recovery client', async () => {
  const root = await mkdtemp(path.join(tmpdir(), 'zinc-openclaw-cancel-'));
  let calls = 0;
  try {
    await expect(
      evaluate(
        { output: path.join(root, 'trial'), baseUrl: 'http://127.0.0.1:9090/v1' },
        {
          discover: async () => ({
            baseUrl: 'http://127.0.0.1:9090/v1',
            model: 'fixture',
            context: 16384,
          }),
          validate: async () => {},
          run: async () => {
            calls++;
            throw Object.assign(new Error('custom cancellation wording'), {
              code: INTERRUPTED_CODE,
            });
          },
        },
      ),
    ).rejects.toThrow('custom cancellation wording');
    expect(calls).toBe(1);
    const report = JSON.parse(await readFile(path.join(root, 'trial/summary.json'), 'utf8'));
    expect(report.interrupted).toBe(true);
    expect(report.passed).toBe(false);
    expect(report.healthyAfter).toBeNull();
    expect(report.results).toHaveLength(1);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});
