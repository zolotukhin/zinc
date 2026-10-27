import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';
import { join } from 'node:path';

test('release packaging and installer contracts use CPU-only native fixtures', () => {
  const result = spawnSync(
    'python3',
    ['-m', 'unittest', 'discover', '-s', 'tools', '-p', 'release_test.py', '-v'],
    {
      cwd: join(import.meta.dir, '..'),
      encoding: 'utf8',
      timeout: 120_000,
    },
  );
  expect(result.status, result.stdout + result.stderr).toBe(0);
}, 130_000);
