import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';

test('native Intel model validation preserves frozen-oracle agreement and complete continuations', () => {
  const result = spawnSync(
    'python3',
    ['-m', 'unittest', 'discover', '-s', 'tools', '-p', 'intel_native_reference_test.py'],
    { cwd: new URL('..', import.meta.url), encoding: 'utf8', timeout: 30_000 },
  );
  expect(result.status, result.stdout + result.stderr).toBe(0);
});
