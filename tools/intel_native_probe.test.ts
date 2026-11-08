import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';

test('native Intel bring-up rejects CPU fallback and incomplete GPU evidence', () => {
  const result = spawnSync(
    'python3',
    ['-m', 'unittest', 'discover', '-s', 'tools', '-p', 'intel_native_probe_test.py'],
    { cwd: new URL('..', import.meta.url), encoding: 'utf8', timeout: 30_000 },
  );
  expect(result.status, result.stdout + result.stderr).toBe(0);
});
