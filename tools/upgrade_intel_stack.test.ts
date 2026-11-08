import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';

test('Intel administrator upgrade defaults to a plan and preserves exact GRUB fallback paths', () => {
  const result = spawnSync(
    'python3',
    ['-m', 'unittest', 'discover', '-s', 'tools', '-p', 'upgrade_intel_stack_test.py'],
    { cwd: new URL('..', import.meta.url), encoding: 'utf8', timeout: 30_000 },
  );
  expect(result.status, result.stdout + result.stderr).toBe(0);
});
