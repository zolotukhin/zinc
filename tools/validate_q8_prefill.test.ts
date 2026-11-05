import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';

test('Q8 prefill regression rejects silent fallback, wrong shapes and invalid logits', () => {
  const result = spawnSync(
    'python3',
    ['-m', 'unittest', 'discover', '-s', 'tools', '-p', 'validate_q8_prefill_test.py'],
    { cwd: new URL('..', import.meta.url), encoding: 'utf8', timeout: 30_000 },
  );
  expect(result.status, result.stdout + result.stderr).toBe(0);
});
