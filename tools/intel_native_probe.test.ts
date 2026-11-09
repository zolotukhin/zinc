import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';

test('native Intel bring-up rejects CPU fallback and incomplete GPU evidence', () => {
  // A cold macOS C++ compiler can exceed Bun's five-second default. Keep
  // Bun's deadline above this child process and its 60-second compiler limit.
  const result = spawnSync(
    'python3',
    ['-m', 'unittest', 'discover', '-s', 'tools', '-p', 'intel_native_probe_test.py'],
    { cwd: new URL('..', import.meta.url), encoding: 'utf8', timeout: 90_000 },
  );
  const diagnostic = [result.stdout, result.stderr, result.error?.message, result.signal]
    .filter(Boolean)
    .join('\n');
  expect(result.status, diagnostic).toBe(0);
}, 100_000);
