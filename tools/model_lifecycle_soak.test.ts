import { expect, test } from 'bun:test';

test('lifecycle soak rejects model, stream, telemetry and ownership regressions without a GPU', () => {
  const result = Bun.spawnSync([
    'python3',
    '-m',
    'unittest',
    'discover',
    '-s',
    'tools',
    '-p',
    'model_lifecycle_soak_test.py',
    '-v',
  ]);
  expect(result.exitCode, result.stdout.toString() + result.stderr.toString()).toBe(0);
});
