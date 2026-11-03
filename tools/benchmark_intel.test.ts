import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';
import { intelModelIds, intelWorkloads } from './intel_workloads.mjs';

test('Intel matrix covers all seven release models using the four canonical workloads', () => {
  expect(intelModelIds).toHaveLength(7);
  for (const id of intelModelIds) {
    const scenarios = intelWorkloads(id);
    expect(scenarios.map((row) => row.id)).toEqual([
      'core',
      'context-medium',
      'context-long',
      'decode-extended',
    ]);
    for (const row of scenarios) {
      expect(row.body.cache_prompt).toBe(false);
      expect(row.body.top_p).toBe(1);
      expect(row.body.seed).toBe(0);
      expect(row.body.messages.length).toBeGreaterThan(0);
    }
  }
  expect(() => intelWorkloads('unknown')).toThrow();
});

test('Intel timing/memory and GPU-only-baseline failure contracts use CPU fixtures', () => {
  const result = spawnSync(
    'python3',
    ['-m', 'unittest', 'discover', '-s', 'tools', '-p', 'benchmark_intel_test.py'],
    { cwd: new URL('..', import.meta.url), encoding: 'utf8', timeout: 30_000 },
  );
  expect(result.status, result.stdout + result.stderr).toBe(0);
});
