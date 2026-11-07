import { expect, test } from 'bun:test';
import { spawnSync } from 'node:child_process';
import { intelModelIds, intelWorkloads } from './intel_workloads.mjs';

test('Intel matrix covers the entire managed catalog using the four canonical workloads', () => {
  // Compile the actual data: source formatting and nonliteral IDs must not
  // hide a model. The exporter only serializes constants, without GPU access.
  const exported = spawnSync('./scripts/zig.sh', ['run', 'src/catalog_manifest.zig', '-lc'], {
    cwd: new URL('..', import.meta.url),
    encoding: 'utf8',
    timeout: 120_000,
  });
  expect(exported.status, exported.stderr).toBe(0);
  const catalogIds = JSON.parse(exported.stdout).map((entry: { id: string }) => entry.id);
  expect(catalogIds.length).toBeGreaterThan(0);
  expect([...intelModelIds].sort()).toEqual(catalogIds.sort());
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
}, 120_000);

test('Intel timing/memory and GPU-only-baseline failure contracts use CPU fixtures', () => {
  const result = spawnSync(
    'python3',
    ['-m', 'unittest', 'discover', '-s', 'tools', '-p', 'benchmark_intel_test.py'],
    { cwd: new URL('..', import.meta.url), encoding: 'utf8', timeout: 30_000 },
  );
  expect(result.status, result.stdout + result.stderr).toBe(0);
});
