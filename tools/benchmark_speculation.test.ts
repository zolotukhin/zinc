import { expect, test } from 'bun:test';
import { requireBenchmarkReference, speculationPair } from './benchmark_speculation.mjs';

test('mismatched or missing MTP settings cannot establish a comparison', () => {
  const on = { speculative_decoding: { enabled: true } };
  const off = { speculative_decoding: { enabled: false } };
  expect(speculationPair(on, off).matched).toBe(false);
  expect(speculationPair(off, on).matched).toBe(false);
  expect(speculationPair(on, {}).matched).toBe(false);
  expect(speculationPair({}, {}).matched).toBe(false);
  expect(speculationPair(on, on).matched).toBe(true);
  expect(speculationPair(off, off).matched).toBe(true);
});

test('a stale or unidentified reference is rejected before benchmarking', () => {
  const expected = '785322d17f2d51e5185cf996a1f4e46d87640841';
  const current = { commit: expected.slice(0, 9), sha256: 'a'.repeat(64) };
  expect(() => requireBenchmarkReference(current, expected)).not.toThrow();
  expect(() => requireBenchmarkReference({ ...current, commit: '5f436dddb' }, expected)).toThrow();
  expect(() => requireBenchmarkReference({ ...current, commit: null }, expected)).toThrow();
  expect(() => requireBenchmarkReference({ ...current, sha256: null }, expected)).toThrow();
});
