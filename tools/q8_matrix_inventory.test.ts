import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read(
  '../site/public/zinc/validation/intel-q8-matrix-range-inventory-2026-11-10.json',
);
const policy = read('../config/release/targets.json');

test('the Q8 range inventory covers every immutable managed model without inference claims', () => {
  expect(report.passed).toBe(true);
  expect(report.source_report_sha256).toBe(
    '85f3ae4e5ff0b7b27975a6a175c4a63dc64492a845648610eed7e22adb4dc1da',
  );
  expect(report.models.map((model) => model.id).sort()).toEqual(
    policy.models.map((model) => model.id).sort(),
  );
  for (const model of report.models) {
    const pin = policy.models.find((entry) => entry.id === model.id);
    expect(model.sha256).toBe(pin.sha256);
    expect(model.size_bytes).toBe(pin.size_bytes);
    expect(model.all_fit_uint32).toBe(true);
    expect(model.q8_matrix_count).toBe(model.matrices.length);
  }
  expect(report.release_qualification).toBe(false);
  expect(report.performance_qualified).toBe(false);
  expect(report.error_type).toBeUndefined();
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('all 704 pinned Q8 tensors fit the host descriptor and byte-address contract', () => {
  let count = 0;
  let maximum = 0;
  for (const model of report.models) {
    let modelMaximum = 0;
    for (const matrix of model.matrices) {
      expect(matrix.name).toMatch(/^[A-Za-z0-9_.-]+$/);
      expect(Number.isSafeInteger(matrix.rows) && Number.isSafeInteger(matrix.columns)).toBe(true);
      expect(matrix.rows).toBeGreaterThan(0);
      expect(matrix.columns).toBeGreaterThan(0);
      expect(matrix.columns % 32).toBe(0);
      expect(matrix.bytes).toBe(matrix.rows * (matrix.columns / 32) * 34);
      expect(matrix.bytes).toBeLessThanOrEqual(0xffffffff);
      modelMaximum = Math.max(modelMaximum, matrix.bytes);
      count++;
    }
    expect(model.maximum_q8_matrix_bytes).toBe(modelMaximum);
    maximum = Math.max(maximum, modelMaximum);
  }
  expect(count).toBe(704);
  expect(maximum).toBe(748 * 1024 * 1024);
});
