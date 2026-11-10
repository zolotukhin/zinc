import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read('../site/public/zinc/validation/intel-native-gdn-operators-2026-11-10.json');
const previous = read('../site/public/zinc/validation/intel-native-exact-cadence-2026-11-10.json');
const failures = read(
  '../site/public/zinc/validation/intel-native-35b-copy-fusion-controls-2026-11-10.json',
);

test('native GDN operators retain their actual FP32 binary, upstream source and official runtime identities', () => {
  expect(report.source_report_sha256).toBe(
    '6ece334710ef6f6b3e62acfc43c661384af80de5c7e0b9af068ceb30f1fd56e1',
  );
  expect(report.test_binary_sha256).toBe(
    'a23ede35689b204dcd34f574e214d8b29cd9672ed487642460ebef35dd90debf',
  );
  expect(report.test_source_sha256).toBe(
    'e0b92b44258e22f217d0cb0d6a488c2479cf2d40bbf9402c71c483c6bdcc66a5',
  );
  expect(report.reference_commit).toBe(previous.reference.commit);
  expect(report.reference_commit).toBe('3b4d6ab3f79651935fba1dd93732f29022a7f5f5');
  for (const [key, value] of Object.entries(report.reference))
    expect(previous.reference[key]).toEqual(value);
  expect(report.native_build).toEqual(previous.native_build);
  expect(report.native_build.settings.GGML_SYCL_F16).toBe('OFF');
  expect(report.native_build.settings.GGML_SYCL_HOST_MEM_FALLBACK).toBe('OFF');
  expect(report.native_stack).toEqual(previous.native_stack);
  expect(report.native_stack.packages).toHaveLength(41);
  expect(report.runtime.runtime).toBe('level_zero');
  const libraries = Object.keys(report.runtime.library_sha256);
  expect(libraries.some((name) => name.startsWith('libsycl.so'))).toBe(true);
  expect(libraries.some((name) => name.startsWith('libur_adapter_level_zero'))).toBe(true);
  expect(report.hardware).toEqual(previous.hardware);
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware_unchanged).toBe(true);
  expect(report.cleanup).toBe('no GPU process remains');
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('all 48 recurrent and cache-fusion numerical cases pass without qualifying the failing model campaign', () => {
  expect(report.settings).toEqual({
    backend: 'SYCL0',
    mode: 'test',
    operation_filter: 'GATED_DELTA_NET.*',
    parallel_workers: 1,
    host_test_concurrency: 4,
  });
  expect(report.completed).toBe(true);
  expect(report.passed).toBe(true);
  expect(report.numerical_passed).toBe(true);
  expect(report.exit_code).toBe(0);
  expect(report.expected_cases).toBe(48);
  expect(report.cases).toHaveLength(48);
  expect(report.summary).toEqual([{ passed: 48, total: 48 }]);
  expect(report.cases.filter((row) => row.operation === 'GATED_DELTA_NET')).toHaveLength(43);
  expect(
    report.cases.filter((row) => row.operation === 'GATED_DELTA_NET_CACHE_FUSION'),
  ).toHaveLength(5);
  expect(new Set(report.cases.map((row) => `${row.operation}(${row.params})`)).size).toBe(48);
  for (const row of report.cases) expect(row.result).toBe('OK');
  for (const count of [1, 33, 64, 65, 100, 127, 200, 256, 257, 512, 513, 2047, 2048]) {
    expect(report.cases.some((row) => row.params.includes(`n_seq_tokens=${count},`))).toBe(true);
  }
  expect(report.cases.some((row) => row.params.includes('permuted=1'))).toBe(true);
  expect(report.cases.some((row) => row.params.includes('kda=1'))).toBe(true);
  expect(report.complete_model_matrix).toBe(false);
  expect(report.performance_qualified).toBe(false);
  expect(report.release_qualification).toBe(false);
  expect(report.native_zinc_backend).toBe(false);
  expect(failures.passed).toBe(false);
  expect(failures.treatments[0].models[0].output_drift).toHaveLength(19);
  expect(failures.treatments[1].protocol_completed).toBe(false);
});
