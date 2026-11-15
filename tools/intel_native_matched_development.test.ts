import { expect, test } from 'bun:test';
import { readFileSync } from 'node:fs';
import { assertPublicArtifact } from './public_artifact_privacy.mjs';

const read = (name: string) => JSON.parse(readFileSync(new URL(name, import.meta.url), 'utf8'));
const report = read(
  '../site/public/zinc/validation/intel-native-matched-development-2026-11-15.json',
);
const lock = readFileSync(
  new URL('../config/intel/oneapi-2026.2-linux-64.explicit.txt', import.meta.url),
  'utf8',
);
const policy = read('../config/release/targets.json');
const identity = (s) =>
  Object.fromEntries(
    ['content_sha256', 'content_bytes', 'promptTokens', 'generatedTokens', 'finish_reason'].map(
      (key) => [key, s[key]],
    ),
  );
const distributions = (rows) =>
  new Set(rows.map((r) => JSON.stringify(r.first_token_top_logprobs))).size;
const sequences = (rows) => new Set(rows.map((r) => JSON.stringify(r.ids))).size;

test('the complete SDK keeps matching official oneDNN development and runtime packages', () => {
  expect(report.native_stack.packages).toHaveLength(42);
  for (const row of report.native_stack.packages) {
    expect(lock).toContain(`#${row.sha256}`);
    expect(row.sha256).toMatch(/^[a-f0-9]{64}$/);
  }
  const runtime = report.native_stack.packages.find((p) => p.name === 'onednn');
  const headers = report.native_stack.packages.find((p) => p.name === 'onednn-devel');
  expect(headers.version).toBe(runtime.version);
  expect(headers.build).toBe(runtime.build);
  expect(headers.sha256).toBe('8970090b393e8c0dc1a14cb4dd47c15d76aba72af28cfd319f43502d5dc8413c');
  expect(report.onednn_development.version).toBe('3.13.2');
  expect(report.onednn_development.adapter_origin).toContain('not an Intel-supplied CMake config');
  expect(report.oneDNN_version_probe).toEqual({
    headers: [3, 13, 2],
    runtime: [3, 13, 2],
    gpu_runtime: 512,
    expected_gpu_runtime: 512,
  });
  expect(report.verified_sdk_guard.old_sdk_rejected).toBe(true);
  expect(report.verified_sdk_guard.native_build.onednn_development.version).toBe('3.13.2');
  expect(report.verified_sdk_guard.adapter.config_sha256).toBe(
    report.onednn_development.sha256['lib/cmake/dnnl/dnnl-config.cmake'],
  );
  expect(report.readiness.passed).toBe(true);
  expect(report.readiness.probe.cases).toBe(78);
  expect(report.readiness.probe.max_absolute_error).toBe(0);
  expect(report.readiness.owned_process_reaped).toBe(true);
  expect(report.native_opencl.device_type).toBe('gpu');
  expect(report.native_opencl.vendor_id).toBe(0x8086);
});

test('the reference was selected before its clean FP32 build and exact-cadence campaign', () => {
  expect(report.reference.commit).toBe('e574a17b062be51b98e239ccd3d3743e3d902333');
  expect(report.reference_selection.commit).toBe(report.reference.commit);
  const dates = [
    report.reference_selection.checked_at_utc,
    report.build_started_at_utc,
    report.build_completed_at_utc,
    report.started_at_utc,
    report.completed_at_utc,
  ].map(Date.parse);
  expect(dates.every(Number.isFinite)).toBe(true);
  expect(dates).toEqual([...dates].sort((a, b) => a - b));
  expect(dates[3] - dates[0]).toBeLessThan(2 * 60 * 60 * 1000);
  expect(report.native_build.settings.GGML_SYCL_F16).toBe('OFF');
  expect(report.native_build.settings.GGML_SYCL_HOST_MEM_FALLBACK).toBe('OFF');
  expect(report.hardware_after).toEqual(report.hardware);
  expect(report.hardware_unchanged).toBe(true);
  expect(report.host_packages_changed).toBe(false);
  expect(report.cleanup).toBe('no GPU process remains');
  expect(() => assertPublicArtifact(report)).not.toThrow();
});

test('all failed full responses and streams remain in the corrected-SDK replay', () => {
  const m = report.model;
  const pin = policy.models.find((p) => p.id === m.id);
  expect(m.context).toBe(pin.context);
  expect(m.model_sha256).toBe(pin.sha256);
  expect(m.native_reference_validation.passed).toBe(true);
  expect(m.native_reference_validation.agreement).toMatchObject({
    total_tokens: 48,
    exact_tokens: 47,
    near_tie_tokens: 1,
  });
  expect(report.settings.extra_json_control_before_sse).toBe(false);
  expect(m.samples).toHaveLength(32);
  expect(m.streams).toHaveLength(8);
  expect(m.processes).toHaveLength(2);
  const first = new Map();
  const drift = [];
  for (const s of m.samples) {
    const actual = identity(s);
    if (!first.has(s.scenario)) first.set(s.scenario, actual);
    const expected = first.get(s.scenario);
    const changed_fields = Object.keys(expected).filter((k) => expected[k] !== actual[k]);
    if (changed_fields.length)
      drift.push({
        lifetime: s.lifetime,
        scenario: s.scenario,
        repeat: s.repeat,
        expected,
        actual,
        changed_fields,
      });
    expect(s.usage.prompt_tokens).toBe(s.promptTokens);
    expect(s.usage.completion_tokens).toBe(s.generatedTokens);
    expect(s.usage.prompt_tokens_details.cached_tokens).toBe(0);
    expect(s.server_timings.cache_n).toBe(0);
    expect(s.server_timings.prompt_n).toBe(s.promptTokens);
    expect(s.server_timings.predicted_n).toBe(s.generatedTokens);
  }
  expect(m.output_drift).toEqual(drift);
  expect(drift).toHaveLength(19);
  const streams = new Map();
  const streamDrift = [];
  for (const s of m.streams) {
    if (!streams.has(s.scenario)) streams.set(s.scenario, s.content_sha256);
    if (streams.get(s.scenario) !== s.content_sha256)
      streamDrift.push({
        lifetime: s.lifetime,
        scenario: s.scenario,
        expected: streams.get(s.scenario),
        actual: s.content_sha256,
      });
  }
  expect(m.stream_drift).toEqual(streamDrift);
  expect(streamDrift).toHaveLength(1);
  for (const scenario of ['core', 'context-medium', 'context-long', 'decode-extended'])
    for (const lifetime of [0, 1]) {
      const rows = m.samples.filter((s) => s.scenario === scenario && s.lifetime === lifetime);
      expect(rows.map((s) => s.repeat)).toEqual([0, 1, 2, 3]);
      expect(rows.map((s) => s.warmup)).toEqual([true, false, false, false]);
    }
  const wanted = report.onednn_development.sha256['lib/libdnnl.so'];
  for (const process of m.processes) {
    expect(process.gpu_offload).toEqual({ backend: 'sycl', offloaded: 41, total: 41 });
    expect(process.native_runtime.library_sha256['libdnnl.so.3.13']).toBe(wanted);
    expect(process.native_runtime.runtime).toBe('level_zero');
    expect(
      Object.keys(process.native_runtime.level_zero_library_sha256).some((k) =>
        k.startsWith('libze_intel_gpu'),
      ),
    ).toBe(true);
  }
});

test('unchanged greedy choices cannot hide varying first-token distributions or later output drift', () => {
  for (const d of report.diagnostics) {
    expect(d.completed).toBe(true);
    expect(d.diagnostic_only).toBe(true);
    expect(d.performance_qualified).toBe(false);
    expect(d.first_token).toHaveLength(16);
    expect(d.full).toHaveLength(8);
    expect(d.fixed_prefix).toHaveLength(16);
    expect(d.fixed_prefix_length).toBe(0);
    expect(d.supplied_generated_prefix).toEqual([]);
    expect(sequences(d.first_token)).toBe(1);
    expect(distributions(d.first_token)).toBe(d.settings.GGML_SYCL_ENABLE_DNN === '0' ? 1 : 16);
    expect(distributions(d.fixed_prefix)).toBe(d.settings.GGML_SYCL_ENABLE_DNN === '0' ? 1 : 16);
    const first = d.full[0].ids;
    expect(d.first_divergent_positions).toEqual(
      d.full
        .slice(1)
        .filter((r) => JSON.stringify(r.ids) !== JSON.stringify(first))
        .map((r) => {
          const position = first.findIndex((id, i) => id !== r.ids[i]);
          return position < 0 ? Math.min(first.length, r.ids.length) : position;
        }),
    );
    expect(sequences(d.full)).toBe(
      d.treatment === 'onednn-gemm-and-reordering-disabled'
        ? 1
        : d.treatment === 'onednn-gemm-disabled'
          ? 2
          : d.treatment === 'reordering-disabled'
            ? 6
            : 7,
    );
    for (const group of [d.first_token, d.full, d.fixed_prefix])
      for (const s of group) {
        expect(s.cache_n).toBe(0);
        expect(s.prompt_n).toBe(58);
        expect(s.first_token_top_logprobs).toHaveLength(8);
      }
  }
  const off = report.diagnostics.find((d) => d.treatment === 'reordering-disabled');
  expect(off.settings).toEqual({ GGML_SYCL_ENABLE_OPT: '0' });
  expect(off.effective_enable_opt).toBe(0);
  expect(off.control_observed).toBe(true);
  expect(sequences(off.full)).toBe(6);
});

test('the first varying observed tensor remains a diagnostic, separate from qualified model support', () => {
  const trace = report.intermediate_trace;
  expect(trace.completed).toBe(true);
  expect(trace.performance_qualified).toBe(false);
  for (const t of trace.treatments) {
    expect(t.logits).toHaveLength(8);
    expect(t.logit_variants).toBe(new Set(t.logits.map((x) => x.fnv1a64)).size);
    expect(t.logit_variants).toBe(8);
  }
  const observed = trace.treatments.find((t) => t.mode === 'all-nodes');
  expect(observed.tensor_counts.every((n) => n === observed.tensor_counts[0])).toBe(true);
  for (const pair of observed.first_tensor_divergence) {
    expect(pair.first).toHaveLength(2);
    for (const t of pair.first)
      expect(t).toMatchObject({
        node: 65,
        name: 'ffn_moe_logits-0',
        op: 'MUL_MAT',
        type: 'f32',
        shape: [256, 58, 1, 1],
        contiguous: true,
      });
    expect(pair.first[0].fnv1a64).not.toBe(pair.first[1].fnv1a64);
  }
  expect(report.completed).toBe(true);
  expect(report.passed).toBe(false);
  for (const key of [
    'performance_qualified',
    'release_qualification',
    'native_zinc_backend',
    'complete_model_matrix',
    'all_models_all_dimensions_better',
  ])
    expect(report[key]).toBe(false);
});

test('unchanging router inputs isolate varying oneDNN matrix outputs, without declaring a root cause', () => {
  const on = report.router_input_trace.treatments[0];
  const off = report.onednn_disabled_intermediate_trace.treatments[0];
  for (const treatment of [on, off])
    for (const slot of [0, 1]) {
      const rows = treatment.router_inputs.filter((r) => r.slot === slot);
      expect(rows).toHaveLength(8);
      expect(new Set(rows.map((r) => r.fnv1a64)).size).toBe(1);
      for (const row of rows) expect(row).toMatchObject({ type: 'f32', contiguous: true });
    }
  expect(on.router_inputs.filter((r) => r.slot === 0).map((r) => r.fnv1a64)).toEqual(
    off.router_inputs.filter((r) => r.slot === 0).map((r) => r.fnv1a64),
  );
  expect(on.logit_variants).toBe(8);
  expect(off.logit_variants).toBe(1);
  expect(off.first_tensor_divergence.every((r) => r.first === null)).toBe(true);
  const normal = report.diagnostics.find((d) => d.treatment === 'onednn-gemm-disabled');
  expect(normal.effective_enable_dnn).toBe(0);
  expect(normal.control_observed).toBe(true);
  expect(sequences(normal.full)).toBe(2);
  const both = report.diagnostics.find(
    (d) => d.treatment === 'onednn-gemm-and-reordering-disabled',
  );
  expect(both.effective_enable_dnn).toBe(0);
  expect(both.effective_enable_opt).toBe(0);
  expect(both.control_observed).toBe(true);
  expect(sequences(both.full)).toBe(1);
  expect(both.performance_qualified).toBe(false);
});

test('the stable small control cannot bypass a failed correctness preflight', () => {
  const control = report.extended_control;
  expect(control.completed).toBe(false);
  expect(control.passed).toBe(false);
  expect(control.performance_qualified).toBe(false);
  expect(control.stage).toBe('native-preflight');
  expect(control.error_type).toBe('ValueError');
  expect(control.full_requests).toBe(0);
  expect(control.streams).toBe(0);
  const check = control.native_reference_validation;
  expect(check.passed).toBe(false);
  expect(check.validation_stage).toBe('aggregate_agreement');
  expect(check.cases).toHaveLength(3);
  expect(check.cases.reduce((n, c) => n + c.comparison.tokens, 0)).toBe(48);
  expect(check.cases.reduce((n, c) => n + c.comparison.near_ties.length, 0)).toBe(2);
});

test('the standalone oneDNN probe retains small numeric variation and explicit determinism separately', () => {
  const p = report.standalone_onednn_router_probe;
  expect(p.completed).toBe(true);
  expect(p.diagnostic_only).toBe(true);
  expect(p.performance_qualified).toBe(false);
  expect(p.native_zinc_backend).toBe(false);
  expect(p.release_qualification).toBe(false);
  expect(p.shape).toEqual({ m: 256, n: 58, k: 2048 });
  expect(p.samples).toHaveLength(64);
  expect(p.source_path).toBe('tools/intel_onednn_matmul_probe.cpp');
  for (const deterministic of [false, true]) {
    const rows = p.samples.filter((r) => r.deterministic === deterministic);
    expect(rows.map((r) => r.run)).toEqual(Array.from({ length: 32 }, (_, i) => i));
    expect(new Set(rows.map((r) => r.fnv1a64)).size).toBe(deterministic ? 1 : 32);
    for (const row of rows) {
      expect(row.guards_intact).toBe(true);
      expect(row.nonfinite).toBe(0);
      expect(row.max_absolute_error).toBeGreaterThanOrEqual(0);
      expect(row.max_absolute_error).toBeLessThan(0.00003);
      expect(row.max_change_from_first).toBeGreaterThanOrEqual(0);
      if (deterministic) {
        expect(row.changed_elements).toBe(0);
        expect(row.max_change_from_first).toBe(0);
      } else if (row.run > 0) expect(row.changed_elements).toBeGreaterThan(0);
    }
  }
  expect(() => assertPublicArtifact(p)).not.toThrow();
});
