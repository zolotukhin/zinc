# Effort 25 — Qwen 3.5 9B prefill (Metal, M4 Max) — STATUS: CLOSED-MERGED

Baseline: 36.5 prefill tok/s, flat across all prompt sizes (a decode-shaped
path), vs llama.cpp 79–445. The 21-cycle loop itself produced no runtime
speedup (all variants 37.7–38.1 vs 38.4 pre-agent baseline), but its analysis
named the real fix — a true layer-major materialized prefill — which landed the
next day in `dbf2eb5a` ("metal: materialize qwen35 prefill prefix"):
local core prefill ~331 tok/s vs llama.cpp ~333, i.e. near parity (recorded in
Effort 28's header). Note: the published metal suite JSON generated
2026-06-13 predates this fix and still shows 36.5. Model: `qwen35-9b-q4k-m`,
dense SSM+attention hybrid, no MoE.

## Landed
- Layer-major materialized prefill for Qwen 3.5 9B (`dbf2eb5a`,
  `src/compute/forward_metal.zig`): per-bucket selectors `isQwen35DensePrefill*`
  (packed full-attn Q/gate Q4_K, KV Q6_K, SSM qkv/gate/tail, dense gate/up,
  down), gates `qwen35Dense9bPrefill*` including fused gate/up+SwiGLU GEMM
  (`qwen35Dense9bPrefillGateUpSwiGLUGemmEnabled`) and down-accumulate GEMM;
  prefix depth via `ZINC_QWEN35_9B_PREFILL_PREFIX_LAYERS` → 36.5 → ~331
  prefill tok/s.
- Materialization counters + first-fallback-reason profile line
  ("qwen35-9b layer-major prefill: ... reason ...",
  `qwen35_dense_prefill_materialized_*` fields) — exactly the observability
  this effort called for.
- Prefill-side Q4/Q4 SSM qkv+gate pair dispatch
  (`isQwen35Dense9bSsmQ4Q4QkvGatePairTarget` inside
  `canUseQwen35SsmQ4Q4QkvGatePair`).
- Harness hardening (`c3ddc6de`, `loops/implement_metal.ts`): fresh runs seed
  `currentBest`/`bestTokPerSec` from the pre-agent baseline; neutral
  `optimization` keeps are rejected on structural efforts.

## Dead ends (do not retry)
- Incremental kernel variants on the token-shaped path — queued one-command
  prefill, partial prefix-depth materialization, packed Q/gate batched
  deinterleave, KV/RoPE fusion, 32-token tiling, small-batch Q4_K/Q6_K tail
  kernels — all neutral (37.7–38.1); only the full layer-major materialized
  path moved the number. (The SwiGLU fusion later shipped, but as a GEMM
  inside that materialized path, not as a matvec variant.)
- Reusing 27B fixed shapes (K=5120/17408, 64 layers) on 9B — different model:
  32 layers, hidden 4096, FFN ~12288, attention gate interleaved with Q by
  head (no contiguous `[Q_all | gate_all]` split).

## Still open
- 9B decode on M4 was only this effort's guard metric and remains ~29 vs
  llama.cpp ~57 tok/s; no dedicated Metal effort covered it as of 2026-06-14.
