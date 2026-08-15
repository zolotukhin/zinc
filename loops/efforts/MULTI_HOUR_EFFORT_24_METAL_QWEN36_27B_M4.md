# Effort 24 — Qwen 3.6 27B dense-hybrid decode (Metal, M4 Max) — STATUS: SUPERSEDED by Effort 28

Baseline 10.48 decode tok/s (public-suite long rows; core rows timed out) → in-loop
best 15.09, published median ~15.2 (suite 2026-06-13), core rows no longer time
out. llama.cpp re-measured at ~21.4–22.0 decode, so ~70% at handoff. All runtime
work integrated to main in `dfa04236` ("metal: integrate M4 Qwen 27B
improvements"). Effort 28 (`MULTI_HOUR_EFFORT_28_METAL_M4_QWEN36_27B_BEAT_LLAMA.md`)
continues the same model with prefill as the primary metric and decode as guard.
Model: `qwen36-27b-q4k-m` — dense hybrid (SSM + dense FFN), NO MoE; never copy
Qwen 3.6 35B-A3B route-pack assumptions here.

## Landed
- Dense Q6_K `ffn_down` (M=5120 K=17408) → llama-style Q6_K DMMV:
  `canUseDenseQ6kSimdgroupDmmvShape` + `dmmv_q6k_llama_pipe` /
  fixed-K `dmmv_q6k_llama_k17408_pipe` (`src/compute/forward_metal.zig`,
  `src/shaders/metal/dmmv_q6k_llama.metal`) → 10.48 → 12.56 tok/s (+19.6%).
- SSM Q6_K `attn_qkv` (M=10240 K=5120) through the same Q6_K DMMV route →
  12.56 → 13.15 tok/s; cross-effort prefill +29% as a side effect.
- SSM Q4_K/Q4_K `attn_qkv`+`attn_gate` pair in one single-axis dual-row
  dispatch: `canUseQwen35SsmQ4Q4QkvGatePair` →
  `src/shaders/metal/dmmv_q4k_qk_dual.metal` → best 15.09 tok/s (`ea522741`).
- Dense Q4_K `ffn_gate`+`ffn_up` through the same dual dispatch:
  `isQwen35DenseGateUpQ4kTarget` / `isQwen35DenseQ4KGateUpSwiGLUTarget`
  (`e9e83050`) — speed-neutral, kept for dispatch reduction.
- Mixed Q6/Q4 SSM qkv+gate pair (`canUseQwen35SsmQ6Q4QkvGatePair`, M_q=0 reuse
  of `dmmv_q4k_qk_q6k_v_pipe`) — in-loop variants were rejected, but this form
  landed in `dfa04236`.
- Neutral fixed-K specializations kept as cleanup: `dmmv_q4k_k17408_pipe`,
  `dmmv_q6k_llama_k17408_pipe`.
- Exact-shape microbench cases `qwen27b_decode_hot`, `qwen27b_dense_gate_up_q4k`,
  `qwen27b_ssm_q4q4`, `qwen27b_ssm_q6_qkv`, `qwen27b_ssm_out_q8`,
  `qwen27b_lm_head_q6k`, `qwen27b_prefill_tail_hot` in
  `benchmarks/metal_q8_shapes.zig` (`790ba581`).
- Harness: `parseTokensGenerated` (`loops/implement_metal.ts`) requires the
  engine form `Generated N tokens in X ms|s`, so prompt prose like
  "generated 160 tokens" can no longer be miscounted (`d2b57ac1`).

## Dead ends (do not retry)
- Enabling the `.qwen35` fused dense Q4_K gate/up+SwiGLU decode path: fewer
  dense barriers (256→192) but slower, 10.42 tok/s.
- Non-SwiGLU dual-projection route for dense Q4_K gate/up: 12.56 → 11.07.
- Fixed-K5120 fused Q4 gate/up+SwiGLU retry (cycle 100): 14.49, reverted.
- Q4 dual nibble-mask vectorization, fixed-K5120 Q4 dual route, tail-free Q4
  dual row alignment, Q4 dual `float2` accumulator cleanup: all in-noise/worse.
- Dense-down tail barrier narrowing, copied dense weights: no gain.
- Reducing dispatch/barrier count alone does not move decode (cycle 81 was
  flat): decode is bandwidth-bound in dense-FFN kernel bodies, not CPU
  encode/command count.

## Still open (inherited by Effort 28)
- Dense FFN Q4 gate/up + Q6 down dominate decode bytes; the remaining ~15 vs
  ~21.4–22.0 gap needs kernel-body memory-traffic work, not more route
  selectors.
