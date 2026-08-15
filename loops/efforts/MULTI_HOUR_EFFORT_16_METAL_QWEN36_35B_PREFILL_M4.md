# Effort 16 — Qwen3.6 35B-A3B prefill (Metal, M4 Max) — STATUS: CLOSED-MERGED

Chat-prompt prefill for `qwen36-35b-a3b-q4k-xl` (hybrid SSM + full-attn + routed MoE)
went **34.1 → 101.9 tok/s (~3x)** across three loop runs, 2026-05-18..28, on the
134-token Effort 16 chat prompt. The 90 tok/s milestone fell at cycle 240 (100.7,
commit 17427b50 — simdgroup-parallel SSM F32 tail reductions); the final 50-cycle
run plateaued at 101.9 (4 kept / 46 reverted, commit a9ee924e). Everything is on
main. Postscript: a later decode loop on the same model silently regressed prefill
101.9 → ~89 through shared SSM/Q8 shaders, which is why the harness now measures the
opposite metric every cycle (`ZINC_CROSS_EFFORT_METRIC` guard,
`loops/implement_metal.ts`). Public perf suite 2026-06-13 (36-token raw prompt):
ZINC 97.4 prompt tok/s vs llama.cpp 299.2 — prompt side remains this model's largest
public gap, while decode leads llama.cpp (81.7 vs 62.5).

## Landed

- Prefill scoring + guardrails in the loop harness — `ZINC_METRIC_MODE=prefill`
  and cross-effort regression check (`loops/implement_metal.ts`).
- Token-major F32 shared-gate MoE combine — `moe_weighted_acc_shared_gate_f32*.metal`
  (incl. `_qwen2048{,_norm,_seed_norm}` finalizers that fold layer output scale and
  next-layer RMSNorm) — 36.4 → 43.4 (commit d4f69c9e).
- Early prompt graph commit — 16-token leading chunk committed before the rest of
  the prompt graph (commit 88265836) — into the mid-40s.
- Buffer/resource-scoped barriers replacing command-wide barriers across
  SSM/attn/router/MoE edges (`resource_barrier_*` accounting in
  `src/compute/forward_metal.zig`; cycles 157–199) — to 51.6.
- Private-repacked Q8 SSM weights + exact-K kernels —
  `dmmv_q8_0_repacked_k2048*/k4096*_qwen*.metal` (commit 624e865c); later reused
  heavily by the decode loop.
- SSM prefill projection chunking — layer-0 qkv/z/alpha/beta projections batched
  over ≤256 prompt tokens with the token-order recurrence untouched:
  `canUseQwenSsmPrefillProjectionChunk` (forward_metal.zig), default-on via
  `defaultQwenSsmPrefillProjectionEnabled`, override `ZINC_QWEN36_35B_SSM_PREFILL_PROJ`.
- 32-token coherence guard — `qwen_ssm_projection_prefill_min_tokens = 32`
  (forward_metal.zig): short raw prompts stay on the per-token path.
- Fused F32 router + top-k for prompt tokens, with input offsets —
  `router_f32_topk_batched.metal` (+ `rms_norm_router_f32_topk_batched`,
  `router_f32_topk_batched_shared_gate`) — 51.6 → 69.9, +35% in one cycle
  (commit 53780317).
- Fused residual+RMSNorm+router top-k — `residual_rms_norm_router_f32_topk.metal`
  (commit 0c077fdb); later the #1 decode hot kernel as well.
- SSM alpha/beta F32 dual tail kernel with simdgroup-parallel row reductions —
  `dmmv_f32_dual_small.metal` (commits a1976fcf, c2276b01, 17427b50) —
  72.1 → 100.7, the single biggest jump of the effort.
- Exact-shape SSM delta+gated-norm kernels (32x128x128, TG128) —
  `ssm_delta_net_gated_norm{,_qwen}.metal`, float4 fast path in
  `ssm_delta_net_prefill.metal` (commits a63aa0b8, a6e896b5).
- 35B validator infrastructure (default-off) — `ZINC_QWEN36_35B_PREFILL_VALIDATE`
  (+`_TOKENS`/`_LAYER`), `ZINC_QWEN36_35B_ROUTE_PACK_VALIDATE_LAYER/_FULL/_FULL_BISECT`,
  `ZINC_QWEN36_35B_ROUTE_PACK_PREFIX_LAYERS`, `ZINC_QWEN36_35B_SSM_PROJ_VALIDATE_LAYER`
  (forward_metal.zig).

## Dead ends (do not retry)

- Route-packed layer-major MoE prefill beyond layer 0 — fastest *incorrect* tree of
  the effort (106.4 tok/s, +4.6%) but never passed the Paris gate across ≥5 escalated
  bisection cycles; survives only as the opt-in candidate
  `ZINC_QWEN36_LAYER0_ROUTE_PACK_PREFILL=1`, default off.
- Narrow Q8/repacked/fixed-K/TG128 retunes (cycles 199–230) — all neutral at ~51;
  only the private repack-upload itself paid. Exact-shape same-cycle A/B or nothing.
- TG64 SSM delta dispatch — 51.0 → 50.6; TG128 restored.
- Generic structural batched prefill for the hybrid — `canUseBatchedPrefill`
  (forward_metal.zig) still rejects `n_experts > 0` / `ssm_d_inner > 0`;
  every win came from token-major fusion + targeted batching, not the Gemma-style
  structural path. Do not blindly relax the gate.
- SSM prefill projection on <32-token prompts — emits repeated `is`; the min-token
  guard is a correctness fence, not a tunable.
- Decode-loop retunes of shared SSM/Q8 shaders — regressed prefill 101.9 → ~89
  invisibly; never accept one without the cross-effort prefill check.

## Still open

- llama.cpp prompt parity (Phase-2 target was 120+; public suite shows 97.4 vs
  299.2 on short raw prompts). Needs either route-pack correctness (see dead end
  above) or a true batched hybrid prefill with per-layer validation — no successor
  effort owns this yet.
