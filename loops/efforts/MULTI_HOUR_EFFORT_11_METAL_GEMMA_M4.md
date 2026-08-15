# Effort 11 — Gemma 4 26B-A4B MoE prefill+decode (Metal, M4 Max) — STATUS: SUPERSEDED by Effort 29

Started at 34 prefill / 30 decode tok/s (26% of llama.cpp overall). Shipped the
GPU-routed structural route-packed MoE prefill path plus decode fusions; the
local harness went 36 → 370 prefill tok/s (breakthrough: cycle 50, Q8_0 MoE
route-column support + clearing a stale `layer_output_scale` structural guard,
90.8 → 329.7), decode ~70 tok/s. Published Metal suite 2026-06-13: 334.8
prefill / 69.43 decode vs llama.cpp 411.8 / 83.5 (81% / 83%). The remaining
public-suite gap is owned by `MULTI_HOUR_EFFORT_29_METAL_M4_GEMMA26_PUBLIC_GAP.md`
— do all new Gemma-26B M4 work there.

## Landed

- GPU-routed batched MoE with a Gemma branch (fused q4_k gate/up, scaled
  expert down, GeGLU, shared expert, post-MoE norm) —
  `canUseGpuRoutedBatchedMoe()` in `src/compute/forward_metal.zig`.
- Structural route-packed batched prefill — `canUseGemmaBatchedPrefill()`;
  profile path names `batched-route-pack` / `batched-route-pack+queued-replay`;
  profile line reports `structural_batched=… route_layers=…`. On the public
  prompt: `structural_batched=yes route_layers=30`.
- Q8_0 MoE route-column prefill — `src/shaders/metal/dmmv_q8_0_moe_cols.metal`
  (the cycle-50 4x unlock).
- Queued token-major prefill fallback with chunk scheduling — `queued_prefill_*`
  profile fields; best exact-20 schedule `[1,5,7,7]`.
- Route-slot machinery — `router_f32_topk_batched(.metal/_shared_gate)`,
  `moe_route_pack(.metal/_blocks)`, `moe_route_ids`, `moe_route_gather`;
  `moe_route_input` scratch sized in token rows (not route-slot rows), gate/up
  reads token rows via `route / k`.
- Grouped expert column kernels with 1–4-route tail specializations —
  `dmmv_q4k_moe_cols(.metal/_geglu)`, `dmmv_q5_1_moe_cols`, `dmmv_q5k_moe_cols`,
  `dmmv_q6k_moe_cols`, `dmmv_q4k_moe_gate_up_geglu` (Q5_1 tails moved 340→368,
  Q4_K GeGLU tails 364→370).
- Scatter/post-norm tail — `moe_route_scatter_scaled`,
  `moe_route_scatter_direct_scaled`, `gemma_moe_post_norm_residual` (+ weighted
  variants).
- Validation guards — `ZINC_GEMMA_MOE_VALIDATE=1`,
  `ZINC_GEMMA_BATCHED_PREFILL=1|validate` (logits parity vs per-token path).
- Exact-shape microbench — `zig build bench-metal-shapes` with case
  `gemma26_prefill_hot` (`benchmarks/metal_q8_shapes.zig`).

## Dead ends (do not retry)

- Q5_1 MoE-down threadgroup width retunes (128/256) → regressed.
- Q5_1 four-rows-per-workgroup variants → regressed twice.
- Q5_1 exact-6-route tail; Q4_K exact-4-route tail; route-pack alt4/block-width
  changes → regressed.
- Store-only MoE accumulate (skip zero-fill) → regressed.
- No-logits zero-token prefill → no total-time gain.
- GPU Q8 LM head for Gemma → GPU wait rose, total time worse.
- CPU Q8 LM-head variants (heap alloc, worker scheduling, row pairing, 16-lane
  dot, skip-store, GPU argmax) → regressed or broke tests.
- Isolated GPU post-MoE post-norm/residual tail micro-change → regressed.
- Fused Q5_1 expert-down + weighted accumulate → regressed.
- Grouped Q4_K column input-addressing changes → regressed.
- Gemma K-as-V V-unit-norm handling in batched prefill → regressed.
- vLLM-style projection-to-activation fused expert kernel → badly regressed.
- Shared-expert dual Q4_K gate/up fusion; 64-thread Q4_K fused gate/up groups
  → regressed.
- Q4_K K-cap widening beyond `K <= 3072` → K=4096 is known bad.
- Shared Q8 gate/up TG=512; shared-expert down Q8 via Apple9 512-thread path
  → regressed.
- Large attention Q/O quad-row Q8; 256-thread paired Q8 K/V override;
  K=2816-specialized paired Q8 shader; fused paired Q8 shared gate/up GeGLU
  → all regressed.
- Q8 `attn_output.weight` K=4096 special shader and repacked layout →
  regressed; the accepted path is the 256-thread runtime Q8 shader.
- Default-on batched prefill without logits-parity validation → regressed.
- Weighted-finalizer, sigmoid/cache, and narrow Q8 finalizer retunes → never
  beat the promoted best.
- Queued-prefill schedule `[1,6,7,6]` → worse than `[1,5,7,7]`.
- Overlapping expert-down/shared GeGLU → regressed.
- mmap→Metal-owned buffer tensor copies → no gain (mmap is fine).
