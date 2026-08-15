# Effort 12 — Gemma 4 31B dense decode+prefill (Metal, M4 Max) — STATUS: CLOSED-MERGED

Started at 0.28 decode tok/s / 72 s prefill on the 14-token chat prompt; ended
at llama.cpp parity. Published Metal suite 2026-06-13
(`site/src/data/zinc-performance.json`): core decode 22.68 vs llama.cpp 22.76
tok/s (tied), prefill 131.1 vs llama.cpp 102.2 tok/s (ZINC ahead); loop best
23.7 decode. The nominal 25-tok/s target was never hit, but llama.cpp itself
sits at ~22.7 on this machine — treat parity as the ceiling. Reopen only if
the bandwidth math changes (quant tier, KV layout). Model: `gemma4-31b-q4k-m`,
60 layers, dim 5376, dense (NOT MoE — Effort 11's routed-MoE machinery does
not apply).

## Landed

- Removed the broken K=5376 specialized kernel (commit `22bbc753`); K=5376
  dense shapes route through the base llama.cpp-port `dmmv_q4k.metal`
  (NSG=2, NR0=2, 64 threads, no TGM input cache) — microbenched 491–629 GB/s
  on all eight 31B decode shapes, at/above the ~480 GB/s Apple9 ceiling.
- Fused dense gate+up+GeGLU — `src/shaders/metal/dmmv_q4k_dense_gate_up_geglu.metal`,
  dispatch kind `dense_gate_up_geglu` in `src/compute/forward_metal.zig`.
- Dense FFN tail fusion — `src/shaders/metal/post_norm_residual_rms_norm_wide.metal`
  (post-norm + residual + next-layer norm in one dispatch).
- `layer_output_scale` folded into the residual RMS norm instead of a
  standalone scale dispatch (`forward_metal.zig`, `layer_output_scales`; a
  non-1.0 scale is also a structural-batched-prefill guard).
- LM-head argmax fusion — `src/shaders/metal/dmmv_q4k_lmhead_argmax.metal`.
- Q6_K llama.cpp-port matvec — `src/shaders/metal/dmmv_q6k_llama.metal`
  (attn V / Q6_K dense shapes).
- Bench coverage — `zig build bench-metal-shapes` case `post_norm_residual_wide`,
  gated to dense Gemma dim-5376 (`benchmarks/metal_q8_shapes.zig`). (The doc's
  older `dense_q4k_5376` case name never landed — unverified.)

Key structural lesson: kernels were never the bottleneck after the k5376
deletion (all ≥ ceiling in microbench); the 57x wall-clock gap was
dispatch/barrier overhead, and the lever was fusion — nearly every barrier in
dense decode is data-dependency-required, so barrier removal without fusion
breaks correctness.

## Dead ends (do not retry)

- `dmmv_q4k_k5376.metal` (TG=512, 16 rows/TG, simdgroup-per-row, 21 KiB TGM
  input cache) → ~5 GB/s on Apple9; wrong architecture, deleted — never
  re-create TGM-cached wide-TG matvec variants.
- Repeated retunes of that kernel (unrolls, dispatch routing, simdgroup-matrix,
  dual gate/up) → 23 of 24 cycles inside noise; nothing moved.
- Command-buffer chunking / async submit for decode → CPU record is ~5 ms/step;
  irrelevant when GPU wait dominates.
- Copying tensors out of mmap into Metal-owned buffers → no gain; mmap is fine.
- Default-on batched prefill without a logits-parity `validate` gate →
  regressed (same conclusion as Effort 11).
- Editing `loops/implement_metal.ts` from inside an agent cycle → forbidden,
  reverted.
- Post-plateau neutral families (cycles 47–66): local `dmmv_q4k` /
  `dmmv_q6k_llama` variants, Q4/Q6 row-pair carry rewrites, Q6 packed-scale /
  output-tail-guard cleanup, `post_norm_residual_rms_norm_wide`
  cache/vectorization variants, LM-head argmax row grouping / 512-thread
  widening, resource-list↔scope barrier swaps → all noise; require fresh
  profile evidence naming the exact family before touching again.
- Chasing decode above ~23 tok/s with kernel work → decode is bandwidth-bound
  and already at llama.cpp parity on this card.
