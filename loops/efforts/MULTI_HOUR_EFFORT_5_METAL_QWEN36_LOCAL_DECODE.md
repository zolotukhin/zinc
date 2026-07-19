# Effort 5 — Qwen 35B-A3B local decode (Metal, M4 Max) — STATUS: CLOSED-MERGED

April 2026 effort to close the local decode gap on the 35B A3B model
(baseline 38.11 tok/s vs llama.cpp 52.83). During the effort window decode
reached ~40.7 (loop cycles) and 47.57 tok/s (Apr 18 suite); follow-on Metal
loop efforts (16 prefill, 24/28 on 27B) continued the same kernel program.
Current published suite (2026-07-08): **zinc 81.7 tok/s decode vs llama.cpp
62.5** on `qwen36-35b-a3b-q4k-xl` — all targets (45/50/53-parity) exceeded.
Note: the doc's numbers predate the May model swap Qwen3.5-35B → Qwen3.6-35B
(commit `e8485949` renamed references in place).

## Landed
- Q8_0 weight repack for SIMD-coalesced access (`b9a0176d`) + nr=2 shader
  coherence fix and Metal Q8_0 test suite (`f563058d`, 2026-04-11) — the
  effort-window decode wins (38.1 → 40.7 → 47.6 tok/s by Apr 18).
- Shape-exact repacked/fixed-K Q8_0 kernel family (this effort's Step 3
  strategy, extended by later loop efforts):
  `src/shaders/metal/dmmv_q8_0_repacked_k2048_nr2_qwen.metal` (+ `_dual_nr2_qwen`,
  `_conv1d`, `_quad`), `dmmv_q8_0_k2048{,_fused_norm,_quad}.metal`,
  `dmmv_q8_0_k4096{,_quad}.metal`; routing in `src/compute/forward_metal.zig`
  (~line 647: LM head 248320×2048 → fixed-K nr=4 repacked path).
- GPU-side greedy argmax incl. fused LM-head variant: `argmax{,_chunks,_pairs}.metal`,
  `dmmv_q4k_lmhead_argmax.metal`, `dmmv_q8_0_argmax.metal`; pipes in
  `forward_metal.zig`; CPU logits only when sampling needs them.
- Benchmark safety: bench tools take the per-GPU process lock
  (`src/main.zig:274`); `bench-metal` / `bench-metal-shapes` steps
  (`build.zig`, `benchmarks/metal_inference.zig`, `metal_q8_shapes.zig`).

## Dead ends (do not retry)
- llama.cpp `--no-repack` A/B → flat (52.87 tok/s); repack is not llama.cpp's edge.
- Broad threadgroup override sweeps → neutral; no blind launch sweeps.
- Generic `q8_0` launch overrides → never beat baseline.
- Shared-expert dual-`q8_0` reuse → no whole-model win, not kept.
- `q8_0` KV vs `f32` KV on Metal → flat (36.49 vs 36.45 tok/s); not a decode lever.
- Overlapping benchmarks on one GPU → both runs collapse to ~30 tok/s;
  the process lock exists precisely for this.
- Repack work, prefix-cache/APC ideas, sampling changes → scoped out then,
  still not decode levers.
