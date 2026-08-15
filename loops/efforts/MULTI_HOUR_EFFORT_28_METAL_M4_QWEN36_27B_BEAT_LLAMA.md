# Effort 28 — Qwen3.6-27B prefill: beat llama.cpp (Metal, M4 Max) — STATUS: CLOSED-MERGED

Model `qwen36-27b-q4k-m` (dense hybrid SSM, 64 layers, full-attn interval 4).
Public-suite baseline: core prefill 15.9 tok/s (15% of llama.cpp's 104.6),
decode 15.4 (70% of 21.95). Root cause was structural: the public path ran
token-major replay for the whole model (`materialized_layer_tokens=0`). The
effort built a materialized layer-major prefill prefix and extended it
interval-by-interval to the full 64-layer body: **97.5 prefill tok/s**
best-kept-correct (~93% of llama.cpp core), decode guarded at ~15.6. Plateau
auto-stop after 20 no-best cycles; the cycle-23 best tree was squashed onto
main in `c8f3b203` ("perf(metal): land M4 Qwen27 prefill wins"). M4 loop work
then moved to Effort 29 (Gemma 26B).

## Landed

- Layer-major queued prefill for the 27B dense hybrid, full 64/64-layer prefix
  (no token-major replay): `qwen35DensePrefillPrefixLayerLimit()` returns
  `full_attn_interval * 16` under `defaultQwen35Dense27bSsmDeltaGatedNormEnabled`
  in `src/compute/forward_metal.zig`. Applies to 32–40-token prompts
  (`qwen_ssm_projection_prefill_min_tokens`,
  `qwen35_dense27b_queued_prefill_max_tokens`). 15.9 → ~95–97.5 tok/s on the
  36-token public core prompt.
- Full-attention 256-wide packed Q+gate shape eligibility (cleared the layer-3
  `q-shape` guard that blocked prefix growth):
  `isQwen35Dense27bFullAttnPackedQGateQ4kTarget` /
  `isQwen35Dense27bFullAttnKvQ6kTarget`. 17 → 23 tok/s at the time.
- Fused Q4_K gate/up SwiGLU GEMM + N48 tile variants:
  `src/shaders/metal/gemm_q4k_gate_up_swiglu.metal` (NR1 function constant),
  pipes `gemm_q4k_gate_up_swiglu_n48_pipe`, `gemm_q4k_n48_pipe`,
  `gemm_q6k_n48_pipe`.
- Q6_K K=5120 LM-head DMMV regrouped 4 → 16 simdgroups/threadgroup:
  `dmmv_q6k_llama_k5120_pipe` dispatched with `block_size = 512`,
  `rows_per_wg = 32`. 97.1 → 97.5 tok/s. (Prefill LM head already computes only
  the final prompt token via input offset — the `lm_head 0.97 GiB` profile
  bucket is not an all-token logits matrix.)
- Prefill evidence/profiling counters: `path evidence` line
  (`materialized_layer_tokens`/`replay_layer_tokens`), `layer-major prefill`
  summary, `next replay blocker` with per-guard status, byte-bucket and
  `hot #N` exact-shape (M/K/bytes/calls) attribution — all in
  `src/compute/forward_metal.zig`.
- Per-dispatch kernel timing probe: `ZINC_METAL_KERNEL_TIMING=1`
  (`src/metal/kernel_timing.zig`); off in normal loop runs.
- Exact-shape Qwen27 benchmark cases in `benchmarks/metal_q8_shapes.zig`
  (`qwen27b_prefill_tail_hot`, `qwen27b_decode_hot`,
  `qwen27b_dense_gate_up_q4k`, `qwen27b_ssm_*`, `qwen27b_lm_head_q6k`) —
  re-added in the landing commit, superseding this doc's earlier note that the
  benchmark did not survive the plateau restore.
- Dense-down direct-accumulate GEMM path exists but is default-OFF
  (`ZINC_QWEN35_9B_PREFILL_DOWN_ACC_GEMM`, default false) — measured non-win
  both on and off.

## Dead ends (do not retry)

- Raising the prefix layer cap while `materialized_layer_tokens=0` → pure noise; fix the named guard blocker first.
- Dual-A shared-memory fused gate/up variant → failed the synthetic comparison test.
- Fused gate/up shared-B one-tile reshuffle → correct but neutral (96.9 vs best 97.5); N48 barrier micro-reduction is low-confidence.
- Effective-N40 and physical NR1=40 gate/up tail tiles (33–40 tokens) → correct but neutral/slower; cutting the 33% column overcompute did not help, hotspot is weight decode/layout, not tail MMA.
- Generic Q4_K/Q6_K N40 dense-down route → regressed 96.7 → 94.1, reverted.
- Q6_K 48-col GEMM variant and plain Q4_K 48-col route for 33–48 tokens → neutral enablement only.
- LM-head simdgroup grouping 16 → 32 → neutral; don't push further without timing proof it's still launch-limited.
- Direct-accumulate dense-down (on) and parallelized accumulate store → neutral or worse prefix GPU time.
- Ping-pong layer-major hidden buffers → fewer barriers, neutral throughput; command scheduling/copyback cleanup is exhausted.
- Vectorized SwiGLU writeback → small non-promoted gain, not in the landed tree (writeback is scalar at HEAD).
- Prior 27B decode fused gate/up attempts → reduced dispatches but lost speed (pre-effort finding).

## Still open

- ~7% core prefill gap to llama.cpp. Byte-weighted GPU split at plateau: dense
  FFN 251 ms of 362 ms total (gate/up 155.5, down 95.6; Q6_K down is the
  largest single sub-bucket). The named remaining lever is quantized weight
  decode / memory layout inside the N48 GEMM kernels, backed by an exact-shape
  microbench (`qwen27b_dense_gate_up_q4k` in `benchmarks/metal_q8_shapes.zig`),
  not more route/tile variants.
- Decode is untouched: ~15.6 vs llama.cpp 21.4–21.9 across scenarios.
- The layer-major prefix only engages for 32–40-token prompts; context-medium /
  context-long public scenarios were never re-measured with it.
- The 27B public-suite matrix was never re-run/published after the landing —
  the site Metal target data is still dated 2026-06-13 and shows the old
  15.9 tok/s rows.
