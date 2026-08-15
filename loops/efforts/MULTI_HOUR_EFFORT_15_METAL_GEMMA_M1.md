# Effort 15 — Gemma 4 26B-A4B MoE decode (Metal, M1 Max) — STATUS: OPEN (never run)

Plan authored 2026-05-23; as of 2026-07-18 **zero cycles have executed** (no M1 Gemma
metal-loop commits exist; the notes file was never filled in). Goal: close ZINC decode
tok/s vs local llama.cpp for `gemma4-26b-a4b-q4k-m` (Gemma 4 26B-A4B MoE, ~4B active,
Q4_K_M, 16.9 GB) on the local M1 Max (Apple7, 32 GPU cores, 32 GB UMA, ~400 GB/s).
Decode speed only; chat mode only (raw completion emits `is is is` and is not a valid
coherence gate). M4 sibling: Effort 11 (`MULTI_HOUR_EFFORT_11_METAL_GEMMA_M4.md`) —
same model, shared kernels, so M1 already inherits everything that landed there. Read
Effort 11's dead ends first; retrying one on M1 requires a fresh `bench-metal-shapes`
number (Apple7 ≠ Apple9), never a hunch.

## Reproduction

```bash
ZINC_MODEL_ID=gemma4-26b-a4b-q4k-m \
ZINC_PROMPT_MODE=chat \
ZINC_TEST_PROMPT="What is the capital of France?" \
ZINC_MAX_TOKENS=128 \
ZINC_TARGET_TOK_PER_SEC=40 ZINC_STOP_ON_TARGET=0 \
ZINC_BENCHMARK_RUNS=5 ZINC_PROFILE_EVERY=1 \
ZINC_BUILD_OPTIMIZE=ReleaseFast \
bun loops/implement_metal.ts --effort 15 --agent claude --cycles 80
# add --resume on every run after the first; a fresh start wipes .metal_optimize/
```

Manual A/B (1 warmup + 3 timed runs, median; `zig build -Doptimize=ReleaseFast` only):

```bash
./zig-out/bin/zinc --model-id gemma4-26b-a4b-q4k-m \
  --prompt "What is the capital of France?" --chat -n 128 --profile
~/Workspace/llama.cpp/build-metal/bin/llama-cli \
  -m ~/Library/Caches/zinc/models/models/gemma4-26b-a4b-q4k-m/model.gguf \
  -p "What is the capital of France?" -n 128 -ngl 99 -fa on --temp 0 --no-warmup
```

Greedy chat output must stay byte-identical and contain `Paris`. `--profile`
aggregates the ~9 s chat prefill with decode — separate the decode-only slice.
Create `loops/efforts/EFFORT_15_NOTES.md` on the first real run.

## Environment facts (verified in code)

- **Q8 KV cache is unavailable — do not propose it.** Gemma 4 has sliding-window
  attention; `defaultKvCacheQ8Enabled` returns false for `.gemma` with
  `rope_freq_base_swa > 0` (src/compute/forward_metal.zig, test
  "defaultKvCacheQ8Enabled disables Gemma ISWA q8 KV cache"). Requires an ISWA
  quantized-cache rotation first — out of scope.
- **`.gemma`-gated kernel unblocking is not the lever** (Effort 14's big win):
  this model *is* `.gemma`, those gates are already open.
- **Already landed via Effort 11 — do not reimplement:** GPU-routed MoE, batched
  top-k routing, route packing, `ZINC_GEMMA_MOE_VALIDATE=1` parity mode
  (forward_metal.zig).

## Kernel map (decode N=1, ZINC ↔ llama.cpp `ggml-metal.metal`)

llama.cpp reference lives at `~/Workspace/llama.cpp/ggml/src/ggml-metal/`
(`ggml-metal.metal`, `ggml-metal-ops.cpp`, `ggml-metal-device.m`). Read-only;
extract techniques, don't copy code.

| Op | ZINC (`src/shaders/metal/`) | llama.cpp |
|---|---|---|
| Q4_K matvec attn Q/K/V/O | `dmmv_q4k.metal`, `dmmv_q4k_qk_dual.metal` | `kernel_mul_mv_q4_K_f32` |
| Flash attention (SWA decode) | `flash_attn.metal` | `kernel_flash_attn_ext_vec` |
| Router logits + top-k | `softmax_topk.metal` | soft_max + argsort |
| Expert gate/up + GeGLU | `dmmv_q4k_moe_gate_up{,_geglu}.metal`, `geglu.metal` | `kernel_mul_mv_id_*` |
| Expert down (Q5_1) | `dmmv_q5_1_moe{,_cols}.metal` | `kernel_mul_mv_id_*` |
| Weighted expert accumulate | `moe_accumulate.metal`, `moe_weighted_acc_{scaled,shared}.metal` | expert-weighted sum |
| LM head | `dmmv_q4k_lmhead{,_norm,_1024}.metal` | `kernel_mul_mv_q4_K_f32` |
| Norms / RoPE | `rms_norm_mul.metal`, `residual_rms_norm.metal`, `add_rms_norm.metal`, `rope_{native,fused,kv_cache_write}.metal` | `kernel_rms_norm_fuse_impl`, `kernel_rope_*` |

## Still open (the entire plan)

- Phase 0: lock ZINC + llama.cpp baselines and decode-slice profile on this M1.
- Structural lead: dense fast paths (`shouldDefaultDenseGemmaBatchedPrefill`,
  `canUseDenseGemmaBatchedPrefill`, and the `n_experts == 0` branches in
  `canUseBatchedPrefill`) have no routed-MoE equivalent — check whether the
  hottest decode bucket is one where the MoE path misses a dense-only fast path.
- Loop discipline: one change per cycle; microbench on the exact hot shape with
  `bench-metal-shapes` (≥5% gate) before whole-model (≥2% / 2× noise gate);
  byte-identical greedy output; Metal-only scope (`src/shaders/metal/`,
  `src/metal/`, Metal branches of `src/gpu/` + `forward_metal.zig`); local
  commits only.
- Terminate on: llama.cpp parity, or 3 consecutive microbench-gate failures on
  3 different kernels, or 6 accepted cycles.
