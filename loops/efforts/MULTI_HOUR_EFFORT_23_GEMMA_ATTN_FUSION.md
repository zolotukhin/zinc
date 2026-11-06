# Effort 23 — gemma attention-path kernel fusion (continue the dense decode lever)

> **Status:** COMPLETE (2026-06-11) — 5 stacked validated dense wins (T1/T3/T4/T5/T6
> on `perf/e23-*` branches, ~+7–8% gemma-31b dense decode); dense fusion lever
> exhausted at this box's ±1% boost-noise floor, remaining fusions are below-floor or
> MoE-only and need clock-locking (sudo unavailable). See the final cycle-log entry.
> Scoped continuation of the two fusions already on main
> (`d134c5a1`): `rms_norm_residual` (post-attn/ffn norm+residual) and
> `rms_norm_rope` (per-head Q/K norm+RoPE) → gemma-31b dense ~28→~30 tok/s
> (~+7%). This effort fuses the **remaining small per-layer dispatches** on the
> gemma attention path, same lever.

Date opened: 2026-06-10. Pairs with `forward_cuda_gemma.zig` + `kernels.cu`.

## Why (the lever, already proven)

gemma decode is **boost-saturated but launch-latency/occupancy bound** (NOT
sync-bound — async is a proven NO-OP/regression, do not touch the command ring).
The only lever is **fewer/fatter kernels**: collapse tiny per-layer dispatches so
the GPU is fed. Two fusions landed (~+4% and ~+6%, stacking to ~+7%). A handful
of small dispatches remain.

## Targets (one per cycle, smallest-risk first)

1. **Per-head V rms_norm (`rms_norm_noweight`) + KV-cache write.** V is normed
   (no weight, never roped) then written to the KV cache — two tiny launches.
   Fuse into one `rms_norm_kvwrite` kernel (norm per head_dim, write to kv_v at
   `pos*kv_dim`). Bit-equivalent.
2. **Pre-attention `rms_norm` → fold the residual-stream read** if a write+read
   round-trip can be dropped (the norm output feeds Q/K/V matvecs; check whether
   a fused norm-into-first-matvec is worth it, else skip — matvecs dominate).
3. **FFN path**: any remaining `rms_norm`/elementwise pairs not already covered
   by `rms_norm_residual` (gemma FFN: pre-ffn norm → GeGLU → post-ffn fused).

## Hard rules (from memory — violating wastes the cycle)

- **Build:** isolated caches ALWAYS — `ZIG_LOCAL_CACHE_DIR=/tmp/lc-$RANDOM
ZIG_GLOBAL_CACHE_DIR=/tmp/gc-$RANDOM ./scripts/zig.sh build cuda-dbg
-Dbackend=cuda -Dshaders=false`; verify the binary hash CHANGED.
- **Do NOT async gemma** (boost-saturated; regression). Fusion only.
- **4090-pinned** (`${ZINC_GPU}`); isolated box dir (`~/zinc-e23`), never
  `~/workspace/zinc`; do not disturb the parallel 5090 work.
- **Boost noise:** interleaved back-to-back A/B (NGEN≥160), same thermal state.
  Never claim a win from one boosted run.

## Validation contract (per cycle)

- `scripts/validate_catalog.sh` → **5/5 token-correct** (the fused kernel must be
  bit-equivalent: token-identical to the pre-fusion path). Break correctness →
  REVERT, document, do not commit.
- Interleaved A/B vs the pre-cycle binary; require a measurable, repeatable gain.
- Commit ONLY this effort's change to `perf/e23-<target>`; push (NOT main).
  Update the cycle log below. Negative result → revert + log it.

## Cycle log

- **2026-06-11 — Target 1 DONE (V rms_norm+KV-write fuse + K KV-write fold). VALIDATED WIN, branch `perf/e23-vk-norm-fuse`, pushed (NOT main).**
  New kernel `rms_norm_kvwrite` (kernels.cu): per-head V plain-normalize (no
  weight) writing STRAIGHT into the V cache at `pos*kv_dim` — replaces the
  `rms_norm_noweight(→v_buf)` + the V half of `kv_cache_write`. Also extended
  `rms_norm_rope` with a `dst_offset` push field so the K head writes its
  norm+roped result straight into `kv_k` at `pos*kv_dim` (Q passes 0 → own
  buffer), folding the K half of `kv_cache_write`. Net: the standalone
  `kv_cache_write` launch + the V round-trip are gone → **2 fewer tiny
  dispatches/layer** on the gemma attention path (both dense + MoE). Bit-equivalent
  (identical normalization, identical cache layout). `kv_cache_write` pipeline
  removed.
  **Build:** isolated caches; variant md5 `8e098573…` ≠ baseline `4278b4c6…`
  (real recompile, not stale). **Correctness:** `validate_catalog` → **5/5
  token-correct** (qwen 3×12/12; gemma4-26b 12/12; gemma4-31b teacher-forced
  11/12 = the documented near-tie, unchanged). **Perf (interleaved back-to-back
  A/B, NGEN=160, 4090, warmup ignored):** gemma-31b DENSE variant wins **4/4
  rounds** — 29.90/29.43, 29.81/29.39, 29.84/29.42, 29.89/29.39 → mean 29.86 vs
  29.41 = **+1.54%** (rock-steady, no boost noise). gemma-26b MoE (same attn-path
  fusion, FFN untouched) 3 rounds 52.18/52.84, 50.65/48.86, 54.08/48.48 → mean
  52.30 vs 50.06 = **+4.48%, net-positive, NO regression** (round 1 near-tie,
  boost-variable). Complements the two landed fusions (norm+residual, qk-norm+rope)
  — different part of the layer, stacks.
  **NEXT (Target 2/3):** pre-attention `rms_norm` fold into the first Q/K/V matvec
  (likely skip — matvecs dominate), or any remaining FFN norm/elementwise pairs.
- **2026-06-11 — Target 3 DONE (per-layer output-scale fold). VALIDATED WIN, branch `perf/e23-los-fold` (stacked on Target 1, commit `e45d7475`), pushed (NOT main).**
  The dense gemma path applied the learned per-layer output scale
  (`blk.N.layer_output_scale.weight`, confirmed present on all 60 gemma-31b
  layers) as a standalone `scalar_mul` command after each FFN block — one tiny
  launch + one command submission per layer. Since it is the layer's LAST write
  to the residual stream, folded it into the post-ffn `rms_norm_residual` via a
  new `rms_norm_residual_scale` kernel: `hidden = s[0]*(hidden + w*x/rms(x))`.
  `ffnBlock` uses it when the scale tensor is present (else plain
  `rms_norm_residual`); `layerOutScale` now self-skips dense layers (only the MoE
  path — final write is a `scale_accumulate` — keeps the standalone op). Removes
  **60 launches + 60 command submissions/token** on the dense path. **Bit-exact**:
  the residual add is the same FMA producing the same f32 value `scalar_mul`
  would re-load, then a plain `*s[0]` — verified gemma-31b 32-tok free-run
  IDENTICAL between baseline and variant binaries. **Build:** isolated caches,
  md5 `a0b2f95b…` baseline → `08c44e74…` variant (real recompile). **Correctness:**
  `validate_catalog` → **5/5 token-correct** (gemma-31b teacher-forced 11/12 = the
  documented near-tie, unchanged). **Perf (interleaved A/B, NGEN=160, 4090,
  warmup ignored):** gemma-31b DENSE — batch1 (4 rounds) variant 30.93 vs 30.56 =
  **+1.19%** (3/4); batch2 (6 rounds) 31.36 vs 30.75 = **+1.97%** (5/6); pooled 10
  rounds variant wins **8/10**, 31.19 vs 30.68 = **+1.66%**. Stacks on Target 1
  (different part of the layer). **Target 2 SKIP confirmed** — pre-attn norm feeds
  3 matvecs (Q/K/V), pre-ffn norm feeds 2 (gate/up); no single-consumer fold and
  the matvecs dominate. Effort 23 net: 2 stacked dense-decode fusion wins
  (~+1.5% Target 1, ~+1.7% Target 3).
- **2026-06-11 — MoE output-scale fold (Target 3 extended to the MoE path). INCONCLUSIVE → REVERTED, not committed.**
  Target 3 fused the per-layer output scale into the dense post-ffn norm+residual
  but explicitly **self-skipped the MoE path** (its final `hidden` write is a
  `scale_accumulate`, not an `rms_norm_residual`), leaving a standalone
  `scalar_mul` (the `layer_output_scale.weight`, confirmed present on the
  gemma4-26b layers) per MoE layer. This cycle folded that scale into the MoE
  combine's final residual add via a new kernel `scale_accumulate_scale`
  (`hidden = s[0]*(hidden + scale*cur)`); `layerOutScale` became a self-skipping
  no-op (both paths now fold). Removes **60 launches + 60 command
  submissions/token** on the gemma4-26b MoE path. **Bit-exact** (commutative
  float multiply over the same residual-add FMA `scalar_mul` would re-load).
  **Build:** isolated caches, md5 `43938aee…` baseline → `304b9936…` variant
  (real recompile). **Correctness:** `validate_catalog` → **5/5 token-correct**,
  with **gemma4-26b 12/12 free-run** (the changed path is bit-identical;
  gemma4-31b dense unchanged, 11/12 teacher-forced near-tie).
  **Perf — UNMEASURABLE in boost noise.** Interleaved A/B (4090, warmup ignored):
  batch1 NGEN=160 ×8 → variant 46.60 vs 44.77 = **+4.10%** (won 5/8); batch2
  NGEN=320 ×10 → variant 46.64 vs 46.31 = **+0.69%** (won 5/10). Pooled 18 rounds:
  variant won **10/18 (55%, coin-flip)**, mean **+2.17%**, but the two batches
  disagree by more than the effect and per-round swings are ±40% (e.g. b1 r2
  baseline +36%, b1 r7 variant +42%; b2 r2 variant +49%, b2 r6 baseline +30%).
  The 26b MoE clock-boosts erratically per fresh process regardless of NGEN, so a
  ~1-3% signal cannot be resolved here. Per the contract ("require a measurable,
  repeatable gain"; "never claim a win from one boosted run") this is **NOT a
  validated win** — and unlike Target 1, the change touches ONLY the MoE path so
  there is no clean dense component to anchor it. **Reverted** (working tree +
  box variant tree restored to HEAD; iso caches cleared). The change is correct
  and strictly fewer-ops (true effect ≥0, never a regression) — recoverable from
  this log for a future **clock-locked** cycle (`nvidia-smi -lgc`, needs sudo the
  agent lacks) that could resolve sub-5% MoE effects. Takeaway: the boost-noisy
  26b MoE path is not A/B-measurable for small fusions on this box without pinned
  clocks; dense-path fusions remain the only cleanly-measurable gemma lever here.
- **2026-06-11 — Target 4 DONE (same-input Q4_K matvec-pair fusion). VALIDATED WIN, branch `perf/e23-dual-matvec` (stacked on T1+T3, commit `067e71df`), pushed (NOT main).**
  Beyond the enumerated T1–T3, the dense path still ran each same-input matvec
  PAIR as two separate launches: FFN `ffn_gate`+`ffn_up` over the pre-ffn norm,
  and attention `attn_q`+`attn_k` over the pre-attn norm (all four Q4_K on
  gemma-31b, confirmed via GGUF: gate/up M=21504 K=5376; q M=8192 / k M=4096
  K=5376). New kernel `dmmv_q4k_fast_dual(a0,a1,x,y0,y1,{M0,M1,K})` fuses a pair
  into ONE launch over `M0+M1` blocks (block `bx<M0` → `a0` row `bx` → `y0`, else
  `a1` row `bx-M0` → `y1`). The per-row Q4_K compute was factored into a shared
  `__device__` helper `zinc_dmmv_q4k_fast_sum` that the standalone `dmmv_q4k_fast`
  now ALSO calls — so each fused block is **bit-identical** to the unfused fast
  path (one arithmetic path, no copy-drift). Wired at both gemma sites behind a
  per-call `both-Q4_K` guard (else the unfused two-launch path). Removes **~2
  kernel-launch boundaries/layer** (~120/token on the 31b: FFN gate/up + attn
  Q/K). **Build:** isolated caches, baseline md5 `3c7bd54c…` → variant `e531f52f…`
  (real recompile; the `.cu` is `@embedFile`'d into the binary). **Correctness:**
  `validate_catalog` → **5/5 token-correct** (qwen 3×12/12; gemma4-26b 12/12;
  gemma4-31b teacher-forced 11/12 = the documented near-tie, unchanged). **Perf
  (interleaved back-to-back A/B, NGEN=160, 4090, warmup ignored):** gemma-31b
  DENSE — batch1 6 rounds variant wins **6/6**, 31.31 vs 30.80 = **+1.67%**;
  batch2 6 rounds variant wins **5/6** (round 1 baseline edged on boost), 31.18
  vs 30.89 = **+0.92%**; **pooled 12 rounds variant wins 11/12, 31.24 vs 30.85 =
  +1.30%** (rock-steady, all winning rounds +0.7%…+2.2%). Stacks on T1 + T3
  (different launches). The attn Q/K fusion also runs on the gemma-26b MoE attn
  path (bit-exact; not separately A/B'd — the 26b MoE is boost-unmeasurable, per
  the prior cycle). Effort 23 net: **3 stacked dense-decode fusion wins** —
  ~+1.5% T1, ~+1.7% T3, ~+1.3% T4.
- **2026-06-11 — Q/K per-head norm+rope fusion (fuse the two rms_norm_rope launches into one). INCONCLUSIVE → REVERTED, not committed.**
  Found uncommitted in the working tree at cycle start (a prior loop cycle wrote it
  but never validated). Beyond T1–T4: the attn path still ran TWO `rms_norm_rope`
  launches/layer — Q (q_buf, dst_offset 0) and K (k_buf → kv_k at `pos*kv_dim`).
  Both share this layer's head_dim/rope_dim/inv_freq/position, so they fuse into one
  launch over `n_head + n_kv_head` blocks (block `< n_head` → Q head; else K head
  into kv_k). The per-head norm+rope arithmetic was factored into a shared
  `__device__ zinc_rms_rope_head` that the standalone `rms_norm_rope` ALSO calls →
  each fused block is **bit-identical** to the unfused launch (same pattern as T4's
  `zinc_dmmv_q4k_fast_sum`). New kernel `rms_norm_rope_qk` + push `RmsRopeQkPush`.
  Removes **one tiny per-head launch/layer** (60/token on the 31b). **Build:**
  isolated caches, baseline md5 `4a28f621…` → variant `3ca1c880…` (real recompile).
  **Correctness:** `validate_catalog` → **5/5 token-correct** (qwen 3×12/12;
  gemma4-26b 12/12; gemma4-31b teacher-forced 11/12 = the documented near-tie,
  unchanged — confirms bit-equivalence). **Perf — UNMEASURABLE in boost noise.**
  Interleaved A/B (4090, warmup ignored), THREE batches that **disagree in sign**:
  b1 NGEN=160 ×6 → variant 31.41 vs 31.38 = **+0.12%** (won 5/6); b2 NGEN=160 ×8 →
  31.63 vs 31.24 = **+1.24%** (won 7/8); b3 NGEN=200 ×8 → 31.32 vs 31.65 =
  **−1.04%** (baseline won 6/8). Pooled 22 rounds: variant 31.46 vs baseline 31.43
  = **+0.10%** (noise floor). The tell: the **variant binary is rock-steady across
  ALL batches** (per-round 31.09–31.53) while the **baseline swings per-process**
  (batch means 31.38 / 31.24 / 31.65) — the apparent b1/b2 gain was just baseline
  running in a low-boost regime, erased when b3's baseline ran hot. Same coin-flip
  signature that rejected the MoE output-scale fold. This fusion removes only ONE
  tiny per-head launch/layer (vs T4's ~2 matvec-pair launches), and that saving sits
  **below this box's ~±1% per-process boost-noise floor** without locked clocks.
  Per the contract ("measurable, repeatable gain"; "never claim a win from one
  boosted run") this is **NOT a validated win**. **Reverted** (working tree + box
  variant tree restored to HEAD; iso caches cleared). The change is correct and
  strictly fewer-ops (true effect ≥0, never a regression) — recoverable from this log
  for a future **clock-locked** cycle (`nvidia-smi -lgc`, needs sudo the agent lacks).
  **Takeaway: the cheap bit-exact dense fusions are now exhausted at this box's
  measurability floor — T1/T3/T4 (each ~+1.3–1.7%, removing ≥2 launches/layer or a
  full round-trip) were the resolvable wins; sub-1% single-tiny-launch fusions like
  this one are below the boost-noise floor and need locked clocks to A/B.**
- **2026-06-11 — Target 5 DONE (V/Q/K per-head norm 3→1 launch fuse). VALIDATED WIN, branch `perf/e23-qkv-norm-fuse` (stacked on T1+T3+T4, commit `984535f7`), pushed (NOT main).**
  Found uncommitted in the working tree at cycle start (a prior loop cycle wrote it,
  never validated). This is the AGGRESSIVE superset of the rejected Q/K-only fusion: it
  collapses ALL THREE per-head norm launches the attention path runs after T1 — V
  plain-normalize+KV-write (`rms_norm_kvwrite`), Q norm+rope and K norm+rope
  (`rms_norm_rope` ×2) — into ONE kernel `rms_norm_rope_qkv` over `n_head + 2*n_kv_head`
  blocks: Q heads (norm+rope → q_buf, offset 0), K heads (norm+rope → kv_k at
  `pos*kv_dim`), V heads (plain norm, no weight/rope → kv_v at `pos*kv_dim`). Each
  branch's arithmetic is copied verbatim from the standalone kernels → **bit-identical**.
  No cross-block hazard (Q in-place, K→kv_k, V→kv_v; no block reads a buffer another
  writes), so `v_src` aliasing `k_buf` on full-attention layers stays safe (k_buf is
  read-only here). Removes **2 tiny launch boundaries/layer** on the gemma attention
  path (dense + MoE; ~120/token on the 31b) — the SAME magnitude as T4, vs the rejected
  Q/K-only fusion which removed only ONE and sat below the noise floor. **Build:**
  isolated caches, baseline md5 `699acae2…` → variant `bd2d7f39…` (real recompile).
  **Correctness:** `validate_catalog` → **5/5 token-correct** (qwen 3×12/12; gemma4-26b
  **12/12 free-run** = the MoE attn path is bit-identical; gemma4-31b teacher-forced
  11/12 = the documented near-tie, unchanged). **Perf (interleaved back-to-back A/B,
  NGEN=160, 4090, warmup ignored) — 5 batches, ALL POSITIVE:** b1 31.76/31.18 +1.87%
  (6/6), b2 31.94/31.39 +1.77% (5/6), b3 31.63/31.49 +0.43% (4/6, baseline ran hot —
  variant still won the batch), b4 31.91/31.14 +2.47% (6/6), b5 31.64/31.10 +1.74%
  (6/6). **Pooled 30 rounds: variant 31.78 vs baseline 31.26 = +1.65%, won 27/30**
  (rock-steady — variant 31.55–31.94 across all batches). Unlike the rejected Q/K-only
  and MoE-scale fusions (whose batches DISAGREED IN SIGN), every batch here is positive
  and the variant wins even when the baseline boosts hot — the T4-resolvability
  signature (removes ≥2 launches/layer). This SUPERSEDES the earlier "cheap bit-exact
  dense fusions exhausted" conclusion: the Q/K-only fuse was below floor because it
  removed ONE launch, but folding V in too crosses back above the floor. Effort 23 net:
  **4 stacked dense-decode fusion wins** — ~+1.5% T1, ~+1.7% T3, ~+1.3% T4, +1.65% T5.
- **2026-06-11 — Target 6 DONE (fold each block's INPUT norm into the PRECEDING
  block's output norm+residual). VALIDATED WIN, branch `perf/e23-norm-boundary-fuse`
  (stacked on T1+T3+T4+T5, commit `07477d65`), pushed (NOT main).** The dense gemma
  layer ran FOUR single-block n_embd reductions: pre-attn `rms_norm`, post-attn
  `rms_norm_residual`, pre-ffn `rms_norm`, post-ffn `rms_norm_residual_scale`. Key
  observation: a post-norm-residual's output (`hidden`) is EXACTLY the input the very
  next pre-norm reads, and BOTH are one-block (grid `{1,1,1}`, 256 threads) reductions
  over the same n_embd vector — so the next pre-norm can run in the SAME launch, right
  after the residual add (a `__syncthreads` makes the just-written `hidden` visible to
  the phase-2 reduction; the intervening barriers also make reusing
  `zinc_block_reduce_sum`'s shared scratch race-free). Two new kernels
  `rms_norm_residual_norm` / `rms_norm_residual_scale_norm` (phase 1 = the exact
  `rms_norm_residual[_scale]` arithmetic, phase 2 = the exact `rms_norm` arithmetic
  re-reading `hidden` from global) fold TWO boundaries per layer: post-attn-residual +
  pre-ffn-norm (within the layer → `ffn_norm_buf`), and post-ffn-residual + the NEXT
  layer's pre-attn-norm (across the layer boundary → `norm_buf`, into layer L+1's
  `attn_norm.weight`). `attentionLayer` skips its pre-attn norm for L>0 (filled by
  ffnBlock(L-1)); only layer 0's pre-attn and the last layer's post-ffn stay
  standalone. **Removes ~2 norm launches/layer (~119/token over 60 layers)** — the same
  ≥2-launch magnitude as T4/T5, the resolvable regime. Gated on `d.n_experts==0` so the
  gemma-26b MoE path (which uses `norm_buf` as router scratch) is BIT-IDENTICAL /
  untouched. Cross-block buffer deps are safe: all commands chain on the one auto-ordered
  CUstream. **Bit-exact:** gemma-31b 40-tok free-run **IDENTICAL** between baseline and
  variant binaries. **Build:** isolated caches, baseline md5 `11bf6728…` → variant
  `44728767…` (real recompile). **Correctness:** `validate_catalog` → **5/5 token-correct**
  (qwen 3×12/12; gemma4-26b **12/12 free-run**; gemma4-31b teacher-forced 11/12 = the
  documented near-tie, unchanged). **Perf (interleaved back-to-back A/B, 4090, warmup
  ignored) — 3 batches ALL POSITIVE:** b1 (6×160) variant 32.50 vs 31.75 = **+2.36%**
  (6/6); b2 (6×160) 32.29 vs 32.09 = **+0.64%** (4/6, baseline boosted hot rounds 1&4 —
  variant still won the batch); b3 (8×200) 32.16 vs 31.74 = **+1.31%** (7/8). **Pooled 20
  rounds: variant 32.30 vs baseline 31.85 = +1.41%, won 17/20.** Variant rock-steady
  (32.09–33.38) while baseline swings per-process (31.47–32.94) and the variant wins even
  when the baseline boosts hot — the T4/T5 resolvability signature, OPPOSITE the
  sign-disagreeing rejected fusions. Stacks on T1+T3+T4+T5 (the four norms it touches are
  distinct launches from the matvec/per-head-norm fusions). Effort 23 net: **5 stacked
  dense-decode fusion wins** — ~+1.5% T1, ~+1.7% T3, ~+1.3% T4, +1.65% T5, +1.41% T6.
- **2026-06-11 — FFN gate+up matvec + GeGLU 3→1 launch fuse. INCONCLUSIVE
  (sign-disagreeing, leans NEGATIVE) → REVERTED, not committed.** Found
  uncommitted in the working tree at cycle start (a prior loop cycle wrote it,
  never validated). Beyond T4 (which fused gate+up into ONE `dmmv_q4k_fast_dual`
  launch over `2*n_ff` blocks, then a SEPARATE `geglu` launch read gate_buf/up_buf
  back): this collapses BOTH matvecs AND the GeGLU elementwise into ONE launch.
  New kernel `ffn_gate_up_geglu` over `n_ff` blocks — each block computes row `bx`
  of gate (a0) AND up (a1) via the shared `zinc_dmmv_q4k_fast_sum` helper (same
  blockDim 64 as the dual → bit-identical f32 g/u), keeps both in registers, and
  writes `gelu(g)*u` to `geglu_buf` (gelu formula copied verbatim from `geglu`).
  Removes the standalone geglu launch + the gate_buf/up_buf global round-trip
  (60/token on the 31b). Gated on both-Q4_K (else the unfused dual+geglu path,
  restored to its pre-T4 two-matvec form in the else branch). **Bit-exact:**
  `validate_catalog` → **5/5 token-correct** (qwen 3×12/12; gemma4-26b 12/12;
  gemma4-31b teacher-forced 11/12 = the documented near-tie, unchanged). **Build:**
  isolated caches, baseline md5 `b6190048…` → variant `07967f26…` (real recompile).
  **Perf — UNMEASURABLE in boost noise, leans NEGATIVE.** Interleaved A/B (4090,
  warmup ignored), THREE batches that **DISAGREE IN SIGN**: b1 (6×160) variant
  30.39 vs baseline 31.21 = **−2.63%** (won 4/6 but two big baseline rounds, swings
  26.29–33.79); b2 (8×160) 32.41 vs 32.55 = **−0.44%** (won 4/8, near-tie); b3
  (8×200) 32.36 vs 32.20 = **+0.50%** (won 6/8). **Pooled 22 rounds: variant 31.84
  vs baseline 32.06 = −0.69%.** Same coin-flip signature that rejected the Q/K-only
  norm+rope and MoE output-scale folds — OPPOSITE the all-positive signature of
  every validated win (T1/T4/T5/T6). **Root cause (a real design cost, not just
  noise):** unlike T4/T5/T6 which removed launches WITHOUT reducing parallelism,
  this fusion HALVES the block count (`n_ff` vs the dual's `2*n_ff`) and
  **serializes** the gate and up reductions within each block (`g` →
  `__syncthreads` → `u`). At batch-1 decode the Q4_K matvec is memory-latency bound
  with already-ample blocks to saturate the 4090, so the lost intra-block
  parallelism roughly cancels the saved geglu launch + round-trip → net ≈ 0,
  leaning negative. Per the contract ("measurable, repeatable gain"; "never claim a
  win from one boosted run") this is **NOT a validated win** → **reverted** (working
  tree + box variant tree restored to HEAD; iso caches + binaries cleared). Unlike
  the prior coin-flip reverts (which were strictly fewer-ops, true effect ≥0), this
  one TRADES parallelism for launches and may be a genuine small regression — NOT
  recoverable as a free future clock-locked win. **Lesson:** the "aggregate ≥2 tiny
  launches" rule (T5) only clears the floor when the fusion doesn't COST
  parallelism; fusing same-input matvecs into ONE block (vs T4/dual's two parallel
  blocks) is a different, worse trade at decode. The T4 dual-matvec (2 parallel
  blocks/pair, separate geglu) remains the right shape for the FFN gate/up pair.
- **2026-06-11 — Attention Q/K/V triple matvec fuse (extend T4 dual → triple,
  fold the standalone V matvec in). INCONCLUSIVE (sign-disagreeing, below floor)
  → REVERTED, not committed.** Found uncommitted in the working tree at cycle
  start (a prior loop cycle wrote it, never validated). After T4 the SWA attention
  path runs the Q/K pair as ONE `dmmv_q4k_fast_dual` launch (over `q_dim+kv_dim`
  blocks) then a SEPARATE standalone V matvec. This folds V into the same launch:
  new kernel `dmmv_q4k_fast_triple` over `M0+M1+M2` blocks (block `<M0`→Q→y0,
  `<M0+M1`→K→y1, else V→y2). V is Q4_K **or** Q6_K (gemma-31b Q4_K_M mixes them
  across SWA layers) — one `v_q6k` push flag selects the branch; the per-row
  compute reuses the shared `zinc_dmmv_q4k_fast_sum` / a newly-factored
  `zinc_dmmv_q6k_fast_sum` device helper (same blockDim 64 as the standalone fast
  path — confirmed `dmmvDispatch` dispatches `dmmv_fast` at blockDim 64 → identical
  summation order → **bit-identical**). Gated on Q&K both-Q4_K + V∈{Q4_K,Q6_K} +
  Wv present (SWA layers); full-attention layers (V = raw K projection, no Wv) and
  the unfused fallback keep the T4 dual. Removes the standalone V launch on the
  ~50 SWA layers of the 31b (~50/token). **PRESERVES parallelism** (Q/K/V still get
  their own blocks — `M0+M1+M2` blocks total, NOT the FFN-fuse's into-one-block
  serialization), so unlike the FFN fuse this is the "good shape" (true effect ≥0).
  **Bit-exact:** `validate_catalog` → **5/5 token-correct** (qwen 3×12/12;
  gemma4-26b 12/12; gemma4-31b teacher-forced 11/12 = the documented near-tie,
  unchanged). **Build:** isolated caches, baseline md5 `9520eb3b…` → variant
  `ab01de92…` (real recompile). **Perf — UNMEASURABLE in boost noise.** Interleaved
  A/B (4090, warmup ignored), THREE batches that **DISAGREE IN SIGN**: b1 (6×160)
  variant 32.49 vs baseline 32.62 = **−0.40%** (won 3/6, baseline hot rounds 33.47
  & 33.23); b2 (8×160) 32.57 vs 32.46 = **+0.33%** (won 5/8, baseline hot rounds
  33.58 & 32.71); b3 (8×200) 32.20 vs 32.03 = **+0.56%** (won 8/8 but baseline ran
  in a cool low-boost regime, margins 0.05–0.36). **Pooled 22 rounds: variant 32.41
  vs baseline 32.35 = +0.19%** (within the ±1% noise floor). Same coin-flip
  signature as the rejected Q/K-only norm+rope and MoE output-scale folds — the
  VARIANT is rock-steady across all batches (~32.4) while the BASELINE swings
  per-process and its occasional hot rounds flip the batch sign. **Root cause:** the
  triple removes only ONE launch per SWA layer (~50/token) — HALF the magnitude of
  the resolvable T4/T5/T6 wins (each ~2 launches/layer, ~120/token), so the saving
  sits below this box's per-process boost-noise floor. Per the contract
  ("measurable, repeatable gain"; "never claim a win from one boosted run") this is
  **NOT a validated win** → **reverted** (working tree + box variant tree restored
  to HEAD; iso caches + binaries cleared). **Unlike the FFN fuse**, this one
  preserves parallelism and is strictly fewer-ops (true effect ≥0, never a
  regression) — **recoverable from this log for a future clock-locked cycle**
  (`nvidia-smi -lgc`, needs sudo the agent lacks) that could resolve sub-1% effects.
  **Lesson re-confirmed (T5):** a single-launch removal that doesn't cost
  parallelism is correct-and-free but still below floor here; only ≥2-launch fusions
  (T4/T5/T6) clear the ±1% boost noise on this box without locked clocks. The T4
  dual remains the right shape for the attention matvecs; folding V to a triple is a
  free-but-unmeasurable extra that wants pinned clocks to bank.
- **2026-06-11 — EFFORT COMPLETE (dense fusion lever exhausted, no implementable
  increment this cycle). No code change.** Cycle-start state: working tree clean
  except the parallel 5090's `site/src/data/zinc-performance.json` (left untouched);
  NO uncommitted fusion pre-staged (unlike the prior 8 cycles). Did a **fresh full
  enumeration** of every per-token launch the dense gemma layer issues after T1–T6:
  attention = `[L0-only pre-attn rms_norm]` · Q/K dual-matvec (T4) · V-matvec (SWA) ·
  `rms_norm_rope_qkv` (T5) · `gemma_attention` · O-matvec · `rms_norm_residual_norm`
  (T6); FFN = gate/up dual-matvec (T4) · `geglu` · down-matvec ·
  `rms_norm_residual_scale_norm` (T6). **Every tiny norm/elementwise/scalar launch is
  already fused** — no standalone `scalar_mul`/`scale`/`add`/single-block-`rms_norm`
  remains on the dense path. The ONLY removable launch left is the V-matvec → Q/K/V
  triple, which removes exactly ONE launch/layer (~50/token, SWA-only) and was already
  implemented+measured+reverted as below-floor (see the triple-matvec cycle above,
  recoverable). The other matvecs (O, down) have UNIQUE inputs → no same-input pair to
  dual (T4's shape needs a shared input). `geglu` can't fuse without serializing
  (rejected FFN-fuse) or recomputing gelu n_embd× per element (down-fuse, ~5376×
  redundant — strictly worse). **Conclusion: no new ≥2-launch, parallelism-preserving,
  bit-exact dense fusion exists.** Remaining un-fused gate/up PAIRS live ONLY on the
  MoE path (gemma4-26b shared-expert `dmmvDispatch`×2 at forward_cuda_gemma.zig:667-668,
  and routed-expert `dmmv_q4k_experts`×2 at :723/:725) — each a T4-style dual candidate,
  but the 26b MoE is **boost-unmeasurable on this box** (proven across 3 cycles: MoE
  output-scale, and the boost-erratic-regardless-of-NGEN signature), so implementing one
  would only reproduce a foregone "INCONCLUSIVE → revert" and burn box time. **Re-confirmed
  the clock-control wall** that gates every below-floor recoverable fusion: on the 4090,
  `nvidia-smi -i 1 -lgc 2400,2400` → "current user does not have permission to change
  clocks"; `sudo -n` → "a password is required". No way to pin clocks → sub-1% dense
  effects and all MoE effects stay unresolvable. **Recommendation: CLOSE Effort 23.** Net
  delivered: **5 stacked validated dense-decode wins** (T1 +1.54%, T3 +1.66%, T4 +1.30%,
  T5 +1.65%, T6 +1.41% — each on its own pushed `perf/e23-*` branch, NOT main; compounding
  ~+7–8% on gemma-31b dense decode atop the two pre-effort fusions). The recoverable
  below-floor/MoE fusions (V-triple dense; MoE shared/routed gate/up duals) are documented
  above and worth banking ONLY in a future clock-locked session (`nvidia-smi -lgc`, needs
  sudo the agent lacks). Box hygiene: cleared `/tmp/lc-* /tmp/gc-* /tmp/cuda-dbg-*`; the
  reusable `~/zinc-e23` rsync tree sits at the T5-era commit with the already-banked T6
  uncommitted in it (harmless — the build helper rsyncs fresh from the Mac repo each cycle).
- **2026-06-11 — CLOSE re-confirmed (no code change). Fresh cycle, independent live
  re-verification of the structural blocker.** Cycle-start state: working tree clean except
  the parallel 5090's `site/src/data/zinc-performance.json` (untouched); NO uncommitted
  fusion pre-staged; no in-progress `perf/*` branch (the current branch
  `perf/e23-norm-boundary-fuse` already contains all 5 stacked wins T1→T6 + docs). Per the
  recoverable-fusion notes, the ONLY way to bank a new increment here is to A/B-resolve a
  below-floor/MoE fusion under **pinned clocks**, so I re-tested the clock wall LIVE on the
  4090 (dev 1): `nvidia-smi -i 1 -lgc 2400,2400` → "The current user does not have permission
  to change clocks for GPU 00000000:70:00.0"; `sudo -n true` → "sudo: a password is required";
  clocks idle 210 MHz / max 3105 MHz, persistence Enabled but unpinnable. The blocker is
  **structural** (no sudo on this WSL2 box per `~/AGENTS.md`), confirmed unchanged. No new
  ≥2-launch, parallelism-preserving, bit-exact DENSE fusion exists (enumeration settled the
  prior two cycles); the recoverable below-floor (V-triple dense) and MoE (shared/routed
  gate/up duals) fusions remain unvalidatable here. Implementing one would only reproduce a
  foregone INCONCLUSIVE→revert and burn box time — explicitly out of scope per the contract
  ("measurable, repeatable gain"). **No code change; nothing committed; no box scratch created
  (only nvidia-smi queries run). Effort 23 stays CLOSED** — net delivered remains the 5 stacked
  validated dense wins (T1 +1.54%, T3 +1.66%, T4 +1.30%, T5 +1.65%, T6 +1.41%), each on its own
  pushed `perf/e23-*` branch, none merged to main. Banking the recoverable fusions needs a
  future clock-locked session (sudo the agent lacks).
