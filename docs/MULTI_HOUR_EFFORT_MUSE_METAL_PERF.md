# Multi-Hour Effort — Muse Glimmer 30B Metal Perf

**Goal:** close ZINC's Muse Glimmer prefill+decode throughput gap vs a locally-built
llama.cpp with merged muse-glimmer support (PR #26841). Apple Silicon Metal, Q4_K_M.

**Reference oracle:** `~/llama-muse-ref/build/bin/llama-cli` (llama.cpp worktree, ref
`muse-ref` = pull/26841/head, `-DGGML_METAL=ON`). Reference on this machine:
**prefill 76.5 t/s, decode 29.3 t/s**.

**Benchmark:** `zinc -m <muse-q4k-m.gguf> --prompt <p> -n <n>` (greedy). Correctness gate
every cycle: coherent + factually-correct completions ("capital of France"→" Paris",
"chemical symbol for iron is"→" Fe"). Math-preserving cycles must stay byte-identical;
the batched prefill is validated byte-identical vs the per-token path
(`ZINC_BATCHED_PREFILL=off`) over 18+ tokens on discriminating prompts.

## Progress

| stage | prefill t/s | decode t/s | note |
|---|--:|--:|---|
| correctness only (baseline) | 12.8 | 11.9 | generic per-token path |
| + decode cmd grouping (`36109d3c`) | 16.6 | 16.7 | math-identical; 52 layers → ~1 cmd buffer |
| + Q6_K fast kernel (cycle 1, `1ed6407d`) | 22.2 | 20.5 | math-identical; down-proj was #1 slow kernel |
| + batched prefill (cycle 2-3, `239155c8`) | **125-153** | 20-22 | layer-major GEMM; weights read once/prompt |
| **reference (llama.cpp)** | **76.5** | **29.3** | target |

Prefill now **exceeds** the reference ~2× on batchable prompts (125 t/s @27 tok, 153 t/s
@54 tok; short <~8-token prompts stay on the per-token path, where there's little to gain).
Decode ≈ 75% of reference. Net vs the correctness-only baseline: **prefill ~12×, decode ~1.85×.**

## Diagnosis (per-kernel profile, `ZINC_METAL_KERNEL_TIMING=1` + byte buckets)

- **Decode is bandwidth-bound on the dense FFN** (66-78% of decode traffic): gate/up
  (M=19968,K=6656 Q4_K) + down (K=19968; ~half Q4_K, ~half Q6_K).
  - Q4_K gate/up already on the reference-style 2-simdgroup base kernel (~440 GB/s) — no
    easy win.
  - Q6_K down/KV was on the legacy SPIRV-Cross `dmmv_q6k` → cycle 1 routed it to the fast
    `dmmv_q6k_llama` port.
  - Remaining decode gap = per-layer non-matmul overhead (~20%/step) + Q6_K's ~1.4× bytes.
- **Prefill was 100% token-major-replay** (weights re-read once *per token*) → cycle 2-3
  routed it to the layer-major batched GEMM path.

## Cycle log

- **cycle 1** (`1ed6407d`): `supportsDenseQ6kSimdgroupDmmvArch` += muse. Muse's Q6_K
  ffn_down (K=19968) + KV projections were on the legacy SPIRV-Cross kernel; the simdgroup
  gate (dense, M%4==0, K%256==0) is satisfied, so they now use `dmmv_q6k_llama`.
  Decode +23% (16.7→20.5), prefill +34% (16.6→22.2). Byte-identical.
- **cycle 2-3** (`239155c8`): batched (layer-major) prefill. Added
  `canUseMuseGlimmerBatchedPrefill` (admits the per-layer attn_gate / QK-norm / post-norms /
  SWA+NoPE the generic gate bars), the weightless embed norm in the batched embed load, the
  per-layer NoPE/window conditioning, and the attention gate (project sigmoid gate into
  scratch.gate before flash, apply after — mirrors the Qwen route-packed path). *Gotcha:*
  first cut projected the gate into scratch.q AFTER flash → WAR hazard vs flash's read of
  scratch.q → visibly wrong tokens; fixed by projecting into scratch.gate before flash.
  Also had to drop the lm_head batched-GEMM-quant gate (muse lm_head is Q5_K but the batched
  path drives the head through dispatchLmHeadWithInputOffset). Prefill 27→125 tok/s (4.6×),
  byte-identical vs per-token over 18 tokens.

## Decode is bandwidth-bound — dispatch fusion does NOT help (tested)

Clean profile at 22.5 tok/s: 1 command buffer/token (grouping optimal), ~732 dispatches +
523 barriers/step, ~15 GiB/token weights at ~378 GB/s effective vs the reference's ~470.
The FFN matmul *kernels* already run at the reference-style ~440 GB/s (per the
`dmmv_q4k` base-kernel note). The ~6 ms/step gap is per-layer non-matmul latency, but it is
**hidden by the concurrent encoder**, not exposed as extra dispatches.

**Tested & reverted (DEAD END — do not re-litigate):** admitting muse to the fused QKV
kernels (`canUseDenseQ4KQKQ6KV` / `canUseDenseQ4KQKV`) so all 52 layers fuse Q+K+V into one
dispatch (+ separate gate) — path evidence confirmed `separate 0`, output byte-identical —
gave **zero** wall-clock change (22.5→22.5). The concurrent encoder already overlaps the
small QKV/gate/norm dispatches behind the bandwidth-bound FFN matmuls, so reducing dispatch
count is not the decode lever. Same logic applies to fused gate+up and the Q5_K lm-head
(`dmmv_q5k_native` is loaded but intentionally unused).

## cycle 4 — speculative decode (`713029fc`) ✅

DFlash (the intended drafter) is infeasible here: it's an EAGLE-style method whose drafter
is a separate DeepSeek-V4 "DSpark" research model (MoE + hyper-connections + Sinkhorn + LoRA)
consuming the target's per-layer hidden states — and no such drafter GGUF exists on the box.
So shipped **model-free prompt-lookup (n-gram) speculative decode** instead, default-on for
Muse (ZINC_SPEC_DECODE=0 to disable):

- Draft = copy what followed the most recent earlier occurrence of the last 3 tokens.
- Verify K+1 tokens in ONE batched forward (`InferenceEngine.verifyTokens` → a per-position
  Q5_K `gemm_q5k` lm-head + CPU argmax tail on the batched-prefill body, gated on the new
  `verify_argmax_out` field — no change to prefillBatched's signature so the regression tests
  hold, only the KvStateNotAvailable window widened 2000→2400).
- Accept the longest confirmed prefix; overwrite rejected-draft KV on the next step.
- **Greedy speculative decoding is exact** — output byte-identical to per-token (validated on
  diverse prompts + long gens).

Key finding: a verify is a full batched forward whose cost is ~fixed regardless of small N
(the batched GEMM's per-pass overhead dominates), so it only pays above ~3 accepted tokens.
Guarded by an **acceptance-EMA gate + adaptive draft length**: no draft / low recent
acceptance → the cheap single-token `decodeStep`, so low-repetition text isn't penalized
(bounded to ~1 startup verify). Measured: copy-heavy (structured / long-context) **1.52×**
(22.6→34.6 tok/s, 6.8 accepted/verify), open-ended ~1.07×, adversarial short code ~-5.6%.

## cycle 5 — spec-decode profiling (draft ceiling 8→24, `9a6e99a8`)

Measured the verify cost directly (ms/verify log): **a verify is ~193 ms FIXED, independent
of N** — 192.6 ms @ N=2, 192.9 ms @ N=3, 195.9 ms @ N=8. The batched GEMM reads the 15 GB
model at only **~78 GB/s** vs decode's matvec **~349 GB/s** — the fixed cost is that
bandwidth inefficiency at small N (the GEMM's large-N tiling wastes bandwidth for a handful
of columns). Consequences:
- Break-even ≈ 193/44 ≈ **4.4 accepted tokens/verify**. Copy-heavy (7/verify) → 1.56×
  (35.5 vs 22.7 tok/s); short-repeat/open-ended fall below it → the EMA gate keeps them on
  the per-token path (neutral).
- Because the cost is fixed, **longer accepted drafts amortize it** — hence the 8→24 ceiling
  (24 accepted → ~8 ms/token). Adaptive-capped, so short-repeat is unaffected.

## cycle 6 — verify kernel dead end + runtime break-even gate (`11ff18f1`)

Attempted the "bandwidth-efficient small-N matvec" lever and it's a **DEAD END** (measured):
- Wrote a dense Q4_K multi-column matvec (`dmmv_q4k_cols.metal`, adapted from moe_cols) to
  replace gemm_q4k at small N. **Slower: 233 vs 196 ms.** Scalar float dots lose to gemm's
  hardware simdgroup matrix units. Reverted.
- Small-NR1 gemm variant (compute only the real columns) is **structurally locked**: the
  shader's A-load needs ≥128 threads and the tiling ties NR1 = 16·(THREADS/64), so THREADS≥128
  forces NR1≥32. Not a define change.
- Root measurement: verify is ~flat ~193–200 ms at N=2 **and** N=8 — the gemm always computes
  a **fixed 32-column tile** regardless of N. So the gemm is near-optimal; the real lever is
  **filling that 32-col tile** with longer accepted drafts, not a new kernel.

Two shipped changes from that:
1. **Full-tile drafting** — drop the adaptive draft-cap (it was throttling on a false
   "cost scales with N" premise); always draft the tile width (ceiling 24→31). Long verbatim
   copies now amortize the fixed verify cost immediately.
2. **Runtime break-even gate** — spec's break-even is `verify_ms / per-token-decode_ms`, and
   decode is **load-dependent** (24 ms/tok quiet ↔ 44 ms/tok loaded; command encoding is
   CPU-side) while verify is GPU-bound (~flat). A static threshold loses on a quiet machine.
   Now the gate MEASURES both live (first 2 tokens calibrate decode; verifies measure verify)
   and only fires when acceptance clears the measured ratio +15%. **Never nets a loss on any
   load; the win scales up automatically when decode is slow or acceptance is high.** Quiet
   machine + ~8/verify ≈ neutral-to-1.05×; loaded or long-copy → up to ~1.8×.

## cycle 7 — wide fused-norm tail for decode (`54d78ab7`) ✅ +5.4% decode

Overnight re-diagnosis at LOW load (2.7–4.7, not the earlier ~12) with isolated microbenches
settled the "where is the gap" question:
- **Matmuls are at hardware peak.** dmmv microbench at Muse's exact shapes: attn/ffn Q4_K
  matvecs 497–527 GB/s (91–96% of M4 Max 546), Q6_K down 465 (85%). gemm_q4k prefill 12.2
  TFLOP/s at every Muse shape. Recomputing llama's pp256 with the *matmul'd* param count
  (27.85B − 1.34B embed − 1.34B lm-head) gives **12.2 TFLOP/s = exactly ZINC's GEMM ceiling** —
  llama is at the ceiling, ZINC prefill is ~10.2 (≈17% exposed overhead).
- **The decode graph is already tight:** both epilogues are fused (attn: post_attn_norm +
  residual + pre_ffn_norm via `dispatchPostNormResidualRmsNormOnCmd`; ffn tail: post_ffn_norm +
  residual + next-input-norm), QK-norm is fused into RoPE (`rope_qk_norm_inplace`). QKV-fusion
  is a confirmed dead end (see above). So the diffuse gap is NOT missing fusion.

**The win** (`54d78ab7`): the two triple-fused epilogue norms run 104 dispatches/token, and for
Muse (hidden_dim=6656) they used the **256-thread** narrow `post_norm_residual_rms_norm` — a
single threadgroup whose two sequential reductions underutilize the core's latency-hiding. 6656
is vec4-aligned and fits the existing **1024-thread** `post_norm_residual_rms_norm_wide`
kernel's MAX_VEC_PER_THREAD=2 register cache exactly, so route Muse's n=6656 tail through it
(was gated to Gemma n=5376 only — a one-condition gate change at `dispatchPostNormResidualRmsNormOnCmd`).
- **decode 24.09 → 25.40 tok/s (+5.4%)** short ctx, **23.68 → 24.6 (+3.9%)** at 256 ctx.
- Greedy byte-identical; batched==per-token byte-identical; factual gate (Paris/Fe/oxygen) intact.
- 9× my pre-estimate — the single-threadgroup reductions were a real bottleneck, not sub-noise.
- Decode gap vs llama: ~10% → **~7.6% at 256 ctx, ~4.5% short**. llama tg48=26.63, pp256=242.

Prefill barely moved (+1%): in the batched path the norms are one-threadgroup-per-token
(N threadgroups already saturate 40 cores), so widening the per-threadgroup count is moot there.
Prefill still does 5 SEPARATE `rms_norm_mul`/layer (262 calls @257-tok) instead of the fused
epilogue decode uses — a possible (marginal ~1-2%, risky) prefill lever, deferred.

## Remaining levers (next)
- **Decode context drop:** 25.4 short → 24.6 @256 ctx (llama flat). ~1.3 ms/tok over 256 ctx is
  the scalar decode-flash per-key overhead (NOT KV bytes — KV read is ~0.07 ms/tok). Not
  byte-identical to touch (softmax order). Fixing it → ~+3% at benchmark ctx.
- **Q6_K down matvec:** 465 GB/s (85%, lowest of the matmuls); biggest single decode contributor
  (down = 102 GiB/decode). Config tuning (rows_per_wg) could recover ~1.7% if headroom exists.
- **P2 — short-prompt batching threshold:** prompts below ~8 tokens still replay per-token.
- logit_scale before softcap (greedy-invariant; needed only for exact logits / sampling).

## cycle 8 — prefill attention-epilogue fusion (`170dbad8`) ✅ +~2% prefill; FFN fusion = DEAD END

`prefillBatched` (the batched layer-major path) applied post_attention_norm as a SEPARATE
`rms_norm_mul` then a fused residual+ffn_norm — 2 dispatches for the attn epilogue, where
decode does 1 (triple-fused `post_norm_residual_rms_norm`). Routed the batched prefill's attn
epilogue through the same triple-fused kernel (batched grid {n_tokens,1,1}, narrow 256-thread
pipe), gated on post_attn_norm_present. **Byte-identical (batched==per-token), dispatches
1096→1044 / barriers 783→731 @289tok, prefill ~198→~202 (+~2%).** Converges the batched path
onto the per-token path's exact math (a genuine simplification).

**DEAD ENDS this cycle (tested, reverted — do not re-litigate):**
- **Prefill FFN-tail fusion** (post_ffn_norm+residual+next-norm, incl. the full option-A that
  skips the top-of-loop input norm + final norm via cross-layer scratch.norm reuse): byte-
  identical, removed 104 MORE barriers (dispatch 1044→940), but **prefill stayed at 204 —
  ZERO speed gain.** Confirms prefill's norm barriers are already hidden behind the GEMMs; the
  17% prefill gap is **in-context GEMM efficiency** (llama runs its GEMM near the isolated
  12.2 TFLOP/s ceiling with ~0 exposed overhead; ZINC ~10.3), NOT the norms/barriers. Removing
  barriers beyond the first (attn) fusion buys nothing. Reverted (complexity w/o benefit).
- **lm-head Q5_K native rows_per_wg 8→16** (512-thread WG): 25.41→25.44, noise. The lm-head is
  weight-bandwidth-bound (~360 GB/s); rows_per_wg only amortizes the tiny shared input cache.
  A real lm-head win needs a better Q5_K *weight-read* pattern (the 176-byte block qh/qs are
  strided), not a config change. Reverted.
- **QK-norm widening:** N/A — `rope_qk_norm_inplace` is already well-parallelized (34 head-
  threadgroups × 64 thr, head_dim=128 reduction). The wide-norm gift was UNIQUE to the 6656-wide
  single-threadgroup epilogue reduction.

## Standing after cycles 7-8 (M4 Max, Q4_K_M, low load)
- **decode 25.4 short / 24.6 @257ctx** vs llama tg48 **26.6** → gap ~4.5% short, ~7.5% @ctx (was ~10%).
- **prefill ~202** vs llama pp256 **242** → ~17% (GEMM-capped; unmoved by fusion).
- Net cycle 7-8: decode +5.4% short / +3.9% @ctx, prefill +2%. Both byte-identical, on main.

## cycle 9 — MEASUREMENT-FAIRNESS CORRECTION (the "17%/10%" gaps were partly artifacts)

Two comparison errors were inflating the perceived gaps:
1. **Prefill N-tile quantization.** The gemm_q4k column tile is 32 wide, so N-not-a-multiple-of-32
   wastes the last tile. **llama is EQUALLY sensitive:** llama pp256=242 but **pp257=216 (−11%)**,
   pp288=241 but pp289=219. My ZINC prefill test used a 257-289 token prompt vs llama-bench's clean
   256 → unfair. At the SAME N: **ZINC pp256=226 vs llama 242 → 6.7%** (steady-state, not 17%).
   FLOP math: llama runs prefill at 12.19 TFLOP/s = 99% of the pipelined-gemm_q4k peak (12.29);
   ZINC at 11.37 = ~7.5% exposed overhead. Barrier microbench: a barrier between two gemm_q4k
   dispatches costs ~2-5% (lost ramp/drain overlap), resource-scoped ≈ scope-scoped for the
   DEPENDENT chain. profileBarrier() uses cmd.barrier() (full MTLBarrierScopeBuffers).
2. **Decode context depth.** llama-bench tg48 generates from ~empty context, so the fair compare
   is ZINC decode at SHORT ctx (25.4), not @257ctx. **llama decode is FLAT** (tg48 @d0/256/512 =
   26.60/26.57/26.55); **ZINC DROPS** (25.4 / 24.6 / ~22). So at the benchmark point decode gap is
   **4.5%**, but it GROWS with context to ~17% @512 — ZINC's biggest real weakness.

**Corrected benchmark standing: prefill 6.7% (pp256), decode 4.5% (tg48).** Both ~5-7%, not 10-17%.

**The real high-value lever = f16 KV cache + f16 flash** (what llama uses). ZINC only has f32 + Q8
flash. The context drop (25.4→22 @512, llama flat) is the scalar-f32 decode flash reading the
growing f32 KV. f16 halves the KV bytes AND enables f16 flash compute → should flatten the curve
like llama, closing the 7-17% @-context gap and helping real long-context use most. NON-byte-
identical (precision change, llama-standard) — being built in cycle 10.

## Highest-EV remaining levers (all NON-trivial — need review, not blind commits)
- **f16 KV cache + flash_attn_f16 (biggest decode lever):** ZINC has only f32 + Q8 KV/flash; llama
  uses f16. The decode **context drop** (25.4 short→24.6 @257ctx; llama is FLAT) is the scalar
  decode-flash per-key f32 compute (NOT KV bytes — those are ~0.07 ms/tok). f16 halves flash
  compute+bytes, matches llama, ~+3% @benchmark ctx and more at long ctx. NON-byte-identical
  (precision change, llama-standard); needs new flash_attn_f16 + f16 kv_cache_write + dtype plumb.
- **Gate-fusion (Q4_K gated o-proj):** fold the attn_gate sigmoid_mul into the o-proj dmmv X-load
  (byte-identical, ~1 barrier/layer). BUT prefill's FFN barrier-removal was neutral, so this is
  likely neutral in decode too (est. ~0.6% EV); needs a hot-loop change to the tuned dmmv_q4k.
- **Q6_K down matvec** (465 GB/s, 85%, lowest matmul): a K=19968-specialized dmmv_q6k_llama
  variant (like the existing k4096/k5120/k17408) might recover ~0.9%. New kernel.
- **Prefill in-context GEMM efficiency:** the real 17% prefill gap — llama keeps the GPU fed
  across the whole graph; ZINC drains between dependent GEMMs. No lever found; likely needs a
  whole-graph scheduling refactor. NOT a single-kernel fix.

NOTE: measurement is currently confounded by CPU contention (parallel agents, load ~12) — the
per-dispatch command encoding is CPU-side, so decode throughput swings 22↔35 tok/s. Back-to-back
spec-vs-nonspec comparisons stay valid; absolute micro-numbers do not. (Cycle 7-8 measured at
low load 2.7–4.7, where the harness `<scratchpad>/muse_ab.sh` gives ±0.01 tok/s repeatability.)

NOTE: measurement is currently confounded by CPU contention (parallel agents, load ~12) — the
per-dispatch command encoding is CPU-side, so decode throughput swings 22↔35 tok/s. Back-to-back
spec-vs-nonspec comparisons stay valid; absolute micro-numbers do not.
