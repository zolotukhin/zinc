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

## Remaining levers (next)
- **D-kernel — decode bandwidth:** close the 378→440 GB/s effective gap by reducing barriers
  that prevent independent matmul overlap (delicate; sub-noise per change).
- **P2 — short-prompt batching threshold:** prompts below ~8 tokens still replay per-token.
- logit_scale before softcap (greedy-invariant; needed only for exact logits / sampling).

NOTE: measurement is currently confounded by CPU contention (parallel agents, load ~12) — the
per-dispatch command encoding is CPU-side, so decode throughput swings 22↔35 tok/s. Back-to-back
spec-vs-nonspec comparisons stay valid; absolute micro-numbers do not.
