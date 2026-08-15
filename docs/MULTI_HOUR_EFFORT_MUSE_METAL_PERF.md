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

## Remaining levers (next)

- **D-struct — speculative decode** is the ONLY structural decode lever left (amortizes the
  ~15 GiB/token weight read across accepted draft tokens). Blocked: no drafter model present
  and no spec-decode infrastructure in ZINC — a large, separate feature.
- **D-kernel — bandwidth:** close the 378→440 GB/s effective gap by reducing barriers that
  prevent independent matmul overlap (delicate; sub-noise per change; ~15% run-to-run
  variance makes iteration unreliable).
- **P2 — short-prompt batching threshold:** prompts below ~8 tokens still replay per-token
  (minor; little to gain).
- logit_scale before softcap (greedy-invariant; needed only for exact logits / sampling).
