---
title: "The gate that keeps Qwen 35B prefill at half of llama.cpp on RDNA4"
date: "2026-04-26"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - qwen3-next
  - qwen35
  - prefill
  - moe
  - ssm
  - mamba
  - llm-inference
  - gpu-kernels
keywords:
  - Qwen 3.5 35B A3B prefill RDNA4
  - Qwen 3.6 35B prefill RDNA4
  - Qwen3-Next 80B A3B Vulkan
  - hybrid MoE SSM prefill
  - gated delta net prefill
  - Mamba2 token-recurrent state
  - canUseBatchedPrefillRdna gate
  - MUL_MAT_ID Vulkan AMD
  - flash_attn_batched RDNA4
  - RDNA4 prefill bandwidth ceiling
  - ZINC Qwen 35B prefill plan
  - llama.cpp Qwen 35B Vulkan
faqs:
  - question: "What is the current ZINC prefill gap to llama.cpp on Qwen 3.5/3.6 35B-A3B?"
    answer: "On a 154-token prompt running on the AMD Radeon AI PRO R9700, ZINC reports 90.24 tok/s prefill while llama.cpp reports about 180 tok/s on the same hardware and weights. ZINC has moved from 78.11 tok/s to 90.24 tok/s over 50 autonomous-loop cycles, but the remaining 2x gap is structural rather than incremental."
  - question: "Why is the gap so much larger on Qwen 35B than on Qwen 3 8B?"
    answer: "Qwen 3 8B is a dense transformer and runs through ZINC's batched prefill path at 164 percent of llama.cpp's prefill on the same card. Qwen 3.5 and 3.6 35B-A3B are hybrid Mixture-of-Experts plus state-space-model architectures, and ZINC's batched prefill path is gated off for any model where n_experts is greater than zero or ssm_d_inner is greater than zero. The flagship runs through a per-token decode loop instead, dispatching about 45,000 GPU workgroup launches per prompt where llama.cpp dispatches under 300."
  - question: "Why did porting llama.cpp's tiled GEMM not close the gap?"
    answer: "We ported all four foundation pieces between cycles 14 and 21: count_experts, mul_mm_q4k, mul_mm_id_q4k, and mul_mmq_q4k. Three of the four stayed dormant with no callers in the actual hot path because the wire-up requires a buffer-layout refactor that did not fit a single autonomous cycle. After 17 cycles of dormant infrastructure, cycle 40 reverted 1,470 lines of code as net technical debt. mul_mm_q4k is still in the tree, wired only into the language-model head where the N=1 case wastes the BN tile, which is exactly the worst case for a tiled GEMM."
  - question: "What is the largest single bucket in the current prefill profile?"
    answer: "The state-space-model bucket at 925 ms out of about 2,110 ms of total GPU phase time, or 44 percent. Inside SSM, the gated delta-net kernel is 449 ms by itself, and the four SSM projection DMMVs add another 318 ms. The MoE bucket is 739 ms in second place. The attention bucket is 333 ms. The MoE share of prefill has been falling as SSM and rms-norm shaders got faster, but SSM has been the largest GPU phase since the cycle-50 profile snapshot."
  - question: "What is the single change that would close the most ground?"
    answer: "Loading the SSM token-recurrent state once per workgroup and walking all 154 prompt tokens inside the kernel, the way llama.cpp's gated_delta_net.cu kernel does on CUDA. Today every token re-reads and re-writes 2 MB of state per SSM layer, which is roughly 18 GB of read traffic and 18 GB of write traffic per prefill. Block-resident state collapses that to 4 MB total per layer and keeps the state vector in registers across the token loop. The expected bucket reduction is ssm_delta from 449 ms to about 150 ms, which is roughly 15 to 20 percent of end-to-end prefill."
excerpt: "ZINC's per-token prefill on Qwen 3.5/3.6 35B-A3B runs at 90.24 tok/s on a Radeon AI PRO R9700. llama.cpp on the same card hits 180 tok/s on the same prompt and weights. The remaining 2x gap is not a kernel-by-kernel gap. It is a single early return in canUseBatchedPrefillRdna that locks every Mixture-of-Experts plus state-space hybrid model onto a per-token decode loop, dispatching 45,000 workgroups per prompt where llama.cpp dispatches 288. Here is what 50 autonomous-loop cycles found, what is still left, and which two changes close most of the remaining ground."
---

A 154-token prompt on Qwen 3.6 35B-A3B at Q4_K_XL runs through ZINC's RDNA4 path at **90.24 tok/s**. The same prompt and weights through [llama.cpp's Vulkan backend](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp) on the same Radeon AI PRO R9700 lands at roughly **180 tok/s**. We started this effort at 78.11 tok/s and have run 50 autonomous-loop optimization cycles to reach 90.24. The remaining 2x is the part that is not going to come from one more shader rewrite.

We want to be specific about what that 2x is, because the shape of the answer is unusual. It is not a per-shader gap. It is not a missing kernel. It is one early return in one Zig function that holds the entire batched prefill path closed for any model where `n_experts > 0` or `ssm_d_inner > 0`. Qwen 3.5 and 3.6 35B-A3B fail both checks. The flagship runs through a per-token decode loop instead.

This post is the field report from those 50 cycles. It covers what moved the number, what looked promising and was reverted as dead infrastructure, where the time actually goes, and the two architectural changes that close most of the remaining ground.

![Per-phase GPU time budget for a 154-token Qwen 3.6 35B-A3B prefill on the Radeon AI PRO R9700 at cycle 50, with llama.cpp's 855 ms wall total shown for comparison.](/blog/qwen35-prefill-phase-budget-cycle50.svg)

The chart is the most useful frame for the rest of this post. ZINC's 1,707 ms wall on this prompt is roughly twice llama.cpp's 855 ms on the same hardware. The four GPU phases inside ZINC's bar are each candidates for an attack, but their absolute sizes lie about how much can come out of each one. The dispatch arithmetic at the bottom of the chart is the part that constrains where the easy wins live.

## What 50 cycles actually moved

Five cycles produced a real best-checkpoint improvement. Every one of them is a per-shader micro-restructure. None of them is a port. Roughly in order:

| Cycle | Result | What changed |
|---|---:|---|
| 22 | 80.61 | Wired count_experts as a post-prefill sweep |
| 28 | 83.07 | rms_norm_dmmv_f32 NUM_ROWS=2 to 1, occupancy-led |
| 42 | 86.94 | rms_norm_dmmv_f32 vec4 reads and writes |
| 46 | 87.82 | rms_norm_mul vec4 plus per-thread register cache |
| 50 | 90.24 | ssm_delta_net 8 threads by 8 rows to 16 threads by 4 rows |

The pattern is the same on every winner. Widen the inner reduction axis, halve the per-thread register slab, double the workgroup count, use cache lines more fully. Cycle 50's shader comment captures it: 16 threads per row read 64 bytes per iteration where the prior 8 threads per row read 32, and the per-thread `reg_state[8]` halves the VGPR pressure that was forcing one wave per SIMD on RDNA4.

These are real wins. They compounded from 78 to 90 tok/s. They are not the 2x.

## What looked structural and reverted as dormant infrastructure

Between cycles 14 and 21 we ported the four pieces of llama.cpp's tiled GEMM with MUL_MAT_ID into the tree. That sequence is the one we keep coming back to in the planning docs:

1. `count_experts.comp` for the per-expert token-count buffer used by tile early-exit.
2. `mul_mm_q4k.comp` as the warp-tiled f32 GEMM with no MoE branch yet.
3. `mul_mm_id_q4k.comp` as the MUL_MAT_ID variant with subgroup-ballot row gather.
4. `mul_mmq_q4k.comp` as the Q8_1-activation integer-dot variant.

All four landed correctly. `mul_mm_q4k` was wired into the language-model head and produced bit-identical output. The other three had zero callers anywhere in the prefill path. In cycle 16 we measured `mul_mm_q4k` at 78.14 tok/s on the language-model dispatch where the prior baseline was 78.55, which is noise. The reason is that the LM head fires once per prefill at N=1, and N=1 is the case where the BN=64 tile of a tiled GEMM is 1.5 percent saturated. The shader is correct and the wiring is correct. It just attacks the wrong dispatch.

Cycle 40 audited the four foundations, found the three dormant pipelines, and reverted 1,470 lines of code. The cycle-40 self-analysis is the cleanest summary of what happened: "porting foundations without wiring them into the actual hot path banks no tok/s, and the wire-up is a high-risk one-cycle refactor needed for buffer layout that the loop has refused to attempt in a single cycle."

The lesson is not that the GEMM port is wrong. The lesson is that wiring is the work, and wiring requires a buffer-layout refactor that does not fit inside one cycle.

## Where the time actually goes

The cycle-50 phase budget on a 154-token Qwen 3.6 35B-A3B prefill totals 2,110 ms of GPU phase time, overlapping into 1,707 ms of wall time through the 2-deep command-buffer pipeline.

| Bucket | ms | Share |
|---|---:|---:|
| SSM total | 925.4 | 44% |
| `ssm_delta` (gated delta-net) | 449.2 | 21% |
| `ssm_proj` (4 DMMVs per layer) | 318.0 | 15% |
| `ssm_out` | 113.3 | |
| `ssm_conv` | 49.9 | |
| `ssm_gnorm` | 47.9 | |
| MoE total | 738.9 | 35% |
| `down` | 256.1 | |
| `gate_up` | 178.4 | |
| `topk` | 114.9 | |
| `router` | 99.9 | |
| Attention | 333.2 | 16% |
| Shared expert | 109.6 | 5% |

The MoE bucket has been shrinking as the rms-norm shaders got faster. The state-space-model bucket has not, because nothing at the structural level has touched it. The cycle 50 win on `ssm_delta` was inside a single shader, not a change to how the SSM block is dispatched. The SSM block on this hybrid is 30 of the 48 layers, and every one of those layers reads and writes 2 MB of token-recurrent state per token. For 154 tokens across 30 layers, that is roughly 18 GB of state-buffer reads and 18 GB of state-buffer writes per prefill. The R9700's 576 GB/s bandwidth ceiling pays for that traffic in real wall time.

## The single early return

The structural reason every per-token bucket is large is one Zig function. `canUseBatchedPrefillRdna` in `src/compute/forward.zig` decides whether the entire batched prefill body in `prefillBatched` can run, or whether the engine falls back to `prefillBatch`, which is a per-token loop calling `decodeStep` 154 times. The relevant lines:

```zig
fn canUseBatchedPrefillRdna(cfg: Config) bool {
    if (cfg.n_experts > 0) return false;
    if (cfg.ssm_d_inner > 0) return false;
    // ...further architecture checks...
    return true;
}
```

Qwen 3.5 and 3.6 35B-A3B fail both checks, because the architecture is hybrid Mixture-of-Experts plus state-space-model. The fall-back path then dispatches every kernel once per token. For 154 tokens, 48 layers, and roughly 6 dispatches per layer through the MoE and SSM bodies, ZINC issues on the order of **45,000 GPU workgroup launches per prefill**. llama.cpp on the same prompt issues fewer than **300**, because every layer's matmul-id, count_experts, and quantize_q8_1 calls operate on the entire batch in one dispatch each.

The two reasons llama.cpp can fold the per-token loop into one dispatch are independent. They have to be ported independently before the gate can drop.

## The first reason: MUL_MAT_ID with a subgroup-ballot gather

llama.cpp's `mul_mm_id_funcs.glsl` builds a per-workgroup `row_ids[BN]` list at the top of the kernel. Each thread computes `id = data_ids[token, slot]`. A single `subgroupBallot` produces a 128-bit mask of which lanes match this workgroup's expert. `subgroupBallotExclusiveBitCount` plus a tiny serial loop across subgroups gives every matching lane its compacted output position. The matmul body that follows reads `data_b[token_idx]` from the gathered list, and the gather itself is folded into the load address. There is no scratch B-tile.

The early-exit at the top of `mul_mm.comp` line 145 reads `data_expert_count[expert_idx]` and bails if `ic * BN >= expert_count`. With 128 experts and 154 tokens at top-8 routing, the average expert sees about 9.6 routed tokens, so 90 percent of workgroups exit before the first load. The total dispatch shape for one MoE FFN matmul-id is `gridX * gridY * gridZ = 11 * 2 * 128 = 2,816 workgroups`. ZINC's per-token MoE dispatches roughly 4,200 workgroups per token, totalling 30,000 across 154 tokens for the same projection. The total work is similar; the dispatch overhead is two orders of magnitude apart.

Wiring this is the deferred cycle-40 refactor. The shader exists. The routing capture buffer exists. The count_experts pipeline exists. The missing piece is the gather plus the scatter that takes per-(token, expert_slot) outputs and accumulates them weighted by the topk weights into per-token output rows.

## The second reason: block-resident SSM state

The Qwen3-Next gated delta-net recurrence is not the Mamba-2 selective-scan recurrence. The right reference is `ggml/src/ggml-cuda/gated_delta_net.cu`, not `ssm-scan.cu`. The CUDA kernel does one block per `(head, sequence)`, loads the full state vector into per-warp registers once, walks all tokens in a `for (t = 0; t < n_tokens; t++)` loop inside the block, and writes the final state once. ZINC's `ssm_delta_net.comp` does one workgroup per `(head, token)` and re-reads the entire state from DRAM on every dispatch.

Block-resident state is roughly 64 KB per head per layer for Qwen 3.5/3.6's `dt_rank * head_v_dim * head_v_dim * f32` shape. That fits in LDS or in the register file with room to spare. The 18 GB of state read traffic per prefill collapses to 4 MB. The dispatch count for `ssm_delta` drops from 154 per layer to 1 per layer.

Parallel-scan over the token axis using a Blelloch tree is the wrong shape for this recurrence. The state update is rank-1 with a per-token gate `g`, not a simple additive prefix sum, and materializing the prefix-state would cost roughly 400 MB per layer per prefill. Neither llama.cpp's CUDA backend nor any production engine does this for gated delta-net. Block-resident state with an in-block token loop is the standard answer.

## The honest part

The two changes above are multi-week ports, not multi-day cycles. The autonomous loop has run 50 cycles on this effort and produced 12 tok/s of progress, which is real and which we are not dismissing. But the loop is now shaped by what fits in a single cycle, and what fits in a single cycle is per-shader micro-restructuring. The remaining ground requires a buffer-layout refactor for the MoE wire-up and a state-buffer-lifetime change for the SSM block. Each of those is a deliberate engineering project with a correctness gate, a CPU reference path, and a validation regime.

There is also a tier of medium-sized wins that we expect to land along the way. Pre-loading Q into shared memory inside `flash_attn.comp` removes the per-iteration re-reads from global memory that today cost roughly 327,000 reads per workgroup per attention layer. Fusing `ssm_gnorm` into the `ssm_out` DMMV removes 4,620 dispatches and barriers per prefill from the critical path. Fusing the K-projection chain into a single `RMS_NORM + MUL + ROPE + KV_CACHE_WRITE` shader removes another 20,000 small dispatches and matches a pattern llama.cpp adopted six months ago. Each of those is a 3 to 5 percent prefill win, and they compound.

We are at 50 percent of llama.cpp on the flagship today. We expect block-resident SSM state plus MUL_MAT_ID wiring to take us into the 75 to 90 percent range. The fusions and the cycle-50 micro-restructure pattern, applied to the remaining MoE inner loops, take us the rest of the way. None of this is speculative. The reference implementations exist, the shaders mostly exist, and the wiring is the work.

The thing that surprised us, after 50 cycles, was how many of the answers were sitting upstream as one-line gates rather than missing kernels. The gate that holds the batched prefill closed for hybrid MoE-plus-SSM models is the most expensive line in our codebase right now. It is also the line that will close last, because it depends on every other piece landing first.
