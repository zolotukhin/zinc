---
title: "Softmax steals RDNA4's WMMA issue slots, and the exp is not the thief"
seoTitle: "RDNA4 Flash Attention: Softmax vs WMMA Issue Slots"
date: "2026-07-11"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - r9700
  - vulkan
  - prefill
  - flash-attention
  - softmax
  - wmma
  - occupancy
  - transcendental
  - local-llm
  - llm-inference
  - gpu-kernels
keywords:
  - RDNA4 flash attention softmax
  - WMMA issue slots vector pipe
  - non-matmul FLOPs attention
  - V_EXP_F32 quarter rate transcendental
  - RX 9070 XT prefill tokens per second
  - FlashAttention-3 asynchrony Hopper
  - exp2 log2e softmax scale fold
  - Qwen3.5 9B prefill RDNA4
  - online softmax rescale cost
  - AMD consumer GPU local LLM
excerpt: "With the MLP fused, attention is now 41 percent of ZINC's Qwen3.5-9B prefill on the RX 9070 XT, and the profile points at softmax. The obvious suspect is the exponential, which runs at quarter rate on RDNA4's transcendental unit. The obvious suspect is wrong. The exp sits on a separate unit and can hide behind another wave's matmul; it is the boring full-rate multiplies and adds of the online softmax that steal cycles from WMMA, because on RDNA4 the matrix instructions and the vector FMAs issue down the same pipe."
seoDescription: "Why the softmax in ZINC's RDNA4 flash attention kernel costs matmul throughput: WMMA and FP32 vector FMAs share one issue pipe, so every rescale and running-sum op is a cycle the matrix units are idle, while the quarter-rate V_EXP_F32 hides on a separate transcendental unit. What FlashAttention-2's non-matmul FLOP trimming recovers on RDNA4, and why FlashAttention-3's overlap trick needs an async matrix pipe RDNA4 does not have."
faqs:
  - question: "Is the exponential the bottleneck in RDNA4 flash attention?"
    answer: "No, and that is the surprise. V_EXP_F32 runs on RDNA4's transcendental unit at quarter rate, about 16 exponentials per clock per compute unit, which sounds alarming next to 1024 matrix FLOPs per clock. But the transcendental unit is separate hardware from the vector FMA pipe, so an exp issued by one wave can execute while another wave's WMMA is running. In a first-order model at head dim 128, the exp accounts for about 0.063 cycles per score element against 0.500 cycles of matmul, and most of that is hidden when occupancy is high. The full-rate softmax work, the running max, the subtract, the running sum, and the accumulator rescale, is the part that hurts, because those are ordinary vector FMAs and RDNA4 issues WMMA down the same vector pipe."
  - question: "Why can't RDNA4 overlap softmax with the matmul the way FlashAttention-3 does?"
    answer: "FlashAttention-3 leans on Hopper's asynchronous Tensor Cores and TMA: a warpgroup can launch a WGMMA and keep running softmax on the CUDA cores while the matrix operation completes in the background, so the two phases interleave inside a single kernel. RDNA4's WMMA is not asynchronous. It is an instruction issued from the wave down the same vector pipe that executes the FP32 FMAs, so a softmax multiply and a matrix multiply cannot both be executing from the same wave. The only overlap RDNA4 offers is across waves, which turns the question of how much softmax is free into a question of occupancy."
  - question: "What does folding log2(e) into the QK scale actually save?"
    answer: "RDNA4's exponential instruction is natively base-2: V_EXP_F32 computes 2^x. Writing exp(x) in the shader makes the compiler emit a full-rate multiply by log2(e) before the transcendental. Folding that constant into the query-key scale that the kernel already applies, so the scale becomes log2(e)/sqrt(head_dim), removes one vector-pipe instruction per score element. It is worth about 0.016 cycles per element in the model, roughly 2.5 percent of the attention kernel, and it costs nothing because the multiply happens once per tile instead of once per element."
  - question: "How much prefill did the softmax work actually buy?"
    answer: "About 8 percent on the attention kernel from trimming the non-matmul instruction count, and another 13 percent from getting the flash attention kernel's register footprint under the 96-VGPR line so it runs sixteen waves instead of ten. Attention fell from roughly 41 percent of prefill wall clock to about 35 percent, and end-to-end Qwen3.5-9B prefill on the RX 9070 XT moved from about 615 tok/s to about 665. The gap to llama.cpp's 973 tok/s pp512 is now roughly 1.46x, down from 1.58x."
draft: false
---

The MLP is no longer the story. Fusing the [SwiGLU intermediate into the down projection](https://zolotukhin.ai/blog/2026-07-10-down-projection-reads-back-a-swiglu-tensor-wider-than-the-hidden-state/) took Qwen3.5-9B prefill on the RX 9070 XT to about 615 tok/s and dropped the MLP block to 22 percent of the profile. Attention inherited the top slot at roughly 41 percent, which is exactly what the last three posts predicted would happen.

So I opened the flash attention kernel expecting the two GEMMs to be the problem, and they were not. The QK-transpose and PV matmuls run on WMMA and behave. The time is going to the softmax between them, and the first instinct about which part of the softmax is a trap.

Everyone's instinct, mine included, is the exponential. RDNA4 executes `V_EXP_F32` on a transcendental unit at [quarter rate](https://chipsandcheese.com/p/microbenchmarking-amds-rdna-3-graphics-architecture), which is roughly 16 exponentials per clock per compute unit against 1024 matrix FLOPs per clock. That ratio looks fatal. It is not, and the reason is the part of the machine nobody puts on a slide: the transcendental unit is separate hardware, and the ordinary multiplies and adds of the softmax are not.

## Non-matmul FLOPs are expensive everywhere, for a specific reason

The general shape of this problem is not new. It is the observation that motivated FlashAttention-2, which explicitly [tweaks the algorithm to reduce non-matmul FLOPs](https://arxiv.org/abs/2307.08691) because on an A100 the hardware has 312 TFLOP/s of FP16 matmul but only 19.5 TFLOP/s of non-matmul FP32. Tri Dao's blog post puts it in one line: [each non-matmul FLOP is 16x more expensive than a matmul FLOP](https://princeton-nlp.github.io/flash-atttention-2/). Attention is the one transformer block where that matters, because the softmax sits directly between two matmuls and touches every element of a quadratic score matrix.

RDNA4's version of that ratio is milder. A compute unit retires 1024 FP16 matrix FLOPs per clock and 128 FP32 vector FLOPs per clock, so a non-matmul FLOP is about 8x a matmul FLOP rather than 16x. On paper the AMD card should care *less* about softmax overhead than the A100 does. It does not work out that way, and the reason is structural rather than arithmetic.

On Nvidia the tensor cores are separate units from the CUDA cores. That separation is what FlashAttention-3 monetizes: it [exploits the asynchrony of Hopper's Tensor Cores and TMA to interleave block-wise matmul and softmax](https://arxiv.org/abs/2407.08608), launching a WGMMA and running the softmax for the previous block while the matrix operation completes in the background. On RDNA4 there is no such thing to exploit. WMMA is an instruction the wave issues down the same vector pipe that executes its FP32 FMAs. A wave running a softmax multiply is a wave not running a matrix multiply, and no amount of scheduling cleverness inside that wave changes it.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-11-rdna4-softmax-issue-slot-budget.svg" alt="A two-panel diagram. The left panel shows one RDNA4 SIMD: an issue port feeding two units, a vector pipe shared by WMMA and FP32 FMA instructions, and a separate quarter-width transcendental unit that runs V_EXP_F32. A red note says softmax max, subtract, sum and rescale go to the vector pipe and steal WMMA slots; a green note says softmax exp goes to the separate unit and can hide behind another wave's WMMA. The right panel is a horizontal bar chart of cycles per score-matrix element on one compute unit. The first bar, QK-transpose and PV on WMMA alone, is 0.500 cycles. The second bar, the flash attention kernel first cut, is 0.594 cycles: the same 0.500 of matmul plus a red segment of 0.094 cycles of softmax on the vector pipe, with a separate hatched strip below showing the 0.063 cycles of exp on the transcendental unit sitting off the critical path. The third bar, the trimmed softmax kernel, is 0.547 cycles, with the red vector-pipe segment cut roughly in half. A footer notes RDNA4 has no asynchronous matrix pipe, so the only overlap available is across waves, which makes occupancy rather than the transcendental rate the thing that decides how much of the softmax is free." loading="lazy" />
  <figcaption>Cycles per score-matrix element on one CU at head dim 128 with a 64-key block. The matmul needs 0.500 cycles; the softmax adds 0.094 cycles of vector-pipe work that directly displaces WMMA, and 0.063 cycles of exp that does not, because the transcendental unit is separate silicon. First-order model, RX 9070 XT rates.</figcaption>
</figure>

## Counting the instructions, not the FLOPs

Put the model on one score element and the picture resolves. At head dim 128, producing one element of the score matrix and consuming it costs 128 multiply-accumulates in the QK-transpose matmul and 128 more in the PV matmul, so 512 FLOPs, and at 1024 matrix FLOPs per clock per CU that is 0.500 cycles. Everything else in the kernel has to be measured against that half cycle.

| Per score element, one CU | ops | unit | rate/clk/CU | cycles | displaces WMMA? |
| --- | ---: | --- | ---: | ---: | --- |
| QK&#8868; + PV matmul | 512 FLOP | matrix (WMMA) | 1024 | 0.500 | — |
| running max, subtract, log2(e) scale, running sum | 4 | vector FMA | 64 | 0.063 | yes |
| accumulator rescale, 64-key block | 2 | vector FMA | 64 | 0.031 | yes |
| exp | 1 | transcendental | 16 | 0.063 | no |

The exp is the smallest line item that everyone is afraid of. Its 0.063 cycles land on a unit the matrix path is not using, so with enough resident waves it disappears into the shadow of somebody else's WMMA. The 0.094 cycles of plain vector work do not disappear anywhere. They are FP32 FMAs on the same pipe as the matrix instructions, and they push the kernel from 0.500 cycles per element to 0.594, which is 19 percent more time than the matmul needs. That is the whole non-matmul tax, and the scary instruction is a third of it.

The accumulator rescale is the line worth staring at. Online softmax keeps a running maximum, and when a new key block raises it, the output accumulator for that query row has to be scaled by the correction factor. That is 128 multiplies per row per block, and with a 64-key block it amortizes to 2 multiplies per score element. It is invisible in a FLOP count and it is a fifth of the kernel's non-matmul time.

## Three trims that do not touch the algorithm

None of the fixes are clever. They are the RDNA4 spelling of FlashAttention-2's instruction-count discipline.

Fold `log2(e)` into the query-key scale. `V_EXP_F32` computes 2^x natively, so writing `exp(x)` in GLSL makes the compiler emit a multiply by log2(e) on the vector pipe, once per score element, forever. The kernel already multiplies the scores by 1/sqrt(head_dim). Making that constant `log2(e)/sqrt(128)` deletes an instruction from the inner loop and changes nothing else.

Do the rescale in packed FP16. The correction factor is one scalar per query row, and the accumulator is a row of 128 values. `v_pk_mul_f16` handles two lanes of the accumulator per instruction, halving the rescale cost from 2 vector ops per element to 1. The accumulator stays FP32 for the running sum; only the correction pass goes packed.

Defer the divide. FlashAttention-2's version of this is not dividing by the running sum on every block and instead carrying an unnormalized output, dividing once in the epilogue. That is one fewer full-rate op per element inside the loop, and it is free.

Together those take the vector-pipe cost from 0.094 cycles per element to about 0.047, and the kernel from 0.594 to 0.547. The softmax overhead drops from 19 percent of the matmul time to 9 percent.

## The other half of the answer is occupancy again

The exp is only free if there is another wave with a WMMA ready to issue while the transcendental unit is busy. That is the same sentence as [yesterday's occupancy post](https://zolotukhin.ai/blog/2026-07-09-vgpr-pressure-caps-fused-rdna4-prefill-gemm-at-nine-waves/), pointed at a different kernel. ZINC's flash attention kernel holds the QK accumulator, the PV accumulator, the running max, and the running sum per query row, and the first honest version came in around 136 VGPRs, which puts it on the ten-wave step of the [RDNA4 occupancy staircase](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/). Ten waves is not much of a shadow to hide an exp in.

Shrinking the query tile so the kernel fits under 96 VGPRs restores all sixteen waves. That costs arithmetic intensity on the QK matmul, which is the usual trade, and here it is worth taking, because the thing being hidden is not just the transcendental but the KV loads too.

| Qwen3.5-9B prefill, RX 9070 XT | prefill | attention share | gap to llama.cpp |
| --- | ---: | ---: | ---: |
| fused MLP, first-cut attention (yesterday) | ~615 tok/s | ~41% | 1.58x |
| trimmed softmax (scale fold, packed rescale, deferred divide) | ~638 tok/s | ~38% | 1.53x |
| trimmed softmax + FA kernel under 96 VGPR | ~665 tok/s | ~35% | 1.46x |
| llama.cpp `pp512` reference | 973 tok/s | — | — |

Roughly 8 percent of the attention kernel came from deleting instructions and another 13 percent from having waves to hide the rest behind. End to end that is 615 to 665 tok/s, an 8 percent prefill gain, which is the smallest step in this series and the one with the least glamour attached to it.

## What this says about porting attention kernels to AMD

The uncomfortable finding is that RDNA4 flash attention is not slow for a reason anyone can fix with a better softmax. It is slow because the card is missing the two hardware features the fast Nvidia kernels are built on. There is no asynchronous matrix pipe, so the FlashAttention-3 trick of interleaving softmax with an in-flight GEMM has nothing to interleave against. And RDNA4 has [no in-register matrix transpose](https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-3/), no `movmatrix` or `ldmatrix.trans`, which is why llama.cpp's RDNA4 flash attention path has to [synthesize one by multiplying against an identity matrix through WMMA](https://github.com/ggml-org/llama.cpp/commit/ea4a321f2a607ca315d998f0656fd255715884a6). AMD's own June 2026 guide presents that identity trick as the recommended workaround, which tells you how much of the RDNA4 attention story is currently spent working around missing instructions rather than using present ones.

That reframes the goal. On Hopper, attention gets fast by overlapping non-matmul work with matmul work. On RDNA4 the only version of overlap available is across waves, which means the levers are the boring ones: issue fewer non-matmul instructions, and keep enough waves resident that the ones you do issue land on a unit somebody else is not using. Both of those are register-file problems wearing a softmax costume.

The gap to llama.cpp is 1.46x now, and for the first time in this series the next step is not obvious. The bytes are fused, the registers are budgeted, the instruction count is trimmed. What is left is a matrix pipe that stalls whenever a wave has anything else to do, and that is not a kernel bug. It is the shape of the chip.
