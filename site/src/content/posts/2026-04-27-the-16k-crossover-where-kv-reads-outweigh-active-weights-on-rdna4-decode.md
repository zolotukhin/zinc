---
title: "The 16k crossover where KV reads outweigh active weights on RDNA4 decode"
date: "2026-04-27"
tags:
  - zinc
  - rdna4
  - amd
  - decode
  - kv-cache
  - moe
  - roofline
  - bandwidth
  - long-context
  - vulkan
  - llm-inference
  - qwen35
keywords:
  - RDNA4 decode bandwidth ceiling
  - KV cache decode roofline
  - Qwen 3.5 35B A3B decode tok/s
  - Radeon AI PRO R9700 memory bandwidth
  - bytes per decode token
  - active weights vs KV cache crossover
  - long context decode local LLM
  - MoE active parameters bandwidth
  - autoregressive decode memory bound
  - flash attention KV read traffic
excerpt: "On a Radeon AI PRO R9700, decoding one token of Qwen 3.5 35B-A3B reads about 2 GB of active weights from VRAM. At 16k context the KV cache hits the same 2 GB, and past that point the KV cache is the larger tenant on every decode step. The bandwidth ceiling falls from roughly 213 tok/s at 8k to 37 tok/s at 128k, and the part of the picture that moves is not the weights."
---

The first time we looked carefully at Qwen 3.5 35B-A3B decode throughput on an AMD Radeon AI PRO R9700, the obvious bandwidth math said the model should comfortably hit two hundred tokens per second. Three billion active parameters at Q4_K_M is roughly two gigabytes of weight reads per token, and the [R9700's 640 GB/s of GDDR6 bandwidth](https://www.tomshardware.com/pc-components/gpus/amd-launches-radeon-ai-pro-r9700-to-challenge-nvidias-ai-market-dominance) divides into that comfortably. The actual decode throughput at 32k context was less than half of that. The reason is not that the kernels were slow. The reason is that nobody in the bandwidth budget was counting the KV cache.

This post is the analytical case for why long-context decode on a 32 GB consumer RDNA4 card crosses a regime boundary somewhere around 16,000 tokens, why the optimization story past that boundary is a different problem from the optimization story in front of it, and why the right unit for thinking about decode performance is bytes per token across both the weights and the KV cache, not just the weights.

The implication is engineering work. The right kernel to optimize first depends on which side of the crossover the user's prompt sits on, and the answer for a 1k-token chat is not the answer for a 64k-token RAG context.

## What an autoregressive decode step actually reads

A single decode step on a memory-bound transformer reads three things from VRAM. The active weights for the layers that fire on this token, the per-token state that the model carries between tokens, and any intermediate activations that spill out of registers and on-chip caches into DRAM. For a quantized model on a card whose compute throughput sits hundreds of times above its memory bandwidth at batch size one, the wall-clock cost of the decode step is set by which of those three is largest, multiplied by how fast the card can pull bytes from DRAM. Pierre Lienhart's [LLM inference series, part five](https://medium.com/@plienhar/llm-inference-series-5-dissecting-model-performance-6144aa93168f) walks the general case in detail. The version of the argument that matters here is the model-specific one.

For Qwen 3.5 35B-A3B at Q4_K_M, the active-weight read per decode token is approximately two gigabytes. The architecture is hybrid: 60 layers split between attention and a Gated DeltaNet state-space block, with a 256-expert MoE feed-forward fired with eight routed experts plus one shared expert per token, as described on the [Hugging Face model card](https://huggingface.co/Qwen/Qwen3.5-35B-A3B). The decode step pulls the attention projections for every layer, the SSM weights for the SSM layers, the routed experts plus the shared expert for the MoE layers, and the embedding plus language-model head. At roughly 4.5 effective bits per weight for Q4_K_M and three billion active parameters, the active-weight bandwidth per token is `3e9 * 4.5 / 8 = 1.69 GB`, which we round to two gigabytes after counting the small overheads on the routing buffers and the per-layer norms.

The per-token state for the SSM layers is about 60 megabytes total, dominated by the gated delta-net hidden state across the 30 SSM layers. That number is constant in context length, because the SSM state is a fixed-size summary of all prior tokens. Yesterday's [gate analysis post](/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4) made this number visible from the prefill side; on the decode side it adds a flat sixty megabytes per token to whatever else the engine is reading.

The KV cache, by contrast, scales linearly with prompt length. At 60 layers, 4 grouped-query KV heads, and a 128-element head dimension, the per-token KV footprint at FP16 is `2 * 60 * 4 * 128 * 2 = 122,880` bytes, exactly 120 KiB. The decode step reads the entire KV slab the attention shader needs, which on a standard flash-attention dispatch is the complete K and V for every prior token across every attention layer. At 8k context that is 0.96 GB of KV reads. At 128k it is 15.36 GB, derived in the same arithmetic the [prior FP16 KV writeup](/blog/2026-04-26-why-fp16-kv-cache-is-the-wrong-default-for-128k-context-on-32gb-rdna4) used to argue against FP16 as a default representation.

## The crossover

The interesting number is the prompt length at which the KV cache reads equal the active-weight reads. For 2 GB of active weights and 120 KiB per token of KV, that crossover lands at roughly 17,000 tokens. We round it to 16k for headline purposes because the difference is below the noise floor of how anyone budgets a context window.

![Decode bytes per token on Qwen 3.5 35B-A3B as a function of context length, with the active-weight floor and three KV cache representations plotted against the 640 GB/s R9700 bandwidth ceiling.](/blog/2026-04-27-rdna4-decode-bandwidth-crossover.svg)

The chart is the post in one image. Below 16k context the curve is dominated by the flat 2 GB of active-weight reads, and the decode tok/s ceiling sits in the 200 to 300 tok/s range regardless of how much KV optimization the engine has done. Above 16k context the KV slope takes over, and the only lever that moves the curve is the KV representation. At 128k the FP16 KV configuration is reading 15.4 GB per token of KV alone, and the 17.4 GB total would saturate the R9700's bandwidth at 37 tok/s in the best case. INT8 KV halves the slope. A Q4 K plus INT8 V configuration along the lines that [llama.cpp's 4-bit KV tracking issue](https://github.com/ggml-org/llama.cpp/issues/6863) discusses cuts it again. The point of the plot is not the absolute numbers, which depend on how cleanly the attention shader actually saturates DRAM. The point is the shape: two regimes, one crossover, and a different optimization target on each side.

The classic [Williams, Waterman, and Patterson roofline paper](https://dl.acm.org/doi/abs/10.1145/1498765.1498785) frames this kind of analysis as an arithmetic-intensity problem. The decode step on a transformer at batch size one is so far below the compute roofline that the only useful axis is the bandwidth ceiling, and the question reduces to how many bytes the kernel has to touch per output token.

## What changes on each side of the crossover

The kernels that win on the short-context side of the crossover are not the kernels that win on the long-context side. This is the part that surprised us.

In the short-context regime, where active-weight reads dominate, the high-leverage optimizations are weight quantization, expert layout, and the cost of the routing scatter. Going from Q4_K_M to a tighter quantization shape shrinks the dominant term proportionally. Routing-aware kernel shapes that read each active expert's weights once and reuse them across all routed tokens in the batch shrink it again. The KV cache on this side of the crossover is a small enough term that flash-attention quality matters less than cache-line discipline on the weight reads. This is the regime the bulk of recent ZINC prefill work has been targeting, because prefill at 154 tokens lives squarely on this side.

In the long-context regime, where KV reads dominate, the active-weight side of the picture is largely solved relative to where the bytes actually are. The high-leverage optimization is the KV cache representation: precision, layout, and the attention shader's ability to dequant inline rather than through a scratch buffer. The well-trodden community result on llama.cpp's [KV quantization discussion](https://github.com/ggml-org/llama.cpp/discussions/5932) is that Q8_0 K and Q8_0 V are within noise of FP16 on most workloads, that K can be quantized further than V before quality breaks down, and that the asymmetric pattern of Q4 K plus Q8 V is the practical floor. Google's [TurboQuant work](https://github.com/ggml-org/llama.cpp/discussions/20969), recently ported to gfx1100, takes that floor further. Each step on that ladder cuts the KV slope of the chart and pushes the decode ceiling back up at long context.

The flat region between 1k and 4k context, where the active-weight slab is roughly twice the KV slab, is where most chat traffic actually lives. That is also where most local-inference benchmark numbers get recorded, which is part of why the long-context cliff is rarely visible in headline tok/s figures. The single-turn `llama-bench` numbers that get quoted around the R9700 sit in the short-context regime by construction.

## The math, made boring on purpose

Spelling the byte budget out as a table makes the optimization picture concrete. Each row is a context length; each column is a candidate KV representation. The total is the sum of two gigabytes of active weights and the per-row KV bytes. The ceiling is 640 divided by the total, in tokens per second.

| Context | FP16 KV bytes | Q8_0 KV bytes | Q4 K + Q8 V bytes | Decode ceiling (FP16 KV) | Decode ceiling (Q8_0 KV) | Decode ceiling (Q4 K + Q8 V) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1k | 0.12 GB | 0.06 GB | 0.045 GB | 302 tok/s | 311 tok/s | 313 tok/s |
| 4k | 0.48 GB | 0.24 GB | 0.18 GB | 258 tok/s | 286 tok/s | 294 tok/s |
| 8k | 0.96 GB | 0.48 GB | 0.36 GB | 216 tok/s | 258 tok/s | 271 tok/s |
| 16k | 1.92 GB | 0.96 GB | 0.72 GB | 163 tok/s | 216 tok/s | 235 tok/s |
| 32k | 3.84 GB | 1.92 GB | 1.44 GB | 110 tok/s | 163 tok/s | 186 tok/s |
| 64k | 7.68 GB | 3.84 GB | 2.88 GB | 66 tok/s | 110 tok/s | 131 tok/s |
| 128k | 15.36 GB | 7.68 GB | 5.76 GB | 37 tok/s | 66 tok/s | 82 tok/s |

The shape of the table is the point. Read down the FP16 KV column and decode tok/s falls by 8x between 1k and 128k. Read across at 128k, and a single quantization step from FP16 to Q8 nearly doubles the ceiling, and a second step to Q4 K plus Q8 V doubles it again on top of that. None of these cells are real benchmark numbers; they are the bandwidth-ceiling math under the standard assumption that the attention kernel approaches memory-bandwidth saturation. The real numbers will sit below the ceilings, and the gap between them is exactly the engineering room a Vulkan attention shader has to grow into.

## Why the per-token state does not change the picture

The 60 megabytes of SSM state that the gated delta-net layers carry between tokens is real bandwidth, but it does not change the argument because it is constant in context length. It moves every row of the table down by an identical amount and shifts the crossover by a few hundred tokens. The reason it deserves a callout is that on this hybrid architecture it is the structural reason ZINC's [SSM state cannot live in registers across the prompt](/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4) yet, and the wall it imposes on prefill has a smaller analog on decode that is worth being explicit about. A future block-resident SSM kernel would not change the crossover for decode either; it would just remove the constant.

The other footnote is the difference between dense 35B at Q4_K_M, which would be reading roughly 20 GB of weights per decode token, and 35B-A3B at Q4_K_M, which is reading roughly 2 GB. The MoE structure shrinks the weight side of the budget by about 10x. That is what gives the KV cache a chance to be the larger term at all. On a dense 35B model, the KV cache stays subordinate to the weight reads even at 128k context, and the entire picture is closer to a flat 32 tok/s decode ceiling at any prompt length the card can hold. The hybrid MoE-plus-SSM architectures that are showing up across Qwen, DeepSeek, and the next round of frontier-class small-active models are exactly the family that makes KV cache representation the load-bearing engineering decision for long context. The asymmetry is structural to the architecture.

## What this means for the ZINC roadmap

The first thing the crossover changes is the order of the work. Going into the second quarter of the year the assumption was that decode performance was a steady-state property of the engine, and the next round of optimization belonged to prefill. The crossover plot makes that assumption wrong for any prompt longer than a chat turn. Long-context decode is its own kernel surface with its own bandwidth bottleneck, and the FP16 KV cache that ZINC ships today is the load-bearing default that gives away most of the room.

The second thing it changes is which kernel rewrite earns the most. An attention shader that reads quantized K and V directly, dequants in the inner loop, and never round-trips through an FP16 scratch buffer is the same shape of work as the [Q8_1 mul_mmq](/blog/2026-04-19-why-q8-1-activations-are-the-next-rdna4-prefill-unlock) port that lifted prefill, applied to a different tensor. The expected payoff is larger, because the bytes the shader is replacing are linear in context length rather than constant.

The third is a benchmark hygiene point. Single-turn tok/s on a 100-token prompt sits in the regime where the active weights are the dominant term, the KV cache is a footnote, and the decode ceiling is roughly 250 tok/s. Multi-turn tok/s on a 32k context sits in the regime where the KV cache is the dominant term, the active weights are the floor, and the decode ceiling is roughly 100 tok/s. Reporting one number without the other is the local-inference equivalent of quoting peak FLOPs without the bandwidth ceiling: directionally true, structurally misleading. The right number to publish is a curve, not a point.

## What comes next

The work is straightforward to enumerate. Quantize the K and V buffers. Teach the attention shader to read them quantized. Measure the decode tok/s curve as a function of context length. Compare the measured curve to the bandwidth ceiling. The gap between the two is the next round of attention-kernel work, and the slope of the curve is the new diagnostic that says whether the engine is paying for KV bytes it could have stopped paying for.

The 16k crossover is not a magic number. It is the prompt length at which a particular hybrid architecture, on a particular consumer card, with a particular default KV representation, swaps which term in the bandwidth budget is largest. The number will move as the active parameter count moves, as the KV representation tightens, and as the next generation of cards lifts the bandwidth ceiling. The shape of the argument will not. Long-context decode on a single consumer-grade card is an attention-kernel problem, not a weight-quantization problem, and the right time to plan for it is before the user has typed the prompt that exposes the cliff.
