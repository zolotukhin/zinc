---
title: "Why RDNA4 long-prefill plateaus on attention, not GEMM"
date: "2026-05-01"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - prefill
  - attention
  - flash-attention
  - cooperative-matrix
  - moe
  - long-context
  - llm-inference
  - qwen35
keywords:
  - RDNA4 long prefill attention
  - flash attention RDNA4 Vulkan
  - QK^T softmax PV crossover
  - attention compute vs GEMM compute
  - Qwen3-30B-A3B prefill bottleneck
  - Radeon AI PRO R9700 long context prefill
  - cooperative matrix attention RDNA4
  - O(N^2) attention compute crossover
  - prefill ceiling local LLM
  - sliding window attention long context
excerpt: "Prefill on the Radeon AI PRO R9700 has two ceilings, not one. The matrix cores fix the GEMM ceiling. They do nothing for the attention ceiling. On Qwen3-30B-A3B the attention FLOP count crosses the active-weight GEMM FLOP count near 16k tokens, and past the crossover the same prefill spends more time in QK^T and PV than in every linear layer combined. Flash attention is what makes that ceiling reachable in the first place. The next lever past it is algorithmic, not numeric."
---

The marketing arc for the [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) is the AI accelerator. Read any of the launch coverage and the headline number is 195 dense FP16 TFLOPS through the second-generation matrix engines. The implicit promise is that prefill, the workload where those engines were always going to matter, scales with that ceiling. For prompts up to a few thousand tokens it does. Past that, it stops.

The bottleneck does not move because the kernel is wrong. It moves because the workload changed shape. Prefill on Qwen3-30B-A3B at short context is dominated by the linear layers, where the compute scales with the number of tokens times the number of active parameters. Prefill at long context is dominated by self-attention, where the compute scales with the square of the number of tokens times the model width. The two terms cross at a sequence length that depends on the model, and on Qwen3-30B-A3B that crossover lands near 16k tokens. Past it, the matrix cores are still a real lever, but the lever they pull on is no longer the binding one.

This post is the math behind that crossover, the reason it lands where it lands on a 35B-A3B-class model, and why fixing it is an algorithmic problem on the attention side rather than a kernel problem on the GEMM side.

## The two ceilings, named carefully

Prefill is the phase that processes the prompt before the first token is sampled. Every linear layer reads the same weight tile from VRAM and multiplies it against every token in the prompt before moving to the next layer. The arithmetic intensity of those linear layers is set by the prompt length: at one token the layer is GEMV, at a thousand tokens it is GEMM, and the crossover from bandwidth-bound to compute-bound is the [32-column DMMV crossover from last week's post on RDNA4 prefill kernels](/blog/2026-04-22-why-rdna4-prefill-wants-a-32-column-dmmv-before-a-gemm).

Self-attention is the other half of the workload. For a single decoder layer at sequence length `N`, the QK^T matrix is `[N, N]` per head, the softmax runs over each row of that matrix, and the PV matmul reads the same `[N, N]` matrix back. The compute count is `4 * H_q * head_dim * N^2` per layer, where `H_q` is the number of query heads and `head_dim` is the per-head dimension. The dependence on `N` is quadratic, and unlike the linear-layer term, that quadratic term has nothing to do with the parameter count.

A linear term and a quadratic term in the same kernel make for a familiar pattern. Below the crossover the linear term dominates. Above it the quadratic does. The matrix cores accelerate both terms, but they cannot move the crossover, because both terms run on the same engines.

## Where the crossover sits on Qwen3-30B-A3B

The publicly documented [Qwen3-30B-A3B model card](https://huggingface.co/Qwen/Qwen3-30B-A3B) gives the specifics. The model has 48 transformer layers, hidden dimension 2048, 32 query heads and 4 key-value heads with grouped-query attention, head dimension 128, and 128 experts with 8 active per token. The full parameter count is 30.5 billion, and the active count per token is roughly 3.3 billion. The [Qwen3 technical report](https://arxiv.org/abs/2505.09388) confirms the active-parameter routing and gives the same shape across the family. The 35B-A3B variant ZINC's earlier prefill posts target shares the same active-parameter footprint and a slightly larger layer count.

Plug those numbers into both terms.

Linear-layer compute for `N` prefill tokens, against 3.3 billion active parameters per token:

`C_linear = 2 * N * 3.3e9 = 6.6e9 * N` FLOPs.

Self-attention compute, summed across 48 layers, with the causal mask halving the work:

`C_attn = 0.5 * 48 * (4 * 32 * 128 * N^2) = 393,000 * N^2` FLOPs.

The two are equal when `6.6e9 * N = 393,000 * N^2`, which gives `N ≈ 16,800`. That number is the per-prompt compute crossover for this model. Below 17k tokens, the linear-layer term carries more FLOPs. Above it, attention does.

For the larger 35B-A3B configuration with a 64-layer stack and a 4096 hidden dimension, the same arithmetic with `H_q * head_dim = 4096` gives an attention coefficient of about 524,000 and a crossover near 12,500 tokens. The number is model-specific, but the order of magnitude is the same. On any A3B-class model on this card, the attention term overtakes the linear-layer term somewhere in the low-to-mid five-figure context range.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-01-rdna4-prefill-attention-vs-gemm-quadratic.svg" alt="Four side-by-side panels for Qwen3-30B-A3B prefill on the AMD Radeon AI PRO R9700, one each for sequence length 1024, 4096, 8192, and 16384 tokens. Each panel shows two squares drawn with area proportional to FLOP count: an orange GEMM square that grows linearly with sequence length, and a blue self-attention square that grows quadratically. At N=1024 the GEMM square dwarfs the attention square at a 94 to 6 ratio. By N=4096 the attention square has grown to roughly a quarter of the GEMM area. By N=8192 attention is one third of the total. At N=16384 the two squares are visually equal in area, marked with a dashed red line labeled attention quadratic catches up. A footer notes the formula and the area-to-FLOP scaling." loading="lazy" />
  <figcaption>Compute share between the linear and attention terms for Qwen3-30B-A3B prefill, with square area proportional to FLOPs. The GEMM term grows in the prompt length; the attention term grows in its square.</figcaption>
</figure>

The picture the panels make is the table the next section will read out in numbers. The leftmost panel is the regime every prefill kernel has been optimized for. The rightmost panel is the regime nobody has spent much engineering time on, because no commodity model could fit a 16k-token prompt on a consumer card until the A3B family arrived with 32 GB of RDNA4 memory underneath it.

## What the numbers say at common prompt lengths

The compute count is one thing. The wall time is the other. Treating both terms as compute-bound at long prefill, and treating the dense matrix throughput on the R9700 as roughly 195 TFLOPS, the ms-per-prefill split looks like this for Qwen3-30B-A3B.

| Prompt length `N` | Linear-layer FLOPs | Attention FLOPs | Linear ms | Attention ms | Attention share |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 6.8e12 | 4.1e11 | 35 | 2 | 6% |
| 4,096 | 2.7e13 | 6.6e12 | 138 | 34 | 19% |
| 8,192 | 5.4e13 | 2.6e13 | 277 | 134 | 33% |
| 16,384 | 1.1e14 | 1.0e14 | 555 | 535 | 49% |
| 32,768 | 2.2e14 | 4.2e14 | 1110 | 2140 | 66% |
| 65,536 | 4.3e14 | 1.7e15 | 2220 | 8560 | 79% |

The wall numbers in the table are the compute-bound floor on this card under the assumption that the kernels actually feed the matrix engines. They are not measurements; they are the lower bounds set by the FLOP count and the spec sheet. What the table makes obvious is the shape of the bend. At 4k tokens, attention is a fifth of the compute. At 16k it is half. At 64k it is four-fifths. The matrix throughput moves both columns by the same factor; it does not change which one binds.

Below the crossover, every prefill optimization that moves the floor is a kernel optimization. The 32-column DMMV port, the cooperative-matrix tiles, the [single push constant fix on `flash_attn_batched.comp` for Gemma 4](/blog/2026-04-24-the-single-push-constant-blocking-gemma-4-prefill-on-rdna4), the [single-vkQueueSubmit work](/blog/2026-04-25-why-one-vkqueuesubmit-per-prompt-is-the-next-quiet-rdna4-prefill-unlock): all of them live to the left of the crossover. They are how prefill on a 4k-token prompt drops from a second to a quarter-second. Past the crossover they are still doing work, but the work they are doing is not the work that bounds the wall.

## Why flash attention does not move the crossover

The temptation when reading the table is to point at flash attention and call it the answer. That misreads the algorithm. The original [FlashAttention paper from Dao et al.](https://arxiv.org/abs/2205.14135) is explicit that the contribution is IO-awareness, not FLOP reduction. The algorithm tiles QK^T into blocks that fit in on-chip SRAM, computes the row-wise softmax incrementally, and avoids ever materializing the full `[N, N]` attention matrix in HBM. The HBM access count drops from `O(N^2)` to `O(N^2 d / M)` where `M` is the SRAM size. That is the IO win, and it is significant: on workloads where the materialization was the bottleneck, the wall time falls in proportion.

The compute count does not. The number of multiply-add operations to compute exact attention is the same with or without tiling. [FlashAttention-2](https://arxiv.org/abs/2307.08691) and the more recent [FlashAttention-3 work for Hopper](https://arxiv.org/abs/2407.08608) both improved the kernel's parallelization and asynchrony, and FlashAttention-3 added FP8 support for the matmul half, but neither one changed the asymptotic FLOP count of exact attention. It is `4 * H_q * head_dim * N^2` per layer no matter how the kernel is scheduled.

This matters on RDNA4 because the [llama.cpp Vulkan backend's cooperative-matrix flash-attention path](https://github.com/ggml-org/llama.cpp/discussions/19890) is already roughly the right kernel. The R9700 community benchmarks on Qwen3-30B-A3B Q4_K_M reach 92% of the card's bandwidth ceiling at decode and pull comparable utilization at prefill below 4k tokens. At 32k tokens the same kernel is doing more work, and the work it is doing is bounded by the matrix throughput on the spec sheet, not by HBM traffic. The kernel is good. The workload it is feeding is bigger.

## Where the cooperative matrix tile actually pays in attention

The case for cooperative-matrix attention is real, but it is the case for raising the prefill ceiling, not for shifting the crossover. With the FlashAttention-2-style block-wise QK^T and PV matmuls running through `VK_KHR_cooperative_matrix` against the second-generation AI accelerators, both halves of the attention compute see the same matrix-throughput multiplier as the linear-layer GEMMs. That is the multiplier that makes the table above achievable on real silicon rather than the theoretical compute-bound floor.

What it does not do is push the attention term off the wall budget. With identical multipliers on both sides, the share of wall time spent in attention at a given sequence length is invariant. Doubling the matrix throughput halves both columns. The 49% attention share at 16k tokens stays a 49% attention share at 16k tokens. The crossover sits where the workload says it sits.

## What this means for the engine roadmap

The practical takeaway is that there are two distinct prefill regimes on this card, and the engine work required to hit ceiling in each one is different. Below the crossover the wall is set by GEMM throughput and dispatch overhead, and the kernel arc that has run through the [last several](/blog/2026-04-22-why-rdna4-prefill-wants-a-32-column-dmmv-before-a-gemm) [posts](/blog/2026-04-23-vulkan-specialization-constants-unlock-rdna4-dmmv-variants) is the arc that closes that gap. Above the crossover the wall is set by attention compute, and the lever stops being a kernel choice.

The levers that actually move the wall above the crossover are algorithmic. Sliding-window attention, which Gemma 4 already uses on five out of every six layers, replaces the `O(N^2)` term with an `O(N * W)` term for a fixed window `W`. Mixture-of-attention layers that route different tokens through different attention spans do the same thing with a softer floor. KV cache eviction, which prunes the keys and values that older tokens have stopped attending to, makes the effective `N` smaller than the prompt length without changing the model. None of those are RDNA4 work. They are model-architecture work that local inference engines either accept or reject when picking which model to ship.

The decode parallel here is the [16k decode crossover](/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode) from earlier in the week. Decode hits a bandwidth wall where the KV cache reads overtake the active-weight reads. Prefill hits a compute wall where the attention quadratic overtakes the linear-layer term. Both walls are workload artifacts. Neither one is a kernel artifact. The difference is that the decode wall has a known answer in KV quantization, and the prefill wall does not have an equally clean equivalent on the attention side.

## What we are watching

The two things to watch from the local-inference seat are the same two things that have been visible on the server side for a year. The first is whether the next generation of consumer-class long-context models commits to sliding-window attention as the default rather than the exception, the way Gemma 4 has and the way the [Mistral 7B paper](https://arxiv.org/abs/2310.06825) argued for in 2023. If they do, the prefill wall on local cards stops being a wall for any prompt the user actually types. The second is whether RDNA4's matrix cores get exposed through cooperative-matrix tiles in the open-source attention kernels at the same quality as `mul_mm_cm2` already exposes them in the linear-layer GEMM path. The Vulkan-side surface is in place. The shaders are mostly in place. What is missing is the same kind of shape audit that found the [single push constant in `flash_attn_batched.comp`](/blog/2026-04-24-the-single-push-constant-blocking-gemma-4-prefill-on-rdna4), only repeated against the cooperative-matrix path.

When both of those land, the prefill wall on a 32 GB RDNA4 card is set by the bandwidth on the spec sheet for prompts under 4k and by algorithmic attention reductions for prompts above 16k. That is a much friendlier wall than the one in the table above.
