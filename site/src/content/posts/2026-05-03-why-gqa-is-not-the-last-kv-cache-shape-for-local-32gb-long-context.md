---
title: "Why GQA is not the last KV-cache shape for local 32 GB long-context decode"
date: "2026-05-03"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - kv-cache
  - long-context
  - attention
  - mla
  - gqa
  - deepseek
  - qwen35
  - llm-inference
keywords:
  - multi-head latent attention RDNA4
  - MLA vs GQA local inference
  - DeepSeek-V3 KV cache 32 GB
  - Qwen3-30B-A3B GQA KV cache size
  - low-rank KV cache compression Vulkan
  - 128k context KV memory budget
  - decoupled RoPE MLA local
  - KV cache shape local LLM
  - long-context decode bandwidth wall
  - Radeon AI PRO R9700 attention shape
excerpt: "Grouped-query attention shrank the KV cache enough to make 128k local context fit on a 32 GB RDNA4 card. It does not shrink it enough to leave headroom. The next reduction is structural, not numerical: low-rank latent attention from the DeepSeek-V2 lineage halves the cache again on a Qwen-class model and turns the bandwidth wall on long-context decode into a different shape entirely."
---

The cleanest argument for grouped-query attention used to be that it was the largest free lunch a transformer architect could ship. Cut the number of K and V heads from thirty-two to four, save seven-eighths of the KV cache, lose almost no quality. Llama 2 took it. Qwen3 took it harder, with eight query heads per KV head. By 2026 every open-weight model worth running at long context on local hardware ships with an aggressive GQA ratio, and the lever has run out of room. The KV cache on Qwen3-30B-A3B at 128k context still takes twelve gigabytes. On a 32 GB RDNA4 card holding a four-bit-quantized weight set, that is the load-bearing fraction of the budget.

The structural fix that actually moves the wall is not a different GQA ratio. It is a different cache shape. [DeepSeek-V2](https://arxiv.org/abs/2405.04434), shipping in May 2024, replaced multi-head and grouped-query attention with a low-rank latent projection: the K and V tensors at each layer compress into a small shared latent vector, the cache stores that vector, and the per-head keys and values are reconstructed during attention. The headline number was a 93.3 percent KV cache reduction against the team's dense 67B MHA model. The relevant number for a local engine running an already-GQA Qwen-class checkpoint is smaller and more interesting.

This post is the math for what multi-latent attention buys back on a 32 GB RDNA4 long-context decode, where GQA already did most of the easy work and where the local bandwidth wall lives in a place GQA cannot reach.

## What GQA leaves on the table

The original [GQA paper from Ainslie and coauthors at EMNLP 2023](https://arxiv.org/abs/2305.13245) framed the work as a generalization of multi-query attention. MHA gives every query head its own K and V. MQA collapses all heads to a single K and V. GQA picks an intermediate group count: enough KV heads to preserve quality, few enough to compress the cache. The arithmetic is linear in the KV-head count. Cut the count by eight, cut the cache by eight.

The Qwen3-30B-A3B [model card on the Qwen3 GitHub release](https://github.com/QwenLM/Qwen3) is a typical aggressive GQA shape: 32 query heads, 4 KV heads, 128 head dim, 48 layers. That gives a per-token KV footprint of `2 * 4 * 128 * 48 * 2 = 98,304` bytes at FP16, or roughly 96 KB per token. Multiply by 128k tokens and the resident KV cache is twelve gigabytes, on the same 32 GB card holding two gigabytes of active weights per decode step plus the rest of the model's resident weight pages.

The decode-side bandwidth math from [the 16k crossover post](/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode) showed where this breaks. Past sixteen thousand resident tokens, the KV bandwidth term overtakes the active-weight term. Past sixty-four thousand, KV is the only term that matters. GQA pushed that crossover from "a few thousand tokens" up to "sixteen thousand," which is real engineering progress, but the curve still rises with context length the way it always did. The slope is set by the per-token cache size, and GQA used the only lever it had: head count. The lever it did not use, because it predates the work, is to compress along the head dimension. That is what MLA does.

## What MLA actually stores

Multi-head latent attention starts from a different assumption than GQA. The keys and values for a given token are not maintained as a stack of full per-head tensors at all. They are held as a single low-rank vector per layer, and the per-head reconstruction is done only when an attention score is being computed. The cache holds the latent. The compute does the rest.

The DeepSeek-V2 paper sets the latent compression dimension `d_c` at five hundred and twelve. The rope-coupled positional information cannot be folded into that latent without breaking position encoding, so MLA carries a small extra `d_h^R = 64` dimension shared across heads to hold the rotary keys. Per layer, per token, the cache holds `d_c + d_h^R = 576` half-precision values. For DeepSeek-V3 with sixty-one layers, that is `576 * 61 * 2 = 70,272` bytes per token. The [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) gives the exact dimensions, and the [DeepSeek-V3 reference implementation on GitHub](https://github.com/deepseek-ai/DeepSeek-V3) confirms the numbers in code.

The quality argument is what makes the cache size relevant rather than just clever. The DeepSeek paper measures MLA against MHA, GQA, and MQA on the same training budget and reports that MLA matches or beats MHA on every benchmark while shrinking the cache below MQA. The mechanism is interesting: a low-rank projection is a form of compression, but it is a learned compression that the training process gets to optimize jointly with the rest of the network. GQA's compression is fixed at the architecture step. MLA's compression has gradients flowing through it.

For a Qwen-class model retrofitted with the same shape, the bytes-per-token math becomes `(512 + 64) * 48 * 2 = 55,296`, or fifty-four kilobytes per token. The full 128k cache drops from twelve gigabytes to under seven. That is not a 28x reduction. It is a 1.8x reduction, because Qwen already cashed in most of GQA's lever. But on a 32 GB card, that 1.8x is the difference between 128k context fitting tightly with no room for batch and 128k context fitting comfortably with room for a second sequence or a longer prompt.

## The shape on the page

The structural difference is easier to read as a diagram than as bytes per token.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-03-kv-cache-shape-mha-gqa-mla.svg" alt="Three vertical stacks of stacked colored blocks compare per-token KV cache shape across MHA, GQA, and MLA on a 48-layer Qwen-class transformer. The MHA stack on the left is the tallest, drawn as 32 thin teal blocks for keys plus 32 thin slate blocks for values, totaling 786 kilobytes per token. The GQA stack in the middle is much shorter, with 4 wider teal key blocks plus 4 slate value blocks, totaling 96 kilobytes per token. The MLA stack on the right is the shortest, drawn as a single wide green block for the 512-dimension shared latent and a thin purple block above it for the 64-dimension decoupled RoPE keys, totaling 54 kilobytes per token. Below each stack a numeric annotation gives the resident KV cache size at 128k context: 96 GB for MHA, 12 GB for GQA, 6.75 GB for MLA. A small note at the bottom of the figure describes the bandwidth tradeoff: MLA does an extra rank-up matmul per attention step in exchange for the smaller cache." loading="lazy" />
  <figcaption>Per-token KV footprint at FP16 across MHA, GQA, and MLA on a 48-layer Qwen-class shape. GQA is a head-count reduction; MLA is a dimensional compression. The two levers compose, but only one of them has any room left.</figcaption>
</figure>

Two things are worth noticing in the figure. The MHA stack is what every model in the open-weight family looked like before Llama 2. It is a useful baseline because it sets the absolute ceiling on how much cache the design space allows. The GQA stack is what every model in the open-weight family looks like today. The MLA stack is what one model family already ships, and what a port to other open-weight checkpoints would look like if the weights were retrained with a low-rank attention block.

The vertical scale on the MLA stack is what makes the rest of the post worth writing. The cache is small enough that the resident memory budget on a 32 GB card stops being the binding constraint. What becomes binding instead is the bandwidth-versus-compute tradeoff at attention time, which is where MLA pays its price.

## What the bandwidth wall looks like under MLA

The recent posts on RDNA4 prefill and decode have been a long argument that the engine on a 32 GB card is bandwidth-bound on long context and compute-bound on short. The [decode roofline post](/blog/2026-04-30-rdna4-matrix-cores-sit-out-the-decode-loop) showed that the matrix cores are unused during decode because the workload is bandwidth-limited. The [long prefill plateau post](/blog/2026-05-01-why-rdna4-long-prefill-plateaus-on-attention-not-gemm) showed that prefill on Qwen3-30B-A3B above sixteen thousand tokens is attention-bound, not GEMM-bound, because the quadratic term in `N` overtakes the linear term.

MLA bends both curves. On the decode side, the per-step KV bandwidth drops by a factor of roughly 1.8 against Qwen-style GQA at the same context. On the prefill side, the dense reconstruction matmul that MLA inserts before each attention block is exactly the kind of dense GEMM the matrix cores were designed for, so the attention term shifts a fraction of its FLOPs out of the bandwidth-bound flash-attention kernel and into the cooperative-matrix path.

The arithmetic on the decode side at 128k context for a Qwen-class shape, treating the [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) as the 644 GB/s bandwidth target and ignoring dispatch overhead, looks like this:

| Attention shape | KV bytes/token | KV at 128k | KV ms/step | Active-weight ms/step | Decode total |
| --- | ---: | ---: | ---: | ---: | ---: |
| MHA (hypothetical) | 786 KB | 96 GB | does not fit | 3.1 | n/a |
| GQA, 4 KV heads (Qwen3) | 96 KB | 12 GB | 19.0 | 3.1 | 22.1 |
| MLA, d_c=512, d_h^R=64 | 54 KB | 6.75 GB | 10.7 | 3.1 + 0.4 | 14.2 |
| MLA + 64k window | 54 KB | 3.4 GB | 5.4 | 3.1 + 0.4 | 8.9 |

The GQA row is what an unmodified Qwen3-30B-A3B decode looks like at 128k. The MLA row is what the same model would look like if a future revision of the same family adopted MLA. The added 0.4 ms is the rank-up matmul that MLA inserts; on bandwidth-bound decode it is essentially free, because the matrix cores are otherwise idle. The gap between rows two and three is roughly seven milliseconds per step, which translates to a 1.6x decode throughput at 128k context.

The fourth row is what happens when MLA composes with a sliding window plus the [four-token attention sink](/blog/2026-05-02-attention-sinks-the-four-kv-tokens-local-long-context-cannot-evict). The cache is small enough that even keeping a 64k window plus the sink stays well inside the bandwidth budget, and the wall time drops below the active-weight floor. That last point is the structural change. With GQA and a 32k window, the engine still spends most of decode reading the cache. With MLA and the same window, the engine starts spending most of decode reading the active weights, which is the regime the matrix cores and the cooperative-matrix path are tuned for.

## The price MLA actually charges

MLA is not free. Three places it costs something real are worth naming clearly.

The first is the per-step attention math. MLA reconstructs the per-head K and V from the latent at every attention step, which is a small dense GEMM with shape `[heads * head_dim, d_c]`. For a 32-head Qwen-class shape that is a `[4096, 512]` matmul per layer per token. At 48 layers and one token per step, that is roughly two hundred million extra FLOPs per decode step. On a memory-bound decode where the matrix cores sit at single-digit utilization, this is invisible. On a heavily batched prefill where the cores are saturated, it shows up as real wall time.

The second is the kernel rewrite. The flash-attention path most local engines run today, including ours, is built around the assumption that the cache holds per-head keys and values laid out contiguously. The MLA path holds a single shared latent. That is not a parameter to a kernel; it is a different kernel. The [llama.cpp MLA optimization PR](https://github.com/ggml-org/llama.cpp/pull/11446) lays out the tradeoff in code: caching the compressed latent uses less memory bandwidth than caching full K and V, but it requires extra compute on every attention step to project the latent back to per-head keys and values. The PR landed for the CPU and CUDA backends; a Vulkan port that keeps cooperative-matrix throughput on the rank-up matmul is the part that has not shipped on RDNA4.

The third is that MLA requires retraining. There is no cheap weight-conversion path from a GQA checkpoint to an MLA checkpoint. The compression matrix `W_DKV` and the up-projection matrices for keys and values are learned, and they have no analytic equivalent to copy from the original K and V projections. Adopting MLA on a Qwen-class model means waiting for a Qwen-class model trained with MLA. The [DeepSeek-V3 reference implementation](https://github.com/deepseek-ai/DeepSeek-V3) is the closest thing to a public template, and even there the MLA blocks are wired into the rest of the architecture in ways that are not trivial to lift out.

## Why this still matters for the local roadmap

The open-weight family is moving in this direction whether the local engines are ready or not. Sebastian Raschka's [side-by-side review of MLA](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison) walks through MLA along the same axis as GQA and concludes that the savings are large enough that future open-weight long-context models will adopt some variant of low-rank attention, even if the precise shape differs from DeepSeek-V3's. The next round of Qwen, Llama, and Mistral checkpoints at 128k or longer native context have a strong incentive to follow.

For an engine pinned to a 32 GB RDNA4 card, that means the long-context decode wall moves from "manage a giant resident KV cache" to "do a small dense rank-up matmul before each attention block." The first is a memory-pressure problem solved by eviction. The second is a kernel problem solved by compute. The matrix cores that sit out the decode loop today have a workload waiting for them in the MLA path, and the bandwidth budget that GQA stretched to its limit gets a final lever pulled.

The invariants on the local long-context decode are stacking up. The four sink tokens stay resident. The recent window stays resident. The KV cache shape moves from grouped to latent. The active-weight footprint stays at two gigabytes per step on a 4-bit MoE. Past that, every decode step on a 32 GB card is a small dense matmul plus a small attention plus a small cache read. That is the shape of local long-context inference after GQA, and it is the shape worth building toward.
