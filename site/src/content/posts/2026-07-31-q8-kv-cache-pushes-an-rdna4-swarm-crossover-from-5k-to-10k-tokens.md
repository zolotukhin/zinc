---
title: "A Q8 KV cache pushes an RDNA4 swarm's crossover from 5k to 10k tokens"
seoTitle: "KV Cache Quantization on RDNA4: Moving the Weight-vs-KV Crossover for a Local Agent Swarm"
date: "2026-07-31"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - kv-cache
  - quantization
  - q8
  - fp8
  - decode
  - agents
  - local-llm
  - llm-inference
keywords:
  - KV cache quantization RX 9070 XT
  - Q8 KV cache local agent swarm
  - RDNA4 decode memory bandwidth KV cache
  - weight stream versus KV cache crossover
  - FP8 KV cache Qwen3 RDNA4
  - llama.cpp cache-type-k q8_0 flash attention
  - KIVI per-channel key per-token value quantization
  - single card agent swarm context budget
excerpt: "Eight agents on one 16 GB RX 9070 XT share a single 5.5 GB weight stream per decode step, so the swarm is fast only until each agent's private KV read grows past that shared cost. At FP16 that crossover lands near 5k tokens per agent; past it, batching mostly stops paying. Quantizing the KV cache to Q8 halves the per-token bytes and moves the crossover to about 10k, and on a 16 GB card it roughly doubles the context the swarm can hold at all. It is a storage-format change, not a kernel change."
seoDescription: "A batched decode step on an RX 9070 XT pays one shared weight stream, about 5.5 GB for Qwen3.5-9B at Q4_K_M, split across all eight agents, but each agent reads its own KV cache and that read does not amortize. Modeled from the measured decode step, FP16 KV traffic for eight agents overtakes the shared weight read at roughly 5k tokens per agent, and past that the swarm's 311 tokens-per-second ceiling sags toward the batch-1 rate. Halving the KV bytes with a Q8 cache moves the crossover to about 10k tokens, lifts the effective batch-8 ceiling at 8k context from 152 to 205 tokens per second, and on a 16 GB card doubles the per-agent context the swarm can hold before it runs out of VRAM. KIVI, vLLM's FP8 KV cache, and llama.cpp's cache-type flags all exploit the same lever. This post models the crossover on a single consumer card and argues the KV storage format is the last cheap knob after continuous batching."
faqs:
  - question: "Why does a local agent swarm slow down as each agent's context grows?"
    answer: "A batched decode step pays one shared weight stream, about 5.5 GB for Qwen3.5-9B at Q4_K_M on an RX 9070 XT, no matter how many agents produce a token that step. That cost amortizes across the batch. But each agent also reads its own KV cache every step, and that read is private and does not amortize. As contexts grow, the summed private KV traffic overtakes the shared weight read, and the swarm's throughput ceiling sags from 311 tokens per second toward the single-agent rate because the step is now dominated by cost that batching cannot share."
  - question: "How much does quantizing the KV cache to Q8 help on a single GPU?"
    answer: "Q8 stores each KV element in one byte instead of two, so it halves the per-token KV bytes and halves the slope of the private KV traffic. Modeled from the measured decode step, that moves the point where KV traffic overtakes the shared weight stream from about 5k tokens per agent to about 10k, lifts the effective batch-8 ceiling at 8k of context from roughly 152 to 205 tokens per second, and on a 16 GB card lets the eight-agent swarm reach about 19k tokens per agent before running out of VRAM instead of about 9k at FP16."
  - question: "Does KV cache quantization hurt output quality?"
    answer: "Q8, meaning 8-bit, KV is close to lossless in practice, and the K cache is more sensitive to quantization than the V cache, which is why llama.cpp lets you set them independently with cache-type-k and cache-type-v. Going more aggressive, to 4-bit or 2-bit, needs a smarter scheme: KIVI showed the key cache should be quantized per channel and the value cache per token because their outlier structure differs, and it holds quality at 2-bit that way. On a single card the safe default is Q8 for both, or Q8 keys and a lighter value cache if the model tolerates it."
  - question: "Is KV cache quantization a kernel change or a storage change?"
    answer: "It is mainly a storage-format change, but it needs the attention kernel to read the quantized format directly. The KV bytes in VRAM shrink, so the per-step KV read shrinks. The one requirement is flash attention: the fused attention kernel must dequantize on the fly inside the kernel, because dequantizing the whole cache to FP16 before each attention step would cost more bandwidth than it saves. With flash attention on, quantized KV is close to free, which is why llama.cpp gates cache-type quantization behind it."
draft: false
---

The single-card agent swarm this series keeps measuring has a fast number and a fine print. The fast number is 311 tokens per second: [eight agents on one RX 9070 XT](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) pulling 5.6 times the aggregate throughput of a single agent. The fine print is that the number is measured at short context, when each agent is holding a few hundred tokens of KV cache. It does not survive contact with a real coding session.

The reason is a split that this series already pulled apart. A decode step is memory bound, and its two big costs behave differently under batching. The model weights are [streamed once per step and shared](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/) across every agent in the batch, so eight agents split one 5.5 GB read. The KV cache is not shared. Each agent reads its own, every step, and that [private read does not amortize](/blog/2026-07-28-batched-swarm-kv-read-does-not-amortize/). So the whole economics of the swarm hinge on one question: at what context length does the private KV traffic grow past the shared weight read?

For Qwen3.5-9B at Q4_K_M with eight agents, the answer at FP16 is about 5,000 tokens each. Below that, the shared weights dominate the step and batching pays off. Above it, the KV reads dominate, and the swarm's ceiling starts sliding toward the throughput of a single agent. The good news is that the crossover is not a law of physics. It is set by how many bytes a KV token costs, and that is a number you can change. Quantizing the KV cache to Q8 halves it and moves the crossover to about 10,000 tokens, which on this workload is the difference between a swarm that stays fast through a real task and one that does not.

## The crossover is just a byte count

Start with the shared side, because it is fixed. Streaming Qwen3.5-9B's Q4_K_M weights is about 5.5 GB, and that read happens once per decode step regardless of batch size. Eight agents in the same step split it eight ways. That shared floor is the [entire reason a swarm beats eight sequential agents](/blog/2026-07-30-static-batching-drains-rdna4-swarm-throughput/).

Now the private side. At FP16, each token of KV cache for this model is about 0.14 MB per agent, summed across all the layers and both the key and value tensors. Every decode step, an agent reads its whole KV cache to attend over its context. So an agent at context C moves roughly 0.14 times C megabytes of KV, and the batch of eight moves eight times that, because none of it is shared. The weight read stays flat at 5.5 GB while the KV read climbs a straight line in C.

Set the two equal. Eight agents' FP16 KV traffic reaches 5.5 GB when 8 times 0.14 times C equals 5,632 MB, which is about 5,000 tokens per agent. That is the crossover. It is not a benchmark, it is arithmetic on a byte count, and it means the swarm's advantage has a shelf life measured in context length.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-31-rdna4-swarm-kv-quant-crossover.svg" alt="A crossover chart on a graphite background titled 'Quantizing the KV cache pushes the swarm's crossover from 5k to 10k tokens'. The main plot has KV context per agent from 0 to 32k tokens on the x-axis and gigabytes streamed per decode step for a batch of eight on the y-axis from 0 to 18. A grey dashed horizontal line sits at 5.5 GB, labelled shared weight stream paid once and split eight ways. An orange line for FP16 KV rises steeply from the origin, crossing the 5.5 GB line at about 5k tokens and running off the top of the chart past 16.5k. A teal line for Q8 KV rises at half the slope, crossing the 5.5 GB line at about 10k tokens, labelled half the bytes half the slope. Two dots mark the crossovers at 5k orange and 10k teal. A right-hand panel titled 'Effective batch-8 decode ceiling' shows paired bars of tokens per second at 4k, 8k, 16k and 32k of per-agent context. FP16 is 205 and 152 as solid orange bars at 4k and 8k then dashed OOM outlines at 16k and 32k. Q8 is 247, 205 and 152 as solid teal bars at 4k, 8k and 16k then a dashed OOM outline at 32k. A dashed line marks the 311 short-context ceiling." loading="lazy" />
  <figcaption>Modeled for Qwen3.5-9B Q4_K_M on one 16 GB RX 9070 XT using the measured decode step t(B,C) = 16.9 + B(1.1 + kC) ms. FP16 KV costs 0.14 MB per token per agent; Q8 halves it. The crossover is where the batched KV read equals the once-per-step 5.5 GB weight read. Q8 moves it from about 5k to 10k tokens and, on a 16 GB card, lets the swarm hold roughly twice the context before it runs out of VRAM.</figcaption>
</figure>

The chart is the whole argument. The orange FP16 line is steep enough that it clears the shared weight read by 5k tokens and leaves the chart entirely past 16.5k, where an agent's own KV read is larger than the full model weight stream. The teal Q8 line does the same work at half the slope, so it reaches the crossover twice as late. Everything to the left of a line's crossover is context where batching still pays; everything to the right is context where the swarm is mostly paying for private reads that no amount of batching can share.

## What the sagging ceiling costs in tokens per second

The crossover is where the loss begins, not where it ends. Fold the growing KV read back into the decode step and the throughput number moves with it. Modeled from the measured step, an eight-agent batch that hits 311 tokens per second at short context is down to about 205 by 4k tokens each, 152 by 8k, and 101 by 16k. The card has not changed. The step is just spending more of itself on reads that eight agents cannot split.

Q8 changes the byte count, so it changes every one of those numbers. Halving the KV bytes halves the KV portion of the step, and the effective ceiling recovers most of a batch size. The table is the same workload at FP16 and Q8, with the card's 16 GB budget included because it turns out to bind first.

| Per-agent context | 8-agent KV, FP16 | 8-agent KV, Q8 | Ceiling, FP16 KV | Ceiling, Q8 KV |
| --- | ---: | ---: | ---: | ---: |
| 4k | 4.4 GB | 2.2 GB | 205 tok/s | 247 tok/s |
| 8k | 8.8 GB | 4.4 GB | 152 tok/s | 205 tok/s |
| 16k | 17.5 GB, over budget | 8.8 GB | does not fit | 152 tok/s |
| 32k | 35 GB, over budget | 17.5 GB, over budget | does not fit | 101 tok/s |

Read the table twice. The first read is speed: at 8k of context, which is an ordinary coding turn once tool output is folded in, Q8 lifts the swarm from 152 to 205 tokens per second, worth about a third more throughput for a storage change. The second read is capacity, and it is the harder limit. With 5.5 GB of weights resident, a 16 GB card has roughly 10 GB left for KV across all eight agents. FP16 exhausts that near 9k tokens each, so an eight-agent FP16 swarm cannot reach 16k of context at all on this card. Q8 stretches the same budget to about 19k. Quantizing the KV is not only how the swarm stays fast, it is how the swarm fits.

## Why Q8 is close to free and where the sharp edges are

The obvious worry is quality. In practice Q8, meaning one byte per KV element instead of two, is close to lossless, and the standard local engines already ship it. In llama.cpp you set it with `--cache-type-k q8_0` and `--cache-type-v q8_0`, and the [community measurements](https://github.com/ggml-org/llama.cpp/discussions/20969) are consistent that Q8 KV is a negligible quality hit for a large VRAM saving. vLLM exposes the same idea as an [FP8 KV cache](https://docs.vllm.ai/en/latest/features/quantization/quantized_kvcache/), storing keys and values in an 8-bit float to halve the cache footprint on the serving path. This is the local-swarm framing of the same [FP8 KV cache](/blog/2026-05-19-fp8-kv-cache-is-the-next-decode-bandwidth-cut-rdna4-already-has-the-wmma-for/) bandwidth cut this blog argued for in May, now aimed at the batched case where it decides whether batching pays off.

The one real requirement is flash attention. Quantized KV only helps if the attention kernel reads the compressed format directly and dequantizes inside the fused kernel. If the engine instead expands the whole cache back to FP16 before each attention step, it moves the full-size bytes anyway and you have paid for nothing. That is why llama.cpp gates cache-type quantization behind flash attention, and it is why an RDNA4 engine wants the [wave32 flash-attention path](/blog/2026-05-11-the-wave32-commit-that-closes-rdna4-long-context-flash-attention-gap/) doing the dequant on chip rather than a separate pass over VRAM.

Push below 8-bit and it stops being free. The key and value caches do not quantize the same way, because their outlier structure is different. The [KIVI paper](https://arxiv.org/abs/2402.02750) from ICML 2024 studied the element distributions and found the key cache should be quantized per channel and the value cache per token, and with that split it held quality at 2-bit while cutting peak memory 2.6 times and enabling up to 4 times the batch size. That is the map for going further than Q8, but it is a real kernel and layout change, not a flag. For a single card the honest default is Q8 on both, or Q8 keys with a lighter value cache if the model tolerates it, which is exactly why the two knobs are separate in the first place.

## The storage format is the last cheap lever

Across the last two weeks the pattern has held: the biggest wins in a local swarm are bookkeeping, not arithmetic. [Continuous batching](/blog/2026-07-30-static-batching-drains-rdna4-swarm-throughput/) recovered the throughput a draining static batch threw away, and it was a scheduler change with no new kernels. KV quantization is the matching move on the memory side. It does not make the card compute faster. It shrinks the private read that batching cannot share, so the shared weight stream stays the dominant cost for twice as long and the swarm holds its ceiling deeper into a real task.

The two levers stack cleanly, and the order is worth stating. Refill the batch so the shared weight stream stays fully amortized, then quantize the KV so the private reads do not overtake it. On a 16 GB RDNA4 card running eight agents, that combination is the difference between a swarm that is fast for the first few hundred tokens and one that is still fast at ten thousand. After that, the levers get expensive: sub-8-bit KV, paged cache layouts, and eventually a bigger card. Q8 is the last one that costs nothing but a flag.
