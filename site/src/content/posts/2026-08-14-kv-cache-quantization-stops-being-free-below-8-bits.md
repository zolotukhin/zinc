---
title: "The KV cache lever the swarm keeps pulling stops being free below 8 bits"
seoTitle: "Sub-8-Bit KV Cache Quantization Is Not a Free Lever for an RDNA4 Agent Swarm"
date: "2026-08-14"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - kv-cache
  - quantization
  - q8-kv-cache
  - accuracy
  - agents
  - local-llm
  - llm-inference
keywords:
  - KV cache quantization accuracy cost
  - q8 KV cache near lossless RDNA4
  - sub 4-bit KV cache perplexity degradation
  - KIVI per-channel key per-token value quantization
  - KVQuant 3-bit KV cache perplexity
  - key cache channel outliers quantization
  - low bit KV cache local swarm RX 9070 XT
  - int4 int2 KV cache quality cliff
excerpt: "Four posts in this series reached for the same lever: quantize the KV cache to q8, halve every byte, double the headroom. That lever is real, and at 8 bits it is close to free. The trouble starts when the swarm wants more and reaches below 8 bits, because the KV cache is not one number to round off. Keys carry large per-channel outliers and values do not, so a single uniform 4-bit or 2-bit knob falls off a quality cliff that the papers only climb back up with per-channel key quantization and explicit outlier isolation. KIVI keeps 2-bit quality only by quantizing keys per-channel and values per-token; KVQuant holds 3-bit to under 0.1 perplexity loss only with pre-RoPE per-channel keys and dense-and-sparse outlier handling. On a single 16 GB RX 9070 XT that means the free win ends at 8 bits, and everything below it is an engineering project with a per-model accuracy budget, not a config flag."
seoDescription: "This series repeatedly used q8 KV cache as a memory lever to fit more agents and longer context on one 16 GB RX 9070 XT running Qwen3.5-9B. At 8 bits the accuracy cost is small and the win is real. Below 8 bits it is not a scalar knob: the key cache has large per-channel outlier structure while the value cache does not, so uniform int4 or int2 quantization degrades sharply. KIVI (ICML 2024) keeps near-baseline quality at 2 bits only by quantizing keys per-channel and values per-token; KVQuant (NeurIPS 2024) reaches under 0.1 perplexity degradation at 3 bits only with pre-RoPE per-channel key quantization and per-vector dense-and-sparse outlier isolation. For a single-card local swarm, the practical rule is: take the near-free 8-bit win, and treat sub-8-bit KV quantization as kernel work with a measured accuracy budget rather than a free lever."
faqs:
  - question: "Is q8 KV cache actually free, or does it cost accuracy?"
    answer: "At 8 bits the accuracy cost is small enough that most deployments treat it as free. Storing keys and values in an 8-bit format instead of fp16 gives each element 256 levels, which is enough dynamic range to hold the attention key and value distributions with little visible quality loss on standard perplexity and task benchmarks. That is why 8-bit and fp8 KV caches ship as a supported option in production engines like vLLM. The caveat is that free applies to the memory and bandwidth, not to correctness guarantees: it is still a lossy transform and should be measured per model, but for a 9B-class model the 8-bit degradation is typically in the noise. The problems in this post begin below 8 bits, not at 8."
  - question: "Why can't I just keep halving the KV cache bit width to fit more agents?"
    answer: "Because the KV cache is not a single distribution you can uniformly round. Attention keys carry a handful of channels with very large magnitudes, the outlier channels, while values are far more uniform. A single per-token quantizer that works fine for values will size its range around a key's outlier channel and crush every other channel in that token into a couple of levels, which is where quality collapses. That is the specific finding behind KIVI: the key cache should be quantized per-channel and the value cache per-token. Halving bit width past 8 without changing the quantization axis is how you walk off the cliff, not a smooth trade."
  - question: "What do the papers actually measure at low bit widths?"
    answer: "KIVI, published at ICML 2024, shows that a tuning-free 2-bit KV cache keeps almost the same quality as fp16, but only with its asymmetric scheme of per-channel keys and per-token values, and it reports about 2.6 times less peak memory and up to 4 times larger batch size as a result. KVQuant, at NeurIPS 2024, reaches under 0.1 perplexity degradation with 3-bit quantization on Wikitext-2 and C4, but only by combining pre-RoPE per-channel key quantization, non-uniform per-layer datatypes, and per-vector dense-and-sparse isolation of outliers. The headline low-bit numbers are real; they are attached to real machinery, not to a uniform rounding flag."
  - question: "What should a single-card local swarm actually do about KV precision?"
    answer: "Take the 8-bit win because it is close to free and it is already the lever earlier posts in this series used to double preemption headroom and push the fit crossover from 5k to 10k tokens. Below 8 bits, stop treating precision as a scalar knob and treat it as a kernel project: quantize keys per-channel and values per-token, isolate the outlier channels, and measure the accuracy on your own model and workload before shipping it. On a 16 GB RX 9070 XT the temptation to keep shrinking KV to admit more agents is constant, and the honest version is that 4-bit and below buy memory only if you are willing to pay for the quantization scheme that keeps them accurate."
draft: false
---

Four different posts in this series have reached for the same move when the card ran short. When [preemption headroom capped out at about two paused turns](/blog/2026-08-12-free-rdna4-preemption-ends-where-paused-kv-fills-the-card/), the fix was a q8 KV cache. When the swarm's [fit crossover sat at 5k tokens](/blog/2026-07-31-q8-kv-cache-pushes-an-rdna4-swarm-crossover-from-5k-to-10k-tokens/), q8 pushed it to 10k. Every time the 16 GB RX 9070 XT filled up, the same lever appeared: store the keys and values in 8 bits instead of 16, halve the KV footprint, and get the headroom back.

That lever is real, and at 8 bits it is close to free. Eight bits give each key and value element 256 levels, which is enough to hold the attention distributions with quality loss that mostly disappears into benchmark noise, which is why an [8-bit or fp8 KV cache ships as a supported option](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache.html) in serving stacks. The problem is what happens next. A single-card swarm is always short on VRAM, so the obvious thought is to keep pulling: if 8 bits doubled the room, surely 4 bits doubles it again, and 2 bits again after that.

It does not work that way, and the reason is worth understanding before you ship it. The KV cache is not one number you can round off uniformly. Below 8 bits it splits into two problems with two different answers, and a uniform low-bit knob walks straight off a quality cliff that the good papers only climb back up with real work.

## The cache is two distributions, not one

Quantizing a tensor means choosing a range and slicing it into levels. Do it well when the values are evenly spread; do it badly when a few values are enormous and the rest are tiny, because the range stretches to cover the outliers and everything else collapses into a handful of indistinguishable levels. So the question for the KV cache is simple: how are its numbers distributed?

The answer, worked out carefully in the [KIVI paper](https://arxiv.org/abs/2402.02750) from ICML 2024, is that keys and values are distributed differently, and that difference is the whole game. The key cache has a small number of channels, the same feature dimensions across every token, that carry very large magnitudes. These are the outlier channels, and they dominate the range. The value cache does not have this structure; its magnitudes are far more uniform across channels.

That single observation dictates the quantization axis. If you quantize a token's key vector as one group, per-token, the outlier channel sets the scale and flattens every other channel in that token. But if you quantize along the channel dimension instead, grouping each channel across tokens, the outlier channel gets its own scale and stops poisoning its neighbors. KIVI's finding, stated plainly, is that the key cache should be quantized per-channel and the value cache per-token. Two tensors, two axes, because they have two shapes of distribution.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-08-14-kv-cache-key-per-channel-value-per-token-quantization.svg" alt="Two grids side by side on a dark slate background, each a small matrix with tokens running down the rows and channels running across the columns. The left grid is labelled key cache and has two full columns shaded bright amber to mark outlier channels that carry large magnitudes across every token, while the rest of the cells are a faint uniform blue. A vertical bracket runs down those columns labelled quantize per-channel, so each channel including the outliers gets its own scale. A red crossed-out horizontal bracket across one row is labelled per-token fails: the outlier channel sets the range and crushes the rest. The right grid is labelled value cache and has no bright columns; every cell is a similar faint blue with no channel structure. A horizontal bracket across one row is labelled quantize per-token, which works because the values are uniform across channels. A footer reads: keys have per-channel outliers so they must be grouped along channels; values are uniform so per-token grouping is fine. This asymmetry is why one uniform low-bit knob falls off a quality cliff." loading="lazy" />
  <figcaption>The key cache (left) has a few channels with large magnitudes, the amber columns, so it must be quantized per-channel to give those outliers their own scale. The value cache (right) has no such channel structure, so per-token quantization is fine. A single uniform per-token knob applied to both crushes the non-outlier key channels, which is the cliff. Structure is illustrative; the per-channel-key, per-token-value split is KIVI's measured result.</figcaption>
</figure>

What the picture is meant to fix in your head is that there is no single right way to quantize the KV cache. The correct axis depends on which tensor you are looking at, and a scheme that ignores the difference is choosing to be wrong about one of them.

## Why 8 bits hides the problem and 4 bits exposes it

At 8 bits you can be sloppy about all of this and get away with it. With 256 levels there is enough resolution that even a range stretched by an outlier channel still leaves the small channels distinguishable. The per-token-versus-per-channel decision barely matters because the quantizer is not starved for levels. This is exactly why the earlier posts in this series could treat q8 as a free lever and never mention accuracy: at that bit width, for a 9B-class model, it very nearly is.

Drop to 4 bits and you have 16 levels. Now the outlier channel eats most of them, and a per-token key quantizer leaves the ordinary channels with two or three levels to describe real variation. Drop to 2 bits and you have four levels total; a mis-chosen axis leaves the informative channels with essentially one. The degradation is not gradual and polite. It is a cliff, and where you fall off it depends entirely on whether the quantization scheme respects the key outliers.

This is the part the free-lever framing quietly skips. The memory math is linear: 8 bits is half of 16, 4 bits is a quarter, 2 bits is an eighth. The accuracy is not linear at all. It is roughly flat down to 8, starts bending at 4, and collapses below that unless you have done something specific to hold it up.

## What the low-bit numbers actually cost

The good news is that near-baseline quality at low bit widths is achievable. The honest news is that the papers reaching it are not flipping a bit-width flag, they are building machinery. KIVI keeps almost the same quality as fp16 at 2 bits, tuning-free, and reports about 2.6 times less peak memory and up to 4 times larger batch size, but that result is inseparable from its asymmetric scheme: per-channel keys, per-token values, with a small full-precision residual for the most recent tokens. Take away the per-channel key handling and the 2-bit number falls apart.

[KVQuant](https://arxiv.org/abs/2401.18079), from NeurIPS 2024, pushes the frontier further and is even clearer about the cost. It reports under 0.1 perplexity degradation at 3 bits on Wikitext-2 and C4, which is remarkable, but it gets there by stacking four separate techniques: per-channel key quantization, quantizing keys before the rotary positional embedding is applied so RoPE does not smear the channel structure, non-uniform per-layer datatypes fit to the actual distributions, and per-vector dense-and-sparse quantization that pulls the outliers out and stores them separately. That is four ideas in a trench coat, and each one is there to defend the same cliff.

The table lines up the memory the swarm is chasing against the accuracy regime the papers actually report, using the per-token KV size this series has used throughout: fp16 KV for Qwen3.5-9B is about 160 KiB per token, or 0.94 GiB for a 6k-token turn.

| KV precision | KV per token | 6k-token turn | Accuracy regime |
| --- | ---: | ---: | --- |
| fp16 | 160 KiB | 0.94 GiB | Baseline |
| 8-bit | 80 KiB | 0.47 GiB | Near-lossless, ships in production engines |
| 4-bit | 40 KiB | 0.23 GiB | Small loss with per-channel grouping; needs care |
| 3-bit | 30 KiB | 0.18 GiB | Under 0.1 ppl loss (KVQuant), with pre-RoPE per-channel keys and outlier isolation |
| 2-bit | 20 KiB | 0.12 GiB | Near-baseline (KIVI), only with per-channel keys and per-token values |

The table says the two things that matter together. The left columns are the memory the swarm wants, and they keep halving, which is exactly why a card-starved swarm keeps eyeing the next row down. The right column is the price of admission, and it stops being blank at the same row where the memory gets interesting. Every row below 8 bits has a condition attached, and the condition is always some version of respect the key outliers. The figures for 4-bit, 3-bit and 2-bit exclude the small overhead of the group scales and the sparse outlier storage, which claws back a few percent of the saving in exchange for the accuracy.

## The RDNA4 kernel angle, since that is where this series lives

None of the per-channel and outlier machinery is free to run, and on a single card the cost lands in the decode kernel. Per-channel key quantization means the scales run along a different axis than the per-token values, so the flash-attention kernel has to dequantize keys and values with two different layouts inside the same inner loop. Pulling outliers into a sparse side channel means the kernel carries a second, irregular read path alongside the dense one. This is the same tension an [earlier post hit with fp8 KV bandwidth](/blog/2026-05-19-fp8-kv-cache-is-the-next-decode-bandwidth-cut-rdna4-already-has-the-wmma-for/): the format that saves the bytes is only a win if the kernel can consume it without giving the time back.

At 8 bits this is manageable, which is another reason 8 bits is the sweet spot for a local engine. A symmetric per-token int8 cache dequantizes with one scale per token and folds cleanly into the existing wave32 flash-attention path this series [committed to for long context](/blog/2026-05-11-the-wave32-commit-that-closes-rdna4-long-context-flash-attention-gap/). The asymmetric per-channel schemes that keep 2-bit and 3-bit accurate do not fold in cleanly; they are a real kernel rewrite with a real occupancy cost, and that cost competes for the same VGPRs and issue slots every other decode optimization in this series has been fighting over.

## What to actually reach for

The rule that falls out of this is narrower than take the biggest quantization you can. It is take the free 8-bit win, and price everything below it. Eight bits is the lever the swarm should pull without hesitation, and it is the one the [q8 crossover post](/blog/2026-07-31-q8-kv-cache-pushes-an-rdna4-swarm-crossover-from-5k-to-10k-tokens/) and the [preemption ceiling post](/blog/2026-08-12-free-rdna4-preemption-ends-where-paused-kv-fills-the-card/) already leaned on, because at that width the accuracy cost is small, the kernel stays simple, and the memory really does halve. That win is banked.

Below 8 bits the framing has to change from lever to project. Four bits and lower buy more room only if you are willing to quantize keys per-channel, isolate the outlier channels, and measure the perplexity and task accuracy on your own model before trusting it, because the cliff is real and its exact location moves with the model. The engines that reach 2-bit and 3-bit without visible loss are not being clever with a flag; they are paying for it with per-channel scales, pre-RoPE handling, and sparse outlier stores, and that bill comes due in the decode kernel on a card that is already tight on registers.

The one-line version is that the KV cache looks like a dial you can keep turning, and it is not. It is two distributions wearing one name, and the moment you turn past 8 bits, you stop rounding a number and start doing engineering.
