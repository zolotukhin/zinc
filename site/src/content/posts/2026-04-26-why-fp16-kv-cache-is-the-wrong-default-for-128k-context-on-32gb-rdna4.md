---
title: "Why FP16 KV cache is the wrong default for 128k context on a 32 GB RDNA4 card"
date: "2026-04-26"
tags:
  - zinc
  - rdna4
  - amd
  - kv-cache
  - quantization
  - long-context
  - vulkan
  - prefill
  - llm-inference
  - turboquant
keywords:
  - KV cache quantization RDNA4
  - 128k context Qwen3.5-35B local inference
  - Radeon AI PRO R9700 32 GB VRAM budget
  - FP16 KV cache memory cost
  - Q8 Q4 KV cache llama.cpp
  - TurboQuant KV cache compression
  - per-token KV bytes math
  - GQA n_kv_heads head_dim KV size
  - Vulkan KV cache layout RDNA4
  - long context local LLM consumer GPU
excerpt: "On a 32 GB Radeon AI PRO R9700 the model weights are not the binding constraint at long context. The KV cache is. At 128k tokens on Qwen3.5-35B-A3B a default FP16 KV cache is 15.4 GiB, which together with the Q4_K_M weights walks straight off the 32 GB ceiling. Quantizing the KV cache is not a polish step. It is the only way the long-context local prompt fits at all."
---

The first time a long prompt runs out of memory on a 32 GB Radeon AI PRO R9700, the instinct is to blame the model. Twenty gigabytes of weights, eight gigabytes of "miscellaneous", and the OS holding the rest, the story goes. The story is wrong. On Qwen3.5-35B-A3B at Q4_K_M the weights are 20.0 GiB, the working memory the engine uses for projections and reductions is well under one gigabyte, and the thing that actually moves with the prompt length is the KV cache. At 128k tokens it is fifteen and a half gigabytes on its own, and on a 32 GB card that is the difference between a useful long prompt and a hard out-of-memory.

This is the constraint that bites every local engine sooner than expected. The weights are static. The KV cache scales linearly with tokens, and on a modern transformer with a non-trivial number of attention heads it is the second-largest tenant in VRAM by the time the prompt is meaningful. On a hosted inference cluster with 80 GB H100s and tensor parallelism, the asymmetry is mostly a budgeting problem. On a single consumer-grade RDNA4 card it is the design problem.

This post is the argument for why FP16 is the wrong default for the KV cache on a 32 GB RDNA4 card, what the per-token arithmetic actually looks like for the models ZINC ships against, and where the floor is once the obvious wins are taken.

## What a KV cache actually costs per token

The size of one token's worth of KV cache is set by the architecture, not the prompt. Multiply two for the K and V tensors, by the number of layers, by the number of KV heads, by the head dimension, by the bytes per element. That product gives the per-token bytes; multiplied by the prompt length it is the total KV bytes that have to live in VRAM for the duration of the request.

For Qwen3.5-35B-A3B the relevant numbers are 60 layers, 4 KV heads after grouped-query attention, and a head dimension of 128. At FP16 that is `2 × 60 × 4 × 128 × 2 = 122,880` bytes per token, which is exactly 120 KiB. At 8k context that is under one gigabyte and nobody notices. At 32k it is 3.84 GiB, which still fits comfortably on the [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) but stops being free. At 128k it is 15.36 GiB, and the math becomes the post.

The Gemma 4 31B family is a different shape but the same underlying problem. Yesterday's [asymmetric GQA writeup](/blog/2026-04-24-the-single-push-constant-blocking-gemma-4-prefill-on-rdna4) noted that Gemma 4 interleaves 50 sliding-window-attention layers at a 1024-token window with 10 full-attention layers at 16 KV heads of dimension 128. The SWA layers are bounded by the window: their KV footprint is constant at roughly 0.4 GiB regardless of prompt length. The 10 full-attention layers carry the rest, and they scale linearly. At 128k tokens the full-attention KV alone runs to 10 GiB at FP16, which is closer to the Qwen number than people who hear "sliding window" expect.

The pattern repeats across every modern model worth running locally. GQA shrinks the per-token cost relative to a multi-head architecture, but the linear factor in tokens is what matters once the prompt grows.

## The 32 GB picture

The chart below is the same prompt and the same model run six different ways on a 32 GB R9700. The top three bars hold KV quantization at FP16 and walk context from 8k to 128k. The bottom three bars hold context at 128k and walk KV quantization from INT8 down to a 3-bit shape in the spirit of Google's [TurboQuant](https://github.com/ggml-org/llama.cpp/discussions/20969) work.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-04-26-rdna4-32gb-kv-cache-memory-budget.svg" alt="Six stacked horizontal bars sharing a 32 GB total VRAM scale on a Radeon AI PRO R9700 with Qwen3.5-35B-A3B Q4_K_M weights. Each bar shows model weights, KV cache, working memory, and any spill past the 32 GB ceiling. The top three bars compare FP16 KV at 8k, 32k, and 128k context, with the 128k FP16 bar visibly crossing the 32 GB ceiling. The bottom three bars compare INT8, Q4, and 3-bit KV quantization at 128k context, all of which fit under the 32 GB line." loading="lazy" />
  <figcaption>The 32 GB ceiling on the R9700 is set by the card. The KV cache is the only segment in this picture that grows with the prompt. The lower three bars show how cheaply that growth can be tamed.</figcaption>
</figure>

The thing to notice is which segment moves and which segments do not. Weights stay at 20 GiB across every bar; that 20 GiB is set the moment a Q4_K_M GGUF is loaded. The working-memory segment, the small green slab on the right of each bar, is dominated by the largest activation tensor the engine has to keep live during a layer and is comfortably under one gigabyte for prefill on this model. Everything that varies between bars is the KV cache.

That is the part of the picture that is easy to misread. A long-context prompt on a 32 GB card is not a model-size problem and not an activation-memory problem. It is a KV-bytes-per-token problem. The right place to spend engineering effort is the KV cache representation.

## Why FP16 is the wrong default

FP16 is the default because it is what the reference implementations write to the KV cache and what the attention shaders read out of it. There is nothing magical about the precision; it is a historical artifact of when the KV cache was small enough that no one had to think about its size. On a 7B model at 2k context with a multi-head attention pattern, the KV cache is 0.5 GiB and FP16 is fine. On a 35B model at 128k context with GQA, FP16 is the load-bearing default that everyone keeps because nobody on a 24 GB or 32 GB consumer card has been able to run that prompt long enough to feel the cost.

The accuracy story has also moved. INT8 KV cache quantization is now well understood; the [llama.cpp 4-bit KV cache discussion](https://github.com/ggml-org/llama.cpp/discussions/5932) reports that Q8_0 K and Q8_0 V are essentially indistinguishable from FP16 in perplexity for the model families llama.cpp ships against, and Q4_0 K with FP16 V (or Q8_0 V) is within noise on most workloads. The corresponding [tracking issue for 4-bit KV in llama.cpp](https://github.com/ggml-org/llama.cpp/issues/6863) walks through the perplexity deltas in detail and points out the well-known asymmetry: K vectors quantize cleanly, V vectors are more fragile, and the right answer is usually a higher precision on V than on K.

The newer point is the 3-bit floor. Google's TurboQuant work, recently [ported to HIP and ROCm on gfx1100](https://github.com/ggml-org/llama.cpp/discussions/21526) and being tracked for the wider llama.cpp tree, claims under-three-bits per KV element with negligible quality loss on long-context workloads. Whether that floor holds across every model family is something every local engine should evaluate on its own evals before flipping the default. What the work establishes for sure is that the 32 GB picture above is not a one-time fix. The KV cache is going to keep getting smaller as the quantization research keeps moving.

## What it costs to read a quantized KV cache on RDNA4

The reason FP16 KV is sticky in a Vulkan backend is that quantized KV requires a dequant step inside flash attention or, better, an attention shader that computes its dot products directly against quantized K and V. Both are real engineering work. ZINC's prefill path, like llama.cpp's Vulkan backend, currently dispatches a [`flash_attn_batched.comp`](https://github.com/ggml-org/llama.cpp/tree/master/ggml/src/ggml-vulkan/vulkan-shaders) that reads K and V as FP16. The Q8_1 activation path described in the [Q8_1 prefill writeup](/blog/2026-04-19-why-q8-1-activations-are-the-next-rdna4-prefill-unlock) is exactly the kind of shader work that has to happen on the KV side too, just for a different tensor.

There are two viable shapes. The first is the offline approach: dequant the KV slab into a scratch FP16 buffer once per layer, then run today's attention shader unchanged. The arithmetic is fine in isolation. On RDNA4 with a peak of roughly 644 GB/s of memory bandwidth, walking the 7.68 GiB INT8 KV slab once and writing 15.36 GiB of FP16 back is 23 GiB of memory traffic, or about 36 ms per prompt at peak. The trouble is what comes next. Attention then reads the FP16 scratch buffer at the same per-token bandwidth cost as before, so the smaller KV bought no bandwidth win on the inner loop. The only thing the offline path saves is steady-state VRAM, and only if the engine can avoid materializing both copies at once.

The second is the inline approach: the attention shader reads INT8 or INT4 K and V directly, dequants in the inner loop, and the FP16 round trip never happens. This is the shape that mul_mmq took for projections and that any serious local engine has to take for attention. The kernel is harder to write but it is the only one that lets the smaller KV cache also be the faster one.

## What a per-token quantization budget looks like

Per-token KV bytes are the right unit for thinking about long-context tradeoffs because the quantity scales linearly with prompt length and shows up everywhere a model card might lie about its real cost. The table below is the same Qwen3.5-35B-A3B architecture under five KV representations, spelled out explicitly.

| KV format | Bytes per token | KV at 32k | KV at 128k | Notes |
| --- | ---: | ---: | ---: | --- |
| FP16 K, FP16 V | 122,880 | 3.84 GiB | 15.36 GiB | The current default. OOM at 128k on a 32 GB card. |
| FP16 K, INT8 V | 92,160 | 2.88 GiB | 11.52 GiB | Cheapest precision-first move; V tolerates Q8 well. |
| INT8 K, INT8 V | 61,440 | 1.92 GiB | 7.68 GiB | The right floor for a portable engine in 2026. |
| Q4 K, INT8 V | 46,080 | 1.44 GiB | 5.76 GiB | Q4 K is the asymmetric pattern most engines land on. |
| TurboQuant-style 3-bit | ~23,040 | 0.72 GiB | 2.88 GiB | Research-grade. Worth tracking, not yet a default. |

The middle three rows are the ones that matter for engineering work shipping this quarter. Going from FP16 KV to INT8 KV halves the KV bytes per token outright. Going one step further to Q4 K with INT8 V cuts it again with little measurable accuracy loss on the workloads the discussions above document. The 3-bit row is what the floor looks like once research-grade compression lands as a production option.

The other thing the table makes obvious is that the bytes-per-token number does not need to be small in absolute terms to matter. At 128k context, 60 KiB per token versus 30 KiB per token is 4 GiB of VRAM either way. On a 32 GB card with a 20 GB model, that 4 GiB is the difference between a usable long prompt and a card that thrashes.

## What this means for the engine

On the engine side, three things follow. The first is that the default should change. Any local engine targeting a 32 GB consumer card and a 128k context window should plan around INT8 KV as the floor and Q4 K with INT8 V as the realistic shipping choice. FP16 KV remains useful as a debug fallback, but it is no longer the right default for the user-visible prompt length the engine wants to advertise.

The second is that the attention kernel has to change with it. Reading quantized K and V inside flash attention is the price of admission. Doing it well on RDNA4 is the same shape of problem as the [32-column DMMV port](/blog/2026-04-22-why-rdna4-prefill-wants-a-32-column-dmmv-before-a-gemm), and it pays for itself in two places: it shrinks the KV bytes the shader has to touch per attention computation, which helps prefill, and it shrinks the per-token bandwidth at decode, which helps the steady-state throughput the user actually feels.

The third is that the right time to do this is now, not after the rest of the prefill optimization arc lands. Every kernel optimization that lands on top of an FP16 KV is going to look like a smaller win once the KV is half the size. The denominator changes. Choosing the KV representation first is the way to make the rest of the optimization story honest.

The 32 GB ceiling on the R9700 is a hard line. The KV cache is the only part of the picture that is easy to move. The work to move it is shaderwise harder than the work that has filled the prior posts in this arc, but it is the work that decides whether 128k context on a single consumer-grade RDNA4 card is a feature or a footnote.
