---
title: "The single push constant blocking Gemma 4 prefill on RDNA4"
date: "2026-04-24"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - gemma
  - gemma4
  - prefill
  - flash-attention
  - gqa
  - llm-inference
  - gpu-kernels
keywords:
  - Gemma 4 31B prefill
  - RDNA4 Gemma inference
  - asymmetric GQA Vulkan
  - q_head_dim kv_head_dim flash attention
  - flash_attn_batched push constant
  - sliding window attention RDNA4
  - gemma3 5:1 SWA layer ratio
  - llama.cpp Gemma Vulkan
  - Radeon AI PRO R9700 Gemma
  - batched prefill grouped query attention
  - ZINC Gemma port
  - prefill bandwidth ceiling consumer AMD
excerpt: "Qwen3-8B prefill on ZINC's RDNA4 batched path runs at 187 tok/s. The same engine on Gemma 4 31B runs at 4.97 tok/s, because Gemma's batched-prefill gate is closed. Five of the six things blocking it open with small, mechanical fixes. The sixth is a single head_dim push constant on flash_attn_batched that quietly assumes Q and KV use the same dimension. Gemma 4's full-attention layers do not. Here is why one push constant became the bottleneck on a 31B model and what the right shape of the fix looks like."
---

The same `R9700` that runs Qwen3-8B prefill at **187 tok/s** through ZINC's batched path runs Gemma 4 31B at **4.97 tok/s** through the per-token path. That is not a model-size story. The bandwidth ceiling for a 31B model on a 576 GB/s card sits well above 30 tok/s. The reason Gemma is on the per-token path is that the batched-prefill gate is closed against `cfg.architecture == .gemma`, and the reason the gate is closed is that opening it produces structurally wrong output until six different things are fixed.

Five of those six things are mechanical and undramatic. They are the kind of port work that anyone who has shipped a multi-architecture inference kernel will recognize: an embedding pre-scale, two extra norm dispatches, per-layer head dim plumbing, a dual RoPE frequency buffer, and a `window_size` push constant on the flash-attention shader. None of them takes more than a day. The sixth one is the one that surprised us, and it is the subject of this post.

The sixth delta is that Gemma 4's full-attention layers use one head dimension for Q and a different head dimension for K and V. The `flash_attn_batched.comp` shader that ZINC ships today has exactly one `head_dim` push constant, and it parameterizes both the Q-side indexing math and the KV-side indexing math from the same value. Whatever number you put there is wrong on one of the two sides. The fix is fifteen lines of shader and three fields on the dispatch call. The interesting part is why one push constant looked correct for two years until a model showed up that violated the unstated assumption.

![AMD Radeon AI PRO R9700, the 32 GB RDNA4 workstation card the Gemma 4 31B numbers in this post are measured on.](/blog/gpu_2.jpg)

## Why batched prefill is the only lever that moves Gemma

The Gemma 4 31B per-token prefill profile on a 49-token prompt lands at 9.85 seconds of GPU time, of which roughly 6.9 seconds is spent on the seven Q4_K projections per layer. That is a per-token weight read of about **27 GB**. Multiplied by 49 tokens, the per-token path reads roughly **1.3 TB** of weights from VRAM to ingest a prompt that fits in a tweet.

The bandwidth floor on the same card with no other costs is `1.3 TB / 576 GB/s ≈ 2.3 s`, so the per-token path is running at roughly five times the floor. Most of that overhead is unavoidable in the per-token shape, because every projection re-reads the full weight tile once per token regardless of how aggressive the inner kernel is. The only structural fix is the same one that took Qwen3-8B from 72 tok/s to 187 tok/s on the same hardware: read each weight row once per prompt, multiply it against N activation columns, accumulate N partial sums.

That move was the subject of [yesterday's post on column-batched DMMV variants](/blog/2026-04-23-vulkan-specialization-constants-unlock-rdna4-dmmv-variants), and the new pipeline cache it enables is what every subsequent architecture port plugs into. For Qwen-style dense LLaMA layers, the plumbing was already in place. For Gemma it is not. The per-token path stays as a fallback because it is correct on every architecture, but it cannot hit the numbers that a 31B model on consumer AMD has to hit to be useful.

## The five deltas that look bigger than they are

Before the head-dim wrinkle, the obvious blockers fell out cleanly. Listing them is the only place this post uses bullet structure, because the point is precisely that none of them was the surprise.

The embedding pre-scale by `sqrt(hidden_dim)` is a single line in the pre-dequant loop. The `post_attention_norm` and `post_ffn_norm` calls between residual additions are the same two extra `dispatchRmsNorm` calls per layer that the per-token path already runs, ported to the batched body. The per-layer head-dim variance between full-attention and sliding-window layers is a `layer_head_dim` argument threaded through the projection, RoPE, KV-cache write, and flash-attention dispatches. The dual RoPE frequency buffers (`rope_freq_base` for full-attention, `rope_freq_base_swa` for the sliding-window layers) are two `freq_buf_handle` slots loaded per layer type. The `window_size` push constant on `flash_attn_batched.comp` is a six-line shader change plus one extra dispatch field, only needed for prompts longer than 1024 tokens.

Each of those is a commit. Each of them lands behind the existing `ZINC_BATCHED_PREFILL=validate` mode, which compares the batched output against the per-token output at the last-token logit level and gates the change on `max_abs_diff` collapsing to zero. The combined diff is roughly 150 lines of Zig and 25 lines of GLSL. That part of the port is not interesting.

What is interesting is what happens when you flip the gate. The first five deltas land, validate mode comes back clean on the SWA layers, and the full-attention layers go silently wrong.

## The sixth delta: asymmetric grouped-query attention

[Gemma 3 introduced a 5:1 ratio](https://huggingface.co/blog/gemma3) of sliding-window-attention layers to full-attention layers, with the SWA window pinned at 1024 tokens. Gemma 4's 31B configuration carries that pattern forward: 50 SWA layers and 10 full-attention layers, interleaved at every sixth position. The SWA layers run with a head dimension of 256 across Q, K, and V, which is the symmetric grouped-query attention shape that every shader in the world assumes by default. The full-attention layers do not.

On the full-attention layers, the Q projection produces 32 heads of dimension 512, for a Q tensor of shape `[hidden, 16384]`. The K and V projections produce 16 heads of dimension 128, for KV tensors of shape `[hidden, 2048]` each. That is grouped-query attention with a different per-head dimension on each side: Q is wider per head than K and V, not just denser in head count. The dot product still works because the inner product is performed against the K head dim, which by construction matches Q's per-head slice that participates in the dot. But the indexing math the shader uses to find each head's slice is parameterized on a single `head_dim` push constant.

The shipped `flash_attn_batched.comp` computes `q_base = head * head_dim` for the Q-side address math and `kv_base = ... * n_kv_heads * head_dim + kv_head * head_dim` for the KV-side address math. On every model ZINC has shipped batched prefill against until now, those two expressions have used the same `head_dim` because every previous model used symmetric GQA. Set `head_dim = 512` for Gemma's full-attention layers and the Q indexing is correct, but the shader strides the KV cache as if each KV head occupied 512 elements, which it does not. Set `head_dim = 128` and the KV math is correct, but Q indexing collides between heads.

This is not an exotic problem in the literature. The original [grouped-query attention paper](https://arxiv.org/abs/2305.13245) by Ainslie et al treats per-head dimension as a free parameter on each side, and the [Gemma reference implementation](https://github.com/google-deepmind/gemma) makes the asymmetry explicit in the model config. It is exotic in production inference shaders, because shipping engines mostly target LLaMA-shaped models where Q, K, and V all share a per-head dimension. The handful of models that break that assumption have either shipped after the shader was written or are recent enough that nobody pushed the path in anger.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/gemma-asymmetric-gqa-head-dim.svg" alt="A two-panel diagram. The left panel shows a Gemma 4 SWA layer with symmetric GQA at head_dim=256, where one push constant correctly drives both Q and KV indexing math. The right panel shows a full-attention layer where Q projects to 32 heads of dimension 512 and K and V project to 16 heads of dimension 128, and a single head_dim push constant cannot satisfy both indexing expressions. A bottom bar describes the fix: split head_dim into q_head_dim and kv_head_dim push constants on flash_attn_batched.comp." loading="lazy" />
  <figcaption>The left layer is what the shipped shader was written for. The right layer is what Gemma 4's flagship architecture demands. One push constant has two truths to tell, and it cannot tell both.</figcaption>
</figure>

The interesting bit the diagram makes obvious is that the asymmetry is not a clever optimization. It is a straightforward design choice in the model: spend per-head capacity on Q, where the attention scores are computed, and let K and V carry the smaller per-head footprint. The footprint asymmetry has shown up in [llama.cpp's Gemma loader plumbing](https://github.com/ggml-org/llama.cpp/issues/21434) too, in a different form, when a `sliding_window_pattern` config field was misread as a `uint32` instead of a `bool`. The shape of the model is forcing inference engines to look at constants they had been treating as obvious.

## Why one push constant was the wrong abstraction

The deeper lesson is about kernel API design, not Gemma. A push constant in a Vulkan shader is the cheapest possible way to thread a runtime value into a compiled pipeline, and the temptation is always to collapse closely-related parameters into a single value when the model architectures of the moment all happen to agree. ZINC's flash-attention shader collapsed `head_dim` for exactly that reason. So has every flash-attention kernel in [llama.cpp's Vulkan backend](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp) and most of the Metal-side function-constant shaders that mirror it. Read the [`flash_attn_cm2.comp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp) header push-constant block and the same single `D` parameter is right there, used both for the Q stride and for the KV stride.

The fix is to split `head_dim` into `q_head_dim` and `kv_head_dim`, propagate both through the indexing math, and pass both from the dispatch site. Roughly fifteen lines of GLSL, three fields on `dispatchFlashAttnBatched`, and the `validate` harness's `max_abs_diff` against the per-token reference drops back to zero on Gemma's full-attention layers without changing anything for the SWA layers or for any other model that happens to set `q_head_dim == kv_head_dim`.

The cost is that every other shader in the tree that touches per-head indexing has to look at itself for the same assumption. The KV-cache write kernel has a `head_dim` push constant of its own. So does the RoPE kernel for Q, separately from the RoPE kernel for K. None of those are wrong on the SWA layers, because the SWA layers are symmetric, and none of them are exercised on the full-attention layers today because Gemma 4's batched gate is still closed. As the rest of the deltas land, each kernel that touches per-head indexing gets the same audit pass and the same two-field split.

## What the fix unlocks

The expected number after the fix lands and the gate opens is **roughly 50 tok/s** on a 105-token Gemma 4 31B prefill on the same R9700, which would be a 10x improvement over the current 4.97 tok/s and would put Gemma 4 in the same prefill regime that Qwen3-8B occupies after the column-batched DMMV port. It is not the asymptotic ceiling, because the next batch of follow-on optimizations (specialization constants on the flash-attention head dim, Q8_1 activations on the SSM-shaped projections, the row-major X layout the effort log keeps coming back to) all stack on top.

The reason the conservative number is 50 tok/s and not the bandwidth-floor 100+ tok/s is that the compute-per-layer ratio on Gemma 4's full-attention layers is harder than on Qwen3-8B. The 16384-wide Q projection is doing real work, and the attention compute is denser per layer because the head dimension is bigger. A 10x prefill speedup is what a careful first pass should land. Anything more is the second-pass story.

The wider point is that the inference-engine work between "we support architecture X" and "we run architecture X at the speed the hardware can sustain" is increasingly about the kernel API surface, not the kernel inner loops. Every architecture that ships now has a constant somewhere that breaks an assumption nobody had bothered to write down. Gemma 4 has six. The other five are easy. The asymmetric-GQA one is a single push constant, and it is the kind of thing every Vulkan-backed local inference engine targeting consumer AMD will hit eventually, because the next generation of open-weight models is going to keep playing with attention shape in ways that the post-LLaMA shaders did not anticipate.

## What comes next

The next ZINC commit in the Gemma 4 prefill arc is the head-dim split on `flash_attn_batched.comp`, with the matching `dispatchFlashAttnBatched` change, plus the SWA `window_size` push constant for prompts longer than 1024 tokens. The first five deltas from the effort log land before that under the still-closed gate, so they can sit as dead code without changing any shipped behavior. Once the head-dim split lands, the gate opens and `validate` mode runs end-to-end against the per-token Gemma 4 reference. The first measured prefill number is the next post.

The broader note for any Vulkan-backed local inference engine is the same one the [DMMV variant post](/blog/2026-04-22-why-rdna4-prefill-wants-a-32-column-dmmv-before-a-gemm) ended on. The shaders that work for today's model lineup are not the shaders that work for next quarter's. Search your push-constant blocks for any parameter named `head_dim`, `n_heads`, or `num_kv_heads` and check whether the kernel's indexing math is actually robust to the case where Q and KV disagree. If the shader assumes they do not, that assumption is going to find a model that breaks it within the year.
