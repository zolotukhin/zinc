---
title: "Why Q8_1 activations are the next RDNA4 prefill unlock"
date: "2026-04-19"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - quantization
  - q8-1
  - mul-mmq
  - prefill
  - llm-inference
  - gpu-kernels
keywords:
  - Q8_1 activation quantization
  - mul_mmq Vulkan
  - RDNA4 prefill optimization
  - AMD consumer GPU LLM inference
  - integer dot product DMMV
  - quantize_q8_1 GLSL
  - llama.cpp mul_mmq
  - wave64 K-parallel DMMV
  - ZINC Qwen3.5 prefill
  - activation quantization LLM
  - subgroup integer reduction
  - RX 9070 inference
  - AI PRO R9700 inference
excerpt: "Weight quantization is everywhere in local LLM inference. Activation quantization is not, and that is the reason RDNA4 prefill on Qwen3.5-35B is still reading FP32 through a dequantize-then-multiply pipeline. Here is why Q8_1 and mul_mmq are the biggest unshipped prefill win on consumer AMD, how llama.cpp routes around the problem ZINC still has, and what the numbers say the upside looks like."
---

Every serious local LLM runtime ships four-bit weights. Almost none of them ship four-bit or eight-bit activations. On AMD RDNA4, that asymmetry costs a factor of two to four on the largest prefill dispatches, and it is the single largest block between ZINC's current `25.67 tok/s` prefill on Qwen3.5-35B and a number that starts with a three-digit prefix.

The good news is that the fix is already in the open. [llama.cpp's Vulkan backend](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp) has run it in production for months, split across two small shaders ([`quantize_q8_1.comp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp) and [`mul_mmq.comp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mmq.comp)) plus a handful of buffer plumbing changes. This post is the concrete case for why that pattern is the next RDNA4 prefill unlock in ZINC, why it matters beyond ZINC for anyone running 35B-class models on consumer AMD, and what the tradeoffs actually look like once you leave the slide.

## The asymmetry at the center of every quantized prefill kernel

Open any weight-quantized LLM kernel and inspect the inner loop. The weights are some flavor of Q4_K, Q5_K, Q6_K, or Q8_0. The activations are FP32 or FP16. The dot product is a float multiply-add: decode a block of weights into FP16 tiles, multiply by the activation element, accumulate. That is the current default in ZINC's `dmmv_q4k_batch.comp` and `dmmv_q8_0_batch.comp` shaders. It is also what llama.cpp's decode path does.

This default is reasonable on the decode side. At decode you do one token at a time, weights dominate bandwidth, and the activation vector is tiny. Whether the activation is FP32 or an int8 rounded from the same FP32 does not change the math that matters.

Prefill is the opposite story. At a 154-token prompt ingestion on Qwen3.5-35B, the SSM projections are fired `4 × 30 × 154 = 18,480` times with the same weight tensors being read from VRAM over and over. Phase timing inside ZINC puts SSM proj alone at about **1.3 seconds** of the roughly **6 seconds** of GPU work. The MoE gate, up, and down projections fire similar multiples inside each of the `128` experts' top-8 routing cohort. The total FP32 multiply-add volume is enormous. The total weight bandwidth is enormous. None of it is being reduced by the fact that the activations are 32-bit.

RDNA4 has a native integer dot path, surfaced through the same [`VK_KHR_cooperative_matrix` extension](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/appendices/VK_KHR_cooperative_matrix.adoc) that powers cooperative-matrix matmul on NVIDIA and via dp4a-style packed integer multiply-adds on the scalar path. That path is not being used in prefill today in most local inference engines that live outside the CUDA ecosystem. It is sitting idle.

![AMD Radeon AI PRO R9700, the workstation RDNA4 card ZINC targets for AMD local inference. The idle integer dot path is on this silicon.](/blog/gpu_1.jpg)

## What Q8_1 actually is, and why activations are easier than weights

Q8_1 is a blocked signed 8-bit format. Each block of 32 activation values shares a single FP16 scale plus a second FP16 holding the block sum. The payload is `32 int8 + 2 fp16 = 36 bytes`, against the `32 × 4 = 128 bytes` of the FP32 activation vector it replaces. Quantizing activations on the fly is a different problem than quantizing weights, and it is easier, not harder.

Weight quantization is a compression problem. The weights are fixed once the model is trained, you get exactly one shot at choosing a quant scheme, and a bad choice is visible in every output token for the life of the model. That is why the [llama.cpp k-quants work](https://github.com/ggerganov/llama.cpp/pull/1684) took months of PR review before a 2-to-6-bit mixed scheme landed as the default, and why the k-quant variants in that PR are still the production spine of every local inference stack that ships quantized weights today.

Activation quantization inside a prefill kernel is a much smaller problem. The activations are only going to live through one layer's worth of projections. You re-quantize the layer's input once, use it across the dozen or so matmul dispatches that feed it, and throw the Q8_1 buffer away when you enter the next layer. A block size of 32 is forgiving: the scale captures the local dynamic range, and typical transformer post-RMSNorm activations fit cleanly into int8 after scaling. The same arithmetic already runs in the decode path at much smaller batch sizes without visible damage to perplexity.

Once you have the Q8_1 buffer, the dot product changes shape. Each output row reads a Q4_K or Q5_K weight block, performs a single dequant step into int8 lanes on the fly, multiplies lane-by-lane against the Q8_1 activation block using a packed integer multiply-add, reduces across the subgroup with `subgroupAdd`, and writes an FP32 partial sum. The final accumulator stays float. Nothing upstream of the dispatch changes.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/q8-1-mul-mmq-dot-product.svg" alt="A two-lane flow diagram. The top lane shows today's RDNA4 prefill path in ZINC: Q4_K weights plus FP32 activations flow into an online dequant step, then a float multiply-add dot product running serially on one wave64 thread per row, producing an FP32 partial sum. The bottom lane shows the mul_mmq path: Q4_K weights unchanged, an upfront quantize-to-Q8_1 pass running once per prompt chunk, a Q8_1 activation buffer reused across every projection in the layer, an integer dot product with 64 subgroup lanes using a packed int8 multiply-add, and the same FP32 partial sum accumulator." loading="lazy" />
  <figcaption>The only difference between the two lanes is what the GPU multiplies. Weights are the same. The accumulator is the same. The inner reduction becomes integer, and the activation bandwidth drops by four.</figcaption>
</figure>

The important thing the diagram hides is how rarely the `quantize_q8_1` pass actually runs. llama.cpp quantizes the activations once per prefill chunk and reuses that buffer for every projection in the layer. On Qwen3.5-35B that is seven reuses per layer at minimum between the four SSM projections and the three MoE gate, up, and down projections. The amortized cost of the quant pass is a rounding error on any prompt longer than a couple of tokens.

## What the numbers say the upside looks like

There are three independent measurements that argue for the same ballpark. None of them is a marketing claim. All three have been run or benchmarked by other teams on roughly the same silicon class, and they are the grounds on which the ZINC effort log currently scopes the Q8_1 port as "Step 10" with an estimated `2x to 4x` upside on the largest prefill DMMVs.

| Workload segment | Current FP path | With Q8_1 + mul_mmq | Why it changes |
| --- | ---: | ---: | --- |
| Q4_K SSM projection, 154 prompt tokens | ~1.30 s | ~0.4 to 0.6 s | Int8 packed mul-add, 64-lane reduction, activations 4x smaller |
| Q5_K MoE gate/up/down, 154 prompt tokens | ~1.60 s | ~0.5 to 0.8 s | Same reduction, grouped across expert cohort |
| FP32 -> Q8_1 quant pass (once per chunk) | 0 s | ~0.02 s | New dispatch, amortized over seven reuses per layer |
| Prefill TTFT, flagship 154-token benchmark | ~13.0 s | ~5 to 7 s projected | Compound of the above plus existing double-buffering |

The first two rows are the heart of the argument. The SSM and MoE buckets are where the time goes today, and they are both DMMV-shaped dispatches against quantized weights with FP32 activations. A realized fraction of the headline `2x to 4x` on the two largest buckets is enough to move the flagship from `25.67 tok/s` prefill into the range where TTFT for a long-context prompt stops being the user-visible bottleneck.

The numbers in the last column are projections, not measured. They assume Step 10 lands on top of a wave64 K-parallel single-column DMMV shader rewrite, which is the foundation that a pair-batch attempt earlier this month demonstrated is a precondition for every column-batching win that follows. The `2x to 4x` figure is the upper bound the effort log currently holds, based on the ratio between the dequant-then-FP32 path and an integer-dot path on the same quantized matmul shapes in the llama.cpp Vulkan codebase. Calling it a guaranteed speedup before ZINC ships the port would be dishonest. Calling it zero would ignore every other consumer GPU runtime that has already measured this win.

## The tradeoff nobody wants to state plainly

Every activation-quant story eventually has to answer whether the model gets worse. In practice, the answer on a Q8_1 block size of 32 against post-RMSNorm transformer activations is "not measurably." llama.cpp ships this format in its quantized matmul default for the Vulkan, CUDA, and HIP backends and has for long enough that any systematic regression would have been caught in the project's perplexity sweeps. The path that does have a known accuracy cost is activation-quant at `int4` or `int2` scale, which is a different conversation and is not on the table here.

The more honest tradeoff is engineering time. A `quantize_q8_1` pass is not complicated, but shipping it correctly means adding a new buffer type (Q8_1 is not just raw int8, it has the per-block FP16 scale and sum), updating every shader that reads activations to understand the new layout, and sequencing the new dispatch into the existing double-buffered prefill pipeline so the CPU can still record token N+1 while the GPU runs token N. Each of those is bounded. None of them is a shader rewrite. All of them together is a week of work if the double-buffer plumbing stays correct, which it does.

The second tradeoff is the one that often surprises people: mul_mmq does not replace cooperative-matrix `mul_mm`. On the largest prefill projections where the column count exceeds DMMV's sweet spot, the right kernel is still the `mul_mm` tiled matmul, for the same reason GEMMs eat GEMVs once N is large enough. Q8_1 plus mul_mmq is the big win in the DMMV regime where most prompt-loop dispatches currently live. Coop-matrix `mul_mm` is the big win for a smaller number of very large projections. The two paths stack, they do not compete, and the effort order treats them as independent bets.

The third tradeoff is that the LM head, the single largest dispatch by output dimension in Qwen3.5-35B, does not benefit. The LM head fires once per prefill and is already excluded from non-terminal prompt tokens by ZINC's dead-tail skip. A specialized matmul variant for the LM head proves nothing on prefill throughput, because it is not in the prefill hot path to begin with.

## What this says about the wider local inference stack

The larger point is that local inference engineering has moved past the era when "we quantized the weights" was enough. On a `35B` hybrid MoE model running on a 32 GB RDNA4 card with 256-bit GDDR6 behind it, the bandwidth ceiling is not the weight read, and it is not the activation write. It is the arithmetic intensity of the inner reduction multiplied by the number of dispatches in the prompt loop. Driving that product down is what separates engines that report prefill numbers competitive with llama.cpp from engines that do not.

The runtimes that do this well already do it. llama.cpp's Vulkan backend does it, and the specific shaders are in the same repository everyone reads. [vllm does a parallel version of this for MoE](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/fused_moe/fused_moe.py) on the CUDA side. Neither project is hiding anything. The code is open, the kernels are documented, and the numbers have been in circulation long enough that there is no excuse left for any local inference project that wants to take RDNA4 prefill seriously to keep the FP32 activation path as its default.

What it means for ZINC is narrower. The next performance suite that goes out with a Q8_1 activation path will close most of the current prefill gap to llama.cpp on this workload, and it will do so without touching decode, without changing quantization, and without a single new line of CPU-side inference logic. For everyone else who is building a runtime against consumer AMD, the lesson is the same one the effort log keeps returning to: the easy wins on this hardware are not in the shaders that already exist. They are in the shapes of the shaders that have not been ported yet.

For background on why this matters for the Qwen3.5-35B flagship in particular, the [stuck-at-25 writeup](/blog/2026-04-18-why-rdna4-prefill-for-qwen-3-5-is-stuck-at-25-tok-s) covers the per-phase budget and the failed attempts that made clear this is the right next bet. The [shaders between 4 and 27 tok/s post](/blog/2026-03-29-the-shaders-standing-between-4-tok-s-and-27-tok-s) is the prior context for how the engine got to where it is, and [every design decision behind ZINC](/blog/2026-04-03-every-design-decision-behind-zinc) spells out the broader shape the runtime is aimed at.
