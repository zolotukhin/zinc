---
title: "Why RDNA4's matrix cores sit out the local LLM decode loop"
date: "2026-04-30"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - decode
  - prefill
  - cooperative-matrix
  - wmma
  - roofline
  - llm-inference
keywords:
  - RDNA4 AI accelerator decode
  - Radeon AI PRO R9700 LLM inference
  - VK_KHR_cooperative_matrix RDNA4
  - WMMA RDNA4 matrix cores
  - LLM decode roofline analysis
  - batch=1 decode arithmetic intensity
  - memory-bound LLM inference AMD
  - Qwen 3.5 35B-A3B decode bandwidth
  - INT4 TOPS local LLM
  - cooperative matrix prefill RDNA4
excerpt: "AMD specs the Radeon AI PRO R9700 at 389 TFLOPS of sparse FP16 and 1557 TOPS of sparse INT4. A local 35B-A3B decode on the same card runs at roughly 50 tokens per second and uses almost none of that compute. The arithmetic intensity of a single decoded token is around two operations per byte. The ridge point of the roofline on this card is north of six hundred. The matrix cores are not the lever for solo decode. They are the lever for prefill, fine-tuning, and any workload that touches the same weight twice."
---

The AMD Radeon AI PRO R9700 product page advertises [389 TFLOPS of sparse FP16 and 1557 TOPS of sparse INT4](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html) on its second-generation AI accelerators. Those are real numbers and the silicon really delivers them under the right workload. The catch is that the workload everybody actually runs on this card most of the day, single-user batch=1 LLM decode, is not that workload. On a Qwen 3.5 35B-A3B decode at 50 tokens per second, the matrix engines are mostly idle. The card is bandwidth-bound by orders of magnitude.

This is not a complaint about the hardware. It is a complaint about how the hardware gets pitched. AI accelerator TOPS only matter for workloads with high arithmetic intensity, and the arithmetic intensity of a single decoded token is set by the model architecture and the batch size, not by the marketing material. Below the ridge point of the roofline, every ops-per-second number on the spec sheet is irrelevant. Decode lives well below the ridge point. So does any other workload that walks the weights once per output.

This post is the careful version of that argument. It walks through where the R9700's ridge point actually sits, why batch=1 decode lands two orders of magnitude to the left of it, and why the same matrix engines are still the right place to spend engineering effort. They are a prefill prize. They are not a decode lever.

## Where the ridge point sits

The roofline model, [introduced by Williams, Waterman, and Patterson at UC Berkeley](https://en.wikipedia.org/wiki/Roofline_model), expresses peak achievable performance as the minimum of two ceilings. The compute ceiling is the chip's peak ops per second. The bandwidth ceiling is the chip's peak memory throughput multiplied by arithmetic intensity, the operations performed per byte read. The ridge point is where the two cross. Workloads to the left of the ridge are bandwidth-bound. Workloads to the right are compute-bound.

For the R9700, the bandwidth ceiling is 644 GB/s of GDDR6 at 256 bits and 20 Gb/s. The dense FP16 matrix ceiling at 1.5x boost is roughly 195 TFLOPS, and the dense INT8 ceiling is roughly 390 TOPS. Sparsity adds a factor of two on top of either if the workload's tensors actually qualify, which inference at batch=1 does not. The ridge points are then 195 TF / 644 GB/s, or about **303 ops per byte for FP16**, and 390 TOPS / 644 GB/s, or about **606 ops per byte for INT8**. With INT4 the ridge sits past 1200 ops per byte. Sparsity-on numbers double those ridge points again.

Those are the numbers that matter when reading the spec sheet. A workload at 5 ops per byte uses about 1% of the FP16 matrix throughput on this card no matter how good the kernel is. A workload at 1 op per byte uses about half a percent. The ceiling exists, but you cannot reach it from the wrong side of the ridge.

## What batch=1 decode actually looks like

The arithmetic intensity of a transformer decode step is straightforward. For one token, the model reads its active weights once and does one matmul against a single activation row per linear layer. The dot product cost is two operations per weight (one multiply, one add). The bytes read are the weight bytes. The arithmetic intensity is therefore two operations per byte, regardless of model size, regardless of quantization, regardless of attention pattern.

That single number is the entire decode story. Increasing model parameters changes both numerator and denominator equally; it does not move the workload along the roofline. Quantizing weights from FP16 to INT4 changes the *bytes read* by 4x but does not change the operations, so it cuts the time by 4x while keeping arithmetic intensity at the same two ops per byte. Smarter kernels do not move the dot. Better caches do not move the dot. The dot is set by the architecture of the workload.

For Qwen 3.5 35B-A3B at Q4_K_M, the active per-token weight bytes are roughly 3 GB. The KV cache adds another 0.12 MB per token at FP16 with this model's GQA shape, and the per-step KV reads scale with prompt length. At a 4k context the per-step KV read is about 0.5 GB. The total per-step bytes are around 3.5 GB and the total per-step ops are around 6 GFLOPs. Arithmetic intensity is **roughly 1.7 ops per byte at 4k context** and falls toward 1 ops per byte as the KV cache grows past the active weights, which is the bandwidth crossover that the [16k crossover post](/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode) walked through.

A decode running at 1.7 ops per byte on a card whose ridge point is 303 ops per byte is using under one percent of the matrix throughput. Whether that one percent ships through the AI accelerators or through the regular vector ALUs is not interesting; the throughput limit is somewhere else.

## The roofline picture

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-04-30-rdna4-decode-roofline.svg" alt="A log-log roofline plot for the AMD Radeon AI PRO R9700. The x-axis is arithmetic intensity in operations per byte, on a log scale from 0.1 to 10000. The y-axis is throughput in GFLOPS, on a log scale from 1 to a million. A diagonal bandwidth line at 644 GB per second rises from the lower-left and meets three flat ceilings: dense FP16 at 195 TFLOPS, dense INT8 at 390 TOPS, and dense INT4 at 780 TOPS. A red dot labeled batch=1 decode sits on the bandwidth diagonal at arithmetic intensity 1.7 and roughly 1.1 TFLOPS. A yellow dot labeled prefill at batch 16 sits further right at intensity 32 and 20 TFLOPS. A second yellow dot labeled prefill at batch 256 sits at the corner of the FP16 ceiling at intensity 512. A dashed vertical line marks the FP16 ridge point at about 303 ops per byte." loading="lazy" />
  <figcaption>The R9700 roofline. Batch=1 decode lives two orders of magnitude to the left of the FP16 ridge. Prefill at modest batch sizes is where the matrix throughput becomes reachable.</figcaption>
</figure>

The thing the chart makes obvious is that the question "do the matrix cores help my decode" has a one-line answer. The decode workload's dot is so far to the left of the ridge that no amount of compute headroom can rescue it. The card spends the entire token-time pulling weights out of memory. The matrix cores wait their turn and do not get one.

The picture also makes the next move obvious. Move the dot rightward and the ceilings start to bind. Prefill is the easy way to move the dot rightward, because prefill processes N tokens against the same weight read. The arithmetic intensity of a length-N prefill is roughly 2N ops per byte for the dense linear layers, modulo attention specifics. At N=64 tokens a prefill workload is at 128 ops per byte and starting to crowd the FP16 ridge. At N=256 it lives well past it.

## Why the cooperative matrix port still matters

The Mesa RADV driver landed [VK_KHR_cooperative_matrix support for RDNA4 in February of last year](https://www.phoronix.com/news/RADV-Lands-RDNA4-Coop-Matrix), following the earlier RDNA3 support. The Khronos extension is the Vulkan-side surface for what AMD calls [WMMA on RDNA3](https://gpuopen.com/learn/wmma_on_rdna3/), the wave-level matrix multiply instructions that drive the AI accelerator path. The [Khronos extension proposal](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_KHR_cooperative_matrix.adoc) lays out the cooperative-matrix types and the SPIR-V intrinsics that compile down to those instructions on hardware that supports them.

Cooperative matrix is the right port to chase, but for the right reason. It is the path that turns a 64-token prefill batch from a bandwidth-and-compute mix into a compute-bound workload that can use the matrix throughput on the spec sheet. The [llama.cpp Vulkan backend](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp) ships separate `mul_mm` and `mul_mm_cm2` shader paths exactly because of this split: the `cm2` path is the cooperative-matrix variant and it kicks in for the larger M-dimension tiles where the matrix throughput buys something. ZINC's prefill plan tracks the same shape.

The mistake to avoid is reading those engineering decisions as a decode story. They are not. A cooperative-matrix `mul_mat_vec` for batch=1 decode would issue the same number of memory transactions as the existing scalar dispatch, finish the matrix instruction in a fraction of the time, and then sit idle waiting for the next weight tile to arrive. The wall time would be unchanged. The decode roofline is not going to move, no matter how many cooperative-matrix dispatches the engine learns to issue, because the dot is to the left of the ridge.

## Where the matrix throughput actually pays

There are three workloads on a single R9700 that genuinely live near the ridge and benefit from the matrix engines. The first is prefill at any non-trivial prompt length, which is the path the cooperative-matrix port targets directly. The second is fine-tuning, where every weight is read once and used against an entire batch of training examples; LoRA fine-tunes on the same card see the matrix throughput because the workload is GEMM, not GEMV. The third is multi-user inference, where batch=1 decodes from independent users get fused into batch=N attention calls and the activations share weight reads across users.

Solo desktop decode is none of those. A single user typing into a chat window sees one token request at a time, against one set of weights, and the matrix engines have nothing to chew on. The path that helps that user is to read fewer bytes per token: smaller weights through quantization, smaller KV cache through KV quantization, and fewer dispatches between layers so the bandwidth-bound stretch sees fewer gaps. All three of those are subjects we have hit in earlier posts, and none of them is about TOPS.

## What this changes about the engine roadmap

The practical takeaway for a local engine roadmap is that "use the matrix cores" is not a single goal. It is two distinct goals with different timelines and different value. The prefill arc, including the [32-column DMMV before a GEMM port](/blog/2026-04-22-why-rdna4-prefill-wants-a-32-column-dmmv-before-a-gemm) and the cooperative-matrix tiles that follow it, is where matrix throughput buys end-user time. The decode arc is where bytes-per-token still wins, and where the [FP16 KV cache argument](/blog/2026-04-26-why-fp16-kv-cache-is-the-wrong-default-for-128k-context-on-32gb-rdna4) and the [single-vkQueueSubmit work](/blog/2026-04-25-why-one-vkqueuesubmit-per-prompt-is-the-next-quiet-rdna4-prefill-unlock) actually move the number.

The deeper takeaway is for anyone reading a GPU spec sheet for an AI workload. The TOPS number on the box describes a ceiling. Whether that ceiling is reachable on the workload in front of you depends on the arithmetic intensity, and the arithmetic intensity is set by the workload, not by the chip. For batch=1 LLM decode, the relevant number on the spec sheet is the bandwidth, not the TOPS. The matrix cores are real and the matrix cores are useful, but they are useful exactly for the workloads that already live on the right side of the ridge. Sit on the wrong side and the spec sheet is a story about the hardware you did not buy.
