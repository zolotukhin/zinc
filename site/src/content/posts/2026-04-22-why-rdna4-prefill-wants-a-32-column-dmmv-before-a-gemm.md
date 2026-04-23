---
title: "Why RDNA4 prefill wants a 32-column DMMV before it wants a GEMM"
date: "2026-04-22"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - prefill
  - qwen3-8b
  - llm-inference
  - gpu-kernels
  - shader-specialization
  - dmmv
keywords:
  - column-batched DMMV Vulkan
  - RDNA4 prefill kernel
  - mul_mat_vec NUM_COLS specialization
  - MAX_COLS 32 shader
  - AMD RDNA4 register allocation
  - Q4_K prefill Vulkan
  - Qwen3-8B prefill RX 9070
  - tiled GEMM vs DMMV LLM
  - llama.cpp mul_mm tiled matmul
  - wave64 VGPR occupancy RDNA4
  - local LLM batched prefill
  - ZINC RDNA4 prefill plan
faqs:
  - question: "What is a column-batched DMMV and how does it differ from a normal DMMV?"
    answer: "A DMMV is a dequantize-and-multiply matrix-vector kernel. It reads one quantized weight block, decodes it, multiplies it by one activation vector, and writes one output element per thread. A column-batched DMMV reads the same weight block once and multiplies it by N different activation vectors from N consecutive prompt tokens, accumulating N output elements per thread. The inner arithmetic is identical. The difference is that the weight block is paid for once instead of N times."
  - question: "Why pick 32 as the column count?"
    answer: "Because the per-column accumulators live in VGPRs, and the register file on an RDNA4 SIMD has to host enough waves to keep the CU occupied. At `MAX_COLS = 32` the shader still compiles to a register footprint that lets the per-token decode path, which shares this pipeline with `num_cols = 1`, hit around 78 tok/s on Qwen3-8B. Raising it to 64 regresses that path to 59 tok/s because occupancy drops. Without a second pipeline variant, 32 is the right shared ceiling."
  - question: "Why not just ship a proper tiled GEMM like llama.cpp's `mul_mm.comp`?"
    answer: "Eventually, yes. A tiled GEMM with cooperative-matrix tiles is the right asymptotic shape once N grows past a few hundred columns. The reason it is not the first move is that the 32-column DMMV closes roughly 96 percent of the weight-read waste at a fraction of the engineering cost, does not require a coop-matrix path for every quant type, and does not touch the per-token decode kernel that already works. GEMM wins the last stretch. DMMV batching wins the first, larger stretch."
  - question: "Does this matter on dense models or only MoE?"
    answer: "Dense models feel the gap the hardest, because every per-token prefill dispatch reads the full Q, K, V, O, gate, up, and down projections out of VRAM. A 103-token Qwen3-8B prompt re-reads 755 GiB of weights. The same prompt with a 32-column DMMV reads 30 GiB. MoE adds routing and expert-cohort batching on top of the same underlying shape, but the dense case is where the argument is cleanest."
  - question: "What stops a local inference engine from simply adopting this?"
    answer: "Nothing structural. The shader exists in ZINC's tree as `dmmv_q4k_batch.comp`, and llama.cpp ships its own column-batched variants behind `NUM_COLS` as a specialization constant. What is missing is the prefill path actually calling into the batched shader with the right chunk size, scratch-buffer lifecycle, and pipeline-state variants for each `num_cols` that the register allocator needs to see. That plumbing is a week of work."
excerpt: "ZINC's per-token RDNA4 prefill on Qwen3-8B runs at 59 tok/s while llama.cpp's Vulkan backend hits 662 tok/s on the same card. The first instinct is to ship a tiled matmul kernel. The real first move is smaller: a column-batched DMMV that reads each weight once and multiplies it by up to 32 prompt tokens at a time. Here is why that shape, not a GEMM, is where an RDNA4 prefill port should start."
---

A fair-benchmark cold run of [Qwen3-8B](https://github.com/QwenLM/Qwen3) at Q4_K_M on an RX 9070 AI PRO gives us the same number twice. ZINC's per-token prefill path reports **59 tok/s** on a 103-token prompt. [llama.cpp's Vulkan backend](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp) on the same machine, same weights, same prompt, reports **662 tok/s**. That is an **11x gap** on a dense model small enough that there is no MoE routing, no SSM, no hybrid architecture to blame.

The first instinct inside every optimization review we have run on this is the same one: ship a proper tiled GEMM, match llama.cpp's `mul_mm.comp` kernel shape, call it done. That instinct is not wrong. It is just ordered wrong. The move that closes most of the gap is a smaller kernel change that has been sitting in ZINC's tree for a month, wired into exactly one call site.

This post is the case for why a column-batched DMMV, not a tiled GEMM, is the correct first kernel for an RDNA4 prefill port. The argument is concrete: weight-read math, register-file math, and a specific shader whose `MAX_COLS = 32` comment is itself a small piece of hard-won RDNA4 knowledge.

## Prefill is bandwidth-bound on weight re-reads, not on arithmetic

The naive shape of a transformer prefill is "the decode loop, run N times." Read the weights, multiply by one activation vector, advance. That shape is correct as math and ruinous as I/O. A Qwen3-8B layer has seven distinct projections that a prefill must fire once per token: Q, K, V, O, gate, up, down. Each projection is a quantized weight tensor living in VRAM.

For a 103-token prompt, the per-token path reads roughly **21 GiB of weights per layer**, across all 36 layers of Qwen3-8B, for a total of **~755 GiB of weight traffic** per prefill. On an RX 9070 AI PRO with 576 GB/s of memory bandwidth, that sets a floor of roughly **1.3 seconds** on prefill walltime just from weight reads, which is essentially the entire measured prefill time of 1.36 seconds. The engine is not compute-bound. It is busy reading the same bytes out of VRAM again and again.

A column-batched kernel reads every weight block once and reuses it across N prompt tokens. On the same 103-token Qwen3-8B prompt, that cuts total weight traffic from 755 GiB down to a floor of **7.4 GiB**, which at 576 GB/s sets a bandwidth lower bound near **13 ms**. The 100x reduction is not hypothetical; it is what the llama.cpp Vulkan backend has been doing for more than a year.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/rdna4-prefill-three-regimes.svg" alt="Two stacked horizontal bar charts comparing three kernel regimes for the same Qwen3-8B 103-token prefill on RDNA4. The top chart compares weight bytes re-read from VRAM: 755 GiB for per-token DMMV, 30 GiB for 32-column batched DMMV, 7.4 GiB for tiled GEMM. The bottom chart compares GPU dispatch counts: 59,000 for per-token, 2,500 for 32-column, 1,500 for tiled GEMM." loading="lazy" />
  <figcaption>The gap between the first two bars in each panel is an order of magnitude. The gap between the second and third is small. Most of the prefill win lives in the first jump, which is exactly the one a column-batched DMMV delivers.</figcaption>
</figure>

The shape that matters is the ratio between those bars. A 32-column DMMV already closes roughly 96% of the weight-read waste and 96% of the dispatch waste. The tiled GEMM closes the last sliver, and it does so at a higher engineering cost because cooperative-matrix paths do not exist for every quant type the engine has to support.

## The kernel ZINC already has

The shader that does this is not theoretical. It sits in `src/shaders/dmmv_q4k_batch.comp` and it has been in the tree long enough to ship with a comment that is itself the reason this post exists:

```glsl
// Max batch size — compile-time constant for register allocation.
// Keeping at 32 because raising to 64 reduces occupancy enough on
// RDNA4 that the shared LM-head path (per-token decode uses this
// same shader with num_cols=1) regresses from 78 → 59 tok/s on
// Qwen3-8B. Prefill only uses num_cols > 1, but the register budget
// is shared across all callers. A dedicated MAX_COLS=64 variant
// with a separate pipeline would be the right fix if the chunk
// reduction becomes worth pursuing — keep that in mind for the
// follow-on tiled mul_mm work.
const uint MAX_COLS = 32u;
```

The inner loop is a one-line change from the per-token version. Instead of multiplying one decoded weight by one activation, it multiplies the same decoded weight by up to `num_cols` activations and accumulates into a register array:

```glsl
for (uint e = 0u; e < 32u; e++) {
    uint byte_val = uint(a_data[qs_base + e]);
    float w_lo = factor_lo * float(byte_val & 0x0Fu) - bias_lo;
    float w_hi = factor_hi * float((byte_val >> 4) & 0x0Fu) - bias_hi;

    // Apply dequantized weight to ALL input columns
    for (uint c = 0u; c < num_cols; c++) {
        uint col_base = x_base + c * K;
        sums[c] += w_lo * x_data[col_base + elem_lo + e]
                 + w_hi * x_data[col_base + elem_hi + e];
    }
}
```

The math is the same DMMV that decode has always used. The difference is the `for (uint c = 0u; c < num_cols; c++)` inner loop and the per-column accumulator array `sums[MAX_COLS]`. Each weight block is fetched, decoded, and held in per-thread registers long enough to feed `num_cols` multiply-adds before the next block is read. Weight bandwidth drops by `num_cols`. Arithmetic grows by the same factor, which is exactly what a GPU wants to hear.

![AMD Radeon AI PRO R9700 from behind, workstation-plain. The GPU this post is arguing at.](/blog/gpu_3.jpg)

## Why 32, and why RDNA4 makes that choice for you

The `MAX_COLS = 32` number is not a magic constant pulled out of a tuning sweep. It falls out of the RDNA4 register file.

An RDNA4 Compute Unit has two SIMDs, and each SIMD carries a **192 KB vector register file** with **1536 VGPRs of 32-bit width per SIMD** in wave32 mode, as catalogued in the [community RDNA4 architecture reference](https://github.com/azhirnov/cpu-gpu-arch/blob/main/gpu/AMD-RDNA4.md). Wave64, which this shader runs under to match the existing wave64 call sites in ZINC's DMMV family, runs one wavefront across two SIMDs and consumes twice the registers per wavefront. Occupancy, the number of concurrent wavefronts a CU can keep in flight, is a direct function of how many VGPRs the shader needs.

Raising `MAX_COLS` from 32 to 64 doubles the per-column accumulator array. The register allocator has to reserve VGPRs for all `MAX_COLS` accumulators whether or not the `num_cols` push constant actually uses them, because the register map is fixed at compile time. At 32 accumulators plus the rest of the shader's working set, occupancy holds high enough to keep the CUs fed. At 64, occupancy drops enough that the shared per-token decode path, which uses the same pipeline with `num_cols = 1`, regresses from **78 tok/s to 59 tok/s** on Qwen3-8B. The prefill side would win from 64. The decode side pays for it. The shared-pipeline constraint picks 32.

The clean fix is to compile two pipeline variants and let each call site pick the right one. That is what llama.cpp does with GLSL [specialization constants](https://github.com/KhronosGroup/GLSL/blob/main/extensions/khr/GLSL_KHR_cooperative_matrix.txt), which bake a compile-time integer into the shader at pipeline-creation time and let the register allocator see a static `for` bound. llama.cpp's [`mul_mat_vec.comp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp) ships with `NUM_COLS` as a specialization constant, and the build script compiles one variant per column count. Each variant gets its own register budget. Decode loads the `NUM_COLS = 1` variant. Prefill loads whichever column count matches the chunk size. No sharing, no regression. That is a narrow, cheap shader-build change, not a runtime rewrite.

## What this does to dispatch count and where the GEMM still wins

The second axis the chart shows is dispatch count. Each `vkCmdDispatch` pays a host-side recording cost of roughly 1 to 3 microseconds on RADV plus a GPU-side launch overhead in the 5 to 20 microsecond range for the dispatch sizes common in LLM inference. On a 103-token Qwen3-8B prefill, the per-token path records roughly 576 dispatches per token across 36 layers, which comes out to about **59,000 dispatches per prompt**. At an optimistic 6 microseconds per dispatch, that is **~354 ms of pure launch overhead**, roughly a quarter of the measured 1.36 s prefill walltime.

A 32-column DMMV chunks the 103 tokens into four passes of 32 columns each, so each projection in each layer fires four batched dispatches instead of 103 per-token dispatches. The seven projections times 36 layers times four chunks gives roughly **1,000 projection dispatches** across the whole prefill, plus another 1,500 or so elementwise dispatches for RoPE, flash attention, SwiGLU, and residuals that are still per-token in the first port. Total dispatches land near **2,500**. Host launch overhead falls from 354 ms toward 15 ms. That alone accounts for a non-trivial fraction of the projected speedup.

A tiled GEMM would collapse the projection dispatches further, to roughly one `mul_mm` dispatch per projection per layer, for **~252 projection dispatches**. The elementwise floor stays the same. Total drops from 2,500 to roughly 1,500. The absolute win is real, but it is much smaller than the launch-overhead win from moving off per-token, and it comes with a compile-time cost: `mul_mm` is a tiled kernel with shared-memory caching and cooperative-matrix tiles, and every new weight quant format needs its own tuned tile size to match the quant's block structure. Q4_K, Q5_K, Q6_K, and Q8_0 are each their own port.

The DMMV batching port is simpler because the inner loop does not change shape. It stays a dequantize-then-multiply reduction, it stays wave64 with 64 threads per row, and the existing infrastructure around `recordBatchDispatchPush` already knows how to hand it the right descriptors. Adding a column axis with a specialization constant is a boring change. Adding a tiled GEMM is a design problem.

## The numbers this is expected to produce

The effort log's current scoping for this work projects landing in the **350 to 500 tok/s** range on first commit, roughly 6x to 8x the current per-token baseline on Qwen3-8B. The argument has three moving parts. Weight-read bandwidth falls by the ratio of 755 to 30 on Q4_K projections. Launch overhead falls from 354 ms toward 15 ms. Elementwise ops and flash attention stay where they are in the first port, which caps the upside until those also get batched. llama.cpp's 662 tok/s is the ceiling that stays above the first-port target, and the remaining gap is precisely the fraction a tiled GEMM and batched elementwise dispatches would claw back.

Calling that first-port number a guaranteed 6x would be dishonest. Calling it a speculative 6x when the weight-read floor, the dispatch floor, and the reference kernel all exist in an open-source sibling engine would be worse. The shape is well-understood. The execution risk is in the glue code around the kernel, not in the kernel itself.

## The tradeoff worth stating plainly

A column-batched DMMV is not a free lunch on three counts, and anyone planning this kind of port should state them out loud.

First, the shared-pipeline constraint is real until you ship specialization variants. On a single-pipeline build like the one sitting in ZINC today, raising `MAX_COLS` past 32 will take occupancy out of the per-token decode path. The port that delivers prefill speedups without regressing decode has to ship per-`num_cols` pipeline variants on day one, not later.

Second, this change does not touch the activation-quantization story. Activations are still read as FP32 against dequantized Q4_K weights, and the inner reduction is still a float accumulator. The [Q8_1 activation path](/blog/2026-04-19-why-q8-1-activations-are-the-next-rdna4-prefill-unlock) is an independent bet that stacks with column batching, not one that competes with it. Both ports together are the shape that gets a 35B MoE model into the interesting range on RDNA4.

Third, the elementwise dispatches that survive the first port will look disproportionately large once the projections stop dominating. RoPE, flash attention, SwiGLU, KV cache writes, and residual adds all stay per-token in the minimal version of the port, and they add up. The follow-up work is to move each of those to a batched shape too, which is already possible for most of them because the Metal side of ZINC has already done it in [`prefillBatched`](/blog/2026-04-20-metal-batched-prefill-38x-speedup-coherent-with-llama-cpp).

## What comes next

The ordering argument generalizes past ZINC. Any consumer-AMD local inference engine that is still running a per-token prefill loop is sitting on the same order-of-magnitude weight-read waste. The correct first move, before the tiled GEMM, before activation quantization, before a new memory layout, is to teach the DMMV kernel a `num_cols` axis and teach the pipeline cache to carry one variant per column count. That is a mechanical change. It ships in a week. It is worth more than any microshader rewrite the same week could produce.

For context on where this fits, the [Qwen3.5-35B prefill postmortem](/blog/2026-04-18-why-rdna4-prefill-for-qwen-3-5-is-stuck-at-25-tok-s) covers why 21 of 24 flat optimization cycles were flat for structural reasons, and the [RDNA4 4 to 27 tok/s deep dive](/blog/2026-03-29-the-shaders-standing-between-4-tok-s-and-27-tok-s) covers the earlier kernel work that set up the current floor. The next post in this arc will be the port's measured numbers once `dmmv_q4k_batch.comp` is wired into the prefill path under a specialization-constant pipeline set. If the weight-read and dispatch math above turns out to be within 20% of reality, this ordering argument holds. If not, the chart above will tell us precisely which bar moved and which did not, and the next correction is a better class of problem than the one we have now.
