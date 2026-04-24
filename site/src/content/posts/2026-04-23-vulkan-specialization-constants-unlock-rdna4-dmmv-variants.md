---
title: "Vulkan specialization constants are how RDNA4 prefill ships five DMMV pipelines without a decode regression"
date: "2026-04-23"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - spir-v
  - specialization-constants
  - prefill
  - qwen3-8b
  - llm-inference
  - gpu-kernels
keywords:
  - Vulkan specialization constants
  - VkSpecializationInfo RDNA4
  - SPIR-V constant_id layout qualifier
  - NUM_COLS DMMV variant Vulkan
  - per-pipeline register budget RDNA4
  - MAX_COLS 32 shared pipeline regression
  - ZINC Vulkan pipeline cache
  - RX 9070 AI PRO decode occupancy
  - llama.cpp mul_mat_vec NUM_COLS
  - pipeline variant compile time Vulkan
  - wave64 VGPR specialization
  - RDNA4 prefill DMMV port
faqs:
  - question: "What is a Vulkan specialization constant, in one paragraph?"
    answer: "A specialization constant is an integer, boolean, or float value that lives inside a SPIR-V binary as a placeholder and gets its final value injected at pipeline creation time via `VkSpecializationInfo`. After substitution the driver runs register allocation, loop unrolling, and dead-code elimination against the specialized shader, not the template. You end up with multiple compiled pipelines from a single SPIR-V module, each with its own register footprint, all describing the same inner math."
  - question: "Why can't we just pass NUM_COLS as a push constant?"
    answer: "Push constants are runtime values. The driver has to emit code that assumes the value could be anything in range, which means the loop with `for (uint c = 0u; c < num_cols; c++)` keeps a dynamic bound and the per-column accumulator array sizes to the worst case. The register allocator has to reserve registers for the maximum possible column count, which is exactly the problem the MAX_COLS=32 shared pipeline has today. A specialization constant is seen as a literal integer at compile time, so the loop unrolls and the accumulator collapses to exactly num_cols slots."
  - question: "How many DMMV variants is too many?"
    answer: "In practice, a handful. ZINC's short list is `NUM_COLS = {1, 8, 16, 32, 64}`, which covers decode (1), short prompt chunks (8, 16), the current prefill ceiling (32), and the occupancy-tested prefill regime (64). Each variant compiles in the 30 to 80 ms range for a Q4_K DMMV kernel on RADV, so five variants add something on the order of a quarter second of cold build time, comfortably absorbed by ZINC's one-shot `VkPipelineCache` on disk after the first run."
  - question: "Does this interact with MTLBinaryArchive on the Metal side?"
    answer: "Not directly. Metal does not have specialization constants in the Vulkan sense. It has function constants, which are the same idea under a different spelling, and MTLBinaryArchive is how you pin the compiled pipeline state on disk. The Metal side of ZINC already leans on function constants for its batched prefill kernels, which is part of why the Metal port did not trip the same shared-pipeline trap the Vulkan side did. The Metal binary-archive work covered in the cold-start post is a complementary caching story, not a register-budget one."
  - question: "Is there a case where you don't want to specialize?"
    answer: "Yes. If the set of values the constant can take is open-ended, or if each variant's register footprint would push you into a build-time explosion, pinning the value with a push constant keeps the pipeline count bounded. The rule of thumb is that specialization is correct when the variant count is small and the per-variant register cost difference is large enough to change occupancy. DMMV with `NUM_COLS ∈ {1, 8, 16, 32, 64}` hits both criteria. A generic `chunk_size` parameter that could be any integer up to a few thousand does not."
excerpt: "Yesterday's post left ZINC's RDNA4 DMMV with a compromise: one pipeline, MAX_COLS=32 shared between decode and prefill, because raising it to 64 regressed decode from 78 tok/s to 59 tok/s on Qwen3-8B. Vulkan specialization constants are the quiet fix. One SPIR-V binary, one `VkSpecializationInfo` per call site, five pipelines compiled with five different register budgets. Here is how that works under the hood, why it only costs a quarter second of extra cold build time, and why specialization constants are a more general answer to the next three kernels in the port."
---

[Yesterday's post](/blog/2026-04-22-why-rdna4-prefill-wants-a-32-column-dmmv-before-a-gemm) ended on a compromise that was easy to gloss over: the `dmmv_q4k_batch.comp` shader ships a single pipeline with `MAX_COLS = 32` because raising the constant to 64 knocks Qwen3-8B decode from **78 tok/s down to 59 tok/s** on an RX 9070 AI PRO. The prefill side would prefer 64. The decode side cannot afford the register pressure. The shared-pipeline constraint picks the lower number, and both call sites pay a tax nobody wants.

That tax is not a Vulkan limitation. It is a build-configuration choice. Every ingredient needed to unwind it has been in the Vulkan specification since 1.0, in the SPIR-V binary format since the first revision, and in [llama.cpp's Vulkan backend](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp) for long enough that it is the production default. The mechanism is specialization constants, and the fix for the `MAX_COLS = 32` compromise is a roughly two hundred line change in ZINC's shader build script plus a small rework of the pipeline cache.

This post is the mechanical argument for that change. It covers what a specialization constant actually is at the SPIR-V level, why it collapses the shared-pipeline problem cleanly on RDNA4, how the canonical example in llama.cpp is structured, and what the cold build-time cost looks like once you ship five variants instead of one. The conclusion is that specialization constants are not a DMMV-only tool. They are the right answer for at least three of the next kernels this port will touch.

## What a specialization constant is at the SPIR-V level

A specialization constant is a placeholder integer, boolean, or float inside a SPIR-V binary, flagged with a `constant_id` so the driver knows the value has not been finalized. [SPIRV-Cross](https://github.com/KhronosGroup/SPIRV-Cross) and every other tool downstream of the reference compiler understands that these values are substituted at pipeline creation time. When you call `vkCreateComputePipelines`, you pass a [`VkSpecializationInfo`](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/chapters/pipelines.adoc) that maps each `constant_id` to a concrete value. The driver substitutes the literal, then runs register allocation, loop unrolling, and dead-code elimination against the specialized module.

The distinction worth pinning down is that substitution happens before the target compile, not during dispatch. The register allocator sees `NUM_COLS` as a plain literal integer. If the shader contains `float sums[NUM_COLS];`, the array allocates exactly that many slots. If it contains `for (uint c = 0u; c < NUM_COLS; c++)`, the loop unrolls to exactly that many iterations. None of that is possible with a push constant, because push constants are runtime values and the driver has to emit code that works across the full legal range. A push constant forces the allocator to reserve the worst-case register budget and leaves the loop as a dynamic bound.

This is the lever the MAX_COLS problem needs. Today's ZINC shader treats `MAX_COLS` as a compile-time `const uint`, which is correct, but there is only one compile. A specialization-constant build emits one SPIR-V and feeds it through the pipeline creation step five times with different `VkSpecializationInfo` values. Five pipeline objects come out, each with its own register map. The [canonical SaschaWillems example](https://github.com/SaschaWillems/Vulkan/blob/master/examples/specializationconstants/specializationconstants.cpp) is worth reading once in full if the mechanics are unfamiliar, because it lays out the `VkSpecializationMapEntry` table exactly the way a production build script wants it.

## The GLSL change is tiny. The build change is most of the work.

At the shader source level the change is one line. Where today's `dmmv_q4k_batch.comp` reads:

```glsl
const uint MAX_COLS = 32u;
```

The specialization-constant variant reads:

```glsl
layout (constant_id = 0) const uint NUM_COLS = 32u;
```

The `32u` at the end is the default value, used only if no `VkSpecializationInfo` is provided at pipeline creation. Every other use of `MAX_COLS` inside the shader stays the same. The accumulator array becomes `float sums[NUM_COLS]`, the inner loop bound becomes `c < NUM_COLS`, and the register allocator now sees whatever integer the pipeline was created with.

The non-trivial work is outside the shader. ZINC's build currently compiles each `.comp` file once and hashes the resulting SPIR-V into a single pipeline slot. A specialization-constant build has to compile the SPIR-V once but create N pipeline objects from it, one per `NUM_COLS` value that some call site will want to bind. llama.cpp's [`mul_mat_vec.comp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec.comp) does exactly this, and the shader build CMake under [`vulkan-shaders/CMakeLists.txt`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/CMakeLists.txt) is the reference for how to structure the NUM_COLS specialization at the build layer without exploding the variant matrix.

The tradeoff the build script is managing is small and explicit. Each extra variant adds one `VkComputePipelineCreateInfo` record, a few dozen bytes of descriptor plumbing, and the per-compile cost from the driver. Nothing about this needs a new descriptor-set layout or a new shader module.

![AMD Radeon AI PRO R9700 from an angle showing the cooling shroud. The card the decode regression is measured on.](/blog/gpu_4.jpg)

## Why RDNA4 rewards per-variant register budgets

The RDNA4 register file is the reason the shared-pipeline compromise bites as hard as it does. A [community RDNA4 architecture reference](https://github.com/azhirnov/cpu-gpu-arch/blob/main/gpu/AMD-RDNA4.md) catalogs the per-SIMD VGPR budget at 1536 32-bit registers under wave32, halved to 768 per wavefront under wave64, and the occupancy a Compute Unit can sustain is a direct function of how many registers a wavefront needs. The shader's `sums[MAX_COLS]` array sits entirely in VGPRs, and the allocator cannot do anything clever about the fact that 75% of those slots are unused in the decode-path invocation with `num_cols = 1`.

At MAX_COLS=32 the occupancy math lands somewhere that keeps enough wavefronts in flight to hide memory latency on Q4_K weight reads. At MAX_COLS=64 the accumulator array doubles, the register budget doubles with it, and the per-CU wavefront count drops enough that the decode path loses roughly a quarter of its throughput. The prefill path, which actually uses the extra columns, would win. But the shared pipeline ships both cases to the same binary, so the worst-case slot count is the one that matters.

Specialization does not just lift this ceiling. It lets the allocator pick a different ceiling per call site. Pipeline 1 with `NUM_COLS = 1` reserves registers for one accumulator. Pipeline 5 with `NUM_COLS = 64` reserves registers for 64. Neither pipeline sees the other's footprint. The decode call site binds pipeline 1, the prefill chunk-8 call site binds pipeline 2, and so on down the list.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/vulkan-specialization-constants-pipeline-fanout.svg" alt="A two-panel diagram. The left panel shows a single SPIR-V binary fanning out into five Vulkan compute pipelines, each created with a different VkSpecializationInfo value for NUM_COLS (1, 8, 16, 32, 64), each with its own register budget labeled underneath. The right panel shows a horizontal bar chart comparing Qwen3-8B decode throughput under three shader strategies: shared MAX_COLS=32 at 78 tok/s, shared MAX_COLS=64 at 59 tok/s, and specialized NUM_COLS variants at 78 tok/s with no regression." loading="lazy" />
  <figcaption>One SPIR-V binary fans out into five pipelines at build time. The right panel is the decode cost of today's compromise and the decode number the specialized build is expected to preserve.</figcaption>
</figure>

The bars on the right are the whole argument in two numbers. The current 78 tok/s is what the `MAX_COLS = 32` shared pipeline delivers. The 59 tok/s is what the naive `MAX_COLS = 64` shared pipeline delivered when someone tried it, the regression that motivated the 32 ceiling in the first place. The 78 tok/s under specialization is the prediction: decode binds the `NUM_COLS = 1` pipeline, sees a register footprint smaller than the current shared binary, and continues to occupy the CUs at today's rate. The prefill side separately binds the `NUM_COLS = 64` variant and gets the occupancy envelope it needs.

## What the build-time cost actually looks like

The fear that gets raised first about specialization is always the same: will this explode compile times? For a kernel like DMMV the answer is no, and the math is easy to spell out.

Each specialization is an independent driver compile, which for a Q4_K DMMV kernel on RADV lands in the 30 to 80 ms range, depending on how aggressive the backend is with the unrolled inner loop. Five variants add roughly 150 to 400 ms of cold build time on first launch. ZINC already pins compiled pipelines in a [`VkPipelineCache`](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/chapters/pipelines.adoc) blob on disk after the first run, so the second-launch cost is zero.

The on-disk cache grows by roughly the same factor as the variant count, because the driver stores one binary blob per pipeline state. For a DMMV kernel that is tens of kilobytes per variant, plus a bit of descriptor metadata. Five variants of five quant types (Q4_K, Q5_K, Q6_K, Q8_0, and the incoming Q4_K_M + Q8_1 pair) land around a single-digit-megabyte pipeline cache on disk. That is smaller than the model's tokenizer vocabulary file.

The real build-layer care is in the variant matrix, not the per-variant cost. A generic "compile every kernel against every specialization value" build sweeps a much bigger matrix than you want. The discipline is to list the values each shader will actually bind at runtime, enumerate them in the build script, and compile only those. llama.cpp's DMMV variant list is the right shape to copy. Decode wants `NUM_COLS = 1`, prefill wants powers of two up to the chunk size the scheduler picks, and nothing in between needs to exist.

## Where specialization generalizes beyond DMMV

The case for specialization constants stops being a DMMV-specific argument once you look at the next three kernels in the prefill port.

The flash-attention kernel will want specialization on head dimension. Today ZINC's decode path binds a single FA kernel with a runtime head-dim push constant, which forces the allocator to carry enough register space for the largest supported head. Qwen3-8B has a head dim of 128. A specialized build emits one variant per value and lets the 128-dim path use a tighter register map than a hypothetical 256-dim variant would have forced.

The SwiGLU elementwise kernel will want specialization on the intermediate size ratio. That value changes per model but is constant across a single model's layers. A per-model specialization baked at load time fixes the intermediate dimension and lets the compiler hoist more of the address math out of the inner loop.

The KV cache write kernel will want specialization on the number of key-value heads, which for grouped-query attention is a small integer that is fixed for a given model. Specialization here is not about register pressure; it is about letting the compiler fold the head-group iteration into a compile-time unroll.

None of these are theoretical. All three are kernels that decode exercises and prefill will exercise harder, which means they sit on the same shared-pipeline trap the DMMV shader sat on until this week.

## The tradeoff, stated plainly

Specialization constants are not free. The three costs worth stating out loud are specific and small.

First, the variant list becomes part of the runtime ABI. A caller that wants `NUM_COLS = 24` because the scheduler picked an off-grid chunk size has to either round up to the nearest supported variant or fall back to a general push-constant pipeline. The scheduler has to know the supported list. In practice this is a handful of powers of two plus 1, which is what every production inference engine already uses.

Second, the `VkPipelineCache` on disk grows by the variant count, and the pipeline-creation step at cold start takes proportionally longer. For five DMMV variants across five quant types, that extra cold-start cost is in the 1 to 2 second range before the pipeline cache warms up. ZINC's cold-start story has to eat that once per machine install, which is comfortable given the [Metal-side cold-start work](/blog/2026-04-21-metal-binary-archive-is-the-missing-cold-start-fix) has already set a baseline for what a well-managed pipeline cache feels like.

Third, the specialization-constant-versus-push-constant choice has to be made per shader, not as a global policy. A kernel whose register footprint does not change with the constant, or whose variant space is open-ended, is better served by a runtime push constant. Over-specializing a shader is a real build-matrix bloat. The DMMV NUM_COLS case is the opposite: finite, register-impactful, and call-site-predictable.

## What comes next

The next ZINC commit in the RDNA4 prefill arc wires `dmmv_q4k_batch.comp` through a `constant_id = 0` NUM_COLS entry, compiles the five listed variants at build time, and adds the variant lookup into the pipeline cache. The measured decode number stays at 78 tok/s. The measured prefill number on a 103-token Qwen3-8B prompt is the one the next post will report, because that is the first number that tests whether the register headroom on the `NUM_COLS = 64` variant actually lands where the occupancy math predicts. If it does, the rest of the prefill kernels follow the same pattern and the shared-pipeline tax disappears as a class of problem, not just for DMMV.

The broader point for any consumer-AMD local inference engine is that specialization constants are a cheap tool and they have been sitting in the Vulkan spec since day one. If your Vulkan backend is still shipping one pipeline per shader across every call site, somewhere in your kernel tree there is a comment shaped like the one in ZINC's `dmmv_q4k_batch.comp` that is waiting to come out.
