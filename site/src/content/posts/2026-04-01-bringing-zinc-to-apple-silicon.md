---
title: "Bringing ZINC to Apple Silicon: from Vulkan to Metal in one engine"
date: "2026-04-01"
tags:
  - zinc
  - apple-silicon
  - metal
  - zig
  - llm-inference
  - gpu-kernels
  - qwen3-5
  - performance
keywords:
  - Apple Silicon LLM inference
  - Metal GPU inference
  - Metal Shading Language
  - Zig Metal backend
  - local LLM Apple Silicon
  - M1 M2 M3 M4 inference
  - ZINC Apple Silicon
  - Metal compute shaders
  - unified memory inference
  - zero-copy model loading
  - MoE Metal kernels
  - Vulkan to Metal port
excerpt: "ZINC now runs natively on Apple Silicon through a Metal backend built from scratch. Not a Vulkan translation layer. Not MLX. Hand-tuned MSL shaders, zero-copy model loading, and the same OpenAI-compatible API. This is the story of how we got there."
---

ZINC started as an AMD story. The whole pitch was simple: consumer GPUs have the bandwidth for real LLM inference, but the software stack ignores them. Build the shaders from scratch, tune them for the hardware, skip the heavyweight frameworks.

That pitch turns out to apply to more than just AMD.

Apple Silicon has unified memory, fast GPU cores, and sits inside tens of millions of machines. Yet the local inference options are either MLX (a Python framework that adds a layer of indirection between you and the GPU) or llama.cpp's Metal backend (which works, but treats Apple Silicon the same way it treats everything else: generically). Nobody has built an inference engine from scratch around what Metal actually offers.

So we did.

As of today, ZINC runs natively on every Apple Silicon Mac from M1 through M5. Same CLI, same chat UI, same OpenAI-compatible API. One `zig build`, one binary, no Python, no MLX, no framework overhead. The engine detects the platform at compile time and builds the right backend: Vulkan on Linux, Metal on macOS.

This post is the story of how that happened, what was harder than expected, and what design decisions made the difference.

## The first question: translate or rebuild?

When you already have a working Vulkan backend, the obvious path is to make it work on macOS. MoltenVK exists. You could, in theory, just run the same SPIR-V shaders through a Vulkan-to-Metal translation layer and call it a day.

We tried that for about five minutes before deciding against it.

The problem is not that MoltenVK does not work. The problem is that it forces you to pretend Apple Silicon is a discrete GPU with separate host and device memory, explicit staging buffers, and a Vulkan-shaped command model. Apple Silicon is none of those things. It has unified memory where CPU and GPU share the same physical pages. It has a command model designed around lightweight encoders, not heavyweight command buffers. It has `newBufferWithBytesNoCopy`, which lets you hand the GPU a pointer to memory you already own and skip the copy entirely.

If you translate Vulkan to Metal, you throw all of that away. You add overhead to simulate a memory model the hardware does not have, and you give up the one feature that makes Apple Silicon uniquely good for inference: the model weights can stay exactly where they are.

So instead of translating, we rebuilt. A native Metal backend with its own shaders, its own loader, and its own forward runtime. The shared parts of ZINC (tokenizer, GGUF parser, HTTP routes, model catalog) stay shared. Everything below the GPU abstraction line is new.

## The Objective-C question

ZINC is written in Zig. Metal is an Objective-C framework. Those two do not naturally mix.

The solution was a deliberate constraint: exactly one Objective-C file in the entire repo. `src/metal/shim.m` is a thin C-ABI wrapper around the Metal API. It exports plain C functions for device creation, buffer allocation, pipeline compilation, command recording, and dispatch. Everything else in the Metal backend is pure Zig calling that C interface.

This matters for two reasons. First, it keeps the Zig code clean. There is no `@cImport` pulling in Objective-C headers scattered across the codebase. There is no ARC happening in inference-hot code paths. The Zig side sees a set of opaque handles and straightforward function calls.

Second, it keeps the maintenance surface small. When Apple changes Metal APIs or adds new features, exactly one file needs to change. The rest of the engine does not know or care that Objective-C exists.

The shim is about 400 lines. It wraps `MTLDevice`, `MTLCommandQueue`, `MTLBuffer`, `MTLComputePipelineState`, `MTLCommandBuffer`, and `MTLComputeCommandEncoder`. That is enough to build a complete inference backend.

## Zero-copy model loading

On Vulkan, loading a model means: mmap the GGUF file, parse the metadata, allocate device-local GPU buffers, stage the weights through a transfer buffer, DMA them to VRAM. It is a well-understood pipeline, but it involves real copies and real upload time. A 20 GB model takes noticeable seconds to load.

On Metal, the same model loads like this: mmap the GGUF file, parse the metadata, wrap each tensor's memory region directly as an `MTLBuffer` using `newBufferWithBytesNoCopy`. Done. No copy. The GPU sees the same physical pages the CPU mapped.

This is the single biggest architectural difference between the two backends. The Vulkan loader and the Metal loader share almost no code below the GGUF parser, because the right design on each platform is fundamentally different.

There is one subtlety that took longer than expected. Metal requires buffer backing memory to be page-aligned. GGUF tensors are not always page-aligned. So the Metal loader wraps the page-aligned region containing each tensor and tracks the byte offset from the page boundary to the actual tensor start. Every shader dispatch then receives a buffer handle plus an offset. It is not complicated once you see it, but getting the alignment arithmetic wrong produces the kind of silent corruption that passes simple tests and fails on real models.

## Writing 31 shaders from scratch

The Vulkan backend has 24 GLSL compute shaders compiled to SPIR-V. The Metal backend has 31 MSL compute shaders. They are not translations of each other.

MSL and GLSL are different languages with different threading models. GLSL thinks in workgroups and invocations. MSL thinks in threadgroups, threads, simdgroups, and thread-position-in-grid. The GLSL shaders use Vulkan push constants for parameters. The MSL shaders use `setBytes:length:atIndex:` to inject small parameter structs inline.

More importantly, the optimization targets are different. On RDNA4, the hot path is wave64 with cooperative matrix operations. On Apple Silicon, the hot path is 32-lane simdgroups with threadgroup memory staging. The kernels that dominate decode time (DMMV, the quantized matrix-vector multiply) have completely different tuning strategies on each platform.

The Q4_K DMMV family got the deepest treatment on Metal. It has general-purpose variants, K-dimension specializations for the common 2048 case, LM-head specializations with wider parallelism, and MoE-specific batched variants. Each one stages the input activation vector in threadgroup memory and reuses it across multiple output rows. That pattern maps naturally to Apple Silicon's memory hierarchy.

Q5_K and Q6_K started as simpler one-thread-per-row kernels. Those were enough for correctness, but they left performance on the table. The real versions now use the same staged-input pattern as Q4_K, which matters because the target model uses mixed quantization across its MoE expert tensors.

## The MoE problem

Everything up to this point was tractable. Build a Metal backend, write some shaders, make inference work. The part that turned out to be genuinely hard was Mixture of Experts.

The Qwen3.5-35B-A3B model is a hybrid architecture: some layers are pure attention, some are SSM (state-space model), and most of the compute lives in MoE layers with 256 experts where 8 are active per token. The MoE layers dominate decode time.

The naive MoE path works like this: compute router logits, read them back to the CPU, do top-k selection, then dispatch each selected expert individually. Gate projection, up projection, SwiGLU activation, down projection, accumulate. For 8 experts per layer across 40 MoE layers, that is a lot of individual GPU dispatches per token.

On the early Metal backend, a profiled 256-token decode run showed 21,141 command buffer commits. Roughly 81 submits per decode step. The `submit/wait` overhead was dominating total traced time. The actual GPU kernels were fast enough. The problem was that we were spending more time launching them than running them.

The fix was a GPU-routed batched MoE path. Instead of reading router logits back to the CPU, a `softmax_topk` shader runs entirely on the GPU, writes the selected expert IDs and weights into a compact routing buffer, and then batched expert projections process all 8 experts in a single dispatch per matrix. The weighted accumulation folds into the residual add without a separate pass.

After that change, the same class of run dropped to 261 command buffer commits. One shared command per decode step. That was the architectural unlock that moved local decode from about 20 tok/s to roughly 30 tok/s.

## Mixed quantization: the real blocker

Getting the batched MoE path working for Q4_K experts was the first milestone. But the target model does not use Q4_K everywhere. Some layers have `q4_k / q4_k / q5_k` expert tensors. Others have `q5_k / q5_k / q6_k`. Any unsupported quantization in any expert tensor of any layer forces that layer back to the slow per-expert fallback path. And if even one layer falls back, the shared-command fast path is disabled for the entire decode step.

This is why Q5_K and Q6_K MoE kernels exist. Not because those quantization types are individually important, but because missing support for them in the batched path blocked the optimization that actually mattered.

Once mixed-quant MoE support was complete, the shared-command path became legal across all MoE layers in the model. That is when the Metal backend started posting real numbers.

## Runtime shader compilation

One decision that surprised people: the Metal backend compiles shaders from MSL source at startup, not from precompiled metallib bundles.

On the Vulkan side, GLSL shaders are compiled to SPIR-V during the build. The binary ships with `.spv` files. On Metal, the binary ships with `.metal` source files and compiles them into `MTLComputePipelineState` objects when the engine starts.

This adds a few seconds to startup, which is not ideal for production. But it made the initial bring-up dramatically simpler. No metallib packaging pipeline. No Xcode project. No separate compilation step. Edit a `.metal` file, rebuild the Zig binary, and the new shader is live. For a period where we were iterating on 31 shaders simultaneously, that feedback loop mattered more than cold-start latency.

The plan is to add precompilation later. The runtime compilation path will stay as a fallback and development convenience.

## Profiling without GPU timestamps

Apple does not expose the same per-dispatch GPU timestamp mechanism that Vulkan does. On the Vulkan backend, you can instrument individual shader dispatches and measure their GPU execution time. On Metal, the engine uses request-scoped CPU-side profiling that tracks:

Recording time. Router CPU time. Submit-and-wait time. GPU-routed MoE recording time. Fallback MoE recording time. Sampling time. Total step time.

This is not as precise as per-dispatch GPU timestamps, but it was enough to identify the real bottleneck. The profiling data is what showed that `submit/wait` dominated traced time and that the MoE command fragmentation was the problem, not tokenizer code or output decoding. Without it, the mixed-quant MoE work would have been guesswork.

## The server just works

One of the better early decisions was to keep the HTTP route layer backend-agnostic. `src/server/runtime.zig` acts as a thin adapter that aliases the right loader, forward runtime, model manager, and sampling API based on which backend is active. The actual route handlers in `routes.zig` never fork into "Metal vs Vulkan" branches.

That means the Metal server serves the same built-in chat UI at `/`, the same `/v1/chat/completions` endpoint with SSE streaming, the same `/health` and `/v1/models` endpoints, and the same managed-model workflow. If you have a client that works with the AMD version of ZINC, it works identically with the Apple Silicon version.

The one current difference: sampling controls. The Metal path uses greedy decoding (temperature=0 argmax on CPU). The Vulkan path supports temperature, top-p, top-k, and repetition penalty via GPU-side sampling. Adding the same controls to Metal is straightforward work but has not been the priority.

## What the numbers look like

On an M1 Pro with 32 GB unified memory running `Qwen3.5-2B-Q4_K_M`:

- CLI plain decode: **~17 tok/s**
- Chat template: **~17 tok/s**

These are early numbers. The M1 Pro is not the fastest Apple Silicon part, and the Metal kernels have not yet received the same depth of tuning that the RDNA4 Vulkan path has had. M4 and M5 benchmarks are coming.

For context, the same model on the AMD RDNA4 test node (Radeon AI PRO R9700, 32 GB, 576 GB/s) runs at about 27 tok/s. The M1 Pro has roughly 200 GB/s memory bandwidth. The numbers are in the right ballpark for the hardware.

## What is next

The Metal backend is functional and shipping. It passes the same test suite, serves the same API, and handles the same models. But there is real optimization work ahead:

**Prefill.** The current Metal prefill path processes one token at a time through the full decode loop. A proper batched prefill would dramatically improve time-to-first-token on longer prompts.

**Sampling.** Adding temperature, top-p, and repetition penalty to the Metal path so it matches the Vulkan server's capabilities.

**M4 and M5 tuning.** The current kernels work across all Apple Silicon generations, but M4 and especially M5 (with its new TensorOps / Neural Accelerator path) deserve generation-specific kernel variants.

**Precompiled shaders.** Switching from runtime MSL compilation to precompiled metallib bundles for faster cold starts.

**More models.** The current validation is on Qwen3.5. Expanding to more architectures is a matter of adding the right tensor mappings and testing.

## The bigger picture

When ZINC started, the thesis was narrow: AMD consumer GPUs are ignored by the mainstream AI stack, and someone should fix that. Adding Apple Silicon support does not change that thesis. It broadens it.

The real thesis is: local GPUs are underused for inference, and the reason is software, not hardware. AMD RDNA4 has the bandwidth. Apple Silicon has the unified memory. Both have the compute. What they lack is an engine that takes each platform seriously enough to build shaders from scratch, tune them for the actual hardware, and ship a single binary that just works.

That is what ZINC is becoming. Two GPU backends, one engine, no heavyweight frameworks. If you have an AMD GPU or an Apple Silicon Mac, you have an inference machine. The software just needs to respect the hardware.

`zig build`, `zinc chat`, and you are running a 2B model locally. That is the bar we set, and that is what ships today.
