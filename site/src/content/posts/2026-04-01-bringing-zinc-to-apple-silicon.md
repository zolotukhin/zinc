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

Twenty-one thousand command buffer commits per decode run. That was the number staring back at us from the Metal profiler when we first got a 35B model running on Apple Silicon. The GPU kernels were fast. The engine was spending most of its time asking the GPU to please start working.

Fixing that took rewriting how Mixture of Experts dispatch works on Metal, building specialized kernels for three quantization types we did not originally plan to support, and rethinking assumptions about command recording that worked fine on Vulkan but fell apart on Apple Silicon. The result: 261 commits per run, a 50% throughput jump, and a Metal backend that now ships as a first-class platform alongside AMD.

This is the story of how ZINC went from "AMD only" to "AMD and Apple Silicon" without turning into two separate engines.

<figure class="diagram-card diagram-wide">

| | AMD (Vulkan) | Apple Silicon (Metal) |
|---|---|---|
| **Memory model** | Discrete VRAM + staging DMA | Unified, shared pages |
| **Shader language** | GLSL 4.60 → SPIR-V | MSL (Metal Shading Language) |
| **Shaders shipped** | 24 compute shaders | 31 compute shaders |
| **Model loading** | mmap → staging buffer → GPU DMA | mmap → `newBufferWithBytesNoCopy` |
| **Threading model** | wave64, cooperative matrix | 32-lane simdgroups, threadgroup staging |
| **Command model** | Pre-recorded command buffers | Lightweight encoders, inline dispatch |

  <figcaption>The two backends share tokenizer, GGUF parser, HTTP routes, and model catalog. Everything below the GPU abstraction line is platform-native.</figcaption>
</figure>

## The first question: translate or rebuild?

When you already have a working Vulkan backend, the obvious path is MoltenVK. Run the same SPIR-V shaders through a translation layer. Call it a day.

We decided against it in about five minutes.

MoltenVK forces you to pretend Apple Silicon is a discrete GPU. Separate host and device memory. Explicit staging buffers. A Vulkan-shaped command model. Apple Silicon is none of those things. It has unified memory where CPU and GPU share the same physical pages. It has `newBufferWithBytesNoCopy`, which lets you hand the GPU a pointer to memory you already own and skip the copy entirely.

Translating Vulkan to Metal throws away the one feature that makes Apple Silicon uniquely good for inference: the model weights can stay exactly where they are.

<figure class="diagram-card diagram-wide">

```
┌─────────────────────────────────────────────────────────────┐
│                        ZINC Engine                          │
├──────────────────────┬──────────────────────────────────────┤
│   Shared layers      │  Tokenizer, GGUF parser, HTTP API,  │
│                      │  model catalog, chat UI, sessions    │
├──────────────────────┼──────────────────────────────────────┤
│   gpu/interface.zig  │  Compile-time backend switch         │
├───────────┬──────────┴───────────┬──────────────────────────┤
│  Vulkan   │                      │  Metal                   │
│  backend  │                      │  backend                 │
├───────────┤                      ├──────────────────────────┤
│ instance  │                      │ device.zig               │
│ buffer    │                      │ buffer.zig               │
│ pipeline  │                      │ pipeline.zig             │
│ command   │                      │ command.zig              │
│ gpu_detect│                      │ shim.m (400 lines ObjC)  │
├───────────┤                      ├──────────────────────────┤
│ 24 GLSL   │                      │ 31 MSL shaders           │
│ → SPIR-V  │                      │ (runtime compiled)       │
├───────────┤                      ├──────────────────────────┤
│ forward   │                      │ forward_metal.zig        │
│ .zig      │                      │                          │
└───────────┘                      └──────────────────────────┘
```

  <figcaption>Backend selection happens at compile time. The inactive backend is not compiled into the binary. macOS builds Metal. Linux builds Vulkan.</figcaption>
</figure>

So instead of translating, we rebuilt. A native Metal backend with its own shaders, its own loader, and its own forward runtime. The shared parts stay shared. Everything below the abstraction line is new.

## The Objective-C question

ZINC is written in Zig. Metal is an Objective-C framework. Those two do not naturally mix.

The solution was a deliberate constraint: exactly one Objective-C file in the entire repo. `src/metal/shim.m` is a thin C-ABI wrapper around the Metal API. It exports plain C functions for device creation, buffer allocation, pipeline compilation, command recording, and dispatch. Everything else in the Metal backend is pure Zig calling that C interface.

The shim is about 400 lines. It wraps `MTLDevice`, `MTLCommandQueue`, `MTLBuffer`, `MTLComputePipelineState`, `MTLCommandBuffer`, and `MTLComputeCommandEncoder`. That is enough to build a complete inference backend.

This keeps the Zig code clean (no ARC in hot paths, no Objective-C headers scattered around) and the maintenance surface small (Apple API changes touch one file).

## Zero-copy model loading

This is the single biggest architectural difference between the two backends.

<figure class="diagram-card diagram-wide">

```
Vulkan model loading (discrete GPU):

  ┌──────────┐    ┌──────────────┐    ┌───────────┐    ┌──────────┐
  │ GGUF     │───▶│ Staging      │───▶│ DMA       │───▶│ Device   │
  │ file     │    │ buffer (CPU) │    │ transfer  │    │ VRAM     │
  │ (mmap)   │    │              │    │           │    │ (GPU)    │
  └──────────┘    └──────────────┘    └───────────┘    └──────────┘
                  ▲ copy 1                ▲ copy 2


Metal model loading (unified memory):

  ┌──────────┐    ┌──────────────────────────────────────────────┐
  │ GGUF     │───▶│ newBufferWithBytesNoCopy                     │
  │ file     │    │                                              │
  │ (mmap)   │    │ Same physical pages. GPU reads mmap directly.│
  └──────────┘    └──────────────────────────────────────────────┘
                  ▲ zero copies
```

  <figcaption>On Vulkan, model weights travel through two copies before the GPU sees them. On Metal, the GPU reads the mmap'd file directly. A 20 GB model loads in the time it takes to parse GGUF metadata.</figcaption>
</figure>

On Vulkan, loading a model means: mmap the GGUF file, allocate device-local GPU buffers, stage weights through a transfer buffer, DMA them to VRAM. Real copies, real upload time.

On Metal: mmap the GGUF file, wrap each tensor's memory region directly as an `MTLBuffer`. Done. The GPU sees the same physical pages the CPU mapped.

There is one subtlety. Metal requires buffer backing memory to be page-aligned. GGUF tensors are not always page-aligned. So the Metal loader wraps the page-aligned region containing each tensor and tracks the byte offset to the actual tensor start. Every shader dispatch receives a buffer handle plus an offset. Getting the alignment arithmetic wrong produces silent corruption that passes simple tests and fails on real models. That one took a while.

## Writing 31 shaders from scratch

The Vulkan backend has 24 GLSL compute shaders compiled to SPIR-V. The Metal backend has 31 MSL compute shaders. They are not translations of each other.

MSL and GLSL have different threading models. GLSL thinks in workgroups and invocations. MSL thinks in threadgroups, threads, simdgroups, and thread-position-in-grid. The optimization targets are different too. On RDNA4, the hot path is wave64 with cooperative matrix operations. On Apple Silicon, the hot path is 32-lane simdgroups with threadgroup memory staging.

<figure class="diagram-card diagram-wide">

| Shader family | Count | Purpose |
|---|---|---|
| **DMMV (quantized matmul-vec)** | 12 | Q4_K, Q5_K, Q6_K, Q8_0, F16, F32 + LM-head and K-dim specializations |
| **MoE DMMV (batched experts)** | 5 | Q4_K, Q5_K, Q6_K batched + K-dim specializations |
| **Elementwise / fusion** | 8 | RMS norm, SwiGLU, RoPE, sigmoid gating, scale-accumulate |
| **Attention** | 3 | Flash attention, KV-cache write, deinterleave |
| **SSM (state-space model)** | 3 | Conv1d, delta-net, gated norm |

  <figcaption>31 MSL compute shaders written from scratch for Apple Silicon. The Q4_K family has the deepest tuning because it appears on the hottest decode-side paths.</figcaption>
</figure>

The Q4_K DMMV family got the deepest treatment. It has general-purpose variants, K-dimension specializations for the common 2048 case, LM-head specializations with wider parallelism, and MoE-specific batched variants. Each stages the input activation vector in threadgroup memory and reuses it across multiple output rows.

Q5_K and Q6_K started as simpler one-thread-per-row kernels. Correctness was easy. Performance was not. The real versions use the same staged-input pattern as Q4_K, because the target model uses mixed quantization across MoE expert tensors. More on why that matters below.

## The MoE problem

Everything up to this point was tractable. Build a Metal backend, write some shaders, make inference work. The part that turned out to be genuinely hard was Mixture of Experts.

The Qwen3.5-35B-A3B model has 256 experts per MoE layer with 8 active per token, across 40 MoE layers. That is where most of the compute lives.

The naive path: compute router logits, read them back to CPU, select top-k experts, dispatch each expert individually. Gate projection, up projection, SwiGLU, down projection, accumulate. Repeat for 8 experts across 40 layers per token.

<figure class="diagram-card diagram-wide">

```
Before: per-expert dispatch (naive path)
──────────────────────────────────────────
Per token, per MoE layer:
  [router] → CPU readback → [expert 1] → [expert 2] → ... → [expert 8] → [accumulate]
                ▲                ▲            ▲                   ▲
          GPU→CPU sync      8 × (gate + up + swiglu + down) = 32 dispatches per layer

  40 MoE layers × ~81 submits/step = 21,141 command buffer commits per decode run
  Result: ~20 tok/s (GPU kernels fast, submit/wait overhead dominates)


After: GPU-routed batched MoE
──────────────────────────────
Per token, per MoE layer:
  [softmax_topk on GPU] → [batched gate+up] → [swiglu_batched] → [batched down] → [moe_weighted_acc]
                                    ▲                                      ▲
                            All 8 experts in one dispatch            Fused residual add

  Entire decode step in 1 shared command buffer = 261 commits per run
  Result: ~30 tok/s
```

  <figcaption>The architectural unlock: moving expert routing to the GPU and batching all 8 expert projections into single dispatches eliminated command-buffer fragmentation. This was worth more than any individual kernel optimization.</figcaption>
</figure>

On the early Metal backend, a profiled 256-token run showed **21,141 command buffer commits**. Roughly 81 submits per decode step. The GPU was fast. We were spending more time launching kernels than running them.

The fix: a GPU-routed batched MoE path. A `softmax_topk` shader runs entirely on the GPU, writes selected expert IDs and weights into a routing buffer, and batched projections process all 8 experts per dispatch. The weighted accumulation folds into the residual add.

After that change: **261 commits**. One shared command per step. Decode went from about 20 tok/s to roughly 30 tok/s.

## Mixed quantization: the real blocker

Getting batched MoE working for Q4_K experts was the first milestone. But the target model does not use Q4_K everywhere. Some layers use `q4_k / q4_k / q5_k`. Others use `q5_k / q5_k / q6_k`.

Here is the catch: any unsupported quantization in any expert tensor of any MoE layer forces that layer back to the per-expert fallback. And if even one layer falls back, the shared-command fast path is disabled for the entire decode step.

This is why Q5_K and Q6_K MoE kernels exist. Not because those types are individually important. Because missing them in the batched path blocked the optimization that actually mattered.

Once mixed-quant support was complete, the shared-command path became legal across all 40 MoE layers. That is when the Metal backend started posting real numbers.

## Runtime shader compilation

One decision that surprised people: the Metal backend compiles shaders from MSL source at startup, not from precompiled metallib bundles.

On Vulkan, GLSL shaders compile to SPIR-V during the build. On Metal, the binary ships with `.metal` source files and compiles them into `MTLComputePipelineState` objects at launch.

This adds a few seconds to startup. But it made bring-up dramatically simpler. No metallib packaging pipeline. No Xcode project. Edit a `.metal` file, rebuild the Zig binary, and the new shader is live. When iterating on 31 shaders simultaneously, that feedback loop matters more than cold-start latency.

Precompilation is planned. The runtime path stays as fallback and development convenience.

## Profiling without GPU timestamps

Apple does not expose the same per-dispatch GPU timestamp mechanism that Vulkan does. The Metal engine uses request-scoped CPU-side profiling:

<figure class="diagram-card diagram-wide">

| Metric | What it tells you |
|---|---|
| **Command recording time** | How long the CPU spends building GPU work |
| **Submit/wait time** | Time blocked waiting for GPU completion |
| **Router CPU time** | Expert selection overhead (GPU-routed path eliminates this) |
| **GPU-routed MoE time** | Batched expert dispatch recording |
| **Fallback MoE time** | Per-expert dispatch recording (the slow path) |
| **Sampling time** | CPU-side argmax |
| **Total step time** | End-to-end per-token wall time |

  <figcaption>Not as precise as per-dispatch GPU timestamps, but enough to find the bottleneck. The profiling data is what proved submit/wait overhead was the problem and directly informed the batched MoE work.</figcaption>
</figure>

This is what showed that `submit/wait` dominated traced time and that MoE command fragmentation was the problem. Without it, the mixed-quant work would have been guesswork.

## The server just works

One of the better early decisions: keep the HTTP route layer backend-agnostic. `src/server/runtime.zig` aliases the right loader, forward runtime, model manager, and sampling API based on the active backend. The route handlers in `routes.zig` never branch on "Metal vs Vulkan."

The Metal server serves the same chat UI at `/`, the same `/v1/chat/completions` with SSE streaming, the same `/health` endpoint, and the same managed-model workflow. If your client works with the AMD version, it works identically on Apple Silicon.

## What the numbers look like

On a **Mac Studio (Mac16,9)** with **Apple M4 Max**, **40-core GPU**, and **64 GB unified memory** running `Qwen3.5-35B-A3B-UD-Q4_K_XL`:

| Metric | Result |
|---|---|
| `bench-metal` plain decode, 256 generated tokens, 1 warmup, 3 measured runs | **35.61 tok/s avg**, `35.58 tok/s` median, `28.1 ms/tok` |
| `bench-metal` prefill on the same run set | **36.2 tok/s avg**, `36.6 tok/s` median |
| Raw HTTP `POST /v1/completions`, `max_tokens=256`, `concurrency=1` | **34.74 tok/s**, `7.37s` avg latency |
| Raw HTTP `POST /v1/completions`, `max_tokens=256`, `concurrency=4` | **34.71 tok/s** aggregate, `18.45s` avg latency, `28.40s` p95 |

These are current validated numbers on the exact machine above, measured on April 2, 2026 with `zig build -Doptimize=ReleaseFast`. The earlier small-model M1 Pro bring-up was useful for proving the backend, but it is no longer the public reference point for Apple Silicon performance.

For context, the same 35B model on the AMD RDNA4 test node (Radeon AI PRO R9700, 32 GB, 576 GB/s) currently runs at **37.95 tok/s** in the clean CLI decode benchmark. That is close enough to make the point: the Metal path is real, not a fallback feature.

## What is next

The Metal backend ships today. Same tests, same API, same models. The optimization roadmap:

**Batched prefill.** Current Metal prefill processes one token at a time. A proper batched path would dramatically improve time-to-first-token.

**Sampling controls.** Temperature, top-p, repetition penalty on Metal to match Vulkan.

**Generation-specific tuning.** M4 Max is now measured and tuned, but the current kernels still aim to run across the whole family. The next step is broader M1/M2/M3 coverage plus generation-specific M5 variants where the hardware justifies them.

**Precompiled shaders.** metallib bundles for faster cold starts.

**More models.** Current validation is Qwen3.5. Expanding to more architectures.

## The bigger picture

When ZINC started, the thesis was narrow: AMD consumer GPUs are ignored by the AI stack, and someone should fix that. Adding Apple Silicon does not change the thesis. It broadens it.

The real thesis: local GPUs are underused for inference, and the reason is software. AMD RDNA4 has the bandwidth. Apple Silicon has the unified memory. Both have the compute. What they lack is an engine that takes each platform seriously enough to build shaders from scratch, tune them for the actual hardware, and ship a single binary that just works.

Two GPU backends. One engine. No heavyweight frameworks. If you have an AMD GPU or a Mac, you have an inference machine.

```bash
zig build -Doptimize=ReleaseFast
./zig-out/bin/zinc chat
```

That is what ships today.
