---
title: "Why cold-start Metal LLMs pay a pipeline-state tax nobody budgets for"
date: "2026-04-20"
tags:
  - zinc
  - apple-silicon
  - metal
  - llm-inference
  - prefill
  - benchmarking
  - gpu-kernels
  - cold-start
keywords:
  - Metal pipeline state compilation
  - MTLComputePipelineState LLM
  - cold start Apple Silicon LLM
  - llama.cpp Metal warmup
  - MoE warmup bug
  - ZINC Metal warmup
  - first Metal dispatch latency
  - MTLBinaryArchive LLM
  - local inference first token
  - MSL compilation cost
  - Metal AIR compile
  - pipeline state object tax
faqs:
  - question: "What is Metal pipeline-state compilation and why does a local LLM pay for it?"
    answer: "A Metal compute shader cannot be dispatched directly. It has to be wrapped in an `MTLComputePipelineState` object, and the object is created from a Metal function plus any specialization constants. Creating it triggers a compile from AIR (Apple's intermediate representation stored inside the metallib) down to GPU machine code for the specific silicon the process is running on. LLM inference issues many distinct kernels per forward pass, so each one pays this compile cost the first time it is dispatched."
  - question: "Is this cost the same as mmap page faulting on a cold start?"
    answer: "No. They are two independent taxes on the same cold start. Page faulting is about moving weight bytes from disk or the file cache into resident memory. Pipeline-state compilation is about turning shader AIR into executable GPU code. A cold process pays both, and you can have a warm file cache that still has to pay the compile cost on every new process."
  - question: "How does llama.cpp avoid this cost?"
    answer: "`llama-server` runs a warmup pass by default that does a dummy forward, which touches every kernel the model's graph will actually use and compiles their pipeline states up front. After warmup, every subsequent prompt dispatches against states that are already cached. The knob is the `--warmup` / `--no-warmup` flag documented in the llama.cpp server README."
  - question: "Why does the warmup misfire on MoE models?"
    answer: "A dense model's warmup dispatch touches every kernel it will ever use. A Mixture of Experts model routes each token to a subset of experts, so a one-shot warmup only triggers pipeline-state compilation for the experts that particular dummy input happened to route to. The others stay uncompiled until a real prompt hits them, and that real prompt pays the tax mid-generation. llama.cpp's tracking issue #11163 documents the same pattern from the opposite side, where most of the MoE weights never get faulted in for the same reason."
  - question: "What would ZINC have to do to fix this on Metal?"
    answer: "Two things. Run a graph-aware warmup that enumerates the kernels each model architecture actually hits and dispatches them once with representative shapes before reporting readiness. For MoE, enumerate all experts in the warmup, not only the ones a random input routes to. Longer term, cache the compiled pipelines to disk with MTLBinaryArchive so a fresh process can load them instead of recompiling."
excerpt: "On Apple Silicon the first prompt after a fresh local LLM binary is slow for two independent reasons. One is mmap page faulting. The other is Metal compiling a compute pipeline state object for every unique shader the model will dispatch. llama.cpp pays this cost once at server warmup. ZINC currently pays it on the first user prompt, every launch."
---

The [cold-CLI benchmark post](/blog/2026-04-19-why-cold-cli-benchmarks-lie-about-apple-silicon-llm-prefill) blamed a 3-to-10 second first-prompt tax on `mmap` page faulting. That explanation is correct, and it is incomplete. Even with a fully warm file cache, the first prompt on a fresh Metal binary is slower than the second. The extra cost has nothing to do with disk. It is the Metal driver compiling a compute pipeline state object for every distinct shader the model is about to dispatch, on first use, in the middle of your prompt.

This post is about that second tax. It is smaller than the page-fault tax, but it is structural, it scales with model architecture rather than weight size, and it is the reason llama.cpp ships a `--warmup` flag and ZINC does not yet. The gap shows up in every cold-CLI comparison where one engine has already paid the tax and the other has not.

## What a pipeline state actually is

On Metal, a compute shader cannot be dispatched directly. The GPU-side executable is wrapped in an `MTLComputePipelineState` object, and that object is created from a function handle plus any specialization constants. The creation call has a real cost. It triggers a backend compile from AIR, Apple's intermediate representation stored inside the `.metallib`, down to the GPU machine code for whichever M-chip the process happens to be running on. The same story applies to the rendering pipeline, which is why `MTLBinaryArchive` exists as a standing mechanism to cache the compiled output across launches.

The call site in llama.cpp's Metal backend is a single line inside [`ggml-metal-device.m`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m): `[lib->device newComputePipelineStateWithFunction:mtl_function error:&error]`. That call is cheap to write and expensive to execute. Tens of milliseconds per kernel is common on M-series hardware. Much larger numbers show up in the wild when the metallib is big and every kernel is seeing the GPU for the first time. [Godot's issue tracker has a good concrete report](https://github.com/godotengine/godot/issues/106757) of the same pattern at extreme scale, where Metal shader compilation dominates first-launch time for a project with many shaders.

## Why LLM inference hits this wall

A transformer forward pass is not one kernel. It is a graph of many. RMSNorm, RoPE, Q4_K dequantize-into-f16, the matmul variants that feed attention, flash attention itself, softmax, SwiGLU, the sampling step. Each one is a distinct Metal function, and when you specialize by head count or tile size you get a family of variants instead of a single entry point.

llama.cpp's `ggml-metal.metal` currently defines `112 kernel void kernel_*` functions and references `519` distinct `"kernel_*"` string literals across its specialization paths. Not all of those get touched by any one model. A dense Llama-3 path at Q4_K hits a smaller subset than a hybrid MoE model that also needs expert routing, top-k, and scatter-add kernels. But even the small subset is dozens of unique pipeline states, each paying the compile tax once.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/metal-pipeline-state-cold-start-tax.svg" alt="A two-lane timeline. The top lane shows the first prompt after a fresh Metal LLM launch, with bars for mmap setup, first-layer weight page faults, Metal pipeline state compilation, and the actual first kernel dispatch. The bottom lane shows the second prompt in the same process, with a short dispatch bar and no pipeline compile or fault work." loading="lazy" />
  <figcaption>The first prompt pays three independent taxes. The second prompt in the same process pays only the last one. Pipeline-state compilation is the orange bar in the middle, and it survives even if the file cache is already warm.</figcaption>
</figure>

The shape that matters is not the absolute millisecond number, because the ms count depends on the chip, the metallib size, and the scheduler. The shape is that pipeline-state compilation is paid once per unique kernel per process, and that cost is invisible on a warm-server benchmark and fully visible on a cold-CLI one.

## MoE breaks the one-shot warmup

The warmup trick is simple. Do a dummy forward pass before the user's first prompt. Every kernel the forward pass needs gets its pipeline state compiled, then cached. The next real prompt dispatches into already-warm state. `llama-server`'s [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) documents the flag as `--warmup, --no-warmup` with warmup enabled by default.

That works for dense models because a single forward touches every kernel the engine will ever run. It breaks for Mixture of Experts. [llama.cpp issue #11163](https://github.com/ggml-org/llama.cpp/issues/11163), "Misc. bug: model warmup doesn't work correctly for MoE models," reports the same failure from the page-cache side: one dummy dispatch only routes to a small set of experts, so only those experts' weights get faulted in and only those experts' kernels get compiled. The rest stays cold. The quote from the opening comment is specific: loading a dense model "will warmup the model correctly, loading the whole thing into OS cache." Loading a big MoE "will only load a small portion (93GB/660GB)."

Pipeline-state caching has the exact same shape. On Qwen3.5-35B-A3B with 128 experts and top-8 routing, a single warmup dispatch touches at most 8 of the expert variants. The remaining experts stay as uncompiled AIR until they get routed to in a real prompt, and the user eats the compile stall mid-generation when it happens. It looks like inference latency jitter and it is really serialized shader compilation.

![Estimated unique Metal pipeline-state count per model family. Ranges are low and high estimates based on llama.cpp ggml-metal.metal kernel and specialization counts.](/blog/metal-pso-kernel-count-by-model.png)

The bar chart above is the reason this matters for the blog's flagship model. A dense 8B path hits roughly 26 to 36 unique pipeline states. A hybrid MoE like Qwen3.5-35B-A3B hits 62 to 84. Double the count, double the per-launch compile tax, and add a long tail where the expert variants not yet routed get compiled one at a time during generation.

## What ZINC needs to do

ZINC's current Metal backend creates its pipeline states lazily, the same way llama.cpp does, which means the first-use cost is front-loaded onto the first real prompt. There is no warmup path. On a cold-CLI benchmark that captures only first-process runs, that cost is present in every measured number. It is a fraction of the cold-CLI noise the earlier post attributed to mmap, and both taxes compound.

The fix is structural. A graph-aware warmup that walks the model's op list once with representative shapes will trigger compilation for every kernel the engine will actually run. MoE support needs the warmup to enumerate every expert, not accept whatever routing a dummy token produces. Both pieces are cheap to build relative to the kernel work already in the engine. The longer-term lever is `MTLBinaryArchive`: persist compiled pipelines to a file next to the model, load them on subsequent launches, skip the compile step entirely. That is the same direction the Godot tracker points toward in its own [first-load shader compilation discussion](https://github.com/godotengine/godot/issues/106757) and several production games use binary archives to keep cold-start latency under control.

## The tradeoff worth stating plainly

Warmup is not free. It adds a fixed cost per process launch, measured in the range of hundreds of milliseconds for dense models and over a second for MoE. On a long-running server, that cost is amortized into nothing. On a single-shot CLI run against a small prompt, it can exceed the prompt's own compute time.

That is why this is a harness question as much as an engine question. A benchmark suite that already distinguishes cold from warm timings, as discussed in the cold-CLI post, should run the engine's warmup separately and report three numbers instead of two: cold-cold (no warmup, no cache), warm-cold (warmup but fresh file cache), and warm-warm (warmup plus warm cache). llama.cpp's server runs at the third number. Most CLI invocations of ZINC today run at the first. Any comparison that mixes them is measuring something neither engineering team actually controls.

The honest next step is to wire the warmup in, measure the difference separately from the page-fault tax, and publish both numbers side by side. That is the only way the rest of the prefill work, the [batched Metal prefill](/blog/2026-04-20-metal-batched-prefill-38x-speedup-coherent-with-llama-cpp) path and the [Q8_1 activation work](/blog/2026-04-19-why-q8-1-activations-are-the-next-rdna4-prefill-unlock) on the RDNA side, lands on a harness that credits each change fairly.
