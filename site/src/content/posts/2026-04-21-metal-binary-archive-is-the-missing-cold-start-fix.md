---
title: "MTLBinaryArchive is the missing half of the Metal LLM cold-start fix"
date: "2026-04-21"
tags:
  - zinc
  - apple-silicon
  - metal
  - llm-inference
  - cold-start
  - pipeline-state
  - shader-cache
  - benchmarking
  - gpu-kernels
keywords:
  - MTLBinaryArchive LLM
  - Metal pipeline state cache
  - Apple Silicon LLM cold start
  - llama.cpp Metal library cache
  - persistent shader cache Metal
  - MTLComputePipelineState cache
  - ZINC Metal cold start
  - Metal binary archive LLM inference
  - pipeline state compilation tax
  - warmup vs binary archive
  - macOS Metal shader compile
  - first prompt latency Metal
faqs:
  - question: "What is MTLBinaryArchive and why does it matter for local LLM inference?"
    answer: "`MTLBinaryArchive` is Apple's on-disk cache for compiled pipeline state objects. When Metal compiles a shader from AIR into GPU machine code, the result can be captured into an archive, written to disk, and loaded by a later process. For an LLM runtime, that is the only mechanism that kills the per-launch pipeline-state compilation tax, because every other cache the runtime can build lives only for the lifetime of a single process."
  - question: "Does llama.cpp already solve this on Metal?"
    answer: "Partially. PR #12265, merged in March 2025, caches the compiled Metal library at the device context level so repeated `llama_context` creations inside the same process do not recompile. That fixes one class of redundant work. It does not touch the cost a fresh CLI invocation pays, because the cache lives in process memory and dies when the process exits. A disk-backed `MTLBinaryArchive` is the next layer, and llama.cpp does not currently ship one for its Metal backend."
  - question: "How big is the first-launch win from binary archives in practice?"
    answer: "Apple's WWDC 2020 session that introduced the API showed a seventeen-hundred-pipeline test app drop from 86 seconds to 3 seconds of pipeline build time on a cold start, roughly 28x, by loading a harvested archive. LLM inference is not seventeen-hundred pipelines; a dense 8B path hits tens of unique states and a hybrid MoE hits closer to a hundred. The absolute savings are smaller, but the shape is the same and it is fully subsumed inside the same first-prompt window the user already stares at."
  - question: "Why can't you just ship one archive and be done?"
    answer: "Because archives are keyed by the compiled GPU machine code, which is keyed by the specific Apple silicon plus the macOS driver version. An archive harvested on an M3 Pro under macOS 15.3 is not guaranteed to hit on an M1 Max under macOS 15.4. The practical shape is a local archive per device per macOS major, regenerated lazily the first time a fresh combination is seen. Misses fall through to normal compile with no user-visible error."
  - question: "What breaks the archive strategy on MoE models?"
    answer: "The same thing that breaks one-shot warmup. A single dummy forward pass on a 128-expert model only routes to eight of them, so only eight experts' kernels get compiled and only those land in the archive. The rest stay uncompiled. The fix is the same on both sides: enumerate every expert explicitly when collecting the archive, not rely on routing to cover the space."
excerpt: "Yesterday's post identified a second tax on Metal LLM cold starts: pipeline-state compilation. llama.cpp's current Metal library cache kills that cost across repeated contexts in one process, but does nothing for a fresh CLI invocation. Apple's `MTLBinaryArchive` API is the only mechanism that makes the compiled pipelines survive a process exit. Here is why ZINC needs to ship one, what it actually costs, and where MoE quietly breaks the approach."
---

[The pipeline-state post from yesterday](/blog/2026-04-20-why-cold-metal-llms-pay-a-pipeline-state-tax) identified two independent taxes on a Metal LLM cold start: `mmap` page faulting for the weight file, and `MTLComputePipelineState` compilation for every unique shader the model will dispatch. The first is a kernel problem. The second is a driver problem. Both show up in the user's first-token latency, and both disappear on the second prompt in the same process.

What was missing from that post, and what matters now, is that the industry's current standard answer for the second tax only half-works. llama.cpp's Metal backend recently landed an in-process library cache in [PR #12265](https://github.com/ggml-org/llama.cpp/pull/12265), and a warmup dispatch has been the default since long before that. Together they make the second prompt in the same process free. They do nothing for a fresh binary. The fix that would make a fresh binary free is `MTLBinaryArchive`, and it has been sitting in Metal since WWDC 2020.

## What the current llama.cpp cache actually covers

BB-fat's PR #12265 was merged by ggerganov on March 11, 2025. The description is one paragraph: "Currently, Metal shaders are recompiled for every llama context initialization, which is redundant and impacts performance when creating multiple contexts. Cache the compiled Metal library at the device context level (`g_ggml_ctx_dev_main`), reusing it for subsequent context initializations." The fix is the right fix for the problem the PR is scoped to. The problem it is scoped to is a long-running server that creates and destroys `llama_context` objects, for example when loading different models or resetting state.

That is a real workload, and the cost it removes is real. It is also strictly narrower than the cost a `llama-cli` invocation pays when a user runs it from a shell and waits for the first token. The cache lives in a global on the device context struct. When the process exits, the struct is destroyed, and the next `llama-cli` run pays the full pipeline-state tax again. The same is true of every runtime that follows the same pattern, ZINC included.

The second layer of the current answer is the warmup dispatch, which is enabled by default in `llama-server` and documented in its [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) as the `--warmup` / `--no-warmup` pair. Warmup makes sure the first real prompt does not hit an uncompiled kernel mid-generation. It pushes the compile cost into a one-shot startup phase instead of spreading it across the user's first forward pass. It does not make the compile cost smaller. On a cold CLI it reorganizes where the pain lands, not how much pain there is.

## What `MTLBinaryArchive` actually is

`MTLBinaryArchive` is a container for compiled pipeline-state objects plus the descriptor keys they were built from. It was introduced in Apple's WWDC 2020 session "Build GPU binaries with Metal" (session 10615). The key methods for a compute runtime are `addComputePipelineFunctions(descriptor:)` to capture a compiled pipeline into an archive, `serialize(to:)` to write the archive to disk, and a `binaryArchives` property on the pipeline descriptor to plumb the archive into pipeline creation.

On a pipeline creation call, Metal searches the descriptor's `binaryArchives` array linearly for a matching function plus specialization-constant key. On a hit it skips compilation and returns the precompiled state. On a miss it falls through to the normal compile path. The closest thing to a collect-then-replay reference implementation in open source is MoltenVK, the Vulkan-on-Metal translation layer that Khronos and Valve ship. Its tracking issue [MoltenVK #1765, "Explore use of Metal Binary Archives for Vulkan pipeline caching"](https://github.com/KhronosGroup/MoltenVK/issues/1765), summarises the current state in one sentence: "pipeline caching in MoltenVK is limited to archiving/reloading the shader MSL source, after conversion from SPIR-V. It does not cache either the compiled MSL or the Metal pipeline state in binary form." The same ticket names `MTLBinaryArchive` as the fix path.

The load side is important. An archive file is authored once, at any time during development or during a first-run collection pass, and then shipped or kept in the user's cache directory. Every subsequent cold start can point at it and skip the tens of milliseconds of per-kernel compilation that a fresh `MTLComputePipelineState` would otherwise cost. The WWDC demo that introduced the API used a seventeen-hundred-pipeline test app and showed the cold-start pipeline-build cost drop from 86 seconds to 3 seconds, a factor of 28. The demo is not an LLM, but the mechanism is the same one.

The other context worth stating is that non-Apple engine authors have independently concluded this is where the real fix lives. Flutter's [engine tracker for "Investigate wiring up Metal Binary Archives on iOS" (issue #60267)](https://github.com/flutter/flutter/issues/60267) frames the problem as raster-thread jank from on-the-fly pipeline creation and names `MTLBinaryArchive` as the planned solution for the same reason a local LLM engine would: any cache that dies with the process is not a fix, it is a workaround.

## How the resolution flow actually works

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/metal-binary-archive-pso-resolution.svg" alt="A two-lane flow diagram. The upper lane shows the default compute pipeline creation path: a pipeline descriptor plus a function handle points at AIR inside the metallib, Metal compiles AIR into GPU machine code on every launch, and the result is a ready pipeline state object kept in RAM for the life of the process. The lower lane shows the path with an MTLBinaryArchive: the descriptor carries a binary-archives array, Metal searches on-disk archives keyed by device and driver, a hit skips compilation and loads the already-compiled ISA directly, a miss falls through to normal compile and can optionally capture the result back into the archive for next launch." loading="lazy" />
  <figcaption>The only difference between the two lanes is the archive lookup at the front. A hit turns compilation into a file read. A miss is harmless, because it falls through to the default compile path and the archive can grow over time as new kernels get captured.</figcaption>
</figure>

The shape a runtime author should notice is that the archive path is additive, not invasive. Wiring it in means setting one property on each pipeline descriptor and teaching the build system how to author and place the archive file. The failure mode is silent fallthrough to the current behaviour. That is the right failure mode for something a runtime will ship to thousands of different `(silicon, macOS)` pairs.

## What the cold-start tax actually looks like by cache tier

Useful numbers for a runtime author come from laying out the three states against each other. The numbers in the table are for a single process launch of Qwen3.5-35B-A3B on an M1 Pro, measured against the same 154-token prompt the rest of the RDNA4 work uses as its reference workload. They are estimates rather than measured millisecond counts, because ZINC does not yet ship any of the caches in question; the ranges come from the prior post's phase decomposition and from typical Metal compile-per-kernel figures on M-series hardware.

| Launch scenario | Weight page faults | Pipeline compile | First-prompt tax |
| --- | ---: | ---: | ---: |
| Cold process, no archive, no warmup | full | every kernel | 3 to 10 s faults plus 0.6 to 1.5 s compile |
| Cold process, no archive, with warmup | full | every kernel, but front-loaded | same total, pain moves from first token to launch |
| Cold process, process library cache hit (PR #12265) | full | every kernel | identical to the no-cache case |
| Cold process, `MTLBinaryArchive` hit on known device | full | skipped | 3 to 10 s faults, zero compile |
| Warm second prompt, same process | none | none | ~0 s |

The row that matters most is the fourth. A binary archive does not touch the page-fault tax, which is the bigger of the two numbers, but it turns the compile tax from a second every launch into a file read. The fifth row is the one every warm-server benchmark secretly reports, and the one no cold CLI can hit on the first invocation without an archive in place.

## What ZINC would have to ship to use this

The engineering work is bounded. ZINC's current Metal pipeline creation path, in `src/metal/shim.m`, is a single call into `[device newComputePipelineStateWithFunction:function options:... error:...]`. Promoting the creation site to a descriptor-based call and setting `binaryArchives = @[archive]` on the descriptor is a mechanical change, and the only real complexity sits in collection and versioning.

Collection is a one-shot pass at first launch on a new `(device, macOS major)` pair. Walk the model's op list, enumerate every kernel the graph will dispatch with representative shapes, add each compiled pipeline into a `MTLBinaryArchive` via `addComputePipelineFunctions(descriptor:)`, and call `serialize(to:)` to persist it. MoE needs the walk to enumerate every expert, not only whichever ones a dummy forward routes to, because [llama.cpp issue #11163](https://github.com/ggml-org/llama.cpp/issues/11163) already documents the symptom for page-faulting and the pipeline-state side has the same shape.

Versioning is the other half. Archive hits are not guaranteed across driver updates, and the right default is to key the archive file on `(device family, macOS major, zinc build hash, model id)` and to lazily regenerate on miss. An archive store under the user's cache directory with one subdirectory per `(device, macOS, model)` tuple is the cleanest place to land it. The Flutter tracker above settles on the same shape for its raster-thread pipelines, and the tradeoff is identical for an LLM runtime.

The size cost is small. A compiled pipeline state is kilobytes, the full archive for a 35B hybrid MoE should live well under a megabyte, and the disk write happens once per fresh `(device, macOS)` pair rather than per launch.

## The tradeoff worth stating plainly

Binary archives do not make the runtime faster on a warm server. They make the cold-CLI shape look more like the warm-server shape. On a `llama-server` that stays up for hours, the archive is a one-line no-op. On a ZINC CLI that the user fires for a single completion, the archive flips the compile tax from "always paid" to "paid once per fresh device-OS combination." That is the whole pitch, and it is worth less than it looks on a bench that only measures warm cases.

It is also worth less than the mmap page-fault tax the earlier post attributed to disk. The archive does nothing for the 3-to-10 second page-fault cost, which stays the dominant component of cold-CLI latency until somebody wires a pre-touch pass that faults the weight file in at launch. The two fixes stack. The right long-term shape is both of them on the same cold start, with the archive bringing the compile tax to zero and the page-fault bringing the weight cost to a single fast sequential read.

## What comes next

The honest next step on ZINC's Metal side is to land the archive write path during the graph-aware warmup that was scoped at the end of yesterday's post, and to key the archive file on the device and macOS version as above. The read path is a single-line change on the pipeline descriptor. Both together are a week of work, and the measurable result should be that the second and third cold-CLI runs on the same machine report the same first-prompt latency as the second prompt in a single-process run, minus only the remaining page-fault cost.

That is the first cold-start fix in the ZINC plan that actually lives on disk. The rest of the work, from the [Q8_1 activation path](/blog/2026-04-19-why-q8-1-activations-are-the-next-rdna4-prefill-unlock) on RDNA4 to the [batched Metal prefill](/blog/2026-04-20-metal-batched-prefill-38x-speedup-coherent-with-llama-cpp) on Apple Silicon, runs inside the warm window. The archive is what makes that warm window start sooner.
