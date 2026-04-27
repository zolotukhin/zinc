---
title: "Why one vkQueueSubmit per prompt is the next quiet RDNA4 prefill unlock"
date: "2026-04-25"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - prefill
  - command-buffers
  - amdgpu
  - llm-inference
  - linux
  - performance
keywords:
  - vkQueueSubmit overhead RDNA4
  - Vulkan command buffer batching
  - amdgpu submit ioctl cost
  - per-layer command buffer LLM inference
  - AMD user mode queues Linux 6.16
  - ZINC RDNA4 submit gap
  - Mesa RADV command stream
  - llama.cpp Vulkan prefill submit
  - prefill CPU stall RDNA4
  - 60-layer Gemma 4 submit budget
excerpt: "ZINC currently records one Vulkan command buffer per transformer layer and calls vkQueueSubmit once for each. On a 60-layer Gemma 4 prefill that is 60 round trips through the amdgpu ioctl path, and the gap between submits is wide enough to land on every flame graph the engine produces. Collapsing those 60 submits into one is a small change at the recording layer with a measurable upside, and it is the right shape of fix to land before AMD's user-mode queues make the per-submit cost smaller but not zero."
---

The 0.6-second `CPU and submit gap` bar in the [RDNA4 prefill phase budget](/blog/2026-04-18-why-rdna4-prefill-for-qwen-3-5-is-stuck-at-25-tok-s) was the most embarrassing slice of that chart. The kernels were doing the work the kernels are supposed to do. The reason 600 milliseconds of wall time on a Qwen3.5-35B prompt was sitting on the CPU side of the timeline is that the Vulkan backend was issuing one `vkQueueSubmit` per layer, and the cost of a `vkQueueSubmit` on `amdgpu` is not zero. On a 60-layer model that adds up fast.

The shape of the fix is small. Record one command buffer per prompt instead of one per layer. Submit it once. Wait once. The shader inner loops are unchanged, the descriptor sets are unchanged, the pipeline cache is unchanged. The change is purely about command-buffer granularity, and it is the kind of plumbing decision that sits well below the "we ported a kernel" headline but compounds with everything above it.

This post is the argument for why the per-prompt submit shape is correct, why it has not been the default, and what the [upcoming AMD user-mode queue path](https://docs.kernel.org/next/gpu/amdgpu/userq.html) does and does not change about that argument. The short version is that user-mode queues lower the floor on per-submit cost but do not move the structural reason to batch. The long version is below.

## What a per-layer submit actually costs

The Vulkan specification is unambiguous about submission being a CPU-heavy operation. The [Khronos command-buffers chapter](https://docs.vulkan.org/spec/latest/chapters/cmdbuffers.html) says explicitly that "submission can be a high overhead operation, and applications should attempt to batch work together into as few calls to `vkQueueSubmit` or `vkQueueSubmit2` as possible." NVIDIA's [Vulkan Dos and Don'ts](https://developer.nvidia.com/blog/vulkan-dos-donts/) phrases it more bluntly: "each `vkQueueSubmit()` has a significant performance cost on CPU, so lower is generally better."

The cost on Mesa RADV with `amdgpu` is the kernel ioctl path. Every submit walks through `vkQueueSubmit` validation in user space, then into the `amdgpu_cs_ioctl` packet build, then into the GPU scheduler queue, then back. The fast case on a recent kernel and a recent RADV is in the low tens of microseconds. The slow case is not. A long-standing [Mesa issue tracking extreme submit overhead](https://gitlab.freedesktop.org/mesa/mesa/-/issues/4330) caught individual submits taking up to 2 milliseconds when the submission thread contended with itself, and even outside of that pathological regime the ordinary cost is in the hundreds of microseconds when the queue depth is non-trivial.

For a 60-layer prompt with one submit per layer, a 200 microsecond per-submit cost is 12 milliseconds of wall time spent purely in the submission machinery. That is not the dominant slice of a long prefill, but it is consistently the largest unforced error in a profile, because it shows up as a flat tax that no kernel optimization can attack.

There is a second cost that does not show up in the per-call timing: the GPU scheduling gap. Between two submits, the command processor on the card can drain the work in flight before the next batch arrives. On a fast kernel and a hot queue, the gap is small. On a cold queue, or under contention with another process on the same `drm` minor, the gap widens to whatever the kernel scheduler decides. RADV's [command-stream documentation](https://docs.mesa3d.org/drivers/radv.html) describes the queue submission as a kernel-mediated step where the driver translates the recorded command stream into hardware-readable packets, and the per-step latency depends on what else the kernel scheduler has on its plate.

## Why the per-layer shape happened

The per-layer shape is not arbitrary. It came out of the natural way every transformer inference engine grows. The first cut of a forward pass is a Python or Zig loop over the model layers, and the simplest way to prove the loop is correct is to make each iteration self-contained: record a command buffer, dispatch every shader the layer needs, submit, wait, validate the output of the layer against a CPU reference, repeat. That shape is a great way to find shader bugs and a poor way to ship the production prefill path.

The pattern persists because most inference engines on Vulkan use a generic compute graph compiler that never accumulated a notion of "the whole prompt is one thing." [llama.cpp's Vulkan backend](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp) is the most-deployed example, and it has a configurable submit cadence based on the size of the work but defaults to a granularity that is closer to per-layer than per-prompt for prefill in many configurations. ZINC inherited the same instinct from its early bring-up, when the validation harness wanted a CPU sync after each layer to compare hidden states against the reference. That harness has been dead code for months. The submit cadence followed it into dead code without anyone noticing.

The per-token decode path does not have this problem. A single decode step has only one set of layer dispatches and naturally lives inside one submit. The prefill path is where the layer count multiplies the submit count, and the prefill path is where the gap shows up.

## The shape of the fix

A per-prompt submit is structurally simple. The recorder walks the same layer loop it walks today, but instead of opening a new command buffer at the top and closing one at the bottom, it appends every dispatch into a single command buffer that lives for the full prompt. At the end of the prompt, one `vkQueueSubmit2` call goes out, with one `VkSubmitInfo2` and one fence. The wait is one `vkWaitForFences` with a timeout sized to the worst-case prefill, not 60 short waits stitched together.

There are three things to be careful about. The first is barrier emission. The per-layer shape lets the recorder treat each layer as fresh state and rebuild every barrier from scratch. A per-prompt shape needs a barrier ledger that survives across layers so the recorder can omit redundant `vkCmdPipelineBarrier` calls between projections that already have a write-after-write dependency satisfied. This is a real diff but a bounded one.

The second is descriptor lifetime. Per-layer recording makes it easy to allocate descriptor sets out of a pool that gets reset between layers. Per-prompt recording either has to size the pool to hold every set the prompt will need, or has to use [`VK_EXT_descriptor_buffer`](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/proposals/VK_EXT_descriptor_buffer.adoc) so the descriptor data lives in a buffer the recorder can address directly. RDNA3 and RDNA4 both support `VK_EXT_descriptor_buffer` in current RADV, and the buffer-resident descriptor model is what every newer engine is moving toward anyway.

The third is the host-side timeline. With per-layer submits, the host can read intermediate KV cache writes between layers if it wants to, for example to stream a KV cache snapshot to disk for prompt caching. With one submit, the host gets to look at the GPU state once at the end. For ZINC's current prompt-caching design that is fine, because the snapshot point is the end of the prompt anyway. For an engine that wanted finer-grained snapshots, the right answer is timeline semaphores within the single command buffer, not a return to per-layer submits.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-04-25-rdna4-vkqueuesubmit-gap-timeline.svg" alt="A horizontal timeline of a 60-layer prefill on RDNA4. The top lane shows the current per-layer submit shape: 60 narrow GPU-work segments separated by 60 small CPU-and-submit gaps that visibly accumulate over the full prompt timeline. A summary at the right of the lane reads twelve to twenty-four milliseconds of submit gap, depending on queue contention. The bottom lane shows the per-prompt submit shape: one continuous GPU-work block with a single submit gap at the front and a single fence wait at the back. A small annotation under the bottom lane reads one vkQueueSubmit, one fence wait, no scheduler hops between layers." loading="lazy" />
  <figcaption>Same prompt, same shaders, same layer count. The top timeline accumulates 60 submit gaps. The bottom timeline pays the submit cost once. The shape of the GPU work is identical.</figcaption>
</figure>

The diagram makes the point obvious. The kernels are not the issue. The recording shape is. Once the per-prompt command buffer lands, the only place a CPU stall can show up between layers is inside the recorder, and the recorder runs in parallel with the GPU on the previous prompt under the standard double-buffered pattern.

## What user-mode queues do and do not change

The Linux 6.16 kernel ships [initial AMD user mode queue support for RDNA3 and RDNA4](https://www.phoronix.com/news/Linux-6.16-AMDGPU-User-Queues), behind a module parameter. The mechanism is exactly what the name says: user-space writes packets directly into a ring buffer that the GPU firmware reads, with no kernel ioctl in the hot path. The [kernel documentation for user queues](https://docs.kernel.org/next/gpu/amdgpu/userq.html) describes the queue as a producer-consumer ring with a write pointer the user managed and a read pointer the firmware advances, and the explicit goal is to "bypass IOCTL calls to the driver to submit work."

The temptation when reading that is to think the per-layer submit problem goes away on its own. It does not. User-mode queues lower the per-submit cost from "ioctl plus kernel scheduler hop" to "memory-mapped ring write plus a doorbell." That is a meaningful reduction, easily an order of magnitude on the CPU side. It does not change the GPU-side scheduling gap between two separate work batches. The command processor still draws a line between submission boundaries, and the firmware still has to pick up the next batch of work from the ring once the previous one has drained.

What user-mode queues do change is the calculus on how much engineering to spend on submit batching for engines that target a wide range of Linux kernels. For ZINC's RDNA4 path, the kernel floor is what is on a current Mesa-friendly distribution, which lags the bleeding kernel by a year. The right design is the one that is correct on a kernel without user queues and inherits the win on a kernel with them. The per-prompt submit shape is exactly that: it removes 59 out of 60 submit-and-gap pairs on every kernel, and on a kernel with user queues the remaining one pair gets cheaper too.

## What the numbers should look like

The arithmetic is straightforward enough to do without measuring. The current RDNA4 path runs one submit per layer for a 60-layer Gemma 4 31B prefill, which is 60 submits per prompt. At a conservative 200 microseconds per submit on a contended `amdgpu` queue, the per-prompt CPU-side cost is 12 milliseconds. At the slow-case 2 millisecond figure from the [Mesa overhead bug](https://gitlab.freedesktop.org/mesa/mesa/-/issues/4330), it is 120 milliseconds. The GPU-side scheduling gap adds an estimated 50 to 200 microseconds per gap, for another 3 to 12 milliseconds. The total is somewhere between 15 and 130 milliseconds of wall time spent on submit overhead per prompt, with most production runs landing in the 20 to 40 millisecond range.

| Submit shape | Submits per 60-layer prompt | Estimated CPU+gap cost | Notes |
| --- | ---: | ---: | --- |
| Per-layer (today) | 60 | 15 to 130 ms | Variance dominated by `amdgpu` queue contention |
| Per-prompt (planned) | 1 | 0.3 to 2.5 ms | One submit, one fence, no scheduler hops between layers |
| Per-prompt + user queue | 1 | < 0.1 ms | Doorbell write, no ioctl, no kernel scheduler hop |

The table is an estimate, not a benchmark. The point is the order of magnitude, not the exact number. On a 4 to 5 second prefill for a 100-token Gemma 4 prompt, eating 25 milliseconds of submit overhead at the small end is a 0.5% tax. On the per-token decode path at 50 tok/s, eating one submit per token is 10 milliseconds of submit overhead per second of decode, which is a 1% tax that nobody currently budgets for. Neither of these is a headline number on its own. Both of them are the kind of flat tax that compounds with every other optimization the engine ships, and removing the tax now is cheaper than removing it after the rest of the kernel work has shrunk the denominator.

## What comes next

The per-prompt submit recording lands inside the same `prefillBatched` entry point that yesterday's [Gemma 4 head-dim work](/blog/2026-04-24-the-single-push-constant-blocking-gemma-4-prefill-on-rdna4) is targeting. The recorder change is a few hundred lines of Zig, mostly bookkeeping for the barrier ledger and the descriptor pool sizing. The per-prompt fence wait replaces the per-layer fence stack with a single `VkFence` and a `VkSemaphore` chain for the host-readable end-of-prompt signal.

The follow-on work is the descriptor buffer port. Once the per-prompt recorder lands and the descriptor pool grows by 60x to hold every layer's sets, the right move is to skip the pool entirely and build the descriptor data once into a `VK_EXT_descriptor_buffer` allocation that the command buffer addresses directly. That move is mechanical after the recording shape is right, and it is the same shape the Metal port has been using through `MTLArgumentEncoder` from the start.

The wider note is that the LLM inference engines that are about to ship across consumer AMD on Linux are running into the same set of submission-side taxes that game engines learned to factor out a decade ago. The kernel-side work on user queues is welcome and will help. The application-side work to stop submitting per layer is the larger and more portable win. Search the engine for any path that calls `vkQueueSubmit` more than once per logical user request, and treat each extra submit as a budget item that has to justify itself.
