---
title: "How we made AMD LLM inference 4x faster on a single GPU"
date: "2026-03-30"
tags:
  - zinc
  - amd
  - rdna4
  - r9700
  - vulkan
  - zig
  - llm-inference
  - qwen3-5
  - performance
keywords:
  - AMD RDNA4 inference
  - AMD LLM inference
  - Radeon AI PRO R9700
  - AMD R9700 benchmark
  - Vulkan LLM inference
  - Zig inference engine
  - Qwen3.5-35B-A3B
  - Qwen3.5-35B-A3B-UD Q4_K_XL
  - local LLM performance
  - AI PRO R9700 benchmark
  - decode throughput
  - GPU resident inference
  - Vulkan command batching
excerpt: "ZINC used to look stuck at about 7 tok/s on AMD RDNA4. The clean ReleaseFast baseline now measures 33.58 tok/s on Qwen3.5-35B-A3B-UD Q4_K_XL on a Radeon AI PRO R9700. This is what changed, which old numbers were misleading, and what still separates us from llama.cpp."
---

For a while, ZINC had an annoying kind of progress problem.

The engine could already run a real 35B model, produce coherent output, and prove that AMD was not a dead end for local inference. But it still felt slow enough that every conversation got pulled back into the same question: does this actually have a path to being useful, or is it still just an interesting bring-up?

That question is starting to get a much better answer.

On March 30, 2026, the clean `ReleaseFast` baseline on the same single RDNA4 GPU measured **33.58 tok/s** on [Qwen3.5-35B-A3B-UD Q4_K_XL](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) running on an **AMD Radeon AI PRO R9700**. That is a little more than a **4x jump** from the old coherent baseline. But the interesting part is not the headline number. It is that the speedup did not come from one heroic trick. It came from a less glamorous mix of real GPU work, fewer CPU round trips, less intrusive measurement, and finally benchmarking the right path.

This post is the bridge between the earlier ZINC story of [basic correctness](/blog/2026-03-27-what-broke-first-when-we-built-zinc-on-amd-rdna4) and the next stage of throughput work. It is the point where the engine stopped looking "alive but slow" and started posting numbers that actually deserve to be compared.

<figure class="diagram-card diagram-wide">
  <img src="/blog/zinc-7-to-33-rdna4.gif" alt="A sped-up screen recording of ZINC running Qwen3.5-35B on the AMD Radeon AI PRO R9700 after the throughput jump, showing the faster AMD RDNA4 decode path in action." loading="lazy" />
  <figcaption>The current clean RDNA4 path is no longer a 7 tok/s science project. It is a 33.58 tok/s baseline with real room left to climb.</figcaption>
</figure>

## Where we started

The early numbers on ZINC were rough for two different reasons.

First, the engine had to become correct. Before that, throughput barely mattered. The forward pass skipped layers, the tokenizer was wrong, flash attention bindings were wrong, and the model output was too broken for performance work to mean anything. That phase bottoms out around the **0.8 tok/s** era from the earlier correctness post. It was slow, but more importantly, it was not yet trustworthy.

Then came the first coherent path. By the time the GPU SSM path and batching work started landing, ZINC could produce coherent English on the 35B Qwen3.5 model, and the decode loop looked stuck at about **7.6 tok/s** on the RDNA4 test node.

That was the number that started to define the project in people's heads: ZINC equals "about 7 tok/s."

The problem is that this was only partly true.

## The 7 tok/s story was real, but incomplete

The old 7 to 16 tok/s numbers in the optimization logs were useful. They captured a real phase of the project. But they also mixed together several things that should have been separated:

- debug-heavy builds
- intrusive profiling and diagnostics
- correctness-oriented loop runs
- clean decode throughput
- reasoning-chat throughput
- server overhead

Those are not the same measurement.

By March 30, the throughput plan had to admit something uncomfortable: part of the reason ZINC looked stuck at 7 tok/s was not that the engine itself had stopped improving. It was that we were still using the wrong KPI for too much of the optimization loop.

The loop was often measuring debug paths. `--profile` still carried too much diagnostic baggage. Mid-token readbacks and extra validation logic were distorting the thing we thought we were trying to optimize. In other words, some of the old slowness was real, and some of it was measurement tax.

That distinction matters, because "the kernels are slow" and "the benchmark path is too intrusive" lead to very different engineering decisions.

## What actually moved the number

The speedup from roughly **7.6 tok/s** to **33.58 tok/s** did not come from one magical kernel rewrite. It came from a stack of changes that all pulled in the same direction.

### 1. More of decode stopped bouncing through the CPU

The biggest structural problem in the early path was not that RDNA4 lacked bandwidth. It was that decode still had too many small GPU to CPU to GPU handoffs.

That was especially painful on Qwen3.5-35B-A3B, because it is not a plain transformer. It combines attention, MoE routing, and SSM layers. The old path kept making the CPU babysit pieces of work that should have stayed on the GPU:

- SSM conv1d
- SSM delta-net updates
- SSM gated norm
- router softmax plus top-k

That meant lots of tiny submit and wait cycles per token. The GPU was not getting a clean chance to run through decode without the host constantly stepping back into the middle.

The moment those paths started becoming GPU-resident, the engine stopped bleeding time into round trips that had nothing to do with model math.

### 2. Command batching started to matter

Once the forward pass was correct and more of the SSM and routing work stayed on-device, command-buffer batching stopped being theoretical cleanup and started becoming real throughput work.

This is one of the less flashy lessons from the project: command overhead matters a lot once the model is actually running correctly. If the runtime keeps rebuilding and resubmitting too much state per layer, it does not matter that the shaders are "fast enough" in isolation. The whole token still pays for all the setup.

The cleaner path reduced that tax enough that the decode number finally moved into the 30 tok/s range.

### 3. We finally separated clean decode from everything else

This was probably the most important mental shift in the whole performance pass.

The raw decode path, the reasoning-chat path, and aggregate server throughput are related, but they are not the same target.

On the clean node, the current numbers now look like this:

| Model | Path | Measured throughput | Hardware |
| --- | --- | --- | --- |
| `Qwen3.5-35B-A3B-UD-Q4_K_XL` | CLI plain decode | **33.58 tok/s** | AMD Radeon AI PRO R9700 (RDNA4, 32 GB) |
| `Qwen3.5-35B-A3B-UD-Q4_K_XL` | Raw `/v1/completions` | **33.55 tok/s** | AMD Radeon AI PRO R9700 (RDNA4, 32 GB) |
| `Qwen3.5-35B-A3B-UD-Q4_K_XL` | Raw `/v1/completions`, `concurrency=4` | **33.98 tok/s aggregate** | AMD Radeon AI PRO R9700 (RDNA4, 32 GB) |
| `Qwen3.5-35B-A3B-UD-Q4_K_XL` | Reasoning chat | **24.94 to 28.56 tok/s** | AMD Radeon AI PRO R9700 (RDNA4, 32 GB) |
| Historical small dense reference | CLI plain decode | **22.93 tok/s** | AMD Radeon AI PRO R9700 (RDNA4, 32 GB) |
| Historical small dense reference | Raw `/v1/completions` | **21.88 tok/s** | AMD Radeon AI PRO R9700 (RDNA4, 32 GB) |

That table immediately makes one thing clear: the raw decode problem is no longer "how do we get above 7 tok/s?" The new problem is more specific:

- keep the clean decode path above 30 tok/s
- lift the reasoning-chat path above 30 tok/s
- drive GPU utilization higher without breaking correctness

Those are much better problems to have.

### 4. The environment stopped moving under our feet

There was a quieter contributor to the improvement too: the benchmark environment got cleaner.

The RDNA4 node is now measured with:

- Mesa pinned to the driver version that does not regress RADV
- `RADV_PERFTEST=coop_matrix`
- clean `ReleaseFast` builds
- competing processes cleared off the node before measurement

That may sound like benchmark housekeeping, but it is not optional. If the machine underneath you is drifting, you do not learn the right lesson from each optimization step.

## The strange result that taught us the most

The older small dense reference is slower.

On the current clean node, that historical small dense reference measures **22.93 tok/s**, while the larger `Qwen3.5-35B-A3B-UD-Q4_K_XL` path measures **33.58 tok/s**.

That is not what people expect if they think model size alone determines speed.

But it is an extremely useful clue. It tells us that today's bottleneck is not just "more parameters means slower decode." It is about actual decode shapes, kernel regimes, architecture mix, and how this runtime behaves on specific workloads. That is a much more interesting problem than a simple bandwidth story.

## Current performance on the Radeon AI PRO R9700

ZINC is now in a much healthier place than it was when the project looked pinned to 7 tok/s:

- the 35B path is coherent and above 33 tok/s on the clean CLI benchmark
- the raw HTTP path is basically at parity with the clean CLI path
- the server is no longer just a toy shell around a slow runtime
- the reasoning-chat path is slower, but now clearly defined as its own optimization target

That does not mean the job is done.

The current llama.cpp baseline on the same node and model is still about **107 tok/s**. ZINC at **33.58 tok/s** models to about **112.5 GB/s**, or only **19.5%** of the card's `576 GB/s` peak bandwidth. There is still a large gap between "the engine is finally healthy" and "the engine is using this GPU as well as it should."

But that is a much more exciting place to be than the old situation, because the next problems are now concrete:

- reducing reasoning-chat overhead
- making profiling lower-perturbation
- reusing more command and descriptor state across decode
- fusing same-input decode work
- retuning kernels by the shapes that actually dominate decode

Those are no longer speculative ideas for an engine that might still be measuring the wrong thing. They are the real next steps for an engine that has finally crossed into credible territory.

## What the 4x speedup actually means

I do not think the most interesting part of this milestone is the number itself.

The interesting part is that ZINC now has a real baseline.

At **7 tok/s**, every discussion kept getting dragged back into the same question: is the project slow because AMD is the wrong target, because Vulkan is the wrong approach, because Zig is the wrong language, or because the implementation is still too unfinished to tell?

At **33.58 tok/s**, that question starts to lose its power.

The answer is becoming much clearer: AMD RDNA4 is not the problem. Vulkan is not the problem. Zig is not the problem. The remaining work is real systems work inside a runtime that is now measurably alive.

That is a much better place to build from.

If you want the broader context around how ZINC got here, the earlier posts tell the story in order: [why we are building ZINC](/blog/2026-03-25-why-we-are-building-zinc), [the local AI rig behind it](/blog/2026-03-26-building-a-local-ai-rig), [what broke first](/blog/2026-03-27-what-broke-first-when-we-built-zinc-on-amd-rdna4), and [the self-improving loop behind the optimization work](/blog/2026-03-28-karpathy-loop-autoresearch-and-the-self-improving-ai-loop-behind-zinc).

This new phase is simpler to describe: the engine is no longer arguing with basic correctness or fake baselines. It is finally arguing with the real performance ceiling.
