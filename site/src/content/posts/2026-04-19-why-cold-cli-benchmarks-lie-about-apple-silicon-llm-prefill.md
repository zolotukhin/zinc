---
title: "Why cold-CLI benchmarks lie about Apple Silicon LLM prefill"
date: "2026-04-19"
tags:
  - zinc
  - apple-silicon
  - metal
  - benchmarking
  - llm-inference
  - prefill
  - mmap
  - performance
keywords:
  - Apple Silicon LLM benchmark
  - cold start prefill
  - mmap page fault benchmark
  - llama.cpp vs zinc benchmark
  - M4 Max LLM benchmark
  - time to first token TTFT
  - warm server benchmark harness
  - zinc performance suite
  - Qwen3.5 35B A3B Metal
  - warm vs cold inference benchmark
  - local LLM inference fairness
  - ZINC_PREFILL_PROFILE
  - llama-server warmup
excerpt: "Our own benchmark harness made ZINC look slower than it is. The April 18 Metal suite reported 1.0 tok/s prefill on Qwen3.5-35B-A3B. The April 15 run reported 2.1. Same engine, same model, same machine. The difference was measurement, not regression. Here is why cold-process CLI benchmarks stop working once the model mmaps 21 GiB of weights."
---

The April 18 Metal suite reported `qwen35-35b-a3b-q4k-xl` core prefill at a median of **1.0 tok/s**. Three days earlier, the same harness reported **2.1 tok/s**. Nothing about the engine changed between those runs that should have halved prefill throughput.

What changed was which three cold CLI launches the suite happened to catch on that machine that day.

This post is about a specific kind of benchmark bug we shipped in our own tooling. It matters beyond ZINC because the same pattern shows up any time you measure a local LLM runtime by spawning a fresh process per run, against a model whose weights are larger than every other thing on the machine combined.

## The asymmetric measurement

Our shared performance suite at `tools/performance_suite.mjs` compares ZINC against [llama.cpp](https://github.com/ggml-org/llama.cpp) across a grid of models, contexts, and prompts. The two engines are not measured the same way.

[llama.cpp runs through `llama-server`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md): one HTTP process per model, launched once, reused across every scenario and every measured run in that model's grid. The server stays loaded. The OS page cache stays warm. The Metal command queue keeps its state.

ZINC runs through a one-shot CLI invocation (`./zig-out/bin/zinc -n … --model-id … --prompt …`), which the suite spawns fresh for every warmup and every measured run. A six-model suite ends up launching about **ninety-six cold ZINC processes** against **six warm llama-server instances**. Every one of those ZINC launches pays a startup cost that llama.cpp is paying exactly once per model.

That cost is not a bug in ZINC's loader. It is the cost of touching the model at all.

## What a cold process actually pays for

The managed Qwen3.5-35B-A3B GGUF we ship is about twenty-one gigabytes on disk. ZINC mmaps it at load time, the same way [llama.cpp does](https://github.com/ggml-org/llama.cpp/issues/91). That part is fast and nearly free, because no weight pages have been touched yet. The process's virtual address space just acquires a 21 GiB window pointing at the file.

The cost shows up on the first GPU dispatch. Metal is about to read a chunk of the weights, so the kernel has to fault in those pages from disk. On Apple Silicon, unified memory means the GPU and the CPU are looking at the same physical pages, but that does not make the first access free. The pages still need to come in from storage, get mapped, and become resident before the compute kernel can dequantize them. Measurements in the community have described this clearly for transformer workloads ([discussion](https://github.com/ggml-org/llama.cpp/discussions/638)): mmap is instant, but a transformer immediately faults in every page it touches, and the first prefill is where that cost lands.

In ZINC on the M4 Max Mac Studio, with a 12-token prompt (“What is the capital of France?”), the numbers come out like this.

First-launch measurements of total prefill time for the whole 12-token prompt, captured in different cold process launches on the same binary: **4.7 s, 10.6 s, 3.4 s, 4.7 s**. The variance is not from the KV cache size. It is the page cache state, the recent disk activity, and the scheduler.

Once the process is warm, the picture is completely different. Decode on the same model on the same machine runs steadily above 47 tok/s. Decode reads the same weight pages the prefill step just faulted in. The second time you touch them, they are already resident, and the per-token cost drops to what the math alone demands.

## The chart that explains the jitter

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/cold-cli-prefill-noise.svg" alt="A horizontal bar chart of six back-to-back ZINC CLI prefill measurements on Qwen3.5-35B-A3B on an M4 Max Mac Studio. Three runs at -c 4096 report 2.4, 2.8, and 3.0 tokens per second. Three runs at -c 131072 report 3.5, 2.6, and 2.5 tokens per second. A warm decode median of about 47 tokens per second is drawn for scale." loading="lazy" />
  <figcaption>Six cold CLI launches, same engine, same model, same machine, same prompt. The prefill number jitters by about 40 percent depending on how much of the cold-fault tax each individual process happens to pay.</figcaption>
</figure>

The six CLI runs above were taken three at a time, back to back, on the same M4 Max, in two KV cache sizes. The per-token compute cost of prefill on this model does not change between them. What changes is the fraction of a 12-token window that goes to reading weights for the first time versus running arithmetic against weights that were already warmed by an earlier phase of the same process.

The stable green bar at the bottom is not a different workload. It is the same GPU, the same shaders, the same KV cache, reading the same weights that are now already resident. That is what the post-cold-load steady state actually looks like.

A benchmark that captures only the three cold launches at the top and ignores the bottom will always report numbers that fluctuate with whatever the OS happened to evict between the previous model and this one. That is the harness we shipped. It is how we reported a 50 percent prefill swing between two runs that had no engine change between them.

## Why llama.cpp did not have this problem

llama.cpp has known about this failure mode for years and routed around it. The canonical benchmarking tool in that project is [`llama-bench`](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench), which loads the model once, does an internal warmup pass, and then runs the repetitions against the already-warm state. The default is five repetitions after one warmup. The `--warmup` flag in `llama-server` plays the same role for the HTTP path.

Community discussion on [mmap behavior for MoE models](https://github.com/ggml-org/llama.cpp/discussions/18758) makes the same observation from the other direction: when the model is larger than RAM or is fighting the page cache for residency, the apparent throughput becomes a function of what fraction of the weights are currently faulted in, not of the engine's math.

The asymmetry in our suite was not intentional. It emerged because ZINC was born CLI-first, and the easiest way to measure a CLI is to run it. llama.cpp had both a CLI and a server well before we did, so the suite's llama.cpp side was built against the warm server path from day one. The ZINC side inherited its shape from the early days when every ZINC benchmark was a one-shot prompt.

That was correct when the model fit in a few gigabytes. It is wrong for a 21 GiB mmap on Apple Silicon.

## Two different questions, two different harnesses

There is a subtler point underneath this. Warm and cold prefill are not the same metric in disguise. They answer different questions.

Cold prefill is the right metric if you want to know how long it takes between a user double-clicking the app and seeing the first token on a fresh machine. It captures page faulting, pipeline compilation, shader loading, and KV allocation. `bench-metal` style microbenchmarks that we cite in earlier posts already answer this question deliberately.

Warm prefill is the right metric if you want to know how fast the engine is at ingesting a prompt when the model is already loaded and in use. That is how nearly every real chat session looks after the first message. It is also the metric llama-server reports, which means any ZINC-vs-llama.cpp comparison should be measured the same way.

Our suite was trying to report warm prefill for llama.cpp and cold prefill for ZINC, then presenting the two numbers side by side. That is not a fair comparison even when both engines are flawless.

## The fix, and what it will not change

The repair is small in scope. We added a `launchLocalZincServer` helper alongside the existing `launchLocalLlamaServer`, routed the ZINC side of the suite through ZINC's own `/v1/completions` endpoint, and kept the CLI path as a fallback for environments where the server cannot start. After the change, a six-model Metal suite runs one ZINC server per model instead of launching a fresh CLI 96 times. The [`--cache-prompt` and slot reuse](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) that keep llama-server warm have direct ZINC equivalents because both engines are just HTTP servers once the measurement is on the server path.

Two things are worth stating plainly so nobody reads this as a performance claim.

Nothing about this change makes the underlying engine faster. The same ZINC binary that reported 1.0 tok/s prefill in the cold harness still has the same per-token cost once the pages are warm. What changes is that the reported number stops depending on OS state that ZINC does not control. The April 18 "regression" against April 15 disappears because it was never a regression.

It also does not eliminate the real cold-start cost. Users who open ZINC for the first time after a reboot still pay the mmap fault-in tax, and that cost deserves its own dedicated metric. We keep the CLI path alive for exactly that reason, and `bench-metal` remains the right tool for cold-process microbenchmarks cited in the earlier [RDNA4 shader post](/blog/2026-03-29-the-shaders-standing-between-4-tok-s-and-27-tok-s) and the [bring-up post](/blog/2026-04-01-bringing-zinc-to-apple-silicon).

## What this says about local inference benchmarking

The broader lesson is not specific to ZINC. Any local inference project that measures itself by spawning a fresh process per run is inheriting a dependency on its host's page cache, scheduler, and disk state. That dependency is usually invisible until the model crosses the size where mmap fault-in dominates the first dispatch.

For 1 GB dense models, cold and warm converge fast enough that the difference fits inside normal run-to-run noise. For 20 GB-class hybrid MoE models, the two regimes are different physics. Pretending they are the same produces headline numbers that move with the wind.

If you are building a benchmark harness for local LLM inference, the two rules that fall out of this are simple. Measure the engine the same way users run it, which on a local machine almost always means a long-lived server rather than a fresh CLI. And when you want to measure cold start on purpose, do it with a tool that advertises that intent, so you do not smuggle it into a metric labeled "throughput."

For ZINC, that is what the next performance suite publishes. The numbers will be lower-variance, closer to the decode steady state the user actually sees, and symmetric with how llama.cpp has been measured all along. For anyone reading those numbers, the bar for "ZINC got slower" just moved a lot closer to "ZINC actually got slower," and away from "the Mac Studio happened to have a cold page cache that afternoon."

That is the measurement we should have been publishing the whole time.

For the background on how ZINC's managed-model surface and benchmark path fit together, see [Qwen3.6-35B-A3B is now generally available](/blog/2026-04-17-qwen-3-6-is-now-generally-available-in-zinc), [every design decision behind ZINC](/blog/2026-04-03-every-design-decision-behind-zinc), and [bringing ZINC to Apple Silicon](/blog/2026-04-01-bringing-zinc-to-apple-silicon). If you just want to run a model, start with [Getting Started](/zinc/docs/getting-started) and [Running ZINC](/zinc/docs/running-zinc).
