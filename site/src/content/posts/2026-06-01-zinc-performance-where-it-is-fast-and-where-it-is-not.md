---
title: "ZINC performance: where it is fast, where it is not, and how to read the numbers"
seoTitle: "ZINC Performance Benchmarks: RDNA, Metal, Intel"
date: "2026-06-01"
tags:
  - zinc
  - performance
  - benchmark
  - rdna4
  - metal
  - apple-silicon
  - intel-arc
  - llama-cpp
  - qwen3
  - qwen-3-6
  - gemma
  - local-llm
  - llm-inference
keywords:
  - ZINC performance
  - ZINC benchmark
  - ZINC vs llama.cpp
  - local LLM performance
  - AMD RDNA4 LLM inference
  - Radeon AI PRO R9700 benchmark
  - Apple Silicon LLM inference
  - Metal LLM benchmark
  - Qwen 3.6 benchmark
  - Gemma 4 benchmark
  - local AI inference benchmark
  - prefill vs decode
  - tokens per second benchmark
faqs:
  - question: "Is ZINC faster than llama.cpp?"
    answer: "Sometimes, but not universally. In the June 1 dashboard data, ZINC is faster than llama.cpp on Qwen 3.6 35B A3B decode on both the RDNA R9700 node and the Apple M4 Max. Across the whole matrix, ZINC still trails llama.cpp in the overall prompt+decode score and especially on prefill and end-to-end harness latency."
  - question: "What is the strongest current ZINC result?"
    answer: "The strongest current result is Qwen 3.6 35B A3B UD Q4_K_XL decode: 127.9 tok/s on the Radeon AI PRO R9700 and 81.07 tok/s on the Apple M4 Max, both ahead of the matching llama.cpp decode medians in the dashboard data."
  - question: "Why can decode win while end-to-end still loses?"
    answer: "Decode measures the hot one-token-at-a-time answer loop. End-to-end includes the whole benchmark path: startup, process or server overhead, request handling, prompt ingestion, and decode. Those are different waits, so one can improve without automatically fixing the others."
  - question: "Will ZINC v0.1 ship for Intel Arc?"
    answer: "No. The Intel Arc Pro B70 data is useful but still experimental. The v0.1 release matrix should stay focused on Linux x86_64 Vulkan and macOS Apple Silicon Metal."
excerpt: "The current ZINC dashboard tells a more interesting story than a single tok/s headline. ZINC is now ahead of llama.cpp on Qwen 3.6 35B A3B decode on both the Radeon AI PRO R9700 and Apple M4 Max. It is not ahead everywhere: prefill remains the broad gap, end-to-end harness latency is still harsh, and Intel Arc is experimental. This post explains how to read the numbers without flattening prefill, decode, and full request latency into one misleading score."
seoDescription: "A data-driven look at ZINC performance across RDNA, Apple Silicon Metal, and Intel Arc, with charts for decode, prefill, and end-to-end benchmark behavior."
---

Quick answer: ZINC is now genuinely competitive in parts of the local LLM inference loop, but the honest statement is narrower than "ZINC is faster." In the current dashboard data, ZINC beats llama.cpp on Qwen 3.6 35B A3B decode on both the Radeon AI PRO R9700 and the Apple M4 Max. It does not beat llama.cpp across the whole matrix, and prefill is still the obvious gap.

That is the useful state of the project before v0.1. The engine is no longer a bring-up curiosity. It has real wins on difficult model shapes. It also has enough data now that the weak spots are visible instead of guessed.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-06-01-zinc-overall-performance-targets.svg" alt="A dashboard-style chart titled ZINC performance snapshot. It shows the best ZINC decode result per target from June 1, 2026 data: AMD RDNA Radeon AI PRO R9700 at 127.9 tok/s on Qwen 3.6 35B A3B, 117.9 percent of llama.cpp decode; Apple Metal M4 Max at 81.07 tok/s on the same model, 110.9 percent of llama.cpp decode; Intel Arc Pro B70 at 43.9 tok/s on Qwen 3 8B and marked experimental." loading="lazy" />
  <figcaption>The headline is phase-specific. The current dashboard shows Qwen 3.6 35B A3B decode ahead of llama.cpp on RDNA and M4 Max, while Intel remains an experimental Vulkan path.</figcaption>
</figure>

The source for the numbers is the current [ZINC benchmark dashboard](/zinc/benchmarks/), backed by `site/src/data/zinc-performance.json`. The RDNA data was collected on one AMD Radeon AI PRO R9700 node with 32 GB of VRAM and 576 GB/s memory bandwidth. The Metal data was collected on an Apple M4 Max. The baseline for the compared rows is llama.cpp on the same model files.

The most important thing to know before reading any of these charts is that "tokens per second" is not one metric. It is at least three different waits.

## Three waits

Prefill is how fast the engine reads the prompt. Decode is how fast it streams each new token after the prompt has been ingested. End-to-end latency is the full wall-clock path the benchmark harness observes, including process or server overhead, request handling, prompt ingestion, and decode.

Those are not interchangeable. A local engine can have a strong decode loop and still feel slow on a long first token. It can have fast prefill and still lose end-to-end because the server path is expensive. It can win one model family and lose another because the model architecture changes the hot buckets.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-06-01-zinc-performance-three-waits.svg" alt="A three-part timeline explaining benchmark waits. Prefill is prompt ingestion and first-token work. Decode is the hot answer loop for each generated token. End-to-end is the full harness or user-visible wait including startup, request path, prompt, and decode. A footer says never ask whether an engine is simply fast; ask which wait got faster." loading="lazy" />
  <figcaption>Decode, prefill, and end-to-end latency are different engineering problems. The current ZINC story only makes sense if we keep them separate.</figcaption>
</figure>

That separation is why the current dashboard is more interesting than a single bar chart. On the flagship Qwen 3.6 35B A3B row, ZINC is ahead in decode on both RDNA and Metal. On prefill, it is still behind. In the end-to-end harness number, it is much further behind, because that path includes more than the GPU's hot token loop.

## The current headline

The strongest ZINC row today is Qwen 3.6 35B A3B UD Q4_K_XL. On the Radeon AI PRO R9700, ZINC decodes it at `127.9 tok/s` against llama.cpp at `108.5 tok/s`, or `117.9%` of the baseline. On the Apple M4 Max, ZINC decodes the same model at `81.07 tok/s` against llama.cpp at `73.09 tok/s`, or `110.9%` of baseline.

That is a real result because the model is not an easy dense transformer. It is a large hybrid model with mixture-of-experts routing and stateful layers. It also fits the exact category that made ZINC worth building: a high-end local GPU running a model that is large enough to be useful and awkward enough that generic dense-kernel intuition is not enough.

The caveat is just as important. Across five compared RDNA models, the dashboard summary has ZINC at `80.5%` of llama.cpp on the overall prompt+decode score. Across five compared Metal models, it has ZINC at `58.9%` on the same overall score. The engine has crossed llama.cpp in specific decode workloads, not across the board.

The right summary is:

| Target | Best ZINC row | ZINC decode | llama.cpp decode | Result |
| --- | ---: | ---: | ---: | --- |
| RDNA / R9700 | Qwen 3.6 35B A3B | `127.9 tok/s` | `108.5 tok/s` | ZINC `117.9%` |
| Metal / M4 Max | Qwen 3.6 35B A3B | `81.07 tok/s` | `73.09 tok/s` | ZINC `110.9%` |
| Intel Arc B70 | Qwen 3 8B | `43.9 tok/s` | `52.14 tok/s` | Experimental |

That last row is intentionally not framed as a product result. Intel Arc is useful research data right now. It is not the v0.1 release story.

## The heatmap is the honest chart

The most honest performance chart is not the headline chart. It is the percent-of-llama heatmap by phase.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-06-01-zinc-performance-phase-heatmap.svg" alt="A heatmap titled Percent of llama.cpp is not one number. Rows include RDNA Qwen 3.6 35B A3B, RDNA Qwen 3.5 9B, RDNA Gemma 4 26B-A4B MoE, Metal Qwen 3.6 35B A3B, and Metal Gemma 4 31B Dense. Columns are decode, prefill, and end-to-end. Green cells show over 100 percent of llama.cpp, yellow 60 to 99 percent, orange 30 to 59 percent, and red below 30 percent." loading="lazy" />
  <figcaption>The pattern is clear: decode has real wins, prefill is the broad gap, and end-to-end remains the harshest metric.</figcaption>
</figure>

On RDNA, the two Qwen rows show why decode matters. Qwen 3.6 35B A3B is `117.9%` of llama.cpp on decode. Qwen 3.5 9B is `111.6%`. Those are not tiny wins. They mean ZINC's hot Vulkan token loop is doing real work on the shapes it has been optimized for.

But those same rows are `38.7%` and `18.4%` of llama.cpp on prefill. That is the visible first-token gap. It means the prompt side still needs structural batching, better state handling, and less per-prompt overhead.

End-to-end is harsher still. In the dashboard, RDNA Qwen 3.6 35B A3B is only `14.3%` of llama.cpp end-to-end. That does not contradict the decode win. It says the full harness path is still expensive. The dashboard notes matter here: ZINC is measured through the CLI path, while the llama.cpp baseline prefers one reusable `llama-server` per model across the scenario matrix. End-to-end is useful, but it is not pure kernel throughput.

That distinction is the entire educational point. Benchmarks become misleading when the chart title says "LLM speed" and the axis silently mixes prompt ingestion, decode, startup, and server reuse.

## RDNA: the strongest current backend

RDNA is the most mature ZINC backend right now. The R9700 has enough VRAM for the 35B-A3B Q4_K_XL class, enough bandwidth to make decode interesting, and a Vulkan stack that has been hammered by the optimization loop for weeks.

The current RDNA dashboard summary:

| Model | ZINC decode | llama.cpp decode | ZINC prefill | llama.cpp prefill |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.6 35B A3B UD Q4_K_XL | `127.9` | `108.5` | `154.27` | `398.82` |
| Qwen 3.5 9B Q4_K_M | `95.39` | `85.51` | `100.79` | `548.94` |
| Gemma 4 26B-A4B MoE Q4_K_M | `89.73` | `102.0` | `89.1` | `497.08` |
| 27B Dense Qwen Q4_K_M | `28.47` | `30.65` | `49.0` | `185.14` |
| Gemma 4 31B Q4_K_M | `24.65` | `28.55` | `41.64` | `201.97` |

The pattern is clear. ZINC is already competitive or ahead on decode for the Qwen paths it has targeted hardest. It is still behind on prompt processing across every listed RDNA model. The difference between those columns is the roadmap.

Why would Qwen 3.6 35B A3B be the strongest row instead of a smaller model? Because active-parameter count and resident-parameter count are different things. The model is large enough to stress memory residency, but each decode token only activates a subset of experts. That makes the hot loop a routing, bandwidth, and state-management problem rather than a dense "multiply every weight every token" problem. ZINC has spent most of its recent engineering on exactly that class of problem.

The smaller and dense rows expose the parts that have not been normalized yet. Gemma and dense Qwen prefill still lean on paths where llama.cpp's mature graph batching and kernel selection are much further ahead.

## Metal: real progress, uneven coverage

The Metal backend is no longer a toy. The current Apple M4 Max data shows the same flagship Qwen 3.6 35B A3B decode win: `81.07 tok/s` for ZINC against `73.09 tok/s` for llama.cpp.

That is a serious signal because Apple Silicon has a very different memory model from the RDNA node. Unified memory, Metal pipeline-state behavior, and Apple GPU threadgroup constraints make the backend a separate engineering project rather than a simple port.

The full Metal table is more uneven:

| Model | ZINC decode | llama.cpp decode | ZINC prefill | llama.cpp prefill |
| --- | ---: | ---: | ---: | ---: |
| Qwen 3.6 35B A3B UD Q4_K_XL | `81.07` | `73.09` | `97.4` | `151.5` |
| Gemma 4 31B Q4_K_M | `21.86` | `23.30` | `132.6` | `84.47` |
| Gemma 4 26B-A4B MoE Q4_K_M | `30.01` | `88.44` | `34.0` | `365.45` |
| Qwen 3.5 9B Q4_K_M | `23.21` | `66.52` | `23.2` | `90.65` |
| 27B Dense Qwen Q4_K_M | `8.93` | `23.32` | `9.6` | `27.87` |

Two things are true at the same time. The flagship row is promising. The average backend is still behind. That is normal for a young backend. It usually means a handful of exact hot shapes have been optimized and the broad catalog still runs through generic or older paths.

The Gemma 4 31B row is a good example of why phase-specific charts matter. Metal ZINC is `157%` of llama.cpp on prefill for that row, but only `93.8%` on decode and `38.2%` end-to-end. If we published only the prefill number, the backend would look solved. If we published only the end-to-end number, it would look hopeless. Neither reading is accurate.

## Intel: useful data, not a release target yet

Intel Arc Pro B70 is in the benchmark data because it is strategically interesting: 32 GB of VRAM, high bandwidth, and matrix hardware that should matter for local inference once the backend grows into it.

But the current ZINC data is partial. In the dashboard, only one Intel ZINC model row succeeds: Qwen 3 8B Q4_K_M. It decodes at `43.9 tok/s` against llama.cpp at `52.14 tok/s`. Prefill is interesting at `49.1 tok/s` against `32.36 tok/s`, but that does not make the backend mature.

For v0.1, Intel should stay out of the binary release matrix. The data is useful for roadmap direction. It is not yet a support promise.

## What the numbers say about the engine

The dashboard tells us five practical things.

First, ZINC's decode loop can be excellent. The Qwen 3.6 35B A3B rows on RDNA and M4 Max are enough evidence for that. A project that could only bring models up would not be ahead of llama.cpp on that shape.

Second, prefill is the next structural priority. The most embarrassing cells in the heatmap are not decode cells. They are prompt-processing cells. That points toward graph-level batching, state reuse, fused prompt kernels, and fewer prompt-side dispatch boundaries.

Third, average backend quality still matters. A single flagship win is not a product. It is a proof that the architecture can win. The work after that is spreading the exact-shape wins across families and prompt regimes.

Fourth, end-to-end must be interpreted carefully. The dashboard's end-to-end numbers are intentionally strict and include more than the GPU's hot path. They are useful because users do wait on the whole path. They are dangerous if presented as "kernel throughput."

Fifth, ZINC needs release discipline now. Once binaries exist, every benchmark claim has to attach to a build, a commit, a model file, and a scenario matrix. That is why `zinc --version`, checksums, release archives, and a draft-only release workflow matter. Performance work without provenance turns into folklore.

## What we should optimize next

The next work is not "make ZINC faster." That is too vague. The numbers point to a smaller set of jobs.

On RDNA:

- Improve prefill for Qwen and Gemma without hurting the current decode wins.
- Keep reducing prompt-side dispatch and state movement.
- Preserve the Qwen 3.6 35B A3B decode lead as a regression gate.
- Treat end-to-end CLI/server overhead as a separate benchmark, not a shader problem.

On Metal:

- Expand exact-shape wins beyond Qwen 3.6 35B A3B.
- Keep measuring cold pipeline-state costs separately from steady-state decode.
- Make broad Gemma and smaller Qwen rows less dependent on generic fallback paths.
- Validate release archives outside the repo, because Metal shader asset lookup is now part of the binary story.

On Intel:

- Keep it experimental.
- Publish data when the matrix succeeds, not when one row does.
- Do not let Intel support leak into v0.1 release language until it has the same hardware validation discipline as RDNA and Metal.

## The honest v0.1 performance message

The v0.1 performance message should be:

> ZINC is an early local inference engine with real decode wins on targeted Qwen workloads. It is not yet a universal llama.cpp replacement. RDNA and Apple Silicon are the first release targets. Prefill and end-to-end latency remain active work.

That sentence is less flashy than "ZINC beats llama.cpp." It is also much more defensible.

The dashboard has become useful precisely because it is uneven. If every cell were green, we would have a marketing page, not an engineering signal. The red cells tell us what to do next. The green cells tell us the project is worth doing.

That is where ZINC is now: no longer proving that it can run, not yet claiming that it wins everywhere, and finally measured well enough that the next work is obvious.
