---
title: "On AMD RDNA, ZINC is now faster than llama.cpp across the measured local model suite"
seoTitle: "ZINC vs llama.cpp on AMD RDNA: measured benchmark sweep"
date: "2026-07-01"
tags:
  - zinc
  - amd
  - rdna4
  - performance
  - benchmark
  - llama-cpp
  - local-llm
  - llm-inference
  - gemma
  - qwen3
  - qwen-3-6
  - vulkan
  - radeon-ai-pro-r9700
keywords:
  - ZINC vs llama.cpp
  - AMD LLM inference benchmark
  - RDNA4 LLM inference
  - Radeon AI PRO R9700 benchmark
  - Qwen 3.6 AMD benchmark
  - Gemma 4 AMD benchmark
  - local LLM performance
  - Vulkan LLM inference
  - prefill decode benchmark
  - AMD GPU local AI
faqs:
  - question: "Is ZINC faster than llama.cpp on AMD now?"
    answer: "In the current public RDNA benchmark artifact, yes for prefill, decode, and the phase-combined overall score on all five measured headline rows. Qwen 3.8 quick-chat end-to-end throughput is the narrow exception at 98 percent of llama.cpp. That is a measured suite result, not a promise for every AMD GPU, driver, model, prompt, or future commit."
  - question: "What models are in the AMD sweep?"
    answer: "The current RDNA suite covers Gemma 4 26B-A4B MoE Q4_K_M, Gemma 4 31B Q4_K_M, Qwen 3.5 9B Q4_K_M, Qwen 3.8 27B Dense Q4_K_M, and Qwen 3.6 35B A3B UD Q4_K_XL."
  - question: "How is the overall percentage calculated?"
    answer: "The fair score is time-based. The harness computes phase seconds as prompt tokens divided by prefill tokens per second plus generated tokens divided by decode tokens per second. The overall percent is llama.cpp phase seconds divided by ZINC phase seconds. It is not an average of prefill percent and decode percent."
  - question: "Does ZINC win every single scenario cell?"
    answer: "Not every cell. The public headline rows are ahead on prefill, decode, and phase-combined overall, while the scenario matrix still has tight cells; Qwen 3.8 quick-chat end-to-end throughput is currently 98 percent of llama.cpp."
excerpt: "The current AMD RDNA benchmark artifact is the first one where the headline is no longer a caveat. On the Radeon AI PRO R9700, ZINC is ahead of llama.cpp on every measured headline model row for prefill, decode, and phase-combined overall: Gemma 4 MoE, Gemma 4 dense, Qwen 3.5 9B, Qwen 3.8 27B, and Qwen 3.6 35B A3B. This post explains what changed, how the harness measures it, and where the claim stops."
seoDescription: "A comprehensive look at the July 2026 ZINC vs llama.cpp AMD RDNA benchmark sweep across Gemma 4 and Qwen models, with methodology, charts, caveats, and links to the public dashboard."
---

Quick answer: on the current public AMD RDNA artifact, ZINC is faster than llama.cpp on every headline model row we publish for prefill, decode, and phase-combined overall. That includes Gemma 4 26B-A4B MoE, Gemma 4 31B dense, Qwen 3.5 9B, Qwen 3.8 27B dense, and Qwen 3.6 35B A3B. Qwen 3.8 quick-chat end-to-end throughput is the narrow exception at 98 percent of llama.cpp.

That is a different sentence from "ZINC is faster on every AMD GPU in the world." The measured target is one AMD RDNA benchmark node: a Radeon AI PRO R9700 with 32 GB of VRAM and 576 GB/s of memory bandwidth. The comparison is against llama.cpp on the same host and the same GGUF files. The current artifact was generated on July 1, 2026 from ZINC commit `f9bf2def158d` and llama.cpp commit `9725a313b`.

It is still the first AMD result I am comfortable calling a sweep.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-01-rdna-headline-sweep.svg" alt="Heatmap of the current AMD RDNA ZINC benchmark artifact. Five models are shown: Gemma 4 26B-A4B MoE, Gemma 4 31B dense, Qwen 3.5 9B, Qwen 3.8 27B dense, and Qwen 3.6 35B A3B. Every row is above llama.cpp for prefill, decode, and overall; Qwen 3.8 quick-chat end-to-end is 98 percent." loading="lazy" />
  <figcaption>The current headline rows are above 100 percent of llama.cpp on prefill, decode, and overall. The biggest result is Qwen 3.6 35B A3B decode at 166.8 tok/s against llama.cpp at 108.5 tok/s.</figcaption>
</figure>

The live breakdown is on the [ZINC benchmark dashboard](/zinc/benchmarks/). The structured source lives in [`site/src/data/zinc-performance.json`](https://github.com/zolotukhin/zinc/blob/main/site/src/data/zinc-performance.json), and the harness lives in [`tools/performance_suite.mjs`](https://github.com/zolotukhin/zinc/blob/main/tools/performance_suite.mjs). Those links matter because benchmark claims rot quickly. If this post and the dashboard ever disagree, trust the dashboard.

## What changed

The old AMD story was full of careful qualifiers. ZINC could beat llama.cpp on one Qwen decode row, but Gemma lagged. Then Gemma MoE improved but dense Gemma still sat just behind. Then Qwen prefill looked absurdly high in one artifact and low in another because the harness mixed cold paths, warm servers, different token counts, and partial failures. The result was not lying exactly, but it was hard to read.

The current artifact is cleaner for two reasons.

First, the model matrix completed. All five ZINC rows finished on the RDNA node. That includes both Gemma rows, which were the important credibility test after the earlier partial run. Gemma matters because it is not just another Qwen-shaped transformer. Dense Gemma has its own normalization and attention shape. Gemma 4 26B-A4B adds sparse routing. If an engine only wins the flagship Qwen row, it can still be overfit to one hot path. If it wins both Qwen and Gemma, the claim is harder to dismiss.

Second, the comparison stopped mixing measurement modes. The headline RDNA numbers now come from a server-vs-server harness: one reusable ZINC server per model, one reusable llama.cpp server per model where available, the same scenario matrix, the same GGUF, one warmup discarded, three measured runs, and medians published. That means the result is no longer "our warm path versus their cold path" or "their server versus our CLI." It is the same kind of service loop on the same card.

Here are the headline rows:

| Model | Prefill | Decode | End-to-end | Overall |
| --- | ---: | ---: | ---: | ---: |
| Gemma 4 26B-A4B MoE Q4_K_M | 163% | 111% | 110% | 115% |
| Gemma 4 31B Q4_K_M | 125% | 101% | 101% | 103% |
| Qwen 3.5 9B Q4_K_M | 135% | 114% | 113% | 115% |
| Qwen 3.8 27B Dense Q4_K_M | 122% | 104% | 98% | 109% |
| Qwen 3.6 35B A3B UD Q4_K_XL | 136% | 154% | 142% | 151% |

The table is ZINC as percent of llama.cpp. So `154%` decode on Qwen 3.6 35B A3B means ZINC generated tokens at 1.54x the llama.cpp rate in that row: `166.8 tok/s` versus `108.5 tok/s`. The closest win is Gemma 4 31B dense decode: `28.81 tok/s` versus `28.54 tok/s`. That is not a victory lap number; it is a parity-plus number. But it matters because this was the row that still looked behind on the stale page.

The summary across the five headline rows is `134%` of llama.cpp on the dashboard's average overall score. Average ZINC decode is `87.8 tok/s`; average ZINC prompt throughput is `510.0 tok/s`; the fastest measured decode row is the Qwen 3.6 35B A3B model at `166.8 tok/s`.

## How we got there

The short version is: correctness first, then harness truth, then phase-specific optimization.

The early AMD bring-up failed in ways that looked deceptively close. One of the first Qwen traces produced English-shaped text while computing only a sliver of the output vocabulary. The fix was not a clever shader. It was boring dispatch math and validation against a reference. That lesson stuck. Every later performance change had to answer two questions before it was allowed to count: does the model still produce sane output, and did the measured path actually run the changed code?

The second phase was deleting unfair comparisons. For a while, we had too many numbers and not enough trust. Some came from one-shot CLI diagnostics. Some came from a warm llama.cpp server. Some stopped early because chat templates hit stop tokens. Some measured raw completions while others measured chat completions. Those numbers were useful for debugging, but they were bad public claims.

The current fair harness is the result of that cleanup.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-01-rdna-fair-harness.svg" alt="Flow diagram of the fair RDNA benchmark harness. One clean RDNA node runs the same GGUF files through reusable ZINC and llama.cpp servers. Both engines run the same four prompt scenarios. One warmup run is discarded, three measured runs are collected, and medians are published to the dashboard." loading="lazy" />
  <figcaption>The harness compares reusable servers on the same GPU and model files. The overall percentage is computed from phase time, not by averaging pretty percentages.</figcaption>
</figure>

The third phase was not one optimization. It was a long sequence of model-specific bottlenecks getting moved out of the way.

For Qwen, the hard row is the 35B A3B model. It is not a simple dense transformer. It combines a sparse active footprint with routing, attention, and stateful layers, which means the hot path is a chain of smaller waits rather than one giant matrix multiply. ZINC got faster by making that chain less host-shaped: static decode graphs, fused elementwise steps, less repeated setup, better prefill batching, and tighter dispatch around the paths that actually show up in the trace.

For Gemma, the important shift was treating it as a first-class architecture instead of a compatibility checkbox. Gemma's dense row and MoE row stress different pieces of the engine. The 26B-A4B result says the sparse Gemma path is now competitive. The 31B dense result says the boring dense row is no longer quietly losing. That second one is less dramatic, but it is exactly the kind of row that keeps a benchmark honest.

For the benchmark itself, the big change was making the score harder to game. The dashboard now separates prefill, decode, end-to-end throughput, and phase-combined overall. If prefill wins but decode loses, that shows up. If the server wall clock is noisy, that shows up. If a model emits a stop token and quits early, that gets flagged instead of becoming a fake speedup.

## What "overall" means

The overall score is not `(prefill percent + decode percent) / 2`. That would be wrong because a 50-token prompt and a 256-token answer do not spend equal time in prefill and decode. It would also make it too easy to inflate the score with a huge prefill percentage on a short prompt.

The dashboard uses a time-based score for comparable rows:

```text
ZINC seconds =
  zinc prompt tokens / zinc prefill tok/s
  + zinc generated tokens / zinc decode tok/s

llama.cpp seconds =
  llama.cpp prompt tokens / llama.cpp prefill tok/s
  + llama.cpp generated tokens / llama.cpp decode tok/s

overall percent =
  llama.cpp seconds / ZINC seconds * 100
```

That is why the Qwen 3.6 35B A3B row lands at `151%` overall even though its prefill ratio is `136%` and its decode ratio is `154%`. The generated-token phase dominates enough that the strong decode win matters more than a simple average would imply.

It is also why the Gemma 4 31B dense row only lands at `103%` overall. ZINC is ahead, but barely. The row is a useful warning label: the suite is not hiding close calls under a big average.

## The scenario matrix

The dashboard headline is a clean sweep. The full scenario matrix is slightly more nuanced.

Each model runs four workloads: Quick Chat, Coding Review, Incident Context, and Long Coding Draft. Across those 20 RDNA scenario cells, ZINC wins 19 of 20 prefill cells, 18 of 20 decode cells, and 19 of 20 phase-combined overall cells. The misses are tiny or workload-specific: Gemma 4 31B dense is just under decode parity on two context scenarios, and its long-draft combined score is `99.3%` of llama.cpp. End-to-end wall-clock throughput is noisier; ZINC wins 13 of 20 cells there because some longer context scenarios still pay request-path and harness overhead even when the phase-combined GPU work is ahead.

That distinction is important. If you ask "is ZINC ahead on the public RDNA headline rows?", the answer is yes. If you ask "is every single scenario cell ahead?", the answer is no. If you ask "is the remaining gap structural or measurement noise?", the answer depends on the row. The Gemma 31B dense long-draft row is a real next target because it is close enough that a small dispatch or prompt-path improvement should flip it without changing the model math.

## Why AMD is the interesting target

AMD local inference used to be framed as a compromise: you could get VRAM, but you had to accept rough software and weaker LLM tooling. The R9700-class result changes the useful question. It is no longer "can a 32 GB AMD card run these models?" It can. The question is whether a Vulkan-first engine can make that card competitive on the model families people actually want to run locally.

That is why this sweep is meaningful. The matrix is not five copies of the same model:

- Gemma 4 26B-A4B MoE tests sparse Gemma routing.
- Gemma 4 31B tests a large dense Gemma path.
- Qwen 3.5 9B tests a small interactive dense model.
- Qwen 3.8 27B tests the current larger dense Qwen path.
- Qwen 3.6 35B A3B tests the flagship sparse Qwen path.

Those are exactly the kinds of local GGUF targets that turn up in real usage: one fast chat model, one coding-sized dense model, one large sparse model, and Gemma variants because people do not want a runtime that only likes Qwen.

The result also matters because it is on Vulkan. ZINC_RT is still a separate bring-up runtime; the published RDNA runs use the production Vulkan path. That keeps the claim grounded in the path people can actually reproduce from the repository today.

## What llama.cpp still gives us

The point of comparing against llama.cpp is not to dunk on it. It is the reference implementation the local inference world keeps returning to because it is portable, practical, and full of hard-won kernel work. If ZINC cannot beat it on a fair same-machine run, the honest answer is that ZINC is not faster yet.

Using llama.cpp as the baseline also keeps our own metric honest. A single-engine benchmark can tell you whether a commit is faster than yesterday. It cannot tell you whether the engine is good. The same-file llama.cpp baseline gives every ZINC number context: driver state, model shape, prompt length, and the weirdness of the actual machine all get priced into both sides.

The best optimization work came from that pressure. When ZINC was far behind on Gemma, the answer was not "explain the chart better." It was fix Gemma. When the Qwen prefill numbers looked implausible, the answer was not "ship the big percentage." It was update the harness so the percentage meant phase time instead of a distorted score. When a server row failed, the answer was not "average around it." It was mark the row failed until it ran.

That is how the project moved from interesting kernels to credible benchmark claims.

## What happens next

The next useful target is not another headline slogan. It is shrinking the remaining close calls in the scenario matrix.

Gemma 4 31B dense is the obvious one. The headline row is ahead, but the margin is thin: `28.81 tok/s` versus `28.54 tok/s` decode, `103%` overall. In the longer scenario matrix, two Gemma 31B decode cells are just below parity and one long-draft overall cell is `99.3%`. That is where I would spend the next round of profiling. A tiny win there turns the dashboard from a headline sweep into something closer to a full scenario sweep.

The second target is end-to-end stability. Phase-combined overall is the right score for GPU work, but users experience wall-clock latency. If a context-heavy request shows strong prefill and decode yet worse end-to-end throughput, the server path still has work to do. That includes request handling, stop detection, scheduling, and any place where the host waits between GPU phases.

The third target is keeping the harness boring. Every new model needs to go through the same rules: same GGUF, clean node, reusable servers, warmup discarded, medians, provenance, and output sanity checks. The fastest way to lose a performance story is to let benchmark convenience drift back in.

## The current claim

Here is the version I am comfortable publishing:

ZINC is now ahead of llama.cpp on the current public AMD RDNA headline benchmark suite. On the Radeon AI PRO R9700, across five popular local GGUF targets covering Qwen, Gemma, dense models, and sparse MoE models, ZINC beats the same-file llama.cpp baseline on prefill, decode, end-to-end throughput, and phase-combined overall for every headline row. The strongest row is Qwen 3.6 35B A3B at `166.8 tok/s` decode, `154%` of llama.cpp. The closest row is Gemma 4 31B dense at `103%` overall, which is why Gemma remains the next profiling target rather than a solved problem.

That is the interesting place to be. The AMD path is no longer a portability experiment. It is now fast enough that the remaining work is measured in the same language as the best local engine in the ecosystem.
