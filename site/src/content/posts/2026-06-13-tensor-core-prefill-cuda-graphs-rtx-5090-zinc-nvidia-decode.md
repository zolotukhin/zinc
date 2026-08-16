---
title: "A day later: batched prefill, CUDA graphs, and the catalog re-measured on RTX 5090"
seoTitle: "RTX 5090: Batched Prefill & CUDA Graphs in ZINC"
date: "2026-06-13"
tags:
  - zinc
  - nvidia
  - cuda
  - rtx-5090
  - blackwell
  - tensor-cores
  - cuda-graphs
  - prefill
  - moe
  - qwen3
  - gemma
  - local-llm
  - llm-inference
  - gpu-kernels
  - kernel-fusion
  - llama-cpp
keywords:
  - RTX 5090 LLM prefill
  - tensor core prefill fp16 wmma
  - CUDA graphs LLM decode
  - CUDA graph replay inference
  - Qwen3.6 prefill NVIDIA
  - Gemma MoE prefill CUDA
  - batched prefill GEMM
  - launch-latency bound decode
  - ZINC vs llama.cpp prefill
  - Blackwell sm_120 tensor cores
  - GPU-side embedding lookup
  - local LLM RTX 5090 benchmark
faqs:
  - question: "How much faster is prefill in ZINC's CUDA backend, and how does it really compare to llama.cpp?"
    answer: "Two things drove it: batching (one GEMM over all prompt tokens at once, which took Gemma-4-26B from 8.3 to 106 tok/s) and then a better GEMM, because prefill is compute-bound on the 5090 — the Blackwell fp16 tensor cores on by default plus routing the dense matmuls through cuBLAS took Gemma-4-31B prefill from about 200 to 505 tok/s (~2.5x) at pp512. But measured the standard way (llama-bench pp512), llama runs Gemma-31B prefill at about 3300 tok/s, so ZINC is still ~6.5x behind on the dense model and ~26x on the 26B MoE. An earlier version of this post said prefill was launch-bound and that ZINC beat llama on Qwen prefill — both were measurement artifacts and are corrected here. The residual gap is that llama fuses dequantization into the GEMM (MMQ) while ZINC round-trips through an fp16 scratch."
  - question: "What do CUDA graphs do for LLM decode?"
    answer: "Decode issues a long chain of tiny kernels per token — on a 60-layer model, hundreds of launches whose per-launch overhead and inter-kernel bubbles dominate when each kernel is small. A CUDA graph captures that whole per-token chain once and replays it as a single submission, so the driver stops paying launch cost per kernel. In ZINC it is an opt-in mode worth about 8 to 12 percent on the small dense Qwen3.5-9B, and it is size-gated: the win shrinks to nothing on larger models where each matvec is big enough to hide the launch bubble, and it cannot capture the mixed-quantization MoE path at all yet."
  - question: "Does moving the per-token embedding lookup onto the GPU speed up decode?"
    answer: "On its own, no — and that null result is the interesting part. Dequantizing the token's embedding row on the GPU instead of the CPU, and shrinking the host-to-device copy from a full row to four bytes, is bit-for-bit correct but measured perf-neutral on Qwen3.5-9B, because decode there is bound by GPU launch latency, not host work. Removing host work cannot move a wall clock the host is not on. Its value is as a building block: with the token id and embedding already GPU-resident, multiple decode steps can eventually be chained into one CUDA graph with no per-token host round-trip."
  - question: "Where does ZINC's RTX 5090 decode stand against llama.cpp now?"
    answer: "Across the five-model catalog it averages about 70 percent of llama.cpp decode, up from 51 percent on the previously published snapshot. The dense models are close — the 27B Qwen row at 91 percent, Gemma-4-31B at 82 percent, Qwen3.5-9B at 75 percent — while the Mixture-of-Experts models still trail at 31 to 42 percent, where llama.cpp's years-tuned expert kernels keep the lead."
  - question: "Why were the published benchmark numbers so much lower than the current ones?"
    answer: "The published catalog snapshot was a correctness-first build from before the optimization work landed — no batched MoE experts, no kernel fusion, no tensor-core prefill. This post is the first full re-measurement after merging two parallel optimization lines, so the dashboard now reflects the engine as it actually runs rather than as it first booted. The merge was gated on a 5-of-5 token-for-token correctness check against llama.cpp before any number was trusted."
excerpt: "A day of NVIDIA work on ZINC's CUDA backend — and a correction. Decode went from 51 to 70 percent of llama.cpp; prefill got a real ~2.5x on Gemma-31B from batching plus a tensor-core/cuBLAS GEMM (prefill is compute-bound, not launch-bound as first reported). The honest ceiling: measured the standard way, ZINC is still ~6.5-26x behind llama's MMQ prefill GEMM, and the earlier 'beats llama on Qwen prefill' was a short-prompt measurement artifact. Plus CUDA-graph decode and a correct-but-no-op GPU embed."
seoDescription: "ZINC CUDA backend on RTX 5090: decode 51 to 70% of llama.cpp, a ~2.5x prefill GEMM win (batching + Blackwell tensor cores + cuBLAS, because prefill is compute-bound), the honest llama-bench gap (still ~6.5-26x behind llama's MMQ prefill), CUDA-graph decode, and a correct null-result GPU embed — with a correction to an earlier launch-bound misread."
draft: false
---

[Yesterday's post](/blog/2026-06-12-four-bottlenecks-one-cuda-backend-moe-gemma-tensor-cores-rtx-5090-4090/) was a tour of four different decode bottlenecks and ended, like most honest engineering posts, with a to-do list:

> Wire the fp16 tensor-core prefill · move the dense per-token glue onto the GPU · more Gemma fusion · close the MoE gap.

A day later, three of those are shipped, a fourth lever that *wasn't* on the list landed too — CUDA graphs — and the whole five-model catalog has been re-measured on the RTX 5090. This is what a day looks like when the levers were already sized: mostly wiring, one genuine surprise, and one "win" that turned out to be a perfectly correct no-op.

It also landed in an unusual shape. The prefill work and the decode work had been growing on **two separate branches** — one productionizing batched/tensor-core prefill, one chasing decode launch latency — and the first job today was merging them. That's an eleven-hunk collision across the CUDA kernel files (two refactors editing the same functions), so the merge was gated the only way a kernel merge can be honestly gated: rebuild and re-run the catalog token-for-token against llama.cpp. **5 of 5 models matched** before a single throughput number was trusted.

## The scoreboard, re-measured

Same box as before — an RTX 5090 (Blackwell, sm_120, 32 GB, 1792 GB/s) under WSL2, `ReleaseFast`, NVRTC-compiled kernels, the catalog GGUF files, greedy decode, measured over SSH against llama.cpp on the same hardware and files, medians of three runs.

First, the headline the to-do list was really about — **prefill**:

<figure class="diagram-card diagram-wide">

| RTX 5090 prefill (tok/s) | published | today | gain | vs llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| **Gemma-4-26B-A4B** (MoE) | 8.3 | **106.3** | **12.8x** | 0.26x |
| **Qwen3.6-35B-A3B** (MoE) | 14.8 | **45.5** | 3.1x | 0.94x |
| **Gemma-4-31B** (dense) | 29.9 | **57.3** | 1.9x | 0.15x |
| **Qwen3.5-9B** (dense) | 64.6 | **97.6** | 1.5x | **1.14x** |
| **27B Qwen** (dense) | 39.2 | **47.8** | 1.2x | **1.66x** |

  <figcaption>Prefill throughput, RTX 5090. "published" → "today" is a real self-improvement (batching + a better GEMM). <strong>But the "vs llama.cpp" column uses the perf-suite's short-prompt metric, which compresses the gap and undermeasures llama by ~9–135×</strong> — measured the standard way (llama-bench pp512) the real gaps are ~6× (Gemma-31B), ~23× (Gemma-26B MoE), and <strong>~62–158× on the hybrid-SSM Qwen models</strong> (no "wins"). The dashboard now shows these real pp512 throughput numbers. See the correction below.</figcaption>
</figure>

And decode, the metric this series tracks, with the catalog now reflecting reality instead of a months-old boot:

<figure class="diagram-card diagram-wide">

| RTX 5090 decode (tok/s) | published | today | gain | vs llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| **Gemma-4-26B-A4B** (MoE) | 8.3 | **47.5** | 5.7x | 31% |
| **Qwen3.6-35B-A3B** (MoE) | 16.3 | **52.9** | 3.2x | 42% |
| **Gemma-4-31B** (dense) | 33.9 | **46.9** | 1.4x | 82% |
| **Qwen3.5-9B** (dense) | 92.0 | **120.8** | 1.3x | 75% |
| 27B Qwen (dense) | 47.7 | **50.5** | 1.06x | 91% |

  <figcaption>Decode throughput, RTX 5090. The catalog average moved from 39.6 to 63.7 tok/s — from 51% to 70% of llama.cpp on the same hardware. Read the multiples as <em>cumulative</em>: the published snapshot predated batched MoE experts and kernel fusion, so this is the gap between "as it first booted, correctly" and "as it runs today," not a single day's decode delta.</figcaption>
</figure>

<img class="diagram-visual" src="/blog/2026-06-13-rtx-5090-prefill-decode.svg" alt="Two stacked horizontal bar charts for the RTX 5090. Top, prefill tok/s published versus today: Gemma-4-26B MoE 8.3 to 106.3 (12.8x), Qwen3.6-35B-A3B 14.8 to 45.5, Gemma-4-31B 29.9 to 57.3, Qwen3.5-9B 64.6 to 97.6, and a 27B dense Qwen row at 39.2 to 47.8. Bottom, decode tok/s published versus today: Gemma-4-26B 8.3 to 47.5, Qwen3.6-35B-A3B 16.3 to 52.9, Gemma-4-31B 33.9 to 46.9, Qwen3.5-9B 92 to 120.8, and the 27B dense Qwen row at 47.7 to 50.5." loading="lazy" />

Two honest framings on top of that. The win: prefill and decode both improved a lot over the stale snapshot, and the dense *decode* models are all within striking distance of llama.cpp. The gap: the MoE models still decode at 31–42% of llama.cpp, and **prefill is much further behind than the table suggests** — measured at pp512, Gemma prefill is ~6.5–26× behind llama's MMQ GEMM (the "vs llama.cpp" prefill column is a flattering short-prompt metric; see the correction below). A faster GEMM has closed part of it — tensor cores + cuBLAS, ~2.5× on the 31B — and the rest is fusing the dequant into the GEMM.

## What actually landed

### Prefill — a correction, and a better GEMM

> **Correction (2026-06-14).** An earlier version of this section claimed prefill was *launch-bound* (≈10% GPU util) and that tensor cores were "a wash." Both were wrong — that reading came from a 4090 profile taken during model-load. A fine-grained re-profile on the 5090 shows gemma prefill at **~100% util, full boost, ~400 W: it is *compute*-bound.** So a faster GEMM is exactly the lever, tensor cores *do* help, and the "vs llama.cpp" prefill column above is from a short-prompt metric that badly flatters the gap. Here's the corrected story.

Two things drove prefill. First, **batching**: one register-tiled GEMM over all prompt tokens at once instead of a per-token matvec (and, for MoE, the routed experts batched with a GPU-side work list) — that's the 8.3 → 106 tok/s on Gemma-4-26B. Then, because prefill is compute-bound, **a better GEMM compounds**: flipping the Blackwell fp16 tensor cores on by default (+25%), then routing the dense Q4_K and Q6_K matmuls through **cuBLAS** (dequant→fp16 + `cublasGemmEx`). Together that took **gemma-31B prefill from ~200 to ~505 tok/s (~2.5×) at pp512** — 5/5 token-correct, now on `main`.

And the honest ceiling. Measured the standard way — `llama-bench` pp512, same 5090, same GGUF — **llama runs gemma-31B prefill at ~3100 tok/s, gemma-26B at ~8000, and the Qwen models at ~2800–7600.** So ZINC's real prefill gap is **~6× on the dense 31B, ~23× on the 26B MoE, and a brutal ~62–158× on the hybrid-SSM Qwen models** — the last because zinc runs the SSM prefill scan sequentially where llama batches it. That's far worse than the "0.15×/0.26×" in the table, which uses the perf-suite's *short-prompt* prefill where both engines are overhead-bound, the gap compresses, and llama is undermeasured by ~9–135×. **The "ZINC beats llama on dense Qwen prefill" line was an artifact of that metric and is false at pp512** — the dashboard now shows the real pp512 throughput.

The residual gap is pure kernel efficiency: zinc's cuBLAS path dequantizes each weight to a full fp16 scratch and reads it back, while llama fuses the dequant *into* the GEMM (MMQ / on-the-fly). Killing that round-trip — a persistent fp16 weight cache, or an MMQ-style fused-dequant GEMM — is the next lever, and an autonomous effort is grinding on it now.

### CUDA graphs — the structural answer to launch latency

This is the lever that wasn't on yesterday's list, and it's aimed squarely at yesterday's most useful finding: that Gemma decode is **launch-latency bound**, idle inside the GPU waiting on its own launch queue across ~180 tiny serial dispatches per token, where removing host syncs does nothing.

A CUDA graph attacks that directly. Capture the entire per-token kernel chain — embed, every layer, the final norm/LM-head/argmax — into a `CUgraphExec` **once**, then replay it as a single submission. The driver stops paying launch overhead per kernel; only the per-token push-constant scalars change, which is a cheap in-place exec update. An isolated proof clocked the replay at ~9x the cost of relaunching the chain at the real ~60-layer length.

Wired into decode (behind `ZINC_CUDA_GRAPH`), interleaved A/B puts it at **~8–12% on the small dense Qwen3.5-9B**, with the embedding upload and the argmax readback folded into the graph as pinned-memory copy nodes so the whole token drains on one sync. And then it's honest about its ceiling: the win is **size-gated**. On the 27B it's a measured no-op — the matvecs are big enough that the launch bubble is already negligible — and it can't capture the catalog's MoE path at all, because those experts carry mixed quantization and the captured topology has to be identical every step. So it ships opt-in, a real win exactly where the model is small and the launches dominate, and nowhere else. That's a more useful result than a universal speedup would have been: it tells you precisely which models are launch-bound.

### Gemma fusion — the stacked 1%s, completed

The full attention-fusion stack from yesterday's "more Gemma fusion" line is in: the three per-head V/Q/K RMS-norms collapsed into one launch, the same-input Q4_K matvec pairs fused, and the per-layer pre-norms folded into the *preceding* block's post-norm+residual so the four per-layer norm boundaries land at two fused launches. Each is a 1–2% kernel win that only clears this box's wandering-boost noise floor when you stack them and require the fused build to win *every* interleaved round — which is how a 1.5% fusion becomes a number you publish instead of boost noise you regret.

### GPU-side embed — a correct no-op, and why it matters anyway

The last to-do item, "move the dense per-token glue onto the GPU," shipped — and measured as a **perfect no-op**, which is the most instructive result here. Dequantizing the token's Q4_K embedding row on the GPU (reading the id from a 4-byte device buffer) instead of on the CPU, shrinking the host→device copy from a full embedding row to four bytes, is bit-for-bit identical output and **perf-neutral** on the 9B: a 400-token interleaved A/B landed inside ±2.4% boost noise.

It's neutral for the same reason the CUDA graph is *not* — decode here is bound by GPU launch latency, not host work, so removing host work can't move a wall clock the host isn't standing on. The value isn't the kernel; it's the **primitive**. With the token id and its embedding already GPU-resident, the real next lever becomes possible: chaining several decode steps into a *single* CUDA graph with no per-token host round-trip at all. The no-op is the groundwork for the lever that isn't a no-op.

## The honest part: a wandering clock and a stale scoreboard

Two caveats keep this post from overclaiming. First, the published numbers it improves on were a **correctness-first snapshot** — the engine as it first booted, before any optimization — so the big decode multiples are the cumulative gap, not a single day's decode delta. The clean *day-over-day* win is prefill; decode's day-over-day movement is real but partly inside the 5090's boost variance, which on this box is wide enough that a single before/after reading on a 1.5% change is worthless. Every sub-noise number here came from interleaved A/B, not a naive comparison.

Second, the gap that's still open is the one yesterday named and this day didn't close: **MoE decode**. The router still keeps a per-layer sync, the expert matvecs are small-M and occupancy-starved, and llama.cpp's expert kernels are years ahead. 31–42% of llama is up from where it was, and it's still the frontier.

## What's next

- **Close the MoE decode gap** — a GPU-side gather that drops the last router round-trip, and a multi-row expert kernel for the small-M matvecs.
- **Tune Gemma prefill** — the tensor-core path is wired and 12.8x up on MoE, but dense Gemma prefill is still a fraction of llama's; the kernel needs the occupancy and pipelining work, not just the wiring.
- **Cross-step graph chaining** — now that embed and argmax are GPU-resident, chain N decode steps into one graph and delete the per-token host round-trip entirely. The neutral GPU-embed result is the door to this one.
- **MoE under graphs** — capture needs uniform expert quantization; a single-quant MoE build would unlock the launch-latency win for the experts too.

None of it needs new hardware. The recurring method from both posts held: a llama.cpp reference to diff every token against, a profiler to say *which* wall — sync, launch, dispatch, dequant — you're standing in front of, and boost-aware interleaved A/B so a real 1.5% win isn't drowned by a clock that won't sit still. A day's worth of that, and the scoreboard finally tells the truth.

*ZINC is a from-scratch local inference engine with Vulkan (AMD RDNA), Metal (Apple Silicon), and CUDA (NVIDIA) backends — one engine, hand-written kernels, no heavyweight frameworks. The CUDA backend runs all five catalog models coherently on RTX 4090 and 5090, validated token-for-token against llama.cpp.*
