---
title: "How we made CUDA LLM decode up to 5× faster: batched MoE experts and kernel fusion in ZINC"
seoTitle: "CUDA LLM decode optimization: batched MoE + kernel fusion (RTX 4090)"
date: "2026-06-11"
tags:
  - zinc
  - cuda
  - nvidia
  - rtx-4090
  - rtx-5090
  - blackwell
  - llm-inference
  - gpu-optimization
  - kernel-fusion
  - mixture-of-experts
  - moe
  - decode-throughput
  - qwen3
  - gemma
  - llama-cpp
  - cuda-kernels
  - local-llm
keywords:
  - CUDA LLM inference optimization
  - batched MoE experts CUDA
  - kernel fusion LLM decode
  - MoE decode throughput RTX 4090
  - local LLM optimization NVIDIA
  - beat llama.cpp decode
  - mixture of experts GPU kernel
  - gemma qwen CUDA tokens per second
  - GPU boost starvation decode
description: "A diagnosis-driven tour of how ZINC's CUDA backend turned its slowest models into front-runners — batching mixture-of-experts decode and fusing gemma's attention kernels — with the benchmarks, the boost-clock gotchas, and the self-driving optimization loop that found the wins while we slept."
seoDescription: "ZINC CUDA backend decode optimization: batched mixture-of-experts experts and gemma kernel fusion took the catalog's slowest models to the front on the RTX 4090, validated token-for-token against llama.cpp."
excerpt: "A mixture-of-experts model was decoding at 9 tok/s on an RTX 4090 — slower than a dense 27B. This is how ZINC's CUDA backend diagnosed boost-clock starvation, batched its MoE experts into two GPU-side kernels (turning the catalog's slowest models into front-runners that outrun the big dense ones), fused gemma's attention dispatches, and kept every change token-for-token correct against llama.cpp."
faqs:
  - question: "Why was a mixture-of-experts model decoding slower than a dense model on the same GPU?"
    answer: "Launch overhead, not arithmetic. ZINC's MoE path read the chosen expert IDs back to the host every layer, a hard CPU-to-GPU sync, then fired a swarm of tiny per-expert matvecs too small to occupy the GPU. The SM clock bounced between about 345 and 2445 MHz, boost-starved, so a 3-billion-active-parameter MoE model decoded near 9 to 10 tok/s, below a dense 27B at 36."
  - question: "How much faster is batched MoE decode in ZINC?"
    answer: "Batching the experts into two GPU-side kernels, one fused gate and up projection and one down projection, with expert IDs read from the router buffer instead of the host, took Qwen3.6-35B-A3B from about 9 to 40 tok/s and Gemma-4-26B-A4B from about 10 to 52 tok/s, both on an RTX 4090 (about 4 to 5x). The MoE models went from the catalog's slowest to decoding above the dense 27B and 31B."
  - question: "Why did not the async CUDA stream ring speed up gemma like it did qwen?"
    answer: "Because gemma was boost-saturated, not sync-starved. Its clock was already pinned near 2715 MHz, so there was no per-token sync idle to reclaim and the async ring's bookkeeping only added overhead. The async ring was the big win on the sync-starved dense qwen path, pushing decode past 104 tok/s and beating llama.cpp's 97. Diagnose the clock before picking the fix."
  - question: "What is kernel fusion and how much does it help LLM decode?"
    answer: "Fusion collapses several tiny per-layer GPU dispatches into one fatter kernel, such as RMSNorm plus residual add, or per-head norm plus RoPE plus KV-cache write. It adds no FLOPs and only removes launch latency, so it helps a boost-saturated path. Each gemma fusion is worth about 1.5 to 6 percent on dense decode and they stack; an autonomous loop landed six bit-equivalent fusions."
  - question: "How do you benchmark GPU decode without fooling yourself?"
    answer: "Validate token-for-token against llama.cpp on every change so a fused kernel must be bit-equivalent; run interleaved A/B by building both binaries and alternating rounds in the same thermal state so boost noise cancels; and force a fresh build with isolated cache directories, verifying the binary hash changed, because a stale cached binary will silently measure your old code."
---

A mixture-of-experts model was decoding at **9 tokens per second** on an RTX 4090. The same GPU runs a *dense* 27B model — which touches far more weights per token — at 36 tok/s. Something was badly wrong, and the fix turned out to have almost nothing to do with arithmetic.

This is the second half of [ZINC's CUDA story](/blog/2026-06-07-how-zinc-got-a-cuda-backend-and-beat-llama-cpp-decode-on-nvidia/). The first half was getting a Zig inference engine to produce a *correct token* on NVIDIA at all. This half is about making it **fast** — and the punchline is that the two biggest wins came from the same realization, applied in opposite directions: **on modern GPUs, LLM decode is usually starved, not compute-bound — and you have to measure which kind of "not fast" you're looking at before you write a single kernel.**

Here is how CUDA LLM decode in ZINC went from "embarrassing" to "beats llama.cpp," what we tried that *didn't* work (and why those failures were the useful part), and the autonomous loop that now hunts these wins on its own.

## The setup: five models, one backend, zero room to be wrong

ZINC is a from-scratch inference engine written in Zig. On the NVIDIA box it has exactly one backend — **CUDA** — because WSL2 exposes no NVIDIA Vulkan ICD, so there is no fallback to hide behind. Everything runs through hand-written kernels compiled at runtime with NVRTC.

The bar is a **five-model catalog**, validated **token-for-token against llama.cpp** on every change:

- `qwen35-9b` — dense hybrid-SSM
- retired 27B Qwen row — dense
- `qwen36-35b-a3b` — mixture-of-experts (128 experts, 8 active)
- `gemma4-31b` — dense, sliding-window attention
- `gemma4-26b-a4b` — mixture-of-experts

"Faster" only counts if all five still produce the **identical greedy token sequence** as llama.cpp afterward. A perf win that breaks correctness is not a win; it is a bug with good marketing.

## Lesson 1: profile the *clock*, not the kernels

The instinct with a slow matvec is to optimize the matvec. That instinct was wrong here. We sampled the SM clock during decode and found two completely different failure modes hiding behind the same low tok/s:

- **The MoE model was boost-*starved*.** Its SM clock bounced bimodally between ~345 MHz and ~2445 MHz. Every layer did a tiny host read-back of which experts were chosen, then fired a swarm of microscopic per-expert matrix-vector multiplies that never fed the GPU enough work to *hold* its boost clock. The GPU spent most of each token idling at its base clock waiting for the CPU.
- **The dense gemma model was boost-*saturated*.** Its clock sat pinned at ~2715 MHz, ~55% utilized, drawing only ~70 W of a 450 W budget. It was *already* boosted; the idle time was GPU-internal launch latency between ~180 tiny serial kernels, not CPU stalls.

Same symptom — low throughput — opposite cause. The starved model needs **fewer synchronizations**; the saturated model needs **fewer, fatter kernels**. Pointing either fix at the wrong model does nothing. (We proved that the embarrassing way: more on the no-op below.)

## Win #1: batched MoE experts — the slowest models overtook the dense ones

The starved MoE path was death by a thousand tiny launches. Each decode layer:

1. computed router logits, then **read the chosen expert IDs back to the host** (a hard CPU↔GPU sync), and
2. looped over the 8 active experts running a few minuscule matvecs each — `M = 512` weight rows, far too small to occupy 128 SMs.

The fix replaces the whole per-expert loop with **two batched kernels** — one launch each for the fused gate/up projection and the down projection — that sweep over *all* active experts in a single grid, reading the expert IDs **GPU-side** straight from the router output buffer. No host read-back, so the entire MoE block runs asynchronously; far fewer, far fatter launches, so the GPU stays fed.

<img class="diagram-visual" src="/blog/2026-06-11-cuda-moe-batching.svg" alt="Bar chart of mixture-of-experts decode throughput before and after batched experts, both on an RTX 4090. Qwen3.6-35B-A3B goes from about 9 tokens per second with the per-expert host-readback loop to about 40 with batched experts, a 4.4 times gain. Gemma-4-26B-A4B goes from about 10 tokens per second to 52, over 5 times. Both are bit-equivalent and token-correct versus llama.cpp, and both end up above the dense 27B and 31B models." loading="lazy" />

On the **RTX 4090**, batching took `qwen36-35b-a3b` from **~9 → ~40 tok/s** (a **4.4×** jump) and the gemma MoE model (`gemma4-26b-a4b`) from **~10 → ~52 tok/s** (over **5×**). Both paths are bit-equivalent — all five catalog models still match llama.cpp token-for-token.

Here is the part that still surprises me: **these MoE models went from the *slowest* things in the catalog to outrunning the big dense ones.** Before batching, the 3-to-4-billion-active-parameter MoE models crawled at ~9–10 tok/s — *below* the dense 27B (36) and 31B (30). After batching they clear **40 and 52 tok/s — above both dense models.** The whole point of MoE is that you only pay for the few experts you activate; that advantage was entirely hostage to launch overhead, and batching set it free.

The lesson generalizes well beyond ZINC: **MoE decode lives or dies on launch count and host round-trips, not on FLOPs.** A 3B-active model that decodes slower than a 27B dense one is almost always paying in synchronization, not arithmetic.

## Win #2: kernel fusion for the boost-saturated path

Gemma's dense model was the opposite problem, so it got the opposite fix. There was no sync to remove — the GPU was already boosted — so the lever was collapsing its swarm of tiny per-layer dispatches into fewer, larger ones. Each fusion is **bit-equivalent** (token-identical to the unfused path) and each was verified with an interleaved A/B:

- **`rms_norm_residual`** — folds the post-attention/FFN RMSNorm and its residual add into one kernel (the residual scale is always 1.0 there), dropping a launch *and* a full hidden-state round-trip per block.
- **`rms_norm_rope`** — fuses the per-head Q/K RMSNorm and rotary embedding into a single launch, staging the normalized head in shared memory.
- **`rms_norm_kvwrite`** — normalizes V straight into the KV cache and folds the K-cache write into the rope kernel via a destination offset, deleting the standalone KV-write launch.

Individually each is worth ~1.5–6% on gemma-31b dense decode, and because they touch different parts of the layer they **stack**. The autonomous optimization loop (below) has since landed **six** of these fusions — adding an output-scale fold, a same-input dual matvec, and a norm-boundary fuse on top of the three above — each one bit-equivalent and individually A/B-verified. Unglamorous, compounding, free.

## The failure that taught us the most: async did *nothing* for gemma

The async command ring — overlapping kernels on a CUDA stream so the CPU never blocks per-op — was the single biggest win on the *dense qwen* path earlier: it took decode from a bimodal ~22–66 up past **104 tok/s, beating llama.cpp's 97**. So the obvious move was to point it at gemma too.

It was a **measured no-op — actually a slight regression.** Why? Because gemma was already boost-saturated. There was no sync-induced idle to reclaim; the ring's bookkeeping just added overhead and held the clock *lower*. qwen benefited because it was *starved*; gemma never was.

We reverted it and wrote the negative result into memory in bold: **do not async gemma's MoE.** Negative results are not wasted cycles — they are the map of where *not* to dig, and on a five-model catalog that map saves more time than any single win.

## How we keep the numbers honest

Decode benchmarking on a boosting GPU is a minefield, and most of our early "wins" were boost noise. The discipline that survived:

- **Token-for-token correctness, every change.** `validate_catalog.sh` runs all five models against a greedy llama.cpp reference. A fused kernel must be bit-equivalent; if a single early token diverges, it reverts. (gemma-31b has one famous EOS-vs-newline near-tie where ZINC's f32 is actually *more* accurate than llama's q8 GPU reference — so the gate has a teacher-forced fallback rather than flagging the more-correct engine.)
- **Interleaved A/B, never sequential.** We build both binaries and run them **back-to-back in the same thermal state**, alternating rounds. A win has to take the majority of rounds. One boosted run proves nothing — a single qwen-35b-a3b run once flashed 60 tok/s against a steady-state of 40, and only the re-runs caught it.
- **The build-cache gotcha.** Zig's cache — *and especially its global cache* — will happily hand you a stale binary that silently measures your *old* code as a no-op. Every benchmark forces a fresh build with isolated cache dirs and **verifies the binary hash actually changed**. This one cost us a confusing afternoon before it went into the runbook.

<img class="diagram-visual" src="/blog/2026-06-11-cuda-catalog-decode.svg" alt="Bar chart of decode throughput in tokens per second on an RTX 4090 for the five-model ZINC CUDA catalog after optimization. Qwen3.5-9B dense is highest near 90, Gemma-4-26B mixture-of-experts about 52, Qwen3.6-35B-A3B mixture-of-experts about 40, the retired 27B dense Qwen row about 36, and Gemma-4-31B dense about 30. The two mixture-of-experts models sit above the larger dense 27B and 31B. All five are five out of five token-correct versus llama.cpp." loading="lazy" />

## The part where the loop optimizes itself

The most fun piece: most of these fusions are now found by an **autonomous optimization loop**. Each perf effort is a scoped markdown plan (`MULTI_HOUR_EFFORT_*.md`) with explicit targets and hard rules. A loop spawns a fresh agent per cycle that reads the plan, makes **one** focused change, builds with isolated caches, runs the 5/5 correctness gate and an interleaved A/B, and commits the change to a `perf/*` branch **only if it is a validated, repeatable win** — otherwise it reverts and logs *why*, so the next cycle never re-tries a dead end.

The gemma fusions in this post? Most were landed by that loop, across a multi-hour run, while we did other things — each with its A/B rounds and 5/5 gate already in the commit message. A single overnight effort produced **six** bit-equivalent fusions before it ran out of cheap launches to collapse. Human review merges the keepers to `main`. It is continuous integration for performance: the loop proposes, the correctness gate disposes.

## Takeaways for anyone optimizing LLM decode on a GPU

1. **Measure the clock first.** Low tok/s has at least two causes — sync-starvation and launch-latency saturation — and they want opposite fixes. The clock trace tells you which.
2. **MoE decode is a launch-count problem.** Batch the experts, read routing GPU-side, kill the host round-trip. Done right, your MoE models overtake the big dense ones instead of trailing them.
3. **Fusion is free money on a saturated path.** Collapse tiny dispatches; they don't add FLOPs, only launch latency, and the savings compound.
4. **Async helps the starved, not the saturated.** Point it at the wrong model and you'll regress. Test, don't assume.
5. **A boosting GPU lies.** Interleave your A/Bs, verify your binary hash, and validate correctness on every single change — or your benchmark is fiction.

ZINC's CUDA backend now runs a five-model catalog **token-correct on the RTX 4090**, with the MoE models decoding **4–5× faster** than where they started — now ahead of the big dense models — and dense decode that beats llama.cpp. And the loop is still running. We'll keep posting the numbers.
