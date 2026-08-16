---
title: "Four bottlenecks, one CUDA backend: MoE, Gemma, and tensor cores on RTX 5090 and 4090"
seoTitle: "RTX 5090 LLM Decode: MoE & Gemma in ZINC's CUDA Backend"
date: "2026-06-12"
tags:
  - zinc
  - nvidia
  - cuda
  - rtx-5090
  - rtx-4090
  - blackwell
  - moe
  - qwen3
  - gemma
  - tensor-cores
  - local-llm
  - llm-inference
  - gpu-kernels
  - kernel-fusion
  - llama-cpp
keywords:
  - RTX 5090 LLM inference
  - RTX 4090 local LLM decode
  - CUDA MoE decode
  - Qwen3.6 35B A3B NVIDIA
  - Gemma local LLM CUDA
  - mixture of experts GPU decode
  - batched MoE experts CUDA
  - tensor core prefill fp16 wmma
  - Blackwell sm_120 tensor cores
  - CUDA kernel fusion decode
  - llama.cpp vs ZINC decode
  - GPU boost clock decode bound
  - dual GPU 5090 vs 4090 decode
  - NVRTC cuda_fp16 mma include
faqs:
  - question: "How fast does Qwen3.6-35B-A3B (MoE) decode on an RTX 5090 in ZINC?"
    answer: "About 37 tok/s, up from 16 — roughly 2.3x. The gain came from reading the chosen expert IDs GPU-side, batching the routed experts into single launches instead of per-expert dispatches, and async-pipelining the MoE command stream so only the router keeps a host sync. Gemma-4-26B-A4B improved even more on the same hardware, from 8.3 to 40.4 tok/s."
  - question: "Does a faster NVIDIA GPU always mean faster LLM decode?"
    answer: "No. The RTX 5090 has 1792 GB/s of bandwidth to the RTX 4090's 1008 GB/s — 1.8x — but decode of a small dense model is often bound by CPU command-building and per-token glue, not GPU bandwidth. Qwen3.5-9B decodes at about 104 tok/s on the 4090 and 118 on the 5090: a real gain, but far short of 1.8x, because the bottleneck is partly off the GPU."
  - question: "Why doesn't the async stream ring speed up Gemma decode like it did Qwen?"
    answer: "Because Gemma is not sync-bound. Profiling showed Gemma-4-31B decode already holding full boost clock at about 55 percent GPU utilization — it is launch-latency bound across roughly 180 tiny serial kernel dispatches per token, not blocked on host syncs. Porting the async ring to it was a validated no-op. The lever that worked was kernel fusion: collapsing the per-layer norm, residual, RoPE, and matvec dispatches into fewer, fatter launches, for about a 7 to 8 percent decode gain."
  - question: "Are the Blackwell tensor cores used for prefill yet?"
    answer: "The build flag that unblocks them landed: NVRTC now passes -I/usr/local/cuda/include, so fp16 and mma.sync kernels compile in-tree. The proven lever is dequantizing the weights to an fp16 scratch buffer once and running a pure fp16 wmma GEMM, which measures about 2.2x over the fp32 register-blocked prefill GEMM. Wiring that fp16 path into the live prefill loop is the immediate next step."
  - question: "Where does ZINC's NVIDIA decode stand against llama.cpp?"
    answer: "On the dense Qwen3.5-9B it is ahead — about 104 tok/s versus llama.cpp's 97 on the same RTX 4090 and model file. Across the full five-model RTX 5090 catalog it averages about 59 percent of llama.cpp decode, up from 51 percent, with the Mixture-of-Experts models the furthest behind and improving fastest (2.3x to 5x). llama.cpp's MoE kernels are heavily tuned; closing that gap is the next chapter."
excerpt: "ZINC's first NVIDIA post beat llama.cpp on one model by fixing one bottleneck. The catalog has five models, and they do not share a bottleneck. Dense decode was sync-bound. MoE was strangled by per-expert dispatch and a router round-trip. Gemma was launch-latency bound, where the async ring is a no-op and only kernel fusion helps. Prefill was dequant-bound, waiting on tensor cores. This is how four different walls fell — Qwen3.6-35B-A3B MoE 16 to 37 tok/s, Gemma-26B MoE 8 to 40 — on RTX 5090 and 4090."
seoDescription: "How ZINC made its whole local-LLM catalog fast on NVIDIA RTX 5090/4090: batched MoE experts and an async router for Qwen3.6 and Gemma, kernel fusion for launch-latency-bound Gemma decode, and the fp16 tensor-core prefill lever — with verified tok/s versus llama.cpp."
draft: false
---

The [first NVIDIA post](/blog/2026-06-07-how-zinc-got-a-cuda-backend-and-beat-llama-cpp-decode-on-nvidia/) ended on a clean number: Qwen3.5-9B decoding at ~104 tok/s on an RTX 4090, ahead of llama.cpp's 97. One model, one bottleneck — per-token sync gaps starving the GPU boost clock — and one fix, an async stream/event ring that removed sixty-four syncs per token.

Then we pointed it at the rest of the catalog, and the clean story fell apart in the most useful way.

Five models — three Qwen, two Gemma, two of them Mixture-of-Experts — and **they do not share a bottleneck**. The async ring that won the 9B did *nothing* for Gemma. The MoE models were strangled by something the dense path never touched. Prefill was bound by a third thing entirely. Each plateau looked, again, like a kernel problem, and was again usually something else — but a *different* something else every time.

This is how four different walls came down.

## The scoreboard first

Same box throughout: an RTX 5090 (Blackwell, sm_120, 170 SMs, 32 GB, 1792 GB/s) and an RTX 4090 (Ada, sm_89, 128 SMs, 24 GB, 1008 GB/s) under WSL2, `ReleaseFast`, NVRTC-compiled kernels, the catalog GGUF files, greedy decode, measured over SSH against llama.cpp on the same hardware and the same files.

<figure class="diagram-card diagram-wide">

| RTX 5090 decode (core) | before | after | gain | bottleneck that fell |
| --- | ---: | ---: | ---: | --- |
| **Gemma-4-26B-A4B** (MoE) | 8.3 | **40.4** | **4.9x** | per-expert dispatch + router readback |
| **Qwen3.6-35B-A3B** (MoE) | 16.3 | **37.4** | **2.3x** | per-expert dispatch + router readback |
| **Qwen3.5-9B** (dense) | 92 | **118** | 1.3x | sync-bound boost starvation |
| **Gemma-4-31B** (dense) | 34 | **39** | 1.15x | launch-latency (fusion, not async) |
| 27B Qwen (dense) | 48 | 42 | flat | — within 5090 boost variance |

  <figcaption>Decode throughput, tok/s, RTX 5090. "before" is the correctness-first / pre-lever build; "after" is the current build. The catalog average moved from 39.6 to 55.4 tok/s — from 51% to 59% of llama.cpp on the same hardware. The two MoE models, the worst performers, improved the most.</figcaption>
</figure>

<img class="diagram-visual" src="/blog/2026-06-12-cuda-catalog-bottlenecks.svg" alt="Horizontal bar chart of RTX 5090 decode throughput before and after each model's bottleneck fix. Gemma-4-26B MoE goes from 8.3 to 40.4 tok/s, a 4.9x gain; Qwen3.6-35B-A3B MoE from 16.3 to 37.4, 2.3x; Qwen3.5-9B dense from 92 to 118, 1.3x; Gemma-4-31B dense from 34 to 39, 1.15x. The muted part of each bar is the before value and the bright part is the gain." loading="lazy" />

Two honest framings sit on top of that table. The win: the dense 9B is *ahead* of llama.cpp (104 vs 97 on the 4090), and every model that was embarrassing is now respectable. The gap: across the catalog ZINC averages ~59% of llama.cpp decode, and the MoE models are still the furthest behind — Gemma-26B's 40.4 is against llama's 113 on that model. llama.cpp's MoE path is years-tuned. The interesting part is not the absolute standing; it is that getting here required diagnosing four unrelated bottlenecks correctly, and the obvious fix was wrong for three of them.

## Bottleneck 1 — dense decode was sync-bound (recap)

The 9B's story is [the previous post](/blog/2026-06-07-how-zinc-got-a-cuda-backend-and-beat-llama-cpp-decode-on-nvidia/), so the one-paragraph version: decode was not compute-bound, it was *sync*-bound. The 32-layer stack issued ~65 blocking `commitAndWait` calls per token, the CPU stalled on each, and in the gaps the GPU dropped its clock — a **525 MHz** median during sync-bound decode against **2520 MHz** during sustained prefill on the same card. The async `CUstream`/`CUevent` ring submits the whole token's work once and lets the CPU run ahead; throughput went from a bimodal 22/66 to a steady ~104, and the GPU held boost. Keep that mechanism in mind, because the next model refuses to play along with it.

## Bottleneck 2 — MoE decode was paying for its routing twice

The Mixture-of-Experts models — Qwen3.6-35B-A3B and Gemma-4-26B-A4B — were the worst performers by far: 16 and 8 tok/s. They had two problems the dense path never had, and both are about *dispatch*, not math.

First, the **router round-trip**. Every MoE layer computes router logits, picks the top-k experts, and then has to act on that choice. The correctness-first path did the obvious thing: compute the logits, `commitAndWait`, read the chosen expert IDs back to the host, and *then* issue the expert work. That host readback is a hard sync in the middle of every single layer — the exact boost-killing stall the async ring was built to remove, except the dense ring couldn't touch it, because it was gated off for MoE (`n_experts == 0`) precisely so the proven dense path stayed untouched.

Second, **per-expert dispatch**. With top-k of 8, the naive loop launched a gate matvec, an up matvec, a SwiGLU, and a down matvec *per chosen expert* — 32 tiny launches per MoE layer, each a kernel that touches one expert's slice. Small, serial, launch-overhead-dominated.

The fix is two moves that compound:

- **Batched experts.** Replace the per-expert loop with single launches — `dmmv_q4k_experts` for the gate/up, `dmmv_q5k_experts` (Qwen) or `dmmv_q5_1_experts` (Gemma) for the down — that read the chosen expert IDs *from a GPU buffer* and process all `n_used` experts in one grid. No host readback to decide what to launch; the kernel indexes the right expert weights itself.
- **Async the MoE stream.** With the IDs read GPU-side, the routed-expert and shared-expert commands no longer depend on a host value mid-layer. Only the router itself keeps a sync. The rest pipelines onto the ordered stream exactly like the dense ring.

The combination moved Qwen3.6-35B-A3B from **16.3 to 37.4 tok/s** and Gemma-4-26B-A4B from **8.3 to 40.4** — a 2.3x and a 4.9x, on a path that had been ignored because "the dense ring already works." It didn't, for these models, because their bottleneck was never the dense ring's bottleneck. The shared expert needed its own correctness care, too: Gemma-26B's per-expert down-projection carries a scale that has to be applied inside the batched async path, not folded away — a one-line fix that was the difference between coherent and confidently wrong.

## Bottleneck 3 — Gemma decode is launch-latency bound, and the async ring is a no-op

Here is where the tidy "async ring fixes decode" narrative breaks, and the break is the most useful result in this post.

Gemma-4-31B is dense. The async ring won the dense 9B. So porting the ring to the dense Gemma path should win again — and it was a clean, correct port. It changed nothing.

The profiler said why. Gemma-31B decode was **not sync-bound**. It was already holding ~2715 MHz — near full boost — at about 55% GPU utilization and only ~70 W of a 450 W budget. There is no boost to recover by removing syncs, because the syncs weren't starving the clock. The ~45% idle is *inside* the GPU: launch latency across roughly **180 tiny serial kernel dispatches per token**. A CPU-side ring that stops the CPU from blocking does nothing for a GPU that's waiting on its own launch queue.

So the same diagnosis tool that said "remove syncs" for Qwen said "remove *launches*" for Gemma. The lever is **kernel fusion**: collapse the small per-layer dispatches into fewer, fatter ones. A run of validated fusions did exactly that —

- fold the post-attention and post-FFN `(rms_norm → scale_accumulate)` pairs into one `rms_norm_residual` launch (the residual scale is 1.0 there, so it's bit-equivalent);
- fuse the per-head Q/K RMS-norm with RoPE into one kernel;
- fuse the per-head V norm with the KV-cache write;
- collapse the three V/Q/K per-head norms into a single launch;
- fuse the same-input Q4_K matvec pairs (gate/up, Q/K);
- fold the per-layer output scale into the preceding norm+residual.

Each is worth around 1 to 2 percent; stacked, they're **about 7 to 8 percent** on Gemma-31B dense decode (34 → 39 tok/s), and they're *complementary* — different parts of the layer — so they add rather than compete.

The part worth dwelling on is the measurement. At these sizes, on a card whose boost clock wanders, a 1.5% kernel win is below the noise floor of a naive before/after — and the 24 GB memory cap on the box makes the 21 GB Gemma weights thrash the page cache on every load, so you cannot just warm it up and read a clean number. The only honest signal came from **interleaved A/B**: run fused and baseline back-to-back, many rounds, same thermal state, and require the fused build to win *every* round. It did, 4 for 4, which is how a 1.5% fusion becomes a number you'll publish instead of boost noise you'll regret.

Same backend, same model family, opposite diagnosis from the dense Qwen. Sync removal and launch removal are different medicines for different diseases, and the GPU tells you which one it has if you read the clock and the utilization instead of guessing.

## The faster card doesn't always win

Worth pulling out, because it's counterintuitive and it's a recurring theme above. The RTX 5090 has **1.8x** the RTX 4090's memory bandwidth. Decode is supposed to be bandwidth-bound. So the 5090 should be ~1.8x faster at decode.

It isn't. Qwen3.5-9B runs ~104 tok/s on the 4090 and ~118 on the 5090 — about 1.15x. The reason is the same reason the async ring mattered: once the matvecs are bandwidth-efficient and the syncs are gone, what's left per token is partly *off* the GPU — embedding dequant, the argmax readback, command building. A faster card can't accelerate the CPU glue. The 5090's bandwidth pulls ahead clearly only where the work is actually big and GPU-resident: the large MoE models and prefill, where it's doing real sustained matrix throughput, not waiting on per-token bookkeeping.

It's a good reminder that "buy more bandwidth" is the wrong instinct when your profiler says the GPU is idle waiting on the host.

## Bottleneck 4 — prefill was dequant-bound, and the tensor-core door just opened

Decode is bandwidth and latency. Prefill is arithmetic — a real GEMM — and it had its own wall.

The fp32 path got most of the way there: a register-blocked 2D-tiled GEMM (64×64 output tile, each of 256 threads computing a 4×4 register micro-tile, weights and activations each reused 64×) hits **5.9x** over running the decode matvec per token, ~9254 GFLOP/s for Q4_K, with Q5_K and Q6_K close behind. That's a real prefill kernel — and still only ~9% of the card's fp32 peak, because at these tile sizes it's occupancy- and latency-bound, not compute-bound.

The next lever is tensor cores, and it was honest about its ceiling and blocked on a build flag. The measured win is *not* the 3–6x a tensor-core headline promises. Dequantize the Q4_K weights to an fp16 scratch buffer **once**, then run a pure fp16 `wmma` GEMM with no dequant inside the inner loop, and you get **~2.2x** over the fp32 register-blocked GEMM (17978 vs 8812 GFLOP/s) — real, but still only ~4% of the fp16 peak, because even that kernel is occupancy-bound and reaching the true peak needs a cuBLAS-class async-pipelined monster. The reason it was only *measured*, never *shipped*: the fp16 and `mma.sync` headers (`cuda_fp16.h`, `mma.h`) need NVRTC to be told where CUDA's includes live, and ZINC's runtime compiler wasn't passing `-I/usr/local/cuda/include`. So the whole fp16 family was stranded outside the shared kernel file.

**That flag just landed.** NVRTC now passes the include path; fp16 and tensor-core kernels compile in-tree, all 25 pipelines build clean, the catalog stays coherent. The proven ~2.2x fp16-scratch prefill kernel is now a wiring job, not a research question — dequant the prefill weights to fp16 once per matrix and route the tiled GEMM through `wmma`. It's the one remaining lever from the last post that was "a one-line change away," and the one line is in.

## Four walls, one tool

Step back and the catalog is a catalog of bottlenecks as much as models:

<figure class="diagram-card diagram-wide">

| Path | Looked like | Actually was | The fix |
| --- | --- | --- | --- |
| Dense decode (Qwen 9B) | slow kernels | per-token host syncs starving boost | async stream/event ring |
| MoE decode (35B-A3B, 26B-A4B) | slow experts | router readback + per-expert dispatch | GPU-side IDs + batched experts + async |
| Dense decode (Gemma 31B) | needs the async ring | launch latency, already boost-saturated | kernel fusion (async is a no-op) |
| Prefill (all) | needs tensor cores | scalar dequant feeding the math | fp16 scratch + wmma (now unblocked) |

  <figcaption>Same `src/cuda` backend, same NVRTC kernel library, same profiler. Four paths, four distinct bottlenecks, and in three of the four the obvious fix — faster kernels, the async ring, tensor cores — was the wrong first move until the profiler said otherwise.</figcaption>
</figure>

The recurring lesson from the first post held, and got sharper: the bottleneck is rarely the thing you're looking at. But the new edge is that **it's not even the same bottleneck across models that share a backend**. A Mixture-of-Experts model and a dense model, or two dense models from different families, can be limited by entirely different layers of the stack — host sync, dispatch count, launch latency, dequant throughput — and a fix that's transformative for one is provably inert for the next. The only thing that generalized was the method: a llama.cpp reference to diff correctness against, a profiler to read the clock and the utilization, boost-aware interleaved A/B so a real 1.5% win isn't drowned by a wandering clock, and a [validation harness](/blog/2026-06-07-how-zinc-got-a-cuda-backend-and-beat-llama-cpp-decode-on-nvidia/) that exits non-zero the instant a token drifts.

## What's next, and what it's worth

The levers are known and sized, which is the good place to be:

- **Wire the fp16 tensor-core prefill** now that NVRTC can compile it — the proven ~2.2x over the fp32 GEMM, for prefill / time-to-first-token.
- **Close the MoE gap to llama.cpp** — the router still keeps a per-layer sync, and the expert matvecs are small-M and occupancy-starved; a GPU-side gather that drops the last router round-trip and a multi-row expert kernel are the next moves.
- **Move the dense per-token glue onto the GPU** — embedding dequant and the argmax readback are now a measurable slice of a 118 tok/s token.
- **More Gemma fusion** — the per-head V norm and the remaining small elementwise dispatches are still separate launches.

None of it needs new hardware. It needs what the rest of this needed: read the clock before you rewrite the kernel, and let the GPU tell you which wall you're actually standing in front of.

*ZINC is a from-scratch local inference engine with Vulkan (AMD RDNA), Metal (Apple Silicon), and now CUDA (NVIDIA) backends — one engine, hand-written kernels, no heavyweight frameworks. The CUDA backend runs all five catalog models coherently on RTX 4090 and 5090.*
