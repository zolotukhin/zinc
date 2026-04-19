---
title: "Why RDNA4 prefill for Qwen3.5-35B is stuck at 25 tok/s"
date: "2026-04-18"
tags:
  - zinc
  - rdna4
  - amd
  - vulkan
  - prefill
  - qwen3-5
  - llm-inference
  - gpu-kernels
  - performance
keywords:
  - RDNA4 prefill
  - AMD RDNA4 LLM inference
  - Qwen3.5 35B A3B prefill
  - Vulkan prefill optimization
  - RX 9070 inference
  - RADV cooperative matrix
  - consumer AMD LLM inference
  - batched DMMV Vulkan
  - SSM projection bottleneck
  - MoE prefill scheduling
  - prefill vs decode GPU
  - llama.cpp Vulkan prefill
  - ZINC RDNA4 prefill
  - time to first token AMD
faqs:
  - question: "Why is prefill slower than decode in ZINC on AMD RDNA4?"
    answer: "Because ZINC's RDNA prefill path still iterates the decode loop once per prompt token. That means each weight tensor is re-read from VRAM per token instead of once per prompt chunk, which is the opposite of what a correctly batched prefill kernel would do."
  - question: "What is the current ZINC prefill number on Qwen3.5-35B?"
    answer: "The best checkpoint after 24 cycles of RDNA4 prefill optimization is 25.67 tok/s on the flagship long-context benchmark, while decode on the same model and hardware runs at roughly 73 tok/s."
  - question: "Which optimization cycles actually moved the prefill number?"
    answer: "Only three. Cycle 4 skipped the final norm, LM head, and last-layer FFN/MoE for non-terminal prefill tokens. Cycle 5 double-buffered the prefill command buffer and embedding staging. Cycle 20 extended cycle 4's skip into the full last-layer attention block."
  - question: "Why did seven barrier-narrowing cycles produce no improvement?"
    answer: "On RDNA4 with RADV, back-to-back compute barriers collapse or reorder inside the driver enough that cosmetic scope narrowing in GLSL does not change what the GPU actually waits for. Removing a barrier by restructuring what reads what is a different experiment, and it is the one worth running."
  - question: "What is the planned architectural fix for ZINC RDNA4 prefill?"
    answer: "Route the four SSM projections for a prompt chunk through one cooperative matrix matmul instead of 154 single-token DMMVs, batch the MoE router the same way, and add a Q8_1 activation quantization step plus a mul_mmq path for the largest DMMVs, following the pattern in llama.cpp's Vulkan backend."
excerpt: "Prefill on Qwen3.5-35B on AMD RDNA4 in ZINC is stuck at 25.67 tok/s while decode runs at 73. After 24 optimization cycles, only three moved the number. Here is what the measurements said about why, and what they say about how local inference engines should actually be tuned on consumer AMD."
---

We ran 24 optimization cycles on RDNA4 prefill for the Qwen3.5-35B-A3B flagship inside ZINC. Three of them moved the number. The other 21 measured flat or negative.

The current best checkpoint is **25.67 tok/s prefill**, on the same machine where the same model decodes at **73.07 tok/s**. That is a runtime where sampling one token is faster than reading one token in. On a 154-token long-context benchmark, our total latency sits near **13 seconds**. [llama.cpp](https://github.com/ggml-org/llama.cpp) on the same hardware, same model, same prompt finishes the whole thing in around **230 milliseconds**.

That gap is not noise. It is not a tuning constant. It is the shape of a deeper problem that only shows up when you try to optimize prompt ingestion on consumer AMD. This post is the honest write-up of what the measurements said, what moved, what did not, and why the story matters for anyone trying to run a 35B-class hybrid model on an RX 9070 locally.

If you want the broader engine context first, read [Every design decision behind ZINC](/blog/2026-04-03-every-design-decision-behind-zinc), [How Mixture of Experts models work in ZINC](/blog/2026-04-04-how-moe-models-work-in-zinc), and the earlier [RDNA4 4 tok/s to 27 tok/s post](/blog/2026-03-29-the-shaders-standing-between-4-tok-s-and-27-tok-s).

## Prefill is not decode in a loop

The naive framing of prompt ingestion is that prefill is just decode repeated N times. Read the weights, produce a hidden state, append to KV, advance. That framing is wrong in the one place it matters: weight bandwidth.

Qwen3.5-35B-A3B reads roughly **1.28 GB of weights per token** on a decode step. The RX 9070 peaks at around 640 GB/s of memory bandwidth on its 256-bit GDDR6 bus. A correctly batched 154-token prefill reads each weight tensor **once** against an N-wide activation block, and the bandwidth ceiling becomes ~200 ms of pure weight traffic plus whatever compute the tiles impose. Sequential prefill reads each weight **154 times** and is bounded by the same 1.28 GB × 154 = 197 GB of weight traffic multiplied by the per-token dispatch tax.

This is why [llama.cpp's Vulkan path](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp) uses one family of kernels for decode and a different family for prefill. The DMMV (dequantized matrix-vector) shaders live at the core of their decode path. For prefill, they compile eight variants of the shader with `NUM_COLS` baked in as a GLSL specialization constant, and when N crosses a tuned threshold they route to a proper mul_mm matmul kernel backed by cooperative matrix tiles. The relevant code sits in [`mul_mat_vec_base.glsl`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_base.glsl) and its sibling shaders. This is not a clever trick. It is the default in any production inference engine that takes prefill seriously.

ZINC's current RDNA path does not do any of that. Our `prefillBatch` iterates the decode loop once per prompt token. That is the single largest reason we are slow.

## What actually moved the number

The three cycles that produced a real best-checkpoint improvement all share one property. None of them tuned a shader. None of them touched quantization. All three restructured **when** work happens, not **how** a single dispatch runs.

| Cycle | Change | Delta | Shape |
| --- | --- | ---: | --- |
| 4 | Skip `final_norm` + LM head + last-layer FFN/MoE for non-terminal prefill tokens | ~+3% | Dead work elimination |
| 5 | Double-buffered prefill command buffer and embedding staging | ~+7% | CPU records N+1 while GPU runs N |
| 20 | Extend cycle 4's skip into the full last-layer attention block | ~+2% | More dead work elimination |

The pattern is blunt. The wins came from telling the GPU to stop running code whose results were never going to be read, and from getting the CPU out of the critical path by recording the next token's command buffer while the previous one was still in flight. Neither change required a new shader.

The flat and negative cycles are the more interesting half of the story.

## What did not move, and why we should stop trying

Seven separate cycles tried narrowing a back-to-back `computeBarrier` into a buffer-scoped barrier. Cumulative movement across those seven cycles was under 0.25 tok/s. On RDNA4 with RADV, the driver collapses or reorders adjacent compute barriers aggressively enough that cosmetic scope narrowing does not change what the GPU actually waits for. Narrowing a barrier in GLSL is a different experiment than removing the barrier by restructuring what reads what, and only the second one is worth running.

A different set of cycles tried pair-batching adjacent prompt tokens through the existing batch DMMV shader (`dmmv_q8_0_batch.comp`) with `num_cols = 2`. Even after a clean wave64 rewrite, this variant measured net-negative. The root cause was not the inner loop. It was the per-layer staging chain around the dispatch: copy column 0 and column 1 activations into an adjacent-column buffer, issue a single batched DMMV, wait on the barrier, split four output buffers back into two per-token paths so downstream ops do not care. That entire chain costs more than the L2 weight-cache win that back-to-back single-column DMMVs already give you for free on RDNA4.

That is the lesson llama.cpp learned years ago. You do not win pair-batching by slapping `num_cols = 2` on a shader that was compiled for `num_cols = 1`. You win it by compiling one shader per column count and letting the register allocator see the static inner loop. Pair-batching without specialization is a net loss on this driver.

A third failed pattern was extending the double-buffered pipeline from 2-deep to 3-deep. Cycle 5's double-buffering already closed the record-submit-wait gap on this workload. Going deeper added nothing because nothing else was starving.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/rdna4-prefill-phase-budget.svg" alt="A horizontal bar chart of the RDNA4 prefill phase budget on Qwen3.5-35B. SSM projections dominate at 1.30 seconds, followed by the rest of the SSM block at 0.50 seconds, MoE expert work at 1.60 seconds, attention at 0.70 seconds, CPU and submit gap at 0.60 seconds, shared expert at 0.08 seconds, and final norm plus LM head at 0.10 seconds." loading="lazy" />
  <figcaption>Phase budget captured with `ZINC_PREFILL_PROFILE=1` on the flagship long-context benchmark. SSM projections alone consume more time than the entire attention block. Any prefill plan that does not attack this bucket will keep missing the number.</figcaption>
</figure>

The chart is the reason we are not chasing micro-wins anymore. SSM proj is a single bucket that has never been batched across prompt tokens, and at 1.3 s it is larger than attention and the final tail combined. The MoE block is larger in total, but it is spread across router, top-k, gate-up, swiglu, down, and weighted accumulation. SSM proj is four adjacent quantized matvecs per layer across 30 layers, called 154 times with the exact same weights. That is the loudest unexploited signal in the whole effort log.

## The RDNA4-specific shape of the problem

Part of what makes this bring-up confusing is that RDNA4 has everything a real prefill path needs. The RX 9070's 64 compute units and 640 GB/s bandwidth are comfortably in the range where 300 tok/s prefill for a 35B A3B model is not exotic. The [GLSL cooperative matrix extension](https://github.com/KhronosGroup/GLSL/blob/main/extensions/khr/GLSL_KHR_cooperative_matrix.txt) is implemented in current Mesa, and RADV now supports `VK_KHR_cooperative_matrix` on RDNA4 with `RADV_PERFTEST=coop_matrix`. We use the same flag in the [Qwen3.6 on AMD and Metal post](/blog/2026-04-17-qwen-3-6-is-now-generally-available-in-zinc) and it works.

The hardware is not the blocker. The issue is that ZINC's RDNA prefill never left the decode-shaped schedule. The dispatch helper that would let us batch matvecs across prompt tokens exists in the source tree. It has zero callers in the prefill hot path. The shader backing it exists. It is flag-gated, untested at scale, and not wired into the runtime's hot loop. We have been pattern-matching "inference engine" onto "decode-style forward pass repeated N times" because that is how the engine was born.

The architectural fix is well-scoped but real. Route the four SSM projections for a prompt chunk through one cooperative matrix matmul instead of 154 single-token DMMVs. Do the same for the MoE router. Quantize the activations once to Q8_1 at the prefill chunk boundary and use the equivalent of llama.cpp's [`quantize_q8_1.comp`](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/vulkan-shaders/quantize_q8_1.comp) plus a mul_mmq path for the largest DMMVs. Keep flash attention on its current [FlashAttention-style](https://github.com/Dao-AILab/flash-attention) decode path only while we are ingesting one token at a time, and switch to a block-ingest attention kernel once the rest of the prefill path stops treating the prompt as a serial stream.

None of this requires inventing a new backend. It requires admitting that decode and prefill are different workloads and deserve different kernels.

## What this means for optimization culture

There is a style of GPU work where every cycle is a clever shader rewrite. A tile size change, a subgroup shuffle, a smaller shared memory footprint, a rebalanced register pressure diagram. That style is deeply satisfying to execute. It is also the style that produced 21 out of 24 flat cycles in this effort.

On consumer AMD, the story is usually structural. The driver collapses what it can, the L2 cache forgives a lot of naive access patterns, and the peak compute is generous enough that micro-optimizations rarely surface in end-to-end numbers. What does surface is whether you are running the right kernel at all, whether the command buffer is full when the GPU is ready, and whether you are re-reading gigabytes of weights that a correctly batched path would have touched once.

That is a less glamorous way to think about optimization work, but it is the one the measurements keep pointing to. The telemetry is already there, behind `ZINC_PREFILL_PROFILE=1`. The dormant helpers are already in the tree. The reference design from llama.cpp is open source and linked above. Every cycle that skips those and tries another shader microvariant is choosing to relitigate a question the evidence has already answered.

## What comes next

The next meaningful move is not a kernel tweak. It is wiring the existing `recordBatchDispatch` helper into the SSM projection call site in the prefill loop, one quantization format at a time, starting with Q8_0 because that is what the SSM weights already use. After that, the router, then the MoE gate-up and down paths, and only then an honest fight with attention on whether prefill should call flash_attn at all or a dedicated block-ingest kernel.

If the plan works, prefill on the flagship should clear Phase 2 (150 tok/s) within a small number of cycles. If it does not, the chart above will tell us exactly which phase did not move, and we will have learned something sharper about RDNA4 than "prefill is slow."

Either way, the boring takeaway stands. On consumer AMD, inference engines are not usually shader-limited. They are schedule-limited. The faster we internalize that, the faster the prefill number starts moving for real.

For the background on the runtime pieces, the most relevant reads are [the earlier RDNA4 shader deep-dive](/blog/2026-03-29-the-shaders-standing-between-4-tok-s-and-27-tok-s), [how we got from 7 to 33 tok/s on AMD RDNA4](/blog/2026-03-30-how-we-moved-zinc-from-7-tok-s-to-33-tok-s-on-amd-rdna4), and the most recent [Qwen3.6 on AMD and Metal](/blog/2026-04-17-qwen-3-6-is-now-generally-available-in-zinc) announcement. If you just want to run a model, start with [Getting Started](/zinc/docs/getting-started) and [Running ZINC](/zinc/docs/running-zinc).
