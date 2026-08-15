---
title: "Admitting a ninth agent stalls the other eight for 8.3 seconds"
seoTitle: "Chunked Prefill on RDNA4: Why Local Schedulers Need a Smaller Chunk"
date: "2026-07-19"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - prefill
  - decode
  - scheduling
  - chunked-prefill
  - continuous-batching
  - agents
  - vulkan
  - local-llm
  - llm-inference
keywords:
  - chunked prefill local LLM
  - prefill decode interference RDNA4
  - stall-free scheduling Sarathi-Serve
  - max_num_batched_tokens chunk size
  - llama.cpp ubatch size 512
  - continuous batching admission latency
  - time to first token agent workload
  - RX 9070 XT prefill 962 tok/s
  - inter token latency local inference
  - decode step budget prefill chunk
excerpt: "Batching eight agents on one RX 9070 XT gives 5.6 times the aggregate tokens, but only while the batch stays still. The moment a ninth agent arrives with an 8,000-token prompt, a prefill-first scheduler freezes the other eight for 8.3 seconds. Chunked prefill fixes the freeze and charges for it, and the chunk size that actually keeps a local decode batch smooth is about 71 tokens, well under a seventh of what llama.cpp and vLLM ship."
seoDescription: "On one RX 9070 XT running Qwen3.5-9B Q4_K_M, prefill is 962 tok/s and a batched decode step is 25.7 ms, which means one decode step has room for about 25 tokens of prefill work. Admitting an 8k prompt unchunked stalls eight running sequences for 8.3 seconds. Chunking at llama.cpp's 512-token physical batch leaves them at 1.8 tok/s; holding them above 10 tok/s needs a 71-token chunk and adds 3.2 seconds to the newcomer's time to first token. Chunked prefill is throughput-neutral and purely redistributes latency, which makes it a question about who is waiting."
faqs:
  - question: "Why does one new request stall a whole batch of running ones?"
    answer: "Because a prefill and a decode step are different shapes, and most schedulers run prefill first. vLLM's own documentation says the default scheduler prioritizes prefills and does not batch prefill and decode into the same batch, which optimizes time to first token but incurs slower inter-token latency. On one RX 9070 XT an 8,000-token prompt takes 8.3 seconds to prefill at 962 tok/s, and for that whole window the eight sequences already decoding produce nothing."
  - question: "What is chunked prefill?"
    answer: "Splitting a long prompt into pieces and attaching one piece to each decode step, so new work is admitted without pausing generation. It comes from Sarathi-Serve, which calls the result a stall-free schedule, and it is now the default in vLLM and SGLang. The prefill still costs the same total compute. It is spread across many steps instead of taken all at once."
  - question: "What chunk size should a local engine use?"
    answer: "Smaller than the server defaults. The right unit is how much prefill work fits inside one decode step, and on an RX 9070 XT that is 962 tok/s times 25.7 ms, about 25 tokens. A 71-token chunk holds every running sequence above 10 tok/s. llama.cpp's physical batch default of 512 leaves them at 1.8 tok/s and vLLM's 2,048 leaves them at 0.5."
  - question: "Does chunked prefill cost throughput?"
    answer: "In the model, no. Doing 8,000 tokens of prefill plus 2,000 tokens of decode takes about 14.75 seconds either way, because the chunked steps are the decode steps that were going to run anyway. In practice it costs something, because a 32-row prefill GEMM does not reach the 962 tok/s a 2,048-row one does, and RDNA4 WMMA tiles are 16 wide so short chunks pad. That penalty is real and I have not measured it."
  - question: "If it is only a latency redistribution, why bother?"
    answer: "Because for a human the two failure modes are not equivalent. Eight seconds of frozen text is intolerable and 19 tok/s is invisible. For agents, where nothing is reading the stream, the argument is weaker and mostly comes down to timeout behavior and fairness between subagents rather than perceived speed."
draft: false
---

Yesterday's post ended on a number I liked: eight concurrent sequences on one RX 9070 XT pull about [311 aggregate tok/s](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) against 55.6 at batch one, because the 5.5 GB of Q4_K_M weights that dominate a decode token get read once for the whole batch. A 5.6x for free.

That number assumes the batch of eight is sitting still. Agent workloads do not sit still. A coding agent spawns subagents when it decides to, not when the scheduler is ready, and each one shows up holding a system prompt, a file listing, and a few thousand tokens of context it wants read before it says anything.

So here is the question that actually decides whether local batching works. A ninth agent arrives with an 8,000-token prompt while eight are mid-generation. What happens to the eight?

## A prefill-first scheduler freezes everyone

In ZINC today, and in most local engines, the answer is that the eight stop.

Prefill and decode are different shapes. Prefill reads the whole prompt at once, so it is a compute-bound matrix-matrix problem. Decode advances one token per sequence, so it is a bandwidth-bound matrix-vector problem. They do not naturally share a step, and the simplest scheduler runs the prefill on its own and resumes decoding afterwards. vLLM's documentation is blunt that this was its own original behaviour: "By default, vLLM scheduler prioritizes prefills and doesn't batch prefill and decode to the same batch. This policy optimizes the TTFT (time to the first token), but incurs slower ITL (inter token latency) and inefficient GPU utilization."

Put ZINC's measured numbers into that. Prefill on Qwen3.5-9B and one RX 9070 XT runs at [962 tok/s](/blog/2026-07-14-a-qwen3-5-9b-chat-turn-spends-most-of-its-wall-clock-in-decode/). An 8,000-token prompt is 8.3 seconds of it. During those 8.3 seconds the eight running sequences advance zero tokens, and each of them was producing a token every 25.7 ms. Their time between tokens goes from 25.7 ms to 8.3 seconds, a factor of 325.

The user-visible version of that is a chat window that stops mid-sentence for eight seconds because something else asked a question. The agent version is worse in a quieter way: eight subagents sit holding their KV cache, producing nothing, while the card does work for a ninth.

## The fix is known, and the local numbers are not the server numbers

The standard answer is chunked prefill, from [Sarathi-Serve](https://arxiv.org/abs/2403.02310). The paper's framing is exact: it "introduces chunked-prefills which splits a prefill request into near equal sized chunks and creates stall-free schedules that adds new requests in a batch without pausing ongoing decodes." Instead of one 8,000-token prefill step, you attach a slice of the prompt to each decode step. Nobody stops. The technique is now the default in vLLM and SGLang, and it works.

The part that does not transfer is the chunk size.

The unit that matters is how much prefill work fits inside one decode step before the step stops feeling like a decode step. On this card that is the prefill rate times the step time: 962 tok/s times 25.7 ms, which is 24.7 tokens. A chunk of about 25 tokens doubles the step. Anything larger means the running sequences are waiting on prefill arithmetic, just in smaller pieces than before.

Now compare that to what ships. llama.cpp splits work into a logical batch and a physical micro-batch, and its [server defaults](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) are `-b, --batch-size N` at 2048 and `-ub, --ubatch-size N` at 512. vLLM's chunked prefill defaults `max_num_batched_tokens` to 2048, and its [tuning docs](https://docs.vllm.ai/en/v0.8.2/performance/optimization.html) are clear about the direction of the knob: "Smaller max_num_batched_tokens achieves better ITL because there are fewer prefills interrupting decodes. Higher max_num_batched_tokens achieves better TTFT." Both defaults are twenty to eighty times larger than one decode step on this card can absorb.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-19-rdna4-prefill-chunk-size-admission-tradeoff.svg" alt="A line chart on a dark teal background titled 'Every prefill chunk size on an RX 9070 XT gives something up'. The horizontal axis is prefill chunk size on a log scale from 32 to 8192 tokens. A solid amber curve on the left axis shows the per-sequence decode rate while a newcomer prefills, falling from 17.0 tokens per second at a 32-token chunk to 10.8 at 64, 6.3 at 128, 3.4 at 256, 1.8 at 512, 0.5 at 2048 and effectively zero at 8192. A dashed cyan curve on the right axis shows the time to admit the 8,000-token prompt, falling in the opposite direction from 14.7 seconds at a 32-token chunk to 11.5 at 64, 9.3 at 256, 8.9 at 512, 8.6 at 2048 and 8.3 at 8192. A shaded amber band covers chunk sizes below 71 tokens, labelled as the region where every agent stays above 10 tokens per second. Two dotted vertical lines mark llama.cpp's 512-token physical batch default and vLLM's 2048-token max_num_batched_tokens default, both far to the right of the band. A footer notes that at both vendor defaults a running sequence emits under two tokens per second for the whole admission window, and that holding all eight agents above 10 tokens per second costs 3.2 extra seconds before the ninth agent sees its first token." loading="lazy" />
  <figcaption>Modeled admission of one 8,000-token prompt into a batch of eight decoding sequences. Qwen3.5-9B Q4_K_M on one RX 9070 XT, using a 962 tok/s prefill rate and a 25.7 ms batched decode step.</figcaption>
</figure>

The two curves cross nothing, which is the point. They move in opposite directions across the entire useful range, and the vendor defaults both sit in the flat right-hand tail where the newcomer has already captured almost all the time-to-first-token it is ever going to get and the running sequences have given up almost everything. Going from a 2,048-token chunk to a 71-token one costs the arriving agent 2.6 seconds of extra wait and buys the eight incumbents a twentyfold improvement in their token rate. That is a lopsided trade in one direction, and the fact that neither default takes it says something about who these schedulers were built for.

| Chunk size | Step time | Steps to admit 8k | Running agents decode at | Newcomer waits |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 59.0 ms | 250 | 17.0 tok/s | 14.7 s |
| 71 | 99.5 ms | 113 | 10.1 tok/s | 11.2 s |
| 128 | 158.8 ms | 63 | 6.3 tok/s | 10.0 s |
| 512 (llama.cpp `-ub`) | 557.9 ms | 16 | 1.8 tok/s | 8.9 s |
| 2048 (vLLM default) | 2154.6 ms | 4 | 0.5 tok/s | 8.6 s |
| none | 8341.7 ms | 1 | 0.1 tok/s | 8.3 s |

Read the last two columns as the whole argument. The rightmost column barely moves, spanning 8.3 to 14.7 seconds across a 256x change in chunk size, because the 8.3 seconds of prefill compute is a constant and everything above it is decode overhead paid repeatedly. The fourth column moves by two orders of magnitude. Chunk size is a knob that costs the arriving request very little and pays the running requests a great deal, right up until the chunk gets small enough that the step is mostly fixed overhead.

## It is a redistribution, not a win

The honest framing is that chunked prefill does not make the card faster.

Work out the total. Doing 8,000 tokens of prefill and 2,000 tokens of decode with a 32-token chunk takes 250 steps at 59.0 ms, about 14.75 seconds. Doing the same work unchunked takes 8.34 seconds of prefill plus 250 decode steps at 25.7 ms, about 14.77 seconds. Those are the same number. The chunked steps are the decode steps that were going to happen anyway, with prefill arithmetic riding along inside them.

So the entire decision is about latency distribution, and that reframes it as a question about who is waiting rather than a question about the GPU. For a person watching text appear, the two failure modes are wildly different: eight seconds of nothing is a bug report, and 19 tok/s instead of 39 is imperceptible because both are faster than reading. For an agent, where no one is watching the stream, the case is weaker. It comes down to whether a subagent's HTTP client times out during the freeze, and whether one agent's long prompt is allowed to starve seven others, which is a fairness property rather than a speed one.

There is also a cost I have not accounted for, and it cuts against small chunks specifically. The model above assumes every chunk prefills at the full 962 tok/s, and it will not. That rate comes from a GEMM with thousands of rows. A 32-row prefill chunk is a short, skinny matmul, and [RDNA4 WMMA operates on 16x16 tiles](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/) that pad whatever you give them, so a 32-row chunk uses two tiles at whatever efficiency a two-tile GEMM reaches. Somewhere below a few hundred rows the effective prefill rate drops and the amber curve in that chart bends down faster than drawn. Where exactly is a measurement I owe.

## What I am changing

The scheduler work from yesterday's post now has a second requirement attached. Promoting the paged KV manager to the default was about holding several sequences at once. This is about letting a new one in without stopping the others, and the two only work together.

The concrete plan is a token budget per step rather than a chunk constant: reserve the decode rows for every in-flight sequence first, then fill the remainder of the step with prefill rows from the admission queue, with the budget expressed as a target step time rather than a token count. A target of 50 ms on this card lands near a 24-token chunk and follows the hardware when the model, the quantization, or the card changes. Then measure the prefill rate at 16, 32, 64 and 128 rows, because that curve decides whether the target is 50 ms or 150.

The broader thing I keep relearning this month is that every local inference number has an assumption underneath it about what the workload looks like. The 5.6x from batching assumed a static batch. The 8.3-second stall is what happens when that assumption meets a workload that arrives in bursts. The scheduler is where those two facts have to be reconciled, and it has been the least interesting file in the repo for four months.
