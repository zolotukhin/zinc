---
title: "Eight concurrent agents pull 5.6 times more tokens from one RX 9070 XT"
seoTitle: "Batch One Is the Local LLM Assumption Agent Workloads Break on RDNA4"
date: "2026-07-18"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - decode
  - batching
  - continuous-batching
  - kv-cache
  - agents
  - vulkan
  - local-llm
  - llm-inference
keywords:
  - local LLM concurrent batching RDNA4
  - batch one decode throughput RX 9070 XT
  - continuous batching local inference
  - shared weight stream decode batching
  - RDNA4 WMMA 16x16x16 padding batch size
  - KV cache VRAM budget 16 GB
  - agentic workload local inference
  - paged KV cache single user engine
  - llama.cpp parallel slots
  - aggregate tokens per second local GPU
excerpt: "A month of kernel work moved one RDNA4 decode token from 25.2 ms to a modeled 18.0 ms. Adding a second concurrent sequence would have added more aggregate throughput than all of it, because 10.7 of those 18 milliseconds are weight bytes that every sequence in a batch reads exactly once. Local engines are built around a single user typing, and the workload on top of them is turning into four agents fanning out at the same time."
seoDescription: "Weight streaming is 10.7 ms of an 18.0 ms Qwen3.5-9B decode token on one RX 9070 XT, and it is the one term that does not scale with batch size. Modeling eight concurrent sequences gives about 311 aggregate tok/s against 55.6 at batch one, a 5.6x, with each sequence still decoding at 38.9 tok/s. RDNA4 WMMA tiles are 16x16x16, so batch 1 and batch 16 issue the same matrix instructions. The real ceiling is VRAM: a 16 GB card holds about 55,000 fp16 KV tokens after weights, which is where the paged KV cache this blog dismissed in May comes back."
faqs:
  - question: "Why does adding sequences barely increase the per-token time?"
    answer: "Because the largest term in the token is shared. Streaming 5.5 GB of Q4_K_M weights takes about 10.7 ms on a 640 GB/s card, and those bytes belong to the model rather than to any one sequence. A decode step that advances eight sequences reads them once and multiplies eight hidden states against them. Only the KV read, the LM head matmul and sampling scale per sequence, and at a 2k context those are small. The modeled token goes from 18.0 ms at one sequence to 25.7 ms at eight, so aggregate throughput rises from 55.6 to about 311 tok/s."
  - question: "Does batching hurt the latency the user actually feels?"
    answer: "It costs some, and less than it looks. In the model each of eight concurrent sequences decodes at 38.9 tok/s instead of 55.6, so a single stream is about 30 percent slower. For a human reading output that difference is invisible, since both are far above reading speed. For agents it does not matter at all, because nothing is watching the tokens arrive. The case where it does matter is one interactive user with one urgent request, which is exactly the case where the batch is one anyway."
  - question: "What stops you from just running 32 sequences?"
    answer: "VRAM, and then arithmetic. Qwen3.5-9B costs about 147 KB per token of fp16 KV cache. After 5.5 GB of weights, a 16 GB RX 9070 XT has room for roughly 55,000 KV tokens in total across every sequence in flight, so sixteen agents get about 3.4k context each. Past a batch of 16 the RDNA4 WMMA tile is full and the matmul stops being free, so the token time starts climbing linearly."
  - question: "Doesn't llama.cpp already do this?"
    answer: "Yes, and that is part of the point. The llama.cpp server exposes `-np, --parallel N` for server slots and enables continuous batching by default. The gap is not the algorithm, it is the default: ZINC sizes a single contiguous KV arena for one sequence because that was the honest read of a local workload a year ago. The workload is what changed."
draft: false
---

The last four posts spent a month of engineering moving one decode token from 25.2 ms down to a modeled 18.0 ms. Command buffer reuse took the CPU rebuild off the critical path. Kernel fusion deleted half the pipeline barriers. Every millisecond of that was real work against real profiler output, and it carried ZINC from 39.6 tok/s to about 55 on one RX 9070 XT.

Adding a second concurrent sequence would have added 49 tok/s of aggregate throughput in an afternoon.

That is not an argument against the kernel work. The kernel work is what makes the batching win as large as it is. It is an argument that ZINC, and most local engines, have been optimizing a workload that is quietly changing shape underneath them. The assumption baked into the whole decode arc is a batch of one: one person, one conversation, one token at a time. The thing actually driving local inference in 2026 is a coding agent that spawns four subagents and waits for all of them.

## The weight read is the only term that does not scale

Take the [post-fusion token budget](/blog/2026-07-17-decode-kernel-fusion-on-rdna4-deletes-the-barrier-not-the-launch/) apart and ask, for each line, whether a second sequence makes it more expensive.

Weight streaming is 10.7 ms of the 18.0. That is 5.5 GB of Q4_K_M weights moving across a 640 GB/s bus, and those bytes do not belong to a sequence. They belong to the model. A decode step that advances eight sequences at once loads each weight tile into registers once and multiplies eight hidden states against it before moving on. Eight sequences, one read.

The KV read is 0.6 ms and it is genuinely per-sequence, because every sequence has its own cache. The small-kernel term grows a little, since the norms and the SwiGLU now process eight rows instead of one. The barrier bubbles do not grow at all, because the same kernel chain with the same barriers serves the whole batch. The LM head splits: reading the 151k-row output embedding is shared, the matmul against it and the sampling are not.

So the term that dominates the token is the term that stays fixed, and the terms that scale are the small ones. That is the whole mechanism. Decode at batch one is not just memory-bound, it is memory-bound on bytes it is willing to share for free.

## RDNA4 pads the batch dimension to 16 whether you fill it or not

There is a second, more specific reason the arithmetic barely moves, and it is baked into the hardware.

RDNA4's matrix cores expose Wave Matrix Multiply Accumulate instructions, and AMD's own [guide to using them](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/) is explicit about the shape: "WMMA operates on matrices of 16x16 dimension only. Thus, if the matrix we are working on is smaller than that, the matrices need to be padded." The M dimension of a decode GEMM is the number of sequences. At batch one you issue a 16x16x16 instruction with fifteen rows of zeros in it.

The consequence is blunt. Batch 1 and batch 16 issue the same matrix instructions and burn the same matrix-core cycles. Sequences two through sixteen ride in padding that the hardware was going to compute anyway.

The honest caveat is that ZINC's batch-one decode path does not use WMMA at all. A one-row GEMM is bandwidth-bound, so the fast path is a dequant-matvec kernel, the same one the [decode loop](/blog/2026-04-30-rdna4-matrix-cores-sit-out-the-decode-loop/) has been sitting on since April while the matrix cores idle. Batching flips which kernel is correct: past roughly four sequences the WMMA path wins, and once you are on it the tile is sixteen rows wide regardless of how many you fill. The matrix engine that decode has been ignoring for three months turns out to be the thing that makes concurrency cheap.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-18-rdna4-decode-concurrency-shared-weight-stream.svg" alt="A two-panel diagram on a deep purple background. The top panel shows five horizontal stacked bars, one per concurrency level, on a shared 0 to 35 millisecond axis, breaking one decode step of Qwen3.5-9B on an RX 9070 XT into five segments: blue weight stream, coral KV read, lime small kernels, pink barriers, and cyan LM head plus sampling. At one sequence the bar is 10.7 plus 0.6 plus 3.2 plus 2.0 plus 1.5, totalling 18.0 milliseconds. At two sequences it is 19.1 milliseconds, at four 21.3, at eight 25.7, and at sixteen 34.5. A dashed blue vertical line at 10.7 milliseconds runs through every bar, annotated that the weight bytes are read once no matter how many sequences read them, because the blue segment is identical in all five rows while the coral KV segment grows from 0.6 to 9.6 milliseconds. The bottom panel is five horizontal lime bars of aggregate throughput across all sequences: 56 tokens per second at one sequence, 105 at two, 188 at four, 311 at eight, and 464 at sixteen, each labelled with the per-sequence rate beside it, falling from 55.6 to 52.4, 46.9, 38.9 and 29.0 tokens per second. A footer notes that RDNA4 WMMA tiles are 16 by 16 by 16, so batch 1 and batch 16 issue the same matrix instructions and the rows below 16 were padding." loading="lazy" />
  <figcaption>Modeled decode step for Qwen3.5-9B Q4_K_M on one RX 9070 XT at about 2k context per sequence, starting from the 18.0 ms post-fusion budget. The blue weight-stream segment is byte-for-byte identical in all five rows; only the coral KV read grows proportionally.</figcaption>
</figure>

The thing to look at is the dashed line. Every bar in the top panel starts with the same 10.7 ms block, because that block is the model reading itself into the compute units, and it happens once per step no matter who is waiting on the result. What grows to its right is KV, which is 0.6 ms per sequence at a 2k context, plus a little kernel and sampling work. Eight sequences cost 43 percent more wall clock per step and produce eight times the tokens, which is where the 5.6x comes from. The bottom panel is the same fact stated two ways: aggregate goes up steeply, per-sequence goes down gently, and even at sixteen concurrent sequences each one is still emitting 29 tokens a second, well above the speed anyone reads at.

## The workload stopped being one person typing

The reason to care about this in July 2026 rather than last year is that the thing sitting on top of a local engine changed.

Anthropic's write-up on [building a multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) is specific about the shape: the lead agent "spins up 3-5 subagents in parallel rather than serially," and the subagents call three or more tools in parallel on top of that. The same post reports that agents use roughly four times the tokens of a chat interaction, and multi-agent systems about fifteen times. That is a workload with a natural concurrency of four or five and an appetite for tokens that a batch of one services badly.

A local engine running that workload at batch one is not just slow, it is spending 10.7 ms of weight streaming per agent per token when it could spend it once for all of them. Four subagents at batch one take four sequential 18.0 ms steps, 72 ms, to advance one token each. Batched, they take 21.3 ms. The card did not get faster. It stopped reading the same five and a half gigabytes four times.

None of this is a new algorithm. The llama.cpp server has shipped `-np, --parallel N` server slots and continuous batching enabled by default for a long time, documented in its [server README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md). The gap in ZINC is not the technique, it is the default.

## Where it stops, and the argument I got wrong in May

Concurrency on a local card is bounded by VRAM, and the bound is tighter than the throughput curve makes it look.

Qwen3.5-9B costs about [147 KB per token](/blog/2026-07-13-attentions-two-matmuls-want-different-number-formats-on-rdna4/) of fp16 KV cache. ZINC sizes its cache at 0.85 of device memory, subtracts the weights, and divides the remainder by that per-token cost. Run the arithmetic on the two RDNA4 cards this blog cares about.

| Card | VRAM | KV budget after 5.5 GB of weights | fp16 KV tokens total | 8 agents each get | 16 agents each get |
| --- | ---: | ---: | ---: | ---: | ---: |
| RX 9070 XT | 16 GB | 8.1 GB | ~55,000 | ~6.9k context | ~3.4k context |
| Radeon AI PRO R9700 | 32 GB | 21.7 GB | ~148,000 | ~18k context | ~9.2k context |

The card has a fixed token budget and concurrency spends it. Batch one hands all 55,000 tokens to a single conversation. Eight agents get under 7k each, which is enough for a tool-calling subagent and nowhere near enough for one reading a large file. An [int8 KV cache](/blog/2026-05-19-fp8-kv-cache-is-the-next-decode-bandwidth-cut-rdna4-already-has-the-wmma-for/) halves the per-token cost to 74 KB and roughly doubles every number in that table, which makes it a concurrency feature as much as a bandwidth one.

Now the part I have to walk back. In May I argued that [paged KV cache is the serving fix a single-user local engine can mostly skip](/blog/2026-05-26-paged-kv-cache-is-the-serving-fix-a-single-user-local-engine-can-mostly-skip/), on the grounds that the fragmentation the [vLLM paper](https://arxiv.org/abs/2309.06180) measured, only 20.4 to 38.2 percent of allocated KV holding real tokens, is a property of many concurrent sequences of different lengths competing for one pool. That reasoning was sound. Its premise was that a local engine has one sequence in flight.

Agents break the premise cleanly. Five subagents arrive together, run for wildly different numbers of turns, finish at different times, and hold contexts ranging from 800 tokens to 30,000. That is variable-length sequences arriving and completing against a shared pool, which is the exact condition PagedAttention was built for, on a card with a quarter of the memory a server has. The conclusion follows the premise, and the premise moved.

## What I am building next

The concrete work is to promote ZINC's paged KV manager from the code path reserved for "the day zinc serves many requests" into the default, with a slot count set at startup and a scheduler that admits a new sequence into the in-flight batch between decode steps rather than queueing it behind a finished one. Then measure, because everything above this line is a model built on a measured 39.6 tok/s baseline, not a benchmark. The number I want on the table is aggregate tok/s at four and eight slots against the 55.6 the single-slot path gives today.

The larger lesson is about which default an engine defends. ZINC has spent four months making a single token cheaper on a card with 64 compute units that a batch of one leaves mostly idle. That was the right work, and it hit the point where the remaining wins are measured in single milliseconds. The next factor of five is not in the kernel. It is in noticing that the 5.5 GB the card reads every 18 milliseconds is the same 5.5 GB for everyone waiting on it.
