---
title: "A Qwen3.5-9B chat turn spends 96 percent of its wall clock in decode"
seoTitle: "Prefill vs Decode Wall Clock: Qwen3.5-9B Chat on RDNA4"
date: "2026-07-14"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - prefill
  - decode
  - kv-cache
  - memory-bandwidth
  - amdahls-law
  - local-llm
  - llm-inference
  - roofline
keywords:
  - prefill vs decode wall clock
  - Qwen3.5-9B chat latency RDNA4
  - decode memory bandwidth bound local LLM
  - RX 9070 XT 640 GB/s decode ceiling
  - Amdahl's law LLM inference
  - time to first token vs tokens per second
  - KV cache bandwidth decode
  - local LLM chat vs RAG workload
  - model bandwidth utilization RDNA4
  - prefill diminishing returns
excerpt: "The last six posts took Qwen3.5-9B prefill on the RX 9070 XT from 219 to 962 tok/s, a 4.4x win that closed the llama.cpp gap. On a normal 2k-in, 2k-out chat turn it shaved the wall clock by 13 percent, and the last post in the series was worth 0.7 percent. Prefill is now 4 percent of a chat turn. Decode is the other 96, and decode is a bandwidth problem the matrix core cannot touch."
seoDescription: "A 2k-in, 2k-out chat turn on Qwen3.5-9B and one RX 9070 XT spends 2.1 seconds in prefill and 50.5 in decode, so prefill is 4 percent of wall clock and decode is 96. Prefill only dominates when the prompt is much longer than the generation, as in RAG and document summary. Decode is bound by the 640 GB/s it takes to stream 5.5 GB of Q4_K_M weights per token, which caps a perfect implementation near 116 tok/s and makes the next order of magnitude a bytes-per-token problem, not a FLOP problem."
faqs:
  - question: "For a normal chat, does prefill speed or decode speed matter more?"
    answer: "Decode, by a wide margin. On Qwen3.5-9B and one RX 9070 XT, a 2k-token prompt prefills in 2.1 seconds at 962 tok/s, and generating 2k tokens takes 50.5 seconds at 39.6 tok/s. That is 4 percent of the wall clock in prefill and 96 percent in decode. Prefill only takes over when the prompt is much longer than the reply, which is the RAG and document-summary shape, not chat."
  - question: "Why did a 4.4x prefill speedup only make a chat turn 13 percent faster?"
    answer: "Amdahl's law. Prefill was about 15 percent of a 2k-in, 2k-out turn before the work started, so even reducing it to zero could not have removed more than 15 percent of the wall clock. The first fix, 219 to 430 tok/s, removed 4.48 seconds and mattered. The last fix, 815 to 962 tok/s, removed 0.37 seconds, because by then prefill was already down to 4 percent of the turn."
  - question: "What limits decode speed on the RX 9070 XT?"
    answer: "Memory bandwidth. At batch one, decoding a token reads every weight in the model once, and the RX 9070 XT moves 640 GB/s across its 256-bit GDDR6 bus. Streaming 5.5 GB of Q4_K_M weights takes about 8.6 ms, which caps a perfect implementation near 116 tok/s regardless of how fast the matrix core is. The matrix core is idle-adjacent during decode because a matrix-vector product has no reuse to feed it."
  - question: "How do you actually make local decode faster then?"
    answer: "Read fewer bytes per token. Smaller quantization lowers the weight bytes directly, an IQ3 build near 4.3 GB lifts the ceiling to about 149 tok/s. Speculative decoding amortizes one weight read across several accepted tokens. A mixture-of-experts model reads only the active experts. Every real decode lever is a bytes-per-token lever, which is why prefill, a compute problem, was the easier win to chase first."
draft: false
---

The last six posts on this blog were about one number. Qwen3.5-9B prefill on a single RX 9070 XT started the month at 219 tok/s and ended it at 962, which is a [4.4x speedup](https://zolotukhin.ai/blog/2026-07-13-attentions-two-matmuls-want-different-number-formats-on-rdna4/) and close enough to llama.cpp's 973 tok/s reference on the same card that I stopped calling it a gap. Six posts of dequant round-trips, VGPR pressure, softmax issue slots, and int8 WMMA, all pointed at prefill.

Here is the part I did not say out loud until now. On a normal chat turn, that entire month of work is worth about 13 percent of the wall clock, and the final post in the series was worth less than one.

That is not a complaint about the work. It is Amdahl's law doing exactly what it does, and it points at where the next order of magnitude actually lives.

## Prefill and decode are not the same size

A language model answers in two phases. Prefill reads the whole prompt at once and builds the KV cache, and because every token is available up front, it is a batched matrix-matrix problem that saturates the matrix core. Decode then generates the reply one token at a time, and each step is a matrix-vector product that reads the entire model to produce a single token.

The two phases run at wildly different speeds for the same reason. Prefill has arithmetic intensity, so it is compute-bound and rewards a faster matrix core. Decode has almost none, so it is bound by how fast weights stream out of VRAM, and the matrix core mostly waits. On Qwen3.5-9B and one RX 9070 XT, ZINC prefills at 962 tok/s and decodes at 39.6. That is a 24x gap between the two phases, and it decides everything about which optimizations pay.

A chat turn is mostly the slow phase. Take a fairly ordinary turn, a 2,000-token prompt and a 2,000-token reply. Prefill costs 2000 / 962, about 2.1 seconds. Decode costs 2000 / 39.6, about 50.5 seconds. Prefill is 4 percent of the wall clock and decode is the other 96.

## The shape of the prompt decides who wins

The 96 percent figure is not universal. It flips hard when the prompt is much longer than the reply, which is exactly the retrieval and summarization shape, and this is the honest case for having done the prefill work at all.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-14-chat-turn-wall-clock-prefill-vs-decode.svg" alt="A two-panel diagram on a dark background. The top panel is five horizontal stacked bars, one per workload, each normalized to 100 percent of that request's wall clock, prefill in amber and decode in blue, with the absolute total in seconds at the right. Short chat, 256 in and 512 out, is 2 percent prefill and 98 percent decode, total 13.2 seconds. Chat turn, 2k in and 2k out, is 4 percent prefill and 96 percent decode, total 52.6 seconds. Code review, 16k in and 1k out, is 40 percent prefill and 60 percent decode, total 41.9 seconds. RAG answer, 32k in and 512 out, is 72 percent prefill and 28 percent decode, total 46.2 seconds. Doc summary, 32k in and 256 out, is 84 percent prefill and 16 percent decode, total 39.8 seconds. The bottom panel is four descending amber bars showing the seconds each post's prefill speedup removed from one 2k-in, 2k-out chat turn: the 219 to 430 tok/s step saved 4.48 seconds, the 430 to 615 step saved 1.40, the 615 to 815 step saved 0.80, and the 815 to 962 step saved only 0.37, because decode stayed at 50.5 seconds the entire time." loading="lazy" />
  <figcaption>Top: wall-clock split by workload shape on Qwen3.5-9B, one RX 9070 XT. Bottom: seconds each prefill speedup removed from a 2k-in, 2k-out chat turn. All decode at 39.6 tok/s, prefill at the stated rate.</figcaption>
</figure>

Read the top panel as a spectrum. When the reply is longer than the prompt, decode swamps everything and prefill is a rounding error. When the prompt is longer than the reply, prefill takes over: a 32k-token document summary that emits 256 tokens spends 84 percent of its time in prefill, and every one of those six posts lands directly on that 84 percent.

The table underneath makes the crossover concrete.

| Workload | Prompt / reply | Prefill | Decode | Total | Prefill share |
| --- | --- | ---: | ---: | ---: | ---: |
| Short chat | 256 / 512 | 0.27 s | 12.9 s | 13.2 s | 2% |
| Chat turn | 2k / 2k | 2.08 s | 50.5 s | 52.6 s | 4% |
| Code review | 16k / 1k | 16.6 s | 25.3 s | 41.9 s | 40% |
| RAG answer | 32k / 512 | 33.3 s | 12.9 s | 46.2 s | 72% |
| Doc summary | 32k / 256 | 33.3 s | 6.5 s | 39.8 s | 84% |

The crossover sits where the prompt is roughly four to five times the reply. Below it you are decode-bound and prefill speed is nearly free money you already spent. Above it you are prefill-bound and time to first token is the number a user feels. Chat lives well below the line, which is why the month of prefill work barely registers on it.

## Amdahl's law was collecting the whole time

The bottom panel of the figure is the part that stung. It plots how many seconds each post's prefill speedup actually removed from one 2k-in, 2k-out chat turn, and the bars fall off a cliff.

The first fix, [219 to 430 tok/s](https://zolotukhin.ai/blog/2026-07-08-the-dequant-scratch-round-trip-is-zincs-last-rdna4-prefill-tax/), cut prefill from 9.13 to 4.65 seconds and removed 4.48 seconds from the turn. That is real. The last fix, [815 to 962 tok/s](https://zolotukhin.ai/blog/2026-07-13-attentions-two-matmuls-want-different-number-formats-on-rdna4/), cut prefill from 2.45 to 2.08 seconds and removed 0.37. Same engineering effort, arguably harder, worth an eighth as much to a chat user.

This is [Amdahl's law](https://en.wikipedia.org/wiki/Amdahl%27s_law) stated as a stopwatch. The overall speedup from improving one part of a system is capped by the fraction of time that part was using. Prefill was about 15 percent of a 2k-in, 2k-out turn when the series began, so driving it to literal zero could never have made the turn more than 15 percent faster. The 4.4x we did get bought 13 percent, and every additional tok/s bought less than the one before it, because each speedup shrank prefill's own share of the wall clock and starved the next optimization of anything to remove.

None of this means the work was wrong. It means the work was right for one workload, the long-prompt one, and I let a chat-shaped intuition quietly ride along on numbers that only ever described document summarization.

## Decode is a bandwidth wall, and the matrix core is on the wrong side of it

If chat is 96 percent decode, then the interesting question is why decode is stuck at 39.6 tok/s, and the answer is not the matrix core.

At batch one, generating a token reads every weight in the model exactly once. Qwen3.5-9B in Q4_K_M is about 5.5 GB. The RX 9070 XT moves 640 GB/s across a 256-bit GDDR6 bus at 20 Gbps, a figure worth [taking from a review](https://www.tomshardware.com/pc-components/gpus/amd-radeon-rx-9070-xt-review) rather than a spec-sheet peak. Stream 5.5 GB at 640 GB/s and you spend 8.6 ms just moving the weights, which caps a flawless implementation at roughly 116 tok/s before it does any arithmetic at all. The matmuls are matrix-vector products with no reuse, so the matrix core has nothing to chew on while it waits for the bus. Every prefill trick from the last month doubled a rate that decode was never using.

ZINC's 39.6 tok/s sits at about a third of that 116 ceiling, so there is headroom in the implementation, but the ceiling itself is fixed by bytes per token, not by FLOPs. This is the same wall [llama.cpp](https://github.com/ggml-org/llama.cpp) hits on the same card, because it is a property of the hardware and the model, not of either engine. You cannot compute your way through it. You can only read fewer bytes.

That reframes the entire next arc. Every decode lever is a bytes-per-token lever. Smaller quantization is the direct one, an [IQ3_M build](https://zolotukhin.ai/blog/2026-05-29-the-imatrix-pass-that-decides-which-qwen3-weights-survive-iq3-m/) near 4.3 GB lifts the ceiling to about 149 tok/s, paid for in perplexity. The int8 KV cache from last week reads fewer KV bytes at long context and already moved 32k decode from 21 to 27 tok/s, which lands where the [KV-versus-weights crossover](https://zolotukhin.ai/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode/) predicted. Speculative decoding is the sideways one, amortizing a single weight read across several accepted draft tokens, though [it has not netted out](https://zolotukhin.ai/blog/2026-05-25-speculative-decoding-on-qwen3-a3b-loses-even-at-100-percent-draft-acceptance/) on a dense 9B for us yet. A mixture-of-experts model is the structural one, reading only the active experts per token.

## What I am changing

The concrete change is to benchmark two numbers from now on and stop quoting one. Prefill throughput describes the RAG and summarization workloads honestly, and I will keep reporting it because those workloads are real and the long-prompt crossover is where prefill genuinely rules. But time to first token and tokens per second are different levers pulling on different phases, and a single "it got faster" hides which one moved.

The lesson is older than this stack and I should have trusted it sooner. Optimize the phase the wall clock is actually sitting in. For a chat turn on a local card, that phase is decode, decode is bound by the 640 GB/s it takes to stream the weights, and the only way through a bandwidth wall is to carry less across it. The matrix core is not invited. The next posts are going to be about bytes.
