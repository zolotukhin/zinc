---
title: "Batching an agent swarm shares the weights but not the KV cache on RDNA4"
seoTitle: "Why a Batched Local Agent Swarm's Speedup Erodes With Context on RDNA4"
date: "2026-07-28"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - agents
  - batching
  - continuous-batching
  - kv-cache
  - decode
  - memory-bandwidth
  - gqa
  - local-llm
  - llm-inference
keywords:
  - batched decode KV cache bandwidth RDNA4
  - agent swarm batching speedup context length
  - shared weight stream vs per request KV cache
  - RX 9070 XT decode memory bound batching
  - why batching speedup falls with long context
  - Qwen3.5-9B KV cache read per token
  - continuous batching diminishing returns local LLM
  - GQA KV cache batched decode bandwidth
excerpt: "Batching eight agents on one RX 9070 XT is only a 5.6x decode win at short context, and that number quietly falls as the agents fill their KV caches. The batch shares one weight stream but every agent still reads its own KV, so past a few thousand tokens the private KV traffic, not the shared weights, sets the decode rate."
seoDescription: "A batched decode step on an RX 9070 XT reads the model weights once for the whole batch but reads each agent's KV cache separately, so the two halves of the memory bill scale differently. Modeling Qwen3.5-9B on one 640 GB/s card: the 5.6x swarm speedup measured at a short 2,400-token context falls to about 3.8x at the 6,900 tokens these agents actually hold, and toward 1.6x as they approach the card's 55,000-token KV budget. This post derives the crossover where private KV reads overtake the shared weight stream, shows why continuous batching has a context-dependent ceiling the throughput posts never named, and argues KV-cache traffic, not the weight stream, is the number a long-running local swarm should design against."
faqs:
  - question: "Does batching multiple agents on one GPU keep its speedup as context grows?"
    answer: "No. Batching amortizes the model weight stream, which is read once per decode step and shared by every agent in the batch, but it does not amortize the KV cache, which each agent reads separately. On one RX 9070 XT running Qwen3.5-9B, the eight-agent decode speedup is about 5.6x when each agent holds a short 2,400-token context, falls to roughly 3.8x at 6,900 tokens, and drops toward 1.6x near the card's 55,000-token KV budget, because the private KV traffic grows with both batch size and context while the shared weight traffic does not."
  - question: "Why does the KV cache read not amortize across a batch the way weights do?"
    answer: "Because the model weights are identical for every sequence, so a batched decode step reads them once and reuses them for all agents, but each agent has its own distinct KV cache that must be read in full to compute its attention. Adding an agent to the batch adds zero weight traffic and a full agent's worth of KV traffic, so as context lengthens the per-token cost becomes dominated by the part batching cannot share."
  - question: "At what context length does the KV read overtake the weight read on an RX 9070 XT?"
    answer: "For Qwen3.5-9B at Q4_K_M with a 5.5 GB weight stream and about 147 KB per fp16 KV token, an eight-agent batch crosses over near 4,700 tokens of context per agent. Below that the shared weight stream is the larger per-token cost and batching pays well; above it the private KV read is larger and each added agent mostly buys itself bandwidth rather than sharing someone else's."
  - question: "How should a local inference engine schedule a long-running agent swarm?"
    answer: "Size the batch against KV-cache traffic, not just against the weight stream. Early agents ride a weight stream that is already paid for, so they are nearly free; once each agent holds thousands of tokens of context, an added agent contributes a full KV read and the marginal throughput per agent collapses. Quantizing the KV cache to fp8 or int8, or holding fewer long histories resident, moves the crossover out and is often worth more than widening the batch."
draft: false
---

Three of the last posts in this series leaned on the same comforting fact: a decode step on an RX 9070 XT streams the whole model once, so if you [run eight agents through that one stream](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) you get 5.6x the tokens for almost free. Yesterday's energy post made the same move, [amortizing the weight stream](/blog/2026-07-27-batching-eight-agents-saves-more-decode-time-than-energy-on-rdna4/) across the batch. Every one of those numbers was measured with the agents holding short contexts.

That is the quiet assumption, and it does not hold. A decode step has two memory costs, not one. The weights are shared by the whole batch, but every agent reads its own KV cache, and nothing about batching makes that read cheaper. As the agents fill their contexts, the part batching cannot share takes over, and the headline 5.6x erodes on its own.

The number worth keeping from this post is smaller and more honest than 5.6x. At the 6,900-token contexts these agents actually [sat on two posts ago](/blog/2026-07-20-swapping-an-idle-agents-kv-cache-beats-recomputing-it-by-177x/), the real eight-agent speedup is closer to 3.8x, and it keeps falling from there.

## The two halves of a decode-step memory bill

A decode step on this card is memory-bound, so its time is set by how many bytes it moves, not how many multiplies it does. Splitting those bytes in two makes the whole argument visible.

The first half is the weight stream. To produce a token, ZINC reads the entire Qwen3.5-9B Q4_K_M model out of VRAM, about 5.5 GB, which is [10.7 ms of a 25.2 ms token](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/). Those weights are the same matrices for every sequence in flight, so a batched step reads them once and feeds all agents. This is the sharing that batching is built on, and it is real: the expensive part is paid once per step, not once per agent.

The second half is the KV cache. Each agent's attention has to read back every key and value it has stored for its own context, and those tensors are private. Qwen3.5-9B costs about [147 KB per fp16 KV token](/blog/2026-07-13-attentions-two-matmuls-want-different-number-formats-on-rdna4/) even after grouped-query attention, the [KV-head sharing scheme](https://arxiv.org/abs/2305.13245) that already cut this number by folding many query heads onto each key/value head. An agent holding 6,900 tokens is reading back about a gigabyte of KV every single decode step, and the agent next to it in the batch reads back its own, separate gigabyte.

Put the two together. For a batch of B agents each holding L tokens, one decode step moves 5.5 GB of shared weights plus B times L times 147 KB of private KV. The step produces B tokens, so per generated token it moves 5.5 GB over B, which shrinks as you add agents, plus L times 147 KB, which does not shrink at all. One term amortizes. The other is fixed per agent and grows with context.

## Where the private read overtakes the shared one

The crossover is the context length where those two per-token terms are equal, and past it the KV read is the bigger number.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-28-batched-swarm-kv-read-does-not-amortize.svg" alt="A line chart on a deep indigo background titled 'Batching shares the weights, not the KV cache'. The horizontal axis is KV context held per agent, from zero to fifty-five thousand tokens. The vertical axis is the effective eight-agent decode speedup, from one to eight times. An amber curve starts at eight times at zero context, where a dashed reference line marks the ideal eight-times ceiling if the KV cache were shared too, and decays steeply: it passes a marked point at 5.6 times near 2,400 tokens labelled 'context at the measured 5.6x turn', a point at 3.8 times near 6,900 tokens labelled '6,900 tokens: the real agent context', and ends at a rose point at 1.6 times at the card's fifty-five-thousand-token KV limit. A dashed vertical line at about 4,700 tokens marks the crossover, and the region to its right is lightly shaded and labelled 'KV read greater than weight read per token, batching mostly stops paying off'. An inset on the right shows three stacked bars of per-token decode bytes for eight agents at 2.4k, 6.9k and 32k tokens of context, each split into an amber 'shared weight divided by eight' segment that stays the same height and a cyan 'private KV per agent' segment that grows until it dominates the tallest bar. A footnote reads: Model 5.5 GB weight stream per step, 147 KB per fp16 KV token, 640 GB/s, speedup equals B times open paren W plus L times k close paren over open paren W plus B times L times k close paren, memory-bound decode." loading="lazy" />
  <figcaption>Modeled for Qwen3.5-9B on one RX 9070 XT: 5.5 GB shared weight stream per decode step, 147 KB per fp16 KV token, 640 GB/s. The eight-agent speedup is B·(W+L·k)/(W+B·L·k), which is 8x only at zero context and decays as each agent's private KV read grows. The crossover near 4,700 tokens is where per-token KV traffic equals the shared per-token weight traffic.</figcaption>
</figure>

Read the inset first. The amber segment, the shared weight stream divided across eight agents, is the same height in every bar, because widening context does nothing to it. The cyan segment, the private KV read, grows with context until at 32k tokens it is most of the bar. The main curve is what that does to throughput: the eight-agent speedup is the full 8x only in the limit of zero context, and it decays from there because the denominator picks up a per-agent KV term the numerator never gets to share.

Plugging in the model's own numbers, the weight stream is 5.5 GB and each KV token is 147 KB, so the eight-agent crossover lands at 5.5e9 divided by (8 times 147e3), about 4,700 tokens. Below that, batching is doing what the throughput posts advertised. Above it, most of what each new agent adds to the batch is its own KV read, and the swarm is closer to eight sequences taking turns than to eight sequences sharing a stream.

| Context per agent | 8-agent speedup | What is setting the pace |
| --- | ---: | --- |
| 0 (ideal limit) | 8.0x | pure weight stream, perfectly shared |
| 2,400 (the measured 5.6x turn) | 5.6x | weights still dominate the per-token cost |
| 4,700 (crossover) | 4.5x | KV read equals weight read |
| 6,900 (real agent context) | 3.8x | private KV is now the larger half |
| 16,000 | 2.6x | KV read is roughly triple the shared weights |
| 55,000 (card KV budget) | 1.6x | almost pure per-agent KV, batch barely helps |

The table is the post in one glance. The 5.6x that anchored three earlier posts corresponds to about 2,400 tokens of context, which is a fresh agent a few turns in. By the time each agent is carrying the 6,900-token working set of a real coding session, the same eight agents are a 3.8x win, and if you let them grow toward the 55,000 tokens the [16 GB card](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html) can hold across the swarm, batching is worth about 1.6x. None of that is an RDNA4 defect. It is the arithmetic of sharing one term and not the other, and it is why the [PagedAttention work](https://arxiv.org/abs/2309.06180) frames KV memory, not model weights, as the thing that limits how many sequences you can actually batch.

## This is the mirror image of the prefill story

There is a satisfying symmetry with prefill worth naming, because it explains why the ceiling exists at all. Prefill is compute-bound and batches beautifully: many tokens share the weight read and keep the matrix units busy, so the model FLOPS utilization that [Pope and colleagues](https://arxiv.org/abs/2211.05102) treat as the master variable is high. Decode is the opposite regime, memory-bound and thin, and batching is an attempt to import prefill's trick by making many sequences share one weight read.

The trick works, but only for the weight read. Prefill shares weights across tokens that all belong to the same sequence and carry no standing KV to re-read. A decode batch shares weights across sequences that each drag a full, private KV history behind them. So batched prefill scales with the batch almost cleanly, while batched decode scales with the batch only until the private KV term catches the shared weight term. The card's [640 GB/s of memory bandwidth](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html) is spent on whichever half is larger, and past the crossover that half is the one batching cannot touch.

## What a long-running swarm should actually tune

The practical takeaway is that batch width is the wrong single knob for a swarm whose agents live a long time. Widening the batch helps a lot when everyone is short and helps almost nothing when everyone is long, and the same eight-agent configuration can be a 5.6x machine in the morning and a 3.8x machine by lunch as the contexts fill. A scheduler that reports throughput without reporting the batch's average context length is hiding the variable that actually moves the number.

The lever that does keep working is shrinking the KV read itself. Every byte cut from the per-token KV cost pushes the crossover to the right and buys back speedup that batching alone cannot. Quantizing the cache to fp8 halves the 147 KB and roughly doubles the crossover context, which is the [decode-bandwidth cut RDNA4 already has the WMMA for](/blog/2026-05-19-fp8-kv-cache-is-the-next-decode-bandwidth-cut-rdna4-already-has-the-wmma-for/); evicting or swapping idle histories keeps fewer long caches resident so the live batch stays on the cheap side of the curve. ZINC's scheduler already tracks live batch size to size speculation, and the same counter multiplied by each agent's context length is exactly the KV-traffic estimate this curve needs.

The framing I want to leave is that batching does one favor, not two. It shares the weight stream, which is a large and honest win at short context, and it charges every agent full price for its own KV cache, which is the bill that grows while nobody is watching. Time and bandwidth are not the same ledger, and the swarm that looks 5.6x faster on a benchmark is a 3.8x machine on real work and a 1.6x one at the limit. Knowing which point on that curve your agents are sitting at is the difference between tuning for the demo and tuning for the day.
