---
title: "A decode weight stream has no reuse, so RDNA4's Infinity Cache runs cold"
seoTitle: "RDNA4's 64 MB Infinity Cache Does Nothing for Local Decode Until an Agent Swarm Shares a Prompt"
date: "2026-07-29"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - infinity-cache
  - memory-bandwidth
  - kv-cache
  - decode
  - cache-hierarchy
  - agents
  - shared-prefix
  - local-llm
  - llm-inference
keywords:
  - RDNA4 Infinity Cache LLM inference
  - RX 9070 XT 64 MB cache decode
  - why Infinity Cache does not help LLM decode
  - last-level cache weight streaming no reuse
  - shared prefix KV cache Infinity Cache
  - RadixAttention on-die cache local GPU
  - effective bandwidth GDDR6 640 GB/s decode
  - agent swarm shared system prompt cache
excerpt: "The 640 GB/s we keep quoting for the RX 9070 XT is the raw GDDR6 number, and that is the honest one, because the card's 64 MB Infinity Cache does almost nothing for a decode token. A streaming weight read has no reuse, so a last-level cache 88 times smaller than the working set is dead weight. The one place it finally earns its keep is a batched agent swarm reading a shared prompt."
seoDescription: "RDNA4's Infinity Cache is a locality machine, and local LLM decode has almost no locality. On an RX 9070 XT, one Qwen3.5-9B decode step streams 5.5 GB of weights read exactly once, so a 64 MB on-die cache, 88x smaller than that working set, adds essentially zero effective bandwidth and decode runs at raw GDDR6 speed. Private KV is the same story. The single read pattern with real reuse is a shared system prompt read once per agent across a batched step, and even there the cache holds only about 445 tokens of KV before it spills. This post separates the reads a last-level cache can and cannot help, and argues the agent swarm is the first local workload that gives RDNA4's Infinity Cache anything to do."
faqs:
  - question: "Does RDNA4's Infinity Cache speed up local LLM decode?"
    answer: "Barely. A decode step on an RX 9070 XT streams the entire model out of VRAM once per token, about 5.5 GB for Qwen3.5-9B at Q4_K_M, and every weight byte is read exactly once. A cache only adds effective bandwidth when the same bytes are re-read before they are evicted, and a 64 MB cache is 88 times smaller than that 5.5 GB working set, so nearly every access is a compulsory miss. Decode weight reads run at the raw 640 GB/s of the GDDR6, which is why that is the number worth quoting."
  - question: "Why does streaming data get no benefit from a last-level cache?"
    answer: "Because a cache pays off on reuse, not on volume. The Infinity Cache holds recently read lines so a later read of the same address hits SRAM instead of DRAM. A weight stream touches each matrix once per token and never comes back to it within that token, and the next token re-reads all 5.5 GB long after the first bytes were evicted. With no temporal reuse and a working set far larger than the cache, the hit rate rounds to zero and effective bandwidth equals raw bandwidth."
  - question: "When does the Infinity Cache actually help a local inference engine?"
    answer: "When many reads hit the same bytes in a short window. In single-user local decode that pattern barely exists, but a batched agent swarm creates one: if eight agents share a system prompt, that prompt's KV cache is read once per agent inside a single batched attention step. The Infinity Cache collapses those eight reads into one GDDR fetch plus seven SRAM hits, for free, as long as the shared prefix fits in 64 MB."
  - question: "How long a shared prompt fits in the RX 9070 XT's 64 MB Infinity Cache?"
    answer: "About 445 tokens. Qwen3.5-9B costs roughly 147 KB per fp16 KV token across all layers, and 64 MB divided by 147 KB is about 445. A 400-token system prompt's KV is about 59 MB and stays resident, so a swarm sharing it reads it from GDDR once per step instead of once per agent. A 2,000-token shared prompt is about 294 MB, so it overflows the cache and most of it falls back to full GDDR traffic."
draft: false
---

The number this series keeps quoting for the RX 9070 XT is 640 GB/s, and it is worth pausing on why that is the honest figure rather than a conservative one. RDNA4's headline memory feature is not the GDDR6 bus. It is the [64 MB Infinity Cache](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html), a large on-die SRAM last-level cache whose entire job in a game is to catch re-reads and make the card behave as if it had far more bandwidth than its bus provides. During local LLM decode it catches almost nothing.

That is the claim I want to defend. A decode step is a streaming read with no reuse, and a cache that is 88 times smaller than the thing being streamed cannot change a streaming read. So the 640 GB/s is not the bus rate with a cautious asterisk; it is the rate decode actually sees, because the Infinity Cache sits idle through the part of the token that costs the most.

The twist, and the reason this is worth a post rather than a footnote, is that the agent swarm the last two weeks have been about is the first local workload that finally gives that cache something to do. Not the weights. A shared prompt.

## A cache pays off on reuse, and decode has none

A cache does exactly one thing: it holds recently touched bytes so a later read of the same address lands in fast SRAM instead of slow DRAM. That only helps if the same bytes are read more than once before they are evicted. Volume is irrelevant; reuse is everything.

Now look at what a decode step reads. To produce one token, ZINC streams the entire Qwen3.5-9B Q4_K_M model out of VRAM, about 5.5 GB, and that stream is [10.7 ms of a 25.2 ms token](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/), the single largest slice. Each matrix is read once, multiplied into the running activation, and never touched again inside that token. The next token re-reads all 5.5 GB from scratch, roughly 25 ms later, long after the first bytes have been pushed out of any 64 MB structure. There is no temporal reuse to catch.

Put the two facts next to each other and the cache's fate is settled. The working set is 5.5 GB, the cache is 64 MB, a ratio of about 88 to 1. Even if the access pattern were friendly, only a little over one percent of the stream could be resident at a time, and the pattern is not friendly, it is a single linear sweep. Every line is a compulsory miss. The Infinity Cache contributes no measurable effective bandwidth to the weight stream, and decode runs at the raw GDDR6 rate. That is the [640 GB/s](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html).

This is the memory-side mirror of a point made earlier in the series about the compute side. RDNA4's [matrix cores sit out the decode loop](/blog/2026-04-30-rdna4-matrix-cores-sit-out-the-decode-loop/) because decode has too little arithmetic to feed them; the Infinity Cache sits out the same loop because decode has too little reuse to fill it. Decode is thin in both currencies the card is built to spend.

## Private KV is the same story

The KV cache does not rescue the cache either, at least not for a single sequence. Each agent's attention reads back every key and value it has stored, and Qwen3.5-9B costs about [147 KB per fp16 KV token](/blog/2026-07-13-attentions-two-matmuls-want-different-number-formats-on-rdna4/) even after grouped-query attention has folded many query heads onto each key-value head. An agent holding a 6,900-token working set is reading roughly a gigabyte of KV every decode step.

That gigabyte is also read once per step. Within a token, each stored KV entry is visited a single time by attention and then dropped. So the private KV read has the same zero-reuse shape as the weight stream, and at 16 times the size of the cache it gets the same non-help. A single agent, no matter how it is tuned, hands the Infinity Cache a workload with nothing to cache.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-29-rdna4-infinity-cache-decode-reuse.svg" alt="A schematic on a near-black slate background titled 'A 64 MB Infinity Cache against a 5.5 GB decode read', subtitled 'RDNA4 (RX 9070 XT) memory path during Qwen3.5-9B decode: what the on-die cache can and cannot hold'. The left half, labelled 'the read path', shows three stacked tiers: a 'Compute units plus WMMA' box at top that consumes weights and KV then discards them, a cyan-outlined 'Infinity Cache, 64 MB SRAM' box in the middle that only pays off on bytes read more than once, and an amber-outlined 'GDDR6, 16 GB, 640 GB/s' box at bottom where every weight byte lives, read once per token. A thick amber arrow labelled 'weight stream 5.5 GB, no reuse, cache skipped' runs straight from GDDR through the cache tier to compute, bypassing it. A cyan arrow labelled 'shared-prefix KV' runs from GDDR into the cache and loops there, annotated 'eight agents, one GDDR read'. The right half, labelled 'working set versus the 64 MB line', has a dashed cyan vertical line marking the 64 MB cache and four horizontal bars: 'weight stream' at 5.5 GB, eighty-eight times the cache; 'private KV, one agent at 6,900 tokens' at 1.0 GB, sixteen times; 'shared prompt KV, 400 tokens' at 59 MB, which fits; and 'shared prompt KV, 2,000 tokens' at 294 MB, which spills with only about 445 tokens staying hot. A callout box reads '64 MB divided by 147 KB per KV token is about 445 tokens, the longest shared prefix the cache can keep resident'." loading="lazy" />
  <figcaption>Modeled for Qwen3.5-9B Q4_K_M on one RX 9070 XT: a 5.5 GB weight stream per decode step, 147 KB per fp16 KV token, 640 GB/s GDDR6, and a 64 MB Infinity Cache. Streaming reads with no reuse gain nothing from a last-level cache; only the shared-prefix KV, read once per agent in a batched step, has the reuse a 64 MB cache can capture, and only while it stays under 64 MB.</figcaption>
</figure>

Read the left side as a decision about which arrow touches the middle tier. The weight stream, the biggest arrow, drives straight past the Infinity Cache because none of its bytes will be asked for twice. The private KV does the same. The only arrow that loops through the cache is the shared-prefix KV, and the right side explains why: it is the only read whose bar sits near or below the 64 MB line, which is the same as saying it is the only read a 64 MB cache can hold onto.

## The swarm creates the one reuse pattern that exists

Here is where the last two weeks pay off. When you [run eight agents on one card](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) and they share a system prompt, that prompt's KV is no longer read once per step. It is read once per agent inside the same batched attention step, because each agent's attention has to attend over the shared prefix as part of its own context. Eight agents, eight reads of the same physical bytes, within a window of a few milliseconds.

That is reuse, and it is exactly what a last-level cache is for. The first agent's attention pulls the shared prefix out of GDDR and into the Infinity Cache; the other seven find it already resident and hit SRAM. The eight GDDR reads collapse to one, and they collapse whether or not the engine explicitly deduplicates the prefix in the kernel. The [prefix sharing that RadixAttention formalizes](https://arxiv.org/abs/2312.07104) as a software structure, the Infinity Cache delivers for free at the hardware level, as long as the shared bytes fit.

The catch is that qualifier. A 64 MB cache holds only so much KV, and the arithmetic is unforgiving: 64 MB divided by 147 KB per token is about 445 tokens. That is the longest shared prefix the cache can keep resident across a batched step. Below it the swarm gets the collapse; above it the prefix spills, and most of it is back to being read once per agent from GDDR.

| Read in a decode step | Bytes | Reads per step | Working set vs 64 MB | Infinity Cache payoff |
| --- | ---: | ---: | ---: | --- |
| Weight stream, Q4_K_M | 5.5 GB | 1 | 88x | none, runs at raw 640 GB/s |
| Private KV, one agent at 6,900 tokens | 1.0 GB | 1 | 16x | none |
| Shared prompt KV, 8 agents, 400 tokens | 59 MB | 8 | 0.9x, fits | up to ~8x fewer GDDR reads on the prefix |
| Shared prompt KV, 8 agents, 2,000 tokens | 294 MB | 8 | 4.6x, spills | only the first ~445 tokens stay hot |

The table is the whole argument in four rows. The two reads that dominate a decode token, the weights and the private KV, are far larger than the cache and read only once, so the cache does nothing for them. The shared-prefix read is the only one with both a small enough footprint and enough reuse to matter, and even it earns its keep only while the prompt stays under a few hundred tokens. A 400-token system prompt is comfortably inside the line; a 2,000-token one, which is an ordinary agent preamble, is nearly five times too big and mostly spills.

## What this changes about tuning a local swarm

The first consequence is to stop expecting the Infinity Cache to show up in a decode benchmark. It will not. A single-stream token generator on this card is a GDDR6 machine with 64 MB of SRAM going almost entirely unused, and any decode tuning that assumes the cache is quietly buying effective bandwidth is tuning against a number that is not there. The [memory-bound regime that Pope and colleagues](https://arxiv.org/abs/2211.05102) treat as the master variable for decode is set by the bus, full stop.

The second consequence is more useful. If the shared-prefix KV is the only read the cache can help, then keeping that prefix short and hot is a real lever, and it is a different lever from the one the [system-prompt post](/blog/2026-07-21-the-system-prompt-a-local-agent-swarm-caches-eight-times-over/) pulled. That post was about capacity: storing the shared prompt once in VRAM instead of eight times so it does not [crowd out the private KV](/blog/2026-07-28-batched-swarm-kv-read-does-not-amortize/). This is about traffic: a shared prompt small enough to live in 64 MB is read from GDDR once per step across the whole batch, not once per agent. The two effects stack. Deduplicating the prefix saves the space, and fitting it under the cache line saves the bandwidth, and a swarm that keeps its common preamble under a few hundred tokens gets both.

There is a design tension worth naming, because it points somewhere. Quantizing the KV cache to fp8, the [decode-bandwidth cut RDNA4 already has the WMMA for](/blog/2026-05-19-fp8-kv-cache-is-the-next-decode-bandwidth-cut-rdna4-already-has-the-wmma-for/), halves the 147 KB per token, which roughly doubles how much shared prefix fits in the cache, from about 445 tokens to about 890. So the same feature that shrinks private KV traffic also widens the window where the Infinity Cache captures shared-prefix reuse. That is a rare case of one change paying off in two ledgers at once, and it is the sort of thing the [PagedAttention line of work](https://arxiv.org/abs/2309.06180) keeps surfacing: the KV cache, not the weights, is where a serving engine's memory decisions actually bite.

The framing I want to leave is that RDNA4's biggest cache is a locality machine, and local decode is a workload almost without locality. For a single user generating tokens, the 64 MB of Infinity Cache is 53.9 billion transistors' worth of SRAM doing very little, and the honest bandwidth number is the raw bus. The agent swarm is the exception, and a narrow one: it manufactures reuse out of a shared prompt, and the cache pays it back exactly up to 64 MB and not a byte further. Knowing that line exists, and keeping the shared context under it, is the difference between a cache that quietly helps every step and one that watches the bus do all the work.
