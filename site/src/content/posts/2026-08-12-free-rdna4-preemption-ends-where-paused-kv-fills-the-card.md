---
title: "Free RDNA4 preemption ends where paused KV caches fill the 16 GB card"
seoTitle: "The VRAM Ceiling on Free Token-Boundary Preemption in an RDNA4 Agent Swarm"
date: "2026-08-12"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - kv-cache
  - preemption
  - scheduling
  - vram
  - gqa
  - quantization
  - agents
  - local-llm
  - llm-inference
keywords:
  - paused KV cache VRAM ceiling RDNA4
  - free preemption memory limit local LLM
  - token-boundary preemption resident KV
  - KV cache size per token GQA Qwen3
  - q8 KV cache doubles resident paused turns
  - RX 9070 XT 16 GB agent swarm memory budget
  - vLLM swap recompute preemption
  - single card local inference KV budget
excerpt: "Yesterday's post argued that preempting a decode slot on an RX 9070 XT is nearly free because the paused turn's KV cache never leaves VRAM. That is true, and it hides a ceiling. A paused turn keeps every byte of its KV resident, so under sustained oversubscription the paused turns stack up in the same 16 GB the running turns are using. The arithmetic is blunt: eight running turns at a 6k-token average already sit at 13.5 GiB, which leaves room for about two average paused turns before the card is full and you are forced back into the swap or recompute you were avoiding. And the turns you actually preempt are the long ones, so a single paused 16k-token refactor can overflow the budget on its own."
seoDescription: "Token-boundary preemption on a single 16 GB RX 9070 XT is cheap because a paused decode turn's KV cache stays resident in VRAM, needing no swap and no recompute to resume. But resident is not free: every paused turn holds its full KV, and those bytes come out of the same 16 GB as the running turns. For Qwen3.5-9B with 40 layers, 8 GQA KV heads and head dim 128, fp16 KV is about 160 KiB per token, so a 6k-token turn is 0.94 GiB. Eight running turns plus weights and runtime reach 13.5 GiB, leaving roughly 2 GiB, about two average paused turns, before the swarm overruns the card and preemption falls back to vLLM-style swap or recompute. Because the turns worth preempting are the long ones, a single paused 16k-token turn at 2.56 GiB can overflow on its own. Quantizing the KV cache to q8 halves every term and roughly doubles the number of paused turns that fit resident."
faqs:
  - question: "Why is token-boundary preemption described as free on an RX 9070 XT?"
    answer: "Because on a single-card local setup the paused turn's KV cache is already in VRAM and never has to move. When you preempt a running turn at a token boundary you simply stop scheduling it; its keys and values stay resident exactly where they were, so resuming it means putting it back in the next batch and continuing from the token it stopped on. There is no copy to host memory and no prefill recompute, which is what makes preemption expensive on serving stacks that swap or recompute the evicted KV. The cost of the eviction itself is at most one decode step, on the order of 25 to 30 milliseconds."
  - question: "So what is the ceiling on that free preemption?"
    answer: "VRAM. Resident is not the same as free. A paused turn keeps every byte of its KV cache, and those bytes share the same 16 GB as the running turns. On an RX 9070 XT running Qwen3.5-9B, model weights and runtime buffers take about 6 GiB, and eight running turns at a 6k-token average context add 7.5 GiB of KV, reaching 13.5 GiB. That leaves roughly 2 GiB of usable headroom, which is about two average paused turns of 0.94 GiB each. Beyond that the card is full and the scheduler has to evict a paused turn's KV to host memory or drop it and recompute later, which is the exact swap or recompute cost that token-boundary preemption was supposed to avoid."
  - question: "How big is the KV cache per token for a 9B model like this?"
    answer: "It depends on the attention shape, not the parameter count. For a configuration with 40 transformer layers, 8 grouped-query-attention key-value heads and a head dimension of 128, each token stores 2 tensors (keys and values) times 8 heads times 128 dimensions times 40 layers, which is 81,920 elements per token. In fp16 that is 163,840 bytes, about 160 KiB per token. A 6k-token turn is therefore about 0.94 GiB, and a 16k-token turn is about 2.56 GiB. Grouped-query attention is what keeps these numbers manageable: with 8 KV heads instead of one per query head, the cache is a fraction of what full multi-head attention would need."
  - question: "Does quantizing the KV cache remove the ceiling?"
    answer: "It moves the ceiling, it does not remove it. Storing keys and values in q8 instead of fp16 halves every KV term, so a 6k-token turn drops from 0.94 GiB to about 0.47 GiB. Eight running turns then take 3.75 GiB instead of 7.5 GiB, and because both the running set and each paused turn shrink, the number of average paused turns that fit climbs from about two to around a dozen. That is the same lever an earlier post measured when q8 pushed the swarm's fit crossover from 5k to 10k tokens of context. It buys headroom, but a long enough paused turn or a deep enough queue still reaches the wall, just later."
draft: false
---

[Yesterday's post](/blog/2026-08-11-a-long-background-turn-stalls-the-foreground-agent-on-rdna4/) made a clean argument. When an interactive foreground turn is stuck behind a long background turn on one RX 9070 XT, you fix the stall by preempting the background turn at a token boundary, and on RDNA4 that preemption is nearly free because the paused turn's KV cache never leaves the 16 GB of VRAM. No swap to host memory, no prefill recompute. Just stop scheduling the turn and pick it back up later from where it left off.

That is correct, and it quietly assumes something that is not always true. It assumes there is room to leave the paused turn resident. Resident is not the same as free. A paused turn keeps every byte of its key-value cache in VRAM, and those bytes come out of the same 16 GB the running turns are already using. Stop enough turns and the resident cache that made preemption cheap is the thing that runs the card out of memory.

This post does the arithmetic that yesterday's skipped. The short version is that free preemption has a ceiling of about two average paused turns, and the turns you actually want to preempt are exactly the ones that blow through it.

## The KV cache is the real budget line

Start with where the 16 GB goes. The running example for this whole series is Qwen3.5-9B on a single [RX 9070 XT](/blog/2026-08-08-littles-law-caps-a-responsive-rdna4-agent-swarm-near-twelve/), quantized to Q4_K_M, which puts the weights near 5.2 GiB. Runtime buffers, the compute scratch and the activation working set, take roughly another 0.8 GiB. Call it 6 GiB of fixed cost before a single token is cached. On a 16 GB card the driver keeps a little back, so usable VRAM is closer to 15.5 GiB. That leaves about 9.5 GiB for the KV cache, and the KV cache is where all the pressure lives.

The size of a KV cache per token is set by the attention shape, not the parameter count. For a model with 40 transformer layers, 8 [grouped-query-attention](https://arxiv.org/abs/2305.13245) key-value heads, and a head dimension of 128, each token stores keys and values across all layers: 2 tensors times 8 heads times 128 dimensions times 40 layers, or 81,920 elements per token. In fp16 that is 163,840 bytes, almost exactly 160 KiB per token. Grouped-query attention, which the [Qwen3 technical report](https://arxiv.org/abs/2505.09388) describes for this family, is what keeps that number small. With 8 KV heads instead of one per query head, the cache is a fraction of what full multi-head attention would store.

At 160 KiB per token, a turn holding 6k tokens of context, a reasonable average for a coding session where the model keeps reading files back, costs about 0.94 GiB of VRAM. A 16k-token turn, a big refactor that has pulled several files into context, costs about 2.56 GiB. Those two numbers are the whole story.

## Eight running turns already sit at 13.5 GiB

The swarm runs eight decode slots. At a 6k-token average, eight running turns hold 7.5 GiB of KV. Add the 6 GiB of weights and runtime and the card is at 13.5 GiB with every slot busy and nothing paused. Against a 15.5 GiB usable ceiling, that is 2 GiB of headroom.

Two gigabytes is about two average paused turns. That is the entire budget for keeping preempted work resident. Preempt one long background turn to admit a foreground turn and you are fine. Preempt a second while the first is still parked and you are near the edge. Preempt a third and the card is over, at which point the scheduler has no choice but to evict a paused turn's KV to host memory or drop it and recompute it later. That is the swap or recompute path yesterday's post was proud to avoid, and it comes back the moment the resident set exceeds VRAM.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-08-12-rdna4-preemption-vram-ceiling-paused-kv.svg" alt="A stacked column chart on a deep teal-black background titled 'Every paused turn keeps its KV in VRAM, so preemption has a ceiling'. The vertical axis is VRAM used on one 16 GB RX 9070 XT running Qwen3.5-9B, from 0 to 18 GiB, with a red dashed ceiling line at 15.5 GiB of usable VRAM. Every column shares a fixed base of about 5.2 GiB of Q4 K M weights and 0.8 GiB of runtime buffers, plus 7.5 GiB of KV for eight running decode turns at a 6k-token average context, bringing the running-only column to 13.5 GiB and leaving 2 GiB of headroom. Moving right, one, two, three and four average paused turns are added at about 0.94 GiB each in amber. Two paused turns still fit at 15.4 GiB; three and four pierce the ceiling and the over-budget slice is drawn in hatched coral labelled forced swap or recompute. A separate rightmost column shows a single long 16k-token background turn adding 2.56 GiB and overflowing the ceiling on its own. A dashed teal line low in the chart tracks the same states with q8 key-value cache, which halves every KV term and never approaches the ceiling." loading="lazy" />
  <figcaption>The same swarm as paused turns accumulate. The base of every column is identical: weights, runtime, and eight running turns at 13.5 GiB. Only the amber paused-KV slice grows. Two average paused turns fit under the 15.5 GiB ceiling; the third and fourth spill into the hatched region, where the card is full and preemption falls back to swap or recompute. The dashed teal line is the same states with a q8 KV cache, which stays far under the ceiling.</figcaption>
</figure>

What the chart makes obvious is that the base never moves. The card is committed to 13.5 GiB the instant all eight slots are busy, so the only variable is how much paused KV you can stack on top before hitting the line. The answer is not many.

## The turns worth preempting are the expensive ones to park

Here is the part that makes the ceiling bite harder than the average suggests. You do not preempt turns at random. You preempt the turn that is holding a slot while doing low-priority work, and the reason it is worth preempting is usually that it is long: a full-file rewrite, a multi-file refactor, a thousand-token generation grinding through its slot. Those turns have big contexts, which means big KV caches.

So the paused set is biased toward exactly the turns that are most expensive to keep resident. The 2 GiB of headroom that holds two average 0.94 GiB turns holds zero long ones. A single paused 16k-token refactor is 2.56 GiB, which overflows the budget by itself. That is the rightmost column in the chart. The mechanism that made preemption cheap, leaving the KV in place, is the same mechanism that fills the card fastest when the preempted turns are the heavy ones.

This is the tension with the [reserve-for-max-context](/blog/2026-08-01-reserve-for-max-kv-cache-fits-four-of-eight-agents-on-rdna4/) result from earlier in the series. If you sized the swarm by reserving each slot's worth of maximum context up front, you would fit far fewer than eight turns, so the swarm overcommits and counts on most turns being short. Overcommitting works right up until preemption asks you to hold a long turn's full cache resident on top of a full running set. The reservation you skipped is the headroom you now do not have.

## Where the ceiling actually lands

The numbers below put the two regimes side by side: the fp16 KV cache the swarm runs by default, and the q8 cache that halves every KV term.

| KV precision | Per running turn (6k ctx) | 8 running turns | Free KV headroom | Avg paused turns that fit | One long 16k turn |
| --- | ---: | ---: | ---: | ---: | --- |
| fp16 | 0.94 GiB | 7.5 GiB | ~2.0 GiB | about 2 | overflows (2.56 GiB) |
| q8 | 0.47 GiB | 3.75 GiB | ~5.75 GiB | about 12 | fits, ~4.5 GiB left |

The table says two useful things. First, the fp16 swarm can keep only about two average turns paused before it is out of room, and it cannot hold even one long paused turn without a fallback. Second, quantizing the KV cache to q8 does not remove the ceiling but it moves it a long way. Halving the KV footprint roughly triples the free headroom, from about 2 GiB to almost 6 GiB, and because each paused turn is now half the size too, the count of average paused turns that fit jumps from about two to around a dozen. That is the same lever the [q8 KV crossover post](/blog/2026-07-31-q8-kv-cache-pushes-an-rdna4-swarm-crossover-from-5k-to-10k-tokens/) measured when q8 pushed the swarm's fit boundary from 5k to 10k tokens of context. Preemption headroom is one more thing that KV precision quietly controls.

None of this contradicts yesterday's post. Token-boundary preemption is still the right tool for priority inversion, and the eviction itself is still one decode step. The correction is narrower: the resume is free only while the paused turn stays resident, and residence is capped by VRAM. Past that cap you are back to the choices [vLLM already enumerates](https://docs.vllm.ai/en/stable/configuration/optimization/), swap the KV out to host memory over PCIe or recompute it from prefill, and the same measurement that found [swapping beats recompute by 177x](/blog/2026-07-20-swapping-an-idle-agents-kv-cache-beats-recomputing-it-by-177x/) only held because the memory was there to swap into.

## What to actually do with a paused turn

The honest scheduling picture is that a single-card swarm has three tiers of cost for a preempted turn, and it should walk down them in order. The cheapest tier is leave it resident, which costs nothing to resume and is available only while VRAM has room. The middle tier is swap its KV to host memory, which costs a PCIe round trip on resume but frees the VRAM immediately. The expensive tier is drop the KV and recompute the turn's prefill when it comes back, which costs real GPU time proportional to context length.

[Iteration-level scheduling](https://www.usenix.org/conference/osdi22/presentation/yu) gives you the clean interruption point to make the choice, and the [paging model from vLLM](https://arxiv.org/abs/2309.06180) gives you the machinery to move KV blocks in and out without fragmenting the pool. What the single-card case adds is a budget the datacenter case mostly ignores: with one 16 GB card there is no second GPU to spill onto, so the resident tier is small and the scheduler crosses into the paying tiers quickly. The right policy keeps the foreground turn resident, keeps the short paused turns resident, and is the first to swap the long paused turns out, because they are the ones eating the budget and the ones cheapest to recover from host memory relative to their size.

The lesson from yesterday still stands: you fix priority inversion by taking the slot back at a token boundary. Today's correction is that taking the slot back does not take the memory back. The paused turn is still sitting in VRAM, and on a 16 GB card there is only room for about two of them before the card decides the question for you.
