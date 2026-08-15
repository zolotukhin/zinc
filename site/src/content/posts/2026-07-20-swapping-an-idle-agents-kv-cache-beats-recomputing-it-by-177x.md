---
title: "Swapping an idle agent's KV cache beats recomputing it by 177x"
seoTitle: "KV Cache Swap vs Recompute on a Local GPU: The 177x Gap on RDNA4"
date: "2026-07-20"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - kv-cache
  - preemption
  - scheduling
  - agents
  - pcie
  - vllm
  - local-llm
  - llm-inference
keywords:
  - KV cache swap vs recompute
  - vLLM preemption mode RECOMPUTE
  - KV cache offload to host RAM
  - PCIe 5.0 KV cache transfer
  - local LLM agent VRAM pressure
  - RX 9070 XT KV cache budget
  - CPU offload KV cache local inference
  - preemption policy local inference engine
  - 147 KB per KV token Qwen3.5-9B
  - idle agent GPU memory
excerpt: "An agent that stops to run a shell command keeps about a gigabyte of KV cache pinned in VRAM while it produces nothing. On an RX 9070 XT, evicting that cache to host DDR5 and bringing it back costs about 41 ms. Rebuilding it with prefill costs 7.2 seconds. Server engines default to the 7.2-second option, and on a local card that default is backwards."
seoDescription: "Qwen3.5-9B costs 147 KB per fp16 KV token, so a 6,900-token agent holds 1.01 GB. Copying that to pinned host memory and back over PCIe 5.0 x16 takes about 41 ms at 50 GB/s. Recomputing it at ZINC's measured 962 tok/s prefill takes 7.2 seconds, a 177x gap that holds at every context length because both costs are linear in tokens. vLLM V1 defaults to RECOMPUTE because a datacenter GPU prefills far faster and its PCIe link is contended. A local box has 96 GB of DDR5 sitting idle and one agent per card, which inverts the decision."
faqs:
  - question: "Why does an idle agent cost anything at all?"
    answer: "Because its KV cache stays resident. Qwen3.5-9B costs about 147 KB per fp16 KV token, so an agent holding 6,900 tokens of context occupies 1.01 GB of VRAM. While that agent is blocked waiting for a shell command or a file read, the gigabyte is doing nothing, and on a 16 GB RX 9070 XT there are only about 55,000 KV tokens of room in total."
  - question: "How much does it cost to move a KV cache to host RAM?"
    answer: "For a 6,900-token agent, 1.01 GB each way. At 50 GB/s of pinned-memory copy over PCIe 5.0 x16 that is 20.3 ms out and 20.3 ms back, about 41 ms round trip. The 50 GB/s figure is a plausible fraction of the 63 GB/s the link can signal, and I have not measured it on this box."
  - question: "Why do vLLM and other servers recompute instead of swapping?"
    answer: "vLLM's docs state that in V1 the default preemption mode is RECOMPUTE rather than SWAP because recomputation has lower overhead in the V1 architecture. That is a reasonable default for a server: a datacenter GPU prefills far faster than 962 tok/s, its PCIe link is shared with other traffic, prefix caching absorbs much of the recompute cost, and host memory in a container is not free. None of those conditions hold on a single-user desktop."
  - question: "Does the 177x ratio change with context length?"
    answer: "No, and that is the useful part. Both costs are linear in tokens, so the ratio is fixed at 177x for a round trip and 354x if you only count the restore. What changes is the absolute number. At 2,000 tokens the choice is 12 ms against 2.1 seconds. At 32,000 tokens it is 188 ms against 33 seconds."
  - question: "What is the catch?"
    answer: "Three things. Pinned host memory has to be reserved up front and is not available to anything else. The transfer competes with any other PCIe traffic, which on a two-card box is a real constraint. And the vLLM paper found swapping degrades badly when blocks are small, because many tiny transfers cannot fill the link. A 16-token block of Qwen3.5-9B KV is 2.35 MB, which is large enough to avoid that, but the point stands that swap performance is a property of how the cache is laid out."
draft: false
---

The last two posts were about time. Eight agents on one RX 9070 XT pull [5.6 times the aggregate tokens](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) of a single sequence, and admitting a ninth [freezes the other eight for 8.3 seconds](/blog/2026-07-19-admitting-a-ninth-agent-stalls-the-other-eight-for-8-3-seconds/) unless the scheduler chunks the prefill. Both of those are questions about how the card spends milliseconds.

This one is about what the card is holding while nothing is happening.

Watch a coding agent for a minute and most of what you see is waiting. It emits a tool call, then the engine has nothing to do for it until a `grep` returns, a test suite finishes, or an HTTP request comes back. During that window the agent produces no tokens. It also does not release a single byte. Qwen3.5-9B costs about [147 KB per fp16 KV token](/blog/2026-07-13-attentions-two-matmuls-want-different-number-formats-on-rdna4/), so an agent sitting on 6,900 tokens of context is holding 1.01 GB of VRAM hostage while it does nothing at all. On a card with room for roughly 55,000 KV tokens total, eight of those is the entire budget.

## The two ways to take a cache back

Serving engines already solved the mechanics. When KV memory runs short, you preempt somebody, and there are exactly two ways to give them their cache back later.

You can copy it to host memory and copy it back, which the [PagedAttention paper](https://arxiv.org/abs/2309.06180) calls swapping. Or you can throw it away and rebuild it from the prompt, which the paper calls recomputation and notes is cheaper than it sounds, because "the tokens generated at decoding can be concatenated with the original user prompt as a new prompt" and rebuilt in a single prefill pass.

vLLM picked one. Its [optimization guide](https://docs.vllm.ai/en/stable/configuration/optimization/) is explicit: "In vLLM V1, the default preemption mode is `RECOMPUTE` rather than `SWAP`, as recomputation has lower overhead in the V1 architecture." The engine even prints a warning naming the mode when it happens, and the same page treats preemption as a symptom to be tuned away rather than a tool to reach for.

Now price both on this card.

Swapping 1.01 GB across PCIe is a straight bandwidth problem. The RX 9070 XT sits on a [PCI Express 5.0](https://www.asus.com/us/motherboards-components/graphics-cards/prime/prime-rx9070xt-o16g/techspec/) x16 link, which signals about 63 GB/s in each direction. Pinned-memory copies do not reach that, so call it 50 GB/s. Out is 20.3 ms, back is 20.3 ms, round trip 41 ms.

Recomputing the same cache means prefilling 6,900 tokens. ZINC's measured prefill on this card and model is [962 tok/s](/blog/2026-07-14-a-qwen3-5-9b-chat-turn-spends-most-of-its-wall-clock-in-decode/). That is 7.2 seconds.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-20-kv-cache-swap-vs-recompute-reclaim-latency.svg" alt="A two-panel diagram on a dark olive-graphite background. The top panel is a horizontal logarithmic time axis running from 1 millisecond to 100 seconds, titled 'How long it takes to get that gigabyte back'. A rose vertical marker at 41 milliseconds is labelled 'swap out plus swap back', with notes that it moves 1.01 GB each way over PCIe 5.0 x16 and that host RAM keeps the tokens verbatim. A lime vertical marker at 7.2 seconds is labelled 'recompute from the prompt', with notes that it is 6,900 tokens of prefill at 962 tokens per second and that this is vLLM V1's default preemption mode. A dashed line connecting the two markers is labelled 177x. A shaded gold band spanning 50 milliseconds to 30 seconds is labelled as where an agent blocked on a tool call typically sits, covering a file read, a grep, a test run or a web request. The swap marker falls to the left of that band and the recompute marker falls inside it. The bottom panel, titled 'And host RAM is the larger tier', shows two horizontal bars drawn to scale: a short rose bar labelled 'on the card, 8.1 GB of VRAM left after weights' holding 55,000 tokens, and a bar six times longer in lime labelled 'in host DDR5, 48 GB pinned out of a 96 GB kit' holding 326,000 tokens. A footer notes that at 147 KB per fp16 KV token the DDR5 already in this machine holds six times the KV cache the card can, one PCIe hop away, and that ZINC uses none of it today." loading="lazy" />
  <figcaption>Modeled cost of reclaiming one agent's 6,900-token KV cache on an RX 9070 XT running Qwen3.5-9B Q4_K_M, against the range of wall-clock windows a coding agent spends blocked on tool calls.</figcaption>
</figure>

The thing to notice is where the shaded band falls relative to the two markers. A swap round trip finishes before almost any tool call does, so evicting a blocked agent is nearly free in wall clock. A recompute lands squarely inside the range of tool-call durations, which means rebuilding a cache can easily cost more than the work the agent went away to do.

## The ratio does not move, only the stakes

Both costs are linear in tokens, so the gap is a constant.

| Context held | KV bytes | Swap out and back at 50 GB/s | Recompute at 962 tok/s | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 | 0.29 GB | 11.8 ms | 2.1 s | 177x |
| 6,900 | 1.01 GB | 40.6 ms | 7.2 s | 177x |
| 16,000 | 2.35 GB | 94.1 ms | 16.6 s | 177x |
| 32,000 | 4.70 GB | 188.2 ms | 33.3 s | 177x |
| 55,000 | 8.09 GB | 323.4 ms | 57.2 s | 177x |

The right-hand column is the same in every row, which is the tell that this is a hardware ratio and not a workload artifact. Per token, a swap round trip costs 5.9 microseconds and a recompute costs 1.04 milliseconds. For recompute to catch up, prefill would have to run at roughly 340,000 tok/s on a 9B model, which is not a number any single card produces.

What does move is the absolute cost, and it moves into territory a person notices very quickly. Two thousand tokens is a difference between a blink and two seconds. Thirty-two thousand tokens is a difference between a blink and half a minute of a dead terminal.

## Why the server default is still right for servers

None of this means vLLM chose wrong. It means the local conditions are different in three ways that all point the same direction.

Prefill rate is the first. A datacenter GPU running an 8B model prefills at rates an order of magnitude or two above 962 tok/s, which shrinks the left side of the comparison without touching the right. The second is PCIe contention. In a tensor-parallel server the link is already carrying collective traffic, and the [paper's own ablation](https://arxiv.org/abs/2309.06180) found that "swapping incurs excessive overhead with small block sizes" because many small transfers cannot fill the bus, while "the overhead of recomputation remains constant across different block sizes." The third is host memory. In a container with a memory limit, gigabytes of pinned host buffer are a cost the operator has to justify.

On a local box every one of those inverts. The prefill rate is slow because the card is a consumer part. The PCIe link is idle, because one process owns the whole machine and the model weights are already resident. And the host memory is not scarce: this rig has [96 GB of DDR5-6000](/blog/2026-03-26-building-a-local-ai-rig/) that spends its day mostly empty. Reserving 48 GB of it for pinned KV buffers gives 326,000 tokens of second-tier cache, six times what the [16 GB card](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html) can hold.

The block-size caveat deserves a direct answer rather than a wave. A 16-token block of Qwen3.5-9B KV is 2.35 MB, which is nowhere near the regime where DMA setup dominates, and a swap-out can gather blocks into one contiguous staging buffer before the copy anyway. That was a real problem for a model with a smaller per-token cache and it is not one here.

## The policy this actually implies

Being 177x cheaper does more than save time. It changes what preemption is for.

At 7.2 seconds, preemption is a failure mode. You only reach for it when you are already out of memory, which is why vLLM's documentation frames repeated preemption as something to fix by raising `gpu_memory_utilization` or lowering `max_num_seqs`. At 41 ms, preemption becomes an ordinary scheduling move. The break-even is simply whether the agent will be gone longer than the round trip, and a tool call that returns in under 41 ms is rare enough that the answer is almost always yes.

So the rule ZINC should implement is short. When a sequence signals it is blocked on an external tool, evict its KV cache to pinned host memory and hand the VRAM to whoever is decoding. Bring it back when the tool result arrives. If the agent's blocked windows are a fraction `b` of its wall clock, the same card holds roughly `1/(1-b)` times as many agents, which for a coding agent that spends more time waiting on the filesystem than generating is a multiple, not a percentage.

There is an obvious tension with the batching result from two days ago, and I want to name it rather than bury it. The 5.6x came from eight sequences sharing one read of the 5.5 GB weight stream, and that only works if all eight are resident and decoding in the same step. A swap tier does not add throughput to that batch. What it does is let the batch be assembled from a much larger pool of agents, so the eight slots stay full of agents that actually have work instead of agents that are waiting on `cargo test`.

## What I owe

Everything above is arithmetic on two measured numbers and one assumed one. The 962 tok/s prefill is real. The 147 KB per token is real. The 50 GB/s is a guess about a link I have not benchmarked, and if the achieved figure on this X870E board turns out to be 25 GB/s, the ratio halves to 88x and nothing about the conclusion changes.

The measurement I want next is not the bandwidth, though. It is the blocked fraction. I have no instrumented number for how much of an agent's wall clock is spent waiting on tools rather than generating tokens, and that single number decides whether a swap tier is worth building or an interesting piece of arithmetic. It is also the easiest thing on this list to measure, which is usually a sign it should have been measured already.
