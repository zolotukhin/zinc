---
title: "A reserve-for-max KV cache fits four of eight agents on a 16 GB RDNA4 card"
seoTitle: "Contiguous vs Paged KV Cache on RDNA4: Why an Agent Swarm Runs Out of VRAM Early"
date: "2026-08-01"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - kv-cache
  - paged-attention
  - pagedattention
  - vllm
  - memory-management
  - agents
  - local-llm
  - llm-inference
keywords:
  - paged KV cache RX 9070 XT agent swarm
  - contiguous vs paged KV cache RDNA4
  - PagedAttention KV cache fragmentation local GPU
  - reserve for max context KV cache waste
  - vLLM PagedAttention 60 to 80 percent waste
  - vAttention contiguous virtual paged physical
  - 16 GB VRAM agent admission wall
  - local coding agent swarm out of memory
excerpt: "A single local user could mostly skip a paged KV cache, and this blog said so in May. An agent swarm reverses that verdict. Eight agents on one 16 GB RX 9070 XT each grow an unpredictable amount of context, so a contiguous cache has to reserve every agent's worst case up front. Reserve 16k tokens each and the card admits four agents, not eight, while most of the reserved VRAM sits empty. Paging commits memory a block at a time and fits all eight in the same budget. This is the fragmentation problem PagedAttention was built for, arriving on a consumer card the moment one user turns into eight."
seoDescription: "On a 16 GB RX 9070 XT running Qwen3.5-9B at Q4_K_M, about 10 GB is left for KV cache after the weights. A contiguous cache must reserve each agent's maximum context before its first token, so provisioning eight agents for 16k tokens each needs 17.9 GB and the card admits only four. Dropping the reservation to 8k fits eight agents but caps every one of them, and at a realistic 3k-token average it wastes 63 percent of the reserved KV budget, squarely inside the 60 to 80 percent that the PagedAttention paper measured for contiguous allocation. A paged cache commits 16-token blocks on demand, fits all eight agents, wastes under one percent, and lets the swarm hold about 71k aggregate tokens before it runs out of VRAM. This post shows why the single-user verdict that a local engine can skip paging flips for an agent swarm, and where vAttention's contiguous-virtual paged-physical design fits on an AMD card."
faqs:
  - question: "Why does a contiguous KV cache waste memory for an agent swarm?"
    answer: "A contiguous cache gives each sequence one unbroken span of VRAM, and it has to reserve that span before the first token because the memory must be contiguous and the context length is not known in advance. For a swarm of eight agents whose contexts grow unpredictably, that means reserving each agent's worst-case context up front. Most agents sit far below their reservation most of the time, so the reserved-but-unused gap is pure waste. On a 16 GB RX 9070 XT with about 10 GB free for KV, reserving 8k tokens per agent for eight agents commits 9 GB but at a realistic 3k-token average only 3.4 GB is live, wasting 63 percent."
  - question: "How many agents fit on a 16 GB card with a contiguous KV cache?"
    answer: "It depends entirely on the context you reserve for each. Qwen3.5-9B at Q4_K_M leaves about 10 GB for KV after 5.5 GB of resident weights, and FP16 KV costs about 0.14 MB per token per agent. Reserve 16k tokens each and one agent's slot is 2.24 GB, so only four agents fit in 10 GB even though at the start of a session each is holding a few hundred tokens. Reserve 8k each and eight agents fit at 1.12 GB apiece, but now no agent can exceed 8k without being evicted. Contiguous allocation forces a trade between how many agents you admit and how much context each may hold."
  - question: "Does a single local user need a paged KV cache?"
    answer: "Mostly not, which is what this blog argued in May. A single-user engine at a batch of one runs a linear conversation and can size one contiguous arena from the same budget math a server uses, with no fragmentation to recover. Paging earns its keep when memory is shared and dynamic across many sequences: a branching tree of reused prefixes, or an agent swarm of independent contexts. The swarm is the case that flips the verdict, because eight unpredictable contexts on one fixed card are exactly the multi-tenant fragmentation problem paging was designed to solve."
  - question: "What is PagedAttention and how much memory does it save?"
    answer: "PagedAttention, from the vLLM paper at SOSP 2023, stores each sequence's KV cache in fixed-size blocks that need not be contiguous in memory, tracked by a per-sequence block table, the same idea as virtual memory paging in an operating system. Because a block is only committed when a sequence actually fills it, there is no reserve-for-max waste. The paper reports that prior contiguous systems wasted 60 to 80 percent of KV memory to fragmentation, while vLLM holds waste under 4 percent, and that this lifts throughput 2 to 4 times by fitting a larger batch."
  - question: "Is paging the only way to fix KV cache fragmentation?"
    answer: "No. vAttention, to appear at ASPLOS 2025, argues that PagedAttention's non-contiguous layout adds programming and kernel overhead, and instead decouples virtual from physical memory: it reserves a contiguous virtual address range per sequence and commits physical pages into it on demand. The attention kernel still reads a flat span while physical memory is never over-reserved. On an AMD card an engine that owns its amdgpu submission path can map and unmap physical buffer objects into a reserved GPU virtual range itself, or use Vulkan sparse residency, getting the same on-demand commit without a software block table."
draft: false
---

In May this blog argued that a paged KV cache is the [serving fix a single-user local engine can mostly skip](/blog/2026-05-26-paged-kv-cache-is-the-serving-fix-a-single-user-local-engine-can-mostly-skip/). The reasoning held: one user at a batch of one runs a linear conversation, so an engine can carve a single contiguous arena from its VRAM budget and never pay for the block table that a multi-tenant server needs. The waste that [PagedAttention](https://arxiv.org/abs/2309.06180) was built to recover is the waste of a crowd, and a single user is not a crowd.

The July series spent two weeks turning that single user into a crowd. [Eight concurrent agents on one RX 9070 XT](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) pull 5.6 times the aggregate throughput of one, and that is the whole case for running a local swarm. But the swarm changes the memory problem underneath it, and it changes it in exactly the direction that makes contiguous allocation break. Eight agents are eight independent contexts, each growing an unpredictable amount, all sharing one fixed 16 GB card. That is a crowd, and the verdict flips.

The sharp version of the flip is an admission number. Provision a contiguous cache to run eight agents at up to 16k tokens each, and the card admits four. The other four never start, while more than half of the memory the first four reserved sits empty. Nothing is wrong with the card. The allocation scheme reserved memory that no agent was using yet, and ran out of room to hand any to the fifth.

## The reservation you have to make before the first token

Start with the budget. Qwen3.5-9B at Q4_K_M keeps about 5.5 GB of weights resident on the RX 9070 XT, which leaves roughly 10 GB of the 16 GB for KV cache across every agent. At FP16 a token of KV costs about 0.14 MB per agent, summed over all layers and both the key and value tensors, the same number the [Q8 crossover post](/blog/2026-07-31-q8-kv-cache-pushes-an-rdna4-swarm-crossover-from-5k-to-10k-tokens/) used a day ago.

A contiguous cache gives each agent one unbroken span of that 10 GB. The catch is that the span has to be reserved before the agent produces its first token, because contiguity is the whole point and you cannot grow a contiguous region into memory another agent is already sitting in. So the engine has to answer a question it cannot actually answer: how long will this agent's context get? It has to assume the worst case and reserve for it.

Reserve 16k tokens per agent and one slot is 16,000 times 0.14 MB, about 2.24 GB. Eight of those is 17.9 GB, which does not fit in 10 GB, so the card admits four agents and stops. This is the same admission wall the series hit when [a ninth agent stalled the other eight](/blog/2026-07-19-admitting-a-ninth-agent-stalls-the-other-eight-for-8-3-seconds/), except now it lands at the fifth, and it lands not because the KV is full but because it was promised away.

## Where the reserved gigabytes actually go

The reservation would be defensible if the agents used it. They do not, at least not most of the time. A coding agent spends most of a session holding a few thousand tokens, not sixteen thousand, and the eight agents in a swarm are almost never all near their ceiling at once. So the reserved slots stand mostly empty, and the empty part is memory the card cannot lend to anyone else.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-08-01-rdna4-swarm-kv-allocation-map.svg" alt="A VRAM allocation map on a dark graphite background titled 'Contiguous reserve-for-max vs paged KV cache on one 16 GB RX 9070 XT'. Two horizontal 16 GB memory bars sit stacked. The top bar, labelled Contiguous reserve-for-max 16k, begins with a 5.5 GB amber block for resident weights, then four wide slate slots for agents one to four, each 2.24 GB, each showing a thin green filled portion near its left edge for about 3k tokens of live KV and a large hatched empty remainder. After the fourth slot a red vertical wall labelled admission wall marks that the 10 GB KV budget is exhausted, and agents five to eight are drawn as greyed-out empty outlines beyond the card's edge with a label eight requested, four admitted. The bottom bar, labelled Paged 16-token blocks, begins with the same 5.5 GB amber weights block, then eight narrow violet stacks of small blocks, one per agent, each sized to its actual 3k-token use, packed tightly with almost no gap, followed by a wide green free-headroom region labelled about 6.6 GB still free, all eight agents admitted. A caption strip at the bottom reads live KV 3.4 GB in both cases; contiguous reserves 9 GB and wastes 63 percent, paged commits on demand and wastes under 1 percent." loading="lazy" />
  <figcaption>Modeled for Qwen3.5-9B Q4_K_M on one 16 GB RX 9070 XT, 5.5 GB weights resident, about 10 GB for KV, FP16 KV at 0.14 MB per token per agent. Both schemes hold the same 3.4 GB of live KV at a 3k-token average. Contiguous reservation carves fixed worst-case slots and strands the empty remainder, hitting an admission wall at the fifth agent. Paging commits 16-token blocks on demand, admits all eight, and leaves the unused budget free for whoever needs it.</figcaption>
</figure>

The map is the argument. Both bars hold the same amount of live KV, the green portions, because the agents are doing the same work. The difference is everything around the green. Contiguous allocation surrounds each agent's live KV with reserved-but-empty space it cannot reclaim, and that reserved space is what runs the card out of room. Paging keeps only what is live plus at most one partly filled block per agent, so the same 3.4 GB of real KV leaves more than 6 GB genuinely free.

## The same three numbers, three ways

Hold the goal fixed at eight agents that may each grow toward 16k tokens, and the allocation scheme is the only variable. The reserve-for-max column is the memory promised before any token exists. The waste column is measured at a realistic mid-session snapshot where each agent is holding about 3k tokens, so the live KV is 8 times 3,000 times 0.14 MB, about 3.4 GB.

| Allocation scheme | Reserved for 8 agents | Fits in 10 GB? | Per-agent context cap | Wasted at 3k average |
| --- | ---: | :---: | :---: | ---: |
| Contiguous, reserve 16k | 17.9 GB | no, admits 4 | 16k | cannot run all 8 |
| Contiguous, reserve 8k | 9.0 GB | yes | 8k hard cap | 63% |
| Paged, 16-token blocks | commit on demand | yes | none until 10 GB | under 1% |

The middle row is the compromise a contiguous engine is pushed into, and it is a bad one. Dropping the reservation to 8k lets all eight agents start, but now none of them can pass 8k tokens, which is an ordinary coding turn once tool output is folded back into the context, exactly the [eighteen-to-one read-back ratio](/blog/2026-07-23-a-local-coding-agent-reads-back-eighteen-tokens-for-every-one-it-writes/) the series measured. And even with the cap, at a 3k average the scheme has reserved 9 GB to hold 3.4 GB, wasting 63 percent. That number is not an accident. The PagedAttention paper found contiguous systems wasted 60 to 80 percent of KV memory to exactly this reserve-for-max fragmentation, and 63 percent lands right in the middle of it. The consumer card reproduces the datacenter result because it is the same problem.

The paged row spends the same 3.4 GB on live KV, commits at most a 16-token block beyond what each agent has filled, and holds waste under one percent, which is what [vLLM reports](https://docs.vllm.ai/en/latest/design/paged_attention.html) for its block manager. It admits all eight agents, imposes no artificial context cap until the real 10 GB is gone, and lets the swarm reach about 71k aggregate tokens before it actually runs out of memory, against the 9 GB the contiguous scheme spent to hold a third of that.

## Why the single-user verdict flipped

The May post was not wrong, and it is worth being precise about why. Paging earns its keep when memory is both shared and dynamically sized across many sequences. A single user at a batch of one has neither property in the common case: one linear conversation, one contiguous arena, no fragmentation to recover, so importing a block table imports cost without benefit. The one exception the post named was a branching tree of reused prefixes, where sibling branches need to share a prefix's blocks. The swarm is a second exception, and a blunter one. Eight agents are eight sequences that do not share a prefix at all, each with its own dynamically growing context, on one fixed card. That is the multi-tenant fragmentation case in full, just arriving on a 16 GB consumer GPU instead of an A100.

Paging is not the only way out, and it is worth naming the alternative because it fits an AMD engine well. [vAttention](https://arxiv.org/abs/2405.04437), to appear at ASPLOS 2025, points out that PagedAttention's block table makes the KV cache non-contiguous in virtual memory, which pushes overhead into every attention kernel that then has to gather blocks. It keeps virtual memory contiguous and pages only physical memory, reserving a flat virtual range per sequence and committing physical pages into it on demand. The attention kernel reads a flat span and never knows the difference, while the card never over-reserves. An engine like zinc that submits directly through the amdgpu path owns its GPU virtual address space, so it can map and unmap physical buffers into a reserved range itself, or lean on Vulkan sparse residency for the same effect, getting on-demand commit without a software block table in the hot loop.

## The order of the memory levers

The July series has been a steady accounting of memory-side levers for a local swarm, and this is the one that decides whether the swarm exists. [Continuous batching](/blog/2026-07-30-static-batching-drains-rdna4-swarm-throughput/) keeps the shared weight stream amortized, and [Q8 KV](/blog/2026-07-31-q8-kv-cache-pushes-an-rdna4-swarm-crossover-from-5k-to-10k-tokens/) halves the private read so the swarm stays fast deeper into a task. Both assume the agents are resident in the first place. On-demand allocation is what gets them resident, because it is the difference between admitting four agents and admitting eight on the same card.

The order to build them in is the order of hard limits. Allocate on demand first, so the card holds the full swarm instead of stranding half its VRAM in empty reservations. Then quantize the KV so the read each agent does stays cheap as its context grows. Then schedule the batch so the shared weights never go to waste. The paged cache a single user could skip is the floor the other two levers stand on, and the swarm is what put it back under them.
