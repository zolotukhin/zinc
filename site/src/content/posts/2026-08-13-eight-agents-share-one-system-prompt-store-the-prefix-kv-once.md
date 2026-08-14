---
title: "Eight agents share one system prompt, so store the prefix KV once"
seoTitle: "Shared-Prefix KV Caching Reclaims VRAM in an RDNA4 Agent Swarm"
date: "2026-08-13"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - kv-cache
  - prefix-cache
  - radixattention
  - prefix-sharing
  - vram
  - agents
  - local-llm
  - llm-inference
keywords:
  - shared prefix KV cache agent swarm
  - store system prompt KV once RDNA4
  - automatic prefix caching vLLM local
  - RadixAttention shared prefix radix tree
  - tool schema prefix KV duplication VRAM
  - RX 9070 XT 16 GB prefix cache reclaim
  - prefix sharing prefill compute saving
  - Hydragen shared prefix attention decode
excerpt: "Yesterday's post found the swarm had about 2 GiB of VRAM headroom on a 16 GB RX 9070 XT, and treated that as scarce. It is scarcer than it needs to be, because a good chunk of it is spent storing the same bytes eight times. Every agent in a coding swarm carries an identical prefix, the system prompt plus the tool schemas, and the naive engine caches a separate copy of that prefix's KV for each of the eight decode slots. A 2,000-token shared prefix is 0.30 GiB per copy at fp16, so eight copies cost 2.44 GiB while one shared copy costs 0.30 GiB. Storing it once reclaims 2.14 GiB, which is roughly the entire headroom the swarm was fighting for. This is the prefix-sharing case an earlier post said a single-user engine could mostly skip, and a swarm cannot."
seoDescription: "In a local agent swarm on one 16 GB RX 9070 XT running Qwen3.5-9B, all eight agents share an identical prefix: the system prompt and the tool-call schemas. A naive engine gives each decode slot its own KV cache and stores that prefix eight times. With 40 layers, 8 GQA KV heads and head dim 128, fp16 KV is about 160 KiB per token, so a 2,000-token shared prefix is 0.30 GiB per copy. Eight copies cost 2.44 GiB; deduplicating to one shared copy costs 0.30 GiB and reclaims 2.14 GiB, close to the entire preemption headroom the swarm otherwise has. This is exactly what automatic prefix caching in vLLM and RadixAttention in SGLang do: hash or radix-match the shared blocks, reference-count them, and store them once. It is a capacity and prefill win, not automatically a decode-bandwidth win, since each agent still reads the shared keys unless a shared-prefix attention kernel like Hydragen batches those reads. An earlier post argued a single-user local engine could mostly skip prefix caching; a swarm of eight concurrent agents with a common prompt flips that conclusion."
faqs:
  - question: "Why does an agent swarm store the same system prompt eight times?"
    answer: "Because the default way to run eight concurrent agents is to give each one its own decode slot with its own KV cache, and each agent's context begins with the same static prefix: the system prompt and the JSON schemas for the tools it can call. When an engine caches KV per slot without any cross-slot sharing, it computes and stores the key-value tensors for that identical prefix once per agent. Eight agents means eight byte-for-byte identical copies of the prefix's KV sitting in the same 16 GB of VRAM. Nothing about the prefix differs between agents, so seven of the eight copies are pure duplication."
  - question: "How much VRAM does deduplicating the shared prefix actually save?"
    answer: "It depends on how long the shared prefix is. For Qwen3.5-9B with 40 layers, 8 grouped-query-attention KV heads and head dimension 128, fp16 KV is about 160 KiB per token. A 2,000-token prefix, a plausible size for a system prompt plus a dozen tool schemas, is therefore about 0.30 GiB per copy. Across eight agents that is 2.44 GiB of KV that is all the same bytes. Storing it once leaves 0.30 GiB and reclaims 2.14 GiB. That is close to the entire preemption headroom an earlier post measured for this swarm, so sharing the prefix roughly doubles the room the scheduler has to work with. A 3,000-token prefix reclaims about 3.2 GiB; a 4,000-token prefix about 4.3 GiB."
  - question: "Is this the same as automatic prefix caching in vLLM or RadixAttention in SGLang?"
    answer: "Yes, it is the same idea applied to a single-card local swarm. vLLM's automatic prefix caching hashes each KV block over its token IDs and its parent block's hash, keeps a global table of physical blocks, and lets any request whose prefix hashes to an existing block point at that block and increment its reference count instead of allocating a new one. SGLang's RadixAttention stores cached prefixes in a radix tree and matches a new request's prefix against the tree, reusing the KV of the longest matching path. Both store a shared prefix once and hand it to every request that begins with it. In a swarm where all eight agents open with the same system prompt, that shared path is long and the hit rate is close to total."
  - question: "Does sharing the prefix also make decoding faster, or only save memory?"
    answer: "By itself it saves memory and saves redundant prefill, not decode bandwidth. Only the first agent has to run the prefix through the model to fill its KV; the other seven get a cache hit and skip that prefill compute entirely, which is a real latency win when an agent spins up. But during decode, each agent's attention still reads over the whole context including the shared prefix keys and values, so even though those keys are stored once, eight agents streaming them separately still pay eight reads. Cutting that requires a shared-prefix attention kernel like Hydragen, which computes attention over the shared prefix and the unique suffixes separately and batches the prefix reads across the whole group. So plain KV sharing is a capacity and prefill win first; the bandwidth win is a second, kernel-level step."
draft: false
---

[Yesterday's post](/blog/2026-08-12-free-rdna4-preemption-ends-where-paused-kv-fills-the-card/) ended on a tight number. A single 16 GB RX 9070 XT running eight decode slots of Qwen3.5-9B sits at about 13.5 GiB with every slot busy, which leaves roughly 2 GiB of headroom, about two average paused turns before preemption falls back to swap or recompute. The whole argument treated that 2 GiB as scarce, something to ration carefully.

It is scarcer than it needs to be, and for an embarrassing reason. A good slice of the KV cache the swarm is spending is eight copies of the same bytes. Every agent in a coding swarm opens its context with an identical prefix, the system prompt and the tool-call schemas, and the straightforward way to run eight agents gives each one its own KV cache with its own private copy of that prefix. Seven of those eight copies are pure duplication.

Store the prefix once instead of eight times and you get most of the headroom back. The arithmetic is not subtle, and it points at a fix that inference servers have shipped for years and a single-card local engine has mostly ignored.

## The prefix is identical, the cache is not

Start with what the eight agents actually share. A coding agent's context is not free-form. It begins with a system prompt that sets its role and rules, followed by the JSON schemas for every tool it can call, read a file, run a command, search, edit, and so on. That whole preamble is static. It is the same tokens in the same order for every agent in the swarm, because they are running the same harness with the same tool set. Only after the preamble does each agent diverge into its own conversation: the files it has read, the task it was given, the output it has generated.

So the context of agent N is a shared head plus a unique tail. The head is identical across all eight. The tail is what makes agent N different from agent M. In KV-cache terms, the head produces exactly the same keys and values for every agent, layer for layer, because attention is deterministic given the same tokens and the same weights.

An engine that caches KV per decode slot, with no awareness that the slots share anything, computes that identical head eight times and stores eight identical copies of its KV. It is the software equivalent of eight people each printing their own copy of the same 40-page manual because no one thought to leave one on the shared desk.

## The arithmetic on a 16 GB card

Put numbers on the head. The KV-cache size per token is set by the attention shape, which for this series is Qwen3.5-9B with 40 transformer layers, 8 [grouped-query-attention](https://arxiv.org/abs/2305.13245) key-value heads, and head dimension 128. That is 2 tensors times 8 heads times 128 dimensions times 40 layers, or 81,920 elements per token, which in fp16 is 163,840 bytes, almost exactly 160 KiB per token. This is the same per-token figure yesterday's post used, so the two posts are measuring the same card.

A system prompt plus a dozen tool schemas is a real chunk of tokens. Call the shared prefix 2,000 tokens, which is on the modest side for an agent harness with a full tool set. At 160 KiB per token, that prefix costs about 0.30 GiB of KV per copy. One agent's copy is fine. The problem is that the naive swarm holds eight of them, 2.44 GiB, and 2.14 GiB of that is redundant.

That 2.14 GiB is not on top of the 13.5 GiB from yesterday, it is inside it. The eight running turns were already counted at 7.5 GiB of KV, and up to 2.44 GiB of that 7.5 is the eight prefix copies. Deduplicate them and the running set's KV drops toward 5.4 GiB, which turns yesterday's 2 GiB of headroom into something closer to 4 GiB. The single cheapest thing the swarm can do to make room is stop storing the same prefix eight times.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-08-13-shared-prefix-kv-radix-tree-rdna4-agent-swarm.svg" alt="A two-panel comparison on a deep plum background. The left panel, labelled naive, shows eight decode rows, one per agent, each beginning with an identical amber block for the same 2,000-token system prompt and tool schemas, followed by a differently sized cool-coloured block for that agent's unique working context. A coral dashed outline groups the eight amber blocks, annotated: same 2,000 tokens, copied 8 times, equals 2.44 GiB of duplicated KV. The right panel, labelled shared prefix, shows a single tall amber node holding the same 2,000-token prefix stored once at 0.30 GiB, with eight curved branches fanning out like a radix tree to the eight agents' unique cool-coloured tails, annotated: one copy, 0.30 GiB, reclaims 2.14 GiB. A footer gives the arithmetic: fp16 KV is about 160 KiB per token, so a 2,000-token prefix is 0.30 GiB per copy, and sharing reclaims about the whole preemption headroom on a 16 GB card." loading="lazy" />
  <figcaption>Left: the naive swarm stores the same 2,000-token prefix once per agent, 2.44 GiB of which 2.14 GiB is duplication. Right: a radix node holds the prefix once and every agent's unique tail branches off it, reclaiming 2.14 GiB. Amber is the shared prefix; each colour is one agent's unique working context. Proportions are illustrative; the GiB figures are computed at 160 KiB per token.</figcaption>
</figure>

What the picture makes clear is that the shared head is a single object with eight tails hanging off it, not eight independent sequences that happen to look alike. Once you draw it that way, storing it eight times looks like the accident it is.

## This is the case a single user gets to skip and a swarm does not

Back in May this blog argued the opposite, and it was right at the time. The post on [paged KV cache being the serving fix a single-user local engine can mostly skip](/blog/2026-05-26-paged-kv-cache-is-the-serving-fix-a-single-user-local-engine-can-mostly-skip/) made the case that prefix sharing and block-granular allocation earn their keep by packing many concurrent sequences, and a desktop assistant running one conversation has nothing to pack. The only exception it flagged was a conversation that branches and shares a prefix, which felt like an edge case for a single user.

A swarm is that edge case as the default. Eight agents with a common system prompt are eight concurrent sequences that share a long prefix, which is precisely the workload prefix sharing was built for. The thing a single user could skip is the thing a swarm has to do.

And the machinery already exists. [vLLM's automatic prefix caching](https://docs.vllm.ai/en/stable/design/automatic_prefix_caching.html) hashes each KV block over its own token IDs and its parent block's hash, keeps a global table of physical blocks, and lets any request whose prefix hashes to an existing block point at that block and bump its reference count rather than allocate a new one. [SGLang's RadixAttention](https://arxiv.org/abs/2312.07104) does the same job with a radix tree: it matches a new request's prefix against the tree and reuses the KV of the longest matching path, storing the shared prefix once. Both are the same move the vLLM [PagedAttention paper](https://arxiv.org/abs/2309.06180) listed as its second win, flexible sharing of KV within and across requests. The point of this post is narrower than any of those systems: on one 16 GB card the shared prefix in an agent swarm is long, the hit rate is close to total, and the memory it frees is exactly the memory the scheduler was short on.

## Where the saving lands as the prefix grows

The reclaim scales with the prefix length, since the duplication is seven extra copies of whatever the prefix costs. The table puts a few plausible prefix sizes side by side for the eight-agent swarm.

| Shared prefix | KV per copy | Eight copies | One shared copy | VRAM reclaimed |
| --- | ---: | ---: | ---: | ---: |
| 1,000 tokens | 0.15 GiB | 1.22 GiB | 0.15 GiB | 1.07 GiB |
| 2,000 tokens | 0.30 GiB | 2.44 GiB | 0.30 GiB | 2.14 GiB |
| 3,000 tokens | 0.46 GiB | 3.66 GiB | 0.46 GiB | 3.20 GiB |
| 4,000 tokens | 0.61 GiB | 4.88 GiB | 0.61 GiB | 4.27 GiB |

The table says the obvious thing and one less obvious thing. The obvious part is that a longer shared prefix means a bigger reclaim, linearly. The less obvious part is that the reclaim is large relative to the card even at modest prefix lengths, because the multiplier is the whole swarm. At 3,000 tokens the swarm gets back 3.2 GiB, which on top of yesterday's 2 GiB headroom is enough to hold three or four extra paused turns resident, or to admit more agents before the [reserve-for-max-context math](/blog/2026-08-01-reserve-for-max-kv-cache-fits-four-of-eight-agents-on-rdna4/) runs out. Prefix length is a knob agent harnesses tend to grow over time as they add tools, and every token added to the shared prefix is multiplied by eight in the naive layout and by one in the shared layout.

## The constraints, stated honestly

Sharing is not free of conditions, and three of them matter. The first is that the match has to be exact. vLLM's block hash chains each block onto its parent's hash and its token IDs, so a cache hit requires the prefix to be byte-for-byte identical up to the block boundary. If the harness injects anything per-agent into the system region, an agent ID, a timestamp, a per-task instruction, it breaks the shared path at the point of divergence and everything after it stops sharing. The fix is a prompt layout discipline: put the truly static prefix first and the per-agent material after it, so the shared head is as long as it can be before the tails split.

The second constraint is that sharing bounds the saving to the prefix, not the whole context. The tails are genuinely different and each still costs its own KV. As agents run longer and their unique contexts grow, the shared prefix becomes a smaller fraction of each agent's total, so the reclaim, while still 2.14 GiB in absolute terms, is a shrinking share of the KV budget. Sharing the prefix buys headroom; it does not change the per-agent growth that eats headroom back.

The third is the one worth being precise about, because it is easy to oversell. Storing the prefix once is a capacity win and a prefill win, not automatically a decode-bandwidth win. Only the first agent runs the prefix through the model; the rest get a cache hit and skip that prefill, which is real saved compute when an agent starts. But during decode each agent's attention still reads over the full context, shared keys included, so eight agents streaming the same prefix keys separately still pay eight reads even though the bytes live in one place. Collapsing those reads needs a shared-prefix attention kernel, which is what [Hydragen](https://arxiv.org/abs/2402.05099) does: it computes attention over the shared prefix and the unique suffixes separately and batches the prefix queries across the group, which reduces the redundant memory reads that dominate large-batch shared-prefix decode. That is a second, kernel-level step. The memory saving comes for free with sharing; the bandwidth saving has to be built.

## What to reach for

The framing that ties this back to the rest of the series is that the swarm's scarcest resource is VRAM, and the first thing to do with a scarce resource is stop wasting it on duplicates. Yesterday's post treated the 2 GiB of headroom as a hard limit on how many paused turns could stay resident. Half of that limit is self-inflicted, because the same 2,000-token prefix is sitting in the card eight times over. Deduplicate it and the limit moves, using machinery that vLLM and SGLang have shipped for years and that this blog previously argued a single-user engine could skip.

The honest one-line version is that an agent swarm is not eight independent users, it is one system prompt with eight tails. The moment you store it that way, the card has room it did not appear to have.
