---
title: "The system prompt a local agent swarm caches eight times over"
seoTitle: "Prefix KV Cache Sharing on a Local GPU: Deduplicating the System Prompt Across Agents"
date: "2026-07-21"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - kv-cache
  - prefix-caching
  - radixattention
  - agents
  - scheduling
  - vllm
  - sglang
  - local-llm
  - llm-inference
keywords:
  - prefix KV cache sharing local LLM
  - RadixAttention system prompt reuse
  - vLLM automatic prefix caching
  - shared system prompt KV cache RDNA4
  - deduplicate prefix cache across agents
  - RX 9070 XT KV budget agents
  - 147 KB per KV token Qwen3.5-9B
  - copy-on-write KV cache local inference
  - tool schema prefix cache
  - local agent swarm VRAM
excerpt: "Launch eight coding agents from one harness and the first few thousand tokens they read are byte-identical: the same system prompt and tool schemas. On the card, that shared preamble is stored eight separate times. On a 16 GB RX 9070 XT a 4,000-token prefix duplicated across eight agents burns 4.7 GB, more than half the KV budget, holding copies of a thing every agent agrees on."
seoDescription: "Qwen3.5-9B costs about 147 KB per fp16 KV token, so a 4,000-token shared prefix is 0.59 GB per agent. Store it once per sequence across eight agents and that is 4.7 GB of a 16 GB card, 51 percent of the ~55,000-token KV budget, spent on byte-identical copies. RadixAttention and vLLM's automatic prefix caching already deduplicate this on servers. The catch is that a hit needs a 100 percent identical prefix, so the harness must keep timestamps and working directories behind the static system-and-tools block, not in front of it."
faqs:
  - question: "Why do eight agents store the same system prompt eight times?"
    answer: "Because a batch of independent sequences is independent by construction. Each agent's KV cache is allocated for its own sequence, and nothing in a plain batching path notices that the first 4,000 tokens of all eight are identical. At about 147 KB per fp16 KV token for Qwen3.5-9B, a 4,000-token prefix is 0.59 GB, and eight copies is 4.7 GB of a 16 GB card."
  - question: "How much VRAM does prefix sharing actually save?"
    answer: "For a 4,000-token prefix across eight agents, sharing collapses eight copies into one and frees seven, which is 4.1 GB. On an RX 9070 XT with roughly 55,000 tokens of KV budget that is 51 percent of the whole budget. The saving scales with the prefix length and the number of agents: it is (N-1) times the prefix size."
  - question: "What already does this?"
    answer: "SGLang's RadixAttention keeps the KV cache of finished requests in a radix tree and automatically reuses any matching prefix, and it is always on because the authors measured no overhead on a miss. vLLM ships the same idea as automatic prefix caching using a hash table of KV blocks, enabled with enable_prefix_caching=True. Both were built for servers, but the memory math is even more favorable on a single-user local box."
  - question: "What breaks prefix sharing?"
    answer: "A cache hit needs a byte-identical prefix. Anthropic's prompt caching docs put it plainly: hits require 100 percent identical prompt segments up to the cached block. If your harness stitches a timestamp, a working directory, or a per-agent id into the top of the system prompt, the shared prefix ends at the first token that differs. The fix is ordering: keep the static system prompt and tool schemas first, and push everything volatile into the suffix."
  - question: "Does sharing the prefix speed up decode?"
    answer: "No. Prefix sharing is a memory and prefill win. vLLM's documentation is explicit that automatic prefix caching only reduces the prefill phase and does not change decode. The value on a local card is that the reclaimed VRAM lets more agents stay resident and decode together, which is where the throughput came from in the first place."
draft: false
---

Launch eight coding agents from the same harness and the first few thousand tokens each one reads are identical. Same system prompt, same tool schemas, the same house rules about how to format a patch and when to run the tests. The eight sequences only diverge once the first agent opens a different file or the user asks it a different question.

On the card, that shared preamble is not shared at all. Each of the eight sequences carries its own copy of the prefix's KV cache, because a batch of independent sequences is exactly that, independent. Qwen3.5-9B costs about [147 KB per fp16 KV token](/blog/2026-07-13-attentions-two-matmuls-want-different-number-formats-on-rdna4/), so a 4,000-token preamble is 0.59 GB. Store it eight times and you have spent 4.7 GB of a 16 GB card holding eight byte-identical copies of the same thing. That is more than half of the [roughly 55,000 tokens of KV budget](/blog/2026-07-20-swapping-an-idle-agents-kv-cache-beats-recomputing-it-by-177x/) the weights leave behind on an RX 9070 XT.

So the most duplicated bytes in a local agent swarm are the ones every agent already agrees on. That is worth fixing, and the fix is well understood on servers.

## The prefix is identical, and a plain batch ignores it

A serving batch treats each sequence as its own tenant. It allocates KV blocks per sequence, fills them during prefill, and reads them during decode. Nothing in that path looks across sequences to notice that agent 3 and agent 7 share their first 4,000 tokens. The redundancy is invisible to the scheduler unless you go looking for it.

Two production engines went looking. SGLang's [RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) keeps the KV cache of finished and in-flight requests in a radix tree instead of discarding it, so "different prompts with the same prefix can share the intermediate KV cache and avoid redundant memory and computation." When a new request arrives, the runtime matches the longest cached prefix, reuses those blocks, and appends only the new tokens as a fresh branch. The tree lives on the CPU, eviction is least-recently-used, and the ablation in the [SGLang paper](https://arxiv.org/abs/2312.07104) found no measurable overhead on a cache miss, which is why the authors leave the feature on unconditionally.

vLLM ships the same idea under a different name. Its [automatic prefix caching](https://docs.vllm.ai/en/stable/features/automatic_prefix_caching/) hashes KV blocks into a table so that "if a new request shares the system prompt with the previous request, the KV cache of the shared prompt can directly be used for the new request without recomputation." No tree, just content-addressed blocks, enabled with a single flag.

Both were designed for the datacenter, where the shared prefix is usually a long system prompt sent by thousands of users. A local agent swarm is a smaller version of the same shape, and the memory arithmetic is if anything more favorable. On a server the duplicated prefix mostly costs recompute on a miss. On one card it costs resident VRAM, N copies of it, in the place where VRAM is scarcest.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-21-shared-prefix-kv-cache-radix-tree-budget.svg" alt="A two-part diagram on a dark teal background. On the left, titled 'What the swarm actually shares', a single emerald rounded box labelled 'system prompt plus tool schemas, 4,000 tokens, one copy' sits as a trunk. A branch node on its right edge fans into eight thin grey curves, each ending in a small grey capsule, labelled '8 agents, each about 2,900 unique tokens'. A note reads 'One trunk, eight leaves. A radix tree stores the trunk exactly once.' On the right, titled 'KV budget on a 16 GB card, about 55,000 tokens', two vertical stacked bars are drawn to the same scale against a dashed line marked 'card full, 8.09 GB'. The left bar, 'store it 8 times', is full: a grey lower block labelled 'suffixes' and a tall violet upper block labelled 'prefix x8', totalling 55,200 tokens. The right bar, 'share it once', reaches only halfway: the same grey suffixes, a thin emerald 'prefix x1' block, and a large dashed emerald region labelled '27,800 tokens free', totalling 27,200 tokens. A curved bracket between the two bar tops is labelled 'reclaimed 4.1 GB'." loading="lazy" />
  <figcaption>Eight agents on one RX 9070 XT running Qwen3.5-9B Q4_K_M. Storing a 4,000-token shared prefix once per sequence fills the card; storing it once frees roughly 28,000 tokens of KV budget for more agents or longer context.</figcaption>
</figure>

The thing to notice is the height difference between the two bars. Both hold the same eight agents doing the same work. The only change is whether the trunk is stored once or eight times, and that single decision is the difference between a full card and a half-empty one.

## The saving is (N minus one) prefixes, and it grows fast

The math is not subtle, which is part of why it is easy to leave on the table. Storing a prefix once instead of once per sequence removes `N - 1` copies. For eight agents that is seven prefixes freed, and the size of each is just the prefix length times 147 KB.

| Shared prefix | One copy | Eight copies | Freed by sharing | Share of 8.09 GB budget |
| ---: | ---: | ---: | ---: | ---: |
| 2,000 tok | 0.29 GB | 2.35 GB | 2.06 GB | 25% |
| 3,000 tok | 0.44 GB | 3.53 GB | 3.09 GB | 38% |
| 4,000 tok | 0.59 GB | 4.70 GB | 4.12 GB | 51% |
| 6,000 tok | 0.88 GB | 7.06 GB | 6.17 GB | 76% |

At a 6,000-token preamble the duplicated copies alone would nearly fill the card before a single agent has read a line of real context, which is not a hypothetical size for a modern agent. A system prompt plus a dozen tool schemas expressed as JSON gets into the thousands of tokens quickly, and every agent in the swarm carries all of it.

Read the table the other way and it becomes a scheduling lever. Freeing 4.1 GB at a 4,000-token prefix is about 28,000 KV tokens, which is enough headroom to admit several more agents, or to hand every existing agent a longer working context, without touching the weights or the decode loop. The reclaimed space is the same space the [eight-agent batching win](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) needs to keep sequences resident and decoding in the same step.

## The catch is that the prefix has to be byte-identical

There is one hard requirement, and getting it wrong quietly turns the feature off. A cache hit needs an exact match. Anthropic's [prompt caching documentation](https://docs.claude.com/en/docs/build-with-claude/prompt-caching) states it directly: "Cache hits require 100% identical prompt segments," and the cached prefix follows a strict order of tools, then system, then messages. The shared prefix ends at the first token where two requests differ.

That single rule decides whether a swarm shares anything at all. If the harness stamps the current time, the working directory, a git branch, or a per-agent identifier into the top of the system prompt, then every agent's prefix diverges at that token and the shared trunk collapses to almost nothing. The same documentation warns about exactly this, describing a breakpoint placed on content that changes every request as a common mistake that yields a fresh cache write and never a read.

The fix is ordering discipline, not new machinery. Keep the static block first: tool schemas, then the fixed system instructions. Push everything volatile, meaning timestamps, the working directory, the task description, the conversation, into the suffix. Structured that way, eight agents share the full static trunk and branch only where they genuinely differ, which is what the tree in the diagram is drawing.

The other cost is bookkeeping. A radix tree or a block hash table has to be maintained as agents start, grow, and exit, and shared blocks have to be treated as read-only so that one agent extending its context cannot mutate a prefix another agent is still reading. SGLang handles this with copy-on-write branching in the tree and reports the CPU-side overhead as small. On a single-user box with one process owning the card, that bookkeeping is cheaper than it is in a contended server, because there are fewer tenants churning the tree.

## Where this sits in the local inference stack

Prefix sharing is a memory and prefill optimization, and it is worth being precise about what it does not do. vLLM's documentation notes that automatic prefix caching "only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens." Decode is untouched. An agent that has already prefilled and is now generating pays the same per-token cost whether its prefix was shared or not.

The value on a local card is indirect and still large. The reclaimed VRAM is what lets more agents stay resident, and resident agents are the precondition for the batched decode that carried the throughput in the first place. Sharing the prefix does not make the batch faster, it makes a bigger batch fit. Stack it against the two other levers this series has measured and the picture is coherent: batching shares the one read of the [weight stream](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) across agents, swapping evicts an idle agent's cache to [host DDR5](/blog/2026-03-26-building-a-local-ai-rig/), and prefix sharing refuses to store the same preamble twice. Each attacks the same scarce resource from a different side.

What I owe here is an honest measurement rather than a model. The 4,000-token figure is a stand-in; the real number is whatever a given harness actually sends, and it is trivial to read off by tokenizing the fixed system-and-tools block once. The engineering question that follows is smaller than it looks, because ZINC already pages its KV cache, and paged blocks are what both RadixAttention and vLLM's hash table are built on. The work is to key those blocks by content and refuse to allocate a second copy when the content already exists. That is a few hundred lines against a benefit that starts at a quarter of the card and climbs from there.
