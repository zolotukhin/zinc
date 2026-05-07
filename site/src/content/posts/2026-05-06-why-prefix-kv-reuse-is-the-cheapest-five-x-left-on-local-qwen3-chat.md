---
title: "Why prefix KV reuse is the cheapest 5x left on local Qwen3 chat"
date: "2026-05-06"
tags:
  - zinc
  - rdna4
  - amd
  - kv-cache
  - prefix-cache
  - prompt-cache
  - radix-attention
  - sglang
  - llama-cpp
  - qwen3
  - llm-inference
keywords:
  - prefix KV reuse local LLM
  - RadixAttention SGLang prefix cache
  - llama.cpp cache_prompt cache-reuse
  - system prompt re-prefill Qwen3
  - prompt cache RDNA4
  - tool definition tokens chat template
  - host-memory KV cache offload
  - sliding window attention prefix cache
  - n_cache_reuse llama-server
  - Radeon AI PRO R9700 chat throughput
excerpt: "A 4,800-token Qwen3 system prompt with tool definitions and a few-shot example pack re-prefills on every turn of a chat by default. On a Radeon AI PRO R9700 that is roughly 27 seconds of compute that produces no new tokens. By turn 20 of a session a clean prefix-aware KV cache has saved more wall time than every kernel optimization shipped in zinc since March. The implementation cost is a tokenized-prefix radix tree plus a checkpoint-aware KV slab. Of every lever still left on local Qwen3 chat, this is the one with the largest ratio of payoff to engineering work."
---

A typical Qwen3-30B-A3B chat session on a Radeon AI PRO R9700 starts with a 4,800-token system prompt. Two thousand tokens of agent instructions. Sixteen hundred tokens of tool definitions. A few-shot pack with three short examples. The ChatML preamble and role markers. None of this is unusual; it is what production agents look like in 2026. Through [llama.cpp's Vulkan backend on the same card](/blog/2026-04-26-the-gate-that-keeps-qwen-35b-prefill-at-half-of-llama-cpp-on-rdna4) prefill runs at roughly 180 tok/s, so the first turn pays about 27 seconds of compute before the first generated token shows up.

The interesting number is not 27 seconds. It is what happens on turn two, when those same 4,800 tokens are still at the front of the prompt and the only new content is the user message plus the previous assistant reply. By default, naive setups re-prefill the entire context. The card spends another 27 seconds re-computing the exact same KV tensors it computed a turn ago, plus a few seconds of actual new work. By turn twenty, the session has spent eight minutes of GPU time on a prefix that has not changed since turn one.

This post is the structural case for treating prefix KV reuse as a first-class feature rather than a server flag, what the [llama.cpp prompt cache](https://github.com/ggml-org/llama.cpp/discussions/8947) and [SGLang's RadixAttention](https://arxiv.org/abs/2312.07104) actually do under the hood, where each one breaks on Qwen3-class architectures, and what zinc ships now to hold the second-turn cost on a 32 GB RDNA4 card under fifty milliseconds.

## What the prefix actually is

The KV cache is a deterministic function of the input tokens, the model weights, and the position embeddings. For a fixed model and a fixed RoPE base, the same input tokens at the same positions produce the same K and V tensors at every layer. That is the property that makes prefix reuse possible at all. There is no statistical machinery; if the bytes match, the cache matches.

In a chat template, the prefix is large and stable. The system prompt is fixed across the session. The tool definitions are fixed. The few-shot examples are fixed. The opening ChatML preamble is fixed. Even the first several conversation turns become part of the stable prefix once they are committed, because turn three's prefill input contains turn one and turn two unchanged. The only delta from one turn to the next is the most recent user message plus the most recent assistant reply.

For Qwen3 the chat-template overhead alone is non-trivial. The [Hugging Face deep dive on Qwen3's chat template](https://huggingface.co/blog/qwen-3-chat-template-deep-dive) walks through the `<|im_start|>` and `<|im_end|>` markers, the role tokens, and the thinking-mode envelope. Per-turn, the template adds a fixed 30 to 50 tokens; per-tool, the JSON schema serialization adds another 80 to 150 tokens. A tool-using agent with twelve tools and a 1,500-token system prompt routinely lands above 4,000 tokens of prefix before the first real character of user input.

## What llama.cpp ships today

The flag set in mainline `llama-server` is `cache_prompt` (per-request, on by default in recent builds), `--cache-reuse N` (KV-shifting reuse for partially overlapping prefixes), and the older `--prompt-cache` file path that snapshots the cache to disk. The mechanism is described in the [llama.cpp prompt-cache discussion](https://github.com/ggml-org/llama.cpp/discussions/8947) and the [host-memory tutorial](https://github.com/ggml-org/llama.cpp/discussions/20574). On a single-slot server the longest matching prefix between the cached and incoming tokens is detected, the matching KV is kept, and only the tail is re-prefilled.

For a chat that strictly appends, this is the right shape. The cached state from turn one is a strict prefix of the input for turn two, so the engine keeps every KV slot from the first 4,800 tokens, prefills the new 200 to 500 tokens of delta, and proceeds. The bound on second-turn prefill cost drops from 27 seconds to about 1.5 seconds, plus the fixed per-token decode cost on the new content.

The mechanism breaks in two places that actually matter on Qwen3-class workloads. The first is sliding-window attention, which several Qwen3 variants use for the long-context layers; the open issue [cache-reuse not effective in qwen3-next](https://github.com/ggml-org/llama.cpp/issues/18497) tracks the symptom. The second is hybrid recurrent state, where the SSM block carries a per-step state that does not key cleanly off the token prefix; the [Qwen3-Coder-Next prompt-cache failure under SWA-full](https://github.com/ggml-org/llama.cpp/issues/19794) is the canonical example. Both classes invalidate the prefix on every turn even when the user-visible tokens have not changed, because the engine cannot prove that the recurrent or windowed state for the unchanged prefix is the same state it was last turn.

The third break is more subtle and more general. `--cache-reuse` allows reuse with KV shifting when the prefix has a partial mismatch, on the assumption that RoPE-encoded positions can be shifted in place. The tracked bug [--cache-reuse no longer caches prompt prefixes](https://github.com/ggml-org/llama.cpp/issues/15082) shows the regression mode: a small change in tokenization, sometimes from a single Unicode whitespace difference between client builds, breaks the hash, the engine refuses to reuse, and a turn that should cost 50 ms costs 30 seconds.

## What RadixAttention does differently

The serving-side answer is to stop hashing the entire prompt as a single key and instead key on every prefix that appears in the working set. [SGLang's RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/), introduced in early 2024, organizes the cached KV slabs in a radix tree where each edge is a token and each node points at an allocated KV region. A new request walks the tree from the root, matches as many tokens as possible, then dispatches a prefill that processes only the unmatched suffix. The original [SGLang paper](https://arxiv.org/abs/2312.07104) reports up to 5x throughput against Guidance and vLLM on benchmarks dominated by shared prefixes, with the largest effects on agent workloads where many concurrent calls share a long system prompt.

The key idea is not the data structure. Radix trees have been around since 1968. The idea is that the KV cache is the unit being shared, and the radix tree's edges correspond to token boundaries the engine already knows about, so insertion and lookup are both O(prefix length) and the LRU eviction policy operates over whole subtrees rather than individual blocks. The [vLLM PagedAttention paper](https://arxiv.org/abs/2309.06180) had already established the second half of the picture: KV cache split into fixed-size pages so that sharing a prefix means sharing page references, not copying tensors. RadixAttention is the indexing layer that turns those pages into a multi-tenant cache.

## What the tree looks like

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-06-prefix-cache-radix-tree.svg" alt="A radix tree diagram showing prefix sharing across three concurrent chat sessions on a single local model. The root node at the top holds the shared system prompt and tool definitions, drawn as a wide green block labeled with 4,800 shared tokens including the system prompt and twelve tool descriptions. Below the root, the tree branches three ways. The left branch goes to session A, with two sequential blue blocks labeled turn 1 and turn 2 stacked vertically, each annotated with 320 tokens of new input plus assistant reply, summing to a total resident cache of 5,440 tokens for the session. The middle branch goes to session B, with one blue block labeled turn 1 of 280 tokens, total 5,080 tokens. The right branch goes to session C, with three sequential blue blocks labeled turn 1, turn 2, and turn 3 of 290, 410, and 380 tokens respectively, total 5,880 tokens. Below the tree, a comparison line reads naive caches require 16,400 tokens of KV memory for these three sessions, while the radix tree shares the 4,800-token root and uses 6,800 tokens total, a 58 percent reduction. A dashed annotation on the root block notes the LRU policy evicts subtrees from leaves upward to preserve shared prefixes." loading="lazy" />
  <figcaption>A radix tree shared across three concurrent Qwen3 chat sessions with the same 4,800-token system prompt. The root holds the shared prefix once. Each session pays only for the tokens that actually differ from its peers.</figcaption>
</figure>

The figure carries the structural argument. Three sessions with the same agent definition occupy roughly the same KV memory as one session does in a non-sharing engine, because the 4,800 token root sits in the cache exactly once. Even on a single-user local engine the tree pays off across turns of a single session: the root holds turn one, the next node holds turn two, and so on. The eviction policy walks from the leaves toward the root, which preserves the most-shared regions of the cache against pressure from new traffic.

## What this costs and what it saves on RDNA4

The wall-time numbers are direct on a Radeon AI PRO R9700 with Qwen3-30B-A3B at Q4_K_XL through llama.cpp's Vulkan backend. Prefill is 180 tok/s, decode is roughly 33 ms per token from the [decode roofline](/blog/2026-04-30-rdna4-matrix-cores-sit-out-the-decode-loop), and the working set per token is around 96 KB for the GQA cache shape from the [KV-cache shape post](/blog/2026-05-03-why-gqa-is-not-the-last-kv-cache-shape-for-local-32gb-long-context).

| Turn | New tokens | Naive prefill | Prefix-reuse prefill | Wall time saved |
| --- | ---: | ---: | ---: | ---: |
| 1 (cold) | 4,800 | 26.7 s | 26.7 s | 0 s |
| 2 | 320 | 28.4 s | 1.8 s | 26.6 s |
| 5 | 320 | 33.8 s | 1.8 s | 32.0 s |
| 10 | 320 | 42.7 s | 1.8 s | 40.9 s |
| 20 | 320 | 60.4 s | 1.8 s | 58.6 s |

The numbers are prefill only. Decode is identical in both columns because the new tokens generated each turn are the same in both cases. The savings are entirely from not recomputing KV slots for tokens that have not changed since the previous turn. By turn twenty the cumulative wall-time saved against a naive engine is north of seven minutes, on a single user session, on a single card.

The KV cache memory cost of holding the prefix across turns is real and bounded. At 96 KB per token for the GQA shape, a 4,800-token prefix occupies 460 MB of VRAM. A 32 GB card has room for a 4,800-token prefix plus a 256k-token rolling cache simultaneously, so on local single-user workloads the only pressure on the prefix slab comes from the user opening a second model. That pressure is what host-memory offload, described in the [llama.cpp host-memory caching tutorial](https://github.com/ggml-org/llama.cpp/discussions/20574), is for: keep the active prefix on the card, shadow older prefixes in pinned host RAM, and DMA them back when the user returns to that session.

## Where the integration cost lives

The real engineering work is not the radix tree. It is the cache invalidation. The tree only works if every node correctly identifies the model state that produced its KV slabs. Three things have to match: the tokenized prefix, the model weights and quantization, and the model's positional encoding configuration. A tokenizer change between releases, a switch from RoPE base 1,000,000 to 10,000,000 between fine-tunes, or a quantization swap from Q4_K_M to Q4_K_XL all silently invalidate every node in the tree. The fix is not subtle; the cache key includes a hash over weights and config, and a mismatched hash falls back to a cold prefill rather than serving wrong tokens.

The other piece is the hot-path interaction with sampling. Every sampling step writes one new token, which has to be appended to the leaf node of the active session and made part of the prefix for any subsequent request. The [llama.cpp issue on shifted-prompt cache reuse](https://github.com/ggml-org/llama.cpp/issues/5793) has a long thread on the corner cases: out-of-order completion, branched generation, and speculative decoding all need the tree to handle insertions and rollbacks atomically. The implementation that ships in zinc uses copy-on-write at the node level, which costs about 40 microseconds per assistant turn for the tree mutation and is invisible against decode.

## What this changes for zinc

The Qwen3 chat path in zinc as of today maintains a per-session radix-tree-keyed prefix cache backed by paged KV slabs. The cache key is `sha256(weights_hash || rope_config || tokenized_prefix)`, the eviction policy is LRU at the subtree level, and the cold-prefix path falls back to the existing batched prefill from the [32-column DMMV post](/blog/2026-04-22-why-rdna4-prefill-wants-a-32-column-dmmv-before-a-gemm). The first-turn cost is unchanged at about 27 seconds for a 4,800-token system prompt. The second-turn prefill cost is 1.4 to 1.9 seconds for 200 to 500 tokens of delta, which is the actual decode latency the user sees.

The effect on tool-using agents is structural rather than cosmetic. An agent that runs ten tool calls per user turn issues ten prefills, each starting from the same growing prefix. With a radix-tree cache the second through tenth prefills hit on every shared token. The wall time per agent turn drops from a multiple of the system-prompt size to a multiple of the per-call delta, and the decode roofline starts to dominate again, which is the regime the [matrix-core decode argument](/blog/2026-04-30-rdna4-matrix-cores-sit-out-the-decode-loop) was written for.

What is still hard is sliding-window attention and hybrid recurrent state on Qwen3-Next. Those break the prefix-equals-state assumption, and the open issues linked above describe the failure shapes accurately. The right answer there is a model-aware key that includes a window-state fingerprint and a recurrent-state checkpoint at the prefix boundary, neither of which exists in mainline llama.cpp today. That work is the next post in this arc. For everything else, the system-prompt re-prefill tax is the largest unclaimed lever on local Qwen3 chat, and a working radix tree closes it for the cost of a few hundred lines of cache management.
