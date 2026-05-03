---
title: "Attention sinks: the four KV tokens local long-context inference cannot evict"
date: "2026-05-02"
tags:
  - zinc
  - rdna4
  - amd
  - kv-cache
  - long-context
  - attention
  - attention-sinks
  - streaming-llm
  - gpt-oss
  - llm-inference
keywords:
  - attention sinks RDNA4 KV cache
  - StreamingLLM local inference
  - first four tokens KV eviction
  - softmax constraint attention sink
  - gpt-oss learned attention sink
  - H2O heavy hitter oracle KV eviction
  - SnapKV local long context
  - Quest query-aware sparsity
  - 128k context 32GB RDNA4
  - attention sink llama.cpp
excerpt: "Every KV eviction scheme for local long-context inference has the same blind spot. Drop the wrong tokens and the model produces garbage; drop the right ones and it stays coherent across a million-token stream. The pivot is not most-recent or heaviest-hitting. It is the first handful of token positions, which the softmax forces every attention head to overweight regardless of content. On a 32 GB RDNA4 card the prefill ceiling and the decode bandwidth wall both end at the same set of four KV slots that cannot be touched."
---

The single result that broke open KV eviction for long-context language models is one a graduate student found by accident in 2023. Guangxuan Xiao at MIT was running a sliding-window attention experiment on Llama-2 and noticed that the moment the window slid past token zero, perplexity collapsed by orders of magnitude. The window itself was fine. The eviction policy was fine. The problem was that token zero was not just another token. The [Efficient Streaming Language Models with Attention Sinks paper](https://arxiv.org/abs/2309.17453) that came out of that experiment is the reason every serious local-inference engine, including ours, treats the first few KV slots differently from every other slot in the cache.

The result has aged into something stranger. OpenAI's [gpt-oss model card](https://cdn.openai.com/pdf/419b6906-9da6-406c-a19d-1bb078ac7637/oai_gpt-oss_model_card.pdf) describes a learned per-head sink as part of the attention layer, with a fixed extra term in the softmax denominator that lets each head "pay no attention to any tokens." A discovery about a quirk of pretrained models is now a deliberate component baked into open-weight architectures. For a local engine running a 128k-context decode on a 32 GB RDNA4 card, that lineage matters in a way that is easy to miss. Whatever KV eviction strategy ships in the engine has to leave those slots alone, and on hybrid MoE-plus-SSM models like the Qwen 35B-A3B family the constraint shows up earlier and harder than the recent prefill-side math suggests.

This post is the structural argument for why a local engine that cares about long-context coherence has only one degree of freedom in its KV eviction policy: the recent window. The sinks are fixed.

## Why the first four tokens are immortal

The mechanism behind the attention sink is uncomfortable in how simple it is. Softmax outputs sum to one. A pretrained transformer is trained against a loss that does not penalize attention-head abstention; the network only sees the post-softmax distribution. When a head has nothing useful to look at on a given step, it cannot output a zero attention vector, because softmax forbids it. It still has to spend its full weight budget somewhere. Empirically, every model trained from scratch settles on the same dumping ground: the first few tokens of the sequence.

[Evan Miller's "Attention Is Off By One" essay](https://www.evanmiller.org/attention-is-off-by-one.html) framed the same observation from the architectural side a year before the StreamingLLM paper. His argument was that the softmax denominator has no slack, and the right fix is to add a constant one to it so heads can downscale to nothing without finding a sacrificial token to dump on. The argument did not change the architecture of the existing pretrained checkpoints, but it explained why those checkpoints behave the way they do, and it predicted that any future architecture that fixed the denominator would no longer need explicit sink tokens. The gpt-oss team appears to have taken something close to that path: a learned bias inside the denominator does the same job as a content-free sink token, except the bias does not consume cache.

The practical fact, for an engine running an existing pretrained checkpoint without that fix, is that the dumping ground in the cache cannot be evicted. Xiao and his coauthors measured what happens when it is. On Llama-2-7B with a 4k window slid forward past the original prompt, perplexity rises from around six to over a thousand by the time the window has moved past the first sequence position. Adding just four anchor tokens at positions zero through three back into the cache, while continuing to evict the rest with a sliding window, drops perplexity back below seven. Four KV slots out of twenty thousand carry the model.

## What this looks like inside a head

The attention pattern that produces this behavior is not subtle. It is not a long tail. It is a single bright stripe down the leftmost columns of the score matrix.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-02-attention-sink-heatmap.svg" alt="Two side-by-side 32-by-32 heatmaps representing causal attention scores from a single head in a pretrained transformer. The left panel, labeled with attention sinks resident, shows a strong yellow stripe down the first three key columns plus a bright recent-token diagonal band, with dim navy filling the rest. The right panel, labeled sliding window only, has the first three columns zeroed and outlined in red as evicted; the recent-window diagonal band is more uneven and the surrounding pattern is noisier. Beneath the right panel, a small line plot shows perplexity over decoded position: a flat blue line near six for the sinks-resident case, and a red line that spikes past a thousand the moment the window slides past position zero." loading="lazy" />
  <figcaption>Two views of the same attention head. With sinks resident, the model parks unused weight on the first four positions and reads coherent text. Without them, the same head's softmax tries to renormalize across content tokens and the output collapses.</figcaption>
</figure>

The figure is what the math says. The model trained with no preferred zero-attention output learned to spend its surplus on positions that always exist and that never change semantic meaning across a sliding window. Removing them does not free attention budget; it forces the budget onto whatever else is in the window, which is a noisy signal. The same paper shows that the precise identity of the sinks barely matters: replacing the first four tokens with the literal token "the" repeated four times recovers most of the win.

## Why this binds harder than the prefill ceiling

The [prefill plateau on RDNA4](/blog/2026-05-01-why-rdna4-long-prefill-plateaus-on-attention-not-gemm) is a compute ceiling. Above sixteen thousand tokens the attention term overtakes the linear-layer term, and the wall stops being a kernel choice. The eviction problem is an algorithmic problem on the same axis, but it lives inside the decode loop rather than the prefill loop, and it has a tighter shape.

For a local user running a long chat against a Qwen 35B-A3B model at Q4_K_M, the active per-token weight read is roughly two gigabytes and the KV cache reads scale with the resident context. The [16k decode crossover post](/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode) walked through where the KV bandwidth term overtakes the active-weight term. Everything past that crossover is a fight over how few KV bytes the engine can read per decoded token while still producing coherent text. The naive answer is to keep only the most recent N tokens, which is the sliding window. The corrected answer, after the StreamingLLM result, is to keep the first four tokens plus the most recent N. The corrected answer adds about half a kilobyte of cache to a hundred-megabyte budget. It is essentially free.

What changes when the model already has a learned sink, the way gpt-oss-20b and gpt-oss-120b do, is that the constraint moves from "keep the first four tokens resident" to "no special handling needed." The learned bias takes over for the role those tokens used to play. For [gpt-oss-20b on Metal](/blog/2026-04-07-how-we-enabled-openai-gpt-oss-20b-in-zinc-on-metal), this is a quiet simplification of the eviction code path. For Qwen, Llama, Mistral, and the rest of the open-weight family without the architectural fix, the four-token rule is mandatory.

## What the eviction landscape actually looks like

The post-StreamingLLM literature can be split into two camps. The first treats the sink as a fixed prior and adds heuristics for which non-sink tokens to keep. The second tries to drop the sink entirely by changing the architecture. Both are interesting; only the first has shipped in production local engines.

[H2O from Zhang and coauthors at NeurIPS 2023](https://arxiv.org/abs/2306.14048) was the first principled extension. It tracks accumulated attention scores per token and keeps the heaviest hitters along with a small recency window. The headline number is up to twenty-nine times throughput on OPT-class models, but the more useful point for a local engine is the hidden assumption: the heavy-hitter list almost always contains the sink positions for free, because they get the most attention by construction. The scoring policy and the StreamingLLM rule end up landing on the same set in practice.

[SnapKV](https://arxiv.org/abs/2404.14469) is the version of the same idea that picks its compressed cache from the prefill scores rather than maintaining a running tally during decode. The tradeoff is that the compression decision is frozen at end-of-prefill, which costs accuracy on questions that drift far from what the prompt was asking. The benefit is that the eviction logic is essentially a pre-decode pass and adds no per-step overhead, which on a bandwidth-bound RDNA4 decode is the right tradeoff.

[Quest from MIT, ICML 2024](https://arxiv.org/abs/2406.10774) is the query-aware variant. It keeps the full cache but only loads the top-K pages relevant to the current query token. The arithmetic on a 32 GB card looks like this: the active-weight read stays the same, but the KV read drops from "all resident pages" to "top eight pages" or wherever K lands. On a 128k-context decode the bandwidth term collapses by an order of magnitude. The catch is that ranking pages is not free; it requires per-page key min-max metadata and a small dot product against the current query. None of that math is hard. It is a kernel that has not yet been ported to Vulkan, and it is on the same shortlist as the cooperative-matrix attention work the engine roadmap already tracks.

## The eviction tier list, made concrete

The ms-per-decode arithmetic for the four schemes on Qwen 35B-A3B at Q4_K_M with a 64k resident context, treating active weights as the 2 GB-per-step floor and the KV bandwidth as the 644 GB/s of GDDR6 on the [Radeon AI PRO R9700](https://www.amd.com/en/products/graphics/workstations/radeon-ai-pro/ai-9000-series/amd-radeon-ai-pro-r9700.html), gives the following lower bounds. Treat them as the bandwidth-bound floor; they assume a perfect kernel and ignore dispatch overhead.

| Eviction policy | Resident KV reads per step | KV ms | Active-weight ms | Total ms | Output |
| --- | ---: | ---: | ---: | ---: | --- |
| Full cache (no eviction) | 2.0 GB | 3.1 | 3.1 | 6.2 | coherent |
| Sliding window (4k) without sinks | 0.13 GB | 0.20 | 3.1 | 3.3 | collapsed |
| Sliding window (4k) plus four sinks | 0.13 GB | 0.20 | 3.1 | 3.3 | coherent |
| H2O 20% retention plus sinks | 0.40 GB | 0.62 | 3.1 | 3.7 | coherent |
| Quest top-8 page selection | 0.06 GB | 0.10 | 3.1 | 3.2 | coherent |

The table reads in two lines. The middle two rows are the same number of KV bytes, the same wall time, and one of them produces text and the other produces noise. The cost of correctness in the sliding-window case is four cache slots. The Quest row is the closest a kernel-side change can get to the Full cache row in coherence at a fraction of the wall time, but it is also the row that does not exist yet on Vulkan.

## What this means for the engine

The engine-level takeaway is that long-context decode on local RDNA4 has two locked-in invariants and one degree of freedom. The first invariant is that the four sink slots stay resident. The second is that the most-recent-N window stays resident, where N is whatever the user's interaction budget says. The free choice is what to do with everything else: keep some of it, score and rank it, page it on demand, or drop it. The literature offers four variations on that choice, and the bandwidth math says any of them are an order-of-magnitude wall-time improvement over the full cache once the resident context grows past 32k tokens.

What the math does not say, and what the engine has to enforce, is that every one of those policies has to have the sink rule wired in by default. The bug surface is shaped like a single missing if-statement: an eviction policy that scores tokens by attention-weight magnitude across the prompt will keep the sinks for free on most prompts, until the day it does not, and then the model will produce garbage on a chat that has been working for a thousand turns. The architectural fix in gpt-oss is the right long-term answer; until the rest of the open-weight family adopts it, the four-token rule is the law.

The relevant prior work in our own kernels is the [llama.cpp Vulkan flash-attention path](https://github.com/ggml-org/llama.cpp/discussions/3443), which has had the StreamingLLM hook open as a discussion thread since the original paper landed. The discussion is still open. The reason it has not closed is that retrofitting the four-token rule into the existing kernel is straightforward; doing it without breaking the kernels that already assume a contiguous KV layout is not. That is the next port on the eviction side, and it is the one that turns long-context decode on a 32 GB card from a memory pressure problem into a bandwidth problem we already know how to solve.

## What we are watching

Two things on the horizon are worth tracking from a local-inference seat. The first is whether the next round of open-weight long-context models follows gpt-oss into bias-style learned sinks, which would let a local engine drop the four-token special case entirely. The second is whether Quest-style query-aware sparsity gets a clean Vulkan port at the same kernel quality the cooperative-matrix flash-attention path already has, because the bandwidth multiplier is large and the ranking math is simple. Both of those would shift the long-context decode wall on a 32 GB RDNA4 card from "manage a giant resident cache" to "page a small resident set on demand." Until they do, the four KV tokens at the start of every prompt are the only ones we cannot move.
