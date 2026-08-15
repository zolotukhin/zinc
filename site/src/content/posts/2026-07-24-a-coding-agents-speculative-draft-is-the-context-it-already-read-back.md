---
title: "A coding agent's speculative draft is the context it already read back"
seoTitle: "Prompt Lookup Decoding Beats a Draft Model for Local Coding Agents on RDNA4"
date: "2026-07-24"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - agents
  - speculative-decoding
  - prompt-lookup
  - decode
  - kv-cache
  - tool-calling
  - local-llm
  - llm-inference
keywords:
  - prompt lookup decoding local LLM
  - n-gram speculative decoding coding agent
  - speculative decoding RDNA4 RX 9070 XT
  - draft model vs prompt lookup
  - input grounded generation copy from context
  - coding agent decode speedup
  - Qwen3.5-9B speculative decode local
  - acceptance rate speculative decoding formula
  - ZINC prompt lookup n-gram
  - agent tool observation reuse decode
excerpt: "A local coding agent reads back far more than it writes, and yesterday that looked like pure cost. It is also a large, free draft model: most of what the agent generates next is already sitting in the file it just read, so n-gram prompt lookup can speculate those tokens for nothing and let a single forward pass confirm a whole block."
seoDescription: "A coding agent copies much of its output from the tool observations it just read, which makes prompt lookup decoding, n-gram speculation with no draft model, a real decode win on a single RX 9070 XT. Because RDNA4 decode is dominated by weight streaming and per-step launch overhead, one forward pass can confirm a block of candidate tokens for about the price of generating one, and the draft costs only a CPU string match. This is why prompt lookup nets out where a draft model did not on Qwen3.5's cheap A3B decode, with modeled per-block speedups of 2x to 6x on copy-heavy spans and a measured 2.4x average on input-grounded tasks."
faqs:
  - question: "What is prompt lookup decoding?"
    answer: "It is speculative decoding with the draft model replaced by a string match. Instead of running a small model to guess the next few tokens, you take the last few tokens the model produced, search the existing context for where that same n-gram appeared before, and propose the tokens that followed it as the candidate block. The full model then verifies the whole block in one forward pass. It is lossless: the output is identical to normal decoding, because any candidate the model would not have produced is rejected. Apoorv Saxena's implementation reports 2x to 4x speedups on input-grounded tasks and is built into Hugging Face transformers and vLLM."
  - question: "Why does prompt lookup work for coding agents specifically?"
    answer: "Because a coding agent's output is heavily copied from its input. When it applies an edit it re-emits lines from the file it just read. When it fixes a test it quotes the identifier or the path from the traceback. When it explains a diff it restates code that is already in context. All of those are exact n-gram matches against the observation the agent read back, so the lookup finds long correct continuations and the model confirms them in a single pass instead of one token at a time."
  - question: "Didn't earlier posts say speculative decoding does not net out on Qwen3.5?"
    answer: "For a draft model, yes. A separate draft model has to run its own forward passes, and on an MoE like Qwen3.5's A3B configuration the target decode is already cheap because only about 3B parameters are active per token, so the draft's relative cost is high and the math does not close. Prompt lookup removes that cost entirely. The draft is a CPU-side string search over the context, not a model, so there is no forward pass to pay for and even modest acceptance is a net win."
  - question: "How much faster is it on one RX 9070 XT?"
    answer: "It depends on how much of the output is copied. The per-block arithmetic from Leviathan et al. gives the expected tokens accepted per forward pass as (1 minus alpha to the power gamma+1) divided by (1 minus alpha), where alpha is the acceptance rate and gamma is the candidate length. On copy-heavy spans where alpha is around 0.7 to 0.9, a gamma of 8 yields roughly 3x to 6x on those spans. Over a whole agent turn the realized speedup is a blend, since novel reasoning tokens do not match, which lines up with the measured 2.4x average on input-grounded tasks."
  - question: "Does prompt lookup help or hurt a local agent swarm?"
    answer: "It helps, because it reduces the number of forward passes needed to produce a given number of tokens, and forward passes are the scarce resource when eight agents share one card. Every block that gets confirmed in one pass is several decode steps the swarm did not have to schedule. The main cost is that a verification pass processes gamma+1 positions instead of one, which is slightly more compute, but decode on RDNA4 is bandwidth and launch bound, not compute bound, so that extra work is close to free."
draft: false
---

Yesterday's post ended on an uncomfortable ratio. A local coding agent [reads back about eighteen tokens for every one it writes](/blog/2026-07-23-a-local-coding-agent-reads-back-eighteen-tokens-for-every-one-it-writes/), because every tool call hands it a file, a grep result, or a test log that is far larger than the thought and the tool call the model produced. That read-back is prefill cost, KV pressure, and wall clock, and the whole framing was that it is a tax you have to manage.

Here is the other half of that observation. The agent is about to copy a lot of that context straight back out. When it applies an edit, it re-emits lines from the file it just read. When it fixes a failing test, it quotes the identifier and the path from the traceback it just ingested. When it writes a commit message or explains a change, it restates code that is already sitting in the context window. The bloated context is not only a cost. It is a large, exact, already-resident draft model, and on a single local card that changes the decode math.

The technique that exploits this is old and boring in the best way. It is speculative decoding with the draft model thrown out.

## The draft model a local agent already has

Regular [speculative decoding](https://arxiv.org/abs/2211.17192) pairs a big target model with a small draft model. The draft guesses the next few tokens cheaply, the target verifies all of them in one forward pass, and because the target's forward pass produces logits for every position at once, you either confirm the whole guessed block or take the tokens up to the first disagreement. It is lossless. The output is exactly what the target would have produced alone. Hugging Face's [assisted generation](https://huggingface.co/blog/assisted-generation) write-up is the clearest walk through of why a single verification pass can stand in for several serial decode steps.

The catch, and the reason [a draft model does not net out on Qwen3.5's A3B decode](/blog/2026-05-25-speculative-decoding-on-qwen3-a3b-loses-even-at-100-percent-draft-acceptance/), is that the draft model is not free. It runs its own forward passes, and when the target is already a mixture-of-experts model with only about 3B active parameters per token, the target decode is cheap enough that the draft's overhead eats most of the gain. [Even at perfect acceptance the arithmetic barely closes](/blog/2026-04-28-why-speculative-decoding-does-not-net-out-on-qwen-35b-a3b/), because you are paying for two models to move one token forward.

Prompt lookup decoding deletes that cost. Instead of a draft model, [Apoorv Saxena's method](https://github.com/apoorvumang/prompt-lookup-decoding) uses a string match: take the last few tokens the model emitted, search the existing context for an earlier place that same n-gram appeared, and propose whatever followed it as the candidate block. There is no model to run. The draft is a sliding-window comparison over token ids, which is CPU work measured in microseconds against a decode step measured in tens of milliseconds. The method is in Hugging Face transformers as `prompt_lookup_num_tokens` and in vLLM as an n-gram speculator, and its own benchmarks report a consistent 2x to 4x on input-grounded tasks with no change to the output.

## Why the verify pass is nearly free on RDNA4

Prompt lookup only pays off if verifying a block is close to the price of generating one token. On RDNA4 it is, and the reason is the same one that runs through the last two weeks of posts. A decode step on one RX 9070 XT running Qwen3.5-9B lands about [39.6 tokens per second](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/), and that time is not spent on arithmetic. It goes to streaming the weights once and to the fixed per-step overhead of launching the decode: the [command submission](/blog/2026-07-16-command-buffer-reuse-is-rdna4s-version-of-the-cuda-graph-decode-win/) and the [barrier between kernels](/blog/2026-07-17-decode-kernel-fusion-on-rdna4-deletes-the-barrier-not-the-launch/) that a single token has to pay in full.

A verification pass over eight candidate positions pays that same weight stream and that same launch overhead once, for all eight. It is a slightly larger batch of work, but decode is bandwidth and launch bound, not compute bound, so processing eight positions instead of one is close to the same wall clock. That is the whole trick: one forward pass, priced like one decode step, can confirm a block of tokens the string matcher pulled straight out of the context.

| One verification step, Qwen3.5-9B on RX 9070 XT | Count | Cost |
| --- | ---: | --- |
| Candidate tokens proposed by string match | 8 | ~microseconds, CPU |
| Model forward pass to confirm them | 1 | ~one decode step |
| Tokens accepted when the edit echoes the file | 6 | emitted for the price of one |
| Serial decode steps avoided | 5 | not scheduled at all |

Read the last two rows together. When the agent is re-emitting a line it just read, the matcher proposes eight tokens, the model confirms six of them in a single pass, and five decode steps that would otherwise have happened one at a time simply never run. The tokens the agent copies are the ones that turn its own bloated context into throughput.

## Acceptance rate is the whole story

How much this helps reduces to one number: the acceptance rate, the fraction of proposed tokens the model actually confirms. [Leviathan, Kalman, and Matias](https://arxiv.org/abs/2211.17192) give the expected tokens accepted per forward pass as a clean function of the acceptance rate `α` and the candidate block length `γ`, which is `(1 - α^(γ+1)) / (1 - α)`. Because the draft costs nothing here, that expected value is also the effective decode speedup.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-24-prompt-lookup-acceptance-speedup-curve.svg" alt="A line chart on a deep indigo background titled 'Prompt lookup speculation: acceptance rate sets the decode speedup'. The horizontal axis is acceptance rate alpha from 0.00 to 0.95, labelled 'fraction of drafted tokens the model confirms'. The vertical axis is effective decode speedup from 0x to 10x. Three curves rise from the bottom left: a cyan curve for continuation length gamma equals 4, a violet curve for gamma equals 8, and a lime curve for gamma equals 16. All three start near 1x at low acceptance and curve upward, with the gamma equals 16 curve rising fastest, reaching about 10x at alpha 0.95, while gamma equals 4 flattens near 4.5x. A dashed horizontal line marks 1x, no speedup. A soft amber vertical band between alpha 0.7 and 0.9 is labelled 'copy-heavy coding spans'. A white marker at about alpha 0.62 and 2.4x is labelled 'prompt lookup, measured, about 2.4x average on input-grounded tasks'. A footnote reads that a draft-model scheme subtracts the draft model's own forward passes while prompt lookup does not, so its whole curve is the net win." loading="lazy" />
  <figcaption>Effective decode speedup as a function of acceptance rate, from the speculative-decoding expectation E = (1 − α^(γ+1)) / (1 − α). Longer candidate blocks help more, but only where acceptance is high. The amber band is where copy-heavy coding spans sit; the white marker is prompt lookup's measured 2.4x average across input-grounded tasks.</figcaption>
</figure>

Two things to notice. First, the curves are almost flat until acceptance climbs past about 0.5, then they turn sharply upward, so this is a technique that does nothing on unpredictable text and a lot on predictable text. Second, a longer candidate block only pays when acceptance is high; at `α` = 0.9 a block of 16 is worth about 8x, but at `α` = 0.6 it is barely better than a block of 4. That is why the parameter matters and why the sensible default block length is short.

The reason coding agents land in the amber band is exactly the read-back asymmetry from yesterday. Prompt lookup's own results make the point: on the multi-turn coding split, [the second turn shows a very high gain](https://github.com/apoorvumang/prompt-lookup-decoding) precisely because there is so much code copying, while open-ended roleplay shows almost none because each token is novel. An agent editing a file is the second-turn coding case on every single step.

## Where it stops helping, and what it changes

The honest limit is that only the copied spans get the speedup. When the agent writes genuinely new reasoning, a chain of thought that is not restating anything in context, the matcher finds nothing and decode falls back to one token at a time at the usual 39.6 per second. So the realized speedup over a whole turn is a blend of fast copied spans and normal novel spans, not a flat multiplier, which is why the measured average across input-grounded tasks is about 2.4x rather than the 6x the copy-heavy spans hit in isolation. Prompt lookup does not make the model think faster. It makes the model stop retyping.

There is also an ambiguity the string match glosses over. When the last n-gram appears in several places in a long context, which continuation do you propose? The current implementation takes the most recent match, which is a fine heuristic for chat but leaves value on the table for a coding agent whose context holds many near-identical lines. A lookup function that preferred matches inside the file the agent is currently editing would almost certainly accept more.

For a swarm, the effect points the right way. [Eight agents on one card](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) are bottlenecked on forward passes, since each decode step streams the shared weights. Every block prompt lookup confirms in one pass is several decode steps the scheduler never has to run, which frees exactly the resource the swarm is short on. The extra compute per verification pass is close to free on a bandwidth-bound card, so the trade is decode steps saved against arithmetic that was going to waste anyway.

## What I am building toward

ZINC already keeps every agent's context resident as [KV cache](/blog/2026-07-20-swapping-an-idle-agents-kv-cache-beats-recomputing-it-by-177x/), which means the raw material for prompt lookup is already in memory. What is missing is the index and the loop: an n-gram lookup over each agent's own token stream, a short candidate block proposed from the most useful match rather than the most recent one, and a verification step folded into the decode kernel so the confirmed tokens land without a second launch.

The framing I want to keep is the inversion of yesterday's. The read-back that makes a local coding agent expensive to prefill is the same read-back that makes it cheap to decode, if the engine is willing to look. The agent already told you most of what it is about to say. The only question is whether the runtime bothers to check the context before it generates a token the file already contains.
