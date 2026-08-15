---
title: "Batching eight agents leaves each one two speculative tokens on RDNA4"
seoTitle: "Speculative Decoding and Batching Compete for the Same Compute Slack on RDNA4"
date: "2026-07-26"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - agents
  - speculative-decoding
  - batching
  - continuous-batching
  - token-tree
  - decode
  - kv-cache
  - local-llm
  - llm-inference
keywords:
  - speculative decoding with batching local LLM
  - optimal speculation length batch size RDNA4
  - batched speculative decoding ragged tensor
  - continuous batching agent swarm RX 9070 XT
  - how many speculative tokens per agent batch
  - Qwen3.5-9B speculative decode batch throughput
  - decode compute budget weight stream RDNA4
  - EQSPEC EXSPEC batch speculative decoding
  - vLLM speculative decoding batch size
  - ZINC agent swarm speculation
excerpt: "The free verification budget from the last two posts was a batch-of-one number. Run the whole agent swarm through one decode step and the batch spends that compute slack itself: at eight agents on one RX 9070 XT there is room for about two speculative tokens each, not two dozen. Batching and speculation are not additive. They draw on the same scarce arithmetic under the weight stream, and speculation makes the batch ragged on top of it."
seoDescription: "Speculative decoding is nearly free on a single sequence because a decode step on one RX 9070 XT wastes most of its arithmetic while it streams weights. That slack is a fixed budget, about 24 tokens of compute per step for Qwen3.5-9B, and batching an agent swarm spends it. At eight concurrent agents the batch already uses eight of those tokens, leaving room for roughly two speculative tokens per agent instead of two dozen, which is the batch-size dependence the synergy-of-batching-and-speculation work predicts. Worse, sequences accept different numbers of drafted tokens, so a batched speculative pass goes ragged and pays alignment overhead. This post models the shared budget for a local swarm and argues for adapting speculation length to the current batch, not fixing it."
faqs:
  - question: "Why does speculative decoding get less effective as batch size grows?"
    answer: "Because speculation is only free when the decode step has spare arithmetic, and batching spends that spare arithmetic. A single decode step on a local card streams the model weights once and does very little compute, so extra candidate tokens ride along under the weight stream for almost nothing. Batching more sequences fills that same compute slack with real tokens, one per sequence, so there is less room left for speculative candidates. The synergy-of-batching-and-speculation analysis makes this precise: the optimal speculation length falls as batch size rises."
  - question: "How many speculative tokens can each agent draft in a batch on one RX 9070 XT?"
    answer: "Roughly 24 divided by the number of agents, minus one. One decode step for Qwen3.5-9B on an RX 9070 XT hides about 24 tokens of compute under a single weight stream. A batch of B agents commits B of those tokens, one per agent, leaving 24 minus B to spread across the batch. At one agent that is 23 draft tokens; at eight agents it is 16 shared, about two each; at sixteen agents there is almost nothing left before the pass costs more than one decode step."
  - question: "What is the ragged tensor problem in batched speculative decoding?"
    answer: "When several sequences are speculated together, each one accepts a different number of drafted tokens in a given pass. That desynchronizes their position IDs, attention masks, and KV-cache lengths, so the batch is no longer a clean rectangle. Implementations either pad every sequence to the longest accepted run, wasting the fast sequences' slots, or realign the tensors, which the EQSPEC analysis shows can consume up to 40 percent of the pass. If handled incorrectly it silently corrupts output."
  - question: "Should a local engine fix the speculation length or adapt it?"
    answer: "Adapt it to the current batch. A fixed draft length that is good at batch one is wasteful at batch eight, and a length tuned for a full batch throws away most of the win when only one agent is active. Because a local agent swarm's batch size changes constantly as agents block on tool calls and resume, the engine should scale the per-agent draft length with the live batch so the total candidate count stays under the compute budget."
  - question: "Does batching plus speculation still beat plain batching for a swarm?"
    answer: "Usually yes, but by less than the single-sequence numbers suggest, and only if the ragged-batch overhead is controlled. Batching already lifts aggregate throughput by amortizing the weight stream across agents. Speculation adds a smaller increment on top because the compute slack it needs is mostly gone, and it introduces alignment cost. The right framing is that the two techniques compete for one resource, so the engine should spend that resource on whichever gives more accepted tokens per pass at the current batch size."
draft: false
---

The last two posts sold speculative decoding as nearly free. A [token tree](/blog/2026-07-25-one-rdna4-verify-pass-has-slack-for-two-dozen-speculative-tokens/) rides along under the weight stream a decode step was going to pay anyway, and one verification pass on an RX 9070 XT has slack for about two dozen candidate tokens before it costs more than a single token. That number, two dozen, was the whole argument. It is also a batch-of-one number, and a local agent swarm is never a batch of one.

Run eight agents through the same card and the picture changes before speculation even starts. The eight agents already share one decode step, because [batching amortizes the weight stream](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/): the weights stream across the bus once and every sequence in the batch reads them. That is exactly why batching works. But it is also why the free speculation budget is already half spent by the time any agent drafts a token. The batch and the draft are drawing on the same account.

The account is compute slack, and it is small. A decode step is bound by streaming weights and paying launch overhead, not by arithmetic, so most of the card's matrix throughput sits idle during a token. Speculation spends that idle arithmetic. So does batching. They are not additive, and treating them as if they were is how you tune a draft length that looks great on a single sequence and quietly does nothing for the swarm.

## The budget is fixed, and the batch is inside it

Start from the one measured floor. A decode step for Qwen3.5-9B on one RX 9070 XT lands about [39.6 tokens per second](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/), roughly 25 milliseconds, almost all of it weight streaming and launch cost. Compute on the same card runs at the 962 tokens per second prefill rate, about 1.04 milliseconds per token of work. Divide the decode floor by the per-token compute and you get the budget: about 24 tokens of arithmetic hide under one weight stream before the compute grows past it. Below 24 the extra work is free; above it each token is priced at the prefill rate. That is the roofline the token-tree post ended on.

A batch of B agents lives inside that same 24-token budget. Each agent contributes one committed token per step, so the batch spends B of the 24 before a single speculative candidate is added. Whatever is left, 24 minus B, is the room for speculation, shared across the whole batch.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-26-batched-speculation-budget-rdna4.svg" alt="A horizontal bar chart on a deep oxblood background titled 'Batching the swarm eats the free speculation budget'. Five bars, one per batch size, each spanning a fixed 24-token compute budget on a shared axis from 0 to 24 tokens of compute per decode step, with a dashed vertical 'budget full' line at 24. Each bar is split into an amber 'committed tokens' segment, one token per agent, and a rose 'free to speculate' segment equal to 24 minus the batch size. The one-agent bar has a tiny amber sliver and a long rose region labelled 23 draft tokens per agent. The two-agent bar shows 11 draft tokens per agent. The four-agent bar shows 4 committed and 5 draft tokens per agent. The eight-agent bar shows 8 committed and 2 draft tokens per agent. The sixteen-agent bar is mostly amber, 16 committed, with a short rose region labelled 0.5 draft tokens per agent. A callout reads 'Batch 1: 23 draft tokens each. Batch 8: 2 draft tokens each. The batch and the draft draw on the same 24-token compute slack.' A lower strip titled 'And the drafts you can afford desync the batch' shows eight thin lanes of differing accepted length, each padded with grey to a dashed 'pass width equals slowest agent' line, with a note that realigning position IDs, masks and KV state costs up to 40 percent of the pass. A footnote gives the model: one weight stream sets a 25.2 millisecond floor at 39.6 tokens per second, compute runs at 962 tokens per second, so batch size times one plus draft length must stay under 24 tokens to hide under the stream." loading="lazy" />
  <figcaption>Modeled for Qwen3.5-9B on one RX 9070 XT from two measured rates: 39.6 tok/s decode sets the floor, 962 tok/s prefill sets the compute cost. The 24-token compute budget is fixed; batching B agents commits B of it, so the per-agent draft length that stays free is about (24 − B) / B. The lower strip shows the second cost: sequences accept different numbers of drafts, so the batched pass goes ragged.</figcaption>
</figure>

Read the bars top to bottom. At one agent, 23 of the 24 tokens are free to draft, which is the number the last post spent on a token tree. At two agents each gets 11. At four, five each. At eight agents the committed tokens take a third of the budget and leave 16 to share, about two speculative tokens per agent. At sixteen the batch fills the budget on its own, and there is nothing left before the pass costs more than a plain decode step. The free ride collapses roughly as one over the batch size.

This is not a ZINC quirk. It is the batch-size dependence that Su and colleagues characterized directly in [the synergy of speculative decoding and batching](https://arxiv.org/abs/2310.18813): the optimal speculation length depends on the batch size, and it falls as the batch grows, because larger batches already use the hardware that speculation was exploiting. Their conclusion was to make the speculation length adaptive rather than fixed, and the roofline above is why.

## The batch also goes ragged

Shrinking budget is the first cost. The second is that the tokens you do draft do not come back cleanly. Speculation is only useful because the model verifies several candidates in one pass and accepts a prefix of them, but different sequences accept different amounts. One agent copying a familiar line from its context might confirm all six drafted tokens; another mid-reasoning confirms one and rejects the rest. After a single pass the sequences in the batch have advanced by different distances.

That is the ragged tensor problem, and a [February 2026 study of batched speculative decoding](https://arxiv.org/abs/2510.22876) found it is not a minor inefficiency. Varying accept counts desynchronize position IDs, attention masks, and KV-cache lengths across the batch, and the paper shows that every existing batch speculative implementation it tested violated output equivalence, producing repetition or gibberish when the misalignment was handled wrong. Their corrected algorithm restores equivalence but measures the alignment overhead at up to 40 percent of the pass, growing superlinearly with batch size, before a scheduler that regroups same-length sequences claws some of it back.

For a swarm this is the expensive corner. The whole point of batching agents is that they share the weight stream, but speculation pushes them out of lockstep exactly when you most want them aligned. You can pad every sequence to the longest accepted run, which wastes the fast agents' slots, or you can pay to realign the tensors. Neither is free, and both scale the wrong way with the number of agents. Speculative decoding must stay [lossless](https://arxiv.org/abs/2211.17192), identical in output to plain decoding, so there is no shortcut of just letting the misalignment slide.

## What the budget says to build

The design consequence is that the per-agent draft length is not a constant to tune once. It is a function of the live batch.

| Live batch | Committed tokens | Free to speculate (total) | Sensible per-agent draft |
| --- | ---: | ---: | ---: |
| 1 agent active | 1 | 23 | a small tree, ~8 to 16 deep |
| 4 agents active | 4 | 20 | ~4 to 5 tokens each |
| 8 agents active | 8 | 16 | ~2 tokens each |
| 16 agents active | 16 | 8 | draft off, or shortest branch only |

The table reads as one rule: keep the total candidate count across the batch under the 24-token budget, and let the batch size decide how that total is split. A local swarm makes this easy to get wrong, because its batch size never sits still. Agents block on a file read, a grep, or a test run and drop out of the batch, then resume and rejoin, so the live count swings between one and a dozen within a single task. A draft length fixed for a full batch wastes almost all of the win when only one agent is awake, and a draft length tuned for a single sequence blows the budget and goes ragged the moment the swarm fills up. Production servers face the same swing and default to conservative speculation for this reason; the [vLLM speculative decoding docs](https://docs.vllm.ai/en/latest/features/speculative_decoding/) note that the gains are largest at low request rates, when the batch is small, and taper as concurrency climbs.

So ZINC's scheduler already knows the one number that sets the draft length: how many agents are currently in the decode batch. The speculation stage should read that count each step and scale the per-agent draft so the batch stays under budget, wide when the swarm is quiet and off when it is full. The [token-tree index](/blog/2026-07-25-one-rdna4-verify-pass-has-slack-for-two-dozen-speculative-tokens/) from yesterday still supplies the candidates; the batch size just decides how many of them are worth proposing.

The framing I want to keep is that batching and speculation are two ways to spend one pool of idle arithmetic, not two independent wins to stack. On a bandwidth-bound card the pool is fixed at about two dozen tokens per step, and the batch is the first claim on it. Speculation gets what is left. At eight agents that is two tokens each, which is still worth taking, but it is a tenth of what a single agent gets, and pretending otherwise is how a swarm ends up paying ragged-batch overhead for a speedup that was never in the budget.
