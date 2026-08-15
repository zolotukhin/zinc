---
title: "Batching eight agents saves more decode time than energy on RDNA4"
seoTitle: "Batching an Agent Swarm on RDNA4 Is a 5.6x Throughput Win but Only a 3.7x Energy Win"
date: "2026-07-27"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - agents
  - energy
  - tokens-per-joule
  - continuous-batching
  - decode
  - memory-bandwidth
  - local-llm
  - llm-inference
keywords:
  - energy per token local LLM RDNA4
  - tokens per joule batching agent swarm
  - RX 9070 XT power draw decode inference
  - does batching save energy LLM inference
  - throughput per watt continuous batching
  - weight streaming energy decode token
  - memory bound decode power RDNA4
  - Qwen3.5-9B energy per token RX 9070 XT
  - amortize weight stream energy batch
  - local agent swarm power efficiency
excerpt: "Batching an agent swarm on one RX 9070 XT turns 39.6 tokens per second into 222, a clean 5.6x. It does not turn 3.5 joules per token into 0.63. It turns it into 0.95, a 3.7x energy win, because the extra tokens draw extra compute power the single-agent number never paid. Batching amortizes the one energy cost that dominates a decode token, streaming the weights, and pays cash for the one it adds."
seoDescription: "A decode token on an RX 9070 XT spends most of its energy moving weights across the bus, not doing arithmetic, so batching an agent swarm that shares one weight stream should cut energy per token. It does, but by less than it cuts latency. Modeling Qwen3.5-9B on one 304W RX 9070 XT: throughput climbs 5.6x from one agent to eight while energy per token falls only 3.7x, because the fixed weight-stream and static power amortizes perfectly across the batch while the compute the batch adds costs real watts. This post separates the two energy terms, shows why the throughput ratio always beats the efficiency ratio, and argues energy per token, not tokens per second, is the number a local swarm should tune against."
faqs:
  - question: "Does batching multiple agents actually reduce energy per token?"
    answer: "Yes, but less than it reduces latency. On one RX 9070 XT running Qwen3.5-9B, going from one agent to eight lifts throughput 5.6x, from 39.6 to about 222 tokens per second, while modeled energy per token falls about 3.7x, from roughly 3.5 to 0.95 joules. The gap exists because the largest energy cost of a decode token is streaming the model weights once, which the whole batch shares, while the extra arithmetic each agent adds draws real compute power that scales with the batch."
  - question: "Why does a decode token spend most of its energy on memory, not compute?"
    answer: "Because decode is memory-bandwidth bound. A single decode step reads the entire model from VRAM to produce one token, so its arithmetic intensity is far below the point where the compute units become the bottleneck. On an RX 9070 XT the weight stream is about 10.7 of a 25.2 millisecond decode token, and moving a byte of weights costs far more energy than the handful of multiplies that byte feeds, which is Horowitz's long-standing observation that data movement dominates arithmetic energy."
  - question: "How much power does an RX 9070 XT draw during LLM decode?"
    answer: "Less than its 304W rated board power, because decode leaves most of the compute units idle. This post models a single-agent decode step at about 140W, dominated by a fixed floor of static and memory-subsystem power near 130W, rising to about 210W at eight batched agents as more compute units light up. Compute-bound prefill, by contrast, approaches the full 304W but processes tokens in parallel, so it is far cheaper per token than single-stream decode."
  - question: "Why is the throughput speedup always larger than the energy-per-token speedup?"
    answer: "Because energy per token equals board power divided by throughput, and batching raises both the numerator and the denominator. Throughput climbs because the shared weight stream now feeds many sequences, but power also climbs because each added sequence runs more arithmetic. The fixed floor power amortizes exactly with throughput, so its per-token cost drops by the full throughput ratio, but the compute term grows, diluting the total. The efficiency ratio can at best equal the throughput ratio, and only if added agents drew no extra power."
  - question: "Should a local inference engine optimize for tokens per second or joules per token?"
    answer: "For a battery, a thermally limited enclosure, or a metered power bill, joules per token is the honest target, and it does not track tokens per second. An engine tuned purely for throughput will happily push a batch into the region where compute power rises faster than useful work, spending watts for a shrinking latency gain. Tracking energy per token tells the scheduler when the next agent added to the batch stops paying for its own power."
draft: false
---

Yesterday's post ended on a compute budget: one decode step on an RX 9070 XT hides about two dozen tokens of arithmetic under a single weight stream, and [batching an agent swarm spends that budget](/blog/2026-07-26-batching-eight-agents-leaves-two-speculative-tokens-rdna4/). That framing was about time. There is a second ledger the throughput posts never opened, and it does not balance the same way.

Run the swarm and count joules instead of milliseconds. Eight agents on one card lift decode throughput from 39.6 tokens per second to about 222, the clean [5.6x we measured](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/). If energy tracked throughput, energy per token would fall by the same 5.6x, from roughly 3.5 joules to 0.63. It does not. It falls to about 0.95 joules, a 3.7x win. The swarm saved more time than it saved energy, and the reason is worth the post.

The short version is that a decode token has two very different energy costs, and batching does opposite things to them. One cost is fixed and gets amortized to almost nothing. The other is variable and the batch pays it in full.

## Where a decode token's energy actually goes

A decode step on this card is bound by moving weights, not by doing math. To generate one token, ZINC streams the entire Qwen3.5-9B Q4_K_M model out of VRAM, roughly 5.5 GB, and that stream is [10.7 ms of a 25.2 ms token](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/), the single largest slice. The arithmetic that consumes those weights is trivial by comparison: a batch-of-one decode has an arithmetic intensity far below the point where the RX 9070 XT's [195 FP16 matrix TFLOPs](https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html) could become the bottleneck. The card is a memory pump with a calculator bolted on, and during decode only the pump is working.

That matters for energy because moving a byte costs far more than computing on it. Horowitz's often-cited figure is that an off-chip DRAM access can burn a thousand times the energy of the arithmetic operation it feeds, and that gap has barely moved across process nodes. The recent measurement work in [Where Do the Joules Go?](https://arxiv.org/abs/2601.22076) makes the same point from the top down: inference time and energy are governed by latent metrics like memory traffic and utilization, not by the nominal FLOP count. So the energy of a decode token is dominated by the weight stream and the static power the card draws just to be on, and only a thin slice is the actual matmul.

Call the first part the floor: static leakage plus the memory subsystem running the bus near its 640 GB/s ceiling. It is there whether the batch holds one agent or eight, because the same 5.5 GB streams either way. Call the second part the compute term: the arithmetic power that scales with how many sequences ride the stream. Batching moves these two in opposite directions.

## The floor amortizes, the compute term does not

Here is the whole argument in one model. I hold the floor power fixed, let the compute term grow with the batch, and read energy per token straight off as board power over throughput. The numbers are modeled for Qwen3.5-9B on one RX 9070 XT, anchored to two measured rates, the 39.6 tokens per second single-agent floor and the 5.6x throughput at eight agents. They are not a power-meter capture.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-27-rdna4-swarm-tokens-per-joule-vs-throughput.svg" alt="A two-panel chart on a deep teal-green background titled 'Batching a swarm saves more time than energy'. The top panel plots two curves against batch size on a horizontal axis from one to eight agents. A gold throughput curve rises steeply from 39.6 tokens per second at one agent to 222 at eight agents, labelled 5.6 times. A rose tokens-per-joule curve rises much more gently from 0.28 at one agent to 1.06 at eight, labelled 3.7 times, and the widening vertical gap between the two curves is shaded to show the efficiency win falling behind the throughput win. A dashed horizontal line marks the compute-bound prefill efficiency ceiling at about 3.2 tokens per joule. The bottom panel is a set of four stacked bars, one per batch size of one, two, four and eight agents, each showing joules per token split into two segments: a large emerald 'shared weight stream plus static floor' segment that shrinks steeply from 3.28 joules at one agent to 0.59 at eight, and a small rose 'compute, per agent' segment that grows slightly from 0.25 to 0.36 joules. A callout reads 'The floor is paid once per step and shared, so its per-token energy falls the full 5.6 times. The compute term is paid per agent, so it rises. Their sum falls only 3.7 times.' A footnote gives the model: floor power 130 watts fixed, compute 10 watts per agent, board power 140 watts at one agent rising to 210 at eight, all under the card's 304 watt rated board power." loading="lazy" />
  <figcaption>Modeled for Qwen3.5-9B on one RX 9070 XT. Board power is a fixed 130W floor (static plus memory subsystem) plus 10W per active agent; throughput is anchored to 39.6 tok/s at one agent and 5.6x at eight. Energy per token is board power over throughput. Throughput climbs 5.6x while energy per token falls 3.7x, because the floor's per-token cost amortizes fully but the compute term grows.</figcaption>
</figure>

Read the bottom panel first. At one agent the floor is 130W spread over 39.6 tokens per second, which is 3.28 joules of floor energy in every token. At eight agents that same 130W is spread over 222 tokens per second, so the floor costs 0.59 joules per token. That is a 5.6x drop, exactly the throughput ratio, because a fixed power divided by a rising throughput falls in lockstep with it. This is the amortization the throughput posts were implicitly banking on: the expensive weight stream is paid once per step and shared by everyone in the batch.

The compute term does the opposite. It is 10W per agent, so it grows with the batch, and because throughput scales sublinearly the per-token compute energy actually creeps up, from 0.25 joules at one agent to 0.36 at eight. It is small, but it is the reason the total does not fall the full 5.6x. Add the two segments and energy per token goes from 3.53 joules to 0.95, a 3.7x improvement riding on a 5.6x throughput improvement.

| Live batch | Throughput (tok/s) | Board power (W) | Energy per token (J) | Tokens per joule |
| --- | ---: | ---: | ---: | ---: |
| 1 agent | 39.6 | 140 | 3.54 | 0.28 |
| 2 agents | 74 | 150 | 2.03 | 0.49 |
| 4 agents | 132 | 170 | 1.29 | 0.78 |
| 8 agents | 222 | 210 | 0.95 | 1.06 |

The table says the same thing the picture does, and it exposes the general rule. Energy per token is power over throughput, and batching lifts both. The efficiency ratio equals the throughput ratio only in the impossible case where added agents draw no extra power. Every real watt the batch adds pulls the energy win below the throughput win. That is not an RDNA4 quirk; it is arithmetic, and it is why [throughput per watt](https://arxiv.org/abs/2601.22076) is reported as its own axis rather than assumed to follow tokens per second.

## Decode is chasing prefill's efficiency and cannot quite reach it

There is a ceiling worth naming, because it explains why batching helps at all. Prefill on this card runs at [962 tokens per second](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/) and, being compute-bound, draws close to the full 304W board power. That is about 0.31 joules per token, an order of magnitude cheaper than a single decode token. Prefill is efficient for the same reason decode is not: it keeps the compute units busy, so the model FLOPS utilization that [Pope and colleagues](https://arxiv.org/abs/2211.05102) treat as the master variable is high, and the fixed weight-read is spread across many tokens processed together.

Batching decode is an attempt to import that trick. Each agent added to the batch raises utilization and spreads the weight stream wider, dragging decode's per-token energy down toward prefill's number. At eight agents we are at 0.95 joules, a third of the way there from the single-agent 3.5. But decode stays memory-bound the whole way, so it never reaches prefill's efficiency; the batch would have to grow large enough to make the card compute-bound, and long before that the [ragged-batch and admission costs](/blog/2026-07-19-admitting-a-ninth-agent-stalls-the-other-eight-for-8-3-seconds/) from earlier in this series eat the gain. The efficiency curve flattens for the same reason the throughput curve does.

For grounding, 0.95 joules per token on a consumer card sits in a believable band. The datacenter stacks on the public [ML.ENERGY leaderboard](https://ml.energy/leaderboard) reach a few tenths of a joule per token on 70B-class models by running large batches on H100-class parts, and a single-stream local card landing near 3.5 joules and a small swarm near 1 joule is the same physics at a smaller scale and a lower batch.

## What this changes for a local swarm

The practical consequence is that tokens per second is the wrong number to tune if what you actually care about is a laptop battery, a thermally capped mini-PC, or a power bill. Those cost functions are in joules, and joules per token does not track tokens per second. A scheduler tuned only for throughput will keep widening the batch into the region where the compute term rises faster than useful work, spending watts for a latency gain that is already flattening.

The number that tells you when to stop is the marginal one: how much energy the next agent adds versus how many tokens it buys. Early in the curve that trade is excellent, because a new agent rides a weight stream that is already paid for. Past the knee the new agent mostly buys itself compute power. ZINC's scheduler already tracks the live batch size to size speculation; the same counter, multiplied against a board-power estimate, tells it where energy per token stops falling. That is the point where adding an agent still helps latency but no longer helps efficiency, and on a power-constrained box that is the point that matters.

The framing I want to keep is that batching does two different favors and gets credit for one. It amortizes the weight stream, which is the honest, large win, and it charges you compute power for the privilege, which is the quiet tax. Time and energy are not the same ledger. The swarm that looks 5.6x faster is 3.7x greener, and knowing which of those two numbers your box is actually limited by is the difference between tuning for a benchmark and tuning for the wall socket.
