---
title: "First-come scheduling starves an RDNA4 swarm; fair queuing shares the card"
seoTitle: "FCFS vs Fair Queuing (VTC) for a Local Agent Swarm on One RDNA4 Card"
date: "2026-08-06"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - scheduling
  - fairness
  - vtc
  - continuous-batching
  - orca
  - agents
  - local-llm
  - llm-inference
keywords:
  - fair scheduling local LLM agent swarm
  - VTC virtual token counter RDNA4
  - FCFS starvation continuous batching
  - vLLM scheduling policy fcfs priority
  - decode slot contention RX 9070 XT
  - agent starvation single GPU
  - work-conserving fair queuing LLM serving
  - which agent decodes next local inference
excerpt: "Chunked prefill decided how a prompt enters the batch, but not which agent gets a decode slot when more agents want one than the card can hold. Once twelve agents contend for eight slots on one RX 9070 XT, first-come scheduling hands the eight earliest agents a permanent lease and starves the rest, while total throughput looks perfectly healthy. Fair queuing keeps the same 311 tokens per second and spreads them, and on a single card the only thing it costs is the KV swap traffic of the agents it rotates out."
seoDescription: "A continuously batched eight-agent swarm on one 16 GB RX 9070 XT running Qwen3.5-9B fills eight decode slots per iteration. When agents accumulate context, the KV budget caps how many fit at once, so twelve active agents can contend for eight slots. First-come-first-served, the default policy in vLLM and the model Orca's iteration-level scheduling assumes, gives the eight earliest agents a slot they hold until they finish; long-lived coding agents rarely finish, so latecomers starve while the aggregate 311 tok/s looks perfectly healthy. The Virtual Token Counter (VTC) from Sheng et al. at Berkeley proves a 2x tight bound on the service difference between any two backlogged clients while staying work-conserving. Modeled on this card, FCFS gives the busiest agent 1,170 tokens in 30 seconds and the starved agent 45; VTC gives every agent about 26 tok/s with a bounded 60-token gap, at identical aggregate throughput. The only cost of fairness on a single card is the KV-cache swap of each rotated-out agent, which the series already priced. This post argues the scheduler's fairness policy, not its throughput, is the lever that decides whether a local swarm feels alive."
faqs:
  - question: "Why does first-come-first-served scheduling starve agents in a local swarm?"
    answer: "Because a continuously batched engine can only decode a fixed number of sequences at once, set by the KV-cache budget, and FCFS gives the first agents to arrive a slot they keep until their request finishes. Local coding agents run long, folding tool output back into context for minutes at a time, so those slots rarely free up. When more agents are active than slots, the latecomers wait behind agents that never leave. On one RX 9070 XT with eight decode slots and twelve active agents, the four that arrive last can get almost no tokens for tens of seconds while aggregate throughput stays near its ceiling."
  - question: "What is the Virtual Token Counter (VTC) fair scheduler?"
    answer: "VTC, from Ying Sheng and coauthors at UC Berkeley, is a fair-queuing algorithm for LLM serving built on continuous batching. It tracks how many input and output tokens each client has been served, weighted by cost, and at each iteration admits the backlogged client with the fewest tokens served so far. The paper proves a 2x tight upper bound on the service difference between any two continuously backlogged clients, while remaining work-conserving, meaning it never leaves a decode slot idle when someone wants it. On a single-user swarm the clients are your own agents, and VTC is what stops one from monopolizing the card."
  - question: "Does fair scheduling cost throughput on a single GPU?"
    answer: "No, when the policy is work-conserving. Both FCFS and VTC keep all eight decode slots full on every iteration, so the card produces the same roughly 311 tokens per second either way; only the distribution across agents changes. What fairness does cost is preemption: rotating an agent out of a slot means evicting its KV cache and restoring it later. On this card, swapping an idle agent's KV to host memory over PCIe beats recomputing it by a large margin, so the fairness tax is bandwidth on a cold path, not lost decode time."
  - question: "Do local inference engines actually let you choose the scheduling policy?"
    answer: "Partly. vLLM exposes a scheduling-policy flag with two choices, fcfs and priority, defaulting to first-come-first-served, and a separate preemption-mode flag that chooses between recompute and swap. Neither option is per-client fairness in the VTC sense; priority still lets a high-priority stream dominate. Research schedulers like VTC implement true fairness but are not yet standard in production engines. An engine like zinc that owns its own scheduler can implement a fair policy directly, which is the argument this post makes for doing so on a shared consumer card."
draft: false
---

The last two posts were about how a prompt gets into the batch. [Chunked prefill](/blog/2026-08-04-chunked-prefill-keeps-an-rdna4-swarm-decoding/) sliced an incoming prompt so it never froze the agents already decoding, and the [disaggregation post](/blog/2026-08-05-datacenter-disaggregates-prefill-decode-single-rdna4-card-fuses/) explained why a single card fuses prefill and decode instead of splitting them across GPUs. Both took the eight decoding agents as a given and worried about the ninth one arriving.

There is a question underneath that neither post answered. When more agents want to decode than the card can hold at once, which ones actually get to run? Chunked prefill decides how a prompt enters. It says nothing about who keeps a decode slot and who waits for one. That decision is a separate policy, and on a shared consumer card it is the difference between a swarm that feels responsive and one where a background agent quietly eats the whole machine.

The default answer almost everywhere is first come, first served. It is simple, it is what [Orca's iteration-level scheduling](https://www.usenix.org/conference/osdi22/presentation/yu) assumes, and it is the [default policy in vLLM](https://docs.vllm.ai/en/latest/serving/engine_args.html). It is also the answer that lets one busy agent starve the rest while every throughput number on your dashboard stays green.

## The card runs fewer agents than you have

Start with why there is contention at all. A continuously batched engine decodes a fixed number of sequences per iteration, and that number is set by the KV cache, not the compute. On a 16 GB RX 9070 XT running Qwen3.5-9B, the [KV budget is around 55,000 tokens](/blog/2026-07-31-q8-kv-cache-pushes-an-rdna4-swarm-crossover-from-5k-to-10k-tokens/), and once agents accumulate real context that budget caps how many fit at once. The [August 1 post](/blog/2026-08-01-reserve-for-max-kv-cache-fits-four-of-eight-agents-on-rdna4/) showed a conservative reserve-for-max policy fitting only four of eight agents; on-demand allocation gets more in, but the ceiling is still finite.

So the realistic picture is not eight agents in eight slots with room to spare. It is more agents active than the card can decode concurrently. Say twelve agents are alive in a coding session, each holding a few thousand tokens of context, and the card can decode eight at a time. Four of them cannot have a slot this iteration. Somebody has to choose which four sit out, and that choice repeats on every one of the roughly thirty-nine iterations the card runs each second.

First come, first served makes the choice by arrival time, and it makes it sticky. Whichever eight agents grabbed a slot first keep it until their current request finishes. That would be fine if requests were short. They are not. A local coding agent is a long-lived thing that [reads far more than it writes](/blog/2026-07-23-a-local-coding-agent-reads-back-eighteen-tokens-for-every-one-it-writes/), folding file after file back into context and decoding steadily for minutes. Under FCFS, an agent that got a slot at the start of the session can hold it more or less forever, and the four latecomers wait for a lease that never expires.

## What starvation looks like when throughput looks fine

The uncomfortable part is that the machine appears healthy the whole time. The eight resident agents keep all eight slots full, the card keeps producing its [311 tokens per second](/blog/2026-07-30-static-batching-drains-rdna4-swarm-throughput/), and if the only thing you watch is aggregate throughput, nothing is wrong. The unfairness is invisible unless you look per agent.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-08-06-fcfs-vs-fair-queuing-agent-service.svg" alt="Two line-chart panels on a deep indigo background titled 'Who gets the eight decode slots when twelve agents want them'. A subtitle reads that twelve agents contend for eight concurrent decode slots on one RX 9070 XT running Qwen3.5-9B, with a batched step of t(8)=16.9+1.1x8=25.7 milliseconds and about 39 tokens per second per slot, and that each panel plots cumulative tokens served per agent over a 30-second window with a shaded band showing the gap between the best- and worst-served agent. The left panel, labelled 'First come, hold until done', shows an amber line for the busiest agent rising straight to 1,170 tokens at 30 seconds and a teal line for the starved agent staying almost flat at about 45 tokens, with a wide red shaded triangle between them labelled 'service gap 1,125 tokens and still widening'. The right panel, labelled 'Fair queuing (VTC)', shows the amber busiest-agent line and the teal worst-served line climbing together to 780 and 720 tokens respectively, nearly overlapping, with a thin red band labelled 'gap 60 tokens, bounded; VTC proves a 2x cap on service difference'. A footer note states that both policies keep all eight slots full every step, so aggregate throughput is the same 311 tokens per second and only the distribution differs." loading="lazy" />
  <figcaption>Modeled for twelve agents contending for eight decode slots on one RX 9070 XT. Cumulative tokens served per agent over 30 seconds. First come, first served lets the busiest agent run flat out while the starved agent barely moves; the gap between them keeps widening. Fair queuing gives every agent a bounded share of the same total throughput.</figcaption>
</figure>

The left panel is the failure. The busiest agent, one that got a slot early and never yielded it, climbs to 1,170 tokens over thirty seconds. A starved latecomer gets 45. Both agents are equally important to the user, and one of them is running at twenty-six times the other's rate purely because of when it happened to start. The service gap is not a transient. It widens for as long as the resident agents keep decoding, which for coding agents is the whole session.

The right panel is what a fair policy does with the identical hardware and the identical eight slots. Every agent, resident or not, gets rotated through the slots so that no one falls too far behind. The busiest agent now reaches 780 instead of 1,170, and the worst-served agent reaches 720 instead of 45. The gap between best and worst is about sixty tokens and, crucially, it stays there. Same card, same total work, distributed so that no agent is invisible.

## Fair queuing has a name and a proof

This is not a new problem, and the fix is not a heuristic. The [Virtual Token Counter](https://arxiv.org/abs/2401.00588), or VTC, from Ying Sheng and coauthors at Berkeley, is a fair scheduler built specifically for the continuous-batching mechanism these posts have been leaning on. It keeps a per-client counter of how much service each one has received, measured as a weighted count of input and output tokens, and on every iteration it admits the backlogged client with the smallest counter. An agent that has been starved has a low counter, so it jumps the queue; an agent that has been hogging the card has a high counter, so it yields.

The paper does something most scheduling work does not: it proves a bound. VTC guarantees a 2x tight upper bound on the difference in service between any two clients that are both continuously backlogged, while staying work-conserving. Those two properties are the whole point. The bound means no agent can run away from the others the way the FCFS busiest agent does. Work-conserving means the fairness is free in throughput terms, because the scheduler never idles a slot that someone could use. That is why both panels of the chart sit under the same 311 tok/s ceiling. Fairness here is not a throughput sacrifice. It is a redistribution of a fixed budget.

It is worth being precise about what the paper targets. VTC was designed for a shared server where different customers send requests at different rates, and its fairness is between those customers. On a single-user box the customers are your own agents, but the math does not care. An interactive agent you are watching and a background indexer you forgot about are two backlogged clients, and without a policy that bounds their service difference, the one you care about can lose to the one you don't.

## The numbers on one card

Here is the same scenario laid out as a table, modeled on the card's measured decode step. Twelve agents, eight slots, the batched step of about 25.7 milliseconds that gives roughly 39 tokens per second per slot and 311 aggregate.

| Policy | Which agents decode | Busiest agent | Worst-served agent | Service gap (30 s) | Aggregate | Added cost |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| First come, first served | first eight to arrive, held | 1,170 tok | 45 tok | 1,125 tok | 311 tok/s | none, four agents frozen |
| Priority by arrival | same eight, reordered | 1,170 tok | 45 tok | 1,125 tok | 311 tok/s | none, same starvation |
| Fair queuing (VTC) | all twelve, rotated | 780 tok | 720 tok | 60 tok | 311 tok/s | KV swap of rotated agents |

Read across the aggregate column and every policy is a tie, which is exactly why throughput dashboards miss this. Read across the service-gap column and the policies could not be more different. FCFS and a naive priority scheme both leave a starved agent at 45 tokens; VTC leaves the worst-off agent at 720. The rightmost column is the only real cost, and it is not throughput.

That cost is preemption. To rotate a resident agent out of a slot so a starved one can run, the engine has to get the resident agent's KV cache out of the way and bring it back later. The series already priced this: [swapping an idle agent's KV cache to host memory beats recomputing it by 177x](/blog/2026-07-20-swapping-an-idle-agents-kv-cache-beats-recomputing-it-by-177x/). So the fairness tax is a few hundred megabytes moving over PCIe on a cold path, not decode cycles on the hot one. On a single card that is a bargain for turning a twenty-six-to-one unfairness into a bounded gap.

## Why a single user should care more, not less

It would be easy to wave this off as a multi-tenant server problem. It is the opposite. A datacenter has rate limits, admission control, and per-customer quotas layered on top of the scheduler precisely because operators expect clients to fight over capacity. The [VTC paper opens](https://arxiv.org/abs/2401.00588) by noting that most major services fall back to crude request rate limits, which under-utilize the hardware and give a poor experience when there is spare capacity. A local swarm has none of that scaffolding. It is one card, your agents, and whatever policy the engine happens to default to, which is almost always FCFS.

And the local workload is the one that triggers starvation hardest. Server traffic is a mix of short and long requests from many users, so any single long request is diluted. A coding swarm is the pathological case: a handful of agents, several of them long-lived, all backlogged, all on one card. That is precisely the setting where FCFS's hold-until-done behavior turns into a permanent lease, and where the agent you are actively watching can be the one that starves because it started a few seconds after the background job.

The pattern from the whole series holds one more time. The [static-versus-continuous batching post](/blog/2026-07-30-static-batching-drains-rdna4-swarm-throughput/) argued the scheduler was the cheapest lever left. Chunked prefill added a second scheduling lever on top. Fair queuing is the third, and it is the one that decides not how fast the swarm runs but whether every agent in it gets to run. Production engines expose enough of the machinery to notice the problem, a [scheduling policy flag and a preemption mode](https://docs.vllm.ai/en/latest/serving/engine_args.html), but not a fair policy in the VTC sense. An engine like zinc that owns its own submission path can. On a shared consumer card, where the person watching is one starved agent away from thinking the whole thing hung, that is the policy worth building.
