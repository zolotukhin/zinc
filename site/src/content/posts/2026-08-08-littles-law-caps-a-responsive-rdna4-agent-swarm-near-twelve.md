---
title: "Little's Law caps a responsive RDNA4 agent swarm near twelve"
seoTitle: "How Many Agents One RDNA4 Card Keeps Responsive: The Response-Time Law"
date: "2026-08-08"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - scheduling
  - queueing-theory
  - littles-law
  - concurrency
  - continuous-batching
  - agents
  - local-llm
  - llm-inference
keywords:
  - how many agents one GPU can run
  - Little's Law LLM serving
  - interactive response time law agent swarm
  - decode slots vs concurrent agents RDNA4
  - Kingman formula LLM latency
  - agent concurrency knee RX 9070 XT
  - saturation point local LLM inference
  - think time between agent turns GPU
excerpt: "The KV budget sets how many agents decode at once on one RX 9070 XT: about eight. That is not the same number as how many agents the card can keep responsive. Little's Law says the second number is larger, because agents spend real time between decode turns running tools and reading files, and while they do their slot is free. The card that decodes eight concurrently keeps roughly twelve agents feeling instant, and the thirteenth is where response time starts climbing instead of throughput."
seoDescription: "On one 16 GB RX 9070 XT running Qwen3.5-9B, the KV cache caps concurrent decode at about eight slots and the card sustains 311 tokens per second. But the number of agents it can keep responsive is set by Little's Law, not the slot count. Each agent loops: decode a turn of about 354 tokens in roughly 9.1 seconds, then spend a few seconds running a tool or reading files before it needs the GPU again. That think time frees the slot, so the interactive response-time law R = N/X - Z puts the saturation knee at about 8 x (1 + Z/S), near twelve agents for a 4-second think time. Below the knee, adding an agent buys throughput; above it, throughput is pinned at 311 tokens per second and each new agent only adds about 1.1 seconds of wait per turn. Kingman's formula explains why the practical target is about 80 percent of that knee: bursty, high-variance agent turns inflate waiting long before utilization reaches one. The lesson for a single-card swarm is to size the agent pool to the knee and treat the think-time ratio, not the GPU, as the real capacity knob."
faqs:
  - question: "How many agents can one RDNA4 card actually keep responsive?"
    answer: "More than the number it decodes at once. On a 16 GB RX 9070 XT running Qwen3.5-9B the KV budget fits about eight concurrent decode slots, but agents do not decode continuously. Each one decodes a turn, then spends a few seconds running a tool or reading files before it needs the GPU again, and during that think time its slot is free for someone else. The interactive response-time law, a consequence of Little's Law, puts the saturation point at roughly the slot count times one plus the ratio of think time to decode time. For a mean turn of about 9.1 seconds and 4 seconds of think time, that is about eight times 1.44, or near twelve agents. Past twelve, throughput is already at its 311 tokens per second ceiling and each added agent only lengthens the queue."
  - question: "What is the interactive response-time law and how does Little's Law give it?"
    answer: "Little's Law states that the average number of items in a stable system equals the arrival rate times the average time each spends there, L = lambda times W. Applied to a closed loop of N agents that each alternate a GPU turn with some think time Z, it rearranges into the interactive response-time law: R = N divided by system throughput X, minus Z. When N is small the card has idle slots, throughput rises with N, and response time stays at its floor. Once throughput hits the card's ceiling it stops rising, so from that point on the only way the equation can balance as N grows is for response time R to climb. The crossover between those two regimes is the knee."
  - question: "Why is the think time between turns the real capacity knob?"
    answer: "Because it decides how far the responsive agent count exceeds the decode-slot count. If agents never paused, think time Z would be zero and the card could keep only as many agents responsive as it has slots, about eight. Every second an agent spends between turns running a tool, waiting on a compiler, or reading files is a second its slot serves someone else, so a larger think-to-decode ratio stretches the knee further past the slot count. A swarm of agents that read a lot between short generations has a high ratio and packs many agents onto one card; a swarm that generates long outputs with little tool use has a low ratio and saturates near the slot count."
  - question: "Why target about 80 percent of the knee instead of running right at it?"
    answer: "Because the response-time law describes averages, and real agent workloads are bursty. Kingman's formula for a general queue shows mean waiting time scales with utilization over one minus utilization, times a variability factor built from the squared coefficients of variation of arrivals and service. Agent turns vary from a 40-token tool check to a 900-token refactor, and agents tend to hit the GPU in waves after a shared tool returns, so both variability terms are nonzero. That pushes waiting time up well before utilization reaches one, so the practical target is a bit below the knee, around 80 percent, where a burst does not turn into a visible stall."
  - question: "Does this replace fair scheduling or reading-speed-aware scheduling?"
    answer: "No, it sits underneath them. Fair queuing decides which agents get the eight slots when more than eight want them, and reading-speed awareness decides how much of a slot the one agent a human is watching actually needs. The response-time law answers a prior question: how many agents you should admit to the session at all before the card stops feeling instant. Size the pool to the knee first, then let fair queuing and per-role policies distribute the slots within it."
draft: false
---

The last two posts argued about who should get a decode slot. [Fair queuing](/blog/2026-08-06-first-come-scheduling-starves-an-rdna4-swarm-fair-queuing-shares-the-card/) stops one agent from starving the rest, and [reading-speed awareness](/blog/2026-08-07-the-foreground-agent-decodes-seven-times-faster-than-you-read/) says the agent a human is watching needs only a thin slice of one. Both took the number of agents as fixed and fought over how to divide the card between them.

There is a question those posts skipped. How many agents should be in the session in the first place? On one RX 9070 XT the KV cache fits about [eight concurrent decode slots](/blog/2026-08-01-reserve-for-max-kv-cache-fits-four-of-eight-agents-on-rdna4/), and it is tempting to read that as the answer: eight agents, one per slot, done. That is the wrong number, and it is wrong in a useful direction. The card can keep more than eight agents feeling instant, and the reason is a sixty-year-old result about queues.

The number that matters is not how many agents decode at once. It is how many can be alive in a coding session before the card stops keeping up with them. Those are different because an agent is not decoding most of the time. It decodes a burst, then goes off to run a test, read a file, or wait on a compiler, and while it does that its slot is free. Counting agents by slots ignores all that idle time, and the idle time is where the extra capacity lives.

## An agent is a loop, not a stream

Start with what one agent actually does. It is not a firehose of tokens. It is a loop: generate a turn of output, hand that output to a tool, wait for the tool, fold the result back into context, then generate the next turn. A [local coding agent reads far more than it writes](/blog/2026-07-23-a-local-coding-agent-reads-back-eighteen-tokens-for-every-one-it-writes/), and every one of those reads is time the agent is not on the GPU.

Put numbers on the loop. A mean agent turn on this card runs about 354 tokens, which is the average of the [mixed workload](/blog/2026-07-30-static-batching-drains-rdna4-swarm-throughput/) this series has been using, everything from a 40-token tool check to a 900-token refactor plan. At the card's per-slot rate of roughly 38.9 tokens per second, that turn takes about 9.1 seconds to decode. Call the time between turns, the tool call and the reading, the think time. Say it is 4 seconds, which is modest for an agent that runs a test or reads a couple of files between generations.

So each agent spends 9.1 seconds needing the GPU and 4 seconds not needing it, a cycle of 13.1 seconds. For a bit less than a third of its life, its decode slot is available to somebody else. That fraction is the whole argument. If you only ever put eight agents on the card, the slot sits idle during every one of those 4-second gaps, and the card is doing less work than it could.

## Little's Law turns the loop into a capacity number

This is exactly the setting John Little formalized in 1961. [Little's Law](https://pubsonline.informs.org/doi/10.1287/opre.9.3.383) says that for any stable system, the average number of items inside it equals the average arrival rate times the average time each item spends there: L equals lambda times W. It is almost embarrassingly general, holding for any arrival pattern and any service discipline as long as the averages are finite, which is why it shows up everywhere from checkout lines to CPU schedulers.

Applied to a closed loop of agents, Little's Law rearranges into a form that performance engineers call the interactive response-time law. If N agents each cycle through a decode turn and a think time Z, and the whole system delivers throughput X turns per second, then the response time an agent waits for its turn is R equals N divided by X, minus Z. That single equation carries the entire story, because X cannot grow forever. It rises as you add agents, right up until the card's decode slots are all busy, and then it flattens at the ceiling this series keeps hitting, 311 tokens per second, or about 0.88 agent-turns per second.

Below that ceiling, adding an agent raises X and R stays pinned at its floor of one turn's decode time. Once X is flat, the equation has only one free variable left, so every new agent has to show up as a bigger R. The crossover, the point where responsiveness stops being free and starts costing wait, sits at roughly the slot count times one plus the think-to-decode ratio. Here that is 8 times 1 plus 4 over 9.1, about 11.5. Round it and the card keeps about twelve agents responsive, not eight.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-08-08-rdna4-swarm-agent-concurrency-knee.svg" alt="A dual-axis line chart on a dark espresso-brown background titled 'Eight decode slots keep about twelve agents responsive, not eight'. A subtitle notes one RX 9070 XT running Qwen3.5-9B with eight decode slots and 311 tokens per second aggregate, a mean agent turn of about 354 tokens taking roughly 9.1 seconds to decode, and about 4 seconds of think time between turns, with the knee at 8 times one plus Z over S, about twelve agents. The horizontal axis is the number of live agents in the session from 0 to 24. The left axis, gold, is aggregate throughput in tokens per second from 0 to 350; a gold line rises linearly at about 27 tokens per second per added agent, then flattens hard at 311 tokens per second. The right axis, terracotta, is mean response time per agent turn in seconds from 0 to 24; a terracotta line stays flat at about 9.1 seconds while slots are free, then bends upward and climbs to about 23 seconds at 24 agents, gaining about 1.1 seconds of wait per added agent. Both lines bend at a green dashed vertical marker at about 11.5 agents labelled 'knee, about 12 agents, 8 slots times one plus Z over S'. The region to the right of the knee is shaded, marking where throughput is pinned and each new agent only adds waiting. A footnote states the interactive response-time law from Little's Law: response time equals agents divided by throughput minus think time, so below the knee the card has idle slots and above it throughput is fixed and the queue grows." loading="lazy" />
  <figcaption>Throughput and per-turn response time as the session grows from one agent to 24, modeled on one RX 9070 XT. Throughput climbs while decode slots are free, then flattens at 311 tokens per second. Response time holds at its 9.1-second floor until the same knee near twelve agents, then rises about 1.1 seconds for every agent added past it.</figcaption>
</figure>

The two curves bend at the same place, and that is the point of the chart. Left of the knee, the gold throughput line is doing the work: each agent you add is worth about 27 more tokens per second, and the terracotta response line does not move, because there is always a free slot waiting. Right of the knee, the gold line is flat. The card is already producing every token it can, so the thirteenth agent does not add throughput. It adds queue, and the terracotta line turns upward and climbs for the rest of the chart.

## What the knee costs on the wrong side

The linear rise past the knee is easy to underrate, so it is worth reading off the exact numbers. Below saturation each agent's turn completes in its bare decode time, about 9.1 seconds, with no waiting. Above saturation the response time grows by the bottleneck's per-turn demand, which on eight slots is 9.1 divided by 8, about 1.1 seconds, for every agent added.

| Live agents | Aggregate throughput | Response time per turn | Regime |
| --- | ---: | ---: | --- |
| 4 | 108 tok/s | 9.1 s | slots idle, card underused |
| 8 | 216 tok/s | 9.1 s | still below the knee |
| 12 | 311 tok/s | ~9.7 s | the knee, card just saturated |
| 16 | 311 tok/s | 14.2 s | throughput pinned, wait growing |
| 24 | 311 tok/s | 23.3 s | every added agent is pure latency |

Read the throughput column down and it stops moving at twelve. Read the response-time column and it does the opposite, more than doubling from the knee to 24 agents while the card produces not one extra token. That is the trap of sizing a swarm by eye. Doubling the agent count from twelve to 24 feels like more parallelism, and on a throughput dashboard nothing looks wrong, but the actual effect is that every agent now waits two and a half times as long for a turn that used to be instant. The parallelism was already spent at the knee.

## Why the practical target sits below the knee

The response-time law describes averages, and averages hide the thing a user actually feels, which is the bad moment. Agent workloads are not smooth. Turn lengths swing across an order of magnitude, and agents tend to arrive at the GPU in waves, several of them finishing a shared tool call at the same instant and all demanding a slot together. That burstiness does not change the average throughput, but it does change the waiting.

[Kingman's formula](https://en.wikipedia.org/wiki/Kingman%27s_formula) is the standard way to see it. For a general single-queue system, mean waiting time is approximately the utilization over one minus utilization, times a variability factor that averages the squared coefficients of variation of the arrival and service processes, times the service time. The utilization term is the part that bites: as the card approaches full, that ratio explodes toward infinity, and any variability at all multiplies the explosion. A workload with bursty arrivals and highly variable turn lengths starts feeling the blowup well before utilization reaches one.

The practical consequence is to leave headroom. Running the card at its exact saturation point means a single burst has no slack to absorb into, and the burst becomes a visible stall. Targeting something like 80 percent of the knee, roughly nine or ten agents rather than twelve here, keeps the average utilization comfortable and gives bursts somewhere to go. The knee is the ceiling; the target is a step below it.

## Where this sits in the stack

None of this replaces the scheduling posts. It sits under them. [Continuous batching](/blog/2026-07-30-static-batching-drains-rdna4-swarm-throughput/) sets the 311 tokens per second ceiling the throughput curve flattens against. [Chunked prefill](/blog/2026-08-04-chunked-prefill-keeps-an-rdna4-swarm-decoding/) keeps a big incoming prompt from freezing the agents already decoding. Fair queuing and reading-speed awareness decide how the eight live slots get divided once more than eight agents want them. The response-time law answers the question that comes before all of those: how many agents to admit to the session at all.

There is a control knob hiding in the derivation, and it is not the GPU. The knee moved past the slot count only because of the think time between turns, the ratio Z over S. An agent that reads a lot between short generations has a high ratio and stretches the knee well past eight; an agent that emits long outputs with little tool use has a low ratio and saturates near the slot count. That means the way to fit more agents on one card is not always a faster kernel or a bigger KV budget. Sometimes it is to shape the workload so agents spend a healthy fraction of each cycle off the GPU, which is exactly what tool-heavy coding agents already do.

Production engines expose almost none of this. vLLM lets you cap concurrency with [max-num-seqs](https://docs.vllm.ai/en/latest/serving/engine_args.html) and pick a scheduling policy, but the number you put there is a guess unless you have measured your agents' think-to-decode ratio, and the default is set for server traffic, not a handful of long-lived local agents. An engine like zinc that owns its own submission path can do better: measure Z and S from the running session, compute the knee, and refuse to over-admit past it. The [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) line of work made the scheduler operate at iteration granularity precisely so it could reason about the batch this finely. On a single consumer card, where there is no autoscaler to hide a bad guess, knowing that eight slots means about twelve agents, and that the twelfth is the last free one, is the difference between a swarm that feels instant and one that quietly queues up behind itself.
