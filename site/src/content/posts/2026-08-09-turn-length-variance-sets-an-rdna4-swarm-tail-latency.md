---
title: "Turn-length variance, not load, sets an RDNA4 agent swarm's tail latency"
seoTitle: "Turn-Length Variance Controls an RDNA4 Agent Swarm's p99 Latency"
date: "2026-08-09"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - scheduling
  - queueing-theory
  - tail-latency
  - preemption
  - continuous-batching
  - agents
  - local-llm
  - llm-inference
keywords:
  - tail latency local LLM agent swarm
  - turn length variance p99 GPU inference
  - preemptive scheduling LLM serving RDNA4
  - least attained service LLM decode
  - Pollaczek-Khinchine variance waiting time
  - continuous batching head of line blocking
  - SRPT unfairness large jobs inference
  - RX 9070 XT decode slot p99
excerpt: "Yesterday's post found the mean: one RX 9070 XT keeps about twelve agents responsive. But the mean is not the number a user feels. Two swarms with the identical average turn length and the identical GPU load can have p99 latencies that differ by more than 2x, and the only thing that changed was the spread of the turn lengths. Preemption looks like the fix, and it does cut the median, but a simulation shows it relocates the tail onto the long turns instead of removing it."
seoDescription: "On one 16 GB RX 9070 XT running Qwen3.5-9B with eight decode slots at 80 percent utilization, tail latency is set by turn-length variance, not by load. A discrete-event M/G/8 simulation holds the mean turn at 354 tokens and the utilization fixed, then varies only the spread. A low-variance workload (CV 0.15) has a median of 9.8 seconds and a p99 of 20 seconds. A high-variance workload (CV 1.13) with the same mean has a faster median of 6.4 seconds but a p99 of 47 seconds, more than double, because the Pollaczek-Khinchine variance term inflates waiting when a 900-token refactor turn holds a slot behind a 40-token tool check. Adding preemptive least-attained-service scheduling drops the median to 3.6 seconds but pushes p99 to about 100 seconds: it does not delete the tail, it moves it onto the long turns, the classic size-based-scheduling tradeoff Bansal and Harchol-Balter analyzed. The practical levers are to shrink the variance itself by chunking long generations, or to choose deliberately which turns absorb the tail, since production engines like vLLM expose only fcfs and priority scheduling with recompute or swap preemption."
faqs:
  - question: "Why does turn-length variance matter more than average load for tail latency?"
    answer: "Because waiting time in a queue does not depend only on how busy the servers are, it depends on how variable the service times are. The Pollaczek-Khinchine result makes this explicit: mean waiting time scales with one plus the squared coefficient of variation of the service time, so a workload with the same mean and the same utilization but a wider spread of turn lengths waits longer on average, and much longer at the tail. On one RX 9070 XT with eight decode slots at 80 percent utilization, a simulation with a fixed mean turn of 354 tokens shows the 99th-percentile response time rising from 20 seconds at low variance to 47 seconds at high variance, while the median actually drops. The load never changed; only the shape of the turn-length distribution did."
  - question: "Does preemption fix the tail-latency problem in an agent swarm?"
    answer: "No, it relocates it. Preemptive least-attained-service scheduling, which keeps advancing whichever turn has generated the fewest tokens so far, cuts the median sharply because short tool-checks stop waiting behind long refactor turns. In the simulation the median falls from 6.4 seconds to 3.6 seconds. But the long turns now get repeatedly bumped, so their response time explodes, and the overall 99th percentile climbs from 47 seconds to about 100 seconds. This is the size-based-scheduling tradeoff Bansal and Harchol-Balter analyzed for SRPT: favoring short jobs helps the many at the expense of the few large ones. You are not removing the tail, you are choosing who sits in it."
  - question: "What actually reduces tail latency on a single-card local swarm?"
    answer: "Two things. First, shrink the variance at the source by chunking long generations into bounded slices so no single turn monopolizes a decode slot for its entire length, which pulls the coefficient of variation down and, by Pollaczek-Khinchine, pulls the tail in with it. Second, decide deliberately which turns are allowed to absorb the tail: protect the interactive turn a human is watching and let a background refactor be the one that waits, rather than letting arrival order decide. Both are policy choices the engine has to make on purpose, because the default of running turns to completion in arrival order gives you the fat tail for free."
  - question: "What scheduling controls do production inference engines expose for this?"
    answer: "Less than the problem needs. vLLM, for example, exposes a scheduling policy that is either first-come-first-served or priority, and a preemption mode that is either recompute or swap. There is no size-aware or least-attained-service option, so the tail behavior you get is whatever arrival order and your priority assignments produce. An engine that owns its own scheduler, like zinc, can measure the turn-length distribution of the running session and pick a policy against it, but off-the-shelf servers leave you with FCFS plus a priority knob and the fat tail that comes with them."
draft: false
---

[Yesterday's post](/blog/2026-08-08-littles-law-caps-a-responsive-rdna4-agent-swarm-near-twelve/) worked out a mean. One RX 9070 XT with eight decode slots keeps about twelve coding agents feeling responsive before response time starts climbing. That number is real and it is useful for sizing a pool, but it is an average, and an average is not what a user notices. What a user notices is the one turn that hangs.

Here is the uncomfortable part. Two swarms can share the exact same average turn length and put the exact same load on the card, and one of them will feel fine while the other stutters. The difference is not the mean and it is not the utilization. It is the variance of the turn lengths, and it lands almost entirely in the tail.

I ran a discrete-event simulation to pin this down, and the result is sharp enough to change how you'd size and schedule a local swarm. Holding the mean turn at 354 tokens and the utilization at 80 percent, moving from a tight turn-length distribution to a bursty one leaves the median roughly unchanged, or even improves it, while more than doubling the 99th-percentile wait. Then I added the obvious fix, preemption, and it made the tail worse in a way that is worth understanding before you reach for it.

## Same mean, same load, different pain

Start with what an agent turn actually looks like on this card. A [local coding agent reads far more than it generates](/blog/2026-07-23-a-local-coding-agent-reads-back-eighteen-tokens-for-every-one-it-writes/), so its generations swing wildly in length. A turn might be a 40-token tool call, or it might be a 900-token refactor plan. The series has been using a mixed workload with a mean of 354 tokens, which at the card's per-slot rate of about 39 tokens per second is a mean service time of 9.1 seconds.

The mean hides the spread. To make the spread the only variable, I built two workloads with the identical 354-token mean and fed them the identical Poisson arrival stream, tuned so eight decode slots sit at 80 percent utilization. The first workload is tight: nearly every turn lands near 354 tokens, a coefficient of variation of 0.15. The second is bursty: 30 percent long turns around 900 tokens and 70 percent short turns around 120 tokens, which works out to the same mean but a coefficient of variation of 1.13. Same average work, same load, same card. Only the shape differs.

The scheduling model matches how [continuous batching](/blog/2026-07-30-static-batching-drains-rdna4-swarm-throughput/) actually behaves. A turn that gets a decode slot holds it until the turn finishes. There is no preemption, so a 900-token refactor occupies its slot for about 23 seconds, and any turn that arrives while all eight slots are busy waits in an admission queue until one frees. That is where the variance does its damage.

## The tail moves, the median barely does

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-08-09-rdna4-swarm-tail-latency-variance.svg" alt="A survival-curve chart on a dark plum background titled 'Turn-length variance sets the tail; preemption just moves it onto the long turns'. The subtitle notes one RX 9070 XT with eight decode slots at about 39 tokens per second per slot, 80 percent utilization, mean turn 354 tokens, and that the survival curve shows the share of turns slower than x. The horizontal axis is per-turn response time in seconds from 0 to 80. The vertical axis is the share of turns slower than that time, on a logarithmic scale from 100 percent down to 0.1 percent, with dashed guides at the 50 percent median line and the 1 percent p99 line. A cyan curve, the low-variance workload with coefficient of variation 0.15 and no preemption, drops steeply and crosses the p99 line at about 20 seconds; its median is 9.8 seconds. A gold curve, the high-variance workload with coefficient of variation 1.13 and no preemption, starts lower for a faster median of 6.4 seconds but has a much fatter tail, crossing the p99 line at about 47 seconds. A coral curve, the same high-variance workload with preemptive least-attained-service scheduling, has the fastest median of 3.6 seconds but the fattest tail of all, still above the 1 percent line at 80 seconds with a p99 near 100 seconds and still climbing. The curves cross each other, showing that favoring short turns lowers the median while raising the tail." loading="lazy" />
  <figcaption>Survival curves for per-turn response time on one RX 9070 XT, from a discrete-event M/G/8 simulation with a fixed 354-token mean turn and 80 percent utilization. The cyan and gold curves change only the turn-length variance; the gold curve has a faster median but a tail more than twice as long. The coral curve adds preemption and pushes the tail out further still.</figcaption>
</figure>

The chart plots the share of turns slower than a given time, so a curve that hugs the left is good and a long right tail is bad. Read the cyan low-variance curve first. It falls off a cliff: half its turns finish by 9.8 seconds and 99 percent finish by 20 seconds. There is no drama because every turn is about the same size, so the queue behaves predictably.

Now the gold high-variance curve, same mean, same load. Its median is actually faster, 6.4 seconds, because most turns are short 120-token generations that clear quickly. But look at where it crosses the 1 percent line: 47 seconds. The 99th-percentile turn takes more than twice as long as in the low-variance case, and the card is doing the identical amount of average work. A short turn that has the bad luck to arrive when several 900-token refactors are holding slots waits behind all of them, and those unlucky waits pile up in the tail.

| Workload (util 0.80, mean 354 tok) | p50 | p90 | p99 | p99.9 |
| --- | ---: | ---: | ---: | ---: |
| Low variance, CV 0.15, no preemption | 9.8 s | 14.1 s | 19.8 s | 25.4 s |
| High variance, CV 1.13, no preemption | 6.4 s | 28.7 s | 46.5 s | 61.9 s |
| High variance, CV 1.13, preemptive LAS | 3.6 s | 30.8 s | ~100 s | ~216 s |

Read the table across and the pattern is clear. Variance trades median for tail: the high-variance row is faster at p50 and far slower at p99. The utilization column, which is the number most dashboards watch, is 0.80 for every row and tells you none of this.

## Why variance lands in the tail

This is not specific to GPUs or to language models. It is a basic property of queues, and the [Pollaczek-Khinchine formula](https://en.wikipedia.org/wiki/Pollaczek%E2%80%93Khinchine_formula) states it directly. For a queue with Poisson arrivals and a general service-time distribution, the mean waiting time is proportional to one plus the squared coefficient of variation of the service time. Hold the mean service time and the utilization fixed, raise the variance, and the waiting term grows on its own.

The intuition is worth keeping. When service times are nearly constant, a slot frees up on a predictable schedule, so a queued turn waits a bounded amount. When service times are heavy-tailed, most of the busy time is consumed by a few very long turns, and anything that arrives during one of those stretches inherits the full length of the giant ahead of it. The average absorbs this because the long turns are rare, but the tail is made of exactly those unlucky arrivals, so the tail is where the variance shows up. That is why the [previous post](/blog/2026-08-08-littles-law-caps-a-responsive-rdna4-agent-swarm-near-twelve/) targeted 80 percent utilization rather than the exact saturation knee: headroom is the cheapest defense against variance, but it does not erase it.

## Preemption cuts the median and moves the tail

If the problem is a short turn stuck behind a long one, the textbook answer is preemption. Do not let a 900-token refactor monopolize a slot for 23 seconds while 40-token tool checks pile up. Continuous batching can already preempt at iteration granularity, and production engines implement it by evicting a sequence's KV cache, either recomputing it later or [swapping it to host memory](/blog/2026-07-20-swapping-an-idle-agents-kv-cache-beats-recomputing-it-by-177x/), which on this rig was 177 times cheaper than recompute.

The oracle-free way to favor short turns is least-attained-service, or LAS: always advance the turns that have generated the fewest tokens so far, on the theory that a turn which has produced little is probably short. It needs no prediction of a turn's final length, just a running count, so it is implementable. The coral curve shows what it does. The median drops to 3.6 seconds, the best of the three, because short turns almost never wait now.

Then follow the coral curve to the right. It has the fattest tail on the chart, still above the 1 percent line at 80 seconds, with a 99th percentile near 100 seconds. LAS did not remove the tail. It moved it. Every time a long refactor turn starts to make progress, a fresh short turn arrives with fewer attained tokens and bumps it, so the long turns get starved and their completion times balloon. The tail did not vanish; it was transferred from unlucky short turns onto the long turns as a group.

This is the [size-based-scheduling tradeoff that Bansal and Harchol-Balter analyzed for SRPT](https://www.cs.cmu.edu/~harchol/Papers/Sigmetrics01.pdf). Their finding is more nuanced than the folklore, that for heavy-tailed workloads the penalty SRPT imposes on large jobs is smaller than people fear, but the direction is not in dispute: policies that favor short jobs pay for it somewhere in the large ones. LAS is harsher than SRPT because it will preempt a turn that is one token from finishing, so in this mixture it inflates the long-turn tail hard. The lesson is not that preemption is bad. It is that at a fixed utilization you do not get to make the tail disappear, you only get to choose who is standing in it.

## What to actually do about it

Two levers survive that reasoning, and neither is a scheduler trick that promises a free lunch.

The first is to attack the variance itself rather than the queue. If a single turn never runs longer than a bounded slice before yielding, the effective service-time distribution narrows, the coefficient of variation falls, and Pollaczek-Khinchine pulls the tail in for you. That is the same instinct behind [chunked prefill](/blog/2026-08-04-chunked-prefill-keeps-an-rdna4-swarm-decoding/), applied to decode: cap how long any one turn holds a slot uninterrupted, round-robin the long turns against each other, and a 900-token refactor stops being a 23-second wall for everyone behind it. You pay a little scheduling overhead and some cache churn, and you buy a shorter tail without starving anyone, because the cap is length-blind.

The second is to decide, on purpose, which turns are allowed to sit in the tail. The [foreground agent a human is watching](/blog/2026-08-07-the-foreground-agent-decodes-seven-times-faster-than-you-read/) should never be the one starved, and a background refactor is a fine thing to make wait. That is a priority decision, and it is the one knob production engines do expose. vLLM, for instance, lets you pick a [scheduling policy of fcfs or priority and a preemption mode of recompute or swap](https://docs.vllm.ai/en/latest/serving/engine_args.html), and nothing size-aware in between. With only those controls, the tail you get is whatever arrival order and your priority assignments hand you.

An engine that owns its own submission path can do better, because it can measure the turn-length distribution of the running session and pick a policy against it instead of guessing. The number to watch is not utilization, which stayed at 0.80 through every case above and told you nothing. It is the coefficient of variation of the turns, because that is the number the tail is actually made of. Size the pool to the mean, as the last post argued, then schedule to the variance, because the mean is the number you report and the tail is the number your users feel.
