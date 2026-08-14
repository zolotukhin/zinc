---
title: "A deeper queue cannot fix an oversubscribed RDNA4 agent swarm"
seoTitle: "Admission Control, Not a Longer Queue, for an Oversubscribed RDNA4 Agent Swarm"
date: "2026-08-10"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - scheduling
  - admission-control
  - load-shedding
  - queueing-theory
  - codel
  - agents
  - local-llm
  - llm-inference
keywords:
  - admission control local LLM agent swarm
  - load shedding GPU inference RDNA4
  - oversubscribed single card agent scheduling
  - interactive response time law inference
  - CoDel delay target RFC 8289 admission
  - vLLM max_num_seqs concurrency cap
  - goodput plateau saturation knee decode slots
  - RX 9070 XT twelve agents queue wait
excerpt: "Once a single RX 9070 XT is past its twelve-agent knee, the card is already delivering every token it can, about 0.88 turns per second. Every agent you add after that buys zero throughput and only lengthens the wait before a turn starts. A discrete-event simulation shows the unbounded queue's p99 wait climbing from 9 to 27 seconds between twelve and twenty-four agents while goodput never moves, which means the fix is not a bigger buffer. It is admission control, and the only real question is whether you shed by count or by delay."
seoDescription: "On one 16 GB RX 9070 XT running Qwen3.5-9B with eight decode slots at about 39 tokens per second each, a closed-loop discrete-event simulation sweeps 8 to 24 concurrent coding agents. Goodput saturates near 0.88 turns per second at a twelve-agent knee predicted by the interactive response time law, so past that point an unbounded queue converts each added agent into pure waiting: p99 queue wait rises from 8.8 seconds at twelve agents to 27 seconds at twenty-four while throughput is pinned. A fixed concurrency cap of twelve, the vLLM max_num_seqs model, holds p99 wait near 13 seconds but defers 37 percent of submissions at twenty-four agents because it sheds by count. A CoDel-style delay target of 8 seconds (RFC 8289) sheds on the metric that matters and defers only 23 percent for a wait a few seconds longer. Neither policy creates throughput; both only choose whether overload appears as latency or as deferral, the load-shedding lesson from the Google SRE book."
faqs:
  - question: "Why does adding more agents past the knee not increase throughput on one card?"
    answer: "Because the card is already saturated. With eight decode slots at about 39 tokens per second each and a mean turn of 354 tokens, the bottleneck delivers roughly 0.88 completed turns per second, and no arrival pattern can pull more work through a station than its service rate allows. The interactive response time law, R = N/X - Z, makes the consequence explicit: once throughput X is pinned at its maximum, adding agents N can only raise response time R, because the equation has no other free variable. In the simulation, goodput is flat near 0.88 turns per second from about a dozen agents onward while the p99 queue wait climbs from 8.8 seconds at twelve agents to 27 seconds at twenty-four. The extra agents are not being served faster or slower on average, they are simply standing in a longer line."
  - question: "Does a bigger queue or buffer help an oversubscribed inference server?"
    answer: "No. A deeper queue changes where the backlog sits, not how fast it drains. When offered load exceeds service capacity, the queue length grows without bound in the open case, or the wait grows linearly with the number of clients in the closed case, and a larger buffer just lets more requests accumulate the same unbounded delay before they are served. The Google SRE book states the practice directly: an overloaded backend should accept only what it can process and reject the rest gracefully, because queueing the excess turns a capacity problem into a latency and reliability problem. The fix for an oversubscribed single-card swarm is admission control that bounds the backlog, not a bigger buffer that hides it."
  - question: "What is the difference between a fixed concurrency cap and a CoDel-style delay target for admission control?"
    answer: "A fixed concurrency cap, like vLLM's max_num_seqs, admits new turns only while the number in the system is below a set count and defers the rest. It bounds the wait, but it sheds by count and is blunt: it starts deferring at a fixed occupancy even when the queued turns are short and would clear quickly. A CoDel-style policy from RFC 8289 instead tracks the actual queue sojourn and only sheds once the minimum delay stays above a target for an interval, so it reacts to the metric a user feels rather than a proxy. In the simulation at twenty-four agents, the fixed cap holds p99 wait near 13 seconds but defers 37 percent of submissions, while the 8-second CoDel target defers only 23 percent for a p99 wait a few seconds longer. One buys tighter latency, the other buys fewer rejections; both bound the system, and neither adds throughput."
  - question: "How many coding agents can a single RX 9070 XT actually keep responsive?"
    answer: "About a dozen, and the number comes from the interactive response time law rather than a guess. With eight decode slots delivering roughly 0.88 turns per second and an agent think time near 4.5 seconds, the saturation knee lands at twelve agents, which matches the concurrency ceiling found earlier in this series. Below the knee the card has spare capacity and queue wait is short; above it, goodput is already maxed and every added agent only lengthens the wait. Twelve is therefore not just where the card is efficient, it is the point past which sizing decisions stop being about throughput and start being about which agents you are willing to defer."
draft: false
---

[Yesterday's post](/blog/2026-08-09-turn-length-variance-sets-an-rdna4-swarm-tail-latency/) ended on a warning: size the pool to the mean, but schedule to the variance, because the tail is the number your users feel. That advice assumes you get to schedule at all. It quietly skips the question that comes first, which is what an RX 9070 XT should do when more agents want in than the card can serve.

The tempting answer is to let them wait. Queues are cheap, memory is cheap, and a request that is queued is at least a request you did not drop. On a single card past its saturation point, that instinct is wrong, and it is wrong in a way you can measure. [Two posts ago I put the responsive ceiling at about twelve agents](/blog/2026-08-08-littles-law-caps-a-responsive-rdna4-agent-swarm-near-twelve/). Once you are past it, the card is already handing you every token it has. Every agent you admit after that buys exactly zero additional throughput and only lengthens the line.

So the fix for an oversubscribed swarm is not a deeper queue. It is admission control: decide, at submission time, whether to let a turn in or turn it away. The only real choice left is how you shed, and a discrete-event simulation shows the two obvious policies pulling in different directions.

## The card is already full at a dozen agents

Start with the capacity, because everything below follows from it. One RX 9070 XT runs eight decode slots at roughly 39 tokens per second each, and the series has been using a mixed coding workload with a mean turn of 354 tokens. That is a mean service time of 9.1 seconds per turn, so eight slots deliver about 0.88 completed turns per second. No arrival pattern can push more work through a station than its service rate allows.

The point where a closed population of agents saturates that station is not folklore, it is the [interactive response time law from Lazowska's queueing text](https://homes.cs.washington.edu/~lazowska/qsp/): R = N/X − Z, where R is response time, N is the number of agents, X is throughput, and Z is think time, the gap an agent spends reading results and calling a tool before its next turn. With a think time near 4.5 seconds, the knee lands at twelve agents, which is exactly the ceiling the earlier post found by a different route.

The law also tells you what breaks past the knee. Once X is pinned at its maximum, the equation has only one free variable left. Add agents and R has no choice but to rise. Throughput cannot absorb them, so waiting does.

## Past the knee, the queue is pure waiting

To see the size of the effect, I ran a closed-loop discrete-event simulation: N agents, each thinking for an exponential 4.5 seconds and then submitting a turn drawn from the same 354-token mixture, served by eight slots. I swept N from 8 to 24 and measured two things that matter, the goodput in completed turns per second and the 99th-percentile queue wait, meaning the time a turn spends waiting before it starts decoding.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-08-10-rdna4-swarm-admission-control-shed-vs-queue.svg" alt="A line chart on a dark plum background titled 'Past the knee, an unbounded queue is pure waiting; admission control trades it for deferral'. The horizontal axis is the number of concurrent coding agents on one RX 9070 XT from 8 to 24. The vertical axis is the 99th-percentile queue wait in seconds before a turn starts decoding, from 0 to 30. A dashed vertical guide at twelve agents is labelled 'knee N*=12: goodput saturates'. A cyan curve, the unbounded queue with no admission control, rises almost linearly from about 9 seconds at twelve agents to 27 seconds at twenty-four agents. A gold curve, a fixed concurrency cap of twelve, rises with the others up to the knee and then flattens near 13 seconds, labelled 'fixed cap holds ~13s wait, but defers 37%'. A coral curve, a CoDel-style delay target of 8 seconds, tracks the cyan unbounded curve almost exactly and keeps climbing to about 25 seconds, labelled 'unbounded & CoDel wait keep climbing'. A legend names the three policies." loading="lazy" />
  <figcaption>p99 queue wait against the number of concurrent agents on one RX 9070 XT, from a closed-loop discrete-event simulation with a fixed 354-token mean turn. Goodput saturates at the twelve-agent knee, so the rising cyan curve is throughput the card cannot deliver, showing up as wait. The gold and coral curves add admission control and bend the wait down by shedding submissions.</figcaption>
</figure>

Read the cyan curve first. Below twelve agents it barely leaves the floor, because the card has slack and a submitted turn usually finds a free slot. Past the knee it climbs in a near-straight line, from 8.8 seconds of p99 wait at twelve agents to 15.5 at sixteen, 21.3 at twenty, and 27 seconds at twenty-four. That climb is the interactive response time law drawn out in wall-clock seconds.

The number the cyan curve hides is goodput, and that is the whole point. Across every one of those agent counts, completed turns per second sat between 0.86 and 0.88. Doubling the agents from twelve to twenty-four moved throughput by less than three percent and tripled the tail wait. The queue did not buy you work. It bought you a longer line for the same work.

## Shed by count, or shed by delay

If a deeper queue is the wrong answer, the right one is to stop admitting turns you cannot serve soon. This is not a GPU idea, it is the load-shedding principle the [Google SRE book states plainly](https://sre.google/sre-book/handling-overload/): an overloaded backend should accept only the requests it can process and reject the rest gracefully, because queueing the excess just converts a capacity problem into a latency and reliability problem. The two natural ways to draw that line behave very differently.

The first is a fixed concurrency cap: admit a turn only while the number in the system is below some count, and defer the rest to a short backoff. This is the control most local engines already expose. [vLLM's `max_num_seqs`](https://docs.vllm.ai/en/latest/serving/engine_args.html) is exactly this knob, a ceiling on sequences in flight, paired with an `fcfs` or `priority` scheduling policy and nothing time-aware in between. Set the cap at twelve and the gold curve flattens: p99 wait holds near 13 seconds no matter how many agents pile up outside.

The second is to shed on the delay itself, the way [CoDel manages a network queue in RFC 8289](https://www.rfc-editor.org/rfc/rfc8289). Instead of counting occupants, track the minimum queue sojourn over a sliding interval and begin dropping only once that minimum stays above a target, here 8 seconds, for longer than the interval. It reacts to the wait a user actually feels rather than to a proxy for it. The coral curve shows the result: it tracks the unbounded queue until the delay genuinely builds, then sheds just enough to keep from running away.

| Policy on one RX 9070 XT | p99 wait @20 | deferred @20 | p99 wait @24 | deferred @24 |
| --- | ---: | ---: | ---: | ---: |
| Unbounded queue | 21.3 s | 0% | 27.0 s | 0% |
| Fixed cap = 12 (`max_num_seqs`) | 12.4 s | 31% | 12.7 s | 37% |
| CoDel target = 8 s | 20.5 s | 10% | 24.7 s | 23% |

The table is the whole argument in four columns. Goodput, left out because it is boring, sat at 0.88 turns per second for all three rows: no policy conjures throughput, they only decide where the overload lands. The fixed cap wins on latency, holding p99 wait near 13 seconds, but it sheds bluntly by count and defers 37 percent of submissions at twenty-four agents, some of them turns that would have cleared in a second or two. The CoDel target sheds by delay, so it turns away far fewer agents, 23 percent instead of 37, at the cost of a wait that runs several seconds longer. That is the same lesson the SRE book draws from years of production overload, and the same reason it warns that a static proxy like queries per second makes a poor capacity signal: the thing worth measuring is the resource or the delay, not a count that stands in for it.

## What to actually reach for

None of this makes the card faster. Twelve agents is the ceiling, and the honest framing is that past it you are no longer sizing for throughput, you are choosing who waits and who gets turned away. A deeper queue pretends that choice does not exist and hands every extra agent the same growing delay. Admission control makes the choice explicit.

If your agents can retry cheaply and you care most about the latency of the ones you do serve, a fixed cap near the knee is the simplest thing that works, and it is already in the engine. If deferral is expensive, an agent bounced off the cap is a stalled developer, then shedding on measured delay keeps more agents in the system for a wait that is only a little longer, and it degrades gracefully when the turn mixture shifts under it. An engine like zinc that owns its own submission path can watch the live sojourn and pick between them, which is more than an `fcfs` cap can do.

The through-line from the last three posts is the same one queueing theory has always insisted on. [Little's Law set the mean](/blog/2026-08-08-littles-law-caps-a-responsive-rdna4-agent-swarm-near-twelve/), [variance set the tail](/blog/2026-08-09-turn-length-variance-sets-an-rdna4-swarm-tail-latency/), and admission control sets the boundary. The card will tell you its capacity whether or not you listen. A queue lets you ignore it for a while, right up until the wait makes the decision for you.
