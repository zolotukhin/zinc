---
title: "A long background turn stalls the foreground agent until RDNA4 preempts the decode slot"
seoTitle: "Token-Boundary Preemption Fixes Priority Inversion in an RDNA4 Agent Swarm"
date: "2026-08-11"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - scheduling
  - preemption
  - priority-inversion
  - continuous-batching
  - kv-cache
  - agents
  - local-llm
  - llm-inference
keywords:
  - decode slot preemption local LLM
  - priority inversion LLM serving
  - token-boundary preemption RDNA4
  - iteration-level scheduling Orca
  - vLLM preemption recompute swap
  - foreground agent tail latency single GPU
  - resident KV cache no recompute
  - RX 9070 XT agent swarm scheduling
excerpt: "On one RX 9070 XT with eight decode slots, the foreground agent usually gets a slot in under a second. The problem is the tail. When it lands behind a long background turn, a slow refactor holding its slot for twenty-plus seconds, it waits the whole thing out: p99 is 6.2 seconds and the worst case is half a minute. That is priority inversion, the same failure that reset Mars Pathfinder, and the fix is not a smarter queue. It is preempting the decode slot at a token boundary, which on RDNA4 costs almost nothing because the paused turn's KV cache never leaves the 16 GB of VRAM."
seoDescription: "A single 16 GB RX 9070 XT running Qwen3.5-9B serves eight decode slots at about 39 tokens per second each. A discrete-event model over a coding-turn mixture with a 354-token mean shows the foreground agent's wait for a free slot is fine on average, p50 0.8 seconds, but the tail is ugly: p90 2.5 seconds, p99 6.2 seconds, worst case about 31 seconds, because a long low-priority background turn holds its slot for the whole turn. This is classic priority inversion, the bug that reset the Mars Pathfinder lander in 1997. Strict priority on the waiting queue does not fix it, because the running turn is the one blocking. Iteration-level scheduling from Orca (OSDI 2022) makes preemption possible at a token boundary, capping the foreground wait near 29 milliseconds regardless of how long the background turn runs. On RDNA4 that preemption is nearly free: the paused turn's KV cache stays resident in the 16 GB of VRAM, so resuming it needs no swap to host memory and no recompute, unlike vLLM's default recompute path."
faqs:
  - question: "Why does the foreground agent sometimes wait several seconds for a decode slot on one RX 9070 XT?"
    answer: "Because all eight decode slots can be busy with background turns when the foreground turn arrives, and without preemption it has to wait for one of those turns to finish. On average that wait is short, about 0.8 seconds at the median, because eight slots finish turns often. The trouble is the tail. Coding turns have a heavy-tailed length distribution, so every so often the foreground turn lands behind a long background turn, a big refactor or file write that occupies its slot for twenty seconds or more. A discrete-event model of eight slots serving a 354-token mean mixture puts the p99 wait at 6.2 seconds and the worst observed wait near 31 seconds. The mean hides the problem; the p99 is what the person watching the cursor actually feels."
  - question: "What is priority inversion in the context of LLM decode scheduling?"
    answer: "Priority inversion is when a high-priority task is blocked by a lower-priority one that is holding a shared resource. The classic case reset the Mars Pathfinder lander in 1997, where a low-priority weather task held a mutex the high-priority bus manager needed. In a local agent swarm the shared resource is a decode slot. A low-priority background turn is admitted to a slot and, because most engines cannot interrupt a running turn, it holds that slot until it finishes generating. If the interactive foreground turn arrives while every slot is occupied by such turns, it is blocked by work that should yield to it. Giving the foreground turn a higher priority in the waiting queue does not help, because the block comes from a turn that is already running, not from the queue order."
  - question: "How does token-boundary preemption fix it, and what does it cost on RDNA4?"
    answer: "Autoregressive decoding advances one token per step across the whole batch, so there is a natural interruption point after every token. Iteration-level scheduling, introduced by Orca at OSDI 2022, reschedules the batch at that granularity. To admit the foreground turn you evict a background turn from its slot at the next token boundary, which takes at most one decode step, about 26 milliseconds at 39 tokens per second, plus a small barrier flush. In the model the foreground wait drops from a 6.2-second p99 to about 29 milliseconds, and that cap holds no matter how long the evicted background turn would have run. On an RX 9070 XT the eviction is nearly free because the paused turn's KV cache stays resident in the 16 GB of VRAM. Resuming it needs no swap to host memory and no prefill recompute, which is the opposite of vLLM's default recompute preemption, where a preempted request throws away its KV and regenerates it later."
  - question: "Can vLLM already preempt a running request for a higher-priority one?"
    answer: "Not as of mid-2026 for this exact case. vLLM has preemption, but it triggers on memory pressure: when the KV cache runs out of blocks, the scheduler evicts running sequences and either recomputes or swaps them later, with recompute the default. Its priority scheduling can evict low-priority requests from the running queue when resources are insufficient, but an open feature request, issue #40004, notes that when the running queue is already at max_num_seqs, a high-priority request waiting to be scheduled cannot preempt a running one. That is precisely the priority-inversion gap. An engine that owns its own scheduler, like zinc, can preempt on priority rather than only on memory, and on RDNA4 it can do so without paying the swap or recompute tax that makes preemption expensive elsewhere."
draft: false
---

The foreground agent, the one whose output you are actually reading, almost always gets a decode slot fast. On a single [RX 9070 XT running eight decode slots](/blog/2026-08-08-littles-law-caps-a-responsive-rdna4-agent-swarm-near-twelve/), the median wait for a free slot is under a second. If that were the whole story there would be nothing to write. The whole story is the tail.

Every so often the foreground turn arrives while all eight slots are busy, and the slot that would free up first is running a long background turn: a big refactor, a full-file rewrite, a thousand-token generation that will hold its slot for the better part of half a minute. There is no free slot and no way to make one, so the foreground turn waits the long turn out. A discrete-event model of that swarm puts the wait at 6.2 seconds at the 99th percentile and about 31 seconds in the worst case. The person watching the cursor does not experience the median. They experience the stall.

That stall has a name from long before local LLMs existed. It is priority inversion, and the fix is not a smarter queue.

## The slot is the shared resource

The setup is the same one this series has been measuring for two weeks. One 16 GB RX 9070 XT runs Qwen3.5-9B with eight decode slots, each producing roughly 39 tokens per second, and the workload is a mixed coding session where a turn averages 354 tokens. Most turns are short, a quick answer or a tool call, but the distribution has a long tail: the occasional turn runs to a couple of thousand tokens, which at 39 tokens per second is nearly a minute on one slot.

[Continuous batching](https://www.usenix.org/conference/osdi22/presentation/yu) keeps all eight slots full by admitting a new turn whenever an old one finishes. That is exactly what you want for throughput. It also means that at any instant the card is committed to eight turns in progress, and a turn, once admitted, keeps its slot until it stops generating. The slot is a shared resource with no natural release point except the end of the turn.

Now give the foreground turn priority. The intuition from the [fair-queuing post](/blog/2026-08-06-first-come-scheduling-starves-an-rdna4-swarm-fair-queuing-shares-the-card/) was that a good scheduler shares the card by turn, and it does, at admission time. But priority in the waiting queue only decides who gets the next slot to open. It says nothing about the eight turns already running. If all eight are long background turns, the highest priority in the world does not conjure a slot. The foreground turn is blocked, not by the queue order, but by work that is already on the card and will not yield.

## This is the Mars Pathfinder bug

The pattern is old enough to have a famous failure attached to it. In July 1997 the Mars Pathfinder lander began resetting itself on the Martian surface. The [cause was a priority inversion](https://www.rapitasystems.com/blog/what-really-happened-software-mars-pathfinder-spacecraft): a low-priority meteorological task held a mutex that a high-priority bus management task needed, and a medium-priority communications task kept the low-priority one from finishing and releasing the lock. The bus manager missed its deadline, a watchdog timer noticed, and the whole system rebooted. The fix was one boolean, priority inheritance on that mutex, uploaded to another planet.

The shape maps cleanly onto a decode swarm. The high-priority task is the foreground turn. The shared resource is the decode slot, or equivalently the KV-cache blocks backing it. The low-priority task is a background turn that acquired a slot and will not give it up until it finishes generating. The medium-priority tasks are all the other background turns keeping the card busy so nothing frees early. The foreground turn misses the only deadline that matters here, the one measured against how fast a human reads, and there is no watchdog to reboot, only a user watching a frozen cursor.

The lesson from Pathfinder is the same lesson here. You do not fix priority inversion by reordering the queue. You fix it by letting the high-priority task take the resource back.

## Preempt at the token boundary

Autoregressive decoding hands you the interruption point for free. Every slot advances exactly one token per step, so after each step there is a clean boundary where the batch can be rebuilt. This is the core idea of [iteration-level scheduling from Orca](https://www.usenix.org/conference/osdi22/presentation/yu), which reschedules the running set at the granularity of a single decode iteration rather than a whole request. Once you schedule per iteration, preemption is not a special operation. It is just choosing not to put a turn back in the next batch.

So when the foreground turn arrives and every slot is full, you evict one background turn at the next token boundary and give its slot to the foreground turn. The cost is at most one decode step, about 26 milliseconds at 39 tokens per second, plus a small barrier flush to drain the in-flight step. The evicted background turn stops where it is and waits for a slot to come back.

The difference in the foreground wait is not subtle. Below is the same discrete-event model, eight slots serving the 354-token mixture, measuring the foreground turn's wait for a slot under the two policies.

| Policy on one RX 9070 XT | p50 wait | p90 wait | p99 wait | worst case |
| --- | ---: | ---: | ---: | ---: |
| No preemption (wait for a slot to free) | 0.76 s | 2.45 s | 6.17 s | ~31 s |
| Token-boundary preemption | 17 ms | 17 ms | 29 ms | 30 ms |

The median barely moves in absolute terms, because the median was already fine. What collapses is the tail. The 99th-percentile wait falls from 6.2 seconds to 29 milliseconds, a factor of about 210, and the worst case stops depending on how unlucky the foreground turn was in which turns happened to be running. That last point is the real win. Without preemption the foreground wait is a function of background turn lengths, so it inherits their heavy tail. With preemption the wait is capped at one decode step no matter what the background turns are doing.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-08-11-decode-slot-preemption-vs-priority-inversion-rdna4.svg" alt="A two-panel timeline diagram on a dark slate-indigo background with a shared time axis from 0 to 32 seconds. The top panel, labelled no preemption, shows decode slot 3 occupied by a gold background bar, a 940-token refactor running about 24 seconds. A cyan dashed line marks the foreground turn arriving at 4 seconds. A red hatched wait bar runs from 4 to 24 seconds labelled 'blocked 20 s, priority inversion', after which the cyan foreground turn finally starts. The bottom panel, labelled token-boundary preemption, shows the same background bar but paused at the 4-second boundary, drawn as a faded dashed segment labelled 'paused at token boundary', while the cyan foreground turn starts within about 26 milliseconds and runs for about 9 seconds. The background bar then resumes as solid gold, labelled 'background resumes from resident KV, no swap, no recompute'." loading="lazy" />
  <figcaption>The same long background turn on one decode slot, without and with token-boundary preemption. Without it, the foreground turn is blocked for the whole background turn, the red hatched span, which is priority inversion drawn out in wall-clock time. With it, the foreground turn is admitted at the next token step and the background turn resumes later from a KV cache that never left VRAM.</figcaption>
</figure>

## Why preemption is cheap on RDNA4 and expensive elsewhere

Preemption has a reputation for being costly, and on most serving stacks it earns it. The expense is not stopping the turn, it is resuming it. When you evict a turn you have to decide what happens to its KV cache, the growing tensor of attention keys and values that represents everything it has generated so far. [vLLM's two preemption modes](https://docs.vllm.ai/en/stable/configuration/optimization/) show the choices. Swap moves those KV blocks out to host memory and copies them back on resume, paying twice over the PCIe bus. Recompute, the default, throws the blocks away and regenerates them by re-running the turn's prefill when it comes back. Either way, resuming a preempted turn costs real work proportional to how much context it had built up. That cost is what makes engines reluctant to preempt for anything short of running out of memory.

On a single RX 9070 XT the calculus is different, because the whole point of a single-card local swarm is that the KV cache already lives in the 16 GB of VRAM and never needed to leave. Preempting a turn does not evict its KV, it just stops scheduling the turn. The blocks sit resident exactly where they were, and resuming the turn means putting it back in the next batch and continuing from the token it stopped on. There is no swap and no recompute, which is the same reason an earlier post found that [swapping an idle agent's KV beats recomputing it by 177 times](/blog/2026-07-20-swapping-an-idle-agents-kv-cache-beats-recomputing-it-by-177x/) when the memory is there to hold it. The resident cache turns preemption from an expensive fallback into a routine scheduling move.

This is where most engines are not yet set up to help. vLLM can preempt, but on memory pressure, not on priority: an [open feature request, issue #40004](https://github.com/vllm-project/vllm/issues/40004), spells out that once the running queue is at `max_num_seqs`, a higher-priority waiting request still cannot preempt a running one. That is the priority-inversion gap stated in the engine's own tracker. Closing it needs a scheduler that will evict a running turn because something more important arrived, not only because the card ran out of blocks. An engine like zinc that owns its own submission and scheduling path can make that call, and on RDNA4 it can make it without paying the resume tax that makes preemption a last resort everywhere else.

## What to reach for

The foreground agent [decodes far faster than you can read it](/blog/2026-08-07-the-foreground-agent-decodes-seven-times-faster-than-you-read/), so the moment it is on the card it feels instant. Everything that makes it feel slow happens before it gets a slot. [Admission control](/blog/2026-08-10-a-deeper-queue-cannot-fix-an-oversubscribed-rdna4-agent-swarm/) decides who gets in at all, and that is the right tool when the card is oversubscribed. Preemption is the tool for the case underneath it, when there is important work waiting and the only thing standing in front of it is unimportant work that happens to have arrived first.

The honest framing is that priority is not a property of a queue, it is a property of what is allowed to interrupt what. A scheduler that can only reorder the waiting line will let a long background refactor sit on a slot for thirty seconds while the person is waiting on a one-line answer. A scheduler that can preempt at a token boundary gives that answer back in the time it takes to finish one token, and on RDNA4 it hands the refactor its slot back afterward with nothing lost. The card was never the bottleneck for the foreground turn. The willingness to take a slot back was.
