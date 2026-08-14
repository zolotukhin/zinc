---
title: "The foreground agent decodes seven times faster than you read"
seoTitle: "Reading-Speed-Aware Scheduling for a Local Agent Swarm on One RDNA4 Card"
date: "2026-08-07"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - scheduling
  - qoe
  - andes
  - reading-speed
  - continuous-batching
  - streaming
  - agents
  - local-llm
  - llm-inference
keywords:
  - reading speed aware LLM scheduling
  - QoE aware serving Andes RDNA4
  - foreground agent token streaming local LLM
  - tokens per second vs reading speed
  - decode slot duty cycle RX 9070 XT
  - client-side token buffer streaming
  - background agent throughput single GPU
  - local agent swarm scheduling objective
excerpt: "Yesterday's post argued a local swarm should schedule its agents fairly. This one argues the foreground agent does not want a fair share at all; it wants to stay ahead of the person reading it. One RX 9070 XT decodes about 39 tokens per second per slot, and an adult reads non-fiction at about 5.3 tokens per second, so the agent whose output you are actually watching has roughly a sevenfold surplus. Every token past reading speed is buffer the eye has not reached, and on one card that surplus is decode time a background agent could be using."
seoDescription: "A foreground coding agent's token stream only has to stay ahead of the human reading it. One RX 9070 XT running Qwen3.5-9B decodes about 39 tokens per second per slot; Brysbaert's 2019 meta-analysis of 190 studies puts adult silent reading at 238 words per minute, about 5.3 tokens per second, so the foreground agent runs roughly 7.4x faster than the reader can consume, and about 4.9x faster even under yesterday's fair-queuing policy. That means one second of decode buffers about 6.4 seconds of reading, and the foreground agent needs only about a 14 percent duty cycle on a slot to never make the reader wait. Everything above reading speed is unread backlog: a dedicated slot piles up more than three minutes of it over 30 seconds. The Andes system (arXiv 2024) formalizes this as Quality-of-Experience, schedules preemptively at token granularity against each request's expected reading speed, and uses a client-side buffer, improving average QoE up to 4.7x over vLLM. The lesson for a single-card swarm: the right objective is per-role. The foreground agent wants read-paced smoothness, the background indexer wants raw throughput, and the surplus from the first pays for the second."
faqs:
  - question: "How much faster does a local agent decode than a person reads?"
    answer: "On one 16 GB RX 9070 XT running Qwen3.5-9B, a resident decode slot produces about 39 tokens per second. Brysbaert's 2019 meta-analysis of 190 reading studies puts average adult silent reading of English non-fiction at 238 words per minute, which at roughly 0.75 words per token is about 5.3 tokens per second. So the foreground agent, the one whose output you are reading, runs about 7.4 times faster than you can consume it. Even under a fair-queuing policy that gives each of several agents about 26 tokens per second, the foreground stream is still nearly 5 times faster than reading speed."
  - question: "What is reading-speed-aware or QoE-aware scheduling?"
    answer: "It is scheduling that treats the goal of a streaming request as keeping the user's token buffer non-empty at their reading pace, rather than maximizing tokens per second. The Andes system from the University of Michigan defines Quality-of-Experience for text streaming this way and schedules preemptively at token granularity, prioritizing requests whose delivery has fallen toward their expected reading speed and de-prioritizing requests already streaming faster than the user can read. It pairs this with a client-side buffer that holds excess tokens and reveals them at reading pace, and reports up to 4.7x higher average QoE than vLLM on the same GPU."
  - question: "Why does streaming a foreground agent faster than reading speed waste the GPU?"
    answer: "Because tokens delivered faster than the reader consumes them just pile up in a buffer the eye has not reached. On this card a dedicated slot delivering 39 tokens per second to a reader who consumes 5.3 accumulates more than three minutes of unread text over a 30-second window. Those decode cycles produced nothing the user perceived any sooner. On a shared single card that same decode time could have advanced a background agent that has real work waiting, so the surplus is not free; it is throughput taken from something else."
  - question: "Does this mean fairness between agents was the wrong goal?"
    answer: "No, it refines it. Fair queuing is the right objective among peers that are equivalent, and yesterday's post used it to stop one agent from starving the others. Reading speed adds that agents are not all equivalent: the one a human is watching has a hard latency floor set by the eye and a large surplus above it, while a background indexer whose output no human reads has no reading-speed floor at all and simply wants throughput. The best single-card policy is per-role, giving the foreground agent a read-paced slice and pouring the rest into background work, not an equal split of the card."
  - question: "When does the reading-speed argument not apply?"
    answer: "It applies only to tokens a human is actively reading. When a foreground agent's output feeds a tool, a compiler, or another agent rather than a person, there is no eye to stay ahead of and the request should be treated as throughput-bound like any background job. It also weakens for content people skim rather than read, such as long code blocks a user scans in seconds, where effective consumption is faster than prose reading speed. And it only matters under contention; if the card is otherwise idle, there is no surplus to reclaim and the agent may as well decode flat out."
draft: false
---

Yesterday's post ended on a clean rule: when several agents share one card, [schedule them fairly](/blog/2026-08-06-first-come-scheduling-starves-an-rdna4-swarm-fair-queuing-shares-the-card/), so no single agent starves the rest. The Virtual Token Counter gives every backlogged agent a bounded share of the same 311 tokens per second, and the card produces exactly as much work either way. Fairness costs nothing in throughput and buys a swarm that feels alive.

There is a fact underneath that rule that makes it incomplete. The agent you are actually looking at, the one filling a chat pane with an explanation while the other seven grind through file reads in the background, is not competing for tokens the way the others are. It is streaming text to a human, and the human is slow. A decode slot on one RX 9070 XT running Qwen3.5-9B produces about [39 tokens per second](/blog/2026-07-30-static-batching-drains-rdna4-swarm-throughput/). A person reads non-fiction at [238 words per minute](https://doi.org/10.1016/j.jml.2019.104047), which is about 5.3 tokens per second. The foreground agent is running seven times faster than you can keep up with.

That gap is the point. Once a stream is that far ahead of the eye, making it faster does nothing a reader can perceive, and on a shared card the cycles that made it faster came out of something that could have used them. The right question for the foreground agent is not how many tokens per second it gets. It is whether its stream stays ahead of the person reading it, and how little of the card that actually takes.

## Reading speed is a real, measured number

The 5.3 figure is worth grounding, because the whole argument rests on it. Marc Brysbaert's 2019 meta-analysis pulled together 190 studies covering 18,573 participants and put average silent reading of English non-fiction at 238 words per minute, with most adults falling between 175 and 300. Fiction runs a little faster, around 260. These are lower than the 300-plus numbers people often quote, because the older figures came from small, unrepresentative samples.

Turning words into tokens needs one conversion. For English, a token is [about three-quarters of a word](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them) in a typical subword tokenizer, so 238 words per minute is roughly 317 tokens per minute, or about 5.3 tokens per second. Call it 5 to 6 tokens per second for a reader who is genuinely reading rather than skimming. That is the rate at which a foreground agent's output is actually being consumed, no matter how fast the GPU can emit it.

Set that against the card. A resident decode slot delivers 39 tokens per second, so the foreground stream arrives at about 7.4 times reading speed. Even if you apply yesterday's fair-queuing policy and the foreground agent is sharing the card down to its VTC allotment of [roughly 26 tokens per second](/blog/2026-08-06-first-come-scheduling-starves-an-rdna4-swarm-fair-queuing-shares-the-card/), it is still arriving at nearly five times the rate the reader can absorb. The surplus does not go away under fair scheduling. It just shrinks a little.

## What the surplus buys, and what it wastes

A surplus that large changes what a decode second is worth. If the slot delivers 39 tokens in a second and the reader drains 5.3, that one second put about 34 tokens into a buffer, which is a little over six seconds of reading. One second of GPU time bought more than six seconds of runway. So the foreground agent does not need the slot continuously. It needs it about one second in every seven, a duty cycle near 14 percent, to keep the reader's buffer from ever running dry.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-08-07-foreground-agent-reading-buffer.svg" alt="Two panels on a dark navy background under the heading 'Reading speed, not throughput, is the ceiling the foreground agent needs'. A subtitle notes one RX 9070 XT running Qwen3.5-9B, a resident decode slot emitting about 39 tokens per second, an adult reading non-fiction at about 5.3 tokens per second or 238 words per minute per Brysbaert 2019, so one second of decode buffers about 6.4 seconds of reading and the foreground agent needs only about 14 percent of a slot to never let the reader wait. The left panel is a teal area chart titled 'Reading buffer of the foreground agent (seconds of reading the user is ahead)' plotting a sawtooth over a 30-second window. The buffer rises steeply to about 6.4 seconds during each one-second decode burst, then drains linearly over six seconds, oscillating between roughly 0.4 and 7.9 seconds and never touching a red dashed 'stall line (buffer empty, reader waits)' at zero. A row of small amber blocks along the bottom, labelled 'decode slot held: 1s on / 6s off', marks the bursts. The right panel is a bar chart titled 'Delivered rate vs what the reader needs' with three bars: a tall amber 'Dedicated slot' bar at 39, a medium blue 'Fair-share (VTC)' bar at 26, and a short green 'Read-paced (Andes)' bar at 5.4, against a red dashed horizontal line labelled 'reading need about 5.3 tokens per second (238 wpm)'. A note explains the surplus of the first two bars above the dashed line is buffer the reader cannot use yet, so it is free for background throughput." loading="lazy" />
  <figcaption>Left: the foreground agent's reading buffer, in seconds of reading the user is ahead, when it holds a decode slot for one second in every seven. The buffer sawtooths but never empties, so the reader never waits. Right: delivered rate under three policies against the reader's actual need. A dedicated slot and a fair share both tower over reading speed; only the read-paced policy sits near it, freeing the rest of the card.</figcaption>
</figure>

The left panel is the buffer over thirty seconds, measured in seconds of reading the user is ahead. Each short burst of decode refills it to about six seconds; between bursts the reader drains it at 5.3 tokens per second. It sawtooths, but at a 14 percent duty cycle it never reaches the stall line, so the person watching sees a smooth, unbroken stream and has no idea the agent spent most of the wall clock not decoding at all.

The right panel is the waste. A dedicated slot delivers 39 tokens per second to a reader who needs 5.3, and over the same thirty seconds it piles up more than three minutes of text the eye has not reached. Those decode cycles were real. They produced nothing the user perceived any sooner, and on a shared card they were taken from the seven background agents that had genuine work queued. Fair queuing narrows the tower from 39 to 26 but does not remove it; the read-paced bar is the only one that sits near what the reader can use.

## Andes already wrote this down

This is not a hunch, and it is not new. A group at the University of Michigan built a serving system called [Andes](https://arxiv.org/abs/2404.16283) around exactly this observation. Their argument is that existing serving systems optimize metrics that are not aligned with user experience, and they define Quality-of-Experience for text streaming in terms of the user's whole interaction timeline: the first token should arrive quickly, and subsequent tokens should arrive at a smooth, digestible pace, meaning at or a little above reading speed rather than as fast as possible.

Andes turns that definition into a scheduler. It preempts at the granularity of a single token and dynamically prioritizes requests by their expected QoE gain, which means a request that has fallen behind its reader's pace gets the card and a request already streaming well ahead of its reader yields. It pairs the scheduler with a client-side buffer that holds the excess tokens and reveals them to the user at the expected pace, which is the same buffer the left panel above is drawing. On the same hardware, Andes reports up to 4.7 times higher average QoE than a state-of-the-art baseline, or the same experience on 61 percent fewer GPU resources. The win does not come from decoding faster. It comes from not spending decode on tokens the reader cannot see yet, and spending it on the requests that are actually behind.

## The objective is per-role, not per-agent

Put this next to yesterday's post and the shape of the right policy comes into focus. Fair queuing treats the agents as equivalent and splits the card evenly. Reading speed says they are not equivalent, and the thing that distinguishes them is whether a human is on the other end.

The foreground agent has a human reader, so it has a hard latency floor at reading speed and a large surplus above it. It wants smoothness, not throughput; past about six tokens per second, more is invisible. A background agent, the indexer folding files into context or the [tool-calling loop that reads back eighteen tokens for every one it writes](/blog/2026-07-23-a-local-coding-agent-reads-back-eighteen-tokens-for-every-one-it-writes/), has no human reading its output at all. It has no reading-speed floor, and it simply wants tokens per second so it can finish and get out of the way. Splitting the card evenly between those two serves neither well. It over-serves the foreground agent, which cannot use the extra, and under-serves the background agent, which can use all of it.

The better policy gives the foreground agent the small read-paced slice it needs to keep its buffer full, then pours the entire surplus into the background agents. Because the whole scheme is [work-conserving](/blog/2026-08-06-first-come-scheduling-starves-an-rdna4-swarm-fair-queuing-shares-the-card/), the card still runs flat out at 311 tokens per second; only the split changes. The person watching the foreground pane sees the same smooth stream they would have seen from a dedicated slot, and the background work finishes meaningfully sooner because it inherited the six-out-of-seven decode seconds the foreground agent did not need.

## Where the argument stops

Two honest limits keep this from being a universal rule. The first is that reading speed only bounds tokens a human reads. The moment a foreground agent's output stops going to a person and starts feeding a tool or another agent, the eye is gone and the request is throughput-bound like any background job. In a coding swarm that switch happens constantly, sometimes several times within one agent's turn, so the scheduler has to know which stream currently has a human attached rather than labeling an agent foreground once and forever.

The second is that reading is not uniform. A user skimming a long code block consumes it far faster than 5.3 tokens per second, and someone carefully reading a dense explanation consumes it slower. A fixed reading rate is a decent default, but Andes treats the expected pace as a per-request parameter for a reason, and a good local implementation would let it vary by content and by how fast the user is actually scrolling.

The pattern this series keeps hitting holds one more time. The wins in a single-card swarm are not new arithmetic; they are decisions about what the fixed card should spend its time on. Continuous batching, on-demand KV allocation, chunked prefill, fair queuing, and now reading-speed awareness are all ways of pointing the same 311 tokens per second at the work that matters. Production engines still schedule the foreground stream as if the user could read at 39 tokens per second, and expose only [coarse policies](https://docs.vllm.ai/en/latest/serving/engine_args.html) to change it. On a shared consumer card, where every decode second the reader cannot use is a second stolen from an agent that could, the stream you are watching does not need to be fast. It needs to be one comfortable step ahead of your eyes, and nothing more.
