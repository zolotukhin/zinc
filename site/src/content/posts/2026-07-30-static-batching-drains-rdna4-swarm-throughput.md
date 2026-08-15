---
title: "Static batching drains an RDNA4 swarm to a third of its throughput"
seoTitle: "Why Static Batching Wastes an RX 9070 XT Agent Swarm and Continuous Batching Wins It Back"
date: "2026-07-30"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - continuous-batching
  - static-batching
  - iteration-level-scheduling
  - scheduling
  - decode
  - throughput
  - agents
  - local-llm
  - llm-inference
keywords:
  - static batching vs continuous batching local LLM
  - continuous batching RX 9070 XT
  - iteration-level scheduling agent swarm
  - RDNA4 decode batch utilization
  - variable generation length GPU underutilization
  - Orca continuous batching single GPU
  - local agent swarm scheduler throughput
  - shared weight stream idle batch slot
excerpt: "Eight agents on one RX 9070 XT hit 311 tokens per second only if all eight run to the last token together. Real agents do not: a tool-call check emits forty tokens while a refactor plan emits nine hundred. Under a static batch the finished lanes still ride the shared weight stream, and the swarm drains to about a third of its own ceiling. The fix is a scheduler change, not a kernel change."
seoDescription: "A batched decode step on an RX 9070 XT pays one shared 5.5 GB weight stream no matter how many agents produce a token, so keeping the batch full is what makes an agent swarm fast. But local agents finish at wildly different lengths, and a static batch holds its width until the last one ends, spending the shared floor on finished lanes. Modeled from the measured decode step, eight agents with realistic output lengths run at only 39 percent utilization, about 122 tokens per second against a 311 ceiling. Continuous batching, the iteration-level scheduling introduced by Orca and now standard in vLLM and TGI, refills a finished slot on the next step and recovers roughly 2.5x. This post models the draining batch on a single consumer card and argues the scheduler is the cheapest lever left in a local swarm."
faqs:
  - question: "Why does a static batch waste GPU time when agents finish at different lengths?"
    answer: "A batched decode step pays one shared weight stream, about 5.5 GB for Qwen3.5-9B at Q4_K_M on an RX 9070 XT, whether one agent or eight produce a token that step. A naive static batch holds its width until the longest sequence finishes, so once a short agent emits its stop token its lane keeps riding the shared step as padding, doing no useful work. With eight agents whose output lengths range from about forty to nine hundred tokens, only 39 percent of the token slots carry real output and effective throughput falls to roughly 122 tokens per second against a 311 ceiling."
  - question: "What is continuous batching and how is it different from static batching?"
    answer: "Continuous batching, also called iteration-level scheduling, decides the batch membership once per decode step instead of once per request. When a sequence emits its stop token, the scheduler drops it and admits a waiting sequence into the freed slot on the very next step, so the batch stays full. Orca introduced it at OSDI 2022 and vLLM and Hugging Face's text-generation-inference made it standard. Static batching instead fixes the batch when it starts and does not change membership until every sequence in it has finished."
  - question: "Does continuous batching help a single-user local agent swarm?"
    answer: "Yes, because a local coding swarm has exactly the workload that punishes static batching: high variance in output length. A tool-call agent may emit forty tokens in the same batch where a reasoning agent emits nine hundred. Iteration-level scheduling keeps the shared weight stream amortized across a full batch instead of letting it drain as short agents exit, which on the modeled RX 9070 XT recovers about 2.5x of effective throughput."
  - question: "Is continuous batching a kernel change or a scheduler change?"
    answer: "It is a scheduler change. The decode kernels are untouched. What changes is when the engine decides batch membership: every iteration rather than every request. The one real complication is that admitting a new agent means running its prefill, which has a different shape than decode, so the scheduler has to interleave prefill chunks with running decodes rather than blocking the batch on a newcomer."
draft: false
---

The number this series has been quoting for a single-card agent swarm is 311 tokens per second: [eight agents on one RX 9070 XT](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) pulling 5.6 times the aggregate throughput of one. That figure has a quiet assumption baked in. It holds only if all eight agents start together and run to their last token together, so every decode step carries eight live sequences.

Real agents do not behave that way. In a coding swarm, one agent fires a tool call and emits forty tokens deciding what to do with the result, while another is halfway through a nine-hundred-token refactor plan. They enter the batch together and leave it minutes apart. The moment their lengths diverge, the question stops being how fast eight agents decode and becomes what the batch does with the slots that empty out early.

The answer, under the batching most local engines ship by default, is that it wastes them. A static batch holds its width until the longest sequence finishes, and every lane that finished early keeps riding the same expensive decode step producing nothing. The swarm drains toward the throughput of a much smaller batch, and on realistic output lengths it gives back most of the win. The fix is old and well understood in the serving world, and it is a scheduler change rather than a kernel change.

## The whole point of batching is a shared weight stream

Start from why batching helps at all. A decode step is memory bound: the [dominant cost is streaming the model weights out of VRAM](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/), about 5.5 GB for Qwen3.5-9B at Q4_K_M, and that stream is paid once per step no matter how many sequences ride it. Put eight agents in the same step and they split one weight read eight ways. That shared floor is the entire reason a swarm is faster than eight sequential agents.

The measured step latency on the RX 9070 XT fits a clean line in batch size. From the [concurrency sweep](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/), a step takes about 18.0 ms at batch 1, 25.7 ms at batch 8, and 34.5 ms at batch 16, which is t(B) = 16.9 + 1.1B milliseconds. The 16.9 ms is the shared floor, mostly the weight stream and fixed launch overhead. The 1.1 ms per agent is the marginal cost of one more sequence's attention and sampling. Aggregate throughput is B divided by t(B), which climbs from 56 tokens per second at batch 1 to 311 at batch 8.

Read that floor the other way and the failure mode is obvious. If a step runs with only two live agents instead of eight, it still pays almost the whole 16.9 ms, but only two tokens come out. The shared cost has nothing to amortize against. A batch that is draining is a batch spending a full-price weight stream on a discount number of tokens.

## What a draining batch actually costs

Take eight agents with a realistic spread of output lengths: a forty-token tool-call check, a ninety-token edit, then 150, 220, 300, 480, 650, and a nine-hundred-token refactor plan, for 2,830 useful tokens in total. A naive static batch fixes its width at eight when the group starts and does not shrink it until the last agent stops. So the batch runs for 900 steps, every step costs the batch-8 latency of 25.7 ms, and the run takes 23.1 seconds.

In that time the batch had 8 times 900, or 7,200, token slots to fill, and only 2,830 of them carried real output. The other 61 percent were padding: finished agents whose lanes the engine kept alive because the batch width was frozen. Utilization is 39 percent, and effective throughput is 2,830 tokens over 23.1 seconds, about 122 tokens per second, against a 311 ceiling the card can hit when the batch stays full.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-30-static-batching-drains-rdna4-swarm-throughput.svg" alt="A two-panel diagram on a deep petrol-green background titled 'A static batch keeps paying the weight stream for finished agents'. The top panel shows eight horizontal lanes on a shared token axis from 0 to 900 tokens, one lane per agent, labelled tool-call check, short edit, grep plus read, unit test run, file rewrite, multi-file patch, reasoning step, and long refactor plan. Each lane has a gold bar whose length is the tokens that agent generates, at 40, 90, 150, 220, 300, 480, 650 and 900 tokens, and the remainder of each lane out to 900 tokens is filled with a grey diagonal hatch labelled wasted padding. A white dashed vertical line at 900 tokens is labelled 'batch ends when the last lane finishes'. A legend notes useful tokens total 2,830 and wasted padding is 8 times 900 equals 7,200 slots, 39 percent useful. The bottom panel, titled 'Effective useful throughput over the whole run', shows three horizontal bars against an axis of aggregate tokens per second from 0 to about 340. A coral bar for naive static batching with width held at 8 reaches 122 tokens per second, labelled 39 percent of the ceiling. An amber bar for a static batch that shrinks the batch reaches 154 tokens per second, labelled still drops with each exit. A mint bar for continuous batching with slots refilled reaches 311 tokens per second, labelled holds batch 8, plus 2.5x over naive, with a dashed mint ceiling line at 311." loading="lazy" />
  <figcaption>Modeled for Qwen3.5-9B Q4_K_M on one RX 9070 XT using the measured decode step t(B) = 16.9 + 1.1B ms. Eight agents with output lengths from 40 to 900 tokens fill only 39 percent of a frozen-width static batch's token slots, so effective throughput lands near 122 tokens per second. Shrinking the batch as agents exit helps but keeps sliding down the floor; refilling the slots holds the batch at eight and the full 311.</figcaption>
</figure>

The gantt on top is the whole argument. Every gold segment is a token that mattered; every hatched segment is the batch paying for a lane that already stopped. A smarter static batcher shrinks its width as agents finish, which avoids literally computing the padding, and that gets you to about 154 tokens per second. But it is still sliding down the same floor. As the batch drops from eight to two to one, the per-token cost climbs back toward the batch-1 rate, so the last agent's final 250 tokens generate at roughly 56 tokens per second while seven-eighths of the card sits idle. Shrinking the batch treats the symptom. The tail is still slow because the tail is running nearly alone.

## Continuous batching refills the slot instead of freezing it

The serving world solved this in 2022. [Orca](https://www.usenix.org/conference/osdi22/presentation/yu) introduced iteration-level scheduling: decide the batch membership once per decode step rather than once per request. Its framing of the problem is exactly the swarm's, that under fixed batching "requests that have finished earlier than other requests in a batch cannot return to the client, while newly arrived requests have to wait until the current batch completely finishes." When a sequence emits its stop token, the scheduler evicts it and slots a waiting sequence into its place on the very next step. The batch stays full, so the shared weight stream stays fully amortized.

The gain is large precisely when output lengths vary, which is Anyscale's finding when they [benchmarked continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference) against static: identical at low length variance, but as variance grows, static batching's throughput collapses while continuous batching holds. Orca reported up to a 36.9 times throughput improvement over a static baseline at matched latency on a large model. A local swarm will not see numbers that dramatic, because its batch is small and its floor is a consumer card's single GDDR6 bus, but the shape is the same: hold the batch full and you hold 311 instead of draining to 122.

| Scheduling policy | Batch width behavior | Utilization | Effective throughput | Where the loss goes |
| --- | --- | ---: | ---: | --- |
| Naive static | Frozen at 8 until last agent stops | 39% | ~122 tok/s | Finished lanes computed as padding |
| Shrinking static | Drops as each agent exits | rises, then low | ~154 tok/s | Floor no longer amortized, slow tail |
| Continuous batching | Refilled to 8 each step from a queue | ~100% while work waits | ~311 tok/s | Bounded by prefill interleave and KV budget |

The table is the argument in three rows. Freezing the batch wastes the most; shrinking it recovers some but leaves a slow tail running against an unamortized floor; refilling it is the only policy that keeps the shared weight stream doing what it is for. And the important thing about that third row is what it does not require: no new kernels, no change to how a decode step runs. It is entirely a decision about which sequences go into the next step.

## The catch is prefill, and it points back at admission

Continuous batching is not free on a single card, and the reason is the same one that made [admitting a ninth agent expensive](/blog/2026-07-19-admitting-a-ninth-agent-stalls-the-other-eight-for-8-3-seconds/). To refill a slot you have to prefill the newcomer's prompt, and prefill is a compute-bound pass with a different shape than the memory-bound decode step. Drop a full prefill into the loop and it stalls every running agent for its duration. So the scheduler cannot just refill blindly; it has to chunk the newcomer's prefill and interleave those chunks with running decodes, which is the same knob the admission post turned. Continuous batching is the decode-side half of a scheduler whose prefill-side half this series already measured.

There is also a ceiling that continuous batching does not lift. Keeping the batch full holds the shared weight stream amortized, but each agent still reads its own KV cache, and that read [does not amortize across the batch](/blog/2026-07-28-batched-swarm-kv-read-does-not-amortize/). As the agents' contexts grow, the private KV traffic swells until it, not the shared weights, sets the step time, and the 311 ceiling itself sags. Continuous batching gets you back to the ceiling. It does not raise it. The two levers stack the way the [energy and throughput levers](/blog/2026-07-27-batching-eight-agents-saves-more-decode-time-than-energy-on-rdna4/) did: refill the slots to reclaim the amortization, then quantize the KV to keep the ceiling from sagging as context grows.

## The scheduler is the cheapest lever left

The pattern across the last two weeks is that the biggest wins in a local swarm keep turning out to be bookkeeping rather than arithmetic. The card does not need faster kernels to go from 122 to 311 tokens per second on this workload. It needs to stop holding dead lanes in the batch. That is a change to when the engine picks batch members, and the serving world has shipped it in [vLLM and text-generation-inference](https://arxiv.org/abs/2309.06180) for years, mostly aimed at datacenter GPUs fielding many users.

The mild surprise is that a single-user local swarm is close to the worst case those systems were built for. A coding agent's output lengths are wildly variable by nature, a quick tool call sitting in the same batch as a long reasoning turn, and high length variance is exactly the condition under which static batching bleeds the most. The workload a hobbyist runs on one consumer card has the same length distribution that makes iteration-level scheduling pay on an A100. Bringing that scheduler down to a 16 GB RDNA4 card is not a port of a datacenter feature so much as the missing half of the swarm's own scheduler, and it is the difference between a batch that drains and one that stays full.
