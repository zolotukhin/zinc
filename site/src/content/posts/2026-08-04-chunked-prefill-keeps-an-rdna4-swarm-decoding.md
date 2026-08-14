---
title: "Chunked prefill keeps an RDNA4 swarm decoding while one agent's prompt loads"
seoTitle: "Chunked Prefill on RDNA4: Stall-Free Scheduling for a Local Agent Swarm"
date: "2026-08-04"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - chunked-prefill
  - continuous-batching
  - scheduling
  - sarathi-serve
  - agents
  - local-llm
  - llm-inference
keywords:
  - chunked prefill RX 9070 XT agent swarm
  - stall-free scheduling RDNA4 local LLM
  - Sarathi-Serve chunked prefill decode latency
  - prefill decode interference continuous batching
  - inter-token latency chunk size tradeoff
  - vLLM enable_chunked_prefill max_num_batched_tokens
  - local agent swarm prefill stall RDNA4
  - Dynamic SplitFuse DeepSpeed-FastGen
excerpt: "A single long prefill runs the GPU flat out, so when one agent in a swarm hands the engine a fresh 1,730-token prompt, a monolithic scheduler freezes the other eight for the whole 8.3 seconds it takes to process. Chunked prefill splits that prompt into 128-token slices and lets the eight decoders emit a token between each slice, cutting the worst-case decode stall from 8.3 seconds to about two thirds of a second for a roughly 3 percent hit to the incoming agent's time-to-first-token. This is the scheduling lever that sits on top of the memory levers, and on a shared consumer card it decides whether the swarm feels alive or frozen."
seoDescription: "On one 16 GB RX 9070 XT running Qwen3.5-9B, eight agents decode by continuous batching, one iteration at a time. A prefill saturates GPU compute, so a monolithic scheduler that runs a ninth agent's 1,730-token prompt in a single iteration stalls every decoding agent for the full 8.3 seconds the series already measured. Chunked prefill, from Sarathi-Serve at OSDI 2024, splits the prompt into equal slices and piggybacks the eight decodes onto each slice, so no decoder waits longer than one iteration. Modeled on this card, a 128-token chunk drops the worst-case decode stall from 8.3 s to about 0.66 s, a 13x cut, for roughly a 3 percent increase in the incoming agent's time-to-first-token; a 32-token chunk reaches 0.2 s at about a 10 percent TTFT cost. The chunk size is the single knob that trades decode tail latency against time-to-first-token, the same tradeoff vLLM exposes as max_num_batched_tokens. This post shows why prefill and decode collide in a continuously batched swarm and where the stall-free schedule lands on RDNA4."
faqs:
  - question: "Why does one agent's prefill stall the whole swarm?"
    answer: "Because a continuously batched engine runs one iteration at a time on the GPU, and a prefill saturates GPU compute. Prefill processes the entire prompt in parallel, so it is compute-bound and fills the machine, while decode processes one token per sequence and is memory-bound. If the scheduler puts a whole prompt into a single iteration, that iteration does nothing but prefill, and every decoding agent waits for it to finish. On one RX 9070 XT a 1,730-token prompt takes about 8.3 seconds to prefill, and a monolithic schedule freezes all eight decoders for the entire span."
  - question: "What is chunked prefill and stall-free scheduling?"
    answer: "Chunked prefill, introduced by Sarathi-Serve at OSDI 2024, splits a prompt into near-equal-sized chunks and processes one chunk per iteration instead of the whole prompt at once. Each iteration also carries the in-flight decode tokens, so the decoders emit a token between every chunk rather than waiting for the full prefill. That is the stall-free schedule: adding a new request never pauses ongoing decodes. Chunked prefill is now the default scheduling strategy in vLLM and SGLang."
  - question: "How much does chunked prefill cut decode latency on RDNA4?"
    answer: "Modeled for a 1,730-token prompt on one RX 9070 XT, a 128-token chunk caps the worst-case decode stall at about 0.66 seconds instead of the 8.3-second monolithic freeze, roughly a 13x reduction, while the incoming agent's time-to-first-token rises only about 3 percent. A smaller 32-token chunk pushes the stall down to about 0.2 seconds, a 40x cut, at roughly a 10 percent TTFT penalty. The chunk size is the knob that trades decode tail latency against time-to-first-token."
  - question: "What is the cost of making the prefill chunks too small?"
    answer: "Two costs. First, every chunk re-reads the KV cache of all previously processed prompt tokens for the attention step, so slicing a prompt into more chunks adds attention work that scales with the number of chunks. Second, each iteration carries fixed overhead for kernel launches and scheduling, so very small chunks multiply that overhead. Below a certain size the iteration also stops being compute-bound and the piggybacked decodes no longer ride for free. The practical range most engines settle on is a few hundred to a couple thousand prefill tokens per iteration."
  - question: "Is chunked prefill the same knob as vLLM's max_num_batched_tokens?"
    answer: "Effectively yes. In vLLM, enabling chunked prefill lets the scheduler prioritize decodes and fit pending prefill tokens into the remaining max_num_batched_tokens budget, chunking any prefill that does not fit. A smaller budget gives better inter-token latency because fewer prefill tokens slow the decodes in each iteration, and a larger budget gives better time-to-first-token because more of the prompt clears per iteration. That budget is the same decode-latency-versus-TTFT lever this post models for the RDNA4 swarm."
draft: false
---

The series has spent two weeks on where an [eight-agent swarm](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) keeps its memory: how much [KV cache](/blog/2026-07-31-q8-kv-cache-pushes-an-rdna4-swarm-crossover-from-5k-to-10k-tokens/) each agent gets, and how to [allocate it on demand](/blog/2026-08-01-reserve-for-max-kv-cache-fits-four-of-eight-agents-on-rdna4/) instead of reserving each one's worst case. Those posts assumed the eight agents were quietly decoding side by side. They usually are. The interesting failure happens the moment a ninth agent shows up with a prompt, or one of the eight folds a large tool observation back into its context and has to prefill it.

That prefill does not politely wait its turn. It runs the GPU flat out, and on a continuously batched engine that processes one iteration at a time, it can freeze every other agent while it works. The series already measured the symptom without naming the cause: [admitting a ninth agent stalled the other eight for 8.3 seconds](/blog/2026-07-19-admitting-a-ninth-agent-stalls-the-other-eight-for-8-3-seconds/). Eight agents in the middle of answering a user went silent for eight full seconds because a ninth agent's prompt was being read.

The fix is not more memory or a faster kernel. It is a scheduling decision about how a prompt is allowed to enter the batch, and it has a name: chunked prefill.

## Prefill and decode want opposite things from the GPU

An LLM request runs in two phases with very different shapes. Prefill reads the whole prompt at once, so it processes hundreds or thousands of tokens in parallel and saturates GPU compute. Decode produces one token at a time, touching the full weight set to make a single token, so it is memory-bound and leaves most of the compute units idle. The [Sarathi-Serve paper](https://www.usenix.org/conference/osdi24/presentation/agrawal) at OSDI 2024 puts it plainly: prefill iterations have high latency but saturate compute, decode iterations have low latency and low compute utilization.

That split is why batching helps decode so much and why the swarm exists at all. Eight memory-bound decodes share one pass over the weights, which is the whole [5.6x throughput win](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) the series opened with. But the same split is what makes prefill dangerous in a shared batch. When a prompt enters, the engine has to run it through the model, and if it runs the entire prompt in one iteration, that iteration is pure prefill. There is no room left for the eight decodes, so they wait.

On one RX 9070 XT the wait is not abstract. A fresh agent's prompt of around 1,730 tokens, roughly the [shared system prompt](/blog/2026-07-21-the-system-prompt-a-local-agent-swarm-caches-eight-times-over/) plus a little context, takes about 8.3 seconds to prefill at the batched-prefill rate this card sustains. For those 8.3 seconds the other eight agents produce nothing.

## Slice the prompt, and let the decodes ride along

Chunked prefill breaks the assumption that a prompt has to enter in one iteration. Instead of feeding all 1,730 tokens at once, the scheduler splits the prompt into near-equal chunks, say 128 tokens each, and processes one chunk per iteration. The important part is what shares the iteration with each chunk: the eight in-flight decodes. Every iteration now does one prefill chunk plus one decode token for each of the eight agents, so the decoders emit a token between every chunk instead of waiting for the whole prompt.

Sarathi-Serve calls the result a stall-free schedule, because adding a new request never pauses ongoing decodes. [DeepSpeed-FastGen](https://arxiv.org/abs/2401.08671) arrived at the same idea independently under the name Dynamic SplitFuse and reported up to 3.7x lower token-level tail latency against vLLM from composing prefill and decode into uniform iterations. The technique has since become the default scheduling strategy in vLLM and SGLang, which is a strong signal that it is the right shape for shared serving rather than a niche trick.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-08-04-rdna4-swarm-chunked-prefill-schedule.svg" alt="A two-panel scheduling gantt on a dark graphite background titled 'One agent's prefill freezes the swarm, or it doesn't'. Both panels show a horizontal time axis from 0 to 9 seconds and eight decoder lanes labelled a1 to a8. In the top panel, labelled Monolithic prefill, each lane emits two small green decode ticks near time zero, then a single wide amber block spans all eight lanes from about 0.6 to 8.9 seconds, labelled agent 9 prefill 1,730 tokens, GPU fully occupied, every decoder waits. A red bracket under the block reads 8.3 s freeze, no agent emits a token, and the lanes show no green ticks during the block. In the bottom panel, labelled Chunked prefill 128-token slices, the same span is broken into fourteen narrow amber slices labelled 14 chunks times 128 tokens, and between every slice all eight lanes emit a green decode tick, so the decoders keep producing tokens throughout. A green bracket over one segment reads each decoder's gap approximately 0.66 s, and a note reads agent 9's first token still lands at about 8.5 s, about 3 percent later than monolithic." loading="lazy" />
  <figcaption>Modeled for Qwen3.5-9B on one 16 GB RX 9070 XT: a 1,730-token prompt at the card's batched-prefill rate, eight in-flight decoders. The monolithic schedule runs the prompt in one iteration and freezes every decoder for 8.3 seconds. Chunking the prompt into 128-token slices lets the eight decoders emit a token between each slice, so no decoder waits longer than one iteration.</figcaption>
</figure>

The diagram is the whole argument. In the top panel the amber prefill owns the GPU and the green decode ticks simply stop for eight seconds. In the bottom panel the prefill is the same total amount of amber, cut into slices, and the decodes keep ticking in the gaps between them. Nothing about the total work changed. The prompt still costs the same 1,730 tokens of prefill. Only the order changed, and the order is what the agents feel.

## The chunk size is the only real knob

Chunking is not free, and the cost lands entirely on the incoming agent. Because the eight decodes now ride in every iteration, and because slicing a prompt makes each chunk re-read the KV cache of the prompt tokens before it, the total prefill takes slightly longer to finish. So the incoming agent's time-to-first-token goes up a little while every other agent's decode latency comes down a lot. The chunk size sets where on that trade you land.

The numbers below are modeled for the same card and prompt, with a per-token prefill cost of about 4.8 milliseconds, a batched eight-agent decode step of about 45 milliseconds, and a fixed per-iteration overhead of about 15 milliseconds. The worst-case decode stall is one iteration, and the incoming agent's time-to-first-token is the sum of all the iterations its prompt takes.

| Prefill chunk | Iterations | Worst-case decode stall | Agent 9 time-to-first-token | TTFT penalty |
| --- | ---: | ---: | ---: | ---: |
| Monolithic (1,730) | 1 | 8.3 s | 8.3 s | baseline |
| 512 tokens | 4 | 2.5 s | 8.4 s | ~1% |
| 128 tokens | 14 | 0.66 s | 8.5 s | ~3% |
| 32 tokens | 55 | 0.20 s | 9.1 s | ~10% |

The middle rows are the point. Dropping from a monolithic prompt to 128-token chunks cuts the worst-case decode stall from 8.3 seconds to about two thirds of a second, a 13x reduction, and the agent whose prompt is being read waits only about 3 percent longer for its first token. Push to 32-token chunks and the stall falls to a fifth of a second, a 40x cut, but now the incoming agent pays a real 10 percent, and the 55 iterations start piling up re-read attention work and launch overhead. Below that the chunks get small enough that the iteration is no longer compute-bound and the piggybacked decodes stop riding for free, which is why no production engine chunks down to single tokens.

This is the same lever vLLM exposes as `max_num_batched_tokens` once [chunked prefill is enabled](https://docs.vllm.ai/en/stable/configuration/optimization/). Its own tuning guidance matches the table exactly: a smaller budget gives better inter-token latency because fewer prefill tokens slow the decodes, and a larger budget gives better time-to-first-token because more of the prompt clears per iteration. The tradeoff is not something the engine can optimize away. It can only be placed, and for an interactive swarm where a frozen agent is the worst thing a user sees, it belongs near the low end.

## Why this matters more on a consumer card than in a datacenter

A datacenter serving stack hides prefill stalls behind sheer parallelism. It has many GPUs, deep request queues, and a scheduler whose job is aggregate throughput under a latency target. A prompt landing on one replica barely registers. The local swarm has none of that cushion. It is one card, eight tenants, and a human watching at least one of them. A single unlucky prefill does not blend into a queue, it freezes a quarter of the visible agents at once, which is exactly the [8.3-second stall](/blog/2026-07-19-admitting-a-ninth-agent-stalls-the-other-eight-for-8-3-seconds/) that motivated this post.

The consumer card also makes prefill stalls more likely, not less. A coding agent's real workload is not a short chat prompt. It is [tool output folded back into context](/blog/2026-07-23-a-local-coding-agent-reads-back-eighteen-tokens-for-every-one-it-writes/), file after file, each fold a fresh prefill of hundreds or thousands of tokens. Eight agents doing that means prefills arrive constantly, not once at session start. Without a stall-free schedule, the swarm spends its life lurching between decode and freeze. With one, the prefills dissolve into the decode stream and the whole thing stays responsive.

There is an RDNA4-specific reason to like chunked prefill beyond latency. Fusing a compute-bound prefill chunk with eight memory-bound decodes in the same iteration is exactly the kind of mixed workload that keeps the card busy on both axes at once, using compute units the pure-decode step leaves idle. On a card whose [Infinity Cache runs cold during decode](/blog/2026-07-29-a-decode-weight-stream-has-no-reuse-so-rdna4-infinity-cache-runs-cold/) and whose [matrix cores sit out the decode loop](/blog/2026-04-30-rdna4-matrix-cores-sit-out-the-decode-loop/), a schedule that folds compute-heavy prefill work into the decode stream is not just a latency fix. It is better hardware utilization for free.

## Where scheduling sits in the stack

The [August 1 post](/blog/2026-08-01-reserve-for-max-kv-cache-fits-four-of-eight-agents-on-rdna4/) ended on the order of the memory levers: allocate the KV cache on demand so the whole swarm fits, then quantize the KV so each agent's read stays cheap, then batch continuously so the shared weights never go to waste. Those levers decide whether the eight agents can coexist at all. Chunked prefill is the lever above them. It decides whether they coexist smoothly, or whether every new prompt is a small outage for everyone else.

The order still holds. Get the agents resident, keep their reads cheap, keep the weight stream amortized, and then schedule the batch so a prefill never becomes a wall. An engine like zinc that owns its submission path can decide chunk sizes per iteration directly, which means the decode-latency-versus-time-to-first-token trade is not a config file it inherits from a serving framework but a property of its own scheduler. On a shared consumer card that control is the difference between eight agents that feel alive and eight agents that keep going quiet whenever a ninth one starts to think.
