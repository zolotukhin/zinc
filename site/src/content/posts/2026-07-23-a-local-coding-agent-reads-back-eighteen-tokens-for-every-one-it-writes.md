---
title: "A local coding agent reads back eighteen tokens for every one it writes"
seoTitle: "Tool Observations Are the Hidden Prefill in Local Agent Loops on RDNA4"
date: "2026-07-23"
tags:
  - zinc
  - amd
  - rdna4
  - rx-9070-xt
  - agents
  - prefill
  - decode
  - kv-cache
  - react
  - tool-calling
  - scheduling
  - local-llm
  - llm-inference
keywords:
  - agent tool observation prefill
  - ReAct loop token asymmetry
  - local LLM agent context growth
  - tool output size KV cache
  - RX 9070 XT prefill decode agent
  - SWE-agent observation truncation
  - chunked prefill tool result local LLM
  - agent KV budget overflow single GPU
  - Qwen3.5-9B agent step wall clock
  - tool result re-prefill cost RDNA4
excerpt: "A coding agent spends its decode budget writing a short thought and a compact tool call, then the harness hands back a file, a grep, or a test log that is far larger than anything the model wrote. That observation re-enters the context as prefill, and on a single local card it is the token traffic that actually fills the KV cache and drives the wall clock."
seoDescription: "In a local agent loop the model writes little per step, a thought plus a tool call, while the tool feeds back a much larger observation that must be prefilled before the next step. Modeled on Qwen3.5-9B on one RX 9070 XT at 962 tok/s prefill and 39.6 tok/s decode, a typical step writes about 110 tokens and reads back about 2,000, an 18-to-1 asymmetry. Those observation tokens dominate the context, drive the per-step wall clock, and overflow the roughly 55,000-token single-card KV budget well before the model's own output would, which is why observation shaping and KV swap are systems levers, not prompt hygiene."
faqs:
  - question: "Why does a local agent read back more tokens than it generates?"
    answer: "Because the model's output per step is small and the tool's output is not. A ReAct step is usually a short reasoning trace plus a compact tool call, on the order of a hundred tokens. The tool then returns an observation: a source file, a grep result, a test log, a web response. Those are routinely thousands of tokens, and every one of them is appended to the context and prefilled before the model runs again. Modeled on Qwen3.5-9B, a step that writes about 110 tokens reads back about 2,000, roughly 18 to 1."
  - question: "Why does the tool observation cost prefill and not decode?"
    answer: "Decode generates one token at a time and reads the whole model per token, so it is memory-bandwidth bound and slow, about 39.6 tokens per second on one RX 9070 XT. Prefill ingests already-known tokens in parallel and is compute bound, about 962 tokens per second on the same card. A tool observation is text the model did not generate, so it enters through the prefill path. A 2,000-token observation is about 2.08 seconds of prefill, on the order of the 2.78 seconds the model spent decoding the step that called the tool."
  - question: "Why do observations overflow the KV cache before the model's own output does?"
    answer: "Because they are 18 times larger per step. On a 16 GB card ZINC has room for about 55,000 fp16 KV tokens after weights. If each step adds about 110 model tokens and about 2,000 observation tokens on top of a 4,000-token system prompt, the running context crosses 55,000 tokens near step 24, and about 95 percent of what filled it came from tools, not the model. That is why the KV pressure that forces a swap or eviction is driven by observation size."
  - question: "Can you just truncate tool outputs to fix this?"
    answer: "Partly, and it is the right first move. SWE-agent's agent-computer interface deliberately shapes observations, windowing file views and suppressing verbose command output, and it improves both quality and cost. But truncation trades context for correctness. Cut a test log too aggressively and the agent loses the stack trace it needed. The systems answer is to treat observation prefill as a first-class cost: keep observations tight, prefill them in small chunks so they do not stall other agents, and swap the resulting KV rather than recompute it."
  - question: "Does the swarm make observation prefill worse?"
    answer: "It sharpens it. Eight agents each hit a tool boundary on their own schedule, so at any moment one of them is likely to drop a multi-thousand-token observation that needs prefilling while the other seven are trying to decode. That is the chunked-prefill admission tension, except it recurs on every tool call rather than once when an agent joins. Small prefill chunks keep a large observation from freezing the swarm."
draft: false
---

A coding agent does not spend most of its context window on its own words. Watch one step of an agent loop and the model writes something short, a line or two of reasoning and a compact tool call, and then the harness hands back something much larger: the contents of a file, a few hundred lines of `grep` output, a failed test run with a stack trace. The model wrote a hundred tokens. It is about to read back a couple thousand.

That asymmetry is easy to miss because it does not show up in the part of the loop we usually profile. We measure how fast the model decodes, because decode is the slow part and the [wall clock of a chat turn is almost entirely decode](/blog/2026-07-14-a-qwen3-5-9b-chat-turn-spends-most-of-its-wall-clock-in-decode/). An agent is not a chat turn. It interleaves generation with tool calls, and every tool call injects text the model never generated. On a single local card, that injected text is the traffic that actually fills the KV cache and sets the pace.

The pattern goes back to the framing that made tool-using agents work in the first place. [ReAct](https://arxiv.org/abs/2210.03629) interleaves a reasoning trace with an action and an observation from the environment, so the model reasons, acts, and then reads what the action returned before reasoning again. The reasoning and the action are what the model produces. The observation is what the world produces, and the world is not concise.

## The model writes a little, the tool returns a lot

Put rough sizes on one step. A thought is a sentence or two, maybe 80 tokens. A tool call is a small JSON object, a name and a few arguments, maybe 30 tokens. So the model emits on the order of 110 tokens per step, and it emits them through the decode path, one at a time, reading the full model for each. On one RX 9070 XT running Qwen3.5-9B, decode is about [39.6 tokens per second](/blog/2026-07-15-weight-streaming-is-under-half-of-an-rdna4-decode-token/), so those 110 tokens cost about 2.78 seconds.

Now the tool runs and returns an observation. A `cat` of a 300-line source file is around 3,600 tokens. A `grep -rn` across a repository is a few hundred to a thousand. A `pytest` run that fails prints its captured output and a traceback, easily one to two thousand tokens. A web fetch or an API response can be larger still. Call a typical observation 2,000 tokens, which is conservative for code work.

That observation is text the model did not generate, so it does not go through decode. It goes through prefill, the compute-bound path that ingests known tokens in parallel at about 962 tokens per second on the same card. Two thousand tokens of observation is about 2.08 seconds of prefill, roughly the same as the 2.78 seconds the model spent generating the step that called the tool. The read-back is not a rounding error next to the generation. It is the same order of magnitude, and it happens every single step.

| One ReAct step, Qwen3.5-9B on RX 9070 XT | Tokens | Path | Wall clock |
| --- | ---: | --- | ---: |
| Thought plus tool call, the model writes | 110 | decode @ 39.6 tok/s | 2.78 s |
| Tool observation, the harness feeds back | 2,000 | prefill @ 962 tok/s | 2.08 s |
| Net added to the KV cache, kept for the task | 2,110 | resident | taxes every later token |

Read the middle row against the top. The model spends its slow decode budget producing 110 tokens, and the tool answers with roughly eighteen times as many, which then have to be prefilled before the next thought can start. The bottom row is the part that keeps costing after the step is over: all 2,110 tokens now sit in the KV cache for the rest of the task, and every future decode token has to read past them.

## The context is mostly things the model never said

Follow that forward over a real task. If a session runs 30 steps, the model has written about 3,300 tokens total, and the tools have fed back about 60,000. Add a 4,000-token system prompt and tool schema, and the context the card is holding is around 67,000 tokens, of which roughly 95 percent is observation text the model never generated. The agent's own output is a thin band riding on top of a much larger pile of tool results.

This matters on a local card specifically because the KV cache is the constraint. On a 16 GB card, ZINC has room for about [55,000 fp16 KV tokens after weights](/blog/2026-07-20-swapping-an-idle-agents-kv-cache-beats-recomputing-it-by-177x/). A task that adds 2,110 tokens per step crosses that budget near step 24, and it crosses it because of the observations, not the reasoning. If the model's output were all that accumulated, the same card would hold hundreds of steps. The tool outputs are what run it out of room.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-07-23-agent-tool-observation-prefill-asymmetry.svg" alt="A two-panel diagram on a warm dark charcoal-brown background. The left panel, titled 'One ReAct step: tokens the model writes vs tokens the tool feeds back', is five horizontal bars sharing a token axis from 0 to 4,000. The top bar, in amber and labelled 'thought plus tool call, the model writes', is tiny at 110 tokens. Below it four longer terracotta bars show tool observations that must be prefilled: 'cat a 300-line source file' at 3,600 tokens, 'grep -rn across the repo' at 900 tokens, 'pytest run log with traceback' at 1,500 tokens, and 'web fetch or API result' at 4,000 tokens. A bracket spans the amber bar and a dashed 2,000-token 'typical observation' marker, annotated 'about 18 tokens read back for every 1 written'. The right panel, titled 'Cumulative context over a 30-step task', is a stacked area chart with steps 0 to 30 on the horizontal axis and cumulative context tokens 0 to 70,000 on the vertical axis. A thin flat slate band at the bottom is the 4,000-token system prompt. A thin amber band above it, labelled 'model output, about 5 percent', grows slowly to 3,300 tokens. A large terracotta area above that, labelled 'tool observations, about 95 percent', grows to 60,000 tokens, so the total reaches 67,300 tokens at step 30. A dashed horizontal line at 55,000 tokens is labelled 'single-card KV budget, 16 GB', and a marker shows the total crossing it near step 24, annotated 'observations overflow the budget, not the model output'." loading="lazy" />
  <figcaption>Modeled for Qwen3.5-9B on one RX 9070 XT. The 962 tok/s prefill and 39.6 tok/s decode rates are measured; the token counts per tool are representative sizes for code work. Left: a step writes about 110 tokens and reads back roughly 2,000. Right: those observations are what push the running context past the card's KV budget, near step 24, not the model's own output.</figcaption>
</figure>

The diagram is a model built on two measured rates and a set of representative tool sizes, not a profiler trace of one session. The exact crossing point moves with how chatty the tools are. The shape does not. As long as an observation is much larger than a thought plus a call, the context fills with tool output, and the KV pressure that eventually forces a [swap or an eviction](/blog/2026-07-20-swapping-an-idle-agents-kv-cache-beats-recomputing-it-by-177x/) is driven by things the model read, not things it wrote.

## Why this lands harder on a swarm

A single agent pays the observation prefill in series, one spike per tool call, and mostly it just waits. A swarm pays it differently. Running [eight agents on one card](/blog/2026-07-18-eight-concurrent-agents-pull-5-6x-more-tokens-from-one-rx-9070-xt/) means eight independent loops each hitting their own tool boundaries, so at almost any moment one of them has just dropped a two-thousand-token observation that needs prefilling while the other seven are trying to decode.

That is the same collision as the [chunked-prefill admission problem](/blog/2026-07-19-admitting-a-ninth-agent-stalls-the-other-eight-for-8-3-seconds/), from a new direction. Admitting a fresh agent means prefilling its prompt while running agents decode, and if you prefill it in one big batch the running agents stall. A tool observation is a prefill of the same kind, except it does not happen once when an agent joins. It happens every time any agent calls a tool, which for a coding loop is constantly. The [vLLM chunked-prefill scheduler](https://docs.vllm.ai/en/stable/configuration/optimization/) exists for exactly this reason: it prioritizes decode and chunks large prefills so they interleave instead of blocking, tuned by `max_num_batched_tokens`. A local agent engine needs that behavior applied to observations, not just to prompts.

There is also a sharing asymmetry worth naming. The [system prompt is shared](/blog/2026-07-21-the-system-prompt-a-local-agent-swarm-caches-eight-times-over/) across all eight agents, so it is stored once and its prefill is amortized. Observations are the opposite. Each agent's tool results are unique to its trajectory, so they cannot be deduplicated across the swarm. The part of the context that is common is cheap and paid once. The part that dominates the context is per-agent and paid every step.

## Observation shaping is a systems lever, not prompt hygiene

The obvious response is to make the tools return less. This is real, and the best evidence for it is that a state-of-the-art agent already does it deliberately. [SWE-agent](https://arxiv.org/abs/2405.15793) built a custom agent-computer interface whose whole thesis is that the interface, including the shape and size of what the tools return, changes how well the agent performs. Their file viewer shows a bounded window rather than dumping a whole file, and verbose command output is suppressed or summarized. They found this helps quality. It also happens to cut exactly the observation prefill this post is about.

The catch is that truncation trades context for correctness, and the trade is not free. Clip a stack trace and the agent loses the line it needed to fix the bug. Window a file too tightly and it pages back and forth, spending more steps and, ironically, more total observation tokens. So observation size is not something to minimize blindly. It is something to treat as a cost with a known price, about one prefill token at 962 per second going in and one KV slot held for the rest of the task, and to spend deliberately.

That reframes three things the last few posts treated separately. Keeping observations tight is worth real effort because each token is paid twice, once as prefill and once as permanent KV. Prefilling observations in [small chunks](/blog/2026-07-19-admitting-a-ninth-agent-stalls-the-other-eight-for-8-3-seconds/) matters because a big observation dropped into a busy swarm freezes the agents that are decoding. And [swapping idle KV](/blog/2026-07-20-swapping-an-idle-agents-kv-cache-beats-recomputing-it-by-177x/) instead of recomputing it matters most for observation-heavy contexts, because recomputing means re-prefilling thousands of tokens the model never generated and could not regenerate.

## What I am building toward

ZINC schedules prefill and decode as if prompts were the only prefill in the system. For an agent loop that assumption is wrong. The dominant prefill is not the prompt the agent starts with. It is the stream of observations it reads back, one per tool call, each larger than the reasoning that requested it, and the scheduler has to treat those observations as first-class prefill work: chunk them, interleave them with the swarm's decode, and account for the KV they leave behind.

The number to keep in mind is the ratio. Roughly eighteen tokens read back for every token written, on a single local card, for a workload that is supposed to be decode-bound. It is decode-bound only if you ignore where the context comes from. Once you count the observations, a local coding agent is a prefill workload wearing a decode workload's clothes, and the engine that serves it well will be the one that schedules the reading, not just the writing.
