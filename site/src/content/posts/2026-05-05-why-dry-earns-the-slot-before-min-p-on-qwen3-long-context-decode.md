---
title: "Why DRY earns the slot before min-p on Qwen3 long-context decode"
date: "2026-05-05"
tags:
  - zinc
  - rdna4
  - amd
  - sampling
  - dry
  - repetition-penalty
  - llama-cpp
  - qwen3
  - llm-inference
  - decoding
keywords:
  - DRY sampler local LLM
  - DRY repetition penalty llama.cpp
  - p-e-w DRY pull request oobabooga
  - sequence-level repetition penalty
  - dry_multiplier dry_base dry_allowed_length
  - sequence_breakers ChatML
  - classical repetition penalty Keskar CTRL
  - frequency penalty looping
  - Qwen3 long-context looping
  - sampler chain order RDNA4
excerpt: "Classical repetition penalty punishes whichever individual tokens have already shown up. Looping is not a token-level event. It is a sequence-level event, and on a 64k-context Qwen3 chat the gap between those two definitions is the difference between a coherent reply and the same paragraph rewritten seventeen times. DRY closes that gap with an exponentially scaled penalty on whatever token would extend a verbatim n-gram from earlier in the prompt, and on a 32 GB RDNA4 decode it costs effectively nothing to run before min-p."
---

A 64k-context Qwen3-30B-A3B chat on a Radeon AI PRO R9700 will, given enough turns and a high enough multiplicative repetition penalty, eventually reproduce the same six-line paragraph for the seventeenth time. The repetition penalty was at 1.15. Frequency penalty at 0.05. Top-p disabled, min-p at 0.05, temperature 0.8. Every individual token in the looping paragraph had already been seen in the prompt, so the rep-penalty pass had pushed each of their logits down by `ln(1.15) ≈ 0.14`. The model produced the loop anyway.

The reason is not subtle. Classical repetition penalty acts on the wrong unit. It punishes individual tokens that have already appeared anywhere in the context window, regardless of order, regardless of whether the new candidate would actually extend a verbatim sequence. Looping is not a token-level event. It is a sequence-level event, and the gap between those two definitions is exactly where the loops survive.

This post is the structural argument for why DRY (Don't Repeat Yourself) is the only repetition sampler with a definition that actually matches what a looping model is doing, why it slots before min-p in the sampler chain rather than after it, and what the cost looks like on a 32 GB RDNA4 single-card decode where every microsecond of host work has to earn its place.

## Why classical repetition penalty fails on a chat template

The classical repetition penalty was [introduced by Keskar and coauthors in the 2019 CTRL paper](https://arxiv.org/abs/1909.05858) as a logit-scaling step: divide the logit of every token that has appeared in the recent context by a penalty factor greater than one. The 1.15 to 1.30 range that ships as the default in most loaders is calibrated against that paper. Its design assumption is that a model is more likely to repeat a token it has already produced, so tilting the distribution away from that token reduces repetition without changing the model's grammar much.

The assumption holds for short contexts and bland baseline samplers. It does not hold for a 64k Qwen3 chat. The chat template alone forces tokens like the role markers, newlines, and special tokens to appear hundreds of times in a long session, and the repetition penalty downscales every one of them on every step. Worse, the penalty is symmetric: a closing quote that the model has used five times legitimately gets the same downscaling as a noun the model is about to fall into a loop on. The penalty distorts grammar and has nothing to say about whether the next token is the start of a fresh continuation or token seven of a verbatim repeat.

The frequency-and-presence variant from the [OpenAI sampling documentation](https://platform.openai.com/docs/api-reference/chat/create#chat-create-frequency_penalty) tightens the symmetry by counting occurrences and scaling proportionally, but it does not change the underlying definition. It is still token-level. The model can loop the same paragraph seventeen times because no individual token in the loop is more frequent than the same token would be in non-looping prose, and the cumulative penalty is still a logit shift of about 0.2, which a slightly confident continuation walks past without effort.

## What DRY actually does

The [DRY repetition penalty was introduced by Philipp Emanuel Weidmann (p-e-w) in oobabooga PR 5677](https://github.com/oobabooga/text-generation-webui/pull/5677) in March 2024 and merged that May. The reframing is small and structural. Instead of asking "has this token shown up recently," DRY asks "would this token extend a verbatim n-gram that already appears in the input." The first question can be answered by a counter. The second requires a suffix-match scan, and that is the entire engineering content of the sampler.

The mechanic, copied from the original PR: walk backward through the input from the current decode position, find the longest suffix that has a verbatim match anywhere earlier in the input, then for every token in the vocabulary, ask how long the match would be if that token were the next one. If that match length `n` is less than a threshold called `allowed_length`, no penalty is applied. Above the threshold, the logit gets a subtractive penalty equal to `multiplier * base^(n - allowed_length)`. The penalty grows exponentially in match length, which is the property that makes the sampler work.

The numbers from the [llama.cpp DRY implementation by wwoodsTM in PR 9702](https://github.com/ggml-org/llama.cpp/pull/9702), merged in October 2024, give the practical picture. The defaults the [llama.cpp server documentation](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) ships are `multiplier = 0.0` (disabled), `base = 1.75`, `allowed_length = 2`, `penalty_last_n = -1` (full context), and a `sequence_breakers` list of `\n`, `:`, `"`, and `*`. The recommended live setting is `multiplier = 0.8`. With those values, a token that would extend a five-token verbatim match earns a logit penalty of `0.8 * 1.75^3 ≈ 4.3`. Extending a nine-token match earns `0.8 * 1.75^7 ≈ 40.2`, the same number that shows up in the debug trace in the original llama.cpp PR conversation. By twelve tokens of match the penalty is over two hundred logits, which is larger than any logit in any current open-weight model.

That last point is the property that makes DRY a hard guarantee rather than a soft preference. The PR description is direct about it: with the right parameter choice, verbatim looping is mathematically impossible. The sampler does not nudge the model away from the loop; it nails the door shut.

## What the curve looks like

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-05-dry-penalty-vs-classical.svg" alt="Two panels comparing the DRY repetition penalty against the classical CTRL-style multiplicative repetition penalty as a function of match length. The left panel is a line chart with match length n on the x axis from zero to twelve and logit penalty on a logarithmic y axis from 0.01 to 1000. A green curve labeled DRY rises sharply from zero at n less than 2 to 0.8 at n equals 2, then 1.4 at n equals 3, then 4.3 at n equals 5, then 23 at n equals 8, then 70 at n equals 10, then 215 at n equals 12, traced as a smooth exponential. A red horizontal dashed line at 0.14 labeled classical repetition penalty 1.15 stays flat across the chart, far below the DRY curve at every value of n above 2. A vertical orange dashed line at n equals 2 marks the allowed length threshold. A shaded yellow region from n equals 8 to n equals 12 is labeled looping zone. The right panel shows a token sequence schematic. The top row shows an earlier prompt span containing the token sequence the cat sat on the mat. The bottom row shows the current decode position with the sequence the cat sat on the highlighted in green as a five-token verbatim match against the earlier span, followed by a candidate next token mat highlighted in red with the annotation match would extend to length 6, DRY penalty equals 0.8 times 1.75 to the fourth equals 7.5 logits, classical repetition penalty equals 0.14 logits." loading="lazy" />
  <figcaption>Penalty magnitude as a function of verbatim-match length, on a logarithmic scale. Classical repetition penalty is flat at about 0.14 logits regardless of context. DRY climbs through three orders of magnitude across realistic match lengths and crosses the maximum logit any current open-weight model produces well before what a human reader would call a loop.</figcaption>
</figure>

The figure carries the entire argument. The classical penalty is a flat line that does not know how long the matching sequence is, so it cannot distinguish a quote that legitimately repeats from a paragraph that is restarting verbatim. DRY scales with the match length that the sampler can actually measure, so its penalty is small when the match is short and unbounded when the match is long. A model cannot win a fight with `1.75^n`.

## The sequence-breakers detail nobody mentions

DRY has one piece of integration cost that the classical penalty does not. A naive implementation immediately runs into the chat template. The ChatML preamble that precedes every assistant message is a ten-to-twelve token sequence that, by construction, occurs verbatim before every prior assistant turn. Without an escape hatch, the first content token of every assistant reply hits a match length of eleven or twelve and gets penalized by hundreds of logits, which guarantees the model never starts a message the same way twice.

The escape hatch is `sequence_breakers`, a list of tokens that interrupt the suffix-match scan. The defaults are tuned for English chat: `\n` and `:` and `"` and `*`. The chat template ends with a newline before the assistant's first content token, so the suffix-match cannot walk back across that newline. The match length resets on every break.

The detail that bites engines integrating DRY for the first time is that the sequence-breaker list has to be tokenized using the loaded model's tokenizer, not as raw bytes. A breaker that maps to multi-byte tokens in one tokenizer and to single-byte tokens in another will silently not break sequences, and the symptom is that the sampler appears to "not work on this model." The `sequence_breakers` field in the llama.cpp server API takes a JSON array of strings exactly so the engine can re-tokenize them per model.

## The sampler chain order again

The [previous post on min-p as the right default truncation sampler](/blog/2026-05-04-why-min-p-is-the-right-default-sampler-for-local-qwen3-decode) made the case that temperature has to run after min-p on the sampler chain. DRY adds a second ordering constraint: it has to run before min-p, not after.

The reason is that DRY's penalty is in logit space. It subtracts a value from the candidate token's logit before any truncation has happened. If min-p runs first, it produces a small set of candidate tokens that the model thinks are likely. If DRY runs after that, on a confident distribution the candidate that would extend a verbatim match is exactly the candidate min-p kept, and DRY's job is to push it out of the kept set. If min-p has already produced a single-element set, DRY is stuck applying a penalty that only changes which single token gets sampled, and on the looping case that single token is the loop continuation. Running DRY first, then min-p, lets the truncation step see the post-DRY distribution and pick a different candidate that is actually likely under the model and not on a loop.

The recommended chain order, which matches the [llama.cpp server sampler chain](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) when DRY is enabled, is: penalties (frequency, repetition, presence) in whatever roles they still play, then DRY, then min-p, then temperature, then the sample. ZINC's sampler chain takes the same shape on by default for Qwen3 and llama-class models.

## What this costs on an RDNA4 decode

The cost concern with DRY is that the suffix-match scan looks superficially quadratic. The naive implementation walks backward through the input on every decode step, comparing the current suffix against every earlier window. On a 64k-context decode, that would be a million-comparison loop per token, which would dominate the per-token budget on a 30 ms decode step.

The original p-e-w PR addressed this with a Z-algorithm implementation that does the scan in linear time. The llama.cpp port keeps the same structure but caps the maximum match length, on the observation that the penalty is exponential and the value past length forty is already in the range of `10^9` and irrelevantly large; capping makes no behavioral difference. The cost on the host CPU is on the order of fifty microseconds per decode token at 64k context, which sits in the same fraction-of-a-percent range as min-p on the [decode roofline](/blog/2026-04-30-rdna4-matrix-cores-sit-out-the-decode-loop).

| Sampler stage at 64k context | Cost per decoded token | Share of 33 ms decode |
| --- | ---: | ---: |
| Logit copy from device | 0.8 to 1.4 ms | 2.4 to 4.2 percent |
| DRY suffix-match scan (capped at 50) | 40 to 80 us | under 0.25 percent |
| Min-p truncation | 60 to 200 us | under 0.6 percent |
| Temperature scale plus draw | 30 to 80 us | under 0.3 percent |

The numbers are noise on the decode budget. The decision to slot DRY in at all is a behavioral decision, not a throughput decision, which is the same shape the min-p argument took yesterday. If a sampler costs tens of microseconds and prevents a class of failures the alternative cannot prevent at any price, it is the correct default.

## What DRY does not catch

The honest part of the argument is what the sampler does not solve. DRY catches verbatim repetition. It does not catch paraphrastic repetition: the model rewriting the same paragraph using slightly different wording each time. It does not catch situational repetition: the model returning to the same topic across turns without copying the surface form. It does not help with attention sinks or with the broken-state failures from KV eviction.

The other half of the honest argument is that paraphrastic repetition is much rarer than verbatim repetition on current-gen open-weight models, and a chat that does not loop verbatim is a chat that mostly does not loop. The PR description from p-e-w is direct about this and the empirical record on the [SillyTavern](https://github.com/SillyTavern/SillyTavern) and [oobabooga textgen wiki Parameters Tab](https://github.com/oobabooga/text-generation-webui/wiki/03-%E2%80%90-Parameters-Tab) communities matches: turn DRY on, drop the classical repetition penalty back to 1.0, and the loops the user actually sees in production almost entirely disappear.

## What this changes for ZINC

The engine-level takeaway is that the default sampler chain for Qwen3-class models on a 32 GB RDNA4 card is not "tune the repetition penalty until the loops stop." It is "drop the classical repetition penalty to 1.0, turn DRY on with `multiplier = 0.8`, `base = 1.75`, `allowed_length = 2`, and the engine's tokenized sequence-breaker set, run min-p at 0.05 after DRY, run temperature last." That is the configuration we ship today. It costs the host CPU under a hundred microseconds per token at 64k context, and it removes the failure mode that the classical penalty was sold as fixing and never actually did.

The thing on the horizon worth watching, in the same spirit as the [Quest query-aware page selection](https://arxiv.org/abs/2406.10774) note from the attention-sinks post, is whether any architecture-level training fix lands that makes the looping behavior less likely at the source. The [original p-e-w PR](https://github.com/oobabooga/text-generation-webui/pull/5677) treats DRY as a sampling fix for a training-side problem. That framing is still right. Until the training side ships a fix, the sampler chain is where the loops have to die, and DRY is the only step on it that actually has the right shape.
