---
title: "Why min-p has aged into the right default sampler for local Qwen3 decode"
date: "2026-05-04"
tags:
  - zinc
  - rdna4
  - amd
  - sampling
  - min-p
  - top-p
  - llama-cpp
  - qwen3
  - llm-inference
  - decoding
keywords:
  - min-p sampling local LLM
  - top-p nucleus sampling failure mode
  - llama.cpp sampler chain default
  - Qwen3 high temperature decode
  - kalomaze min-p pull request
  - dynamic truncation sampler
  - temperature after min-p
  - Schaeffer min-p reanalysis ICLR
  - local inference sampler default
  - RDNA4 decode sampler cost
excerpt: "Top-p was the default truncation sampler for six years, and on a 151,936-token vocabulary at temperature 1.4 it still occasionally hands the sampler a noise token while three good continuations sit in the same nucleus. Min-p ties the threshold to the model's own top-token probability, which behaves better at high temperature on the kinds of distributions Qwen3 actually produces. The 2025 Stanford reanalysis weakened the original paper's quality claims, but on a local 32 GB RDNA4 decode the practical answer is unchanged: ship min-p as a first-class sampler, run temperature last, and stop pretending the nucleus is doing the work."
---

A 1200-token Qwen3 chat reply on a Radeon AI PRO R9700 at temperature 1.4 with top-p set to 0.95 still occasionally emits a token that looks pulled out of nowhere. The model was confident. The top six logits covered more than half the probability mass. The sampler then drew from a long tail of tokens whose individual probabilities were each under one percent but whose cumulative weight pulled the nucleus past the 0.95 threshold. The sampler did exactly what the algorithm specified. The output read like a typo.

That failure mode is the case [Ari Holtzman's 2019 nucleus-sampling paper](https://arxiv.org/abs/1904.09751) was already aware of. The paper treats top-p as a fix for the bland-and-repetitive failure of low-temperature greedy decoding. It does not claim immunity to the opposite failure: at high temperature, even a confident distribution gets flattened enough that low-probability tokens cumulatively cross the nucleus threshold and become eligible. For a six-year-old paper that survived because the problem it solved was urgent at the time, that is a reasonable scope. For a 2026 local engine running a 151,936-token Qwen3 vocabulary at the temperatures users want for creative writing and tool-call diversity, the scope has aged poorly.

This post is the structural reason the local-inference community converged on min-p as the truncation step that runs before temperature, what the recent reanalysis from Stanford actually says about the original claims, and why a local engine on a 32 GB RDNA4 card should ship min-p as a first-class sampler regardless of which side of that argument lands.

## What top-p does well, and what it does badly

The nucleus is dynamic in the right direction. When the model is confident, the cumulative top-p threshold is reached after a handful of tokens. When the model is uncertain, the nucleus widens to include the long tail of plausible continuations. This is the property that made top-p a better default than top-k for years: top-k has to commit to a single tail size at compile time, and the right tail size depends on the entropy of the next-token distribution.

The problem is that top-p's threshold is over the wrong axis. The cumulative probability mass is held fixed at, say, 0.95, regardless of how that mass is distributed inside the nucleus. On a confident distribution where the top three tokens hold 0.93 of the mass, the nucleus grabs a fourth token whose individual probability is 0.005, plus another 0.005 and another, until the running sum crosses 0.95. The fourth, fifth, and sixth tokens may individually be hundreds of times less likely than the top one. They are still admitted. At temperature 1.0 the gap is small. At temperature 1.4 the same logits are flattened enough that the tail contains thousands of tokens that each carry a thousandth of a percent, and the nucleus of size 0.95 can pull in dozens of them.

The [original min-p PR by kalomaze on llama.cpp](https://github.com/ggml-org/llama.cpp/pull/3841) framed it the practical way: the threshold should scale with the top token, not with the tail. If the most likely token has probability 0.6, then with min-p set to 0.05 only tokens with at least 0.03 probability are admitted. If the most likely token has probability 0.05, the threshold drops to 0.0025, and the sampler is permissive in exactly the regime where the model is uncertain enough to deserve diversity. The threshold rides the model's confidence rather than chasing a fixed cumulative mass.

## What the truncation actually looks like

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-04-min-p-vs-top-p-truncation.svg" alt="Two probability-distribution panels comparing nucleus top-p sampling against min-p sampling on a Qwen3-style vocabulary at temperature 1.4. Both panels show the same bar chart of token probabilities ranked by descending value: a tall green bar for the top token at probability 0.20, then bars at 0.13, 0.09, 0.07, 0.05, and 0.04, then a long sloping tail of small bars from 0.0099 down to 0.001. The top panel shades the entire visible chart and labels it top-p admits 41 tokens, with 37 tail tokens at p under 0.01 each, indicating the cumulative-0.95 cutoff lands well past the 30 visible bars. The bottom panel shows the same distribution with a horizontal blue dashed line at min-p threshold equal to 0.05 times the top probability, which is 0.010. Only the top six bars sit above that line and are filled solid; every smaller bar is hollow. A label reads min-p admits six tokens, all with individual probability above the 0.010 cutoff." loading="lazy" />
  <figcaption>Two truncation strategies on the same flattened distribution. Top-p chases a cumulative mass and admits a long tail of low-probability tokens. Min-p sets the threshold relative to the most likely token and stops there.</figcaption>
</figure>

The figure is the entire argument in shape. The model produced a clean unimodal distribution with six good continuations sitting above a noise floor. Top-p admits the six continuations and then keeps walking down the tail until the cumulative sum crosses 0.95, by which point the sampler is choosing between dozens of tokens that each carry a fraction of a percent. Min-p stops at the floor itself.

What min-p gives up is the semi-formal guarantee that 95 percent of the model's mass is reachable. What it buys back is that no token can be sampled whose individual probability is less than 5 percent of the top token's probability, regardless of how flat the distribution becomes. On a high-temperature draw that is the property the user wanted in the first place.

## The Stanford reanalysis and what it actually says

The case for min-p does not get a free pass. The [Schaeffer, Kazdan, and Denisov-Blanch reanalysis from Stanford](https://arxiv.org/abs/2506.13681), published in mid-2025, examined the four lines of evidence in the [original min-p paper by Nguyen and coauthors](https://arxiv.org/abs/2407.01082) and found the human-evaluation results, the NLP benchmark results, and the LLM-as-a-judge results all weaker than the paper claimed once the omitted runs and statistical-test choices were corrected. The paper had been the eighteenth-highest-scoring submission to ICLR 2025 and an oral presentation. The reanalysis is not a refutation of min-p as a sampler; it is a refutation of the claim that min-p uniformly outperforms the alternatives across quality, diversity, and the explicit quality-versus-diversity frontier.

The honest read is that the paper oversold a real but narrower result. The reanalysis does not show min-p hurts. It shows the headline numbers were overstated, that some of the original benchmark wins do not survive corrected sampling-temperature handling, and that the human study had design problems the original peer review missed.

For a local engine, that reframes the decision but does not flip it. The practical case for min-p was never the benchmark wins. It was the failure-mode coverage at high temperature on a long-tailed vocabulary. That case is structural: top-p admits noise tokens at high temperature on flattened distributions. Min-p does not. Whether that translates into measurable quality gains on GPQA and AlpacaEval is the question the reanalysis answered with "less than the paper said." Whether it translates into fewer broken outputs in the seat in front of the user is the question this post is about, and that one is an architectural question, not a benchmark question.

## What this costs on an RDNA4 decode

The cost of swapping the truncation step on the sampler chain is essentially zero on the decode roofline. The [batch=1 decode roofline post](/blog/2026-04-30-rdna4-matrix-cores-sit-out-the-decode-loop) walked through why the wall-time floor on a Qwen3-30B-A3B decode step on the R9700 is set by the ~3 GB of active-weight reads at Q4_K_M and the KV cache reads scaling with context. The sampler runs after the logits land back on the host. A 151,936-entry sort plus a single threshold comparison is a few hundred microseconds on a modern CPU; the truncation walk is linear in the number of tokens above the threshold, which for any real distribution is a few thousand tokens at most.

| Sampler stage | Cost per decoded token | Share of 33 ms decode |
| --- | ---: | ---: |
| Logit copy from device | 0.6 to 1.2 ms | 2 to 4 percent |
| Repetition penalty walk | 50 to 200 us | under 0.6 percent |
| Top-p truncation | 80 to 250 us | under 0.8 percent |
| Min-p truncation | 60 to 200 us | under 0.6 percent |
| Temperature scale plus draw | 30 to 80 us | under 0.3 percent |

The numbers are noise on the decode budget. The decision to use one truncation sampler over another is not a kernel decision and there is no compute argument for either. The argument is over which one produces fewer broken tokens, and the structural argument favors min-p exactly when temperature climbs.

## The chain order matters

The detail that gets the implementation wrong is sampler ordering. The [llama.cpp default sampler chain](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) puts temperature at the end, after all truncation steps. That ordering is not a stylistic choice. Temperature has to run after min-p, because min-p compares probabilities relative to the top token, and applying temperature first redistributes the probability mass across the truncated set in a way that defeats the point. If temperature runs after min-p, the truncation has already locked in the model-confident set and only the within-set distribution gets reshaped. If temperature runs before min-p, the threshold gets computed against an already-flattened distribution and the floor moves down with the temperature, which is the failure mode the sampler was meant to avoid.

Every local-engine bug report we have seen on min-p quality reduces to one of two things: the sampler was disabled because top-p was set to a non-default value, or temperature was being applied before min-p in the chain. Neither is a min-p problem. Both are integration problems, and the fix is to keep the chain order penalties, then min-p, then temperature, and to disable top-p when min-p is on. ZINC's sampler chain takes the same shape.

## What this changes for a local engine

The takeaway for an engine that ships a default sampler is that the decision is not between two roughly equivalent truncation strategies. It is between one strategy whose failure mode shows up exactly in the high-temperature regime users are asking for and another strategy whose failure mode is harder to articulate but whose practical hits are smaller. The Stanford reanalysis tightens the claim that min-p is a uniform improvement; it does not change the failure-mode argument. The community defaults at [SillyTavern](https://github.com/SillyTavern/SillyTavern) and the [Open-WebUI integration with Ollama](https://github.com/open-webui/open-webui/issues/4278) reflect the same conclusion from a different direction: power users converge on min-p plus temperature, top-p disabled.

The harder question, for an engine that targets RDNA4 single-card decode at 30 to 50 tokens per second, is whether to take a position at all. The defensible answer is that any sampler that costs hundreds of microseconds on a thirty-three-millisecond per-token budget should be picked for behavior, not throughput, and that the behavior at high temperature on a 151,936-token vocabulary is what the user actually feels. Min-p with a default of 0.05, temperature last on the chain, and top-p disabled is the configuration that holds up across the model families a local engine has to support, and it is the configuration we ship in ZINC's default sampler today.

The next thing on this axis is whether the [DRY repetition penalty](https://github.com/ggml-org/llama.cpp/pull/9702) earns its slot before min-p on long-context generation. That is an argument for another post.
