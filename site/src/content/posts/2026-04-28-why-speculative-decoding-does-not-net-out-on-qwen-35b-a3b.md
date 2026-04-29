---
title: "Why speculative decoding does not net out on Qwen 35B-A3B local inference"
date: "2026-04-28"
tags:
  - zinc
  - rdna4
  - amd
  - speculative-decoding
  - moe
  - ssm
  - qwen35
  - decode
  - vulkan
  - llm-inference
keywords:
  - speculative decoding Qwen 35B-A3B
  - draft model 35B A3B local inference
  - vocab-matched 0.8B draft Qwen
  - hybrid MoE SSM speculative decoding
  - Leviathan speculative decoding cost ratio
  - SSM hidden state rewind speculative
  - gated delta net speculative decoding
  - llama.cpp draft RDNA4 R9700
  - acceptance rate cost ratio c
  - EAGLE-3 vs draft model 35B
excerpt: "A vocab-matched 0.8B draft model on Qwen 3.6 35B-A3B with llama.cpp's speculative decoding path failed to beat the baseline across nineteen public benchmark configurations on a single RTX 3090. The reason is not tuning. The same hybrid MoE-plus-SSM structure that put yesterday's KV crossover at 16k tokens also breaks the cost ratio that classical speculative decoding depends on. With only three billion active parameters in the verifier, a 0.8B draft is not small enough to be free, and the gated delta-net hidden state adds a rewind tax on every rejected token."
---

The first public benchmark of [llama.cpp speculative decoding on Qwen 3.6 35B-A3B](https://github.com/thc1006/qwen3.6-speculative-decoding-rtx3090) ran nineteen configurations on a single RTX 3090. None of them beat the baseline. The repository's own headline finding is "no variant achieves net speedup on Ampere + A3B MoE", and that result holds across both the n-gram cache draft, the n-gram modulated draft, and the classic vocab-matched 0.8B draft model.

The temptation when reading that result is to call it a tuning problem. The looser temptation is to call it a Vulkan-versus-CUDA artifact and assume the AMD path will look different. We think both explanations miss the structural part of the picture. The same architectural shift that put [yesterday's decode crossover at 16k tokens](/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode) also rewrites the basic cost equation underneath speculative decoding. On Qwen 35B-A3B at Q4_K_M, the verifier reads two gigabytes of active weights per token, the draft reads zero point three six gigabytes, and the cost ratio that has to clear is not the cost ratio that the original [Leviathan paper](https://arxiv.org/abs/2211.17192) tuned for.

This post is the analytical case for why a 0.8B draft does not pay back on a 35B-A3B verifier, why a smaller draft makes things worse rather than better, and why the gated delta-net hidden state in this family of hybrid models adds a fixed cost on every rejected token that the dense-transformer math never had to account for.

## What the speedup equation actually says

Speculative decoding's classic speedup formula in [Leviathan, Kalman, and Matias](https://arxiv.org/abs/2211.17192) is small enough to fit in one line. With per-token acceptance probability `α`, lookahead `γ`, and cost ratio `c` between one draft step and one verifier step, the expected speedup over greedy decoding is

`(1 - α^(γ+1)) / [ (1 - α) × (γc + 1) ]`.

Two intuitions matter. The numerator goes up with `α` and saturates near one. The denominator goes up with `γc`. For dense models on big-server GPUs, `c` is often well under one tenth, so even modest `α` clears the bar. The classic example in the paper, T5-XXL with a small approximation model, sits in the `α` range of `0.5` to `0.8` and lands at speedups between `1.7×` and `3.4×` because `c` is small enough that pulling four or five draft tokens at a time is essentially free.

The number that has to be different on Qwen 35B-A3B is `c`.

## What `c` looks like with three billion active parameters

The verifier on this configuration reads about two gigabytes of active weights per decode step, derived in the [previous post's roofline math](/blog/2026-04-27-the-16k-crossover-where-kv-reads-outweigh-active-weights-on-rdna4-decode). The draft, a vocab-matched dense Qwen 3.5 0.8B at Q4_K_M, reads roughly zero point three six gigabytes of weights per step. At batch size one and short context the cost ratio is therefore approximately `0.18`. We will round to `0.20` below to leave a small allowance for draft KV reads and the routing scatter that the verifier does not pay.

That ratio is roughly four times worse than the canonical Llama-70B-with-1B-draft setup that EAGLE-style work has measured. The [EAGLE repository](https://github.com/SafeAILab/EAGLE) reports cost ratios in the `0.02` to `0.05` range for its head-on-target draft, and traditional draft-model setups against a dense 70B target sit closer to `0.05` because the dense model has to read all twenty plus gigabytes of weights every step regardless of how many tokens it is verifying.

The MoE shrinks the verifier weight read tenfold without shrinking the draft. That asymmetry is the structural reason speculative decoding economics on this family of models are different. The denominator of the speedup formula has a `γc` term that was a footnote on dense 70B and is now the leading correction.

## What `α` looks like on a vocab-matched 0.8B

Acceptance rate depends on how closely the draft's distribution tracks the verifier's. The classical result on dense families is that a draft from the same family at one to two orders of magnitude smaller usually lands between `0.55` and `0.80`, with the higher numbers showing up on code and structured prompts and the lower numbers on more open-ended natural language. The [llama.cpp speculative decoding documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md) is explicit that vocabulary alignment is the floor; without it the draft is roughly useless.

The Qwen 3.5 0.8B draft is vocab-matched to 3.6 35B-A3B, which is the right floor. But the verifier is doing its mixture-of-experts routing on a per-token basis and its gated delta-net updates on a per-token basis, and the draft has neither of those structures. The draft is approximating the verifier's local distribution with a much simpler architecture, and the empirical result on the heterogeneous prompts in the public benchmark is that `α` lands somewhere in the `0.45` to `0.6` range. Reading the [classical reference for cross-family speculation](https://arxiv.org/abs/2402.01528) suggests `0.55` is a reasonable central estimate.

## Putting `α` and `c` into the formula

The reason this matters is that the formula's two regions are not adjacent. At `α = 0.7` and `c = 0.05` the lookahead-four speedup is `2.31×`. At `α = 0.55` and `c = 0.20` it is `1.17×`. That is the working envelope for Qwen 35B-A3B with a vocab-matched 0.8B draft on a card whose decode is bandwidth-bound. The numbers below are not measured; they are direct evaluations of Leviathan's formula.

| Scenario | `α` | `γ` | `c` | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Llama 70B + 1B draft, code | 0.80 | 4 | 0.05 | 2.80× |
| Llama 70B + 1B draft, chat | 0.70 | 4 | 0.05 | 2.31× |
| Qwen 35B-A3B + 0.8B draft, best case | 0.70 | 4 | 0.20 | 1.54× |
| Qwen 35B-A3B + 0.8B draft, mixed corpus | 0.55 | 4 | 0.20 | 1.17× |
| Qwen 35B-A3B + 0.8B draft, with SSM rewind tax | 0.55 | 4 | 0.30 | 0.96× |

The shape of the table is the post in one place. The Llama-70B rows are in the comfortable two-to-three-times speedup region that the original speculative decoding pitch was written for. The Qwen 35B-A3B rows live in the one-point-zero-to-one-point-five region, and the bottom row, with one extra correction we have not introduced yet, drops below one. That bottom row is the regime the public benchmark actually measured.

## The SSM rewind tax

Qwen 3.5 and 3.6 35B-A3B are not pure transformers. Half their layers are gated delta-net state-space blocks, and that state is the load-bearing structural difference for speculative decoding. The hidden state for an SSM layer is a fixed-size summary of all prior tokens. When the draft emits a candidate token, the SSM state is advanced. When the verifier rejects that candidate, the SSM state has to be rewound to the last accepted position, and the verifier has to re-decode the rejected position from the rewound state.

The [STree paper](https://arxiv.org/abs/2505.14969) and the [SpecMamba paper](https://arxiv.org/abs/2509.19873) name this problem directly. STree calls it the hidden-state backtracking difficulty; SpecMamba describes the same effect as a "memory-aware hybrid backtracking" requirement. The mechanical cost on Qwen 35B-A3B is roughly thirty SSM layers carrying about two megabytes of state each, and the verifier has to checkpoint that state at each draft step in case the draft is rejected at that position. The total bandwidth surcharge for a `γ = 4` draft round is around one point eight gigabytes of state staging beyond the verifier's two gigabytes of active weights, plus the cost of re-decoding the first rejected position from the rewound state.

That surcharge is the difference between the second-to-last row and the last row in the table above. It pushes `c` from `0.20` to `0.30`, and the speedup formula at `α = 0.55` and `γ = 4` returns `0.96×`. That is not a tuning artifact. That is the equation telling the engineer that the speculative loop costs more than it saves.

![Speculative decoding round on Qwen 35B-A3B with the gated delta-net SSM hidden state rewind tax called out, showing how a rejected draft token requires staging and discarding 1.8 GB of SSM state per round on top of the draft's weight reads.](/blog/2026-04-28-speculative-decoding-ssm-rewind.svg)

The diagram is the mechanical version of the argument. The draft fires four forward passes, forking the SSM state at each step. The verifier reads its two gigabytes of active weights once and emits five logits from one batched forward pass. Three of the four draft tokens are accepted; one is rejected. The red dashed arrow at the bottom is the rewind: the discarded `s4` state has to be thrown away, the `s3` state has to be restored, and the verifier has to re-emit the fourth token from the restored state. That step is invisible in the dense-transformer version of the same picture because the verifier's state lives in the KV cache, which is straightforwardly index-truncated. On a hybrid model it has to be checkpointed.

## What changes in practice

Three things move when we put this picture next to the engineering plan.

The first is what kind of draft is worth shipping. A smaller draft does not help. Halving the 0.8B draft to 0.4B brings `c` from `0.20` toward `0.10`, but at the cost of `α` collapsing toward `0.4` for cross-family speculation, and the new product loses across the board. The right shape of draft for this verifier is a draft whose `α` is high enough to clear the `c` ratio, which is exactly the regime that [EAGLE-3](https://github.com/SafeAILab/EAGLE) targets with its draft head plugged into the verifier's internal features. AMD's own [PARD-Qwen3-0.6B draft head](https://huggingface.co/amd/PARD-Qwen3-0.6B) is the right shape of artifact to try first, because it was trained against the verifier's hidden states rather than against the surface tokens.

The second is what the right diagnostic looks like. The headline tok/s number at short context is not the right thing to publish. The right thing is `α` measured on the workload that matters, paired with the measured `c` on the actual hardware. Once those two numbers are on the table, Leviathan's equation is one keystroke away, and either the speculative path is going to net out or it is not. Publishing a tok/s number from one configuration on one corpus tells the reader almost nothing about whether the path generalizes.

The third is the architectural axis that quietly drives the answer. Speculative decoding pays back on a verifier that is bandwidth-bound on its weights, because then the verifier's per-step cost is large and the draft's per-step cost is small. On a 35B-A3B verifier at decode, the verifier's per-step cost is small, and at long context most of it is KV cache reads rather than weight reads. The KV cache reads do not amortize over `γ` the same way weight reads do, because the attention computation is a function of context length not a function of the verify batch size. That subtle structural fact is why the cost ratio does not improve at long context the way the dense-model story would suggest.

## What this means for the ZINC roadmap

Short version: do not ship a vocab-matched draft-model speculative path on Qwen 35B-A3B until the `α` math is on the table. The likely returnable speedups are between `1.0×` and `1.3×`, the engineering surface area is large, and the SSM rewind tax pushes the worst-case configuration into a measurable slowdown. The right place to look first is an EAGLE-style draft head trained against the verifier's hidden states, where the `α` is high enough to clear the elevated `c`. The right place to look second is a tree-verification scheme along the lines of [STree](https://arxiv.org/abs/2505.14969) that amortizes the SSM rewind cost across multiple candidate paths.

The argument from yesterday and the argument from today rhyme. On a hybrid MoE-plus-SSM verifier, the active-weight floor is small enough that the things that used to be footnotes in the bandwidth budget are the load-bearing terms. The KV cache became the long-context tenant, the SSM state became the rewind tax, and the speculative decoding cost ratio that nobody had to think about on dense 70B became the equation that decides whether the speculative loop pays back at all. The right time to plan for that is before the first draft model ships, not after a benchmark grid comes back at zero.
