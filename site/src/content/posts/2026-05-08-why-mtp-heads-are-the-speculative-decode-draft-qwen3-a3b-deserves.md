---
title: "Why MTP heads are the speculative decode draft Qwen3 A3B always needed on RDNA4"
seoTitle: "MTP Speculative Decoding for Qwen3"
date: "2026-05-08"
tags:
  - zinc
  - rdna4
  - amd
  - speculative-decoding
  - mtp
  - multi-token-prediction
  - qwen3
  - qwen3-next
  - llama-cpp
  - vllm
  - llm-inference
keywords:
  - MTP head speculative decoding local
  - Qwen3-Next multi-token prediction RDNA4
  - llama.cpp PR 22673 mtp draft
  - GDN partial rollback PR 22400
  - hidden-state aligned draft Qwen3
  - qwen3_next_mtp vllm speculative-config
  - Leviathan speedup formula MTP
  - draft-model vs MTP cost ratio
  - 27B dense Qwen MTP 2.5x decode
  - SSM rewind tax speculative decode
excerpt: "The April 28 argument that draft-model speculative decoding does not net out on Qwen 35B-A3B was the right read at the time. Two pieces moved in the next ten days. PR 22400 made gated DeltaNet rollback partial instead of full, eliminating the SSM rewind tax. PR 22673 wired MTP heads into llama.cpp as a built-in draft, and the measured speedup on a 27B dense Qwen checkpoint is 2.5x at γ=3 with a 0.72 acceptance rate. The cost ratio that ruined the 0.8B draft drops by an order of magnitude when the draft is one transformer layer attached to the verifier's last hidden state, and on a 32 GB RDNA4 card the only thing standing between local Qwen3 and that 2.5x is a Vulkan kernel that has not landed yet."
seoDescription: "Why MTP heads are the right speculative decoding draft for Qwen3: hidden-state alignment, low cost ratio, and partial Gated DeltaNet rollback."
---

Quick answer: MTP heads are a better speculative decoding draft for Qwen3-style models because they are trained against the verifier's hidden state and cost roughly one extra layer, not a separate dense draft model. That fixes the cost-ratio problem that makes a vocab-matched 0.8B draft unattractive on hybrid MoE/SSM verifiers.

The [argument we made on April 28](/blog/2026-04-28-why-speculative-decoding-does-not-net-out-on-qwen-35b-a3b/) was that a vocab-matched 0.8B draft model could not pay for itself on Qwen 35B-A3B. The Leviathan speedup formula said so directly: at acceptance rate `α = 0.55`, lookahead `γ = 4`, and cost ratio `c = 0.20`, the expected speedup was `1.17×`, and the gated DeltaNet rewind tax pushed `c` to `0.30`, dropping the formula to `0.96×`. The public benchmark on a 3090 measured exactly that: zero speedup across nineteen configurations.

Two things moved in the ten days that followed. The first was [PR 22400 by am17an in llama.cpp](https://github.com/ggml-org/llama.cpp/pull/22400), which kept gated DeltaNet intermediates per token so a rejected draft only triggers a partial rollback instead of a checkpoint restore. On a 5090 with a Q4_K_M Qwen3.5-27B target and a Q8_0 Qwen3.5-0.8B draft, the same configuration that lost on master shipped a `1.78×` speedup, going from 70.95 to 126.06 tok/s on a doubly-linked-list code prompt. The SSM rewind tax was real, but it was structurally fixable.

The second was [PR 22673 by the same author](https://github.com/ggml-org/llama.cpp/pull/22673), opened May 4, which wired Multi-Token Prediction heads into the speculative path as a built-in draft. The measured aggregate on a 27B dense Qwen checkpoint at Q8 across nine prompt categories was `2.5×` over baseline at γ=3 with a 0.72 acceptance rate, and `2.4×` at γ=2 with a 0.83 acceptance rate. Neither configuration shipped a separate draft model file; the draft was the MTP head loaded from the same GGUF.

Together those two changes inverted the conclusion of the April 28 post. Speculative decoding does net out on this family of models. It just does not net out with the draft you reach for first.

## What an MTP head is, briefly

Multi-Token Prediction was [introduced as a training objective by Gloeckle and coauthors at ICML 2024](https://arxiv.org/abs/2404.19737). The idea is that at every position the model emits next-token logits as usual, plus a small number of additional heads that predict the token at offset `+2`, `+3`, and so on. The recommended setup in the paper was four extra heads, with the auxiliary loss aggregated alongside the standard cross-entropy.

DeepSeek-V3 picked it up at scale, and [Sebastian Raschka's gallery entry on MTP](https://sebastianraschka.com/llm-architecture-gallery/mtp/) tracks the broader adoption: GLM-5, Qwen3-Next, Tencent Hy3-preview, Step 3.5 Flash, Nemotron 3 Super, and several more all ship MTP heads. The reason is not the training-time signal in isolation. It is that the same heads can be repurposed at inference as draft tokens, with the verifier and draft sharing a trunk.

In Qwen3-Next that mechanism is named directly. The [model card for Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct) lists Multi-Token Prediction as one of the four model-level features, alongside hybrid attention, sparse MoE, and stability optimizations. The recommended SGLang invocation in the same model card uses `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-num-draft-tokens 4`, and the [vLLM Qwen3-Next recipe](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-Next.html) exposes the same path under `--speculative-config '{"method":"qwen3_next_mtp","num_speculative_tokens":2}'`.

The mechanical part of the design is what matters here. The MTP head is one transformer block, sometimes called a NextN block, that sits on top of the main trunk and consumes the trunk's last-layer hidden state. Its parameter count is roughly `1/L` of the trunk's, where `L` is the layer count. For a 64-layer 27B target the head holds about 430 MB at Q8. For an 80B-A3B target with 48 layers and one extra MTP layer it holds roughly 1.5 GB at Q8 of which about 0.4 GB is active per token because of the MoE routing.

## Why the cost ratio collapses

The April 28 post worked the cost ratio in one direction: `c` is verifier work over draft work, so a verifier that activates only 3B parameters at a step makes a 0.8B draft expensive on a relative basis. With MoE active weights at 2 GB and a 0.8B draft at 0.36 GB, `c` lands near 0.20 before any rewind taxes are added. That stayed honest.

What changes with an MTP head is the draft itself. The MTP head reads the verifier's last hidden state and emits logits through a single transformer block, so its compute cost is `1/L` of the verifier's. On the 27B dense Qwen checkpoint at Q8 the verifier reads about 27 GB of weights per step and the head reads roughly 0.43 GB, putting `c` near 0.016 even before any cache reuse. On Qwen3-Next-80B-A3B with one MTP layer the same arithmetic puts active `c` near 0.13, four times better than the 0.8B-draft case in the prior post.

The acceptance rate moves the other way at the same time. A 0.8B draft from a different family approximates the verifier's distribution; an MTP head was trained against the verifier's hidden state at exactly the offsets it has to predict. The PR 22673 numbers measure 0.72 to 0.83 acceptance depending on `γ`, which is roughly the range DeepSeek-V3 reports for [its own MTP-1 configuration at "above 80%" acceptance per the V3 technical report](https://huggingface.co/deepseek-ai/DeepSeek-V3). The April 28 post's central estimate of `α = 0.55` for a cross-family draft holds for cross-family drafts. It does not hold for hidden-state-aligned heads.

The product is what matters. With `α = 0.83`, `γ = 3`, and `c = 0.04`, Leviathan's formula returns `(1 − 0.83⁴) / [(1 − 0.83) × (3 × 0.04 + 1)] ≈ 2.7×`. The PR's measured `2.5×` lands within fifteen percent of that closed form, which is the sort of agreement engineers should see when a model is right.

## Why partial rollback was the structural unlock

PR 22400's contribution is small in lines of code and large in implication. The previous `seq_rm` path on hybrid models did a full rollback to the most recent checkpoint when even one draft token was rejected, because the gated DeltaNet recurrence was advanced inside a fused kernel that wrote one final state and discarded its intermediates. A draft of length four with one rejection at the tail meant rolling back to before the draft started and re-decoding three accepted tokens.

The PR widens the GDN tensor by `1 + n_rs_seq` groups and writes the per-token intermediate state into a sliding ring, so that on rejection the engine can index directly to the position of the last accepted token. The rolled-back work shrinks from `O(γ)` to `O(γ − k)` where `k` is the number of accepted tokens, and the verifier's re-decode happens only over the rejected suffix. The 1.78x measurement on Qwen3.5-27B with the legacy 0.8B draft is what `c = 0.20` actually buys when the rewind tax is paid only for what was rejected, not for the entire draft window.

The same primitive is what PR 22673 leans on. The MTP head emits its three candidates against the verifier's per-token GDN intermediates, so when one is rejected the engine resumes from the corresponding intermediate and the verifier produces the right hidden state for the next round without re-prefilling. Without partial rollback the MTP path would still net out on the 27B dense Qwen checkpoint, where there is no GDN, but it would not net out on Qwen3-Next or Qwen3.6 35B-A3B where half the layers are linear. The two PRs are not independent; the second one needs the first.

## What the numbers actually look like

The benchmark grid in PR 22673 measures the 27B dense Qwen checkpoint at Q8 on a DGX Spark across nine prompt categories, with the same prompts run under three configurations.

| Configuration | Aggregate tok/s | Speedup vs baseline | Aggregate acceptance |
| --- | ---: | ---: | ---: |
| Baseline (no spec) | 7.0 | 1.00× | n/a |
| MTP draft, γ=2 | 15.7 | 2.24× | 0.83 |
| MTP draft, γ=3 | 16.8 | 2.40× | 0.72 |
| 0.8B Qwen3.5 draft, γ=16 with partial rollback | 17.5 | 2.50× | 0.68 |

The 0.8B draft row is the one to read carefully. With `γ = 16` and partial rollback it ties the MTP head on aggregate tok/s, but its acceptance rate is lower and its variance across prompts is higher. The MTP head's worst category is translation at 13.9 tok/s; the draft model's worst category is `explain_concept` at 12.7 tok/s. The MTP head also avoids distributing a separate GGUF artifact, which matters for a local engine where users do not want to manage two model files per target model.

The shape of the table holds for the dense 27B target. On Qwen3.6 35B-A3B and Qwen3-Next-80B-A3B the cross-family draft is still expected to lose because the MoE pulls verifier active weights down to 3 GB and the draft cost ratio cannot follow. The MTP head, by construction, scales its size with the trunk layer count rather than with the active expert count, so its `c` is close to `1/L` regardless of the MoE shape.

<figure class="diagram-card diagram-wide">
  <img class="diagram-visual" src="/blog/2026-05-08-mtp-vs-draft-spec-decode.svg" alt="A two-panel architectural comparison. The left panel shows the April 28 draft-model speculative decoding picture on Qwen 3.6 35B-A3B: a small 0.8B draft model emits four candidate tokens autoregressively, the 35B-A3B verifier checks them in parallel, one is rejected, and a red curved arrow shows a 1.8 GB gated DeltaNet state rewind on the rejected token. The label below reads cost ratio c equals 0.30, acceptance alpha equals 0.55, gamma equals 4, Leviathan speedup equals 0.96 times. The right panel shows the May 8 MTP-head picture: a single 48-layer Qwen3 trunk feeds its last hidden state into a one-layer MTP head drawn as a dashed border, the head emits three candidate tokens that the same trunk verifies in one batched forward pass, and a small state ring next to the trunk holds per-token gated DeltaNet intermediates so only the rejected suffix is rolled back. The label below reads cost ratio c equals 0.04, acceptance alpha equals 0.83, gamma equals 3, Leviathan speedup equals 2.5 times. A header at the top reads two speculative decode shapes for Qwen3 with a subhead explaining the cost-ratio shift. A footer notes that the MTP path is opt-in via spec-type mtp in llama.cpp PR 22673 and qwen3 next mtp in vLLM." loading="lazy" />
  <figcaption>Same family of models, different draft. The MTP head shifts both terms in Leviathan's formula at once: the cost ratio drops because the draft is one layer of the trunk, and the acceptance rate rises because the draft consumes the verifier's own hidden state.</figcaption>
</figure>

The diagram is the structural argument in shape. The April 28 picture has two physically separate models with their own forward passes, their own memory budgets, and a discard-and-redo penalty on the verifier whenever a draft is rejected. The May 8 picture has one model with one extra block, and a per-token state ring that turns rejection into a one-token rewind instead of a one-window rewind.

## What is still missing on the RDNA4 path

The bug at the bottom of PR 22673 is the part of this story that has not finished. The Vulkan backend, which is the path the Radeon AI PRO R9700 uses, returns garbage tokens with an acceptance rate of 0.013 on the same configuration that runs cleanly on CUDA. The author's response in the thread is that the PR depends on partial-rollback support that the Vulkan kernel for the gated DeltaNet path has not implemented yet. A separate effort, [PR 22587 for a row-per-warp CUDA kernel for `GATED_DELTA_NET`](https://github.com/ggml-org/llama.cpp/pull/22587), is the analogous CUDA-side performance work; the Vulkan equivalent has not landed.

This is the only thing standing between local Qwen3-Next on a 32 GB RDNA4 card and the same `2.5×` decode unlock that the CUDA path measured. The kernel is small. The change is to keep the per-token GDN intermediates in a sliding ring, expose a write-position parameter, and let the speculative path index into the ring on rejection. Most of the design work is already done in the CUDA kernel; porting it is a matter of adapting one Vulkan compute shader and updating the spec-decoding seq_rm dispatch.

The right operating point for zinc on a 32 GB card is `γ = 3` with the MTP head as the draft, and the host-memory state shadow [from the May 7 prefix-cache argument](/blog/2026-05-07-why-qwen3-next-gated-deltanet-breaks-the-prefix-cache-local-engines-just-built/) running underneath. The two designs are complementary. The state plane keeps multi-turn prefill cheap; the MTP head keeps decode fast. Neither one is the long-promised free lunch, but together they cover the two largest wall-time taxes that local Qwen3 was paying through April.

The takeaway from rerunning the April 28 cost ratios with the May 8 numbers is narrow and specific. Speculative decoding on hybrid MoE models is a draft-design problem, not a fundamental impossibility. Hand it the right draft and the equation prints `2.5×`. Hand it a draft that was not trained against the verifier's features and the equation prints `0.96×`. The local-inference field has been working the second variant for years; the first variant arrived ten days ago in two pull requests, and the only thing left to ship on RDNA4 is one Vulkan kernel.
