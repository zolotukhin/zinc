---
title: "How Mixture of Experts models work in ZINC"
date: "2026-04-04"
tags:
  - zinc
  - moe
  - mixture-of-experts
  - llm-inference
  - qwen3-5
  - gemma4
  - gguf
  - vulkan
  - metal
  - gpu-kernels
keywords:
  - Mixture of Experts models
  - MoE models
  - MoE inference
  - sparse expert routing
  - top-k expert routing
  - shared expert MoE
  - GPU MoE inference
  - Qwen3.5 35B-A3B
  - Gemma 4 26B-A4B
  - GGUF MoE inference
  - Vulkan MoE inference
  - Metal MoE inference
  - llama.cpp alternative for MoE
  - vLLM MoE routing
  - ZINC Mixture of Experts
excerpt: "Mixture of Experts (MoE) models only run a small set of experts per token, but the routing details decide whether inference is elegant or slow. This post explains how MoE models work, which Qwen and Gemma MoE models ZINC supports today, and how ZINC executes router top-k, batched expert kernels, and shared-expert paths on GPU."
---

Mixture of Experts (MoE) models are the current answer to a very practical LLM question: how do you make the model bigger without paying dense-model cost on every token? The short answer is sparse routing. Instead of running one giant feed-forward network everywhere, an MoE model asks a router which experts should handle the current token, runs only the top-k experts, and combines their outputs. The long answer is that MoE inference lives or dies on the routing implementation, which is exactly why [ZINC](/zinc) treats MoE as a first-class runtime problem instead of "just another model family."

This post explains what MoE models are, which MoE variants ZINC currently supports, and how the ZINC runtime executes them on Vulkan and Metal. If you want broader engine context first, start with [the design decisions behind ZINC](/blog/2026-04-03-every-design-decision-behind-zinc) and [the shader work that moved routing back onto the GPU](/blog/2026-03-29-the-shaders-standing-between-4-tok-s-and-27-tok-s).

## What a Mixture of Experts model actually does

At a high level, an MoE layer replaces one dense FFN with a small router plus a bank of experts:

1. The token's hidden state enters the FFN block.
2. A router projects that hidden state into one logit per expert.
3. The model keeps only the top-k experts for that token.
4. Only those selected experts run.
5. Their outputs are weighted and added together.
6. Some architectures also run a shared expert that every token sees.

That last point matters. "MoE" is not one architecture. Some models are transformer-only MoE stacks. Some are hybrid models where attention is interleaved with SSM layers. Some include a shared expert path in addition to routed experts. Some use SwiGLU, some use GEGLU, and the norm placement is not always the same.

<figure class="diagram-card diagram-compact">
  <div class="flow-diagram" role="img" aria-label="A flow diagram showing an MoE layer from hidden state through FFN norm, router projection, top-k selection, expert projections, activation, down projection, optional shared expert paths, weighted accumulation, and final MoE output plus residual.">
    <div class="flow-main">
      <div class="flow-node">Hidden state</div>
      <div class="flow-arrow">↓</div>
      <div class="flow-node">FFN norm</div>
      <div class="flow-arrow">↓</div>
      <div class="flow-node flow-node-accent">Router projection</div>
      <div class="flow-arrow">↓</div>
      <div class="flow-node">Softmax + top-k</div>
      <div class="flow-arrow">↓</div>
      <div class="flow-node">Selected expert IDs + weights</div>
      <div class="flow-arrow">↓</div>
      <div class="flow-node">Expert gate + up projections</div>
      <div class="flow-arrow">↓</div>
      <div class="flow-node">SwiGLU / GEGLU</div>
      <div class="flow-arrow">↓</div>
      <div class="flow-node">Expert down projections</div>
      <div class="flow-arrow">↓</div>
      <div class="flow-node">Weighted accumulation</div>
      <div class="flow-arrow">↓</div>
      <div class="flow-node flow-node-output">MoE output + residual</div>
    </div>
    <div class="flow-branches">
      <div class="flow-branch-label">Optional side paths folded into weighted accumulation</div>
      <div class="flow-branch-row">
        <span class="flow-pill">Shared expert</span>
        <span class="flow-plus">+</span>
        <span class="flow-pill">Shared gate scalar</span>
      </div>
    </div>
  </div>
  <figcaption>A minimal MoE layer view: route the token, run only the selected experts, optionally fold in a shared expert, then collapse everything back into one FFN output.</figcaption>
</figure>

Dense parts of the model still run for every token. Attention still runs. Norms still run. The embedding path still runs. The output projection still runs. MoE makes the FFN sparse, not the entire forward pass.

That distinction is why MoE inference is often more subtle than people expect. The "saved compute" story is real, but the new bottlenecks become router latency, expert batching, memory traffic, and whatever synchronization the runtime inserts between those steps.

## The MoE models ZINC has today

Today, ZINC's managed catalog contains two MoE entries:

| Model | Catalog status | What it is in practice | ZINC notes |
| --- | --- | --- | --- |
| `qwen35-35b-a3b-q4k-xl` | supported | Qwen3.5 35B-A3B hybrid MoE with SSM + full-attention scheduling and a shared expert path | Validated managed model on RDNA4 and Apple Silicon |
| `gemma4-12b-q4k-m` | experimental | Gemma 4 26B-A4B MoE with Gemma-specific norms and GEGLU FFN activation | Managed, but still marked experimental |

There is an important nuance here.

ZINC's parser recognizes several GGUF architecture strings related to sparse expert models:

- `qwen2moe`
- `qwen3moe`
- `qwen35moe`
- `gpt-oss`
- `gpt_oss`
- `openai-moe`

But the current managed Qwen3.5 35B-A3B entry is not loaded through the generic `qwen35moe` path. It maps to `qwen35`, which is ZINC's hybrid Qwen3.5-style runtime with SSM metadata, a `full_attention_interval`, and MoE FFNs layered on top. That is a meaningful distinction. In other words, Qwen3.5 35B-A3B in ZINC is not "just a transformer with experts." It is a hybrid architecture with sparse FFNs.

Gemma 4 is different again. When ZINC sees a Gemma-family config with `n_experts > 0`, it switches to a Gemma-specific MoE decode graph. That path keeps Gemma's norm placement, uses GEGLU instead of SwiGLU, and applies a post-FFN norm before the residual add.

If you want a useful mental model, think of ZINC as supporting three MoE shapes today:

- generic transformer MoE for `qwen2_moe` and `gpt_oss`
- hybrid Qwen3.5-style MoE for `qwen35`
- Gemma MoE for `gemma4`

That is also why broad comparisons like "MoE in vLLM" or "MoE in llama.cpp" can be a little misleading. They are useful as ecosystem references, but the actual runtime pressure depends on which MoE family you are talking about.

## How ZINC decides that a GGUF is MoE

MoE in ZINC starts as GGUF metadata, not as a hardcoded model list.

When the loader reads a model, it extracts the fields that define sparse routing:

| GGUF field | What ZINC uses it for |
| --- | --- |
| `*.expert_count` | Total number of experts in the layer |
| `*.expert_used_count` | Top-k: how many experts actually run for each token |
| `*.expert_feed_forward_length` | Per-expert intermediate width |
| `*.expert_shared_feed_forward_length` | Shared-expert intermediate width, when present |
| `*.full_attention_interval` | Hybrid schedule for models like Qwen3.5 |

The shared expert handling is worth calling out separately because real GGUFs are not always tidy. Some Qwen3.5 GGUFs omit the shared-expert intermediate dimension in metadata. ZINC handles that by falling back to the actual tensor shapes of `ffn_gate_shexp.weight`, `ffn_up_shexp.weight`, or `ffn_down_shexp.weight` and inferring the width from there.

That is one of those details that sounds boring until it breaks. If the runtime gets the shared-expert shape wrong, the model may still produce text, but the text quality drifts because the math is wrong in a way that is easy to miss. We hit exactly that class of bug during the [RDNA4 bring-up work](/blog/2026-03-27-what-broke-first-when-we-built-zinc-on-amd-rdna4).

Hybrid Qwen3.5-style models add one more twist. When SSM metadata exists, ZINC also tracks `full_attention_interval`. If the GGUF does not provide it, the loader defaults that interval to `4`, which means the runtime treats every fourth layer as a full attention layer and the others as SSM layers. That scheduling decision is part of how ZINC interprets the model, not a cosmetic detail.

## How MoE inference works inside ZINC

The actual decode-time MoE path in ZINC is straightforward in structure and deliberately GPU-heavy in execution.

### 1. Normalize the FFN input

Before routing, ZINC runs the appropriate FFN norm for the current architecture:

- Qwen3.5 reuses `post_attention_norm.weight` as the FFN norm
- Gemma uses its own `ffn_norm.weight`
- dense models stay on the ordinary dense FFN path

Once the hidden state is normalized, ZINC can route or project it without another host-side reshape or staging step.

### 2. Compute router logits on the GPU

Each MoE layer has a router projection, `ffn_gate_inp.weight`, which ZINC runs as a DMMV against the normalized hidden state. The output is one logit per expert in `router_logits_buf`.

This is the first place where ZINC refuses to treat MoE like an afterthought. Router math is part of the forward pass, so it stays on device by default.

### 3. Run fused `softmax_topk` on the GPU

ZINC's `softmax_topk` shader does three things in one GPU dispatch:

1. softmax over all expert logits
2. top-k selection
3. renormalization of the selected expert weights

The shader writes a compact routing buffer:

- first `k` entries: expert IDs
- next `k` entries: normalized routing weights

That output format matters because the next kernels can consume it directly without CPU unpacking. The entire point is to avoid the classic MoE anti-pattern: project router logits on the GPU, read them back to the CPU for one tiny decision, then upload the result again.

### 4. Dispatch all selected experts in parallel

After routing, ZINC's fast path batches all selected experts together instead of dispatching them one by one.

The runtime records:

1. expert gate projection for all selected experts
2. expert up projection for all selected experts
3. activation across the batched expert buffer
4. expert down projection for all selected experts
5. weighted accumulation back into one FFN result

For the common top-k pattern, that collapses what would otherwise be a long sequence of per-expert launches into five MoE stages. In the top-8 case the code comment is blunt about the win: expert-related dispatches drop from 32 to 5, and the corresponding barriers drop from 32 to 4.

That batching strategy is one of the main reasons MoE inference in ZINC looks different from a naive implementation. The problem is not just "run fewer experts." The problem is "run the selected experts without turning sparse routing into a dispatch-latency machine."

### 5. Use the right activation for the family

Activation is architecture-specific:

- Qwen-style MoE paths use SwiGLU
- Gemma MoE uses GEGLU

That difference is small at the blog-post level, but it matters in the runtime because the activation shader is part of the routed expert path. ZINC does not pretend that all MoE FFNs are interchangeable.

### 6. Accumulate expert outputs with routing weights

Once the selected experts have produced their down-projected outputs, ZINC runs `moe_weighted_acc`, which computes the weighted sum of all selected expert outputs into `moe_out_buf`.

This is the step that turns "k different expert answers" back into one FFN result for the layer.

If the full GPU path is not available for a given quantization or platform state, ZINC falls back to a CPU path:

- copy router logits back to host
- run CPU `topKSoftmax`
- dispatch experts serially
- accumulate with scalar weights

That fallback exists for correctness and portability, but it is not the path you want to live on for throughput.

### 7. Run the shared expert when the model has one

This part is easy to skip conceptually and very expensive to skip in practice.

For models with shared-expert tensors, ZINC runs the shared expert alongside the routed experts on every token:

1. `ffn_gate_shexp.weight` and `ffn_up_shexp.weight`
2. SwiGLU or GEGLU
3. `ffn_down_shexp.weight`
4. optional gating through `ffn_gate_inp_shexp.weight`

If a shared gate exists and the GPU path is present, ZINC uses `sigmoid_scale_acc` so the accumulation stays on-device. If the gate exists but the GPU helper is unavailable, it reads back the scalar, computes the sigmoid on CPU, and falls back to a scalar accumulate. If no shared gate exists at all, the shared expert is added with weight `1.0`.

This is not decorative extra math. Qwen3.5-quality output depends on it. Earlier in the project, leaving the shared expert out produced a model that looked superficially alive while still being mathematically wrong.

### 8. Finish with architecture-specific cleanup

After routed experts and shared expert work are combined:

- Gemma can apply `post_ffw_norm.weight` to the MoE output
- the final FFN result is added back into the residual stream
- decode continues to the next layer

At that point the MoE layer has collapsed back into the same interface as any other layer output: one hidden-state vector, ready for the next block.

## Why this implementation matters for MoE inference

MoE models are supposed to trade dense compute for sparse compute. If the runtime turns that trade into CPU round-trips, tiny dispatches, or repeated buffer synchronization, the advantage erodes fast.

That is the engineering reason the ZINC MoE path is built around:

- on-device router projection
- fused GPU top-k routing
- batched expert dispatch
- GPU weighted accumulation
- explicit shared-expert support

The alternative is easy to describe and hard to scale:

1. run router projection
2. copy logits back
3. choose experts on CPU
4. dispatch experts one by one
5. accumulate with repeated barriers

That works. It is also how an MoE engine quietly becomes a latency engine instead.

This is especially relevant if you are comparing MoE inference stacks across projects. llama.cpp, vLLM, TensorRT-LLM, and other runtimes all make different tradeoffs around batching, hardware assumptions, and fallback behavior. ZINC's version is deliberately optimized for local Vulkan and Metal inference on the hardware people actually own, especially AMD RDNA4 and Apple Silicon, not for a CUDA-only datacenter story.

## FAQ

### Are MoE models always faster than dense models?

No. MoE models reduce the amount of FFN work each token performs, but they add routing, expert selection, expert fan-out, and weighted accumulation. If the runtime is sloppy, those new costs can eat the gain.

### What does top-k routing mean in an MoE model?

It means the router scores every expert, keeps only the `k` highest-scoring experts for the current token, renormalizes those weights, and ignores the rest for that token.

### Why does the shared expert matter?

Because some MoE models do not send tokens only through routed experts. They also send every token through a shared expert path. If you omit that path, the model can still emit text while being systematically wrong.

### Are all MoE models implemented the same way in ZINC?

No. ZINC has a generic transformer-MoE path, a hybrid Qwen3.5 path, and a Gemma MoE path. The routing idea is shared, but the surrounding layer structure, activations, norms, and scheduling differ.

If you want to run one of the managed MoE models yourself, the fastest path is still the same: start from the [ZINC docs](/zinc/docs/getting-started), pull a managed model, and then inspect the architecture in [the spec page](/zinc/docs/spec). The interesting part of MoE is not that the model has experts. The interesting part is how much real inference engineering is required to make those experts behave like one coherent layer.
