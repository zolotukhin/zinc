---
title: "The broken Vulkan shaders keeping our AMD RDNA4 inference engine stuck at 4 tok/s"
date: "2026-03-29"
tags:
  - zinc
  - vulkan
  - radv
  - gpu-shader
  - shader-debugging
  - glsl-compute
  - rdna4
  - qwen3-5
  - local-llm-inference
  - ssm
  - moe
keywords:
  - Vulkan shader debugging
  - GPU shader debugging
  - AMD RDNA4 inference
  - local LLM inference
  - GLSL compute shader debugging
  - Vulkan compute shaders for AI inference
  - RADV driver crash
  - Mesa RADV
  - MoE routing shader
  - softmax top-k shader
  - SSM delta net shader
  - state space model GPU inference
  - GPU resident inference
  - CPU GPU synchronization overhead
  - Qwen3.5-35B-A3B GGUF
  - llama.cpp alternative Vulkan
excerpt: "ZINC could already run Qwen3.5-35B-A3B on AMD RDNA4, but local LLM inference was stuck at 4 tok/s because the Vulkan compute shaders behind SSM and MoE routing were still wrong. This is how we debugged RADV crashes, recurrent state drift, and CPU-GPU round trips to move decode back onto the GPU."
---

ZINC already knew how to talk. That was not the problem anymore.

By the time we reached this phase, the engine was generating coherent English on [Unsloth's exact `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/blob/main/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf) and matching llama.cpp closely enough that correctness bugs were finally visible as engineering bugs instead of vague model weirdness. The bad news was that our Vulkan inference path on AMD RDNA4 was doing that work at roughly **4 tok/s**, while the hardware should have been capable of far more.

The reason was not that AMD RDNA4 was too slow. The reason was that we were still making the CPU babysit parts of the forward pass that should have stayed on the GPU.

If you build local LLM inference systems, you have probably seen some version of this failure mode. The kernels look fine. The profiler says the GPU is active. The tokens are technically coming out. But the actual decode loop is full of tiny synchronization wounds: read a buffer back to the CPU, do one small piece of math in host code, upload the result again, wait on another fence, and repeat. You do not die from one of those wounds. You die from all of them together.

That was exactly where ZINC got stuck.

This post picks up after [what broke first in local LLM inference on AMD RDNA4](/blog/2026-03-27-what-broke-first-when-we-built-zinc-on-amd-rdna4) and after [the loop that made those bugs cheaper to chase](/blog/2026-03-28-karpathy-loop-autoresearch-and-the-self-improving-ai-loop-behind-zinc). The next bottleneck was more low-level: not tokenizer correctness, not graph structure, but shader correctness in the last CPU-owned parts of the decode path.

## The real problem was not compute. It was round trips.

Qwen3.5-35B-A3B is not a plain transformer. It is a hybrid of attention, mixture-of-experts routing, and structured state space layers. In ZINC's current architecture, 30 of the 40 layers use the SSM path, and all 40 layers still need routing decisions for MoE.

Before the shader work, that meant every decode token kept bouncing between GPU and CPU:

- 30 SSM layers projected on GPU, then read intermediate tensors back to the CPU for conv1d, delta-net state update, and gated norm, then uploaded the results again.
- 40 MoE layers read router logits back to the CPU for softmax plus top-k expert selection.
- Shared expert gating still had its own scalar handoff path too.

In aggregate, that decode path was making roughly **151 Vulkan submit/wait steps per token**. At around **0.3 ms** of overhead per submit, you can burn tens of milliseconds per token before the actual model math even gets a fair chance.

<figure class="diagram-card diagram-wide">
  <img src="/blog/shader-roundtrip-tax.svg" alt="A diagram comparing the old CPU-managed decode path with repeated GPU to CPU readbacks against the newer GPU-resident path that keeps SSM and routing work on-device." loading="lazy" />
  <figcaption>The first throughput win was not some magical kernel speedup. It was cutting the CPU out of the middle of decode.</figcaption>
</figure>

That is why the next milestone was so clear. We wrote four compute shaders specifically to move the remaining CPU-side work onto the GPU:

| Shader | Replaces | Why it matters |
| --- | --- | --- |
| `ssm_conv1d.comp` | CPU conv1d + SiLU | Removes one readback across 30 layers |
| `ssm_delta_net.comp` | CPU recurrent state update | Removes the hardest CPU SSM step |
| `ssm_gated_norm.comp` | CPU gated normalization | Keeps the whole SSM chain on-device |
| `softmax_topk.comp` | CPU router softmax + top-k | Removes 40 routing readbacks per token |

The pipelines existed. The descriptor setup existed. The runtime had GPU and CPU paths with automatic fallback. The only thing missing was the hardest thing: **the shaders had to be right**.

If you want to inspect the exact implementation surface, the current kernels live in [`softmax_topk.comp`](https://github.com/zolotukhin/zinc/blob/main/src/shaders/softmax_topk.comp), [`ssm_conv1d.comp`](https://github.com/zolotukhin/zinc/blob/main/src/shaders/ssm_conv1d.comp), [`ssm_delta_net.comp`](https://github.com/zolotukhin/zinc/blob/main/src/shaders/ssm_delta_net.comp), and [`ssm_gated_norm.comp`](https://github.com/zolotukhin/zinc/blob/main/src/shaders/ssm_gated_norm.comp), with the dispatch side wired through [`compute/elementwise.zig`](https://github.com/zolotukhin/zinc/blob/main/src/compute/elementwise.zig) and [`compute/forward.zig`](https://github.com/zolotukhin/zinc/blob/main/src/compute/forward.zig).

## Why this class of bug is nasty

There are at least three pleasant ways a shader can fail:

1. It does not compile.
2. It crashes immediately.
3. It writes obviously insane values.

None of those are what happened here.

The shader failures that mattered lived in the more annoying middle ground:

- one shader crashed RADV only under a specific subgroup pattern,
- three shaders ran and wrote plausible-looking buffers,
- and the final model output got worse in a way that looked like "the model is still kind of broken" rather than "this particular kernel is wrong."

That is a much worse debugging surface. Once you are in that regime, you are no longer asking "does the shader work?" You are asking "which part of this 40-layer forward pass diverges first, by how much, and because of what assumption?"

## Bug 1: `softmax_topk.comp` crashed RADV on RDNA4

The MoE routing shader sounds simple on paper:

1. load 256 router logits,
2. softmax them,
3. find the top-8 experts,
4. renormalize the selected weights.

On paper, that is tiny work. One workgroup. Shared memory. Perfectly reasonable GPU task.

The problem was not the algorithm. It was the subgroup implementation.

The original top-k selection used subgroup ballot operations to identify the winning lane:

```glsl
bool is_winner = (local_best == wave_best) && (local_best >= 0.0);
uvec4 ballot = subgroupBallot(is_winner);
uint first_winner = subgroupBallotFindLSB(ballot);
uint winning_idx = subgroupBroadcast(local_best_idx, first_winner);
```

That pattern is elegant. It is also exactly the kind of thing that can land in the seam between "valid SPIR-V" and "this particular driver backend hates it."

On our RDNA4 test hardware under RADV, the trouble was the interaction between:

- wave64 subgroup semantics,
- `uvec4` ballot results where only part of the mask is meaningful,
- and a broadcast index derived from the ballot path itself.

The crash was not a mystery about the model. It was a driver-compatibility problem triggered by a perfectly ordinary-looking subgroup trick.

The fix was much less fancy and much more robust: stop being clever, write local winners into shared memory, and let thread 0 scan the 64 candidates.

```glsl
// Each thread writes its best local candidate
s_local_val[tid] = best_val;
s_local_idx[tid] = best_idx;
barrier();

// Thread 0 scans all 64 lanes and picks the winner
if (tid == 0) {
    float global_best = -1.0;
    uint global_idx = 0;
    for (uint t = 0; t < 64; t++) {
        if (s_local_val[t] > global_best) {
            global_best = s_local_val[t];
            global_idx = s_local_idx[t];
        }
    }
    output_data[ki] = global_idx;
    output_data[k + ki] = floatBitsToUint(global_best);
    s_probs[global_idx] = -1.0;
}
```

If you work in this space, this is one of the recurring lessons: the "more GPU-native" version is not always the better systems choice. A slightly dumber shared-memory reduction can be a better production design than a prettier subgroup path if it survives the actual driver stack you run.

## Bug 2: the SSM shaders ran, but the math drifted across tokens

The SSM side was harder because the failure was not a crash. It was wrong output.

The GPU chain there is:

1. `ssm_conv1d.comp`
2. `ssm_delta_net.comp`
3. `ssm_gated_norm.comp`

If any one of those is wrong, the output of all 30 SSM layers is wrong, and the model still "works" just enough to waste your time.

<figure class="diagram-card diagram-wide">
  <img src="/blog/shader-fault-lines.svg" alt="A diagram showing the SSM chain of conv1d, delta-net, and gated norm on one side, plus the MoE router on the other, with the specific failure modes labeled." loading="lazy" />
  <figcaption>The shader work split into two classes of pain: a router shader that crashed fast, and an SSM chain that failed quietly.</figcaption>
</figure>

The delta-net shader was the most suspicious one because it sits at the intersection of four failure-prone details:

- the logical layout of `conv_out`,
- normalization and scaling order,
- persistent recurrent state updates,
- and f16 versus f32 tensor interpretation.

The buffer layout alone is enough to cause subtle corruption. The shader assumes `conv_out` is packed as:

```glsl
// conv_out layout: [Q(qk_dim), K(qk_dim), V(d_inner)]
uint qk_dim = d_state * n_group;
uint q_offset = k_hi * d_state;
uint k_offset = qk_dim + k_hi * d_state;
uint v_offset = 2 * qk_dim + h * head_v_dim;
```

That is consistent with the CPU reference, but it is still a dangerous place to live. The semantic names only matter if the GGUF tensor packing, the CPU slicing, and the GPU slicing all agree on what those three regions actually mean. If one side silently treats the same 8,192 floats as `[x, K, V]` while the other treats them as `[Q, K, V]`, the shader is "correct" only in the narrowest possible sense.

Then there is the state itself.

Each head carries a **128 x 128** recurrent matrix. With 32 heads per layer and 30 SSM layers, that is a lot of persistent GPU state being updated token after token. The shader decays that state, computes a row-wise dot product against K, performs an outer-product style update, and then reads out against Q. Nothing about that math is conceptually exotic. What makes it painful is that tiny numerical differences get a long runway to compound.

The CPU code normalizes Q and then scales it. The GPU shader fused those two operations:

```glsl
float inv_norm = inversesqrt(max(partial, 1e-12));
float q_scale = inv_norm / sqrt(float(d_state));
for (uint i = tid; i < len; i += 64) {
    s_q[i] *= q_scale;
}
```

Mathematically, that is equivalent to normalize-then-scale. Numerically, it is not identical. In a delta-net style recurrent update, "not identical" can become "visibly wrong" much faster than people expect, because the next token is using state that already contains the previous rounding error.

That is the part of GPU debugging people outside inference engineering often underestimate. Once a recurrent state is involved, a tiny layout or precision bug is not local anymore. It is persistent.

## The interesting part was not writing the shaders. It was isolating them.

At this point, the debugging plan stopped being glamorous and became professional.

The only sensible strategy was to compare the GPU and CPU chains stage by stage:

- GPU conv1d output versus CPU conv1d output
- GPU delta-net output versus CPU delta-net output
- GPU gated norm output versus CPU gated norm output

That sounds pedestrian. It is also exactly what prevents a week of random guessing.

The point was not to prove "the GPU path is broken." We already knew that. The point was to find the **first divergence**. In systems like this, the first wrong tensor is almost always the most useful fact in the whole investigation.

Once you know that conv1d matches and delta-net diverges, you stop wasting time on the wrong files. Once you know the router crashes only on the ballot path, you stop blaming tensor data. Once you know the Q/K/V split is the same on CPU and GPU, you stop inventing architecture theories and start looking at numerical order or state update semantics.

This is one of those areas where shader work and systems work are the same discipline. The good move is usually not a clever move. The good move is to reduce the uncertainty surface until only one plausible bug class remains.

## If you are debugging Vulkan compute shaders for inference

Three habits mattered more than any single trick:

- Find the first wrong tensor, not the final wrong token. Compare GPU and CPU stage by stage until one buffer diverges.
- Treat driver behavior as part of the design surface. A subgroup trick that is elegant in SPIR-V but unstable on RADV is still a bad production kernel.
- Be suspicious of anything recurrent. Once state carries across tokens, tiny layout or precision mistakes stop being local bugs and start becoming system-level drift.

That is the part of shader work people often under-budget. The hard part is not always writing the kernel. The hard part is building a debugging loop that tells you whether the kernel is wrong, numerically unstable, driver-hostile, or just looking at the wrong slice of memory.

## Why this matters if you build inference engines

The lesson here is not "GLSL on AMD is fragile," even though sometimes it is.

The real lesson is that once you build an inference stack below the comfortable framework layer, the hard problems stop respecting clean boundaries.

The routing bug was not just "a shader bug." It was a subgroup design decision colliding with a real driver.

The SSM bug was not just "a math bug." It was model layout, recurrent state, normalization order, and tensor precision all sharing one failure surface.

And the performance story was not just "optimize kernels." It was "remove host round-trips that make decent kernels irrelevant."

That is why this phase felt so important to write down. People in this industry often talk about inference performance as if the work begins at occupancy, tiling, and cache reuse. In practice, there is a whole earlier layer of pain where you are still trying to earn the right to optimize.

For ZINC, that meant getting shader correctness to the point where a GPU-resident decode path is not just faster in theory, but trustworthy enough to enable the next round of real optimization.

## What changes after these shaders are solid

Once these shaders are correct, a lot of downstream work gets simpler immediately:

- the GPU SSM path can stay active by default,
- the GPU MoE router can stay active by default,
- command-buffer batching can do real work instead of waiting on CPU round trips,
- and the decode loop can stop spending so much of its life in submission overhead.

That does not magically get ZINC to the full llama.cpp baseline. There is still more work after that: GPU-side expert dispatch, hotter weight reuse, better cache behavior, and occupancy tuning. But this shader milestone is the one that turns the rest of the performance work from speculative into concrete.

It is the difference between "the engine is technically alive" and "the engine is finally on the right side of the PCIe boundary."

If you want the broader context around these failures, the earlier pieces connect directly to this one: [why we are building ZINC](/blog/2026-03-25-why-we-are-building-zinc), [the home AI rig behind it](/blog/2026-03-26-building-a-local-ai-rig), [the early correctness bugs in the forward pass](/blog/2026-03-27-what-broke-first-when-we-built-zinc-on-amd-rdna4), and [the optimization loop we use to hunt these bugs faster](/blog/2026-03-28-karpathy-loop-autoresearch-and-the-self-improving-ai-loop-behind-zinc).

This shader phase is what sits between those stories. It is the moment where the project stops arguing with basic model correctness and starts arguing with the GPU at the level that actually matters.
