# The Self-Improving Loop: How an AI Agent Found Bugs We Couldn't See

When we set out to build ZINC's forward pass for Qwen3.5-35B-A3B — a hybrid attention + SSM + MoE architecture with 40 layers, 256 experts, and delta-net recurrent blocks — we knew the complexity would be extreme. We wrote the initial implementation by hand: 1,400+ lines of Zig code orchestrating Vulkan compute shaders, CPU-side state management, and MoE expert routing with GPU↔CPU synchronization.

The model loaded. The shaders compiled. Tokens came out. But the output was garbage — multilingual word soup that responded to different prompts but never formed a coherent sentence.

We spent a full day of manual debugging: verified the tokenizer matched llama.cpp exactly, confirmed the embedding and RMS norm produced bit-identical results against a CPU reference, traced the Q8_0 DMMV dispatch bug that left 97% of logits at zero. Each fix improved something but coherence remained elusive.

Then we pointed the optimization loop at the problem. Over 3 runs and 113 total cycles, the agent found bugs we never would have caught by reading the code.

## How the Loop Works

Each cycle follows a simple pattern:

1. **rsync** the local source to the remote RDNA4 test node (AMD Radeon AI PRO R9700, 32GB)
2. **Build** via `zig build` (compiles Zig + GLSL shaders to SPIR-V)
3. **Run** ZINC with a test prompt, capture stdout/stderr
4. **Diagnose** — parse tok/s, check for crashes, detect garbage output
5. **Spawn Claude** with full context: build errors, run output, architecture description, history of what was tried, what failed, accumulated ideas from all previous cycles
6. Claude reads the source, makes ONE focused change
7. **Verify** — rsync + rebuild + rerun
8. **Keep or revert** based on whether the change improved anything

The agent sees the RDNA4 hardware constraints, the Qwen3.5 architecture spec, the full list of already-fixed issues, and — critically — the self-analysis from its own previous cycles. Each cycle's reasoning feeds into the next.

## Run 1: The DMMV Performance Loop (29 cycles, 9 kept)

The first loop run focused on performance before the full forward pass existed. It optimized the DMMV (decode matrix-vector multiply) shader — the core operation that dominates inference throughput. Key wins:

- **Pre-allocated staging buffers** — eliminated per-token `vkAllocateMemory` calls
- **1-thread-per-row DMMV** — fixed 12.5% thread utilization when K=2048 (only 8 Q4_K blocks per row, but 64 threads per workgroup, leaving 56 idle)
- **Fused sub-block pairs** — extracted both nibbles per `qs` byte load, halving memory traffic
- **Shared memory input vector caching** — loaded the input vector into LDS cooperatively to eliminate L1 cache contention with the 72KB weight working set
- **Batched prefill** — processed all prompt tokens in a single GPU submission

Result: **7,531 → 9,769 tok/s prefill** (before the full transformer loop existed).

## Run 2: The Failed Run (43 cycles, 0 kept)

This run launched before the loop's diagnosis was updated with the forward pass architecture details. The agent had no context about the hybrid attention+SSM+MoE structure or what bugs had already been fixed. It spent 43 cycles trying various approaches but none passed the keep criteria — the output remained garbage and every change either broke the build or didn't improve anything.

**Lesson learned**: the loop is only as good as its prompt context. Without knowing what's already been fixed and what the architecture looks like, the agent wastes cycles rediscovering known issues.

## Run 3: The Correctness Loop (44 cycles, 40 kept)

This is where things got interesting. Armed with full architecture documentation, a list of 16 already-fixed issues, and specific debugging hints, the agent systematically dismantled a cascade of correctness bugs.

### The Wave32 Revelation (Cycle #3)

The very first substantive fix was something we never considered: **RADV might use wave32 subgroups instead of wave64**.

Our DMMV shaders accumulated partial sums across 64 threads using `subgroupAdd()`, assuming all 64 threads formed a single subgroup (wave64). On RDNA4, the hardware supports wave64, but RADV's compiler can choose wave32 for some dispatches. With wave32, only 32 threads reduce together — the other 32 threads' partial sums are silently discarded.

```glsl
// BEFORE: assumes wave64
sum0 = subgroupAdd(sum0);
if (tid == 0u) y_data[row0] = sum0;  // Missing half the dot product!

// AFTER: handles both wave32 and wave64
sum0 = subgroupAdd(sum0);
if (gl_NumSubgroups > 1u) {
    if (subgroupElect()) s_sum0[gl_SubgroupID] = sum0;
    barrier();
    if (tid == 0u) sum0 = s_sum0[0] + s_sum0[1];
}
if (tid == 0u) y_data[row0] = sum0;
```

This fix affected Q8_0, F16, and the RMS norm shader — every dot product in the forward pass was losing up to half its value on certain dispatches. The agent identified this by reasoning about why the output was wrong despite correct embeddings, and checking RADV's subgroup behavior.

### The Q4_K Sub-Block Pairing Bug (Cycle #19)

This was the most consequential single fix. The Q4_K dequantization shader pairs each byte's low and high nibbles with scale/min values from "sub-blocks." Our implementation paired sub-blocks `(sp, sp+4)` — meaning nibble 0 used scale 0 and nibble 4 used scale 4:

```glsl
// WRONG: stride-4 pairing
float sc_lo = decode_scale(ks_off, sp);      // sub-block 0
float sc_hi = decode_scale(ks_off, sp + 4u); // sub-block 4
uint x_sb_lo = x_local + sp * 32u;           // elements 0-31
uint x_sb_hi = x_local + (sp + 4u) * 32u;    // elements 128-159
```

But GGML's actual layout uses consecutive pairs `(2*sp, 2*sp+1)`:

```glsl
// CORRECT: consecutive pairing
float sc_lo = decode_scale(ks_off, sb_lo);    // sub-block 0
float sc_hi = decode_scale(ks_off, sb_hi);    // sub-block 1
uint x_sb_lo = x_local + sb_lo * 32u;         // elements 0-31
uint x_sb_hi = x_local + sb_hi * 32u;         // elements 32-63
```

Every Q4_K matrix-vector multiply in the model — all MoE expert gate and up projections across all 40 layers — was using the wrong scale values for half its elements. The agent found this by adding a CPU reference comparison for the embed→norm→LM_head path, confirming Q8_0 was correct, then reasoning that Q4_K must be the divergence point.

The tok/s jumped from 2.3 to 2.4 after this fix, but more importantly, the logit distribution changed significantly — a sign that the actual computation was now different.

### The Q4_K SPEC_K Bug (Cycle #9)

The Q4_K shader used a specialization constant `SPEC_K` (set to `hidden_dim=2048` at pipeline creation) for its inner loop bounds. This worked for attention projections where K always equals hidden_dim. But MoE expert down-projections have K=512 (intermediate_dim), and the O-projection has K=4096 (q_dim).

With `SPEC_K=2048`, the down-projection shader would iterate over 8 blocks per row (2048/256) instead of the correct 2 blocks (512/256), reading 6 blocks of garbage data from neighboring experts.

```zig
// WRONG: compile-time constant doesn't match runtime K
const blocks_per_row = SPEC_K / Q4K_BLOCK_SIZE;  // always 8

// CORRECT: use push-constant K which varies per dispatch
const blocks_per_row = K / Q4K_BLOCK_SIZE;  // 2 for down, 8 for gate/up
```

### The Shared Expert Dimension Mismatch (Cycle #26)

Qwen3.5 has two types of FFN: routed experts (256 experts, top-8 selected) with intermediate_dim=512, and a shared expert that processes every token with intermediate_dim=5632. Our code used the same `inter_dim` for both — which happened to be 512 (the per-expert value from `expert_feed_forward_length`).

The shared expert was computing only 9% of its intended intermediate representation (512/5632), essentially reducing it to noise. Since the shared expert runs on every token in every layer, this corrupted the hidden state progressively across all 40 layers.

The agent identified this by reading the GGUF metadata more carefully:
```
expert_feed_forward_length = 512        ← per-expert (routed)
expert_shared_feed_forward_length = 5632 ← shared expert (much larger!)
```

### The Conv1d Split Order (Cycles #27-28)

The SSM layers project through a fused `wqkv` weight and then split the output into Q, K, V components after conv1d. The agent found that our split order was wrong — we were extracting `[V, K, Q]` instead of `[Q, K, V]`:

```
// WRONG
V = conv_out[0..d_inner]            // 4096 elements
K = conv_out[d_inner..d_inner+qk]   // 2048 elements
Q = conv_out[d_inner+qk..end]       // 2048 elements

// CORRECT (matching llama.cpp)
Q = conv_out[0..qk_dim]             // 2048 elements
K = conv_out[qk_dim..2*qk_dim]      // 2048 elements
V = conv_out[2*qk_dim..end]         // 4096 elements
```

Since 30/40 layers are SSM, this meant Q was receiving V's data, K was getting the wrong slice, and V was a mix of Q and K. The agent traced this by reading `qwen35moe.cpp`'s `build_layer_attn_linear` function on the remote node.

### The Buffer Overflow (Cycle #14)

The attention Q+gate projection outputs `n_heads × head_dim × 2` floats (16 × 256 × 2 = 8192), but `attn_out_buf` was sized for only `q_dim` (4096 floats). Heads 10-15's data wrote past the buffer end, corrupting `o_proj_buf` which was allocated immediately after in GPU memory. The attention output for the last 6 heads was garbage.

### The Q5_K Dequant Bug (Cycles #20, #44)

The Q5_K shader had the same sub-block pairing bug as Q4_K but with an additional twist: Q5_K's element layout is interleaved differently than Q4_K. The first fix (cycle #20) applied the Q4_K-style consecutive pairing. A later cycle (#44) caught that Q5_K actually uses a different interleaving scheme where low and high nibble elements alternate: `y[2l] = low_nibble, y[2l+1] = high_nibble`.

This affected the expert down-projections (which use Q5_K in this model), meaning every MoE FFN's output projection was computing wrong values.

## The Stall Problem

After cycle #26, the loop hit a plateau. The output changed from multilingual garbage to ASCII numbers (`", 11111 110  31 311..."`) — closer to English but still not coherent. The first token `,` actually matched llama.cpp's second-most-likely prediction, suggesting the model was getting close.

But cycles 27-41 kept producing the same output. The keep logic was accepting "no regression" changes that didn't actually change anything — empty commits that passed verification because the output didn't get worse.

We diagnosed this by analyzing the loop's state:
- 40 out of 41 cycles were "kept" — but many were no-ops
- `garbageOutput` and `coherentText` weren't being tracked in the cycle history
- The stall wasn't detected, so the agent kept trying incremental tweaks instead of switching strategies

The fix: reject changes that don't change the output text, detect stalls after 3+ cycles with identical output, and inject stronger hints telling the agent to try fundamentally different approaches (SSH to read reference code, add CPU reference comparisons, disable subsystems to isolate).

## What the Loop Taught Us

### 1. The cascade effect is real

Every bug we found had downstream consequences that masked other bugs. The Q4_K sub-block pairing was wrong, which made MoE experts produce wrong output, which corrupted the hidden state, which made the attention layers attend to wrong values, which produced wrong KV cache entries, which made all subsequent tokens wrong. Fixing one bug often revealed three more.

### 2. GPU shader bugs are invisible

Unlike CPU code where you can printf values, GPU compute shaders fail silently. A shader that reads the wrong memory address doesn't crash — it just produces wrong numbers that look plausible. The wave32 subgroup issue is a perfect example: the dot product was "half wrong" but still produced non-zero, non-NaN values. Only by comparing against a CPU reference could you catch it.

### 3. The agent needs architecture context

Run 2 (0 kept out of 43 cycles) vs Run 3 (40 kept out of 44 cycles) demonstrates the critical importance of prompt context. The same agent with the same tools, but with a detailed architecture description and a list of already-fixed issues, was 10x more productive.

### 4. Stall detection is essential

Without detecting when the loop is stuck, cycles are wasted on no-op changes. The agent needs to know when its incremental approach isn't working and switch to a fundamentally different strategy — like reading the reference implementation or adding CPU-side verification.

### 5. Quantization is a minefield

Three of the top bugs (Q4_K sub-block pairing, Q5_K interleaving, SPEC_K loop bounds) were in the quantized dequantization shaders. These shaders implement complex bit-packing schemes where a single index error corrupts every element. The GGML reference code is the only reliable ground truth, and the mapping from GGML's C to GLSL is non-trivial.

## By the Numbers

| Metric | Run 1 (perf) | Run 2 (no context) | Run 3 (full context) |
|--------|:---:|:---:|:---:|
| **Cycles** | 29 | 43 | 44 |
| **Kept** | 9 (31%) | 0 (0%) | 40 (91%) |
| **Critical bugs found** | 0 | 0 | 8 |
| **Tok/s improvement** | 7.5K→9.8K prefill | — | 0→4.0 decode |
| **Output quality** | N/A (no forward pass) | Unchanged (garbage) | Multilingual garbage → ASCII/numbers |

The 8 critical bugs found by the loop (wave32 reduction, Q4_K pairing, SPEC_K bounds, shared expert dim, conv1d split, buffer overflow, Q5_K interleave, SSM conv ordering) represent errors that a human reviewer would likely miss on code inspection — they require understanding the interaction between the GLSL shader's bit-level operations, the GGUF tensor layout, and the Vulkan dispatch parameters.

## Current State

After all three loop runs plus manual debugging, ZINC's forward pass:
- Loads and processes 733 tensors (21 GB) on the RDNA4 GPU
- Runs all 40 transformer layers: 10 attention + 30 SSM + 40 MoE FFN
- Tokenizes correctly (matches llama.cpp bit-for-bit)
- Produces fully-computed logits (248K vocabulary, no zeros)
- Generates at ~4 tok/s (vs llama.cpp baseline of 107 tok/s)
- Output responds to different prompts but is not yet coherent

The model is close — the first generated token for "The capital of France is" is now `,` which matches llama.cpp's second-most-likely prediction (logprob -1.62). The remaining distance to coherent output likely lies in a few more dequantization or scaling bugs that the next loop run, armed with better stall detection and output quality tracking, should find.

## The Loop's Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   optimize_zinc.ts                       │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ buildAndRun()                                     │   │
│  │  rsync → zig build → run zinc → parse output     │   │
│  │  → detect phase (fix/optimize)                    │   │
│  │  → isGarbageOutput(), isCoherentText()            │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │ buildPrompt()                                     │   │
│  │  Diagnosis + architecture + fixed issues          │   │
│  │  + history + failed approaches + stall detection  │   │
│  │  + last cycle's self-analysis + accumulated ideas │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │ Claude Agent                                      │   │
│  │  Reads source → makes ONE focused change          │   │
│  │  → outputs @@@DESCRIPTION, @@@SELF_ANALYSIS,      │   │
│  │    @@@NEXT_IDEAS                                   │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │ Verify + Keep/Revert                              │   │
│  │  rsync → rebuild → rerun → compare output         │   │
│  │  Keep if: coherent, or fewer errors, or more       │   │
│  │    tokens, or output actually changed              │   │
│  │  Revert if: no-op, regression, or broke coherence  │   │
│  └──────────────────────────────────────────────────┘   │
│                         │                                │
│  ┌──────────────────────▼───────────────────────────┐   │
│  │ State Management                                  │   │
│  │  Track: cycles, failed approaches, ideas,          │   │
│  │    output snippets, coherence, stall count          │   │
│  │  Persist to JSON for resume across sessions        │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

The loop runs on a local Mac, rsyncing source to a remote RDNA4 node for build and execution. Claude operates in the local environment, editing Zig and GLSL source files. The verification step ensures every change is tested on real hardware before being committed.
