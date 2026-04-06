# From 11 to 139 tok/s: A Performance Journey on AMD RDNA4

*Technical deep-dive into optimizing a Vulkan inference engine for hybrid SSM/MoE models on consumer AMD GPUs.*

## The Setup

**Hardware**: AMD RX 9070 (RDNA4) — 64 CUs, 576 GB/s memory bandwidth, 16 GB VRAM
**Model**: Qwen3.5-35B-A3B — a hybrid architecture with 40 layers, 256 MoE experts (top-8 active), delta-net SSM layers, and standard attention layers interleaved every 4th layer
**Engine**: ZINC — Zig host + GLSL compute shaders + Vulkan 1.3
**Competitor**: llama.cpp at 102 tok/s on the same hardware

## The Starting Point: 11 tok/s

When we first got the Qwen3.5-35B-A3B model producing coherent output on ZINC, the decode speed was 11 tokens per second — about 91 ms per token. That's nearly 10x slower than llama.cpp.

The theoretical minimum for this model is ~2.2 ms/token (450 tok/s), based on reading ~1.28 GB of weights per token at 576 GB/s peak bandwidth. Our 91 ms meant we were achieving only 2.4% of the GPU's memory bandwidth — 97.6% of the time was wasted on something other than useful memory reads.

## Finding the Bugs (Before Performance)

Before we could optimize, we had to make the model work at all. Five bugs stood between us and coherent output:

### Bug 1: IMROPE Frequency Reset

Qwen3.5 uses Interleaved Multi-dimensional RoPE (IMROPE) with sections `[11, 11, 10, 0]` — 32 rotation pairs split across three groups. Our frequency precomputation treated each section independently, resetting the exponent to zero at each boundary:

```
Section 0: freq[k] = 1/base^(k/11)  for k=0..10
Section 1: freq[k] = 1/base^(k/11)  for k=0..10  ← RESET TO 1.0 AT PAIR 11!
Section 2: freq[k] = 1/base^(k/10)  for k=0..9   ← RESET TO 1.0 AT PAIR 22!
```

The correct computation uses a single global progression: `freq[k] = 1/base^(2k/rope_dim)` for all 32 pairs. For text generation where all position IDs are identical, IMROPE reduces to standard NeoX-style RoPE — the section boundaries only matter for multi-modal (vision) inputs.

**Why it was invisible at position 0**: The rotation angle is `position × frequency`. At position 0, every angle is zero regardless of frequency, so all rotations are identity. The bug only manifested at position 1+, causing progressive divergence during multi-token prefill.

### Bug 2: Q5_K DMMV Half-Elements

The Q5_K dequantized matrix-vector multiply shader processed only 16 of 32 bytes per sub-block group, silently dropping half the dot-product terms:

```glsl
// BUG: each thread processes 4 elements, 4 threads = 16 total
uint e_start = slice * 4u;
for (uint e = e_start; e < e_start + 4u; e++) { ... }

// FIX: each thread processes 8 elements, 4 threads = 32 total
uint e_start = slice * 8u;
for (uint e = e_start; e < e_start + 8u; e++) { ... }
```

Every Q5_K tensor — which included the SSM QKV projection and output weights — produced subtly wrong results. The model could still generate text, but it was incoherent.

### Bug 3: MoE Router Initialization

The softmax top-k shader for MoE expert routing initialized its winner search with `-1.0` instead of `-infinity`:

```glsl
// BUG: if all expert logits < -1.0, expert 0 always "wins"
float global_best = -1.0;

// FIX: proper -inf initialization
float global_best = -1.0 / 0.0;
```

With 256 fine-grained experts (each with only 512 intermediate dimensions), router logits were routinely below -1.0. The model silently routed every token to expert 0, regardless of the router's actual prediction.

### Bug 4: Gemma3 Metal Backend

Five separate issues prevented Gemma3 from working on Apple Silicon:
- Missing embedding scaling by `√hidden_dim` (~62x magnitude error)
- FFN norm weight loading order reversed (used post-attention norm instead of FFN norm)
- Missing post-attention and post-FFN RMS norms (Gemma-specific architecture requirement)
- Wrong activation function (SwiGLU instead of GEGLU)

### Bug 5: Qwen3.5 BOS Token

A hardcoded fallback prepended BOS token ID 1 (the `"` character) to every Qwen3.5 prompt, shifting all position indices by one and corrupting attention patterns.

## Phase 0: The 3.5x Speedup (11 → 39 tok/s)

With the model producing correct output, we turned to performance. The profiler told a clear story:

| Component | Time | % of Token |
|-----------|------|-----------|
| SSM CPU roundtrips (30 × submitAndWait) | ~50 ms | 55% |
| SSM CPU compute (conv1d + delta-net) | ~12 ms | 13% |
| MoE GPU path | ~11 ms | 12% |
| Everything else | ~18 ms | 20% |

**The dominant bottleneck was architectural, not computational.** Each of the 30 SSM layers forced a GPU→CPU→GPU roundtrip: flush the command buffer, wait for GPU completion, run conv1d and delta-net on CPU, upload results, restart the command buffer. At ~1.5 ms per synchronization point, the 30 roundtrips alone consumed 45+ ms — more than the entire token budget of llama.cpp.

The GPU SSM shaders (conv1d, delta-net state update, gated normalization) already existed in the codebase. They'd been disabled because of correctness concerns about the delta-net update rule, but careful comparison against the CPU reference showed they produced identical results.

The fix was a single line change: instead of gating on architecture type (`!has_delta_net`), gate on shader availability:

```zig
// Before: force CPU for all delta-net models
const has_delta_net = config.full_attn_interval > 1;
const use_gpu_ssm = pipeline_ssm_conv1d != null and !has_delta_net;

// After: use GPU when all shaders are available
const use_gpu_ssm = pipeline_ssm_conv1d != null and
    pipeline_ssm_delta_net != null and
    pipeline_ssm_gated_norm != null;
```

This eliminated all 30 `submitAndWait` calls. The entire 40-layer decode now runs in a single command buffer submission.

**Results:**
- Qwen3.5-35B-A3B: 11 → 38.8 tok/s (3.5×)
- Qwen3.5-2B: 33.8 → 138.6 tok/s (4.1×)
- Bandwidth utilization: 2.4% → 21.8%

## Phase 1: The Shared Memory Revelation (39 → 48 tok/s)

### Profiling exposes the real bottleneck

With GPU timestamps enabled, we finally had per-phase timing for every dispatch:

| Phase | Time (40 layers) | % of Token |
|-------|-------------------|-----------|
| MoE router DMMV | 8.89 ms | 23% |
| Shared expert projections | 7.73 ms | 20% |
| Delta-net SSM | 4.77 ms | 12% |
| MoE softmax top-k | 2.82 ms | 7% |
| MoE expert gate+up | 2.70 ms | 7% |
| Attention | 2.60 ms | 7% |
| MoE expert down | 2.06 ms | 5% |
| SSM projections | 1.87 ms | 5% |
| Everything else | ~5.5 ms | 14% |

The **MoE router was the #1 bottleneck at 8.89 ms** — 23% of total token time for a tiny 256×2048 matrix multiplication. This was baffling. A 2MB weight matrix at 576 GB/s peak bandwidth should take 3.5 µs. We were 2,500× slower than theoretical.

### The diagnosis

The f32 DMMV shader was the simplest in the codebase:

```glsl
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= M) return;
    float sum = 0.0;
    for (uint j = 0; j < K; j++) {
        sum += w[a_offset/4 + row*K + j] * x[x_offset/4 + j];
    }
    y[y_offset/4 + row] = sum;
}
```

No shared memory. Each of the 256 threads independently read the same 8KB input vector from global memory. With 256 threads × 2048 floats × 4 bytes = **2MB of redundant reads** — doubling the memory traffic for every router dispatch. Across 40 layers, that's 80MB of wasted bandwidth.

And with only `ceil(256/64) = 4` workgroups, only 4 of the GPU's 64 CUs were active — 6.25% compute utilization. The memory subsystem couldn't hide latency with so few wavefronts in flight.

### 1a. Q4_K MoE DMMV: packed reads (marginal gain)

We first tried rewriting the MoE expert DMMV (`dmmv_q4k_moe.comp`) from `uint8_t` byte access to packed `uint32` reads with `vec4` dot products. Each Q4_K block (144 bytes) went from 144 individual byte reads to 36 aligned 4-byte reads.

**Result: ~0.3 tok/s improvement.** The RADV driver was already smart enough to coalesce the byte reads into wider loads behind the scenes. The real bottleneck was elsewhere.

### 1a′. F32 DMMV: shared memory input caching (major win)

The fix was adding a single `shared float s_x[K]` array and cooperative loading:

```glsl
shared float s_x[SPEC_K];

void main() {
    uint tid = gl_LocalInvocationID.x;
    // All 64 threads cooperate to load x once
    for (uint i = tid; i < K; i += 64) {
        s_x[i] = x[x_base + i];
    }
    barrier();
    // Each thread computes its row using shared memory
    for (uint j = 0; j < K; j += 4) {
        sum += w[w_base+j] * s_x[j] + w[w_base+j+1] * s_x[j+1]
             + w[w_base+j+2] * s_x[j+2] + w[w_base+j+3] * s_x[j+3];
    }
}
```

**Results (GPU timestamps, per-phase):**

| Phase | Before | After | Change |
|-------|--------|-------|--------|
| MoE router | 8.89 ms | 2.40 ms | **-73%** |
| Shared expert proj | 7.73 ms | 2.18 ms | **-72%** |
| MoE expert gate+up | 2.70 ms | 1.44 ms | -47% |
| SSM total | 8.07 ms | 5.71 ms | -29% |
| GPU total | 38.7 ms | 21.2 ms | **-45%** |
| **Overall** | **39 tok/s** | **48 tok/s** | **+23%** |

The shared expert projection improvement was unexpected — it uses Q8_0, not f32. But by cutting the f32 router's cache footprint in half, we reduced contention in the L2 cache for every other dispatch on the GPU. This cascading effect across all phases was larger than the direct savings.

### Current breakdown at 48 tok/s (~21 ms/tok)

| Phase | Time | % of Token |
|-------|------|-----------|
| SSM (delta-net + proj + conv + norm) | 5.71 ms | 27% |
| MoE (router + topk + gate/up + down + acc) | 7.16 ms | 34% |
| Attention | 3.39 ms | 16% |
| Shared expert | 2.85 ms | 13% |
| Tail (norm + LM head + argmax) | 0.90 ms | 4% |
| CPU overhead | 0.60 ms | 3% |

## The Gap to Close

| Milestone | tok/s | BW% | Status |
|-----------|-------|-----|--------|
| Starting point | 11 | 2.4% | ✅ Bugs fixed |
| Phase 0: GPU SSM | 39 | 21.8% | ✅ Done |
| Phase 1a: Shared memory + packed DMMV | 48 | 27% | ✅ Done |
| Phase 1b-c: Kernel fusion | 55-65 | 30-36% | 🔄 In progress |
| Phase 2: Infrastructure | 70-80 | 39-44% | Planned |
| Phase 3: Advanced | 100+ | 55%+ | Planned |
| llama.cpp reference | 102 | ~23%* | Target |

*llama.cpp's bandwidth calculation differs because it uses chunked prefill for SSM layers, reading more bytes per token but with better parallelism.

## Lessons Learned

1. **The biggest performance bug was a correctness fix.** Disabling GPU SSM for delta-net models was the right call when the output was wrong, but it introduced a 3.5x performance regression. The fix was to make the GPU path correct, not to keep it disabled.

2. **Position 0 is a liar.** Three of our five bugs were invisible at position 0 (IMROPE frequencies, Q5_K half-elements, MoE routing). Every value was correct for the first token. Multi-token regression tests are essential.

3. **-1.0 ≠ -∞ when your experts are tiny.** Fine-grained MoE with 256 experts produces very small router logits. What works for 8 experts breaks silently at 256.

4. **Source-level regression tests catch shader bugs.** We can't run GPU tests in CI, but `@embedFile` lets us assert structural properties of shader source: element counts, initialization values, access patterns. These caught every regression we tested for.

5. **Profile before you optimize.** The Q4_K packed reads were the "obvious" optimization — the MoE expert DMMV was supposedly the bottleneck. Profiling revealed the real bottleneck was the trivial f32 router shader, where a missing shared memory declaration wasted more bandwidth than the Q4_K byte access pattern.

6. **Cache effects cascade.** Fixing one shader's memory access pattern improved performance across every other dispatch on the GPU. The L2 cache is a shared resource — reducing pressure in one kernel directly benefits all others running on adjacent CUs.

---

*Hardware details: AMD Radeon RX 9070 (RDNA4 / GFX1201), 64 CUs, 16 GB GDDR6 @ 576 GB/s, Mesa RADV driver, Vulkan 1.3. All benchmarks at Q4_K_M quantization with greedy sampling.*
