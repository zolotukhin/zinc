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

## Phase 1: Kernel-Level Optimizations (target: 39 → 55-65 tok/s)

*Status: In progress*

### Current bottleneck breakdown at 39 tok/s (~26 ms/tok)

With the SSM CPU roundtrip eliminated, the time is now distributed across:

| Component | Est. Time | Opportunity |
|-----------|-----------|-------------|
| MoE expert DMMVs (gate+up+down, 40 layers) | ~10-12 ms | Byte→uint32 access pattern |
| Shared expert DMMVs (40 layers) | ~5-6 ms | Fuse gate+up reads |
| SSM GPU (proj+conv+delta+norm+out, 30 layers) | ~4-5 ms | Occupancy tuning |
| MoE router + softmax_topk (40 layers) | ~2-3 ms | Fuse into single kernel |
| Attention (10 layers) | ~2-3 ms | Already efficient |
| Final tail (norm + LM head + argmax) | ~1 ms | — |

### 1a. Q4_K MoE DMMV: byte access → packed uint32

The MoE expert DMMV shader (`dmmv_q4k_moe.comp`) reads Q4_K weight data as individual `uint8_t` bytes. Each thread performs ~32 scattered byte reads per sub-block — terrible for memory coalescing on GPUs that prefer 32-bit or wider aligned loads.

The non-MoE Q4_K shader already uses packed `uint` reads. Porting this to the MoE variant should improve cache utilization and coalesced memory access.

### 1b. Fuse router DMMV + softmax top-k

The MoE router produces 256 float logits (a tiny DMMV: 256 × 2048), followed by a separate dispatch for top-k selection. Since 256 floats fit in shared memory, a fused kernel can compute the dot products and immediately select the top-8 experts in one pass — eliminating a barrier and dispatch per layer.

### 1c. Fuse shared expert gate+up

Both shared expert projections read the same input (`ffn_norm_buf`). A fused kernel reads the input once and writes both outputs, halving the input bandwidth.

## The Gap to Close

| Milestone | tok/s | BW% | Status |
|-----------|-------|-----|--------|
| Starting point | 11 | 2.4% | ✅ Bugs fixed |
| Phase 0: GPU SSM | 39 | 21.8% | ✅ Done |
| Phase 1: Kernel fusion | 55-65 | 30-36% | 🔄 In progress |
| Phase 2: Infrastructure | 70-80 | 39-44% | Planned |
| Phase 3: Advanced | 100+ | 55%+ | Planned |
| llama.cpp reference | 102 | ~23%* | Target |

*llama.cpp's bandwidth calculation differs because it uses chunked prefill for SSM layers, reading more bytes per token but with better parallelism.

## Lessons Learned

1. **The biggest performance bug was a correctness fix.** Disabling GPU SSM for delta-net models was the right call when the output was wrong, but it introduced a 3.5x performance regression. The fix was to make the GPU path correct, not to keep it disabled.

2. **Position 0 is a liar.** Three of our five bugs were invisible at position 0 (IMROPE frequencies, Q5_K half-elements, MoE routing). Every value was correct for the first token. Multi-token regression tests are essential.

3. **-1.0 ≠ -∞ when your experts are tiny.** Fine-grained MoE with 256 experts produces very small router logits. What works for 8 experts breaks silently at 256.

4. **Source-level regression tests catch shader bugs.** We can't run GPU tests in CI, but `@embedFile` lets us assert structural properties of shader source: element counts, initialization values, access patterns. These caught every regression we tested for.

---

*Hardware details: AMD Radeon RX 9070 (RDNA4 / GFX1201), 64 CUs, 16 GB GDDR6 @ 576 GB/s, Mesa RADV driver, Vulkan 1.3. All benchmarks at Q4_K_M quantization with greedy sampling.*
