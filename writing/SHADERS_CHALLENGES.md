# GPU Shader Challenges: Getting SSM and MoE Routing Off the CPU

## The Problem in One Sentence

ZINC's forward pass produces correct, coherent text at 4 tok/s using CPU-side SSM and MoE routing. Four GPU compute shaders were written to replace this CPU work, but the softmax+top-k shader crashes the RADV driver and the SSM shaders produce wrong output. Fixing these shaders is the only thing standing between 4 tok/s and 27+ tok/s.

## How We Got Here

### Phase 3b: Correctness (complete)

The self-improving optimization loop ran 186 cycles and fixed 13 correctness bugs in the forward pass. ZINC now generates coherent English matching llama.cpp quality on Qwen3.5-35B-A3B (a hybrid attention + SSM + MoE model with 40 layers, 256 experts, top-8 routing).

### Phase 3c: Performance (current)

Profiling revealed the bottleneck isn't GPU compute speed -- it's CPU-GPU synchronization. Each decode token makes ~151 `vkQueueSubmit` + `vkWaitForFences` calls because:

- **30 SSM layers** each do: GPU projections --> readback to CPU --> CPU conv1d + delta-net + gated norm --> upload back to GPU --> GPU output projection. That's 4 submits per SSM layer = **120 submits**.
- **40 MoE layers** each readback router logits for CPU softmax + top-k = **40 submits**.
- **40 shared expert gates** each readback 1 scalar for CPU sigmoid = **40 more** (but some are batched).

At ~0.3ms per submit, that's ~45ms of pure Vulkan overhead per token, plus the CPU SSM computation time on 30 layers.

### The GPU Shader Approach

We wrote 4 new GLSL compute shaders to move all CPU-side computation to GPU:

| Shader | Replaces | Submits Eliminated |
|--------|----------|--------------------|
| `ssm_conv1d.comp` | CPU conv1d + SiLU (30 layers) | 30 readbacks |
| `ssm_delta_net.comp` | CPU delta-net state update (30 layers) | (same 30 readbacks) |
| `ssm_gated_norm.comp` | CPU RMS norm x SiLU gate (30 layers) | (same 30 readbacks) |
| `softmax_topk.comp` | CPU softmax + top-k routing (40 layers) | 40 readbacks |

Plus `sigmoid_scale_acc.comp` (already existed) for the shared expert gate.

The pipeline infrastructure, buffer allocations, descriptor set helpers, and integration code are all complete. The shaders compile. The forward pass has runtime-gated GPU/CPU paths. **But the shaders produce wrong results or crash.**

---

## Bug 1: softmax_topk.comp Crashes RADV

### What It Does

Selects the top-8 experts from 256 softmax probabilities, entirely on GPU. Single workgroup (64 threads), 256 floats in shared memory.

### The Algorithm

```glsl
// 1. Load 256 logits to shared memory, compute softmax
// 2. For each of k=8 rounds:
//    - Find the max probability (parallel reduction)
//    - Record the winning expert ID and weight
//    - Mask it out, repeat
// 3. Renormalize the 8 selected weights
```

### Why It Crashes

The top-k selection uses subgroup ballot operations to find which thread holds the maximum value:

```glsl
bool is_winner = (local_best == wave_best) && (local_best >= 0.0);
uvec4 ballot = subgroupBallot(is_winner);
uint first_winner = subgroupBallotFindLSB(ballot);
uint winning_idx = subgroupBroadcast(local_best_idx, first_winner);
```

On RADV (Mesa's Vulkan driver for AMD GPUs), this pattern is problematic at wave64:

1. `subgroupBallot()` returns a `uvec4` (128 bits) but only 64 bits are meaningful at wave64
2. `subgroupBallotFindLSB()` may scan unused bits containing garbage
3. `subgroupBroadcast()` with a non-uniform index derived from ballot operations may trigger undefined behavior in the RADV compiler's subgroup lowering

The extensions used are `GL_KHR_shader_subgroup_arithmetic` and `GL_KHR_shader_subgroup_ballot`. RADV supports these, but the interaction between ballot and broadcast at wave64 has known edge cases.

### Hardware Context

From the [RDNA4 ISA manual](docs/GPU_REFERENCE.md):

> Wave64 executes over 2 cycles on each SIMD. Subgroup operations at wave64 aggregate across all 64 lanes. The ballot result covers 64 lanes packed into the lower 2 components of uvec4.

The AMD Radeon AI PRO R9700 (our test hardware) has 64 CUs, each with 2 SIMDs. At wave64, a single workgroup of 64 threads maps to exactly one wave -- no cross-wave synchronization needed. The issue is purely in RADV's SPIR-V lowering of the ballot intrinsics.

### Possible Fixes

**Option A: Replace ballot with shared memory reduction**

```glsl
// Instead of ballot + broadcast:
shared uint s_winner_idx;
shared float s_winner_val;

// Each thread writes its local max to shared memory
if (local_best > s_winner_val) {  // Race condition, but...
    atomicMax(s_winner_val_bits, floatBitsToUint(local_best));
}
barrier();
// Thread that matches the atomic max writes its index
if (floatBitsToUint(local_best) == s_winner_val_bits) {
    s_winner_idx = local_best_idx;
}
barrier();
```

This avoids all ballot operations. Slightly slower due to atomic operations, but 256 experts / 8 rounds is tiny work.

**Option B: Use subgroupMax + subgroupShuffle instead of ballot**

```glsl
float wave_best = subgroupMax(local_best);
// Each thread checks if it has the winner
// Use subgroupMin on the lane ID where best matches
uint my_lane = gl_SubgroupInvocationID;
uint winner_lane = (local_best == wave_best) ? my_lane : 0xFFFFFFFF;
winner_lane = subgroupMin(winner_lane);
uint winning_idx = subgroupShuffle(local_best_idx, winner_lane);
```

This uses shuffle (supported on RADV) instead of ballot.

**Option C: CPU fallback for now, GPU router later**

The CPU softmax + top-k only takes ~0.01ms per layer. The 40 readbacks are the expensive part (not the softmax itself). A better approach might be to keep CPU routing but eliminate the readback by using a **pre-recorded command buffer with indirect dispatch** -- the GPU writes expert IDs to a buffer, and the CPU reads them after all layers complete (one submit instead of 40).

---

## Bug 2: SSM Shaders Produce Wrong Output

Three shaders form a chain: conv1d --> delta_net --> gated_norm. Any bug in the chain corrupts the final output for all 30 SSM layers (75% of the model).

### The Data Flow

```
                    GPU Buffers
                    ----------
attn_out_buf  <-- DMMV: QKV projection (conv_channels = 8192 floats)
gate_buf      <-- DMMV: Z gate (d_inner = 4096 floats)
router_logits <-- DMMV: Alpha/DT (dt_rank = 32 floats)
down_buf      <-- DMMV: Beta (dt_rank = 32 floats)
        |
        v
[ssm_conv1d.comp]
  reads: attn_out_buf (current input) + conv kernel tensor + persistent conv state
  writes: swiglu_buf (conv output, 8192 floats)
  state: shifts sliding window, writes new input
        |
        v
[ssm_delta_net.comp]
  reads: swiglu_buf (conv output) + alpha + beta + ssm_a + dt_bias
  reads+writes: persistent recurrent state (32 heads x 128x128 = 2MB per layer)
  writes: attn_out_buf (delta-net readout, 4096 floats)
        |
        v
[ssm_gated_norm.comp]
  reads: attn_out_buf (delta output) + gate_buf (Z gate) + norm weights
  writes: swiglu_buf (final SSM output, 4096 floats)
        |
        v
  DMMV: ssm_out projection (swiglu_buf --> o_proj_buf)
  Residual: hidden_buf += o_proj_buf
```

### Challenge 1: The conv_out Layout

The conv1d output buffer contains three logical sections:

```
conv_out[0 .. d_inner]                              = x  (value-like, 4096 floats)
conv_out[d_inner .. d_inner + n_group*d_state]       = K groups (16 groups x 128 = 2048 floats)
conv_out[d_inner + n_group*d_state .. conv_channels] = V groups (16 groups x 128 = 2048 floats)
```

But the delta-net shader's comment says the layout is `[Q(n_group*d_state), K(n_group*d_state), V(d_inner)]` -- the **opposite order**. The CPU code has:

```zig
// CPU reference (forward.zig)
const qk_dim = d_state * n_group;       // 128 * 16 = 2048
var q_ssm = conv_out[0..qk_dim];        // First 2048 = "Q" (actually K-group-sized)
var k_ssm = conv_out[qk_dim..2*qk_dim]; // Next 2048 = "K"
const v_ssm = conv_out[2*qk_dim..][0..d_inner]; // Last 4096 = "V"
```

The GPU shader uses:
```glsl
uint qk_dim = d_state * n_group;
uint q_offset = k_hi * d_state;           // within first qk_dim elements
uint k_offset = qk_dim + k_hi * d_state;  // within second qk_dim elements
uint v_offset = 2 * qk_dim + h * head_v_dim; // within last d_inner elements
```

These match each other, but the actual semantic mapping (which part is Q, which is K, which is V from the model's perspective) depends on how `attn_qkv.weight` is structured. If the GGUF tensor packs them as `[x, K_groups, V_groups]` rather than `[Q, K, V]`, the naming is misleading but the indexing is correct as long as CPU and GPU agree.

**The risk**: any mismatch in this 3-way split corrupts all 30 SSM layers.

### Challenge 2: Delta-Net State Matrix at Scale

Each head maintains a 128x128 state matrix (16,384 floats = 64KB). With 32 heads per layer and 30 SSM layers, the total persistent state is:

```
32 heads x 128 x 128 x 4 bytes = 2 MB per layer
30 layers x 2 MB = 60 MB total GPU state
```

The delta-net update per head involves:
1. **Decay**: multiply all 16,384 elements by `exp(gate)` -- a scalar
2. **Outer product update**: for each of 128 rows, compute `sk = dot(state[row], k)`, then `d = beta * (v[row] - sk)`, then `state[row][col] += k[col] * d`
3. **Readout**: for each of 128 rows, compute `out[row] = dot(state[row], q)`

With 64 threads handling 128 rows (2 rows per thread), each thread iterates over 128 columns per row for the dot products and updates. That's `128 x 128 = 16,384` memory accesses to device-local GPU memory per head.

**The 64KB state doesn't fit in shared memory** (RDNA4 LDS is 64KB per CU in CU mode, but shared memory per workgroup is configurable up to 64KB). The shader uses device-local buffer access instead, which is slower but correct.

**Numerical precision concern**: the state matrices accumulate across all tokens. Small floating-point errors in the decay or outer product compound over hundreds of tokens. The CPU path uses `f32` throughout. The GPU path also uses `f32`, but floating-point operation ordering differs (GPU processes rows in parallel, CPU processes rows sequentially). Over 256 tokens, this can cause visible output divergence.

### Challenge 3: Q Scaling and Normalization Order

The CPU code normalizes Q/K vectors, then applies a `1/sqrt(d_state)` scale to Q:

```zig
// CPU: normalize first, scale second
l2Normalize(q_ssm[h * d_state ..][0..d_state]);
// ... later ...
const q_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_state)));
for (q_ssm) |*v| v.* *= q_scale;
```

The GPU shader fuses normalization and scaling into one pass:

```glsl
// GPU: fused normalize + scale
float inv_norm = inversesqrt(max(partial, 1e-12));
float q_scale = inv_norm / sqrt(float(d_state));
for (uint i = tid; i < len; i += 64) {
    s_q[i] *= q_scale;
}
```

Mathematically equivalent, but floating-point associativity means `(x / norm) * scale != x * (scale / norm)` due to rounding. For the delta-net, this small difference gets amplified through the `sk = dot(state, k)` computation, then through the `state += k * d` outer product update, then through all subsequent tokens.

### Challenge 4: f16 Tensor Weight Detection

Some model tensors (conv kernel, ssm_a, dt_bias, ssm_norm weights) may be stored as f16 in the GGUF file. The CPU path reads these via `readMmapFloats()` which handles f16-->f32 conversion. The GPU path reads directly from the GPU tensor buffer using dual buffer aliases:

```glsl
layout(set = 0, binding = 1) readonly buffer ConvKernelF32 { float conv_kernel_f32[]; };
layout(set = 0, binding = 1) readonly buffer ConvKernelF16 { float16_t conv_kernel_f16[]; };
```

The Zig integration determines the type at dispatch time:

```zig
const conv_kernel_is_f16 = conv_tensor.info.type_ == .f16;
try self.elementwise.recordSsmConv1d(&self.decode_cmd, ds,
    conv_channels, d_conv, conv_kernel_is_f16);
```

This works for the Qwen3.5 model where tensors are consistently typed. But there's a subtle issue: the GPU tensor buffer was uploaded via DMA from the GGUF mmap. If the tensor is stored as f16, the GPU buffer contains raw f16 bytes. Reading via `float16_t` and casting to `float` may produce slightly different values than the CPU path's `readMmapFloats()` which goes through Zig's `@floatCast`.

For the conv kernel (8192 channels x 4 taps = 32,768 weights), even a 1-ULP difference per weight accumulates through the convolution.

---

## The Debugging Approach

### Step 1: Isolate Which Shader Is Wrong

Add diagnostic readbacks after each shader in the GPU SSM chain:

```
GPU conv1d output vs CPU conv1d output (8192 floats, layer 0)
GPU delta_net output vs CPU delta_net output (4096 floats, layer 0)
GPU gated_norm output vs CPU gated_norm output (4096 floats, layer 0)
```

The CPU reference already logs `SSM_DBG L0` values for layer 0. Adding GPU readbacks and comparing pinpoints which shader diverges.

### Step 2: Fix the Divergent Shader

Based on the analysis above, the most likely causes (in order):

1. **conv_out layout mismatch** in delta_net -- the Q/K/V extraction offsets may not match the actual GGUF tensor layout for this model
2. **Q scale order** in delta_net -- the fused normalize+scale may diverge enough to flip argmax decisions
3. **State update sequencing** -- the GPU processes rows in parallel within a thread (2 rows per thread), while the CPU processes all rows sequentially. If there are inter-row dependencies (there shouldn't be, but worth verifying), this could cause divergence
4. **f16 conv kernel rounding** -- if the conv kernel is f16, slight rounding differences compound through 8192-channel convolution

### Step 3: Fix softmax_topk

Replace the ballot operations with shared memory or shuffle-based top-k. The algorithmic approach is sound (repeated parallel max with masking), only the implementation uses RADV-incompatible subgroup intrinsics.

---

## Hardware Specs (Test Node)

| Spec | Value |
|------|-------|
| GPU | AMD Radeon AI PRO R9700 (RDNA4, gfx1201) |
| Compute Units | 64 |
| VRAM | 32 GB GDDR6 |
| Memory Bandwidth | 576 GB/s |
| L0 Vector Cache | 32 KB per CU |
| L2 Cache | 8 MB |
| Infinity Cache | 64 MB |
| LDS per CU | 64 KB (CU mode) |
| Wave Size | 64 (optimal for decode DMMV) |
| Driver | RADV (Mesa 25.0.7) |
| SPIR-V Compiler | glslc from shaderc 2023.8 (system, newer versions break RADV) |
| VK_KHR_cooperative_matrix | Supported (16x16x16 WMMA, wave32 only) |

### Memory Budget

```
Model weights:        21.0 GB (Qwen3.5-35B-A3B Q4_K_XL)
KV cache:              2.0 GB (4096 context, 40 layers)
SSM conv state:        0.004 GB (40 layers x 96 KB)
SSM recurrent state:   0.08 GB (40 layers x 2 MB)
Intermediate buffers:  0.1 GB (hidden, norm, gate, up, down, swiglu, moe_out, etc.)
                      -------
Total:                ~23.2 GB of 32 GB VRAM
```

### Theoretical Performance Bounds

```
Memory-bandwidth floor: 21 GB / 576 GB/s = 36.5 ms/tok = 27.4 tok/s
  (every weight read once per token, no caching)

llama.cpp baseline:     107 tok/s (same model, same hardware)
  (implies ~4x better than bandwidth floor -- L2/Infinity Cache reuse)

ZINC current:           4.3 tok/s (CPU SSM + 151 submits/token)
ZINC target:            27+ tok/s (milestone 1), 107+ tok/s (stretch)
```

---

## Scope of the Fix

### What's Already Done (no code changes needed)

- 4 GLSL compute shaders (411 lines total)
- 6 Vulkan compute pipelines with record methods
- Persistent GPU state buffers (conv: 3.75 MB, recurrent: 80 MB)
- Router output buffer (64 bytes, host-visible)
- Descriptor set helpers (writeDescSet4, writeDescSet7)
- `runSsmLayerGpu()` integration with buffer binding chain
- GPU softmax_topk router integration with expert ID readback
- GPU sigmoid_scale_acc shared expert gate integration
- Command buffer batching across all 40 layers
- `--profile` flag with Vulkan timestamp query pool
- Runtime GPU/CPU path selection (automatic fallback)
- All 40 build tests passing

### What Needs Fixing (estimated scope)

| Fix | Files | Lines | Difficulty |
|-----|-------|-------|------------|
| softmax_topk RADV crash | `softmax_topk.comp` | ~30 lines rewrite of top-k loop | Medium |
| SSM shader correctness | `ssm_delta_net.comp` + maybe `ssm_conv1d.comp` | ~20-50 lines | Hard (requires GPU debugging) |
| Diagnostic readbacks for debugging | `forward.zig` | ~40 lines (temporary) | Easy |

### What Happens After the Fix

Once the GPU shaders produce correct output:
- GPU SSM path activates automatically (runtime pipeline check)
- GPU router activates automatically
- GPU shared expert gate activates automatically
- Command buffer batching reduces submits from ~151 to ~42 per token
- Expected throughput: 10-27+ tok/s (depending on shader occupancy)

Further optimization (27 --> 107 tok/s) requires:
- GPU-side expert dispatch (eliminate remaining 40 MoE readbacks)
- L2 cache optimization (keep weights hot across layers)
- Shader occupancy tuning (profile DMMV workgroup sizes)
- Possibly multi-queue overlap (compute + transfer)
