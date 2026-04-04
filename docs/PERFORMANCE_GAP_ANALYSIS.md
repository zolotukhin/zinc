# ZINC vs llama.cpp Performance Gap Analysis

Date: 2026-04-04
Model: `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`
GPU: AMD Radeon AI PRO R9700 (RDNA4, 32 GB, 576 GB/s, 64 CUs)

## Current Numbers

| Metric | ZINC | llama.cpp | Gap |
|--------|------|-----------|-----|
| RDNA4 35B decode | 37.95 tok/s (26.3 ms/tok) | ~107 tok/s (9.3 ms/tok) | **2.8x** |
| RDNA4 35B prefill | Not reported | ~223 tok/s | — |
| Modeled bandwidth | 127 GB/s (22.1%) | ~360 GB/s (62.5%) | 2.8x |

Theoretical ceiling: 576 GB/s / ~1.5 GB weights per token = ~384 tok/s max.
llama.cpp at 107 tok/s uses ~28% of peak bandwidth. ZINC at 38 tok/s uses ~22%.

## ZINC Token Breakdown

From `DECODE_THROUGHPUT_PLAN.md` profiling (37.95 tok/s baseline):

| Phase | GPU Time |
|-------|----------|
| Routed MoE total | 10.24 ms |
| SSM total | 5.67 ms |
| Shared expert total | 5.35 ms |
| Attention | 3.58 ms |
| Final tail | 0.90 ms |
| **Total** | **~26.3 ms** |

CPU record time is only ~0.58 ms. GPU execution dominates at ~37 ms (in profile mode).
1022 descriptor allocs + 1022 descriptor writes per token.
Submit-to-completion adds ~0.95 ms over measured GPU work.

### SSM Breakdown

| Sub-phase | Time |
|-----------|------|
| Projections | 1.59 ms |
| Conv1d | 0.26 ms |
| Delta-net update | 3.08 ms |
| Gated norm | 0.23 ms |
| Out projection + residual | 0.76 ms |

### Routed MoE Breakdown

| Sub-phase | Time |
|-----------|------|
| Router projection | 5.23 ms |
| Softmax top-k | 2.21 ms |
| Gate + up expert projections | 1.55 ms |
| SwiGLU | 0.27 ms |
| Down projection | 1.16 ms |
| Weighted accumulate | 0.25 ms |

### Shared Expert Breakdown

| Sub-phase | Time |
|-----------|------|
| Gate + up projections | 4.69 ms |
| SwiGLU | 0.27 ms |
| Down projection | 0.37 ms |
| Gate accumulate | 0.26 ms |

## Hot-Kernel Microbenchmark Results

16-slot rotating working set, March 31, 2026:

| Kernel | Bandwidth | % of Peak |
|--------|-----------|-----------|
| q8_shared_gate_up | 414.2 GB/s | 71.9% |
| q8_shared_down | 399.7 GB/s | 69.4% |
| q8_ssm_out | 376.0 GB/s | 65.3% |
| q8_router | 319.7 GB/s | 55.5% |
| ssm_delta | ~36.0 GB/s | 6.2% |

## llama.cpp Vulkan Backend Characteristics

From `research/llama_cpp_analysis.md` and `docs/RDNA4_TUNING.md`:

### Architecture Detection

- RDNA4 (gfx1201) classified as `AMD_RDNA3` — no RDNA4-specific enum
- No entry in `gpu_pipeline_configs`, falls through to driver default (wave64)

### Matmul Path Selection

- MMVQ (integer dot product quantized path) requires `GL_EXT_integer_dot_product`
- Default Ubuntu glslc (shaderc 2023.8) doesn't support it
- Newer glslc produces SPIR-V that causes 5x RADV regression
- Falls back to FP16 DMMV path with `rm_kq=2` (2 rows per workgroup)

### Op Fusion (already implemented in llama.cpp)

| Fusion | Pattern | Dispatches Saved |
|--------|---------|-----------------|
| MULTI_ADD | N consecutive ADDs → 1 | ~280 |
| RMS_NORM_MUL | RMS_NORM + MUL → 1 | ~131 |
| TOPK_MOE | SOFTMAX+ARGSORT+GET_ROWS+SUM_ROWS+CLAMP+DIV → 1 | ~360 |
| MUL_MAT_ID_MUL | MUL_MAT_ID + MUL → 1 | ~39 |
| MUL_MAT_ADD | MUL_MAT + ADD → 1 | ~9 |
| GLU | SILU/GELU + MUL | ~80 |

Total compute graph for Qwen3.5-35B-A3B single token: 3728 nodes, 2356 dispatchable, ~1500 after fusions.

### llama.cpp Profiling on RDNA4

Per-token decode: ~10.2 ms total (63% matmul, 35% non-matmul, <1% dispatch overhead).

DMMV bandwidth utilization by shape:

| Shape | Bandwidth % |
|-------|------------|
| 248320 x 2048 (vocab output) | 93.2% |
| 8192 x 2048 (large attn) | 83.6% |
| 4096 x 2048 (medium attn) | 66.1% |
| 512 x 2048 (MoE expert, Q4_K) | 59.6% |
| 32 x 2048 (small) | 2.7% |

Command buffer submission: `nodes_per_submit=100` is already optimal. Testing 10000 = same, 10 = -8%.

Concurrent scaling: 4 slots at 108 tok/s each = 432 aggregate (linear, GPU not saturated by single stream).

## Detailed Findings with Code References

### Finding 1: SSM GPU Path Guard (`forward.zig:1931`)

```zig
const use_gpu_ssm = self.elementwise.pipeline_ssm_conv1d != null and config.architecture != .qwen35;
```

This line explicitly excludes Qwen3.5 from the GPU SSM path. When the architecture
is `.qwen35`, the condition evaluates false and all 30 SSM layers go through
`runSsmLayerCpu` (`forward.zig:1944`).

**However**, the profiling data reports "0 CPU fallbacks" on the benchmark path,
which means either:
- The architecture enum is not `.qwen35` at runtime (detected as something else), or
- The fallback counter (`cpu_ssm_fallbacks`) doesn't cover this specific path

This needs verification: check the startup log line `FASTPATH: gpu_ssm=...` to confirm
which path is actually active.

If the CPU path IS active, the per-layer cost is catastrophic (`forward.zig:2594-2943`):
1. 4 GPU DMMV projections recorded into command buffer
2. **GPU→CPU readback** (`submitAndWait` at ~2658) — copies ~20KB to host
3. CPU conv1d (~2692-2702)
4. CPU delta-net state update (~2807-2830) — O(dt_rank * head_v_dim²) = O(32 * 128 * 128)
5. CPU gated RMS norm (~2891-2904)
6. **CPU→GPU upload** of ssm_output (~2919-2927) — writes d_inner floats to staging
7. GPU ssm_out DMMV + residual — one more dispatch + sync

For 30 SSM layers, this creates ~60 GPU-CPU sync barriers per token. Each
`submitAndWait` stalls the GPU pipeline entirely (~0.5-1.0ms per sync).

**Fix**: Remove the architecture guard, qualify on shader availability only:
```zig
const use_gpu_ssm = self.elementwise.pipeline_ssm_conv1d != null and
    self.elementwise.pipeline_ssm_delta_net != null and
    self.elementwise.pipeline_ssm_gated_norm != null;
```

**Estimated impact if CPU fallback is active**: ~15-20 ms saved (potential doubling of throughput).

### Finding 2: Descriptor Sets Allocated Per-Dispatch (`forward.zig:2558`)

Every DMMV call allocates a fresh descriptor set from the pool and writes 3 buffer
bindings:

```zig
const ds = try self.allocDescSet(pip.descriptor_set_layout);
self.writeDescSet3(ds, tensor.gpu_buffer.handle, ...);
```

Per-token descriptor set count breakdown:
- Per attention layer: ~15 descriptor sets (rms_norm, q/k/v projections, deinterleave,
  sigmoid_mul, rope_q, rope_k, attn, o_proj, residual, ffn_norm, gate, up, down, residual)
- Per SSM layer: ~8 descriptor sets
- Per MoE layer: ~12 descriptor sets (router, softmax_topk, gate_moe×8, up_moe×8,
  swiglu, down_moe×8, weighted_acc, shared_gate, shared_up, shared_swiglu, shared_down,
  shared_gate_acc, residual)
- Final: ~3 (final_norm, lm_head, argmax)

**Total: ~500+ descriptor set allocations and writes per token.**

Then at the top of each token (`forward.zig:1461`), `vkResetDescriptorPool` frees ALL
previously allocated descriptor sets, forcing complete reallocation.

llama.cpp pre-allocates and caches descriptor sets. For decode, the command buffer is
recorded once with all bindings, then replayed with only push constants updated.

**Fix**: Pre-allocate descriptor sets during `init()` for each fixed buffer combination.
Since ZINC uses a fixed set of intermediate buffers (hidden_buf, norm_buf, q_buf, etc.),
descriptor sets can be pre-created. At decode time, only `vkCmdPushConstants` changes.

**Estimated impact**: ~3-5 ms saved.

### Finding 3: Command Buffer Re-Recorded Every Token (`forward.zig:1461-1463`)

```zig
_ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
try self.decode_cmd.reset();
try self.decode_cmd.begin();
```

The entire command buffer (with ~500+ dispatches, barriers, and descriptor bindings)
is re-recorded from scratch every single token. The `vkResetDescriptorPool` frees all
previously allocated descriptor sets.

**Current measured cost**: ~0.58 ms CPU record time (not the primary blocker at 38 tok/s,
but becomes proportionally more significant as GPU time shrinks).

llama.cpp records the decode command buffer once, then re-submits with updated push
constants via `vkCmdPushDescriptorSetWithTemplate` (VK_KHR_push_descriptor).

**Fix options**:
1. Pre-record command buffer at init time (requires pre-allocated descriptor sets)
2. Use VK_KHR_push_descriptor to update buffer bindings inline without descriptor alloc
3. At minimum, only re-record when layer structure changes (it doesn't — same every token)

**Estimated impact**: ~2-4 ms saved (CPU-side), but enables GPU-CPU pipelining which
could hide remaining CPU overhead entirely.

### Finding 4: Global Memory Barriers (`command.zig:259-264`)

```zig
pub fn computeBarrier(self: *const CommandBuffer) void {
    self.recordMemoryBarrier(computeBarrierSpec());
}
```

Every `computeBarrier()` call issues a full global `VkMemoryBarrier`, which stalls ALL
GPU memory traffic. These are called after every single dispatch — ~100+ per token.

llama.cpp uses buffer-specific barriers (`VkBufferMemoryBarrier`) that only wait on the
specific buffer that was written, allowing independent dispatches to proceed.

For the serial decode loop this is moderate impact (each step depends on previous), but
it would help in the MoE batched path where gate and up projections are independent.

**Fix**: Replace global barriers with buffer-specific barriers where possible.

**Estimated impact**: ~1-2 ms.

### Finding 5: CPU Embedding Dequant (`forward.zig:1403-1418`)

```zig
fn embedToken(self: *InferenceEngine, token_id: u32) !void {
    ...
    dequantRow(mmap[data_start..], safe_id, hidden_dim, embd.info.type_, staging_f32[0..hidden_dim]);
}
```

Every token requires CPU-side dequantization of the embedding row from the mmap data,
followed by a GPU upload. For Q4_K with hidden_dim=2048: ~144 bytes quantized expanded
to 8192 bytes of F32, then copied to staging buffer and uploaded.

llama.cpp uses GPU-side embedding lookup where possible, uploading only the 4-byte
token ID instead of the full 8KB dequantized row.

ZINC has `embed_dequant_q4k.comp` already written but marked "future" and not wired
into the decode path.

**Fix**: Wire the GPU embedding shader into `embedToken()`, upload only token_id.

**Estimated impact**: ~0.5-1 ms.

### Finding 6: MoE Expert Shared Input Re-Loading (`forward.zig:1984-2080`)

The GPU MoE path dispatches all 8 experts via Y workgroups (line ~2033, `n_used` as Y
dimension). Each expert's workgroup loads the input vector into shared memory cooperatively.

Since all experts share the same input (`x_expert_stride=0`), the input vector is loaded
8 times across 8 Y workgroups — 8x redundant shared memory loads.

llama.cpp processes experts serially on GPU in a single kernel loop, amortizing the
input vector load once across all experts.

**Fix**: Either loop over experts inside a single kernel, or use a two-phase approach
where the input is loaded to shared memory once by a coordinator workgroup.

**Estimated impact**: ~0.5 ms.

### Finding 7: Q8_0 Shader Strided Access Pattern (`dmmv_q8_0.comp:44-100`)

The Q8_0 shader uses a strided pattern where each thread processes blocks tid, tid+64,
tid+128, etc. for 2 rows, then reduces via subgroup operations. This is less
cache-friendly than the Q4_K shader's 1-thread-per-row approach because:

- All 64 threads hammer the same weight blocks simultaneously → L1 cache contention
- Strided access pattern is bad for cache line utilization
- Subgroup reduction adds overhead vs direct thread-to-output mapping

llama.cpp uses a 1-subgroup-per-row approach where each subgroup (wave) handles one
output row cooperatively, with better vectorized weight loading.

**Fix**: Evaluate 1-subgroup-per-row pattern for Q8_0, matching Q4_K approach.

**Estimated impact**: moderate, affects all q8_0 projections (SSM, shared expert, router).

### Finding 8: No Cooperative Matrix for Decode

`coop_matmul.comp` exists but is marked "future" for batched prefill. For decode
(single-token matmul-vec), cooperative matrix is not applicable — DMMV is
memory-bandwidth-bound, not compute-bound. This is the correct design choice.

The real question is whether DMMV bandwidth utilization is high enough on the shapes
that matter. For Q4_K at 144 bytes/256 elements = 0.5625 bytes/element, a single DMMV
with M=2048 rows reads ~1.15 MB weights + 8 KB input. At 576 GB/s theoretical minimum
is ~2 us per DMMV. With ~40 DMMVs per layer and 40 layers, that's ~3.2 ms theoretical
minimum vs 26.3 ms observed — the gap is synchronization and kernel overhead.

### Finding 9: ssm_delta_net Occupancy

`ssm_delta_net.spv` shader stats from RADV:
- 48 VGPRs
- 1536 bytes LDS
- 441 instructions
- Inverse throughput 569

This limits GPU occupancy. At 6.2% bandwidth utilization it's the least efficient kernel,
but accounts for 3.08 ms (11.7% of token time) because the delta-net state update is
compute-bound (O(d_inner × d_state) recurrent update) rather than memory-bound like DMMV.

Already improved from 7.71 ms → 3.08 ms via row-tiled rewrite. Further gains require
reducing VGPR pressure or restructuring the compute pattern.

## Summary: Root Cause Priority

| Priority | Finding | Est. Impact | Fix Difficulty |
|----------|---------|-------------|----------------|
| **#1** | SSM CPU fallback (if active) | ~15-20 ms | Low — remove architecture guard |
| **#2** | DMMV kernel bandwidth on medium shapes | ~5-8 ms | Medium — kernel tuning |
| **#3** | Descriptor set alloc/write per dispatch | ~3-5 ms | Medium — pre-allocate |
| **#4** | Command buffer re-recording | ~2-4 ms | Medium — pre-record |
| **#5** | Global memory barriers | ~1-2 ms | Low — buffer barriers |
| **#6** | CPU embedding dequant | ~0.5-1 ms | Low — GPU embedding shader |
| **#7** | MoE shared input re-loading | ~0.5 ms | Low — adjust kernel |

If Finding #1 (SSM CPU fallback) is active, fixing it alone could push from ~38 to ~70-90 tok/s.
Combined with #2-#4, ZINC should approach 90-100+ tok/s.

## Decode Loop Data Flow (forward.zig)

```
Token ID
  → CPU dequant embedding (token_embd.weight)           [Finding 5]
  → Upload to hidden_buf via staging
  → For each layer 0..39:
      ├─ RMS norm (hidden → norm_buf)
      ├─ QKV projection (DMMV: attn_q.weight × norm_buf → q/k/v)
      ├─ IF attention layer (every 4th):
      │   ├─ Deinterleave Q+gate → separate buffers
      │   ├─ Sigmoid gate × Q
      │   ├─ RoPE on Q and K (partial, rope_dim=64 of head_dim=256)
      │   ├─ Write K,V to KV cache at position
      │   ├─ Flash attention (Q × cached K/V → attn_out)
      │   └─ Output projection (DMMV: o_proj × attn_out)
      ├─ ELSE SSM layer:
      │   ├─ Conv1d (GPU shader)                          [Finding 1]
      │   ├─ Delta-net state update (GPU shader)          [Finding 9]
      │   ├─ Gated RMS norm (GPU shader)
      │   └─ Output projection (DMMV: ssm_out × ssm_hidden)  [Finding 7]
      ├─ Residual add (hidden += layer_output)
      ├─ FFN norm (RMS norm)
      ├─ MoE routing:
      │   ├─ Gate projection (DMMV: gate_exps × ffn_norm → router_logits)  [Finding 2]
      │   ├─ GPU softmax + top-k → expert IDs + weights
      │   └─ For each expert: gate+up (DMMV) → SwiGLU → down (DMMV) → scale_accumulate
      │       (input vector loaded 8x across expert workgroups)            [Finding 6]
      ├─ Shared expert: same structure, single expert, sigmoid-gated
      └─ Residual add (hidden += ffn_output)
  → Final RMS norm
  → LM head projection (DMMV: output.weight × norm → logits)
  → GPU argmax → readback 4-byte token ID
  → Next token
```

## What Hasn't Worked (Dead Ends)

From `DECODE_THROUGHPUT_PLAN.md`:

- Staging the whole `x` vector in LDS for `q8_0` — regressed
- Forcing different `q8_0` inner-loop reuse patterns — regressed
- Removing full-softmax from `softmax_topk` — noise
- Spin-wait on fences (`vkGetFenceStatus`) — no improvement
- subgroupBallot in `softmax_topk` — crashes RADV at wave64
- wave32 for DMMV — regression (wave64 is optimal for memory-bound)
- WG_SIZE_LARGE=256 — regression
- `rm_kq>2` — 75% regression
- Clock forcing — 23% regression
- f16 KV cache — no improvement
- THP (transparent huge pages) — no improvement

## Path to 107+ tok/s

### Short term (38 → 50 tok/s, already planned)

The `DECODE_THROUGHPUT_PLAN.md` has a 7-phase plan targeting 50 tok/s:

1. Keep the profiler, use it as the gate
2. Attack hot kernels: router/top-k → shared expert → attention/SSM
3. Eliminate descriptor churn (pre-allocate persistent descriptor sets)
4. Command buffer replay (pre-record skeleton, only push constants dynamic)
5. Benchmark real decode shapes against RDNA4 reference utilization
6. Audit barriers, remove unnecessarily strong global barriers
7. Concurrency (after 50 tok/s)

### Medium term (50 → 80 tok/s)

- Fused MoE expert kernel (gate+up+SwiGLU+down in one dispatch per expert)
- GPU embedding lookup (eliminate 8KB CPU→GPU upload per token)
- Shader-stats-guided kernel tuning (reduce VGPR pressure, improve occupancy)
- Buffer-specific barriers where independent dispatches exist

### Long term (80 → 107+ tok/s)

- Weight caching in L2/Infinity Cache (llama.cpp likely benefits from this)
- Multi-queue overlap for independent layer operations
- Persistent threads for small-shape kernels
- RDNA4-specific tuning (no engine currently has this)
- Integer dot product hardware path (depends on glslc/RADV fix)

## Key Source References

| File | Line(s) | What |
|------|---------|------|
| `forward.zig` | 1461-1463 | Command buffer reset + re-record every token |
| `forward.zig` | 1931 | SSM GPU path gate (`config.architecture != .qwen35`) |
| `forward.zig` | 1940-1944 | SSM GPU vs CPU dispatch |
| `forward.zig` | 2558 | Per-dispatch descriptor set allocation |
| `dmmv.zig` | — | DMMV dispatch with per-call descriptor alloc |
| `command.zig` | 259-264 | Global `VkMemoryBarrier` in `computeBarrier()` |
| `llama_cpp_analysis.md` | — | llama.cpp Vulkan backend analysis |
| `DECODE_THROUGHPUT_PLAN.md` | — | Current optimization roadmap |
| `RDNA4_TUNING.md` | — | Hardware reference and llama.cpp profiling data |
| `GPU_REFERENCE.md` | — | RDNA3/RDNA4 hardware reference |

## Environment Requirements (Critical for Reproducing)

- Mesa 25.0.7 pinned (25.2.8 causes ~14% RADV regression)
- GECC disabled (`amdgpu.ras_enable=0`)
- `RADV_PERFTEST=coop_matrix` set
- System glslc only (shaderc 2023.8; newer causes 5x regression)
- llama.cpp build 3306dba with `-DGGML_VULKAN=ON -O3 -march=znver4`
- llama.cpp flags: `-ngl 99 --device Vulkan0 --parallel 4 -c 32768 -ctk q8_0 -ctv q8_0 -b 4096 -ub 1024 --mlock --flash-attn`
