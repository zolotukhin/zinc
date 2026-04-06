# RDNA4 Performance Plan: Qwen3.5-35B-A3B → 100+ tok/s

**Target**: Beat llama.cpp (102 tok/s) on AMD RX 9070 (RDNA4, 576 GB/s, 64 CUs)
**Current**: 11 tok/s (91 ms/tok) — 10x gap
**Model**: Qwen3.5-35B-A3B MoE — 40 layers, 256 experts (top-8), hybrid SSM+attention

## 1. Theoretical Limits

### Bytes read per token

Not all 21 GB is read — MoE only activates 8/256 experts:

| Component | Layers | Quant | Per-token bytes |
|-----------|--------|-------|-----------------|
| SSM projections (qkv+gate+alpha+beta+out) | 30 | Q8_0 | 420 MB |
| Shared expert (gate+up+down) | 40 | Q8_0 | 240 MB |
| LM head | 1 | Q4_K | 289 MB |
| MoE experts (8/256, gate+up+down) | 40 | Q4_K/Q5_K | 192 MB |
| Attention Q/K/V/O | 10 | Q4_K | 120 MB |
| MoE router | 40 | F32/Q4_K | 12 MB |
| Norms, conv kernels, SSM constants | 40 | F32 | ~5 MB |
| **Total** | | | **~1.28 GB** |

### Speed limits

| Metric | Value |
|--------|-------|
| Peak BW | 576 GB/s |
| Theoretical max | 1.28 GB / 576 GB/s = **2.22 ms = 450 tok/s** |
| Realistic ceiling (35-40% BW) | ~170-200 tok/s |
| llama.cpp achieved | 102 tok/s (9.8 ms, ~131 GB/s = 23% BW) |
| **ZINC current** | **11 tok/s (91 ms, ~14 GB/s = 2.4% BW)** |

## 2. Where Time Is Spent Now

### The SSM CPU roundtrip (the dominant bottleneck)

30 of 40 layers use the CPU SSM path (`runSsmLayerCpu`). Each layer does:

1. Record 4 DMMV dispatches
2. Record GPU→host copy
3. **`submitAndWait`** — flush GPU, stall CPU (~1.5 ms per sync)
4. CPU conv1d (8192 channels × 4 taps)
5. CPU delta-net (32 heads × 128×128 state update)
6. CPU gated norm
7. Upload result, restart command buffer
8. Continue MoE in new command buffer

This creates **30 GPU pipeline flushes per token**.

### Estimated breakdown at 91 ms/tok

| Component | Time | % |
|-----------|------|---|
| SSM CPU roundtrips (30 × submitAndWait) | ~45-55 ms | 55% |
| SSM CPU compute (conv1d + delta-net) | ~10-15 ms | 13% |
| MoE GPU path (40 layers) | ~10-12 ms | 12% |
| Shared expert (40 layers) | ~5-6 ms | 6% |
| Attention (10 layers) | ~3-4 ms | 4% |
| Command buffer overhead (30+ re-recordings) | ~3-5 ms | 4% |
| Final tail (norm + LM head + argmax) | ~1 ms | 1% |
| **Total** | **~91 ms** | |

## 3. Optimization Roadmap

### Phase 0: Re-enable GPU SSM (11 → 35-40 tok/s) — 1 day

The GPU SSM shaders already exist and were working before the delta-net guard was added. The fix:

```zig
// forward.zig line 1997-1998
// FROM:
const has_delta_net = config.full_attn_interval > 1;
const use_gpu_ssm = self.elementwise.pipeline_ssm_conv1d != null and !has_delta_net;
// TO:
const use_gpu_ssm = self.elementwise.pipeline_ssm_conv1d != null and
    self.elementwise.pipeline_ssm_delta_net != null and
    self.elementwise.pipeline_ssm_gated_norm != null;
```

This eliminates all 30 `submitAndWait` calls, keeping the entire decode in a single command buffer. Historical data shows 38 tok/s when this path was active.

**Validation**: Compare GPU delta-net output vs CPU reference at layer 0 position 0 using existing diagnostic infrastructure.

### Phase 1: Kernel-Level Optimizations (38 → 55-65 tok/s) — 1-2 weeks

**1a. Q4_K MoE DMMV — packed reads**

Current `dmmv_q4k_moe.comp` uses `uint8_t` byte access. Each thread reads individual bytes from scattered addresses — terrible for coalescing. The non-MoE `dmmv_q4k.comp` uses packed `uint` reads.

Fix: Rewrite MoE shader to load Q4_K blocks as 36 `uint32` words instead of 144 individual bytes.

Expected: 1-3 ms saved across 40 MoE layers.

**1b. Fuse router DMMV + softmax_topk**

Router output (256 floats) fits in shared memory. A fused kernel computes the 256 dot products and immediately selects top-8, eliminating one barrier + dispatch per layer.

Expected: ~3-4 ms saved across 40 layers.

**1c. Fuse shared expert gate+up**

Both read the same input (`ffn_norm_buf`). A fused kernel reads the input once and writes both gate and up outputs.

Expected: ~1-2 ms saved.

### Phase 2: Infrastructure (55 → 70-80 tok/s) — 1-2 weeks

**2a. Pre-allocate descriptor sets**

Currently ~1000 descriptor sets are allocated per token. Since buffer bindings are fixed, pre-allocate at init time.

Expected: ~2-3 ms CPU savings.

**2b. Pre-record command buffer skeleton**

The command buffer structure is identical every decode step. Pre-record once, replay with push constant updates for position-dependent values.

Expected: ~0.5 ms + enables GPU-CPU pipelining.

**2c. Buffer-specific barriers**

Replace global `VkMemoryBarrier` with buffer-specific barriers where dispatches are independent (gate/up, expert batches).

Expected: ~1-2 ms by allowing overlapped execution.

### Phase 3: Advanced (70 → 100+ tok/s) — 2-4 weeks

**3a. Fused MoE expert kernel**

Single dispatch per MoE layer: gate+up → SwiGLU → down with intermediates in shared memory/registers.

**3b. Delta-net occupancy improvement**

Tile the 128×128 state matrix update across multiple warps. Current implementation is sequential per-column.

**3c. Multi-queue overlap**

Use separate compute queues for independent operations (e.g., SSM on queue 1, MoE prefetch on queue 2).

**3d. Integer dot product (IDP) path**

RDNA4 supports IDP. Once RADV+glslc compatibility is confirmed, use it for Q4_K/Q8_0.

### Milestones

| Phase | Target | Key change |
|-------|--------|------------|
| Phase 0 | 35-40 tok/s | GPU SSM enabled |
| Phase 1 | 55-65 tok/s | Packed MoE DMMV, fused router+topk |
| Phase 2 | 70-80 tok/s | Pre-alloc descriptors, buffer barriers |
| Phase 3 | 100+ tok/s | Fused MoE, delta-net occupancy, multi-queue |

## 4. Micro-benchmarks

Each layer type needs an isolated benchmark to measure throughput and identify the gap from theoretical peak.

### 4.1 DMMV Shape Sweep

Measure bandwidth utilization for every DMMV shape used in the model:

| Test | M | K | Quant | Expected BW% |
|------|---|---|-------|-------------|
| ssm_qkv | 8320 | 2048 | Q8_0 | 60-70% |
| ssm_out | 2048 | 4096 | Q8_0 | 50-60% |
| attn_q | 8192 | 2048 | Q4_K | 55-65% |
| attn_o | 2048 | 4096 | Q4_K | 45-55% |
| moe_expert_gate | 512 | 2048 | Q4_K | 30-40% |
| moe_expert_down | 2048 | 512 | Q5_K | 25-35% |
| shared_gate | 512 | 2048 | Q8_0 | 35-45% |
| lm_head | 248320 | 2048 | Q4_K | 65-75% |
| moe_router | 256 | 2048 | F32 | 15-25% |
| ssm_alpha | 32 | 2048 | Q8_0 | 5-10% |

Small shapes (M<256) will have very low utilization — these are candidates for kernel fusion.

### 4.2 MoE Expert Batched Dispatch

Compare dispatch strategies for 8 experts:
- 8 Y-workgroups (current batched path)
- 8 sequential dispatches (baseline)
- 1 fused expert kernel (target)

### 4.3 SSM Layer Isolation

Run a single GPU SSM layer (proj → conv1d → delta-net → gated_norm → out → residual) in a tight loop with GPU timestamps at each stage.

### 4.4 Barrier Cost

Measure time for N dispatches with:
- Global compute barriers (current)
- Buffer-specific barriers (target)
- No barriers (theoretical minimum)

### 4.5 Command Buffer Recording

Measure CPU time to record the full 40-layer command buffer without submitting. Identifies descriptor allocation and recording overhead.

## 5. How to Run Benchmarks

```bash
# Build with profiling
zig build -Doptimize=ReleaseFast

# Run with per-phase profiling
ZINC_PROFILE=1 ./zig-out/bin/zinc -m model.gguf --prompt "Hello" --chat -n 100

# Run isolated DMMV shape benchmark (when implemented)
./zig-out/bin/zinc-bench-shapes --device 0 --iterations 1000

# Compare with llama.cpp
/root/llama.cpp/build/bin/llama-bench -m model.gguf -ngl 999 -n 128
```

## 6. Critical Files

| File | What to change |
|------|---------------|
| `src/compute/forward.zig:1997` | SSM path guard |
| `src/shaders/dmmv_q4k_moe.comp` | Byte→uint32 reads |
| `src/shaders/softmax_topk.comp` | Fuse with router |
| `src/shaders/ssm_delta_net.comp` | Occupancy tuning |
| `src/bench_hot_decode.zig` | Extend with shape sweep |
| `src/compute/dmmv.zig` | Fused expert dispatch |
