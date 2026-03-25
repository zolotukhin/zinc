# RDNA4 Tuning Guide

Findings from extensive profiling of LLM inference on AMD Radeon AI PRO R9700 (RDNA4, gfx1201).

## Hardware Specifications
- **GPU**: 64 CUs, wave64, 32KB L1/CU, 6MB L2
- **Memory**: 32GB GDDR6, 576 GB/s bandwidth
- **Vulkan**: VK_KHR_cooperative_matrix 16x16x16
- **Architecture**: gfx1201 (detected as AMD_RDNA3 by llama.cpp)

## Critical: Disable GPU ECC (GECC)

RDNA4 enables GECC by default, which silently consumes ~10% memory bandwidth for error correction. For inference workloads where occasional bit flips are acceptable, disabling it gives a significant speedup.

```bash
# Add to /etc/default/grub:
GRUB_CMDLINE_LINUX_DEFAULT="... amdgpu.ras_enable=0"
# Then: update-grub && reboot
```

**Measured impact**: 101 tok/s → 110 tok/s (+9%) on Qwen3.5-35B-A3B Q4_K

## RADV Driver Configuration

```bash
# Enable cooperative matrix support
export RADV_PERFTEST=coop_matrix
```

Without this, all matmul operations fall back to scalar shaders — massive performance loss.

## Per-Token Decode Profiling

Profiled with `GGML_VK_PERF_LOGGER=1` on Qwen3.5-35B-A3B (Q4_K_XL, SSM+attention hybrid MoE).

### Time Breakdown (per token)
| Component | Time (ms) | % of Total |
|-----------|----------|------------|
| Matmul compute | 6.5 | 63% |
| Non-matmul compute | 3.6 | 35% |
| Vulkan dispatch overhead | ~0.1 | <1% |
| **Total** | **~10.2** | |

### Matmul Bandwidth Utilization
| Operation | BW Utilization | Time/token |
|-----------|---------------|------------|
| Vocab output (m=248320, k=2048) | **93.2%** | 1006 us |
| Large attention (m=8192, k=2048) | **83.6%** | 1481 us |
| Medium attention (m=4096, k=2048) | 66.1% | 682 us |
| MoE experts (q4_K, m=512, k=2048) | 59.6% | 1073 us |
| Small matmul (m=32, k=2048) | 2.7% | 272 us |

Large matmuls are near bandwidth-optimal. Small matmuls can't saturate the memory subsystem.

### Non-Matmul Ops (per token)
| Op | Dispatches | Total Time |
|----|-----------|------------|
| RMS_NORM_MUL (fused) | 131 | 593 us |
| MUL (element-wise) | 110 | 365 us |
| GET_ROWS | 122 | 338 us |
| SIGMOID | 80 | 267 us |
| MULTI_ADD (fused) | 80 | 256 us |
| GLU (fused) | 80 | 250 us |
| SILU | 60 | 143 us |
| L2_NORM | 60 | 125 us |
| SSM_CONV | 30 | 150 us |
| GATED_DELTA_NET | 30 | 128 us |

### Compute Graph Stats
- Total graph nodes: 3728
- Dispatchable ops: 2356
- After existing fusions: ~1500 dispatches
- Dispatch overhead: ~0.1ms (negligible — measured 0.016µs per dispatch)

## Vulkan Dispatch Overhead (Micro-benchmark)

Raw Vulkan dispatch cost measured on RDNA4:

| Test | Result |
|------|--------|
| Single dispatch (record+submit+wait) | 33 us |
| 1500 empty dispatches (GPU time) | 24 us = **0.016 us/dispatch** |
| 1500 dispatches (wall time) | 85 us = 0.057 us/dispatch |
| Pre-recorded command buffer replay | 54 us for 1500 dispatches |

**Key insight**: Dispatch overhead is negligible. The 2-5µs per "dispatch" seen in profiling is real kernel execution time on small memory-bound tensors.

## Concurrent Request Scaling

| Concurrent Slots | Per-slot tok/s | Aggregate tok/s |
|-----------------|----------------|-----------------|
| 1 | 110 | 110 |
| 4 | 108 | 432 |

Linear scaling — the GPU is not saturated by a single decode request.

## What Doesn't Help

| Optimization | Result | Notes |
|-------------|--------|-------|
| Wave32 for DMMV | No improvement | Driver's default wave64 is optimal |
| DMMV_WG_SIZE_LARGE (256 threads) | No improvement | Too many idle threads for small K |
| rm_kq > 2 (rows per workgroup) | **-75% regression** | Wave64 can't handle 4+ rows |
| GPU clock forcing (profile_peak) | **-23% regression** | Power throttling on memory-bound work |
| f16 KV cache (vs q8_0) | No change | KV ops are negligible |
| Flash attention on/off | No change | Tiny fraction of decode time |
| CPU thread count (1-16) | No change | Workload is 100% GPU-bound |
| THP=always (vs madvise) | Marginal | Model weights are in GPU VRAM |

## SPIR-V Toolchain Compatibility

**Critical**: Newer versions of shaderc/spirv-tools produce SPIR-V that RADV (ACO compiler) handles poorly — up to 5x slower.

| glslc Version | RADV Compatibility | Performance |
|--------------|-------------------|-------------|
| shaderc 2023.8 (Ubuntu 24.04) | Excellent | 110 tok/s |
| shaderc v2026.2-dev | **Broken** | 19-25 tok/s |

The newer glslc adds `NonWritable`/`NonReadable` decorations and different control flow that RADV's ACO optimizer can't handle efficiently.

**Recommendation**: Use the system-provided glslc from Ubuntu packages, not a custom-built version.

## SMU Firmware Compatibility

Kernel 6.17 has SMU driver IF v0x2e, while RDNA4 firmware expects v0x32. This mismatch limits max GPU clock to 2200 MHz instead of 2350 MHz.

Kernel 6.14 or earlier may have a compatible SMU driver version.
