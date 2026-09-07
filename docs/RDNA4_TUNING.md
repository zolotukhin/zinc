# RDNA4 Tuning Guide

> This guide mixes general RDNA4 platform findings, driver/toolchain notes, and external baseline measurements. Treat it as tuning context, not as the current ZINC benchmark leaderboard.

Findings from extensive profiling of LLM inference on AMD Radeon AI PRO R9700 (RDNA4, gfx1201). The guide covers two backend paths: the **Vulkan backend** through Mesa RADV (current production, `-Dbackend=vulkan`), and the **ZINC_RT backend** built on direct PM4 submission through `amdgpu` (M0 scaffolding in tree, `-Dbackend=zinc_rt`). Every firmware, kernel, and driver setting documented below applies to both backends, because both sit on top of the same `amdgpu` kernel driver.

## Current measured peak (Vulkan backend, Qwen3.6-35B-A3B Q4_K_XL)

| Metric | ZINC (Vulkan + RADV) | llama.cpp (Vulkan) | Ratio |
|---|---:|---:|---:|
| Decode tok/s | 117.07 | 104.47 | 1.12x |
| Prefill tok/s | 88.08 | 181.95 | 0.48x |
| Bandwidth utilization (decode) | 31% of 576 GB/s | ~28% | — |
| Per-token weight traffic | ~1.57 GiB | ~1.57 GiB | — |

Decode beats llama.cpp by 12%. Prefill still trails — the hybrid MoE plus SSM architecture exposes more parallelism than the current dispatcher exploits, and the dense-model batched-prefill work is covered in [How ZINC's RDNA4 batched prefill went from 42 to 208 tok/s](/blog/2026-06-05-how-zinc-rdna4-batched-prefill-went-from-42-to-208-tok-s/). The remaining hybrid work lives in the ZINC_RT chunked-prefill plan in `docs/ZINC_RT_DESIGN.md` §18.7. The 31% bandwidth utilization at peak decode is the headroom number to watch — the theoretical ceiling at 100% BW is ~365 tok/s, and the path to closing the gap is no longer one missing fused kernel.

## Hardware Specifications
- **GPU**: 64 CUs, wave64, 32KB L0 vector cache/CU, 8MB L2
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

**Measured impact**: 101 tok/s → 110 tok/s (+9%) on Qwen3.6-35B-A3B Q4_K

## RADV Driver Configuration

```bash
# Enable cooperative matrix support
export RADV_PERFTEST=coop_matrix
```

Without this, all matmul operations fall back to scalar shaders — massive performance loss.

## Per-Token Decode Profiling

Profiled with `GGML_VK_PERF_LOGGER=1` on Qwen3.6-35B-A3B (Q4_K_XL, SSM+attention hybrid MoE).

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

### Q5_K DMMV row packing (recent)

The Q5_K dequantize-matmul-vector shader was rebuilt to pack two output rows per workgroup. On a 32-row Q5_K matmul this halves the workgroup count, doubles the per-WG accumulator footprint, and reuses the same dequantized block across both rows. Net effect on decode is a measurable few percent on Q5_K-heavy models like Qwen3-8B, with no correctness changes. The pattern is documented in `src/shaders/dmmv_q5k.comp`; the same row-packing idea is the next candidate for Q3_K and Q6_K, both of which currently dispatch one row per WG.

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

Linear scaling — the GPU is not saturated by a single decode request. Aggregate scaling beyond four slots requires paged KV (the current Vulkan backend's flat KV cache OOMs at sixteen concurrent slots on Qwen3.6-35B-A3B). The paged KV v2 layout in `docs/ZINC_RT_DESIGN.md` §19 targets ~2 100 aggregate tok/s at sixteen slots on the same R9700.

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

## Mesa Version Sensitivity

The RADV ACO compiler in Mesa is the SPIR-V to PM4 path. Two version cliffs are documented:

| Mesa version | Status on R9700 | Notes |
|---|---|---|
| 25.0.7 | Recommended | Current bench-node version |
| 25.2.8 | **~14% RADV regression** | Avoid until upstream ACO regressions are reverted |

The regression manifests on the Q4_K DMMV path most strongly. We pin Mesa via the system package manager and do not auto-upgrade the bench node.

## Load-time Q4_0 Re-quantization

GGUF models commonly ship Q8_0 for tensors the publisher considered precision-sensitive: SSM in-projection (`attn_qkv`), SSM out-projection, attention gate, and sometimes the router. On Qwen3.6-35B-A3B the SSM `attn_qkv` is `[8192, 2048]` per layer in Q8_0, ~17 MiB, streamed every decode token. Across 30 SSM layers that is ~510 MiB of Q8_0 weights traversed per token, well past L3.

On the T-CPU autopilot path, adding `attn_qkv` to the Q8_0 → Q4_0 re-quantize-at-load list moved decode from 32.8 to 37.6 tok/s and prefill from 28.5 to 32.7 tok/s on the 9800X3D bench, with output staying coherent. The model is already Q4_K everywhere else; per-weight noise on the SSM in-proj is averaged out by the L2-normalized delta-net recurrence one layer downstream. The same lever applies on RDNA4 wherever a tensor is loaded Q8_0 by default and the kernel has a Q4_0 variant. See `forward_zinc_rt.zig`'s `q4_candidates` list for the current set.

## NextN/MTP speculative decoding (Qwen 3.8 GGUFs with an appended NextN block)

Qwen 3.8 GGUFs that carry the model's NextN block use it automatically on the
Vulkan backend, in the CLI and in the OpenAI-compatible server. Each cycle
drafts two tokens with the NextN block, verifies the seed plus drafts in one
3-token pass of the full model, restores the DeltaNet/conv state at the
accepted boundary when a draft is rejected, and catches the NextN block up over
the committed rows. Greedy output is bit-identical to ordinary decode.

R9700, Qwen 3.8 27B Q4_K_M, 96 greedy tokens, server path:

| | decode tok/s |
|---|---:|
| `ZINC_MTP=0` | 33.0 |
| `ZINC_MTP=1` (default) | 49.9 |

Scope and knobs:

- Greedy requests only (default `temperature`, `top_p`, repetition penalty);
  sampled requests take the ordinary path.
- Server requests that reuse a cached prompt prefix (`session_id` clients)
  fall back to ordinary decode for now; a full-prompt prefill is required to
  prime the NextN block.
- `ZINC_MTP=0` disables MTP. `ZINC_MTP_COLS=0` disables the column-parallel
  Q4_K/Q5_K/Q6_K matvec route used by the verification batch (A/B only);
  `ZINC_MTP_COLS_ROWS` (2/4/8, default 4) and `ZINC_MTP_LMHEAD_ROWS` (default 8)
  pick its rows-per-workgroup variants.
- `ZINC_MTP_DRAFT_VOCAB=<rows>` limits the draft lm-head to the first N
  (frequency-ordered) vocabulary rows; default 131072 for vocabularies above
  160K rows, 0 = all rows. Verification always scores the full vocabulary, so
  this only trades draft acceptance for draft cost.
- `ZINC_MTP_DP4A=1` runs the Q4_K verification matvecs on int8 dp4a with
  Q8_1-quantized activations: ~3% faster, but the output is no longer
  bit-identical to plain decode (off by default).
- `ZINC_MTP_DRAFTS` (1–3, default 2) sets the draft window.
- `ZINC_MTP_PROFILE=1` with `--profile` prints per-phase GPU totals for the
  verification pass (one submit per layer function, slower).
- Pass `-c <tokens>` on the CLI: its auto-sized context fills VRAM to the
  margin and slows the state restores; the server planner leaves headroom.
- `ZINC_VK_QUEUE_FAMILY=<n>` overrides the Vulkan queue family (diagnostic;
  families 0 and 1 perform the same on the R9700).

## ZINC_RT, this guide, and what changes

The Vulkan-specific advice above is about driver, firmware, and toolchain. **All of it still applies under ZINC_RT** because ZINC_RT uses the same `amdgpu` kernel driver. Disable GECC. Stay on Mesa 25.0.7 (when running anything that links libvulkan, including dev tooling and CI shader compilation). Stay on kernel 6.14 if you can. Pin shaderc 2023.8.

What ZINC_RT changes is the userspace layer above the kernel driver. The Vulkan tax this guide measures — 33 µs per `vkQueueSubmit` plus fence, 80 µs command-buffer re-record on a 1500-node graph, the 5x glslc regression risk — does not apply to ZINC_RT's PM4-direct path. ZINC_RT submits via a ring-buffer write plus a doorbell MMIO store, total CPU-to-CP latency around 150-500 ns. The trade is that ZINC_RT is at M0 today (T1 KFD smoke dispatch verified on R9700; full IR lowering is M2). The Vulkan path is the production target until ZINC_RT clears its M1 gate at 140 tok/s decode.

For the long-form story of why ZINC_RT exists at all, the head-to-head architectural comparison vs ROCm and Vulkan, the multitenant batching architecture, and the falsification criteria, see:

- `docs/ZINC_RT_DESIGN.md` — the canonical ZINC_RT design
- The blog post "ROCm vs Vulkan vs ZINC_RT: inside the decision to write our own GPU runtime for local LLM inference on AMD RDNA4" at `/blog/inside-the-decision-to-write-our-own-gpu-runtime-for-local-llm-inference`

The two backends will both ship indefinitely. The cross-backend logit-equality test in CI is what keeps them honest.
