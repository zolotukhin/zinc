# Feature Specification: Apple Silicon Inference (Metal Backend)

**Feature Branch**: `005-apple-silicon-inference`
**Created**: 2026-03-28
**Status**: Draft
**Hardware**: Mac Studio M4 Max, 64 GB, 546 GB/s, 40-core GPU
**Model**: Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf (20.7 GB, MoE 35B/3B active)

## Problem Statement

ZINC currently only runs inference on Linux with Vulkan (RDNA4). Apple Silicon has 546 GB/s unified memory bandwidth on M4 Max — enough for 65-120 tok/s on this MoE model — but there is no Metal backend. Adding Apple Silicon support unlocks:

1. Local development and testing without SSH to the remote node
2. A large market of Mac users who want fast local inference
3. Competitive advantage: current engines leave 25-45% of bandwidth on the table

## Competitive Analysis

| Engine | Architecture | Single-req tok/s (M4 Max, Qwen3.5-35B-A3B Q4) | Parallel support |
|--------|-------------|-----------------------------------------------|------------------|
| llama.cpp Metal | C → ggml → Metal | ~35 | No batching |
| mlx-lm | Python → MLX → Metal | ~65 | No (sequential) |
| vllm-mlx | Python → MLX → Metal | ~65 (single) / ~160 (5 req) | Continuous batching |
| MetalRT | C++ → Metal direct | ~75-80 | Unknown |
| **ZINC (target)** | **Zig → Metal direct** | **80+** | **Continuous batching** |

### Why ZINC Can Be Fastest

1. **Zig → Metal direct**: No Python interpreter, no framework abstraction. Same "straight to Metal" philosophy as MetalRT but with Zig's zero-overhead FFI
2. **Static graph pre-recording**: Command buffers recorded once, replayed per token. MLX rebuilds the graph dynamically every step
3. **Zero-copy everything**: UMA means mmap → Metal buffer, no staging, no DMA
4. **GGUF K-quants**: Better quality than MLX's affine quants at same bit width, with custom MSL dequant kernels
5. **Batched expert dispatch**: MoE routing can batch same-expert calls across parallel requests

## User Scenarios & Testing

### User Story 1 — Fast Single-Request Decode (Priority: P1)

A user runs ZINC on their Mac with a large MoE model and gets fast, responsive text generation — faster than any other local engine.

**Acceptance Criteria**:
1. **Given** ZINC loaded with Qwen3.5-35B-A3B Q4_K_XL on M4 Max 64 GB, **When** generating 256 tokens, **Then** decode throughput is ≥80 tok/s
2. **Given** the same setup, **When** generating text, **Then** output is identical to the Vulkan backend (bitwise-identical logits not required, but same greedy-sampled tokens)
3. **Given** the same setup, **Then** model loads in <2 seconds (mmap, zero-copy)

### User Story 2 — Parallel Request Serving (Priority: P2)

A developer runs ZINC as a local API server and sends multiple concurrent requests, achieving higher aggregate throughput than any competing engine.

**Acceptance Criteria**:
1. **Given** 5 concurrent requests, **Then** aggregate throughput is ≥200 tok/s
2. **Given** parallel requests, **Then** each individual request still gets ≥40 tok/s (no starvation)
3. **Given** the OpenAI-compatible API, **Then** responses stream correctly with SSE
4. **Given** 16 concurrent requests (max supported), **Then** server does not OOM and gracefully rejects request 17+

### User Story 3 — Cross-Platform GGUF (Priority: P1)

The same GGUF model file works on both the RDNA4 Linux node (Vulkan) and the Mac Studio (Metal) with no conversion or re-quantization.

**Acceptance Criteria**:
1. **Given** `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf`, **When** loaded on Metal, **Then** all tensors parse correctly and dequantize to the same values as the Vulkan path

## Architecture Overview

```
src/
├── vulkan/          # Existing Vulkan backend (Linux/RDNA4)
│   ├── instance.zig
│   ├── buffer.zig
│   ├── pipeline.zig
│   └── command.zig
├── metal/           # New Metal backend (macOS/Apple Silicon)
│   ├── shim.m       # Thin ObjC layer exposing C API
│   ├── shim.h       # C header for Zig @cImport
│   ├── device.zig   # MTLDevice, command queue init
│   ├── buffer.zig   # Metal buffer (SharedMode, zero-copy mmap)
│   ├── pipeline.zig # MSL → compute pipeline
│   └── command.zig  # Command buffer recording + dispatch
├── gpu/             # Backend-agnostic GPU abstraction
│   ├── interface.zig # GpuBackend union: .vulkan | .metal
│   ├── buffer.zig   # Unified buffer handle
│   └── dispatch.zig # Unified compute dispatch
├── compute/         # Unchanged — uses gpu/ abstraction
├── model/           # Unchanged — GGUF parsing, tokenizer
└── shaders/
    ├── *.comp       # Existing GLSL (Vulkan)
    └── metal/       # MSL compute kernels
        ├── dmmv_q4k.metal
        ├── dmmv_q8_0.metal
        ├── flash_attn.metal
        ├── rms_norm_mul.metal
        ├── swiglu.metal
        ├── rope_fused.metal
        └── ...
```

### Backend Selection

```
if (builtin.os.tag == .macos) → Metal backend
else if (builtin.os.tag == .linux) → Vulkan backend
```

Compile-time selection via `@import("builtin")`. No runtime polymorphism overhead.

### GPU Abstraction Layer

The `gpu/interface.zig` provides a minimal tagged union:

```zig
pub const GpuBackend = union(enum) {
    vulkan: VulkanInstance,
    metal: MetalDevice,

    pub fn createBuffer(self: *@This(), size: usize, opts: BufferOpts) Buffer { ... }
    pub fn dispatch(self: *@This(), pipeline: Pipeline, grid: [3]u32, push: anytype) void { ... }
};
```

This replaces direct Vulkan calls in `compute/` with backend-agnostic dispatch. The abstraction is **zero-cost at comptime** — Zig's comptime generics mean the backend is resolved at compile time, producing identical codegen to direct Vulkan or Metal calls. No vtables, no switches, no runtime polymorphism. Vulkan path performance must not regress.

### Metal Shader Strategy

**Phase 1**: Cross-compile existing SPIR-V → MSL via SPIRV-Cross for non-cooperative-matrix shaders. Gets basic inference working quickly.

**Phase 2**: Hand-optimized MSL kernels using:
- `simdgroup_matrix` (8x8 matrix ops, available since M1)
- `simdgroup_async_copy` (overlapped compute + memory loads)
- 32-thread simdgroup tuning (vs 64-thread workgroups for RDNA4)
- Dispatch-time workgroup sizes (Metal allows this, Vulkan doesn't)

**Phase 3**: `cooperative_tensor` (Metal 4, MSL 4.0) when M5 becomes available — provides neural accelerator access from compute shaders.

### Zero-Copy Model Loading

```
Current (Vulkan):  mmap → staging buffer → vkCmdCopy → device-local buffer
Metal:             mmap → newBufferWithBytesNoCopy → done
```

The GGUF mmap pointer is wrapped directly as a Metal shared buffer. GPU reads model weights from the same physical pages the OS paged in. No copies, no uploads, no staging.

### Continuous Batching Design

```
Requests → Scheduler → Batch assembly (group by decode step)
                              ↓
                     Pre-recorded command buffer (per batch-size)
                              ↓
                     Single GPU submit (all sequences)
                              ↓
                     Logits readback → per-sequence sampling
                              ↓
                     Token emission (SSE streaming)
```

KV cache uses paged allocation (16-token pages, same as Vulkan path). Pages are shared-mode Metal buffers — no CPU-GPU sync needed for page table updates.

**Context length**: Default 4096 tokens (matching Vulkan). Configurable up to 32768 via `--context-length` CLI flag. At 32K with F32 KV, cache requires ~2.5 GB — well within 64 GB headroom after model weights.

## Hardware Requirements

- **Minimum**: Any Apple Silicon Mac (M1+), macOS 14 Sonoma+
- **Baseline Metal features** (M1+): `simdgroup_matrix` 8x8, threadgroup shared memory, compute shaders
- **Enhanced** (M2+): `simdgroup_async_copy` for overlapped memory loads in flash attention
- **Future** (M5+): `cooperative_tensor` / Metal 4 TensorOps for neural accelerator access from compute shaders
- Chip-specific optimizations selected at runtime via `gpu_detect.zig` (same pattern as RDNA3/4 detection on Vulkan path)

## Error Handling

Fail-fast at startup with actionable messages:
- **No Apple Silicon**: "Error: ZINC Metal backend requires Apple Silicon (M1+). This machine has [chip]. Exiting."
- **Model exceeds memory**: "Error: Model requires ~X GB but only Y GB available. Use a smaller quantization or shorter context length."
- **Metal unavailable**: "Error: Metal framework not found. Ensure macOS 14+ and Xcode Command Line Tools are installed."
- **Max requests exceeded**: Server returns HTTP 503 with `{"error": "max concurrent requests (16) reached"}` — does not crash.
- **GPU command timeout**: Log error, abort current request, remain available for new requests.

No CPU fallback. No silent degradation.

## Non-Goals (This Spec)

- Neural Engine / ANE integration (requires CoreML, limited context, not competitive for >1B models yet)
- Distributed multi-Mac inference (Exo-style clustering)
- Training or fine-tuning
- Non-Apple GPU support on macOS (eGPU is dead)
- CPU fallback inference (fail fast instead)

## Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| SPIRV-Cross output is slow | Medium | Delays Phase 1 | Start with 2-3 hand-written MSL kernels for critical path (dmmv_q4k, flash_attn) |
| ObjC shim adds overhead | Low | Minor | Measure — llama.cpp proves this is negligible |
| 64 GB not enough for Q4 + long context KV | Low | Limits context | KV quantization (TurboQuant, already designed) |
| simdgroup_matrix slower than expected | Medium | Perf miss | Benchmark vs scalar path early, optimize later |
| MoE expert dispatch is scattered | Medium | Bandwidth waste | Batch same-expert calls, prefetch expert pages |

## Clarifications

### Session 2026-03-28

- Q: Minimum Apple Silicon chip? → A: M1+ (all Apple Silicon), with per-chip optimizations layered on (M2+ async copy, M5+ cooperative tensor)
- Q: GPU abstraction strategy? → A: Thin gpu/ abstraction (buffer + dispatch only) with comptime backend selection. Must be zero-cost — no runtime polymorphism, no vtables, no performance regression on either backend.
- Q: Maximum context length on Metal? → A: 4096 default (match Vulkan), configurable up to 32768 via CLI flag to leverage 64 GB UMA headroom.
- Q: Maximum concurrent requests? → A: 16 concurrent (matches vllm-mlx benchmark ceiling). KV cache memory partitioned dynamically — smaller contexts use less.
- Q: Failure behavior on unsupported/OOM conditions? → A: Fail fast — detect at startup, print actionable error message (e.g., "requires Apple Silicon", "model exceeds available memory"), exit non-zero. No CPU fallback.

## Dependencies

- Xcode Command Line Tools (for ObjC compilation)
- SPIRV-Cross (for Phase 1 shader translation)
- Metal framework (macOS SDK)
- Existing: GGUF parser, tokenizer, compute graph IR, forward loop
