# Implementation Plan: Apple Silicon Inference (Metal Backend)

**Branch**: `004-apple-silicon-inference` | **Date**: 2026-03-28 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/004-apple-silicon-inference/spec.md`

## Summary

Add a Metal compute backend to ZINC enabling inference on Apple Silicon Macs (M1+), targeting ≥80 tok/s single-request decode and ≥200 tok/s aggregate at 5 concurrent requests for Qwen3.5-35B-A3B Q4_K_XL on M4 Max. Uses an ObjC shim (C API) for Metal access from Zig, zero-copy model loading via UMA, and MSL compute kernels (initially cross-compiled from SPIR-V, then hand-optimized with simdgroup_matrix).

## Technical Context

**Language/Version**: Zig 0.15.2+ (host), MSL / Metal Shading Language (GPU shaders), Objective-C (thin shim)
**Primary Dependencies**: Metal.framework, Foundation.framework, SPIRV-Cross (build tool), Xcode Command Line Tools
**Storage**: GGUF model files (mmap'd, zero-copy via `newBufferWithBytesNoCopy`)
**Testing**: `zig build test` (unit tests), numerical validation against Vulkan backend and llama.cpp reference
**Target Platform**: macOS 14+ on Apple Silicon (M1, M2, M3, M4 families)
**Project Type**: CLI + inference server (OpenAI-compatible API)
**Performance Goals**: ≥80 tok/s single decode, ≥200 tok/s aggregate (5 req), ≥75% bandwidth utilization, <2s model load
**Constraints**: 64 GB M4 Max dev hardware, 20.7 GB model, 4096 default context (configurable to 32768), 16 max concurrent requests
**Scale/Scope**: Single Mac, local inference server, same model/quant formats as Vulkan path

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Performance-First | **PASS** | 80+ tok/s target, 75%+ bandwidth utilization, profiling-driven optimization phases |
| II. RDNA4-Native | **VIOLATION** | Metal is not RDNA4. See Complexity Tracking. |
| III. Zig Systems Correctness | **PASS** | All host code in Zig, ObjC shim is thin C API only, comptime backend selection, explicit buffer management |
| IV. Vulkan-First | **VIOLATION** | Adding Metal as a second GPU backend. See Complexity Tracking. |
| V. Production Serving | **PASS** | Continuous batching, paged KV cache, OpenAI API, 16 concurrent requests, SSE streaming |
| VI. Correctness Validation | **PASS** | Cross-backend greedy-sampled token matching, llama.cpp reference validation |

### Post-Design Re-Check

| Principle | Status | Notes |
|-----------|--------|-------|
| II. RDNA4-Native | **JUSTIFIED** | Metal backend is additive — zero changes to Vulkan path, comptime isolation ensures no RDNA4 regression |
| IV. Vulkan-First | **JUSTIFIED** | Metal is a second first-class backend, not a replacement. Vulkan remains primary on Linux. Constitution principle governs not adding ROCm (which doesn't support consumer RDNA); Metal serves a different hardware family entirely |

## Project Structure

### Documentation (this feature)

```text
specs/004-apple-silicon-inference/
├── plan.md              # This file
├── research.md          # Phase 0 output (completed)
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output (completed)
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── main.zig                    # CLI entry, arg parsing — add --context-length flag
├── gpu/                        # NEW: Backend-agnostic GPU abstraction
│   ├── interface.zig           #   Comptime GpuBackend: .vulkan | .metal
│   ├── buffer.zig              #   Unified buffer handle
│   └── dispatch.zig            #   Unified compute dispatch
├── metal/                      # NEW: Metal backend (macOS only)
│   ├── shim.h                  #   C API header for Zig @cImport
│   ├── shim.m                  #   ObjC implementation (MTLDevice, buffers, pipelines, dispatch)
│   ├── device.zig              #   Metal device init, command queue, chip detection
│   ├── buffer.zig              #   Metal shared-mode buffers, mmap wrapping
│   ├── pipeline.zig            #   MSL → MTLComputePipelineState
│   └── command.zig             #   Command buffer recording + dispatch
├── vulkan/                     # EXISTING: Vulkan backend (Linux only) — unchanged
│   ├── instance.zig
│   ├── buffer.zig
│   ├── pipeline.zig
│   ├── command.zig
│   └── gpu_detect.zig
├── compute/                    # MODIFIED: Use gpu/ abstraction instead of direct vulkan/ calls
│   ├── forward.zig             #   Refactor to use gpu.Buffer, gpu.dispatch()
│   ├── dmmv.zig                #   Backend-agnostic dispatch
│   ├── elementwise.zig
│   ├── attention.zig
│   └── graph.zig               #   Unchanged
├── model/                      # MODIFIED: Loader uses gpu/ for buffer creation
│   ├── gguf.zig                #   Unchanged
│   ├── loader.zig              #   Metal: mmap wrap, Vulkan: staging upload
│   ├── architecture.zig        #   Unchanged
│   └── tokenizer.zig           #   Unchanged
├── scheduler/                  # EXISTING: Extended for Metal
│   ├── scheduler.zig
│   └── request.zig
├── server/                     # EXISTING: HTTP server
│   └── http.zig
└── shaders/
    ├── *.comp                  # EXISTING: GLSL compute shaders (Vulkan)
    └── metal/                  # NEW: MSL compute kernels
        ├── dmmv_q4k.metal
        ├── dmmv_q8_0.metal
        ├── dmmv_q5k.metal
        ├── dmmv_q6k.metal
        ├── dmmv_f16.metal
        ├── dmmv_f32.metal
        ├── flash_attn.metal
        ├── rms_norm_mul.metal
        ├── swiglu.metal
        ├── rope_fused.metal
        ├── deinterleave.metal
        ├── sigmoid_mul.metal
        ├── vadd.metal
        ├── scale_accumulate.metal
        ├── embed_dequant_q4k.metal
        ├── ssm_conv1d.metal
        ├── ssm_delta_net.metal
        └── ssm_gated_norm.metal
```

**Structure Decision**: Metal backend mirrors the Vulkan module layout (device/buffer/pipeline/command) for consistency. A thin `gpu/` abstraction layer sits between `compute/` and the backends, using Zig comptime generics to eliminate runtime overhead. Shaders live in `src/shaders/metal/` parallel to existing `.comp` files.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| II. RDNA4-Native — Metal is not RDNA4 | Apple Silicon has 30%+ of developer market; local dev/test without SSH to remote node; competitive opportunity (current engines underperform) | "RDNA4-only" would permanently exclude macOS users. Metal backend is fully additive — comptime isolation means zero changes to any Vulkan code path. RDNA4 remains first-class. |
| IV. Vulkan-First — Metal is a second GPU backend | Vulkan does not exist on macOS (MoltenVK has prohibitive overhead for compute). Metal is the only viable GPU API on Apple Silicon. | Running Vulkan-over-MoltenVK measured 40-60% slower than native Metal in llama.cpp benchmarks. The Vulkan-First principle was designed to reject ROCm (no consumer RDNA support), not to prevent supporting additional hardware families. |

## Phase 0: Research (Completed)

Research findings documented in [research.md](research.md). Key decisions:

| Decision | Rationale | Alternatives Considered |
|----------|-----------|------------------------|
| ObjC shim (C API) over zig-objc | Production-proven (llama.cpp pattern), debuggable, no dependency on third-party bindings | zig-objc (verbose msgSend), metal-cpp (C++ interop fragile from Zig), auto-generated bindings (unstable) |
| GGUF format (keep existing) | Cross-platform — same file on Vulkan and Metal, K-quants better quality than MLX affine | MLX safetensors (separate model files, worse quant quality) |
| SPIRV-Cross for initial shaders | Fast path to working inference — 15+ shaders translated automatically | Write all MSL from scratch (slower, blocks end-to-end testing) |
| Comptime backend selection | Zero runtime cost, identical codegen to direct calls | Runtime tag dispatch (unnecessary overhead), separate forward.zig (maintenance nightmare) |
| 32-thread workgroups (Metal) | Apple GPU simdgroup width = 32 (vs AMD wave64 = 64) | 64-thread (wastes half simdgroup, lower occupancy) |
| M1+ minimum target | All Apple Silicon supports simdgroup_matrix; maximizes user base | M4-only (excludes most Mac users for no shader benefit) |

## Phase 1: Design & Contracts

### Data Model

See [data-model.md](data-model.md).

### Contracts

See [contracts/](contracts/).

### Quickstart

See [quickstart.md](quickstart.md) — baseline measurement commands for llama.cpp Metal, mlx-lm, and vllm-mlx.

## Implementation Phases

### Phase 1: Metal Backend Foundation

**Goal**: Single token generated correctly on Metal.

1. ObjC shim (`src/metal/shim.m` + `shim.h`) — C API for device, buffer, pipeline, command, dispatch
2. Zig Metal wrappers (`src/metal/device.zig`, `buffer.zig`, `pipeline.zig`, `command.zig`)
3. Build system — macOS: compile shim.m, link Metal+Foundation, skip Vulkan; Linux: unchanged
4. GPU abstraction (`src/gpu/interface.zig`) — comptime generic, refactor compute/ to use it
5. Cross-compile shaders via SPIRV-Cross (critical path: dmmv_q4k, flash_attn, rms_norm_mul, swiglu, rope)
6. Zero-copy model loading — mmap → `mtl_wrap_mmap()` in loader.zig
7. End-to-end single-token decode — validate against Vulkan output

**Milestone**: `zig build && ./zig-out/bin/zinc -m model.gguf --prompt "Hello"` on macOS.

### Phase 2: Performance Optimization

**Goal**: ≥80 tok/s single-request decode.

1. Benchmark Phase 1 output (expect ~20-40 tok/s from cross-compiled shaders)
2. Hand-optimized MSL kernels: dmmv_q4k (simdgroup_matrix), flash_attn (simdgroup_async_copy for M2+)
3. Fused ops tuning (32-thread simdgroups, workgroup size optimization)
4. Pre-compile all MSL into single .metallib at build time
5. Profile with Metal System Trace — target ≥75% bandwidth utilization

**Milestone**: ≥80 tok/s on M4 Max 64 GB with Qwen3.5-35B-A3B Q4_K_XL.

### Phase 3: Parallel Requests & Server

**Goal**: ≥200 tok/s aggregate at 5 concurrent requests.

1. Extend scheduler for concurrent request KV page allocation
2. Batched dispatch — single GPU submit for all sequences in batch
3. OpenAI-compatible API with SSE streaming on Metal
4. Max 16 concurrent, HTTP 503 on overflow
5. Benchmark parallel throughput

**Milestone**: 5 concurrent `curl` requests, each ≥40 tok/s, aggregate ≥200 tok/s.

### Phase 4: Polish

1. macOS CI (GitHub Actions)
2. Model compatibility testing (Llama-3, Mistral, DeepSeek)
3. All quant types validated: Q4_K, Q5_K, Q6_K, Q8_0, F16
4. Documentation updates (AGENTS.md, docs/SPEC.md)
