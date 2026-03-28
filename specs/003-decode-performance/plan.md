# Implementation Plan: Decode Performance Optimization

**Branch**: `003-decode-performance` | **Date**: 2026-03-28 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/003-decode-performance/spec.md`

## Summary

Optimize ZINC decode throughput from 4 tok/s to 107+ tok/s on RDNA4 by eliminating CPU-GPU synchronization overhead. The current forward pass makes ~151 vkQueueSubmit+vkWaitForFences calls per token because MoE routing, shared expert gating, and SSM computation all require CPU readback. The approach: (1) profile to confirm bottleneck, (2) move all CPU-side computation to GPU shaders, (3) batch command buffers across layers, (4) tune shader occupancy. Phased success criteria: ≥27 tok/s (milestone 1, bandwidth floor), ≥107 tok/s (target, matching llama.cpp).

## Technical Context

**Language/Version**: Zig 0.14-dev (host), GLSL 4.60 (shaders) compiled to SPIR-V via system glslc (shaderc 2023.8)
**Primary Dependencies**: Vulkan 1.3 (RADV driver, Mesa 25.0.7), VK_KHR_cooperative_matrix
**Storage**: GGUF model files, memory-mapped with DMA to GPU VRAM
**Testing**: `zig build test` (26 build tests), manual decode correctness comparison vs llama.cpp
**Target Platform**: Linux (Ubuntu), AMD Radeon AI PRO R9700 (RDNA4, 32 GB, 576 GB/s)
**Project Type**: CLI inference engine (server mode planned in Phase 4)
**Performance Goals**: ≥27 tok/s milestone 1 (memory-bandwidth floor), ≥107 tok/s target (llama.cpp parity)
**Constraints**: wave64 only, system glslc 2023.8 only (newer causes 5x regression on RADV), single-request decode
**Scale/Scope**: Single model (Qwen3.5-35B-A3B Q4_K_XL, 21 GB, 40 layers, 256 experts top-8)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|-----------|--------|-------|
| I. Performance-First | ✅ PASS | This feature IS the performance optimization. All changes profiling-driven. |
| II. RDNA4-Native | ✅ PASS | All new shaders target wave64, 64 CUs, 32KB L1. No cross-vendor compromises. |
| III. Zig Systems Correctness | ✅ PASS | All host code in Zig. GPU buffers explicitly managed. No hidden allocations (removing per-token alloc/free in SSM). |
| IV. Vulkan-First | ✅ PASS | All GPU work via GLSL compute → SPIR-V. System glslc 2023.8 only. |
| V. Production Serving | ⚠️ NOTE | Feature focuses on single-request decode. This is prerequisite for multi-request (Phase 4). No violation — single-request throughput is foundational. |
| VI. Correctness Validation | ✅ PASS | Token-exact match required vs pre-optimization output. Router and gate values validated within tolerance. |

**Command buffer strategy** (from Technical Constraints): "Pre-record static decode graph, replay via vkQueueSubmit" — this feature partially implements this by batching within/across layers. Full pre-recording requires GPU-side routing first.

No violations. Proceeding to Phase 0.

## Project Structure

### Documentation (this feature)

```text
specs/003-decode-performance/
├── plan.md              # This file
├── research.md          # Phase 0: bottleneck analysis, shader design decisions
├── data-model.md        # Phase 1: GPU buffer entities and state management
├── quickstart.md        # Phase 1: how to validate the optimization
├── checklists/
│   └── requirements.md  # Spec quality checklist
└── tasks.md             # Phase 2 output (via /speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── compute/
│   └── forward.zig          # PRIMARY: decode loop, submit batching, GPU SSM integration
├── shaders/
│   ├── softmax_topk.comp    # NEW: GPU MoE router (replaces CPU topKSoftmax)
│   ├── ssm_conv1d.comp      # NEW: GPU conv1d + SiLU (replaces CPU conv1d)
│   ├── ssm_delta_net.comp   # NEW: GPU delta-net state update (replaces CPU delta-net)
│   ├── ssm_gated_norm.comp  # NEW: GPU RMS norm × SiLU gate (replaces CPU gated norm)
│   ├── sigmoid_scale_acc.comp  # EXISTING: used for shared expert gate on GPU
│   ├── dmmv_q4k.comp        # EXISTING: may need occupancy tuning
│   └── ... (16 existing shaders unchanged)
├── vulkan/
│   ├── command.zig           # EXISTING: command buffer recording/submit (may need timestamp query support)
│   └── instance.zig          # EXISTING: device init (may need query pool creation)
├── model/
│   └── loader.zig            # EXISTING: config parsing (unchanged)
└── main.zig                  # EXISTING: CLI args (add --profile flag, update --max-tokens)
```

**Structure Decision**: All changes are within the existing `src/` structure. 4 new shader files, modifications to `forward.zig` (primary), minor changes to `command.zig`, `instance.zig`, and `main.zig`. No new directories.

## Complexity Tracking

No constitution violations. Table not needed.
