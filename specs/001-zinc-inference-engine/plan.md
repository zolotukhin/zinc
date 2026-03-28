# Implementation Plan: ZINC LLM Inference Engine

**Branch**: `001-zinc-inference-engine` | **Date**: 2026-03-25 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `specs/001-zinc-inference-engine/spec.md`

## Summary

Build a full-stack LLM inference engine in Zig + Vulkan compute shaders, targeting AMD RDNA3/RDNA4 consumer GPUs. The engine loads GGUF models, runs inference through hand-tuned GPU kernels achieving 90%+ bandwidth utilization, serves an OpenAI-compatible HTTP API with continuous batching and paged KV cache, and optionally compresses KV cache using TurboQuant for 5x memory reduction.

## Technical Context

**Language/Version**: Zig 0.15.2+ (host code) + GLSL (compute shaders compiled to SPIR-V)
**Primary Dependencies**: Vulkan 1.3 API (direct C ABI calls), system glslc (shaderc 2023.8)
**Storage**: Memory-mapped GGUF files, GPU VRAM buffers (no database)
**Testing**: `zig build test` (unit tests) + `bun test` (TypeScript integration/API tests)
**Target Platform**: Linux (primary), macOS (build only — no GPU inference)
**Project Type**: Inference server (daemon with HTTP API) + CLI tool
**Performance Goals**: 110+ tok/s single-request, 432+ tok/s aggregate (4 concurrent), 2800+ tok/s prefill, 90%+ DMMV bandwidth utilization
**Current State (2026-03-27)**: Full 40-layer forward pass running at 4 tok/s. Output not yet coherent (ASCII numbers instead of English). 8 critical bugs found and fixed by optimization loop. Tokenizer verified correct, embedding+norm verified bit-identical to CPU reference.
**Constraints**: GPU memory-bound (576 GB/s RDNA4), wave64 optimal but RADV may use wave32 (shaders must handle both), system glslc only (newer versions cause 5x regression), RADV_PERFTEST=coop_matrix required, Mesa 25.0.7 pinned (25.2.8 has 14% regression)
**Scale/Scope**: 4-8 concurrent requests, 8K-32K context, models up to 70B Q4_K on 32GB VRAM
**Target Model**: Qwen3.5-35B-A3B Q4_K_XL (hybrid attention+SSM+MoE, 40 layers, 256 experts top-8, delta-net recurrent blocks)

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- [x] **Performance-First (I)**: All kernel designs include bandwidth utilization targets; profiling data required for validation
- [x] **RDNA4-Native (II)**: Shaders target wave64, 64 CUs, 32KB L1/CU, 6MB L2; no generic GPU fallbacks
- [x] **Zig Systems Correctness (III)**: All host code in Zig; explicit GPU memory management; comptime for dispatch tables
- [x] **Vulkan-First (IV)**: Vulkan compute only; GLSL→SPIR-V via system glslc; cooperative matrix leveraged
- [x] **Production Serving (V)**: Continuous batching + paged KV cache + OpenAI API from Phase 2
- [x] **Correctness Validation (VI)**: All outputs validated against llama.cpp/PyTorch reference; >99.5% cosine similarity

## Project Structure

### Documentation (this feature)

```text
specs/001-zinc-inference-engine/
├── plan.md              # This file
├── research.md          # Technical decisions and research
├── data-model.md        # Core data structures
├── quickstart.md        # Validation scenarios
├── contracts/           # API contracts
│   └── openai-api.md    # OpenAI-compatible endpoint spec
└── tasks.md             # Task breakdown (from /speckit.tasks)
```

### Source Code (repository root)

```text
src/
├── main.zig                 # Entry point, CLI parsing, server startup
├── vulkan/
│   ├── instance.zig         # Vulkan instance, device, queue setup
│   ├── buffer.zig           # GPU buffer allocation and management
│   ├── pipeline.zig         # Compute pipeline creation and management
│   ├── command.zig          # Command buffer recording and replay
│   └── gpu_detect.zig       # GPU capability detection and auto-tuning
├── model/
│   ├── gguf.zig             # GGUF file parser and memory mapper
│   ├── loader.zig           # Model loading orchestration (GGUF → GPU buffers)
│   ├── architecture.zig     # Architecture-specific graph builders (LLaMA, MoE, Mamba)
│   └── tokenizer.zig        # Tokenizer interface
├── compute/
│   ├── graph.zig            # Compute graph definition and execution
│   ├── dmmv.zig             # Decode matmul-vec dispatch
│   ├── elementwise.zig      # Fused element-wise kernel dispatch
│   ├── attention.zig        # Flash attention dispatch
│   └── moe.zig              # MoE expert routing dispatch
├── scheduler/
│   ├── scheduler.zig        # Continuous batching scheduler
│   ├── kv_cache.zig         # Paged KV cache management
│   └── request.zig          # Request state and lifecycle
├── turboquant/
│   ├── lloyd_max.zig        # Lloyd-Max codebook solver (CPU)
│   ├── rotation.zig         # Orthogonal matrix generation via QR (CPU)
│   ├── qjl.zig              # QJL projection matrix generation
│   ├── compress.zig         # GPU compression dispatch
│   └── config.zig           # TurboQuant configuration and CLI
├── server/
│   ├── http.zig             # HTTP server (Zig std.http)
│   ├── routes.zig           # OpenAI-compatible API route handlers
│   └── sse.zig              # Server-sent events streaming
└── shaders/
    ├── dmmv_q4k.comp        # Q4_K decode matmul-vec
    ├── dmmv_q8_0.comp       # Q8_0 decode matmul-vec
    ├── dmmv_f16.comp        # F16 decode matmul-vec
    ├── rms_norm_mul.comp    # Fused RMS norm + scale multiply
    ├── swiglu.comp          # Fused SiLU(x) * y
    ├── sigmoid_mul.comp     # Fused sigmoid(x) * y (SSM gating)
    ├── rope_fused.comp      # Fused RoPE + reshape + cache write
    ├── softmax_topk.comp    # Fused softmax + top-k (MoE routing)
    ├── flash_attn.comp      # Paged flash attention (GQA)
    ├── coop_matmul.comp     # Cooperative matrix matmul (prefill)
    ├── tq_quantize_keys.comp      # TurboQuant key compression
    ├── tq_quantize_values.comp    # TurboQuant value compression
    ├── tq_attention_scores.comp   # Asymmetric attention on compressed KV
    └── tq_decompress_values.comp  # Weighted value reconstruction

benchmarks/
├── bandwidth.zig            # Memory bandwidth microbenchmark
└── dispatch.zig             # Vulkan dispatch overhead microbenchmark

loops/
├── test_api.ts              # TypeScript API integration tests
└── test_streaming.ts        # SSE streaming tests

docs/
├── SPEC.md                  # Technical specification (existing)
├── TURBOQUANT_SPEC.md       # TurboQuant specification (existing)
├── RDNA4_TUNING.md          # RDNA4 tuning guide (existing)
└── API.md                   # API reference (existing)

research/
├── llama_cpp_analysis.md    # llama.cpp Vulkan backend analysis (existing)
└── turboquant-pytorch-master/  # PyTorch reference implementation (existing)
```

**Structure Decision**: Single project with Zig source tree. Shaders are compiled at build time via `build.zig`. The project is a single binary that acts as both CLI tool and HTTP server. No frontend; the interface is the HTTP API.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| Multiple shader files (~14) | Each quant format and fused op needs a dedicated optimized shader | A single generic shader cannot achieve 90%+ bandwidth utilization — specialization constants help but per-format shaders are needed for the inner loops |
| TurboQuant subsystem | Adds significant complexity (6 new files + 4 shaders) | Required to fit concurrent requests in VRAM on 16GB cards; simpler quantization (round-to-nearest) introduces biased inner products that degrade attention quality |
