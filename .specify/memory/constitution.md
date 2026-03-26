<!--
Sync Impact Report:
- Version change: 0.0.0 → 1.0.0
- Added principles: Performance-First, RDNA4-Native, Zig Systems Correctness, Vulkan-First, Production Serving, Correctness Validation
- Added sections: Technical Constraints, Development Workflow
- Templates requiring updates: ✅ spec-template.md (no changes needed), ✅ plan-template.md (no changes needed), ✅ tasks-template.md (no changes needed)
- Follow-up TODOs: none
-->

# ZINC Constitution

## Core Principles

### I. Performance-First

Every design decision MUST optimize for memory bandwidth utilization on AMD consumer GPUs. The target is 90%+ bandwidth utilization on large matmuls and measurable improvement on all kernel fusions. Performance claims MUST be backed by profiling data, not assumptions. If a design choice trades performance for convenience, it MUST be justified with measurements showing negligible impact.

### II. RDNA4-Native

ZINC targets AMD RDNA3/RDNA4 consumer GPUs specifically — not generic GPU support. Shaders MUST be hand-tuned for wave64, 64 CUs, 32KB L1/CU, and 6MB L2. Generic "works on any GPU" approaches are rejected in favor of architecture-specific optimization. Intel Arc and NVIDIA support may follow, but RDNA4 is the first-class target and MUST NOT be compromised for cross-vendor compatibility.

### III. Zig Systems Correctness

All host-side code MUST be written in Zig. No hidden allocations, no undefined behavior, no runtime overhead from safety checks. GPU memory management MUST be explicit — every allocation tracked, every buffer lifetime clear. Comptime MUST be used for kernel dispatch tables, format-specific code paths, and configuration that can be resolved at build time. C ABI compatibility MUST be maintained for direct Vulkan API calls with zero bindings overhead.

### IV. Vulkan-First

Vulkan compute is the only GPU backend. ROCm is explicitly excluded because it does not support RDNA3/RDNA4 consumer GPUs. All GPU work MUST use GLSL compute shaders compiled to SPIR-V via the system glslc (shaderc 2023.8 from Ubuntu packages). Newer glslc versions that produce RADV-incompatible SPIR-V MUST NOT be used. The cooperative matrix extension (VK_KHR_cooperative_matrix) MUST be leveraged for matmul acceleration.

### V. Production Serving

ZINC is an inference server, not a CLI tool. Every feature MUST support concurrent requests via continuous batching and paged KV cache. The OpenAI-compatible API is the primary interface. Single-request performance matters, but multi-request aggregate throughput is the primary metric. KV cache design MUST support page-level management (allocate/free/defrag) for concurrent request serving.

### VI. Correctness Validation

All numerical results MUST be validated against a reference implementation (llama.cpp or PyTorch). Attention accuracy MUST maintain >99.5% cosine similarity against FP16 baseline. Quantization methods MUST use mathematically unbiased estimators where inner products are involved (e.g., TurboQuant QJL correction for KV cache). Codebook and rotation matrix generation MUST be reproducible with seeded PRNGs and validated against reference implementations with the same seeds.

## Technical Constraints

- **SPIR-V toolchain**: System glslc (shaderc 2023.8) only — newer versions cause 5x regression on RADV
- **Wave size**: wave64 is optimal; wave32 measured slower on RDNA4
- **Quantization formats**: Q4_K, Q5_K, Q6_K, Q8_0, F16 (phase 1); TurboQuant KV compression (phase 2+)
- **Model formats**: GGUF only — memory-mapped, DMA to GPU VRAM
- **Supported architectures**: LLaMA/Mistral/Qwen (transformer), Qwen MoE, Mamba/Jamba (SSM hybrid)
- **Command buffer strategy**: Pre-record static decode graph, replay via vkQueueSubmit

## Development Workflow

- **Profiling-driven**: Changes to GPU kernels MUST include before/after profiling data
- **Reference validation**: New shader implementations MUST pass numerical comparison against llama.cpp or PyTorch reference
- **Incremental phases**: Development follows Phase 0 (scaffold) → Phase 1 (single request) → Phase 2 (server + batching) → Phase 3 (performance) → Phase 4 (production)
- **Build system**: `zig build` for all targets; `zig build test` for Zig unit tests + `bun test` for TypeScript integration tests

## Governance

This constitution governs all ZINC development decisions. Amendments require:
1. Documented rationale with profiling data or technical justification
2. Review by project maintainers
3. Backward compatibility assessment for API and model format changes

Principle violations MUST be documented in the Complexity Tracking section of implementation plans with specific justification for why the simpler approach is insufficient.

**Version**: 1.0.0 | **Ratified**: 2026-03-25 | **Last Amended**: 2026-03-25
