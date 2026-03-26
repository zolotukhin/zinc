# Research: ZINC Inference Engine

## Decision Log

### D1: Zig over Rust/C++

**Decision**: Use Zig 0.15.2+ for all host-side code.
**Rationale**: No hidden allocations (critical for GPU memory management), C ABI compatible (zero-overhead Vulkan calls), comptime for kernel dispatch tables, cross-compilation to single binary.
**Alternatives considered**: Rust (borrow checker overhead on GPU buffer management, FFI friction with Vulkan C API), C++ (undefined behavior, complex build systems, 15K+ lines problem seen in llama.cpp ggml-vulkan.cpp).

### D2: Vulkan Compute over ROCm/HIP

**Decision**: Vulkan compute shaders are the only GPU backend.
**Rationale**: ROCm does not support RDNA3/RDNA4 consumer GPUs — only MI-series datacenter GPUs. Vulkan works on all AMD GPUs via RADV driver. Measured dispatch overhead of 0.016µs/dispatch makes Vulkan viable for high-dispatch-count inference.
**Alternatives considered**: ROCm/HIP (doesn't work on target hardware), OpenCL (no cooperative matrix extension, worse driver support), DirectX (Windows only).

### D3: System glslc (shaderc 2023.8)

**Decision**: Compile GLSL→SPIR-V using the system-provided glslc from Ubuntu 24.04 packages only.
**Rationale**: Newer glslc versions (v2026.2-dev) add NonWritable/NonReadable decorations and different control flow that RADV's ACO compiler handles poorly — measured 5x performance regression (110 tok/s → 19-25 tok/s).
**Alternatives considered**: Custom glslc build (causes regression), HLSL→SPIR-V via DXC (adds dependency, no benefit), hand-written SPIR-V (unmaintainable).

### D4: GGUF as the only model format

**Decision**: Support GGUF format exclusively.
**Rationale**: GGUF is the dominant format for quantized consumer models. It supports memory-mapping for DMA to GPU, includes architecture metadata for graph construction, and supports split files for large models.
**Alternatives considered**: SafeTensors (no quantization metadata, requires separate config), ONNX (heavy runtime, not designed for quantized inference), custom format (ecosystem fragmentation).

### D5: Paged KV Cache (vLLM-style)

**Decision**: 16-token pages with page table mapping (sequence_id, position) → GPU page.
**Rationale**: Enables efficient concurrent request serving without pre-allocating worst-case KV memory per request. Copy-on-write for shared prefixes. LRU eviction under memory pressure.
**Alternatives considered**: Contiguous KV cache per request (wastes VRAM, limits concurrency — llama.cpp's current approach), ring buffer (doesn't support variable-length requests well).

### D6: TurboQuant for KV Cache Compression

**Decision**: Implement TurboQuant (ICLR 2026) with 2/3/4-bit support.
**Rationale**: Mathematically unbiased inner product estimator — critical for attention quality. 5x compression at 3-bit with >99.5% attention cosine similarity. Validated on real Qwen2.5-3B data in reference implementation.
**Alternatives considered**: Round-to-nearest quantization (biased inner products degrade attention), KIVI (similar approach but less optimal codebooks), no compression (limits concurrent requests on 16GB cards).

### D7: Tokenizer Strategy

**Decision**: Shell out to external tokenizer (sentencepiece/tiktoken) initially, implement native Zig tokenizer in Phase 4.
**Rationale**: Tokenizer implementation is complex (BPE, vocabulary handling) and orthogonal to the core GPU inference work. Using an external process unblocks inference development. Native implementation deferred to production phase.
**Alternatives considered**: Native Zig from day 1 (delays core GPU work), C library binding (adds build complexity).

### D8: Chat Template Strategy

**Decision**: Support a fixed set of common templates (ChatML, Llama, Mistral) with string-based substitution. Full Jinja2 deferred.
**Rationale**: Full Jinja2 requires a template engine (complex to implement in Zig). The most common models use simple role-based templates. Covering ChatML + Llama + Mistral handles >90% of use cases.
**Alternatives considered**: Full Jinja2 in Zig (massive effort), shell out to Python Jinja2 (adds Python dependency).

## RDNA4-Specific Findings

Source: `docs/RDNA4_TUNING.md` and `research/llama_cpp_analysis.md`

- **Wave64 is optimal**: wave32 measured slower on RDNA4 for DMMV
- **GPU ECC**: Enabled by default, costs ~10% bandwidth. Disable with `amdgpu.ras_enable=0`
- **Cooperative matrix**: Requires `RADV_PERFTEST=coop_matrix` environment variable
- **Dispatch overhead**: 0.016µs per dispatch — negligible; 1500 dispatches in 24µs GPU time
- **Command buffer replay**: Pre-recorded replay of 1500 dispatches takes 54µs
- **SMU firmware mismatch**: Kernel 6.17 limits RDNA4 to 2200 MHz (should be 2350 MHz)
- **Clock forcing hurts**: `profile_peak` causes -23% regression due to power throttling on memory-bound work
- **DMMV_WG_SIZE_LARGE (256 threads)**: No improvement — too many idle threads for small K
- **rm_kq > 2**: -75% regression — wave64 can't handle 4+ rows per workgroup

## llama.cpp Analysis Summary

Source: `research/llama_cpp_analysis.md`

- RDNA4 (gfx1201) is misclassified as AMD_RDNA3 — no RDNA4-specific enum
- No entry in `gpu_pipeline_configs` for RDNA3, so subgroup size falls through to driver default
- MMVQ (integer dot product) path blocked: requires newer glslc which causes 5x regression
- Already fuses: MULTI_ADD, RMS_NORM_MUL, TOPK_MOE, MUL_MAT_ID_MUL, MUL_MAT_ADD, GLU
- Remaining fusion opportunities: sigmoid_mul (58x), silu_mul (51x)
- 15K+ lines of C++ in ggml-vulkan.cpp — hard to modify; struct layout changes can cause 20% regression
