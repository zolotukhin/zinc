# Feature Specification: ZINC LLM Inference Engine

**Feature Branch**: `001-zinc-inference-engine`
**Created**: 2026-03-25
**Status**: Draft
**Input**: Full-stack LLM inference engine for AMD RDNA3/RDNA4 consumer GPUs using Zig + Vulkan compute, with OpenAI-compatible API, continuous batching, paged KV cache, and TurboQuant KV cache compression.

## User Scenarios & Testing

### User Story 1 - Single-Request Inference (Priority: P1)

A developer loads a GGUF model onto an AMD RDNA4 GPU and generates text from a prompt. The engine reads the GGUF file, uploads weights to GPU VRAM, runs the transformer forward pass using hand-tuned Vulkan compute shaders, and streams generated tokens back.

**Why this priority**: This is the foundational capability — nothing else works without a correct, performant single-request inference path. Validates the entire GPU kernel library, model loading, and compute graph.

**Independent Test**: Load Qwen3.5-35B-A3B Q4_K_XL GGUF on an RDNA4 GPU, generate 256 tokens from a fixed prompt, verify output is coherent English matching llama.cpp quality. Compare first 10 generated tokens against llama.cpp server output for the same prompt.

**Current State (2026-03-27)**: Forward pass runs all 40 layers (10 attention + 30 SSM + 40 MoE FFN) at 4 tok/s. Tokenizer matches llama.cpp exactly. Embedding + RMS norm verified bit-identical to CPU reference. Output is ASCII numbers/punctuation instead of coherent English — correctness debugging in progress. 8 critical bugs found and fixed by the self-improving optimization loop (wave32 subgroup reduction, Q4_K/Q5_K sub-block pairing, SPEC_K bounds, shared expert dimension, conv1d split order, buffer overflow, SSM conv ordering).

**Acceptance Scenarios**:

1. **Given** a valid GGUF model file and an RDNA4 GPU, **When** the user runs `zinc -m model.gguf --prompt "Hello"`, **Then** the engine generates coherent text at 110+ tok/s for Qwen3.5-35B-A3B Q4_K on AI PRO R9700.
2. **Given** a GGUF model with Q4_K quantization, **When** inference runs, **Then** DMMV bandwidth utilization is 67-93% of theoretical (576 GB/s on AI PRO R9700).
3. **Given** a model with RoPE, RMS norm, SwiGLU, and softmax ops, **When** the forward pass executes, **Then** fused kernels (RMS_NORM_MUL, SWIGLU, ROPE_FUSED) reduce non-matmul compute time by >30% vs unfused baseline.

---

### User Story 2 - Multi-Request Server with OpenAI API (Priority: P2)

A team deploys ZINC as an inference server serving multiple concurrent clients via the OpenAI-compatible HTTP API. Requests are continuously batched, KV cache is paged, and SSE streaming delivers tokens as they're generated.

**Why this priority**: Production serving is ZINC's differentiator over llama.cpp. Continuous batching and paged KV cache enable efficient multi-request serving that llama.cpp's Vulkan backend cannot do.

**Independent Test**: Start the server, send 4 concurrent `/v1/chat/completions` requests with `stream: true`, verify all 4 streams return coherent tokens at 100+ tok/s each (400+ aggregate) with no cross-request contamination.

**Acceptance Scenarios**:

1. **Given** a running ZINC server with a loaded model, **When** 4 concurrent chat completion requests arrive, **Then** each request receives 100+ tok/s with 400+ tok/s aggregate throughput.
2. **Given** paged KV cache with 16-token pages, **When** requests arrive and complete, **Then** pages are allocated and freed correctly with no memory leaks over 1000 request cycles.
3. **Given** a streaming request, **When** tokens are generated, **Then** SSE events are delivered within 1ms of token generation with correct `data: {"choices": [...]}` format.
4. **Given** the `/health` endpoint, **When** queried, **Then** it returns GPU stats (VRAM, temperature, clock), model info, and active request count.

---

### User Story 3 - TurboQuant KV Cache Compression (Priority: P3)

A user enables TurboQuant KV cache compression to serve more concurrent requests or longer contexts on limited VRAM. The engine compresses KV cache using random rotation + Lloyd-Max quantization + QJL residual correction, achieving 5x compression at 3-bit with <0.5% attention accuracy loss.

**Why this priority**: Extends the production serving capability. On RX 9070 XT (16GB), TurboQuant enables 8+ concurrent 8K sessions on an 8B model instead of 4. On AI PRO R9700, it enables 70B models with concurrent requests where FP16 KV wouldn't fit.

**Independent Test**: Load Qwen3-8B Q4_K, enable `--kv-quant 3`, generate text, verify attention cosine similarity >99.5% against FP16 KV baseline and actual VRAM savings match theoretical 5x compression ratio.

**Acceptance Scenarios**:

1. **Given** `--kv-quant 3` is enabled, **When** keys are compressed, **Then** Stage 1 (Lloyd-Max) + Stage 2 (QJL) produce compressed pages that are 5x smaller than FP16 pages (102 bytes vs 512 bytes per K+V token at d=128).
2. **Given** compressed KV cache, **When** attention scores are computed asymmetrically, **Then** cosine similarity with FP16 baseline is >99.5% and inner product bias is <0.001 over 10K random vectors.
3. **Given** an RX 9070 XT with 16GB VRAM and Qwen3-8B Q4_K loaded, **When** 8 concurrent 8K-context requests run with TQ-3bit, **Then** all requests are served without OOM (vs 4 max with FP16 KV).

---

### User Story 4 - MoE and SSM/Mamba Architecture Support (Priority: P4)

A user loads a Qwen MoE or Mamba/Jamba hybrid model. The engine correctly handles expert routing (softmax + top-k selection), sparse expert computation, and SSM state-space operations alongside standard attention.

**Why this priority**: Extends model coverage to the architectures most relevant for efficient inference (MoE for quality/compute tradeoff, SSM for linear-time sequence processing).

**Independent Test**: Load Qwen3.5-35B-A3B (SSM+attention hybrid MoE), generate text, validate output against llama.cpp reference. Expert routing must correctly select top-k experts per token.

**Acceptance Scenarios**:

1. **Given** a Qwen MoE GGUF model, **When** the forward pass runs, **Then** the SOFTMAX_TOPK fused kernel correctly routes tokens to the top-k experts and the MUL_MAT_ID_MUL fusion computes expert outputs correctly.
2. **Given** a Mamba/Jamba hybrid model, **When** inference runs, **Then** SSM convolution, gated delta net, and sigmoid-multiply operations produce results matching the reference implementation.

---

### Edge Cases

- What happens when VRAM is insufficient for the model? Engine MUST report the shortfall in GB and refuse to load rather than crashing.
- What happens when a GGUF file is corrupt or uses an unsupported quant type? Engine MUST report the specific issue (bad magic, unsupported quant format) and exit cleanly.
- What happens when a client disconnects mid-stream? Server MUST free the KV cache pages and request slot within one scheduler tick.
- What happens when GPU ECC is enabled (consuming ~10% bandwidth)? Engine SHOULD detect and warn about suboptimal configuration.
- What happens with sequence lengths beyond the model's trained context? Engine MUST either clamp or report the limit.

## Requirements

### Functional Requirements

- **FR-001**: Engine MUST load GGUF model files, including split multi-part files (model-00001-of-00003.gguf), via memory-mapping with DMA to GPU VRAM.
- **FR-002**: Engine MUST support Q4_K, Q5_K, Q6_K, Q8_0, and F16 quantization formats for weight tensors.
- **FR-003**: Engine MUST implement hand-tuned GLSL compute shaders for DMMV (decode matmul-vec) achieving 90%+ bandwidth utilization on matrices with m >= 8192.
- **FR-004**: Engine MUST implement fused element-wise kernels: RMS_NORM_MUL, SWIGLU, SIGMOID_MUL, ROPE_FUSED, SOFTMAX_TOPK.
- **FR-005**: Engine MUST implement paged flash attention with 256-token blocks, 16-token pages, and GQA support.
- **FR-006**: Engine MUST pre-record the decode command buffer and replay via vkQueueSubmit (measured: 54us for 1500 dispatches).
- **FR-007**: Engine MUST serve an OpenAI-compatible HTTP API with POST /v1/chat/completions (streaming + non-streaming), POST /v1/completions, POST /v1/embeddings, GET /v1/models, GET /health.
- **FR-008**: Engine MUST implement continuous batching with a scheduler that collects, prioritizes, and batches pending requests.
- **FR-009**: Engine MUST implement paged KV cache with page table mapping (sequence_id, position) to GPU pages, supporting allocate/free/defrag and copy-on-write for shared prefixes.
- **FR-010**: Engine MUST support TurboQuant KV cache compression at 2, 3, and 4 bits via the `--kv-quant` CLI flag.
- **FR-011**: Engine MUST implement Lloyd-Max codebook solver and Householder QR rotation matrix generation on CPU at model load time.
- **FR-012**: Engine MUST implement asymmetric attention (query rotation optimization) for compressed KV cache, avoiding full decompression.
- **FR-013**: Engine MUST auto-detect GPU capabilities (vendor, VRAM, bandwidth, CU count, wave size, cooperative matrix support) and tune kernel parameters accordingly.
- **FR-014**: Engine MUST support LLaMA/Mistral/Qwen transformer, Qwen MoE, and Mamba/Jamba SSM hybrid architectures.

### Key Entities

- **Model**: GGUF header + tensor info + GPU buffers + architecture metadata. Represents a loaded model ready for inference.
- **KVPage**: A 16-token page of key-value cache data, either FP16 or TurboQuant-compressed. Managed by the page table.
- **Request**: An active inference request with its sequence state, KV cache page mappings, and generation parameters (temperature, max_tokens, etc.).
- **GpuConfig**: Auto-detected GPU capabilities and derived tuning parameters (workgroup sizes, tile dimensions, flash attention block size).
- **TurboQuantConfig**: Per-model compression parameters including rotation matrices (Pi), QJL projection matrix (S), Lloyd-Max centroids, and bit-width settings.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Single-request generation achieves 110+ tok/s on Qwen3.5-35B-A3B Q4_K on AI PRO R9700 (32GB, 576 GB/s).
- **SC-002**: 4 concurrent requests achieve 108+ tok/s each (432+ tok/s aggregate) with zero per-slot degradation.
- **SC-003**: Prefill throughput achieves 2800+ tok/s at pp512 context length.
- **SC-004**: DMMV bandwidth utilization is 93%+ for large matrices (m >= 248320) and 83%+ for medium matrices (m >= 8192).
- **SC-005**: TurboQuant 3-bit KV compression achieves 5x memory reduction with >99.5% attention cosine similarity vs FP16 baseline.
- **SC-006**: RX 9070 XT (16GB) serves 8+ concurrent 8K-context sessions on Qwen3-8B Q4_K with TQ-3bit enabled.
- **SC-007**: All GPU kernel outputs achieve >99.5% cosine similarity against llama.cpp or PyTorch reference implementation.
- **SC-008**: OpenAI API compatibility allows drop-in replacement for existing clients using /v1/chat/completions with streaming.

## Assumptions

- Target hardware is AMD RDNA3/RDNA4 consumer GPUs with Vulkan 1.3+ support and RADV or AMDVLK driver.
- System glslc is shaderc 2023.8 from Ubuntu 24.04 packages (newer versions produce RADV-incompatible SPIR-V).
- Models are distributed in GGUF format with standard architecture metadata.
- GPU ECC is disabled (`amdgpu.ras_enable=0`) for optimal bandwidth — engine warns if detected.
- RADV_PERFTEST=coop_matrix is set for cooperative matrix support.
- Linux is the primary platform; macOS builds for development only (no GPU inference).
- Tokenizer implementation strategy (native Zig vs external sentencepiece/tiktoken) is deferred to implementation planning.
- Chat template support scope (full Jinja2 vs subset) is deferred to implementation planning.
