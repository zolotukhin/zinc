# ZINC Technical Specification

## 1. GPU Kernel Library

### 1.1 Matrix Multiplication (DMMV — Decode MatMul-Vec)
The critical path. Single-token decode is memory-bandwidth bound.

```
Target: 90%+ memory bandwidth utilization on RDNA4 (>500 GB/s effective)
Measured on llama.cpp: 67-93% depending on matrix size
```

**Approach:**
- Hand-written GLSL compute shaders compiled to SPIR-V with the system glslc
- Specialization constants for matrix dimensions (M, K, quant type)
- RDNA4-specific tuning: wave64, 64 CUs, 32KB L0 vector cache/CU, 8MB L2
- Cooperative matrix 16x16x16 for prefill/batched decode
- Per-quant-type optimized dequantize+dot-product kernels

**Quantization support (phase 1):**
- Q4_K, Q5_K, Q6_K (K-quants — best quality/size)
- Q8_0 (attention weights)
- F16 (KV cache, small tensors)

### 1.2 Fused Element-wise Kernels
From profiling: ~3.6ms per token is non-matmul compute on small tensors.

| Fusion | Ops | Estimated Savings |
|--------|-----|-------------------|
| RMS_NORM_MUL | RMS_NORM + scale multiply | ~500us |
| SWIGLU | SILU(x) * y | Standard in modern models |
| SIGMOID_MUL | sigmoid(x) * y | SSM/Mamba gating |
| ROPE_FUSED | RoPE + reshape + cache write | ~200us |
| SOFTMAX_TOPK | softmax + top-k selection | MoE routing |

Each fusion is a single .comp shader with multiple input bindings,
eliminating intermediate memory traffic.

### 1.3 Flash Attention
Paged flash attention for efficient KV cache with concurrent requests.

- Block size: 256 (matching RDNA4 L1 cache)
- Paged KV: 16-token pages (like vLLM)
- Supports GQA (grouped-query attention)
- Cooperative matrix acceleration for QxK^T and attention*V

### 1.4 Command Buffer Strategy
Key insight from benchmarking: Vulkan dispatch overhead is only 0.016us.
The real cost is kernel execution on small tensors.

- **Pre-record** the decode command buffer once (graph is static per-token)
- **Replay** via `vkQueueSubmit` — only 54us for 1500 dispatches
- **Dynamic data** via push constants and descriptor set updates (no re-recording)
- **Double-buffer** command buffers for pipeline overlap

## 2. Model Loading

### 2.1 GGUF Format Support
Read GGUF files directly — the most common format for quantized models.

```zig
const Model = struct {
    header: GGUFHeader,
    tensors: []TensorInfo,
    gpu_buffers: []VulkanBuffer,
    metadata: std.StringHashMap(MetaValue),
};
```

- Memory-map the file, DMA weights directly to GPU VRAM
- Support split files (model-00001-of-00003.gguf)
- Parse architecture metadata to build compute graph

### 2.2 Supported Architectures
- Qwen3 / Qwen3.5 (standard transformer + MoE variants)
- Gemma 3 (GeGLU activation, Gemma-specific normalization)
- OpenAI GPT-OSS (MoE with OAI SwiGLU, MXFP4 experts, attention sinks, ISWA, YaRN RoPE)
- Mamba / Jamba (SSM hybrid — Vulkan only)

## 3. Request Scheduler

### 3.1 Continuous Batching

```
Scheduler loop:
  1. Collect pending requests (new + in-progress)
  2. Sort by priority (time-in-queue, request type)
  3. Form batch: select up to max_batch_size tokens
  4. Prefill new sequences (batched)
  5. Decode one token per active sequence (batched)
  6. Check stopping conditions, emit completed tokens
  7. Manage KV cache pages (allocate/free/defrag)
```

### 3.2 KV Cache Management
Paged attention with page table — like vLLM's PagedAttention.

- **Page size**: 16 tokens x head_dim x num_layers x 2 (K+V)
- **Page table**: maps (sequence_id, position) -> GPU page
- **Copy-on-write**: shared prompt prefixes reuse pages
- **Eviction**: LRU when memory pressure

## 4. API Server

### 4.1 OpenAI-Compatible Endpoints
```
POST /v1/chat/completions    — chat inference (streaming + non-streaming)
POST /v1/completions         — text completion
POST /v1/embeddings          — embedding extraction
GET  /v1/models              — list loaded models
GET  /health                 — server health + GPU stats
```

### 4.2 Zig HTTP Server
- Zero-copy request parsing
- SSE streaming for token-by-token output
- Connection pooling for concurrent clients

## 5. GPU Auto-Configuration

```zig
const GpuConfig = struct {
    vendor: enum { amd_rdna3, amd_rdna4, intel_arc, nvidia },
    vram_gb: u32,
    bandwidth_gbps: u32,
    compute_units: u32,
    wave_size: u32,
    coopmat_support: bool,
    coopmat_tile: [3]u32,
    l1_cache_kb: u32,
    l2_cache_mb: u32,
    max_workgroup_size: u32,

    // Derived tuning parameters
    dmmv_workgroup_size: u32,
    dmmv_rows_per_workgroup: u32,
    matmul_tile_m: u32,
    matmul_tile_n: u32,
    flash_attn_block_size: u32,
};
```

## 6. Open Questions

1. **Tokenizer**: Implement in Zig or shell out to sentencepiece/tiktoken?
2. **Chat templates**: Full Jinja2 or subset?
3. **Multi-GPU**: Tensor parallelism over PCIe or pipeline parallelism?
4. **Windows support**: Vulkan works on Windows too
5. **Intel Arc support**: Same Vulkan path, different tuning
