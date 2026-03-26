# Data Model: ZINC Inference Engine

## Core Entities

### Model

Represents a loaded GGUF model ready for inference.

**Fields**:
- `header`: GGUF file header (magic, version, tensor count, metadata count)
- `tensors`: Array of tensor descriptors (name, type, dimensions, GPU buffer offset)
- `gpu_buffers`: Array of Vulkan buffers holding weight data in GPU VRAM
- `metadata`: String-keyed metadata from GGUF (architecture, context length, head count, etc.)
- `architecture`: Parsed architecture type (LLaMA, QwenMoE, Mamba, Jamba)
- `config`: Derived model config (n_layers, n_heads, head_dim, n_kv_heads, vocab_size, etc.)

**Relationships**: Owns many TensorInfo, references GpuConfig, optionally references TurboQuantConfig.

**State transitions**: Unloaded → Loading (memory-mapping + DMA) → Ready → Unloaded.

### GpuConfig

Auto-detected GPU capabilities and derived tuning parameters.

**Fields**:
- `vendor`: GPU vendor/architecture (amd_rdna3, amd_rdna4, intel_arc, nvidia)
- `vram_gb`, `bandwidth_gbps`, `compute_units`, `wave_size`: Hardware specs
- `coopmat_support`, `coopmat_tile`: Cooperative matrix capability
- `l1_cache_kb`, `l2_cache_mb`, `max_workgroup_size`: Cache hierarchy
- `dmmv_workgroup_size`, `dmmv_rows_per_workgroup`: Derived matmul tuning
- `matmul_tile_m`, `matmul_tile_n`: Cooperative matmul tile sizes
- `flash_attn_block_size`: Flash attention block size (256 for RDNA4)

**Relationships**: Referenced by Model, Scheduler, all kernel dispatchers.

### KVPage

A 16-token page of key-value cache data.

**Fields**:
- `format`: Page format (fp16, turboquant_2bit, turboquant_3bit, turboquant_4bit)
- `keys`: Key data — either FP16 buffer or compressed (MSE indices + QJL signs + residual norms + vec norms)
- `values`: Value data — either FP16 buffer or compressed (MSE indices + vec norms)
- `ref_count`: Reference count for copy-on-write
- `last_access`: Timestamp for LRU eviction

**Relationships**: Owned by PageTable, referenced by Request.

### PageTable

Maps sequence positions to KV cache pages.

**Fields**:
- `entries`: Map of (sequence_id, page_index) → KVPage pointer
- `free_pages`: Pool of unallocated pages
- `total_pages`, `used_pages`: Capacity tracking

**Relationships**: Owns KVPage pool, referenced by Scheduler.

### Request

An active inference request with its state.

**Fields**:
- `id`: Unique request identifier
- `state`: Request state (pending, prefilling, decoding, completed, cancelled)
- `prompt_tokens`: Tokenized input sequence
- `generated_tokens`: Tokens generated so far
- `kv_page_map`: This request's KV cache page mappings
- `params`: Generation parameters (temperature, top_p, top_k, max_tokens, stop sequences)
- `time_in_queue`: For priority scheduling
- `sse_connection`: SSE stream handle for token-by-token output (if streaming)

**Relationships**: References KVPages via PageTable, owned by Scheduler.

**State transitions**: Pending → Prefilling → Decoding → Completed/Cancelled.

### TurboQuantConfig

Per-model TurboQuant compression configuration.

**Fields**:
- `enabled`: Whether TQ compression is active
- `key_bits`, `value_bits`: Bit widths (2, 3, or 4)
- `min_seq_compress`: Minimum sequence length before compression kicks in
- `head_dim`: Model head dimension (typically 128)
- `n_levels`: 2^(bits-1) for keys, 2^bits for values

### TurboQuantMatrices

Per-head precomputed constants for TurboQuant.

**Fields**:
- `Pi`: Random orthogonal rotation matrix (d × d) f32
- `Pi_T`: Transpose of Pi (d × d) f32
- `S`: QJL projection matrix (m × d) f32 (keys only)
- `centroids`: Lloyd-Max codebook (n_levels values) f32

**Relationships**: Generated once at model load time per unique (layer, head) pair. Referenced by TQ compression and attention shaders.

## Compressed Page Layout (d=128, 3-bit)

| Component | Size per token | Format |
|-----------|---------------|--------|
| Key MSE indices | 48 B (12 × u32, padded) | 2-bit packed, 16B aligned |
| Key QJL signs | 16 B (4 × u32) | 1-bit packed |
| Key residual norm | 2 B | f16 |
| Key vec norm | 2 B | f16 |
| Value MSE indices | 48 B (12 × u32, padded) | 3-bit packed, 16B aligned |
| Value vec norm | 2 B | f16 |
| **Total K+V** | **102 B** | vs 512 B FP16 = **5x compression** |
