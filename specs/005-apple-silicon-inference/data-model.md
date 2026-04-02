# Data Model: Apple Silicon Inference

## Entities

### MetalDevice

Wraps `MTLDevice` via ObjC shim. One per process.

| Field | Type | Description |
|-------|------|-------------|
| ctx | `*MetalCtx` | Opaque C pointer to ObjC shim context |
| chip | `ChipFamily` | Detected chip: `.m1`, `.m2`, `.m3`, `.m4` |
| max_buffer_size | `usize` | MTLDevice.maxBufferLength (typically ~75% of RAM) |
| has_async_copy | `bool` | M2+ simdgroup_async_copy support |
| simdgroup_size | `u32` | Always 32 on Apple Silicon |
| memory_bytes | `u64` | Total unified memory (e.g., 64 GB) |

### MetalBuffer

Wraps `MTLBuffer`. Two creation modes: allocate (new shared buffer) or wrap (existing mmap pointer).

| Field | Type | Description |
|-------|------|-------------|
| handle | `*MetalBuf` | Opaque C pointer to MTLBuffer |
| size | `usize` | Buffer size in bytes |
| cpu_ptr | `?[*]u8` | Direct CPU pointer (SharedMode — always non-null) |
| is_mmap_wrapped | `bool` | True if wrapping external mmap, false if owned |

### MetalPipeline

Wraps `MTLComputePipelineState` created from MSL source or precompiled metallib.

| Field | Type | Description |
|-------|------|-------------|
| handle | `*MetalPipe` | Opaque C pointer to pipeline state |
| max_threads_per_threadgroup | `u32` | Hardware limit (typically 1024) |
| threadgroup_memory_size | `u32` | Shared memory used by this kernel |

### GpuBackend (tagged union)

Comptime-resolved abstraction over Vulkan and Metal.

```zig
pub const GpuBackend = union(enum) {
    vulkan: VulkanInstance,  // Linux
    metal: MetalDevice,      // macOS
};
```

Zig comptime ensures only one variant is ever compiled per target OS. Methods:
- `createBuffer(size, opts) → Buffer`
- `wrapMmap(ptr, size) → Buffer` (Metal: zero-copy, Vulkan: staging upload)
- `createPipeline(shader_source) → Pipeline`
- `beginCommand() → CommandBuffer`
- `dispatch(pipeline, grid, block, push_data, buffers)`
- `commitAndWait()`
- `destroy()`

### Buffer (unified handle)

```zig
pub const Buffer = union(enum) {
    vulkan: vulkan.Buffer,
    metal: MetalBuffer,

    pub fn ptr(self: @This()) ?[*]u8;  // Metal: always available, Vulkan: only for mapped
    pub fn size(self: @This()) usize;
};
```

### ChipFamily

Runtime-detected Apple Silicon generation. Used for feature gating.

```zig
pub const ChipFamily = enum {
    m1,    // Apple 7 GPU family, simdgroup_matrix
    m2,    // Apple 8, + simdgroup_async_copy
    m3,    // Apple 8+, + dynamic caching
    m4,    // Apple 9, + ray tracing (not used)
    unknown,

    pub fn hasAsyncCopy(self: @This()) bool {
        return self != .m1 and self != .unknown;
    }
};
```

## Relationships

```
GpuBackend (1) ──owns──> MetalDevice (1) ──creates──> MetalBuffer (N)
                                          ──creates──> MetalPipeline (N)

Model.load() ──uses──> GpuBackend.wrapMmap() ──creates──> MetalBuffer (per tensor)

InferenceEngine ──owns──> GpuBackend
                ──owns──> Buffer[] (intermediates: hidden, norm, q, k, v, etc.)
                ──owns──> Pipeline[] (dmmv, flash_attn, rms_norm, swiglu, etc.)
                ──owns──> KvCache[layers] ──contains──> Buffer (k_cache, v_cache per layer)
```

## State Transitions

### MetalDevice Lifecycle

```
Uninitialized → mtl_init() → Ready → [create buffers, pipelines, dispatch] → mtl_destroy() → Destroyed
```

### Request Lifecycle (unchanged from Vulkan, extended for Metal)

```
Pending → Scheduled (KV pages assigned) → Decoding (tokens generated) → Complete (all tokens emitted)
                                              ↓ (on error/timeout)
                                           Failed (pages freed, HTTP 500/504)
```

### Command Buffer (per decode step)

```
Begin → [bind pipeline, set buffers, dispatch]* → Commit → Wait → Complete
```

On Metal, command buffers are non-reusable (unlike Vulkan's replayable pre-recorded buffers). However, the dispatch sequence is identical each step — the "pre-recording" advantage comes from having the Zig-side dispatch loop pre-computed (no per-token graph walks).

## Scale Assumptions

| Dimension | Value | Source |
|-----------|-------|--------|
| Model weights | 20.7 GB (Q4_K_XL) | GGUF file |
| Available after model load | ~40 GB (64 GB - 21 GB - OS) | M4 Max 64 GB |
| KV cache per token per layer | 2 × kv_dim × 4 bytes = 2048 bytes | F32 KV, kv_dim=256 |
| KV cache per token (40 layers) | ~80 KB | 40 × 2048 |
| KV cache at 4096 context | ~320 MB | 4096 × 80 KB |
| KV cache at 32768 context | ~2.5 GB | 32768 × 80 KB |
| Max concurrent × max context | 16 × 32768 = ~40 GB | Would exhaust memory — dynamic allocation prevents this |
| Typical: 8 concurrent × 4096 | ~2.5 GB | Comfortable in 40 GB headroom |
