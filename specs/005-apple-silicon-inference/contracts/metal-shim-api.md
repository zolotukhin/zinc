# Contract: Metal Shim C API

The ObjC shim (`src/metal/shim.m`) exposes this C API via `src/metal/shim.h`. This is the **only** interface between Zig and Metal — all Metal/ObjC calls are encapsulated here.

## Types

```c
typedef struct MetalCtx MetalCtx;     // Opaque: MTLDevice + MTLCommandQueue
typedef struct MetalBuf MetalBuf;     // Opaque: MTLBuffer
typedef struct MetalPipe MetalPipe;   // Opaque: MTLComputePipelineState
typedef struct MetalCmd MetalCmd;     // Opaque: MTLCommandBuffer + MTLComputeCommandEncoder
```

## Functions

### Device Lifecycle

```c
// Initialize Metal device and command queue. Returns NULL on failure (no Apple Silicon, no Metal).
MetalCtx* mtl_init(void);

// Destroy device and release all Metal objects.
void mtl_destroy(MetalCtx* ctx);

// Query device properties.
uint64_t mtl_max_buffer_size(MetalCtx* ctx);
uint64_t mtl_total_memory(MetalCtx* ctx);
uint32_t mtl_chip_family(MetalCtx* ctx);  // Returns ChipFamily enum value
```

### Buffer Management

```c
// Allocate a new shared-mode buffer. cpu_ptr receives direct CPU pointer.
MetalBuf* mtl_create_buffer(MetalCtx* ctx, size_t size, void** cpu_ptr);

// Wrap an existing mmap'd pointer as a Metal buffer (zero-copy).
// ptr must be page-aligned. size must be page-aligned.
// The caller retains ownership of the underlying memory.
MetalBuf* mtl_wrap_mmap(MetalCtx* ctx, void* ptr, size_t size);

// Get CPU pointer to buffer contents.
void* mtl_buffer_contents(MetalBuf* buf);

// Free a buffer. For mmap-wrapped buffers, only releases the Metal wrapper.
void mtl_free_buffer(MetalBuf* buf);
```

### Pipeline (Shader) Management

```c
// Create compute pipeline from MSL source string.
// fn_name is the kernel function name within the source.
// Returns NULL on compilation failure (logs error).
MetalPipe* mtl_create_pipeline(MetalCtx* ctx, const char* msl_source, const char* fn_name);

// Create compute pipeline from precompiled metallib data.
MetalPipe* mtl_create_pipeline_from_lib(MetalCtx* ctx, const void* lib_data, size_t lib_size, const char* fn_name);

// Query pipeline properties.
uint32_t mtl_pipeline_max_threads(MetalPipe* pipe);

// Free a pipeline.
void mtl_free_pipeline(MetalPipe* pipe);
```

### Command Buffer & Dispatch

```c
// Begin a new command buffer. Only one active at a time per context.
MetalCmd* mtl_begin_command(MetalCtx* ctx);

// Dispatch a compute kernel.
// grid[3]: threadgroups per dimension
// block[3]: threads per threadgroup per dimension
// bufs/n_bufs: Metal buffers bound at indices 0..n_bufs-1
// push_data/push_size: bytes copied to buffer index n_bufs (push constants equivalent)
void mtl_dispatch(MetalCmd* cmd, MetalPipe* pipe,
                  const uint32_t grid[3], const uint32_t block[3],
                  MetalBuf** bufs, uint32_t n_bufs,
                  const void* push_data, size_t push_size);

// Insert a compute barrier (equivalent to vkCmdPipelineBarrier).
void mtl_barrier(MetalCmd* cmd);

// Commit command buffer and wait for GPU completion (synchronous).
void mtl_commit_and_wait(MetalCmd* cmd);

// Commit command buffer without waiting (asynchronous).
// Use mtl_wait() to synchronize later.
void mtl_commit_async(MetalCmd* cmd);
void mtl_wait(MetalCmd* cmd);
```

## Invariants

1. `mtl_init()` must be called before any other function. Returns NULL if Metal is unavailable.
2. `mtl_wrap_mmap()` pointer and size must be page-aligned (4096 bytes on Apple Silicon).
3. Only one `MetalCmd` can be active per `MetalCtx` at a time. `mtl_commit_and_wait()` or `mtl_commit_async()` must be called before `mtl_begin_command()` again.
4. `push_data` size must not exceed 4096 bytes (Metal argument buffer limit).
5. Buffer indices in `mtl_dispatch()` are bound sequentially starting at index 0.
6. All functions are thread-unsafe — callers must synchronize externally.

## Error Handling

- `mtl_init()` returns NULL → caller prints error and exits
- `mtl_create_pipeline()` returns NULL → MSL compilation failed, error logged to stderr
- `mtl_create_buffer()` returns NULL → allocation failed (OOM)
- `mtl_wrap_mmap()` returns NULL → pointer not page-aligned or Metal rejected the mapping
- GPU command timeout → `mtl_commit_and_wait()` returns after MTLCommandBuffer.status == .error
