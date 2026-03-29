#ifndef ZINC_METAL_SHIM_H
#define ZINC_METAL_SHIM_H

#include <stdint.h>
#include <stddef.h>

// Opaque types
typedef struct MetalCtx MetalCtx;
typedef struct MetalBuf MetalBuf;
typedef struct MetalPipe MetalPipe;
typedef struct MetalCmd MetalCmd;

// Device lifecycle
MetalCtx* mtl_init(void);
void mtl_destroy(MetalCtx* ctx);
uint64_t mtl_max_buffer_size(MetalCtx* ctx);
uint64_t mtl_total_memory(MetalCtx* ctx);
uint32_t mtl_chip_family(MetalCtx* ctx);

// Buffer management
MetalBuf* mtl_create_buffer(MetalCtx* ctx, size_t size, void** cpu_ptr);
MetalBuf* mtl_wrap_mmap(MetalCtx* ctx, void* ptr, size_t size);
void* mtl_buffer_contents(MetalBuf* buf);
void mtl_free_buffer(MetalBuf* buf);

// Pipeline management
MetalPipe* mtl_create_pipeline(MetalCtx* ctx, const char* msl_source, const char* fn_name);
MetalPipe* mtl_create_pipeline_from_lib(MetalCtx* ctx, const void* lib_data, size_t lib_size, const char* fn_name);
uint32_t mtl_pipeline_max_threads(MetalPipe* pipe);
void mtl_free_pipeline(MetalPipe* pipe);

// Command buffer & dispatch
MetalCmd* mtl_begin_command(MetalCtx* ctx);
void mtl_dispatch(MetalCmd* cmd, MetalPipe* pipe,
                  const uint32_t grid[3], const uint32_t block[3],
                  MetalBuf** bufs, uint32_t n_bufs,
                  const void* push_data, size_t push_size);
void mtl_dispatch_v2(MetalCmd* cmd, MetalPipe* pipe,
                     const uint32_t grid[3], const uint32_t block[3],
                     MetalBuf** bufs, uint32_t n_bufs,
                     const void* push_data, size_t push_size,
                     uint32_t push_idx);
void mtl_barrier(MetalCmd* cmd);
void mtl_commit_and_wait(MetalCmd* cmd);
void mtl_commit_async(MetalCmd* cmd);
void mtl_wait(MetalCmd* cmd);

#endif // ZINC_METAL_SHIM_H
