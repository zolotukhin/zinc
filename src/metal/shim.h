#ifndef ZINC_METAL_SHIM_H
#define ZINC_METAL_SHIM_H

#include <stdint.h>
#include <stddef.h>

// Opaque types
typedef struct MetalCtx MetalCtx;
typedef struct MetalBuf MetalBuf;
typedef struct MetalPipe MetalPipe;
typedef struct MetalCmd MetalCmd;
typedef struct MetalRSet MetalRSet;

enum {
    ZINC_MTL_GPU_FAMILY_APPLE7 = 1007,
    ZINC_MTL_GPU_FAMILY_APPLE8 = 1008,
    ZINC_MTL_GPU_FAMILY_APPLE9 = 1009,
    ZINC_MTL_GPU_FAMILY_APPLE10 = 1010,
    ZINC_MTL_GPU_FAMILY_MAC2 = 2002,
};

// Device lifecycle
MetalCtx* mtl_init(void);
void mtl_destroy(MetalCtx* ctx);
uint64_t mtl_max_buffer_size(MetalCtx* ctx);
uint64_t mtl_total_memory(MetalCtx* ctx);
uint32_t mtl_chip_family(MetalCtx* ctx);
uint8_t mtl_supports_family(MetalCtx* ctx, uint32_t family);
uint8_t mtl_has_unified_memory(MetalCtx* ctx);
uint8_t mtl_supports_raytracing(MetalCtx* ctx);
uint64_t mtl_recommended_max_working_set_size(MetalCtx* ctx);
uint64_t mtl_max_threadgroup_memory_length(MetalCtx* ctx);

// Buffer management
MetalBuf* mtl_create_buffer(MetalCtx* ctx, size_t size, void** cpu_ptr);
MetalBuf* mtl_create_private_buffer(MetalCtx* ctx, size_t size);
MetalBuf* mtl_wrap_mmap(MetalCtx* ctx, void* ptr, size_t size);
void* mtl_buffer_contents(MetalBuf* buf);
void mtl_free_buffer(MetalBuf* buf);

// Pipeline management
MetalPipe* mtl_create_pipeline(MetalCtx* ctx, const char* msl_source, const char* fn_name);
MetalPipe* mtl_create_pipeline_quiet(MetalCtx* ctx, const char* msl_source, const char* fn_name);
MetalPipe* mtl_create_pipeline_from_lib(MetalCtx* ctx, const void* lib_data, size_t lib_size, const char* fn_name);
uint32_t mtl_pipeline_max_threads(MetalPipe* pipe);
uint32_t mtl_pipeline_thread_execution_width(MetalPipe* pipe);
uint32_t mtl_pipeline_static_threadgroup_memory_length(MetalPipe* pipe);
void mtl_free_pipeline(MetalPipe* pipe);

// Command buffer & dispatch
MetalCmd* mtl_begin_command(MetalCtx* ctx);
MetalCmd* mtl_begin_command_mode(MetalCtx* ctx, uint8_t serial);
void mtl_dispatch(MetalCmd* cmd, MetalPipe* pipe,
                  const uint32_t grid[3], const uint32_t block[3],
                  MetalBuf** bufs, uint32_t n_bufs,
                  const void* push_data, size_t push_size);
void mtl_dispatch_v2(MetalCmd* cmd, MetalPipe* pipe,
                     const uint32_t grid[3], const uint32_t block[3],
                     MetalBuf** bufs, uint32_t n_bufs,
                     const void* push_data, size_t push_size,
                     uint32_t push_idx);
void mtl_dispatch_v2_tgmem(MetalCmd* cmd, MetalPipe* pipe,
                     const uint32_t grid[3], const uint32_t block[3],
                     MetalBuf** bufs, uint32_t n_bufs,
                     const void* push_data, size_t push_size,
                     uint32_t push_idx, uint32_t tg_mem_size);
void mtl_barrier(MetalCmd* cmd);
void mtl_commit_and_wait(MetalCmd* cmd);
void mtl_commit_async(MetalCmd* cmd);
void mtl_wait(MetalCmd* cmd);

// Residency set management (macOS 15+). Wires GPU buffers down so they don't
// page-fault on cold access. Adapted from llama.cpp ggml-metal-device.m
// `ggml_metal_buffer_rset_init`. Returns NULL on systems where the API is
// unavailable; all subsequent calls become no-ops in that case.
MetalRSet* mtl_rset_create(MetalCtx* ctx, uint32_t initial_capacity);
void mtl_rset_add_buffer(MetalRSet* rset, MetalBuf* buf);
void mtl_rset_commit_and_request(MetalRSet* rset);
void mtl_rset_free(MetalRSet* rset);
uint8_t mtl_rset_supported(void);

#endif // ZINC_METAL_SHIM_H
