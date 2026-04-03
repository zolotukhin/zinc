// ZINC Metal shim — thin C API wrapping Apple Metal framework.
// This is the ONLY Objective-C file in the project. All Metal/ObjC calls are encapsulated here.
// See specs/005-apple-silicon-inference/contracts/metal-shim-api.md for the contract.
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include "shim.h"
#include <stdio.h>
#include <mach/mach.h>

#ifndef MTLGPUFamilyApple10
#define MTLGPUFamilyApple10 ((MTLGPUFamily)1010)
#endif

// --- Opaque struct definitions ---

struct MetalCtx {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
};

struct MetalBuf {
    id<MTLBuffer> buffer;
    size_t size;
    int is_mmap;
};

struct MetalPipe {
    id<MTLComputePipelineState> state;
};

struct MetalCmd {
    id<MTLCommandBuffer> cmd_buf;
    id<MTLComputeCommandEncoder> encoder;
};

// --- Device lifecycle ---

MetalCtx* mtl_init(void) {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        fprintf(stderr, "Error: Metal device not available. Apple Silicon required.\n");
        return NULL;
    }

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        fprintf(stderr, "Error: Failed to create Metal command queue.\n");
        return NULL;
    }

    MetalCtx* ctx = (MetalCtx*)calloc(1, sizeof(MetalCtx));
    if (!ctx) return NULL;

    ctx->device = device;
    ctx->queue = queue;
    return ctx;
}

void mtl_destroy(MetalCtx* ctx) {
    if (!ctx) return;
    ctx->queue = nil;
    ctx->device = nil;
    free(ctx);
}

uint64_t mtl_max_buffer_size(MetalCtx* ctx) {
    if (!ctx) return 0;
    return (uint64_t)[ctx->device maxBufferLength];
}

uint64_t mtl_total_memory(MetalCtx* ctx) {
    if (!ctx) return 0;
    // Use Mach API to get total physical memory (unified memory on Apple Silicon)
    mach_port_t host = mach_host_self();
    vm_size_t page_size;
    host_page_size(host, &page_size);

    mach_msg_type_number_t count = HOST_BASIC_INFO_COUNT;
    host_basic_info_data_t info;
    if (host_info(host, HOST_BASIC_INFO, (host_info_t)&info, &count) == KERN_SUCCESS) {
        return (uint64_t)info.max_mem;
    }
    // Fallback: recommendedMaxWorkingSetSize is GPU-specific but useful
    return (uint64_t)[ctx->device recommendedMaxWorkingSetSize];
}

uint32_t mtl_chip_family(MetalCtx* ctx) {
    if (!ctx) return 0; // unknown

    if ([ctx->device supportsFamily:MTLGPUFamilyApple10]) return 10;
    if ([ctx->device supportsFamily:MTLGPUFamilyApple9]) return 9;
    if ([ctx->device supportsFamily:MTLGPUFamilyApple8]) return 8;
    if ([ctx->device supportsFamily:MTLGPUFamilyApple7]) return 7;

    return 0; // unknown / pre-M1
}

uint8_t mtl_supports_family(MetalCtx* ctx, uint32_t family) {
    if (!ctx) return 0;
    return [ctx->device supportsFamily:(MTLGPUFamily)family] ? 1 : 0;
}

uint8_t mtl_has_unified_memory(MetalCtx* ctx) {
    if (!ctx) return 0;
    return [ctx->device hasUnifiedMemory] ? 1 : 0;
}

uint8_t mtl_supports_raytracing(MetalCtx* ctx) {
    if (!ctx) return 0;
    return [ctx->device supportsRaytracing] ? 1 : 0;
}

uint64_t mtl_recommended_max_working_set_size(MetalCtx* ctx) {
    if (!ctx) return 0;
    return (uint64_t)[ctx->device recommendedMaxWorkingSetSize];
}

uint64_t mtl_max_threadgroup_memory_length(MetalCtx* ctx) {
    if (!ctx) return 0;
    return (uint64_t)[ctx->device maxThreadgroupMemoryLength];
}

// --- Buffer management ---

MetalBuf* mtl_create_buffer(MetalCtx* ctx, size_t size, void** cpu_ptr) {
    if (!ctx || size == 0) return NULL;

    id<MTLBuffer> buffer = [ctx->device newBufferWithLength:size
                                                    options:MTLResourceStorageModeShared];
    if (!buffer) {
        fprintf(stderr, "Error: Failed to allocate Metal buffer of %zu bytes.\n", size);
        return NULL;
    }

    MetalBuf* buf = (MetalBuf*)calloc(1, sizeof(MetalBuf));
    if (!buf) return NULL;

    buf->buffer = buffer;
    buf->size = size;
    buf->is_mmap = 0;

    if (cpu_ptr) {
        *cpu_ptr = [buffer contents];
    }

    return buf;
}

MetalBuf* mtl_create_private_buffer(MetalCtx* ctx, size_t size) {
    if (!ctx || size == 0) return NULL;

    id<MTLBuffer> buffer = [ctx->device newBufferWithLength:size
                                                    options:MTLResourceStorageModePrivate];
    if (!buffer) {
        fprintf(stderr, "Error: Failed to allocate private Metal buffer of %zu bytes.\n", size);
        return NULL;
    }

    MetalBuf* buf = (MetalBuf*)calloc(1, sizeof(MetalBuf));
    if (!buf) return NULL;

    buf->buffer = buffer;
    buf->size = size;
    buf->is_mmap = 0;
    return buf;
}

MetalBuf* mtl_wrap_mmap(MetalCtx* ctx, void* ptr, size_t size) {
    if (!ctx || !ptr || size == 0) return NULL;

    // Verify page alignment (4096 on Apple Silicon)
    if ((uintptr_t)ptr % 4096 != 0) {
        fprintf(stderr, "Error: mmap pointer %p is not page-aligned (4096).\n", ptr);
        return NULL;
    }
    // Round size up to page boundary
    size_t aligned_size = (size + 4095) & ~(size_t)4095;

    id<MTLBuffer> buffer = [ctx->device newBufferWithBytesNoCopy:ptr
                                                          length:aligned_size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];
    if (!buffer) {
        fprintf(stderr, "Error: Failed to wrap mmap'd region (%p, %zu bytes) as Metal buffer.\n", ptr, size);
        return NULL;
    }

    MetalBuf* buf = (MetalBuf*)calloc(1, sizeof(MetalBuf));
    if (!buf) return NULL;

    buf->buffer = buffer;
    buf->size = size;
    buf->is_mmap = 1;
    return buf;
}

void* mtl_buffer_contents(MetalBuf* buf) {
    if (!buf) return NULL;
    return [buf->buffer contents];
}

void mtl_free_buffer(MetalBuf* buf) {
    if (!buf) return;
    buf->buffer = nil; // ARC releases the MTLBuffer
    free(buf);
}

// --- Pipeline management ---

MetalPipe* mtl_create_pipeline(MetalCtx* ctx, const char* msl_source, const char* fn_name) {
    if (!ctx || !msl_source || !fn_name) return NULL;

    NSError* error = nil;
    NSString* source = [NSString stringWithUTF8String:msl_source];
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = YES;

    id<MTLLibrary> library = [ctx->device newLibraryWithSource:source options:options error:&error];
    if (!library) {
        fprintf(stderr, "Error: MSL compilation failed for '%s': %s\n",
                fn_name, [[error localizedDescription] UTF8String]);
        return NULL;
    }

    NSString* name = [NSString stringWithUTF8String:fn_name];
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) {
        fprintf(stderr, "Error: Kernel function '%s' not found in compiled MSL.\n", fn_name);
        return NULL;
    }

    id<MTLComputePipelineState> state = [ctx->device newComputePipelineStateWithFunction:function
                                                                                   error:&error];
    if (!state) {
        fprintf(stderr, "Error: Failed to create compute pipeline for '%s': %s\n",
                fn_name, [[error localizedDescription] UTF8String]);
        return NULL;
    }

    MetalPipe* pipe = (MetalPipe*)calloc(1, sizeof(MetalPipe));
    if (!pipe) return NULL;
    pipe->state = state;
    return pipe;
}

MetalPipe* mtl_create_pipeline_from_lib(MetalCtx* ctx, const void* lib_data, size_t lib_size, const char* fn_name) {
    if (!ctx || !lib_data || lib_size == 0 || !fn_name) return NULL;

    NSError* error = nil;
    dispatch_data_t data = dispatch_data_create(lib_data, lib_size, NULL, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    id<MTLLibrary> library = [ctx->device newLibraryWithData:data error:&error];
    if (!library) {
        fprintf(stderr, "Error: Failed to load metallib for '%s': %s\n",
                fn_name, [[error localizedDescription] UTF8String]);
        return NULL;
    }

    NSString* name = [NSString stringWithUTF8String:fn_name];
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (!function) {
        fprintf(stderr, "Error: Kernel function '%s' not found in metallib.\n", fn_name);
        return NULL;
    }

    id<MTLComputePipelineState> state = [ctx->device newComputePipelineStateWithFunction:function
                                                                                   error:&error];
    if (!state) {
        fprintf(stderr, "Error: Failed to create pipeline from metallib for '%s': %s\n",
                fn_name, [[error localizedDescription] UTF8String]);
        return NULL;
    }

    MetalPipe* pipe = (MetalPipe*)calloc(1, sizeof(MetalPipe));
    if (!pipe) return NULL;
    pipe->state = state;
    return pipe;
}

uint32_t mtl_pipeline_max_threads(MetalPipe* pipe) {
    if (!pipe) return 0;
    return (uint32_t)[pipe->state maxTotalThreadsPerThreadgroup];
}

uint32_t mtl_pipeline_thread_execution_width(MetalPipe* pipe) {
    if (!pipe) return 0;
    return (uint32_t)[pipe->state threadExecutionWidth];
}

uint32_t mtl_pipeline_static_threadgroup_memory_length(MetalPipe* pipe) {
    if (!pipe) return 0;
    return (uint32_t)[pipe->state staticThreadgroupMemoryLength];
}

void mtl_free_pipeline(MetalPipe* pipe) {
    if (!pipe) return;
    pipe->state = nil;
    free(pipe);
}

// --- Command buffer & dispatch ---

MetalCmd* mtl_begin_command(MetalCtx* ctx) {
    if (!ctx) return NULL;

    id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
    if (!cmd_buf) {
        fprintf(stderr, "Error: Failed to create Metal command buffer.\n");
        return NULL;
    }

    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent];
    if (!encoder) {
        fprintf(stderr, "Error: Failed to create compute command encoder.\n");
        return NULL;
    }

    MetalCmd* cmd = (MetalCmd*)calloc(1, sizeof(MetalCmd));
    if (!cmd) return NULL;
    cmd->cmd_buf = cmd_buf;
    cmd->encoder = encoder;
    return cmd;
}

void mtl_dispatch(MetalCmd* cmd, MetalPipe* pipe,
                  const uint32_t grid[3], const uint32_t block[3],
                  MetalBuf** bufs, uint32_t n_bufs,
                  const void* push_data, size_t push_size) {
    if (!cmd || !pipe) return;

    [cmd->encoder setComputePipelineState:pipe->state];

    // Bind data buffers at indices 0..n_bufs-1
    for (uint32_t i = 0; i < n_bufs; i++) {
        if (bufs[i]) {
            [cmd->encoder setBuffer:bufs[i]->buffer offset:0 atIndex:i];
        }
    }

    // Bind push constants as a buffer at index n_bufs (equivalent to Vulkan push constants)
    if (push_data && push_size > 0) {
        [cmd->encoder setBytes:push_data length:push_size atIndex:n_bufs];
    }

    MTLSize threadgroups = MTLSizeMake(grid[0], grid[1], grid[2]);
    MTLSize threads_per = MTLSizeMake(block[0], block[1], block[2]);
    [cmd->encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per];
}

void mtl_dispatch_v2(MetalCmd* cmd, MetalPipe* pipe,
                     const uint32_t grid[3], const uint32_t block[3],
                     MetalBuf** bufs, uint32_t n_bufs,
                     const void* push_data, size_t push_size,
                     uint32_t push_idx) {
    if (!cmd || !pipe) return;

    [cmd->encoder setComputePipelineState:pipe->state];

    // Bind data buffers, shifting indices to skip push_idx
    for (uint32_t i = 0; i < n_bufs; i++) {
        uint32_t slot = (i < push_idx) ? i : (i + 1);
        if (bufs[i]) {
            [cmd->encoder setBuffer:bufs[i]->buffer offset:0 atIndex:slot];
        }
    }

    // Bind push constants at push_idx via setBytes (inlined into command buffer)
    if (push_data && push_size > 0) {
        [cmd->encoder setBytes:push_data length:push_size atIndex:push_idx];
    }

    MTLSize threadgroups = MTLSizeMake(grid[0], grid[1], grid[2]);
    MTLSize threads_per = MTLSizeMake(block[0], block[1], block[2]);
    [cmd->encoder dispatchThreadgroups:threadgroups threadsPerThreadgroup:threads_per];
}

void mtl_barrier(MetalCmd* cmd) {
    if (!cmd) return;
    // Lightweight in-encoder memory barrier — ensures all buffer writes from
    // previous dispatches are visible to subsequent dispatches, without
    // destroying and recreating the compute command encoder.
    // Available since macOS 10.14; Apple Silicon requires macOS 11+.
    [cmd->encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
}

void mtl_commit_and_wait(MetalCmd* cmd) {
    if (!cmd) return;
    [cmd->encoder endEncoding];
    [cmd->cmd_buf commit];
    [cmd->cmd_buf waitUntilCompleted];

    if ([cmd->cmd_buf status] == MTLCommandBufferStatusError) {
        fprintf(stderr, "Error: Metal command buffer failed: %s\n",
                [[cmd->cmd_buf.error localizedDescription] UTF8String]);
    }

    cmd->encoder = nil;
    cmd->cmd_buf = nil;
    free(cmd);
}

void mtl_commit_async(MetalCmd* cmd) {
    if (!cmd) return;
    [cmd->encoder endEncoding];
    cmd->encoder = nil;
    [cmd->cmd_buf commit];
}

void mtl_wait(MetalCmd* cmd) {
    if (!cmd) return;
    [cmd->cmd_buf waitUntilCompleted];

    if ([cmd->cmd_buf status] == MTLCommandBufferStatusError) {
        fprintf(stderr, "Error: Metal command buffer failed: %s\n",
                [[cmd->cmd_buf.error localizedDescription] UTF8String]);
    }

    cmd->cmd_buf = nil;
    free(cmd);
}
