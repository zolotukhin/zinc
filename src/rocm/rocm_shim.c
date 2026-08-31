// ROCm/HIP implementation of ZINC's accelerator C ABI.
//
// The Zig CUDA orchestration is intentionally API-agnostic above cuda_shim.h:
// buffers, pipelines, commands, and graph handles are opaque. Implementing the
// same ABI here lets ROCm reuse the mature model loader and forward scheduler
// while hipRTC compiles the portable subset of src/shaders/cuda/kernels.cu.

#include "cuda_shim.h"

#include <hip/hip_runtime_api.h>
#include <hip/hiprtc.h>
#include <hipblas/hipblas.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DISPATCH_BUFS 32

struct CudaCtx {
    int device;
    hipStream_t stream;
    hipblasHandle_t hipblas;
    hipDeviceProp_t props;
    hipModule_t kernel_module;
};

struct CudaBuf {
    void* dptr;
    size_t size;
    void* host;
    int owns;
    int owns_host;
};

struct CudaPipe {
    hipFunction_t fn;
    hipModule_t owned_module;
    char* name;
};

struct CudaCmd {
    hipStream_t stream;
    hipEvent_t event;
};

struct CudaGraph {
    hipGraphExec_t exec;
    int have_exec;
};

static _Thread_local char g_err[2048];

static void set_err(const char* where, const char* what) {
    snprintf(g_err, sizeof g_err, "%s: %s", where, what ? what : "(null)");
    fprintf(stderr, "[rocm_shim] %s\n", g_err);
}

static int hip_ok(hipError_t result, const char* where) {
    if (result == hipSuccess) return 1;
    set_err(where, hipGetErrorString(result));
    return 0;
}

const char* cuda_last_error(void) {
    return g_err;
}

// ---- Device lifecycle -------------------------------------------------------

CudaCtx* cuda_init(int device_index) {
    g_err[0] = 0;

    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || device_index < 0 || device_index >= count) {
        return NULL; // callers probe ascending ordinals; out-of-range is normal
    }
    if (!hip_ok(hipSetDevice(device_index), "hipSetDevice")) return NULL;

    CudaCtx* ctx = (CudaCtx*)calloc(1, sizeof *ctx);
    if (!ctx) {
        set_err("cuda_init", "out of memory");
        return NULL;
    }
    ctx->device = device_index;
    if (!hip_ok(hipGetDeviceProperties(&ctx->props, device_index), "hipGetDeviceProperties") ||
        !hip_ok(hipStreamCreateWithFlags(&ctx->stream, hipStreamNonBlocking), "hipStreamCreateWithFlags")) {
        free(ctx);
        return NULL;
    }

    // hipBLAS is used only by the optional large-prefill path. Failure is
    // non-fatal because the ROCm milestone defaults to portable matvec kernels.
    if (hipblasCreate(&ctx->hipblas) == HIPBLAS_STATUS_SUCCESS) {
        hipblasSetStream(ctx->hipblas, ctx->stream);
    } else {
        ctx->hipblas = NULL;
    }
    return ctx;
}

void cuda_destroy(CudaCtx* ctx) {
    if (!ctx) return;
    hipSetDevice(ctx->device);
    if (ctx->stream) hipStreamSynchronize(ctx->stream);
    if (ctx->kernel_module) hipModuleUnload(ctx->kernel_module);
    if (ctx->hipblas) hipblasDestroy(ctx->hipblas);
    if (ctx->stream) hipStreamDestroy(ctx->stream);
    free(ctx);
}

uint64_t cuda_total_memory(CudaCtx* ctx) {
    return ctx ? (uint64_t)ctx->props.totalGlobalMem : 0;
}

uint64_t cuda_free_memory(CudaCtx* ctx) {
    if (!ctx) return 0;
    hipSetDevice(ctx->device);
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    if (hipMemGetInfo(&free_bytes, &total_bytes) != hipSuccess) return 0;
    return (uint64_t)free_bytes;
}

uint32_t cuda_sm_count(CudaCtx* ctx) {
    return ctx ? (uint32_t)ctx->props.multiProcessorCount : 0;
}

uint32_t cuda_compute_capability(CudaCtx* ctx) {
    if (!ctx) return 0;
    // CudaDevice.initBest ranks this value. CU count selects the discrete GPU
    // over an integrated adapter, while the cap keeps NVIDIA cubin loading off.
    uint32_t cus = (uint32_t)ctx->props.multiProcessorCount;
    return cus > 88 ? 88 : (cus ? cus : 1);
}

uint32_t cuda_max_threads_per_block(CudaCtx* ctx) {
    return ctx ? (uint32_t)ctx->props.maxThreadsPerBlock : 0;
}

uint32_t cuda_max_shared_mem_per_block(CudaCtx* ctx) {
    if (!ctx) return 0;
    int value = 0;
    if (hipDeviceGetAttribute(&value, hipDeviceAttributeMaxSharedMemoryPerBlock, ctx->device) != hipSuccess) {
        return (uint32_t)ctx->props.sharedMemPerBlock;
    }
    return (uint32_t)value;
}

uint32_t cuda_warp_size(CudaCtx* ctx) {
    return ctx ? (uint32_t)ctx->props.warpSize : 0;
}

void cuda_device_name(CudaCtx* ctx, char* out, size_t cap) {
    if (!out || cap == 0) return;
    if (!ctx) {
        out[0] = 0;
        return;
    }
    snprintf(out, cap, "%s [%s]", ctx->props.name, ctx->props.gcnArchName);
    out[cap - 1] = 0;
}

void cuda_device_arch(CudaCtx* ctx, char* out, size_t cap) {
    if (!out || cap == 0) return;
    if (!ctx) {
        out[0] = 0;
        return;
    }
    snprintf(out, cap, "%s", ctx->props.gcnArchName);
    out[cap - 1] = 0;
}

// ---- Buffer management ------------------------------------------------------

CudaBuf* cuda_create_buffer(CudaCtx* ctx, size_t size) {
    if (!ctx) return NULL;
    hipSetDevice(ctx->device);
    CudaBuf* buf = (CudaBuf*)calloc(1, sizeof *buf);
    if (!buf) {
        set_err("cuda_create_buffer", "out of memory");
        return NULL;
    }
    if (!hip_ok(hipMalloc(&buf->dptr, size ? size : 1), "hipMalloc")) {
        free(buf);
        return NULL;
    }
    buf->size = size;
    buf->owns = 1;
    return buf;
}

CudaBuf* cuda_create_buffer_staged(CudaCtx* ctx, size_t size, void** cpu_ptr) {
    CudaBuf* buf = cuda_create_buffer(ctx, size);
    if (!buf) return NULL;
    if (!hip_ok(hipHostMalloc(&buf->host, size ? size : 1, hipHostMallocDefault), "hipHostMalloc")) {
        cuda_free_buffer(buf);
        return NULL;
    }
    buf->owns_host = 1;
    if (cpu_ptr) *cpu_ptr = buf->host;
    return buf;
}

CudaBuf* cuda_upload_mmap(CudaCtx* ctx, const void* host_ptr, size_t size) {
    CudaBuf* buf = cuda_create_buffer(ctx, size);
    if (!buf) return NULL;
    if (!hip_ok(hipMemcpy(buf->dptr, host_ptr, size, hipMemcpyHostToDevice), "hipMemcpy(mmap)")) {
        cuda_free_buffer(buf);
        return NULL;
    }
    return buf;
}

CudaBuf* cuda_alias_buffer(CudaBuf* base, size_t offset, size_t size) {
    if (!base || offset > base->size || size > base->size - offset) {
        set_err("cuda_alias_buffer", "view exceeds parent buffer");
        return NULL;
    }
    CudaBuf* buf = (CudaBuf*)calloc(1, sizeof *buf);
    if (!buf) {
        set_err("cuda_alias_buffer", "out of memory");
        return NULL;
    }
    buf->dptr = (void*)((unsigned char*)base->dptr + offset);
    buf->size = size;
    return buf;
}

uint64_t cuda_buffer_device_ptr(CudaBuf* buf) {
    return buf ? (uint64_t)(uintptr_t)buf->dptr : 0;
}

void cuda_upload(CudaCtx* ctx, CudaBuf* buf, const void* src, size_t size) {
    if (!ctx || !buf) return;
    if (!hip_ok(hipMemcpyAsync(buf->dptr, src, size, hipMemcpyHostToDevice, ctx->stream), "hipMemcpyAsync(H2D)")) return;
    hip_ok(hipStreamSynchronize(ctx->stream), "hipStreamSynchronize(H2D)");
}

void cuda_download(CudaCtx* ctx, CudaBuf* buf, void* dst, size_t size) {
    if (!ctx || !buf) return;
    if (!hip_ok(hipMemcpyAsync(dst, buf->dptr, size, hipMemcpyDeviceToHost, ctx->stream), "hipMemcpyAsync(D2H)")) return;
    hip_ok(hipStreamSynchronize(ctx->stream), "hipStreamSynchronize(D2H)");
}

void cuda_upload_async(CudaCtx* ctx, CudaBuf* buf, const void* src, size_t size) {
    if (ctx && buf) hip_ok(hipMemcpyAsync(buf->dptr, src, size, hipMemcpyHostToDevice, ctx->stream), "hipMemcpyAsync(H2D)");
}

void cuda_download_async(CudaCtx* ctx, CudaBuf* buf, void* dst, size_t size) {
    if (ctx && buf) hip_ok(hipMemcpyAsync(dst, buf->dptr, size, hipMemcpyDeviceToHost, ctx->stream), "hipMemcpyAsync(D2H)");
}

void* cuda_alloc_host(size_t size) {
    void* ptr = NULL;
    if (!hip_ok(hipHostMalloc(&ptr, size ? size : 1, hipHostMallocDefault), "hipHostMalloc")) return NULL;
    return ptr;
}

void cuda_free_host(void* ptr) {
    if (ptr) hipHostFree(ptr);
}

void cuda_free_buffer(CudaBuf* buf) {
    if (!buf) return;
    if (buf->owns && buf->dptr) hipFree(buf->dptr);
    if (buf->owns_host && buf->host) hipHostFree(buf->host);
    free(buf);
}

// ---- Pipeline management (hipRTC) ------------------------------------------

static int compile_kernel_module(CudaCtx* ctx, const char* source,
                                 const char* const* extra_opts, uint32_t n_extra_opts) {
    if (ctx->kernel_module) return 1;

    hiprtcProgram program;
    hiprtcResult result = hiprtcCreateProgram(&program, source, "zinc_kernels.hip", 0, NULL, NULL);
    if (result != HIPRTC_SUCCESS) {
        set_err("hiprtcCreateProgram", hiprtcGetErrorString(result));
        return 0;
    }

    char arch_opt[192];
    snprintf(arch_opt, sizeof arch_opt, "--offload-arch=%s", ctx->props.gcnArchName);
    const char* rocm_path = getenv("ROCM_PATH");
    if (!rocm_path || !rocm_path[0]) rocm_path = getenv("ROCM_HOME");
    if (!rocm_path || !rocm_path[0]) rocm_path = "/opt/rocm";
    char include_opt[1024];
    snprintf(include_opt, sizeof include_opt, "-I%s/include", rocm_path);

    const uint32_t base_count = 5;
    const char** opts = (const char**)calloc(base_count + n_extra_opts, sizeof *opts);
    if (!opts) {
        hiprtcDestroyProgram(&program);
        set_err("compile_kernel_module", "out of memory");
        return 0;
    }
    opts[0] = arch_opt;
    opts[1] = "--std=c++17";
    opts[2] = "-DZINC_ROCM=1";
    opts[3] = include_opt;
    opts[4] = "-O3";
    for (uint32_t i = 0; i < n_extra_opts; ++i) opts[base_count + i] = extra_opts[i];

    result = hiprtcCompileProgram(program, (int)(base_count + n_extra_opts), opts);
    free(opts);
    if (result != HIPRTC_SUCCESS) {
        size_t log_size = 0;
        hiprtcGetProgramLogSize(program, &log_size);
        char* log = (char*)malloc(log_size + 1);
        if (log) {
            hiprtcGetProgramLog(program, log);
            log[log_size] = 0;
            fprintf(stderr, "[rocm_shim] hipRTC log:\n%s\n", log);
            free(log);
        }
        set_err("hiprtcCompileProgram", hiprtcGetErrorString(result));
        hiprtcDestroyProgram(&program);
        return 0;
    }

    size_t code_size = 0;
    result = hiprtcGetCodeSize(program, &code_size);
    if (result != HIPRTC_SUCCESS || code_size == 0) {
        set_err("hiprtcGetCodeSize", hiprtcGetErrorString(result));
        hiprtcDestroyProgram(&program);
        return 0;
    }
    char* code = (char*)malloc(code_size);
    if (!code) {
        hiprtcDestroyProgram(&program);
        set_err("compile_kernel_module", "out of memory for code object");
        return 0;
    }
    result = hiprtcGetCode(program, code);
    hiprtcDestroyProgram(&program);
    if (result != HIPRTC_SUCCESS) {
        free(code);
        set_err("hiprtcGetCode", hiprtcGetErrorString(result));
        return 0;
    }
    const char* dump_path = getenv("ZINC_ROCM_DUMP_CODE");
    if (dump_path && dump_path[0]) {
        FILE* dump = fopen(dump_path, "wb");
        if (dump) {
            fwrite(code, 1, code_size, dump);
            fclose(dump);
        }
    }
    hipError_t load_result = hipModuleLoadData(&ctx->kernel_module, code);
    free(code);
    return hip_ok(load_result, "hipModuleLoadData");
}

CudaPipe* cuda_create_pipeline(CudaCtx* ctx, const char* source, const char* fn_name,
                               const char* const* opts, uint32_t n_opts) {
    if (!ctx || !source || !fn_name) return NULL;
    hipSetDevice(ctx->device);
    if (!compile_kernel_module(ctx, source, opts, n_opts)) return NULL;

    CudaPipe* pipe = (CudaPipe*)calloc(1, sizeof *pipe);
    if (!pipe) {
        set_err("cuda_create_pipeline", "out of memory");
        return NULL;
    }
    if (!hip_ok(hipModuleGetFunction(&pipe->fn, ctx->kernel_module, fn_name), fn_name)) {
        free(pipe);
        return NULL;
    }
    pipe->name = (char*)malloc(strlen(fn_name) + 1);
    if (pipe->name) strcpy(pipe->name, fn_name);
    if (getenv("ZINC_ROCM_KERNEL_ATTRS") && strstr(fn_name, "wmma_i8")) {
        int regs = 0, local = 0, shared = 0, max_threads = 0;
        hipFuncGetAttribute(&regs, HIP_FUNC_ATTRIBUTE_NUM_REGS, pipe->fn);
        hipFuncGetAttribute(&local, HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, pipe->fn);
        hipFuncGetAttribute(&shared, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES, pipe->fn);
        hipFuncGetAttribute(&max_threads, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK, pipe->fn);
        fprintf(stderr, "[rocm_shim] %s: regs=%d local=%d shared=%d max_threads=%d\n",
                fn_name, regs, local, shared, max_threads);
    }
    return pipe;
}

CudaPipe* cuda_create_pipeline_from_image(CudaCtx* ctx, const void* image, size_t image_size,
                                          const char* fn_name) {
    (void)image_size;
    if (!ctx || !image || !fn_name) return NULL;
    hipModule_t module = NULL;
    if (!hip_ok(hipModuleLoadData(&module, image), "hipModuleLoadData(image)")) return NULL;
    CudaPipe* pipe = (CudaPipe*)calloc(1, sizeof *pipe);
    if (!pipe) {
        hipModuleUnload(module);
        return NULL;
    }
    if (!hip_ok(hipModuleGetFunction(&pipe->fn, module, fn_name), fn_name)) {
        hipModuleUnload(module);
        free(pipe);
        return NULL;
    }
    pipe->owned_module = module;
    pipe->name = (char*)malloc(strlen(fn_name) + 1);
    if (pipe->name) strcpy(pipe->name, fn_name);
    return pipe;
}

static uint32_t function_attribute(CudaPipe* pipe, hipFunction_attribute attr) {
    if (!pipe) return 0;
    int value = 0;
    if (hipFuncGetAttribute(&value, attr, pipe->fn) != hipSuccess) return 0;
    return (uint32_t)value;
}

uint32_t cuda_pipeline_max_threads(CudaPipe* pipe) {
    return function_attribute(pipe, HIP_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
}

uint32_t cuda_pipeline_shared_mem(CudaPipe* pipe) {
    return function_attribute(pipe, HIP_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES);
}

void cuda_pipeline_set_max_dynamic_shared(CudaPipe* pipe, uint32_t bytes) {
    if (pipe) hipFuncSetAttribute(pipe->fn, HIP_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)bytes);
}

void cuda_free_pipeline(CudaPipe* pipe) {
    if (!pipe) return;
    if (pipe->owned_module) hipModuleUnload(pipe->owned_module);
    free(pipe->name);
    free(pipe); // hipRTC module ownership belongs to CudaCtx
}

// ---- Command submission -----------------------------------------------------

CudaCmd* cuda_begin_command(CudaCtx* ctx) {
    if (!ctx) return NULL;
    CudaCmd* cmd = (CudaCmd*)calloc(1, sizeof *cmd);
    if (!cmd) {
        set_err("cuda_begin_command", "out of memory");
        return NULL;
    }
    cmd->stream = ctx->stream;
    if (!hip_ok(hipEventCreateWithFlags(&cmd->event, hipEventDisableTiming), "hipEventCreateWithFlags")) {
        free(cmd);
        return NULL;
    }
    return cmd;
}

void cuda_dispatch(CudaCmd* cmd, CudaPipe* pipe,
                   const uint32_t grid[3], const uint32_t block[3],
                   CudaBuf** bufs, uint32_t n_bufs,
                   const void* push_data, size_t push_size,
                   uint32_t shared_bytes) {
    (void)push_size;
    if (!cmd || !pipe || n_bufs > MAX_DISPATCH_BUFS) {
        set_err("cuda_dispatch", "invalid dispatch");
        return;
    }
    void* device_ptrs[MAX_DISPATCH_BUFS];
    void* args[MAX_DISPATCH_BUFS + 1];
    for (uint32_t i = 0; i < n_bufs; ++i) {
        device_ptrs[i] = bufs[i]->dptr;
        args[i] = &device_ptrs[i];
    }
    uint32_t n_args = n_bufs;
    if (push_data) args[n_args++] = (void*)push_data;
    (void)n_args;

    const char* timing_filter = getenv("ZINC_ROCM_TIME_KERNEL");
    const int time_kernel = timing_filter && pipe->name && strstr(pipe->name, timing_filter);
    hipEvent_t start = NULL, stop = NULL;
    if (time_kernel && hipEventCreate(&start) == hipSuccess && hipEventCreate(&stop) == hipSuccess)
        hipEventRecord(start, cmd->stream);

    hip_ok(hipModuleLaunchKernel(pipe->fn,
                                grid[0], grid[1], grid[2],
                                block[0], block[1], block[2],
                                shared_bytes, cmd->stream, args, NULL),
           "hipModuleLaunchKernel");

    if (start && stop) {
        hipEventRecord(stop, cmd->stream);
        hipEventSynchronize(stop);
        float ms = 0.0f;
        hipEventElapsedTime(&ms, start, stop);
        fprintf(stderr, "[rocm_time] %s %.4f ms grid=%ux%u\n",
                pipe->name, ms, grid[0], grid[1]);
    }
    if (start) hipEventDestroy(start);
    if (stop) hipEventDestroy(stop);
}

void cuda_barrier(CudaCmd* cmd) {
    (void)cmd; // a single HIP stream is ordered
}

void cuda_commit_and_wait(CudaCmd* cmd) {
    if (!cmd) return;
    hipEventRecord(cmd->event, cmd->stream);
    hipStreamSynchronize(cmd->stream);
    hipEventDestroy(cmd->event);
    free(cmd);
}

void cuda_commit_async(CudaCmd* cmd) {
    if (cmd) hipEventRecord(cmd->event, cmd->stream);
}

void cuda_wait(CudaCmd* cmd) {
    if (!cmd) return;
    hipEventSynchronize(cmd->event);
    hipEventDestroy(cmd->event);
    free(cmd);
}

void cuda_release_completed(CudaCmd* cmd) {
    if (!cmd) return;
    if (cmd->event) hipEventDestroy(cmd->event);
    free(cmd);
}

// ---- hipBLAS ---------------------------------------------------------------

void cuda_cublas_hgemm(CudaCtx* ctx, unsigned M, unsigned N, unsigned K,
                       CudaBuf* W, CudaBuf* A, CudaBuf* Y, float beta) {
    if (!ctx || !ctx->hipblas || !W || !A || !Y) {
        set_err("cuda_cublas_hgemm", "hipBLAS unavailable");
        return;
    }
    const float alpha = 1.0f;
    hipblasStatus_t status = hipblasGemmEx(
        ctx->hipblas, HIPBLAS_OP_T, HIPBLAS_OP_N,
        (int)M, (int)N, (int)K,
        &alpha, W->dptr, HIP_R_16F, (int)K,
        A->dptr, HIP_R_16F, (int)K,
        &beta, Y->dptr, HIP_R_32F, (int)M,
        HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    if (status != HIPBLAS_STATUS_SUCCESS) set_err("hipblasGemmEx", "failed");
}

// ---- HIP graphs (decode capture/replay) -------------------------------------
// The CUDA scheduler records an invariant per-token kernel chain and updates
// only its node parameters on subsequent tokens. HIP exposes the same capture,
// exec-update, and launch primitives, including async copies from pinned host
// memory, so the orchestration can be shared without a ROCm-specific graph path.
int cuda_graph_supported(void) { return 1; }

CudaGraph* cuda_graph_create(void) {
    CudaGraph* graph = (CudaGraph*)calloc(1, sizeof *graph);
    if (!graph) set_err("cuda_graph_create", "out of memory");
    return graph;
}

int cuda_graph_begin(CudaCtx* ctx) {
    if (!ctx) return 0;
    hipSetDevice(ctx->device);
    return hip_ok(hipStreamBeginCapture(ctx->stream, hipStreamCaptureModeRelaxed),
                  "hipStreamBeginCapture");
}

int cuda_graph_end_launch(CudaCtx* ctx, CudaGraph* cached) {
    if (!ctx || !cached) return 0;
    hipGraph_t graph = NULL;
    if (!hip_ok(hipStreamEndCapture(ctx->stream, &graph), "hipStreamEndCapture")) return 0;

    if (cached->have_exec) {
        hipGraphNode_t error_node = NULL;
        hipGraphExecUpdateResult update_result = hipGraphExecUpdateError;
        if (hipGraphExecUpdate(cached->exec, graph, &error_node, &update_result) != hipSuccess ||
            update_result != hipGraphExecUpdateSuccess) {
            hipGraphExecDestroy(cached->exec);
            cached->exec = NULL;
            cached->have_exec = 0;
        }
    }
    if (!cached->have_exec) {
        if (!hip_ok(hipGraphInstantiate(&cached->exec, graph, NULL, NULL, 0),
                    "hipGraphInstantiate")) {
            hipGraphDestroy(graph);
            return 0;
        }
        cached->have_exec = 1;
    }
    hipGraphDestroy(graph);
    if (!hip_ok(hipGraphLaunch(cached->exec, ctx->stream), "hipGraphLaunch")) return 0;
    return hip_ok(hipStreamSynchronize(ctx->stream), "hipStreamSynchronize(graph)");
}

void cuda_graph_free(CudaGraph* graph) {
    if (!graph) return;
    if (graph->have_exec) hipGraphExecDestroy(graph->exec);
    free(graph);
}
