// CUDA backend C implementation for ZINC. See cuda_shim.h for the ABI contract.
// Uses the CUDA Driver API (cu*) + NVRTC for runtime kernel compilation.
// Link: -lcuda -lnvrtc  (libcuda.so in /usr/lib/wsl/lib on WSL).

#include "cuda_shim.h"
#include <cuda.h>
#include <nvrtc.h>
#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_DISPATCH_BUFS 32

struct CudaCtx { CUdevice dev; CUcontext ctx; CUstream stream; cublasHandle_t cublas; };
struct CudaBuf { CUdeviceptr dptr; size_t size; void* host; int owns; int owns_host; };
struct CudaPipe { CUmodule mod; CUfunction fn; };
struct CudaCmd { CUstream stream; CUevent event; };

static __thread char g_err[1024];

static void set_err(const char* where, const char* what) {
    snprintf(g_err, sizeof g_err, "%s: %s", where, what ? what : "(null)");
    fprintf(stderr, "[cuda_shim] %s\n", g_err);
}
static int cu_ok(CUresult r, const char* where) {
    if (r == CUDA_SUCCESS) return 1;
    const char* s = NULL; cuGetErrorString(r, &s);
    set_err(where, s);
    return 0;
}
const char* cuda_last_error(void) { return g_err; }

// ---- Device lifecycle --------------------------------------------------------
CudaCtx* cuda_init(int device_index) {
    g_err[0] = 0;
    static int inited = 0;
    if (!inited) { if (!cu_ok(cuInit(0), "cuInit")) return NULL; inited = 1; }
    CudaCtx* c = (CudaCtx*)calloc(1, sizeof *c);
    if (!c) { set_err("cuda_init", "oom"); return NULL; }
    // Silent on failure: callers (e.g. initBest) probe ascending ordinals to
    // enumerate devices, so an out-of-range index here is expected, not an error.
    if (cuDeviceGet(&c->dev, device_index) != CUDA_SUCCESS) { free(c); return NULL; }
    // Use the device's primary context (shared with the runtime API).
    if (!cu_ok(cuDevicePrimaryCtxRetain(&c->ctx, c->dev), "cuDevicePrimaryCtxRetain")) { free(c); return NULL; }
    if (!cu_ok(cuCtxSetCurrent(c->ctx), "cuCtxSetCurrent")) { free(c); return NULL; }
    if (!cu_ok(cuStreamCreate(&c->stream, CU_STREAM_NON_BLOCKING), "cuStreamCreate")) { free(c); return NULL; }
    // Effort 26 cycle 9: cuBLAS handle for the prefill dense Q4_K GEMM (created
    // lazily-safe here; binds to the current primary context). Non-fatal on
    // failure — the cuBLAS prefill path is opt-in and falls back if absent.
    if (cublasCreate(&c->cublas) == CUBLAS_STATUS_SUCCESS) {
        cublasSetStream(c->cublas, c->stream);
        cublasSetMathMode(c->cublas, CUBLAS_TENSOR_OP_MATH);
    } else {
        c->cublas = NULL;
    }
    return c;
}
void cuda_destroy(CudaCtx* c) {
    if (!c) return;
    if (c->cublas) cublasDestroy(c->cublas);
    if (c->stream) cuStreamDestroy(c->stream);
    if (c->ctx) cuDevicePrimaryCtxRelease(c->dev);
    free(c);
}
// Effort 26 cycle 9 — dense prefill Q4_K GEMM via cuBLAS fp16 tensor cores.
// Computes Y[N,M] (token-major row-major) = A[N,K]·W[M,K]^T. In cuBLAS's
// column-major terms: C(M×N) = op(W)·op(A) with op(W)=T (W is row-major M×K =
// col-major K×M, lda=K) and op(A)=N (A is row-major N×K = col-major K×N, ldb=K);
// C is col-major M×N (ldc=M) = row-major N×M = the token-major Y. fp16 in, fp32
// accumulate/out — matches gemm_q4k_tc's fp16-multiply / fp32-accumulate.
void cuda_cublas_hgemm(CudaCtx* c, unsigned M, unsigned N, unsigned K,
                       CudaBuf* W, CudaBuf* A, CudaBuf* Y, float beta) {
    if (!c || !c->cublas) { set_err("cuda_cublas_hgemm", "no cublas handle"); return; }
    const float alpha = 1.0f;
    cublasStatus_t s = cublasGemmEx(
        c->cublas, CUBLAS_OP_T, CUBLAS_OP_N, (int)M, (int)N, (int)K,
        &alpha, (const void*)W->dptr, CUDA_R_16F, (int)K,
        (const void*)A->dptr, CUDA_R_16F, (int)K,
        &beta, (void*)Y->dptr, CUDA_R_32F, (int)M,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) set_err("cuda_cublas_hgemm", "cublasGemmEx failed");
}

void cuda_cublas_hgemm_strided_batched(CudaCtx* c, unsigned trans_a, unsigned trans_b,
                                       unsigned M, unsigned N, unsigned K,
                                       CudaBuf* A, size_t a_offset, unsigned lda, int64_t stride_a,
                                       CudaBuf* B, size_t b_offset, unsigned ldb, int64_t stride_b,
                                       CudaBuf* C, size_t c_offset, unsigned ldc, int64_t stride_c,
                                       unsigned batch_count, float beta) {
    if (!c || !c->cublas || !A || !B || !C) {
        set_err("cuda_cublas_hgemm_strided_batched", "cuBLAS unavailable");
        return;
    }
    const float alpha = 1.0f;
    const void* ap = (const void*)(uintptr_t)(A->dptr + a_offset);
    const void* bp = (const void*)(uintptr_t)(B->dptr + b_offset);
    void* cp = (void*)(uintptr_t)(C->dptr + c_offset);
    cublasStatus_t s = cublasGemmStridedBatchedEx(
        c->cublas,
        trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
        trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
        (int)M, (int)N, (int)K,
        &alpha, ap, CUDA_R_16F, (int)lda, (long long int)stride_a,
        bp, CUDA_R_16F, (int)ldb, (long long int)stride_b,
        &beta, cp, CUDA_R_32F, (int)ldc, (long long int)stride_c,
        (int)batch_count, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (s != CUBLAS_STATUS_SUCCESS) set_err("cuda_cublas_hgemm_strided_batched", "cublasGemmStridedBatchedEx failed");
}

void cuda_cublas_sgemm_strided_batched(CudaCtx* c, unsigned trans_a, unsigned trans_b,
                                       unsigned M, unsigned N, unsigned K,
                                       CudaBuf* A, size_t a_offset, unsigned lda, int64_t stride_a,
                                       CudaBuf* B, size_t b_offset, unsigned ldb, int64_t stride_b,
                                       CudaBuf* C, size_t c_offset, unsigned ldc, int64_t stride_c,
                                       unsigned batch_count, float beta) {
    if (!c || !c->cublas || !A || !B || !C) {
        set_err("cuda_cublas_sgemm_strided_batched", "cuBLAS unavailable");
        return;
    }
    const float alpha = 1.0f;
    const float* ap = (const float*)(uintptr_t)(A->dptr + a_offset);
    const float* bp = (const float*)(uintptr_t)(B->dptr + b_offset);
    float* cp = (float*)(uintptr_t)(C->dptr + c_offset);
    cublasStatus_t s = cublasSgemmStridedBatched(
        c->cublas,
        trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
        trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
        (int)M, (int)N, (int)K,
        &alpha, ap, (int)lda, (long long int)stride_a,
        bp, (int)ldb, (long long int)stride_b,
        &beta, cp, (int)ldc, (long long int)stride_c,
        (int)batch_count);
    if (s != CUBLAS_STATUS_SUCCESS) set_err("cuda_cublas_sgemm_strided_batched", "cublasSgemmStridedBatched failed");
}
static int dev_attr(CUdevice d, CUdevice_attribute a) {
    int v = 0; cuDeviceGetAttribute(&v, a, d); return v;
}
uint64_t cuda_total_memory(CudaCtx* c) { size_t b = 0; cuDeviceTotalMem(&b, c->dev); return b; }
uint64_t cuda_free_memory(CudaCtx* c) { size_t f = 0, t = 0; cuCtxSetCurrent(c->ctx); cuMemGetInfo(&f, &t); return f; }
uint32_t cuda_sm_count(CudaCtx* c) { return (uint32_t)dev_attr(c->dev, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT); }
uint32_t cuda_compute_capability(CudaCtx* c) {
    int mj = dev_attr(c->dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    int mn = dev_attr(c->dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    return (uint32_t)(mj * 10 + mn);
}
uint32_t cuda_max_threads_per_block(CudaCtx* c) { return (uint32_t)dev_attr(c->dev, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK); }
uint32_t cuda_max_shared_mem_per_block(CudaCtx* c) { return (uint32_t)dev_attr(c->dev, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN); }
uint32_t cuda_warp_size(CudaCtx* c) { return (uint32_t)dev_attr(c->dev, CU_DEVICE_ATTRIBUTE_WARP_SIZE); }
void cuda_device_name(CudaCtx* c, char* out, size_t cap) {
    if (cap == 0) return;
    if (cuDeviceGetName(out, (int)cap, c->dev) != CUDA_SUCCESS) out[0] = 0;
    out[cap - 1] = 0;
}
void cuda_device_arch(CudaCtx* c, char* out, size_t cap) {
    if (cap == 0) return;
    if (!c) { out[0] = 0; return; }
    int mj = dev_attr(c->dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
    int mn = dev_attr(c->dev, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
    snprintf(out, cap, "sm_%d%d", mj, mn);
    out[cap - 1] = 0;
}

// ---- Buffer management -------------------------------------------------------
CudaBuf* cuda_create_buffer(CudaCtx* c, size_t size) {
    cuCtxSetCurrent(c->ctx);
    CudaBuf* b = (CudaBuf*)calloc(1, sizeof *b);
    if (!b) { set_err("cuda_create_buffer", "oom"); return NULL; }
    if (!cu_ok(cuMemAlloc(&b->dptr, size ? size : 1), "cuMemAlloc")) { free(b); return NULL; }
    b->size = size; b->owns = 1;
    return b;
}
CudaBuf* cuda_create_buffer_staged(CudaCtx* c, size_t size, void** cpu_ptr) {
    CudaBuf* b = cuda_create_buffer(c, size);
    if (!b) return NULL;
    if (!cu_ok(cuMemAllocHost(&b->host, size ? size : 1), "cuMemAllocHost")) { cuda_free_buffer(b); return NULL; }
    b->owns_host = 1;
    if (cpu_ptr) *cpu_ptr = b->host;
    return b;
}
CudaBuf* cuda_upload_mmap(CudaCtx* c, const void* host_ptr, size_t size) {
    CudaBuf* b = cuda_create_buffer(c, size);
    if (!b) return NULL;
    if (!cu_ok(cuMemcpyHtoD(b->dptr, host_ptr, size), "cuMemcpyHtoD(mmap)")) { cuda_free_buffer(b); return NULL; }
    return b;
}
CudaBuf* cuda_alias_buffer(CudaBuf* base, size_t offset, size_t size) {
    CudaBuf* b = (CudaBuf*)calloc(1, sizeof *b);
    if (!b) { set_err("cuda_alias_buffer", "oom"); return NULL; }
    b->dptr = base->dptr + offset; b->size = size; b->owns = 0;
    return b;
}
uint64_t cuda_buffer_device_ptr(CudaBuf* b) { return (uint64_t)b->dptr; }
void cuda_upload(CudaCtx* c, CudaBuf* b, const void* src, size_t size) {
    cuCtxSetCurrent(c->ctx);
    if (!cu_ok(cuMemcpyHtoDAsync(b->dptr, src, size, c->stream), "cuMemcpyHtoDAsync")) return;
    cuStreamSynchronize(c->stream);
}
void cuda_download(CudaCtx* c, CudaBuf* b, void* dst, size_t size) {
    cuCtxSetCurrent(c->ctx);
    if (!cu_ok(cuMemcpyDtoHAsync(dst, b->dptr, size, c->stream), "cuMemcpyDtoHAsync")) return;
    cuStreamSynchronize(c->stream);
}
// Async variants: enqueue the copy on the ctx stream and return WITHOUT syncing.
// Issued between cuda_graph_begin/end_launch they become memcpy graph nodes, so
// the embed H2D and argmax D2H ride the single graph launch instead of each
// costing a WSL2 sync round-trip. Host side must be pinned (cuda_alloc_host).
void cuda_upload_async(CudaCtx* c, CudaBuf* b, const void* src, size_t size) {
    cuMemcpyHtoDAsync(b->dptr, src, size, c->stream);
}
void cuda_download_async(CudaCtx* c, CudaBuf* b, void* dst, size_t size) {
    cuMemcpyDtoHAsync(dst, b->dptr, size, c->stream);
}
void* cuda_alloc_host(size_t size) {
    void* p = NULL;
    if (!cu_ok(cuMemAllocHost(&p, size ? size : 1), "cuMemAllocHost")) return NULL;
    return p;
}
void cuda_free_host(void* p) { if (p) cuMemFreeHost(p); }
void cuda_free_buffer(CudaBuf* b) {
    if (!b) return;
    if (b->owns && b->dptr) cuMemFree(b->dptr);
    if (b->owns_host && b->host) cuMemFreeHost(b->host);
    free(b);
}

// ---- Pipeline management (NVRTC) ---------------------------------------------
CudaPipe* cuda_create_pipeline(CudaCtx* c, const char* src, const char* fn_name,
                               const char* const* opts, uint32_t n_opts) {
    cuCtxSetCurrent(c->ctx);
    nvrtcProgram prog;
    if (nvrtcCreateProgram(&prog, src, "zinc_kernel.cu", 0, NULL, NULL) != NVRTC_SUCCESS) {
        set_err("nvrtcCreateProgram", "failed"); return NULL;
    }
    // Default arch = the running device's cc (e.g. sm_120 / sm_89).
    char arch[32];
    snprintf(arch, sizeof arch, "--gpu-architecture=sm_%u", cuda_compute_capability(c));
    // -I the CUDA toolkit headers so kernels may #include <cuda_fp16.h> / <mma.h>
    // (Blackwell tensor-core / fp16 prefill GEMMs). NVRTC ignores the flag for
    // kernels that don't include them, so this is additive for the existing set.
    const char* base_opts[3] = { arch, "--std=c++17", "-I/usr/local/cuda/include" };
    uint32_t total = 3 + n_opts;
    const char** all = (const char**)malloc(sizeof(char*) * total);
    all[0] = base_opts[0]; all[1] = base_opts[1]; all[2] = base_opts[2];
    for (uint32_t i = 0; i < n_opts; i++) all[3 + i] = opts[i];
    nvrtcResult cr = nvrtcCompileProgram(prog, (int)total, all);
    free(all);
    if (cr != NVRTC_SUCCESS) {
        size_t logsz = 0; nvrtcGetProgramLogSize(prog, &logsz);
        char* log = (char*)malloc(logsz + 1);
        if (log) { nvrtcGetProgramLog(prog, log); log[logsz] = 0; fprintf(stderr, "[cuda_shim] NVRTC log:\n%s\n", log); free(log); }
        set_err("nvrtcCompileProgram", nvrtcGetErrorString(cr));
        nvrtcDestroyProgram(&prog); return NULL;
    }
    // Use nvrtcGetCUBIN (direct machine code) instead of nvrtcGetPTX + driver JIT.
    // The driver JIT can generate incorrect SASS for mma.sync inline PTX on some
    // architectures. CUBIN is compiled by NVRTC's internal ptxas, which is correct.
    size_t cubinsz = 0;
    if (nvrtcGetCUBINSize(prog, &cubinsz) != NVRTC_SUCCESS || cubinsz == 0) {
        // Fallback to PTX if CUBIN not available
        size_t ptxsz = 0; nvrtcGetPTXSize(prog, &ptxsz);
        char* ptx = (char*)malloc(ptxsz);
        nvrtcGetPTX(prog, ptx);
        nvrtcDestroyProgram(&prog);
        CudaPipe* p = (CudaPipe*)calloc(1, sizeof *p);
        if (!p) { free(ptx); set_err("cuda_create_pipeline", "oom"); return NULL; }
        if (!cu_ok(cuModuleLoadData(&p->mod, ptx), "cuModuleLoadData")) { free(ptx); free(p); return NULL; }
        free(ptx);
        if (!cu_ok(cuModuleGetFunction(&p->fn, p->mod, fn_name), "cuModuleGetFunction")) {
            cuModuleUnload(p->mod); free(p); return NULL;
        }
        return p;
    }
    char* cubin = (char*)malloc(cubinsz);
    nvrtcResult cubin_ret = nvrtcGetCUBIN(prog, cubin);
    nvrtcDestroyProgram(&prog);
    if (cubin_ret != NVRTC_SUCCESS) { free(cubin); set_err("nvrtcGetCUBIN", nvrtcGetErrorString(cubin_ret)); return NULL; }

    CudaPipe* p = (CudaPipe*)calloc(1, sizeof *p);
    if (!p) { free(cubin); set_err("cuda_create_pipeline", "oom"); return NULL; }
    if (!cu_ok(cuModuleLoadData(&p->mod, cubin), "cuModuleLoadData(cubin)")) { free(cubin); free(p); return NULL; }
    free(cubin);
    if (!cu_ok(cuModuleGetFunction(&p->fn, p->mod, fn_name), "cuModuleGetFunction")) {
        cuModuleUnload(p->mod); free(p); return NULL;
    }
    return p;
}
CudaPipe* cuda_create_pipeline_from_image(CudaCtx* c, const void* image, size_t image_size, const char* fn_name) {
    (void)image_size;
    cuCtxSetCurrent(c->ctx);
    CudaPipe* p = (CudaPipe*)calloc(1, sizeof *p);
    if (!p) { set_err("cuda_create_pipeline_from_image", "oom"); return NULL; }
    if (!cu_ok(cuModuleLoadData(&p->mod, image), "cuModuleLoadData(image)")) { free(p); return NULL; }
    if (!cu_ok(cuModuleGetFunction(&p->fn, p->mod, fn_name), "cuModuleGetFunction")) {
        cuModuleUnload(p->mod); free(p); return NULL;
    }
    return p;
}
static int func_attr(CUfunction f, CUfunction_attribute a) { int v = 0; cuFuncGetAttribute(&v, a, f); return v; }
uint32_t cuda_pipeline_max_threads(CudaPipe* p) { return (uint32_t)func_attr(p->fn, CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK); }
uint32_t cuda_pipeline_shared_mem(CudaPipe* p) { return (uint32_t)func_attr(p->fn, CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES); }
void cuda_pipeline_set_max_dynamic_shared(CudaPipe* p, uint32_t bytes) {
    cuFuncSetAttribute(p->fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)bytes);
}
void cuda_free_pipeline(CudaPipe* p) { if (!p) return; if (p->mod) cuModuleUnload(p->mod); free(p); }

// ---- Command / dispatch ------------------------------------------------------
CudaCmd* cuda_begin_command(CudaCtx* c) {
    cuCtxSetCurrent(c->ctx);
    CudaCmd* m = (CudaCmd*)calloc(1, sizeof *m);
    if (!m) { set_err("cuda_begin_command", "oom"); return NULL; }
    m->stream = c->stream;
    if (!cu_ok(cuEventCreate(&m->event, CU_EVENT_DEFAULT), "cuEventCreate")) { free(m); return NULL; }
    return m;
}
void cuda_dispatch(CudaCmd* m, CudaPipe* p,
                   const uint32_t grid[3], const uint32_t block[3],
                   CudaBuf** bufs, uint32_t n_bufs,
                   const void* push_data, size_t push_size,
                   uint32_t shared_bytes) {
    (void)push_size;
    if (n_bufs > MAX_DISPATCH_BUFS) { set_err("cuda_dispatch", "too many buffers"); return; }
    // kernelParams[i] points to the i-th arg's value: &dptr for each buffer,
    // then the push struct bytes for the trailing by-value param.
    CUdeviceptr dptrs[MAX_DISPATCH_BUFS];
    void* args[MAX_DISPATCH_BUFS + 1];
    for (uint32_t i = 0; i < n_bufs; i++) { dptrs[i] = bufs[i]->dptr; args[i] = &dptrs[i]; }
    uint32_t nargs = n_bufs;
    if (push_data) { args[nargs++] = (void*)push_data; }
    cu_ok(cuLaunchKernel(p->fn, grid[0], grid[1], grid[2], block[0], block[1], block[2],
                         shared_bytes, m->stream, args, NULL), "cuLaunchKernel");
}
void cuda_barrier(CudaCmd* m) { (void)m; /* single stream is implicitly ordered */ }
void cuda_commit_and_wait(CudaCmd* m) {
    cuEventRecord(m->event, m->stream);
    cuStreamSynchronize(m->stream);
    cuEventDestroy(m->event); free(m);
}
void cuda_commit_async(CudaCmd* m) { cuEventRecord(m->event, m->stream); }
void cuda_wait(CudaCmd* m) { cuEventSynchronize(m->event); cuEventDestroy(m->event); free(m); }
void cuda_release_completed(CudaCmd* m) { if (!m) return; if (m->event) cuEventDestroy(m->event); free(m); }

// ---- CUDA Graphs (decode replay, Effort 25) ---------------------------------
// See cuda_shim.h. Capture the per-step kernel chain on the ctx stream and
// replay it as one graph launch — eliminating the per-kernel launch + inter-
// kernel-bubble latency that dominates the launch-bound decode regime (~10% GPU
// util). The exec is cached and updated in place across steps; on an update
// failure (e.g. an incompatible structural change) it is re-instantiated from
// the freshly captured graph, so the launched exec always reflects the current
// step's parameters and is bit-identical to the un-captured chain.
struct CudaGraph { CUgraphExec exec; int have_exec; };

int cuda_graph_supported(void) { return 1; }
CudaGraph* cuda_graph_create(void) {
    CudaGraph* g = (CudaGraph*)calloc(1, sizeof *g);
    if (!g) { set_err("cuda_graph_create", "oom"); return NULL; }
    return g;
}
int cuda_graph_begin(CudaCtx* c) {
    cuCtxSetCurrent(c->ctx);
    return cu_ok(cuStreamBeginCapture(c->stream, CU_STREAM_CAPTURE_MODE_RELAXED), "cuStreamBeginCapture");
}
int cuda_graph_end_launch(CudaCtx* c, CudaGraph* g) {
    CUgraph graph = NULL;
    if (!cu_ok(cuStreamEndCapture(c->stream, &graph), "cuStreamEndCapture")) return 0;
    if (g->have_exec) {
        // Topology is invariant across decode steps (same layers/kernels/order);
        // only per-token push-constant scalars change → an in-place exec update
        // is cheap. If the update is rejected, fall back to a re-instantiate.
        CUgraphExecUpdateResultInfo info;
        memset(&info, 0, sizeof info);
        if (cuGraphExecUpdate(g->exec, graph, &info) != CUDA_SUCCESS) {
            cuGraphExecDestroy(g->exec);
            g->have_exec = 0;
        }
    }
    if (!g->have_exec) {
        if (!cu_ok(cuGraphInstantiate(&g->exec, graph, 0), "cuGraphInstantiate")) {
            cuGraphDestroy(graph);
            return 0;
        }
        g->have_exec = 1;
    }
    cuGraphDestroy(graph);
    int ok = cu_ok(cuGraphLaunch(g->exec, c->stream), "cuGraphLaunch");
    cuStreamSynchronize(c->stream);
    return ok;
}
void cuda_graph_free(CudaGraph* g) {
    if (!g) return;
    if (g->have_exec) cuGraphExecDestroy(g->exec);
    free(g);
}
