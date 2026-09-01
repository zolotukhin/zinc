#ifndef ZINC_CUDA_SHIM_H
#define ZINC_CUDA_SHIM_H

// C ABI for the ZINC CUDA backend — the boundary between Zig and the CUDA
// Driver/Runtime + NVRTC. Mirrors src/metal/shim.h so the compute layer can
// drive Metal and CUDA through structurally identical surfaces.
//
// Memory model: unlike Metal (Apple unified memory), CUDA device memory is NOT
// host-visible. Weights/activations live in device buffers; host<->device moves
// are explicit (cuda_upload / cuda_download), staged through pinned host memory.
//
// Dispatch ABI: every kernel is authored as
//     __global__ void name(P0* buf0, P1* buf1, ..., Push pc)
// i.e. the bound buffers come first (as device pointers, in bind order) and a
// single trailing push-constant struct passed by value. cuda_dispatch builds the
// cuLaunchKernel argument vector as [&dptr0, &dptr1, ..., push_data] accordingly.
//
// Command/stream model: a CudaCmd wraps one CUstream plus a per-command CUevent.
// Work is enqueued on the stream (async by default); the event marks completion.
//   - cuda_commit_and_wait : record event, block until the stream drains.
//   - cuda_commit_async    : record event, return immediately (GPU runs while the
//                            CPU builds the next command — the overlap the Metal
//                            backend gets from commitAsync).
//   - cuda_wait            : block on a previously async-committed command's event.
//   - cuda_release_completed: free a command already known complete (queue-ordered).

#include <stdint.h>
#include <stddef.h>

// Opaque handles
typedef struct CudaCtx CudaCtx;     // device + context + default compute stream
typedef struct CudaBuf CudaBuf;     // device allocation (+ optional pinned host mirror)
typedef struct CudaPipe CudaPipe;   // compiled CUfunction (+ its CUmodule)
typedef struct CudaCmd CudaCmd;     // stream batch + completion event
typedef struct CudaGraph CudaGraph; // cached captured per-step kernel chain (CUgraphExec)

// ---- Device lifecycle --------------------------------------------------------
// device_index selects the CUDA device (PCI-bus order is the caller's concern;
// pass the resolved index). Returns NULL on failure.
CudaCtx* cuda_init(int device_index);
void     cuda_destroy(CudaCtx* ctx);
uint64_t cuda_total_memory(CudaCtx* ctx);          // total VRAM, bytes
uint64_t cuda_free_memory(CudaCtx* ctx);           // currently free VRAM, bytes
uint32_t cuda_sm_count(CudaCtx* ctx);              // multiprocessor count
uint32_t cuda_compute_capability(CudaCtx* ctx);    // major*10 + minor (e.g. 120, 89)
uint32_t cuda_max_threads_per_block(CudaCtx* ctx);
uint32_t cuda_max_shared_mem_per_block(CudaCtx* ctx);
uint32_t cuda_warp_size(CudaCtx* ctx);             // 32 on all NVIDIA
// Fills name_out (NUL-terminated, up to cap bytes) with the device name.
void     cuda_device_name(CudaCtx* ctx, char* name_out, size_t cap);
// Fills arch_out with the native compilation target (for example sm_120 or
// gfx1201). This is diagnostic metadata; callers must not infer capabilities
// from the string.
void     cuda_device_arch(CudaCtx* ctx, char* arch_out, size_t cap);

// ---- Buffer management -------------------------------------------------------
// Device-local buffer (the common case for weights/activations/state).
CudaBuf* cuda_create_buffer(CudaCtx* ctx, size_t size);
// Device buffer with a paired pinned-host staging mirror; *cpu_ptr receives the
// host pointer for fast cuda_upload/cuda_download (cudaHostAlloc).
CudaBuf* cuda_create_buffer_staged(CudaCtx* ctx, size_t size, void** cpu_ptr);
// Register an existing host mapping (e.g. mmap'd weights) as pinned and copy to
// device — the CUDA analogue of mtl_wrap_mmap (which is zero-copy on Apple).
CudaBuf* cuda_upload_mmap(CudaCtx* ctx, const void* host_ptr, size_t size);
// Sub-buffer view sharing the parent's device allocation (no new alloc).
CudaBuf* cuda_alias_buffer(CudaBuf* base, size_t offset, size_t size);
// Raw device pointer (CUdeviceptr) as an integer, for arg packing / aliasing.
uint64_t cuda_buffer_device_ptr(CudaBuf* buf);
// Explicit transfers (synchronous on the ctx stream).
void     cuda_upload(CudaCtx* ctx, CudaBuf* buf, const void* src, size_t size);
void     cuda_download(CudaCtx* ctx, CudaBuf* buf, void* dst, size_t size);
// Async (no-sync) transfers on the ctx stream — capturable into a CUDA graph so
// the per-step embed H2D and argmax D2H fold into the single graph launch
// instead of costing a separate stream sync each. Use PINNED host memory
// (cuda_alloc_host) for the host side so the captured copy is truly async.
void     cuda_upload_async(CudaCtx* ctx, CudaBuf* buf, const void* src, size_t size);
void     cuda_download_async(CudaCtx* ctx, CudaBuf* buf, void* dst, size_t size);
// Pinned (page-locked) host allocation, required for async graph-captured copies.
void*    cuda_alloc_host(size_t size);
void     cuda_free_host(void* ptr);
void     cuda_free_buffer(CudaBuf* buf);

// ---- Pipeline management -----------------------------------------------------
// NVRTC-compile `cu_source` for the running device's arch and resolve `fn_name`.
// `opts`/`n_opts` are extra NVRTC options (may be NULL/0). Returns NULL on
// compile/link failure (compile log goes to stderr).
CudaPipe* cuda_create_pipeline(CudaCtx* ctx, const char* cu_source, const char* fn_name,
                               const char* const* opts, uint32_t n_opts);
// Load a precompiled cubin/PTX module and resolve `fn_name` (offline nvcc path).
CudaPipe* cuda_create_pipeline_from_image(CudaCtx* ctx, const void* image, size_t image_size,
                                          const char* fn_name);
uint32_t  cuda_pipeline_max_threads(CudaPipe* pipe);
uint32_t  cuda_pipeline_shared_mem(CudaPipe* pipe);        // static shared bytes
// Opt-in to a larger dynamic shared-memory cap for this kernel (Blackwell/Ada).
void      cuda_pipeline_set_max_dynamic_shared(CudaPipe* pipe, uint32_t bytes);
void      cuda_free_pipeline(CudaPipe* pipe);

// ---- Command / dispatch ------------------------------------------------------
CudaCmd* cuda_begin_command(CudaCtx* ctx);
// grid/block are [x,y,z]. `bufs` are bound in order as the leading kernel
// pointer args; `push_data` (push_size bytes) is the trailing by-value arg.
// `shared_bytes` is dynamic shared memory for this launch (0 if none).
void cuda_dispatch(CudaCmd* cmd, CudaPipe* pipe,
                   const uint32_t grid[3], const uint32_t block[3],
                   CudaBuf** bufs, uint32_t n_bufs,
                   const void* push_data, size_t push_size,
                   uint32_t shared_bytes);
// Same-stream launches are implicitly ordered, so this is a no-op for a single
// stream; kept for surface parity and future multi-stream (event) use.
void cuda_barrier(CudaCmd* cmd);
void cuda_commit_and_wait(CudaCmd* cmd);
void cuda_commit_async(CudaCmd* cmd);
void cuda_wait(CudaCmd* cmd);
void cuda_release_completed(CudaCmd* cmd);

// ---- cuBLAS prefill GEMM (Effort 26 cycle 9) ---------------------------------
// Y[N,M] (token-major, row-major [T=N, M]) = A[N,K]·W[M,K]^T, fp16 inputs, fp32
// accumulate. W and A are fp16 device buffers (W dequant'd from Q4_K, A downcast
// from f32 by the caller's kernels on the SAME ctx stream — cuBLAS runs on that
// stream so it is correctly ordered after them). Y is an fp32 device buffer.
// `beta` is the cublas accumulate factor on Y (0 = overwrite, used by prefill;
// 1 = accumulate the residual in place, used by the B>27 serving decode GEMMs
// whose acc_mode==1 weights — O-proj / FFN-down / SSM-out — fold into Y).
void cuda_cublas_hgemm(CudaCtx* ctx, unsigned M, unsigned N, unsigned K,
                       CudaBuf* W, CudaBuf* A, CudaBuf* Y, float beta);

// The same operation over equally-strided matrix views. Strides are expressed
// in elements (not bytes), matching cuBLAS/hipBLAS. Muse uses batch=2 for its
// two GQA KV heads, cutting attention from four BLAS submissions to two.
void cuda_cublas_hgemm_strided_batched(CudaCtx* ctx, unsigned trans_a, unsigned trans_b,
                                       unsigned M, unsigned N, unsigned K,
                                       CudaBuf* A, size_t a_offset, unsigned lda, int64_t stride_a,
                                       CudaBuf* B, size_t b_offset, unsigned ldb, int64_t stride_b,
                                       CudaBuf* C, size_t c_offset, unsigned ldc, int64_t stride_c,
                                       unsigned batch_count, float beta);

void cuda_cublas_sgemm_strided_batched(CudaCtx* ctx, unsigned trans_a, unsigned trans_b,
                                       unsigned M, unsigned N, unsigned K,
                                       CudaBuf* A, size_t a_offset, unsigned lda, int64_t stride_a,
                                       CudaBuf* B, size_t b_offset, unsigned ldb, int64_t stride_b,
                                       CudaBuf* C, size_t c_offset, unsigned ldc, int64_t stride_c,
                                       unsigned batch_count, float beta);

// ---- CUDA Graphs (decode replay, Effort 25) ----------------------------------
// Capture the per-decode-step kernel chain once and replay it as a SINGLE graph
// launch, collapsing the ~480 per-kernel launches + inter-kernel GPU bubbles of
// a launch-bound step into one submission. Per-step usage:
//     cuda_graph_begin(ctx);            // start capturing the ctx stream
//     ... issue the normal cuda_dispatch chain (captured, NOT executed) ...
//     cuda_graph_end_launch(ctx, g);    // end capture, instantiate-or-update the
//                                        // cached exec, launch it, sync the stream
// The exec is cached in `g` and updated in place across steps (per-token params —
// position, seq_len, KV offset — change but topology is invariant); on an update
// failure it is re-instantiated from the freshly captured graph, so the launched
// exec always matches the current step's parameters (bit-identical to the
// un-captured chain). MUST NOT be used around a chain that synchronizes or reads
// back mid-capture (e.g. the MoE router host readback).
int        cuda_graph_supported(void);                    // 1 when capture/replay is implemented
CudaGraph* cuda_graph_create(void);
int        cuda_graph_begin(CudaCtx* ctx);                 // 1 on success, 0 on failure
int        cuda_graph_end_launch(CudaCtx* ctx, CudaGraph* graph); // 1 on success
void       cuda_graph_free(CudaGraph* graph);

// ---- Diagnostics -------------------------------------------------------------
// Last CUDA/NVRTC error string for the calling thread ("" if none). Not freed.
const char* cuda_last_error(void);

#endif // ZINC_CUDA_SHIM_H
