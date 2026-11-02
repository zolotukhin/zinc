//! Shared CUDA/ROCm compute pipeline wrapper for native kernel compilation.
//! CUDA uses NVRTC with an `sm_XY` target; ROCm uses hipRTC with a `gfx*` target.
//! The selected C shim owns native modules and kernel handles.
//! @section CUDA Runtime
//! Each pipeline holds an opaque module/function pair obtained from the shim.
//! Use `createPipeline` for runtime compilation or `createPipelineFromImage` for
//! an image supported by the selected backend (cubin/PTX on CUDA). Free with
//! `freePipeline` when done.
const std = @import("std");
const shim = @import("c.zig").shim;

/// A compiled CUDA kernel ready for dispatch.
pub const CudaPipeline = struct {
    /// Opaque handle to the C shim pipeline object (CUmodule + CUfunction).
    handle: ?*shim.CudaPipe,
    /// Maximum threads per block the kernel supports.
    max_threads: u32,
    /// Bytes of static shared memory the kernel declares.
    shared_mem: u32,
    /// Optional debug label (typically the kernel name).
    name: ?[]const u8 = null,
};

/// NVRTC-compile `cu_source` and resolve `fn_name` for dispatch.
/// @param ctx  Active CUDA context; must not be null.
/// @param cu_source  Null-terminated CUDA C source string passed directly to NVRTC.
/// @param fn_name  Null-terminated name of the kernel function to extract from the compiled module.
/// @returns A `CudaPipeline` with populated `max_threads` and `shared_mem` fields, or `error.CudaPipelineCreateFailed`.
pub fn createPipeline(ctx: ?*shim.CudaCtx, cu_source: [*:0]const u8, fn_name: [*:0]const u8) !CudaPipeline {
    const handle = shim.cuda_create_pipeline(ctx, cu_source, fn_name, null, 0);
    if (handle == null) return error.CudaPipelineCreateFailed;
    return .{
        .handle = handle,
        .max_threads = shim.cuda_pipeline_max_threads(handle),
        .shared_mem = shim.cuda_pipeline_shared_mem(handle),
    };
}

/// Load a kernel from a precompiled cubin/PTX image (offline nvcc path).
/// @param ctx  Active CUDA context; must not be null.
/// @param image  Pointer to the raw cubin or PTX image bytes.
/// @param image_size  Byte length of `image`.
/// @param fn_name  Null-terminated name of the kernel function to locate in the loaded module.
/// @returns A `CudaPipeline` with populated `max_threads` and `shared_mem` fields, or `error.CudaPipelineCreateFailed`.
pub fn createPipelineFromImage(ctx: ?*shim.CudaCtx, image: [*]const u8, image_size: usize, fn_name: [*:0]const u8) !CudaPipeline {
    const handle = shim.cuda_create_pipeline_from_image(ctx, @ptrCast(image), image_size, fn_name);
    if (handle == null) return error.CudaPipelineCreateFailed;
    return .{
        .handle = handle,
        .max_threads = shim.cuda_pipeline_max_threads(handle),
        .shared_mem = shim.cuda_pipeline_shared_mem(handle),
    };
}

/// Opt this kernel into a larger dynamic shared-memory cap (Ada/Blackwell).
/// @param pipe  Pipeline whose dynamic shared-memory limit to raise.
/// @param bytes  New maximum dynamic shared memory per block in bytes.
/// @note No-op when `pipe.handle` is null. On Ada/Blackwell this lifts the
///       default 48 KB barrier by calling `cuFuncSetAttribute` on the shim side.
pub fn setMaxDynamicShared(pipe: *CudaPipeline, bytes: u32) void {
    if (pipe.handle) |h| shim.cuda_pipeline_set_max_dynamic_shared(h, bytes);
}

/// Release the pipeline handle. Safe to call with a null handle.
pub fn freePipeline(pipe: *CudaPipeline) void {
    if (pipe.handle) |h| {
        shim.cuda_free_pipeline(h);
        pipe.handle = null;
    }
}
