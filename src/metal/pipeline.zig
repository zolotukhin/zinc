//! Metal compute pipeline wrapper — MSL source or precompiled metallib.
//!
//! It compiles or loads Metal kernels, exposes the threadgroup capabilities
//! that dispatch code needs for tuning, and keeps pipeline lifecycle handling
//! out of the higher-level runtime and benchmark paths.
//! @section Metal Runtime
const std = @import("std");
const shim = @import("c.zig").shim;

/// A compiled Metal compute pipeline state ready for dispatch.
pub const MetalPipeline = struct {
    /// Opaque handle to the C shim pipeline object.
    handle: ?*shim.MetalPipe,
    /// Maximum threads the pipeline supports per threadgroup.
    max_threads_per_threadgroup: u32,
    /// SIMD execution width (warp size) for this pipeline.
    thread_execution_width: u32,
    /// Bytes of threadgroup memory statically allocated by the kernel.
    static_threadgroup_memory_length: u32,
};

/// Compile an MSL source string into a compute pipeline for the given function name.
pub fn createPipeline(ctx: ?*shim.MetalCtx, msl_source: [*:0]const u8, fn_name: [*:0]const u8) !MetalPipeline {
    const handle = shim.mtl_create_pipeline(ctx, msl_source, fn_name);
    if (handle == null) return error.MetalPipelineCreateFailed;
    return .{
        .handle = handle,
        .max_threads_per_threadgroup = shim.mtl_pipeline_max_threads(handle),
        .thread_execution_width = shim.mtl_pipeline_thread_execution_width(handle),
        .static_threadgroup_memory_length = shim.mtl_pipeline_static_threadgroup_memory_length(handle),
    };
}

/// Create a compute pipeline from a precompiled metallib binary blob.
pub fn createPipelineFromLib(ctx: ?*shim.MetalCtx, lib_data: [*]const u8, lib_size: usize, fn_name: [*:0]const u8) !MetalPipeline {
    const handle = shim.mtl_create_pipeline_from_lib(ctx, lib_data, lib_size, fn_name);
    if (handle == null) return error.MetalPipelineCreateFailed;
    return .{
        .handle = handle,
        .max_threads_per_threadgroup = shim.mtl_pipeline_max_threads(handle),
        .thread_execution_width = shim.mtl_pipeline_thread_execution_width(handle),
        .static_threadgroup_memory_length = shim.mtl_pipeline_static_threadgroup_memory_length(handle),
    };
}

/// Release the pipeline handle. Safe to call with a null handle.
pub fn freePipeline(pipe: *MetalPipeline) void {
    if (pipe.handle) |h| {
        shim.mtl_free_pipeline(h);
        pipe.handle = null;
    }
}

test "MetalPipeline struct size" {
    try std.testing.expect(@sizeOf(MetalPipeline) <= 24);
}

test "createPipeline compiles simple MSL kernel" {
    const device_shim = @import("c.zig").shim;
    const ctx = device_shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer device_shim.mtl_destroy(ctx);

    const msl_source =
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void main0(device float* out [[buffer(0)]],
        \\                   uint id [[thread_position_in_grid]]) {
        \\    out[id] = float(id);
        \\}
    ;

    var pipe = try createPipeline(ctx, msl_source, "main0");
    defer freePipeline(&pipe);

    try std.testing.expect(pipe.handle != null);
    try std.testing.expect(pipe.max_threads_per_threadgroup > 0);
    try std.testing.expect(pipe.thread_execution_width > 0);
    try std.testing.expect(pipe.max_threads_per_threadgroup >= pipe.thread_execution_width);
    try std.testing.expect(pipe.max_threads_per_threadgroup >= 32); // Apple GPU minimum
}

test "createPipeline fails on invalid MSL" {
    const device_shim = @import("c.zig").shim;
    const ctx = device_shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer device_shim.mtl_destroy(ctx);

    try std.testing.expectError(
        error.MetalPipelineCreateFailed,
        createPipeline(ctx, "this is not valid MSL", "main0"),
    );
}

test "createPipeline fails on wrong function name" {
    const device_shim = @import("c.zig").shim;
    const ctx = device_shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer device_shim.mtl_destroy(ctx);

    const msl_source =
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void my_kernel(device float* out [[buffer(0)]],
        \\                      uint id [[thread_position_in_grid]]) {
        \\    out[id] = 1.0;
        \\}
    ;

    try std.testing.expectError(
        error.MetalPipelineCreateFailed,
        createPipeline(ctx, msl_source, "nonexistent_function"),
    );
}

test "freePipeline with null handle is safe" {
    var pipe = MetalPipeline{
        .handle = null,
        .max_threads_per_threadgroup = 0,
        .thread_execution_width = 0,
        .static_threadgroup_memory_length = 0,
    };
    freePipeline(&pipe); // should not crash
}
