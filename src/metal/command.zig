//! Metal command buffer wrapper — dispatch recording and GPU synchronization.
//!
//! This module is the low-level bridge between ZINC's compute dispatchers and
//! the Objective-C Metal shim. It records compute work, barriers, and timing
//! mode so higher-level inference code can stay backend-agnostic.
//! @section Metal Runtime
const std = @import("std");
const shim = @import("c.zig").shim;
const MetalBuffer = @import("buffer.zig").MetalBuffer;
const MetalPipeline = @import("pipeline.zig").MetalPipeline;

/// Encoder policy used when opening a Metal compute command buffer.
pub const CommandEncoderMode = enum(u8) {
    concurrent = 0,
    serial = 1,
};

/// A recorded command buffer that encodes compute dispatches for the GPU.
pub const MetalCommand = struct {
    /// Opaque handle to the C shim command buffer.
    handle: ?*shim.MetalCmd,
    dispatch_count: u32,
    barrier_count: u32,
    barrier_enabled: bool,

    /// Encode a compute dispatch binding buffers, push constants, grid, and block sizes.
    pub fn dispatch(
        self: *MetalCommand,
        pipe: *const MetalPipeline,
        grid: [3]u32,
        block: [3]u32,
        bufs: []const *const MetalBuffer,
        push_data: ?*const anyopaque,
        push_size: usize,
    ) void {
        if (self.handle == null or pipe.handle == null) return;
        self.dispatch_count += 1;

        // Build array of MetalBuf pointers for the C shim
        var c_bufs: [32]?*shim.MetalBuf = .{null} ** 32;
        const n_bufs: u32 = @intCast(@min(bufs.len, 32));
        for (bufs[0..n_bufs], 0..n_bufs) |b, i| {
            c_bufs[i] = b.handle;
        }

        shim.mtl_dispatch(
            self.handle,
            pipe.handle,
            &grid,
            &block,
            @ptrCast(&c_bufs),
            n_bufs,
            push_data,
            push_size,
        );
    }

    /// Dispatch with explicit push constant buffer index.
    /// SPIRV-Cross compiled shaders place push constants at a specific buffer index
    /// (often 0 or 1). Data buffers in `bufs` are bound at all other indices in order,
    /// skipping push_idx.
    pub fn dispatchV2(
        self: *MetalCommand,
        pipe: *const MetalPipeline,
        grid: [3]u32,
        block: [3]u32,
        bufs: []const *const MetalBuffer,
        push_data: ?*const anyopaque,
        push_size: usize,
        push_idx: u32,
    ) void {
        if (self.handle == null or pipe.handle == null) return;
        self.dispatch_count += 1;

        var c_bufs: [32]?*shim.MetalBuf = .{null} ** 32;
        const n_bufs: u32 = @intCast(@min(bufs.len, 32));
        for (bufs[0..n_bufs], 0..n_bufs) |b, i| {
            c_bufs[i] = b.handle;
        }

        shim.mtl_dispatch_v2(
            self.handle,
            pipe.handle,
            &grid,
            &block,
            @ptrCast(&c_bufs),
            n_bufs,
            push_data,
            push_size,
            push_idx,
        );
    }

    /// Dispatch with explicit threadgroup memory allocation.
    pub fn dispatchV2WithTgMem(
        self: *MetalCommand,
        pipe: *const MetalPipeline,
        grid: [3]u32,
        block: [3]u32,
        bufs: []const *const MetalBuffer,
        push_data: ?*const anyopaque,
        push_size: usize,
        push_idx: u32,
        tg_mem_size: u32,
    ) void {
        if (self.handle == null or pipe.handle == null) return;

        var c_bufs: [32]?*shim.MetalBuf = .{null} ** 32;
        const n_bufs: u32 = @intCast(@min(bufs.len, 32));
        for (bufs[0..n_bufs], 0..n_bufs) |b, i| {
            c_bufs[i] = b.handle;
        }

        shim.mtl_dispatch_v2_tgmem(
            self.handle,
            pipe.handle,
            &grid,
            &block,
            @ptrCast(&c_bufs),
            n_bufs,
            push_data,
            push_size,
            push_idx,
            tg_mem_size,
        );
    }

    /// Insert a memory barrier ensuring all prior dispatches complete before subsequent ones.
    pub fn barrier(self: *MetalCommand) void {
        if (!self.barrier_enabled) return;
        if (self.handle) |h| {
            self.barrier_count += 1;
            shim.mtl_barrier(h);
        }
    }

    /// Commit the command buffer to the GPU and block until execution completes.
    pub fn commitAndWait(self: *MetalCommand) void {
        if (self.handle) |h| {
            shim.mtl_commit_and_wait(h);
            self.handle = null;
        }
    }

    /// Commit the command buffer for async GPU execution; call `wait` later to synchronize.
    pub fn commitAsync(self: *MetalCommand) void {
        if (self.handle) |h| {
            shim.mtl_commit_async(h);
            // handle stays valid — call wait() later
        }
    }

    /// Block until an async-committed command buffer finishes execution.
    pub fn wait(self: *MetalCommand) void {
        if (self.handle) |h| {
            shim.mtl_wait(h);
            self.handle = null;
        }
    }
};

/// Allocate a new command buffer from the given Metal context.
pub fn beginCommand(ctx: ?*shim.MetalCtx) !MetalCommand {
    return beginCommandWithMode(ctx, .concurrent);
}

/// Allocate a command buffer using the requested encoder/barrier policy.
pub fn beginCommandWithMode(ctx: ?*shim.MetalCtx, mode: CommandEncoderMode) !MetalCommand {
    const handle = shim.mtl_begin_command_mode(ctx, @intFromEnum(mode));
    if (handle == null) return error.MetalCommandBufferFailed;
    return .{
        .handle = handle,
        .dispatch_count = 0,
        .barrier_count = 0,
        .barrier_enabled = mode == .concurrent,
    };
}

const buffer_mod = @import("buffer.zig");
const pipeline_mod = @import("pipeline.zig");

test "beginCommand and commitAndWait" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var cmd = try beginCommand(ctx);
    // Commit immediately with no dispatches — should succeed
    cmd.commitAndWait();
    try std.testing.expect(cmd.handle == null); // handle cleared after commit
}

test "full GPU dispatch: write indices via compute shader" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    // Create output buffer (256 floats = 1024 bytes)
    const n: u32 = 256;
    var out_buf = try buffer_mod.createBuffer(ctx, n * @sizeOf(f32));
    defer buffer_mod.freeBuffer(&out_buf);

    // Zero the buffer
    @memset(out_buf.cpu_ptr.?[0 .. n * @sizeOf(f32)], 0);

    // Compile a kernel that writes thread_id to each element
    const msl =
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void main0(device float* out [[buffer(0)]],
        \\                   uint id [[thread_position_in_grid]]) {
        \\    out[id] = float(id) * 2.0 + 1.0;
        \\}
    ;
    var pipe = try pipeline_mod.createPipeline(ctx, msl, "main0");
    defer pipeline_mod.freePipeline(&pipe);

    // Dispatch
    var cmd = try beginCommand(ctx);
    const bufs = [_]*const MetalBuffer{&out_buf};
    cmd.dispatch(&pipe, .{ n / 32, 1, 1 }, .{ 32, 1, 1 }, &bufs, null, 0);
    cmd.commitAndWait();

    // Read back and verify
    const result: [*]const f32 = @ptrCast(@alignCast(out_buf.cpu_ptr.?));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 0.001); // 0*2+1
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[1], 0.001); // 1*2+1
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), result[2], 0.001); // 2*2+1
    try std.testing.expectApproxEqAbs(@as(f32, 511.0), result[255], 0.001); // 255*2+1
}

test "GPU dispatch with push constants" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    const n: u32 = 64;
    var out_buf = try buffer_mod.createBuffer(ctx, n * @sizeOf(f32));
    defer buffer_mod.freeBuffer(&out_buf);
    @memset(out_buf.cpu_ptr.?[0 .. n * @sizeOf(f32)], 0);

    // Kernel that reads a scale factor from push constants (buffer index 1)
    const msl =
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\struct Params { uint n; float scale; };
        \\kernel void main0(device float* out [[buffer(0)]],
        \\                   constant Params& params [[buffer(1)]],
        \\                   uint id [[thread_position_in_grid]]) {
        \\    if (id < params.n) {
        \\        out[id] = float(id) * params.scale;
        \\    }
        \\}
    ;
    var pipe = try pipeline_mod.createPipeline(ctx, msl, "main0");
    defer pipeline_mod.freePipeline(&pipe);

    const Params = extern struct { n: u32, scale: f32 };
    const params = Params{ .n = n, .scale = 3.0 };

    var cmd = try beginCommand(ctx);
    const bufs = [_]*const MetalBuffer{&out_buf};
    cmd.dispatch(&pipe, .{ 2, 1, 1 }, .{ 32, 1, 1 }, &bufs, &params, @sizeOf(Params));
    cmd.commitAndWait();

    const result: [*]const f32 = @ptrCast(@alignCast(out_buf.cpu_ptr.?));
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 0.001); // 0*3
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), result[1], 0.001); // 1*3
    try std.testing.expectApproxEqAbs(@as(f32, 30.0), result[10], 0.001); // 10*3
    try std.testing.expectApproxEqAbs(@as(f32, 189.0), result[63], 0.001); // 63*3
}

test "barrier separates dispatches" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    const n: u32 = 64;
    var buf = try buffer_mod.createBuffer(ctx, n * @sizeOf(f32));
    defer buffer_mod.freeBuffer(&buf);
    @memset(buf.cpu_ptr.?[0 .. n * @sizeOf(f32)], 0);

    // First kernel: write id to buffer
    const msl_write =
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void main0(device float* data [[buffer(0)]],
        \\                   uint id [[thread_position_in_grid]]) {
        \\    data[id] = float(id);
        \\}
    ;
    // Second kernel: double each element
    const msl_double =
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void main0(device float* data [[buffer(0)]],
        \\                   uint id [[thread_position_in_grid]]) {
        \\    data[id] = data[id] * 2.0;
        \\}
    ;

    var pipe_write = try pipeline_mod.createPipeline(ctx, msl_write, "main0");
    defer pipeline_mod.freePipeline(&pipe_write);
    var pipe_double = try pipeline_mod.createPipeline(ctx, msl_double, "main0");
    defer pipeline_mod.freePipeline(&pipe_double);

    var cmd = try beginCommand(ctx);
    const bufs = [_]*const MetalBuffer{&buf};

    // Write, barrier, then double
    cmd.dispatch(&pipe_write, .{ 2, 1, 1 }, .{ 32, 1, 1 }, &bufs, null, 0);
    cmd.barrier();
    cmd.dispatch(&pipe_double, .{ 2, 1, 1 }, .{ 32, 1, 1 }, &bufs, null, 0);
    cmd.commitAndWait();

    const result: [*]const f32 = @ptrCast(@alignCast(buf.cpu_ptr.?));
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), result[0], 0.001); // 0*2
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[1], 0.001); // 1*2
    try std.testing.expectApproxEqAbs(@as(f32, 20.0), result[10], 0.001); // 10*2
    try std.testing.expectApproxEqAbs(@as(f32, 126.0), result[63], 0.001); // 63*2
}

test "concurrent encoder with barrier preserves buffer write-read chains" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    const n: u32 = 64;
    var buf = try buffer_mod.createBuffer(ctx, n * @sizeOf(f32));
    defer buffer_mod.freeBuffer(&buf);
    @memset(buf.cpu_ptr.?[0 .. n * @sizeOf(f32)], 0);

    const msl_write =
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void main0(device float* data [[buffer(0)]],
        \\                   uint id [[thread_position_in_grid]]) {
        \\    data[id] = float(id);
        \\}
    ;
    const msl_add =
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void main0(device float* data [[buffer(0)]],
        \\                   uint id [[thread_position_in_grid]]) {
        \\    data[id] = data[id] + 1.0;
        \\}
    ;

    var pipe_write = try pipeline_mod.createPipeline(ctx, msl_write, "main0");
    defer pipeline_mod.freePipeline(&pipe_write);
    var pipe_add = try pipeline_mod.createPipeline(ctx, msl_add, "main0");
    defer pipeline_mod.freePipeline(&pipe_add);

    var cmd = try beginCommand(ctx);
    const bufs = [_]*const MetalBuffer{&buf};

    cmd.dispatch(&pipe_write, .{ 2, 1, 1 }, .{ 32, 1, 1 }, &bufs, null, 0);
    cmd.barrier(); // concurrent encoder requires explicit barrier for write-then-read
    cmd.dispatch(&pipe_add, .{ 2, 1, 1 }, .{ 32, 1, 1 }, &bufs, null, 0);
    cmd.commitAndWait();

    const result: [*]const f32 = @ptrCast(@alignCast(buf.cpu_ptr.?));
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 2.0), result[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 11.0), result[10], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 64.0), result[63], 0.001);
}
