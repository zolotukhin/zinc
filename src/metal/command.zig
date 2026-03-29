//! Metal command buffer wrapper — dispatch recording and GPU synchronization.
const std = @import("std");
const shim = @import("c.zig").shim;
const MetalBuffer = @import("buffer.zig").MetalBuffer;
const MetalPipeline = @import("pipeline.zig").MetalPipeline;

pub const MetalCommand = struct {
    handle: ?*shim.MetalCmd,

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

    pub fn barrier(self: *MetalCommand) void {
        if (self.handle) |h| shim.mtl_barrier(h);
    }

    pub fn commitAndWait(self: *MetalCommand) void {
        if (self.handle) |h| {
            shim.mtl_commit_and_wait(h);
            self.handle = null;
        }
    }

    pub fn commitAsync(self: *MetalCommand) void {
        if (self.handle) |h| {
            shim.mtl_commit_async(h);
            // handle stays valid — call wait() later
        }
    }

    pub fn wait(self: *MetalCommand) void {
        if (self.handle) |h| {
            shim.mtl_wait(h);
            self.handle = null;
        }
    }
};

pub fn beginCommand(ctx: ?*shim.MetalCtx) !MetalCommand {
    const handle = shim.mtl_begin_command(ctx);
    if (handle == null) return error.MetalCommandBufferFailed;
    return .{ .handle = handle };
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
