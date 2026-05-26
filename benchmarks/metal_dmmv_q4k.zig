// Microbench for the Q4_K decode matvec kernel (`dmmv_q4k.metal`).
//
// Decode-shape diagnostic for Effort 14. Measures isolated per-kernel
// throughput on the Qwen3-8B hot shapes (N=1), so we can tell whether
// the 8.6 tok/s end-to-end decode is matvec-bound or dispatch-bound.
//
// If a single matvec runs near memory-bandwidth-bound here but the
// end-to-end decode does not, the lever is the dispatch/barrier path,
// not the matvec kernel.

const std = @import("std");

fn nanoTimestamp() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}
const support = @import("zinc_bench_support");
const metal_device = support.metal_device;
const metal_command = support.metal_command;
const metal_pipeline = support.metal_pipeline;
const metal_buffer = support.metal_buffer;
const process_lock = support.process_lock;
const shim = support.metal_c.shim;

const MetalBuffer = metal_buffer.MetalBuffer;
const MetalPipeline = metal_pipeline.MetalPipeline;

pub const std_options = std.Options{ .log_level = .warn };

// Mirrors `DmmvPush` in src/compute/forward_metal.zig (buffer 1 of
// dmmv_q4k.metal). Push slot 1 in the dispatch — push slot 0 is the
// "kernel meta" slot that some pipelines use; dmmv_q4k takes the push
// at buffer(1) per its kernel signature.
const DmmvPush = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    x_offset: u32,
    y_offset: u32,
};

fn loadShaderPipeline(io: std.Io, ctx: ?*shim.MetalCtx, name: []const u8) !MetalPipeline {
    var path_buf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "src/shaders/metal/{s}.metal", .{name}) catch return error.PathTooLong;
    const file = std.Io.Dir.cwd().openFile(io, path, .{}) catch return error.ShaderNotFound;
    defer file.close();
    const stat = try file.stat();
    if (stat.size > 1024 * 1024) return error.ShaderTooLarge;
    var source_buf: [1024 * 1024 + 1]u8 = undefined;
    const bytes_read = try file.readAll(source_buf[0 .. source_buf.len - 1]);
    source_buf[bytes_read] = 0;
    var fn_buf: [16]u8 = undefined;
    const fn_name = try std.fmt.bufPrintZ(&fn_buf, "main0", .{});
    return metal_pipeline.createPipeline(ctx, @ptrCast(&source_buf), fn_name);
}

// Fill Q4_K buffer with deterministic pseudo-random bytes + small scales
// so dequantized values stay in a reasonable range. Output correctness is
// not measured here — only kernel throughput.
fn fillQ4K(buf: *MetalBuffer, num_blocks: usize) void {
    const ptr: [*]u8 = @ptrCast(buf.cpu_ptr.?);
    var rng = std.Random.DefaultPrng.init(42);
    const rand = rng.random();
    const bytes = num_blocks * 144;
    for (0..bytes) |i| ptr[i] = rand.int(u8);
    var i: usize = 0;
    while (i < bytes) : (i += 144) {
        const d: f16 = 0.0078;
        const dmin: f16 = 0.0039;
        std.mem.writeInt(u16, ptr[i..][0..2], @bitCast(d), .little);
        std.mem.writeInt(u16, ptr[i + 2 ..][0..2], @bitCast(dmin), .little);
    }
}

fn fillF32(buf: *MetalBuffer, n: usize) void {
    const ptr: [*]f32 = @ptrCast(@alignCast(buf.cpu_ptr.?));
    var rng = std.Random.DefaultPrng.init(1);
    const rand = rng.random();
    for (0..n) |j| ptr[j] = (rand.float(f32) - 0.5) * 2.0;
}

const BenchCase = struct {
    label: []const u8,
    M: u32,
    K: u32,
};

// Time per kernel dispatch, in nanoseconds. The kernel writes to a
// host-visible buffer Y of length M floats. ROWS_PER_WG = 4 matches the
// runtime dispatch (see forward_metal.zig:3497).
const ROWS_PER_WG: u32 = 4;
const BLOCK_SIZE: u32 = 64;

fn benchShape(
    ctx: ?*shim.MetalCtx,
    pipe: *const MetalPipeline,
    w_buf: *const MetalBuffer,
    x_buf: *const MetalBuffer,
    y_buf: *const MetalBuffer,
    M: u32,
    K: u32,
    warmup: u32,
    iters: u32,
) !f64 {
    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ w_buf, x_buf, y_buf };
    const grid_x = (M + ROWS_PER_WG - 1) / ROWS_PER_WG;
    const grid = [_]u32{ grid_x, 1, 1 };
    const block = [_]u32{ BLOCK_SIZE, 1, 1 };

    // Warmup runs through commitAndWait (full GPU-side warm-up + driver
    // pipeline ramp).
    for (0..warmup) |_| {
        var cmd = try metal_command.beginCommand(ctx);
        cmd.dispatchV2(pipe, grid, block, &bufs, &push, @sizeOf(DmmvPush), 1);
        cmd.commitAndWait();
    }

    // Timed: enqueue all iters into a single command buffer so we measure
    // per-kernel GPU time amortized over the iterations, NOT one
    // commitAndWait per iter. This isolates kernel time from CPU
    // command-buffer overhead.
    const start = nanoTimestamp();
    var cmd = try metal_command.beginCommand(ctx);
    for (0..iters) |_| {
        cmd.dispatchV2(pipe, grid, block, &bufs, &push, @sizeOf(DmmvPush), 1);
    }
    cmd.commitAndWait();
    const end = nanoTimestamp();

    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iters));
}

pub fn main(init: std.process.Init) !void {
    const io = init.io;
    const allocator = init.gpa;

    var device = try metal_device.MetalDevice.init(allocator, 0);
    defer device.deinit();

    var gpu_lock = process_lock.acquire(.metal, device.selected_device_index) catch |err| {
        support.reportGpuProcessLockError(err, .metal, device.selected_device_index);
    };
    defer gpu_lock.deinit();

    var pipe = try loadShaderPipeline(io, device.ctx, "dmmv_q4k");
    defer metal_pipeline.freePipeline(&pipe);

    // Qwen3-8B Q4_K_M decode shapes (N=1).
    // hidden_dim=4096, n_heads=32, n_kv_heads=8, head_dim=128,
    // intermediate_dim=12288.
    const cases = [_]BenchCase{
        .{ .label = "attn_q       M=4096  K=4096 ", .M = 4096, .K = 4096 },
        .{ .label = "attn_k/v     M=1024  K=4096 ", .M = 1024, .K = 4096 },
        .{ .label = "attn_o       M=4096  K=4096 ", .M = 4096, .K = 4096 },
        .{ .label = "ffn_up/gate  M=12288 K=4096 ", .M = 12288, .K = 4096 },
        .{ .label = "ffn_down     M=4096  K=12288", .M = 4096, .K = 12288 },
    };

    const max_M: u32 = 12288;
    const max_K: u32 = 12288;
    const max_blocks: usize = @as(usize, max_M) * (max_K / 256);
    var w_buf = try metal_buffer.createBuffer(device.ctx, max_blocks * 144);
    defer metal_buffer.freeBuffer(&w_buf);
    fillQ4K(&w_buf, max_blocks);

    var x_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, max_K) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&x_buf);
    fillF32(&x_buf, max_K);

    var y_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, max_M) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&y_buf);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &stdout_buffer);
    try stdout.interface.print(
        "dmmv_q4k decode microbenchmark (N=1) | GPU={s}\n",
        .{@tagName(device.chip)},
    );
    try stdout.interface.print(
        "Peak DRAM bandwidth on this class: ~410 GB/s (M1 Max), ~546 GB/s (M4 Max)\n\n",
        .{},
    );

    const total_warmup: u32 = 50;
    const total_iters: u32 = 500;

    try stdout.interface.print(
        "{s:<32} {s:>10} {s:>10} {s:>10} {s:>9} {s:>9}\n",
        .{ "shape", "weight MB", "us/iter", "GB/s (W)", "vs 410", "vs 546" },
    );
    for (cases) |c| {
        const weight_bytes: u64 = @as(u64, c.M) * (c.K / 256) * 144;
        const ns = try benchShape(device.ctx, &pipe, &w_buf, &x_buf, &y_buf, c.M, c.K, total_warmup, total_iters);
        const w_gb: f64 = @as(f64, @floatFromInt(weight_bytes)) / 1e9;
        const gbs = w_gb / (ns / 1e9);
        const pct_m1: f64 = gbs / 410.0 * 100.0;
        const pct_m4: f64 = gbs / 546.0 * 100.0;
        const weight_mb: f64 = @as(f64, @floatFromInt(weight_bytes)) / (1024.0 * 1024.0);
        try stdout.interface.print(
            "{s:<32} {d:>10.2} {d:>10.2} {d:>10.1} {d:>8.1}% {d:>8.1}%\n",
            .{ c.label, weight_mb, ns / 1e3, gbs, pct_m1, pct_m4 },
        );
    }
    try stdout.interface.print("\n", .{});

    // Compute the implied per-token kernel sum for a 36-layer Qwen3-8B
    // decode step. This is the *kernel-only* lower bound; the real path
    // also runs attention, RoPE, norms, sampling, etc.
    var per_token_ns: f64 = 0;
    {
        const n_layers: f64 = 36;
        var attn_q_ns: f64 = 0;
        var attn_kv_ns: f64 = 0;
        var attn_o_ns: f64 = 0;
        var ffn_up_ns: f64 = 0;
        var ffn_down_ns: f64 = 0;
        attn_q_ns = try benchShape(device.ctx, &pipe, &w_buf, &x_buf, &y_buf, 4096, 4096, 20, 100);
        attn_kv_ns = try benchShape(device.ctx, &pipe, &w_buf, &x_buf, &y_buf, 1024, 4096, 20, 100);
        attn_o_ns = try benchShape(device.ctx, &pipe, &w_buf, &x_buf, &y_buf, 4096, 4096, 20, 100);
        ffn_up_ns = try benchShape(device.ctx, &pipe, &w_buf, &x_buf, &y_buf, 12288, 4096, 20, 100);
        ffn_down_ns = try benchShape(device.ctx, &pipe, &w_buf, &x_buf, &y_buf, 4096, 12288, 20, 100);
        // Per layer: attn_q + 2*attn_kv + attn_o + 2*ffn_up + ffn_down
        // (gate and up are two separate matvecs on the same K).
        const per_layer_ns = attn_q_ns + 2 * attn_kv_ns + attn_o_ns + 2 * ffn_up_ns + ffn_down_ns;
        per_token_ns = per_layer_ns * n_layers;
    }
    const per_token_ms = per_token_ns / 1e6;
    const tok_per_sec = 1e9 / per_token_ns;
    try stdout.interface.print(
        "Implied 36-layer Q4_K matvec lower bound (ignores attn/norm/RoPE/lm_head):\n",
        .{},
    );
    try stdout.interface.print(
        "  {d:.2} ms/token => {d:.1} tok/s (kernel-only ceiling for matvec)\n",
        .{ per_token_ms, tok_per_sec },
    );
    try stdout.interface.print(
        "  Compare against current ZINC end-to-end decode: ~8.6 tok/s (116 ms/token).\n",
        .{},
    );
    try stdout.interface.flush();
}
