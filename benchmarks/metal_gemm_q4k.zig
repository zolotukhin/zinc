const std = @import("std");
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

// Mirrors the GemmPush struct in src/shaders/metal/gemm_q4k.metal.
const GemmPush = extern struct {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    ne12: i32,
    _pad0: u32 = 0,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    ne0: i32,
    ne1: i32,
    src0_off: u32,
};

fn loadShaderPipeline(ctx: ?*shim.MetalCtx, name: []const u8) !MetalPipeline {
    var path_buf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "src/shaders/metal/{s}.metal", .{name}) catch return error.PathTooLong;
    const file = std.fs.cwd().openFile(path, .{}) catch return error.ShaderNotFound;
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

// Fill Q4_K buffer with deterministic pseudo-random bytes + small scales so
// dequantized values stay in a reasonable range. Output correctness is not
// measured here — only kernel throughput.
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

fn benchShape(
    ctx: ?*shim.MetalCtx,
    pipe: *const MetalPipeline,
    w_buf: *const MetalBuffer,
    x_buf: *const MetalBuffer,
    y_buf: *const MetalBuffer,
    M: u32,
    K: u32,
    N: u32,
    warmup: u32,
    iters: u32,
) !f64 {
    const push = GemmPush{
        .ne00 = @intCast(K),
        .ne02 = 1,
        .nb01 = @as(u64, K / 256) * 144,
        .nb02 = 0,
        .ne12 = 1,
        .nb10 = 4,
        .nb11 = @as(u64, K) * 4,
        .nb12 = 0,
        .ne0 = @intCast(M),
        .ne1 = @intCast(N),
        .src0_off = 0,
    };
    const bufs = [_]*const MetalBuffer{ w_buf, x_buf, y_buf };
    const grid_x = (N + 31) / 32;
    const grid_y = (M + 63) / 64;
    const grid = [_]u32{ grid_x, grid_y, 1 };
    const block = [_]u32{ 128, 1, 1 };
    const tg_mem: u32 = 8192;

    for (0..warmup) |_| {
        var cmd = try metal_command.beginCommand(ctx);
        cmd.dispatchV2WithTgMem(pipe, grid, block, &bufs, &push, @sizeOf(GemmPush), 0, tg_mem);
        cmd.commitAndWait();
    }

    const start = std.time.nanoTimestamp();
    var cmd = try metal_command.beginCommand(ctx);
    for (0..iters) |_| {
        cmd.dispatchV2WithTgMem(pipe, grid, block, &bufs, &push, @sizeOf(GemmPush), 0, tg_mem);
    }
    cmd.commitAndWait();
    const end = std.time.nanoTimestamp();

    return @as(f64, @floatFromInt(end - start)) / @as(f64, @floatFromInt(iters));
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var device = try metal_device.MetalDevice.init(allocator, 0);
    defer device.deinit();

    var gpu_lock = process_lock.acquire(.metal, device.selected_device_index) catch |err| {
        support.reportGpuProcessLockError(err, .metal, device.selected_device_index);
    };
    defer gpu_lock.deinit();

    var pipe = try loadShaderPipeline(device.ctx, "gemm_q4k");
    defer metal_pipeline.freePipeline(&pipe);

    // Qwen3-8B shapes we care about.
    const cases = [_]BenchCase{
        .{ .label = "attn_q/out  (M=4096 K=4096)", .M = 4096, .K = 4096 },
        .{ .label = "attn_kv     (M=1024 K=4096)", .M = 1024, .K = 4096 },
        .{ .label = "ffn_up/gate (M=12288 K=4096)", .M = 12288, .K = 4096 },
        .{ .label = "ffn_down    (M=4096 K=12288)", .M = 4096, .K = 12288 },
    };

    // Allocate buffers sized for the largest case, reuse across shapes.
    const max_M: u32 = 12288;
    const max_K: u32 = 12288;
    const max_N: u32 = 512;
    const max_blocks: usize = @as(usize, max_M) * (max_K / 256);
    var w_buf = try metal_buffer.createBuffer(device.ctx, max_blocks * 144);
    defer metal_buffer.freeBuffer(&w_buf);
    fillQ4K(&w_buf, max_blocks);

    var x_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, max_N) * max_K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&x_buf);
    fillF32(&x_buf, max_N * max_K);

    var y_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, max_N) * max_M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&y_buf);

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);
    try stdout.interface.print(
        "gemm_q4k microbenchmark | GPU={s} | tgmem_max={d} KiB\n\n",
        .{ @tagName(device.chip), device.maxThreadgroupMemoryLength() / 1024 },
    );

    const token_sizes = [_]u32{ 1, 4, 16, 64, 128, 256, 411, 512 };

    for (cases) |c| {
        const weight_bytes: u64 = @as(u64, c.M) * (c.K / 256) * 144;
        try stdout.interface.print(
            "=== {s} | weight={d:.2} MiB ===\n",
            .{ c.label, @as(f64, @floatFromInt(weight_bytes)) / (1024.0 * 1024.0) },
        );
        try stdout.interface.print("{s:>4} {s:>10} {s:>10} {s:>12} {s:>14}\n", .{ "N", "ms", "GFLOP/s", "GB/s (W)", "ns per (M,N)" });
        for (token_sizes) |N| {
            const ns = try benchShape(device.ctx, &pipe, &w_buf, &x_buf, &y_buf, c.M, c.K, N, 5, 30);
            const flops: f64 = @as(f64, @floatFromInt(c.M)) * @as(f64, @floatFromInt(c.K)) * @as(f64, @floatFromInt(N)) * 2.0;
            const gflops = flops / ns;
            const w_gb: f64 = @as(f64, @floatFromInt(weight_bytes)) / 1e9;
            const gbs = w_gb / (ns / 1e9);
            const ns_per_token: f64 = ns / @as(f64, @floatFromInt(N));
            try stdout.interface.print("{d:>4} {d:>10.3} {d:>10.1} {d:>12.1} {d:>14.3}\n", .{ N, ns / 1e6, gflops, gbs, ns_per_token });
        }
        try stdout.interface.print("\n", .{});
    }
    try stdout.interface.flush();
}
