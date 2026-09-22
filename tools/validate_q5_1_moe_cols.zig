//! Kernel-level check for `dmmv_q5_1_moe_cols.comp` — the route-packed Q5_1 MoE
//! down projection that dominates Gemma MoE prefill.
//!
//! Four rewrites of that shader were each provably equivalent on paper and each
//! silently wrong on the GPU, because the only signal available was end-to-end
//! model output: the model kept generating, it just answered differently. This
//! runs the shader on synthetic weights and compares every output against a CPU
//! Q5_1 dequant reference, so a wrong kernel fails in seconds with the exact
//! route and row that diverged.
//!
//!   zig build validate-moe-cols && ./zig-out/bin/zinc-validate-q5-1-moe-cols
//!
//! Exit code is non-zero when any element exceeds the tolerance, so it works as
//! a gate before touching that shader.
const std = @import("std");

const vulkan = @import("vulkan");
const vk = vulkan.vk;
const instance_mod = vulkan.instance;
const Instance = vulkan.Instance;
const buffer = vulkan.buffer;
const Buffer = vulkan.Buffer;
const command = vulkan.command;
const pipeline_mod = vulkan.pipeline;

/// Mirrors MoeColsDmmvPushConstants in src/compute/dmmv.zig.
const MoeColsPush = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    expert_stride: u32,
    x_offset: u32,
    y_offset: u32,
    ids_stride: u32,
    x_route_divisor: u32,
    accumulate: u32 = 0,
};

const q5_1_block_size: usize = 32;
const q5_1_block_bytes: usize = 24;
/// The shader packs eight token routes per active block (NUM_COLS).
const cols_per_block: u32 = 8;

fn halfToFloat(bits: u16) f32 {
    return @floatCast(@as(f16, @bitCast(bits)));
}

/// CPU reference for one Q5_1 row: q = nibble | (qh bit << 4), value = q*d + m.
fn dequantQ5_1Row(raw: []const u8, row: usize, K: usize, out: []f32) void {
    const blocks = K / q5_1_block_size;
    const row_base = row * blocks * q5_1_block_bytes;
    for (0..blocks) |b| {
        const base = row_base + b * q5_1_block_bytes;
        const d = halfToFloat(std.mem.readInt(u16, raw[base..][0..2], .little));
        const m = halfToFloat(std.mem.readInt(u16, raw[base + 2 ..][0..2], .little));
        const qh = std.mem.readInt(u32, raw[base + 4 ..][0..4], .little);
        for (0..16) |j| {
            const byte = raw[base + 8 + j];
            const lo: u32 = (byte & 0x0F) | (((qh >> @intCast(j)) & 1) << 4);
            const hi: u32 = (byte >> 4) | (((qh >> @intCast(j + 16)) & 1) << 4);
            out[b * q5_1_block_size + j] = @as(f32, @floatFromInt(lo)) * d + m;
            out[b * q5_1_block_size + j + 16] = @as(f32, @floatFromInt(hi)) * d + m;
        }
    }
}

const Shape = struct {
    name: []const u8,
    M: u32, // output rows (hidden_dim)
    K: u32, // reduction length (inter_dim)
    n_experts: u32,
    routes: u32, // token routes packed across experts
    accumulate: bool,
    /// Check every Nth row. Production-scale shapes would otherwise spend
    /// minutes in the CPU reference; the GPU still computes all of them.
    row_step: u32 = 1,
};

fn fillDeterministic(weight: []u8, input: []f32, seed: u64) void {
    var rng = std.Random.DefaultPrng.init(seed);
    const r = rng.random();
    // Full-range quants and scales: a kernel that mixes up the fifth bit or the
    // min only shows up when both halves of the range are exercised.
    for (weight) |*b| b.* = r.int(u8);
    var i: usize = 0;
    while (i < weight.len) : (i += q5_1_block_bytes) {
        const d: f16 = @floatCast(0.0625 * (r.float(f32) * 2.0 - 1.0));
        const m: f16 = @floatCast(0.25 * (r.float(f32) * 2.0 - 1.0));
        std.mem.writeInt(u16, weight[i..][0..2], @bitCast(d), .little);
        std.mem.writeInt(u16, weight[i + 2 ..][0..2], @bitCast(m), .little);
    }
    for (input) |*v| v.* = r.float(f32) * 2.0 - 1.0;
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const shader_dir = std.posix.getenv("ZINC_SHADER_DIR") orelse "zig-out/share/zinc/shaders";
    var path_buf: [512]u8 = undefined;
    const shader_path = try std.fmt.bufPrintZ(&path_buf, "{s}/dmmv_q5_1_moe_cols.spv", .{shader_dir});

    const device_index = if (std.posix.getenv("ZINC_GPU")) |raw|
        std.fmt.parseInt(u32, raw, 10) catch instance_mod.auto_select_device_index
    else
        instance_mod.auto_select_device_index;

    var instance = try Instance.init(allocator, device_index);
    defer instance.deinit();
    if (instance.push_descriptor_fn == null) return error.PushDescriptorsUnavailable;

    var pool = try command.CommandPool.init(&instance);
    defer pool.deinit();

    var pipe = try pipeline_mod.createFromSpirvWithOptions(
        &instance,
        shader_path,
        6,
        @sizeOf(MoeColsPush),
        &.{},
        .{ .required_subgroup_size = 64, .require_full_subgroups = true, .push_descriptors = true },
        allocator,
    );
    defer pipe.deinit();

    const shapes = [_]Shape{
        // The shape this kernel actually runs in production (gemma4-26b-a4b).
        .{ .name = "gemma4-26b-a4b down", .M = 2816, .K = 704, .n_experts = 4, .routes = 37, .accumulate = false },
        // Ragged tail: routes not a multiple of the eight-column block.
        .{ .name = "ragged routes", .M = 512, .K = 704, .n_experts = 3, .routes = 19, .accumulate = false },
        // Accumulate mode, and a K with an odd block count per row.
        .{ .name = "accumulate, odd blocks", .M = 256, .K = 352, .n_experts = 2, .routes = 8, .accumulate = true },
        // Single route, single expert: the degenerate case.
        .{ .name = "single route", .M = 128, .K = 704, .n_experts = 1, .routes = 1, .accumulate = false },
        // Production scale: a real prompt fans hundreds of routes across the
        // model's full expert set, which is where a kernel that looks correct on
        // a handful of blocks can still go wrong.
        .{ .name = "prod scale 128 experts", .M = 2816, .K = 704, .n_experts = 128, .routes = 512, .accumulate = false, .row_step = 16 },
    };

    var failures: usize = 0;
    for (shapes) |shape| {
        const failed = try runShape(&instance, &pool, &pipe, shape, allocator);
        if (failed) failures += 1;
    }
    if (failures != 0) {
        std.debug.print("\n{d} of {d} shapes FAILED\n", .{ failures, shapes.len });
        std.process.exit(1);
    }
    std.debug.print("\nall {d} shapes matched the CPU reference\n", .{shapes.len});
}

fn runShape(
    instance: *Instance,
    pool: *command.CommandPool,
    pipe: *pipeline_mod.Pipeline,
    shape: Shape,
    allocator: std.mem.Allocator,
) !bool {
    const M: usize = shape.M;
    const K: usize = shape.K;
    const blocks_per_row = K / q5_1_block_size;
    const row_bytes = blocks_per_row * q5_1_block_bytes;
    const expert_stride = M * row_bytes;
    const routes: usize = shape.routes;
    const ids_stride: u32 = @intCast(routes);

    // Route ids are packed expert-major, the way moe_route_pack.comp emits them:
    // every expert owns a contiguous run of route slots.
    const counts = try allocator.alloc(u32, shape.n_experts);
    defer allocator.free(counts);
    const ids = try allocator.alloc(u32, shape.n_experts * routes);
    defer allocator.free(ids);
    @memset(ids, std.math.maxInt(u32));
    var route_expert = try allocator.alloc(u32, routes);
    defer allocator.free(route_expert);

    var active_blocks: std.ArrayList(u32) = .{};
    defer active_blocks.deinit(allocator);
    var next_route: u32 = 0;
    for (0..shape.n_experts) |e| {
        // Spread the routes across experts, last expert takes the remainder.
        const share: u32 = @intCast(if (e == shape.n_experts - 1)
            routes - next_route
        else
            routes / shape.n_experts);
        counts[e] = share;
        for (0..share) |k| {
            ids[e * routes + k] = next_route;
            route_expert[next_route] = @intCast(e);
            next_route += 1;
        }
        var block: u32 = 0;
        while (block * cols_per_block < share) : (block += 1) {
            try active_blocks.append(allocator, @as(u32, @intCast(e)) | (block << 16));
        }
    }

    var weight_buf = try Buffer.initHostVisibleStorage(instance, shape.n_experts * expert_stride);
    defer weight_buf.deinit();
    var input_buf = try Buffer.initHostVisibleStorage(instance, routes * K * @sizeOf(f32));
    defer input_buf.deinit();
    var output_buf = try Buffer.initHostVisibleStorage(instance, routes * M * @sizeOf(f32));
    defer output_buf.deinit();
    var counts_buf = try Buffer.initHostVisibleStorage(instance, shape.n_experts * @sizeOf(u32));
    defer counts_buf.deinit();
    var ids_buf = try Buffer.initHostVisibleStorage(instance, ids.len * @sizeOf(u32));
    defer ids_buf.deinit();
    var active_buf = try Buffer.initHostVisibleStorage(instance, active_blocks.items.len * @sizeOf(u32));
    defer active_buf.deinit();

    const weight = weight_buf.mapped.?[0 .. shape.n_experts * expert_stride];
    const input: [*]f32 = @ptrCast(@alignCast(input_buf.mapped.?));
    const output: [*]f32 = @ptrCast(@alignCast(output_buf.mapped.?));
    fillDeterministic(weight, input[0 .. routes * K], 0xA5A5 +% shape.M);

    const counts_gpu: [*]u32 = @ptrCast(@alignCast(counts_buf.mapped.?));
    @memcpy(counts_gpu[0..counts.len], counts);
    const ids_gpu: [*]u32 = @ptrCast(@alignCast(ids_buf.mapped.?));
    @memcpy(ids_gpu[0..ids.len], ids);
    const active_gpu: [*]u32 = @ptrCast(@alignCast(active_buf.mapped.?));
    @memcpy(active_gpu[0..active_blocks.items.len], active_blocks.items);

    // Seed the output so accumulate mode has something to add onto.
    for (0..routes * M) |i| {
        output[i] = if (shape.accumulate) @as(f32, @floatFromInt(i % 13)) * 0.125 - 0.5 else 0.0;
    }
    const seeded = try allocator.alloc(f32, routes * M);
    defer allocator.free(seeded);
    @memcpy(seeded, output[0 .. routes * M]);

    const push = MoeColsPush{
        .M = @intCast(M),
        .K = @intCast(K),
        .a_offset = 0,
        .expert_stride = @intCast(expert_stride),
        .x_offset = 0,
        .y_offset = 0,
        .ids_stride = ids_stride,
        .x_route_divisor = 1,
        .accumulate = if (shape.accumulate) 1 else 0,
    };
    const infos = [_]vk.c.VkDescriptorBufferInfo{
        .{ .buffer = weight_buf.handle, .offset = 0, .range = weight_buf.size },
        .{ .buffer = input_buf.handle, .offset = 0, .range = input_buf.size },
        .{ .buffer = output_buf.handle, .offset = 0, .range = output_buf.size },
        .{ .buffer = counts_buf.handle, .offset = 0, .range = counts_buf.size },
        .{ .buffer = ids_buf.handle, .offset = 0, .range = ids_buf.size },
        .{ .buffer = active_buf.handle, .offset = 0, .range = active_buf.size },
    };

    var cmd = try command.CommandBuffer.init(instance, pool);
    defer cmd.deinit(pool);
    try cmd.beginOneTime();
    // Grid x covers the rows (the shader takes eight rows per workgroup but the
    // production dispatch sizes x by M/4, so mirror that); y is one per active block.
    cmd.pushDescAndDispatch(
        pipe,
        instance.push_descriptor_fn,
        infos[0..],
        std.mem.asBytes(&push),
        @intCast((M + 3) / 4),
        @intCast(active_blocks.items.len),
        1,
    );
    try cmd.end();
    try cmd.submitAndWait(instance.compute_queue);

    const ref_row = try allocator.alloc(f32, K);
    defer allocator.free(ref_row);

    var max_diff: f32 = 0.0;
    var worst_route: usize = 0;
    var worst_row: usize = 0;
    for (0..routes) |route| {
        const expert = route_expert[route];
        const matrix = weight[expert * expert_stride ..][0..expert_stride];
        const x = input[route * K .. (route + 1) * K];
        var row: usize = 0;
        while (row < M) : (row += shape.row_step) {
            dequantQ5_1Row(matrix, row, K, ref_row);
            var expected: f32 = if (shape.accumulate) seeded[route * M + row] else 0.0;
            for (0..K) |i| expected += ref_row[i] * x[i];
            const diff = @abs(expected - output[route * M + row]);
            if (diff > max_diff) {
                max_diff = diff;
                worst_route = route;
                worst_row = row;
            }
        }
    }

    // Tolerance scales with the reduction length: the shader splits the row
    // across lanes, so the summation order differs from the serial reference.
    const tolerance: f32 = 2e-3 * @as(f32, @floatFromInt(K)) / 704.0;
    const ok = max_diff <= tolerance;
    std.debug.print("{s:<24} M={d:<5} K={d:<4} experts={d} routes={d:<3} acc={} max_diff={d:.6} {s}\n", .{
        shape.name,               shape.M, shape.K, shape.n_experts, shape.routes, shape.accumulate, max_diff,
        if (ok) "ok" else "FAIL",
    });
    if (!ok) {
        std.debug.print("    worst at route {d} row {d} (tolerance {d:.6})\n", .{ worst_route, worst_row, tolerance });
    }
    return !ok;
}
