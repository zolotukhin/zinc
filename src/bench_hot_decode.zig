//! Hot-path decode kernel microbenchmarks.
//! Measures per-dispatch GPU latency, memory bandwidth, and VRAM utilisation
//! for individual compute kernels (DMMV, SSM delta-net) in isolation.
//! Run via `zig build hot-bench -Doptimize=ReleaseFast`.
//! @section Inference Runtime
const std = @import("std");
const vk = @import("vulkan/vk.zig");
const Instance = @import("vulkan/instance.zig").Instance;
const CommandPool = @import("vulkan/command.zig").CommandPool;
const CommandBuffer = @import("vulkan/command.zig").CommandBuffer;
const Buffer = @import("vulkan/buffer.zig").Buffer;
const buffer_mod = @import("vulkan/buffer.zig");
const gpu_detect = @import("vulkan/gpu_detect.zig");
const DmmvDispatch = @import("compute/dmmv.zig").DmmvDispatch;
const ElementwiseDispatch = @import("compute/elementwise.zig").ElementwiseDispatch;
const elementwise = @import("compute/elementwise.zig");
const GGMLType = @import("model/gguf.zig").GGMLType;
const loader = @import("model/loader.zig");

const log = std.log.scoped(.hot_bench);

const BenchModelConfig = struct {
    hidden_dim: u32 = 2048,
    n_experts: u32 = 256,
    shared_expert_intermediate_dim: u32 = 512,
    ssm_d_inner: u32 = 4096,
    ssm_dt_rank: u32 = 32,
    ssm_d_state: u32 = 128,
    ssm_n_group: u32 = 16,
};

const BenchKind = enum {
    dmmv_q8_0,
    ssm_delta_net,
};

const BenchCase = struct {
    name: []const u8,
    kind: BenchKind,
    M: u32 = 0,
    K: u32 = 0,
    d_inner: u32 = 0,
    dt_rank: u32 = 0,
    d_state: u32 = 0,
    n_group: u32 = 0,

    fn describe(self: BenchCase, buf: []u8) []const u8 {
        return switch (self.kind) {
            .dmmv_q8_0 => std.fmt.bufPrint(buf, "q8_0 M={d} K={d}", .{ self.M, self.K }) catch self.name,
            .ssm_delta_net => std.fmt.bufPrint(buf, "delta d_inner={d} dt_rank={d} d_state={d} n_group={d}", .{
                self.d_inner,
                self.dt_rank,
                self.d_state,
                self.n_group,
            }) catch self.name,
        };
    }
};

const BenchResult = struct {
    name: []const u8,
    kind: BenchKind,
    gpu_ms_per_iter: f64,
    wall_ms_per_iter: f64,
    overhead_ms_per_iter: f64,
    effective_gbps: f64,
    utilization_pct: f64,
    bytes_per_iter: u64,
    iterations: u32,
    M: u32 = 0,
    K: u32 = 0,
    d_inner: u32 = 0,
    dt_rank: u32 = 0,
    d_state: u32 = 0,
    n_group: u32 = 0,
};

const Args = struct {
    device: u32 = 0,
    iterations: u32 = 200,
    warmup: u32 = 25,
    working_set: u32 = 16,
    shader_dir: ?[]const u8 = null,
    model_path: ?[]const u8 = null,
    case_filter: ?[]const u8 = null,
    json: bool = false,

    fn deinit(self: *Args, allocator: std.mem.Allocator) void {
        if (self.shader_dir) |path| allocator.free(path);
        if (self.model_path) |path| allocator.free(path);
        if (self.case_filter) |path| allocator.free(path);
        self.* = undefined;
    }
};

const TimestampTimer = struct {
    pool: vk.c.VkQueryPool,
    period_ns: f64,
    device: vk.c.VkDevice,

    fn init(instance: *const Instance) !TimestampTimer {
        const pool_info = vk.c.VkQueryPoolCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queryType = vk.c.VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = 2,
            .pipelineStatistics = 0,
        };
        var pool: vk.c.VkQueryPool = null;
        const result = vk.c.vkCreateQueryPool(instance.device, &pool_info, null, &pool);
        if (result != vk.c.VK_SUCCESS) return error.QueryPoolCreateFailed;
        return .{
            .pool = pool,
            .period_ns = @as(f64, instance.device_props.limits.timestampPeriod),
            .device = instance.device,
        };
    }

    fn deinit(self: *TimestampTimer) void {
        vk.c.vkDestroyQueryPool(self.device, self.pool, null);
        self.* = undefined;
    }

    fn writeStart(self: *const TimestampTimer, cmd: *const CommandBuffer) void {
        vk.c.vkCmdResetQueryPool(cmd.handle, self.pool, 0, 2);
        vk.c.vkCmdWriteTimestamp(cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, self.pool, 0);
    }

    fn writeEnd(self: *const TimestampTimer, cmd: *const CommandBuffer) void {
        vk.c.vkCmdWriteTimestamp(cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, self.pool, 1);
    }

    fn elapsedMs(self: *const TimestampTimer) !f64 {
        var timestamps: [2]u64 = undefined;
        const qr = vk.c.vkGetQueryPoolResults(
            self.device,
            self.pool,
            0,
            2,
            2 * @sizeOf(u64),
            &timestamps,
            @sizeOf(u64),
            vk.c.VK_QUERY_RESULT_64_BIT | vk.c.VK_QUERY_RESULT_WAIT_BIT,
        );
        if (qr != vk.c.VK_SUCCESS) return error.QueryReadFailed;
        const elapsed_ns = @as(f64, @floatFromInt(timestamps[1] -| timestamps[0])) * self.period_ns;
        return elapsed_ns / 1e6;
    }
};

fn parseArgs(allocator: std.mem.Allocator) !Args {
    const argv = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, argv);

    var args = Args{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            printUsage();
            std.process.exit(0);
        } else if (std.mem.eql(u8, arg, "--device") or std.mem.eql(u8, arg, "-d")) {
            i += 1;
            if (i >= argv.len) return error.MissingArgument;
            args.device = try std.fmt.parseUnsigned(u32, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--iterations")) {
            i += 1;
            if (i >= argv.len) return error.MissingArgument;
            args.iterations = try std.fmt.parseUnsigned(u32, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--warmup")) {
            i += 1;
            if (i >= argv.len) return error.MissingArgument;
            args.warmup = try std.fmt.parseUnsigned(u32, argv[i], 10);
        } else if (std.mem.eql(u8, arg, "--working-set")) {
            i += 1;
            if (i >= argv.len) return error.MissingArgument;
            args.working_set = @max(1, try std.fmt.parseUnsigned(u32, argv[i], 10));
        } else if (std.mem.eql(u8, arg, "--shader-dir")) {
            i += 1;
            if (i >= argv.len) return error.MissingArgument;
            args.shader_dir = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--model") or std.mem.eql(u8, arg, "-m")) {
            i += 1;
            if (i >= argv.len) return error.MissingArgument;
            args.model_path = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--case")) {
            i += 1;
            if (i >= argv.len) return error.MissingArgument;
            args.case_filter = try allocator.dupe(u8, argv[i]);
        } else if (std.mem.eql(u8, arg, "--json")) {
            args.json = true;
        } else {
            log.err("Unknown argument: {s}", .{arg});
            printUsage();
            return error.InvalidArgument;
        }
    }
    return args;
}

fn printUsage() void {
    std.debug.print(
        \\Usage: zinc-hot-bench [options]
        \\  -m, --model <path>       Optional GGUF path to derive exact hot shapes
        \\  -d, --device <id>        Vulkan device index (default: 0)
        \\  --shader-dir <path>      Shader directory (default: zig-out/share/zinc/shaders)
        \\  --iterations <n>         Timed iterations per case (default: 200)
        \\  --warmup <n>             Warmup iterations per case (default: 25)
        \\  --working-set <n>        Rotate across N buffer sets to reduce cache-hot bias (default: 16)
        \\  --case <name>            Run one case: q8_router | q8_shared_gate_up | q8_shared_down | q8_ssm_out | ssm_delta
        \\  --json                   Emit JSON instead of log lines
        \\  -h, --help               Show this help
        \\
    , .{});
}

fn resolveShaderDir(allocator: std.mem.Allocator, override: ?[]const u8) ![]u8 {
    if (override) |path| return allocator.dupe(u8, path);

    const candidates = [_][]const u8{
        "zig-out/share/zinc/shaders",
        "share/zinc/shaders",
    };
    for (candidates) |candidate| {
        std.fs.cwd().access(candidate, .{}) catch continue;
        return allocator.dupe(u8, candidate);
    }

    const exe_path = try std.fs.selfExePathAlloc(allocator);
    defer allocator.free(exe_path);
    const exe_dir = std.fs.path.dirname(exe_path) orelse ".";
    const derived = try std.fs.path.join(allocator, &.{ exe_dir, "..", "share", "zinc", "shaders" });
    errdefer allocator.free(derived);
    std.fs.cwd().access(derived, .{}) catch return error.ShaderDirNotFound;
    return derived;
}

fn loadBenchModelConfig(path: ?[]const u8, allocator: std.mem.Allocator) !BenchModelConfig {
    var cfg = BenchModelConfig{};
    if (path) |model_path| {
        const inspected = try loader.inspectConfig(model_path, allocator);
        if (inspected.hidden_dim > 0) cfg.hidden_dim = inspected.hidden_dim;
        if (inspected.n_experts > 0) cfg.n_experts = inspected.n_experts;
        if (inspected.shared_expert_intermediate_dim > 0) cfg.shared_expert_intermediate_dim = inspected.shared_expert_intermediate_dim;
        if (inspected.ssm_d_inner > 0) cfg.ssm_d_inner = inspected.ssm_d_inner;
        if (inspected.ssm_dt_rank > 0) cfg.ssm_dt_rank = inspected.ssm_dt_rank;
        if (inspected.ssm_d_state > 0) cfg.ssm_d_state = inspected.ssm_d_state;
        if (inspected.ssm_n_group > 0) cfg.ssm_n_group = inspected.ssm_n_group;
    }
    return cfg;
}

fn buildCases(cfg: BenchModelConfig) [5]BenchCase {
    return .{
        .{ .name = "q8_router", .kind = .dmmv_q8_0, .M = cfg.n_experts, .K = cfg.hidden_dim },
        .{ .name = "q8_shared_gate_up", .kind = .dmmv_q8_0, .M = cfg.shared_expert_intermediate_dim, .K = cfg.hidden_dim },
        .{ .name = "q8_shared_down", .kind = .dmmv_q8_0, .M = cfg.hidden_dim, .K = cfg.shared_expert_intermediate_dim },
        .{ .name = "q8_ssm_out", .kind = .dmmv_q8_0, .M = cfg.hidden_dim, .K = cfg.ssm_d_inner },
        .{
            .name = "ssm_delta",
            .kind = .ssm_delta_net,
            .d_inner = cfg.ssm_d_inner,
            .dt_rank = cfg.ssm_dt_rank,
            .d_state = cfg.ssm_d_state,
            .n_group = cfg.ssm_n_group,
        },
    };
}

fn matchesCaseFilter(case_name: []const u8, filter: ?[]const u8) bool {
    if (filter == null) return true;
    return std.mem.eql(u8, case_name, filter.?);
}

fn createStorageBuffer(instance: *const Instance, size: usize) !Buffer {
    return Buffer.initDeviceLocal(instance, size, vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
}

fn initDeviceLocalWithBytes(instance: *const Instance, cmd_pool: *const CommandPool, bytes: []const u8) !Buffer {
    var staging = try Buffer.initStaging(instance, bytes.len);
    defer staging.deinit();
    staging.upload(bytes);
    var device_buf = try createStorageBuffer(instance, bytes.len);
    errdefer device_buf.deinit();
    try buffer_mod.copyBuffer(instance, cmd_pool.handle, &staging, &device_buf, bytes.len);
    return device_buf;
}

fn allocDescSet(device: vk.c.VkDevice, pool: vk.c.VkDescriptorPool, layout: vk.c.VkDescriptorSetLayout) !vk.c.VkDescriptorSet {
    const alloc_info = vk.c.VkDescriptorSetAllocateInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
        .pNext = null,
        .descriptorPool = pool,
        .descriptorSetCount = 1,
        .pSetLayouts = &layout,
    };
    var ds: vk.c.VkDescriptorSet = null;
    const result = vk.c.vkAllocateDescriptorSets(device, &alloc_info, &ds);
    if (result != vk.c.VK_SUCCESS) return error.DescriptorSetAllocFailed;
    return ds;
}

fn writeDescSet(comptime N: usize, device: vk.c.VkDevice, ds: vk.c.VkDescriptorSet, infos: *[N]vk.c.VkDescriptorBufferInfo) void {
    var writes: [N]vk.c.VkWriteDescriptorSet = undefined;
    for (0..N) |i| {
        writes[i] = .{
            .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = null,
            .dstSet = ds,
            .dstBinding = @intCast(i),
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = null,
            .pBufferInfo = &infos[i],
            .pTexelBufferView = null,
        };
    }
    vk.c.vkUpdateDescriptorSets(device, N, &writes, 0, null);
}

fn q8MatrixBytes(M: u32, K: u32) u64 {
    const blocks_per_row = @as(u64, (K + GGMLType.q8_0.blockSize() - 1) / GGMLType.q8_0.blockSize());
    return @as(u64, M) * blocks_per_row * GGMLType.q8_0.bytesPerBlock();
}

fn dmmvBytesPerIter(M: u32, K: u32) u64 {
    return q8MatrixBytes(M, K) + @as(u64, K) * @sizeOf(f32) + @as(u64, M) * @sizeOf(f32);
}

fn ssmDeltaApproxBytesPerIter(d_inner: u32, dt_rank: u32, d_state: u32, n_group: u32) u64 {
    const head_v_dim = d_inner / dt_rank;
    const conv_channels = d_inner + 2 * n_group * d_state;
    const conv_out_bytes = @as(u64, conv_channels) * @sizeOf(f32);
    const alpha_beta_bytes = @as(u64, dt_rank * 2) * @sizeOf(f32);
    const output_bytes = @as(u64, d_inner) * @sizeOf(f32);
    const state_bytes = @as(u64, dt_rank) * @as(u64, head_v_dim) * @as(u64, head_v_dim) * @sizeOf(f32);
    const param_bytes = @as(u64, dt_rank * 2) * @sizeOf(f16);
    return conv_out_bytes + alpha_beta_bytes + output_bytes + param_bytes + state_bytes * 2;
}

fn fillQ80Weights(dst: []u8, M: u32, K: u32, salt: u32) void {
    const block_bytes: usize = GGMLType.q8_0.bytesPerBlock();
    const blocks_per_row: usize = (K + GGMLType.q8_0.blockSize() - 1) / GGMLType.q8_0.blockSize();
    const scale: f16 = 0.125;
    const scale_bytes = std.mem.asBytes(&scale);
    var row: u32 = 0;
    while (row < M) : (row += 1) {
        var block: u32 = 0;
        while (block < blocks_per_row) : (block += 1) {
            const off = (@as(usize, row) * blocks_per_row + @as(usize, block)) * block_bytes;
            @memcpy(dst[off..][0..2], scale_bytes);
            for (0..32) |i| {
                const lane: i32 = @intCast(i);
                const mixed = @mod(lane + @as(i32, @intCast(row)) + @as(i32, @intCast(block)) + @as(i32, @intCast(salt)), 15);
                const pattern: i8 = @intCast(mixed - 7);
                dst[off + 2 + i] = @bitCast(pattern);
            }
        }
    }
}

fn fillF32Pattern(dst: []f32, scale: f32, salt: u32) void {
    for (dst, 0..) |*value, i| {
        const lane = @as(f32, @floatFromInt(((i + salt) % 17) + 1));
        value.* = lane * scale;
    }
}

fn fillF16Pattern(dst: []f16, scale: f16, salt: u32) void {
    for (dst, 0..) |*value, i| {
        const lane: f16 = @floatFromInt(((i + salt) % 7) + 1);
        value.* = lane * scale;
    }
}

const DmmvSlot = struct {
    weights: Buffer,
    x: Buffer,
    y: Buffer,
    descriptor_set: vk.c.VkDescriptorSet,

    fn deinit(self: *DmmvSlot) void {
        self.weights.deinit();
        self.x.deinit();
        self.y.deinit();
        self.* = undefined;
    }
};

const SsmDeltaSlot = struct {
    conv: Buffer,
    dt_bias: Buffer,
    alpha: Buffer,
    beta: Buffer,
    ssm_a: Buffer,
    state: Buffer,
    output: Buffer,
    descriptor_set: vk.c.VkDescriptorSet,

    fn deinit(self: *SsmDeltaSlot) void {
        self.conv.deinit();
        self.dt_bias.deinit();
        self.alpha.deinit();
        self.beta.deinit();
        self.ssm_a.deinit();
        self.state.deinit();
        self.output.deinit();
        self.* = undefined;
    }
};

fn recordRepeatedDmmv(
    cmd: *const CommandBuffer,
    timer: *const TimestampTimer,
    dispatch: *const DmmvDispatch,
    descriptor_sets: []const vk.c.VkDescriptorSet,
    M: u32,
    K: u32,
    iterations: u32,
) !void {
    timer.writeStart(cmd);
    var i: u32 = 0;
    while (i < iterations) : (i += 1) {
        const ds = descriptor_sets[i % descriptor_sets.len];
        try dispatch.recordDispatch(cmd, .q8_0, ds, M, K, 0, 0, 0);
    }
    timer.writeEnd(cmd);
}

fn recordRepeatedSsmDelta(
    cmd: *const CommandBuffer,
    timer: *const TimestampTimer,
    dispatch: *const ElementwiseDispatch,
    descriptor_sets: []const vk.c.VkDescriptorSet,
    push: elementwise.SsmDeltaNetPush,
    iterations: u32,
) !void {
    timer.writeStart(cmd);
    var i: u32 = 0;
    while (i < iterations) : (i += 1) {
        const ds = descriptor_sets[i % descriptor_sets.len];
        try dispatch.recordSsmDeltaNet(cmd, ds, push);
        if (i + 1 < iterations) cmd.computeBarrier();
    }
    timer.writeEnd(cmd);
}

fn runRecorded(
    cmd: *CommandBuffer,
    queue: vk.c.VkQueue,
    timer: *const TimestampTimer,
) !struct { gpu_ms: f64, wall_ms: f64 } {
    try cmd.end();
    const wall_start = std.time.nanoTimestamp();
    try cmd.submitAndWait(queue);
    const wall_end = std.time.nanoTimestamp();
    return .{
        .gpu_ms = try timer.elapsedMs(),
        .wall_ms = @as(f64, @floatFromInt(wall_end - wall_start)) / 1_000_000.0,
    };
}

fn warmupRecorded(cmd: *CommandBuffer, queue: vk.c.VkQueue) !void {
    try cmd.end();
    try cmd.submitAndWait(queue);
}

fn runDmmvCase(
    allocator: std.mem.Allocator,
    instance: *const Instance,
    queue: vk.c.VkQueue,
    cmd_pool: *const CommandPool,
    cmd: *CommandBuffer,
    timer: *const TimestampTimer,
    dispatch: *const DmmvDispatch,
    gpu_config: *const gpu_detect.GpuConfig,
    case: BenchCase,
    iterations: u32,
    warmup: u32,
    working_set: u32,
) !BenchResult {
    const weight_bytes = q8MatrixBytes(case.M, case.K);
    const slot_count: usize = @intCast(working_set);

    const weight_blob = try allocator.alloc(u8, @intCast(weight_bytes));
    defer allocator.free(weight_blob);

    const x_host = try allocator.alloc(f32, case.K);
    defer allocator.free(x_host);

    const y_host = try allocator.alloc(f32, case.M);
    defer allocator.free(y_host);
    const pipeline = dispatch.pipelineForType(.q8_0) orelse return error.ShaderNotLoaded;
    const slots = try allocator.alloc(DmmvSlot, slot_count);
    defer allocator.free(slots);
    var init_count: usize = 0;
    errdefer {
        for (slots[0..init_count]) |*slot| slot.deinit();
    }

    while (init_count < slot_count) : (init_count += 1) {
        fillQ80Weights(weight_blob, case.M, case.K, @intCast(init_count * 13));
        fillF32Pattern(x_host, 0.03125, @intCast(init_count * 17));
        @memset(y_host, 0.0);

        slots[init_count].weights = try initDeviceLocalWithBytes(instance, cmd_pool, weight_blob);
        slots[init_count].x = try initDeviceLocalWithBytes(instance, cmd_pool, std.mem.sliceAsBytes(x_host));
        slots[init_count].y = try initDeviceLocalWithBytes(instance, cmd_pool, std.mem.sliceAsBytes(y_host));
        slots[init_count].descriptor_set = try allocDescSet(instance.device, dispatch.descriptor_pool, pipeline.descriptor_set_layout);
        var infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = slots[init_count].weights.handle, .offset = 0, .range = slots[init_count].weights.size },
            .{ .buffer = slots[init_count].x.handle, .offset = 0, .range = slots[init_count].x.size },
            .{ .buffer = slots[init_count].y.handle, .offset = 0, .range = slots[init_count].y.size },
        };
        writeDescSet(3, instance.device, slots[init_count].descriptor_set, &infos);
    }
    defer {
        for (slots[0..slot_count]) |*slot| slot.deinit();
    }

    const descriptor_sets = try allocator.alloc(vk.c.VkDescriptorSet, slot_count);
    defer allocator.free(descriptor_sets);
    for (slots, 0..) |slot, i| descriptor_sets[i] = slot.descriptor_set;

    try cmd.reset();
    try cmd.beginOneTime();
    try recordRepeatedDmmv(cmd, timer, dispatch, descriptor_sets, case.M, case.K, warmup);
    try warmupRecorded(cmd, queue);

    try cmd.reset();
    try cmd.beginOneTime();
    try recordRepeatedDmmv(cmd, timer, dispatch, descriptor_sets, case.M, case.K, iterations);
    const measured = try runRecorded(cmd, queue, timer);

    const bytes_per_iter = dmmvBytesPerIter(case.M, case.K);
    const eff_gbps = (@as(f64, @floatFromInt(bytes_per_iter)) * @as(f64, @floatFromInt(iterations))) / (measured.gpu_ms / 1000.0) / 1_000_000_000.0;
    const utilization = if (gpu_config.bandwidth_gbps > 0) eff_gbps / @as(f64, @floatFromInt(gpu_config.bandwidth_gbps)) * 100.0 else 0.0;

    return .{
        .name = case.name,
        .kind = case.kind,
        .gpu_ms_per_iter = measured.gpu_ms / @as(f64, @floatFromInt(iterations)),
        .wall_ms_per_iter = measured.wall_ms / @as(f64, @floatFromInt(iterations)),
        .overhead_ms_per_iter = @max(0.0, measured.wall_ms - measured.gpu_ms) / @as(f64, @floatFromInt(iterations)),
        .effective_gbps = eff_gbps,
        .utilization_pct = utilization,
        .bytes_per_iter = bytes_per_iter,
        .iterations = iterations,
        .M = case.M,
        .K = case.K,
    };
}

fn runSsmDeltaCase(
    allocator: std.mem.Allocator,
    instance: *const Instance,
    queue: vk.c.VkQueue,
    cmd_pool: *const CommandPool,
    cmd: *CommandBuffer,
    timer: *const TimestampTimer,
    dispatch: *const ElementwiseDispatch,
    gpu_config: *const gpu_detect.GpuConfig,
    case: BenchCase,
    iterations: u32,
    warmup: u32,
    working_set: u32,
) !BenchResult {
    const d_inner = case.d_inner;
    const dt_rank = case.dt_rank;
    const d_state = case.d_state;
    const n_group = case.n_group;
    const head_v_dim = d_inner / dt_rank;
    const conv_channels = d_inner + 2 * n_group * d_state;
    const state_len: usize = @intCast(@as(u64, dt_rank) * @as(u64, head_v_dim) * @as(u64, head_v_dim));
    const slot_count: usize = @intCast(working_set);

    const conv_out = try allocator.alloc(f32, conv_channels);
    defer allocator.free(conv_out);
    const alpha = try allocator.alloc(f32, dt_rank);
    defer allocator.free(alpha);
    const beta = try allocator.alloc(f32, dt_rank);
    defer allocator.free(beta);
    const state = try allocator.alloc(f32, state_len);
    defer allocator.free(state);
    const output = try allocator.alloc(f32, d_inner);
    defer allocator.free(output);
    const dt_bias = try allocator.alloc(f16, dt_rank);
    defer allocator.free(dt_bias);
    const ssm_a = try allocator.alloc(f16, dt_rank);
    defer allocator.free(ssm_a);

    const pipeline = dispatch.pipeline_ssm_delta_net orelse return error.ShaderNotLoaded;
    const slots = try allocator.alloc(SsmDeltaSlot, slot_count);
    defer allocator.free(slots);
    var init_count: usize = 0;
    errdefer {
        for (slots[0..init_count]) |*slot| slot.deinit();
    }

    while (init_count < slot_count) : (init_count += 1) {
        fillF32Pattern(conv_out, 0.0078125, @intCast(init_count * 11));
        fillF32Pattern(alpha, 0.125, @intCast(init_count * 3));
        fillF32Pattern(beta, 0.25, @intCast(init_count * 5));
        fillF32Pattern(state, 0.00048828125, @intCast(init_count * 7));
        @memset(output, 0.0);
        fillF16Pattern(dt_bias, @as(f16, 0.0625), @intCast(init_count * 2));
        fillF16Pattern(ssm_a, @as(f16, -0.03125), @intCast(init_count * 4));

        slots[init_count].conv = try initDeviceLocalWithBytes(instance, cmd_pool, std.mem.sliceAsBytes(conv_out));
        slots[init_count].dt_bias = try initDeviceLocalWithBytes(instance, cmd_pool, std.mem.sliceAsBytes(dt_bias));
        slots[init_count].alpha = try initDeviceLocalWithBytes(instance, cmd_pool, std.mem.sliceAsBytes(alpha));
        slots[init_count].beta = try initDeviceLocalWithBytes(instance, cmd_pool, std.mem.sliceAsBytes(beta));
        slots[init_count].ssm_a = try initDeviceLocalWithBytes(instance, cmd_pool, std.mem.sliceAsBytes(ssm_a));
        slots[init_count].state = try initDeviceLocalWithBytes(instance, cmd_pool, std.mem.sliceAsBytes(state));
        slots[init_count].output = try initDeviceLocalWithBytes(instance, cmd_pool, std.mem.sliceAsBytes(output));
        slots[init_count].descriptor_set = try allocDescSet(instance.device, dispatch.descriptor_pool, pipeline.descriptor_set_layout);
        var infos = [7]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = slots[init_count].conv.handle, .offset = 0, .range = slots[init_count].conv.size },
            .{ .buffer = slots[init_count].dt_bias.handle, .offset = 0, .range = slots[init_count].dt_bias.size },
            .{ .buffer = slots[init_count].alpha.handle, .offset = 0, .range = slots[init_count].alpha.size },
            .{ .buffer = slots[init_count].beta.handle, .offset = 0, .range = slots[init_count].beta.size },
            .{ .buffer = slots[init_count].ssm_a.handle, .offset = 0, .range = slots[init_count].ssm_a.size },
            .{ .buffer = slots[init_count].state.handle, .offset = 0, .range = slots[init_count].state.size },
            .{ .buffer = slots[init_count].output.handle, .offset = 0, .range = slots[init_count].output.size },
        };
        writeDescSet(7, instance.device, slots[init_count].descriptor_set, &infos);
    }
    defer {
        for (slots[0..slot_count]) |*slot| slot.deinit();
    }

    const descriptor_sets = try allocator.alloc(vk.c.VkDescriptorSet, slot_count);
    defer allocator.free(descriptor_sets);
    for (slots, 0..) |slot, i| descriptor_sets[i] = slot.descriptor_set;

    const push = elementwise.SsmDeltaNetPush{
        .d_inner = d_inner,
        .dt_rank = dt_rank,
        .head_v_dim = head_v_dim,
        .d_state = d_state,
        .n_group = n_group,
        .ssm_a_is_f16 = 1,
        .dt_bias_is_f16 = 1,
        .has_dt_bias = 1,
        .has_ssm_a = 1,
    };

    try cmd.reset();
    try cmd.beginOneTime();
    try recordRepeatedSsmDelta(cmd, timer, dispatch, descriptor_sets, push, warmup);
    try warmupRecorded(cmd, queue);

    try cmd.reset();
    try cmd.beginOneTime();
    try recordRepeatedSsmDelta(cmd, timer, dispatch, descriptor_sets, push, iterations);
    const measured = try runRecorded(cmd, queue, timer);

    const bytes_per_iter = ssmDeltaApproxBytesPerIter(d_inner, dt_rank, d_state, n_group);
    const eff_gbps = (@as(f64, @floatFromInt(bytes_per_iter)) * @as(f64, @floatFromInt(iterations))) / (measured.gpu_ms / 1000.0) / 1_000_000_000.0;
    const utilization = if (gpu_config.bandwidth_gbps > 0) eff_gbps / @as(f64, @floatFromInt(gpu_config.bandwidth_gbps)) * 100.0 else 0.0;

    return .{
        .name = case.name,
        .kind = case.kind,
        .gpu_ms_per_iter = measured.gpu_ms / @as(f64, @floatFromInt(iterations)),
        .wall_ms_per_iter = measured.wall_ms / @as(f64, @floatFromInt(iterations)),
        .overhead_ms_per_iter = @max(0.0, measured.wall_ms - measured.gpu_ms) / @as(f64, @floatFromInt(iterations)),
        .effective_gbps = eff_gbps,
        .utilization_pct = utilization,
        .bytes_per_iter = bytes_per_iter,
        .iterations = iterations,
        .d_inner = d_inner,
        .dt_rank = dt_rank,
        .d_state = d_state,
        .n_group = n_group,
    };
}

fn printResults(results: []const BenchResult) void {
    for (results) |result| {
        switch (result.kind) {
            .dmmv_q8_0 => log.info("{s}: gpu={d:.3} ms/iter wall={d:.3} ms/iter overhead={d:.3} ms/iter | {d:.1} GB/s | {d:.1}% of peak | {d} B/iter | M={d} K={d}", .{
                result.name,
                result.gpu_ms_per_iter,
                result.wall_ms_per_iter,
                result.overhead_ms_per_iter,
                result.effective_gbps,
                result.utilization_pct,
                result.bytes_per_iter,
                result.M,
                result.K,
            }),
            .ssm_delta_net => log.info("{s}: gpu={d:.3} ms/iter wall={d:.3} ms/iter overhead={d:.3} ms/iter | approx {d:.1} GB/s | {d:.1}% of peak | {d} B/iter | d_inner={d} dt_rank={d} d_state={d} n_group={d}", .{
                result.name,
                result.gpu_ms_per_iter,
                result.wall_ms_per_iter,
                result.overhead_ms_per_iter,
                result.effective_gbps,
                result.utilization_pct,
                result.bytes_per_iter,
                result.d_inner,
                result.dt_rank,
                result.d_state,
                result.n_group,
            }),
        }
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer {
        const leaked = gpa.deinit();
        if (leaked == .leak) log.err("memory leak detected", .{});
    }
    const allocator = gpa.allocator();

    var args = try parseArgs(allocator);
    defer args.deinit(allocator);
    const shader_dir = try resolveShaderDir(allocator, args.shader_dir);
    defer allocator.free(shader_dir);

    const model_cfg = try loadBenchModelConfig(args.model_path, allocator);
    const cases = buildCases(model_cfg);

    var instance = try Instance.init(allocator, args.device);
    defer instance.deinit();
    const gpu_cfg = gpu_detect.detect(&instance);

    var cmd_pool = try CommandPool.init(&instance);
    defer cmd_pool.deinit();
    var cmd = try CommandBuffer.init(&instance, &cmd_pool);
    defer cmd.deinit(&cmd_pool);
    var timer = try TimestampTimer.init(&instance);
    defer timer.deinit();

    var dmmv = try DmmvDispatch.init(&instance, &gpu_cfg, shader_dir, @max(model_cfg.hidden_dim, model_cfg.ssm_d_inner), allocator);
    defer dmmv.deinit();
    var elt = try ElementwiseDispatch.init(&instance, shader_dir, allocator);
    defer elt.deinit();

    log.info("GPU: {s} | BW {d} GB/s | shader_dir={s}", .{ gpu_cfg.nameSlice(), gpu_cfg.bandwidth_gbps, shader_dir });
    if (args.model_path) |model_path| {
        log.info("Shape source: {s}", .{model_path});
    } else {
        log.info("Shape source: built-in Qwen3.5-35B defaults", .{});
    }
    log.info("Working set: {d} rotating buffer sets", .{args.working_set});

    var results: std.ArrayList(BenchResult) = .{};
    defer results.deinit(allocator);

    for (cases) |case| {
        if (!matchesCaseFilter(case.name, args.case_filter)) continue;
        var desc_buf: [128]u8 = undefined;
        log.info("Running {s} ({s}) warmup={d} iterations={d}", .{
            case.name,
            case.describe(&desc_buf),
            args.warmup,
            args.iterations,
        });
        const result = switch (case.kind) {
            .dmmv_q8_0 => try runDmmvCase(allocator, &instance, instance.compute_queue, &cmd_pool, &cmd, &timer, &dmmv, &gpu_cfg, case, args.iterations, args.warmup, args.working_set),
            .ssm_delta_net => try runSsmDeltaCase(allocator, &instance, instance.compute_queue, &cmd_pool, &cmd, &timer, &elt, &gpu_cfg, case, args.iterations, args.warmup, args.working_set),
        };
        try results.append(allocator, result);
    }

    if (results.items.len == 0) return error.NoCasesMatched;

    if (args.json) {
        var stdout = std.fs.File.stdout().writerStreaming(&.{});
        defer stdout.end() catch {};
        try std.json.Stringify.value(results.items, .{ .whitespace = .indent_2 }, &stdout.interface);
        try stdout.interface.writeByte('\n');
    } else {
        printResults(results.items);
        log.info("Tip: rerun with RADV_DEBUG=shaderstats to inspect q8_0 and ssm_delta_net register/LDS pressure.", .{});
    }
}

test "q8 dmmv bytes model matches 256x2048 router shape" {
    try std.testing.expectEqual(@as(u64, 566_272), dmmvBytesPerIter(256, 2048));
}

test "bench defaults match current qwen35 hot shapes" {
    const cases = buildCases(.{});
    try std.testing.expectEqual(@as(u32, 256), cases[0].M);
    try std.testing.expectEqual(@as(u32, 512), cases[1].M);
    try std.testing.expectEqual(@as(u32, 2048), cases[2].M);
    try std.testing.expectEqual(@as(u32, 4096), cases[3].K);
    try std.testing.expectEqual(@as(u32, 32), cases[4].dt_rank);
}
