const std = @import("std");
const support = @import("zinc_bench_support");
const metal_device = support.metal_device;
const metal_loader = support.metal_loader;
const metal_command = support.metal_command;
const metal_pipeline = support.metal_pipeline;
const metal_buffer = support.metal_buffer;
const gguf = support.gguf;
const shim = support.metal_c.shim;
const process_lock = support.process_lock;
const forward_metal = support.forward_metal;

const GGMLType = gguf.GGMLType;
const MetalBuffer = metal_buffer.MetalBuffer;
const MetalPipeline = metal_pipeline.MetalPipeline;

pub const std_options = std.Options{
    .log_level = .warn,
};

const DmmvPush = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    x_offset: u32,
    y_offset: u32,
};

const DualQ8DmmvPush = extern struct {
    M0: u32,
    M1: u32,
    K: u32,
    a0_offset: u32,
    a1_offset: u32,
    x_offset: u32,
    y0_offset: u32,
    y1_offset: u32,
};

const RmsNormPush = extern struct {
    n: u32,
    eps: f32,
};

const MoeDmmvPush = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    expert_stride: u32,
    x_expert_stride: u32,
    x_offset: u32,
    y_offset: u32,
};

const CaseId = enum {
    all,
    lm_head,
    attn_q,
    attn_k,
    attn_v,
    attn_out,
    ssm_qkv,
    ssm_gate,
    ssm_dual,
    ssm_out,
    router,
    shared_gate,
    shared_up,
    shared_down,
    shared_dual,
    moe_gate,
    moe_up,
    moe_down,
};

const PipelineMode = enum {
    runtime,
    k2048,
    both,
};

const Config = struct {
    model_path: ?[]const u8 = null,
    device_index: u32 = 0,
    warmup_iterations: u32 = 25,
    iterations: u32 = 200,
    case_id: CaseId = .all,
    pipeline_mode: PipelineMode = .both,
    threadgroup_size: ?u32 = null,
    show_help: bool = false,
};

const HotCase = struct {
    key: []const u8,
    label: []const u8,
    tensor0: *const metal_loader.LoadedTensor,
    tensor1: ?*const metal_loader.LoadedTensor = null,
    norm_tensor: ?*const metal_loader.LoadedTensor = null,
    rows0: u32,
    rows1: u32 = 0,
    cols: u32,
    expert_slots: u32 = 1,
    x_expert_stride: u32 = 0,

    fn isDual(self: @This()) bool {
        return self.tensor1 != null;
    }

    fn isMoe(self: @This()) bool {
        return self.expert_slots > 1 and self.tensor1 == null;
    }

    fn totalRows(self: @This()) u32 {
        return self.rows0 + self.rows1;
    }
};

const PipelineSelection = struct {
    shader_name: []const u8,
    variant_label: []const u8,
    pipe: MetalPipeline,
    push_idx: u32,
    rows_per_wg: u32,
    block_size: u32,
};

const BenchResult = struct {
    case_key: []const u8,
    variant_label: []const u8,
    shader_name: []const u8,
    tensor_name: []const u8,
    rows: u32,
    cols: u32,
    expert_slots: u32 = 1,
    x_expert_stride: u32 = 0,
    iterations: u32,
    block_size: u32,
    rows_per_wg: u32,
    thread_execution_width: u32,
    static_threadgroup_memory_length: u32,
    weight_bytes_per_iter: u64,
    total_ms: f64,
    ms_per_iter: f64,
    gbps: f64,
    checksum: f64,
    output: []f32,
};

const DualBenchResult = struct {
    case_key: []const u8,
    variant_label: []const u8,
    shader_name: []const u8,
    tensor0_name: []const u8,
    tensor1_name: []const u8,
    rows0: u32,
    rows1: u32,
    cols: u32,
    iterations: u32,
    block_size: u32,
    rows_per_wg: u32,
    thread_execution_width: u32,
    static_threadgroup_memory_length: u32,
    weight_bytes_per_iter: u64,
    total_ms: f64,
    ms_per_iter: f64,
    gbps: f64,
    checksum0: f64,
    checksum1: f64,
    output0: []f32,
    output1: []f32,
};

const CompareSummary = struct {
    max_abs: f32,
    mean_abs: f64,
};

fn helpText() []const u8 {
    return
        \\Usage: zinc-bench-metal-shapes -m <model.gguf> [options]
        \\
        \\Benchmarks the exact local hot q8_0 Metal shapes from the real GGUF model.
        \\
        \\Options:
        \\  -m, --model <path>         GGUF model path (required)
        \\  -d, --device <index>       Metal device index (default: 0)
        \\  --case <name>              all | lm_head | attn_q | attn_k | attn_v | attn_out
        \\                            | ssm_qkv | ssm_gate | ssm_dual | ssm_out
        \\                            | router | shared_gate | shared_up | shared_down | shared_dual
        \\                            | moe_gate | moe_up | moe_down
        \\  --pipeline <mode>          runtime | k2048 | both (default: both)
        \\  --iterations <n>           Timed dispatches per case (default: 200)
        \\  --warmup <n>               Warmup dispatches per case (default: 25)
        \\  --tg <threads>             Override threadgroup size
        \\  -h, --help                 Show this help text
        \\
        \\Examples:
        \\  zig build bench-metal-shapes -- -m /Users/zolotukhin/models/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf
        \\  zig build bench-metal-shapes -- -m model.gguf --case lm_head --pipeline both
        \\
    ;
}

fn parseU32(arg: []const u8) !u32 {
    return std.fmt.parseUnsigned(u32, arg, 10);
}

fn parseCaseId(arg: []const u8) !CaseId {
    if (std.mem.eql(u8, arg, "all")) return .all;
    if (std.mem.eql(u8, arg, "lm_head")) return .lm_head;
    if (std.mem.eql(u8, arg, "attn_q")) return .attn_q;
    if (std.mem.eql(u8, arg, "attn_k")) return .attn_k;
    if (std.mem.eql(u8, arg, "attn_v")) return .attn_v;
    if (std.mem.eql(u8, arg, "attn_out")) return .attn_out;
    if (std.mem.eql(u8, arg, "ssm_qkv")) return .ssm_qkv;
    if (std.mem.eql(u8, arg, "ssm_gate")) return .ssm_gate;
    if (std.mem.eql(u8, arg, "ssm_dual")) return .ssm_dual;
    if (std.mem.eql(u8, arg, "ssm_out")) return .ssm_out;
    if (std.mem.eql(u8, arg, "router")) return .router;
    if (std.mem.eql(u8, arg, "shared_gate")) return .shared_gate;
    if (std.mem.eql(u8, arg, "shared_up")) return .shared_up;
    if (std.mem.eql(u8, arg, "shared_down")) return .shared_down;
    if (std.mem.eql(u8, arg, "shared_dual")) return .shared_dual;
    if (std.mem.eql(u8, arg, "moe_gate")) return .moe_gate;
    if (std.mem.eql(u8, arg, "moe_up")) return .moe_up;
    if (std.mem.eql(u8, arg, "moe_down")) return .moe_down;
    return error.InvalidCase;
}

fn parsePipelineMode(arg: []const u8) !PipelineMode {
    if (std.mem.eql(u8, arg, "runtime")) return .runtime;
    if (std.mem.eql(u8, arg, "k2048")) return .k2048;
    if (std.mem.eql(u8, arg, "both")) return .both;
    return error.InvalidPipelineMode;
}

fn parseArgs(args: []const [:0]const u8) !Config {
    var config = Config{};
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            config.show_help = true;
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return error.MissingModelPath;
            config.model_path = args[i];
        } else if (std.mem.eql(u8, arg, "-d") or std.mem.eql(u8, arg, "--device")) {
            i += 1;
            if (i >= args.len) return error.MissingDeviceIndex;
            config.device_index = try parseU32(args[i]);
        } else if (std.mem.eql(u8, arg, "--case")) {
            i += 1;
            if (i >= args.len) return error.MissingCase;
            config.case_id = try parseCaseId(args[i]);
        } else if (std.mem.eql(u8, arg, "--pipeline")) {
            i += 1;
            if (i >= args.len) return error.MissingPipelineMode;
            config.pipeline_mode = try parsePipelineMode(args[i]);
        } else if (std.mem.eql(u8, arg, "--iterations")) {
            i += 1;
            if (i >= args.len) return error.MissingIterations;
            config.iterations = try parseU32(args[i]);
        } else if (std.mem.eql(u8, arg, "--warmup")) {
            i += 1;
            if (i >= args.len) return error.MissingWarmupIterations;
            config.warmup_iterations = try parseU32(args[i]);
        } else if (std.mem.eql(u8, arg, "--tg")) {
            i += 1;
            if (i >= args.len) return error.MissingThreadgroupSize;
            config.threadgroup_size = try parseU32(args[i]);
        } else {
            return error.UnknownArgument;
        }
    }

    if (!config.show_help and config.model_path == null) return error.MissingModelPath;
    if (config.iterations == 0) return error.InvalidIterations;
    return config;
}

fn loadShaderPipeline(ctx: ?*shim.MetalCtx, name: []const u8) !MetalPipeline {
    var path_buf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "src/shaders/metal/{s}.metal", .{name}) catch return error.PathTooLong;
    const file = std.fs.cwd().openFile(path, .{}) catch return error.ShaderNotFound;
    defer file.close();
    const stat = try file.stat();
    if (stat.size > 1024 * 1024) return error.ShaderTooLarge;

    var source_buf: [1024 * 1024 + 1]u8 = undefined;
    const bytes_read = try file.readAll(source_buf[0..source_buf.len - 1]);
    source_buf[bytes_read] = 0;

    var fn_buf: [16]u8 = undefined;
    const fn_name = try std.fmt.bufPrintZ(&fn_buf, "main0", .{});
    return metal_pipeline.createPipeline(ctx, @ptrCast(&source_buf), fn_name);
}

fn tensorRows(tensor: *const metal_loader.LoadedTensor) !u32 {
    if (tensor.info.n_dims < 2) return error.InvalidTensorShape;
    return std.math.cast(u32, tensor.info.dims[1]) orelse error.InvalidTensorShape;
}

fn tensorCols(tensor: *const metal_loader.LoadedTensor) !u32 {
    if (tensor.info.n_dims < 1) return error.InvalidTensorShape;
    return std.math.cast(u32, tensor.info.dims[0]) orelse error.InvalidTensorShape;
}

fn findTensorByName(model: *const metal_loader.Model, name: []const u8) ?*const metal_loader.LoadedTensor {
    for (model.tensors.items) |*tensor| {
        if (std.mem.eql(u8, tensor.info.name, name)) return tensor;
    }
    return null;
}

fn parseLayerIndex(name: []const u8) ?u32 {
    if (!std.mem.startsWith(u8, name, "blk.")) return null;
    const rest = name[4..];
    const end = std.mem.indexOfScalar(u8, rest, '.') orelse return null;
    return std.fmt.parseUnsigned(u32, rest[0..end], 10) catch null;
}

fn findLayerTensor(model: *const metal_loader.Model, layer: u32, suffix: []const u8) ?*const metal_loader.LoadedTensor {
    var name_buf: [128]u8 = undefined;
    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.{s}", .{ layer, suffix }) catch return null;
    return findTensorByName(model, name);
}

fn findTensorBySuffixAndShape(
    model: *const metal_loader.Model,
    suffix: []const u8,
    quant_type: GGMLType,
    expected_rows: u32,
    expected_cols: u32,
) ?*const metal_loader.LoadedTensor {
    for (model.tensors.items) |*tensor| {
        if (tensor.info.type_ != quant_type) continue;
        if (!std.mem.endsWith(u8, tensor.info.name, suffix)) continue;
        const rows = tensorRows(tensor) catch continue;
        const cols = tensorCols(tensor) catch continue;
        if (rows == expected_rows and cols == expected_cols) return tensor;
    }
    return null;
}

fn resolveHotCase(model: *const metal_loader.Model, case_id: CaseId) !HotCase {
    return switch (case_id) {
        .lm_head => blk: {
            const tensor = findTensorByName(model, "output.weight") orelse
                findTensorByName(model, "token_embd.weight") orelse return error.MissingLmHeadTensor;
            if (tensor.info.type_ != .q8_0) return error.ExpectedQ8Tensor;
            break :blk .{
                .key = "lm_head",
                .label = "LM head",
                .tensor0 = tensor,
                .rows0 = try tensorRows(tensor),
                .cols = try tensorCols(tensor),
            };
        },
        .attn_q => blk: {
            const tensor = findLayerTensor(model, 3, "attn_q.weight") orelse return error.MissingTensor;
            if (tensor.info.type_ != .q8_0) return error.ExpectedQ8Tensor;
            break :blk .{
                .key = "attn_q",
                .label = "Full attention Q",
                .tensor0 = tensor,
                .rows0 = try tensorRows(tensor),
                .cols = try tensorCols(tensor),
            };
        },
        .attn_k => blk: {
            const tensor = findLayerTensor(model, 3, "attn_k.weight") orelse return error.MissingTensor;
            if (tensor.info.type_ != .q8_0) return error.ExpectedQ8Tensor;
            break :blk .{
                .key = "attn_k",
                .label = "Full attention K",
                .tensor0 = tensor,
                .rows0 = try tensorRows(tensor),
                .cols = try tensorCols(tensor),
            };
        },
        .attn_v => blk: {
            const tensor = findLayerTensor(model, 3, "attn_v.weight") orelse return error.MissingTensor;
            if (tensor.info.type_ != .q8_0) return error.ExpectedQ8Tensor;
            break :blk .{
                .key = "attn_v",
                .label = "Full attention V",
                .tensor0 = tensor,
                .rows0 = try tensorRows(tensor),
                .cols = try tensorCols(tensor),
            };
        },
        .attn_out => blk: {
            const tensor = findLayerTensor(model, 3, "attn_output.weight") orelse return error.MissingTensor;
            if (tensor.info.type_ != .q8_0) return error.ExpectedQ8Tensor;
            break :blk .{
                .key = "attn_out",
                .label = "Full attention output",
                .tensor0 = tensor,
                .rows0 = try tensorRows(tensor),
                .cols = try tensorCols(tensor),
            };
        },
        .ssm_qkv => blk: {
            const tensor = findTensorBySuffixAndShape(model, "attn_qkv.weight", .q8_0, 8192, 2048) orelse
                return error.MissingSsmQkvTensor;
            break :blk .{
                .key = "ssm_qkv",
                .label = "SSM qkv",
                .tensor0 = tensor,
                .rows0 = 8192,
                .cols = 2048,
            };
        },
        .ssm_gate => blk: {
            const tensor = findTensorBySuffixAndShape(model, "attn_gate.weight", .q8_0, 4096, 2048) orelse
                return error.MissingSsmGateTensor;
            break :blk .{
                .key = "ssm_gate",
                .label = "SSM gate",
                .tensor0 = tensor,
                .rows0 = 4096,
                .cols = 2048,
            };
        },
        .ssm_dual => blk: {
            const qkv = findTensorBySuffixAndShape(model, "attn_qkv.weight", .q8_0, 8192, 2048) orelse
                return error.MissingSsmQkvTensor;
            const gate = findTensorBySuffixAndShape(model, "attn_gate.weight", .q8_0, 4096, 2048) orelse
                return error.MissingSsmGateTensor;
            const layer = parseLayerIndex(qkv.info.name) orelse return error.InvalidTensorName;
            const norm = findLayerTensor(model, layer, "attn_norm.weight") orelse
                return error.MissingAttnNormTensor;
            break :blk .{
                .key = "ssm_dual",
                .label = "SSM qkv + gate",
                .tensor0 = qkv,
                .tensor1 = gate,
                .norm_tensor = norm,
                .rows0 = 8192,
                .rows1 = 4096,
                .cols = 2048,
            };
        },
        .ssm_out => blk: {
            const tensor = findTensorBySuffixAndShape(model, "ssm_out.weight", .q8_0, 2048, 4096) orelse
                return error.MissingSsmOutTensor;
            break :blk .{
                .key = "ssm_out",
                .label = "SSM out",
                .tensor0 = tensor,
                .rows0 = 2048,
                .cols = 4096,
            };
        },
        .router => blk: {
            const tensor = findLayerTensor(model, 0, "ffn_gate_inp.weight") orelse
                return error.MissingRouterTensor;
            if (tensor.info.type_ != .q8_0) return error.ExpectedQ8Tensor;
            const rows = try tensorRows(tensor);
            const cols = try tensorCols(tensor);
            break :blk .{
                .key = "router",
                .label = "MoE router",
                .tensor0 = tensor,
                .rows0 = rows,
                .cols = cols,
            };
        },
        .shared_gate => blk: {
            const inter_dim = model.config.shared_expert_intermediate_dim;
            const hidden_dim = model.config.hidden_dim;
            const tensor = findTensorBySuffixAndShape(model, "ffn_gate_shexp.weight", .q8_0, inter_dim, hidden_dim) orelse
                return error.MissingSharedGateTensor;
            break :blk .{
                .key = "shared_gate",
                .label = "Shared expert gate",
                .tensor0 = tensor,
                .rows0 = inter_dim,
                .cols = hidden_dim,
            };
        },
        .shared_up => blk: {
            const inter_dim = model.config.shared_expert_intermediate_dim;
            const hidden_dim = model.config.hidden_dim;
            const tensor = findTensorBySuffixAndShape(model, "ffn_up_shexp.weight", .q8_0, inter_dim, hidden_dim) orelse
                return error.MissingSharedUpTensor;
            break :blk .{
                .key = "shared_up",
                .label = "Shared expert up",
                .tensor0 = tensor,
                .rows0 = inter_dim,
                .cols = hidden_dim,
            };
        },
        .shared_down => blk: {
            const inter_dim = model.config.shared_expert_intermediate_dim;
            const hidden_dim = model.config.hidden_dim;
            const tensor = findTensorBySuffixAndShape(model, "ffn_down_shexp.weight", .q8_0, hidden_dim, inter_dim) orelse
                return error.MissingSharedDownTensor;
            break :blk .{
                .key = "shared_down",
                .label = "Shared expert down",
                .tensor0 = tensor,
                .rows0 = hidden_dim,
                .cols = inter_dim,
            };
        },
        .shared_dual => blk: {
            const inter_dim = model.config.shared_expert_intermediate_dim;
            const hidden_dim = model.config.hidden_dim;
            const gate = findTensorBySuffixAndShape(model, "ffn_gate_shexp.weight", .q8_0, inter_dim, hidden_dim) orelse
                return error.MissingSharedGateTensor;
            const up = findTensorBySuffixAndShape(model, "ffn_up_shexp.weight", .q8_0, inter_dim, hidden_dim) orelse
                return error.MissingSharedUpTensor;
            break :blk .{
                .key = "shared_dual",
                .label = "Shared expert gate + up",
                .tensor0 = gate,
                .tensor1 = up,
                .rows0 = inter_dim,
                .rows1 = inter_dim,
                .cols = hidden_dim,
            };
        },
        .moe_gate => blk: {
            const inter_dim = model.config.intermediate_dim;
            const hidden_dim = model.config.hidden_dim;
            const tensor = findTensorBySuffixAndShape(model, "ffn_gate_exps.weight", .q4_k, inter_dim, hidden_dim) orelse
                findTensorBySuffixAndShape(model, "ffn_gate_exps.weight", .q5_k, inter_dim, hidden_dim) orelse
                findTensorBySuffixAndShape(model, "ffn_gate_exps.weight", .q6_k, inter_dim, hidden_dim) orelse
                return error.MissingMoeGateTensor;
            break :blk .{
                .key = "moe_gate",
                .label = "MoE gate experts",
                .tensor0 = tensor,
                .rows0 = inter_dim,
                .cols = hidden_dim,
                .expert_slots = model.config.n_experts_used,
                .x_expert_stride = 0,
            };
        },
        .moe_up => blk: {
            const inter_dim = model.config.intermediate_dim;
            const hidden_dim = model.config.hidden_dim;
            const tensor = findTensorBySuffixAndShape(model, "ffn_up_exps.weight", .q4_k, inter_dim, hidden_dim) orelse
                findTensorBySuffixAndShape(model, "ffn_up_exps.weight", .q5_k, inter_dim, hidden_dim) orelse
                findTensorBySuffixAndShape(model, "ffn_up_exps.weight", .q6_k, inter_dim, hidden_dim) orelse
                return error.MissingMoeUpTensor;
            break :blk .{
                .key = "moe_up",
                .label = "MoE up experts",
                .tensor0 = tensor,
                .rows0 = inter_dim,
                .cols = hidden_dim,
                .expert_slots = model.config.n_experts_used,
                .x_expert_stride = 0,
            };
        },
        .moe_down => blk: {
            const inter_dim = model.config.intermediate_dim;
            const hidden_dim = model.config.hidden_dim;
            const tensor = findTensorBySuffixAndShape(model, "ffn_down_exps.weight", .q4_k, hidden_dim, inter_dim) orelse
                findTensorBySuffixAndShape(model, "ffn_down_exps.weight", .q5_k, hidden_dim, inter_dim) orelse
                findTensorBySuffixAndShape(model, "ffn_down_exps.weight", .q6_k, hidden_dim, inter_dim) orelse
                return error.MissingMoeDownTensor;
            break :blk .{
                .key = "moe_down",
                .label = "MoE down experts",
                .tensor0 = tensor,
                .rows0 = hidden_dim,
                .cols = inter_dim,
                .expert_slots = model.config.n_experts_used,
                .x_expert_stride = inter_dim,
            };
        },
        .all => error.InvalidCase,
    };
}

fn tensorPageOffset(model: *const metal_loader.Model, tensor: *const metal_loader.LoadedTensor) u32 {
    const data_offset: u64 = model.gguf_file.tensor_data_offset + tensor.info.offset;
    const aligned_offset = (data_offset / 4096) * 4096;
    return @intCast(data_offset - aligned_offset);
}

fn weightBytesPerIter(quant_type: GGMLType, rows: u32, cols: u32) u64 {
    const block_size = quant_type.blockSize();
    const bytes_per_block = quant_type.bytesPerBlock();
    if (block_size == 0 or bytes_per_block == 0) {
        return @as(u64, rows) * @as(u64, cols) * @sizeOf(f32);
    }
    return @as(u64, rows) * @as(u64, cols / block_size) * @as(u64, bytes_per_block);
}

fn fillInputBuffer(buf: *MetalBuffer, cols: u32) void {
    const ptr: [*]f32 = @ptrCast(@alignCast(buf.cpu_ptr.?));
    for (0..cols) |i| {
        const raw: i32 = @intCast((i * 13 + 7) % 29);
        ptr[i] = 0.125 * @as(f32, @floatFromInt(raw - 14));
    }
}

fn fillMoeInputBuffer(buf: *MetalBuffer, cols: u32, expert_slots: u32, x_expert_stride: u32) void {
    if (x_expert_stride == 0) {
        fillInputBuffer(buf, cols);
        return;
    }

    const ptr: [*]f32 = @ptrCast(@alignCast(buf.cpu_ptr.?));
    for (0..expert_slots) |slot| {
        const base = slot * x_expert_stride;
        for (0..cols) |i| {
            const raw: i32 = @intCast((slot * 19 + i * 13 + 7) % 41);
            ptr[base + i] = 0.0625 * @as(f32, @floatFromInt(raw - 20));
        }
    }
}

fn fillRoutingBuffer(buf: *MetalBuffer, n_experts: u32, expert_slots: u32) void {
    const ptr: [*]u32 = @ptrCast(@alignCast(buf.cpu_ptr.?));
    for (0..expert_slots) |slot| {
        ptr[slot] = @intCast((slot * 37 + 11) % n_experts);
    }
}

fn loadNormWeightsBuffer(
    ctx: ?*shim.MetalCtx,
    model: *const metal_loader.Model,
    tensor: *const metal_loader.LoadedTensor,
    cols: u32,
) !MetalBuffer {
    const mmap = model.mmap_data orelse return error.NoMmapData;
    const buf = try metal_buffer.createBuffer(ctx, @as(usize, cols) * @sizeOf(f32));
    const dst: [*]f32 = @ptrCast(@alignCast(buf.cpu_ptr.?));
    const data_offset: usize = @intCast(model.gguf_file.tensor_data_offset + tensor.info.offset);
    forward_metal.dequantRow(mmap[data_offset..], 0, cols, tensor.info.type_, dst[0..cols]);
    return buf;
}

fn chooseBlockSize(pipe: *const MetalPipeline, cols: u32, override: ?u32) !u32 {
    const simd_width = if (pipe.thread_execution_width > 0) pipe.thread_execution_width else @as(u32, 32);
    if (override) |tg| {
        if (tg == 0 or tg > pipe.max_threads_per_threadgroup) return error.InvalidThreadgroupSize;
        if (tg % simd_width != 0) return error.InvalidThreadgroupSize;
        return tg;
    }
    if (cols <= 4096 and pipe.thread_execution_width == 32 and pipe.max_threads_per_threadgroup >= 256) {
        return 256;
    }
    return 64;
}

fn selectPipeline(ctx: ?*shim.MetalCtx, variant: PipelineMode, cols: u32, threadgroup_override: ?u32) !PipelineSelection {
    return switch (variant) {
        .runtime => blk: {
            var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0");
            const block_size = try chooseBlockSize(&pipe, cols, threadgroup_override);
            const simd_width = if (pipe.thread_execution_width > 0) pipe.thread_execution_width else @as(u32, 32);
            break :blk .{
                .shader_name = "dmmv_q8_0",
                .variant_label = "runtime",
                .pipe = pipe,
                .push_idx = 0,
                .rows_per_wg = block_size / simd_width,
                .block_size = block_size,
            };
        },
        .k2048 => blk: {
            if (cols > 2048) return error.K2048OnlySupportsK2048OrLess;
            var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_k2048");
            const block_size = try chooseBlockSize(&pipe, cols, threadgroup_override);
            const simd_width = if (pipe.thread_execution_width > 0) pipe.thread_execution_width else @as(u32, 32);
            break :blk .{
                .shader_name = "dmmv_q8_0_k2048",
                .variant_label = "k2048",
                .pipe = pipe,
                .push_idx = 0,
                .rows_per_wg = block_size / simd_width,
                .block_size = block_size,
            };
        },
        .both => unreachable,
    };
}

fn selectDualPipeline(ctx: ?*shim.MetalCtx, cols: u32, threadgroup_override: ?u32) !PipelineSelection {
    const pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_dual");
    const simd_width = if (pipe.thread_execution_width > 0) pipe.thread_execution_width else @as(u32, 32);
    const block_size = if (threadgroup_override) |tg| blk: {
        if (tg == 0 or tg > pipe.max_threads_per_threadgroup) return error.InvalidThreadgroupSize;
        if (tg % simd_width != 0) return error.InvalidThreadgroupSize;
        break :blk tg;
    } else blk: {
        if (cols <= 2048 and pipe.max_threads_per_threadgroup >= 512) break :blk @as(u32, 512);
        break :blk simd_width;
    };
    return .{
        .shader_name = "dmmv_q8_0_dual",
        .variant_label = "dual",
        .pipe = pipe,
        .push_idx = 0,
        .rows_per_wg = block_size / simd_width,
        .block_size = block_size,
    };
}

fn selectFusedDualPipeline(ctx: ?*shim.MetalCtx, cols: u32, threadgroup_override: ?u32) !PipelineSelection {
    const pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_dual_fused_norm");
    const simd_width = if (pipe.thread_execution_width > 0) pipe.thread_execution_width else @as(u32, 32);
    const block_size = if (threadgroup_override) |tg| blk: {
        if (tg == 0 or tg > pipe.max_threads_per_threadgroup) return error.InvalidThreadgroupSize;
        if (tg % simd_width != 0) return error.InvalidThreadgroupSize;
        break :blk tg;
    } else blk: {
        if (cols <= 2048 and pipe.max_threads_per_threadgroup >= 512) break :blk @as(u32, 512);
        break :blk simd_width;
    };
    return .{
        .shader_name = "dmmv_q8_0_dual_fused_norm",
        .variant_label = "fused",
        .pipe = pipe,
        .push_idx = 0,
        .rows_per_wg = block_size / simd_width,
        .block_size = block_size,
    };
}

fn selectMoePipeline(
    ctx: ?*shim.MetalCtx,
    chip: metal_device.GpuFamily,
    quant_type: GGMLType,
    variant: PipelineMode,
    rows: u32,
    cols: u32,
    x_expert_stride: u32,
) !PipelineSelection {
    return switch (quant_type) {
        .q4_k => switch (variant) {
            .runtime => if (cols <= 2048 and chip.isM5Class() and x_expert_stride != 0 and rows >= 1024) .{
                .shader_name = "dmmv_q4k_moe_k2048_1024",
                .variant_label = "runtime",
                .pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe_k2048_1024"),
                .push_idx = 1,
                .rows_per_wg = 32,
                .block_size = 1024,
            } else if (cols <= 2048) .{
                .shader_name = "dmmv_q4k_moe_k2048",
                .variant_label = "runtime",
                .pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe_k2048"),
                .push_idx = 1,
                .rows_per_wg = 16,
                .block_size = 512,
            } else .{
                .shader_name = "dmmv_q4k_moe",
                .variant_label = "runtime",
                .pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe"),
                .push_idx = 1,
                .rows_per_wg = 8,
                .block_size = 256,
            },
            .k2048 => if (cols <= 2048) .{
                .shader_name = "dmmv_q4k_moe_k2048",
                .variant_label = "k2048",
                .pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe_k2048"),
                .push_idx = 1,
                .rows_per_wg = 16,
                .block_size = 512,
            } else return error.K2048OnlySupportsK2048OrLess,
            .both => if (cols <= 2048 and x_expert_stride != 0 and rows >= 1024) .{
                .shader_name = "dmmv_q4k_moe_k2048_1024",
                .variant_label = "k2048_1024",
                .pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe_k2048_1024"),
                .push_idx = 1,
                .rows_per_wg = 32,
                .block_size = 1024,
            } else .{
                .shader_name = "dmmv_q4k_moe",
                .variant_label = "base",
                .pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe"),
                .push_idx = 1,
                .rows_per_wg = 8,
                .block_size = 256,
            },
        },
        .q5_k => switch (variant) {
            .runtime => if (cols <= 2048) .{
                .shader_name = "dmmv_q5k_moe_k2048",
                .variant_label = "runtime",
                .pipe = try loadShaderPipeline(ctx, "dmmv_q5k_moe_k2048"),
                .push_idx = 1,
                .rows_per_wg = 16,
                .block_size = 512,
            } else .{
                .shader_name = "dmmv_q5k_moe",
                .variant_label = "runtime",
                .pipe = try loadShaderPipeline(ctx, "dmmv_q5k_moe"),
                .push_idx = 1,
                .rows_per_wg = 8,
                .block_size = 256,
            },
            .k2048 => if (cols <= 2048) .{
                .shader_name = "dmmv_q5k_moe_k2048",
                .variant_label = "k2048",
                .pipe = try loadShaderPipeline(ctx, "dmmv_q5k_moe_k2048"),
                .push_idx = 1,
                .rows_per_wg = 16,
                .block_size = 512,
            } else return error.K2048OnlySupportsK2048OrLess,
            .both => .{
                .shader_name = "dmmv_q5k_moe",
                .variant_label = "base",
                .pipe = try loadShaderPipeline(ctx, "dmmv_q5k_moe"),
                .push_idx = 1,
                .rows_per_wg = 8,
                .block_size = 256,
            },
        },
        .q6_k => .{
            .shader_name = "dmmv_q6k_moe",
            .variant_label = if (variant == .runtime) "runtime" else "base",
            .pipe = try loadShaderPipeline(ctx, "dmmv_q6k_moe"),
            .push_idx = 1,
            .rows_per_wg = 8,
            .block_size = 256,
        },
        else => error.ExpectedExpertQuantTensor,
    };
}

fn runDispatchBatch(
    ctx: ?*shim.MetalCtx,
    selection: *const PipelineSelection,
    tensor: *const metal_loader.LoadedTensor,
    model: *const metal_loader.Model,
    input_buf: *const MetalBuffer,
    output_buf: *const MetalBuffer,
    rows: u32,
    cols: u32,
    dispatches: u32,
) !void {
    if (dispatches == 0) return;
    const push = DmmvPush{
        .M = rows,
        .K = cols,
        .a_offset = tensorPageOffset(model, tensor),
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &tensor.gpu_buffer, input_buf, output_buf };
    const workgroups = (rows + selection.rows_per_wg - 1) / selection.rows_per_wg;

    var cmd = try metal_command.beginCommand(ctx);
    for (0..dispatches) |_| {
        cmd.dispatchV2(
            &selection.pipe,
            .{ workgroups, 1, 1 },
            .{ selection.block_size, 1, 1 },
            &bufs,
            &push,
            @sizeOf(DmmvPush),
            selection.push_idx,
        );
    }
    cmd.commitAndWait();
}

fn runMoeDispatchBatch(
    ctx: ?*shim.MetalCtx,
    selection: *const PipelineSelection,
    tensor: *const metal_loader.LoadedTensor,
    model: *const metal_loader.Model,
    input_buf: *const MetalBuffer,
    output_buf: *const MetalBuffer,
    routing_buf: *const MetalBuffer,
    rows: u32,
    cols: u32,
    expert_slots: u32,
    x_expert_stride: u32,
    dispatches: u32,
) !void {
    if (dispatches == 0) return;
    const push = MoeDmmvPush{
        .M = rows,
        .K = cols,
        .a_offset = tensorPageOffset(model, tensor),
        .expert_stride = @intCast(weightBytesPerIter(tensor.info.type_, rows, cols)),
        .x_expert_stride = x_expert_stride,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &tensor.gpu_buffer, input_buf, output_buf, routing_buf };
    const workgroups = (rows + selection.rows_per_wg - 1) / selection.rows_per_wg;

    var cmd = try metal_command.beginCommand(ctx);
    for (0..dispatches) |_| {
        cmd.dispatchV2(
            &selection.pipe,
            .{ workgroups, expert_slots, 1 },
            .{ selection.block_size, 1, 1 },
            &bufs,
            &push,
            @sizeOf(MoeDmmvPush),
            selection.push_idx,
        );
    }
    cmd.commitAndWait();
}

fn runSeparateDualDispatchBatch(
    ctx: ?*shim.MetalCtx,
    selection: *const PipelineSelection,
    tensor0: *const metal_loader.LoadedTensor,
    tensor1: *const metal_loader.LoadedTensor,
    model: *const metal_loader.Model,
    input_buf: *const MetalBuffer,
    output0_buf: *const MetalBuffer,
    output1_buf: *const MetalBuffer,
    rows0: u32,
    rows1: u32,
    cols: u32,
    dispatches: u32,
) !void {
    if (dispatches == 0) return;
    const push0 = DmmvPush{
        .M = rows0,
        .K = cols,
        .a_offset = tensorPageOffset(model, tensor0),
        .x_offset = 0,
        .y_offset = 0,
    };
    const push1 = DmmvPush{
        .M = rows1,
        .K = cols,
        .a_offset = tensorPageOffset(model, tensor1),
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs0 = [_]*const MetalBuffer{ &tensor0.gpu_buffer, input_buf, output0_buf };
    const bufs1 = [_]*const MetalBuffer{ &tensor1.gpu_buffer, input_buf, output1_buf };
    const workgroups0 = (rows0 + selection.rows_per_wg - 1) / selection.rows_per_wg;
    const workgroups1 = (rows1 + selection.rows_per_wg - 1) / selection.rows_per_wg;

    var cmd = try metal_command.beginCommand(ctx);
    for (0..dispatches) |_| {
        cmd.dispatchV2(
            &selection.pipe,
            .{ workgroups0, 1, 1 },
            .{ selection.block_size, 1, 1 },
            &bufs0,
            &push0,
            @sizeOf(DmmvPush),
            selection.push_idx,
        );
        cmd.dispatchV2(
            &selection.pipe,
            .{ workgroups1, 1, 1 },
            .{ selection.block_size, 1, 1 },
            &bufs1,
            &push1,
            @sizeOf(DmmvPush),
            selection.push_idx,
        );
    }
    cmd.commitAndWait();
}

fn runDualDispatchBatch(
    ctx: ?*shim.MetalCtx,
    selection: *const PipelineSelection,
    tensor0: *const metal_loader.LoadedTensor,
    tensor1: *const metal_loader.LoadedTensor,
    model: *const metal_loader.Model,
    input_buf: *const MetalBuffer,
    output0_buf: *const MetalBuffer,
    output1_buf: *const MetalBuffer,
    rows0: u32,
    rows1: u32,
    cols: u32,
    dispatches: u32,
) !void {
    if (dispatches == 0) return;
    const push = DualQ8DmmvPush{
        .M0 = rows0,
        .M1 = rows1,
        .K = cols,
        .a0_offset = tensorPageOffset(model, tensor0),
        .a1_offset = tensorPageOffset(model, tensor1),
        .x_offset = 0,
        .y0_offset = 0,
        .y1_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &tensor0.gpu_buffer, &tensor1.gpu_buffer, input_buf, output0_buf, output1_buf };
    const workgroups = (rows0 + rows1 + selection.rows_per_wg - 1) / selection.rows_per_wg;

    var cmd = try metal_command.beginCommand(ctx);
    for (0..dispatches) |_| {
        cmd.dispatchV2(
            &selection.pipe,
            .{ workgroups, 1, 1 },
            .{ selection.block_size, 1, 1 },
            &bufs,
            &push,
            @sizeOf(DualQ8DmmvPush),
            selection.push_idx,
        );
    }
    cmd.commitAndWait();
}

fn runRmsNormDualDispatchBatch(
    ctx: ?*shim.MetalCtx,
    rms_norm_pipe: *const MetalPipeline,
    dual_selection: *const PipelineSelection,
    tensor0: *const metal_loader.LoadedTensor,
    tensor1: *const metal_loader.LoadedTensor,
    model: *const metal_loader.Model,
    hidden_buf: *const MetalBuffer,
    norm_buf: *const MetalBuffer,
    norm_weight_buf: *const MetalBuffer,
    output0_buf: *const MetalBuffer,
    output1_buf: *const MetalBuffer,
    rows0: u32,
    rows1: u32,
    cols: u32,
    dispatches: u32,
) !void {
    if (dispatches == 0) return;
    const rms_push = RmsNormPush{ .n = cols, .eps = 1e-6 };
    const rms_bufs = [_]*const MetalBuffer{ hidden_buf, norm_buf, norm_weight_buf };
    const rms_tg = if (rms_norm_pipe.thread_execution_width > 0 and
        rms_norm_pipe.thread_execution_width <= rms_norm_pipe.max_threads_per_threadgroup)
        rms_norm_pipe.thread_execution_width
    else
        @as(u32, 32);

    const dual_push = DualQ8DmmvPush{
        .M0 = rows0,
        .M1 = rows1,
        .K = cols,
        .a0_offset = tensorPageOffset(model, tensor0),
        .a1_offset = tensorPageOffset(model, tensor1),
        .x_offset = 0,
        .y0_offset = 0,
        .y1_offset = 0,
    };
    const dual_bufs = [_]*const MetalBuffer{ &tensor0.gpu_buffer, &tensor1.gpu_buffer, norm_buf, output0_buf, output1_buf };
    const dual_workgroups = (rows0 + rows1 + dual_selection.rows_per_wg - 1) / dual_selection.rows_per_wg;

    var cmd = try metal_command.beginCommand(ctx);
    for (0..dispatches) |_| {
        cmd.dispatchV2(
            rms_norm_pipe,
            .{ 1, 1, 1 },
            .{ rms_tg, 1, 1 },
            &rms_bufs,
            &rms_push,
            @sizeOf(RmsNormPush),
            0,
        );
        cmd.barrier();
        cmd.dispatchV2(
            &dual_selection.pipe,
            .{ dual_workgroups, 1, 1 },
            .{ dual_selection.block_size, 1, 1 },
            &dual_bufs,
            &dual_push,
            @sizeOf(DualQ8DmmvPush),
            dual_selection.push_idx,
        );
    }
    cmd.commitAndWait();
}

fn runFusedDualDispatchBatch(
    ctx: ?*shim.MetalCtx,
    selection: *const PipelineSelection,
    tensor0: *const metal_loader.LoadedTensor,
    tensor1: *const metal_loader.LoadedTensor,
    model: *const metal_loader.Model,
    hidden_buf: *const MetalBuffer,
    norm_weight_buf: *const MetalBuffer,
    output0_buf: *const MetalBuffer,
    output1_buf: *const MetalBuffer,
    rows0: u32,
    rows1: u32,
    cols: u32,
    dispatches: u32,
) !void {
    if (dispatches == 0) return;
    const push = DualQ8DmmvPush{
        .M0 = rows0,
        .M1 = rows1,
        .K = cols,
        .a0_offset = tensorPageOffset(model, tensor0),
        .a1_offset = tensorPageOffset(model, tensor1),
        .x_offset = 0,
        .y0_offset = 0,
        .y1_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &tensor0.gpu_buffer, &tensor1.gpu_buffer, hidden_buf, output0_buf, output1_buf, norm_weight_buf };
    const workgroups = (rows0 + rows1 + selection.rows_per_wg - 1) / selection.rows_per_wg;

    var cmd = try metal_command.beginCommand(ctx);
    for (0..dispatches) |_| {
        cmd.dispatchV2(
            &selection.pipe,
            .{ workgroups, 1, 1 },
            .{ selection.block_size, 1, 1 },
            &bufs,
            &push,
            @sizeOf(DualQ8DmmvPush),
            selection.push_idx,
        );
    }
    cmd.commitAndWait();
}

fn copyOutput(allocator: std.mem.Allocator, output_buf: *const MetalBuffer, rows: u32) ![]f32 {
    const ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    return try allocator.dupe(f32, ptr[0..rows]);
}

fn checksumOutput(output: []const f32) f64 {
    if (output.len == 0) return 0.0;
    const probe = @min(output.len, 32);
    var sum: f64 = 0.0;
    for (0..probe) |i| {
        sum += @as(f64, output[i]);
    }
    sum += @as(f64, output[output.len - 1]);
    return sum;
}

fn compareOutputs(lhs: []const f32, rhs: []const f32) CompareSummary {
    const n = @min(lhs.len, rhs.len);
    var max_abs: f32 = 0.0;
    var sum_abs: f64 = 0.0;
    for (0..n) |i| {
        const diff = @abs(lhs[i] - rhs[i]);
        if (diff > max_abs) max_abs = diff;
        sum_abs += @as(f64, diff);
    }
    return .{
        .max_abs = max_abs,
        .mean_abs = if (n == 0) 0.0 else sum_abs / @as(f64, @floatFromInt(n)),
    };
}

fn benchmarkVariant(
    allocator: std.mem.Allocator,
    device: *const metal_device.MetalDevice,
    model: *const metal_loader.Model,
    hot_case: HotCase,
    warmup_iterations: u32,
    iterations: u32,
    variant: PipelineMode,
    threadgroup_override: ?u32,
) !BenchResult {
    if (hot_case.isDual()) return error.DualCaseRequiresDualBenchmark;
    if (hot_case.isMoe()) return error.MoeCaseRequiresMoeBenchmark;
    var selection = try selectPipeline(device.ctx, variant, hot_case.cols, threadgroup_override);
    defer metal_pipeline.freePipeline(&selection.pipe);

    var input_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.cols) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.rows0) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    fillInputBuffer(&input_buf, hot_case.cols);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    try runDispatchBatch(device.ctx, &selection, hot_case.tensor0, model, &input_buf, &output_buf, hot_case.rows0, hot_case.cols, warmup_iterations);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    const start_ns = std.time.nanoTimestamp();
    try runDispatchBatch(device.ctx, &selection, hot_case.tensor0, model, &input_buf, &output_buf, hot_case.rows0, hot_case.cols, iterations);
    const elapsed_ns = std.time.nanoTimestamp() - start_ns;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms;
    const ms_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;
    const weight_bytes = weightBytesPerIter(hot_case.tensor0.info.type_, hot_case.rows0, hot_case.cols);
    const total_bytes = weight_bytes * iterations;
    const output_copy = try copyOutput(allocator, &output_buf, hot_case.rows0);

    return .{
        .case_key = hot_case.key,
        .variant_label = selection.variant_label,
        .shader_name = selection.shader_name,
        .tensor_name = hot_case.tensor0.info.name,
        .rows = hot_case.rows0,
        .cols = hot_case.cols,
        .expert_slots = 1,
        .x_expert_stride = 0,
        .iterations = iterations,
        .block_size = selection.block_size,
        .rows_per_wg = selection.rows_per_wg,
        .thread_execution_width = selection.pipe.thread_execution_width,
        .static_threadgroup_memory_length = selection.pipe.static_threadgroup_memory_length,
        .weight_bytes_per_iter = weight_bytes,
        .total_ms = elapsed_ms,
        .ms_per_iter = ms_per_iter,
        .gbps = (@as(f64, @floatFromInt(total_bytes)) / seconds) / 1_000_000_000.0,
        .checksum = checksumOutput(output_copy),
        .output = output_copy,
    };
}

fn benchmarkMoeVariant(
    allocator: std.mem.Allocator,
    device: *const metal_device.MetalDevice,
    model: *const metal_loader.Model,
    hot_case: HotCase,
    warmup_iterations: u32,
    iterations: u32,
    variant: PipelineMode,
) !BenchResult {
    if (!hot_case.isMoe()) return error.ExpectedMoeCase;
    var selection = try selectMoePipeline(
        device.ctx,
        device.chip,
        hot_case.tensor0.info.type_,
        variant,
        hot_case.rows0,
        hot_case.cols,
        hot_case.x_expert_stride,
    );
    defer metal_pipeline.freePipeline(&selection.pipe);

    const input_len = if (hot_case.x_expert_stride == 0)
        hot_case.cols
    else
        hot_case.x_expert_stride * hot_case.expert_slots;
    var input_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, input_len) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.rows0) * hot_case.expert_slots * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);
    var routing_buf = try metal_buffer.createBuffer(device.ctx, @max(@as(usize, hot_case.expert_slots) * @sizeOf(u32), 4));
    defer metal_buffer.freeBuffer(&routing_buf);

    fillMoeInputBuffer(&input_buf, hot_case.cols, hot_case.expert_slots, hot_case.x_expert_stride);
    fillRoutingBuffer(&routing_buf, model.config.n_experts, hot_case.expert_slots);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    try runMoeDispatchBatch(device.ctx, &selection, hot_case.tensor0, model, &input_buf, &output_buf, &routing_buf, hot_case.rows0, hot_case.cols, hot_case.expert_slots, hot_case.x_expert_stride, warmup_iterations);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    const start_ns = std.time.nanoTimestamp();
    try runMoeDispatchBatch(device.ctx, &selection, hot_case.tensor0, model, &input_buf, &output_buf, &routing_buf, hot_case.rows0, hot_case.cols, hot_case.expert_slots, hot_case.x_expert_stride, iterations);
    const elapsed_ns = std.time.nanoTimestamp() - start_ns;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms;
    const ms_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;
    const weight_bytes = weightBytesPerIter(hot_case.tensor0.info.type_, hot_case.rows0, hot_case.cols) * hot_case.expert_slots;
    const total_bytes = weight_bytes * iterations;
    const output_copy = try copyOutput(allocator, &output_buf, hot_case.rows0 * hot_case.expert_slots);

    return .{
        .case_key = hot_case.key,
        .variant_label = selection.variant_label,
        .shader_name = selection.shader_name,
        .tensor_name = hot_case.tensor0.info.name,
        .rows = hot_case.rows0,
        .cols = hot_case.cols,
        .expert_slots = hot_case.expert_slots,
        .x_expert_stride = hot_case.x_expert_stride,
        .iterations = iterations,
        .block_size = selection.block_size,
        .rows_per_wg = selection.rows_per_wg,
        .thread_execution_width = selection.pipe.thread_execution_width,
        .static_threadgroup_memory_length = selection.pipe.static_threadgroup_memory_length,
        .weight_bytes_per_iter = weight_bytes,
        .total_ms = elapsed_ms,
        .ms_per_iter = ms_per_iter,
        .gbps = (@as(f64, @floatFromInt(total_bytes)) / seconds) / 1_000_000_000.0,
        .checksum = checksumOutput(output_copy),
        .output = output_copy,
    };
}

fn benchmarkSeparateDualVariant(
    allocator: std.mem.Allocator,
    device: *const metal_device.MetalDevice,
    model: *const metal_loader.Model,
    hot_case: HotCase,
    warmup_iterations: u32,
    iterations: u32,
    threadgroup_override: ?u32,
) !DualBenchResult {
    const tensor1 = hot_case.tensor1 orelse return error.ExpectedDualCase;
    var selection = try selectPipeline(device.ctx, .runtime, hot_case.cols, threadgroup_override);
    defer metal_pipeline.freePipeline(&selection.pipe);

    var input_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.cols) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output0_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.rows0) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output0_buf);
    var output1_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.rows1) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output1_buf);

    fillInputBuffer(&input_buf, hot_case.cols);
    @memset(output0_buf.cpu_ptr.?[0..output0_buf.size], 0);
    @memset(output1_buf.cpu_ptr.?[0..output1_buf.size], 0);

    try runSeparateDualDispatchBatch(device.ctx, &selection, hot_case.tensor0, tensor1, model, &input_buf, &output0_buf, &output1_buf, hot_case.rows0, hot_case.rows1, hot_case.cols, warmup_iterations);
    @memset(output0_buf.cpu_ptr.?[0..output0_buf.size], 0);
    @memset(output1_buf.cpu_ptr.?[0..output1_buf.size], 0);

    const start_ns = std.time.nanoTimestamp();
    try runSeparateDualDispatchBatch(device.ctx, &selection, hot_case.tensor0, tensor1, model, &input_buf, &output0_buf, &output1_buf, hot_case.rows0, hot_case.rows1, hot_case.cols, iterations);
    const elapsed_ns = std.time.nanoTimestamp() - start_ns;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms;
    const ms_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;
    const weight_bytes = weightBytesPerIter(hot_case.tensor0.info.type_, hot_case.totalRows(), hot_case.cols);
    const total_bytes = weight_bytes * iterations;
    const output0_copy = try copyOutput(allocator, &output0_buf, hot_case.rows0);
    const output1_copy = try copyOutput(allocator, &output1_buf, hot_case.rows1);

    return .{
        .case_key = hot_case.key,
        .variant_label = "separate",
        .shader_name = selection.shader_name,
        .tensor0_name = hot_case.tensor0.info.name,
        .tensor1_name = tensor1.info.name,
        .rows0 = hot_case.rows0,
        .rows1 = hot_case.rows1,
        .cols = hot_case.cols,
        .iterations = iterations,
        .block_size = selection.block_size,
        .rows_per_wg = selection.rows_per_wg,
        .thread_execution_width = selection.pipe.thread_execution_width,
        .static_threadgroup_memory_length = selection.pipe.static_threadgroup_memory_length,
        .weight_bytes_per_iter = weight_bytes,
        .total_ms = elapsed_ms,
        .ms_per_iter = ms_per_iter,
        .gbps = (@as(f64, @floatFromInt(total_bytes)) / seconds) / 1_000_000_000.0,
        .checksum0 = checksumOutput(output0_copy),
        .checksum1 = checksumOutput(output1_copy),
        .output0 = output0_copy,
        .output1 = output1_copy,
    };
}

fn benchmarkDualVariant(
    allocator: std.mem.Allocator,
    device: *const metal_device.MetalDevice,
    model: *const metal_loader.Model,
    hot_case: HotCase,
    warmup_iterations: u32,
    iterations: u32,
    threadgroup_override: ?u32,
) !DualBenchResult {
    const tensor1 = hot_case.tensor1 orelse return error.ExpectedDualCase;
    var selection = try selectDualPipeline(device.ctx, hot_case.cols, threadgroup_override);
    defer metal_pipeline.freePipeline(&selection.pipe);

    var input_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.cols) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output0_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.rows0) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output0_buf);
    var output1_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.rows1) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output1_buf);

    fillInputBuffer(&input_buf, hot_case.cols);
    @memset(output0_buf.cpu_ptr.?[0..output0_buf.size], 0);
    @memset(output1_buf.cpu_ptr.?[0..output1_buf.size], 0);

    try runDualDispatchBatch(device.ctx, &selection, hot_case.tensor0, tensor1, model, &input_buf, &output0_buf, &output1_buf, hot_case.rows0, hot_case.rows1, hot_case.cols, warmup_iterations);
    @memset(output0_buf.cpu_ptr.?[0..output0_buf.size], 0);
    @memset(output1_buf.cpu_ptr.?[0..output1_buf.size], 0);

    const start_ns = std.time.nanoTimestamp();
    try runDualDispatchBatch(device.ctx, &selection, hot_case.tensor0, tensor1, model, &input_buf, &output0_buf, &output1_buf, hot_case.rows0, hot_case.rows1, hot_case.cols, iterations);
    const elapsed_ns = std.time.nanoTimestamp() - start_ns;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms;
    const ms_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;
    const weight_bytes = weightBytesPerIter(hot_case.tensor0.info.type_, hot_case.totalRows(), hot_case.cols);
    const total_bytes = weight_bytes * iterations;
    const output0_copy = try copyOutput(allocator, &output0_buf, hot_case.rows0);
    const output1_copy = try copyOutput(allocator, &output1_buf, hot_case.rows1);

    return .{
        .case_key = hot_case.key,
        .variant_label = selection.variant_label,
        .shader_name = selection.shader_name,
        .tensor0_name = hot_case.tensor0.info.name,
        .tensor1_name = tensor1.info.name,
        .rows0 = hot_case.rows0,
        .rows1 = hot_case.rows1,
        .cols = hot_case.cols,
        .iterations = iterations,
        .block_size = selection.block_size,
        .rows_per_wg = selection.rows_per_wg,
        .thread_execution_width = selection.pipe.thread_execution_width,
        .static_threadgroup_memory_length = selection.pipe.static_threadgroup_memory_length,
        .weight_bytes_per_iter = weight_bytes,
        .total_ms = elapsed_ms,
        .ms_per_iter = ms_per_iter,
        .gbps = (@as(f64, @floatFromInt(total_bytes)) / seconds) / 1_000_000_000.0,
        .checksum0 = checksumOutput(output0_copy),
        .checksum1 = checksumOutput(output1_copy),
        .output0 = output0_copy,
        .output1 = output1_copy,
    };
}

fn benchmarkRmsNormDualVariant(
    allocator: std.mem.Allocator,
    device: *const metal_device.MetalDevice,
    model: *const metal_loader.Model,
    hot_case: HotCase,
    warmup_iterations: u32,
    iterations: u32,
    threadgroup_override: ?u32,
) !DualBenchResult {
    const tensor1 = hot_case.tensor1 orelse return error.ExpectedDualCase;
    const norm_tensor = hot_case.norm_tensor orelse return error.MissingAttnNormTensor;
    var rms_norm_pipe = try loadShaderPipeline(device.ctx, "rms_norm_mul");
    defer metal_pipeline.freePipeline(&rms_norm_pipe);
    var dual_selection = try selectDualPipeline(device.ctx, hot_case.cols, threadgroup_override);
    defer metal_pipeline.freePipeline(&dual_selection.pipe);

    var hidden_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.cols) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&hidden_buf);
    var norm_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.cols) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&norm_buf);
    var norm_weight_buf = try loadNormWeightsBuffer(device.ctx, model, norm_tensor, hot_case.cols);
    defer metal_buffer.freeBuffer(&norm_weight_buf);
    var output0_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.rows0) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output0_buf);
    var output1_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.rows1) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output1_buf);

    fillInputBuffer(&hidden_buf, hot_case.cols);
    @memset(norm_buf.cpu_ptr.?[0..norm_buf.size], 0);
    @memset(output0_buf.cpu_ptr.?[0..output0_buf.size], 0);
    @memset(output1_buf.cpu_ptr.?[0..output1_buf.size], 0);

    try runRmsNormDualDispatchBatch(device.ctx, &rms_norm_pipe, &dual_selection, hot_case.tensor0, tensor1, model, &hidden_buf, &norm_buf, &norm_weight_buf, &output0_buf, &output1_buf, hot_case.rows0, hot_case.rows1, hot_case.cols, warmup_iterations);
    @memset(norm_buf.cpu_ptr.?[0..norm_buf.size], 0);
    @memset(output0_buf.cpu_ptr.?[0..output0_buf.size], 0);
    @memset(output1_buf.cpu_ptr.?[0..output1_buf.size], 0);

    const start_ns = std.time.nanoTimestamp();
    try runRmsNormDualDispatchBatch(device.ctx, &rms_norm_pipe, &dual_selection, hot_case.tensor0, tensor1, model, &hidden_buf, &norm_buf, &norm_weight_buf, &output0_buf, &output1_buf, hot_case.rows0, hot_case.rows1, hot_case.cols, iterations);
    const elapsed_ns = std.time.nanoTimestamp() - start_ns;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms;
    const ms_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;
    const weight_bytes = weightBytesPerIter(hot_case.tensor0.info.type_, hot_case.totalRows(), hot_case.cols);
    const total_bytes = weight_bytes * iterations;
    const output0_copy = try copyOutput(allocator, &output0_buf, hot_case.rows0);
    const output1_copy = try copyOutput(allocator, &output1_buf, hot_case.rows1);

    return .{
        .case_key = hot_case.key,
        .variant_label = "rms+dual",
        .shader_name = "rms_norm_mul+dmmv_q8_0_dual",
        .tensor0_name = hot_case.tensor0.info.name,
        .tensor1_name = tensor1.info.name,
        .rows0 = hot_case.rows0,
        .rows1 = hot_case.rows1,
        .cols = hot_case.cols,
        .iterations = iterations,
        .block_size = dual_selection.block_size,
        .rows_per_wg = dual_selection.rows_per_wg,
        .thread_execution_width = dual_selection.pipe.thread_execution_width,
        .static_threadgroup_memory_length = dual_selection.pipe.static_threadgroup_memory_length,
        .weight_bytes_per_iter = weight_bytes,
        .total_ms = elapsed_ms,
        .ms_per_iter = ms_per_iter,
        .gbps = (@as(f64, @floatFromInt(total_bytes)) / seconds) / 1_000_000_000.0,
        .checksum0 = checksumOutput(output0_copy),
        .checksum1 = checksumOutput(output1_copy),
        .output0 = output0_copy,
        .output1 = output1_copy,
    };
}

fn benchmarkFusedDualVariant(
    allocator: std.mem.Allocator,
    device: *const metal_device.MetalDevice,
    model: *const metal_loader.Model,
    hot_case: HotCase,
    warmup_iterations: u32,
    iterations: u32,
    threadgroup_override: ?u32,
) !DualBenchResult {
    const tensor1 = hot_case.tensor1 orelse return error.ExpectedDualCase;
    const norm_tensor = hot_case.norm_tensor orelse return error.MissingAttnNormTensor;
    var selection = try selectFusedDualPipeline(device.ctx, hot_case.cols, threadgroup_override);
    defer metal_pipeline.freePipeline(&selection.pipe);

    var hidden_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.cols) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&hidden_buf);
    var norm_weight_buf = try loadNormWeightsBuffer(device.ctx, model, norm_tensor, hot_case.cols);
    defer metal_buffer.freeBuffer(&norm_weight_buf);
    var output0_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.rows0) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output0_buf);
    var output1_buf = try metal_buffer.createBuffer(device.ctx, @as(usize, hot_case.rows1) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output1_buf);

    fillInputBuffer(&hidden_buf, hot_case.cols);
    @memset(output0_buf.cpu_ptr.?[0..output0_buf.size], 0);
    @memset(output1_buf.cpu_ptr.?[0..output1_buf.size], 0);

    try runFusedDualDispatchBatch(device.ctx, &selection, hot_case.tensor0, tensor1, model, &hidden_buf, &norm_weight_buf, &output0_buf, &output1_buf, hot_case.rows0, hot_case.rows1, hot_case.cols, warmup_iterations);
    @memset(output0_buf.cpu_ptr.?[0..output0_buf.size], 0);
    @memset(output1_buf.cpu_ptr.?[0..output1_buf.size], 0);

    const start_ns = std.time.nanoTimestamp();
    try runFusedDualDispatchBatch(device.ctx, &selection, hot_case.tensor0, tensor1, model, &hidden_buf, &norm_weight_buf, &output0_buf, &output1_buf, hot_case.rows0, hot_case.rows1, hot_case.cols, iterations);
    const elapsed_ns = std.time.nanoTimestamp() - start_ns;
    const elapsed_ms = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms;
    const ms_per_iter = elapsed_ms / @as(f64, @floatFromInt(iterations));
    const seconds = @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_s;
    const weight_bytes = weightBytesPerIter(hot_case.tensor0.info.type_, hot_case.totalRows(), hot_case.cols);
    const total_bytes = weight_bytes * iterations;
    const output0_copy = try copyOutput(allocator, &output0_buf, hot_case.rows0);
    const output1_copy = try copyOutput(allocator, &output1_buf, hot_case.rows1);

    return .{
        .case_key = hot_case.key,
        .variant_label = selection.variant_label,
        .shader_name = selection.shader_name,
        .tensor0_name = hot_case.tensor0.info.name,
        .tensor1_name = tensor1.info.name,
        .rows0 = hot_case.rows0,
        .rows1 = hot_case.rows1,
        .cols = hot_case.cols,
        .iterations = iterations,
        .block_size = selection.block_size,
        .rows_per_wg = selection.rows_per_wg,
        .thread_execution_width = selection.pipe.thread_execution_width,
        .static_threadgroup_memory_length = selection.pipe.static_threadgroup_memory_length,
        .weight_bytes_per_iter = weight_bytes,
        .total_ms = elapsed_ms,
        .ms_per_iter = ms_per_iter,
        .gbps = (@as(f64, @floatFromInt(total_bytes)) / seconds) / 1_000_000_000.0,
        .checksum0 = checksumOutput(output0_copy),
        .checksum1 = checksumOutput(output1_copy),
        .output0 = output0_copy,
        .output1 = output1_copy,
    };
}

fn printBenchResult(stdout: anytype, result: BenchResult) !void {
    try stdout.interface.print(
        "  {s}: shader={s} tg={d} rows/wg={d} tew={d} tgmem={d}B | {d:.3} ms/iter | {d:.2} GB/s | checksum {d:.6}\n",
        .{
            result.variant_label,
            result.shader_name,
            result.block_size,
            result.rows_per_wg,
            result.thread_execution_width,
            result.static_threadgroup_memory_length,
            result.ms_per_iter,
            result.gbps,
            result.checksum,
        },
    );
}

fn printDualBenchResult(stdout: anytype, result: DualBenchResult) !void {
    try stdout.interface.print(
        "  {s}: shader={s} tg={d} rows/wg={d} tew={d} tgmem={d}B | {d:.3} ms/iter | {d:.2} GB/s | checksum0 {d:.6} | checksum1 {d:.6}\n",
        .{
            result.variant_label,
            result.shader_name,
            result.block_size,
            result.rows_per_wg,
            result.thread_execution_width,
            result.static_threadgroup_memory_length,
            result.ms_per_iter,
            result.gbps,
            result.checksum0,
            result.checksum1,
        },
    );
}

fn compareDualOutputs(lhs: DualBenchResult, rhs: DualBenchResult) CompareSummary {
    const first = compareOutputs(lhs.output0, rhs.output0);
    const second = compareOutputs(lhs.output1, rhs.output1);
    const total_len = lhs.output0.len + lhs.output1.len;
    const weighted_sum = first.mean_abs * @as(f64, @floatFromInt(lhs.output0.len)) +
        second.mean_abs * @as(f64, @floatFromInt(lhs.output1.len));
    return .{
        .max_abs = @max(first.max_abs, second.max_abs),
        .mean_abs = if (total_len == 0) 0.0 else weighted_sum / @as(f64, @floatFromInt(total_len)),
    };
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const config = parseArgs(args) catch |err| {
        std.fs.File.stderr().writeAll(helpText()) catch {};
        return err;
    };
    if (config.show_help) {
        try std.fs.File.stdout().writeAll(helpText());
        return;
    }

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);

    var device = try metal_device.MetalDevice.init(allocator, config.device_index);
    defer device.deinit();

    var gpu_process_lock = process_lock.acquire(.metal, device.selected_device_index) catch |err| {
        support.reportGpuProcessLockError(err, .metal, device.selected_device_index);
    };
    defer gpu_process_lock.deinit();

    var model = try metal_loader.load(config.model_path.?, device.ctx, allocator);
    defer model.deinit();

    const hot_case_ids = [_]CaseId{ .lm_head, .attn_q, .attn_k, .attn_v, .attn_out, .ssm_qkv, .ssm_gate, .ssm_dual, .ssm_out, .router, .shared_gate, .shared_up, .shared_down, .shared_dual, .moe_gate, .moe_up, .moe_down };

    try stdout.interface.print("Metal q8 exact-shape benchmark\n", .{});
    try stdout.interface.print("Model: {s}\n", .{config.model_path.?});
    try stdout.interface.print(
        "GPU: {s} | unified={} | tgmem={d} KiB | working-set={d} GiB\n",
        .{
            @tagName(device.chip),
            device.hasUnifiedMemory(),
            device.maxThreadgroupMemoryLength() / 1024,
            device.recommendedMaxWorkingSetSize() / (1024 * 1024 * 1024),
        },
    );
    try stdout.interface.print(
        "Warmup={d} | iterations={d} | pipeline={s} | tg_override={s}\n\n",
        .{
            config.warmup_iterations,
            config.iterations,
            @tagName(config.pipeline_mode),
            if (config.threadgroup_size) |_| "set" else "auto",
        },
    );

    for (hot_case_ids) |case_id| {
        if (config.case_id != .all and case_id != config.case_id) continue;

        const hot_case = try resolveHotCase(&model, case_id);
        if (hot_case.isDual()) {
            const tensor1 = hot_case.tensor1.?;
            try stdout.interface.print(
                "Case {s}: {s} | tensors={s} + {s} | quant={s} + {s} | M0={d} M1={d} K={d} | weight {d:.2} MiB/iter\n",
                .{
                    hot_case.key,
                    hot_case.label,
                    hot_case.tensor0.info.name,
                    tensor1.info.name,
                    @tagName(hot_case.tensor0.info.type_),
                    @tagName(tensor1.info.type_),
                    hot_case.rows0,
                    hot_case.rows1,
                    hot_case.cols,
                    @as(f64, @floatFromInt(weightBytesPerIter(hot_case.tensor0.info.type_, hot_case.totalRows(), hot_case.cols))) / (1024.0 * 1024.0),
                },
            );

            var separate_result: ?DualBenchResult = null;
            defer if (separate_result) |*result| {
                allocator.free(result.output0);
                allocator.free(result.output1);
            };
            var rms_dual_result: ?DualBenchResult = null;
            defer if (rms_dual_result) |*result| {
                allocator.free(result.output0);
                allocator.free(result.output1);
            };
            var dual_result: ?DualBenchResult = null;
            defer if (dual_result) |*result| {
                allocator.free(result.output0);
                allocator.free(result.output1);
            };
            var fused_result: ?DualBenchResult = null;
            defer if (fused_result) |*result| {
                allocator.free(result.output0);
                allocator.free(result.output1);
            };

            if (hot_case.norm_tensor != null) {
                fused_result = try benchmarkFusedDualVariant(
                    allocator,
                    &device,
                    &model,
                    hot_case,
                    config.warmup_iterations,
                    config.iterations,
                    config.threadgroup_size,
                );
                try printDualBenchResult(&stdout, fused_result.?);

                rms_dual_result = try benchmarkRmsNormDualVariant(
                    allocator,
                    &device,
                    &model,
                    hot_case,
                    config.warmup_iterations,
                    config.iterations,
                    config.threadgroup_size,
                );
                try printDualBenchResult(&stdout, rms_dual_result.?);
            }

            if (config.pipeline_mode == .runtime or config.pipeline_mode == .both) {
                dual_result = try benchmarkDualVariant(
                    allocator,
                    &device,
                    &model,
                    hot_case,
                    config.warmup_iterations,
                    config.iterations,
                    config.threadgroup_size,
                );
                try printDualBenchResult(&stdout, dual_result.?);
            }

            if (config.pipeline_mode == .both) {
                separate_result = try benchmarkSeparateDualVariant(
                    allocator,
                    &device,
                    &model,
                    hot_case,
                    config.warmup_iterations,
                    config.iterations,
                    config.threadgroup_size,
                );
                try printDualBenchResult(&stdout, separate_result.?);
            } else if (config.pipeline_mode == .k2048) {
                return error.K2048UnsupportedForDualCase;
            }

            if (fused_result != null and rms_dual_result != null) {
                const diff = compareDualOutputs(rms_dual_result.?, fused_result.?);
                const delta_pct = if (rms_dual_result.?.ms_per_iter > 0.0)
                    ((rms_dual_result.?.ms_per_iter - fused_result.?.ms_per_iter) / rms_dual_result.?.ms_per_iter) * 100.0
                else
                    0.0;
                try stdout.interface.print(
                    "  fused vs rms+dual: {d:.2}% ms/iter | max_abs {d:.6} | mean_abs {d:.6}\n",
                    .{ delta_pct, diff.max_abs, diff.mean_abs },
                );
            }

            if (separate_result != null and dual_result != null) {
                const diff = compareDualOutputs(separate_result.?, dual_result.?);
                const delta_pct = if (separate_result.?.ms_per_iter > 0.0)
                    ((separate_result.?.ms_per_iter - dual_result.?.ms_per_iter) / separate_result.?.ms_per_iter) * 100.0
                else
                    0.0;
                try stdout.interface.print(
                    "  dual vs separate: {d:.2}% ms/iter | max_abs {d:.6} | mean_abs {d:.6}\n",
                    .{ delta_pct, diff.max_abs, diff.mean_abs },
                );
            }
        } else if (hot_case.isMoe()) {
            try stdout.interface.print(
                "Case {s}: {s} | tensor={s} | quant={s} | M={d} K={d} | experts={d} | x_stride={d} | weight {d:.2} MiB/iter\n",
                .{
                    hot_case.key,
                    hot_case.label,
                    hot_case.tensor0.info.name,
                    @tagName(hot_case.tensor0.info.type_),
                    hot_case.rows0,
                    hot_case.cols,
                    hot_case.expert_slots,
                    hot_case.x_expert_stride,
                    @as(f64, @floatFromInt(weightBytesPerIter(hot_case.tensor0.info.type_, hot_case.rows0, hot_case.cols) * hot_case.expert_slots)) / (1024.0 * 1024.0),
                },
            );

            var runtime_result: ?BenchResult = null;
            defer if (runtime_result) |*result| allocator.free(result.output);
            var alt_result: ?BenchResult = null;
            defer if (alt_result) |*result| allocator.free(result.output);

            runtime_result = try benchmarkMoeVariant(
                allocator,
                &device,
                &model,
                hot_case,
                config.warmup_iterations,
                config.iterations,
                .runtime,
            );
            try printBenchResult(&stdout, runtime_result.?);

            if (config.pipeline_mode == .both) {
                alt_result = switch (hot_case.tensor0.info.type_) {
                    .q4_k => try benchmarkMoeVariant(
                        allocator,
                        &device,
                        &model,
                        hot_case,
                        config.warmup_iterations,
                        config.iterations,
                        .both,
                    ),
                    .q5_k => try benchmarkMoeVariant(
                        allocator,
                        &device,
                        &model,
                        hot_case,
                        config.warmup_iterations,
                        config.iterations,
                        .both,
                    ),
                    else => null,
                };
                if (alt_result) |result| {
                    try printBenchResult(&stdout, result);
                }
            } else if (config.pipeline_mode == .k2048 and hot_case.tensor0.info.type_ != .q6_k) {
                alt_result = try benchmarkMoeVariant(
                    allocator,
                    &device,
                    &model,
                    hot_case,
                    config.warmup_iterations,
                    config.iterations,
                    .k2048,
                );
                try printBenchResult(&stdout, alt_result.?);
            }

            if (runtime_result != null and alt_result != null) {
                const diff = compareOutputs(runtime_result.?.output, alt_result.?.output);
                const delta_pct = if (runtime_result.?.ms_per_iter > 0.0)
                    ((runtime_result.?.ms_per_iter - alt_result.?.ms_per_iter) / runtime_result.?.ms_per_iter) * 100.0
                else
                    0.0;
                try stdout.interface.print(
                    "  delta: {d:.2}% ms/iter | max_abs {d:.6} | mean_abs {d:.6}\n",
                    .{ delta_pct, diff.max_abs, diff.mean_abs },
                );
            }
        } else {
            try stdout.interface.print(
                "Case {s}: {s} | tensor={s} | quant={s} | M={d} K={d} | weight {d:.2} MiB/iter\n",
                .{
                    hot_case.key,
                    hot_case.label,
                    hot_case.tensor0.info.name,
                    @tagName(hot_case.tensor0.info.type_),
                    hot_case.rows0,
                    hot_case.cols,
                    @as(f64, @floatFromInt(weightBytesPerIter(hot_case.tensor0.info.type_, hot_case.rows0, hot_case.cols))) / (1024.0 * 1024.0),
                },
            );

            var runtime_result: ?BenchResult = null;
            defer if (runtime_result) |*result| allocator.free(result.output);
            var k2048_result: ?BenchResult = null;
            defer if (k2048_result) |*result| allocator.free(result.output);

            if (config.pipeline_mode == .runtime or config.pipeline_mode == .both) {
                runtime_result = try benchmarkVariant(
                    allocator,
                    &device,
                    &model,
                    hot_case,
                    config.warmup_iterations,
                    config.iterations,
                    .runtime,
                    config.threadgroup_size,
                );
                try printBenchResult(&stdout, runtime_result.?);
            }

            if (config.pipeline_mode == .k2048 or config.pipeline_mode == .both) {
                if (hot_case.cols <= 2048) {
                    k2048_result = try benchmarkVariant(
                        allocator,
                        &device,
                        &model,
                        hot_case,
                        config.warmup_iterations,
                        config.iterations,
                        .k2048,
                        config.threadgroup_size,
                    );
                    try printBenchResult(&stdout, k2048_result.?);
                } else if (config.pipeline_mode == .both) {
                    try stdout.interface.print("  k2048: skipped (K={d} > 2048)\n", .{hot_case.cols});
                } else {
                    return error.K2048OnlySupportsK2048OrLess;
                }
            }

            if (runtime_result != null and k2048_result != null) {
                const diff = compareOutputs(runtime_result.?.output, k2048_result.?.output);
                const delta_pct = if (runtime_result.?.ms_per_iter > 0.0)
                    ((runtime_result.?.ms_per_iter - k2048_result.?.ms_per_iter) / runtime_result.?.ms_per_iter) * 100.0
                else
                    0.0;
                try stdout.interface.print(
                    "  delta: {d:.2}% ms/iter | max_abs {d:.6} | mean_abs {d:.6}\n",
                    .{ delta_pct, diff.max_abs, diff.mean_abs },
                );
            }
        }

        try stdout.interface.print("\n", .{});
    }
    try stdout.interface.flush();
}
