//! Vulkan text inference for the fixed-loop Inkling 2 Mini decoder.
//! @section Inference Runtime
//!
//! Each chronological layer visit has private convolution state and a packed FP16 attention cache. Single-token command buffers replay with an updated token and position.
const std = @import("std");
const inkling = @import("../model/inkling.zig");
const loader = @import("../model/loader.zig");
const gguf = @import("../model/gguf.zig");
const vk = @import("../vulkan/vk.zig").c;
const Instance = @import("../vulkan/instance.zig").Instance;
const Buffer = @import("../vulkan/buffer.zig").Buffer;
const command = @import("../vulkan/command.zig");
const pipeline = @import("../vulkan/pipeline.zig");

const log = std.log.scoped(.inkling_vulkan);
const tensors = @import("../model/inkling_weights.zig").Tensors(loader.Model, loader.LoadedTensor);
const Layout = inkling.Layout;
const Weight = tensors.Weight;
const Binding = vk.VkDescriptorBufferInfo;
const batch = 64;
const threads = 64;
const attention_tile = 128;
const logit_partials = 128;
const decode_buckets = 1 + std.math.log2_int_ceil(u32, groups(inkling.max_context, attention_tile));
const max_bindings = 10;
const empty = Buffer{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = null };

const Embed = extern struct { width: u32, kind: u32 };
const Quantize = extern struct { width: u32 };
const Matrix = extern struct { rows: u32, cols: u32, routes: u32, per_route: u32, blocked: u32 };
const Repack = extern struct { blocks: u32, groups: u32 };
const Output = extern struct { rows: u32, cols: u32 };
const Norm = extern struct { width: u32, out_stride: u32, out_offset: u32, eps: f32, weighted: u32 };
const Mix = extern struct { hidden: u32, capacity: u32, first_token: u32, blocks: u32, block_ids: [inkling.max_depth_blocks]u32 };
const Count = extern struct { n: u32 };
const Conv = extern struct {
    channels: u32,
    taps: u32,
    count: u32,
    history_offset: u32,
    dynamic_head_dim: u32,
    residual: u32,
    scale: f32,
};
const Clip = extern struct { dim: u32, heads: u32, scale_floor: u32, limit: f32, alpha: f32 };
const KvWrite = extern struct { dim: u32, heads: u32, capacity: u32, clip: f32, eps: f32 };
const Attention = extern struct {
    dim: u32,
    heads: u32,
    kv_heads: u32,
    capacity: u32,
    window: u32,
    relative_dim: u32,
    relative_extent: u32,
    tiles: u32,
};
const Route = extern struct { experts: u32, active: u32, biased: u32, scale: f32 };
const Situ = extern struct { width: u32, beta: f32, linear_beta: f32 };
const Combine = extern struct { width: u32, active: u32 };
const Argmax = extern struct { n: u32, phase: u32 };

const Kernel = enum {
    embed,
    mat_f32,
    mat_f16,
    mat_bf16,
    mat_q4_0,
    mat_q4_0_q8,
    mat_q8_0,
    mat_q4_k,
    mat_q5_k,
    mat_q6_k,
    output_q6_k,
    quantize,
    repack_q4_0,
    norm,
    mix,
    copy,
    control,
    conv,
    clip,
    kv,
    attention,
    attention_merge,
    route,
    situ,
    combine,
    argmax,

    fn program(self: Kernel) Shader {
        return switch (self) {
            .embed => Shader.of("inkling_embed", 3, Embed),
            .mat_f32 => Shader.matrix(.f32),
            .mat_f16 => Shader.matrix(.f16),
            .mat_bf16 => Shader.matrix(.bf16),
            .mat_q4_0 => Shader.matrix(.q4_0),
            .mat_q4_0_q8 => Shader.of("inkling_q4_0_q8", 4, Matrix),
            .mat_q8_0 => Shader.matrix(.q8_0),
            .mat_q4_k => Shader.matrix(.q4_k),
            .mat_q5_k => Shader.matrix(.q5_k),
            .mat_q6_k => Shader.matrix(.q6_k),
            .output_q6_k => Shader.of("inkling_q6_k_head", 3, Output),
            .quantize => Shader.of("inkling_quantize", 2, Quantize),
            .repack_q4_0 => Shader.of("inkling_repack_q4_0", 1, Repack),
            .norm => Shader.of("inkling_norm", 3, Norm),
            .mix => Shader.of("inkling_mix", 4, Mix),
            .copy => Shader.of("inkling_copy", 2, Count),
            .control => Shader.of("inkling_control", 4, Count),
            .conv => Shader.of("inkling_conv", 7, Conv),
            .clip => Shader.of("inkling_clip", 3, Clip),
            .kv => Shader.of("inkling_kv", 5, KvWrite),
            .attention => Shader.of("inkling_attention", 10, Attention),
            .attention_merge => Shader.of("inkling_attention_merge", 4, Attention),
            .route => Shader.of("inkling_route", 4, Route),
            .situ => Shader.of("inkling_situ", 2, Situ),
            .combine => Shader.of("inkling_combine", 3, Combine),
            .argmax => Shader.of("argmax", 3, Argmax),
        };
    }

    fn forMatrix(kind: gguf.GGMLType) Kernel {
        return switch (kind) {
            .f32 => .mat_f32,
            .f16 => .mat_f16,
            .bf16 => .mat_bf16,
            .q4_0 => .mat_q4_0,
            .q8_0 => .mat_q8_0,
            .q4_k => .mat_q4_k,
            .q5_k => .mat_q5_k,
            .q6_k => .mat_q6_k,
            else => unreachable, // Checked by the shared Inkling tensor reader.
        };
    }
};
const Pipelines = std.EnumArray(Kernel, ?pipeline.Pipeline);

const Shader = struct {
    name: []const u8,
    bindings: u32,
    push_size: u32,
    quant: ?pipeline.SpecConst = null,

    fn of(name: []const u8, bindings: u32, comptime Params: type) Shader {
        return .{ .name = name, .bindings = bindings, .push_size = @sizeOf(Params) };
    }

    fn matrix(kind: gguf.GGMLType) Shader {
        var shader = of("inkling_matvec", 4, Matrix);
        shader.quant = .{ .id = 0, .value = @intFromEnum(kind) };
        return shader;
    }
};

const Cache = struct {
    capacity: u32 = 0,
    k: Buffer = empty,
    v: Buffer = empty,
    convolution: Buffer = empty,

    fn capacityFor(d: Layout, layer: u32, ctx: u32) u32 {
        return if (d.sliding[layer]) @min(ctx, d.window + batch - 1) else ctx;
    }

    fn kvBytes(d: Layout, capacity: u32) u64 {
        return @as(u64, capacity) * d.kv_heads * groups(d.head_dim, 2) * @sizeOf(u32);
    }

    fn convBytes(d: Layout) u64 {
        return @as(u64, 2 * (d.kv_heads * d.head_dim + d.hidden)) * (d.taps - 1) * @sizeOf(f32);
    }

    fn deinit(self: *Cache) void {
        if (self.k.handle != null) self.k.deinit();
        if (self.v.handle != null) self.v.deinit();
        if (self.convolution.handle != null) self.convolution.deinit();
    }
};

const Scratch = struct {
    tokens: Buffer = empty,
    cur: Buffer = empty,
    prelude: Buffer = empty,
    previous_pass: Buffer = empty,
    saved: Buffer = empty,
    messages: Buffer = empty,
    inputs: Buffer = empty,
    paired: Buffer = empty,
    rank: Buffer = empty,
    gain: Buffer = empty,
    forget: Buffer = empty,
    q: Buffer = empty,
    k_raw: Buffer = empty,
    v_raw: Buffer = empty,
    k: Buffer = empty,
    v: Buffer = empty,
    relative: Buffer = empty,
    sink: Buffer = empty,
    gate: Buffer = empty,
    k_taps: Buffer = empty,
    v_taps: Buffer = empty,
    attn_partial: Buffer = empty,
    attention: Buffer = empty,
    attn_raw: Buffer = empty,
    attn_hidden: Buffer = empty,
    route_logits: Buffer = empty,
    route_ids: Buffer = empty,
    route_weights: Buffer = empty,
    gate_up: Buffer = empty,
    expert_hidden: Buffer = empty,
    expert_result: Buffer = empty,
    ffn_raw: Buffer = empty,
    logits: Buffer = empty,
    candidates: Buffer = empty,
    next: Buffer = empty,
    quantized_input: Buffer = empty,

    fn sizes(d: Layout, ctx: u32) [std.meta.fields(Scratch).len]u64 {
        const h = @as(u64, batch) * d.hidden * @sizeOf(f32);
        const q = @as(u64, batch) * d.heads * d.head_dim * @sizeOf(f32);
        const kv = @as(u64, batch) * d.kv_heads * d.head_dim * @sizeOf(f32);
        const route = @as(u64, batch) * d.active * @sizeOf(f32);
        const taps = @as(u64, batch) * d.kv_heads * d.taps * @sizeOf(f32);
        const bytes = .{
            .tokens = (batch + 1) * @sizeOf(u32),
            .cur = h,
            .prelude = h,
            .previous_pass = h,
            .saved = h * groups(d.visitCount(), d.block_size),
            .messages = 4 * h,
            .inputs = 3 * h,
            .paired = 2 * h,
            .rank = @as(u64, batch) * d.gate_rank * @sizeOf(f32),
            .gain = h,
            .forget = h,
            .q = q,
            .k_raw = kv,
            .v_raw = kv,
            .k = kv,
            .v = kv,
            .relative = @as(u64, batch) * d.heads * d.relative_dim * @sizeOf(f32),
            .sink = @as(u64, batch) * d.heads * @sizeOf(f32),
            .gate = q,
            .k_taps = taps,
            .v_taps = taps,
            .attn_partial = @as(u64, batch) * d.heads * groups(ctx, attention_tile) * (d.head_dim + 2) * @sizeOf(f32),
            .attention = q,
            .attn_raw = h,
            .attn_hidden = h,
            .route_logits = @as(u64, batch) * d.experts * @sizeOf(f32),
            .route_ids = route,
            .route_weights = route,
            .gate_up = route * 2 * d.expert_width,
            .expert_hidden = route * d.expert_width,
            .expert_result = route * d.hidden,
            .ffn_raw = h,
            .logits = @as(u64, d.vocab) * @sizeOf(f32),
            .candidates = logit_partials * 2 * @sizeOf(u32),
            .next = @sizeOf(u32),
            .quantized_input = @as(u64, batch) * groups(@max(2 * d.hidden, d.heads * d.head_dim, d.active * d.expert_width, d.gate_rank), 32) * 36,
        };
        var result: [std.meta.fields(Scratch).len]u64 = undefined;
        inline for (std.meta.fields(Scratch), 0..) |field, i| result[i] = @field(bytes, field.name);
        return result;
    }

    fn init(instance: *const Instance, d: Layout, ctx: u32) !Scratch {
        var scratch: Scratch = .{};
        errdefer scratch.deinit();
        const bytes = sizes(d, ctx);
        inline for (std.meta.fields(Scratch), bytes) |field, size| {
            @field(scratch, field.name) = try deviceBuffer(instance, size);
        }
        return scratch;
    }

    fn deinit(self: *Scratch) void {
        inline for (std.meta.fields(Scratch)) |field| {
            const buf = &@field(self, field.name);
            if (buf.handle != null) buf.deinit();
        }
    }
};

pub const RuntimeBytes = struct {
    context: u64 = 0,
    device: u64 = 0,
};

pub fn runtimeBytes(d: Layout, ctx: u32, storage_limit: u64) ?RuntimeBytes {
    var usage: RuntimeBytes = .{};
    for (Scratch.sizes(d, ctx)) |size| {
        if (size > storage_limit) return null;
        usage.device += size;
    }
    const conv_bytes = Cache.convBytes(d);
    if (conv_bytes > storage_limit) return null;
    for (0..d.visitCount()) |visit| {
        const layer = d.physicalLayer(@intCast(visit));
        const kv_bytes = Cache.kvBytes(d, Cache.capacityFor(d, layer, ctx));
        if (kv_bytes > storage_limit) return null;
        usage.context += 2 * kv_bytes;
        usage.device += 2 * kv_bytes + conv_bytes;
    }
    return usage;
}

pub fn contextCapacity(d: Layout, ceiling: u32, budget: u64, storage_limit: u64) u32 {
    var low: u32 = 0;
    var high = ceiling;
    while (low < high) {
        const mid = low + (high - low + 1) / 2;
        const usage = runtimeBytes(d, mid, storage_limit);
        if (usage != null and usage.?.device <= budget) low = mid else high = mid - 1;
    }
    return low;
}

pub const Forward = struct {
    allocator: std.mem.Allocator,
    instance: *const Instance,
    d: Layout,
    max_ctx: u32,
    usage: RuntimeBytes,
    layers: []tensors.Layer,
    visits: []Cache,
    embed: Weight,
    embed_norm: Weight,
    output: Weight,
    output_norm: Weight,
    output_query: Weight,
    scratch: Scratch,
    input: Buffer,
    result: Buffer,
    readback: Buffer,
    cmd_pool: command.CommandPool,
    work_cmd: command.CommandBuffer,
    decode_cmds: [decode_buckets]?command.CommandBuffer = @splat(null),
    descriptor_pool: vk.VkDescriptorPool = null,
    pipes: Pipelines = Pipelines.initFill(null),
    next_position: u32 = 0,

    pub fn init(model: *loader.Model, instance: *const Instance, shader_dir: []const u8, allocator: std.mem.Allocator) !Forward {
        const d = try Layout.read(&model.gguf_file);
        const limits = instance.device_props.limits;
        if (limits.maxPerStageDescriptorStorageBuffers < max_bindings or limits.maxDescriptorSetStorageBuffers < max_bindings)
            return error.InklingStorageDescriptorLimit;
        const ceiling = @min(model.config.context_length, limits.maxComputeWorkGroupCount[2] *| attention_tile);
        const budget = instance.vramBytes() -| model.deviceTensorBytes() -| inkling.vulkan_memory_reserve;
        const max_ctx = contextCapacity(d, ceiling, budget, limits.maxStorageBufferRange);
        if (max_ctx == 0) return error.ContextLengthDoesNotFit;
        const usage = runtimeBytes(d, max_ctx, limits.maxStorageBufferRange).?;

        const embed = try tensors.modelWeight(model, "token_embd.weight", &.{ d.hidden, d.padded_vocab }, .matrix);
        const embed_norm = try tensors.modelWeight(model, "token_embd_norm.weight", &.{d.hidden}, .float);
        const output = try tensors.modelWeight(model, "output.weight", &.{ d.hidden, d.padded_vocab }, .matrix);
        const output_norm = try tensors.modelWeight(model, "output_norm.weight", &.{d.hidden}, .float);
        const output_query = try tensors.modelWeight(model, "output_res_query.weight", &.{d.hidden}, .float);
        const layers = try allocator.alloc(tensors.Layer, d.layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| layer.* = try tensors.Layer.read(model, d, @intCast(i));

        const visits = try allocator.alloc(Cache, d.visitCount());
        @memset(visits, .{});
        errdefer {
            for (visits) |*visit| visit.deinit();
            allocator.free(visits);
        }
        for (visits, 0..) |*visit, i| {
            visit.capacity = Cache.capacityFor(d, d.physicalLayer(@intCast(i)), max_ctx);
            const kv_bytes = Cache.kvBytes(d, visit.capacity);
            visit.k = try deviceBuffer(instance, kv_bytes);
            visit.v = try deviceBuffer(instance, kv_bytes);
            visit.convolution = try deviceBuffer(instance, Cache.convBytes(d));
        }

        var scratch = try Scratch.init(instance, d, max_ctx);
        errdefer scratch.deinit();
        var input = try Buffer.initStaging(instance, scratch.tokens.size);
        errdefer input.deinit();
        var result = try Buffer.initReadback(instance, scratch.next.size);
        errdefer result.deinit();
        var readback = try Buffer.initReadback(instance, scratch.logits.size);
        errdefer readback.deinit();
        var cmd_pool = try command.CommandPool.init(instance);
        errdefer cmd_pool.deinit();
        var work_cmd = try command.CommandBuffer.init(instance, &cmd_pool);
        errdefer work_cmd.deinit(&cmd_pool);

        var self: Forward = .{
            .allocator = allocator,
            .instance = instance,
            .d = d,
            .max_ctx = max_ctx,
            .usage = usage,
            .layers = layers,
            .visits = visits,
            .embed = embed,
            .embed_norm = embed_norm,
            .output = output,
            .output_norm = output_norm,
            .output_query = output_query,
            .scratch = scratch,
            .input = input,
            .result = result,
            .readback = readback,
            .cmd_pool = cmd_pool,
            .work_cmd = work_cmd,
        };
        errdefer self.freePipelines();
        const push_desc = instance.push_descriptor_fn != null;
        if (!push_desc) self.descriptor_pool = try createDescriptorPool(instance, d.visitCount());
        const use_q8 = instance.caps.integer_dot_product and
            std.mem.eql(u8, std.posix.getenv("ZINC_INKLING_Q8") orelse "1", "1");
        const repack = use_q8 and limits.maxComputeSharedMemorySize >= 4608 * @sizeOf(u32) and
            std.mem.eql(u8, std.posix.getenv("ZINC_INKLING_BLOCK_Q4") orelse "1", "1");
        for (std.enums.values(Kernel)) |kernel| {
            if (!use_q8 and (kernel == .quantize or kernel == .mat_q4_0_q8)) continue;
            if (!repack and kernel == .repack_q4_0) continue;
            const program = kernel.program();
            const spv = try std.fmt.allocPrint(allocator, "{s}/{s}.spv", .{ shader_dir, program.name });
            defer allocator.free(spv);
            self.pipes.set(kernel, try pipeline.createFromSpirvWithOptions(
                instance,
                spv,
                program.bindings,
                program.push_size,
                if (program.quant) |spec| &.{spec} else &.{},
                .{ .required_subgroup_size = 64, .require_full_subgroups = true, .push_descriptors = push_desc },
                allocator,
            ));
        }
        if (repack) try self.repackExperts();
        if (max_ctx < model.config.context_length) {
            log.info("Context trimmed from {d} to {d} tokens to fit the Vulkan memory budget", .{ model.config.context_length, max_ctx });
        }
        log.info("Inkling 2 Mini: {d} stored layers, {d} layer visits, {d} context", .{ d.layers, d.visitCount(), max_ctx });
        return self;
    }

    pub fn deinit(self: *Forward) void {
        for (&self.decode_cmds) |*entry| {
            if (entry.*) |*cmd| cmd.deinit(&self.cmd_pool);
        }
        self.work_cmd.deinit(&self.cmd_pool);
        self.cmd_pool.deinit();
        self.freePipelines();
        self.readback.deinit();
        self.result.deinit();
        self.input.deinit();
        self.scratch.deinit();
        for (self.visits) |*visit| visit.deinit();
        self.allocator.free(self.visits);
        self.allocator.free(self.layers);
        self.* = undefined;
    }

    fn freePipelines(self: *Forward) void {
        if (self.descriptor_pool != null) vk.vkDestroyDescriptorPool(self.instance.device, self.descriptor_pool, null);
        for (&self.pipes.values) |*entry| {
            if (entry.*) |*pipe| pipe.deinit();
        }
    }

    fn repacked(self: *const Forward, w: Weight) bool {
        return self.pipes.get(.repack_q4_0) != null and w.gpu_buffer.mapped == null and
            w.info.type_ == .q4_0 and w.info.dims[0] % 128 == 0 and w.info.dims[1] % 4 == 0;
    }

    fn repackExperts(self: *Forward) !void {
        const cmd = &self.work_cmd;
        try cmd.beginOneTime();
        cmd.transferToComputeBarrier();
        for (self.layers) |layer| {
            for ([_]Weight{ layer.gate_up, layer.down }) |w| {
                if (!self.repacked(w)) continue;
                const n: u32 = @intCast(w.info.dims[1] * w.info.dims[2] / 4);
                const x = @min(n, self.instance.device_props.limits.maxComputeWorkGroupCount[0]);
                try self.dispatch(cmd, .repack_q4_0, .{ x, 1, groups(n, x) }, &.{weight(w)}, Repack{
                    .blocks = @intCast(w.info.dims[0] / 32),
                    .groups = n,
                });
            }
        }
        try cmd.end();
        try cmd.submitAndWait(self.instance.compute_queue);
    }

    pub fn prefill(self: *Forward, ids: []const u32, position: u32) !void {
        if (position > self.max_ctx or ids.len > self.max_ctx - position) return error.ContextLengthExceeded;
        var offset: u32 = 0;
        while (offset < ids.len) {
            const end: u32 = @intCast(@min(ids.len, offset + batch));
            try self.runChunk(ids[offset..end], position + offset, end == ids.len);
            offset = end;
        }
    }

    pub fn step(self: *Forward, token: u32, position: u32, output: bool) !void {
        try self.runChunk(&.{token}, position, output);
    }

    pub fn greedy(self: *const Forward) u32 {
        return std.mem.bytesAsValue(u32, self.result.mapped.?[0..@sizeOf(u32)]).*;
    }

    pub fn logits(self: *const Forward) []const f32 {
        const ptr: [*]const f32 = @ptrCast(@alignCast(self.readback.mapped.?));
        return ptr[0..self.d.vocab];
    }

    fn runChunk(self: *Forward, ids: []const u32, position: u32, output: bool) !void {
        if (position >= self.max_ctx or ids.len > self.max_ctx - position) return error.ContextLengthExceeded;
        if (position != 0 and self.next_position != position) return error.NonContiguousInklingPosition;
        for (ids) |id| if (id >= self.d.vocab) return error.InvalidInklingToken;
        const count: u32 = @intCast(ids.len);
        var input: [batch + 1]u32 = undefined;
        input[0] = position;
        @memcpy(input[1..][0..count], ids);
        self.input.upload(std.mem.sliceAsBytes(input[0 .. count + 1]));

        errdefer self.next_position = 0;
        const tiles = groups(position + count, attention_tile);
        if (count == 1 and output and self.descriptor_pool == null) {
            const bucket = std.math.log2_int_ceil(u32, tiles);
            const cached = &self.decode_cmds[bucket];
            if (cached.* == null) {
                var cmd = try command.CommandBuffer.init(self.instance, &self.cmd_pool);
                errdefer cmd.deinit(&self.cmd_pool);
                const budget = @min(@as(u32, 1) << @intCast(bucket), groups(self.max_ctx, attention_tile));
                try self.record(&cmd, count, budget, output);
                cached.* = cmd;
            }
            try cached.*.?.submitAndWait(self.instance.compute_queue);
        } else {
            try self.work_cmd.reset();
            if (self.descriptor_pool != null and vk.vkResetDescriptorPool(self.instance.device, self.descriptor_pool, 0) != vk.VK_SUCCESS)
                return error.DescriptorPoolResetFailed;
            try self.record(&self.work_cmd, count, tiles, output);
            try self.work_cmd.submitAndWait(self.instance.compute_queue);
        }
        self.next_position = position + count;
        if (output and self.greedy() >= self.d.vocab) return error.InvalidInklingOutput;
    }

    fn record(self: *const Forward, cmd: *command.CommandBuffer, count: u32, tiles: u32, output: bool) !void {
        const d = self.d;
        const s = &self.scratch;
        try cmd.begin();
        cmd.computeReadWriteBarrier();
        copyBuffer(cmd, &self.input, &s.tokens, (count + 1) * @sizeOf(u32));
        cmd.transferToComputeBarrier();

        try self.dispatch(cmd, .embed, .{ groups(d.hidden, threads), count, 1 }, &.{ weight(self.embed), bind(&s.tokens), bind(&s.cur) }, Embed{
            .width = d.hidden,
            .kind = @intFromEnum(self.embed.info.type_),
        });
        try self.norm(cmd, bind(&s.cur), self.embed_norm, bind(&s.cur), count, d.hidden, 0);

        var traversal: inkling.Traversal = .{};
        for (self.visits, 0..) |*cache, index| {
            const visit = traversal.visit(d, @intCast(index));
            if (visit.save_prelude) {
                try self.copy(cmd, bind(&s.cur), bind(&s.prelude), count * d.hidden);
            } else if (visit.restart_pass) {
                try self.copy(cmd, bind(&s.cur), bind(&s.previous_pass), count * d.hidden);
                try self.copy(cmd, bind(&s.prelude), bind(&s.cur), count * d.hidden);
            }
            const layer = &self.layers[visit.physical];
            if (visit.uses_message) {
                try self.mix(cmd, &s.previous_pass, layer.loop.?.query, &s.messages, traversal.bank, count, 4, 0);
            }
            try self.attention(cmd, layer, cache, visit, tiles, count);
            if (visit.save_block) |id| {
                try self.norm(cmd, bind(&s.cur), null, bind(&s.saved), count, d.hidden, id * batch * d.hidden);
            }
            try self.ffn(cmd, layer, cache, traversal.active, visit.uses_message, count);
        }
        if (output) {
            const in = rowSlice(&s.inputs, d.hidden, 0);
            try self.mix(cmd, &s.cur, self.output_query, &s.inputs, traversal.active, 1, 1, count - 1);
            try self.norm(cmd, in, self.output_norm, in, 1, d.hidden, 0);
            if (self.output.info.type_ == .q6_k) {
                const n = groups(d.vocab, 8);
                const x = @min(n, self.instance.device_props.limits.maxComputeWorkGroupCount[0]);
                try self.dispatch(cmd, .output_q6_k, .{ x, 1, groups(n, x) }, &.{
                    weight(self.output), in, bind(&s.logits),
                }, Output{ .rows = d.vocab, .cols = d.hidden });
            } else {
                try self.matrix(cmd, self.output, in, bind(&s.logits), d.vocab, 1, 0, false);
            }
            const args = &.{ bind(&s.logits), bind(&s.candidates), bind(&s.next) };
            try self.dispatch(cmd, .argmax, .{ logit_partials, 1, 1 }, args, Argmax{ .n = d.vocab, .phase = 0 });
            try self.dispatch(cmd, .argmax, .{ 1, 1, 1 }, args, Argmax{ .n = logit_partials, .phase = 1 });
            cmd.computeToTransferBarrier();
            copyBuffer(cmd, &s.next, &self.result, s.next.size);
            // A cached chat prefix can be reused with different sampling parameters.
            copyBuffer(cmd, &s.logits, &self.readback, s.logits.size);
        }
        try cmd.end();
    }

    fn matrix(self: *const Forward, cmd: *command.CommandBuffer, w: Weight, input: Binding, out: Binding, rows: u32, count: u32, routes: u32, per_route: bool) !void {
        const use_packed = self.pipes.get(.mat_q4_0_q8) != null and w.info.type_ == .q4_0;
        if (use_packed) try self.quantize(cmd, input, @intCast(w.info.dims[0]), if (per_route) count * routes else count);
        try self.recordMatrix(cmd, w, input, out, rows, count, routes, per_route, use_packed);
        cmd.computeReadWriteBarrier();
    }

    fn recordMatrix(self: *const Forward, cmd: *command.CommandBuffer, w: Weight, input: Binding, out: Binding, rows: u32, count: u32, routes: u32, per_route: bool, use_packed: bool) !void {
        const workgroups = groups(rows, 4);
        const x = @min(workgroups, self.instance.device_props.limits.maxComputeWorkGroupCount[0]);
        const kernel = if (use_packed) Kernel.mat_q4_0_q8 else Kernel.forMatrix(w.info.type_);
        try self.dispatchIndependent(cmd, kernel, .{ x, count * @max(routes, 1), groups(workgroups, x) }, &.{
            weight(w), if (use_packed) bind(&self.scratch.quantized_input) else input, bind(&self.scratch.route_ids), out,
        }, Matrix{
            .rows = rows,
            .cols = @intCast(w.info.dims[0]),
            .routes = routes,
            .per_route = @intFromBool(per_route),
            .blocked = @intFromBool(routes != 0 and self.repacked(w)),
        });
    }

    fn quantize(self: *const Forward, cmd: *command.CommandBuffer, input: Binding, width: u32, count: u32) !void {
        try self.dispatch(cmd, .quantize, .{ groups(width, 256), count, 1 }, &.{ input, bind(&self.scratch.quantized_input) }, Quantize{ .width = width });
    }

    fn project(self: *const Forward, cmd: *command.CommandBuffer, w: Weight, input: Binding, out: Binding, count: u32) !void {
        try self.matrix(cmd, w, input, out, @intCast(w.info.dims[1]), count, 0, false);
    }

    const Projection = struct { w: Weight, out: *const Buffer };

    fn projectShared(self: *const Forward, cmd: *command.CommandBuffer, input: Binding, projections: []const Projection, count: u32) !void {
        var input_packed = false;
        for (projections) |p| {
            const use_packed = self.pipes.get(.mat_q4_0_q8) != null and p.w.info.type_ == .q4_0;
            if (use_packed and !input_packed) {
                try self.quantize(cmd, input, @intCast(p.w.info.dims[0]), count);
                input_packed = true;
            }
            try self.recordMatrix(cmd, p.w, input, bind(p.out), @intCast(p.w.info.dims[1]), count, 0, false, use_packed);
        }
        cmd.computeReadWriteBarrier();
    }

    fn copy(self: *const Forward, cmd: *command.CommandBuffer, input: Binding, out: Binding, n: u32) !void {
        try self.dispatch(cmd, .copy, .{ groups(n, threads), 1, 1 }, &.{ input, out }, Count{ .n = n });
    }

    fn norm(self: *const Forward, cmd: *command.CommandBuffer, input: Binding, w: ?Weight, out: Binding, count: u32, stride: u32, offset: u32) !void {
        try self.dispatch(cmd, .norm, .{ count, 1, 1 }, &.{ input, if (w) |v| weight(v) else input, out }, Norm{
            .width = self.d.hidden,
            .out_stride = stride,
            .out_offset = offset,
            .eps = if (w != null) self.d.eps else 1e-6,
            .weighted = @intFromBool(w != null),
        });
    }

    fn mix(self: *const Forward, cmd: *command.CommandBuffer, current: *const Buffer, query: Weight, out: *const Buffer, depth: inkling.Depth, count: u32, queries: u32, first_token: u32) !void {
        try self.dispatch(cmd, .mix, .{ count, queries, 1 }, &.{ bind(current), bind(&self.scratch.saved), weight(query), bind(out) }, Mix{
            .hidden = self.d.hidden,
            .capacity = batch,
            .first_token = first_token,
            .blocks = depth.len,
            .block_ids = depth.ids,
        });
    }

    fn control(self: *const Forward, cmd: *command.CommandBuffer, w: tensors.Controller, input: Binding, i: u32, count: u32) !void {
        const s = &self.scratch;
        const h = self.d.hidden;
        const msg = rowSlice(&s.messages, h, i);
        const pair = bind(&s.paired);
        try self.norm(cmd, input, null, pair, count, 2 * h, 0);
        try self.norm(cmd, msg, null, pair, count, 2 * h, h);
        try self.project(cmd, w.down, pair, bind(&s.rank), count);
        try self.project(cmd, w.gain, bind(&s.rank), bind(&s.gain), count);
        try self.project(cmd, w.forget, bind(&s.rank), bind(&s.forget), count);
        try self.dispatch(cmd, .control, .{ groups(count * h, threads), 1, 1 }, &.{
            input, msg, bind(&s.gain), bind(&s.forget),
        }, Count{ .n = count * h });
    }

    fn conv(self: *const Forward, cmd: *command.CommandBuffer, cache: *const Cache, raw: *const Buffer, w: Weight, dynamic: ?*const Buffer, residual: ?*const Buffer, out: *const Buffer, offset: u32, count: u32) !void {
        const d = self.d;
        const n: u32 = @intCast(w.info.dims[1]);
        try self.dispatch(cmd, .conv, .{ groups(n, threads), 1, 1 }, &.{
            bind(raw),                 weight(w), bind(dynamic orelse raw),   bind(&cache.convolution),
            bind(residual orelse raw), bind(out), bind(&self.scratch.tokens),
        }, Conv{
            .channels = n,
            .taps = d.taps,
            .count = count,
            .history_offset = offset,
            .dynamic_head_dim = if (dynamic != null) d.head_dim else 0,
            .residual = @intFromBool(residual != null),
            .scale = 1.0 / @sqrt(@as(f32, @floatFromInt((if (dynamic != null) 2 * d.taps else d.taps) + 1))),
        });
    }

    fn attention(self: *const Forward, cmd: *command.CommandBuffer, layer: *const tensors.Layer, cache: *const Cache, visit: inkling.Traversal.Step, tile_budget: u32, count: u32) !void {
        const d = self.d;
        const s = &self.scratch;
        const kv = d.kv_heads * d.head_dim;
        const local = d.sliding[visit.physical];
        const distinct = !local or visit.uses_message;
        const q_in = rowSlice(&s.inputs, d.hidden, 0);
        const k_in = if (distinct) rowSlice(&s.inputs, d.hidden, 1) else q_in;
        const v_in = if (distinct) rowSlice(&s.inputs, d.hidden, 2) else q_in;
        if (layer.attn_query) |query| {
            try self.mix(cmd, &s.cur, query, &s.inputs, visit.attn_depth, count, if (local) 1 else 3, 0);
        } else {
            try self.copy(cmd, bind(&s.cur), q_in, count * d.hidden);
            if (!local) {
                try self.copy(cmd, bind(&s.cur), k_in, count * d.hidden);
                try self.copy(cmd, bind(&s.cur), v_in, count * d.hidden);
            }
        }
        if (local and visit.uses_message) {
            try self.copy(cmd, q_in, k_in, count * d.hidden);
            try self.copy(cmd, q_in, v_in, count * d.hidden);
        }
        const inputs = [_]Binding{ q_in, k_in, v_in };
        for (inputs[0..if (distinct) @as(usize, 3) else 1], 0..) |input, i| {
            if (visit.uses_message) try self.control(cmd, layer.loop.?.controls[i], input, @intCast(i), count);
            try self.norm(cmd, input, layer.input_norms[i], input, count, d.hidden, 0);
        }
        const query = [_]Projection{
            .{ .w = layer.q, .out = &s.q },
            .{ .w = layer.relative, .out = &s.relative },
            .{ .w = layer.sink, .out = &s.sink },
            .{ .w = layer.gate, .out = &s.gate },
        };
        const key: Projection = .{ .w = layer.k, .out = &s.k_raw };
        const value: Projection = .{ .w = layer.v, .out = &s.v_raw };
        if (distinct) {
            try self.projectShared(cmd, q_in, &query, count);
            const keys = [_]Projection{ key, .{ .w = layer.k_conv_proj orelse layer.k, .out = &s.k_taps } };
            const values = [_]Projection{ value, .{ .w = layer.v_conv_proj orelse layer.v, .out = &s.v_taps } };
            try self.projectShared(cmd, k_in, keys[0..if (local) 1 else 2], count);
            try self.projectShared(cmd, v_in, values[0..if (local) 1 else 2], count);
        } else {
            const all = query ++ [_]Projection{ key, value };
            try self.projectShared(cmd, q_in, &all, count);
        }
        try self.conv(cmd, cache, &s.k_raw, layer.k_conv, if (local) null else &s.k_taps, null, &s.k, 0, count);
        try self.conv(cmd, cache, &s.v_raw, layer.v_conv, if (local) null else &s.v_taps, null, &s.v, kv, count);
        const alpha = if (local) 0 else d.scale_alpha;
        try self.dispatch(cmd, .clip, .{ d.heads, count, 1 }, &.{ bind(&s.q), weight(layer.q_clip), bind(&s.tokens) }, Clip{
            .dim = d.head_dim,
            .heads = d.heads,
            .scale_floor = d.scale_floor,
            .limit = d.qk_clip,
            .alpha = alpha,
        });
        try self.dispatch(cmd, .clip, .{ d.heads, count, 1 }, &.{ bind(&s.relative), weight(layer.rel_clip), bind(&s.tokens) }, Clip{
            .dim = d.relative_dim,
            .heads = d.heads,
            .scale_floor = d.scale_floor,
            .limit = d.relative_clip,
            .alpha = alpha,
        });
        try self.dispatch(cmd, .kv, .{ d.kv_heads, count, 1 }, &.{
            bind(&s.k), bind(&s.v), bind(&cache.k), bind(&cache.v), bind(&s.tokens),
        }, KvWrite{ .dim = d.head_dim, .heads = d.kv_heads, .capacity = cache.capacity, .clip = d.qk_clip, .eps = d.eps });

        const tiles = if (local) @min(tile_budget, groups(d.window, attention_tile)) else tile_budget;
        const params = Attention{
            .dim = d.head_dim,
            .heads = d.heads,
            .kv_heads = d.kv_heads,
            .capacity = cache.capacity,
            .window = if (local) d.window else 0,
            .relative_dim = d.relative_dim,
            .relative_extent = if (local) d.window else d.relative_extent,
            .tiles = tiles,
        };
        try self.dispatch(cmd, .attention, .{ d.heads, count, tiles }, &.{
            bind(&s.q),            bind(&s.relative), weight(layer.rel_bias), bind(&cache.k), bind(&cache.v),
            bind(&s.attn_partial), bind(&s.tokens),   bind(&s.sink),          bind(&s.gate),  bind(&s.attention),
        }, params);
        if (tiles > 1) try self.dispatch(cmd, .attention_merge, .{ d.heads, count, 1 }, &.{
            bind(&s.attn_partial), bind(&s.sink), bind(&s.gate), bind(&s.attention),
        }, params);
        try self.project(cmd, layer.output, bind(&s.attention), bind(&s.attn_raw), count);
        try self.conv(cmd, cache, &s.attn_raw, layer.attn_post, null, if (visit.save_block != null) null else &s.cur, &s.attn_hidden, 2 * kv, count);
    }

    fn ffn(self: *const Forward, cmd: *command.CommandBuffer, layer: *const tensors.Layer, cache: *const Cache, depth: inkling.Depth, message: bool, count: u32) !void {
        const d = self.d;
        const s = &self.scratch;
        const input = rowSlice(&s.inputs, d.hidden, 0);
        try self.mix(cmd, &s.attn_hidden, layer.ffn_query, &s.inputs, depth, count, 1, 0);
        if (message) try self.control(cmd, layer.loop.?.controls[3], input, 3, count);
        try self.norm(cmd, input, layer.ffn_norm, input, count, d.hidden, 0);
        try self.project(cmd, layer.router, input, bind(&s.route_logits), count);
        try self.dispatch(cmd, .route, .{ count, 1, 1 }, &.{
            bind(&s.route_logits), if (layer.router_bias) |bias| weight(bias) else bind(&s.route_logits),
            bind(&s.route_ids),    bind(&s.route_weights),
        }, Route{ .experts = d.experts, .active = d.active, .biased = @intFromBool(layer.router_bias != null), .scale = d.router_scale });
        try self.matrix(cmd, layer.gate_up, input, bind(&s.gate_up), 2 * d.expert_width, count, d.active, false);
        try self.dispatch(cmd, .situ, .{ groups(d.expert_width, threads), count * d.active, 1 }, &.{ bind(&s.gate_up), bind(&s.expert_hidden) }, Situ{
            .width = d.expert_width,
            .beta = d.situ_beta,
            .linear_beta = d.situ_linear_beta,
        });
        try self.matrix(cmd, layer.down, bind(&s.expert_hidden), bind(&s.expert_result), d.hidden, count, d.active, true);
        try self.dispatch(cmd, .combine, .{ groups(d.hidden, threads), count, 1 }, &.{
            bind(&s.expert_result), bind(&s.route_weights), bind(&s.ffn_raw),
        }, Combine{ .width = d.hidden, .active = d.active });
        const offset = 2 * d.kv_heads * d.head_dim + d.hidden;
        try self.conv(cmd, cache, &s.ffn_raw, layer.ffn_post, null, &s.attn_hidden, &s.cur, offset, count);
    }

    fn dispatch(self: *const Forward, cmd: *command.CommandBuffer, kernel: Kernel, grid: [3]u32, buffers: []const Binding, params: anytype) !void {
        try self.dispatchIndependent(cmd, kernel, grid, buffers, params);
        cmd.computeReadWriteBarrier();
    }

    fn dispatchIndependent(self: *const Forward, cmd: *command.CommandBuffer, kernel: Kernel, grid: [3]u32, buffers: []const Binding, params: anytype) !void {
        const pipe = &self.pipes.getPtrConst(kernel).*.?;
        const data = std.mem.asBytes(&params);
        const spec = kernel.program();
        std.debug.assert(buffers.len == spec.bindings and data.len == spec.push_size);
        if (pipe.uses_push_descriptors) {
            cmd.pushDescAndDispatch(pipe, self.instance.push_descriptor_fn, buffers, data, grid[0], grid[1], grid[2]);
        } else {
            const set = try self.descriptors(pipe, buffers);
            cmd.dispatchWithPush(pipe, set, data, grid[0], grid[1], grid[2]);
        }
    }

    fn descriptors(self: *const Forward, pipe: *const pipeline.Pipeline, buffers: []const Binding) !vk.VkDescriptorSet {
        const info = vk.VkDescriptorSetAllocateInfo{
            .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = self.descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &pipe.descriptor_set_layout,
        };
        var set: vk.VkDescriptorSet = null;
        if (vk.vkAllocateDescriptorSets(self.instance.device, &info, &set) != vk.VK_SUCCESS) return error.DescriptorSetAllocFailed;
        var writes: [max_bindings]vk.VkWriteDescriptorSet = undefined;
        for (buffers, 0..) |*buffer, i| {
            writes[i] = .{
                .sType = vk.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = set,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = buffer,
                .pTexelBufferView = null,
            };
        }
        vk.vkUpdateDescriptorSets(self.instance.device, @intCast(buffers.len), &writes, 0, null);
        return set;
    }
};

fn createDescriptorPool(instance: *const Instance, visits: u32) !vk.VkDescriptorPool {
    // Each visit records at most 56 dispatches including loop controls. Pool
    // space is reused once its command buffer's submission has finished.
    const sets = 64 * visits + 16;
    const size = vk.VkDescriptorPoolSize{ .type = vk.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, .descriptorCount = sets * max_bindings };
    const info = vk.VkDescriptorPoolCreateInfo{
        .sType = vk.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .maxSets = sets,
        .poolSizeCount = 1,
        .pPoolSizes = &size,
    };
    var pool: vk.VkDescriptorPool = null;
    if (vk.vkCreateDescriptorPool(instance.device, &info, null, &pool) != vk.VK_SUCCESS) return error.DescriptorPoolCreateFailed;
    return pool;
}

fn bind(buf: *const Buffer) Binding {
    return .{ .buffer = buf.handle, .offset = 0, .range = buf.size };
}

fn weight(w: Weight) Binding {
    return bind(&w.gpu_buffer);
}

fn rowSlice(buf: *const Buffer, width: u32, index: u32) Binding {
    const size = @as(u64, batch) * width * @sizeOf(f32);
    return .{ .buffer = buf.handle, .offset = index * size, .range = size };
}

fn groups(n: u32, width: u32) u32 {
    return std.math.divCeil(u32, n, width) catch unreachable;
}

fn deviceBuffer(instance: *const Instance, size: u64) !Buffer {
    return Buffer.initDeviceLocal(instance, size, vk.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
}

fn copyBuffer(cmd: *const command.CommandBuffer, src: *const Buffer, dst: *const Buffer, size: u64) void {
    const region = vk.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = size };
    vk.vkCmdCopyBuffer(cmd.handle, src.handle, dst.handle, 1, &region);
}
