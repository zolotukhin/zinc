//! ROCm text inference for the fixed-loop Inkling 2 Mini architecture.
//! @section Inference Runtime
//!
//! Prefill and decode share a layer schedule and pair each visit's attention
//! cache with the causal convolution history. The looped visits share weights.
const std = @import("std");
const gguf = @import("../model/gguf.zig");
const inkling = @import("../model/inkling.zig");
const loader = @import("../model/loader_cuda.zig");
const pipeline = @import("../cuda/pipeline.zig");
const buffer = @import("../cuda/buffer.zig");
const command = @import("../cuda/command.zig");
const shim = @import("../cuda/c.zig").shim;

const log = std.log.scoped(.forward_inkling);
const Buffer = buffer.CudaBuffer;
const tensors = @import("../model/inkling_weights.zig").Tensors(loader.Model, loader.LoadedTensor);
const Weight = tensors.Weight;
const Layer = tensors.Layer;
const Controller = tensors.Controller;
const modelWeight = tensors.modelWeight;
const Layout = inkling.Layout;
const empty_buffer = Buffer{ .handle = null, .size = 0 };
const threads = 64;
const batch = 64;
const attn_tile = 128;
const logit_partials = 128;
const graph_buckets = 1 + @as(usize, std.math.log2_int_ceil(u32, inkling.max_context / attn_tile));

const Kernel = enum {
    inkling_embed,
    inkling_norm,
    inkling_mix,
    inkling_pair_norm,
    inkling_control,
    inkling_control_rows,
    inkling_copy,
    inkling_matvec,
    inkling_matvec4,
    inkling_experts4,
    inkling_f32_decode,
    inkling_quantize,
    inkling_repack_q4,
    inkling_q4_0_q8,
    inkling_q4_0_wmma,
    build_expert_order_padded_4,
    build_expert_order_padded_16,
    inkling_conv,
    inkling_clip,
    inkling_kv,
    inkling_qkv_decode,
    inkling_attention,
    inkling_attention_merge,
    inkling_route,
    inkling_situ,
    inkling_combine,
    inkling_combine_conv,
    inkling_argmax_partial,
    inkling_argmax,
    dmmv_q6k_fast,
};
const kernel_fields = std.meta.fields(Kernel);
const empty_pipeline = pipeline.CudaPipeline{ .handle = null, .max_threads = 0, .shared_mem = 0 };

const Matrix = extern struct { rows: u32, cols: u32, kind: u32, routes: u32, per_route: u32, blocked: u32 = 0 };
const MatrixTile = extern struct { rows: u32, cols: u32, count: u32, routes: u32, per_route: u32, blocked: u32 };
const Repack = extern struct { blocks: u32 };
const ExpertOrder = extern struct { count: u32, used: u32, experts: u32, stride: u32, padded: u32 };
const Quantize = extern struct { cols: u32 };
const Dmmv = extern struct { rows: u32, cols: u32, weights_offset: u32 = 0, input_offset: u32 = 0, output_offset: u32 = 0, accum: u32 = 0 };
const Embed = extern struct { hidden: u32, kind: u32 };
const Norm = extern struct { width: u32, offset: u32, eps: f32, weighted: u32 };
const Mix = extern struct { hidden: u32, capacity: u32, first_token: u32, blocks: u32, block_ids: [inkling.max_depth_blocks]u32 };
const PairNorm = extern struct { hidden: u32 };
const ControlRows = extern struct { width: u32, rank: u32 };
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
const DecodeKv = extern struct {
    dim: u32,
    heads: u32,
    kv_heads: u32,
    rel_dim: u32,
    taps: u32,
    scale_floor: u32,
    capacity: u32,
    dynamic: u32,
    q_clip: f32,
    rel_clip: f32,
    alpha: f32,
    eps: f32,
    conv_scale: f32,
};
const Attention = extern struct {
    dim: u32,
    heads: u32,
    kv_heads: u32,
    capacity: u32,
    window: u32,
    rel_dim: u32,
    rel_extent: u32,
    tiles: u32,
};
const Route = extern struct { experts: u32, active: u32, biased: u32, scale: f32 };
const Situ = extern struct { width: u32, beta: f32, linear_beta: f32 };
const Combine = extern struct { hidden: u32, active: u32 };
const CombineConv = extern struct { combine: Combine, taps: u32, offset: u32, scale: f32 };
const Argmax = extern struct { vocab: u32, partials: u32, cap: f32 };

const Projection = struct { weight: Weight, out: *const Buffer };

const Cache = struct {
    capacity: u32 = 0,
    k: Buffer = empty_buffer,
    v: Buffer = empty_buffer,
    convolution: Buffer = empty_buffer,

    fn deinit(self: *Cache) void {
        buffer.freeBuffer(&self.k);
        buffer.freeBuffer(&self.v);
        buffer.freeBuffer(&self.convolution);
    }
};

const Scratch = struct {
    tokens: Buffer = empty_buffer,
    cur: Buffer = empty_buffer,
    prelude: Buffer = empty_buffer,
    previous_pass: Buffer = empty_buffer,
    saved: Buffer = empty_buffer,
    messages: Buffer = empty_buffer,
    inputs: Buffer = empty_buffer,
    paired: Buffer = empty_buffer,
    rank: Buffer = empty_buffer,
    gain: Buffer = empty_buffer,
    forget: Buffer = empty_buffer,
    q: Buffer = empty_buffer,
    k_raw: Buffer = empty_buffer,
    v_raw: Buffer = empty_buffer,
    k: Buffer = empty_buffer,
    v: Buffer = empty_buffer,
    relative: Buffer = empty_buffer,
    sink: Buffer = empty_buffer,
    gate: Buffer = empty_buffer,
    k_taps: Buffer = empty_buffer,
    v_taps: Buffer = empty_buffer,
    attn_partial: Buffer = empty_buffer,
    attention: Buffer = empty_buffer,
    attn_raw: Buffer = empty_buffer,
    attn_hidden: Buffer = empty_buffer,
    route_logits: Buffer = empty_buffer,
    route_ids: Buffer = empty_buffer,
    route_weights: Buffer = empty_buffer,
    expert_order: Buffer = empty_buffer,
    expert_groups: Buffer = empty_buffer,
    gate_up: Buffer = empty_buffer,
    expert_hidden: Buffer = empty_buffer,
    expert_result: Buffer = empty_buffer,
    ffn_raw: Buffer = empty_buffer,
    logits: Buffer = empty_buffer,
    candidates: Buffer = empty_buffer,
    next: Buffer = empty_buffer,
    q8: Buffer = empty_buffer,

    fn init(ctx: ?*shim.CudaCtx, d: Layout, max_ctx: u32, q8: bool) !Scratch {
        const h = @as(usize, batch) * d.hidden * 4;
        const q = @as(usize, batch) * d.heads * d.head_dim * 4;
        const kv = @as(usize, batch) * d.kv_heads * d.head_dim * 4;
        const route = @as(usize, batch) * d.active;
        const expert_groups = expertGroups(batch, d.active, d.experts, 4);
        const expert_order = @max(expert_groups * 4, if (q8) expertGroups(batch, d.active, d.experts, 16) * 16 else 0);
        const sizes = .{
            .tokens = (batch + 1) * @sizeOf(u32),
            .cur = h,
            .prelude = h,
            .previous_pass = h,
            .saved = h * inkling.max_depth_blocks,
            .messages = 4 * h,
            .inputs = 3 * h,
            .paired = 2 * h,
            .rank = @as(usize, batch) * d.gate_rank * 4,
            .gain = h,
            .forget = h,
            .q = q,
            .k_raw = kv,
            .v_raw = kv,
            .k = kv,
            .v = kv,
            .relative = @as(usize, batch) * d.heads * d.relative_dim * 4,
            .sink = @as(usize, batch) * d.heads * 4,
            .gate = q,
            .k_taps = @as(usize, batch) * d.kv_heads * d.taps * 4,
            .v_taps = @as(usize, batch) * d.kv_heads * d.taps * 4,
            .attn_partial = @as(usize, batch) * d.heads * groups(max_ctx, attn_tile) * (d.head_dim + 2) * 4,
            .attention = q,
            .attn_raw = h,
            .attn_hidden = h,
            .route_logits = @as(usize, batch) * d.experts * 4,
            .route_ids = route * 4,
            .route_weights = route * 4,
            .expert_order = @as(usize, expert_order) * @sizeOf(u32),
            .expert_groups = @as(usize, expert_groups) * @sizeOf(u32),
            .gate_up = route * 2 * d.expert_width * 4,
            .expert_hidden = route * d.expert_width * 4,
            .expert_result = route * d.hidden * 4,
            .ffn_raw = h,
            .logits = @as(usize, d.vocab) * 4,
            .candidates = logit_partials * 2 * 4,
            .next = 4,
            .q8 = if (q8) @as(usize, batch) * @max(d.hidden, d.heads * d.head_dim, d.active * d.expert_width) / 32 * 36 else 0,
        };
        var self: Scratch = .{};
        errdefer self.deinit();
        inline for (std.meta.fields(Scratch)) |field| {
            const size = @field(sizes, field.name);
            if (size != 0) @field(self, field.name) = try buffer.createBuffer(ctx, size);
        }
        return self;
    }

    fn deinit(self: *Scratch) void {
        inline for (std.meta.fields(Scratch)) |field| buffer.freeBuffer(&@field(self, field.name));
    }
};

const Depth = inkling.Depth;

pub const ForwardInkling = struct {
    allocator: std.mem.Allocator,
    ctx: ?*shim.CudaCtx,
    d: Layout,
    max_ctx: u32,
    layers: []Layer,
    visits: []Cache,
    embed: Weight,
    embed_norm: Weight,
    output: Weight,
    output_norm: Weight,
    output_query: Weight,
    scratch: Scratch,
    in: [3]Buffer = @splat(empty_buffer),
    messages: [4]Buffer = @splat(empty_buffer),
    pipes: [kernel_fields.len]pipeline.CudaPipeline = @splat(empty_pipeline),
    next_position: u32 = 0,
    use_q8: bool,
    blocked_experts: bool = false,
    use_graphs: bool,
    graphs: [graph_buckets]?*shim.CudaGraph = @splat(null),

    pub fn init(allocator: std.mem.Allocator, model: *loader.Model, max_ctx: u32) !ForwardInkling {
        const d = try Layout.read(&model.gguf_file);
        if (max_ctx == 0 or max_ctx > d.context) return error.InvalidContextLength;
        if (model.inkling_expert_layout == .invalid) return error.InklingDispatchFailed;
        const embed = try modelWeight(model, "token_embd.weight", &.{ d.hidden, d.padded_vocab }, .matrix);
        const embed_norm = try modelWeight(model, "token_embd_norm.weight", &.{d.hidden}, .float);
        const output = try modelWeight(model, "output.weight", &.{ d.hidden, d.padded_vocab }, .matrix);
        const output_norm = try modelWeight(model, "output_norm.weight", &.{d.hidden}, .float);
        const output_query = try modelWeight(model, "output_res_query.weight", &.{d.hidden}, .float);

        const layers = try allocator.alloc(Layer, d.layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| layer.* = try Layer.read(model, d, @intCast(i));

        const visits = try allocator.alloc(Cache, d.visitCount());
        @memset(visits, .{});
        errdefer {
            for (visits) |*visit| visit.deinit();
            allocator.free(visits);
        }
        const kv = d.kv_heads * d.head_dim;
        for (visits, 0..) |*visit, i| {
            const is_local = d.sliding[d.physicalLayer(@intCast(i))];
            visit.capacity = if (is_local) @min(max_ctx, d.window + batch - 1) else max_ctx;
            const bytes = @as(usize, visit.capacity) * kv * @sizeOf(f16);
            visit.k = try buffer.createBuffer(model.ctx, bytes);
            visit.v = try buffer.createBuffer(model.ctx, bytes);
            visit.convolution = try buffer.createBuffer(model.ctx, @as(usize, 2 * kv + 2 * d.hidden) * (d.taps - 1) * @sizeOf(f32));
        }

        const use_q8 = model.inkling_expert_layout == .blocked or tuningEnabled("ZINC_INKLING_Q8");
        const use_graphs = tuningEnabled("ZINC_INKLING_GRAPH") and std.posix.getenv("ZINC_ROCM_TIME_KERNEL") == null;
        var scratch = try Scratch.init(model.ctx, d, max_ctx, use_q8);
        errdefer scratch.deinit();

        var self: ForwardInkling = .{
            .allocator = allocator,
            .ctx = model.ctx,
            .d = d,
            .max_ctx = max_ctx,
            .layers = layers,
            .visits = visits,
            .embed = embed,
            .embed_norm = embed_norm,
            .output = output,
            .output_norm = output_norm,
            .output_query = output_query,
            .scratch = scratch,
            .use_q8 = use_q8,
            .use_graphs = use_graphs,
        };
        errdefer self.freeViewsAndPipelines();
        const row_size = @as(usize, d.hidden) * batch * @sizeOf(f32);
        for (&self.in, 0..) |*view, i| view.* = try buffer.aliasBuffer(&self.scratch.inputs, i * row_size, row_size);
        for (&self.messages, 0..) |*view, i| view.* = try buffer.aliasBuffer(&self.scratch.messages, i * row_size, row_size);
        inline for (kernel_fields, 0..) |field, i| {
            self.pipes[i] = try pipeline.createPipeline(model.ctx, @import("cuda_kernels.zig").source, field.name);
        }
        if (use_q8 and tuningEnabled("ZINC_INKLING_BLOCK_Q4") and model.inkling_expert_layout == .original)
            try self.repackExperts(model);
        self.blocked_experts = model.inkling_expert_layout == .blocked;
        log.info("Inkling 2 Mini: {d} stored layers, {d} layer visits, {d} context", .{ d.layers, d.visitCount(), max_ctx });
        return self;
    }

    fn repackExperts(self: *const ForwardInkling, model: *loader.Model) !void {
        const d = self.d;
        if (d.hidden % 128 != 0 or d.expert_width % 128 != 0 or d.hidden > 4096 or d.expert_width > 4096) return;
        for (self.layers) |layer| {
            if (layer.gate_up.info.type_ != .q4_0 or layer.down.info.type_ != .q4_0) return;
        }
        var cmd = try command.beginCommand(self.ctx);
        model.inkling_expert_layout = .invalid;
        for (self.layers) |layer| {
            for ([_]Weight{ layer.gate_up, layer.down }) |w| {
                const p = Repack{ .blocks = @intCast(w.info.dims[0] / 32) };
                const groups_of_rows: u32 = @intCast(w.info.dims[1] * d.experts / 4);
                cmd.dispatch(&self.pipes[@intFromEnum(Kernel.inkling_repack_q4)], .{ groups_of_rows, 1, 1 }, .{ threads, 1, 1 }, &.{&w.gpu_buffer}, &p, @sizeOf(Repack), 4 * p.blocks * 18);
            }
        }
        if (shim.rocm_commit_and_wait_checked(cmd.handle) == 0) return error.InklingDispatchFailed;
        model.inkling_expert_layout = .blocked;
    }

    fn freeViewsAndPipelines(self: *ForwardInkling) void {
        for (&self.in) |*view| buffer.freeBuffer(view);
        for (&self.messages) |*view| buffer.freeBuffer(view);
        for (&self.pipes) |*pipe| pipeline.freePipeline(pipe);
    }

    pub fn deinit(self: *ForwardInkling) void {
        for (self.graphs) |graph| shim.cuda_graph_free(graph);
        self.freeViewsAndPipelines();
        self.scratch.deinit();
        for (self.visits) |*visit| visit.deinit();
        self.allocator.free(self.visits);
        self.allocator.free(self.layers);
    }

    pub fn allocSlots(self: *const ForwardInkling, count: u32, ctx: u32) !void {
        if (count != 1 or ctx != self.max_ctx) return error.InvalidParallelism;
    }

    pub fn resetSlot(self: *ForwardInkling, slot: u32) !void {
        if (slot != 0) return error.InvalidSlot;
        self.next_position = 0;
    }

    pub fn decodeBatch(self: *ForwardInkling, ids: []const u32, positions: []const u32, slots: []const u32, out: []u32) !void {
        if (ids.len != 1 or positions.len != 1 or slots.len != 1 or out.len != 1 or slots[0] != 0)
            return error.InvalidParallelism;
        out[0] = try self.decodeStep(ids[0], positions[0], true);
    }

    pub fn prefillBatchedSlot(self: *ForwardInkling, ids: []const u32, slot: u32) !u32 {
        try self.resetSlot(slot);
        return self.prefillBatched(ids);
    }

    pub fn prefillBatched(self: *ForwardInkling, ids: []const u32) !u32 {
        if (ids.len == 0 or ids.len > self.max_ctx) return error.InvalidInklingBatch;
        var start: usize = 0;
        while (start < ids.len) {
            const end = @min(ids.len, start + batch);
            const result = try self.runChunk(ids[start..end], @intCast(start), end == ids.len);
            if (end == ids.len) return result;
            start = end;
        }
        unreachable;
    }

    pub fn prefillStep(self: *ForwardInkling, id: u32, position: u32) !void {
        _ = try self.runChunk(&.{id}, position, false);
    }

    pub fn decodeStep(self: *ForwardInkling, id: u32, position: u32, run_layers: bool) !u32 {
        if (!run_layers) return error.InklingRequiresFullStep;
        return self.runChunk(&.{id}, position, true);
    }

    fn dispatch(self: *const ForwardInkling, cmd: *command.CudaCommand, kernel: Kernel, grid: [3]u32, args: []const *const Buffer, params: anytype) void {
        cmd.dispatch(&self.pipes[@intFromEnum(kernel)], grid, .{ threads, 1, 1 }, args, &params, @sizeOf(@TypeOf(params)), 0);
    }

    fn runChunk(self: *ForwardInkling, ids: []const u32, position: u32, sample: bool) !u32 {
        if (ids.len == 0 or ids.len > batch or position >= self.max_ctx or ids.len > self.max_ctx - position)
            return error.InvalidInklingBatch;
        if (position != 0 and position != self.next_position) return error.NonContiguousInklingPosition;
        for (ids) |id| if (id >= self.d.vocab) return error.InvalidInklingToken;

        const count: u32 = @intCast(ids.len);
        var input: [batch + 1]u32 = undefined;
        input[0] = position;
        @memcpy(input[1..][0..count], ids);
        buffer.upload(self.ctx, &self.scratch.tokens, std.mem.sliceAsBytes(input[0 .. count + 1]));
        self.executeChunk(count, groups(position + count, attn_tile), sample) catch |err| {
            self.next_position = 0;
            return err;
        };
        self.next_position = position + count;
        if (!sample) return 0;
        var next_id = self.d.vocab;
        buffer.download(self.ctx, &self.scratch.next, std.mem.asBytes(&next_id));
        if (next_id >= self.d.vocab) return error.InvalidInklingOutput;
        return next_id;
    }

    fn executeChunk(self: *ForwardInkling, count: u32, tiles: u32, sample: bool) !void {
        if (self.use_graphs and count == 1 and sample) {
            // The graph reads the token and position from device memory. Cache
            // power-of-two tile budgets so replay also works across requests.
            const bucket = std.math.log2_int_ceil(u32, tiles);
            const cached = &self.graphs[bucket];
            if (cached.*) |graph| {
                if (shim.rocm_graph_replay(self.ctx, graph) != 0) return;
                self.use_graphs = false;
                return error.InklingDispatchFailed;
            }
            if (shim.cuda_graph_create()) |graph| {
                var kept = false;
                defer if (!kept) shim.cuda_graph_free(graph);
                var cmd = try command.beginCommand(self.ctx);
                if (shim.cuda_graph_begin(self.ctx) != 0) {
                    self.recordChunk(&cmd, count, @min(@as(u32, 1) << @intCast(bucket), groups(self.max_ctx, attn_tile)), sample);
                    const status = shim.rocm_graph_end_recorded(self.ctx, graph, cmd.handle);
                    if (status == 1) {
                        cached.* = graph;
                        kept = true;
                        return;
                    }
                    if (status < 0) {
                        self.use_graphs = false;
                        return error.InklingDispatchFailed;
                    }
                } else {
                    cmd.releaseCompleted();
                }
            }
            self.use_graphs = false;
        }
        var cmd = try command.beginCommand(self.ctx);
        self.recordChunk(&cmd, count, tiles, sample);
        if (shim.rocm_commit_and_wait_checked(cmd.handle) == 0) return error.InklingDispatchFailed;
    }

    fn recordChunk(self: *const ForwardInkling, cmd: *command.CudaCommand, count: u32, tiles: u32, sample: bool) void {
        const d = self.d;
        const s = &self.scratch;
        self.dispatch(cmd, .inkling_embed, .{ groups(d.hidden, threads), count, 1 }, &.{ &self.embed.gpu_buffer, &s.tokens, &s.cur }, Embed{
            .hidden = d.hidden,
            .kind = @intFromEnum(self.embed.info.type_),
        });
        self.norm(cmd, &s.cur, self.embed_norm, &s.cur, count, 0);

        var traversal: inkling.Traversal = .{};
        for (self.visits, 0..) |*cache, index| {
            const step = traversal.visit(d, @intCast(index));
            if (step.save_prelude) {
                self.copy(cmd, &s.cur, &s.prelude, count * d.hidden);
            } else if (step.restart_pass) {
                self.copy(cmd, &s.cur, &s.previous_pass, count * d.hidden);
                self.copy(cmd, &s.prelude, &s.cur, count * d.hidden);
            }
            const layer = &self.layers[step.physical];
            if (step.uses_message) {
                self.mix(cmd, &s.previous_pass, layer.loop.?.query, &s.messages, traversal.bank, count, 4, 0);
            }
            self.attention(cmd, layer, cache, step.attn_depth, d.sliding[step.physical], step.uses_message, step.save_block != null, tiles, count);
            if (step.save_block) |id| self.norm(cmd, &s.cur, null, &s.saved, count, id * batch);
            self.ffn(cmd, layer, cache, traversal.active, step.uses_message, count);
        }

        if (sample) {
            self.mix(cmd, &s.cur, self.output_query, &s.inputs, traversal.active, 1, 1, count - 1);
            self.norm(cmd, &self.in[0], self.output_norm, &self.in[0], 1, 0);
            if (self.output.info.type_ == .q6_k) {
                self.dispatch(cmd, .dmmv_q6k_fast, .{ d.vocab, 1, 1 }, &.{ &self.output.gpu_buffer, &self.in[0], &s.logits }, Dmmv{ .rows = d.vocab, .cols = d.hidden });
            } else {
                self.matrix(cmd, self.output, &self.in[0], &s.logits, d.vocab, 1, 0, false, 0);
            }
            const arg = Argmax{ .vocab = d.vocab, .partials = logit_partials, .cap = d.logit_cap };
            self.dispatch(cmd, .inkling_argmax_partial, .{ logit_partials, 1, 1 }, &.{ &s.logits, &s.candidates }, arg);
            self.dispatch(cmd, .inkling_argmax, .{ 1, 1, 1 }, &.{ &s.candidates, &s.next }, arg);
        }
    }

    fn matrix(self: *const ForwardInkling, cmd: *command.CudaCommand, w: Weight, x: *const Buffer, y: *const Buffer, rows: u32, count: u32, routes: u32, per_route: bool, tile: u32) void {
        const s = &self.scratch;
        const params = Matrix{
            .rows = rows,
            .cols = @intCast(w.info.dims[0]),
            .kind = @intFromEnum(w.info.type_),
            .routes = routes,
            .per_route = @intFromBool(per_route),
            .blocked = @intFromBool(self.blocked_experts and routes != 0),
        };
        const bindings = [_]*const Buffer{ &w.gpu_buffer, x, &s.route_ids, y };
        if (count == 1 and routes == 0 and w.info.type_ == .f32 and params.cols % 4 == 0) {
            self.dispatch(cmd, .inkling_f32_decode, .{ groups(rows, 2), 1, 1 }, &.{ &w.gpu_buffer, x, y }, params);
            return;
        }
        if (self.use_q8 and count >= 8 and routes == 0 and w.info.type_ == .f32 and
            shim.rocm_try_sgemm(self.ctx, rows, params.cols, count, w.gpu_buffer.handle, x.handle, y.handle) != 0)
            return;

        const use_wmma = tile == 16 or (routes == 0 and count >= 8);
        if (self.use_q8 and w.info.type_ == .q4_0 and (count == 1 or use_wmma or params.blocked != 0)) {
            self.packQ8(cmd, x, params.cols, if (per_route) count * routes else count);
            self.packedQ4(cmd, w, y, params, count, use_wmma);
        } else if (tile == 4) {
            const tiles = expertGroups(count, routes, self.d.experts, 4);
            self.dispatch(cmd, .inkling_experts4, .{ groups(rows, 4), tiles, 1 }, &.{
                &w.gpu_buffer, x, &s.expert_order, &s.expert_groups, y,
            }, params);
        } else if (routes == 0 and count % 4 == 0) {
            self.dispatch(cmd, .inkling_matvec4, .{ groups(rows, 4), count / 4, 1 }, &bindings, params);
        } else {
            self.dispatch(cmd, .inkling_matvec, .{ groups(rows, 4), count * @max(routes, 1), 1 }, &bindings, params);
        }
    }

    fn packQ8(self: *const ForwardInkling, cmd: *command.CudaCommand, x: *const Buffer, cols: u32, count: u32) void {
        self.dispatch(cmd, .inkling_quantize, .{ groups(cols, threads), count, 1 }, &.{ x, &self.scratch.q8 }, Quantize{ .cols = cols });
    }

    fn packedQ4(self: *const ForwardInkling, cmd: *command.CudaCommand, w: Weight, y: *const Buffer, p: Matrix, count: u32, use_wmma: bool) void {
        const s = &self.scratch;
        if (!use_wmma) {
            self.dispatch(cmd, .inkling_q4_0_q8, .{ groups(p.rows, 4), count * @max(p.routes, 1), 1 }, &.{
                &w.gpu_buffer, &s.q8, &s.route_ids, y,
            }, p);
        } else {
            const tiles = if (p.routes == 0) groups(count, 16) else expertGroups(count, p.routes, self.d.experts, 16);
            self.dispatch(cmd, .inkling_q4_0_wmma, .{ groups(p.rows, 32), tiles, 1 }, &.{
                &w.gpu_buffer, &s.q8, &s.expert_order, &s.expert_groups, y,
            }, MatrixTile{ .rows = p.rows, .cols = p.cols, .count = count, .routes = p.routes, .per_route = p.per_route, .blocked = p.blocked });
        }
    }

    fn project(self: *const ForwardInkling, cmd: *command.CudaCommand, w: Weight, x: *const Buffer, y: *const Buffer, count: u32) void {
        self.matrix(cmd, w, x, y, @intCast(w.info.dims[1]), count, 0, false, 0);
    }

    fn projectSharedInput(self: *const ForwardInkling, cmd: *command.CudaCommand, x: *const Buffer, projects: []const Projection, count: u32) void {
        var prepared = false;
        for (projects) |p| {
            if (self.use_q8 and (count == 1 or count >= 8) and p.weight.info.type_ == .q4_0) {
                const cols: u32 = @intCast(p.weight.info.dims[0]);
                if (!prepared) {
                    self.packQ8(cmd, x, cols, count);
                    prepared = true;
                }
                self.packedQ4(cmd, p.weight, p.out, .{
                    .rows = @intCast(p.weight.info.dims[1]),
                    .cols = cols,
                    .kind = @intFromEnum(gguf.GGMLType.q4_0),
                    .routes = 0,
                    .per_route = 0,
                }, count, count >= 8);
            } else {
                self.project(cmd, p.weight, x, p.out, count);
            }
        }
    }

    fn norm(self: *const ForwardInkling, cmd: *command.CudaCommand, x: *const Buffer, w: ?Weight, y: *const Buffer, count: u32, offset: u32) void {
        self.dispatch(cmd, .inkling_norm, .{ count, 1, 1 }, &.{ x, if (w) |weight| &weight.gpu_buffer else x, y }, Norm{
            .width = self.d.hidden,
            .offset = offset,
            .eps = if (w != null) self.d.eps else 1e-6,
            .weighted = @intFromBool(w != null),
        });
    }

    fn copy(self: *const ForwardInkling, cmd: *command.CudaCommand, x: *const Buffer, y: *const Buffer, len: u32) void {
        self.dispatch(cmd, .inkling_copy, .{ groups(len, threads), 1, 1 }, &.{ x, y }, Count{ .n = len });
    }

    fn mix(self: *const ForwardInkling, cmd: *command.CudaCommand, cur: *const Buffer, query: Weight, out: *const Buffer, depth: Depth, count: u32, queries: u32, first_token: u32) void {
        var params = Mix{
            .hidden = self.d.hidden,
            .capacity = batch,
            .first_token = first_token,
            .blocks = depth.len,
            .block_ids = @splat(0),
        };
        @memcpy(params.block_ids[0..depth.len], depth.slice());
        self.dispatch(cmd, .inkling_mix, .{ count, queries, 1 }, &.{ cur, &self.scratch.saved, &query.gpu_buffer, out }, params);
    }

    fn control(self: *const ForwardInkling, cmd: *command.CudaCommand, weights: Controller, x: *const Buffer, index: usize, count: u32) void {
        const s = &self.scratch;
        const msg = &self.messages[index];
        self.dispatch(cmd, .inkling_pair_norm, .{ count, 2, 1 }, &.{ x, msg, &s.paired }, PairNorm{ .hidden = self.d.hidden });
        self.project(cmd, weights.down, &s.paired, &s.rank, count);
        if (count == 1 and weights.gain.info.type_ == .f32 and weights.forget.info.type_ == .f32) {
            self.dispatch(cmd, .inkling_control_rows, .{ groups(self.d.hidden, 4), 1, 1 }, &.{
                x, msg, &s.rank, &weights.gain.gpu_buffer, &weights.forget.gpu_buffer,
            }, ControlRows{ .width = self.d.hidden, .rank = self.d.gate_rank });
            return;
        }
        self.project(cmd, weights.gain, &s.rank, &s.gain, count);
        self.project(cmd, weights.forget, &s.rank, &s.forget, count);
        const elems = count * self.d.hidden;
        self.dispatch(cmd, .inkling_control, .{ groups(elems, threads), 1, 1 }, &.{ x, msg, &s.gain, &s.forget }, Count{ .n = elems });
    }

    fn conv(self: *const ForwardInkling, cmd: *command.CudaCommand, cache: *const Cache, x: *const Buffer, w: Weight, dynamic: ?*const Buffer, residual: ?*const Buffer, y: *const Buffer, offset: u32, count: u32) void {
        const taps = self.d.taps;
        const n: u32 = @intCast(w.info.dims[1]);
        self.dispatch(cmd, .inkling_conv, .{ groups(n, threads), 1, 1 }, &.{
            x, &w.gpu_buffer, dynamic orelse x, &cache.convolution, residual orelse x, y, &self.scratch.tokens,
        }, Conv{
            .channels = n,
            .taps = taps,
            .count = count,
            .history_offset = offset,
            .dynamic_head_dim = if (dynamic != null) self.d.head_dim else 0,
            .residual = @intFromBool(residual != null),
            .scale = convolutionScale(taps, dynamic != null),
        });
    }

    fn attention(self: *const ForwardInkling, cmd: *command.CudaCommand, l: *const Layer, cache: *const Cache, depth: Depth, local: bool, with_message: bool, block_start: bool, tile_budget: u32, count: u32) void {
        const d = self.d;
        const s = &self.scratch;
        const kv = d.kv_heads * d.head_dim;
        const queries: u32 = if (local) 1 else 3;
        if (l.attn_query) |query| {
            self.mix(cmd, &s.cur, query, &s.inputs, depth, count, queries, 0);
        } else {
            self.copy(cmd, &s.cur, &self.in[0], count * d.hidden);
            if (!local) {
                self.copy(cmd, &s.cur, &self.in[1], count * d.hidden);
                self.copy(cmd, &s.cur, &self.in[2], count * d.hidden);
            }
        }
        if (local and with_message) {
            self.copy(cmd, &self.in[0], &self.in[1], count * d.hidden);
            self.copy(cmd, &self.in[0], &self.in[2], count * d.hidden);
        }
        const distinct = !local or with_message;
        for (0..if (distinct) @as(usize, 3) else 1) |i| {
            if (with_message) self.control(cmd, l.loop.?.controls[i], &self.in[i], i, count);
            self.norm(cmd, &self.in[i], l.input_norms[i], &self.in[i], count, 0);
        }
        const query = [_]Projection{
            .{ .weight = l.q, .out = &s.q },
            .{ .weight = l.relative, .out = &s.relative },
            .{ .weight = l.sink, .out = &s.sink },
            .{ .weight = l.gate, .out = &s.gate },
        };
        const key_projection: Projection = .{ .weight = l.k, .out = &s.k_raw };
        const value_projection: Projection = .{ .weight = l.v, .out = &s.v_raw };
        if (distinct) {
            self.projectSharedInput(cmd, &self.in[0], &query, count);
            const key = [_]Projection{ key_projection, .{ .weight = l.k_conv_proj orelse l.k, .out = &s.k_taps } };
            const value = [_]Projection{ value_projection, .{ .weight = l.v_conv_proj orelse l.v, .out = &s.v_taps } };
            self.projectSharedInput(cmd, &self.in[1], key[0..if (local) 1 else 2], count);
            self.projectSharedInput(cmd, &self.in[2], value[0..if (local) 1 else 2], count);
        } else {
            const all = query ++ [_]Projection{ key_projection, value_projection };
            self.projectSharedInput(cmd, &self.in[0], &all, count);
        }
        self.finishAttentionInputs(cmd, l, cache, local, count);
        const tiles = if (local) @min(tile_budget, groups(d.window, attn_tile)) else tile_budget;
        const params = Attention{
            .dim = d.head_dim,
            .heads = d.heads,
            .kv_heads = d.kv_heads,
            .capacity = cache.capacity,
            .window = if (local) d.window else 0,
            .rel_dim = d.relative_dim,
            .rel_extent = if (local) d.window else d.relative_extent,
            .tiles = tiles,
        };
        self.dispatch(cmd, .inkling_attention, .{ d.heads, count, tiles }, &.{
            &s.q,            &s.relative, &l.rel_bias.gpu_buffer, &cache.k, &cache.v,
            &s.attn_partial, &s.tokens,   &s.sink,                &s.gate,  &s.attention,
        }, params);
        if (tiles > 1) self.dispatch(cmd, .inkling_attention_merge, .{ d.heads, count, 1 }, &.{
            &s.attn_partial, &s.sink, &s.gate, &s.attention,
        }, params);
        self.project(cmd, l.output, &s.attention, &s.attn_raw, count);
        self.conv(cmd, cache, &s.attn_raw, l.attn_post, null, if (block_start) null else &s.cur, &s.attn_hidden, 2 * kv, count);
    }

    fn finishAttentionInputs(self: *const ForwardInkling, cmd: *command.CudaCommand, l: *const Layer, cache: *const Cache, local: bool, count: u32) void {
        const d = self.d;
        const s = &self.scratch;
        const alpha = if (local) 0 else d.scale_alpha;
        if (count == 1) {
            self.dispatch(cmd, .inkling_qkv_decode, .{ 2 * d.heads + d.kv_heads, 1, 1 }, &.{
                &s.q,                 &s.relative,          &l.q_clip.gpu_buffer, &l.rel_clip.gpu_buffer, &s.k_raw,           &s.v_raw,
                &l.k_conv.gpu_buffer, &l.v_conv.gpu_buffer, &s.k_taps,            &s.v_taps,              &cache.convolution, &cache.k,
                &cache.v,             &s.tokens,
            }, DecodeKv{
                .dim = d.head_dim,
                .heads = d.heads,
                .kv_heads = d.kv_heads,
                .rel_dim = d.relative_dim,
                .taps = d.taps,
                .scale_floor = d.scale_floor,
                .capacity = cache.capacity,
                .dynamic = @intFromBool(!local),
                .q_clip = d.qk_clip,
                .rel_clip = d.relative_clip,
                .alpha = alpha,
                .eps = d.eps,
                .conv_scale = convolutionScale(d.taps, !local),
            });
            return;
        }
        self.conv(cmd, cache, &s.k_raw, l.k_conv, if (local) null else &s.k_taps, null, &s.k, 0, count);
        self.conv(cmd, cache, &s.v_raw, l.v_conv, if (local) null else &s.v_taps, null, &s.v, d.kv_heads * d.head_dim, count);
        self.dispatch(cmd, .inkling_clip, .{ d.heads, count, 1 }, &.{ &s.q, &l.q_clip.gpu_buffer, &s.tokens }, Clip{
            .dim = d.head_dim,
            .heads = d.heads,
            .scale_floor = d.scale_floor,
            .limit = d.qk_clip,
            .alpha = alpha,
        });
        self.dispatch(cmd, .inkling_clip, .{ d.heads, count, 1 }, &.{ &s.relative, &l.rel_clip.gpu_buffer, &s.tokens }, Clip{
            .dim = d.relative_dim,
            .heads = d.heads,
            .scale_floor = d.scale_floor,
            .limit = d.relative_clip,
            .alpha = alpha,
        });
        self.dispatch(cmd, .inkling_kv, .{ d.kv_heads, count, 1 }, &.{ &s.k, &s.v, &cache.k, &cache.v, &s.tokens }, KvWrite{
            .dim = d.head_dim,
            .heads = d.kv_heads,
            .capacity = cache.capacity,
            .clip = d.qk_clip,
            .eps = d.eps,
        });
    }

    fn ffn(self: *const ForwardInkling, cmd: *command.CudaCommand, l: *const Layer, cache: *const Cache, depth: Depth, with_message: bool, count: u32) void {
        const d = self.d;
        const s = &self.scratch;
        self.mix(cmd, &s.attn_hidden, l.ffn_query, &s.inputs, depth, count, 1, 0);
        if (with_message) self.control(cmd, l.loop.?.controls[3], &self.in[0], 3, count);
        self.norm(cmd, &self.in[0], l.ffn_norm, &self.in[0], count, 0);
        self.project(cmd, l.router, &self.in[0], &s.route_logits, count);
        self.dispatch(cmd, .inkling_route, .{ count, 1, 1 }, &.{
            &s.route_logits, if (l.router_bias) |bias| &bias.gpu_buffer else &s.route_logits, &s.route_ids, &s.route_weights,
        }, Route{
            .experts = d.experts,
            .active = d.active,
            .biased = @intFromBool(l.router_bias != null),
            .scale = d.router_scale,
        });
        const tile: u32 = if (count < 8) 0 else if (self.use_q8 and l.gate_up.info.type_ == .q4_0 and l.down.info.type_ == .q4_0) 16 else 4;
        if (tile != 0) {
            const tiles = expertGroups(count, d.active, d.experts, tile);
            self.dispatch(cmd, if (tile == 16) .build_expert_order_padded_16 else .build_expert_order_padded_4, .{ 1, 1, 1 }, &.{
                &s.route_ids, &s.expert_order, &s.expert_groups,
            }, ExpertOrder{
                .count = count,
                .used = d.active,
                .experts = d.experts,
                .stride = d.active,
                .padded = tiles * tile,
            });
        }
        self.matrix(cmd, l.gate_up, &self.in[0], &s.gate_up, 2 * d.expert_width, count, d.active, false, tile);
        self.dispatch(cmd, .inkling_situ, .{ groups(d.expert_width, threads), count * d.active, 1 }, &.{ &s.gate_up, &s.expert_hidden }, Situ{
            .width = d.expert_width,
            .beta = d.situ_beta,
            .linear_beta = d.situ_linear_beta,
        });
        self.matrix(cmd, l.down, &s.expert_hidden, &s.expert_result, d.hidden, count, d.active, true, tile);
        const combine = Combine{ .hidden = d.hidden, .active = d.active };
        const offset = 2 * d.kv_heads * d.head_dim + d.hidden;
        if (count == 1) {
            self.dispatch(cmd, .inkling_combine_conv, .{ groups(d.hidden, threads), 1, 1 }, &.{
                &s.expert_result, &s.route_weights, &l.ffn_post.gpu_buffer, &cache.convolution,
                &s.attn_hidden,   &s.cur,           &s.tokens,
            }, CombineConv{ .combine = combine, .taps = d.taps, .offset = offset, .scale = convolutionScale(d.taps, false) });
            return;
        }
        self.dispatch(cmd, .inkling_combine, .{ groups(d.hidden, threads), count, 1 }, &.{
            &s.expert_result, &s.route_weights, &s.ffn_raw,
        }, combine);
        self.conv(cmd, cache, &s.ffn_raw, l.ffn_post, null, &s.attn_hidden, &s.cur, offset, count);
    }
};

fn convolutionScale(taps: u32, dynamic: bool) f32 {
    return 1.0 / @sqrt(@as(f32, @floatFromInt((if (dynamic) taps * 2 else taps) + 1)));
}

fn tuningEnabled(name: [:0]const u8) bool {
    return std.mem.eql(u8, std.posix.getenv(name) orelse "1", "1");
}

fn groups(n: u32, width: u32) u32 {
    return (n + width - 1) / width;
}

fn expertGroups(count: u32, active: u32, experts: u32, tile: u32) u32 {
    const routes = count * active;
    return (routes + (tile - 1) * @min(routes, experts)) / tile;
}
