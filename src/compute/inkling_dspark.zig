//! Greedy ROCm drafter for the Inkling 2 Mini DSpark companion GGUF.
//! @section Inference Runtime
//!
//! Uses hidden states from the target to propose tokens for verification by the full model.
const std = @import("std");
const host_io = @import("host_io");
const gguf = @import("../model/gguf.zig");
const loader = @import("../model/loader_cuda.zig");
const buffer = @import("../cuda/buffer.zig");
const command = @import("../cuda/command.zig");
const pipeline = @import("../cuda/pipeline.zig");
const shim = @import("../cuda/c.zig").shim;

const Buffer = buffer.CudaBuffer;
const empty = Buffer{ .handle = null, .size = 0 };
const d_hidden = 2304;
const d_ff = 9216;
const d_head = 128;
const d_heads = 24;
const d_kv_heads = 8;
const d_kv = d_kv_heads * d_head;
const d_rank = 256;
const draft_layers = 3;
const target_taps = [_]u32{ 5, 23, 41, 48 };
const target_batch = 64;
const block_rows = 8;
pub const max_drafts = block_rows - 1;
const attn_tile = 128;
const partials = 128;
const eps = 1e-6;

const Kernel = enum {
    dspark_capture,
    dspark_feature_mix,
    dspark_embed,
    dspark_q8_wmma,
    dspark_qk_rope,
    dspark_store_kv,
    dspark_attention,
    dspark_attention_merge,
    dspark_add,
    dspark_swiglu,
    dspark_joint,
    dspark_markov_update,
    dspark_markov_logits,
    dspark_pick,
    inkling_norm,
    inkling_quantize,
    inkling_matvec,
    inkling_argmax_partial,
    dmmv_q6k_btok2,
    dmmv_q6k_btok3,
    dmmv_q6k_btok4,
    dmmv_q6k_btok5,
    dmmv_q6k_btok6,
    dmmv_q6k_btok7,
    dmmv_q6k_btok8,
};
const kernel_fields = @typeInfo(Kernel).@"enum".field_names;
const empty_pipeline = pipeline.CudaPipeline{ .handle = null, .max_threads = 0, .shared_mem = 0 };

const Feature = extern struct { width: u32, stride: u32, index: u32 };
const FeatureMix = extern struct { width: u32, stride: u32, first: u32, eps: f32 };
const Embed = extern struct { width: u32, kind: u32, eps: f32 };
const Matmul = extern struct { rows: u32, cols: u32, count: u32 };
const Matvec = extern struct { rows: u32, cols: u32, kind: u32, routes: u32 = 0, per_route: u32 = 0, blocked: u32 = 0 };
const Dmmv = extern struct { rows: u32, cols: u32, weights_offset: u32 = 0, input_offset: u32 = 0, out_offset: u32 = 0, accum: u32 = 0 };
const Rope = extern struct { heads: u32, dim: u32, position: u32, eps: f32 };
const Kv = extern struct { width: u32, position: u32, capacity: u32 };
const Attention = extern struct { dim: u32, heads: u32, kv_heads: u32, position: u32, capacity: u32, first: u32, rows: u32, tiles: u32 };
const Norm = extern struct { width: u32, offset: u32 = 0, eps: f32, weighted: u32 = 1 };
const Count = extern struct { n: u32 };
const Quantize = extern struct { cols: u32 };
const Markov = extern struct { width: u32, rank: u32, slot: u32 };
const Head = extern struct { vocab: u32, rank: u32, slot: u32 };
const Pick = extern struct { partials: u32, slot: u32 };
const Argmax = extern struct { vocab: u32, partials: u32, cap: f32 };

const LayerWeights = struct {
    ctx_mix: Buffer = empty,
    ctx_norm: Buffer = empty,
    ctx_k: Buffer = empty,
    ctx_v: Buffer = empty,
    ctx_k_norm: Buffer = empty,
    q: Buffer = empty,
    k: Buffer = empty,
    v: Buffer = empty,
    o: Buffer = empty,
    q_norm: Buffer = empty,
    k_norm: Buffer = empty,
    attn_norm: Buffer = empty,
    ffn_norm: Buffer = empty,
    gate: Buffer = empty,
    up: Buffer = empty,
    down: Buffer = empty,

    fn deinit(self: *LayerWeights) void {
        inline for (@typeInfo(LayerWeights).@"struct".field_names) |field| buffer.freeBuffer(&@field(self, field));
    }
};

const Weights = struct {
    arena: Buffer = empty,
    layers: [draft_layers]LayerWeights = @splat(.{}),
    mix: Buffer = empty,
    mix_norm: Buffer = empty,
    embed_norm: Buffer = empty,
    mask: Buffer = empty,
    output_norm: Buffer = empty,
    joint: Buffer = empty,
    joint_bias: Buffer = empty,
    markov_w1: Buffer = empty,
    markov_w2: Buffer = empty,
    markov_bias: Buffer = empty,

    fn load(allocator: std.mem.Allocator, ctx: ?*shim.CudaCtx, path: []const u8, vocab: u32) !Weights {
        const file = try std.Io.Dir.cwd().openFile(host_io.io, path, .{});
        defer file.close(host_io.io);
        const stat = try file.stat(host_io.io);
        const mapped = try std.posix.mmap(null, stat.size, .{ .READ = true }, .{ .TYPE = .PRIVATE }, file.handle, 0);
        defer std.posix.munmap(mapped);
        var g = try gguf.parse(mapped, allocator);
        defer g.deinit();
        if (!std.mem.eql(u8, g.getString("general.architecture") orelse "", "dflash") or
            (g.getU32("dflash.block_count") orelse 0) != draft_layers or
            (g.getU32("dflash.block_size") orelse 0) != block_rows or
            (g.getU32("dflash.embedding_length") orelse 0) != d_hidden or
            (g.getU32("dflash.feed_forward_length") orelse 0) != d_ff or
            (g.getU32("dflash.attention.head_count") orelse 0) != d_heads or
            (g.getU32("dflash.attention.head_count_kv") orelse 0) != d_kv_heads or
            (g.getU32("dflash.rope.dimension_count") orelse 0) != d_head or
            (g.getF32("dflash.rope.freq_base") orelse 0) != 1_000_000 or
            (g.getU32("dflash.attention.sliding_window") orelse 0) != 1025 or
            !(g.getBool("dflash.has_markov_context") orelse false) or
            (g.getBool("dflash.sample_from_anchor") orelse true))
            return error.UnsupportedInklingDraft;
        const taps = g.metadata.get("dflash.target_layers") orelse return error.UnsupportedInklingDraft;
        if (taps != .array or taps.array.len != target_taps.len) return error.UnsupportedInklingDraft;
        for (taps.array, target_taps) |entry, id| if (entry.asU32() != id) return error.UnsupportedInklingDraft;

        var result: Weights = .{};
        errdefer result.deinit();
        const base = g.tensor_data_offset;
        if (base >= mapped.len) return error.InvalidDraftTensor;
        result.arena = try buffer.uploadMmap(ctx, mapped.ptr + base, mapped.len - base);
        const arena = &result.arena;
        result.mix = try view(&g, arena, "markov_ctx_mix.weight", .f32, &.{4});
        result.mix_norm = try view(&g, arena, "markov_ctx_norm.weight", .f32, &.{d_hidden});
        result.embed_norm = try view(&g, arena, "token_embd_norm.weight", .f32, &.{d_hidden});
        result.mask = try view(&g, arena, "mask_embd.weight", .f32, &.{ d_hidden, max_drafts });
        result.output_norm = try view(&g, arena, "output_norm.weight", .f32, &.{d_hidden});
        result.joint = try view(&g, arena, "joint_proj.weight", .bf16, &.{ 2 * (d_hidden + d_rank), 3 * d_rank });
        result.joint_bias = try view(&g, arena, "joint_proj.bias", .f32, &.{3 * d_rank});
        result.markov_w1 = try view(&g, arena, "markov_w1.weight", .bf16, &.{ d_rank, vocab });
        result.markov_w2 = try view(&g, arena, "markov_w2.weight", .bf16, &.{ d_rank, vocab });
        result.markov_bias = try view(&g, arena, "markov_w2.bias", .f32, &.{vocab});

        for (&result.layers, 0..) |*layer, i| {
            layer.ctx_mix = try layerView(&g, arena, i, "ctx_mix", .f32, &.{4});
            layer.ctx_norm = try layerView(&g, arena, i, "ctx_norm", .f32, &.{d_hidden});
            layer.ctx_k = try layerView(&g, arena, i, "attn_k_ctx", .q8_0, &.{ d_hidden, d_kv });
            layer.ctx_v = try layerView(&g, arena, i, "attn_v_ctx", .q8_0, &.{ d_hidden, d_kv });
            layer.ctx_k_norm = try layerView(&g, arena, i, "attn_k_ctx_norm", .f32, &.{d_head});
            layer.q = try layerView(&g, arena, i, "attn_q", .q8_0, &.{ d_hidden, d_heads * d_head });
            layer.k = try layerView(&g, arena, i, "attn_k", .q8_0, &.{ d_hidden, d_kv });
            layer.v = try layerView(&g, arena, i, "attn_v", .q8_0, &.{ d_hidden, d_kv });
            layer.o = try layerView(&g, arena, i, "attn_output", .q8_0, &.{ d_heads * d_head, d_hidden });
            layer.q_norm = try layerView(&g, arena, i, "attn_q_norm", .f32, &.{d_head});
            layer.k_norm = try layerView(&g, arena, i, "attn_k_norm", .f32, &.{d_head});
            layer.attn_norm = try layerView(&g, arena, i, "attn_norm", .f32, &.{d_hidden});
            layer.ffn_norm = try layerView(&g, arena, i, "ffn_norm", .f32, &.{d_hidden});
            layer.gate = try layerView(&g, arena, i, "ffn_gate", .q8_0, &.{ d_hidden, d_ff });
            layer.up = try layerView(&g, arena, i, "ffn_up", .q8_0, &.{ d_hidden, d_ff });
            layer.down = try layerView(&g, arena, i, "ffn_down", .q8_0, &.{ d_ff, d_hidden });
        }
        return result;
    }

    fn deinit(self: *Weights) void {
        for (&self.layers) |*layer| layer.deinit();
        inline for (@typeInfo(Weights).@"struct".field_names) |field| {
            if (comptime !std.mem.eql(u8, field, "arena") and !std.mem.eql(u8, field, "layers"))
                buffer.freeBuffer(&@field(self, field));
        }
        buffer.freeBuffer(&self.arena);
    }
};

fn view(g: *const gguf.GGUFFile, arena: *const Buffer, name: []const u8, kind: gguf.GGMLType, dims: []const u64) !Buffer {
    for (g.tensors.items) |info| {
        if (!std.mem.eql(u8, info.name, name)) continue;
        if (info.type_ != kind or info.n_dims != dims.len or !std.mem.eql(u64, info.dims[0..dims.len], dims))
            return error.InvalidDraftTensor;
        return buffer.aliasBuffer(arena, @intCast(info.offset), @intCast(info.sizeBytes()));
    }
    return error.MissingDraftTensor;
}

fn layerView(g: *const gguf.GGUFFile, arena: *const Buffer, layer: usize, name: []const u8, kind: gguf.GGMLType, dims: []const u64) !Buffer {
    var scratch: [80]u8 = undefined;
    const key = try std.fmt.bufPrint(&scratch, "blk.{d}.{s}.weight", .{ layer, name });
    return view(g, arena, key, kind, dims);
}

const Cache = struct {
    k: Buffer = empty,
    v: Buffer = empty,
    capacity: u32 = 0,
    window: u32 = 0,
};

const Scratch = struct {
    features: Buffer = empty,
    ctx_mix: Buffer = empty,
    f_rnn: Buffer = empty,
    q: Buffer = empty,
    k: Buffer = empty,
    v: Buffer = empty,
    noise_keys: Buffer = empty,
    noise_values: Buffer = empty,
    frequency: Buffer = empty,
    q8: Buffer = empty,
    cur: Buffer = empty,
    norm: Buffer = empty,
    reduced: Buffer = empty,
    partial: Buffer = empty,
    residual: Buffer = empty,
    gate: Buffer = empty,
    up: Buffer = empty,
    logits: Buffer = empty,
    bias_logits: Buffer = empty,
    state: Buffer = empty,
    joint: Buffer = empty,
    rnn: Buffer = empty,
    markov: Buffer = empty,
    candidates: Buffer = empty,
    tokens: Buffer = empty,

    fn init(ctx: ?*shim.CudaCtx, max_ctx: u32, vocab: u32) !Scratch {
        const fp = @sizeOf(f32);
        const u = @sizeOf(u32);
        const sizes = .{
            .features = target_taps.len * target_batch * d_hidden * fp,
            .ctx_mix = target_batch * d_hidden * fp,
            .f_rnn = d_hidden * fp,
            .q = block_rows * d_heads * d_head * fp,
            .k = target_batch * d_kv * fp,
            .v = target_batch * d_kv * fp,
            .noise_keys = block_rows * d_kv * @sizeOf(f16),
            .noise_values = block_rows * d_kv * @sizeOf(f16),
            .frequency = d_head / 2 * fp,
            .q8 = @max(target_batch * d_hidden, block_rows * d_ff) / 32 * 36,
            .cur = block_rows * d_hidden * fp,
            .norm = block_rows * d_hidden * fp,
            .reduced = block_rows * d_heads * d_head * fp,
            .partial = @as(usize, block_rows * d_heads * (d_head + 2)) * divCeil(max_ctx, attn_tile) * fp,
            .residual = block_rows * d_hidden * fp,
            .gate = block_rows * d_ff * fp,
            .up = block_rows * d_ff * fp,
            .logits = @as(usize, block_rows) * vocab * fp,
            .bias_logits = @as(usize, vocab) * fp,
            .state = d_rank * fp,
            .joint = 2 * (d_hidden + d_rank) * fp,
            .rnn = 3 * d_rank * fp,
            .markov = d_rank * fp,
            .candidates = partials * 2 * u,
            .tokens = block_rows * u,
        };
        var result: Scratch = .{};
        errdefer result.deinit();
        inline for (@typeInfo(Scratch).@"struct".field_names) |field|
            @field(result, field) = try buffer.createBuffer(ctx, @field(sizes, field));
        var frequency: [d_head / 2]f32 = undefined;
        for (&frequency, 0..) |*value, i| value.* = std.math.pow(f32, 1_000_000.0, -@as(f32, @floatFromInt(i * 2)) / d_head);
        buffer.upload(ctx, &result.frequency, std.mem.asBytes(&frequency));
        return result;
    }

    fn deinit(self: *Scratch) void {
        inline for (@typeInfo(Scratch).@"struct".field_names) |field| buffer.freeBuffer(&@field(self, field));
    }
};

pub const Draft = struct {
    ctx: ?*shim.CudaCtx,
    embed: *const loader.LoadedTensor,
    output: *const loader.LoadedTensor,
    vocab: u32,
    max_ctx: u32,
    weights: Weights,
    scratch: Scratch,
    cache: [draft_layers]Cache = @splat(.{}),
    pipes: [kernel_fields.len]pipeline.CudaPipeline = @splat(empty_pipeline),
    position: u32 = 0,

    pub fn init(allocator: std.mem.Allocator, ctx: ?*shim.CudaCtx, path: []const u8, max_ctx: u32, vocab: u32, embed: *const loader.LoadedTensor, output: *const loader.LoadedTensor) !Draft {
        if (output.info.type_ != .q6_k or output.info.dims[0] != d_hidden or embed.info.dims[0] != d_hidden)
            return error.UnsupportedInklingDraftTarget;
        var weights = try Weights.load(allocator, ctx, path, vocab);
        errdefer weights.deinit();
        var scratch = try Scratch.init(ctx, max_ctx, vocab);
        errdefer scratch.deinit();
        var self: Draft = .{ .ctx = ctx, .embed = embed, .output = output, .vocab = vocab, .max_ctx = max_ctx, .weights = weights, .scratch = scratch };
        errdefer self.freePipesAndCache();

        for (&self.cache, 0..) |*cache, i| {
            cache.window = if (i == 1) 0 else 1025;
            cache.capacity = if (cache.window == 0) max_ctx else @min(max_ctx, cache.window);
            const bytes = @as(usize, d_kv) * cache.capacity * @sizeOf(f16);
            cache.k = try buffer.createBuffer(ctx, bytes);
            cache.v = try buffer.createBuffer(ctx, bytes);
        }
        inline for (kernel_fields, 0..) |field, i|
            self.pipes[i] = try pipeline.createPipeline(ctx, @import("cuda_kernels.zig").source, field);
        return self;
    }

    fn freePipesAndCache(self: *Draft) void {
        for (&self.pipes) |*pipe| pipeline.freePipeline(pipe);
        for (&self.cache) |*cache| {
            buffer.freeBuffer(&cache.k);
            buffer.freeBuffer(&cache.v);
        }
    }

    pub fn deinit(self: *Draft) void {
        self.freePipesAndCache();
        self.scratch.deinit();
        self.weights.deinit();
    }

    fn dispatch(self: *const Draft, cmd: *command.CudaCommand, kernel: Kernel, grid: [3]u32, args: []const *const Buffer, params: anytype) void {
        cmd.dispatch(&self.pipes[@backingInt(kernel)], grid, .{ 64, 1, 1 }, args, &params, @sizeOf(@TypeOf(params)), 0);
    }

    pub fn capture(self: *const Draft, cmd: *command.CudaCommand, source: *const Buffer, count: u32, visit: u32) void {
        inline for (target_taps, 0..) |tap, i| {
            if (tap == visit) self.dispatch(cmd, .dspark_capture, .{ divCeil(d_hidden, 64), count, 1 }, &.{ source, &self.scratch.features }, Feature{
                .width = d_hidden,
                .stride = target_batch * d_hidden,
                .index = i,
            });
        }
    }

    fn mix(self: *const Draft, cmd: *command.CudaCommand, w: *const Buffer, scale: *const Buffer, out: *const Buffer, first: u32, n: u32) void {
        self.dispatch(cmd, .dspark_feature_mix, .{ n, 1, 1 }, &.{ &self.scratch.features, w, scale, out }, FeatureMix{
            .width = d_hidden,
            .stride = target_batch * d_hidden,
            .first = first,
            .eps = eps,
        });
    }

    fn norm(self: *const Draft, cmd: *command.CudaCommand, source: *const Buffer, weights: *const Buffer, out: *const Buffer, count: u32) void {
        self.dispatch(cmd, .inkling_norm, .{ count, 1, 1 }, &.{ source, weights, out }, Norm{ .width = d_hidden, .eps = eps });
    }

    fn pack(self: *const Draft, cmd: *command.CudaCommand, x: *const Buffer, cols: u32, count: u32) void {
        self.dispatch(cmd, .inkling_quantize, .{ divCeil(cols, 64), count, 1 }, &.{ x, &self.scratch.q8 }, Quantize{ .cols = cols });
    }

    fn matmul(self: *const Draft, cmd: *command.CudaCommand, w: *const Buffer, out: *const Buffer, rows: u32, cols: u32, count: u32) void {
        self.dispatch(cmd, .dspark_q8_wmma, .{ divCeil(rows, 32), divCeil(count, 16), 1 }, &.{ w, &self.scratch.q8, out }, Matmul{
            .rows = rows,
            .cols = cols,
            .count = count,
        });
    }

    fn rope(self: *const Draft, cmd: *command.CudaCommand, x: *const Buffer, w: *const Buffer, heads: u32, count: u32, pos: u32) void {
        self.dispatch(cmd, .dspark_qk_rope, .{ heads, count, 1 }, &.{ x, w, &self.scratch.frequency }, Rope{
            .heads = heads,
            .dim = d_head,
            .position = pos,
            .eps = eps,
        });
    }

    fn store(self: *const Draft, cmd: *command.CudaCommand, keys: *const Buffer, values: *const Buffer, count: u32, pos: u32, capacity: u32) void {
        self.dispatch(cmd, .dspark_store_kv, .{ divCeil(d_kv, 64), count, 1 }, &.{ &self.scratch.k, &self.scratch.v, keys, values }, Kv{
            .width = d_kv,
            .position = pos,
            .capacity = capacity,
        });
    }

    pub fn inject(self: *Draft, pos: u32, count: u32) !void {
        if (count == 0 or count > target_batch or pos + count > self.max_ctx or (pos != 0 and pos != self.position))
            return error.NonContiguousDraftContext;
        const s = &self.scratch;
        var cmd = try command.beginCommand(self.ctx);
        for (&self.weights.layers, &self.cache) |*w, *cache| {
            self.mix(&cmd, &w.ctx_mix, &w.ctx_norm, &s.ctx_mix, 0, count);
            self.pack(&cmd, &s.ctx_mix, d_hidden, count);
            self.matmul(&cmd, &w.ctx_k, &s.k, d_kv, d_hidden, count);
            self.matmul(&cmd, &w.ctx_v, &s.v, d_kv, d_hidden, count);
            self.rope(&cmd, &s.k, &w.ctx_k_norm, d_kv_heads, count, pos);
            self.store(&cmd, &cache.k, &cache.v, count, pos, cache.capacity);
        }
        self.mix(&cmd, &self.weights.mix, &self.weights.mix_norm, &s.f_rnn, count - 1, 1);
        if (shim.rocm_commit_and_wait_checked(cmd.handle) == 0) return error.DraftDispatchFailed;
        self.position = pos + count;
    }

    pub fn propose(self: *const Draft, anchor: u32, pos: u32, out: []u32) !void {
        const count: u32 = @intCast(out.len + 1);
        if (out.len == 0 or count > block_rows or anchor >= self.vocab or pos != self.position or pos + count > self.max_ctx)
            return error.InvalidDraftWindow;
        const s = &self.scratch;
        buffer.upload(self.ctx, &s.tokens, std.mem.asBytes(&anchor));
        var cmd = try command.beginCommand(self.ctx);
        self.dispatch(&cmd, .dspark_embed, .{ count, 1, 1 }, &.{ &self.embed.gpu_buffer, &self.weights.embed_norm, &self.weights.mask, &s.tokens, &s.cur }, Embed{
            .width = d_hidden,
            .kind = @backingInt(self.embed.info.type_),
            .eps = eps,
        });
        for (&self.weights.layers, &self.cache) |*w, *cache| {
            self.norm(&cmd, &s.cur, &w.attn_norm, &s.norm, count);
            self.pack(&cmd, &s.norm, d_hidden, count);
            self.matmul(&cmd, &w.q, &s.q, d_heads * d_head, d_hidden, count);
            self.matmul(&cmd, &w.k, &s.k, d_kv, d_hidden, count);
            self.matmul(&cmd, &w.v, &s.v, d_kv, d_hidden, count);
            self.rope(&cmd, &s.q, &w.q_norm, d_heads, count, pos);
            self.rope(&cmd, &s.k, &w.k_norm, d_kv_heads, count, pos);
            self.store(&cmd, &s.noise_keys, &s.noise_values, count, 0, block_rows);
            const first = if (cache.window == 0) 0 else (pos + 1) -| cache.window;
            const tiles = divCeil(pos + count - first, attn_tile);
            const a = Attention{ .dim = d_head, .heads = d_heads, .kv_heads = d_kv_heads, .position = pos, .capacity = cache.capacity, .first = first, .rows = count, .tiles = tiles };
            self.dispatch(&cmd, .dspark_attention, .{ d_heads, count, tiles }, &.{ &s.q, &cache.k, &cache.v, &s.noise_keys, &s.noise_values, &s.partial }, a);
            self.dispatch(&cmd, .dspark_attention_merge, .{ d_heads, count, 1 }, &.{ &s.partial, &s.reduced }, a);
            self.pack(&cmd, &s.reduced, d_heads * d_head, count);
            self.matmul(&cmd, &w.o, &s.residual, d_hidden, d_heads * d_head, count);
            self.add(&cmd, &s.cur, &s.residual, count * d_hidden);
            self.norm(&cmd, &s.cur, &w.ffn_norm, &s.norm, count);
            self.pack(&cmd, &s.norm, d_hidden, count);
            self.matmul(&cmd, &w.gate, &s.gate, d_ff, d_hidden, count);
            self.matmul(&cmd, &w.up, &s.up, d_ff, d_hidden, count);
            const n = count * d_ff;
            self.dispatch(&cmd, .dspark_swiglu, .{ divCeil(n, 64), 1, 1 }, &.{ &s.gate, &s.up }, Count{ .n = n });
            self.pack(&cmd, &s.gate, d_ff, count);
            self.matmul(&cmd, &w.down, &s.residual, d_hidden, d_ff, count);
            self.add(&cmd, &s.cur, &s.residual, count * d_hidden);
        }
        self.norm(&cmd, &s.cur, &self.weights.output_norm, &s.norm, count);
        const head: Kernel = switch (count) {
            2 => .dmmv_q6k_btok2,
            3 => .dmmv_q6k_btok3,
            4 => .dmmv_q6k_btok4,
            5 => .dmmv_q6k_btok5,
            6 => .dmmv_q6k_btok6,
            7 => .dmmv_q6k_btok7,
            8 => .dmmv_q6k_btok8,
            else => unreachable,
        };
        self.dispatch(&cmd, head, .{ self.vocab, 1, 1 }, &.{ &self.output.gpu_buffer, &s.norm, &s.logits }, Dmmv{ .rows = self.vocab, .cols = d_hidden });
        var slot: u32 = 1;
        while (slot < count) : (slot += 1) self.markovStep(&cmd, slot);
        if (shim.rocm_commit_and_wait_checked(cmd.handle) == 0) return error.DraftDispatchFailed;
        var ids: [block_rows]u32 = @splat(0xffffffff);
        buffer.download(self.ctx, &s.tokens, std.mem.sliceAsBytes(ids[0..count]));
        for (ids[1..count]) |tok| if (tok >= self.vocab) return error.InvalidDraftToken;
        @memcpy(out, ids[1..count]);
    }

    fn add(self: *const Draft, cmd: *command.CudaCommand, dst: *const Buffer, src: *const Buffer, n: u32) void {
        self.dispatch(cmd, .dspark_add, .{ divCeil(n, 64), 1, 1 }, &.{ dst, src }, Count{ .n = n });
    }

    fn markovStep(self: *const Draft, cmd: *command.CudaCommand, slot: u32) void {
        const s = &self.scratch;
        const w = &self.weights;
        const m = Markov{ .width = d_hidden, .rank = d_rank, .slot = slot };
        self.dispatch(cmd, .dspark_joint, .{ divCeil(2 * (d_hidden + d_rank), 64), 1, 1 }, &.{ &s.state, &w.markov_w1, &s.norm, &s.f_rnn, &s.tokens, &s.joint }, m);
        self.dispatch(cmd, .inkling_matvec, .{ divCeil(3 * d_rank, 4), 1, 1 }, &.{ &w.joint, &s.joint, &s.tokens, &s.rnn }, Matvec{
            .rows = 3 * d_rank,
            .cols = 2 * (d_hidden + d_rank),
            .kind = @backingInt(gguf.GGMLType.bf16),
        });
        self.dispatch(cmd, .dspark_markov_update, .{ divCeil(d_rank, 64), 1, 1 }, &.{ &s.rnn, &w.joint_bias, &s.state, &s.markov }, m);
        self.dispatch(cmd, .dspark_markov_logits, .{ divCeil(self.vocab, 4), 1, 1 }, &.{ &w.markov_w2, &w.markov_bias, &s.markov, &s.logits, &s.bias_logits }, Head{ .vocab = self.vocab, .rank = d_rank, .slot = slot });
        self.dispatch(cmd, .inkling_argmax_partial, .{ partials, 1, 1 }, &.{ &s.bias_logits, &s.candidates }, Argmax{ .vocab = self.vocab, .partials = partials, .cap = 0 });
        self.dispatch(cmd, .dspark_pick, .{ 1, 1, 1 }, &.{ &s.candidates, &s.tokens }, Pick{ .partials = partials, .slot = slot });
    }
};

fn divCeil(n: u32, width: u32) u32 {
    return (n + width - 1) / width;
}
