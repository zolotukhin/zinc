//! GGUF metadata and execution layout for the fixed-loop Inkling 2 Mini decoder.
//! @section Model Format & Loading
//!
//! The layout maps chronological visits to physical weight layers while
//! keeping an independent cache slot for each visit.
const std = @import("std");
pub const default_context_length: u32 = 4096;
const gguf = @import("gguf.zig");

pub const architecture = "inkling2_loop";
pub const max_layers = 64;
pub const max_visits = 128;
pub const max_depth_blocks = 16;
pub const max_context = 1_048_576;
/// Headroom for Vulkan allocations and display use outside the tensor/cache plan.
pub const vulkan_memory_reserve = 256 * 1024 * 1024;

pub const Layout = struct {
    layers: u32,
    hidden: u32,
    heads: u32,
    kv_heads: u32,
    head_dim: u32,
    experts: u32,
    active: u32,
    expert_width: u32,
    vocab: u32,
    padded_vocab: u32,
    context: u32,
    eps: f32,
    window: u32,
    sliding: [max_layers]bool,
    taps: u32,
    block_size: u32,
    relative_dim: u32,
    relative_extent: u32,
    qk_clip: f32,
    relative_clip: f32,
    scale_floor: u32,
    scale_alpha: f32,
    router_scale: f32,
    situ_beta: f32,
    situ_linear_beta: f32,
    logit_cap: f32,
    core_start: u32,
    core_end: u32,
    passes: u32,
    gate_rank: u32,

    pub fn read(file: *const gguf.GGUFFile) !Layout {
        const layers = try integer(file, "block_count");
        if (layers == 0 or layers > max_layers) return error.InvalidInklingMetadata;
        const pattern = file.metadata.get(architecture ++ ".attention.sliding_window_pattern") orelse
            return error.InvalidInklingMetadata;
        var sliding: [max_layers]bool = @splat(false);
        if (pattern != .array or pattern.array.len != layers) return error.InvalidInklingMetadata;
        for (pattern.array, 0..) |item, i| sliding[i] = item.asBool() orelse return error.InvalidInklingMetadata;
        const layout: Layout = .{
            .layers = layers,
            .hidden = try integer(file, "embedding_length"),
            .heads = try uniformInteger(file, "attention.head_count", layers),
            .kv_heads = try uniformInteger(file, "attention.head_count_kv", layers),
            .head_dim = try integer(file, "attention.key_length"),
            .experts = try integer(file, "expert_count"),
            .active = try integer(file, "expert_used_count"),
            .expert_width = try uniformInteger(file, "expert_feed_forward_length", layers),
            .padded_vocab = try integer(file, "vocab_size"),
            .vocab = try integer(file, "vocab_size_unpadded"),
            .context = try integer(file, "context_length"),
            .eps = try positive(file, "attention.layer_norm_rms_epsilon"),
            .window = try integer(file, "attention.sliding_window"),
            .sliding = sliding,
            .taps = try integer(file, "shortconv.l_cache"),
            .block_size = try integer(file, "attn_res.block_size"),
            .relative_dim = try integer(file, "attention.relative_length"),
            .relative_extent = try integer(file, "attention.relative_extent"),
            .qk_clip = try positive(file, "attention.qk_clip"),
            .relative_clip = try positive(file, "attention.relative_clip"),
            .scale_floor = try integer(file, "attention.log_scale_floor"),
            .scale_alpha = try nonnegative(file, "attention.log_scale_alpha", null),
            .router_scale = try nonnegative(file, "expert_weights_scale", 1),
            .situ_beta = try positive(file, "activation.situ_beta"),
            .situ_linear_beta = try positive(file, "activation.situ_linear_beta"),
            .logit_cap = try nonnegative(file, "final_logit_softcapping", 0),
            .core_start = try integer(file, "loop.start"),
            .core_end = try integer(file, "loop.end"),
            .passes = try integer(file, "loop.count"),
            .gate_rank = try integer(file, "loop.gate_rank"),
        };
        if (file.getBool(architecture ++ ".loop.varying") orelse false) return error.UnsupportedInklingLayout;
        if ((file.getU32(architecture ++ ".attention.value_length") orelse layout.head_dim) != layout.head_dim or
            (file.getU32(architecture ++ ".attention.key_length_swa") orelse layout.head_dim) != layout.head_dim or
            (file.getU32(architecture ++ ".attention.value_length_swa") orelse layout.head_dim) != layout.head_dim)
            return error.UnsupportedInklingLayout;
        if (layout.heads > 64 or layout.head_dim > 128 or layout.heads % layout.kv_heads != 0 or
            layout.experts > 256 or layout.active > @min(layout.experts, 16) or layout.relative_dim > 64 or
            layout.hidden > 8192 or layout.expert_width > 8192 or layout.gate_rank > 256 or
            layout.padded_vocab > 1_000_000 or layout.vocab > layout.padded_vocab or
            layout.context > max_context or layout.window > layout.context or layout.relative_extent > layout.context or
            layout.taps < 2 or layout.taps > 64 or
            layout.core_start >= layout.core_end or layout.core_end > layers or
            layout.core_start % layout.block_size != 0 or layout.core_end % layout.block_size != 0 or
            layout.passes < 2 or layout.passes > 8)
            return error.UnsupportedInklingLayout;
        if (layout.visitCount() > max_visits or (layout.visitCount() - 1) / layout.block_size + 1 > max_depth_blocks)
            return error.UnsupportedInklingLayout;
        const tokens = file.metadata.get("tokenizer.ggml.tokens") orelse return error.InvalidInklingMetadata;
        if (tokens != .array or tokens.array.len != layout.vocab) return error.InvalidInklingMetadata;
        return layout;
    }

    pub fn visitCount(self: Layout) u32 {
        return self.layers + (self.passes - 1) * (self.core_end - self.core_start);
    }

    /// Traverse the prefix, the loop body once per pass, and then the suffix.
    /// The visit index is the cache slot. The physical index only selects weights.
    pub fn physicalLayer(self: Layout, visit: u32) u32 {
        if (visit < self.core_start) return visit;
        const span = self.core_end - self.core_start;
        const loop_visits = self.passes * span;
        const offset = visit - self.core_start;
        if (offset < loop_visits) return self.core_start + offset % span;
        return self.core_end + offset - loop_visits;
    }
};

pub const Depth = struct {
    ids: [max_depth_blocks]u32 = @splat(0),
    len: u32 = 0,

    pub fn slice(self: *const Depth) []const u32 {
        return self.ids[0..self.len];
    }

    fn append(self: *Depth, id: u32) void {
        self.ids[self.len] = id;
        self.len += 1;
    }
};

/// The saved residual blocks visible at each visit. On a loop restart the
/// active depth is restored to the prelude and the prior pass becomes the
/// separate message bank.
pub const Traversal = struct {
    active: Depth = .{},
    prelude: Depth = .{},
    bank: Depth = .{},

    pub const Step = struct {
        physical: u32,
        uses_message: bool,
        save_prelude: bool,
        restart_pass: bool,
        attn_depth: Depth,
        save_block: ?u32,
    };

    pub fn visit(self: *Traversal, d: Layout, index: u32) Step {
        const physical = d.physicalLayer(index);
        const span = d.core_end - d.core_start;
        const uses_message = index >= d.core_start + span and index < d.core_start + d.passes * span;
        const save_prelude = index == d.core_start;
        const restart_pass = uses_message and physical == d.core_start;
        if (save_prelude) {
            self.prelude = self.active;
        } else if (restart_pass) {
            self.bank = .{};
            for (self.active.slice()[self.prelude.len + 1 ..]) |id| self.bank.append(id);
            self.active = self.prelude;
        }
        const attn_depth = self.active;
        const save_block = if (physical % d.block_size == 0) index / d.block_size else null;
        if (save_block) |id| self.active.append(id);
        return .{
            .physical = physical,
            .uses_message = uses_message,
            .save_prelude = save_prelude,
            .restart_pass = restart_pass,
            .attn_depth = attn_depth,
            .save_block = save_block,
        };
    }
};

fn integer(file: *const gguf.GGUFFile, comptime field: []const u8) !u32 {
    const value = file.getU32(architecture ++ "." ++ field) orelse return error.InvalidInklingMetadata;
    return if (value > 0) value else error.InvalidInklingMetadata;
}

fn uniformInteger(file: *const gguf.GGUFFile, comptime field: []const u8, layers: u32) !u32 {
    const key = architecture ++ "." ++ field;
    const value = file.metadata.get(key) orelse return error.InvalidInklingMetadata;
    if (value.asU32() != null) return integer(file, field);
    if (value != .array or value.array.len != layers) return error.InvalidInklingMetadata;
    const first = value.array[0].asU32() orelse return error.InvalidInklingMetadata;
    if (first == 0) return error.InvalidInklingMetadata;
    for (value.array[1..]) |item| {
        if (item.asU32() != first) return error.UnsupportedInklingLayout;
    }
    return first;
}

fn nonnegative(file: *const gguf.GGUFFile, comptime field: []const u8, default: ?f32) !f32 {
    const value = file.getF32(architecture ++ "." ++ field) orelse default orelse return error.InvalidInklingMetadata;
    if (!std.math.isFinite(value) or value < 0) return error.InvalidInklingMetadata;
    return value;
}

fn positive(file: *const gguf.GGUFFile, comptime field: []const u8) !f32 {
    const value = try nonnegative(file, field, null);
    return if (value > 0) value else error.InvalidInklingMetadata;
}

test "loop visits share only the middle layers' weights" {
    var layout: Layout = undefined;
    layout.layers = 24;
    layout.core_start = 6;
    layout.core_end = 18;
    layout.passes = 3;
    layout.block_size = 6;
    try std.testing.expectEqual(@as(u32, 48), layout.visitCount());
    const boundaries = [_][2]u32{ .{ 0, 0 }, .{ 5, 5 }, .{ 6, 6 }, .{ 17, 17 }, .{ 18, 6 }, .{ 29, 17 }, .{ 30, 6 }, .{ 41, 17 }, .{ 42, 18 }, .{ 47, 23 } };
    for (boundaries) |pair| try std.testing.expectEqual(pair[1], layout.physicalLayer(pair[0]));

    var traversal: Traversal = .{};
    for (0..layout.visitCount()) |index| {
        const step = traversal.visit(layout, @intCast(index));
        if (index == 18 or index == 30) {
            try std.testing.expect(step.restart_pass);
            try std.testing.expectEqualSlices(u32, &.{0}, step.attn_depth.slice());
            try std.testing.expectEqualSlices(u32, &.{@as(u32, @intCast(index / 6 - 1))}, traversal.bank.slice());
        }
    }
    try std.testing.expectEqualSlices(u32, &.{ 0, 5, 6, 7 }, traversal.active.slice());
}
