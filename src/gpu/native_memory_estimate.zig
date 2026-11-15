//! Metadata-only ROCm/CUDA preflight estimate. Includes context and serving
//! slots; lazy prefill buffers and driver/compiler allocations remain estimates.
//! This must not be advertised as an allocation or inference success test.
//! @section Hardware Detection
const std = @import("std");
const ModelConfig = @import("../model/config.zig").ModelConfig;
const kv_storage = @import("../compute/kv_storage.zig");
const memory_plan = @import("memory_plan.zig");

pub const Estimate = struct {
    weights_bytes: u64,
    kv_bytes: u64,
    recurrent_bytes: u64,
    runtime_reserve_bytes: u64,
    estimated_total_bytes: u64,
    context_length: u32,
    slots: u32,
};

/// Use the native server's F32 default (or explicit Q8). Multiple slots are
/// additive to the forward's original single-sequence KV/recurrent buffers.
/// Gemma metadata can combine per-layer maxima: this is deliberately an
/// estimate, not a reason to reject a model that has not tried allocating.
pub fn estimate(config: ModelConfig, weights: u64, context: u32, slots: u32, format: kv_storage.Format) !Estimate {
    if (context == 0 or slots == 0 or config.n_layers == 0 or config.head_dim == 0 or config.n_kv_heads == 0) return error.InvalidModelDimensions;
    if (config.ssm_d_inner > 0 and (config.ssm_dt_rank == 0 or config.ssm_d_conv == 0)) return error.InvalidModelDimensions;
    const copies = if (slots > 1) try std.math.add(u64, slots, 1) else 1;
    const kv_dim = try std.math.mul(u32, config.n_kv_heads, config.head_dim);
    var decoder_config = config;
    decoder_config.n_nextn_layers = 0;
    const decoder_kv_layers = memory_plan.kvLayerCount(decoder_config);
    const kv_layers = try std.math.add(u32, decoder_kv_layers, config.n_nextn_layers);
    const per_row: u64 = try format.rowBytes(kv_dim);
    const kv_bytes = try mul(&.{ per_row, context, kv_layers, 2, copies });
    const ssm_layers = config.n_layers - decoder_kv_layers;
    const recurrent_per_layer: u64 = if (config.ssm_d_inner > 0) blk: {
        const conv_channels = try std.math.add(u64, config.ssm_d_inner, try mul(&.{ 2, config.ssm_n_group, config.ssm_d_state }));
        const conv = try mul(&.{ config.ssm_d_conv - 1, conv_channels, 4 });
        const head = config.ssm_d_inner / config.ssm_dt_rank;
        const state = try mul(&.{ config.ssm_dt_rank, head, head, 4 });
        break :blk try std.math.add(u64, conv, state);
    } else 0;
    const recurrent_bytes = try mul(&.{ recurrent_per_layer, ssm_layers, copies });
    // A visible reserve, rather than treating weights + KV as the entire
    // runtime. Actual lazy scratch, optional caches and JIT costs can be larger.
    const reserve: u64 = 256 * 1024 * 1024;
    const total = try std.math.add(u64, try std.math.add(u64, weights, kv_bytes), try std.math.add(u64, recurrent_bytes, reserve));
    return .{ .weights_bytes = weights, .kv_bytes = kv_bytes, .recurrent_bytes = recurrent_bytes, .runtime_reserve_bytes = reserve, .estimated_total_bytes = total, .context_length = context, .slots = slots };
}

fn mul(values: []const u64) !u64 {
    var product: u64 = 1;
    for (values) |value| product = try std.math.mul(u64, product, value);
    return product;
}

test "native memory includes context, additive serving slots and Q8 scales" {
    var config = std.mem.zeroes(ModelConfig);
    config.n_layers = 4;
    config.n_kv_heads = 2;
    config.head_dim = 128;
    const one = try estimate(config, 1000, 2048, 1, .f32);
    const longer = try estimate(config, 1000, 4096, 1, .f32);
    const parallel = try estimate(config, 1000, 2048, 4, .f32);
    const q8 = try estimate(config, 1000, 2048, 1, .q8);
    try std.testing.expectEqual(one.kv_bytes * 2, longer.kv_bytes);
    try std.testing.expectEqual(one.kv_bytes * 5, parallel.kv_bytes);
    try std.testing.expectEqual(one.kv_bytes / 128 * 36, q8.kv_bytes);
    config.n_nextn_layers = 1;
    const draft = try estimate(config, 1000, 2048, 1, .f32);
    try std.testing.expectEqual(one.kv_bytes / 4 * 5, draft.kv_bytes);
    config.n_nextn_layers = 0;
    try std.testing.expect(one.estimated_total_bytes > one.weights_bytes + one.kv_bytes);
    config.ssm_d_inner = 1024;
    config.ssm_d_conv = 4;
    config.ssm_dt_rank = 16;
    config.ssm_n_group = 1;
    config.ssm_d_state = 64;
    config.full_attn_interval = 2;
    const hybrid = try estimate(config, 1000, 2048, 1, .f32);
    try std.testing.expect(hybrid.recurrent_bytes > 0);
    try std.testing.expectEqual(one.kv_bytes / 2, hybrid.kv_bytes);
    try std.testing.expectError(error.Overflow, estimate(config, std.math.maxInt(u64), 2048, 1, .f32));
    config.head_dim = 0;
    try std.testing.expectError(error.InvalidModelDimensions, estimate(config, 1000, 2048, 1, .f32));
}
