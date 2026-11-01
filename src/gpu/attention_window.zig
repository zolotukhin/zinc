//! Causal sliding-window bounds, independent of a GPU API.
//! @section Inference Runtime
//!
//! Share model-layer selection and causal bounds between single-query decode
//! and batched prefill, so every query sees the same intended token history.
const std = @import("std");
const ModelConfig = @import("../model/config.zig").ModelConfig;

/// Gemma 4's sliding layers use the smaller head width and SWA RoPE base.
/// Global layers retain the complete prefix even though the model has a window.
pub fn forLayer(config: ModelConfig, layer_head_dim: u32) u32 {
    return if (config.architecture == .gemma and config.rope_freq_base_swa > 0 and layer_head_dim < config.head_dim)
        config.sliding_window_size
    else
        0;
}

/// `seq_len` includes the query token. The pinned llama.cpp STANDARD mask
/// excludes a key when query_position - key_position >= window_size.
pub fn start(seq_len: u32, window_size: u32) u32 {
    return if (window_size > 0 and seq_len > window_size) seq_len - window_size else 0;
}

test "Gemma sliding and global layers retain distinct attention histories" {
    var cfg = std.mem.zeroes(ModelConfig);
    cfg.architecture = .gemma;
    cfg.head_dim = 512;
    cfg.rope_freq_base_swa = 10000;
    cfg.sliding_window_size = 1024;
    try std.testing.expectEqual(@as(u32, 1024), forLayer(cfg, 256));
    try std.testing.expectEqual(@as(u32, 0), forLayer(cfg, 512));
    cfg.architecture = .qwen35;
    try std.testing.expectEqual(@as(u32, 0), forLayer(cfg, 256));
    cfg.architecture = .gemma;
    cfg.rope_freq_base_swa = 0;
    try std.testing.expectEqual(@as(u32, 0), forLayer(cfg, 256));
}

test "sliding window includes current token and crosses pages without an off-by-one" {
    for ([_]u32{ 1, 1023, 1024, 1025, 1039, 1040, 1041, 2049 }) |seq_len| {
        for ([_]u32{ 0, 1, 16, 1024 }) |window| {
            const first = start(seq_len, window);
            try std.testing.expectEqual(@min(seq_len, if (window > 0) window else seq_len), seq_len - first);
            for (0..seq_len) |key| {
                const masked_by_reference = window > 0 and seq_len - 1 - key >= window;
                try std.testing.expectEqual(!masked_by_reference, key >= first);
            }
        }
    }
    // Consecutive queries in a prefill chunk need different lower bounds.
    try std.testing.expectEqual(@as(u32, 0), start(1024, 1024));
    try std.testing.expectEqual(@as(u32, 1), start(1025, 1024));
    try std.testing.expectEqual(@as(u32, 2), start(1026, 1024));
}
