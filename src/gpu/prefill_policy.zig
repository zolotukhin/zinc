//! Correctness defaults for backend inference acceleration.
//! @section Inference Runtime
//!
//! Extra activation quantization is separate from a GGUF's weight format.
//! Keep paths that change qualified continuations behind an explicit override.
const std = @import("std");
const ModelConfig = @import("../model/config.zig").ModelConfig;
const GGMLType = @import("../model/gguf.zig").GGMLType;

/// The dense Qwen2/3 Q8_0 path shares weights across prompt tokens using the
/// existing F32-activation tiled GEMM. Do not broaden Intel's other prefill
/// defaults or accept a missing pipeline as evidence of support.
pub fn vulkanDenseQ8PrefillCompatible(config: ModelConfig, push_descriptors: bool, q8_gemm: bool) bool {
    // GGUF qwen2 and qwen3 intentionally share the .qwen2 forward graph.
    return config.architecture == .qwen2 and config.n_experts == 0 and
        config.ssm_d_inner == 0 and config.sliding_window_size == 0 and
        config.full_attn_interval <= 1 and push_descriptors and q8_gemm;
}

/// Limit the qualified Intel default to a complete Q8 graph: LM head plus
/// seven attention/FFN projections. Mixed quantizations retain their existing
/// defaults; a missing projection cannot silently establish eligibility.
pub fn vulkanDenseQ8PrefillDefault(config: ModelConfig, intel: bool, push_descriptors: bool, q8_gemm: bool, types: []const GGMLType) bool {
    if (!intel or config.n_layers == 0 or types.len != 8 or
        !vulkanDenseQ8PrefillCompatible(config, push_descriptors, q8_gemm)) return false;
    for (types) |quant| if (quant != .q8_0) return false;
    return true;
}

/// Intel Q8 GEMM has no serial-shader column limit. A bounded 384-token
/// batch avoids repeated graph submissions; the matched starter diagnostic
/// improved medium/long prefill by 53%/87% over the 96-token control. Gemma's
/// independent tuning override must not silently change this Q8 bound.
pub fn intelBatchedPrefillChunkDefault(gemma_moe: bool, dense_q8: bool, gemma_limit: u32) u32 {
    if (gemma_moe) return gemma_limit;
    return if (dense_q8) 384 else 96;
}

test "Intel Q8 batching is bounded independently of Gemma precision tuning" {
    for ([_]u32{ 64, 384, 4096 }) |gemma_limit| {
        try std.testing.expectEqual(gemma_limit, intelBatchedPrefillChunkDefault(true, false, gemma_limit));
        try std.testing.expectEqual(@as(u32, 384), intelBatchedPrefillChunkDefault(false, true, gemma_limit));
        try std.testing.expectEqual(@as(u32, 96), intelBatchedPrefillChunkDefault(false, false, gemma_limit));
    }
}

test "qualified Intel Q8 default requires every projection and preserves mixed-quant defaults" {
    var cfg = std.mem.zeroes(ModelConfig);
    cfg.architecture = .qwen2;
    cfg.n_layers = 28;
    var types: [8]GGMLType = @splat(.q8_0);
    try std.testing.expect(vulkanDenseQ8PrefillDefault(cfg, true, true, true, &types));
    try std.testing.expect(!vulkanDenseQ8PrefillDefault(cfg, false, true, true, &types));
    try std.testing.expect(!vulkanDenseQ8PrefillDefault(cfg, true, false, true, &types));
    try std.testing.expect(!vulkanDenseQ8PrefillDefault(cfg, true, true, false, &types));
    try std.testing.expect(!vulkanDenseQ8PrefillDefault(cfg, true, true, true, types[0..7]));
    for (0..types.len) |index| {
        for ([_]GGMLType{ .q4_k, .q6_k, .f16, .f32 }) |quant| {
            types[index] = quant;
            try std.testing.expect(!vulkanDenseQ8PrefillDefault(cfg, true, true, true, &types));
        }
        types[index] = .q8_0;
    }
    cfg.n_layers = 0;
    try std.testing.expect(!vulkanDenseQ8PrefillDefault(cfg, true, true, true, &types));
    cfg.n_layers = 28;
    cfg.architecture = .qwen35;
    try std.testing.expect(!vulkanDenseQ8PrefillDefault(cfg, true, true, true, &types));
}

test "dense Qwen Q8 prefill requires its exact graph and F32 activation kernel" {
    var cfg = std.mem.zeroes(ModelConfig);
    cfg.architecture = .qwen2;
    for ([_]u32{ 0, 1 }) |interval| {
        cfg.full_attn_interval = interval;
        try std.testing.expect(vulkanDenseQ8PrefillCompatible(cfg, true, true));
    }
    try std.testing.expect(!vulkanDenseQ8PrefillCompatible(cfg, false, true));
    try std.testing.expect(!vulkanDenseQ8PrefillCompatible(cfg, true, false));
    cfg.n_experts = 8;
    try std.testing.expect(!vulkanDenseQ8PrefillCompatible(cfg, true, true));
    cfg.n_experts = 0;
    cfg.ssm_d_inner = 1024;
    try std.testing.expect(!vulkanDenseQ8PrefillCompatible(cfg, true, true));
    cfg.ssm_d_inner = 0;
    cfg.sliding_window_size = 1024;
    try std.testing.expect(!vulkanDenseQ8PrefillCompatible(cfg, true, true));
    cfg.sliding_window_size = 0;
    cfg.full_attn_interval = 4;
    try std.testing.expect(!vulkanDenseQ8PrefillCompatible(cfg, true, true));
    cfg.full_attn_interval = 1;
    for ([_]@TypeOf(cfg.architecture){ .gemma, .qwen35, .muse_glimmer, .gpt_oss }) |arch| {
        cfg.architecture = arch;
        try std.testing.expect(!vulkanDenseQ8PrefillCompatible(cfg, true, true));
    }
}

/// Carrying a NextN block establishes model compatibility, not verifier
/// correctness. Intel Qwen 3.8 MTP changes qualified greedy choices, while
/// ordinary decode matches the native reference. R9700 controls match with MTP
/// both on and off; preserve that default and keep Intel speculation explicit.
pub fn vulkanMtpRequested(eligible: bool, intel: bool, override: ?[]const u8) bool {
    if (!eligible) return false;
    const raw = override orelse return !intel;
    return raw.len > 0 and
        !std.mem.eql(u8, raw, "0") and
        !std.ascii.eqlIgnoreCase(raw, "off") and
        !std.ascii.eqlIgnoreCase(raw, "false") and
        !std.ascii.eqlIgnoreCase(raw, "no");
}

test "Intel NextN compatibility does not silently enable unqualified speculation" {
    try std.testing.expect(!vulkanMtpRequested(true, true, null));
    try std.testing.expect(vulkanMtpRequested(true, false, null));
    try std.testing.expect(!vulkanMtpRequested(false, false, null));
    for ([_][]const u8{ "", "0", "off", "FALSE", "No" }) |value| {
        try std.testing.expect(!vulkanMtpRequested(true, true, value));
        try std.testing.expect(!vulkanMtpRequested(true, false, value));
    }
    for ([_][]const u8{ "1", "true", "ON" }) |value| {
        try std.testing.expect(vulkanMtpRequested(true, true, value));
        try std.testing.expect(vulkanMtpRequested(true, false, value));
        try std.testing.expect(!vulkanMtpRequested(false, true, value));
    }
}

/// Dense Gemma's Q8 activation prefill changes the copper continuation after
/// a long prefix. F32-activation batched prefill retains the reference choice.
pub fn vulkanActivationQuantDefault(config: ModelConfig) bool {
    return config.architecture != .gemma or config.n_experts != 0;
}

test "dense Gemma uses F32 prefill activations while other existing defaults remain" {
    var cfg = std.mem.zeroes(ModelConfig);
    cfg.architecture = .gemma;
    cfg.hidden_dim = 5376;
    try std.testing.expect(!vulkanActivationQuantDefault(cfg));
    cfg.n_experts = 128;
    try std.testing.expect(vulkanActivationQuantDefault(cfg));
    cfg.n_experts = 0;
    cfg.architecture = .muse_glimmer;
    try std.testing.expect(vulkanActivationQuantDefault(cfg));
    cfg.architecture = .qwen35;
    try std.testing.expect(vulkanActivationQuantDefault(cfg));
}
