//! Correctness defaults for backend inference acceleration.
//! @section Inference Runtime
//!
//! Extra activation quantization is separate from a GGUF's weight format.
//! Keep paths that change qualified continuations behind an explicit override.
const std = @import("std");
const ModelConfig = @import("../model/config.zig").ModelConfig;

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
