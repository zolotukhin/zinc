//! Correctness defaults for backend prefill paths.
//! @section Inference Runtime
//!
//! Extra activation quantization is separate from a GGUF's weight format.
//! Keep paths that change qualified continuations behind an explicit override.
const std = @import("std");
const ModelConfig = @import("../model/config.zig").ModelConfig;

/// Dense Gemma's Q8 activation prefill changes the copper continuation after
/// a long prefix. F32-activation batched prefill retains the reference choice.
pub fn vulkanActivationQuantDefault(config: ModelConfig) bool {
    return config.architecture != .gemma or config.n_experts != 0;
}

/// ROCm Gemma MoE's batched path changes the first extended copper token by
/// more than the reference near-tie allowance. Serial prefill retains all
/// reference choices; keep the unqualified batched path behind an opt-in.
pub fn rocmGemmaBatchedDefault(config: ModelConfig) bool {
    return config.architecture != .gemma or config.n_experts == 0;
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

test "ROCm Gemma MoE defaults to serial prefill while other model paths retain batching" {
    var cfg = std.mem.zeroes(ModelConfig);
    cfg.architecture = .gemma;
    try std.testing.expect(rocmGemmaBatchedDefault(cfg));
    cfg.n_experts = 128;
    try std.testing.expect(!rocmGemmaBatchedDefault(cfg));
    cfg.architecture = .muse_glimmer;
    try std.testing.expect(rocmGemmaBatchedDefault(cfg));
    cfg.architecture = .qwen35;
    try std.testing.expect(rocmGemmaBatchedDefault(cfg));
}
