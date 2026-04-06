//! Platform-independent model types shared by Vulkan and Metal backends.
//! The actual extraction logic lives in loader.zig (Vulkan) and will be
//! duplicated for Metal when needed (the extraction is pure GGUF parsing,
//! but loader.zig has Vulkan imports at the top level).
const std = @import("std");

/// Supported model families inferred from GGUF architecture metadata.
pub const Architecture = enum {
    mistral,
    qwen2,
    qwen2_moe,
    qwen35,
    mamba,
    jamba,
    gemma,
    gpt_oss,
    unknown,
};

/// Normalized model dimensions and routing metadata extracted from GGUF fields.
pub const ModelConfig = struct {
    architecture: Architecture,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    hidden_dim: u32,
    intermediate_dim: u32,
    vocab_size: u32,
    context_length: u32,
    rope_freq_base: f32,
    rope_freq_base_swa: f32 = 0,
    rms_norm_eps: f32 = 1e-6,
    n_experts: u32,
    n_experts_used: u32,
    rope_dim: u32,
    ssm_d_conv: u32,
    ssm_d_inner: u32,
    ssm_d_state: u32,
    ssm_dt_rank: u32,
    ssm_n_group: u32,
    full_attn_interval: u32,
    shared_expert_intermediate_dim: u32,
    final_logit_softcapping: f32 = 0.0,
    attn_scale: f32 = 0.0,
    sliding_window_size: u32 = 0,
    rope_scaling_factor: f32 = 0.0,
    rope_original_context: u32 = 0,
    rope_sections: [4]u32 = .{ 0, 0, 0, 0 },
};

/// Parse architecture string from GGUF metadata.
pub fn parseArchitecture(arch_str: []const u8) Architecture {
    if (std.mem.eql(u8, arch_str, "llama")) return .unknown;
    if (std.mem.eql(u8, arch_str, "mistral")) return .mistral;
    if (std.mem.eql(u8, arch_str, "qwen2")) return .qwen2;
    if (std.mem.eql(u8, arch_str, "qwen3")) return .qwen2;
    if (std.mem.eql(u8, arch_str, "qwen2moe")) return .qwen2_moe;
    if (std.mem.eql(u8, arch_str, "qwen3moe")) return .qwen2_moe;
    if (std.mem.eql(u8, arch_str, "qwen35moe")) return .qwen2_moe;
    if (std.mem.eql(u8, arch_str, "qwen35")) return .qwen35;
    if (std.mem.eql(u8, arch_str, "mamba")) return .mamba;
    if (std.mem.eql(u8, arch_str, "jamba")) return .jamba;
    if (std.mem.eql(u8, arch_str, "gemma")) return .gemma;
    if (std.mem.eql(u8, arch_str, "gemma2")) return .gemma;
    if (std.mem.eql(u8, arch_str, "gemma3")) return .gemma;
    if (std.mem.eql(u8, arch_str, "gemma4")) return .gemma;
    if (std.mem.eql(u8, arch_str, "gpt-oss")) return .gpt_oss;
    if (std.mem.eql(u8, arch_str, "gpt_oss")) return .gpt_oss;
    if (std.mem.eql(u8, arch_str, "openai-moe")) return .gpt_oss;
    return .unknown;
}

test "parseArchitecture" {
    try std.testing.expectEqual(Architecture.unknown, parseArchitecture("llama"));
    try std.testing.expectEqual(Architecture.qwen2, parseArchitecture("qwen2"));
    try std.testing.expectEqual(Architecture.qwen2, parseArchitecture("qwen3"));
    try std.testing.expectEqual(Architecture.qwen2_moe, parseArchitecture("qwen2moe"));
    try std.testing.expectEqual(Architecture.qwen2_moe, parseArchitecture("qwen3moe"));
    try std.testing.expectEqual(Architecture.qwen2_moe, parseArchitecture("qwen35moe"));
    try std.testing.expectEqual(Architecture.qwen35, parseArchitecture("qwen35"));
    try std.testing.expectEqual(Architecture.mamba, parseArchitecture("mamba"));
    try std.testing.expectEqual(Architecture.gemma, parseArchitecture("gemma"));
    try std.testing.expectEqual(Architecture.gemma, parseArchitecture("gemma2"));
    try std.testing.expectEqual(Architecture.gemma, parseArchitecture("gemma3"));
    try std.testing.expectEqual(Architecture.gemma, parseArchitecture("gemma4"));
    try std.testing.expectEqual(Architecture.unknown, parseArchitecture("gpt2"));
}
