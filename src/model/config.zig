//! Platform-independent model types shared by Vulkan and Metal backends.
//! @section Model Format & Loading
//! The actual extraction logic lives in loader.zig (Vulkan) and
//! loader_metal.zig (Metal). Both share this config type for GGUF parsing,
//! but keep separate loaders because loader.zig has Vulkan imports at the top level.
const std = @import("std");

/// Supported model families inferred from GGUF architecture metadata.
/// Multiple GGUF architecture strings may collapse to a single variant —
/// for example `"llama"` maps to `.mistral` and `"qwen3"` maps to `.qwen2`
/// because their forward-pass implementations are identical.
/// `.unknown` is returned for any unrecognised string.
pub const Architecture = enum {
    mistral,
    qwen2,
    qwen2_moe,
    qwen35,
    mamba,
    jamba,
    gemma,
    gpt_oss,
    /// Meta Muse Glimmer 30B (2026): dense causal transformer that combines the
    /// Gemma skeleton (pre/post attn+ffn RMSNorm, sliding-window attention with a
    /// global layer every 4th, SwiGLU, final-logit softcapping) with Qwen-style
    /// gated attention (separate `attn_gate` tensor) and QK-norm. Dense (no MoE),
    /// GQA 32/2 at head_dim 128. GGUF `general.architecture = "muse-glimmer"`.
    muse_glimmer,
    unknown,
};

/// Normalized model dimensions and routing metadata extracted from GGUF fields.
/// SSM-specific fields (`ssm_d_*`, `ssm_n_group`, `full_attn_interval`) default to
/// zero and are only meaningful for Mamba/Jamba/Qwen3.5 hybrid architectures.
/// RoPE section fields (`rope_sections`, `rope_attn_factor`, `rope_scaling_factor`,
/// `rope_original_context`) are used only by the `qwen35` IMRoPE scheme.
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
    /// Multiplier applied to the final logits before softcapping (Cohere/Muse
    /// Glimmer `logit_scale`). 0.0 means "no scaling" (the default for models
    /// that omit the key).
    logit_scale: f32 = 0.0,
    attn_scale: f32 = 0.0,
    sliding_window_size: u32 = 0,
    rope_scaling_factor: f32 = 0.0,
    rope_attn_factor: f32 = 1.0,
    rope_original_context: u32 = 0,
    rope_sections: [4]u32 = .{ 0, 0, 0, 0 },
};

/// Map a GGUF `general.architecture` string to an `Architecture` variant.
/// The mapping is many-to-one: architecturally equivalent families share a variant
/// (e.g. `"llama"` → `.mistral`, `"qwen3"` → `.qwen2`), so callers must not
/// assume the variant name matches the original GGUF string.
/// @param arch_str The raw architecture string from GGUF metadata, e.g. `"qwen2"`.
/// @returns The matching `Architecture` variant, or `.unknown` if unrecognised.
pub fn parseArchitecture(arch_str: []const u8) Architecture {
    if (std.mem.eql(u8, arch_str, "mistral")) return .mistral;
    // LLaMA 2/3.x are architecturally identical to Mistral: dense attention,
    // dense FFN, no Q/K norms, GQA, RoPE. Map both to the same enum so the
    // existing Mistral forward path handles them.
    if (std.mem.eql(u8, arch_str, "llama")) return .mistral;
    if (std.mem.eql(u8, arch_str, "qwen2")) return .qwen2;
    if (std.mem.eql(u8, arch_str, "qwen3")) return .qwen2;
    if (std.mem.eql(u8, arch_str, "qwen2moe")) return .qwen2_moe;
    if (std.mem.eql(u8, arch_str, "qwen3moe")) return .qwen2_moe;
    if (std.mem.eql(u8, arch_str, "qwen35moe")) return .qwen2_moe;
    if (std.mem.eql(u8, arch_str, "qwen3_5_moe")) return .qwen2_moe;
    if (std.mem.eql(u8, arch_str, "qwen36moe")) return .qwen2_moe;
    if (std.mem.eql(u8, arch_str, "qwen3_6_moe")) return .qwen2_moe;
    if (std.mem.eql(u8, arch_str, "qwen35")) return .qwen35;
    if (std.mem.eql(u8, arch_str, "qwen3_5")) return .qwen35;
    if (std.mem.eql(u8, arch_str, "qwen3_5_text")) return .qwen35;
    if (std.mem.eql(u8, arch_str, "qwen36")) return .qwen35;
    if (std.mem.eql(u8, arch_str, "qwen3_6")) return .qwen35;
    if (std.mem.eql(u8, arch_str, "qwen3_6_text")) return .qwen35;
    if (std.mem.eql(u8, arch_str, "mamba")) return .mamba;
    if (std.mem.eql(u8, arch_str, "jamba")) return .jamba;
    if (std.mem.eql(u8, arch_str, "gemma")) return .gemma;
    if (std.mem.eql(u8, arch_str, "gemma2")) return .gemma;
    if (std.mem.eql(u8, arch_str, "gemma4")) return .gemma;
    if (std.mem.eql(u8, arch_str, "gpt-oss")) return .gpt_oss;
    if (std.mem.eql(u8, arch_str, "gpt_oss")) return .gpt_oss;
    if (std.mem.eql(u8, arch_str, "openai-moe")) return .gpt_oss;
    if (std.mem.eql(u8, arch_str, "muse-glimmer")) return .muse_glimmer;
    if (std.mem.eql(u8, arch_str, "muse_glimmer")) return .muse_glimmer;
    if (std.mem.eql(u8, arch_str, "museglimmer")) return .muse_glimmer;
    return .unknown;
}

test "parseArchitecture" {
    try std.testing.expectEqual(Architecture.qwen2, parseArchitecture("qwen2"));
    try std.testing.expectEqual(Architecture.qwen2, parseArchitecture("qwen3"));
    try std.testing.expectEqual(Architecture.qwen2_moe, parseArchitecture("qwen2moe"));
    try std.testing.expectEqual(Architecture.qwen2_moe, parseArchitecture("qwen3moe"));
    try std.testing.expectEqual(Architecture.qwen2_moe, parseArchitecture("qwen35moe"));
    try std.testing.expectEqual(Architecture.qwen2_moe, parseArchitecture("qwen3_5_moe"));
    try std.testing.expectEqual(Architecture.qwen2_moe, parseArchitecture("qwen36moe"));
    try std.testing.expectEqual(Architecture.qwen2_moe, parseArchitecture("qwen3_6_moe"));
    try std.testing.expectEqual(Architecture.qwen35, parseArchitecture("qwen35"));
    try std.testing.expectEqual(Architecture.qwen35, parseArchitecture("qwen3_5"));
    try std.testing.expectEqual(Architecture.qwen35, parseArchitecture("qwen3_5_text"));
    try std.testing.expectEqual(Architecture.qwen35, parseArchitecture("qwen36"));
    try std.testing.expectEqual(Architecture.qwen35, parseArchitecture("qwen3_6"));
    try std.testing.expectEqual(Architecture.qwen35, parseArchitecture("qwen3_6_text"));
    try std.testing.expectEqual(Architecture.mamba, parseArchitecture("mamba"));
    try std.testing.expectEqual(Architecture.gemma, parseArchitecture("gemma"));
    try std.testing.expectEqual(Architecture.gemma, parseArchitecture("gemma2"));
    try std.testing.expectEqual(Architecture.gemma, parseArchitecture("gemma4"));
    try std.testing.expectEqual(Architecture.mistral, parseArchitecture("mistral"));
    try std.testing.expectEqual(Architecture.mistral, parseArchitecture("llama"));
    try std.testing.expectEqual(Architecture.gpt_oss, parseArchitecture("gpt-oss"));
    try std.testing.expectEqual(Architecture.muse_glimmer, parseArchitecture("muse-glimmer"));
    try std.testing.expectEqual(Architecture.muse_glimmer, parseArchitecture("muse_glimmer"));
    try std.testing.expectEqual(Architecture.unknown, parseArchitecture("gpt2"));
}
