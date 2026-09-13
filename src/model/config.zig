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
    /// Decoder layers executed by the ordinary autoregressive forward pass.
    /// GGUF `block_count` may also include appended NextN/MTP draft blocks;
    /// those are tracked separately in `n_nextn_layers`.
    n_layers: u32,
    n_nextn_layers: u32 = 0,
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

pub const LayerCounts = struct {
    decoder: u32,
    nextn: u32,
};

/// Split GGUF `block_count` into ordinary decoder and appended NextN/MTP layers.
/// Invalid metadata is ignored conservatively so existing models keep their
/// original decoder count instead of underflowing or becoming empty.
pub fn normalizeLayerCounts(block_count: u32, declared_nextn: u32) LayerCounts {
    if (declared_nextn == 0 or declared_nextn >= block_count) {
        return .{ .decoder = block_count, .nextn = 0 };
    }
    return .{ .decoder = block_count - declared_nextn, .nextn = declared_nextn };
}

test "normalizeLayerCounts excludes appended NextN blocks" {
    try std.testing.expectEqual(LayerCounts{ .decoder = 64, .nextn = 1 }, normalizeLayerCounts(65, 1));
    try std.testing.expectEqual(LayerCounts{ .decoder = 64, .nextn = 0 }, normalizeLayerCounts(64, 0));
    try std.testing.expectEqual(LayerCounts{ .decoder = 1, .nextn = 0 }, normalizeLayerCounts(1, 1));
}

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

/// Muse Glimmer rotates adjacent dim pairs (llama.cpp LLAMA_ROPE_TYPE_NORM:
/// (2i, 2i+1) with frequency i), while ZINC's rope kernels pair (i, i+half).
/// Reordering each head's Q/K output rows to [even dims..., odd dims...] (and
/// the per-dim q_norm/k_norm weights the same way) makes the kernels' pairing
/// equal to the trained one: pair (i, i+half) of the new layout is (2i, 2i+1)
/// of the original. Scores are dot products over the permuted dims (invariant);
/// V and the output projection are untouched. Quantized rows move whole
/// (blocks run along K), so no requantization.
pub fn museRopePermutedCopy(allocator: std.mem.Allocator, name: []const u8, src: []const u8, n_elems: u64, hidden_dim: u64, head_dim: u64) !?[]u8 {
    const is_q = std.mem.endsWith(u8, name, "attn_q.weight");
    const is_k = std.mem.endsWith(u8, name, "attn_k.weight");
    const is_norm = std.mem.endsWith(u8, name, "attn_q_norm.weight") or std.mem.endsWith(u8, name, "attn_k_norm.weight");
    if (!(is_q or is_k or is_norm)) return null;
    if (head_dim == 0 or head_dim % 2 != 0) return null;
    const half = head_dim / 2;
    const out = try allocator.alloc(u8, src.len);
    errdefer allocator.free(out);
    if (is_norm) {
        if (n_elems != head_dim or n_elems == 0) {
            allocator.free(out);
            return null;
        }
        const esz: usize = src.len / @as(usize, @intCast(n_elems));
        for (0..@intCast(half)) |i| {
            @memcpy(out[i * esz ..][0..esz], src[(2 * i) * esz ..][0..esz]);
            @memcpy(out[(i + @as(usize, @intCast(half))) * esz ..][0..esz], src[(2 * i + 1) * esz ..][0..esz]);
        }
        return out;
    }
    if (hidden_dim == 0 or n_elems % hidden_dim != 0) {
        allocator.free(out);
        return null;
    }
    const rows: usize = @intCast(n_elems / hidden_dim);
    if (rows % @as(usize, @intCast(head_dim)) != 0 or @as(usize, src.len) % rows != 0) {
        allocator.free(out);
        return null;
    }
    const row_bytes: usize = src.len / rows;
    const hd: usize = @intCast(head_dim);
    const hf: usize = @intCast(half);
    var h: usize = 0;
    while (h < rows / hd) : (h += 1) {
        for (0..hf) |i| {
            const dst_even = (h * hd + i) * row_bytes;
            const dst_odd = (h * hd + hf + i) * row_bytes;
            const src_even = (h * hd + 2 * i) * row_bytes;
            const src_odd = (h * hd + 2 * i + 1) * row_bytes;
            @memcpy(out[dst_even..][0..row_bytes], src[src_even..][0..row_bytes]);
            @memcpy(out[dst_odd..][0..row_bytes], src[src_odd..][0..row_bytes]);
        }
    }
    return out;
}

