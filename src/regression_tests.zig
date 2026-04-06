//! Source-level regression guards for bugs that are hard to cover with unit-only GPU tests.
const std = @import("std");

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    try std.testing.expect(std.mem.indexOf(u8, haystack, needle) != null);
}

fn expectNotContains(haystack: []const u8, needle: []const u8) !void {
    try std.testing.expect(std.mem.indexOf(u8, haystack, needle) == null);
}

fn expectContainsNear(haystack: []const u8, marker: []const u8, needle: []const u8, window: usize) !void {
    const start = std.mem.indexOf(u8, haystack, marker) orelse return error.TestExpectedEqual;
    const end = @min(start + window, haystack.len);
    try std.testing.expect(std.mem.indexOf(u8, haystack[start..end], needle) != null);
}

fn expectMultiSubgroupFallback(shader_src: []const u8, reduce_name: []const u8) !void {
    try expectContains(shader_src, "gl_NumSubgroups > 1u");
    try expectContains(shader_src, "gl_SubgroupInvocationID == 0u");
    try expectContains(shader_src, reduce_name);
    try expectContains(shader_src, "barrier();");
}

test "decode loop keeps transfer-copy split for packed Q and gate" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "Qwen3Next packs per-head [Q(head_dim), gate(head_dim)] blocks.");
    try expectContains(src, "self.decode_cmd.computeToTransferBarrier();");
    try expectContains(src, "self.decode_cmd.transferToComputeBarrier();");
}

test "decode loop applies packed attention gate after flash attention" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "// Flash attention");
    try expectContains(src, "self.writeDescSet3(gds, self.attn_out_buf.handle");
}

test "decode loop keeps compute-to-transfer barrier before KV cache writes" {
    const src = @embedFile("compute/forward.zig");
    try expectContainsNear(src, "// KV cache write", "self.decode_cmd.computeToTransferBarrier();", 220);
}

test "decode loop keeps layer-boundary compute barrier after FFN residual" {
    const src = @embedFile("compute/forward.zig");
    try expectContainsNear(src, "The next layer immediately reads hidden_buf as its input.", "self.decode_cmd.computeBarrier();", 120);
}

test "prefill resets per-request state before processing prompt tokens" {
    const src = @embedFile("compute/forward.zig");
    try expectContainsNear(src, "pub fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "try self.resetRequestState();", 400);
}

test "softmax_topk shader keeps RADV-safe shared-memory winner scan" {
    const src = @embedFile("shaders/softmax_topk.comp");
    try expectContains(src, "shared float s_local_val[64];");
    try expectContains(src, "shared uint  s_local_idx[64];");
    try expectContains(src, "for (uint t = 0; t < 64; t++)");
    try expectContains(src, "s_logits[global_idx] = -1.0 / 0.0;");
    try expectNotContains(src, "GL_KHR_shader_subgroup_ballot");
    try expectNotContains(src, "subgroupBroadcast(");
}

test "softmax_topk shader renormalizes only selected logits" {
    const src = @embedFile("shaders/softmax_topk.comp");
    try expectContains(src, "shared float s_logits[256];");
    try expectContains(src, "float max_logit = -1.0 / 0.0;");
    try expectContains(src, "exp(uintBitsToFloat(output_data[k + i]) - max_logit)");
    try expectNotContains(src, "shared float s_probs[256];");
}

test "flash_attn shader keeps multi-subgroup fallback" {
    const src = @embedFile("shaders/flash_attn.comp");
    try expectContains(src, "subgroupMax");
    try expectContains(src, "subgroupAdd");
    try expectMultiSubgroupFallback(src, "s_reduce_scalar");
}

test "ssm_delta_net shader keeps multi-subgroup fallback" {
    const src = @embedFile("shaders/ssm_delta_net.comp");
    try expectContains(src, "subgroupAdd");
    try expectMultiSubgroupFallback(src, "s_reduce_scalar");
}

test "Q5_K shader keeps GGML contiguous half ordering" {
    const src = @embedFile("shaders/dmmv_q5k.comp");
    try expectContains(src, "low nibble");
    try expectContains(src, "high nibble");
    try expectContains(src, "x_grp + e]");
    try expectContains(src, "x_grp + 32u + e]");
    try expectNotContains(src, "2u * e");
}

test "Q5_K shader processes all 32 qs bytes per group_pair (not 16)" {
    // Regression: Q5_K DMMV previously used slice*4 and loop of 4, processing
    // only 16 of 32 qs bytes per group_pair. This silently dropped half the
    // dot-product terms, producing wrong results for Q5_K tensors (Qwen3.5 SSM).
    const src = @embedFile("shaders/dmmv_q5k.comp");
    try expectContains(src, "slice * 8u");
    try expectContains(src, "e_start + 8u");
    try expectNotContains(src, "slice * 4u");
    try expectNotContains(src, "e_start + 4u");
}

test "Q5_K MoE shader keeps GGML contiguous half ordering" {
    const src = @embedFile("shaders/dmmv_q5k_moe.comp");
    try expectContains(src, "x[x_grp + e]");
    try expectContains(src, "x[x_grp + 32u + e]");
    try expectNotContains(src, "2u * e");
}

test "IMROPE frequency uses global pair index, not per-section reset" {
    // Regression: IMROPE precomputation used per-section independent exponents,
    // resetting to 0 at each section boundary. For text IMROPE (all position IDs
    // equal), frequencies must use a single global progression: freq[k] = 1/base^(2k/rope_dim).
    // The per-section code caused pairs at section boundaries (11, 22) to get freq=1.0
    // instead of the correct monotonically decreasing values.
    const src = @embedFile("compute/forward.zig");
    // Must use total_pairs (global), not sec_pairs (per-section)
    try expectContains(src, "total_pairs = config.rope_sections[0] + config.rope_sections[1]");
    try expectContains(src, "for (0..total_pairs)");
    // Must NOT have per-section loop that resets exponents
    try expectNotContains(src, "for (0..sec_pairs)");
}

test "Metal Gemma embedding scaling applied before debug logging" {
    // Regression: Metal backend was missing sqrt(hidden_dim) embedding scaling
    // for Gemma models, causing ~62x smaller initial hidden states.
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "Gemma models scale embeddings by sqrt(hidden_dim).");
    try expectContains(src, "config.architecture == .gemma");
}

test "Metal FFN norm prefers ffn_norm over post_attention_norm" {
    // Regression: Metal used post_attention_norm.weight as FFN norm (wrong for Gemma
    // where both exist). Must prefer ffn_norm.weight, falling back to post_attention_norm.
    const src = @embedFile("compute/forward_metal.zig");
    // The ffn_norm_bufs init should try ffn_norm FIRST
    try expectContainsNear(src, "FFN norm: prefer ffn_norm.weight", "findLayerTensor(model, layer, \"ffn_norm.weight\")", 200);
}

test "Metal supports Gemma post-attention and post-FFN norms" {
    // Regression: Metal was missing post_attention_norm and post_ffw_norm dispatches,
    // which Gemma3 requires for correctness.
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "post_attn_norm_bufs");
    try expectContains(src, "post_ffw_norm_bufs");
    try expectContains(src, "post_ffw_norm.weight");
}

test "softmax_topk shader uses -inf for global_best init, not -1.0" {
    // Regression: softmax_topk used -1.0 as the initial value for the global
    // winner search. When router logits are all < -1.0, this silently selects
    // expert 0 instead of the actual best expert, corrupting MoE routing.
    const src = @embedFile("shaders/softmax_topk.comp");
    // global_best must be -inf
    try expectContains(src, "float global_best = -1.0 / 0.0;");
    // Must NOT use -1.0 as init
    try expectNotContains(src, "float global_best = -1.0;");
}

test "Metal loads GEGLU pipeline for Gemma activation" {
    // Regression: Metal used SwiGLU for all models, but Gemma requires GEGLU.
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "geglu_pipe");
    try expectContains(src, "cfg.architecture == .gemma");
}

test "router_logits_buf sized for max(n_experts, ssm_dt_rank)" {
    // Regression: router_logits_buf was sized for n_experts=1 (non-MoE) but SSM alpha
    // projection writes dt_rank floats. Buffer overflow corrupted alpha[1..15].
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "@max(if (config.n_experts > 0) config.n_experts else @as(u32, 1), config.ssm_dt_rank)");
}

test "GPU SSM path enabled when all three shaders are available" {
    // The GPU SSM path requires conv1d + delta-net + gated_norm shaders.
    // Must NOT gate on architecture enum (qwen35 vs qwen2_moe confusion).
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "pipeline_ssm_conv1d != null");
    try expectContains(src, "pipeline_ssm_delta_net != null");
    try expectContains(src, "pipeline_ssm_gated_norm != null");
    try expectNotContains(src, "config.architecture != .qwen35");
    try expectNotContains(src, "!has_delta_net");
}

test "Vulkan FFN norm prefers ffn_norm over post_attention_norm" {
    // Same fix as Metal — Vulkan must also prefer ffn_norm.weight first.
    const src = @embedFile("compute/forward.zig");
    try expectContainsNear(src, "ffn_norm_tensor", "findLayerTensor(layer, \"ffn_norm.weight\")", 120);
}

test "Vulkan Gemma embedding scaling matches Metal" {
    // Both backends must scale Gemma embeddings by sqrt(hidden_dim).
    const vulkan_src = @embedFile("compute/forward.zig");
    try expectContains(vulkan_src, "Gemma models scale embeddings by sqrt(hidden_dim).");
}

test "Vulkan post-attention norm applied before attn residual" {
    // Gemma3 requires RMS norm on o_proj output before residual add.
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "Gemma post-attention norm: RMS norm on o_proj output before residual add");
}

test "Vulkan post-FFN norm applied before FFN residual" {
    // Gemma3 requires RMS norm on down_proj output before residual add.
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "Gemma post-FFN norm: RMS norm on down_proj output before residual add");
}

test "rope_sections loaded from GGUF metadata" {
    // IMROPE requires rope.dimension_sections from GGUF for Qwen3.5 models.
    const src = @embedFile("model/loader.zig");
    try expectContains(src, "rope.dimension_sections");
}

test "RoPE shader supports freq buffer path for IMROPE" {
    // When freq_base_bits=0, the RoPE shader reads precomputed frequencies from
    // binding 2 instead of computing from freq_base. This supports IMROPE and
    // proportional RoPE (Gemma 4).
    const src = @embedFile("shaders/rope_fused.comp");
    try expectContains(src, "freq_base_bits == 0u");
    try expectContains(src, "inv_freq[i]");
}

test "F32 DMMV uses K-parallel reduction via subgroupAdd" {
    // Performance: F32 DMMV must use K-parallel (1 row per workgroup, 64 threads
    // collaborate via subgroupAdd) instead of M-parallel (64 rows per workgroup,
    // 1 thread per row). K-parallel gives M workgroups instead of ceil(M/64),
    // dramatically improving GPU utilization for small M (MoE router, M=256).
    const src = @embedFile("shaders/dmmv_f32.comp");
    try expectContains(src, "subgroupAdd");
    try expectContains(src, "shared float s_x[SPEC_K]");
    try expectContains(src, "row = gl_WorkGroupID.x");
    // Must NOT use gl_GlobalInvocationID (old M-parallel pattern)
    try expectNotContains(src, "gl_GlobalInvocationID");
}

test "Q4_K MoE shader uses packed uint32 reads, not byte access" {
    // Performance: Q4_K MoE DMMV must use uint32 packed reads (36 u32 per block)
    // instead of uint8_t byte access (144 individual reads). The packed path
    // gives 4x fewer memory transactions and enables vec4 dot products.
    const src = @embedFile("shaders/dmmv_q4k_moe.comp");
    // Must use uint buffer, not uint8_t
    try expectContains(src, "uint a_u32[]");
    try expectNotContains(src, "uint8_t a_data[]");
    // Must use vec4 dot products
    try expectContains(src, "unpack_nibbles_lo");
    try expectContains(src, "unpack_nibbles_hi");
    try expectContains(src, "dot(vec4(factor_lo)");
    // Must NOT have individual byte reads
    try expectNotContains(src, "a_data[");
}

test "Q5_K MoE shader processes all 32 elements per sub-block pair" {
    // The MoE Q5_K shader must iterate e from 0 to 31 (not 0..15 like the
    // old dense Q5_K bug). Each sub-block pair has 32 bytes of qs data.
    const src = @embedFile("shaders/dmmv_q5k_moe.comp");
    try expectContains(src, "for (uint e = 0; e < 32; e++)");
}

test "chat UI derives the model link from the reported model name" {
    const src = @embedFile("server/chat.html");
    try expectContains(src, "const chatStateKey='zinc.chat.state.v3';");
    try expectContains(src, "function restoreChatState()");
    try expectContains(src, "function clearConversation()");
    try expectContains(src, "id=\"cb\" class=\"btn btn-clear\"");
    try expectContains(src, "@media (max-width:720px)");
    try expectContains(src, "function modelHrefForName(name)");
    try expectContains(src, "function switchableModels()");
    try expectContains(src, "function activeModel()");
    try expectContains(src, "function scheduleHealthRefresh(delay)");
    try expectContains(src, "function refreshHealth()");
    try expectContains(src, "setModelTag(d.model)");
    try expectContains(src, "setGpuMemory(d);");
    try expectContains(src, "restoreChatState();");
    try expectContains(src, "CB.addEventListener('click',clearConversation);");
    try expectContains(src, "setCurrentModel(current);");
    try expectContains(src, "await Promise.allSettled([refreshHealth(),refreshModels()]);");
    try expectContains(src, "fetch(base+'/models/activate'");
    try expectContains(src, "m.managed&&m.installed&&m.supported_on_current_gpu&&m.fits_current_gpu");
    try expectNotContains(src, "setCurrentModel(selectedModel());");
    try expectNotContains(src, "href=\"https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF\"");
}
