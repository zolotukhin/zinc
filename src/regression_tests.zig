//! Source-level regression guards for bugs that are hard to cover with unit-only GPU tests.
const builtin = @import("builtin");
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

test "decode loop keeps deinterleave split for packed Q and gate" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "Qwen3Next packs per-head [Q(head_dim), gate(head_dim)] blocks.");
    try expectContains(src, "Deinterleave Q+gate using compute shader");
    try expectContains(src, "pipeline_deinterleave");
}

test "decode loop applies packed attention gate after flash attention" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "// Flash attention");
    try expectContains(src, "self.writeDescSet3(gds, self.attn_out_buf.handle");
}

test "decode loop keeps compute-to-transfer barrier before KV cache writes" {
    const src = @embedFile("compute/forward.zig");
    const marker = std.mem.indexOf(u8, src, "Transfer fallback: Q RoPE before barrier (original order preserved)") orelse return error.TestExpectedEqual;
    const fallback_src = src[marker..@min(marker + 1600, src.len)];
    try expectContains(fallback_src, "self.decode_cmd.computeAndTransferBarrier();");
    try expectContains(fallback_src, "vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.k_buf.handle, self.kv_k_cache[layer_idx].handle");
    try expectContains(fallback_src, "self.decode_cmd.transferToComputeBarrier();");
}

test "decode loop keeps layer-boundary compute barrier after FFN residual" {
    const src = @embedFile("compute/forward.zig");
    try expectContainsNear(src, "// FFN residual: hidden_buf += down_buf", "self.decode_cmd.computeBarrier();", 1200);
}

test "prefill resets per-request state before processing prompt tokens" {
    const src = @embedFile("compute/forward.zig");
    try expectContainsNear(src, "pub fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "try self.resetRequestState(target_context_tokens);", 900);
}

test "Metal prefill preserves cached prefixes instead of resetting unconditionally" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContainsNear(src, "pub fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "if (state.position == 0 and state.generated_tokens.items.len == 0)", 900);
    try expectContainsNear(src, "pub fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "return error.KvStateNotAvailable;", 1400);
}

test "Metal prefillBatched gates on env flag and supported architecture" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "ZINC_BATCHED_PREFILL");
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "batchedPrefillMode()", 600);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "canUseBatchedPrefill(self)", 600);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "return self.prefillBatch(state, prompt_tokens);", 1200);
}

test "Metal prefillBatched validate path diffs last-token logits within 1e-3" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "if (mode == .validate)", 12000);
    try expectContainsNear(src, "if (mode == .validate)", "const tol: f32 = 1e-3;", 1500);
    try expectContainsNear(src, "if (mode == .validate)", "try self.prefillBatch(state, prompt_tokens);", 1500);
}

test "Metal prefillBatched uses gemm/rope batched dispatch helpers" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "dispatchGemmBatchedOnCmd", 12000);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "dispatchRopeBatchedOnCmd", 12000);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "dispatchFlashAttnBatchedOnCmd", 12000);
}

test "Metal prefillBatched routes Q8 KV cache through flash_attn_batched_q8" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "if (self.kv_cache_q8)", 12000);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "dispatchFlashAttnBatchedQ8OnCmd", 12000);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "dispatchKvCacheWriteBatchedQ8OnCmd", 12000);
}

test "Metal prefillBatched supports prefix reuse by extending KV at state.position" {
    const src = @embedFile("compute/forward_metal.zig");
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "const position_base: u32 = state.position;", 2000);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "return error.KvStateNotAvailable;", 2000);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "const kv_len = position_base + n_tokens;", 12000);
    try expectContainsNear(src, "pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "self.position = position_base + n_tokens;", 12000);
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

test "Q5_0 and Q5_1 DMMV launch with 2 rows per workgroup" {
    const src = @embedFile("compute/dmmv.zig");
    try expectContains(src, ".q5_0, .q5_1, .mxfp4, .q8_0, .f16 => (M + 1) / 2");
    try expectContains(src, ".q4_k, .q5_0, .q5_1, .q6_k => (M + 1) / 2");
}

test "GPT-OSS routing keeps SOFTMAX_WEIGHT expert selection path" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "GPT-OSS uses this SOFTMAX_WEIGHT routing rule instead of softmax-over-all-experts.");
    try expectContains(src, "if (config.architecture == .gpt_oss) {\n                        topKSoftmaxWeight(router_logits, n_used, expert_ids[0..n_used], expert_weights[0..n_used]);\n                    } else {");
}

test "GPT-OSS FFN keeps OAI SwiGLU and bias-add dispatches" {
    const forward_src = @embedFile("compute/forward.zig");
    const elementwise_src = @embedFile("compute/elementwise.zig");
    try expectContainsNear(forward_src, "if (self.model.config.architecture == .gpt_oss) {", "return self.dispatchSwigluOai(gate_buf, gate_size, up_buf, up_size, output_buf, output_size, n_elements);", 200);
    try expectContains(forward_src, "try self.dispatchBiasAddSlice(self.gate_buf.handle, self.gate_buf.size, bias, eid * inter_dim, inter_dim);");
    try expectContains(forward_src, "try self.dispatchBiasAddSlice(self.down_buf.handle, hidden_size, bias, eid * hidden_dim, hidden_dim);");
    try expectContains(elementwise_src, "pub fn recordSwigluOai(");
    try expectContains(elementwise_src, "pub fn recordBiasAdd(");
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
    // which Gemma requires for correctness.
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
    try expectContainsNear(src, "FFN norm: prefer ffn_norm.weight", "const ffn_norm_tensor = lt.ffn_norm orelse", 500);
    try expectContains(src, "lt.post_attention_norm orelse return error.TensorNotFound;");
}

test "Vulkan Gemma embedding scaling matches Metal" {
    // Both backends must scale Gemma embeddings by sqrt(hidden_dim).
    const vulkan_src = @embedFile("compute/forward.zig");
    try expectContains(vulkan_src, "Gemma models scale embeddings by sqrt(hidden_dim).");
}

test "Vulkan post-attention norm applied before attn residual" {
    // Gemma requires RMS norm on o_proj output before residual add.
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "Gemma post-attention norm: RMS norm on o_proj output before residual add");
}

test "Vulkan post-FFN norm applied before FFN residual" {
    // Gemma requires RMS norm on down_proj output before residual add.
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

test "YaRN RoPE attention scale stays wired through Vulkan RoPE dispatch" {
    const forward_src = @embedFile("compute/forward.zig");
    const rope_src = @embedFile("shaders/rope_fused.comp");
    const norm_rope_src = @embedFile("shaders/norm_rope.comp");
    try expectContains(forward_src, "const rope_attn_scale = if (use_yarn_rope) effectiveRopeAttnScale(config) else 1.0;");
    try expectContains(forward_src, "const push = RopePush{\n                .stride = stride,\n                .rope_dim = rope_dim,\n                .n_heads = n_heads,\n                .position = position,\n                .freq_base_bits = @bitCast(freq_base),\n                .attn_scale_bits = @bitCast(attn_scale),");
    try expectContains(forward_src, "const push = NormRopePush{\n            .head_dim = head_dim,\n            .rope_dim = rope_dim,\n            .n_heads = n_heads,\n            .position = position,\n            .freq_base_bits = @bitCast(freq_base),\n            .attn_scale_bits = @bitCast(attn_scale),");
    try expectContains(rope_src, "float attn_scale = attn_scale_bits != 0u ? uintBitsToFloat(attn_scale_bits) : 1.0;");
    try expectContains(rope_src, "float cos_t = cos(theta) * attn_scale;");
    try expectContains(norm_rope_src, "float attn_scale = attn_scale_bits != 0u ? uintBitsToFloat(attn_scale_bits) : 1.0;");
    try expectContains(norm_rope_src, "float cos_t = cos(theta) * attn_scale;");
}

test "flash attention sink buffer stays in final normalization" {
    const src = @embedFile("shaders/flash_attn.comp");
    try expectContains(src, "layout(set = 0, binding = 5) readonly  buffer Sinks");
    try expectContains(src, "float sink_val = sink_data[head];");
    try expectContains(src, "final_sum = s_sum_old * rescale + exp(sink_val - sink_max);");
    try expectContains(src, "o_data[o_base + d] = s_out[d] * rescale * inv_sum;");
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

test "Metal flash_attn supports head_dim=512 for Gemma 4 global attention layers" {
    // Regression: FLASH_MAX_HEAD_DIM was 256. Gemma 4 has mixed attention where
    // SWA layers use head_dim=256 but global layers (every 6th) use head_dim=512.
    // The threadgroup arrays were too small and per-thread loops used `if (tid < vec4_dim)`
    // instead of strided loops, leaving the second half of Q/acc uninitialized → NaN.
    const src = @embedFile("shaders/metal/flash_attn.metal");
    try expectContains(src, "FLASH_MAX_HEAD_DIM = 512");
    // Must use strided loops, not single-pass `if (tid < vec4_dim)`
    try expectContains(src, "for (uint i = tid; i < vec4_dim; i += FLASH_TG_SIZE)");
    try expectContains(src, "for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE)");
    try expectNotContains(src, "if (tid < vec4_dim)");
}

test "Metal flash_attn_q8 supports head_dim=512" {
    const src = @embedFile("shaders/metal/flash_attn_q8.metal");
    try expectContains(src, "FLASH_MAX_HEAD_DIM = 512");
    try expectContains(src, "for (uint i = tid; i < vec4_dim; i += FLASH_TG_SIZE)");
    try expectContains(src, "for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE)");
    try expectNotContains(src, "if (tid < vec4_dim)");
}

test "Metal flash_attn_batched supports head_dim=512" {
    const src = @embedFile("shaders/metal/flash_attn_batched.metal");
    try expectContains(src, "FLASH_MAX_HEAD_DIM = 512");
    try expectContains(src, "for (uint i = tid; i < vec4_dim; i += FLASH_TG_SIZE)");
    try expectContains(src, "for (uint vi = tid; vi < vec4_dim; vi += FLASH_TG_SIZE)");
    try expectNotContains(src, "if (tid < vec4_dim)");
}

test "Metal Q8_0 DMMV uses float dot products, not half (overflow at large norm values)" {
    // Regression: Q8_0 shader converted float input to half4 for dot products.
    // Gemma 4 attn_norm weights up to ~300 produce norm_buf values up to ~3000;
    // int8(127) × half(3000) = 381,000 overflows f16 max (65504) → -inf.
    // Fix: use float4 dot products in all Q8_0 DMMV variants.
    //
    // Also: quants must be read via packed_char4 (not int* cast) because Q8_0
    // quants start at byte offset 2 within 34-byte blocks — misaligned for int*.
    const src = @embedFile("shaders/metal/dmmv_q8_0.metal");
    try expectContains(src, "packed_char4");
    try expectContains(src, "dot(float4(");
    // Must NOT convert input to half
    try expectNotContains(src, "half4 x = half4(");
    try expectNotContains(src, "half4 q_half");
    // Must NOT use misaligned int* cast for quant reads
    try expectNotContains(src, "device const int*)(blk");
}

test "Metal Q8_0 k2048 DMMV uses float dot products" {
    const src = @embedFile("shaders/metal/dmmv_q8_0_k2048.metal");
    try expectContains(src, "packed_char4");
    try expectContains(src, "dot(float4(");
    try expectNotContains(src, "half4 x = half4(");
    try expectNotContains(src, "half4 q_half");
    try expectNotContains(src, "device const int*)(blk");
}

test "Metal Q8_0 dual DMMV uses float dot products" {
    const src = @embedFile("shaders/metal/dmmv_q8_0_dual.metal");
    try expectContains(src, "float4 q_f = float4(q)");
    try expectContains(src, "dot(q_f, x_f)");
    try expectNotContains(src, "half4 x = half4(");
    try expectNotContains(src, "half4 q_half");
}

test "Metal Q5_1 DMMV shader exists and uses factored d*sum(q*x)+m*sum(x)" {
    // Q5_1 expert down projections in Gemma 4 26B-A4B MoE were falling back to CPU.
    const src = @embedFile("shaders/metal/dmmv_q5_1.metal");
    // Q5_1 block: 24 bytes (d=f16 + m=f16 + qh=u32 + qs=16 bytes)
    try expectContains(src, "bpb = 24");
    // Factored dot product: d * sum(q*x) + m * sum(x)
    try expectContains(src, "d * sum_qx + m * sum_x");
    // Must read min value from bytes 2-3
    try expectContains(src, "half*)(block + 2)");
    // Must read qh from bytes 4-7
    try expectContains(src, "block[4]");
}

test "Vulkan Q5_1 DMMV shader exists and uses factored dot product" {
    const src = @embedFile("shaders/dmmv_q5_1.comp");
    try expectContains(src, "Q5_1_BYTES      = 24");
    try expectContains(src, "d * sum_qx + m * sum_x");
}

test "Vulkan flash_attn supports head_dim up to 512" {
    // Gemma 4 global attention layers use head_dim=512.
    // The Vulkan shader uses shared memory sized for 512 and strided loops.
    const src = @embedFile("shaders/flash_attn.comp");
    try expectContains(src, "s_out[512]");
    // Uses strided loop (tid increments by 64) that naturally handles any head_dim
    try expectContains(src, "for (uint d = tid; d < head_dim; d += 64u)");
}

test "Metal forward derives per-layer head_dim from attn_q_norm tensor" {
    // Regression: Gemma 4 has mixed head_dim per layer (256 for SWA, 512 for global).
    // The Metal forward must derive head_dim from attn_q_norm or attn_k_norm tensors,
    // not use the global config.head_dim for all layers.
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "if (lt.attn_q_norm) |qn|");
    try expectContainsNear(src, "if (lt.attn_q_norm) |qn|", "head_dim = @intCast(qn.info.numElements())", 200);
}

test "Metal forward handles use_k_as_v for Gemma global attention layers" {
    // Regression: Gemma 4 global attention layers have no attn_v tensor — they share K as V.
    const src = @embedFile("compute/forward_metal.zig");
    try expectContains(src, "use_k_as_v = lt.attn_v == null and cfg.architecture == .gemma");
}

test "Vulkan forward handles use_k_as_v for Gemma global attention layers" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "use_k_as_v = lt.attn_v == null and config.architecture == .gemma");
}

test "Vulkan forward derives per-layer head_dim from attn_q_norm tensor" {
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "layer_head_dim");
    try expectContains(src, "layer_kv_dim");
    try expectContains(src, "layer_n_kv_heads");
}

test "Vulkan forward handles fused ffn_gate_up_exps for Gemma 4 MoE" {
    // Gemma 4 26B-A4B uses fused ffn_gate_up_exps instead of separate gate/up tensors.
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "fused_gate_up = lt.ffn_gate_up_exps");
    try expectContains(src, "up_base_offset");
}

test "Vulkan forward uses GEGLU activation for Gemma architecture" {
    // Gemma models use GEGLU, not SwiGLU. The dispatchFfnActivation helper
    // selects the right shader based on architecture.
    const src = @embedFile("compute/forward.zig");
    try expectContains(src, "fn dispatchFfnActivation(");
    try expectContains(src, "fn dispatchGeglu(");
    try expectContains(src, "self.model.config.architecture == .gemma");
    // All MoE/FFN activation calls must use dispatchFfnActivation, not dispatchSwiglu directly
    try expectNotContains(src, "try self.dispatchSwiglu(");
}

test "Gemma 4 12B MoE catalog entry has correct download URL with UD prefix" {
    const src = @embedFile("model/catalog.zig");
    // The Unsloth Dynamic quantization uses UD- prefix in filenames
    try expectContains(src, "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf");
    // Must NOT have the old incorrect filename without UD-
    try expectNotContains(src, "gemma-4-26B-A4B-it-Q4_K_M.gguf");
}

test "Q5_0 shader reads qh via byte assembly, not unaligned uint32 cast" {
    // Regression guard: the Q5_0 block stores qh at byte offset 2 within a 22-byte block.
    // Reading via *((device const uint*)&block[2]) silently returns wrong values on Apple
    // Silicon for non-4-byte-aligned addresses. The fix reads bytes individually.
    const src = @embedFile("shaders/metal/dmmv_q5_0.metal");
    // Must NOT contain the broken unaligned cast pattern
    try expectNotContains(src, "uint*)&block[2]");
    try expectNotContains(src, "uint*)(block + 2)");
    // Must contain the safe byte-by-byte assembly
    try expectContains(src, "uint(block[2])");
    try expectContains(src, "uint(block[3])");
    try expectContains(src, "uint(block[4])");
    try expectContains(src, "uint(block[5])");
}

test "Q5_0 dequantRow matches expected values for known block" {
    if (builtin.os.tag != .macos) return error.SkipZigTest;

    const forward_metal = @import("compute/forward_metal.zig");
    // Build a Q5_0 block: d=0.5, qh=0x0000FFFF (bits 0-15 set), qs all 0x53 (lo=3, hi=5)
    // Element j (0-15): lo=3, bit_lo=1 → quant=3|(1<<4)=19 → value=0.5*(19-16)=1.5
    // Element 16+j:     hi=5, bit_hi=0 → quant=5|(0<<4)=5  → value=0.5*(5-16)=-5.5
    var block: [22]u8 = undefined;
    const d_bits: u16 = @bitCast(@as(f16, 0.5));
    block[0] = @truncate(d_bits);
    block[1] = @truncate(d_bits >> 8);
    block[2] = 0xFF;
    block[3] = 0xFF;
    block[4] = 0x00;
    block[5] = 0x00; // qh = 0x0000FFFF
    @memset(block[6..22], 0x53); // lo=3, hi=5
    var output: [32]f32 = undefined;
    forward_metal.dequantRow(&block, 0, 32, .q5_0, &output);
    for (0..16) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, 1.5), output[j], 0.001);
    }
    for (16..32) |j| {
        try std.testing.expectApproxEqAbs(@as(f32, -5.5), output[j], 0.001);
    }
}
