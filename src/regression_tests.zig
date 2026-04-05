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

test "Q5_K MoE shader keeps GGML contiguous half ordering" {
    const src = @embedFile("shaders/dmmv_q5k_moe.comp");
    try expectContains(src, "x[x_grp + e]");
    try expectContains(src, "x[x_grp + 32u + e]");
    try expectNotContains(src, "2u * e");
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
