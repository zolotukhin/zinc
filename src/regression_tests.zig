//! Source-level regression guards for bugs that are hard to cover with unit-only GPU tests.
const std = @import("std");

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    try std.testing.expect(std.mem.indexOf(u8, haystack, needle) != null);
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

test "decode loop keeps compute-to-transfer barrier before Q and gate copies" {
    const src = @embedFile("compute/forward.zig");
    try expectContainsNear(src, "The next step is a transfer copy", "self.decode_cmd.computeToTransferBarrier();", 160);
}

test "decode loop keeps compute-to-transfer barrier before KV cache writes" {
    const src = @embedFile("compute/forward.zig");
    try expectContainsNear(src, "// KV cache write", "self.decode_cmd.computeToTransferBarrier();", 160);
}

test "decode loop keeps layer-boundary compute barrier after FFN residual" {
    const src = @embedFile("compute/forward.zig");
    try expectContainsNear(src, "The next layer immediately reads hidden_buf as its input.", "self.decode_cmd.computeBarrier();", 120);
}

test "prefill resets per-request state before processing prompt tokens" {
    const src = @embedFile("compute/forward.zig");
    try expectContainsNear(src, "pub fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {", "try self.resetRequestState();", 400);
}

test "softmax_topk shader keeps multi-subgroup fallback" {
    const src = @embedFile("shaders/softmax_topk.comp");
    try expectContains(src, "subgroupMax");
    try expectContains(src, "subgroupAdd");
    try expectMultiSubgroupFallback(src, "s_reduce_val");
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
