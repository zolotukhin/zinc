//! Host-only routing for accelerator CLI/debug prefill. Keep this policy shared
//! so the diagnostic harness exercises the same arithmetic as the production CLI.
const std = @import("std");

pub const Options = struct {
    rocm: bool,
    hybrid_dense: bool,
    mtp_prepared: bool = false,
    enabled: bool = true,
};

pub fn useBatched(tokens: usize, options: Options) bool {
    if (!options.enabled or tokens < 2) return false;
    // Dense Qwen projections below 32 tokens already use per-token matvecs.
    // Batching only the attention/SSM arithmetic can still change greedy output:
    // Qwen 3.6 27B's 19-token release prompt diverged at generated token 16.
    // Use serial prefill at these lengths, like the HTTP scheduler's threshold.
    // MTP needs batched hidden rows for priming; keep that path and MoE's routed
    // prefill unchanged. Long prompts must stay batched to meet the 4K timeout.
    return !options.rocm or !options.hybrid_dense or options.mtp_prepared or tokens >= 32;
}

test "short dense ROCm release prompts retain serial arithmetic" {
    const options: Options = .{ .rocm = true, .hybrid_dense = true };
    for ([_]usize{ 0, 1, 2, 18, 19, 22, 31 }) |tokens| {
        try std.testing.expect(!useBatched(tokens, options));
    }
    // Guard the boundary and the required long-context workload. Globally
    // disabling batching fixed the continuation but timed out at 4K.
    for ([_]usize{ 32, 33, 4032, 4096 }) |tokens| {
        try std.testing.expect(useBatched(tokens, options));
    }
}

test "MTP retains batched hidden rows and explicit opt-out wins" {
    const options: Options = .{ .rocm = true, .hybrid_dense = true, .mtp_prepared = true };
    for ([_]usize{ 2, 19, 31, 32, 4032 }) |tokens| {
        try std.testing.expect(useBatched(tokens, options));
        var disabled = options;
        disabled.enabled = false;
        try std.testing.expect(!useBatched(tokens, disabled));
    }
    try std.testing.expect(!useBatched(1, options));
}

test "MoE, Gemma and CUDA keep their existing prefill routing" {
    const unchanged = [_]Options{
        .{ .rocm = true, .hybrid_dense = false },
        .{ .rocm = false, .hybrid_dense = true },
        .{ .rocm = false, .hybrid_dense = false },
    };
    for (unchanged) |options| {
        try std.testing.expect(!useBatched(0, options));
        try std.testing.expect(!useBatched(1, options));
        try std.testing.expect(useBatched(2, options));
        try std.testing.expect(useBatched(31, options));
        try std.testing.expect(useBatched(32, options));
    }
}
