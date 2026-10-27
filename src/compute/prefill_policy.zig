//! Host-only routing for accelerator CLI, debug and slot prefill. Share the
//! decisions so prompt batching and SSM reduction selection cannot drift apart.
const std = @import("std");

pub const batch_threshold: usize = 32;

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
    return !options.rocm or !options.hybrid_dense or options.mtp_prepared or tokens >= batch_threshold;
}

pub const SlotOptions = struct {
    rocm: bool,
    moe: bool,
    slots: u32,
    kv_f32: bool,
    b1_matvec: bool,
};

/// The MoE CLI batches short prompts with the warp SSM scan. Serial HTTP
/// prefill instead chose "box" where the CLI/reference chose "fridge". Merely
/// enabling slot batching still selects the block scan, so both the scheduler
/// and slot scan must use this decision. Restrict it to the F32 single-slot B=1
/// path; concurrent serving needs its existing batched-decode reduction.
pub fn useCliMoeSlotPrefill(tokens: usize, options: SlotOptions) bool {
    return options.rocm and options.moe and options.slots == 1 and
        options.kv_f32 and options.b1_matvec and tokens > 1 and tokens < batch_threshold;
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

test "short MoE slot prefill preserves CLI arithmetic at the batching boundary" {
    const options: SlotOptions = .{ .rocm = true, .moe = true, .slots = 1, .kv_f32 = true, .b1_matvec = true };
    for ([_]usize{ 2, 18, 19, 22, 31 }) |tokens| {
        try std.testing.expect(useCliMoeSlotPrefill(tokens, options));
    }
    for ([_]usize{ 0, 1, 32, 33, 4032, 4096 }) |tokens| {
        try std.testing.expect(!useCliMoeSlotPrefill(tokens, options));
    }
}

test "MoE slot exception leaves other backends, KV formats and concurrent decode alone" {
    const options: SlotOptions = .{ .rocm = true, .moe = true, .slots = 1, .kv_f32 = true, .b1_matvec = true };
    inline for (.{ "rocm", "moe", "kv_f32", "b1_matvec" }) |field| {
        var disabled = options;
        @field(disabled, field) = false;
        try std.testing.expect(!useCliMoeSlotPrefill(19, disabled));
    }
    for ([_]u32{ 0, 2, 4 }) |slots| {
        var concurrent = options;
        concurrent.slots = slots;
        try std.testing.expect(!useCliMoeSlotPrefill(19, concurrent));
    }
}
