// Vulkan Dispatch Overhead Benchmark
//
// Measures raw Vulkan dispatch cost on RDNA4:
// - Single dispatch (record + submit + wait)
// - N dispatches in one command buffer (GPU overhead)
// - Pre-recorded command buffer replay
//
// Expected results from RDNA4 profiling:
// - Single dispatch: ~33us
// - 1500 dispatches GPU time: ~24us (0.016us/dispatch)
// - Pre-recorded replay: ~54us for 1500 dispatches

const std = @import("std");

const log = std.log.scoped(.dispatch);

pub const DispatchResult = struct {
    name: []const u8,
    n_dispatches: u32,
    total_us: u64,
    per_dispatch_us: f64,
};

/// Expected performance targets from RDNA4 profiling.
pub const targets = struct {
    pub const single_dispatch_us: f64 = 50.0; // should be ~33us
    pub const batch_1500_per_dispatch_us: f64 = 0.1; // should be ~0.016us
    pub const replay_1500_us: u64 = 100; // should be ~54us
};

/// Print dispatch benchmark results.
pub fn printResults(results: []const DispatchResult) void {
    log.info("=== Vulkan Dispatch Overhead Benchmark ===", .{});
    for (results) |r| {
        log.info("{s}: {d} dispatches | {d}us total | {d:.3}us/dispatch", .{
            r.name, r.n_dispatches, r.total_us, r.per_dispatch_us,
        });
    }
}

test "DispatchResult" {
    const r = DispatchResult{
        .name = "test",
        .n_dispatches = 1500,
        .total_us = 24,
        .per_dispatch_us = 0.016,
    };
    try std.testing.expectEqual(@as(u32, 1500), r.n_dispatches);
}
