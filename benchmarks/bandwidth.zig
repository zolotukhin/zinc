// DMMV Bandwidth Utilization Benchmark
//
// Measures effective memory bandwidth for each DMMV quant type at various
// matrix sizes. Run on RDNA4 hardware to validate 90%+ utilization on
// large matmuls.
//
// Usage: Compiled and run on the remote test node as part of `zig build bench`
// See build.zig for benchmark target configuration.

const std = @import("std");

const log = std.log.scoped(.bandwidth);

/// Benchmark configuration for a single matrix size.
pub const BenchConfig = struct {
    name: []const u8,
    M: u32,
    K: u32,
    quant: []const u8, // "q4_k", "q8_0", "f16"
    iterations: u32,
};

/// Result of a bandwidth benchmark run.
pub const BenchResult = struct {
    name: []const u8,
    M: u32,
    K: u32,
    quant_name: []const u8,
    bytes_read: u64,
    time_us: u64,
    bandwidth_gbps: f64,
    utilization_pct: f64,
};

/// Standard benchmark configurations matching RDNA4 profiling data.
pub const standard_configs = [_]BenchConfig{
    .{ .name = "vocab_output", .M = 248320, .K = 2048, .quant = "q4_k", .iterations = 10 },
    .{ .name = "large_attn", .M = 8192, .K = 2048, .quant = "q4_k", .iterations = 20 },
    .{ .name = "medium_attn", .M = 4096, .K = 2048, .quant = "q4_k", .iterations = 30 },
    .{ .name = "moe_expert", .M = 512, .K = 2048, .quant = "q4_k", .iterations = 50 },
    .{ .name = "small_matmul", .M = 32, .K = 2048, .quant = "q4_k", .iterations = 100 },
    .{ .name = "large_f16", .M = 8192, .K = 2048, .quant = "f16", .iterations = 20 },
    .{ .name = "q8_0_large", .M = 8192, .K = 2048, .quant = "q8_0", .iterations = 20 },
};

/// Target bandwidth utilization thresholds.
pub const targets = struct {
    pub const vocab_output_pct: f64 = 90.0; // 93.2% measured in llama.cpp
    pub const large_attn_pct: f64 = 80.0; // 83.6% measured
    pub const medium_attn_pct: f64 = 60.0; // 66.1% measured
};

/// Print benchmark results in a formatted table.
pub fn printResults(results: []const BenchResult) void {
    log.info("=== DMMV Bandwidth Benchmark ===", .{});
    for (results) |r| {
        const status: []const u8 = if (r.utilization_pct >= 80.0) "PASS" else if (r.utilization_pct >= 60.0) "OK" else "LOW";
        log.info("{s}: M={d} K={d} {s} | {d}us | {d:.1} GB/s | {d:.1}% [{s}]", .{
            r.name, r.M, r.K, r.quant_name, r.time_us,
            r.bandwidth_gbps, r.utilization_pct, status,
        });
    }
}

test "BenchConfig sizes" {
    try std.testing.expectEqual(@as(usize, 7), standard_configs.len);
}
