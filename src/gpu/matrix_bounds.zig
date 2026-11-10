//! Validate Vulkan Q8 GEMM byte ranges before recording GPU work.
//! @section Inference Runtime
//!
//! Check matrix extents and descriptor sizes with overflow-safe host arithmetic.
//! These checks run before GPU access and preserve the existing shader ABI.
const std = @import("std");

/// The Vulkan Q8 GEMM shaders use 32-bit byte offsets, and their descriptors bind
/// the complete weight buffer. Reject unsupported ranges before either an
/// offset wraps in the shader or the descriptor exceeds the 32-bit range.
/// Alignment is checked separately by the Q8 dispatch caller.
/// Current forward call sites pass a tensor's gpu_buffer.size with zero weight
/// offset; loader.zig allocates that buffer using tensor_info.sizeBytes(). The
/// model file's total size is never used as this descriptor range.
pub fn q8MatrixFits(rows: u32, columns: u32, offset: u32, buffer_bytes: u64) bool {
    if (rows == 0 or columns == 0 or (columns & 31) != 0) return false;
    if (buffer_bytes > std.math.maxInt(u32) or offset > buffer_bytes) return false;
    const row_bytes = @as(u64, columns / 32) * 34;
    // Divide the remaining range instead of multiplying rows * row_bytes:
    // even that intermediate can overflow u64 for arbitrary u32 dimensions.
    return @as(u64, rows) <= (buffer_bytes - offset) / row_bytes;
}

test "Q8 matrix range includes nonzero offsets and the exact last block" {
    try std.testing.expect(q8MatrixFits(3, 32, 2, 104));
    try std.testing.expect(!q8MatrixFits(3, 32, 2, 103));
    try std.testing.expect(q8MatrixFits(2, 64, 0, 136));
    try std.testing.expect(!q8MatrixFits(2, 64, 0, 135));
    try std.testing.expect(!q8MatrixFits(1, 32, 36, 34));
}

test "Q8 matrix range rejects the first overflowing block and oversized descriptors" {
    const limit: u64 = std.math.maxInt(u32);
    const rows: u32 = @intCast(limit / 34);
    try std.testing.expect(q8MatrixFits(rows, 32, 0, limit));
    try std.testing.expect(q8MatrixFits(rows, 32, 16, limit));
    try std.testing.expect(!q8MatrixFits(rows, 32, 18, limit));
    try std.testing.expect(!q8MatrixFits(rows + 1, 32, 0, limit));
    try std.testing.expect(!q8MatrixFits(1, 32, 0, limit + 1));
    try std.testing.expect(!q8MatrixFits(1, std.math.maxInt(u32) - 31, 0, limit));
}

test "Q8 matrix range handles malformed and maximum dimensions without overflow" {
    for ([_][2]u32{ .{ 0, 32 }, .{ 1, 0 }, .{ 1, 31 }, .{ 1, 33 }, .{ std.math.maxInt(u32), std.math.maxInt(u32) - 31 } }) |shape| {
        try std.testing.expect(!q8MatrixFits(shape[0], shape[1], 0, std.math.maxInt(u32)));
    }
    try std.testing.expect(!q8MatrixFits(1, 32, std.math.maxInt(u32), std.math.maxInt(u32)));
}

test "Q8 division guard matches an independent wide-integer range calculation" {
    const limit: u64 = std.math.maxInt(u32);
    const rows_at_limit: u32 = @intCast(limit / 34);
    const rows = [_]u32{ 0, 1, 2, 3, rows_at_limit - 1, rows_at_limit, rows_at_limit + 1, std.math.maxInt(u32) };
    const columns = [_]u32{ 0, 1, 31, 32, 64, 96, 1024, std.math.maxInt(u32) - 31, std.math.maxInt(u32) };
    const offsets = [_]u32{ 0, 1, 2, 16, 18, std.math.maxInt(u32) - 1, std.math.maxInt(u32) };
    const sizes = [_]u64{ 0, 33, 34, 35, 104, limit, limit + 1, std.math.maxInt(u64) };
    for (rows) |m| for (columns) |k| for (offsets) |offset| for (sizes) |size| {
        const end = @as(u128, offset) + @as(u128, m) * (k / 32) * 34;
        const expected = m > 0 and k > 0 and (k & 31) == 0 and size <= limit and end <= size;
        try std.testing.expectEqual(expected, q8MatrixFits(m, k, offset, size));
    };
}
