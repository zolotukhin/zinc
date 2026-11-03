//! Bound temporary model-upload memory without changing tensor bytes.
//! @section Inference Runtime
//!
//! Plan exact byte windows for reusable staging, including large offsets and tails.
//! The host-only contracts run on every backend without opening a GPU.
const std = @import("std");

pub const staging_limit_bytes: u64 = 64 * 1024 * 1024;

pub fn stagingCapacity(largest_tensor_bytes: u64) u64 {
    return @min(largest_tensor_bytes, staging_limit_bytes);
}

pub const Chunk = struct { offset: u64, size: u64 };

/// Upload each byte once, including a final partial staging window.
pub const Chunks = struct {
    total: u64,
    capacity: u64,
    offset: u64 = 0,

    pub fn init(total: u64, capacity: u64) !Chunks {
        if (total != 0 and capacity == 0) return error.ZeroUploadCapacity;
        return .{ .total = total, .capacity = capacity };
    }

    pub fn next(self: *Chunks) ?Chunk {
        if (self.offset == self.total) return null;
        const size = @min(self.total - self.offset, self.capacity);
        const chunk: Chunk = .{ .offset = self.offset, .size = size };
        self.offset += size;
        return chunk;
    }
};

test "upload staging is capped and small tensors do not reserve the full cap" {
    try std.testing.expectEqual(@as(u64, 0), stagingCapacity(0));
    try std.testing.expectEqual(@as(u64, 34), stagingCapacity(34));
    try std.testing.expectEqual(staging_limit_bytes, stagingCapacity(std.math.maxInt(u64)));
}

test "upload windows preserve every byte and surrounding guards" {
    var input: [129]u8 = undefined;
    for (&input, 0..) |*byte, i| byte.* = @truncate(i * 17);
    var output: [137]u8 = @splat(0xA5);
    var chunks = try Chunks.init(input.len, 16);
    var count: usize = 0;
    while (chunks.next()) |chunk| {
        const start: usize = @intCast(chunk.offset);
        const size: usize = @intCast(chunk.size);
        @memcpy(output[3 + start ..][0..size], input[start..][0..size]);
        count += 1;
    }
    try std.testing.expectEqual(@as(usize, 9), count);
    try std.testing.expectEqualSlices(u8, &input, output[3..132]);
    for (output[0..3]) |byte| try std.testing.expectEqual(@as(u8, 0xA5), byte);
    for (output[132..]) |byte| try std.testing.expectEqual(@as(u8, 0xA5), byte);
}

test "upload windows handle empty tensors, exact windows and large tails" {
    var empty = try Chunks.init(0, 0);
    try std.testing.expectEqual(null, empty.next());
    try std.testing.expectError(error.ZeroUploadCapacity, Chunks.init(1, 0));
    var chunks = try Chunks.init(staging_limit_bytes * 2 + 34, staging_limit_bytes);
    try std.testing.expectEqual(Chunk{ .offset = 0, .size = staging_limit_bytes }, chunks.next().?);
    try std.testing.expectEqual(Chunk{ .offset = staging_limit_bytes, .size = staging_limit_bytes }, chunks.next().?);
    try std.testing.expectEqual(Chunk{ .offset = staging_limit_bytes * 2, .size = 34 }, chunks.next().?);
    try std.testing.expectEqual(null, chunks.next());
    var exact = try Chunks.init(64, 64);
    try std.testing.expectEqual(Chunk{ .offset = 0, .size = 64 }, exact.next().?);
    try std.testing.expectEqual(null, exact.next());
}

test "upload byte bounds do not overflow near the address-space limit" {
    const maximum = std.math.maxInt(u64);
    var chunks = try Chunks.init(maximum, maximum - 7);
    try std.testing.expectEqual(Chunk{ .offset = 0, .size = maximum - 7 }, chunks.next().?);
    try std.testing.expectEqual(Chunk{ .offset = maximum - 7, .size = 7 }, chunks.next().?);
    try std.testing.expectEqual(null, chunks.next());
}
