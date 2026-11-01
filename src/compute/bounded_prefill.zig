//! Retry scratch allocation before prefill mutates KV state. A failed whole-
//! prompt allocation must not turn an otherwise usable GPU into serial prefill.
const std = @import("std");

/// Share Qwen's scratch limit knob without accepting a zero-sized chunk or
/// letting a malformed setting replace the original prompt capacity.
pub fn limit(tokens: u32, setting: ?[]const u8) u32 {
    const text = setting orelse return tokens;
    const parsed = std.fmt.parseInt(u32, text, 10) catch return tokens;
    return if (parsed == 0) tokens else @min(tokens, parsed);
}

/// The allocator must release partial allocations on failure. Only device OOM
/// is retryable; execution, host allocation and shape errors keep their cause.
pub fn fit(context: anytype, tokens: u32, comptime allocate: anytype) !u32 {
    if (tokens == 0) return error.EmptyPrompt;
    var capacity = tokens;
    while (true) {
        _ = allocate(context, capacity) catch |err| {
            if (err != error.CudaBufferAllocFailed or capacity <= 32) return err;
            capacity = @max(32, capacity / 2);
            continue;
        };
        return capacity;
    }
}

const AllocationProbe = struct {
    limit: u32,
    failure: ?anyerror = null,
    attempts: [16]u32 = @splat(0),
    count: usize = 0,

    fn allocate(self: *AllocationProbe, capacity: u32) !void {
        self.attempts[self.count] = capacity;
        self.count += 1;
        if (self.failure) |err| return err;
        if (capacity > self.limit) return error.CudaBufferAllocFailed;
    }
};

test "available scratch preserves the whole prompt and short tail capacity" {
    for ([_]u32{ 1, 19, 31, 32, 33, 4032, 5037 }) |tokens| {
        var probe: AllocationProbe = .{ .limit = tokens };
        try std.testing.expectEqual(tokens, try fit(&probe, tokens, AllocationProbe.allocate));
        try std.testing.expectEqual(@as(usize, 1), probe.count);
    }
}

test "explicit chunk limits preserve short prompts and reject invalid capacities" {
    try std.testing.expectEqual(@as(u32, 5037), limit(5037, null));
    for ([_][]const u8{ "0", "-1", "garbage", "4294967296", "" }) |setting|
        try std.testing.expectEqual(@as(u32, 5037), limit(5037, setting));
    try std.testing.expectEqual(@as(u32, 16), limit(5037, "16"));
    try std.testing.expectEqual(@as(u32, 1), limit(5037, "1"));
    try std.testing.expectEqual(@as(u32, 19), limit(19, "128"));
}

test "device OOM shrinks a coding prompt to a reusable batch before execution" {
    var probe: AllocationProbe = .{ .limit = 128 };
    try std.testing.expectEqual(@as(u32, 78), try fit(&probe, 5037, AllocationProbe.allocate));
    try std.testing.expectEqualSlices(u32, &.{ 5037, 2518, 1259, 629, 314, 157, 78 }, probe.attempts[0..probe.count]);
}

test "exhausted device memory terminates at the minimum batch" {
    var probe: AllocationProbe = .{ .limit = 31 };
    try std.testing.expectError(error.CudaBufferAllocFailed, fit(&probe, 129, AllocationProbe.allocate));
    try std.testing.expectEqualSlices(u32, &.{ 129, 64, 32 }, probe.attempts[0..probe.count]);
    var short: AllocationProbe = .{ .limit = 0 };
    try std.testing.expectError(error.CudaBufferAllocFailed, fit(&short, 1, AllocationProbe.allocate));
    try std.testing.expectEqual(@as(usize, 1), short.count);
}

test "execution and host allocation errors do not trigger scratch retries" {
    for ([_]anyerror{ error.GpuExecutionFailed, error.OutOfMemory, error.InvalidShape }) |err| {
        var probe: AllocationProbe = .{ .limit = 0, .failure = err };
        try std.testing.expectError(err, fit(&probe, 5037, AllocationProbe.allocate));
        try std.testing.expectEqual(@as(usize, 1), probe.count);
    }
    var empty: AllocationProbe = .{ .limit = 0 };
    try std.testing.expectError(error.EmptyPrompt, fit(&empty, 0, AllocationProbe.allocate));
    try std.testing.expectEqual(@as(usize, 0), empty.count);
}
