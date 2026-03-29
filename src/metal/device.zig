//! Metal device wrapper — macOS Apple Silicon GPU backend.
//! Stub implementation: T005 will fill in the full MetalDevice.
const std = @import("std");
const shim = @import("c.zig").shim;

pub const ChipFamily = enum(u32) {
    m1 = 1,
    m2 = 2,
    m3 = 3,
    m4 = 4,
    unknown = 0,

    pub fn hasAsyncCopy(self: @This()) bool {
        return self != .m1 and self != .unknown;
    }
};

pub const MetalDevice = struct {
    ctx: ?*shim.MetalCtx,
    chip: ChipFamily,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, _: u32) !MetalDevice {
        const ctx = shim.mtl_init();
        if (ctx == null) {
            return error.MetalInitFailed;
        }
        const family_raw = shim.mtl_chip_family(ctx);
        const chip: ChipFamily = if (family_raw >= 1 and family_raw <= 4)
            @enumFromInt(family_raw)
        else
            .unknown;
        return .{
            .ctx = ctx,
            .chip = chip,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *MetalDevice) void {
        if (self.ctx) |ctx| {
            shim.mtl_destroy(ctx);
            self.ctx = null;
        }
    }

    pub fn maxBufferSize(self: *const MetalDevice) u64 {
        if (self.ctx) |ctx| return shim.mtl_max_buffer_size(ctx);
        return 0;
    }

    pub fn totalMemory(self: *const MetalDevice) u64 {
        if (self.ctx) |ctx| return shim.mtl_total_memory(ctx);
        return 0;
    }
};

test "ChipFamily hasAsyncCopy" {
    try std.testing.expect(!ChipFamily.m1.hasAsyncCopy());
    try std.testing.expect(ChipFamily.m2.hasAsyncCopy());
    try std.testing.expect(ChipFamily.m3.hasAsyncCopy());
    try std.testing.expect(ChipFamily.m4.hasAsyncCopy());
    try std.testing.expect(!ChipFamily.unknown.hasAsyncCopy());
}

test "MetalDevice init and deinit" {
    var device = try MetalDevice.init(std.testing.allocator, 0);
    defer device.deinit();

    // Device should be initialized
    try std.testing.expect(device.ctx != null);

    // Chip should be detected (on Apple Silicon)
    try std.testing.expect(device.chip != .unknown);
}

test "MetalDevice reports nonzero memory" {
    var device = try MetalDevice.init(std.testing.allocator, 0);
    defer device.deinit();

    // Total memory should be >0 (any Apple Silicon has at least 8 GB)
    const total = device.totalMemory();
    try std.testing.expect(total > 0);
    try std.testing.expect(total >= 8 * 1024 * 1024 * 1024); // >= 8 GB

    // Max buffer size should be >0 and <= total memory
    const max_buf = device.maxBufferSize();
    try std.testing.expect(max_buf > 0);
    try std.testing.expect(max_buf <= total);
}

test "MetalDevice double deinit is safe" {
    var device = try MetalDevice.init(std.testing.allocator, 0);
    device.deinit();
    // Second deinit should be a no-op (ctx is null)
    device.deinit();
}
