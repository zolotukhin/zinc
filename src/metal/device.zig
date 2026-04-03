//! Metal device wrapper — macOS Apple Silicon GPU backend.
//! Stub implementation: T005 will fill in the full MetalDevice.
const std = @import("std");
const shim = @import("c.zig").shim;

pub const GpuFamily = enum(u32) {
    apple7 = 7,
    apple8 = 8,
    apple9 = 9,
    apple10 = 10,
    unknown = 0,

    pub fn isApple9OrNewer(self: @This()) bool {
        return switch (self) {
            .apple9, .apple10 => true,
            else => false,
        };
    }

    pub fn isM5Class(self: @This()) bool {
        return self == .apple10;
    }
};

pub const MetalCapabilities = struct {
    supports_apple7: bool,
    supports_apple8: bool,
    supports_apple9: bool,
    supports_apple10: bool,
    supports_mac2: bool,
    has_unified_memory: bool,
    supports_raytracing: bool,
    recommended_max_working_set_size: u64,
    max_threadgroup_memory_length: u64,
};

pub const MetalDevice = struct {
    ctx: ?*shim.MetalCtx,
    chip: GpuFamily,
    caps: MetalCapabilities,
    selected_device_index: u32,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, _: u32) !MetalDevice {
        const ctx = shim.mtl_init();
        if (ctx == null) {
            return error.MetalInitFailed;
        }
        const family_raw = shim.mtl_chip_family(ctx);
        const chip: GpuFamily = if (family_raw == 7 or family_raw == 8 or family_raw == 9 or family_raw == 10)
            @enumFromInt(family_raw)
        else
            .unknown;
        return .{
            .ctx = ctx,
            .chip = chip,
            .caps = .{
                .supports_apple7 = shim.mtl_supports_family(ctx, shim.ZINC_MTL_GPU_FAMILY_APPLE7) != 0,
                .supports_apple8 = shim.mtl_supports_family(ctx, shim.ZINC_MTL_GPU_FAMILY_APPLE8) != 0,
                .supports_apple9 = shim.mtl_supports_family(ctx, shim.ZINC_MTL_GPU_FAMILY_APPLE9) != 0,
                .supports_apple10 = shim.mtl_supports_family(ctx, shim.ZINC_MTL_GPU_FAMILY_APPLE10) != 0,
                .supports_mac2 = shim.mtl_supports_family(ctx, shim.ZINC_MTL_GPU_FAMILY_MAC2) != 0,
                .has_unified_memory = shim.mtl_has_unified_memory(ctx) != 0,
                .supports_raytracing = shim.mtl_supports_raytracing(ctx) != 0,
                .recommended_max_working_set_size = shim.mtl_recommended_max_working_set_size(ctx),
                .max_threadgroup_memory_length = shim.mtl_max_threadgroup_memory_length(ctx),
            },
            .selected_device_index = 0,
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

    pub fn recommendedMaxWorkingSetSize(self: *const MetalDevice) u64 {
        return self.caps.recommended_max_working_set_size;
    }

    pub fn maxThreadgroupMemoryLength(self: *const MetalDevice) u64 {
        return self.caps.max_threadgroup_memory_length;
    }

    pub fn hasUnifiedMemory(self: *const MetalDevice) bool {
        return self.caps.has_unified_memory;
    }

    pub fn supportsRaytracing(self: *const MetalDevice) bool {
        return self.caps.supports_raytracing;
    }
};

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

    try std.testing.expect(device.hasUnifiedMemory());
    try std.testing.expect(device.maxThreadgroupMemoryLength() > 0);
    try std.testing.expect(device.recommendedMaxWorkingSetSize() > 0);
}

test "MetalDevice double deinit is safe" {
    var device = try MetalDevice.init(std.testing.allocator, 0);
    device.deinit();
    // Second deinit should be a no-op (ctx is null)
    device.deinit();
}

test "GpuFamily helpers" {
    try std.testing.expect(!GpuFamily.apple7.isApple9OrNewer());
    try std.testing.expect(!GpuFamily.apple8.isApple9OrNewer());
    try std.testing.expect(GpuFamily.apple9.isApple9OrNewer());
    try std.testing.expect(GpuFamily.apple10.isApple9OrNewer());
    try std.testing.expect(!GpuFamily.apple9.isM5Class());
    try std.testing.expect(GpuFamily.apple10.isM5Class());
}
