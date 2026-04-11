//! Metal buffer wrapper — shared-mode GPU buffers with zero-copy mmap support.
//!
//! The Metal backend allocates shared-storage buffers so CPU code, the model
//! loader, and GPU kernels can all see the same memory without an explicit
//! staging copy. This module owns that wrapper and its mmap integration.
const std = @import("std");
const shim = @import("c.zig").shim;

pub const MetalBuffer = struct {
    handle: ?*shim.MetalBuf,
    size: usize,
    /// Direct CPU pointer — always non-null for SharedMode buffers on Apple Silicon.
    cpu_ptr: ?[*]u8,
    /// True if wrapping an external mmap, false if buffer owns its memory.
    is_mmap_wrapped: bool,
    /// True if this buffer holds Q8_0 data in the SIMD-coalesced repacked layout.
    is_repacked_q8: bool = false,

    pub fn contents(self: *const MetalBuffer) ?[*]u8 {
        return self.cpu_ptr;
    }

    /// Mapped pointer alias (compatibility with Vulkan buffer interface pattern).
    pub fn mapped(self: *const MetalBuffer) ?[*]u8 {
        return self.cpu_ptr;
    }
};

pub fn createBuffer(ctx: ?*shim.MetalCtx, size: usize) !MetalBuffer {
    var cpu_ptr: ?*anyopaque = null;
    const handle = shim.mtl_create_buffer(ctx, size, &cpu_ptr);
    if (handle == null) return error.MetalBufferAllocFailed;
    return .{
        .handle = handle,
        .size = size,
        .cpu_ptr = @ptrCast(cpu_ptr),
        .is_mmap_wrapped = false,
    };
}

pub fn createPrivateBuffer(ctx: ?*shim.MetalCtx, size: usize) !MetalBuffer {
    const handle = shim.mtl_create_private_buffer(ctx, size);
    if (handle == null) return error.MetalBufferAllocFailed;
    return .{
        .handle = handle,
        .size = size,
        .cpu_ptr = null,
        .is_mmap_wrapped = false,
    };
}

pub fn wrapMmap(ctx: ?*shim.MetalCtx, ptr: [*]u8, size: usize) !MetalBuffer {
    const handle = shim.mtl_wrap_mmap(ctx, ptr, size);
    if (handle == null) return error.MetalMmapWrapFailed;
    return .{
        .handle = handle,
        .size = size,
        .cpu_ptr = ptr,
        .is_mmap_wrapped = true,
    };
}

pub fn freeBuffer(buf: *MetalBuffer) void {
    if (buf.handle) |h| {
        shim.mtl_free_buffer(h);
        buf.handle = null;
    }
}

test "MetalBuffer struct size" {
    try std.testing.expect(@sizeOf(MetalBuffer) <= 32);
}

test "createBuffer allocates shared memory with CPU pointer" {
    const device_shim = @import("c.zig").shim;
    const ctx = device_shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer device_shim.mtl_destroy(ctx);

    var buf = try createBuffer(ctx, 4096);
    defer freeBuffer(&buf);

    try std.testing.expect(buf.handle != null);
    try std.testing.expectEqual(@as(usize, 4096), buf.size);
    try std.testing.expect(buf.cpu_ptr != null);
    try std.testing.expect(!buf.is_mmap_wrapped);

    // Write and read back through CPU pointer (SharedMode = CPU+GPU same memory)
    const ptr = buf.cpu_ptr.?;
    ptr[0] = 0xAB;
    ptr[1] = 0xCD;
    ptr[4095] = 0xEF;
    try std.testing.expectEqual(@as(u8, 0xAB), ptr[0]);
    try std.testing.expectEqual(@as(u8, 0xCD), ptr[1]);
    try std.testing.expectEqual(@as(u8, 0xEF), ptr[4095]);
}

test "createBuffer with various sizes" {
    const device_shim = @import("c.zig").shim;
    const ctx = device_shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer device_shim.mtl_destroy(ctx);

    // Small buffer
    var small = try createBuffer(ctx, 64);
    defer freeBuffer(&small);
    try std.testing.expectEqual(@as(usize, 64), small.size);

    // Large buffer (1 MB)
    var large = try createBuffer(ctx, 1024 * 1024);
    defer freeBuffer(&large);
    try std.testing.expectEqual(@as(usize, 1024 * 1024), large.size);
}

test "createPrivateBuffer allocates GPU-only memory without CPU pointer" {
    const device_shim = @import("c.zig").shim;
    const ctx = device_shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer device_shim.mtl_destroy(ctx);

    var buf = try createPrivateBuffer(ctx, 4096);
    defer freeBuffer(&buf);

    try std.testing.expect(buf.handle != null);
    try std.testing.expectEqual(@as(usize, 4096), buf.size);
    try std.testing.expect(buf.cpu_ptr == null);
    try std.testing.expect(!buf.is_mmap_wrapped);
}

test "createBuffer with zero size returns error" {
    const device_shim = @import("c.zig").shim;
    const ctx = device_shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer device_shim.mtl_destroy(ctx);

    try std.testing.expectError(error.MetalBufferAllocFailed, createBuffer(ctx, 0));
}

test "wrapMmap wraps page-aligned memory" {
    const device_shim = @import("c.zig").shim;
    const ctx = device_shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer device_shim.mtl_destroy(ctx);

    // mmap a page-aligned region
    const size = 4096 * 4; // 16 KB
    const data = try std.posix.mmap(
        null,
        size,
        std.posix.PROT.READ | std.posix.PROT.WRITE,
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
        -1,
        0,
    );
    defer std.posix.munmap(data);

    // Write test pattern
    data[0] = 0x42;
    data[size - 1] = 0x99;

    var buf = try wrapMmap(ctx, data.ptr, size);
    defer freeBuffer(&buf);

    try std.testing.expect(buf.handle != null);
    try std.testing.expectEqual(@as(usize, size), buf.size);
    try std.testing.expect(buf.is_mmap_wrapped);

    // CPU pointer should point to the original mmap'd memory
    try std.testing.expectEqual(@as(u8, 0x42), buf.cpu_ptr.?[0]);
    try std.testing.expectEqual(@as(u8, 0x99), buf.cpu_ptr.?[size - 1]);
}

test "freeBuffer with null handle is safe" {
    var buf = MetalBuffer{
        .handle = null,
        .size = 0,
        .cpu_ptr = null,
        .is_mmap_wrapped = false,
    };
    freeBuffer(&buf); // should not crash
}

test "contents and mapped return same pointer" {
    const device_shim = @import("c.zig").shim;
    const ctx = device_shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer device_shim.mtl_destroy(ctx);

    var buf = try createBuffer(ctx, 256);
    defer freeBuffer(&buf);

    try std.testing.expectEqual(buf.contents(), buf.mapped());
    try std.testing.expect(buf.contents() != null);
}
