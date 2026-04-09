//! Cross-process GPU reservation lock keyed by backend and selected device.
//!
//! ZINC uses a filesystem lock to stop multiple inference processes from
//! loading different models onto the same physical GPU at once, which would
//! otherwise produce confusing OOM failures and unstable benchmark results.
const std = @import("std");

pub const Backend = enum {
    vulkan,
    metal,
};

pub const ProcessLock = struct {
    file: ?std.fs.File = null,

    pub fn isHeld(self: *const ProcessLock) bool {
        return self.file != null;
    }

    pub fn deinit(self: *ProcessLock) void {
        if (self.file) |file| {
            file.close();
            self.file = null;
        }
    }
};

pub const AcquireError = std.fs.File.OpenError || error{
    GpuAlreadyReserved,
    LockPathTooLong,
};

pub fn lockPath(buffer: []u8, backend: Backend, device_index: u32) error{LockPathTooLong}![]const u8 {
    return std.fmt.bufPrint(buffer, "/tmp/zinc-gpu-{s}-{d}.lock", .{
        @tagName(backend),
        device_index,
    }) catch error.LockPathTooLong;
}

pub fn acquire(backend: Backend, device_index: u32) AcquireError!ProcessLock {
    var path_buffer: [64]u8 = undefined;
    const path = try lockPath(&path_buffer, backend, device_index);
    const file = std.fs.createFileAbsolute(path, .{
        .read = true,
        .truncate = false,
        .lock = .exclusive,
        .lock_nonblocking = true,
    }) catch |err| switch (err) {
        error.WouldBlock => return error.GpuAlreadyReserved,
        else => return err,
    };
    return .{ .file = file };
}

test "lockPath includes backend and device index" {
    var buffer: [64]u8 = undefined;
    const vulkan_path = try lockPath(&buffer, .vulkan, 3);
    try std.testing.expectEqualStrings("/tmp/zinc-gpu-vulkan-3.lock", vulkan_path);

    const metal_path = try lockPath(&buffer, .metal, 0);
    try std.testing.expectEqualStrings("/tmp/zinc-gpu-metal-0.lock", metal_path);
}
