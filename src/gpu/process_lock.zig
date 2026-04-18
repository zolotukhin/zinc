//! Cross-process GPU reservation lock keyed by backend and selected device.
//!
//! ZINC uses a filesystem lock to stop multiple inference processes from
//! loading different models onto the same physical GPU at once, which would
//! otherwise produce confusing OOM failures and unstable benchmark results.
//! @section Inference Runtime
const std = @import("std");

/// Backend identifier encoded into the shared GPU lockfile name.
pub const Backend = enum {
    vulkan,
    metal,
};

/// Cross-process lock handle that reserves one backend/device pair.
pub const ProcessLock = struct {
    file: ?std.fs.File = null,

    /// Return whether the lock currently owns an open lockfile handle.
    pub fn isHeld(self: *const ProcessLock) bool {
        return self.file != null;
    }

    /// Release the held lockfile handle, if any.
    pub fn deinit(self: *ProcessLock) void {
        if (self.file) |file| {
            file.close();
            self.file = null;
        }
    }
};

/// Errors returned while acquiring a backend/device GPU reservation lock.
pub const AcquireError = std.fs.File.OpenError || error{
    GpuAlreadyReserved,
    LockPathTooLong,
};

/// Format the lockfile path for a backend/device pair into the caller buffer.
pub fn lockPath(buffer: []u8, backend: Backend, device_index: u32) error{LockPathTooLong}![]const u8 {
    return std.fmt.bufPrint(buffer, "/tmp/zinc-gpu-{s}-{d}.lock", .{
        @tagName(backend),
        device_index,
    }) catch error.LockPathTooLong;
}

/// Acquire the cross-process GPU reservation lock for a backend/device pair.
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
