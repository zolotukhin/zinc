//! Cross-process GPU reservation lock keyed by backend and selected device.
//! @section Inference Runtime
//!
//! ZINC uses a filesystem lock to stop multiple inference processes from
//! loading different models onto the same physical GPU at once, which would
//! otherwise produce confusing OOM failures and unstable benchmark results.
const std = @import("std");
const host_io = @import("host_io");

/// Backend identifier encoded into the shared GPU lockfile name.
pub const Backend = enum {
    vulkan,
    metal,
};

/// Cross-process lock handle that reserves one backend/device pair.
pub const ProcessLock = struct {
    file: ?std.Io.File = null,

    /// Return whether the lock currently owns an open lockfile handle.
    pub fn isHeld(self: *const ProcessLock) bool {
        return self.file != null;
    }

    /// Close this process's OS file reservation. There are no thread waiters:
    /// acquire() opens a separate handle with lock_nonblocking=true. Releasing
    /// permits another process (or a later model activation) to acquire anew.
    pub fn release(self: *ProcessLock) void {
        if (self.file) |file| {
            file.close(host_io.io);
            self.file = null;
        }
    }

    /// Scope cleanup uses the same idempotent file release as model unload.
    pub fn deinit(self: *ProcessLock) void {
        self.release();
    }
};

/// Errors returned while acquiring a backend/device GPU reservation lock.
pub const AcquireError = std.Io.File.OpenError || error{
    GpuAlreadyReserved,
    LockPathTooLong,
};

/// Format the lockfile path for a backend/device pair into the caller-supplied buffer.
///
/// @param buffer Destination slice for the formatted path; must be large enough to hold the formatted path.
/// @param backend The GPU backend whose tag name is embedded in the path.
/// @param device_index Zero-based device index embedded in the path.
/// @returns A slice into `buffer` holding the formatted path string, e.g. `/tmp/zinc-gpu-vulkan-0.lock`.
/// @note Returns `error.LockPathTooLong` if `buffer` is too small to hold the formatted path.
pub fn lockPath(buffer: []u8, backend: Backend, device_index: u32) error{LockPathTooLong}![]const u8 {
    return std.fmt.bufPrint(buffer, "/tmp/zinc-gpu-{s}-{d}.lock", .{
        @tagName(backend),
        device_index,
    }) catch error.LockPathTooLong;
}

/// Acquire the cross-process GPU reservation lock for a backend/device pair.
///
/// Opens the lockfile in non-blocking exclusive mode so that a second process
/// attempting to claim the same GPU immediately receives `error.GpuAlreadyReserved`
/// rather than blocking indefinitely.
///
/// @param backend The GPU backend to reserve.
/// @param device_index Zero-based index of the device to reserve within that backend.
/// @returns A `ProcessLock` holding the open lockfile handle; caller must call `deinit` to release.
/// @note Returns `error.GpuAlreadyReserved` if another process already holds the lock.
pub fn acquire(backend: Backend, device_index: u32) AcquireError!ProcessLock {
    var path_buffer: [64]u8 = undefined;
    const path = try lockPath(&path_buffer, backend, device_index);
    const file = std.Io.Dir.createFileAbsolute(host_io.io, path, .{
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

test "GPU reservation rejects a second handle and releases on close" {
    // A synthetic index reserves no physical GPU and does not share live locks.
    const index = 1_000_000 + @as(u32, @intCast(std.c.getpid()));
    var buffer: [64]u8 = undefined;
    const path = try lockPath(&buffer, .vulkan, index);
    defer std.Io.Dir.deleteFileAbsolute(host_io.io, path) catch {};
    var first = try acquire(.vulkan, index);
    defer first.deinit();
    try std.testing.expectError(error.GpuAlreadyReserved, acquire(.vulkan, index));
    first.release();
    first.release();
    try std.testing.expect(!first.isHeld());
    var replacement = try acquire(.vulkan, index);
    defer replacement.deinit();
    try std.testing.expect(replacement.isHeld());
}
