//! CPU integration contract for the actual server's idle-heap reclaimer.
//! Runs in a fresh process so allocator tuning cannot affect other tests.
const std = @import("std");
const builtin = @import("builtin");
const host_memory = @import("host_memory");

extern "c" fn mallopt(parameter: c_int, value: c_int) c_int;
extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
extern "c" fn read(fd: c_int, buffer: [*]u8, size: usize) isize;
extern "c" fn close(fd: c_int) c_int;

fn residentBytes() !usize {
    const fd = open("/proc/self/smaps_rollup", 0);
    if (fd < 0) return error.ResidentTelemetryUnavailable;
    defer _ = close(fd);
    var buffer: [8192]u8 = undefined;
    const count = read(fd, &buffer, buffer.len);
    if (count <= 0) return error.ResidentTelemetryUnavailable;
    var lines = std.mem.tokenizeScalar(u8, buffer[0..@intCast(count)], '\n');
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "Rss:")) continue;
        var fields = std.mem.tokenizeAny(u8, line["Rss:".len..], " \t");
        return (try std.fmt.parseInt(usize, fields.next() orelse return error.MissingResidentBytes, 10)) * 1024;
    }
    return error.MissingResidentBytes;
}

pub fn main() !void {
    if (comptime !(builtin.link_libc and builtin.target.os.tag == .linux and builtin.target.abi.isGnu())) {
        std.debug.print("Host-memory reclaim probe skipped: glibc Linux only.\n", .{});
        return;
    }

    // glibc malloc.h: disable automatic trimming (M_TRIM_THRESHOLD=-1) and
    // mmap allocations (M_MMAP_MAX=-4) in this test process only. The freed
    // region then stays resident until the production reclaimer runs.
    if (mallopt(-1, -1) != 1 or mallopt(-4, 0) != 1) return error.CannotControlProbeAllocator;
    const allocator = std.heap.c_allocator;
    const live = try allocator.alloc(u8, 64 * 1024);
    defer allocator.free(live);
    @memset(live, 0x5a);
    const retired = try allocator.alloc(u8, 64 * 1024 * 1024);
    var offset: usize = 0;
    while (offset < retired.len) : (offset += 4096) {
        const byte: *volatile u8 = @ptrCast(&retired[offset]);
        byte.* = 0xa5;
    }
    allocator.free(retired);

    const before = try residentBytes();
    host_memory.reclaimIdlePages();
    const after = try residentBytes();
    if (after >= before or before - after < 32 * 1024 * 1024) return error.UnusedPagesRemainResident;
    for (live) |*byte| {
        const observed: *volatile u8 = @ptrCast(byte);
        if (observed.* != 0x5a) return error.LiveAllocationChanged;
    }
    std.debug.print("Host-memory reclaim probe: {d} resident bytes returned; live allocation preserved.\n", .{before - after});
}
