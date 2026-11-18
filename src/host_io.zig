//! Process I/O shared by host orchestration and the standalone tools.
//! Production entrypoints install the process's standard I/O implementation;
//! unit tests use the test runner's implementation.
const std = @import("std");
const builtin = @import("builtin");

var fallback: std.Io.Threaded = .init_single_threaded;
pub var io: std.Io = if (builtin.is_test) std.testing.io else fallback.io();
var process_args: std.process.Args = .{ .vector = &.{} };

pub fn init(process: std.process.Init) void {
    io = process.io;
    process_args = process.minimal.args;
}

pub fn argsAlloc(allocator: std.mem.Allocator) ![][:0]u8 {
    var result: std.ArrayList([:0]u8) = .empty;
    errdefer {
        for (result.items) |arg| allocator.free(arg);
        result.deinit(allocator);
    }
    var iterator = try process_args.iterateAllocator(allocator);
    defer iterator.deinit();
    while (iterator.next()) |arg| {
        const owned = try allocator.dupeSentinel(u8, arg, 0);
        errdefer allocator.free(owned);
        try result.append(allocator, owned);
    }
    return result.toOwnedSlice(allocator);
}

pub fn argsFree(allocator: std.mem.Allocator, args: []const [:0]u8) void {
    for (args) |arg| allocator.free(arg);
    allocator.free(args);
}

pub fn argsWithAllocator(allocator: std.mem.Allocator) !std.process.Args.Iterator {
    return process_args.iterateAllocator(allocator);
}

/// Read the live process environment, including changes made after startup.
pub fn getenv(name: []const u8) ?[:0]const u8 {
    var buffer: [4096]u8 = undefined;
    if (name.len >= buffer.len or std.mem.indexOfScalar(u8, name, 0) != null) return null;
    @memcpy(buffer[0..name.len], name);
    buffer[name.len] = 0;
    const value = std.c.getenv(buffer[0..name.len :0]) orelse return null;
    return std.mem.span(value);
}

pub fn repeatText(comptime text: []const u8, comptime count: usize) [text.len * count]u8 {
    var result: [text.len * count]u8 = undefined;
    for (0..count) |i| @memcpy(result[i * text.len ..][0..text.len], text);
    return result;
}

pub fn timestamp() i64 {
    return @intCast(@divFloor(nanoTimestamp(), std.time.ns_per_s));
}

pub fn getEnvVarOwned(allocator: std.mem.Allocator, name: []const u8) ![]u8 {
    const value = getenv(name) orelse return error.EnvironmentVariableNotFound;
    return allocator.dupe(u8, value);
}

pub const Mutex = struct {
    value: std.Io.Mutex = .init,

    pub fn lock(self: *Mutex) void {
        self.value.lockUncancelable(io);
    }

    pub fn unlock(self: *Mutex) void {
        self.value.unlock(io);
    }

    pub fn tryLock(self: *Mutex) bool {
        return self.value.tryLock();
    }
};

/// Reader leases keep model resources alive until every handler finishes.
/// A lifecycle change uses tryLock so a busy model returns a retryable error.
pub const RwLock = struct {
    value: std.Io.RwLock = .init,

    pub fn lockShared(self: *RwLock) void {
        self.value.lockSharedUncancelable(io);
    }
    pub fn unlockShared(self: *RwLock) void {
        self.value.unlockShared(io);
    }
    pub fn tryLockShared(self: *RwLock) bool {
        return self.value.tryLockShared(io);
    }
    pub fn tryLock(self: *RwLock) bool {
        return self.value.tryLock(io);
    }
    pub fn unlock(self: *RwLock) void {
        self.value.unlock(io);
    }
};

pub const Condition = struct {
    value: std.Io.Condition = .init,

    pub fn wait(self: *Condition, mutex: *Mutex) void {
        self.value.waitUncancelable(io, &mutex.value);
    }

    pub fn signal(self: *Condition) void {
        self.value.signal(io);
    }

    pub fn broadcast(self: *Condition) void {
        self.value.broadcast(io);
    }
};

/// Bounded CPU executor for the reference runtime's optional general pool.
/// The dedicated allocation-free decode pool remains its preferred hot path.
pub const Pool = struct {
    backend: std.Io.Threaded,
    n_jobs: usize,

    pub const Options = struct {
        allocator: std.mem.Allocator,
        n_jobs: usize,
    };

    pub fn init(self: *Pool, options: Options) !void {
        self.* = .{
            .backend = .init(options.allocator, .{
                .async_limit = .limited(options.n_jobs),
                .concurrent_limit = .limited(options.n_jobs),
            }),
            .n_jobs = options.n_jobs,
        };
    }

    pub fn deinit(self: *Pool) void {
        self.backend.deinit();
        self.* = undefined;
    }

    pub fn spawnWg(self: *Pool, group: *std.Io.Group, comptime function: anytype, args: anytype) void {
        group.async(self.backend.io(), function, args);
    }

    pub fn waitAndWork(self: *Pool, group: *std.Io.Group) void {
        group.await(self.backend.io()) catch unreachable;
    }
};

pub fn readToEndAlloc(file: std.Io.File, allocator: std.mem.Allocator, limit: usize) ![]u8 {
    var reader = file.reader(io, &.{});
    return reader.interface.allocRemaining(allocator, .limited(limit));
}

pub fn readAll(file: std.Io.File, buffer: []u8) !usize {
    var reader = file.readerStreaming(io, &.{});
    return reader.interface.readSliceShort(buffer);
}

pub fn read(file: std.Io.File, buffer: []u8) !usize {
    if (buffer.len == 0) return 0;
    return file.readStreaming(io, &.{buffer}) catch |err| switch (err) {
        error.EndOfStream => 0,
        else => return err,
    };
}

pub fn writeSocketAll(stream: std.Io.net.Stream, bytes: []const u8) !void {
    var writer = stream.writer(io, &.{});
    try writer.interface.writeAll(bytes);
}

/// Send a frame without copying its payload or issuing one write per fragment.
/// The writer handles short writes and restores the supplied slice descriptors.
pub fn writeSocketVecAll(stream: std.Io.net.Stream, buffers: [][]const u8) !void {
    var writer = stream.writer(io, &.{});
    try writer.interface.writeVecAll(buffers);
}

pub fn readSocket(stream: std.Io.net.Stream, bytes: []u8) !usize {
    var buffers = [_][]u8{bytes};
    const result = try (try io.operate(.{ .net_read = .{
        .socket_handle = stream.socket.handle,
        .data = &buffers,
    } })).net_read;
    return result.data_len;
}

pub fn nanoTimestamp() i128 {
    return std.Io.Timestamp.now(io, .real).nanoseconds;
}

pub fn milliTimestamp() i64 {
    return @intCast(@divFloor(nanoTimestamp(), std.time.ns_per_ms));
}

pub fn sleep(nanoseconds: u64) void {
    io.sleep(.fromNanoseconds(nanoseconds), .awake) catch {};
}

pub const Timer = struct {
    started: std.Io.Timestamp,

    pub fn start() !Timer {
        return .{ .started = .now(io, .awake) };
    }

    pub fn read(self: *const Timer) u64 {
        const now = std.Io.Timestamp.now(io, .awake);
        return @intCast(@max(0, self.started.durationTo(now).nanoseconds));
    }

    pub fn reset(self: *Timer) void {
        self.started = .now(io, .awake);
    }

    pub fn lap(self: *Timer) u64 {
        const now = std.Io.Timestamp.now(io, .awake);
        const elapsed = self.started.durationTo(now);
        self.started = now;
        return @intCast(@max(0, elapsed.nanoseconds));
    }
};

test "environment reads observe changes after process initialization" {
    const Environment = struct {
        extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
        extern "c" fn unsetenv(name: [*:0]const u8) c_int;
    };
    const key = "ZINC_HOST_IO_TEST_ENVIRONMENT";
    const previous = if (getenv(key)) |value| try std.testing.allocator.dupeSentinel(u8, value, 0) else null;
    defer {
        if (previous) |value| {
            _ = Environment.setenv(key, value, 1);
            std.testing.allocator.free(value);
        } else _ = Environment.unsetenv(key);
    }
    try std.testing.expectEqual(@as(c_int, 0), Environment.setenv(key, "first", 1));
    const first = try getEnvVarOwned(std.testing.allocator, key);
    defer std.testing.allocator.free(first);
    try std.testing.expectEqual(@as(c_int, 0), Environment.setenv(key, "second", 1));
    const second = try getEnvVarOwned(std.testing.allocator, key);
    defer std.testing.allocator.free(second);
    try std.testing.expectEqualStrings("first", first);
    try std.testing.expectEqualStrings("second", second);
}

test "streaming file reads preserve short reads and zero-byte EOF" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "data", .data = "abc" });
    const file = try tmp.dir.openFile(io, "data", .{});
    defer file.close(io);
    try std.testing.expectEqual(@as(usize, 0), try read(file, &.{}));
    var buffer: [8]u8 = undefined;
    try std.testing.expectEqual(@as(usize, 3), try read(file, &buffer));
    try std.testing.expectEqualStrings("abc", buffer[0..3]);
    try std.testing.expectEqual(@as(usize, 0), try read(file, &buffer));
    try std.testing.expectEqual(@as(usize, 0), try read(file, &buffer));
}

test "CPU executor completes every borrowed task before its group returns" {
    const Task = struct {
        count: *std.atomic.Value(u32),
        output: *u32,
        value: u32,

        fn run(self: *@This()) void {
            self.output.* = self.value;
            _ = self.count.fetchAdd(1, .monotonic);
        }
    };
    for ([_]u32{ 0, 1, 3 }) |workers| {
        var pool: Pool = undefined;
        try pool.init(.{ .allocator = std.heap.smp_allocator, .n_jobs = workers });
        defer pool.deinit();
        for (0..3) |_| {
            var group: std.Io.Group = .init;
            var count: std.atomic.Value(u32) = .init(0);
            var output: [17]u32 = @splat(0);
            var tasks: [output.len]Task = undefined;
            for (&tasks, &output, 0..) |*task, *value, i| {
                task.* = .{ .count = &count, .output = value, .value = @intCast(i + 1) };
                pool.spawnWg(&group, Task.run, .{task});
            }
            pool.waitAndWork(&group);
            try std.testing.expectEqual(@as(u32, output.len), count.load(.monotonic));
            for (output, 0..) |value, i| try std.testing.expectEqual(@as(u32, @intCast(i + 1)), value);
        }
    }
}
