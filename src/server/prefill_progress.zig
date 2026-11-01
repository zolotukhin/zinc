//! Streaming-prefill liveness probes. A graceful TCP half-close is ambiguous;
//! periodic SSE comments detect a reader that has gone away without rejecting
//! clients that close their upload side while continuing to read.
const std = @import("std");
const host_io = @import("host_io");

pub fn ConnectionProgress(comptime Connection: type) type {
    return struct {
        const Self = @This();
        conn: *Connection,
        streaming: bool = true,
        last_keepalive_ns: i128,
        cancelled: bool = false,

        pub fn poll(context: *anyopaque) bool {
            const self: *Self = @ptrCast(@alignCast(context));
            return self.pollAt(host_io.nanoTimestamp());
        }

        fn pollAt(self: *Self, now_ns: i128) bool {
            if (self.cancelled) return false;
            if (self.conn.isPeerClosed()) {
                self.cancelled = true;
                return false;
            }
            if (self.streaming and now_ns - self.last_keepalive_ns >= std.time.ns_per_s) {
                self.conn.writeSseComment("prefill") catch {
                    self.cancelled = true;
                    return false;
                };
                self.last_keepalive_ns = now_ns;
            }
            return true;
        }
    };
}

test "prefill keepalive tolerates half-close and stops on reset or failed write" {
    const FakeConnection = struct {
        reset: bool = false,
        write_failed: bool = false,
        writes: usize = 0,
        fn isPeerClosed(self: *@This()) bool {
            return self.reset;
        }
        fn writeSseComment(self: *@This(), text: []const u8) !void {
            try std.testing.expectEqualStrings("prefill", text);
            self.writes += 1;
            if (self.write_failed) return error.BrokenPipe;
        }
    };
    var conn = FakeConnection{};
    var progress = ConnectionProgress(FakeConnection){ .conn = &conn, .last_keepalive_ns = 0 };
    try std.testing.expect(progress.pollAt(0));
    try std.testing.expect(progress.pollAt(std.time.ns_per_s));
    try std.testing.expect(progress.pollAt(std.time.ns_per_s + 1));
    try std.testing.expectEqual(@as(usize, 1), conn.writes);
    conn.write_failed = true;
    try std.testing.expect(!progress.pollAt(2 * std.time.ns_per_s));
    conn.write_failed = false;
    try std.testing.expect(!progress.pollAt(3 * std.time.ns_per_s));
    try std.testing.expectEqual(@as(usize, 2), conn.writes);
    var reset = FakeConnection{ .reset = true };
    var stopped = ConnectionProgress(FakeConnection){ .conn = &reset, .last_keepalive_ns = 0 };
    try std.testing.expect(!stopped.pollAt(std.time.ns_per_s));
    try std.testing.expectEqual(@as(usize, 0), reset.writes);
    var json_conn = FakeConnection{};
    var json_progress = ConnectionProgress(FakeConnection){
        .conn = &json_conn,
        .streaming = false,
        .last_keepalive_ns = 0,
    };
    try std.testing.expect(json_progress.pollAt(2 * std.time.ns_per_s));
    try std.testing.expectEqual(@as(usize, 0), json_conn.writes);
    json_conn.reset = true;
    try std.testing.expect(!json_progress.pollAt(3 * std.time.ns_per_s));
    try std.testing.expectEqual(@as(usize, 0), json_conn.writes);
}
