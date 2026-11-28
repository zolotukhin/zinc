//! Per-request draft-depth adjustment for Inkling 2 Mini DSpark.
//! @section Inference Runtime
//!
//! Adjusts the proposal window using recent target acceptance and resets it at each request.
const std = @import("std");

pub const Depth = struct {
    depth: u32,
    ceiling: u32,
    rounds: u32 = 0,
    accepted: u32 = 0,

    pub fn init(limit: u32) Depth {
        return .{ .depth = @min(limit, 2), .ceiling = limit };
    }

    pub fn observe(self: *Depth, proposed: u32, accepted: u32) void {
        if (proposed != self.depth) return; // The request is near its output/context limit.
        self.rounds += 1;
        self.accepted += accepted;
        if (self.rounds < 6) return;
        const ratio = self.accepted * 100;
        const offered = self.rounds * proposed;
        if (self.depth == 2 and ratio >= offered * 85) {
            self.depth = @min(self.ceiling, 3);
        } else if (self.depth == 3) {
            if (ratio >= offered * 85) {
                self.depth = self.ceiling;
            } else if (ratio <= offered * 56) {
                self.depth = 2;
            }
        } else if (self.depth > 3 and ratio <= offered * 55) {
            self.ceiling = 3;
            self.depth = 2;
        }
        self.rounds = 0;
        self.accepted = 0;
    }
};

test "expands on high acceptance and backs off after a low-yield wide block" {
    var d = Depth.init(7);
    for (0..6) |_| d.observe(2, 2);
    try std.testing.expectEqual(@as(u32, 3), d.depth);
    for (0..6) |_| d.observe(3, 3);
    try std.testing.expectEqual(@as(u32, 7), d.depth);
    for (0..6) |_| d.observe(7, 3);
    try std.testing.expectEqual(@as(u32, 2), d.depth);
    for (0..6) |_| d.observe(2, 2);
    try std.testing.expectEqual(@as(u32, 3), d.depth);
    for (0..6) |_| d.observe(3, 3);
    try std.testing.expectEqual(@as(u32, 3), d.depth);
}

test "a request with little room does not update the active draft window" {
    var d = Depth.init(3);
    for (0..12) |_| d.observe(1, 1);
    try std.testing.expectEqual(@as(u32, 2), d.depth);
    try std.testing.expectEqual(@as(u32, 0), d.rounds);
}
