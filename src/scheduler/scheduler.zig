//! Continuous batching scheduler for concurrent inference requests.
//! @section Scheduler
//! Manages multiple active requests, assigns KV cache slots, and
//! dispatches batched prefill and decode operations to the GPU.
const std = @import("std");
const Request = @import("request.zig").Request;
const RequestState = @import("request.zig").RequestState;
const GenerationParams = @import("request.zig").GenerationParams;

const log = std.log.scoped(.scheduler);

/// Scheduler that manages concurrent inference requests.
pub const Scheduler = struct {
    /// Active requests indexed by slot ID.
    slots: []?Request,
    /// Maximum number of concurrent requests.
    max_parallel: u32,
    /// Next request ID counter.
    next_id: u64,
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,

    /// Initialize the scheduler with a fixed number of concurrent request slots.
    /// @param allocator Allocator for the slot array.
    /// @param max_parallel Maximum number of concurrent requests.
    /// @returns A Scheduler with all slots initially empty.
    pub fn init(allocator: std.mem.Allocator, max_parallel: u32) !Scheduler {
        const slots = try allocator.alloc(?Request, max_parallel);
        @memset(slots, null);
        log.info("Scheduler ready: {d} slots", .{max_parallel});
        return .{
            .slots = slots,
            .max_parallel = max_parallel,
            .next_id = 1,
            .allocator = allocator,
        };
    }

    /// Submit a new request and assign it to a free slot.
    /// @param self Scheduler to submit to.
    /// @param prompt_tokens Tokenized prompt for the request.
    /// @param params Generation parameters (max_tokens, temperature, etc.).
    /// @returns The assigned slot ID.
    /// @note Returns error.AllSlotsBusy if no free slots are available.
    pub fn submit(self: *Scheduler, prompt_tokens: []const u32, params: GenerationParams) !u32 {
        // Find a free slot
        for (self.slots, 0..) |*slot, i| {
            if (slot.* == null) {
                const id = self.next_id;
                self.next_id += 1;
                var req = Request.init(self.allocator, id, prompt_tokens, params);
                req.slot_id = @intCast(i);
                slot.* = req;
                log.info("Request {d} assigned to slot {d} ({d} prompt tokens)", .{ id, i, prompt_tokens.len });
                return @intCast(i);
            }
        }
        return error.AllSlotsBusy;
    }

    /// Check if all slots are occupied.
    /// @param self Scheduler to query.
    /// @returns True if every slot holds an active request.
    pub fn isFull(self: *const Scheduler) bool {
        return self.activeCount() >= self.max_parallel;
    }

    /// Get the number of active (non-null) requests.
    /// @param self Scheduler to query.
    /// @returns Count of occupied slots.
    pub fn activeCount(self: *const Scheduler) u32 {
        var count: u32 = 0;
        for (self.slots) |slot| {
            if (slot != null) count += 1;
        }
        return count;
    }

    /// Return slot IDs of requests in the prefilling state.
    /// @returns Empty slice (stub — allocation strategy TBD).
    pub fn pendingPrefill(self: *Scheduler) []u32 {
        // Returns slot IDs of pending requests
        // TODO: implement with proper allocation
        _ = self;
        return &.{};
    }

    /// Return slot IDs of requests in the decoding state.
    /// @returns Empty slice (stub — allocation strategy TBD).
    pub fn activeDecoding(self: *Scheduler) []u32 {
        _ = self;
        return &.{};
    }

    /// Release a completed or cancelled request's slot, freeing its resources.
    /// @param self Scheduler to release from.
    /// @param slot_id Slot index to free.
    pub fn release(self: *Scheduler, slot_id: u32) void {
        if (slot_id < self.slots.len) {
            if (self.slots[slot_id]) |*req| {
                req.deinit();
                self.slots[slot_id] = null;
                log.info("Released slot {d}", .{slot_id});
            }
        }
    }

    /// Tear down all active requests and free the slot array.
    /// @param self Scheduler to destroy.
    pub fn deinit(self: *Scheduler) void {
        for (self.slots) |*slot| {
            if (slot.*) |*req| req.deinit();
        }
        self.allocator.free(self.slots);
    }
};

test "Scheduler submit and release" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 4);
    defer sched.deinit();

    try std.testing.expectEqual(@as(u32, 0), sched.activeCount());

    const slot0 = try sched.submit(&.{ 1, 2, 3 }, .{});
    try std.testing.expectEqual(@as(u32, 0), slot0);
    try std.testing.expectEqual(@as(u32, 1), sched.activeCount());

    const slot1 = try sched.submit(&.{ 4, 5 }, .{});
    try std.testing.expectEqual(@as(u32, 1), slot1);
    try std.testing.expectEqual(@as(u32, 2), sched.activeCount());

    sched.release(0);
    try std.testing.expectEqual(@as(u32, 1), sched.activeCount());
}

test "Scheduler full" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 2);
    defer sched.deinit();

    _ = try sched.submit(&.{1}, .{});
    _ = try sched.submit(&.{2}, .{});
    try std.testing.expectError(error.AllSlotsBusy, sched.submit(&.{3}, .{}));
}

test "Scheduler isFull" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 2);
    defer sched.deinit();

    try std.testing.expect(!sched.isFull());
    _ = try sched.submit(&.{1}, .{});
    try std.testing.expect(!sched.isFull());
    _ = try sched.submit(&.{2}, .{});
    try std.testing.expect(sched.isFull());
    sched.release(0);
    try std.testing.expect(!sched.isFull());
}

test "Scheduler release and reuse slot" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 1);
    defer sched.deinit();

    const s1 = try sched.submit(&.{10}, .{});
    try std.testing.expectEqual(@as(u32, 0), s1);
    sched.release(s1);

    // Same slot should be reusable
    const s2 = try sched.submit(&.{20}, .{});
    try std.testing.expectEqual(@as(u32, 0), s2);
    sched.release(s2);
}

test "Scheduler request IDs increment" {
    const allocator = std.testing.allocator;
    var sched = try Scheduler.init(allocator, 4);
    defer sched.deinit();

    _ = try sched.submit(&.{1}, .{});
    _ = try sched.submit(&.{2}, .{});

    // Request IDs should be 0 and 1 (or some incrementing sequence)
    // Check slots have different request objects
    try std.testing.expect(sched.slots[0] != null);
    try std.testing.expect(sched.slots[1] != null);
    try std.testing.expect(sched.slots[0].?.id != sched.slots[1].?.id);
}
