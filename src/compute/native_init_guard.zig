//! Own partial native forward allocations until initialization succeeds.
//! Model replacement must release a failed constructor before restoring the
//! previous model. A fallible struct literal alone cannot provide that cleanup.
const std = @import("std");

pub fn Guard(comptime Ops: type) type {
    return struct {
        const Self = @This();
        const Resource = union(enum) {
            buffer: Ops.Buffer,
            pipeline: Ops.Pipeline,
            host: struct { bytes: []u8, alignment: std.mem.Alignment },
        };
        allocator: std.mem.Allocator,
        ctx: Ops.Context,
        resources: std.ArrayList(Resource) = .empty,
        committed: bool = false,

        pub fn init(allocator: std.mem.Allocator, ctx: Ops.Context) Self {
            return .{ .allocator = allocator, .ctx = ctx };
        }

        /// Successful forwards retain their existing individual ownership.
        pub fn commit(self: *Self) void {
            self.committed = true;
        }

        pub fn deinit(self: *Self) void {
            if (!self.committed) {
                var i = self.resources.items.len;
                while (i > 0) {
                    i -= 1;
                    switch (self.resources.items[i]) {
                        .buffer => |value| {
                            var owned = value;
                            Ops.freeBuffer(&owned);
                        },
                        .pipeline => |value| {
                            var owned = value;
                            Ops.freePipeline(&owned);
                        },
                        .host => |memory| self.allocator.rawFree(memory.bytes, memory.alignment, @returnAddress()),
                    }
                }
            }
            self.resources.deinit(self.allocator);
        }

        pub fn createBuffer(self: *Self, size: usize) !Ops.Buffer {
            var value = try Ops.createBuffer(self.ctx, size);
            errdefer Ops.freeBuffer(&value);
            try self.resources.append(self.allocator, .{ .buffer = value });
            return value;
        }

        pub fn createPipeline(self: *Self, source: [*:0]const u8, name: [*:0]const u8) !Ops.Pipeline {
            var value = try Ops.createPipeline(self.ctx, source, name);
            errdefer Ops.freePipeline(&value);
            try self.resources.append(self.allocator, .{ .pipeline = value });
            return value;
        }

        pub fn createPipelineFromImage(self: *Self, source: [*]const u8, size: usize, name: [*:0]const u8) !Ops.Pipeline {
            var value = try Ops.createPipelineFromImage(self.ctx, source, size, name);
            errdefer Ops.freePipeline(&value);
            try self.resources.append(self.allocator, .{ .pipeline = value });
            return value;
        }

        /// Track only allocations transferred into the returned forward.
        /// Temporaries keep their normal local defer and must not use this.
        pub fn alloc(self: *Self, comptime T: type, count: usize) ![]T {
            const value = try self.allocator.alloc(T, count);
            errdefer self.allocator.free(value);
            if (@sizeOf(T) != 0 and count != 0) {
                try self.resources.append(self.allocator, .{ .host = .{
                    .bytes = std.mem.sliceAsBytes(value),
                    .alignment = .of(T),
                } });
            }
            return value;
        }
    };
}

pub const Native = Guard(struct {
    const buffers = @import("../cuda/buffer.zig");
    const pipelines = @import("../cuda/pipeline.zig");
    pub const Context = ?*@import("../cuda/c.zig").shim.CudaCtx;
    pub const Buffer = buffers.CudaBuffer;
    pub const Pipeline = pipelines.CudaPipeline;
    pub const createBuffer = buffers.createBuffer;
    pub const freeBuffer = buffers.freeBuffer;
    pub const createPipeline = pipelines.createPipeline;
    pub const createPipelineFromImage = pipelines.createPipelineFromImage;
    pub const freePipeline = pipelines.freePipeline;
});

const FakeState = struct {
    created: usize = 0,
    live: usize = 0,
    fail_at: usize = std.math.maxInt(usize),
    freed: [8]usize = @splat(0),
    freed_count: usize = 0,
};
const FakeOps = struct {
    pub const Context = *FakeState;
    pub const Buffer = struct { state: *FakeState, id: usize };
    pub const Pipeline = Buffer;
    pub fn createBuffer(state: Context, _: usize) !Buffer {
        if (state.created == state.fail_at) return error.InjectedNativeFailure;
        state.created += 1;
        state.live += 1;
        return .{ .state = state, .id = state.created };
    }
    pub fn createPipeline(state: Context, _: [*:0]const u8, _: [*:0]const u8) !Pipeline {
        return createBuffer(state, 1);
    }
    pub fn createPipelineFromImage(state: Context, _: [*]const u8, _: usize, _: [*:0]const u8) !Pipeline {
        return createBuffer(state, 1);
    }
    pub fn freeBuffer(value: *Buffer) void {
        std.debug.assert(value.id != 0);
        value.state.live -= 1;
        value.state.freed[value.state.freed_count] = value.id;
        value.state.freed_count += 1;
        value.id = 0;
    }
    pub const freePipeline = freeBuffer;
};

fn failedConstructor(allocator: std.mem.Allocator, state: *FakeState) !void {
    var guard = Guard(FakeOps).init(allocator, state);
    defer guard.deinit();
    _ = try guard.createPipeline("source", "first");
    _ = try guard.alloc(u128, 7);
    _ = try guard.createBuffer(1024);
    _ = try guard.alloc(u32, 5);
    _ = try guard.createPipelineFromImage("image", 5, "last");
    return error.InjectedConstructorFailure;
}

test "partial native initialization unwinds buffers pipelines and aligned host allocations" {
    for (0..4) |failure| {
        var state = FakeState{ .fail_at = failure };
        const result = failedConstructor(std.testing.allocator, &state);
        if (failure < 3) try std.testing.expectError(error.InjectedNativeFailure, result) else try std.testing.expectError(error.InjectedConstructorFailure, result);
        try std.testing.expectEqual(@as(usize, 0), state.live);
        for (state.freed[0..state.freed_count], 0..) |id, i| try std.testing.expectEqual(state.created - i, id);
    }
}

fn allocationFailure(allocator: std.mem.Allocator) !void {
    var state = FakeState{};
    defer std.debug.assert(state.live == 0);
    failedConstructor(allocator, &state) catch |err| switch (err) {
        error.InjectedConstructorFailure => {},
        else => return err,
    };
}

test "native initialization frees a handle even when recording its ownership fails" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, allocationFailure, .{});
}

test "successful native initialization transfers ownership without freeing live resources" {
    var state = FakeState{};
    var guard = Guard(FakeOps).init(std.testing.allocator, &state);
    var buf = try guard.createBuffer(128);
    var pipe = try guard.createPipeline("source", "kernel");
    const memory = try guard.alloc(u128, 3);
    guard.commit();
    guard.deinit();
    try std.testing.expectEqual(@as(usize, 2), state.live);
    FakeOps.freeBuffer(&buf);
    FakeOps.freePipeline(&pipe);
    std.testing.allocator.free(memory);
    try std.testing.expectEqual(@as(usize, 0), state.live);
}
