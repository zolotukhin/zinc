//! Resource replacement for accelerator model switching.
//! @section API Server
//!
//! The caller holds an exclusive lifecycle lease. The previous model is freed
//! before loading its replacement, since two model allocations may not fit.
const std = @import("std");

/// Context supplies `current`, `unload()`, and `load(entry)`. Resource entries
/// remain owned by the immutable model inventory throughout replacement.
/// Optional `validate(entry)` rejects metadata or budgets before unloading.
pub fn activate(context: anytype, entry: anytype) !void {
    if (context.current) |current| {
        if (std.mem.eql(u8, current.entry.id, entry.id)) return;
    }
    // Reject bad metadata/budgets while the current model is still resident.
    if (@hasDecl(@TypeOf(context.*), "validate")) try context.validate(entry);
    const previous = if (context.current) |current| current.entry else null;
    context.unload();
    context.current = context.load(entry) catch |err| {
        if (previous) |old| context.current = context.load(old) catch null;
        return err;
    };
}

const FakeEntry = struct { id: []const u8 };
const FakeResource = struct { entry: *const FakeEntry };
const FakeManager = struct {
    current: ?*FakeResource = null,
    value: FakeResource = undefined,
    fail_new: bool = false,
    fail_restore: bool = false,
    calls: [8]u8 = @splat(0),
    n: usize = 0,

    pub fn load(self: *@This(), entry: *const FakeEntry) !*FakeResource {
        self.calls[self.n] = entry.id[0];
        self.n += 1;
        if (entry.id[0] == 'b' and self.fail_new) return error.NewModelFailed;
        if (entry.id[0] == 'a' and self.fail_restore) return error.RestoreFailed;
        self.value = .{ .entry = entry };
        return &self.value;
    }

    pub fn unload(self: *@This()) void {
        if (self.current != null) {
            self.calls[self.n] = '-';
            self.n += 1;
        }
        self.current = null;
    }
};

test "model replacement frees old resources before loading new and leaves same model intact" {
    const a = FakeEntry{ .id = "a" };
    const b = FakeEntry{ .id = "b" };
    var manager = FakeManager{};
    try activate(&manager, &a);
    try activate(&manager, &a);
    try activate(&manager, &b);
    try std.testing.expectEqualStrings("a-b", manager.calls[0..manager.n]);
    try std.testing.expectEqualStrings("b", manager.current.?.entry.id);
}

test "model replacement restores the previous model if a new load fails" {
    const a = FakeEntry{ .id = "a" };
    const b = FakeEntry{ .id = "b" };
    var manager = FakeManager{ .fail_new = true };
    try activate(&manager, &a);
    try std.testing.expectError(error.NewModelFailed, activate(&manager, &b));
    try std.testing.expectEqualStrings("a-ba", manager.calls[0..manager.n]);
    try std.testing.expectEqualStrings("a", manager.current.?.entry.id);
}

test "model replacement stays unloaded if both replacement and restore fail" {
    const a = FakeEntry{ .id = "a" };
    const b = FakeEntry{ .id = "b" };
    var manager = FakeManager{ .fail_new = true };
    try activate(&manager, &a);
    manager.fail_restore = true;
    try std.testing.expectError(error.NewModelFailed, activate(&manager, &b));
    try std.testing.expect(manager.current == null);
}

test "model replacement can reload after explicit unload" {
    const a = FakeEntry{ .id = "a" };
    var manager = FakeManager{};
    try activate(&manager, &a);
    manager.unload();
    try activate(&manager, &a);
    try std.testing.expectEqualStrings("a-a", manager.calls[0..manager.n]);
    try std.testing.expectEqualStrings("a", manager.current.?.entry.id);
}

test "replacement validation failure preserves resident resources" {
    const Checked = struct {
        current: ?*FakeResource,
        pub fn validate(_: *@This(), _: *const FakeEntry) !void {
            return error.InvalidModel;
        }
        pub fn unload(_: *@This()) void {
            @panic("validation must precede unload");
        }
        pub fn load(_: *@This(), _: *const FakeEntry) !*FakeResource {
            @panic("validation must precede load");
        }
    };
    const a = FakeEntry{ .id = "a" };
    const b = FakeEntry{ .id = "b" };
    var resident = FakeResource{ .entry = &a };
    var manager = Checked{ .current = &resident };
    try std.testing.expectError(error.InvalidModel, activate(&manager, &b));
    try std.testing.expect(manager.current == &resident);
}
