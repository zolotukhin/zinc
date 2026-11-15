//! Structured diagnostics shared by every GPU backend. Human output and JSON
//! record the same checks; callers never need to scrape terminal formatting.
//! @section Hardware Detection
const std = @import("std");
const build_info = @import("build_info.zig");

pub const Status = enum {
    ok,
    warn,
    fail,
    skip,
    pub fn label(self: Status) []const u8 {
        return switch (self) {
            .ok => "OK",
            .warn => "WARN",
            .fail => "FAIL",
            .skip => "SKIP",
        };
    }
};
pub const Result = enum { ready, ready_with_warnings, not_ready };
pub const Check = struct { name: []const u8, status: Status, message: []const u8 };
pub const Memory = struct {
    source: enum { estimate, catalog },
    weights_bytes: u64,
    runtime_bytes: u64,
    kv_bytes: u64,
    total_bytes: u64,
    budget_bytes: u64,
    context_length: u32,
    slots: u32 = 1,
    recurrent_bytes: ?u64 = null,
    runtime_reserve_bytes: ?u64 = null,
};

pub const Report = struct {
    allocator: std.mem.Allocator,
    checks: std.ArrayList(Check) = .empty,
    device_name: ?[]u8 = null,
    memory_budget_bytes: ?u64 = null,
    model: ?[]u8 = null,
    memory: ?Memory = null,
    fatal_error: ?[]const u8 = null,
    ok: u32 = 0,
    warn: u32 = 0,
    fail: u32 = 0,
    skip: u32 = 0,

    pub fn init(allocator: std.mem.Allocator) Report {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *Report) void {
        if (self.device_name) |name| self.allocator.free(name);
        if (self.model) |model| self.allocator.free(model);
        for (self.checks.items) |check| {
            self.allocator.free(check.name);
            self.allocator.free(check.message);
        }
        self.checks.deinit(self.allocator);
    }

    pub fn setModel(self: *Report, id: []const u8) !void {
        if (!std.unicode.utf8ValidateSlice(id)) return error.InvalidUtf8;
        const copy = try self.allocator.dupe(u8, id);
        if (self.model) |old| self.allocator.free(old);
        self.model = copy;
    }

    /// Error names have static storage: even an exhausted allocator can emit
    /// a valid failure document through the streaming JSON writer.
    pub fn markFatal(self: *Report, err: anyerror) void {
        self.fatal_error = @errorName(err);
        if (self.fail == 0) self.fail = 1;
    }

    pub fn setDevice(self: *Report, name: []const u8, budget: u64) !void {
        const copy = try self.allocator.dupe(u8, name);
        if (self.device_name) |old| self.allocator.free(old);
        self.device_name = copy;
        self.memory_budget_bytes = budget;
    }

    pub fn add(self: *Report, name: []const u8, status: Status, comptime fmt: []const u8, args: anytype) !void {
        const owned_name = try self.allocator.dupe(u8, name);
        errdefer self.allocator.free(owned_name);
        const message = try std.fmt.allocPrint(self.allocator, fmt, args);
        errdefer self.allocator.free(message);
        if (!std.unicode.utf8ValidateSlice(message)) return error.InvalidUtf8;
        try self.checks.append(self.allocator, .{ .name = owned_name, .status = status, .message = message });
        switch (status) {
            .ok => self.ok += 1,
            .warn => self.warn += 1,
            .fail => self.fail += 1,
            .skip => self.skip += 1,
        }
    }

    pub fn result(self: Report) Result {
        if (self.fail > 0 or self.checks.items.len == 0) return .not_ready;
        return if (self.warn > 0) .ready_with_warnings else .ready;
    }

    pub fn writeJson(self: Report, writer: *std.Io.Writer) !void {
        try std.json.Stringify.value(.{
            .schema_version = 1,
            .build = build_info.Metadata{},
            .backend = build_info.backend,
            .result = self.result(),
            .@"error" = self.fatal_error,
            .device = .{ .name = self.device_name, .memory_budget_bytes = self.memory_budget_bytes },
            .model = self.model,
            .memory = self.memory,
            .summary = .{ .ok = self.ok, .warn = self.warn, .fail = self.fail, .skip = self.skip },
            .checks = self.checks.items,
        }, .{}, writer);
        try writer.writeByte('\n');
    }
};

test "diagnostic verdict and JSON stay consistent on warnings and failure" {
    var report = Report.init(std.testing.allocator);
    defer report.deinit();
    try std.testing.expectEqual(Result.not_ready, report.result());
    try report.add("device", .ok, "Found {s}", .{"GPU"});
    try std.testing.expectEqual(Result.ready, report.result());
    try report.add("model fit", .warn, "Estimate only: {d} bytes", .{123});
    try std.testing.expectEqual(Result.ready_with_warnings, report.result());
    try report.add("model", .fail, "Invalid name: {s}", .{"quoted\"\\\n"});
    try std.testing.expectEqual(Result.not_ready, report.result());
    var buffer: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try report.writeJson(&writer);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, writer.buffered(), .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("not_ready", parsed.value.object.get("result").?.string);
    try std.testing.expectEqual(@as(usize, 3), parsed.value.object.get("checks").?.array.items.len);
    try std.testing.expectEqual(@as(i64, 1), parsed.value.object.get("summary").?.object.get("fail").?.integer);
    try std.testing.expectEqualStrings("Invalid name: quoted\"\\\n", parsed.value.object.get("checks").?.array.items[2].object.get("message").?.string);
}

test "allocation failures still produce a valid failure report without allocating" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    var report = Report.init(failing.allocator());
    defer report.deinit();
    try std.testing.expectError(error.OutOfMemory, report.setModel("model"));
    try std.testing.expectError(error.OutOfMemory, report.add("device", .ok, "Found GPU", .{}));
    report.markFatal(error.OutOfMemory);
    var buffer: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try report.writeJson(&writer);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, writer.buffered(), .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("not_ready", parsed.value.object.get("result").?.string);
    try std.testing.expectEqualStrings("OutOfMemory", parsed.value.object.get("error").?.string);
}
