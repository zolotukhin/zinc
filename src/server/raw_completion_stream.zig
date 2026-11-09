//! Raw completions use text chunks, without chat templates or stop-tag cleanup.
//! Keep generation incremental so a disconnected reader releases the GPU.
const std = @import("std");

pub const Piece = struct {
    bytes: []const u8,
    eog: bool = false,
};

pub const Options = struct {
    id: []const u8,
    created: i64,
    model: []const u8,
    prompt_tokens: usize,
    requested_max_tokens: u32,
    max_tokens: u32,
    include_usage: bool,
};

/// Retain a split UTF-8 character until another token completes it. At the
/// output budget, the unfinished tail is omitted, like non-streaming JSON.
pub fn completeUtf8Prefix(text: []const u8) []const u8 {
    if (text.len == 0) return text;
    var start = text.len - 1;
    while (start > 0 and text[start] & 0xc0 == 0x80) : (start -= 1) {}
    const length = std.unicode.utf8ByteSequenceLength(text[start]) catch return text;
    return if (text.len - start < length) text[0..start] else text;
}

fn chunk(conn: anytype, options: Options, text: []const u8, finish_reason: ?[]const u8) !void {
    var buffer: [16384]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try std.json.Stringify.value(.{
        .id = options.id,
        .object = "text_completion",
        .created = options.created,
        .model = options.model,
        .choices = .{.{ .index = @as(u32, 0), .text = text, .finish_reason = finish_reason }},
    }, .{}, &writer);
    try conn.writeSseEvent(writer.buffered());
}

/// `source.next` samples exactly one token, including EOG. Do not precompute
/// the response: first content and cancellation must precede the final token.
pub fn run(conn: anytype, source: anytype, options: Options) !usize {
    var pending: [1024]u8 = undefined;
    var pending_len: usize = 0;
    var generated: usize = 0;
    var ended_on_eog = false;
    while (generated < options.max_tokens) {
        if (conn.isPeerClosed()) return error.RequestCancelled;
        const piece: Piece = try source.next();
        generated += 1;
        if (piece.eog) {
            ended_on_eog = true;
            break;
        }
        if (piece.bytes.len > pending.len - pending_len) return error.TokenTooLarge;
        @memcpy(pending[pending_len..][0..piece.bytes.len], piece.bytes);
        pending_len += piece.bytes.len;
        const complete = completeUtf8Prefix(pending[0..pending_len]);
        if (!std.unicode.utf8ValidateSlice(complete)) return error.InvalidUtf8;
        if (complete.len > 0) try chunk(conn, options, complete, null);
        const remaining = pending_len - complete.len;
        std.mem.copyForwards(u8, pending[0..remaining], pending[complete.len..pending_len]);
        pending_len = remaining;
    }
    if (conn.isPeerClosed()) return error.RequestCancelled;
    const reason: []const u8 = if (!ended_on_eog and options.requested_max_tokens > 0 and generated >= options.max_tokens) "length" else "stop";
    try chunk(conn, options, "", reason);
    if (options.include_usage) {
        var buffer: [16384]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buffer);
        try std.json.Stringify.value(.{
            .id = options.id,
            .object = "text_completion",
            .created = options.created,
            .model = options.model,
            .choices = @as([0]struct {}, .{}),
            .usage = .{ .prompt_tokens = options.prompt_tokens, .completion_tokens = generated, .total_tokens = options.prompt_tokens + generated },
        }, .{}, &writer);
        try conn.writeSseEvent(writer.buffered());
    }
    try conn.writeSseDone();
    return generated;
}

const FakeSource = struct {
    pieces: []const Piece,
    calls: usize = 0,
    fail_at: ?usize = null,
    pub fn next(self: *@This()) !Piece {
        if (self.fail_at == self.calls) return error.DecodeFailed;
        if (self.calls >= self.pieces.len) return error.UnexpectedRead;
        const piece = self.pieces[self.calls];
        self.calls += 1;
        return piece;
    }
};

const FakeConnection = struct {
    events: std.ArrayList([]u8) = .empty,
    done: bool = false,
    closed: bool = false,
    fail_at: ?usize = null,
    pub fn writeSseEvent(self: *@This(), event: []const u8) !void {
        if (self.fail_at == self.events.items.len) return error.BrokenPipe;
        const copy = try std.testing.allocator.dupe(u8, event);
        errdefer std.testing.allocator.free(copy);
        try self.events.append(std.testing.allocator, copy);
    }
    pub fn isPeerClosed(self: *@This()) bool {
        return self.closed;
    }
    pub fn writeSseDone(self: *@This()) !void {
        self.done = true;
    }
    fn deinit(self: *@This()) void {
        for (self.events.items) |event| std.testing.allocator.free(event);
        self.events.deinit(std.testing.allocator);
    }
};

fn testOptions(limit: u32) Options {
    return .{ .id = "cmpl-test", .created = 123, .model = "quoted\"model\n東京", .prompt_tokens = 42, .requested_max_tokens = limit, .max_tokens = limit, .include_usage = true };
}

test "raw SSE keeps split Unicode, metadata and exact usage at every token budget" {
    const pieces = [_]Piece{ .{ .bytes = "caf" }, .{ .bytes = "\xc3" }, .{ .bytes = "\xa9 \xf0" }, .{ .bytes = "\x9f\x9a" }, .{ .bytes = "\x80\n\"\\" } };
    var all: std.ArrayList(u8) = .empty;
    defer all.deinit(std.testing.allocator);
    for (pieces, 1..) |piece, limit| {
        try all.appendSlice(std.testing.allocator, piece.bytes);
        var source = FakeSource{ .pieces = &pieces };
        var conn = FakeConnection{};
        defer conn.deinit();
        try std.testing.expectEqual(limit, try run(&conn, &source, testOptions(@intCast(limit))));
        try std.testing.expect(conn.done);
        var text: std.ArrayList(u8) = .empty;
        defer text.deinit(std.testing.allocator);
        for (conn.events.items, 0..) |event, index| {
            const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, event, .{});
            defer parsed.deinit();
            const object = parsed.value.object;
            try std.testing.expectEqualStrings("text_completion", object.get("object").?.string);
            try std.testing.expectEqualStrings(testOptions(1).model, object.get("model").?.string);
            const choices = object.get("choices").?.array.items;
            if (index == conn.events.items.len - 1) {
                try std.testing.expectEqual(@as(usize, 0), choices.len);
                const usage = object.get("usage").?.object;
                try std.testing.expectEqual(@as(i64, @intCast(limit)), usage.get("completion_tokens").?.integer);
                try std.testing.expectEqual(@as(i64, @intCast(42 + limit)), usage.get("total_tokens").?.integer);
            } else {
                const choice = choices[0].object;
                try std.testing.expect(choice.get("delta") == null);
                try text.appendSlice(std.testing.allocator, choice.get("text").?.string);
                if (index == conn.events.items.len - 2) try std.testing.expectEqualStrings("length", choice.get("finish_reason").?.string);
            }
        }
        try std.testing.expectEqualStrings(completeUtf8Prefix(all.items), text.items);
    }
}

test "raw SSE counts EOG without exposing it and supports empty or exhausted budgets" {
    const pieces = [_]Piece{ .{ .bytes = "answer" }, .{ .bytes = "<eog>", .eog = true } };
    for ([_]u32{ 0, 1, 2, 3 }) |limit| {
        var source = FakeSource{ .pieces = &pieces };
        var conn = FakeConnection{};
        defer conn.deinit();
        var options = testOptions(limit);
        options.include_usage = false;
        const count = try run(&conn, &source, options);
        try std.testing.expectEqual(@min(limit, 2), count);
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, conn.events.items[conn.events.items.len - 1], .{});
        defer parsed.deinit();
        const reason = parsed.value.object.get("choices").?.array.items[0].object.get("finish_reason").?.string;
        try std.testing.expectEqualStrings(if (limit == 1) "length" else "stop", reason);
        for (conn.events.items) |event| try std.testing.expect(std.mem.indexOf(u8, event, "<eog>") == null and std.mem.indexOf(u8, event, "\"usage\"") == null);
    }
    var source = FakeSource{ .pieces = &pieces };
    var conn = FakeConnection{};
    defer conn.deinit();
    var options = testOptions(0);
    options.requested_max_tokens = 16;
    try std.testing.expectEqual(@as(usize, 0), try run(&conn, &source, options));
    try std.testing.expect(std.mem.indexOf(u8, conn.events.items[0], "\"finish_reason\":\"length\"") != null);
}

test "raw streaming stops sampling on disconnect or write failure and never completes a failed decode" {
    const pieces = [_]Piece{ .{ .bytes = "first" }, .{ .bytes = "second" } };
    var source = FakeSource{ .pieces = &pieces };
    var conn = FakeConnection{ .closed = true };
    defer conn.deinit();
    try std.testing.expectError(error.RequestCancelled, run(&conn, &source, testOptions(2)));
    try std.testing.expectEqual(@as(usize, 0), source.calls);
    conn.closed = false;
    conn.fail_at = 0;
    try std.testing.expectError(error.BrokenPipe, run(&conn, &source, testOptions(2)));
    try std.testing.expectEqual(@as(usize, 1), source.calls);
    conn.fail_at = null;
    source.calls = 0;
    source.fail_at = 1;
    try std.testing.expectError(error.DecodeFailed, run(&conn, &source, testOptions(2)));
    try std.testing.expectEqual(@as(usize, 1), conn.events.items.len);
    try std.testing.expect(!conn.done);
    var invalid = FakeSource{ .pieces = &.{.{ .bytes = "\xff" }} };
    try std.testing.expectError(error.InvalidUtf8, run(&conn, &invalid, testOptions(1)));
}
