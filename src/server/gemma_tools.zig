//! Gemma 4's native function-call grammar. Strings use raw <|"|> delimiters;
//! JSON escaping, braces and newlines inside them are ordinary string bytes.
const std = @import("std");

pub const call_open = "<|tool_call>";
pub const call_close = "<tool_call|>";
pub const string_delimiter = "<|\"|>";

pub const Call = struct { name: []const u8, arguments: std.json.Value };

/// Hide only an initial thought channel. Markers inside native string
/// arguments are source data and must never be rewritten or trimmed.
pub fn visibleText(text: []const u8) []const u8 {
    const prefix = "<|channel>thought\n";
    var visible = std.mem.trimStart(u8, text, " \t\r\n");
    if (std.mem.startsWith(u8, prefix, visible)) return "";
    if (std.mem.startsWith(u8, visible, prefix)) {
        const end = std.mem.indexOfPos(u8, visible, prefix.len, "<channel|>") orelse return "";
        visible = visible[end + "<channel|>".len ..];
    }
    return visible;
}

pub fn findClose(text: []const u8, start: usize) ?usize {
    var in_string = false;
    var i = start;
    while (i < text.len) {
        if (std.mem.startsWith(u8, text[i..], string_delimiter)) {
            in_string = !in_string;
            i += string_delimiter.len;
        } else if (!in_string and std.mem.startsWith(u8, text[i..], call_close)) {
            return i;
        } else i += 1;
    }
    return null;
}

const Parser = struct {
    text: []const u8,
    pos: usize = 0,
    allocator: std.mem.Allocator,

    fn whitespace(self: *Parser) void {
        while (self.pos < self.text.len and std.ascii.isWhitespace(self.text[self.pos])) : (self.pos += 1) {}
    }

    fn take(self: *Parser, byte: u8) bool {
        self.whitespace();
        if (self.pos == self.text.len or self.text[self.pos] != byte) return false;
        self.pos += 1;
        return true;
    }

    fn string(self: *Parser) ![]const u8 {
        if (!std.mem.startsWith(u8, self.text[self.pos..], string_delimiter)) return error.InvalidToolCall;
        self.pos += string_delimiter.len;
        const end = std.mem.indexOfPos(u8, self.text, self.pos, string_delimiter) orelse return error.InvalidToolCall;
        const result = self.text[self.pos..end];
        if (!std.unicode.utf8ValidateSlice(result)) return error.InvalidToolCall;
        self.pos = end + string_delimiter.len;
        return result;
    }

    fn key(self: *Parser) ![]const u8 {
        self.whitespace();
        if (std.mem.startsWith(u8, self.text[self.pos..], string_delimiter)) return self.string();
        const start = self.pos;
        while (self.pos < self.text.len and std.mem.indexOfScalar(u8, ":,{}[] \t\r\n<>", self.text[self.pos]) == null) : (self.pos += 1) {}
        if (self.pos == start) return error.InvalidToolCall;
        return self.text[start..self.pos];
    }

    fn value(self: *Parser, depth: u8) anyerror!std.json.Value {
        if (depth == 0) return error.InvalidToolCall;
        self.whitespace();
        if (self.pos == self.text.len) return error.InvalidToolCall;
        if (std.mem.startsWith(u8, self.text[self.pos..], string_delimiter)) return .{ .string = try self.string() };
        if (self.take('{')) {
            var object: std.json.ObjectMap = .empty;
            if (self.take('}')) return .{ .object = object };
            while (true) {
                const name = try self.key();
                if (!self.take(':') or object.contains(name)) return error.InvalidToolCall;
                const item = try self.value(depth - 1);
                try object.put(self.allocator, name, item);
                if (self.take('}')) break;
                if (!self.take(',')) return error.InvalidToolCall;
            }
            return .{ .object = object };
        }
        if (self.take('[')) {
            var array: std.array_list.Managed(std.json.Value) = .init(self.allocator);
            if (self.take(']')) return .{ .array = array };
            while (true) {
                try array.append(try self.value(depth - 1));
                if (self.take(']')) break;
                if (!self.take(',')) return error.InvalidToolCall;
            }
            return .{ .array = array };
        }
        const start = self.pos;
        while (self.pos < self.text.len and std.mem.indexOfScalar(u8, ",]} \t\r\n", self.text[self.pos]) == null) : (self.pos += 1) {}
        if (self.pos == start) return error.InvalidToolCall;
        const scalar = std.json.parseFromSliceLeaky(std.json.Value, self.allocator, self.text[start..self.pos], .{}) catch |err| return if (err == error.OutOfMemory) err else error.InvalidToolCall;
        return switch (scalar) {
            .null, .bool, .integer, .float, .number_string => scalar,
            else => error.InvalidToolCall,
        };
    }
};

/// Allocations belong to the supplied arena. Incomplete/invalid calls are never
/// repaired into executable requests. The top-level arguments must be an object.
pub fn parse(inner: []const u8, allocator: std.mem.Allocator) !?Call {
    const input = std.mem.trim(u8, inner, " \t\r\n");
    if (!std.mem.startsWith(u8, input, "call:")) return null;
    const brace = std.mem.indexOfScalar(u8, input, '{') orelse return null;
    const name = std.mem.trim(u8, input["call:".len..brace], " \t\r\n");
    if (!validName(name)) return null;
    var parser = Parser{ .text = input, .pos = brace, .allocator = allocator };
    const arguments = parser.value(64) catch |err| return if (err == error.InvalidToolCall) null else err;
    parser.whitespace();
    if (parser.pos != input.len or arguments != .object) return null;
    return .{ .name = name, .arguments = arguments };
}

fn validName(name: []const u8) bool {
    if (name.len == 0) return false;
    for (name) |c| if (!std.ascii.isAlphanumeric(c) and c != '_' and c != '-') return false;
    return true;
}

fn appendString(buf: *std.ArrayList(u8), a: std.mem.Allocator, value: []const u8) !void {
    // The native grammar has no escape for its own raw-string delimiter.
    if (std.mem.indexOf(u8, value, string_delimiter) != null) return error.InvalidToolArguments;
    try buf.appendSlice(a, string_delimiter);
    try buf.appendSlice(a, value);
    try buf.appendSlice(a, string_delimiter);
}

fn lessThan(_: void, lhs: []const u8, rhs: []const u8) bool {
    return std.mem.lessThan(u8, lhs, rhs);
}

fn sortedKeys(object: std.json.ObjectMap, a: std.mem.Allocator) ![][]const u8 {
    const keys = try a.dupe([]const u8, object.keys());
    std.mem.sort([]const u8, keys, {}, lessThan);
    return keys;
}

pub fn appendValue(buf: *std.ArrayList(u8), a: std.mem.Allocator, value: std.json.Value, depth: u8) anyerror!void {
    if (depth == 0) return error.InvalidToolArguments;
    switch (value) {
        .string => try appendString(buf, a, value.string),
        .object => |object| {
            try buf.append(a, '{');
            const keys = try sortedKeys(object, a);
            defer a.free(keys);
            for (keys, 0..) |key, i| {
                if (i > 0) try buf.append(a, ',');
                // Native call/result keys are bare, including nested objects.
                if (key.len == 0 or std.mem.indexOfAny(u8, key, ":,{}[] \t\r\n<>") != null) return error.InvalidToolArguments;
                try buf.print(a, "{s}:", .{key});
                try appendValue(buf, a, object.get(key).?, depth - 1);
            }
            try buf.append(a, '}');
        },
        .array => |array| {
            try buf.append(a, '[');
            for (array.items, 0..) |item, i| {
                if (i > 0) try buf.append(a, ',');
                try appendValue(buf, a, item, depth - 1);
            }
            try buf.append(a, ']');
        },
        else => {
            const json = try std.json.Stringify.valueAlloc(a, value, .{});
            defer a.free(json);
            try buf.appendSlice(a, json);
        },
    }
}

fn appendType(buf: *std.ArrayList(u8), a: std.mem.Allocator, kind: std.json.Value) !void {
    if (kind == .array) {
        try buf.append(a, '[');
        for (kind.array.items, 0..) |item, i| {
            if (i > 0) try buf.append(a, ',');
            try appendType(buf, a, item);
        }
        try buf.append(a, ']');
    } else if (kind == .string) {
        const upper = try a.dupe(u8, kind.string);
        defer a.free(upper);
        for (upper) |*c| c.* = std.ascii.toUpper(c.*);
        try appendString(buf, a, upper);
    } else return error.InvalidToolArguments;
}

fn appendSchema(buf: *std.ArrayList(u8), a: std.mem.Allocator, schema: std.json.Value, depth: u8, root: bool) anyerror!void {
    if (schema != .object or depth == 0) return error.InvalidToolArguments;
    try buf.append(a, '{');
    var first = true;
    const fields: []const []const u8 = if (root) &.{ "properties", "required", "type" } else &.{ "description", "enum", "items", "nullable", "properties", "required", "type" };
    for (fields) |field| {
        const item = schema.object.get(field) orelse continue;
        if (item == .null) continue;
        if ((item == .array and item.array.items.len == 0) or (item == .object and item.object.count() == 0)) continue;
        if (!first) try buf.append(a, ',');
        first = false;
        try buf.print(a, "{s}:", .{field});
        if (std.mem.eql(u8, field, "type")) {
            try appendType(buf, a, item);
        } else if (std.mem.eql(u8, field, "properties") and item == .object) {
            try buf.append(a, '{');
            const keys = try sortedKeys(item.object, a);
            defer a.free(keys);
            for (keys, 0..) |key, i| {
                if (i > 0) try buf.append(a, ',');
                if (!validName(key)) return error.InvalidToolArguments;
                try buf.print(a, "{s}:", .{key});
                try appendSchema(buf, a, item.object.get(key).?, depth - 1, false);
            }
            try buf.append(a, '}');
        } else if (std.mem.eql(u8, field, "items")) {
            try appendSchema(buf, a, item, depth - 1, false);
        } else try appendValue(buf, a, item, depth - 1);
    }
    try buf.append(a, '}');
}

pub fn appendDefinition(buf: *std.ArrayList(u8), a: std.mem.Allocator, name: []const u8, description: []const u8, parameters: []const u8) !void {
    if (!validName(name)) return error.InvalidToolArguments;
    const schema = try std.json.parseFromSlice(std.json.Value, a, parameters, .{});
    defer schema.deinit();
    try buf.print(a, "<|tool>declaration:{s}{{description:", .{name});
    try appendString(buf, a, description);
    try buf.appendSlice(a, ",parameters:");
    try appendSchema(buf, a, schema.value, 64, true);
    try buf.appendSlice(a, "}<tool|>");
}

pub fn appendCall(buf: *std.ArrayList(u8), a: std.mem.Allocator, name: []const u8, arguments: []const u8) !void {
    if (!validName(name)) return error.InvalidToolArguments;
    const parsed = try std.json.parseFromSlice(std.json.Value, a, arguments, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidToolArguments;
    try buf.print(a, "{s}call:{s}", .{ call_open, name });
    try appendValue(buf, a, parsed.value, 64);
    try buf.appendSlice(a, call_close);
}

pub fn appendResult(buf: *std.ArrayList(u8), a: std.mem.Allocator, name: []const u8, content: []const u8) !void {
    if (!validName(name)) return error.InvalidToolArguments;
    try buf.print(a, "<|tool_response>response:{s}{{value:", .{name});
    // OpenAI tool content is text, even when its bytes happen to look like JSON.
    try appendString(buf, a, content);
    try buf.appendSlice(a, "}<tool_response|>");
}

test "Gemma native calls round-trip nested types and literal source bytes" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const json = "{\"path\":\"café/東京.ts\",\"code\":\"const x = {a: \\\"<think>\\\"};\\n// <tool_call|>\\n\",\"options\":{\"ok\":true,\"n\":-1.5,\"missing\":null,\"items\":[1,\"false\"]}}";
    var buf: std.ArrayList(u8) = .empty;
    try appendCall(&buf, a, "write", json);
    const end = findClose(buf.items, call_open.len).?;
    try std.testing.expectEqual(buf.items.len - call_close.len, end);
    const call = (try parse(buf.items[call_open.len..end], a)).?;
    const expected = try std.json.parseFromSliceLeaky(std.json.Value, a, json, .{});
    try std.testing.expectEqualStrings("write", call.name);
    try std.testing.expectEqualStrings(expected.object.get("code").?.string, call.arguments.object.get("code").?.string);
    try std.testing.expectEqualStrings("café/東京.ts", call.arguments.object.get("path").?.string);
    const options = call.arguments.object.get("options").?.object;
    try std.testing.expect(options.get("ok").?.bool);
    try std.testing.expectEqual(@as(f64, -1.5), options.get("n").?.float);
    try std.testing.expect(options.get("missing").? == .null);
    try std.testing.expectEqualStrings("false", options.get("items").?.array.items[1].string);
}

test "Gemma malformed calls remain non-executable" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    for ([_][]const u8{
        "call:x{a:<|\"|>unfinished}", "call:x{a:1,a:2}",      "call:x{a:unquoted}",
        "call:x{a:[1,]}",             "call:x{a:1} trailing", "call:x[]",
        "call:bad name{}",
    }) |input| try std.testing.expect(try parse(input, arena.allocator()) == null);
}

test "Gemma definitions match the native schema scaffold" {
    const a = std.testing.allocator;
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(a);
    try appendDefinition(&buf, a, "read", "Read a file", "{\"type\":\"object\",\"required\":[\"path\"],\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"File path\"}}}");
    try std.testing.expectEqualStrings("<|tool>declaration:read{description:<|\"|>Read a file<|\"|>,parameters:{properties:{path:{description:<|\"|>File path<|\"|>,type:<|\"|>STRING<|\"|>}},required:[<|\"|>path<|\"|>],type:<|\"|>OBJECT<|\"|>}}<tool|>", buf.items);
}
