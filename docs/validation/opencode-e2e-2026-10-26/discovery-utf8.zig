const std = @import("std");
const discovery = @import("discovery");
test "invalid UTF8 basename is a recoverable error" {
    const invalid = [_][]const u8{
        "\xff.gguf", "bad\xc0\xaf.gguf", "bad\xe2\x82.gguf",
        "bad\xed\xa0\x80.gguf", "bad\xf4\x90\x80\x80.gguf", "bad\x80.gguf",
    };
    for (invalid) |name| {
        const raw_path = try std.fmt.allocPrint(std.testing.allocator, "/fixtures/parent/{s}", .{name});
        defer std.testing.allocator.free(raw_path);
        const result = discovery.serializeModels(std.testing.allocator, raw_path, 262144);
        if (result) |bytes| {
            defer std.testing.allocator.free(bytes);
            return error.InvalidBasenameWasSerialized;
        } else |err| {
            try std.testing.expectEqual(error.InvalidUtf8, err);
        }
    }
}
test "valid basename stays a string and parent bytes remain private" {
    const names = [_][]const u8{"model.gguf", "测试模型.gguf", "quoted\"\\\n\t.gguf", "max-\xf4\x8f\xbf\xbf.gguf"};
    for (names) |name| {
        const raw_path = try std.fmt.allocPrint(std.testing.allocator, "/fixtures/\xff-private/{s}", .{name});
        defer std.testing.allocator.free(raw_path);
        const bytes = try discovery.serializeModels(std.testing.allocator, raw_path, 32768);
        defer std.testing.allocator.free(bytes);
        try std.testing.expect(std.unicode.utf8ValidateSlice(bytes));
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, bytes, .{});
        defer parsed.deinit();
        try std.testing.expectEqualStrings("list", parsed.value.object.get("object").?.string);
        const model = parsed.value.object.get("data").?.array.items[0].object;
        try std.testing.expectEqualStrings(name, model.get("id").?.string);
        try std.testing.expectEqualStrings("zinc", model.get("owned_by").?.string);
        try std.testing.expectEqual(@as(i64, 0), model.get("created").?.integer);
        try std.testing.expectEqual(@as(i64, 32768), model.get("context_length").?.integer);
        try std.testing.expect(std.mem.indexOf(u8, bytes, "private") == null);
    }
}
test "invalid basename fails before allocation and valid names preserve OOM" {
    var failing = std.testing.FailingAllocator.init(std.testing.allocator, .{ .fail_index = 0 });
    try std.testing.expectError(error.InvalidUtf8, discovery.serializeModels(failing.allocator(), "bad\xff.gguf", 32768));
    try std.testing.expectError(error.OutOfMemory, discovery.serializeModels(failing.allocator(), "valid.gguf", 32768));
}
