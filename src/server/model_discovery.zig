//! Model discovery serializer for the OpenAI-compatible /v1/models endpoint.
//! @section API Server
//! Produces an allocator-owned JSON byte slice listing the loaded model with
//! its basename as `id`, a fixed `owned_by` of "zinc", and per-slot context
//! length. Never exposes parent directories or other private metadata.
const std = @import("std");

pub const ChatModel = struct {
    display_name: ?[]const u8 = null,
    homepage_url: ?[]const u8 = null,
    supports_thinking_toggle: bool = false,
};

/// Describe the already-loaded native model to the shared browser UI. This is
/// a single active, manually loaded model, not a hot-switchable catalog entry.
pub fn serializeChatModel(allocator: std.mem.Allocator, model_path: []const u8, context_length: u32, options: ChatModel) ![]u8 {
    const model_id = std.fs.path.basename(model_path);
    if (!std.unicode.utf8ValidateSlice(model_id)) return error.InvalidUtf8;
    return std.json.Stringify.valueAlloc(allocator, .{
        .object = "list",
        .data = &.{.{
            .id = model_id,
            .object = "model",
            .created = 0,
            .owned_by = "zinc",
            .context_length = context_length,
            .display_name = options.display_name orelse model_id,
            .homepage_url = options.homepage_url,
            .active = true,
            .managed = false,
            .installed = true,
            .supports_thinking_toggle = options.supports_thinking_toggle,
        }},
    }, .{});
}

/// Model discovery serializer — produces OpenAI-compatible /v1/models JSON.
/// Returns an allocator-owned byte slice; caller must free with the same allocator.
pub fn serializeModels(allocator: std.mem.Allocator, model_path: []const u8, context_length: u32) ![]u8 {
    const model_id = std.fs.path.basename(model_path);
    if (!std.unicode.utf8ValidateSlice(model_id)) return error.InvalidUtf8;
    return std.json.Stringify.valueAlloc(allocator, .{
        .object = "list",
        .data = &.{.{
            .id = model_id,
            .object = "model",
            .created = 0,
            .owned_by = "zinc",
            .context_length = context_length,
        }},
    }, .{});
}

test "serializeModels basic" {
    const allocator = std.testing.allocator;
    const result = try serializeModels(allocator, "/path/to/model.gguf", 8192);
    defer allocator.free(result);
    const expected = "{\"object\":\"list\",\"data\":[{\"id\":\"model.gguf\",\"object\":\"model\",\"created\":0,\"owned_by\":\"zinc\",\"context_length\":8192}]}";
    try std.testing.expectEqualStrings(expected, result);
}

test "native browser discovery identifies only the loaded model and its real controls" {
    const bytes = try serializeChatModel(std.testing.allocator, "/fixtures/private/Qwen\"27B.gguf", 65536, .{
        .display_name = "Qwen 27B",
        .homepage_url = "https://example.com/model",
        .supports_thinking_toggle = true,
    });
    defer std.testing.allocator.free(bytes);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, bytes, .{});
    defer parsed.deinit();
    const models = parsed.value.object.get("data").?.array.items;
    try std.testing.expectEqual(@as(usize, 1), models.len);
    const model = models[0].object;
    try std.testing.expectEqualStrings("Qwen\"27B.gguf", model.get("id").?.string);
    try std.testing.expectEqualStrings("Qwen 27B", model.get("display_name").?.string);
    try std.testing.expectEqual(@as(i64, 65536), model.get("context_length").?.integer);
    try std.testing.expect(model.get("active").?.bool and model.get("installed").?.bool);
    try std.testing.expect(!model.get("managed").?.bool);
    try std.testing.expect(model.get("supports_thinking_toggle").?.bool);
    const legacy_bytes = try serializeModels(std.testing.allocator, "/fixtures/private/Qwen\"27B.gguf", 65536);
    defer std.testing.allocator.free(legacy_bytes);
    const legacy = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, legacy_bytes, .{});
    defer legacy.deinit();
    const original = legacy.value.object.get("data").?.array.items[0].object;
    // Browser additions preserve every pre-existing OpenAI discovery field.
    inline for (.{ "id", "object", "owned_by" }) |key| try std.testing.expectEqualStrings(original.get(key).?.string, model.get(key).?.string);
    inline for (.{ "created", "context_length" }) |key| try std.testing.expectEqual(original.get(key).?.integer, model.get(key).?.integer);
    try std.testing.expect(std.mem.indexOf(u8, bytes, "private") == null);
    try std.testing.expectError(error.InvalidUtf8, serializeChatModel(std.testing.allocator, "bad\xff.gguf", 65536, .{}));

    const unknown = try serializeChatModel(std.testing.allocator, "unknown.gguf", 32768, .{});
    defer std.testing.allocator.free(unknown);
    const fallback = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, unknown, .{});
    defer fallback.deinit();
    const item = fallback.value.object.get("data").?.array.items[0].object;
    try std.testing.expectEqualStrings("unknown.gguf", item.get("display_name").?.string);
    try std.testing.expect(!item.get("supports_thinking_toggle").?.bool);
}

test "serializeModels basename includes extension" {
    const allocator = std.testing.allocator;
    const result = try serializeModels(allocator, "/fixtures/models/qwen35-9b-q4k-m.gguf", 16384);
    defer allocator.free(result);
    const expected = "{\"object\":\"list\",\"data\":[{\"id\":\"qwen35-9b-q4k-m.gguf\",\"object\":\"model\",\"created\":0,\"owned_by\":\"zinc\",\"context_length\":16384}]}";
    try std.testing.expectEqualStrings(expected, result);
}

test "serializeModels privacy - no parent directories" {
    const allocator = std.testing.allocator;
    const result = try serializeModels(allocator, "/fixtures/models/my-model.gguf", 4096);
    defer allocator.free(result);
    const expected = "{\"object\":\"list\",\"data\":[{\"id\":\"my-model.gguf\",\"object\":\"model\",\"created\":0,\"owned_by\":\"zinc\",\"context_length\":4096}]}";
    try std.testing.expectEqualStrings(expected, result);
}

test "serializeModels JSON escaping quotes" {
    const allocator = std.testing.allocator;
    const result = try serializeModels(allocator, "test-model\"quotes.gguf", 2048);
    defer allocator.free(result);
    const expected = "{\"object\":\"list\",\"data\":[{\"id\":\"test-model\\\"quotes.gguf\",\"object\":\"model\",\"created\":0,\"owned_by\":\"zinc\",\"context_length\":2048}]}";
    try std.testing.expectEqualStrings(expected, result);
}

test "serializeModels JSON escaping backslash" {
    const allocator = std.testing.allocator;
    const result = try serializeModels(allocator, "test-model\\backslash.gguf", 2048);
    defer allocator.free(result);
    const expected = "{\"object\":\"list\",\"data\":[{\"id\":\"test-model\\\\backslash.gguf\",\"object\":\"model\",\"created\":0,\"owned_by\":\"zinc\",\"context_length\":2048}]}";
    try std.testing.expectEqualStrings(expected, result);
}

test "serializeModels JSON escaping newline" {
    const allocator = std.testing.allocator;
    const result = try serializeModels(allocator, "test-model\nnewline.gguf", 2048);
    defer allocator.free(result);
    const expected = "{\"object\":\"list\",\"data\":[{\"id\":\"test-model\\nnewline.gguf\",\"object\":\"model\",\"created\":0,\"owned_by\":\"zinc\",\"context_length\":2048}]}";
    try std.testing.expectEqualStrings(expected, result);
}

test "serializeModels JSON escaping tab" {
    const allocator = std.testing.allocator;
    const result = try serializeModels(allocator, "test-model\ttab.gguf", 2048);
    defer allocator.free(result);
    const expected = "{\"object\":\"list\",\"data\":[{\"id\":\"test-model\\ttab.gguf\",\"object\":\"model\",\"created\":0,\"owned_by\":\"zinc\",\"context_length\":2048}]}";
    try std.testing.expectEqualStrings(expected, result);
}

test "serializeModels JSON unicode" {
    const allocator = std.testing.allocator;
    const result = try serializeModels(allocator, "测试模型.gguf", 2048);
    defer allocator.free(result);
    const expected = "{\"object\":\"list\",\"data\":[{\"id\":\"测试模型.gguf\",\"object\":\"model\",\"created\":0,\"owned_by\":\"zinc\",\"context_length\":2048}]}";
    try std.testing.expectEqualStrings(expected, result);
}

test "serializeModels JSON mixed special chars" {
    const allocator = std.testing.allocator;
    const result = try serializeModels(allocator, "test\"model\\with\nmixed\tchars.gguf", 2048);
    defer allocator.free(result);
    const expected = "{\"object\":\"list\",\"data\":[{\"id\":\"test\\\"model\\\\with\\nmixed\\tchars.gguf\",\"object\":\"model\",\"created\":0,\"owned_by\":\"zinc\",\"context_length\":2048}]}";
    try std.testing.expectEqualStrings(expected, result);
}

test "serializeModels owns a complete allocation for a long model name" {
    const allocator = std.testing.allocator;
    const name = try allocator.alloc(u8, 20000);
    defer allocator.free(name);
    @memset(name, 'x');
    const model_path = try std.fmt.allocPrint(allocator, "/fixtures/parent/{s}", .{name});
    defer allocator.free(model_path);
    const result = try serializeModels(allocator, model_path, 32768);
    defer allocator.free(result);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, result, .{});
    defer parsed.deinit();
    const model = parsed.value.object.get("data").?.array.items[0].object;
    try std.testing.expectEqualStrings(name, model.get("id").?.string);
    try std.testing.expectEqual(@as(i64, 0), model.get("created").?.integer);
    try std.testing.expectEqual(@as(i64, 32768), model.get("context_length").?.integer);
    try std.testing.expect(std.mem.indexOf(u8, result, "/fixtures/parent") == null);
}

test "invalid UTF8 basename is a recoverable error" {
    const invalid = [_][]const u8{
        "\xff.gguf",            "bad\xc0\xaf.gguf",         "bad\xe2\x82.gguf",
        "bad\xed\xa0\x80.gguf", "bad\xf4\x90\x80\x80.gguf", "bad\x80.gguf",
    };
    for (invalid) |name| {
        const raw_path = try std.fmt.allocPrint(std.testing.allocator, "/fixtures/parent/{s}", .{name});
        defer std.testing.allocator.free(raw_path);
        const result = serializeModels(std.testing.allocator, raw_path, 262144);
        if (result) |bytes| {
            defer std.testing.allocator.free(bytes);
            return error.InvalidBasenameWasSerialized;
        } else |err| {
            try std.testing.expectEqual(error.InvalidUtf8, err);
        }
    }
}
test "valid basename stays a string and parent bytes remain private" {
    const names = [_][]const u8{ "model.gguf", "测试模型.gguf", "quoted\"\\\n\t.gguf", "max-\xf4\x8f\xbf\xbf.gguf" };
    for (names) |name| {
        const raw_path = try std.fmt.allocPrint(std.testing.allocator, "/fixtures/\xff-private/{s}", .{name});
        defer std.testing.allocator.free(raw_path);
        const bytes = try serializeModels(std.testing.allocator, raw_path, 32768);
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
    try std.testing.expectError(error.InvalidUtf8, serializeModels(failing.allocator(), "bad\xff.gguf", 32768));
    try std.testing.expectError(error.OutOfMemory, serializeModels(failing.allocator(), "valid.gguf", 32768));
}
