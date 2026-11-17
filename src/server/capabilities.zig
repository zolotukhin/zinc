//! Versioned, backend-independent client discovery. These are protocol features,
//! not claims about a model's tool quality or hardware qualification.
//! @section API Server
const std = @import("std");
const build_info = @import("../build_info.zig");
pub const Backend = @import("../cli_help.zig").Backend;

pub const Capabilities = struct {
    chat_completions: bool = true,
    text_completions: bool = true,
    chat_streaming: bool = true,
    text_streaming: bool,
    model_management: bool,
    model_switching: bool,
    browser_ui: bool,
    sampling: bool,
    tool_calls: enum { model_dependent } = .model_dependent,
    chat_session_reuse: bool,
};

pub const Info = struct {
    schema_version: u32 = 1,
    build: build_info.Metadata = .{},
    backend: Backend,
    active_model: ?[]const u8,
    context_length: ?u32,
    capabilities: Capabilities,
};

pub fn currentBackend() Backend {
    // Validate the compile-time build option before producing an executable.
    // An invalid package identity must not become a runtime discovery panic.
    return comptime (std.meta.stringToEnum(Backend, build_info.backend) orelse @compileError("Client discovery requires a Vulkan, Metal, ROCm or CUDA build"));
}

pub fn info(backend: Backend, model: ?[]const u8, context: ?u32, reuse: bool) Info {
    const accelerator = backend == .rocm or backend == .cuda;
    return .{
        .backend = backend,
        .active_model = model,
        .context_length = context,
        .capabilities = .{
            .text_streaming = accelerator,
            .model_management = !accelerator,
            .model_switching = !accelerator,
            .browser_ui = backend != .cuda,
            .sampling = !accelerator,
            .chat_session_reuse = backend == .vulkan and model != null and reuse,
        },
    };
}

pub fn writeJson(value: Info, writer: *std.Io.Writer) !void {
    try std.json.Stringify.value(value, .{}, writer);
}

/// ROCm/CUDA /v1/models identifies a loaded file by basename, never its parent.
pub fn nativeInfo(backend: Backend, path: ?[]const u8, context: u32) !Info {
    const id = if (path) |p| std.fs.path.basename(p) else null;
    if (id) |name| if (!std.unicode.utf8ValidateSlice(name)) return error.InvalidUtf8;
    return info(backend, id, if (id != null) context else null, false);
}

/// Native health retains its legacy status/backend/context keys and adds the
/// same discovery contract as Vulkan/Metal health and /v1/capabilities.
pub fn writeHealth(value: Info, writer: *std.Io.Writer) !void {
    try std.json.Stringify.value(.{
        .status = "ok",
        .schema_version = value.schema_version,
        .build = value.build,
        .backend = value.backend,
        // The shared chat UI reads this legacy display-name key. Native IDs
        // remain the GGUF basename and never expose the parent directory.
        .model = value.active_model orelse "",
        .active_model = value.active_model,
        .context_length = value.context_length,
        .capabilities = value.capabilities,
    }, .{}, writer);
}

/// Own model strings after releasing the model-manager lock. A concurrent hot
/// switch must not invalidate discovery while the HTTP response is serialized.
pub const Snapshot = struct {
    value: Info,
    display_name: []u8,

    pub fn init(allocator: std.mem.Allocator, backend: Backend, model: ?[]const u8, display: []const u8, context: ?u32, reuse: bool) !Snapshot {
        if (!std.unicode.utf8ValidateSlice(display)) return error.InvalidUtf8;
        if (model) |id| if (!std.unicode.utf8ValidateSlice(id)) return error.InvalidUtf8;
        const name = try allocator.dupe(u8, display);
        errdefer allocator.free(name);
        const id = if (model) |m| try allocator.dupe(u8, m) else null;
        return .{ .value = info(backend, id, context, reuse), .display_name = name };
    }

    pub fn deinit(self: *Snapshot, allocator: std.mem.Allocator) void {
        if (self.value.active_model) |id| allocator.free(id);
        allocator.free(self.display_name);
    }
};

test "discovery owns model identity across a hot switch and rejects invalid UTF8" {
    var id = [_]u8{ 'a', 'b', 'c' };
    var snapshot = try Snapshot.init(std.testing.allocator, .vulkan, &id, &id, 4096, true);
    defer snapshot.deinit(std.testing.allocator);
    id[0] = 'z';
    try std.testing.expectEqualStrings("abc", snapshot.value.active_model.?);
    try std.testing.expectEqualStrings("abc", snapshot.display_name);
    try std.testing.expectError(error.InvalidUtf8, Snapshot.init(std.testing.allocator, .rocm, "bad\xff", "model", 2048, false));
}

test "all backends share discovery schema and advertise only supported features" {
    inline for (.{ Backend.vulkan, Backend.metal, Backend.rocm, Backend.cuda }) |backend| {
        const value = info(backend, "model\"\\\n.gguf", 8192, true);
        const accelerator = backend == .rocm or backend == .cuda;
        try std.testing.expectEqual(accelerator, value.capabilities.text_streaming);
        try std.testing.expectEqual(!accelerator, value.capabilities.model_switching);
        try std.testing.expectEqual(backend != .cuda, value.capabilities.browser_ui);
        try std.testing.expectEqual(backend == .vulkan, value.capabilities.chat_session_reuse);
        var buffer: [4096]u8 = undefined;
        var writer = std.Io.Writer.fixed(&buffer);
        try writeJson(value, &writer);
        const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, writer.buffered(), .{});
        defer parsed.deinit();
        try std.testing.expectEqualStrings(@tagName(backend), parsed.value.object.get("backend").?.string);
        try std.testing.expectEqualStrings(value.active_model.?, parsed.value.object.get("active_model").?.string);
        try std.testing.expectEqual(@as(i64, 8192), parsed.value.object.get("context_length").?.integer);
    }
    const empty = info(.vulkan, null, null, true);
    try std.testing.expect(!empty.capabilities.chat_session_reuse);
    try std.testing.expect(empty.active_model == null and empty.context_length == null);
}

test "native health and discovery share identity without leaking parent paths" {
    const value = try nativeInfo(.rocm, "/fixtures/private/model\".gguf", 16384);
    var buffer: [4096]u8 = undefined;
    var writer = std.Io.Writer.fixed(&buffer);
    try writeHealth(value, &writer);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, writer.buffered(), .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("model\".gguf", parsed.value.object.get("active_model").?.string);
    try std.testing.expectEqualStrings("model\".gguf", parsed.value.object.get("model").?.string);
    try std.testing.expectEqualStrings("ok", parsed.value.object.get("status").?.string);
    try std.testing.expect(std.mem.indexOf(u8, writer.buffered(), "private") == null);
    try std.testing.expectError(error.InvalidUtf8, nativeInfo(.rocm, "/fixtures/bad\xff.gguf", 2048));
    try std.testing.expect((try nativeInfo(.cuda, null, 2048)).active_model == null);

    writer = std.Io.Writer.fixed(&buffer);
    try writeHealth(try nativeInfo(.rocm, null, 2048), &writer);
    const empty = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, writer.buffered(), .{});
    defer empty.deinit();
    try std.testing.expectEqualStrings("", empty.value.object.get("model").?.string);
    try std.testing.expect(empty.value.object.get("active_model").? == .null);
}
