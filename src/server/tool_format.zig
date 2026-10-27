//! Pluggable tool-calling format dispatch for chat completions.
//! @section Tool Calling
//! Qwen ChatML and Gemma 4 use their checkpoint-native tool dialects.
//! `NoopToolFormat` is the fallback for templates without tool support.
const std = @import("std");
const TemplateKind = @import("../model/tokenizer.zig").Tokenizer.TemplateKind;
const Tokenizer = @import("../model/tokenizer.zig").Tokenizer;

const log = std.log.scoped(.tool_format);
pub const gemma = @import("gemma_tools.zig");

var global_tool_call_id = std.atomic.Value(u64).init(0);

fn allocToolCallId(allocator: std.mem.Allocator) ![]u8 {
    const id = global_tool_call_id.fetchAdd(1, .monotonic);
    return std.fmt.allocPrint(allocator, "call_{d}", .{id});
}

fn isWhitespaceOnly(s: []const u8) bool {
    for (s) |c| switch (c) {
        ' ', '\t', '\r', '\n' => {},
        else => return false,
    };
    return true;
}

/// One tool definition extracted from the request's `tools` array.
pub const ToolDefinition = struct {
    name: []const u8,
    description: []const u8,
    /// Raw JSON object representing the parameters schema. Not validated.
    parameters_json: []const u8,
};

/// One parsed tool call extracted from assistant output.
pub const ToolCall = struct {
    /// Generated as "call_<n>" by the parser. OpenAI requires non-empty id.
    id: []const u8,
    name: []const u8,
    /// Raw JSON object string for the tool's arguments.
    arguments_json: []const u8,
};

/// Result of parsing a non-streaming assistant message: the prose the user
/// should see and any tool invocations the model emitted. Returned by
/// `ToolFormat.parseAssistantToolCalls` and consumed by the chat completions
/// response builder when deciding between a `content` reply and a `tool_calls`
/// reply.
pub const ParsedAssistantOutput = struct {
    /// Anything outside `<tool_call>...</tool_call>` blocks.
    text_content: []const u8,
    /// Empty slice if no tool calls were detected.
    tool_calls: []const ToolCall,
};

/// Disposition of a streaming chunk after the `StreamingDetector` has inspected
/// it. The chat completions streamer uses this to decide whether to forward
/// bytes as a `content` delta, swallow them for further inspection, or flush
/// a finished tool call.
pub const FeedResult = enum {
    /// Bytes pass through to the SSE content delta.
    emit_as_content,
    /// Bytes are buffered internally; do not emit.
    hold,
    /// A complete tool_call was just parsed; pull it via takePendingToolCall.
    tool_call_complete,
};

/// Streaming-mode tool-call detector. The chat completions handler feeds each
/// decoded chunk through this state machine; the detector buffers bytes that
/// might be the start of a `<tool_call>` tag and flushes them either as
/// content deltas or as parsed tool calls. One detector per streaming request.
pub const StreamingDetector = struct {
    native_gemma: bool = false,
    /// Request schemas distinguish XML string values from numbers/objects.
    tools: []const ToolDefinition = &.{},
    /// Bytes pending emission as content delta.
    content_pending: std.ArrayList(u8) = .empty,
    /// Bytes being inspected for an incomplete <tool_call> open tag.
    hold_buf: std.ArrayList(u8) = .empty,
    /// Tool calls fully parsed and waiting for the caller to drain.
    pending_calls: std.ArrayList(ToolCall) = .empty,
    allocator: std.mem.Allocator,

    /// Maximum bytes to hold while waiting for an open tag to disambiguate.
    /// "<tool_call>" is 11 bytes; we use 24 for safety margin.
    const max_hold = 24;

    /// Free the internal buffers and any pending tool-call payloads. Safe to
    /// call once at end-of-stream regardless of how many `feed` calls happened.
    pub fn deinit(self: *StreamingDetector) void {
        self.content_pending.deinit(self.allocator);
        self.hold_buf.deinit(self.allocator);
        for (self.pending_calls.items) |c| {
            self.allocator.free(c.id);
            self.allocator.free(c.name);
            self.allocator.free(c.arguments_json);
        }
        self.pending_calls.deinit(self.allocator);
    }

    /// Push the next decoded chunk into the detector and return how the caller
    /// should react. The detector retains ownership of `chunk`'s contribution
    /// to the internal buffer; callers should drain via `takeContentDelta` and
    /// `takePendingToolCall` between feeds.
    /// @param chunk Newly decoded model bytes.
    /// @returns A `FeedResult` indicating whether content is ready to emit, a
    ///     tool call is ready to consume, or the bytes are still being held.
    pub fn feed(self: *StreamingDetector, chunk: []const u8) !FeedResult {
        const open = if (self.native_gemma) gemma.call_open else tool_call_open;
        const close = if (self.native_gemma) gemma.call_close else tool_call_close;
        try self.hold_buf.appendSlice(self.allocator, chunk);
        while (true) {
            if (std.mem.indexOf(u8, self.hold_buf.items, open)) |open_at| {
                if (open_at > 0) {
                    try self.content_pending.appendSlice(self.allocator, self.hold_buf.items[0..open_at]);
                    std.mem.copyForwards(u8, self.hold_buf.items, self.hold_buf.items[open_at..]);
                    self.hold_buf.shrinkRetainingCapacity(self.hold_buf.items.len - open_at);
                }
                const close_at = (if (self.native_gemma) gemma.findClose(self.hold_buf.items, open.len) else findCallClose(self.hold_buf.items, open.len)) orelse break;
                const inner = std.mem.trim(u8, self.hold_buf.items[open.len..close_at], " \t\r\n");
                const after_close = close_at + close.len;
                if (try parseDialectCall(inner, self.tools, self.allocator, self.native_gemma)) |call| {
                    errdefer {
                        self.allocator.free(call.id);
                        self.allocator.free(call.name);
                        self.allocator.free(call.arguments_json);
                    }
                    try self.pending_calls.append(self.allocator, call);
                } else {
                    try self.content_pending.appendSlice(self.allocator, self.hold_buf.items[0..after_close]);
                }
                std.mem.copyForwards(u8, self.hold_buf.items, self.hold_buf.items[after_close..]);
                self.hold_buf.shrinkRetainingCapacity(self.hold_buf.items.len - after_close);
                continue;
            }
            // Retain only a suffix that can grow into the opening delimiter.
            var held = @min(self.hold_buf.items.len, open.len - 1);
            while (held > 0 and !std.mem.endsWith(u8, self.hold_buf.items, open[0..held])) : (held -= 1) {}
            const emit = self.hold_buf.items.len - held;
            try self.content_pending.appendSlice(self.allocator, self.hold_buf.items[0..emit]);
            std.mem.copyForwards(u8, self.hold_buf.items, self.hold_buf.items[emit..]);
            self.hold_buf.shrinkRetainingCapacity(held);
            break;
        }
        if (self.pending_calls.items.len > 0) return .tool_call_complete;
        if (self.content_pending.items.len > 0) return .emit_as_content;
        return .hold;
    }

    /// Drain pending content bytes. The returned slice aliases the detector's
    /// internal buffer — consume it before the next feed call (subsequent feeds
    /// will overwrite the same allocation). The detector retains ownership and
    /// the buffer is freed by deinit.
    pub fn takeContentDelta(self: *StreamingDetector) []const u8 {
        const out = self.content_pending.items;
        // clearRetainingCapacity poisons removed bytes in safety builds. This
        // API borrows those bytes until the next feed, so only reset the length.
        self.content_pending.items.len = 0;
        return out;
    }

    /// Drain one fully parsed tool call, FIFO order. Caller takes ownership of
    /// the returned `id`/`name`/`arguments_json` slices and must free them with
    /// the same allocator that initialized this detector. Returns null when the
    /// queue is empty.
    pub fn takePendingToolCall(self: *StreamingDetector) ?ToolCall {
        if (self.pending_calls.items.len == 0) return null;
        return self.pending_calls.orderedRemove(0);
    }

    /// Flush all remaining buffered bytes as content at end of stream. Any bytes
    /// still held in the speculative-match buffer are appended to the pending
    /// content buffer, and the combined slice is returned. The returned slice
    /// aliases the internal buffer; consume before calling deinit.
    pub fn finalize(self: *StreamingDetector) []const u8 {
        const held = self.hold_buf.items;
        if (held.len == 0) return self.content_pending.items;
        self.content_pending.appendSlice(self.allocator, held) catch return self.content_pending.items;
        self.hold_buf.clearRetainingCapacity();
        return self.content_pending.items;
    }
};

/// Vtable interface to a per-template tool-call format. Lets the chat
/// completions path render tool definitions, parse tool calls out of model
/// output, and create matching streaming detectors without knowing whether
/// the active template is ChatML, llama3, or anything else. Concrete
/// implementations are minted via `forTemplate` or the per-format factories
/// `chatmlToolFormat` / `noopToolFormat`.
pub const ToolFormat = struct {
    ptr: *anyopaque,
    vtable: *const VTable,
    native_xml: bool = false,
    native_gemma: bool = false,
    supports_tools: bool = false,

    pub fn usesJsonPrefill(self: ToolFormat) bool {
        return self.supports_tools and !self.native_xml and !self.native_gemma;
    }

    /// Function-pointer table that concrete format implementations fill in.
    pub const VTable = struct {
        renderToolDefinitions: *const fn (
            ctx: *anyopaque,
            tools: []const ToolDefinition,
            buf: *std.ArrayList(u8),
            allocator: std.mem.Allocator,
        ) anyerror!void,

        renderToolResultMessage: *const fn (
            ctx: *anyopaque,
            tool_call_id: []const u8,
            content: []const u8,
            buf: *std.ArrayList(u8),
            allocator: std.mem.Allocator,
        ) anyerror!void,

        parseAssistantToolCalls: *const fn (
            ctx: *anyopaque,
            model_output: []const u8,
            tools: []const ToolDefinition,
            allocator: std.mem.Allocator,
        ) anyerror!ParsedAssistantOutput,

        newStreamingDetector: *const fn (
            ctx: *anyopaque,
            allocator: std.mem.Allocator,
        ) anyerror!*StreamingDetector,
    };

    /// Append the format-specific rendering of `tools` (e.g. Qwen's
    /// `# Tools\n<tools>...</tools>` block) to `buf`, ready to be spliced into
    /// the system message. Noop formats append nothing.
    pub fn renderToolDefinitions(
        self: ToolFormat,
        tools: []const ToolDefinition,
        buf: *std.ArrayList(u8),
        allocator: std.mem.Allocator,
    ) anyerror!void {
        return self.vtable.renderToolDefinitions(self.ptr, tools, buf, allocator);
    }

    /// Append the format-specific tool-result message (e.g. ChatML's
    /// `<tool_response>...</tool_response>`) to `buf`. Used when replaying
    /// `role: "tool"` history entries into the prompt.
    pub fn renderToolResultMessage(
        self: ToolFormat,
        tool_call_id: []const u8,
        content: []const u8,
        buf: *std.ArrayList(u8),
        allocator: std.mem.Allocator,
    ) anyerror!void {
        return self.vtable.renderToolResultMessage(self.ptr, tool_call_id, content, buf, allocator);
    }

    /// Split a complete (non-streaming) assistant response into prose plus a
    /// list of structured tool calls. Allocates the returned slices with the
    /// caller's allocator; ownership transfers to the caller.
    pub fn parseAssistantToolCalls(
        self: ToolFormat,
        model_output: []const u8,
        allocator: std.mem.Allocator,
    ) anyerror!ParsedAssistantOutput {
        return self.vtable.parseAssistantToolCalls(self.ptr, model_output, &.{}, allocator);
    }

    pub fn parseAssistantToolCallsWithTools(self: ToolFormat, model_output: []const u8, tools: []const ToolDefinition, allocator: std.mem.Allocator) !ParsedAssistantOutput {
        return self.vtable.parseAssistantToolCalls(self.ptr, model_output, tools, allocator);
    }

    /// Create a fresh streaming-mode detector for one chat completion stream.
    /// Caller owns the returned pointer and must `deinit` + `destroy` it.
    pub fn newStreamingDetector(
        self: ToolFormat,
        allocator: std.mem.Allocator,
    ) anyerror!*StreamingDetector {
        return self.vtable.newStreamingDetector(self.ptr, allocator);
    }
};

// ============================================================
// NoopToolFormat — silent fallback for non-ChatML templates.
// ============================================================

/// Silent fallback `ToolFormat` for templates that don't have a tool-call
/// dialect wired in (everything except ChatML today). Definition rendering is
/// a no-op, parsing returns the model output verbatim with no tool calls, and
/// the streaming detector treats everything as content.
pub const NoopToolFormat = struct {
    fn render_defs(_: *anyopaque, _: []const ToolDefinition, _: *std.ArrayList(u8), _: std.mem.Allocator) !void {}

    fn render_result(_: *anyopaque, _: []const u8, content: []const u8, buf: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        try buf.appendSlice(allocator, content);
    }

    fn parse_calls(_: *anyopaque, model_output: []const u8, _: []const ToolDefinition, allocator: std.mem.Allocator) !ParsedAssistantOutput {
        _ = allocator;
        return .{ .text_content = model_output, .tool_calls = &.{} };
    }

    fn new_detector(_: *anyopaque, allocator: std.mem.Allocator) !*StreamingDetector {
        const d = try allocator.create(StreamingDetector);
        d.* = .{ .allocator = allocator };
        return d;
    }

    const vtable = ToolFormat.VTable{
        .renderToolDefinitions = render_defs,
        .renderToolResultMessage = render_result,
        .parseAssistantToolCalls = parse_calls,
        .newStreamingDetector = new_detector,
    };

    var instance: u8 = 0; // dummy ctx pointer; Noop has no state
};

/// Build a `ToolFormat` backed by `NoopToolFormat`. Returned value is cheap to
/// pass around and shares a single static instance.
pub fn noopToolFormat() ToolFormat {
    return .{
        .ptr = @ptrCast(&NoopToolFormat.instance),
        .vtable = &NoopToolFormat.vtable,
    };
}

// ============================================================
// Factory: pick the right ToolFormat for a template kind.
// ============================================================

/// Pick the right `ToolFormat` for a chat template family. ChatML maps to the
/// Qwen3-style `<tool_call>` dialect; everything else falls through to the
/// no-op format (tools field accepted but silently ignored downstream).
pub fn forTemplate(template_kind: TemplateKind) ToolFormat {
    return switch (template_kind) {
        .chatml => chatmlToolFormat(),
        else => noopToolFormat(),
    };
}

/// Newer Qwen checkpoints publish XML function/parameter tags in their own
/// GGUF template. Keep the original JSON dialect for other ChatML checkpoints.
pub fn forTokenizer(tokenizer: *const Tokenizer) ToolFormat {
    if (tokenizer.detectTemplateKind() == .gemma and std.mem.indexOf(u8, tokenizer.chat_template orelse "", "<|tool_call>") != null) return gemmaToolFormat();
    if (tokenizer.detectTemplateKind() == .chatml and std.mem.indexOf(u8, tokenizer.chat_template orelse "", "<function=") != null) {
        var format = chatmlToolFormat();
        format.vtable = &vtable_chatml_xml;
        format.native_xml = true;
        return format;
    }
    return forTemplate(tokenizer.detectTemplateKind());
}

// ============================================================
// ChatMLToolFormat — Qwen3-family tool calling.
// ============================================================

const tool_call_open = "<tool_call>";
const tool_call_close = "</tool_call>";

fn findCallClose(text: []const u8, start: usize) ?usize {
    var first = start;
    while (first < text.len and std.ascii.isWhitespace(text[first])) : (first += 1) {}
    const json = first < text.len and text[first] == '{';
    const xml = std.mem.startsWith(u8, text[first..], "<function=");
    var in_string = false;
    var escaped = false;
    var i = first;
    while (i < text.len) : (i += 1) {
        if (json) {
            if (in_string) {
                if (escaped) escaped = false else if (text[i] == '\\') escaped = true else if (text[i] == '"') in_string = false;
                continue;
            }
            if (text[i] == '"') {
                in_string = true;
                continue;
            }
        }
        if (std.mem.startsWith(u8, text[i..], tool_call_close)) {
            if (!xml or std.mem.endsWith(u8, std.mem.trimEnd(u8, text[first..i], " \t\r\n"), "</function>")) return i;
        }
    }
    return null;
}

fn schemaAllowsString(schema: std.json.Value, depth: u8) bool {
    if (depth == 0 or schema != .object) return true;
    if (schema.object.get("type")) |kind| {
        if (kind == .string) return std.mem.eql(u8, kind.string, "string");
        if (kind == .array) {
            for (kind.array.items) |item| if (item == .string and std.mem.eql(u8, item.string, "string")) return true;
            return false;
        }
    }
    for ([_][]const u8{ "anyOf", "oneOf" }) |key| {
        if (schema.object.get(key)) |alternatives| {
            if (alternatives != .array) return true;
            for (alternatives.array.items) |item| if (schemaAllowsString(item, depth - 1)) return true;
            return false;
        }
    }
    return true;
}

fn stripXmlEnvelopeNewlines(value: []const u8) []const u8 {
    var out = value;
    if (std.mem.startsWith(u8, out, "\r\n")) out = out[2..] else if (std.mem.startsWith(u8, out, "\n")) out = out[1..];
    if (std.mem.endsWith(u8, out, "\r\n")) out = out[0 .. out.len - 2] else if (std.mem.endsWith(u8, out, "\n")) out = out[0 .. out.len - 1];
    return out;
}

fn parsedCall(allocator: std.mem.Allocator, name: []const u8, args: std.json.Value) !?ToolCall {
    if (name.len == 0) return null;
    const id = try allocToolCallId(allocator);
    errdefer allocator.free(id);
    const owned_name = try allocator.dupe(u8, name);
    errdefer allocator.free(owned_name);
    const arguments = try std.json.Stringify.valueAlloc(allocator, args, .{});
    return .{ .id = id, .name = owned_name, .arguments_json = arguments };
}

/// Parse only complete native XML or JSON calls. Malformed blocks stay text;
/// schemas type XML values without guessing that source-code strings are JSON.
fn parseCallInner(inner: []const u8, tools: []const ToolDefinition, allocator: std.mem.Allocator) !?ToolCall {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    if (!std.mem.startsWith(u8, inner, "<function=")) {
        const json = std.json.parseFromSliceLeaky(std.json.Value, a, inner, .{}) catch return null;
        if (json != .object) return null;
        const name = json.object.get("name") orelse return null;
        if (name != .string) return null;
        return parsedCall(allocator, name.string, json.object.get("arguments") orelse .{ .object = .empty });
    }
    const name_end = std.mem.indexOfScalar(u8, inner, '>') orelse return null;
    const name = inner["<function=".len..name_end];
    if (name.len == 0 or std.mem.indexOfAny(u8, name, "<>\n\r\t ") != null) return null;
    var schema: std.json.Value = .null;
    for (tools) |tool| {
        if (std.mem.eql(u8, tool.name, name)) {
            schema = std.json.parseFromSliceLeaky(std.json.Value, a, tool.parameters_json, .{}) catch .null;
            break;
        }
    }
    var args: std.json.Value = .{ .object = .empty };
    var cursor = name_end + 1;
    while (true) {
        while (cursor < inner.len and std.ascii.isWhitespace(inner[cursor])) : (cursor += 1) {}
        if (std.mem.startsWith(u8, inner[cursor..], "</function>")) {
            cursor += "</function>".len;
            if (!isWhitespaceOnly(inner[cursor..])) return null;
            return parsedCall(allocator, name, args);
        }
        if (!std.mem.startsWith(u8, inner[cursor..], "<parameter=")) return null;
        const key_start = cursor + "<parameter=".len;
        const key_end = std.mem.indexOfScalarPos(u8, inner, key_start, '>') orelse return null;
        const key = inner[key_start..key_end];
        if (key.len == 0 or args.object.contains(key)) return null;
        const value_start = key_end + 1;
        const value_end = std.mem.indexOfPos(u8, inner, value_start, "</parameter>") orelse return null;
        const raw = stripXmlEnvelopeNewlines(inner[value_start..value_end]);
        var value: std.json.Value = .{ .string = raw };
        if (schema == .object) {
            if (schema.object.get("properties")) |properties| {
                if (properties == .object) {
                    if (properties.object.get(key)) |property| {
                        if (!schemaAllowsString(property, 8)) {
                            // Native Qwen/Jinja XML also spells scalar booleans
                            // True/False and null as None. Decode these only
                            // when the schema excludes strings, preserving
                            // source-code and JSON-looking string arguments.
                            const scalar = std.mem.trim(u8, raw, " \t\r\n");
                            value = if (std.ascii.eqlIgnoreCase(scalar, "true"))
                                .{ .bool = true }
                            else if (std.ascii.eqlIgnoreCase(scalar, "false"))
                                .{ .bool = false }
                            else if (std.mem.eql(u8, scalar, "None"))
                                .null
                            else
                                std.json.parseFromSliceLeaky(std.json.Value, a, raw, .{}) catch .{ .string = raw };
                        }
                    }
                }
            }
        }
        try args.object.put(a, key, value);
        cursor = value_end + "</parameter>".len;
    }
}

/// Replay SDK tool history in the checkpoint's native dialect.
pub fn appendAssistantCall(buf: *std.ArrayList(u8), allocator: std.mem.Allocator, name: []const u8, arguments: []const u8, native_xml: bool) !void {
    if (!native_xml) {
        const escaped_name = try std.json.Stringify.valueAlloc(allocator, name, .{});
        defer allocator.free(escaped_name);
        try buf.print(allocator, "<tool_call>\n{{\"name\":{s},\"arguments\":{s}}}\n</tool_call>\n", .{ escaped_name, arguments });
        return;
    }
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, arguments, .{});
    defer parsed.deinit();
    if (parsed.value != .object) return error.InvalidToolArguments;
    try buf.print(allocator, "<tool_call>\n<function={s}>\n", .{name});
    var it = parsed.value.object.iterator();
    while (it.next()) |entry| {
        try buf.print(allocator, "<parameter={s}>\n", .{entry.key_ptr.*});
        if (entry.value_ptr.* == .string) {
            try buf.appendSlice(allocator, entry.value_ptr.string);
        } else {
            const json = try std.json.Stringify.valueAlloc(allocator, entry.value_ptr.*, .{});
            defer allocator.free(json);
            try buf.appendSlice(allocator, json);
        }
        try buf.appendSlice(allocator, "\n</parameter>\n");
    }
    try buf.appendSlice(allocator, "</function>\n</tool_call>\n");
}

test "ChatML tool descriptions remain valid JSON with quotes newlines and Unicode" {
    const allocator = std.testing.allocator;
    const description = "Read a \"file\".\nUse a JSON path, including café 東京.\n\tControl: \x01";
    const tools = [_]ToolDefinition{.{ .name = "read", .description = description, .parameters_json = "{\"type\":\"object\"}" }};
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(allocator);
    try chatmlToolFormat().renderToolDefinitions(&tools, &buf, allocator);
    var lines = std.mem.splitScalar(u8, buf.items, '\n');
    var found = false;
    while (lines.next()) |line| {
        if (!std.mem.startsWith(u8, line, "{\"type\"")) continue;
        const parsed = try std.json.parseFromSlice(std.json.Value, allocator, line, .{});
        defer parsed.deinit();
        const function = parsed.value.object.get("function").?.object;
        try std.testing.expectEqualStrings("read", function.get("name").?.string);
        try std.testing.expectEqualStrings(description, function.get("description").?.string);
        found = true;
    }
    try std.testing.expect(found);
}

fn chatml_render_defs(
    _: *anyopaque,
    tools: []const ToolDefinition,
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
) !void {
    return chatmlRenderDefs(tools, buf, allocator, false);
}

fn chatml_xml_render_defs(_: *anyopaque, tools: []const ToolDefinition, buf: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
    return chatmlRenderDefs(tools, buf, allocator, true);
}

fn chatmlRenderDefs(tools: []const ToolDefinition, buf: *std.ArrayList(u8), allocator: std.mem.Allocator, native_xml: bool) !void {
    if (tools.len == 0) return;

    // Qwen/ChatML scaffold. Each signature must be a valid JSON line even when
    // client descriptions contain Markdown, quoted examples, or newlines.
    try buf.appendSlice(allocator,
        \\
        \\# Tools
        \\
        \\You may call one or more functions to assist with the user query.
        \\
        \\You are provided with function signatures within <tools></tools> XML tags:
        \\<tools>
        \\
    );

    for (tools) |tool| {
        const name_json = try std.json.Stringify.valueAlloc(allocator, tool.name, .{});
        defer allocator.free(name_json);
        const description_json = try std.json.Stringify.valueAlloc(allocator, tool.description, .{});
        defer allocator.free(description_json);
        try buf.appendSlice(allocator, "{\"type\": \"function\", \"function\": {\"name\": ");
        try buf.appendSlice(allocator, name_json);
        try buf.appendSlice(allocator, ", \"description\": ");
        try buf.appendSlice(allocator, description_json);
        try buf.appendSlice(allocator, ", \"parameters\": ");
        try buf.appendSlice(allocator, tool.parameters_json);
        try buf.appendSlice(allocator, "}}\n");
    }

    if (native_xml) {
        try buf.appendSlice(allocator,
            \\</tools>
            \\
            \\If you choose to call a function, reply in this format:
            \\<tool_call>
            \\<function=example_function_name>
            \\<parameter=example_parameter_name>
            \\value
            \\</parameter>
            \\</function>
            \\</tool_call>
            \\Use the exact names from the signatures and include every required parameter. Parameter values can span multiple lines. Put any explanation before the call, with no text after it.
        );
        return;
    }
    try buf.appendSlice(allocator,
        \\</tools>
        \\
        \\For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
        \\<tool_call>
        \\{"name": "function_name", "arguments": {"argument_name": "value"}}
        \\</tool_call>
        \\Use the exact function and argument names from the signatures. Quote every string value and keep arguments inside the arguments object.
    );
}

fn chatml_render_result(
    _: *anyopaque,
    tool_call_id: []const u8,
    content: []const u8,
    buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
) !void {
    _ = tool_call_id; // Qwen3 doesn't render the call id
    try buf.appendSlice(allocator, "<tool_response>\n");
    try buf.appendSlice(allocator, content);
    if (content.len == 0 or content[content.len - 1] != '\n') {
        try buf.appendSlice(allocator, "\n");
    }
    try buf.appendSlice(allocator, "</tool_response>\n");
}

fn chatml_fallback_block(
    text_buf: *std.ArrayList(u8),
    allocator: std.mem.Allocator,
    inner: []const u8,
) !void {
    try text_buf.appendSlice(allocator, tool_call_open);
    try text_buf.appendSlice(allocator, "\n");
    try text_buf.appendSlice(allocator, inner);
    try text_buf.appendSlice(allocator, "\n");
    try text_buf.appendSlice(allocator, tool_call_close);
}

fn chatml_parse_calls(_: *anyopaque, model_output: []const u8, tools: []const ToolDefinition, allocator: std.mem.Allocator) !ParsedAssistantOutput {
    var calls: std.ArrayList(ToolCall) = .empty;
    errdefer {
        for (calls.items) |call| {
            allocator.free(call.id);
            allocator.free(call.name);
            allocator.free(call.arguments_json);
        }
        calls.deinit(allocator);
    }
    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(allocator);
    var cursor: usize = 0;
    var just_called = false;
    while (std.mem.indexOfPos(u8, model_output, cursor, tool_call_open)) |open| {
        const between = model_output[cursor..open];
        if (!just_called or !isWhitespaceOnly(between)) try text.appendSlice(allocator, between);
        const begin = open + tool_call_open.len;
        const close = findCallClose(model_output, begin) orelse {
            try text.appendSlice(allocator, model_output[open..]);
            cursor = model_output.len;
            break;
        };
        const inner = std.mem.trim(u8, model_output[begin..close], " \t\r\n");
        if (try parseCallInner(inner, tools, allocator)) |call| {
            errdefer {
                allocator.free(call.id);
                allocator.free(call.name);
                allocator.free(call.arguments_json);
            }
            try calls.append(allocator, call);
            just_called = true;
        } else {
            try chatml_fallback_block(&text, allocator, inner);
            just_called = false;
        }
        cursor = close + tool_call_close.len;
    }
    if (cursor < model_output.len and (!just_called or !isWhitespaceOnly(model_output[cursor..]))) try text.appendSlice(allocator, model_output[cursor..]);
    const content = try text.toOwnedSlice(allocator);
    errdefer allocator.free(content);
    return .{ .text_content = content, .tool_calls = try calls.toOwnedSlice(allocator) };
}

fn chatml_new_detector(_: *anyopaque, allocator: std.mem.Allocator) !*StreamingDetector {
    const d = try allocator.create(StreamingDetector);
    d.* = .{ .allocator = allocator };
    return d;
}

const vtable_chatml = ToolFormat.VTable{
    .renderToolDefinitions = chatml_render_defs,
    .renderToolResultMessage = chatml_render_result,
    .parseAssistantToolCalls = chatml_parse_calls,
    .newStreamingDetector = chatml_new_detector,
};

const vtable_chatml_xml = ToolFormat.VTable{
    .renderToolDefinitions = chatml_xml_render_defs,
    .renderToolResultMessage = chatml_render_result,
    .parseAssistantToolCalls = chatml_parse_calls,
    .newStreamingDetector = chatml_new_detector,
};

var chatml_instance: u8 = 0;

/// Build a `ToolFormat` that emits the Qwen3-family `<tool_call>...` dialect.
/// Used for any template detected as ChatML. Returned value is cheap to pass
/// around and shares a single static instance.
pub fn chatmlToolFormat() ToolFormat {
    return .{
        .ptr = @ptrCast(&chatml_instance),
        .vtable = &vtable_chatml,
        .supports_tools = true,
    };
}

fn parseDialectCall(inner: []const u8, tools: []const ToolDefinition, allocator: std.mem.Allocator, native_gemma: bool) !?ToolCall {
    if (!native_gemma) return parseCallInner(inner, tools, allocator);
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();
    const call = (try gemma.parse(inner, arena.allocator())) orelse return null;
    // Never promote an unadvertised function into an executable SDK tool call.
    for (tools) |tool| if (std.mem.eql(u8, tool.name, call.name)) return parsedCall(allocator, call.name, call.arguments);
    return null;
}

fn gemma_render_defs(_: *anyopaque, tools: []const ToolDefinition, buf: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
    for (tools) |tool| try gemma.appendDefinition(buf, allocator, tool.name, tool.description, tool.parameters_json);
}

fn gemma_render_result(_: *anyopaque, name: []const u8, content: []const u8, buf: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
    return gemma.appendResult(buf, allocator, name, content);
}

fn gemma_parse_calls(_: *anyopaque, text: []const u8, tools: []const ToolDefinition, allocator: std.mem.Allocator) !ParsedAssistantOutput {
    var detector = StreamingDetector{ .allocator = allocator, .native_gemma = true, .tools = tools };
    defer detector.deinit();
    _ = try detector.feed(gemma.visibleText(text));
    const content = try allocator.dupe(u8, detector.finalize());
    errdefer allocator.free(content);
    const calls = try detector.pending_calls.toOwnedSlice(allocator);
    return .{ .text_content = content, .tool_calls = calls };
}

fn gemma_new_detector(_: *anyopaque, allocator: std.mem.Allocator) !*StreamingDetector {
    const d = try allocator.create(StreamingDetector);
    d.* = .{ .allocator = allocator, .native_gemma = true };
    return d;
}

const vtable_gemma = ToolFormat.VTable{
    .renderToolDefinitions = gemma_render_defs,
    .renderToolResultMessage = gemma_render_result,
    .parseAssistantToolCalls = gemma_parse_calls,
    .newStreamingDetector = gemma_new_detector,
};

pub fn gemmaToolFormat() ToolFormat {
    return .{ .ptr = @ptrCast(&chatml_instance), .vtable = &vtable_gemma, .native_gemma = true, .supports_tools = true };
}

// ============================================================
// Tests
// ============================================================

test "NoopToolFormat.renderToolDefinitions is a no-op" {
    const tf = noopToolFormat();
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    const tools = [_]ToolDefinition{
        .{ .name = "foo", .description = "bar", .parameters_json = "{}" },
    };
    try tf.renderToolDefinitions(&tools, &buf, std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), buf.items.len);
}

test "NoopToolFormat.parseAssistantToolCalls returns text_content unchanged" {
    const tf = noopToolFormat();
    const result = try tf.parseAssistantToolCalls("hello world", std.testing.allocator);
    try std.testing.expectEqualStrings("hello world", result.text_content);
    try std.testing.expectEqual(@as(usize, 0), result.tool_calls.len);
}

test "NoopToolFormat.renderToolResultMessage appends raw content" {
    const tf = noopToolFormat();
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    try tf.renderToolResultMessage("call_0", "result text", &buf, std.testing.allocator);
    try std.testing.expectEqualStrings("result text", buf.items);
}

test "forTemplate returns a usable ToolFormat for every kind" {
    inline for (.{ .chatml, .llama3, .gemma, .openai_moe, .generic }) |kind| {
        const tf = forTemplate(kind);
        const result = try tf.parseAssistantToolCalls("x", std.testing.allocator);
        if (kind == .chatml) {
            std.testing.allocator.free(result.text_content);
            std.testing.allocator.free(result.tool_calls);
        }
    }
}

test "ChatMLToolFormat.parseAssistantToolCalls extracts a single tool call" {
    const tf = chatmlToolFormat();
    const output =
        \\Let me check that.
        \\<tool_call>
        \\{"name": "Bash", "arguments": {"command": "ls /"}}
        \\</tool_call>
    ;
    const parsed = try tf.parseAssistantToolCalls(output, std.testing.allocator);
    defer {
        for (parsed.tool_calls) |c| {
            std.testing.allocator.free(c.id);
            std.testing.allocator.free(c.name);
            std.testing.allocator.free(c.arguments_json);
        }
        std.testing.allocator.free(parsed.tool_calls);
        std.testing.allocator.free(parsed.text_content);
    }

    try std.testing.expectEqualStrings("Let me check that.\n", parsed.text_content);
    try std.testing.expectEqual(@as(usize, 1), parsed.tool_calls.len);
    try std.testing.expect(parsed.tool_calls[0].id.len > 0);
    try std.testing.expectEqualStrings("Bash", parsed.tool_calls[0].name);
    try std.testing.expectEqualStrings(
        \\{"command":"ls /"}
    , parsed.tool_calls[0].arguments_json);
}

test "ChatMLToolFormat.parseAssistantToolCalls extracts multiple parallel calls" {
    const tf = chatmlToolFormat();
    const output =
        \\Let me run two commands.
        \\<tool_call>
        \\{"name": "Bash", "arguments": {"command": "ls /"}}
        \\</tool_call>
        \\<tool_call>
        \\{"name": "Read", "arguments": {"path": "/etc/hostname"}}
        \\</tool_call>
    ;
    const parsed = try tf.parseAssistantToolCalls(output, std.testing.allocator);
    defer {
        for (parsed.tool_calls) |c| {
            std.testing.allocator.free(c.id);
            std.testing.allocator.free(c.name);
            std.testing.allocator.free(c.arguments_json);
        }
        std.testing.allocator.free(parsed.tool_calls);
        std.testing.allocator.free(parsed.text_content);
    }

    try std.testing.expectEqual(@as(usize, 2), parsed.tool_calls.len);
    try std.testing.expect(parsed.tool_calls[0].id.len > 0);
    try std.testing.expectEqualStrings("Bash", parsed.tool_calls[0].name);
    try std.testing.expect(parsed.tool_calls[1].id.len > 0);
    try std.testing.expect(!std.mem.eql(u8, parsed.tool_calls[0].id, parsed.tool_calls[1].id));
    try std.testing.expectEqualStrings("Read", parsed.tool_calls[1].name);
    try std.testing.expectEqualStrings("Let me run two commands.\n", parsed.text_content);
}

test "ChatMLToolFormat.parseAssistantToolCalls falls back to text on malformed JSON" {
    const tf = chatmlToolFormat();
    const output =
        \\Let me try.
        \\<tool_call>
        \\{not valid json}
        \\</tool_call>
        \\<tool_call>
        \\{"name": "OK", "arguments": {}}
        \\</tool_call>
    ;
    const parsed = try tf.parseAssistantToolCalls(output, std.testing.allocator);
    defer {
        for (parsed.tool_calls) |c| {
            std.testing.allocator.free(c.id);
            std.testing.allocator.free(c.name);
            std.testing.allocator.free(c.arguments_json);
        }
        std.testing.allocator.free(parsed.tool_calls);
        std.testing.allocator.free(parsed.text_content);
    }

    // The valid call still extracted and gets a non-empty id.
    try std.testing.expectEqual(@as(usize, 1), parsed.tool_calls.len);
    try std.testing.expect(parsed.tool_calls[0].id.len > 0);
    try std.testing.expectEqualStrings("OK", parsed.tool_calls[0].name);
    // The malformed block (with tags) appears in text_content.
    try std.testing.expect(std.mem.indexOf(u8, parsed.text_content, "<tool_call>\n{not valid json}\n</tool_call>") != null);
}

test "ChatMLToolFormat.renderToolDefinitions emits Qwen3 verbatim tool prompt" {
    const tf = chatmlToolFormat();
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    const tools = [_]ToolDefinition{
        .{
            .name = "Bash",
            .description = "Execute a bash command.",
            .parameters_json =
            \\{"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}
            ,
        },
    };
    try tf.renderToolDefinitions(&tools, &buf, std.testing.allocator);

    try std.testing.expect(std.mem.indexOf(u8, buf.items, "# Tools") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "<tools>") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "</tools>") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"name\": \"Bash\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "\"description\": \"Execute a bash command.\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, buf.items, "<tool_call>") != null);
}

test "ChatMLToolFormat.renderToolResultMessage wraps content in tool_response tags" {
    const tf = chatmlToolFormat();
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    try tf.renderToolResultMessage("call_0", "exit code 0\nfile: hello.txt\n", &buf, std.testing.allocator);
    try std.testing.expectEqualStrings(
        \\<tool_response>
        \\exit code 0
        \\file: hello.txt
        \\</tool_response>
        \\
    , buf.items);
}

test "StreamingDetector.feed emits normal text as content" {
    const tf = chatmlToolFormat();
    const detector = try tf.newStreamingDetector(std.testing.allocator);
    defer {
        detector.deinit();
        std.testing.allocator.destroy(detector);
    }
    const result = try detector.feed("Hello, world!");
    try std.testing.expectEqual(FeedResult.emit_as_content, result);
}

test "StreamingDetector.feed preserves prefix content with a complete call" {
    const tf = chatmlToolFormat();
    const detector = try tf.newStreamingDetector(std.testing.allocator);
    defer {
        detector.deinit();
        std.testing.allocator.destroy(detector);
    }

    const chunk = "Sure!\n<tool_call>\n{\"name\":\"X\",\"arguments\":{}}\n</tool_call>";

    // The caller drains content before calls, including when both arrive in
    // the same token. No extra feed is needed at end of generation.
    const r1 = try detector.feed(chunk);
    try std.testing.expectEqual(FeedResult.tool_call_complete, r1);
    const content = detector.takeContentDelta();
    try std.testing.expectEqualStrings("Sure!\n", content);

    const call = detector.takePendingToolCall() orelse return error.NoCall;
    defer {
        std.testing.allocator.free(call.id);
        std.testing.allocator.free(call.name);
        std.testing.allocator.free(call.arguments_json);
    }
    try std.testing.expectEqualStrings("X", call.name);
    try std.testing.expect(call.id.len > 0);
}

test "StreamingDetector flushes false-positive partial tag as content" {
    const tf = chatmlToolFormat();
    const detector = try tf.newStreamingDetector(std.testing.allocator);
    defer {
        detector.deinit();
        std.testing.allocator.destroy(detector);
    }

    // First chunk ends with "<th" — might be start of <tool_call>
    _ = try detector.feed("Hello <th");

    // Second chunk "ink>" makes "<think>" which is NOT a tool_call.
    const r2 = try detector.feed("ink>");
    try std.testing.expectEqual(FeedResult.emit_as_content, r2);

    const content = detector.takeContentDelta();
    // Verify some bytes were flushed as content (exact split may vary)
    try std.testing.expect(content.len > 0);
}

test "forTemplate returns ChatMLToolFormat for chatml kind" {
    const tf = forTemplate(.chatml);
    var buf: std.ArrayList(u8) = .empty;
    defer buf.deinit(std.testing.allocator);
    const tools = [_]ToolDefinition{
        .{ .name = "f", .description = "d", .parameters_json = "{}" },
    };
    try tf.renderToolDefinitions(&tools, &buf, std.testing.allocator);
    // ChatML emits a non-empty buffer; Noop emits empty.
    try std.testing.expect(buf.items.len > 0);
}

test "forTemplate returns NoopToolFormat for non-chatml kinds" {
    inline for (.{ .llama3, .gemma, .openai_moe, .generic }) |kind| {
        const tf = forTemplate(kind);
        var buf: std.ArrayList(u8) = .empty;
        defer buf.deinit(std.testing.allocator);
        const tools = [_]ToolDefinition{
            .{ .name = "f", .description = "d", .parameters_json = "{}" },
        };
        try tf.renderToolDefinitions(&tools, &buf, std.testing.allocator);
        try std.testing.expectEqual(@as(usize, 0), buf.items.len);
    }
}

const xml_test_tools = [_]ToolDefinition{.{
    .name = "edit",
    .description = "Edit a file.",
    .parameters_json =
    \\{"type":"object","properties":{"filePath":{"type":"string"},"oldString":{"type":"string"},"newString":{"type":"string"},"offset":{"type":"integer"},"replaceAll":{"type":"boolean"},"options":{"type":"object"},"lines":{"type":"array"},"nullable":{"type":["integer","null"]},"flexible":{"anyOf":[{"type":"integer"},{"type":"string"}]}}}
    ,
}};

fn freeTestCall(allocator: std.mem.Allocator, call: ToolCall) void {
    allocator.free(call.id);
    allocator.free(call.name);
    allocator.free(call.arguments_json);
}

test "native XML arguments preserve source strings and schema types" {
    const a = std.testing.allocator;
    const wire =
        \\<tool_call>
        \\<function=edit>
        \\<parameter=filePath>
        \\src/export utils/表.mjs
        \\</parameter>
        \\<parameter=oldString>
        \\0
        \\</parameter>
        \\<parameter=newString>
        \\  const x = "café 東京 🚀";
        \\  return { a: true };
        \\</parameter>
        \\<parameter=offset>12</parameter>
        \\<parameter=replaceAll>false</parameter>
        \\<parameter=options>{"safe":true}</parameter>
        \\<parameter=lines>[1,"two"]</parameter>
        \\<parameter=nullable>null</parameter>
        \\<parameter=flexible>42</parameter>
        \\</function>
        \\</tool_call>
    ;
    const parsed = try chatmlToolFormat().parseAssistantToolCallsWithTools(wire, &xml_test_tools, a);
    defer a.free(parsed.text_content);
    defer a.free(parsed.tool_calls);
    defer for (parsed.tool_calls) |call| freeTestCall(a, call);
    try std.testing.expectEqual(@as(usize, 1), parsed.tool_calls.len);
    const args = try std.json.parseFromSlice(std.json.Value, a, parsed.tool_calls[0].arguments_json, .{});
    defer args.deinit();
    const obj = args.value.object;
    try std.testing.expectEqualStrings("src/export utils/表.mjs", obj.get("filePath").?.string);
    try std.testing.expectEqualStrings("0", obj.get("oldString").?.string);
    try std.testing.expectEqualStrings("  const x = \"café 東京 🚀\";\n  return { a: true };", obj.get("newString").?.string);
    try std.testing.expectEqual(@as(i64, 12), obj.get("offset").?.integer);
    try std.testing.expectEqual(false, obj.get("replaceAll").?.bool);
    try std.testing.expectEqual(true, obj.get("options").?.object.get("safe").?.bool);
    try std.testing.expectEqual(@as(usize, 2), obj.get("lines").?.array.items.len);
    try std.testing.expect(obj.get("nullable").? == .null);
    try std.testing.expectEqualStrings("42", obj.get("flexible").?.string);
}

test "native XML history round trip preserves multiline whitespace and JSON-looking strings" {
    const a = std.testing.allocator;
    const arguments = "{\"filePath\":\"src/表.mjs\",\"oldString\":\"\\n  old\\n\",\"newString\":\"{\\\"literal\\\":true}\\n\\n\",\"replaceAll\":true,\"offset\":0}";
    var wire: std.ArrayList(u8) = .empty;
    defer wire.deinit(a);
    try appendAssistantCall(&wire, a, "edit", arguments, true);
    const parsed = try chatmlToolFormat().parseAssistantToolCallsWithTools(wire.items, &xml_test_tools, a);
    defer a.free(parsed.text_content);
    defer a.free(parsed.tool_calls);
    defer for (parsed.tool_calls) |call| freeTestCall(a, call);
    try std.testing.expectEqual(@as(usize, 1), parsed.tool_calls.len);
    try std.testing.expectEqualStrings(arguments, parsed.tool_calls[0].arguments_json);
}

test "tool streams accept every byte split and multiple calls without losing chunks" {
    const a = std.testing.allocator;
    const wire = "Before\n<tool_call>\n<function=edit>\n<parameter=filePath>src/表.mjs</parameter>\n<parameter=replaceAll>true</parameter>\n</function>\n</tool_call>\n<tool_call>{\"name\":\"read\",\"arguments\":{\"filePath\":\"café </tool_call>\"}}</tool_call>After";
    for (0..wire.len + 1) |split| {
        var detector = StreamingDetector{ .allocator = a, .tools = &xml_test_tools };
        defer detector.deinit();
        _ = try detector.feed(wire[0..split]);
        // Deliberately feed again before draining to exercise queued calls.
        _ = try detector.feed(wire[split..]);
        const content = detector.takeContentDelta();
        try std.testing.expectEqualStrings("Before\n\nAfter", content);
        const first = detector.takePendingToolCall() orelse return error.MissingXmlCall;
        defer freeTestCall(a, first);
        const second = detector.takePendingToolCall() orelse return error.MissingJsonCall;
        defer freeTestCall(a, second);
        try std.testing.expectEqualStrings("edit", first.name);
        try std.testing.expectEqualStrings("{\"filePath\":\"src/表.mjs\",\"replaceAll\":true}", first.arguments_json);
        try std.testing.expectEqualStrings("read", second.name);
        try std.testing.expectEqualStrings("{\"filePath\":\"café </tool_call>\"}", second.arguments_json);
        try std.testing.expect(!std.mem.eql(u8, first.id, second.id));
        try std.testing.expectEqual(@as(?ToolCall, null), detector.takePendingToolCall());
        try std.testing.expectEqualStrings("", detector.finalize());
    }
}

test "malformed native XML stays visible and never invents tool arguments" {
    const a = std.testing.allocator;
    const malformed = [_][]const u8{
        "<function=edit><parameter=filePath>file</function>",
        "<function=edit><parameter=filePath>a</parameter><parameter=filePath>b</parameter></function>",
        "<function=edit><parameter=filePath>a</parameter></function>extra",
        "<function=><parameter=filePath>a</parameter></function>",
        "{\"function\":\"read\",\"filePath\": file}",
    };
    for (malformed) |inner| {
        const wire = try std.fmt.allocPrint(a, "<tool_call>{s}</tool_call>", .{inner});
        defer a.free(wire);
        var detector = StreamingDetector{ .allocator = a, .tools = &xml_test_tools };
        defer detector.deinit();
        _ = try detector.feed(wire);
        try std.testing.expectEqual(@as(?ToolCall, null), detector.takePendingToolCall());
        try std.testing.expectEqualStrings(wire, detector.finalize());
    }
}

test "native Qwen XML decodes Python boolean and null scalars only for typed fields" {
    const a = std.testing.allocator;
    for ([_][]const u8{ "True", "False" }, [_]bool{ true, false }) |spelling, expected| {
        const wire = try std.fmt.allocPrint(a, "<tool_call><function=edit><parameter=replaceAll>{s}</parameter><parameter=nullable>None</parameter><parameter=oldString>{s}</parameter></function></tool_call>", .{ spelling, spelling });
        defer a.free(wire);
        const parsed = try chatmlToolFormat().parseAssistantToolCallsWithTools(wire, &xml_test_tools, a);
        defer a.free(parsed.text_content);
        defer a.free(parsed.tool_calls);
        defer for (parsed.tool_calls) |call| freeTestCall(a, call);
        try std.testing.expectEqual(@as(usize, 1), parsed.tool_calls.len);
        const args = try std.json.parseFromSlice(std.json.Value, a, parsed.tool_calls[0].arguments_json, .{});
        defer args.deinit();
        try std.testing.expectEqual(expected, args.value.object.get("replaceAll").?.bool);
        try std.testing.expect(args.value.object.get("nullable").? == .null);
        try std.testing.expectEqualStrings(spelling, args.value.object.get("oldString").?.string);
    }
}

test "Gemma streaming detects byte-split multiple calls and preserves raw argument markers" {
    const a = std.testing.allocator;
    const tools = [_]ToolDefinition{.{ .name = "write", .description = "", .parameters_json = "{}" }};
    const detector = try gemmaToolFormat().newStreamingDetector(a);
    detector.tools = &tools;
    defer {
        detector.deinit();
        a.destroy(detector);
    }
    const input = "Prose. <|tool_call>call:write{code:<|\"|>東京\n<think><channel|><tool_call|><|\"|>}<tool_call|><|tool_call>call:write{code:<|\"|>second<|\"|>}<tool_call|> Done.";
    var content: std.ArrayList(u8) = .empty;
    defer content.deinit(a);
    var calls: std.ArrayList(ToolCall) = .empty;
    defer {
        for (calls.items) |call| {
            a.free(call.id);
            a.free(call.name);
            a.free(call.arguments_json);
        }
        calls.deinit(a);
    }
    for (input, 0..) |_, i| {
        _ = try detector.feed(input[i..][0..1]);
        try content.appendSlice(a, detector.takeContentDelta());
        while (detector.takePendingToolCall()) |call| try calls.append(a, call);
    }
    try content.appendSlice(a, detector.finalize());
    try std.testing.expectEqualStrings("Prose.  Done.", content.items);
    try std.testing.expectEqual(@as(usize, 2), calls.items.len);
    try std.testing.expect(!std.mem.eql(u8, calls.items[0].id, calls.items[1].id));
    const args = try std.json.parseFromSlice(std.json.Value, a, calls.items[0].arguments_json, .{});
    defer args.deinit();
    try std.testing.expectEqualStrings("東京\n<think><channel|><tool_call|>", args.value.object.get("code").?.string);
}

test "Gemma unknown and incomplete calls remain text" {
    const a = std.testing.allocator;
    const tools = [_]ToolDefinition{.{ .name = "read", .description = "", .parameters_json = "{}" }};
    const input = "<|tool_call>call:unknown{}<tool_call|><|tool_call>call:read{path:<|\"|>unfinished";
    const result = try gemmaToolFormat().parseAssistantToolCallsWithTools(input, &tools, a);
    defer {
        a.free(result.text_content);
        a.free(result.tool_calls);
    }
    try std.testing.expectEqual(@as(usize, 0), result.tool_calls.len);
    try std.testing.expectEqualStrings(input, result.text_content);
}
