//! Route dispatcher and endpoint handlers for the OpenAI-compatible API.
//! @section API Server
//! Handles /v1/chat/completions, /v1/completions, /v1/models, /health,
//! and a built-in chat UI. Supports both streaming (SSE) and non-streaming responses.
const std = @import("std");
const http = @import("http.zig");
const runtime = @import("runtime.zig");
const forward_mod = runtime.forward_mod;
const tokenizer_mod = runtime.tokenizer_mod;
const Model = runtime.Model;

const log = std.log.scoped(.routes);

/// Handle one HTTP connection: parse request, dispatch to endpoint, send response.
/// @param conn Active client connection to read from and write to.
/// @param engine Inference engine for running generation.
/// @param tokenizer Tokenizer for prompt encoding and token decoding.
/// @param model Loaded model (used for model name in API responses).
/// @param allocator Allocator for per-request temporaries.
pub fn handleConnection(
    conn: *http.Connection,
    engine: *forward_mod.InferenceEngine,
    tokenizer: *tokenizer_mod.Tokenizer,
    model: *Model,
    allocator: std.mem.Allocator,
) !void {
    const request = conn.readRequest() catch |err| {
        log.warn("Failed to parse request: {s}", .{@errorName(err)});
        conn.sendError(400, "invalid_request_error", "Malformed HTTP request") catch {};
        return;
    };

    log.info("{s} {s}", .{ @tagName(request.method), request.path });

    // Route dispatch
    if (request.method == .GET and std.mem.eql(u8, request.path, "/health")) {
        try handleHealth(conn, engine, model);
    } else if (request.method == .GET and std.mem.eql(u8, request.path, "/v1/models")) {
        try handleModels(conn, model);
    } else if (request.method == .POST and std.mem.eql(u8, request.path, "/v1/chat/completions")) {
        try handleChatCompletions(conn, engine, tokenizer, model, request.body, allocator);
    } else if (request.method == .POST and std.mem.eql(u8, request.path, "/v1/completions")) {
        try handleCompletions(conn, engine, tokenizer, model, request.body, allocator);
    } else if (request.method == .OPTIONS) {
        // CORS preflight
        try conn.sendJson(200, "{}");
    } else if (request.method == .GET and (std.mem.eql(u8, request.path, "/") or std.mem.eql(u8, request.path, "/chat"))) {
        try serveChatUi(conn);
    } else {
        try conn.sendError(404, "not_found", "Unknown endpoint");
    }
}

// ── /health ──────────────────────────────────────────────────

fn handleHealth(conn: *http.Connection, engine: *const forward_mod.InferenceEngine, model: *const Model) !void {
    _ = engine;
    var buf: [1024]u8 = undefined;
    const model_name = modelName(model);
    const body = std.fmt.bufPrint(&buf,
        \\{{"status":"ok","model":"{s}"}}
    , .{model_name}) catch return error.BufferTooSmall;
    try conn.sendJson(200, body);
}

// ── /v1/models ───────────────────────────────────────────────

fn handleModels(conn: *http.Connection, model: *const Model) !void {
    var buf: [1024]u8 = undefined;
    const model_name = modelName(model);
    const ts = @divTrunc(std.time.timestamp(), 1);
    const body = std.fmt.bufPrint(&buf,
        \\{{"object":"list","data":[{{"id":"{s}","object":"model","created":{d},"owned_by":"zinc"}}]}}
    , .{ model_name, ts }) catch return error.BufferTooSmall;
    try conn.sendJson(200, body);
}

// ── /v1/chat/completions ─────────────────────────────────────

const chat_stop_strs = [_][]const u8{
    "<|im_end|>",
    "<|im_start|>",
};

fn findFirstStop(text: []const u8, stop_strs: []const []const u8) ?usize {
    var first: ?usize = null;
    for (stop_strs) |stop| {
        if (std.mem.indexOf(u8, text, stop)) |pos| {
            if (first == null or pos < first.?) {
                first = pos;
            }
        }
    }
    return first;
}

fn handleChatCompletions(
    conn: *http.Connection,
    engine: *forward_mod.InferenceEngine,
    tokenizer: *tokenizer_mod.Tokenizer,
    model: *const Model,
    body: []const u8,
    allocator: std.mem.Allocator,
) !void {
    var parsed = parseRequestBody(body, allocator) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON in request body");
        return;
    };
    defer parsed.deinit();

    const messages = parsed.messages;
    if (messages.len == 0) {
        try conn.sendError(400, "invalid_request_error", "Field 'messages' is required");
        return;
    }

    const roles = allocator.alloc([]const u8, messages.len) catch {
        try conn.sendError(500, "internal_error", "Out of memory");
        return;
    };
    defer allocator.free(roles);
    const contents = allocator.alloc([]const u8, messages.len) catch {
        try conn.sendError(500, "internal_error", "Out of memory");
        return;
    };
    defer allocator.free(contents);

    var prompt_cap: usize = 32;
    for (messages, 0..) |msg, i| {
        roles[i] = normalizeRole(msg.role);
        contents[i] = msg.content;
        prompt_cap += roles[i].len + msg.content.len + 32;
    }

    const prompt_buf = allocator.alloc(u8, prompt_cap) catch {
        try conn.sendError(500, "internal_error", "Out of memory");
        return;
    };
    defer allocator.free(prompt_buf);

    const prompt = tokenizer.applyChatTemplate(roles, contents, prompt_buf) catch {
        try conn.sendError(400, "invalid_request_error", "Prompt too long");
        return;
    };

    // Tokenize
    const raw_tokens = tokenizer.encode(prompt) catch {
        try conn.sendError(500, "internal_error", "Tokenization failed");
        return;
    };
    defer allocator.free(raw_tokens);

    const prompt_tokens = allocator.alloc(u32, raw_tokens.len + 1) catch {
        try conn.sendError(500, "internal_error", "Out of memory");
        return;
    };
    defer allocator.free(prompt_tokens);
    prompt_tokens[0] = tokenizer.bosId();
    @memcpy(prompt_tokens[1..], raw_tokens);
    const model_name = modelName(model);
    const ts = @divTrunc(std.time.timestamp(), 1);
    const max_tokens = parsed.max_tokens;
    const req_id = "chatcmpl-zinc0001"; // TODO: T013 unique IDs

    if (parsed.stream) {
        // Streaming path — per-token SSE delivery
        conn.sendSseStart() catch return;

        // Send first chunk with role
        {
            var chunk_buf: [1024]u8 = undefined;
            const chunk = std.fmt.bufPrint(&chunk_buf,
                \\{{"id":"{s}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"role":"assistant"}},"finish_reason":null}}]}}
            , .{ req_id, ts, model_name }) catch return;
            conn.writeSseEvent(chunk) catch return;
        }

        // Prefill prompt tokens
        var state = forward_mod.DecodeState.init(allocator);
        defer state.deinit();
        engine.prefillBatch(&state, prompt_tokens) catch {
            conn.writeSseDone() catch {};
            return;
        };

        // Decode loop with buffered stop detection.
        // Tokens are buffered and only sent once we confirm they're not part of <|im_end|>.
        const eos = tokenizer.eosId();
        const stop_strs = chat_stop_strs[0..];
        var pending_tokens: [16]u32 = undefined; // tokens waiting to be sent
        var pending_count: usize = 0;
        var gen_text_buf: [4096]u8 = undefined; // accumulated decoded text for stop check
        var gen_text_len: usize = 0;
        var sent_text_len: usize = 0; // how much of gen_text has been confirmed safe to send
        var stopped = false;

        var generated: u32 = 0;
        if (prompt_tokens.len > 0 and max_tokens > 0) {
            var prev_token = engine.sampleGreedy();
            generated = 1;

            while (generated <= max_tokens and prev_token != eos and !stopped) {
                // Accumulate this token's decoded text
                var dec_buf: [256]u8 = undefined;
                const tok_text = tokenizer.decodeToken(prev_token, &dec_buf);
                if (gen_text_len + tok_text.len < gen_text_buf.len) {
                    @memcpy(gen_text_buf[gen_text_len..][0..tok_text.len], tok_text);
                    gen_text_len += tok_text.len;
                }

                // Add to pending queue
                if (pending_count < pending_tokens.len) {
                    pending_tokens[pending_count] = prev_token;
                    pending_count += 1;
                }

                // Check for full stop match against all stop strings
                if (findFirstStop(gen_text_buf[0..gen_text_len], stop_strs)) |_| {
                    stopped = true;
                }
                if (stopped) break;

                // Check if any suffix could be a prefix of any stop string
                var is_partial = false;
                for (stop_strs) |ss| {
                    const check_len = @min(gen_text_len, ss.len - 1);
                    var sl: usize = 1;
                    while (sl <= check_len) : (sl += 1) {
                        const suffix = gen_text_buf[gen_text_len - sl .. gen_text_len];
                        if (std.mem.startsWith(u8, ss, suffix)) {
                            is_partial = true;
                            break;
                        }
                    }
                    if (is_partial) break;
                }

                if (!is_partial) {
                    // Safe to send all pending tokens
                    for (pending_tokens[0..pending_count]) |tid| {
                        streamToken(conn, tid, tokenizer, req_id, ts, model_name) catch return;
                    }
                    pending_count = 0;
                    sent_text_len = gen_text_len;
                }

                // Generate next token
                if (generated < max_tokens) {
                    engine.decodeStep(&state, prev_token) catch break;
                    prev_token = engine.sampleGreedy();
                    generated += 1;
                } else break;
            }

            // Flush any remaining pending tokens (only if we didn't hit stop)
            if (!stopped) {
                for (pending_tokens[0..pending_count]) |tid| {
                    streamToken(conn, tid, tokenizer, req_id, ts, model_name) catch return;
                }
            }
        }

        // Final chunk with finish_reason
        {
            var chunk_buf: [1024]u8 = undefined;
            const chunk = std.fmt.bufPrint(&chunk_buf,
                \\{{"id":"{s}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}
            , .{ req_id, ts, model_name }) catch "";
            conn.writeSseEvent(chunk) catch return;
        }

        conn.writeSseDone() catch return;
    } else {
        // Non-streaming: use same prefill+decode loop with stop detection
        var state2 = forward_mod.DecodeState.init(allocator);
        defer state2.deinit();
        engine.prefillBatch(&state2, prompt_tokens) catch {
            try conn.sendError(500, "internal_error", "Prefill failed");
            return;
        };
        var text_buf: std.ArrayList(u8) = .{};
        defer text_buf.deinit(allocator);
        var ns_gen: u32 = 0;
        const ns_eos = tokenizer.eosId();
        const ns_stops = chat_stop_strs[0..];
        if (prompt_tokens.len > 0 and max_tokens > 0) {
            var prev = engine.sampleGreedy();
            ns_gen = 1;
            while (ns_gen <= max_tokens and prev != ns_eos) {
                var decode_buf2: [256]u8 = undefined;
                const tok_utf8 = tokenizer.decodeToken(prev, &decode_buf2);
                text_buf.appendSlice(allocator, tok_utf8) catch break;
                const hit = if (findFirstStop(text_buf.items, ns_stops)) |pos| blk: {
                    text_buf.shrinkRetainingCapacity(pos);
                    break :blk true;
                } else false;
                if (hit) break;
                engine.decodeStep(&state2, prev) catch break;
                prev = engine.sampleGreedy();
                ns_gen += 1;
            }
        }

        // Escape the full text for JSON
        var escaped_buf: [16384]u8 = undefined;
        const escaped_text = jsonEscape(text_buf.items, &escaped_buf);

        var resp_buf: [32768]u8 = undefined;
        const resp = std.fmt.bufPrint(&resp_buf,
            \\{{"id":"{s}","object":"chat.completion","created":{d},"model":"{s}","choices":[{{"index":0,"message":{{"role":"assistant","content":"{s}"}},"finish_reason":"stop"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
        , .{
            req_id,                     ts,                model_name,
            escaped_text,               prompt_tokens.len, ns_gen,
            prompt_tokens.len + ns_gen,
        }) catch {
            try conn.sendError(500, "internal_error", "Response too large");
            return;
        };
        try conn.sendJson(200, resp);
    }
}

// ── /v1/completions ──────────────────────────────────────────

fn handleCompletions(
    conn: *http.Connection,
    engine: *forward_mod.InferenceEngine,
    tokenizer: *tokenizer_mod.Tokenizer,
    model: *const Model,
    body: []const u8,
    allocator: std.mem.Allocator,
) !void {
    var parsed = parseRequestBody(body, allocator) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON");
        return;
    };
    defer parsed.deinit();

    if (parsed.prompt.len == 0) {
        try conn.sendError(400, "invalid_request_error", "Field 'prompt' is required");
        return;
    }

    // Tokenize raw prompt (no chat template)
    const raw_tokens = tokenizer.encode(parsed.prompt) catch {
        try conn.sendError(500, "internal_error", "Tokenization failed");
        return;
    };
    defer allocator.free(raw_tokens);

    const prompt_tokens = allocator.alloc(u32, raw_tokens.len + 1) catch {
        try conn.sendError(500, "internal_error", "Out of memory");
        return;
    };
    defer allocator.free(prompt_tokens);
    prompt_tokens[0] = tokenizer.bosId();
    @memcpy(prompt_tokens[1..], raw_tokens);

    const model_name = modelName(model);
    const ts = @divTrunc(std.time.timestamp(), 1);
    const req_id = "cmpl-zinc0001";

    const output_tokens = forward_mod.generate(engine, prompt_tokens, parsed.max_tokens, tokenizer.eosId(), allocator) catch {
        try conn.sendError(500, "internal_error", "Generation failed");
        return;
    };
    defer allocator.free(output_tokens);

    var text_buf: std.ArrayList(u8) = .{};
    defer text_buf.deinit(allocator);
    for (output_tokens) |tid| {
        const t = if (tid < tokenizer.vocab.len) tokenizer.vocab[tid] else "<?>";
        text_buf.appendSlice(allocator, t) catch break;
    }

    var escaped_buf: [16384]u8 = undefined;
    const escaped_text = jsonEscape(text_buf.items, &escaped_buf);

    var resp_buf: [32768]u8 = undefined;
    const resp = std.fmt.bufPrint(&resp_buf,
        \\{{"id":"{s}","object":"text_completion","created":{d},"model":"{s}","choices":[{{"index":0,"text":"{s}","finish_reason":"stop"}}],"usage":{{"prompt_tokens":{d},"completion_tokens":{d},"total_tokens":{d}}}}}
    , .{
        req_id,                                ts,                model_name,
        escaped_text,                          prompt_tokens.len, output_tokens.len,
        prompt_tokens.len + output_tokens.len,
    }) catch {
        try conn.sendError(500, "internal_error", "Response too large");
        return;
    };
    try conn.sendJson(200, resp);
}

// ── Helpers ──────────────────────────────────────────────────

const RawMessage = struct {
    role: []const u8 = "",
    content: []const u8 = "",
};

const RawRequestBody = struct {
    messages: ?[]RawMessage = null,
    prompt: []const u8 = "",
    max_tokens: u32 = 256,
    stream: bool = false,
    temperature: f32 = 1.0,
};

const JsonMessage = struct {
    role: []u8 = &[_]u8{},
    content: []u8 = &[_]u8{},
};

const ParsedRequest = struct {
    messages: []JsonMessage = &[_]JsonMessage{},
    prompt: []u8 = &[_]u8{},
    max_tokens: u32 = 256,
    stream: bool = false,
    temperature: f32 = 1.0,
    allocator: std.mem.Allocator,

    fn deinit(self: *ParsedRequest) void {
        for (self.messages) |msg| {
            if (msg.role.len > 0) self.allocator.free(msg.role);
            if (msg.content.len > 0) self.allocator.free(msg.content);
        }
        if (self.messages.len > 0) self.allocator.free(self.messages);
        if (self.prompt.len > 0) self.allocator.free(self.prompt);
        self.* = undefined;
    }
};

fn parseRequestBody(body: []const u8, allocator: std.mem.Allocator) !ParsedRequest {
    const parsed = try std.json.parseFromSlice(RawRequestBody, allocator, body, .{
        .ignore_unknown_fields = true,
        .duplicate_field_behavior = .use_last,
        .allocate = .alloc_always,
    });
    defer parsed.deinit();

    var result = ParsedRequest{
        .allocator = allocator,
    };
    errdefer result.deinit();

    result.max_tokens = parsed.value.max_tokens;
    result.stream = parsed.value.stream;
    result.temperature = parsed.value.temperature;
    result.prompt = try decodeJsonText(allocator, parsed.value.prompt);

    if (parsed.value.messages) |messages| {
        result.messages = try allocator.alloc(JsonMessage, messages.len);
        for (messages, 0..) |msg, i| {
            result.messages[i] = .{
                .role = try decodeJsonText(allocator, msg.role),
                .content = try decodeJsonText(allocator, msg.content),
            };
        }
    }

    return result;
}

fn decodeJsonText(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    if (input.len == 0) return &[_]u8{};

    var out: std.ArrayList(u8) = .{};
    defer out.deinit(allocator);

    var i: usize = 0;
    while (i < input.len) : (i += 1) {
        if (input[i] != '\\' or i + 1 >= input.len) {
            try out.append(allocator, input[i]);
            continue;
        }

        i += 1;
        switch (input[i]) {
            '"' => try out.append(allocator, '"'),
            '\\' => try out.append(allocator, '\\'),
            '/' => try out.append(allocator, '/'),
            'b' => try out.append(allocator, 0x08),
            'f' => try out.append(allocator, 0x0c),
            'n' => try out.append(allocator, '\n'),
            'r' => try out.append(allocator, '\r'),
            't' => try out.append(allocator, '\t'),
            else => try out.append(allocator, input[i]),
        }
    }

    return out.toOwnedSlice(allocator);
}

fn normalizeRole(role: []const u8) []const u8 {
    if (std.mem.eql(u8, role, "system") or
        std.mem.eql(u8, role, "user") or
        std.mem.eql(u8, role, "assistant"))
    {
        return role;
    }
    return "user";
}

test "decodeJsonText handles common escape sequences" {
    const decoded = try decodeJsonText(std.testing.allocator, "hello\\n\\\"world\\\"\\\\");
    defer if (decoded.len > 0) std.testing.allocator.free(decoded);
    try std.testing.expectEqualStrings("hello\n\"world\"\\", decoded);
}

test "decodeJsonText empty string" {
    const decoded = try decodeJsonText(std.testing.allocator, "");
    defer if (decoded.len > 0) std.testing.allocator.free(decoded);
    try std.testing.expectEqual(@as(usize, 0), decoded.len);
}

test "parseRequestBody extracts stream flag and user message" {
    const body = "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}";
    var parsed = try parseRequestBody(body, std.testing.allocator);
    defer parsed.deinit();
    try std.testing.expect(parsed.stream);
    try std.testing.expectEqual(@as(usize, 1), parsed.messages.len);
    try std.testing.expectEqualStrings("user", parsed.messages[0].role);
    try std.testing.expectEqualStrings("hello", parsed.messages[0].content);
}

test "parseRequestBody extracts max_tokens and prompt" {
    const body = "{\"model\":\"qwen\",\"prompt\":\"test\",\"max_tokens\":128}";
    var parsed = try parseRequestBody(body, std.testing.allocator);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(u32, 128), parsed.max_tokens);
    try std.testing.expectEqualStrings("test", parsed.prompt);
}

test "parseRequestBody defaults when fields missing" {
    const body = "{\"model\":\"qwen\"}";
    var parsed = try parseRequestBody(body, std.testing.allocator);
    defer parsed.deinit();
    try std.testing.expect(!parsed.stream);
    try std.testing.expectEqual(@as(u32, 256), parsed.max_tokens);
    try std.testing.expectEqual(@as(usize, 0), parsed.messages.len);
    try std.testing.expectEqualStrings("", parsed.prompt);
}

test "parseRequestBody handles escaped content and multiple messages" {
    const body =
        \\{"messages":[{"role":"assistant","content":"literal \"content\":\"noise\""},{"role":"user","content":"line1\\nline2"}],"stream":true}
    ;
    var parsed = try parseRequestBody(body, std.testing.allocator);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 2), parsed.messages.len);
    try std.testing.expectEqualStrings("assistant", parsed.messages[0].role);
    try std.testing.expectEqualStrings("literal \"content\":\"noise\"", parsed.messages[0].content);
    try std.testing.expectEqualStrings("line1\nline2", parsed.messages[1].content);
}

test "normalizeRole falls back to user" {
    try std.testing.expectEqualStrings("user", normalizeRole("tool"));
    try std.testing.expectEqualStrings("assistant", normalizeRole("assistant"));
}

test "ParsedRequest defaults" {
    var result = ParsedRequest{
        .allocator = std.testing.allocator,
    };
    defer result.deinit();
    try std.testing.expectEqual(@as(usize, 0), result.messages.len);
    try std.testing.expectEqual(@as(usize, 0), result.prompt.len);
    try std.testing.expectEqual(@as(u32, 256), result.max_tokens);
    try std.testing.expect(!result.stream);
    try std.testing.expectEqual(@as(f32, 1.0), result.temperature);
}

fn jsonEscape(input: []const u8, buf: []u8) []const u8 {
    var out: usize = 0;
    for (input) |c| {
        if (out + 2 >= buf.len) break;
        switch (c) {
            '"' => {
                buf[out] = '\\';
                buf[out + 1] = '"';
                out += 2;
            },
            '\\' => {
                buf[out] = '\\';
                buf[out + 1] = '\\';
                out += 2;
            },
            '\n' => {
                buf[out] = '\\';
                buf[out + 1] = 'n';
                out += 2;
            },
            '\r' => {
                buf[out] = '\\';
                buf[out + 1] = 'r';
                out += 2;
            },
            '\t' => {
                buf[out] = '\\';
                buf[out + 1] = 't';
                out += 2;
            },
            else => {
                buf[out] = c;
                out += 1;
            },
        }
    }
    return buf[0..out];
}

/// Send a single token as an SSE ChatCompletionChunk event.
fn streamToken(
    conn: *http.Connection,
    token_id: u32,
    tokenizer: *const tokenizer_mod.Tokenizer,
    req_id: []const u8,
    ts: i64,
    model_name: []const u8,
) !void {
    // Decode GPT-2 byte encoding to real UTF-8
    var decode_buf: [256]u8 = undefined;
    const token_text = tokenizer.decodeToken(token_id, &decode_buf);
    var escaped_buf: [512]u8 = undefined;
    const escaped = jsonEscape(token_text, &escaped_buf);
    var chunk_buf: [1024]u8 = undefined;
    const chunk = std.fmt.bufPrint(&chunk_buf,
        \\{{"id":"{s}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"content":"{s}"}},"finish_reason":null}}]}}
    , .{ req_id, ts, model_name, escaped }) catch return error.BufferTooSmall;
    try conn.writeSseEvent(chunk);
}

// ── Built-in Chat UI ─────────────────────────────────────────

fn serveChatUi(conn: *http.Connection) !void {
    const html = @embedFile("chat.html");
    var buf: [256]u8 = undefined;
    const header = std.fmt.bufPrint(&buf, "HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{html.len}) catch return error.HeaderTooLarge;
    try conn.stream.writeAll(header);
    try conn.stream.writeAll(html);
}

fn modelName(model: *const Model) []const u8 {
    _ = model;
    return "qwen3.5-35b"; // TODO: derive from GGUF metadata
}

test "jsonEscape handles special characters" {
    var buf: [64]u8 = undefined;
    const result = jsonEscape("hello \"world\"\nfoo", &buf);
    try std.testing.expectEqualStrings("hello \\\"world\\\"\\nfoo", result);
}

test "jsonEscape handles tabs and backslashes" {
    var buf: [64]u8 = undefined;
    const result = jsonEscape("a\\b\tc", &buf);
    try std.testing.expectEqualStrings("a\\\\b\\tc", result);
}

test "jsonEscape empty string" {
    var buf: [64]u8 = undefined;
    const result = jsonEscape("", &buf);
    try std.testing.expectEqualStrings("", result);
}

test "jsonEscape carriage return" {
    var buf: [32]u8 = undefined;
    const result = jsonEscape("a\rb", &buf);
    try std.testing.expectEqualStrings("a\\rb", result);
}

test "jsonEscape plain ASCII passthrough" {
    var buf: [64]u8 = undefined;
    const result = jsonEscape("Hello, World! 123", &buf);
    try std.testing.expectEqualStrings("Hello, World! 123", result);
}

test "findFirstStop picks earliest chat control marker" {
    const text = "Hello<|im_start|>assistant<|im_end|>";
    try std.testing.expectEqual(@as(?usize, 5), findFirstStop(text, chat_stop_strs[0..]));
}

test "findFirstStop returns null when no chat stop marker exists" {
    try std.testing.expectEqual(@as(?usize, null), findFirstStop("Hello there", chat_stop_strs[0..]));
}
