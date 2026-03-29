//! Route dispatcher and endpoint handlers for the OpenAI-compatible API.
//! Handles /v1/chat/completions, /v1/completions, /v1/models, /health.
const std = @import("std");
const http = @import("http.zig");
const forward_mod = @import("../compute/forward.zig");
const tokenizer_mod = @import("../model/tokenizer.zig");
const Model = @import("../model/loader.zig").Model;

const log = std.log.scoped(.routes);

/// Handle one HTTP connection: parse request, dispatch to endpoint, send response.
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

fn handleChatCompletions(
    conn: *http.Connection,
    engine: *forward_mod.InferenceEngine,
    tokenizer: *tokenizer_mod.Tokenizer,
    model: *const Model,
    body: []const u8,
    allocator: std.mem.Allocator,
) !void {
    // Parse essential fields from JSON body
    const parsed = parseJsonFields(body) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON in request body");
        return;
    };

    if (parsed.messages_content.len == 0) {
        try conn.sendError(400, "invalid_request_error", "Field 'messages' is required");
        return;
    }

    // Apply chat template: wrap content with ChatML tags
    var prompt_buf: [8192]u8 = undefined;
    const prompt = std.fmt.bufPrint(&prompt_buf,
        \\<|im_start|>user
        \\{s}<|im_end|>
        \\<|im_start|>assistant
        \\
    , .{parsed.messages_content}) catch {
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

        // Decode loop: generate and stream one token at a time
        const eos = tokenizer.eosId();
        // Stop detection: accumulate generated text and check for <|im_end|>
        var gen_text_buf: [4096]u8 = undefined;
        var gen_text_len: usize = 0;
        const stop_str = "<|im_end|>";

        var generated: u32 = 0;
        if (prompt_tokens.len > 0 and max_tokens > 0) {
            const first = engine.sampleGreedy();
            generated = 1;
            if (first != eos) {
                const tok_text = if (first < tokenizer.vocab.len) tokenizer.vocab[first] else "";
                // Accumulate text for stop detection
                if (gen_text_len + tok_text.len < gen_text_buf.len) {
                    @memcpy(gen_text_buf[gen_text_len..][0..tok_text.len], tok_text);
                    gen_text_len += tok_text.len;
                }
                if (std.mem.indexOf(u8, gen_text_buf[0..gen_text_len], stop_str) != null) {
                    // Hit stop — don't send this token
                } else {
                    streamToken(conn, first, tokenizer, req_id, ts, model_name) catch return;
                }
            }
            var prev_token = first;
            while (generated < max_tokens and prev_token != eos) {
                // Check if we've hit the stop string
                if (std.mem.indexOf(u8, gen_text_buf[0..gen_text_len], stop_str) != null) break;

                engine.decodeStep(&state, prev_token) catch break;
                const next = engine.sampleGreedy();
                generated += 1;
                prev_token = next;
                if (next == eos) break;

                const tok_text = if (next < tokenizer.vocab.len) tokenizer.vocab[next] else "";
                if (gen_text_len + tok_text.len < gen_text_buf.len) {
                    @memcpy(gen_text_buf[gen_text_len..][0..tok_text.len], tok_text);
                    gen_text_len += tok_text.len;
                }
                if (std.mem.indexOf(u8, gen_text_buf[0..gen_text_len], stop_str) != null) break;

                streamToken(conn, next, tokenizer, req_id, ts, model_name) catch return;
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
        const ns_stop = "<|im_end|>";
        if (prompt_tokens.len > 0 and max_tokens > 0) {
            var prev = engine.sampleGreedy();
            ns_gen = 1;
            while (ns_gen <= max_tokens and prev != ns_eos) {
                var decode_buf2: [256]u8 = undefined;
                const tok_utf8 = tokenizer.decodeToken(prev, &decode_buf2);
                text_buf.appendSlice(allocator, tok_utf8) catch break;
                if (std.mem.indexOf(u8, text_buf.items, ns_stop)) |pos| {
                    text_buf.shrinkRetainingCapacity(pos);
                    break;
                }
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
            req_id, ts, model_name,
            escaped_text,
            prompt_tokens.len, ns_gen, prompt_tokens.len + ns_gen,
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
    const parsed = parseJsonFields(body) catch {
        try conn.sendError(400, "invalid_request_error", "Invalid JSON");
        return;
    };

    if (parsed.prompt_text.len == 0) {
        try conn.sendError(400, "invalid_request_error", "Field 'prompt' is required");
        return;
    }

    // Tokenize raw prompt (no chat template)
    const raw_tokens = tokenizer.encode(parsed.prompt_text) catch {
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
        req_id, ts, model_name,
        escaped_text,
        prompt_tokens.len, output_tokens.len, prompt_tokens.len + output_tokens.len,
    }) catch {
        try conn.sendError(500, "internal_error", "Response too large");
        return;
    };
    try conn.sendJson(200, resp);
}

// ── Helpers ──────────────────────────────────────────────────

/// Minimal JSON field extraction (no full parser — just find key fields).
const ParsedRequest = struct {
    messages_content: []const u8, // last user message content
    prompt_text: []const u8, // raw prompt for /v1/completions
    max_tokens: u32,
    stream: bool,
    temperature: f32,
};

fn parseJsonFields(body: []const u8) !ParsedRequest {
    var result = ParsedRequest{
        .messages_content = "",
        .prompt_text = "",
        .max_tokens = 256,
        .stream = false,
        .temperature = 1.0,
    };

    // Extract "stream":true/false
    if (std.mem.indexOf(u8, body, "\"stream\":true") != null or
        std.mem.indexOf(u8, body, "\"stream\": true") != null)
    {
        result.stream = true;
    }

    // Extract "max_tokens":N
    if (std.mem.indexOf(u8, body, "\"max_tokens\":")) |pos| {
        const start = pos + "\"max_tokens\":".len;
        const trimmed = std.mem.trim(u8, body[start..@min(start + 10, body.len)], " ");
        result.max_tokens = std.fmt.parseInt(u32, trimmed[0..findNumEnd(trimmed)], 10) catch 256;
    } else if (std.mem.indexOf(u8, body, "\"max_tokens\": ")) |pos| {
        const start = pos + "\"max_tokens\": ".len;
        result.max_tokens = std.fmt.parseInt(u32, body[start..@min(start + 10, body.len)][0..findNumEnd(body[start..@min(start + 10, body.len)])], 10) catch 256;
    }

    // Extract last "content":"..." from messages
    if (std.mem.lastIndexOf(u8, body, "\"content\":\"")) |pos| {
        const start = pos + "\"content\":\"".len;
        if (findStringEnd(body[start..])) |end| {
            result.messages_content = body[start .. start + end];
        }
    } else if (std.mem.lastIndexOf(u8, body, "\"content\": \"")) |pos| {
        const start = pos + "\"content\": \"".len;
        if (findStringEnd(body[start..])) |end| {
            result.messages_content = body[start .. start + end];
        }
    }

    // Extract "prompt":"..."
    if (std.mem.indexOf(u8, body, "\"prompt\":\"")) |pos| {
        const start = pos + "\"prompt\":\"".len;
        if (findStringEnd(body[start..])) |end| {
            result.prompt_text = body[start .. start + end];
        }
    } else if (std.mem.indexOf(u8, body, "\"prompt\": \"")) |pos| {
        const start = pos + "\"prompt\": \"".len;
        if (findStringEnd(body[start..])) |end| {
            result.prompt_text = body[start .. start + end];
        }
    }

    return result;
}

fn findNumEnd(s: []const u8) usize {
    for (s, 0..) |c, i| {
        if (c < '0' or c > '9') return i;
    }
    return s.len;
}

fn findStringEnd(s: []const u8) ?usize {
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\\') { i += 1; continue; } // skip escaped chars
        if (s[i] == '"') return i;
    }
    return null;
}

fn jsonEscape(input: []const u8, buf: []u8) []const u8 {
    var out: usize = 0;
    for (input) |c| {
        if (out + 2 >= buf.len) break;
        switch (c) {
            '"' => { buf[out] = '\\'; buf[out + 1] = '"'; out += 2; },
            '\\' => { buf[out] = '\\'; buf[out + 1] = '\\'; out += 2; },
            '\n' => { buf[out] = '\\'; buf[out + 1] = 'n'; out += 2; },
            '\r' => { buf[out] = '\\'; buf[out + 1] = 'r'; out += 2; },
            '\t' => { buf[out] = '\\'; buf[out + 1] = 't'; out += 2; },
            else => { buf[out] = c; out += 1; },
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

test "parseJsonFields extracts stream flag" {
    const body = "{\"model\":\"qwen\",\"messages\":[{\"role\":\"user\",\"content\":\"hello\"}],\"stream\":true}";
    const parsed = try parseJsonFields(body);
    try std.testing.expect(parsed.stream);
    try std.testing.expectEqualStrings("hello", parsed.messages_content);
}

test "parseJsonFields extracts max_tokens" {
    const body = "{\"model\":\"qwen\",\"prompt\":\"test\",\"max_tokens\":128}";
    const parsed = try parseJsonFields(body);
    try std.testing.expectEqual(@as(u32, 128), parsed.max_tokens);
    try std.testing.expectEqualStrings("test", parsed.prompt_text);
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

test "parseJsonFields defaults when fields missing" {
    const body = "{\"model\":\"qwen\"}";
    const parsed = try parseJsonFields(body);
    try std.testing.expect(!parsed.stream);
    try std.testing.expectEqual(@as(u32, 256), parsed.max_tokens);
    try std.testing.expectEqualStrings("", parsed.messages_content);
    try std.testing.expectEqualStrings("", parsed.prompt_text);
}

test "parseJsonFields stream false explicit" {
    const body = "{\"model\":\"qwen\",\"stream\":false}";
    const parsed = try parseJsonFields(body);
    try std.testing.expect(!parsed.stream);
}

test "parseJsonFields extracts content with spaces" {
    const body = "{\"model\":\"q\",\"messages\":[{\"role\":\"user\",\"content\": \"hello world\"}],\"stream\": true}";
    const parsed = try parseJsonFields(body);
    try std.testing.expect(parsed.stream);
    try std.testing.expectEqualStrings("hello world", parsed.messages_content);
}

test "parseJsonFields max_tokens with spaces" {
    const body = "{\"max_tokens\": 64}";
    const parsed = try parseJsonFields(body);
    try std.testing.expectEqual(@as(u32, 64), parsed.max_tokens);
}

test "findStringEnd handles escaped quotes" {
    // Input after opening quote: hello \"inner\" end"rest
    // Escaped \" at positions 6-7 and 14-15, real " at position 19
    const s = "hello \\\"inner\\\" end\"rest";
    const end = findStringEnd(s);
    try std.testing.expectEqual(@as(?usize, 19), end);
}

test "findNumEnd extracts digits" {
    try std.testing.expectEqual(@as(usize, 3), findNumEnd("123abc"));
    try std.testing.expectEqual(@as(usize, 0), findNumEnd("abc"));
    try std.testing.expectEqual(@as(usize, 5), findNumEnd("99999"));
}

test "parseJsonFields handles multiline content" {
    const body = "{\"messages\":[{\"role\":\"user\",\"content\":\"line1\\nline2\"}]}";
    const parsed = try parseJsonFields(body);
    try std.testing.expectEqualStrings("line1\\nline2", parsed.messages_content);
}

test "parseJsonFields handles multiple messages picks last content" {
    const body =
        \\{"messages":[{"role":"system","content":"sys"},{"role":"user","content":"usr"}]}
    ;
    const parsed = try parseJsonFields(body);
    // lastIndexOf should find the last "content" which is "usr"
    try std.testing.expectEqualStrings("usr", parsed.messages_content);
}

test "parseJsonFields max_tokens large value" {
    const body = "{\"max_tokens\":4096}";
    const parsed = try parseJsonFields(body);
    try std.testing.expectEqual(@as(u32, 4096), parsed.max_tokens);
}

test "parseJsonFields prompt with special chars" {
    const body = "{\"prompt\":\"What is 2+2?\"}";
    const parsed = try parseJsonFields(body);
    try std.testing.expectEqualStrings("What is 2+2?", parsed.prompt_text);
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

test "findStringEnd no closing quote returns null" {
    try std.testing.expectEqual(@as(?usize, null), findStringEnd("no close"));
}

test "findStringEnd immediate close" {
    try std.testing.expectEqual(@as(?usize, 0), findStringEnd("\"rest"));
}
