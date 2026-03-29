//! Route dispatcher and endpoint handlers for the OpenAI-compatible API.
//! @section API Server
//! Handles /v1/chat/completions, /v1/completions, /v1/models, /health,
//! and a built-in chat UI. Supports both streaming (SSE) and non-streaming responses.
const std = @import("std");
const http = @import("http.zig");
const forward_mod = @import("../compute/forward.zig");
const tokenizer_mod = @import("../model/tokenizer.zig");
const Model = @import("../model/loader.zig").Model;

const log = std.log.scoped(.routes);

pub const ServerState = struct {
    started_at: i64,
    active_requests: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    queued_requests: std.atomic.Value(u32) = std.atomic.Value(u32).init(0),
    generation_mutex: std.Thread.Mutex = .{},

    pub fn init(started_at: i64) ServerState {
        return .{ .started_at = started_at };
    }

    pub fn uptimeSeconds(self: *const ServerState, now: i64) u64 {
        return @intCast(@max(now - self.started_at, 0));
    }

    pub fn snapshot(self: *const ServerState, now: i64) HealthSnapshot {
        return .{
            .active_requests = self.active_requests.load(.monotonic),
            .queued_requests = self.queued_requests.load(.monotonic),
            .uptime_seconds = self.uptimeSeconds(now),
        };
    }
};

const HealthSnapshot = struct {
    active_requests: u32,
    queued_requests: u32,
    uptime_seconds: u64,
};

const GenerationGuard = struct {
    state: *ServerState,

    fn acquire(state: *ServerState) GenerationGuard {
        _ = state.queued_requests.fetchAdd(1, .monotonic);
        state.generation_mutex.lock();
        _ = state.queued_requests.fetchSub(1, .monotonic);
        _ = state.active_requests.fetchAdd(1, .monotonic);
        return .{ .state = state };
    }

    fn release(self: *GenerationGuard) void {
        _ = self.state.active_requests.fetchSub(1, .monotonic);
        self.state.generation_mutex.unlock();
    }
};

/// Handle one HTTP connection: parse request, dispatch to endpoint, send response.
/// @param conn Active client connection to read from and write to.
/// @param engine Inference engine for running generation.
/// @param tokenizer Tokenizer for prompt encoding and token decoding.
/// @param model Loaded model (used for model name in API responses).
/// @param server_state Shared server metrics and generation lock.
/// @param allocator Allocator for per-request temporaries.
pub fn handleConnection(
    conn: *http.Connection,
    engine: *forward_mod.InferenceEngine,
    tokenizer: *tokenizer_mod.Tokenizer,
    model: *Model,
    server_state: *ServerState,
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
        try handleHealth(conn, model, server_state);
    } else if (request.method == .GET and std.mem.eql(u8, request.path, "/v1/models")) {
        try handleModels(conn, model);
    } else if (request.method == .POST and std.mem.eql(u8, request.path, "/v1/chat/completions")) {
        try handleChatCompletions(conn, engine, tokenizer, model, server_state, request.body, allocator);
    } else if (request.method == .POST and std.mem.eql(u8, request.path, "/v1/completions")) {
        try handleCompletions(conn, engine, tokenizer, model, server_state, request.body, allocator);
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

fn buildHealthJson(server_state: *const ServerState, model_name: []const u8, buf: []u8) ![]const u8 {
    const now = std.time.timestamp();
    const snapshot = server_state.snapshot(now);
    return std.fmt.bufPrint(buf,
        \\{{"status":"ok","model":"{s}","active_requests":{d},"queued_requests":{d},"uptime_seconds":{d}}}
    , .{ model_name, snapshot.active_requests, snapshot.queued_requests, snapshot.uptime_seconds });
}

fn handleHealth(conn: *http.Connection, model: *const Model, server_state: *const ServerState) !void {
    var buf: [1024]u8 = undefined;
    const body = buildHealthJson(server_state, modelName(model), &buf) catch return error.BufferTooSmall;
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

fn buildChatPrompt(tokenizer: *const tokenizer_mod.Tokenizer, user_content: []const u8, buf: []u8) ![]const u8 {
    const roles = [_][]const u8{"user"};
    const contents = [_][]const u8{user_content};
    return tokenizer.applyChatTemplate(&roles, &contents, buf);
}

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
    server_state: *ServerState,
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

    var prompt_buf: [8192]u8 = undefined;
    const prompt = buildChatPrompt(tokenizer, parsed.messages_content, &prompt_buf) catch |err| {
        if (err == error.BufferTooSmall) {
            try conn.sendError(400, "invalid_request_error", "Prompt too long");
            return;
        }
        try conn.sendError(500, "internal_error", "Prompt formatting failed");
        return;
    };

    // Tokenize
    const raw_tokens = tokenizer.encode(prompt) catch {
        try conn.sendError(500, "internal_error", "Tokenization failed");
        return;
    };
    defer allocator.free(raw_tokens);

    const prepend_bos = tokenizer.shouldPrependBos();
    const bos_extra: usize = if (prepend_bos) 1 else 0;
    const prompt_tokens = allocator.alloc(u32, raw_tokens.len + bos_extra) catch {
        try conn.sendError(500, "internal_error", "Out of memory");
        return;
    };
    defer allocator.free(prompt_tokens);
    if (prepend_bos) {
        prompt_tokens[0] = tokenizer.bosId();
        @memcpy(prompt_tokens[1..], raw_tokens);
    } else {
        @memcpy(prompt_tokens, raw_tokens);
    }

    const model_name = modelName(model);
    const ts = @divTrunc(std.time.timestamp(), 1);
    const max_tokens = parsed.max_tokens;
    const req_id = "chatcmpl-zinc0001"; // TODO: T013 unique IDs
    var generation_guard = GenerationGuard.acquire(server_state);
    defer generation_guard.release();

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
        if (conn.isPeerClosed()) return;

        // Prefill prompt tokens
        var state = forward_mod.DecodeState.init(allocator);
        defer state.deinit();
        engine.prefillBatch(&state, prompt_tokens) catch {
            conn.writeSseDone() catch {};
            return;
        };
        if (conn.isPeerClosed()) return;

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
                if (conn.isPeerClosed()) return;

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
                    if (conn.isPeerClosed()) return;
                    engine.decodeStep(&state, prev_token) catch break;
                    if (conn.isPeerClosed()) return;
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
    server_state: *ServerState,
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

    const prepend_bos = tokenizer.shouldPrependBos();
    const bos_extra: usize = if (prepend_bos) 1 else 0;
    const prompt_tokens = allocator.alloc(u32, raw_tokens.len + bos_extra) catch {
        try conn.sendError(500, "internal_error", "Out of memory");
        return;
    };
    defer allocator.free(prompt_tokens);
    if (prepend_bos) {
        prompt_tokens[0] = tokenizer.bosId();
        @memcpy(prompt_tokens[1..], raw_tokens);
    } else {
        @memcpy(prompt_tokens, raw_tokens);
    }

    const model_name = modelName(model);
    const ts = @divTrunc(std.time.timestamp(), 1);
    const req_id = "cmpl-zinc0001";
    var generation_guard = GenerationGuard.acquire(server_state);
    defer generation_guard.release();

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
        if (s[i] == '\\') {
            i += 1;
            continue;
        } // skip escaped chars
        if (s[i] == '"') return i;
    }
    return null;
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

fn fallbackModelName(model: *const Model) []const u8 {
    return switch (model.config.architecture) {
        .qwen35 => "qwen3.5",
        .qwen2_moe => "qwen3.5-35b",
        .qwen2 => "qwen2",
        .llama => "llama",
        .mistral => "mistral",
        .mamba => "mamba",
        .jamba => "jamba",
        .unknown => "zinc-model",
    };
}

fn modelName(model: *const Model) []const u8 {
    return model.gguf_file.getString("general.basename") orelse
        model.gguf_file.getString("general.name") orelse
        fallbackModelName(model);
}

fn makeTestTokenizer(chat_template: ?[]const u8) tokenizer_mod.Tokenizer {
    return .{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .chat_template = chat_template,
        .allocator = std.testing.allocator,
    };
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

test "buildChatPrompt uses tokenizer chat template helper" {
    var tok = makeTestTokenizer(null);
    defer tok.token_to_id.deinit();

    var buf: [256]u8 = undefined;
    const prompt = try buildChatPrompt(&tok, "hello", &buf);
    try std.testing.expectEqualStrings("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n", prompt);
}

test "ServerState snapshot tracks active queued and uptime" {
    var state = ServerState.init(100);
    _ = state.active_requests.fetchAdd(1, .monotonic);
    _ = state.queued_requests.fetchAdd(2, .monotonic);

    const snapshot = state.snapshot(112);
    try std.testing.expectEqual(@as(u32, 1), snapshot.active_requests);
    try std.testing.expectEqual(@as(u32, 2), snapshot.queued_requests);
    try std.testing.expectEqual(@as(u64, 12), snapshot.uptime_seconds);
}

test "buildHealthJson includes request counts and uptime" {
    var state = ServerState.init(std.time.timestamp() - 5);
    _ = state.active_requests.fetchAdd(1, .monotonic);
    _ = state.queued_requests.fetchAdd(1, .monotonic);

    var buf: [256]u8 = undefined;
    const body = try buildHealthJson(&state, "qwen3.5-35b", &buf);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"status\":\"ok\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"model\":\"qwen3.5-35b\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"active_requests\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"queued_requests\":1") != null);
    try std.testing.expect(std.mem.indexOf(u8, body, "\"uptime_seconds\":") != null);
}

test "findFirstStop picks earliest chat control marker" {
    const text = "Hello<|im_start|>assistant<|im_end|>";
    try std.testing.expectEqual(@as(?usize, 5), findFirstStop(text, chat_stop_strs[0..]));
}

test "findFirstStop returns null when no chat stop marker exists" {
    try std.testing.expectEqual(@as(?usize, null), findFirstStop("Hello there", chat_stop_strs[0..]));
}
