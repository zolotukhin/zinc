//! Active generation session for a streaming request.
//! Each session owns a connection, decode state, and generation progress.
const std = @import("std");
const http = @import("http.zig");
const forward_mod = @import("../compute/forward.zig");
const tokenizer_mod = @import("../model/tokenizer.zig");

const log = std.log.scoped(.session);

pub const SessionState = enum {
    prefilling,
    decoding,
    completed,
    failed,
};

/// An active streaming generation session.
pub const Session = struct {
    conn: http.Connection,
    state: SessionState,
    decode_state: forward_mod.DecodeState,
    prev_token: u32,
    tokens_generated: u32,
    max_tokens: u32,
    eos_token_id: u32,
    req_id: [32]u8,
    req_id_len: usize,
    model_name: []const u8,
    created_ts: i64,
    is_streaming: bool,
    // Non-streaming: accumulate tokens
    output_tokens: std.ArrayList(u32),
    allocator: std.mem.Allocator,

    pub fn init(
        conn: http.Connection,
        max_tokens: u32,
        eos_token_id: u32,
        model_name: []const u8,
        is_streaming: bool,
        allocator: std.mem.Allocator,
    ) Session {
        var id_buf: [32]u8 = undefined;
        const ts = std.time.timestamp();
        const id_len = (std.fmt.bufPrint(&id_buf, "chatcmpl-{x}", .{@as(u64, @bitCast(ts))}) catch &id_buf).len;
        return .{
            .conn = conn,
            .state = .prefilling,
            .decode_state = forward_mod.DecodeState.init(allocator),
            .prev_token = 0,
            .tokens_generated = 0,
            .max_tokens = max_tokens,
            .eos_token_id = eos_token_id,
            .req_id = id_buf,
            .req_id_len = id_len,
            .model_name = model_name,
            .created_ts = @divTrunc(std.time.timestamp(), 1),
            .is_streaming = is_streaming,
            .output_tokens = .{},
            .allocator = allocator,
        };
    }

    pub fn reqId(self: *const Session) []const u8 {
        return self.req_id[0..self.req_id_len];
    }

    /// Send one token as an SSE event. Returns false if connection is broken.
    pub fn sendToken(self: *Session, token_id: u32, tokenizer: *const tokenizer_mod.Tokenizer) bool {
        if (!self.is_streaming) {
            self.output_tokens.append(self.allocator, token_id) catch return false;
            return true;
        }
        const token_text = if (token_id < tokenizer.vocab.len) tokenizer.vocab[token_id] else "<?>";
        var escaped_buf: [512]u8 = undefined;
        const escaped = jsonEscape(token_text, &escaped_buf);
        var chunk_buf: [1024]u8 = undefined;
        const chunk = std.fmt.bufPrint(&chunk_buf,
            \\{{"id":"{s}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{"content":"{s}"}},"finish_reason":null}}]}}
        , .{ self.reqId(), self.created_ts, self.model_name, escaped }) catch return false;
        self.conn.writeSseEvent(chunk) catch return false;
        return true;
    }

    /// Send the final [DONE] event and close the stream.
    pub fn finish(self: *Session) void {
        if (self.is_streaming) {
            var chunk_buf: [512]u8 = undefined;
            const chunk = std.fmt.bufPrint(&chunk_buf,
                \\{{"id":"{s}","object":"chat.completion.chunk","created":{d},"model":"{s}","choices":[{{"index":0,"delta":{{}},"finish_reason":"stop"}}]}}
            , .{ self.reqId(), self.created_ts, self.model_name }) catch "";
            self.conn.writeSseEvent(chunk) catch {};
            self.conn.writeSseDone() catch {};
        }
        self.state = .completed;
    }

    pub fn deinit(self: *Session) void {
        self.decode_state.deinit();
        self.output_tokens.deinit(self.allocator);
        self.conn.close();
    }

    // JSON escape (duplicated from routes to avoid circular import)
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
};

test "Session reqId format" {
    // Test only the ID generation, not the full session lifecycle
    var id_buf: [32]u8 = undefined;
    const ts = std.time.timestamp();
    const id = std.fmt.bufPrint(&id_buf, "chatcmpl-{x}", .{@as(u64, @bitCast(ts))}) catch "chatcmpl-0";
    try std.testing.expect(std.mem.startsWith(u8, id, "chatcmpl-"));
    try std.testing.expect(id.len > 9);
}

test "SessionState enum values" {
    try std.testing.expect(@intFromEnum(SessionState.prefilling) != @intFromEnum(SessionState.decoding));
    try std.testing.expect(@intFromEnum(SessionState.completed) != @intFromEnum(SessionState.failed));
}
