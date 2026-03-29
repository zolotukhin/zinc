//! Minimal HTTP/1.1 server for the OpenAI-compatible inference API.
//! Supports request parsing, JSON responses, SSE streaming headers, and error responses.
const std = @import("std");

const log = std.log.scoped(.http);

/// Active client connection with request/response capabilities.
pub const Connection = struct {
    stream: std.net.Stream,
    allocator: std.mem.Allocator,
    // Read buffer for request parsing
    read_buf: [65536]u8 = undefined,
    read_len: usize = 0,

    /// Read and parse an HTTP/1.1 request. Reads headers + body based on Content-Length.
    pub fn readRequest(self: *Connection) !Request {
        // Read data until we find \r\n\r\n (end of headers)
        var total: usize = 0;
        var header_end: ?usize = null;
        while (total < self.read_buf.len) {
            const n = self.stream.read(self.read_buf[total..]) catch |err| {
                if (total > 0) break;
                return err;
            };
            if (n == 0) break; // connection closed
            total += n;
            // Search for header terminator
            if (total >= 4) {
                const search_start = if (total > n + 3) total - n - 3 else 0;
                for (search_start..total - 3) |i| {
                    if (std.mem.eql(u8, self.read_buf[i .. i + 4], "\r\n\r\n")) {
                        header_end = i + 4;
                        break;
                    }
                }
                if (header_end != null) break;
            }
        }
        self.read_len = total;
        const hdr_end = header_end orelse return error.MalformedRequest;
        const header_str = self.read_buf[0..hdr_end];

        // Parse request line: METHOD PATH HTTP/1.1\r\n
        const first_line_end = std.mem.indexOf(u8, header_str, "\r\n") orelse return error.MalformedRequest;
        const request_line = header_str[0..first_line_end];

        // Split by spaces
        var method: Method = .UNKNOWN;
        var path: []const u8 = "/";
        var part: usize = 0;
        var start: usize = 0;
        for (request_line, 0..) |c, i| {
            if (c == ' ' or i == request_line.len - 1) {
                const end = if (c == ' ') i else i + 1;
                const token = request_line[start..end];
                if (part == 0) {
                    if (std.mem.eql(u8, token, "GET")) method = .GET
                    else if (std.mem.eql(u8, token, "POST")) method = .POST
                    else if (std.mem.eql(u8, token, "OPTIONS")) method = .OPTIONS;
                } else if (part == 1) {
                    path = token;
                }
                part += 1;
                start = i + 1;
            }
        }

        // Extract Content-Length
        var content_length: usize = 0;
        var line_start: usize = first_line_end + 2;
        while (line_start < hdr_end) {
            const line_end = std.mem.indexOf(u8, header_str[line_start..], "\r\n") orelse break;
            const line = header_str[line_start .. line_start + line_end];
            if (line.len > 16 and (line[0] == 'C' or line[0] == 'c')) {
                // Case-insensitive Content-Length check
                const lower = "content-length: ";
                if (line.len > lower.len) {
                    var matches = true;
                    for (lower, 0..) |lc, ci| {
                        const hc = if (line[ci] >= 'A' and line[ci] <= 'Z') line[ci] + 32 else line[ci];
                        if (hc != lc) { matches = false; break; }
                    }
                    if (matches) {
                        content_length = std.fmt.parseInt(usize, std.mem.trim(u8, line[lower.len..], " \t"), 10) catch 0;
                    }
                }
            }
            line_start += line_end + 2;
        }

        // Read remaining body if needed
        const body_already = total - hdr_end;
        if (body_already < content_length) {
            const remaining = content_length - body_already;
            const buf_remaining = self.read_buf.len - total;
            if (remaining > buf_remaining) return error.RequestTooLarge;
            var read_so_far: usize = 0;
            while (read_so_far < remaining) {
                const n = try self.stream.read(self.read_buf[total + read_so_far .. total + read_so_far + remaining - read_so_far]);
                if (n == 0) break;
                read_so_far += n;
            }
            self.read_len = total + read_so_far;
        }

        const body = if (content_length > 0) self.read_buf[hdr_end .. hdr_end + content_length] else "";
        return Request{ .method = method, .path = path, .body = body };
    }

    /// Send a JSON response with status code.
    pub fn sendJson(self: *Connection, status: u16, body: []const u8) !void {
        var buf: [512]u8 = undefined;
        const status_text = switch (status) {
            200 => "OK", 400 => "Bad Request", 404 => "Not Found",
            429 => "Too Many Requests", 500 => "Internal Server Error",
            503 => "Service Unavailable", else => "OK",
        };
        const header = std.fmt.bufPrint(&buf, "HTTP/1.1 {d} {s}\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n", .{ status, status_text, body.len }) catch return error.HeaderTooLarge;
        try self.stream.writeAll(header);
        try self.stream.writeAll(body);
    }

    /// Send an OpenAI-format error response.
    pub fn sendError(self: *Connection, status: u16, err_type: []const u8, message: []const u8) !void {
        var buf: [2048]u8 = undefined;
        const body = std.fmt.bufPrint(&buf, "{{\"error\":{{\"message\":\"{s}\",\"type\":\"{s}\",\"code\":{d}}}}}", .{ message, err_type, status }) catch return error.HeaderTooLarge;
        try self.sendJson(status, body);
    }

    /// Send SSE streaming response headers (no body yet — caller writes events via stream).
    pub fn sendSseStart(self: *Connection) !void {
        const header = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nAccess-Control-Allow-Origin: *\r\nConnection: keep-alive\r\nTransfer-Encoding: chunked\r\n\r\n";
        try self.stream.writeAll(header);
    }

    /// Write a single SSE chunk (chunked transfer encoding).
    pub fn writeSseEvent(self: *Connection, data: []const u8) !void {
        // Chunked format: {size_hex}\r\n{data}\r\n
        var size_buf: [16]u8 = undefined;
        // data: {json}\n\n = data.len + "data: ".len + "\n\n".len
        const event_prefix = "data: ";
        const event_suffix = "\n\n";
        const chunk_len = event_prefix.len + data.len + event_suffix.len;
        const size_str = std.fmt.bufPrint(&size_buf, "{x}\r\n", .{chunk_len}) catch unreachable;
        try self.stream.writeAll(size_str);
        try self.stream.writeAll(event_prefix);
        try self.stream.writeAll(data);
        try self.stream.writeAll(event_suffix);
        try self.stream.writeAll("\r\n");
    }

    /// Write the final SSE [DONE] event and close the chunked stream.
    pub fn writeSseDone(self: *Connection) !void {
        try self.writeSseEvent("[DONE]");
        // Chunked terminator: 0\r\n\r\n
        try self.stream.writeAll("0\r\n\r\n");
    }

    /// Close the connection stream.
    pub fn close(self: *Connection) void {
        self.stream.close();
    }
};

/// HTTP request methods.
pub const Method = enum { GET, POST, OPTIONS, UNKNOWN };

/// Parsed HTTP request.
pub const Request = struct {
    method: Method,
    path: []const u8,
    body: []const u8,
};

/// HTTP server that binds and listens on a TCP port.
pub const Server = struct {
    listener: std.net.Server,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, port: u16) !Server {
        const address = std.net.Address.initIp4(.{ 0, 0, 0, 0 }, port);
        const listener = try address.listen(.{ .reuse_address = true });
        return Server{ .listener = listener, .allocator = allocator };
    }

    pub fn accept(self: *Server) !Connection {
        const conn = try self.listener.accept();
        return Connection{ .stream = conn.stream, .allocator = self.allocator };
    }

    pub fn deinit(self: *Server) void {
        self.listener.deinit();
    }
};

test "Server struct size" {
    try std.testing.expect(@sizeOf(Server) > 0);
}

test "Method enum has expected values" {
    try std.testing.expect(@intFromEnum(Method.GET) != @intFromEnum(Method.POST));
}
