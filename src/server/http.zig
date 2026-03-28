//! Minimal HTTP server for the OpenAI-compatible inference API.
//! @section HTTP Server
//! Listens on a configurable port and routes requests to the inference
//! engine. Supports streaming via SSE for chat completion endpoints.
const std = @import("std");

const log = std.log.scoped(.http);

/// Active client connection with request state.
pub const Connection = struct {
    stream: std.net.Stream,
    allocator: std.mem.Allocator,

    pub fn readRequest(self: *Connection) !Request {
        _ = self;
        // TODO: parse HTTP request
        return Request{ .method = .GET, .path = "/health", .body = "" };
    }

    pub fn sendResponse(self: *Connection, status: u16, content_type: []const u8, body: []const u8) !void {
        var buf: [4096]u8 = undefined;
        const header = std.fmt.bufPrint(&buf, "HTTP/1.1 {d} OK\r\nContent-Type: {s}\r\nContent-Length: {d}\r\nAccess-Control-Allow-Origin: *\r\n\r\n", .{ status, content_type, body.len }) catch return error.HeaderTooLarge;
        try self.stream.writeAll(header);
        try self.stream.writeAll(body);
    }

    pub fn close(self: *Connection) void {
        self.stream.close();
    }
};

pub const Method = enum { GET, POST, OPTIONS, UNKNOWN };

pub const Request = struct {
    method: Method,
    path: []const u8,
    body: []const u8,
};

/// HTTP server that accepts connections and dispatches to a handler.
pub const Server = struct {
    listener: std.net.Server,
    allocator: std.mem.Allocator,

    /// Bind to the given port and start listening.
    pub fn init(allocator: std.mem.Allocator, port: u16) !Server {
        const address = std.net.Address.initIp4(.{ 0, 0, 0, 0 }, port);
        const listener = try address.listen(.{
            .reuse_address = true,
        });
        log.info("Listening on port {d}", .{port});
        return Server{ .listener = listener, .allocator = allocator };
    }

    /// Accept the next connection. Blocks until a client connects.
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
