const std = @import("std");

const log = std.log.scoped(.tokenizer);

pub const TokenizerBackend = enum {
    sentencepiece,
    tiktoken,
};

/// Tokenizer that shells out to an external process for BPE encoding/decoding.
/// This is a temporary solution — native Zig tokenizer planned for later phases.
pub const Tokenizer = struct {
    backend: TokenizerBackend,
    model_path: []const u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, backend: TokenizerBackend) Tokenizer {
        return Tokenizer{
            .backend = backend,
            .model_path = model_path,
            .allocator = allocator,
        };
    }

    /// Encode text to token IDs by calling external tokenizer.
    pub fn encode(self: *const Tokenizer, text: []const u8) ![]u32 {
        const cmd = switch (self.backend) {
            .sentencepiece => &[_][]const u8{
                "python3", "-c",
                "import sys; from sentencepiece import SentencePieceProcessor; " ++
                    "sp = SentencePieceProcessor(model_file=sys.argv[1]); " ++
                    "tokens = sp.encode(sys.stdin.read()); " ++
                    "[print(t) for t in tokens]",
                self.model_path,
            },
            .tiktoken => &[_][]const u8{
                "python3", "-c",
                "import sys, tiktoken; " ++
                    "enc = tiktoken.get_encoding('cl100k_base'); " ++
                    "tokens = enc.encode(sys.stdin.read()); " ++
                    "[print(t) for t in tokens]",
            },
        };

        var child = std.process.Child.init(cmd, self.allocator);
        child.stdin_behavior = .pipe;
        child.stdout_behavior = .pipe;
        child.stderr_behavior = .pipe;

        try child.spawn();

        // Write input text
        if (child.stdin) |stdin| {
            stdin.writeAll(text) catch {};
            stdin.close();
            child.stdin = null;
        }

        // Read output
        const max_output: usize = 1024 * 1024; // 1MB max
        const stdout_data = try child.stdout.?.readToEndAlloc(self.allocator, max_output);
        defer self.allocator.free(stdout_data);

        _ = try child.wait();

        // Parse token IDs from stdout (one per line)
        var tokens: std.ArrayList(u32) = .{};
        errdefer tokens.deinit(self.allocator);

        var lines = std.mem.splitScalar(u8, stdout_data, '\n');
        while (lines.next()) |line| {
            const trimmed = std.mem.trim(u8, line, &[_]u8{ ' ', '\t', '\r' });
            if (trimmed.len == 0) continue;
            const id = std.fmt.parseInt(u32, trimmed, 10) catch continue;
            try tokens.append(self.allocator, id);
        }

        return try tokens.toOwnedSlice(self.allocator);
    }

    /// Decode token IDs to text by calling external tokenizer.
    pub fn decode(self: *const Tokenizer, tokens: []const u32) ![]u8 {
        // Build token list as space-separated string
        var token_str: std.ArrayList(u8) = .{};
        defer token_str.deinit(self.allocator);

        for (tokens, 0..) |t, i| {
            if (i > 0) try token_str.append(self.allocator, ' ');
            var buf: [16]u8 = undefined;
            const s = std.fmt.bufPrint(&buf, "{d}", .{t}) catch unreachable;
            try token_str.appendSlice(self.allocator, s);
        }

        const cmd = switch (self.backend) {
            .sentencepiece => &[_][]const u8{
                "python3", "-c",
                "import sys; from sentencepiece import SentencePieceProcessor; " ++
                    "sp = SentencePieceProcessor(model_file=sys.argv[1]); " ++
                    "ids = list(map(int, sys.stdin.read().split())); " ++
                    "print(sp.decode(ids), end='')",
                self.model_path,
            },
            .tiktoken => &[_][]const u8{
                "python3", "-c",
                "import sys, tiktoken; " ++
                    "enc = tiktoken.get_encoding('cl100k_base'); " ++
                    "ids = list(map(int, sys.stdin.read().split())); " ++
                    "print(enc.decode(ids), end='')",
            },
        };

        var child = std.process.Child.init(cmd, self.allocator);
        child.stdin_behavior = .pipe;
        child.stdout_behavior = .pipe;
        child.stderr_behavior = .pipe;

        try child.spawn();

        if (child.stdin) |stdin| {
            stdin.writeAll(token_str.items) catch {};
            stdin.close();
            child.stdin = null;
        }

        const max_output: usize = 1024 * 1024;
        const result = try child.stdout.?.readToEndAlloc(self.allocator, max_output);

        _ = try child.wait();

        return result;
    }

    pub fn deinit(self: *Tokenizer) void {
        _ = self;
        // No resources to free — paths are borrowed
    }
};

test "Tokenizer init" {
    const tok = Tokenizer.init(std.testing.allocator, "model.spm", .sentencepiece);
    try std.testing.expectEqual(TokenizerBackend.sentencepiece, tok.backend);
    try std.testing.expectEqualStrings("model.spm", tok.model_path);
}
