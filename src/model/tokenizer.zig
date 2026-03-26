//! Bridge UTF-8 text and token IDs through an external tokenizer backend.
//! @section Tokenization
//! The current implementation shells out to Python-backed tokenizers so the
//! runtime can handle prompts and decoded output before a native tokenizer lands.
const std = @import("std");

const log = std.log.scoped(.tokenizer);

/// Tokenizer backends supported by the current shell-out implementation.
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

    /// Construct a tokenizer wrapper for a specific backend and model path.
    /// @param allocator Allocator used for process IO buffers.
    /// @param model_path Model or tokenizer path consumed by the backend.
    /// @param backend Backend implementation to invoke for encode/decode.
    /// @returns A Tokenizer value that borrows the provided model path.
    pub fn init(allocator: std.mem.Allocator, model_path: []const u8, backend: TokenizerBackend) Tokenizer {
        return Tokenizer{
            .backend = backend,
            .model_path = model_path,
            .allocator = allocator,
        };
    }

    /// Encode text into token IDs by invoking the configured backend.
    /// @param self Tokenizer configuration and allocator.
    /// @param text UTF-8 prompt text to encode.
    /// @returns A heap-allocated slice of token IDs.
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
        child.stdin_behavior = .Pipe;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;

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

    /// Decode token IDs back into UTF-8 text by invoking the configured backend.
    /// @param self Tokenizer configuration and allocator.
    /// @param tokens Token IDs to decode in order.
    /// @returns A heap-allocated UTF-8 byte slice containing the decoded text.
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
        child.stdin_behavior = .Pipe;
        child.stdout_behavior = .Pipe;
        child.stderr_behavior = .Pipe;

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

    /// Release tokenizer-owned resources.
    /// @note The current implementation only borrows the model path, so deinit is a no-op.
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
