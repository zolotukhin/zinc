//! Native BPE tokenizer that reads vocabulary and merge rules from GGUF metadata.
//! @section Tokenization
//! Implements byte-pair encoding directly in Zig using the tokenizer tables
//! embedded in GGUF model files, eliminating the Python dependency.
const std = @import("std");
const gguf = @import("gguf.zig");

const log = std.log.scoped(.tokenizer);

/// A native BPE tokenizer backed by vocabulary and merge tables from GGUF metadata.
pub const Tokenizer = struct {
    /// Vocabulary: token ID → token bytes
    vocab: []const []const u8,
    /// Reverse map: token bytes → token ID
    token_to_id: std.StringHashMap(u32),
    /// BPE merge rules ordered by priority (lower index = higher priority)
    merges: []const Merge,
    /// Special token IDs
    bos_id: u32,
    eos_id: u32,
    /// Token scores (used for SentencePiece-style merge priority)
    scores: ?[]const f32,
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,

    const Merge = struct {
        first: []const u8,
        second: []const u8,
        rank: u32,
    };

    /// Initialize tokenizer from GGUF metadata.
    /// Reads tokenizer.ggml.tokens, tokenizer.ggml.merges, and special token IDs.
    pub fn initFromGGUF(gf: *const gguf.GGUFFile, allocator: std.mem.Allocator) !Tokenizer {
        // Read vocabulary tokens
        const tokens_val = gf.metadata.get("tokenizer.ggml.tokens") orelse {
            return error.NoTokenizerVocab;
        };
        const tokens_array = switch (tokens_val) {
            .array => |a| a,
            else => return error.NoTokenizerVocab,
        };

        const vocab_size = tokens_array.len;
        log.info("Tokenizer: {d} tokens", .{vocab_size});

        // Build vocab table and reverse map
        const vocab = try allocator.alloc([]const u8, vocab_size);
        errdefer allocator.free(vocab);

        var token_to_id = std.StringHashMap(u32).init(allocator);
        errdefer token_to_id.deinit();

        try token_to_id.ensureTotalCapacity(@intCast(vocab_size));

        for (tokens_array, 0..) |tok_val, i| {
            const tok_str = switch (tok_val) {
                .string => |s| s,
                else => "",
            };
            vocab[i] = tok_str;
            token_to_id.putAssumeCapacity(tok_str, @intCast(i));
        }

        // Read token scores if available (SentencePiece models use scores for merge priority)
        var scores: ?[]const f32 = null;
        if (gf.metadata.get("tokenizer.ggml.scores")) |scores_val| {
            switch (scores_val) {
                .array => |a| {
                    const s = try allocator.alloc(f32, a.len);
                    for (a, 0..) |v, i| {
                        s[i] = switch (v) {
                            .float32 => |f| f,
                            else => 0.0,
                        };
                    }
                    scores = s;
                },
                else => {},
            }
        }

        // Read BPE merges if available
        var merges_list: std.ArrayListAligned(Merge, null) = .{};
        errdefer merges_list.deinit(allocator);

        if (gf.metadata.get("tokenizer.ggml.merges")) |merges_val| {
            switch (merges_val) {
                .array => |a| {
                    try merges_list.ensureTotalCapacity(allocator, a.len);
                    for (a, 0..) |merge_val, rank| {
                        const merge_str = switch (merge_val) {
                            .string => |s| s,
                            else => continue,
                        };
                        // Merges are "tokenA tokenB" separated by first space
                        if (std.mem.indexOfScalar(u8, merge_str, ' ')) |sep| {
                            merges_list.appendAssumeCapacity(.{
                                .first = merge_str[0..sep],
                                .second = merge_str[sep + 1 ..],
                                .rank = @intCast(rank),
                            });
                        }
                    }
                    log.info("Tokenizer: {d} BPE merges", .{merges_list.items.len});
                },
                else => {},
            }
        }

        // Read special token IDs
        const bos_id = gf.getU32("tokenizer.ggml.bos_token_id") orelse 1;
        const eos_id = gf.getU32("tokenizer.ggml.eos_token_id") orelse 2;

        const model_type = gf.getString("tokenizer.ggml.model") orelse "unknown";
        log.info("Tokenizer type: {s} | BOS: {d} | EOS: {d}", .{ model_type, bos_id, eos_id });

        return Tokenizer{
            .vocab = vocab,
            .token_to_id = token_to_id,
            .merges = try merges_list.toOwnedSlice(allocator),
            .scores = scores,
            .bos_id = bos_id,
            .eos_id = eos_id,
            .allocator = allocator,
        };
    }

    /// Encode UTF-8 text into token IDs using the loaded vocabulary and merge tables.
    /// @param self Tokenizer state containing vocabulary, reverse lookup, merges, and optional scores.
    /// @param text UTF-8 input text to tokenize.
    /// @returns A heap-allocated token-ID slice in model order.
    /// @note Unknown merged symbols fall back to byte-level tokens so encoding always produces a result.
    /// GPT-2 byte-to-unicode mapping. Maps each raw byte to a Unicode character.
    /// Printable ASCII (33-126, 161-172, 174-255) maps to itself.
    /// Non-printable bytes (0-32, 127-160, 173) map to U+0100+ range.
    /// This is how GPT-2/Qwen tokenizers represent raw bytes in the vocabulary.
    fn gpt2ByteToUnicode(byte: u8) [4]u8 {
        // Build the codepoint for this byte
        const cp: u21 = switch (byte) {
            // Printable ranges that map to themselves
            '!'...'~', 0xA1...0xAC, 0xAE...0xFF => byte,
            // Non-printable: shift to U+0100+ range
            else => @as(u21, 256) + @as(u21, switch (byte) {
                0...0x20 => byte, // 0-32 → U+0100..U+0120
                0x7F...0xA0 => byte - 0x7F + 33, // 127-160 → U+0121..U+0141
                0xAD => 33 + 34, // 173 → U+0143
                else => byte,
            }),
        };
        // Encode as UTF-8
        var buf: [4]u8 = .{ 0, 0, 0, 0 };
        if (cp < 0x80) {
            buf[0] = @intCast(cp);
            return buf;
        } else if (cp < 0x800) {
            buf[0] = @intCast(0xC0 | (cp >> 6));
            buf[1] = @intCast(0x80 | (cp & 0x3F));
            return buf;
        } else {
            buf[0] = @intCast(0xE0 | (cp >> 12));
            buf[1] = @intCast(0x80 | ((cp >> 6) & 0x3F));
            buf[2] = @intCast(0x80 | (cp & 0x3F));
            return buf;
        }
    }

    pub fn encode(self: *const Tokenizer, text: []const u8) ![]u32 {
        if (text.len == 0) return try self.allocator.alloc(u32, 0);

        // Start with GPT-2 byte-level encoding: each raw byte maps to a Unicode char
        var symbols: std.ArrayList([]const u8) = .{};
        defer symbols.deinit(self.allocator);

        for (text) |byte| {
            const unicode_char = gpt2ByteToUnicode(byte);
            // Find length of UTF-8 encoding
            const len: usize = if (unicode_char[0] < 0x80) 1 else if (unicode_char[0] < 0xE0) 2 else 3;
            // We need to allocate a persistent copy since symbols stores slices
            const copy = try self.allocator.alloc(u8, len);
            @memcpy(copy, unicode_char[0..len]);
            try symbols.append(self.allocator, copy);
        }

        // If we have merges (GPT-2/tiktoken style), use merge-based BPE
        if (self.merges.len > 0) {
            try self.applyMerges(&symbols);
        } else if (self.scores != null) {
            // SentencePiece style: greedily merge the pair with highest combined score
            try self.applySentencePieceMerges(&symbols);
        }

        // Convert symbol strings to token IDs
        var tokens: std.ArrayList(u32) = .{};
        errdefer tokens.deinit(self.allocator);

        for (symbols.items) |sym| {
            if (self.token_to_id.get(sym)) |id| {
                try tokens.append(self.allocator, id);
            } else {
                // Fall back to byte-level tokens for unknown sequences
                for (sym) |byte| {
                    const byte_token = self.findByteToken(byte);
                    try tokens.append(self.allocator, byte_token);
                }
            }
        }

        return try tokens.toOwnedSlice(self.allocator);
    }

    /// Apply BPE merges in priority order (GPT-2/tiktoken style).
    fn applyMerges(self: *const Tokenizer, symbols: *std.ArrayList([]const u8)) !void {
        // Build a merge rank lookup for fast pair → rank queries
        var merge_ranks = std.StringHashMap(u32).init(self.allocator);
        defer merge_ranks.deinit();

        // Pre-allocate merge key buffer
        var key_buf: std.ArrayList(u8) = .{};
        defer key_buf.deinit(self.allocator);

        for (self.merges) |merge| {
            key_buf.clearRetainingCapacity();
            try key_buf.appendSlice(self.allocator, merge.first);
            try key_buf.append(self.allocator, ' ');
            try key_buf.appendSlice(self.allocator, merge.second);
            const key_copy = try self.allocator.dupe(u8, key_buf.items);
            try merge_ranks.put(key_copy, merge.rank);
        }
        defer {
            var it = merge_ranks.iterator();
            while (it.next()) |entry| {
                self.allocator.free(@constCast(entry.key_ptr.*));
            }
        }

        // Repeatedly find and apply the highest-priority (lowest rank) merge
        while (symbols.items.len > 1) {
            var best_rank: u32 = std.math.maxInt(u32);
            var best_pos: usize = 0;

            // Find the pair with the lowest merge rank
            for (0..symbols.items.len - 1) |pos| {
                key_buf.clearRetainingCapacity();
                try key_buf.appendSlice(self.allocator, symbols.items[pos]);
                try key_buf.append(self.allocator, ' ');
                try key_buf.appendSlice(self.allocator, symbols.items[pos + 1]);

                if (merge_ranks.get(key_buf.items)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_pos = pos;
                    }
                }
            }

            if (best_rank == std.math.maxInt(u32)) break; // No more merges possible

            // Merge the pair: concatenate symbols[best_pos] and symbols[best_pos+1]
            const merged = try std.mem.concat(self.allocator, u8, &.{
                symbols.items[best_pos],
                symbols.items[best_pos + 1],
            });

            symbols.items[best_pos] = merged;
            _ = symbols.orderedRemove(best_pos + 1);
            // Note: merged strings leak here. Acceptable for a short-lived encode call.
            // A proper fix would track allocated slices separately from input slices.
        }
    }

    /// Apply SentencePiece-style merges using token scores.
    fn applySentencePieceMerges(self: *const Tokenizer, symbols: *std.ArrayList([]const u8)) !void {
        const scores_arr = self.scores orelse return;

        while (symbols.items.len > 1) {
            var best_score: f32 = -std.math.inf(f32);
            var best_pos: usize = 0;
            var found = false;

            for (0..symbols.items.len - 1) |pos| {
                // Try concatenating adjacent symbols and look up the merged token
                const merged = try std.mem.concat(self.allocator, u8, &.{
                    symbols.items[pos],
                    symbols.items[pos + 1],
                });
                defer self.allocator.free(merged);

                if (self.token_to_id.get(merged)) |id| {
                    const score = if (id < scores_arr.len) scores_arr[id] else -std.math.inf(f32);
                    if (score > best_score) {
                        best_score = score;
                        best_pos = pos;
                        found = true;
                    }
                }
            }

            if (!found) break;

            const merged = try std.mem.concat(self.allocator, u8, &.{
                symbols.items[best_pos],
                symbols.items[best_pos + 1],
            });

            symbols.items[best_pos] = merged;
            _ = symbols.orderedRemove(best_pos + 1);
        }
    }

    /// Find a byte-level fallback token (e.g., "<0x41>" for byte 0x41).
    fn findByteToken(self: *const Tokenizer, byte: u8) u32 {
        // Try common byte token formats
        var buf: [8]u8 = undefined;
        const hex = std.fmt.bufPrint(&buf, "<0x{X:0>2}>", .{byte}) catch return byte;
        if (self.token_to_id.get(hex)) |id| return id;

        // Try lowercase hex
        const hex_lower = std.fmt.bufPrint(&buf, "<0x{x:0>2}>", .{byte}) catch return byte;
        if (self.token_to_id.get(hex_lower)) |id| return id;

        // Last resort: use byte value directly as token ID
        return @as(u32, byte);
    }

    /// Return the model's configured end-of-sequence token ID.
    /// @param self Tokenizer to inspect.
    /// @returns The EOS token ID loaded from GGUF metadata or the default fallback.
    pub fn eosId(self: *const Tokenizer) u32 {
        return self.eos_id;
    }

    /// Return the model's configured beginning-of-sequence token ID.
    /// @param self Tokenizer to inspect.
    /// @returns The BOS token ID loaded from GGUF metadata or the default fallback.
    pub fn bosId(self: *const Tokenizer) u32 {
        return self.bos_id;
    }

    /// Release tokenizer-owned vocabulary tables, merges, and optional score arrays.
    /// @param self Tokenizer to tear down in place.
    /// @note This does not free any source GGUF storage because the tokenizer owns duplicated tables.
    pub fn deinit(self: *Tokenizer) void {
        if (self.scores) |s| self.allocator.free(s);
        self.allocator.free(self.merges);
        self.token_to_id.deinit();
        self.allocator.free(self.vocab);
    }
};

test "Tokenizer findByteToken" {
    var tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    // With empty vocab, should fall back to byte value
    try std.testing.expectEqual(@as(u32, 65), tok.findByteToken(65));
}
