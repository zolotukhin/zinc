//! Native BPE tokenizer that reads vocabulary and merge rules from GGUF metadata.
//! @section Tokenization
//! Implements byte-pair encoding directly in Zig using the tokenizer tables
//! embedded in GGUF model files, eliminating external tokenizer dependencies.
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
    bos_id: ?u32,
    eos_id: u32,
    /// Whether prompts should be prefixed with BOS automatically.
    prepend_bos: bool,
    /// Token scores (used for SentencePiece-style merge priority)
    scores: ?[]const f32,
    /// Chat template string from GGUF metadata, or null
    chat_template: ?[]const u8 = null,
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
        log.debug("Tokenizer: {d} tokens", .{vocab_size});

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
                    log.debug("Tokenizer: {d} BPE merges", .{merges_list.items.len});
                },
                else => {},
            }
        }

        // Read special token IDs
        const bos_id = gf.getU32("tokenizer.ggml.bos_token_id");
        const eos_id = gf.getU32("tokenizer.ggml.eos_token_id") orelse 2;
        const prepend_bos = gf.getBool("tokenizer.ggml.add_bos_token") orelse (bos_id != null);

        const model_type = gf.getString("tokenizer.ggml.model") orelse "unknown";
        if (bos_id) |id| {
            log.debug("Tokenizer type: {s} | BOS: {d} | prepend_bos={} | EOS: {d}", .{
                model_type,
                id,
                prepend_bos,
                eos_id,
            });
        } else {
            log.debug("Tokenizer type: {s} | BOS: none | prepend_bos={} | EOS: {d}", .{
                model_type,
                prepend_bos,
                eos_id,
            });
        }

        const chat_template = gf.getString("tokenizer.chat_template");
        if (chat_template) |tmpl| log.debug("Chat template: {d} chars", .{tmpl.len});

        return Tokenizer{
            .vocab = vocab,
            .token_to_id = token_to_id,
            .merges = try merges_list.toOwnedSlice(allocator),
            .scores = scores,
            .bos_id = bos_id,
            .eos_id = eos_id,
            .prepend_bos = prepend_bos,
            .chat_template = chat_template,
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

    /// Encode UTF-8 text into a token ID slice using BPE merges from the GGUF vocabulary.
    /// Encode UTF-8 text into token IDs using BPE merges from the GGUF vocabulary.
    pub fn encode(self: *const Tokenizer, text: []const u8) ![]u32 {
        if (text.len == 0) return try self.allocator.alloc(u32, 0);

        // Start with GPT-2 byte-level encoding: each raw byte maps to a Unicode char
        var symbols: std.ArrayList([]const u8) = .{};
        defer symbols.deinit(self.allocator);

        var owned_symbols: std.ArrayList([]u8) = .{};
        defer {
            for (owned_symbols.items) |sym| self.allocator.free(sym);
            owned_symbols.deinit(self.allocator);
        }

        for (text) |byte| {
            const unicode_char = gpt2ByteToUnicode(byte);
            // Find length of UTF-8 encoding
            const len: usize = if (unicode_char[0] < 0x80) 1 else if (unicode_char[0] < 0xE0) 2 else 3;
            // We need to allocate a persistent copy since symbols stores slices
            const copy = try self.allocator.alloc(u8, len);
            @memcpy(copy, unicode_char[0..len]);
            try owned_symbols.append(self.allocator, copy);
            try symbols.append(self.allocator, copy);
        }

        // If we have merges (GPT-2/tiktoken style), use merge-based BPE
        if (self.merges.len > 0) {
            try self.applyMerges(&symbols, &owned_symbols);
        } else if (self.scores != null) {
            // SentencePiece style: greedily merge the pair with highest combined score
            try self.applySentencePieceMerges(&symbols, &owned_symbols);
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
    fn applyMerges(self: *const Tokenizer, symbols: *std.ArrayList([]const u8), owned_symbols: *std.ArrayList([]u8)) !void {
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
            try owned_symbols.append(self.allocator, merged);

            symbols.items[best_pos] = merged;
            _ = symbols.orderedRemove(best_pos + 1);
        }
    }

    /// Apply SentencePiece-style merges using token scores.
    fn applySentencePieceMerges(self: *const Tokenizer, symbols: *std.ArrayList([]const u8), owned_symbols: *std.ArrayList([]u8)) !void {
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
            try owned_symbols.append(self.allocator, merged);

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
        return self.bos_id orelse self.eos_id;
    }

    /// Whether prompt construction should prepend BOS automatically.
    pub fn shouldPrependBos(self: *const Tokenizer) bool {
        return self.prepend_bos and self.bos_id != null;
    }

    /// Release tokenizer-owned vocabulary tables, merges, and optional score arrays.
    /// @param self Tokenizer to tear down in place.
    /// @note This does not free any source GGUF storage because the tokenizer owns duplicated tables.
    /// Decode a token to UTF-8 text, reversing GPT-2 byte encoding.
    /// The vocab stores tokens in GPT-2 Unicode representation where each original byte
    /// is mapped to a Unicode character. This reverses that mapping.
    pub fn decodeToken(self: *const Tokenizer, token_id: u32, buf: []u8) []const u8 {
        if (token_id >= self.vocab.len) return "";
        const gpt2_text = self.vocab[token_id];
        var out: usize = 0;
        var i: usize = 0;
        while (i < gpt2_text.len and out < buf.len) {
            // Parse one UTF-8 codepoint from the GPT-2 encoded token
            const byte0 = gpt2_text[i];
            var cp: u21 = 0;
            var cp_len: usize = 1;
            if (byte0 < 0x80) {
                cp = byte0;
            } else if (byte0 < 0xE0 and i + 1 < gpt2_text.len) {
                cp = (@as(u21, byte0 & 0x1F) << 6) | @as(u21, gpt2_text[i + 1] & 0x3F);
                cp_len = 2;
            } else if (byte0 < 0xF0 and i + 2 < gpt2_text.len) {
                cp = (@as(u21, byte0 & 0x0F) << 12) | (@as(u21, gpt2_text[i + 1] & 0x3F) << 6) | @as(u21, gpt2_text[i + 2] & 0x3F);
                cp_len = 3;
            } else {
                i += 1;
                continue;
            }
            i += cp_len;
            // Reverse the GPT-2 byte-to-unicode mapping
            const orig_byte: u8 = if (cp < 256) blk: {
                // Direct mapping: printable ASCII and high bytes
                break :blk @intCast(cp);
            } else if (cp >= 256 and cp < 256 + 33) blk: {
                // 0-32 were mapped to U+0100..U+0120
                break :blk @intCast(cp - 256);
            } else if (cp >= 256 + 33 and cp < 256 + 33 + 34) blk: {
                // 127-160 were mapped to U+0121..U+0142
                break :blk @intCast(cp - 256 - 33 + 0x7F);
            } else if (cp == 256 + 33 + 34) blk: {
                // 173 (0xAD) was mapped to U+0143
                break :blk 0xAD;
            } else blk: {
                break :blk '?';
            };
            buf[out] = orig_byte;
            out += 1;
        }
        return buf[0..out];
    }

    /// Apply chat template to role/content pairs. Returns formatted prompt in buf.
    pub fn applyChatTemplate(self: *const Tokenizer, roles: []const []const u8, contents: []const []const u8, buf: []u8) ![]const u8 {
        var pos: usize = 0;
        const use_chatml = if (self.chat_template) |tmpl|
            std.mem.indexOf(u8, tmpl, "im_start") != null
        else
            true;
        const n = @min(roles.len, contents.len);
        if (use_chatml) {
            for (0..n) |i| {
                const written = std.fmt.bufPrint(buf[pos..], "<|im_start|>{s}\n{s}<|im_end|>\n", .{ roles[i], contents[i] }) catch return error.BufferTooSmall;
                pos += written.len;
            }
            const suffix = std.fmt.bufPrint(buf[pos..], "<|im_start|>assistant\n", .{}) catch return error.BufferTooSmall;
            pos += suffix.len;
        } else {
            for (0..n) |i| {
                const written = std.fmt.bufPrint(buf[pos..], "[{s}]: {s}\n", .{ roles[i], contents[i] }) catch return error.BufferTooSmall;
                pos += written.len;
            }
        }
        return buf[0..pos];
    }

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
        .prepend_bos = true,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    // With empty vocab, should fall back to byte value
    try std.testing.expectEqual(@as(u32, 65), tok.findByteToken(65));
}

test "shouldPrependBos is false when BOS metadata is absent" {
    var tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = null,
        .eos_id = 2,
        .prepend_bos = false,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    try std.testing.expect(!tok.shouldPrependBos());
    try std.testing.expectEqual(@as(u32, 2), tok.bosId());
}

test "decodeToken converts GPT-2 leading-space marker back to ASCII space" {
    const vocab = [_][]const u8{"\xC4\xA0Paris"};
    var tok = Tokenizer{
        .vocab = &vocab,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    var buf: [32]u8 = undefined;
    const decoded = tok.decodeToken(0, &buf);
    try std.testing.expectEqualStrings(" Paris", decoded);
}

test "decodeToken converts GPT-2 remapped newline back to byte 0x0A" {
    const vocab = [_][]const u8{"\xC4\x8A"};
    var tok = Tokenizer{
        .vocab = &vocab,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    var buf: [8]u8 = undefined;
    const decoded = tok.decodeToken(0, &buf);
    try std.testing.expectEqualStrings("\n", decoded);
}

test "applyChatTemplate ChatML format" {
    const tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .chat_template = null, // null defaults to ChatML
        .allocator = std.testing.allocator,
    };
    var buf: [1024]u8 = undefined;
    const roles = [_][]const u8{"user"};
    const contents = [_][]const u8{"Hello"};
    const result = try tok.applyChatTemplate(&roles, &contents, &buf);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>user") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Hello") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_end|>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>assistant") != null);
}

test "applyChatTemplate with im_start template uses ChatML" {
    const tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .chat_template = "{%- for m in messages %}<|im_start|>{{m.role}}\n{{m.content}}<|im_end|>{%- endfor %}",
        .allocator = std.testing.allocator,
    };
    var buf: [1024]u8 = undefined;
    const roles = [_][]const u8{ "system", "user" };
    const contents = [_][]const u8{ "You help.", "Hi" };
    const result = try tok.applyChatTemplate(&roles, &contents, &buf);
    // Should have both messages
    try std.testing.expect(std.mem.indexOf(u8, result, "system") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "You help.") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Hi") != null);
}

test "applyChatTemplate non-ChatML fallback" {
    const tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .chat_template = "some other template without im tags",
        .allocator = std.testing.allocator,
    };
    var buf: [1024]u8 = undefined;
    const roles = [_][]const u8{"user"};
    const contents = [_][]const u8{"test"};
    const result = try tok.applyChatTemplate(&roles, &contents, &buf);
    // Should use [role]: content format
    try std.testing.expect(std.mem.indexOf(u8, result, "[user]: test") != null);
    // Should NOT have ChatML tags
    try std.testing.expect(std.mem.indexOf(u8, result, "im_start") == null);
}

test "applyChatTemplate empty messages" {
    const tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .chat_template = null,
        .allocator = std.testing.allocator,
    };
    var buf: [1024]u8 = undefined;
    const roles = [_][]const u8{};
    const contents = [_][]const u8{};
    const result = try tok.applyChatTemplate(&roles, &contents, &buf);
    // Should just have the assistant prefix
    try std.testing.expectEqualStrings("<|im_start|>assistant\n", result);
}

test "encode frees temporary buffers for BPE merges" {
    const vocab = [_][]const u8{ "h", "i", "hi" };
    const merges = [_]Tokenizer.Merge{.{
        .first = "h",
        .second = "i",
        .rank = 0,
    }};
    var tok = Tokenizer{
        .vocab = &vocab,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &merges,
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    try tok.token_to_id.put("h", 0);
    try tok.token_to_id.put("i", 1);
    try tok.token_to_id.put("hi", 2);

    const tokens = try tok.encode("hi");
    defer std.testing.allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 2), tokens[0]);
}

test "encode frees temporary buffers for sentencepiece merges" {
    const vocab = [_][]const u8{ "a", "b", "ab" };
    const scores = try std.testing.allocator.dupe(f32, &[_]f32{ -1.0, -1.0, 2.0 });
    var tok = Tokenizer{
        .vocab = &vocab,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = scores,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();
    defer std.testing.allocator.free(scores);

    try tok.token_to_id.put("a", 0);
    try tok.token_to_id.put("b", 1);
    try tok.token_to_id.put("ab", 2);

    const tokens = try tok.encode("ab");
    defer std.testing.allocator.free(tokens);

    try std.testing.expectEqual(@as(usize, 1), tokens.len);
    try std.testing.expectEqual(@as(u32, 2), tokens[0]);
}
