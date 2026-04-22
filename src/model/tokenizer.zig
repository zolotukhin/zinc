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
    /// Whether prompt construction should append EOS.
    add_eos_token: bool = false,
    /// Token scores (used for SentencePiece-style merge priority)
    scores: ?[]const f32,
    /// Chat template string from GGUF metadata, or null
    chat_template: ?[]const u8 = null,
    /// Byte-level BPE pretokenizer style for GPT-2/Qwen-family vocabularies.
    pretokenizer: Pretokenizer = .legacy,
    /// Precomputed "first second" → rank lookup used by applyMerges. Built
    /// once at init so each BPE encode (many per prompt) doesn't rebuild a
    /// 151k-entry hashmap — doing that per call cost multi-minute latency on
    /// the second chat request because cumulative GPA allocator pressure
    /// made each rebuild orders of magnitude slower than the first.
    merge_ranks: std.StringHashMap(u32) = undefined,
    /// Whether merge_ranks has been populated (false for test constructors
    /// that don't go through initFromGGUF).
    merge_ranks_ready: bool = false,
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,

    const Merge = struct {
        first: []const u8,
        second: []const u8,
        rank: u32,
    };

    const Pretokenizer = enum {
        legacy,
        gpt2_ascii,
        gemma4_bpe,
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

        // Read special token IDs.
        const bos_id = gf.getU32("tokenizer.ggml.bos_token_id");
        const eos_id = gf.getU32("tokenizer.ggml.eos_token_id") orelse 2;
        const model_type = gf.getString("tokenizer.ggml.model") orelse "unknown";
        const architecture = gf.getString("general.architecture") orelse "";
        const prepend_bos = gf.getBool("tokenizer.ggml.add_bos_token") orelse blk: {
            // Default: prepend BOS when a BOS token ID is defined.
            // Llama 3 uses GPT2 tokenizer format but requires BOS (128000).
            // Qwen3/3.5 explicitly omit BOS metadata — do NOT prepend.
            // GPT-OSS defines a BOS token but llama.cpp does not prepend it for prompts.
            if (std.mem.eql(u8, architecture, "gpt-oss")) break :blk false;
            break :blk bos_id != null;
        };
        const add_eos_token = gf.getBool("tokenizer.ggml.add_eos_token") orelse false;
        if (bos_id) |id| {
            log.debug("Tokenizer type: {s} | BOS: {d} | prepend_bos={} | EOS: {d} | add_eos={}", .{
                model_type,
                id,
                prepend_bos,
                eos_id,
                add_eos_token,
            });
        } else {
            log.debug("Tokenizer type: {s} | BOS: none | prepend_bos={} | EOS: {d} | add_eos={}", .{
                model_type,
                prepend_bos,
                eos_id,
                add_eos_token,
            });
        }

        const chat_template = gf.getString("tokenizer.chat_template");
        if (chat_template) |tmpl| log.debug("Chat template: {d} chars", .{tmpl.len});
        const pre_name = gf.getString("tokenizer.ggml.pre") orelse "";
        const pretokenizer: Pretokenizer = if (std.mem.eql(u8, model_type, "gemma4") or std.mem.eql(u8, pre_name, "gemma4"))
            .gemma4_bpe
        else if (scores == null and merges_list.items.len > 0 and
            (std.mem.eql(u8, model_type, "gpt2") or
                std.mem.eql(u8, pre_name, "qwen2") or
                std.mem.eql(u8, pre_name, "qwen35")))
            .gpt2_ascii
        else
            .legacy;

        const merges_owned = try merges_list.toOwnedSlice(allocator);

        // Precompute "first second" → rank lookup. Populating this at init
        // is O(n) once, vs O(n) every BPE encode call. With ~150k merges
        // and ~10 BPE chunks per chat prompt, the cached path saves ~1.5M
        // key allocations per chat — which matters because repeated per-call
        // rebuilds made each chat after the first hang for minutes.
        var merge_ranks = std.StringHashMap(u32).init(allocator);
        errdefer {
            var it_rank = merge_ranks.iterator();
            while (it_rank.next()) |entry| allocator.free(@constCast(entry.key_ptr.*));
            merge_ranks.deinit();
        }
        try merge_ranks.ensureTotalCapacity(@intCast(merges_owned.len));
        var key_buf: std.ArrayList(u8) = .{};
        defer key_buf.deinit(allocator);
        for (merges_owned) |merge| {
            key_buf.clearRetainingCapacity();
            try key_buf.appendSlice(allocator, merge.first);
            try key_buf.append(allocator, ' ');
            try key_buf.appendSlice(allocator, merge.second);
            const key_copy = try allocator.dupe(u8, key_buf.items);
            try merge_ranks.put(key_copy, merge.rank);
        }

        return Tokenizer{
            .vocab = vocab,
            .token_to_id = token_to_id,
            .merges = merges_owned,
            .scores = scores,
            .bos_id = bos_id,
            .eos_id = eos_id,
            .prepend_bos = prepend_bos,
            .add_eos_token = add_eos_token,
            .chat_template = chat_template,
            .pretokenizer = pretokenizer,
            .merge_ranks = merge_ranks,
            .merge_ranks_ready = true,
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

    fn utf8SequenceLength(byte0: u8) usize {
        if (byte0 < 0x80) return 1;
        if ((byte0 & 0xE0) == 0xC0) return 2;
        if ((byte0 & 0xF0) == 0xE0) return 3;
        if ((byte0 & 0xF8) == 0xF0) return 4;
        return 1;
    }

    /// Encode UTF-8 text into a token ID slice using the tokenizer's own allocator.
    /// Callers must free the returned slice with `freeEncoded`, not an arbitrary request allocator.
    fn encodeChunk(self: *const Tokenizer, text: []const u8) ![]u32 {
        if (text.len == 0) return try self.allocator.alloc(u32, 0);

        // Start with GPT-2 byte-level encoding: each raw byte maps to a Unicode char
        var symbols: std.ArrayList([]const u8) = .{};
        defer symbols.deinit(self.allocator);

        var owned_symbols: std.ArrayList([]u8) = .{};
        defer {
            for (owned_symbols.items) |sym| self.allocator.free(sym);
            owned_symbols.deinit(self.allocator);
        }

        // SentencePiece and Gemma4 BPE use raw UTF-8 symbols with spaces normalized to ▁.
        // GPT-2/Qwen use byte-level Unicode remapping.
        const use_spm_style_bpe = self.scores != null or self.pretokenizer == .gemma4_bpe;
        if (use_spm_style_bpe) {
            var pos: usize = 0;
            while (pos < text.len) {
                if (text[pos] == ' ') {
                    const copy = try self.allocator.alloc(u8, 3);
                    copy[0] = 0xE2;
                    copy[1] = 0x96;
                    copy[2] = 0x81;
                    try owned_symbols.append(self.allocator, copy);
                    try symbols.append(self.allocator, copy);
                    pos += 1;
                    continue;
                }

                const seq_len = @min(utf8SequenceLength(text[pos]), text.len - pos);
                const copy = try self.allocator.alloc(u8, seq_len);
                @memcpy(copy, text[pos .. pos + seq_len]);
                try owned_symbols.append(self.allocator, copy);
                try symbols.append(self.allocator, copy);
                pos += seq_len;
            }
        } else {
            for (text) |byte| {
                const unicode_char = gpt2ByteToUnicode(byte);
                const len: usize = if (unicode_char[0] < 0x80) 1 else if (unicode_char[0] < 0xE0) 2 else 3;
                const copy = try self.allocator.alloc(u8, len);
                @memcpy(copy, unicode_char[0..len]);
                try owned_symbols.append(self.allocator, copy);
                try symbols.append(self.allocator, copy);
            }
        }

        // Gemma4 uses SentencePiece-style BPE. Its GGUF metadata carries both
        // token scores and a merge list, but the scored piece merges match the
        // reference tokenizer behavior more closely than the raw merge ranks.
        if (self.pretokenizer == .gemma4_bpe and self.scores != null) {
            try self.applySentencePieceMerges(&symbols, &owned_symbols);
        } else if (self.merges.len > 0) {
            // GPT-2/tiktoken-style merge ranks.
            try self.applyMerges(&symbols, &owned_symbols);
        } else if (self.scores != null) {
            // SentencePiece-style scored piece merges.
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

    fn isAsciiLetter(byte: u8) bool {
        return (byte >= 'a' and byte <= 'z') or (byte >= 'A' and byte <= 'Z');
    }

    fn isAsciiDigit(byte: u8) bool {
        return byte >= '0' and byte <= '9';
    }

    fn isAsciiSpace(byte: u8) bool {
        return byte == ' ' or byte == '\t' or byte == '\n' or byte == '\r';
    }

    fn matchAsciiContraction(text: []const u8, start: usize) usize {
        if (start >= text.len or text[start] != '\'') return 0;
        const rest = text[start..];
        const candidates = [_][]const u8{ "'re", "'ve", "'ll", "'s", "'t", "'m", "'d" };
        for (candidates) |candidate| {
            if (rest.len >= candidate.len and std.ascii.eqlIgnoreCase(rest[0..candidate.len], candidate)) {
                return candidate.len;
            }
        }
        return 0;
    }

    fn nextGpt2PretokenChunk(text: []const u8, pos: *usize) []const u8 {
        if (pos.* >= text.len) return text[text.len..text.len];

        const start = pos.*;
        var i = start;

        const const_len = matchAsciiContraction(text, start);
        if (const_len > 0) {
            pos.* = start + const_len;
            return text[start..pos.*];
        }

        if (text[i] == ' ') {
            if (i + 1 < text.len and isAsciiLetter(text[i + 1])) {
                i += 2;
                while (i < text.len and isAsciiLetter(text[i])) : (i += 1) {}
                pos.* = i;
                return text[start..i];
            }
            if (i + 1 < text.len and isAsciiDigit(text[i + 1])) {
                i += 2;
                while (i < text.len and isAsciiDigit(text[i])) : (i += 1) {}
                pos.* = i;
                return text[start..i];
            }
            if (i + 1 < text.len and !isAsciiSpace(text[i + 1]) and !isAsciiLetter(text[i + 1]) and !isAsciiDigit(text[i + 1])) {
                i += 2;
                while (i < text.len and !isAsciiSpace(text[i]) and !isAsciiLetter(text[i]) and !isAsciiDigit(text[i])) : (i += 1) {}
                pos.* = i;
                return text[start..i];
            }
            i += 1;
            while (i < text.len and isAsciiSpace(text[i])) : (i += 1) {}
            pos.* = i;
            return text[start..i];
        }

        if (isAsciiLetter(text[i])) {
            i += 1;
            while (i < text.len and isAsciiLetter(text[i])) : (i += 1) {}
            pos.* = i;
            return text[start..i];
        }

        if (isAsciiDigit(text[i])) {
            i += 1;
            while (i < text.len and isAsciiDigit(text[i])) : (i += 1) {}
            pos.* = i;
            return text[start..i];
        }

        if (isAsciiSpace(text[i])) {
            i += 1;
            while (i < text.len and isAsciiSpace(text[i])) : (i += 1) {}
            pos.* = i;
            return text[start..i];
        }

        i += 1;
        while (i < text.len and !isAsciiSpace(text[i]) and !isAsciiLetter(text[i]) and !isAsciiDigit(text[i])) : (i += 1) {}
        pos.* = i;
        return text[start..i];
    }

    fn nextGemma4PretokenChunk(text: []const u8, pos: *usize) []const u8 {
        if (pos.* >= text.len) return text[text.len..text.len];

        const start = pos.*;
        const is_newline = text[start] == '\n';
        var i = start + 1;
        while (i < text.len and (text[i] == '\n') == is_newline) : (i += 1) {}
        pos.* = i;
        return text[start..i];
    }
    /// Encode UTF-8 text into token IDs using the tokenizer's pretokenizer and merges.
    pub fn encode(self: *const Tokenizer, text: []const u8) ![]u32 {
        if (text.len == 0) return try self.allocator.alloc(u32, 0);
        if (self.pretokenizer == .gemma4_bpe) {
            var tokens: std.ArrayList(u32) = .{};
            errdefer tokens.deinit(self.allocator);

            var pos: usize = 0;
            while (pos < text.len) {
                const chunk = nextGemma4PretokenChunk(text, &pos);
                if (chunk.len == 0) break;

                if (self.token_to_id.get(chunk)) |id| {
                    try tokens.append(self.allocator, id);
                    continue;
                }

                const chunk_tokens = try self.encodeChunk(chunk);
                defer self.allocator.free(chunk_tokens);
                try tokens.appendSlice(self.allocator, chunk_tokens);
            }

            return try tokens.toOwnedSlice(self.allocator);
        }
        if (self.scores != null or self.merges.len == 0 or self.pretokenizer == .legacy) {
            return self.encodeChunk(text);
        }

        var tokens: std.ArrayList(u32) = .{};
        errdefer tokens.deinit(self.allocator);

        var pos: usize = 0;
        while (pos < text.len) {
            const chunk = nextGpt2PretokenChunk(text, &pos);
            if (chunk.len == 0) break;
            const chunk_tokens = try self.encodeChunk(chunk);
            defer self.allocator.free(chunk_tokens);
            try tokens.appendSlice(self.allocator, chunk_tokens);
        }

        return try tokens.toOwnedSlice(self.allocator);
    }

    /// Release a token slice returned by `encode`.
    pub fn freeEncoded(self: *const Tokenizer, tokens: []u32) void {
        self.allocator.free(tokens);
    }

    /// Encode a prompt and prepend BOS when the model expects it.
    /// The returned slice is allocated with `allocator`, so server routes can use a
    /// per-request allocator while the tokenizer keeps owning its internal scratch buffers.
    /// Special tokens (e.g. `<|start_header_id|>`) are resolved via `token_to_id` rather
    /// than being BPE-encoded character by character.
    pub fn encodePrompt(self: *const Tokenizer, text: []const u8, allocator: std.mem.Allocator) ![]u32 {
        const raw_tokens = try self.encodeWithSpecialTokens(text, allocator);
        defer allocator.free(raw_tokens);

        // Determine whether BOS should be prepended, but avoid duplicating it when
        // the chat template already emitted the BOS token as its first special token.
        var prepend_bos = self.shouldPrependBos();
        if (prepend_bos and raw_tokens.len > 0 and raw_tokens[0] == self.bosId()) {
            prepend_bos = false;
        }

        const bos_extra: usize = if (prepend_bos) 1 else 0;
        const prompt_tokens = try allocator.alloc(u32, raw_tokens.len + bos_extra);

        if (prepend_bos) {
            prompt_tokens[0] = self.bosId();
            @memcpy(prompt_tokens[1..], raw_tokens);
        } else {
            @memcpy(prompt_tokens, raw_tokens);
        }

        return prompt_tokens;
    }

    /// Encode text that may contain special token markers (e.g. `<|...|>`).
    /// Special tokens that exist in the vocabulary are mapped directly to their
    /// token IDs; the remaining text segments are BPE-encoded normally.
    /// The returned slice is allocated with `allocator`.
    fn encodeWithSpecialTokens(self: *const Tokenizer, text: []const u8, allocator: std.mem.Allocator) ![]u32 {
        // Fast path: if there are no potential special token markers, just BPE-encode.
        // Check for both `<|...|>` (GPT-2/Llama) and `<...>` (Gemma) style markers.
        if (std.mem.indexOf(u8, text, "<") == null) {
            const bpe = try self.encode(text);
            defer self.freeEncoded(bpe);
            const out = try allocator.alloc(u32, bpe.len);
            @memcpy(out, bpe);
            return out;
        }

        var tokens: std.ArrayList(u32) = .{};
        errdefer tokens.deinit(allocator);

        var pos: usize = 0;
        while (pos < text.len) {
            // Look for the next `<` marker (handles both `<|...|>` and `<...>` formats).
            if (std.mem.indexOfPos(u8, text, pos, "<")) |start| {
                // Try `<|...|>` first, then `<...>`
                // Try to match a special token starting at `start`.
                // Checks <|...|> (GPT-2/Llama) first, then <...> (Gemma).
                var special_end: usize = 0;
                var special_candidate: []const u8 = "";
                var found_special = false;
                // Try <|...|> (GPT-2/Llama style)
                if (start + 2 < text.len and text[start + 1] == '|') {
                    if (std.mem.indexOfPos(u8, text, start + 2, "|>")) |pipe_end| {
                        const candidate = text[start .. pipe_end + 2];
                        if (self.token_to_id.get(candidate) != null) {
                            special_end = pipe_end + 2;
                            special_candidate = candidate;
                            found_special = true;
                        }
                    }
                }
                // Try <...> (Gemma style: <start_of_turn>, <end_of_turn>, etc.)
                if (!found_special) {
                    if (std.mem.indexOfPos(u8, text, start + 1, ">")) |gt_pos| {
                        const candidate = text[start .. gt_pos + 1];
                        if (self.token_to_id.get(candidate) != null) {
                            special_end = gt_pos + 1;
                            special_candidate = candidate;
                            found_special = true;
                        }
                    }
                }

                if (found_special) {
                    const end = special_end;
                    const candidate = special_candidate;

                    if (self.token_to_id.get(candidate)) |special_id| {
                        // BPE-encode any text before this special token.
                        if (start > pos) {
                            const bpe = try self.encode(text[pos..start]);
                            defer self.freeEncoded(bpe);
                            try tokens.appendSlice(allocator, bpe);
                        }
                        try tokens.append(allocator, special_id);
                        pos = end;
                        continue;
                    }
                }
                // The `<...>` pattern was not a known special token.
                // BPE-encode everything up to and including `<` so the outer
                // loop can continue scanning after it.
                const chunk_end = start + 1;
                const bpe = try self.encode(text[pos..chunk_end]);
                defer self.freeEncoded(bpe);
                try tokens.appendSlice(allocator, bpe);
                pos = chunk_end;
            } else {
                // No more `<|` markers — BPE-encode the rest.
                const bpe = try self.encode(text[pos..]);
                defer self.freeEncoded(bpe);
                try tokens.appendSlice(allocator, bpe);
                pos = text.len;
            }
        }

        return try tokens.toOwnedSlice(allocator);
    }

    /// Apply BPE merges in priority order (GPT-2/tiktoken style).
    fn applyMerges(self: *const Tokenizer, symbols: *std.ArrayList([]const u8), owned_symbols: *std.ArrayList([]u8)) !void {
        // Rank lookup is precomputed once at init and shared across every BPE
        // encode call — see Tokenizer.merge_ranks for the rationale.
        const merge_ranks = if (self.merge_ranks_ready) &self.merge_ranks else blk: {
            // Fallback for test constructors that synthesize a Tokenizer
            // directly without calling initFromGGUF.
            break :blk null;
        };
        var fallback_ranks: std.StringHashMap(u32) = undefined;
        var fallback_owned = false;
        defer if (fallback_owned) {
            var it = fallback_ranks.iterator();
            while (it.next()) |entry| self.allocator.free(@constCast(entry.key_ptr.*));
            fallback_ranks.deinit();
        };
        const ranks_ptr: *const std.StringHashMap(u32) = merge_ranks orelse blk: {
            fallback_ranks = std.StringHashMap(u32).init(self.allocator);
            fallback_owned = true;
            var kb: std.ArrayList(u8) = .{};
            defer kb.deinit(self.allocator);
            for (self.merges) |merge| {
                kb.clearRetainingCapacity();
                try kb.appendSlice(self.allocator, merge.first);
                try kb.append(self.allocator, ' ');
                try kb.appendSlice(self.allocator, merge.second);
                const key_copy = try self.allocator.dupe(u8, kb.items);
                try fallback_ranks.put(key_copy, merge.rank);
            }
            break :blk &fallback_ranks;
        };

        // Pre-allocate merge key buffer
        var key_buf: std.ArrayList(u8) = .{};
        defer key_buf.deinit(self.allocator);

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

                if (ranks_ptr.get(key_buf.items)) |rank| {
                    if (rank < best_rank) {
                        best_rank = rank;
                        best_pos = pos;
                    }
                }
            }

            if (best_rank == std.math.maxInt(u32)) break; // No more merges possible

            // Merge the pair: concatenate symbols[best_pos] and symbols[best_pos+1]
            const old_left = symbols.items[best_pos];
            const old_right = symbols.items[best_pos + 1];
            const merged = try std.mem.concat(self.allocator, u8, &.{
                old_left,
                old_right,
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

            const old_left = symbols.items[best_pos];
            const old_right = symbols.items[best_pos + 1];
            const merged = try std.mem.concat(self.allocator, u8, &.{
                old_left,
                old_right,
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

    /// Whether a sampled token ends the current generation turn.
    ///
    /// Always terminates on the configured EOS. Gemma 4 additionally uses
    /// `<eos>=1` and `</s>=212` alongside the primary `<turn|>=106` EOS — we
    /// treat those as EOG too when the chat template is Gemma, but not for
    /// other tokenizers (Qwen token 1 is a plain `"` character).
    pub fn isEndOfGeneration(self: *const Tokenizer, token: u32) bool {
        if (token == self.eos_id) return true;
        const tmpl = self.chat_template orelse return false;
        const is_gemma4 = std.mem.indexOf(u8, tmpl, "<|turn>") != null;
        if (is_gemma4 and (token == 1 or token == 212)) return true;
        return false;
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

    /// Build prompt tokens following GGUF BOS/EOS policy.
    pub fn preparePromptTokens(self: *const Tokenizer, raw_tokens: []const u32) ![]u32 {
        const prefix_len: usize = if (self.shouldPrependBos()) 1 else 0;
        const suffix_len: usize = if (self.add_eos_token) 1 else 0;
        const prompt_tokens = try self.allocator.alloc(u32, raw_tokens.len + prefix_len + suffix_len);

        var pos: usize = 0;
        if (self.shouldPrependBos()) {
            prompt_tokens[pos] = self.bosId();
            pos += 1;
        }
        @memcpy(prompt_tokens[pos .. pos + raw_tokens.len], raw_tokens);
        pos += raw_tokens.len;
        if (self.add_eos_token) {
            prompt_tokens[pos] = self.eos_id;
        }

        return prompt_tokens;
    }

    /// Decode a token ID to UTF-8 text, reversing the GPT-2 byte-to-unicode mapping.
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
            } else if (cp == 0x2581) blk: {
                // SentencePiece word boundary marker (▁) → space
                break :blk ' ';
            } else blk: {
                // Pass through raw UTF-8 for non-ASCII codepoints (CJK, emoji, etc.)
                // instead of replacing with '?'
                if (cp_len <= buf.len - out) {
                    @memcpy(buf[out..][0..cp_len], gpt2_text[i - cp_len ..][0..cp_len]);
                    out += cp_len;
                    continue;
                }
                break :blk '?';
            };
            buf[out] = orig_byte;
            out += 1;
        }
        return buf[0..out];
    }

    /// Options controlling chat template rendering behavior.
    pub const ChatTemplateOptions = struct {
        enable_thinking: ?bool = null,
        add_generation_prompt: bool = true,
        /// When true, skip the thinking template entirely even if the tokenizer supports it.
        skip_thinking_template: bool = false,
    };

    fn appendTrimmed(dst: []u8, pos: *usize, text: []const u8) !void {
        const trimmed = std.mem.trim(u8, text, " \t\r\n");
        if (pos.* + trimmed.len > dst.len) return error.BufferTooSmall;
        @memcpy(dst[pos.*..][0..trimmed.len], trimmed);
        pos.* += trimmed.len;
    }

    fn appendGemma4ThinkingStripped(dst: []u8, pos: *usize, text: []const u8) !void {
        var src_pos: usize = 0;
        var wrote_any = false;
        while (src_pos < text.len) {
            if (std.mem.indexOfPos(u8, text, src_pos, "<|channel>")) |open_idx| {
                const chunk = if (wrote_any) text[src_pos..open_idx] else std.mem.trimLeft(u8, text[src_pos..open_idx], " \t\r\n");
                if (pos.* + chunk.len > dst.len) return error.BufferTooSmall;
                @memcpy(dst[pos.*..][0..chunk.len], chunk);
                pos.* += chunk.len;
                wrote_any = wrote_any or chunk.len > 0;

                if (std.mem.indexOfPos(u8, text, open_idx, "<channel|>")) |close_idx| {
                    src_pos = close_idx + "<channel|>".len;
                } else {
                    break;
                }
            } else {
                const chunk = if (wrote_any) std.mem.trimRight(u8, text[src_pos..], " \t\r\n") else std.mem.trim(u8, text[src_pos..], " \t\r\n");
                if (pos.* + chunk.len > dst.len) return error.BufferTooSmall;
                @memcpy(dst[pos.*..][0..chunk.len], chunk);
                pos.* += chunk.len;
                break;
            }
        }
    }

    /// Return whether the model's chat template supports an explicit thinking toggle.
    pub fn supportsThinkingToggle(self: *const Tokenizer) bool {
        return if (self.chat_template) |tmpl|
            std.mem.indexOf(u8, tmpl, "enable_thinking") != null and
                (std.mem.indexOf(u8, tmpl, "<think>") != null or
                    std.mem.indexOf(u8, tmpl, "<|think|>") != null)
        else
            false;
    }

    /// Apply chat template to role/content pairs. Returns formatted prompt in buf.
    pub fn applyChatTemplate(self: *const Tokenizer, roles: []const []const u8, contents: []const []const u8, buf: []u8) ![]const u8 {
        return self.applyChatTemplateWithOptions(roles, contents, .{}, buf);
    }

    /// Apply chat template to role/content pairs with explicit thinking control.
    pub fn applyChatTemplateWithOptions(self: *const Tokenizer, roles: []const []const u8, contents: []const []const u8, options: ChatTemplateOptions, buf: []u8) ![]const u8 {
        var pos: usize = 0;
        const template_kind = self.detectTemplateKind();
        const supports_thinking = self.supportsThinkingToggle();
        const n = @min(roles.len, contents.len);
        switch (template_kind) {
            .chatml => {
                for (0..n) |i| {
                    const written = std.fmt.bufPrint(buf[pos..], "<|im_start|>{s}\n{s}<|im_end|>\n", .{ roles[i], contents[i] }) catch return error.BufferTooSmall;
                    pos += written.len;
                }
                if (options.add_generation_prompt) {
                    const suffix = if (supports_thinking and !options.skip_thinking_template) blk: {
                        if (options.enable_thinking orelse false) {
                            break :blk std.fmt.bufPrint(buf[pos..], "<|im_start|>assistant\n<think>\n", .{}) catch return error.BufferTooSmall;
                        }
                        break :blk std.fmt.bufPrint(buf[pos..], "<|im_start|>assistant\n<think>\n\n</think>\n\n", .{}) catch return error.BufferTooSmall;
                    } else std.fmt.bufPrint(buf[pos..], "<|im_start|>assistant\n", .{}) catch return error.BufferTooSmall;
                    pos += suffix.len;
                }
            },
            .llama3 => {
                const header = std.fmt.bufPrint(buf[pos..], "<|begin_of_text|>", .{}) catch return error.BufferTooSmall;
                pos += header.len;
                for (0..n) |i| {
                    const written = std.fmt.bufPrint(buf[pos..], "<|start_header_id|>{s}<|end_header_id|>\n\n{s}<|eot_id|>", .{ roles[i], contents[i] }) catch return error.BufferTooSmall;
                    pos += written.len;
                }
                if (options.add_generation_prompt) {
                    const suffix = std.fmt.bufPrint(buf[pos..], "<|start_header_id|>assistant<|end_header_id|>\n\n", .{}) catch return error.BufferTooSmall;
                    pos += suffix.len;
                }
            },
            .gemma => {
                // Gemma format: <bos> + per-message <turn_start>role\ncontent<turn_end>\n
                // Gemma 2/3 use <start_of_turn>/<end_of_turn>, Gemma 4 uses <|turn>/<turn|>.
                const tmpl = self.chat_template orelse "";
                const is_gemma4 = std.mem.indexOf(u8, tmpl, "<|turn>") != null;
                const turn_start: []const u8 = if (is_gemma4) "<|turn>" else "<start_of_turn>";
                const turn_end: []const u8 = if (is_gemma4) "<turn|>" else "<end_of_turn>";
                const bos = std.fmt.bufPrint(buf[pos..], "<bos>", .{}) catch return error.BufferTooSmall;
                pos += bos.len;
                if (is_gemma4) {
                    var start_idx: usize = 0;
                    const thinking_enabled = options.enable_thinking orelse false;
                    const has_leading_system = n > 0 and
                        (std.mem.eql(u8, roles[0], "system") or std.mem.eql(u8, roles[0], "developer"));
                    if (thinking_enabled or has_leading_system) {
                        const system_turn = std.fmt.bufPrint(buf[pos..], "{s}system\n", .{turn_start}) catch return error.BufferTooSmall;
                        pos += system_turn.len;
                        if (thinking_enabled) {
                            const think_token = std.fmt.bufPrint(buf[pos..], "<|think|>", .{}) catch return error.BufferTooSmall;
                            pos += think_token.len;
                        }
                        if (has_leading_system) {
                            const system_text = std.fmt.bufPrint(buf[pos..], "{s}", .{std.mem.trim(u8, contents[0], " \t\r\n")}) catch return error.BufferTooSmall;
                            pos += system_text.len;
                            start_idx = 1;
                        }
                        const system_end = std.fmt.bufPrint(buf[pos..], "{s}\n", .{turn_end}) catch return error.BufferTooSmall;
                        pos += system_end.len;
                    }
                    for (start_idx..n) |i| {
                        const rendered_role = if (std.mem.eql(u8, roles[i], "assistant")) "model" else roles[i];
                        const prefix = std.fmt.bufPrint(buf[pos..], "{s}{s}\n", .{ turn_start, rendered_role }) catch return error.BufferTooSmall;
                        pos += prefix.len;
                        if (std.mem.eql(u8, rendered_role, "model")) {
                            try appendGemma4ThinkingStripped(buf, &pos, contents[i]);
                        } else {
                            try appendTrimmed(buf, &pos, contents[i]);
                        }
                        const suffix = std.fmt.bufPrint(buf[pos..], "{s}\n", .{turn_end}) catch return error.BufferTooSmall;
                        pos += suffix.len;
                    }
                    if (options.add_generation_prompt) {
                        const suffix = std.fmt.bufPrint(buf[pos..], "{s}model\n", .{turn_start}) catch return error.BufferTooSmall;
                        pos += suffix.len;
                        if (!options.skip_thinking_template and !thinking_enabled) {
                            const channel = std.fmt.bufPrint(buf[pos..], "<|channel>thought\n<channel|>", .{}) catch return error.BufferTooSmall;
                            pos += channel.len;
                        }
                    }
                } else {
                    for (0..n) |i| {
                        const written = std.fmt.bufPrint(buf[pos..], "{s}{s}\n{s}{s}\n", .{ turn_start, roles[i], contents[i], turn_end }) catch return error.BufferTooSmall;
                        pos += written.len;
                    }
                    if (options.add_generation_prompt) {
                        const suffix = std.fmt.bufPrint(buf[pos..], "{s}model\n", .{turn_start}) catch return error.BufferTooSmall;
                        pos += suffix.len;
                    }
                }
            },
            .openai_moe => {
                // OpenAI MoE (gpt-oss): <|start|>role<|message|>content<|end|>
                for (0..n) |i| {
                    const end_tag = if (std.mem.eql(u8, roles[i], "assistant")) "<|return|>" else "<|end|>";
                    const written = std.fmt.bufPrint(buf[pos..], "<|start|>{s}<|message|>{s}{s}", .{ roles[i], contents[i], end_tag }) catch return error.BufferTooSmall;
                    pos += written.len;
                }
                if (options.add_generation_prompt) {
                    // Match llama.cpp: omit <|message|> — model generates it
                    const suffix = std.fmt.bufPrint(buf[pos..], "<|start|>assistant", .{}) catch return error.BufferTooSmall;
                    pos += suffix.len;
                }
            },
            .generic => {
                for (0..n) |i| {
                    const written = std.fmt.bufPrint(buf[pos..], "[{s}]: {s}\n", .{ roles[i], contents[i] }) catch return error.BufferTooSmall;
                    pos += written.len;
                }
            },
        }
        return buf[0..pos];
    }

    const TemplateKind = enum { chatml, llama3, gemma, openai_moe, generic };

    /// Return the detected chat template kind as a human-readable string (e.g. "chatml", "openai_moe").
    pub fn detectTemplateKindName(self: *const Tokenizer) []const u8 {
        return @tagName(self.detectTemplateKind());
    }

    fn detectTemplateKind(self: *const Tokenizer) TemplateKind {
        const tmpl = self.chat_template orelse return .chatml;
        if (std.mem.indexOf(u8, tmpl, "im_start") != null) return .chatml;
        if (std.mem.indexOf(u8, tmpl, "start_header_id") != null) return .llama3;
        if (std.mem.indexOf(u8, tmpl, "start_of_turn") != null or
            std.mem.indexOf(u8, tmpl, "<|turn>") != null) return .gemma;
        if (std.mem.indexOf(u8, tmpl, "<|start|>") != null and
            std.mem.indexOf(u8, tmpl, "<|message|>") != null) return .openai_moe;
        return .generic;
    }

    /// Release tokenizer-owned vocabulary tables, merges, and optional score arrays.
    pub fn deinit(self: *Tokenizer) void {
        if (self.merge_ranks_ready) {
            var it = self.merge_ranks.iterator();
            while (it.next()) |entry| self.allocator.free(@constCast(entry.key_ptr.*));
            self.merge_ranks.deinit();
            self.merge_ranks_ready = false;
        }
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

test "initFromGGUF populates merge_ranks cache when merges are present" {
    // Regression guard for the BPE merge-rank cache. If initFromGGUF ever
    // stops populating merge_ranks, every applyMerges call will hit the
    // fallback path and rebuild a 151k-entry hashmap per chat chunk —
    // which made each chat request after the first take tens of minutes.
    const allocator = std.testing.allocator;

    var gf = gguf.GGUFFile{
        .version = .v3,
        .tensor_count = 0,
        .metadata = .{},
        .tensors = .{},
        .tensor_data_offset = 0,
        .allocator = allocator,
    };
    defer gf.deinit();

    const tokens = try allocator.alloc(gguf.MetadataValue, 4);
    tokens[0] = .{ .string = try allocator.dupe(u8, "a") };
    tokens[1] = .{ .string = try allocator.dupe(u8, "b") };
    tokens[2] = .{ .string = try allocator.dupe(u8, "ab") };
    tokens[3] = .{ .string = try allocator.dupe(u8, "c") };

    const merges = try allocator.alloc(gguf.MetadataValue, 1);
    merges[0] = .{ .string = try allocator.dupe(u8, "a b") };

    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.tokens"), .{ .array = tokens });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.merges"), .{ .array = merges });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.model"), .{ .string = try allocator.dupe(u8, "gpt2") });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "general.architecture"), .{ .string = try allocator.dupe(u8, "qwen35") });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.eos_token_id"), .{ .uint32 = 3 });

    var tok = try Tokenizer.initFromGGUF(&gf, allocator);
    defer tok.deinit();

    try std.testing.expect(tok.merge_ranks_ready);
    try std.testing.expectEqual(@as(u32, 0), tok.merge_ranks.get("a b").?);
}

test "initFromGGUF omits BOS for qwen35 family (no BOS in GGUF)" {
    const allocator = std.testing.allocator;

    var gf = gguf.GGUFFile{
        .version = .v3,
        .tensor_count = 0,
        .metadata = .{},
        .tensors = .{},
        .tensor_data_offset = 0,
        .allocator = allocator,
    };
    defer gf.deinit();

    const tokens = try allocator.alloc(gguf.MetadataValue, 3);
    tokens[0] = .{ .string = try allocator.dupe(u8, "a") };
    tokens[1] = .{ .string = try allocator.dupe(u8, "b") };
    tokens[2] = .{ .string = try allocator.dupe(u8, "c") };

    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.tokens"), .{ .array = tokens });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.model"), .{ .string = try allocator.dupe(u8, "gpt2") });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "general.architecture"), .{ .string = try allocator.dupe(u8, "qwen35") });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.eos_token_id"), .{ .uint32 = 2 });

    var tok = try Tokenizer.initFromGGUF(&gf, allocator);
    defer tok.deinit();

    // Qwen3.5 GGUFs omit BOS metadata — should NOT prepend BOS
    try std.testing.expect(!tok.shouldPrependBos());
}

test "initFromGGUF omits BOS for gpt-oss prompts by default" {
    const allocator = std.testing.allocator;

    var gf = gguf.GGUFFile{
        .version = .v3,
        .tensor_count = 0,
        .metadata = .{},
        .tensors = .{},
        .tensor_data_offset = 0,
        .allocator = allocator,
    };
    defer gf.deinit();

    const tokens = try allocator.alloc(gguf.MetadataValue, 3);
    tokens[0] = .{ .string = try allocator.dupe(u8, "a") };
    tokens[1] = .{ .string = try allocator.dupe(u8, "b") };
    tokens[2] = .{ .string = try allocator.dupe(u8, "c") };

    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.tokens"), .{ .array = tokens });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.model"), .{ .string = try allocator.dupe(u8, "gpt2") });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "general.architecture"), .{ .string = try allocator.dupe(u8, "gpt-oss") });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.bos_token_id"), .{ .uint32 = 199998 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.eos_token_id"), .{ .uint32 = 200002 });

    var tok = try Tokenizer.initFromGGUF(&gf, allocator);
    defer tok.deinit();

    try std.testing.expect(!tok.shouldPrependBos());
}

test "initFromGGUF respects gemma4 add_bos_token=false" {
    const allocator = std.testing.allocator;

    var gf = gguf.GGUFFile{
        .version = .v3,
        .tensor_count = 0,
        .metadata = .{},
        .tensors = .{},
        .tensor_data_offset = 0,
        .allocator = allocator,
    };
    defer gf.deinit();

    const tokens = try allocator.alloc(gguf.MetadataValue, 3);
    tokens[0] = .{ .string = try allocator.dupe(u8, "a") };
    tokens[1] = .{ .string = try allocator.dupe(u8, "b") };
    tokens[2] = .{ .string = try allocator.dupe(u8, "<bos>") };

    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.tokens"), .{ .array = tokens });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.model"), .{ .string = try allocator.dupe(u8, "gemma4") });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "general.architecture"), .{ .string = try allocator.dupe(u8, "gemma4") });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.bos_token_id"), .{ .uint32 = 2 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.eos_token_id"), .{ .uint32 = 1 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.add_bos_token"), .{ .bool_ = false });

    var tok = try Tokenizer.initFromGGUF(&gf, allocator);
    defer tok.deinit();

    try std.testing.expect(!tok.shouldPrependBos());
}

test "initFromGGUF respects gemma4 add_bos_token=true" {
    const allocator = std.testing.allocator;

    var gf = gguf.GGUFFile{
        .version = .v3,
        .tensor_count = 0,
        .metadata = .{},
        .tensors = .{},
        .tensor_data_offset = 0,
        .allocator = allocator,
    };
    defer gf.deinit();

    const tokens = try allocator.alloc(gguf.MetadataValue, 3);
    tokens[0] = .{ .string = try allocator.dupe(u8, "a") };
    tokens[1] = .{ .string = try allocator.dupe(u8, "b") };
    tokens[2] = .{ .string = try allocator.dupe(u8, "<bos>") };

    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.tokens"), .{ .array = tokens });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.model"), .{ .string = try allocator.dupe(u8, "gemma4") });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "general.architecture"), .{ .string = try allocator.dupe(u8, "gemma4") });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.bos_token_id"), .{ .uint32 = 2 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.eos_token_id"), .{ .uint32 = 1 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "tokenizer.ggml.add_bos_token"), .{ .bool_ = true });

    var tok = try Tokenizer.initFromGGUF(&gf, allocator);
    defer tok.deinit();

    try std.testing.expect(tok.shouldPrependBos());
}

test "preparePromptTokens skips BOS when prepend_bos is false" {
    var tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .bos_id = @as(u32, 11),
        .eos_id = 42,
        .prepend_bos = false,
        .scores = null,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    const raw = [_]u32{ 760, 6511, 314, 9338, 369 };
    const prompt = try tok.preparePromptTokens(&raw);
    defer std.testing.allocator.free(prompt);

    try std.testing.expectEqualSlices(u32, &raw, prompt);
}

test "preparePromptTokens prepends BOS when prepend_bos is true" {
    var tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .scores = null,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    const raw = [_]u32{ 10, 20, 30 };
    const prompt = try tok.preparePromptTokens(&raw);
    defer std.testing.allocator.free(prompt);

    try std.testing.expectEqual(@as(usize, 4), prompt.len);
    try std.testing.expectEqual(@as(u32, 1), prompt[0]);
    try std.testing.expectEqualSlices(u32, &raw, prompt[1..]);
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

test "encodePrompt supports distinct tokenizer and output allocators" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();

    var tok = Tokenizer{
        .vocab = &.{ "h", "i" },
        .token_to_id = std.StringHashMap(u32).init(arena.allocator()),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .allocator = arena.allocator(),
    };
    defer tok.token_to_id.deinit();

    try tok.token_to_id.put("h", 10);
    try tok.token_to_id.put("i", 11);

    const prompt_tokens = try tok.encodePrompt("hi", std.testing.allocator);
    defer std.testing.allocator.free(prompt_tokens);

    try std.testing.expectEqualSlices(u32, &.{ 1, 10, 11 }, prompt_tokens);
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

test "applyChatTemplate gemma4 defaults to closed thought channel prompt" {
    var tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 2,
        .eos_id = 106,
        .prepend_bos = true,
        .chat_template = "{%- if enable_thinking is defined and enable_thinking -%}<|turn>system\n<|think|><turn|>\n{%- endif -%}<|turn>{{ role }}\n{{ content }}<turn|>",
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    var buf: [256]u8 = undefined;
    const roles = [_][]const u8{"user"};
    const contents = [_][]const u8{"Hello"};
    const result = try tok.applyChatTemplate(&roles, &contents, &buf);

    try std.testing.expect(std.mem.indexOf(u8, result, "<bos><|turn>user\nHello<turn|>\n<|turn>model\n<|channel>thought\n<channel|>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|turn>assistant\n") == null);
}

test "applyChatTemplate gemma4 rewrites assistant history to model role" {
    var tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 2,
        .eos_id = 106,
        .prepend_bos = true,
        .chat_template = "{%- if enable_thinking is defined and enable_thinking -%}<|turn>system\n<|think|><turn|>\n{%- endif -%}<|turn>{{ role }}\n{{ content }}<turn|>",
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    var buf: [256]u8 = undefined;
    const roles = [_][]const u8{ "user", "assistant" };
    const contents = [_][]const u8{ "Hello", "Hi there" };
    const result = try tok.applyChatTemplateWithOptions(&roles, &contents, .{ .add_generation_prompt = false }, &buf);

    try std.testing.expect(std.mem.indexOf(u8, result, "<|turn>assistant\n") == null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|turn>model\nHi there<turn|>\n") != null);
}

test "applyChatTemplate gemma4 thinking mode emits synthetic system turn" {
    var tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 2,
        .eos_id = 106,
        .prepend_bos = true,
        .chat_template = "{%- if enable_thinking is defined and enable_thinking -%}<|turn>system\n<|think|><turn|>\n{%- endif -%}<|turn>{{ role }}\n{{ content }}<turn|>",
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    var buf: [256]u8 = undefined;
    const roles = [_][]const u8{"user"};
    const contents = [_][]const u8{"Hello"};
    const result = try tok.applyChatTemplateWithOptions(&roles, &contents, .{ .enable_thinking = true }, &buf);

    try std.testing.expect(std.mem.indexOf(u8, result, "<bos><|turn>system\n<|think|><turn|>\n<|turn>user\nHello<turn|>\n<|turn>model\n") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|channel>thought\n<channel|>") == null);
}

test "applyChatTemplate llama3 template uses header tokens" {
    const tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .chat_template = "{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>{% endfor %}",
        .allocator = std.testing.allocator,
    };
    var buf: [1024]u8 = undefined;
    const roles = [_][]const u8{ "system", "user" };
    const contents = [_][]const u8{ "You help.", "Hi" };
    const result = try tok.applyChatTemplate(&roles, &contents, &buf);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|begin_of_text|>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|start_header_id|>system<|end_header_id|>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "You help.<|eot_id|>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|start_header_id|>user<|end_header_id|>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "Hi<|eot_id|>") != null);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|start_header_id|>assistant<|end_header_id|>") != null);
}

test "applyChatTemplate qwen thinking template emits empty think block for generation" {
    var tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .chat_template =
        \\{%- if add_generation_prompt %}
        \\  {{- '<|im_start|>assistant\n' }}
        \\  {%- if enable_thinking is defined and enable_thinking is true %}
        \\    {{- '<think>\n' }}
        \\  {%- else %}
        \\    {{- '<think>\n\n</think>\n\n' }}
        \\  {%- endif %}
        \\{%- endif %}
        ,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    var buf: [1024]u8 = undefined;
    const roles = [_][]const u8{"user"};
    const contents = [_][]const u8{"Hello"};
    const result = try tok.applyChatTemplate(&roles, &contents, &buf);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>user\nHello<|im_end|>\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, result, "<|im_start|>assistant\n<think>\n\n</think>\n\n"));
}

test "applyChatTemplateWithOptions emits open think block when enabled" {
    var tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .chat_template =
        \\{%- if add_generation_prompt %}
        \\  {{- '<|im_start|>assistant\n' }}
        \\  {%- if enable_thinking is defined and enable_thinking is true %}
        \\    {{- '<think>\n' }}
        \\  {%- else %}
        \\    {{- '<think>\n\n</think>\n\n' }}
        \\  {%- endif %}
        \\{%- endif %}
        ,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    var buf: [1024]u8 = undefined;
    const roles = [_][]const u8{"user"};
    const contents = [_][]const u8{"Hello"};
    const result = try tok.applyChatTemplateWithOptions(&roles, &contents, .{ .enable_thinking = true }, &buf);
    try std.testing.expect(std.mem.indexOf(u8, result, "<|im_start|>user\nHello<|im_end|>\n") != null);
    try std.testing.expect(std.mem.endsWith(u8, result, "<|im_start|>assistant\n<think>\n"));
    try std.testing.expect(std.mem.indexOf(u8, result, "</think>") == null);
}

test "supportsThinkingToggle detects qwen enable_thinking templates" {
    var tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .chat_template =
        \\{%- if add_generation_prompt %}
        \\  {{- '<|im_start|>assistant\n' }}
        \\  {%- if enable_thinking is defined and enable_thinking is true %}
        \\    {{- '<think>\n' }}
        \\  {%- else %}
        \\    {{- '<think>\n\n</think>\n\n' }}
        \\  {%- endif %}
        \\{%- endif %}
        ,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();
    try std.testing.expect(tok.supportsThinkingToggle());
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

test "applyChatTemplateWithOptions can omit generation prompt" {
    var tok = Tokenizer{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .bos_id = null,
        .eos_id = 2,
        .prepend_bos = false,
        .scores = null,
        .chat_template = null,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    const roles = [_][]const u8{ "user", "assistant" };
    const contents = [_][]const u8{ "hello", "world" };
    var buf: [256]u8 = undefined;
    const result = try tok.applyChatTemplateWithOptions(&roles, &contents, .{ .add_generation_prompt = false }, &buf);
    try std.testing.expectEqualStrings("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\nworld<|im_end|>\n", result);
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

test "encode gemma4_bpe normalizes spaces to ▁ before BPE merges" {
    const vocab = [_][]const u8{ "▁", "a", "b", "▁a", "▁ab" };
    const merges = [_]Tokenizer.Merge{
        .{ .first = "▁", .second = "a", .rank = 0 },
        .{ .first = "▁a", .second = "b", .rank = 1 },
    };
    var tok = Tokenizer{
        .vocab = &vocab,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &merges,
        .scores = null,
        .bos_id = null,
        .eos_id = 0,
        .prepend_bos = false,
        .pretokenizer = .gemma4_bpe,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    try tok.token_to_id.put("▁", 0);
    try tok.token_to_id.put("a", 1);
    try tok.token_to_id.put("b", 2);
    try tok.token_to_id.put("▁a", 3);
    try tok.token_to_id.put("▁ab", 4);

    const tokens = try tok.encode(" ab");
    defer std.testing.allocator.free(tokens);

    try std.testing.expectEqualSlices(u32, &.{4}, tokens);
}

test "encode gemma4_bpe keeps newline runs as direct tokens when present" {
    const vocab = [_][]const u8{ "\n\n", "\n", "a" };
    const merges = [_]Tokenizer.Merge{};
    var tok = Tokenizer{
        .vocab = &vocab,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &merges,
        .scores = null,
        .bos_id = null,
        .eos_id = 0,
        .prepend_bos = false,
        .pretokenizer = .gemma4_bpe,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    try tok.token_to_id.put("\n\n", 0);
    try tok.token_to_id.put("\n", 1);
    try tok.token_to_id.put("a", 2);

    const tokens = try tok.encode("\n\n");
    defer std.testing.allocator.free(tokens);

    try std.testing.expectEqualSlices(u32, &.{0}, tokens);
}

test "encode gemma4_bpe prefers score merges over merge ranks when both exist" {
    const vocab = [_][]const u8{ "a", "b", "c", "ab", "bc" };
    const merges = [_]Tokenizer.Merge{
        .{ .first = "a", .second = "b", .rank = 0 },
        .{ .first = "b", .second = "c", .rank = 1 },
    };
    const scores = try std.testing.allocator.dupe(f32, &[_]f32{
        -1.0, // a
        -1.0, // b
        -1.0, // c
        0.5, // ab
        2.0, // bc
    });
    defer std.testing.allocator.free(scores);

    var tok = Tokenizer{
        .vocab = &vocab,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &merges,
        .scores = scores,
        .bos_id = null,
        .eos_id = 0,
        .prepend_bos = false,
        .pretokenizer = .gemma4_bpe,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    try tok.token_to_id.put("a", 0);
    try tok.token_to_id.put("b", 1);
    try tok.token_to_id.put("c", 2);
    try tok.token_to_id.put("ab", 3);
    try tok.token_to_id.put("bc", 4);

    const tokens = try tok.encode("abc");
    defer std.testing.allocator.free(tokens);

    try std.testing.expectEqualSlices(u32, &.{ 0, 4 }, tokens);
}

test "encodeWithSpecialTokens maps known special tokens to their IDs" {
    // Build a small vocabulary: single-char GPT-2 tokens + two special tokens.
    // GPT-2 BPE stores printable ASCII as-is, so "h"/"i" match the byte encoding.
    const vocab = [_][]const u8{ "h", "i", "<|special|>" };
    var tok = Tokenizer{
        .vocab = &vocab,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = null,
        .eos_id = 2,
        .prepend_bos = false,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    try tok.token_to_id.put("h", 0);
    try tok.token_to_id.put("i", 1);
    try tok.token_to_id.put("<|special|>", 2);

    const tokens = try tok.encodeWithSpecialTokens("hi<|special|>hi", std.testing.allocator);
    defer std.testing.allocator.free(tokens);

    // Expect: h(0), i(1), <|special|>(2), h(0), i(1)
    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 0, 1 }, tokens);
}

test "encodeWithSpecialTokens falls back for unknown <|...|> patterns" {
    const vocab = [_][]const u8{ "h", "i" };
    var tok = Tokenizer{
        .vocab = &vocab,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = null,
        .eos_id = 1,
        .prepend_bos = false,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    try tok.token_to_id.put("h", 0);
    try tok.token_to_id.put("i", 1);

    // <|nope|> is not in the vocab, so it should be BPE-encoded character by character.
    const tokens = try tok.encodeWithSpecialTokens("h<|nope|>i", std.testing.allocator);
    defer std.testing.allocator.free(tokens);

    // First and last token must be h(0) and i(1); middle tokens are byte-fallback.
    try std.testing.expect(tokens.len > 2);
    try std.testing.expectEqual(@as(u32, 0), tokens[0]);
    try std.testing.expectEqual(@as(u32, 1), tokens[tokens.len - 1]);
}

test "encodeWithSpecialTokens fast path when no special markers present" {
    const vocab = [_][]const u8{ "h", "i" };
    var tok = Tokenizer{
        .vocab = &vocab,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = null,
        .eos_id = 1,
        .prepend_bos = false,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    try tok.token_to_id.put("h", 0);
    try tok.token_to_id.put("i", 1);

    const tokens = try tok.encodeWithSpecialTokens("hi", std.testing.allocator);
    defer std.testing.allocator.free(tokens);

    try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, tokens);
}

test "encodePrompt skips duplicate BOS when chat template already emits it" {
    // Simulate Llama 3 scenario: BOS is token 100, and the text starts with <|begin_of_text|>.
    const vocab = [_][]const u8{ "h", "i", "<|begin_of_text|>" };
    var tok = Tokenizer{
        .vocab = &vocab,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 2, // <|begin_of_text|> is token 2 in our mini vocab
        .eos_id = 1,
        .prepend_bos = true,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    try tok.token_to_id.put("h", 0);
    try tok.token_to_id.put("i", 1);
    try tok.token_to_id.put("<|begin_of_text|>", 2);

    const tokens = try tok.encodePrompt("<|begin_of_text|>hi", std.testing.allocator);
    defer std.testing.allocator.free(tokens);

    // BOS (2) should appear exactly once, not twice.
    try std.testing.expectEqualSlices(u32, &.{ 2, 0, 1 }, tokens);
}

test "encodePrompt prepends BOS when text has no leading special BOS" {
    const vocab = [_][]const u8{ "h", "i", "<|special|>" };
    var tok = Tokenizer{
        .vocab = &vocab,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 2,
        .eos_id = 1,
        .prepend_bos = true,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    try tok.token_to_id.put("h", 0);
    try tok.token_to_id.put("i", 1);
    try tok.token_to_id.put("<|special|>", 2);

    const tokens = try tok.encodePrompt("hi", std.testing.allocator);
    defer std.testing.allocator.free(tokens);

    // BOS (2) should be prepended since the text doesn't start with a BOS token.
    try std.testing.expectEqualSlices(u32, &.{ 2, 0, 1 }, tokens);
}

test "encodeWithSpecialTokens handles consecutive special tokens" {
    const vocab = [_][]const u8{ "<|a|>", "<|b|>", "<|c|>" };
    var tok = Tokenizer{
        .vocab = &vocab,
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = null,
        .eos_id = 0,
        .prepend_bos = false,
        .allocator = std.testing.allocator,
    };
    defer tok.token_to_id.deinit();

    try tok.token_to_id.put("<|a|>", 0);
    try tok.token_to_id.put("<|b|>", 1);
    try tok.token_to_id.put("<|c|>", 2);

    const tokens = try tok.encodeWithSpecialTokens("<|a|><|b|><|c|>", std.testing.allocator);
    defer std.testing.allocator.free(tokens);

    try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2 }, tokens);
}
