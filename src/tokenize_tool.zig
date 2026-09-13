//! CPU-only tokenizer check. Parses a GGUF's metadata, tokenizes a text file
//! with ZINC's tokenizer through `encodePrompt` (the path the CLI and server
//! use) and prints the token ids as a JSON array on stderr. It loads no tensors
//! and opens no GPU device, so it can run beside a live server and be diffed
//! against the reference `llama-tokenize --ids`.
//! Usage: zinc-tokenize <model.gguf> <text-file>
//! @section Tokenization
const std = @import("std");
const gguf = @import("model/gguf.zig");
const tokenizer_mod = @import("model/tokenizer.zig");

/// Entry point: `zinc-tokenize <model.gguf> <text-file>`.
pub fn main() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .{};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);
    if (args.len != 3) {
        std.debug.print("usage: zinc-tokenize <model.gguf> <text-file>\n", .{});
        std.process.exit(2);
    }

    const file = try std.fs.cwd().openFile(args[1], .{});
    defer file.close();
    const stat = try file.stat();
    const data = try std.posix.mmap(null, stat.size, std.posix.PROT.READ, .{ .TYPE = .PRIVATE }, file.handle, 0);
    defer std.posix.munmap(data);

    var gf = try gguf.parseWithOptions(data, allocator, .{ .log_summary = false });
    defer gf.deinit();
    var tokenizer = try tokenizer_mod.Tokenizer.initFromGGUF(&gf, allocator);
    defer tokenizer.deinit();

    const text = try std.fs.cwd().readFileAlloc(allocator, args[2], 256 * 1024 * 1024);
    defer allocator.free(text);
    const ids = try tokenizer.encodePrompt(text, allocator);
    defer allocator.free(ids);

    var out: std.ArrayList(u8) = .{};
    defer out.deinit(allocator);
    try out.append(allocator, '[');
    for (ids, 0..) |id, i| {
        if (i > 0) try out.appendSlice(allocator, ", ");
        var buf: [16]u8 = undefined;
        try out.appendSlice(allocator, try std.fmt.bufPrint(&buf, "{d}", .{id}));
    }
    try out.append(allocator, ']');
    std.debug.print("{s}\n", .{out.items});
}

test {
    _ = @import("model/tokenizer.zig");
    _ = @import("model/unicode_classes.zig");
}
