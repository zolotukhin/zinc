const std = @import("std");
const host_io = @import("host_io");
const gguf = @import("gguf");

fn printKey(gf: *const gguf.GGUFFile, key: []const u8) void {
    std.debug.print("{s}=", .{key});
    if (gf.metadata.get(key)) |v| {
        switch (v) {
            .string => |s| std.debug.print("string:{s}\n", .{s}),
            .uint32 => |x| std.debug.print("u32:{d}\n", .{x}),
            .bool_ => |x| std.debug.print("bool:{}\n", .{x}),
            .array => |arr| {
                const elem_type = if (arr.len > 0) @tagName(arr[0]) else "empty";
                std.debug.print("array[len={d},first_type={s}]\n", .{ arr.len, elem_type });
            },
            else => std.debug.print("{s}\n", .{@tagName(v)}),
        }
    } else {
        std.debug.print("MISSING\n", .{});
    }
}

pub fn main(process_init: std.process.Init) !void {
    host_io.init(process_init);
    var gpa = std.heap.DebugAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try host_io.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.skip();
    const path = args.next() orelse return error.MissingArg;

    const file = try std.Io.Dir.cwd().openFile(host_io.io, path, .{});
    defer {
        var f = file;
        f.close(host_io.io);
    }
    const stat = try file.stat(host_io.io);
    const mmap_data = try std.posix.mmap(
        null,
        stat.size,
        .{ .READ = true },
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    defer std.posix.munmap(mmap_data);

    var gf = try gguf.parseWithOptions(mmap_data, allocator, .{ .log_summary = false });
    defer gf.deinit();

    const keys = [_][]const u8{
        "general.architecture",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
        "tokenizer.chat_template",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.merges",
    };
    for (keys) |key| printKey(&gf, key);
}
