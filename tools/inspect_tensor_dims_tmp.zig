const std = @import("std");
const gguf = @import("../src/model/gguf.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.skip();
    const path = args.next() orelse return error.MissingArg;

    const file = try std.fs.cwd().openFile(path, .{});
    defer {
        var f = file;
        f.close();
    }
    const stat = try file.stat();
    const mmap_data = try std.posix.mmap(
        null,
        stat.size,
        std.posix.PROT.READ,
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    defer std.posix.munmap(mmap_data);

    var gf = try gguf.parseWithOptions(mmap_data, allocator, .{ .log_summary = false });
    defer gf.deinit();

    while (args.next()) |tensor_name| {
        if (gf.findTensor(tensor_name)) |t| {
            std.debug.print("{s}: type={s} dims=({d},{d},{d},{d}) elems={d} bytes={d}\n", .{
                tensor_name,
                @tagName(t.type_),
                t.dims[0],
                t.dims[1],
                t.dims[2],
                t.dims[3],
                t.numElements(),
                t.sizeBytes(),
            });
        } else {
            std.debug.print("{s}: MISSING\n", .{tensor_name});
        }
    }
}
