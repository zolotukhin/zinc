const std = @import("std");
const host_io = @import("host_io");
const gguf = @import("gguf");

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
