const std = @import("std");
const vulkan = @import("vulkan.zig");

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const banner =
        \\ZINC — Zig INferenCe Engine for AMD GPUs
        \\
        \\Usage: zinc [options]
        \\  -m, --model <path>     Path to GGUF model file
        \\  -p, --port <port>      Server port (default: 8080)
        \\  -d, --device <id>      Vulkan device index (default: 0)
        \\  -c, --context <size>   Context length (default: 4096)
        \\  --parallel <n>         Max concurrent requests (default: 4)
        \\
    ;
    try std.fs.File.stdout().writeAll(banner);

    // Enumerate Vulkan devices
    try vulkan.init(allocator);
}

test "basic" {
    // Placeholder
    try std.testing.expect(true);
}
