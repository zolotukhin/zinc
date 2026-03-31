const std = @import("std");

fn dumpStruct(comptime T: type, name: []const u8, writer: anytype) !void {
    const info = @typeInfo(T);
    if (info != .@"struct") return;
    
    try writer.print("### {s}\n\n", .{name});
    try writer.print("- **Total Size**: {d} bytes\n", .{@sizeOf(T)});
    try writer.print("- **Alignment**: {d} bytes\n\n", .{@alignOf(T)});
    
    try writer.print("| Offset | Size | Align | Field | Type |\n", .{});
    try writer.print("|--------|------|-------|-------|------|\n", .{});
    
    inline for (info.@"struct".fields) |f| {
        try writer.print("| {d} | {d} | {d} | `{s}` | `{s}` |\n", .{
            @offsetOf(T, f.name),
            @sizeOf(f.type),
            f.alignment,
            f.name,
            @typeName(f.type),
        });
    }
    try writer.print("\n", .{});
}

const main_mod = @import("main.zig");
const modules = .{
    .{ "loader", @import("model/loader.zig") },
    .{ "forward", @import("compute/forward.zig") },
};

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();

    var stdout_buffer: [4096]u8 = undefined;
    var stdout = std.fs.File.stdout().writerStreaming(&stdout_buffer);

    try stdout.interface.print("# ZINC Struct Memory Layouts\n\n", .{});
    
    inline for (modules) |mod| {
        try stdout.interface.print("## Module: {s}\n\n", .{mod[0]});
        
        const mod_info = @typeInfo(mod[1]);
        if (mod_info == .@"struct") {
            inline for (mod_info.@"struct".decls) |decl| {
                const decl_val = @field(mod[1], decl.name);
                const T = @TypeOf(decl_val);
                if (T == type) {
                    const t_info = @typeInfo(decl_val);
                    if (t_info == .@"struct") {
                        try dumpStruct(decl_val, decl.name, &stdout.interface);
                    }
                }
            }
        }
    }

    try stdout.interface.flush();
}
