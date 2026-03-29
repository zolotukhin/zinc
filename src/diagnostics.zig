const std = @import("std");
const builtin = @import("builtin");

/// Run system diagnostics and output to stdout.
pub fn run(allocator: std.mem.Allocator) !void {
    const stdout = std.io.getStdOut().writer();
    try stdout.print("\n=== ZINC System Diagnostics ===\n", .{});

    // 1. Check OS
    try stdout.print("OS            : ", .{});
    if (builtin.os.tag != .linux) {
        try stdout.print("{s} [WARN: Linux is recommended for production AMD RDNA4]\n", .{@tagName(builtin.os.tag)});
    } else {
        try stdout.print("Linux [OK]\n", .{});
    }

    // Mesa and GECC checks are Linux/AMD specific
    if (builtin.os.tag == .linux) {
        // 2. Check Mesa Version
        try stdout.print("Mesa          : ", .{});
        if (getMesaVersion(allocator)) |ver| {
            if (std.mem.indexOf(u8, ver, "25.0.7") != null) {
                try stdout.print("{s} [OK]\n", .{ver});
            } else {
                try stdout.print("{s} [WARN: 25.0.7 is recommended to avoid regressions]\n", .{ver});
            }
            allocator.free(ver);
        } else |_| {
            try stdout.print("Not found (mesa-vulkan-drivers not installed or dpkg missing?)\n", .{});
        }

        // 3. Check GECC (RAS)
        try stdout.print("GECC (RAS)    : ", .{});
        if (getGeccStatus(allocator)) |status| {
            const trimmed = std.mem.trim(u8, status, " \n\r\t");
            if (std.mem.eql(u8, trimmed, "0")) {
                try stdout.print("Disabled [OK]\n", .{});
            } else {
                try stdout.print("Enabled ({s}) [WARN: Should be disabled (0) for max performance]\n", .{trimmed});
            }
            allocator.free(status);
        } else |_| {
            try stdout.print("Unknown (could not read /sys/module/amdgpu/parameters/ras_enable)\n", .{});
        }
    }

    // 4. Check RADV_PERFTEST
    try stdout.print("RADV_PERFTEST : ", .{});
    if (std.posix.getenv("RADV_PERFTEST")) |val| {
        if (std.mem.indexOf(u8, val, "coop_matrix") != null) {
            try stdout.print("{s} [OK]\n", .{val});
        } else {
            try stdout.print("{s} [WARN: Should include 'coop_matrix' for maximum bandwidth]\n", .{val});
        }
    } else {
        try stdout.print("Not set [WARN: Should be 'coop_matrix' for maximum bandwidth]\n", .{});
    }

    try stdout.print("===============================\n\n", .{});
}

fn getMesaVersion(allocator: std.mem.Allocator) ![]u8 {
    var child = std.process.Child.init(&[_][]const u8{ "dpkg-query", "-W", "-f=${Version}", "mesa-vulkan-drivers" }, allocator);
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Ignore;

    try child.spawn();
    const stdout = try child.stdout.?.readToEndAlloc(allocator, 1024);
    const term = try child.wait();

    if (term != .Exited or term.Exited != 0 or stdout.len == 0) {
        allocator.free(stdout);
        return error.QueryFailed;
    }
    return stdout;
}

fn getGeccStatus(allocator: std.mem.Allocator) ![]u8 {
    const file = std.fs.openFileAbsolute("/sys/module/amdgpu/parameters/ras_enable", .{}) catch return error.NotFound;
    defer file.close();
    return try file.readToEndAlloc(allocator, 64);
}
