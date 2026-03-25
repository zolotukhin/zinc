const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    // Main executable
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    // Homebrew paths (macOS)
    exe_mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    exe_mod.addSystemIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    exe_mod.linkSystemLibrary("vulkan", .{});

    const exe = b.addExecutable(.{
        .name = "zinc",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    // Run step
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    const run_step = b.step("run", "Run ZINC inference server");
    run_step.dependOn(&run_cmd.step);

    // Tests
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_mod.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
    test_mod.addSystemIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    test_mod.linkSystemLibrary("vulkan", .{});

    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });

    const run_unit_tests = b.addRunArtifact(unit_tests);
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);

    // Bun tests (loops/ TypeScript)
    const bun_tests = b.addSystemCommand(&.{ "bun", "test" });
    test_step.dependOn(&bun_tests.step);
}
