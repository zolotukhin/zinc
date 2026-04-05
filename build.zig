const std = @import("std");

fn configureVulkanModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    module: *std.Build.Module,
) void {
    switch (target.result.os.tag) {
        .macos => {
            module.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
            module.addSystemIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
            module.linkSystemLibrary("vulkan", .{});
        },
        .windows => {
            const vulkan_sdk = b.graph.env_map.get("VULKAN_SDK") orelse
                b.graph.env_map.get("VK_SDK_PATH") orelse
                @panic("Windows builds require the LunarG Vulkan SDK. Install it and restart your shell so VULKAN_SDK is available.");
            const lib_dir = if (target.result.cpu.arch == .x86) "Lib32" else "Lib";

            module.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ vulkan_sdk, "Include" }) });
            module.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ vulkan_sdk, lib_dir }) });
            module.linkSystemLibrary("vulkan-1", .{});
        },
        else => {
            module.linkSystemLibrary("vulkan", .{});
        },
    }
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    var optimize = b.standardOptimizeOption(.{});
    if (b.option(bool, "release", "Deprecated compatibility flag; prefer -Doptimize")) |release| {
        optimize = if (release) .ReleaseFast else .Debug;
    }
    const full_tests = b.option(bool, "full-tests", "Require integration smoke tests and fail when their environment is missing") orelse false;
    const install_hot_bench = b.option(bool, "install-hot-bench", "Install the zinc-hot-bench binary as part of the default install step") orelse false;

    const is_linux = target.result.os.tag == .linux;
    const is_macos = target.result.os.tag == .macos;

    // --- Shader compilation: GLSL .comp → SPIR-V .spv ---
    // Only compiled when glslc is available (Linux build node).
    // On macOS, shaders are skipped (build-only, no GPU inference).
    const shader_dir = "src/shaders";
    const shader_sources = .{
        "dmmv_q4k",
        "dmmv_q8_0",
        "dmmv_q5k",
        "dmmv_q6k",
        "dmmv_f16",
        "dmmv_f32",
        "rms_norm_mul",
        "swiglu",
        "geglu",
        "sigmoid_mul",
        "rope_fused",
        "softmax_topk",
        "flash_attn",
        "coop_matmul",
        "deinterleave",
        "vadd",
        "scale_accumulate",
        "scale_acc_sigmoid",
        "sigmoid_scale_acc",
        "argmax",
        "embed_dequant_q4k",
        "ssm_conv1d",
        "ssm_delta_net",
        "ssm_gated_norm",
        "tq_quantize_keys",
        "tq_quantize_values",
        "tq_attention_scores",
        "tq_decompress_values",
        "dmmv_q4k_moe",
        "dmmv_q5k_moe",
        "dmmv_q6k_moe",
        "moe_weighted_acc",
    };

    const compile_shaders = b.option(bool, "shaders", "Compile GLSL shaders to SPIR-V (requires glslc)") orelse is_linux;

    if (compile_shaders) {
        inline for (shader_sources) |name| {
            const comp_file = shader_dir ++ "/" ++ name ++ ".comp";
            const spv_file = name ++ ".spv";

            const compile_cmd = b.addSystemCommand(&.{
                "glslc",
                "--target-env=vulkan1.3",
                "-O",
                "-o",
            });
            const spv_output = compile_cmd.addOutputFileArg(spv_file);
            compile_cmd.addFileArg(b.path(comp_file));

            b.getInstallStep().dependOn(&b.addInstallFile(spv_output, "share/zinc/shaders/" ++ spv_file).step);
        }
    }

    // --- Main executable ---
    const exe_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    if (is_macos) {
        exe_mod.addCSourceFile(.{
            .file = b.path("src/metal/shim.m"),
            .flags = &.{ "-fobjc-arc", "-fmodules" },
        });
        exe_mod.addIncludePath(b.path("src/metal"));
        exe_mod.linkFramework("Metal", .{});
        exe_mod.linkFramework("Foundation", .{});
    } else {
        configureVulkanModule(b, target, exe_mod);
    }

    const exe = b.addExecutable(.{
        .name = "zinc",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    const hot_bench_mod = b.createModule(.{
        .root_source_file = b.path("src/bench_hot_decode.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    configureVulkanModule(b, target, hot_bench_mod);

    const hot_bench = b.addExecutable(.{
        .name = "zinc-hot-bench",
        .root_module = hot_bench_mod,
    });

    if (install_hot_bench) {
        b.installArtifact(hot_bench);
    }

    // --- Documentation ---
    const docs_step = b.step("docs", "Generate Zig documentation");
    const docs_install = b.addInstallDirectory(.{
        .source_dir = exe.getEmittedDocs(),
        .install_dir = .prefix,
        .install_subdir = "docs",
    });
    docs_step.dependOn(&docs_install.step);

    // --- Run step ---
    const run_cmd = b.addRunArtifact(exe);
    run_cmd.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
    const run_step = b.step("run", "Run ZINC inference engine");
    run_step.dependOn(&run_cmd.step);

    const run_hot_bench = b.addRunArtifact(hot_bench);
    run_hot_bench.step.dependOn(b.getInstallStep());
    if (b.args) |args| {
        run_hot_bench.addArgs(args);
    }
    const hot_bench_step = b.step("hot-bench", "Run hot decode microbenchmarks");
    hot_bench_step.dependOn(&run_hot_bench.step);

    if (is_macos) {
        const bench_mod = b.createModule(.{
            .root_source_file = b.path("benchmarks/metal_inference.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        const bench_support_mod = b.createModule(.{
            .root_source_file = b.path("src/bench_support.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        bench_support_mod.addIncludePath(b.path("src/metal"));
        bench_mod.addImport("zinc_bench_support", bench_support_mod);
        bench_mod.addCSourceFile(.{
            .file = b.path("src/metal/shim.m"),
            .flags = &.{ "-fobjc-arc", "-fmodules" },
        });
        bench_mod.addIncludePath(b.path("src/metal"));
        bench_mod.linkFramework("Metal", .{});
        bench_mod.linkFramework("Foundation", .{});

        const bench_exe = b.addExecutable(.{
            .name = "zinc-bench-metal",
            .root_module = bench_mod,
        });
        b.installArtifact(bench_exe);

        const bench_run = b.addRunArtifact(bench_exe);
        if (b.args) |args| {
            bench_run.addArgs(args);
        }

        const bench_metal_step = b.step("bench-metal", "Run the Metal inference benchmark (ReleaseFast)");
        bench_metal_step.dependOn(&bench_run.step);

        const bench_step = b.step("bench", "Run benchmarks");
        bench_step.dependOn(&bench_run.step);

        const bench_shapes_mod = b.createModule(.{
            .root_source_file = b.path("benchmarks/metal_q8_shapes.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        bench_shapes_mod.addImport("zinc_bench_support", bench_support_mod);
        bench_shapes_mod.addCSourceFile(.{
            .file = b.path("src/metal/shim.m"),
            .flags = &.{ "-fobjc-arc", "-fmodules" },
        });
        bench_shapes_mod.addIncludePath(b.path("src/metal"));
        bench_shapes_mod.linkFramework("Metal", .{});
        bench_shapes_mod.linkFramework("Foundation", .{});

        const bench_shapes_exe = b.addExecutable(.{
            .name = "zinc-bench-metal-shapes",
            .root_module = bench_shapes_mod,
        });
        b.installArtifact(bench_shapes_exe);

        const bench_shapes_run = b.addRunArtifact(bench_shapes_exe);
        if (b.args) |args| {
            bench_shapes_run.addArgs(args);
        }

        const bench_metal_shapes_step = b.step("bench-metal-shapes", "Run exact-shape Metal q8 hot benchmarks (ReleaseFast)");
        bench_metal_shapes_step.dependOn(&bench_shapes_run.step);
    }

    // --- Unit tests ---
    const test_mod = b.createModule(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    if (is_macos) {
        test_mod.addCSourceFile(.{
            .file = b.path("src/metal/shim.m"),
            .flags = &.{ "-fobjc-arc", "-fmodules" },
        });
        test_mod.addIncludePath(b.path("src/metal"));
        test_mod.linkFramework("Metal", .{});
        test_mod.linkFramework("Foundation", .{});
    } else {
        configureVulkanModule(b, target, test_mod);
    }

    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    const run_bun_tests = b.addSystemCommand(&.{ "bun", "test" });
    run_bun_tests.setCwd(b.path("."));
    run_bun_tests.setEnvironmentVariable("ZINC_REQUIRE_FULL_TESTS", if (full_tests) "1" else "0");

    const print_summary = b.addSystemCommand(&.{ "bun", "tools/print_test_summary.ts" });
    print_summary.setCwd(b.path("."));
    print_summary.setEnvironmentVariable("ZINC_REQUIRE_FULL_TESTS", if (full_tests) "1" else "0");
    print_summary.step.dependOn(&run_unit_tests.step);
    print_summary.step.dependOn(&run_bun_tests.step);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&print_summary.step);
}
