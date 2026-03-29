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
    // Default to ReleaseFast for inference performance (override with -Drelease=false for debug)
    const release = b.option(bool, "release", "Build optimized binary (default: true)") orelse true;
    const optimize: std.builtin.OptimizeMode = if (release) .ReleaseFast else .Debug;

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

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
