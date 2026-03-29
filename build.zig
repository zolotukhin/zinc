const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

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

    // Platform-specific GPU backend
    if (is_macos) {
        // Metal backend: compile ObjC shim, link Apple frameworks
        exe_mod.addCSourceFile(.{
            .file = b.path("src/metal/shim.m"),
            .flags = &.{ "-fobjc-arc", "-fmodules" },
        });
        exe_mod.addIncludePath(b.path("src/metal"));
        exe_mod.linkFramework("Metal", .{});
        exe_mod.linkFramework("Foundation", .{});
    } else {
        // Vulkan backend (Linux)
        exe_mod.linkSystemLibrary("vulkan", .{});
    }

    const exe = b.addExecutable(.{
        .name = "zinc",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

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
        test_mod.linkSystemLibrary("vulkan", .{});
    }

    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_unit_tests.step);
}
