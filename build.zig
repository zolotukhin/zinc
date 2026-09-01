const std = @import("std");

const Backend = enum {
    auto,
    vulkan,
    metal,
    cuda,
    rocm,
    zinc_rt,
};

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

fn configureCudaModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    module: *std.Build.Module,
) void {
    // CUDA backend — Linux/WSL2 + NVIDIA only. On the WSL2 deployment box NVIDIA
    // exposes CUDA, not Vulkan, so ZINC's kernels reach the GPU through a native
    // CUDA path: the C shim (Driver API + NVRTC) in src/cuda/cuda_shim.c plus the
    // Zig wrappers in src/cuda/{device,buffer,pipeline,command}.zig. Mirrors
    // configureVulkanModule. See docs/cuda-backend.md §6.
    _ = target;
    const cuda_home = b.graph.env_map.get("CUDA_HOME") orelse "/usr/local/cuda";
    module.addCSourceFile(.{
        .file = b.path("src/cuda/cuda_shim.c"),
        .flags = &.{"-std=c11"},
    });
    module.addIncludePath(b.path("src/cuda"));
    module.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ cuda_home, "include" }) });
    module.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ cuda_home, "lib64" }) });
    // libcuda.so (the driver stub) lives in the WSL passthrough dir, not lib64.
    module.addLibraryPath(.{ .cwd_relative = "/usr/lib/wsl/lib" });
    module.linkSystemLibrary("cuda", .{}); // CUDA Driver API
    module.linkSystemLibrary("nvrtc", .{}); // runtime kernel compilation
    // Effort 26 cycle 9: cuBLAS for the prefill dense Q4_K GEMM (dequant→fp16 +
    // cublasGemmEx fp16 TC, ~6× the hand-written gemm_q4k_tc). Opt-in via
    // ZINC_BATCHED_CUBLAS at runtime; the handle is created in cuda_init.
    module.linkSystemLibrary("cublas", .{});
    module.linkSystemLibrary("cudart", .{}); // cublas runtime dependency
}

fn configureRocmModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    module: *std.Build.Module,
) void {
    // ROCm backend — Linux + AMD only. The Zig compute/model layers reuse the
    // mature CUDA-side orchestration through its stable C ABI, while this shim
    // implements that ABI with HIP, hipRTC, and hipBLAS.
    _ = target;
    const rocm_home = b.graph.env_map.get("ROCM_PATH") orelse
        b.graph.env_map.get("ROCM_HOME") orelse
        "/opt/rocm";
    module.addCSourceFile(.{
        .file = b.path("src/rocm/rocm_shim.c"),
        .flags = &.{ "-std=c11", "-D__HIP_PLATFORM_AMD__=1" },
    });
    module.addIncludePath(b.path("src/cuda"));
    module.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ rocm_home, "include" }) });
    module.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ rocm_home, "lib" }) });
    module.linkSystemLibrary("amdhip64", .{});
    module.linkSystemLibrary("hiprtc", .{});
    module.linkSystemLibrary("hipblas", .{});
}

fn resolveBunExe(b: *std.Build) []const u8 {
    if (b.graph.env_map.get("BUN_EXE")) |bun_exe| return bun_exe;
    if (std.fs.accessAbsolute("/root/.bun/bin/bun", .{})) |_| return "/root/.bun/bin/bun" else |_| {}
    return "bun";
}

fn addBunDirToPath(b: *std.Build, run: *std.Build.Step.Run, bun_exe: []const u8) void {
    if (!std.fs.path.isAbsolute(bun_exe)) return;
    const bun_dir = std.fs.path.dirname(bun_exe) orelse return;
    const old_path = b.graph.env_map.get("PATH") orelse "";
    const path = if (old_path.len == 0)
        bun_dir
    else
        b.fmt("{s}:{s}", .{ bun_dir, old_path });
    run.setEnvironmentVariable("PATH", path);
}

pub fn build(b: *std.Build) void {
    const requested_backend = b.option(Backend, "backend", "Select inference backend: auto, vulkan, metal, cuda, rocm, zinc_rt") orelse .auto;
    const target = b.standardTargetOptions(.{
        .default_target = if (requested_backend == .zinc_rt)
            .{ .cpu_model = .native }
        else
            .{},
    });
    var optimize = b.standardOptimizeOption(.{});
    if (b.option(bool, "release", "Deprecated compatibility flag; prefer -Doptimize")) |release| {
        optimize = if (release) .ReleaseFast else .Debug;
    }
    const build_version = b.option([]const u8, "version", "Version string embedded in `zinc --version`") orelse
        b.graph.env_map.get("ZINC_VERSION") orelse
        "dev";
    const build_commit = b.option([]const u8, "commit", "Git commit hash embedded in `zinc --version`") orelse
        b.graph.env_map.get("ZINC_COMMIT") orelse
        b.graph.env_map.get("GITHUB_SHA") orelse
        "unknown";
    const full_tests = b.option(bool, "full-tests", "Require integration smoke tests and fail when their environment is missing") orelse false;
    const install_hot_bench = b.option(bool, "install-hot-bench", "Install the zinc-hot-bench binary as part of the default install step") orelse false;

    // Rolling Linux distros can ship CRT objects with sections Zig's bundled
    // LLD does not understand yet. Let local builders override libc paths
    // without baking machine-specific files into the repository.
    if (std.fs.cwd().access(".build-support/libc.conf", .{})) |_| {
        b.libc_file = ".build-support/libc.conf";
    } else |_| {}

    const is_linux = target.result.os.tag == .linux;
    const is_macos = target.result.os.tag == .macos;
    const selected_backend: Backend = switch (requested_backend) {
        .auto => if (is_macos) .metal else .vulkan,
        else => requested_backend,
    };

    if (selected_backend == .metal and !is_macos) {
        @panic("-Dbackend=metal currently requires a macOS target");
    }
    if (selected_backend == .vulkan and !is_linux) {
        @panic("-Dbackend=vulkan currently requires a Linux target");
    }
    if (selected_backend == .cuda and !is_linux) {
        @panic("-Dbackend=cuda currently requires a Linux target (NVIDIA + CUDA toolkit)");
    }
    if (selected_backend == .rocm and !is_linux) {
        @panic("-Dbackend=rocm currently requires a Linux target (AMD + ROCm toolkit)");
    }

    const build_options = b.addOptions();
    build_options.addOption([]const u8, "version", build_version);
    build_options.addOption([]const u8, "commit", build_commit);
    build_options.addOption(
        []const u8,
        "target",
        b.fmt("{s}-{s}-{s}", .{
            @tagName(target.result.cpu.arch),
            @tagName(target.result.os.tag),
            @tagName(target.result.abi),
        }),
    );
    build_options.addOption([]const u8, "optimize", @tagName(optimize));
    build_options.addOption([]const u8, "backend", @tagName(selected_backend));

    const zinc_rt_gguf_mod = b.createModule(.{
        .root_source_file = b.path("src/model/gguf.zig"),
        .target = target,
        .optimize = optimize,
    });
    const zinc_rt_lib_mod = b.createModule(.{
        .root_source_file = b.path("src/zinc_rt/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    zinc_rt_lib_mod.addImport("gguf", zinc_rt_gguf_mod);
    const forward_zinc_rt_mod = b.createModule(.{
        .root_source_file = b.path("src/compute/forward_zinc_rt.zig"),
        .target = target,
        .optimize = optimize,
    });
    forward_zinc_rt_mod.addImport("gguf", zinc_rt_gguf_mod);
    forward_zinc_rt_mod.addImport("zinc_rt", zinc_rt_lib_mod);

    // --- Shader compilation: GLSL .comp → SPIR-V .spv ---
    // Only compiled when glslc is available (Linux build node).
    // On macOS, shaders are skipped (build-only, no GPU inference).
    const shader_dir = "src/shaders";
    const shader_sources = .{
        "dmmv_q4k",
        "dmmv_q8_0",
        "dmmv_q8_0_batch",
        "dmmv_q8_0_kpar_batch",
        "dmmv_q8_0_wide",
        "dmmv_q8_0_wide4",
        "dmmv_q8_0_q8_1",
        "dmmv_q4k_q8_1",
        "dmmv_q8_0_fused_pair",
        "dmmv_q5k",
        "dmmv_q6k",
        "dmmv_q6k_wide",
        "dmmv_f16",
        "dmmv_f32",
        "rms_norm_mul",
        "swiglu",
        "swiglu_oai",
        "geglu",
        "sigmoid_mul",
        "rope_fused",
        "softmax_topk",
        "softmax_topk_v2",
        "softmax_top1",
        "softmax_top1_batch",
        "softmax_topk_batch",
        "router_f32_batch",
        "flash_attn",
        "flash_attn_split_merge",
        "deinterleave",
        "deinterleave_batched",
        "vadd",
        "scale_accumulate",
        "bias_add",
        "scale_in_place",
        "mul_elementwise",
        "per_expert_scale",
        "sigmoid_scale_acc",
        "argmax",
        "ssm_conv1d",
        "ssm_qk_norm",
        "ssm_delta_net",
        "ssm_delta_net_cols8",
        "ssm_delta_net_cols8_normed",
        "ssm_gated_norm",
        "ssm_gated_norm_batched",
        "dmmv_mxfp4",
        "dmmv_q5_0",
        "dmmv_q5_1",
        "dmmv_q5_1_acc",
        "dmmv_q4k_moe",
        "dmmv_q4k_moe_kpar",
        "dmmv_q4k_moe_cols",
        "dmmv_q4k_moe_cols_q8_1",
        "dmmv_q4k_fused_gate_up_moe",
        "dmmv_q4k_fused_gate_up_swiglu_moe",
        "dmmv_q4k_fused_gate_up_swiglu",
        "dmmv_q4k_fused_gate_up_geglu",
        "dmmv_q4k_fused_gate_up_geglu_pair",
        "dmmv_q4k_moe_fused_gate_up_geglu",
        "dmmv_q4k_moe_fused_gate_up_geglu_batch_top1",
        "dmmv_q4k_moe_fused_gate_up_geglu_cols_top1",
        "dmmv_q4k_moe_fused_gate_up_swiglu_cols_top1",
        "dmmv_q4k_moe_fused_gate_up_swiglu_cols_top1_q8_1",
        "dmmv_q8_0_fused_gate_up_swiglu",
        "dmmv_q8_0_fused_gate_up_swiglu_gate",
        "dmmv_q8_0_fused_gate_up_geglu",
        "dmmv_q8_0_fused_gate_up_geglu4",
        "dmmv_q8_0_sigmoid_acc",
        "dmmv_mxfp4_moe",
        "dmmv_q5_1_moe",
        "dmmv_q5_1_moe_cols",
        "dmmv_q5k_moe",
        "dmmv_q5k_moe_kpar",
        "dmmv_q5k_moe_cols",
        "dmmv_q5k_moe_cols_q8_1",
        "dmmv_q6k_moe",
        "dmmv_q6k_moe_cols",
        "moe_weighted_acc",
        "moe_weighted_acc_batch",
        "moe_weighted_acc_scaled_batch",
        "sigmoid_scale_acc_batch",
        "dmmv_q4k_batch",
        "dmmv_q4k_batch_kpar",
        "dmmv_q6k_batch",
        "dmmv_q6k_batch_kpar",
        "kv_cache_write",
        "norm_rope",
        "quantize_q8_1",
        // Batched prefill shaders — ported from the Metal backend so the
        // Vulkan/RDNA side can share the prefillBatched orchestration.
        "rope_batched",
        "flash_attn_batched",
        "kv_cache_write_batched",
        "residual_rms_norm",
        "post_norm_residual_rms_norm",
        "rms_norm_add",
        "rms_norm_add_vec4",
        "dmmv_q4k_wide",
        "dmmv_q4k_moe_batched",
        "dmmv_q4k_moe_fused_down_acc",
        "dmmv_q5k_moe_fused_down_acc",
        "dmmv_q5k_moe_fused_down_acc_q8_1",
        "dmmv_q5_1_moe_fused_down_acc",
        "dmmv_q5_1_moe_fused_down_acc_scaled",
        "dmmv_q5_1_moe_down_acc_scaled_batch_top1",
        "dmmv_q8_0_moe_fused_down_acc_scaled",
        "dmmv_q4k_o_proj_merge",
        "rms_norm_dmmv_f32",
        "rms_norm_scale_dmmv_f32",
        "rms_norm_scale_dmmv_f32_batch",
        "rms_norm_dmmv_q4k_alpha_beta",
        "qk_norm_rope_kv_write",
        "qk_norm_rope_kv_write_batched",
        "k_norm_rope_kv_write_batched",
        // Effort-6 GEMM port: tiled Q4_K dense GEMM plus its routed MoE
        // gather sibling. Both are compile-registered foundations; production
        // prefill still uses route-packed columns until the routed GEMM has a
        // layer replay validator.
        "mul_mm_q4k",
        "mul_mm_id_q4k",
        "mul_mm_q4k_gate_up_swiglu",
        "mul_mm_q4k_gate_up_geglu",
        "mul_mm_q4k_gate_up_geglu_full",
        "mul_mm_q4k_gate_up_geglu_tail8",
        "mul_mm_q6k",
        "mul_mm_q6k_tail8",
        "count_experts",
        // Previously-orphaned shaders: these .comp files were added by their
        // cycles (dmmv_f32_dual_batch + ssm_conv1d_batched in effort-15 cycle 9;
        // mul_mm_q6k_full + mul_mm_q4k_gate_up_swiglu_full in cycles 43/44) and
        // wired into forward.zig/dmmv.zig, but were never added here — so clean
        // builds silently ran fallback kernels and the benchmark measured the
        // wrong code (the effort-15 79.63 tok/s artifact). The shader-install
        // parity guard in loops/optimize_perf.ts now fails loud if any
        // src/shaders/*.comp is missing from this list.
        "dmmv_f32_dual_batch",
        "ssm_conv1d_batched",
        "mul_mm_q6k_full",
        "mul_mm_q6k_full_down_acc",
        "mul_mm_q4k_down_acc",
        "mul_mm_q4k_down_acc_wide",
        "mul_mm_q4k_tail8",
        "mul_mm_q4k_gate_up_swiglu_full",
        // Vulkan port of the Metal MoE route-pack kernel; the .comp was added
        // without registering it here, so the parity guard flagged it.
        "moe_route_pack",
        // Run-1 cycle-13 (Q5_K dense projection) and cycle-23 (DP4a Q6_K
        // dense-down GEMM + per-32-block activation quantizer): each cycle
        // committed its .comp via `git add src/` but the matching
        // shader_sources update was uncommitted — a subsequent rejected
        // cycle's `git checkout -- build.zig` wiped the registration, so
        // the parity guard flagged them on the next baseline. The loop's
        // commit now also stages build.zig to prevent recurrence.
        "mul_mm_q5k",
        "mul_mm_q5k_wide",
        "mul_mm_q8_0",
        "mul_mm_q8_0_full_dp4a",
        "mul_mm_q6k_full_dp4a",
        "mul_mm_q6k_full_dp4a_bm64_n64_acc",
        "mul_mm_q6k_full_dp4a_mmq64_n64_acc",
        "mul_mm_q6k_full_dp4a_q8_1",
        "mul_mm_q6k_full_dp4a_q8_1_bm64_n64",
        "quantize_act_q8",
        "mul_mm_q4k_gate_up_swiglu_full_dp4a",
        "mul_mm_q4k_gate_up_swiglu_full_dp4a_q8",
        "mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_bm64",
        "mul_mm_q4k_gate_up_geglu_n1_dp4a_q8",
        "mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1",
        "mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_bm64",
        "mul_mm_q4k_full_dp4a",
        "mul_mm_q4k_full_dp4a_bm64_n64_acc",
        "mul_mm_q4k_full_dp4a_mmq64_n64_acc",
        "mul_mm_q5k_full_dp4a",
        "quantize_act_q8_1",
        // Effort-15 cycle 16: fuses residual+RMS norm + Q8_1 activation
        // quantize for the Qwen3.6-27B dense FFN prefill DP4a path. Drops the
        // separate quantize_act_q8_1 dispatch + barrier inside dense_ffn_gateup
        // per SSM-fed layer-major segment.
        "residual_rms_norm_quant_q8_1",
        // Effort-15 cycle 11: token-batched SSM fused gated norm. Replaces the
        // per-token pushDescAndDispatch loop in the layer-major SSM segment
        // (Qwen3.6-27B context-medium prefill) with a single (dt_rank,
        // n_tokens, 1) dispatch — ~280 dispatch records dropped per SSM
        // segment.
        "ssm_gated_norm_batch_tok",
        "ssm_gated_norm_batch_tok_fused",
    };

    // SPIR-V is only the Vulkan backend's kernel format. CUDA uses NVRTC `.cu`
    // kernels (and Metal uses MSL), so don't require glslc for those backends.
    const compile_shaders = b.option(bool, "shaders", "Compile GLSL shaders to SPIR-V (requires glslc)") orelse (selected_backend == .vulkan);

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
    if (selected_backend == .metal) {
        const metal_shaders_install = b.addInstallDirectory(.{
            .source_dir = b.path("src/shaders/metal"),
            .install_dir = .prefix,
            .install_subdir = "share/zinc/shaders/metal",
        });
        b.getInstallStep().dependOn(&metal_shaders_install.step);
    }

    // --- Main executable ---
    const exe_mod = b.createModule(.{
        .root_source_file = b.path(if (selected_backend == .zinc_rt) "src/zinc_rt/main.zig" else "src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    exe_mod.addOptions("build_options", build_options);

    if (selected_backend == .zinc_rt) {
        // M0 T-CPU scaffold is pure Zig. GPU tier linking starts when
        // forward_zinc_rt is wired to a concrete direct-submission tier.
        exe_mod.addImport("gguf", zinc_rt_gguf_mod);
        exe_mod.addImport("zinc_rt", zinc_rt_lib_mod);
        exe_mod.addImport("forward_zinc_rt", forward_zinc_rt_mod);
    } else if (is_macos) {
        exe_mod.addCSourceFile(.{
            .file = b.path("src/metal/shim.m"),
            .flags = &.{"-fobjc-arc"},
        });
        exe_mod.addIncludePath(b.path("src/metal"));
        exe_mod.linkFramework("Metal", .{});
        exe_mod.linkFramework("Foundation", .{});
    } else if (selected_backend == .cuda) {
        configureCudaModule(b, target, exe_mod);
    } else if (selected_backend == .rocm) {
        configureRocmModule(b, target, exe_mod);
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
    hot_bench_mod.addOptions("build_options", build_options);
    configureVulkanModule(b, target, hot_bench_mod);

    const hot_bench = b.addExecutable(.{
        .name = "zinc-hot-bench",
        .root_module = hot_bench_mod,
    });

    if (install_hot_bench) {
        b.installArtifact(hot_bench);
    }

    // --- CUDA primitive-layer smoke (Linux/WSL2 + NVIDIA only) ---
    // Builds & runs src/cuda/smoke.zig standalone — independent of the main exe
    // and the gpu/interface.zig dispatch — so the src/cuda/* primitive layer can
    // be validated before forward_cuda exists. `zig build cuda-smoke`. See
    // docs/cuda-backend.md §6.2 and Effort 20/21.
    if (is_linux) {
        const cuda_smoke_mod = b.createModule(.{
            .root_source_file = b.path("src/cuda/smoke.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        configureCudaModule(b, target, cuda_smoke_mod);
        const cuda_smoke_exe = b.addExecutable(.{
            .name = "cuda-smoke",
            .root_module = cuda_smoke_mod,
        });
        const run_cuda_smoke = b.addRunArtifact(cuda_smoke_exe);
        if (b.args) |args| run_cuda_smoke.addArgs(args);
        const cuda_smoke_step = b.step("cuda-smoke", "Build & run the CUDA primitive-layer smoke test (Linux/NVIDIA)");
        cuda_smoke_step.dependOn(&run_cuda_smoke.step);

        // --- CUDA model loader load-test (Linux/WSL2 + NVIDIA only) ---
        // Builds & runs src/cuda/loadtest.zig standalone — exercises
        // src/model/loader_cuda.zig end to end (mmap + GGUF parse + H2D upload
        // of every tensor) independent of forward_cuda. Pass the model path as
        // a trailing arg: `zig build cuda-loadtest -Dbackend=cuda -- model.gguf`.
        // Rooted at src/ (root file src/loadtest_cuda.zig) so the single module
        // can reach both model/* (loader, gguf, config) and cuda/* (device,
        // buffer, c) — mirroring how src/main.zig spans both subtrees. Keeping
        // it one module is what lets the loader's internal `../cuda/*` imports
        // resolve.
        const cuda_loadtest_mod = b.createModule(.{
            .root_source_file = b.path("src/loadtest_cuda.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        configureCudaModule(b, target, cuda_loadtest_mod);
        const cuda_loadtest_exe = b.addExecutable(.{
            .name = "cuda-loadtest",
            .root_module = cuda_loadtest_mod,
        });
        const run_cuda_loadtest = b.addRunArtifact(cuda_loadtest_exe);
        if (b.args) |args| run_cuda_loadtest.addArgs(args);
        const cuda_loadtest_step = b.step("cuda-loadtest", "Build & run the CUDA model loader load-test (Linux/NVIDIA)");
        cuda_loadtest_step.dependOn(&run_cuda_loadtest.step);

        // --- CUDA greedy-decode driver (Linux/WSL2 + NVIDIA only) ---
        // Builds & runs src/run_cuda.zig — drives src/compute/forward_cuda.zig
        // end to end (loader + forward + argmax) for a single greedy token.
        // `zig build cuda-run -Dbackend=cuda -- [token] [v0|v1|v2] [model.gguf]`.
        // Rooted at src/ (like loadtest) so model/*, cuda/*, and compute/*
        // resolve as one module.
        const cuda_run_mod = b.createModule(.{
            .root_source_file = b.path("src/run_cuda.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        configureCudaModule(b, target, cuda_run_mod);
        const cuda_run_exe = b.addExecutable(.{
            .name = "cuda-run",
            .root_module = cuda_run_mod,
        });
        const run_cuda_run = b.addRunArtifact(cuda_run_exe);
        if (b.args) |args| run_cuda_run.addArgs(args);
        const cuda_run_step = b.step("cuda-run", "Build & run the CUDA greedy-decode driver (Linux/NVIDIA)");
        cuda_run_step.dependOn(&run_cuda_run.step);

        // CUDA per-layer debug dump — drives forward_cuda layer-by-layer via the
        // public hooks, printing the residual-stream norm after each layer, to
        // diff against a reference implementation per-layer reference and pinpoint divergence.
        const cuda_dbg_mod = b.createModule(.{
            .root_source_file = b.path("src/dbg_cuda.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        configureCudaModule(b, target, cuda_dbg_mod);
        const cuda_dbg_exe = b.addExecutable(.{ .name = "cuda-dbg", .root_module = cuda_dbg_mod });
        const run_cuda_dbg = b.addRunArtifact(cuda_dbg_exe);
        if (b.args) |args| run_cuda_dbg.addArgs(args);
        const cuda_dbg_step = b.step("cuda-dbg", "Build & run the CUDA per-layer debug dump (Linux/NVIDIA)");
        cuda_dbg_step.dependOn(&run_cuda_dbg.step);
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
            .flags = &.{"-fobjc-arc"},
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
            .flags = &.{"-fobjc-arc"},
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

        const bench_gemm_q4k_mod = b.createModule(.{
            .root_source_file = b.path("benchmarks/metal_gemm_q4k.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        bench_gemm_q4k_mod.addImport("zinc_bench_support", bench_support_mod);
        bench_gemm_q4k_mod.addCSourceFile(.{
            .file = b.path("src/metal/shim.m"),
            .flags = &.{"-fobjc-arc"},
        });
        bench_gemm_q4k_mod.addIncludePath(b.path("src/metal"));
        bench_gemm_q4k_mod.linkFramework("Metal", .{});
        bench_gemm_q4k_mod.linkFramework("Foundation", .{});

        const bench_gemm_q4k_exe = b.addExecutable(.{
            .name = "zinc-bench-metal-gemm-q4k",
            .root_module = bench_gemm_q4k_mod,
        });
        b.installArtifact(bench_gemm_q4k_exe);

        const bench_gemm_q4k_run = b.addRunArtifact(bench_gemm_q4k_exe);
        if (b.args) |args| {
            bench_gemm_q4k_run.addArgs(args);
        }

        const bench_gemm_q4k_step = b.step("bench-metal-gemm-q4k", "Run gemm_q4k microbenchmark (ReleaseFast)");
        bench_gemm_q4k_step.dependOn(&bench_gemm_q4k_run.step);

        const bench_dmmv_q4k_mod = b.createModule(.{
            .root_source_file = b.path("benchmarks/metal_dmmv_q4k.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        bench_dmmv_q4k_mod.addImport("zinc_bench_support", bench_support_mod);
        bench_dmmv_q4k_mod.addCSourceFile(.{
            .file = b.path("src/metal/shim.m"),
            .flags = &.{"-fobjc-arc"},
        });
        bench_dmmv_q4k_mod.addIncludePath(b.path("src/metal"));
        bench_dmmv_q4k_mod.linkFramework("Metal", .{});
        bench_dmmv_q4k_mod.linkFramework("Foundation", .{});

        const bench_dmmv_q4k_exe = b.addExecutable(.{
            .name = "zinc-bench-metal-dmmv-q4k",
            .root_module = bench_dmmv_q4k_mod,
        });
        b.installArtifact(bench_dmmv_q4k_exe);

        const bench_dmmv_q4k_run = b.addRunArtifact(bench_dmmv_q4k_exe);
        if (b.args) |args| {
            bench_dmmv_q4k_run.addArgs(args);
        }

        const bench_dmmv_q4k_step = b.step("bench-metal-dmmv-q4k", "Run dmmv_q4k decode microbenchmark (ReleaseFast)");
        bench_dmmv_q4k_step.dependOn(&bench_dmmv_q4k_run.step);
    }

    // --- Unit tests ---
    const test_mod = b.createModule(.{
        .root_source_file = b.path(if (selected_backend == .zinc_rt) "src/zinc_rt/test_root.zig" else "src/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    test_mod.addOptions("build_options", build_options);
    if (selected_backend == .zinc_rt) {
        // Pure Zig T-CPU tests; no platform GPU runtime linked.
        test_mod.addImport("gguf", zinc_rt_gguf_mod);
        test_mod.addImport("zinc_rt", zinc_rt_lib_mod);
        test_mod.addImport("forward_zinc_rt", forward_zinc_rt_mod);
    } else if (is_macos) {
        test_mod.addCSourceFile(.{
            .file = b.path("src/metal/shim.m"),
            .flags = &.{"-fobjc-arc"},
        });
        test_mod.addIncludePath(b.path("src/metal"));
        test_mod.linkFramework("Metal", .{});
        test_mod.linkFramework("Foundation", .{});
    } else if (selected_backend == .cuda) {
        configureCudaModule(b, target, test_mod);
    } else if (selected_backend == .rocm) {
        configureRocmModule(b, target, test_mod);
    } else {
        configureVulkanModule(b, target, test_mod);
    }

    const unit_tests = b.addTest(.{
        .root_module = test_mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);

    const zinc_rt_test_mod = b.createModule(.{
        .root_source_file = b.path("src/zinc_rt/test_root.zig"),
        .target = target,
        .optimize = optimize,
    });
    zinc_rt_test_mod.addImport("gguf", zinc_rt_gguf_mod);
    zinc_rt_test_mod.addImport("zinc_rt", zinc_rt_lib_mod);
    zinc_rt_test_mod.addImport("forward_zinc_rt", forward_zinc_rt_mod);
    const zinc_rt_unit_tests = b.addTest(.{
        .name = "zinc-rt-ir-smoke",
        .root_module = zinc_rt_test_mod,
    });
    const run_zinc_rt_unit_tests = b.addRunArtifact(zinc_rt_unit_tests);
    // In partial mode (`full_tests = false`) restrict `bun test` to the
    // fast unit-test files. The slow `tests/test_qwen_smoke.test.ts`
    // file launches multiple managed servers and loads four GGUFs
    // (Qwen 3.5 9B, Qwen 3.6 35B, Qwen 3.8 27B, plus the API model),
    // which together run several minutes on this Mac
    // Studio — well past the harness's 120s `runCommand` timeout for
    // `zig build test`, so even though the smoke tests themselves pass
    // the parent spawn was being killed and `testExitCode` came back
    // `-1`, causing the harness to revert otherwise-good changes.
    // Full mode still runs every test file so the user's local
    // `zig build test --full-tests` (or whatever flag wires
    // `full_tests = true`) is unchanged.
    const bun_exe = resolveBunExe(b);
    const run_bun_tests = if (full_tests)
        b.addSystemCommand(&.{ bun_exe, "test" })
    else
        b.addSystemCommand(&.{
            bun_exe,     "test",
            "loops/",    "tools/",
            "site/src/", "tests/chat_ui_markdown.test.ts",
        });
    run_bun_tests.setCwd(b.path("."));
    addBunDirToPath(b, run_bun_tests, bun_exe);
    run_bun_tests.setEnvironmentVariable("ZINC_REQUIRE_FULL_TESTS", if (full_tests) "1" else "0");
    // Keep implement_metal.ts prompt tests deterministic under the outer
    // optimization harness. Those tests construct Qwen/Gemma states directly,
    // so parent-process model and metric env must not decide their branch.
    run_bun_tests.setEnvironmentVariable("ZINC_MODEL_ID", "qwen36-35b-a3b-q4k-xl");
    run_bun_tests.setEnvironmentVariable("ZINC_MODEL", "");
    run_bun_tests.setEnvironmentVariable("ZINC_METRIC_MODE", "decode");
    // Pin ZINC_TARGET_TOK_PER_SEC to the implement_metal.ts default (50)
    // so the harness's parent-process value (e.g. 26) does not leak into
    // the buildPrompt unit tests in loops/implement_metal.test.ts, which
    // rely on tokPerSec=36 falling under target to render the "below
    // target" diagnosis (samples list + variance warning). Without this,
    // two tests ("includes benchmark samples in diagnosis", "warns when
    // benchmark samples are too noisy for direction") fail with a
    // "TARGET REACHED" prompt instead.
    run_bun_tests.setEnvironmentVariable("ZINC_TARGET_TOK_PER_SEC", "50");

    const print_summary = b.addSystemCommand(&.{ bun_exe, "tools/print_test_summary.ts" });
    print_summary.setCwd(b.path("."));
    addBunDirToPath(b, print_summary, bun_exe);
    print_summary.setEnvironmentVariable("ZINC_REQUIRE_FULL_TESTS", if (full_tests) "1" else "0");
    print_summary.step.dependOn(&run_unit_tests.step);
    print_summary.step.dependOn(&run_zinc_rt_unit_tests.step);
    print_summary.step.dependOn(&run_bun_tests.step);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&print_summary.step);
}
