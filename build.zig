const std = @import("std");

const Backend = enum {
    auto,
    vulkan,
    metal,
    cuda,
    rocm,
    zinc_rt,
};

// Preserve per-checkout CRT overrides with the 0.17 artifact API.
fn configureLibc(b: *std.Build, artifact: *std.Build.Step.Compile) void {
    if (std.Io.Dir.cwd().access(b.graph.io, ".build-support/libc.conf", .{})) |_| {
        artifact.setLibCFile(b.path(".build-support/libc.conf"));
    } else |_| {}
}

fn addExecutable(b: *std.Build, options: std.Build.ExecutableOptions) *std.Build.Step.Compile {
    const artifact = b.addExecutable(options);
    configureLibc(b, artifact);
    return artifact;
}

fn addTest(b: *std.Build, options: std.Build.TestOptions) *std.Build.Step.Compile {
    const artifact = b.addTest(options);
    configureLibc(b, artifact);
    return artifact;
}

fn createModule(b: *std.Build, options: std.Build.Module.CreateOptions) *std.Build.Module {
    const module = b.createModule(options);
    module.addImport("host_io", b.modules.get("host_io").?);
    return module;
}

fn addMetalBindings(b: *std.Build, target: std.Build.ResolvedTarget, module: *std.Build.Module) void {
    // An explicit minimum macOS target is considered cross-compilation even
    // on ARM64 macOS. Supply the chosen SDK's framework/library paths rather
    // than relying on native SDK discovery (which skips those paths then).
    if (b.graph.environ_map.get("SDKROOT")) |sdk| {
        if (sdk.len > 0) {
            module.addFrameworkPath(.{ .cwd_relative = b.pathJoin(&.{ sdk, "System", "Library", "Frameworks" }) });
            module.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ sdk, "usr", "lib" }) });
            module.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sdk, "usr", "include" }) });
        }
    }
    const translated = b.addTranslateC(.{
        .root_source_file = b.path("src/metal/shim.h"),
        .target = target,
        .optimize = module.optimize.?,
    });
    module.addImport("metal_c", translated.createModule());
}

fn addCudaBindings(b: *std.Build, target: std.Build.ResolvedTarget, module: *std.Build.Module) void {
    const translated = b.addTranslateC(.{
        .root_source_file = b.path("src/cuda/cuda_shim.h"),
        .target = target,
        .optimize = module.optimize.?,
    });
    module.addImport("cuda_c", translated.createModule());
}

fn configureVulkanModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    module: *std.Build.Module,
) void {
    const translated = b.addTranslateC(.{
        .root_source_file = b.path("src/vulkan/headers.h"),
        .target = target,
        .optimize = module.optimize.?,
    });
    if (target.result.os.tag == .macos) {
        translated.addSystemIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
    } else if (target.result.os.tag == .windows) {
        const sdk = b.graph.environ_map.get("VULKAN_SDK") orelse
            b.graph.environ_map.get("VK_SDK_PATH") orelse @panic("Vulkan SDK is required");
        translated.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ sdk, "Include" }) });
    }
    module.addImport("vulkan_c", translated.createModule());
    switch (target.result.os.tag) {
        .macos => {
            module.addLibraryPath(.{ .cwd_relative = "/opt/homebrew/lib" });
            module.addSystemIncludePath(.{ .cwd_relative = "/opt/homebrew/include" });
            module.linkSystemLibrary("vulkan", .{});
        },
        .windows => {
            const vulkan_sdk = b.graph.environ_map.get("VULKAN_SDK") orelse
                b.graph.environ_map.get("VK_SDK_PATH") orelse
                @panic("Windows builds require the LunarG Vulkan SDK. Install it and restart your shell so VULKAN_SDK is available.");
            const lib_dir = if (target.result.cpu.arch == .x86) "Lib32" else "Lib";

            module.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ vulkan_sdk, "Include" }) });
            module.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ vulkan_sdk, lib_dir }) });
            module.linkSystemLibrary("vulkan-1", .{});
        },
        else => {
            // An explicit glibc floor disables Zig's native library search.
            // Release containers provide their own loader library directory.
            if (b.graph.environ_map.get("ZINC_VULKAN_LIBRARY_PATH")) |library_path| {
                module.addLibraryPath(.{ .cwd_relative = library_path });
            }
            module.linkSystemLibrary("vulkan", .{});
        },
    }
}

fn configureCudaModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    module: *std.Build.Module,
) void {
    addCudaBindings(b, target, module);
    // CUDA backend — Linux/WSL2 + NVIDIA only. On the WSL2 deployment box NVIDIA
    // exposes CUDA, not Vulkan, so ZINC's kernels reach the GPU through a native
    // CUDA path: the C shim (Driver API + NVRTC) in src/cuda/cuda_shim.c plus the
    // Zig wrappers in src/cuda/{device,buffer,pipeline,command}.zig. Mirrors
    // configureVulkanModule. See docs/cuda-backend.md §6.
    const cuda_home = b.graph.environ_map.get("CUDA_HOME") orelse "/usr/local/cuda";
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
    addCudaBindings(b, target, module);
    // ROCm backend — Linux + AMD only. The Zig compute/model layers reuse the
    // mature CUDA-side orchestration through its stable C ABI, while this shim
    // implements that ABI with HIP, hipRTC, and hipBLAS.
    const rocm_home = b.graph.environ_map.get("ROCM_PATH") orelse
        b.graph.environ_map.get("ROCM_HOME") orelse
        "/opt/rocm";
    module.addCSourceFile(.{
        .file = b.path("src/rocm/rocm_shim.c"),
        .flags = &.{ "-std=c11", "-D__HIP_PLATFORM_AMD__=1" },
    });
    module.addIncludePath(b.path("src/cuda"));
    module.addSystemIncludePath(.{ .cwd_relative = b.pathJoin(&.{ rocm_home, "include" }) });
    module.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ rocm_home, "lib" }) });
    // ROCm installs outside the system loader's default directories. Preserve
    // its runtime location so an extracted package needs no shell setup.
    module.addRPath(.{ .cwd_relative = b.pathJoin(&.{ rocm_home, "lib" }) });
    module.linkSystemLibrary("amdhip64", .{});
    module.linkSystemLibrary("hiprtc", .{});
    module.linkSystemLibrary("hipblas", .{});
}

fn resolveBunExe(b: *std.Build) []const u8 {
    if (b.graph.environ_map.get("BUN_EXE")) |bun_exe| return bun_exe;
    if (b.graph.environ_map.get("HOME")) |home| {
        const bun_exe = b.pathJoin(&.{ home, ".bun", "bin", "bun" });
        if (std.Io.Dir.accessAbsolute(b.graph.io, bun_exe, .{})) |_| return bun_exe else |_| {}
    }
    return "bun";
}

fn addBunDirToPath(b: *std.Build, run: *std.Build.Step.Run, bun_exe: []const u8) void {
    if (!std.fs.path.isAbsolute(bun_exe)) return;
    const bun_dir = std.fs.path.dirname(bun_exe) orelse return;
    const old_path = b.graph.environ_map.get("PATH") orelse "";
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
        b.graph.environ_map.get("ZINC_VERSION") orelse
        "dev";
    const build_commit = b.option([]const u8, "commit", "Git commit hash embedded in `zinc --version`") orelse
        b.graph.environ_map.get("ZINC_COMMIT") orelse
        b.graph.environ_map.get("GITHUB_SHA") orelse
        "unknown";
    const full_tests = b.option(bool, "full-tests", "Require integration smoke tests and fail when their environment is missing") orelse false;
    const requested_gpu_tests = b.option(bool, "gpu-tests", "Enable hardware unit tests on a dedicated GPU worker") orelse false;
    const gpu_tests = full_tests or requested_gpu_tests;
    const install_hot_bench = b.option(bool, "install-hot-bench", "Install the zinc-hot-bench binary as part of the default install step") orelse false;

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

    _ = b.addModule("host_io", .{
        .root_source_file = b.path("src/host_io.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });

    const zinc_rt_gguf_mod = createModule(b, .{
        .root_source_file = b.path("src/model/gguf.zig"),
        .target = target,
        .optimize = optimize,
    });
    const zinc_rt_lib_mod = createModule(b, .{
        .root_source_file = b.path("src/zinc_rt/lib.zig"),
        .target = target,
        .optimize = optimize,
    });
    zinc_rt_lib_mod.addImport("gguf", zinc_rt_gguf_mod);
    const forward_zinc_rt_mod = createModule(b, .{
        .root_source_file = b.path("src/compute/forward_zinc_rt.zig"),
        .target = target,
        .optimize = optimize,
    });
    forward_zinc_rt_mod.addImport("gguf", zinc_rt_gguf_mod);
    forward_zinc_rt_mod.addImport("zinc_rt", zinc_rt_lib_mod);

    // --- Shader compilation: GLSL .comp → SPIR-V .spv ---
    // Only compiled when glslc is available (Linux build node).
    // Metal builds install their MSL sources instead of compiling SPIR-V.
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
        "dmmv_q6k_rows4",
        "dmmv_q4k_rows4",
        "dmmv_q6k_rows4_cols",
        "dmmv_q4k_rows4_cols",
        "dmmv_q5k_rows4_cols",
        "dmmv_q4k_rows4_cols3",
        "dmmv_q4k_rows4_cols4",
        "dmmv_q6k_rows4_cols3",
        "dmmv_q6k_rows4_cols4",
        "dmmv_q5k_rows4_cols3",
        "dmmv_q5k_rows4_cols4",
        "dmmv_q6k_rows8_cols3",
        "dmmv_q6k_rows8_cols4",
        "dmmv_q4k_rows8_cols3",
        "dmmv_q4k_rows8_cols4",
        "dmmv_q4k_rows2_cols",
        "dmmv_q4k_rows8_cols",
        "dmmv_q6k_rows2_cols",
        "dmmv_q6k_rows8_cols",
        "dmmv_q4k_fused_gate_up_swiglu_cols",
        "dmmv_q4k_fused_gate_up_swiglu_cols3",
        "dmmv_q4k_fused_gate_up_swiglu_cols4",
        "dmmv_q4k_q8_1_cols",
        "dmmv_f16",
        "dmmv_f32",
        "rms_norm_mul",
        "rms_norm_mul_wide",
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
        "flash_attn_f16kv",
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
        "ssm_delta_net_cols8_hist",
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
        "dmmv_q4k_moe_fused_gate_up_geglu_cols_top1_q8_1",
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
        "kv_cache_write_f16kv",
        "norm_rope",
        "quantize_q8_1",
        // Batched prefill shaders — ported from the Metal backend so the
        // Vulkan/RDNA side can share the prefillBatched orchestration.
        "rope_batched",
        "flash_attn_batched",
        "flash_attn_batched_qt",
        "flash_attn_batched_gqa",
        "flash_attn_batched_gqa_f16kv",
        "flash_attn_batched_tile",
        "flash_attn_batched_tile_f16kv",
        "flash_attn_q8kv",
        "flash_attn_gqa",
        "flash_attn_gqa_f16kv",
        "flash_attn_gqa_q8kv",
        "flash_attn_batched_tile_q8kv",
        "flash_attn_batched_tile_q8mmq",
        "flash_attn_batched_tile_q8mmq_lds",
        "flash_attn_q8mmq",
        "qk_norm_rope_kv_write_q8kv",
        "qk_norm_rope_kv_write_batched_q8kv",
        "kv_cache_write_q8kv",
        "kv_cache_write_batched_q8kv",
        "kv_cache_write_single_q8kv",
        "flash_attn_batched_qt_f16kv",
        "flash_attn_batched_f16kv",
        "kv_cache_write_batched",
        "kv_cache_write_batched_f16kv",
        "kv_cache_write_single_f16kv",
        "residual_rms_norm",
        "residual_rms_norm_wide",
        "post_norm_residual_rms_norm",
        "gemma_moe_tail_fused",
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
        "rms_norm_scale_dmmv_f32_topk",
        "rms_norm_scale_dmmv_f32_batch",
        "rms_norm_dmmv_q4k_alpha_beta",
        "rms_norm_dmmv_alpha_beta_ksplit",
        "qk_norm_rope_kv_write",
        "qk_norm_rope_kv_write_f16kv",
        "qk_norm_rope_kv_write_batched",
        "qk_norm_rope_kv_write_batched_f16kv",
        "k_norm_rope_kv_write_batched",
        "k_norm_rope_kv_write_batched_f16kv",
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
        "ssm_conv1d_batched_hist",
        "embed_gather",
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
    const release_shaders_step = b.step("release-shaders", "Compile and install SPIR-V without building executables");

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

            const install_shader = b.addInstallFile(spv_output, "share/zinc/shaders/" ++ spv_file);
            b.getInstallStep().dependOn(&install_shader.step);
            release_shaders_step.dependOn(&install_shader.step);
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

    // Release packages verify the complete installed asset set, not merely
    // that one shader exists. ROCm/CUDA kernels are embedded in the binary.
    var runtime_assets: std.ArrayList([]const u8) = .empty;
    if (selected_backend == .vulkan) {
        inline for (shader_sources) |name| {
            runtime_assets.append(b.allocator, b.fmt("share/zinc/shaders/{s}.spv", .{name})) catch @panic("OOM");
        }
    } else if (selected_backend == .metal) {
        var directory = b.root.openDir(b.graph.io, "src/shaders/metal", .{ .iterate = true }) catch @panic("missing Metal shaders");
        defer directory.close(b.graph.io);
        var iterator = directory.iterate();
        while (iterator.next(b.graph.io) catch @panic("cannot enumerate Metal shaders")) |entry| {
            if (entry.kind == .file and std.mem.endsWith(u8, entry.name, ".metal")) {
                runtime_assets.append(b.allocator, b.fmt("share/zinc/shaders/metal/{s}", .{entry.name})) catch @panic("OOM");
            }
        }
    }
    std.mem.sort([]const u8, runtime_assets.items, {}, struct {
        fn lessThan(_: void, a: []const u8, c: []const u8) bool {
            return std.mem.lessThan(u8, a, c);
        }
    }.lessThan);
    const asset_manifest = std.json.Stringify.valueAlloc(b.allocator, .{
        .schema_version = 1,
        .backend = @tagName(selected_backend),
        .files = runtime_assets.items,
    }, .{ .whitespace = .indent_2 }) catch @panic("OOM");
    const generated_assets = b.addWriteFiles();
    b.getInstallStep().dependOn(&b.addInstallFile(generated_assets.add("asset-manifest.json", asset_manifest), "share/zinc/asset-manifest.json").step);

    // --- Main executable ---
    const exe_mod = createModule(b, .{
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
        addMetalBindings(b, target, exe_mod);
        exe_mod.linkFramework("Metal", .{});
        exe_mod.linkFramework("Foundation", .{});
    } else if (selected_backend == .cuda) {
        configureCudaModule(b, target, exe_mod);
    } else if (selected_backend == .rocm) {
        configureRocmModule(b, target, exe_mod);
    } else {
        configureVulkanModule(b, target, exe_mod);
    }

    const exe = addExecutable(b, .{
        .name = "zinc",
        .root_module = exe_mod,
    });

    b.installArtifact(exe);

    if (is_linux) {
        // The C ABI header is portable. A check-only artifact lets Linux CI
        // analyze CUDA orchestration without a CUDA SDK or a GPU context.
        const cuda_check_options = b.addOptions();
        cuda_check_options.addOption([]const u8, "version", build_version);
        cuda_check_options.addOption([]const u8, "commit", build_commit);
        cuda_check_options.addOption([]const u8, "target", target.result.zigTriple(b.allocator) catch @panic("out of memory"));
        cuda_check_options.addOption([]const u8, "optimize", @tagName(optimize));
        cuda_check_options.addOption([]const u8, "backend", "cuda");
        const cuda_check_step = b.step("check-cuda", "Type-check CUDA host code without linking or running a GPU");
        for ([_][]const u8{
            "src/main.zig", "src/cuda/smoke.zig", "src/loadtest_cuda.zig", "src/run_cuda.zig", "src/dbg_cuda.zig",
        }, 0..) |root, i| {
            const cuda_check_mod = createModule(b, .{
                .root_source_file = b.path(root),
                .target = target,
                .optimize = optimize,
                .link_libc = true,
            });
            cuda_check_mod.addOptions("build_options", cuda_check_options);
            addCudaBindings(b, target, cuda_check_mod);
            const cuda_check = addExecutable(b, .{
                .name = b.fmt("zinc-cuda-check-{d}", .{i}),
                .root_module = cuda_check_mod,
            });
            cuda_check_step.dependOn(&cuda_check.step);
        }
    }

    const analyzer_mod = createModule(b, .{
        .root_source_file = b.path("src/zig-struct-analyzer.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    analyzer_mod.addOptions("build_options", build_options);
    addMetalBindings(b, target, analyzer_mod);
    if (!is_macos) {
        configureVulkanModule(b, target, analyzer_mod);
    }
    const analyzer = addExecutable(b, .{
        .name = "zinc-struct-analyzer",
        .root_module = analyzer_mod,
    });
    const analyze_step = b.step("analyze-structs", "Print layouts for the generated API documentation probe");
    analyze_step.dependOn(&b.addRunArtifact(analyzer).step);

    const check_tools_step = b.step("check-tools", "Compile standalone tools without opening a model or GPU");
    const metadata_tools = .{
        .{ "zinc-inspect-qwen-meta", "tools/inspect_qwen_meta.zig" },
        .{ "zinc-inspect-gemma-meta", "tools/inspect_gemma_meta.zig" },
        .{ "zinc-inspect-tokenizer-meta", "tools/inspect_tokenizer_meta.zig" },
        .{ "zinc-inspect-tensor-dims", "tools/inspect_tensor_dims.zig" },
    };
    inline for (metadata_tools) |tool| {
        const tool_mod = createModule(b, .{
            .root_source_file = b.path(tool[1]),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        tool_mod.addImport("gguf", zinc_rt_gguf_mod);
        const tool_exe = addExecutable(b, .{ .name = tool[0], .root_module = tool_mod });
        check_tools_step.dependOn(&tool_exe.step);
    }

    const layouts_mod = createModule(b, .{
        .root_source_file = b.path("src/dump_struct_layouts.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    layouts_mod.addOptions("build_options", build_options);
    addMetalBindings(b, target, layouts_mod);
    if (!is_macos) configureVulkanModule(b, target, layouts_mod);
    const layouts_exe = addExecutable(b, .{ .name = "zinc-struct-layouts", .root_module = layouts_mod });
    check_tools_step.dependOn(&layouts_exe.step);
    b.step("dump-struct-layouts", "Print public loader and forward struct layouts")
        .dependOn(&b.addRunArtifact(layouts_exe).step);

    const hot_bench_mod = createModule(b, .{
        .root_source_file = b.path("src/bench_hot_decode.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    hot_bench_mod.addOptions("build_options", build_options);
    configureVulkanModule(b, target, hot_bench_mod);

    const hot_bench = addExecutable(b, .{
        .name = "zinc-hot-bench",
        .root_module = hot_bench_mod,
    });

    if (install_hot_bench) {
        b.installArtifact(hot_bench);
    }

    // CPU-only tokenizer check (`zig build tokenize-tool`): parses GGUF metadata
    // and prints ZINC's prompt token ids without opening a GPU device, for
    // diffing against the reference tokenizer on any machine.
    const tokenize_tool_mod = createModule(b, .{
        .root_source_file = b.path("src/tokenize_tool.zig"),
        .target = target,
        .optimize = optimize,
    });
    const tokenize_tool = addExecutable(b, .{
        .name = "zinc-tokenize",
        .root_module = tokenize_tool_mod,
    });
    const tokenize_tool_step = b.step("tokenize-tool", "Build the CPU-only GGUF tokenizer check (zig-out/bin/zinc-tokenize)");
    tokenize_tool_step.dependOn(&b.addInstallArtifact(tokenize_tool, .{}).step);

    const tokenize_bench = addExecutable(b, .{
        .name = "zinc-tokenize-bench",
        .root_module = createModule(b, .{
            .root_source_file = b.path("src/bench_tokenize.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });
    b.step("tokenize-bench", "Build the CPU-only prompt tokenization benchmark")
        .dependOn(&b.addInstallArtifact(tokenize_bench, .{}).step);
    check_tools_step.dependOn(&tokenize_tool.step);
    check_tools_step.dependOn(&tokenize_bench.step);
    if (!is_macos) check_tools_step.dependOn(&hot_bench.step);

    // Kernel-level check for the Q5_1 route-packed MoE down projection
    // (`zig build validate-moe-cols`). Runs the shader against a CPU Q5_1
    // dequant reference on synthetic weights, so a wrong kernel fails with the
    // route and row that diverged instead of silently changing model output.
    // Vulkan only: it opens a device and loads the compiled SPIR-V.
    if (selected_backend == .vulkan) {
        const moe_cols_mod = createModule(b, .{
            .root_source_file = b.path("tools/validate_q5_1_moe_cols.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        const vulkan_mod = createModule(b, .{
            .root_source_file = b.path("src/vulkan/mod.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        configureVulkanModule(b, target, vulkan_mod);
        moe_cols_mod.addImport("vulkan", vulkan_mod);
        configureVulkanModule(b, target, moe_cols_mod);
        const moe_cols_exe = addExecutable(b, .{
            .name = "zinc-validate-q5-1-moe-cols",
            .root_module = moe_cols_mod,
        });
        const moe_cols_step = b.step("validate-moe-cols", "Check dmmv_q5_1_moe_cols against a CPU reference");
        moe_cols_step.dependOn(&b.addInstallArtifact(moe_cols_exe, .{}).step);
        check_tools_step.dependOn(&moe_cols_exe.step);
        // The validator loads the installed SPIR-V, so it has to depend on the
        // shader compile/install — otherwise editing the shader and running this
        // step happily validates the previous build's .spv.
        moe_cols_step.dependOn(b.getInstallStep());

        const q6k_cols_mod = createModule(b, .{
            .root_source_file = b.path("tools/validate_q6k_moe_cols.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        q6k_cols_mod.addImport("vulkan", vulkan_mod);
        configureVulkanModule(b, target, q6k_cols_mod);
        const q6k_cols_exe = addExecutable(b, .{
            .name = "zinc-validate-q6k-moe-cols",
            .root_module = q6k_cols_mod,
        });
        check_tools_step.dependOn(&q6k_cols_exe.step);
        const q6k_cols_step = b.step("validate-q6k-moe-cols", "Build the Q6_K MoE projection validator");
        q6k_cols_step.dependOn(&b.addInstallArtifact(q6k_cols_exe, .{}).step);
        q6k_cols_step.dependOn(b.getInstallStep());
    }

    // --- CUDA primitive-layer smoke (Linux/WSL2 + NVIDIA only) ---
    // Builds & runs src/cuda/smoke.zig standalone — independent of the main exe
    // and the gpu/interface.zig dispatch — so the src/cuda/* primitive layer can
    // be validated before forward_cuda exists. `zig build cuda-smoke`. See
    // docs/cuda-backend.md §6.2 and Effort 20/21.
    if (is_linux) {
        const cuda_smoke_mod = createModule(b, .{
            .root_source_file = b.path("src/cuda/smoke.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        configureCudaModule(b, target, cuda_smoke_mod);
        const cuda_smoke_exe = addExecutable(b, .{
            .name = "cuda-smoke",
            .root_module = cuda_smoke_mod,
        });
        const run_cuda_smoke = b.addRunArtifact(cuda_smoke_exe);
        run_cuda_smoke.addPassthruArgs();
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
        const cuda_loadtest_mod = createModule(b, .{
            .root_source_file = b.path("src/loadtest_cuda.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        configureCudaModule(b, target, cuda_loadtest_mod);
        const cuda_loadtest_exe = addExecutable(b, .{
            .name = "cuda-loadtest",
            .root_module = cuda_loadtest_mod,
        });
        const run_cuda_loadtest = b.addRunArtifact(cuda_loadtest_exe);
        run_cuda_loadtest.addPassthruArgs();
        const cuda_loadtest_step = b.step("cuda-loadtest", "Build & run the CUDA model loader load-test (Linux/NVIDIA)");
        cuda_loadtest_step.dependOn(&run_cuda_loadtest.step);

        // --- CUDA greedy-decode driver (Linux/WSL2 + NVIDIA only) ---
        // Builds & runs src/run_cuda.zig — drives src/compute/forward_cuda.zig
        // end to end (loader + forward + argmax) for a single greedy token.
        // `zig build cuda-run -Dbackend=cuda -- [token] [v0|v1|v2] [model.gguf]`.
        // Rooted at src/ (like loadtest) so model/*, cuda/*, and compute/*
        // resolve as one module.
        const cuda_run_mod = createModule(b, .{
            .root_source_file = b.path("src/run_cuda.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        configureCudaModule(b, target, cuda_run_mod);
        const cuda_run_exe = addExecutable(b, .{
            .name = "cuda-run",
            .root_module = cuda_run_mod,
        });
        const run_cuda_run = b.addRunArtifact(cuda_run_exe);
        run_cuda_run.addPassthruArgs();
        const cuda_run_step = b.step("cuda-run", "Build & run the CUDA greedy-decode driver (Linux/NVIDIA)");
        cuda_run_step.dependOn(&run_cuda_run.step);

        // CUDA per-layer debug dump — drives forward_cuda layer-by-layer via the
        // public hooks, printing the residual-stream norm after each layer, to
        // diff against a reference implementation per-layer reference and pinpoint divergence.
        const cuda_dbg_mod = createModule(b, .{
            .root_source_file = b.path("src/dbg_cuda.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        configureCudaModule(b, target, cuda_dbg_mod);
        const cuda_dbg_exe = addExecutable(b, .{ .name = "cuda-dbg", .root_module = cuda_dbg_mod });
        const run_cuda_dbg = b.addRunArtifact(cuda_dbg_exe);
        run_cuda_dbg.addPassthruArgs();
        const cuda_dbg_step = b.step("cuda-dbg", "Build & run the CUDA per-layer debug dump (Linux/NVIDIA)");
        cuda_dbg_step.dependOn(&run_cuda_dbg.step);
    }

    if (selected_backend == .rocm) {
        const rocm_dbg_mod = createModule(b, .{
            .root_source_file = b.path("src/dbg_cuda.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        rocm_dbg_mod.addOptions("build_options", build_options);
        configureRocmModule(b, target, rocm_dbg_mod);
        const rocm_dbg_exe = addExecutable(b, .{ .name = "rocm-dbg", .root_module = rocm_dbg_mod });
        const run_rocm_dbg = b.addRunArtifact(rocm_dbg_exe);
        run_rocm_dbg.addPassthruArgs();
        b.step("rocm-dbg", "Build and run ROCm diagnostics on a dedicated GPU").dependOn(&run_rocm_dbg.step);
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
    run_cmd.addPassthruArgs();
    const run_step = b.step("run", "Run ZINC inference engine");
    run_step.dependOn(&run_cmd.step);

    const run_hot_bench = b.addRunArtifact(hot_bench);
    run_hot_bench.step.dependOn(b.getInstallStep());
    run_hot_bench.addPassthruArgs();
    const hot_bench_step = b.step("hot-bench", "Run hot decode microbenchmarks");
    hot_bench_step.dependOn(&run_hot_bench.step);

    var metal_bench_cli_tests: ?*std.Build.Step.Compile = null;
    if (is_macos) {
        const bench_mod = createModule(b, .{
            .root_source_file = b.path("benchmarks/metal_inference.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        const bench_support_mod = createModule(b, .{
            .root_source_file = b.path("src/bench_support.zig"),
            .target = target,
            .optimize = .ReleaseFast,
            .link_libc = true,
        });
        bench_support_mod.addIncludePath(b.path("src/metal"));
        addMetalBindings(b, target, bench_support_mod);
        bench_mod.addImport("zinc_bench_support", bench_support_mod);
        bench_mod.addCSourceFile(.{
            .file = b.path("src/metal/shim.m"),
            .flags = &.{"-fobjc-arc"},
        });
        bench_mod.addIncludePath(b.path("src/metal"));
        addMetalBindings(b, target, bench_mod);
        bench_mod.linkFramework("Metal", .{});
        bench_mod.linkFramework("Foundation", .{});
        const bench_exe = addExecutable(b, .{
            .name = "zinc-bench-metal",
            .root_module = bench_mod,
        });
        b.installArtifact(bench_exe);
        metal_bench_cli_tests = addTest(b, .{
            .name = "metal-bench-cli-tests",
            .root_module = bench_mod,
            .filters = &.{"Metal benchmark CLI"},
        });

        const bench_run = b.addRunArtifact(bench_exe);
        bench_run.addPassthruArgs();

        const bench_metal_step = b.step("bench-metal", "Run the Metal inference benchmark (ReleaseFast)");
        bench_metal_step.dependOn(&bench_run.step);
        const bench_step = b.step("bench", "Run benchmarks");
        bench_step.dependOn(&bench_run.step);

        const bench_shapes_mod = createModule(b, .{
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
        addMetalBindings(b, target, bench_shapes_mod);
        bench_shapes_mod.linkFramework("Metal", .{});
        bench_shapes_mod.linkFramework("Foundation", .{});

        const bench_shapes_exe = addExecutable(b, .{
            .name = "zinc-bench-metal-shapes",
            .root_module = bench_shapes_mod,
        });
        b.installArtifact(bench_shapes_exe);

        const bench_shapes_run = b.addRunArtifact(bench_shapes_exe);
        bench_shapes_run.addPassthruArgs();

        const bench_metal_shapes_step = b.step("bench-metal-shapes", "Run exact-shape Metal q8 hot benchmarks (ReleaseFast)");
        bench_metal_shapes_step.dependOn(&bench_shapes_run.step);

        const bench_gemm_q4k_mod = createModule(b, .{
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
        addMetalBindings(b, target, bench_gemm_q4k_mod);
        bench_gemm_q4k_mod.linkFramework("Metal", .{});
        bench_gemm_q4k_mod.linkFramework("Foundation", .{});

        const bench_gemm_q4k_exe = addExecutable(b, .{
            .name = "zinc-bench-metal-gemm-q4k",
            .root_module = bench_gemm_q4k_mod,
        });
        b.installArtifact(bench_gemm_q4k_exe);

        const bench_gemm_q4k_run = b.addRunArtifact(bench_gemm_q4k_exe);
        bench_gemm_q4k_run.addPassthruArgs();

        const bench_gemm_q4k_step = b.step("bench-metal-gemm-q4k", "Run gemm_q4k microbenchmark (ReleaseFast)");
        bench_gemm_q4k_step.dependOn(&bench_gemm_q4k_run.step);

        const bench_dmmv_q4k_mod = createModule(b, .{
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
        addMetalBindings(b, target, bench_dmmv_q4k_mod);
        bench_dmmv_q4k_mod.linkFramework("Metal", .{});
        bench_dmmv_q4k_mod.linkFramework("Foundation", .{});

        const bench_dmmv_q4k_exe = addExecutable(b, .{
            .name = "zinc-bench-metal-dmmv-q4k",
            .root_module = bench_dmmv_q4k_mod,
        });
        b.installArtifact(bench_dmmv_q4k_exe);

        const bench_dmmv_q4k_run = b.addRunArtifact(bench_dmmv_q4k_exe);
        bench_dmmv_q4k_run.addPassthruArgs();

        const bench_dmmv_q4k_step = b.step("bench-metal-dmmv-q4k", "Run dmmv_q4k decode microbenchmark (ReleaseFast)");
        bench_dmmv_q4k_step.dependOn(&bench_dmmv_q4k_run.step);
    }

    // --- Unit tests ---
    const test_mod = createModule(b, .{
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
            .flags = if (gpu_tests) &.{"-fobjc-arc"} else &.{ "-fobjc-arc", "-DZINC_CPU_ONLY_TESTS=1" },
        });
        test_mod.addIncludePath(b.path("src/metal"));
        addMetalBindings(b, target, test_mod);
        test_mod.linkFramework("Metal", .{});
        test_mod.linkFramework("Foundation", .{});
    } else if (selected_backend == .cuda) {
        configureCudaModule(b, target, test_mod);
    } else if (selected_backend == .rocm) {
        configureRocmModule(b, target, test_mod);
    } else {
        configureVulkanModule(b, target, test_mod);
    }

    const unit_tests = addTest(b, .{
        .root_module = test_mod,
    });
    const run_unit_tests = b.addRunArtifact(unit_tests);
    run_unit_tests.setEnvironmentVariable("ZINC_RUN_GPU_TESTS", if (gpu_tests) "1" else "0");

    const zinc_rt_unit_tests = addTest(b, .{
        .name = "zinc-rt-ir-smoke",
        .root_module = zinc_rt_lib_mod,
    });
    const run_zinc_rt_unit_tests = b.addRunArtifact(zinc_rt_unit_tests);
    const forward_zinc_rt_tests = addTest(b, .{
        .name = "zinc-rt-forward-tests",
        .root_module = forward_zinc_rt_mod,
    });
    const run_forward_zinc_rt_tests = b.addRunArtifact(forward_zinc_rt_tests);
    const host_io_tests = addTest(b, .{
        .name = "host-io-tests",
        .root_module = b.modules.get("host_io").?,
    });
    const run_host_io_tests = b.addRunArtifact(host_io_tests);
    // The minimal server is only reached by CUDA/ROCm executable paths, so
    // ordinary root-module tests can leave its embedded tests unanalyzed.
    // Keep discovery's pure CPU contract covered on every backend.
    const model_discovery_tests = addTest(b, .{
        .name = "model-discovery-tests",
        .root_module = createModule(b, .{
            .root_source_file = b.path("src/server/model_discovery.zig"),
            .target = target,
            .optimize = optimize,
        }),
    });
    const run_model_discovery_tests = b.addRunArtifact(model_discovery_tests);

    // Compile Metal tests on a workstation without opening a GPU or a model.
    const test_build_step = b.step("test-build", "Compile all Zig tests without executing them");
    test_build_step.dependOn(&unit_tests.step);
    test_build_step.dependOn(&zinc_rt_unit_tests.step);
    test_build_step.dependOn(&forward_zinc_rt_tests.step);
    test_build_step.dependOn(&host_io_tests.step);
    test_build_step.dependOn(&model_discovery_tests.step);
    if (metal_bench_cli_tests) |tests| test_build_step.dependOn(&tests.step);
    // The default test target is CPU-only. Full mode includes model-loading
    // and live API smoke tests and must run on a dedicated GPU worker.
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
    run_bun_tests.setEnvironmentVariable("ZINC_REQUIRE_STRUCT_LAYOUTS", "1");
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
    print_summary.step.dependOn(&run_forward_zinc_rt_tests.step);
    print_summary.step.dependOn(&run_host_io_tests.step);
    print_summary.step.dependOn(&run_model_discovery_tests.step);
    print_summary.step.dependOn(&run_bun_tests.step);
    if (metal_bench_cli_tests) |tests| print_summary.step.dependOn(&b.addRunArtifact(tests).step);

    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&print_summary.step);
}
