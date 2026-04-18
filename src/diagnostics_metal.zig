//! Apple Silicon diagnostics and managed-model fit reporting for Metal.
//!
//! The Metal diagnostics path inspects the default device, reports unified
//! memory and feature support, and optionally projects how a selected GGUF or
//! managed model fits once runtime allocations and KV reservation are included.
//! @section Hardware Detection
const std = @import("std");
const builtin = @import("builtin");
const metal_c = @import("metal/c.zig").shim;
const metal_device = @import("metal/device.zig");
const metal_pipeline = @import("metal/pipeline.zig");
const forward_metal = @import("compute/forward_metal.zig");
const memory_plan = @import("gpu/memory_plan.zig");
const config_mod = @import("model/config.zig");
const loader_metal = @import("model/loader_metal.zig");

const MetalDevice = metal_device.MetalDevice;
const ModelConfig = config_mod.ModelConfig;
const ModelInspection = loader_metal.ModelInspection;

/// Configuration for a Metal diagnostics run.
pub const Options = struct {
    /// GPU device index (ignored on Metal; always uses the system default).
    device_index: u32 = 0,
    /// Path to a GGUF model file for inspection, or null to skip.
    model_path: ?[]const u8 = null,
    /// Requested runtime context ceiling from CLI/server configuration.
    requested_context_length: ?u32 = null,
    /// Managed model catalog entry, if selected via `--model-id`.
    managed_model: ?ManagedModelInfo = null,
    /// Directory containing Metal shader source files.
    shader_dir: []const u8 = "src/shaders/metal",
};

/// Catalog metadata for a managed (downloadable) model.
pub const ManagedModelInfo = struct {
    /// Unique model identifier used in the catalog.
    id: []const u8,
    /// Human-readable model name shown in UI.
    display_name: []const u8,
    /// GGUF filename within the local cache.
    file_name: []const u8,
    /// On-disk size in bytes.
    size_bytes: u64,
    /// Minimum VRAM required for inference.
    required_vram_bytes: u64,
    /// Short status string (e.g. "installed", "available").
    status_label: []const u8,
};

const Styles = struct {
    enabled: bool,

    fn detect(stdout_file: std.fs.File) Styles {
        const no_color = std.posix.getenv("NO_COLOR") != null;
        const force_color =
            isTruthy(std.posix.getenv("FORCE_COLOR")) or
            isTruthy(std.posix.getenv("CLICOLOR_FORCE"));
        return .{
            .enabled = shouldUseColor(stdout_file.isTty(), std.posix.getenv("TERM"), no_color, force_color),
        };
    }

    fn statusCode(self: Styles, status: CheckStatus) []const u8 {
        _ = self;
        return switch (status) {
            .ok => "1;32",
            .warn => "1;33",
            .fail => "1;31",
            .skip => "1;34",
        };
    }
};

const CheckStatus = enum {
    ok,
    warn,
    fail,
    skip,

    fn label(self: CheckStatus) []const u8 {
        return switch (self) {
            .ok => "OK",
            .warn => "WARN",
            .fail => "FAIL",
            .skip => "SKIP",
        };
    }
};

const Summary = struct {
    ok: u32 = 0,
    warn: u32 = 0,
    fail: u32 = 0,
    skip: u32 = 0,

    fn record(self: *Summary, status: CheckStatus) void {
        switch (status) {
            .ok => self.ok += 1,
            .warn => self.warn += 1,
            .fail => self.fail += 1,
            .skip => self.skip += 1,
        }
    }
};

const ShaderSourceCheck = struct {
    found: usize = 0,
    total: usize = required_shader_files.len,
    first_missing: ?[]const u8 = null,
};

const PipelineCompileCheck = struct {
    thread_execution_width: u32,
    max_threads_per_threadgroup: u32,
    static_threadgroup_memory_length: u32,
};

/// Estimated unified-memory breakdown for running a model on Apple Silicon.
pub const UnifiedFitEstimate = struct {
    /// Tensor weight payload bytes mapped into unified memory.
    weights_bytes: u64,
    /// Activation, KV-cache, and scratch buffer bytes in unified memory.
    runtime_unified_bytes: u64,
    /// Sum of weights + runtime bytes in unified memory.
    total_unified_bytes: u64,
    /// Recommended working-set budget reported by the Metal driver.
    recommended_working_set_bytes: u64,
    /// Total physical memory on the device.
    total_memory_bytes: u64,
    /// KV cache bytes included in the runtime estimate.
    kv_cache_bytes: u64,
    /// Current runtime context cap used for KV cache sizing.
    max_ctx: u32,
    /// Maximum context the current UMA budget could sustain if runtime caps were lifted.
    budget_max_ctx: u32,

    fn headroomBytes(self: UnifiedFitEstimate) i128 {
        return @as(i128, self.recommended_working_set_bytes) - @as(i128, self.total_unified_bytes);
    }

    fn fitStatus(self: UnifiedFitEstimate) CheckStatus {
        return fitStatusForUnifiedBytes(
            self.total_unified_bytes,
            self.recommended_working_set_bytes,
            self.total_memory_bytes,
        );
    }
};

const required_shader_files = [_][]const u8{
    "dmmv_q4k.metal",
    "dmmv_q4k_k2048.metal",
    "dmmv_q4k_lmhead.metal",
    "dmmv_q4k_lmhead_1024.metal",
    "dmmv_q5k.metal",
    "dmmv_q5k_moe.metal",
    "dmmv_q6k.metal",
    "dmmv_q6k_moe.metal",
    "dmmv_q8_0.metal",
    "dmmv_f16.metal",
    "dmmv_f32.metal",
    "dmmv_q4k_moe.metal",
    "dmmv_q4k_moe_k2048.metal",
    "dmmv_q4k_moe_k2048_1024.metal",
    "deinterleave.metal",
    "flash_attn.metal",
    "kv_cache_write.metal",
    "rope_fused.metal",
    "sigmoid_mul.metal",
    "swiglu.metal",
    "swiglu_batched.metal",
    "scale_accumulate.metal",
    "rms_norm_mul.metal",
    "moe_accumulate.metal",
    "moe_accumulate_batched.metal",
    "softmax_topk.metal",
    "sigmoid_scale_acc.metal",
    "moe_weighted_acc.metal",
    "ssm_conv1d.metal",
    "ssm_delta_net.metal",
    "ssm_gated_norm.metal",
};

/// Run Metal system diagnostics and output a readable preflight report to stdout.
pub fn run(opts: Options, allocator: std.mem.Allocator) !void {
    const stdout_file = std.fs.File.stdout();
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = stdout_file.writerStreaming(&stdout_buffer);
    const stdout = &stdout_writer.interface;
    const styles = Styles.detect(stdout_file);
    var summary = Summary{};

    try stdout.print("\n=== ZINC System Diagnostics ===\n", .{});
    try stdout.flush();

    const step1_start = std.time.nanoTimestamp();
    try printStepHeader(stdout, styles, 1, 4, "Host Environment");
    try printStatusLine(stdout, styles, &summary, if (builtin.os.tag == .macos) .ok else .fail, "OS", "{s}", .{@tagName(builtin.os.tag)});
    try printStatusLine(stdout, styles, &summary, if (builtin.cpu.arch == .aarch64) .ok else .fail, "CPU arch", "{s}", .{@tagName(builtin.cpu.arch)});
    try printStatusLine(stdout, styles, &summary, .ok, "Backend", "Metal", .{});
    try printStepDuration(stdout, styles, step1_start);

    var maybe_device: ?MetalDevice = null;
    defer if (maybe_device) |*device| device.deinit();

    const step2_start = std.time.nanoTimestamp();
    try printStepHeader(stdout, styles, 2, 4, "Metal Device");
    try printCheckingLine(stdout, styles, "Metal device init and capability query");
    if (MetalDevice.init(allocator, opts.device_index)) |device| {
        maybe_device = device;
        if (opts.device_index == 0) {
            try printStatusLine(stdout, styles, &summary, .ok, "Metal init", "Initialized default system device", .{});
        } else {
            try printStatusLine(
                stdout,
                styles,
                &summary,
                .warn,
                "Metal init",
                "Initialized default system device (requested index {d} ignored on Metal)",
                .{opts.device_index},
            );
        }
        try printStatusLine(
            stdout,
            styles,
            &summary,
            if (device.chip == .unknown) .warn else .ok,
            "GPU family",
            "{s}",
            .{@tagName(device.chip)},
        );
        try printStatusLine(
            stdout,
            styles,
            &summary,
            if (device.hasUnifiedMemory()) .ok else .fail,
            "Unified memory",
            "{any}",
            .{device.hasUnifiedMemory()},
        );
        try printStatusLine(
            stdout,
            styles,
            &summary,
            if (device.recommendedMaxWorkingSetSize() > 0) .ok else .warn,
            "Working-set budget",
            "{d:.2} GiB",
            .{bytesToGiB(device.recommendedMaxWorkingSetSize())},
        );
        try printStatusLine(
            stdout,
            styles,
            &summary,
            if (device.maxBufferSize() > 0) .ok else .fail,
            "Max buffer size",
            "{d:.2} GiB",
            .{bytesToGiB(device.maxBufferSize())},
        );
        try printDetailLine(stdout, "  total memory", "{d:.2} GiB", .{bytesToGiB(device.totalMemory())});
        try printDetailLine(stdout, "  threadgroup memory", "{d} KiB", .{device.maxThreadgroupMemoryLength() / 1024});
        try printDetailLine(stdout, "  raytracing", "{any}", .{device.supportsRaytracing()});
        try printDetailLine(
            stdout,
            "  families",
            "apple7={any} apple8={any} apple9={any} apple10={any} mac2={any}",
            .{
                device.caps.supports_apple7,
                device.caps.supports_apple8,
                device.caps.supports_apple9,
                device.caps.supports_apple10,
                device.caps.supports_mac2,
            },
        );
    } else |err| {
        try printStatusLine(stdout, styles, &summary, .fail, "Metal init", "Initialization failed: {s}", .{@errorName(err)});
    }
    try printStepDuration(stdout, styles, step2_start);

    const step3_start = std.time.nanoTimestamp();
    try printStepHeader(stdout, styles, 3, 4, "Metal Runtime Assets");
    try printCheckingLine(stdout, styles, "Metal shader sources");
    const shader_dir = if (opts.shader_dir.len == 0) "src/shaders/metal" else opts.shader_dir;
    if (inspectShaderSources(shader_dir)) |check| {
        if (check.found == check.total) {
            try printStatusLine(stdout, styles, &summary, .ok, "Shader sources", "{d}/{d} found in {s}", .{ check.found, check.total, shader_dir });
        } else {
            try printStatusLine(
                stdout,
                styles,
                &summary,
                .fail,
                "Shader sources",
                "{d}/{d} found in {s} (missing {s})",
                .{ check.found, check.total, shader_dir, check.first_missing.? },
            );
        }
    } else |err| {
        try printStatusLine(stdout, styles, &summary, .fail, "Shader sources", "Unable to open {s}: {s}", .{ shader_dir, @errorName(err) });
    }

    try printCheckingLine(stdout, styles, "MSL pipeline smoke compile");
    if (maybe_device) |*device| {
        if (compileSmokePipeline(device.ctx)) |compile_check| {
            try printStatusLine(
                stdout,
                styles,
                &summary,
                .ok,
                "MSL compile",
                "tw={d} max={d} stgmem={d}",
                .{
                    compile_check.thread_execution_width,
                    compile_check.max_threads_per_threadgroup,
                    compile_check.static_threadgroup_memory_length,
                },
            );
        } else |err| {
            try printStatusLine(stdout, styles, &summary, .fail, "MSL compile", "Pipeline creation failed: {s}", .{@errorName(err)});
        }
    } else {
        try printStatusLine(stdout, styles, &summary, .skip, "MSL compile", "Skipped (Metal device unavailable)", .{});
    }
    try printStepDuration(stdout, styles, step3_start);

    const step4_start = std.time.nanoTimestamp();
    try printStepHeader(stdout, styles, 4, 4, "Model File");
    if (opts.managed_model) |managed| {
        try printStatusLine(stdout, styles, &summary, .ok, "Managed model", "{s} ({s})", .{ managed.id, managed.status_label });
        try printDetailLine(stdout, "  display name", "{s}", .{managed.display_name});
        try printDetailLine(stdout, "  catalog size", "{d:.2} GiB", .{bytesToGiB(managed.size_bytes)});
    }
    try printModelCheck(stdout, styles, &summary, opts.model_path, opts.managed_model, maybe_device, opts.requested_context_length, allocator);
    try printStepDuration(stdout, styles, step4_start);

    try printSummary(stdout, summary);
    try printVerdict(stdout, styles, summary);

    if (summary.fail != 0) {
        return error.DiagnosticsFailed;
    }

    try stdout.flush();
}

fn printModelCheck(
    writer: anytype,
    styles: Styles,
    summary: *Summary,
    model_path: ?[]const u8,
    managed_model: ?ManagedModelInfo,
    maybe_device: ?MetalDevice,
    requested_context_length: ?u32,
    allocator: std.mem.Allocator,
) !void {
    if (model_path) |path| {
        try printCheckingLine(writer, styles, "GGUF model header and unified-memory fit");
        const inspection = loader_metal.inspectModel(path, allocator) catch |err| {
            try printStatusLine(writer, styles, summary, .fail, "GGUF", "Inspection failed for {s}: {s}", .{ path, @errorName(err) });
            return;
        };

        try printStatusLine(writer, styles, summary, .ok, "GGUF", "{s}", .{path});
        try printDetailLine(
            writer,
            "  architecture",
            "{s} | {d} layers | {d} heads ({d} KV) | dim {d} | vocab {d}",
            .{
                @tagName(inspection.config.architecture),
                inspection.config.n_layers,
                inspection.config.n_heads,
                inspection.config.n_kv_heads,
                inspection.config.hidden_dim,
                inspection.config.vocab_size,
            },
        );
        try printDetailLine(writer, "  file size", "{d:.2} GiB", .{bytesToGiB(inspection.file_size)});
        try printDetailLine(writer, "  tensor payload", "{d:.2} GiB across {d} tensors", .{ bytesToGiB(inspection.tensor_bytes), inspection.tensor_count });
        try printDetailLine(writer, "  metadata entries", "{d}", .{inspection.metadata_count});

        if (maybe_device) |device| {
            const fit = estimateUnifiedFit(inspection, device.recommendedMaxWorkingSetSize(), device.totalMemory(), requested_context_length);
            const status = fit.fitStatus();
            const total_memory_status = fitStatusForUnifiedBytes(fit.total_unified_bytes, fit.total_memory_bytes, fit.total_memory_bytes);
            const headroom = fit.headroomBytes();

            if (fit.total_unified_bytes > fit.total_memory_bytes) {
                try printStatusLine(
                    writer,
                    styles,
                    summary,
                    status,
                    "Unified-memory fit",
                    "{d:.2} / {d:.2} GiB total memory (over by {d:.2} GiB)",
                    .{
                        bytesToGiB(fit.total_unified_bytes),
                        bytesToGiB(fit.total_memory_bytes),
                        bytesToGiB(@intCast(fit.total_unified_bytes - fit.total_memory_bytes)),
                    },
                );
            } else if (fit.total_unified_bytes > fit.recommended_working_set_bytes) {
                try printStatusLine(
                    writer,
                    styles,
                    summary,
                    status,
                    "Unified-memory fit",
                    "{d:.2} / {d:.2} GiB recommended working set (over by {d:.2} GiB, still within total memory)",
                    .{
                        bytesToGiB(fit.total_unified_bytes),
                        bytesToGiB(fit.recommended_working_set_bytes),
                        bytesToGiB(@intCast(-headroom)),
                    },
                );
            } else {
                try printStatusLine(
                    writer,
                    styles,
                    summary,
                    status,
                    "Unified-memory fit",
                    "{d:.2} / {d:.2} GiB recommended working set (headroom {d:.2} GiB)",
                    .{
                        bytesToGiB(fit.total_unified_bytes),
                        bytesToGiB(fit.recommended_working_set_bytes),
                        bytesToGiB(@intCast(headroom)),
                    },
                );
            }

            try printDetailLine(writer, "  weights", "{d:.2} GiB", .{bytesToGiB(fit.weights_bytes)});
            try printDetailLine(writer, "  runtime shared buffers", "{d:.2} GiB", .{bytesToGiB(fit.runtime_unified_bytes)});
            try printDetailLine(writer, "  KV cache", "{d:.2} GiB (ctx cap {d})", .{ bytesToGiB(fit.kv_cache_bytes), fit.max_ctx });
            try printDetailLine(writer, "  budget-fit ctx", "{d} tokens at current UMA budget", .{fit.budget_max_ctx});
            try printDetailLine(writer, "  total memory fit", "{s}", .{total_memory_status.label()});
            try printDetailLine(writer, "  note", "Conservative UMA estimate counts shared runtime buffers and KV cache", .{});
        } else {
            try printStatusLine(writer, styles, summary, .skip, "Unified-memory fit", "Skipped (Metal device unavailable)", .{});
        }
        return;
    }

    if (managed_model) |managed| {
        if (maybe_device) |device| {
            const status = fitStatusForUnifiedBytes(managed.size_bytes, device.recommendedMaxWorkingSetSize(), device.totalMemory());
            try printStatusLine(
                writer,
                styles,
                summary,
                status,
                "Catalog size fit",
                "{d:.2} GiB of weights vs {d:.2} GiB recommended working set",
                .{ bytesToGiB(managed.size_bytes), bytesToGiB(device.recommendedMaxWorkingSetSize()) },
            );
            try printStatusLine(writer, styles, summary, .skip, "GGUF inspection", "Managed model is not installed locally; exact GGUF check skipped", .{});
        } else {
            try printStatusLine(writer, styles, summary, .skip, "GGUF inspection", "Managed model not installed locally and Metal device unavailable", .{});
        }
        return;
    }

    try printStatusLine(writer, styles, summary, .skip, "Model", "No model specified", .{});
}

/// Estimate unified-memory usage for a model given Apple Silicon memory constraints.
pub fn estimateUnifiedFit(
    inspection: ModelInspection,
    recommended_working_set_bytes: u64,
    total_memory_bytes: u64,
    requested_context_length: ?u32,
) UnifiedFitEstimate {
    const config = inspection.config;
    const profile = memory_plan.profile(config);
    const max_ctx = memory_plan.requestedContextTokens(
        config,
        requested_context_length,
        forward_metal.runtime_context_cap,
    );
    const kv_cache_bytes = profile.deviceLocalContextBytes(max_ctx);
    const runtime_unified_bytes = profile.runtimeUnifiedBytes(max_ctx);

    const recommended = if (recommended_working_set_bytes > 0) recommended_working_set_bytes else total_memory_bytes;
    const budget_max_ctx = profile.maxContextTokensForUnifiedBudget(
        inspection.tensor_bytes,
        recommended,
        config.context_length,
    );

    return .{
        .weights_bytes = inspection.tensor_bytes,
        .runtime_unified_bytes = runtime_unified_bytes,
        .total_unified_bytes = inspection.tensor_bytes + runtime_unified_bytes,
        .recommended_working_set_bytes = recommended,
        .total_memory_bytes = total_memory_bytes,
        .kv_cache_bytes = kv_cache_bytes,
        .max_ctx = max_ctx,
        .budget_max_ctx = budget_max_ctx,
    };
}

fn inspectShaderSourcesInDir(dir: std.fs.Dir) ShaderSourceCheck {
    var check = ShaderSourceCheck{};
    for (required_shader_files) |name| {
        const file = dir.openFile(name, .{}) catch {
            if (check.first_missing == null) check.first_missing = name;
            continue;
        };
        var close_file = file;
        close_file.close();
        check.found += 1;
    }
    return check;
}

fn inspectShaderSources(dir_path: []const u8) !ShaderSourceCheck {
    var dir = try std.fs.cwd().openDir(dir_path, .{});
    defer dir.close();
    return inspectShaderSourcesInDir(dir);
}

fn compileSmokePipeline(ctx: ?*metal_c.MetalCtx) !PipelineCompileCheck {
    const msl_source =
        \\#include <metal_stdlib>
        \\using namespace metal;
        \\kernel void main0(device float* out [[buffer(0)]],
        \\                   uint id [[thread_position_in_grid]]) {
        \\    out[id] = float(id);
        \\}
    ;

    var pipe = try metal_pipeline.createPipeline(ctx, msl_source, "main0");
    defer metal_pipeline.freePipeline(&pipe);

    return .{
        .thread_execution_width = pipe.thread_execution_width,
        .max_threads_per_threadgroup = pipe.max_threads_per_threadgroup,
        .static_threadgroup_memory_length = pipe.static_threadgroup_memory_length,
    };
}

fn fitStatusForUnifiedBytes(required_bytes: u64, recommended_working_set_bytes: u64, total_memory_bytes: u64) CheckStatus {
    if (required_bytes > total_memory_bytes) return .fail;
    if (required_bytes > recommended_working_set_bytes) return .warn;
    const total_headroom = @as(i128, total_memory_bytes) - @as(i128, required_bytes);
    if (total_headroom < 1024 * 1024 * 1024) return .warn;
    return .ok;
}

fn printStepHeader(writer: anytype, styles: Styles, index: u32, total: u32, title: []const u8) !void {
    try printStyled(writer, styles, "1;36", "\n[{d}/{d}] {s}\n", .{ index, total, title });
    try writer.flush();
}

fn printCheckingLine(writer: anytype, styles: Styles, label: []const u8) !void {
    try printStyled(writer, styles, "2", "  Checking {s}...\n", .{label});
    try writer.flush();
}

fn printStatusLine(
    writer: anytype,
    styles: Styles,
    summary: *Summary,
    status: CheckStatus,
    label: []const u8,
    comptime value_fmt: []const u8,
    value_args: anytype,
) !void {
    summary.record(status);
    try writer.print("  {s}: ", .{label});
    try writer.print(value_fmt, value_args);
    try writer.print(" ", .{});
    try printStyled(writer, styles, styles.statusCode(status), "[{s}]", .{status.label()});
    try writer.print("\n", .{});
    try writer.flush();
}

fn printDetailLine(writer: anytype, label: []const u8, comptime value_fmt: []const u8, value_args: anytype) !void {
    try writer.print("  {s}: ", .{label});
    try writer.print(value_fmt, value_args);
    try writer.print("\n", .{});
    try writer.flush();
}

fn printStepDuration(writer: anytype, styles: Styles, start_ns: i128) !void {
    const elapsed_ns = std.time.nanoTimestamp() - start_ns;
    try printStyled(writer, styles, "2", "  Step time: ", .{});
    try printDurationValue(writer, elapsed_ns);
    try writer.print("\n", .{});
    try writer.flush();
}

fn printSummary(writer: anytype, summary: Summary) !void {
    try writer.print(
        "\nSummary       : {d} ok, {d} warn, {d} fail, {d} skip\n",
        .{ summary.ok, summary.warn, summary.fail, summary.skip },
    );
    try writer.flush();
}

fn printVerdict(writer: anytype, styles: Styles, summary: Summary) !void {
    if (summary.fail != 0) {
        try writer.print("Verdict       : ", .{});
        try printStyled(writer, styles, "1;31", "NOT READY [FAIL]\n", .{});
    } else if (summary.warn != 0) {
        try writer.print("Verdict       : ", .{});
        try printStyled(writer, styles, "1;33", "READY WITH WARNINGS [WARN]\n", .{});
    } else {
        try writer.print("Verdict       : ", .{});
        try printStyled(writer, styles, "1;32", "READY [OK]\n", .{});
    }
    try writer.flush();
}

fn printStyled(writer: anytype, styles: Styles, code: []const u8, comptime fmt: []const u8, args: anytype) !void {
    if (styles.enabled) {
        try writer.print("\x1b[{s}m", .{code});
        try writer.print(fmt, args);
        try writer.print("\x1b[0m", .{});
    } else {
        try writer.print(fmt, args);
    }
}

fn printDurationValue(writer: anytype, elapsed_ns: i128) !void {
    if (elapsed_ns >= std.time.ns_per_s) {
        try writer.print("{d:.2} s", .{@as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(std.time.ns_per_s))});
    } else if (elapsed_ns >= std.time.ns_per_ms) {
        try writer.print("{d:.2} ms", .{@as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(std.time.ns_per_ms))});
    } else if (elapsed_ns >= std.time.ns_per_us) {
        try writer.print("{d:.2} us", .{@as(f64, @floatFromInt(elapsed_ns)) / @as(f64, @floatFromInt(std.time.ns_per_us))});
    } else {
        try writer.print("{d} ns", .{elapsed_ns});
    }
}

fn shouldUseColor(stdout_is_tty: bool, term: ?[]const u8, no_color: bool, force_color: bool) bool {
    if (no_color) return false;
    if (force_color) return true;
    if (!stdout_is_tty) return false;
    if (term) |term_value| {
        if (std.mem.eql(u8, term_value, "dumb")) return false;
    }
    return true;
}

fn isTruthy(value: ?[]const u8) bool {
    const text = value orelse return false;
    return std.mem.eql(u8, text, "1") or std.mem.eql(u8, text, "true") or std.mem.eql(u8, text, "yes");
}

fn bytesToGiB(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
}

test "inspectShaderSources reports all shaders present" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    for (required_shader_files) |name| {
        try tmp.dir.writeFile(.{ .sub_path = name, .data = "" });
    }

    const check = inspectShaderSourcesInDir(tmp.dir);
    try std.testing.expectEqual(required_shader_files.len, check.found);
    try std.testing.expectEqual(required_shader_files.len, check.total);
    try std.testing.expect(check.first_missing == null);
}

test "fitStatusForUnifiedBytes distinguishes ok, warn, and fail" {
    try std.testing.expectEqual(CheckStatus.ok, fitStatusForUnifiedBytes(8 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024, 32 * 1024 * 1024 * 1024));
    try std.testing.expectEqual(CheckStatus.warn, fitStatusForUnifiedBytes(20 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024, 32 * 1024 * 1024 * 1024));
    try std.testing.expectEqual(CheckStatus.fail, fitStatusForUnifiedBytes(40 * 1024 * 1024 * 1024, 32 * 1024 * 1024 * 1024, 32 * 1024 * 1024 * 1024));
}

test "estimateUnifiedFit includes weights, runtime buffers, and kv cache" {
    const config = ModelConfig{
        .architecture = .qwen2_moe,
        .n_layers = 40,
        .n_heads = 16,
        .n_kv_heads = 2,
        .head_dim = 256,
        .hidden_dim = 2048,
        .intermediate_dim = 512,
        .vocab_size = 248320,
        .context_length = 32768,
        .rope_freq_base = 10000000.0,
        .rms_norm_eps = 1e-6,
        .n_experts = 256,
        .n_experts_used = 8,
        .rope_dim = 64,
        .ssm_d_conv = 4,
        .ssm_d_inner = 4096,
        .ssm_d_state = 128,
        .ssm_dt_rank = 32,
        .ssm_n_group = 16,
        .full_attn_interval = 4,
        .shared_expert_intermediate_dim = 512,
    };
    const inspection = ModelInspection{
        .config = config,
        .file_size = 0,
        .tensor_bytes = 21 * 1024 * 1024 * 1024,
        .tensor_count = 733,
        .metadata_count = 52,
    };

    const fit = estimateUnifiedFit(inspection, 48 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024, null);
    try std.testing.expectEqual(@as(u32, 4096), fit.max_ctx);
    try std.testing.expect(fit.budget_max_ctx >= fit.max_ctx);
    try std.testing.expect(fit.weights_bytes == inspection.tensor_bytes);
    try std.testing.expect(fit.kv_cache_bytes > 0);
    try std.testing.expect(fit.runtime_unified_bytes > fit.kv_cache_bytes);
    try std.testing.expect(fit.total_unified_bytes > fit.weights_bytes);
    try std.testing.expectEqual(CheckStatus.ok, fit.fitStatus());
}

test "estimateUnifiedFit respects requested context ceiling on Metal" {
    const config = ModelConfig{
        .architecture = .qwen2_moe,
        .n_layers = 40,
        .n_heads = 16,
        .n_kv_heads = 2,
        .head_dim = 256,
        .hidden_dim = 2048,
        .intermediate_dim = 512,
        .vocab_size = 248320,
        .context_length = 32768,
        .rope_freq_base = 10000000.0,
        .rms_norm_eps = 1e-6,
        .n_experts = 256,
        .n_experts_used = 8,
        .rope_dim = 64,
        .ssm_d_conv = 4,
        .ssm_d_inner = 4096,
        .ssm_d_state = 128,
        .ssm_dt_rank = 32,
        .ssm_n_group = 16,
        .full_attn_interval = 4,
        .shared_expert_intermediate_dim = 512,
    };
    const inspection = ModelInspection{
        .config = config,
        .file_size = 0,
        .tensor_bytes = 21 * 1024 * 1024 * 1024,
        .tensor_count = 733,
        .metadata_count = 52,
    };

    const requested_small = estimateUnifiedFit(inspection, 48 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024, 2048);
    try std.testing.expectEqual(@as(u32, 2048), requested_small.max_ctx);

    const requested_large = estimateUnifiedFit(inspection, 48 * 1024 * 1024 * 1024, 64 * 1024 * 1024 * 1024, 8192);
    try std.testing.expectEqual(@as(u32, forward_metal.runtime_context_cap), requested_large.max_ctx);
}
