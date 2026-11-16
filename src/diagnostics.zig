//! Vulkan system diagnostics (`zinc --check`).
//! Probes the host environment, Vulkan driver, GPU capabilities,
//! shader assets, and optional GGUF model fit to produce a human-readable
//! preflight report on stdout.
//! @section Hardware Detection
const std = @import("std");
const diagnostic_report = @import("diagnostic_report.zig");
const host_io = @import("host_io");
const builtin = @import("builtin");
const gguf = @import("model/gguf.zig");
const memory_plan = @import("gpu/memory_plan.zig");
const kv_dtype = @import("compute/kv_dtype.zig");
const loader_mod = @import("model/loader.zig");
const gpu_detect = @import("vulkan/gpu_detect.zig");
const instance_mod = @import("vulkan/instance.zig");
const vk = @import("vulkan/vk.zig");

const Instance = instance_mod.Instance;
const ModelConfig = loader_mod.ModelConfig;
const ModelInspection = loader_mod.ModelInspection;

/// Configuration for a Vulkan diagnostics run.
pub const Options = struct {
    /// Optional structured sink; the same checks feed human and JSON reports.
    report: ?*diagnostic_report.Report = null,
    output_writer: ?*std.Io.Writer = null,
    /// GPU device index to probe, or auto-select when unset by CLI.
    device_index: u32 = instance_mod.auto_select_device_index,
    /// Path to a GGUF model file for inspection, or null to skip.
    model_path: ?[]const u8 = null,
    /// Requested runtime context ceiling from CLI/server configuration.
    requested_context_length: ?u32 = null,
    /// Managed model catalog entry, if selected via `--model-id`.
    managed_model: ?ManagedModelInfo = null,
    /// Directory containing compiled SPIR-V shader assets.
    shader_dir: []const u8 = "zig-out/share/zinc/shaders",
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

    fn detect(stdout_file: std.Io.File) Styles {
        const no_color = host_io.getenv("NO_COLOR") != null;
        const force_color =
            isTruthy(host_io.getenv("FORCE_COLOR")) or
            isTruthy(host_io.getenv("CLICOLOR_FORCE"));
        return .{
            .enabled = shouldUseColor((stdout_file.isTty(host_io.io) catch false), host_io.getenv("TERM"), no_color, force_color),
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

const CheckStatus = diagnostic_report.Status;

const Summary = struct {
    report: ?*diagnostic_report.Report = null,
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

const ShaderAssetCheck = struct {
    found: usize = 0,
    total: usize = required_shader_files.len,
    first_missing: ?[]const u8 = null,
};

const VulkanProbe = struct {
    instance: Instance,
    gpu_config: gpu_detect.GpuConfig,
    requested_index: u32,
    selected_index: u32,
    device_count: u32,

    fn deinit(self: *VulkanProbe) void {
        self.instance.deinit();
        self.* = undefined;
    }
};

/// Estimated VRAM breakdown for running a model on a discrete GPU.
pub const FitEstimate = struct {
    /// Tensor weight payload bytes uploaded to device-local VRAM.
    weights_bytes: u64,
    /// All device-local runtime bytes: activation/scratch, KV cache, and SSM state.
    runtime_device_local_bytes: u64,
    /// Staging buffers allocated in host-visible memory.
    host_visible_bytes: u64,
    /// KV cache bytes allocated in device-local VRAM.
    kv_cache_bytes: u64,
    /// SSM convolution and recurrent state bytes on the GPU.
    gpu_ssm_bytes: u64,
    /// Sum of weights and all device-local runtime bytes (includes KV cache and SSM state).
    total_device_local_bytes: u64,
    /// Available VRAM budget reported by the Vulkan driver.
    vram_budget_bytes: u64,
    /// Current runtime context cap used for KV cache sizing.
    max_ctx: u32,
    /// Maximum context the current VRAM budget could sustain if runtime caps were lifted.
    budget_max_ctx: u32,

    fn headroomBytes(self: FitEstimate) i128 {
        return @as(i128, self.vram_budget_bytes) - @as(i128, self.total_device_local_bytes);
    }

    fn fitStatus(self: FitEstimate) CheckStatus {
        const headroom = self.headroomBytes();
        if (headroom < 0) return .fail;
        if (headroom < 1024 * 1024 * 1024) return .warn;
        return .ok;
    }
};

const GgufHeader = struct {
    version: u32,
    tensor_count: u64,
    metadata_count: u64,
};

const required_shader_files = [_][]const u8{
    "dmmv_q4k.spv",
    "dmmv_q5k.spv",
    "dmmv_q6k.spv",
    "dmmv_q8_0.spv",
    "dmmv_f16.spv",
    "dmmv_f32.spv",
    "dmmv_q4k_moe.spv",
    "dmmv_q5k_moe.spv",
    "flash_attn.spv",
    "rms_norm_mul.spv",
    "swiglu.spv",
    "swiglu_oai.spv",
    "rope_fused.spv",
    "deinterleave.spv",
    "sigmoid_mul.spv",
    "vadd.spv",
    "scale_accumulate.spv",
    "bias_add.spv",
    "ssm_conv1d.spv",
    "ssm_delta_net.spv",
    "ssm_gated_norm.spv",
    "softmax_topk.spv",
    "sigmoid_scale_acc.spv",
    "moe_weighted_acc.spv",
};

/// Run the five-step preflight diagnostics sequence and print a human-readable report to stdout.
/// Probes host OS, Vulkan driver, compiled shader assets, GPU device capabilities, and
/// (optionally) a GGUF model file. Returns `error.DiagnosticsFailed` if any check is marked FAIL.
/// @param opts Diagnostic configuration: device index, optional model path, shader dir, etc.
/// @param allocator Used for Vulkan device enumeration and model inspection scratch space.
pub fn run(opts: Options, allocator: std.mem.Allocator) !void {
    const stdout_file = std.Io.File.stdout();
    var stdout_buffer: [4096]u8 = undefined;
    var stdout_writer = stdout_file.writerStreaming(host_io.io, &stdout_buffer);
    const stdout = opts.output_writer orelse &stdout_writer.interface;
    const styles = Styles.detect(stdout_file);
    var summary = Summary{ .report = opts.report };

    try stdout.print("\n=== ZINC System Diagnostics ===\n", .{});
    try stdout.flush();

    const step1_start = host_io.nanoTimestamp();
    try printStepHeader(stdout, styles, 1, 5, "Host Environment");
    const os_status: CheckStatus = if (builtin.os.tag == .linux) .ok else .warn;
    try printStatusLine(stdout, styles, &summary, os_status, "OS", "{s}", .{@tagName(builtin.os.tag)});
    try printStepDuration(stdout, styles, step1_start);

    var vulkan_probe: ?VulkanProbe = null;
    var vulkan_probe_err: ?anyerror = null;
    if (probeVulkan(allocator, opts.device_index)) |probe| {
        vulkan_probe = probe;
    } else |err| {
        vulkan_probe_err = err;
    }
    defer {
        if (vulkan_probe) |*probe| probe.deinit();
    }

    const step2_start = host_io.nanoTimestamp();
    const driver_vendor: ?gpu_detect.GpuVendor = if (vulkan_probe) |*probe| probe.gpu_config.vendor else null;
    try printStepHeader(stdout, styles, 2, 5, driverStepTitle(driver_vendor));
    try printLinuxDriverChecks(stdout, styles, &summary, allocator, if (vulkan_probe) |*probe| probe else null);
    try printStepDuration(stdout, styles, step2_start);

    const step3_start = host_io.nanoTimestamp();
    try printStepHeader(stdout, styles, 3, 5, "Runtime Assets");
    try printCheckingLine(stdout, styles, "compiled shader assets");
    try printShaderAssets(stdout, styles, &summary, opts.shader_dir);
    try printStepDuration(stdout, styles, step3_start);

    const step4_start = host_io.nanoTimestamp();
    try printStepHeader(stdout, styles, 4, 5, "Vulkan Device");
    try printCheckingLine(stdout, styles, "Vulkan loader, device enumeration, and logical device init");
    var vram_budget_bytes: ?u64 = null;
    if (vulkan_probe) |*probe| {
        vram_budget_bytes = probe.instance.vramBytes();
        if (opts.report) |report| try report.setDevice(probe.gpu_config.nameSlice(), vram_budget_bytes.?);
        try printVulkanProbe(stdout, styles, &summary, probe);
    } else if (vulkan_probe_err) |err| {
        const status: CheckStatus = if (builtin.os.tag == .linux) .fail else .warn;
        try printStatusLine(stdout, styles, &summary, status, "Vulkan", "Initialization failed: {s}", .{@errorName(err)});
    }
    try printStepDuration(stdout, styles, step4_start);

    const step5_start = host_io.nanoTimestamp();
    try printStepHeader(stdout, styles, 5, 5, "Model File");
    if (opts.managed_model != null) {
        try printCheckingLine(stdout, styles, "managed model compatibility");
    } else if (opts.model_path != null) {
        try printCheckingLine(stdout, styles, "GGUF model header");
    }
    try printModelCheck(stdout, styles, &summary, opts.model_path, opts.managed_model, vram_budget_bytes, opts.requested_context_length, allocator);
    try printStepDuration(stdout, styles, step5_start);

    try printSummary(stdout, summary);
    try printVerdict(stdout, styles, summary);

    if (summary.fail != 0) {
        return error.DiagnosticsFailed;
    }

    try stdout.flush();
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
    if (summary.report) |report| try report.add(label, status, value_fmt, value_args);
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
    const elapsed_ns = host_io.nanoTimestamp() - start_ns;
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

fn driverStepTitle(vendor: ?gpu_detect.GpuVendor) []const u8 {
    const actual = vendor orelse return "Linux GPU Driver Checks";
    if (isAmdVendor(actual)) return "Linux AMD Driver Checks";
    if (isIntelVendor(actual)) return "Linux Intel Driver Checks";
    return "Linux GPU Driver Checks";
}

fn printLinuxDriverChecks(
    writer: anytype,
    styles: Styles,
    summary: *Summary,
    allocator: std.mem.Allocator,
    probe: ?*const VulkanProbe,
) !void {
    if (builtin.os.tag != .linux) {
        try printStatusLine(writer, styles, summary, .skip, "GPU driver", "Linux-only check", .{});
        try printStatusLine(writer, styles, summary, .skip, "RADV_PERFTEST", "Linux-only check", .{});
        try printStatusLine(writer, styles, summary, .skip, "GECC (RAS)", "Linux-only check", .{});
        return;
    }

    const actual = probe orelse {
        try printStatusLine(writer, styles, summary, .skip, "GPU driver", "Skipped until Vulkan device probe completes", .{});
        return;
    };

    if (isAmdVendor(actual.gpu_config.vendor)) {
        try printCheckingLine(writer, styles, "Mesa Vulkan driver package");
        try printMesa(writer, styles, summary, allocator);
        try printCheckingLine(writer, styles, "RADV cooperative-matrix flag");
        try printRadvPerftest(writer, styles, summary);
        try printCheckingLine(writer, styles, "GECC / RAS status");
        try printGecc(writer, styles, summary, allocator);
        return;
    }

    if (isIntelVendor(actual.gpu_config.vendor)) {
        try printStatusLine(
            writer,
            styles,
            summary,
            .ok,
            "Intel Vulkan",
            "ANV-compatible device detected ({s})",
            .{actual.gpu_config.nameSlice()},
        );
        try printStatusLine(writer, styles, summary, .skip, "RADV_PERFTEST", "AMD RADV-only tuning flag", .{});
        try printStatusLine(writer, styles, summary, .skip, "GECC (RAS)", "AMD-only reliability check", .{});
        return;
    }

    try printStatusLine(
        writer,
        styles,
        summary,
        .ok,
        "Vulkan driver",
        "Device detected ({s})",
        .{actual.gpu_config.nameSlice()},
    );
    try printStatusLine(writer, styles, summary, .skip, "RADV_PERFTEST", "AMD RADV-only tuning flag", .{});
    try printStatusLine(writer, styles, summary, .skip, "GECC (RAS)", "AMD-only reliability check", .{});
}

fn isAmdVendor(vendor: gpu_detect.GpuVendor) bool {
    return switch (vendor) {
        .amd_rdna3, .amd_rdna4, .amd_rdna4_apu, .amd_other => true,
        else => false,
    };
}

fn isIntelVendor(vendor: gpu_detect.GpuVendor) bool {
    return switch (vendor) {
        .intel_arc, .intel_arc_xe2 => true,
        else => false,
    };
}

fn printRadvPerftest(writer: anytype, styles: Styles, summary: *Summary) !void {
    if (host_io.getenv("RADV_PERFTEST")) |val| {
        if (std.mem.indexOf(u8, val, "coop_matrix") != null) {
            try printStatusLine(writer, styles, summary, .ok, "RADV_PERFTEST", "{s}", .{val});
        } else {
            try printStatusLine(writer, styles, summary, .warn, "RADV_PERFTEST", "{s} (missing coop_matrix)", .{val});
        }
    } else {
        try printStatusLine(writer, styles, summary, .warn, "RADV_PERFTEST", "Not set in current shell", .{});
    }
}

fn printMesa(writer: anytype, styles: Styles, summary: *Summary, allocator: std.mem.Allocator) !void {
    if (getMesaVersion(allocator)) |ver| {
        defer allocator.free(ver);

        if (std.mem.indexOf(u8, ver, "25.0.7") != null) {
            try printStatusLine(writer, styles, summary, .ok, "Mesa", "{s}", .{ver});
        } else {
            try printStatusLine(writer, styles, summary, .warn, "Mesa", "{s} (25.0.7 recommended)", .{ver});
        }
    } else |_| {
        try printStatusLine(writer, styles, summary, .warn, "Mesa", "Not found (dpkg-query or mesa-vulkan-drivers missing)", .{});
    }
}

fn printGecc(writer: anytype, styles: Styles, summary: *Summary, allocator: std.mem.Allocator) !void {
    if (getGeccStatus(allocator)) |status| {
        defer allocator.free(status);

        const trimmed = std.mem.trim(u8, status, " \n\r\t");
        if (std.mem.eql(u8, trimmed, "0")) {
            try printStatusLine(writer, styles, summary, .ok, "GECC (RAS)", "Disabled", .{});
        } else {
            try printStatusLine(writer, styles, summary, .warn, "GECC (RAS)", "Enabled ({s})", .{trimmed});
        }
    } else |_| {
        try printStatusLine(writer, styles, summary, .warn, "GECC (RAS)", "Unknown (could not read ras_enable)", .{});
    }
}

fn printShaderAssets(writer: anytype, styles: Styles, summary: *Summary, shader_dir: []const u8) !void {
    var dir = std.Io.Dir.cwd().openDir(host_io.io, shader_dir, .{}) catch |err| {
        const status: CheckStatus = if (builtin.os.tag == .linux) .fail else .warn;
        try printStatusLine(writer, styles, summary, status, "Shader dir", "{s} ({s})", .{ shader_dir, @errorName(err) });
        return;
    };
    defer dir.close(host_io.io);

    try printStatusLine(writer, styles, summary, .ok, "Shader dir", "{s}", .{shader_dir});

    const shader_check = inspectShaderAssets(dir);
    if (shader_check.found == shader_check.total) {
        try printStatusLine(writer, styles, summary, .ok, "Required shaders", "{d}/{d} present", .{ shader_check.found, shader_check.total });
    } else {
        const status: CheckStatus = if (builtin.os.tag == .linux) .fail else .warn;
        try printStatusLine(
            writer,
            styles,
            summary,
            status,
            "Required shaders",
            "{d}/{d} present (first missing: {s})",
            .{ shader_check.found, shader_check.total, shader_check.first_missing.? },
        );
    }
}

fn printVulkanProbe(writer: anytype, styles: Styles, summary: *Summary, probe: *const VulkanProbe) !void {
    try printStatusLine(writer, styles, summary, .ok, "Vulkan", "Instance and logical device initialized", .{});
    try printDetailLine(writer, "Visible GPUs", "{d}", .{probe.device_count});
    if (probe.requested_index == instance_mod.auto_select_device_index) {
        try printDetailLine(writer, "Selected GPU", "#{d} {s} (auto)", .{ probe.selected_index, probe.gpu_config.nameSlice() });
    } else if (probe.requested_index != probe.selected_index) {
        try printDetailLine(writer, "Selected GPU", "#{d} (requested #{d} out of range)", .{ probe.selected_index, probe.requested_index });
    } else {
        try printDetailLine(writer, "Selected GPU", "#{d} {s}", .{ probe.selected_index, probe.gpu_config.nameSlice() });
    }
    try printDetailLine(writer, "Queue family", "{d}", .{probe.instance.compute_queue_family});
    try printStatusLine(
        writer,
        styles,
        summary,
        .ok,
        "GPU tuning",
        "{s} | {d} MB VRAM | {d} GB/s | wave{d} | coopmat={s}",
        .{
            @tagName(probe.gpu_config.vendor),
            probe.gpu_config.vram_mb,
            probe.gpu_config.bandwidth_gbps,
            probe.gpu_config.wave_size,
            if (probe.gpu_config.coopmat_support) "yes" else "no",
        },
    );
}

fn printModelCheck(
    writer: anytype,
    styles: Styles,
    summary: *Summary,
    model_path: ?[]const u8,
    managed_model: ?ManagedModelInfo,
    vram_budget_bytes: ?u64,
    requested_context_length: ?u32,
    allocator: std.mem.Allocator,
) !void {
    if (managed_model) |managed| {
        if (model_path) |path| {
            const inspection = loader_mod.inspectModel(path, allocator) catch |err| {
                try printStatusLine(writer, styles, summary, .fail, "Managed model", "{s} ({s}) ({s})", .{ managed.display_name, managed.id, @errorName(err) });
                return;
            };

            try printStatusLine(writer, styles, summary, .ok, "Managed model", "{s} ({s})", .{ managed.display_name, managed.id });
            try printDetailLine(writer, "Resolved path", "{s}", .{path});
            try printInspectedModelDetails(writer, styles, summary, inspection, vram_budget_bytes, requested_context_length);
            return;
        }

        try printStatusLine(writer, styles, summary, .ok, "Managed model", "{s} ({s})", .{ managed.display_name, managed.id });
        try printDetailLine(writer, "Catalog file", "{s}", .{managed.file_name});
        try printDetailLine(writer, "Catalog size", "{d:.2} GiB", .{bytesToGiB(managed.size_bytes)});
        try printDetailLine(writer, "Catalog status", "{s}", .{managed.status_label});
        try printDetailLine(writer, "Installation", "Not present in local cache; using catalog metadata", .{});

        if (vram_budget_bytes) |budget| {
            const fit_status = fitStatusForBytes(managed.required_vram_bytes, budget);
            const headroom = @as(i128, budget) - @as(i128, managed.required_vram_bytes);
            if (headroom >= 0) {
                try printStatusLine(
                    writer,
                    styles,
                    summary,
                    fit_status,
                    "VRAM fit (catalog)",
                    "{d:.2} / {d:.2} GiB device-local (headroom {d:.2} GiB)",
                    .{
                        bytesToGiB(managed.required_vram_bytes),
                        bytesToGiB(budget),
                        bytesToGiB(@intCast(headroom)),
                    },
                );
            } else {
                try printStatusLine(
                    writer,
                    styles,
                    summary,
                    fit_status,
                    "VRAM fit (catalog)",
                    "{d:.2} / {d:.2} GiB device-local (over by {d:.2} GiB)",
                    .{
                        bytesToGiB(managed.required_vram_bytes),
                        bytesToGiB(budget),
                        bytesToGiB(@intCast(-headroom)),
                    },
                );
            }
        } else {
            try printDetailLine(writer, "Catalog VRAM requirement", "{d:.2} GiB device-local", .{bytesToGiB(managed.required_vram_bytes)});
        }
        return;
    }

    const path = model_path orelse {
        try printStatusLine(writer, styles, summary, .skip, "Model", "Skipped (pass -m/--model or --model-id to validate a model)", .{});
        return;
    };

    const inspection = loader_mod.inspectModel(path, allocator) catch |err| {
        try printStatusLine(writer, styles, summary, .fail, "Model", "{s} ({s})", .{ path, @errorName(err) });
        return;
    };

    try printStatusLine(writer, styles, summary, .ok, "Model", "{s}", .{path});
    try printInspectedModelDetails(writer, styles, summary, inspection, vram_budget_bytes, requested_context_length);
}

fn printInspectedModelDetails(
    writer: anytype,
    styles: Styles,
    summary: *Summary,
    inspection: ModelInspection,
    vram_budget_bytes: ?u64,
    requested_context_length: ?u32,
) !void {
    try printDetailLine(writer, "File size", "{d:.2} GiB", .{bytesToGiB(inspection.file_size)});
    try printDetailLine(writer, "Tensor upload", "{d:.2} GiB device-local weights", .{bytesToGiB(inspection.tensor_bytes)});
    try printDetailLine(writer, "GGUF header", "{d} tensors | {d} metadata entries", .{ inspection.tensor_count, inspection.metadata_count });
    try printDetailLine(
        writer,
        "Model config",
        "{d} layers | {d} heads ({d} KV) | dim {d} | ctx {d}",
        .{
            inspection.config.n_layers,
            inspection.config.n_heads,
            inspection.config.n_kv_heads,
            inspection.config.hidden_dim,
            inspection.config.context_length,
        },
    );

    if (vram_budget_bytes) |budget| {
        const fit = estimateFit(inspection, budget, requested_context_length);
        if (summary.report) |report| report.memory = .{
            .source = .estimate,
            .weights_bytes = fit.weights_bytes,
            .runtime_bytes = fit.runtime_device_local_bytes - fit.kv_cache_bytes,
            .kv_bytes = fit.kv_cache_bytes,
            .total_bytes = fit.total_device_local_bytes,
            .budget_bytes = budget,
            .context_length = fit.max_ctx,
        };
        const fit_status = fit.fitStatus();
        const headroom = fit.headroomBytes();
        if (headroom >= 0) {
            try printStatusLine(
                writer,
                styles,
                summary,
                fit_status,
                "VRAM fit",
                "{d:.2} / {d:.2} GiB device-local (headroom {d:.2} GiB)",
                .{
                    bytesToGiB(fit.total_device_local_bytes),
                    bytesToGiB(fit.vram_budget_bytes),
                    bytesToGiB(@intCast(headroom)),
                },
            );
        } else {
            try printStatusLine(
                writer,
                styles,
                summary,
                fit_status,
                "VRAM fit",
                "{d:.2} / {d:.2} GiB device-local (over by {d:.2} GiB)",
                .{
                    bytesToGiB(fit.total_device_local_bytes),
                    bytesToGiB(fit.vram_budget_bytes),
                    bytesToGiB(@intCast(-headroom)),
                },
            );
        }
        try printDetailLine(writer, "  weights", "{d:.2} GiB", .{bytesToGiB(fit.weights_bytes)});
        try printDetailLine(writer, "  runtime device-local", "{d:.2} GiB", .{bytesToGiB(fit.runtime_device_local_bytes)});
        try printDetailLine(writer, "  KV cache", "{d:.2} GiB (ctx cap {d})", .{ bytesToGiB(fit.kv_cache_bytes), fit.max_ctx });
        try printDetailLine(writer, "  budget-fit ctx", "{d} tokens at current VRAM budget", .{fit.budget_max_ctx});
        try printDetailLine(writer, "  GPU SSM state", "{d:.2} GiB", .{bytesToGiB(fit.gpu_ssm_bytes)});
        try printDetailLine(writer, "  host-visible staging", "{d:.2} GiB (reported separately)", .{bytesToGiB(fit.host_visible_bytes)});
        try printDetailLine(writer, "  note", "Fit excludes Vulkan alignment, descriptor pools, query pools, and driver overhead", .{});
    } else {
        try printStatusLine(writer, styles, summary, .skip, "VRAM fit", "Skipped (GPU probe not available)", .{});
    }
}

fn fitStatusForBytes(required_bytes: u64, budget_bytes: u64) CheckStatus {
    const headroom = @as(i128, budget_bytes) - @as(i128, required_bytes);
    if (headroom < 0) return .fail;
    if (headroom < 1024 * 1024 * 1024) return .warn;
    return .ok;
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

fn inspectShaderAssets(dir: std.Io.Dir) ShaderAssetCheck {
    var check = ShaderAssetCheck{};
    for (required_shader_files) |name| {
        const file = dir.openFile(host_io.io, name, .{}) catch {
            if (check.first_missing == null) check.first_missing = name;
            continue;
        };
        var close_file = file;
        close_file.close(host_io.io);
        check.found += 1;
    }
    return check;
}

fn readGgufHeader(file: std.Io.File) !GgufHeader {
    var buf: [24]u8 = undefined;
    const n = try file.preadAll(&buf, 0);
    if (n < buf.len) return error.ShortRead;

    const magic = std.mem.readInt(u32, buf[0..4], .little);
    if (magic != gguf.GGUF_MAGIC) return error.InvalidMagic;

    return .{
        .version = std.mem.readInt(u32, buf[4..8], .little),
        .tensor_count = std.mem.readInt(u64, buf[8..16], .little),
        .metadata_count = std.mem.readInt(u64, buf[16..24], .little),
    };
}

/// Compute a `FitEstimate` that breaks down VRAM usage for a model at a given budget.
/// Derives weights, KV cache, SSM state, and runtime buffer sizes from the model
/// config, inspected projection widths and selected Vulkan KV layout. Returns
/// per-category byte counts without evaluating fit status.
/// @param inspection GGUF model inspection result supplying tensor byte count and config.
/// @param vram_budget_bytes Device-local VRAM available, as reported by the Vulkan driver.
/// @param requested_context_length Optional context ceiling override; null uses the model default.
/// @returns A `FitEstimate` with per-category byte counts and a budget-headroom context maximum.
pub fn estimateFit(inspection: ModelInspection, vram_budget_bytes: u64, requested_context_length: ?u32) FitEstimate {
    // This is the runtime's environment-driven selector, not a fixed default.
    return estimateFitWithKvBlockBytes(inspection, vram_budget_bytes, requested_context_length, kv_dtype.bytesPer32Elems());
}

fn estimateFitWithKvBlockBytes(inspection: ModelInspection, vram_budget_bytes: u64, requested_context_length: ?u32, kv_block_bytes: u64) FitEstimate {
    const config = inspection.config;
    var profile = memory_plan.profileWithKvBlockBytes(config, kv_block_bytes);
    if (inspection.kv_dimension_sum) |dimensions| {
        // Match Vulkan's selected F16/F32/Q8 layout and actual Gemma layer
        // widths. F32 metadata maxima falsely rejected 8K on a 32 GB card.
        // Inspected rows are aligned, but callers can also construct this
        // public summary. Round up without overflowing dimensions + 31.
        const blocks = dimensions / 32 + @intFromBool(dimensions % 32 != 0);
        profile.device_local_bytes_per_token = blocks * kv_block_bytes * 2;
    }
    const max_ctx = memory_plan.effectiveContextCeiling(config, requested_context_length);
    const kv_cache_bytes = profile.deviceLocalContextBytes(max_ctx);
    const runtime_device_local_bytes = profile.runtimeDeviceLocalBytes(max_ctx);
    const host_visible_bytes = profile.runtimeHostVisibleBytes(max_ctx);
    const budget_max_ctx = profile.maxContextTokensForDeviceLocalBudget(
        inspection.tensor_bytes,
        vram_budget_bytes,
        config.context_length,
    );

    return .{
        .weights_bytes = inspection.tensor_bytes,
        .runtime_device_local_bytes = runtime_device_local_bytes,
        .host_visible_bytes = host_visible_bytes,
        .kv_cache_bytes = kv_cache_bytes,
        .gpu_ssm_bytes = profile.gpu_ssm_bytes,
        .total_device_local_bytes = inspection.tensor_bytes + runtime_device_local_bytes,
        .vram_budget_bytes = vram_budget_bytes,
        .max_ctx = max_ctx,
        .budget_max_ctx = budget_max_ctx,
    };
}

fn bytesToGiB(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
}

fn probeVulkan(allocator: std.mem.Allocator, preferred_device: u32) !VulkanProbe {
    const app_info = vk.c.VkApplicationInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = null,
        .pApplicationName = "zinc",
        .applicationVersion = vk.c.VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "zinc",
        .engineVersion = vk.c.VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = vk.c.VK_API_VERSION_1_3,
    };

    const create_info = vk.c.VkInstanceCreateInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
    };

    var handle: vk.c.VkInstance = null;
    const create_result = vk.c.vkCreateInstance(&create_info, null, &handle);
    if (create_result != vk.c.VK_SUCCESS) return error.VulkanInitFailed;
    errdefer vk.c.vkDestroyInstance(handle, null);

    var dev_count: u32 = 0;
    _ = vk.c.vkEnumeratePhysicalDevices(handle, &dev_count, null);
    if (dev_count == 0) return error.NoDevices;

    const phys_devices = try allocator.alloc(vk.c.VkPhysicalDevice, dev_count);
    defer allocator.free(phys_devices);
    _ = vk.c.vkEnumeratePhysicalDevices(handle, &dev_count, phys_devices.ptr);

    const selected_index = try instance_mod.selectPhysicalDeviceIndex(allocator, phys_devices, dev_count, preferred_device);
    const physical_device = phys_devices[selected_index];

    var device_props: vk.c.VkPhysicalDeviceProperties = undefined;
    vk.c.vkGetPhysicalDeviceProperties(physical_device, &device_props);

    var mem_props: vk.c.VkPhysicalDeviceMemoryProperties = undefined;
    vk.c.vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

    var qf_count: u32 = 0;
    vk.c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count, null);
    const qf_props = try allocator.alloc(vk.c.VkQueueFamilyProperties, qf_count);
    defer allocator.free(qf_props);
    vk.c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count, qf_props.ptr);

    var compute_family: ?u32 = null;
    for (qf_props[0..qf_count], 0..) |qf, qi| {
        if (qf.queueFlags & vk.c.VK_QUEUE_COMPUTE_BIT != 0 and qf.queueFlags & vk.c.VK_QUEUE_GRAPHICS_BIT == 0) {
            compute_family = @intCast(qi);
            break;
        }
    }
    if (compute_family == null) {
        for (qf_props[0..qf_count], 0..) |qf, qi| {
            if (qf.queueFlags & vk.c.VK_QUEUE_COMPUTE_BIT != 0) {
                compute_family = @intCast(qi);
                break;
            }
        }
    }
    const compute_queue_family = compute_family orelse return error.NoComputeQueue;

    const queue_priority: f32 = 1.0;
    const queue_create_info = vk.c.VkDeviceQueueCreateInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .queueFamilyIndex = compute_queue_family,
        .queueCount = 1,
        .pQueuePriorities = &queue_priority,
    };

    const device_create_info = vk.c.VkDeviceCreateInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .queueCreateInfoCount = 1,
        .pQueueCreateInfos = &queue_create_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
        .pEnabledFeatures = null,
    };

    var device: vk.c.VkDevice = null;
    const device_result = vk.c.vkCreateDevice(physical_device, &device_create_info, null, &device);
    if (device_result != vk.c.VK_SUCCESS) return error.DeviceCreateFailed;
    errdefer vk.c.vkDestroyDevice(device, null);

    var compute_queue: vk.c.VkQueue = null;
    vk.c.vkGetDeviceQueue(device, compute_queue_family, 0, &compute_queue);

    var instance = Instance{
        .handle = handle,
        .physical_device = physical_device,
        .device = device,
        .compute_queue = compute_queue,
        .compute_queue_family = compute_queue_family,
        .device_props = device_props,
        .mem_props = mem_props,
        .selected_device_index = selected_index,
        .allocator = allocator,
    };

    return .{
        .instance = instance,
        .gpu_config = gpu_detect.detect(&instance),
        .requested_index = preferred_device,
        .selected_index = selected_index,
        .device_count = dev_count,
    };
}

fn getMesaVersion(allocator: std.mem.Allocator) ![]u8 {
    var child = try std.process.spawn(host_io.io, .{
        .argv = &.{ "dpkg-query", "-W", "-f=${Version}", "mesa-vulkan-drivers" },
        .stdout = .pipe,
        .stderr = .ignore,
    });
    defer child.kill(host_io.io);
    const stdout = try host_io.readToEndAlloc(child.stdout.?, allocator, 1024);
    const term = try child.wait(host_io.io);

    if (term != .exited or term.exited != 0 or stdout.len == 0) {
        allocator.free(stdout);
        return error.QueryFailed;
    }
    return stdout;
}

fn getGeccStatus(allocator: std.mem.Allocator) ![]u8 {
    const file = std.Io.Dir.openFileAbsolute(host_io.io, "/sys/module/amdgpu/parameters/ras_enable", .{}) catch return error.NotFound;
    defer {
        var close_file = file;
        close_file.close(host_io.io);
    }
    return try host_io.readToEndAlloc(file, allocator, 64);
}

test "inspectShaderAssets reports all shaders present" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    for (required_shader_files) |name| {
        try tmp.dir.writeFile(host_io.io, .{ .sub_path = name, .data = "" });
    }

    const check = inspectShaderAssets(tmp.dir);
    try std.testing.expectEqual(required_shader_files.len, check.found);
    try std.testing.expectEqual(required_shader_files.len, check.total);
    try std.testing.expect(check.first_missing == null);
}

test "inspectShaderAssets reports first missing shader" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    try tmp.dir.writeFile(host_io.io, .{ .sub_path = required_shader_files[0], .data = "" });

    const check = inspectShaderAssets(tmp.dir);
    try std.testing.expectEqual(@as(usize, 1), check.found);
    try std.testing.expectEqualStrings(required_shader_files[1], check.first_missing.?);
}

test "estimateFit includes weights, kv cache, and runtime buffers" {
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

    const fit = estimateFit(inspection, 32 * 1024 * 1024 * 1024, 4096);
    try std.testing.expectEqual(@as(u32, 4096), fit.max_ctx);
    try std.testing.expect(fit.budget_max_ctx >= fit.max_ctx);
    try std.testing.expect(fit.weights_bytes == inspection.tensor_bytes);
    try std.testing.expect(fit.kv_cache_bytes > 0);
    try std.testing.expect(fit.runtime_device_local_bytes > fit.kv_cache_bytes);
    try std.testing.expect(fit.host_visible_bytes > 0);
    try std.testing.expect(fit.total_device_local_bytes > fit.weights_bytes);
}

test "Gemma 31B diagnostics budget actual projections and every Vulkan KV layout" {
    var config: ModelConfig = std.mem.zeroes(ModelConfig);
    config.architecture = .gemma;
    config.n_layers = 60;
    config.n_heads = 32;
    config.n_kv_heads = 16;
    config.head_dim = 512;
    config.hidden_dim = 5376;
    config.intermediate_dim = 21504;
    config.vocab_size = 262144;
    config.context_length = 131072;
    const inspection = ModelInspection{
        .config = config,
        .file_size = 0,
        .tensor_bytes = 18323733440,
        .kv_dimension_sum = 50 * 4096 + 10 * 2048,
        .tensor_count = 0,
        .metadata_count = 0,
    };
    const card = @as(u64, 32) * 1024 * 1024 * 1024;
    const half = estimateFitWithKvBlockBytes(inspection, card, 8192, 64);
    const full = estimateFitWithKvBlockBytes(inspection, card, 8192, 128);
    const quantized = estimateFitWithKvBlockBytes(inspection, card, 8192, 36);
    try std.testing.expectEqual(@as(u64, 7381975040), half.kv_cache_bytes);
    try std.testing.expectEqual(half.kv_cache_bytes * 2, full.kv_cache_bytes);
    try std.testing.expectEqual(half.kv_cache_bytes / 64 * 36, quantized.kv_cache_bytes);
    try std.testing.expectEqual(inspection.tensor_bytes, half.weights_bytes);
    try std.testing.expectEqual(CheckStatus.ok, half.fitStatus());
    try std.testing.expect(half.budget_max_ctx >= 8192);
    try std.testing.expect(full.total_device_local_bytes > half.total_device_local_bytes);
    try std.testing.expect(quantized.total_device_local_bytes < half.total_device_local_bytes);
    // The reported 47 GiB came from both old assumptions together. Retain
    // conservative metadata fallback and a real oversized-context rejection.
    var unknown = inspection;
    unknown.kv_dimension_sum = null;
    try std.testing.expectEqual(CheckStatus.fail, estimateFitWithKvBlockBytes(unknown, card, 8192, 128).fitStatus());
    try std.testing.expectEqual(CheckStatus.fail, estimateFitWithKvBlockBytes(inspection, card, 32768, 64).fitStatus());
    var capped = inspection;
    capped.config.context_length = 4096;
    try std.testing.expectEqual(@as(u32, 4096), estimateFitWithKvBlockBytes(capped, card, 8192, 64).max_ctx);
    var unaligned = inspection;
    unaligned.kv_dimension_sum = 1001;
    try std.testing.expectEqual(@as(u64, 32 * 64 * 2 * 8192), estimateFitWithKvBlockBytes(unaligned, card, 8192, 64).kv_cache_bytes);
}

test "uniform diagnostics preserve KV precision when projection metadata is unavailable" {
    var config: ModelConfig = std.mem.zeroes(ModelConfig);
    config.n_layers = 4;
    config.n_heads = 8;
    config.n_kv_heads = 2;
    config.head_dim = 128;
    config.hidden_dim = 1024;
    config.context_length = 8192;
    const inspection = ModelInspection{ .config = config, .file_size = 0, .tensor_bytes = 1024, .tensor_count = 0, .metadata_count = 0 };
    for ([_]u64{ 64, 128, 36 }) |block_bytes| {
        const fit = estimateFitWithKvBlockBytes(inspection, 1024 * 1024 * 1024, 4096, block_bytes);
        try std.testing.expectEqual(@as(u64, 4 * (256 / 32) * 2 * 4096) * block_bytes, fit.kv_cache_bytes);
    }
}

test "fitStatusForBytes distinguishes fit, warning margin, and overflow" {
    try std.testing.expectEqual(CheckStatus.ok, fitStatusForBytes(4 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024));
    try std.testing.expectEqual(CheckStatus.warn, fitStatusForBytes(7 * 1024 * 1024 * 1024 + 512 * 1024 * 1024, 8 * 1024 * 1024 * 1024));
    try std.testing.expectEqual(CheckStatus.fail, fitStatusForBytes(9 * 1024 * 1024 * 1024, 8 * 1024 * 1024 * 1024));
}

test "shouldUseColor respects tty and overrides" {
    try std.testing.expect(shouldUseColor(true, "xterm-256color", false, false));
    try std.testing.expect(!shouldUseColor(false, "xterm-256color", false, false));
    try std.testing.expect(!shouldUseColor(true, "dumb", false, false));
    try std.testing.expect(!shouldUseColor(true, "xterm-256color", true, false));
    try std.testing.expect(shouldUseColor(false, "dumb", false, true));
}

test "isTruthy accepts common true values" {
    try std.testing.expect(isTruthy("1"));
    try std.testing.expect(isTruthy("true"));
    try std.testing.expect(isTruthy("yes"));
    try std.testing.expect(!isTruthy("0"));
    try std.testing.expect(!isTruthy(null));
}
