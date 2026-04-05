//! Managed active-model runtime state for the HTTP server and CLI startup.
//! @section Managed Models
//! ZINC still loads one model into memory at a time. This manager keeps the
//! current engine/tokenizer/model bundle together and handles serialized swaps.
const std = @import("std");
const loader_mod = @import("../model/loader.zig");
const tokenizer_mod = @import("../model/tokenizer.zig");
const catalog_mod = @import("../model/catalog.zig");
const managed_mod = @import("../model/managed.zig");
const diagnostics_mod = @import("../diagnostics.zig");
const forward_mod = @import("../compute/forward.zig");
const process_lock_mod = @import("../gpu/process_lock.zig");
const gpu_detect = @import("../vulkan/gpu_detect.zig");
const instance_mod = @import("../vulkan/instance.zig");
const CommandPool = @import("../vulkan/command.zig").CommandPool;

const Instance = instance_mod.Instance;

/// Describes which model to load: a filesystem path and an optional managed-catalog ID.
pub const LoadSpec = struct {
    model_path: []const u8,
    managed_id: ?[]const u8 = null,
};

/// Flat representation of a catalog model for JSON serialization to API clients.
pub const ModelSummary = struct {
    id: []const u8,
    display_name: []const u8,
    release_date: []const u8,
    homepage_url: []const u8,
    family: []const u8,
    quantization: []const u8,
    installed: bool,
    active: bool,
    managed: bool,
    supported_on_current_gpu: bool,
    fits_current_gpu: bool,
    required_vram_bytes: u64,
    size_bytes: u64,
    exact_fit: bool,
    status_label: []const u8,
    supports_thinking_toggle: bool,
};

/// Snapshot of the full model catalog annotated with the current GPU profile.
pub const ModelCatalogView = struct {
    profile: []const u8,
    data: []ModelSummary,

    /// Frees the owned summary slice.
    pub fn deinit(self: *ModelCatalogView, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }
};

/// Bundle of model, tokenizer, and inference engine that represents a fully loaded model.
pub const LoadedResources = struct {
    model: loader_mod.Model,
    tokenizer: tokenizer_mod.Tokenizer,
    engine: forward_mod.InferenceEngine,
    model_path: []u8,
    managed_id: ?[]u8,
    display_name: []u8,
    weights_bytes: u64,
    runtime_device_local_bytes: u64,
    context_reserved_bytes: u64,
    context_capacity_tokens: u32,
    context_bytes_per_token: u64,
    device_local_bytes: u64,
    device_local_budget_bytes: u64,

    fn deinit(self: *LoadedResources, instance: *const Instance, allocator: std.mem.Allocator) void {
        self.engine.deinit();
        self.tokenizer.deinit();
        self.model.deinit(instance);
        allocator.free(self.model_path);
        if (self.managed_id) |id| allocator.free(id);
        allocator.free(self.display_name);
        self.* = undefined;
    }
};

/// Thread-safe owner of the currently active model, providing load, swap, and catalog queries.
pub const ModelManager = struct {
    allocator: std.mem.Allocator,
    instance: *const Instance,
    gpu_config: gpu_detect.GpuConfig,
    vram_budget_bytes: u64,
    shader_dir: []const u8,
    state_mutex: std.Thread.Mutex = .{},
    gpu_process_lock: process_lock_mod.ProcessLock = .{},
    current: ?*LoadedResources,

    /// Outcome of a managed model removal, including whether it was unloaded from the GPU.
    pub const RemoveResult = struct {
        unloaded_from_gpu: bool,
        cleared_active_selection: bool,
        removed: managed_mod.RemoveInstalledModelResult,
    };

    /// Creates a manager and immediately loads the model described by `spec`.
    pub fn init(
        spec: LoadSpec,
        instance: *const Instance,
        gpu_config_value: gpu_detect.GpuConfig,
        shader_dir: []const u8,
        allocator: std.mem.Allocator,
    ) !ModelManager {
        var gpu_process_lock = try process_lock_mod.acquire(.vulkan, instance.selected_device_index);
        errdefer gpu_process_lock.deinit();
        const current = try allocator.create(LoadedResources);
        errdefer allocator.destroy(current);
        try loadResourcesInto(current, spec, instance, gpu_config_value, shader_dir, allocator);
        return .{
            .allocator = allocator,
            .instance = instance,
            .gpu_config = gpu_config_value,
            .vram_budget_bytes = instance.vramBytes(),
            .shader_dir = shader_dir,
            .gpu_process_lock = gpu_process_lock,
            .current = current,
        };
    }

    /// Creates a manager with no model loaded (server starts idle).
    pub fn initEmpty(
        instance: *const Instance,
        gpu_config_value: gpu_detect.GpuConfig,
        shader_dir: []const u8,
        allocator: std.mem.Allocator,
    ) ModelManager {
        return .{
            .allocator = allocator,
            .instance = instance,
            .gpu_config = gpu_config_value,
            .vram_budget_bytes = instance.vramBytes(),
            .shader_dir = shader_dir,
            .current = null,
        };
    }

    /// Tears down the loaded model (if any) and releases all owned resources.
    pub fn deinit(self: *ModelManager) void {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();
        if (self.current) |current| {
            current.deinit(self.instance, self.allocator);
            self.allocator.destroy(current);
        }
        self.gpu_process_lock.deinit();
    }

    /// Returns a pointer to the active model resources, or null if none is loaded.
    pub fn currentResources(self: *ModelManager) ?*LoadedResources {
        return self.current;
    }

    /// Returns the human-readable name of the active model, or `"none"`.
    pub fn activeDisplayName(self: *ModelManager) []const u8 {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();
        return if (self.current) |current| current.display_name else "none";
    }

    /// Returns the catalog profile string for the detected GPU (e.g. `"amd_rdna4"`).
    pub fn catalogProfile(self: *const ModelManager) []const u8 {
        return catalog_mod.profileForGpu(self.gpu_config);
    }

    /// VRAM accounting breakdown for the currently loaded model.
    pub const MemoryUsage = struct {
        weights_bytes: u64,
        runtime_device_local_bytes: u64,
        context_reserved_bytes: u64,
        context_capacity_tokens: u32,
        context_bytes_per_token: u64,
        device_local_bytes: u64,
        device_local_budget_bytes: u64,

        /// Returns the effective context length, clamped to the available capacity.
        pub fn activeContextTokens(self: @This(), requested_tokens: u32) u32 {
            return @min(requested_tokens, self.context_capacity_tokens);
        }

        /// Returns the VRAM bytes required for the effective context length.
        pub fn activeContextBytes(self: @This(), requested_tokens: u32) u64 {
            return @as(u64, self.activeContextTokens(requested_tokens)) * self.context_bytes_per_token;
        }
    };

    /// Snapshots the VRAM usage of the active model, or returns zeroes if idle.
    pub fn currentMemoryUsage(self: *ModelManager) MemoryUsage {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();
        if (self.current) |current| {
            return .{
                .weights_bytes = current.weights_bytes,
                .runtime_device_local_bytes = current.runtime_device_local_bytes,
                .context_reserved_bytes = current.context_reserved_bytes,
                .context_capacity_tokens = current.context_capacity_tokens,
                .context_bytes_per_token = current.context_bytes_per_token,
                .device_local_bytes = current.device_local_bytes,
                .device_local_budget_bytes = current.device_local_budget_bytes,
            };
        }
        return .{
            .weights_bytes = 0,
            .runtime_device_local_bytes = 0,
            .context_reserved_bytes = 0,
            .context_capacity_tokens = 0,
            .context_bytes_per_token = 0,
            .device_local_bytes = 0,
            .device_local_budget_bytes = self.vram_budget_bytes,
        };
    }

    /// Builds a catalog snapshot with install/active/fit status for every entry.
    /// When `include_all` is false, entries unsupported on the current GPU are excluded.
    pub fn collectCatalogView(self: *ModelManager, allocator: std.mem.Allocator, include_all: bool) !ModelCatalogView {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();

        const profile = self.catalogProfile();
        const active_managed_id = if (self.current) |current| current.managed_id else null;
        const active_display_name = if (self.current) |current| current.display_name else "none";
        const active_supports_thinking_toggle = if (self.current) |current| current.tokenizer.supportsThinkingToggle() else false;

        var list: std.ArrayList(ModelSummary) = .{};
        defer list.deinit(allocator);

        for (catalog_mod.entries) |entry| {
            const tested_profile_match = catalog_mod.supportsProfile(entry, profile);

            const installed = managed_mod.isInstalled(entry.id, allocator);
            const fit = managed_mod.describeFit(entry, self.vram_budget_bytes, allocator) catch managed_mod.ModelFit{
                .required_vram_bytes = entry.required_vram_bytes,
                .fits_current_gpu = catalog_mod.fitsGpu(entry, self.vram_budget_bytes),
                .exact = false,
            };
            const supported_now = tested_profile_match and fit.fits_current_gpu;
            if (!include_all and !supported_now) continue;

            const status_label = if (supported_now)
                @tagName(entry.status)
            else if (tested_profile_match)
                "too-large"
            else
                "hidden";

            try list.append(allocator, .{
                .id = entry.id,
                .display_name = entry.display_name,
                .release_date = entry.release_date,
                .homepage_url = entry.homepage_url,
                .family = entry.family,
                .quantization = entry.quantization,
                .installed = installed,
                .active = active_managed_id != null and std.mem.eql(u8, active_managed_id.?, entry.id),
                .managed = true,
                .supported_on_current_gpu = supported_now,
                .fits_current_gpu = fit.fits_current_gpu,
                .required_vram_bytes = fit.required_vram_bytes,
                .size_bytes = entry.size_bytes,
                .exact_fit = fit.exact,
                .status_label = status_label,
                .supports_thinking_toggle = active_managed_id != null and std.mem.eql(u8, active_managed_id.?, entry.id) and active_supports_thinking_toggle and entry.thinking_stable,
            });
        }

        if (self.current != null and active_managed_id == null) {
            try list.append(allocator, .{
                .id = active_display_name,
                .display_name = active_display_name,
                .release_date = "",
                .homepage_url = "",
                .family = "",
                .quantization = "",
                .installed = true,
                .active = true,
                .managed = false,
                .supported_on_current_gpu = true,
                .fits_current_gpu = true,
                .required_vram_bytes = 0,
                .size_bytes = 0,
                .exact_fit = true,
                .status_label = "raw",
                .supports_thinking_toggle = active_supports_thinking_toggle,
            });
        }

        return .{
            .profile = profile,
            .data = try list.toOwnedSlice(allocator),
        };
    }

    /// Returns true if the given catalog entry is compatible with and fits the current GPU.
    pub fn supportsManagedEntry(self: *ModelManager, entry: catalog_mod.CatalogEntry, allocator: std.mem.Allocator) bool {
        const fit = managed_mod.describeFit(entry, self.vram_budget_bytes, allocator) catch managed_mod.ModelFit{
            .required_vram_bytes = entry.required_vram_bytes,
            .fits_current_gpu = catalog_mod.fitsGpu(entry, self.vram_budget_bytes),
            .exact = false,
        };
        return catalog_mod.supportsProfile(entry, self.catalogProfile()) and fit.fits_current_gpu;
    }

    /// Caller must already hold the shared generation lock.
    pub fn activateManagedModel(self: *ModelManager, model_id: []const u8, persist_active: bool) !void {
        const entry = catalog_mod.find(model_id) orelse return error.UnknownManagedModel;
        if (!catalog_mod.supportsProfile(entry.*, self.catalogProfile())) return error.ModelUnsupportedOnThisGpu;
        if (!managed_mod.isInstalled(model_id, self.allocator)) return error.ModelNotInstalled;

        const fit = try managed_mod.verifyActiveSelectionFits(model_id, self.vram_budget_bytes, self.allocator);
        if (!fit.fits_current_gpu) return error.ModelDoesNotFit;

        self.state_mutex.lock();
        defer self.state_mutex.unlock();

        if (self.current) |current| {
            if (current.managed_id) |active_id| {
                if (std.mem.eql(u8, active_id, model_id)) {
                    if (persist_active) try managed_mod.writeActiveSelection(model_id, self.allocator);
                    return;
                }
            }
        }

        const new_path = try managed_mod.resolveInstalledModelPath(model_id, self.allocator);
        defer self.allocator.free(new_path);
        var acquired_gpu_lock = false;
        if (!self.gpu_process_lock.isHeld()) {
            self.gpu_process_lock = try process_lock_mod.acquire(.vulkan, self.instance.selected_device_index);
            acquired_gpu_lock = true;
            errdefer if (acquired_gpu_lock) self.gpu_process_lock.deinit();
        }
        const switched = try self.allocator.create(LoadedResources);
        errdefer self.allocator.destroy(switched);
        loadResourcesInto(switched, .{ .model_path = new_path, .managed_id = model_id }, self.instance, self.gpu_config, self.shader_dir, self.allocator) catch |switch_err| {
            return switch_err;
        };

        const previous = self.current;
        self.current = switched;
        acquired_gpu_lock = false;
        if (previous) |old| {
            old.deinit(self.instance, self.allocator);
            self.allocator.destroy(old);
        }

        if (persist_active) try managed_mod.writeActiveSelection(model_id, self.allocator);
    }

    /// Caller must already hold the shared generation lock.
    pub fn removeManagedModel(self: *ModelManager, model_id: []const u8, force: bool) !RemoveResult {
        var unloaded_from_gpu = false;
        var previous: ?*LoadedResources = null;

        self.state_mutex.lock();
        if (self.current) |current| {
            if (current.managed_id) |active_id| {
                if (std.mem.eql(u8, active_id, model_id)) {
                    if (!force) {
                        self.state_mutex.unlock();
                        return error.ModelLoadedInGpu;
                    }
                    previous = current;
                    self.current = null;
                    unloaded_from_gpu = true;
                }
            }
        }
        self.state_mutex.unlock();

        if (previous) |resources| {
            resources.deinit(self.instance, self.allocator);
            self.allocator.destroy(resources);
        }
        if (unloaded_from_gpu) {
            self.gpu_process_lock.deinit();
        }

        const removed = try managed_mod.removeInstalledModel(model_id, self.allocator);
        const cleared_active_selection = try managed_mod.clearActiveSelectionIfMatches(model_id, self.allocator);

        return .{
            .unloaded_from_gpu = unloaded_from_gpu,
            .cleared_active_selection = cleared_active_selection,
            .removed = removed,
        };
    }
};

fn loadResourcesInto(
    resources: *LoadedResources,
    spec: LoadSpec,
    instance: *const Instance,
    gpu_config_value: gpu_detect.GpuConfig,
    shader_dir: []const u8,
    allocator: std.mem.Allocator,
) !void {
    var cmd_pool = try CommandPool.init(instance);
    defer cmd_pool.deinit();

    resources.* = undefined;
    const inspection = try loader_mod.inspectModel(spec.model_path, allocator);
    const fit = diagnostics_mod.estimateFit(inspection, instance.vramBytes());

    resources.model = try loader_mod.load(spec.model_path, instance, &cmd_pool, allocator);
    errdefer resources.model.deinit(instance);

    resources.tokenizer = try tokenizer_mod.Tokenizer.initFromGGUF(&resources.model.gguf_file, allocator);
    errdefer resources.tokenizer.deinit();

    // Important: the engine stores a Model pointer. Initialize it against the
    // stable model field inside the final LoadedResources storage.
    resources.engine = try forward_mod.InferenceEngine.init(&resources.model, instance, gpu_config_value, shader_dir, allocator);
    errdefer resources.engine.deinit();

    resources.model_path = try allocator.dupe(u8, spec.model_path);
    errdefer allocator.free(resources.model_path);

    resources.managed_id = if (spec.managed_id) |id| try allocator.dupe(u8, id) else null;
    errdefer if (resources.managed_id) |id| allocator.free(id);

    resources.display_name = try allocator.dupe(u8, modelDisplayName(&resources.model));
    errdefer allocator.free(resources.display_name);
    resources.weights_bytes = inspection.tensor_bytes;
    resources.runtime_device_local_bytes = fit.runtime_device_local_bytes;
    resources.context_reserved_bytes = fit.kv_cache_bytes;
    resources.context_capacity_tokens = fit.max_ctx;
    resources.context_bytes_per_token = if (fit.max_ctx == 0) 0 else @divTrunc(fit.kv_cache_bytes, fit.max_ctx);
    resources.device_local_bytes = fit.total_device_local_bytes;
    resources.device_local_budget_bytes = fit.vram_budget_bytes;

    std.debug.assert(resources.engine.model == &resources.model);
}

fn fallbackModelName(model: *const loader_mod.Model) []const u8 {
    return switch (model.config.architecture) {
        .qwen35 => "qwen3.5",
        .qwen2_moe => "qwen3.5-35b",
        .qwen2 => "qwen2",
        .mistral => "mistral",
        .mamba => "mamba",
        .jamba => "jamba",
        .gemma => "gemma",
        .gpt_oss => "gpt-oss-20b",
        .unknown => "zinc-model",
    };
}

fn modelDisplayName(model: *const loader_mod.Model) []const u8 {
    return model.gguf_file.getString("general.basename") orelse
        model.gguf_file.getString("general.name") orelse
        fallbackModelName(model);
}

test "collectCatalogView marks active managed model" {
    var fake = ModelManager{
        .allocator = std.testing.allocator,
        .instance = undefined,
        .gpu_config = .{
            .vendor = .amd_rdna4,
            .device_name = undefined,
            .device_name_len = 0,
            .vram_mb = 32624,
            .bandwidth_gbps = 576,
            .compute_units = 64,
            .wave_size = 64,
            .coopmat_support = true,
            .l1_cache_kb = 32,
            .l2_cache_mb = 6,
            .max_workgroup_size = 1024,
            .dmmv_workgroup_size = 64,
            .dmmv_rows_per_workgroup = 2,
            .matmul_tile_m = 16,
            .matmul_tile_n = 16,
            .flash_attn_block_size = 256,
        },
        .vram_budget_bytes = 32 * 1024 * 1024 * 1024,
        .shader_dir = "zig-out/share/zinc/shaders",
        .current = undefined,
    };
    var current = LoadedResources{
        .model = undefined,
        .tokenizer = .{
            .vocab = &.{},
            .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
            .merges = &.{},
            .scores = null,
            .bos_id = 1,
            .eos_id = 2,
            .prepend_bos = true,
            .chat_template =
            \\{%- if add_generation_prompt %}
            \\  {{- '<|im_start|>assistant\n' }}
            \\  {%- if enable_thinking is defined and enable_thinking is true %}
            \\    {{- '<think>\n' }}
            \\  {%- else %}
            \\    {{- '<think>\n\n</think>\n\n' }}
            \\  {%- endif %}
            \\{%- endif %}
            ,
            .allocator = std.testing.allocator,
        },
        .engine = undefined,
        .model_path = try std.testing.allocator.dupe(u8, "/tmp/test.gguf"),
        .managed_id = try std.testing.allocator.dupe(u8, "qwen35-35b-a3b-q4k-xl"),
        .display_name = try std.testing.allocator.dupe(u8, "Qwen3.5 35B-A3B UD Q4_K_XL"),
        .weights_bytes = 20 * 1024 * 1024 * 1024,
        .runtime_device_local_bytes = 1024 * 1024 * 1024,
        .context_reserved_bytes = 768 * 1024 * 1024,
        .context_capacity_tokens = 4096,
        .context_bytes_per_token = 192 * 1024,
        .device_local_bytes = 21 * 1024 * 1024 * 1024,
        .device_local_budget_bytes = 32 * 1024 * 1024 * 1024,
    };
    fake.current = &current;
    defer {
        if (fake.current) |loaded| {
            loaded.tokenizer.token_to_id.deinit();
            std.testing.allocator.free(loaded.model_path);
            if (loaded.managed_id) |id| std.testing.allocator.free(id);
            std.testing.allocator.free(loaded.display_name);
        }
    }

    var view = try fake.collectCatalogView(std.testing.allocator, false);
    defer view.deinit(std.testing.allocator);

    try std.testing.expect(view.data.len >= 1);
    var saw_active = false;
    for (view.data) |entry| {
        if (std.mem.eql(u8, entry.id, "qwen35-35b-a3b-q4k-xl")) {
            saw_active = true;
            try std.testing.expect(entry.active);
            try std.testing.expect(entry.supports_thinking_toggle);
            try std.testing.expectEqualStrings("2026-02-16", entry.release_date);
        }
    }
    try std.testing.expect(saw_active);
}

test "currentMemoryUsage reports empty state when no model is loaded" {
    var fake = ModelManager{
        .allocator = std.testing.allocator,
        .instance = undefined,
        .gpu_config = .{
            .vendor = .amd_rdna4,
            .device_name = undefined,
            .device_name_len = 0,
            .vram_mb = 32624,
            .bandwidth_gbps = 576,
            .compute_units = 64,
            .wave_size = 64,
            .coopmat_support = true,
            .l1_cache_kb = 32,
            .l2_cache_mb = 6,
            .max_workgroup_size = 1024,
            .dmmv_workgroup_size = 64,
            .dmmv_rows_per_workgroup = 2,
            .matmul_tile_m = 16,
            .matmul_tile_n = 16,
            .flash_attn_block_size = 256,
        },
        .vram_budget_bytes = 32 * 1024 * 1024 * 1024,
        .shader_dir = "zig-out/share/zinc/shaders",
        .current = null,
    };

    const usage = fake.currentMemoryUsage();
    try std.testing.expectEqual(@as(u64, 0), usage.weights_bytes);
    try std.testing.expectEqual(@as(u64, 0), usage.runtime_device_local_bytes);
    try std.testing.expectEqual(@as(u64, 0), usage.context_reserved_bytes);
    try std.testing.expectEqual(@as(u32, 0), usage.context_capacity_tokens);
    try std.testing.expectEqual(@as(u64, 0), usage.context_bytes_per_token);
    try std.testing.expectEqual(@as(u64, 0), usage.device_local_bytes);
    try std.testing.expectEqual(@as(u64, 32 * 1024 * 1024 * 1024), usage.device_local_budget_bytes);
    try std.testing.expectEqualStrings("none", fake.activeDisplayName());
}

test "removeManagedModel refuses loaded active model without force" {
    var fake = ModelManager{
        .allocator = std.testing.allocator,
        .instance = undefined,
        .gpu_config = .{
            .vendor = .amd_rdna4,
            .device_name = undefined,
            .device_name_len = 0,
            .vram_mb = 32624,
            .bandwidth_gbps = 576,
            .compute_units = 64,
            .wave_size = 64,
            .coopmat_support = true,
            .l1_cache_kb = 32,
            .l2_cache_mb = 6,
            .max_workgroup_size = 1024,
            .dmmv_workgroup_size = 64,
            .dmmv_rows_per_workgroup = 2,
            .matmul_tile_m = 16,
            .matmul_tile_n = 16,
            .flash_attn_block_size = 256,
        },
        .vram_budget_bytes = 32 * 1024 * 1024 * 1024,
        .shader_dir = "zig-out/share/zinc/shaders",
        .current = undefined,
    };
    var current = LoadedResources{
        .model = undefined,
        .tokenizer = .{
            .vocab = &.{},
            .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
            .merges = &.{},
            .scores = null,
            .bos_id = 1,
            .eos_id = 2,
            .prepend_bos = true,
            .chat_template = null,
            .allocator = std.testing.allocator,
        },
        .engine = undefined,
        .model_path = try std.testing.allocator.dupe(u8, "/tmp/test.gguf"),
        .managed_id = try std.testing.allocator.dupe(u8, "qwen35-35b-a3b-q4k-xl"),
        .display_name = try std.testing.allocator.dupe(u8, "Qwen3.5 35B-A3B UD Q4_K_XL"),
        .weights_bytes = 20 * 1024 * 1024 * 1024,
        .runtime_device_local_bytes = 1024 * 1024 * 1024,
        .context_reserved_bytes = 768 * 1024 * 1024,
        .context_capacity_tokens = 4096,
        .context_bytes_per_token = 192 * 1024,
        .device_local_bytes = 21 * 1024 * 1024 * 1024,
        .device_local_budget_bytes = 32 * 1024 * 1024 * 1024,
    };
    fake.current = &current;
    defer {
        fake.current.?.tokenizer.token_to_id.deinit();
        std.testing.allocator.free(fake.current.?.model_path);
        if (fake.current.?.managed_id) |id| std.testing.allocator.free(id);
        std.testing.allocator.free(fake.current.?.display_name);
    }

    try std.testing.expectError(error.ModelLoadedInGpu, fake.removeManagedModel("qwen35-35b-a3b-q4k-xl", false));
    try std.testing.expect(fake.current == &current);
}
