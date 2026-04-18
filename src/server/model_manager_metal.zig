//! Metal-backed active-model runtime state for the HTTP server.
//!
//! The server uses this manager to load one Metal model at a time, track the
//! tokenizer and runtime allocations that belong to it, and project fit/status
//! information back into the OpenAI-compatible model-management endpoints.
//! @section API Server
const std = @import("std");
const catalog_mod = @import("../model/catalog.zig");
const config_mod = @import("../model/config.zig");
const managed_mod = @import("../model/managed.zig");
const loader_mod = @import("../model/loader_metal.zig");
const tokenizer_mod = @import("../model/tokenizer.zig");
const forward_mod = @import("../compute/forward_metal.zig");
const memory_plan = @import("../gpu/memory_plan.zig");
const process_lock_mod = @import("../gpu/process_lock.zig");
const metal_device = @import("../metal/device.zig");

const ModelConfig = config_mod.ModelConfig;
const MetalDevice = metal_device.MetalDevice;

/// Identifies a model to load: a GGUF file path and optional managed-catalog id.
pub const LoadSpec = struct {
    model_path: []const u8,
    managed_id: ?[]const u8 = null,
    requested_context_length: ?u32 = null,
};

/// Compact view of one catalog entry for the HTTP `/v1/models` response.
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

/// Snapshot of the full model catalog, filtered by the current GPU profile.
pub const ModelCatalogView = struct {
    profile: []const u8,
    data: []ModelSummary,

    /// Free the owned catalog summary slice.
    pub fn deinit(self: *ModelCatalogView, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }
};

/// All GPU and host resources for a loaded model: weights, tokenizer, and inference engine.
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

    fn deinit(self: *LoadedResources, allocator: std.mem.Allocator) void {
        self.engine.deinit();
        self.tokenizer.deinit();
        self.model.deinit();
        allocator.free(self.model_path);
        if (self.managed_id) |id| allocator.free(id);
        allocator.free(self.display_name);
        self.* = undefined;
    }
};

/// Thread-safe manager for the currently active model on the Metal backend.
/// Handles loading, hot-swapping, catalog queries, and VRAM budget enforcement.
pub const ModelManager = struct {
    allocator: std.mem.Allocator,
    device: *const MetalDevice,
    profile: []const u8,
    vram_budget_bytes: u64,
    state_mutex: std.Thread.Mutex = .{},
    gpu_process_lock: process_lock_mod.ProcessLock = .{},
    requested_context_length: ?u32 = null,
    current: ?*LoadedResources,

    pub const RemoveResult = struct {
        unloaded_from_gpu: bool,
        cleared_active_selection: bool,
        removed: managed_mod.RemoveInstalledModelResult,
    };

    pub const MemoryUsage = struct {
        weights_bytes: u64,
        runtime_device_local_bytes: u64,
        context_reserved_bytes: u64,
        context_capacity_tokens: u32,
        context_bytes_per_token: u64,
        device_local_bytes: u64,
        device_local_budget_bytes: u64,

        /// Clamp an active-request token count to the loaded model's capacity.
        pub fn activeContextTokens(self: @This(), requested_tokens: u32) u32 {
            return @min(requested_tokens, self.context_capacity_tokens);
        }

        /// Return the bytes consumed by the clamped active context token count.
        pub fn activeContextBytes(self: @This(), requested_tokens: u32) u64 {
            return @as(u64, self.activeContextTokens(requested_tokens)) * self.context_bytes_per_token;
        }
    };

    /// Create a manager and eagerly load the requested Metal model.
    pub fn init(
        spec: LoadSpec,
        device: *const MetalDevice,
        allocator: std.mem.Allocator,
    ) !ModelManager {
        var gpu_process_lock = try process_lock_mod.acquire(.metal, device.selected_device_index);
        errdefer gpu_process_lock.deinit();
        const current = try allocator.create(LoadedResources);
        errdefer allocator.destroy(current);
        try loadResourcesInto(current, spec, device, allocator);
        return .{
            .allocator = allocator,
            .device = device,
            .profile = catalog_mod.profileForMetal(),
            .vram_budget_bytes = memoryBudget(device),
            .gpu_process_lock = gpu_process_lock,
            .requested_context_length = spec.requested_context_length,
            .current = current,
        };
    }

    /// Create an idle manager with no model currently loaded.
    pub fn initEmpty(
        device: *const MetalDevice,
        requested_context_length: ?u32,
        allocator: std.mem.Allocator,
    ) ModelManager {
        return .{
            .allocator = allocator,
            .device = device,
            .profile = catalog_mod.profileForMetal(),
            .vram_budget_bytes = memoryBudget(device),
            .requested_context_length = requested_context_length,
            .current = null,
        };
    }

    /// Tear down the active model, if any, and release the GPU process lock.
    pub fn deinit(self: *ModelManager) void {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();
        if (self.current) |current| {
            current.deinit(self.allocator);
            self.allocator.destroy(current);
        }
        self.gpu_process_lock.deinit();
    }

    /// Return the currently loaded resource bundle, or null when idle.
    pub fn currentResources(self: *ModelManager) ?*LoadedResources {
        return self.current;
    }

    /// Return the active model display name, or `"none"` when no model is loaded.
    pub fn activeDisplayName(self: *ModelManager) []const u8 {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();
        return if (self.current) |current| current.display_name else "none";
    }

    /// Return the catalog profile string used for the active Metal device.
    pub fn catalogProfile(self: *const ModelManager) []const u8 {
        return self.profile;
    }

    /// Snapshot memory usage for the active model, or zeroes when idle.
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

    /// Build a catalog view annotated with install, fit, and active-model status.
    pub fn collectCatalogView(self: *ModelManager, allocator: std.mem.Allocator, include_all: bool) !ModelCatalogView {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();

        const active_catalog_entry = if (self.current) |current|
            catalog_mod.findForLoadedModel(current.managed_id, current.model_path, current.display_name)
        else
            null;
        const active_catalog_id = if (active_catalog_entry) |entry| entry.id else null;
        const active_display_name = if (self.current) |current| current.display_name else "none";
        const active_supports_thinking_toggle = if (self.current) |current| current.tokenizer.supportsThinkingToggle() else false;

        var list: std.ArrayList(ModelSummary) = .{};
        defer list.deinit(allocator);

        for (catalog_mod.entries) |entry| {
            const tested_profile_match = catalog_mod.supportsProfile(entry, self.profile);
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
                .active = active_catalog_id != null and std.mem.eql(u8, active_catalog_id.?, entry.id),
                .managed = true,
                .supported_on_current_gpu = supported_now,
                .fits_current_gpu = fit.fits_current_gpu,
                .required_vram_bytes = fit.required_vram_bytes,
                .size_bytes = entry.size_bytes,
                .exact_fit = fit.exact,
                .status_label = status_label,
                .supports_thinking_toggle = active_catalog_id != null and std.mem.eql(u8, active_catalog_id.?, entry.id) and active_supports_thinking_toggle and entry.thinking_stable,
            });
        }

        if (self.current != null and active_catalog_id == null) {
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
            .profile = self.profile,
            .data = try list.toOwnedSlice(allocator),
        };
    }

    /// Return whether a managed catalog entry is supported and fits on the active device.
    pub fn supportsManagedEntry(self: *ModelManager, entry: catalog_mod.CatalogEntry, allocator: std.mem.Allocator) bool {
        const fit = managed_mod.describeFit(entry, self.vram_budget_bytes, allocator) catch managed_mod.ModelFit{
            .required_vram_bytes = entry.required_vram_bytes,
            .fits_current_gpu = catalog_mod.fitsGpu(entry, self.vram_budget_bytes),
            .exact = false,
        };
        return catalog_mod.supportsProfile(entry, self.profile) and fit.fits_current_gpu;
    }

    /// Caller must already hold the shared generation lock.
    pub fn activateManagedModel(self: *ModelManager, model_id: []const u8, persist_active: bool) !void {
        const entry = catalog_mod.find(model_id) orelse return error.UnknownManagedModel;
        if (!catalog_mod.supportsProfile(entry.*, self.profile)) return error.ModelUnsupportedOnThisGpu;
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
            self.gpu_process_lock = try process_lock_mod.acquire(.metal, self.device.selected_device_index);
            acquired_gpu_lock = true;
            errdefer if (acquired_gpu_lock) self.gpu_process_lock.deinit();
        }
        const switched = try self.allocator.create(LoadedResources);
        errdefer self.allocator.destroy(switched);
        try loadResourcesInto(switched, .{
            .model_path = new_path,
            .managed_id = model_id,
            .requested_context_length = self.requested_context_length,
        }, self.device, self.allocator);

        const previous = self.current;
        self.current = switched;
        acquired_gpu_lock = false;
        if (previous) |old| {
            old.deinit(self.allocator);
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
            resources.deinit(self.allocator);
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
    device: *const MetalDevice,
    allocator: std.mem.Allocator,
) !void {
    resources.* = undefined;

    resources.model = try loader_mod.load(spec.model_path, device.ctx, allocator);
    errdefer resources.model.deinit();
    memory_plan.applyRequestedContextLimit(&resources.model.config, spec.requested_context_length);

    resources.tokenizer = try tokenizer_mod.Tokenizer.initFromGGUF(&resources.model.gguf_file, allocator);
    errdefer resources.tokenizer.deinit();

    resources.engine = try forward_mod.InferenceEngine.init(&resources.model, device, allocator, .{});
    errdefer resources.engine.deinit();

    resources.model_path = try allocator.dupe(u8, spec.model_path);
    errdefer allocator.free(resources.model_path);

    resources.managed_id = if (spec.managed_id) |id| try allocator.dupe(u8, id) else null;
    errdefer if (resources.managed_id) |id| allocator.free(id);

    resources.display_name = try allocator.dupe(u8, modelDisplayName(&resources.model));
    errdefer allocator.free(resources.display_name);

    const weights_bytes = tensorBytes(&resources.model);
    const usage = estimateMemoryUsage(
        resources.model.config,
        weights_bytes,
        memoryBudget(device),
        resources.engine.max_context_tokens,
    );
    resources.weights_bytes = usage.weights_bytes;
    resources.runtime_device_local_bytes = usage.runtime_device_local_bytes;
    resources.context_reserved_bytes = usage.context_reserved_bytes;
    resources.context_capacity_tokens = usage.context_capacity_tokens;
    resources.context_bytes_per_token = usage.context_bytes_per_token;
    resources.device_local_bytes = usage.device_local_bytes;
    resources.device_local_budget_bytes = usage.device_local_budget_bytes;
}

fn tensorBytes(model: *const loader_mod.Model) u64 {
    var total: u64 = 0;
    for (model.gguf_file.tensors.items) |tensor_info| {
        total += tensor_info.sizeBytes();
    }
    return total;
}

fn memoryBudget(device: *const MetalDevice) u64 {
    const working_set = device.recommendedMaxWorkingSetSize();
    if (working_set > 0) return working_set;
    return device.totalMemory();
}

fn estimateMemoryUsage(config: ModelConfig, weights_bytes: u64, budget_bytes: u64, runtime_ctx: u32) ModelManager.MemoryUsage {
    const profile = memory_plan.profile(config);
    const kv_cache_bytes = profile.deviceLocalContextBytes(runtime_ctx);
    const runtime_device_local_bytes = profile.runtimeDeviceLocalBytes(runtime_ctx);

    return .{
        .weights_bytes = weights_bytes,
        .runtime_device_local_bytes = runtime_device_local_bytes,
        .context_reserved_bytes = kv_cache_bytes,
        .context_capacity_tokens = runtime_ctx,
        .context_bytes_per_token = if (runtime_ctx == 0) 0 else @divTrunc(kv_cache_bytes, runtime_ctx),
        .device_local_bytes = weights_bytes + runtime_device_local_bytes,
        .device_local_budget_bytes = budget_bytes,
    };
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

test "MemoryUsage active context bytes clamp to capacity" {
    const usage = ModelManager.MemoryUsage{
        .weights_bytes = 0,
        .runtime_device_local_bytes = 0,
        .context_reserved_bytes = 1024,
        .context_capacity_tokens = 4,
        .context_bytes_per_token = 256,
        .device_local_bytes = 0,
        .device_local_budget_bytes = 0,
    };
    try std.testing.expectEqual(@as(u32, 4), usage.activeContextTokens(99));
    try std.testing.expectEqual(@as(u64, 1024), usage.activeContextBytes(99));
}

test "estimateMemoryUsage follows the reserved runtime context" {
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

    const usage = estimateMemoryUsage(config, 21 * 1024 * 1024 * 1024, 48 * 1024 * 1024 * 1024, 2048);
    try std.testing.expectEqual(@as(u32, 2048), usage.context_capacity_tokens);
    try std.testing.expect(usage.context_reserved_bytes > 0);
    try std.testing.expect(usage.runtime_device_local_bytes > usage.context_reserved_bytes);
}

test "collectCatalogView exposes validated qwen thinking toggles on Metal" {
    var fake = ModelManager{
        .allocator = std.testing.allocator,
        .device = undefined,
        .profile = catalog_mod.profileForMetal(),
        .vram_budget_bytes = 64 * 1024 * 1024 * 1024,
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
        .model_path = try std.testing.allocator.dupe(u8, "/tmp/model.gguf"),
        .managed_id = try std.testing.allocator.dupe(u8, "qwen35-35b-a3b-q4k-xl"),
        .display_name = try std.testing.allocator.dupe(u8, "Qwen3.5 35B-A3B UD Q4_K_XL"),
        .weights_bytes = 0,
        .runtime_device_local_bytes = 0,
        .context_reserved_bytes = 0,
        .context_capacity_tokens = 0,
        .context_bytes_per_token = 0,
        .device_local_bytes = 0,
        .device_local_budget_bytes = 64 * 1024 * 1024 * 1024,
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

    var saw_active = false;
    for (view.data) |entry| {
        if (std.mem.eql(u8, entry.id, "qwen35-35b-a3b-q4k-xl")) {
            saw_active = true;
            try std.testing.expect(entry.active);
            try std.testing.expect(entry.supports_thinking_toggle);
        }
    }
    try std.testing.expect(saw_active);
}

test "collectCatalogView preserves qwen36 thinking toggle for raw matched path on Metal" {
    var fake = ModelManager{
        .allocator = std.testing.allocator,
        .device = undefined,
        .profile = catalog_mod.profileForMetal(),
        .vram_budget_bytes = 64 * 1024 * 1024 * 1024,
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
        .model_path = try std.testing.allocator.dupe(u8, "/Users/test/Library/Caches/zinc/models/models/qwen36-35b-a3b-q4k-xl/model.gguf"),
        .managed_id = null,
        .display_name = try std.testing.allocator.dupe(u8, "Qwen3.6-35B-A3B-UD-Q4_K_XL"),
        .weights_bytes = 0,
        .runtime_device_local_bytes = 0,
        .context_reserved_bytes = 0,
        .context_capacity_tokens = 0,
        .context_bytes_per_token = 0,
        .device_local_bytes = 0,
        .device_local_budget_bytes = 64 * 1024 * 1024 * 1024,
    };
    fake.current = &current;
    defer {
        if (fake.current) |loaded| {
            loaded.tokenizer.token_to_id.deinit();
            std.testing.allocator.free(loaded.model_path);
            std.testing.allocator.free(loaded.display_name);
        }
    }

    var view = try fake.collectCatalogView(std.testing.allocator, false);
    defer view.deinit(std.testing.allocator);

    var qwen36_active = false;
    for (view.data) |entry| {
        if (std.mem.eql(u8, entry.id, "qwen36-35b-a3b-q4k-xl")) {
            qwen36_active = true;
            try std.testing.expect(entry.active);
            try std.testing.expect(entry.supports_thinking_toggle);
        }
        try std.testing.expect(!std.mem.eql(u8, entry.id, "Qwen3.6-35B-A3B-UD-Q4_K_XL"));
    }
    try std.testing.expect(qwen36_active);
}
