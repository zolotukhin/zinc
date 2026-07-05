//! Managed active-model runtime state for the HTTP server and CLI startup.
//! @section Managed Models
//! ZINC still loads one model into memory at a time. This manager keeps the
//! current engine/tokenizer/model bundle together and handles serialized swaps.
const std = @import("std");
const loader_mod = @import("../model/loader.zig");
const tokenizer_mod = @import("../model/tokenizer.zig");
const catalog_mod = @import("../model/catalog.zig");
const managed_mod = @import("../model/managed.zig");
const forward_mod = @import("../compute/forward.zig");
const memory_plan = @import("../gpu/memory_plan.zig");
const process_lock_mod = @import("../gpu/process_lock.zig");
const gpu_detect = @import("../vulkan/gpu_detect.zig");
const instance_mod = @import("../vulkan/instance.zig");
const CommandPool = @import("../vulkan/command.zig").CommandPool;

const Instance = instance_mod.Instance;

/// Describes which model to load: a filesystem path, an optional managed-catalog ID, and an optional context-length override.
pub const LoadSpec = struct {
    model_path: []const u8,
    managed_id: ?[]const u8 = null,
    requested_context_length: ?u32 = null,
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
    /// VRAM required when MoE expert tensors are offloaded to host RAM under
    /// `ZINC_OFFLOAD_MOE_EXPERTS=1`. Equal to required_vram_bytes for dense
    /// models. Lets clients show users that a too-large model would fit with
    /// the offload escape hatch.
    required_vram_with_offload_bytes: u64,
    /// True when the model doesn't fit in VRAM but does fit if MoE expert
    /// tensors are offloaded to host RAM. Mutually exclusive with
    /// `fits_current_gpu`.
    requires_offload_to_fit: bool,
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
    requested_context_length: ?u32 = null,
    current: ?*LoadedResources,

    /// Outcome of a managed model removal, including whether it was unloaded from the GPU.
    pub const RemoveResult = struct {
        unloaded_from_gpu: bool,
        cleared_active_selection: bool,
        removed: managed_mod.RemoveInstalledModelResult,
    };

    /// Creates a manager and immediately loads the model described by `spec`.
    /// Acquires the per-device GPU process lock before loading.
    /// @param spec Path, optional catalog ID, and optional context-length override for the model to load.
    /// @param instance Vulkan instance that owns the selected GPU device.
    /// @param gpu_config_value Detected GPU capabilities used for shader selection and catalog filtering.
    /// @param shader_dir Filesystem path to the directory containing compiled SPIR-V shaders.
    /// @param allocator Used for all heap allocations owned by this manager.
    /// @returns An initialised `ModelManager` with `current` pointing to the loaded resources,
    ///   or an error if loading fails.
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
            .requested_context_length = spec.requested_context_length,
            .current = current,
        };
    }

    /// Creates a manager with no model loaded; the server starts idle and the GPU lock is not held.
    /// @param instance Vulkan instance that owns the selected GPU device.
    /// @param gpu_config_value Detected GPU capabilities used for catalog filtering.
    /// @param shader_dir Filesystem path to the directory containing compiled SPIR-V shaders.
    /// @param requested_context_length Optional token-count override applied when a model is later activated.
    /// @param allocator Used for all heap allocations owned by this manager.
    pub fn initEmpty(
        instance: *const Instance,
        gpu_config_value: gpu_detect.GpuConfig,
        shader_dir: []const u8,
        requested_context_length: ?u32,
        allocator: std.mem.Allocator,
    ) ModelManager {
        return .{
            .allocator = allocator,
            .instance = instance,
            .gpu_config = gpu_config_value,
            .vram_budget_bytes = instance.vramBytes(),
            .shader_dir = shader_dir,
            .requested_context_length = requested_context_length,
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

    /// Returns the catalog profile string for the detected GPU (e.g. `"amd-rdna4-32gb"`).
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

        /// Returns the effective context token count: `requested_tokens` clamped to `context_capacity_tokens`.
        /// @param requested_tokens The number of context tokens the caller wants to use.
        /// @returns The usable token count, which may be less than requested if the model was loaded with a smaller window.
        pub fn activeContextTokens(self: @This(), requested_tokens: u32) u32 {
            return @min(requested_tokens, self.context_capacity_tokens);
        }

        /// Returns the device-local VRAM bytes consumed by the effective context window.
        /// @param requested_tokens The desired context length in tokens; clamped via `activeContextTokens`.
        /// @returns Bytes = clamped token count × `context_bytes_per_token`.
        pub fn activeContextBytes(self: @This(), requested_tokens: u32) u64 {
            return @as(u64, self.activeContextTokens(requested_tokens)) * self.context_bytes_per_token;
        }
    };

    /// Snapshots the VRAM usage of the active model, or returns zeroes with the full VRAM budget in `device_local_budget_bytes` if idle.
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
            const tested_profile_match = catalog_mod.supportsProfile(entry, profile);

            const installed = managed_mod.isInstalled(entry.id, allocator);
            const fit = managed_mod.describeFit(entry, self.vram_budget_bytes, allocator) catch managed_mod.ModelFit{
                .required_vram_bytes = entry.required_vram_bytes,
                .fits_current_gpu = catalog_mod.fitsGpu(entry, self.vram_budget_bytes),
                .exact = false,
                .required_vram_with_offload_bytes = catalog_mod.requiredVramWithOffload(entry),
                .fit_state = catalog_mod.fitState(entry, self.vram_budget_bytes),
            };
            // Currently-active entries are kept even when the catalog's
            // conservative required_vram_bytes exceeds the live budget —
            // the model is demonstrably running, so hiding it from
            // /v1/models just confuses clients that want to query by id.
            const is_active_entry = active_catalog_id != null and std.mem.eql(u8, active_catalog_id.?, entry.id);
            const supported_now = is_active_entry or (tested_profile_match and fit.fits_current_gpu);
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
                .required_vram_with_offload_bytes = fit.required_vram_with_offload_bytes,
                .requires_offload_to_fit = fit.fit_state == .fits_with_offload,
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
                .required_vram_with_offload_bytes = 0,
                .requires_offload_to_fit = false,
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

    /// Reports whether a catalog entry is both GPU-architecture-compatible and fits within the current VRAM budget.
    /// @param entry The catalog entry to evaluate.
    /// @param allocator Used for temporary allocations during fit computation; no long-lived allocation is made.
    /// @returns `true` when the entry matches the detected GPU profile and `describeFit` reports it fits.
    pub fn supportsManagedEntry(self: *ModelManager, entry: catalog_mod.CatalogEntry, allocator: std.mem.Allocator) bool {
        const fit = managed_mod.describeFit(entry, self.vram_budget_bytes, allocator) catch managed_mod.ModelFit{
            .required_vram_bytes = entry.required_vram_bytes,
            .fits_current_gpu = catalog_mod.fitsGpu(entry, self.vram_budget_bytes),
            .exact = false,
            .required_vram_with_offload_bytes = catalog_mod.requiredVramWithOffload(entry),
            .fit_state = catalog_mod.fitState(entry, self.vram_budget_bytes),
        };
        return catalog_mod.supportsProfile(entry, self.catalogProfile()) and fit.fits_current_gpu;
    }

    /// Loads and activates a managed catalog model, replacing any currently loaded model.
    /// If the requested model is already active the function returns immediately (optionally
    /// persisting the selection). The GPU process lock is acquired if not already held.
    /// @note Caller must hold the shared generation lock before calling this function.
    /// @param model_id Catalog entry ID to activate; must be installed on disk.
    /// @param persist_active When true, writes the selection to the active-model file so it
    ///   survives process restarts.
    /// @param force When false, an entry whose tested profiles do not include the
    ///   detected GPU is rejected with `error.ModelUnsupportedOnThisGpu`; when true,
    ///   activation proceeds on the untested profile with only a logged warning.
    /// @returns `error.UnknownManagedModel` if the ID is not in the catalog,
    ///   `error.ModelUnsupportedOnThisGpu` if the entry is not validated on this GPU
    ///   and `force` is false, `error.ModelNotInstalled` if the weights file is
    ///   absent, or `error.ModelDoesNotFit` if the model exceeds the VRAM budget.
    pub fn activateManagedModel(self: *ModelManager, model_id: []const u8, persist_active: bool, force: bool) !void {
        const entry = catalog_mod.find(model_id) orelse return error.UnknownManagedModel;
        if (!catalog_mod.supportsProfile(entry.*, self.catalogProfile())) {
            // Untested GPU profile: require an explicit opt-in (parity with the
            // CLI `model use --force`). The VRAM fit check below still enforces
            // real capacity when the caller does opt in.
            if (!force) return error.ModelUnsupportedOnThisGpu;
            std.log.scoped(.model_manager).warn(
                "{s} is not validated on this GPU profile ({s}); activating anyway (force)",
                .{ entry.display_name, self.catalogProfile() },
            );
        }
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
        loadResourcesInto(switched, .{
            .model_path = new_path,
            .managed_id = model_id,
            .requested_context_length = self.requested_context_length,
        }, self.instance, self.gpu_config, self.shader_dir, self.allocator) catch |switch_err| {
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

    /// Uninstalls a managed model from disk and, if it is currently loaded, optionally evicts it from the GPU.
    /// @note Caller must hold the shared generation lock before calling this function.
    /// @param model_id Catalog entry ID of the model to remove.
    /// @param force When `false`, returns `error.ModelLoadedInGpu` if the model is currently
    ///   active. When `true`, the model is unloaded from the GPU before deletion.
    /// @returns A `RemoveResult` describing whether the GPU was cleared and whether the
    ///   active-selection file was updated.
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
    resources.model = try loader_mod.load(spec.model_path, instance, &cmd_pool, allocator);
    errdefer resources.model.deinit(instance);

    // Derive a context length from the Vulkan device's VRAM budget when the
    // caller doesn't pin one. Matches the Metal path — see
    // `memory_plan.autoContextTokensForDeviceBudget` for the vLLM-inspired math.
    const effective_requested = spec.requested_context_length orelse memory_plan.autoContextTokensForDeviceBudget(
        memory_plan.profile(resources.model.config),
        tensorBytes(&resources.model),
        instance.vramBytes(),
        resources.model.config.context_length,
    );
    memory_plan.applyRequestedContextLimit(&resources.model.config, effective_requested);

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
    const weights_bytes = tensorBytes(&resources.model);
    const profile = memory_plan.profile(resources.model.config);
    const runtime_ctx = resources.engine.max_context_tokens;
    const kv_cache_bytes = profile.deviceLocalContextBytes(runtime_ctx);
    const runtime_device_local_bytes = profile.runtimeDeviceLocalBytes(runtime_ctx);
    resources.weights_bytes = weights_bytes;
    resources.runtime_device_local_bytes = runtime_device_local_bytes;
    resources.context_reserved_bytes = kv_cache_bytes;
    resources.context_capacity_tokens = runtime_ctx;
    resources.context_bytes_per_token = if (runtime_ctx == 0) 0 else @divTrunc(kv_cache_bytes, runtime_ctx);
    resources.device_local_bytes = weights_bytes + runtime_device_local_bytes;
    resources.device_local_budget_bytes = instance.vramBytes();

    std.debug.assert(resources.engine.model == &resources.model);
}

fn tensorBytes(model: *const loader_mod.Model) u64 {
    // Only count device-local tensors against the VRAM budget. MoE expert
    // tensors offloaded to host-visible memory live in system RAM and do
    // not consume VRAM. See loader.shouldOffloadToHost for the rule.
    var total: u64 = 0;
    for (model.gguf_file.tensors.items) |tensor_info| {
        if (loader_mod.shouldOffloadToHost(tensor_info.name)) continue;
        total += tensor_info.sizeBytes();
    }
    return total;
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
        .managed_id = try std.testing.allocator.dupe(u8, "qwen35-9b-q4k-m"),
        .display_name = try std.testing.allocator.dupe(u8, "Qwen3.5 9B Q4_K_M"),
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
        if (std.mem.eql(u8, entry.id, "qwen35-9b-q4k-m")) {
            saw_active = true;
            try std.testing.expect(entry.active);
            try std.testing.expect(entry.supports_thinking_toggle);
            try std.testing.expectEqualStrings("2026-02-28", entry.release_date);
        }
    }
    try std.testing.expect(saw_active);
}

test "collectCatalogView enables qwen36 A3B thinking toggle for raw matched path" {
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
        .model_path = try std.testing.allocator.dupe(u8, "/tmp/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf"),
        .managed_id = null,
        .display_name = try std.testing.allocator.dupe(u8, "Qwen3.6-35B-A3B-UD-Q4_K_XL"),
        .weights_bytes = 20 * 1024 * 1024 * 1024,
        .runtime_device_local_bytes = 1024 * 1024 * 1024,
        .context_reserved_bytes = 768 * 1024 * 1024 * 1024,
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
        .managed_id = try std.testing.allocator.dupe(u8, "qwen36-35b-a3b-q4k-xl"),
        .display_name = try std.testing.allocator.dupe(u8, "Qwen3.6 35B-A3B UD Q4_K_XL"),
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

    try std.testing.expectError(error.ModelLoadedInGpu, fake.removeManagedModel("qwen36-35b-a3b-q4k-xl", false));
    try std.testing.expect(fake.current == &current);
}
