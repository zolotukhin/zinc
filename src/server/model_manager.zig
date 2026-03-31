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
const gpu_detect = @import("../vulkan/gpu_detect.zig");
const instance_mod = @import("../vulkan/instance.zig");
const CommandPool = @import("../vulkan/command.zig").CommandPool;

const Instance = instance_mod.Instance;

pub const LoadSpec = struct {
    model_path: []const u8,
    managed_id: ?[]const u8 = null,
};

pub const ModelSummary = struct {
    id: []const u8,
    display_name: []const u8,
    homepage_url: []const u8,
    installed: bool,
    active: bool,
    managed: bool,
    supported_on_current_gpu: bool,
    fits_current_gpu: bool,
    required_vram_bytes: u64,
    exact_fit: bool,
    status_label: []const u8,
    supports_thinking_toggle: bool,
};

pub const ModelCatalogView = struct {
    profile: []const u8,
    data: []ModelSummary,

    pub fn deinit(self: *ModelCatalogView, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        self.* = undefined;
    }
};

pub const LoadedResources = struct {
    model: loader_mod.Model,
    tokenizer: tokenizer_mod.Tokenizer,
    engine: forward_mod.InferenceEngine,
    model_path: []u8,
    managed_id: ?[]u8,
    display_name: []u8,
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

pub const ModelManager = struct {
    allocator: std.mem.Allocator,
    instance: *const Instance,
    gpu_config: gpu_detect.GpuConfig,
    vram_budget_bytes: u64,
    shader_dir: []const u8,
    state_mutex: std.Thread.Mutex = .{},
    current: *LoadedResources,

    pub fn init(
        spec: LoadSpec,
        instance: *const Instance,
        gpu_config_value: gpu_detect.GpuConfig,
        shader_dir: []const u8,
        allocator: std.mem.Allocator,
    ) !ModelManager {
        const current = try allocator.create(LoadedResources);
        errdefer allocator.destroy(current);
        try loadResourcesInto(current, spec, instance, gpu_config_value, shader_dir, allocator);
        return .{
            .allocator = allocator,
            .instance = instance,
            .gpu_config = gpu_config_value,
            .vram_budget_bytes = instance.vramBytes(),
            .shader_dir = shader_dir,
            .current = current,
        };
    }

    pub fn deinit(self: *ModelManager) void {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();
        self.current.deinit(self.instance, self.allocator);
        self.allocator.destroy(self.current);
    }

    pub fn currentResources(self: *ModelManager) *LoadedResources {
        return self.current;
    }

    pub fn activeDisplayName(self: *ModelManager) []const u8 {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();
        return self.current.display_name;
    }

    pub const MemoryUsage = struct {
        device_local_bytes: u64,
        device_local_budget_bytes: u64,
    };

    pub fn currentMemoryUsage(self: *ModelManager) MemoryUsage {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();
        return .{
            .device_local_bytes = self.current.device_local_bytes,
            .device_local_budget_bytes = self.current.device_local_budget_bytes,
        };
    }

    pub fn collectCatalogView(self: *ModelManager, allocator: std.mem.Allocator, include_all: bool) !ModelCatalogView {
        self.state_mutex.lock();
        defer self.state_mutex.unlock();

        const profile = catalog_mod.profileForGpu(self.gpu_config);
        const active_managed_id = self.current.managed_id;
        const active_display_name = self.current.display_name;
        const active_supports_thinking_toggle = self.current.tokenizer.supportsThinkingToggle();

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
                .homepage_url = entry.homepage_url,
                .installed = installed,
                .active = active_managed_id != null and std.mem.eql(u8, active_managed_id.?, entry.id),
                .managed = true,
                .supported_on_current_gpu = supported_now,
                .fits_current_gpu = fit.fits_current_gpu,
                .required_vram_bytes = fit.required_vram_bytes,
                .exact_fit = fit.exact,
                .status_label = status_label,
                .supports_thinking_toggle = active_managed_id != null and std.mem.eql(u8, active_managed_id.?, entry.id) and active_supports_thinking_toggle,
            });
        }

        if (active_managed_id == null) {
            try list.append(allocator, .{
                .id = active_display_name,
                .display_name = active_display_name,
                .homepage_url = "",
                .installed = true,
                .active = true,
                .managed = false,
                .supported_on_current_gpu = true,
                .fits_current_gpu = true,
                .required_vram_bytes = 0,
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

    /// Caller must already hold the shared generation lock.
    pub fn activateManagedModel(self: *ModelManager, model_id: []const u8, persist_active: bool) !void {
        const entry = catalog_mod.find(model_id) orelse return error.UnknownManagedModel;
        const profile = catalog_mod.profileForGpu(self.gpu_config);
        if (!catalog_mod.supportsProfile(entry.*, profile)) return error.ModelUnsupportedOnThisGpu;
        if (!managed_mod.isInstalled(model_id, self.allocator)) return error.ModelNotInstalled;

        const fit = try managed_mod.verifyActiveSelectionFits(model_id, self.vram_budget_bytes, self.allocator);
        if (!fit.fits_current_gpu) return error.ModelDoesNotFit;

        self.state_mutex.lock();
        defer self.state_mutex.unlock();

        if (self.current.managed_id) |active_id| {
            if (std.mem.eql(u8, active_id, model_id)) {
                if (persist_active) try managed_mod.writeActiveSelection(model_id, self.allocator);
                return;
            }
        }

        const new_path = try managed_mod.resolveInstalledModelPath(model_id, self.allocator);
        defer self.allocator.free(new_path);
        const switched = try self.allocator.create(LoadedResources);
        errdefer self.allocator.destroy(switched);
        loadResourcesInto(switched, .{ .model_path = new_path, .managed_id = model_id }, self.instance, self.gpu_config, self.shader_dir, self.allocator) catch |switch_err| {
            return switch_err;
        };

        const previous = self.current;
        self.current = switched;
        previous.deinit(self.instance, self.allocator);
        self.allocator.destroy(previous);

        if (persist_active) try managed_mod.writeActiveSelection(model_id, self.allocator);
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
    resources.device_local_bytes = fit.total_device_local_bytes;
    resources.device_local_budget_bytes = fit.vram_budget_bytes;

    std.debug.assert(resources.engine.model == &resources.model);
}

fn fallbackModelName(model: *const loader_mod.Model) []const u8 {
    return switch (model.config.architecture) {
        .qwen35 => "qwen3.5",
        .qwen2_moe => "qwen3.5-35b",
        .qwen2 => "qwen2",
        .llama => "llama",
        .mistral => "mistral",
        .mamba => "mamba",
        .jamba => "jamba",
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
        .device_local_bytes = 21 * 1024 * 1024 * 1024,
        .device_local_budget_bytes = 32 * 1024 * 1024 * 1024,
    };
    fake.current = &current;
    defer {
        fake.current.tokenizer.token_to_id.deinit();
        std.testing.allocator.free(fake.current.model_path);
        if (fake.current.managed_id) |id| std.testing.allocator.free(id);
        std.testing.allocator.free(fake.current.display_name);
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
        }
    }
    try std.testing.expect(saw_active);
}
