//! Managed model cache, active-model selection, and download helpers.
//! @section Managed Models
//! These helpers back the `zinc model ...` CLI and the server-side active-model
//! switching flow.
const std = @import("std");
const host_io = @import("host_io");
const builtin = @import("builtin");
const gpu = @import("../gpu/interface.zig");
const catalog = @import("catalog.zig");
const diagnostics = if (gpu.is_metal) @import("../diagnostics_metal.zig") else @import("../diagnostics.zig");
const loader_mod = if (gpu.is_metal) @import("loader_metal.zig") else @import("loader.zig");

/// The compiled runtime owns its own GPU-index namespace and capability cache.
pub const catalog_backend: catalog.CatalogBackend = if (gpu.is_metal) .metal else if (gpu.is_rocm) .rocm else if (gpu.is_cuda) .cuda else .vulkan;

/// Resolved cache and config directory paths for the current platform.
pub const RuntimePaths = struct {
    cache_root: []u8,
    config_root: []u8,

    /// Frees the owned path slices and invalidates the struct.
    pub fn deinit(self: *RuntimePaths, allocator: std.mem.Allocator) void {
        allocator.free(self.cache_root);
        allocator.free(self.config_root);
        self.* = undefined;
    }
};

/// VRAM budget check result for a single catalog model.
pub const ModelFit = struct {
    required_vram_bytes: u64,
    fits_current_gpu: bool,
    exact: bool,
    /// VRAM required when MoE expert tensors are offloaded to host RAM via
    /// `ZINC_OFFLOAD_MOE_EXPERTS=1`. Equal to `required_vram_bytes` for dense
    /// models. For MoE models, this is meaningfully smaller and lets users
    /// fit a model that would otherwise exceed the VRAM budget.
    required_vram_with_offload_bytes: u64,
    /// Tri-state offload-aware fit assessment.
    fit_state: catalog.FitState,
};

/// The currently active managed model as persisted in the config directory.
pub const ActiveSelection = struct {
    model_id: []u8,
    selected_at_unix: i64,

    /// Frees the owned model_id slice and invalidates the struct.
    pub fn deinit(self: *ActiveSelection, allocator: std.mem.Allocator) void {
        allocator.free(self.model_id);
        self.* = undefined;
    }
};

/// Cached GPU capability profile used to avoid re-probing the device on every run.
pub const CachedGpuProfile = struct {
    profile: []u8,
    device_name: []u8,
    vram_budget_bytes: u64,
    cached_at_unix: i64,

    /// Frees the owned string slices and invalidates the struct.
    pub fn deinit(self: *CachedGpuProfile, allocator: std.mem.Allocator) void {
        allocator.free(self.profile);
        allocator.free(self.device_name);
        self.* = undefined;
    }
};

/// On-disk manifest written alongside an installed GGUF model file.
///
/// `sha256` is optional: hand-rolled manifests for symlinked local files
/// (e.g. `download_url:"local"`) frequently omit it because the user can't
/// or won't precompute the digest of an existing GGUF. Treat it as a hint,
/// not a required field — the catalog's `sha256` is the source of truth
/// when a checksum needs to be enforced.
pub const InstalledManifest = struct {
    size_bytes: u64,
    sha256: ?[]u8,
    required_vram_bytes: ?u64,
    /// Bytes of MoE expert tensors in the installed file. Nullable because
    /// older manifests written before the offload-aware loader existed don't
    /// carry it; callers fall back to the catalog estimate in that case.
    offloadable_vram_bytes: ?u64 = null,

    /// Frees the owned sha256 slice (if any) and invalidates the struct.
    pub fn deinit(self: *InstalledManifest, allocator: std.mem.Allocator) void {
        if (self.sha256) |s| allocator.free(s);
        self.* = undefined;
    }
};

/// Outcome of a model removal: which artifacts were actually deleted.
pub const RemoveInstalledModelResult = struct {
    deleted_model: bool,
    deleted_manifest: bool,
    removed_dir: bool,
};

/// Callback hooks for observing model download lifecycle events.
pub const DownloadObserver = struct {
    context: ?*anyopaque = null,
    on_start: ?*const fn (context: ?*anyopaque, total_bytes: ?u64) void = null,
    on_progress: ?*const fn (context: ?*anyopaque, downloaded_bytes: u64, total_bytes: ?u64) void = null,
    on_verifying: ?*const fn (context: ?*anyopaque, downloaded_bytes: u64) void = null,
    on_complete: ?*const fn (context: ?*anyopaque, downloaded_bytes: u64) void = null,
};

const progress_bar_width = 28;
const progress_update_interval_ns = 150 * std.time.ns_per_ms;
const redirect_buffer_len = 4096;

const UpstreamArtifactMetadata = struct {
    size_bytes: ?u64 = null,
    sha256_hex: ?[64]u8 = null,
};

fn estimateRequiredBytes(inspection: loader_mod.ModelInspection) u64 {
    if (comptime gpu.is_metal) {
        return diagnostics.estimateUnifiedFit(inspection, std.math.maxInt(u64), std.math.maxInt(u64), 4096).total_unified_bytes;
    }
    return diagnostics.estimateFit(inspection, std.math.maxInt(u64), 4096).total_device_local_bytes;
}

fn findHeaderValue(head: std.http.Client.Response.Head, name: []const u8) ?[]const u8 {
    var it = head.iterateHeaders();
    while (it.next()) |header| {
        if (std.ascii.eqlIgnoreCase(header.name, name)) return header.value;
    }
    return null;
}

fn canonicalizeSha256Header(value: []const u8) ?[64]u8 {
    const trimmed = std.mem.trim(u8, value, "\" ");
    if (trimmed.len != 64) return null;

    var out: [64]u8 = undefined;
    for (trimmed, 0..) |c, i| {
        if (!std.ascii.isHex(c)) return null;
        out[i] = std.ascii.toLower(c);
    }
    return out;
}

fn fetchUpstreamArtifactMetadata(client: *std.http.Client, uri: std.Uri) !UpstreamArtifactMetadata {
    var req = try client.request(.HEAD, uri, .{ .redirect_behavior = .unhandled });
    defer req.deinit();
    try req.sendBodiless();

    const response = try req.receiveHead(&.{});
    return try extractUpstreamArtifactMetadata(response.head);
}

fn extractUpstreamArtifactMetadata(head: std.http.Client.Response.Head) !UpstreamArtifactMetadata {
    const status_class = head.status.class();
    if (status_class != .success and status_class != .redirect) return error.DownloadFailed;

    var metadata = UpstreamArtifactMetadata{};
    if (findHeaderValue(head, "x-linked-size")) |value| {
        metadata.size_bytes = std.fmt.parseInt(u64, std.mem.trim(u8, value, " "), 10) catch return error.InvalidUpstreamMetadata;
    } else if (status_class == .success) {
        metadata.size_bytes = head.content_length;
    }
    if (findHeaderValue(head, "x-linked-etag")) |value| {
        metadata.sha256_hex = canonicalizeSha256Header(value);
    }
    return metadata;
}

fn preflightCatalogDrift(
    entry: catalog.CatalogEntry,
    metadata: UpstreamArtifactMetadata,
    writer: anytype,
) !void {
    if (metadata.size_bytes) |size_bytes| {
        // size_bytes == 0 means the entry carries no pinned size (e.g. `-hf`
        // downloads), so there is no catalog expectation to drift from.
        if (entry.size_bytes != 0 and size_bytes != entry.size_bytes) {
            try writer.print(
                "Catalog note: upstream size is {d:.2} GiB but pinned catalog size is {d:.2} GiB; continuing because sha256 pin is authoritative.\n",
                .{ bytesToGiB(size_bytes), bytesToGiB(entry.size_bytes) },
            );
        }
    }

    if (metadata.sha256_hex) |sha256_hex| {
        if (entry.sha256.len == 0) {
            // No pinned sha256 — accept whatever the remote provides.
        } else if (!std.ascii.eqlIgnoreCase(sha256_hex[0..], entry.sha256)) {
            try writer.print(
                "Catalog stale: upstream sha256 is {s} but pinned catalog sha256 is {s}\n",
                .{ sha256_hex[0..], entry.sha256 },
            );
            try writer.writeAll("Refresh src/model/catalog.zig before pulling this managed model.\n");
            return error.CatalogMetadataMismatch;
        }
    }
}

/// Returns the resolved cache and config root directories for the current platform.
pub fn runtimePaths(allocator: std.mem.Allocator) !RuntimePaths {
    return .{
        .cache_root = try resolveCacheRoot(allocator),
        .config_root = try resolveConfigRoot(allocator),
    };
}

// A model id names a single directory under `<cache_root>/models/`. Ids
// arrive from the catalog, `-hf` cache-id derivation, CLI arguments, server
// request bodies, and the active-model config file; restricting them to one
// filesystem-safe path component here makes every id-keyed cache path
// (resolve / isInstalled / remove) traversal-proof regardless of caller.
fn isValidModelId(id: []const u8) bool {
    if (id.len == 0 or id.len > 128) return false;
    if (id[0] == '.' or id[0] == '-') return false;
    for (id) |c| {
        switch (c) {
            'A'...'Z', 'a'...'z', '0'...'9', '.', '_', '-' => {},
            else => return false,
        }
    }
    return true;
}

/// Returns the absolute path to the installed GGUF file for the given model id.
///
/// @param model_id Catalog model identifier (e.g. `"qwen3-2b"`).
/// @param allocator Used to build the path; caller owns the returned slice.
/// @returns Heap-allocated absolute path `<cache_root>/models/<model_id>/model.gguf`.
/// @note Ids that are not a single filesystem-safe path component fail with
/// `error.InvalidModelId`, so no caller can resolve a path outside the cache.
pub fn resolveInstalledModelPath(model_id: []const u8, allocator: std.mem.Allocator) ![]u8 {
    if (!isValidModelId(model_id)) return error.InvalidModelId;
    var paths = try runtimePaths(allocator);
    defer paths.deinit(allocator);
    return try std.fs.path.join(allocator, &.{ paths.cache_root, "models", model_id, "model.gguf" });
}

/// Returns the absolute path to the manifest JSON for the given model id.
///
/// @param model_id Catalog model identifier.
/// @param allocator Used to build the path; caller owns the returned slice.
/// @returns Heap-allocated absolute path `<cache_root>/models/<model_id>/manifest.json`.
/// @note Ids that are not a single filesystem-safe path component fail with
/// `error.InvalidModelId`, so no caller can resolve a path outside the cache.
pub fn resolveManifestPath(model_id: []const u8, allocator: std.mem.Allocator) ![]u8 {
    if (!isValidModelId(model_id)) return error.InvalidModelId;
    var paths = try runtimePaths(allocator);
    defer paths.deinit(allocator);
    return try std.fs.path.join(allocator, &.{ paths.cache_root, "models", model_id, "manifest.json" });
}

/// Returns the absolute path to the active-model config file.
pub fn resolveActiveConfigPath(allocator: std.mem.Allocator) ![]u8 {
    const root = try resolveConfigRoot(allocator);
    defer allocator.free(root);
    return try std.fs.path.join(allocator, &.{ root, "active-model.json" });
}

/// Returns the absolute path to the cached GPU profile JSON for the given device index.
///
/// @param device_index Zero-based index of the GPU device; encoded in the filename.
/// @param allocator Used to build the path; caller owns the returned slice.
/// @returns Heap-allocated absolute path `<config_root>/gpu-profile-<backend>-device-<device_index>.json`.
pub fn resolveGpuProfileCachePath(device_index: u32, allocator: std.mem.Allocator) ![]u8 {
    const root = try resolveConfigRoot(allocator);
    defer allocator.free(root);
    return try gpuProfileCachePath(root, catalog_backend, device_index, allocator);
}

fn gpuProfileCachePath(root: []const u8, backend: catalog.CatalogBackend, device_index: u32, allocator: std.mem.Allocator) ![]u8 {
    // Legacy files have no backend identity. Ignore them instead of reusing a
    // Vulkan device index or a stale NVIDIA profile from a ROCm invocation.
    return try std.fmt.allocPrint(allocator, "{s}/gpu-profile-{s}-device-{d}.json", .{ root, @tagName(backend), device_index });
}

/// Returns true if the model GGUF file exists in the local cache.
pub fn isInstalled(model_id: []const u8, allocator: std.mem.Allocator) bool {
    const path = resolveInstalledModelPath(model_id, allocator) catch return false;
    defer allocator.free(path);
    std.Io.Dir.accessAbsolute(host_io.io, path, .{}) catch return false;
    return true;
}

/// Reads the persisted active-model selection, or returns null if none is set.
pub fn readActiveSelection(allocator: std.mem.Allocator) !?ActiveSelection {
    const path = try resolveActiveConfigPath(allocator);
    defer allocator.free(path);

    const file = std.Io.Dir.openFileAbsolute(host_io.io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer {
        var close_file = file;
        close_file.close(host_io.io);
    }

    const data = try host_io.readToEndAlloc(file, allocator, 4096);
    defer allocator.free(data);

    const model_id = extractJsonStringField(data, "active_model_id") orelse return error.InvalidActiveModelConfig;
    const selected_at = extractJsonI64Field(data, "selected_at_unix") orelse 0;

    return .{
        .model_id = try allocator.dupe(u8, model_id),
        .selected_at_unix = selected_at,
    };
}

/// Persists the given model id as the active selection with the current timestamp.
pub fn writeActiveSelection(model_id: []const u8, allocator: std.mem.Allocator) !void {
    const path = try resolveActiveConfigPath(allocator);
    defer allocator.free(path);
    try ensureParentDir(path);

    const file = try std.Io.Dir.createFileAbsolute(host_io.io, path, .{ .truncate = true });
    defer {
        var close_file = file;
        close_file.close(host_io.io);
    }

    var file_buffer: [1024]u8 = undefined;
    var file_writer = file.writerStreaming(host_io.io, &file_buffer);
    try file_writer.interface.print(
        \\{{"active_model_id":"{s}","selected_at_unix":{d}}}
    , .{ model_id, host_io.timestamp() });
    try file_writer.interface.flush();
}

/// Removes the active-model config file. Returns true if a file was deleted.
pub fn clearActiveSelection(allocator: std.mem.Allocator) !bool {
    const path = try resolveActiveConfigPath(allocator);
    defer allocator.free(path);
    return deleteFileIfExistsAbsolute(path);
}

/// Clears the active selection only if it currently points to the given model id.
pub fn clearActiveSelectionIfMatches(model_id: []const u8, allocator: std.mem.Allocator) !bool {
    var active = try readActiveSelection(allocator);
    defer if (active) |*selection| selection.deinit(allocator);
    if (active) |selection| {
        if (std.mem.eql(u8, selection.model_id, model_id)) {
            return try clearActiveSelection(allocator);
        }
    }
    return false;
}

/// Reads the cached GPU profile for the given device, or returns null if not cached.
pub fn readCachedGpuProfile(device_index: u32, allocator: std.mem.Allocator) !?CachedGpuProfile {
    const path = try resolveGpuProfileCachePath(device_index, allocator);
    defer allocator.free(path);

    const file = std.Io.Dir.openFileAbsolute(host_io.io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer {
        var close_file = file;
        close_file.close(host_io.io);
    }

    const data = try host_io.readToEndAlloc(file, allocator, 4096);
    defer allocator.free(data);

    return try parseGpuProfileCache(data, catalog_backend, allocator);
}

const StoredGpuProfile = struct {
    schema_version: u32,
    backend: catalog.CatalogBackend,
    profile: []const u8,
    device_name: []const u8,
    vram_budget_bytes: u64,
    cached_at_unix: i64,
};

fn parseGpuProfileCache(data: []const u8, backend: catalog.CatalogBackend, allocator: std.mem.Allocator) !?CachedGpuProfile {
    const parsed = std.json.parseFromSlice(StoredGpuProfile, allocator, data, .{ .allocate = .alloc_always }) catch |err| switch (err) {
        error.OutOfMemory => return err,
        else => return null, // Advisory metadata; malformed/legacy data needs a fresh probe.
    };
    defer parsed.deinit();
    const stored = parsed.value;
    if (stored.schema_version != 2 or stored.backend != backend) return null;
    const profile = try allocator.dupe(u8, stored.profile);
    errdefer allocator.free(profile);
    return .{ .profile = profile, .device_name = try allocator.dupe(u8, stored.device_name), .vram_budget_bytes = stored.vram_budget_bytes, .cached_at_unix = stored.cached_at_unix };
}

/// Persists a GPU capability profile to disk for the given device index.
///
/// Overwrites any existing cache file for that device.
/// @param device_index Zero-based GPU index; used to derive the cache file name.
/// @param profile Opaque profile string (e.g. `"rdna4"`) returned by the diagnostics layer.
/// @param device_name Human-readable device name stored for informational display.
/// @param vram_budget_bytes Effective VRAM budget in bytes for this device.
/// @param allocator Used only for path construction; nothing is retained after the call.
pub fn writeCachedGpuProfile(
    device_index: u32,
    profile: []const u8,
    device_name: []const u8,
    vram_budget_bytes: u64,
    allocator: std.mem.Allocator,
) !void {
    const path = try resolveGpuProfileCachePath(device_index, allocator);
    defer allocator.free(path);
    try ensureParentDir(path);

    const file = try std.Io.Dir.createFileAbsolute(host_io.io, path, .{ .truncate = true });
    defer {
        var close_file = file;
        close_file.close(host_io.io);
    }

    var file_buffer: [2048]u8 = undefined;
    var file_writer = file.writerStreaming(host_io.io, &file_buffer);
    try std.json.Stringify.value(StoredGpuProfile{ .schema_version = 2, .backend = catalog_backend, .profile = profile, .device_name = device_name, .vram_budget_bytes = vram_budget_bytes, .cached_at_unix = host_io.timestamp() }, .{}, &file_writer.interface);
    try file_writer.interface.flush();
}

/// Returns a VRAM fit assessment for a catalog model against the given budget.
///
/// If the model is installed, the on-disk manifest is consulted first (most
/// accurate); if the manifest lacks VRAM fields, the installed GGUF is
/// inspected directly and the manifest is updated. For models that are not yet
/// installed, catalog static estimates are used and `ModelFit.exact` is false.
/// @param entry Catalog entry describing the model to assess.
/// @param vram_budget_bytes Available VRAM in bytes on the target device.
/// @param allocator Used for path resolution and manifest I/O; nothing is retained.
/// @returns `ModelFit` with exact=true when data came from the installed model (manifest or file inspection), false for catalog estimates.
pub fn describeFit(entry: catalog.CatalogEntry, vram_budget_bytes: u64, allocator: std.mem.Allocator) !ModelFit {
    if (isInstalled(entry.id, allocator)) {
        const installed_path = try resolveInstalledModelPath(entry.id, allocator);
        defer allocator.free(installed_path);
        const manifest_path = try resolveManifestPath(entry.id, allocator);
        defer allocator.free(manifest_path);

        // Existing manifests don't carry offloadable_bytes; fall back to the
        // catalog estimate for that field. New manifests written below carry
        // the exact value from the GGUF inspection.
        if (try readInstalledManifest(manifest_path, allocator)) |manifest| {
            defer {
                var owned = manifest;
                owned.deinit(allocator);
            }
            if (manifest.required_vram_bytes) |required_vram_bytes| {
                const offloadable = manifest.offloadable_vram_bytes orelse entry.offloadable_vram_bytes;
                const with_offload = if (offloadable >= required_vram_bytes) 0 else required_vram_bytes - offloadable;
                return makeModelFit(required_vram_bytes, offloadable, with_offload, vram_budget_bytes, true);
            }
        }

        const inspection = try loader_mod.inspectModel(installed_path, allocator);
        const required_bytes = estimateRequiredBytes(inspection);
        const offloadable = inspection.offloadable_tensor_bytes;
        const with_offload = if (offloadable >= required_bytes) 0 else required_bytes - offloadable;
        try writeManifest(manifest_path, entry, inspection.file_size, required_bytes, offloadable, allocator);
        return makeModelFit(required_bytes, offloadable, with_offload, vram_budget_bytes, true);
    }

    // No installed file: fall back to the catalog's static estimates.
    return .{
        .required_vram_bytes = entry.required_vram_bytes,
        .fits_current_gpu = catalog.fitsGpu(entry, vram_budget_bytes),
        .exact = false,
        .required_vram_with_offload_bytes = catalog.requiredVramWithOffload(entry),
        .fit_state = catalog.fitState(entry, vram_budget_bytes),
    };
}

fn makeModelFit(required: u64, offloadable: u64, with_offload: u64, budget: u64, exact: bool) ModelFit {
    const fits = required <= budget;
    const state: catalog.FitState = if (fits)
        .fits
    else if (offloadable > 0 and with_offload <= budget)
        .fits_with_offload
    else
        .does_not_fit;
    return .{
        .required_vram_bytes = required,
        .fits_current_gpu = fits,
        .exact = exact,
        .required_vram_with_offload_bytes = if (with_offload == 0) required else with_offload,
        .fit_state = state,
    };
}

/// Verifies that the named model is installed and fits in the given VRAM budget.
///
/// Returns `error.UnknownManagedModel` if `model_id` is not in the catalog, or
/// `error.ModelNotInstalled` if the GGUF file is absent from the local cache.
/// @param model_id Catalog model identifier to check.
/// @param vram_budget_bytes Available VRAM in bytes on the target device.
/// @param allocator Forwarded to `describeFit` for path and manifest I/O.
/// @returns `ModelFit` describing the exact VRAM requirement and fit state.
pub fn verifyActiveSelectionFits(model_id: []const u8, vram_budget_bytes: u64, allocator: std.mem.Allocator) !ModelFit {
    const entry = catalog.find(model_id) orelse return error.UnknownManagedModel;
    if (!isInstalled(model_id, allocator)) return error.ModelNotInstalled;
    return describeFit(entry.*, vram_budget_bytes, allocator);
}

/// Deletes an installed model's GGUF, manifest, and (if empty) its directory.
pub fn removeInstalledModel(model_id: []const u8, allocator: std.mem.Allocator) !RemoveInstalledModelResult {
    const model_dir = try resolveInstalledModelDir(model_id, allocator);
    defer allocator.free(model_dir);
    const model_path = try resolveInstalledModelPath(model_id, allocator);
    defer allocator.free(model_path);
    const manifest_path = try resolveManifestPath(model_id, allocator);
    defer allocator.free(manifest_path);

    return removeInstalledModelAtPaths(model_dir, model_path, manifest_path);
}

/// Downloads and installs a model from the catalog, verifying its sha256 checksum.
pub fn pullModel(entry: catalog.CatalogEntry, allocator: std.mem.Allocator, writer: anytype) !void {
    try pullModelWithObserver(entry, allocator, writer, null);
}

/// Downloads and installs a model, reporting progress via an optional observer.
///
/// If the model is already installed and its sha256 matches the catalog, the
/// function returns immediately without re-downloading. A `.partial` staging
/// file is used so an interrupted download does not corrupt the cache.
/// @param entry Catalog entry that supplies the download URL and expected sha256.
/// @param allocator Used for HTTP client, path construction, and temporary buffers.
/// @param writer Receives human-readable status lines and the download progress bar.
/// @param observer Optional lifecycle callbacks for programmatic progress tracking; pass null to skip.
pub fn pullModelWithObserver(
    entry: catalog.CatalogEntry,
    allocator: std.mem.Allocator,
    writer: anytype,
    observer: ?*const DownloadObserver,
) !void {
    if (!isValidModelId(entry.id)) return error.InvalidModelId;
    var paths = try runtimePaths(allocator);
    defer paths.deinit(allocator);

    const models_dir = try std.fs.path.join(allocator, &.{ paths.cache_root, "models", entry.id });
    defer allocator.free(models_dir);
    const final_path = try std.fs.path.join(allocator, &.{ models_dir, "model.gguf" });
    defer allocator.free(final_path);
    const manifest_path = try std.fs.path.join(allocator, &.{ models_dir, "manifest.json" });
    defer allocator.free(manifest_path);

    if (isInstalled(entry.id, allocator)) {
        // No pinned sha256 (e.g. `-hf` downloads): trust the cached file.
        if (entry.sha256.len == 0) {
            try writer.print("Already installed: {s}\n", .{final_path});
            return;
        }
        const actual_sha = try computeFileSha256Hex(final_path, allocator);
        defer allocator.free(actual_sha);
        if (std.ascii.eqlIgnoreCase(actual_sha, entry.sha256)) {
            try writer.print("Already installed: {s}\n", .{final_path});
            return;
        }
        try writer.print("Cached file checksum mismatch, replacing: {s}\n", .{final_path});
        std.Io.Dir.deleteFileAbsolute(host_io.io, final_path) catch {};
    }

    const downloads_dir = try std.fs.path.join(allocator, &.{ paths.cache_root, "downloads" });
    defer allocator.free(downloads_dir);
    try std.Io.Dir.cwd().createDirPath(host_io.io, downloads_dir);
    try std.Io.Dir.cwd().createDirPath(host_io.io, models_dir);

    const partial_name = try std.fmt.allocPrint(allocator, "{s}.partial", .{entry.id});
    defer allocator.free(partial_name);
    const partial_path = try std.fs.path.join(allocator, &.{ downloads_dir, partial_name });
    defer allocator.free(partial_path);
    std.Io.Dir.deleteFileAbsolute(host_io.io, partial_path) catch {};

    try writer.print("Resolving model: {s}\n", .{entry.id});
    try writer.print("Downloading: {s}\n", .{entry.download_url});

    var client: std.http.Client = .{ .allocator = allocator, .io = host_io.io };
    defer client.deinit();

    const uri = try std.Uri.parse(entry.download_url);
    const upstream_metadata = try fetchUpstreamArtifactMetadata(&client, uri);
    try preflightCatalogDrift(entry, upstream_metadata, writer);

    const partial_file = try std.Io.Dir.createFileAbsolute(host_io.io, partial_path, .{ .truncate = true });
    defer {
        var close_file = partial_file;
        close_file.close(host_io.io);
    }
    errdefer std.Io.Dir.deleteFileAbsolute(host_io.io, partial_path) catch {};

    var file_buffer: [16 * 1024]u8 = undefined;
    var file_writer = partial_file.writerStreaming(host_io.io, &file_buffer);

    var req = try client.request(.GET, uri, .{});
    defer req.deinit();
    try req.sendBodiless();

    var redirect_buffer: [redirect_buffer_len]u8 = undefined;
    var response = try req.receiveHead(&redirect_buffer);
    if (response.head.status.class() != .success) return error.DownloadFailed;

    const total_bytes = response.head.content_length;
    if (total_bytes) |total| {
        try writer.print("Size: {d:.2} GiB\n", .{bytesToGiB(total)});
    } else {
        try writer.writeAll("Size: unknown\n");
    }
    if (observer) |obs| {
        if (obs.on_start) |cb| cb(obs.context, total_bytes);
    }

    var transfer_buffer: [64]u8 = undefined;
    var reader = response.reader(&transfer_buffer);
    var download_buffer: [64 * 1024]u8 = undefined;
    var downloaded_bytes: u64 = 0;
    var progress_timer = try host_io.Timer.start();
    var last_progress_ns: u64 = 0;
    var progress_started = false;

    try writeDownloadProgress(writer, 0, total_bytes, 0, false);
    progress_started = true;

    while (true) {
        // Guard: stop once all expected bytes are received.
        // Zig 0.15 std.http panics if we read past the content-length boundary.
        if (total_bytes) |total| {
            if (downloaded_bytes >= total) break;
        }
        const n = reader.readSliceShort(&download_buffer) catch |err| switch (err) {
            error.ReadFailed => return response.bodyErr().?,
        };
        if (n == 0) break;
        try file_writer.interface.writeAll(download_buffer[0..n]);
        downloaded_bytes += n;
        if (observer) |obs| {
            if (obs.on_progress) |cb| cb(obs.context, downloaded_bytes, total_bytes);
        }
        const elapsed_ns = progress_timer.read();
        if (elapsed_ns - last_progress_ns >= progress_update_interval_ns) {
            try writeDownloadProgress(writer, downloaded_bytes, total_bytes, elapsed_ns, false);
            last_progress_ns = elapsed_ns;
        }
    }

    try file_writer.interface.flush();
    if (progress_started) {
        try writeDownloadProgress(writer, downloaded_bytes, total_bytes, progress_timer.read(), true);
    }

    const stat = try partial_file.stat(host_io.io);
    if (observer) |obs| {
        if (obs.on_verifying) |cb| cb(obs.context, stat.size);
    }

    // Prefer the pinned catalog digest; entries without one (`-hf` synthesizes
    // them from the Hugging Face manifest, which may omit `lfs.sha256`) fall
    // back to the digest the upstream HEAD preflight advertised (Hugging
    // Face's x-linked-etag is the LFS sha256), so those downloads still get
    // end-to-end integrity. Only when neither source has a digest is the
    // file installed unverified — loudly, not silently.
    var upstream_sha_buf: [64]u8 = undefined;
    const expected_sha: []const u8 = blk: {
        if (entry.sha256.len > 0) break :blk entry.sha256;
        if (upstream_metadata.sha256_hex) |hex| {
            upstream_sha_buf = hex;
            break :blk upstream_sha_buf[0..];
        }
        break :blk "";
    };
    if (expected_sha.len > 0) {
        try writer.writeAll("Verifying sha256...\n");
        const actual_sha = try computeFileSha256Hex(partial_path, allocator);
        defer allocator.free(actual_sha);
        if (!std.ascii.eqlIgnoreCase(actual_sha, expected_sha)) {
            return error.ChecksumMismatch;
        }
    } else {
        try writer.writeAll("Warning: neither the catalog nor upstream provided a sha256; installing without integrity verification.\n");
    }

    std.Io.Dir.deleteFileAbsolute(host_io.io, final_path) catch {};
    try std.Io.Dir.renameAbsolute(partial_path, final_path, host_io.io);
    const inspection = try loader_mod.inspectModel(final_path, allocator);
    try writeManifest(manifest_path, entry, stat.size, estimateRequiredBytes(inspection), inspection.offloadable_tensor_bytes, allocator);
    if (observer) |obs| {
        if (obs.on_complete) |cb| cb(obs.context, stat.size);
    }
    if (expected_sha.len > 0) {
        try writer.print("sha256 verified\nInstalled: {s}\n", .{final_path});
    } else {
        try writer.print("Installed (unverified): {s}\n", .{final_path});
    }
}

/// Converts a byte count to gibibytes (GiB) as a floating-point value.
pub fn bytesToGiB(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1024.0 * 1024.0 * 1024.0);
}

fn writeDownloadProgress(
    writer: anytype,
    downloaded_bytes: u64,
    total_bytes: ?u64,
    elapsed_ns: u64,
    done: bool,
) !void {
    var bar_storage: [progress_bar_width]u8 = undefined;
    const bar = buildProgressBar(&bar_storage, downloaded_bytes, total_bytes);
    const elapsed_seconds = if (elapsed_ns == 0) 0.0 else @as(f64, @floatFromInt(elapsed_ns)) / @as(f64, std.time.ns_per_s);
    const mib_per_second = if (elapsed_seconds > 0.0)
        (@as(f64, @floatFromInt(downloaded_bytes)) / (1024.0 * 1024.0)) / elapsed_seconds
    else
        0.0;

    if (total_bytes) |total| {
        const percent = if (total == 0) 100.0 else (@as(f64, @floatFromInt(downloaded_bytes)) * 100.0) / @as(f64, @floatFromInt(total));
        try writer.print(
            "\rDownloading [{s}] {d: >5.1}%  {d:.2}/{d:.2} GiB  {d:.1} MiB/s",
            .{
                bar,
                @min(percent, 100.0),
                bytesToGiB(downloaded_bytes),
                bytesToGiB(total),
                mib_per_second,
            },
        );
    } else {
        try writer.print(
            "\rDownloading [{s}]  {d:.2} GiB  {d:.1} MiB/s",
            .{
                bar,
                bytesToGiB(downloaded_bytes),
                mib_per_second,
            },
        );
    }

    if (done) {
        try writer.writeAll("\n");
    }
    try writer.flush();
}

fn buildProgressBar(storage: *[progress_bar_width]u8, downloaded_bytes: u64, total_bytes: ?u64) []const u8 {
    const ratio: f64 = if (total_bytes) |total|
        if (total == 0)
            1.0
        else
            @min(@as(f64, 1.0), @as(f64, @floatFromInt(downloaded_bytes)) / @as(f64, @floatFromInt(total)))
    else if (downloaded_bytes == 0)
        0.0
    else
        0.35;
    const filled = @min(progress_bar_width, @as(usize, @intFromFloat(@floor(ratio * @as(f64, progress_bar_width)))));

    for (storage, 0..) |*slot, idx| {
        slot.* = if (idx < filled) '=' else ' ';
    }
    if (filled > 0 and filled < progress_bar_width) {
        storage[filled - 1] = '>';
    } else if (filled == progress_bar_width and progress_bar_width > 0) {
        storage[progress_bar_width - 1] = '=';
    }
    return storage[0..];
}

fn writeManifest(
    path: []const u8,
    entry: catalog.CatalogEntry,
    size_bytes: u64,
    required_vram_bytes: u64,
    offloadable_vram_bytes: u64,
    allocator: std.mem.Allocator,
) !void {
    _ = allocator;
    try ensureParentDir(path);
    const file = try std.Io.Dir.createFileAbsolute(host_io.io, path, .{ .truncate = true });
    defer {
        var close_file = file;
        close_file.close(host_io.io);
    }
    var file_buffer: [2048]u8 = undefined;
    var file_writer = file.writerStreaming(host_io.io, &file_buffer);
    try file_writer.interface.print(
        \\{{"id":"{s}","display_name":"{s}","installed_at_unix":{d},"size_bytes":{d},"required_vram_bytes":{d},"offloadable_vram_bytes":{d},"sha256":"{s}","download_url":"{s}"}}
    , .{
        entry.id,
        entry.display_name,
        host_io.timestamp(),
        size_bytes,
        required_vram_bytes,
        offloadable_vram_bytes,
        entry.sha256,
        entry.download_url,
    });
    try file_writer.interface.flush();
}

fn readInstalledManifest(path: []const u8, allocator: std.mem.Allocator) !?InstalledManifest {
    const file = std.Io.Dir.openFileAbsolute(host_io.io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer {
        var close_file = file;
        close_file.close(host_io.io);
    }

    const data = try host_io.readToEndAlloc(file, allocator, 4096);
    defer allocator.free(data);

    const size_i64 = extractJsonI64Field(data, "size_bytes") orelse return error.InvalidManifest;
    const sha256_opt = extractJsonStringField(data, "sha256");
    const required_vram_i64 = extractJsonI64Field(data, "required_vram_bytes");
    const offloadable_vram_i64 = extractJsonI64Field(data, "offloadable_vram_bytes");
    if (size_i64 < 0) return error.InvalidManifest;
    if (required_vram_i64) |v| if (v < 0) return error.InvalidManifest;
    if (offloadable_vram_i64) |v| if (v < 0) return error.InvalidManifest;

    return .{
        .size_bytes = @intCast(size_i64),
        .sha256 = if (sha256_opt) |s| try allocator.dupe(u8, s) else null,
        .required_vram_bytes = if (required_vram_i64) |v| @intCast(v) else null,
        .offloadable_vram_bytes = if (offloadable_vram_i64) |v| @intCast(v) else null,
    };
}

fn computeFileSha256Hex(path: []const u8, allocator: std.mem.Allocator) ![]u8 {
    const file = try std.Io.Dir.openFileAbsolute(host_io.io, path, .{});
    defer {
        var close_file = file;
        close_file.close(host_io.io);
    }

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [64 * 1024]u8 = undefined;
    while (true) {
        const n = try host_io.read(file, &buf);
        if (n == 0) break;
        hasher.update(buf[0..n]);
    }

    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    return allocator.dupe(u8, &hex);
}

fn resolveInstalledModelDir(model_id: []const u8, allocator: std.mem.Allocator) ![]u8 {
    if (!isValidModelId(model_id)) return error.InvalidModelId;
    var paths = try runtimePaths(allocator);
    defer paths.deinit(allocator);
    return try std.fs.path.join(allocator, &.{ paths.cache_root, "models", model_id });
}

fn removeInstalledModelAtPaths(model_dir: []const u8, model_path: []const u8, manifest_path: []const u8) !RemoveInstalledModelResult {
    const had_model = pathExistsAbsolute(model_path);
    const had_manifest = pathExistsAbsolute(manifest_path);
    if (!had_model and !had_manifest) return error.ModelNotInstalled;

    return .{
        .deleted_model = try deleteFileIfExistsAbsolute(model_path),
        .deleted_manifest = try deleteFileIfExistsAbsolute(manifest_path),
        .removed_dir = deleteDirIfEmptyAbsolute(model_dir) catch |err| switch (err) {
            error.FileNotFound => false,
            error.DirNotEmpty => false,
            else => return err,
        },
    };
}

fn resolveCacheRoot(allocator: std.mem.Allocator) ![]u8 {
    const xdg_cache = host_io.getenv("XDG_CACHE_HOME");
    const home = host_io.getenv("HOME");
    return resolveCacheRootForEnv(allocator, builtin.os.tag, xdg_cache, home);
}

fn resolveConfigRoot(allocator: std.mem.Allocator) ![]u8 {
    const xdg_config = host_io.getenv("XDG_CONFIG_HOME");
    const home = host_io.getenv("HOME");
    return resolveConfigRootForEnv(allocator, builtin.os.tag, xdg_config, home);
}

fn resolveCacheRootForEnv(
    allocator: std.mem.Allocator,
    os_tag: std.Target.Os.Tag,
    xdg_cache_home: ?[:0]const u8,
    home: ?[:0]const u8,
) ![]u8 {
    if (xdg_cache_home) |xdg| {
        return std.fs.path.join(allocator, &.{ xdg, "zinc", "models" });
    }
    const home_value = home orelse return error.MissingHomeDirectory;
    return switch (os_tag) {
        .macos => std.fs.path.join(allocator, &.{ home_value, "Library", "Caches", "zinc", "models" }),
        else => std.fs.path.join(allocator, &.{ home_value, ".cache", "zinc", "models" }),
    };
}

fn resolveConfigRootForEnv(
    allocator: std.mem.Allocator,
    os_tag: std.Target.Os.Tag,
    xdg_config_home: ?[:0]const u8,
    home: ?[:0]const u8,
) ![]u8 {
    if (xdg_config_home) |xdg| {
        return std.fs.path.join(allocator, &.{ xdg, "zinc" });
    }
    const home_value = home orelse return error.MissingHomeDirectory;
    return switch (os_tag) {
        .macos => std.fs.path.join(allocator, &.{ home_value, "Library", "Application Support", "zinc" }),
        else => std.fs.path.join(allocator, &.{ home_value, ".config", "zinc" }),
    };
}

fn ensureParentDir(path: []const u8) !void {
    const parent = std.fs.path.dirname(path) orelse return;
    try std.Io.Dir.cwd().createDirPath(host_io.io, parent);
}

fn pathExistsAbsolute(path: []const u8) bool {
    std.Io.Dir.accessAbsolute(host_io.io, path, .{}) catch return false;
    return true;
}

fn dirExistsAbsolute(path: []const u8) bool {
    const dir = std.Io.Dir.openDirAbsolute(host_io.io, path, .{}) catch return false;
    defer {
        var close_dir = dir;
        close_dir.close(host_io.io);
    }
    return true;
}

fn deleteFileIfExistsAbsolute(path: []const u8) !bool {
    std.Io.Dir.deleteFileAbsolute(host_io.io, path) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    return true;
}

fn deleteDirIfEmptyAbsolute(path: []const u8) !bool {
    std.Io.Dir.deleteDirAbsolute(host_io.io, path) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    return true;
}

fn extractJsonStringField(body: []const u8, key: []const u8) ?[]const u8 {
    var needle_buf: [128]u8 = undefined;
    const needle = std.fmt.bufPrint(&needle_buf, "\"{s}\":\"", .{key}) catch return null;
    if (std.mem.indexOf(u8, body, needle)) |pos| {
        const start = pos + needle.len;
        return body[start .. start + (findStringEnd(body[start..]) orelse return null)];
    }
    const spaced = std.fmt.bufPrint(&needle_buf, "\"{s}\": \"", .{key}) catch return null;
    if (std.mem.indexOf(u8, body, spaced)) |pos| {
        const start = pos + spaced.len;
        return body[start .. start + (findStringEnd(body[start..]) orelse return null)];
    }
    return null;
}

fn extractJsonI64Field(body: []const u8, key: []const u8) ?i64 {
    var needle_buf: [128]u8 = undefined;
    const needle = std.fmt.bufPrint(&needle_buf, "\"{s}\":", .{key}) catch return null;
    const pos = std.mem.indexOf(u8, body, needle) orelse return null;
    const start = pos + needle.len;
    const trimmed = std.mem.trimStart(u8, body[start..], " ");
    const end = findNumEnd(trimmed);
    if (end == 0) return null;
    return std.fmt.parseInt(i64, trimmed[0..end], 10) catch null;
}

fn findNumEnd(s: []const u8) usize {
    for (s, 0..) |c, i| {
        if (i == 0 and c == '-') continue;
        if (c < '0' or c > '9') return i;
    }
    return s.len;
}

fn findStringEnd(s: []const u8) ?usize {
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] == '\\') {
            i += 1;
            continue;
        }
        if (s[i] == '"') return i;
    }
    return null;
}

test "isValidModelId accepts catalog and hf cache ids" {
    try std.testing.expect(isValidModelId("qwen35-9b-q4k-m"));
    try std.testing.expect(isValidModelId("hf--unsloth-qwen3.5-9b-gguf--q4_k_m"));
}

test "isValidModelId rejects ids that are not a single safe path component" {
    try std.testing.expect(!isValidModelId(""));
    try std.testing.expect(!isValidModelId("."));
    try std.testing.expect(!isValidModelId(".."));
    try std.testing.expect(!isValidModelId("../../../home/user"));
    try std.testing.expect(!isValidModelId("a/b"));
    try std.testing.expect(!isValidModelId("a\\b"));
    try std.testing.expect(!isValidModelId(".hidden"));
    try std.testing.expect(!isValidModelId("-flag-like"));
    try std.testing.expect(!isValidModelId("id with space"));
    try std.testing.expect(!isValidModelId(&host_io.repeatText("x", 129)));
}

test "model paths cannot be resolved for traversal ids" {
    try std.testing.expectError(error.InvalidModelId, resolveInstalledModelPath("../escape", std.testing.allocator));
    try std.testing.expectError(error.InvalidModelId, resolveManifestPath("../escape", std.testing.allocator));
    try std.testing.expectError(error.InvalidModelId, resolveInstalledModelDir("../escape", std.testing.allocator));
    try std.testing.expect(!isInstalled("../escape", std.testing.allocator));
    try std.testing.expectError(error.InvalidModelId, removeInstalledModel("../escape", std.testing.allocator));
}

test "resolve cache root prefers XDG cache home" {
    const path = try resolveCacheRootForEnv(std.testing.allocator, .linux, "/tmp/cache", "/Users/example");
    defer std.testing.allocator.free(path);
    try std.testing.expectEqualStrings("/tmp/cache/zinc/models", path);
}

test "resolve config root falls back to home on linux" {
    const path = try resolveConfigRootForEnv(std.testing.allocator, .linux, null, "/home/zinc");
    defer std.testing.allocator.free(path);
    try std.testing.expectEqualStrings("/home/zinc/.config/zinc", path);
}

test "resolve config root uses application support on macos" {
    const path = try resolveConfigRootForEnv(std.testing.allocator, .macos, null, "/Users/zinc");
    defer std.testing.allocator.free(path);
    try std.testing.expectEqualStrings("/Users/zinc/Library/Application Support/zinc", path);
}

test "active selection roundtrip via explicit config path" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_root = try tmp.dir.realPathFileAlloc(host_io.io, ".", std.testing.allocator);
    defer std.testing.allocator.free(config_root);
    const config_path = try std.fs.path.join(std.testing.allocator, &.{ config_root, "active-model.json" });
    defer std.testing.allocator.free(config_path);

    try ensureParentDir(config_path);
    const file = try std.Io.Dir.createFileAbsolute(host_io.io, config_path, .{ .truncate = true });
    defer {
        var close_file = file;
        close_file.close(host_io.io);
    }
    try file.writeStreamingAll(host_io.io, "{\"active_model_id\":\"qwen35-9b-q4k-m\",\"selected_at_unix\":42}");

    const opened = try std.Io.Dir.openFileAbsolute(host_io.io, config_path, .{});
    defer {
        var close_file = opened;
        close_file.close(host_io.io);
    }
    const data = try host_io.readToEndAlloc(opened, std.testing.allocator, 256);
    defer std.testing.allocator.free(data);
    try std.testing.expectEqualStrings("qwen35-9b-q4k-m", extractJsonStringField(data, "active_model_id").?);
    try std.testing.expectEqual(@as(?i64, 42), extractJsonI64Field(data, "selected_at_unix"));
}

test "extractJsonI64Field reads positive integer fields" {
    const body = "{\"vram_budget_bytes\":34359738368,\"cached_at_unix\":123}";
    try std.testing.expectEqual(@as(?i64, 34_359_738_368), extractJsonI64Field(body, "vram_budget_bytes"));
    try std.testing.expectEqual(@as(?i64, 123), extractJsonI64Field(body, "cached_at_unix"));
}

test "buildProgressBar reflects partial and complete downloads" {
    var partial: [progress_bar_width]u8 = undefined;
    const partial_bar = buildProgressBar(&partial, 50, 100);
    try std.testing.expectEqual(@as(usize, progress_bar_width), partial_bar.len);
    try std.testing.expect(std.mem.indexOfScalar(u8, partial_bar, '>') != null);

    var full: [progress_bar_width]u8 = undefined;
    const full_bar = buildProgressBar(&full, 100, 100);
    try std.testing.expectEqual(@as(usize, progress_bar_width), full_bar.len);
    try std.testing.expect(std.mem.indexOfScalar(u8, full_bar, ' ') == null);
}

test "extractUpstreamArtifactMetadata reads x-linked headers from redirect response" {
    const response_bytes =
        "HTTP/1.1 302 Found\r\n" ++
        "x-linked-size: 22241950336\r\n" ++
        "x-linked-etag: \"1b0ac637dfa092bbba2793977db9485a40c4f8b42df5fe342f0076d61b66ae83\"\r\n" ++
        "location: https://example.invalid/file.gguf\r\n\r\n";

    const head = try std.http.Client.Response.Head.parse(response_bytes);
    const metadata = try extractUpstreamArtifactMetadata(head);

    try std.testing.expectEqual(@as(?u64, 22_241_950_336), metadata.size_bytes);
    try std.testing.expect(metadata.sha256_hex != null);
    try std.testing.expectEqualStrings(
        "1b0ac637dfa092bbba2793977db9485a40c4f8b42df5fe342f0076d61b66ae83",
        metadata.sha256_hex.?[0..],
    );
}

test "preflightCatalogDrift fails fast on upstream drift" {
    const entry = catalog.find("qwen35-9b-q4k-m") orelse return error.TestExpectedEqual;
    const drifted = UpstreamArtifactMetadata{
        .size_bytes = entry.size_bytes + 1,
        .sha256_hex = canonicalizeSha256Header("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa").?,
    };

    const Sink = struct {
        fn print(_: *@This(), comptime _: []const u8, _: anytype) !void {}
        fn writeAll(_: *@This(), _: []const u8) !void {}
    };
    var sink = Sink{};

    try std.testing.expectError(error.CatalogMetadataMismatch, preflightCatalogDrift(entry.*, drifted, &sink));
}

test "preflightCatalogDrift allows size-only drift" {
    const entry = catalog.find("qwen35-9b-q4k-m") orelse return error.TestExpectedEqual;
    const drifted = UpstreamArtifactMetadata{
        .size_bytes = entry.size_bytes + 1,
        .sha256_hex = canonicalizeSha256Header(entry.sha256).?,
    };

    const Sink = struct {
        fn print(_: *@This(), comptime _: []const u8, _: anytype) !void {}
        fn writeAll(_: *@This(), _: []const u8) !void {}
    };
    var sink = Sink{};

    try preflightCatalogDrift(entry.*, drifted, &sink);
}

test "removeInstalledModelAtPaths deletes known artifacts and empty dir" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realPathFileAlloc(host_io.io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root);
    const model_dir = try std.fs.path.join(std.testing.allocator, &.{ root, "models", "qwen35-9b-q4k-m" });
    defer std.testing.allocator.free(model_dir);
    try std.Io.Dir.cwd().createDirPath(host_io.io, model_dir);

    const model_path = try std.fs.path.join(std.testing.allocator, &.{ model_dir, "model.gguf" });
    defer std.testing.allocator.free(model_path);
    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ model_dir, "manifest.json" });
    defer std.testing.allocator.free(manifest_path);

    {
        const file = try std.Io.Dir.createFileAbsolute(host_io.io, model_path, .{});
        defer {
            var close_file = file;
            close_file.close(host_io.io);
        }
    }
    {
        const file = try std.Io.Dir.createFileAbsolute(host_io.io, manifest_path, .{});
        defer {
            var close_file = file;
            close_file.close(host_io.io);
        }
    }

    const result = try removeInstalledModelAtPaths(model_dir, model_path, manifest_path);
    try std.testing.expect(result.deleted_model);
    try std.testing.expect(result.deleted_manifest);
    try std.testing.expect(result.removed_dir);
    try std.testing.expect(!pathExistsAbsolute(model_path));
    try std.testing.expect(!pathExistsAbsolute(manifest_path));
    try std.testing.expect(!dirExistsAbsolute(model_dir));
}

test "removeInstalledModelAtPaths keeps non-empty directory" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const root = try tmp.dir.realPathFileAlloc(host_io.io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root);
    const model_dir = try std.fs.path.join(std.testing.allocator, &.{ root, "models", "qwen35-9b-q4k-m" });
    defer std.testing.allocator.free(model_dir);
    try std.Io.Dir.cwd().createDirPath(host_io.io, model_dir);

    const model_path = try std.fs.path.join(std.testing.allocator, &.{ model_dir, "model.gguf" });
    defer std.testing.allocator.free(model_path);
    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ model_dir, "manifest.json" });
    defer std.testing.allocator.free(manifest_path);
    const extra_path = try std.fs.path.join(std.testing.allocator, &.{ model_dir, "notes.txt" });
    defer std.testing.allocator.free(extra_path);

    for ([_][]const u8{ model_path, manifest_path, extra_path }) |path| {
        const file = try std.Io.Dir.createFileAbsolute(host_io.io, path, .{});
        defer {
            var close_file = file;
            close_file.close(host_io.io);
        }
    }

    const result = try removeInstalledModelAtPaths(model_dir, model_path, manifest_path);
    try std.testing.expect(result.deleted_model);
    try std.testing.expect(result.deleted_manifest);
    try std.testing.expect(!result.removed_dir);
    try std.testing.expect(pathExistsAbsolute(extra_path));
    try std.testing.expect(dirExistsAbsolute(model_dir));
}

test "active selection pointing to non-catalog model is detectable" {
    // Simulates the scenario where active-model.json references a model
    // that is no longer in the catalog.
    const stale_id = "nonexistent-fixture-model";
    try std.testing.expect(catalog.find(stale_id) == null);

    // A valid catalog model should be findable.
    const valid_id = "qwen35-9b-q4k-m";
    try std.testing.expect(catalog.find(valid_id) != null);
}

test "active selection roundtrip rejects non-catalog model on validate" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();

    const config_root = try tmp.dir.realPathFileAlloc(host_io.io, ".", std.testing.allocator);
    defer std.testing.allocator.free(config_root);
    const config_path = try std.fs.path.join(std.testing.allocator, &.{ config_root, "active-model.json" });
    defer std.testing.allocator.free(config_path);

    try ensureParentDir(config_path);
    {
        const file = try std.Io.Dir.createFileAbsolute(host_io.io, config_path, .{ .truncate = true });
        defer {
            var close_file = file;
            close_file.close(host_io.io);
        }
        try file.writeStreamingAll(host_io.io, "{\"active_model_id\":\"nonexistent-fixture-model\",\"selected_at_unix\":42}");
    }

    // Read back — the selection is parseable but points to a removed model.
    const opened = try std.Io.Dir.openFileAbsolute(host_io.io, config_path, .{});
    defer {
        var close_file = opened;
        close_file.close(host_io.io);
    }
    const data = try host_io.readToEndAlloc(opened, std.testing.allocator, 256);
    defer std.testing.allocator.free(data);

    const model_id = extractJsonStringField(data, "active_model_id").?;
    try std.testing.expectEqualStrings("nonexistent-fixture-model", model_id);
    // The model is not in the catalog — this is the check that should happen at startup.
    try std.testing.expect(catalog.find(model_id) == null);
}

test "writeManifest then readInstalledManifest round-trips offloadable_vram_bytes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realPathFileAlloc(host_io.io, ".", std.testing.allocator);
    defer std.testing.allocator.free(dir_path);
    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ dir_path, "manifest.json" });
    defer std.testing.allocator.free(manifest_path);

    const entry = catalog.find("qwen36-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    try writeManifest(manifest_path, entry.*, 22_360_456_160, 23_106_019_926, 19_327_352_832, std.testing.allocator);

    var manifest = (try readInstalledManifest(manifest_path, std.testing.allocator)) orelse return error.TestExpectedEqual;
    defer manifest.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u64, 22_360_456_160), manifest.size_bytes);
    try std.testing.expectEqual(@as(?u64, 23_106_019_926), manifest.required_vram_bytes);
    try std.testing.expectEqual(@as(?u64, 19_327_352_832), manifest.offloadable_vram_bytes);
}

test "describeFit uninstalled returns catalog estimate with fit_state populated" {
    // Synthesize a CatalogEntry whose id is unique enough that isInstalled()
    // returns false against any real cache root. This exercises the
    // catalog-estimate branch of describeFit without filesystem fixtures.
    const synthetic = catalog.CatalogEntry{
        .id = "phase-e-test-fixture-not-installed-zzzz",
        .display_name = "Phase E Fixture",
        .release_date = "2099-12-31",
        .family = "qwen3.5",
        .format = "gguf",
        .quantization = "Q4_K_XL",
        .file_name = "fixture.gguf",
        .homepage_url = "",
        .download_url = "",
        .sha256 = "",
        .size_bytes = 22 * 1024 * 1024 * 1024,
        .required_vram_bytes = 22 * 1024 * 1024 * 1024,
        .offloadable_vram_bytes = 18 * 1024 * 1024 * 1024,
        .default_context_length = 4096,
        .recommended_for_chat = false,
        .thinking_stable = true,
        .status = .experimental,
        .tested_profiles = &.{},
    };

    // 16 GiB budget: doesn't fit straight (22 > 16), fits with offload (4 <= 16).
    const fit = try describeFit(synthetic, 16 * 1024 * 1024 * 1024, std.testing.allocator);
    try std.testing.expectEqual(false, fit.fits_current_gpu);
    try std.testing.expectEqual(false, fit.exact);
    try std.testing.expectEqual(catalog.FitState.fits_with_offload, fit.fit_state);
    try std.testing.expectEqual(@as(u64, 22 * 1024 * 1024 * 1024), fit.required_vram_bytes);
    try std.testing.expectEqual(@as(u64, 4 * 1024 * 1024 * 1024), fit.required_vram_with_offload_bytes);

    // 32 GiB budget: fits straight. fit_state collapses to .fits.
    const big_fit = try describeFit(synthetic, 32 * 1024 * 1024 * 1024, std.testing.allocator);
    try std.testing.expectEqual(true, big_fit.fits_current_gpu);
    try std.testing.expectEqual(catalog.FitState.fits, big_fit.fit_state);

    // 2 GiB budget: too small even with offload.
    const small_fit = try describeFit(synthetic, 2 * 1024 * 1024 * 1024, std.testing.allocator);
    try std.testing.expectEqual(false, small_fit.fits_current_gpu);
    try std.testing.expectEqual(catalog.FitState.does_not_fit, small_fit.fit_state);
}

test "readInstalledManifest tolerates old format without offloadable_vram_bytes" {
    // Backward compat: manifests written before the offload field was added
    // must still parse, with offloadable_vram_bytes = null. describeFit then
    // falls back to the catalog estimate for that model.
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir_path = try tmp.dir.realPathFileAlloc(host_io.io, ".", std.testing.allocator);
    defer std.testing.allocator.free(dir_path);
    const manifest_path = try std.fs.path.join(std.testing.allocator, &.{ dir_path, "manifest.json" });
    defer std.testing.allocator.free(manifest_path);

    {
        const file = try std.Io.Dir.createFileAbsolute(host_io.io, manifest_path, .{ .truncate = true });
        defer {
            var close_file = file;
            close_file.close(host_io.io);
        }
        try file.writeStreamingAll(host_io.io,
            \\{"id":"qwen35-9b-q4k-m","display_name":"Qwen 3.5 9B","installed_at_unix":1700000000,"size_bytes":5650000000,"required_vram_bytes":6442450944,"sha256":"","download_url":"local"}
        );
    }

    var manifest = (try readInstalledManifest(manifest_path, std.testing.allocator)) orelse return error.TestExpectedEqual;
    defer manifest.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(u64, 5_650_000_000), manifest.size_bytes);
    try std.testing.expectEqual(@as(?u64, 6_442_450_944), manifest.required_vram_bytes);
    try std.testing.expectEqual(@as(?u64, null), manifest.offloadable_vram_bytes);
}

test "GPU capability caches separate runtime backends even at the same device index" {
    var paths: [4][]u8 = undefined;
    var allocated: usize = 0;
    defer for (paths[0..allocated]) |path| std.testing.allocator.free(path);
    const backends = [_]catalog.CatalogBackend{ .vulkan, .rocm, .cuda, .metal };
    for (backends, 0..) |backend, i| {
        paths[i] = try gpuProfileCachePath("/config", backend, 0, std.testing.allocator);
        allocated += 1;
    }
    try std.testing.expectEqualStrings("/config/gpu-profile-rocm-device-0.json", paths[1]);
    for (paths, 0..) |path, i| {
        for (paths[i + 1 ..]) |other| try std.testing.expect(!std.mem.eql(u8, path, other));
        try std.testing.expect(!std.mem.eql(u8, path, "/config/gpu-profile-device-0.json"));
    }
}

test "GPU cache metadata rejects legacy and different backends and roundtrips quoted device names" {
    const stored = StoredGpuProfile{ .schema_version = 2, .backend = .rocm, .profile = "amd-rdna4-32gb", .device_name = "Radeon \"quoted\" \\ device", .vram_budget_bytes = 30 * 1024 * 1024 * 1024, .cached_at_unix = 42 };
    const data = try std.json.Stringify.valueAlloc(std.testing.allocator, stored, .{});
    defer std.testing.allocator.free(data);
    var cached = (try parseGpuProfileCache(data, .rocm, std.testing.allocator)) orelse return error.TestExpectedEqual;
    defer cached.deinit(std.testing.allocator);
    try std.testing.expectEqualStrings(stored.profile, cached.profile);
    try std.testing.expectEqualStrings(stored.device_name, cached.device_name);
    try std.testing.expectEqual(stored.vram_budget_bytes, cached.vram_budget_bytes);
    for ([_]catalog.CatalogBackend{ .vulkan, .cuda, .metal }) |backend| {
        try std.testing.expect((try parseGpuProfileCache(data, backend, std.testing.allocator)) == null);
    }
    for ([_][]const u8{
        "{\"profile\":\"nvidia-cuda\",\"device_name\":\"old\",\"vram_budget_bytes\":1,\"cached_at_unix\":42}",
        "{\"schema_version\":1,\"backend\":\"rocm\",\"profile\":\"old\",\"device_name\":\"old\",\"vram_budget_bytes\":1,\"cached_at_unix\":42}",
        "{partial",
        "{}",
    }) |invalid| {
        try std.testing.expect((try parseGpuProfileCache(invalid, .rocm, std.testing.allocator)) == null);
    }
}
