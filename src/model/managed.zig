//! Managed model cache, active-model selection, and download helpers.
//! @section Managed Models
//! These helpers back the `zinc model ...` CLI and the server-side active-model
//! switching flow.
const std = @import("std");
const builtin = @import("builtin");
const catalog = @import("catalog.zig");
const diagnostics = @import("../diagnostics.zig");
const loader_mod = @import("loader.zig");

pub const RuntimePaths = struct {
    cache_root: []u8,
    config_root: []u8,

    pub fn deinit(self: *RuntimePaths, allocator: std.mem.Allocator) void {
        allocator.free(self.cache_root);
        allocator.free(self.config_root);
        self.* = undefined;
    }
};

pub const ModelFit = struct {
    required_vram_bytes: u64,
    fits_current_gpu: bool,
    exact: bool,
};

pub const ActiveSelection = struct {
    model_id: []u8,
    selected_at_unix: i64,

    pub fn deinit(self: *ActiveSelection, allocator: std.mem.Allocator) void {
        allocator.free(self.model_id);
        self.* = undefined;
    }
};

pub const CachedGpuProfile = struct {
    profile: []u8,
    device_name: []u8,
    vram_budget_bytes: u64,
    cached_at_unix: i64,

    pub fn deinit(self: *CachedGpuProfile, allocator: std.mem.Allocator) void {
        allocator.free(self.profile);
        allocator.free(self.device_name);
        self.* = undefined;
    }
};

pub const InstalledManifest = struct {
    size_bytes: u64,
    sha256: []u8,
    required_vram_bytes: ?u64,

    pub fn deinit(self: *InstalledManifest, allocator: std.mem.Allocator) void {
        allocator.free(self.sha256);
        self.* = undefined;
    }
};

pub const DownloadObserver = struct {
    context: ?*anyopaque = null,
    on_start: ?*const fn (context: ?*anyopaque, total_bytes: ?u64) void = null,
    on_progress: ?*const fn (context: ?*anyopaque, downloaded_bytes: u64, total_bytes: ?u64) void = null,
    on_verifying: ?*const fn (context: ?*anyopaque, downloaded_bytes: u64) void = null,
    on_complete: ?*const fn (context: ?*anyopaque, downloaded_bytes: u64) void = null,
};

const progress_bar_width = 28;
const progress_update_interval_ns = 150 * std.time.ns_per_ms;

pub fn runtimePaths(allocator: std.mem.Allocator) !RuntimePaths {
    return .{
        .cache_root = try resolveCacheRoot(allocator),
        .config_root = try resolveConfigRoot(allocator),
    };
}

pub fn resolveInstalledModelPath(model_id: []const u8, allocator: std.mem.Allocator) ![]u8 {
    var paths = try runtimePaths(allocator);
    defer paths.deinit(allocator);
    return try std.fs.path.join(allocator, &.{ paths.cache_root, "models", model_id, "model.gguf" });
}

pub fn resolveManifestPath(model_id: []const u8, allocator: std.mem.Allocator) ![]u8 {
    var paths = try runtimePaths(allocator);
    defer paths.deinit(allocator);
    return try std.fs.path.join(allocator, &.{ paths.cache_root, "models", model_id, "manifest.json" });
}

pub fn resolveActiveConfigPath(allocator: std.mem.Allocator) ![]u8 {
    const root = try resolveConfigRoot(allocator);
    defer allocator.free(root);
    return try std.fs.path.join(allocator, &.{ root, "active-model.json" });
}

pub fn resolveGpuProfileCachePath(device_index: u32, allocator: std.mem.Allocator) ![]u8 {
    const root = try resolveConfigRoot(allocator);
    defer allocator.free(root);
    return try std.fmt.allocPrint(allocator, "{s}/gpu-profile-device-{d}.json", .{ root, device_index });
}

pub fn isInstalled(model_id: []const u8, allocator: std.mem.Allocator) bool {
    const path = resolveInstalledModelPath(model_id, allocator) catch return false;
    defer allocator.free(path);
    std.fs.accessAbsolute(path, .{}) catch return false;
    return true;
}

pub fn readActiveSelection(allocator: std.mem.Allocator) !?ActiveSelection {
    const path = try resolveActiveConfigPath(allocator);
    defer allocator.free(path);

    const file = std.fs.openFileAbsolute(path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer {
        var close_file = file;
        close_file.close();
    }

    const data = try file.readToEndAlloc(allocator, 4096);
    defer allocator.free(data);

    const model_id = extractJsonStringField(data, "active_model_id") orelse return error.InvalidActiveModelConfig;
    const selected_at = extractJsonI64Field(data, "selected_at_unix") orelse 0;

    return .{
        .model_id = try allocator.dupe(u8, model_id),
        .selected_at_unix = selected_at,
    };
}

pub fn writeActiveSelection(model_id: []const u8, allocator: std.mem.Allocator) !void {
    const path = try resolveActiveConfigPath(allocator);
    defer allocator.free(path);
    try ensureParentDir(path);

    const file = try std.fs.createFileAbsolute(path, .{ .truncate = true });
    defer {
        var close_file = file;
        close_file.close();
    }

    var file_buffer: [1024]u8 = undefined;
    var writer = file.writerStreaming(&file_buffer);
    try writer.interface.print(
        \\{{"active_model_id":"{s}","selected_at_unix":{d}}}
    , .{ model_id, std.time.timestamp() });
    try writer.interface.flush();
}

pub fn readCachedGpuProfile(device_index: u32, allocator: std.mem.Allocator) !?CachedGpuProfile {
    const path = try resolveGpuProfileCachePath(device_index, allocator);
    defer allocator.free(path);

    const file = std.fs.openFileAbsolute(path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer {
        var close_file = file;
        close_file.close();
    }

    const data = try file.readToEndAlloc(allocator, 4096);
    defer allocator.free(data);

    const profile = extractJsonStringField(data, "profile") orelse return error.InvalidGpuProfileCache;
    const device_name = extractJsonStringField(data, "device_name") orelse return error.InvalidGpuProfileCache;
    const vram_budget_i64 = extractJsonI64Field(data, "vram_budget_bytes") orelse return error.InvalidGpuProfileCache;
    const cached_at = extractJsonI64Field(data, "cached_at_unix") orelse 0;
    if (vram_budget_i64 < 0) return error.InvalidGpuProfileCache;

    return .{
        .profile = try allocator.dupe(u8, profile),
        .device_name = try allocator.dupe(u8, device_name),
        .vram_budget_bytes = @intCast(vram_budget_i64),
        .cached_at_unix = cached_at,
    };
}

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

    const file = try std.fs.createFileAbsolute(path, .{ .truncate = true });
    defer {
        var close_file = file;
        close_file.close();
    }

    var file_buffer: [2048]u8 = undefined;
    var writer = file.writerStreaming(&file_buffer);
    try writer.interface.print(
        \\{{"profile":"{s}","device_name":"{s}","vram_budget_bytes":{d},"cached_at_unix":{d}}}
    , .{
        profile,
        device_name,
        vram_budget_bytes,
        std.time.timestamp(),
    });
    try writer.interface.flush();
}

pub fn describeFit(entry: catalog.CatalogEntry, vram_budget_bytes: u64, allocator: std.mem.Allocator) !ModelFit {
    if (isInstalled(entry.id, allocator)) {
        const installed_path = try resolveInstalledModelPath(entry.id, allocator);
        defer allocator.free(installed_path);
        const manifest_path = try resolveManifestPath(entry.id, allocator);
        defer allocator.free(manifest_path);

        if (try readInstalledManifest(manifest_path, allocator)) |manifest| {
            defer {
                var owned = manifest;
                owned.deinit(allocator);
            }
            if (manifest.required_vram_bytes) |required_vram_bytes| {
                return .{
                    .required_vram_bytes = required_vram_bytes,
                    .fits_current_gpu = required_vram_bytes <= vram_budget_bytes,
                    .exact = true,
                };
            }
        }

        const inspection = try loader_mod.inspectModel(installed_path, allocator);
        const fit = diagnostics.estimateFit(inspection, vram_budget_bytes);
        try writeManifest(manifest_path, entry, inspection.file_size, fit.total_device_local_bytes, allocator);
        return .{
            .required_vram_bytes = fit.total_device_local_bytes,
            .fits_current_gpu = fit.total_device_local_bytes <= vram_budget_bytes,
            .exact = true,
        };
    }

    return .{
        .required_vram_bytes = entry.required_vram_bytes,
        .fits_current_gpu = catalog.fitsGpu(entry, vram_budget_bytes),
        .exact = false,
    };
}

pub fn verifyActiveSelectionFits(model_id: []const u8, vram_budget_bytes: u64, allocator: std.mem.Allocator) !ModelFit {
    const entry = catalog.find(model_id) orelse return error.UnknownManagedModel;
    if (!isInstalled(model_id, allocator)) return error.ModelNotInstalled;
    return describeFit(entry.*, vram_budget_bytes, allocator);
}

pub fn pullModel(entry: catalog.CatalogEntry, allocator: std.mem.Allocator, writer: anytype) !void {
    try pullModelWithObserver(entry, allocator, writer, null);
}

pub fn pullModelWithObserver(
    entry: catalog.CatalogEntry,
    allocator: std.mem.Allocator,
    writer: anytype,
    observer: ?*const DownloadObserver,
) !void {
    var paths = try runtimePaths(allocator);
    defer paths.deinit(allocator);

    const models_dir = try std.fs.path.join(allocator, &.{ paths.cache_root, "models", entry.id });
    defer allocator.free(models_dir);
    const final_path = try std.fs.path.join(allocator, &.{ models_dir, "model.gguf" });
    defer allocator.free(final_path);
    const manifest_path = try std.fs.path.join(allocator, &.{ models_dir, "manifest.json" });
    defer allocator.free(manifest_path);

    if (isInstalled(entry.id, allocator)) {
        const actual_sha = try computeFileSha256Hex(final_path, allocator);
        defer allocator.free(actual_sha);
        if (std.ascii.eqlIgnoreCase(actual_sha, entry.sha256)) {
            try writer.print("Already installed: {s}\n", .{final_path});
            return;
        }
        try writer.print("Cached file checksum mismatch, replacing: {s}\n", .{final_path});
        std.fs.deleteFileAbsolute(final_path) catch {};
    }

    const downloads_dir = try std.fs.path.join(allocator, &.{ paths.cache_root, "downloads" });
    defer allocator.free(downloads_dir);
    try std.fs.cwd().makePath(downloads_dir);
    try std.fs.cwd().makePath(models_dir);

    const partial_name = try std.fmt.allocPrint(allocator, "{s}.partial", .{entry.id});
    defer allocator.free(partial_name);
    const partial_path = try std.fs.path.join(allocator, &.{ downloads_dir, partial_name });
    defer allocator.free(partial_path);
    std.fs.deleteFileAbsolute(partial_path) catch {};

    try writer.print("Resolving model: {s}\n", .{entry.id});
    try writer.print("Downloading: {s}\n", .{entry.download_url});

    const partial_file = try std.fs.createFileAbsolute(partial_path, .{ .truncate = true });
    defer {
        var close_file = partial_file;
        close_file.close();
    }
    errdefer std.fs.deleteFileAbsolute(partial_path) catch {};

    var file_buffer: [16 * 1024]u8 = undefined;
    var file_writer = partial_file.writerStreaming(&file_buffer);

    var client: std.http.Client = .{ .allocator = allocator };
    defer client.deinit();

    const uri = try std.Uri.parse(entry.download_url);
    var req = try client.request(.GET, uri, .{});
    defer req.deinit();
    try req.sendBodiless();

    var response = try req.receiveHead(&.{});
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
    var progress_timer = try std.time.Timer.start();
    var last_progress_ns: u64 = 0;
    var progress_started = false;

    try writeDownloadProgress(writer, 0, total_bytes, 0, false);
    progress_started = true;

    while (true) {
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

    const stat = try partial_file.stat();
    try writer.writeAll("Verifying sha256...\n");
    if (observer) |obs| {
        if (obs.on_verifying) |cb| cb(obs.context, stat.size);
    }

    const actual_sha = try computeFileSha256Hex(partial_path, allocator);
    defer allocator.free(actual_sha);
    if (!std.ascii.eqlIgnoreCase(actual_sha, entry.sha256)) {
        return error.ChecksumMismatch;
    }

    std.fs.deleteFileAbsolute(final_path) catch {};
    try std.fs.renameAbsolute(partial_path, final_path);
    const inspection = try loader_mod.inspectModel(final_path, allocator);
    const fit = diagnostics.estimateFit(inspection, std.math.maxInt(u64));
    try writeManifest(manifest_path, entry, stat.size, fit.total_device_local_bytes, allocator);
    if (observer) |obs| {
        if (obs.on_complete) |cb| cb(obs.context, stat.size);
    }
    try writer.print("sha256 verified\nInstalled: {s}\n", .{final_path});
}

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
    allocator: std.mem.Allocator,
) !void {
    _ = allocator;
    try ensureParentDir(path);
    const file = try std.fs.createFileAbsolute(path, .{ .truncate = true });
    defer {
        var close_file = file;
        close_file.close();
    }
    var file_buffer: [2048]u8 = undefined;
    var writer = file.writerStreaming(&file_buffer);
    try writer.interface.print(
        \\{{"id":"{s}","display_name":"{s}","installed_at_unix":{d},"size_bytes":{d},"required_vram_bytes":{d},"sha256":"{s}","download_url":"{s}"}}
    , .{
        entry.id,
        entry.display_name,
        std.time.timestamp(),
        size_bytes,
        required_vram_bytes,
        entry.sha256,
        entry.download_url,
    });
    try writer.interface.flush();
}

fn readInstalledManifest(path: []const u8, allocator: std.mem.Allocator) !?InstalledManifest {
    const file = std.fs.openFileAbsolute(path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer {
        var close_file = file;
        close_file.close();
    }

    const data = try file.readToEndAlloc(allocator, 4096);
    defer allocator.free(data);

    const size_i64 = extractJsonI64Field(data, "size_bytes") orelse return error.InvalidManifest;
    const sha256 = extractJsonStringField(data, "sha256") orelse return error.InvalidManifest;
    const required_vram_i64 = extractJsonI64Field(data, "required_vram_bytes");
    if (size_i64 < 0) return error.InvalidManifest;
    if (required_vram_i64) |v| if (v < 0) return error.InvalidManifest;

    return .{
        .size_bytes = @intCast(size_i64),
        .sha256 = try allocator.dupe(u8, sha256),
        .required_vram_bytes = if (required_vram_i64) |v| @intCast(v) else null,
    };
}

fn computeFileSha256Hex(path: []const u8, allocator: std.mem.Allocator) ![]u8 {
    const file = try std.fs.openFileAbsolute(path, .{});
    defer {
        var close_file = file;
        close_file.close();
    }

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var buf: [64 * 1024]u8 = undefined;
    while (true) {
        const n = try file.read(&buf);
        if (n == 0) break;
        hasher.update(buf[0..n]);
    }

    var digest: [32]u8 = undefined;
    hasher.final(&digest);
    const hex = std.fmt.bytesToHex(digest, .lower);
    return allocator.dupe(u8, &hex);
}

fn resolveCacheRoot(allocator: std.mem.Allocator) ![]u8 {
    const xdg_cache = std.posix.getenv("XDG_CACHE_HOME");
    const home = std.posix.getenv("HOME");
    return resolveCacheRootForEnv(allocator, builtin.os.tag, xdg_cache, home);
}

fn resolveConfigRoot(allocator: std.mem.Allocator) ![]u8 {
    const xdg_config = std.posix.getenv("XDG_CONFIG_HOME");
    const home = std.posix.getenv("HOME");
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
    try std.fs.cwd().makePath(parent);
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
    const trimmed = std.mem.trimLeft(u8, body[start..], " ");
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

    const config_root = try tmp.dir.realpathAlloc(std.testing.allocator, ".");
    defer std.testing.allocator.free(config_root);
    const config_path = try std.fs.path.join(std.testing.allocator, &.{ config_root, "active-model.json" });
    defer std.testing.allocator.free(config_path);

    try ensureParentDir(config_path);
    const file = try std.fs.createFileAbsolute(config_path, .{ .truncate = true });
    defer {
        var close_file = file;
        close_file.close();
    }
    try file.writeAll("{\"active_model_id\":\"qwen35-2b-q4k-m\",\"selected_at_unix\":42}");

    const opened = try std.fs.openFileAbsolute(config_path, .{});
    defer {
        var close_file = opened;
        close_file.close();
    }
    const data = try opened.readToEndAlloc(std.testing.allocator, 256);
    defer std.testing.allocator.free(data);
    try std.testing.expectEqualStrings("qwen35-2b-q4k-m", extractJsonStringField(data, "active_model_id").?);
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
