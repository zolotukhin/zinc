//! Installed model inventory for accelerator HTTP model switching.
//! @section API Server
//!
//! Only the startup model's directory and curated managed-cache locations are
//! inspected. Clients select an advertised ID; they never supply a load path.
const std = @import("std");
const gpu = @import("../gpu/interface.zig");
const catalog = @import("../model/catalog.zig");
const managed = @import("../model/managed.zig");

pub const Entry = struct {
    id: []const u8 = "",
    display_name: []const u8 = "",
    path: []const u8 = "",
    homepage_url: []const u8 = "",
    family: []const u8 = "",
    quantization: []const u8 = "",
    size_bytes: u64 = 0,
    required_vram_bytes: u64 = 0,
    rocm_16gb_context_length: ?u32 = null,
    installed: bool = true,
    managed: bool = false,
    can_activate: bool = true,
    supported_on_current_gpu: bool = true,
    fits_current_gpu: bool = true,
    fit_source: []const u8 = "estimate",

    fn deinit(self: *Entry, allocator: std.mem.Allocator) void {
        inline for (.{ "id", "display_name", "path", "homepage_url", "family", "quantization" }) |field| allocator.free(@field(self, field));
        self.* = .{};
    }
};

pub const Inventory = struct {
    allocator: std.mem.Allocator,
    entries: std.ArrayList(Entry) = .{},
    /// Borrows the ID owned by the startup entry in `entries`.
    initial_model_id: []const u8 = "",

    pub fn init(allocator: std.mem.Allocator, initial_path: []const u8, budget_bytes: u64, rocm_profile: ?[]const u8) !Inventory {
        return collect(allocator, initial_path, budget_bytes, true, rocm_profile);
    }

    /// `include_managed` permits hermetic filesystem tests without inspecting
    /// the user's cache. Production calls `init` and includes installed models.
    pub fn collect(allocator: std.mem.Allocator, initial_path: []const u8, budget_bytes: u64, include_managed: bool, rocm_profile: ?[]const u8) !Inventory {
        var self = Inventory{ .allocator = allocator };
        errdefer self.deinit();
        const absolute = try std.fs.cwd().realpathAlloc(allocator, initial_path);
        defer allocator.free(absolute);
        // The explicitly supplied startup path is authorized by the operator.
        // Canonicalize it first, then admit only the resulting regular file.
        try self.addFile(absolute, budget_bytes, false, true, rocm_profile);
        if (self.entries.items.len == 0) return error.InvalidInitialModel;
        self.initial_model_id = self.entries.items[0].id;

        const parent = std.fs.path.dirname(absolute) orelse return error.InvalidInitialModel;
        var dir = try std.fs.openDirAbsolute(parent, .{ .iterate = true });
        defer dir.close();
        var iterator = dir.iterate();
        while (try iterator.next()) |item| {
            if (item.kind != .file or !isGguf(item.name)) continue;
            const path = try std.fs.path.join(allocator, &.{ parent, item.name });
            defer allocator.free(path);
            try self.addFile(path, budget_bytes, false, false, rocm_profile);
        }
        if (include_managed) {
            for (catalog.entries) |entry| {
                const path = managed.resolveInstalledModelPath(entry.id, allocator) catch |err| switch (err) {
                    error.OutOfMemory => return err,
                    else => continue,
                };
                defer allocator.free(path);
                self.addFile(path, budget_bytes, true, false, rocm_profile) catch |err| switch (err) {
                    error.OutOfMemory => return err,
                    else => continue,
                };
            }
        }
        return self;
    }

    pub fn deinit(self: *Inventory) void {
        for (self.entries.items) |*entry| entry.deinit(self.allocator);
        self.entries.deinit(self.allocator);
        self.initial_model_id = "";
    }

    /// Exact lookup only: strings resembling paths have no special meaning.
    pub fn find(self: *const Inventory, id: []const u8) ?*const Entry {
        for (self.entries.items) |*entry| if (std.mem.eql(u8, entry.id, id)) return entry;
        return null;
    }

    fn addFile(self: *Inventory, path: []const u8, budget_bytes: u64, is_managed: bool, is_initial: bool, rocm_profile: ?[]const u8) !void {
        if (!isGguf(std.fs.path.basename(path))) return;
        const size = regularFileSize(path) catch return;
        const canonical = try std.fs.cwd().realpathAlloc(self.allocator, path);
        defer self.allocator.free(canonical);
        for (self.entries.items) |entry| if (std.mem.eql(u8, entry.path, canonical)) return;
        const known = catalog.findForLoadedModel(null, canonical, "");
        if (!gpu.is_rocm and !is_initial) {
            if (known) |entry| {
                if (std.mem.eql(u8, entry.family, "inkling2")) return;
            }
        }
        var entry = Entry{};
        errdefer entry.deinit(self.allocator);
        entry.id = if (known) |model| try self.allocator.dupe(u8, model.id) else blk: {
            var digest: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(canonical, &digest, .{});
            break :blk try std.fmt.allocPrint(self.allocator, "local-{s}", .{std.fmt.bytesToHex(digest[0..12], .lower)});
        };
        // A curated file in the startup directory takes precedence over another
        // copy in the cache. Either ID always resolves to one recorded path.
        if (self.find(entry.id) != null) {
            entry.deinit(self.allocator);
            return;
        }
        entry.display_name = try self.allocator.dupe(u8, if (known) |model| model.display_name else std.fs.path.basename(canonical));
        entry.path = try self.allocator.dupe(u8, canonical);
        entry.homepage_url = try self.allocator.dupe(u8, if (known) |model| model.homepage_url else "");
        entry.family = try self.allocator.dupe(u8, if (known) |model| model.family else "");
        entry.quantization = try self.allocator.dupe(u8, if (known) |model| model.quantization else "");
        entry.size_bytes = size;
        if (rocm_profile != null and known != null)
            entry.rocm_16gb_context_length = catalog.rocm16GbContext(known.?.*, rocm_profile.?);
        entry.required_vram_bytes = if (entry.rocm_16gb_context_length != null)
            catalog.rocm_16gb_min_vram_bytes
        else if (known) |model|
            model.required_vram_bytes
        else
            size;
        entry.fits_current_gpu = budget_bytes == 0 or entry.required_vram_bytes <= budget_bytes;
        entry.fit_source = if (known != null) "catalog" else "estimate";
        // An unrecognized sibling is not implicitly validated merely because
        // its extension is GGUF. The startup file already loaded successfully.
        entry.supported_on_current_gpu = known != null or is_initial;
        entry.can_activate = entry.supported_on_current_gpu;
        entry.managed = is_managed;
        try self.entries.append(self.allocator, entry);
    }
};

fn isGguf(name: []const u8) bool {
    return name.len > 5 and std.ascii.eqlIgnoreCase(name[name.len - 5 ..], ".gguf");
}

/// Iteration checks the directory entry itself, so a symlink named `*.gguf`
/// cannot add a model from outside the authorized directory/cache locations.
fn regularFileSize(path: []const u8) !u64 {
    const parent = std.fs.path.dirname(path) orelse return error.InvalidPath;
    var dir = try std.fs.cwd().openDir(parent, .{ .iterate = true });
    defer dir.close();
    const name = std.fs.path.basename(path);
    var iterator = dir.iterate();
    while (try iterator.next()) |item| {
        if (!std.mem.eql(u8, item.name, name)) continue;
        if (item.kind != .file) return error.NotRegularFile;
        const stat = try dir.statFile(name);
        if (stat.kind != .file) return error.NotRegularFile;
        return stat.size;
    }
    return error.FileNotFound;
}

test "accelerator inventory discovers installed siblings without following symlinks or directories" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{ .sub_path = "start.gguf", .data = "weights" });
    try tmp.dir.writeFile(.{ .sub_path = "Other.GGUF", .data = "other" });
    try tmp.dir.writeFile(.{ .sub_path = "notes.txt", .data = "ignore" });
    try tmp.dir.makeDir("nested.gguf");
    try tmp.dir.symLink("start.gguf", "alias.gguf", .{});
    const path = try tmp.dir.realpathAlloc(allocator, "start.gguf");
    defer allocator.free(path);
    var inventory = try Inventory.collect(allocator, path, 64, false, null);
    defer inventory.deinit();
    try std.testing.expectEqual(@as(usize, 2), inventory.entries.items.len);
    const initial = inventory.find(inventory.initial_model_id).?;
    try std.testing.expectEqualStrings("start.gguf", initial.display_name);
    try std.testing.expectEqualStrings(path, initial.path);
    try std.testing.expect(initial.fits_current_gpu);
    try std.testing.expect(inventory.find("../start.gguf") == null);
    try std.testing.expect(inventory.find(path) == null);
    try std.testing.expect(inventory.find("start.gguf") == null);
    for (inventory.entries.items) |entry| {
        if (std.mem.eql(u8, entry.display_name, "Other.GGUF")) {
            try std.testing.expect(!entry.can_activate);
            try std.testing.expect(!entry.supported_on_current_gpu);
            try std.testing.expectEqualStrings("estimate", entry.fit_source);
        }
    }
}

test "accelerator inventory matches curated names and preserves fit metadata" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const known = catalog.find("qwen38-27b-q4k-m").?;
    try tmp.dir.writeFile(.{ .sub_path = known.file_name, .data = "fixture" });
    const path = try tmp.dir.realpathAlloc(allocator, known.file_name);
    defer allocator.free(path);
    var inventory = try Inventory.collect(allocator, path, 1024, false, null);
    defer inventory.deinit();
    try std.testing.expectEqual(@as(usize, 1), inventory.entries.items.len);
    const entry = inventory.find(known.id).?;
    try std.testing.expectEqualStrings(known.display_name, entry.display_name);
    try std.testing.expectEqualStrings(known.homepage_url, entry.homepage_url);
    try std.testing.expectEqual(known.required_vram_bytes, entry.required_vram_bytes);
    try std.testing.expect(!entry.fits_current_gpu);
    try std.testing.expectEqualStrings(known.id, inventory.initial_model_id);
    try std.testing.expect(entry.can_activate);
    try std.testing.expectEqualStrings("catalog", entry.fit_source);

    var rocm = try Inventory.collect(allocator, path, catalog.rocm_16gb_min_vram_bytes, false, "amd-rdna4-16gb");
    defer rocm.deinit();
    const compact = rocm.find(known.id).?;
    try std.testing.expect(compact.fits_current_gpu);
    try std.testing.expectEqual(catalog.rocm_16gb_min_vram_bytes, compact.required_vram_bytes);
    try std.testing.expectEqual(@as(?u32, 2048), compact.rocm_16gb_context_length);
}

test "accelerator inventory does not discover models recursively" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(.{ .sub_path = "start.gguf", .data = "fixture" });
    try tmp.dir.makeDir("subdir");
    try tmp.dir.writeFile(.{ .sub_path = "subdir/hidden.gguf", .data = "hidden" });
    const path = try tmp.dir.realpathAlloc(allocator, "start.gguf");
    defer allocator.free(path);
    var inventory = try Inventory.collect(allocator, path, 0, false, null);
    defer inventory.deinit();
    try std.testing.expectEqual(@as(usize, 1), inventory.entries.items.len);
}

fn allocationFixture(allocator: std.mem.Allocator, path: []const u8) !void {
    var inventory = try Inventory.collect(allocator, path, 0, false, null);
    defer inventory.deinit();
    try std.testing.expectEqual(@as(usize, 1), inventory.entries.items.len);
}

test "accelerator inventory cleans partial allocations and duplicate catalog ids" {
    const allocator = std.testing.allocator;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const known = catalog.find("qwen38-27b-q4k-m").?;
    try tmp.dir.writeFile(.{ .sub_path = known.file_name, .data = "fixture" });
    // Compact punctuation matching maps both files to the same advertised ID.
    try tmp.dir.writeFile(.{ .sub_path = "Qwen3_8-27B-Q4_K_M.gguf", .data = "duplicate" });
    const path = try tmp.dir.realpathAlloc(allocator, known.file_name);
    defer allocator.free(path);
    try std.testing.checkAllAllocationFailures(allocator, allocationFixture, .{path});
}
