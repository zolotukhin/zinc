//! Curated catalog of ZINC-supported managed GGUF models.
//! @section Managed Models
//! The catalog is intentionally small and only includes models that ZINC has
//! explicitly validated for the listed GPU profiles.
const std = @import("std");
const gpu_detect = @import("../vulkan/gpu_detect.zig");

/// Lifecycle status of a catalog entry, controlling visibility and UI treatment.
pub const CatalogStatus = enum {
    supported,
    experimental,
    hidden,
    deprecated,
};

/// A single managed-model entry describing its identity, download location,
/// hardware requirements, and tested GPU profiles.
pub const CatalogEntry = struct {
    id: []const u8,
    display_name: []const u8,
    /// Upstream model-family release date in YYYY-MM-DD format.
    release_date: []const u8,
    family: []const u8,
    format: []const u8,
    quantization: []const u8,
    file_name: []const u8,
    homepage_url: []const u8,
    download_url: []const u8,
    sha256: []const u8,
    size_bytes: u64,
    required_vram_bytes: u64,
    default_context_length: u32,
    recommended_for_chat: bool,
    /// Whether the model produces stable, useful output when thinking is enabled.
    /// Small models may loop or fail to separate reasoning from answers.
    thinking_stable: bool,
    status: CatalogStatus,
    tested_profiles: []const []const u8,
};

/// Shared GPU profile string used for all Apple Silicon (Metal) devices.
pub const apple_silicon_profile = "apple-silicon";

/// The complete list of ZINC-validated managed models available for download.
pub const entries = [_]CatalogEntry{
    .{
        .id = "qwen35-35b-a3b-q4k-xl",
        .display_name = "Qwen3.5 35B-A3B UD Q4_K_XL",
        .release_date = "2026-02-16",
        .family = "qwen3.5",
        .format = "gguf",
        .quantization = "UD-Q4_K_XL",
        .file_name = "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf",
        .homepage_url = "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF",
        .download_url = "https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/resolve/main/Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf?download=true",
        .sha256 = "1b0ac637dfa092bbba2793977db9485a40c4f8b42df5fe342f0076d61b66ae83",
        .size_bytes = 22_241_950_336,
        .required_vram_bytes = 22_987_514_102,
        .default_context_length = 4096,
        .recommended_for_chat = true,
        .thinking_stable = true,
        .status = .supported,
        .tested_profiles = &.{
            "amd-rdna4-32gb",
            apple_silicon_profile,
            "intel-arc",
        },
    },
    .{
        .id = "qwen36-35b-a3b-q4k-xl",
        .display_name = "Qwen3.6 35B-A3B UD Q4_K_XL",
        .release_date = "2026-04-15",
        .family = "qwen3.6",
        .format = "gguf",
        .quantization = "UD-Q4_K_XL",
        .file_name = "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        .homepage_url = "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF",
        .download_url = "https://huggingface.co/unsloth/Qwen3.6-35B-A3B-GGUF/resolve/main/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf?download=true",
        .sha256 = "",
        .size_bytes = 22_360_456_160,
        .required_vram_bytes = 23_106_019_926,
        .default_context_length = 4096,
        .recommended_for_chat = true,
        .thinking_stable = true,
        .status = .supported,
        .tested_profiles = &.{
            "amd-rdna4-32gb",
            apple_silicon_profile,
            "intel-arc",
        },
    },
    .{
        .id = "gpt-oss-20b-q4k-m",
        .display_name = "OpenAI GPT-OSS 20B Q4_K_M",
        .release_date = "2025-06-25",
        .family = "gpt-oss",
        .format = "gguf",
        .quantization = "Q4_K_M",
        .file_name = "openai_gpt-oss-20b-Q4_K_M.gguf",
        .homepage_url = "https://huggingface.co/bartowski/openai_gpt-oss-20b-GGUF",
        .download_url = "https://huggingface.co/bartowski/openai_gpt-oss-20b-GGUF/resolve/main/openai_gpt-oss-20b-Q4_K_M.gguf?download=true",
        .sha256 = "86a21df11afa5a40031ec1974e368ae0ab561ee3995f4d08ff432e8b2b7af9fc",
        .size_bytes = 11_673_418_816,
        .required_vram_bytes = 14 * 1024 * 1024 * 1024,
        .default_context_length = 4096,
        .recommended_for_chat = true,
        .thinking_stable = true,
        .status = .supported,
        .tested_profiles = &.{
            apple_silicon_profile,
            "intel-arc",
        },
    },
    .{
        .id = "qwen3-8b-q4k-m",
        .display_name = "Qwen3 8B Q4_K_M",
        .release_date = "2025-04-29",
        .family = "qwen3",
        .format = "gguf",
        .quantization = "Q4_K_M",
        .file_name = "Qwen3-8B-Q4_K_M.gguf",
        .homepage_url = "https://huggingface.co/unsloth/Qwen3-8B-GGUF",
        .download_url = "https://huggingface.co/unsloth/Qwen3-8B-GGUF/resolve/main/Qwen3-8B-Q4_K_M.gguf?download=true",
        .sha256 = "120307ba529eb2439d6c430d94104dabd578497bc7bfe7e322b5d9933b449bd4",
        .size_bytes = 5_027_784_512,
        .required_vram_bytes = 6 * 1024 * 1024 * 1024,
        .default_context_length = 4096,
        .recommended_for_chat = true,
        .thinking_stable = true,
        .status = .supported,
        .tested_profiles = &.{
            "amd-rdna4-32gb",
            apple_silicon_profile,
            "intel-arc",
        },
    },
    .{
        .id = "gemma4-31b-q4k-m",
        .display_name = "Gemma 4 31B Q4_K_M",
        .release_date = "2026-04-02",
        .family = "gemma4",
        .format = "gguf",
        .quantization = "Q4_K_M",
        .file_name = "gemma-4-31B-it-Q4_K_M.gguf",
        .homepage_url = "https://huggingface.co/unsloth/gemma-4-31B-it-GGUF",
        .download_url = "https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/gemma-4-31B-it-Q4_K_M.gguf?download=true",
        .sha256 = "",
        .size_bytes = 19_650_000_000,
        .required_vram_bytes = 21 * 1024 * 1024 * 1024,
        .default_context_length = 4096,
        .recommended_for_chat = true,
        .thinking_stable = true,
        .status = .supported,
        .tested_profiles = &.{
            "amd-rdna4-32gb",
            apple_silicon_profile,
            "intel-arc",
        },
    },
    .{
        .id = "gemma4-12b-q4k-m",
        .display_name = "Gemma 4 12B (26B-A4B MoE) Q4_K_M",
        .release_date = "2026-04-02",
        .family = "gemma4",
        .format = "gguf",
        .quantization = "Q4_K_M",
        .file_name = "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
        .homepage_url = "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF",
        .download_url = "https://huggingface.co/unsloth/gemma-4-26B-A4B-it-GGUF/resolve/main/gemma-4-26B-A4B-it-UD-Q4_K_M.gguf?download=true",
        .sha256 = "",
        .size_bytes = 16_868_236_288,
        .required_vram_bytes = 16 * 1024 * 1024 * 1024,
        .default_context_length = 4096,
        .recommended_for_chat = true,
        .thinking_stable = true,
        .status = .experimental,
        .tested_profiles = &.{
            "amd-rdna4-32gb",
            apple_silicon_profile,
            "intel-arc",
        },
    },
};

/// Look up a catalog entry by its short identifier, returning null if not found.
pub fn find(id: []const u8) ?*const CatalogEntry {
    for (&entries) |*entry| {
        if (std.mem.eql(u8, entry.id, id)) return entry;
    }
    return null;
}

fn eqlCompactAsciiCaseInsensitive(a: []const u8, b: []const u8) bool {
    var ia: usize = 0;
    var ib: usize = 0;

    while (true) {
        while (ia < a.len and !std.ascii.isAlphanumeric(a[ia])) ia += 1;
        while (ib < b.len and !std.ascii.isAlphanumeric(b[ib])) ib += 1;

        const done_a = ia >= a.len;
        const done_b = ib >= b.len;
        if (done_a or done_b) return done_a and done_b;

        if (std.ascii.toLower(a[ia]) != std.ascii.toLower(b[ib])) return false;
        ia += 1;
        ib += 1;
    }
}

/// Match a loaded model back to a catalog entry, even when it was opened from a
/// raw path instead of a managed model id.
pub fn findForLoadedModel(managed_id: ?[]const u8, model_path: []const u8, display_name: []const u8) ?*const CatalogEntry {
    if (managed_id) |id| {
        if (find(id)) |entry| return entry;
    }

    const base_name = std.fs.path.basename(model_path);
    if (std.mem.eql(u8, base_name, "model.gguf")) {
        if (std.fs.path.dirname(model_path)) |parent_path| {
            const parent_name = std.fs.path.basename(parent_path);
            if (find(parent_name)) |entry| return entry;
        }
    }

    for (&entries) |*entry| {
        if (std.mem.eql(u8, base_name, entry.file_name) or eqlCompactAsciiCaseInsensitive(base_name, entry.file_name)) {
            return entry;
        }
        if (eqlCompactAsciiCaseInsensitive(display_name, entry.display_name) or
            eqlCompactAsciiCaseInsensitive(display_name, entry.file_name))
        {
            return entry;
        }
    }

    return null;
}

/// Map a detected Vulkan GPU configuration to its catalog profile string.
pub fn profileForGpu(config: gpu_detect.GpuConfig) []const u8 {
    return switch (config.vendor) {
        .amd_rdna4 => if (config.vram_mb >= 28 * 1024) "amd-rdna4-32gb" else if (config.vram_mb >= 14 * 1024) "amd-rdna4-16gb" else "amd-rdna4-small",
        .amd_rdna4_apu => "amd-rdna4-apu",
        .amd_rdna3 => if (config.vram_mb >= 14 * 1024) "amd-rdna3-16gb" else "amd-rdna3-small",
        .amd_other => "amd-other",
        .nvidia => "nvidia",
        .intel_arc_xe2 => "intel-arc",
        .intel_arc => "intel-arc",
        .unknown => "unknown",
    };
}

/// Return the catalog profile string for Apple Silicon Metal devices.
pub fn profileForMetal() []const u8 {
    return apple_silicon_profile;
}

/// Return whether the entry has been tested on the given GPU profile.
pub fn supportsProfile(entry: CatalogEntry, profile: []const u8) bool {
    for (entry.tested_profiles) |tested| {
        if (std.mem.eql(u8, tested, profile)) return true;
    }
    return false;
}

/// Return whether the model's VRAM requirement fits within the given budget.
pub fn fitsGpu(entry: CatalogEntry, vram_budget_bytes: u64) bool {
    return entry.required_vram_bytes <= vram_budget_bytes;
}

/// Return whether the model is both tested on the given profile and fits in VRAM.
pub fn supportedOnCurrentGpu(entry: CatalogEntry, profile: []const u8, vram_budget_bytes: u64) bool {
    return supportsProfile(entry, profile) and fitsGpu(entry, vram_budget_bytes);
}

/// Map a catalog family string to the GGUF architecture string that models in
/// that family use. Returns null for unrecognized families — the caller should
/// treat that as an error (a catalog entry with no known architecture mapping).
pub fn ggufArchForFamily(family: []const u8) ?[]const u8 {
    const families = .{
        .{ "qwen3.6", "qwen35" },
        .{ "qwen3.5", "qwen35" },
        .{ "qwen3", "qwen3" },
        .{ "qwen2.5", "qwen2" },
        .{ "qwen2", "qwen2" },
        .{ "mistral", "mistral" },
        .{ "gemma4", "gemma4" },
        .{ "gemma2", "gemma2" },
        .{ "gemma", "gemma" },
        .{ "mamba", "mamba" },
        .{ "jamba", "jamba" },
        .{ "gpt-oss", "gpt-oss" },
    };
    inline for (families) |pair| {
        if (std.mem.eql(u8, family, pair[0])) return pair[1];
    }
    return null;
}

test "every catalog entry maps to a supported architecture" {
    const config_mod = @import("config.zig");
    for (&entries) |entry| {
        const gguf_arch = ggufArchForFamily(entry.family) orelse {
            std.debug.print("FAIL: catalog entry '{s}' has family '{s}' with no known GGUF architecture mapping\n", .{ entry.id, entry.family });
            return error.TestExpectedEqual;
        };
        const arch = config_mod.parseArchitecture(gguf_arch);
        if (arch == .unknown) {
            std.debug.print("FAIL: catalog entry '{s}' (family '{s}') maps to GGUF arch '{s}' which parseArchitecture returns .unknown\n", .{ entry.id, entry.family, gguf_arch });
            return error.TestExpectedEqual;
        }
    }
}

test "catalog IDs are unique" {
    for (&entries, 0..) |a, i| {
        for (entries[i + 1 ..]) |b| {
            if (std.mem.eql(u8, a.id, b.id)) {
                std.debug.print("FAIL: duplicate catalog ID '{s}'\n", .{a.id});
                return error.TestExpectedEqual;
            }
        }
    }
}

test "find returns null for unknown model" {
    try std.testing.expect(find("nonexistent-model-id") == null);
}

test "find returns known entry" {
    const entry = find("qwen3-8b-q4k-m") orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("Qwen3 8B Q4_K_M", entry.display_name);
    try std.testing.expectEqualStrings("2025-04-29", entry.release_date);
}

test "find returns known qwen3.6 entry" {
    const entry = find("qwen36-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("Qwen3.6 35B-A3B UD Q4_K_XL", entry.display_name);
    try std.testing.expectEqualStrings("2026-04-15", entry.release_date);
    try std.testing.expectEqualStrings("qwen3.6", entry.family);
    try std.testing.expect(entry.recommended_for_chat);
    try std.testing.expect(entry.thinking_stable);
    try std.testing.expect(entry.status == .supported);
}

test "qwen3.6 family reuses qwen35 gguf architecture mapping" {
    try std.testing.expectEqualStrings("qwen35", ggufArchForFamily("qwen3.6") orelse return error.TestExpectedEqual);
}

test "findForLoadedModel matches managed-cache qwen36 path" {
    const entry = findForLoadedModel(
        null,
        "/Users/test/Library/Caches/zinc/models/models/qwen36-35b-a3b-q4k-xl/model.gguf",
        "Qwen3.6-35B-A3B-UD-Q4_K_XL",
    ) orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("qwen36-35b-a3b-q4k-xl", entry.id);
}

test "findForLoadedModel matches raw filename and loose display name" {
    const entry = findForLoadedModel(
        null,
        "/tmp/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        "Qwen3.6 35B A3B UD Q4 K XL",
    ) orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("qwen36-35b-a3b-q4k-xl", entry.id);
}

test "profileForGpu maps RDNA4 32 GB boards" {
    const config = gpu_detect.GpuConfig{
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
    };
    try std.testing.expectEqualStrings("amd-rdna4-32gb", profileForGpu(config));
}

test "profileForMetal returns apple silicon profile" {
    try std.testing.expectEqualStrings(apple_silicon_profile, profileForMetal());
}

test "fitsGpu compares against required vram" {
    const entry = find("qwen35-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    try std.testing.expect(fitsGpu(entry.*, 24 * 1024 * 1024 * 1024));
    try std.testing.expect(!fitsGpu(entry.*, 20 * 1024 * 1024 * 1024));
}

test "supportedOnCurrentGpu requires both tested profile and fit" {
    const entry = find("qwen35-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    try std.testing.expect(supportedOnCurrentGpu(entry.*, "amd-rdna4-32gb", 24 * 1024 * 1024 * 1024));
    try std.testing.expect(!supportedOnCurrentGpu(entry.*, "amd-rdna4-16gb", 24 * 1024 * 1024 * 1024));
    try std.testing.expect(!supportedOnCurrentGpu(entry.*, "amd-rdna4-32gb", 20 * 1024 * 1024 * 1024));
}

test "qwen thinking stability flags track validated chat behavior" {
    const qwen3 = find("qwen3-8b-q4k-m") orelse return error.TestExpectedEqual;
    try std.testing.expect(qwen3.recommended_for_chat);
    try std.testing.expect(qwen3.thinking_stable);

    const qwen35 = find("qwen35-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    try std.testing.expect(qwen35.thinking_stable);

    const qwen36 = find("qwen36-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    try std.testing.expect(qwen36.thinking_stable);
}
