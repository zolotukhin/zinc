//! Curated catalog of ZINC-supported managed GGUF models.
//! @section Managed Models
//! The catalog is intentionally small and only includes models that ZINC has
//! explicitly validated for the listed GPU profiles.
const std = @import("std");
const gpu_detect = @import("../vulkan/gpu_detect.zig");

pub const CatalogStatus = enum {
    supported,
    experimental,
    hidden,
    deprecated,
};

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

pub const entries = [_]CatalogEntry{
    .{
        .id = "qwen35-2b-q4k-m",
        .display_name = "Qwen3.5 2B Q4_K_M",
        .release_date = "2026-02-16",
        .family = "qwen3.5",
        .format = "gguf",
        .quantization = "Q4_K_M",
        .file_name = "Qwen3.5-2B-Q4_K_M.gguf",
        .homepage_url = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF",
        .download_url = "https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf?download=true",
        .sha256 = "aaf42c8b7c3cab2bf3d69c355048d4a0ee9973d48f16c731c0520ee914699223",
        .size_bytes = 1_374_389_535,
        .required_vram_bytes = 3 * 1024 * 1024 * 1024,
        .default_context_length = 4096,
        .recommended_for_chat = true,
        .thinking_stable = false,
        .status = .supported,
        .tested_profiles = &.{
            "amd-rdna4-32gb",
        },
    },
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
        .sha256 = "70f7b15b38aa5d778e3ecd7158fc35ecc7b9c36d11696d4ec59a116b75a78158",
        .size_bytes = 21_152_743_629,
        .required_vram_bytes = 22_987_514_102,
        .default_context_length = 4096,
        .recommended_for_chat = true,
        .thinking_stable = true,
        .status = .supported,
        .tested_profiles = &.{
            "amd-rdna4-32gb",
        },
    },
};

pub fn find(id: []const u8) ?*const CatalogEntry {
    for (&entries) |*entry| {
        if (std.mem.eql(u8, entry.id, id)) return entry;
    }
    return null;
}

pub fn profileForGpu(config: gpu_detect.GpuConfig) []const u8 {
    return switch (config.vendor) {
        .amd_rdna4 => if (config.vram_mb >= 28 * 1024) "amd-rdna4-32gb" else if (config.vram_mb >= 14 * 1024) "amd-rdna4-16gb" else "amd-rdna4-small",
        .amd_rdna4_apu => "amd-rdna4-apu",
        .amd_rdna3 => if (config.vram_mb >= 14 * 1024) "amd-rdna3-16gb" else "amd-rdna3-small",
        .amd_other => "amd-other",
        .nvidia => "nvidia",
        .intel_arc => "intel-arc",
        .unknown => "unknown",
    };
}

pub fn supportsProfile(entry: CatalogEntry, profile: []const u8) bool {
    for (entry.tested_profiles) |tested| {
        if (std.mem.eql(u8, tested, profile)) return true;
    }
    return false;
}

pub fn fitsGpu(entry: CatalogEntry, vram_budget_bytes: u64) bool {
    return entry.required_vram_bytes <= vram_budget_bytes;
}

pub fn supportedOnCurrentGpu(entry: CatalogEntry, profile: []const u8, vram_budget_bytes: u64) bool {
    return supportsProfile(entry, profile) and fitsGpu(entry, vram_budget_bytes);
}

test "find returns known entry" {
    const entry = find("qwen35-2b-q4k-m") orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("Qwen3.5 2B Q4_K_M", entry.display_name);
    try std.testing.expectEqualStrings("2026-02-16", entry.release_date);
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

test "thinking_stable is false for 2B and true for 35B" {
    const small = find("qwen35-2b-q4k-m") orelse return error.TestExpectedEqual;
    try std.testing.expect(!small.thinking_stable);
    const large = find("qwen35-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    try std.testing.expect(large.thinking_stable);
}
