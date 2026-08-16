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
    /// Bytes of MoE expert tensors that move to host RAM when
    /// `ZINC_OFFLOAD_MOE_EXPERTS=1`. Zero for dense models (no benefit
    /// from offload). For MoE entries this is a catalog estimate; the
    /// loader/inspector will overwrite with the exact tensor sum once
    /// the model is installed.
    offloadable_vram_bytes: u64 = 0,
    default_context_length: u32,
    recommended_for_chat: bool,
    /// Whether the model produces stable, useful output when thinking is enabled.
    /// Small models may loop or fail to separate reasoning from answers.
    thinking_stable: bool,
    status: CatalogStatus,
    tested_profiles: []const []const u8,
};

/// VRAM-fit assessment for a catalog entry against a specific GPU budget.
pub const FitState = enum {
    /// Model fits in VRAM with no special configuration.
    fits,
    /// Model fits only when `ZINC_OFFLOAD_MOE_EXPERTS=1` moves MoE expert
    /// tensors to host RAM. Requires a recent driver and willingness to
    /// pay PCIe latency on expert reads.
    fits_with_offload,
    /// Model is too large for this GPU even with offload.
    does_not_fit,
};

/// Shared GPU profile string used for all Apple Silicon (Metal) devices.
pub const apple_silicon_profile = "apple-silicon";

/// The complete list of ZINC-validated managed models available for download.
pub const entries = [_]CatalogEntry{
    .{
        .id = "qwen38-27b-q4k-m",
        .display_name = "Qwen3.8 27B Dense Q4_K_M",
        .release_date = "2026-08-14",
        .family = "qwen3.8",
        .format = "gguf",
        .quantization = "Q4_K_M",
        .file_name = "Qwen3.8-27B-Q4_K_M.gguf",
        .homepage_url = "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF",
        .download_url = "https://huggingface.co/unsloth/Qwen3.8-27B-GGUF/resolve/main/Qwen3.8-27B-Q4_K_M.gguf?download=true",
        .sha256 = "7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169",
        .size_bytes = 17_106_775_008,
        // Qwen3.8-27B retains Qwen3.6-27B's qwen35 text architecture and
        // dimensions. Validated 32 GB-class RDNA4 and Apple Silicon systems
        // have enough headroom for weights, the capped context, and scratch.
        .required_vram_bytes = 20 * 1024 * 1024 * 1024,
        .default_context_length = 4096,
        .recommended_for_chat = true,
        .thinking_stable = true,
        .status = .supported,
        .tested_profiles = &.{
            "amd-rdna4-32gb",
            apple_silicon_profile,
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
        // 35B-A3B: same architecture as Qwen3.5; ≈ 18 GiB of offloadable experts.
        .offloadable_vram_bytes = 18 * 1024 * 1024 * 1024,
        .default_context_length = 4096,
        // Raw completion is validated on Intel/RDNA/Metal. The model exposes
        // the Qwen thinking scaffold; chat is still not the recommended default
        // workflow for this MoE pack because short direct prompts can be brittle.
        .recommended_for_chat = false,
        .thinking_stable = true,
        .status = .supported,
        .tested_profiles = &.{
            "amd-rdna4-32gb",
            apple_silicon_profile,
            "intel-arc",
            nvidia_cuda_profile,
        },
    },
    .{
        .id = "qwen36-27b-q4k-m",
        .display_name = "Qwen3.6 27B Dense Q4_K_M",
        .release_date = "2026-04-22",
        .family = "qwen3.6",
        .format = "gguf",
        .quantization = "Q4_K_M",
        .file_name = "Qwen3.6-27B-Q4_K_M.gguf",
        .homepage_url = "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF",
        .download_url = "https://huggingface.co/unsloth/Qwen3.6-27B-GGUF/resolve/main/Qwen3.6-27B-Q4_K_M.gguf?download=true",
        .sha256 = "",
        .size_bytes = 16_817_244_384,
        .required_vram_bytes = 20 * 1024 * 1024 * 1024,
        .default_context_length = 4096,
        .recommended_for_chat = true,
        .thinking_stable = true,
        .status = .experimental,
        .tested_profiles = &.{
            "amd-rdna4-32gb",
            apple_silicon_profile,
            "intel-arc",
            nvidia_cuda_profile,
        },
    },
    .{
        .id = "qwen35-9b-q4k-m",
        .display_name = "Qwen 3.5 9B Q4_K_M",
        .release_date = "2026-02-28",
        .family = "qwen3.5",
        .format = "gguf",
        .quantization = "Q4_K_M",
        .file_name = "Qwen3.5-9B-Q4_K_M.gguf",
        .homepage_url = "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF",
        .download_url = "https://huggingface.co/unsloth/Qwen3.5-9B-GGUF/resolve/main/Qwen3.5-9B-Q4_K_M.gguf?download=true",
        .sha256 = "03b74727a860a56338e042c4420bb3f04b2fec5734175f4cb9fa853daf52b7e8",
        .size_bytes = 5_680_522_464,
        .required_vram_bytes = 7 * 1024 * 1024 * 1024,
        .default_context_length = 4096,
        .recommended_for_chat = true,
        .thinking_stable = true,
        .status = .supported,
        .tested_profiles = &.{
            "amd-rdna4-32gb",
            "amd-rdna4-16gb",
            apple_silicon_profile,
            "intel-arc",
            nvidia_cuda_profile,
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
            nvidia_cuda_profile,
        },
    },
    .{
        .id = "gemma4-26b-a4b-q4k-m",
        .display_name = "Gemma 4 26B-A4B MoE Q4_K_M",
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
        // Gemma4 26B-A4B MoE (~4B active, ~22B inactive experts) at Q4_K_M ≈ 11 GiB.
        .offloadable_vram_bytes = 11 * 1024 * 1024 * 1024,
        .default_context_length = 4096,
        .recommended_for_chat = true,
        .thinking_stable = true,
        .status = .supported,
        .tested_profiles = &.{
            "amd-rdna4-32gb",
            apple_silicon_profile,
            "intel-arc",
            nvidia_cuda_profile,
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
/// raw path instead of a managed-model id. Resolution order: managed id → parent
/// directory name (for managed-cache `model.gguf` paths) → file-name exact match
/// → file-name compact-case-insensitive match → display-name fuzzy match.
/// @param managed_id Optional catalog id previously recorded for this model; tried first.
/// @param model_path Absolute filesystem path to the GGUF file being loaded.
/// @param display_name Human-readable name used as a last-resort fuzzy key.
/// @returns Pointer into `entries`, or null when no catalog entry matches.
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
/// RDNA4 is split by VRAM tier (≥28 GiB → "amd-rdna4-32gb", ≥14 GiB →
/// "amd-rdna4-16gb", otherwise "amd-rdna4-small"); RDNA3 is split by VRAM
/// tier (≥14 GiB → "amd-rdna3-16gb", otherwise "amd-rdna3-small"); all
/// other vendors map to a single string each.
/// @param config GPU vendor/VRAM description produced by `gpu_detect`.
/// @returns Catalog profile key that can be matched against `CatalogEntry.tested_profiles`.
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

/// Shared GPU profile string used for all NVIDIA (CUDA) devices.
pub const nvidia_cuda_profile = "nvidia-cuda";

/// Return the catalog profile string for NVIDIA CUDA devices. The CUDA backend
/// does not split by VRAM tier the way Vulkan/RDNA4 does — fit is decided by the
/// live `freeMemory()` budget — so a single profile string is sufficient.
pub fn profileForCuda() []const u8 {
    return nvidia_cuda_profile;
}

/// Return whether the entry has been tested on the given GPU profile.
/// @param entry Catalog entry to check.
/// @param profile Profile string such as `"amd-rdna4-32gb"` or `apple_silicon_profile`.
/// @returns true if `profile` appears in `entry.tested_profiles`.
pub fn supportsProfile(entry: CatalogEntry, profile: []const u8) bool {
    for (entry.tested_profiles) |tested| {
        if (std.mem.eql(u8, tested, profile)) return true;
    }
    return false;
}

/// Return whether the model's VRAM requirement fits within the given budget
/// without enabling MoE offload. Strict — does not consider the offload escape
/// hatch. Use `fitState` for the offload-aware tri-state assessment.
pub fn fitsGpu(entry: CatalogEntry, vram_budget_bytes: u64) bool {
    return entry.required_vram_bytes <= vram_budget_bytes;
}

/// VRAM required when MoE expert tensors are offloaded to host RAM. Equal to
/// `required_vram_bytes` for dense models (no offloadable tensors).
pub fn requiredVramWithOffload(entry: CatalogEntry) u64 {
    if (entry.offloadable_vram_bytes >= entry.required_vram_bytes) return 0;
    return entry.required_vram_bytes - entry.offloadable_vram_bytes;
}

/// Tri-state fit assessment that distinguishes "fits as-is" from "fits only
/// with `ZINC_OFFLOAD_MOE_EXPERTS=1`". Use this to surface the offload escape
/// hatch to users when a model would otherwise look unsupported.
pub fn fitState(entry: CatalogEntry, vram_budget_bytes: u64) FitState {
    if (entry.required_vram_bytes <= vram_budget_bytes) return .fits;
    if (entry.offloadable_vram_bytes > 0 and requiredVramWithOffload(entry) <= vram_budget_bytes) return .fits_with_offload;
    return .does_not_fit;
}

/// Return true when the model needs `ZINC_OFFLOAD_MOE_EXPERTS=1` to fit
/// (does not fit by itself but does fit with offload enabled).
pub fn requiresOffloadToFit(entry: CatalogEntry, vram_budget_bytes: u64) bool {
    return fitState(entry, vram_budget_bytes) == .fits_with_offload;
}

/// Return whether the model is both tested on the given profile and fits in VRAM
/// without MoE offload. Equivalent to `supportsProfile and fitsGpu`.
/// @param entry Catalog entry to evaluate.
/// @param profile Detected GPU profile string (e.g. from `profileForGpu`).
/// @param vram_budget_bytes Available VRAM budget in bytes.
/// @returns true only when both conditions hold; does not consider offload.
pub fn supportedOnCurrentGpu(entry: CatalogEntry, profile: []const u8, vram_budget_bytes: u64) bool {
    return supportsProfile(entry, profile) and fitsGpu(entry, vram_budget_bytes);
}

/// Map a catalog family string to the GGUF architecture string that models in
/// that family use. Returns null for unrecognized families — the caller should
/// treat that as an error (a catalog entry with no known architecture mapping).
/// Note that Qwen 3.5, 3.6, and 3.8 all map to `"qwen35"` because their GGUFs
/// declare the SSM+attention hybrid text architecture under that name.
/// @param family Value of `CatalogEntry.family` (e.g. `"gemma4"`, `"qwen3.6"`).
/// @returns GGUF architecture identifier, or null if the family is not recognized.
pub fn ggufArchForFamily(family: []const u8) ?[]const u8 {
    const families = .{
        .{ "qwen3.8", "qwen35" },
        .{ "qwen3.6", "qwen35" },
        // Qwen 3.5 is a dense SSM+attention hybrid (the GGUF declares the
        // "qwen35" architecture), the same family ZINC drives for Qwen 3.6 —
        // not a plain transformer.
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
    const entry = find("qwen35-9b-q4k-m") orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("Qwen 3.5 9B Q4_K_M", entry.display_name);
    try std.testing.expectEqualStrings("2026-02-28", entry.release_date);
}

test "find returns known qwen3.6 entry" {
    const entry = find("qwen36-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("Qwen3.6 35B-A3B UD Q4_K_XL", entry.display_name);
    try std.testing.expectEqualStrings("2026-04-15", entry.release_date);
    try std.testing.expectEqualStrings("qwen3.6", entry.family);
    try std.testing.expect(!entry.recommended_for_chat);
    try std.testing.expect(entry.thinking_stable);
    try std.testing.expect(entry.status == .supported);
}

test "find returns qwen3.6 27b dense entry" {
    const entry = find("qwen36-27b-q4k-m") orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("Qwen3.6 27B Dense Q4_K_M", entry.display_name);
    try std.testing.expectEqualStrings("2026-04-22", entry.release_date);
    try std.testing.expectEqualStrings("qwen3.6", entry.family);
    try std.testing.expectEqualStrings("Qwen3.6-27B-Q4_K_M.gguf", entry.file_name);
    try std.testing.expect(entry.recommended_for_chat);
    try std.testing.expect(entry.thinking_stable);
    try std.testing.expect(entry.status == .experimental);
}

test "find returns qwen3.8 27b dense entry" {
    const entry = find("qwen38-27b-q4k-m") orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("Qwen3.8 27B Dense Q4_K_M", entry.display_name);
    try std.testing.expectEqualStrings("2026-08-14", entry.release_date);
    try std.testing.expectEqualStrings("qwen3.8", entry.family);
    try std.testing.expectEqualStrings("Qwen3.8-27B-Q4_K_M.gguf", entry.file_name);
    try std.testing.expectEqualStrings("7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169", entry.sha256);
    try std.testing.expectEqual(@as(u64, 17_106_775_008), entry.size_bytes);
    try std.testing.expect(entry.recommended_for_chat);
    try std.testing.expect(entry.thinking_stable);
    try std.testing.expect(entry.status == .supported);
    try std.testing.expect(supportsProfile(entry.*, "amd-rdna4-32gb"));
    try std.testing.expect(supportsProfile(entry.*, apple_silicon_profile));
}

test "qwen3.6 family reuses qwen35 gguf architecture mapping" {
    try std.testing.expectEqualStrings("qwen35", ggufArchForFamily("qwen3.6") orelse return error.TestExpectedEqual);
}

test "qwen3.8 family reuses qwen35 gguf architecture mapping" {
    try std.testing.expectEqualStrings("qwen35", ggufArchForFamily("qwen3.8") orelse return error.TestExpectedEqual);
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

test "findForLoadedModel matches qwen36 27b dense filename" {
    const entry = findForLoadedModel(
        null,
        "/root/models/Qwen3.6-27B-Q4_K_M.gguf",
        "Qwen3.6 27B Q4 K M",
    ) orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("qwen36-27b-q4k-m", entry.id);
}

test "findForLoadedModel matches qwen38 27b dense filename" {
    const entry = findForLoadedModel(
        null,
        "/root/models/Qwen3.8-27B-Q4_K_M.gguf",
        "Qwen3.8 27B Q4 K M",
    ) orelse return error.TestExpectedEqual;
    try std.testing.expectEqualStrings("qwen38-27b-q4k-m", entry.id);
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
    const entry = find("qwen36-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    try std.testing.expect(fitsGpu(entry.*, 24 * 1024 * 1024 * 1024));
    try std.testing.expect(!fitsGpu(entry.*, 20 * 1024 * 1024 * 1024));
}

test "fitState distinguishes fits, fits_with_offload, does_not_fit" {
    const moe = find("qwen36-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    // 32 GiB: fits everything, no offload needed.
    try std.testing.expectEqual(FitState.fits, fitState(moe.*, 32 * 1024 * 1024 * 1024));
    // 16 GiB: doesn't fit straight (needs ~22 GiB) but fits with offload (~3.5 GiB).
    try std.testing.expectEqual(FitState.fits_with_offload, fitState(moe.*, 16 * 1024 * 1024 * 1024));
    // 2 GiB: too small even with offload.
    try std.testing.expectEqual(FitState.does_not_fit, fitState(moe.*, 2 * 1024 * 1024 * 1024));
}

test "fitState for dense model never returns fits_with_offload" {
    const dense = find("qwen35-9b-q4k-m") orelse return error.TestExpectedEqual;
    // Fits in 8 GiB.
    try std.testing.expectEqual(FitState.fits, fitState(dense.*, 8 * 1024 * 1024 * 1024));
    // Doesn't fit in 4 GiB and can't be helped by offload (no expert tensors).
    try std.testing.expectEqual(FitState.does_not_fit, fitState(dense.*, 4 * 1024 * 1024 * 1024));
}

test "requiresOffloadToFit only true when straight fit fails but offload fits" {
    const moe = find("qwen36-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    try std.testing.expect(!requiresOffloadToFit(moe.*, 32 * 1024 * 1024 * 1024)); // straight fit
    try std.testing.expect(requiresOffloadToFit(moe.*, 16 * 1024 * 1024 * 1024)); // offload-only
    try std.testing.expect(!requiresOffloadToFit(moe.*, 2 * 1024 * 1024 * 1024)); // too small either way
}

test "requiredVramWithOffload subtracts offloadable share" {
    const moe = find("qwen36-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    const without = moe.required_vram_bytes;
    const with = requiredVramWithOffload(moe.*);
    try std.testing.expect(with < without);
    try std.testing.expectEqual(without - moe.offloadable_vram_bytes, with);
}

test "requiredVramWithOffload returns required_vram_bytes for dense models" {
    const dense = find("qwen35-9b-q4k-m") orelse return error.TestExpectedEqual;
    try std.testing.expectEqual(dense.required_vram_bytes, requiredVramWithOffload(dense.*));
}

test "supportedOnCurrentGpu requires both tested profile and fit" {
    const entry = find("qwen36-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    try std.testing.expect(supportedOnCurrentGpu(entry.*, "amd-rdna4-32gb", 24 * 1024 * 1024 * 1024));
    try std.testing.expect(!supportedOnCurrentGpu(entry.*, "amd-rdna4-16gb", 24 * 1024 * 1024 * 1024));
    try std.testing.expect(!supportedOnCurrentGpu(entry.*, "amd-rdna4-32gb", 20 * 1024 * 1024 * 1024));
}

test "qwen3.8 27b is visible on validated RDNA and Apple Silicon profiles" {
    const entry = find("qwen38-27b-q4k-m") orelse return error.TestExpectedEqual;
    try std.testing.expect(entry.status == .supported);
    try std.testing.expect(supportedOnCurrentGpu(entry.*, "amd-rdna4-32gb", 32 * 1024 * 1024 * 1024));
    try std.testing.expect(!supportedOnCurrentGpu(entry.*, "amd-rdna4-16gb", 32 * 1024 * 1024 * 1024));
    try std.testing.expect(supportedOnCurrentGpu(entry.*, apple_silicon_profile, 32 * 1024 * 1024 * 1024));
}

test "qwen thinking stability flags track validated chat behavior" {
    const qwen3 = find("qwen35-9b-q4k-m") orelse return error.TestExpectedEqual;
    try std.testing.expect(qwen3.recommended_for_chat);
    try std.testing.expect(qwen3.thinking_stable);

    const qwen36 = find("qwen36-35b-a3b-q4k-xl") orelse return error.TestExpectedEqual;
    try std.testing.expect(!qwen36.recommended_for_chat);
    try std.testing.expect(qwen36.thinking_stable);

    const qwen36_dense = find("qwen36-27b-q4k-m") orelse return error.TestExpectedEqual;
    try std.testing.expect(qwen36_dense.thinking_stable);
}

test "supported qwen catalog entries expose validated thinking toggles" {
    // The chat UI only shows thinking when the active catalog entry marks it as
    // stable. Keep every supported Qwen entry on the validated path.
    for (&entries) |entry| {
        if (entry.status == .supported and std.mem.startsWith(u8, entry.family, "qwen")) {
            try std.testing.expect(entry.thinking_stable);
        }
    }
}

test "supported catalog entries do not force-hide thinking metadata" {
    // `supported` models are the default user-facing choices. If a future entry
    // still needs the thinking toggle hidden, keep it experimental until chat
    // behavior is validated and this invariant can stay true.
    for (&entries) |entry| {
        if (entry.status == .supported) {
            try std.testing.expect(entry.thinking_stable);
        }
    }
}
