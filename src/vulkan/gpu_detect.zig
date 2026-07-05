//! Inspect the selected Vulkan device and derive architecture-specific tuning defaults.
//! @section Hardware Detection
//! The heuristics here convert raw Vulkan device properties into the settings
//! used by DMMV, matmul, and attention dispatch code.
const std = @import("std");
const vk = @import("vk.zig");
const Instance = @import("instance.zig").Instance;
const builtin = @import("builtin");

const log = std.log.scoped(.gpu_detect);

/// GPU vendor and architecture buckets used by ZINC's tuning heuristics.
pub const GpuVendor = enum {
    amd_rdna3,
    amd_rdna4,
    /// AMD Strix Halo APU iGPU (gfx1151). Architecturally RDNA 3.5, not RDNA4,
    /// but it exposes cooperative matrix like RDNA3/RDNA4, so ZINC buckets it
    /// with the RDNA4 tuning path (hence the name). Bandwidth is constrained by
    /// the shared LPDDR5X system-memory bus (~256 GB/s on 128 GB configs)
    /// rather than dedicated GDDR6, so tuning defaults differ from amd_rdna4.
    amd_rdna4_apu,
    amd_other,
    nvidia,
    /// Intel Arc Xe2 (Battlemage, B-series): min subgroup 16, default subgroup 32.
    intel_arc_xe2,
    /// Intel Arc Xe-HPG (Alchemist, A-series): SIMD32 subgroup.
    intel_arc,
    unknown,
};

/// Auto-detected GPU capabilities and derived tuning parameters.
pub const GpuConfig = struct {
    /// GPU vendor and architecture.
    vendor: GpuVendor,
    /// Device name bytes (null-padded).
    device_name: [256]u8,
    /// Valid bytes in device_name.
    device_name_len: usize,
    /// Total VRAM in MB.
    vram_mb: u32,
    /// Memory bandwidth in GB/s.
    bandwidth_gbps: u32,
    /// Number of compute units.
    compute_units: u32,
    /// SIMD wave width (from Vulkan subgroup size: e.g. 16, 32, or 64).
    wave_size: u32,
    /// Cooperative matrix support.
    coopmat_support: bool,

    // Cache hierarchy
    /// L0/L1 cache per CU in KB.
    l1_cache_kb: u32,
    /// L2 cache in MB.
    l2_cache_mb: u32,
    /// Max workgroup size.
    max_workgroup_size: u32,

    // Derived tuning parameters
    /// Number of invocations per DMMV workgroup; set equal to `wave_size` at detection time.
    dmmv_workgroup_size: u32,
    /// Number of output rows processed per DMMV workgroup; 2 for wave64 (AMD RDNA), 1 for wave32/wave16.
    dmmv_rows_per_workgroup: u32,
    /// Coop matrix tile height.
    matmul_tile_m: u32,
    /// Coop matrix tile width.
    matmul_tile_n: u32,
    /// Flash attention block size.
    flash_attn_block_size: u32,

    /// Return the device name as a byte slice covering only the populated prefix.
    /// @returns A slice of `device_name[0..device_name_len]`; no null terminator or padding included.
    pub fn nameSlice(self: *const GpuConfig) []const u8 {
        return self.device_name[0..self.device_name_len];
    }

    /// Log the detected GPU name, vendor, memory, wave size, cooperative-matrix support, and all derived tuning parameters at info level.
    pub fn log_info(self: *const GpuConfig) void {
        log.info("GPU: {s}", .{self.nameSlice()});
        log.info("  Vendor: {s}", .{@tagName(self.vendor)});
        log.info("  VRAM: {d} MB | BW: {d} GB/s | CUs: {d}", .{ self.vram_mb, self.bandwidth_gbps, self.compute_units });
        log.info("  Wave size: {d} | Max WG: {d}", .{ self.wave_size, self.max_workgroup_size });
        log.info("  CoopMat: {s}", .{if (self.coopmat_support) "yes" else "no"});
        log.info("  DMMV WG: {d} x {d} rows | MatMul tile: {d}x{d}", .{
            self.dmmv_workgroup_size,
            self.dmmv_rows_per_workgroup,
            self.matmul_tile_m,
            self.matmul_tile_n,
        });
        log.info("  Flash attn block: {d}", .{self.flash_attn_block_size});
    }
};

/// Inspect Vulkan device properties and derive runtime tuning defaults.
/// @param instance Active Vulkan instance whose selected physical device should be classified.
/// @returns A GpuConfig populated from Vulkan limits plus ZINC-specific vendor heuristics.
/// @note The classification is heuristic and intentionally biased toward sensible defaults rather than exact SKU detection.
pub fn detect(instance: *const Instance) GpuConfig {
    const props = instance.device_props;
    const name_slice = std.mem.sliceTo(&props.deviceName, 0);

    // Use the Vulkan default subgroup size. Subgroup size control exposes a
    // min/max range, but the minimum is not the size used by ordinary compute
    // pipelines. Intel Battlemage, for example, reports 16..32 while running
    // unrequested compute shaders at subgroupSize=32.
    const subgroup_size: u32 = chooseDefaultSubgroupSize(
        instance.caps.default_subgroup_size,
        instance.caps.min_subgroup_size,
    );

    var config = GpuConfig{
        .vendor = .unknown,
        .device_name = undefined,
        .device_name_len = 0,
        .vram_mb = @intCast(instance.vramBytes() / (1024 * 1024)),
        .bandwidth_gbps = 0,
        .compute_units = 0,
        .wave_size = subgroup_size,
        .coopmat_support = instance.caps.cooperative_matrix,
        .l1_cache_kb = 16,
        .l2_cache_mb = 2,
        .max_workgroup_size = props.limits.maxComputeWorkGroupSize[0],
        .dmmv_workgroup_size = 64,
        .dmmv_rows_per_workgroup = 2,
        .matmul_tile_m = 16,
        .matmul_tile_n = 16,
        .flash_attn_block_size = 128,
    };

    // Copy name
    config.device_name_len = @min(name_slice.len, config.device_name.len);
    @memcpy(config.device_name[0..config.device_name_len], name_slice[0..config.device_name_len]);

    // Identify vendor and architecture
    if (props.vendorID == 0x1002) {
        // AMD — differentiate RDNA3 vs RDNA4 by device ID ranges
        config.vendor = classifyAmd(props.deviceID, name_slice);

        switch (config.vendor) {
            .amd_rdna4 => {
                // Discrete RDNA4 (Navi 48/44): RX 9070 XT / R9700
                // 32KB L1/CU, 6MB L2, up to 64 CUs, 576-640 GB/s GDDR6
                config.l1_cache_kb = 32;
                config.l2_cache_mb = 6;
                config.bandwidth_gbps = 576;
                config.compute_units = 64;
                config.flash_attn_block_size = 256;
                config.matmul_tile_m = 16;
                config.matmul_tile_n = 16;
            },
            .amd_rdna4_apu => {
                // Strix Halo APU iGPU (gfx1151), architecturally RDNA 3.5.
                // Exposes cooperative matrix like RDNA3/RDNA4, so it shares the
                // RDNA4 tuning bucket; memory bandwidth is limited by the
                // shared LPDDR5X-8000/8533 bus.
                // Radeon 8060S (32 CUs, 256-bit bus): ~256 GB/s peak.
                // Higher-end Strix Halo SKUs use the same gfx1151 target.
                config.l1_cache_kb = 32;
                config.l2_cache_mb = 4;
                config.bandwidth_gbps = 256; // shared LPDDR5X, not GDDR6
                config.compute_units = 32;
                config.flash_attn_block_size = 256;
                config.matmul_tile_m = 16;
                config.matmul_tile_n = 16;
            },
            .amd_rdna3 => {
                // RDNA3: 32KB L1/CU, 6MB L2
                config.l1_cache_kb = 32;
                config.l2_cache_mb = 6;
                config.bandwidth_gbps = 480;
                config.compute_units = 48;
                config.flash_attn_block_size = 256;
            },
            else => {
                config.bandwidth_gbps = 256;
                config.compute_units = 32;
            },
        }
        // AMD RDNA3/4 always uses wave64 for optimal scheduling, regardless of
        // what min_subgroup_size reports (AMD supports both 32 and 64, but 64
        // is the native width and what our shaders are tuned for).
        config.wave_size = 64;

        // Nudge users who forgot the documented RADV tuning flag. ZINC is built
        // and benchmarked with RADV_PERFTEST=coop_matrix; without it, RADV may
        // not expose cooperative matrix (some Mesa builds gate it behind this
        // flag), which on APUs such as Strix Halo can prevent startup entirely.
        if (builtin.os.tag == .linux) {
            // RADV_PERFTEST is a comma-separated flag list; match whole tokens
            // so coop_matrix / coop_matrix2 count but a substring inside an
            // unrelated flag name does not.
            var has_coopmat = false;
            if (std.posix.getenv("RADV_PERFTEST")) |val| {
                var flags = std.mem.splitScalar(u8, val, ',');
                while (flags.next()) |flag| {
                    const token = std.mem.trim(u8, flag, " ");
                    if (std.mem.eql(u8, token, "coop_matrix") or std.mem.eql(u8, token, "coop_matrix2")) {
                        has_coopmat = true;
                        break;
                    }
                }
            }
            if (!has_coopmat) {
                log.warn("RADV_PERFTEST=coop_matrix is not set; ZINC is tuned to run with it. Prefix your command (e.g. `RADV_PERFTEST=coop_matrix zinc ...`) for best performance and RADV compatibility.", .{});
            }
        }
    } else if (props.vendorID == 0x10de) {
        // NVIDIA
        config.vendor = .nvidia;
        config.bandwidth_gbps = 512;
    } else if (props.vendorID == 0x8086) {
        config.vendor = classifyIntel(props.deviceID, name_slice);
        if (config.vendor == .intel_arc_xe2) {
            // Xe2 (Battlemage, Arc B-series). Per-SKU bandwidth ranges from
            // 224 GB/s (B50) to 608 GB/s (B70 / B65 / B580) on the supported
            // cards; 640 GB/s here is a high-water-mark headroom assumption,
            // not a per-SKU lookup. See docs/INTEL_GPU_REFERENCE.md for the
            // actual per-card table.
            config.bandwidth_gbps = 640;
            config.compute_units = 32;
            // Xe2 has 256 KB L1 per Xe-core; on B70's 32 Xe-cores that totals
            // 8 MB across the GPU. The shared GPU-wide L2 size is driver-
            // reported and varies by SKU; 8 MB is a planner placeholder.
            config.l1_cache_kb = 256;
            config.l2_cache_mb = 8;
        } else {
            // Xe-HPG (Alchemist, Arc A-series). A770: 512 GB/s, 32 Xe-cores.
            // L1/SLM is configurable on Alchemist; keep a conservative L1 hint.
            config.bandwidth_gbps = 512;
            config.compute_units = 32;
            config.l1_cache_kb = 64;
            config.l2_cache_mb = 8;
        }
        config.flash_attn_block_size = 256;
    }

    // Derive DMMV parameters from wave size
    // wave64 → workgroup 64, 2 rows; wave32 → workgroup 32, 1 row
    config.dmmv_workgroup_size = config.wave_size;
    config.dmmv_rows_per_workgroup = if (config.wave_size == 64) 2 else 1;

    return config;
}

/// Classify AMD GPU architecture from device ID and name.
fn classifyAmd(device_id: u32, name: []const u8) GpuVendor {
    // gfx1201 = RDNA4 Navi 48 (RX 9070 / 9070 XT / 9070 GRE / Radeon AI PRO R9700)
    // gfx1200 = RDNA4 Navi 44 (RX 9060 / 9060 XT)
    // gfx1100-gfx1103 = RDNA3
    _ = device_id;

    // Strix Halo APU iGPU (gfx1151, architecturally RDNA 3.5).
    // Covers Radeon 8060S / 8050S and any other Strix Halo iGPU variant.
    // Must be checked before the discrete RDNA4 branch because the device
    // name contains neither "9070" nor "R9700".
    if (containsIgnoreCase(name, "gfx1151") or
        containsIgnoreCase(name, "8060") or
        containsIgnoreCase(name, "8050"))
    {
        return .amd_rdna4_apu;
    }

    // Discrete RDNA4: Navi 48 (gfx1201 / RX 9070 / R9700) and Navi 44 (gfx1200 / RX 9060)
    if (containsIgnoreCase(name, "gfx1200") or
        containsIgnoreCase(name, "gfx1201") or
        containsIgnoreCase(name, "9070") or
        containsIgnoreCase(name, "9060") or
        containsIgnoreCase(name, "R9700") or
        containsIgnoreCase(name, "RDNA4"))
    {
        return .amd_rdna4;
    }

    if (containsIgnoreCase(name, "gfx1100") or
        containsIgnoreCase(name, "gfx1101") or
        containsIgnoreCase(name, "gfx1102") or
        containsIgnoreCase(name, "gfx1103") or
        containsIgnoreCase(name, "7900") or
        containsIgnoreCase(name, "7800") or
        containsIgnoreCase(name, "7700") or
        containsIgnoreCase(name, "7600"))
    {
        return .amd_rdna3;
    }

    return .amd_other;
}

/// Classify Intel GPU architecture from device ID and name.
fn classifyIntel(device_id: u32, name: []const u8) GpuVendor {
    _ = device_id;
    // Xe2 = Battlemage (Arc B-series): Mesa ANV reports these as
    // "Intel(R) Graphics (BMG Gxx)" where BMG is the Battlemage codename.
    // Retail names like "Arc B580", "Arc B570", "Arc Pro B70" may also appear.
    // Xe-HPG = Alchemist (Arc A-series): default for other Intel Arc GPUs.
    if (containsIgnoreCase(name, "bmg") or
        containsIgnoreCase(name, "xe2") or
        containsIgnoreCase(name, "battlemage") or
        containsIgnoreCase(name, " b5") or
        containsIgnoreCase(name, " b7"))
    {
        return .intel_arc_xe2;
    }
    return .intel_arc;
}

fn containsIgnoreCase(haystack: []const u8, needle: []const u8) bool {
    if (needle.len > haystack.len) return false;
    var i: usize = 0;
    while (i + needle.len <= haystack.len) : (i += 1) {
        var match = true;
        for (0..needle.len) |j| {
            if (std.ascii.toLower(haystack[i + j]) != std.ascii.toLower(needle[j])) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }
    return false;
}

fn chooseDefaultSubgroupSize(default_subgroup_size: u32, min_subgroup_size: u32) u32 {
    if (default_subgroup_size > 0) return default_subgroup_size;
    if (min_subgroup_size > 0) return min_subgroup_size;
    return 32;
}

test "containsIgnoreCase" {
    try std.testing.expect(containsIgnoreCase("AMD Radeon RX 9070 XT", "9070"));
    try std.testing.expect(containsIgnoreCase("RADV GFX1201", "gfx1201"));
    try std.testing.expect(!containsIgnoreCase("NVIDIA RTX 4090", "9070"));
}

test "classifyAmd — discrete RDNA4" {
    try std.testing.expectEqual(GpuVendor.amd_rdna4, classifyAmd(0x7480, "AMD Radeon RX 9070 XT (RADV GFX1201)"));
    try std.testing.expectEqual(GpuVendor.amd_rdna4, classifyAmd(0x7481, "Radeon RX 9060 XT (RADV GFX1200)"));
    try std.testing.expectEqual(GpuVendor.amd_rdna4, classifyAmd(0x7461, "Radeon AI PRO R9700 (RADV GFX1201)"));
}

test "classifyAmd — Strix Halo RDNA4 iGPU" {
    // Real RADV device name reported on a Minisforum MS-S1 / AMD Ryzen AI MAX+ 395
    try std.testing.expectEqual(GpuVendor.amd_rdna4_apu, classifyAmd(0x1586, "Radeon 8060S Graphics (RADV GFX1151)"));
    try std.testing.expectEqual(GpuVendor.amd_rdna4_apu, classifyAmd(0x1585, "Radeon 8050S Graphics (RADV GFX1151)"));
    // Ensure iGPU is NOT mis-classified as discrete RDNA4
    const vendor = classifyAmd(0x1586, "Radeon 8060S Graphics (RADV GFX1151)");
    try std.testing.expect(vendor != .amd_rdna4);
}

test "classifyAmd — RDNA3" {
    try std.testing.expectEqual(GpuVendor.amd_rdna3, classifyAmd(0x744C, "AMD Radeon RX 7900 XTX (RADV GFX1100)"));
}

test "classifyIntel — Arc B-series (Xe2)" {
    try std.testing.expectEqual(GpuVendor.intel_arc_xe2, classifyIntel(0x0000, "Intel Arc Pro B70"));
    try std.testing.expectEqual(GpuVendor.intel_arc_xe2, classifyIntel(0x0000, "Intel Arc B580"));
    // Real Mesa ANV device name on Arc B70 (BMG G31)
    try std.testing.expectEqual(GpuVendor.intel_arc_xe2, classifyIntel(0x0000, "Intel(R) Graphics (BMG G31)"));
}

test "classifyIntel — Arc A-series (Xe-HPG)" {
    try std.testing.expectEqual(GpuVendor.intel_arc, classifyIntel(0x0000, "Intel Arc A770"));
    try std.testing.expectEqual(GpuVendor.intel_arc, classifyIntel(0x0000, "Intel Arc A380"));
}

test "wave_size fallback and DMMV derivation" {
    // Verify the subgroup size fallback and wave_size → DMMV parameter mapping.
    // wave64 (AMD): workgroup 64, 2 rows per workgroup
    {
        const wave: u32 = 64;
        const wg = wave;
        const rows: u32 = if (wave == 64) 2 else 1;
        try std.testing.expectEqual(@as(u32, 64), wg);
        try std.testing.expectEqual(@as(u32, 2), rows);
    }
    // wave32 (NVIDIA): workgroup 32, 1 row
    {
        const wave: u32 = 32;
        const wg = wave;
        const rows: u32 = if (wave == 64) 2 else 1;
        try std.testing.expectEqual(@as(u32, 32), wg);
        try std.testing.expectEqual(@as(u32, 1), rows);
    }
    // wave16 (Intel Xe2): workgroup 16, 1 row
    {
        const wave: u32 = 16;
        const wg = wave;
        const rows: u32 = if (wave == 64) 2 else 1;
        try std.testing.expectEqual(@as(u32, 16), wg);
        try std.testing.expectEqual(@as(u32, 1), rows);
    }
    // Intel Battlemage commonly reports min=16/max=32 while the default
    // subgroup width used by ordinary pipelines is 32.
    {
        try std.testing.expectEqual(@as(u32, 32), chooseDefaultSubgroupSize(32, 16));
    }
    // Fallback when default subgroup is unavailable but min_subgroup_size is set.
    {
        try std.testing.expectEqual(@as(u32, 16), chooseDefaultSubgroupSize(0, 16));
    }
    // Final fallback when neither field is available.
    {
        try std.testing.expectEqual(@as(u32, 32), chooseDefaultSubgroupSize(0, 0));
    }
}

test "GpuConfig name" {
    var config = GpuConfig{
        .vendor = .amd_rdna4,
        .device_name = undefined,
        .device_name_len = 4,
        .vram_mb = 32768,
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
    @memcpy(config.device_name[0..4], "test");
    try std.testing.expectEqualStrings("test", config.nameSlice());
}
