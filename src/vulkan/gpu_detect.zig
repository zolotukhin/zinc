//! Inspect the selected Vulkan device and derive architecture-specific tuning defaults.
//! @section Hardware Detection
//! The heuristics here convert raw Vulkan device properties into the settings
//! used by DMMV, matmul, and attention dispatch code.
const std = @import("std");
const vk = @import("vk.zig");
const Instance = @import("instance.zig").Instance;

const log = std.log.scoped(.gpu_detect);

/// GPU vendor and architecture buckets used by ZINC's tuning heuristics.
pub const GpuVendor = enum {
    amd_rdna3,
    amd_rdna4,
    /// AMD RDNA4 iGPU in a unified-memory APU (e.g. Strix Halo / gfx1151).
    /// Same ISA and cooperative-matrix support as discrete RDNA4, but bandwidth
    /// is constrained by the shared LPDDR5X system-memory bus (~256 GB/s on
    /// 128 GB configs) rather than dedicated GDDR6.  Tuning defaults differ
    /// from amd_rdna4 accordingly.
    amd_rdna4_apu,
    amd_other,
    nvidia,
    /// Intel Arc Xe2 (Battlemage, B-series): SIMD16 subgroup.
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
    /// DMMV workgroup size.
    dmmv_workgroup_size: u32,
    /// Rows per DMMV workgroup.
    dmmv_rows_per_workgroup: u32,
    /// Coop matrix tile height.
    matmul_tile_m: u32,
    /// Coop matrix tile width.
    matmul_tile_n: u32,
    /// Flash attention block size.
    flash_attn_block_size: u32,

    /// Return the active device name as a trimmed byte slice.
    /// @returns The populated prefix of `device_name`.
    pub fn nameSlice(self: *const GpuConfig) []const u8 {
        return self.device_name[0..self.device_name_len];
    }

    /// Log the detected GPU properties and derived tuning parameters.
    /// @param self Derived GPU configuration for the selected device.
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

    // Use Vulkan-reported subgroup size when available, otherwise fall back to 32.
    const subgroup_size: u32 = if (instance.caps.min_subgroup_size > 0)
        instance.caps.min_subgroup_size
    else
        16;

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
                // RDNA4 iGPU in unified-memory APU (gfx1151 / Strix Halo)
                // Same ISA and CoopMat support as discrete RDNA4, but memory
                // bandwidth is limited by the shared LPDDR5X-8000/8533 bus.
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
    } else if (props.vendorID == 0x10de) {
        // NVIDIA
        config.vendor = .nvidia;
        config.bandwidth_gbps = 512;
    } else if (props.vendorID == 0x8086) {
        config.vendor = classifyIntel(props.deviceID, name_slice);
        if (config.vendor == .intel_arc_xe2) {
            // Xe2 (Battlemage, Arc B-series).
            // B770 Pro: 32 Xe2-HPG cores, 256-bit GDDR6 @ 20 Gbps ≈ 640 GB/s.
            config.bandwidth_gbps = 640;
            config.compute_units = 32;
            config.l1_cache_kb = 64;
            config.l2_cache_mb = 8;
        } else {
            // Xe-HPG (Alchemist, Arc A-series).
            // A770: 512 GB/s, 32 Xe-cores.
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
    // gfx1200/gfx1201 = RDNA4 (Navi 48/44)
    // gfx1100-gfx1103 = RDNA3
    // Device IDs: RDNA4 starts at 0x15xx range
    _ = device_id;

    // RDNA4 iGPU: Strix Halo APU (gfx1151)
    // Covers Radeon 8060S / 8050S and any other Strix Halo iGPU variant.
    // Must be checked before the discrete RDNA4 branch because the device
    // name contains neither "9070" nor "R9700".
    if (containsIgnoreCase(name, "gfx1151") or
        containsIgnoreCase(name, "8060") or
        containsIgnoreCase(name, "8050"))
    {
        return .amd_rdna4_apu;
    }

    // Discrete RDNA4: Navi 48 (gfx1200 / RX 9070) and Navi 44 (gfx1201 / RX 9060 / R9700)
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
    // Retail names like "Arc B580" / "Arc B770" may also appear.
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

test "containsIgnoreCase" {
    try std.testing.expect(containsIgnoreCase("AMD Radeon RX 9070 XT", "9070"));
    try std.testing.expect(containsIgnoreCase("RADV GFX1201", "gfx1201"));
    try std.testing.expect(!containsIgnoreCase("NVIDIA RTX 4090", "9070"));
}

test "classifyAmd — discrete RDNA4" {
    try std.testing.expectEqual(GpuVendor.amd_rdna4, classifyAmd(0x7480, "AMD Radeon RX 9070 XT (RADV GFX1200)"));
    try std.testing.expectEqual(GpuVendor.amd_rdna4, classifyAmd(0x7481, "Radeon RX 9060 XT (RADV GFX1201)"));
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
    try std.testing.expectEqual(GpuVendor.intel_arc_xe2, classifyIntel(0x0000, "Intel Arc B770 Pro"));
    try std.testing.expectEqual(GpuVendor.intel_arc_xe2, classifyIntel(0x0000, "Intel Arc B580"));
    // Real Mesa ANV device name on Arc B70 (BMG G31)
    try std.testing.expectEqual(GpuVendor.intel_arc_xe2, classifyIntel(0x0000, "Intel(R) Graphics (BMG G31)"));
}

test "classifyIntel — Arc A-series (Xe-HPG)" {
    try std.testing.expectEqual(GpuVendor.intel_arc, classifyIntel(0x0000, "Intel Arc A770"));
    try std.testing.expectEqual(GpuVendor.intel_arc, classifyIntel(0x0000, "Intel Arc A380"));
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
