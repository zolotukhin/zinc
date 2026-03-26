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
    amd_other,
    nvidia,
    intel_arc,
    unknown,
};

/// Auto-detected GPU capabilities and derived tuning parameters.
pub const GpuConfig = struct {
    vendor: GpuVendor,
    device_name: [256]u8,
    device_name_len: usize,
    vram_mb: u32,
    bandwidth_gbps: u32,
    compute_units: u32,
    wave_size: u32,
    coopmat_support: bool,

    // Cache hierarchy
    l1_cache_kb: u32,
    l2_cache_mb: u32,
    max_workgroup_size: u32,

    // Derived tuning parameters
    dmmv_workgroup_size: u32,
    dmmv_rows_per_workgroup: u32,
    matmul_tile_m: u32,
    matmul_tile_n: u32,
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

/// Detect GPU capabilities and derive tuning parameters.
pub fn detect(instance: *const Instance) GpuConfig {
    const props = instance.device_props;
    const name_slice = std.mem.sliceTo(&props.deviceName, 0);

    var config = GpuConfig{
        .vendor = .unknown,
        .device_name = undefined,
        .device_name_len = 0,
        .vram_mb = @intCast(instance.vramBytes() / (1024 * 1024)),
        .bandwidth_gbps = 0,
        .compute_units = 0,
        .wave_size = 32,
        .coopmat_support = false,
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
        config.wave_size = 64; // wave64 optimal on RDNA3/4

        switch (config.vendor) {
            .amd_rdna4 => {
                // RDNA4: 32KB L1/CU, 6MB L2, 64 CUs (9070 XT), 576 GB/s
                config.l1_cache_kb = 32;
                config.l2_cache_mb = 6;
                config.bandwidth_gbps = 576;
                config.compute_units = 64;
                config.coopmat_support = true;
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
                config.coopmat_support = true;
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
        config.wave_size = 32;
        config.bandwidth_gbps = 512;
    } else if (props.vendorID == 0x8086) {
        // Intel
        config.vendor = .intel_arc;
        config.wave_size = 32;
        config.bandwidth_gbps = 256;
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

    // Heuristic: check for RDNA4 keywords in device name
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
