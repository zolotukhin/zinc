//! KV-cache storage type for the Vulkan backend, decided once per process from
//! the environment. Shader element indices are always in units of elements
//! (or vec4s), never bytes, so the same dispatch code drives every layout:
//!
//!   f32  4 B/elem  (ZINC_KV_F16=0)
//!   f16  2 B/elem  (default)
//!   q8   32 int8 + one f16 scale per 32-element block, padded to 36 bytes
//!        (ZINC_KV_Q8=1). Rows are block-aligned because head_dim and kv_dim
//!        are multiples of 32, so a vec4 index i lives in block i/8 at uint
//!        i%8, and the block's scale is its uint 8.
const std = @import("std");

pub const Layout = enum { f32, f16, q8 };

fn envOn(name: []const u8, default: bool) bool {
    const raw = std.posix.getenv(name) orelse return default;
    if (raw.len == 0) return default;
    return !(std.mem.eql(u8, raw, "0") or std.ascii.eqlIgnoreCase(raw, "false") or std.ascii.eqlIgnoreCase(raw, "off"));
}

pub fn layout() Layout {
    if (envOn("ZINC_KV_Q8", false)) return .q8;
    return if (envOn("ZINC_KV_F16", true)) .f16 else .f32;
}

pub fn f16Enabled() bool {
    return layout() != .f32;
}

pub fn isQ8() bool {
    return layout() == .q8;
}

/// Bytes a 32-element block occupies. Every KV row is a whole number of blocks.
pub fn bytesPer32Elems() u64 {
    return switch (layout()) {
        .f32 => 128,
        .f16 => 64,
        .q8 => 36,
    };
}

/// Bytes for `elems` elements of cache; `elems` must be a multiple of 32.
pub fn rowBytes(elems: u64) u64 {
    return (elems / 32) * bytesPer32Elems();
}

/// Bytes per element for the two uncompressed layouts. Callers that can see a
/// q8 cache must size rows with rowBytes() instead.
pub fn elementBytes() u64 {
    return switch (layout()) {
        .f32 => 4,
        .f16 => 2,
        .q8 => unreachable,
    };
}

/// Shader variant for the current layout: `base`, `base ++ "_f16kv"`, or
/// `base ++ "_q8kv"`. Variants that do not exist for a layout are reported as
/// missing pipelines at load time.
pub fn shaderName(comptime base: []const u8) []const u8 {
    return switch (layout()) {
        .f32 => base,
        .f16 => base ++ "_f16kv",
        .q8 => base ++ "_q8kv",
    };
}
