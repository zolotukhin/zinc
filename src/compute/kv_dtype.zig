//! KV cache element type for the Vulkan backend.
//!
//! The cache dominates the context budget: on a hybrid model such as Qwen 3.8
//! 27B (17 attending layers of 65, kv_dim 1024) f32 costs 136 KB per token and
//! f16 costs 68 KB, so f16 doubles the context that fits on a card and halves
//! the bytes attention reads per token. f16 is the usual storage choice for
//! KV caches and the comparison runtime's default.
//! Set ZINC_KV_F16=0 to fall back to f32 (A/B and numerics checks); the choice
//! is fixed at engine init, so every shader that touches the cache is loaded
//! from the matching variant and no call site has to know.
const std = @import("std");

pub fn f16Enabled() bool {
    const raw = std.posix.getenv("ZINC_KV_F16") orelse return true;
    if (raw.len == 0) return true;
    return !(std.mem.eql(u8, raw, "0") or std.ascii.eqlIgnoreCase(raw, "false") or std.ascii.eqlIgnoreCase(raw, "off"));
}

/// Bytes per stored K/V element.
pub fn elementBytes() u64 {
    return if (f16Enabled()) 2 else 4;
}

/// Shader basename for a KV-touching kernel: the `_f16kv` sibling when the
/// cache is f16.
pub fn shaderName(comptime base: []const u8) []const u8 {
    return if (f16Enabled()) base ++ "_f16kv" else base;
}
