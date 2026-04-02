//! GPU backend abstraction — comptime-resolved, zero runtime overhead.
//! On macOS → Metal backend. On Linux → Vulkan backend.
const builtin = @import("builtin");

pub const is_metal = builtin.os.tag == .macos;
pub const is_vulkan = builtin.os.tag == .linux;

// Backend-specific module imports, resolved at comptime.
// Only the active backend's code is compiled.
pub const backend = if (is_metal)
    @import("../metal/device.zig")
else
    @import("../vulkan/instance.zig");

pub const vk = if (is_vulkan) @import("../vulkan/vk.zig") else struct {};
pub const buffer_mod = if (is_vulkan) @import("../vulkan/buffer.zig") else struct {};
pub const pipeline_mod = if (is_vulkan) @import("../vulkan/pipeline.zig") else struct {};
pub const command_mod = if (is_vulkan) @import("../vulkan/command.zig") else struct {};
pub const gpu_detect_mod = if (is_vulkan) @import("../vulkan/gpu_detect.zig") else struct {};

const std = @import("std");

test "backend selection is correct for this platform" {
    if (builtin.os.tag == .macos) {
        try std.testing.expect(is_metal);
        try std.testing.expect(!is_vulkan);
    } else if (builtin.os.tag == .linux) {
        try std.testing.expect(is_vulkan);
        try std.testing.expect(!is_metal);
    }
}

test "backend module resolves to correct type" {
    if (is_metal) {
        // On macOS, backend should be the Metal device module
        const MetalDevice = backend.MetalDevice;
        try std.testing.expect(@sizeOf(MetalDevice) > 0);
    }
}
