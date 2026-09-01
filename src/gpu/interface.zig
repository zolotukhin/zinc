//! GPU backend abstraction — comptime-resolved, zero runtime overhead.
//! macOS → Metal. Linux → Vulkan by default, CUDA with `-Dbackend=cuda`,
//! or ROCm/HIP with `-Dbackend=rocm`.
//! @section Inference Runtime
const builtin = @import("builtin");
const std = @import("std");
const build_options = @import("build_options");

/// True when compiling for macOS (Metal backend).
pub const is_metal = builtin.os.tag == .macos;
/// True when compiling for Linux with `-Dbackend=rocm` (AMD + HIP).
pub const is_rocm = builtin.os.tag == .linux and std.mem.eql(u8, build_options.backend, "rocm");
/// True for the CUDA-family orchestration shared by native CUDA and ROCm/HIP.
pub const is_cuda = builtin.os.tag == .linux and
    (std.mem.eql(u8, build_options.backend, "cuda") or is_rocm);
/// True when compiling for Linux with the Vulkan backend (the Linux default).
pub const is_vulkan = builtin.os.tag == .linux and !is_cuda;

// Backend-specific module imports, resolved at comptime.
// Only the active backend's code is compiled.
/// Platform-specific GPU device module (Metal / CUDA / Vulkan).
pub const backend = if (is_metal)
    @import("../metal/device.zig")
else if (is_cuda)
    @import("../cuda/device.zig")
else
    @import("../vulkan/instance.zig");

/// Vulkan C bindings (empty struct off-Vulkan).
pub const vk = if (is_vulkan) @import("../vulkan/vk.zig") else struct {};
/// Device buffer allocation + H2D/D2H staging (CUDA / Vulkan).
pub const buffer_mod = if (is_cuda)
    @import("../cuda/buffer.zig")
else if (is_vulkan)
    @import("../vulkan/buffer.zig")
else
    struct {};
/// Compute pipeline creation (CUDA NVRTC / Vulkan).
pub const pipeline_mod = if (is_cuda)
    @import("../cuda/pipeline.zig")
else if (is_vulkan)
    @import("../vulkan/pipeline.zig")
else
    struct {};
/// Command/stream recording (CUDA streams+events / Vulkan command buffers).
pub const command_mod = if (is_cuda)
    @import("../cuda/command.zig")
else if (is_vulkan)
    @import("../vulkan/command.zig")
else
    struct {};
/// GPU detection / tuning (Vulkan only for now; CUDA caps come from device.zig).
pub const gpu_detect_mod = if (is_vulkan) @import("../vulkan/gpu_detect.zig") else struct {};

test "backend selection is correct for this platform" {
    if (builtin.os.tag == .macos) {
        try std.testing.expect(is_metal);
        try std.testing.expect(!is_vulkan);
    } else if (builtin.os.tag == .linux) {
        try std.testing.expect(is_vulkan != is_cuda);
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
