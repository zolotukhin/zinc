//! Aggregate entry point for the Vulkan primitives.
//!
//! The engine imports these files directly by relative path, but a standalone
//! tool cannot reach outside its own module tree, so kernel-level validators
//! (`tools/validate_*_moe_cols.zig`) import this as the `vulkan` module instead.
const std = @import("std");

pub const vk = @import("vk.zig");
pub const instance = @import("instance.zig");
pub const Instance = instance.Instance;
pub const buffer = @import("buffer.zig");
pub const Buffer = buffer.Buffer;
pub const command = @import("command.zig");
pub const pipeline = @import("pipeline.zig");
pub const gpu_detect = @import("gpu_detect.zig");

test "aggregate module exposes the primitives tools need" {
    try std.testing.expect(@hasDecl(instance, "Instance"));
    try std.testing.expect(@hasDecl(buffer, "Buffer"));
    try std.testing.expect(@hasDecl(command, "CommandBuffer"));
    try std.testing.expect(@hasDecl(pipeline, "createFromSpirvWithOptions"));
}
