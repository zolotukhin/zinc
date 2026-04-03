const std = @import("std");

pub const metal_device = @import("metal/device.zig");
pub const metal_loader = @import("model/loader_metal.zig");
pub const metal_buffer = @import("metal/buffer.zig");
pub const metal_command = @import("metal/command.zig");
pub const metal_pipeline = @import("metal/pipeline.zig");
pub const metal_c = @import("metal/c.zig");
pub const gguf = @import("model/gguf.zig");
pub const tokenizer_mod = @import("model/tokenizer.zig");
pub const forward_metal = @import("compute/forward_metal.zig");
pub const process_lock = @import("gpu/process_lock.zig");

pub fn reportGpuProcessLockError(err: anyerror, backend: process_lock.Backend, device_index: u32) noreturn {
    switch (err) {
        error.GpuAlreadyReserved => std.log.err(
            "GPU {s}:{d} is already reserved by another zinc process. Stop the other instance before loading a second model on the same GPU.",
            .{ @tagName(backend), device_index },
        ),
        else => std.log.err("Failed to acquire GPU process lock for {s}:{d}: {s}", .{
            @tagName(backend),
            device_index,
            @errorName(err),
        }),
    }
    std.process.exit(1);
}
