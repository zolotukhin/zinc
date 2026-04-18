//! Shared helpers for benchmark and standalone runner entrypoints.
//!
//! This module re-exports the Metal runtime pieces that the benchmark tools
//! need and centralizes the GPU-process-lock error path so the small bench
//! binaries do not duplicate server/runtime boilerplate.
//! @section Inference Runtime
const std = @import("std");

/// Metal device module re-export used by benchmark binaries.
pub const metal_device = @import("metal/device.zig");
/// Metal loader module re-export used by benchmark binaries.
pub const metal_loader = @import("model/loader_metal.zig");
/// Metal buffer helper re-export used by benchmark binaries.
pub const metal_buffer = @import("metal/buffer.zig");
/// Metal command helper re-export used by benchmark binaries.
pub const metal_command = @import("metal/command.zig");
/// Metal pipeline helper re-export used by benchmark binaries.
pub const metal_pipeline = @import("metal/pipeline.zig");
/// Raw Metal shim re-export used by benchmark binaries.
pub const metal_c = @import("metal/c.zig");
/// GGUF parser re-export used by benchmark binaries.
pub const gguf = @import("model/gguf.zig");
/// Tokenizer module re-export used by benchmark binaries.
pub const tokenizer_mod = @import("model/tokenizer.zig");
/// Metal inference runtime re-export used by benchmark binaries.
pub const forward_metal = @import("compute/forward_metal.zig");
/// Cross-process GPU lock helper re-export used by benchmark binaries.
pub const process_lock = @import("gpu/process_lock.zig");

/// Log a user-facing GPU-process-lock error and terminate the benchmark binary.
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
