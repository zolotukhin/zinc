//! CUDA command wrapper — kernel dispatch and stream/event synchronization
//! (mirrors src/metal/command.zig). A CudaCommand wraps the context's CUstream
//! plus a backend-owned completion event. ROCm creates that event lazily in
//! `commitAsync`, while CUDA may allocate it when the command begins.
//! @section CUDA Runtime
const std = @import("std");
const shim = @import("c.zig").shim;
const CudaBuffer = @import("buffer.zig").CudaBuffer;
const CudaPipeline = @import("pipeline.zig").CudaPipeline;

/// A recorded stream batch that launches compute kernels on the GPU.
pub const CudaCommand = struct {
    /// Opaque handle to the C shim command (stream + completion event).
    handle: ?*shim.CudaCmd,
    dispatch_count: u32 = 0,

    /// Launch a kernel on this command's stream.
    /// Bound `bufs` become the leading device-pointer args (capped at 32);
    /// `push_data` (push_size bytes) is the trailing by-value push-constant arg.
    /// @param pipe     Compiled CUDA pipeline (kernel + argument layout) to execute.
    /// @param grid     3-D grid dimensions in thread-blocks (x, y, z).
    /// @param block    3-D thread-block dimensions (x, y, z).
    /// @param bufs     Device buffers passed as pointer args before the push constant.
    /// @param push_data Pointer to the push-constant struct, or null if unused.
    /// @param push_size Byte size of the push-constant struct.
    /// @param shared_bytes Dynamic shared memory in bytes to allocate per block.
    pub fn dispatch(
        self: *CudaCommand,
        pipe: *const CudaPipeline,
        grid: [3]u32,
        block: [3]u32,
        bufs: []const *const CudaBuffer,
        push_data: ?*const anyopaque,
        push_size: usize,
        shared_bytes: u32,
    ) void {
        if (self.handle == null or pipe.handle == null) return;
        self.dispatch_count += 1;

        var c_bufs: [32]?*shim.CudaBuf = undefined;
        const n_bufs: u32 = @intCast(@min(bufs.len, 32));
        for (bufs[0..n_bufs], 0..n_bufs) |b, i| {
            c_bufs[i] = b.handle;
        }

        shim.cuda_dispatch(
            self.handle,
            pipe.handle,
            &grid,
            &block,
            @ptrCast(&c_bufs),
            n_bufs,
            push_data,
            push_size,
            shared_bytes,
        );
    }

    /// Same-stream launches are implicitly ordered; no-op for a single stream.
    pub fn barrier(self: *CudaCommand) void {
        if (self.handle) |h| shim.cuda_barrier(h);
    }

    /// Record completion and block until the stream drains.
    pub fn commitAndWait(self: *CudaCommand) void {
        if (self.handle) |h| {
            shim.cuda_commit_and_wait(h);
            self.handle = null;
        }
    }

    /// Record completion and return immediately; call `wait` later to sync.
    pub fn commitAsync(self: *CudaCommand) void {
        if (self.handle) |h| {
            shim.cuda_commit_async(h);
            // handle stays valid — call wait() later
        }
    }

    /// Block on a previously async-committed command's completion event.
    pub fn wait(self: *CudaCommand) void {
        if (self.handle) |h| {
            shim.cuda_wait(h);
            self.handle = null;
        }
    }

    /// Release a command whose completion is guaranteed by a later queue-ordered
    /// synchronization (e.g. the caller waited on a subsequent command in the
    /// same stream).  Frees the shim handle without issuing another wait.
    pub fn releaseCompleted(self: *CudaCommand) void {
        if (self.handle) |h| {
            shim.cuda_release_completed(h);
            self.handle = null;
        }
    }
};

/// Begin a new command batch on the given context. The backend controls whether
/// its completion event is allocated now or lazily by `commitAsync`.
/// @param ctx    Active CUDA context that owns the stream; must not be null.
/// @returns      A fresh `CudaCommand` ready for kernel dispatches.
/// @note         Returns `error.CudaCommandFailed` if the shim cannot allocate
///               the underlying command resources.
pub fn beginCommand(ctx: ?*shim.CudaCtx) !CudaCommand {
    const handle = shim.cuda_begin_command(ctx);
    if (handle == null) return error.CudaCommandFailed;
    return .{ .handle = handle };
}
