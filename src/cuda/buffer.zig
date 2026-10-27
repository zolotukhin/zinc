//! CUDA buffer wrapper — device-local allocations with optional pinned staging.
//!
//! Unlike Metal (Apple unified memory), CUDA device memory is NOT CPU-visible;
//! host<->device transfers are explicit (`upload`/`download`), staged through
//! pinned host memory. Mirrors src/metal/buffer.zig.
//! @section CUDA Runtime
const std = @import("std");
const shim = @import("c.zig").shim;

/// CUDA device buffer handle plus optional pinned-host staging mirror.
pub const CudaBuffer = struct {
    handle: ?*shim.CudaBuf,
    size: usize,
    /// Pinned host staging pointer for staged buffers (null for plain device buffers).
    host_ptr: ?[*]u8 = null,
    /// False for lightweight aliases into a larger buffer owned elsewhere.
    owns_handle: bool = true,

    /// Raw device pointer (CUdeviceptr as u64) for kernel arg packing / aliasing.
    pub fn devicePtr(self: *const CudaBuffer) u64 {
        if (self.handle) |h| return shim.cuda_buffer_device_ptr(h);
        return 0;
    }

    /// Pinned host staging pointer, if this buffer was created staged.
    pub fn contents(self: *const CudaBuffer) ?[*]u8 {
        return self.host_ptr;
    }
};

/// Allocate a device-local buffer (the common case for weights/activations/state).
/// @param ctx  CUDA context that owns the allocation.
/// @param size Number of bytes to allocate on the device.
/// @returns A `CudaBuffer` with no host staging pointer; use `createBufferStaged` when CPU access is needed.
/// @note Returns `error.CudaBufferAllocFailed` if the shim returns a null handle.
pub fn createBuffer(ctx: ?*shim.CudaCtx, size: usize) !CudaBuffer {
    const handle = shim.cuda_create_buffer(ctx, size);
    if (handle == null) return error.CudaBufferAllocFailed;
    return .{ .handle = handle, .size = size };
}

/// Allocate a device buffer paired with a pinned-host staging mirror for fast
/// `upload`/`download`. The host pointer is exposed via `contents()`.
/// @param ctx  CUDA context that owns the allocation.
/// @param size Number of bytes to allocate on both the device and in pinned host memory.
/// @returns A `CudaBuffer` whose `host_ptr` field is non-null; `contents()` returns the pinned staging address.
/// @note Returns `error.CudaBufferAllocFailed` if the shim returns a null handle.
pub fn createBufferStaged(ctx: ?*shim.CudaCtx, size: usize) !CudaBuffer {
    var cpu_ptr: ?*anyopaque = null;
    const handle = shim.cuda_create_buffer_staged(ctx, size, &cpu_ptr);
    if (handle == null) return error.CudaBufferAllocFailed;
    return .{ .handle = handle, .size = size, .host_ptr = @ptrCast(cpu_ptr) };
}

/// Copy an existing host mapping (e.g. mmap'd weights) to a new device-local buffer.
/// Unlike Metal's zero-copy wrapMmap, this performs a full host-to-device copy.
/// @param ctx      CUDA context that will own the resulting device allocation.
/// @param host_ptr Pointer to the host memory region to copy from (typically an mmap'd file mapping).
/// @param size     Number of bytes to transfer.
/// @returns A device-local `CudaBuffer`; `host_ptr` is null (data lives only on device after this call).
/// @note Returns `error.CudaMmapUploadFailed` if the shim returns a null handle.
pub fn uploadMmap(ctx: ?*shim.CudaCtx, host_ptr: *const anyopaque, size: usize) !CudaBuffer {
    const handle = shim.cuda_upload_mmap(ctx, host_ptr, size);
    if (handle == null) return error.CudaMmapUploadFailed;
    return .{ .handle = handle, .size = size };
}

/// Create a lightweight view into an existing buffer's device allocation.
/// @param base   Parent buffer whose device memory is aliased.
/// @param offset Byte offset from the start of `base`'s device allocation.
/// @param size   Number of bytes the alias covers.
/// @returns A `CudaBuffer` with `owns_handle = false`; calling `freeBuffer` on it releases only the wrapper, not the parent's device memory.
/// @note Returns `error.CudaBufferAllocFailed` if the shim returns a null handle.
pub fn aliasBuffer(base: *const CudaBuffer, offset: usize, size: usize) !CudaBuffer {
    const handle = shim.cuda_alias_buffer(base.handle, offset, size);
    if (handle == null) return error.CudaBufferAllocFailed;
    // CUDA and ROCm set the C handle's owns flag to zero. The Zig flag mirrors
    // that ownership; releasing this handle never releases base's allocation.
    return .{ .handle = handle, .size = size, .owns_handle = false };
}

/// Free a buffer handle (the shim only releases device memory if this buffer
/// owns it — aliases just free the wrapper). Safe with a null handle.
pub fn freeBuffer(buf: *CudaBuffer) void {
    if (buf.handle) |h| {
        // The backend checks CudaBuf.owns, then always destroys the wrapper.
        // Non-owning views must still be passed here to avoid metadata leaks.
        shim.cuda_free_buffer(h);
        buf.handle = null;
    }
}

/// Copy bytes from host to device (synchronous on the context stream).
/// @param ctx  CUDA context whose stream is used for the transfer.
/// @param buf  Destination device buffer; must be at least `data.len` bytes.
/// @param data Source slice on the host.
pub fn upload(ctx: ?*shim.CudaCtx, buf: *const CudaBuffer, data: []const u8) void {
    shim.cuda_upload(ctx, buf.handle, @ptrCast(data.ptr), data.len);
}

/// Copy bytes from device to host (synchronous on the context stream).
/// @param ctx CUDA context whose stream is used for the transfer.
/// @param buf Source device buffer; must be at least `dst.len` bytes.
/// @param dst Destination slice on the host.
pub fn download(ctx: ?*shim.CudaCtx, buf: *const CudaBuffer, dst: []u8) void {
    shim.cuda_download(ctx, buf.handle, @ptrCast(dst.ptr), dst.len);
}

/// Async host→device copy (no stream sync). Capturable into a CUDA graph; the
/// host side must be pinned (see `allocHost`). @see download(Async) for D2H.
pub fn uploadAsync(ctx: ?*shim.CudaCtx, buf: *const CudaBuffer, data: []const u8) void {
    shim.cuda_upload_async(ctx, buf.handle, @ptrCast(data.ptr), data.len);
}

/// Async device→host copy (no stream sync). Capturable into a CUDA graph; the
/// destination must be pinned host memory (see `allocHost`).
pub fn downloadAsync(ctx: ?*shim.CudaCtx, buf: *const CudaBuffer, dst: []u8) void {
    shim.cuda_download_async(ctx, buf.handle, @ptrCast(dst.ptr), dst.len);
}

/// Pinned (page-locked) host allocation of `n` `T` values, required as the host
/// endpoint of an async graph-captured copy. Free with `freeHost`.
pub fn allocHost(comptime T: type, n: usize) ![]T {
    const p = shim.cuda_alloc_host(n * @sizeOf(T)) orelse return error.OutOfMemory;
    const tp: [*]T = @ptrCast(@alignCast(p));
    return tp[0..n];
}

/// Free a pinned host allocation from `allocHost`.
pub fn freeHost(slice: anytype) void {
    shim.cuda_free_host(@ptrCast(@constCast(slice.ptr)));
}
