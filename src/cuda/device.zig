//! CUDA device wrapper — NVIDIA GPU backend (mirrors src/metal/device.zig).
//!
//! Owns CUDA context init and capability queries used by the loader,
//! diagnostics, and CUDA inference runtime.
//! @section CUDA Runtime
const std = @import("std");
const shim = @import("c.zig").shim;

/// Capability snapshot queried once from the active CUDA device.
pub const CudaCapabilities = struct {
    /// major*10 + minor, e.g. 120 = sm_120 (Blackwell), 89 = sm_89 (Ada).
    compute_capability: u32,
    sm_count: u32,
    warp_size: u32,
    max_threads_per_block: u32,
    max_shared_mem_per_block: u32,
    total_memory: u64,
};

/// Active CUDA device wrapper plus capability metadata used by the backend.
pub const CudaDevice = struct {
    ctx: ?*shim.CudaCtx,
    caps: CudaCapabilities,
    device_index: u32,
    allocator: std.mem.Allocator,

    /// Initialize a specific CUDA device by index and query its capabilities.
    /// @param allocator Allocator stored for future use by the backend.
    /// @param device_index Zero-based CUDA device ordinal.
    /// @returns Initialised `CudaDevice` with a live context, or `error.CudaInitFailed` if the shim rejects the index.
    pub fn init(allocator: std.mem.Allocator, device_index: u32) !CudaDevice {
        const ctx = shim.cuda_init(@intCast(device_index));
        if (ctx == null) return error.CudaInitFailed;
        return .{
            .ctx = ctx,
            .caps = queryCaps(ctx),
            .device_index = device_index,
            .allocator = allocator,
        };
    }

    /// Initialize the highest-compute-capability device (prefer 5090 over 4090).
    /// Probes up to 16 device indices, selects the one with the largest compute capability
    /// value, then opens a final context on that device via `init`.
    /// @param allocator Forwarded to `init` for the selected device.
    /// @returns Initialised `CudaDevice` for the best device, or `error.CudaNoDevice` if no device is found.
    pub fn initBest(allocator: std.mem.Allocator) !CudaDevice {
        // Allow explicit device override (e.g. ZINC_CUDA_DEVICE=1 for 4090)
        if (std.posix.getenv("ZINC_CUDA_DEVICE")) |dev_str| {
            if (std.fmt.parseInt(u32, dev_str, 10)) |dev_idx| {
                return init(allocator, dev_idx);
            } else |_| {}
        }
        var best_index: i64 = -1;
        var best_cc: u32 = 0;
        var idx: u32 = 0;
        while (idx < 16) : (idx += 1) {
            const ctx = shim.cuda_init(@intCast(idx));
            if (ctx == null) break;
            const cc = shim.cuda_compute_capability(ctx);
            if (cc > best_cc) {
                best_cc = cc;
                best_index = @intCast(idx);
            }
            shim.cuda_destroy(ctx);
        }
        if (best_index < 0) return error.CudaNoDevice;
        return init(allocator, @intCast(best_index));
    }

    fn queryCaps(ctx: ?*shim.CudaCtx) CudaCapabilities {
        return .{
            .compute_capability = shim.cuda_compute_capability(ctx),
            .sm_count = shim.cuda_sm_count(ctx),
            .warp_size = shim.cuda_warp_size(ctx),
            .max_threads_per_block = shim.cuda_max_threads_per_block(ctx),
            .max_shared_mem_per_block = shim.cuda_max_shared_mem_per_block(ctx),
            .total_memory = shim.cuda_total_memory(ctx),
        };
    }

    /// Destroy the active CUDA context.
    pub fn deinit(self: *CudaDevice) void {
        if (self.ctx) |ctx| {
            shim.cuda_destroy(ctx);
            self.ctx = null;
        }
    }

    /// Total device memory (VRAM capacity) in bytes.
    pub fn totalMemory(self: *const CudaDevice) u64 {
        return self.caps.total_memory;
    }

    /// Currently free device memory in bytes; 0 if the context has been destroyed.
    pub fn freeMemory(self: *const CudaDevice) u64 {
        if (self.ctx) |ctx| return shim.cuda_free_memory(ctx);
        return 0;
    }

    /// Compute capability encoded as `major*10 + minor` (e.g. 120 = sm_120).
    pub fn computeCapability(self: *const CudaDevice) u32 {
        return self.caps.compute_capability;
    }

    /// Number of streaming multiprocessors (SMs) on the device.
    pub fn smCount(self: *const CudaDevice) u32 {
        return self.caps.sm_count;
    }

    /// Threads per warp (32 on all current NVIDIA hardware).
    pub fn warpSize(self: *const CudaDevice) u32 {
        return self.caps.warp_size;
    }

    /// Maximum shared memory per block available with opt-in dynamic allocation, in bytes.
    pub fn maxSharedMemPerBlock(self: *const CudaDevice) u64 {
        return self.caps.max_shared_mem_per_block;
    }

    /// Copy the device name into `buf` and return the NUL-trimmed slice.
    /// @param buf Caller-supplied scratch buffer; 64–256 bytes is typically sufficient.
    /// @returns Slice into `buf` containing the device name without a trailing NUL, or an empty slice if the context has been destroyed.
    pub fn name(self: *const CudaDevice, buf: []u8) []const u8 {
        if (self.ctx) |ctx| {
            shim.cuda_device_name(ctx, buf.ptr, buf.len);
            const len = std.mem.indexOfScalar(u8, buf, 0) orelse buf.len;
            return buf[0..len];
        }
        return "";
    }

    /// Copy the native kernel compilation target (for example `sm_120` or
    /// `gfx1201`) into `buf` and return the NUL-trimmed slice.
    pub fn arch(self: *const CudaDevice, buf: []u8) []const u8 {
        if (self.ctx) |ctx| {
            shim.cuda_device_arch(ctx, buf.ptr, buf.len);
            const len = std.mem.indexOfScalar(u8, buf, 0) orelse buf.len;
            return buf[0..len];
        }
        return "";
    }
};
