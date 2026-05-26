//! AMDGPU KFD (`/dev/kfd`) bring-up for the T1 PM4-direct tier.
//!
//! The design's T1 tier submits PM4 packets straight to the AMD command
//! processor. On Linux that means the KFD compute path (`/dev/kfd` +
//! `AMDKFD_IOC_*`), the same userspace ABI ROCm/HSA and tinygrad ride on. It
//! works on every `amdgpu` kernel that ships KFD and does not depend on the
//! experimental user-mode-queue (UMQ / `DRM_IOCTL_AMDGPU_USERQ`) ABI, which on
//! the R9700 (gfx1201, kernel 6.17, `uni_mes` firmware) the kernel rejects with
//! "Usermode queue is not supported for this IP" — see `umq.zig` and §14 of the
//! design doc.
//!
//! This module brings up the T1 PM4-direct path on the kernel ABI that works:
//!   * open `/dev/kfd` + the render node, `AMDKFD_IOC_GET_VERSION`,
//!   * match the render minor against `/sys/.../kfd/topology/nodes/*`,
//!   * `AMDKFD_IOC_ACQUIRE_VM`, `AMDKFD_IOC_GET_PROCESS_APERTURES_NEW`,
//!   * `bringUpPath`: reserve a VA window inside the GPUVM aperture,
//!     `ALLOC_MEMORY_OF_GPU` (GTT, writable) + `mmap` + `MAP_MEMORY_TO_GPU` +
//!     a CPU round-trip, then `UNMAP_MEMORY_FROM_GPU` + `FREE_MEMORY_OF_GPU`,
//!   * `createComputeQueueSmokePath`: allocate the ring / wptr / rptr / EOP /
//!     CWSR buffer objects the way `kfd_queue_acquire_buffers` validates them,
//!     `AMDKFD_IOC_CREATE_QUEUE` (PM4 compute), stage a couple of PM4 NOP
//!     packets into the ring, then `AMDKFD_IOC_DESTROY_QUEUE` and tear down.
//! Ringing the doorbell and retiring a PM4 fence is the next bring-up step.
//! @section Inference Runtime
const std = @import("std");
const builtin = @import("builtin");
const packet = @import("packet.zig");

const linux = std.os.linux;

/// Default DRM render node used when the caller does not supply one. The
/// renderD128 minor is the standard single-GPU choice on AMD Linux systems.
pub const default_render_node = "/dev/dri/renderD128";
/// Path to the KFD compute device used for every AMDKFD_IOC_* ioctl.
pub const kfd_device_node = "/dev/kfd";
/// Sysfs root that enumerates KFD topology nodes (one subdirectory per node,
/// each carrying `gpu_id` and a `properties` file that drives queue sizing).
pub const topology_nodes_dir = "/sys/devices/virtual/kfd/kfd/topology/nodes";
/// Minimum KFD ABI major version required for the PM4 compute path; the
/// bring-up fails fast with `kfd_version_too_old` below this number.
pub const min_kfd_major: u32 = 1;

const kfd_ioctl_base: u8 = 'K';

// AMDKFD_IOC_* command numbers (uapi/linux/kfd_ioctl.h).
const kfd_ioc_get_version_nr: u8 = 0x01;
const kfd_ioc_create_queue_nr: u8 = 0x02;
const kfd_ioc_destroy_queue_nr: u8 = 0x03;
const kfd_ioc_get_process_apertures_new_nr: u8 = 0x14;
const kfd_ioc_acquire_vm_nr: u8 = 0x15;
const kfd_ioc_alloc_memory_of_gpu_nr: u8 = 0x16;
const kfd_ioc_free_memory_of_gpu_nr: u8 = 0x17;
const kfd_ioc_map_memory_to_gpu_nr: u8 = 0x18;
const kfd_ioc_unmap_memory_from_gpu_nr: u8 = 0x19;

// KFD_IOC_ALLOC_MEM_FLAGS_*.
/// Allocate the BO out of device VRAM (local frame buffer).
pub const ALLOC_MEM_FLAGS_VRAM: u32 = 1 << 0;
/// Allocate the BO out of system GTT memory (the path used by every smoke BO).
pub const ALLOC_MEM_FLAGS_GTT: u32 = 1 << 1;
/// Pin a userptr range as the BO backing; not used by the bring-up smoke path.
pub const ALLOC_MEM_FLAGS_USERPTR: u32 = 1 << 2;
/// Allocate a doorbell page so a userspace queue can ring its wptr doorbell.
pub const ALLOC_MEM_FLAGS_DOORBELL: u32 = 1 << 3;
/// Request a CPU-coherent BO (snooped on x86); needed by ring/wptr/rptr BOs.
pub const ALLOC_MEM_FLAGS_COHERENT: u32 = 1 << 26;
/// Mark the BO as PCIe-visible / exportable to peer devices.
pub const ALLOC_MEM_FLAGS_PUBLIC: u32 = 1 << 29;
/// Mark the BO as containing GPU-executable code (shader binaries).
pub const ALLOC_MEM_FLAGS_EXECUTABLE: u32 = 1 << 30;
/// Map the BO with write permission on the GPU side (default for smoke BOs).
pub const ALLOC_MEM_FLAGS_WRITABLE: u32 = 1 << 31;

// KFD_IOC_QUEUE_TYPE_*.
/// PM4 compute queue type passed to `AMDKFD_IOC_CREATE_QUEUE` for MEC pipes.
pub const QUEUE_TYPE_COMPUTE: u32 = 0x0;
/// SDMA queue type; copy engine queue, not used by the PM4 compute bring-up.
pub const QUEUE_TYPE_SDMA: u32 = 0x1;
/// AQL compute queue type (HSA packet processor format) used by ROCm/HSA.
pub const QUEUE_TYPE_COMPUTE_AQL: u32 = 0x2;

/// `AMDKFD_IOC_GET_VERSION` ioctl args — the KFD ABI version reported by the
/// running kernel. Matches `struct kfd_ioctl_get_version_args` from uapi.
pub const GetVersionArgs = extern struct {
    major_version: u32,
    minor_version: u32,
};

/// `AMDKFD_IOC_ACQUIRE_VM` ioctl args — binds the calling process's GPUVM to
/// the DRM render node fd so subsequent allocations land in the right address
/// space.
pub const AcquireVmArgs = extern struct {
    drm_fd: u32,
    gpu_id: u32,
};

/// One device aperture entry returned by `GET_PROCESS_APERTURES_NEW`: the LDS,
/// scratch, and GPUVM windows that this process is allowed to use on the
/// indicated `gpu_id`. The bring-up only consumes `gpuvm_base/limit`.
pub const ProcessDeviceApertures = extern struct {
    lds_base: u64,
    lds_limit: u64,
    scratch_base: u64,
    scratch_limit: u64,
    gpuvm_base: u64,
    gpuvm_limit: u64,
    gpu_id: u32,
    _pad: u32,
};

/// `AMDKFD_IOC_GET_PROCESS_APERTURES_NEW` ioctl args — caller supplies a
/// pointer to an array of `ProcessDeviceApertures` plus its length; the kernel
/// fills the array and updates `num_of_nodes` with the actual count.
pub const GetProcessAperturesNewArgs = extern struct {
    kfd_process_device_apertures_ptr: u64,
    num_of_nodes: u32,
    _pad: u32,
};

/// `AMDKFD_IOC_ALLOC_MEMORY_OF_GPU` ioctl args. The caller chooses the GPU VA
/// (`va_addr`) and the kernel returns a BO `handle` plus an `mmap_offset` for
/// the DRM render node so the BO can be CPU-mapped.
pub const AllocMemoryOfGpuArgs = extern struct {
    va_addr: u64,
    size: u64,
    handle: u64,
    mmap_offset: u64,
    gpu_id: u32,
    flags: u32,
};

/// `AMDKFD_IOC_FREE_MEMORY_OF_GPU` ioctl args — release the BO referenced by
/// `handle` (must already be unmapped from every GPU it was mapped to).
pub const FreeMemoryOfGpuArgs = extern struct {
    handle: u64,
};

/// `AMDKFD_IOC_MAP_MEMORY_TO_GPU` / `UNMAP_MEMORY_FROM_GPU` ioctl args. Same
/// layout for both directions; the kernel updates `n_success` with the number
/// of devices the BO was successfully (un)mapped on.
pub const MapMemoryToGpuArgs = extern struct {
    handle: u64,
    device_ids_array_ptr: u64,
    n_devices: u32,
    n_success: u32,
};

/// `AMDKFD_IOC_CREATE_QUEUE` ioctl args — describes the PM4 compute queue the
/// kernel should map onto the MEC. Every BO address (ring/wptr/rptr/EOP/CWSR)
/// must already be allocated and mapped to the same GPU before the ioctl.
pub const CreateQueueArgs = extern struct {
    ring_base_address: u64,
    write_pointer_address: u64,
    read_pointer_address: u64,
    doorbell_offset: u64,
    ring_size: u32,
    gpu_id: u32,
    queue_type: u32,
    queue_percentage: u32,
    queue_priority: u32,
    queue_id: u32,
    eop_buffer_address: u64,
    eop_buffer_size: u64,
    ctx_save_restore_address: u64,
    ctx_save_restore_size: u32,
    ctl_stack_size: u32,
};

/// `AMDKFD_IOC_DESTROY_QUEUE` ioctl args — release the queue identified by
/// `queue_id` (the value the kernel returned from `CREATE_QUEUE`).
pub const DestroyQueueArgs = extern struct {
    queue_id: u32,
    _pad: u32,
};

const ioc_get_version = linux.IOCTL.IOR(kfd_ioctl_base, kfd_ioc_get_version_nr, GetVersionArgs);
const ioc_acquire_vm = linux.IOCTL.IOW(kfd_ioctl_base, kfd_ioc_acquire_vm_nr, AcquireVmArgs);
const ioc_get_process_apertures_new = linux.IOCTL.IOWR(kfd_ioctl_base, kfd_ioc_get_process_apertures_new_nr, GetProcessAperturesNewArgs);
const ioc_alloc_memory_of_gpu = linux.IOCTL.IOWR(kfd_ioctl_base, kfd_ioc_alloc_memory_of_gpu_nr, AllocMemoryOfGpuArgs);
const ioc_free_memory_of_gpu = linux.IOCTL.IOW(kfd_ioctl_base, kfd_ioc_free_memory_of_gpu_nr, FreeMemoryOfGpuArgs);
const ioc_map_memory_to_gpu = linux.IOCTL.IOWR(kfd_ioctl_base, kfd_ioc_map_memory_to_gpu_nr, MapMemoryToGpuArgs);
const ioc_unmap_memory_from_gpu = linux.IOCTL.IOWR(kfd_ioctl_base, kfd_ioc_unmap_memory_from_gpu_nr, MapMemoryToGpuArgs);
const ioc_create_queue = linux.IOCTL.IOWR(kfd_ioctl_base, kfd_ioc_create_queue_nr, CreateQueueArgs);
const ioc_destroy_queue = linux.IOCTL.IOWR(kfd_ioctl_base, kfd_ioc_destroy_queue_nr, DestroyQueueArgs);

/// One GPU topology node parsed from `topology_nodes_dir`. Carries the values
/// the queue bring-up needs to validate `CREATE_QUEUE`: `gpu_id`, the GFX IP
/// target version, CU/SIMD counts that drive CWSR sizing, and the canonical
/// `cwsr_size` / `ctl_stack_size` advertised by the kernel.
pub const TopologyNode = struct {
    node_index: u32,
    gpu_id: u32,
    gfx_target_version: u32 = 0,
    simd_count: u32 = 0,
    simd_per_cu: u32 = 0,
    num_xcc: u32 = 1,
    cwsr_size: u32 = 0,
    ctl_stack_size: u32 = 0,
    drm_render_minor: u32 = 0,
};

/// Outcome categories for `bringUpPath`. Every non-`ok` value identifies the
/// exact bring-up step that failed (open, version check, aperture lookup, VA
/// reservation, alloc, map, mmap, CPU round-trip, unmap, free).
pub const BringUpStatus = enum {
    ok,
    unsupported_os,
    kfd_open_failed,
    render_node_open_failed,
    get_version_failed,
    kfd_version_too_old,
    topology_unreadable,
    topology_node_missing,
    acquire_vm_failed,
    apertures_failed,
    aperture_missing,
    va_reservation_failed,
    va_outside_aperture,
    alloc_failed,
    map_failed,
    map_incomplete,
    mmap_failed,
    roundtrip_mismatch,
    unmap_failed,
    free_failed,
};

/// Full bring-up report produced by `bringUpPath`. Captures the status plus
/// every observable value the smoke run collected (KFD ABI version, topology
/// info, the GPUVM window, the reserved VA, and the errno of the last failing
/// ioctl when applicable) so callers can render diagnostics without rerunning.
pub const BringUpResult = struct {
    status: BringUpStatus,
    render_node: []const u8 = default_render_node,
    kfd_version_major: u32 = 0,
    kfd_version_minor: u32 = 0,
    gpu_id: u32 = 0,
    gfx_target_version: u32 = 0,
    simd_count: u32 = 0,
    cwsr_size: u32 = 0,
    ctl_stack_size: u32 = 0,
    gpuvm_base: u64 = 0,
    gpuvm_limit: u64 = 0,
    scratch_va: u64 = 0,
    errno: ?linux.E = null,

    /// True when every bring-up step succeeded (`status == .ok`).
    pub fn ok(self: BringUpResult) bool {
        return self.status == .ok;
    }
};

var last_ioctl_errno: ?linux.E = null;

/// Returns the errno of the most recent failing ioctl, or `null` if the last
/// ioctl succeeded. Cleared at the start of each `ioctlChecked` call.
pub fn lastErrno() ?linux.E {
    return last_ioctl_errno;
}

const IoctlError = error{IoctlFailed};

fn ioctlChecked(fd: std.posix.fd_t, request: u32, arg: usize) IoctlError!void {
    last_ioctl_errno = null;
    const rc = linux.ioctl(fd, request, arg);
    const err = linux.errno(rc);
    if (err != .SUCCESS) {
        last_ioctl_errno = err;
        return error.IoctlFailed;
    }
}

/// Render minor encoded in a `/dev/dri/renderD<minor>` path, e.g. 128.
pub fn renderMinorOf(render_node: []const u8) ?u32 {
    const marker = "renderD";
    const idx = std.mem.lastIndexOf(u8, render_node, marker) orelse return null;
    const tail = render_node[idx + marker.len ..];
    var end: usize = 0;
    while (end < tail.len and std.ascii.isDigit(tail[end])) : (end += 1) {}
    if (end == 0) return null;
    return std.fmt.parseInt(u32, tail[0..end], 10) catch null;
}

fn readSysU32(io: std.Io, dir: std.Io.Dir, sub_path: []const u8) ?u32 {
    var buf: [64]u8 = undefined;
    const file = dir.openFile(io, sub_path, .{}) catch return null;
    defer file.close(io);
    const n = file.readStreaming(io, &.{&buf}) catch return null;
    const trimmed = std.mem.trim(u8, buf[0..n], " \t\r\n");
    return std.fmt.parseInt(u32, trimmed, 10) catch null;
}

fn readPropertyU32(properties: []const u8, key: []const u8) ?u32 {
    var it = std.mem.tokenizeAny(u8, properties, "\n");
    while (it.next()) |line| {
        var fields = std.mem.tokenizeAny(u8, line, " \t");
        const name = fields.next() orelse continue;
        if (!std.mem.eql(u8, name, key)) continue;
        const value = fields.next() orelse return null;
        return std.fmt.parseInt(u32, value, 10) catch null;
    }
    return null;
}

/// Scan the KFD topology for the GPU node backing `render_minor`.
pub fn findTopologyNode(io: std.Io, render_minor: u32) ?TopologyNode {
    if (builtin.os.tag != .linux) return null;
    var nodes_dir = std.Io.Dir.openDirAbsolute(io, topology_nodes_dir, .{ .iterate = true }) catch return null;
    defer nodes_dir.close(io);
    var it = nodes_dir.iterate();
    while (it.next(io) catch null) |entry| {
        const node_index = std.fmt.parseInt(u32, entry.name, 10) catch continue;
        var node_dir = nodes_dir.openDir(io, entry.name, .{}) catch continue;
        defer node_dir.close(io);
        const gpu_id = readSysU32(io, node_dir, "gpu_id") orelse 0;
        if (gpu_id == 0) continue; // CPU-only topology node.
        var prop_buf: [4096]u8 = undefined;
        const prop_file = node_dir.openFile(io, "properties", .{}) catch continue;
        defer prop_file.close(io);
        const prop_len = prop_file.readStreaming(io, &.{&prop_buf}) catch continue;
        const properties = prop_buf[0..prop_len];
        const drm_render_minor = readPropertyU32(properties, "drm_render_minor") orelse continue;
        if (drm_render_minor != render_minor) continue;
        return .{
            .node_index = node_index,
            .gpu_id = gpu_id,
            .gfx_target_version = readPropertyU32(properties, "gfx_target_version") orelse 0,
            .simd_count = readPropertyU32(properties, "simd_count") orelse 0,
            .simd_per_cu = readPropertyU32(properties, "simd_per_cu") orelse 0,
            .num_xcc = readPropertyU32(properties, "num_xcc") orelse 1,
            .cwsr_size = readPropertyU32(properties, "cwsr_size") orelse 0,
            .ctl_stack_size = readPropertyU32(properties, "ctl_stack_size") orelse 0,
            .drm_render_minor = drm_render_minor,
        };
    }
    return null;
}

/// Cheap reachability check used by `engine.autoTier()` — no ioctls, no GPU VM.
pub fn reachable(io: std.Io) bool {
    if (builtin.os.tag != .linux) return false;
    const kfd = std.Io.Dir.openFileAbsolute(io, kfd_device_node, .{ .mode = .read_write }) catch return false;
    kfd.close(io);
    const minor = renderMinorOf(default_render_node) orelse return false;
    const node = findTopologyNode(io, minor) orelse return false;
    return node.gpu_id != 0;
}

/// Run `bringUpPath` against `default_render_node` ("/dev/dri/renderD128").
/// @returns Status + diagnostics for the GPUVM round-trip.
pub fn bringUpDefault(io: std.Io) BringUpResult {
    return bringUpPath(io, default_render_node);
}

/// End-to-end KFD bring-up: open `/dev/kfd` and the supplied render node,
/// check the ABI version, acquire the GPUVM, look up the matching topology
/// aperture, reserve a 64 KiB VA window inside it, allocate + mmap +
/// MAP_MEMORY_TO_GPU a 4 KiB GTT scratch BO, write/read a magic value to
/// verify the round-trip, then unmap and free everything. Every failure point
/// is captured in the returned `BringUpResult` (status + last errno) so a
/// non-Linux caller or a partial-permissions environment can still get a
/// useful diagnostic without panicking.
/// @param render_node Path to the DRM render node (e.g. `/dev/dri/renderD128`).
/// @returns A populated `BringUpResult`; `.ok()` is true only on a clean
/// round-trip.
/// @note Off Linux this short-circuits to `unsupported_os` and performs no IO.
pub fn bringUpPath(io: std.Io, render_node: []const u8) BringUpResult {
    if (builtin.os.tag != .linux) {
        return .{ .status = .unsupported_os, .render_node = render_node };
    }

    var result: BringUpResult = .{ .status = .ok, .render_node = render_node };

    const minor = renderMinorOf(render_node) orelse return fail(result, .topology_unreadable);
    const topo = findTopologyNode(io, minor) orelse return fail(result, .topology_node_missing);
    result.gpu_id = topo.gpu_id;
    result.gfx_target_version = topo.gfx_target_version;
    result.simd_count = topo.simd_count;
    result.cwsr_size = topo.cwsr_size;
    result.ctl_stack_size = topo.ctl_stack_size;

    const kfd = std.Io.Dir.openFileAbsolute(io, kfd_device_node, .{ .mode = .read_write }) catch return fail(result, .kfd_open_failed);
    defer kfd.close(io);
    const drm = std.Io.Dir.openFileAbsolute(io, render_node, .{ .mode = .read_write }) catch return fail(result, .render_node_open_failed);
    defer drm.close(io);

    var version: GetVersionArgs = std.mem.zeroes(GetVersionArgs);
    ioctlChecked(kfd.handle, ioc_get_version, @intFromPtr(&version)) catch return failErrno(result, .get_version_failed);
    result.kfd_version_major = version.major_version;
    result.kfd_version_minor = version.minor_version;
    if (version.major_version < min_kfd_major) return fail(result, .kfd_version_too_old);

    var acquire: AcquireVmArgs = .{ .drm_fd = @intCast(drm.handle), .gpu_id = topo.gpu_id };
    ioctlChecked(kfd.handle, ioc_acquire_vm, @intFromPtr(&acquire)) catch return failErrno(result, .acquire_vm_failed);

    var apertures: [16]ProcessDeviceApertures = std.mem.zeroes([16]ProcessDeviceApertures);
    var apertures_args: GetProcessAperturesNewArgs = .{
        .kfd_process_device_apertures_ptr = @intFromPtr(&apertures),
        .num_of_nodes = apertures.len,
        ._pad = 0,
    };
    ioctlChecked(kfd.handle, ioc_get_process_apertures_new, @intFromPtr(&apertures_args)) catch return failErrno(result, .apertures_failed);
    const aperture = blk: {
        const node_count: u32 = @min(apertures_args.num_of_nodes, @as(u32, apertures.len));
        var i: u32 = 0;
        while (i < node_count) : (i += 1) {
            if (apertures[i].gpu_id == topo.gpu_id) break :blk apertures[i];
        }
        return fail(result, .aperture_missing);
    };
    result.gpuvm_base = aperture.gpuvm_base;
    result.gpuvm_limit = aperture.gpuvm_limit;

    // Reserve a small VA window. The kernel-picked address is a valid CPU VA;
    // on gfx9+ the GPUVM aperture covers the canonical low half so the same
    // address doubles as the GPU VA (the ROCt/tinygrad pattern).
    const window_size: usize = 0x10000;
    const window = std.posix.mmap(
        null,
        window_size,
        .{},
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
        -1,
        0,
    ) catch return fail(result, .va_reservation_failed);
    var window_active = true;
    defer if (window_active) std.posix.munmap(window);

    const window_addr: u64 = @intFromPtr(window.ptr);
    result.scratch_va = window_addr;
    if (window_addr < aperture.gpuvm_base or window_addr + @as(u64, window_size) > aperture.gpuvm_limit) {
        return fail(result, .va_outside_aperture);
    }

    const scratch_size: usize = 0x1000;
    var alloc_args: AllocMemoryOfGpuArgs = .{
        .va_addr = window_addr,
        .size = @as(u64, scratch_size),
        .handle = 0,
        .mmap_offset = 0,
        .gpu_id = topo.gpu_id,
        .flags = ALLOC_MEM_FLAGS_GTT | ALLOC_MEM_FLAGS_WRITABLE,
    };
    ioctlChecked(kfd.handle, ioc_alloc_memory_of_gpu, @intFromPtr(&alloc_args)) catch return failErrno(result, .alloc_failed);
    const scratch_handle = alloc_args.handle;
    var freed = false;
    defer if (!freed) {
        var free_args: FreeMemoryOfGpuArgs = .{ .handle = scratch_handle };
        ioctlChecked(kfd.handle, ioc_free_memory_of_gpu, @intFromPtr(&free_args)) catch {};
    };

    // FIXED mmap over the reservation: the BO's GPU VA and CPU VA are the same
    // address. Tearing down the reservation slice at function exit covers it.
    const scratch_map = std.posix.mmap(
        window.ptr,
        scratch_size,
        std.posix.PROT{ .READ = true, .WRITE = true },
        .{ .TYPE = .SHARED, .FIXED = true },
        drm.handle,
        alloc_args.mmap_offset,
    ) catch return fail(result, .mmap_failed);

    var device_ids = [_]u32{topo.gpu_id};
    var map_args: MapMemoryToGpuArgs = .{
        .handle = scratch_handle,
        .device_ids_array_ptr = @intFromPtr(&device_ids),
        .n_devices = 1,
        .n_success = 0,
    };
    ioctlChecked(kfd.handle, ioc_map_memory_to_gpu, @intFromPtr(&map_args)) catch return failErrno(result, .map_failed);
    if (map_args.n_success != 1) return fail(result, .map_incomplete);
    var unmapped = false;
    defer if (!unmapped) {
        var unmap_args: MapMemoryToGpuArgs = .{
            .handle = scratch_handle,
            .device_ids_array_ptr = @intFromPtr(&device_ids),
            .n_devices = 1,
            .n_success = 0,
        };
        ioctlChecked(kfd.handle, ioc_unmap_memory_from_gpu, @intFromPtr(&unmap_args)) catch {};
    };

    // CPU round-trip through the mapped scratch BO.
    const magic: u64 = 0x5A494E435F525431; // "ZINC_RT1"
    const cell: *volatile u64 = @ptrCast(@alignCast(scratch_map.ptr));
    cell.* = magic;
    if (cell.* != magic) return fail(result, .roundtrip_mismatch);

    var unmap_args: MapMemoryToGpuArgs = .{
        .handle = scratch_handle,
        .device_ids_array_ptr = @intFromPtr(&device_ids),
        .n_devices = 1,
        .n_success = 0,
    };
    ioctlChecked(kfd.handle, ioc_unmap_memory_from_gpu, @intFromPtr(&unmap_args)) catch return failErrno(result, .unmap_failed);
    unmapped = true;

    var free_args: FreeMemoryOfGpuArgs = .{ .handle = scratch_handle };
    ioctlChecked(kfd.handle, ioc_free_memory_of_gpu, @intFromPtr(&free_args)) catch return failErrno(result, .free_failed);
    freed = true;

    std.posix.munmap(window);
    window_active = false;

    result.status = .ok;
    return result;
}

fn fail(base: BringUpResult, status: BringUpStatus) BringUpResult {
    var result = base;
    result.status = status;
    return result;
}

fn failErrno(base: BringUpResult, status: BringUpStatus) BringUpResult {
    var result = fail(base, status);
    result.errno = last_ioctl_errno;
    return result;
}

// ===========================================================================
// T1 PM4 compute-queue create/destroy bring-up
//
// `AMDKFD_IOC_CREATE_QUEUE` for `KFD_IOC_QUEUE_TYPE_COMPUTE` requires the caller
// to hand the kernel ring / write-pointer / read-pointer / EOP / CWSR buffer
// objects; `kfd_queue_acquire_buffers` validates each one's size against the
// topology node's properties before the queue is mapped onto the MEC. This
// smoke path constructs those BOs with the verified sizing rules, creates the
// queue, stages a couple of PM4 NOP packets into the ring (not yet submitted —
// the doorbell ring + retired fence is the next step), then destroys the queue
// and tears everything down. Decode still runs on T-CPU after admission.
// ===========================================================================

/// Outcome categories for the PM4 compute-queue smoke path. Each non-`ok`
/// value identifies a specific failure step (BO allocation, map,
/// `CREATE_QUEUE`, `DESTROY_QUEUE`) so callers can render bring-up reports.
pub const ComputeQueueSmokeStatus = enum {
    ok,
    unsupported_os,
    topology_node_missing,
    kfd_open_failed,
    render_node_open_failed,
    get_version_failed,
    kfd_version_too_old,
    acquire_vm_failed,
    apertures_failed,
    aperture_missing,
    va_reservation_failed,
    bo_alloc_failed,
    bo_map_failed,
    create_queue_failed,
    destroy_queue_failed,
};

/// Full report for the PM4 compute-queue smoke run. Captures the status plus
/// every observable value the run produced (BO VAs, queue id, doorbell offset,
/// initial wptr/rptr, number of PM4 NOP dwords staged, and the errno of the
/// last failing ioctl) so a higher-level CLI can render bring-up output.
pub const ComputeQueueSmokeResult = struct {
    status: ComputeQueueSmokeStatus,
    render_node: []const u8 = default_render_node,
    gpu_id: u32 = 0,
    gfx_target_version: u32 = 0,
    queue_id: u32 = 0,
    doorbell_offset: u64 = 0,
    ring_va: u64 = 0,
    ring_size: u32 = 0,
    wptr_va: u64 = 0,
    rptr_va: u64 = 0,
    eop_va: u64 = 0,
    eop_size: u64 = 0,
    cwsr_va: u64 = 0,
    cwsr_bo_size: u64 = 0,
    ctx_save_restore_size: u32 = 0,
    ctl_stack_size: u32 = 0,
    create_queue_arg_size: u32 = @sizeOf(CreateQueueArgs),
    wptr_value: u64 = 0,
    rptr_value: u64 = 0,
    nop_dwords_staged: u32 = 0,
    errno: ?linux.E = null,
    destroy_errno: ?linux.E = null,

    /// True when the queue was created, NOPs staged, and the queue destroyed
    /// without any ioctl failure.
    pub fn ok(self: ComputeQueueSmokeResult) bool {
        return self.status == .ok;
    }
};

const page_size: usize = 4096;
const default_ring_size: usize = 0x10000;

/// Round `value` up to the next multiple of `alignment`. Returns `value`
/// unchanged when `alignment` is zero or `value` is already aligned.
/// @param value Number to be rounded up.
/// @param alignment Power-of-two (or any non-zero) boundary to align to.
/// @returns Smallest multiple of `alignment` that is `>= value`.
pub fn alignUp(value: u64, alignment: u64) u64 {
    if (alignment == 0) return value;
    const rem = value % alignment;
    return if (rem == 0) value else value + (alignment - rem);
}

/// CWSR buffer-object size the kernel's `kfd_queue_acquire_buffers` validates
/// against: `align_up((cwsr_size + debug_memory_size) * num_xcc, PAGE)`, where
/// for gfx ≥ 10.1.x `debug_memory_size = align_up((simd_count / simd_per_cu /
/// num_xcc) * 32 * 32, 64)`. Verified on the R9700 (gfx1201): cwsr_size
/// 0x1d47000, debug 0x10000, BO 0x1d57000.
pub fn computeCwsrBoSize(
    cwsr_size: u32,
    simd_count: u32,
    simd_per_cu_in: u32,
    num_xcc_in: u32,
    gfx_target_version: u32,
) u64 {
    const num_xcc: u64 = if (num_xcc_in == 0) 1 else num_xcc_in;
    const simd_per_cu: u64 = if (simd_per_cu_in != 0)
        simd_per_cu_in
    else if (gfx_target_version >= 100000 and gfx_target_version < 130000)
        2 // RDNA: 2 SIMD32 per CU
    else
        4; // GCN/CDNA fallback
    const cu_num: u64 = if (simd_count == 0 or simd_per_cu == 0)
        0
    else
        @as(u64, simd_count) / simd_per_cu / num_xcc;
    const debug_memory_size: u64 = if (gfx_target_version >= 100100)
        alignUp(cu_num * 32 * 32, 64)
    else
        0;
    return alignUp((@as(u64, cwsr_size) + debug_memory_size) * num_xcc, page_size);
}

const BoAllocError = error{
    VaReservationFailed,
    VaOutsideAperture,
    BoAllocFailed,
    BoMmapFailed,
    BoMapFailed,
    BoMapIncomplete,
};

const AllocedBo = struct {
    handle: u64,
    va: u64,
    reservation: []align(std.heap.page_size_min) u8,
    cpu: ?[]align(std.heap.page_size_min) u8,
};

/// Reserve a fresh VA window inside the GPUVM aperture, `ALLOC_MEMORY_OF_GPU`
/// (GTT, writable) at that VA, optionally `mmap` it `MAP_FIXED` over the
/// reservation for CPU access, then `MAP_MEMORY_TO_GPU`. On error every partial
/// step is undone before returning. On success the caller owns `reservation`
/// (munmap when done — that also drops the FIXED CPU mapping) and `handle`
/// (`FREE_MEMORY_OF_GPU` when done).
fn allocQueueBo(
    kfd: std.Io.File,
    drm: std.Io.File,
    gpu_id: u32,
    aperture: ProcessDeviceApertures,
    size: usize,
    want_cpu: bool,
) BoAllocError!AllocedBo {
    const reservation = std.posix.mmap(
        null,
        size,
        .{},
        .{ .TYPE = .PRIVATE, .ANONYMOUS = true },
        -1,
        0,
    ) catch return error.VaReservationFailed;
    errdefer std.posix.munmap(reservation);

    const va: u64 = @intFromPtr(reservation.ptr);
    if (va < aperture.gpuvm_base or va + @as(u64, size) > aperture.gpuvm_limit) {
        return error.VaOutsideAperture;
    }

    var alloc_args: AllocMemoryOfGpuArgs = .{
        .va_addr = va,
        .size = @as(u64, size),
        .handle = 0,
        .mmap_offset = 0,
        .gpu_id = gpu_id,
        .flags = ALLOC_MEM_FLAGS_GTT | ALLOC_MEM_FLAGS_WRITABLE,
    };
    ioctlChecked(kfd.handle, ioc_alloc_memory_of_gpu, @intFromPtr(&alloc_args)) catch return error.BoAllocFailed;
    errdefer {
        var free_args: FreeMemoryOfGpuArgs = .{ .handle = alloc_args.handle };
        ioctlChecked(kfd.handle, ioc_free_memory_of_gpu, @intFromPtr(&free_args)) catch {};
    }

    var cpu_map: ?[]align(std.heap.page_size_min) u8 = null;
    if (want_cpu) {
        const mapped = std.posix.mmap(
            reservation.ptr,
            size,
            std.posix.PROT{ .READ = true, .WRITE = true },
            .{ .TYPE = .SHARED, .FIXED = true },
            drm.handle,
            alloc_args.mmap_offset,
        ) catch return error.BoMmapFailed;
        @memset(mapped, 0);
        cpu_map = mapped;
    }

    var device_ids = [_]u32{gpu_id};
    var map_args: MapMemoryToGpuArgs = .{
        .handle = alloc_args.handle,
        .device_ids_array_ptr = @intFromPtr(&device_ids),
        .n_devices = 1,
        .n_success = 0,
    };
    ioctlChecked(kfd.handle, ioc_map_memory_to_gpu, @intFromPtr(&map_args)) catch return error.BoMapFailed;
    if (map_args.n_success != 1) return error.BoMapIncomplete;

    return .{ .handle = alloc_args.handle, .va = va, .reservation = reservation, .cpu = cpu_map };
}

fn cqFail(base: ComputeQueueSmokeResult, status: ComputeQueueSmokeStatus) ComputeQueueSmokeResult {
    var result = base;
    result.status = status;
    return result;
}

fn cqFailErrno(base: ComputeQueueSmokeResult, status: ComputeQueueSmokeStatus) ComputeQueueSmokeResult {
    var result = cqFail(base, status);
    result.errno = last_ioctl_errno;
    return result;
}

fn cqFromAllocErr(base: ComputeQueueSmokeResult, err: BoAllocError) ComputeQueueSmokeResult {
    return switch (err) {
        error.VaReservationFailed, error.VaOutsideAperture => cqFail(base, .va_reservation_failed),
        error.BoAllocFailed => cqFailErrno(base, .bo_alloc_failed),
        error.BoMmapFailed, error.BoMapFailed, error.BoMapIncomplete => cqFailErrno(base, .bo_map_failed),
    };
}

/// Run `createComputeQueueSmokePath` against `default_render_node`.
/// @returns The compute-queue smoke result; `.ok()` is true on a clean run.
pub fn createComputeQueueSmokeDefault(io: std.Io) ComputeQueueSmokeResult {
    return createComputeQueueSmokePath(io, default_render_node);
}

/// Stand up a PM4 compute queue end-to-end on the given render node. Allocates
/// the ring / wptr / rptr / EOP / CWSR buffer objects with the sizing rules
/// `kfd_queue_acquire_buffers` validates, calls `AMDKFD_IOC_CREATE_QUEUE`,
/// stages a couple of PM4 NOP packets into the ring (no doorbell yet), then
/// destroys the queue and tears everything down. Every BO and ioctl failure
/// is captured in the returned report so the bring-up smoke test can render
/// the exact step that failed.
/// @param render_node DRM render node backing the target GPU.
/// @returns The full smoke report; `.ok()` is true on a clean create/destroy.
/// @note Off Linux this short-circuits to `unsupported_os` without IO.
pub fn createComputeQueueSmokePath(io: std.Io, render_node: []const u8) ComputeQueueSmokeResult {
    if (builtin.os.tag != .linux) {
        return .{ .status = .unsupported_os, .render_node = render_node };
    }

    var result: ComputeQueueSmokeResult = .{ .status = .ok, .render_node = render_node };

    const minor = renderMinorOf(render_node) orelse return cqFail(result, .topology_node_missing);
    const topo = findTopologyNode(io, minor) orelse return cqFail(result, .topology_node_missing);
    result.gpu_id = topo.gpu_id;
    result.gfx_target_version = topo.gfx_target_version;
    result.ctx_save_restore_size = topo.cwsr_size;
    result.ctl_stack_size = topo.ctl_stack_size;

    const kfd = std.Io.Dir.openFileAbsolute(io, kfd_device_node, .{ .mode = .read_write }) catch return cqFail(result, .kfd_open_failed);
    defer kfd.close(io);
    const drm = std.Io.Dir.openFileAbsolute(io, render_node, .{ .mode = .read_write }) catch return cqFail(result, .render_node_open_failed);
    defer drm.close(io);

    var version: GetVersionArgs = std.mem.zeroes(GetVersionArgs);
    ioctlChecked(kfd.handle, ioc_get_version, @intFromPtr(&version)) catch return cqFailErrno(result, .get_version_failed);
    if (version.major_version < min_kfd_major) return cqFail(result, .kfd_version_too_old);

    var acquire: AcquireVmArgs = .{ .drm_fd = @intCast(drm.handle), .gpu_id = topo.gpu_id };
    ioctlChecked(kfd.handle, ioc_acquire_vm, @intFromPtr(&acquire)) catch return cqFailErrno(result, .acquire_vm_failed);

    var apertures: [16]ProcessDeviceApertures = std.mem.zeroes([16]ProcessDeviceApertures);
    var apertures_args: GetProcessAperturesNewArgs = .{
        .kfd_process_device_apertures_ptr = @intFromPtr(&apertures),
        .num_of_nodes = apertures.len,
        ._pad = 0,
    };
    ioctlChecked(kfd.handle, ioc_get_process_apertures_new, @intFromPtr(&apertures_args)) catch return cqFailErrno(result, .apertures_failed);
    const aperture = blk: {
        const node_count: u32 = @min(apertures_args.num_of_nodes, @as(u32, apertures.len));
        var i: u32 = 0;
        while (i < node_count) : (i += 1) {
            if (apertures[i].gpu_id == topo.gpu_id) break :blk apertures[i];
        }
        return cqFail(result, .aperture_missing);
    };

    const ring_size: usize = default_ring_size;
    const eop_size: usize = page_size; // node_props.eop_buffer_size == 4096 for gfx ≥ 8.0.0
    const cwsr_bo_size: usize = @intCast(computeCwsrBoSize(
        topo.cwsr_size,
        topo.simd_count,
        topo.simd_per_cu,
        topo.num_xcc,
        topo.gfx_target_version,
    ));
    result.ring_size = @intCast(ring_size);
    result.eop_size = eop_size;
    result.cwsr_bo_size = cwsr_bo_size;

    // BOs to release in reverse order at scope exit (after DESTROY_QUEUE).
    var bo_handles: [5]u64 = .{ 0, 0, 0, 0, 0 };
    var bo_reservations: [5][]align(std.heap.page_size_min) u8 = undefined;
    var bo_count: usize = 0;
    defer {
        var i: usize = bo_count;
        while (i > 0) {
            i -= 1;
            var device_ids = [_]u32{topo.gpu_id};
            var unmap_args: MapMemoryToGpuArgs = .{
                .handle = bo_handles[i],
                .device_ids_array_ptr = @intFromPtr(&device_ids),
                .n_devices = 1,
                .n_success = 0,
            };
            ioctlChecked(kfd.handle, ioc_unmap_memory_from_gpu, @intFromPtr(&unmap_args)) catch {};
            var free_args: FreeMemoryOfGpuArgs = .{ .handle = bo_handles[i] };
            ioctlChecked(kfd.handle, ioc_free_memory_of_gpu, @intFromPtr(&free_args)) catch {};
            std.posix.munmap(bo_reservations[i]);
        }
    }

    const ring_bo = allocQueueBo(kfd, drm, topo.gpu_id, aperture, ring_size, true) catch |e| return cqFromAllocErr(result, e);
    bo_handles[bo_count] = ring_bo.handle;
    bo_reservations[bo_count] = ring_bo.reservation;
    bo_count += 1;

    const wptr_bo = allocQueueBo(kfd, drm, topo.gpu_id, aperture, page_size, true) catch |e| return cqFromAllocErr(result, e);
    bo_handles[bo_count] = wptr_bo.handle;
    bo_reservations[bo_count] = wptr_bo.reservation;
    bo_count += 1;

    const rptr_bo = allocQueueBo(kfd, drm, topo.gpu_id, aperture, page_size, true) catch |e| return cqFromAllocErr(result, e);
    bo_handles[bo_count] = rptr_bo.handle;
    bo_reservations[bo_count] = rptr_bo.reservation;
    bo_count += 1;

    const eop_bo = allocQueueBo(kfd, drm, topo.gpu_id, aperture, eop_size, false) catch |e| return cqFromAllocErr(result, e);
    bo_handles[bo_count] = eop_bo.handle;
    bo_reservations[bo_count] = eop_bo.reservation;
    bo_count += 1;

    const cwsr_bo = allocQueueBo(kfd, drm, topo.gpu_id, aperture, cwsr_bo_size, false) catch |e| return cqFromAllocErr(result, e);
    bo_handles[bo_count] = cwsr_bo.handle;
    bo_reservations[bo_count] = cwsr_bo.reservation;
    bo_count += 1;

    result.ring_va = ring_bo.va;
    result.wptr_va = wptr_bo.va;
    result.rptr_va = rptr_bo.va;
    result.eop_va = eop_bo.va;
    result.cwsr_va = cwsr_bo.va;

    var create_args: CreateQueueArgs = std.mem.zeroes(CreateQueueArgs);
    create_args.ring_base_address = ring_bo.va;
    create_args.write_pointer_address = wptr_bo.va;
    create_args.read_pointer_address = rptr_bo.va;
    create_args.doorbell_offset = 0;
    create_args.ring_size = @intCast(ring_size);
    create_args.gpu_id = topo.gpu_id;
    create_args.queue_type = QUEUE_TYPE_COMPUTE;
    create_args.queue_percentage = 100;
    create_args.queue_priority = 7;
    create_args.queue_id = 0;
    create_args.eop_buffer_address = eop_bo.va;
    create_args.eop_buffer_size = @as(u64, eop_size);
    create_args.ctx_save_restore_address = cwsr_bo.va;
    create_args.ctx_save_restore_size = topo.cwsr_size;
    create_args.ctl_stack_size = topo.ctl_stack_size;
    ioctlChecked(kfd.handle, ioc_create_queue, @intFromPtr(&create_args)) catch return cqFailErrno(result, .create_queue_failed);

    result.queue_id = create_args.queue_id;
    result.doorbell_offset = create_args.doorbell_offset;

    // Fresh PM4 ring: read/write pointers both start at 0.
    if (wptr_bo.cpu) |m| result.wptr_value = @as(*const volatile u64, @ptrCast(@alignCast(m.ptr))).*;
    if (rptr_bo.cpu) |m| result.rptr_value = @as(*const volatile u64, @ptrCast(@alignCast(m.ptr))).*;

    // Stage a couple of PM4 NOP packets into the ring (not submitted yet; the
    // doorbell ring + retired fence is the next bring-up step). This exercises
    // the PM4 packet builder against real GPU-mapped ring memory.
    if (ring_bo.cpu) |m| {
        const ring_words = @as([*]u32, @ptrCast(@alignCast(m.ptr)))[0 .. ring_size / @sizeOf(u32)];
        var builder = packet.PacketBuilder.init(ring_words);
        builder.writeNop(3) catch {};
        builder.writeNop(1) catch {};
        result.nop_dwords_staged = @intCast(builder.written().len);
    }

    var destroy_args: DestroyQueueArgs = .{ .queue_id = create_args.queue_id, ._pad = 0 };
    ioctlChecked(kfd.handle, ioc_destroy_queue, @intFromPtr(&destroy_args)) catch {
        var r = result;
        r.status = .destroy_queue_failed;
        r.destroy_errno = last_ioctl_errno;
        return r;
    };

    result.status = .ok;
    return result;
}

/// Render `gfx_target_version` (e.g. 120001) as a GFX target string ("gfx1201").
pub fn formatGfxTarget(buf: []u8, gfx_target_version: u32) []const u8 {
    if (gfx_target_version == 0) return "gfx?";
    const major = gfx_target_version / 10000;
    const minor = (gfx_target_version / 100) % 100;
    const step = gfx_target_version % 100;
    const rendered = std.fmt.bufPrint(buf, "gfx{d}{x}{x}", .{ major, minor, step }) catch return "gfx?";
    return rendered;
}

test "kfd ioctl arg layout is stable" {
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(GetVersionArgs));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(AcquireVmArgs));
    try std.testing.expectEqual(@as(usize, 56), @sizeOf(ProcessDeviceApertures));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(GetProcessAperturesNewArgs));
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(AllocMemoryOfGpuArgs));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(FreeMemoryOfGpuArgs));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(MapMemoryToGpuArgs));
    try std.testing.expectEqual(@as(usize, 88), @sizeOf(CreateQueueArgs));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(DestroyQueueArgs));
}

test "kfd ioctl request numbers match uapi" {
    // _IOR('K', 0x01, struct kfd_ioctl_get_version_args)         size 8
    try std.testing.expectEqual(@as(u32, 0x80084b01), ioc_get_version);
    // _IOW('K', 0x15, struct kfd_ioctl_acquire_vm_args)          size 8
    try std.testing.expectEqual(@as(u32, 0x40084b15), ioc_acquire_vm);
    // _IOWR('K', 0x14, struct kfd_ioctl_get_process_apertures_new_args) size 16
    try std.testing.expectEqual(@as(u32, 0xc0104b14), ioc_get_process_apertures_new);
    // _IOWR('K', 0x16, struct kfd_ioctl_alloc_memory_of_gpu_args) size 40
    try std.testing.expectEqual(@as(u32, 0xc0284b16), ioc_alloc_memory_of_gpu);
    // _IOWR('K', 0x18, struct kfd_ioctl_map_memory_to_gpu_args)   size 24
    try std.testing.expectEqual(@as(u32, 0xc0184b18), ioc_map_memory_to_gpu);
    // _IOWR('K', 0x02, struct kfd_ioctl_create_queue_args)        size 88
    try std.testing.expectEqual(@as(u32, 0xc0584b02), ioc_create_queue);
}

test "renderMinorOf parses render node paths" {
    try std.testing.expectEqual(@as(?u32, 128), renderMinorOf("/dev/dri/renderD128"));
    try std.testing.expectEqual(@as(?u32, 129), renderMinorOf("/dev/dri/renderD129"));
    try std.testing.expectEqual(@as(?u32, null), renderMinorOf("/dev/dri/card1"));
}

test "readPropertyU32 extracts KFD topology fields" {
    const props = "simd_count 128\ngfx_target_version 120001\ncwsr_size 30699520\nctl_stack_size 28672\n";
    try std.testing.expectEqual(@as(?u32, 128), readPropertyU32(props, "simd_count"));
    try std.testing.expectEqual(@as(?u32, 120001), readPropertyU32(props, "gfx_target_version"));
    try std.testing.expectEqual(@as(?u32, 30699520), readPropertyU32(props, "cwsr_size"));
    try std.testing.expectEqual(@as(?u32, null), readPropertyU32(props, "missing_key"));
}

test "formatGfxTarget renders RDNA targets" {
    var buf: [16]u8 = undefined;
    try std.testing.expectEqualStrings("gfx1201", formatGfxTarget(&buf, 120001));
    try std.testing.expectEqualStrings("gfx1036", formatGfxTarget(&buf, 100306));
    try std.testing.expectEqualStrings("gfx?", formatGfxTarget(&buf, 0));
}

test "bringUpPath reports unsupported_os off Linux" {
    if (builtin.os.tag == .linux) return error.SkipZigTest;
    const result = bringUpPath(default_render_node);
    try std.testing.expectEqual(BringUpStatus.unsupported_os, result.status);
    try std.testing.expect(!result.ok());
}

test "reachable is false off Linux" {
    if (builtin.os.tag == .linux) return error.SkipZigTest;
    try std.testing.expect(!reachable());
}

test "computeCwsrBoSize matches the verified R9700 (gfx1201) reference" {
    // simd_count 128, simd_per_cu 2, num_xcc 1 → cu_num 64; debug 0x10000.
    try std.testing.expectEqual(@as(u64, 0x1d57000), computeCwsrBoSize(0x1d47000, 128, 2, 1, 120001));
    // simd_per_cu missing → RDNA fallback of 2 gives the same answer.
    try std.testing.expectEqual(@as(u64, 0x1d57000), computeCwsrBoSize(0x1d47000, 128, 0, 1, 120001));
}

test "alignUp rounds up to the requested boundary" {
    try std.testing.expectEqual(@as(u64, 4096), alignUp(4096, 4096));
    try std.testing.expectEqual(@as(u64, 8192), alignUp(4097, 4096));
    try std.testing.expectEqual(@as(u64, 65536), alignUp(64 * 32 * 32, 64));
}

test "createComputeQueueSmokePath reports unsupported_os off Linux" {
    if (builtin.os.tag == .linux) return error.SkipZigTest;
    const result = createComputeQueueSmokePath(default_render_node);
    try std.testing.expectEqual(ComputeQueueSmokeStatus.unsupported_os, result.status);
    try std.testing.expect(!result.ok());
}
