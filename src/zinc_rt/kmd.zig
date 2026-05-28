//! Thin AMDGPU kernel-driver queries used by direct ZINC_RT tiers.
//!
//! This file intentionally starts with capability discovery only. T2 UMQ queue
//! creation needs the same UAPI definitions, but selection must first prove the
//! kernel exposes compute user queues instead of relying on kernel version alone.
//! @section Inference Runtime
const std = @import("std");
const builtin = @import("builtin");

const linux = std.os.linux;

/// Outcome of probing the AMDGPU render node for compute user-queue support.
/// Each variant maps to a specific failure mode when discovering whether the
/// kernel exposes the UMQ surface ZINC_RT tier 2 needs.
pub const QueryStatus = enum {
    available,
    unsupported_os,
    render_node_open_failed,
    hw_ip_info_failed,
    compute_userq_unsupported,
    compute_userq_slots_missing,
    fw_metadata_failed,
    invalid_fw_metadata,
};

/// Compute user-queue capability metadata reported by the kernel.
/// Captures the slot count and the EOP (end-of-pipe) scratch buffer sizing the
/// driver requires when creating a compute UMQ.
pub const ComputeUserqInfo = struct {
    userq_slots: u32,
    eop_size: u32,
    eop_alignment: u32,
};

/// Combined result returned by `queryComputeUserq`.
/// Carries the discovery status plus optional capability info and the errno
/// captured from the failing ioctl, when applicable.
pub const QueryResult = struct {
    status: QueryStatus,
    info: ?ComputeUserqInfo = null,
    errno: ?linux.E = null,
};

const drm_ioctl_base: u8 = 'd';
const drm_command_base: u8 = 0x40;

/// AMDGPU HW IP type selector for the compute pipe used by `AMDGPU_INFO` queries.
pub const AMDGPU_HW_IP_COMPUTE: u32 = 1;
/// `AMDGPU_INFO` sub-query that returns `DrmAmdgpuInfoHwIp` for a given HW IP.
pub const AMDGPU_INFO_HW_IP_INFO: u32 = 0x02;
/// `AMDGPU_INFO` sub-query that returns user-queue firmware area metadata (`DrmAmdgpuInfoUqMetadata`).
pub const AMDGPU_INFO_UQ_FW_AREAS: u32 = 0x24;
/// `DRM_AMDGPU_USERQ` op code that allocates a new user-mode queue.
pub const AMDGPU_USERQ_OP_CREATE: u32 = 1;
/// `DRM_AMDGPU_USERQ` op code that releases a previously created user-mode queue.
pub const AMDGPU_USERQ_OP_FREE: u32 = 2;

/// GEM domain flag requesting allocation in system GTT memory.
pub const AMDGPU_GEM_DOMAIN_GTT: u64 = 0x2;
/// GEM domain flag requesting allocation in device VRAM.
pub const AMDGPU_GEM_DOMAIN_VRAM: u64 = 0x4;
/// GEM domain flag requesting allocation in the MMIO doorbell aperture.
pub const AMDGPU_GEM_DOMAIN_DOORBELL: u64 = 0x40;

/// GEM creation flag asking the kernel to map GTT memory as CPU write-combined for fast streaming writes.
pub const AMDGPU_GEM_CREATE_CPU_GTT_USWC: u64 = 1 << 2;
/// GEM creation flag asking the kernel to zero-fill VRAM allocations before returning the BO.
pub const AMDGPU_GEM_CREATE_VRAM_CLEARED: u64 = 1 << 3;
/// GEM creation flag keeping the BO permanently mapped in the device VM so it never needs revalidation.
pub const AMDGPU_GEM_CREATE_VM_ALWAYS_VALID: u64 = 1 << 6;

/// `DRM_AMDGPU_GEM_VA` operation that binds a BO into the device virtual address space.
pub const AMDGPU_VA_OP_MAP: u32 = 1;
/// VA mapping flag granting GPU read access to the mapped range.
pub const AMDGPU_VM_PAGE_READABLE: u32 = 1 << 1;
/// VA mapping flag granting GPU write access to the mapped range.
pub const AMDGPU_VM_PAGE_WRITEABLE: u32 = 1 << 2;
/// VA mapping flag granting GPU shader-execute access to the mapped range.
pub const AMDGPU_VM_PAGE_EXECUTABLE: u32 = 1 << 3;
/// VA mapping flag selecting the default memory type (MTYPE) for the GPU page table entry.
pub const AMDGPU_VM_MTYPE_DEFAULT: u32 = 0 << 5;

const drm_amdgpu_gem_create_nr: u8 = 0x00;
const drm_amdgpu_gem_mmap_nr: u8 = 0x01;
const drm_amdgpu_info_nr: u8 = 0x05;
const drm_amdgpu_gem_va_nr: u8 = 0x08;
const drm_amdgpu_userq_nr: u8 = 0x16;

/// Input layout for the `DRM_IOCTL_AMDGPU_GEM_CREATE` ioctl.
/// Mirrors the kernel UAPI struct describing the requested buffer size, alignment, domain mask, and creation flags.
pub const DrmAmdgpuGemCreateIn = extern struct {
    bo_size: u64,
    alignment: u64,
    domains: u64,
    domain_flags: u64,
};

/// Output layout returned by `DRM_IOCTL_AMDGPU_GEM_CREATE`, holding the freshly allocated BO handle.
pub const DrmAmdgpuGemCreateOut = extern struct {
    handle: u32,
    _pad: u32,
};

/// Tagged union packing the in/out forms of the GEM-create ioctl into the same buffer the kernel reads and writes.
pub const DrmAmdgpuGemCreate = extern union {
    in: DrmAmdgpuGemCreateIn,
    out: DrmAmdgpuGemCreateOut,
};

/// Input layout for `DRM_IOCTL_AMDGPU_GEM_MMAP`, identifying the BO to expose to userspace.
pub const DrmAmdgpuGemMmapIn = extern struct {
    handle: u32,
    _pad: u32,
};

/// Output layout returned by `DRM_IOCTL_AMDGPU_GEM_MMAP` with the file offset to pass to `mmap`.
pub const DrmAmdgpuGemMmapOut = extern struct {
    addr_ptr: u64,
};

/// Tagged union packing the in/out forms of the GEM-mmap ioctl into one shared buffer.
pub const DrmAmdgpuGemMmap = extern union {
    in: DrmAmdgpuGemMmapIn,
    out: DrmAmdgpuGemMmapOut,
};

/// Argument layout for `DRM_IOCTL_AMDGPU_GEM_VA`, the ioctl that maps a BO into the GPU virtual address space.
/// Encodes the BO handle, VA operation, page-permission flags, target VA range, and any syncobj fence handles.
pub const DrmAmdgpuGemVa = extern struct {
    handle: u32,
    _pad: u32,
    operation: u32,
    flags: u32,
    va_address: u64,
    offset_in_bo: u64,
    map_size: u64,
    vm_timeline_point: u64,
    vm_timeline_syncobj_out: u32,
    num_syncobj_handles: u32,
    input_fence_syncobj_handles: u64,
};

/// Argument layout for `DRM_IOCTL_AMDGPU_INFO`, the generic info-query ioctl.
/// `return_pointer`/`return_size` describe a userspace output buffer; `query` selects a sub-query whose discriminator-specific parameters live in `query_data`.
pub const DrmAmdgpuInfo = extern struct {
    return_pointer: u64,
    return_size: u32,
    query: u32,
    query_data: extern union {
        mode_crtc: extern struct {
            id: u32,
            _pad: u32,
        },
        query_hw_ip: extern struct {
            type: u32,
            ip_instance: u32,
        },
        read_mmr_reg: extern struct {
            dword_offset: u32,
            count: u32,
            instance: u32,
            flags: u32,
        },
        query_fw: extern struct {
            fw_type: u32,
            ip_instance: u32,
            index: u32,
            _pad: u32,
        },
        vbios_info: extern struct {
            type: u32,
            offset: u32,
        },
        sensor_info: extern struct {
            type: u32,
        },
        video_cap: extern struct {
            type: u32,
        },
    },
};

/// Output buffer for `AMDGPU_INFO_HW_IP_INFO`.
/// Reports the HW IP version and capabilities, ring-buffer alignment requirements, the bitmask of available kernel rings, and the count of user-queue slots — the last field is what tier 2 reads to confirm UMQ support.
pub const DrmAmdgpuInfoHwIp = extern struct {
    hw_ip_version_major: u32,
    hw_ip_version_minor: u32,
    capabilities_flags: u64,
    ib_start_alignment: u32,
    ib_size_alignment: u32,
    available_rings: u32,
    ip_discovery_version: u32,
    userq_num_slots: u32,
};

/// Output buffer for `AMDGPU_INFO_UQ_FW_AREAS`, sized per IP type.
/// Reports the per-queue scratch buffers the firmware needs: shadow/CSA areas for GFX, the EOP buffer for compute, and the CSA area for SDMA.
pub const DrmAmdgpuInfoUqMetadata = extern union {
    gfx: extern struct {
        shadow_size: u32,
        shadow_alignment: u32,
        csa_size: u32,
        csa_alignment: u32,
    },
    compute: extern struct {
        eop_size: u32,
        eop_alignment: u32,
    },
    sdma: extern struct {
        csa_size: u32,
        csa_alignment: u32,
    },
};

/// Input layout for the `DRM_IOCTL_AMDGPU_USERQ` ioctl.
/// Describes the create/free op, the target IP (compute, gfx, sdma), the doorbell BO and offset to ring, the VA ranges of the queue ring buffer plus its read/write pointers, and a pointer to the IP-specific MQD blob.
pub const DrmAmdgpuUserqIn = extern struct {
    op: u32,
    queue_id: u32,
    ip_type: u32,
    doorbell_handle: u32,
    doorbell_offset: u32,
    flags: u32,
    queue_va: u64,
    queue_size: u64,
    rptr_va: u64,
    wptr_va: u64,
    mqd: u64,
    mqd_size: u64,
};

/// Output layout returned by a successful `AMDGPU_USERQ_OP_CREATE`, containing the kernel-assigned queue id used by subsequent ioctls.
pub const DrmAmdgpuUserqOut = extern struct {
    queue_id: u32,
    _pad: u32,
};

/// Tagged union packing the in/out forms of the user-queue ioctl into the same buffer.
pub const DrmAmdgpuUserq = extern union {
    in: DrmAmdgpuUserqIn,
    out: DrmAmdgpuUserqOut,
};

/// GFX11 compute MQD payload pointed at by `DrmAmdgpuUserqIn.mqd`.
/// Currently only the EOP scratch VA is required; matches the kernel's `drm_amdgpu_userq_mqd_compute_gfx11` layout.
pub const DrmAmdgpuUserqMqdComputeGfx11 = extern struct {
    eop_va: u64,
};

/// Thin handle for a kernel-managed GEM buffer object.
/// Pairs the kernel GEM handle with the allocation size so callers can re-issue ioctls (mmap, VA map) without re-querying the size.
pub const Bo = struct {
    handle: u32,
    size: u64,
};

const drm_ioctl_amdgpu_gem_create = linux.IOCTL.IOWR(
    drm_ioctl_base,
    drm_command_base + drm_amdgpu_gem_create_nr,
    DrmAmdgpuGemCreate,
);

const drm_ioctl_amdgpu_gem_mmap = linux.IOCTL.IOWR(
    drm_ioctl_base,
    drm_command_base + drm_amdgpu_gem_mmap_nr,
    DrmAmdgpuGemMmap,
);

const drm_ioctl_amdgpu_info = linux.IOCTL.IOW(
    drm_ioctl_base,
    drm_command_base + drm_amdgpu_info_nr,
    DrmAmdgpuInfo,
);

const drm_ioctl_amdgpu_gem_va = linux.IOCTL.IOW(
    drm_ioctl_base,
    drm_command_base + drm_amdgpu_gem_va_nr,
    DrmAmdgpuGemVa,
);

const drm_ioctl_amdgpu_userq = linux.IOCTL.IOWR(
    drm_ioctl_base,
    drm_command_base + drm_amdgpu_userq_nr,
    DrmAmdgpuUserq,
);

/// Probe an AMDGPU render node to decide whether the compute user-mode-queue surface is usable.
/// Opens the render node, asks the kernel for compute HW IP info plus user-queue firmware areas, and validates that the queue slots and EOP buffer parameters look sane.
/// @param render_node Absolute path to the DRI render node (e.g. `/dev/dri/renderD128`).
/// @returns A `QueryResult` whose `status` field identifies the precise failure mode or `.available` on success, with `info` populated when compute UMQ is ready to use.
/// @note Returns `.unsupported_os` immediately on non-Linux builds without opening any file.
pub fn queryComputeUserq(io: std.Io, render_node: []const u8) QueryResult {
    if (builtin.os.tag != .linux) {
        return .{ .status = .unsupported_os };
    }

    var file = std.Io.Dir.openFileAbsolute(io, render_node, .{ .mode = .read_write }) catch {
        return .{ .status = .render_node_open_failed };
    };
    defer file.close(io);

    const hw_ip = queryHwIp(file, AMDGPU_HW_IP_COMPUTE) catch |err| {
        return .{
            .status = .hw_ip_info_failed,
            .errno = errorToErrno(err),
        };
    };

    if (hw_ip.available_rings == 0) {
        return .{ .status = .compute_userq_unsupported };
    }
    if (hw_ip.userq_num_slots == 0) {
        return .{ .status = .compute_userq_slots_missing };
    }

    const metadata = queryUqMetadataCompute(file) catch |err| {
        return .{
            .status = .fw_metadata_failed,
            .errno = errorToErrno(err),
        };
    };

    if (metadata.compute.eop_size == 0 or metadata.compute.eop_alignment == 0) {
        return .{ .status = .invalid_fw_metadata };
    }

    return .{
        .status = .available,
        .info = .{
            .userq_slots = hw_ip.userq_num_slots,
            .eop_size = metadata.compute.eop_size,
            .eop_alignment = metadata.compute.eop_alignment,
        },
    };
}

/// Issue `AMDGPU_INFO_HW_IP_INFO` for the given HW IP type on an open render-node file.
/// @param file Open file handle for the AMDGPU render node.
/// @param ip_type IP selector constant such as `AMDGPU_HW_IP_COMPUTE`.
/// @returns The kernel-filled `DrmAmdgpuInfoHwIp` for IP instance 0.
/// @note Returns `error.IoctlFailed` on failure; inspect `lastErrno()` for the captured errno.
pub fn queryHwIp(file: std.Io.File, ip_type: u32) !DrmAmdgpuInfoHwIp {
    var out: DrmAmdgpuInfoHwIp = std.mem.zeroes(DrmAmdgpuInfoHwIp);
    var query: DrmAmdgpuInfo = std.mem.zeroes(DrmAmdgpuInfo);
    query.return_pointer = @intFromPtr(&out);
    query.return_size = @sizeOf(DrmAmdgpuInfoHwIp);
    query.query = AMDGPU_INFO_HW_IP_INFO;
    query.query_data.query_hw_ip = .{
        .type = ip_type,
        .ip_instance = 0,
    };
    try ioctlRaw(file, drm_ioctl_amdgpu_info, @intFromPtr(&query));
    return out;
}

fn queryUqMetadataCompute(file: std.Io.File) !DrmAmdgpuInfoUqMetadata {
    var out: DrmAmdgpuInfoUqMetadata = std.mem.zeroes(DrmAmdgpuInfoUqMetadata);
    var query: DrmAmdgpuInfo = std.mem.zeroes(DrmAmdgpuInfo);
    query.return_pointer = @intFromPtr(&out);
    query.return_size = @sizeOf(DrmAmdgpuInfoUqMetadata);
    query.query = AMDGPU_INFO_UQ_FW_AREAS;
    query.query_data.query_hw_ip = .{
        .type = AMDGPU_HW_IP_COMPUTE,
        .ip_instance = 0,
    };
    try ioctlRaw(file, drm_ioctl_amdgpu_info, @intFromPtr(&query));
    return out;
}

/// Allocate a GEM buffer object via `DRM_IOCTL_AMDGPU_GEM_CREATE`.
/// @param file Open file handle for the AMDGPU render node.
/// @param size Buffer size in bytes.
/// @param alignment Required base alignment in bytes.
/// @param domains Bitmask of `AMDGPU_GEM_DOMAIN_*` constants selecting GTT, VRAM, or doorbell aperture.
/// @param flags Bitmask of `AMDGPU_GEM_CREATE_*` creation flags (e.g. USWC, VRAM-cleared, always-valid).
/// @returns A `Bo` wrapping the kernel GEM handle and the requested size for later mmap or VA-map calls.
pub fn createGem(file: std.Io.File, size: u64, alignment: u64, domains: u64, flags: u64) !Bo {
    var create: DrmAmdgpuGemCreate = std.mem.zeroes(DrmAmdgpuGemCreate);
    create.in = .{
        .bo_size = size,
        .alignment = alignment,
        .domains = domains,
        .domain_flags = flags,
    };
    try ioctlRaw(file, drm_ioctl_amdgpu_gem_create, @intFromPtr(&create));
    return .{
        .handle = create.out.handle,
        .size = size,
    };
}

/// Map a previously created GEM BO into the calling process's address space.
/// First queries the kernel for the BO's mmap offset, then issues a shared `mmap` against the render-node fd at that offset.
/// @param file Open file handle for the AMDGPU render node that owns the BO.
/// @param bo The buffer object handle and size returned by `createGem`.
/// @param prot Standard POSIX `PROT_*` page-protection flags.
/// @returns A page-aligned byte slice covering the BO; the slice length equals `bo.size`.
pub fn mmapGem(file: std.Io.File, bo: Bo, prot: std.posix.PROT) ![]align(std.heap.page_size_min) u8 {
    var mmap_args: DrmAmdgpuGemMmap = std.mem.zeroes(DrmAmdgpuGemMmap);
    mmap_args.in = .{
        .handle = bo.handle,
        ._pad = 0,
    };
    try ioctlRaw(file, drm_ioctl_amdgpu_gem_mmap, @intFromPtr(&mmap_args));

    return std.posix.mmap(
        null,
        @intCast(bo.size),
        prot,
        .{ .TYPE = .SHARED },
        file.handle,
        mmap_args.out.addr_ptr,
    );
}

/// Bind a GEM BO into the device virtual address space at a caller-chosen VA.
/// Performs an `AMDGPU_VA_OP_MAP` over the full BO size starting at offset 0.
/// @param file Open file handle for the AMDGPU render node that owns the BO.
/// @param bo The buffer object handle and size returned by `createGem`.
/// @param va Target GPU virtual address; must satisfy the page alignment the kernel enforces.
/// @param flags Bitmask of `AMDGPU_VM_PAGE_*` and `AMDGPU_VM_MTYPE_*` permission/cache flags.
pub fn mapGemVa(file: std.Io.File, bo: Bo, va: u64, flags: u32) !void {
    var args: DrmAmdgpuGemVa = std.mem.zeroes(DrmAmdgpuGemVa);
    args.handle = bo.handle;
    args.operation = AMDGPU_VA_OP_MAP;
    args.flags = flags;
    args.va_address = va;
    args.offset_in_bo = 0;
    args.map_size = bo.size;
    try ioctlRaw(file, drm_ioctl_amdgpu_gem_va, @intFromPtr(&args));
}

/// Create a compute user-mode queue via `DRM_IOCTL_AMDGPU_USERQ`.
/// Builds a GFX11 compute MQD pointing at the caller-supplied EOP scratch VA and submits an `AMDGPU_USERQ_OP_CREATE`.
/// @param file Open file handle for the AMDGPU render node.
/// @param doorbell_handle GEM handle of the doorbell BO the kernel will use to wake this queue.
/// @param doorbell_offset Doorbell slot offset within the doorbell BO assigned to this queue.
/// @param queue_va Device VA of the ring buffer backing the queue.
/// @param queue_size Ring buffer size in bytes.
/// @param rptr_va Device VA of the queue read-pointer word.
/// @param wptr_va Device VA of the queue write-pointer word.
/// @param eop_va Device VA of the end-of-pipe scratch buffer required by GFX11 compute firmware.
/// @param flags Driver-specific creation flags forwarded as-is to the kernel.
/// @returns The kernel-assigned queue id used in later doorbell rings and the matching `freeUserq` call.
pub fn createComputeUserq(
    file: std.Io.File,
    doorbell_handle: u32,
    doorbell_offset: u32,
    queue_va: u64,
    queue_size: u64,
    rptr_va: u64,
    wptr_va: u64,
    eop_va: u64,
    flags: u32,
) !u32 {
    var mqd: DrmAmdgpuUserqMqdComputeGfx11 = .{ .eop_va = eop_va };
    var args: DrmAmdgpuUserq = std.mem.zeroes(DrmAmdgpuUserq);
    args.in = .{
        .op = AMDGPU_USERQ_OP_CREATE,
        .queue_id = 0,
        .ip_type = AMDGPU_HW_IP_COMPUTE,
        .doorbell_handle = doorbell_handle,
        .doorbell_offset = doorbell_offset,
        .flags = flags,
        .queue_va = queue_va,
        .queue_size = queue_size,
        .rptr_va = rptr_va,
        .wptr_va = wptr_va,
        .mqd = @intFromPtr(&mqd),
        .mqd_size = @sizeOf(DrmAmdgpuUserqMqdComputeGfx11),
    };
    try ioctlRaw(file, drm_ioctl_amdgpu_userq, @intFromPtr(&args));
    return args.out.queue_id;
}

/// Release a user-mode queue previously returned by `createComputeUserq` via `AMDGPU_USERQ_OP_FREE`.
/// @param file Open file handle for the AMDGPU render node that owns the queue.
/// @param queue_id Kernel-assigned queue id to free.
pub fn freeUserq(file: std.Io.File, queue_id: u32) !void {
    var args: DrmAmdgpuUserq = std.mem.zeroes(DrmAmdgpuUserq);
    args.in = .{
        .op = AMDGPU_USERQ_OP_FREE,
        .queue_id = queue_id,
        .ip_type = 0,
        .doorbell_handle = 0,
        .doorbell_offset = 0,
        .flags = 0,
        .queue_va = 0,
        .queue_size = 0,
        .rptr_va = 0,
        .wptr_va = 0,
        .mqd = 0,
        .mqd_size = 0,
    };
    try ioctlRaw(file, drm_ioctl_amdgpu_userq, @intFromPtr(&args));
}

const IoctlError = error{IoctlFailed};

var last_ioctl_errno: ?linux.E = null;

/// Return the errno captured by the most recent ioctl performed through this module.
/// @returns The captured `linux.E` value, or `null` if the previous call succeeded or no call has been made yet.
/// @note The module clears the saved errno at the start of every ioctl, so the value is only meaningful immediately after an `error.IoctlFailed`.
pub fn lastErrno() ?linux.E {
    return last_ioctl_errno;
}

fn ioctlRaw(file: std.Io.File, request: u32, arg: usize) IoctlError!void {
    last_ioctl_errno = null;
    const rc = linux.ioctl(file.handle, request, arg);
    const err = linux.errno(rc);
    if (err != .SUCCESS) {
        last_ioctl_errno = err;
        return error.IoctlFailed;
    }
}

fn errorToErrno(err: anyerror) ?linux.E {
    return switch (err) {
        error.IoctlFailed => last_ioctl_errno,
        else => null,
    };
}

test "amdgpu info ioctl uapi layout is stable for userq discovery" {
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(DrmAmdgpuGemCreate));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(DrmAmdgpuGemMmap));
    try std.testing.expectEqual(@as(usize, 64), @sizeOf(DrmAmdgpuGemVa));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(DrmAmdgpuInfo));
    try std.testing.expectEqual(@as(usize, 40), @sizeOf(DrmAmdgpuInfoHwIp));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(DrmAmdgpuInfoUqMetadata));
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(DrmAmdgpuUserqIn));
    try std.testing.expectEqual(@as(usize, 72), @sizeOf(DrmAmdgpuUserq));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(DrmAmdgpuUserqMqdComputeGfx11));
}

test "queryComputeUserq is disabled on non-Linux hosts" {
    if (builtin.os.tag == .linux) return error.SkipZigTest;
    const result = queryComputeUserq("/dev/dri/renderD128");
    try std.testing.expectEqual(QueryStatus.unsupported_os, result.status);
}
