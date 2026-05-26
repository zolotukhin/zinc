//! AMDGPU DRM command-submission (CS) path — bring-up of the RADV / radeonsi
//! PM4 submission foundation.
//!
//! T1 PM4-direct reaches the AMD command processor through three Linux ABIs:
//!   * `DRM_IOCTL_AMDGPU_USERQ` — the user-mode-queue ABI; the bench-node
//!     firmware reports zero compute USERQ slots, so it is unusable here
//!     (see `umq.zig`).
//!   * `/dev/kfd` `AMDKFD_IOC_CREATE_QUEUE` + a doorbell ring — works to create
//!     a raw `QUEUE_TYPE_COMPUTE` queue, but the MES never retires the PM4 we
//!     stage in it on this kernel (see `kfd.zig`).
//!   * `DRM_IOCTL_AMDGPU_CS` — the kernel-managed command-submission UAPI every
//!     AMD userspace driver (RADV, radeonsi, amdvlk) rides. The kernel owns the
//!     ring / doorbell / MES bookkeeping; userspace hands it an indirect buffer
//!     (IB) of PM4 and waits on the retired fence. This is the reliable
//!     foundation the GPU compute dispatch lowers onto.
//!
//! This module brings the CS path's first retired PM4 batch up as a
//! benchmark-visible gate: open the render node, query the compute HW IP,
//! allocate an amdgpu context, create a persistent BO list for a GTT
//! indirect-buffer BO plus a signal BO, map them into the GPU VM at low VAs,
//! submit two PM4 `WRITE_DATA` streams through `DRM_IOCTL_AMDGPU_CS` using the
//! same context/BO list, and wait for the returned fences with
//! `DRM_IOCTL_AMDGPU_WAIT_CS`.
//!
//! This is not the final T1/T2 ring from the design; it is the kernel-managed
//! CS baseline used to validate packet bytes, BO residency, VM mapping, and
//! fence retirement before lowering real decode slices onto the direct tiers.
//! @section Inference Runtime
const std = @import("std");
const builtin = @import("builtin");
const kmd = @import("../kmd.zig");
const packet = @import("packet.zig");

const linux = std.os.linux;

/// Default DRM render node used by the CS bring-up gate when no path is provided.
pub const default_render_node = "/dev/dri/renderD128";

// HW IP block ids (uapi/drm/amdgpu_drm.h).
/// amdgpu HW IP block id for the graphics ring (uapi/drm/amdgpu_drm.h).
pub const AMDGPU_HW_IP_GFX: u32 = 0;
/// amdgpu HW IP block id for the async compute ring used by ZINC submissions.
pub const AMDGPU_HW_IP_COMPUTE: u32 = 1;

// Context ops.
/// `DRM_AMDGPU_CTX` op selector for allocating a new submission context.
pub const AMDGPU_CTX_OP_ALLOC_CTX: u32 = 1;
/// `DRM_AMDGPU_CTX` op selector for releasing a previously allocated context.
pub const AMDGPU_CTX_OP_FREE_CTX: u32 = 2;

// BO-list ops.
/// `DRM_AMDGPU_BO_LIST` op selector to create a residency BO list handle.
pub const AMDGPU_BO_LIST_OP_CREATE: u32 = 0;
/// `DRM_AMDGPU_BO_LIST` op selector to destroy a previously created BO list.
pub const AMDGPU_BO_LIST_OP_DESTROY: u32 = 1;

// CS chunk ids.
/// CS chunk id for an indirect-buffer descriptor (`DrmAmdgpuCsChunkIb`).
pub const AMDGPU_CHUNK_ID_IB: u32 = 0x01;
/// CS chunk id for an inline BO-handles list, an alternative to a pre-created BO list.
pub const AMDGPU_CHUNK_ID_BO_HANDLES: u32 = 0x06;

// IB flags.
/// IB flag instructing the kernel to emit a memory-sync packet around the IB
/// so writes from the BO list reach DRAM before/after the dispatch.
pub const AMDGPU_IB_FLAG_EMIT_MEM_SYNC: u32 = 1 << 6;

const drm_ioctl_base: u8 = 'd';
const drm_command_base: u8 = 0x40;
const drm_amdgpu_ctx_nr: u8 = 0x02;
const drm_amdgpu_bo_list_nr: u8 = 0x03;
const drm_amdgpu_cs_nr: u8 = 0x04;
const drm_amdgpu_wait_cs_nr: u8 = 0x09;

/// Input payload of `DRM_IOCTL_AMDGPU_CTX`: selects an op and carries the
/// caller-supplied `ctx_id` and submission priority for that op.
pub const DrmAmdgpuCtxIn = extern struct {
    op: u32,
    flags: u32,
    ctx_id: u32,
    priority: i32,
};

/// Output payload of `AMDGPU_CTX_OP_ALLOC_CTX`: the kernel-assigned context id
/// returned in the same `DrmAmdgpuCtx` union after a successful allocation.
pub const DrmAmdgpuCtxOutAlloc = extern struct {
    ctx_id: u32,
    _pad: u32,
};

/// Output payload of the `AMDGPU_CTX_OP_QUERY_STATE` op: GPU reset state and
/// hang counter for the queried context (unused on the bring-up path).
pub const DrmAmdgpuCtxOutState = extern struct {
    flags: u64,
    hangs: u32,
    reset_status: u32,
};

/// Tagged union passed to `DRM_IOCTL_AMDGPU_CTX` covering the input request
/// and the two output shapes (alloc / query-state).
pub const DrmAmdgpuCtx = extern union {
    in: DrmAmdgpuCtxIn,
    out_alloc: DrmAmdgpuCtxOutAlloc,
    out_state: DrmAmdgpuCtxOutState,
};

/// Input payload of `DRM_IOCTL_AMDGPU_BO_LIST`: op selector plus a pointer to
/// an array of `DrmAmdgpuBoListEntry` describing the BOs the submission must
/// keep resident.
pub const DrmAmdgpuBoListIn = extern struct {
    operation: u32,
    list_handle: u32,
    bo_number: u32,
    bo_info_size: u32,
    bo_info_ptr: u64,
};

/// Output payload of `AMDGPU_BO_LIST_OP_CREATE`: the kernel-assigned BO list
/// handle referenced from subsequent CS submissions.
pub const DrmAmdgpuBoListOut = extern struct {
    list_handle: u32,
    _pad: u32,
};

/// Tagged union passed to `DRM_IOCTL_AMDGPU_BO_LIST` covering input and output.
pub const DrmAmdgpuBoList = extern union {
    in: DrmAmdgpuBoListIn,
    out: DrmAmdgpuBoListOut,
};

/// Single residency entry inside a BO list: the GEM handle to make resident
/// and a kernel-visible priority hint for eviction.
pub const DrmAmdgpuBoListEntry = extern struct {
    bo_handle: u32,
    bo_priority: u32,
};

/// One chunk inside a `DRM_IOCTL_AMDGPU_CS` submission: a typed sub-payload
/// (`chunk_id`, length in dwords, pointer to the chunk data).
pub const DrmAmdgpuCsChunk = extern struct {
    chunk_id: u32,
    length_dw: u32,
    chunk_data: u64,
};

/// Input payload of `DRM_IOCTL_AMDGPU_CS`: binds a context, BO list and an
/// array of typed chunks (the IB descriptor lives in one of those chunks).
pub const DrmAmdgpuCsIn = extern struct {
    ctx_id: u32,
    bo_list_handle: u32,
    num_chunks: u32,
    flags: u32,
    chunks: u64,
};

/// Output payload of `DRM_IOCTL_AMDGPU_CS`: the fence handle the caller waits
/// on via `DRM_IOCTL_AMDGPU_WAIT_CS` for the submission to retire.
pub const DrmAmdgpuCsOut = extern struct {
    handle: u64,
};

/// Tagged union passed to `DRM_IOCTL_AMDGPU_CS` covering input and output.
pub const DrmAmdgpuCs = extern union {
    in: DrmAmdgpuCsIn,
    out: DrmAmdgpuCsOut,
};

/// Chunk payload for `AMDGPU_CHUNK_ID_IB`: describes the indirect-buffer VA,
/// its size in bytes, the target IP type/ring, and submission flags such as
/// `AMDGPU_IB_FLAG_EMIT_MEM_SYNC`.
pub const DrmAmdgpuCsChunkIb = extern struct {
    _pad: u32,
    flags: u32,
    va_start: u64,
    ib_bytes: u32,
    ip_type: u32,
    ip_instance: u32,
    ring: u32,
};

/// Input payload of `DRM_IOCTL_AMDGPU_WAIT_CS`: identifies the fence to wait
/// on (by `handle`/`ctx_id` against a specific IP/ring) and the timeout.
pub const DrmAmdgpuWaitCsIn = extern struct {
    handle: u64,
    timeout: u64,
    ip_type: u32,
    ip_instance: u32,
    ring: u32,
    ctx_id: u32,
};

/// Output payload of `DRM_IOCTL_AMDGPU_WAIT_CS`: zero on successful retirement,
/// nonzero on timeout or fence error.
pub const DrmAmdgpuWaitCsOut = extern struct {
    status: u64,
};

/// Tagged union passed to `DRM_IOCTL_AMDGPU_WAIT_CS` covering input and output.
pub const DrmAmdgpuWaitCs = extern union {
    in: DrmAmdgpuWaitCsIn,
    out: DrmAmdgpuWaitCsOut,
};

const ioc_ctx = linux.IOCTL.IOWR(drm_ioctl_base, drm_command_base + drm_amdgpu_ctx_nr, DrmAmdgpuCtx);
const ioc_bo_list = linux.IOCTL.IOWR(drm_ioctl_base, drm_command_base + drm_amdgpu_bo_list_nr, DrmAmdgpuBoList);
const ioc_cs = linux.IOCTL.IOWR(drm_ioctl_base, drm_command_base + drm_amdgpu_cs_nr, DrmAmdgpuCs);
const ioc_wait_cs = linux.IOCTL.IOWR(drm_ioctl_base, drm_command_base + drm_amdgpu_wait_cs_nr, DrmAmdgpuWaitCs);

const compute_pgm_rsrc1_value: u32 = 0xe0000001;
const compute_pgm_rsrc2_argmax_top2_value: u32 = 0x90; // 8 user SGPRs + workgroup-id-x.
const shader_offset_argmax_top2: usize = 0x000;
const shader_offset_rms_norm_elem0: usize = 0x100;
const shader_offset_dmmv_f32_row_range: usize = 0x200;

// gfx1201 one-wave kernel assembled with:
//   llvm-mc-20 -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj
//
// ABI:
//   s[2:3] = output GTT pointer
//   s4     = ordered score 0
//   s5     = ordered score 1
//   s6     = token 0
//   s7     = token 1
//
// It computes the top-2 argmax choice and stores the selected token.
const argmax_top2_gfx1201 = [_]u32{
    0xbf090504, // s_cmp_ge_u32 s4, s5
    0x98040706, // s_cselect_b32 s4, s6, s7
    0x7e020280, // v_mov_b32_e32 v1, 0
    0x7e000204, // v_mov_b32_e32 v0, s4
    0xee068002, // global_store_b32 v1, v0, s[2:3]
    0x00000000,
    0x00000001,
    0xbf800000, // s_nop 0
    0xbfb60003, // s_sendmsg(MSG_DEALLOC_VGPRS)
    0xbfb00000, // s_endpgm
};

// gfx1201 one-wave kernel assembled with:
//   llvm-mc-20 -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj
//
// ABI:
//   s[0:1] = input GTT pointer: f32 hidden0, f32 inv_rms, f32 weight0
//   s[2:3] = output GTT pointer
//
// It computes one real final-RMS-norm element:
//   output[0] = hidden0 * inv_rms * weight0
const rms_norm_elem0_gfx1201 = [_]u32{
    0x7e060280, // v_mov_b32_e32 v3, 0
    0xee050000, 0x00000000, 0x00000003, // global_load_b32 v0, v3, s[0:1]
    0xee050000, 0x00000001, 0x00000403, // global_load_b32 v1, v3, s[0:1] offset:4
    0xee050000, 0x00000002, 0x00000803, // global_load_b32 v2, v3, s[0:1] offset:8
    0xbf8903f7, // s_waitcnt vmcnt(0)
    0x10000101, // v_mul_f32_e32 v0, v1, v0
    0x10000102, // v_mul_f32_e32 v0, v2, v0
    0xee068002, 0x00000000, 0x00000003, // global_store_b32 v3, v0, s[2:3]
    0xbf800000, // s_nop 0
    0xbfb60003, // s_sendmsg(MSG_DEALLOC_VGPRS)
    0xbfb00000, // s_endpgm
};

// gfx1201 wave64 F32 DMMV row-range kernel assembled with:
//   llvm-mc-20 -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj
//
// ABI:
//   s[0:1] = input f32 vector pointer
//   s[2:3] = output f32 row-result pointer
//   s[4:5] = f32 weight rows pointer
//   s6     = cols, multiple of 64
//   s7     = unused
//   s8     = workgroup_id_x, one wave per row
//
// A single GPU workitem serially computes a compact row range and stores
// output[row]. This is intentionally small and row-oriented: it is the first
// consumed model row range, not the final DMMV kernel shape.
const dmmv_f32_row_range_gfx1201 = [_]u32{
    0xbe890080, // s_mov_b32 s9, 0
    0xbf090709, // s_cmp_ge_u32 s9, s7
    0xbfa2001a, // s_cbranch_scc1 +26
    0x960a0609, // s_mul_i32 s10, s9, s6
    0x840a820a, // s_lshl_b32 s10, s10, 2
    0xbe8b0080, // s_mov_b32 s11, 0
    0x7e020280, // v_mov_b32_e32 v1, 0
    0xbf09060b, // s_cmp_ge_u32 s11, s6
    0xbfa2000d, // s_cbranch_scc1 +13
    0x840c820b, // s_lshl_b32 s12, s11, 2
    0x7e04020c, // v_mov_b32_e32 v2, s12
    0x4a06040a, // v_add_nc_u32_e32 v3, s10, v2
    0xee050000, 0x00000004, 0x00000002, // global_load_b32 v4, v2, s[0:1]
    0xee050004, 0x00000005, 0x00000003, // global_load_b32 v5, v3, s[4:5]
    0xbf8903f7, // s_waitcnt vmcnt(0)
    0x56020b04, // v_fmac_f32_e32 v1, v4, v5
    0x800b810b, // s_add_co_u32 s11, s11, 1
    0xbfa0fff1, // s_branch -15
    0x840c8209, // s_lshl_b32 s12, s9, 2
    0x7e04020c, // v_mov_b32_e32 v2, s12
    0xee068002, 0x00800000, 0x00000002, // global_store_b32 v2, v1, s[2:3]
    0x80098109, // s_add_co_u32 s9, s9, 1
    0xbfa0ffe4, // s_branch -28
    0xbf800000, // s_nop 0
    0xbfb60003, // s_sendmsg(MSG_DEALLOC_VGPRS)
    0xbfb00000, // s_endpgm
};

/// Outcome classification for the CS bring-up smoke gate.
/// Each variant maps to a specific failure point in the open → submit → wait
/// pipeline, so the benchmark UI can attribute a regression to render-node
/// access, kernel ABI mismatch, BO/VA setup, submission, or fence retirement.
pub const SmokeStatus = enum {
    ok,
    unsupported_os,
    render_node_open_failed,
    hw_ip_query_failed,
    no_rings,
    ctx_alloc_failed,
    va_reservation_failed,
    ib_bo_failed,
    ib_map_failed,
    ib_va_failed,
    signal_bo_failed,
    signal_map_failed,
    signal_va_failed,
    bo_list_failed,
    cs_submit_failed,
    wait_cs_failed,
    wait_timeout,
    signal_check_failed,
};

/// Structured result returned by the CS bring-up smoke gate.
/// Captures the rendezvous addresses, kernel-assigned handles, observed signal
/// value, fence handles, and the final `SmokeStatus` so benchmark output can
/// surface a precise failure mode without re-running the path.
pub const SmokeResult = struct {
    status: SmokeStatus,
    render_node: []const u8 = default_render_node,
    ip_type: u32 = 0,
    available_rings: u32 = 0,
    ctx_id: u32 = 0,
    bo_list_handle: u32 = 0,
    ib_va: u64 = 0,
    ib_bytes: u32 = 0,
    signal_va: u64 = 0,
    signal_value: u64 = 0,
    submit_count: u32 = 0,
    first_fence_handle: u64 = 0,
    fence_handle: u64 = 0,
    wait_status: u64 = 0,
    errno: ?linux.E = null,

    /// Returns true when both PM4 submissions retired and the signal BO read
    /// back the expected sentinel value.
    pub fn ok(self: SmokeResult) bool {
        return self.status == .ok;
    }
};

/// Per-token CS submission context for the PM4 bring-up tiers.
///
/// Owns the long-lived amdgpu context, BO list, and the GPU-mapped indirect-
/// buffer / input / output / signal / shader buffers used by the
/// `copyU32`, `argmaxTop2`, `rmsNormElement0` and `dmmvF32RowRange` dispatches.
/// Reused across many submissions so each decode step only re-records PM4 into
/// the existing IB and re-submits via `DRM_IOCTL_AMDGPU_CS`.
pub const TokenBoundary = struct {
    io: std.Io,
    file: std.Io.File,
    ctx_id: u32,
    ip_type: u32,
    bo_list_handle: u32,
    ib_va: u64,
    input_va: u64,
    output_va: u64,
    signal_va: u64,
    shader_va: u64,
    ib_map: []align(std.heap.page_size_min) u8,
    input_map: []align(std.heap.page_size_min) u8,
    output_map: []align(std.heap.page_size_min) u8,
    signal_map: []align(std.heap.page_size_min) u8,
    shader_map: []align(std.heap.page_size_min) u8,
    builder: packet.PacketBuilder,
    submit_count: u32 = 0,
    last_fence_handle: u64 = 0,
    last_wait_status: u64 = 0,
    last_ib_bytes: u32 = 0,

    /// Open the canonical render node (`default_render_node`) and finish the
    /// full CS bring-up: context, BO list, IB / input / output / signal /
    /// shader buffers, all mapped into a low GPU VA range.
    /// @returns A ready `TokenBoundary` whose `builder` can record PM4 immediately.
    pub fn initDefault(io: std.Io) !TokenBoundary {
        return initPath(io, default_render_node);
    }

    /// Open the given render node and bring up the full CS submission state.
    ///
    /// Allocates an amdgpu context, creates GTT-backed BOs for the indirect
    /// buffer, input scratch (~2 MiB), output, signal and shader pages, maps
    /// each into a fixed low GPU VA so the kernel does not need to re-bind
    /// them per submission, uploads the three gfx1201 PM4 kernels into the
    /// shader page, and creates a persistent BO list referencing all five BOs.
    /// @param render_node Absolute path to the amdgpu DRM render node (e.g. `/dev/dri/renderD128`).
    /// @returns A ready `TokenBoundary` on success; the relevant `error.*Failed` variant otherwise.
    /// @note Linux-only; returns `error.UnsupportedOs` on other platforms.
    pub fn initPath(io: std.Io, render_node: []const u8) !TokenBoundary {
        if (builtin.os.tag != .linux) return error.UnsupportedOs;

        var file = std.Io.Dir.openFileAbsolute(io, render_node, .{ .mode = .read_write }) catch return error.RenderNodeOpenFailed;
        errdefer file.close(io);

        const ip_type: u32 = AMDGPU_HW_IP_COMPUTE;
        const hw_ip = kmd.queryHwIp(file, ip_type) catch return error.HwIpQueryFailed;
        if (hw_ip.available_rings == 0) return error.NoComputeRings;

        var ctx: DrmAmdgpuCtx = std.mem.zeroes(DrmAmdgpuCtx);
        ctx.in = .{ .op = AMDGPU_CTX_OP_ALLOC_CTX, .flags = 0, .ctx_id = 0, .priority = 0 };
        ioctlRaw(file, ioc_ctx, @intFromPtr(&ctx)) catch return error.CtxAllocFailed;
        const ctx_id = ctx.out_alloc.ctx_id;
        errdefer freeContext(file, ctx_id);

        const ib_bo_size: usize = 64 * 1024;
        const input_bo_size: usize = 2 * 1024 * 1024;
        const page_size: usize = 4096;
        const ib_va: u64 = 0x1_0200_0000;
        const input_va: u64 = ib_va + ib_bo_size;
        const output_va: u64 = input_va + input_bo_size;
        const signal_va: u64 = output_va + page_size;
        const shader_va: u64 = signal_va + page_size;

        const ib_bo = kmd.createGem(file, ib_bo_size, 256, kmd.AMDGPU_GEM_DOMAIN_GTT, kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC) catch return error.IbBoFailed;
        const ib_map = kmd.mmapGem(file, ib_bo, std.posix.PROT{ .READ = true, .WRITE = true }) catch return error.IbMapFailed;
        errdefer std.posix.munmap(ib_map);
        const exec_va_flags = kmd.AMDGPU_VM_PAGE_READABLE | kmd.AMDGPU_VM_PAGE_WRITEABLE | kmd.AMDGPU_VM_PAGE_EXECUTABLE | kmd.AMDGPU_VM_MTYPE_DEFAULT;
        kmd.mapGemVa(file, ib_bo, ib_va, exec_va_flags) catch return error.IbVaFailed;

        const data_va_flags = kmd.AMDGPU_VM_PAGE_READABLE | kmd.AMDGPU_VM_PAGE_WRITEABLE | kmd.AMDGPU_VM_MTYPE_DEFAULT;
        const input_bo = kmd.createGem(file, input_bo_size, 256, kmd.AMDGPU_GEM_DOMAIN_GTT, kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC) catch return error.InputBoFailed;
        const input_map = kmd.mmapGem(file, input_bo, std.posix.PROT{ .READ = true, .WRITE = true }) catch return error.InputMapFailed;
        errdefer std.posix.munmap(input_map);
        kmd.mapGemVa(file, input_bo, input_va, data_va_flags) catch return error.InputVaFailed;

        const output_bo = kmd.createGem(file, page_size, 256, kmd.AMDGPU_GEM_DOMAIN_GTT, kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC) catch return error.OutputBoFailed;
        const output_map = kmd.mmapGem(file, output_bo, std.posix.PROT{ .READ = true, .WRITE = true }) catch return error.OutputMapFailed;
        errdefer std.posix.munmap(output_map);
        kmd.mapGemVa(file, output_bo, output_va, data_va_flags) catch return error.OutputVaFailed;

        const signal_bo = kmd.createGem(file, page_size, 256, kmd.AMDGPU_GEM_DOMAIN_GTT, kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC) catch return error.SignalBoFailed;
        const signal_map = kmd.mmapGem(file, signal_bo, std.posix.PROT{ .READ = true, .WRITE = true }) catch return error.SignalMapFailed;
        errdefer std.posix.munmap(signal_map);
        kmd.mapGemVa(file, signal_bo, signal_va, data_va_flags) catch return error.SignalVaFailed;

        const shader_bo = kmd.createGem(file, page_size, 256, kmd.AMDGPU_GEM_DOMAIN_GTT, kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC) catch return error.ShaderBoFailed;
        const shader_map = kmd.mmapGem(file, shader_bo, std.posix.PROT{ .READ = true, .WRITE = true }) catch return error.ShaderMapFailed;
        errdefer std.posix.munmap(shader_map);
        kmd.mapGemVa(file, shader_bo, shader_va, exec_va_flags) catch return error.ShaderVaFailed;

        const shader_words = @as([*]u32, @ptrCast(@alignCast(shader_map.ptr)))[0 .. page_size / @sizeOf(u32)];
        for (shader_words) |*word| word.* = 0xbfb00000;
        for (argmax_top2_gfx1201, 0..) |word, i| shader_words[shader_offset_argmax_top2 / @sizeOf(u32) + i] = word;
        for (rms_norm_elem0_gfx1201, 0..) |word, i| shader_words[shader_offset_rms_norm_elem0 / @sizeOf(u32) + i] = word;
        for (dmmv_f32_row_range_gfx1201, 0..) |word, i| shader_words[shader_offset_dmmv_f32_row_range / @sizeOf(u32) + i] = word;
        storeFence();

        var bo_entries = [_]DrmAmdgpuBoListEntry{
            .{ .bo_handle = ib_bo.handle, .bo_priority = 0 },
            .{ .bo_handle = input_bo.handle, .bo_priority = 0 },
            .{ .bo_handle = output_bo.handle, .bo_priority = 0 },
            .{ .bo_handle = signal_bo.handle, .bo_priority = 0 },
            .{ .bo_handle = shader_bo.handle, .bo_priority = 0 },
        };
        var bo_list: DrmAmdgpuBoList = std.mem.zeroes(DrmAmdgpuBoList);
        bo_list.in = .{
            .operation = AMDGPU_BO_LIST_OP_CREATE,
            .list_handle = 0,
            .bo_number = bo_entries.len,
            .bo_info_size = @sizeOf(DrmAmdgpuBoListEntry),
            .bo_info_ptr = @intFromPtr(&bo_entries),
        };
        ioctlRaw(file, ioc_bo_list, @intFromPtr(&bo_list)) catch return error.BoListFailed;
        const bo_list_handle = bo_list.out.list_handle;
        errdefer destroyBoList(file, bo_list_handle);

        const ib_words = @as([*]u32, @ptrCast(@alignCast(ib_map.ptr)))[0 .. ib_bo_size / @sizeOf(u32)];
        return .{
            .io = io,
            .file = file,
            .ctx_id = ctx_id,
            .ip_type = ip_type,
            .bo_list_handle = bo_list_handle,
            .ib_va = ib_va,
            .input_va = input_va,
            .output_va = output_va,
            .signal_va = signal_va,
            .shader_va = shader_va,
            .ib_map = ib_map,
            .input_map = input_map,
            .output_map = output_map,
            .signal_map = signal_map,
            .shader_map = shader_map,
            .builder = packet.PacketBuilder.init(ib_words),
        };
    }

    /// Tear down every kernel resource the `init*` paths created: destroy the
    /// BO list, free the amdgpu context, `munmap` each CPU mapping, and close
    /// the render-node file descriptor.
    /// @note Leaves the struct in an `undefined` state; do not reuse it.
    pub fn deinit(self: *TokenBoundary) void {
        destroyBoList(self.file, self.bo_list_handle);
        freeContext(self.file, self.ctx_id);
        std.posix.munmap(self.signal_map);
        std.posix.munmap(self.output_map);
        std.posix.munmap(self.input_map);
        std.posix.munmap(self.shader_map);
        std.posix.munmap(self.ib_map);
        self.file.close(self.io);
        self.* = undefined;
    }

    /// Round-trip one `u32` through the GPU as the simplest end-to-end gate:
    /// PM4 `COPY_DATA` from the input page to the output page, plus a
    /// `WRITE_DATA` of a per-submission sentinel into the signal page.
    /// @param value 32-bit payload to copy.
    /// @returns The value the GPU wrote into `output_map[0]`.
    /// @note Returns `error.SignalMismatch` if the post-fence signal value does not match the expected sentinel.
    pub fn copyU32(self: *TokenBoundary, value: u32) !u32 {
        const input_words: [*]volatile u32 = @ptrCast(@alignCast(self.input_map.ptr));
        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        input_words[0] = value;
        output_words[0] = 0xffff_ffff;
        signal_words[0] = 0;
        signal_words[1] = 0;

        const signal_expected: u64 = 0x5A494E435254_1000 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);
        try self.builder.copyData32(self.input_va, self.output_va);
        try self.builder.writeData64(self.signal_va, signal_expected);
        try self.builder.padToAlignment(64);
        storeFence();

        var ib_chunk_data: DrmAmdgpuCsChunkIb = .{
            ._pad = 0,
            .flags = AMDGPU_IB_FLAG_EMIT_MEM_SYNC,
            .va_start = self.ib_va,
            .ib_bytes = 0,
            .ip_type = self.ip_type,
            .ip_instance = 0,
            .ring = 0,
        };
        var chunks = [_]DrmAmdgpuCsChunk{.{
            .chunk_id = AMDGPU_CHUNK_ID_IB,
            .length_dw = @sizeOf(DrmAmdgpuCsChunkIb) / @sizeOf(u32),
            .chunk_data = @intFromPtr(&ib_chunk_data),
        }};
        var chunk_ptrs = [_]u64{@intFromPtr(&chunks[0])};
        self.last_fence_handle = try submitBuilderAndWait(
            self.file,
            self.ctx_id,
            self.ip_type,
            self.bo_list_handle,
            &self.builder,
            &ib_chunk_data,
            &chunk_ptrs,
            &self.last_ib_bytes,
            &self.last_wait_status,
        );
        self.submit_count += 1;

        const signal_value = @as(u64, signal_words[0]) | (@as(u64, signal_words[1]) << 32);
        if (signal_value != signal_expected) return error.SignalMismatch;
        return output_words[0];
    }

    /// Alias for `copyU32` framed as the per-token decode pulse: prove the
    /// GPU produced a token by round-tripping its id through a real PM4
    /// submission and fence wait.
    /// @param token_id Token id to round-trip through the GPU.
    /// @returns The id the GPU echoed back into the output page.
    pub fn produceToken(self: *TokenBoundary, token_id: u32) !u32 {
        return self.copyU32(token_id);
    }

    /// Dispatch the gfx1201 top-2 argmax kernel and return the selected token.
    ///
    /// Loads the argmax program into the compute SGPRs, packs the output VA
    /// and the two `(score, token)` pairs into `compute_user_data_0..7`, fires
    /// one workgroup, then waits on the signal sentinel before reading the
    /// kernel-chosen token from the output page.
    /// @param token0 First candidate token id.
    /// @param score0 Logit/score for `token0` (compared via ordered f32 bits).
    /// @param token1 Second candidate token id.
    /// @param score1 Logit/score for `token1`.
    /// @returns Whichever of `token0`/`token1` the kernel selected.
    /// @note Returns `error.SignalMismatch` on fence mismatch or `error.ArgmaxTop2InvalidToken` if the kernel writes anything else.
    pub fn argmaxTop2(
        self: *TokenBoundary,
        token0: u32,
        score0: f32,
        token1: u32,
        score1: f32,
    ) !u32 {
        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        output_words[0] = 0xffff_ffff;
        signal_words[0] = 0;
        signal_words[1] = 0;

        const signal_expected: u64 = 0x5A494E435254_3000 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_argmax_top2;
        const pgm_lo: u32 = @truncate(pgm_va >> 8);
        const pgm_hi: u32 = @truncate(pgm_va >> 40);
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ pgm_lo, pgm_hi });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            compute_pgm_rsrc1_value,
            compute_pgm_rsrc2_argmax_top2_value,
        });
        try self.builder.setShRegOne(packet.sh_reg_pgm_rsrc3, 0);
        try self.builder.setShReg(packet.sh_reg_num_thread_x, &[_]u32{ 1, 1, 1 });
        try self.builder.setShReg(packet.sh_reg_resource_limits, &[_]u32{
            0,
            0xffff_ffff,
            0xffff_ffff,
        });

        const out_lo: u32 = @truncate(self.output_va);
        const out_hi: u32 = @truncate(self.output_va >> 32);
        try self.builder.setShReg(packet.compute_user_data_0, &[_]u32{
            0,
            0,
            out_lo,
            out_hi,
            orderedF32(score0),
            orderedF32(score1),
            token0,
            token1,
        });
        try self.builder.dispatchDirectInitiator(1, 1, 1, packet.dispatch_initiator_compute);
        try self.builder.writeData64(self.signal_va, signal_expected);
        try self.builder.padToAlignment(64);
        storeFence();

        var ib_chunk_data: DrmAmdgpuCsChunkIb = .{
            ._pad = 0,
            .flags = AMDGPU_IB_FLAG_EMIT_MEM_SYNC,
            .va_start = self.ib_va,
            .ib_bytes = 0,
            .ip_type = self.ip_type,
            .ip_instance = 0,
            .ring = 0,
        };
        var chunks = [_]DrmAmdgpuCsChunk{.{
            .chunk_id = AMDGPU_CHUNK_ID_IB,
            .length_dw = @sizeOf(DrmAmdgpuCsChunkIb) / @sizeOf(u32),
            .chunk_data = @intFromPtr(&ib_chunk_data),
        }};
        var chunk_ptrs = [_]u64{@intFromPtr(&chunks[0])};
        self.last_fence_handle = try submitBuilderAndWait(
            self.file,
            self.ctx_id,
            self.ip_type,
            self.bo_list_handle,
            &self.builder,
            &ib_chunk_data,
            &chunk_ptrs,
            &self.last_ib_bytes,
            &self.last_wait_status,
        );
        self.submit_count += 1;

        const signal_value = @as(u64, signal_words[0]) | (@as(u64, signal_words[1]) << 32);
        if (signal_value != signal_expected) return error.SignalMismatch;
        const selected = output_words[0];
        if (selected != token0 and selected != token1) return error.ArgmaxTop2InvalidToken;
        return selected;
    }

    /// Dispatch the single-element gfx1201 final-RMS-norm kernel.
    ///
    /// Stores `hidden0 * inv_rms * weight0` into `output_map[0]` via a real
    /// PM4 dispatch on the compute ring, with a signal sentinel verifying
    /// retirement.
    /// @param hidden0 First hidden-state element.
    /// @param inv_rms Pre-computed inverse RMS scale.
    /// @param weight0 First RMS-norm weight.
    /// @returns The fused `hidden0 * inv_rms * weight0` value the GPU produced.
    /// @note Returns `error.SignalMismatch` if the signal sentinel does not match the expected per-submission value.
    pub fn rmsNormElement0(
        self: *TokenBoundary,
        hidden0: f32,
        inv_rms: f32,
        weight0: f32,
    ) !f32 {
        const input_words: [*]volatile u32 = @ptrCast(@alignCast(self.input_map.ptr));
        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        input_words[0] = @bitCast(hidden0);
        input_words[1] = @bitCast(inv_rms);
        input_words[2] = @bitCast(weight0);
        output_words[0] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_4000 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_rms_norm_elem0;
        const pgm_lo: u32 = @truncate(pgm_va >> 8);
        const pgm_hi: u32 = @truncate(pgm_va >> 40);
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ pgm_lo, pgm_hi });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            compute_pgm_rsrc1_value,
            compute_pgm_rsrc2_argmax_top2_value,
        });
        try self.builder.setShRegOne(packet.sh_reg_pgm_rsrc3, 0);
        try self.builder.setShReg(packet.sh_reg_num_thread_x, &[_]u32{ 1, 1, 1 });
        try self.builder.setShReg(packet.sh_reg_resource_limits, &[_]u32{
            0,
            0xffff_ffff,
            0xffff_ffff,
        });

        const in_lo: u32 = @truncate(self.input_va);
        const in_hi: u32 = @truncate(self.input_va >> 32);
        const out_lo: u32 = @truncate(self.output_va);
        const out_hi: u32 = @truncate(self.output_va >> 32);
        try self.builder.setShReg(packet.compute_user_data_0, &[_]u32{
            in_lo,
            in_hi,
            out_lo,
            out_hi,
            0,
            0,
            0,
            0,
        });
        try self.builder.dispatchDirectInitiator(1, 1, 1, packet.dispatch_initiator_compute);
        try self.builder.writeData64(self.signal_va, signal_expected);
        try self.builder.padToAlignment(64);
        storeFence();

        var ib_chunk_data: DrmAmdgpuCsChunkIb = .{
            ._pad = 0,
            .flags = AMDGPU_IB_FLAG_EMIT_MEM_SYNC,
            .va_start = self.ib_va,
            .ib_bytes = 0,
            .ip_type = self.ip_type,
            .ip_instance = 0,
            .ring = 0,
        };
        var chunks = [_]DrmAmdgpuCsChunk{.{
            .chunk_id = AMDGPU_CHUNK_ID_IB,
            .length_dw = @sizeOf(DrmAmdgpuCsChunkIb) / @sizeOf(u32),
            .chunk_data = @intFromPtr(&ib_chunk_data),
        }};
        var chunk_ptrs = [_]u64{@intFromPtr(&chunks[0])};
        self.last_fence_handle = try submitBuilderAndWait(
            self.file,
            self.ctx_id,
            self.ip_type,
            self.bo_list_handle,
            &self.builder,
            &ib_chunk_data,
            &chunk_ptrs,
            &self.last_ib_bytes,
            &self.last_wait_status,
        );
        self.submit_count += 1;

        const signal_value = @as(u64, signal_words[0]) | (@as(u64, signal_words[1]) << 32);
        if (signal_value != signal_expected) return error.SignalMismatch;
        return @bitCast(output_words[0]);
    }

    /// Dispatch the gfx1201 row-range f32 dense matrix-vector kernel.
    ///
    /// Copies the input vector and the row-major f32 weight block into the
    /// shared input page (64-byte aligned), records PM4 that points the
    /// kernel at the input/weights/output pages and the `rows`/`cols`
    /// arguments, and waits on the signal sentinel before reading `output`.
    /// This is the first row-oriented dense compute kernel the CS path runs;
    /// `cols` must be a multiple of 64 and `output` must hold at least `rows`
    /// elements.
    /// @param input Input activation vector of length `cols`.
    /// @param weights_f32 Row-major weight bytes; must hold at least `rows*cols*4` bytes.
    /// @param rows Number of output rows to compute.
    /// @param cols Inner dimension; must be a multiple of 64.
    /// @param output Output slice receiving `rows` f32 values.
    /// @note Returns `error.ShapeMismatch`, `error.InputTooLarge`, `error.OutputTooLarge`, or `error.SignalMismatch` on invalid shapes or signal-readback failure.
    pub fn dmmvF32RowRange(
        self: *TokenBoundary,
        input: []const f32,
        weights_f32: []const u8,
        rows: u32,
        cols: u32,
        output: []f32,
    ) !void {
        if (rows == 0 or cols == 0 or cols % 64 != 0) return error.ShapeMismatch;
        if (output.len < rows) return error.ShapeMismatch;
        const row_bytes: usize = @as(usize, cols) * @sizeOf(f32);
        const weights_bytes = @as(usize, rows) * row_bytes;
        if (weights_f32.len < weights_bytes) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        if (weight_off + weights_bytes > self.input_map.len) return error.InputTooLarge;
        if (@as(usize, rows) * @sizeOf(f32) > self.output_map.len) return error.OutputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);
        @memcpy(self.input_map[weight_off..][0..weights_bytes], weights_f32[0..weights_bytes]);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..rows) |i| output_words[i] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_5000 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_dmmv_f32_row_range;
        const pgm_lo: u32 = @truncate(pgm_va >> 8);
        const pgm_hi: u32 = @truncate(pgm_va >> 40);
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ pgm_lo, pgm_hi });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            compute_pgm_rsrc1_value,
            compute_pgm_rsrc2_argmax_top2_value,
        });
        try self.builder.setShRegOne(packet.sh_reg_pgm_rsrc3, 0);
        try self.builder.setShReg(packet.sh_reg_num_thread_x, &[_]u32{ 1, 1, 1 });
        try self.builder.setShReg(packet.sh_reg_resource_limits, &[_]u32{
            0,
            0xffff_ffff,
            0xffff_ffff,
        });

        const in_lo: u32 = @truncate(self.input_va);
        const in_hi: u32 = @truncate(self.input_va >> 32);
        const out_lo: u32 = @truncate(self.output_va);
        const out_hi: u32 = @truncate(self.output_va >> 32);
        const weight_va = self.input_va + @as(u64, weight_off);
        const weight_lo: u32 = @truncate(weight_va);
        const weight_hi: u32 = @truncate(weight_va >> 32);
        try self.builder.setShReg(packet.compute_user_data_0, &[_]u32{
            in_lo,
            in_hi,
            out_lo,
            out_hi,
            weight_lo,
            weight_hi,
            cols,
            rows,
        });
        try self.builder.dispatchDirectInitiator(1, 1, 1, packet.dispatch_initiator_compute);
        try self.builder.writeData64(self.signal_va, signal_expected);
        try self.builder.padToAlignment(64);
        storeFence();

        var ib_chunk_data: DrmAmdgpuCsChunkIb = .{
            ._pad = 0,
            .flags = AMDGPU_IB_FLAG_EMIT_MEM_SYNC,
            .va_start = self.ib_va,
            .ib_bytes = 0,
            .ip_type = self.ip_type,
            .ip_instance = 0,
            .ring = 0,
        };
        var chunks = [_]DrmAmdgpuCsChunk{.{
            .chunk_id = AMDGPU_CHUNK_ID_IB,
            .length_dw = @sizeOf(DrmAmdgpuCsChunkIb) / @sizeOf(u32),
            .chunk_data = @intFromPtr(&ib_chunk_data),
        }};
        var chunk_ptrs = [_]u64{@intFromPtr(&chunks[0])};
        self.last_fence_handle = try submitBuilderAndWait(
            self.file,
            self.ctx_id,
            self.ip_type,
            self.bo_list_handle,
            &self.builder,
            &ib_chunk_data,
            &chunk_ptrs,
            &self.last_ib_bytes,
            &self.last_wait_status,
        );
        self.submit_count += 1;

        const signal_value = @as(u64, signal_words[0]) | (@as(u64, signal_words[1]) << 32);
        if (signal_value != signal_expected) return error.SignalMismatch;
        for (0..rows) |i| output[i] = @bitCast(output_words[i]);
    }
};

fn orderedF32(value: f32) u32 {
    const bits: u32 = @bitCast(value);
    if ((bits & 0x8000_0000) != 0) return ~bits;
    return bits ^ 0x8000_0000;
}

var last_errno: ?linux.E = null;

/// Errno captured from the most recent `ioctl` issued by this module, or null
/// if the call succeeded. Useful for surfacing a precise reason after a
/// `SmokeResult.status` indicates a kernel-side failure.
/// @returns The latest captured `linux.E` value, or null when there was no error.
pub fn lastErrno() ?linux.E {
    return last_errno;
}

/// Run the bring-up smoke gate against `default_render_node`.
/// @returns A `SmokeResult` summarizing whether the two PM4 submissions retired and the signal sentinel matched.
pub fn setupSmokeDefault() SmokeResult {
    return submitNopSmokePath(default_render_node);
}

/// Run the bring-up smoke gate against the given DRM render node path.
/// @param render_node Absolute path to the amdgpu DRM render node to test.
/// @returns A `SmokeResult` describing the open → submit → wait outcome.
pub fn setupSmokePath(render_node: []const u8) SmokeResult {
    return submitNopSmokePath(render_node);
}

/// Backwards-compatible alias for `setupSmokeDefault` named for the underlying
/// PM4 NOP+WRITE_DATA stream that exercises the CS path.
/// @returns A `SmokeResult` describing the bring-up outcome on `default_render_node`.
pub fn submitNopSmokeDefault(io: std.Io) SmokeResult {
    return submitNopSmokePath(io, default_render_node);
}

/// Full bring-up smoke implementation: open the render node, query compute IP,
/// allocate a context, create GTT-backed IB + signal BOs and map them at fixed
/// low GPU VAs, build a PM4 NOP + `WRITE_DATA` stream, submit it twice through
/// `DRM_IOCTL_AMDGPU_CS`, and verify each fence retires with the expected
/// signal sentinel in the signal BO.
/// @param render_node Absolute path to the amdgpu DRM render node to exercise.
/// @returns A `SmokeResult` whose `status` pinpoints the failure stage, or `.ok` on success.
/// @note Returns `.unsupported_os` immediately on non-Linux hosts; never throws.
pub fn submitNopSmokePath(io: std.Io, render_node: []const u8) SmokeResult {
    if (builtin.os.tag != .linux) return .{ .status = .unsupported_os, .render_node = render_node };

    last_errno = null;
    var result: SmokeResult = .{ .status = .ok, .render_node = render_node };

    var file = std.Io.Dir.openFileAbsolute(io, render_node, .{ .mode = .read_write }) catch {
        result.status = .render_node_open_failed;
        return result;
    };
    defer file.close(io);

    // Use the compute ring: this is the path Vulkan compute and future
    // ZINC_RT decode dispatches target.
    const ip_type: u32 = AMDGPU_HW_IP_COMPUTE;
    const hw_ip = kmd.queryHwIp(file, ip_type) catch {
        result.errno = kmd.lastErrno();
        result.status = .hw_ip_query_failed;
        return result;
    };
    result.ip_type = ip_type;
    result.available_rings = hw_ip.available_rings;
    if (hw_ip.available_rings == 0) {
        result.status = .no_rings;
        return result;
    }

    // Allocate (and, on exit, free) an amdgpu submission context.
    var ctx: DrmAmdgpuCtx = std.mem.zeroes(DrmAmdgpuCtx);
    ctx.in = .{ .op = AMDGPU_CTX_OP_ALLOC_CTX, .flags = 0, .ctx_id = 0, .priority = 0 };
    ioctlRaw(file, ioc_ctx, @intFromPtr(&ctx)) catch {
        result.errno = last_errno;
        result.status = .ctx_alloc_failed;
        return result;
    };
    const ctx_id = ctx.out_alloc.ctx_id;
    result.ctx_id = ctx_id;
    defer {
        var free_ctx: DrmAmdgpuCtx = std.mem.zeroes(DrmAmdgpuCtx);
        free_ctx.in = .{ .op = AMDGPU_CTX_OP_FREE_CTX, .flags = 0, .ctx_id = ctx_id, .priority = 0 };
        ioctlRaw(file, ioc_ctx, @intFromPtr(&free_ctx)) catch {};
    }

    // Reserve a low GPU VA. Earlier bring-up used the CPU mmap address as the
    // GPU VA, which can land outside the amdgpu VM aperture on this kernel and
    // page-fault the CP. This fixed low VA is aligned well beyond the advertised
    // IB start alignment and is private to this short-lived DRM file.
    const ib_bo_size: usize = 64 * 1024;
    const ib_va: u64 = 0x1_0000_0000;
    result.ib_va = ib_va;
    const signal_bo_size: usize = 4096;
    const signal_va: u64 = ib_va + ib_bo_size;
    const signal_expected_1: u64 = 0x5A494E435254_0001; // "ZINCRT\0\1"
    const signal_expected_2: u64 = 0x5A494E435254_0002; // "ZINCRT\0\2"
    result.signal_va = signal_va;

    // IB BO: GTT, CPU write-combined. Create it, map it for the CPU, map it
    // into the GPU VM (read/write/execute), build a PM4 NOP stream into it,
    // and store-fence the write-combined bytes out to DRAM before submission.
    const ib_bo = kmd.createGem(file, ib_bo_size, 256, kmd.AMDGPU_GEM_DOMAIN_GTT, kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC) catch {
        result.errno = kmd.lastErrno();
        result.status = .ib_bo_failed;
        return result;
    };
    const ib_map = kmd.mmapGem(file, ib_bo, std.posix.PROT{ .READ = true, .WRITE = true }) catch {
        result.errno = kmd.lastErrno();
        result.status = .ib_map_failed;
        return result;
    };
    defer std.posix.munmap(ib_map);

    const va_flags = kmd.AMDGPU_VM_PAGE_READABLE | kmd.AMDGPU_VM_PAGE_WRITEABLE | kmd.AMDGPU_VM_PAGE_EXECUTABLE | kmd.AMDGPU_VM_MTYPE_DEFAULT;
    kmd.mapGemVa(file, ib_bo, ib_va, va_flags) catch {
        result.errno = kmd.lastErrno();
        result.status = .ib_va_failed;
        return result;
    };

    // Small GTT signal BO. A PM4 WRITE_DATA packet writes `signal_expected`
    // here, then the CPU verifies the mapped value after WAIT_CS retires.
    const signal_bo = kmd.createGem(file, signal_bo_size, 256, kmd.AMDGPU_GEM_DOMAIN_GTT, kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC) catch {
        result.errno = kmd.lastErrno();
        result.status = .signal_bo_failed;
        return result;
    };
    const signal_map = kmd.mmapGem(file, signal_bo, std.posix.PROT{ .READ = true, .WRITE = true }) catch {
        result.errno = kmd.lastErrno();
        result.status = .signal_map_failed;
        return result;
    };
    defer std.posix.munmap(signal_map);

    const signal_va_flags = kmd.AMDGPU_VM_PAGE_READABLE | kmd.AMDGPU_VM_PAGE_WRITEABLE | kmd.AMDGPU_VM_MTYPE_DEFAULT;
    kmd.mapGemVa(file, signal_bo, signal_va, signal_va_flags) catch {
        result.errno = kmd.lastErrno();
        result.status = .signal_va_failed;
        return result;
    };
    @memset(signal_map[0..@sizeOf(u64)], 0);

    var bo_entries = [_]DrmAmdgpuBoListEntry{
        .{ .bo_handle = ib_bo.handle, .bo_priority = 0 },
        .{ .bo_handle = signal_bo.handle, .bo_priority = 0 },
    };
    var bo_list: DrmAmdgpuBoList = std.mem.zeroes(DrmAmdgpuBoList);
    bo_list.in = .{
        .operation = AMDGPU_BO_LIST_OP_CREATE,
        .list_handle = 0,
        .bo_number = bo_entries.len,
        .bo_info_size = @sizeOf(DrmAmdgpuBoListEntry),
        .bo_info_ptr = @intFromPtr(&bo_entries),
    };
    ioctlRaw(file, ioc_bo_list, @intFromPtr(&bo_list)) catch {
        result.errno = last_errno;
        result.status = .bo_list_failed;
        return result;
    };
    const bo_list_handle = bo_list.out.list_handle;
    result.bo_list_handle = bo_list_handle;
    defer {
        var destroy_list: DrmAmdgpuBoList = std.mem.zeroes(DrmAmdgpuBoList);
        destroy_list.in = .{
            .operation = AMDGPU_BO_LIST_OP_DESTROY,
            .list_handle = bo_list_handle,
            .bo_number = 0,
            .bo_info_size = 0,
            .bo_info_ptr = 0,
        };
        ioctlRaw(file, ioc_bo_list, @intFromPtr(&destroy_list)) catch {};
    }

    const ib_words = @as([*]u32, @ptrCast(@alignCast(ib_map.ptr)))[0 .. ib_bo_size / @sizeOf(u32)];
    var builder = packet.PacketBuilder.init(ib_words);
    var ib_chunk_data: DrmAmdgpuCsChunkIb = .{
        ._pad = 0,
        .flags = AMDGPU_IB_FLAG_EMIT_MEM_SYNC,
        .va_start = ib_va,
        .ib_bytes = 0,
        .ip_type = ip_type,
        .ip_instance = 0,
        .ring = 0,
    };
    var chunks = [_]DrmAmdgpuCsChunk{.{
        .chunk_id = AMDGPU_CHUNK_ID_IB,
        .length_dw = @sizeOf(DrmAmdgpuCsChunkIb) / @sizeOf(u32),
        .chunk_data = @intFromPtr(&ib_chunk_data),
    }};
    var chunk_ptrs = [_]u64{@intFromPtr(&chunks[0])};

    result.first_fence_handle = submitWriteDataAndWait(
        file,
        ctx_id,
        ip_type,
        bo_list_handle,
        &builder,
        &ib_chunk_data,
        &chunk_ptrs,
        signal_map,
        signal_va,
        signal_expected_1,
        &result.ib_bytes,
        &result.wait_status,
    ) catch |err| return submitFail(result, err);
    result.submit_count = 1;

    const first_signal_words: [*]volatile u32 = @ptrCast(@alignCast(signal_map.ptr));
    const first_signal = @as(u64, first_signal_words[0]) | (@as(u64, first_signal_words[1]) << 32);
    if (first_signal != signal_expected_1) {
        result.signal_value = first_signal;
        result.status = .signal_check_failed;
        return result;
    }

    result.fence_handle = submitWriteDataAndWait(
        file,
        ctx_id,
        ip_type,
        bo_list_handle,
        &builder,
        &ib_chunk_data,
        &chunk_ptrs,
        signal_map,
        signal_va,
        signal_expected_2,
        &result.ib_bytes,
        &result.wait_status,
    ) catch |err| return submitFail(result, err);
    result.submit_count = 2;

    const signal_words: [*]volatile u32 = @ptrCast(@alignCast(signal_map.ptr));
    result.signal_value = @as(u64, signal_words[0]) | (@as(u64, signal_words[1]) << 32);
    if (result.signal_value != signal_expected_2) {
        result.status = .signal_check_failed;
        return result;
    }

    result.status = .ok;
    return result;
}

fn submitWriteDataAndWait(
    file: std.Io.File,
    ctx_id: u32,
    ip_type: u32,
    bo_list_handle: u32,
    builder: *packet.PacketBuilder,
    ib_chunk_data: *DrmAmdgpuCsChunkIb,
    chunk_ptrs: []u64,
    signal_map: []align(std.heap.page_size_min) u8,
    signal_va: u64,
    signal_expected: u64,
    ib_bytes_out: *u32,
    wait_status_out: *u64,
) SubmitError!u64 {
    @memset(signal_map[0..@sizeOf(u64)], 0);
    builder.reset();
    try builder.writeNop(3);
    try builder.writeData64(signal_va, signal_expected);
    try builder.writeNop(1);
    try builder.padToAlignment(64);
    ib_chunk_data.ib_bytes = @intCast(builder.written().len * @sizeOf(u32));
    ib_bytes_out.* = ib_chunk_data.ib_bytes;
    storeFence();

    return submitBuilderAndWait(
        file,
        ctx_id,
        ip_type,
        bo_list_handle,
        builder,
        ib_chunk_data,
        chunk_ptrs,
        ib_bytes_out,
        wait_status_out,
    );
}

fn submitBuilderAndWait(
    file: std.Io.File,
    ctx_id: u32,
    ip_type: u32,
    bo_list_handle: u32,
    builder: *const packet.PacketBuilder,
    ib_chunk_data: *DrmAmdgpuCsChunkIb,
    chunk_ptrs: []u64,
    ib_bytes_out: *u32,
    wait_status_out: *u64,
) SubmitError!u64 {
    ib_chunk_data.ib_bytes = @intCast(builder.written().len * @sizeOf(u32));
    ib_bytes_out.* = ib_chunk_data.ib_bytes;

    var submit: DrmAmdgpuCs = std.mem.zeroes(DrmAmdgpuCs);
    submit.in = .{
        .ctx_id = ctx_id,
        .bo_list_handle = bo_list_handle,
        .num_chunks = @intCast(chunk_ptrs.len),
        .flags = 0,
        .chunks = @intFromPtr(chunk_ptrs.ptr),
    };
    ioctlRaw(file, ioc_cs, @intFromPtr(&submit)) catch return error.SubmitFailed;

    var wait: DrmAmdgpuWaitCs = std.mem.zeroes(DrmAmdgpuWaitCs);
    wait.in = .{
        .handle = submit.out.handle,
        .timeout = std.math.maxInt(u64),
        .ip_type = ip_type,
        .ip_instance = 0,
        .ring = 0,
        .ctx_id = ctx_id,
    };
    ioctlRaw(file, ioc_wait_cs, @intFromPtr(&wait)) catch return error.WaitFailed;
    wait_status_out.* = wait.out.status;
    if (wait.out.status != 0) {
        return error.WaitTimedOut;
    }
    return submit.out.handle;
}

const SubmitError = error{ SubmitFailed, WaitFailed, WaitTimedOut, OutOfSpace };

fn submitFail(result: SmokeResult, err: SubmitError) SmokeResult {
    var failed = result;
    failed.errno = last_errno;
    failed.status = switch (err) {
        error.SubmitFailed => .cs_submit_failed,
        error.WaitFailed => .wait_cs_failed,
        error.WaitTimedOut => .wait_timeout,
        error.OutOfSpace => .cs_submit_failed,
    };
    return failed;
}

fn ioctlRaw(file: std.Io.File, request: u32, arg: usize) error{IoctlFailed}!void {
    last_errno = null;
    const rc = linux.ioctl(file.handle, request, arg);
    const err = linux.errno(rc);
    if (err != .SUCCESS) {
        last_errno = err;
        return error.IoctlFailed;
    }
}

fn freeContext(file: std.Io.File, ctx_id: u32) void {
    var free_ctx: DrmAmdgpuCtx = std.mem.zeroes(DrmAmdgpuCtx);
    free_ctx.in = .{ .op = AMDGPU_CTX_OP_FREE_CTX, .flags = 0, .ctx_id = ctx_id, .priority = 0 };
    ioctlRaw(file, ioc_ctx, @intFromPtr(&free_ctx)) catch {};
}

fn destroyBoList(file: std.Io.File, bo_list_handle: u32) void {
    var destroy_list: DrmAmdgpuBoList = std.mem.zeroes(DrmAmdgpuBoList);
    destroy_list.in = .{
        .operation = AMDGPU_BO_LIST_OP_DESTROY,
        .list_handle = bo_list_handle,
        .bo_number = 0,
        .bo_info_size = 0,
        .bo_info_ptr = 0,
    };
    ioctlRaw(file, ioc_bo_list, @intFromPtr(&destroy_list)) catch {};
}

/// Drain write-combined CPU stores so they are globally visible (in DRAM) — on
/// x86 `sfence` evicts the WC buffers; elsewhere a compiler barrier is the best
/// this bring-up gate needs (it only runs meaningfully on the x86 bench node).
inline fn storeFence() void {
    switch (builtin.target.cpu.arch) {
        .x86, .x86_64 => asm volatile ("sfence" ::: .{ .memory = true }),
        else => asm volatile ("" ::: .{ .memory = true }),
    }
}

test "amdgpu cs uapi layout is stable" {
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(DrmAmdgpuCtxIn));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(DrmAmdgpuCtxOutAlloc));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(DrmAmdgpuCtxOutState));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(DrmAmdgpuCtx));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(DrmAmdgpuBoListIn));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(DrmAmdgpuBoListOut));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(DrmAmdgpuBoList));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(DrmAmdgpuBoListEntry));
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(DrmAmdgpuCsChunk));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(DrmAmdgpuCsIn));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(DrmAmdgpuCsOut));
    try std.testing.expectEqual(@as(usize, 24), @sizeOf(DrmAmdgpuCs));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(DrmAmdgpuCsChunkIb));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(DrmAmdgpuWaitCsIn));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(DrmAmdgpuWaitCsOut));
    try std.testing.expectEqual(@as(usize, 32), @sizeOf(DrmAmdgpuWaitCs));
}

test "amdgpu ctx ioctl number matches uapi" {
    // _IOWR('d', DRM_COMMAND_BASE + DRM_AMDGPU_CTX, union drm_amdgpu_ctx)  size 16
    try std.testing.expectEqual(@as(u32, 0xc0106442), ioc_ctx);
    // _IOWR('d', DRM_COMMAND_BASE + DRM_AMDGPU_BO_LIST, union drm_amdgpu_bo_list) size 24
    try std.testing.expectEqual(@as(u32, 0xc0186443), ioc_bo_list);
    // _IOWR('d', DRM_COMMAND_BASE + DRM_AMDGPU_CS, union drm_amdgpu_cs)    size 24
    try std.testing.expectEqual(@as(u32, 0xc0186444), ioc_cs);
    // _IOWR('d', DRM_COMMAND_BASE + DRM_AMDGPU_WAIT_CS, union drm_amdgpu_wait_cs) size 32
    try std.testing.expectEqual(@as(u32, 0xc0206449), ioc_wait_cs);
}

test "setupSmokePath reports unsupported_os off Linux" {
    if (builtin.os.tag == .linux) return error.SkipZigTest;
    const r = setupSmokePath(default_render_node);
    try std.testing.expectEqual(SmokeStatus.unsupported_os, r.status);
    try std.testing.expect(!r.ok());
}
