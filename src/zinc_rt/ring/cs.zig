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
//! indirect-buffer BO plus data/signal/shader BOs, map them into the GPU VM at
//! low VAs, submit PM4 streams through `DRM_IOCTL_AMDGPU_CS` using the same
//! context/BO list, and wait for the returned fences with
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
const compute_pgm_rsrc1_vgpr12_value: u32 = (compute_pgm_rsrc1_value & ~@as(u32, 0x3f)) | 0x2;
const compute_pgm_rsrc1_vgpr16_value: u32 = (compute_pgm_rsrc1_value & ~@as(u32, 0x3f)) | 0x3;
const compute_pgm_rsrc1_vgpr32_value: u32 = (compute_pgm_rsrc1_value & ~@as(u32, 0x3f)) | 0x7;
const compute_pgm_rsrc2_argmax_top2_value: u32 = 0x90; // 8 user SGPRs + workgroup-id-x.
const compute_pgm_rsrc2_user8_vgpr_workitem_x_value: u32 = (8 << 1) | (1 << 11);
// 8 user SGPRs + ENABLE_SGPR_WORKGROUP_ID_X (bit 7 -> ttmp9 on gfx11/12) + VGPR
// workitem id x (bit 11). For multi-workgroup grid dispatches that index output
// rows by workgroup_id_x.
const compute_pgm_rsrc2_user8_wgid_vgpr_x_value: u32 = (8 << 1) | (1 << 7) | (1 << 11);

fn supportsEmbeddedGfx12Kernels(hw_ip: kmd.DrmAmdgpuInfoHwIp) bool {
    return hw_ip.hw_ip_version_major == 12;
}
const shader_offset_argmax_top2: usize = 0x000;
const shader_offset_rms_norm_elem0: usize = 0x100;
const shader_offset_dmmv_f32_row_range: usize = 0x200;
const shader_offset_dmmv_q4_0_row_range: usize = 0x300;
const shader_offset_dmmv_q8_0_row_range: usize = 0x500;
const shader_offset_argmax_u32_range: usize = 0x600;
const shader_offset_dmmv_q4_0_argmax_row_range: usize = 0x700;
const shader_offset_dmmv_q4_0_row_range_parallel: usize = 0x900;
const shader_offset_dmmv_q8_0_row_range_parallel: usize = 0xb00;
const shader_offset_dmmv_q4_0_row_partial64: usize = 0xd00;
// TGID-delivery probe + future resident-weight kernels live in the second 4 KiB
// page; the first page (0x000-0xfff) is full.
const shader_offset_tgid_probe: usize = 0x1000;
const shader_page_bytes: usize = 8192;

// gfx1201 multi-workgroup TGID-delivery probe. Each workgroup stores its
// workgroup_id_x at output[workgroup_id_x]; a correct multi-WG dispatch yields
// [0,1,2,...]. Assembled with llvm-mc -triple=amdgcn-amd-amdhsa -mcpu=gfx1201.
// ABI: s[0:1]=output u32 ptr, s8=workgroup_id_x (rsrc2 must enable SGPR WGID_X
// with 8 preceding user SGPRs).
// Exhaustive gfx1201 TGID-slot probe: each workgroup, for each candidate SGPR
// s8..s19, stores its candidate value at output[candidate*16 + candidate_index]
// (guarded by `< groups`, so garbage candidates are skipped, fault-safe). The
// candidate slot that reads [0,1,2,...] across workgroups is the one carrying
// workgroup_id_x. Writes signature 0x53475052 ("SGPR") at word 960 as a
// kernel-ran marker. ABI: s[0:1]=output, s[2:3]=signal, s4=groups, s5/s6=signal
// value, s8..=candidate SGPRs.
const shader_offset_tgid_dump: usize = 0x1200;
const tgid_dump_gfx1201 = [_]u32{
    0xbf090408, 0xbfa20006, 0x84148608, 0x7e000214, 0x7e020208, 0xee068000, 0x00800000, 0x00000000,
    0xbf090409, 0xbfa20007, 0x84148609, 0x80148414, 0x7e000214, 0x7e020209, 0xee068000, 0x00800000,
    0x00000000, 0xbf09040a, 0xbfa20007, 0x8414860a, 0x80148814, 0x7e000214, 0x7e02020a, 0xee068000,
    0x00800000, 0x00000000, 0xbf09040b, 0xbfa20007, 0x8414860b, 0x80148c14, 0x7e000214, 0x7e02020b,
    0xee068000, 0x00800000, 0x00000000, 0xbf09040c, 0xbfa20007, 0x8414860c, 0x80149014, 0x7e000214,
    0x7e02020c, 0xee068000, 0x00800000, 0x00000000, 0xbf09040d, 0xbfa20007, 0x8414860d, 0x80149414,
    0x7e000214, 0x7e02020d, 0xee068000, 0x00800000, 0x00000000, 0xbf09040e, 0xbfa20007, 0x8414860e,
    0x80149814, 0x7e000214, 0x7e02020e, 0xee068000, 0x00800000, 0x00000000, 0xbf09040f, 0xbfa20007,
    0x8414860f, 0x80149c14, 0x7e000214, 0x7e02020f, 0xee068000, 0x00800000, 0x00000000, 0xbf090410,
    0xbfa20007, 0x84148610, 0x8014a014, 0x7e000214, 0x7e020210, 0xee068000, 0x00800000, 0x00000000,
    0xbf090411, 0xbfa20007, 0x84148611, 0x8014a414, 0x7e000214, 0x7e020211, 0xee068000, 0x00800000,
    0x00000000, 0xbf090412, 0xbfa20007, 0x84148612, 0x8014a814, 0x7e000214, 0x7e020212, 0xee068000,
    0x00800000, 0x00000000, 0xbf090413, 0xbfa20007, 0x84148613, 0x8014ac14, 0x7e000214, 0x7e020213,
    0xee068000, 0x00800000, 0x00000000, 0x7e0002ff, 0x00000f00, 0x7e0202ff, 0x53475052, 0xee068000,
    0x00800000, 0x00000000, 0x7e000280, 0x7e020205, 0xee068002, 0x00800000, 0x00000000, 0x7e000284,
    0x7e020206, 0xee068002, 0x00800000, 0x00000000, 0xbf8903f7, 0xbfb60003, 0xbfb00000,
};

// Grid-over-rows Q4_0 dequant-matvec (src/zinc_rt/isa/gfx1201/dmmv_q4_0_resident_grid.s).
// One wave per workgroup; workgroup_id_x (ttmp9) selects the 64-row block, so a
// grid of ceil(rows/64) workgroups covers all rows in one submit across all CUs.
// ABI: s[0:1]=input, s[2:3]=output, s[4:5]=Q4_0 weights, s6=cols, s7=total_rows.
const shader_offset_dmmv_q4_0_resident_grid: usize = 0x1400;
const dmmv_q4_0_resident_grid_gfx1201 = [_]u32{
    0x84088675, 0x4a100008, 0x7e1e0207, 0x7d921f08, 0xbfa5003b, 0x7e020280, 0x850a8506, 0xd72c0009,
    0x0002100a, 0xd72c0009, 0x00021292, 0xbe8b0080, 0xbf090a0b, 0xbfa2002d, 0x960d920b, 0x4a14120d,
    0xee048004, 0x00000002, 0x0000000a, 0x960ea00b, 0xbe8f0080, 0xbf8903f7, 0x7e041702, 0xbf09900f,
    0xbfa20020, 0x8010820d, 0x80100f10, 0x4a161210, 0xee040004, 0x00000003, 0x0000000b, 0x80110f0e,
    0x84118211, 0x7e180211, 0xee050000, 0x00000006, 0x0000000c, 0x80120f0e, 0x80129012, 0x84128212,
    0x7e1a0212, 0xee050000, 0x00000007, 0x0000000d, 0xbf8903f7, 0x3608068f, 0x320a0684, 0x4a0808c8,
    0x4a0a0ac8, 0x7e080b04, 0x7e0a0b05, 0x10080902, 0x56020d04, 0x100a0b02, 0x56020f05, 0x800f810f,
    0xbfa0ffde, 0x800b810b, 0xbfa0ffd1, 0x301c1082, 0xee068002, 0x00800000, 0x0000000e, 0xbfa00000,
    0xbf800000, 0xbfb60003, 0xbfb00000,
};

const tgid_probe_gfx1201 = [_]u32{
    0x7e020275, // v_mov_b32_e32 v1, ttmp9   (v1 = workgroup_id_x on gfx11/gfx12)
    0x30000282, // v_lshlrev_b32_e32 v0, 2, v1   (v0 = id*4 byte offset)
    0xee068000, 0x00800000, 0x00000000, // global_store_b32 v0, v1, s[0:1]
    0xbf8903f7, // s_waitcnt vmcnt(0)
    0xbfb60003, // s_sendmsg sendmsg(MSG_DEALLOC_VGPRS)
    0xbfb00000, // s_endpgm
};

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

// gfx1201 one-workitem ordered-u32 argmax row-range kernel assembled with:
//   llvm-mc-20 -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj
//
// ABI:
//   s[0:1] = ordered-score u32 input pointer
//   s[2:3] = output pointer: u32 selected_token, u32 ordered_score
//   s4     = rows
//   s5     = start_row
//
// The host maps f32 scores into sortable u32 keys before dispatch. The kernel
// does the row-range argmax and stores the absolute token id.
const argmax_u32_range_gfx1201 = [_]u32{
    0x7e000280,
    0x7e060280,
    0xbe890081,
    0xee050000,
    0x00000001,
    0x00000000,
    0xbf8903f7,
    0xbf090409,
    0xbfa2000c,
    0x840a8209,
    0x7e00020a,
    0xee050000,
    0x00000002,
    0x00000000,
    0xbf8903f7,
    0x7e080209,
    0x7c980302,
    0x02020501,
    0x02060903,
    0x80098109,
    0xbfa0fff2,
    0x4a060605,
    0x7e000280,
    0xee068002,
    0x01800000,
    0x00000000,
    0x7e000284,
    0xee068002,
    0x00800000,
    0x00000000,
    0xbf800000,
    0xbfb60003,
    0xbfb00000,
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

// gfx1201 wave64 Q4_0 DMMV row-range kernel assembled with:
//   llvm-mc-20 -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj
//
// ABI:
//   s[0:1] = input f32 vector pointer
//   s[2:3] = output f32 row-result pointer
//   s[4:5] = Q4_0 weight rows pointer
//   s6     = cols, multiple of 32
//   s7     = rows
//   s8     = workgroup_id_x, unused
//
// A single GPU workitem serially computes a compact Q4_0 row range. This is
// deliberately a correctness-oriented model-slice kernel, not the final
// K-parallel DMMV shape.
const dmmv_q4_0_row_range_gfx1201 = [_]u32{
    0xbe890080,
    0x850a8506,
    0xbf090709,
    0xbfa2003b,
    0x7e020280,
    0x960c0a09,
    0x960c920c,
    0xbe8b0080,
    0xbf090a0b,
    0xbfa2002e,
    0x960d920b,
    0x800d0d0c,
    0x7e00020d,
    0xee048004,
    0x00000002,
    0x00000000,
    0x960ea00b,
    0xbe8f0080,
    0xbf8903f7,
    0x7e041702,
    0xbf09900f,
    0xbfa20020,
    0x8010820d,
    0x80100f10,
    0x7e000210,
    0xee040004,
    0x00000003,
    0x00000000,
    0x80110f0e,
    0x84118211,
    0x7e000211,
    0xee050000,
    0x00000006,
    0x00000000,
    0x80120f0e,
    0x80129012,
    0x84128212,
    0x7e000212,
    0xee050000,
    0x00000007,
    0x00000000,
    0xbf8903f7,
    0x3608068f,
    0x320a0684,
    0x4a0808c8,
    0x4a0a0ac8,
    0x7e080b04,
    0x7e0a0b05,
    0x10080902,
    0x56020d04,
    0x100a0b02,
    0x56020f05,
    0x800f810f,
    0xbfa0ffde,
    0x800b810b,
    0xbfa0ffd0,
    0x84108209,
    0x7e000210,
    0xee068002,
    0x00800000,
    0x00000000,
    0x80098109,
    0xbfa0ffc3,
    0xbf800000,
    0xbfb60003,
    0xbfb00000,
};

// gfx1201 wave64 Q4_0 DMMV row-range + argmax kernel assembled with:
//   llvm-mc-20 -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj
//
// ABI:
//   s[0:1] = input f32 vector pointer
//   s[2:3] = output pointer: u32 local row, f32 score bits
//   s[4:5] = Q4_0 weight rows pointer
//   s6     = cols, multiple of 32
//   s7     = rows
//
// One workitem serially computes a compact row range and keeps the best row in
// registers. This lets the LM-head M1 slice consume a GPU-selected candidate
// without a second direct argmax dispatch over GPU-written logits.
const dmmv_q4_0_argmax_row_range_gfx1201 = [_]u32{
    0xbe890080,
    0x850a8506,
    0x7e1002ff,
    0xff800000,
    0x7e120280,
    0xbf090709,
    0xbfa2003a,
    0x7e020280,
    0x960c0a09,
    0x960c920c,
    0xbe8b0080,
    0xbf090a0b,
    0xbfa2002e,
    0x960d920b,
    0x800d0d0c,
    0x7e00020d,
    0xee048004,
    0x00000002,
    0x00000000,
    0x960ea00b,
    0xbe8f0080,
    0xbf8903f7,
    0x7e041702,
    0xbf09900f,
    0xbfa20020,
    0x8010820d,
    0x80100f10,
    0x7e000210,
    0xee040004,
    0x00000003,
    0x00000000,
    0x80110f0e,
    0x84118211,
    0x7e000211,
    0xee050000,
    0x00000006,
    0x00000000,
    0x80120f0e,
    0x80129012,
    0x84128212,
    0x7e000212,
    0xee050000,
    0x00000007,
    0x00000000,
    0xbf8903f7,
    0x3608068f,
    0x320a0684,
    0x4a0808c8,
    0x4a0a0ac8,
    0x7e080b04,
    0x7e0a0b05,
    0x10080902,
    0x56020d04,
    0x100a0b02,
    0x56020f05,
    0x800f810f,
    0xbfa0ffde,
    0x800b810b,
    0xbfa0ffd0,
    0x7c281101,
    0x7e140209,
    0x02100308,
    0x02121509,
    0x80098109,
    0xbfa0ffc4,
    0x7e000280,
    0xee068002,
    0x04800000,
    0x00000000,
    0x7e000284,
    0xee068002,
    0x04000000,
    0x00000000,
    0xbf800000,
    0xbfb60003,
    0xbfb00000,
};

// gfx1201 wave64 Q4_0 DMMV row-range kernel assembled with:
//   llvm-mc-20 -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj
//
// ABI:
//   s[0:1] = input f32 vector pointer
//   s[2:3] = output f32 row-result pointer
//   s[4:5] = Q4_0 weight rows pointer
//   s6     = cols, multiple of 32
//   s7     = rows, currently must be 64
//   v0     = workitem_id_x, row id
//
// One wave evaluates a 64-row Q4_0 window: each lane owns one row and walks K
// serially. This gives the consumed LM-head window a real row-parallel model
// slice without depending on unvalidated multi-workgroup TGID delivery.
const dmmv_q4_0_row_range_parallel_gfx1201 = [_]u32{
    0xbf06c007,
    0xbfa1003a,
    0x7e100300,
    0x7e020280,
    0x850a8506,
    0x1612100a,
    0x16121292,
    0xbe8b0080,
    0xbf090a0b,
    0xbfa2002d,
    0x960d920b,
    0x4a14120d,
    0xee048004,
    0x00000002,
    0x0000000a,
    0x960ea00b,
    0xbe8f0080,
    0xbf8903f7,
    0x7e041702,
    0xbf09900f,
    0xbfa20020,
    0x8010820d,
    0x80100f10,
    0x4a161210,
    0xee040004,
    0x00000003,
    0x0000000b,
    0x80110f0e,
    0x84118211,
    0x7e180211,
    0xee050000,
    0x00000006,
    0x0000000c,
    0x80120f0e,
    0x80129012,
    0x84128212,
    0x7e1a0212,
    0xee050000,
    0x00000007,
    0x0000000d,
    0xbf8903f7,
    0x3608068f,
    0x320a0684,
    0x4a0808c8,
    0x4a0a0ac8,
    0x7e080b04,
    0x7e0a0b05,
    0x10080902,
    0x56020d04,
    0x100a0b02,
    0x56020f05,
    0x800f810f,
    0xbfa0ffde,
    0x800b810b,
    0xbfa0ffd1,
    0x301c1082,
    0xee068002,
    0x00800000,
    0x0000000e,
    0xbfa00000,
    0xbf800000,
    0xbfb60003,
    0xbfb00000,
};

// gfx1201 wave64 Q8_0 DMMV row-range kernel assembled with:
//   llvm-mc-20 -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj
//
// ABI:
//   s[0:1] = input f32 vector pointer
//   s[2:3] = output f32 row-result pointer
//   s[4:5] = Q8_0 weight rows pointer
//   s6     = cols, multiple of 32
//   s7     = rows
//   s8     = workgroup_id_x, unused
//
// A single GPU workitem serially computes a compact Q8_0 row range. This is a
// correctness-oriented exact source-format model-slice kernel, not the final
// K-parallel DMMV shape.
const dmmv_q8_0_row_range_gfx1201 = [_]u32{
    0xbe890080,
    0x850a8506,
    0xbf090709,
    0xbfa2002d,
    0x7e020280,
    0x960c0a09,
    0x960ca20c,
    0xbe8b0080,
    0xbf090a0b,
    0xbfa20020,
    0x960da20b,
    0x800d0d0c,
    0x7e00020d,
    0xee048004,
    0x00000002,
    0x00000000,
    0x960ea00b,
    0xbe8f0080,
    0xbf8903f7,
    0x7e041702,
    0xbf09a00f,
    0xbfa20012,
    0x8010820d,
    0x80100f10,
    0x7e000210,
    0xee044004,
    0x00000003,
    0x00000000,
    0x80110f0e,
    0x84118211,
    0x7e000211,
    0xee050000,
    0x00000006,
    0x00000000,
    0xbf8903f7,
    0x7e060b03,
    0x10060702,
    0x56020d03,
    0x800f810f,
    0xbfa0ffec,
    0x800b810b,
    0xbfa0ffde,
    0x84108209,
    0x7e000210,
    0xee068002,
    0x00800000,
    0x00000000,
    0x80098109,
    0xbfa0ffd1,
    0xbf800000,
    0xbfb60003,
    0xbfb00000,
};

// gfx1201 wave64 Q8_0 DMMV row-range kernel assembled with:
//   llvm-mc-20 -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj
//
// ABI:
//   s[0:1] = input f32 vector pointer
//   s[2:3] = output f32 row-result pointer
//   s[4:5] = Q8_0 weight rows pointer
//   s6     = cols, multiple of 32
//   s7     = rows, currently must be 64
//   v0     = workitem_id_x, row id
//
// One wave evaluates a 64-row Q8_0 range: each lane owns one row and walks K
// serially. The current forward path uses this for the consumed SSM
// alpha+beta row-range slice (32 alpha rows + 32 beta rows).
const dmmv_q8_0_row_range_parallel_gfx1201 = [_]u32{
    0xbf06c007,
    0xbfa1002c,
    0x7e100300,
    0x7e020280,
    0x850a8506,
    0x1612100a,
    0x161212a2,
    0xbe8b0080,
    0xbf090a0b,
    0xbfa2001f,
    0x960da20b,
    0x4a14120d,
    0xee048004,
    0x00000002,
    0x0000000a,
    0x960ea00b,
    0xbe8f0080,
    0xbf8903f7,
    0x7e041702,
    0xbf09a00f,
    0xbfa20012,
    0x8010820d,
    0x80100f10,
    0x4a161210,
    0xee044004,
    0x00000003,
    0x0000000b,
    0x80110f0e,
    0x84118211,
    0x7e180211,
    0xee050000,
    0x00000006,
    0x0000000c,
    0xbf8903f7,
    0x7e060b03,
    0x10060702,
    0x56020d03,
    0x800f810f,
    0xbfa0ffec,
    0x800b810b,
    0xbfa0ffdf,
    0x301c1082,
    0xee068002,
    0x00800000,
    0x0000000e,
    0xbfa00000,
    0xbf800000,
    0xbfb60003,
    0xbfb00000,
};

// gfx1201 wave64 Q4_0 single-row partial-sum DMMV kernel assembled with:
//   llvm-mc-20 -triple=amdgcn-amd-amdhsa -mcpu=gfx1201 -filetype=obj
//
// ABI:
//   s[0:1] = input f32 vector pointer
//   s[2:3] = output f32 partial sums pointer (64 floats)
//   s[4:5] = one Q4_0 weight row pointer
//   s6     = cols, multiple of 32
//   s7     = unused
//   v0     = workitem_id_x / lane id
//
// Lanes 0..31 each accumulate one element from every Q4_0 block, then the host
// reduces the 64 partials. This is scoped to one row so LM-head selected-row
// validation can consume a K-parallel GPU-produced value without staging a
// 64-row window.
const dmmv_q4_0_row_partial64_gfx1201 = [_]u32{
    0x7e100300,
    0x7e020280,
    0x362a109f,
    0x362c108f,
    0x850a8506,
    0xbe8b0080,
    0xbf090a0b,
    0xbfa2001e,
    0x960d920b,
    0x7e14020d,
    0xee048004,
    0x00000002,
    0x0000000a,
    0x960ea00b,
    0x8411820e,
    0x30182a82,
    0x4a181811,
    0xee050000,
    0x00000006,
    0x0000000c,
    0x8010820d,
    0x4a162c10,
    0xee040004,
    0x00000003,
    0x0000000b,
    0xbf8903f7,
    0x7e041702,
    0x3608068f,
    0x320a0684,
    0x7e3c0290,
    0x7c923d15,
    0x02080905,
    0x4a0808c8,
    0x7e080b04,
    0x10080902,
    0x56020d04,
    0x800b810b,
    0xbfa0ffe0,
    0x7e3c02a0,
    0x7c923d08,
    0x02020280,
    0x301c1082,
    0xee068002,
    0x00800000,
    0x0000000e,
    0xbf800000,
    0xbfb60003,
    0xbfb00000,
};

comptime {
    std.debug.assert(shader_offset_argmax_top2 + argmax_top2_gfx1201.len * @sizeOf(u32) <= shader_offset_rms_norm_elem0);
    std.debug.assert(shader_offset_rms_norm_elem0 + rms_norm_elem0_gfx1201.len * @sizeOf(u32) <= shader_offset_dmmv_f32_row_range);
    std.debug.assert(shader_offset_dmmv_f32_row_range + dmmv_f32_row_range_gfx1201.len * @sizeOf(u32) <= shader_offset_dmmv_q4_0_row_range);
    std.debug.assert(shader_offset_dmmv_q4_0_row_range + dmmv_q4_0_row_range_gfx1201.len * @sizeOf(u32) <= shader_offset_dmmv_q8_0_row_range);
    std.debug.assert(shader_offset_dmmv_q8_0_row_range + dmmv_q8_0_row_range_gfx1201.len * @sizeOf(u32) <= shader_offset_argmax_u32_range);
    std.debug.assert(shader_offset_argmax_u32_range + argmax_u32_range_gfx1201.len * @sizeOf(u32) <= shader_offset_dmmv_q4_0_argmax_row_range);
    std.debug.assert(shader_offset_dmmv_q4_0_argmax_row_range + dmmv_q4_0_argmax_row_range_gfx1201.len * @sizeOf(u32) <= shader_offset_dmmv_q4_0_row_range_parallel);
    std.debug.assert(shader_offset_dmmv_q4_0_row_range_parallel + dmmv_q4_0_row_range_parallel_gfx1201.len * @sizeOf(u32) <= shader_offset_dmmv_q8_0_row_range_parallel);
    std.debug.assert(shader_offset_dmmv_q8_0_row_range_parallel + dmmv_q8_0_row_range_parallel_gfx1201.len * @sizeOf(u32) <= shader_offset_dmmv_q4_0_row_partial64);
    std.debug.assert(shader_offset_dmmv_q4_0_row_partial64 + dmmv_q4_0_row_partial64_gfx1201.len * @sizeOf(u32) <= shader_offset_tgid_probe);
    std.debug.assert(shader_offset_tgid_probe + tgid_probe_gfx1201.len * @sizeOf(u32) <= shader_offset_tgid_dump);
    std.debug.assert(shader_offset_tgid_dump + tgid_dump_gfx1201.len * @sizeOf(u32) <= shader_offset_dmmv_q4_0_resident_grid);
    std.debug.assert(shader_offset_dmmv_q4_0_resident_grid + dmmv_q4_0_resident_grid_gfx1201.len * @sizeOf(u32) <= shader_page_bytes);
}

/// Result produced by the ordered-score argmax row-range kernel.
pub const ArgmaxRangeResult = struct {
    token: u32,
    ordered_score: u32,
};

/// Result produced by a quantized DMMV row-range kernel that performs its own
/// in-kernel argmax over the computed rows.
pub const DmmvArgmaxResult = struct {
    row: u32,
    score: f32,
};

/// In-flight Q4_0 row-range dispatch submitted through `DRM_IOCTL_AMDGPU_CS`.
///
/// Callers can overlap independent CPU work with this fence, then call
/// `finish` before reusing the `TokenBoundary` maps or PM4 builder.
pub const PendingDmmvQ4_0RowRangeParallelChunks = struct {
    boundary: *TokenBoundary,
    fence_handle: u64,
    signal_expected: u64,
    rows: u32,

    /// Wait for the submitted CS fence, validate the signal sentinel, and copy
    /// the GPU-written row scores into `output`.
    pub fn finish(self: PendingDmmvQ4_0RowRangeParallelChunks, output: []f32) !void {
        if (output.len < self.rows) return error.ShapeMismatch;

        const boundary = self.boundary;
        try boundary.waitFence(self.fence_handle);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(boundary.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(boundary.signal_map.ptr));
        const signal_value = @as(u64, signal_words[0]) | (@as(u64, signal_words[1]) << 32);
        if (signal_value != self.signal_expected) return error.SignalMismatch;
        for (0..self.rows) |i| output[i] = @bitCast(output_words[i]);
    }

    /// Retire the fence when the caller must abandon the result after submit.
    pub fn waitDiscard(self: PendingDmmvQ4_0RowRangeParallelChunks) void {
        self.boundary.waitFence(self.fence_handle) catch {};
    }
};

/// Q4_0 row window staged once into the long-lived CS input BO.
///
/// This is still GTT-backed bring-up memory, not the final device-local weight
/// residency model, but it lets the executed M1 LM-head proof stop copying the
/// same weight rows into the dispatch scratch on every run.
pub const ResidentDmmvQ4_0Rows = struct {
    weight_off: usize,
    rows: u32,
    cols: u32,
    row_bytes: usize,
    bytes: usize,
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
    file: std.fs.File,
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
    pub fn initDefault() !TokenBoundary {
        return initPath(default_render_node);
    }

    /// Open the given render node and bring up the full CS submission state.
    ///
    /// Allocates an amdgpu context, creates GTT-backed BOs for the indirect
    /// buffer, input scratch (~2 MiB), output, signal and shader pages, maps
    /// each into a fixed low GPU VA so the kernel does not need to re-bind
    /// them per submission, uploads the gfx1201 PM4 kernels into the
    /// shader page, and creates a persistent BO list referencing all five BOs.
    /// @param render_node Absolute path to the amdgpu DRM render node (e.g. `/dev/dri/renderD128`).
    /// @returns A ready `TokenBoundary` on success; the relevant `error.*Failed` variant otherwise.
    /// @note Linux-only; returns `error.UnsupportedOs` on other platforms.
    pub fn initPath(render_node: []const u8) !TokenBoundary {
        if (builtin.os.tag != .linux) return error.UnsupportedOs;

        var file = std.fs.openFileAbsolute(render_node, .{ .mode = .read_write }) catch return error.RenderNodeOpenFailed;
        errdefer file.close();

        const ip_type: u32 = AMDGPU_HW_IP_COMPUTE;
        const hw_ip = kmd.queryHwIp(file, ip_type) catch return error.HwIpQueryFailed;
        if (hw_ip.available_rings == 0) return error.NoComputeRings;
        if (!supportsEmbeddedGfx12Kernels(hw_ip)) return error.UnsupportedComputeIp;

        var ctx: DrmAmdgpuCtx = std.mem.zeroes(DrmAmdgpuCtx);
        ctx.in = .{ .op = AMDGPU_CTX_OP_ALLOC_CTX, .flags = 0, .ctx_id = 0, .priority = 0 };
        ioctlRaw(file, ioc_ctx, @intFromPtr(&ctx)) catch return error.CtxAllocFailed;
        const ctx_id = ctx.out_alloc.ctx_id;
        errdefer freeContext(file, ctx_id);

        const ib_bo_size: usize = 64 * 1024;
        // The M1 LM-head verifier can now stage the whole default 4096-row
        // Q4_0 cap (about 4.5 MiB at K=2048) so one tracked decode token can
        // consume a fully GPU-produced capped LM-head argmax.
        const input_bo_size: usize = 6 * 1024 * 1024;
        const page_size: usize = 4096;
        const output_bo_size: usize = 32 * 1024;
        const ib_va: u64 = 0x1_0200_0000;
        const input_va: u64 = ib_va + ib_bo_size;
        const output_va: u64 = input_va + input_bo_size;
        const signal_va: u64 = output_va + output_bo_size;
        const shader_va: u64 = signal_va + page_size;

        const ib_bo = kmd.createGem(file, ib_bo_size, 256, kmd.AMDGPU_GEM_DOMAIN_GTT, kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC) catch return error.IbBoFailed;
        const ib_map = kmd.mmapGem(file, ib_bo, std.posix.PROT.READ | std.posix.PROT.WRITE) catch return error.IbMapFailed;
        errdefer std.posix.munmap(ib_map);
        const exec_va_flags = kmd.AMDGPU_VM_PAGE_READABLE | kmd.AMDGPU_VM_PAGE_WRITEABLE | kmd.AMDGPU_VM_PAGE_EXECUTABLE | kmd.AMDGPU_VM_MTYPE_DEFAULT;
        kmd.mapGemVa(file, ib_bo, ib_va, exec_va_flags) catch return error.IbVaFailed;

        const data_va_flags = kmd.AMDGPU_VM_PAGE_READABLE | kmd.AMDGPU_VM_PAGE_WRITEABLE | kmd.AMDGPU_VM_MTYPE_DEFAULT;
        const input_bo = kmd.createGem(file, input_bo_size, 256, kmd.AMDGPU_GEM_DOMAIN_GTT, kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC) catch return error.InputBoFailed;
        const input_map = kmd.mmapGem(file, input_bo, std.posix.PROT.READ | std.posix.PROT.WRITE) catch return error.InputMapFailed;
        errdefer std.posix.munmap(input_map);
        kmd.mapGemVa(file, input_bo, input_va, data_va_flags) catch return error.InputVaFailed;

        const output_bo = kmd.createGem(file, output_bo_size, 256, kmd.AMDGPU_GEM_DOMAIN_GTT, kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC) catch return error.OutputBoFailed;
        const output_map = kmd.mmapGem(file, output_bo, std.posix.PROT.READ | std.posix.PROT.WRITE) catch return error.OutputMapFailed;
        errdefer std.posix.munmap(output_map);
        kmd.mapGemVa(file, output_bo, output_va, data_va_flags) catch return error.OutputVaFailed;

        const signal_bo = kmd.createGem(file, page_size, 256, kmd.AMDGPU_GEM_DOMAIN_GTT, kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC) catch return error.SignalBoFailed;
        const signal_map = kmd.mmapGem(file, signal_bo, std.posix.PROT.READ | std.posix.PROT.WRITE) catch return error.SignalMapFailed;
        errdefer std.posix.munmap(signal_map);
        kmd.mapGemVa(file, signal_bo, signal_va, data_va_flags) catch return error.SignalVaFailed;

        const shader_bo = kmd.createGem(file, shader_page_bytes, 256, kmd.AMDGPU_GEM_DOMAIN_GTT, kmd.AMDGPU_GEM_CREATE_CPU_GTT_USWC) catch return error.ShaderBoFailed;
        const shader_map = kmd.mmapGem(file, shader_bo, std.posix.PROT.READ | std.posix.PROT.WRITE) catch return error.ShaderMapFailed;
        errdefer std.posix.munmap(shader_map);
        kmd.mapGemVa(file, shader_bo, shader_va, exec_va_flags) catch return error.ShaderVaFailed;

        const shader_words = @as([*]u32, @ptrCast(@alignCast(shader_map.ptr)))[0 .. shader_page_bytes / @sizeOf(u32)];
        for (shader_words) |*word| word.* = 0xbfb00000;
        for (argmax_top2_gfx1201, 0..) |word, i| shader_words[shader_offset_argmax_top2 / @sizeOf(u32) + i] = word;
        for (rms_norm_elem0_gfx1201, 0..) |word, i| shader_words[shader_offset_rms_norm_elem0 / @sizeOf(u32) + i] = word;
        for (dmmv_f32_row_range_gfx1201, 0..) |word, i| shader_words[shader_offset_dmmv_f32_row_range / @sizeOf(u32) + i] = word;
        for (dmmv_q4_0_row_range_gfx1201, 0..) |word, i| shader_words[shader_offset_dmmv_q4_0_row_range / @sizeOf(u32) + i] = word;
        for (dmmv_q8_0_row_range_gfx1201, 0..) |word, i| shader_words[shader_offset_dmmv_q8_0_row_range / @sizeOf(u32) + i] = word;
        for (argmax_u32_range_gfx1201, 0..) |word, i| shader_words[shader_offset_argmax_u32_range / @sizeOf(u32) + i] = word;
        for (dmmv_q4_0_argmax_row_range_gfx1201, 0..) |word, i| shader_words[shader_offset_dmmv_q4_0_argmax_row_range / @sizeOf(u32) + i] = word;
        for (dmmv_q4_0_row_range_parallel_gfx1201, 0..) |word, i| shader_words[shader_offset_dmmv_q4_0_row_range_parallel / @sizeOf(u32) + i] = word;
        for (dmmv_q8_0_row_range_parallel_gfx1201, 0..) |word, i| shader_words[shader_offset_dmmv_q8_0_row_range_parallel / @sizeOf(u32) + i] = word;
        for (dmmv_q4_0_row_partial64_gfx1201, 0..) |word, i| shader_words[shader_offset_dmmv_q4_0_row_partial64 / @sizeOf(u32) + i] = word;
        for (tgid_probe_gfx1201, 0..) |word, i| shader_words[shader_offset_tgid_probe / @sizeOf(u32) + i] = word;
        for (tgid_dump_gfx1201, 0..) |word, i| shader_words[shader_offset_tgid_dump / @sizeOf(u32) + i] = word;
        for (dmmv_q4_0_resident_grid_gfx1201, 0..) |word, i| shader_words[shader_offset_dmmv_q4_0_resident_grid / @sizeOf(u32) + i] = word;
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
        self.file.close();
        self.* = undefined;
    }

    fn waitFence(self: *TokenBoundary, fence_handle: u64) SubmitError!void {
        try waitSubmittedFence(
            self.file,
            self.ctx_id,
            self.ip_type,
            fence_handle,
            &self.last_wait_status,
        );
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
    /// Loads the argmax program into the compute SGPRs, packs the output VA,
    /// two ordered scores, and two token ids into `compute_user_data_2..7`, fires
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

    /// Dispatch the gfx1201 ordered-score row-range argmax kernel.
    ///
    /// Converts `scores` into sortable u32 keys, copies them into the shared
    /// input page, then lets the compute ring select the max row. The returned
    /// token id is absolute: `start_row + local_best`.
    /// @param scores F32 logit/score row range to select from.
    /// @param start_row Absolute token row corresponding to `scores[0]`.
    /// @returns The selected absolute token id and the ordered score key the GPU stored.
    pub fn argmaxF32Range(
        self: *TokenBoundary,
        scores: []const f32,
        start_row: u32,
    ) !ArgmaxRangeResult {
        if (scores.len == 0 or scores.len > std.math.maxInt(u32)) return error.ShapeMismatch;
        const rows: u32 = @intCast(scores.len);
        if (@as(usize, rows) * @sizeOf(u32) > self.input_map.len) return error.InputTooLarge;

        const input_words: [*]volatile u32 = @ptrCast(@alignCast(self.input_map.ptr));
        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (scores, 0..) |score, i| input_words[i] = orderedF32(score);
        output_words[0] = 0xffff_ffff;
        output_words[1] = 0;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_9000 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_argmax_u32_range;
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
            rows,
            start_row,
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
        const token = output_words[0];
        if (token < start_row or token >= start_row + rows) return error.ArgmaxRangeInvalidToken;
        return .{
            .token = token,
            .ordered_score = output_words[1],
        };
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

    /// Dispatch one F32 DMMV kernel over two packed row ranges sharing the same input vector.
    ///
    /// This is used by the M1 SSM bridge to consume alpha and beta projection
    /// rows in one CS submission when both tensors are F32. The existing
    /// row-range shader sees one compact `rows_a + rows_b` matrix; output rows
    /// are written in that same order.
    pub fn dmmvF32TwoRowRanges(
        self: *TokenBoundary,
        input: []const f32,
        weights_a_f32: []const u8,
        rows_a: u32,
        weights_b_f32: []const u8,
        rows_b: u32,
        cols: u32,
        output: []f32,
    ) !void {
        if (rows_a == 0 or rows_b == 0 or cols == 0 or cols % 64 != 0) return error.ShapeMismatch;
        const rows = rows_a + rows_b;
        if (rows < rows_a or input.len < cols or output.len < rows) return error.ShapeMismatch;

        const row_bytes: usize = @as(usize, cols) * @sizeOf(f32);
        const weights_a_bytes = @as(usize, rows_a) * row_bytes;
        const weights_b_bytes = @as(usize, rows_b) * row_bytes;
        if (weights_a_f32.len < weights_a_bytes or weights_b_f32.len < weights_b_bytes) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        const weight_b_off = weight_off + weights_a_bytes;
        const weights_bytes = weights_a_bytes + weights_b_bytes;
        if (weight_off + weights_bytes > self.input_map.len) return error.InputTooLarge;
        if (@as(usize, rows) * @sizeOf(f32) > self.output_map.len) return error.OutputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);
        @memcpy(self.input_map[weight_off..][0..weights_a_bytes], weights_a_f32[0..weights_a_bytes]);
        @memcpy(self.input_map[weight_b_off..][0..weights_b_bytes], weights_b_f32[0..weights_b_bytes]);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..rows) |i| output_words[i] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_5100 | @as(u64, self.submit_count + 1);
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

    /// Dispatch the gfx1201 row-range Q4_0 matrix-vector kernel.
    ///
    /// Copies the input vector and raw GGML Q4_0 rows into the shared input page,
    /// records PM4 for one serial workitem over `rows`, and reads back one f32
    /// result per row. This intentionally validates real quantized model bytes
    /// through the native CS path while the full K-parallel DMMV kernel is still
    /// under construction.
    /// @param input Input activation vector of length `cols`.
    /// @param weights_q4_0 Row-major GGML Q4_0 row bytes; must hold at least `rows * (cols/32*18)` bytes.
    /// @param rows Number of output rows to compute.
    /// @param cols Inner dimension; must be a multiple of 32.
    /// @param output Output slice receiving `rows` f32 values.
    pub fn dmmvQ4_0RowRange(
        self: *TokenBoundary,
        input: []const f32,
        weights_q4_0: []const u8,
        rows: u32,
        cols: u32,
        output: []f32,
    ) !void {
        if (rows == 0 or cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        if (input.len < cols or output.len < rows) return error.ShapeMismatch;
        const row_bytes: usize = (@as(usize, cols) / 32) * 18;
        const weights_bytes = @as(usize, rows) * row_bytes;
        if (weights_q4_0.len < weights_bytes) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        if (weight_off + weights_bytes > self.input_map.len) return error.InputTooLarge;
        if (@as(usize, rows) * @sizeOf(f32) > self.output_map.len) return error.OutputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);
        @memcpy(self.input_map[weight_off..][0..weights_bytes], weights_q4_0[0..weights_bytes]);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..rows) |i| output_words[i] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_6000 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_dmmv_q4_0_row_range;
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

    /// Dispatch the wave-lane gfx1201 Q4_0 matrix-vector kernel for exactly 64 rows in parallel.
    ///
    /// Stages the same source-format rows as `dmmvQ4_0RowRange`, but launches
    /// one wave64 workgroup where each lane computes one row. Intended for
    /// 64-row LM-head prefix/window ranges where the GPU row values participate
    /// in choosing the sampled token.
    /// @param input Input activation vector of length `cols`.
    /// @param weights_q4_0 Row-major GGML Q4_0 row bytes; must hold exactly 64 rows.
    /// @param rows Must be exactly 64; any other value returns `error.ShapeMismatch`.
    /// @param cols Inner dimension; must be a multiple of 32.
    /// @param output Output slice receiving 64 f32 values (one per row).
    /// @note Returns `error.SignalMismatch` if the post-fence signal value does not match the expected sentinel.
    pub fn dmmvQ4_0RowRangeParallel(
        self: *TokenBoundary,
        input: []const f32,
        weights_q4_0: []const u8,
        rows: u32,
        cols: u32,
        output: []f32,
    ) !void {
        if (rows != 64 or cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        if (input.len < cols or output.len < rows) return error.ShapeMismatch;
        const row_bytes: usize = (@as(usize, cols) / 32) * 18;
        const weights_bytes = @as(usize, rows) * row_bytes;
        if (weights_q4_0.len < weights_bytes) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        if (weight_off + weights_bytes > self.input_map.len) return error.InputTooLarge;
        if (@as(usize, rows) * @sizeOf(f32) > self.output_map.len) return error.OutputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);
        @memcpy(self.input_map[weight_off..][0..weights_bytes], weights_q4_0[0..weights_bytes]);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..rows) |i| output_words[i] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_B000 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_dmmv_q4_0_row_range_parallel;
        const pgm_lo: u32 = @truncate(pgm_va >> 8);
        const pgm_hi: u32 = @truncate(pgm_va >> 40);
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ pgm_lo, pgm_hi });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            compute_pgm_rsrc1_vgpr16_value,
            compute_pgm_rsrc2_user8_vgpr_workitem_x_value,
        });
        try self.builder.setShRegOne(packet.sh_reg_pgm_rsrc3, 0);
        try self.builder.setShReg(packet.sh_reg_num_thread_x, &[_]u32{ 64, 1, 1 });
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
        try self.builder.releaseMemSignal(self.signal_va, signal_expected);
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

    /// Dispatch the multi-workgroup TGID-delivery probe: `groups` workgroups,
    /// one workitem each, each storing its `workgroup_id_x` (SGPR s8) at
    /// `out[workgroup_id_x]`. A correct dispatch yields `out[0..groups] =
    /// [0,1,...,groups-1]`; if TGID is not delivered, every workgroup collides
    /// at `out[0]`. Validates the grid-over-rows premise of the resident DMMV.
    pub fn sgprTgidProbe(self: *TokenBoundary, groups: u32, out: []u32) !void {
        if (groups == 0 or out.len < groups) return error.ShapeMismatch;
        if (@as(usize, groups) * @sizeOf(u32) > self.output_map.len) return error.OutputTooLarge;

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..groups) |i| output_words[i] = 0xffff_ffff;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_B000 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_tgid_probe;
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ @truncate(pgm_va >> 8), @truncate(pgm_va >> 40) });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            compute_pgm_rsrc1_value,
            compute_pgm_rsrc2_argmax_top2_value, // 8 user SGPRs + ENABLE_SGPR_WORKGROUP_ID_X
        });
        try self.builder.setShRegOne(packet.sh_reg_pgm_rsrc3, 0);
        try self.builder.setShReg(packet.sh_reg_num_thread_x, &[_]u32{ 1, 1, 1 });
        try self.builder.setShReg(packet.sh_reg_resource_limits, &[_]u32{ 0, 0xffff_ffff, 0xffff_ffff });

        const out_lo: u32 = @truncate(self.output_va);
        const out_hi: u32 = @truncate(self.output_va >> 32);
        try self.builder.setShReg(packet.compute_user_data_0, &[_]u32{ out_lo, out_hi, 0, 0, 0, 0, 0, 0 });
        try self.builder.dispatchDirectInitiator(groups, 1, 1, packet.dispatch_initiator_compute);
        try self.builder.releaseMemSignal(self.signal_va, signal_expected);
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
        for (0..groups) |i| out[i] = output_words[i];
    }

    /// Dispatch the exhaustive TGID-slot probe (`tgid_dump`): `groups`
    /// workgroups scan candidate SGPRs s8..s19 and store each candidate value
    /// via a value-as-index trick. Fills `out[0..out.len]` from the output BO;
    /// the caller decodes which candidate slot reads [0,1,...] across
    /// workgroups. `out[960]` reads 0x53475052 ("SGPR") iff the kernel ran.
    pub fn sgprTgidDump(self: *TokenBoundary, groups: u32, out: []u32) !void {
        if (groups == 0) return error.ShapeMismatch;
        if (out.len * @sizeOf(u32) > self.output_map.len) return error.OutputTooLarge;
        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        const zero_words = @min(self.output_map.len / @sizeOf(u32), @as(usize, 1024));
        for (0..zero_words) |i| output_words[i] = 0xffff_ffff;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_B000 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_tgid_dump;
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ @truncate(pgm_va >> 8), @truncate(pgm_va >> 40) });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            compute_pgm_rsrc1_value,
            compute_pgm_rsrc2_argmax_top2_value, // 8 user SGPRs + ENABLE_SGPR_WORKGROUP_ID_X
        });
        try self.builder.setShRegOne(packet.sh_reg_pgm_rsrc3, 0);
        try self.builder.setShReg(packet.sh_reg_num_thread_x, &[_]u32{ 1, 1, 1 });
        try self.builder.setShReg(packet.sh_reg_resource_limits, &[_]u32{ 0, 0xffff_ffff, 0xffff_ffff });

        const out_lo: u32 = @truncate(self.output_va);
        const out_hi: u32 = @truncate(self.output_va >> 32);
        const sig_lo: u32 = @truncate(self.signal_va);
        const sig_hi: u32 = @truncate(self.signal_va >> 32);
        try self.builder.setShReg(packet.compute_user_data_0, &[_]u32{
            out_lo,                           out_hi,
            sig_lo,                           sig_hi,
            groups,                           @truncate(signal_expected),
            @truncate(signal_expected >> 32), 0,
        });
        try self.builder.dispatchDirectInitiator(groups, 1, 1, packet.dispatch_initiator_compute);
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
        for (0..out.len) |i| out[i] = output_words[i];
    }

    /// Grid-over-rows Q4_0 dequant-matvec. Stages `input` + `weights_q4_0` into
    /// the input BO, dispatches ceil(rows/64) workgroups (each picks its 64-row
    /// block from workgroup_id_x/ttmp9), and reads back `rows` results in ONE
    /// submit. This is the correctness path (weights staged per call); the perf
    /// path points s[4:5] at a previously staged resident row window.
    pub fn dmmvQ4_0ResidentGrid(
        self: *TokenBoundary,
        input: []const f32,
        weights_q4_0: []const u8,
        rows: u32,
        cols: u32,
        output: []f32,
    ) !void {
        const pending = try self.beginDmmvQ4_0ResidentGrid(input, weights_q4_0, rows, cols);
        try pending.finish(output);
    }

    /// Submit the staged-weight grid-over-rows Q4_0 DMMV and return before the
    /// fence wait. Callers can overlap CPU tail work with the resident-grid
    /// dispatch, then finish through the returned pending handle.
    pub fn beginDmmvQ4_0ResidentGrid(
        self: *TokenBoundary,
        input: []const f32,
        weights_q4_0: []const u8,
        rows: u32,
        cols: u32,
    ) !PendingDmmvQ4_0RowRangeParallelChunks {
        if (rows == 0 or cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        if (input.len < cols) return error.ShapeMismatch;
        const row_bytes: usize = (@as(usize, cols) / 32) * 18;
        const weights_bytes = @as(usize, rows) * row_bytes;
        if (weights_q4_0.len < weights_bytes) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        if (weight_off + weights_bytes > self.input_map.len) return error.InputTooLarge;

        @memcpy(self.input_map[weight_off..][0..weights_bytes], weights_q4_0[0..weights_bytes]);
        return self.beginDmmvQ4_0ResidentGridAtWeightVa(
            input,
            rows,
            cols,
            row_bytes,
            self.input_va + @as(u64, weight_off),
            weight_off,
        );
    }

    /// Submit the resident-weight grid-over-rows Q4_0 DMMV. Only the activation
    /// vector is copied; weights are read from the long-lived resident region.
    pub fn beginDmmvQ4_0ResidentGridResident(
        self: *TokenBoundary,
        input: []const f32,
        resident: ResidentDmmvQ4_0Rows,
    ) !PendingDmmvQ4_0RowRangeParallelChunks {
        if (resident.weight_off + resident.bytes > self.input_map.len) return error.InputTooLarge;
        return self.beginDmmvQ4_0ResidentGridAtWeightVa(
            input,
            resident.rows,
            resident.cols,
            resident.row_bytes,
            self.input_va + @as(u64, resident.weight_off),
            resident.weight_off,
        );
    }

    fn beginDmmvQ4_0ResidentGridAtWeightVa(
        self: *TokenBoundary,
        input: []const f32,
        rows: u32,
        cols: u32,
        row_bytes: usize,
        weight_va: u64,
        scratch_weight_off: usize,
    ) !PendingDmmvQ4_0RowRangeParallelChunks {
        if (rows == 0 or cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        if (input.len < cols) return error.ShapeMismatch;
        if (row_bytes == 0) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        if (input_bytes.len > scratch_weight_off) return error.InputTooLarge;
        if (@as(usize, rows) * @sizeOf(f32) > self.output_map.len) return error.OutputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..rows) |i| output_words[i] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_B000 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_dmmv_q4_0_resident_grid;
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ @truncate(pgm_va >> 8), @truncate(pgm_va >> 40) });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            compute_pgm_rsrc1_vgpr16_value,
            compute_pgm_rsrc2_user8_wgid_vgpr_x_value,
        });
        try self.builder.setShRegOne(packet.sh_reg_pgm_rsrc3, 0);
        try self.builder.setShReg(packet.sh_reg_num_thread_x, &[_]u32{ 64, 1, 1 });
        try self.builder.setShReg(packet.sh_reg_resource_limits, &[_]u32{ 0, 0xffff_ffff, 0xffff_ffff });

        const in_lo: u32 = @truncate(self.input_va);
        const in_hi: u32 = @truncate(self.input_va >> 32);
        const out_lo: u32 = @truncate(self.output_va);
        const out_hi: u32 = @truncate(self.output_va >> 32);
        const weight_lo: u32 = @truncate(weight_va);
        const weight_hi: u32 = @truncate(weight_va >> 32);
        try self.builder.setShReg(packet.compute_user_data_0, &[_]u32{
            in_lo, in_hi, out_lo, out_hi, weight_lo, weight_hi, cols, rows,
        });
        const groups: u32 = (rows + 63) / 64;
        try self.builder.dispatchDirectInitiator(groups, 1, 1, packet.dispatch_initiator_compute);
        try self.builder.releaseMemSignal(self.signal_va, signal_expected);
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
        self.last_fence_handle = try submitBuilder(
            self.file,
            self.ctx_id,
            self.bo_list_handle,
            &self.builder,
            &ib_chunk_data,
            &chunk_ptrs,
            &self.last_ib_bytes,
        );
        self.submit_count += 1;

        return .{
            .boundary = self,
            .fence_handle = self.last_fence_handle,
            .signal_expected = signal_expected,
            .rows = rows,
        };
    }

    /// One-shot check that a `size`-byte VRAM BO is CPU-mappable (large-BAR) and
    /// round-trips a host write/read. Gates the resident-weight LM-head matvec,
    /// which only beats the CPU if the weight lives in VRAM (576 GB/s) rather
    /// than GTT (PCIe-bound). Leaks the probe BO (reclaimed at process exit).
    pub fn probeVramMappable(self: *TokenBoundary, size: usize) !bool {
        const bo = kmd.createGem(
            self.file,
            size,
            256,
            kmd.AMDGPU_GEM_DOMAIN_VRAM,
            kmd.AMDGPU_GEM_CREATE_VRAM_CLEARED | kmd.AMDGPU_GEM_CREATE_CPU_ACCESS_REQUIRED,
        ) catch return false;
        const map = kmd.mmapGem(self.file, bo, std.posix.PROT.READ | std.posix.PROT.WRITE) catch return false;
        defer std.posix.munmap(map);
        const words: [*]volatile u32 = @ptrCast(@alignCast(map.ptr));
        const last = size / @sizeOf(u32) - 1;
        words[0] = 0xdead_beef;
        words[last] = 0xcafe_babe;
        storeFence();
        return words[0] == 0xdead_beef and words[last] == 0xcafe_babe;
    }

    /// Dispatch the wave64 Q4_0 single-row partial-sum kernel and reduce it on the host.
    ///
    /// This lowers one real model row through a K-parallel CS kernel: lanes
    /// 0..31 each accumulate one column per Q4_0 block, the kernel writes 64
    /// partial sums, and the host performs the final scalar reduction. The
    /// caller may pass `partials_out` to inspect the raw lane outputs.
    /// @param input Input activation vector of length `cols`.
    /// @param row_q4_0 Raw GGML Q4_0 bytes for one row; must hold `(cols/32)*18` bytes.
    /// @param cols Inner dimension; must be a multiple of 32.
    /// @param partials_out Optional output slice receiving 64 f32 partial sums.
    /// @returns Reduced dot-product score.
    pub fn dmmvQ4_0RowPartial64(
        self: *TokenBoundary,
        input: []const f32,
        row_q4_0: []const u8,
        cols: u32,
        partials_out: ?[]f32,
    ) !f32 {
        if (cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        if (input.len < cols) return error.ShapeMismatch;
        if (partials_out) |partials| {
            if (partials.len < 64) return error.ShapeMismatch;
        }
        const row_bytes: usize = (@as(usize, cols) / 32) * 18;
        if (row_q4_0.len < row_bytes) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        if (weight_off + row_bytes > self.input_map.len) return error.InputTooLarge;
        if (64 * @sizeOf(f32) > self.output_map.len) return error.OutputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);
        @memcpy(self.input_map[weight_off..][0..row_bytes], row_q4_0[0..row_bytes]);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..64) |i| output_words[i] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_C000 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_dmmv_q4_0_row_partial64;
        const pgm_lo: u32 = @truncate(pgm_va >> 8);
        const pgm_hi: u32 = @truncate(pgm_va >> 40);
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ pgm_lo, pgm_hi });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            compute_pgm_rsrc1_vgpr32_value,
            compute_pgm_rsrc2_user8_vgpr_workitem_x_value,
        });
        try self.builder.setShRegOne(packet.sh_reg_pgm_rsrc3, 0);
        try self.builder.setShReg(packet.sh_reg_num_thread_x, &[_]u32{ 64, 1, 1 });
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
            0,
        });
        try self.builder.dispatchDirectInitiator(1, 1, 1, packet.dispatch_initiator_compute);
        try self.builder.releaseMemSignal(self.signal_va, signal_expected);
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

        var sum: f32 = 0.0;
        for (0..64) |i| {
            const partial: f32 = @bitCast(output_words[i]);
            if (!std.math.isFinite(partial)) return error.NonFiniteOutput;
            if (partials_out) |partials| partials[i] = partial;
            sum += partial;
        }
        return sum;
    }

    /// Dispatch one or more 64-row wave-lane Q4_0 DMMV chunks in a single CS submission.
    ///
    /// Each chunk runs the same source-format row-parallel shader as
    /// `dmmvQ4_0RowRangeParallel`; this helper stages a larger adjacent row
    /// window, emits one dispatch per 64-row chunk into the same IB, and waits
    /// once after the final release fence. The output slice receives all rows
    /// in order.
    /// @param input Input activation vector of length `cols`.
    /// @param weights_q4_0 Row-major GGML Q4_0 row bytes; must hold `rows` rows.
    /// @param rows Number of rows to compute; must be a positive multiple of 64.
    /// @param cols Inner dimension; must be a multiple of 32.
    /// @param output Output slice receiving `rows` f32 values.
    pub fn dmmvQ4_0RowRangeParallelChunks(
        self: *TokenBoundary,
        input: []const f32,
        weights_q4_0: []const u8,
        rows: u32,
        cols: u32,
        output: []f32,
    ) !void {
        const pending = try self.beginDmmvQ4_0RowRangeParallelChunks(input, weights_q4_0, rows, cols);
        try pending.finish(output);
    }

    /// Stage a Q4_0 row window into the high end of the long-lived input BO.
    ///
    /// The returned handle is valid until another caller overwrites the same
    /// resident region. Default M1 generation uses it for the first LM-head
    /// prefix proof, before any broad validation slices can reuse the scratch.
    pub fn stageDmmvQ4_0RowsResident(
        self: *TokenBoundary,
        weights_q4_0: []const u8,
        rows: u32,
        cols: u32,
    ) !ResidentDmmvQ4_0Rows {
        if (rows == 0 or rows % 64 != 0 or cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        const row_bytes: usize = (@as(usize, cols) / 32) * 18;
        const weights_bytes = @as(usize, rows) * row_bytes;
        if (weights_q4_0.len < weights_bytes) return error.ShapeMismatch;
        if (weights_bytes > self.input_map.len) return error.InputTooLarge;

        const unaligned_off = self.input_map.len - weights_bytes;
        const weight_off = unaligned_off - (unaligned_off % 64);
        if (weight_off == 0) return error.InputTooLarge;

        @memcpy(self.input_map[weight_off..][0..weights_bytes], weights_q4_0[0..weights_bytes]);
        storeFence();
        return .{
            .weight_off = weight_off,
            .rows = rows,
            .cols = cols,
            .row_bytes = row_bytes,
            .bytes = weights_bytes,
        };
    }

    /// Dispatch a previously staged resident Q4_0 row window.
    ///
    /// Only the activation vector is copied for this submission; PM4 points the
    /// shader at the resident weight VA in the shared input BO.
    pub fn beginDmmvQ4_0RowRangeParallelChunksResident(
        self: *TokenBoundary,
        input: []const f32,
        resident: ResidentDmmvQ4_0Rows,
    ) !PendingDmmvQ4_0RowRangeParallelChunks {
        if (resident.weight_off + resident.bytes > self.input_map.len) return error.InputTooLarge;
        return self.beginDmmvQ4_0RowRangeParallelChunksAtWeightVa(
            input,
            resident.rows,
            resident.cols,
            resident.row_bytes,
            self.input_va + @as(u64, resident.weight_off),
            resident.weight_off,
        );
    }

    /// Submit one or more 64-row wave-lane Q4_0 chunks and return before
    /// waiting for the CS fence. The shared input/output/signal maps and PM4
    /// builder must not be reused until the returned dispatch is finished.
    pub fn beginDmmvQ4_0RowRangeParallelChunks(
        self: *TokenBoundary,
        input: []const f32,
        weights_q4_0: []const u8,
        rows: u32,
        cols: u32,
    ) !PendingDmmvQ4_0RowRangeParallelChunks {
        if (rows == 0 or rows % 64 != 0 or cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        if (input.len < cols) return error.ShapeMismatch;
        const row_bytes: usize = (@as(usize, cols) / 32) * 18;
        const weights_bytes = @as(usize, rows) * row_bytes;
        if (weights_q4_0.len < weights_bytes) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        if (weight_off + weights_bytes > self.input_map.len) return error.InputTooLarge;

        @memcpy(self.input_map[weight_off..][0..weights_bytes], weights_q4_0[0..weights_bytes]);
        return self.beginDmmvQ4_0RowRangeParallelChunksAtWeightVa(
            input,
            rows,
            cols,
            row_bytes,
            self.input_va + @as(u64, weight_off),
            weight_off,
        );
    }

    fn beginDmmvQ4_0RowRangeParallelChunksAtWeightVa(
        self: *TokenBoundary,
        input: []const f32,
        rows: u32,
        cols: u32,
        row_bytes: usize,
        weight_va: u64,
        scratch_weight_off: usize,
    ) !PendingDmmvQ4_0RowRangeParallelChunks {
        if (rows == 0 or rows % 64 != 0 or cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        if (input.len < cols) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        if (input_bytes.len > scratch_weight_off) return error.InputTooLarge;
        if (@as(usize, rows) * @sizeOf(f32) > self.output_map.len) return error.OutputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..rows) |i| output_words[i] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_B100 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_dmmv_q4_0_row_range_parallel;
        const pgm_lo: u32 = @truncate(pgm_va >> 8);
        const pgm_hi: u32 = @truncate(pgm_va >> 40);
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ pgm_lo, pgm_hi });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            compute_pgm_rsrc1_vgpr16_value,
            compute_pgm_rsrc2_user8_vgpr_workitem_x_value,
        });
        try self.builder.setShRegOne(packet.sh_reg_pgm_rsrc3, 0);
        try self.builder.setShReg(packet.sh_reg_num_thread_x, &[_]u32{ 64, 1, 1 });
        try self.builder.setShReg(packet.sh_reg_resource_limits, &[_]u32{
            0,
            0xffff_ffff,
            0xffff_ffff,
        });

        const in_lo: u32 = @truncate(self.input_va);
        const in_hi: u32 = @truncate(self.input_va >> 32);
        var row_start: u32 = 0;
        while (row_start < rows) : (row_start += 64) {
            const out_va = self.output_va + @as(u64, row_start) * @sizeOf(f32);
            const chunk_weight_va = weight_va + @as(u64, row_start) * @as(u64, row_bytes);
            try self.builder.setShReg(packet.compute_user_data_0, &[_]u32{
                in_lo,
                in_hi,
                @truncate(out_va),
                @truncate(out_va >> 32),
                @truncate(chunk_weight_va),
                @truncate(chunk_weight_va >> 32),
                cols,
                64,
            });
            try self.builder.dispatchDirectInitiator(1, 1, 1, packet.dispatch_initiator_compute);
        }
        try self.builder.releaseMemSignal(self.signal_va, signal_expected);
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
        self.last_fence_handle = try submitBuilder(
            self.file,
            self.ctx_id,
            self.bo_list_handle,
            &self.builder,
            &ib_chunk_data,
            &chunk_ptrs,
            &self.last_ib_bytes,
        );
        self.submit_count += 1;

        return .{
            .boundary = self,
            .fence_handle = self.last_fence_handle,
            .signal_expected = signal_expected,
            .rows = rows,
        };
    }

    /// Dispatch two 64-row Q4_0 DMMV ranges that share one input vector in one CS submission.
    ///
    /// The two ranges are staged back-to-back, then the existing 64-lane Q4_0
    /// row-parallel kernel is dispatched twice in the same IB. The output slice
    /// receives A's 64 rows first and B's 64 rows second. This is used by the
    /// M1 forward bridge to consume a wider routed MoE gate/up slice without
    /// adding a second fence wait.
    /// @param input Input activation vector of length `cols`.
    /// @param weights_a_q4_0 Row-major GGML Q4_0 bytes for range A; must hold exactly 64 rows.
    /// @param weights_b_q4_0 Row-major GGML Q4_0 bytes for range B; must hold exactly 64 rows.
    /// @param cols Inner dimension; must be a multiple of 32.
    /// @param output Output slice receiving 128 f32 values: A rows first, then B rows.
    pub fn dmmvQ4_0TwoRowRangesParallel64(
        self: *TokenBoundary,
        input: []const f32,
        weights_a_q4_0: []const u8,
        weights_b_q4_0: []const u8,
        cols: u32,
        output: []f32,
    ) !void {
        const rows_per_range: u32 = 64;
        const rows: u32 = rows_per_range * 2;
        if (cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        if (input.len < cols or output.len < rows) return error.ShapeMismatch;

        const row_bytes: usize = (@as(usize, cols) / 32) * 18;
        const range_bytes = @as(usize, rows_per_range) * row_bytes;
        const weights_bytes = range_bytes * 2;
        if (weights_a_q4_0.len < range_bytes or weights_b_q4_0.len < range_bytes) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        const weight_b_off = weight_off + range_bytes;
        if (weight_off + weights_bytes > self.input_map.len) return error.InputTooLarge;
        if (@as(usize, rows) * @sizeOf(f32) > self.output_map.len) return error.OutputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);
        @memcpy(self.input_map[weight_off..][0..range_bytes], weights_a_q4_0[0..range_bytes]);
        @memcpy(self.input_map[weight_b_off..][0..range_bytes], weights_b_q4_0[0..range_bytes]);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..rows) |i| output_words[i] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_B200 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_dmmv_q4_0_row_range_parallel;
        const pgm_lo: u32 = @truncate(pgm_va >> 8);
        const pgm_hi: u32 = @truncate(pgm_va >> 40);
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ pgm_lo, pgm_hi });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            compute_pgm_rsrc1_vgpr16_value,
            compute_pgm_rsrc2_user8_vgpr_workitem_x_value,
        });
        try self.builder.setShRegOne(packet.sh_reg_pgm_rsrc3, 0);
        try self.builder.setShReg(packet.sh_reg_num_thread_x, &[_]u32{ 64, 1, 1 });
        try self.builder.setShReg(packet.sh_reg_resource_limits, &[_]u32{
            0,
            0xffff_ffff,
            0xffff_ffff,
        });

        const in_lo: u32 = @truncate(self.input_va);
        const in_hi: u32 = @truncate(self.input_va >> 32);
        const weight_va = self.input_va + @as(u64, weight_off);
        var row_start: u32 = 0;
        while (row_start < rows) : (row_start += rows_per_range) {
            const out_va = self.output_va + @as(u64, row_start) * @sizeOf(f32);
            const chunk_weight_va = weight_va + @as(u64, row_start) * @as(u64, row_bytes);
            try self.builder.setShReg(packet.compute_user_data_0, &[_]u32{
                in_lo,
                in_hi,
                @truncate(out_va),
                @truncate(out_va >> 32),
                @truncate(chunk_weight_va),
                @truncate(chunk_weight_va >> 32),
                cols,
                rows_per_range,
            });
            try self.builder.dispatchDirectInitiator(1, 1, 1, packet.dispatch_initiator_compute);
        }
        try self.builder.releaseMemSignal(self.signal_va, signal_expected);
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

    /// Dispatch four 64-row Q4_0 DMMV ranges that share one input vector in one CS submission.
    ///
    /// This is the batched companion to `dmmvQ4_0TwoRowRangesParallel64` for
    /// decode-phase MoE gate/up validation across two routed experts. The
    /// output slice receives A, B, C, D ranges in order.
    pub fn dmmvQ4_0FourRowRangesParallel64(
        self: *TokenBoundary,
        input: []const f32,
        weights_a_q4_0: []const u8,
        weights_b_q4_0: []const u8,
        weights_c_q4_0: []const u8,
        weights_d_q4_0: []const u8,
        cols: u32,
        output: []f32,
    ) !void {
        const rows_per_range: u32 = 64;
        const rows: u32 = rows_per_range * 4;
        if (cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        if (input.len < cols or output.len < rows) return error.ShapeMismatch;

        const row_bytes: usize = (@as(usize, cols) / 32) * 18;
        const range_bytes = @as(usize, rows_per_range) * row_bytes;
        const weights_bytes = range_bytes * 4;
        if (weights_a_q4_0.len < range_bytes or
            weights_b_q4_0.len < range_bytes or
            weights_c_q4_0.len < range_bytes or
            weights_d_q4_0.len < range_bytes)
        {
            return error.ShapeMismatch;
        }

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        if (weight_off + weights_bytes > self.input_map.len) return error.InputTooLarge;
        if (@as(usize, rows) * @sizeOf(f32) > self.output_map.len) return error.OutputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);
        @memcpy(self.input_map[weight_off..][0..range_bytes], weights_a_q4_0[0..range_bytes]);
        @memcpy(self.input_map[weight_off + range_bytes ..][0..range_bytes], weights_b_q4_0[0..range_bytes]);
        @memcpy(self.input_map[weight_off + range_bytes * 2 ..][0..range_bytes], weights_c_q4_0[0..range_bytes]);
        @memcpy(self.input_map[weight_off + range_bytes * 3 ..][0..range_bytes], weights_d_q4_0[0..range_bytes]);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..rows) |i| output_words[i] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_B300 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_dmmv_q4_0_row_range_parallel;
        const pgm_lo: u32 = @truncate(pgm_va >> 8);
        const pgm_hi: u32 = @truncate(pgm_va >> 40);
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ pgm_lo, pgm_hi });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            compute_pgm_rsrc1_vgpr16_value,
            compute_pgm_rsrc2_user8_vgpr_workitem_x_value,
        });
        try self.builder.setShRegOne(packet.sh_reg_pgm_rsrc3, 0);
        try self.builder.setShReg(packet.sh_reg_num_thread_x, &[_]u32{ 64, 1, 1 });
        try self.builder.setShReg(packet.sh_reg_resource_limits, &[_]u32{
            0,
            0xffff_ffff,
            0xffff_ffff,
        });

        const in_lo: u32 = @truncate(self.input_va);
        const in_hi: u32 = @truncate(self.input_va >> 32);
        const weight_va = self.input_va + @as(u64, weight_off);
        var row_start: u32 = 0;
        while (row_start < rows) : (row_start += rows_per_range) {
            const out_va = self.output_va + @as(u64, row_start) * @sizeOf(f32);
            const chunk_weight_va = weight_va + @as(u64, row_start) * @as(u64, row_bytes);
            try self.builder.setShReg(packet.compute_user_data_0, &[_]u32{
                in_lo,
                in_hi,
                @truncate(out_va),
                @truncate(out_va >> 32),
                @truncate(chunk_weight_va),
                @truncate(chunk_weight_va >> 32),
                cols,
                rows_per_range,
            });
            try self.builder.dispatchDirectInitiator(1, 1, 1, packet.dispatch_initiator_compute);
        }
        try self.builder.releaseMemSignal(self.signal_va, signal_expected);
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

    /// Dispatch Q4_0 DMMV for two arbitrary model rows staged back-to-back.
    ///
    /// The caller supplies two individual source-format rows, which are packed
    /// into the shared staging page as a compact two-row matrix. This lets the
    /// current forward path obtain both LM-head top-2 scores from one real
    /// DMMV row-range submission even when the rows are not adjacent in vocab.
    /// @param input Input activation vector of length `cols`.
    /// @param row_a_q4_0 Raw GGML Q4_0 bytes for the first row; must hold at least `(cols/32)*18` bytes.
    /// @param row_b_q4_0 Raw GGML Q4_0 bytes for the second row; same size requirement as `row_a_q4_0`.
    /// @param cols Inner dimension; must be a multiple of 32.
    /// @param output Output slice receiving 2 f32 values: `output[0]` for row A, `output[1]` for row B.
    /// @note Returns `error.SignalMismatch` if the post-fence signal sentinel does not match.
    pub fn dmmvQ4_0TwoRows(
        self: *TokenBoundary,
        input: []const f32,
        row_a_q4_0: []const u8,
        row_b_q4_0: []const u8,
        cols: u32,
        output: []f32,
    ) !void {
        if (cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        if (input.len < cols or output.len < 2) return error.ShapeMismatch;
        const rows: u32 = 2;
        const row_bytes: usize = (@as(usize, cols) / 32) * 18;
        if (row_a_q4_0.len < row_bytes or row_b_q4_0.len < row_bytes) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        const weights_bytes = @as(usize, rows) * row_bytes;
        if (weight_off + weights_bytes > self.input_map.len) return error.InputTooLarge;
        if (@as(usize, rows) * @sizeOf(f32) > self.output_map.len) return error.OutputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);
        @memcpy(self.input_map[weight_off..][0..row_bytes], row_a_q4_0[0..row_bytes]);
        @memcpy(self.input_map[weight_off + row_bytes ..][0..row_bytes], row_b_q4_0[0..row_bytes]);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..rows) |i| output_words[i] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_6100 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_dmmv_q4_0_row_range;
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
        output[0] = @bitCast(output_words[0]);
        output[1] = @bitCast(output_words[1]);
    }

    /// Dispatch the gfx1201 Q4_0 row-range DMMV kernel that performs argmax in
    /// the same submission.
    ///
    /// The method stages the exact same source-format input and Q4_0 rows as
    /// `dmmvQ4_0RowRange`, but the kernel only stores the local best row and
    /// score. The forward path uses this for LM-head prefix/window candidates
    /// so a GPU-produced model value can directly participate in sampling
    /// without a follow-up direct argmax dispatch over copied logits.
    pub fn dmmvQ4_0ArgmaxRowRange(
        self: *TokenBoundary,
        input: []const f32,
        weights_q4_0: []const u8,
        rows: u32,
        cols: u32,
    ) !DmmvArgmaxResult {
        if (rows == 0 or cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        if (input.len < cols) return error.ShapeMismatch;
        const row_bytes: usize = (@as(usize, cols) / 32) * 18;
        const weights_bytes = @as(usize, rows) * row_bytes;
        if (weights_q4_0.len < weights_bytes) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        if (weight_off + weights_bytes > self.input_map.len) return error.InputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);
        @memcpy(self.input_map[weight_off..][0..weights_bytes], weights_q4_0[0..weights_bytes]);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        output_words[0] = 0xffff_ffff;
        output_words[1] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_A000 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_dmmv_q4_0_argmax_row_range;
        const pgm_lo: u32 = @truncate(pgm_va >> 8);
        const pgm_hi: u32 = @truncate(pgm_va >> 40);
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ pgm_lo, pgm_hi });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            compute_pgm_rsrc1_vgpr12_value,
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
        const row = output_words[0];
        if (row >= rows) return error.ArgmaxRangeInvalidToken;
        return .{
            .row = row,
            .score = @bitCast(output_words[1]),
        };
    }

    /// Dispatch the gfx1201 row-range Q8_0 matrix-vector kernel.
    ///
    /// Copies the input vector and raw GGML Q8_0 rows into the shared input
    /// page, records PM4 for one serial workitem over `rows`, and reads back one
    /// f32 result per row. This keeps source-format Q8_0 model-slice validation
    /// exact while the final K-parallel DMMV kernel is still under construction.
    /// @param input Input activation vector of length `cols`.
    /// @param weights_q8_0 Row-major GGML Q8_0 row bytes; must hold at least `rows * (cols/32*34)` bytes.
    /// @param rows Number of output rows to compute.
    /// @param cols Inner dimension; must be a multiple of 32.
    /// @param output Output slice receiving `rows` f32 values.
    pub fn dmmvQ8_0RowRange(
        self: *TokenBoundary,
        input: []const f32,
        weights_q8_0: []const u8,
        rows: u32,
        cols: u32,
        output: []f32,
    ) !void {
        if (rows == 0 or cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        if (input.len < cols or output.len < rows) return error.ShapeMismatch;
        const row_bytes: usize = (@as(usize, cols) / 32) * 34;
        const weights_bytes = @as(usize, rows) * row_bytes;
        if (weights_q8_0.len < weights_bytes) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        if (weight_off + weights_bytes > self.input_map.len) return error.InputTooLarge;
        if (@as(usize, rows) * @sizeOf(f32) > self.output_map.len) return error.OutputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);
        @memcpy(self.input_map[weight_off..][0..weights_bytes], weights_q8_0[0..weights_bytes]);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..rows) |i| output_words[i] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_7000 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_dmmv_q8_0_row_range;
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

    /// Dispatch one gfx1201 Q8_0 DMMV kernel over two adjacent logical row
    /// ranges that share the same input vector.
    ///
    /// The method packs `weights_a` followed by `weights_b` into the staging
    /// page, then runs the same compact Q8_0 row-range kernel over
    /// `rows_a + rows_b` rows. The output slice receives A's rows first and B's
    /// rows second. This is used by the M1 bridge to consume paired SSM
    /// alpha/beta projections without paying two CS submissions for the same
    /// activation vector.
    /// @param input Input activation vector of length `cols`.
    /// @param weights_a_q8_0 Row-major GGML Q8_0 bytes for range A; must hold at least `rows_a * (cols/32*34)` bytes.
    /// @param rows_a Number of rows in the A range.
    /// @param weights_b_q8_0 Row-major GGML Q8_0 bytes for range B; must hold at least `rows_b * (cols/32*34)` bytes.
    /// @param rows_b Number of rows in the B range.
    /// @param cols Inner dimension; must be a multiple of 32.
    /// @param output Output slice receiving `rows_a + rows_b` f32 values: A rows first, then B rows.
    /// @note Returns `error.SignalMismatch` if the post-fence signal sentinel does not match.
    pub fn dmmvQ8_0TwoRowRanges(
        self: *TokenBoundary,
        input: []const f32,
        weights_a_q8_0: []const u8,
        rows_a: u32,
        weights_b_q8_0: []const u8,
        rows_b: u32,
        cols: u32,
        output: []f32,
    ) !void {
        return self.dmmvQ8_0TwoRowRangesImpl(input, weights_a_q8_0, rows_a, weights_b_q8_0, rows_b, cols, output, false);
    }

    /// Dispatch one wave64 Q8_0 DMMV kernel over two packed row ranges totalling exactly 64 rows.
    ///
    /// This is the row-parallel companion to `dmmvQ8_0TwoRowRanges` for the
    /// current SSM alpha+beta shape: 32 alpha rows plus 32 beta rows. Each lane
    /// computes one row from the packed staging block, eliminating the serial
    /// per-row loop of the scalar variant.
    /// @param input Input activation vector of length `cols`.
    /// @param weights_a_q8_0 Row-major GGML Q8_0 bytes for range A; must hold at least `rows_a * (cols/32*34)` bytes.
    /// @param rows_a Number of rows in the A range; `rows_a + rows_b` must equal 64.
    /// @param weights_b_q8_0 Row-major GGML Q8_0 bytes for range B; must hold at least `rows_b * (cols/32*34)` bytes.
    /// @param rows_b Number of rows in the B range.
    /// @param cols Inner dimension; must be a multiple of 32.
    /// @param output Output slice receiving exactly 64 f32 values: A rows first, then B rows.
    /// @note Returns `error.ShapeMismatch` if `rows_a + rows_b != 64`. Returns `error.SignalMismatch` on sentinel mismatch.
    pub fn dmmvQ8_0TwoRowRangesParallel64(
        self: *TokenBoundary,
        input: []const f32,
        weights_a_q8_0: []const u8,
        rows_a: u32,
        weights_b_q8_0: []const u8,
        rows_b: u32,
        cols: u32,
        output: []f32,
    ) !void {
        return self.dmmvQ8_0TwoRowRangesImpl(input, weights_a_q8_0, rows_a, weights_b_q8_0, rows_b, cols, output, true);
    }

    /// Dispatch one or more adjacent 64-row Q8_0 row-parallel chunks in one CS submission.
    ///
    /// The helper stages a contiguous Q8_0 row window once, emits one
    /// row-parallel dispatch per 64 rows into the same IB, and waits once after
    /// the final release fence. The default router verifier uses this to keep
    /// full 256-expert coverage without four independent `amdgpu_cs` waits.
    /// @param input Input activation vector of length `cols`.
    /// @param weights_q8_0 Row-major GGML Q8_0 rows; must hold exactly `rows` rows.
    /// @param rows Number of rows to compute; must be a positive multiple of 64.
    /// @param cols Inner dimension; must be a multiple of 32.
    /// @param output Output slice receiving `rows` f32 values.
    pub fn dmmvQ8_0RowRangeParallelChunks(
        self: *TokenBoundary,
        input: []const f32,
        weights_q8_0: []const u8,
        rows: u32,
        cols: u32,
        output: []f32,
    ) !void {
        const chunk_rows: u32 = 64;
        if (rows == 0 or rows % chunk_rows != 0 or cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        if (input.len < cols or output.len < rows) return error.ShapeMismatch;

        const row_bytes: usize = (@as(usize, cols) / 32) * 34;
        const weights_bytes = @as(usize, rows) * row_bytes;
        if (weights_q8_0.len < weights_bytes) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        if (weight_off + weights_bytes > self.input_map.len) return error.InputTooLarge;
        if (@as(usize, rows) * @sizeOf(f32) > self.output_map.len) return error.OutputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);
        @memcpy(self.input_map[weight_off..][0..weights_bytes], weights_q8_0[0..weights_bytes]);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..rows) |i| output_words[i] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_expected: u64 = 0x5A494E435254_8200 | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const pgm_va = self.shader_va + shader_offset_dmmv_q8_0_row_range_parallel;
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ @truncate(pgm_va >> 8), @truncate(pgm_va >> 40) });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            compute_pgm_rsrc1_vgpr16_value,
            compute_pgm_rsrc2_user8_vgpr_workitem_x_value,
        });
        try self.builder.setShRegOne(packet.sh_reg_pgm_rsrc3, 0);
        try self.builder.setShReg(packet.sh_reg_num_thread_x, &[_]u32{ chunk_rows, 1, 1 });
        try self.builder.setShReg(packet.sh_reg_resource_limits, &[_]u32{
            0,
            0xffff_ffff,
            0xffff_ffff,
        });

        const in_lo: u32 = @truncate(self.input_va);
        const in_hi: u32 = @truncate(self.input_va >> 32);
        const weight_va = self.input_va + @as(u64, weight_off);
        var row_start: u32 = 0;
        while (row_start < rows) : (row_start += chunk_rows) {
            const out_va = self.output_va + @as(u64, row_start) * @sizeOf(f32);
            const chunk_weight_va = weight_va + @as(u64, row_start) * @as(u64, row_bytes);
            try self.builder.setShReg(packet.compute_user_data_0, &[_]u32{
                in_lo,
                in_hi,
                @truncate(out_va),
                @truncate(out_va >> 32),
                @truncate(chunk_weight_va),
                @truncate(chunk_weight_va >> 32),
                cols,
                chunk_rows,
            });
            try self.builder.dispatchDirectInitiator(1, 1, 1, packet.dispatch_initiator_compute);
        }
        try self.builder.releaseMemSignal(self.signal_va, signal_expected);
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

    fn dmmvQ8_0TwoRowRangesImpl(
        self: *TokenBoundary,
        input: []const f32,
        weights_a_q8_0: []const u8,
        rows_a: u32,
        weights_b_q8_0: []const u8,
        rows_b: u32,
        cols: u32,
        output: []f32,
        parallel64: bool,
    ) !void {
        if (rows_a == 0 or rows_b == 0 or cols == 0 or cols % 32 != 0) return error.ShapeMismatch;
        const rows = rows_a + rows_b;
        if (rows < rows_a or input.len < cols or output.len < rows) return error.ShapeMismatch;
        if (parallel64 and rows != 64) return error.ShapeMismatch;

        const row_bytes: usize = (@as(usize, cols) / 32) * 34;
        const weights_a_bytes = @as(usize, rows_a) * row_bytes;
        const weights_b_bytes = @as(usize, rows_b) * row_bytes;
        if (weights_a_q8_0.len < weights_a_bytes or weights_b_q8_0.len < weights_b_bytes) return error.ShapeMismatch;

        const input_bytes = std.mem.sliceAsBytes(input[0..cols]);
        const weight_off = std.mem.alignForward(usize, input_bytes.len, 64);
        const weight_b_off = weight_off + weights_a_bytes;
        const weights_bytes = weights_a_bytes + weights_b_bytes;
        if (weight_off + weights_bytes > self.input_map.len) return error.InputTooLarge;
        if (@as(usize, rows) * @sizeOf(f32) > self.output_map.len) return error.OutputTooLarge;

        @memcpy(self.input_map[0..input_bytes.len], input_bytes);
        @memcpy(self.input_map[weight_off..][0..weights_a_bytes], weights_a_q8_0[0..weights_a_bytes]);
        @memcpy(self.input_map[weight_b_off..][0..weights_b_bytes], weights_b_q8_0[0..weights_b_bytes]);

        const output_words: [*]volatile u32 = @ptrCast(@alignCast(self.output_map.ptr));
        const signal_words: [*]volatile u32 = @ptrCast(@alignCast(self.signal_map.ptr));
        for (0..rows) |i| output_words[i] = 0x7fc0_0000;
        signal_words[0] = 0;
        signal_words[1] = 0;
        storeFence();

        const signal_base: u64 = if (parallel64) @as(u64, 0x5A494E435254_8100) else @as(u64, 0x5A494E435254_8000);
        const signal_expected: u64 = signal_base | @as(u64, self.submit_count + 1);
        self.builder.reset();
        try self.builder.writeNop(1);

        const shader_offset: usize = if (parallel64) shader_offset_dmmv_q8_0_row_range_parallel else shader_offset_dmmv_q8_0_row_range;
        const pgm_va = self.shader_va + shader_offset;
        const pgm_lo: u32 = @truncate(pgm_va >> 8);
        const pgm_hi: u32 = @truncate(pgm_va >> 40);
        const pgm_rsrc1: u32 = if (parallel64) compute_pgm_rsrc1_vgpr16_value else compute_pgm_rsrc1_value;
        const pgm_rsrc2: u32 = if (parallel64) compute_pgm_rsrc2_user8_vgpr_workitem_x_value else compute_pgm_rsrc2_argmax_top2_value;
        try self.builder.setShReg(packet.sh_reg_pgm_lo, &[_]u32{ pgm_lo, pgm_hi });
        try self.builder.setShReg(packet.sh_reg_pgm_rsrc1, &[_]u32{
            pgm_rsrc1,
            pgm_rsrc2,
        });
        try self.builder.setShRegOne(packet.sh_reg_pgm_rsrc3, 0);
        const thread_x: u32 = if (parallel64) 64 else 1;
        try self.builder.setShReg(packet.sh_reg_num_thread_x, &[_]u32{ thread_x, 1, 1 });
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
        if (parallel64) {
            try self.builder.releaseMemSignal(self.signal_va, signal_expected);
        } else {
            try self.builder.writeData64(self.signal_va, signal_expected);
        }
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
pub fn submitNopSmokeDefault() SmokeResult {
    return submitNopSmokePath(default_render_node);
}

/// Full bring-up smoke implementation: open the render node, query compute IP,
/// allocate a context, create GTT-backed IB + signal BOs and map them at fixed
/// low GPU VAs, build a PM4 NOP + `WRITE_DATA` stream, submit it twice through
/// `DRM_IOCTL_AMDGPU_CS`, and verify each fence retires with the expected
/// signal sentinel in the signal BO.
/// @param render_node Absolute path to the amdgpu DRM render node to exercise.
/// @returns A `SmokeResult` whose `status` pinpoints the failure stage, or `.ok` on success.
/// @note Returns `.unsupported_os` immediately on non-Linux hosts; never throws.
pub fn submitNopSmokePath(render_node: []const u8) SmokeResult {
    if (builtin.os.tag != .linux) return .{ .status = .unsupported_os, .render_node = render_node };

    last_errno = null;
    var result: SmokeResult = .{ .status = .ok, .render_node = render_node };

    var file = std.fs.openFileAbsolute(render_node, .{ .mode = .read_write }) catch {
        result.status = .render_node_open_failed;
        return result;
    };
    defer file.close();

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
    const ib_map = kmd.mmapGem(file, ib_bo, std.posix.PROT.READ | std.posix.PROT.WRITE) catch {
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
    const signal_map = kmd.mmapGem(file, signal_bo, std.posix.PROT.READ | std.posix.PROT.WRITE) catch {
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
    file: std.fs.File,
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
    file: std.fs.File,
    ctx_id: u32,
    ip_type: u32,
    bo_list_handle: u32,
    builder: *const packet.PacketBuilder,
    ib_chunk_data: *DrmAmdgpuCsChunkIb,
    chunk_ptrs: []u64,
    ib_bytes_out: *u32,
    wait_status_out: *u64,
) SubmitError!u64 {
    const fence_handle = try submitBuilder(
        file,
        ctx_id,
        bo_list_handle,
        builder,
        ib_chunk_data,
        chunk_ptrs,
        ib_bytes_out,
    );
    try waitSubmittedFence(file, ctx_id, ip_type, fence_handle, wait_status_out);
    return fence_handle;
}

fn submitBuilder(
    file: std.fs.File,
    ctx_id: u32,
    bo_list_handle: u32,
    builder: *const packet.PacketBuilder,
    ib_chunk_data: *DrmAmdgpuCsChunkIb,
    chunk_ptrs: []u64,
    ib_bytes_out: *u32,
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
    return submit.out.handle;
}

fn waitSubmittedFence(
    file: std.fs.File,
    ctx_id: u32,
    ip_type: u32,
    fence_handle: u64,
    wait_status_out: *u64,
) SubmitError!void {
    var wait: DrmAmdgpuWaitCs = std.mem.zeroes(DrmAmdgpuWaitCs);
    wait.in = .{
        .handle = fence_handle,
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

fn ioctlRaw(file: std.fs.File, request: u32, arg: usize) error{IoctlFailed}!void {
    last_errno = null;
    const rc = linux.ioctl(file.handle, request, arg);
    const err = linux.E.init(rc);
    if (err != .SUCCESS) {
        last_errno = err;
        return error.IoctlFailed;
    }
}

fn freeContext(file: std.fs.File, ctx_id: u32) void {
    var free_ctx: DrmAmdgpuCtx = std.mem.zeroes(DrmAmdgpuCtx);
    free_ctx.in = .{ .op = AMDGPU_CTX_OP_FREE_CTX, .flags = 0, .ctx_id = ctx_id, .priority = 0 };
    ioctlRaw(file, ioc_ctx, @intFromPtr(&free_ctx)) catch {};
}

fn destroyBoList(file: std.fs.File, bo_list_handle: u32) void {
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

test "gfx1201 multi-WG TGID delivery + VRAM residency probe" {
    // Requires the real amdgpu render node; skips cleanly off-Linux / no GPU.
    var boundary = TokenBoundary.initDefault() catch |e| {
        std.debug.print("\n[probe] SKIP (no GPU render node): {s}\n", .{@errorName(e)});
        return;
    };
    defer boundary.deinit();

    var out = [_]u32{0} ** 8;
    boundary.sgprTgidProbe(4, out[0..4]) catch |e| {
        std.debug.print("\n[probe] TGID dispatch FAILED: {s}\n", .{@errorName(e)});
        return e;
    };
    std.debug.print(
        "\n[probe] TGID out[0..4] = {{ {d}, {d}, {d}, {d} }} (expect 0,1,2,3 = multi-WG OK)\n",
        .{ out[0], out[1], out[2], out[3] },
    );

    const vram_ok = boundary.probeVramMappable(400 * 1024 * 1024) catch |e| blk: {
        std.debug.print("[probe] VRAM probe error: {s}\n", .{@errorName(e)});
        break :blk false;
    };
    std.debug.print("[probe] VRAM 400MB CPU-mappable + roundtrip: {}\n", .{vram_ok});
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

test "embedded gfx12 kernels are not armed on older compute IPs" {
    var hw_ip: kmd.DrmAmdgpuInfoHwIp = std.mem.zeroes(kmd.DrmAmdgpuInfoHwIp);
    hw_ip.hw_ip_version_major = 10;
    try std.testing.expect(!supportsEmbeddedGfx12Kernels(hw_ip));

    hw_ip.hw_ip_version_major = 11;
    try std.testing.expect(!supportsEmbeddedGfx12Kernels(hw_ip));

    hw_ip.hw_ip_version_major = 12;
    try std.testing.expect(supportsEmbeddedGfx12Kernels(hw_ip));

    hw_ip.hw_ip_version_major = 13;
    try std.testing.expect(!supportsEmbeddedGfx12Kernels(hw_ip));
}

test "setupSmokePath reports unsupported_os off Linux" {
    if (builtin.os.tag == .linux) return error.SkipZigTest;
    const r = setupSmokePath(default_render_node);
    try std.testing.expectEqual(SmokeStatus.unsupported_os, r.status);
    try std.testing.expect(!r.ok());
}
