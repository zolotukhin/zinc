//! Wrap the decode-time matrix-vector shader family used for projection ops.
//! @section Shader Dispatch
//! This helper selects quantization-specific DMMV pipelines and records the
//! push constants and workgroup sizes needed for single-token decode.
const std = @import("std");
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const PushDescriptorFn = @import("../vulkan/instance.zig").PushDescriptorFn;
const Pipeline = @import("../vulkan/pipeline.zig").Pipeline;
const pipeline_mod = @import("../vulkan/pipeline.zig");
const CommandBuffer = @import("../vulkan/command.zig").CommandBuffer;
const Buffer = @import("../vulkan/buffer.zig").Buffer;
const GpuConfig = @import("../vulkan/gpu_detect.zig").GpuConfig;
const GGMLType = @import("../model/gguf.zig").GGMLType;

const log = std.log.scoped(.dmmv);
const descriptor_pool_max_sets: u32 = 256;
const descriptors_per_set: u32 = 3;

/// Push constants for DMMV shaders (must match GLSL layout).
pub const DmmvPushConstants = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    x_offset: u32,
    y_offset: u32,
    acc_mode: u32 = 0, // 0 = overwrite (y = result), 1 = accumulate (y += result)
};

test "MoE route-packed columns support Q6_K alignment" {
    try std.testing.expect(moeColsKAligned(.q4_k, 256));
    try std.testing.expect(moeColsKAligned(.q5_k, 256));
    try std.testing.expect(moeColsKAligned(.q6_k, 256));
    try std.testing.expect(moeColsKAligned(.q5_1, 32));
    try std.testing.expect(!moeColsKAligned(.q6_k, 128));
    try std.testing.expect(!moeColsKAligned(.q8_0, 256));
}

/// Push constants for batch DMMV shaders (prefill: multiple columns).
pub const BatchDmmvPushConstants = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    x_offset: u32,
    y_offset: u32,
    num_cols: u32,
};

/// Push constants for MoE DMMV shaders (must match GLSL layout).
/// Batched expert dispatch: workgroup Y dimension selects expert slot.
pub const MoeDmmvPushConstants = extern struct {
    M: u32,
    K: u32,
    expert_stride: u32,
    x_expert_stride: u32,
    x_offset: u32,
    y_offset: u32,
};

/// Push constants for the cross-token batched MoE DMMV
/// (src/shaders/dmmv_q4k_moe_batched.comp). Each WG handles one
/// (row_pair, expert_slot, token_idx) triple; the routing buffer is
/// flattened [n_tokens × n_experts_used] of expert IDs.
pub const MoeBatchedDmmvPushConstants = extern struct {
    M: u32,
    K: u32,
    expert_stride: u32,
    n_experts_used: u32,
    x_token_stride: u32,
    y_token_stride: u32,
};

/// Push constants for grouped route-column MoE DMMV shaders.
pub const MoeColsDmmvPushConstants = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    expert_stride: u32,
    x_offset: u32,
    y_offset: u32,
    ids_stride: u32,
    x_route_divisor: u32,
    accumulate: u32 = 0,
};

fn moeColsKAligned(quant_type: GGMLType, K: u32) bool {
    if (K == 0) return false;
    return switch (quant_type) {
        .q5_1 => (K & 31) == 0,
        .q4_k, .q5_k, .q6_k => (K & 255) == 0,
        else => false,
    };
}

/// Push constants for the fused MoE down + weighted_acc shader
/// (src/shaders/dmmv_q4k_moe_fused_down_acc.comp). Same layout as
/// MoeDmmvPushConstants plus n_used (the expert loop is internal to
/// the shader so the dispatch grid drops the Y=n_experts_used dim).
pub const MoeFusedDownAccPushConstants = extern struct {
    M: u32,
    K: u32,
    expert_stride: u32,
    x_expert_stride: u32,
    x_offset: u32,
    y_offset: u32,
    n_used: u32,
    acc_mode: u32 = 0, // 0 = overwrite, 1 = accumulate into Y
};

/// Push constants for Gemma's packed Q4_K MoE gate+up+GEGLU shader.
/// The shader reads expert ids from the routing buffer, so `expert_stride`
/// spans the full packed gate+up expert and `up_offset` selects the up half.
pub const MoeGateUpGegluPushConstants = extern struct {
    M: u32,
    K: u32,
    expert_stride: u32,
    up_offset: u32,
    x_offset: u32,
    y_offset: u32,
};

/// Push constants for Gemma short-prefill token-batched top-1 MoE shaders.
pub const GemmaTop1MoeBatchPushConstants = extern struct {
    M: u32,
    K: u32,
    expert_stride: u32,
    up_or_routing_stride: u32,
    routing_stride: u32 = 0,
};

/// Push constants for route-packed Gemma short-prefill top-1 gate/up+GEGLU.
pub const GemmaTop1GateUpColsPushConstants = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    expert_stride: u32,
    up_offset: u32,
    y_offset: u32,
    ids_stride: u32,
    x_route_divisor: u32,
    x_token_base: u32 = 0,
};

/// Push constants for the fused split-K merge + o_proj DMMV-acc shader
/// (src/shaders/dmmv_q4k_o_proj_merge.comp). Adds the merge-pass parameters
/// to the standard DmmvPushConstants so a single dispatch reads partials,
/// computes per-head LSE merge weights, stages attn_out into LDS, and runs
/// the Q4_K matmul with residual accumulation.
pub const OprojMergePushConstants = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    x_offset: u32,
    y_offset: u32,
    acc_mode: u32 = 1,
    n_heads: u32,
    n_i_chunks: u32,
    sink_offset: u32,
    head_dim: u32,
};

/// Push constants for the fused Q8_0 DMMV + sigmoid-gated scale-accumulate
/// shader (src/shaders/dmmv_q8_0_sigmoid_acc.comp). Same prefix as
/// DmmvPushConstants but `acc_mode` is replaced by `gate_offset` (the
/// shader always accumulates and always sigmoid-gates; gate_offset selects
/// which f32 in the gate buffer holds the shexp_gate scalar — typically 0).
pub const DmmvSigmoidAccPushConstants = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    x_offset: u32,
    y_offset: u32,
    gate_offset: u32,
};

/// Push constants for the Gemma CPU-MoE fused gate+up+GEGLU shader.
/// Gate/up offsets are byte offsets into the same packed Q4_K expert tensor;
/// x/y offsets follow the standard DMMV byte convention.
pub const DmmvGateUpGegluPushConstants = extern struct {
    M: u32,
    K: u32,
    gate_offset: u32,
    up_offset: u32,
    x_offset: u32,
    y_offset: u32,
};

/// Push constants for quantized DMMV fused with `y += scale * dot(W, x)`.
pub const DmmvScaleAccPushConstants = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    x_offset: u32,
    y_offset: u32,
    scale_bits: u32,
};

/// Push constants for the fused Q8_0 pair DMMV shader. Computes two
/// independent Q8_0 matvecs that share one F32 input vector.
pub const DmmvQ8PairPushConstants = extern struct {
    M0: u32,
    M1: u32,
    K: u32,
    a0_offset: u32 = 0,
    a1_offset: u32 = 0,
    x_offset: u32 = 0,
    y0_offset: u32 = 0,
    y1_offset: u32 = 0,
};

/// Push constants for the quantize_q8_1 shader.
/// `ne` = number of f32 input elements (must be a multiple of 32).
/// `num_blocks` = ne / 32. Pass explicitly so the shader does not have to divide.
pub const QuantizeQ8_1Push = extern struct {
    ne: u32,
    num_blocks: u32,
};

/// Push constants for `count_experts.comp` (effort-6 Step 3 helper). Mirrors
/// the reference implementation's count_experts push so the shader is structurally identical
/// to the upstream version. All strides are in u32 units (not bytes).
///
/// For the prefill routing capture buffer with layout
///   slot(token, layer) = (token * n_layers + layer) * (2 * n_experts_used)
/// where the first n_experts_used u32s are expert IDs and the second
/// n_experts_used u32s are f32 weights, configure as:
///   ne00 = n_experts_used                (cells per token row)
///   ne01 = n_tokens                      (number of token rows)
///   nb00 = 1                             (consecutive within slot)
///   nb01 = 2 * n_experts_used * n_layers (skip n_layers slots per row step)
///   a_offset = layer * 2 * n_experts_used (jump to layer's slot in token 0)
pub const CountExpertsPush = extern struct {
    ne00: u32,
    ne01: u32,
    nb00: u32,
    nb01: u32,
    a_offset: u32,
};

/// Push constants for `moe_route_pack` — turns per-token expert routing into
/// expert-grouped work lists (per-expert counts, packed token IDs, the active-block
/// list, and the indirect dispatch args consumed by the MoE columns kernels).
pub const MoeRoutePackPush = extern struct {
    n_tokens: u32,
    n_experts: u32,
    k: u32,
    routing_stride: u32,
    ids_stride: u32,
    gate_up_workgroups_x: u32,
    down_workgroups_x: u32,
    routing_token_base: u32,
};

/// Push constants for `mul_mm_q4k.comp` (effort-6 Step 1 of 5 foundation:
/// tiled Q4_K dense GEMM). Mirrors the dispatch-side argument shape needed
/// to address an M×K Q4_K weight tensor against a K×N f32 activation tile.
/// All offsets follow the existing dmmv convention:
///   `a_offset` is in BYTES (the shader divides by 4 to index a_u32[]),
///   `b_offset` and `d_offset` are in FLOATS.
/// Layout for B/D is column-major: B[col][k] = data_b[b_offset + col*stride_b + k].
pub const MulMmQ4KPush = extern struct {
    M: u32,
    N: u32,
    K: u32,
    stride_b: u32,
    stride_d: u32,
    a_offset: u32,
    b_offset: u32,
    d_offset: u32,
};

/// Push constants for `mul_mm_id_q4k.comp`, the routed MoE gather sibling
/// of `MulMmQ4KPush`. Offsets follow existing GEMM conventions: `a_offset`
/// and `batch_stride_a` are bytes, `b_offset`, `d_offset`, `stride_b`,
/// `stride_d`, and `batch_stride_d` are f32 elements, and `ids_offset` is
/// in u32 elements.
pub const MulMmIdQ4KPush = extern struct {
    M: u32,
    K: u32,
    stride_b: u32,
    stride_d: u32,
    batch_stride_a: u32,
    batch_stride_d: u32,
    nei0: u32,
    nei1: u32,
    nbi1: u32,
    a_offset: u32,
    b_offset: u32,
    d_offset: u32,
    ids_offset: u32,
};

/// Push constants for the int8 DP4a full-tile Q6_K dense-down GEMM.
pub const MulMmQ6KDp4aPush = extern struct {
    M: u32,
    N: u32,
    K: u32,
    stride_b_packed: u32, // per-column uints in the packed int8 activation (K/4)
    stride_b_scale: u32, // per-column floats in the activation scale (K/32)
    stride_d: u32, // per-column floats in output (M)
    a_offset: u32, // byte offset into Q6_K weights
    d_offset: u32, // float offset into output
};

/// Push constants for the one-shot per-32-block activation int8 quantizer.
pub const QuantizeActPush = extern struct {
    n_tokens: u32,
    K: u32,
    blocks_per_token: u32,
    total_blocks: u32,
};

/// Push constants for the int8 DP4a full-tile Q4_K gate+up+SwiGLU GEMM.
/// Same fields as `MulMmQ6KDp4aPush` but `stride_b_scale` counts vec2 entries
/// (one per 32-block) so the shader can fetch (scale, dsum) in one read.
pub const MulMmQ4KGateUpDp4aPush = extern struct {
    M: u32,
    N: u32,
    K: u32,
    stride_b_packed: u32, // per-column uints in the packed int8 activation (K/4)
    stride_b_scale: u32, // per-column vec2s (scale, dsum) in the activation scale (K/32)
    stride_d: u32, // per-column floats in output (M)
    a_offset: u32, // byte offset into Q4_K weights (same value for gate & up SSBOs)
    d_offset: u32, // float offset into output
};

/// Push constants for the int8 DP4a full-tile Q4_K gate+up+SwiGLU GEMM that
/// emits Q8_0-style packed activations directly (fused SwiGLU + quantize for
/// the Qwen3.6-27B dense-down DP4a path).
pub const MulMmQ4KGateUpDp4aQ8Push = extern struct {
    M: u32,
    N: u32,
    K: u32,
    stride_b_packed: u32, // per-column uints in input packed int8 activation (K/4)
    stride_b_scale: u32, // per-column vec2s in input scale buffer (K/32)
    stride_d_packed: u32, // per-token uints in output packed swiglu (M/4)
    stride_d_scale: u32, // per-token floats in output scale buffer (M/32)
    a_offset: u32, // byte offset into Q4_K weights (gate AND up)
    b_packed_offset: u32, // uint offset into input packed activation
    b_scale_offset: u32, // vec2/float offset into input scale buffer
    d_packed_offset: u32, // uint offset into output packed buffer
    d_scale_offset: u32, // float offset into output scale buffer
};

/// Size in bytes of a single Q8_1 output block (32 int8 values + f16 d + f16 d*sum).
pub const Q8_1_BLOCK_BYTES: u32 = 36;

/// Manages DMMV pipelines for different quantization types.
pub const DmmvDispatch = struct {
    /// Q4K pipeline, or null.
    pipeline_q4k: ?Pipeline,
    /// Q4K wide-vocab variant (NUM_ROWS=8) for tall matrices like the Gemma
    /// 4 31B LM head (M=262144). Same binding layout as pipeline_q4k — swap
    /// in at the call site when M is large enough to benefit from 4× fewer
    /// workgroups with 4× more hidden-vector reuse per workgroup.
    pipeline_q4k_wide: ?Pipeline,
    /// Cross-token batched MoE DMMV. Same Q4_K weight layout as
    /// pipeline_q4k_moe_kpar but the dispatch grid adds a token_idx
    /// dimension (grid.z) so one dispatch covers all N prompt tokens'
    /// MoE FFN work for a given (gate, up, or down) projection. Routing
    /// buffer is flattened to [n_tokens × n_experts_used]. Foundation for
    /// batched MoE prefill on qwen35moe / qwen36moe — pipeline registered,
    /// dispatch helper available, but not yet wired into prefillBatched.
    pipeline_q4k_moe_batched: ?Pipeline,
    /// Q5K pipeline, or null.
    pipeline_q5k: ?Pipeline,
    /// Q6K pipeline, or null.
    pipeline_q6k: ?Pipeline,
    /// Q6K wide-vocab variant (NUM_ROWS=8) for tall LM heads.
    pipeline_q6k_wide: ?Pipeline,
    /// MXFP4 pipeline, or null.
    pipeline_mxfp4: ?Pipeline,
    /// Q5_0 pipeline, or null.
    pipeline_q5_0: ?Pipeline,
    /// Q5_1 pipeline, or null.
    pipeline_q5_1: ?Pipeline,
    /// Q8 0 pipeline, or null.
    pipeline_q8_0: ?Pipeline,
    /// Q8_0 one-row-per-thread batch-style variant for very tall matrices.
    pipeline_q8_0_batch: ?Pipeline,
    /// Q8_0 K-parallel batch variant for multi-column prefill (one WG per row,
    /// 64 threads cooperate on K, acc_mode > 1 = num_cols).
    pipeline_q8_0_kpar_batch: ?Pipeline,
    /// Q8_0 pipeline with SPEC_BLOCKS_PER_ROW=64 (K=2048).
    pipeline_q8_0_spec64: ?Pipeline,
    /// Q8_0 pipeline with SPEC_BLOCKS_PER_ROW=128 (K=4096).
    pipeline_q8_0_spec128: ?Pipeline,
    /// Q8_0 wide-vocab LM-head variant. Same two-row workgroup shape and
    /// binding layout as pipeline_q8_0, but shares x-vector loads across rows.
    pipeline_q8_0_wide: ?Pipeline,
    /// Q8_0 wide-vocab LM-head variant with four rows per workgroup.
    pipeline_q8_0_wide4: ?Pipeline,
    /// Q8_0 x Q8_1 integer-dot DMMV. Binding 1 is a quantized Q8_1 activation
    /// buffer produced by pipeline_quantize_q8_1.
    pipeline_q8_0_q8_1: ?Pipeline,
    /// Q4_K x Q8_1 integer-dot DMMV. Binding 1 is a quantized Q8_1 activation
    /// buffer produced by pipeline_quantize_q8_1.
    pipeline_q4k_q8_1: ?Pipeline,
    /// Fused Q8_0 pair DMMV. Five bindings: A0, A1, X, Y0, Y1. Used as an
    /// opt-in SSM projection experiment for wqkv + z/gate, whose matrices
    /// share the same normalized hidden vector.
    pipeline_q8_0_fused_pair: ?Pipeline,
    /// F16 pipeline, or null.
    pipeline_f16: ?Pipeline,
    /// F32 pipeline, or null.
    pipeline_f32: ?Pipeline,
    /// Batch Q4K pipeline for prefill (3 bindings: A, X_batch, Y_batch).
    pipeline_q4k_batch: ?Pipeline,
    /// Batch Q4K pipeline, K-parallel wave64 variant (3 bindings). Enabled
    /// via ZINC_Q4K_BATCH_KPAR=1. One WG per row with 64-way K parallelism;
    /// same binding shape as pipeline_q4k_batch.
    pipeline_q4k_batch_kpar: ?Pipeline,
    /// Batch Q6K pipeline for prefill (3 bindings: A, X_batch, Y_batch).
    /// Unlocks batched prefill for Q4_K_M checkpoints — the attn_v and
    /// ffn_down tensors are Q6_K in that layout, so the all-Q4_K gate
    /// was previously rejecting every real catalog model.
    pipeline_q6k_batch: ?Pipeline,
    /// Batch Q6K pipeline, K-parallel wave64 variant. Same binding shape
    /// as pipeline_q6k_batch. Enabled by default alongside the Q4_K
    /// kpar variant so Q4_K_M models (with Q6_K on attn_v / ffn_down)
    /// don't regress to the serial shader for those projections.
    pipeline_q6k_batch_kpar: ?Pipeline,
    /// MoE Q4K pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_q4k_moe: ?Pipeline,
    /// Experimental K-parallel Q4K MoE pipeline (same 4 bindings, wave64 subgroupAdd).
    pipeline_q4k_moe_kpar: ?Pipeline,
    /// Grouped top-1 prefill Q4_K MoE DMMV over route-packed token columns.
    pipeline_q4k_moe_cols: ?Pipeline,
    /// Q8_1-input sibling for grouped Q4_K MoE down projection probes.
    pipeline_q4k_moe_cols_q8_1: ?Pipeline,
    /// Fused gate+up Q4_K MoE pipeline (6 bindings: W_gate, W_up, X, Y_gate,
    /// Y_up, routing). Halves the dispatch count for the MoE gate+up phase
    /// and reads the shared input once per block.
    pipeline_q4k_fused_gate_up_moe: ?Pipeline,
    /// Cycle 154: same shader as pipeline_q4k_fused_gate_up_moe but with
    /// SPEC_BLOCKS_PER_ROW=8 baked in at pipeline-spec time. Used when
    /// hidden_dim=2048 (Qwen 3.5/3.6 MoE catalog), so the inner block loop
    /// has a compile-time bound and the SPIR-V → AMDGPU compiler can unroll
    /// + fold the per-block index arithmetic. Falls back to the unspec'd
    /// pipeline_q4k_fused_gate_up_moe for other K values.
    pipeline_q4k_fused_gate_up_moe_spec8: ?Pipeline,
    /// MoE-specific fused gate+up+SwiGLU variant (5 bindings: W_gate, W_up,
    /// X, activatedY, routing). Writes silu(gate) * up straight into the
    /// per-expert activation buffer so the forward path can skip the separate
    /// MoE SwiGLU dispatch.
    pipeline_q4k_fused_gate_up_swiglu_moe: ?Pipeline,
    /// Same shader as pipeline_q4k_fused_gate_up_swiglu_moe with
    /// SPEC_BLOCKS_PER_ROW=8 baked for hidden_dim=2048 Qwen A3B experts.
    pipeline_q4k_fused_gate_up_swiglu_moe_spec8: ?Pipeline,
    /// Fused gate+up+SwiGLU Q4_K dense pipeline (4 bindings: W_gate, W_up,
    /// X, swiglu_out). Replaces the dense FFN front-end dispatch trio
    /// (gate DMMV + up DMMV + swiglu element-wise) with a single dispatch
    /// that computes silu(W_gate·x) * (W_up·x) inline. Eliminates gate_buf
    /// and up_buf from the dense decode datapath. Saves one global compute
    /// barrier per layer (gate+up → swiglu), and reads the shared input
    /// once per block. Same DmmvPushConstants layout as pipeline_q4k.
    pipeline_q4k_fused_gate_up_swiglu: ?Pipeline,
    /// Same shader as pipeline_q4k_fused_gate_up_swiglu with NUM_ROWS=1 via
    /// specialization constant. Targets Qwen3.6-27B's wide dense FFN gate/up
    /// shape where the regular NUM_ROWS=2 fused path was register-pressure
    /// sensitive.
    pipeline_q4k_fused_gate_up_swiglu_row1: ?Pipeline,
    /// Gemma CPU-routed MoE expert front-end fusion. Gemma packs expert
    /// gate/up rows into one Q4_K tensor, so this takes independent byte
    /// offsets into that tensor and writes GEGLU(gate, up) directly.
    pipeline_q4k_fused_gate_up_geglu: ?Pipeline,
    /// Gemma shared-expert front-end fusion for separate Q4_K gate/up tensors.
    /// Same binding shape as the dense SwiGLU pair shader, but applies GEGLU.
    pipeline_q4k_fused_gate_up_geglu_pair: ?Pipeline,
    /// Gemma batched MoE front-end: reads CPU-selected expert ids from the
    /// routing buffer and writes GEGLU activations for all selected experts
    /// into contiguous activation slabs.
    pipeline_q4k_moe_fused_gate_up_geglu: ?Pipeline,
    /// Gemma short-prefill token-batched top-1 fused gate+up+GEGLU.
    pipeline_q4k_moe_fused_gate_up_geglu_batch_top1: ?Pipeline,
    /// Gemma short-prefill route-packed top-1 fused gate+up+GEGLU.
    pipeline_q4k_moe_fused_gate_up_geglu_cols_top1: ?Pipeline,
    /// Qwen short-prefill route-packed top-1 fused gate+up+SwiGLU for
    /// separate Q4_K gate/up expert tensors.
    pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1: ?Pipeline,
    /// DP4a/Q8_1 sibling of the grouped Qwen top-1 gate+up+SwiGLU path.
    pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1_q8_1: ?Pipeline,
    /// Q8_0 sibling of pipeline_q4k_fused_gate_up_swiglu. Targets the
    /// shared expert in Qwen 3.5 / 3.6 MoE packs where shared FFN
    /// weights ship as Q8_0 (rather than Q4_K). Same 4-binding layout
    /// (W_gate, W_up, X, swiglu_out) and same DmmvPushConstants. Used
    /// to fuse the per-token shared-expert (gate DMMV + up DMMV +
    /// SwiGLU) trio into one dispatch.
    pipeline_q8_0_fused_gate_up_swiglu: ?Pipeline,
    /// Q8_0 fused shared-expert gate+up+SwiGLU plus the f32 shared gate
    /// scalar (`ffn_gate_inp_shexp`) computed by WG 0. Saves the separate
    /// one-row f32 DMMV before the shared down tail.
    pipeline_q8_0_fused_gate_up_swiglu_gate: ?Pipeline,
    /// Gemma shared-expert Q8_0 gate+up fusion with inline GEGLU.
    pipeline_q8_0_fused_gate_up_geglu: ?Pipeline,
    /// Four-row sibling of the Gemma shared-expert Q8_0 gate+up GEGLU fusion.
    pipeline_q8_0_fused_gate_up_geglu4: ?Pipeline,
    /// Q8_0 DMMV fused with sigmoid-gated scale-accumulate. Replaces
    /// the (down_shexp DMMV → barrier → sigmoid_scale_acc) pair on the
    /// Qwen 3.5 / 3.6 shared-expert tail when the shared down weights
    /// are Q8_0. 4 bindings (W, X=gate_buf, Y=hidden_buf, gate=router_logits_buf).
    /// Push: DmmvSigmoidAccPushConstants. Saves 1 dispatch + 1 barrier
    /// per layer per token.
    pipeline_q8_0_sigmoid_acc: ?Pipeline,
    /// Fused split-K merge + Q4_K o_proj DMMV-acc pipeline (4 bindings:
    /// W_o, partial_attn_out_buf, hidden_buf, sinks). Replaces the
    /// (flash_attn_split_merge → o_proj DMMV-acc) pair with one dispatch:
    /// each WG reads per-head M, L, computes LSE merge weights with sink
    /// fold-in, stages the merged attn_out (hidden_dim floats) into LDS,
    /// then runs the standard Q4_K matmul reading the B-vector from LDS
    /// and accumulating into hidden_buf. Saves 1 dispatch + 1 barrier per
    /// attention layer when split-K is active. Push constants:
    /// OprojMergePushConstants. Gated behind ZINC_FUSED_OPROJ_MERGE.
    pipeline_q4k_o_proj_merge: ?Pipeline,
    /// Fused Q4_K MoE down + weighted_acc pipeline (4 bindings: A, X,
    /// Y=hidden_buf, routing). Each WG owns NUM_ROWS=2 hidden rows and
    /// loops over n_used experts internally, eliminating the separate
    /// moe_weighted_acc dispatch on call sites where Y can be hidden_buf
    /// directly (no post_ffw_norm, no ffn_down_exps_scale).
    pipeline_q4k_moe_fused_down_acc: ?Pipeline,
    /// Q5_K analogue of pipeline_q4k_moe_fused_down_acc. Same 4-binding
    /// shape and MoeFusedDownAccPushConstants layout. Targets Q4_K_M / XL
    /// packs where the down projection ships as Q5_K (e.g.
    /// Qwen3.6-35B-A3B-UD-Q4_K_XL) so the fused path can engage there too.
    pipeline_q5k_moe_fused_down_acc: ?Pipeline,
    /// Q5_K fused routed down+acc over Q8_1-quantized per-expert SwiGLU
    /// activations. A3B decode specialization for K=512.
    pipeline_q5k_moe_fused_down_acc_q8_1: ?Pipeline,
    /// MoE Q5K pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_q5k_moe: ?Pipeline,
    /// Experimental K-parallel Q5K MoE pipeline (same 4 bindings, wave64 subgroupAdd).
    pipeline_q5k_moe_kpar: ?Pipeline,
    /// Grouped top-1 prefill Q5_K MoE DMMV over route-packed token columns.
    pipeline_q5k_moe_cols: ?Pipeline,
    /// Q8_1-input sibling for grouped Q5_K MoE down projection probes.
    pipeline_q5k_moe_cols_q8_1: ?Pipeline,
    /// MoE MXFP4 pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_mxfp4_moe: ?Pipeline,
    /// MoE Q5_1 pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_q5_1_moe: ?Pipeline,
    /// Grouped Gemma Q5_1 MoE down DMMV over route-packed token columns.
    pipeline_q5_1_moe_cols: ?Pipeline,
    /// Q5_1 DMMV fused with scaled accumulation. Used by Gemma CPU-routed
    /// MoE to fold down projection + weighted accumulation into one dispatch.
    pipeline_q5_1_acc: ?Pipeline,
    /// Gemma Q5_1 MoE down projection fused across all selected experts.
    /// Reads ids/weights from the routing buffer and writes the accumulated
    /// hidden vector directly, replacing per-expert down+acc dispatches.
    pipeline_q5_1_moe_fused_down_acc: ?Pipeline,
    /// Same as pipeline_q5_1_moe_fused_down_acc, but also reads
    /// ffn_down_exps.scale on-GPU. This lets Gemma use GPU top-k without
    /// CPU-side routing weight patch-up.
    pipeline_q5_1_moe_fused_down_acc_scaled: ?Pipeline,
    /// Gemma short-prefill token-batched top-1 Q5_1 down + scaled accumulate.
    pipeline_q5_1_moe_down_acc_scaled_batch_top1: ?Pipeline,
    /// Q8_0 sibling of pipeline_q5_1_moe_fused_down_acc_scaled for Gemma
    /// layers whose down experts are stored as Q8_0.
    pipeline_q8_0_moe_fused_down_acc_scaled: ?Pipeline,
    /// MoE Q6K pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_q6k_moe: ?Pipeline,
    /// Grouped top-1 prefill Q6_K MoE DMMV over route-packed token columns.
    /// Covers mixed-quant Qwen A3B layers whose down experts are Q6_K.
    pipeline_q6k_moe_cols: ?Pipeline,
    /// Foundation for future mul_mmq work: quantize an F32 activation into
    /// Q8_1 blocks. 2 bindings (A f32 vec4 in, D u32 stream out), push
    /// constants {ne, num_blocks}. Activation pre-quantizer preserved as
    /// reusable infrastructure (per effort-6 plan); the SSM proj mmq DMMV
    /// consumers (dmmv_q8_0_q8_1, dmmv_q4k_q8_1) share this exact
    /// 36-byte-per-block layout.
    pipeline_quantize_q8_1: ?Pipeline,
    /// Effort-6 Step 3 foundation for the routed tiled GEMM path.
    /// Counts how many tokens were routed to each expert. Output is a
    /// `[n_experts]` u32 buffer consumed by the dormant mul_mm_id_q4k path
    /// for per-expert early exits. Pipeline loads at startup; no production
    /// callers until replay validation lands. 2 bindings (A routing in,
    /// D counts out), push = CountExpertsPush.
    pipeline_count_experts: ?Pipeline,
    /// Packs a token-major route cache into expert-major active blocks.
    pipeline_moe_route_pack: ?Pipeline,
    /// Effort-6 Step 1 of 5 foundation: tiled Q4_K dense GEMM. First port
    /// of the reference implementation's mul_mm.comp #ifndef COOPMAT branch adapted to ZINC's
    /// wave64 / Q4_K conventions. WG produces a 32×16 output tile with 64
    /// threads; BK=32 (one Q4_K sub-block) per outer step. Pipeline loads
    /// at startup; this cycle wires it under ZINC_MUL_MM_LM_HEAD=1 to the
    /// final-tail LM head dispatch as a correctness-exercise (LM head
    /// fires once per prefill so the perf-hotpath impact is small). The
    /// routed mul_mm_id_q4k sibling is registered separately. 3 bindings
    /// (A weights, B f32 activations, D f32 outputs), push = MulMmQ4KPush.
    pipeline_mul_mm_q4k: ?Pipeline,
    /// Routed MoE gather sibling of pipeline_mul_mm_q4k. The shader consumes
    /// token-major route ids plus per-expert counts and scatters outputs by
    /// token/slot. It is compile-registered as a validation target; production
    /// prefill still uses route-packed columns until this path has numeric
    /// replay coverage.
    pipeline_mul_mm_id_q4k: ?Pipeline,
    /// Same GEMM as pipeline_mul_mm_q4k but the output ACCUMULATES into
    /// d_data (d[idx] += result). Used by the fused dense-FFN residual path
    /// to eliminate the standalone barrier + scale_accumulate dispatch.
    /// Fires for the Qwen3.5-9B whose down_proj is Q4_K.
    pipeline_mul_mm_q4k_down_acc: ?Pipeline,
    /// Wide-tile (BN=64) variant of the fused-residual Q4_K dense-down GEMM.
    /// Halves weight VRAM traffic for N=64 prefill (1 WG/M-tile vs 2).
    pipeline_mul_mm_q4k_down_acc_wide: ?Pipeline,
    /// Batched dense FFN front-end for Qwen3.6-27B: gate/up Q4_K GEMMs plus
    /// SwiGLU in one tiled dispatch.
    pipeline_mul_mm_q4k_gate_up_swiglu: ?Pipeline,
    /// Batched dense FFN front-end for Gemma 4: gate/up Q4_K GEMMs plus
    /// GEGLU in one tiled dispatch.
    pipeline_mul_mm_q4k_gate_up_geglu: ?Pipeline,
    /// Branchless full-tile variant of the fused Q4_K gate/up/GEGLU GEMM.
    pipeline_mul_mm_q4k_gate_up_geglu_full: ?Pipeline,
    /// Narrow BN=8 Q4_K gate/up/GEGLU GEMM for Gemma short-prompt tails.
    pipeline_mul_mm_q4k_gate_up_geglu_tail8: ?Pipeline,
    /// Branchless full-tile variant of the fused Q4_K gate/up/SwiGLU GEMM.
    pipeline_mul_mm_q4k_gate_up_swiglu_full: ?Pipeline,
    /// Tiled Q6_K dense GEMM for Qwen3.6-27B batched dense-down prefill.
    pipeline_mul_mm_q6k: ?Pipeline,
    /// Narrow BN=8 Q4_K GEMM for Gemma 31B short-prompt projection tails.
    pipeline_mul_mm_q4k_tail8: ?Pipeline,
    /// Branchless full-tile Q6_K GEMM. Host routes only 32-aligned M/N tiles here.
    pipeline_mul_mm_q6k_full: ?Pipeline,
    /// Same GEMM as pipeline_mul_mm_q6k_full but the output ACCUMULATES into
    /// d_data (d[idx] += result). Used by the fused dense-FFN residual path
    /// to eliminate the standalone barrier + scale_accumulate dispatch.
    pipeline_mul_mm_q6k_full_down_acc: ?Pipeline,
    /// Narrow BN=8 Q6_K GEMM for Gemma 31B short-prompt tails.
    pipeline_mul_mm_q6k_tail8: ?Pipeline,
    /// Tiled Q5_K dense GEMM for Qwen3.6-27B batched SSM out projection prefill.
    pipeline_mul_mm_q5k: ?Pipeline,
    /// Wide-tile (BN=64) variant of the Q5_K GEMM for SSM projections.
    /// Halves weight VRAM traffic for N=64 prefill.
    pipeline_mul_mm_q5k_wide: ?Pipeline,
    /// Tiled Q8_0 dense GEMM for Qwen3.6 A3B batched SSM out projection prefill.
    pipeline_mul_mm_q8_0: ?Pipeline,
    /// int8 DP4a full-tile Q8_0 GEMM over Q8_0-style pre-quantized activations.
    pipeline_mul_mm_q8_0_full_dp4a: ?Pipeline,
    /// int8 DP4a full-tile Q6_K dense-down GEMM (Qwen3.6-27B prefill).
    pipeline_mul_mm_q6k_full_dp4a: ?Pipeline,
    /// Same Q6_K DP4a dense-down shader with K=12288 specialized for
    /// Qwen3.5-9B dense FFN down projections.
    pipeline_mul_mm_q6k_full_dp4a_k12288: ?Pipeline,
    /// K=12288, BN=64 guarded sibling for Qwen3.5-9B long dense-down bodies.
    pipeline_mul_mm_q6k_full_dp4a_k12288_n64_ragged: ?Pipeline,
    /// K=12288, BN=64 two-slice guarded sibling for Qwen3.5-9B long dense-down bodies.
    pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bk2_ragged: ?Pipeline,
    /// K=12288, BM=64/BN=64 two-slice guarded sibling for Qwen3.5-9B long dense-down bodies.
    pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bm64_ragged: ?Pipeline,
    /// K=17408, BN=64 sibling for short prompts padded to a 64-column body.
    pipeline_mul_mm_q6k_full_dp4a_k17408_n64: ?Pipeline,
    /// K=17408, BN=64 two-slice sibling used only by exact 64-token bodies.
    pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2: ?Pipeline,
    /// K=17408, BM=64/BN=64 sibling for 64-aligned overwrite bodies.
    pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64: ?Pipeline,
    /// K=17408, BM=64/BN=64 two-wave sibling that accumulates exact 64-token bodies.
    pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc: ?Pipeline,
    /// K=17408, BM=64/BN=64 low-register accumulate sibling for exact 64-token bodies.
    pipeline_mul_mm_q6k_full_dp4a_k17408_n64_mmq64_acc: ?Pipeline,
    /// Exact 64-token sibling that consumes K-block-major interleaved Q8 activations.
    pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc_interleaved: ?Pipeline,
    /// K=17408, BN=64 two-slice sibling that accumulates into the residual.
    pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2_acc: ?Pipeline,
    /// K=17408, BN=64 guarded sibling for long dense-hybrid ragged bodies.
    pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged: ?Pipeline,
    /// K=17408, BM=64/BN=64 guarded sibling for long dense-hybrid ragged overwrite bodies.
    pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged_bm64: ?Pipeline,
    /// K=17408, BN=40 sibling for 33-40 token dense-hybrid 27B prompts.
    pipeline_mul_mm_q6k_full_dp4a_k17408_n40: ?Pipeline,
    /// Same Q6_K DP4a dense-down shader with K=21504 specialized for
    /// Gemma 4 31B dense FFN down projections.
    pipeline_mul_mm_q6k_full_dp4a_k21504: ?Pipeline,
    /// K=21504, BM=64/BN=64 sibling for Gemma 4 31B Q6_K dense-down 49-64 token bodies.
    pipeline_mul_mm_q6k_full_dp4a_k21504_n64_bm64: ?Pipeline,
    /// K=21504, BN=64 sibling for Gemma 4 31B Q6_K dense-down 49-64 token bodies.
    pipeline_mul_mm_q6k_full_dp4a_k21504_n64: ?Pipeline,
    /// K=21504, BN=8 sibling for Gemma 4 31B dense-down ragged 65-72 token tails.
    pipeline_mul_mm_q6k_full_dp4a_k21504_n8: ?Pipeline,
    /// K=21504, BN=72 guarded sibling for Gemma 4 31B Q6_K dense-down 65-72 token prompts.
    pipeline_mul_mm_q6k_full_dp4a_k21504_n72: ?Pipeline,
    /// int8 DP4a full-tile Q6_K GEMM that reads a Q8_1 activation layout
    /// (vec2 scale_dsum per 32-block, dsum unused). Used by the Qwen3.6-27B
    /// SSM wqkv prefill projection so it can share the Q8_1 quantize_act
    /// pass with the Q4_K z projection — saves one quantize+barrier per SSM
    /// layer compared to dispatching a separate Q8_0 quantize for wqkv.
    pipeline_mul_mm_q6k_full_dp4a_q8_1: ?Pipeline,
    /// K=5120, BM=64/BN=64 exact SSM wqkv prefill projection.
    pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_bm64: ?Pipeline,
    /// K=5120, BN=64 guarded sibling for dense-hybrid 27B SSM wqkv ragged bodies.
    pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_ragged: ?Pipeline,
    /// K=5120, BN=40 sibling for 33-40 token dense-hybrid 27B SSM wqkv prefill.
    pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n40: ?Pipeline,
    /// One-shot per-32-block activation int8 quantizer for the DP4a down path.
    pipeline_quantize_act_q8: ?Pipeline,
    /// int8 DP4a full-tile Q4_K gate+up+SwiGLU GEMM (Qwen3.6-27B prefill).
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a: ?Pipeline,
    /// Same DP4a full-tile Q4_K gate/up GEMM specialized for Gemma GEGLU.
    pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a: ?Pipeline,
    /// int8 DP4a full-tile Q4_K gate+up+SwiGLU GEMM that fuses Q8_0-style
    /// activation quantize at the output (Qwen3.6-27B prefill), so the
    /// downstream dense-down DP4a kernel can skip the standalone
    /// quantize_act_q8 dispatch + barrier.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8: ?Pipeline,
    /// Same Q8_0 fused-output producer specialized to Gemma's GEGLU activation.
    pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8: ?Pipeline,
    /// BN=64 sibling for Gemma's 96-column padded short-prompt GEGLU producer.
    pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_n64: ?Pipeline,
    /// K=5376 specialization for Gemma 4 31B dense GEGLU gate/up projections.
    pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376: ?Pipeline,
    /// K=5376, BN=64 sibling for Gemma 4 31B's 96-column padded prompt shape.
    pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n64: ?Pipeline,
    /// K=5376, BN=8 sibling for Gemma 4 31B's 65-72 token ragged GEGLU tail.
    pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n8: ?Pipeline,
    /// Single-token Gemma 4 31B dense GEGLU producer. Uses all lanes for N=1
    /// instead of running the BN=8 tiled kernel with seven guarded columns.
    pipeline_mul_mm_q4k_gate_up_geglu_n1_dp4a_q8: ?Pipeline,
    /// K=4096 specialization of the Q8_0 fused gate/up producer for
    /// Qwen3.5-9B dense FFN gate/up projections.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096: ?Pipeline,
    /// K=4096, N-tile=64 specialization for the Qwen3.5-9B 64-token
    /// long-draft prefill shape. It halves repeated Q4_K gate/up weight tile
    /// loads versus the default 32-column tile.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64: ?Pipeline,
    /// K=4096, BN=64 guarded sibling for Qwen3.5-9B long ragged prefill bodies.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64_ragged: ?Pipeline,
    /// K=5120 specialization for Qwen dense-hybrid 27B gate/up projections.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120: ?Pipeline,
    /// K=5120, BN=64 sibling for short prompts padded to a 64-column body.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64: ?Pipeline,
    /// K=5120, BM=64/BN=64 sibling for dense-hybrid 27B 64-token gate/up bodies.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64: ?Pipeline,
    /// Exact 64-token producer that writes K-block-major interleaved Q8 output for dense-down.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64_interleaved: ?Pipeline,
    /// K=5120, BN=64 guarded sibling for long dense-hybrid ragged bodies.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged: ?Pipeline,
    /// K=5120, BM=64/BN=64 guarded sibling for long dense-hybrid ragged bodies.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged_bm64: ?Pipeline,
    /// K=5120, BN=40 sibling for 33-40 token dense-hybrid 27B prompts.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n40: ?Pipeline,
    /// Q8_1 (scale + dsum vec2) sibling of the Q8_0 fused producer. Q4_K-down
    /// needs the dsum bias-correction term; this variant lets the Q4_K-down
    /// DP4a (mul_mm_q4k_full_dp4a) skip its standalone quantize_act_q8_1
    /// dispatch + barrier per Q4_K-down layer.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1: ?Pipeline,
    /// Same Q8_1 fused-output producer specialized to Gemma's GEGLU activation.
    pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1: ?Pipeline,
    /// BN=64 sibling for Gemma's 96-column padded short-prompt GEGLU producer.
    pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_n64: ?Pipeline,
    /// K=5376 specialization for Gemma 4 31B dense GEGLU gate/up projections.
    pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376: ?Pipeline,
    /// K=5376, BN=64 sibling for Gemma 4 31B's 96-column padded prompt shape.
    pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n64: ?Pipeline,
    /// K=5376, BN=8 sibling for Gemma 4 31B's Q4_K-down ragged GEGLU tail.
    pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n8: ?Pipeline,
    /// K=4096 specialization of the Q8_1 fused gate/up producer for
    /// Qwen3.5-9B dense FFN gate/up projections when down is Q4_K.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096: ?Pipeline,
    /// K=4096, N-tile=64 specialization of the Q8_1 fused gate/up producer
    /// for the Qwen3.5-9B 64-token long-draft prefill shape.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64: ?Pipeline,
    /// K=4096, BN=64 guarded sibling for Qwen3.5-9B long ragged Q4_K-down bodies.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64_ragged: ?Pipeline,
    /// K=5120 specialization for Qwen dense-hybrid 27B gate/up projections.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120: ?Pipeline,
    /// K=5120, BN=64 sibling for short prompts padded to a 64-column body.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64: ?Pipeline,
    /// K=5120, BM=64/BN=64 sibling for dense-hybrid 27B 64-token gate/up bodies.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64: ?Pipeline,
    /// Exact 64-token producer that writes K-block-major interleaved Q8_1 output for dense-down.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64_interleaved: ?Pipeline,
    /// K=5120, BN=64 guarded sibling for long dense-hybrid ragged bodies.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged: ?Pipeline,
    /// K=5120, BM=64/BN=64 guarded sibling for long dense-hybrid ragged bodies.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged_bm64: ?Pipeline,
    /// K=5120, BN=40 sibling for 33-40 token dense-hybrid 27B prompts.
    pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n40: ?Pipeline,
    /// int8 DP4a full-tile Q4_K single-projection GEMM (no gate, no SwiGLU)
    /// used by the Qwen3.6-27B SSM z prefill projection. Same Q8_1 activation
    /// layout as the gate+up variant; 4 bindings (A weights, B packed, B
    /// scale_dsum, D f32 out).
    pipeline_mul_mm_q4k_full_dp4a: ?Pipeline,
    /// K=2816, BN=8 sibling for Gemma 26B Q4_K LM-head decode experiments.
    pipeline_mul_mm_q4k_full_dp4a_k2816_n8: ?Pipeline,
    /// K=5376, BN=8 sibling for Gemma 31B Q4_K LM-head decode experiments.
    pipeline_mul_mm_q4k_full_dp4a_k5376_n8: ?Pipeline,
    /// K=5120, BM=64/BN=64 exact SSM z prefill projection.
    pipeline_mul_mm_q4k_full_dp4a_k5120_n64_bm64: ?Pipeline,
    /// K=5120, BN=64 guarded sibling for dense-hybrid 27B SSM z ragged bodies.
    pipeline_mul_mm_q4k_full_dp4a_k5120_n64_ragged: ?Pipeline,
    /// Same Q4_K DP4a shader with K=12288 specialized for Qwen3.5-9B dense
    /// FFN down projections when a layer's down tensor is Q4_K.
    pipeline_mul_mm_q4k_full_dp4a_k12288: ?Pipeline,
    /// K=12288, BN=64 guarded sibling for Qwen3.5-9B long Q4_K dense-down bodies.
    pipeline_mul_mm_q4k_full_dp4a_k12288_n64_ragged: ?Pipeline,
    /// K=12288, BN=64 two-slice guarded sibling for Qwen3.5-9B long Q4_K dense-down bodies.
    pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bk2_ragged: ?Pipeline,
    /// K=12288, BM=64/BN=64 two-slice guarded sibling for Qwen3.5-9B long Q4_K dense-down bodies.
    pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bm64_ragged: ?Pipeline,
    /// K=5120, BN=40 sibling for 33-40 token dense-hybrid 27B SSM z prefill.
    pipeline_mul_mm_q4k_full_dp4a_k5120_n40: ?Pipeline,
    /// K=17408, BN=64 sibling for Qwen dense-hybrid 27B Q4_K dense-down bodies.
    pipeline_mul_mm_q4k_full_dp4a_k17408_n64: ?Pipeline,
    /// K=17408, BN=64 two-slice sibling used only by exact 64-token bodies.
    pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2: ?Pipeline,
    /// K=17408, BM=64/BN=64 sibling for 64-aligned overwrite bodies.
    pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64: ?Pipeline,
    /// K=17408, BM=64/BN=64 two-wave sibling that accumulates exact 64-token bodies.
    pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc: ?Pipeline,
    /// K=17408, BM=64/BN=64 low-register accumulate sibling for exact 64-token bodies.
    pipeline_mul_mm_q4k_full_dp4a_k17408_n64_mmq64_acc: ?Pipeline,
    /// Exact 64-token sibling that consumes K-block-major interleaved Q8_1 activations.
    pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc_interleaved: ?Pipeline,
    /// K=17408, BN=64 two-slice sibling that accumulates into the residual.
    pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2_acc: ?Pipeline,
    /// K=17408, BN=64 guarded sibling for long Q4_K dense-down ragged bodies.
    pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged: ?Pipeline,
    /// K=17408, BM=64/BN=64 guarded sibling for long Q4_K dense-down ragged overwrite bodies.
    pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged_bm64: ?Pipeline,
    /// K=17408, BN=40 sibling for 33-40 token dense-hybrid 27B Q4_K dense-down bodies.
    pipeline_mul_mm_q4k_full_dp4a_k17408_n40: ?Pipeline,
    /// Same Q4_K DP4a shader with K=21504 specialized for Gemma 4 31B dense
    /// FFN down projections when a layer's down tensor is Q4_K.
    pipeline_mul_mm_q4k_full_dp4a_k21504: ?Pipeline,
    /// K=21504, BN=64 sibling for Gemma 4 31B Q4_K dense-down 49-72 token bodies.
    pipeline_mul_mm_q4k_full_dp4a_k21504_n64: ?Pipeline,
    /// K=21504, BN=8 sibling for Gemma 4 31B Q4_K dense-down ragged tails.
    pipeline_mul_mm_q4k_full_dp4a_k21504_n8: ?Pipeline,
    /// int8 DP4a full-tile single Q5_K GEMM for the Qwen3.6-27B SSM out
    /// projection. Same push/binding layout as the Q4_K variant — the only
    /// difference is the 5-bit weight unpack (qs nibble | qh hi bit).
    pipeline_mul_mm_q5k_full_dp4a: ?Pipeline,
    /// K=4096, two-slice sibling for Qwen3.5-9B SSM Q5_K prefill projections.
    pipeline_mul_mm_q5k_full_dp4a_k4096_bk2: ?Pipeline,
    /// K=6144, BN=40 sibling for 33-40 token dense-hybrid 27B SSM out prefill.
    pipeline_mul_mm_q5k_full_dp4a_k6144_n40: ?Pipeline,
    /// Q8_1-style activation quantizer (packed int8 + per-block scale,dsum)
    /// for the DP4a gate/up path's Q4_K bias-correction term.
    pipeline_quantize_act_q8_1: ?Pipeline,
    /// Descriptor pool for this dispatch.
    descriptor_pool: vk.c.VkDescriptorPool,
    /// Logical device.
    device: vk.c.VkDevice,

    /// Create the DMMV dispatch wrapper and load the supported quantized pipelines.
    /// @param instance Active Vulkan instance and logical device.
    /// @param gpu_config Derived GPU tuning parameters (wave size, push-descriptor support).
    /// @param shader_dir Directory containing compiled SPIR-V shader binaries (`.spv` files).
    /// @param hidden_dim Maximum K value used by the Q4_K and F32 shaders' shared-memory array; must be >= hidden_dim, inter_dim, q_dim, and d_inner.
    /// @param allocator Allocator used for temporary pipeline creation state.
    /// @returns A fully-initialised DmmvDispatch ready to record projection work; missing shaders are silently set to null.
    pub fn init(
        /// Vulkan instance.
        instance: *const Instance,
        /// GPU capabilities.
        gpu_config: *const GpuConfig,
        shader_dir: []const u8,
        /// Hidden state width.
        hidden_dim: u32,
        /// Allocator for owned resources.
        allocator: std.mem.Allocator,
    ) !DmmvDispatch {
        const use_wave64 = gpu_config.wave_size == 64;

        // Create descriptor pool
        const pool_size = vk.c.VkDescriptorPoolSize{
            .type = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            // Size for runtime reuse plus rotating hot-bench working sets.
            .descriptorCount = descriptor_pool_max_sets * descriptors_per_set,
        };
        const pool_info = vk.c.VkDescriptorPoolCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.c.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = descriptor_pool_max_sets,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        var descriptor_pool: vk.c.VkDescriptorPool = null;
        if (instance.push_descriptor_fn == null) {
            const result = vk.c.vkCreateDescriptorPool(instance.device, &pool_info, null, &descriptor_pool);
            if (result != vk.c.VK_SUCCESS) return error.DescriptorPoolCreateFailed;
        }

        // Load pipelines (3 bindings: A matrix, x vector, y output)
        const push_size = @sizeOf(DmmvPushConstants);

        // Specialization constant: SPEC_K (id=1) = max_k to size the shared memory
        // array in the Q4_K shader (s_x[SPEC_K]). Must be >= the largest K value
        // used in any Q4_K dispatch (hidden_dim, inter_dim, q_dim, d_inner).
        const spec_k = [_]pipeline_mod.SpecConst{.{ .id = 1, .value = hidden_dim }};
        const has_push_desc = instance.push_descriptor_fn != null;
        const push_desc_options = pipeline_mod.PipelineOptions{
            .push_descriptors = has_push_desc,
        };
        const dp4a_wave32_env = std.posix.getenv("ZINC_DP4A_WAVE32");
        const requested_dp4a_subgroup: u32 = if (dp4a_wave32_env != null and std.mem.eql(u8, dp4a_wave32_env.?, "1")) 32 else 64;
        const push_desc_wave64_options = pipeline_mod.PipelineOptions{
            .required_subgroup_size = requested_dp4a_subgroup,
            .require_full_subgroups = true,
            .push_descriptors = has_push_desc,
        };
        // On non-wave64 GPUs (e.g. Intel Xe2 with SIMD16), requesting
        // required_subgroup_size=64 is outside the supported range and silently
        // dropped by the driver. Use plain push_desc_options instead; the shader
        // multi-subgroup merge path handles the cross-subgroup reduction.
        const effective_wave64_options: pipeline_mod.PipelineOptions = if (use_wave64)
            push_desc_wave64_options
        else
            push_desc_options;

        var path_buf: [512]u8 = undefined;

        const q4k_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k = pipeline_mod.createFromSpirvWithOptions(instance, q4k_path, 3, push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q4_K shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Wide-vocab Q4_K variant (NUM_ROWS=8). Same binding layout — used by
        // the LM-head dispatch on models with vocab ≥ 100_000 where the
        // default NUM_ROWS=2 would spawn hundreds of thousands of workgroups
        // and thrash the L1 cache with redundant hidden-vector reads.
        const q4k_wide_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_wide.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_wide = pipeline_mod.createFromSpirvWithOptions(instance, q4k_wide_path, 3, push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q4_K wide shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q8_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0 = pipeline_mod.createFromSpirvWithOptions(instance, q8_path, 3, push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q8_batch_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_batch = pipeline_mod.createFromSpirvWithOptions(instance, q8_batch_path, 3, push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q8_0 batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q8_kpar_batch_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_kpar_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_kpar_batch = pipeline_mod.createFromSpirvWithOptions(instance, q8_kpar_batch_path, 3, push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q8_0 kpar batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q8_spec64 = [_]pipeline_mod.SpecConst{.{ .id = 2, .value = 64 }};
        const q8_spec64_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_spec64 = pipeline_mod.createFromSpirvWithOptions(instance, q8_spec64_path, 3, push_size, &q8_spec64, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 spec64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q8_spec128 = [_]pipeline_mod.SpecConst{.{ .id = 2, .value = 128 }};
        const q8_spec128_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_spec128 = pipeline_mod.createFromSpirvWithOptions(instance, q8_spec128_path, 3, push_size, &q8_spec128, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 spec128 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q8_wide_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_wide.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_wide = pipeline_mod.createFromSpirvWithOptions(instance, q8_wide_path, 3, push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 wide shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q8_wide4_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_wide4.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_wide4 = pipeline_mod.createFromSpirvWithOptions(instance, q8_wide4_path, 3, push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 wide4 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q8_q81_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_q8_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_q8_1 = pipeline_mod.createFromSpirvWithOptions(instance, q8_q81_path, 3, push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 x Q8_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q4_q81_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_q8_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_q8_1 = pipeline_mod.createFromSpirvWithOptions(instance, q4_q81_path, 3, push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K x Q8_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q8_pair_push_size = @sizeOf(DmmvQ8PairPushConstants);
        const q8_pair_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_fused_pair.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_fused_pair = pipeline_mod.createFromSpirvWithOptions(instance, q8_pair_path, 5, q8_pair_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 fused-pair shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const mxfp4_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_mxfp4.spv", .{shader_dir}) catch unreachable;
        const pipeline_mxfp4 = pipeline_mod.createFromSpirvWithOptions(instance, mxfp4_path, 3, push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("MXFP4 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q5_0_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5_0.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5_0 = pipeline_mod.createFromSpirvWithOptions(instance, q5_0_path, 3, push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_0 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q5_1_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5_1 = pipeline_mod.createFromSpirvWithOptions(instance, q5_1_path, 3, push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q5_1_acc_push_size = @sizeOf(DmmvScaleAccPushConstants);
        const q5_1_acc_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5_1_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5_1_acc = pipeline_mod.createFromSpirvWithOptions(instance, q5_1_acc_path, 3, q5_1_acc_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_1 scaled-acc shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q5k_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5k.spv", .{shader_dir}) catch unreachable;
        // Q5_K K-parallel shader: cross-subgroup shared-memory reduction handles
        // wave64 (1 subgroup), wave32 (2 subgroups), and SIMD16 (4 subgroups).
        const pipeline_q5k = pipeline_mod.createFromSpirvWithOptions(instance, q5k_path, 3, push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_K shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q6k_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q6k.spv", .{shader_dir}) catch unreachable;
        const pipeline_q6k = pipeline_mod.createFromSpirvWithOptions(instance, q6k_path, 3, push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q6_K shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q6k_wide_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q6k_wide.spv", .{shader_dir}) catch unreachable;
        const pipeline_q6k_wide = pipeline_mod.createFromSpirvWithOptions(instance, q6k_wide_path, 3, push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q6_K wide shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const f16_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_f16.spv", .{shader_dir}) catch unreachable;
        const pipeline_f16 = pipeline_mod.createFromSpirvWithOptions(instance, f16_path, 3, push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("F16 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const f32_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_f32.spv", .{shader_dir}) catch unreachable;
        const pipeline_f32 = pipeline_mod.createFromSpirvWithOptions(instance, f32_path, 3, push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("F32 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Batch DMMV for prefill: 3 bindings (A, X_batch, Y_batch), batch push constants
        const batch_push_size = @sizeOf(BatchDmmvPushConstants);
        const q4k_batch_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_batch = pipeline_mod.createFromSpirvWithOptions(instance, q4k_batch_path, 3, batch_push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q4_K batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q6k_batch_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q6k_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_q6k_batch = pipeline_mod.createFromSpirvWithOptions(instance, q6k_batch_path, 3, batch_push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q6_K batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q4k_batch_kpar_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_batch_kpar.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_batch_kpar = pipeline_mod.createFromSpirvWithOptions(instance, q4k_batch_kpar_path, 3, batch_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K batch kpar shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q6k_batch_kpar_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q6k_batch_kpar.spv", .{shader_dir}) catch unreachable;
        const pipeline_q6k_batch_kpar = pipeline_mod.createFromSpirvWithOptions(instance, q6k_batch_kpar_path, 3, batch_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q6_K batch kpar shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // MoE DMMV pipelines: 4 bindings (A, x, y, routing), different push constants
        const moe_push_size = @sizeOf(MoeDmmvPushConstants);

        const q4k_moe_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_moe = pipeline_mod.createFromSpirvWithOptions(instance, q4k_moe_path, 4, moe_push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q4_K MoE shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // K-parallel Q4_K MoE variant: wave64 subgroupAdd reduction, no shared s_x array.
        // Experimental — enabled only when ZINC_MOE_KPAR=1 in forward.zig.
        const q4k_moe_kpar_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_moe_kpar.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_moe_kpar = pipeline_mod.createFromSpirvWithOptions(instance, q4k_moe_kpar_path, 4, moe_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K MoE kpar shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const moe_cols_push_size = @sizeOf(MoeColsDmmvPushConstants);
        const q4k_moe_cols_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_moe_cols.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_moe_cols = pipeline_mod.createFromSpirvWithOptions(instance, q4k_moe_cols_path, 6, moe_cols_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K grouped MoE cols shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_q4k_moe_cols != null) {
            log.info("dmmv_q4k_moe_cols pipeline loaded (grouped top-1 MoE prefill)", .{});
        }
        const q4k_moe_cols_q8_1_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_moe_cols_q8_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_moe_cols_q8_1 = pipeline_mod.createFromSpirvWithOptions(instance, q4k_moe_cols_q8_1_path, 7, moe_cols_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K grouped MoE cols Q8_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_q4k_moe_cols_q8_1 != null) {
            log.info("dmmv_q4k_moe_cols_q8_1 pipeline loaded (grouped top-1 MoE prefill Q8_1 down)", .{});
        }

        // Cross-token batched MoE Q4_K DMMV. Same 4-binding shape as kpar
        // (A, X, Y, routing) but with the larger MoeBatchedDmmvPushConstants
        // struct that adds n_experts_used / x_token_stride / y_token_stride.
        // Dispatch grid is (M+1)/2, n_experts_used, n_tokens.
        const moe_batched_push_size = @sizeOf(MoeBatchedDmmvPushConstants);
        const q4k_moe_batched_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_moe_batched.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_moe_batched = pipeline_mod.createFromSpirvWithOptions(instance, q4k_moe_batched_path, 4, moe_batched_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K MoE batched shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_q4k_moe_batched != null) {
            log.info("dmmv_q4k_moe_batched pipeline loaded (cross-token MoE; not yet wired)", .{});
        }

        // Fused gate+up Q4_K MoE: reads expert_input_buf once per block and
        // writes to both gate_buf and up_buf. 6 bindings (W_gate, W_up, X,
        // Y_gate, Y_up, routing). Same MoeDmmvPushConstants as kpar.
        const q4k_fused_gate_up_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_fused_gate_up_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_fused_gate_up_moe = pipeline_mod.createFromSpirvWithOptions(instance, q4k_fused_gate_up_path, 6, moe_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K MoE fused gate+up shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q4k_fused_gate_up_spec8 = [_]pipeline_mod.SpecConst{.{ .id = 0, .value = 8 }};
        const pipeline_q4k_fused_gate_up_moe_spec8 = pipeline_mod.createFromSpirvWithOptions(instance, q4k_fused_gate_up_path, 6, moe_push_size, &q4k_fused_gate_up_spec8, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K MoE fused gate+up spec8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q4k_fused_gate_up_swiglu_moe_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_fused_gate_up_swiglu_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_fused_gate_up_swiglu_moe = pipeline_mod.createFromSpirvWithOptions(instance, q4k_fused_gate_up_swiglu_moe_path, 5, moe_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K MoE fused gate+up+SwiGLU shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const pipeline_q4k_fused_gate_up_swiglu_moe_spec8 = pipeline_mod.createFromSpirvWithOptions(instance, q4k_fused_gate_up_swiglu_moe_path, 5, moe_push_size, &q4k_fused_gate_up_spec8, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K MoE fused gate+up+SwiGLU spec8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Dense fused gate+up+SwiGLU: reads ffn_norm_buf once per block,
        // dot-products against W_gate and W_up, applies silu(gate)*up
        // inline, and writes a single swiglu_buf row. 4 bindings (W_gate,
        // W_up, X, swiglu). Same DmmvPushConstants as pipeline_q4k.
        const q4k_fused_gate_up_swiglu_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_fused_gate_up_swiglu.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_fused_gate_up_swiglu = pipeline_mod.createFromSpirvWithOptions(instance, q4k_fused_gate_up_swiglu_path, 4, push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K dense fused gate+up+SwiGLU shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q4k_fused_gate_up_swiglu_row1_spec = [_]pipeline_mod.SpecConst{.{ .id = 0, .value = 1 }};
        const pipeline_q4k_fused_gate_up_swiglu_row1 = pipeline_mod.createFromSpirvWithOptions(instance, q4k_fused_gate_up_swiglu_path, 4, push_size, &q4k_fused_gate_up_swiglu_row1_spec, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K dense fused gate+up+SwiGLU row1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q4k_fused_gate_up_geglu_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_fused_gate_up_geglu.spv", .{shader_dir}) catch unreachable;
        const gate_up_geglu_push_size = @sizeOf(DmmvGateUpGegluPushConstants);
        const pipeline_q4k_fused_gate_up_geglu = pipeline_mod.createFromSpirvWithOptions(instance, q4k_fused_gate_up_geglu_path, 3, gate_up_geglu_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K Gemma fused gate+up+GEGLU shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q4k_fused_gate_up_geglu_pair_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_fused_gate_up_geglu_pair.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_fused_gate_up_geglu_pair = pipeline_mod.createFromSpirvWithOptions(instance, q4k_fused_gate_up_geglu_pair_path, 4, push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K Gemma shared fused gate+up+GEGLU pair shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q4k_moe_fused_gate_up_geglu_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_moe_fused_gate_up_geglu.spv", .{shader_dir}) catch unreachable;
        const moe_gate_up_geglu_push_size = @sizeOf(MoeGateUpGegluPushConstants);
        const pipeline_q4k_moe_fused_gate_up_geglu = pipeline_mod.createFromSpirvWithOptions(instance, q4k_moe_fused_gate_up_geglu_path, 4, moe_gate_up_geglu_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K Gemma MoE fused gate+up+GEGLU shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q4k_moe_fused_gate_up_geglu_batch_top1_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_moe_fused_gate_up_geglu_batch_top1.spv", .{shader_dir}) catch unreachable;
        const gemma_top1_batch_push_size = @sizeOf(GemmaTop1MoeBatchPushConstants);
        const pipeline_q4k_moe_fused_gate_up_geglu_batch_top1 = pipeline_mod.createFromSpirvWithOptions(instance, q4k_moe_fused_gate_up_geglu_batch_top1_path, 4, gemma_top1_batch_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K Gemma top-1 batched gate+up+GEGLU shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q4k_moe_fused_gate_up_geglu_cols_top1_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_moe_fused_gate_up_geglu_cols_top1.spv", .{shader_dir}) catch unreachable;
        const gemma_top1_gate_up_cols_push_size = @sizeOf(GemmaTop1GateUpColsPushConstants);
        const pipeline_q4k_moe_fused_gate_up_geglu_cols_top1 = pipeline_mod.createFromSpirvWithOptions(instance, q4k_moe_fused_gate_up_geglu_cols_top1_path, 6, gemma_top1_gate_up_cols_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K Gemma grouped top-1 gate+up+GEGLU shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_q4k_moe_fused_gate_up_geglu_cols_top1 != null) {
            log.info("dmmv_q4k_moe_fused_gate_up_geglu_cols_top1 pipeline loaded (grouped Gemma top-1 MoE prefill gate/up)", .{});
        }
        const q4k_moe_fused_gate_up_swiglu_cols_top1_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_moe_fused_gate_up_swiglu_cols_top1.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1 = pipeline_mod.createFromSpirvWithOptions(instance, q4k_moe_fused_gate_up_swiglu_cols_top1_path, 7, gemma_top1_gate_up_cols_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K Qwen grouped top-1 gate+up+SwiGLU shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1 != null) {
            log.info("dmmv_q4k_moe_fused_gate_up_swiglu_cols_top1 pipeline loaded (grouped Qwen top-1 MoE prefill gate/up)", .{});
        }
        const q4k_moe_fused_gate_up_swiglu_cols_top1_q8_1_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_moe_fused_gate_up_swiglu_cols_top1_q8_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1_q8_1 = pipeline_mod.createFromSpirvWithOptions(instance, q4k_moe_fused_gate_up_swiglu_cols_top1_q8_1_path, 8, gemma_top1_gate_up_cols_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K Qwen grouped top-1 Q8_1 gate+up+SwiGLU shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1_q8_1 != null) {
            log.info("dmmv_q4k_moe_fused_gate_up_swiglu_cols_top1_q8_1 pipeline loaded (grouped Qwen top-1 MoE prefill Q8_1 gate/up)", .{});
        }

        // Q8_0 fused gate+up+SwiGLU. 4 bindings, same push struct as the
        // Q4_K variant. Used by the shared expert path on Qwen 3.5 / 3.6
        // MoE packs where shared FFN weights are Q8_0 (rather than Q4_K).
        const q8_0_fused_gate_up_swiglu_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_fused_gate_up_swiglu.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_fused_gate_up_swiglu = pipeline_mod.createFromSpirvWithOptions(instance, q8_0_fused_gate_up_swiglu_path, 4, push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 fused gate+up+SwiGLU shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q8_0_fused_gate_up_swiglu_gate_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_fused_gate_up_swiglu_gate.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_fused_gate_up_swiglu_gate = pipeline_mod.createFromSpirvWithOptions(instance, q8_0_fused_gate_up_swiglu_gate_path, 6, push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 fused gate+up+SwiGLU+gate shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q8_0_fused_gate_up_geglu_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_fused_gate_up_geglu.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_fused_gate_up_geglu = pipeline_mod.createFromSpirvWithOptions(instance, q8_0_fused_gate_up_geglu_path, 4, push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 fused gate+up+GEGLU shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q8_0_fused_gate_up_geglu4_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_fused_gate_up_geglu4.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_fused_gate_up_geglu4 = pipeline_mod.createFromSpirvWithOptions(instance, q8_0_fused_gate_up_geglu4_path, 4, push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q8_0 fused gate+up+GEGLU4 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Q8_0 DMMV fused with sigmoid-gated scale-accumulate. Replaces the
        // (down_shexp DMMV → barrier → sigmoid_scale_acc) pair on the
        // Qwen 3.5 / 3.6 shared-expert tail. 4 bindings (W, X=gate_buf,
        // Y=hidden_buf, gate=router_logits_buf). Push: DmmvSigmoidAccPushConstants.
        const sigmoid_acc_push_size = @sizeOf(DmmvSigmoidAccPushConstants);
        const q8_0_sigmoid_acc_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_sigmoid_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_sigmoid_acc = pipeline_mod.createFromSpirvWithOptions(instance, q8_0_sigmoid_acc_path, 4, sigmoid_acc_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 fused sigmoid-acc shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Fused split-K merge + Q4_K o_proj DMMV-acc. 4 bindings (W_o,
        // partial_attn_out_buf, hidden_buf, sinks). Push uses
        // OprojMergePushConstants — same DmmvPushConstants prefix plus
        // n_heads, n_i_chunks, sink_offset, head_dim. Used when
        // ZINC_FUSED_OPROJ_MERGE=1 and split-K is active to fold the
        // merge dispatch into o_proj.
        const oproj_merge_push_size = @sizeOf(OprojMergePushConstants);
        const q4k_o_proj_merge_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_o_proj_merge.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_o_proj_merge = pipeline_mod.createFromSpirvWithOptions(instance, q4k_o_proj_merge_path, 4, oproj_merge_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K fused o_proj+merge shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Fused Q4_K MoE down + weighted_acc: each WG accumulates over
        // n_used experts and writes hidden_buf[row] += sum (NUM_ROWS=2).
        // 4 bindings (A, X=swiglu_buf, Y=hidden_buf, routing). Push struct
        // adds n_used (the expert loop is internal to the shader so the
        // dispatch grid drops the Y dim used by kpar).
        const fused_down_acc_push_size = @sizeOf(MoeFusedDownAccPushConstants);
        const q4k_fused_down_acc_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_moe_fused_down_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_moe_fused_down_acc = pipeline_mod.createFromSpirvWithOptions(instance, q4k_fused_down_acc_path, 4, fused_down_acc_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K MoE fused down+acc shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Q5_K analogue of the fused down+acc shader. Targets Q4_K_M / XL
        // packs whose down projection is Q5_K (Qwen3.6-35B-A3B-UD-Q4_K_XL).
        const q5k_fused_down_acc_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5k_moe_fused_down_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5k_moe_fused_down_acc = pipeline_mod.createFromSpirvWithOptions(instance, q5k_fused_down_acc_path, 4, fused_down_acc_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_K MoE fused down+acc shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q5k_fused_down_acc_q81_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5k_moe_fused_down_acc_q8_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5k_moe_fused_down_acc_q8_1 = pipeline_mod.createFromSpirvWithOptions(instance, q5k_fused_down_acc_q81_path, 4, fused_down_acc_push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q5_K MoE fused down+acc Q8_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q5_1_fused_down_acc_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5_1_moe_fused_down_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5_1_moe_fused_down_acc = pipeline_mod.createFromSpirvWithOptions(instance, q5_1_fused_down_acc_path, 4, fused_down_acc_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_1 Gemma MoE fused down+acc shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q5_1_fused_down_acc_scaled_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5_1_moe_fused_down_acc_scaled.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5_1_moe_fused_down_acc_scaled = pipeline_mod.createFromSpirvWithOptions(instance, q5_1_fused_down_acc_scaled_path, 5, fused_down_acc_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_1 Gemma MoE scaled fused down+acc shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q5_1_down_acc_scaled_batch_top1_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5_1_moe_down_acc_scaled_batch_top1.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5_1_moe_down_acc_scaled_batch_top1 = pipeline_mod.createFromSpirvWithOptions(instance, q5_1_down_acc_scaled_batch_top1_path, 5, gemma_top1_batch_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_1 Gemma top-1 batched scaled down+acc shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q8_0_fused_down_acc_scaled_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_moe_fused_down_acc_scaled.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_moe_fused_down_acc_scaled = pipeline_mod.createFromSpirvWithOptions(instance, q8_0_fused_down_acc_scaled_path, 5, fused_down_acc_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 Gemma MoE scaled fused down+acc shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q5k_moe_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5k_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5k_moe = pipeline_mod.createFromSpirvWithOptions(instance, q5k_moe_path, 4, moe_push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q5_K MoE shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // K-parallel Q5_K MoE variant: wave64 subgroupAdd reduction (targets the
        // ~713 ms MoE down bucket in the Qwen3.6-35B flagship prefill).
        const q5k_moe_kpar_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5k_moe_kpar.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5k_moe_kpar = pipeline_mod.createFromSpirvWithOptions(instance, q5k_moe_kpar_path, 4, moe_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_K MoE kpar shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q5k_moe_cols_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5k_moe_cols.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5k_moe_cols = pipeline_mod.createFromSpirvWithOptions(instance, q5k_moe_cols_path, 6, moe_cols_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_K grouped MoE cols shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_q5k_moe_cols != null) {
            log.info("dmmv_q5k_moe_cols pipeline loaded (grouped top-1 MoE prefill)", .{});
        }
        const q5k_moe_cols_q8_1_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5k_moe_cols_q8_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5k_moe_cols_q8_1 = pipeline_mod.createFromSpirvWithOptions(instance, q5k_moe_cols_q8_1_path, 7, moe_cols_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_K grouped MoE cols Q8_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_q5k_moe_cols_q8_1 != null) {
            log.info("dmmv_q5k_moe_cols_q8_1 pipeline loaded (grouped top-1 MoE prefill Q8_1 down)", .{});
        }

        const mxfp4_moe_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_mxfp4_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_mxfp4_moe = pipeline_mod.createFromSpirvWithOptions(instance, mxfp4_moe_path, 4, moe_push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("MXFP4 MoE shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q5_1_moe_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5_1_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5_1_moe = pipeline_mod.createFromSpirvWithOptions(instance, q5_1_moe_path, 4, moe_push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q5_1 MoE shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q5_1_moe_cols_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5_1_moe_cols.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5_1_moe_cols = pipeline_mod.createFromSpirvWithOptions(instance, q5_1_moe_cols_path, 6, moe_cols_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_1 grouped MoE cols shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_q5_1_moe_cols != null) {
            log.info("dmmv_q5_1_moe_cols pipeline loaded (grouped Gemma Q5_1 MoE prefill)", .{});
        }

        const q6k_moe_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q6k_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_q6k_moe = pipeline_mod.createFromSpirvWithOptions(instance, q6k_moe_path, 4, moe_push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q6_K MoE shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const q6k_moe_cols_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q6k_moe_cols.spv", .{shader_dir}) catch unreachable;
        const pipeline_q6k_moe_cols = pipeline_mod.createFromSpirvWithOptions(instance, q6k_moe_cols_path, 6, moe_cols_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q6_K grouped MoE cols shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_q6k_moe_cols != null) {
            log.info("dmmv_q6k_moe_cols pipeline loaded (grouped top-1 MoE prefill)", .{});
        }

        if (pipeline_q4k_moe != null and pipeline_q5k_moe != null and pipeline_q6k_moe != null) {
            log.info("MoE DMMV pipelines loaded — GPU expert dispatch enabled (no readback)", .{});
        }

        // Foundation for mul_mmq: quantize F32 activations into Q8_1 blocks.
        // 2 bindings (A, D), push = QuantizeQ8_1Push {ne, num_blocks}.
        const q81_push_size = @sizeOf(QuantizeQ8_1Push);
        const q81_path = std.fmt.bufPrint(&path_buf, "{s}/quantize_q8_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_quantize_q8_1 = pipeline_mod.createFromSpirvWithOptions(instance, q81_path, 2, q81_push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("quantize_q8_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_quantize_q8_1 != null) {
            log.info("quantize_q8_1 pipeline loaded (mul_mmq foundation)", .{});
        }

        // Effort 6 Step 3: foundation helper for the routed GEMM path.
        // Counts tokens-per-expert from a per-(token, layer) routing buffer.
        // 2 bindings (A routing in, D counts out), push = CountExpertsPush.
        const count_experts_push_size = @sizeOf(CountExpertsPush);
        const count_experts_path = std.fmt.bufPrint(&path_buf, "{s}/count_experts.spv", .{shader_dir}) catch unreachable;
        const pipeline_count_experts = pipeline_mod.createFromSpirvWithOptions(instance, count_experts_path, 2, count_experts_push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("count_experts shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_count_experts != null) {
            log.info("count_experts pipeline loaded (routed GEMM foundation; not yet wired)", .{});
        }

        const moe_route_pack_push_size = @sizeOf(MoeRoutePackPush);
        const moe_route_pack_path = std.fmt.bufPrint(&path_buf, "{s}/moe_route_pack.spv", .{shader_dir}) catch unreachable;
        const pipeline_moe_route_pack = pipeline_mod.createFromSpirvWithOptions(instance, moe_route_pack_path, 6, moe_route_pack_push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("moe_route_pack shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_moe_route_pack != null) {
            log.info("moe_route_pack pipeline loaded (grouped top-1 MoE prefill route blocks)", .{});
        }

        // Effort-6 Step 1 of 5: tiled Q4_K dense GEMM foundation. Single
        // wave64 subgroup per WG (BM=32 × BN=16 output tile), so request
        // wave64 specialization. 3 bindings (A weights, B f32, D f32),
        // push = MulMmQ4KPush.
        const mul_mm_q4k_push_size = @sizeOf(MulMmQ4KPush);
        const mul_mm_q4k_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q4k.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_path, 3, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k != null) {
            log.info("mul_mm_q4k pipeline loaded (tiled Q4_K dense GEMM; LM head opt-in via ZINC_MUL_MM_LM_HEAD=1)", .{});
        }
        const mul_mm_id_q4k_push_size = @sizeOf(MulMmIdQ4KPush);
        const mul_mm_id_q4k_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_id_q4k.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_id_q4k = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_id_q4k_path, 5, mul_mm_id_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_id_q4k shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_id_q4k != null) {
            log.info("mul_mm_id_q4k pipeline loaded (routed Q4_K MoE GEMM foundation; not yet selected)", .{});
        }
        const mul_mm_q4k_down_acc_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q4k_down_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_down_acc = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_down_acc_path, 3, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_down_acc shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_down_acc != null) {
            log.info("mul_mm_q4k_down_acc pipeline loaded (fused-residual Q4_K dense-down GEMM)", .{});
        }
        const mul_mm_q4k_down_acc_wide_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q4k_down_acc_wide.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_down_acc_wide = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_down_acc_wide_path, 3, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_down_acc_wide shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_down_acc_wide != null) {
            log.info("mul_mm_q4k_down_acc_wide pipeline loaded (BN=64 fused-residual Q4_K dense-down GEMM)", .{});
        }
        const mul_mm_q4k_tail8_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q4k_tail8.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_tail8 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_tail8_path, 3, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_tail8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_tail8 != null) {
            log.info("mul_mm_q4k_tail8 pipeline loaded (Gemma 31B Q4_K <=8-token projection tails)", .{});
        }
        const mul_mm_q4k_gate_up_swiglu_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q4k_gate_up_swiglu.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_gate_up_swiglu = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gate_up_swiglu_path, 4, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu != null) {
            log.info("mul_mm_q4k_gate_up_swiglu pipeline loaded (Qwen dense-hybrid 27B batched FFN)", .{});
        }
        const mul_mm_q4k_gate_up_geglu_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q4k_gate_up_geglu.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_gate_up_geglu = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gate_up_geglu_path, 4, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu != null) {
            log.info("mul_mm_q4k_gate_up_geglu pipeline loaded (Gemma 4 batched dense FFN)", .{});
        }
        const mul_mm_q4k_gate_up_geglu_full_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q4k_gate_up_geglu_full.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_gate_up_geglu_full = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gate_up_geglu_full_path, 4, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_full shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_full != null) {
            log.info("mul_mm_q4k_gate_up_geglu_full pipeline loaded (branchless Gemma 4 dense FFN full tiles)", .{});
        }
        const mul_mm_q4k_gate_up_geglu_tail8_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q4k_gate_up_geglu_tail8.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_gate_up_geglu_tail8 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gate_up_geglu_tail8_path, 4, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_tail8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_tail8 != null) {
            log.info("mul_mm_q4k_gate_up_geglu_tail8 pipeline loaded (Gemma 31B Q4_K gate/up <=8-token prefill tails)", .{});
        }
        const mul_mm_q4k_gate_up_swiglu_full_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q4k_gate_up_swiglu_full.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_gate_up_swiglu_full = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gate_up_swiglu_full_path, 4, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full pipeline loaded (branchless Qwen dense-hybrid 27B FFN full tiles)", .{});
        }
        const mul_mm_q6k_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q6k.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q6k = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_path, 3, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k != null) {
            log.info("mul_mm_q6k pipeline loaded (Qwen dense-hybrid 27B batched Q6_K prefill projections)", .{});
        }
        const mul_mm_q6k_full_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q6k_full.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q6k_full = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_path, 3, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full != null) {
            log.info("mul_mm_q6k_full pipeline loaded (branchless full-tile Q6_K prefill GEMM)", .{});
        }
        const mul_mm_q6k_full_down_acc_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q6k_full_down_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q6k_full_down_acc = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_down_acc_path, 3, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_down_acc shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_down_acc != null) {
            log.info("mul_mm_q6k_full_down_acc pipeline loaded (fused-residual Q6_K dense-down GEMM)", .{});
        }
        const mul_mm_q6k_tail8_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q6k_tail8.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q6k_tail8 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_tail8_path, 3, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_tail8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_tail8 != null) {
            log.info("mul_mm_q6k_tail8 pipeline loaded (Gemma 31B Q6_K <=8-token prefill tails)", .{});
        }
        const mul_mm_q5k_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q5k.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q5k = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q5k_path, 3, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q5k shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q5k != null) {
            log.info("mul_mm_q5k pipeline loaded (Qwen dense-hybrid 27B batched Q5_K SSM-out prefill projection)", .{});
        }
        const mul_mm_q5k_wide_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q5k_wide.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q5k_wide = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q5k_wide_path, 3, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q5k_wide shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q5k_wide != null) {
            log.info("mul_mm_q5k_wide pipeline loaded (BN=64 Q5_K SSM prefill projection)", .{});
        }
        const mul_mm_q8_0_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q8_0.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q8_0 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q8_0_path, 3, mul_mm_q4k_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q8_0 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q8_0 != null) {
            log.info("mul_mm_q8_0 pipeline loaded (Qwen3.6 A3B batched Q8_0 SSM-out prefill projection)", .{});
        }
        // int8 DP4a dense-down: 4-binding GEMM (weights/packed-acts/scales/out) and
        // the 3-binding activation quantizer. Both only succeed when the device
        // enabled shaderIntegerDotProduct; otherwise the host path falls back to f32.
        const mul_mm_q8_0_full_dp4a_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q8_0_full_dp4a.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q8_0_full_dp4a = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q8_0_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q8_0_full_dp4a shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q8_0_full_dp4a != null) {
            log.info("mul_mm_q8_0_full_dp4a pipeline loaded (int8 DP4a Q8_0 prefill GEMM)", .{});
        }
        const mul_mm_q6k_full_dp4a_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q6k_full_dp4a.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q6k_full_dp4a = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a != null) {
            log.info("mul_mm_q6k_full_dp4a pipeline loaded (int8 DP4a Qwen dense-hybrid 27B down prefill)", .{});
        }
        const spec_k_4096 = [_]pipeline_mod.SpecConst{.{ .id = 0, .value = 4096 }};
        const spec_k_4096_n64 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 4096 },
            .{ .id = 1, .value = 64 },
        };
        const spec_k_4096_n64_gateup_ragged = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 4096 },
            .{ .id = 1, .value = 64 },
            .{ .id = 3, .value = 1 },
        };
        const spec_k_4096_bk2 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 4096 },
            .{ .id = 3, .value = 2 },
        };
        const spec_k_5120 = [_]pipeline_mod.SpecConst{.{ .id = 0, .value = 5120 }};
        const spec_k_5120_n64 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 5120 },
            .{ .id = 1, .value = 64 },
        };
        const spec_k_5120_n64_interleaved = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 5120 },
            .{ .id = 1, .value = 64 },
            .{ .id = 4, .value = 1 },
        };
        const spec_k_5120_n40 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 5120 },
            .{ .id = 1, .value = 40 },
        };
        const spec_k_5120_n64_ragged = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 5120 },
            .{ .id = 1, .value = 64 },
            .{ .id = 3, .value = 1 },
        };
        const spec_k_5120_n64_q6_q8_1_ragged = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 5120 },
            .{ .id = 1, .value = 64 },
            .{ .id = 2, .value = 1 },
        };
        const spec_k_12288 = [_]pipeline_mod.SpecConst{.{ .id = 0, .value = 12288 }};
        const spec_k_12288_n64_ragged = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 12288 },
            .{ .id = 1, .value = 64 },
            .{ .id = 2, .value = 1 },
        };
        const spec_k_12288_n64_bk2_ragged = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 12288 },
            .{ .id = 1, .value = 64 },
            .{ .id = 2, .value = 1 },
            .{ .id = 3, .value = 2 },
        };
        const spec_k_17408_n64 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 17408 },
            .{ .id = 1, .value = 64 },
        };
        const spec_k_17408_n64_bk2 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 17408 },
            .{ .id = 1, .value = 64 },
            .{ .id = 3, .value = 2 },
        };
        const spec_k_17408_n64_bk2_acc = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 17408 },
            .{ .id = 1, .value = 64 },
            .{ .id = 3, .value = 2 },
            .{ .id = 4, .value = 1 },
        };
        const spec_k_17408_n64_bk2_acc_interleaved = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 17408 },
            .{ .id = 1, .value = 64 },
            .{ .id = 3, .value = 2 },
            .{ .id = 4, .value = 1 },
            .{ .id = 5, .value = 1 },
        };
        const spec_k_17408_mmq64 = [_]pipeline_mod.SpecConst{.{ .id = 0, .value = 17408 }};
        const spec_k_17408_n64_ragged = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 17408 },
            .{ .id = 1, .value = 64 },
            .{ .id = 2, .value = 1 },
        };
        const spec_k_17408_n64_bk2_ragged = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 17408 },
            .{ .id = 1, .value = 64 },
            .{ .id = 2, .value = 1 },
            .{ .id = 3, .value = 2 },
        };
        const spec_k_21504 = [_]pipeline_mod.SpecConst{.{ .id = 0, .value = 21504 }};
        const spec_k_21504_n64 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 21504 },
            .{ .id = 1, .value = 64 },
        };
        const spec_k_21504_n64_bk2 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 21504 },
            .{ .id = 1, .value = 64 },
            .{ .id = 3, .value = 2 },
        };
        const spec_k_2816_n8 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 2816 },
            .{ .id = 1, .value = 8 },
        };
        const spec_k_5376_n8 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 5376 },
            .{ .id = 1, .value = 8 },
        };
        const spec_k_21504_n8 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 21504 },
            .{ .id = 1, .value = 8 },
        };
        const spec_k_21504_n72_ragged = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 21504 },
            .{ .id = 1, .value = 72 },
            .{ .id = 2, .value = 1 },
        };
        // The dense-hybrid 40-token single-projection shapes benefit from
        // staging two K slices per LDS phase. Wider body/tail shapes keep the
        // default one-slice path to avoid excess LDS and register pressure.
        const spec_k_5120_n40_bk2 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 5120 },
            .{ .id = 1, .value = 40 },
            .{ .id = 3, .value = 2 },
        };
        const spec_k_6144_n40_bk2 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 6144 },
            .{ .id = 1, .value = 40 },
            .{ .id = 3, .value = 2 },
        };
        const spec_k_17408_n40_bk2 = [_]pipeline_mod.SpecConst{
            .{ .id = 0, .value = 17408 },
            .{ .id = 1, .value = 40 },
            .{ .id = 3, .value = 2 },
        };
        const geglu_q8_k5376_spec = [_]pipeline_mod.SpecConst{ .{ .id = 0, .value = 5376 }, .{ .id = 2, .value = 1 } };
        const geglu_q8_k5376_n64_spec = [_]pipeline_mod.SpecConst{ .{ .id = 0, .value = 5376 }, .{ .id = 1, .value = 64 }, .{ .id = 2, .value = 1 } };
        const geglu_q8_k5376_n8_spec = [_]pipeline_mod.SpecConst{ .{ .id = 0, .value = 5376 }, .{ .id = 1, .value = 8 }, .{ .id = 2, .value = 1 } };
        const pipeline_mul_mm_q6k_full_dp4a_k12288 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_12288, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=12288 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k12288 != null) {
            log.info("mul_mm_q6k_full_dp4a K=12288 pipeline loaded (Qwen3.5-9B dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k12288_n64_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_12288_n64_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=12288 BN=64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k12288_n64_ragged != null) {
            log.info("mul_mm_q6k_full_dp4a K=12288 BN=64 ragged pipeline loaded (Qwen3.5-9B long dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bk2_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_12288_n64_bk2_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=12288 BN=64 BK2 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bk2_ragged != null) {
            log.info("mul_mm_q6k_full_dp4a K=12288 BN=64 BK2 ragged pipeline loaded (Qwen3.5-9B long dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k17408_n64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_17408_n64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=17408 BN=64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k17408_n64 != null) {
            log.info("mul_mm_q6k_full_dp4a K=17408 BN=64 pipeline loaded (Qwen dense-hybrid 27B 64-token dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_17408_n64_bk2, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=17408 BN=64 BK2 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2 != null) {
            log.info("mul_mm_q6k_full_dp4a K=17408 BN=64 BK2 pipeline loaded (Qwen dense-hybrid 27B exact 64-token dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2_acc = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_17408_n64_bk2_acc, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=17408 BN=64 BK2 accumulate shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2_acc != null) {
            log.info("mul_mm_q6k_full_dp4a K=17408 BN=64 BK2 accumulate pipeline loaded (Qwen dense-hybrid 27B exact 64-token dense-down prefill)", .{});
        }
        var mul_mm_q6k_full_dp4a_bm64_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const mul_mm_q6k_full_dp4a_bm64_path = std.fmt.bufPrint(&mul_mm_q6k_full_dp4a_bm64_path_buf, "{s}/mul_mm_q6k_full_dp4a_bm64_n64_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bm64_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_bm64_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_12288_n64_bk2_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=12288 BM64 BN64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bm64_ragged != null) {
            log.info("mul_mm_q6k_full_dp4a K=12288 BM64 BN64 ragged pipeline loaded (Qwen3.5-9B long dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_bm64_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_17408_n64_bk2, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=17408 BM64 BN64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64 != null) {
            log.info("mul_mm_q6k_full_dp4a K=17408 BM64 BN64 pipeline loaded (Qwen dense-hybrid 27B 64-aligned dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_bm64_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_17408_n64_bk2_acc, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=17408 BM64 BN64 accumulate shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc != null) {
            log.info("mul_mm_q6k_full_dp4a K=17408 BM64 BN64 accumulate pipeline loaded (Qwen dense-hybrid 27B exact 64-token dense-down prefill)", .{});
        }
        var mul_mm_q6k_full_dp4a_mmq64_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const mul_mm_q6k_full_dp4a_mmq64_path = std.fmt.bufPrint(&mul_mm_q6k_full_dp4a_mmq64_path_buf, "{s}/mul_mm_q6k_full_dp4a_mmq64_n64_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q6k_full_dp4a_k17408_n64_mmq64_acc = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_mmq64_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_17408_mmq64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=17408 MMQ64 accumulate shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k17408_n64_mmq64_acc != null) {
            log.info("mul_mm_q6k_full_dp4a K=17408 MMQ64 accumulate pipeline loaded (Qwen dense-hybrid 27B exact 64-token dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc_interleaved = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_bm64_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_17408_n64_bk2_acc_interleaved, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=17408 BM64 BN64 interleaved accumulate shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc_interleaved != null) {
            log.info("mul_mm_q6k_full_dp4a K=17408 BM64 BN64 interleaved accumulate pipeline loaded (Qwen dense-hybrid 27B exact 64-token dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_17408_n64_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=17408 BN=64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged != null) {
            log.info("mul_mm_q6k_full_dp4a K=17408 BN=64 ragged pipeline loaded (Qwen dense-hybrid 27B long dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged_bm64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_bm64_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_17408_n64_bk2_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=17408 BM64 BN64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged_bm64 != null) {
            log.info("mul_mm_q6k_full_dp4a K=17408 BM64 BN64 ragged pipeline loaded (Qwen dense-hybrid 27B long dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k17408_n40 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_17408_n40_bk2, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=17408 BN=40 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k17408_n40 != null) {
            log.info("mul_mm_q6k_full_dp4a K=17408 BN=40 pipeline loaded (Qwen dense-hybrid 27B 40-token dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k21504 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_21504, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=21504 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k21504 != null) {
            log.info("mul_mm_q6k_full_dp4a K=21504 pipeline loaded (Gemma 4 31B dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k21504_n64_bm64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_bm64_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_21504_n64_bk2, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=21504 BM64 BN64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k21504_n64_bm64 != null) {
            log.info("mul_mm_q6k_full_dp4a K=21504 BM64 BN64 pipeline loaded (Gemma 4 31B Q6_K dense-down 64-token prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k21504_n64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_21504_n64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=21504 BN=64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k21504_n64 != null) {
            log.info("mul_mm_q6k_full_dp4a K=21504 BN=64 pipeline loaded (Gemma 4 31B Q6_K dense-down 64-token prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k21504_n8 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_21504_n8, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=21504 BN=8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k21504_n8 != null) {
            log.info("mul_mm_q6k_full_dp4a K=21504 BN=8 pipeline loaded (Gemma 4 31B dense-down ragged tail)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_k21504_n72 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_21504_n72_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a K=21504 BN=72 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_k21504_n72 != null) {
            log.info("mul_mm_q6k_full_dp4a K=21504 BN=72 ragged pipeline loaded (Gemma 4 31B Q6_K dense-down 65-72 token prompts)", .{});
        }
        // Q8_1-input variant of the Q6_K DP4a GEMM (vec2 scale_dsum, dsum
        // unused). Lets the Qwen3.6-27B SSM wqkv projection share a single
        // Q8_1 quantize of scratch_norm with the Q4_K z projection, saving one
        // quantize_act + barrier per SSM layer. Same 4-binding layout and push
        // constants as the Q8_0-input variant; the only difference is the SclB
        // binding type.
        const mul_mm_q6k_full_dp4a_q8_1_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q6k_full_dp4a_q8_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q6k_full_dp4a_q8_1 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_q8_1_path, 4, @sizeOf(MulMmQ6KDp4aPush), &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a_q8_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_q8_1 != null) {
            log.info("mul_mm_q6k_full_dp4a_q8_1 pipeline loaded (int8 DP4a Qwen dense-hybrid 27B SSM wqkv prefill, Q8_1 input)", .{});
        }
        var mul_mm_q6k_full_dp4a_q8_1_bm64_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const mul_mm_q6k_full_dp4a_q8_1_bm64_path = std.fmt.bufPrint(&mul_mm_q6k_full_dp4a_q8_1_bm64_path_buf, "{s}/mul_mm_q6k_full_dp4a_q8_1_bm64_n64.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_bm64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_q8_1_bm64_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_5120_n64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a_q8_1 K=5120 BM64 BN64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_bm64 != null) {
            log.info("mul_mm_q6k_full_dp4a_q8_1 K=5120 BM64 BN64 pipeline loaded (Qwen dense-hybrid 27B SSM wqkv prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_q8_1_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_5120_n64_q6_q8_1_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a_q8_1 K=5120 BN=64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_ragged != null) {
            log.info("mul_mm_q6k_full_dp4a_q8_1 K=5120 BN=64 ragged pipeline loaded (Qwen dense-hybrid 27B SSM wqkv prefill)", .{});
        }
        const pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n40 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q6k_full_dp4a_q8_1_path, 4, @sizeOf(MulMmQ6KDp4aPush), &spec_k_5120_n40_bk2, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q6k_full_dp4a_q8_1 K=5120 BN=40 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n40 != null) {
            log.info("mul_mm_q6k_full_dp4a_q8_1 K=5120 BN=40 pipeline loaded (Qwen dense-hybrid 27B 40-token SSM wqkv prefill)", .{});
        }
        const quantize_act_q8_path = std.fmt.bufPrint(&path_buf, "{s}/quantize_act_q8.spv", .{shader_dir}) catch unreachable;
        const pipeline_quantize_act_q8 = pipeline_mod.createFromSpirvWithOptions(instance, quantize_act_q8_path, 3, @sizeOf(QuantizeActPush), &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("quantize_act_q8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_quantize_act_q8 != null) {
            log.info("quantize_act_q8 pipeline loaded (DP4a activation quantizer)", .{});
        }
        // int8 DP4a dense gate+up+SwiGLU (Qwen3.6-27B prefill). 5-binding GEMM
        // (gate weights, up weights, packed-acts, (scale,dsum), out) and the
        // 3-binding Q8_1-style activation quantizer.
        const mul_mm_q4k_gateup_dp4a_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q4k_gate_up_swiglu_full_dp4a.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_path, 5, @sizeOf(MulMmQ4KGateUpDp4aPush), &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a pipeline loaded (int8 DP4a Qwen dense-hybrid 27B gate+up prefill)", .{});
        }
        const geglu_activation_spec = [_]pipeline_mod.SpecConst{.{ .id = 0, .value = 1 }};
        const pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_path, 5, @sizeOf(MulMmQ4KGateUpDp4aPush), &geglu_activation_spec, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_full_dp4a shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a != null) {
            log.info("mul_mm_q4k_gate_up_geglu_full_dp4a pipeline loaded (int8 DP4a Gemma dense gate+up prefill)", .{});
        }
        // Variant that emits Q8_0-style packed activation directly, so the
        // downstream dense-down DP4a kernel can skip the standalone
        // quantize_act_q8 dispatch + barrier. Same gate/up SSBOs and Q8_1 input
        // activation as the f32-output variant; replaces the single f32 output
        // buffer with (packed int8, scale) pair.
        const mul_mm_q4k_gateup_dp4a_q8_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q4k_gate_up_swiglu_full_dp4a_q8.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 pipeline loaded (int8 DP4a Qwen dense-hybrid 27B gate+up prefill with fused Q8_0 output)", .{});
        }
        const geglu_q8_activation_spec = [_]pipeline_mod.SpecConst{.{ .id = 2, .value = 1 }};
        const geglu_q8_n64_spec = [_]pipeline_mod.SpecConst{ .{ .id = 1, .value = 64 }, .{ .id = 2, .value = 1 } };
        const pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &geglu_q8_activation_spec, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_full_dp4a_q8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8 != null) {
            log.info("mul_mm_q4k_gate_up_geglu_full_dp4a_q8 pipeline loaded (int8 DP4a Gemma dense gate+up prefill with fused Q8_0 output)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_n64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &geglu_q8_n64_spec, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_full_dp4a_q8 BN=64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_n64 != null) {
            log.info("mul_mm_q4k_gate_up_geglu_full_dp4a_q8 BN=64 pipeline loaded (Gemma 31B 96-column dense gate+up prefill split)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &geglu_q8_k5376_spec, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_full_dp4a_q8 K=5376 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376 != null) {
            log.info("mul_mm_q4k_gate_up_geglu_full_dp4a_q8 K=5376 pipeline loaded (Gemma 4 31B dense gate+up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &geglu_q8_k5376_n64_spec, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_full_dp4a_q8 K=5376 N64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n64 != null) {
            log.info("mul_mm_q4k_gate_up_geglu_full_dp4a_q8 K=5376 N64 pipeline loaded (Gemma 4 31B 64-token dense gate+up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n8 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &geglu_q8_k5376_n8_spec, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_full_dp4a_q8 K=5376 N8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n8 != null) {
            log.info("mul_mm_q4k_gate_up_geglu_full_dp4a_q8 K=5376 N8 pipeline loaded (Gemma 4 31B dense GEGLU ragged tail)", .{});
        }
        var mul_mm_q4k_gateup_geglu_n1_dp4a_q8_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const mul_mm_q4k_gateup_geglu_n1_dp4a_q8_path = std.fmt.bufPrint(&mul_mm_q4k_gateup_geglu_n1_dp4a_q8_path_buf, "{s}/mul_mm_q4k_gate_up_geglu_n1_dp4a_q8.spv", .{shader_dir}) catch unreachable;
        const geglu_n1_k5376_spec = [_]pipeline_mod.SpecConst{.{ .id = 0, .value = 5376 }};
        const pipeline_mul_mm_q4k_gate_up_geglu_n1_dp4a_q8 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_geglu_n1_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &geglu_n1_k5376_spec, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_n1_dp4a_q8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_n1_dp4a_q8 != null) {
            log.info("mul_mm_q4k_gate_up_geglu_n1_dp4a_q8 pipeline loaded (Gemma 4 31B single-token dense GEGLU)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_4096, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=4096 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=4096 pipeline loaded (Qwen3.5-9B dense gate+up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_4096_n64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=4096 N64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=4096 N64 pipeline loaded (Qwen3.5-9B 64-token dense gate+up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_4096_n64_gateup_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=4096 N64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64_ragged != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=4096 N64 ragged pipeline loaded (Qwen3.5-9B long dense gate+up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 pipeline loaded (Qwen dense-hybrid 27B gate/up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120_n64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 N64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 N64 pipeline loaded (Qwen dense-hybrid 27B 64-token gate/up prefill)", .{});
        }
        var mul_mm_q4k_gateup_dp4a_q8_bm64_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const mul_mm_q4k_gateup_dp4a_q8_bm64_path = std.fmt.bufPrint(&mul_mm_q4k_gateup_dp4a_q8_bm64_path_buf, "{s}/mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_bm64.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_bm64_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120_n64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 BM64 N64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 BM64 N64 pipeline loaded (Qwen dense-hybrid 27B 64-token gate/up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64_interleaved = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_bm64_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120_n64_interleaved, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 BM64 N64 interleaved shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64_interleaved != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 BM64 N64 interleaved pipeline loaded (Qwen dense-hybrid 27B 64-token gate/up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120_n64_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 N64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 N64 ragged pipeline loaded (Qwen dense-hybrid 27B long gate/up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged_bm64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_bm64_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120_n64_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 BM64 N64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged_bm64 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 BM64 N64 ragged pipeline loaded (Qwen dense-hybrid 27B long gate/up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n40 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120_n40, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 N40 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n40 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 K=5120 N40 pipeline loaded (Qwen dense-hybrid 27B 40-token gate/up prefill)", .{});
        }
        // Q4_K-down sibling of the Q8_0 fused producer. Same gate/up SSBOs,
        // same Q8_1 input activation; the only differences are the output
        // scale buffer (vec2 scale+dsum instead of f32 scale) and the
        // per-cluster isum reduction so the dsum bias-correction term is
        // valid for the downstream mul_mm_q4k_full_dp4a (Q4_K-down) kernel.
        const mul_mm_q4k_gateup_dp4a_q8_1_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 pipeline loaded (int8 DP4a Qwen dense-hybrid 27B gate+up prefill with fused Q8_1 output for Q4_K-down)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &geglu_q8_activation_spec, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1 != null) {
            log.info("mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1 pipeline loaded (int8 DP4a Gemma dense gate+up prefill with fused Q8_1 output for Q4_K-down)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_n64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &geglu_q8_n64_spec, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1 BN=64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_n64 != null) {
            log.info("mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1 BN=64 pipeline loaded (Gemma 31B 96-column dense gate+up prefill split)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &geglu_q8_k5376_spec, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1 K=5376 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376 != null) {
            log.info("mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1 K=5376 pipeline loaded (Gemma 4 31B dense gate+up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &geglu_q8_k5376_n64_spec, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1 K=5376 N64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n64 != null) {
            log.info("mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1 K=5376 N64 pipeline loaded (Gemma 4 31B 64-token dense gate+up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n8 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &geglu_q8_k5376_n8_spec, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1 K=5376 N8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n8 != null) {
            log.info("mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1 K=5376 N8 pipeline loaded (Gemma 4 31B Q4_K-down ragged GEGLU tail)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_4096, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=4096 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=4096 pipeline loaded (Qwen3.5-9B dense gate+up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_4096_n64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=4096 N64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=4096 N64 pipeline loaded (Qwen3.5-9B 64-token dense gate+up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_4096_n64_gateup_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=4096 N64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64_ragged != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=4096 N64 ragged pipeline loaded (Qwen3.5-9B long dense gate+up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 pipeline loaded (Qwen dense-hybrid 27B gate/up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120_n64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 N64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 N64 pipeline loaded (Qwen dense-hybrid 27B 64-token gate/up prefill)", .{});
        }
        var mul_mm_q4k_gateup_dp4a_q8_1_bm64_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const mul_mm_q4k_gateup_dp4a_q8_1_bm64_path = std.fmt.bufPrint(&mul_mm_q4k_gateup_dp4a_q8_1_bm64_path_buf, "{s}/mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_bm64.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_bm64_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120_n64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 BM64 N64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 BM64 N64 pipeline loaded (Qwen dense-hybrid 27B 64-token gate/up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64_interleaved = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_bm64_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120_n64_interleaved, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 BM64 N64 interleaved shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64_interleaved != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 BM64 N64 interleaved pipeline loaded (Qwen dense-hybrid 27B 64-token gate/up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120_n64_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 N64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 N64 ragged pipeline loaded (Qwen dense-hybrid 27B long gate/up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged_bm64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_bm64_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120_n64_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 BM64 N64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged_bm64 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 BM64 N64 ragged pipeline loaded (Qwen dense-hybrid 27B long gate/up prefill)", .{});
        }
        const pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n40 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_gateup_dp4a_q8_1_path, 6, @sizeOf(MulMmQ4KGateUpDp4aQ8Push), &spec_k_5120_n40, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 N40 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n40 != null) {
            log.info("mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 K=5120 N40 pipeline loaded (Qwen dense-hybrid 27B 40-token gate/up prefill)", .{});
        }
        const quantize_act_q8_1_path = std.fmt.bufPrint(&path_buf, "{s}/quantize_act_q8_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_quantize_act_q8_1 = pipeline_mod.createFromSpirvWithOptions(instance, quantize_act_q8_1_path, 3, @sizeOf(QuantizeActPush), &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("quantize_act_q8_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_quantize_act_q8_1 != null) {
            log.info("quantize_act_q8_1 pipeline loaded (DP4a activation quantizer w/ dsum)", .{});
        }
        // int8 DP4a single Q4_K projection for the Qwen3.6-27B SSM z prefill
        // path (M=d_inner, K=hidden_dim). Reuses MulMmQ4KGateUpDp4aPush layout
        // (M/N/K + Q8_1 strides + a_offset/d_offset). 4 bindings.
        const mul_mm_q4k_full_dp4a_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q4k_full_dp4a.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_full_dp4a = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a != null) {
            log.info("mul_mm_q4k_full_dp4a pipeline loaded (int8 DP4a Qwen dense-hybrid 27B SSM z prefill)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k2816_n8 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_2816_n8, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=2816 BN=8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k2816_n8 != null) {
            log.info("mul_mm_q4k_full_dp4a K=2816 BN=8 pipeline loaded (Gemma 26B Q4_K LM-head decode)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k5376_n8 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_5376_n8, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=5376 BN=8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k5376_n8 != null) {
            log.info("mul_mm_q4k_full_dp4a K=5376 BN=8 pipeline loaded (Gemma 31B Q4_K LM-head decode)", .{});
        }
        var mul_mm_q4k_full_dp4a_bm64_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const mul_mm_q4k_full_dp4a_bm64_path = std.fmt.bufPrint(&mul_mm_q4k_full_dp4a_bm64_path_buf, "{s}/mul_mm_q4k_full_dp4a_bm64_n64_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_full_dp4a_k5120_n64_bm64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_bm64_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_5120_n64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=5120 BM64 BN64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k5120_n64_bm64 != null) {
            log.info("mul_mm_q4k_full_dp4a K=5120 BM64 BN64 pipeline loaded (Qwen dense-hybrid 27B SSM z prefill)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k5120_n64_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_5120_n64_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=5120 BN=64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k5120_n64_ragged != null) {
            log.info("mul_mm_q4k_full_dp4a K=5120 BN=64 ragged pipeline loaded (Qwen dense-hybrid 27B SSM z prefill)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k5120_n40 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_5120_n40_bk2, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=5120 BN=40 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k5120_n40 != null) {
            log.info("mul_mm_q4k_full_dp4a K=5120 BN=40 pipeline loaded (Qwen dense-hybrid 27B SSM z 40-token prefill)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k12288 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_12288, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=12288 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k12288 != null) {
            log.info("mul_mm_q4k_full_dp4a K=12288 pipeline loaded (Qwen3.5-9B dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k12288_n64_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_12288_n64_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=12288 BN=64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k12288_n64_ragged != null) {
            log.info("mul_mm_q4k_full_dp4a K=12288 BN=64 ragged pipeline loaded (Qwen3.5-9B long dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bk2_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_12288_n64_bk2_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=12288 BN=64 BK2 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bk2_ragged != null) {
            log.info("mul_mm_q4k_full_dp4a K=12288 BN=64 BK2 ragged pipeline loaded (Qwen3.5-9B long dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bm64_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_bm64_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_12288_n64_bk2_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=12288 BM64 BN64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bm64_ragged != null) {
            log.info("mul_mm_q4k_full_dp4a K=12288 BM64 BN64 ragged pipeline loaded (Qwen3.5-9B long dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k17408_n64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_17408_n64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=17408 BN=64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k17408_n64 != null) {
            log.info("mul_mm_q4k_full_dp4a K=17408 BN=64 pipeline loaded (Qwen dense-hybrid 27B Q4_K dense-down 64-token bodies)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_17408_n64_bk2, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=17408 BN=64 BK2 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2 != null) {
            log.info("mul_mm_q4k_full_dp4a K=17408 BN=64 BK2 pipeline loaded (Qwen dense-hybrid 27B Q4_K dense-down exact 64-token body)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2_acc = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_17408_n64_bk2_acc, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=17408 BN=64 BK2 accumulate shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2_acc != null) {
            log.info("mul_mm_q4k_full_dp4a K=17408 BN=64 BK2 accumulate pipeline loaded (Qwen dense-hybrid 27B Q4_K dense-down exact 64-token body)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_bm64_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_17408_n64_bk2, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=17408 BM64 BN64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64 != null) {
            log.info("mul_mm_q4k_full_dp4a K=17408 BM64 BN64 pipeline loaded (Qwen dense-hybrid 27B Q4_K dense-down 64-aligned bodies)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_bm64_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_17408_n64_bk2_acc, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=17408 BM64 BN64 accumulate shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc != null) {
            log.info("mul_mm_q4k_full_dp4a K=17408 BM64 BN64 accumulate pipeline loaded (Qwen dense-hybrid 27B Q4_K dense-down exact 64-token body)", .{});
        }
        var mul_mm_q4k_full_dp4a_mmq64_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const mul_mm_q4k_full_dp4a_mmq64_path = std.fmt.bufPrint(&mul_mm_q4k_full_dp4a_mmq64_path_buf, "{s}/mul_mm_q4k_full_dp4a_mmq64_n64_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q4k_full_dp4a_k17408_n64_mmq64_acc = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_mmq64_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_17408_mmq64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=17408 MMQ64 accumulate shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k17408_n64_mmq64_acc != null) {
            log.info("mul_mm_q4k_full_dp4a K=17408 MMQ64 accumulate pipeline loaded (Qwen dense-hybrid 27B Q4_K dense-down exact 64-token body)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc_interleaved = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_bm64_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_17408_n64_bk2_acc_interleaved, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=17408 BM64 BN64 interleaved accumulate shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc_interleaved != null) {
            log.info("mul_mm_q4k_full_dp4a K=17408 BM64 BN64 interleaved accumulate pipeline loaded (Qwen dense-hybrid 27B Q4_K dense-down exact 64-token body)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_17408_n64_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=17408 BN=64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged != null) {
            log.info("mul_mm_q4k_full_dp4a K=17408 BN=64 ragged pipeline loaded (Qwen dense-hybrid 27B Q4_K dense-down long bodies)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged_bm64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_bm64_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_17408_n64_bk2_ragged, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=17408 BM64 BN64 ragged shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged_bm64 != null) {
            log.info("mul_mm_q4k_full_dp4a K=17408 BM64 BN64 ragged pipeline loaded (Qwen dense-hybrid 27B Q4_K dense-down long bodies)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k17408_n40 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_17408_n40_bk2, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=17408 BN=40 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k17408_n40 != null) {
            log.info("mul_mm_q4k_full_dp4a K=17408 BN=40 pipeline loaded (Qwen dense-hybrid 27B Q4_K dense-down 40-token bodies)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k21504 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_21504, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=21504 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k21504 != null) {
            log.info("mul_mm_q4k_full_dp4a K=21504 pipeline loaded (Gemma 4 31B dense-down prefill)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k21504_n64 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_21504_n64, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=21504 BN=64 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k21504_n64 != null) {
            log.info("mul_mm_q4k_full_dp4a K=21504 BN=64 pipeline loaded (Gemma 4 31B Q4_K dense-down 64-token bodies)", .{});
        }
        const pipeline_mul_mm_q4k_full_dp4a_k21504_n8 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q4k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_21504_n8, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q4k_full_dp4a K=21504 BN=8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q4k_full_dp4a_k21504_n8 != null) {
            log.info("mul_mm_q4k_full_dp4a K=21504 BN=8 pipeline loaded (Gemma 4 31B Q4_K dense-down ragged tail)", .{});
        }
        // int8 DP4a single Q5_K projection for the Qwen3.6-27B SSM out prefill
        // path (M=hidden_dim, K=d_inner). Same push layout as the Q4_K variant
        // (the 5-bit weight unpack is the only kernel-side difference). 4 bindings.
        const mul_mm_q5k_full_dp4a_path = std.fmt.bufPrint(&path_buf, "{s}/mul_mm_q5k_full_dp4a.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_mm_q5k_full_dp4a = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q5k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q5k_full_dp4a shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q5k_full_dp4a != null) {
            log.info("mul_mm_q5k_full_dp4a pipeline loaded (int8 DP4a Qwen dense-hybrid 27B SSM out prefill)", .{});
        }
        const pipeline_mul_mm_q5k_full_dp4a_k4096_bk2 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q5k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_4096_bk2, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q5k_full_dp4a K=4096 BK2 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q5k_full_dp4a_k4096_bk2 != null) {
            log.info("mul_mm_q5k_full_dp4a K=4096 BK2 pipeline loaded (Qwen3.5-9B SSM Q5_K prefill)", .{});
        }
        const pipeline_mul_mm_q5k_full_dp4a_k6144_n40 = pipeline_mod.createFromSpirvWithOptions(instance, mul_mm_q5k_full_dp4a_path, 4, @sizeOf(MulMmQ4KGateUpDp4aPush), &spec_k_6144_n40_bk2, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("mul_mm_q5k_full_dp4a K=6144 BN=40 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_mul_mm_q5k_full_dp4a_k6144_n40 != null) {
            log.info("mul_mm_q5k_full_dp4a K=6144 BN=40 pipeline loaded (Qwen dense-hybrid 27B SSM out 40-token prefill)", .{});
        }
        return DmmvDispatch{
            .pipeline_q4k = pipeline_q4k,
            .pipeline_q4k_wide = pipeline_q4k_wide,
            .pipeline_mxfp4 = pipeline_mxfp4,
            .pipeline_q5_0 = pipeline_q5_0,
            .pipeline_q5_1 = pipeline_q5_1,
            .pipeline_q5k = pipeline_q5k,
            .pipeline_q6k = pipeline_q6k,
            .pipeline_q6k_wide = pipeline_q6k_wide,
            .pipeline_q8_0 = pipeline_q8_0,
            .pipeline_q8_0_batch = pipeline_q8_0_batch,
            .pipeline_q8_0_kpar_batch = pipeline_q8_0_kpar_batch,
            .pipeline_q8_0_spec64 = pipeline_q8_0_spec64,
            .pipeline_q8_0_spec128 = pipeline_q8_0_spec128,
            .pipeline_q8_0_wide = pipeline_q8_0_wide,
            .pipeline_q8_0_wide4 = pipeline_q8_0_wide4,
            .pipeline_q8_0_q8_1 = pipeline_q8_0_q8_1,
            .pipeline_q4k_q8_1 = pipeline_q4k_q8_1,
            .pipeline_q8_0_fused_pair = pipeline_q8_0_fused_pair,
            .pipeline_f16 = pipeline_f16,
            .pipeline_f32 = pipeline_f32,
            .pipeline_q4k_batch = pipeline_q4k_batch,
            .pipeline_q4k_batch_kpar = pipeline_q4k_batch_kpar,
            .pipeline_q6k_batch = pipeline_q6k_batch,
            .pipeline_q6k_batch_kpar = pipeline_q6k_batch_kpar,
            .pipeline_q4k_moe = pipeline_q4k_moe,
            .pipeline_q4k_moe_kpar = pipeline_q4k_moe_kpar,
            .pipeline_q4k_moe_cols = pipeline_q4k_moe_cols,
            .pipeline_q4k_moe_cols_q8_1 = pipeline_q4k_moe_cols_q8_1,
            .pipeline_q4k_moe_batched = pipeline_q4k_moe_batched,
            .pipeline_q4k_fused_gate_up_moe = pipeline_q4k_fused_gate_up_moe,
            .pipeline_q4k_fused_gate_up_moe_spec8 = pipeline_q4k_fused_gate_up_moe_spec8,
            .pipeline_q4k_fused_gate_up_swiglu_moe = pipeline_q4k_fused_gate_up_swiglu_moe,
            .pipeline_q4k_fused_gate_up_swiglu_moe_spec8 = pipeline_q4k_fused_gate_up_swiglu_moe_spec8,
            .pipeline_q4k_fused_gate_up_swiglu = pipeline_q4k_fused_gate_up_swiglu,
            .pipeline_q4k_fused_gate_up_swiglu_row1 = pipeline_q4k_fused_gate_up_swiglu_row1,
            .pipeline_q4k_fused_gate_up_geglu = pipeline_q4k_fused_gate_up_geglu,
            .pipeline_q4k_fused_gate_up_geglu_pair = pipeline_q4k_fused_gate_up_geglu_pair,
            .pipeline_q4k_moe_fused_gate_up_geglu = pipeline_q4k_moe_fused_gate_up_geglu,
            .pipeline_q4k_moe_fused_gate_up_geglu_batch_top1 = pipeline_q4k_moe_fused_gate_up_geglu_batch_top1,
            .pipeline_q4k_moe_fused_gate_up_geglu_cols_top1 = pipeline_q4k_moe_fused_gate_up_geglu_cols_top1,
            .pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1 = pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1,
            .pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1_q8_1 = pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1_q8_1,
            .pipeline_q8_0_fused_gate_up_swiglu = pipeline_q8_0_fused_gate_up_swiglu,
            .pipeline_q8_0_fused_gate_up_swiglu_gate = pipeline_q8_0_fused_gate_up_swiglu_gate,
            .pipeline_q8_0_fused_gate_up_geglu = pipeline_q8_0_fused_gate_up_geglu,
            .pipeline_q8_0_fused_gate_up_geglu4 = pipeline_q8_0_fused_gate_up_geglu4,
            .pipeline_q8_0_sigmoid_acc = pipeline_q8_0_sigmoid_acc,
            .pipeline_q4k_o_proj_merge = pipeline_q4k_o_proj_merge,
            .pipeline_q4k_moe_fused_down_acc = pipeline_q4k_moe_fused_down_acc,
            .pipeline_q5k_moe_fused_down_acc = pipeline_q5k_moe_fused_down_acc,
            .pipeline_q5k_moe_fused_down_acc_q8_1 = pipeline_q5k_moe_fused_down_acc_q8_1,
            .pipeline_mxfp4_moe = pipeline_mxfp4_moe,
            .pipeline_q5_1_moe = pipeline_q5_1_moe,
            .pipeline_q5_1_moe_cols = pipeline_q5_1_moe_cols,
            .pipeline_q5_1_acc = pipeline_q5_1_acc,
            .pipeline_q5_1_moe_fused_down_acc = pipeline_q5_1_moe_fused_down_acc,
            .pipeline_q5_1_moe_fused_down_acc_scaled = pipeline_q5_1_moe_fused_down_acc_scaled,
            .pipeline_q5_1_moe_down_acc_scaled_batch_top1 = pipeline_q5_1_moe_down_acc_scaled_batch_top1,
            .pipeline_q8_0_moe_fused_down_acc_scaled = pipeline_q8_0_moe_fused_down_acc_scaled,
            .pipeline_q5k_moe = pipeline_q5k_moe,
            .pipeline_q5k_moe_kpar = pipeline_q5k_moe_kpar,
            .pipeline_q5k_moe_cols = pipeline_q5k_moe_cols,
            .pipeline_q5k_moe_cols_q8_1 = pipeline_q5k_moe_cols_q8_1,
            .pipeline_q6k_moe = pipeline_q6k_moe,
            .pipeline_q6k_moe_cols = pipeline_q6k_moe_cols,
            .pipeline_quantize_q8_1 = pipeline_quantize_q8_1,
            .pipeline_count_experts = pipeline_count_experts,
            .pipeline_moe_route_pack = pipeline_moe_route_pack,
            .pipeline_mul_mm_q4k = pipeline_mul_mm_q4k,
            .pipeline_mul_mm_id_q4k = pipeline_mul_mm_id_q4k,
            .pipeline_mul_mm_q4k_down_acc = pipeline_mul_mm_q4k_down_acc,
            .pipeline_mul_mm_q4k_down_acc_wide = pipeline_mul_mm_q4k_down_acc_wide,
            .pipeline_mul_mm_q4k_tail8 = pipeline_mul_mm_q4k_tail8,
            .pipeline_mul_mm_q4k_gate_up_swiglu = pipeline_mul_mm_q4k_gate_up_swiglu,
            .pipeline_mul_mm_q4k_gate_up_geglu = pipeline_mul_mm_q4k_gate_up_geglu,
            .pipeline_mul_mm_q4k_gate_up_geglu_full = pipeline_mul_mm_q4k_gate_up_geglu_full,
            .pipeline_mul_mm_q4k_gate_up_geglu_tail8 = pipeline_mul_mm_q4k_gate_up_geglu_tail8,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full = pipeline_mul_mm_q4k_gate_up_swiglu_full,
            .pipeline_mul_mm_q6k = pipeline_mul_mm_q6k,
            .pipeline_mul_mm_q6k_full = pipeline_mul_mm_q6k_full,
            .pipeline_mul_mm_q6k_full_down_acc = pipeline_mul_mm_q6k_full_down_acc,
            .pipeline_mul_mm_q6k_tail8 = pipeline_mul_mm_q6k_tail8,
            .pipeline_mul_mm_q6k_full_dp4a = pipeline_mul_mm_q6k_full_dp4a,
            .pipeline_mul_mm_q6k_full_dp4a_k12288 = pipeline_mul_mm_q6k_full_dp4a_k12288,
            .pipeline_mul_mm_q6k_full_dp4a_k12288_n64_ragged = pipeline_mul_mm_q6k_full_dp4a_k12288_n64_ragged,
            .pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bk2_ragged = pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bk2_ragged,
            .pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bm64_ragged = pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bm64_ragged,
            .pipeline_mul_mm_q6k_full_dp4a_k17408_n64 = pipeline_mul_mm_q6k_full_dp4a_k17408_n64,
            .pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2 = pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2,
            .pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64 = pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64,
            .pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc = pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc,
            .pipeline_mul_mm_q6k_full_dp4a_k17408_n64_mmq64_acc = pipeline_mul_mm_q6k_full_dp4a_k17408_n64_mmq64_acc,
            .pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc_interleaved = pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc_interleaved,
            .pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2_acc = pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2_acc,
            .pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged = pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged,
            .pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged_bm64 = pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged_bm64,
            .pipeline_mul_mm_q6k_full_dp4a_k17408_n40 = pipeline_mul_mm_q6k_full_dp4a_k17408_n40,
            .pipeline_mul_mm_q6k_full_dp4a_k21504 = pipeline_mul_mm_q6k_full_dp4a_k21504,
            .pipeline_mul_mm_q6k_full_dp4a_k21504_n64_bm64 = pipeline_mul_mm_q6k_full_dp4a_k21504_n64_bm64,
            .pipeline_mul_mm_q6k_full_dp4a_k21504_n64 = pipeline_mul_mm_q6k_full_dp4a_k21504_n64,
            .pipeline_mul_mm_q6k_full_dp4a_k21504_n8 = pipeline_mul_mm_q6k_full_dp4a_k21504_n8,
            .pipeline_mul_mm_q6k_full_dp4a_k21504_n72 = pipeline_mul_mm_q6k_full_dp4a_k21504_n72,
            .pipeline_mul_mm_q6k_full_dp4a_q8_1 = pipeline_mul_mm_q6k_full_dp4a_q8_1,
            .pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_bm64 = pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_bm64,
            .pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_ragged = pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_ragged,
            .pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n40 = pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n40,
            .pipeline_quantize_act_q8 = pipeline_quantize_act_q8,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a,
            .pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a = pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8,
            .pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8 = pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8,
            .pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_n64 = pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_n64,
            .pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376 = pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376,
            .pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n64 = pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n64,
            .pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n8 = pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n8,
            .pipeline_mul_mm_q4k_gate_up_geglu_n1_dp4a_q8 = pipeline_mul_mm_q4k_gate_up_geglu_n1_dp4a_q8,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64_ragged = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64_ragged,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64_interleaved = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64_interleaved,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged_bm64 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged_bm64,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n40 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n40,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1,
            .pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1 = pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1,
            .pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_n64 = pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_n64,
            .pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376 = pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376,
            .pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n64 = pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n64,
            .pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n8 = pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n8,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64_ragged = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64_ragged,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64_interleaved = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64_interleaved,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged_bm64 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged_bm64,
            .pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n40 = pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n40,
            .pipeline_mul_mm_q4k_full_dp4a = pipeline_mul_mm_q4k_full_dp4a,
            .pipeline_mul_mm_q4k_full_dp4a_k2816_n8 = pipeline_mul_mm_q4k_full_dp4a_k2816_n8,
            .pipeline_mul_mm_q4k_full_dp4a_k5376_n8 = pipeline_mul_mm_q4k_full_dp4a_k5376_n8,
            .pipeline_mul_mm_q4k_full_dp4a_k5120_n64_bm64 = pipeline_mul_mm_q4k_full_dp4a_k5120_n64_bm64,
            .pipeline_mul_mm_q4k_full_dp4a_k5120_n64_ragged = pipeline_mul_mm_q4k_full_dp4a_k5120_n64_ragged,
            .pipeline_mul_mm_q4k_full_dp4a_k12288 = pipeline_mul_mm_q4k_full_dp4a_k12288,
            .pipeline_mul_mm_q4k_full_dp4a_k12288_n64_ragged = pipeline_mul_mm_q4k_full_dp4a_k12288_n64_ragged,
            .pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bk2_ragged = pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bk2_ragged,
            .pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bm64_ragged = pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bm64_ragged,
            .pipeline_mul_mm_q4k_full_dp4a_k5120_n40 = pipeline_mul_mm_q4k_full_dp4a_k5120_n40,
            .pipeline_mul_mm_q4k_full_dp4a_k17408_n64 = pipeline_mul_mm_q4k_full_dp4a_k17408_n64,
            .pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2 = pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2,
            .pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64 = pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64,
            .pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc = pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc,
            .pipeline_mul_mm_q4k_full_dp4a_k17408_n64_mmq64_acc = pipeline_mul_mm_q4k_full_dp4a_k17408_n64_mmq64_acc,
            .pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc_interleaved = pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc_interleaved,
            .pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2_acc = pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2_acc,
            .pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged = pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged,
            .pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged_bm64 = pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged_bm64,
            .pipeline_mul_mm_q4k_full_dp4a_k17408_n40 = pipeline_mul_mm_q4k_full_dp4a_k17408_n40,
            .pipeline_mul_mm_q4k_full_dp4a_k21504 = pipeline_mul_mm_q4k_full_dp4a_k21504,
            .pipeline_mul_mm_q4k_full_dp4a_k21504_n64 = pipeline_mul_mm_q4k_full_dp4a_k21504_n64,
            .pipeline_mul_mm_q4k_full_dp4a_k21504_n8 = pipeline_mul_mm_q4k_full_dp4a_k21504_n8,
            .pipeline_mul_mm_q5k_full_dp4a = pipeline_mul_mm_q5k_full_dp4a,
            .pipeline_mul_mm_q5k_full_dp4a_k4096_bk2 = pipeline_mul_mm_q5k_full_dp4a_k4096_bk2,
            .pipeline_mul_mm_q5k_full_dp4a_k6144_n40 = pipeline_mul_mm_q5k_full_dp4a_k6144_n40,
            .pipeline_quantize_act_q8_1 = pipeline_quantize_act_q8_1,
            .pipeline_mul_mm_q5k = pipeline_mul_mm_q5k,
            .pipeline_mul_mm_q5k_wide = pipeline_mul_mm_q5k_wide,
            .pipeline_mul_mm_q8_0 = pipeline_mul_mm_q8_0,
            .pipeline_mul_mm_q8_0_full_dp4a = pipeline_mul_mm_q8_0_full_dp4a,
            .descriptor_pool = descriptor_pool,
            .device = instance.device,
        };
    }

    /// Select the quantization-specific pipeline used for a weight matrix format.
    /// @param self Dispatch wrapper containing the loaded DMMV pipelines.
    /// @param quant_type GGML quantization format for the weight matrix.
    /// @returns A pipeline pointer when that quantization format has a loaded shader implementation.
    /// @note Unsupported or unloaded formats return `null` so callers can surface `error.UnsupportedQuantType`.
    pub fn pipelineForType(self: *const DmmvDispatch, quant_type: GGMLType) ?*const Pipeline {
        return switch (quant_type) {
            .q4_k => if (self.pipeline_q4k) |*p| p else null,
            .mxfp4 => if (self.pipeline_mxfp4) |*p| p else null,
            .q5_0 => if (self.pipeline_q5_0) |*p| p else null,
            .q5_1 => if (self.pipeline_q5_1) |*p| p else null,
            .q5_k => if (self.pipeline_q5k) |*p| p else null,
            .q6_k => if (self.pipeline_q6k) |*p| p else null,
            .q8_0 => if (self.pipeline_q8_0) |*p| p else null,
            .f16 => if (self.pipeline_f16) |*p| p else null,
            .f32 => if (self.pipeline_f32) |*p| p else null,
            else => null,
        };
    }

    /// Select the MoE-specific pipeline for the given weight format (4 bindings: A, x, y, routing).
    /// @param self Dispatch wrapper containing the loaded DMMV pipelines.
    /// @param quant_type GGML quantization format for the MoE expert weight matrix.
    /// @returns A pipeline pointer when a MoE shader is loaded for the format, or null for unsupported types.
    pub fn moePipelineForType(self: *const DmmvDispatch, quant_type: GGMLType) ?*const Pipeline {
        return switch (quant_type) {
            .q4_k => if (self.pipeline_q4k_moe) |*p| p else null,
            .mxfp4 => if (self.pipeline_mxfp4_moe) |*p| p else null,
            .q5_1 => if (self.pipeline_q5_1_moe) |*p| p else null,
            .q5_k => if (self.pipeline_q5k_moe) |*p| p else null,
            .q6_k => if (self.pipeline_q6k_moe) |*p| p else null,
            else => null,
        };
    }

    /// Record a batched MoE DMMV dispatch where all experts run in parallel via Y workgroups.
    /// @param cmd Command buffer to record into.
    /// @param quant_type Weight quantization; resolved via `moePipelineForType`.
    /// @param descriptor_set Descriptor set with bindings A, x, y, routing.
    /// @param M Output rows (weight rows per expert).
    /// @param K Contraction width (shared across all experts).
    /// @param expert_stride Byte stride between consecutive experts in the stacked weight tensor.
    /// @param n_experts_y Number of experts to process; becomes the Y workgroup dimension.
    /// @param x_expert_stride Element stride between consecutive experts' input vectors (0 = shared input; K = per-expert).
    /// @param x_offset Element offset into the input buffer.
    /// @param y_offset Element offset into the output buffer.
    /// @returns `error.UnsupportedQuantType` when no MoE pipeline is loaded for `quant_type`.
    pub fn recordMoeDispatch(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        quant_type: GGMLType,
        descriptor_set: vk.c.VkDescriptorSet,
        M: u32,
        K: u32,
        expert_stride: u32,
        n_experts_y: u32,
        x_expert_stride: u32,
        x_offset: u32,
        y_offset: u32,
    ) !void {
        const pip = self.moePipelineForType(quant_type) orelse return error.UnsupportedQuantType;

        const push = MoeDmmvPushConstants{
            .M = M,
            .K = K,
            .expert_stride = expert_stride,
            .x_expert_stride = x_expert_stride,
            .x_offset = x_offset,
            .y_offset = y_offset,
        };

        const workgroups_x = switch (quant_type) {
            .mxfp4, .q8_0, .f16 => (M + 1) / 2,
            else => (M + 63) / 64,
        };

        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups_x, n_experts_y, 1);
    }

    /// Record a cross-token batched MoE DMMV dispatch. Reads N tokens'
    /// inputs from `X_batch[N × K]`, dispatches one WG per (row_pair,
    /// expert_slot, token_idx), routes via flattened
    /// `routing[N × n_experts_used]`, writes to
    /// `Y_batch[N × n_experts_used × M]`. Q4_K only for now.
    pub fn recordMoeBatchedDispatch(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        x_buf: vk.c.VkBuffer,
        x_size: vk.c.VkDeviceSize,
        y_buf: vk.c.VkBuffer,
        y_size: vk.c.VkDeviceSize,
        routing_buf: vk.c.VkBuffer,
        routing_size: vk.c.VkDeviceSize,
        M: u32,
        K: u32,
        expert_stride: u32,
        n_experts_used: u32,
        n_tokens: u32,
        x_token_stride: u32,
        y_token_stride: u32,
    ) !void {
        const pip = if (self.pipeline_q4k_moe_batched) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        const push = MoeBatchedDmmvPushConstants{
            .M = M,
            .K = K,
            .expert_stride = expert_stride,
            .n_experts_used = n_experts_used,
            .x_token_stride = x_token_stride,
            .y_token_stride = y_token_stride,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = x_buf, .offset = 0, .range = x_size },
            .{ .buffer = y_buf, .offset = 0, .range = y_size },
            .{ .buffer = routing_buf, .offset = 0, .range = routing_size },
        };
        const wg_x = (M + 1) / 2;
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            wg_x,
            n_experts_used,
            n_tokens,
        );
    }

    /// Record the single-workgroup `moe_route_pack` dispatch that compacts per-token
    /// expert routing into expert-grouped work lists for the MoE columns kernels.
    ///
    /// Reads `routing_buf`/`ids_buf` and writes the per-expert `counts_buf`, the packed
    /// `active_blocks_buf` + `active_count_buf`, and `dispatch_args_buf` for the
    /// subsequent indirect MoE dispatch. Each buffer is paired with its byte size.
    /// @param cmd Command buffer to record into.
    /// @param push_desc_fn Optional push-descriptor function (null uses bound descriptor sets).
    /// @param n_tokens Number of tokens being routed.
    /// @param n_experts Total expert count in the layer.
    /// @param k Experts selected per token (top-k).
    /// @param routing_stride Per-token stride (elements) into the routing buffer.
    /// @param ids_stride Per-token stride into the packed IDs buffer.
    /// @param gate_up_workgroups_x Workgroups-X to encode into the gate/up columns dispatch args.
    /// @param down_workgroups_x Workgroups-X to encode into the down columns dispatch args.
    /// @param routing_token_base First token offset within the routing buffer.
    /// @returns `error.PipelineNotLoaded` if the route-pack pipeline is absent, or
    ///   `error.InvalidArgument` on a zero token/expert/k/stride/workgroup count.
    pub fn recordMoeRoutePack(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        routing_buf: vk.c.VkBuffer,
        routing_size: vk.c.VkDeviceSize,
        counts_buf: vk.c.VkBuffer,
        counts_size: vk.c.VkDeviceSize,
        ids_buf: vk.c.VkBuffer,
        ids_size: vk.c.VkDeviceSize,
        active_count_buf: vk.c.VkBuffer,
        active_count_size: vk.c.VkDeviceSize,
        active_blocks_buf: vk.c.VkBuffer,
        active_blocks_size: vk.c.VkDeviceSize,
        dispatch_args_buf: vk.c.VkBuffer,
        dispatch_args_size: vk.c.VkDeviceSize,
        n_tokens: u32,
        n_experts: u32,
        k: u32,
        routing_stride: u32,
        ids_stride: u32,
        gate_up_workgroups_x: u32,
        down_workgroups_x: u32,
        routing_token_base: u32,
    ) !void {
        const pip = if (self.pipeline_moe_route_pack) |*p| p else return error.PipelineNotLoaded;
        if (n_tokens == 0 or n_experts == 0 or k == 0 or ids_stride == 0) return error.InvalidArgument;
        if (gate_up_workgroups_x == 0 or down_workgroups_x == 0) return error.InvalidArgument;
        const push = MoeRoutePackPush{
            .n_tokens = n_tokens,
            .n_experts = n_experts,
            .k = k,
            .routing_stride = routing_stride,
            .ids_stride = ids_stride,
            .gate_up_workgroups_x = gate_up_workgroups_x,
            .down_workgroups_x = down_workgroups_x,
            .routing_token_base = routing_token_base,
        };
        const infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = routing_buf, .offset = 0, .range = routing_size },
            .{ .buffer = counts_buf, .offset = 0, .range = counts_size },
            .{ .buffer = ids_buf, .offset = 0, .range = ids_size },
            .{ .buffer = active_count_buf, .offset = 0, .range = active_count_size },
            .{ .buffer = active_blocks_buf, .offset = 0, .range = active_blocks_size },
            .{ .buffer = dispatch_args_buf, .offset = 0, .range = dispatch_args_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            1,
            1,
            1,
        );
    }

    /// Record an **indirect** MoE columns DMMV — the per-expert weight×activation
    /// matvec whose workgroup count is read from `indirect_buf` rather than the host.
    ///
    /// Indirect form: the active-block count is GPU-resident (produced by
    /// `recordMoeRoutePack`), so the host never stalls on a readback. Each buffer is
    /// paired with its byte size.
    /// @param cmd Command buffer to record into.
    /// @param push_desc_fn Optional push-descriptor function (null uses bound descriptor sets).
    /// @param quant_type Weight quantization; supports K-quant columns plus Gemma Q5_1 down rows.
    /// @param indirect_buf Buffer holding the `VkDispatchIndirectCommand` workgroup dims.
    /// @param indirect_offset Byte offset of the dispatch args within `indirect_buf`.
    /// @param M Output rows (expert weight rows).
    /// @param K Contraction width; must be a multiple of 256.
    /// @param expert_stride Per-expert stride (elements) into the weight buffer.
    /// @param ids_stride Per-token stride into the packed IDs buffer.
    /// @param x_route_divisor Divisor mapping output rows back to source activation rows.
    /// @param accumulate When true, add into `y_buf` instead of overwriting it.
    /// @returns `error.UnsupportedQuantType` for unsupported weights, or
    ///   `error.InvalidArgument` on zero M/K/ids_stride or incompatible K alignment.
    pub fn recordMoeColsDispatchIndirect(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        quant_type: GGMLType,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        x_buf: vk.c.VkBuffer,
        x_size: vk.c.VkDeviceSize,
        y_buf: vk.c.VkBuffer,
        y_size: vk.c.VkDeviceSize,
        counts_buf: vk.c.VkBuffer,
        counts_size: vk.c.VkDeviceSize,
        ids_buf: vk.c.VkBuffer,
        ids_size: vk.c.VkDeviceSize,
        active_blocks_buf: vk.c.VkBuffer,
        active_blocks_size: vk.c.VkDeviceSize,
        indirect_buf: vk.c.VkBuffer,
        indirect_offset: vk.c.VkDeviceSize,
        M: u32,
        K: u32,
        expert_stride: u32,
        ids_stride: u32,
        x_route_divisor: u32,
        a_offset: u32,
        x_offset: u32,
        y_offset: u32,
        accumulate: bool,
    ) !void {
        const pip = switch (quant_type) {
            .q4_k => if (self.pipeline_q4k_moe_cols) |*p| p else return error.UnsupportedQuantType,
            .q5_k => if (self.pipeline_q5k_moe_cols) |*p| p else return error.UnsupportedQuantType,
            .q5_1 => if (self.pipeline_q5_1_moe_cols) |*p| p else return error.UnsupportedQuantType,
            .q6_k => if (self.pipeline_q6k_moe_cols) |*p| p else return error.UnsupportedQuantType,
            else => return error.UnsupportedQuantType,
        };
        if (M == 0 or K == 0 or ids_stride == 0) return error.InvalidArgument;
        if (!moeColsKAligned(quant_type, K)) return error.InvalidArgument;
        const push = MoeColsDmmvPushConstants{
            .M = M,
            .K = K,
            .a_offset = a_offset,
            .expert_stride = expert_stride,
            .x_offset = x_offset,
            .y_offset = y_offset,
            .ids_stride = ids_stride,
            .x_route_divisor = @max(x_route_divisor, 1),
            .accumulate = if (accumulate) 1 else 0,
        };
        const infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = x_buf, .offset = 0, .range = x_size },
            .{ .buffer = y_buf, .offset = 0, .range = y_size },
            .{ .buffer = counts_buf, .offset = 0, .range = counts_size },
            .{ .buffer = ids_buf, .offset = 0, .range = ids_size },
            .{ .buffer = active_blocks_buf, .offset = 0, .range = active_blocks_size },
        };
        cmd.pushDescAndDispatchIndirect(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            indirect_buf,
            indirect_offset,
        );
    }

    pub fn recordMoeColsQ8_1DispatchIndirect(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        quant_type: GGMLType,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        x_packed_buf: vk.c.VkBuffer,
        x_packed_size: vk.c.VkDeviceSize,
        x_scale_dsum_buf: vk.c.VkBuffer,
        x_scale_dsum_size: vk.c.VkDeviceSize,
        y_buf: vk.c.VkBuffer,
        y_size: vk.c.VkDeviceSize,
        counts_buf: vk.c.VkBuffer,
        counts_size: vk.c.VkDeviceSize,
        ids_buf: vk.c.VkBuffer,
        ids_size: vk.c.VkDeviceSize,
        active_blocks_buf: vk.c.VkBuffer,
        active_blocks_size: vk.c.VkDeviceSize,
        indirect_buf: vk.c.VkBuffer,
        indirect_offset: vk.c.VkDeviceSize,
        M: u32,
        K: u32,
        expert_stride: u32,
        ids_stride: u32,
        x_route_divisor: u32,
        a_offset: u32,
        x_offset: u32,
        y_offset: u32,
        accumulate: bool,
    ) !void {
        const pip = switch (quant_type) {
            .q4_k => if (self.pipeline_q4k_moe_cols_q8_1) |*p| p else return error.UnsupportedQuantType,
            .q5_k => if (self.pipeline_q5k_moe_cols_q8_1) |*p| p else return error.UnsupportedQuantType,
            else => return error.UnsupportedQuantType,
        };
        if (M == 0 or K == 0 or ids_stride == 0) return error.InvalidArgument;
        if ((K & 255) != 0) return error.InvalidArgument;
        const push = MoeColsDmmvPushConstants{
            .M = M,
            .K = K,
            .a_offset = a_offset,
            .expert_stride = expert_stride,
            .x_offset = x_offset,
            .y_offset = y_offset,
            .ids_stride = ids_stride,
            .x_route_divisor = @max(x_route_divisor, 1),
            .accumulate = if (accumulate) 1 else 0,
        };
        const infos = [7]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = x_packed_buf, .offset = 0, .range = x_packed_size },
            .{ .buffer = x_scale_dsum_buf, .offset = 0, .range = x_scale_dsum_size },
            .{ .buffer = y_buf, .offset = 0, .range = y_size },
            .{ .buffer = counts_buf, .offset = 0, .range = counts_size },
            .{ .buffer = ids_buf, .offset = 0, .range = ids_size },
            .{ .buffer = active_blocks_buf, .offset = 0, .range = active_blocks_size },
        };
        cmd.pushDescAndDispatchIndirect(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            indirect_buf,
            indirect_offset,
        );
    }

    pub fn recordGemmaTop1GateUpGegluColsDispatchIndirect(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        x_buf: vk.c.VkBuffer,
        x_size: vk.c.VkDeviceSize,
        y_buf: vk.c.VkBuffer,
        y_size: vk.c.VkDeviceSize,
        counts_buf: vk.c.VkBuffer,
        counts_size: vk.c.VkDeviceSize,
        ids_buf: vk.c.VkBuffer,
        ids_size: vk.c.VkDeviceSize,
        active_blocks_buf: vk.c.VkBuffer,
        active_blocks_size: vk.c.VkDeviceSize,
        indirect_buf: vk.c.VkBuffer,
        indirect_offset: vk.c.VkDeviceSize,
        M: u32,
        K: u32,
        expert_stride: u32,
        up_offset: u32,
        ids_stride: u32,
        x_route_divisor: u32,
        a_offset: u32,
        y_offset: u32,
    ) !void {
        const pip = if (self.pipeline_q4k_moe_fused_gate_up_geglu_cols_top1) |*p| p else return error.PipelineNotLoaded;
        if (M == 0 or K == 0 or ids_stride == 0) return error.InvalidArgument;
        if ((K & 255) != 0) return error.InvalidArgument;
        const push = GemmaTop1GateUpColsPushConstants{
            .M = M,
            .K = K,
            .a_offset = a_offset,
            .expert_stride = expert_stride,
            .up_offset = up_offset,
            .y_offset = y_offset,
            .ids_stride = ids_stride,
            .x_route_divisor = @max(x_route_divisor, 1),
            .x_token_base = 0,
        };
        const infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = x_buf, .offset = 0, .range = x_size },
            .{ .buffer = y_buf, .offset = 0, .range = y_size },
            .{ .buffer = counts_buf, .offset = 0, .range = counts_size },
            .{ .buffer = ids_buf, .offset = 0, .range = ids_size },
            .{ .buffer = active_blocks_buf, .offset = 0, .range = active_blocks_size },
        };
        cmd.pushDescAndDispatchIndirect(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            indirect_buf,
            indirect_offset,
        );
    }

    pub fn recordQwenTop1GateUpSwigluColsDispatchIndirect(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        x_buf: vk.c.VkBuffer,
        x_size: vk.c.VkDeviceSize,
        y_buf: vk.c.VkBuffer,
        y_size: vk.c.VkDeviceSize,
        counts_buf: vk.c.VkBuffer,
        counts_size: vk.c.VkDeviceSize,
        ids_buf: vk.c.VkBuffer,
        ids_size: vk.c.VkDeviceSize,
        active_blocks_buf: vk.c.VkBuffer,
        active_blocks_size: vk.c.VkDeviceSize,
        indirect_buf: vk.c.VkBuffer,
        indirect_offset: vk.c.VkDeviceSize,
        M: u32,
        K: u32,
        expert_stride: u32,
        ids_stride: u32,
        x_route_divisor: u32,
        x_token_base: u32,
        a_offset: u32,
        y_offset: u32,
    ) !void {
        const pip = if (self.pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1) |*p| p else return error.PipelineNotLoaded;
        if (M == 0 or K == 0 or ids_stride == 0) return error.InvalidArgument;
        if ((K & 255) != 0) return error.InvalidArgument;
        const push = GemmaTop1GateUpColsPushConstants{
            .M = M,
            .K = K,
            .a_offset = a_offset,
            .expert_stride = expert_stride,
            .up_offset = 0,
            .y_offset = y_offset,
            .ids_stride = ids_stride,
            .x_route_divisor = @max(x_route_divisor, 1),
            .x_token_base = x_token_base,
        };
        const infos = [7]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = x_buf, .offset = 0, .range = x_size },
            .{ .buffer = y_buf, .offset = 0, .range = y_size },
            .{ .buffer = counts_buf, .offset = 0, .range = counts_size },
            .{ .buffer = ids_buf, .offset = 0, .range = ids_size },
            .{ .buffer = active_blocks_buf, .offset = 0, .range = active_blocks_size },
        };
        cmd.pushDescAndDispatchIndirect(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            indirect_buf,
            indirect_offset,
        );
    }

    pub fn recordQwenTop1GateUpSwigluColsQ8_1DispatchIndirect(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        x_packed_buf: vk.c.VkBuffer,
        x_packed_size: vk.c.VkDeviceSize,
        x_scale_dsum_buf: vk.c.VkBuffer,
        x_scale_dsum_size: vk.c.VkDeviceSize,
        y_buf: vk.c.VkBuffer,
        y_size: vk.c.VkDeviceSize,
        counts_buf: vk.c.VkBuffer,
        counts_size: vk.c.VkDeviceSize,
        ids_buf: vk.c.VkBuffer,
        ids_size: vk.c.VkDeviceSize,
        active_blocks_buf: vk.c.VkBuffer,
        active_blocks_size: vk.c.VkDeviceSize,
        indirect_buf: vk.c.VkBuffer,
        indirect_offset: vk.c.VkDeviceSize,
        M: u32,
        K: u32,
        expert_stride: u32,
        ids_stride: u32,
        x_route_divisor: u32,
        x_token_base: u32,
        a_offset: u32,
        y_offset: u32,
    ) !void {
        const pip = if (self.pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1_q8_1) |*p| p else return error.PipelineNotLoaded;
        if (M == 0 or K == 0 or ids_stride == 0) return error.InvalidArgument;
        if ((K & 255) != 0) return error.InvalidArgument;
        const push = GemmaTop1GateUpColsPushConstants{
            .M = M,
            .K = K,
            .a_offset = a_offset,
            .expert_stride = expert_stride,
            .up_offset = 0,
            .y_offset = y_offset,
            .ids_stride = ids_stride,
            .x_route_divisor = @max(x_route_divisor, 1),
            .x_token_base = x_token_base,
        };
        const infos = [8]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = x_packed_buf, .offset = 0, .range = x_packed_size },
            .{ .buffer = x_scale_dsum_buf, .offset = 0, .range = x_scale_dsum_size },
            .{ .buffer = y_buf, .offset = 0, .range = y_size },
            .{ .buffer = counts_buf, .offset = 0, .range = counts_size },
            .{ .buffer = ids_buf, .offset = 0, .range = ids_size },
            .{ .buffer = active_blocks_buf, .offset = 0, .range = active_blocks_size },
        };
        cmd.pushDescAndDispatchIndirect(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            indirect_buf,
            indirect_offset,
        );
    }

    /// Record a **direct** MoE columns DMMV using a host-known `active_block_count`
    /// (the non-indirect sibling of `recordMoeColsDispatchIndirect`).
    ///
    /// Prefer this when the active block count is already known on the CPU; otherwise
    /// use the indirect form to avoid a GPU→CPU readback. This variant always
    /// overwrites `y_buf` (no accumulate). Each buffer is paired with its byte size.
    /// @param cmd Command buffer to record into.
    /// @param push_desc_fn Optional push-descriptor function (null uses bound descriptor sets).
    /// @param quant_type Weight quantization; supports route-packed K-quant columns plus Gemma Q5_1 down rows.
    /// @param M Output rows (expert weight rows).
    /// @param K Contraction width; must match the quantization block alignment.
    /// @param expert_stride Per-expert stride (elements) into the weight buffer.
    /// @param active_block_count Number of active expert blocks to dispatch (workgroups-Y).
    /// @param ids_stride Per-token stride into the packed IDs buffer.
    /// @param x_route_divisor Divisor mapping output rows back to source activation rows.
    /// @returns `error.UnsupportedQuantType` for unsupported weights, or
    ///   `error.InvalidArgument` on zero M/K/active_block_count/ids_stride or incompatible K alignment.
    pub fn recordMoeColsDispatch(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        quant_type: GGMLType,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        x_buf: vk.c.VkBuffer,
        x_size: vk.c.VkDeviceSize,
        y_buf: vk.c.VkBuffer,
        y_size: vk.c.VkDeviceSize,
        counts_buf: vk.c.VkBuffer,
        counts_size: vk.c.VkDeviceSize,
        ids_buf: vk.c.VkBuffer,
        ids_size: vk.c.VkDeviceSize,
        active_blocks_buf: vk.c.VkBuffer,
        active_blocks_size: vk.c.VkDeviceSize,
        M: u32,
        K: u32,
        expert_stride: u32,
        active_block_count: u32,
        ids_stride: u32,
        x_route_divisor: u32,
        a_offset: u32,
        x_offset: u32,
        y_offset: u32,
    ) !void {
        const pip = switch (quant_type) {
            .q4_k => if (self.pipeline_q4k_moe_cols) |*p| p else return error.UnsupportedQuantType,
            .q5_k => if (self.pipeline_q5k_moe_cols) |*p| p else return error.UnsupportedQuantType,
            .q5_1 => if (self.pipeline_q5_1_moe_cols) |*p| p else return error.UnsupportedQuantType,
            .q6_k => if (self.pipeline_q6k_moe_cols) |*p| p else return error.UnsupportedQuantType,
            else => return error.UnsupportedQuantType,
        };
        if (M == 0 or K == 0 or active_block_count == 0 or ids_stride == 0) return error.InvalidArgument;
        if (!moeColsKAligned(quant_type, K)) return error.InvalidArgument;
        const push = MoeColsDmmvPushConstants{
            .M = M,
            .K = K,
            .a_offset = a_offset,
            .expert_stride = expert_stride,
            .x_offset = x_offset,
            .y_offset = y_offset,
            .ids_stride = ids_stride,
            .x_route_divisor = @max(x_route_divisor, 1),
            .accumulate = 0,
        };
        const infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = x_buf, .offset = 0, .range = x_size },
            .{ .buffer = y_buf, .offset = 0, .range = y_size },
            .{ .buffer = counts_buf, .offset = 0, .range = counts_size },
            .{ .buffer = ids_buf, .offset = 0, .range = ids_size },
            .{ .buffer = active_blocks_buf, .offset = 0, .range = active_blocks_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            (M + 3) / 4,
            active_block_count,
            1,
        );
    }

    /// Record a decode-time matrix-vector multiply dispatch.
    /// @param self Dispatch wrapper containing the quantization-specific pipelines.
    /// @param cmd Command buffer currently being recorded.
    /// @param quant_type GGML quantization format for the weight matrix.
    /// @param descriptor_set Descriptor set containing matrix, input vector, and output buffers.
    /// @param M Output row count.
    /// @param K Input feature width.
    /// @param a_offset Byte offset for the weight matrix.
    /// @param x_offset Byte offset for the input vector.
    /// @param y_offset Byte offset for the output vector.
    /// @returns `error.UnsupportedQuantType` when no pipeline is available for `quant_type`.
    /// @note The helper uses one workgroup per 2 output rows for most quantized formats.
    pub fn recordDispatch(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        /// Quantization type.
        quant_type: GGMLType,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        M: u32,
        K: u32,
        /// Weight buffer byte offset.
        a_offset: u32,
        /// Input buffer byte offset.
        x_offset: u32,
        /// Output buffer byte offset.
        y_offset: u32,
    ) !void {
        const pip = self.pipelineForType(quant_type) orelse return error.UnsupportedQuantType;

        const push = DmmvPushConstants{
            .M = M,
            .K = K,
            .a_offset = a_offset,
            .x_offset = x_offset,
            .y_offset = y_offset,
        };

        // K-parallel (NUM_ROWS=2) for most DMMVs.
        // For very large M (LM head with M>64K), K-parallel creates too many workgroups.
        // Use the batch shader (1 thread per row, 64 rows/WG) which has fewer WGs and
        // better memory access patterns for large fan-out.
        const use_kparallel = switch (quant_type) {
            .q4_k => M <= 65536,
            .q5_k, .q6_k => true,
            else => false,
        };
        const workgroups_x = if (use_kparallel) switch (quant_type) {
            .q4_k, .q5_k, .q6_k => (M + 1) / 2,
            else => unreachable,
        } else switch (quant_type) {
            .q4_k => blk: {
                // Use batch shader in single-column mode (1 thread per row)
                if (self.pipeline_q4k_batch) |*batch_pip| {
                    const batch_push = BatchDmmvPushConstants{
                        .M = M,
                        .K = K,
                        .a_offset = a_offset,
                        .x_offset = x_offset,
                        .y_offset = y_offset,
                        .num_cols = 1,
                    };
                    cmd.dispatchWithPush(batch_pip, descriptor_set, std.mem.asBytes(&batch_push), (M + 63) / 64, 1, 1);
                    return;
                }
                break :blk (M + 1) / 2; // fallback to K-parallel
            },
            .q5_0, .q5_1, .mxfp4, .q8_0, .f16 => (M + 1) / 2,
            .f32 => M, // K-parallel: 64 threads per row via subgroupAdd
            else => (M + 63) / 64,
        };

        cmd.dispatchWithPush(
            pip,
            descriptor_set,
            std.mem.asBytes(&push),
            workgroups_x,
            1,
            1,
        );
    }

    /// Record a push-descriptor batch DMMV dispatch covering `num_cols` token columns.
    /// Bindings: 0 = A weight matrix, 1 = X_batch (K × num_cols, column-major),
    /// 2 = Y_batch (M × num_cols, column-major).
    /// @param cmd Command buffer to record into.
    /// @param quant_type Weight quantization; only Q4_K and Q6_K batch shaders are supported.
    /// @param push_desc_fn Push-descriptor function pointer (null falls back to bound descriptor sets).
    /// @param M Output row count (weight matrix rows).
    /// @param K Contraction width (hidden dimension).
    /// @param num_cols Number of token columns to process in this batch.
    /// @returns `error.UnsupportedQuantType` if no batch shader is loaded for `quant_type`.
    pub fn recordBatchDispatchPush(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        quant_type: GGMLType,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        x_buf: vk.c.VkBuffer,
        x_size: vk.c.VkDeviceSize,
        y_buf: vk.c.VkBuffer,
        y_size: vk.c.VkDeviceSize,
        M: u32,
        K: u32,
        a_offset: u32,
        x_offset: u32,
        y_offset: u32,
        num_cols: u32,
    ) !void {
        const pip = switch (quant_type) {
            .q4_k => if (self.pipeline_q4k_batch) |*p| p else return error.UnsupportedQuantType,
            .q6_k => if (self.pipeline_q6k_batch) |*p| p else return error.UnsupportedQuantType,
            else => return error.UnsupportedQuantType,
        };
        const push = BatchDmmvPushConstants{
            .M = M,
            .K = K,
            .a_offset = a_offset,
            .x_offset = x_offset,
            .y_offset = y_offset,
            .num_cols = num_cols,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = x_buf, .offset = 0, .range = x_size },
            .{ .buffer = y_buf, .offset = 0, .range = y_size },
        };
        const workgroups_x = (M + 63) / 64;
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            workgroups_x,
            1,
            1,
        );
    }

    /// Record a dispatch that quantizes `ne` f32 elements from `a_buf` into
    /// Q8_1 blocks (36 bytes each) in `d_buf`. Foundation for mul_mmq — no
    /// production callers yet. Requires `ne` to be a multiple of 32.
    /// Returns `error.PipelineNotLoaded` when the shader is unavailable,
    /// `error.InvalidArgument` when ne is not a multiple of 32.
    pub fn recordQuantizeQ8_1(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        ne: u32,
    ) !void {
        const pip = if (self.pipeline_quantize_q8_1) |*p| p else return error.PipelineNotLoaded;
        if (ne == 0 or (ne & 31) != 0) return error.InvalidArgument;
        const num_blocks = ne >> 5;
        const push = QuantizeQ8_1Push{ .ne = ne, .num_blocks = num_blocks };
        const infos = [2]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        // 4 blocks per workgroup.
        const wg_x = (num_blocks + 3) / 4;
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            wg_x,
            1,
            1,
        );
    }

    /// Effort 6 Step 3: dispatch the count_experts shader. For one layer,
    /// scan a routing buffer that stores per-(token, layer) topk expert IDs
    /// and produce a `[n_experts]` u32 count buffer.
    ///
    /// Layout assumed for `routing_buf`:
    ///   slot(token, layer) starts at byte offset
    ///       (token * n_layers + layer) * (2 * n_experts_used) * 4
    ///   the first n_experts_used u32s are expert IDs, the next n_experts_used
    ///   u32s are f32 weights (mirrors router_output_buf packing in
    ///   forward.zig:5316+).
    ///
    /// `counts_buf` must be sized for at least `n_experts * sizeof(u32)`.
    /// Caller is responsible for clearing or overwriting it (the shader
    /// writes one element per expert, indexed by gl_WorkGroupID.x).
    ///
    /// Returns `error.PipelineNotLoaded` if the count_experts shader is
    /// unavailable, `error.InvalidArgument` for zero token counts.
    pub fn recordCountExperts(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        routing_buf: vk.c.VkBuffer,
        routing_size: vk.c.VkDeviceSize,
        counts_buf: vk.c.VkBuffer,
        counts_size: vk.c.VkDeviceSize,
        n_tokens: u32,
        n_layers: u32,
        layer: u32,
        n_experts_used: u32,
        n_experts: u32,
        /// Byte offset into `counts_buf` where this dispatch's `n_experts`
        /// u32 outputs should land. Must be aligned to the device's storage
        /// buffer offset alignment (typically 4-256 B; n_experts × 4 is a
        /// natural multiple). Pass 0 to write to the start of the buffer.
        d_offset_bytes: vk.c.VkDeviceSize,
    ) !void {
        const pip = if (self.pipeline_count_experts) |*p| p else return error.PipelineNotLoaded;
        if (n_tokens == 0 or n_experts == 0 or n_experts_used == 0 or n_layers == 0) {
            return error.InvalidArgument;
        }
        if (layer >= n_layers) return error.InvalidArgument;
        const out_bytes: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, n_experts) * @sizeOf(u32);
        if (d_offset_bytes + out_bytes > counts_size) return error.InvalidArgument;
        const slot_stride_u32: u32 = 2 * n_experts_used;
        const push = CountExpertsPush{
            .ne00 = n_experts_used,
            .ne01 = n_tokens,
            .nb00 = 1,
            .nb01 = slot_stride_u32 * n_layers,
            .a_offset = layer * slot_stride_u32,
        };
        const infos = [2]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = routing_buf, .offset = 0, .range = routing_size },
            .{ .buffer = counts_buf, .offset = d_offset_bytes, .range = out_bytes },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            n_experts,
            1,
            1,
        );
    }

    /// Effort-6 Step 1: dispatch the tiled Q4_K dense GEMM
    /// (`mul_mm_q4k.comp`). Computes D[M, N] = A[M, K] (Q4_K) × B[K, N] (f32),
    /// where B and D are column-major (B[col][k] = data_b[b_offset +
    /// col*stride_b + k], analogously for D).
    ///
    /// Tile shape: WG = 64 threads producing a 32 × 32 output tile.
    /// Dispatch grid: ((M+31)/32) × ((N+31)/32) × 1.
    ///
    /// Constraints:
    /// - K must be a multiple of 256 (Q4_K super-block size).
    /// - `a_offset` is in BYTES; `b_offset` and `d_offset` are in FLOATS.
    /// - Caller is responsible for any preceding clear of D.
    ///
    /// Returns `error.PipelineNotLoaded` if mul_mm_q4k.spv isn't loaded,
    /// `error.InvalidArgument` for K-misaligned inputs.
    pub fn recordMulMmQ4K(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        // BM=32, BN=32 in the shader; keep these in sync.
        const wg_x = (M + 31) / 32;
        const wg_y = (N + 31) / 32;
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            wg_x,
            wg_y,
            1,
        );
    }

    /// Dispatch the routed tiled Q4_K MoE GEMM (`mul_mm_id_q4k.comp`).
    /// One dispatch covers all experts: grid.x tiles output rows, grid.y
    /// tiles routed token/slot columns per expert, and grid.z is expert id.
    /// `counts_buf[e]` must contain the number of token/slot pairs routed to
    /// expert `e`; `ids_buf` is the token-major top-k route table scanned by
    /// the shader. This path is a validation foundation and is not selected
    /// by production prefill yet.
    pub fn recordMulMmIdQ4K(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        ids_buf: vk.c.VkBuffer,
        ids_size: vk.c.VkDeviceSize,
        counts_buf: vk.c.VkBuffer,
        counts_size: vk.c.VkDeviceSize,
        M: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        batch_stride_a: u32,
        batch_stride_d: u32,
        nei0: u32,
        nei1: u32,
        nbi1: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
        ids_offset: u32,
        n_experts: u32,
        n_tile_y: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_id_q4k) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or n_experts == 0 or n_tile_y == 0) return error.InvalidArgument;
        if (stride_b == 0 or stride_d == 0 or batch_stride_a == 0 or batch_stride_d == 0) return error.InvalidArgument;
        if (nei0 == 0 or nei1 == 0 or nbi1 == 0) return error.InvalidArgument;
        const push = MulMmIdQ4KPush{
            .M = M,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .batch_stride_a = batch_stride_a,
            .batch_stride_d = batch_stride_d,
            .nei0 = nei0,
            .nei1 = nei1,
            .nbi1 = nbi1,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
            .ids_offset = ids_offset,
        };
        const infos = [5]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
            .{ .buffer = ids_buf, .offset = 0, .range = ids_size },
            .{ .buffer = counts_buf, .offset = 0, .range = counts_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            (M + 31) / 32,
            n_tile_y,
            n_experts,
        );
    }

    /// Tiled Q4_K dense GEMM with fused-residual accumulate: identical to
    /// recordMulMmQ4K but the output ACCUMULATES into D (d[idx] += result)
    /// instead of overwriting. Eliminates the standalone barrier +
    /// scale_accumulate dispatch for the dense-FFN residual add. Ragged
    /// shapes (M or N not multiples of 32) are handled by boundary checks
    /// in the shader.
    /// @param M Output rows (weight rows, i.e. hidden_dim for down projection).
    /// @param N Token batch size (number of columns).
    /// @param K Contraction width; must be a multiple of 256.
    /// @returns `error.PipelineNotLoaded` if the acc pipeline is absent, or `error.InvalidArgument` for zero/misaligned K or zero M/N.
    pub fn recordMulMmQ4KDownAcc(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_down_acc) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        // BM=32, BN=32 in the shader; keep these in sync.
        const wg_x = (M + 31) / 32;
        const wg_y = (N + 31) / 32;
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            wg_x,
            wg_y,
            1,
        );
    }

    /// Wide-tile (BN=64) variant of recordMulMmQ4KDownAcc. Each WG produces a
    /// 32×64 output tile. Caller must ensure N is a multiple of 64 for full
    /// efficiency (the shader handles ragged N via bounds checks).
    pub fn recordMulMmQ4KDownAccWide(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_down_acc_wide) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        // BM=32, BN=64 in the shader.
        const wg_x = (M + 31) / 32;
        const wg_y = (N + 63) / 64;
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            wg_x,
            wg_y,
            1,
        );
    }

    /// Tiled Q4_K batched dense FFN front-end: computes silu(gate_weight * B) * (up_weight * B) directly into D.
    /// Ragged (M or N not multiples of 32) shapes are handled by boundary checks in the shader.
    /// @param M Output rows (gate/up weight rows, i.e. inter_dim).
    /// @param N Token batch size (number of columns).
    /// @param K Contraction width; must be a multiple of 256.
    /// @returns `error.PipelineNotLoaded` if the gate+up+SwiGLU pipeline is absent, or `error.InvalidArgument` for zero/misaligned K or zero M/N.
    pub fn recordMulMmQ4KGateUpSwiglu(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_gate_up_swiglu) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        const wg_x = (M + 31) / 32;
        const wg_y = (N + 31) / 32;
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            wg_x,
            wg_y,
            1,
        );
    }

    /// BN=8 Q4_K GEMM for narrow token tails after the generic 32-column path.
    /// Requires row-aligned M; N may be 1..8 and is bounds-checked in-shader.
    pub fn recordMulMmQ4KTail8(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_tail8) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or N > 8 or (M & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            1,
            1,
        );
    }

    /// Tiled Q4_K batched Gemma dense FFN front-end: computes gelu(gate_weight * B) * (up_weight * B) directly into D.
    /// Same binding and push layout as recordMulMmQ4KGateUpSwiglu.
    pub fn recordMulMmQ4KGateUpGeglu(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_gate_up_geglu) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        const wg_x = (M + 31) / 32;
        const wg_y = (N + 31) / 32;
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            wg_x,
            wg_y,
            1,
        );
    }

    /// Branchless full-tile Q4_K gate/up/GEGLU GEMM for Gemma dense prefill.
    /// Requires M and N to be multiples of 32; ragged token tails use the
    /// checked recordMulMmQ4KGateUpGeglu path.
    pub fn recordMulMmQ4KGateUpGegluFull(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_gate_up_geglu_full) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0 or (N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            N / 32,
            1,
        );
    }

    /// BN=8 Q4_K gate/up/GEGLU GEMM for narrow token tails after the
    /// 32-column full-tile path. Requires row-aligned M; N may be 1..8.
    pub fn recordMulMmQ4KGateUpGegluTail8(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_gate_up_geglu_tail8) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or N > 8 or (M & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            1,
            1,
        );
    }

    /// Branchless full-tile Q4_K gate/up/SwiGLU GEMM.
    /// Requires M and N to be multiples of 32; ragged token tails use the
    /// checked recordMulMmQ4KGateUpSwiglu path.
    pub fn recordMulMmQ4KGateUpSwigluFull(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0 or (N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            N / 32,
            1,
        );
    }

    /// Tiled Q6_K dense GEMM. Same push/layout as recordMulMmQ4K.
    /// Used by Qwen3.6-27B layer-major prefill for dense-down and SSM wqkv.
    pub fn recordMulMmQ6K(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q6k) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        const wg_x = (M + 31) / 32;
        const wg_y = (N + 31) / 32;
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            wg_x,
            wg_y,
            1,
        );
    }

    /// BN=8 Q6_K GEMM for narrow token tails after the 32-column full-tile path.
    /// Requires row-aligned M; N may be 1..8 and is bounds-checked in-shader.
    pub fn recordMulMmQ6KTail8(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q6k_tail8) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or N > 8 or (M & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            1,
            1,
        );
    }

    /// Tiled Q5_K dense GEMM. Same push/layout as recordMulMmQ4K/recordMulMmQ6K.
    /// Used by Qwen3.6-27B layer-major prefill for the SSM out projection, which
    /// otherwise falls through to the dmmv_q5k one-WG-per-row batched path.
    pub fn recordMulMmQ5K(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q5k) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        const wg_x = (M + 31) / 32;
        const wg_y = (N + 31) / 32;
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            wg_x,
            wg_y,
            1,
        );
    }

    /// Wide-tile (BN=64) variant of recordMulMmQ5K. Each WG produces a 32×64
    /// output tile. Halves weight VRAM traffic for N=64 prefill.
    pub fn recordMulMmQ5KWide(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q5k_wide) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        // BM=32, BN=64 in the shader.
        const wg_x = (M + 31) / 32;
        const wg_y = (N + 63) / 64;
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            wg_x,
            wg_y,
            1,
        );
    }

    /// Tiled Q8_0 dense GEMM for Qwen3.6 A3B layer-major prefill (SSM out projection).
    /// Uses the same `MulMmQ4KPush` layout as `recordMulMmQ4K`, but K must be a multiple of 32 (not 256).
    /// @returns `error.PipelineNotLoaded` if the Q8_0 GEMM pipeline is absent, or `error.InvalidArgument` for K not a multiple of 32 or zero M/N.
    pub fn recordMulMmQ8_0(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q8_0) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 31) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            (M + 31) / 32,
            (N + 31) / 32,
            1,
        );
    }

    /// Record an int8 DP4a full-tile Q8_0 GEMM. The activation must already
    /// be quantized with recordQuantizeActQ8. Ragged token tails stay on the
    /// f32 recordMulMmQ8_0 path at the call site.
    pub fn recordMulMmQ8_0FullDp4a(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_buf: vk.c.VkBuffer,
        b_scale_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q8_0_full_dp4a) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 31) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0 or (N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ6KDp4aPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d = M,
            .a_offset = a_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_buf, .offset = 0, .range = b_scale_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            N / 32,
            1,
        );
    }

    /// Branchless full-tile Q6_K GEMM for 32-aligned token counts; the host routes only 32-aligned M/N tiles here.
    /// @note M and N must both be multiples of 32; ragged tails must use `recordMulMmQ6K` instead.
    /// @returns `error.PipelineNotLoaded` if the full-tile pipeline is absent, or `error.InvalidArgument` for misaligned or zero dimensions.
    pub fn recordMulMmQ6KFull(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q6k_full) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0 or (N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            N / 32,
            1,
        );
    }

    /// Fused-residual variant of `recordMulMmQ6KFull`: the output ACCUMULATES
    /// into d_data (d[idx] += result) instead of overwriting. This eliminates
    /// the standalone barrier + scale_accumulate dispatch that adds the
    /// dense-FFN residual. Same shape constraints as `recordMulMmQ6KFull`.
    pub fn recordMulMmQ6KFullDownAcc(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        stride_b: u32,
        stride_d: u32,
        a_offset: u32,
        b_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q6k_full_down_acc) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0 or (N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b = stride_b,
            .stride_d = stride_d,
            .a_offset = a_offset,
            .b_offset = b_offset,
            .d_offset = d_offset,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_buf, .offset = 0, .range = b_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            N / 32,
            1,
        );
    }

    /// Quantize an f32 activation matrix to packed int8 + per-32-block scales
    /// (one shot, no per-tile redundancy) for the int8 DP4a dense-down GEMM.
    /// @param src_buf token-major f32 activation [n_tokens][K].
    /// @param dst_packed_buf token-major packed int8 [n_tokens][K/4] (4 lanes/uint).
    /// @param dst_scale_buf token-major f32 scale [n_tokens][K/32].
    pub fn recordQuantizeActQ8(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        src_buf: vk.c.VkBuffer,
        src_size: vk.c.VkDeviceSize,
        dst_packed_buf: vk.c.VkBuffer,
        dst_packed_size: vk.c.VkDeviceSize,
        dst_scale_buf: vk.c.VkBuffer,
        dst_scale_size: vk.c.VkDeviceSize,
        n_tokens: u32,
        K: u32,
    ) !void {
        const pip = if (self.pipeline_quantize_act_q8) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 31) != 0 or n_tokens == 0) return error.InvalidArgument;
        const blocks_per_token = K / 32;
        const total_blocks = n_tokens * blocks_per_token;
        const push = QuantizeActPush{
            .n_tokens = n_tokens,
            .K = K,
            .blocks_per_token = blocks_per_token,
            .total_blocks = total_blocks,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = src_buf, .offset = 0, .range = src_size },
            .{ .buffer = dst_packed_buf, .offset = 0, .range = dst_packed_size },
            .{ .buffer = dst_scale_buf, .offset = 0, .range = dst_scale_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            (total_blocks + 255) / 256,
            1,
            1,
        );
    }

    /// Record the int8 DP4a full-tile Q6_K dense-down GEMM. Weights are Q6_K;
    /// the activation arrives pre-quantized (packed int8 + per-32-block f32 scale)
    /// from recordQuantizeActQ8. Output is token-major f32 [N][M].
    pub fn recordMulMmQ6KFullDp4a(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_buf: vk.c.VkBuffer,
        b_scale_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_offset: u32,
        accumulate: bool,
    ) !void {
        const bm64_acc_enabled = if (std.posix.getenv("ZINC_QWEN36_27B_BM64_DOWN_ACC")) |raw| !std.mem.eql(u8, raw, "0") else true;
        const mmq64_acc_enabled = if (std.posix.getenv("ZINC_QWEN36_27B_MMQ64_DOWN")) |raw| !std.mem.eql(u8, raw, "0") else false;
        const k12288_bk2_enabled = if (std.posix.getenv("ZINC_QWEN35_9B_K12288_BK2")) |raw| !std.mem.eql(u8, raw, "0") else true;
        const k12288_bm64_down_enabled = if (std.posix.getenv("ZINC_QWEN35_9B_BM64_DOWN")) |raw| !std.mem.eql(u8, raw, "0") else true;
        const use_n64_bm64 = !accumulate and K == 17408 and N >= 64 and (N & 63) == 0 and (M & 63) == 0 and self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64 != null;
        const use_k21504_n64_bm64 = !accumulate and K == 21504 and N == 64 and (M & 63) == 0 and self.pipeline_mul_mm_q6k_full_dp4a_k21504_n64_bm64 != null;
        const use_exact_n64_mmq64_acc = mmq64_acc_enabled and accumulate and K == 17408 and N == 64 and (M & 63) == 0 and self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_mmq64_acc != null;
        const use_exact_n64_bm64_acc = !use_exact_n64_mmq64_acc and bm64_acc_enabled and accumulate and K == 17408 and N == 64 and (M & 63) == 0 and self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc != null;
        const use_exact_n64_acc = accumulate and K == 17408 and N == 64 and self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2_acc != null;
        const use_exact_n64_bk2 = !accumulate and K == 17408 and N == 64 and self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2 != null;
        const use_k12288_ragged_n64_bm64 = k12288_bm64_down_enabled and !accumulate and K == 12288 and N > 64 and (N & 63) != 0 and (M & 63) == 0 and self.pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bm64_ragged != null;
        const use_k12288_ragged_n64_bk2 = !use_k12288_ragged_n64_bm64 and k12288_bk2_enabled and !accumulate and K == 12288 and N > 64 and (N & 63) != 0 and self.pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bk2_ragged != null;
        const use_k12288_ragged_n64 = !accumulate and K == 12288 and N > 64 and (N & 63) != 0 and (use_k12288_ragged_n64_bm64 or use_k12288_ragged_n64_bk2 or self.pipeline_mul_mm_q6k_full_dp4a_k12288_n64_ragged != null);
        const use_ragged_n64_bm64 = !accumulate and K == 17408 and N > 64 and (N & 63) != 0 and (M & 63) == 0 and self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged_bm64 != null;
        const use_ragged_n64 = K == 17408 and N > 64 and (N & 63) != 0 and (use_ragged_n64_bm64 or self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged != null);
        if (accumulate and !(use_exact_n64_mmq64_acc or use_exact_n64_bm64_acc or use_exact_n64_acc)) return error.PipelineNotLoaded;
        const n_tile: u32 = if (use_n64_bm64 or use_k21504_n64_bm64 or use_exact_n64_mmq64_acc or use_exact_n64_bm64_acc)
            64
        else if (K == 17408 and N == 40 and self.pipeline_mul_mm_q6k_full_dp4a_k17408_n40 != null)
            40
        else if (use_k12288_ragged_n64)
            64
        else if (use_exact_n64_acc or use_exact_n64_bk2)
            64
        else if (use_ragged_n64)
            64
        else if (K == 17408 and N >= 64 and (N & 63) == 0 and self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64 != null)
            64
        else if (K == 21504 and N == 64 and self.pipeline_mul_mm_q6k_full_dp4a_k21504_n64 != null)
            64
        else
            32;
        const pip = blk: {
            if (use_ragged_n64_bm64) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged_bm64) |*p| break :blk p;
            }
            if (use_n64_bm64) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64) |*p| break :blk p;
            }
            if (use_k21504_n64_bm64) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k21504_n64_bm64) |*p| break :blk p;
            }
            if (use_exact_n64_mmq64_acc) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_mmq64_acc) |*p| break :blk p;
            }
            if (use_exact_n64_bm64_acc) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc) |*p| break :blk p;
            }
            if (use_exact_n64_acc) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2_acc) |*p| break :blk p;
            }
            if (use_exact_n64_bk2) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2) |*p| break :blk p;
            }
            if (use_ragged_n64) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged) |*p| break :blk p;
            }
            if (use_k12288_ragged_n64_bm64) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bm64_ragged) |*p| break :blk p;
            }
            if (use_k12288_ragged_n64_bk2) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bk2_ragged) |*p| break :blk p;
            }
            if (use_k12288_ragged_n64) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k12288_n64_ragged) |*p| break :blk p;
            }
            if (K == 17408 and n_tile == 40) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n40) |*p| break :blk p;
            }
            if (K == 17408 and n_tile == 64) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64) |*p| break :blk p;
            }
            if (K == 12288) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k12288) |*p| break :blk p;
            }
            if (K == 21504 and n_tile == 64) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k21504_n64) |*p| break :blk p;
            }
            if (K == 21504) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k21504) |*p| break :blk p;
            }
            break :blk if (self.pipeline_mul_mm_q6k_full_dp4a) |*p| p else return error.PipelineNotLoaded;
        };
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0) return error.InvalidArgument;
        if (n_tile == 40) {
            if (N != 40) return error.InvalidArgument;
        } else if (n_tile == 64) {
            if (!(use_ragged_n64 or use_k12288_ragged_n64) and (N & 63) != 0) return error.InvalidArgument;
        } else if ((N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ6KDp4aPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d = M,
            .a_offset = a_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_buf, .offset = 0, .range = b_scale_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            if (use_n64_bm64 or use_k21504_n64_bm64 or use_exact_n64_mmq64_acc or use_exact_n64_bm64_acc or use_ragged_n64_bm64 or use_k12288_ragged_n64_bm64) M / 64 else M / 32,
            if (use_ragged_n64 or use_k12288_ragged_n64) (N + n_tile - 1) / n_tile else N / n_tile,
            1,
        );
    }

    pub fn recordMulMmQ6KFullDp4aN64InterleavedAcc(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_buf: vk.c.VkBuffer,
        b_scale_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc_interleaved) |*p| p else return error.PipelineNotLoaded;
        if (M == 0 or N != 64 or K != 17408 or (M & 63) != 0) return error.InvalidArgument;
        const push = MulMmQ6KDp4aPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d = M,
            .a_offset = a_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_buf, .offset = 0, .range = b_scale_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 64,
            1,
            1,
        );
    }

    /// Guarded BN=72 int8 DP4a Q6_K dense-down GEMM for Gemma 4 31B's
    /// 65-72 token public prompt shape. This covers the 64-column body and
    /// <=8-column ragged tail in one pass over the K=21504 down weights.
    pub fn recordMulMmQ6KRagged72Dp4a(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_buf: vk.c.VkBuffer,
        b_scale_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q6k_full_dp4a_k21504_n72) |*p| p else return error.PipelineNotLoaded;
        if (K != 21504) return error.InvalidArgument;
        if (M == 0 or (M & 31) != 0) return error.InvalidArgument;
        if (N <= 64 or N > 72) return error.InvalidArgument;
        const push = MulMmQ6KDp4aPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d = M,
            .a_offset = a_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_buf, .offset = 0, .range = b_scale_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            1,
            1,
        );
    }

    /// BN=8 int8 DP4a Q6_K dense-down tail for Gemma 4 31B short prompts.
    /// The packed/scaled activation descriptors are offset to the first tail
    /// token, while the output uses d_offset so the token-major destination
    /// layout remains contiguous with the full 64-token prefix.
    pub fn recordMulMmQ6KTail8Dp4a(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_buf: vk.c.VkBuffer,
        b_scale_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        b_packed_offset: u32,
        b_scale_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = blk: {
            if (K == 21504) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_k21504_n8) |*p| break :blk p;
            }
            return error.PipelineNotLoaded;
        };
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or N > 8 or (M & 31) != 0) return error.InvalidArgument;
        const b_packed_byte_offset: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, b_packed_offset) * @sizeOf(u32);
        const b_scale_byte_offset: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, b_scale_offset) * @sizeOf(f32);
        if (b_packed_byte_offset >= b_packed_size or b_scale_byte_offset >= b_scale_size) return error.InvalidArgument;
        const push = MulMmQ6KDp4aPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d = M,
            .a_offset = a_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_packed_buf, .offset = b_packed_byte_offset, .range = b_packed_size - b_packed_byte_offset },
            .{ .buffer = b_scale_buf, .offset = b_scale_byte_offset, .range = b_scale_size - b_scale_byte_offset },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            1,
            1,
        );
    }

    /// Q8_1-input variant: same Q6_K DP4a GEMM but the activation scale
    /// buffer is `vec2 b_scale_dsum[]` per 32-block (Q4_K-style layout). The
    /// shader reads `.x` only — `.y` (dsum) is unused since Q6_K weights have
    /// no per-block bias term. Used by the Qwen3.6-27B SSM wqkv projection so
    /// it can share a single Q8_1 quantize of scratch_norm with the Q4_K z
    /// projection. Push constant stride_b_scale = K/32 (number of vec2 entries
    /// per token), same numeric value as the Q8_0 variant since the indexing
    /// is in typed-element units.
    pub fn recordMulMmQ6KFullDp4aQ8_1(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_offset: u32,
    ) !void {
        const use_k5120_n64_bm64 = K == 5120 and N == 64 and (M & 63) == 0 and self.pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_bm64 != null;
        const use_ragged_n64 = K == 5120 and N > 64 and (N & 63) != 0 and self.pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_ragged != null;
        const n_tile: u32 = if (use_k5120_n64_bm64)
            64
        else if (K == 5120 and N == 40 and self.pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n40 != null)
            40
        else if (use_ragged_n64)
            64
        else
            32;
        const pip = blk: {
            if (use_k5120_n64_bm64) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_bm64) |*p| break :blk p;
            }
            if (use_ragged_n64) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_ragged) |*p| break :blk p;
            }
            if (K == 5120 and n_tile == 40) {
                if (self.pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n40) |*p| break :blk p;
            }
            break :blk if (self.pipeline_mul_mm_q6k_full_dp4a_q8_1) |*p| p else return error.PipelineNotLoaded;
        };
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0) return error.InvalidArgument;
        if (n_tile == 40) {
            if (N != 40) return error.InvalidArgument;
        } else if (n_tile == 64) {
            if (!use_ragged_n64 and (N & 63) != 0) return error.InvalidArgument;
        } else if ((N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ6KDp4aPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d = M,
            .a_offset = a_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_dsum_buf, .offset = 0, .range = b_scale_dsum_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            if (use_k5120_n64_bm64) M / 64 else M / 32,
            if (use_ragged_n64) (N + n_tile - 1) / n_tile else N / n_tile,
            1,
        );
    }

    /// Q8_1-style activation quantize: packed int8 + per-32-block (scale, dsum)
    /// for the DP4a Q4_K gate+up GEMM bias-correction term.
    pub fn recordQuantizeActQ8_1(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        src_buf: vk.c.VkBuffer,
        src_size: vk.c.VkDeviceSize,
        dst_packed_buf: vk.c.VkBuffer,
        dst_packed_size: vk.c.VkDeviceSize,
        dst_scale_dsum_buf: vk.c.VkBuffer,
        dst_scale_dsum_size: vk.c.VkDeviceSize,
        n_tokens: u32,
        K: u32,
    ) !void {
        const pip = if (self.pipeline_quantize_act_q8_1) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 31) != 0 or n_tokens == 0) return error.InvalidArgument;
        const blocks_per_token = K / 32;
        const total_blocks = n_tokens * blocks_per_token;
        const push = QuantizeActPush{
            .n_tokens = n_tokens,
            .K = K,
            .blocks_per_token = blocks_per_token,
            .total_blocks = total_blocks,
        };
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = src_buf, .offset = 0, .range = src_size },
            .{ .buffer = dst_packed_buf, .offset = 0, .range = dst_packed_size },
            .{ .buffer = dst_scale_dsum_buf, .offset = 0, .range = dst_scale_dsum_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            (total_blocks + 255) / 256,
            1,
            1,
        );
    }

    /// int8 DP4a full-tile Q4_K gate+up+SwiGLU GEMM for Qwen3.6-27B dense FFN
    /// prefill. Activations arrive pre-quantized from recordQuantizeActQ8_1
    /// (packed int8 + per-32-block (scale, dsum)). Output is silu(gate)*up,
    /// token-major f32 [N][M].
    pub fn recordMulMmQ4KGateUpSwigluFullDp4a(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0 or (N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KGateUpDp4aPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d = M,
            .a_offset = a_offset,
            .d_offset = d_offset,
        };
        const infos = [5]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_dsum_buf, .offset = 0, .range = b_scale_dsum_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            N / 32,
            1,
        );
    }

    /// int8 DP4a full-tile Q4_K gate+up+GEGLU GEMM for Gemma dense FFN
    /// prefill. Activations arrive pre-quantized from recordQuantizeActQ8_1
    /// (packed int8 + per-32-block (scale, dsum)). Output is gelu(gate)*up,
    /// token-major f32 [N][M].
    pub fn recordMulMmQ4KGateUpGegluFullDp4a(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0 or (N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KGateUpDp4aPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d = M,
            .a_offset = a_offset,
            .d_offset = d_offset,
        };
        const infos = [5]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_dsum_buf, .offset = 0, .range = b_scale_dsum_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            N / 32,
            1,
        );
    }

    /// int8 DP4a full-tile Q4_K gate+up+SwiGLU GEMM that emits Q8_0-style
    /// packed activation directly. Output layout matches quantize_act_q8.comp
    /// (per-token packed int8 + per-32-block scale), so the downstream
    /// dense-down DP4a kernel can consume it directly without the standalone
    /// quantize_act_q8 dispatch + barrier. The f32 SwiGLU intermediate is
    /// never written to global memory.
    pub fn recordMulMmQ4KGateUpSwigluFullDp4aQ8(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_packed_buf: vk.c.VkBuffer,
        d_packed_size: vk.c.VkDeviceSize,
        d_scale_buf: vk.c.VkBuffer,
        d_scale_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_packed_offset: u32,
        d_scale_offset: u32,
    ) !void {
        const use_bm64_n64 = K == 5120 and N >= 64 and (N & 63) == 0 and (M & 63) == 0 and self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64 != null;
        const use_bm64_ragged_n64 = K == 5120 and N > 64 and (N & 63) != 0 and (M & 63) == 0 and self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged_bm64 != null;
        const use_k4096_ragged_n64 = K == 4096 and N > 64 and (N & 63) != 0 and self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64_ragged != null;
        const use_k5120_ragged_n64 = K == 5120 and N > 64 and (N & 63) != 0 and (use_bm64_ragged_n64 or self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged != null);
        const use_ragged_n64 = use_k4096_ragged_n64 or use_k5120_ragged_n64;
        const n_tile: u32 = if (N == 40 and K == 5120 and self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n40 != null)
            40
        else if (use_ragged_n64)
            64
        else if (N >= 64 and (N & 63) == 0 and ((K == 4096 and self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64 != null) or (K == 5120 and self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64 != null)))
            64
        else
            32;
        const m_tile: u32 = if (use_bm64_n64 or use_bm64_ragged_n64) 64 else 32;
        const pip = blk: {
            if (use_bm64_ragged_n64) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged_bm64) |*p| break :blk p;
            }
            if (use_bm64_n64) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64) |*p| break :blk p;
            }
            if (use_ragged_n64) {
                if (use_k4096_ragged_n64) {
                    if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64_ragged) |*p| break :blk p;
                }
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged) |*p| break :blk p;
            }
            if (K == 5120 and n_tile == 40) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n40) |*p| break :blk p;
            }
            if (K == 4096 and n_tile == 64) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64) |*p| break :blk p;
            }
            if (K == 5120 and n_tile == 64) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64) |*p| break :blk p;
            }
            if (K == 4096) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096) |*p| break :blk p;
            }
            if (K == 5120) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120) |*p| break :blk p;
            }
            break :blk if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8) |*p| p else return error.PipelineNotLoaded;
        };
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0) return error.InvalidArgument;
        if (n_tile == 40) {
            if (N != 40) return error.InvalidArgument;
        } else if (n_tile == 64) {
            if (!use_ragged_n64 and (N & 63) != 0) return error.InvalidArgument;
        } else if ((N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KGateUpDp4aQ8Push{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d_packed = M / 4,
            .stride_d_scale = M / 32,
            .a_offset = a_offset,
            .b_packed_offset = 0,
            .b_scale_offset = 0,
            .d_packed_offset = d_packed_offset,
            .d_scale_offset = d_scale_offset,
        };
        const infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_dsum_buf, .offset = 0, .range = b_scale_dsum_size },
            .{ .buffer = d_packed_buf, .offset = 0, .range = d_packed_size },
            .{ .buffer = d_scale_buf, .offset = 0, .range = d_scale_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / m_tile,
            if (use_ragged_n64) (N + n_tile - 1) / n_tile else N / n_tile,
            1,
        );
    }

    pub fn recordMulMmQ4KGateUpSwigluFullDp4aQ8N64Interleaved(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_packed_buf: vk.c.VkBuffer,
        d_packed_size: vk.c.VkDeviceSize,
        d_scale_buf: vk.c.VkBuffer,
        d_scale_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_packed_offset: u32,
        d_scale_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64_interleaved) |*p| p else return error.PipelineNotLoaded;
        if (M == 0 or N != 64 or K != 5120 or (M & 63) != 0) return error.InvalidArgument;
        const push = MulMmQ4KGateUpDp4aQ8Push{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d_packed = M / 4,
            .stride_d_scale = M / 32,
            .a_offset = a_offset,
            .b_packed_offset = 0,
            .b_scale_offset = 0,
            .d_packed_offset = d_packed_offset,
            .d_scale_offset = d_scale_offset,
        };
        const infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_dsum_buf, .offset = 0, .range = b_scale_dsum_size },
            .{ .buffer = d_packed_buf, .offset = 0, .range = d_packed_size },
            .{ .buffer = d_scale_buf, .offset = 0, .range = d_scale_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 64,
            1,
            1,
        );
    }

    /// Gemma GEGLU variant of recordMulMmQ4KGateUpSwigluFullDp4aQ8. Emits
    /// Q8_0-style packed GEGLU activation for Q6_K dense-down DP4a.
    pub fn recordMulMmQ4KGateUpGegluFullDp4aQ8(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_packed_buf: vk.c.VkBuffer,
        d_packed_size: vk.c.VkDeviceSize,
        d_scale_buf: vk.c.VkBuffer,
        d_scale_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        b_packed_offset: u32,
        b_scale_offset: u32,
        d_packed_offset: u32,
        d_scale_offset: u32,
    ) !void {
        const n_tile: u32 = if (K == 5376 and N <= 8 and self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n8 != null)
            8
        else if (N == 64 and (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n64 != null or self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_n64 != null))
            64
        else
            32;
        const pip = blk: {
            if (K == 5376 and n_tile == 8) {
                if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n8) |*p| break :blk p;
            }
            if (K == 5376 and n_tile == 64) {
                if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n64) |*p| break :blk p;
            }
            if (K == 5376) {
                if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376) |*p| break :blk p;
            }
            if (n_tile == 64) {
                if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_n64) |*p| break :blk p;
            }
            break :blk if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8) |*p| p else return error.PipelineNotLoaded;
        };
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0) return error.InvalidArgument;
        if (n_tile == 8) {
            if (N > 8) return error.InvalidArgument;
        } else if ((N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KGateUpDp4aQ8Push{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d_packed = M / 4,
            .stride_d_scale = M / 32,
            .a_offset = a_offset,
            .b_packed_offset = b_packed_offset,
            .b_scale_offset = b_scale_offset,
            .d_packed_offset = d_packed_offset,
            .d_scale_offset = d_scale_offset,
        };
        const infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_dsum_buf, .offset = 0, .range = b_scale_dsum_size },
            .{ .buffer = d_packed_buf, .offset = 0, .range = d_packed_size },
            .{ .buffer = d_scale_buf, .offset = 0, .range = d_scale_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            (N + n_tile - 1) / n_tile,
            1,
        );
    }

    /// Single-token Gemma GEGLU producer for Q6_K-down decode. Same bindings
    /// and push layout as recordMulMmQ4KGateUpGegluFullDp4aQ8, but fixed to
    /// one token so all lanes do useful work instead of guarding seven BN=8
    /// columns.
    pub fn recordMulMmQ4KGateUpGegluN1Dp4aQ8(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_packed_buf: vk.c.VkBuffer,
        d_packed_size: vk.c.VkDeviceSize,
        d_scale_buf: vk.c.VkBuffer,
        d_scale_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        b_packed_offset: u32,
        b_scale_offset: u32,
        d_packed_offset: u32,
        d_scale_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_gate_up_geglu_n1_dp4a_q8) |*p| p else return error.PipelineNotLoaded;
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N != 1 or (M & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KGateUpDp4aQ8Push{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d_packed = M / 4,
            .stride_d_scale = M / 32,
            .a_offset = a_offset,
            .b_packed_offset = b_packed_offset,
            .b_scale_offset = b_scale_offset,
            .d_packed_offset = d_packed_offset,
            .d_scale_offset = d_scale_offset,
        };
        const infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_dsum_buf, .offset = 0, .range = b_scale_dsum_size },
            .{ .buffer = d_packed_buf, .offset = 0, .range = d_packed_size },
            .{ .buffer = d_scale_buf, .offset = 0, .range = d_scale_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            1,
            1,
        );
    }

    /// Q4_K-down sibling of recordMulMmQ4KGateUpSwigluFullDp4aQ8. Same fused
    /// gate+up+SwiGLU DP4a GEMM, but emits Q8_1-style activation (packed int8
    /// + per-32-block (scale, dsum) vec2) so the downstream mul_mm_q4k_full_dp4a
    /// (Q4_K-down) consumer can skip the standalone quantize_act_q8_1 dispatch
    /// + barrier. dsum = scale * sum(int8_lanes) is computed inside the kernel
    /// via subgroupClusteredAdd cluster_size=TPR_M=8, so there's no LDS
    /// round-trip beyond the GEMM's existing barriers. Caller is responsible
    /// for sizing d_scale_dsum_buf as 2x the Q8_0 scale buffer (per-block
    /// vec2 instead of per-block float).
    pub fn recordMulMmQ4KGateUpSwigluFullDp4aQ8_1(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_packed_buf: vk.c.VkBuffer,
        d_packed_size: vk.c.VkDeviceSize,
        d_scale_dsum_buf: vk.c.VkBuffer,
        d_scale_dsum_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_packed_offset: u32,
        d_scale_dsum_offset: u32,
    ) !void {
        const use_bm64_n64 = K == 5120 and N >= 64 and (N & 63) == 0 and (M & 63) == 0 and self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64 != null;
        const use_bm64_ragged_n64 = K == 5120 and N > 64 and (N & 63) != 0 and (M & 63) == 0 and self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged_bm64 != null;
        const use_k4096_ragged_n64 = K == 4096 and N > 64 and (N & 63) != 0 and self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64_ragged != null;
        const use_k5120_ragged_n64 = K == 5120 and N > 64 and (N & 63) != 0 and (use_bm64_ragged_n64 or self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged != null);
        const use_ragged_n64 = use_k4096_ragged_n64 or use_k5120_ragged_n64;
        const n_tile: u32 = if (N == 40 and K == 5120 and self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n40 != null)
            40
        else if (use_ragged_n64)
            64
        else if (N >= 64 and (N & 63) == 0 and ((K == 4096 and self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64 != null) or (K == 5120 and self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64 != null)))
            64
        else
            32;
        const m_tile: u32 = if (use_bm64_n64 or use_bm64_ragged_n64) 64 else 32;
        const pip = blk: {
            if (use_bm64_ragged_n64) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged_bm64) |*p| break :blk p;
            }
            if (use_bm64_n64) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64) |*p| break :blk p;
            }
            if (use_ragged_n64) {
                if (use_k4096_ragged_n64) {
                    if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64_ragged) |*p| break :blk p;
                }
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged) |*p| break :blk p;
            }
            if (K == 5120 and n_tile == 40) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n40) |*p| break :blk p;
            }
            if (K == 4096 and n_tile == 64) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64) |*p| break :blk p;
            }
            if (K == 5120 and n_tile == 64) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64) |*p| break :blk p;
            }
            if (K == 4096) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096) |*p| break :blk p;
            }
            if (K == 5120) {
                if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120) |*p| break :blk p;
            }
            break :blk if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1) |*p| p else return error.PipelineNotLoaded;
        };
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0) return error.InvalidArgument;
        if (n_tile == 40) {
            if (N != 40) return error.InvalidArgument;
        } else if (n_tile == 64) {
            if (!use_ragged_n64 and (N & 63) != 0) return error.InvalidArgument;
        } else if ((N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KGateUpDp4aQ8Push{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d_packed = M / 4,
            .stride_d_scale = M / 32,
            .a_offset = a_offset,
            .b_packed_offset = 0,
            .b_scale_offset = 0,
            .d_packed_offset = d_packed_offset,
            .d_scale_offset = d_scale_dsum_offset,
        };
        const infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_dsum_buf, .offset = 0, .range = b_scale_dsum_size },
            .{ .buffer = d_packed_buf, .offset = 0, .range = d_packed_size },
            .{ .buffer = d_scale_dsum_buf, .offset = 0, .range = d_scale_dsum_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / m_tile,
            if (use_ragged_n64) (N + n_tile - 1) / n_tile else N / n_tile,
            1,
        );
    }

    pub fn recordMulMmQ4KGateUpSwigluFullDp4aQ8_1N64Interleaved(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_packed_buf: vk.c.VkBuffer,
        d_packed_size: vk.c.VkDeviceSize,
        d_scale_dsum_buf: vk.c.VkBuffer,
        d_scale_dsum_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_packed_offset: u32,
        d_scale_dsum_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64_interleaved) |*p| p else return error.PipelineNotLoaded;
        if (M == 0 or N != 64 or K != 5120 or (M & 63) != 0) return error.InvalidArgument;
        const push = MulMmQ4KGateUpDp4aQ8Push{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d_packed = M / 4,
            .stride_d_scale = M / 32,
            .a_offset = a_offset,
            .b_packed_offset = 0,
            .b_scale_offset = 0,
            .d_packed_offset = d_packed_offset,
            .d_scale_offset = d_scale_dsum_offset,
        };
        const infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_dsum_buf, .offset = 0, .range = b_scale_dsum_size },
            .{ .buffer = d_packed_buf, .offset = 0, .range = d_packed_size },
            .{ .buffer = d_scale_dsum_buf, .offset = 0, .range = d_scale_dsum_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 64,
            1,
            1,
        );
    }

    /// Gemma GEGLU variant of recordMulMmQ4KGateUpSwigluFullDp4aQ8_1. Emits
    /// Q8_1-style packed GEGLU activation for Q4_K dense-down DP4a.
    pub fn recordMulMmQ4KGateUpGegluFullDp4aQ8_1(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_packed_buf: vk.c.VkBuffer,
        d_packed_size: vk.c.VkDeviceSize,
        d_scale_dsum_buf: vk.c.VkBuffer,
        d_scale_dsum_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        b_packed_offset: u32,
        b_scale_offset: u32,
        d_packed_offset: u32,
        d_scale_dsum_offset: u32,
    ) !void {
        const n_tile: u32 = if (K == 5376 and N <= 8 and self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n8 != null)
            8
        else if (N == 64 and (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n64 != null or self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_n64 != null))
            64
        else
            32;
        const pip = blk: {
            if (K == 5376 and n_tile == 8) {
                if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n8) |*p| break :blk p;
            }
            if (K == 5376 and n_tile == 64) {
                if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n64) |*p| break :blk p;
            }
            if (K == 5376) {
                if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376) |*p| break :blk p;
            }
            if (n_tile == 64) {
                if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_n64) |*p| break :blk p;
            }
            break :blk if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1) |*p| p else return error.PipelineNotLoaded;
        };
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0) return error.InvalidArgument;
        if (n_tile == 8) {
            if (N > 8) return error.InvalidArgument;
        } else if ((N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KGateUpDp4aQ8Push{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d_packed = M / 4,
            .stride_d_scale = M / 32,
            .a_offset = a_offset,
            .b_packed_offset = b_packed_offset,
            .b_scale_offset = b_scale_offset,
            .d_packed_offset = d_packed_offset,
            .d_scale_offset = d_scale_dsum_offset,
        };
        const infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = gate_buf, .offset = 0, .range = gate_size },
            .{ .buffer = up_buf, .offset = 0, .range = up_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_dsum_buf, .offset = 0, .range = b_scale_dsum_size },
            .{ .buffer = d_packed_buf, .offset = 0, .range = d_packed_size },
            .{ .buffer = d_scale_dsum_buf, .offset = 0, .range = d_scale_dsum_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            (N + n_tile - 1) / n_tile,
            1,
        );
    }

    /// int8 DP4a full-tile single Q5_K GEMM (no fused activation). Used by the
    /// Qwen3.6-27B SSM out prefill projection (M=hidden_dim, K=d_inner).
    /// Activations arrive pre-quantized (packed int8 + per-32-block (scale,
    /// dsum)) from recordQuantizeActQ8_1. Output is token-major f32 [N][M].
    /// Same push/binding shape as recordMulMmQ4KFullDp4a — the only difference
    /// is the 5-bit weight unpack inside the kernel.
    pub fn recordMulMmQ5KFullDp4a(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_offset: u32,
    ) !void {
        const qwen35_ssm_q5_bk2_enabled = if (std.posix.getenv("ZINC_QWEN35_9B_SSM_Q5_BK2")) |raw| !std.mem.eql(u8, raw, "0") else true;
        const use_k4096_bk2 = qwen35_ssm_q5_bk2_enabled and K == 4096 and self.pipeline_mul_mm_q5k_full_dp4a_k4096_bk2 != null;
        const n_tile: u32 = if (K == 6144 and N == 40 and self.pipeline_mul_mm_q5k_full_dp4a_k6144_n40 != null)
            40
        else
            32;
        const pip = blk: {
            if (use_k4096_bk2) {
                if (self.pipeline_mul_mm_q5k_full_dp4a_k4096_bk2) |*p| break :blk p;
            }
            if (K == 6144 and n_tile == 40) {
                if (self.pipeline_mul_mm_q5k_full_dp4a_k6144_n40) |*p| break :blk p;
            }
            break :blk if (self.pipeline_mul_mm_q5k_full_dp4a) |*p| p else return error.PipelineNotLoaded;
        };
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0) return error.InvalidArgument;
        if (n_tile == 40) {
            if (N != 40) return error.InvalidArgument;
        } else if ((N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KGateUpDp4aPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d = M,
            .a_offset = a_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_dsum_buf, .offset = 0, .range = b_scale_dsum_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            N / n_tile,
            1,
        );
    }

    /// int8 DP4a full-tile single Q4_K GEMM (no fused activation). Used by the
    /// Qwen3.6-27B SSM z prefill projection. Activations arrive pre-quantized
    /// (packed int8 + per-32-block (scale, dsum)) from recordQuantizeActQ8_1.
    /// Output is token-major f32 [N][M].
    pub fn recordMulMmQ4KFullDp4a(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_offset: u32,
        accumulate: bool,
    ) !void {
        const bm64_acc_enabled = if (std.posix.getenv("ZINC_QWEN36_27B_BM64_DOWN_ACC")) |raw| !std.mem.eql(u8, raw, "0") else true;
        const mmq64_acc_enabled = if (std.posix.getenv("ZINC_QWEN36_27B_MMQ64_DOWN")) |raw| !std.mem.eql(u8, raw, "0") else false;
        const k12288_bk2_enabled = if (std.posix.getenv("ZINC_QWEN35_9B_K12288_BK2")) |raw| !std.mem.eql(u8, raw, "0") else true;
        const k12288_bm64_down_enabled = if (std.posix.getenv("ZINC_QWEN35_9B_BM64_DOWN")) |raw| !std.mem.eql(u8, raw, "0") else true;
        const use_k5120_n64_bm64 = !accumulate and K == 5120 and N == 64 and (M & 63) == 0 and self.pipeline_mul_mm_q4k_full_dp4a_k5120_n64_bm64 != null;
        const use_k5120_ragged_n64 = !accumulate and K == 5120 and N > 64 and (N & 63) != 0 and self.pipeline_mul_mm_q4k_full_dp4a_k5120_n64_ragged != null;
        const use_k12288_ragged_n64_bm64 = k12288_bm64_down_enabled and !accumulate and K == 12288 and N > 64 and (N & 63) != 0 and (M & 63) == 0 and self.pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bm64_ragged != null;
        const use_k12288_ragged_n64_bk2 = !use_k12288_ragged_n64_bm64 and k12288_bk2_enabled and !accumulate and K == 12288 and N > 64 and (N & 63) != 0 and self.pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bk2_ragged != null;
        const use_k12288_ragged_n64 = !accumulate and K == 12288 and N > 64 and (N & 63) != 0 and (use_k12288_ragged_n64_bm64 or use_k12288_ragged_n64_bk2 or self.pipeline_mul_mm_q4k_full_dp4a_k12288_n64_ragged != null);
        const use_n64_bm64 = !accumulate and K == 17408 and N >= 64 and (N & 63) == 0 and (M & 63) == 0 and self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64 != null;
        const use_exact_n64_mmq64_acc = mmq64_acc_enabled and accumulate and K == 17408 and N == 64 and (M & 63) == 0 and self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_mmq64_acc != null;
        const use_exact_n64_bm64_acc = !use_exact_n64_mmq64_acc and bm64_acc_enabled and accumulate and K == 17408 and N == 64 and (M & 63) == 0 and self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc != null;
        const use_exact_n64_acc = accumulate and K == 17408 and N == 64 and self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2_acc != null;
        const use_exact_n64_bk2 = !accumulate and K == 17408 and N == 64 and self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2 != null;
        const use_ragged_n64_bm64 = !accumulate and K == 17408 and N > 64 and (N & 63) != 0 and (M & 63) == 0 and self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged_bm64 != null;
        const use_ragged_n64 = K == 17408 and N > 64 and (N & 63) != 0 and (use_ragged_n64_bm64 or self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged != null);
        if (accumulate and !(use_exact_n64_mmq64_acc or use_exact_n64_bm64_acc or use_exact_n64_acc)) return error.PipelineNotLoaded;
        const n_tile: u32 = if (use_k5120_n64_bm64 or use_n64_bm64 or use_exact_n64_mmq64_acc or use_exact_n64_bm64_acc)
            64
        else if (use_k5120_ragged_n64)
            64
        else if (use_k12288_ragged_n64)
            64
        else if (K == 5120 and N == 40 and self.pipeline_mul_mm_q4k_full_dp4a_k5120_n40 != null)
            40
        else if (K == 17408 and N == 40 and self.pipeline_mul_mm_q4k_full_dp4a_k17408_n40 != null)
            40
        else if (use_exact_n64_acc or use_exact_n64_bk2)
            64
        else if (use_ragged_n64)
            64
        else if (K == 17408 and N == 64 and self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64 != null)
            64
        else if (K == 17408 and N >= 64 and (N & 63) == 0 and self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64 != null)
            64
        else if (K == 21504 and N == 64 and self.pipeline_mul_mm_q4k_full_dp4a_k21504_n64 != null)
            64
        else
            32;
        const pip = blk: {
            if (use_k5120_n64_bm64) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k5120_n64_bm64) |*p| break :blk p;
            }
            if (use_k5120_ragged_n64) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k5120_n64_ragged) |*p| break :blk p;
            }
            if (use_k12288_ragged_n64_bm64) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bm64_ragged) |*p| break :blk p;
            }
            if (use_k12288_ragged_n64_bk2) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bk2_ragged) |*p| break :blk p;
            }
            if (use_k12288_ragged_n64) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k12288_n64_ragged) |*p| break :blk p;
            }
            if (use_ragged_n64_bm64) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged_bm64) |*p| break :blk p;
            }
            if (use_n64_bm64) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64) |*p| break :blk p;
            }
            if (use_exact_n64_mmq64_acc) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_mmq64_acc) |*p| break :blk p;
            }
            if (use_exact_n64_bm64_acc) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc) |*p| break :blk p;
            }
            if (use_exact_n64_acc) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2_acc) |*p| break :blk p;
            }
            if (use_ragged_n64) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged) |*p| break :blk p;
            }
            if (K == 5120 and n_tile == 40) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k5120_n40) |*p| break :blk p;
            }
            if (K == 17408 and n_tile == 40) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n40) |*p| break :blk p;
            }
            if (use_exact_n64_bk2) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2) |*p| break :blk p;
            }
            if (K == 17408 and n_tile == 64) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64) |*p| break :blk p;
            }
            if (K == 21504 and N == 64) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k21504_n64) |*p| break :blk p;
            }
            if (K == 12288) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k12288) |*p| break :blk p;
            }
            if (K == 21504) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k21504) |*p| break :blk p;
            }
            break :blk if (self.pipeline_mul_mm_q4k_full_dp4a) |*p| p else return error.PipelineNotLoaded;
        };
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or (M & 31) != 0) return error.InvalidArgument;
        if (n_tile == 40) {
            if (N != 40) return error.InvalidArgument;
        } else if (n_tile == 64) {
            if (!(use_ragged_n64 or use_k5120_ragged_n64 or use_k12288_ragged_n64) and (N & 63) != 0) return error.InvalidArgument;
        } else if ((N & 31) != 0) return error.InvalidArgument;
        const push = MulMmQ4KGateUpDp4aPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d = M,
            .a_offset = a_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_dsum_buf, .offset = 0, .range = b_scale_dsum_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            if (use_k5120_n64_bm64 or use_n64_bm64 or use_exact_n64_mmq64_acc or use_exact_n64_bm64_acc or use_ragged_n64_bm64 or use_k12288_ragged_n64_bm64) M / 64 else M / 32,
            if (use_ragged_n64 or use_k5120_ragged_n64 or use_k12288_ragged_n64) (N + n_tile - 1) / n_tile else N / n_tile,
            1,
        );
    }

    pub fn recordMulMmQ4KFullDp4aN64InterleavedAcc(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc_interleaved) |*p| p else return error.PipelineNotLoaded;
        if (M == 0 or N != 64 or K != 17408 or (M & 63) != 0) return error.InvalidArgument;
        const push = MulMmQ4KGateUpDp4aPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d = M,
            .a_offset = a_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_packed_buf, .offset = 0, .range = b_packed_size },
            .{ .buffer = b_scale_dsum_buf, .offset = 0, .range = b_scale_dsum_size },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 64,
            1,
            1,
        );
    }

    /// BN=8 int8 DP4a Q4_K dense-down tail for Gemma 4 31B short prompts.
    /// Descriptor offsets select the tail token slice of the pre-quantized
    /// Q8_1 activation buffers; d_offset keeps the f32 output token-major.
    pub fn recordMulMmQ4KTail8Dp4a(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        push_desc_fn: ?PushDescriptorFn,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_packed_buf: vk.c.VkBuffer,
        b_packed_size: vk.c.VkDeviceSize,
        b_scale_dsum_buf: vk.c.VkBuffer,
        b_scale_dsum_size: vk.c.VkDeviceSize,
        d_buf: vk.c.VkBuffer,
        d_size: vk.c.VkDeviceSize,
        M: u32,
        N: u32,
        K: u32,
        a_offset: u32,
        b_packed_offset: u32,
        b_scale_dsum_offset: u32,
        d_offset: u32,
    ) !void {
        const pip = blk: {
            if (K == 2816) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k2816_n8) |*p| break :blk p;
            }
            if (K == 5376) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k5376_n8) |*p| break :blk p;
            }
            if (K == 21504) {
                if (self.pipeline_mul_mm_q4k_full_dp4a_k21504_n8) |*p| break :blk p;
            }
            return error.PipelineNotLoaded;
        };
        if (K == 0 or (K & 255) != 0) return error.InvalidArgument;
        if (M == 0 or N == 0 or N > 8 or (M & 31) != 0) return error.InvalidArgument;
        const b_packed_byte_offset: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, b_packed_offset) * @sizeOf(u32);
        const b_scale_dsum_byte_offset: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, b_scale_dsum_offset) * 2 * @sizeOf(f32);
        if (b_packed_byte_offset >= b_packed_size or b_scale_dsum_byte_offset >= b_scale_dsum_size) return error.InvalidArgument;
        const push = MulMmQ4KGateUpDp4aPush{
            .M = M,
            .N = N,
            .K = K,
            .stride_b_packed = K / 4,
            .stride_b_scale = K / 32,
            .stride_d = M,
            .a_offset = a_offset,
            .d_offset = d_offset,
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = a_buf, .offset = 0, .range = a_size },
            .{ .buffer = b_packed_buf, .offset = b_packed_byte_offset, .range = b_packed_size - b_packed_byte_offset },
            .{ .buffer = b_scale_dsum_buf, .offset = b_scale_dsum_byte_offset, .range = b_scale_dsum_size - b_scale_dsum_byte_offset },
            .{ .buffer = d_buf, .offset = 0, .range = d_size },
        };
        cmd.pushDescAndDispatch(
            pip,
            push_desc_fn,
            infos[0..],
            std.mem.asBytes(&push),
            M / 32,
            1,
            1,
        );
    }

    /// Destroy the loaded pipelines and descriptor pool.
    /// @param self Dispatch wrapper to tear down in place.
    pub fn deinit(self: *DmmvDispatch) void {
        if (self.pipeline_q4k) |*p| p.deinit();
        if (self.pipeline_q4k_wide) |*p| p.deinit();
        if (self.pipeline_q5_1) |*p| p.deinit();
        if (self.pipeline_q5k) |*p| p.deinit();
        if (self.pipeline_q6k) |*p| p.deinit();
        if (self.pipeline_q6k_wide) |*p| p.deinit();
        if (self.pipeline_q8_0) |*p| p.deinit();
        if (self.pipeline_q8_0_batch) |*p| p.deinit();
        if (self.pipeline_q8_0_kpar_batch) |*p| p.deinit();
        if (self.pipeline_q8_0_spec64) |*p| p.deinit();
        if (self.pipeline_q8_0_spec128) |*p| p.deinit();
        if (self.pipeline_q8_0_wide) |*p| p.deinit();
        if (self.pipeline_q8_0_wide4) |*p| p.deinit();
        if (self.pipeline_q8_0_q8_1) |*p| p.deinit();
        if (self.pipeline_q4k_q8_1) |*p| p.deinit();
        if (self.pipeline_q8_0_fused_pair) |*p| p.deinit();
        if (self.pipeline_f16) |*p| p.deinit();
        if (self.pipeline_f32) |*p| p.deinit();
        if (self.pipeline_q4k_batch) |*p| p.deinit();
        if (self.pipeline_q4k_batch_kpar) |*p| p.deinit();
        if (self.pipeline_q6k_batch) |*p| p.deinit();
        if (self.pipeline_q6k_batch_kpar) |*p| p.deinit();
        if (self.pipeline_q4k_moe) |*p| p.deinit();
        if (self.pipeline_q4k_moe_kpar) |*p| p.deinit();
        if (self.pipeline_q4k_moe_cols) |*p| p.deinit();
        if (self.pipeline_q4k_moe_cols_q8_1) |*p| p.deinit();
        if (self.pipeline_q4k_moe_batched) |*p| p.deinit();
        if (self.pipeline_q4k_fused_gate_up_moe) |*p| p.deinit();
        if (self.pipeline_q4k_fused_gate_up_moe_spec8) |*p| p.deinit();
        if (self.pipeline_q4k_fused_gate_up_swiglu_moe) |*p| p.deinit();
        if (self.pipeline_q4k_fused_gate_up_swiglu_moe_spec8) |*p| p.deinit();
        if (self.pipeline_q4k_fused_gate_up_swiglu) |*p| p.deinit();
        if (self.pipeline_q4k_fused_gate_up_swiglu_row1) |*p| p.deinit();
        if (self.pipeline_q4k_fused_gate_up_geglu) |*p| p.deinit();
        if (self.pipeline_q4k_fused_gate_up_geglu_pair) |*p| p.deinit();
        if (self.pipeline_q4k_moe_fused_gate_up_geglu) |*p| p.deinit();
        if (self.pipeline_q4k_moe_fused_gate_up_geglu_batch_top1) |*p| p.deinit();
        if (self.pipeline_q4k_moe_fused_gate_up_geglu_cols_top1) |*p| p.deinit();
        if (self.pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1) |*p| p.deinit();
        if (self.pipeline_q4k_moe_fused_gate_up_swiglu_cols_top1_q8_1) |*p| p.deinit();
        if (self.pipeline_q8_0_fused_gate_up_swiglu) |*p| p.deinit();
        if (self.pipeline_q8_0_fused_gate_up_swiglu_gate) |*p| p.deinit();
        if (self.pipeline_q8_0_fused_gate_up_geglu) |*p| p.deinit();
        if (self.pipeline_q8_0_fused_gate_up_geglu4) |*p| p.deinit();
        if (self.pipeline_q8_0_sigmoid_acc) |*p| p.deinit();
        if (self.pipeline_q4k_o_proj_merge) |*p| p.deinit();
        if (self.pipeline_q4k_moe_fused_down_acc) |*p| p.deinit();
        if (self.pipeline_q5k_moe_fused_down_acc) |*p| p.deinit();
        if (self.pipeline_q5k_moe_fused_down_acc_q8_1) |*p| p.deinit();
        if (self.pipeline_q5k_moe) |*p| p.deinit();
        if (self.pipeline_q5k_moe_kpar) |*p| p.deinit();
        if (self.pipeline_q5k_moe_cols) |*p| p.deinit();
        if (self.pipeline_q5k_moe_cols_q8_1) |*p| p.deinit();
        if (self.pipeline_q5_1_moe_cols) |*p| p.deinit();
        if (self.pipeline_q5_1_acc) |*p| p.deinit();
        if (self.pipeline_q5_1_moe_fused_down_acc) |*p| p.deinit();
        if (self.pipeline_q5_1_moe_fused_down_acc_scaled) |*p| p.deinit();
        if (self.pipeline_q5_1_moe_down_acc_scaled_batch_top1) |*p| p.deinit();
        if (self.pipeline_q8_0_moe_fused_down_acc_scaled) |*p| p.deinit();
        if (self.pipeline_q6k_moe) |*p| p.deinit();
        if (self.pipeline_q6k_moe_cols) |*p| p.deinit();
        if (self.pipeline_quantize_q8_1) |*p| p.deinit();
        if (self.pipeline_count_experts) |*p| p.deinit();
        if (self.pipeline_moe_route_pack) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k) |*p| p.deinit();
        if (self.pipeline_mul_mm_id_q4k) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_down_acc) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_down_acc_wide) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_tail8) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_full) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_tail8) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_down_acc) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_tail8) |*p| p.deinit();
        if (self.pipeline_mul_mm_q5k) |*p| p.deinit();
        if (self.pipeline_mul_mm_q5k_wide) |*p| p.deinit();
        if (self.pipeline_mul_mm_q8_0) |*p| p.deinit();
        if (self.pipeline_mul_mm_q8_0_full_dp4a) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k12288) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k12288_n64_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bk2_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k12288_n64_bm64_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_mmq64_acc) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bm64_acc_interleaved) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_bk2_acc) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n64_ragged_bm64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k17408_n40) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k21504) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k21504_n64_bm64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k21504_n64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k21504_n8) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_k21504_n72) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_q8_1) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_bm64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n64_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q6k_full_dp4a_q8_1_k5120_n40) |*p| p.deinit();
        if (self.pipeline_quantize_act_q8) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_n64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_k5376_n8) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_n1_dp4a_q8) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k4096_n64_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_bm64_interleaved) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n64_ragged_bm64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_k5120_n40) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_n64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_geglu_full_dp4a_q8_1_k5376_n8) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k4096_n64_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_bm64_interleaved) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n64_ragged_bm64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_gate_up_swiglu_full_dp4a_q8_1_k5120_n40) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k2816_n8) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k5376_n8) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k5120_n64_bm64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k5120_n64_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k12288) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k12288_n64_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bk2_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k12288_n64_bm64_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k5120_n40) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_mmq64_acc) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bm64_acc_interleaved) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_bk2_acc) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n64_ragged_bm64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k17408_n40) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k21504) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k21504_n64) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k_full_dp4a_k21504_n8) |*p| p.deinit();
        if (self.pipeline_mul_mm_q5k_full_dp4a) |*p| p.deinit();
        if (self.pipeline_mul_mm_q5k_full_dp4a_k4096_bk2) |*p| p.deinit();
        if (self.pipeline_mul_mm_q5k_full_dp4a_k6144_n40) |*p| p.deinit();
        if (self.pipeline_quantize_act_q8_1) |*p| p.deinit();
        if (self.pipeline_mxfp4) |*p| p.deinit();
        vk.c.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.* = undefined;
    }
};
