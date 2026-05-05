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

/// Push constants for the quantize_q8_1 shader.
/// `ne` = number of f32 input elements (must be a multiple of 32).
/// `num_blocks` = ne / 32. Pass explicitly so the shader does not have to divide.
pub const QuantizeQ8_1Push = extern struct {
    ne: u32,
    num_blocks: u32,
};

/// Push constants for `count_experts.comp` (effort-6 Step 3 helper). Mirrors
/// llama.cpp's count_experts push so the shader is structurally identical
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
    /// MXFP4 pipeline, or null.
    pipeline_mxfp4: ?Pipeline,
    /// Q5_0 pipeline, or null.
    pipeline_q5_0: ?Pipeline,
    /// Q5_1 pipeline, or null.
    pipeline_q5_1: ?Pipeline,
    /// Q8 0 pipeline, or null.
    pipeline_q8_0: ?Pipeline,
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
    /// Fused gate+up Q4_K MoE pipeline (6 bindings: W_gate, W_up, X, Y_gate,
    /// Y_up, routing). Halves the dispatch count for the MoE gate+up phase
    /// and reads the shared input once per block.
    pipeline_q4k_fused_gate_up_moe: ?Pipeline,
    /// Fused gate+up+SwiGLU Q4_K dense pipeline (4 bindings: W_gate, W_up,
    /// X, swiglu_out). Replaces the dense FFN front-end dispatch trio
    /// (gate DMMV + up DMMV + swiglu element-wise) with a single dispatch
    /// that computes silu(W_gate·x) * (W_up·x) inline. Eliminates gate_buf
    /// and up_buf from the dense decode datapath. Saves one global compute
    /// barrier per layer (gate+up → swiglu), and reads the shared input
    /// once per block. Same DmmvPushConstants layout as pipeline_q4k.
    pipeline_q4k_fused_gate_up_swiglu: ?Pipeline,
    /// Q8_0 sibling of pipeline_q4k_fused_gate_up_swiglu. Targets the
    /// shared expert in Qwen 3.5 / 3.6 MoE packs where shared FFN
    /// weights ship as Q8_0 (rather than Q4_K). Same 4-binding layout
    /// (W_gate, W_up, X, swiglu_out) and same DmmvPushConstants. Used
    /// to fuse the per-token shared-expert (gate DMMV + up DMMV +
    /// SwiGLU) trio into one dispatch.
    pipeline_q8_0_fused_gate_up_swiglu: ?Pipeline,
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
    /// MoE Q5K pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_q5k_moe: ?Pipeline,
    /// Experimental K-parallel Q5K MoE pipeline (same 4 bindings, wave64 subgroupAdd).
    pipeline_q5k_moe_kpar: ?Pipeline,
    /// MoE MXFP4 pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_mxfp4_moe: ?Pipeline,
    /// MoE Q5_1 pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_q5_1_moe: ?Pipeline,
    /// MoE Q6K pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_q6k_moe: ?Pipeline,
    /// Foundation for future mul_mmq work: quantize an F32 activation into
    /// Q8_1 blocks. 2 bindings (A f32 vec4 in, D u32 stream out), push
    /// constants {ne, num_blocks}. Activation pre-quantizer preserved as
    /// reusable infrastructure (per effort-6 plan); the SSM proj mmq DMMV
    /// consumers (dmmv_q8_0_q8_1, dmmv_q4k_q8_1) were measured flat in
    /// cycles 8 + 33 of the loop and removed in cycle 60's pivot.
    pipeline_quantize_q8_1: ?Pipeline,
    /// Effort-6 Step 3 foundation for the MUL_MAT_ID tiled GEMM port.
    /// Counts how many tokens were routed to each expert. Output is a
    /// `[n_experts]` u32 buffer that the future MUL_MAT_ID GEMM uses for the
    /// per-expert early-exit. Pipeline loads at startup; no production
    /// callers until the tiled GEMM (Steps 1+2) lands. 2 bindings (A routing
    /// in, D counts out), push = CountExpertsPush.
    pipeline_count_experts: ?Pipeline,
    /// Effort-6 Step 1 of 5 foundation: tiled Q4_K dense GEMM. First port
    /// of llama.cpp's mul_mm.comp #ifndef COOPMAT branch adapted to ZINC's
    /// wave64 / Q4_K conventions. WG produces a 32×16 output tile with 64
    /// threads; BK=32 (one Q4_K sub-block) per outer step. Pipeline loads
    /// at startup; this cycle wires it under ZINC_MUL_MM_LM_HEAD=1 to the
    /// final-tail LM head dispatch as a correctness-exercise (LM head
    /// fires once per prefill so the perf-hotpath impact is small). Step 2
    /// will add the MUL_MAT_ID variant. 3 bindings (A weights, B f32
    /// activations, D f32 outputs), push = MulMmQ4KPush.
    pipeline_mul_mm_q4k: ?Pipeline,
    /// Descriptor pool for this dispatch.
    descriptor_pool: vk.c.VkDescriptorPool,
    /// Logical device.
    device: vk.c.VkDevice,

    /// Create the DMMV dispatch wrapper and load the supported quantized pipelines.
    /// @param instance Active Vulkan instance and logical device.
    /// @param gpu_config Derived GPU tuning parameters.
    /// @param shader_dir Directory containing compiled SPIR-V shader binaries.
    /// @param allocator Allocator used for temporary pipeline creation state.
    /// @returns A DmmvDispatch ready to record projection work.
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
        const push_desc_wave64_options = pipeline_mod.PipelineOptions{
            .required_subgroup_size = 64,
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
        const pipeline_q4k_batch_kpar = pipeline_mod.createFromSpirvWithOptions(instance, q4k_batch_kpar_path, 3, batch_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K batch kpar shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q6k_batch_kpar_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q6k_batch_kpar.spv", .{shader_dir}) catch unreachable;
        const pipeline_q6k_batch_kpar = pipeline_mod.createFromSpirvWithOptions(instance, q6k_batch_kpar_path, 3, batch_push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
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

        // Dense fused gate+up+SwiGLU: reads ffn_norm_buf once per block,
        // dot-products against W_gate and W_up, applies silu(gate)*up
        // inline, and writes a single swiglu_buf row. 4 bindings (W_gate,
        // W_up, X, swiglu). Same DmmvPushConstants as pipeline_q4k.
        const q4k_fused_gate_up_swiglu_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_fused_gate_up_swiglu.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_fused_gate_up_swiglu = pipeline_mod.createFromSpirvWithOptions(instance, q4k_fused_gate_up_swiglu_path, 4, push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q4_K dense fused gate+up+SwiGLU shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Q8_0 fused gate+up+SwiGLU. 4 bindings, same push struct as the
        // Q4_K variant. Used by the shared expert path on Qwen 3.5 / 3.6
        // MoE packs where shared FFN weights are Q8_0 (rather than Q4_K).
        const q8_0_fused_gate_up_swiglu_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_fused_gate_up_swiglu.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_fused_gate_up_swiglu = pipeline_mod.createFromSpirvWithOptions(instance, q8_0_fused_gate_up_swiglu_path, 4, push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 fused gate+up+SwiGLU shader not loaded: {s}", .{@errorName(err)});
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

        const q5k_moe_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5k_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5k_moe = pipeline_mod.createFromSpirvWithOptions(instance, q5k_moe_path, 4, moe_push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q5_K MoE shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // K-parallel Q5_K MoE variant: wave64 subgroupAdd reduction (targets the
        // ~713 ms MoE down bucket in the Qwen3.5-35B flagship prefill).
        const q5k_moe_kpar_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5k_moe_kpar.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5k_moe_kpar = pipeline_mod.createFromSpirvWithOptions(instance, q5k_moe_kpar_path, 4, moe_push_size, &.{}, effective_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_K MoE kpar shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

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

        const q6k_moe_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q6k_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_q6k_moe = pipeline_mod.createFromSpirvWithOptions(instance, q6k_moe_path, 4, moe_push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q6_K MoE shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

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

        // Effort 6 Step 3: foundation helper for MUL_MAT_ID GEMM port.
        // Counts tokens-per-expert from a per-(token, layer) routing buffer.
        // 2 bindings (A routing in, D counts out), push = CountExpertsPush.
        const count_experts_push_size = @sizeOf(CountExpertsPush);
        const count_experts_path = std.fmt.bufPrint(&path_buf, "{s}/count_experts.spv", .{shader_dir}) catch unreachable;
        const pipeline_count_experts = pipeline_mod.createFromSpirvWithOptions(instance, count_experts_path, 2, count_experts_push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("count_experts shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_count_experts != null) {
            log.info("count_experts pipeline loaded (MUL_MAT_ID GEMM foundation; not yet wired)", .{});
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

        return DmmvDispatch{
            .pipeline_q4k = pipeline_q4k,
            .pipeline_q4k_wide = pipeline_q4k_wide,
            .pipeline_mxfp4 = pipeline_mxfp4,
            .pipeline_q5_0 = pipeline_q5_0,
            .pipeline_q5_1 = pipeline_q5_1,
            .pipeline_q5k = pipeline_q5k,
            .pipeline_q6k = pipeline_q6k,
            .pipeline_q8_0 = pipeline_q8_0,
            .pipeline_f16 = pipeline_f16,
            .pipeline_f32 = pipeline_f32,
            .pipeline_q4k_batch = pipeline_q4k_batch,
            .pipeline_q4k_batch_kpar = pipeline_q4k_batch_kpar,
            .pipeline_q6k_batch = pipeline_q6k_batch,
            .pipeline_q6k_batch_kpar = pipeline_q6k_batch_kpar,
            .pipeline_q4k_moe = pipeline_q4k_moe,
            .pipeline_q4k_moe_kpar = pipeline_q4k_moe_kpar,
            .pipeline_q4k_moe_batched = pipeline_q4k_moe_batched,
            .pipeline_q4k_fused_gate_up_moe = pipeline_q4k_fused_gate_up_moe,
            .pipeline_q4k_fused_gate_up_swiglu = pipeline_q4k_fused_gate_up_swiglu,
            .pipeline_q8_0_fused_gate_up_swiglu = pipeline_q8_0_fused_gate_up_swiglu,
            .pipeline_q8_0_sigmoid_acc = pipeline_q8_0_sigmoid_acc,
            .pipeline_q4k_o_proj_merge = pipeline_q4k_o_proj_merge,
            .pipeline_q4k_moe_fused_down_acc = pipeline_q4k_moe_fused_down_acc,
            .pipeline_q5k_moe_fused_down_acc = pipeline_q5k_moe_fused_down_acc,
            .pipeline_mxfp4_moe = pipeline_mxfp4_moe,
            .pipeline_q5_1_moe = pipeline_q5_1_moe,
            .pipeline_q5k_moe = pipeline_q5k_moe,
            .pipeline_q5k_moe_kpar = pipeline_q5k_moe_kpar,
            .pipeline_q6k_moe = pipeline_q6k_moe,
            .pipeline_quantize_q8_1 = pipeline_quantize_q8_1,
            .pipeline_count_experts = pipeline_count_experts,
            .pipeline_mul_mm_q4k = pipeline_mul_mm_q4k,
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

    /// Select the MoE quantization-specific pipeline (4 bindings: A, x, y, routing).
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

    /// Record a batched MoE DMMV dispatch — all experts run in parallel via Y workgroups.
    /// expert_stride: bytes per expert in stacked weight tensor.
    /// n_experts_y: number of experts to process (dispatched as Y workgroups).
    /// x_expert_stride: elements between experts' inputs (0=shared input, K=per-expert).
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
    /// @note The helper uses one workgroup per 64 output rows.
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

    /// Push-descriptor batch DMMV dispatch.
    /// Bindings order: 0 = A (weight), 1 = X_batch (K × num_cols, column-major),
    /// 2 = Y_batch (M × num_cols, column-major).
    /// Returns error.UnsupportedQuantType if the batch shader isn't loaded for this quant type.
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
    /// Tile shape: WG = 64 threads producing a 32 × 16 output tile.
    /// Dispatch grid: ((M+31)/32) × ((N+15)/16) × 1.
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
        // BM=32, BN=16 in the shader; keep these in sync.
        const wg_x = (M + 31) / 32;
        const wg_y = (N + 15) / 16;
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

    /// Destroy the loaded pipelines and descriptor pool.
    /// @param self Dispatch wrapper to tear down in place.
    pub fn deinit(self: *DmmvDispatch) void {
        if (self.pipeline_q4k) |*p| p.deinit();
        if (self.pipeline_q4k_wide) |*p| p.deinit();
        if (self.pipeline_q5_1) |*p| p.deinit();
        if (self.pipeline_q5k) |*p| p.deinit();
        if (self.pipeline_q6k) |*p| p.deinit();
        if (self.pipeline_q8_0) |*p| p.deinit();
        if (self.pipeline_f16) |*p| p.deinit();
        if (self.pipeline_f32) |*p| p.deinit();
        if (self.pipeline_q4k_batch) |*p| p.deinit();
        if (self.pipeline_q4k_batch_kpar) |*p| p.deinit();
        if (self.pipeline_q6k_batch) |*p| p.deinit();
        if (self.pipeline_q6k_batch_kpar) |*p| p.deinit();
        if (self.pipeline_q4k_moe) |*p| p.deinit();
        if (self.pipeline_q4k_moe_kpar) |*p| p.deinit();
        if (self.pipeline_q4k_moe_batched) |*p| p.deinit();
        if (self.pipeline_q4k_fused_gate_up_moe) |*p| p.deinit();
        if (self.pipeline_q4k_fused_gate_up_swiglu) |*p| p.deinit();
        if (self.pipeline_q8_0_fused_gate_up_swiglu) |*p| p.deinit();
        if (self.pipeline_q8_0_sigmoid_acc) |*p| p.deinit();
        if (self.pipeline_q4k_o_proj_merge) |*p| p.deinit();
        if (self.pipeline_q4k_moe_fused_down_acc) |*p| p.deinit();
        if (self.pipeline_q5k_moe_fused_down_acc) |*p| p.deinit();
        if (self.pipeline_q5k_moe) |*p| p.deinit();
        if (self.pipeline_q5k_moe_kpar) |*p| p.deinit();
        if (self.pipeline_q6k_moe) |*p| p.deinit();
        if (self.pipeline_quantize_q8_1) |*p| p.deinit();
        if (self.pipeline_count_experts) |*p| p.deinit();
        if (self.pipeline_mul_mm_q4k) |*p| p.deinit();
        vk.c.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.* = undefined;
    }
};
