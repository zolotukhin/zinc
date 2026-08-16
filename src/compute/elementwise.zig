//! Wrap the fused element-wise shader family used by the decode loop.
//! @section Shader Dispatch
//! This helper loads the RMS norm, SwiGLU, and RoPE pipelines and records the
//! push constants needed for their dispatches.
const std = @import("std");
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const Pipeline = @import("../vulkan/pipeline.zig").Pipeline;
const pipeline_mod = @import("../vulkan/pipeline.zig");
const CommandBuffer = @import("../vulkan/command.zig").CommandBuffer;

const log = std.log.scoped(.elementwise);
const descriptor_pool_max_sets: u32 = 256;
const max_storage_buffers_per_set: u32 = 8;

/// Push constants for RMS norm shader.
pub const RmsNormPush = extern struct {
    N: u32,
    eps_bits: u32, // float bits reinterpreted as u32
};

/// Push constants for SwiGLU shader.
pub const SwigluPush = extern struct {
    N: u32,
};

/// Push constants for vector add shader.
const VaddPush = extern struct {
    N: u32,
};

/// Push constants for deinterleave shader.
pub const DeinterleavePush = extern struct {
    head_dim: u32,
    n_heads: u32,
};

/// Push constants for sigmoid multiply shader.
pub const SigmoidMulPush = extern struct {
    N: u32,
};

/// Push constants for scale-accumulate shader.
pub const ScaleAccPush = extern struct {
    N: u32,
    scale_bits: u32, // float reinterpreted as u32
};

/// Push constants for bias add shader.
pub const BiasAddPush = extern struct {
    N: u32,
    src_offset: u32,
};

/// Push constants for RoPE shader (with partial rotation / IMRoPE support).
pub const RopePush = extern struct {
    stride: u32, // full head dimension (distance between heads in memory)
    rope_dim: u32, // number of dimensions to rotate (<= stride)
    n_heads: u32,
    position: u32,
    freq_base_bits: u32, // float bits reinterpreted as u32
    attn_scale_bits: u32, // YaRN magnitude scale (1.0 for plain RoPE)
};

/// Push constants for rope_batched (multi-token prefill variant).
/// Layout mirrors src/shaders/rope_batched.comp.
pub const RopeBatchedPush = extern struct {
    stride: u32,
    rope_dim: u32,
    n_heads: u32,
    position_base: u32,
    freq_base_bits: u32,
    attn_scale_bits: u32,
};

/// Push constants for SSM conv1d + SiLU shader.
pub const SsmConv1dPush = extern struct {
    conv_channels: u32,
    d_conv: u32,
    kernel_is_f16: u32,
    // Circular state buffer rotation (0..d_conv-2). Host advances per
    // token; reset to 0 when ssm conv state is zeroed (resetRequestState).
    state_offset: u32,
};

/// Push constants for the batched SSM conv1d shader.
pub const SsmConv1dBatchedPush = extern struct {
    conv_channels: u32,
    d_conv: u32,
    kernel_is_f16: u32,
    state_offset: u32,
    n_tokens: u32,
};

/// Push constants for batched f32 dual DMMV (SSM alpha/beta).
pub const F32DualBatchPush = extern struct {
    M: u32,
    K: u32,
    stride_x: u32,
    stride_y: u32,
};

/// Push constants for SSM delta-net state update shader.
pub const SsmDeltaNetPush = extern struct {
    d_inner: u32,
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    n_group: u32,
    ssm_a_is_f16: u32,
    dt_bias_is_f16: u32,
    has_dt_bias: u32,
    has_ssm_a: u32,
    // A3: token-loop fold inside the shader. n_tok=1 keeps the shader
    // structurally equivalent to the pre-A3 form (state hoisted to
    // registers but only one iteration). n_tok>1 amortizes one
    // state-buffer DRAM round-trip across n_tok prefill tokens.
    n_tok: u32,
    conv_stride_tok: u32, // floats: 2*qk_dim + d_inner
    ab_stride_tok: u32, // floats: dt_rank
    y_stride_tok: u32, // floats: d_inner
};

/// Push constants for the SSM Q/K RMS-norm shader. Drives the per-group
/// normalization applied to query and key projections inside Mamba/SSM blocks.
pub const SsmQkNormPush = extern struct {
    d_state: u32,
    n_group: u32,
    qk_dim: u32,
};

/// Push constants for SSM gated norm shader.
pub const SsmGatedNormPush = extern struct {
    d_inner: u32,
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    norm_per_head: u32,
    n_tok: u32 = 1,
};

/// Push constants for softmax + top-k MoE router shader.
pub const SoftmaxTopkPush = extern struct {
    n_experts: u32,
    k: u32,
    /// Optional positive scale applied to selected logits before softmax.
    /// Use floatBitsToInt(1.0) for the normal unscaled path.
    scale_bits: u32 = @bitCast(@as(f32, 1.0)),
};

/// Push constants for token-batched f32 router matvec.
pub const RouterF32BatchPush = extern struct {
    M: u32,
    K: u32,
    n_tokens: u32,
    stride_x: u32,
    stride_y: u32,
};

/// Push constants for token-batched Gemma router RMS norm + scale + f32 DMMV.
pub const RmsNormScaleDmmvF32BatchPush = extern struct {
    M: u32,
    K: u32,
    n_tokens: u32,
    eps_bits: u32,
};

/// Push constants for token-batched MoE top-k.
pub const SoftmaxTopkBatchPush = extern struct {
    n_experts: u32,
    k: u32,
    scale_bits: u32,
    token_base: u32,
    n_tokens: u32,
    logits_stride: u32,
    output_stride: u32,
};

/// Push constants for batched MoE weighted accumulate shader.
/// Sums all expert outputs at once: a[i] = sum_j(weight_j * b[j*src_stride+i]).
pub const MoeWeightedAccPush = extern struct {
    N: u32,
    n_used: u32,
    src_stride: u32,
};

/// Push constants for the **batched** MoE weighted-accumulate shader.
/// Sums each token's `n_used` selected-expert outputs across a token batch in one
/// dispatch: `a[t,i] = sum_j(weight_{t,j} * b[...])`.
pub const MoeWeightedAccBatchPush = extern struct {
    hidden_dim: u32,
    n_tokens: u32,
    n_used: u32,
    routing_stride: u32,
    routing_token_base: u32,
    accum_token_base: u32,
};

/// Push constants for the Gemma batched MoE weighted-accumulate shader.
/// Same route-major contract as `MoeWeightedAccBatchPush`, with an extra
/// per-expert scale offset for model-specific down-projection scales.
pub const MoeWeightedAccScaledBatchPush = extern struct {
    hidden_dim: u32,
    n_tokens: u32,
    n_used: u32,
    routing_stride: u32,
    routing_token_base: u32,
    accum_token_base: u32,
    scale_offset: u32,
};

/// Push constants for the **batched** `sigmoid_scale_acc` shader.
/// Applies a per-token sigmoid-gated shared-expert add across a token batch:
/// `accum[t,i] += sigmoid(gate_t) * src[t,i]`.
pub const SigmoidScaleAccBatchPush = extern struct {
    hidden_dim: u32,
    n_tokens: u32,
    accum_token_base: u32,
};

/// Push constants for KV cache write compute shader.
pub const KvCacheWritePush = extern struct {
    kv_dim: u32,
    dst_offset: u32,
};

/// Push constants for batched KV cache write (prefillBatched path).
/// Matches src/shaders/kv_cache_write_batched.comp.
pub const KvCacheWriteBatchedPush = extern struct {
    kv_dim: u32,
    n_tokens: u32,
    page_size: u32,
    base_token: u32,
};

/// Push constants for fused residual-add + RMS norm
/// (src/shaders/residual_rms_norm.comp). One dispatch per `n_tokens`
/// workgroups replaces a scale_accumulate → barrier → rms_norm_mul chain.
pub const ResidualRmsNormPush = extern struct {
    n: u32,
    eps_bits: u32,
    scale_bits: u32,
};

/// Push constants for fused post-norm + residual-add + RMS norm
/// (src/shaders/post_norm_residual_rms_norm.comp). One dispatch replaces
/// Gemma's post_attention_norm -> barrier -> residual_rms_norm sequence.
pub const PostNormResidualRmsNormPush = extern struct {
    n: u32,
    eps: f32,
    hidden_scale: f32 = 1.0,
};

/// Push constants for fused residual-add + RMS norm + Q8_1 activation quantize
/// (src/shaders/residual_rms_norm_quant_q8_1.comp). Same residual/RMS math as
/// ResidualRmsNormPush, but also emits packed int8 lanes + (scale, dsum) so
/// the downstream Qwen3.6-27B dense FFN DP4a gate+up GEMM can skip its
/// separate quantize_act_q8_1 dispatch.
pub const ResidualRmsNormQuantQ8_1Push = extern struct {
    n: u32,
    eps_bits: u32,
    scale_bits: u32,
    blocks_per_token: u32,
    stride_packed: u32,
    write_norm_out: u32,
};

/// Push constants for fused RMS norm + RoPE shader.
pub const NormRopePush = extern struct {
    head_dim: u32,
    rope_dim: u32,
    n_heads: u32,
    position: u32,
    freq_base_bits: u32,
    attn_scale_bits: u32,
    eps_bits: u32,
};

/// Push constants for fused rmsnorm(src) + hidden accumulate shader
/// (src/shaders/rms_norm_add.comp). Used by Gemma prefillBatched to fold
/// post_ffw_norm + residual add into one dispatch.
pub const RmsNormAddPush = extern struct {
    n: u32,
    eps: f32,
};

/// Push constants for fused RMS norm + f32 router DMMV shader
/// (src/shaders/rms_norm_dmmv_f32.comp). Folds the per-MoE-layer
/// rms_norm_mul → router DMMV pair into a single dispatch on
/// architectures whose router weights are f32 (Qwen 3.5/3.6 etc).
pub const RmsNormDmmvF32Push = extern struct {
    M: u32, // router output rows (= n_experts)
    K: u32, // hidden_dim
    eps_bits: u32, // RMS norm epsilon (f32 bits)
};

/// Push constants for fused RMS norm + Q4_K alpha+beta SSM proj DMMV
/// (src/shaders/rms_norm_dmmv_q4k_alpha_beta.comp). Folds the
/// per-SSM-layer (rms_norm_mul → alpha DMMV → beta DMMV) trio into a
/// single dispatch on the qwen35moe / qwen36moe SSM proj fast path.
pub const RmsNormDmmvQ4kAlphaBetaPush = extern struct {
    M: u32, // alpha rows == beta rows (= dt_rank)
    K: u32, // hidden_dim (must be multiple of 256 for Q4_K)
    eps_bits: u32, // RMS norm epsilon (f32 bits)
};

/// Push constants for fused Q+K norm + RoPE + KV cache write shader
/// (src/shaders/qk_norm_rope_kv_write.comp). Folds the per-attention-layer
/// (Q norm+rope → K norm+rope → kv_cache_write) trio on Qwen 3 family
/// dense attention into a single dispatch.
pub const QkNormRopeKvWritePush = extern struct {
    head_dim: u32,
    rope_dim: u32,
    n_q_heads: u32,
    n_k_heads: u32,
    position: u32,
    freq_base_bits: u32, // 0 ⇒ use freq buffer
    attn_scale_bits: u32, // 0 ⇒ scale = 1.0
    eps_bits: u32,
    dst_offset: u32, // physical_token * kv_dim (in floats)
    v_norm: u32 = 0, // 1 ⇒ unit-RMS-normalize V while writing kv_v
};

/// Batched SWA variant of QkNormRopeKvWritePush.
/// Binding 4 is the KV page table, so this variant computes RoPE frequencies
/// from freq_base_bits instead of reading a frequency buffer.
pub const QkNormRopeKvWriteBatchedPush = extern struct {
    head_dim: u32,
    rope_dim: u32,
    n_q_heads: u32,
    n_k_heads: u32,
    n_tokens: u32,
    page_size: u32,
    base_token: u32,
    freq_base_bits: u32,
    attn_scale_bits: u32,
    eps_bits: u32,
    v_norm: u32 = 0,
};

/// Batched full-attention K/V sibling used when Q must keep the precomputed
/// RoPE frequency buffer binding. It fuses K RMS norm, K RoPE, optional V unit
/// norm, and paged KV cache write. Q norm/RoPE stays on the existing path.
pub const KNormRopeKvWriteBatchedPush = extern struct {
    head_dim: u32,
    rope_dim: u32,
    n_k_heads: u32,
    n_tokens: u32,
    page_size: u32,
    base_token: u32,
    freq_base_bits: u32,
    attn_scale_bits: u32,
    eps_bits: u32,
    v_norm: u32 = 0,
};

/// Manages element-wise fused kernel pipelines.
pub const ElementwiseDispatch = struct {
    /// RMS NORM pipeline, or null.
    pipeline_rms_norm: ?Pipeline,
    /// RMS norm plus hidden-store pipeline for Qwen3.6 27B prefix partial decode.
    pipeline_rms_norm_store_hidden: ?Pipeline,
    /// SWIGLU pipeline, or null.
    pipeline_swiglu: ?Pipeline,
    /// OAI SWIGLU pipeline (gpt-oss), or null.
    pipeline_swiglu_oai: ?Pipeline,
    /// GEGLU pipeline (GELU-gated, used by Gemma), or null.
    pipeline_geglu: ?Pipeline,
    /// ROPE pipeline, or null.
    pipeline_rope: ?Pipeline,
    /// Batched RoPE pipeline used by the RDNA prefillBatched path, or null.
    pipeline_rope_batched: ?Pipeline,
    /// DEINTERLEAVE pipeline, or null.
    pipeline_deinterleave: ?Pipeline,
    /// Token-batched DEINTERLEAVE pipeline, or null. Used by Qwen3.6-27B
    /// layer-major full-attn prefill to split packed Q+gate across n_tokens
    /// in one dispatch.
    pipeline_deinterleave_batched: ?Pipeline,
    /// SIGMOID MUL pipeline, or null.
    pipeline_sigmoid_mul: ?Pipeline,
    /// VADD pipeline, or null.
    pipeline_vadd: ?Pipeline,
    /// SCALE ACC pipeline, or null.
    pipeline_scale_acc: ?Pipeline,
    /// BIAS ADD pipeline, or null.
    pipeline_bias_add: ?Pipeline,
    /// In-place scale pipeline: `data[i] *= scale` (1 binding, Gemma 4 per-layer output scaling).
    pipeline_scale_in_place: ?Pipeline,
    /// Element-wise multiply pipeline: `a[i] *= b[i]` (2 bindings, used for ffn_gate_inp.scale).
    pipeline_mul_elementwise: ?Pipeline,
    /// Per-expert scale pipeline: `down[i] *= scales[expert] * routing[expert]` (3 bindings).
    pipeline_per_expert_scale: ?Pipeline,
    /// SSM CONV1D pipeline, or null.
    pipeline_ssm_conv1d: ?Pipeline,
    /// Batched SSM CONV1D pipeline, or null.
    pipeline_ssm_conv1d_batched: ?Pipeline,
    /// Batched f32 alpha/beta SSM projection pipeline, or null.
    pipeline_dmmv_f32_dual_batch: ?Pipeline,
    /// In-place SSM Q/K normalization pipeline, or null.
    pipeline_ssm_qk_norm: ?Pipeline,
    /// SSM DELTA NET pipeline, or null.
    pipeline_ssm_delta_net: ?Pipeline,
    /// SSM DELTA NET cols8 pipeline, or null.
    pipeline_ssm_delta_net_cols8: ?Pipeline,
    /// SSM DELTA NET cols8 pipeline for pre-normalized Q/K, or null.
    pipeline_ssm_delta_net_cols8_normed: ?Pipeline,
    /// SSM GATED NORM pipeline, or null.
    pipeline_ssm_gated_norm: ?Pipeline,
    /// Effort-15 cycle 11: token-batched SSM GATED NORM pipeline. Identical
    /// 4-binding semantics to pipeline_ssm_gated_norm but the grid is
    /// (dt_rank, n_tokens, 1) — each workgroup handles one (head, token) pair
    /// instead of dispatching the per-token loop from the host. Used by
    /// prefillQwen36RunSsmLayerToFfnNorm when n_tokens > 1.
    pipeline_ssm_gated_norm_batch_tok: ?Pipeline,
    /// Fused token-loop variant: each WG processes one head across ALL tokens
    /// (grid is (dt_rank, 1, 1)), eliminating per-token WG launch overhead.
    pipeline_ssm_gated_norm_batch_tok_fused: ?Pipeline,
    /// SOFTMAX TOPK pipeline, or null.
    pipeline_softmax_topk: ?Pipeline,
    /// SOFTMAX TOPK v2 (subgroup-parallel reduction), or null.
    pipeline_softmax_topk_v2: ?Pipeline,
    /// TOP-1 MoE router fast path, or null.
    pipeline_softmax_top1: ?Pipeline,
    /// Token-batched top-1 MoE router fast path, or null.
    pipeline_softmax_top1_batch: ?Pipeline,
    /// Token-batched f32 MoE router matvec, or null.
    pipeline_router_f32_batch: ?Pipeline,
    /// Token-batched Gemma router RMS norm + scale + f32 DMMV, or null.
    pipeline_rms_norm_scale_dmmv_f32_batch: ?Pipeline,
    /// Token-batched MoE top-k, or null.
    pipeline_softmax_topk_batch: ?Pipeline,
    /// SIGMOID SCALE ACC pipeline: a[i] += sigmoid(c[0]) * b[i], 3 bindings.
    pipeline_sigmoid_scale_acc: ?Pipeline,
    /// MOE WEIGHTED ACC pipeline: a[i] += routing_weight * b[i], 3 bindings (accum, src, routing).
    pipeline_moe_weighted_acc: ?Pipeline,
    /// Batched route-major MoE weighted accumulate, 3 bindings.
    pipeline_moe_weighted_acc_batch: ?Pipeline,
    /// Batched route-major weighted accumulate with per-expert scale, 4 bindings.
    pipeline_moe_weighted_acc_scaled_batch: ?Pipeline,
    /// Batched per-token sigmoid-gated accumulate, 3 bindings.
    pipeline_sigmoid_scale_acc_batch: ?Pipeline,
    /// KV CACHE WRITE pipeline: compute-based KV cache copy, 4 bindings (k_src, k_dst, v_src, v_dst).
    pipeline_kv_cache_write: ?Pipeline,
    /// Batched KV cache write for prefillBatched (5 bindings, page-aware).
    pipeline_kv_cache_write_batched: ?Pipeline,
    /// Fused residual-add + RMS norm for prefillBatched (4 bindings).
    pipeline_residual_rms_norm: ?Pipeline,
    /// Fused Gemma post-attention norm + residual-add + FFN RMS norm (5 bindings).
    pipeline_post_norm_residual_rms_norm: ?Pipeline,
    /// Fused residual-add + RMS norm + Q8_1 quantize for the Qwen3.6-27B
    /// dense FFN prefill DP4a path (6 bindings: hidden(rw), residual,
    /// norm_out, weights, packed_i8, scale_dsum). Saves one quantize_act_q8_1
    /// dispatch + barrier per SSM-fed layer-major segment.
    pipeline_residual_rms_norm_quant_q8_1: ?Pipeline,
    /// Fused RMS norm + residual-accumulate (hidden += weight * rmsnorm(src)),
    /// 3 bindings (hidden, src, weights). Used by Gemma's post_ffw_norm tail.
    pipeline_rms_norm_add: ?Pipeline,
    /// Vec4 variant of pipeline_rms_norm_add for hidden dimensions divisible by 4.
    pipeline_rms_norm_add_vec4: ?Pipeline,
    /// NORM ROPE pipeline: fused RMS norm + RoPE per head, 3 bindings (data, weight, freq).
    pipeline_norm_rope: ?Pipeline,
    /// Fused RMS norm + f32 router DMMV pipeline (5 bindings: hidden,
    /// ffn_norm_weights, router_weights, ffn_norm_buf, router_logits_buf).
    /// Replaces (rms_norm_mul → router DMMV) for Qwen-style MoE layers
    /// whose router uses f32 weights. Saves 1 dispatch + 1 barrier per
    /// MoE layer (~30 layers on Qwen 3.6 35B-A3B).
    pipeline_rms_norm_dmmv_f32: ?Pipeline,
    /// Gemma router variant: unit RMS norm + ffn_gate_inp.scale + f32
    /// router DMMV in one dispatch. The expert/shared FFN norm remains
    /// separate because Gemma routes on a different normalized vector.
    pipeline_rms_norm_scale_dmmv_f32: ?Pipeline,
    /// Fused RMS norm + Q4_K alpha/beta SSM proj DMMV pipeline (7 bindings:
    /// hidden, attn_norm_w, alpha_w, beta_w, norm_buf, alpha_out, beta_out).
    /// Replaces (rms_norm_mul → alpha DMMV → beta DMMV) trio at the start
    /// of each SSM layer's proj phase on qwen35moe / qwen36moe (30 SSM
    /// layers each). WG 0 also writes norm_buf so downstream wqkv/z DMMVs
    /// see the pre-normalized hidden vector.
    pipeline_rms_norm_dmmv_q4k_alpha_beta: ?Pipeline,
    /// Fused Q+K norm + RoPE + KV cache write pipeline (8 bindings:
    /// q_data, q_norm_w, k_src, k_norm_w, freq_buf, kv_k_cache, v_src,
    /// kv_v_cache). Replaces (Q norm+rope → K norm+rope → kv_cache_write)
    /// on Qwen 3 dense attention layers, saving 2 dispatches + 1 barrier
    /// per attention layer.
    pipeline_qk_norm_rope_kv_write: ?Pipeline,
    /// Batched Gemma SWA sibling of pipeline_qk_norm_rope_kv_write. Binding 4 is
    /// the KV page table, so this path uses rope_freq_base_swa rather than a
    /// precomputed frequency buffer.
    pipeline_qk_norm_rope_kv_write_batched: ?Pipeline,
    /// Batched Gemma full-attention K/V sibling. Keeps the precomputed frequency
    /// buffer binding by leaving Q norm/RoPE separate and fusing only K/V cache
    /// production into one paged dispatch.
    pipeline_k_norm_rope_kv_write_batched: ?Pipeline,
    /// Descriptor pool for this dispatch.
    descriptor_pool: vk.c.VkDescriptorPool,
    /// Logical device.
    device: vk.c.VkDevice,

    /// Create the fused element-wise dispatch wrapper and load its shaders.
    /// @param instance Active Vulkan instance and logical device.
    /// @param shader_dir Directory containing compiled SPIR-V shader binaries.
    /// @param allocator Allocator used for temporary pipeline creation state.
    /// @returns An ElementwiseDispatch ready to record element-wise passes.
    pub fn init(
        /// Vulkan instance.
        instance: *const Instance,
        shader_dir: []const u8,
        /// Allocator for owned resources.
        allocator: std.mem.Allocator,
    ) !ElementwiseDispatch {
        const pool_size = vk.c.VkDescriptorPoolSize{
            .type = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            // The largest elementwise descriptor set binds 8 storage buffers.
            // Keep enough descriptors for runtime reuse plus rotating hot-bench
            // working sets.
            .descriptorCount = descriptor_pool_max_sets * max_storage_buffers_per_set,
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

        var path_buf: [512]u8 = undefined;
        const push_options = pipeline_mod.PipelineOptions{
            .push_descriptors = instance.push_descriptor_fn != null,
        };
        const push_wave64_options = pipeline_mod.PipelineOptions{
            .required_subgroup_size = 64,
            .require_full_subgroups = true,
            .push_descriptors = instance.push_descriptor_fn != null,
        };

        // RMS norm: 2 inputs (x, weight) + 1 output = 3 bindings
        const rms_path = std.fmt.bufPrint(&path_buf, "{s}/rms_norm_mul.spv", .{shader_dir}) catch unreachable;
        const pipeline_rms_norm = pipeline_mod.createFromSpirvWithOptions(instance, rms_path, 3, @sizeOf(RmsNormPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("rms_norm_mul shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        // Reuses the previously-unwired ssm_gated_norm_batched shader slot so
        // build.zig's existing shader manifest installs this prefix-only helper.
        const rms_store_path = std.fmt.bufPrint(&path_buf, "{s}/ssm_gated_norm_batched.spv", .{shader_dir}) catch unreachable;
        const pipeline_rms_norm_store_hidden = pipeline_mod.createFromSpirvWithOptions(instance, rms_store_path, 4, @sizeOf(RmsNormPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("rms_norm_store_hidden shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // SwiGLU: 2 inputs (gate, up) + 1 output = 3 bindings
        const swiglu_path = std.fmt.bufPrint(&path_buf, "{s}/swiglu.spv", .{shader_dir}) catch unreachable;
        const pipeline_swiglu = pipeline_mod.createFromSpirvWithOptions(instance, swiglu_path, 3, @sizeOf(SwigluPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("swiglu shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // GPT-OSS OAI SwiGLU: same bindings as SwiGLU, different activation.
        const swiglu_oai_path = std.fmt.bufPrint(&path_buf, "{s}/swiglu_oai.spv", .{shader_dir}) catch unreachable;
        const pipeline_swiglu_oai = pipeline_mod.createFromSpirvWithOptions(instance, swiglu_oai_path, 3, @sizeOf(SwigluPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("swiglu_oai shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // GEGLU: 2 inputs (gate, up) + 1 output = 3 bindings (same layout as SwiGLU)
        const geglu_path = std.fmt.bufPrint(&path_buf, "{s}/geglu.spv", .{shader_dir}) catch unreachable;
        const pipeline_geglu = pipeline_mod.createFromSpirvWithOptions(instance, geglu_path, 3, @sizeOf(SwigluPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("geglu shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // RoPE: 1 input + 1 output + 1 freq_buf = 3 bindings
        const rope_path = std.fmt.bufPrint(&path_buf, "{s}/rope_fused.spv", .{shader_dir}) catch unreachable;
        const pipeline_rope = pipeline_mod.createFromSpirvWithOptions(instance, rope_path, 3, @sizeOf(RopePush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("rope_fused shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // RoPE batched: same 3 bindings, processes N tokens per dispatch via grid.y.
        const rope_batched_path = std.fmt.bufPrint(&path_buf, "{s}/rope_batched.spv", .{shader_dir}) catch unreachable;
        const pipeline_rope_batched = pipeline_mod.createFromSpirvWithOptions(instance, rope_batched_path, 3, @sizeOf(RopeBatchedPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("rope_batched shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // deinterleave: 1 input + 2 outputs = 3 bindings
        const deinterleave_path = std.fmt.bufPrint(&path_buf, "{s}/deinterleave.spv", .{shader_dir}) catch unreachable;
        const pipeline_deinterleave = pipeline_mod.createFromSpirvWithOptions(instance, deinterleave_path, 3, @sizeOf(DeinterleavePush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("deinterleave shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // deinterleave_batched: 1 input + 2 outputs = 3 bindings (token-batched
        // sibling of deinterleave, dispatched as (heads*head_dim/64, n_tokens, 1)).
        const deinterleave_batched_path = std.fmt.bufPrint(&path_buf, "{s}/deinterleave_batched.spv", .{shader_dir}) catch unreachable;
        const pipeline_deinterleave_batched = pipeline_mod.createFromSpirvWithOptions(instance, deinterleave_batched_path, 3, @sizeOf(DeinterleavePush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("deinterleave_batched shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // sigmoid_mul: 2 inputs + 1 output = 3 bindings
        const sigmoid_path = std.fmt.bufPrint(&path_buf, "{s}/sigmoid_mul.spv", .{shader_dir}) catch unreachable;
        const pipeline_sigmoid_mul = pipeline_mod.createFromSpirvWithOptions(instance, sigmoid_path, 3, @sizeOf(SigmoidMulPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("sigmoid_mul shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // vadd: 2 inputs + 1 output = 3 bindings
        const vadd_path = std.fmt.bufPrint(&path_buf, "{s}/vadd.spv", .{shader_dir}) catch unreachable;
        const pipeline_vadd = pipeline_mod.createFromSpirvWithOptions(instance, vadd_path, 3, @sizeOf(VaddPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("vadd shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // scale_accumulate: 1 read-write + 1 read = 2 bindings
        const sacc_path = std.fmt.bufPrint(&path_buf, "{s}/scale_accumulate.spv", .{shader_dir}) catch unreachable;
        const pipeline_scale_acc = pipeline_mod.createFromSpirvWithOptions(instance, sacc_path, 2, @sizeOf(ScaleAccPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("scale_accumulate shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // bias_add: out[i] += bias[src_offset + i], 2 bindings (output, bias)
        const bias_add_path = std.fmt.bufPrint(&path_buf, "{s}/bias_add.spv", .{shader_dir}) catch unreachable;
        const pipeline_bias_add = pipeline_mod.createFromSpirvWithOptions(instance, bias_add_path, 2, @sizeOf(BiasAddPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("bias_add shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // scale_in_place: 1 read-write binding (Gemma 4 per-layer output scaling)
        const sip_path = std.fmt.bufPrint(&path_buf, "{s}/scale_in_place.spv", .{shader_dir}) catch unreachable;
        const pipeline_scale_in_place = pipeline_mod.createFromSpirvWithOptions(instance, sip_path, 1, @sizeOf(ScaleAccPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("scale_in_place shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // mul_elementwise: 2 bindings (a *= b) — for ffn_gate_inp.scale
        const MulElemPush = extern struct { N: u32 };
        const mul_path = std.fmt.bufPrint(&path_buf, "{s}/mul_elementwise.spv", .{shader_dir}) catch unreachable;
        const pipeline_mul_elementwise = pipeline_mod.createFromSpirvWithOptions(instance, mul_path, 2, @sizeOf(MulElemPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("mul_elementwise shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // per_expert_scale: 3 bindings (down, scales, routing) — for ffn_down_exps.scale
        const PerExpertPush = extern struct { hidden_dim: u32, n_used: u32 };
        const pes_path = std.fmt.bufPrint(&path_buf, "{s}/per_expert_scale.spv", .{shader_dir}) catch unreachable;
        const pipeline_per_expert_scale = pipeline_mod.createFromSpirvWithOptions(instance, pes_path, 3, @sizeOf(PerExpertPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("per_expert_scale shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // SSM conv1d + SiLU: 4 bindings (input, kernel, state, output)
        const conv1d_path = std.fmt.bufPrint(&path_buf, "{s}/ssm_conv1d.spv", .{shader_dir}) catch unreachable;
        const pipeline_ssm_conv1d = pipeline_mod.createFromSpirvWithOptions(instance, conv1d_path, 4, @sizeOf(SsmConv1dPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("ssm_conv1d shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const conv1d_batched_path = std.fmt.bufPrint(&path_buf, "{s}/ssm_conv1d_batched.spv", .{shader_dir}) catch unreachable;
        const pipeline_ssm_conv1d_batched = pipeline_mod.createFromSpirvWithOptions(instance, conv1d_batched_path, 3, @sizeOf(SsmConv1dBatchedPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("ssm_conv1d_batched shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const f32_dual_batch_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_f32_dual_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_dmmv_f32_dual_batch = pipeline_mod.createFromSpirvWithOptions(instance, f32_dual_batch_path, 5, @sizeOf(F32DualBatchPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("dmmv_f32_dual_batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const qk_norm_path = std.fmt.bufPrint(&path_buf, "{s}/ssm_qk_norm.spv", .{shader_dir}) catch unreachable;
        const pipeline_ssm_qk_norm = pipeline_mod.createFromSpirvWithOptions(instance, qk_norm_path, 1, @sizeOf(SsmQkNormPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("ssm_qk_norm shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // SSM delta-net: 7 bindings (conv_out, dt_bias, alpha, beta, ssm_a, state, output)
        const delta_path = std.fmt.bufPrint(&path_buf, "{s}/ssm_delta_net.spv", .{shader_dir}) catch unreachable;
        const pipeline_ssm_delta_net = pipeline_mod.createFromSpirvWithOptions(instance, delta_path, 7, @sizeOf(SsmDeltaNetPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("ssm_delta_net shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const delta_cols8_path = std.fmt.bufPrint(&path_buf, "{s}/ssm_delta_net_cols8.spv", .{shader_dir}) catch unreachable;
        const pipeline_ssm_delta_net_cols8 = pipeline_mod.createFromSpirvWithOptions(instance, delta_cols8_path, 7, @sizeOf(SsmDeltaNetPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("ssm_delta_net_cols8 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const delta_cols8_normed_path = std.fmt.bufPrint(&path_buf, "{s}/ssm_delta_net_cols8_normed.spv", .{shader_dir}) catch unreachable;
        const pipeline_ssm_delta_net_cols8_normed = pipeline_mod.createFromSpirvWithOptions(instance, delta_cols8_normed_path, 7, @sizeOf(SsmDeltaNetPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("ssm_delta_net_cols8_normed shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // SSM gated norm: 4 bindings (delta_output, z_gate, norm_weights, output)
        const gnorm_path = std.fmt.bufPrint(&path_buf, "{s}/ssm_gated_norm.spv", .{shader_dir}) catch unreachable;
        const pipeline_ssm_gated_norm = pipeline_mod.createFromSpirvWithOptions(instance, gnorm_path, 4, @sizeOf(SsmGatedNormPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("ssm_gated_norm shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        // Effort-15 cycle 11: token-batched SSM gated norm. Same 4-binding
        // layout and push struct as ssm_gated_norm.spv; only the dispatch grid
        // differs (Y dim = n_tokens).
        const gnorm_batch_tok_path = std.fmt.bufPrint(&path_buf, "{s}/ssm_gated_norm_batch_tok.spv", .{shader_dir}) catch unreachable;
        const pipeline_ssm_gated_norm_batch_tok = pipeline_mod.createFromSpirvWithOptions(instance, gnorm_batch_tok_path, 4, @sizeOf(SsmGatedNormPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("ssm_gated_norm_batch_tok shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_ssm_gated_norm_batch_tok != null) {
            log.info("ssm_gated_norm_batch_tok pipeline loaded (Qwen dense-hybrid 27B SSM gated norm token-batched dispatch)", .{});
        }
        // Fused token-loop variant: one WG per head, all tokens processed
        // internally via n_tok push constant.
        const gnorm_batch_tok_fused_path = std.fmt.bufPrint(&path_buf, "{s}/ssm_gated_norm_batch_tok_fused.spv", .{shader_dir}) catch unreachable;
        const pipeline_ssm_gated_norm_batch_tok_fused = pipeline_mod.createFromSpirvWithOptions(instance, gnorm_batch_tok_fused_path, 4, @sizeOf(SsmGatedNormPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("ssm_gated_norm_batch_tok_fused shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Softmax + top-k: 2 bindings (logits, output)
        const topk_path = std.fmt.bufPrint(&path_buf, "{s}/softmax_topk.spv", .{shader_dir}) catch unreachable;
        const pipeline_softmax_topk = pipeline_mod.createFromSpirvWithOptions(instance, topk_path, 2, @sizeOf(SoftmaxTopkPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("softmax_topk shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        // Softmax + top-k v2: subgroup-parallel reduction (subgroupMax/Min/Shuffle).
        const topk_v2_path = std.fmt.bufPrint(&path_buf, "{s}/softmax_topk_v2.spv", .{shader_dir}) catch unreachable;
        const pipeline_softmax_topk_v2 = pipeline_mod.createFromSpirvWithOptions(instance, topk_v2_path, 2, @sizeOf(SoftmaxTopkPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("softmax_topk_v2 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const top1_path = std.fmt.bufPrint(&path_buf, "{s}/softmax_top1.spv", .{shader_dir}) catch unreachable;
        const pipeline_softmax_top1 = pipeline_mod.createFromSpirvWithOptions(instance, top1_path, 2, @sizeOf(SoftmaxTopkPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("softmax_top1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const top1_batch_path = std.fmt.bufPrint(&path_buf, "{s}/softmax_top1_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_softmax_top1_batch = pipeline_mod.createFromSpirvWithOptions(instance, top1_batch_path, 2, @sizeOf(SoftmaxTopkBatchPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("softmax_top1_batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const router_f32_batch_path = std.fmt.bufPrint(&path_buf, "{s}/router_f32_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_router_f32_batch = pipeline_mod.createFromSpirvWithOptions(instance, router_f32_batch_path, 3, @sizeOf(RouterF32BatchPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("router_f32_batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const rms_norm_scale_router_batch_path = std.fmt.bufPrint(&path_buf, "{s}/rms_norm_scale_dmmv_f32_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_rms_norm_scale_dmmv_f32_batch = pipeline_mod.createFromSpirvWithOptions(instance, rms_norm_scale_router_batch_path, 4, @sizeOf(RmsNormScaleDmmvF32BatchPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("rms_norm_scale_dmmv_f32_batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const topk_batch_path = std.fmt.bufPrint(&path_buf, "{s}/softmax_topk_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_softmax_topk_batch = pipeline_mod.createFromSpirvWithOptions(instance, topk_batch_path, 2, @sizeOf(SoftmaxTopkBatchPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("softmax_topk_batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // sigmoid_scale_acc: a[i] += sigmoid(c[0]) * b[i], 3 bindings (accum, src, gate)
        const ssa_path = std.fmt.bufPrint(&path_buf, "{s}/sigmoid_scale_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_sigmoid_scale_acc = pipeline_mod.createFromSpirvWithOptions(instance, ssa_path, 3, @sizeOf(ScaleAccPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("sigmoid_scale_acc shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // moe_weighted_acc: a[i] += routing_weight * b[i], 3 bindings (accum, src, routing)
        const mwa_path = std.fmt.bufPrint(&path_buf, "{s}/moe_weighted_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_moe_weighted_acc = pipeline_mod.createFromSpirvWithOptions(instance, mwa_path, 3, @sizeOf(MoeWeightedAccPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("moe_weighted_acc shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const mwa_batch_path = std.fmt.bufPrint(&path_buf, "{s}/moe_weighted_acc_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_moe_weighted_acc_batch = pipeline_mod.createFromSpirvWithOptions(instance, mwa_batch_path, 3, @sizeOf(MoeWeightedAccBatchPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("moe_weighted_acc_batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const mwa_scaled_batch_path = std.fmt.bufPrint(&path_buf, "{s}/moe_weighted_acc_scaled_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_moe_weighted_acc_scaled_batch = pipeline_mod.createFromSpirvWithOptions(instance, mwa_scaled_batch_path, 4, @sizeOf(MoeWeightedAccScaledBatchPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("moe_weighted_acc_scaled_batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const ssa_batch_path = std.fmt.bufPrint(&path_buf, "{s}/sigmoid_scale_acc_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_sigmoid_scale_acc_batch = pipeline_mod.createFromSpirvWithOptions(instance, ssa_batch_path, 3, @sizeOf(SigmoidScaleAccBatchPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("sigmoid_scale_acc_batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // kv_cache_write: 4 bindings (k_src, k_dst, v_src, v_dst)
        const kvcw_path = std.fmt.bufPrint(&path_buf, "{s}/kv_cache_write.spv", .{shader_dir}) catch unreachable;
        const pipeline_kv_cache_write = pipeline_mod.createFromSpirvWithOptions(instance, kvcw_path, 4, @sizeOf(KvCacheWritePush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("kv_cache_write shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // kv_cache_write_batched: 5 bindings (k_src, k_dst, v_src, v_dst, page_table).
        // Writes N tokens' K/V into their paged slots in one dispatch — replaces
        // the per-token vkCmdCopyBuffer loop that prefillBatched used to emit.
        const kvcwb_path = std.fmt.bufPrint(&path_buf, "{s}/kv_cache_write_batched.spv", .{shader_dir}) catch unreachable;
        const pipeline_kv_cache_write_batched = pipeline_mod.createFromSpirvWithOptions(instance, kvcwb_path, 5, @sizeOf(KvCacheWriteBatchedPush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("kv_cache_write_batched shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // residual_rms_norm: 4 bindings (hidden, residual, norm_out, weights).
        // Fuses scale_accumulate + rms_norm_mul so prefillBatched saves one
        // dispatch + one barrier per residual per layer.
        const resnorm_path = std.fmt.bufPrint(&path_buf, "{s}/residual_rms_norm.spv", .{shader_dir}) catch unreachable;
        const pipeline_residual_rms_norm = pipeline_mod.createFromSpirvWithOptions(instance, resnorm_path, 4, @sizeOf(ResidualRmsNormPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("residual_rms_norm shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // post_norm_residual_rms_norm: 5 bindings (hidden, residual,
        // post_norm_weights, norm_out, ffn_norm_weights). Fuses Gemma's
        // post_attention_norm + residual add + ffn_norm sequence.
        const post_norm_resnorm_path = std.fmt.bufPrint(&path_buf, "{s}/post_norm_residual_rms_norm.spv", .{shader_dir}) catch unreachable;
        const pipeline_post_norm_residual_rms_norm = pipeline_mod.createFromSpirvWithOptions(instance, post_norm_resnorm_path, 5, @sizeOf(PostNormResidualRmsNormPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("post_norm_residual_rms_norm shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // residual_rms_norm_quant_q8_1: 6 bindings (hidden, residual, norm_out,
        // weights, packed_i8, scale_dsum). Used by the Qwen3.6-27B dense FFN
        // prefill DP4a path to fuse the quantize_act_q8_1 dispatch into the
        // upstream SSM-out residual+RMS-norm. Workgroup size is 256 (not 64
        // like residual_rms_norm) to align cleanly with 32-block subgroup
        // clusters; the host enforces (hidden_dim % 256) == 0 before
        // dispatching.
        const resnorm_q8_1_path = std.fmt.bufPrint(&path_buf, "{s}/residual_rms_norm_quant_q8_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_residual_rms_norm_quant_q8_1 = pipeline_mod.createFromSpirvWithOptions(instance, resnorm_q8_1_path, 6, @sizeOf(ResidualRmsNormQuantQ8_1Push), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("residual_rms_norm_quant_q8_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        if (pipeline_residual_rms_norm_quant_q8_1 != null) {
            log.info("residual_rms_norm_quant_q8_1 pipeline loaded (Qwen dense-hybrid 27B FFN DP4a input fusion)", .{});
        }

        // norm_rope: fused RMS norm + RoPE, 3 bindings (data, weight, freq)
        const norm_rope_path = std.fmt.bufPrint(&path_buf, "{s}/norm_rope.spv", .{shader_dir}) catch unreachable;
        const pipeline_norm_rope = pipeline_mod.createFromSpirvWithOptions(instance, norm_rope_path, 3, @sizeOf(NormRopePush), &.{}, push_options, allocator) catch |err| blk: {
            log.warn("norm_rope shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // rms_norm_add: fused rmsnorm(src) + hidden accumulate, 3 bindings
        // (hidden, src, weights). Used by Gemma's post_ffw_norm + residual tail
        // to save one dispatch + one barrier per layer.
        const rms_norm_add_path = std.fmt.bufPrint(&path_buf, "{s}/rms_norm_add.spv", .{shader_dir}) catch unreachable;
        const pipeline_rms_norm_add = pipeline_mod.createFromSpirvWithOptions(instance, rms_norm_add_path, 3, @sizeOf(RmsNormAddPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("rms_norm_add shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };
        const rms_norm_add_vec4_path = std.fmt.bufPrint(&path_buf, "{s}/rms_norm_add_vec4.spv", .{shader_dir}) catch unreachable;
        const pipeline_rms_norm_add_vec4 = pipeline_mod.createFromSpirvWithOptions(instance, rms_norm_add_vec4_path, 3, @sizeOf(RmsNormAddPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("rms_norm_add_vec4 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // rms_norm_dmmv_f32: fused RMS norm + f32 router DMMV, 5 bindings
        // (hidden, ffn_norm_w, router_w, ffn_norm_buf, router_logits_buf).
        // Targets the per-MoE-layer (rms_norm_mul → router DMMV) pair on
        // architectures whose router weights are f32 (Qwen 3.5/3.6 etc).
        const rms_norm_dmmv_f32_path = std.fmt.bufPrint(&path_buf, "{s}/rms_norm_dmmv_f32.spv", .{shader_dir}) catch unreachable;
        const pipeline_rms_norm_dmmv_f32 = pipeline_mod.createFromSpirvWithOptions(instance, rms_norm_dmmv_f32_path, 5, @sizeOf(RmsNormDmmvF32Push), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("rms_norm_dmmv_f32 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // rms_norm_scale_dmmv_f32: Gemma MoE router fast path, 4 bindings
        // (hidden, ffn_gate_inp.scale, router_w, router_logits_buf).
        const rms_norm_scale_dmmv_f32_path = std.fmt.bufPrint(&path_buf, "{s}/rms_norm_scale_dmmv_f32.spv", .{shader_dir}) catch unreachable;
        const pipeline_rms_norm_scale_dmmv_f32 = pipeline_mod.createFromSpirvWithOptions(instance, rms_norm_scale_dmmv_f32_path, 4, @sizeOf(RmsNormDmmvF32Push), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("rms_norm_scale_dmmv_f32 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // rms_norm_dmmv_q4k_alpha_beta: fused RMS norm + Q4_K alpha/beta
        // SSM proj DMMV, 7 bindings (hidden, attn_norm_w, alpha_w, beta_w,
        // norm_buf, alpha_out, beta_out). Targets the per-SSM-layer
        // (rms_norm_mul → alpha DMMV → beta DMMV) trio on qwen35moe /
        // qwen36moe (alpha/beta have M=dt_rank, both Q4_K in Q4_K_M / XL).
        const rms_norm_dmmv_q4k_alpha_beta_path = std.fmt.bufPrint(&path_buf, "{s}/rms_norm_dmmv_q4k_alpha_beta.spv", .{shader_dir}) catch unreachable;
        const pipeline_rms_norm_dmmv_q4k_alpha_beta = pipeline_mod.createFromSpirvWithOptions(instance, rms_norm_dmmv_q4k_alpha_beta_path, 7, @sizeOf(RmsNormDmmvQ4kAlphaBetaPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("rms_norm_dmmv_q4k_alpha_beta shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // qk_norm_rope_kv_write: fused Q+K norm + RoPE + KV cache write,
        // 8 bindings (q_data, q_norm_w, k_src, k_norm_w, freq_buf,
        // kv_k_cache, v_src, kv_v_cache). Targets the per-attention-layer
        // (Q norm+rope → K norm+rope → kv_cache_write) trio on Qwen 3
        // family dense attention. Saves 2 dispatches + 1 barrier per layer.
        const qk_norm_rope_kv_write_path = std.fmt.bufPrint(&path_buf, "{s}/qk_norm_rope_kv_write.spv", .{shader_dir}) catch unreachable;
        const pipeline_qk_norm_rope_kv_write = pipeline_mod.createFromSpirvWithOptions(instance, qk_norm_rope_kv_write_path, 8, @sizeOf(QkNormRopeKvWritePush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("qk_norm_rope_kv_write shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const qk_norm_rope_kv_write_batched_path = std.fmt.bufPrint(&path_buf, "{s}/qk_norm_rope_kv_write_batched.spv", .{shader_dir}) catch unreachable;
        const pipeline_qk_norm_rope_kv_write_batched = pipeline_mod.createFromSpirvWithOptions(instance, qk_norm_rope_kv_write_batched_path, 8, @sizeOf(QkNormRopeKvWriteBatchedPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("qk_norm_rope_kv_write_batched shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const k_norm_rope_kv_write_batched_path = std.fmt.bufPrint(&path_buf, "{s}/k_norm_rope_kv_write_batched.spv", .{shader_dir}) catch unreachable;
        const pipeline_k_norm_rope_kv_write_batched = pipeline_mod.createFromSpirvWithOptions(instance, k_norm_rope_kv_write_batched_path, 7, @sizeOf(KNormRopeKvWriteBatchedPush), &.{}, push_wave64_options, allocator) catch |err| blk: {
            log.warn("k_norm_rope_kv_write_batched shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        return ElementwiseDispatch{
            .pipeline_rms_norm = pipeline_rms_norm,
            .pipeline_rms_norm_store_hidden = pipeline_rms_norm_store_hidden,
            .pipeline_swiglu = pipeline_swiglu,
            .pipeline_swiglu_oai = pipeline_swiglu_oai,
            .pipeline_geglu = pipeline_geglu,
            .pipeline_rope = pipeline_rope,
            .pipeline_rope_batched = pipeline_rope_batched,
            .pipeline_deinterleave = pipeline_deinterleave,
            .pipeline_deinterleave_batched = pipeline_deinterleave_batched,
            .pipeline_sigmoid_mul = pipeline_sigmoid_mul,
            .pipeline_vadd = pipeline_vadd,
            .pipeline_scale_acc = pipeline_scale_acc,
            .pipeline_bias_add = pipeline_bias_add,
            .pipeline_scale_in_place = pipeline_scale_in_place,
            .pipeline_mul_elementwise = pipeline_mul_elementwise,
            .pipeline_per_expert_scale = pipeline_per_expert_scale,
            .pipeline_ssm_conv1d = pipeline_ssm_conv1d,
            .pipeline_ssm_conv1d_batched = pipeline_ssm_conv1d_batched,
            .pipeline_dmmv_f32_dual_batch = pipeline_dmmv_f32_dual_batch,
            .pipeline_ssm_qk_norm = pipeline_ssm_qk_norm,
            .pipeline_ssm_delta_net = pipeline_ssm_delta_net,
            .pipeline_ssm_delta_net_cols8 = pipeline_ssm_delta_net_cols8,
            .pipeline_ssm_delta_net_cols8_normed = pipeline_ssm_delta_net_cols8_normed,
            .pipeline_ssm_gated_norm = pipeline_ssm_gated_norm,
            .pipeline_ssm_gated_norm_batch_tok = pipeline_ssm_gated_norm_batch_tok,
            .pipeline_ssm_gated_norm_batch_tok_fused = pipeline_ssm_gated_norm_batch_tok_fused,
            .pipeline_softmax_topk = pipeline_softmax_topk,
            .pipeline_softmax_topk_v2 = pipeline_softmax_topk_v2,
            .pipeline_softmax_top1 = pipeline_softmax_top1,
            .pipeline_softmax_top1_batch = pipeline_softmax_top1_batch,
            .pipeline_router_f32_batch = pipeline_router_f32_batch,
            .pipeline_rms_norm_scale_dmmv_f32_batch = pipeline_rms_norm_scale_dmmv_f32_batch,
            .pipeline_softmax_topk_batch = pipeline_softmax_topk_batch,
            .pipeline_sigmoid_scale_acc = pipeline_sigmoid_scale_acc,
            .pipeline_moe_weighted_acc = pipeline_moe_weighted_acc,
            .pipeline_moe_weighted_acc_batch = pipeline_moe_weighted_acc_batch,
            .pipeline_moe_weighted_acc_scaled_batch = pipeline_moe_weighted_acc_scaled_batch,
            .pipeline_sigmoid_scale_acc_batch = pipeline_sigmoid_scale_acc_batch,
            .pipeline_kv_cache_write = pipeline_kv_cache_write,
            .pipeline_kv_cache_write_batched = pipeline_kv_cache_write_batched,
            .pipeline_residual_rms_norm = pipeline_residual_rms_norm,
            .pipeline_post_norm_residual_rms_norm = pipeline_post_norm_residual_rms_norm,
            .pipeline_residual_rms_norm_quant_q8_1 = pipeline_residual_rms_norm_quant_q8_1,
            .pipeline_rms_norm_add = pipeline_rms_norm_add,
            .pipeline_rms_norm_add_vec4 = pipeline_rms_norm_add_vec4,
            .pipeline_norm_rope = pipeline_norm_rope,
            .pipeline_rms_norm_dmmv_f32 = pipeline_rms_norm_dmmv_f32,
            .pipeline_rms_norm_scale_dmmv_f32 = pipeline_rms_norm_scale_dmmv_f32,
            .pipeline_rms_norm_dmmv_q4k_alpha_beta = pipeline_rms_norm_dmmv_q4k_alpha_beta,
            .pipeline_qk_norm_rope_kv_write = pipeline_qk_norm_rope_kv_write,
            .pipeline_qk_norm_rope_kv_write_batched = pipeline_qk_norm_rope_kv_write_batched,
            .pipeline_k_norm_rope_kv_write_batched = pipeline_k_norm_rope_kv_write_batched,
            .descriptor_pool = descriptor_pool,
            .device = instance.device,
        };
    }

    /// Record an RMS-norm-plus-scale dispatch for a batch of tokens.
    ///
    /// This binds the fused normalization shader used before attention and MLP
    /// projections so each token is normalized against its hidden dimension.
    /// @param self Dispatch wrapper containing the RMS norm pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set containing input, weight, and output buffers.
    /// @param hidden_dim Hidden width processed per token.
    /// @param n_tokens Number of tokens covered by the dispatch.
    /// @param eps Numerical stability epsilon passed to the shader.
    /// @returns `error.ShaderNotLoaded` when the RMS norm pipeline is unavailable.
    /// @note The helper dispatches one workgroup per token.
    pub fn recordRmsNorm(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        /// Hidden state width.
        hidden_dim: u32,
        n_tokens: u32,
        eps: f32,
    ) !void {
        const pip = if (self.pipeline_rms_norm) |*p| p else return error.ShaderNotLoaded;
        const push = RmsNormPush{
            .N = hidden_dim,
            .eps_bits = @bitCast(eps),
        };
        // One workgroup per token
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), n_tokens, 1, 1);
    }

    /// Record a SwiGLU activation dispatch.
    /// @param self Dispatch wrapper containing the SwiGLU pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set containing gate, up, and output buffers.
    /// @param n_elements Total number of output elements to compute.
    /// @returns `error.ShaderNotLoaded` when the SwiGLU pipeline is unavailable.
    /// @note Workgroups are sized as `ceil(n_elements / 64)`.
    pub fn recordSwiglu(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
    ) !void {
        const pip = if (self.pipeline_swiglu) |*p| p else return error.ShaderNotLoaded;
        const push = SwigluPush{ .N = n_elements };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record a GPT-OSS / OAI-variant SwiGLU activation dispatch.
    /// Uses the same 3-binding layout as `recordSwiglu` (gate, up → output) but
    /// selects the swiglu_oai shader whose activation function matches gpt-oss.
    /// @param self Dispatch wrapper containing the OAI SwiGLU pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set containing gate, up, and output buffers.
    /// @param n_elements Total number of output elements to compute.
    /// @returns `error.ShaderNotLoaded` when the OAI SwiGLU pipeline is unavailable.
    /// @note Workgroups are sized as `ceil(n_elements / 64)`.
    pub fn recordSwigluOai(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
    ) !void {
        const pip = if (self.pipeline_swiglu_oai) |*p| p else return error.ShaderNotLoaded;
        const push = SwigluPush{ .N = n_elements };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record an in-place bias add dispatch: `out[i] += bias[src_offset + i]`.
    /// @param self Dispatch wrapper containing the bias add pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set with two bindings: output buffer (rw) and bias buffer (ro).
    /// @param n_elements Number of elements to update.
    /// @param src_offset Element offset into the bias buffer (allows a shared bias tensor to be sliced).
    /// @returns `error.ShaderNotLoaded` when the bias add pipeline is unavailable.
    pub fn recordBiasAdd(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
        src_offset: u32,
    ) !void {
        const pip = if (self.pipeline_bias_add) |*p| p else return error.ShaderNotLoaded;
        const push = BiasAddPush{
            .N = n_elements,
            .src_offset = src_offset,
        };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record a GEGLU activation dispatch (GELU-gated, used by Gemma).
    /// Same buffer layout as SwiGLU: gate, up → output.
    pub fn recordGeglu(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
    ) !void {
        const pip = if (self.pipeline_geglu) |*p| p else return error.ShaderNotLoaded;
        const push = SwigluPush{ .N = n_elements };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record a RoPE dispatch with partial rotation support (IMRoPE).
    /// Rotates the first `rope_dim` dimensions of each attention head at the
    /// given sequence position; the remaining `stride - rope_dim` dimensions
    /// are copied unchanged, enabling interleaved-masked (IMRoPE) layouts.
    /// @param self Dispatch wrapper containing the RoPE pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set with three bindings: input, output, and freq buffer.
    /// @param stride Full head dimension in f32 elements (distance between heads in the buffer).
    /// @param rope_dim Number of dimensions to rotate (must be <= stride; pass stride for plain RoPE).
    /// @param n_heads Number of query heads to rotate; one workgroup is dispatched per head.
    /// @param position Current decode token position used to compute rotation angles.
    /// @param freq_base Base frequency for the sinusoidal schedule (e.g. 10000.0 for standard RoPE).
    /// @param attn_scale YaRN magnitude scale applied after rotation; use 1.0 for plain RoPE.
    /// @returns `error.ShaderNotLoaded` when the RoPE pipeline is unavailable.
    pub fn recordRope(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        stride: u32,
        /// RoPE dimensions (0 = all).
        rope_dim: u32,
        /// Number of query heads.
        n_heads: u32,
        /// Current token position.
        position: u32,
        freq_base: f32,
        attn_scale: f32,
    ) !void {
        const pip = if (self.pipeline_rope) |*p| p else return error.ShaderNotLoaded;
        const push = RopePush{
            .stride = stride,
            .rope_dim = rope_dim,
            .n_heads = n_heads,
            .position = position,
            .freq_base_bits = @bitCast(freq_base),
            .attn_scale_bits = @bitCast(attn_scale),
        };
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), n_heads, 1, 1);
    }

    /// Record a batched RoPE dispatch that rotates N tokens at consecutive
    /// positions [position_base, position_base + n_tokens) in a single call.
    /// Grid is (n_heads, n_tokens, 1); each (head, token) workgroup rotates
    /// `rope_dim` elements of the token's head slice.
    pub fn recordRoPEBatched(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        stride: u32,
        rope_dim: u32,
        n_heads: u32,
        position_base: u32,
        n_tokens: u32,
        freq_base: f32,
        attn_scale: f32,
    ) !void {
        const pip = if (self.pipeline_rope_batched) |*p| p else return error.ShaderNotLoaded;
        const push = RopeBatchedPush{
            .stride = stride,
            .rope_dim = rope_dim,
            .n_heads = n_heads,
            .position_base = position_base,
            .freq_base_bits = @bitCast(freq_base),
            .attn_scale_bits = @bitCast(attn_scale),
        };
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), n_heads, n_tokens, 1);
    }

    /// Record a deinterleave dispatch: split element-interleaved Q+gate into separate buffers.
    pub fn recordDeinterleave(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        /// Per-head dimension.
        head_dim: u32,
        /// Number of query heads.
        n_heads: u32,
    ) !void {
        const pip = if (self.pipeline_deinterleave) |*p| p else return error.ShaderNotLoaded;
        const push = DeinterleavePush{ .head_dim = head_dim, .n_heads = n_heads };
        const total = head_dim * n_heads;
        const workgroups = (total + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record a token-batched deinterleave dispatch.
    /// Splits each token's packed `[Q(head_dim), gate(head_dim)]` interleaved
    /// per-head layout into separate Q and gate output buffers in one dispatch.
    /// Grid is `(ceil(head_dim * n_heads / 64), n_tokens, 1)`.
    /// @param self Dispatch wrapper containing the batched deinterleave pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set with 3 bindings: packed input, Q output, gate output.
    /// @param head_dim Per-head dimension in elements.
    /// @param n_heads Number of query heads per token.
    /// @param n_tokens Number of tokens to process (Y dimension of the dispatch grid).
    /// @returns `error.ShaderNotLoaded` when the batched deinterleave pipeline is unavailable.
    pub fn recordDeinterleaveBatched(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        head_dim: u32,
        n_heads: u32,
        n_tokens: u32,
    ) !void {
        const pip = if (self.pipeline_deinterleave_batched) |*p| p else return error.ShaderNotLoaded;
        const push = DeinterleavePush{ .head_dim = head_dim, .n_heads = n_heads };
        const total = head_dim * n_heads;
        const workgroups_x = (total + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups_x, n_tokens, 1);
    }

    /// Record a sigmoid multiply dispatch: out = input * sigmoid(gate).
    pub fn recordSigmoidMul(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
    ) !void {
        const pip = if (self.pipeline_sigmoid_mul) |*p| p else return error.ShaderNotLoaded;
        const push = SigmoidMulPush{ .N = n_elements };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record a vector add dispatch: c = a + b.
    pub fn recordVadd(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
    ) !void {
        const pip = if (self.pipeline_vadd) |*p| p else return error.ShaderNotLoaded;
        const push = VaddPush{ .N = n_elements };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record a scale-accumulate dispatch: a[i] += scale * b[i].
    /// Vec4-coalesced: each thread handles one vec4 (4 f32 elements). Caller
    /// must pass n_elements divisible by 4; every in-tree caller already does.
    pub fn recordScaleAcc(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
        scale: f32,
    ) !void {
        const pip = if (self.pipeline_scale_acc) |*p| p else return error.ShaderNotLoaded;
        const push = ScaleAccPush{ .N = n_elements, .scale_bits = @bitCast(scale) };
        const workgroups = (n_elements + 255) / 256;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record an in-place element-wise scale dispatch: `data[i] *= scale`.
    /// @param self Dispatch wrapper containing the scale-in-place pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set with one binding: the buffer to scale in place.
    /// @param n_elements Number of f32 elements to scale.
    /// @param scale Scalar multiplier applied to every element.
    /// @returns `error.ShaderNotLoaded` when the scale-in-place pipeline is unavailable.
    pub fn recordScaleInPlace(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
        scale: f32,
    ) !void {
        const pip = if (self.pipeline_scale_in_place) |*p| p else return error.ShaderNotLoaded;
        const push = ScaleAccPush{ .N = n_elements, .scale_bits = @bitCast(scale) };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record a single-token SSM depthwise conv1d + SiLU dispatch.
    /// Reads the current SSM conv state via `state_offset` (a rotating index into
    /// the circular state buffer), applies a depthwise conv kernel of width
    /// `d_conv`, and writes the SiLU-activated output in-place.
    /// @param self Dispatch wrapper containing the SSM conv1d pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set with four bindings: input, kernel, state, output.
    /// @param conv_channels Number of SSM channels (width of the depthwise conv).
    /// @param d_conv Kernel width of the depthwise convolution.
    /// @param kernel_is_f16 True when the kernel weight buffer is f16; false for f32.
    /// @param state_offset Current rotation index (0..d_conv-2) into the circular state buffer.
    /// @returns `error.ShaderNotLoaded` when the SSM conv1d pipeline is unavailable.
    pub fn recordSsmConv1d(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        conv_channels: u32,
        d_conv: u32,
        kernel_is_f16: bool,
        state_offset: u32,
    ) !void {
        const pip = if (self.pipeline_ssm_conv1d) |*p| p else return error.ShaderNotLoaded;
        const push = SsmConv1dPush{
            .conv_channels = conv_channels,
            .d_conv = d_conv,
            .kernel_is_f16 = if (kernel_is_f16) 1 else 0,
            .state_offset = state_offset,
        };
        const workgroups = (conv_channels + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record an in-place SSM Q/K RMS-norm dispatch.
    /// Applies per-group RMS normalization to the concatenated Q and K projections
    /// inside a Mamba/DeltaNet SSM block; dispatches one workgroup per group.
    /// @param self Dispatch wrapper containing the SSM Q/K norm pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set with one binding: the Q+K buffer (in-place).
    /// @param d_state Per-group state dimension (qk_dim = d_state * n_group).
    /// @param n_group Number of normalization groups; one workgroup per group.
    /// @returns `error.ShaderNotLoaded` when the SSM Q/K norm pipeline is unavailable.
    pub fn recordSsmQkNorm(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        d_state: u32,
        n_group: u32,
    ) !void {
        const pip = if (self.pipeline_ssm_qk_norm) |*p| p else return error.ShaderNotLoaded;
        const push = SsmQkNormPush{
            .d_state = d_state,
            .n_group = n_group,
            .qk_dim = d_state * n_group,
        };
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), n_group, 1, 1);
    }

    /// Record an SSM DeltaNet state-update dispatch (baseline variant).
    /// Executes the DeltaNet recurrence over a single token (or `push.n_tok`
    /// prefill tokens when n_tok > 1).  Grid is `(dt_rank, head_v_dim, 1)` —
    /// one wave64 workgroup per (head, row) pair.
    /// @param self Dispatch wrapper containing the SSM delta-net pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set with 7 bindings: conv_out, dt_bias, alpha, beta, ssm_a, state, output.
    /// @param push Fully populated push-constant struct describing the SSM dimensions and flags.
    /// @returns `error.ShaderNotLoaded` when the SSM delta-net pipeline is unavailable.
    pub fn recordSsmDeltaNet(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        push: SsmDeltaNetPush,
    ) !void {
        const pip = if (self.pipeline_ssm_delta_net) |*p| p else return error.ShaderNotLoaded;
        // 64t×1r: one WG per (head, row) pair — see ssm_delta_net.comp
        const row_blocks = push.head_v_dim;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), push.dt_rank, row_blocks, 1);
    }

    /// Record an SSM DeltaNet state-update dispatch using the cols8 tiled variant.
    /// Each wave64 workgroup processes four output rows (head_v_dim / 4 workgroups
    /// per head), improving register reuse relative to the baseline 1-row shader.
    /// @param self Dispatch wrapper containing the SSM delta-net cols8 pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set with 7 bindings (same layout as `recordSsmDeltaNet`).
    /// @param push Fully populated push-constant struct describing the SSM dimensions and flags.
    /// @returns `error.ShaderNotLoaded` when the SSM delta-net cols8 pipeline is unavailable.
    pub fn recordSsmDeltaNetCols8(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        push: SsmDeltaNetPush,
    ) !void {
        const pip = if (self.pipeline_ssm_delta_net_cols8) |*p| p else return error.ShaderNotLoaded;
        // ssm_delta_net_cols8 currently maps four output rows per wave64.
        const row_blocks = (push.head_v_dim + 3) / 4;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), push.dt_rank, row_blocks, 1);
    }

    /// Record an SSM DeltaNet state-update dispatch using the cols8 normed variant.
    /// Identical semantics to `recordSsmDeltaNetCols8` but selects the shader
    /// that expects Q/K inputs to be pre-normalized (skipping the in-shader norm).
    /// Each wave64 workgroup processes eight output rows (head_v_dim / 8 workgroups per head).
    /// @param self Dispatch wrapper containing the SSM delta-net cols8 normed pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set with 7 bindings (same layout as `recordSsmDeltaNet`).
    /// @param push Fully populated push-constant struct describing the SSM dimensions and flags.
    /// @returns `error.ShaderNotLoaded` when the SSM delta-net cols8 normed pipeline is unavailable.
    pub fn recordSsmDeltaNetCols8Normed(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        push: SsmDeltaNetPush,
    ) !void {
        const pip = if (self.pipeline_ssm_delta_net_cols8_normed) |*p| p else return error.ShaderNotLoaded;
        const row_blocks = (push.head_v_dim + 7) / 8;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), push.dt_rank, row_blocks, 1);
    }

    /// Record an SSM gated norm dispatch: applies z-gate * RMS-norm(delta_output).
    /// Dispatches one wave64 workgroup per head (`push.dt_rank` workgroups total).
    /// @param self Dispatch wrapper containing the SSM gated norm pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set with 4 bindings: delta_output, z_gate, norm_weights, output.
    /// @param push Push-constant struct specifying d_inner, dt_rank, head_v_dim, d_state, and norm_per_head.
    /// @returns `error.ShaderNotLoaded` when the SSM gated norm pipeline is unavailable.
    pub fn recordSsmGatedNorm(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        push: SsmGatedNormPush,
    ) !void {
        const pip = if (self.pipeline_ssm_gated_norm) |*p| p else return error.ShaderNotLoaded;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), push.dt_rank, 1, 1);
    }

    /// Record softmax + top-k MoE router dispatch.
    pub fn recordSoftmaxTopk(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        n_experts: u32,
        k: u32,
    ) !void {
        const pip = if (self.pipeline_softmax_topk) |*p| p else return error.ShaderNotLoaded;
        const push = SoftmaxTopkPush{ .n_experts = n_experts, .k = k };
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), 1, 1, 1);
    }

    /// Record sigmoid-gated scale-accumulate: a[i] += sigmoid(c[0]) * b[i].
    pub fn recordSigmoidScaleAcc(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
    ) !void {
        const pip = if (self.pipeline_sigmoid_scale_acc) |*p| p else return error.ShaderNotLoaded;
        // Push constant only needs N (uses same layout as ScaleAccPush but only N is read)
        const push = ScaleAccPush{ .N = n_elements, .scale_bits = 0 };
        const workgroups = (n_elements + 255) / 256;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record a MoE weighted accumulate dispatch: `a[i] += routing_weight[j] * b[j*src_stride + i]`
    /// summed over `n_used` selected experts.  Routing weights are read from the GPU
    /// routing buffer (binding 2), not from a push constant.
    /// @param self Dispatch wrapper containing the MoE weighted accumulate pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set with 3 bindings: accum (rw), src experts, routing weights.
    /// @param n_elements Hidden dimension of the accumulation buffer (elements updated per token).
    /// @param n_used Number of selected experts whose outputs are summed.
    /// @param src_stride Elements per expert in the source buffer (typically equal to n_elements).
    /// @returns `error.ShaderNotLoaded` when the MoE weighted accumulate pipeline is unavailable.
    pub fn recordMoeWeightedAcc(
        self: *const ElementwiseDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
        n_used: u32,
        src_stride: u32,
    ) !void {
        const pip = if (self.pipeline_moe_weighted_acc) |*p| p else return error.ShaderNotLoaded;
        const push = MoeWeightedAccPush{ .N = n_elements, .n_used = n_used, .src_stride = src_stride };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Destroy the loaded pipelines and descriptor pool.
    /// @param self Dispatch wrapper to tear down in place.
    pub fn deinit(self: *ElementwiseDispatch) void {
        if (self.pipeline_rms_norm) |*p| p.deinit();
        if (self.pipeline_rms_norm_store_hidden) |*p| p.deinit();
        if (self.pipeline_swiglu) |*p| p.deinit();
        if (self.pipeline_swiglu_oai) |*p| p.deinit();
        if (self.pipeline_geglu) |*p| p.deinit();
        if (self.pipeline_rope) |*p| p.deinit();
        if (self.pipeline_rope_batched) |*p| p.deinit();
        if (self.pipeline_deinterleave) |*p| p.deinit();
        if (self.pipeline_deinterleave_batched) |*p| p.deinit();
        if (self.pipeline_sigmoid_mul) |*p| p.deinit();
        if (self.pipeline_vadd) |*p| p.deinit();
        if (self.pipeline_scale_acc) |*p| p.deinit();
        if (self.pipeline_bias_add) |*p| p.deinit();
        if (self.pipeline_scale_in_place) |*p| p.deinit();
        if (self.pipeline_mul_elementwise) |*p| p.deinit();
        if (self.pipeline_per_expert_scale) |*p| p.deinit();
        if (self.pipeline_ssm_conv1d) |*p| p.deinit();
        if (self.pipeline_ssm_conv1d_batched) |*p| p.deinit();
        if (self.pipeline_dmmv_f32_dual_batch) |*p| p.deinit();
        if (self.pipeline_ssm_qk_norm) |*p| p.deinit();
        if (self.pipeline_ssm_delta_net) |*p| p.deinit();
        if (self.pipeline_ssm_delta_net_cols8) |*p| p.deinit();
        if (self.pipeline_ssm_delta_net_cols8_normed) |*p| p.deinit();
        if (self.pipeline_ssm_gated_norm) |*p| p.deinit();
        if (self.pipeline_ssm_gated_norm_batch_tok) |*p| p.deinit();
        if (self.pipeline_ssm_gated_norm_batch_tok_fused) |*p| p.deinit();
        if (self.pipeline_softmax_topk) |*p| p.deinit();
        if (self.pipeline_softmax_topk_v2) |*p| p.deinit();
        if (self.pipeline_softmax_top1) |*p| p.deinit();
        if (self.pipeline_softmax_top1_batch) |*p| p.deinit();
        if (self.pipeline_router_f32_batch) |*p| p.deinit();
        if (self.pipeline_rms_norm_scale_dmmv_f32_batch) |*p| p.deinit();
        if (self.pipeline_softmax_topk_batch) |*p| p.deinit();
        if (self.pipeline_sigmoid_scale_acc) |*p| p.deinit();
        if (self.pipeline_moe_weighted_acc) |*p| p.deinit();
        if (self.pipeline_moe_weighted_acc_batch) |*p| p.deinit();
        if (self.pipeline_moe_weighted_acc_scaled_batch) |*p| p.deinit();
        if (self.pipeline_sigmoid_scale_acc_batch) |*p| p.deinit();
        if (self.pipeline_kv_cache_write) |*p| p.deinit();
        if (self.pipeline_kv_cache_write_batched) |*p| p.deinit();
        if (self.pipeline_residual_rms_norm) |*p| p.deinit();
        if (self.pipeline_post_norm_residual_rms_norm) |*p| p.deinit();
        if (self.pipeline_residual_rms_norm_quant_q8_1) |*p| p.deinit();
        if (self.pipeline_rms_norm_add) |*p| p.deinit();
        if (self.pipeline_rms_norm_add_vec4) |*p| p.deinit();
        if (self.pipeline_norm_rope) |*p| p.deinit();
        if (self.pipeline_rms_norm_dmmv_f32) |*p| p.deinit();
        if (self.pipeline_rms_norm_scale_dmmv_f32) |*p| p.deinit();
        if (self.pipeline_rms_norm_dmmv_q4k_alpha_beta) |*p| p.deinit();
        if (self.pipeline_qk_norm_rope_kv_write) |*p| p.deinit();
        if (self.pipeline_qk_norm_rope_kv_write_batched) |*p| p.deinit();
        if (self.pipeline_k_norm_rope_kv_write_batched) |*p| p.deinit();
        vk.c.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.* = undefined;
    }
};
