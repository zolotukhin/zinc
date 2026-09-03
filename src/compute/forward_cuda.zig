//! CUDA forward pass for the dense `qwen35` hybrid-SSM model (Qwen 3.5 9B).
//!
//! M2 bring-up — incremental greedy decode of a single token on NVIDIA. The CUDA
//! backend modules (device/buffer/pipeline/command) and the kernel library
//! (`src/shaders/cuda/kernels.cu`, NVRTC-compiled) are orchestrated here into a
//! real forward pass. Weights upload VERBATIM-quantized (Q4_K/Q5_K/Q6_K/Q8_0
//! blocks) via loader_cuda and are dequantized inside the DMMV kernels.
//!
//! Layer schedule (qwen35, full_attention_interval=4): layer L is full attention
//! when ((L+1) % 4 == 0) → L in {3,7,11,15,19,23,27,31}; the other 24 layers are
//! gated-delta-net SSM. Every layer is followed by a SwiGLU FFN block.
//!
//! @section Inference Runtime
const std = @import("std");
const build_options = @import("build_options");
const buffer = @import("../cuda/buffer.zig");
const pipeline = @import("../cuda/pipeline.zig");
const command = @import("../cuda/command.zig");
const shim = @import("../cuda/c.zig").shim;
const gguf = @import("../model/gguf.zig");
const loader = @import("../model/loader_cuda.zig");

const log = std.log.scoped(.cuda_fwd);
const CudaBuffer = buffer.CudaBuffer;
const CudaPipeline = pipeline.CudaPipeline;
const LoadedTensor = loader.LoadedTensor;
const is_rocm = std.mem.eql(u8, build_options.backend, "rocm");
// Kept as one constant because all decode call sites must use the same launch
// geometry; the dequantizing matvec is measured fastest with two wave32 warps.
const dmmv_fast_block: u32 = 64;

/// The CUDA kernel library, bundled into the binary and NVRTC-compiled on load.
const KERNELS_CU = @embedFile("../shaders/cuda/kernels.cu");

// ---- kernel push-constant structs (must byte-match kernels.cu) --------------
const RmsPush = extern struct { N: u32, eps: f32 };
const RmsQ8Push = extern struct { N: u32, eps: f32, T: u32 };
const DmmvPush = extern struct {
    M: u32,
    K: u32,
    a_offset: u32 = 0,
    x_offset: u32 = 0,
    y_offset: u32 = 0,
    acc_mode: u32 = 0,
};
const DmmvPairPush = extern struct { M0: u32, M1: u32, K: u32, pair_reduce: u32 = 0 };
const RopePush = extern struct {
    stride: u32,
    rope_dim: u32,
    n_heads: u32,
    position: u32,
    freq_base_bits: u32,
    attn_scale_bits: u32,
};
const AttnPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_len: u32,
    attn_scale_bits: u32,
    sink_offset: u32,
};
const KvWritePush = extern struct { kv_dim: u32, dst_offset: u32 };
const ConvPush = extern struct {
    conv_channels: u32,
    d_conv: u32,
    kernel_is_f16: u32,
    state_offset: u32,
};
const ConvPreparePush = extern struct {
    conv_channels: u32,
    d_conv: u32,
    kernel_is_f16: u32,
    state_offset: u32,
    dt_rank: u32,
    d_inner: u32,
    d_state: u32,
    n_group: u32,
    ssm_a_is_f16: u32,
    dt_bias_is_f16: u32,
};
const DeltaNetPush = extern struct {
    d_inner: u32,
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    n_group: u32,
    ssm_a_is_f16: u32,
    dt_bias_is_f16: u32,
    has_dt_bias: u32,
    has_ssm_a: u32,
    n_tok: u32,
    conv_stride_tok: u32,
    ab_stride_tok: u32,
    y_stride_tok: u32,
    preprocessed: u32,
};

const DeltaNetWarpPush = extern struct {
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    n_group: u32,
    ssm_a_is_f16: u32,
    dt_bias_is_f16: u32,
    has_dt_bias: u32,
    has_ssm_a: u32,
    n_tok: u32,
    conv_stride_tok: u32,
    ab_stride_tok: u32,
    y_stride_tok: u32,
};
const DeltaNetPreparePush = extern struct {
    dt_rank: u32,
    d_state: u32,
    n_group: u32,
    ssm_a_is_f16: u32,
    dt_bias_is_f16: u32,
    has_dt_bias: u32,
    has_ssm_a: u32,
    n_tok: u32,
    conv_stride_tok: u32,
    ab_stride_tok: u32,
};
const DeltaNetColWarpPush = extern struct {
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    n_group: u32,
    n_tok: u32,
    conv_stride_tok: u32,
    ab_stride_tok: u32,
    y_stride_tok: u32,
    fast_reduce: u32,
};
const DeltaNetChunkedPush = extern struct {
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    n_group: u32,
    ssm_a_is_f16: u32,
    dt_bias_is_f16: u32,
    has_dt_bias: u32,
    has_ssm_a: u32,
    n_tok: u32,
    conv_stride_tok: u32,
    ab_stride_tok: u32,
    y_stride_tok: u32,
};
const GatedNormPush = extern struct {
    d_inner: u32,
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    norm_per_head: u32,
};
// Effort 28 4c-2b: batched-decode SSM-inner kernels (byte-match kernels.cu).
const ConvSeqPush = extern struct {
    conv_channels: u32,
    d_conv: u32,
    kernel_is_f16: u32,
    conv_state_len: u32,
};
const GatedNormSeqPush = extern struct {
    d_inner: u32,
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    norm_per_head: u32,
};
const DeltaNetSeqPush = extern struct {
    d_inner: u32,
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    n_group: u32,
    ssm_a_is_f16: u32,
    dt_bias_is_f16: u32,
    has_dt_bias: u32,
    has_ssm_a: u32,
    conv_stride_tok: u32,
    ab_stride_tok: u32,
    y_stride_tok: u32,
    ssm_state_len: u32,
};
const SwigluPush = extern struct { N: u32 };
const SigmoidMulPush = extern struct { N: u32 };
const DeintPush = extern struct { head_dim: u32, n_head: u32 };
const QwenQkvPush = extern struct {
    head_dim: u32,
    eps: f32,
    rope_dim: u32,
    n_head: u32,
    n_kv_head: u32,
    position: u32,
    kv_offset: u32,
};
// Effort 28 4c-2: fused batched-decode attn front-end (byte-match kernels.cu).
const QwenQkvSeqPush = extern struct {
    head_dim: u32,
    eps: f32,
    rope_dim: u32,
    n_head: u32,
    n_kv_head: u32,
    slot_ctx: u32,
};
const AttnSlotPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    slot_ctx: u32,
    attn_scale_bits: u32,
    sink_offset: u32,
};
// Effort 28 4c: batched GEMM push (must byte-match `struct GemmPush` in kernels.cu).
// Output Y is token-major [T, M]; offsets are in BYTES; acc_mode 1 => Y += .
const GemmPush = extern struct {
    M: u32,
    K: u32,
    T: u32,
    a_offset: u32 = 0,
    x_offset: u32 = 0,
    y_offset: u32 = 0,
    acc_mode: u32 = 0,
    q8_stride: u32 = 0,
};
const ArgmaxPush = extern struct { N: u32 };
const ArgmaxV2Push = extern struct { N: u32, partials: u32 };
const EmbedPush = extern struct { K: u32, vocab: u32 };
// MoE router/combine kernels (byte-match kernels.cu).
const TopkPush = extern struct { n_experts: u32, k: u32 };
const MoeAccPush = extern struct { N: u32, n_used: u32, src_stride: u32 };
const SigmoidAccPush = extern struct { N: u32 };
// Batched MoE expert matvec: one launch over all n_used experts, GPU-side ids.
const ExpertsPush = extern struct { M: u32, K: u32, slice: u32, x_stride: u32, n_used: u32, base: u32 = 0, fuse_activation: u32 = 0 };
// Effort 29 T2: token-batched (grid.y = T) MoE prefill twins — process ALL T
// prompt tokens' routed/shared experts in one launch each (vs the per-token loop).
const ExpertsBatchPush = extern struct { M: u32, K: u32, slice: u32, x_stride: u32, n_used: u32, base: u32, routing_stride: u32, x_tok_stride: u32, y_tok_stride: u32 };
const MoeAccBatchPush = extern struct { N: u32, n_used: u32, src_stride: u32, a_tok_stride: u32, b_tok_stride: u32, routing_stride: u32 };
const MatvecBatchPush = extern struct { M: u32, K: u32, x_tok_stride: u32, y_tok_stride: u32 };
// T2 (qwen MoE port): grouped Tensor-core routed-expert gate/up GEMM. Twins of
// the gemma push structs (forward_cuda_gemma.zig) — the SAME kernels
// (build_expert_order_padded + gemm_q4k_experts_grouped_tc) serve both paths.
const BuildOrderPadPush = extern struct { T: u32, n_used: u32, n_experts: u32, routing_stride: u32, max_pos: u32 };
const GroupedTCPush = extern struct { M: u32, K: u32, base: u32, gu_full: u32, dst_tok_stride: u32 };
const GroupedI8Push = extern struct { M: u32, K: u32, T: u32, base: u32, expert_stride: u32, dst_tok_stride: u32, route_stride: u32 };
// Down-projection grouped-TC (Q5_K): A is the SwiGLU output indexed per work-item
// (token*n_used + slot), so it carries `slice`/`n_used` instead of `base`/`gu_full`.
const GroupedTCDownPush = extern struct { M: u32, K: u32, slice: u32, n_used: u32, dst_tok_stride: u32 };
// Effort 26 T0 (qwen batched prefill): cuBLAS fp16-TC dense GEMM helpers + the
// batched SSM/util kernels. Must byte-match kernels.cu.
const F32ToF16Push = extern struct { N: u32 };
const QuantActPush = extern struct { K: u32, T: u32 };
const GateUpSwigluPush = extern struct { M: u32, K: u32, T: u32, gate_a_offset: u32 = 0, up_a_offset: u32 = 0, x_offset: u32 = 0, y_offset: u32 = 0 };
const GemmQ8Push = extern struct { M: u32, K: u32, T: u32, a_offset: u32, x_q8_stride: u32 };
const DequantQ4KPush = extern struct { M: u32, K: u32, a_offset: u32 = 0 };
const DequantQ5KPush = extern struct { M: u32, K: u32, a_offset: u32 = 0 };
const DequantQ6KPush = extern struct { M: u32, K: u32, a_offset: u32 = 0 };
const DequantQ8_0Push = extern struct { M: u32, K: u32, a_offset: u32 = 0 };
const AddPush = extern struct { N: u32 };
const CopyPush = extern struct { N: u32, src_offset: u32 = 0, dst_offset: u32 = 0 };
const ConcatRowsPush = extern struct { N: u32, T: u32 };
const ConvBatchPush = extern struct { conv_channels: u32, d_conv: u32, kernel_is_f16: u32, n_tok: u32, state_offset: u32 };
const GatedNormBatchPush = extern struct { d_inner: u32, dt_rank: u32, head_v_dim: u32, d_state: u32, norm_per_head: u32, n_tok: u32 };
// Effort 26 T0: batched attention-inner twins (grid.y = T).
const DeintBatchPush = extern struct { head_dim: u32, n_head: u32, T: u32 };
const RopeBatchPush = extern struct { stride: u32, rope_dim: u32, n_heads: u32, base_position: u32, freq_base_bits: u32, attn_scale_bits: u32 };
const KvWriteBatchPush = extern struct { kv_dim: u32, dst_base: u32, T: u32 };
const AttnBatchPush = extern struct { head_dim: u32, n_heads: u32, n_kv_heads: u32, T: u32, attn_scale_bits: u32, sink_offset: u32, base_position: u32 };

/// Map a GGUF quant type to its DMMV kernel pipeline (indexes ForwardCuda.dmmv).
fn dmmvIdx(t: gguf.GGMLType) usize {
    return switch (t) {
        .q4_k => 0,
        .q5_k => 1,
        .q6_k => 2,
        .q8_0 => 3,
        .f32 => 4,
        else => 0,
    };
}

/// Effort 28: a stacked-MoE expert weight type that has a GPU-side `dmmv_*_experts`
/// kernel (reads the chosen expert id from `router_out_buf` GPU-side, no host
/// readback). The async `batched_experts` path is enabled only when ALL of
/// gate/up/down are supported; any other type (q8_0/f32) keeps the host-id
/// fallback (correct, but per-row sync). Adding q6_k (cycle 2026-06-15) brings the
/// 5 mixed-quant Q6_K-expert layers of qwen36-35b-a3b onto the async/collapse path.
fn expertsSupported(t: gguf.GGMLType) bool {
    return switch (t) {
        .q4_k, .q5_k, .q6_k => true,
        else => false,
    };
}

/// Derived dims (one decode token, qwen35).
const Derived = struct {
    n_embd: u32,
    n_ff: u32,
    n_head: u32,
    n_kv_head: u32,
    head_dim: u32,
    q_dim: u32, // n_head * head_dim = 4096
    kv_dim: u32, // n_kv_head * head_dim = 1024
    vocab: u32,
    rms_eps: f32,
    n_layers: u32,
    full_attn_interval: u32,
    rope_dim: u32,
    rope_freq_base: f32,
    // SSM
    d_conv: u32,
    d_inner: u32,
    d_state: u32,
    dt_rank: u32,
    n_group: u32,
    head_v_dim: u32, // d_inner / dt_rank = 128
    conv_channels: u32, // d_inner + 2*n_group*d_state = 8192
    conv_state_len: u32, // (d_conv-1) * conv_channels per ssm layer
    ssm_state_len: u32, // dt_rank * head_v_dim * head_v_dim per ssm layer
    // MoE (0 for the dense qwen35; >0 for the qwen2_moe qwen36)
    n_experts: u32, // total routed experts (256)
    n_experts_used: u32, // top-k active experts per token (8)
    shexp_ff: u32, // shared-expert intermediate dim (512)

    fn from(c: anytype) Derived {
        const head_dim = c.head_dim;
        const q_dim = c.n_heads * head_dim;
        const kv_dim = c.n_kv_heads * head_dim;
        const head_v_dim = c.ssm_d_inner / c.ssm_dt_rank;
        const conv_channels = c.ssm_d_inner + 2 * c.ssm_n_group * c.ssm_d_state;
        return .{
            .n_embd = c.hidden_dim,
            .n_ff = c.intermediate_dim,
            .n_head = c.n_heads,
            .n_kv_head = c.n_kv_heads,
            .head_dim = head_dim,
            .q_dim = q_dim,
            .kv_dim = kv_dim,
            .vocab = c.vocab_size,
            .rms_eps = c.rms_norm_eps,
            .n_layers = c.n_layers,
            .full_attn_interval = c.full_attn_interval,
            .rope_dim = c.rope_dim,
            .rope_freq_base = c.rope_freq_base,
            .d_conv = c.ssm_d_conv,
            .d_inner = c.ssm_d_inner,
            .d_state = c.ssm_d_state,
            .dt_rank = c.ssm_dt_rank,
            .n_group = c.ssm_n_group,
            .head_v_dim = head_v_dim,
            .conv_channels = conv_channels,
            .conv_state_len = (c.ssm_d_conv - 1) * conv_channels,
            .ssm_state_len = c.ssm_dt_rank * head_v_dim * head_v_dim,
            .n_experts = c.n_experts,
            .n_experts_used = c.n_experts_used,
            .shexp_ff = c.shared_expert_intermediate_dim,
        };
    }
};

/// All NVRTC-compiled kernels used by the decode path.
const Pipelines = struct {
    rms_norm: CudaPipeline,
    rms_norm_quant_q8: CudaPipeline, // RMS norm + packed Q8_1 activation
    dmmv: [5]CudaPipeline, // q4k, q5k, q6k, q8_0, f32
    dmmv_fast: [4]CudaPipeline, // q4k_fast, q5k_fast, q6k_fast, q8_0_fast (block=64)
    dmmv_q4k_q8_fast: CudaPipeline, // experimental Q4_K x Q8_1 decode matvec
    dmmv_q5k_q8_fast: CudaPipeline, // experimental Q5_K x Q8_1 decode matvec
    dmmv_q4k_q8_btok2: CudaPipeline, // two-token Q4_K x packed-Q8 verifier
    dmmv_q6k_q8_btok2: CudaPipeline, // two-token Q6_K x packed-Q8 verifier
    dmmv_q4k_gate_up_swiglu: CudaPipeline, // fused dense-decode FFN gate/up/SwiGLU
    dmmv_q4k_gate_up_swiglu_q8: CudaPipeline, // experimental Q8_1 activation decode FFN
    dmmv_q4k_gate_up_swiglu_q8_btok2: CudaPipeline, // two-token packed-Q8 FFN
    dmmv_q6k_q8_fast: CudaPipeline, // experimental Q6_K x Q8_1 decode matvec
    dmmv_f32_dual: CudaPipeline, // fused SSM alpha/beta projection
    dmmv_q4k_pair: CudaPipeline, // true same-input Q4_K projection pair
    dmmv_q4k_pair_q8: CudaPipeline, // experimental paired Q4_K x Q8_1 matvec
    dmmv_q4k_pair_q8_btok2: CudaPipeline, // two-token packed-Q8 projection pair
    // Effort 28 4c: batched-decode GEMM (one weight read amortized over B rows).
    gemm: [4]CudaPipeline, // q4k, q5k, q6k, q8_0 tiled_v2
    gemm_f32: CudaPipeline, // f32 weights (e.g. some ssm projections)
    gemm_q4k_tc: CudaPipeline, // tensor-core Q4_K GEMM (ZINC_PREFILL_TC=1)
    gemm_q4k_dp4a: CudaPipeline, // DP4a int8 Q4_K GEMM (ZINC_PREFILL_DP4A=1)
    gemm_q4k_wmma_i8: CudaPipeline, // gfx12 int8-WMMA Q4_K x Q8_1 GEMM
    gemm_q5k_wmma_i8: CudaPipeline, // gfx12 int8-WMMA Q5_K x Q8_1 GEMM
    gemm_q6k_wmma_i8: CudaPipeline, // gfx12 int8-WMMA Q6_K x Q8_1 GEMM
    gemm_q4k_wmma_i8_t48: CudaPipeline, // exact 48-token short-prompt tile
    gemm_q4k_wmma_i8_t48_direct: CudaPipeline, // register-direct Q4_K short-prompt tile
    gemm_q5k_wmma_i8_t48: CudaPipeline,
    gemm_q6k_wmma_i8_t48: CudaPipeline,
    gemm_q4k_wmma_i8_t64: CudaPipeline, // exact 64-token prompt tile
    gemm_q5k_wmma_i8_t64: CudaPipeline,
    gemm_q6k_wmma_i8_t64: CudaPipeline,
    gemm_q4k_wmma_i8_t80: CudaPipeline, // experimental 80-token prompt tile
    gemm_q5k_wmma_i8_t80: CudaPipeline,
    gemm_q6k_wmma_i8_t80: CudaPipeline,
    gemm_q8_0_wmma_i8: CudaPipeline,
    gemm_q8_0_wmma_i8_t48: CudaPipeline,
    gemm_q8_0_wmma_i8_t64: CudaPipeline,
    gemm_q8_0_wmma_i8_t80: CudaPipeline,
    gemm_q8_0_wmma_i8_t16: CudaPipeline,
    gemm_q4k_wmma_i8_t16: CudaPipeline, // short remainder tile after full 112-token tiles
    gemm_q5k_wmma_i8_t16: CudaPipeline,
    gemm_q6k_wmma_i8_t16: CudaPipeline,
    gemm_f16_tc: CudaPipeline, // fp16 pre-dequanted TC GEMM (ZINC_PREFILL_F16=1)
    gemm_q4k_tc_lowsmem: CudaPipeline, // 8KB-shared TC GEMM (3x occupancy)
    gemm_q4k_mma_lowsmem: CudaPipeline, // inline-PTX fp16 MMA path
    gemm_q4k_gate_up_swiglu: CudaPipeline, // fused gate+up+SwiGLU TC GEMM
    gemm_q8_0_tc_lowsmem: CudaPipeline, // 16KB-shared TC Q8_0 GEMM (shared experts)
    gemm_q6k_tc_lowsmem: CudaPipeline, // 16KB-shared TC Q6_K GEMM
    gemm_q5k_tc_lowsmem: CudaPipeline, // 16KB-shared TC Q5_K GEMM
    quantize_act_q8: CudaPipeline, // float→Q8_0 activation pre-quantizer
    gemm_q4k_q8_dp4a: CudaPipeline, // Q4_K × Q8_0 DP4a GEMM (pre-quant path)
    // Effort 28: Q4_K token-BATCH matvec for small-B decode (idx B-2, B=2..8).
    // Reads each weight row once + amortizes the dequant over B tokens, dodging
    // the 64-tile padding waste of the batched GEMM at small B (opt-in).
    dmmv_q4k_btok: [26]CudaPipeline, // Effort 28: token-batch matvec B=2..27 (idx B-2)
    dmmv_q6k_btok: [26]CudaPipeline, // Effort 28: Q6_K small-B token-batch matvec
    dmmv_q5k_btok: [26]CudaPipeline, // Effort 28: Q5_K small-B token-batch matvec
    dmmv_q8_0_btok: [26]CudaPipeline, // Effort 28: Q8_0 small-B token-batch matvec
    rope: CudaPipeline,
    kv_cache_write: CudaPipeline,
    naive_attention: CudaPipeline,
    // Effort 28 4c-2: fused batched-decode attention (grid.y=B per-seq) twins.
    qwen_norm_rope_qkv_seq: CudaPipeline,
    qwen_norm_rope_qkv: CudaPipeline,
    naive_attention_batched_seq: CudaPipeline,
    ssm_conv1d_seq: CudaPipeline,
    ssm_delta_net_seq: CudaPipeline,
    ssm_gated_norm_seq: CudaPipeline,
    sigmoid_mul: CudaPipeline,
    sigmoid_mul_quant_q8: CudaPipeline,
    deinterleave: CudaPipeline,
    ssm_conv1d: CudaPipeline,
    ssm_conv1d_prepare: CudaPipeline,
    ssm_delta_net: CudaPipeline,
    ssm_delta_net_prepare: CudaPipeline, // normalize Q/K + activate gate/beta once per token/head
    ssm_delta_net_col_warp: CudaPipeline, // block-compatible state scan using four wave32 columns
    ssm_delta_net_col_warp_history: CudaPipeline, // speculative state after each row
    ssm_delta_net_warp: CudaPipeline, // warp-level scan (no __syncthreads in hot loop)
    ssm_delta_net_warp_history: CudaPipeline, // speculative state after each row
    ssm_delta_net_chunked: CudaPipeline, // chunked parallel delta-net (ZINC_SSM_CHUNKED)
    ssm_gated_norm: CudaPipeline,
    ssm_gated_norm_quant_q8: CudaPipeline,
    swiglu: CudaPipeline,
    argmax: CudaPipeline,
    argmax_partials: CudaPipeline,
    argmax_finalize: CudaPipeline,
    embed_q4k: CudaPipeline, // GPU-side Q4_K embedding-row dequant (Effort 25 c5)
    // MoE (qwen2_moe). Compiled unconditionally; only dispatched when n_experts>0.
    softmax_topk: CudaPipeline,
    softmax_topk_batched: CudaPipeline, // Effort 29 T2: per-token top-k over all T
    moe_weighted_acc: CudaPipeline,
    sigmoid_scale_acc: CudaPipeline,
    dmmv_q4k_experts: CudaPipeline, // batched gate/up over all experts
    dmmv_q5k_experts: CudaPipeline, // batched gate/up or down over all experts
    dmmv_q6k_experts: CudaPipeline, // batched down (the 4 Q6_K layers of 35b-a3b)
    // Effort 29 T2: token-batched (grid.y = T) MoE prefill twins.
    dmmv_q4k_experts_batched: CudaPipeline,
    dmmv_q5k_experts_batched: CudaPipeline,
    dmmv_q6k_experts_batched: CudaPipeline,
    dmmv_f32_batched: CudaPipeline, // router logits + shared gate scalar (all T)
    moe_weighted_acc_batched: CudaPipeline, // routed combine (all T)
    sigmoid_scale_acc_batched: CudaPipeline, // shared-expert gating (all T)
    // T2 (qwen MoE port): single-launch padded grouped Tensor-core gate/up GEMM
    // over the routed experts (ZINC_MOE_TC). Same kernels as the gemma path.
    build_expert_order_padded: CudaPipeline,
    build_expert_order_padded_16: CudaPipeline,
    gemm_q4k_experts_grouped_tc: CudaPipeline,
    gemm_q4k_experts_grouped_i8: CudaPipeline,
    gemm_q4k_experts_grouped_i8_direct: CudaPipeline,
    gemm_q4k_experts_grouped_i8_t16: CudaPipeline,
    gemm_q4k_experts_grouped_i8_t16_direct: CudaPipeline,
    gemm_q4k_experts_grouped_i8_t16_m64: CudaPipeline,
    gemm_q4k_experts_grouped_i8_t16_m32: CudaPipeline,
    gemm_q5k_experts_grouped_i8: CudaPipeline,
    gemm_q5k_experts_grouped_i8_t16: CudaPipeline,
    gemm_q5k_experts_grouped_i8_t16_m32: CudaPipeline,
    gemm_q6k_experts_grouped_i8: CudaPipeline,
    gemm_q6k_experts_grouped_i8_t16: CudaPipeline,
    gemm_q6k_experts_grouped_i8_t16_m32: CudaPipeline,
    gemm_q5k_experts_grouped_tc: CudaPipeline,
    gemm_q6k_experts_grouped_tc: CudaPipeline, // down twin for the 4 Q6_K layers
    // Effort 26 T0: batched-prefill GEMM + SSM kernels (qwen prefillBatched).
    dequant_q4k_to_f16: CudaPipeline, // full Q4_K weight [M,K] → fp16 for cuBLAS
    dequant_q5k_to_f16: CudaPipeline, // full Q5_K weight [M,K] → fp16 for cuBLAS (qwen attn_qkv/ssm_out)
    dequant_q6k_to_f16: CudaPipeline, // full Q6_K weight [M,K] → fp16 for cuBLAS
    dequant_q8_0_to_f16: CudaPipeline, // full Q8_0 weight [M,K] → fp16 for cuBLAS (shexp)
    f32_to_f16: CudaPipeline, // activation downcast [T,K] → fp16 for cuBLAS
    add_inplace: CudaPipeline, // residual fold: hidden += projection
    copy_f32: CudaPipeline, // device-local state checkpoint / row copy
    concat_rows: CudaPipeline, // NextN [token embedding | trunk hidden] input
    ssm_conv1d_batched: CudaPipeline, // one launch over all T (circular state)
    ssm_conv1d_batched_history: CudaPipeline, // speculative state after each row
    ssm_gated_norm_batched: CudaPipeline, // grid.y = T (stateless per token)
    // Effort 26 T0: batched attention-inner kernels (collapse the per-token loop).
    deinterleave_batched: CudaPipeline,
    rope_batched: CudaPipeline,
    kv_cache_write_batched: CudaPipeline,
    attention_causal_batched: CudaPipeline,
    attention_causal_v2: CudaPipeline, // coalesced warp-per-key qwen prefill attention (ZINC_ATTN_V2)
};

/// Token-major scratch for the batched prefill path (Effort 26 T0). Allocated
/// lazily by `ensureBatch(T)`; one buffer per [T, dim] activation. Buffers are
/// generously named per role; SSM-only and FFN-only buffers coexist (a layer
/// uses one or the other). Sized so a single allocation serves both the dense
/// qwen35 and the qwen36 MoE catalog rows.
const BatchScratch = struct {
    t_cap: u32,
    hidden: CudaBuffer, // [T, n_embd] residual stream
    norm: CudaBuffer, // [T, n_embd] pre-block norm output
    o: CudaBuffer, // [T, n_embd] projection output folded into hidden
    qfull: CudaBuffer, // [T, 2*q_dim] packed [Q|gate] (attn)
    q: CudaBuffer, // [T, q_dim]
    attn_gate: CudaBuffer, // [T, q_dim] deinterleaved attn gate
    k: CudaBuffer, // [T, kv_dim]
    v: CudaBuffer, // [T, kv_dim]
    attn_out: CudaBuffer, // [T, q_dim]
    qkv: CudaBuffer, // [T, conv_channels] ssm qkv projection (conv input)
    conv_out: CudaBuffer, // [T, conv_channels] conv1d output
    z: CudaBuffer, // [T, d_inner] ssm z-gate
    alpha: CudaBuffer, // [T, dt_rank]
    beta: CudaBuffer, // [T, dt_rank]
    delta_out: CudaBuffer, // [T, d_inner] delta-net scan output
    ssm_gn: CudaBuffer, // [T, d_inner] gated-norm output (ssm out-proj input)
    ffn_norm: CudaBuffer, // [T, n_embd]
    gate_ff: CudaBuffer, // [T, n_ff]
    up_ff: CudaBuffer, // [T, n_ff]
    swiglu_ff: CudaBuffer, // [T, n_ff]
    act_f16: CudaBuffer, // [T, maxK] fp16 activation for cuBLAS
    w_f16: CudaBuffer, // [maxM*maxK] fp16 dequant'd weight for cuBLAS
    act_q8: CudaBuffer, // [T, maxK/32*34] Q8_0 pre-quantized activation for DP4a
    // Effort 29 T2: token-major MoE prefill scratch (n_experts>0; size-1 stubs on
    // the dense qwen35). Mirrors the per-token moeFfnBlock buffers, batched over T.
    router_logits_e: CudaBuffer, // [T, n_experts]
    router_table_e: CudaBuffer, // [T, 2*n_used] (ids then weight-bits per token)
    gate_e: CudaBuffer, // [T, n_used*ef] routed gate projection (slot-major/token)
    up_e: CudaBuffer, // [T, n_used*ef] routed up projection
    swiglu_e: CudaBuffer, // [T, n_used*ef] routed SwiGLU
    down_e: CudaBuffer, // [T, n_used*n_embd] routed down (slot-major per token)
    gate_scalar_e: CudaBuffer, // [T] shared-expert sigmoid gate logit per token
    // T2 (qwen MoE port): padded counting-sort order + per-tile expert id for the
    // single-launch grouped-TC gate/up GEMM (ZINC_MOE_TC). Stubs on dense qwen35.
    padded_order: CudaBuffer, // [P + 64*n_experts] u32 ((token<<16|slot) or INV)
    tile_expert: CudaBuffer, // [(P + 64*n_experts)/64] u32 (expert id per 64-tile)
};

/// Effort 28 4c: token-major [B, dim] activation scratch for batched decode.
/// Distinct from the single-token scratch on ForwardCuda — additive, never
/// aliases it. Sized to the per-row max over the attention / SSM / FFN blocks so
/// every batched projection (and the per-row inner on row b's slice) has room.
const DecodeBatch = struct {
    b_cap: u32,
    hidden: CudaBuffer, // [B, n_embd] residual stream
    norm: CudaBuffer, // [B, n_embd] pre-block rms norm
    qfull: CudaBuffer, // [B, 2*q_dim] packed [Q|gate] (attn)
    q: CudaBuffer, // [B, q_dim]
    k: CudaBuffer, // [B, kv_dim]
    v: CudaBuffer, // [B, kv_dim]
    gate: CudaBuffer, // [B, max(q_dim, d_inner, n_ff)] attn gate / ssm z / ffn gate
    attn_out: CudaBuffer, // [B, max(q_dim, conv_channels, d_inner)] attn out / ssm qkv / delta_out
    swiglu: CudaBuffer, // [B, max(n_ff, conv_channels, d_inner)] conv_out / gated_norm / ffn swiglu
    up: CudaBuffer, // [B, n_ff] ffn up
    alpha: CudaBuffer, // [B, dt_rank] ssm alpha
    beta: CudaBuffer, // [B, dt_rank] ssm beta
    ssm_delta: CudaBuffer, // [B, d_inner] ssm delta-net output (dedicated stride)
    ssm_y: CudaBuffer, // [B, d_inner] ssm gated-norm output → out-proj
    ffn_norm: CudaBuffer, // [B, n_embd] ffn pre-norm
    moe_down: CudaBuffer, // [B, n_embd] batched MoE shared-expert down output (pre-scale)
    moe_gate_scalar: CudaBuffer, // [B] batched MoE shared-expert sigmoid gate logit
    // Effort 29 T3: fp16 scratch for the B>27 serving-decode cuBLAS GEMM (one
    // dequant'd weight at a time + the B activation columns). Sized like the
    // prefill BatchScratch's w_f16/act_f16 (largest decode weight / B*max_k).
    w_f16: CudaBuffer, // [max_w] one dequant'd weight (fp16) for cuBLAS
    act_f16: CudaBuffer, // [B, max_k] fp16 activation column block for cuBLAS

    fn alloc(ctx: ?*shim.CudaCtx, d: Derived, b_cap: u32) !DecodeBatch {
        const f4 = @sizeOf(f32);
        const dt = @max(@as(u32, 1), d.dt_rank);
        const di = @max(@as(u32, 1), d.d_inner);
        const gate_n = @max(d.q_dim, @max(d.d_inner, d.n_ff));
        const aout_n = @max(d.q_dim, @max(d.conv_channels, d.d_inner));
        const swig_n = @max(d.n_ff, @max(d.conv_channels, d.d_inner));
        const max_k = @max(@max(d.n_embd, d.q_dim), @max(d.d_inner, d.n_ff));
        const max_w = @max(@max(d.conv_channels, 2 * d.q_dim), d.n_ff) * @max(d.n_embd, d.n_ff);
        return .{
            .b_cap = b_cap,
            .hidden = try buffer.createBuffer(ctx, b_cap * d.n_embd * f4),
            .norm = try buffer.createBuffer(ctx, b_cap * d.n_embd * f4),
            .qfull = try buffer.createBuffer(ctx, b_cap * 2 * d.q_dim * f4),
            .q = try buffer.createBuffer(ctx, b_cap * d.q_dim * f4),
            .k = try buffer.createBuffer(ctx, b_cap * d.kv_dim * f4),
            .v = try buffer.createBuffer(ctx, b_cap * d.kv_dim * f4),
            .gate = try buffer.createBuffer(ctx, b_cap * gate_n * f4),
            .attn_out = try buffer.createBuffer(ctx, b_cap * aout_n * f4),
            .swiglu = try buffer.createBuffer(ctx, b_cap * swig_n * f4),
            .up = try buffer.createBuffer(ctx, b_cap * d.n_ff * f4),
            .alpha = try buffer.createBuffer(ctx, b_cap * dt * f4),
            .beta = try buffer.createBuffer(ctx, b_cap * dt * f4),
            .ssm_delta = try buffer.createBuffer(ctx, b_cap * di * f4),
            .ssm_y = try buffer.createBuffer(ctx, b_cap * di * f4),
            .ffn_norm = try buffer.createBuffer(ctx, b_cap * d.n_embd * f4),
            .moe_down = try buffer.createBuffer(ctx, b_cap * d.n_embd * f4),
            .moe_gate_scalar = try buffer.createBuffer(ctx, b_cap * f4),
            .w_f16 = try buffer.createBuffer(ctx, max_w * @sizeOf(u16)),
            .act_f16 = try buffer.createBuffer(ctx, b_cap * max_k * @sizeOf(u16)),
        };
    }

    fn free(self: *DecodeBatch) void {
        inline for (.{ &self.hidden, &self.norm, &self.qfull, &self.q, &self.k, &self.v, &self.gate, &self.attn_out, &self.swiglu, &self.up, &self.alpha, &self.beta, &self.ssm_delta, &self.ssm_y, &self.ffn_norm, &self.moe_down, &self.moe_gate_scalar, &self.w_f16, &self.act_f16 }) |b| {
            buffer.freeBuffer(b);
        }
    }
};

const mtp_max_draft: u32 = 3;
const mtp_max_verify: u32 = mtp_max_draft + 1;

/// Lazily allocated state for Qwen3.8's appended NextN block. The target's
/// attention KV remains in ForwardCuda's ordinary layer arrays; this owns only
/// rollback copies for recurrent layers and the small host/device staging used
/// by draft/verify cycles.
const MtpState = struct {
    recurrent_snapshot: []CudaBuffer,
    conv_snapshot: []CudaBuffer,
    conv_off_snapshot: []u32,
    verify_logits: CudaBuffer, // [mtp_max_verify, vocab]
    verify_argmax: CudaBuffer, // [mtp_max_verify] u32
    h_input: CudaBuffer, // one target/MTP hidden row for single-token drafting
    pending_h: []f32, // normalized target hidden preceding the next seed token
    target_h: []f32, // downloaded verification h rows
    pair_h: []f32, // shifted h rows paired with accepted target tokens
    primed: bool = false,
    cycles: u32 = 0,
    drafted: u32 = 0,
    accepted: u32 = 0,
    draft_ns: u64 = 0,
    target_ns: u64 = 0,
    restore_ns: u64 = 0,
    catchup_ns: u64 = 0,

    fn deinit(self: *MtpState, allocator: std.mem.Allocator) void {
        for (self.recurrent_snapshot) |*b| buffer.freeBuffer(b);
        for (self.conv_snapshot) |*b| buffer.freeBuffer(b);
        allocator.free(self.recurrent_snapshot);
        allocator.free(self.conv_snapshot);
        allocator.free(self.conv_off_snapshot);
        buffer.freeBuffer(&self.verify_logits);
        buffer.freeBuffer(&self.verify_argmax);
        buffer.freeBuffer(&self.h_input);
        allocator.free(self.pending_h);
        allocator.free(self.target_h);
        allocator.free(self.pair_h);
        self.* = undefined;
    }
};

pub const MtpCycleResult = struct {
    next_token: u32,
    n_drafted: u32,
    n_accepted: u32,
    drafts: [mtp_max_draft]u32,
};

pub const MtpPerf = struct {
    cycles: u32,
    draft_ms: f64,
    target_ms: f64,
    restore_ms: f64,
    catchup_ms: f64,
};

/// Per-token GPU forward state for qwen35 greedy decode.
pub const ForwardCuda = struct {
    allocator: std.mem.Allocator,
    ctx: ?*shim.CudaCtx,
    model: *loader.Model,
    d: Derived,
    max_ctx: u32,

    pipes: Pipelines,

    // activation scratch (device, f32)
    hidden: CudaBuffer,
    norm_buf: CudaBuffer,
    qfull_buf: CudaBuffer, // [2*q_dim] packed [Q|gate] interleaved per head (attn)
    q_buf: CudaBuffer,
    k_buf: CudaBuffer,
    v_buf: CudaBuffer,
    gate_buf: CudaBuffer, // max(n_embd, n_ff) for FFN gate / attn-z gate
    attn_out_buf: CudaBuffer, // q_dim for attn; conv_channels for SSM qkv
    ffn_norm_buf: CudaBuffer,
    up_buf: CudaBuffer,
    swiglu_buf: CudaBuffer,
    router_buf: CudaBuffer, // ssm alpha [dt_rank]
    down_buf: CudaBuffer, // ssm beta [dt_rank]; MoE: slot-major down outputs [n_used*n_embd]
    logits_buf: CudaBuffer,
    argmax_buf: CudaBuffer, // u32 x1
    argmax_partial_buf: CudaBuffer, // 128 f32 values followed by 128 u32 indices
    tok_in_buf: CudaBuffer, // u32 x1: device token id, source for the GPU embed lookup
    use_fused_decode_ffn: bool = false,
    use_decode_pair_reduce: bool = false,
    use_q4_pair_reduce: bool = false,
    use_decode_q8_ffn: bool = false,
    use_decode_q8_q6: bool = false,
    use_decode_q8_lm: bool = false,
    use_decode_q8_q4: bool = false,
    use_decode_q8_q6_proj: bool = false,
    use_decode_q8_q5: bool = false,
    use_decode_q8_q4_pair: bool = false,
    use_argmax_v2: bool = false,
    use_decode_ssm_col_warp: bool = false,
    use_decode_ssm_fast: bool = false,
    use_fused_ssm_f32: bool = false,
    use_fused_q4_pairs: bool = false,
    use_fused_attn_frontend: bool = false,
    use_rocm_rms_q8: bool = false,
    lightweight_rocm_commands: bool = false,
    // async decode command ring (dense path): commitAsync'd layer commands are
    // stashed here and freed after the tail commitAndWait drains the shared
    // CUstream. Defaults so the init literal need not list them.
    pending: [1024]command.CudaCommand = undefined,
    n_pending: u32 = 0,
    // CUDA-graph decode replay (Effort 25, behind ZINC_CUDA_GRAPH). When set, the
    // dense per-step kernel chain is stream-captured into a cached CUgraphExec and
    // replayed as ONE graph launch — collapsing the ~480 launch-bound kernel
    // dispatches/token into a single submission. `capturing` is true only while
    // recording, switching `submit`/attention to a no-sync capture path.
    graph: ?*shim.CudaGraph = null,
    capturing: bool = false,
    // Effort 28: CUDA-graph replay for the BATCHED dense-decode step (qwen35),
    // one cached CUgraphExec per distinct batch size B (index B). Opt-in via
    // ZINC_BATCH_GRAPH (`batch_graph_on`) / harness `batch_graph_force`. The
    // captured region is sync-free thanks to the dense launch-collapse (commit
    // 3c6ca650): every batched block rides the async `submit` no-sync path under
    // `capturing`, and the per-step scalars (positions/slots/tokens) live in
    // device scratch uploaded BEFORE capture, so the topology + push-constants are
    // invariant across steps → `cuGraphExecUpdate` is a cheap in-place no-op.
    // MoE is now ALSO capturable when `moe_graph_capturable` (cycle 2026-06-15):
    // after `4439e0be` (GPU-side expert-id for ALL supported quants) + `6516c010`
    // (per-row sync collapse), the batched MoE routed + shared path contains NO
    // host readback and NO commitAndWait — `submit`/`waitPending` are no-ops under
    // `capturing` — and the expert id is read GPU-side from a device buffer (not a
    // push constant), so topology + push-constants stay invariant across steps. Set
    // only by `decodeBatch`.
    batch_graph: [9]?*shim.CudaGraph = .{null} ** 9,
    batch_graph_on: bool = false,
    batch_graph_force: ?bool = null,
    // True when EVERY MoE layer's gate/up/down expert tensors have a GPU-side
    // `dmmv_*_experts` kernel (`expertsSupported`) → no host id readback anywhere on
    // the routed path → the batched MoE decode step is stream-capturable. Computed
    // once at init (false for dense models, which use the `n_experts==0` graph gate).
    moe_graph_capturable: bool = false,
    // GPU-side embedding (Effort 25 cycle 5): when the token_embd tensor is Q4_K,
    // dequant the token's row on-GPU (reading the id from `tok_in_buf`) instead of
    // a per-token CPU dequant + full-row H2D. `embed_weight` points at the resident
    // token_embd.weight device buffer; null/false → fall back to the CPU path.
    embed_weight: ?*const CudaBuffer = null,
    embed_gpu: bool = false,
    // Effort 28 B==1 matvec fast path (qwen analog of ForwardGemma.decode_b1).
    // When a `decodeBatch` step batches a single sequence — every per-token
    // prefill and any single-client decode — its per-layer projection/FFN GEMMs
    // waste 63/64 of the 64×64 batched tile on one row. Set true only when B==1
    // AND `b1MatvecOn()` (default-on, opt out ZINC_BATCH_B1_MATVEC=0); routes
    // those GEMMs through the tuned `dmmvDispatch` matvec (exactly what production
    // `decodeStep` uses). Never set by `decodeStep`/`prefillStep`, so the serial
    // path is untouched. `decode_b1_force` (harness-only) overrides the env gate
    // for an in-process A/B (null → env default).
    decode_b1: bool = false,
    decode_b1_force: ?bool = null,
    // Effort 28: at small batch B (2..8) the 64×64 batched GEMM wastes 56-62/64
    // row-slots and goes compute-bound on tile padding (head-to-head: B=8 100%
    // util, slow). Set true in `decodeBatch` for 2≤B≤8 when `mrowMatvecOn()`
    // (opt-in, ZINC_BATCH_MROW=1); routes Q4_K projection/FFN GEMMs through the
    // token-batch matvec (`dmmv_q4k_btok`, weight read once + dequant amortized
    // over B → bandwidth-bound). Never set by `decodeStep`/`prefillStep`.
    // `decode_mrow_force` (harness-only) overrides the env gate for an in-process
    // A/B (null → env default).
    decode_mrow: bool = false,
    decode_mrow_force: ?bool = null,
    // Effort 28 MoE: batch the shared expert across ALL B rows of a decodeBatch
    // step. The shared-expert gate/up/down are DENSE weights (not per-expert), so
    // the per-row `moeFfnRowDecode` loop reads them B× redundantly; running them as
    // ONE GEMM each over B rows reads each weight ONCE. This is only a win through
    // the bandwidth-bound `btok` matvec (it rides the `decode_mrow` gate); through
    // the tile GEMM it loses to the per-row matvec at small B (tile-padding tax).
    // Enabled when `decode_mrow` AND `moeSharedBatchedOn()` (default-on, opt-out
    // ZINC_BATCH_MOE_SHARED=0); `_force` overrides the env gate for an in-process
    // A/B. Never set by decodeStep/prefillStep (they run the single-seq path).
    moe_shared_batched: bool = false,
    moe_shared_batched_force: ?bool = null,
    // Effort 28 MoE launch-collapse: in the batched-decode MoE block the B per-row
    // `moeFfnRoutedRowDecode` calls each ended with a blocking `waitPending` (to make
    // the shared per-row scratch safe to reuse next row). For the pure-GPU async
    // (`batched_experts`) path that sync is UNNECESSARY — all commands ride ONE
    // shared CUstream in-order, so the rows queue back-to-back and drain together at
    // the layer tail, removing B CPU↔GPU round-trips per MoE layer per token (the
    // boost-starved qwen MoE decode is launch-bound). Default-on when the batched
    // MoE path runs; `_force` lets the harness A/B it (false → restore per-row sync).
    moe_collapse: bool = true,
    moe_collapse_force: ?bool = null,
    // Effort 28 DENSE batched-decode launch-collapse: the per-layer batched blocks
    // (`attentionLayerBatchedDecode`/`ssmLayerBatchedDecode`/`ffnBlockBatchedDecode`)
    // each ended with a blocking `commitAndWait` so the shared per-layer scratch
    // (db.norm/db.qfull/db.gate/…) was safe to reuse next layer. That sync is
    // UNNECESSARY — all commands ride ONE shared CUstream in-order, so layer L+1's
    // writes serialize after layer L's reads with no host round-trip. Route those
    // blocks through the async `submit` ring instead and drain once at the decode
    // tail → removes ~2 CPU↔GPU round-trips per layer per token (boost-starved
    // launch-bound decode), AND is the structural prerequisite for graph-capturing
    // the batched step (a captured region must contain no commitAndWait). Default-on
    // when the batched path runs; `_force` lets the harness A/B it (false → restore
    // per-layer sync). Set only by `decodeBatch`; serial decodeStep never touches it.
    decode_collapse: bool = true,
    decode_collapse_force: ?bool = null,
    // MoE scratch (only used when n_experts > 0)
    router_logits_buf: CudaBuffer, // [n_experts] f32 router logits
    router_out_buf: CudaBuffer, // [2*n_experts_used] u32: ids then weight-bits
    gate_scalar_buf: CudaBuffer, // [1] f32 shared-expert sigmoid gate logit
    host_embed: []f32, // PINNED: async embed H2D source (graph-capturable)
    host_tok: []u32, // PINNED: async argmax D2H dest (graph-capturable), len 1
    host_tok_in: []u32, // PINNED: token-id H2D source for the GPU embed lookup, len 1
    host_router_ids: []u32, // [n_experts_used] downloaded expert ids

    // constant buffers
    inv_freq: CudaBuffer, // [rope_dim/2]
    sinks: CudaBuffer, // [n_layers*n_head] NaN

    // KV cache per attention layer (indexed by layer; only attn layers populated)
    kv_k: []CudaBuffer,
    kv_v: []CudaBuffer,
    // SSM state per layer (only ssm layers populated)
    ssm_conv_state: []CudaBuffer,
    ssm_state: []CudaBuffer,
    conv_off: []u32, // per-layer circular conv state offset
    layer_is_attn: []bool, // per-layer: true=attention, false=SSM (detected from tensor names)

    // Effort 26 T0: batched prefill (qwen). Lazily-allocated token-major scratch
    // + the cuBLAS dense-GEMM toggles (mirrors the gemma prefillBatched path).
    batch: ?BatchScratch = null,
    // Qwen3.8 model-native speculative decode. Allocated only for CLI requests
    // that actually opt into/use the appended NextN layer, so ordinary serving
    // keeps its existing memory footprint.
    mtp: ?MtpState = null,
    use_spec_btok: bool = false,
    capture_spec_state: bool = false,
    use_attn_v2: bool = false, // ZINC_ATTN_V2: coalesced warp-per-key qwen prefill attention (token-tolerance)
    use_cublas: bool = false, // dense Q4_K/Q6_K prefill GEMMs via cuBLAS fp16 TC
    use_cublas_q4: bool = false, // route Q4_K dense GEMMs through cuBLAS
    use_cublas_q5: bool = false, // also route Q5_K dense GEMMs through cuBLAS (qwen attn_qkv/ssm_out)
    use_cublas_q6: bool = false, // also route Q6_K dense GEMMs through cuBLAS
    use_cublas_q8: bool = false, // also route Q8_0 dense GEMMs through cuBLAS (qwen36 shexp)
    cublas_min_t: u32 = 256, // cuBLAS for T>=256 (vendor wgmma, but needs large N for efficient tiling). Override with ZINC_CUBLAS_MIN_T
    // T2 (qwen MoE port): route the routed gate/up experts through the single-launch
    // padded grouped Tensor-core GEMM (build_expert_order_padded +
    // gemm_q4k_experts_grouped_tc) instead of the dmmv_*_experts_batched matvec.
    // Q4_K-only (the down + any non-Q4_K gate/up layer stay on the matvec). fp16 →
    // token-tolerance gate, not bit-identical. DEFAULT-ON (opt out ZINC_MOE_TC=0/off),
    // T-gated by moe_tc_min_t (the padded tiles are mostly empty at small T). An
    // EXPLICIT truthy ZINC_MOE_TC=1 bypasses the T-gate (lets validate_catalog
    // exercise the TC path with the short catalog prompt). Mirrors the gemma path.
    use_tc_experts: bool = false,
    tc_experts_forced: bool = false,
    // Crossover measured on the 4090 / qwen36-35b-a3b (256 experts, 8 active): the
    // grouped-TC win is gated higher than the gemma path's 256 because 256 experts
    // each pad to a 64-token tile — at small T the per-expert run (≈T·8/256) is far
    // below 64, so the padded tiles are mostly empty and the wasted TC work cancels
    // the weight-traffic win. The original 1536 gate was set when only gate/up ran
    // on TC; once the FULL expert FFN went on TC (Q5_K + Q6_K down-proj added) the
    // per-tile fixed cost amortizes over the whole FFN → the crossover dropped, the
    // same way the gemma gate fell 512→256 after its Q5_1 down-on-TC. Re-measured on a
    // CLEAN 4090 (order-alternated A/B, forced TC vs matvec, full FFN on TC, 4 rounds
    // each): T=768 4/4 win (TC/MV 1.27-1.51, matvec capped ~415 vs TC 503-590), T=640
    // 4/4 win (1.08-1.49, worst MV-first round +8.4% — clear of the ±1.6% order-bias
    // floor), T=512 MIXED (0.91/1.31/1.15/1.02 — a TC-loss + a tie round). So 640 is
    // the zero-regression crossover; lowered 1024→640 to route the [640,1024) band to
    // the grouped-TC experts (more prompts hit the 29× row's biggest lever). Below 640
    // the proven _batched matvec stays (256 experts → <20 tok/64-tile = mostly-empty
    // padded tiles, the wasted wmma cancels the weight-traffic win).
    moe_tc_min_t: u32 = 64,
    // Effort 29 T3: route the B>27 SERVING-decode dense GEMMs through the same
    // cuBLAS fp16-TC path (dequant W→fp16 + cublasGemmEx) instead of the f32
    // register-tiled gemm[idx]. btok covers B≤27 (bandwidth-bound, no round-trip);
    // past it the tile GEMM pads B to 64 and runs f32 ALU math, so cuBLAS's TC
    // throughput should win. Default-on, opt out ZINC_SERVE_CUBLAS=0. Set per
    // decodeBatch from the gate below; min-B tunable via ZINC_SERVE_CUBLAS_MINB.
    serve_cublas: bool = false,
    serve_cublas_force: ?bool = null,
    serve_cublas_min_b: u32 = 28,
    // Effort 29 T3 option-b: resident fp16 weight cache for the SERVING-decode
    // cuBLAS path. cycle-7's serve_cublas re-dequants the FULL weight to fp16
    // EVERY decode step (~8× btok's weight traffic — cycle-8 quantified the
    // per-step time as dequant-dominated). Serving keeps weights resident, so
    // dequant each dense weight ONCE into a resident fp16 buffer (keyed by its
    // device-ptr) and have the per-step GEMM read the cached fp16 directly,
    // killing the round-trip. Opt-in (ZINC_SERVE_WCACHE=1) — fp16 costs ~3.5×
    // the Q4_K bytes in VRAM, a serving batch-capacity trade; budget-capped
    // (ZINC_SERVE_WCACHE_MB, default 24 GB) with graceful per-GEMM fallback when
    // the cap or VRAM is exhausted. Lazily allocated on first serving GEMM.
    serve_wcache: ?std.AutoHashMap(usize, CudaBuffer) = null,
    serve_wcache_on: bool = false,
    serve_wcache_force: ?bool = null,
    serve_wcache_bytes: usize = 0,

    // When true, ssmLayerBatched uses the block-reduce kernel (state-compatible
    // with the decode ssm_delta_net_seq kernel). Set by prefillBatchedSlot so
    // the batched prefill's SSM state is bit-compatible with subsequent decode.
    force_block_ssm: bool = false,

    // Lazy fp16 weight cache for prefill GEMMs. Keyed by weight device-ptr.
    // On first prefill call, each Q4_K weight is dequanted to fp16 once and
    // cached. Subsequent calls hit the cache → skip the dequant kernel entirely
    // → cuBLAS runs at full fp16 TC throughput (no DRAM round-trip).
    // Decode path is unaffected (uses original Q4_K weights for dmmv).
    // Default-on for ROCm MoE when at least 6 GiB remains after loading the
    // model; opt out with ZINC_PRE_DEQUANT=0. Dense models and tighter devices
    // retain the quantized path. VRAM cost is bounded by the tensors reached.
    w_f16_cache: ?std.AutoHashMap(usize, buffer.CudaBuffer) = null,
    w_f16_cache_bytes: usize = 0,
    w_f16_cache_cap: usize = 8 * 1024 * 1024 * 1024,
    serve_wcache_cap: usize = 24 * 1024 * 1024 * 1024,

    // Effort 28 inc 4 — per-SEQUENCE slot state for batched decode (null until
    // allocSlotState; freed by freeSlotState). Mirrors the gemma slot KV but also
    // carries the hybrid-SSM per-slot conv + recurrent state: attn layers populate
    // the KV slots [n_slots*slot_ctx*kv_dim], ssm layers the conv [n_slots*
    // conv_state_len] + recurrent [n_slots*ssm_state_len] slots; the "wrong" layer
    // type gets a 1-elem stub so free is uniform. ADDITIVE — the single-sequence
    // kv_k/kv_v/ssm_conv_state/ssm_state above stay the production decodeStep path.
    kv_k_slots: ?[]CudaBuffer = null,
    kv_v_slots: ?[]CudaBuffer = null,
    ssm_conv_slots: ?[]CudaBuffer = null,
    ssm_state_slots: ?[]CudaBuffer = null,
    n_slots: u32 = 0,
    slot_ctx: u32 = 0,
    // E28 degradation fix: persistent device scratch for the per-step batched
    // decode positions[]/slots[] uploads — allocated ONCE alongside the slot
    // state (sized to n_slots ≥ B), reused every `decodeBatch` step so the
    // serving loop no longer cudaMalloc/cudaFrees two u32 buffers per decoded
    // token (per-step alloc/free fragments the allocator → monotonic collapse).
    pos_scratch: ?CudaBuffer = null,
    slots_scratch: ?CudaBuffer = null,
    // E28 lever C suspect #2 (qwen port of the gemma `bf4ad2a6` fix): persistent
    // scratch to collapse the per-step embed + tail B-serial GPU round-trips into
    // single batched ops. `argmax_scratch` [n_slots] u32 collects every row's tail
    // argmax (ONE B-wide download vs B downloads/token); `embed_host`
    // [n_slots*n_embd] stages the CPU embed (ONE upload vs B); `tok_scratch`
    // [n_slots] u32 holds the B token ids for the GPU embed path (ONE command
    // buffer vs B commitAndWaits). All sized to n_slots ≥ B, alloc'd in
    // allocSlotState / freed in freeSlotState. ADDITIVE — production decodeStep
    // (single-seq argmax_buf/host_embed/tok_in_buf) is untouched.
    argmax_scratch: ?CudaBuffer = null,
    embed_host: ?[]f32 = null,
    tok_scratch: ?CudaBuffer = null,

    // Effort 28 4c: token-major [B, dim] activation scratch for batched decode.
    // Lazily allocated on the first decodeBatch call, grown if B exceeds cap.
    // ADDITIVE — never aliases the single-sequence scratch above.
    decode_batch: ?DecodeBatch = null,

    /// Compile kernels, allocate every device buffer, upload inv_freq + sinks,
    /// zero the KV cache and SSM state.
    pub fn init(allocator: std.mem.Allocator, model: *loader.Model, max_ctx: u32) !ForwardCuda {
        const ctx = model.ctx;
        // The CUDA forward implements the qwen35/qwen36 hybrid-SSM family (dense +
        // qwen2_moe). Reject other architectures (e.g. gemma4) cleanly here rather
        // than dividing by zero on the absent SSM dims downstream.
        switch (model.config.architecture) {
            .qwen35, .qwen2_moe => {},
            else => |a| {
                log.err("CUDA backend: architecture '{s}' is not implemented (only qwen35/qwen36 family)", .{@tagName(a)});
                return error.UnsupportedArchitecture;
            },
        }
        const d = Derived.from(model.config);

        const src = try allocator.dupeZ(u8, KERNELS_CU);
        defer allocator.free(src);

        var pipes: Pipelines = undefined;
        pipes.rms_norm = try pipeline.createPipeline(ctx, src.ptr, "rms_norm");
        pipes.rms_norm_quant_q8 = try pipeline.createPipeline(ctx, src.ptr, "rms_norm_quant_q8_0");
        pipes.dmmv[0] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k");
        pipes.dmmv[1] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5k");
        pipes.dmmv[2] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q6k");
        pipes.dmmv[3] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q8_0");
        pipes.dmmv[4] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_f32");
        // Fast single-row matvec variants (84-90% bandwidth peak vs ~12-15% base);
        // same DmmvPush ABI + [weight, x, y] buffer order, dispatched at block=64.
        pipes.dmmv_fast[0] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_fast");
        pipes.dmmv_fast[1] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5k_fast");
        pipes.dmmv_fast[2] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q6k_fast");
        pipes.dmmv_fast[3] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q8_0_fast");
        pipes.dmmv_q4k_q8_fast = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_q8_fast");
        pipes.dmmv_q5k_q8_fast = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5k_q8_fast");
        pipes.dmmv_q4k_q8_btok2 = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_q8_btok2");
        pipes.dmmv_q6k_q8_btok2 = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q6k_q8_btok2");
        pipes.dmmv_q4k_gate_up_swiglu = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_gate_up_swiglu");
        pipes.dmmv_q4k_gate_up_swiglu_q8 = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_gate_up_swiglu_q8");
        pipes.dmmv_q4k_gate_up_swiglu_q8_btok2 = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_gate_up_swiglu_q8_btok2");
        pipes.dmmv_q6k_q8_fast = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q6k_q8_fast");
        pipes.dmmv_f32_dual = try pipeline.createPipeline(ctx, src.ptr, "dmmv_f32_dual");
        pipes.dmmv_q4k_pair = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_pair");
        pipes.dmmv_q4k_pair_q8 = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_pair_q8");
        pipes.dmmv_q4k_pair_q8_btok2 = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_pair_q8_btok2");
        pipes.rope = try pipeline.createPipeline(ctx, src.ptr, "rope");
        pipes.kv_cache_write = try pipeline.createPipeline(ctx, src.ptr, "kv_cache_write");
        pipes.naive_attention = try pipeline.createPipeline(ctx, src.ptr, "naive_attention");
        pipes.qwen_norm_rope_qkv_seq = try pipeline.createPipeline(ctx, src.ptr, "qwen_norm_rope_qkv_seq");
        pipes.qwen_norm_rope_qkv = try pipeline.createPipeline(ctx, src.ptr, "qwen_norm_rope_qkv");
        pipes.naive_attention_batched_seq = try pipeline.createPipeline(ctx, src.ptr, "naive_attention_batched_seq");
        pipes.ssm_conv1d_seq = try pipeline.createPipeline(ctx, src.ptr, "ssm_conv1d_seq");
        pipes.ssm_delta_net_seq = try pipeline.createPipeline(ctx, src.ptr, "ssm_delta_net_seq");
        pipes.ssm_gated_norm_seq = try pipeline.createPipeline(ctx, src.ptr, "ssm_gated_norm_seq");
        pipes.sigmoid_mul = try pipeline.createPipeline(ctx, src.ptr, "sigmoid_mul");
        pipes.sigmoid_mul_quant_q8 = try pipeline.createPipeline(ctx, src.ptr, "sigmoid_mul_quant_q8_0");
        pipes.deinterleave = try pipeline.createPipeline(ctx, src.ptr, "deinterleave_qgate");
        pipes.ssm_conv1d = try pipeline.createPipeline(ctx, src.ptr, "ssm_conv1d");
        pipes.ssm_conv1d_prepare = try pipeline.createPipeline(ctx, src.ptr, "ssm_conv1d_prepare");
        pipes.ssm_delta_net = try pipeline.createPipeline(ctx, src.ptr, "ssm_delta_net");
        pipes.ssm_delta_net_prepare = try pipeline.createPipeline(ctx, src.ptr, "ssm_delta_net_prepare");
        pipes.ssm_delta_net_col_warp = try pipeline.createPipeline(ctx, src.ptr, "ssm_delta_net_col_warp");
        pipes.ssm_delta_net_col_warp_history = try pipeline.createPipeline(ctx, src.ptr, "ssm_delta_net_col_warp_history");
        pipes.ssm_delta_net_warp = try pipeline.createPipeline(ctx, src.ptr, "ssm_delta_net_warp");
        pipes.ssm_delta_net_warp_history = try pipeline.createPipeline(ctx, src.ptr, "ssm_delta_net_warp_history");
        pipes.ssm_delta_net_chunked = try pipeline.createPipeline(ctx, src.ptr, "ssm_delta_net_chunked");
        // Chunked kernel needs ~49KB shared memory (k_norm[32KB] + M[16KB] + misc)
        pipeline.setMaxDynamicShared(&pipes.ssm_delta_net_chunked, 64 * 128 * @sizeOf(f32) + 64 * 64 * @sizeOf(f32) + 2 * 64 * @sizeOf(f32));
        pipes.ssm_gated_norm = try pipeline.createPipeline(ctx, src.ptr, "ssm_gated_norm");
        pipes.ssm_gated_norm_quant_q8 = try pipeline.createPipeline(ctx, src.ptr, "ssm_gated_norm_quant_q8_0");
        pipes.swiglu = try pipeline.createPipeline(ctx, src.ptr, "swiglu");
        pipes.argmax = try pipeline.createPipeline(ctx, src.ptr, "argmax");
        pipes.argmax_partials = try pipeline.createPipeline(ctx, src.ptr, "argmax_partials");
        pipes.argmax_finalize = try pipeline.createPipeline(ctx, src.ptr, "argmax_finalize");
        pipes.embed_q4k = try pipeline.createPipeline(ctx, src.ptr, "embed_lookup_q4k");
        pipes.softmax_topk = try pipeline.createPipeline(ctx, src.ptr, "softmax_topk");
        pipes.softmax_topk_batched = try pipeline.createPipeline(ctx, src.ptr, "softmax_topk_batched");
        pipes.moe_weighted_acc = try pipeline.createPipeline(ctx, src.ptr, "moe_weighted_acc");
        pipes.sigmoid_scale_acc = try pipeline.createPipeline(ctx, src.ptr, "sigmoid_scale_acc");
        pipes.dmmv_q4k_experts = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_experts");
        pipes.dmmv_q5k_experts = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5k_experts");
        pipes.dmmv_q6k_experts = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q6k_experts");
        pipes.dmmv_q4k_experts_batched = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_experts_batched");
        pipes.dmmv_q5k_experts_batched = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5k_experts_batched");
        pipes.dmmv_q6k_experts_batched = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q6k_experts_batched");
        pipes.dmmv_f32_batched = try pipeline.createPipeline(ctx, src.ptr, "dmmv_f32_batched");
        pipes.moe_weighted_acc_batched = try pipeline.createPipeline(ctx, src.ptr, "moe_weighted_acc_batched");
        pipes.sigmoid_scale_acc_batched = try pipeline.createPipeline(ctx, src.ptr, "sigmoid_scale_acc_batched");
        // T2 (qwen MoE port): grouped Tensor-core routed gate/up experts.
        pipes.build_expert_order_padded = try pipeline.createPipeline(ctx, src.ptr, "build_expert_order_padded");
        pipes.build_expert_order_padded_16 = try pipeline.createPipeline(ctx, src.ptr, "build_expert_order_padded_16");
        pipes.gemm_q4k_experts_grouped_tc = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_experts_grouped_tc");
        pipes.gemm_q4k_experts_grouped_i8 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_experts_grouped_i8");
        pipes.gemm_q4k_experts_grouped_i8_direct = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_experts_grouped_i8_direct");
        pipes.gemm_q4k_experts_grouped_i8_t16 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_experts_grouped_i8_t16");
        pipes.gemm_q4k_experts_grouped_i8_t16_direct = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_experts_grouped_i8_t16_direct");
        pipes.gemm_q4k_experts_grouped_i8_t16_m64 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_experts_grouped_i8_t16_m64");
        pipes.gemm_q4k_experts_grouped_i8_t16_m32 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_experts_grouped_i8_t16_m32");
        pipes.gemm_q5k_experts_grouped_i8 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q5k_experts_grouped_i8");
        pipes.gemm_q5k_experts_grouped_i8_t16 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q5k_experts_grouped_i8_t16");
        pipes.gemm_q5k_experts_grouped_i8_t16_m32 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q5k_experts_grouped_i8_t16_m32");
        pipes.gemm_q6k_experts_grouped_i8 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_experts_grouped_i8");
        pipes.gemm_q6k_experts_grouped_i8_t16 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_experts_grouped_i8_t16");
        pipes.gemm_q6k_experts_grouped_i8_t16_m32 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_experts_grouped_i8_t16_m32");
        pipes.gemm_q5k_experts_grouped_tc = try pipeline.createPipeline(ctx, src.ptr, "gemm_q5k_experts_grouped_tc");
        pipes.gemm_q6k_experts_grouped_tc = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_experts_grouped_tc");
        // Effort 26 T0: batched-prefill GEMM + SSM kernels.
        pipes.dequant_q4k_to_f16 = try pipeline.createPipeline(ctx, src.ptr, "dequant_q4k_to_f16");
        pipes.dequant_q5k_to_f16 = try pipeline.createPipeline(ctx, src.ptr, "dequant_q5k_to_f16");
        pipes.dequant_q6k_to_f16 = try pipeline.createPipeline(ctx, src.ptr, "dequant_q6k_to_f16");
        pipes.dequant_q8_0_to_f16 = try pipeline.createPipeline(ctx, src.ptr, "dequant_q8_0_to_f16");
        pipes.f32_to_f16 = try pipeline.createPipeline(ctx, src.ptr, "f32_to_f16");
        pipes.add_inplace = try pipeline.createPipeline(ctx, src.ptr, "add_inplace");
        pipes.copy_f32 = try pipeline.createPipeline(ctx, src.ptr, "copy_f32");
        pipes.concat_rows = try pipeline.createPipeline(ctx, src.ptr, "concat_rows");
        pipes.ssm_conv1d_batched = try pipeline.createPipeline(ctx, src.ptr, "ssm_conv1d_batched");
        pipes.ssm_conv1d_batched_history = try pipeline.createPipeline(ctx, src.ptr, "ssm_conv1d_batched_history");
        pipes.ssm_gated_norm_batched = try pipeline.createPipeline(ctx, src.ptr, "ssm_gated_norm_batched");
        pipes.deinterleave_batched = try pipeline.createPipeline(ctx, src.ptr, "deinterleave_qgate_batched");
        pipes.rope_batched = try pipeline.createPipeline(ctx, src.ptr, "rope_batched");
        pipes.kv_cache_write_batched = try pipeline.createPipeline(ctx, src.ptr, "kv_cache_write_batched");
        pipes.attention_causal_batched = try pipeline.createPipeline(ctx, src.ptr, "attention_causal_batched");
        pipes.attention_causal_v2 = try pipeline.createPipeline(ctx, src.ptr, "attention_causal_batched_v2");
        // Effort 28 4c: batched-decode GEMMs (weights read once over B rows).
        pipes.gemm[0] = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_tiled_v2");
        pipes.gemm[1] = try pipeline.createPipeline(ctx, src.ptr, "gemm_q5k_tiled_v2");
        pipes.gemm[2] = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_tiled_v2");
        pipes.gemm[3] = try pipeline.createPipeline(ctx, src.ptr, "gemm_q8_0_tiled_v2");
        pipes.gemm_f32 = try pipeline.createPipeline(ctx, src.ptr, "gemm_f32_tiled_v2");
        pipes.gemm_q4k_tc = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_tc");
        pipes.gemm_q4k_dp4a = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_dp4a");
        pipes.gemm_q4k_wmma_i8 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_wmma_i8");
        pipes.gemm_q5k_wmma_i8 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q5k_wmma_i8");
        pipes.gemm_q6k_wmma_i8 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_wmma_i8");
        pipes.gemm_q4k_wmma_i8_t48 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_wmma_i8_t48");
        pipes.gemm_q4k_wmma_i8_t48_direct = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_wmma_i8_t48_direct");
        pipes.gemm_q5k_wmma_i8_t48 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q5k_wmma_i8_t48");
        pipes.gemm_q6k_wmma_i8_t48 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_wmma_i8_t48");
        pipes.gemm_q4k_wmma_i8_t64 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_wmma_i8_t64");
        pipes.gemm_q5k_wmma_i8_t64 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q5k_wmma_i8_t64");
        pipes.gemm_q6k_wmma_i8_t64 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_wmma_i8_t64");
        pipes.gemm_q4k_wmma_i8_t80 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_wmma_i8_t80");
        pipes.gemm_q5k_wmma_i8_t80 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q5k_wmma_i8_t80");
        pipes.gemm_q6k_wmma_i8_t80 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_wmma_i8_t80");
        pipes.gemm_q8_0_wmma_i8 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q8_0_wmma_i8");
        pipes.gemm_q8_0_wmma_i8_t48 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q8_0_wmma_i8_t48");
        pipes.gemm_q8_0_wmma_i8_t64 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q8_0_wmma_i8_t64");
        pipes.gemm_q8_0_wmma_i8_t80 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q8_0_wmma_i8_t80");
        pipes.gemm_q8_0_wmma_i8_t16 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q8_0_wmma_i8_t16");
        pipes.gemm_q4k_wmma_i8_t16 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_wmma_i8_t16");
        pipes.gemm_q5k_wmma_i8_t16 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q5k_wmma_i8_t16");
        pipes.gemm_q6k_wmma_i8_t16 = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_wmma_i8_t16");
        pipes.gemm_f16_tc = try pipeline.createPipeline(ctx, src.ptr, "gemm_f16_tc");
        pipes.gemm_q4k_tc_lowsmem = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_tc_lowsmem");
        // mma.sync kernel: use nvcc-compiled cubin (NVRTC generates wrong SASS for mma.sync)
        pipes.gemm_q4k_mma_lowsmem = blk: {
            const cc = shim.cuda_compute_capability(ctx);
            log.info("mma.sync cubin: cc={d}, trying nvcc cubin path", .{cc});
            if (cc >= 89) {
                const cubin_data = if (cc >= 100) @embedFile("../cuda/cubins/mma_kernel.sm120.fatbin") else @embedFile("../cuda/cubins/mma_kernel.sm89.fatbin");
                log.info("mma.sync cubin: {d} bytes embedded", .{cubin_data.len});
                if (pipeline.createPipelineFromImage(ctx, cubin_data.ptr, cubin_data.len, "gemm_q4k_mma_lowsmem")) |pipe| {
                    log.info("mma.sync: nvcc cubin loaded OK", .{});
                    break :blk pipe;
                } else |err| {
                    log.warn("mma.sync: cubin load failed ({s}), falling back to NVRTC", .{@errorName(err)});
                }
            } else {
                log.info("mma.sync: cc<89, using NVRTC", .{});
            }
            break :blk pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_mma_lowsmem") catch
                try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_tc_lowsmem");
        };
        pipes.gemm_q4k_gate_up_swiglu = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_gate_up_swiglu_lowsmem");
        pipes.gemm_q8_0_tc_lowsmem = try pipeline.createPipeline(ctx, src.ptr, "gemm_q8_0_tc_lowsmem");
        pipes.gemm_q6k_tc_lowsmem = try pipeline.createPipeline(ctx, src.ptr, "gemm_q6k_tc_lowsmem");
        pipes.gemm_q5k_tc_lowsmem = try pipeline.createPipeline(ctx, src.ptr, "gemm_q5k_tc_lowsmem");
        pipes.quantize_act_q8 = try pipeline.createPipeline(ctx, src.ptr, "quantize_act_q8_0");
        pipes.gemm_q4k_q8_dp4a = try pipeline.createPipeline(ctx, src.ptr, "gemm_q4k_q8_dp4a");
        // Effort 28: token-batch matvecs B=2..16 (idx B-2) for every common decode
        // quant — Q4_K covers most proj/gate/up; Q6_K/Q5_K/Q8_0 cover the residual
        // O-proj/FFN-down/SSM-out on mixed-quant layers. B=9..16 extends btok past
        // the old 8-cap into the higher-concurrency serving regime (btok stays
        // bandwidth-bound to the ~B≈27 roofline crossover vs the padded tile GEMM).
        inline for ([_][]const u8{ "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27" }, 0..) |suf, i| {
            pipes.dmmv_q4k_btok[i] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q4k_btok" ++ suf);
            pipes.dmmv_q6k_btok[i] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q6k_btok" ++ suf);
            pipes.dmmv_q5k_btok[i] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q5k_btok" ++ suf);
            pipes.dmmv_q8_0_btok[i] = try pipeline.createPipeline(ctx, src.ptr, "dmmv_q8_0_btok" ++ suf);
        }
        log.info("nvrtc: compiled {d} kernel pipelines", .{182});

        const f4 = @sizeOf(f32);
        const max_act = @max(d.n_ff, d.conv_channels); // 12288 vs 8192 → 12288
        // Batched MoE buffers hold all n_used experts' gate/up/swiglu (slot-major).
        const moe_act = @max(max_act, d.n_experts_used * d.n_ff);
        // MoE: down_buf is slot-major [n_used * n_embd]; keep ≥64 for the SSM beta reuse.
        const down_elems = @max(@as(u32, 64), d.n_experts_used * d.n_embd);
        // Tiny-but-nonzero stubs so the dense path (n_experts==0) still allocates/free uniformly.
        const router_logits_elems = @max(@as(u32, 1), d.n_experts);
        const router_out_elems = @max(@as(u32, 1), 2 * d.n_experts_used);
        const state_layers = d.n_layers + model.config.n_nextn_layers;
        var self = ForwardCuda{
            .allocator = allocator,
            .ctx = ctx,
            .model = model,
            .d = d,
            .max_ctx = max_ctx,
            .pipes = pipes,
            .hidden = try buffer.createBuffer(ctx, d.n_embd * f4),
            .norm_buf = try buffer.createBuffer(ctx, d.n_embd * f4),
            .qfull_buf = try buffer.createBuffer(ctx, 2 * d.q_dim * f4),
            .q_buf = try buffer.createBuffer(ctx, d.q_dim * f4),
            .k_buf = try buffer.createBuffer(ctx, d.kv_dim * f4),
            .v_buf = try buffer.createBuffer(ctx, d.kv_dim * f4),
            .gate_buf = try buffer.createBuffer(ctx, moe_act * f4),
            .attn_out_buf = try buffer.createBuffer(ctx, d.conv_channels * f4),
            .ffn_norm_buf = try buffer.createBuffer(ctx, d.n_embd * f4),
            .up_buf = try buffer.createBuffer(ctx, moe_act * f4),
            .swiglu_buf = try buffer.createBuffer(ctx, moe_act * f4),
            .router_buf = try buffer.createBuffer(ctx, 64 * f4),
            .down_buf = try buffer.createBuffer(ctx, down_elems * f4),
            .logits_buf = try buffer.createBuffer(ctx, d.vocab * f4),
            .argmax_buf = try buffer.createBuffer(ctx, @sizeOf(u32)),
            .argmax_partial_buf = try buffer.createBuffer(ctx, 128 * (@sizeOf(f32) + @sizeOf(u32))),
            .tok_in_buf = try buffer.createBuffer(ctx, @sizeOf(u32)),
            .router_logits_buf = try buffer.createBuffer(ctx, router_logits_elems * f4),
            .router_out_buf = try buffer.createBuffer(ctx, router_out_elems * @sizeOf(u32)),
            .gate_scalar_buf = try buffer.createBuffer(ctx, f4),
            .host_embed = try buffer.allocHost(f32, d.n_embd),
            .host_tok = try buffer.allocHost(u32, 1),
            .host_tok_in = try buffer.allocHost(u32, 1),
            .host_router_ids = try allocator.alloc(u32, @max(@as(u32, 1), d.n_experts_used)),
            .inv_freq = try buffer.createBuffer(ctx, (d.rope_dim / 2) * f4),
            .sinks = try buffer.createBuffer(ctx, state_layers * d.n_head * f4),
            .kv_k = try allocator.alloc(CudaBuffer, state_layers),
            .kv_v = try allocator.alloc(CudaBuffer, state_layers),
            .ssm_conv_state = try allocator.alloc(CudaBuffer, state_layers),
            .ssm_state = try allocator.alloc(CudaBuffer, state_layers),
            .conv_off = try allocator.alloc(u32, state_layers),
            .layer_is_attn = try allocator.alloc(bool, state_layers),
        };

        if (comptime is_rocm) {
            const fused_ffn = std.posix.getenv("ZINC_ROCM_FUSED_FFN");
            self.use_fused_decode_ffn = if (fused_ffn) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            if (self.use_fused_decode_ffn) {
                log.info("ROCm fused dense FFN: Q4_K gate/up/SwiGLU share one kernel", .{});
            }
            const pair_reduce = std.posix.getenv("ZINC_ROCM_DECODE_PAIR_REDUCE");
            self.use_decode_pair_reduce = if (pair_reduce) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            const q4_pair_reduce = std.posix.getenv("ZINC_ROCM_Q4_PAIR_REDUCE");
            self.use_q4_pair_reduce = if (q4_pair_reduce) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            const decode_q8_ffn = std.posix.getenv("ZINC_ROCM_DECODE_Q8_FFN");
            self.use_decode_q8_ffn = if (decode_q8_ffn) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            const decode_q8_q6 = std.posix.getenv("ZINC_ROCM_DECODE_Q8_Q6");
            self.use_decode_q8_q6 = if (decode_q8_q6) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            const decode_q8_lm = std.posix.getenv("ZINC_ROCM_DECODE_Q8_LM");
            self.use_decode_q8_lm = if (decode_q8_lm) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            const decode_q8_q4 = std.posix.getenv("ZINC_ROCM_DECODE_Q8_Q4");
            self.use_decode_q8_q4 = if (decode_q8_q4) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            const decode_q8_q6_proj = std.posix.getenv("ZINC_ROCM_DECODE_Q8_Q6_PROJ");
            self.use_decode_q8_q6_proj = if (decode_q8_q6_proj) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            const decode_q8_q5 = std.posix.getenv("ZINC_ROCM_DECODE_Q8_Q5");
            self.use_decode_q8_q5 = if (decode_q8_q5) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            const decode_q8_q4_pair = std.posix.getenv("ZINC_ROCM_DECODE_Q8_Q4_PAIR");
            self.use_decode_q8_q4_pair = if (decode_q8_q4_pair) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            const argmax_v2 = std.posix.getenv("ZINC_ROCM_ARGMAX_V2");
            self.use_argmax_v2 = if (argmax_v2) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            const decode_ssm_col_warp = std.posix.getenv("ZINC_ROCM_DECODE_SSM_COL_WARP");
            self.use_decode_ssm_col_warp = if (decode_ssm_col_warp) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            const decode_ssm_fast = std.posix.getenv("ZINC_ROCM_DECODE_SSM_FAST");
            self.use_decode_ssm_fast = if (decode_ssm_fast) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            const fused_ssm_f32 = std.posix.getenv("ZINC_ROCM_FUSED_SSM_F32");
            self.use_fused_ssm_f32 = if (fused_ssm_f32) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            if (self.use_fused_ssm_f32) {
                log.info("ROCm fused SSM F32: alpha/beta projections share one kernel", .{});
            }
            const fused_q4_pairs = std.posix.getenv("ZINC_ROCM_FUSED_Q4_PAIRS");
            self.use_fused_q4_pairs = if (fused_q4_pairs) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            if (self.use_fused_q4_pairs) {
                log.info("ROCm fused Q4_K pairs: attention K/V and SSM qkv/gate share kernels", .{});
            }
            const fused_attn = std.posix.getenv("ZINC_ROCM_FUSED_ATTN_FRONTEND");
            self.use_fused_attn_frontend = if (fused_attn) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            if (self.use_fused_attn_frontend) {
                log.info("ROCm fused attention front-end: deinterleave, norm, RoPE, and KV write share one kernel", .{});
            }
            const rms_q8 = std.posix.getenv("ZINC_ROCM_RMS_Q8");
            self.use_rocm_rms_q8 = if (rms_q8) |v| !(std.mem.eql(u8, v, "0") or
                std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
                std.ascii.eqlIgnoreCase(v, "no")) else true;
            if (self.use_rocm_rms_q8) {
                log.info("ROCm fused RMS + Q8 activation packing enabled", .{});
            }
            // A single ordered HIP stream does not require a completion event for
            // every asynchronously submitted command group. The final tail or an
            // explicit waitPending fence drains all prior launches. This is a
            // measured win across dense, hybrid, and MoE Qwen on the R9700; retain
            // an explicit `=0` fallback for diagnosis.
            self.lightweight_rocm_commands = envFlag("ZINC_ROCM_LIGHT_COMMANDS", true);
            self.cublas_min_t = 1;
        }

        // GPU-side embed: enable only when token_embd.weight is Q4_K and the row
        // length is a whole number of 256-superblocks (always true for these
        // dense qwen models). Other quants (e.g. the 35B MoE's Q8_0) keep the CPU
        // dequant path. The device buffer pointer is stable for the model's life.
        if (model.get("token_embd.weight")) |t| {
            self.embed_weight = &t.gpu_buffer;
            self.embed_gpu = (t.info.type_ == .q4_k) and (d.n_embd % 256 == 0);
        }

        // Per-layer KV / SSM state, sized only for the layer type. We still
        // allocate a tiny stub for the "wrong" type so deinit is uniform.
        for (0..state_layers) |li| {
            const L: u32 = @intCast(li);
            self.conv_off[li] = 0;
            // Detect layer type from tensor presence: attention layers have
            // attn_q.weight, SSM layers have attn_qkv.weight.
            self.layer_is_attn[li] = model.getLayer(L, "attn_q.weight") != null;
            if (li < 8 or li >= d.n_layers) log.info("  layer {d}: {s}", .{ li, if (self.layer_is_attn[li]) "attention" else "SSM" });
            if (li == 0 and d.n_experts > 0) {
                if (model.getLayer(L, "ffn_gate_exps.weight")) |w| log.info("  ffn_gate_exps: type={} dims={any}", .{ w.info.type_, w.info.dims[0..@min(w.info.n_dims, 3)] });
                if (model.getLayer(L, "ffn_down_exps.weight")) |w| log.info("  ffn_down_exps: type={} dims={any}", .{ w.info.type_, w.info.dims[0..@min(w.info.n_dims, 3)] });
                if (model.getLayer(L, "ffn_gate_shexp.weight")) |w| log.info("  ffn_gate_shexp: type={} dims={any}", .{ w.info.type_, w.info.dims[0..@min(w.info.n_dims, 3)] });
                if (model.getLayer(L, "ffn_down_shexp.weight")) |w| log.info("  ffn_down_shexp: type={} dims={any}", .{ w.info.type_, w.info.dims[0..@min(w.info.n_dims, 3)] });
                if (model.getLayer(L, "ffn_gate_inp.weight")) |w| log.info("  ffn_gate_inp: type={} dims={any}", .{ w.info.type_, w.info.dims[0..@min(w.info.n_dims, 3)] });
            }
            if (self.layer_is_attn[li]) {
                self.kv_k[li] = try buffer.createBuffer(ctx, max_ctx * d.kv_dim * f4);
                self.kv_v[li] = try buffer.createBuffer(ctx, max_ctx * d.kv_dim * f4);
                self.ssm_conv_state[li] = try buffer.createBuffer(ctx, f4);
                self.ssm_state[li] = try buffer.createBuffer(ctx, f4);
            } else {
                self.kv_k[li] = try buffer.createBuffer(ctx, f4);
                self.kv_v[li] = try buffer.createBuffer(ctx, f4);
                self.ssm_conv_state[li] = try buffer.createBuffer(ctx, d.conv_state_len * f4);
                self.ssm_state[li] = try buffer.createBuffer(ctx, d.ssm_state_len * f4);
                // zero-init the SSM state
                try zeroBuffer(allocator, ctx, &self.ssm_conv_state[li], d.conv_state_len);
                try zeroBuffer(allocator, ctx, &self.ssm_state[li], d.ssm_state_len);
            }
        }

        // inv_freq[k] = 1 / (freq_base ^ (2k / rope_dim)), k = 0 .. rope_dim/2-1
        {
            const half = d.rope_dim / 2;
            const hf = try allocator.alloc(f32, half);
            defer allocator.free(hf);
            for (0..half) |k| {
                const exp = @as(f32, @floatFromInt(2 * k)) / @as(f32, @floatFromInt(d.rope_dim));
                hf[k] = 1.0 / std.math.pow(f32, d.rope_freq_base, exp);
            }
            buffer.upload(ctx, &self.inv_freq, std.mem.sliceAsBytes(hf));
        }

        // sinks: all NaN → attention sink disabled.
        {
            const n = state_layers * d.n_head;
            const sk = try allocator.alloc(f32, n);
            defer allocator.free(sk);
            @memset(sk, std.math.nan(f32));
            buffer.upload(ctx, &self.sinks, std.mem.sliceAsBytes(sk));
        }

        // CUDA-graph decode replay. Capturable MoE (Effort-27 C2: qwen36-35b-a3b
        // decode +62%, output token-identical to non-graph) is DEFAULT-ON; opt out
        // with ZINC_CUDA_GRAPH=0/off/false/no. The dense path (n_experts==0) is now
        // DEFAULT-ON for SMALL dense models only (Effort-27 C3: qwen35-9b +6.7% on
        // this 4090); large dense (qwen36-27b) regresses ~3% so stays OPT-IN — see the
        // n_embd gate below. Effort-25 first proved the dense graph but left it opt-IN;
        // C3 productizes it for the size where it wins. The MoE path
        // is capturable ONLY when every layer takes
        // the batched async expert path (Effort-27 C1): it reads expert ids GPU-side
        // with NO host readback, so the whole MoE step is a static stream. Before C1's
        // q5k/q6k experts kernels, mixed-quant layers fell to a per-slot fallback that
        // synced + read ids back mid-block (illegal during capture) — why Effort-25 C4
        // found the catalog MoE non-capturable. Non-capturable MoE runs the async
        // per-step path (self.graph stays null).
        // Largest n_embd that still nets a win from the dense decode graph (see the
        // gate below). qwen35-9b (4096) wins; qwen36-27b (5120) regresses.
        const dense_graph_max_embd: u32 = 4096;
        const graphs_supported = shim.cuda_graph_supported() != 0;
        const graph_env = std.posix.getenv("ZINC_CUDA_GRAPH");
        var graph_off = false;
        var graph_optin = false;
        if (graph_env) |e| {
            graph_off = std.mem.eql(u8, e, "0") or std.ascii.eqlIgnoreCase(e, "off") or
                std.ascii.eqlIgnoreCase(e, "false") or std.ascii.eqlIgnoreCase(e, "no");
            graph_optin = !graph_off;
        }
        if (graphs_supported and !graph_off) {
            if (d.n_experts == 0) {
                // Dense decode graph (Effort-25 proof; Effort-27 C3 productized).
                // DEFAULT-ON only for SMALL dense models (n_embd <= 4096, e.g.
                // qwen35-9b: +6.7% on the 4090) where per-launch overhead dominates.
                // Large dense (qwen36-27b n_embd=5120) is OPT-IN: its big per-layer
                // matvecs swamp the launch-bubbles the graph removes, so the graph
                // re-instantiate cost makes it a ~3% REGRESSION (C3 measured 0.967
                // median, 4 rounds) — same size-gating Effort-25 C4 saw on MoE graphs.
                if (d.n_embd <= dense_graph_max_embd or graph_optin) {
                    self.graph = shim.cuda_graph_create();
                    log.info("CUDA graph: dense decode enabled (n_embd={d}, opt out ZINC_CUDA_GRAPH=0)", .{d.n_embd});
                }
            } else if (self.moeGraphCapturable()) {
                self.graph = shim.cuda_graph_create();
                log.info("CUDA graph: capturable MoE decode default-on (opt out ZINC_CUDA_GRAPH=0)", .{});
            } else {
                log.info("CUDA graph: MoE has a non-batched expert layer (per-slot fallback) — graph disabled", .{});
            }
        }
        // Effort 28: is the batched MoE decode step stream-capturable? Only when
        // EVERY MoE layer is on the GPU-side `batched_experts` path (all of
        // gate/up/down `expertsSupported`) — else a layer host-gathers its expert
        // ids (download + commitAndWait), illegal mid-capture. Dense models stay on
        // the `n_experts==0` gate. Scanned once here (model weights are loaded).
        if (d.n_experts > 0) {
            self.moe_graph_capturable = blk: {
                var L: u32 = 0;
                while (L < d.n_layers) : (L += 1) {
                    const wge = self.model.getLayer(L, "ffn_gate_exps.weight") orelse break :blk false;
                    const wue = self.model.getLayer(L, "ffn_up_exps.weight") orelse break :blk false;
                    const wde = self.model.getLayer(L, "ffn_down_exps.weight") orelse break :blk false;
                    if (!expertsSupported(wge.info.type_) or !expertsSupported(wue.info.type_) or !expertsSupported(wde.info.type_)) break :blk false;
                }
                break :blk true;
            };
        }
        // Effort 28: CUDA-graph replay for the BATCHED decode step. Per-B execs are
        // created lazily on first use in `decodeBatch`.
        // - MoE-capturable models (every routed layer reads expert ids GPU-side, no
        //   host readback) are DEFAULT-ON: validated WIN on a CLEAN 5090 window
        //   (qwen36-35b-a3b B=4, 3 boost-comparable rounds, graph ON +36..+46% agg,
        //   token-identical BATCH_GATE+SANITY) — the boost-starved MoE step recovers
        //   its per-step launch bubble via graph replay (mirrors e27's serial
        //   MoE-graph default-on). Opt OUT with ZINC_BATCH_GRAPH=0/off/false/no.
        // - Dense models stay OPT-IN (measured NEUTRAL-to-slightly-negative on a
        //   clean window — the dense batched step is already launch-collapsed): env
        //   must be set truthy.
        const bg_env = std.posix.getenv("ZINC_BATCH_GRAPH");
        const bg_falsey = if (bg_env) |v| (std.mem.eql(u8, v, "0") or
            std.ascii.eqlIgnoreCase(v, "off") or std.ascii.eqlIgnoreCase(v, "false") or
            std.ascii.eqlIgnoreCase(v, "no")) else false;
        if (graphs_supported and self.moe_graph_capturable) {
            self.batch_graph_on = !bg_falsey; // default-on for MoE-capturable, opt-out
        } else if (graphs_supported and d.n_experts == 0) {
            self.batch_graph_on = (bg_env != null) and !bg_falsey; // dense opt-in
        }
        if (self.batch_graph_on) {
            log.info("ZINC_BATCH_GRAPH: batched decode will replay via per-B captured CUDA graphs (moe_capturable={})", .{self.moe_graph_capturable});
        }

        // Qwen MoE's resident Q8 shared weights are a proven prefill win. Let
        // each lazy allocation decide whether it fits instead of sampling free
        // VRAM once during init: after a sequence of large benchmark processes,
        // ROCm can report temporarily fragmented free space at that instant.
        const pre_dequant_default = is_rocm and d.n_experts > 0;
        if (envFlag("ZINC_PRE_DEQUANT", pre_dequant_default) or envFlag("ZINC_PREFILL_F16", false)) {
            self.w_f16_cache = std.AutoHashMap(usize, buffer.CudaBuffer).init(allocator);
            self.w_f16_cache_cap = @as(usize, envU32("ZINC_PRE_DEQUANT_MB", 8 * 1024)) * 1024 * 1024;
            log.info("persistent fp16 prefill weight cache enabled (cap {d} MiB)", .{self.w_f16_cache_cap / (1024 * 1024)});
        }

        return self;
    }

    pub fn deinit(self: *ForwardCuda) void {
        @setEvalBranchQuota(2000);
        const a = self.allocator;
        self.freeBatch();
        if (self.mtp) |*state| state.deinit(a);
        if (self.graph) |g| shim.cuda_graph_free(g);
        for (&self.batch_graph) |*bg| if (bg.*) |g| shim.cuda_graph_free(g);
        inline for (.{ &self.hidden, &self.norm_buf, &self.qfull_buf, &self.q_buf, &self.k_buf, &self.v_buf, &self.gate_buf, &self.attn_out_buf, &self.ffn_norm_buf, &self.up_buf, &self.swiglu_buf, &self.router_buf, &self.down_buf, &self.logits_buf, &self.argmax_buf, &self.argmax_partial_buf, &self.tok_in_buf, &self.router_logits_buf, &self.router_out_buf, &self.gate_scalar_buf, &self.inv_freq, &self.sinks }) |b| {
            buffer.freeBuffer(b);
        }
        for (self.kv_k) |*b| buffer.freeBuffer(b);
        for (self.kv_v) |*b| buffer.freeBuffer(b);
        for (self.ssm_conv_state) |*b| buffer.freeBuffer(b);
        for (self.ssm_state) |*b| buffer.freeBuffer(b);
        if (self.decode_batch) |*db| db.free();
        if (self.serve_wcache) |*cache| {
            var it = cache.valueIterator();
            while (it.next()) |buf| buffer.freeBuffer(buf);
            cache.deinit();
        }
        if (self.w_f16_cache) |*cache| {
            var it = cache.valueIterator();
            while (it.next()) |buf| buffer.freeBuffer(buf);
            cache.deinit();
        }
        self.freeSlotState();
        a.free(self.kv_k);
        a.free(self.kv_v);
        a.free(self.ssm_conv_state);
        a.free(self.ssm_state);
        a.free(self.conv_off);
        a.free(self.layer_is_attn);
        buffer.freeHost(self.host_embed);
        buffer.freeHost(self.host_tok);
        buffer.freeHost(self.host_tok_in);
        a.free(self.host_router_ids);
        inline for (std.meta.fields(Pipelines)) |f| {
            if (comptime std.mem.eql(u8, f.name, "dmmv")) {
                for (&self.pipes.dmmv) |*p| pipeline.freePipeline(p);
            } else if (comptime std.mem.eql(u8, f.name, "dmmv_fast")) {
                for (&self.pipes.dmmv_fast) |*p| pipeline.freePipeline(p);
            } else if (comptime std.mem.eql(u8, f.name, "gemm")) {
                for (&self.pipes.gemm) |*p| pipeline.freePipeline(p);
            } else if (comptime std.mem.eql(u8, f.name, "dmmv_q4k_btok")) {
                for (&self.pipes.dmmv_q4k_btok) |*p| pipeline.freePipeline(p);
            } else if (comptime std.mem.eql(u8, f.name, "dmmv_q6k_btok")) {
                for (&self.pipes.dmmv_q6k_btok) |*p| pipeline.freePipeline(p);
            } else if (comptime std.mem.eql(u8, f.name, "dmmv_q5k_btok")) {
                for (&self.pipes.dmmv_q5k_btok) |*p| pipeline.freePipeline(p);
            } else if (comptime std.mem.eql(u8, f.name, "dmmv_q8_0_btok")) {
                for (&self.pipes.dmmv_q8_0_btok) |*p| pipeline.freePipeline(p);
            } else {
                pipeline.freePipeline(&@field(self.pipes, f.name));
            }
        }
        self.* = undefined;
    }

    // ---- Effort 28 inc 4: per-sequence slot state (batched decode plumbing) ----

    /// Allocate per-sequence slot state for `n_slots` concurrent sequences, each
    /// with up to `slot_ctx` positions. Attn layers get slot KV; ssm layers get
    /// per-slot conv + recurrent state (zero-initialised). Additive — production
    /// single-sequence buffers are untouched. Re-callable (frees first).
    pub fn allocSlotState(self: *ForwardCuda, n_slots: u32, slot_ctx: u32) !void {
        self.freeSlotState();
        const ctx = self.ctx;
        const f4 = @sizeOf(f32);
        const d = self.d;
        const kk = try self.allocator.alloc(CudaBuffer, d.n_layers);
        const vv = try self.allocator.alloc(CudaBuffer, d.n_layers);
        const cc = try self.allocator.alloc(CudaBuffer, d.n_layers);
        const ss = try self.allocator.alloc(CudaBuffer, d.n_layers);
        for (0..d.n_layers) |li| {
            const L: u32 = @intCast(li);
            if (self.layer_is_attn[L]) {
                const kv_bytes = @as(usize, n_slots) * slot_ctx * d.kv_dim * f4;
                kk[li] = try buffer.createBuffer(ctx, kv_bytes);
                vv[li] = try buffer.createBuffer(ctx, kv_bytes);
                cc[li] = try buffer.createBuffer(ctx, f4);
                ss[li] = try buffer.createBuffer(ctx, f4);
            } else {
                kk[li] = try buffer.createBuffer(ctx, f4);
                vv[li] = try buffer.createBuffer(ctx, f4);
                cc[li] = try buffer.createBuffer(ctx, @as(usize, n_slots) * d.conv_state_len * f4);
                ss[li] = try buffer.createBuffer(ctx, @as(usize, n_slots) * d.ssm_state_len * f4);
                try zeroBuffer(self.allocator, ctx, &cc[li], n_slots * d.conv_state_len);
                try zeroBuffer(self.allocator, ctx, &ss[li], n_slots * d.ssm_state_len);
            }
        }
        self.kv_k_slots = kk;
        self.kv_v_slots = vv;
        self.ssm_conv_slots = cc;
        self.ssm_state_slots = ss;
        self.n_slots = n_slots;
        self.slot_ctx = slot_ctx;
        // Persistent per-step decode scratch (see field comment): sized to the
        // max batch (n_slots) so every decodeBatch step just re-uploads into it.
        self.pos_scratch = try buffer.createBuffer(ctx, @as(usize, n_slots) * @sizeOf(u32));
        self.slots_scratch = try buffer.createBuffer(ctx, @as(usize, n_slots) * @sizeOf(u32));
        // Suspect-#2 port: persistent tail/embed scratch (see field comment).
        self.argmax_scratch = try buffer.createBuffer(ctx, @as(usize, n_slots) * @sizeOf(u32));
        self.tok_scratch = try buffer.createBuffer(ctx, @as(usize, n_slots) * @sizeOf(u32));
        self.embed_host = try self.allocator.alloc(f32, @as(usize, n_slots) * d.n_embd);
    }

    pub fn freeSlotState(self: *ForwardCuda) void {
        inline for (.{ &self.kv_k_slots, &self.kv_v_slots, &self.ssm_conv_slots, &self.ssm_state_slots }) |field| {
            if (field.*) |bufs| {
                for (bufs) |*b| buffer.freeBuffer(b);
                self.allocator.free(bufs);
                field.* = null;
            }
        }
        inline for (.{ &self.pos_scratch, &self.slots_scratch, &self.argmax_scratch, &self.tok_scratch }) |field| {
            if (field.*) |*b| {
                buffer.freeBuffer(b);
                field.* = null;
            }
        }
        if (self.embed_host) |h| {
            self.allocator.free(h);
            self.embed_host = null;
        }
        self.n_slots = 0;
        self.slot_ctx = 0;
    }

    /// Byte offset of slot `slot`'s K/V for position `pos` in attn layer `L`'s slot
    /// KV: (slot*slot_ctx + pos)*kv_dim*sizeof(f32). The future per-seq kv-write +
    /// slot attention kernels (4b/4c) will use this exact indexing.
    pub fn slotKvOffsetBytes(self: *const ForwardCuda, slot: u32, pos: u32) usize {
        return (@as(usize, slot) * self.slot_ctx + pos) * self.d.kv_dim * @sizeOf(f32);
    }
    /// Byte offset of slot `slot`'s conv ring (ssm layer): slot*conv_state_len*f4.
    pub fn slotConvOffsetBytes(self: *const ForwardCuda, slot: u32) usize {
        return @as(usize, slot) * self.d.conv_state_len * @sizeOf(f32);
    }
    /// Byte offset of slot `slot`'s recurrent state (ssm layer): slot*ssm_state_len*f4.
    pub fn slotStateOffsetBytes(self: *const ForwardCuda, slot: u32) usize {
        return @as(usize, slot) * self.d.ssm_state_len * @sizeOf(f32);
    }

    /// Effort 28 increment 4 (qwen serving): zero a REUSED slot's SSM conv ring +
    /// recurrent state so a new request prefilling from pos=0 into this slot starts
    /// from clean state. The serving engine reuses slots across requests
    /// (nslots < concurrent clients) and qwen's SSM state is ACCUMULATED (not
    /// position-indexed), so without this a reused slot inherits the prior request's
    /// recurrent state → corruption. The attention slot KV needs NO reset (it is
    /// position-indexed and overwritten on prefill; reads never reach beyond the
    /// current sequence's positions — same reason gemma's slot KV reuses cleanly).
    /// No-op if slot state is unallocated. Call on ADMIT, before per-token prefill.
    pub fn resetSlot(self: *ForwardCuda, slot: u32) !void {
        const cc = self.ssm_conv_slots orelse return;
        const ss = self.ssm_state_slots orelse return;
        const d = self.d;
        const f4 = @sizeOf(f32);
        std.debug.assert(slot < self.n_slots);
        var L: u32 = 0;
        while (L < d.n_layers) : (L += 1) {
            if (self.layer_is_attn[L]) continue; // attn KV is position-overwritten
            var convst = try buffer.aliasBuffer(&cc[L], self.slotConvOffsetBytes(slot), d.conv_state_len * f4);
            defer buffer.freeBuffer(&convst);
            try zeroBuffer(self.allocator, self.ctx, &convst, d.conv_state_len);
            var recst = try buffer.aliasBuffer(&ss[L], self.slotStateOffsetBytes(slot), d.ssm_state_len * f4);
            defer buffer.freeBuffer(&recst);
            try zeroBuffer(self.allocator, self.ctx, &recst, d.ssm_state_len);
        }
    }

    /// Upload distinct data into two device regions of `buf` and read both back to
    /// prove they do NOT alias (the slot-offset arithmetic carves non-overlapping
    /// per-sequence regions). Returns false if either write clobbered the other.
    fn nonOverlapCheck(self: *ForwardCuda, buf: *CudaBuffer, off_a: usize, off_b: usize, n: u32) !bool {
        const ctx = self.ctx;
        const f4 = @sizeOf(f32);
        const a = try self.allocator.alloc(f32, n);
        defer self.allocator.free(a);
        const b = try self.allocator.alloc(f32, n);
        defer self.allocator.free(b);
        const rd = try self.allocator.alloc(f32, n);
        defer self.allocator.free(rd);
        for (a, 0..) |*v, i| v.* = 7.5 + @as(f32, @floatFromInt(i));
        @memset(b, -3.0);
        var va = try buffer.aliasBuffer(buf, off_a, n * f4);
        defer buffer.freeBuffer(&va);
        var vb = try buffer.aliasBuffer(buf, off_b, n * f4);
        defer buffer.freeBuffer(&vb);
        buffer.upload(ctx, &vb, std.mem.sliceAsBytes(b));
        buffer.upload(ctx, &va, std.mem.sliceAsBytes(a));
        buffer.download(ctx, &va, std.mem.sliceAsBytes(rd));
        for (a, rd) |x, y| if (x != y) return false;
        buffer.download(ctx, &vb, std.mem.sliceAsBytes(rd));
        for (b, rd) |x, y| if (x != y) return false;
        return true;
    }

    /// Sub-step 4a plumbing smoke: prove the slot-offset arithmetic round-trips and
    /// that distinct slots map to NON-overlapping device regions for BOTH the
    /// attention KV and the SSM conv + recurrent state. Requires allocSlotState
    /// with n_slots >= 2 (so slot 0 and slot n_slots-1 differ). Returns true on
    /// success.
    pub fn slotStateSmoke(self: *ForwardCuda) !bool {
        if (self.kv_k_slots == null or self.n_slots < 2 or self.slot_ctx == 0) return error.SlotStateNotAllocated;
        const d = self.d;
        var attn_l: ?u32 = null;
        var ssm_l: ?u32 = null;
        for (0..d.n_layers) |li| {
            const L: u32 = @intCast(li);
            if (self.layer_is_attn[L]) {
                if (attn_l == null) attn_l = L;
            } else if (ssm_l == null) ssm_l = L;
        }
        // Attn KV: slot 0 / pos 0 vs the last slot / last position.
        if (attn_l) |L| {
            const ks = &self.kv_k_slots.?[L];
            const off_a = self.slotKvOffsetBytes(0, 0);
            const off_b = self.slotKvOffsetBytes(self.n_slots - 1, self.slot_ctx - 1);
            if (!try self.nonOverlapCheck(ks, off_a, off_b, d.kv_dim)) return false;
        }
        // SSM conv + recurrent: slot 0 vs the last slot.
        if (ssm_l) |L| {
            const cs = &self.ssm_conv_slots.?[L];
            if (!try self.nonOverlapCheck(cs, self.slotConvOffsetBytes(0), self.slotConvOffsetBytes(self.n_slots - 1), d.conv_state_len)) return false;
            const ss = &self.ssm_state_slots.?[L];
            if (!try self.nonOverlapCheck(ss, self.slotStateOffsetBytes(0), self.slotStateOffsetBytes(self.n_slots - 1), d.ssm_state_len)) return false;
        }
        return true;
    }

    /// Reset the production single-sequence recurrent state to a fresh-process
    /// start: zero every ssm layer's conv ring + recurrent state and clear the
    /// per-layer conv offset. The attention KV cache needs no reset — it is
    /// position-indexed and each sequence overwrites from pos 0, reading only what
    /// it wrote. Used by the batch harness to make the per-sequence SERIAL
    /// reference truly single-sequence (without this, the unindexed SSM recurrent
    /// state leaks from one reference sequence into the next). Additive — never
    /// called on the real decode/prefill path; that path runs one sequence/process.
    pub fn resetState(self: *ForwardCuda) !void {
        const d = self.d;
        for (0..d.n_layers) |li| {
            const L: u32 = @intCast(li);
            self.conv_off[li] = 0;
            if (!self.layer_is_attn[L]) {
                try zeroBuffer(self.allocator, self.ctx, &self.ssm_conv_state[li], d.conv_state_len);
                try zeroBuffer(self.allocator, self.ctx, &self.ssm_state[li], d.ssm_state_len);
            }
        }
    }

    /// Run one greedy decode step for `token` at sequence position `pos`,
    /// returning the argmax token id. v0: embed → final rms_norm → LM head →
    /// argmax (layers are gated by `run_layers`).
    pub fn decodeStep(self: *ForwardCuda, token: u32, pos: u32, run_layers: bool) !u32 {
        const d = self.d;
        const ctx = self.ctx;

        // EMBED. GPU path (Q4_K token_embd): stage the token id into the tiny pinned
        // host_tok_in and dequant its row on-GPU into `hidden` (no per-token CPU
        // dequant; H2D shrinks from a full n_embd row to 4 bytes). CPU fallback:
        // dequant the row into the pinned host_embed and upload it.
        const use_graph = run_layers and self.graph != null;
        if (self.embed_gpu) {
            self.host_tok_in[0] = token;
        } else {
            self.model.dequantEmbeddingRow(token, self.host_embed);
            // Non-graph path uploads (sync) here. The graph path defers the H2D into
            // decodeStepGraph so it is CAPTURED as the graph's first node — pinned.
            if (!use_graph) buffer.upload(ctx, &self.hidden, std.mem.sliceAsBytes(self.host_embed));
        }

        // TAIL tensors (resolved up-front so the graph and async paths share them).
        const out_norm = self.model.get("output_norm.weight") orelse return error.MissingTensor;
        const lm_head = self.model.get("output.weight") orelse return error.MissingTensor;

        // CUDA-graph replay path: capture the full dense per-step chain (embed +
        // layers + tail + argmax D2H) and launch it as one graph. Bit-identical
        // to the async chain (same kernels, same order, same single stream).
        if (use_graph) {
            return try self.decodeStepGraph(pos, &out_norm.gpu_buffer, &lm_head.gpu_buffer, lm_head.info.type_);
        }

        // Non-graph GPU embed: tiny token H2D, then the embed dispatch leads the
        // stream (ordered before the layer ops; drained by the tail commitAndWait).
        if (self.embed_gpu) {
            buffer.upload(ctx, &self.tok_in_buf, std.mem.sliceAsBytes(self.host_tok_in));
            try self.recordEmbed();
        }

        if (run_layers) {
            var L: u32 = 0;
            while (L < d.n_layers) : (L += 1) {
                if (self.layer_is_attn[L]) {
                    try self.attentionLayer(L, pos);
                } else {
                    try self.ssmLayer(L);
                }
                if (d.n_experts > 0) try self.moeFfnBlock(L) else try self.ffnBlock(L);
            }
        }

        // TAIL: final rms_norm → LM head → argmax.
        var cmd = try command.beginCommand(ctx);
        self.tailDispatch(&cmd, &out_norm.gpu_buffer, &lm_head.gpu_buffer, lm_head.info.type_);
        cmd.commitAndWait(); // drains the whole stream incl. the async layer ops
        self.drainPending(); // free the stashed async commands (completion guaranteed)

        var tok: u32 = 0;
        buffer.download(ctx, &self.argmax_buf, std.mem.asBytes(&tok));
        return tok;
    }

    /// Prefill helper: run every layer (build the KV cache) but SKIP the tail
    /// rms_norm + LM head + argmax. A prompt-internal token's logits are never
    /// used — only the final prompt token's argmax seeds generation — so the
    /// (vocab x n_embd) head matvec is pure waste on T-1 of the T prompt tokens.
    /// It is a large share for the MoE models, whose active per-token forward is
    /// small next to the full-vocab head. The async layer ops are drained here
    /// (waitPending) so the ring is freed each token; bit-identical generation.
    pub fn prefillStep(self: *ForwardCuda, token: u32, pos: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        self.model.dequantEmbeddingRow(token, self.host_embed);
        buffer.upload(ctx, &self.hidden, std.mem.sliceAsBytes(self.host_embed));
        var L: u32 = 0;
        while (L < d.n_layers) : (L += 1) {
            if (self.layer_is_attn[L]) {
                try self.attentionLayer(L, pos);
            } else {
                try self.ssmLayer(L);
            }
            if (d.n_experts > 0) try self.moeFfnBlock(L) else try self.ffnBlock(L);
        }
        self.waitPending(); // drain the async layer ops; no logits for prompt-internal tokens
    }

    pub fn prefillBatched(self: *ForwardCuda, tokens: []const u32) !u32 {
        const d = self.d;
        const ctx = self.ctx;
        const T: u32 = @intCast(tokens.len);
        const f4 = @sizeOf(f32);
        const profile = std.posix.getenv("ZINC_PREFILL_PROFILE") != null;
        var prof_attn: i64 = 0;
        var prof_ssm: i64 = 0;
        var prof_ffn: i64 = 0;
        // cuBLAS dense-GEMM defaults (mirror the gemma path): on unless opted out.
        self.use_attn_v2 = prefillAttnV2On();
        self.use_cublas = cublasDefaultOn();
        // On ROCm, Q4/Q5/Q6 use the native pre-quantized Q8 kernels: repeated
        // hipBLAS dequantization was slower on gfx1201. F32 (and Q8_0) still use
        // hipBLAS. CUDA keeps its established cuBLAS quant defaults.
        self.use_cublas_q4 = self.use_cublas and !is_rocm and std.posix.getenv("ZINC_BATCHED_CUBLAS_NOQ4") == null;
        self.use_cublas_q5 = self.use_cublas and !is_rocm and std.posix.getenv("ZINC_BATCHED_CUBLAS_NOQ5") == null;
        self.use_cublas_q6 = self.use_cublas and !is_rocm and std.posix.getenv("ZINC_BATCHED_CUBLAS_NOQ6") == null;
        self.use_cublas_q8 = self.use_cublas and std.posix.getenv("ZINC_BATCHED_CUBLAS_NOQ8") == null;
        // ROCm initializes this threshold to 1 because hipBLAS F32 projection
        // batching wins even on short prompts; CUDA retains its existing gate.
        if (std.posix.getenv("ZINC_CUBLAS_MIN_T")) |mt| {
            if (std.fmt.parseInt(u32, mt, 10)) |val| self.cublas_min_t = val else |_| {}
        }
        // Effort 29 T2: token-batched MoE FFN (qwen36). DEFAULT-ON; opt out to the
        // proven per-token loop via ZINC_QWEN_MOE_BATCHED=0 (for the A/B baseline).
        const moe_batched = qwenMoeBatchedOn();
        // T2 (qwen MoE port): grouped Tensor-core routed gate/up experts
        // (ZINC_MOE_TC). The grouped expert kernels are still inert compatibility
        // symbols in the ROCm module, so dispatch the exact token-batched matvec
        // path there until native gfx12 grouped kernels exist. Calling the stubs
        // leaves routed-output scratch stale across requests and corrupts a reused
        // server after its first prompt.
        self.use_tc_experts = moeTcDefaultOn();
        self.tc_experts_forced = moeTcForced();
        if (is_rocm) self.moe_tc_min_t = 16;
        if (profile and d.n_experts > 0) {
            log.info("MoE prefill policy: T={d} enabled={} forced={} min_t={d}", .{ T, self.use_tc_experts, self.tc_experts_forced, self.moe_tc_min_t });
        }

        const b = try self.ensureBatch(T);

        // EMBED all T tokens → hidden [T, n_embd] (qwen has no embedding scale).
        const host = try self.allocator.alloc(f32, T * d.n_embd);
        defer self.allocator.free(host);
        for (0..T) |t| self.model.dequantEmbeddingRow(tokens[t], host[t * d.n_embd ..][0..d.n_embd]);
        buffer.upload(ctx, &b.hidden, std.mem.sliceAsBytes(host));

        var L: u32 = 0;
        while (L < d.n_layers) : (L += 1) {
            if (profile) {
                self.waitPending();
                const t0 = std.time.milliTimestamp();
                if (self.layer_is_attn[L]) {
                    try self.attentionLayerBatched(L, T, b, 0);
                } else {
                    try self.ssmLayerBatched(L, T, b);
                }
                self.waitPending();
                const dt0 = std.time.milliTimestamp() - t0;
                if (self.layer_is_attn[L]) prof_attn += dt0 else prof_ssm += dt0;
                const t1 = std.time.milliTimestamp();
                if (d.n_experts > 0) {
                    if (moe_batched and self.moeBatchedSupported(L)) {
                        try self.moeFfnBlockBatched(L, T, b);
                    } else {
                        const saved_hidden = self.hidden;
                        var t: u32 = 0;
                        while (t < T) : (t += 1) {
                            self.hidden = try buffer.aliasBuffer(&b.hidden, t * d.n_embd * f4, d.n_embd * f4);
                            try self.moeFfnBlock(L);
                            buffer.freeBuffer(&self.hidden);
                        }
                        self.hidden = saved_hidden;
                    }
                } else {
                    try self.ffnBlockBatched(L, T, b);
                }
                self.waitPending();
                prof_ffn += std.time.milliTimestamp() - t1;
            } else {
                if (self.layer_is_attn[L]) {
                    try self.attentionLayerBatched(L, T, b, 0);
                } else {
                    try self.ssmLayerBatched(L, T, b);
                }
                if (d.n_experts > 0) {
                    if (moe_batched and self.moeBatchedSupported(L)) {
                        try self.moeFfnBlockBatched(L, T, b);
                    } else {
                        const saved_hidden = self.hidden;
                        var t: u32 = 0;
                        while (t < T) : (t += 1) {
                            self.hidden = try buffer.aliasBuffer(&b.hidden, t * d.n_embd * f4, d.n_embd * f4);
                            try self.moeFfnBlock(L);
                            buffer.freeBuffer(&self.hidden);
                        }
                        self.hidden = saved_hidden;
                    }
                } else {
                    try self.ffnBlockBatched(L, T, b);
                }
            }
        }
        self.waitPending(); // drain every layer's async commands before the tail.

        if (profile) {
            const total = prof_attn + prof_ssm + prof_ffn;
            std.log.info("PREFILL_PROFILE: attn={d}ms ssm={d}ms ffn={d}ms total={d}ms (T={d})", .{
                prof_attn, prof_ssm, prof_ffn, total, T,
            });
        }

        // TAIL on the last token only: rms_norm → LM head → argmax.
        const last = T - 1;
        const out_norm = self.model.get("output_norm.weight") orelse return error.MissingTensor;
        const lm_head = self.model.get("output.weight") orelse return error.MissingTensor;
        var hid_last = try buffer.aliasBuffer(&b.hidden, last * d.n_embd * f4, d.n_embd * f4);
        defer buffer.freeBuffer(&hid_last);

        var cmd = try command.beginCommand(ctx);
        const want_q8 = self.use_decode_q8_lm and lm_head.info.type_ == .q6_k;
        const norm_q8_ready = self.rmsNormDecodeDispatch(&cmd, &hid_last, &out_norm.gpu_buffer, &self.norm_buf, d.n_embd, want_q8);
        self.lmHeadDispatch(&cmd, &lm_head.gpu_buffer, lm_head.info.type_, norm_q8_ready);
        self.argmaxDispatch(&cmd, &self.argmax_buf);
        cmd.commitAndWait();

        var tok: u32 = 0;
        buffer.download(ctx, &self.argmax_buf, std.mem.asBytes(&tok));
        return tok;
    }

    /// Whether this model/runtime can use its appended Qwen3.8 NextN block.
    /// ROCm enables it by default; ZINC_MTP=0 is the explicit compatibility
    /// escape hatch. CUDA can opt in with ZINC_MTP=1 while that path matures.
    pub fn mtpEnabled(self: *const ForwardCuda) bool {
        if (!envFlag("ZINC_MTP", is_rocm)) return false;
        if (self.model.config.n_nextn_layers != 1 or self.d.n_experts != 0) return false;
        const L = self.d.n_layers;
        return self.model.getLayer(L, "nextn.eh_proj.weight") != null and
            self.model.getLayer(L, "nextn.enorm.weight") != null and
            self.model.getLayer(L, "nextn.hnorm.weight") != null and
            self.model.getLayer(L, "attn_q.weight") != null and
            self.model.getLayer(L, "ffn_gate.weight") != null;
    }

    fn ensureMtpState(self: *ForwardCuda) !*MtpState {
        if (self.mtp) |*state| return state;

        const d = self.d;
        const a = self.allocator;
        const f4 = @sizeOf(f32);
        const n_layers: usize = @intCast(d.n_layers);

        const recurrent = try a.alloc(CudaBuffer, n_layers);
        var n_recurrent: usize = 0;
        errdefer {
            for (recurrent[0..n_recurrent]) |*b| buffer.freeBuffer(b);
            a.free(recurrent);
        }
        const conv = try a.alloc(CudaBuffer, n_layers);
        var n_conv: usize = 0;
        errdefer {
            for (conv[0..n_conv]) |*b| buffer.freeBuffer(b);
            a.free(conv);
        }
        for (0..n_layers) |li| {
            const rec_elems: u32 = if (self.layer_is_attn[li]) 1 else d.ssm_state_len;
            const conv_elems: u32 = if (self.layer_is_attn[li]) 1 else d.conv_state_len;
            recurrent[li] = try buffer.createBuffer(self.ctx, @as(usize, rec_elems) * mtp_max_verify * f4);
            n_recurrent += 1;
            conv[li] = try buffer.createBuffer(self.ctx, @as(usize, conv_elems) * mtp_max_verify * f4);
            n_conv += 1;
        }

        const conv_off = try a.alloc(u32, n_layers);
        errdefer a.free(conv_off);
        const verify_logits = try buffer.createBuffer(self.ctx, @as(usize, mtp_max_verify) * d.vocab * f4);
        errdefer {
            var b = verify_logits;
            buffer.freeBuffer(&b);
        }
        const verify_argmax = try buffer.createBuffer(self.ctx, @as(usize, mtp_max_verify) * @sizeOf(u32));
        errdefer {
            var b = verify_argmax;
            buffer.freeBuffer(&b);
        }
        const h_input = try buffer.createBuffer(self.ctx, @as(usize, d.n_embd) * f4);
        errdefer {
            var b = h_input;
            buffer.freeBuffer(&b);
        }
        const pending_h = try a.alloc(f32, d.n_embd);
        errdefer a.free(pending_h);
        const target_h = try a.alloc(f32, @as(usize, mtp_max_verify) * d.n_embd);
        errdefer a.free(target_h);
        const pair_h = try a.alloc(f32, @as(usize, mtp_max_verify) * d.n_embd);
        errdefer a.free(pair_h);

        @memset(pending_h, 0);
        self.mtp = .{
            .recurrent_snapshot = recurrent,
            .conv_snapshot = conv,
            .conv_off_snapshot = conv_off,
            .verify_logits = verify_logits,
            .verify_argmax = verify_argmax,
            .h_input = h_input,
            .pending_h = pending_h,
            .target_h = target_h,
            .pair_h = pair_h,
        };
        return &self.mtp.?;
    }

    fn mtpHeadNorm(self: *ForwardCuda) ?*const LoadedTensor {
        const L = self.d.n_layers;
        return self.model.getLayer(L, "nextn.shared_head_norm.weight") orelse
            self.model.get("output_norm.weight");
    }

    fn mtpHead(self: *ForwardCuda) ?*const LoadedTensor {
        const L = self.d.n_layers;
        return self.model.getLayer(L, "nextn.shared_head_head.weight") orelse
            self.model.get("output.weight");
    }

    /// Allocate the speculative context before prompt timing begins. This is
    /// model/runtime setup (equivalent to creating llama.cpp's draft context),
    /// not prompt processing, and can involve hundreds of MiB of rollback rows.
    pub fn mtpPrepare(self: *ForwardCuda) !bool {
        if (!self.mtpEnabled()) return false;
        _ = try self.ensureMtpState();
        return true;
    }

    /// Save the target's pre-verification circular offsets. The speculative SSM
    /// kernels record the full recurrent/conv state after each candidate row,
    /// so rejection can select an exact accepted boundary without replay.
    fn mtpCheckpointTarget(self: *ForwardCuda) !void {
        self.waitPending();
        const state = &self.mtp.?;
        @memcpy(state.conv_off_snapshot, self.conv_off[0..self.d.n_layers]);
    }

    fn mtpRestoreTarget(self: *ForwardCuda, committed: u32) !void {
        self.waitPending();
        const state = &self.mtp.?;
        std.debug.assert(committed >= 1 and committed <= mtp_max_verify);

        var cmd = try command.beginCommand(self.ctx);
        for (0..self.d.n_layers) |li| {
            if (self.layer_is_attn[li]) continue;
            self.conv_off[li] = (state.conv_off_snapshot[li] + committed) % (self.d.d_conv - 1);
            const rec = CopyPush{ .N = self.d.ssm_state_len, .src_offset = (committed - 1) * self.d.ssm_state_len };
            cmd.dispatch(&self.pipes.copy_f32, .{ ceilDiv(rec.N, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &state.recurrent_snapshot[li], &self.ssm_state[li] }, &rec, @sizeOf(CopyPush), 0);
            const cv = CopyPush{ .N = self.d.conv_state_len, .src_offset = (committed - 1) * self.d.conv_state_len };
            cmd.dispatch(&self.pipes.copy_f32, .{ ceilDiv(cv.N, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &state.conv_snapshot[li], &self.ssm_conv_state[li] }, &cv, @sizeOf(CopyPush), 0);
        }
        cmd.commitAndWait();
    }

    /// Run the appended NextN block over already-committed target tokens. The
    /// shifted target hidden rows are supplied by the caller: token at p pairs
    /// with target h at p-1. This fills/repairs the draft head's own KV cache.
    fn mtpCatchup(self: *ForwardCuda, tokens: []const u32, h_pairs: []const f32, base_position: u32) !void {
        const T: u32 = @intCast(tokens.len);
        if (T == 0) return;
        const d = self.d;
        const rows: usize = @intCast(T);
        const width: usize = @intCast(d.n_embd);
        std.debug.assert(h_pairs.len == rows * width);

        const b = try self.ensureBatch(T);
        const embeddings = try self.allocator.alloc(f32, rows * width);
        defer self.allocator.free(embeddings);
        for (0..rows) |t| self.model.dequantEmbeddingRow(tokens[t], embeddings[t * width ..][0..width]);
        buffer.upload(self.ctx, &b.hidden, std.mem.sliceAsBytes(embeddings));
        buffer.upload(self.ctx, &b.o, std.mem.sliceAsBytes(h_pairs));

        const L = d.n_layers;
        const enorm = self.layer(L, "nextn.enorm.weight");
        const hnorm = self.layer(L, "nextn.hnorm.weight");
        const eh = self.layer(L, "nextn.eh_proj.weight");
        const wan = self.layer(L, "attn_norm.weight");
        const wk = self.layer(L, "attn_k.weight");
        const wv = self.layer(L, "attn_v.weight");
        const wkn = self.layer(L, "attn_k_norm.weight");
        const saved_spec_btok = self.use_spec_btok;
        const saved_cublas_q8 = self.use_cublas_q8;
        self.use_spec_btok = T >= 2;
        // The Q8_0 EH projection is the only large Q8 matrix here. ROCm's
        // native tile avoids a per-call fp16 dequant plus hipBLAS setup and is
        // dramatically faster for both the 48-row prime and 1–2-row repair.
        if (is_rocm) self.use_cublas_q8 = false;
        defer {
            self.use_spec_btok = saved_spec_btok;
            self.use_cublas_q8 = saved_cublas_q8;
        }

        var cmd = try command.beginCommand(self.ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &enorm.gpu_buffer, &b.norm }, &rms, @sizeOf(RmsPush), 0);
        cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.o, &hnorm.gpu_buffer, &b.ffn_norm }, &rms, @sizeOf(RmsPush), 0);
        const concat = ConcatRowsPush{ .N = d.n_embd, .T = T };
        cmd.dispatch(&self.pipes.concat_rows, .{ ceilDiv(T * d.n_embd, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &b.norm, &b.ffn_norm, &b.gate_ff }, &concat, @sizeOf(ConcatRowsPush), 0);
        self.gemmDispatchPrefill(&cmd, eh, &b.gate_ff, &b.hidden, d.n_embd, 2 * d.n_embd, T);

        // Catch-up persists only this block's K/V. Its attention result, output
        // projection and FFN are discarded because committed rows use target
        // h as the next cycle's cross-input. Skipping them removes roughly an
        // entire draft block of dead work per verification cycle.
        cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &wan.gpu_buffer, &b.norm }, &rms, @sizeOf(RmsPush), 0);
        self.gemmDispatchPrefill(&cmd, wk, &b.norm, &b.k, d.kv_dim, d.n_embd, T);
        self.gemmDispatchPrefill(&cmd, wv, &b.norm, &b.v, d.kv_dim, d.n_embd, T);
        const rms_h = RmsPush{ .N = d.head_dim, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ d.n_kv_head * T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.k, &wkn.gpu_buffer, &b.k }, &rms_h, @sizeOf(RmsPush), 0);
        const rope_k = RopeBatchPush{ .stride = d.head_dim, .rope_dim = d.rope_dim, .n_heads = d.n_kv_head, .base_position = base_position, .freq_base_bits = 0, .attn_scale_bits = 0 };
        cmd.dispatch(&self.pipes.rope_batched, .{ d.n_kv_head, T, 1 }, .{ 256, 1, 1 }, &.{ &b.k, &b.k, &self.inv_freq }, &rope_k, @sizeOf(RopeBatchPush), 0);
        const kvw = KvWriteBatchPush{ .kv_dim = d.kv_dim, .dst_base = base_position, .T = T };
        cmd.dispatch(&self.pipes.kv_cache_write_batched, .{ ceilDiv(d.kv_dim, 64), T, 1 }, .{ 64, 1, 1 }, &.{ &b.k, &self.kv_k[L], &b.v, &self.kv_v[L] }, &kvw, @sizeOf(KvWriteBatchPush), 0);
        self.submit(cmd);
        self.waitPending();
    }

    /// One autoregressive call to the single NextN head. h_input holds either
    /// the preceding target hidden row or the prior draft head's normalized row;
    /// the latter is copied back in-place for recursive draft steps.
    fn mtpDraftStep(self: *ForwardCuda, token: u32, pos: u32) !u32 {
        const d = self.d;
        const L = d.n_layers;
        const state = &self.mtp.?;
        const enorm = self.layer(L, "nextn.enorm.weight");
        const hnorm = self.layer(L, "nextn.hnorm.weight");
        const eh = self.layer(L, "nextn.eh_proj.weight");
        const head_norm = self.mtpHeadNorm() orelse return error.MissingTensor;
        const head = self.mtpHead() orelse return error.MissingTensor;

        self.model.dequantEmbeddingRow(token, self.host_embed);
        buffer.upload(self.ctx, &self.hidden, std.mem.sliceAsBytes(self.host_embed));

        var cmd = try command.beginCommand(self.ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &enorm.gpu_buffer, &self.ffn_norm_buf }, &rms, @sizeOf(RmsPush), 0);
        cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &state.h_input, &hnorm.gpu_buffer, &self.up_buf }, &rms, @sizeOf(RmsPush), 0);
        const concat = ConcatRowsPush{ .N = d.n_embd, .T = 1 };
        cmd.dispatch(&self.pipes.concat_rows, .{ ceilDiv(d.n_embd, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &self.ffn_norm_buf, &self.up_buf, &self.gate_buf }, &concat, @sizeOf(ConcatRowsPush), 0);
        self.dmmvDispatch(&cmd, eh, &self.gate_buf, &self.hidden, d.n_embd, 2 * d.n_embd, 0, 0);
        self.submit(cmd);

        try self.attentionLayer(L, pos);
        try self.ffnBlock(L);

        cmd = try command.beginCommand(self.ctx);
        const want_q8 = self.use_decode_q8_lm and head.info.type_ == .q6_k;
        const norm_q8_ready = self.rmsNormDecodeDispatch(&cmd, &self.hidden, &head_norm.gpu_buffer, &self.norm_buf, d.n_embd, want_q8);
        const save_h = CopyPush{ .N = d.n_embd };
        cmd.dispatch(&self.pipes.copy_f32, .{ ceilDiv(d.n_embd, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &self.norm_buf, &state.h_input }, &save_h, @sizeOf(CopyPush), 0);
        self.lmHeadDispatch(&cmd, &head.gpu_buffer, head.info.type_, norm_q8_ready);
        self.argmaxDispatch(&cmd, &self.argmax_buf);
        cmd.commitAndWait();
        self.drainPending();

        var tok: u32 = 0;
        buffer.download(self.ctx, &self.argmax_buf, std.mem.asBytes(&tok));
        return tok;
    }

    /// Run 1..4 proposed tokens through the real target in one weight-amortized
    /// batch. Every normalized final hidden row is retained for NextN catch-up;
    /// when `predictions` is non-null, each row's greedy next token is returned.
    fn mtpTargetBatch(self: *ForwardCuda, tokens: []const u32, base_position: u32, predictions: ?*[mtp_max_verify]u32) !void {
        const T: u32 = @intCast(tokens.len);
        std.debug.assert(T >= 1 and T <= mtp_max_verify);
        const d = self.d;
        const rows: usize = @intCast(T);
        const width: usize = @intCast(d.n_embd);
        const b = try self.ensureBatch(T);

        const embeddings = try self.allocator.alloc(f32, rows * width);
        defer self.allocator.free(embeddings);
        for (0..rows) |t| self.model.dequantEmbeddingRow(tokens[t], embeddings[t * width ..][0..width]);
        buffer.upload(self.ctx, &b.hidden, std.mem.sliceAsBytes(embeddings));

        const saved_spec_btok = self.use_spec_btok;
        const saved_capture = self.capture_spec_state;
        self.use_spec_btok = T >= 2;
        self.capture_spec_state = predictions != null;
        defer {
            self.use_spec_btok = saved_spec_btok;
            self.capture_spec_state = saved_capture;
        }

        var L: u32 = 0;
        while (L < d.n_layers) : (L += 1) {
            if (self.layer_is_attn[L]) {
                try self.attentionLayerBatched(L, T, b, base_position);
            } else {
                try self.ssmLayerBatched(L, T, b);
            }
            try self.ffnBlockBatched(L, T, b);
        }
        self.waitPending();

        const out_norm = self.model.get("output_norm.weight") orelse return error.MissingTensor;
        var cmd = try command.beginCommand(self.ctx);

        var logits_rows: [mtp_max_verify]CudaBuffer = undefined;
        var argmax_rows: [mtp_max_verify]CudaBuffer = undefined;
        var n_alias: usize = 0;
        defer {
            for (0..n_alias) |i| {
                buffer.freeBuffer(&logits_rows[i]);
                buffer.freeBuffer(&argmax_rows[i]);
            }
        }
        if (predictions != null) {
            const head = self.model.get("output.weight") orelse return error.MissingTensor;
            const head_uses_q8 = self.prefillUsesQ8(head, d.vocab, T);
            const norm_q8_ready = self.rmsNormPrefillDispatch(&cmd, &b.hidden, &out_norm.gpu_buffer, &b.norm, d.n_embd, T, head_uses_q8);
            self.gemmDispatchPrefillImpl(&cmd, head, &b.norm, &self.mtp.?.verify_logits, d.vocab, d.n_embd, T, norm_q8_ready);
            for (0..rows) |t| {
                logits_rows[t] = try buffer.aliasBuffer(&self.mtp.?.verify_logits, t * d.vocab * @sizeOf(f32), d.vocab * @sizeOf(f32));
                argmax_rows[t] = try buffer.aliasBuffer(&self.mtp.?.verify_argmax, t * @sizeOf(u32), @sizeOf(u32));
                n_alias += 1;
                self.argmaxDispatchFrom(&cmd, &logits_rows[t], &argmax_rows[t]);
            }
        } else {
            const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
            cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &out_norm.gpu_buffer, &b.norm }, &rms, @sizeOf(RmsPush), 0);
        }
        cmd.commitAndWait();

        buffer.download(self.ctx, &b.norm, std.mem.sliceAsBytes(self.mtp.?.target_h[0 .. rows * width]));
        if (predictions) |out| {
            buffer.download(self.ctx, &self.mtp.?.verify_argmax, std.mem.sliceAsBytes(out[0..rows]));
        }
    }

    /// Prime the draft head from the target prompt hidden rows left by
    /// prefillBatched. Initialization is charged to prefill, matching llama.cpp.
    pub fn mtpPrime(self: *ForwardCuda, prompt_tokens: []const u32) !bool {
        if (!self.mtpEnabled() or prompt_tokens.len == 0 or self.batch == null) return false;
        const d = self.d;
        const T: u32 = @intCast(prompt_tokens.len);
        const rows: usize = prompt_tokens.len;
        const width: usize = @intCast(d.n_embd);
        const b = &self.batch.?;
        if (b.t_cap < T) return false;
        _ = try self.ensureMtpState();

        const out_norm = self.model.get("output_norm.weight") orelse return error.MissingTensor;
        var cmd = try command.beginCommand(self.ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &out_norm.gpu_buffer, &b.norm }, &rms, @sizeOf(RmsPush), 0);
        cmd.commitAndWait();

        const target_h = try self.allocator.alloc(f32, rows * width);
        defer self.allocator.free(target_h);
        const pairs = try self.allocator.alloc(f32, rows * width);
        defer self.allocator.free(pairs);
        buffer.download(self.ctx, &b.norm, std.mem.sliceAsBytes(target_h));
        @memset(pairs, 0);
        if (rows > 1) @memcpy(pairs[width..], target_h[0 .. (rows - 1) * width]);
        @memcpy(self.mtp.?.pending_h, target_h[(rows - 1) * width ..][0..width]);

        try self.mtpCatchup(prompt_tokens, pairs, 0);
        self.mtp.?.primed = true;
        log.info("Qwen NextN/MTP enabled: {d} prompt rows primed, draft window={d}", .{ T, @min(mtp_max_draft, envU32("ZINC_MTP_DRAFTS", 1)) });
        return true;
    }

    pub fn mtpPerf(self: *const ForwardCuda) MtpPerf {
        const state = &self.mtp.?;
        return .{
            .cycles = state.cycles,
            .draft_ms = @as(f64, @floatFromInt(state.draft_ns)) / 1_000_000.0,
            .target_ms = @as(f64, @floatFromInt(state.target_ns)) / 1_000_000.0,
            .restore_ms = @as(f64, @floatFromInt(state.restore_ns)) / 1_000_000.0,
            .catchup_ms = @as(f64, @floatFromInt(state.catchup_ns)) / 1_000_000.0,
        };
    }

    /// Draft up to three tokens, verify them in one real-target batch, restore
    /// the recurrent state at the accepted boundary, then repair the NextN KV.
    /// The caller commits `seed` plus the returned accepted draft prefix only.
    pub fn mtpCycle(self: *ForwardCuda, seed: u32, pos: u32, max_drafts: u32, eos_id: u32) !MtpCycleResult {
        if (self.mtp == null or !self.mtp.?.primed) return error.MtpNotPrimed;
        const n_limit = @min(max_drafts, @min(mtp_max_draft, @max(@as(u32, 1), envU32("ZINC_MTP_DRAFTS", 1))));
        std.debug.assert(n_limit > 0);

        var phase_timer = try std.time.Timer.start();
        buffer.upload(self.ctx, &self.mtp.?.h_input, std.mem.sliceAsBytes(self.mtp.?.pending_h));
        var drafts: [mtp_max_draft]u32 = @splat(0);
        var n_drafted: u32 = 0;
        var fed = seed;
        while (n_drafted < n_limit) : (n_drafted += 1) {
            const draft = try self.mtpDraftStep(fed, pos + n_drafted);
            drafts[n_drafted] = draft;
            fed = draft;
            if (draft == eos_id) {
                n_drafted += 1;
                break;
            }
        }
        self.mtp.?.draft_ns += phase_timer.lap();

        var target_tokens: [mtp_max_verify]u32 = @splat(0);
        target_tokens[0] = seed;
        @memcpy(target_tokens[1 .. 1 + n_drafted], drafts[0..n_drafted]);
        try self.mtpCheckpointTarget();
        var predictions: [mtp_max_verify]u32 = @splat(0);
        try self.mtpTargetBatch(target_tokens[0 .. n_drafted + 1], pos, &predictions);
        self.mtp.?.target_ns += phase_timer.lap();

        var accepted: u32 = 0;
        while (accepted < n_drafted and predictions[accepted] == drafts[accepted]) : (accepted += 1) {}
        const committed = accepted + 1;
        if (accepted < n_drafted) {
            try self.mtpRestoreTarget(committed);
        }
        self.mtp.?.restore_ns += phase_timer.lap();

        const width: usize = @intCast(self.d.n_embd);
        @memcpy(self.mtp.?.pair_h[0..width], self.mtp.?.pending_h);
        var row: usize = 1;
        while (row < committed) : (row += 1) {
            @memcpy(self.mtp.?.pair_h[row * width ..][0..width], self.mtp.?.target_h[(row - 1) * width ..][0..width]);
        }
        @memcpy(self.mtp.?.pending_h, self.mtp.?.target_h[(committed - 1) * width ..][0..width]);
        try self.mtpCatchup(target_tokens[0..committed], self.mtp.?.pair_h[0 .. @as(usize, committed) * width], pos);
        self.mtp.?.catchup_ns += phase_timer.lap();

        self.mtp.?.cycles += 1;
        self.mtp.?.drafted += n_drafted;
        self.mtp.?.accepted += accepted;
        return .{
            .next_token = predictions[accepted],
            .n_drafted = n_drafted,
            .n_accepted = accepted,
            .drafts = drafts,
        };
    }

    /// Batched prefill that writes KV cache + SSM state into the given slot's
    /// per-sequence region (instead of the single-seq scratch). Used by the
    /// serve engine so server-mode prefill gets the 5× batched-GEMM speedup
    /// instead of falling back to per-token decodeBatch (B=1).
    pub fn prefillBatchedSlot(self: *ForwardCuda, tokens: []const u32, slot: u32) !u32 {
        const d = self.d;
        const f4 = @sizeOf(f32);
        const T: u32 = @intCast(tokens.len);
        if (self.kv_k_slots == null) return error.SlotStateNotAllocated;

        // Save single-seq state pointers + conv offsets.
        const saved_kv_k = self.kv_k;
        const saved_kv_v = self.kv_v;
        const saved_ssm = self.ssm_state;
        const saved_conv = self.ssm_conv_state;
        const saved_conv_off = try self.allocator.dupe(u32, self.conv_off);
        defer self.allocator.free(saved_conv_off);
        @memset(self.conv_off, 0);

        // Alias KV/SSM state to the slot's per-sequence region.
        var alias_err: ?anyerror = null;
        L_alias: for (0..d.n_layers) |li| {
            const L: u32 = @intCast(li);
            if (self.layer_is_attn[L]) {
                const kv_off = @as(usize, slot) * self.slot_ctx * d.kv_dim * f4;
                self.kv_k[L] = buffer.aliasBuffer(&self.kv_k_slots.?[L], kv_off, T * d.kv_dim * f4) catch |e| {
                    alias_err = e;
                    break :L_alias;
                };
                self.kv_v[L] = buffer.aliasBuffer(&self.kv_v_slots.?[L], kv_off, T * d.kv_dim * f4) catch |e| {
                    alias_err = e;
                    break :L_alias;
                };
            } else {
                self.ssm_state[L] = buffer.aliasBuffer(&self.ssm_state_slots.?[L], self.slotStateOffsetBytes(slot), d.ssm_state_len * f4) catch |e| {
                    alias_err = e;
                    break :L_alias;
                };
                self.ssm_conv_state[L] = buffer.aliasBuffer(&self.ssm_conv_slots.?[L], self.slotConvOffsetBytes(slot), d.conv_state_len * f4) catch |e| {
                    alias_err = e;
                    break :L_alias;
                };
            }
        }
        if (alias_err) |e| {
            self.restorePrefillAliases(saved_kv_k, saved_kv_v, saved_ssm, saved_conv, d.n_layers);
            @memcpy(self.conv_off, saved_conv_off);
            return e;
        }

        // Run batched prefill — writes land in the slot's aliased state.
        // The warp kernel's Q/K norm uses a different FP reduction tree than
        // the decode kernel → state divergence during decode. Force the block-
        // reduce kernel for slot-aware prefill to ensure bit-compatible state.
        self.force_block_ssm = true;
        const tok = self.prefillBatched(tokens) catch |err| {
            self.restorePrefillAliases(saved_kv_k, saved_kv_v, saved_ssm, saved_conv, d.n_layers);
            @memcpy(self.conv_off, saved_conv_off);
            return err;
        };

        self.restorePrefillAliases(saved_kv_k, saved_kv_v, saved_ssm, saved_conv, d.n_layers);
        self.force_block_ssm = false;
        @memcpy(self.conv_off, saved_conv_off);
        return tok;
    }

    fn restorePrefillAliases(self: *ForwardCuda, saved_kk: []CudaBuffer, saved_vv: []CudaBuffer, saved_ss: []CudaBuffer, saved_cc: []CudaBuffer, n_layers: u32) void {
        var i: u32 = 0;
        while (i < n_layers) : (i += 1) {
            // Only free the buffers that were actually aliased (per layer type).
            // Attention layers: kv_k/kv_v aliased; ssm_state/conv untouched.
            // SSM layers: ssm_state/conv aliased; kv_k/kv_v untouched.
            if (self.layer_is_attn[i]) {
                buffer.freeBuffer(&self.kv_k[i]);
                buffer.freeBuffer(&self.kv_v[i]);
            } else {
                buffer.freeBuffer(&self.ssm_state[i]);
                buffer.freeBuffer(&self.ssm_conv_state[i]);
            }
        }
        self.kv_k = saved_kk;
        self.kv_v = saved_vv;
        self.ssm_state = saved_ss;
        self.ssm_conv_state = saved_cc;
    }
    /// GEMM over all T tokens; the deinterleave-gate, per-head q/k norm, RoPE,
    /// KV-write, causal attention and sigmoid-gate are each ONE batched launch
    /// over all T tokens (Effort 26 T0), bit-identical to the old per-token loop
    /// (token t at sequence position t). qwen35-9b pp512 +32% vs per-token.
    fn attentionLayerBatched(self: *ForwardCuda, L: u32, T: u32, b: *BatchScratch, base_position: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const wq = self.layer(L, "attn_q.weight");
        const wk = self.layer(L, "attn_k.weight");
        const wv = self.layer(L, "attn_v.weight");
        const wqn = self.layer(L, "attn_q_norm.weight");
        const wkn = self.layer(L, "attn_k_norm.weight");
        const wo = self.layer(L, "attn_output.weight");
        const wan = self.layer(L, "attn_norm.weight");

        var cmd = try command.beginCommand(ctx);
        const wq_uses_q8 = self.prefillUsesQ8(wq, 2 * d.q_dim, T);
        const norm_q8_ready = self.rmsNormPrefillDispatch(&cmd, &b.hidden, &wan.gpu_buffer, &b.norm, d.n_embd, T, wq_uses_q8);
        self.gemmDispatchPrefillImpl(&cmd, wq, &b.norm, &b.qfull, 2 * d.q_dim, d.n_embd, T, norm_q8_ready);
        const reuse_q8_qk = prefillQ8ReuseOn() and
            wq_uses_q8 and self.prefillUsesQ8(wk, d.kv_dim, T);
        const use_spec_kv_pair = self.use_spec_btok and T == 2 and mtpQ8TypeOn(0) and reuse_q8_qk and
            wk.info.type_ == .q4_k and wv.info.type_ == .q4_k;
        if (use_spec_kv_pair) {
            self.dmmvQ4PairQ8Btok2(&cmd, &wk.gpu_buffer, &wv.gpu_buffer, &b.k, &b.v, d.kv_dim, d.kv_dim, d.n_embd);
        } else {
            self.gemmDispatchPrefillImpl(&cmd, wk, &b.norm, &b.k, d.kv_dim, d.n_embd, T, reuse_q8_qk);
            const reuse_q8_qkv = reuse_q8_qk and self.prefillUsesQ8(wv, d.kv_dim, T);
            self.gemmDispatchPrefillImpl(&cmd, wv, &b.norm, &b.v, d.kv_dim, d.n_embd, T, reuse_q8_qkv);
        }

        // Effort 26 T0: batched attention inner — each of the per-token ops below
        // is collapsed into ONE launch over all T tokens (grid.y = T, or grid.x
        // scaled by T for the per-head norms), bit-identical to the per-token loop
        // (token t at sequence position t). Kernels in a stream run in order, so
        // the batched KV write completes before the causal attention reads it.
        // Deinterleave the packed [Q|gate] projection for all T tokens.
        const deint = DeintBatchPush{ .head_dim = d.head_dim, .n_head = d.n_head, .T = T };
        cmd.dispatch(&self.pipes.deinterleave_batched, .{ ceilDiv(d.q_dim, 256), T, 1 }, .{ 256, 1, 1 }, &.{ &b.qfull, &b.q, &b.attn_gate }, &deint, @sizeOf(DeintBatchPush), 0);
        // Per-head q/k RMS norm: one block per (token, head) row of head_dim.
        const rms_h = RmsPush{ .N = d.head_dim, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ d.n_head * T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.q, &wqn.gpu_buffer, &b.q }, &rms_h, @sizeOf(RmsPush), 0);
        cmd.dispatch(&self.pipes.rms_norm, .{ d.n_kv_head * T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.k, &wkn.gpu_buffer, &b.k }, &rms_h, @sizeOf(RmsPush), 0);
        // RoPE q/k over all T at their absolute positions.
        const rope_q = RopeBatchPush{ .stride = d.head_dim, .rope_dim = d.rope_dim, .n_heads = d.n_head, .base_position = base_position, .freq_base_bits = 0, .attn_scale_bits = 0 };
        cmd.dispatch(&self.pipes.rope_batched, .{ d.n_head, T, 1 }, .{ 256, 1, 1 }, &.{ &b.q, &b.q, &self.inv_freq }, &rope_q, @sizeOf(RopeBatchPush), 0);
        const rope_k = RopeBatchPush{ .stride = d.head_dim, .rope_dim = d.rope_dim, .n_heads = d.n_kv_head, .base_position = base_position, .freq_base_bits = 0, .attn_scale_bits = 0 };
        cmd.dispatch(&self.pipes.rope_batched, .{ d.n_kv_head, T, 1 }, .{ 256, 1, 1 }, &.{ &b.k, &b.k, &self.inv_freq }, &rope_k, @sizeOf(RopeBatchPush), 0);
        // Write all T tokens' K/V into the cache at positions 0..T-1.
        const kvw = KvWriteBatchPush{ .kv_dim = d.kv_dim, .dst_base = base_position, .T = T };
        cmd.dispatch(&self.pipes.kv_cache_write_batched, .{ ceilDiv(d.kv_dim, 64), T, 1 }, .{ 64, 1, 1 }, &.{ &b.k, &self.kv_k[L], &b.v, &self.kv_v[L] }, &kvw, @sizeOf(KvWriteBatchPush), 0);
        // Causal attention for all T queries (block (head, t), seq_len = t+1).
        const attn = AttnBatchPush{ .head_dim = d.head_dim, .n_heads = d.n_head, .n_kv_heads = d.n_kv_head, .T = T, .attn_scale_bits = 0, .sink_offset = L * d.n_head, .base_position = base_position };
        if (self.use_attn_v2) {
            cmd.dispatch(&self.pipes.attention_causal_v2, .{ d.n_head, T, 1 }, .{ 256, 1, 1 }, &.{ &b.q, &self.kv_k[L], &self.kv_v[L], &self.sinks, &b.attn_out }, &attn, @sizeOf(AttnBatchPush), (base_position + T + d.head_dim) * 4);
        } else {
            cmd.dispatch(&self.pipes.attention_causal_batched, .{ d.n_head, T, 1 }, .{ 256, 1, 1 }, &.{ &b.q, &self.kv_k[L], &self.kv_v[L], &self.sinks, &b.attn_out }, &attn, @sizeOf(AttnBatchPush), (base_position + T) * 4);
        }
        // Sigmoid attention gate, element-wise over all [T, q_dim].
        const sm = SigmoidMulPush{ .N = T * d.q_dim };
        cmd.dispatch(&self.pipes.sigmoid_mul, .{ ceilDiv(T * d.q_dim, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.attn_out, &b.attn_gate, &b.attn_out }, &sm, @sizeOf(SigmoidMulPush), 0);
        // O projection → b.o, then fold into the residual stream.
        self.gemmDispatchPrefill(&cmd, wo, &b.attn_out, &b.o, d.n_embd, d.q_dim, T);
        const add = AddPush{ .N = T * d.n_embd };
        cmd.dispatch(&self.pipes.add_inplace, .{ ceilDiv(T * d.n_embd, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &b.o }, &add, @sizeOf(AddPush), 0);
        self.submit(cmd);
    }

    /// Token-major SSM (gated delta-net) block: pre-norm + qkv/z/alpha/beta + out
    /// projections via batched GEMM; conv1d in ONE batched launch (circular state
    /// advances internally); the delta-net scan in ONE launch (`n_tok=T`,
    /// recurrence preserved); gated-norm batched over T. Bit-identical to ssmLayer.
    fn ssmLayerBatched(self: *ForwardCuda, L: u32, T: u32, b: *BatchScratch) !void {
        const d = self.d;
        const ctx = self.ctx;
        const wan = self.layer(L, "attn_norm.weight");
        const wqkv = self.layer(L, "attn_qkv.weight");
        const wz = self.layer(L, "attn_gate.weight");
        const walpha = self.layer(L, "ssm_alpha.weight");
        const wbeta = self.layer(L, "ssm_beta.weight");
        const wconv = self.layer(L, "ssm_conv1d.weight");
        const wdt = self.layer(L, "ssm_dt.bias");
        const wa = self.layer(L, "ssm_a");
        const wnorm = self.layer(L, "ssm_norm.weight");
        const wout = self.layer(L, "ssm_out.weight");

        const ssm_profile = std.posix.getenv("ZINC_SSM_PROFILE") != null;
        var prof_pre: i64 = 0;
        var prof_post: i64 = 0;

        // When NOT profiling, use a single command buffer (original behavior).
        // When profiling, split into pre-scan / scan / post-scan for timing.
        var cmd = try command.beginCommand(ctx);

        // === Pre-scan: RMS norm + GEMMs + conv1d ===
        const wqkv_uses_q8 = self.prefillUsesQ8(wqkv, d.conv_channels, T);
        const norm_q8_ready = self.rmsNormPrefillDispatch(&cmd, &b.hidden, &wan.gpu_buffer, &b.norm, d.n_embd, T, wqkv_uses_q8);
        const reuse_q8_qkv_z = prefillQ8ReuseOn() and
            wqkv_uses_q8 and self.prefillUsesQ8(wz, d.d_inner, T);
        const use_spec_qkv_z_pair = self.use_spec_btok and T == 2 and mtpQ8TypeOn(0) and reuse_q8_qkv_z and
            wqkv.info.type_ == .q4_k and wz.info.type_ == .q4_k;
        if (use_spec_qkv_z_pair) {
            self.dmmvQ4PairQ8Btok2(&cmd, &wqkv.gpu_buffer, &wz.gpu_buffer, &b.qkv, &b.z, d.conv_channels, d.d_inner, d.n_embd);
        } else {
            self.gemmDispatchPrefillImpl(&cmd, wqkv, &b.norm, &b.qkv, d.conv_channels, d.n_embd, T, norm_q8_ready);
            self.gemmDispatchPrefillImpl(&cmd, wz, &b.norm, &b.z, d.d_inner, d.n_embd, T, reuse_q8_qkv_z);
        }
        self.gemmDispatchPrefill(&cmd, walpha, &b.norm, &b.alpha, d.dt_rank, d.n_embd, T);
        self.gemmDispatchPrefill(&cmd, wbeta, &b.norm, &b.beta, d.dt_rank, d.n_embd, T);
        const conv = ConvBatchPush{ .conv_channels = d.conv_channels, .d_conv = d.d_conv, .kernel_is_f16 = boolU32(wconv.info.type_ == .f16), .n_tok = T, .state_offset = self.conv_off[L] };
        if (self.capture_spec_state) {
            cmd.dispatch(&self.pipes.ssm_conv1d_batched_history, .{ ceilDiv(d.conv_channels, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.qkv, &wconv.gpu_buffer, &self.ssm_conv_state[L], &b.conv_out, &self.mtp.?.conv_snapshot[L] }, &conv, @sizeOf(ConvBatchPush), 0);
        } else {
            cmd.dispatch(&self.pipes.ssm_conv1d_batched, .{ ceilDiv(d.conv_channels, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.qkv, &wconv.gpu_buffer, &self.ssm_conv_state[L], &b.conv_out }, &conv, @sizeOf(ConvBatchPush), 0);
        }
        self.conv_off[L] = (self.conv_off[L] + T) % (d.d_conv - 1);

        if (ssm_profile) {
            const t0 = std.time.milliTimestamp();
            self.submit(cmd);
            self.waitPending();
            prof_pre = std.time.milliTimestamp() - t0;
        }

        // === Delta-net scan ===
        if (ssm_profile) cmd = try command.beginCommand(ctx);
        const use_chunked = !self.capture_spec_state and std.posix.getenv("ZINC_SSM_CHUNKED") != null and
            std.mem.eql(u8, std.posix.getenv("ZINC_SSM_CHUNKED").?, "1");
        // Delta-net scan: warp-level kernel (no __syncthreads in hot loop).
        // FIXED (2026-06-30): warp kernel now computes S@k (row-wise) matching
        // the block kernel. Safe for all T values.
        // A/B toggle: ZINC_SSM_WARP=0 falls back to the block-reduce kernel.
        const use_warp = (self.capture_spec_state and !is_rocm) or
            (!self.capture_spec_state and !self.force_block_ssm and !use_chunked and
                (std.posix.getenv("ZINC_SSM_WARP") == null or
                    !std.mem.eql(u8, std.posix.getenv("ZINC_SSM_WARP").?, "0")));
        const use_preprocessed = !use_chunked and !use_warp and d.d_state == d.head_v_dim and d.d_state <= 128 and ssmPreparedOn();
        const use_col_warp = use_preprocessed and ssmColWarpOn();
        if (ssm_profile and L == 0) log.info("SSM_PROFILE L0: prepared={}", .{use_preprocessed});
        if (use_preprocessed) {
            const prep = DeltaNetPreparePush{
                .dt_rank = d.dt_rank,
                .d_state = d.d_state,
                .n_group = d.n_group,
                .ssm_a_is_f16 = boolU32(wa.info.type_ == .f16),
                .dt_bias_is_f16 = boolU32(wdt.info.type_ == .f16),
                .has_dt_bias = 1,
                .has_ssm_a = 1,
                .n_tok = T,
                .conv_stride_tok = d.conv_channels,
                .ab_stride_tok = d.dt_rank,
            };
            cmd.dispatch(&self.pipes.ssm_delta_net_prepare, .{ d.n_group, T, 1 }, .{ d.d_state, 1, 1 }, &.{ &b.conv_out, &wdt.gpu_buffer, &b.alpha, &b.beta, &wa.gpu_buffer }, &prep, @sizeOf(DeltaNetPreparePush), 0);
        }
        var prof_scan: i64 = 0;
        if (use_chunked) {
            const dn_ch = DeltaNetChunkedPush{
                .dt_rank = d.dt_rank,
                .head_v_dim = d.head_v_dim,
                .d_state = d.d_state,
                .n_group = d.n_group,
                .ssm_a_is_f16 = boolU32(wa.info.type_ == .f16),
                .dt_bias_is_f16 = boolU32(wdt.info.type_ == .f16),
                .has_dt_bias = 1,
                .has_ssm_a = 1,
                .n_tok = T,
                .conv_stride_tok = d.conv_channels,
                .ab_stride_tok = d.dt_rank,
                .y_stride_tok = d.d_inner,
            };
            const smem_size = 64 * d.head_v_dim * @sizeOf(f32) + 64 * 64 * @sizeOf(f32) + 2 * 64 * @sizeOf(f32);
            cmd.dispatch(&self.pipes.ssm_delta_net_chunked, .{ d.dt_rank, ceilDiv(d.head_v_dim, 4), 1 }, .{ 32, 4, 1 }, &.{ &b.conv_out, &wdt.gpu_buffer, &b.alpha, &b.beta, &wa.gpu_buffer, &self.ssm_state[L], &b.delta_out }, &dn_ch, @sizeOf(DeltaNetChunkedPush), smem_size);
        } else if (use_warp) {
            const dn_warp = DeltaNetWarpPush{
                .dt_rank = d.dt_rank,
                .head_v_dim = d.head_v_dim,
                .d_state = d.d_state,
                .n_group = d.n_group,
                .ssm_a_is_f16 = boolU32(wa.info.type_ == .f16),
                .dt_bias_is_f16 = boolU32(wdt.info.type_ == .f16),
                .has_dt_bias = 1,
                .has_ssm_a = 1,
                .n_tok = T,
                .conv_stride_tok = d.conv_channels,
                .ab_stride_tok = d.dt_rank,
                .y_stride_tok = d.d_inner,
            };
            if (self.capture_spec_state) {
                cmd.dispatch(&self.pipes.ssm_delta_net_warp_history, .{ d.dt_rank, ceilDiv(d.head_v_dim, 4), 1 }, .{ 32, 4, 1 }, &.{ &b.conv_out, &wdt.gpu_buffer, &b.alpha, &b.beta, &wa.gpu_buffer, &self.ssm_state[L], &b.delta_out, &self.mtp.?.recurrent_snapshot[L] }, &dn_warp, @sizeOf(DeltaNetWarpPush), 2 * T * @sizeOf(f32));
            } else {
                cmd.dispatch(&self.pipes.ssm_delta_net_warp, .{ d.dt_rank, ceilDiv(d.head_v_dim, 4), 1 }, .{ 32, 4, 1 }, &.{ &b.conv_out, &wdt.gpu_buffer, &b.alpha, &b.beta, &wa.gpu_buffer, &self.ssm_state[L], &b.delta_out }, &dn_warp, @sizeOf(DeltaNetWarpPush), 2 * T * @sizeOf(f32));
            }
        } else if (use_col_warp) {
            const dn_col = DeltaNetColWarpPush{
                .dt_rank = d.dt_rank,
                .head_v_dim = d.head_v_dim,
                .d_state = d.d_state,
                .n_group = d.n_group,
                .n_tok = T,
                .conv_stride_tok = d.conv_channels,
                .ab_stride_tok = d.dt_rank,
                .y_stride_tok = d.d_inner,
                .fast_reduce = boolU32(if (self.capture_spec_state) self.use_decode_ssm_fast else ssmColWarpFastOn()),
            };
            if (self.capture_spec_state) {
                cmd.dispatch(&self.pipes.ssm_delta_net_col_warp_history, .{ d.dt_rank, ceilDiv(d.head_v_dim, 4), 1 }, .{ 32, 4, 1 }, &.{ &b.conv_out, &b.alpha, &b.beta, &self.ssm_state[L], &b.delta_out, &self.mtp.?.recurrent_snapshot[L] }, &dn_col, @sizeOf(DeltaNetColWarpPush), 0);
            } else {
                cmd.dispatch(&self.pipes.ssm_delta_net_col_warp, .{ d.dt_rank, ceilDiv(d.head_v_dim, 4), 1 }, .{ 32, 4, 1 }, &.{ &b.conv_out, &b.alpha, &b.beta, &self.ssm_state[L], &b.delta_out }, &dn_col, @sizeOf(DeltaNetColWarpPush), 0);
            }
        } else {
            const dn = DeltaNetPush{
                .d_inner = d.d_inner,
                .dt_rank = d.dt_rank,
                .head_v_dim = d.head_v_dim,
                .d_state = d.d_state,
                .n_group = d.n_group,
                .ssm_a_is_f16 = boolU32(wa.info.type_ == .f16),
                .dt_bias_is_f16 = boolU32(wdt.info.type_ == .f16),
                .has_dt_bias = 1,
                .has_ssm_a = 1,
                .n_tok = T,
                .conv_stride_tok = d.conv_channels,
                .ab_stride_tok = d.dt_rank,
                .y_stride_tok = d.d_inner,
                .preprocessed = boolU32(use_preprocessed),
            };
            cmd.dispatch(&self.pipes.ssm_delta_net, .{ d.dt_rank, d.head_v_dim, 1 }, .{ d.head_v_dim, 1, 1 }, &.{ &b.conv_out, &wdt.gpu_buffer, &b.alpha, &b.beta, &wa.gpu_buffer, &self.ssm_state[L], &b.delta_out }, &dn, @sizeOf(DeltaNetPush), 0);
        }

        if (ssm_profile) {
            const t0 = std.time.milliTimestamp();
            self.submit(cmd);
            self.waitPending();
            prof_scan = std.time.milliTimestamp() - t0;
            cmd = try command.beginCommand(ctx);
        }

        // === Post-scan: gated norm + output GEMM + residual ===
        const norm_per_head: u32 = if (wnorm.info.numElements() == d.d_inner) 1 else 0;
        const gn = GatedNormBatchPush{ .d_inner = d.d_inner, .dt_rank = d.dt_rank, .head_v_dim = d.head_v_dim, .d_state = d.d_state, .norm_per_head = norm_per_head, .n_tok = T };
        cmd.dispatch(&self.pipes.ssm_gated_norm_batched, .{ d.dt_rank, T, 1 }, .{ d.head_v_dim, 1, 1 }, &.{ &b.delta_out, &b.z, &wnorm.gpu_buffer, &b.ssm_gn }, &gn, @sizeOf(GatedNormBatchPush), 0);
        self.gemmDispatchPrefill(&cmd, wout, &b.ssm_gn, &b.o, d.n_embd, d.d_inner, T);
        const add = AddPush{ .N = T * d.n_embd };
        cmd.dispatch(&self.pipes.add_inplace, .{ ceilDiv(T * d.n_embd, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &b.o }, &add, @sizeOf(AddPush), 0);

        if (ssm_profile) {
            const t0 = std.time.milliTimestamp();
            self.submit(cmd);
            self.waitPending();
            prof_post = std.time.milliTimestamp() - t0;
            if (L == 0)
                log.info("SSM_PROFILE L0: pre={d}ms scan={d}ms post={d}ms (T={d})", .{ prof_pre, prof_scan, prof_post, T });
        } else {
            self.submit(cmd);
        }
    }

    /// Token-major dense SwiGLU FFN: pre-norm + gate/up/down via batched GEMM,
    /// element-wise SwiGLU over [T, n_ff], fold down into the residual stream.
    fn ffnBlockBatched(self: *ForwardCuda, L: u32, T: u32, b: *BatchScratch) !void {
        const d = self.d;
        const ctx = self.ctx;
        const wfn = self.layer(L, "post_attention_norm.weight");
        const wgate = self.layer(L, "ffn_gate.weight");
        const wup = self.layer(L, "ffn_up.weight");
        const wdown = self.layer(L, "ffn_down.weight");

        // Fused gate+up+SwiGLU when both are Q4_K and the tile is big enough
        const fuse_gate_up_swiglu = prefillGateUpSwigluOn();
        const dense_gate_up_uses_cublas = self.use_cublas and T >= self.cublas_min_t;
        const can_fuse_gate_up = wgate.info.type_ == .q4_k and wup.info.type_ == .q4_k and
            T >= 32 and d.n_ff >= 64 and !dense_gate_up_uses_cublas and fuse_gate_up_swiglu;
        const can_fuse_spec_gate_up = is_rocm and self.use_spec_btok and T == 2 and mtpQ8TypeOn(0) and
            wgate.info.type_ == .q4_k and wup.info.type_ == .q4_k;

        var cmd = try command.beginCommand(ctx);
        const wgate_uses_q8 = can_fuse_spec_gate_up or (!can_fuse_gate_up and self.prefillUsesQ8(wgate, d.n_ff, T));
        const norm_q8_ready = self.rmsNormPrefillDispatch(&cmd, &b.hidden, &wfn.gpu_buffer, &b.ffn_norm, d.n_embd, T, wgate_uses_q8);
        if (can_fuse_spec_gate_up) {
            if (!norm_q8_ready) {
                const qp = QuantActPush{ .K = d.n_embd, .T = T };
                cmd.dispatch(&self.pipes.quantize_act_q8, .{ ceilDiv(d.n_embd, 256), T, 1 }, .{ 256, 1, 1 }, &.{ &b.ffn_norm, &b.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
            }
            const gp = DmmvPush{ .M = d.n_ff, .K = d.n_embd };
            cmd.dispatch(&self.pipes.dmmv_q4k_gate_up_swiglu_q8_btok2, .{ d.n_ff, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ &wgate.gpu_buffer, &wup.gpu_buffer, &b.act_q8, &b.swiglu_ff }, &gp, @sizeOf(DmmvPush), 0);
        } else if (can_fuse_gate_up) {
            const gp = GateUpSwigluPush{ .M = d.n_ff, .K = d.n_embd, .T = T };
            cmd.dispatch(&self.pipes.gemm_q4k_gate_up_swiglu, .{ ceilDiv(d.n_ff, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &wgate.gpu_buffer, &wup.gpu_buffer, &b.ffn_norm, &b.swiglu_ff }, &gp, @sizeOf(GateUpSwigluPush), 0);
        } else {
            self.gemmDispatchPrefillImpl(&cmd, wgate, &b.ffn_norm, &b.gate_ff, d.n_ff, d.n_embd, T, norm_q8_ready);
            const reuse_q8_gate_up = prefillQ8ReuseOn() and
                wgate_uses_q8 and self.prefillUsesQ8(wup, d.n_ff, T);
            self.gemmDispatchPrefillImpl(&cmd, wup, &b.ffn_norm, &b.up_ff, d.n_ff, d.n_embd, T, reuse_q8_gate_up);
            const sg = SwigluPush{ .N = T * d.n_ff };
            cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(T * d.n_ff, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.gate_ff, &b.up_ff, &b.swiglu_ff }, &sg, @sizeOf(SwigluPush), 0);
        }
        self.gemmDispatchPrefill(&cmd, wdown, &b.swiglu_ff, &b.o, d.n_embd, d.n_ff, T);
        const add = AddPush{ .N = T * d.n_embd };
        cmd.dispatch(&self.pipes.add_inplace, .{ ceilDiv(T * d.n_embd, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &b.o }, &add, @sizeOf(AddPush), 0);
        self.submit(cmd);
    }

    /// Map a stacked-expert quant to its token-batched (grid.y = T) experts kernel.
    fn expertsPipeBatched(self: *ForwardCuda, t: gguf.GGMLType) ?*CudaPipeline {
        return switch (t) {
            .q4_k => &self.pipes.dmmv_q4k_experts_batched,
            .q5_k => &self.pipes.dmmv_q5k_experts_batched,
            .q6_k => &self.pipes.dmmv_q6k_experts_batched,
            else => null,
        };
    }

    /// True when layer L's MoE block can run the token-batched prefill path: every
    /// routed expert tensor has a batched kernel and the router / shared gate-scalar
    /// weights are F32 (handled by dmmv_f32_batched). qwen36-35b-a3b is all-supported
    /// (Q4_K/Q5_K/Q6_K experts, F32 router); a future unsupported quant falls back.
    fn moeBatchedSupported(self: *ForwardCuda, L: u32) bool {
        const wge = self.layer(L, "ffn_gate_exps.weight");
        const wue = self.layer(L, "ffn_up_exps.weight");
        const wde = self.layer(L, "ffn_down_exps.weight");
        const wrouter = self.layer(L, "ffn_gate_inp.weight");
        const wgi = self.layer(L, "ffn_gate_inp_shexp.weight");
        return self.expertsPipeBatched(wge.info.type_) != null and
            self.expertsPipeBatched(wue.info.type_) != null and
            self.expertsPipeBatched(wde.info.type_) != null and
            wrouter.info.type_ == .f32 and wgi.info.type_ == .f32;
    }

    /// Effort 29 T2: token-batched qwen2-MoE FFN over ALL T prompt tokens — the
    /// batched twin of the per-token `moeFfnBlock` loop that qwen36-35b-a3b prefill
    /// used to run (router + 8 routed experts + shared expert ≈ 14 launches × T ×
    /// 40 layers → launch-bound, pp256 ~50 t/s). Each heavy step is ONE launch over
    /// all T tokens (grid.y = T), reading token-major buffers; every per-(token)
    /// kernel is byte-for-byte the single-token kernel's, so the result is
    /// bit-identical to the per-token loop. Caller-gated by `moeBatchedSupported(L)`.
    fn moeFfnBlockBatched(self: *ForwardCuda, L: u32, T: u32, b: *BatchScratch) !void {
        const d = self.d;
        const ctx = self.ctx;
        const n_used = d.n_experts_used;
        const ef = d.n_ff; // routed-expert intermediate (512)
        const sf = d.shexp_ff; // shared-expert intermediate (512)
        const rt_stride = 2 * n_used;
        const wfn = self.layer(L, "post_attention_norm.weight");
        const wrouter = self.layer(L, "ffn_gate_inp.weight"); // [n_embd, n_experts] F32
        const wge = self.layer(L, "ffn_gate_exps.weight");
        const wue = self.layer(L, "ffn_up_exps.weight");
        const wde = self.layer(L, "ffn_down_exps.weight");
        const gate_slice = expertSliceBytes(wge.info.type_, ef, d.n_embd);
        const up_slice = expertSliceBytes(wue.info.type_, ef, d.n_embd);
        const down_slice = expertSliceBytes(wde.info.type_, d.n_embd, ef);
        const gate_pipe = self.expertsPipeBatched(wge.info.type_).?;
        const up_pipe = self.expertsPipeBatched(wue.info.type_).?;
        const down_pipe = self.expertsPipeBatched(wde.info.type_).?;

        var cmd = try command.beginCommand(ctx);
        // --- Router: batched rms_norm → F32 logits → per-token top-k softmax. -----
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.hidden, &wfn.gpu_buffer, &b.ffn_norm }, &rms, @sizeOf(RmsPush), 0);
        const rl = MatvecBatchPush{ .M = d.n_experts, .K = d.n_embd, .x_tok_stride = d.n_embd, .y_tok_stride = d.n_experts };
        cmd.dispatch(&self.pipes.dmmv_f32_batched, .{ d.n_experts, T, 1 }, .{ 256, 1, 1 }, &.{ &wrouter.gpu_buffer, &b.ffn_norm, &b.router_logits_e }, &rl, @sizeOf(MatvecBatchPush), 0);
        const tk = TopkPush{ .n_experts = d.n_experts, .k = n_used };
        cmd.dispatch(&self.pipes.softmax_topk_batched, .{ T, 1, 1 }, .{ 64, 1, 1 }, &.{ &b.router_logits_e, &b.router_table_e }, &tk, @sizeOf(TopkPush), 0);

        // --- Routed experts: gate/up → SwiGLU → down, slot-major per token. -------
        // T2 port: when ZINC_MOE_TC is on (T-gated) and both gate/up are Q4_K, run a
        // single-launch padded grouped Tensor-core GEMM each instead of the per-token
        // matvec — the matvec re-reads every routed token's expert weight (≈T·n_used/
        // n_experts× redundant), the grouped TC reads each expert weight ~once/tile.
        // The down + any non-Q4_K layer (the model's 1 Q5_K gate/up layer) stay on the
        // matvec. fp16 → token-tolerance, not bit-identical.
        const use_tc = self.use_tc_experts and (self.tc_experts_forced or T >= self.moe_tc_min_t) and
            wge.info.type_ == .q4_k and wue.info.type_ == .q4_k;
        if (std.posix.getenv("ZINC_PREFILL_PROFILE") != null and L < 2) {
            log.info("MoE layer {d}: gate={} up={} grouped={}", .{ L, wge.info.type_, wue.info.type_, use_tc });
        }
        if (use_tc) {
            const P = n_used * T;
            // gfx12 integer WMMA consumes native 16-route fragments. Avoid padding
            // every short-prompt expert bucket to the CUDA path's 64-route tile.
            // With 256 experts and eight routes/token, even a 322-token prompt
            // averages only ten rows per expert. Keep the native 16-route tile
            // through this sparse range; switching to 64 at T=256 padded most
            // long-context buckets with zeros and halved useful throughput.
            const rocm_route16 = is_rocm and T < 512;
            const route_tile: u32 = if (rocm_route16) 16 else 64;
            const max_pos = P + route_tile * d.n_experts;
            const max_tiles = max_pos / route_tile;
            // Padded counting sort of the (token,slot) work-items by expert, each run
            // padded to one route tile with a per-tile expert id — GPU-side, no readback.
            const bo = BuildOrderPadPush{ .T = T, .n_used = n_used, .n_experts = d.n_experts, .routing_stride = rt_stride, .max_pos = max_pos };
            const order_pipe = if (route_tile == 16) &self.pipes.build_expert_order_padded_16 else &self.pipes.build_expert_order_padded;
            cmd.dispatch(order_pipe, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &b.router_table_e, &b.padded_order, &b.tile_expert }, &bo, @sizeOf(BuildOrderPadPush), 0);
            if (comptime is_rocm) {
                const qp = QuantActPush{ .K = d.n_embd, .T = T };
                cmd.dispatch(&self.pipes.quantize_act_q8, .{ ceilDiv(d.n_embd, 256), T, 1 }, .{ 256, 1, 1 }, &.{ &b.ffn_norm, &b.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
            }
            // gate (wge) + up (wue) are SEPARATE tensors (gemma fuses them) → per-expert
            // stride = the whole slice, base 0. A = the per-token ffn_norm row.
            if (comptime is_rocm) {
                const use_m32 = route_tile == 16 and std.posix.getenv("ZINC_MOE_M32") != null;
                const use_m64 = route_tile == 16 and !use_m32 and std.posix.getenv("ZINC_MOE_M64") != null;
                const use_direct = route_tile == 16 and !use_m32 and !use_m64 and envFlag("ZINC_MOE_DIRECT_A", is_rocm);
                const grouped_pipe = if (use_m32)
                    &self.pipes.gemm_q4k_experts_grouped_i8_t16_m32
                else if (use_m64)
                    &self.pipes.gemm_q4k_experts_grouped_i8_t16_m64
                else if (use_direct)
                    &self.pipes.gemm_q4k_experts_grouped_i8_t16_direct
                else if (envFlag("ZINC_MOE_DIRECT_A", is_rocm))
                    &self.pipes.gemm_q4k_experts_grouped_i8_direct
                else if (route_tile == 16)
                    &self.pipes.gemm_q4k_experts_grouped_i8_t16
                else
                    &self.pipes.gemm_q4k_experts_grouped_i8;
                const grouped_rows: u32 = if (use_m32) 32 else if (use_m64) 64 else 128;
                const grouped_threads: u32 = if (use_m32) 64 else if (use_m64) 128 else 256;
                const pg = GroupedI8Push{ .M = ef, .K = d.n_embd, .T = T, .base = 0, .expert_stride = gate_slice, .dst_tok_stride = n_used * ef, .route_stride = 0 };
                cmd.dispatch(grouped_pipe, .{ ceilDiv(ef, grouped_rows), max_tiles, 1 }, .{ grouped_threads, 1, 1 }, &.{ &wge.gpu_buffer, &b.act_q8, &b.padded_order, &b.tile_expert, &b.gate_e }, &pg, @sizeOf(GroupedI8Push), 0);
                const pu = GroupedI8Push{ .M = ef, .K = d.n_embd, .T = T, .base = 0, .expert_stride = up_slice, .dst_tok_stride = n_used * ef, .route_stride = 0 };
                cmd.dispatch(grouped_pipe, .{ ceilDiv(ef, grouped_rows), max_tiles, 1 }, .{ grouped_threads, 1, 1 }, &.{ &wue.gpu_buffer, &b.act_q8, &b.padded_order, &b.tile_expert, &b.up_e }, &pu, @sizeOf(GroupedI8Push), 0);
            } else {
                const pg = GroupedTCPush{ .M = ef, .K = d.n_embd, .base = 0, .gu_full = gate_slice, .dst_tok_stride = n_used * ef };
                cmd.dispatch(&self.pipes.gemm_q4k_experts_grouped_tc, .{ ceilDiv(ef, 64), max_tiles, 1 }, .{ 256, 1, 1 }, &.{ &wge.gpu_buffer, &b.ffn_norm, &b.padded_order, &b.tile_expert, &b.gate_e }, &pg, @sizeOf(GroupedTCPush), 0);
                const pu = GroupedTCPush{ .M = ef, .K = d.n_embd, .base = 0, .gu_full = up_slice, .dst_tok_stride = n_used * ef };
                cmd.dispatch(&self.pipes.gemm_q4k_experts_grouped_tc, .{ ceilDiv(ef, 64), max_tiles, 1 }, .{ 256, 1, 1 }, &.{ &wue.gpu_buffer, &b.ffn_norm, &b.padded_order, &b.tile_expert, &b.up_e }, &pu, @sizeOf(GroupedTCPush), 0);
            }
        } else {
            const pg = ExpertsBatchPush{ .M = ef, .K = d.n_embd, .slice = gate_slice, .x_stride = 0, .n_used = n_used, .base = 0, .routing_stride = rt_stride, .x_tok_stride = d.n_embd, .y_tok_stride = n_used * ef };
            cmd.dispatch(gate_pipe, .{ n_used * ef, T, 1 }, .{ 64, 1, 1 }, &.{ &wge.gpu_buffer, &b.ffn_norm, &b.gate_e, &b.router_table_e }, &pg, @sizeOf(ExpertsBatchPush), 0);
            const pu = ExpertsBatchPush{ .M = ef, .K = d.n_embd, .slice = up_slice, .x_stride = 0, .n_used = n_used, .base = 0, .routing_stride = rt_stride, .x_tok_stride = d.n_embd, .y_tok_stride = n_used * ef };
            cmd.dispatch(up_pipe, .{ n_used * ef, T, 1 }, .{ 64, 1, 1 }, &.{ &wue.gpu_buffer, &b.ffn_norm, &b.up_e, &b.router_table_e }, &pu, @sizeOf(ExpertsBatchPush), 0);
        }
        const sg = SwigluPush{ .N = T * n_used * ef };
        cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(T * n_used * ef, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.gate_e, &b.up_e, &b.swiglu_e }, &sg, @sizeOf(SwigluPush), 0);
        // Down projection: when the gate/up grouped-TC path ran (use_tc), put down on
        // TC too — reuse the SAME padded order/tile_expert (same routing) with a
        // grouped-TC GEMM whose A is the per-work-item SwiGLU output. Down is 1/3 of
        // expert FLOPs and the matvec re-reads each expert's whole down weight per
        // routed token. qwen36-35b-a3b down = Q5_K ×36 + Q6_K ×4 — a per-quant kernel
        // each (the Q6_K matvec is the heaviest, 6-bit weight traffic). Any other
        // quant falls back to the matvec.
        const down_tc_pipe: ?*CudaPipeline = if (!is_rocm and use_tc and moeDownTcOn()) switch (wde.info.type_) {
            .q5_k => &self.pipes.gemm_q5k_experts_grouped_tc,
            .q6_k => if (moeDownQ6kTcOn()) &self.pipes.gemm_q6k_experts_grouped_tc else null, // the 4 Q6_K layers
            else => null,
        } else null;
        const rocm_route16 = is_rocm and T < 512;
        const rocm_down_m32 = rocm_route16 and std.posix.getenv("ZINC_MOE_M32") != null;
        const rocm_down_pipe: ?*CudaPipeline = if (is_rocm and use_tc and moeDownTcOn()) switch (wde.info.type_) {
            .q5_k => if (rocm_down_m32)
                &self.pipes.gemm_q5k_experts_grouped_i8_t16_m32
            else if (rocm_route16)
                &self.pipes.gemm_q5k_experts_grouped_i8_t16
            else
                &self.pipes.gemm_q5k_experts_grouped_i8,
            .q6_k => if (moeDownQ6kTcOn())
                if (rocm_down_m32)
                    &self.pipes.gemm_q6k_experts_grouped_i8_t16_m32
                else if (rocm_route16)
                    &self.pipes.gemm_q6k_experts_grouped_i8_t16
                else
                    &self.pipes.gemm_q6k_experts_grouped_i8
            else
                null,
            else => null,
        } else null;
        if (rocm_down_pipe) |dpipe| {
            const Pd = n_used * T;
            const route_tile: u32 = if (rocm_route16) 16 else 64;
            const max_pos_d = Pd + route_tile * d.n_experts;
            const max_tiles_d = max_pos_d / route_tile;
            // Reuse act_q8 after gate/up: each routed SwiGLU row is now its own
            // activation row, addressed through (token,slot) in padded_order.
            const qp = QuantActPush{ .K = ef, .T = Pd };
            cmd.dispatch(&self.pipes.quantize_act_q8, .{ ceilDiv(ef, 256), Pd, 1 }, .{ 256, 1, 1 }, &.{ &b.swiglu_e, &b.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
            const pd = GroupedI8Push{ .M = d.n_embd, .K = ef, .T = Pd, .base = 0, .expert_stride = down_slice, .dst_tok_stride = n_used * d.n_embd, .route_stride = n_used };
            const down_rows: u32 = if (rocm_down_m32) 32 else 128;
            const down_threads: u32 = if (rocm_down_m32) 64 else 256;
            cmd.dispatch(dpipe, .{ ceilDiv(d.n_embd, down_rows), max_tiles_d, 1 }, .{ down_threads, 1, 1 }, &.{ &wde.gpu_buffer, &b.act_q8, &b.padded_order, &b.tile_expert, &b.down_e }, &pd, @sizeOf(GroupedI8Push), 0);
        } else if (down_tc_pipe) |dpipe| {
            const Pd = n_used * T;
            const max_pos_d = Pd + 64 * d.n_experts;
            const max_tiles_d = max_pos_d >> 6;
            const pd = GroupedTCDownPush{ .M = d.n_embd, .K = ef, .slice = down_slice, .n_used = n_used, .dst_tok_stride = n_used * d.n_embd };
            cmd.dispatch(dpipe, .{ ceilDiv(d.n_embd, 64), max_tiles_d, 1 }, .{ 256, 1, 1 }, &.{ &wde.gpu_buffer, &b.swiglu_e, &b.padded_order, &b.tile_expert, &b.down_e }, &pd, @sizeOf(GroupedTCDownPush), 0);
        } else {
            const pd = ExpertsBatchPush{ .M = d.n_embd, .K = ef, .slice = down_slice, .x_stride = ef, .n_used = n_used, .base = 0, .routing_stride = rt_stride, .x_tok_stride = n_used * ef, .y_tok_stride = n_used * d.n_embd };
            cmd.dispatch(down_pipe, .{ n_used * d.n_embd, T, 1 }, .{ 64, 1, 1 }, &.{ &wde.gpu_buffer, &b.swiglu_e, &b.down_e, &b.router_table_e }, &pd, @sizeOf(ExpertsBatchPush), 0);
        }
        // Weighted combine of each token's n_used routed-down slices → hidden +=.
        const ma = MoeAccBatchPush{ .N = d.n_embd, .n_used = n_used, .src_stride = d.n_embd, .a_tok_stride = d.n_embd, .b_tok_stride = n_used * d.n_embd, .routing_stride = rt_stride };
        cmd.dispatch(&self.pipes.moe_weighted_acc_batched, .{ ceilDiv(d.n_embd, 64), T, 1 }, .{ 64, 1, 1 }, &.{ &b.hidden, &b.down_e, &b.router_table_e }, &ma, @sizeOf(MoeAccBatchPush), 0);

        // --- Shared expert: gate/up (dense GEMM) → SwiGLU → down, sigmoid-gated. ---
        const wgs = self.layer(L, "ffn_gate_shexp.weight");
        const wus = self.layer(L, "ffn_up_shexp.weight");
        const wds = self.layer(L, "ffn_down_shexp.weight");
        const wgi = self.layer(L, "ffn_gate_inp_shexp.weight"); // [n_embd, 1] F32
        const fuse_gate_up_swiglu = prefillGateUpSwigluOn();
        const shared_gate_up_uses_cublas = self.use_cublas and T >= self.cublas_min_t;
        const can_fuse_shexp = wgs.info.type_ == .q4_k and wus.info.type_ == .q4_k and
            T >= 32 and sf >= 64 and !shared_gate_up_uses_cublas and fuse_gate_up_swiglu;
        if (can_fuse_shexp) {
            const gp = GateUpSwigluPush{ .M = sf, .K = d.n_embd, .T = T };
            cmd.dispatch(&self.pipes.gemm_q4k_gate_up_swiglu, .{ ceilDiv(sf, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &wgs.gpu_buffer, &wus.gpu_buffer, &b.ffn_norm, &b.swiglu_ff }, &gp, @sizeOf(GateUpSwigluPush), 0);
        } else {
            self.gemmDispatchPrefill(&cmd, wgs, &b.ffn_norm, &b.gate_ff, sf, d.n_embd, T);
            const reuse_q8_gate_up = prefillQ8ReuseOn() and
                self.prefillUsesQ8(wgs, sf, T) and self.prefillUsesQ8(wus, sf, T);
            self.gemmDispatchPrefillImpl(&cmd, wus, &b.ffn_norm, &b.up_ff, sf, d.n_embd, T, reuse_q8_gate_up);
            const ssg = SwigluPush{ .N = T * sf };
            cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(T * sf, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &b.gate_ff, &b.up_ff, &b.swiglu_ff }, &ssg, @sizeOf(SwigluPush), 0);
        }
        // Shared-expert gate scalar = sigmoid(W_gi · norm) per token (1 row, all T).
        const gs = MatvecBatchPush{ .M = 1, .K = d.n_embd, .x_tok_stride = d.n_embd, .y_tok_stride = 1 };
        cmd.dispatch(&self.pipes.dmmv_f32_batched, .{ 1, T, 1 }, .{ 256, 1, 1 }, &.{ &wgi.gpu_buffer, &b.ffn_norm, &b.gate_scalar_e }, &gs, @sizeOf(MatvecBatchPush), 0);
        self.gemmDispatchPrefill(&cmd, wds, &b.swiglu_ff, &b.o, d.n_embd, sf, T);
        const ssa = MoeAccBatchPush{ .N = d.n_embd, .n_used = n_used, .src_stride = d.n_embd, .a_tok_stride = d.n_embd, .b_tok_stride = d.n_embd, .routing_stride = rt_stride };
        cmd.dispatch(&self.pipes.sigmoid_scale_acc_batched, .{ ceilDiv(d.n_embd, 64), T, 1 }, .{ 64, 1, 1 }, &.{ &b.hidden, &b.o, &b.gate_scalar_e }, &ssa, @sizeOf(MoeAccBatchPush), 0);
        self.submit(cmd);
    }

    /// Batched dense GEMM y[T,M] = x[T,K] · W[M,K]ᵀ. cuBLAS fp16 tensor cores for
    /// Q4_K (idx 0) and Q6_K (idx 2) once T amortizes the dequant→fp16 round-trip
    /// (T >= cublas_min_t); otherwise (and for other quants / short prompts) loop
    /// the proven per-token matvec over the token-major buffers (bit-identical to
    /// prefillStep). Always OVERWRITES y (residual folds use add_inplace).
    fn gemmDispatchPrefill(self: *ForwardCuda, cmd: *command.CudaCommand, w: *const LoadedTensor, x: *const CudaBuffer, y: *const CudaBuffer, M: u32, K: u32, T: u32) void {
        self.gemmDispatchPrefillImpl(cmd, w, x, y, M, K, T, false);
    }

    /// Whether this projection takes the shared Q8-activation GEMM path. Paired
    /// projections can use this to safely retain the quantized activation tile.
    fn prefillUsesQ8(self: *ForwardCuda, w: *const LoadedTensor, M: u32, T: u32) bool {
        const idx = dmmvIdx(w.info.type_);
        // The two-row speculative verifier mirrors decode's packed-Q8 matvecs.
        // Besides halving activation bandwidth versus f32 btok, using the same
        // quantization and reduction path keeps verification logits closely
        // aligned with ordinary greedy decode.
        if (is_rocm and self.use_spec_btok and T == 2 and self.batch != null and
            (idx == 0 or idx == 2) and M >= 1 and mtpQ8TypeOn(idx)) return true;
        if (self.use_cublas and T >= self.cublas_min_t and
            ((idx == 0 and self.use_cublas_q4) or (idx == 1 and self.use_cublas_q5) or
                (idx == 2 and self.use_cublas_q6) or (idx == 3 and self.use_cublas_q8))) return false;
        const use_f16 = idx == 0 and std.posix.getenv("ZINC_PREFILL_F16") != null;
        const use_mma = !use_f16 and idx == 0 and std.posix.getenv("ZINC_PREFILL_MMA") != null;
        const use_dp4a = !use_f16 and !use_mma and idx == 0 and std.posix.getenv("ZINC_PREFILL_DP4A") != null;
        return self.batch != null and T >= 32 and M >= 64 and !use_f16 and !use_mma and !use_dp4a and
            (idx == 0 or (is_rocm and (idx == 1 or idx == 2 or idx == 3))) and prefillQ8On();
    }

    fn gemmDispatchPrefillImpl(self: *ForwardCuda, cmd: *command.CudaCommand, w: *const LoadedTensor, x: *const CudaBuffer, y: *const CudaBuffer, M: u32, K: u32, T: u32, reuse_q8: bool) void {
        const idx = dmmvIdx(w.info.type_);

        // Plain-F32 projections are small in Qwen3.5's SSM (alpha/beta are
        // [48, 5120]) but occur twice per layer. The old fallback emitted one
        // dmmv launch per token: 212 * 2 * 48 = 20,352 launches for the reference
        // prompt. This kernel preserves the exact 256-thread reduction while
        // moving the token dimension into grid.y, collapsing that to 96 launches.
        if (idx == 4) {
            const bp = MatvecBatchPush{
                .M = M,
                .K = K,
                .x_tok_stride = K,
                .y_tok_stride = M,
            };
            cmd.dispatch(&self.pipes.dmmv_f32_batched, .{ M, T, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &bp, @sizeOf(MatvecBatchPush), 0);
            return;
        }
        if (is_rocm and self.use_spec_btok and T == 2 and (idx == 0 or idx == 2) and self.batch != null and mtpQ8TypeOn(idx)) {
            const b = &self.batch.?;
            if (!reuse_q8) {
                const qp = QuantActPush{ .K = K, .T = T };
                cmd.dispatch(&self.pipes.quantize_act_q8, .{ ceilDiv(K, 256), T, 1 }, .{ 256, 1, 1 }, &.{ x, &b.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
            }
            const pipe: *CudaPipeline = switch (idx) {
                0 => &self.pipes.dmmv_q4k_q8_btok2,
                2 => &self.pipes.dmmv_q6k_q8_btok2,
                else => unreachable,
            };
            const push = DmmvPush{ .M = M, .K = K };
            // Decode's Q6_K dense FFN down projection deliberately uses one
            // wave; mirror that work partition so each verifier row follows
            // the same accumulation tree. Other projections use two waves.
            const block: u32 = if (idx == 2 and M == self.d.n_embd and K == self.d.n_ff) 32 else dmmv_fast_block;
            cmd.dispatch(pipe, .{ M, 1, 1 }, .{ block, 1, 1 }, &.{ &w.gpu_buffer, &b.act_q8, y }, &push, @sizeOf(DmmvPush), 0);
            return;
        }
        // Speculative target verification and MTP catch-up operate on 2..4
        // consecutive rows. The ordinary short-prefill fallback launches one
        // independent matvec per row and rereads every weight T times. Reuse the
        // serving btok kernels here: each weight row is decoded once and dotted
        // against every token, which is the key amortization behind verification.
        if (self.use_spec_btok and T >= 2 and T <= 27 and idx < 4) {
            const pipe: *CudaPipeline = switch (idx) {
                0 => &self.pipes.dmmv_q4k_btok[T - 2],
                1 => &self.pipes.dmmv_q5k_btok[T - 2],
                2 => &self.pipes.dmmv_q6k_btok[T - 2],
                3 => &self.pipes.dmmv_q8_0_btok[T - 2],
                else => unreachable,
            };
            const push = DmmvPush{ .M = M, .K = K };
            cmd.dispatch(pipe, .{ M, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(DmmvPush), 0);
            return;
        }
        if (self.use_cublas and T >= self.cublas_min_t and ((idx == 0 and self.use_cublas_q4) or (idx == 1 and self.use_cublas_q5) or (idx == 2 and self.use_cublas_q6) or (idx == 3 and self.use_cublas_q8))) {
            const b = &self.batch.?;

            // Determine fp16 weight source: lazy cache (persistent) or scratch
            // (per-GEMM dequant round-trip). The cache eliminates the dequant
            // kernel on 2nd+ prefill calls → ~30% GEMM speedup.
            var w_f16_handle = b.w_f16.handle; // default: scratch
            var need_dequant = true;

            if (self.w_f16_cache) |*cache| {
                const key = @intFromPtr(w.gpu_buffer.handle);
                if (cache.getPtr(key)) |cached| {
                    // Cache HIT: use persistent fp16 weight, skip dequant
                    w_f16_handle = cached.handle;
                    need_dequant = false;
                } else cache_fill: {
                    // Cache MISS: try to allocate persistent buffer + dequant into it
                    const bytes = @as(usize, M) * K * @sizeOf(u16);
                    const reserve: u64 = 512 * 1024 * 1024;
                    if (self.w_f16_cache_bytes + bytes > self.w_f16_cache_cap or
                        shim.cuda_free_memory(self.ctx) < @as(u64, bytes) + reserve)
                    {
                        break :cache_fill;
                    }
                    cache.ensureUnusedCapacity(1) catch break :cache_fill;
                    const new_buf = buffer.createBuffer(self.ctx, bytes) catch null;
                    if (new_buf) |dq_buf| {
                        if (idx == 0) {
                            const dq = DequantQ4KPush{ .M = M, .K = K };
                            cmd.dispatch(&self.pipes.dequant_q4k_to_f16, .{ ceilDiv(M * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, &dq_buf }, &dq, @sizeOf(DequantQ4KPush), 0);
                        } else if (idx == 1) {
                            const dq = DequantQ5KPush{ .M = M, .K = K };
                            cmd.dispatch(&self.pipes.dequant_q5k_to_f16, .{ ceilDiv(M * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, &dq_buf }, &dq, @sizeOf(DequantQ5KPush), 0);
                        } else if (idx == 2) {
                            const dq = DequantQ6KPush{ .M = M, .K = K };
                            cmd.dispatch(&self.pipes.dequant_q6k_to_f16, .{ ceilDiv(M * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, &dq_buf }, &dq, @sizeOf(DequantQ6KPush), 0);
                        } else {
                            const dq = DequantQ8_0Push{ .M = M, .K = K };
                            cmd.dispatch(&self.pipes.dequant_q8_0_to_f16, .{ ceilDiv(M * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, &dq_buf }, &dq, @sizeOf(DequantQ8_0Push), 0);
                        }
                        cache.putAssumeCapacity(key, dq_buf);
                        self.w_f16_cache_bytes += bytes;
                        w_f16_handle = dq_buf.handle;
                        need_dequant = false;
                    }
                    // else: alloc failed → fall through to scratch dequant below
                }
            }

            // Scratch dequant (first call or cache alloc failure)
            if (need_dequant) {
                if (idx == 0) {
                    const dq = DequantQ4KPush{ .M = M, .K = K };
                    cmd.dispatch(&self.pipes.dequant_q4k_to_f16, .{ ceilDiv(M * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, &b.w_f16 }, &dq, @sizeOf(DequantQ4KPush), 0);
                } else if (idx == 1) {
                    const dq = DequantQ5KPush{ .M = M, .K = K };
                    cmd.dispatch(&self.pipes.dequant_q5k_to_f16, .{ ceilDiv(M * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, &b.w_f16 }, &dq, @sizeOf(DequantQ5KPush), 0);
                } else if (idx == 2) {
                    const dq = DequantQ6KPush{ .M = M, .K = K };
                    cmd.dispatch(&self.pipes.dequant_q6k_to_f16, .{ ceilDiv(M * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, &b.w_f16 }, &dq, @sizeOf(DequantQ6KPush), 0);
                } else {
                    const dq = DequantQ8_0Push{ .M = M, .K = K };
                    cmd.dispatch(&self.pipes.dequant_q8_0_to_f16, .{ ceilDiv(M * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, &b.w_f16 }, &dq, @sizeOf(DequantQ8_0Push), 0);
                }
                w_f16_handle = b.w_f16.handle;
            }

            const cvt = F32ToF16Push{ .N = T * K };
            cmd.dispatch(&self.pipes.f32_to_f16, .{ ceilDiv(T * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ x, &b.act_f16 }, &cvt, @sizeOf(F32ToF16Push), 0);
            shim.cuda_cublas_hgemm(self.ctx, @intCast(M), @intCast(T), @intCast(K), w_f16_handle, b.act_f16.handle, y.handle, 0.0);
            return;
        }
        // Fallback: batched tiled GEMM for T >= 32 (reads weight once per tile
        // instead of Tx). Per-token DMMV for T < 32 (avoids 64-token tile waste).
        // ZINC_PREFILL_F16=1: use pre-dequanted fp16 weights (skip dequant entirely)
        // when a cached fp16 version exists. 10-30x faster GEMM.
        // ZINC_PREFILL_DP4A=1: DP4a int8 dot product.
        // ZINC_PREFILL_TC=0: opt out of Q4_K tensor-core (default on).
        // ZINC_PREFILL_Q8_TC=0: opt out of the Q8_0 tensor-core shared-expert path.
        const use_f16 = idx == 0 and std.posix.getenv("ZINC_PREFILL_F16") != null;
        const use_mma = !use_f16 and idx == 0 and std.posix.getenv("ZINC_PREFILL_MMA") != null;
        const use_dp4a = !use_f16 and !use_mma and idx == 0 and std.posix.getenv("ZINC_PREFILL_DP4A") != null;
        const use_q8 = !use_f16 and !use_mma and !use_dp4a and
            (idx == 0 or (is_rocm and (idx == 1 or idx == 2))) and prefillQ8On();
        const use_lowsmem = !use_f16 and !use_mma and !use_dp4a and !use_q8 and idx == 0 and (std.posix.getenv("ZINC_PREFILL_LOWSMEM") == null or
            !std.mem.eql(u8, std.posix.getenv("ZINC_PREFILL_LOWSMEM").?, "0"));
        const use_tc = !use_f16 and !use_dp4a and !use_lowsmem and idx == 0 and (std.posix.getenv("ZINC_PREFILL_TC") == null or
            !std.mem.eql(u8, std.posix.getenv("ZINC_PREFILL_TC").?, "0"));

        // Check for cached fp16 weight when ZINC_PREFILL_F16 is set
        if (use_f16) {
            if (self.w_f16_cache) |*cache| {
                const key = @intFromPtr(w.gpu_buffer.handle);
                if (cache.getPtr(key)) |cached| {
                    // Cache HIT: dispatch fp16 TC GEMM (no dequant!)
                    const push = GemmPush{ .M = M, .K = K, .T = T, .a_offset = 0 };
                    cmd.dispatch(&self.pipes.gemm_f16_tc, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ cached, x, y }, &push, @sizeOf(GemmPush), 0);
                    return;
                }
                // Cache MISS: dequant to fp16 and cache for future use
                const new_buf = buffer.createBuffer(self.ctx, @as(usize, M) * K * @sizeOf(u16)) catch null;
                if (new_buf) |dq_buf| {
                    if (idx == 0) {
                        const dq = DequantQ4KPush{ .M = M, .K = K };
                        cmd.dispatch(&self.pipes.dequant_q4k_to_f16, .{ ceilDiv(M * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, &dq_buf }, &dq, @sizeOf(DequantQ4KPush), 0);
                        cache.put(key, dq_buf) catch {};
                        if (cache.getPtr(key)) |cached| {
                            const push = GemmPush{ .M = M, .K = K, .T = T, .a_offset = 0 };
                            cmd.dispatch(&self.pipes.gemm_f16_tc, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ cached, x, y }, &push, @sizeOf(GemmPush), 0);
                            return;
                        }
                    }
                }
                // Fall through to normal path if cache alloc failed
            }
        }
        if (T >= 32 and M >= 64 and idx < 4) {
            const push = GemmPush{ .M = M, .K = K, .T = T };
            if (use_mma) {
                cmd.dispatch(&self.pipes.gemm_q4k_mma_lowsmem, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(GemmPush), 0);
            } else if (use_q8) {
                // Pre-quant tiled DP4a: quantize input to Q8_0, then tiled DP4a GEMM
                if (self.batch) |b| {
                    if (!reuse_q8) {
                        const qp = QuantActPush{ .K = K, .T = T };
                        cmd.dispatch(&self.pipes.quantize_act_q8, .{ ceilDiv(K, 256), T, 1 }, .{ 256, 1, 1 }, &.{ x, &b.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
                    }
                    const q8push = GemmPush{ .M = M, .K = K, .T = T, .q8_stride = K / 32 * 36 };
                    if (comptime is_rocm) {
                        const tuned_tiles = prefillWmmaTilesOn();
                        const use_t48 = tuned_tiles and T <= 48;
                        const use_t64 = !use_t48 and tuned_tiles and T <= 64;
                        const use_t80 = !use_t48 and !use_t64 and prefillWmmaT80On() and T <= 80;
                        const qpipe = if (use_t48) switch (idx) {
                            0 => if (prefillWmmaQ4DirectOn()) &self.pipes.gemm_q4k_wmma_i8_t48_direct else &self.pipes.gemm_q4k_wmma_i8_t48,
                            1 => &self.pipes.gemm_q5k_wmma_i8_t48,
                            2 => &self.pipes.gemm_q6k_wmma_i8_t48,
                            else => &self.pipes.gemm_q8_0_wmma_i8_t48,
                        } else if (use_t64) switch (idx) {
                            0 => &self.pipes.gemm_q4k_wmma_i8_t64,
                            1 => &self.pipes.gemm_q5k_wmma_i8_t64,
                            2 => &self.pipes.gemm_q6k_wmma_i8_t64,
                            else => &self.pipes.gemm_q8_0_wmma_i8_t64,
                        } else if (use_t80) switch (idx) {
                            0 => &self.pipes.gemm_q4k_wmma_i8_t80,
                            1 => &self.pipes.gemm_q5k_wmma_i8_t80,
                            2 => &self.pipes.gemm_q6k_wmma_i8_t80,
                            else => &self.pipes.gemm_q8_0_wmma_i8_t80,
                        } else switch (idx) {
                            0 => &self.pipes.gemm_q4k_wmma_i8,
                            1 => &self.pipes.gemm_q5k_wmma_i8,
                            2 => &self.pipes.gemm_q6k_wmma_i8,
                            else => &self.pipes.gemm_q8_0_wmma_i8,
                        };
                        const tail = T % 112;
                        // Long prompts usually end in a partial 112-column tile.
                        // Reuse the already-compiled short-prompt kernels for that
                        // remainder instead of making the generic seven-fragment
                        // kernel compute as many as 96 throwaway columns.
                        const tail_tile: u32 = if (prefillWmmaTailOn() and tuned_tiles and
                            !use_t48 and !use_t64 and !use_t80 and T > 112 and tail > 0)
                            if (tail <= 16) 16 else if (tail <= 48) 48 else if (tail <= 64) 64 else if (tail <= 80 and prefillWmmaT80On()) 80 else 0
                        else
                            0;
                        const use_specialized_tail = tail_tile != 0;
                        const main_tiles = if (use_specialized_tail) T / 112 else ceilDiv(T, 112);
                        cmd.dispatch(qpipe, .{ ceilDiv(M, 128), main_tiles, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, &b.act_q8, y }, &q8push, @sizeOf(GemmPush), 0);
                        if (use_specialized_tail) {
                            const tail_pipe = switch (tail_tile) {
                                16 => switch (idx) {
                                    0 => &self.pipes.gemm_q4k_wmma_i8_t16,
                                    1 => &self.pipes.gemm_q5k_wmma_i8_t16,
                                    2 => &self.pipes.gemm_q6k_wmma_i8_t16,
                                    else => &self.pipes.gemm_q8_0_wmma_i8_t16,
                                },
                                48 => switch (idx) {
                                    0 => if (prefillWmmaQ4DirectOn()) &self.pipes.gemm_q4k_wmma_i8_t48_direct else &self.pipes.gemm_q4k_wmma_i8_t48,
                                    1 => &self.pipes.gemm_q5k_wmma_i8_t48,
                                    2 => &self.pipes.gemm_q6k_wmma_i8_t48,
                                    else => &self.pipes.gemm_q8_0_wmma_i8_t48,
                                },
                                64 => switch (idx) {
                                    0 => &self.pipes.gemm_q4k_wmma_i8_t64,
                                    1 => &self.pipes.gemm_q5k_wmma_i8_t64,
                                    2 => &self.pipes.gemm_q6k_wmma_i8_t64,
                                    else => &self.pipes.gemm_q8_0_wmma_i8_t64,
                                },
                                80 => switch (idx) {
                                    0 => &self.pipes.gemm_q4k_wmma_i8_t80,
                                    1 => &self.pipes.gemm_q5k_wmma_i8_t80,
                                    2 => &self.pipes.gemm_q6k_wmma_i8_t80,
                                    else => &self.pipes.gemm_q8_0_wmma_i8_t80,
                                },
                                else => unreachable,
                            };
                            var tail_push = q8push;
                            tail_push.x_offset = (T - tail) * K * 4;
                            cmd.dispatch(tail_pipe, .{ ceilDiv(M, 128), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, &b.act_q8, y }, &tail_push, @sizeOf(GemmPush), 0);
                        }
                    } else {
                        cmd.dispatch(&self.pipes.gemm_q4k_dp4a, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, &b.act_q8, y }, &q8push, @sizeOf(GemmPush), 0);
                    }
                } else {
                    cmd.dispatch(&self.pipes.gemm_q4k_tc_lowsmem, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(GemmPush), 0);
                }
            } else if (use_dp4a) {
                cmd.dispatch(&self.pipes.gemm_q4k_dp4a, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, x, y }, &push, @sizeOf(GemmPush), 0);
            } else if (idx == 3 and q8TcLowsmemOn()) {
                cmd.dispatch(&self.pipes.gemm_q8_0_tc_lowsmem, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(GemmPush), 0);
            } else if (idx == 2) {
                cmd.dispatch(&self.pipes.gemm_q6k_tc_lowsmem, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(GemmPush), 0);
            } else if (idx == 1) {
                cmd.dispatch(&self.pipes.gemm_q5k_tc_lowsmem, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(GemmPush), 0);
            } else if (use_lowsmem) {
                cmd.dispatch(&self.pipes.gemm_q4k_tc_lowsmem, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(GemmPush), 0);
            } else if (use_tc) {
                cmd.dispatch(&self.pipes.gemm_q4k_tc, .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(GemmPush), 0);
            } else {
                cmd.dispatch(&self.pipes.gemm[idx], .{ ceilDiv(M, 64), ceilDiv(T, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(GemmPush), 0);
            }
        } else {
            const push = DmmvPush{ .M = M, .K = K };
            if (idx < 4) {
                cmd.dispatch(&self.pipes.dmmv_fast[idx], .{ M, T, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(DmmvPush), 0);
            } else {
                var t: u32 = 0;
                while (t < T) : (t += 1) {
                    const tp = DmmvPush{ .M = M, .K = K, .x_offset = t * K * 4, .y_offset = t * M * 4 };
                    cmd.dispatch(&self.pipes.dmmv[idx], .{ M, 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &tp, @sizeOf(DmmvPush), 0);
                }
            }
        }
    }

    /// Allocate (or reuse) the token-major batched scratch for T tokens.
    fn ensureBatch(self: *ForwardCuda, T: u32) !*BatchScratch {
        if (self.batch) |*bb| {
            if (bb.t_cap >= T) return bb;
            self.freeBatch();
        }
        const d = self.d;
        const ctx = self.ctx;
        const f4 = @sizeOf(f32);
        const max_k = @max(@max(d.n_embd, d.q_dim), @max(d.d_inner, d.n_ff));
        const max_w = @max(@max(d.conv_channels, 2 * d.q_dim), d.n_ff) * @max(d.n_embd, d.n_ff);
        // Effort 29 T2: MoE prefill scratch. n_ff is the routed-expert intermediate
        // (ef) on the MoE rows; gate/up/swiglu_ff also serve the shared expert
        // (shexp_ff), so size them to the larger of the two. The routed buffers are
        // size-1 stubs on the dense qwen35 (n_experts==0).
        const max_ff = @max(d.n_ff, d.shexp_ff);
        const moe = d.n_experts > 0;
        const nu = @max(@as(u32, 1), d.n_experts_used);
        const e_gu = if (moe) T * nu * d.n_ff * f4 else f4; // [T, n_used*ef]
        const e_dn = if (moe) T * nu * d.n_embd * f4 else f4; // [T, n_used*n_embd]
        self.batch = BatchScratch{
            .t_cap = T,
            .hidden = try buffer.createBuffer(ctx, T * d.n_embd * f4),
            .norm = try buffer.createBuffer(ctx, T * d.n_embd * f4),
            .o = try buffer.createBuffer(ctx, T * d.n_embd * f4),
            .qfull = try buffer.createBuffer(ctx, T * 2 * d.q_dim * f4),
            .q = try buffer.createBuffer(ctx, T * d.q_dim * f4),
            .attn_gate = try buffer.createBuffer(ctx, T * d.q_dim * f4),
            .k = try buffer.createBuffer(ctx, T * d.kv_dim * f4),
            .v = try buffer.createBuffer(ctx, T * d.kv_dim * f4),
            .attn_out = try buffer.createBuffer(ctx, T * d.q_dim * f4),
            .qkv = try buffer.createBuffer(ctx, T * d.conv_channels * f4),
            .conv_out = try buffer.createBuffer(ctx, T * d.conv_channels * f4),
            .z = try buffer.createBuffer(ctx, T * d.d_inner * f4),
            .alpha = try buffer.createBuffer(ctx, T * d.dt_rank * f4),
            .beta = try buffer.createBuffer(ctx, T * d.dt_rank * f4),
            .delta_out = try buffer.createBuffer(ctx, T * d.d_inner * f4),
            .ssm_gn = try buffer.createBuffer(ctx, T * d.d_inner * f4),
            .ffn_norm = try buffer.createBuffer(ctx, T * d.n_embd * f4),
            .gate_ff = try buffer.createBuffer(ctx, T * max_ff * f4),
            .up_ff = try buffer.createBuffer(ctx, T * max_ff * f4),
            .swiglu_ff = try buffer.createBuffer(ctx, T * max_ff * f4),
            .act_f16 = try buffer.createBuffer(ctx, T * max_k * @sizeOf(u16)),
            .w_f16 = try buffer.createBuffer(ctx, max_w * @sizeOf(u16)),
            .act_q8 = try buffer.createBuffer(ctx, T * (@max(max_k, nu * d.n_ff) / 32) * 36),
            .router_logits_e = try buffer.createBuffer(ctx, if (moe) T * d.n_experts * f4 else f4),
            .router_table_e = try buffer.createBuffer(ctx, if (moe) T * 2 * nu * f4 else f4),
            .gate_e = try buffer.createBuffer(ctx, e_gu),
            .up_e = try buffer.createBuffer(ctx, e_gu),
            .swiglu_e = try buffer.createBuffer(ctx, e_gu),
            .down_e = try buffer.createBuffer(ctx, e_dn),
            .gate_scalar_e = try buffer.createBuffer(ctx, if (moe) T * f4 else f4),
            .padded_order = try buffer.createBuffer(ctx, if (moe) @max(@as(u32, 1), T * nu + 64 * d.n_experts) * @sizeOf(u32) else f4),
            .tile_expert = try buffer.createBuffer(ctx, if (moe) @max(@as(u32, 1), (T * nu >> 4) + d.n_experts + 2) * @sizeOf(u32) else f4),
        };
        return &self.batch.?;
    }

    fn freeBatch(self: *ForwardCuda) void {
        if (self.batch) |*bb| {
            inline for (.{ &bb.hidden, &bb.norm, &bb.o, &bb.qfull, &bb.q, &bb.attn_gate, &bb.k, &bb.v, &bb.attn_out, &bb.qkv, &bb.conv_out, &bb.z, &bb.alpha, &bb.beta, &bb.delta_out, &bb.ssm_gn, &bb.ffn_norm, &bb.gate_ff, &bb.up_ff, &bb.swiglu_ff, &bb.act_f16, &bb.w_f16, &bb.act_q8, &bb.router_logits_e, &bb.router_table_e, &bb.gate_e, &bb.up_e, &bb.swiglu_e, &bb.down_e, &bb.gate_scalar_e, &bb.padded_order, &bb.tile_expert }) |buf| {
                buffer.freeBuffer(buf);
            }
            self.batch = null;
        }
    }

    pub fn decodeBatch(self: *ForwardCuda, tokens: []const u32, positions: []const u32, slots: []const u32, out: []u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        if (self.kv_k_slots == null) return error.SlotStateNotAllocated;
        const B: u32 = @intCast(tokens.len);
        std.debug.assert(positions.len == B and slots.len == B and out.len == B);
        const f4 = @sizeOf(f32);

        // A single active HTTP request should use the same launch-collapsed,
        // fused matvec chain as CLI decode, with only the recurrent/KV buffers
        // redirected to its scheduler slot. The generic B-wide path otherwise
        // pays three synchronous metadata uploads and misses the serial fusion
        // set at B=1. True concurrent batches continue through the code below.
        if (B == 1 and b1MatvecOn()) {
            out[0] = try self.decodeSlotStep(tokens[0], positions[0], slots[0]);
            return;
        }

        // Effort 28 B==1 matvec fast path: when this step batches a single
        // sequence, route the per-layer projection/FFN GEMMs (`gemmDispatch`) to
        // the tuned matvec. `defer` clears it so an early-return error never leaks
        // the flag past this call (production decodeStep/prefillStep never set it).
        self.decode_b1 = (B == 1) and (self.decode_b1_force orelse b1MatvecOn());
        defer self.decode_b1 = false;
        // Small-B (2..8) Q4_K token-batch matvec fast path (opt-in). `defer` clears
        // it so an early-return error never leaks the flag past this call.
        self.decode_mrow = (B >= 2 and B <= 27) and (self.decode_mrow_force orelse mrowMatvecOn());
        defer self.decode_mrow = false;
        // MoE shared-expert batching RIDES the btok path: batching the dense shared
        // expert over B rows is only a win when the batched GEMM is the bandwidth-
        // bound `btok` matvec — through the 64×64 tile GEMM it is SLOWER than the
        // per-row matvec at small B (tile-padding tax, measured −16% at B=4). So gate
        // it on `decode_mrow` (= mrow/btok active) → no regression on the default
        // mrow-off path. `defer` clears it so a forced A/B value never leaks.
        self.moe_shared_batched = self.decode_mrow and (self.moe_shared_batched_force orelse moeSharedBatchedOn());
        defer self.moe_shared_batched = false;
        // MoE launch-collapse (async routed-expert path drops its per-row sync).
        // Default-on; harness `_force` restores the per-row sync for an A/B. `defer`
        // resets the live flag (the default field value stays true for serial paths
        // that never touch it — harmless since they never run the routed-batch path).
        self.moe_collapse = self.moe_collapse_force orelse true;
        defer self.moe_collapse = true;
        // DENSE batched-decode launch-collapse (async submit + single tail drain).
        // Default-on; `_force` restores the per-layer sync for the harness A/B.
        self.decode_collapse = self.decode_collapse_force orelse true;
        defer self.decode_collapse = true;
        // Effort 29 T3: route the B>27 serving GEMMs through cuBLAS fp16 TC (the
        // f32 tile GEMM's regime). Independent of decode_mrow (btok owns B≤27). The
        // min-B is tunable for the A/B sweep (ZINC_SERVE_CUBLAS_MINB). `defer` clears
        // the live flag so a forced value never leaks past this call.
        if (std.posix.getenv("ZINC_SERVE_CUBLAS_MINB")) |mb| {
            self.serve_cublas_min_b = std.fmt.parseInt(u32, mb, 10) catch self.serve_cublas_min_b;
        }
        self.serve_cublas = (B >= self.serve_cublas_min_b) and (self.serve_cublas_force orelse serveCublasOn());
        defer self.serve_cublas = false;
        // Effort 29 T3 option-b: resident fp16 weight cache for the serving GEMMs.
        // Only meaningful when serve_cublas engages; lazily inits the map + reads
        // the budget cap on first enable. ADDITIVE — the cache only changes WHERE
        // the fp16 weight comes from (resident vs per-step scratch), the fp16 bytes
        // are bit-identical to the per-step dequant → same token-correctness.
        self.serve_wcache_on = self.serve_cublas and (self.serve_wcache_force orelse serveWcacheOn());
        if (self.serve_wcache_on and self.serve_wcache == null) {
            self.serve_wcache = std.AutoHashMap(usize, CudaBuffer).init(self.allocator);
            if (std.posix.getenv("ZINC_SERVE_WCACHE_MB")) |mb| {
                if (std.fmt.parseInt(usize, mb, 10) catch null) |v| self.serve_wcache_cap = v * 1024 * 1024;
            }
        }

        const db = try self.ensureDecodeBatch(B);
        const out_norm = self.model.get("output_norm.weight") orelse return error.MissingTensor;
        const lm_head = self.model.get("output.weight") orelse return error.MissingTensor;

        // EMBED all B rows into db.hidden, SAME kernel/weight as serial decodeStep
        // so the projection inputs are bit-aligned. Suspect-#2 port: collapse the
        // per-row B-serial round-trips. GPU path: upload all B token ids ONCE to
        // tok_scratch, then ONE command buffer dispatches embed_q4k per row (the
        // kernel reads tok[0] → alias tok_scratch at bi·4 for the bi-th id) into
        // each row's db.hidden slice → ONE commitAndWait (was B). CPU path: dequant
        // all B rows into the persistent embed_host, ONE upload to the contiguous
        // [B,n_embd] db.hidden (was B uploads).
        var bi: u32 = 0;
        while (bi < B) : (bi += 1) std.debug.assert(slots[bi] < self.n_slots and positions[bi] < self.slot_ctx);
        if (self.embed_gpu) {
            const tok_buf = &self.tok_scratch.?;
            buffer.upload(ctx, tok_buf, std.mem.sliceAsBytes(tokens));
            const push = EmbedPush{ .K = d.n_embd, .vocab = d.vocab };
            const nsb = d.n_embd / 256;
            var cmd = try command.beginCommand(ctx);
            bi = 0;
            while (bi < B) : (bi += 1) {
                var hrow = try buffer.aliasBuffer(&db.hidden, bi * d.n_embd * f4, d.n_embd * f4);
                defer buffer.freeBuffer(&hrow);
                var trow = try buffer.aliasBuffer(tok_buf, bi * @sizeOf(u32), @sizeOf(u32));
                defer buffer.freeBuffer(&trow);
                cmd.dispatch(&self.pipes.embed_q4k, .{ nsb, 1, 1 }, .{ 256, 1, 1 }, &.{ self.embed_weight.?, &trow, &hrow }, &push, @sizeOf(EmbedPush), 0);
            }
            cmd.commitAndWait();
        } else {
            const host = self.embed_host.?[0 .. B * d.n_embd];
            bi = 0;
            while (bi < B) : (bi += 1) {
                self.model.dequantEmbeddingRow(tokens[bi], host[bi * d.n_embd ..][0..d.n_embd]);
            }
            buffer.upload(ctx, &db.hidden, std.mem.sliceAsBytes(host));
        }

        // 4c-step-2: upload per-seq positions[]/slots[] ONCE (same for all layers)
        // to device u32 buffers for the fused batched attention kernels.
        // E28 degradation fix: reuse the persistent pos/slots scratch (allocated
        // in allocSlotState, sized to n_slots ≥ B) instead of a cudaMalloc/free
        // pair per step. The fused batched kernels read only the first B entries.
        std.debug.assert(B <= self.n_slots);
        const pos_buf = &self.pos_scratch.?;
        const slot_buf = &self.slots_scratch.?;
        buffer.upload(ctx, pos_buf, std.mem.sliceAsBytes(positions));
        buffer.upload(ctx, slot_buf, std.mem.sliceAsBytes(slots));
        var max_seq_len: u32 = 1;
        for (positions) |p| max_seq_len = @max(max_seq_len, p + 1);

        // Effort 28 CUDA-graph replay (opt-in): the batched-decode step is sync-free
        // after the launch-collapse, so capture the whole layer chain + tail into a
        // per-B cached exec and replay it as ONE graph launch. The embed + pos/slot
        // uploads above already ran (sync) into device scratch, so the captured
        // kernels read the current step's data. Requires `decode_collapse` (no
        // commitAndWait inside a captured region). For MoE, additionally requires the
        // capturable batched routed+shared path (`moe_shared_batched` + `moe_collapse`,
        // both ride `decode_mrow`) and `moe_graph_capturable` (no host id readback on
        // ANY MoE layer) — else the routed loop host-gathers ids mid-capture.
        const moe_graph_ok = d.n_experts == 0 or
            (self.moe_graph_capturable and self.moe_shared_batched and self.moe_collapse);
        const use_graph = shim.cuda_graph_supported() != 0 and moe_graph_ok and self.decode_collapse and (self.batch_graph_force orelse self.batch_graph_on);
        if (use_graph) {
            return try self.decodeBatchGraph(B, db, pos_buf, slot_buf, max_seq_len, &out_norm.gpu_buffer, &lm_head.gpu_buffer, lm_head.info.type_, out);
        }

        // Layer-major: each block reads/writes db.hidden over ALL B rows. The big
        // projection/FFN GEMMs read each weight ONCE for all B rows (the
        // amortization win — 4c-step-1). Both the attention inner (4c-step-2a) AND
        // the SSM inner (conv1d/delta-net/gated-norm, 4c-step-2b) are now FUSED into
        // batched grid.y/z=B kernels — each block is ONE command buffer / one sync.
        var L: u32 = 0;
        while (L < d.n_layers) : (L += 1) {
            if (self.layer_is_attn[L]) {
                try self.attentionLayerBatchedDecode(L, B, db, pos_buf, slot_buf, max_seq_len);
            } else {
                try self.ssmLayerBatchedDecode(L, B, db, pos_buf, slot_buf);
            }
            if (d.n_experts > 0) try self.moeFfnBlockBatchedDecode(L, B, db) else try self.ffnBlockBatchedDecode(L, B, db);
        }

        // TAIL: rms_norm → LM head → argmax for all B rows. Suspect-#2 port: chain
        // every row into ONE command buffer + ONE commitAndWait + ONE B-wide
        // download (was B serial commitAndWait+download per decoded token). On one
        // stream the dispatches execute strictly in order, so reusing the single
        // norm_buf/logits_buf scratch across rows is hazard-free (row b+1's rms_norm
        // can't run until row b's LM head consumed norm_buf) → identical math to the
        // per-row form. Only the argmax OUTPUT is per-row → each writes its own slot
        // of argmax_scratch (aliased bi·4), downloaded once into out[0..B].
        const argmax_out = &self.argmax_scratch.?;
        var cmd = try command.beginCommand(ctx);
        bi = 0;
        while (bi < B) : (bi += 1) {
            var hrow = try buffer.aliasBuffer(&db.hidden, bi * d.n_embd * f4, d.n_embd * f4);
            defer buffer.freeBuffer(&hrow);
            var am_slot = try buffer.aliasBuffer(argmax_out, bi * @sizeOf(u32), @sizeOf(u32));
            defer buffer.freeBuffer(&am_slot);
            const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &hrow, &out_norm.gpu_buffer, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
            self.lmHeadDispatch(&cmd, &lm_head.gpu_buffer, lm_head.info.type_, false);
            self.argmaxDispatch(&cmd, &am_slot);
        }
        cmd.commitAndWait();
        // The tail commitAndWait drained the shared stream (incl. any async layer
        // commands stashed by the dense launch-collapse) → their completion is
        // guaranteed; free the stashed handles. No-op when n_pending==0 (e.g. the
        // MoE path drains the ring each layer via its `waitPending`).
        self.drainPending();
        buffer.download(ctx, argmax_out, std.mem.sliceAsBytes(out));
    }

    /// Single-request serving step: production serial scratch and fused kernels,
    /// scheduler-owned slot state. Alias wrappers may be released after dispatch
    /// because HIP/CUDA copies raw kernel arguments at launch; the parent slot
    /// allocations remain alive for the server lifetime. The tail is the sole
    /// stream synchronization, matching `decodeStep`.
    fn decodeSlotStep(self: *ForwardCuda, token: u32, pos: u32, slot: u32) !u32 {
        const d = self.d;
        const ctx = self.ctx;
        std.debug.assert(slot < self.n_slots and pos < self.slot_ctx);

        if (self.embed_gpu) {
            self.host_tok_in[0] = token;
            buffer.upload(ctx, &self.tok_in_buf, std.mem.sliceAsBytes(self.host_tok_in));
            try self.recordEmbed();
        } else {
            self.model.dequantEmbeddingRow(token, self.host_embed);
            buffer.upload(ctx, &self.hidden, std.mem.sliceAsBytes(self.host_embed));
        }

        var L: u32 = 0;
        while (L < d.n_layers) : (L += 1) {
            if (self.layer_is_attn[L]) {
                try self.attentionLayerSlot(L, pos, slot);
            } else {
                try self.ssmLayerSlot(L, pos, slot);
            }
            if (d.n_experts > 0) try self.moeFfnBlock(L) else try self.ffnBlock(L);
        }

        const out_norm = self.model.get("output_norm.weight") orelse return error.MissingTensor;
        const lm_head = self.model.get("output.weight") orelse return error.MissingTensor;
        var cmd = try command.beginCommand(ctx);
        self.tailDispatch(&cmd, &out_norm.gpu_buffer, &lm_head.gpu_buffer, lm_head.info.type_);
        cmd.commitAndWait();
        self.drainPending();

        var tok: u32 = 0;
        buffer.download(ctx, &self.argmax_buf, std.mem.asBytes(&tok));
        return tok;
    }

    /// Effort 28: dense batched-decode step via CUDA-graph replay (one cached exec
    /// per B). The embed + per-seq pos/slot/token uploads already ran (sync) into
    /// device scratch in `decodeBatch`, so `db.hidden`/`pos_buf`/`slot_buf` hold this
    /// step's data; capture the layer chain + tail (all on ONE stream via the
    /// `submit` no-sync path under `capturing`) and launch as a single graph. Across
    /// steps the topology AND push-constants are invariant (every per-seq scalar is
    /// read from a device buffer, not a push constant) → `cuGraphExecUpdate` is a
    /// cheap in-place no-op. Bit-identical to the non-graph batched chain (same
    /// kernels, same order, same single stream) → token-identical to N serial runs.
    /// Dense OR MoE (MoE gated by the caller on `moe_graph_capturable` — every layer
    /// reads expert ids GPU-side, no host readback — plus `moe_shared_batched` +
    /// `moe_collapse` so the routed+shared path is the no-sync batched form).
    fn decodeBatchGraph(self: *ForwardCuda, B: u32, db: *DecodeBatch, pos_buf: *const CudaBuffer, slot_buf: *const CudaBuffer, max_seq_len: u32, out_norm: *const CudaBuffer, lm_head: *const CudaBuffer, lm_type: gguf.GGMLType, out: []u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const f4 = @sizeOf(f32);
        if (self.batch_graph[B] == null) self.batch_graph[B] = shim.cuda_graph_create();
        const g = self.batch_graph[B].?;

        self.capturing = true;
        _ = shim.cuda_graph_begin(ctx);
        // Layer chain — each batched block rides the async `submit` ring, which under
        // `capturing` records onto the captured stream with no event/sync. The
        // attention smem is sized at the constant slot_ctx (not data-dependent
        // max_seq_len) so the captured kernel-node shape is invariant across steps.
        var L: u32 = 0;
        while (L < d.n_layers) : (L += 1) {
            if (self.layer_is_attn[L]) {
                try self.attentionLayerBatchedDecode(L, B, db, pos_buf, slot_buf, max_seq_len);
            } else {
                try self.ssmLayerBatchedDecode(L, B, db, pos_buf, slot_buf);
            }
            // MoE: the capturable batched routed+shared path (gated above on
            // moe_graph_capturable + moe_shared_batched + moe_collapse → no host
            // readback, submit/waitPending are no-ops under `capturing`).
            if (d.n_experts > 0) try self.moeFfnBlockBatchedDecode(L, B, db) else try self.ffnBlockBatchedDecode(L, B, db);
        }
        // TAIL: rms_norm → LM head → argmax for all B rows, recorded into ONE command
        // buffer (no commitAndWait — captured). On one stream the dispatches execute
        // in order so reusing norm_buf/logits_buf across rows is hazard-free; only the
        // argmax OUTPUT is per-row (each writes its own argmax_scratch slot).
        const argmax_out = &self.argmax_scratch.?;
        var cmd = try command.beginCommand(ctx);
        var bi: u32 = 0;
        while (bi < B) : (bi += 1) {
            var hrow = try buffer.aliasBuffer(&db.hidden, bi * d.n_embd * f4, d.n_embd * f4);
            defer buffer.freeBuffer(&hrow);
            var am_slot = try buffer.aliasBuffer(argmax_out, bi * @sizeOf(u32), @sizeOf(u32));
            defer buffer.freeBuffer(&am_slot);
            const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &hrow, out_norm, &self.norm_buf }, &rms, @sizeOf(RmsPush), 0);
            self.lmHeadDispatch(&cmd, lm_head, lm_type, false);
            self.argmaxDispatch(&cmd, &am_slot);
        }
        cmd.releaseCompleted(); // captured onto the stream; no event record / no sync
        self.capturing = false;

        // End capture, instantiate-or-update the cached exec, launch + ONE sync
        // (drains the whole captured layer chain + tail).
        _ = shim.cuda_graph_end_launch(ctx, g);
        // The argmax slots are now resident on the device; one B-wide D2H.
        buffer.download(ctx, argmax_out, std.mem.sliceAsBytes(out));
    }

    /// (Re)allocate the [B, dim] batched-decode scratch, growing if B exceeds cap.
    fn ensureDecodeBatch(self: *ForwardCuda, B: u32) !*DecodeBatch {
        if (self.decode_batch) |*db| {
            if (db.b_cap >= B) return db;
            db.free();
            self.decode_batch = null;
        }
        self.decode_batch = try DecodeBatch.alloc(self.ctx, self.d, B);
        return &self.decode_batch.?;
    }

    /// Batched GEMM y[B,M] = x[B,K] · Wᵀ for B rows, reading W ONCE (token-major
    /// buffers; acc_mode 1 accumulates into y for the residual). The dense
    /// projection / FFN weights are all q4k/q5k/q6k/q8_0 (tiled gemm) or f32.
    fn gemmDispatch(self: *ForwardCuda, cmd: *command.CudaCommand, w: *const LoadedTensor, x: *const CudaBuffer, y: *const CudaBuffer, M: u32, K: u32, B: u32, acc_mode: u32) void {
        // Effort 28: B==1 decode → tuned matvec (decodeStep's path). At B==1 the
        // single-row x/y are contiguous (a_offset 0) and dmmvDispatch honors the
        // residual acc_mode (O-proj / FFN-down / SSM-out use acc_mode 1), so this
        // is a drop-in for the batched-GEMM tile across every decode weight type
        // (q4k/q5k/q6k/q8_0/f32). Only set on a B==1 decodeBatch step.
        if (B == 1 and self.decode_b1) {
            self.dmmvDispatch(cmd, w, x, y, M, K, acc_mode, 0);
            return;
        }
        // Effort 28: small-B Q4_K token-batch matvec. x/y are token-major [B,*]
        // (a_offset/x_offset/y_offset 0), and the kernel honors acc_mode for the
        // residual GEMMs (O-proj / FFN-down / SSM-out). Reads each weight row once,
        // amortizing the dequant over the B tokens → bandwidth-bound, no tile waste.
        if (self.decode_mrow and B >= 2 and B <= 27) {
            // Q4_K covers most proj/gate/up; Q6_K/Q5_K/Q8_0 cover the residual
            // O-proj/FFN-down/SSM-out on mixed-quant layers. Each btok is
            // bit-identical-per-row to its *_fast matvec → token-identical decode.
            const btok: ?*CudaPipeline = switch (w.info.type_) {
                .q4_k => &self.pipes.dmmv_q4k_btok[B - 2],
                .q6_k => &self.pipes.dmmv_q6k_btok[B - 2],
                .q5_k => &self.pipes.dmmv_q5k_btok[B - 2],
                .q8_0 => &self.pipes.dmmv_q8_0_btok[B - 2],
                else => null,
            };
            if (btok) |pipe| {
                const push = DmmvPush{ .M = M, .K = K, .acc_mode = acc_mode };
                cmd.dispatch(pipe, .{ M, 1, 1 }, .{ 64, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(DmmvPush), 0);
                return;
            }
        }
        // Effort 29 T3: the B>27 serving regime. btok stops at B=27 (float sum[B]
        // register ceiling); past it the default f32 tile GEMM (below) pads B to 64
        // and runs ALU math. Route the same dense quants the prefill cuBLAS path
        // handles (Q4_K/Q6_K/Q8_0) through cuBLAS fp16 TC instead: dequant W→fp16
        // (db.w_f16, reused per GEMM, stream-ordered) + x→fp16 (db.act_f16) + one
        // cublasGemmEx. acc_mode==1 (O-proj/down/ssm-out) folds into the residual y
        // via beta=1; acc_mode==0 overwrites (beta=0). Token-correctness-tolerance
        // gate (fp16 TC reduction order differs from the f32 tile GEMM). q5_k/f32
        // have no dequant→fp16 kernel → fall through to the tile GEMM.
        if (self.serve_cublas) {
            const ci = dmmvIdx(w.info.type_);
            if (ci == 0 or ci == 2 or ci == 3) {
                const db = &self.decode_batch.?;
                // Effort 29 T3 option-b: pick the fp16 weight SOURCE. With the
                // resident cache on, a HIT reuses the once-dequant'd buffer (skip
                // the dequant launch entirely); a MISS dequants ONCE into a freshly
                // allocated resident buffer (budget/VRAM permitting) and reuses it
                // forever after. Cache off / cap-exhausted / alloc-fail → the
                // per-step scratch `db.w_f16` (original cycle-7 round-trip path).
                var dq_target: *const CudaBuffer = &db.w_f16;
                var whandle = db.w_f16.handle;
                var need_dequant = true;
                if (self.serve_wcache_on and self.serve_wcache != null) {
                    const cache = &self.serve_wcache.?;
                    const key = @intFromPtr(w.gpu_buffer.handle);
                    if (cache.getPtr(key)) |cached| {
                        whandle = cached.handle;
                        need_dequant = false;
                    } else {
                        const wbytes = @as(usize, M) * @as(usize, K) * @sizeOf(u16);
                        if (self.serve_wcache_bytes + wbytes <= self.serve_wcache_cap) {
                            if (buffer.createBuffer(self.ctx, wbytes)) |newbuf| {
                                cache.put(key, newbuf) catch {
                                    buffer.freeBuffer(@constCast(&newbuf));
                                };
                                if (cache.getPtr(key)) |cp| {
                                    self.serve_wcache_bytes += wbytes;
                                    dq_target = cp; // dequant ONCE into the resident buffer
                                    whandle = cp.handle;
                                }
                            } else |_| {}
                        }
                    }
                }
                if (need_dequant) {
                    if (ci == 0) {
                        const dq = DequantQ4KPush{ .M = M, .K = K };
                        cmd.dispatch(&self.pipes.dequant_q4k_to_f16, .{ ceilDiv(M * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, dq_target }, &dq, @sizeOf(DequantQ4KPush), 0);
                    } else if (ci == 2) {
                        const dq = DequantQ6KPush{ .M = M, .K = K };
                        cmd.dispatch(&self.pipes.dequant_q6k_to_f16, .{ ceilDiv(M * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, dq_target }, &dq, @sizeOf(DequantQ6KPush), 0);
                    } else {
                        const dq = DequantQ8_0Push{ .M = M, .K = K };
                        cmd.dispatch(&self.pipes.dequant_q8_0_to_f16, .{ ceilDiv(M * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, dq_target }, &dq, @sizeOf(DequantQ8_0Push), 0);
                    }
                }
                const cvt = F32ToF16Push{ .N = B * K };
                cmd.dispatch(&self.pipes.f32_to_f16, .{ ceilDiv(B * K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ x, &db.act_f16 }, &cvt, @sizeOf(F32ToF16Push), 0);
                const beta: f32 = if (acc_mode == 1) 1.0 else 0.0;
                shim.cuda_cublas_hgemm(self.ctx, @intCast(M), @intCast(B), @intCast(K), whandle, db.act_f16.handle, y.handle, beta);
                return;
            }
        }
        const idx = dmmvIdx(w.info.type_);
        const push = GemmPush{ .M = M, .K = K, .T = B, .acc_mode = acc_mode };
        if (w.info.type_ == .f32) {
            cmd.dispatch(&self.pipes.gemm_f32, .{ ceilDiv(M, 64), ceilDiv(B, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(GemmPush), 0);
        } else {
            cmd.dispatch(&self.pipes.gemm[idx], .{ ceilDiv(M, 64), ceilDiv(B, 64), 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(GemmPush), 0);
        }
    }

    /// 4c-step-2 attention block: pre-norm + Q(+gate)/K/V + O projections run as
    /// GEMMs over ALL B rows (each weight read once), AND the per-seq attention
    /// inner is now FUSED into two batched grid.y=B launches — collapsing the
    /// per-row deinterleave / q-k norm / RoPE / KV-write / softmax loop (mirrors
    /// gemma's 1c). One command buffer, ONE commitAndWait (was B per-row syncs):
    ///   1. batched pre-norm + Q(+gate)/K/V projections (GEMMs over B rows)
    ///   2. qwen_norm_rope_qkv_seq — fused deinterleave + per-head q/k norm + RoPE
    ///      + slot KV write + gate extract over (n_head+2*n_kv_head, B) at per-seq
    ///      positions[]/slots[] (device buffers)
    ///   3. naive_attention_batched_seq — per-(head,b) softmax(QK^T)V + sink over
    ///      each row's slot KV history, seq_len=positions[b]+1
    ///   4. sigmoid_mul — sigmoid-z gate over all B rows (token-major [B,q_dim])
    ///   5. batched O projection, accumulate into db.hidden (residual)
    /// Per-(head,b) arithmetic is copied from the per-row kernels → token-identical
    /// to N serial runs (ARGMAX-identical given the GEMM reduction order).
    fn attentionLayerBatchedDecode(self: *ForwardCuda, L: u32, B: u32, db: *DecodeBatch, pos_buf: *const CudaBuffer, slot_buf: *const CudaBuffer, max_seq_len: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const wq = self.layer(L, "attn_q.weight");
        const wk = self.layer(L, "attn_k.weight");
        const wv = self.layer(L, "attn_v.weight");
        const wqn = self.layer(L, "attn_q_norm.weight");
        const wkn = self.layer(L, "attn_k_norm.weight");
        const wo = self.layer(L, "attn_output.weight");
        const wan = self.layer(L, "attn_norm.weight");

        var cmd = try command.beginCommand(ctx);

        // 1. Batched pre-attn norm + Q(+gate)/K/V projections over B rows.
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ B, 1, 1 }, .{ 256, 1, 1 }, &.{ &db.hidden, &wan.gpu_buffer, &db.norm }, &rms, @sizeOf(RmsPush), 0);
        self.gemmDispatch(&cmd, wq, &db.norm, &db.qfull, 2 * d.q_dim, d.n_embd, B, 0);
        self.gemmDispatch(&cmd, wk, &db.norm, &db.k, d.kv_dim, d.n_embd, B, 0);
        self.gemmDispatch(&cmd, wv, &db.norm, &db.v, d.kv_dim, d.n_embd, B, 0);

        // 2. Fused per-seq deinterleave + q/k norm + RoPE + slot KV write + gate
        //    extract over (n_head + 2*n_kv_head, B). Writes db.q, db.gate, slot K/V.
        const qkv = QwenQkvSeqPush{ .head_dim = d.head_dim, .eps = d.rms_eps, .rope_dim = d.rope_dim, .n_head = d.n_head, .n_kv_head = d.n_kv_head, .slot_ctx = self.slot_ctx };
        const nr_sh: u32 = d.head_dim * 4;
        cmd.dispatch(&self.pipes.qwen_norm_rope_qkv_seq, .{ d.n_head + 2 * d.n_kv_head, B, 1 }, .{ 256, 1, 1 }, &.{ &db.qfull, &db.k, &db.v, &wqn.gpu_buffer, &wkn.gpu_buffer, &self.inv_freq, &db.q, &db.gate, &self.kv_k_slots.?[L], &self.kv_v_slots.?[L], pos_buf, slot_buf }, &qkv, @sizeOf(QwenQkvSeqPush), nr_sh);

        // 3. Batched per-seq attention over each row's slot KV history.
        const attn = AttnSlotPush{ .head_dim = d.head_dim, .n_heads = d.n_head, .n_kv_heads = d.n_kv_head, .slot_ctx = self.slot_ctx, .attn_scale_bits = 0, .sink_offset = L * d.n_head };
        // The kernel's dynamic shared mem holds up to `seq_len`=positions[b]+1 f32
        // scores per row. Under graph capture, request the constant MAX (slot_ctx)
        // so the captured kernel-node smem size is invariant as positions grow →
        // the cached exec updates in place instead of re-instantiating each step.
        // The kernel still uses only seq_len entries (mirrors the serial path).
        const attn_smem: u32 = if (self.capturing) self.slot_ctx * 4 else max_seq_len * 4;
        cmd.dispatch(&self.pipes.naive_attention_batched_seq, .{ d.n_head, B, 1 }, .{ 256, 1, 1 }, &.{ &db.q, &self.kv_k_slots.?[L], &self.kv_v_slots.?[L], &self.sinks, &db.attn_out, pos_buf, slot_buf }, &attn, @sizeOf(AttnSlotPush), attn_smem);

        // 4. Per-seq sigmoid-z gate over all B rows (token-major [B,q_dim] contiguous).
        const sm = SigmoidMulPush{ .N = B * d.q_dim };
        cmd.dispatch(&self.pipes.sigmoid_mul, .{ ceilDiv(B * d.q_dim, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &db.attn_out, &db.gate, &db.attn_out }, &sm, @sizeOf(SigmoidMulPush), 0);

        // 5. Batched O projection, accumulate into db.hidden (residual).
        self.gemmDispatch(&cmd, wo, &db.attn_out, &db.hidden, d.n_embd, d.q_dim, B, 1);
        if (self.decode_collapse) self.submit(cmd) else cmd.commitAndWait();
    }

    /// 4c-step-2b SSM block: pre-norm + qkv/z/alpha/beta + out projections run as
    /// GEMMs over ALL B rows (each weight read once), AND the conv1d / delta-net
    /// scan / gated-norm inner is now FUSED into three batched grid.y/z=B kernels
    /// (the genuinely-new hybrid-SSM batched work; gemma had no SSM). Collapses the
    /// per-row decode loop from B per-row `commitAndWait` syncs to ONE command
    /// buffer / ONE commitAndWait (mirrors 4c-step-2a's attention fusion):
    ///   1. batched pre-norm + qkv/z/alpha/beta projections (GEMMs over B rows)
    ///   2. ssm_conv1d_seq — per-(channel,b) depthwise causal conv + SiLU over each
    ///      row's slot conv ring at per-row state_offset = positions[b] % (d_conv-1)
    ///   3. ssm_delta_net_seq — per-(head,row,b) gated delta-net scan over each
    ///      row's slot recurrent state (one decode token per row)
    ///   4. ssm_gated_norm_seq — per-(head,b) gated RMS-norm over token-major rows
    ///   5. batched out projection, accumulate into db.hidden (residual)
    /// Per-element arithmetic is copied verbatim from the per-row kernels (slot/row
    /// bases derived from positions[]/slots[]) → token-identical to N serial runs.
    fn ssmLayerBatchedDecode(self: *ForwardCuda, L: u32, B: u32, db: *DecodeBatch, pos_buf: *const CudaBuffer, slot_buf: *const CudaBuffer) !void {
        const d = self.d;
        const ctx = self.ctx;
        const wan = self.layer(L, "attn_norm.weight");
        const wqkv = self.layer(L, "attn_qkv.weight");
        const wz = self.layer(L, "attn_gate.weight");
        const walpha = self.layer(L, "ssm_alpha.weight");
        const wbeta = self.layer(L, "ssm_beta.weight");
        const wconv = self.layer(L, "ssm_conv1d.weight");
        const wdt = self.layer(L, "ssm_dt.bias");
        const wa = self.layer(L, "ssm_a");
        const wnorm = self.layer(L, "ssm_norm.weight");
        const wout = self.layer(L, "ssm_out.weight");

        var cmd = try command.beginCommand(ctx);

        // 1. Batched pre-norm + qkv / z / alpha / beta projections over B rows.
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ B, 1, 1 }, .{ 256, 1, 1 }, &.{ &db.hidden, &wan.gpu_buffer, &db.norm }, &rms, @sizeOf(RmsPush), 0);
        self.gemmDispatch(&cmd, wqkv, &db.norm, &db.attn_out, d.conv_channels, d.n_embd, B, 0);
        self.gemmDispatch(&cmd, wz, &db.norm, &db.gate, d.d_inner, d.n_embd, B, 0);
        self.gemmDispatch(&cmd, walpha, &db.norm, &db.alpha, d.dt_rank, d.n_embd, B, 0);
        self.gemmDispatch(&cmd, wbeta, &db.norm, &db.beta, d.dt_rank, d.n_embd, B, 0);

        // 2. Batched conv1d over (ceilDiv(conv_channels,64), B): each row reads/writes
        //    its slot conv ring at per-row state_offset = positions[b] % (d_conv-1).
        const conv = ConvSeqPush{ .conv_channels = d.conv_channels, .d_conv = d.d_conv, .kernel_is_f16 = boolU32(wconv.info.type_ == .f16), .conv_state_len = d.conv_state_len };
        cmd.dispatch(&self.pipes.ssm_conv1d_seq, .{ ceilDiv(d.conv_channels, 64), B, 1 }, .{ 64, 1, 1 }, &.{ &db.attn_out, &wconv.gpu_buffer, &self.ssm_conv_slots.?[L], &db.swiglu, pos_buf, slot_buf }, &conv, @sizeOf(ConvSeqPush), 0);

        // 3. Batched delta-net scan over (dt_rank, head_v_dim, B): each row reads/writes
        //    its slot recurrent state (one decode token per row).
        const dn = DeltaNetSeqPush{
            .d_inner = d.d_inner,
            .dt_rank = d.dt_rank,
            .head_v_dim = d.head_v_dim,
            .d_state = d.d_state,
            .n_group = d.n_group,
            .ssm_a_is_f16 = boolU32(wa.info.type_ == .f16),
            .dt_bias_is_f16 = boolU32(wdt.info.type_ == .f16),
            .has_dt_bias = 1,
            .has_ssm_a = 1,
            .conv_stride_tok = d.conv_channels,
            .ab_stride_tok = d.dt_rank,
            .y_stride_tok = d.d_inner,
            .ssm_state_len = d.ssm_state_len,
        };
        cmd.dispatch(&self.pipes.ssm_delta_net_seq, .{ d.dt_rank, d.head_v_dim, B }, .{ d.head_v_dim, 1, 1 }, &.{ &db.swiglu, &wdt.gpu_buffer, &db.alpha, &db.beta, &wa.gpu_buffer, &self.ssm_state_slots.?[L], &db.ssm_delta, slot_buf }, &dn, @sizeOf(DeltaNetSeqPush), 0);

        // 4. Batched gated norm over (dt_rank, B): position/slot-independent.
        const norm_per_head: u32 = if (wnorm.info.numElements() == d.d_inner) 1 else 0;
        const gn = GatedNormSeqPush{ .d_inner = d.d_inner, .dt_rank = d.dt_rank, .head_v_dim = d.head_v_dim, .d_state = d.d_state, .norm_per_head = norm_per_head };
        cmd.dispatch(&self.pipes.ssm_gated_norm_seq, .{ d.dt_rank, B, 1 }, .{ d.head_v_dim, 1, 1 }, &.{ &db.ssm_delta, &db.gate, &wnorm.gpu_buffer, &db.ssm_y }, &gn, @sizeOf(GatedNormSeqPush), 0);

        // 5. Batched out projection, accumulate into db.hidden (residual).
        self.gemmDispatch(&cmd, wout, &db.ssm_y, &db.hidden, d.n_embd, d.d_inner, B, 1);
        if (self.decode_collapse) self.submit(cmd) else cmd.commitAndWait();
    }

    /// 4c dense FFN block: fully batched over B rows (pre-norm + gate/up GEMMs +
    /// SwiGLU + down GEMM accumulate). Position/slot-independent. Mirrors `ffnBlock`.
    fn ffnBlockBatchedDecode(self: *ForwardCuda, L: u32, B: u32, db: *DecodeBatch) !void {
        const d = self.d;
        const ctx = self.ctx;
        const wfn = self.layer(L, "post_attention_norm.weight");
        const wgate = self.layer(L, "ffn_gate.weight");
        const wup = self.layer(L, "ffn_up.weight");
        const wdown = self.layer(L, "ffn_down.weight");

        var cmd = try command.beginCommand(ctx);
        const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ B, 1, 1 }, .{ 256, 1, 1 }, &.{ &db.hidden, &wfn.gpu_buffer, &db.ffn_norm }, &rms, @sizeOf(RmsPush), 0);
        self.gemmDispatch(&cmd, wgate, &db.ffn_norm, &db.gate, d.n_ff, d.n_embd, B, 0);
        self.gemmDispatch(&cmd, wup, &db.ffn_norm, &db.up, d.n_ff, d.n_embd, B, 0);
        const sg = SwigluPush{ .N = B * d.n_ff };
        cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(B * d.n_ff, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &db.gate, &db.up, &db.swiglu }, &sg, @sizeOf(SwigluPush), 0);
        self.gemmDispatch(&cmd, wdown, &db.swiglu, &db.hidden, d.n_embd, d.n_ff, B, 1);
        if (self.decode_collapse) self.submit(cmd) else cmd.commitAndWait();
    }

    /// 4d: batched-decode MoE FFN block (qwen36-35b-a3b). The MoE FFN is STATELESS
    /// (no KV, no SSM recurrent state — it reads `hidden`, routes to experts, and
    /// accumulates the routed + shared-expert output back into `hidden`), so unlike
    /// the attention/SSM blocks it needs NO per-slot state and NO per-seq position.
    /// Each of the B rows is therefore an independent single-token MoE FFN: loop the
    /// rows, aliasing row b's slice of `db.hidden`, and run the production block math
    /// VERBATIM on it via `moeFfnRowDecode` (so batched == N-serial token-identical
    /// by construction). The per-row scratch (`ffn_norm_buf`/`gate_buf`/… and the
    /// host expert-id readback) is shared across rows, so each row runs to
    /// completion (`waitPending`) before the next reuses it. Fusing the per-row MoE
    /// over B rows (router GEMM [B,n_experts] + `build_expert_order` over B·n_used)
    /// is a later throughput sub-step; this first establishes correctness (mirrors
    /// 4b → 4c for attention/SSM). Production `moeFfnBlock`/`decodeStep` UNTOUCHED.
    ///
    /// Effort 28 MoE perf: the SHARED expert (gate/up/down) is a DENSE FFN — the same
    /// weights for every row — so the per-row loop reads them B× redundantly. When
    /// `moe_shared_batched` (default-on), split each row into router+routed-experts
    /// only (`moeFfnRowDecode` → routed accumulate into its `db.hidden` row), then run
    /// the shared expert ONCE batched over all B rows (`moeSharedExpertBatched`,
    /// reading each shared weight once). The shared input is the batched pre-norm
    /// `db.ffn_norm` (computed up front before the routed accumulation overwrites
    /// `db.hidden`), and the per-row routed path reads its own `db.ffn_norm` slice
    /// (so the per-row rms_norm is also collapsed to one launch over B rows). Math is
    /// byte-identical per row to the per-row shared expert (ARGMAX-identical given the
    /// GEMM reduction order) → batched == N-serial token-identical. When off (the A/B
    /// arm), fall back to the original full per-row `moeFfnRowDecode`.
    fn moeFfnBlockBatchedDecode(self: *ForwardCuda, L: u32, B: u32, db: *DecodeBatch) !void {
        const d = self.d;
        const f4 = @sizeOf(f32);
        if (!self.moe_shared_batched) {
            // A/B fallback: the original per-row MoE FFN (router + routed + shared).
            var bi: u32 = 0;
            while (bi < B) : (bi += 1) {
                var hrow = try buffer.aliasBuffer(&db.hidden, bi * d.n_embd * f4, d.n_embd * f4);
                defer buffer.freeBuffer(&hrow);
                try self.moeFfnRowDecode(L, &hrow);
            }
            return;
        }
        // Batched pre-norm for ALL B rows — the shared-expert input AND each routed
        // row's router/expert input. One launch over grid B (was one rms_norm per
        // row inside `moeFfnRowDecode`). Same per-row 256-thread reduction →
        // bit-identical per row. Computed BEFORE the routed loop accumulates into
        // db.hidden, so the shared expert reads the correct pre-norm.
        const wfn = self.layer(L, "post_attention_norm.weight");
        {
            var cmd = try command.beginCommand(self.ctx);
            const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
            cmd.dispatch(&self.pipes.rms_norm, .{ B, 1, 1 }, .{ 256, 1, 1 }, &.{ &db.hidden, &wfn.gpu_buffer, &db.ffn_norm }, &rms, @sizeOf(RmsPush), 0);
            self.submit(cmd); // async on the shared stream; consumed in-order below
        }
        // Per-row router + routed experts: reads row b's db.ffn_norm slice, routes,
        // accumulates the routed (weighted) expert output into row b's db.hidden.
        var bi: u32 = 0;
        while (bi < B) : (bi += 1) {
            var nrow = try buffer.aliasBuffer(&db.ffn_norm, bi * d.n_embd * f4, d.n_embd * f4);
            var hrow = try buffer.aliasBuffer(&db.hidden, bi * d.n_embd * f4, d.n_embd * f4);
            defer {
                buffer.freeBuffer(&nrow);
                buffer.freeBuffer(&hrow);
            }
            try self.moeFfnRoutedRowDecode(L, &nrow, &hrow);
        }
        // Shared expert ONCE over all B rows (each shared weight read once).
        try self.moeSharedExpertBatched(L, B, db);
    }

    /// Batched MoE shared expert over all B rows (the `db.ffn_norm` pre-norm): the
    /// shared gate/up/down are DENSE weights, so they run as ONE `gemmDispatch` each
    /// over the [B, n_embd] norm (reading each weight once) — the
    /// `ffnBlockBatchedDecode` pattern. `swiglu` over the contiguous [B, sf] gate/up,
    /// the per-row sigmoid(gate-logit)·down scale is element-wise so it stays a tiny
    /// per-row dispatch (the weight reads — the cost — are batched). The down output
    /// goes to `db.moe_down` (NOT accumulated by the GEMM) because it must be scaled
    /// by sigmoid(gate logit) before accumulating into db.hidden. Byte-identical math
    /// per row to the per-row shared expert (ARGMAX-identical given GEMM reduction).
    fn moeSharedExpertBatched(self: *ForwardCuda, L: u32, B: u32, db: *DecodeBatch) !void {
        const d = self.d;
        const f4 = @sizeOf(f32);
        const sf = d.shexp_ff;
        const wgs = self.layer(L, "ffn_gate_shexp.weight");
        const wus = self.layer(L, "ffn_up_shexp.weight");
        const wds = self.layer(L, "ffn_down_shexp.weight");
        const wgi = self.layer(L, "ffn_gate_inp_shexp.weight"); // [hidden, 1] F32
        var cmd = try command.beginCommand(self.ctx);
        self.gemmDispatch(&cmd, wgs, &db.ffn_norm, &db.gate, sf, d.n_embd, B, 0);
        self.gemmDispatch(&cmd, wus, &db.ffn_norm, &db.up, sf, d.n_embd, B, 0);
        const sg = SwigluPush{ .N = B * sf };
        cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(B * sf, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &db.gate, &db.up, &db.swiglu }, &sg, @sizeOf(SwigluPush), 0);
        self.gemmDispatch(&cmd, wgi, &db.ffn_norm, &db.moe_gate_scalar, 1, d.n_embd, B, 0);
        self.gemmDispatch(&cmd, wds, &db.swiglu, &db.moe_down, d.n_embd, sf, B, 0);
        const ss = SigmoidAccPush{ .N = d.n_embd };
        var bi: u32 = 0;
        while (bi < B) : (bi += 1) {
            var hrow = try buffer.aliasBuffer(&db.hidden, bi * d.n_embd * f4, d.n_embd * f4);
            var drow = try buffer.aliasBuffer(&db.moe_down, bi * d.n_embd * f4, d.n_embd * f4);
            var srow = try buffer.aliasBuffer(&db.moe_gate_scalar, bi * f4, f4);
            defer {
                buffer.freeBuffer(&hrow);
                buffer.freeBuffer(&drow);
                buffer.freeBuffer(&srow);
            }
            cmd.dispatch(&self.pipes.sigmoid_scale_acc, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &hrow, &drow, &srow }, &ss, @sizeOf(SigmoidAccPush), 0);
        }
        self.submit(cmd);
        self.waitPending();
    }

    /// Router + routed-experts only (NO shared expert), reading the caller's
    /// precomputed `norm_row` (a `db.ffn_norm` slice) instead of recomputing the
    /// per-row rms_norm, and accumulating the routed (weighted) expert output into
    /// `hidden_row`. The shared expert is run batched afterward by
    /// `moeSharedExpertBatched`. Body is the router+routed half of `moeFfnRowDecode`
    /// VERBATIM (same self.* per-row scratch, same `waitPending` so it is safe to
    /// reuse for the next row); split out so the dense shared expert can amortize
    /// across B rows. Used only by `moeFfnBlockBatchedDecode` (decode-batch path).
    fn moeFfnRoutedRowDecode(self: *ForwardCuda, L: u32, norm_row: *const CudaBuffer, hidden_row: *const CudaBuffer) !void {
        const d = self.d;
        const ctx = self.ctx;
        const n_used = d.n_experts_used;
        const ef = d.n_ff; // expert_ff = intermediate_dim
        const wrouter = self.layer(L, "ffn_gate_inp.weight"); // [hidden, n_experts] F32
        const wge = self.layer(L, "ffn_gate_exps.weight"); // stacked [hidden, ef, n_experts]
        const wue = self.layer(L, "ffn_up_exps.weight");
        const wde = self.layer(L, "ffn_down_exps.weight"); // stacked [ef, hidden, n_experts]

        const gate_slice = expertSliceBytes(wge.info.type_, ef, d.n_embd);
        const up_slice = expertSliceBytes(wue.info.type_, ef, d.n_embd);
        const down_slice = expertSliceBytes(wde.info.type_, d.n_embd, ef);
        // GPU-side async experts (no host id readback) when ALL of gate/up/down have
        // a `dmmv_*_experts` kernel. Adding q6_k (this cycle) brings the 5 mixed-quant
        // Q6_K-expert layers onto this path so they ride the launch-collapse too.
        const batched_experts = expertsSupported(wge.info.type_) and expertsSupported(wue.info.type_) and expertsSupported(wde.info.type_);

        // --- Router: logits → top-k softmax (rms_norm precomputed → norm_row). ----
        {
            var cmd = try command.beginCommand(ctx);
            self.dmmvDispatch(&cmd, wrouter, norm_row, &self.router_logits_buf, d.n_experts, d.n_embd, 0, 0);
            const tk = TopkPush{ .n_experts = d.n_experts, .k = n_used };
            cmd.dispatch(&self.pipes.softmax_topk, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &.{ &self.router_logits_buf, &self.router_out_buf }, &tk, @sizeOf(TopkPush), 0);
            if (batched_experts) {
                self.submit(cmd); // async: experts read the ids GPU-side, no readback
            } else {
                cmd.commitAndWait(); // sync: the fallback host-gathers the ids next
                self.drainPending();
            }
        }

        if (!batched_experts) {
            buffer.download(ctx, &self.router_out_buf, std.mem.sliceAsBytes(self.host_router_ids[0..n_used]));
        }

        // --- Routed experts → SwiGLU → down, slot-major into down_buf, then combine.
        {
            var cmd = try command.beginCommand(ctx);
            if (batched_experts) {
                const nrows_gu = n_used * ef;
                self.expertsDispatch(&cmd, wge, norm_row, &self.gate_buf, &self.router_out_buf, ef, d.n_embd, gate_slice, 0, n_used);
                self.expertsDispatch(&cmd, wue, norm_row, &self.up_buf, &self.router_out_buf, ef, d.n_embd, up_slice, 0, n_used);
                const sg = SwigluPush{ .N = nrows_gu };
                cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(nrows_gu, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
                self.expertsDispatch(&cmd, wde, &self.swiglu_buf, &self.down_buf, &self.router_out_buf, d.n_embd, ef, down_slice, ef, n_used);
            } else {
                const sg = SwigluPush{ .N = ef };
                var j: u32 = 0;
                while (j < n_used) : (j += 1) {
                    const id = self.host_router_ids[j];
                    self.dmmvDispatch(&cmd, wge, norm_row, &self.gate_buf, ef, d.n_embd, 0, id * gate_slice);
                    self.dmmvDispatch(&cmd, wue, norm_row, &self.up_buf, ef, d.n_embd, 0, id * up_slice);
                    cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(ef, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
                    const down_push = DmmvPush{ .M = d.n_embd, .K = ef, .acc_mode = 0, .a_offset = id * down_slice, .y_offset = j * d.n_embd * @sizeOf(f32) };
                    const didx = dmmvIdx(wde.info.type_);
                    if (didx < 4) {
                        cmd.dispatch(&self.pipes.dmmv_fast[didx], .{ d.n_embd, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ &wde.gpu_buffer, &self.swiglu_buf, &self.down_buf }, &down_push, @sizeOf(DmmvPush), 0);
                    } else {
                        cmd.dispatch(&self.pipes.dmmv[didx], .{ d.n_embd, 1, 1 }, .{ 256, 1, 1 }, &.{ &wde.gpu_buffer, &self.swiglu_buf, &self.down_buf }, &down_push, @sizeOf(DmmvPush), 0);
                    }
                }
            }
            const ma = MoeAccPush{ .N = d.n_embd, .n_used = n_used, .src_stride = d.n_embd };
            cmd.dispatch(&self.pipes.moe_weighted_acc, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ hidden_row, &self.down_buf, &self.router_out_buf }, &ma, @sizeOf(MoeAccPush), 0);
            self.submit(cmd);
        }
        // LAUNCH-COLLAPSE: the async (`batched_experts`) path is pure-GPU and every
        // command rides the context's ONE shared CUstream (strictly in-order), so
        // the next row's writes to the shared scratch (router_*/gate_buf/up_buf/
        // swiglu_buf/down_buf) serialize AFTER this row's reads — no per-row host
        // sync is needed. The B rows' commands queue back-to-back and are drained
        // together by the layer tail (`moeSharedExpertBatched`'s `waitPending`),
        // removing B blocking CPU↔GPU round-trips per MoE layer per token (the
        // boost-starved qwen MoE decode is launch-bound). The host-id FALLBACK
        // path STILL syncs per row — it downloads `router_out_buf` to host-gather
        // the expert ids and reuses the host `host_router_ids` scratch next row.
        if (!batched_experts or !self.moe_collapse) self.waitPending();
    }

    /// Per-row MoE FFN: the production `moeFfnBlock` body VERBATIM but reading/
    /// accumulating into the caller's `hidden_row` (a `db.hidden` row alias) instead
    /// of the single-seq `self.hidden`, and run SYNCHRONOUSLY (`waitPending` at the
    /// end) so the shared per-row scratch is safe to reuse for the next row. Keeping
    /// this a separate verbatim copy leaves production `moeFfnBlock` byte-identical.
    fn moeFfnRowDecode(self: *ForwardCuda, L: u32, hidden_row: *const CudaBuffer) !void {
        const d = self.d;
        const ctx = self.ctx;
        const n_used = d.n_experts_used;
        const ef = d.n_ff; // expert_ff = intermediate_dim
        const wfn = self.layer(L, "post_attention_norm.weight");
        const wrouter = self.layer(L, "ffn_gate_inp.weight"); // [hidden, n_experts] F32
        const wge = self.layer(L, "ffn_gate_exps.weight"); // stacked [hidden, ef, n_experts]
        const wue = self.layer(L, "ffn_up_exps.weight");
        const wde = self.layer(L, "ffn_down_exps.weight"); // stacked [ef, hidden, n_experts]

        const gate_slice = expertSliceBytes(wge.info.type_, ef, d.n_embd);
        const up_slice = expertSliceBytes(wue.info.type_, ef, d.n_embd);
        const down_slice = expertSliceBytes(wde.info.type_, d.n_embd, ef);
        const batched_experts = dmmvIdx(wge.info.type_) == 0 and dmmvIdx(wue.info.type_) == 0 and dmmvIdx(wde.info.type_) == 1;

        // --- Router: rms_norm → logits → top-k softmax. -----------------------
        {
            var cmd = try command.beginCommand(ctx);
            const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ hidden_row, &wfn.gpu_buffer, &self.ffn_norm_buf }, &rms, @sizeOf(RmsPush), 0);
            self.dmmvDispatch(&cmd, wrouter, &self.ffn_norm_buf, &self.router_logits_buf, d.n_experts, d.n_embd, 0, 0);
            const tk = TopkPush{ .n_experts = d.n_experts, .k = n_used };
            cmd.dispatch(&self.pipes.softmax_topk, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &.{ &self.router_logits_buf, &self.router_out_buf }, &tk, @sizeOf(TopkPush), 0);
            if (batched_experts) {
                self.submit(cmd); // async: experts read the ids GPU-side, no readback
            } else {
                cmd.commitAndWait(); // sync: the fallback host-gathers the ids next
                self.drainPending();
            }
        }

        if (!batched_experts) {
            buffer.download(ctx, &self.router_out_buf, std.mem.sliceAsBytes(self.host_router_ids[0..n_used]));
        }

        // --- Routed experts → SwiGLU → down, slot-major into down_buf.
        {
            var cmd = try command.beginCommand(ctx);
            if (batched_experts) {
                const nrows_gu = n_used * ef;
                const pg = ExpertsPush{ .M = ef, .K = d.n_embd, .slice = gate_slice, .x_stride = 0, .n_used = n_used };
                cmd.dispatch(&self.pipes.dmmv_q4k_experts, .{ nrows_gu, 1, 1 }, .{ 64, 1, 1 }, &.{ &wge.gpu_buffer, &self.ffn_norm_buf, &self.gate_buf, &self.router_out_buf }, &pg, @sizeOf(ExpertsPush), 0);
                const pu = ExpertsPush{ .M = ef, .K = d.n_embd, .slice = up_slice, .x_stride = 0, .n_used = n_used };
                cmd.dispatch(&self.pipes.dmmv_q4k_experts, .{ nrows_gu, 1, 1 }, .{ 64, 1, 1 }, &.{ &wue.gpu_buffer, &self.ffn_norm_buf, &self.up_buf, &self.router_out_buf }, &pu, @sizeOf(ExpertsPush), 0);
                const sg = SwigluPush{ .N = nrows_gu };
                cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(nrows_gu, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
                const pd = ExpertsPush{ .M = d.n_embd, .K = ef, .slice = down_slice, .x_stride = ef, .n_used = n_used };
                cmd.dispatch(&self.pipes.dmmv_q5k_experts, .{ n_used * d.n_embd, 1, 1 }, .{ 64, 1, 1 }, &.{ &wde.gpu_buffer, &self.swiglu_buf, &self.down_buf, &self.router_out_buf }, &pd, @sizeOf(ExpertsPush), 0);
            } else {
                const sg = SwigluPush{ .N = ef };
                var j: u32 = 0;
                while (j < n_used) : (j += 1) {
                    const id = self.host_router_ids[j];
                    self.dmmvDispatch(&cmd, wge, &self.ffn_norm_buf, &self.gate_buf, ef, d.n_embd, 0, id * gate_slice);
                    self.dmmvDispatch(&cmd, wue, &self.ffn_norm_buf, &self.up_buf, ef, d.n_embd, 0, id * up_slice);
                    cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(ef, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
                    const down_push = DmmvPush{ .M = d.n_embd, .K = ef, .acc_mode = 0, .a_offset = id * down_slice, .y_offset = j * d.n_embd * @sizeOf(f32) };
                    const didx = dmmvIdx(wde.info.type_);
                    if (didx < 4) {
                        cmd.dispatch(&self.pipes.dmmv_fast[didx], .{ d.n_embd, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ &wde.gpu_buffer, &self.swiglu_buf, &self.down_buf }, &down_push, @sizeOf(DmmvPush), 0);
                    } else {
                        cmd.dispatch(&self.pipes.dmmv[didx], .{ d.n_embd, 1, 1 }, .{ 256, 1, 1 }, &.{ &wde.gpu_buffer, &self.swiglu_buf, &self.down_buf }, &down_push, @sizeOf(DmmvPush), 0);
                    }
                }
            }
            const ma = MoeAccPush{ .N = d.n_embd, .n_used = n_used, .src_stride = d.n_embd };
            cmd.dispatch(&self.pipes.moe_weighted_acc, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ hidden_row, &self.down_buf, &self.router_out_buf }, &ma, @sizeOf(MoeAccPush), 0);
            self.submit(cmd);
        }

        // --- Shared expert: gate/up/SwiGLU/down, scaled by sigmoid(gate logit).
        {
            const wgs = self.layer(L, "ffn_gate_shexp.weight");
            const wus = self.layer(L, "ffn_up_shexp.weight");
            const wds = self.layer(L, "ffn_down_shexp.weight");
            const wgi = self.layer(L, "ffn_gate_inp_shexp.weight"); // [hidden, 1] F32
            const sf = d.shexp_ff;
            var cmd = try command.beginCommand(ctx);
            self.dmmvDispatch(&cmd, wgs, &self.ffn_norm_buf, &self.gate_buf, sf, d.n_embd, 0, 0);
            self.dmmvDispatch(&cmd, wus, &self.ffn_norm_buf, &self.up_buf, sf, d.n_embd, 0, 0);
            const sg = SwigluPush{ .N = sf };
            cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(sf, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
            self.dmmvDispatch(&cmd, wgi, &self.ffn_norm_buf, &self.gate_scalar_buf, 1, d.n_embd, 0, 0);
            self.dmmvDispatch(&cmd, wds, &self.swiglu_buf, &self.down_buf, d.n_embd, sf, 0, 0);
            const ss = SigmoidAccPush{ .N = d.n_embd };
            cmd.dispatch(&self.pipes.sigmoid_scale_acc, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ hidden_row, &self.down_buf, &self.gate_scalar_buf }, &ss, @sizeOf(SigmoidAccPush), 0);
            self.submit(cmd);
        }
        self.waitPending(); // sync: shared per-row scratch reused by the next row
    }

    /// Slot-state variant of `attentionLayer`: identical block math, but the KV
    /// cache read/write target sequence `slot`'s region of the slot KV (aliased to
    /// its base), and the kernels run SYNCHRONOUSLY (commitAndWait) so the per-row
    /// slot aliases are safe to free on return. The kv-write/attention offsets are
    /// pos-relative within the slot's aliased region — byte-for-byte the same
    /// values as the single-seq path. 4c fuses this into a batched per-seq kernel.
    fn attentionLayerSlot(self: *ForwardCuda, L: u32, pos: u32, slot: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const f4 = @sizeOf(f32);
        const wq = self.layer(L, "attn_q.weight"); // packed [Q | gate], M = 2*q_dim
        const wk = self.layer(L, "attn_k.weight");
        const wv = self.layer(L, "attn_v.weight");
        const wqn = self.layer(L, "attn_q_norm.weight");
        const wkn = self.layer(L, "attn_k_norm.weight");
        const wo = self.layer(L, "attn_output.weight");
        const wan = self.layer(L, "attn_norm.weight");

        // Alias sequence `slot`'s KV region to its base; pos-relative offsets below.
        const kv_base = @as(usize, slot) * self.slot_ctx * d.kv_dim * f4;
        const kv_span = @as(usize, self.slot_ctx) * d.kv_dim * f4;
        var kk = try buffer.aliasBuffer(&self.kv_k_slots.?[L], kv_base, kv_span);
        var vv = try buffer.aliasBuffer(&self.kv_v_slots.?[L], kv_base, kv_span);
        defer {
            buffer.freeBuffer(&kk);
            buffer.freeBuffer(&vv);
        }

        var cmd = try command.beginCommand(ctx);
        const norm_q8_ready = self.rmsNormDecodeDispatch(&cmd, &self.hidden, &wan.gpu_buffer, &self.norm_buf, d.n_embd, self.use_decode_q8_q4 and d.n_experts == 0 and wq.info.type_ == .q4_k);
        self.dmmvPreparedQ8Dispatch(&cmd, wq, &self.norm_buf, &self.qfull_buf, 2 * d.q_dim, d.n_embd, 0, 0, norm_q8_ready);
        if (self.use_fused_q4_pairs and wk.info.type_ == .q4_k and wv.info.type_ == .q4_k) {
            self.dmmvQ4PairDispatch(&cmd, &wk.gpu_buffer, &wv.gpu_buffer, &self.norm_buf, &self.k_buf, &self.v_buf, d.kv_dim, d.kv_dim, d.n_embd, norm_q8_ready);
        } else {
            self.dmmvPreparedQ8Dispatch(&cmd, wk, &self.norm_buf, &self.k_buf, d.kv_dim, d.n_embd, 0, 0, norm_q8_ready);
            self.dmmvPreparedQ8Dispatch(&cmd, wv, &self.norm_buf, &self.v_buf, d.kv_dim, d.n_embd, 0, 0, norm_q8_ready);
        }
        self.attentionFrontendDispatch(&cmd, &wqn.gpu_buffer, &wkn.gpu_buffer, &kk, &vv, pos);
        const seq_len = pos + 1;
        const attn = AttnPush{ .head_dim = d.head_dim, .n_heads = d.n_head, .n_kv_heads = d.n_kv_head, .seq_len = seq_len, .attn_scale_bits = 0, .sink_offset = L * d.n_head };
        const attn_smem: u32 = seq_len * 4;
        cmd.dispatch(&self.pipes.naive_attention, .{ d.n_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.q_buf, &kk, &vv, &self.sinks, &self.attn_out_buf }, &attn, @sizeOf(AttnPush), attn_smem);
        const o_q8_ready = self.decodePreparedQ8Eligible(wo.info.type_, d.n_embd, d.q_dim);
        if (o_q8_ready) {
            const qp = QuantActPush{ .K = d.q_dim, .T = 1 };
            cmd.dispatch(&self.pipes.sigmoid_mul_quant_q8, .{ ceilDiv(d.q_dim, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &self.attn_out_buf, &self.gate_buf, &self.batch.?.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
            self.dmmvPreparedQ8Dispatch(&cmd, wo, &self.attn_out_buf, &self.hidden, d.n_embd, d.q_dim, 1, 0, true);
        } else {
            const sm = SigmoidMulPush{ .N = d.q_dim };
            cmd.dispatch(&self.pipes.sigmoid_mul, .{ ceilDiv(d.q_dim, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.attn_out_buf, &self.gate_buf, &self.attn_out_buf }, &sm, @sizeOf(SigmoidMulPush), 0);
            self.dmmvDispatch(&cmd, wo, &self.attn_out_buf, &self.hidden, d.n_embd, d.q_dim, 1, 0);
        }
        self.submit(cmd); // parent slot allocations outlive the queued alias pointers
    }

    /// Slot-state variant of `ssmLayer`: identical block math, but the conv ring
    /// + recurrent state read/write sequence `slot`'s aliased region, and the conv
    /// ring offset is DERIVED from the position (`pos % (d_conv-1)`) instead of the
    /// per-layer host counter — production advances `conv_off[L]` one step per
    /// token, so at position p it equals `p % (d_conv-1)`; deriving it makes each
    /// slot's conv offset self-consistent without extra per-slot host state. Runs
    /// SYNCHRONOUSLY so the per-row aliases are safe to free on return.
    fn ssmLayerSlot(self: *ForwardCuda, L: u32, pos: u32, slot: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const f4 = @sizeOf(f32);
        const wan = self.layer(L, "attn_norm.weight");
        const wqkv = self.layer(L, "attn_qkv.weight");
        const wz = self.layer(L, "attn_gate.weight");
        const walpha = self.layer(L, "ssm_alpha.weight");
        const wbeta = self.layer(L, "ssm_beta.weight");
        const wconv = self.layer(L, "ssm_conv1d.weight");
        const wdt = self.layer(L, "ssm_dt.bias");
        const wa = self.layer(L, "ssm_a");
        const wnorm = self.layer(L, "ssm_norm.weight");
        const wout = self.layer(L, "ssm_out.weight");

        // Alias sequence `slot`'s conv ring + recurrent state to their bases.
        var convst = try buffer.aliasBuffer(&self.ssm_conv_slots.?[L], self.slotConvOffsetBytes(slot), d.conv_state_len * f4);
        var recst = try buffer.aliasBuffer(&self.ssm_state_slots.?[L], self.slotStateOffsetBytes(slot), d.ssm_state_len * f4);
        defer {
            buffer.freeBuffer(&convst);
            buffer.freeBuffer(&recst);
        }
        const conv_off = pos % (d.d_conv - 1);

        var cmd = try command.beginCommand(ctx);
        const fused_q4_pair = self.use_fused_q4_pairs and wqkv.info.type_ == .q4_k and wz.info.type_ == .q4_k;
        const want_pair_q8 = fused_q4_pair and self.use_decode_q8_q4_pair and d.n_experts == 0;
        const want_individual_q8 = !fused_q4_pair and
            self.decodePreparedQ8Eligible(wqkv.info.type_, d.conv_channels, d.n_embd) and
            self.decodePreparedQ8Eligible(wz.info.type_, d.d_inner, d.n_embd);
        const norm_q8_ready = self.rmsNormDecodeDispatch(&cmd, &self.hidden, &wan.gpu_buffer, &self.norm_buf, d.n_embd, want_pair_q8 or want_individual_q8);
        if (fused_q4_pair) {
            self.dmmvQ4PairDispatch(&cmd, &wqkv.gpu_buffer, &wz.gpu_buffer, &self.norm_buf, &self.attn_out_buf, &self.gate_buf, d.conv_channels, d.d_inner, d.n_embd, norm_q8_ready);
        } else {
            self.dmmvPreparedQ8Dispatch(&cmd, wqkv, &self.norm_buf, &self.attn_out_buf, d.conv_channels, d.n_embd, 0, 0, norm_q8_ready);
            self.dmmvPreparedQ8Dispatch(&cmd, wz, &self.norm_buf, &self.gate_buf, d.d_inner, d.n_embd, 0, 0, norm_q8_ready);
        }
        if (self.use_fused_ssm_f32 and walpha.info.type_ == .f32 and wbeta.info.type_ == .f32) {
            const ab = DmmvPush{ .M = d.dt_rank, .K = d.n_embd };
            cmd.dispatch(&self.pipes.dmmv_f32_dual, .{ d.dt_rank, 1, 1 }, .{ 256, 1, 1 }, &.{ &walpha.gpu_buffer, &wbeta.gpu_buffer, &self.norm_buf, &self.router_buf, &self.down_buf }, &ab, @sizeOf(DmmvPush), 0);
        } else {
            self.dmmvDispatch(&cmd, walpha, &self.norm_buf, &self.router_buf, d.dt_rank, d.n_embd, 0, 0);
            self.dmmvDispatch(&cmd, wbeta, &self.norm_buf, &self.down_buf, d.dt_rank, d.n_embd, 0, 0);
        }
        const conv_prepared = is_rocm and self.use_decode_ssm_col_warp and
            d.d_state == d.head_v_dim and d.d_state <= 128 and (d.d_state & 31) == 0;
        if (conv_prepared) {
            const cp = ConvPreparePush{
                .conv_channels = d.conv_channels,
                .d_conv = d.d_conv,
                .kernel_is_f16 = boolU32(wconv.info.type_ == .f16),
                .state_offset = conv_off,
                .dt_rank = d.dt_rank,
                .d_inner = d.d_inner,
                .d_state = d.d_state,
                .n_group = d.n_group,
                .ssm_a_is_f16 = boolU32(wa.info.type_ == .f16),
                .dt_bias_is_f16 = boolU32(wdt.info.type_ == .f16),
            };
            cmd.dispatch(&self.pipes.ssm_conv1d_prepare, .{ @max(d.n_group, ceilDiv(d.d_inner, d.d_state)), 1, 1 }, .{ d.d_state, 1, 1 }, &.{ &self.attn_out_buf, &wconv.gpu_buffer, &convst, &self.swiglu_buf, &wdt.gpu_buffer, &self.router_buf, &self.down_buf, &wa.gpu_buffer }, &cp, @sizeOf(ConvPreparePush), 0);
        } else {
            const conv = ConvPush{ .conv_channels = d.conv_channels, .d_conv = d.d_conv, .kernel_is_f16 = boolU32(wconv.info.type_ == .f16), .state_offset = conv_off };
            cmd.dispatch(&self.pipes.ssm_conv1d, .{ ceilDiv(d.conv_channels, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.attn_out_buf, &wconv.gpu_buffer, &convst, &self.swiglu_buf }, &conv, @sizeOf(ConvPush), 0);
        }
        self.deltaNetDecodeDispatch(&cmd, wdt, wa, &recst, conv_prepared);
        const norm_per_head: u32 = if (wnorm.info.numElements() == d.d_inner) 1 else 0;
        const gn = GatedNormPush{ .d_inner = d.d_inner, .dt_rank = d.dt_rank, .head_v_dim = d.head_v_dim, .d_state = d.d_state, .norm_per_head = norm_per_head };
        const out_q8_ready = (d.head_v_dim & 31) == 0 and self.decodePreparedQ8Eligible(wout.info.type_, d.n_embd, d.d_inner);
        if (out_q8_ready) {
            cmd.dispatch(&self.pipes.ssm_gated_norm_quant_q8, .{ d.dt_rank, 1, 1 }, .{ d.head_v_dim, 1, 1 }, &.{ &self.attn_out_buf, &self.gate_buf, &wnorm.gpu_buffer, &self.batch.?.act_q8 }, &gn, @sizeOf(GatedNormPush), 0);
            self.dmmvPreparedQ8Dispatch(&cmd, wout, &self.swiglu_buf, &self.hidden, d.n_embd, d.d_inner, 1, 0, true);
        } else {
            cmd.dispatch(&self.pipes.ssm_gated_norm, .{ d.dt_rank, 1, 1 }, .{ d.head_v_dim, 1, 1 }, &.{ &self.attn_out_buf, &self.gate_buf, &wnorm.gpu_buffer, &self.swiglu_buf }, &gn, @sizeOf(GatedNormPush), 0);
            self.dmmvDispatch(&cmd, wout, &self.swiglu_buf, &self.hidden, d.n_embd, d.d_inner, 1, 0);
        }
        self.submit(cmd); // parent slot allocations outlive the queued alias pointers
    }

    /// Dense decode step via CUDA-graph replay (Effort 25). `hidden` already holds
    /// the embedded token. Stream-captures the layer chain + tail into the cached
    /// CUgraphExec and launches it as one submission, eliminating the per-kernel
    /// launch + inter-kernel-bubble latency of the ~480-kernel launch-bound chain.
    /// While `capturing`, `submit` and the attention kernel take a no-sync,
    /// shape-invariant path so the captured topology is identical every step
    /// (only per-token push-constant scalars change → cheap in-place exec update).
    fn decodeStepGraph(self: *ForwardCuda, pos: u32, out_norm: *const CudaBuffer, lm_head: *const CudaBuffer, lm_type: gguf.GGMLType) !u32 {
        const d = self.d;
        const ctx = self.ctx;

        self.capturing = true;
        _ = shim.cuda_graph_begin(ctx);
        // EMBED as the graph's first node(s). GPU path: a 4-byte token-id H2D then
        // the embed_lookup dispatch dequants the row into `hidden` on-device (the
        // per-token CPU dequant is gone entirely). CPU fallback: H2D the pre-
        // dequantized pinned host_embed row. Either way it rides the one launch.
        if (self.embed_gpu) {
            buffer.uploadAsync(ctx, &self.tok_in_buf, std.mem.sliceAsBytes(self.host_tok_in));
            try self.recordEmbed();
        } else {
            buffer.uploadAsync(ctx, &self.hidden, std.mem.sliceAsBytes(self.host_embed));
        }
        var L: u32 = 0;
        while (L < d.n_layers) : (L += 1) {
            if (self.layer_is_attn[L]) {
                try self.attentionLayer(L, pos);
            } else {
                try self.ssmLayer(L);
            }
            // MoE path is captured too when every layer is batched (moeGraphCapturable
            // gated this at setup); the batched expert path is a static stream.
            if (d.n_experts > 0) try self.moeFfnBlock(L) else try self.ffnBlock(L);
        }
        var cmd = try command.beginCommand(ctx);
        self.tailDispatch(&cmd, out_norm, lm_head, lm_type);
        cmd.releaseCompleted(); // captured onto the stream; no event record / no sync
        // ARGMAX D2H as the graph's last node (pinned host_tok), so the predicted
        // token lands in host memory after the single graph-launch sync — no extra
        // per-token download sync.
        buffer.downloadAsync(ctx, &self.argmax_buf, std.mem.sliceAsBytes(self.host_tok));
        self.capturing = false;

        // End capture, instantiate-or-update the cached exec, launch + ONE sync
        // (drains the embed H2D, layer chain, tail, and argmax D2H together).
        _ = shim.cuda_graph_end_launch(ctx, self.graph.?);

        return self.host_tok[0];
    }

    /// Dispatch the GPU-side embedding lookup (Q4_K): dequant the token's row
    /// (id in `tok_in_buf`) into `hidden`. Routed through `submit` so it captures
    /// cleanly as the graph's first compute node and otherwise rides the async
    /// ring. One block per 256-superblock, 256 threads (one output each).
    fn recordEmbed(self: *ForwardCuda) !void {
        const d = self.d;
        var cmd = try command.beginCommand(self.ctx);
        const push = EmbedPush{ .K = d.n_embd, .vocab = d.vocab };
        const nsb = d.n_embd / 256;
        cmd.dispatch(&self.pipes.embed_q4k, .{ nsb, 1, 1 }, .{ 256, 1, 1 }, &.{ self.embed_weight.?, &self.tok_in_buf, &self.hidden }, &push, @sizeOf(EmbedPush), 0);
        self.submit(cmd);
    }

    fn lmHeadDispatch(self: *ForwardCuda, cmd: *command.CudaCommand, lm_head: *const CudaBuffer, lm_type: gguf.GGMLType, q8_ready: bool) void {
        const d = self.d;
        const lm = DmmvPush{ .M = d.vocab, .K = d.n_embd };
        const lm_idx = dmmvIdx(lm_type);
        if (self.use_decode_q8_lm and lm_type == .q6_k and self.batch != null and (d.n_embd & 255) == 0) {
            if (!q8_ready) {
                const qp = QuantActPush{ .K = d.n_embd, .T = 1 };
                cmd.dispatch(&self.pipes.quantize_act_q8, .{ ceilDiv(d.n_embd, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &self.norm_buf, &self.batch.?.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
            }
            cmd.dispatch(&self.pipes.dmmv_q6k_q8_fast, .{ d.vocab, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ lm_head, &self.batch.?.act_q8, &self.logits_buf }, &lm, @sizeOf(DmmvPush), 0);
        } else if (lm_idx < 4) {
            cmd.dispatch(&self.pipes.dmmv_fast[lm_idx], .{ d.vocab, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ lm_head, &self.norm_buf, &self.logits_buf }, &lm, @sizeOf(DmmvPush), 0);
        } else {
            cmd.dispatch(&self.pipes.dmmv[lm_idx], .{ d.vocab, 1, 1 }, .{ 256, 1, 1 }, &.{ lm_head, &self.norm_buf, &self.logits_buf }, &lm, @sizeOf(DmmvPush), 0);
        }
    }

    /// Record the decode tail (final rms_norm → LM head matvec → argmax) onto
    /// `cmd`. Shared by the async and graph-capture paths.
    fn tailDispatch(self: *ForwardCuda, cmd: *command.CudaCommand, out_norm: *const CudaBuffer, lm_head: *const CudaBuffer, lm_type: gguf.GGMLType) void {
        const d = self.d;
        const want_q8 = self.use_decode_q8_lm and lm_type == .q6_k;
        const norm_q8_ready = self.rmsNormDecodeDispatch(cmd, &self.hidden, out_norm, &self.norm_buf, d.n_embd, want_q8);
        self.lmHeadDispatch(cmd, lm_head, lm_type, norm_q8_ready);
        self.argmaxDispatch(cmd, &self.argmax_buf);
    }

    /// Record the qwen attention front-end.  The fused ROCm path replaces six
    /// tiny launch-bound kernels with one block-per-head kernel while preserving
    /// the standalone reduction order and the exact cache layout.
    fn attentionFrontendDispatch(
        self: *ForwardCuda,
        cmd: *command.CudaCommand,
        wqn: *const CudaBuffer,
        wkn: *const CudaBuffer,
        kk: *const CudaBuffer,
        vv: *const CudaBuffer,
        pos: u32,
    ) void {
        const d = self.d;
        if (self.use_fused_attn_frontend) {
            const qkv = QwenQkvPush{
                .head_dim = d.head_dim,
                .eps = d.rms_eps,
                .rope_dim = d.rope_dim,
                .n_head = d.n_head,
                .n_kv_head = d.n_kv_head,
                .position = pos,
                .kv_offset = pos * d.kv_dim,
            };
            cmd.dispatch(
                &self.pipes.qwen_norm_rope_qkv,
                .{ d.n_head + 2 * d.n_kv_head, 1, 1 },
                .{ 256, 1, 1 },
                &.{ &self.qfull_buf, &self.k_buf, &self.v_buf, wqn, wkn, &self.inv_freq, &self.q_buf, &self.gate_buf, kk, vv },
                &qkv,
                @sizeOf(QwenQkvPush),
                d.head_dim * @sizeOf(f32),
            );
            return;
        }

        const deint = DeintPush{ .head_dim = d.head_dim, .n_head = d.n_head };
        cmd.dispatch(&self.pipes.deinterleave, .{ ceilDiv(d.q_dim, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &self.qfull_buf, &self.q_buf, &self.gate_buf }, &deint, @sizeOf(DeintPush), 0);
        const rms_h = RmsPush{ .N = d.head_dim, .eps = d.rms_eps };
        cmd.dispatch(&self.pipes.rms_norm, .{ d.n_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.q_buf, wqn, &self.q_buf }, &rms_h, @sizeOf(RmsPush), 0);
        cmd.dispatch(&self.pipes.rms_norm, .{ d.n_kv_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.k_buf, wkn, &self.k_buf }, &rms_h, @sizeOf(RmsPush), 0);
        const rope_q = RopePush{ .stride = d.head_dim, .rope_dim = d.rope_dim, .n_heads = d.n_head, .position = pos, .freq_base_bits = 0, .attn_scale_bits = 0 };
        cmd.dispatch(&self.pipes.rope, .{ d.n_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.q_buf, &self.q_buf, &self.inv_freq }, &rope_q, @sizeOf(RopePush), 0);
        const rope_k = RopePush{ .stride = d.head_dim, .rope_dim = d.rope_dim, .n_heads = d.n_kv_head, .position = pos, .freq_base_bits = 0, .attn_scale_bits = 0 };
        cmd.dispatch(&self.pipes.rope, .{ d.n_kv_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.k_buf, &self.k_buf, &self.inv_freq }, &rope_k, @sizeOf(RopePush), 0);
        const kvw = KvWritePush{ .kv_dim = d.kv_dim, .dst_offset = pos * d.kv_dim };
        cmd.dispatch(&self.pipes.kv_cache_write, .{ ceilDiv(d.kv_dim, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.k_buf, kk, &self.v_buf, vv }, &kvw, @sizeOf(KvWritePush), 0);
    }

    // ---- per-block builders -------------------------------------------------

    fn attentionLayer(self: *ForwardCuda, L: u32, pos: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const wq = self.layer(L, "attn_q.weight"); // packed [Q | gate], M = 2*q_dim
        const wk = self.layer(L, "attn_k.weight");
        const wv = self.layer(L, "attn_v.weight");
        const wqn = self.layer(L, "attn_q_norm.weight");
        const wkn = self.layer(L, "attn_k_norm.weight");
        const wo = self.layer(L, "attn_output.weight");
        const wan = self.layer(L, "attn_norm.weight");

        var cmd = try command.beginCommand(ctx);
        const norm_q8_ready = self.rmsNormDecodeDispatch(&cmd, &self.hidden, &wan.gpu_buffer, &self.norm_buf, d.n_embd, self.use_decode_q8_q4 and d.n_experts == 0 and wq.info.type_ == .q4_k);
        // Q+gate, K, V projections from norm_buf.
        // qwen35 packs the attention gate INTO attn_q: wq outputs 2*q_dim, laid out
        // per head as [Q(head_dim) | gate(head_dim)] interleaved across heads
        // ([Q0,g0,Q1,g1,...]). Project the full 2*q_dim, then deinterleave into
        // contiguous q_buf (the Q halves) and gate_buf (the gate halves).
        self.dmmvPreparedQ8Dispatch(&cmd, wq, &self.norm_buf, &self.qfull_buf, 2 * d.q_dim, d.n_embd, 0, 0, norm_q8_ready);
        if (self.use_fused_q4_pairs and wk.info.type_ == .q4_k and wv.info.type_ == .q4_k) {
            self.dmmvQ4PairDispatch(&cmd, &wk.gpu_buffer, &wv.gpu_buffer, &self.norm_buf, &self.k_buf, &self.v_buf, d.kv_dim, d.kv_dim, d.n_embd, norm_q8_ready);
        } else {
            self.dmmvPreparedQ8Dispatch(&cmd, wk, &self.norm_buf, &self.k_buf, d.kv_dim, d.n_embd, 0, 0, norm_q8_ready);
            self.dmmvPreparedQ8Dispatch(&cmd, wv, &self.norm_buf, &self.v_buf, d.kv_dim, d.n_embd, 0, 0, norm_q8_ready);
        }
        self.attentionFrontendDispatch(&cmd, &wqn.gpu_buffer, &wkn.gpu_buffer, &self.kv_k[L], &self.kv_v[L], pos);
        // attention: out → attn_out_buf
        const seq_len = pos + 1;
        const attn = AttnPush{ .head_dim = d.head_dim, .n_heads = d.n_head, .n_kv_heads = d.n_kv_head, .seq_len = seq_len, .attn_scale_bits = 0, .sink_offset = L * d.n_head };
        // naive_attention's dynamic shared mem holds `seq_len` f32 scores. Under
        // graph capture, request the MAX (max_ctx) so the kernel node's shared-mem
        // size is constant across steps — keeping the captured topology invariant
        // (only push scalars change) so the exec updates in place instead of
        // re-instantiating each token. The kernel still uses only seq_len entries.
        const attn_smem: u32 = if (self.capturing) self.max_ctx * 4 else seq_len * 4;
        cmd.dispatch(&self.pipes.naive_attention, .{ d.n_head, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.q_buf, &self.kv_k[L], &self.kv_v[L], &self.sinks, &self.attn_out_buf }, &attn, @sizeOf(AttnPush), attn_smem);
        // gate: attn_out *= sigmoid(gate)
        const o_q8_ready = self.decodePreparedQ8Eligible(wo.info.type_, d.n_embd, d.q_dim);
        if (o_q8_ready) {
            const qp = QuantActPush{ .K = d.q_dim, .T = 1 };
            cmd.dispatch(&self.pipes.sigmoid_mul_quant_q8, .{ ceilDiv(d.q_dim, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &self.attn_out_buf, &self.gate_buf, &self.batch.?.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
            self.dmmvPreparedQ8Dispatch(&cmd, wo, &self.attn_out_buf, &self.hidden, d.n_embd, d.q_dim, 1, 0, true);
        } else {
            const sm = SigmoidMulPush{ .N = d.q_dim };
            cmd.dispatch(&self.pipes.sigmoid_mul, .{ ceilDiv(d.q_dim, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.attn_out_buf, &self.gate_buf, &self.attn_out_buf }, &sm, @sizeOf(SigmoidMulPush), 0);
            // O projection, accumulate into hidden
            self.dmmvDispatch(&cmd, wo, &self.attn_out_buf, &self.hidden, d.n_embd, d.q_dim, 1, 0);
        }
        self.submit(cmd);
    }

    fn ssmLayer(self: *ForwardCuda, L: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const wan = self.layer(L, "attn_norm.weight");
        const wqkv = self.layer(L, "attn_qkv.weight");
        const wz = self.layer(L, "attn_gate.weight");
        const walpha = self.layer(L, "ssm_alpha.weight");
        const wbeta = self.layer(L, "ssm_beta.weight");
        const wconv = self.layer(L, "ssm_conv1d.weight");
        const wdt = self.layer(L, "ssm_dt.bias");
        const wa = self.layer(L, "ssm_a");
        const wnorm = self.layer(L, "ssm_norm.weight");
        const wout = self.layer(L, "ssm_out.weight");

        var cmd = try command.beginCommand(ctx);
        const fused_q4_pair = self.use_fused_q4_pairs and wqkv.info.type_ == .q4_k and wz.info.type_ == .q4_k;
        const want_pair_q8 = fused_q4_pair and self.use_decode_q8_q4_pair and d.n_experts == 0;
        const want_individual_q8 = !fused_q4_pair and
            self.decodePreparedQ8Eligible(wqkv.info.type_, d.conv_channels, d.n_embd) and
            self.decodePreparedQ8Eligible(wz.info.type_, d.d_inner, d.n_embd);
        const norm_q8_ready = self.rmsNormDecodeDispatch(&cmd, &self.hidden, &wan.gpu_buffer, &self.norm_buf, d.n_embd, want_pair_q8 or want_individual_q8);
        // qkv (conv_channels) and z-gate (d_inner)
        if (fused_q4_pair) {
            self.dmmvQ4PairDispatch(&cmd, &wqkv.gpu_buffer, &wz.gpu_buffer, &self.norm_buf, &self.attn_out_buf, &self.gate_buf, d.conv_channels, d.d_inner, d.n_embd, norm_q8_ready);
        } else {
            self.dmmvPreparedQ8Dispatch(&cmd, wqkv, &self.norm_buf, &self.attn_out_buf, d.conv_channels, d.n_embd, 0, 0, norm_q8_ready);
            self.dmmvPreparedQ8Dispatch(&cmd, wz, &self.norm_buf, &self.gate_buf, d.d_inner, d.n_embd, 0, 0, norm_q8_ready);
        }
        // alpha, beta (dt_rank)
        if (self.use_fused_ssm_f32 and walpha.info.type_ == .f32 and wbeta.info.type_ == .f32) {
            const ab = DmmvPush{ .M = d.dt_rank, .K = d.n_embd };
            cmd.dispatch(&self.pipes.dmmv_f32_dual, .{ d.dt_rank, 1, 1 }, .{ 256, 1, 1 }, &.{ &walpha.gpu_buffer, &wbeta.gpu_buffer, &self.norm_buf, &self.router_buf, &self.down_buf }, &ab, @sizeOf(DmmvPush), 0);
        } else {
            self.dmmvDispatch(&cmd, walpha, &self.norm_buf, &self.router_buf, d.dt_rank, d.n_embd, 0, 0);
            self.dmmvDispatch(&cmd, wbeta, &self.norm_buf, &self.down_buf, d.dt_rank, d.n_embd, 0, 0);
        }
        // conv1d (+ SiLU) over the qkv stream → swiglu_buf (conv_out)
        const conv_prepared = is_rocm and self.use_decode_ssm_col_warp and
            d.d_state == d.head_v_dim and d.d_state <= 128 and (d.d_state & 31) == 0;
        if (conv_prepared) {
            const cp = ConvPreparePush{
                .conv_channels = d.conv_channels,
                .d_conv = d.d_conv,
                .kernel_is_f16 = boolU32(wconv.info.type_ == .f16),
                .state_offset = self.conv_off[L],
                .dt_rank = d.dt_rank,
                .d_inner = d.d_inner,
                .d_state = d.d_state,
                .n_group = d.n_group,
                .ssm_a_is_f16 = boolU32(wa.info.type_ == .f16),
                .dt_bias_is_f16 = boolU32(wdt.info.type_ == .f16),
            };
            cmd.dispatch(&self.pipes.ssm_conv1d_prepare, .{ @max(d.n_group, ceilDiv(d.d_inner, d.d_state)), 1, 1 }, .{ d.d_state, 1, 1 }, &.{ &self.attn_out_buf, &wconv.gpu_buffer, &self.ssm_conv_state[L], &self.swiglu_buf, &wdt.gpu_buffer, &self.router_buf, &self.down_buf, &wa.gpu_buffer }, &cp, @sizeOf(ConvPreparePush), 0);
        } else {
            const conv = ConvPush{ .conv_channels = d.conv_channels, .d_conv = d.d_conv, .kernel_is_f16 = boolU32(wconv.info.type_ == .f16), .state_offset = self.conv_off[L] };
            cmd.dispatch(&self.pipes.ssm_conv1d, .{ ceilDiv(d.conv_channels, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.attn_out_buf, &wconv.gpu_buffer, &self.ssm_conv_state[L], &self.swiglu_buf }, &conv, @sizeOf(ConvPush), 0);
        }
        // delta-net scan: conv_out + dt_bias + alpha + beta + ssm_a + state → attn_out_buf (delta_out)
        self.deltaNetDecodeDispatch(&cmd, wdt, wa, &self.ssm_state[L], conv_prepared);
        // gated norm: (delta_out, z) → swiglu_buf, or directly into the
        // packed activation consumed by the output projection.
        const norm_per_head: u32 = if (wnorm.info.numElements() == d.d_inner) 1 else 0;
        const gn = GatedNormPush{ .d_inner = d.d_inner, .dt_rank = d.dt_rank, .head_v_dim = d.head_v_dim, .d_state = d.d_state, .norm_per_head = norm_per_head };
        const out_q8_ready = (d.head_v_dim & 31) == 0 and self.decodePreparedQ8Eligible(wout.info.type_, d.n_embd, d.d_inner);
        if (out_q8_ready) {
            cmd.dispatch(&self.pipes.ssm_gated_norm_quant_q8, .{ d.dt_rank, 1, 1 }, .{ d.head_v_dim, 1, 1 }, &.{ &self.attn_out_buf, &self.gate_buf, &wnorm.gpu_buffer, &self.batch.?.act_q8 }, &gn, @sizeOf(GatedNormPush), 0);
            self.dmmvPreparedQ8Dispatch(&cmd, wout, &self.swiglu_buf, &self.hidden, d.n_embd, d.d_inner, 1, 0, true);
        } else {
            cmd.dispatch(&self.pipes.ssm_gated_norm, .{ d.dt_rank, 1, 1 }, .{ d.head_v_dim, 1, 1 }, &.{ &self.attn_out_buf, &self.gate_buf, &wnorm.gpu_buffer, &self.swiglu_buf }, &gn, @sizeOf(GatedNormPush), 0);
            // out projection, accumulate into hidden
            self.dmmvDispatch(&cmd, wout, &self.swiglu_buf, &self.hidden, d.n_embd, d.d_inner, 1, 0);
        }
        self.submit(cmd);

        // advance circular conv offset (host), AFTER this token's conv.
        self.conv_off[L] = (self.conv_off[L] + 1) % (d.d_conv - 1);
    }

    fn ffnBlock(self: *ForwardCuda, L: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        // qwen35 names the FFN norm `post_attention_norm.weight` (not ffn_norm).
        const wfn = self.layer(L, "post_attention_norm.weight");
        const wgate = self.layer(L, "ffn_gate.weight");
        const wup = self.layer(L, "ffn_up.weight");
        const wdown = self.layer(L, "ffn_down.weight");

        var cmd = try command.beginCommand(ctx);
        const ffn_q8 = self.use_fused_decode_ffn and self.use_decode_q8_ffn and
            wgate.info.type_ == .q4_k and wup.info.type_ == .q4_k and
            self.batch != null and (d.n_embd & 255) == 0;
        const norm_q8_ready = self.rmsNormDecodeDispatch(&cmd, &self.hidden, &wfn.gpu_buffer, &self.ffn_norm_buf, d.n_embd, ffn_q8);
        if (self.use_fused_decode_ffn and wgate.info.type_ == .q4_k and wup.info.type_ == .q4_k and (d.n_embd & 255) == 0) {
            const gu = DmmvPush{ .M = d.n_ff, .K = d.n_embd, .acc_mode = boolU32(self.use_decode_pair_reduce) };
            if (self.use_decode_q8_ffn and self.batch != null) {
                if (!norm_q8_ready) {
                    const qp = QuantActPush{ .K = d.n_embd, .T = 1 };
                    cmd.dispatch(&self.pipes.quantize_act_q8, .{ ceilDiv(d.n_embd, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &self.ffn_norm_buf, &self.batch.?.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
                }
                // The shared Q8 kernel uses acc_mode as its activation selector
                // (0 = SiLU, 1 = GELU for Gemma), not as a reduction toggle.
                // Qwen 3.5/3.6/3.8 FFNs are SwiGLU; passing the pair-reduction
                // policy here accidentally selected GEGLU on the default path.
                var gu_silu = gu;
                gu_silu.acc_mode = 0;
                cmd.dispatch(&self.pipes.dmmv_q4k_gate_up_swiglu_q8, .{ d.n_ff, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ &wgate.gpu_buffer, &wup.gpu_buffer, &self.batch.?.act_q8, &self.swiglu_buf }, &gu_silu, @sizeOf(DmmvPush), 0);
            } else {
                cmd.dispatch(&self.pipes.dmmv_q4k_gate_up_swiglu, .{ d.n_ff, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ &wgate.gpu_buffer, &wup.gpu_buffer, &self.ffn_norm_buf, &self.swiglu_buf }, &gu, @sizeOf(DmmvPush), 0);
            }
        } else {
            self.dmmvDispatch(&cmd, wgate, &self.ffn_norm_buf, &self.gate_buf, d.n_ff, d.n_embd, 0, 0);
            self.dmmvDispatch(&cmd, wup, &self.ffn_norm_buf, &self.up_buf, d.n_ff, d.n_embd, 0, 0);
            const sg = SwigluPush{ .N = d.n_ff };
            cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(d.n_ff, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
        }
        if (self.use_decode_q8_q6 and wdown.info.type_ == .q6_k and self.batch != null and (d.n_ff & 255) == 0) {
            const qp = QuantActPush{ .K = d.n_ff, .T = 1 };
            cmd.dispatch(&self.pipes.quantize_act_q8, .{ ceilDiv(d.n_ff, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ &self.swiglu_buf, &self.batch.?.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
            const down = DmmvPush{ .M = d.n_embd, .K = d.n_ff, .acc_mode = 1 };
            cmd.dispatch(&self.pipes.dmmv_q6k_q8_fast, .{ d.n_embd, 1, 1 }, .{ 32, 1, 1 }, &.{ &wdown.gpu_buffer, &self.batch.?.act_q8, &self.hidden }, &down, @sizeOf(DmmvPush), 0);
        } else {
            self.dmmvDispatch(&cmd, wdown, &self.swiglu_buf, &self.hidden, d.n_embd, d.n_ff, 1, 0);
        }
        self.submit(cmd);
    }

    /// MoE FFN block (qwen2_moe / qwen36). Replaces the dense ffnBlock when
    /// n_experts>0. Steps: rms_norm → router logits → top-k softmax → for each of
    /// the k routed experts run its gate/up/SwiGLU/down (addressed by a byte slice
    /// into the stacked expert weight) into a slot-major down_buf → weighted
    /// accumulate into hidden → shared expert (sigmoid-gated) accumulate into
    /// hidden. The routed combine and shared expert both += into hidden, so the
    /// residual is already present (no separate add).
    /// The batched single-launch routed-expert kernel for a quant, or null if we
    /// only have the per-slot path. q4k/q5k serve gate/up (and q5k down); q6k
    /// serves the 4 Q6_K ffn_down layers of qwen36-35b-a3b. When every routed
    /// tensor has one, the whole MoE block runs async with no host readback.
    fn expertsPipe(self: *ForwardCuda, t: gguf.GGMLType) ?*CudaPipeline {
        return switch (t) {
            .q4_k => &self.pipes.dmmv_q4k_experts,
            .q5_k => &self.pipes.dmmv_q5k_experts,
            .q6_k => &self.pipes.dmmv_q6k_experts,
            else => null,
        };
    }

    /// MoE decode is graph-capturable iff EVERY layer takes the batched async
    /// expert path — i.e. each layer's gate/up/down expert tensors all have a
    /// batched experts kernel (`expertsPipe != null`). Any layer that would fall
    /// to the per-slot path syncs + reads ids back to the host mid-block, which is
    /// illegal during stream capture. Mirrors the `batched_experts` predicate in
    /// `moeFfnBlock`, checked over all layers up front.
    fn moeGraphCapturable(self: *ForwardCuda) bool {
        var L: u32 = 0;
        while (L < self.d.n_layers) : (L += 1) {
            const wge = self.layer(L, "ffn_gate_exps.weight");
            const wue = self.layer(L, "ffn_up_exps.weight");
            const wde = self.layer(L, "ffn_down_exps.weight");
            if (self.expertsPipe(wge.info.type_) == null) return false;
            if (self.expertsPipe(wue.info.type_) == null) return false;
            if (self.expertsPipe(wde.info.type_) == null) return false;
        }
        return true;
    }

    fn moeFfnBlock(self: *ForwardCuda, L: u32) !void {
        const d = self.d;
        const ctx = self.ctx;
        const n_used = d.n_experts_used;
        const ef = d.n_ff; // expert_ff = intermediate_dim = 512
        const wfn = self.layer(L, "post_attention_norm.weight");
        const wrouter = self.layer(L, "ffn_gate_inp.weight"); // [hidden, n_experts] F32
        const wge = self.layer(L, "ffn_gate_exps.weight"); // stacked [hidden, ef, n_experts]
        const wue = self.layer(L, "ffn_up_exps.weight");
        const wde = self.layer(L, "ffn_down_exps.weight"); // stacked [ef, hidden, n_experts]

        // Per-expert byte strides into the stacked tensors (quant may vary per layer).
        const gate_slice = expertSliceBytes(wge.info.type_, ef, d.n_embd);
        const up_slice = expertSliceBytes(wue.info.type_, ef, d.n_embd);
        const down_slice = expertSliceBytes(wde.info.type_, d.n_embd, ef);
        // Fast batched path: one launch over all experts, ids read GPU-side, no
        // host readback (so the whole MoE block stays async — no boost-starving
        // per-layer drain). Engages whenever every routed tensor has a batched
        // experts kernel: gate/up in {q4k,q5k}, down in {q5k,q6k}. The catalog
        // 35b-a3b has 4 Q6_K-down + 1 Q5_K-gate/up layers that USED to fall to
        // the sync per-slot path; q6k/q5k experts kernels keep them async too.
        // Else (any unsupported quant) the per-slot host path (correct for any).
        const gate_pipe = self.expertsPipe(wge.info.type_);
        const up_pipe = self.expertsPipe(wue.info.type_);
        const down_pipe = self.expertsPipe(wde.info.type_);
        const batched_experts = gate_pipe != null and up_pipe != null and down_pipe != null;

        // --- Router: rms_norm → logits → top-k softmax. -----------------------
        {
            var cmd = try command.beginCommand(ctx);
            const rms = RmsPush{ .N = d.n_embd, .eps = d.rms_eps };
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ &self.hidden, &wfn.gpu_buffer, &self.ffn_norm_buf }, &rms, @sizeOf(RmsPush), 0);
            self.dmmvDispatch(&cmd, wrouter, &self.ffn_norm_buf, &self.router_logits_buf, d.n_experts, d.n_embd, 0, 0);
            const tk = TopkPush{ .n_experts = d.n_experts, .k = n_used };
            cmd.dispatch(&self.pipes.softmax_topk, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &.{ &self.router_logits_buf, &self.router_out_buf }, &tk, @sizeOf(TopkPush), 0);
            if (batched_experts) {
                self.submit(cmd); // async: experts read the ids GPU-side, no readback
            } else {
                cmd.commitAndWait(); // sync: the fallback host-gathers the ids next
                self.drainPending();
            }
        }

        // Fallback only: download the chosen expert ids for the per-slot path.
        if (!batched_experts) {
            buffer.download(ctx, &self.router_out_buf, std.mem.sliceAsBytes(self.host_router_ids[0..n_used]));
        }

        // Debug: dump router state for first decode step at specific layers
        if ((L == 0 or L == 3 or L == 39) and std.posix.getenv("ZINC_MOE_DEBUG") != null and !self.capturing) {
            self.waitPending();
            var logits: [256]f32 = undefined;
            buffer.download(ctx, &self.router_logits_buf, std.mem.sliceAsBytes(logits[0..d.n_experts]));
            var router_out: [16]u32 = undefined;
            buffer.download(ctx, &self.router_out_buf, std.mem.sliceAsBytes(router_out[0 .. n_used * 2]));
            var max_logit: f32 = -1e30;
            var max_id: u32 = 0;
            for (0..d.n_experts) |i| {
                if (logits[i] > max_logit) {
                    max_logit = logits[i];
                    max_id = @intCast(i);
                }
            }
            log.info("MOE_DEBUG L{d}: max_logit={d:.4} at expert {d}, top-8 ids=[{d},{d},{d},{d},{d},{d},{d},{d}], weights=[{d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4},{d:.4}]", .{
                L,                                          max_logit,                                  max_id,
                router_out[0],                              router_out[1],                              router_out[2],
                router_out[3],                              router_out[4],                              router_out[5],
                router_out[6],                              router_out[7],                              @as(f32, @bitCast(router_out[n_used])),
                @as(f32, @bitCast(router_out[n_used + 1])), @as(f32, @bitCast(router_out[n_used + 2])), @as(f32, @bitCast(router_out[n_used + 3])),
                @as(f32, @bitCast(router_out[n_used + 4])), @as(f32, @bitCast(router_out[n_used + 5])), @as(f32, @bitCast(router_out[n_used + 6])),
                @as(f32, @bitCast(router_out[n_used + 7])),
            });
        }

        // --- Routed experts → SwiGLU → down, slot-major into down_buf.
        {
            var cmd = try command.beginCommand(ctx);
            if (batched_experts) {
                const nrows_gu = n_used * ef;
                const pg = ExpertsPush{ .M = ef, .K = d.n_embd, .slice = gate_slice, .x_stride = 0, .n_used = n_used };
                cmd.dispatch(gate_pipe.?, .{ nrows_gu, 1, 1 }, .{ 64, 1, 1 }, &.{ &wge.gpu_buffer, &self.ffn_norm_buf, &self.gate_buf, &self.router_out_buf }, &pg, @sizeOf(ExpertsPush), 0);
                const pu = ExpertsPush{ .M = ef, .K = d.n_embd, .slice = up_slice, .x_stride = 0, .n_used = n_used };
                cmd.dispatch(up_pipe.?, .{ nrows_gu, 1, 1 }, .{ 64, 1, 1 }, &.{ &wue.gpu_buffer, &self.ffn_norm_buf, &self.up_buf, &self.router_out_buf }, &pu, @sizeOf(ExpertsPush), 0);
                const sg = SwigluPush{ .N = nrows_gu };
                cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(nrows_gu, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
                const pd = ExpertsPush{ .M = d.n_embd, .K = ef, .slice = down_slice, .x_stride = ef, .n_used = n_used };
                cmd.dispatch(down_pipe.?, .{ n_used * d.n_embd, 1, 1 }, .{ 64, 1, 1 }, &.{ &wde.gpu_buffer, &self.swiglu_buf, &self.down_buf, &self.router_out_buf }, &pd, @sizeOf(ExpertsPush), 0);
            } else {
                const sg = SwigluPush{ .N = ef };
                var j: u32 = 0;
                while (j < n_used) : (j += 1) {
                    const id = self.host_router_ids[j];
                    self.dmmvDispatch(&cmd, wge, &self.ffn_norm_buf, &self.gate_buf, ef, d.n_embd, 0, id * gate_slice);
                    self.dmmvDispatch(&cmd, wue, &self.ffn_norm_buf, &self.up_buf, ef, d.n_embd, 0, id * up_slice);
                    cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(ef, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
                    const down_push = DmmvPush{ .M = d.n_embd, .K = ef, .acc_mode = 0, .a_offset = id * down_slice, .y_offset = j * d.n_embd * @sizeOf(f32) };
                    const didx = dmmvIdx(wde.info.type_);
                    if (didx < 4) {
                        cmd.dispatch(&self.pipes.dmmv_fast[didx], .{ d.n_embd, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ &wde.gpu_buffer, &self.swiglu_buf, &self.down_buf }, &down_push, @sizeOf(DmmvPush), 0);
                    } else {
                        cmd.dispatch(&self.pipes.dmmv[didx], .{ d.n_embd, 1, 1 }, .{ 256, 1, 1 }, &.{ &wde.gpu_buffer, &self.swiglu_buf, &self.down_buf }, &down_push, @sizeOf(DmmvPush), 0);
                    }
                }
            }
            // weighted combine of the k slot outputs into hidden.
            const ma = MoeAccPush{ .N = d.n_embd, .n_used = n_used, .src_stride = d.n_embd };
            cmd.dispatch(&self.pipes.moe_weighted_acc, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.hidden, &self.down_buf, &self.router_out_buf }, &ma, @sizeOf(MoeAccPush), 0);
            self.submit(cmd);
        }

        // --- Shared expert: gate/up/SwiGLU/down, scaled by sigmoid(gate logit).
        {
            const wgs = self.layer(L, "ffn_gate_shexp.weight");
            const wus = self.layer(L, "ffn_up_shexp.weight");
            const wds = self.layer(L, "ffn_down_shexp.weight");
            const wgi = self.layer(L, "ffn_gate_inp_shexp.weight"); // [hidden, 1] F32
            const sf = d.shexp_ff;
            var cmd = try command.beginCommand(ctx);
            self.dmmvDispatch(&cmd, wgs, &self.ffn_norm_buf, &self.gate_buf, sf, d.n_embd, 0, 0);
            self.dmmvDispatch(&cmd, wus, &self.ffn_norm_buf, &self.up_buf, sf, d.n_embd, 0, 0);
            const sg = SwigluPush{ .N = sf };
            cmd.dispatch(&self.pipes.swiglu, .{ ceilDiv(sf, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.gate_buf, &self.up_buf, &self.swiglu_buf }, &sg, @sizeOf(SwigluPush), 0);
            // shared-expert gate scalar = W_gi · norm (1×hidden).
            self.dmmvDispatch(&cmd, wgi, &self.ffn_norm_buf, &self.gate_scalar_buf, 1, d.n_embd, 0, 0);
            // down → down_buf[0..hidden], then hidden += sigmoid(gate)*down.
            self.dmmvDispatch(&cmd, wds, &self.swiglu_buf, &self.down_buf, d.n_embd, sf, 0, 0);
            const ss = SigmoidAccPush{ .N = d.n_embd };
            cmd.dispatch(&self.pipes.sigmoid_scale_acc, .{ ceilDiv(d.n_embd, 64), 1, 1 }, .{ 64, 1, 1 }, &.{ &self.hidden, &self.down_buf, &self.gate_scalar_buf }, &ss, @sizeOf(SigmoidAccPush), 0);
            self.submit(cmd); // async: deferred like the dense path
        }

        // Debug: dump hidden state norm after MoE FFN
        if ((L == 0 or L == 3 or L == 39) and std.posix.getenv("ZINC_MOE_DEBUG") != null and !self.capturing) {
            self.waitPending();
            var hidden_sample: [8]f32 = undefined;
            buffer.download(ctx, &self.hidden, std.mem.sliceAsBytes(hidden_sample[0..8]));
            var gate_s: [1]f32 = undefined;
            buffer.download(ctx, &self.gate_scalar_buf, std.mem.sliceAsBytes(gate_s[0..1]));
            log.info("MOE_DEBUG L{d} post: hidden[0..3]=[{d:.4},{d:.4},{d:.4},{d:.4}] gate_s={d:.4} sig={d:.4}", .{
                L,
                hidden_sample[0],
                hidden_sample[1],
                hidden_sample[2],
                hidden_sample[3],
                gate_s[0],
                1.0 / (1.0 + @exp(-gate_s[0])),
            });
        }
    }

    // Public wrappers for the per-block builders (used by the v1 driver to run
    // and inspect one block type at a time).
    pub fn attentionLayerPub(self: *ForwardCuda, L: u32, pos: u32) !void {
        try self.attentionLayer(L, pos);
        self.waitPending(); // dump callers read hidden right after — make it sync
    }
    pub fn ssmLayerPub(self: *ForwardCuda, L: u32) !void {
        try self.ssmLayer(L);
        self.waitPending();
    }
    pub fn ffnBlockPub(self: *ForwardCuda, L: u32) !void {
        try self.ffnBlock(L);
        self.waitPending();
    }

    /// Async decode command ring. For the dense path (n_experts==0) each layer-op
    /// commits asynchronously and is stashed; the ops pipeline on the shared
    /// CUstream (the CPU never blocks per-op), then the tail's single
    /// commitAndWait drains the stream and `drainPending` frees the events. This
    /// removes the ~0.4 ms/op WSL2 CPU↔GPU sync round-trip (and the boost-
    /// starvation those idle gaps cause — see Effort 21 Cycles 7–8). MoE layers
    /// also pipeline now: only the per-layer router stays synchronous (it reads
    /// the chosen expert ids back to the host mid-block); the routed + shared
    /// expert commands defer like the dense path, so a MoE token drops from
    /// ~4 syncs/layer to 1 (the router) — recovering the starved GPU boost.
    fn submit(self: *ForwardCuda, cmd: command.CudaCommand) void {
        var c = cmd;
        if (self.capturing) {
            // Graph capture: the dispatches are already recorded onto the captured
            // stream. Don't record a completion event or sync — just free the
            // (unused) command handle; the single graph launch + its sync drains
            // all captured work at the end of the step.
            c.releaseCompleted();
            return;
        }
        if (self.lightweight_rocm_commands) {
            // cuda_dispatch has already enqueued each launch on the context's
            // ordered stream. Release the unused event now; the decode tail or
            // waitPending fence remains the sole synchronization point.
            c.releaseCompleted();
            return;
        }
        if (self.n_pending < self.pending.len) {
            c.commitAsync();
            self.pending[self.n_pending] = c;
            self.n_pending += 1;
        } else {
            c.commitAndWait();
        }
    }

    /// Free the stashed async commands. Safe once a later same-stream
    /// commitAndWait (the tail) has drained the stream — completion is guaranteed.
    fn drainPending(self: *ForwardCuda) void {
        var i: u32 = 0;
        while (i < self.n_pending) : (i += 1) self.pending[i].releaseCompleted();
        self.n_pending = 0;
    }

    /// Wait on + free the stashed async commands, for callers that read GPU
    /// results before any tail sync (the per-block Pub wrappers).
    fn waitPending(self: *ForwardCuda) void {
        if (self.lightweight_rocm_commands) {
            var fence = command.beginCommand(self.ctx) catch return;
            fence.commitAndWait();
            return;
        }
        var i: u32 = 0;
        while (i < self.n_pending) : (i += 1) self.pending[i].wait();
        self.n_pending = 0;
    }

    /// Download the current hidden state (for sanity checks).
    pub fn readHidden(self: *ForwardCuda, out: []f32) void {
        buffer.download(self.ctx, &self.hidden, std.mem.sliceAsBytes(out[0..self.d.n_embd]));
    }

    /// Download the top of the logits buffer (for top-k reporting).
    pub fn readLogits(self: *ForwardCuda, out: []f32) void {
        buffer.download(self.ctx, &self.logits_buf, std.mem.sliceAsBytes(out[0..@min(out.len, self.d.vocab)]));
    }

    // ---- helpers ------------------------------------------------------------

    /// Decode-only normalization helper. When the next projection consumes the
    /// ROCm packed-Q8 activation, produce the normalized f32 row and that packed
    /// view together. Returns true when `batch.act_q8` is ready for immediate
    /// reuse by one or more same-input matvecs.
    fn rmsNormDecodeDispatch(
        self: *ForwardCuda,
        cmd: *command.CudaCommand,
        x: *const CudaBuffer,
        w: *const CudaBuffer,
        y: *const CudaBuffer,
        N: u32,
        want_q8: bool,
    ) bool {
        const fuse = self.use_rocm_rms_q8 and want_q8 and self.batch != null and (N & 255) == 0;
        if (fuse) {
            const rms_q8 = RmsQ8Push{ .N = N, .eps = self.d.rms_eps, .T = 1 };
            cmd.dispatch(&self.pipes.rms_norm_quant_q8, .{ 1, 1, 1 }, .{ 1024, 1, 1 }, &.{ x, w, y, &self.batch.?.act_q8 }, &rms_q8, @sizeOf(RmsQ8Push), 0);
        } else {
            const rms = RmsPush{ .N = N, .eps = self.d.rms_eps };
            cmd.dispatch(&self.pipes.rms_norm, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ x, w, y }, &rms, @sizeOf(RmsPush), 0);
        }
        return fuse;
    }

    /// Batched-prefill twin of `rmsNormDecodeDispatch`. One 1024-thread block
    /// owns each token, so all prompt rows remain independent while Q8 packing
    /// is folded into the normalization pass used by the first projection.
    fn rmsNormPrefillDispatch(
        self: *ForwardCuda,
        cmd: *command.CudaCommand,
        x: *const CudaBuffer,
        w: *const CudaBuffer,
        y: *const CudaBuffer,
        N: u32,
        T: u32,
        want_q8: bool,
    ) bool {
        const fuse = self.use_rocm_rms_q8 and want_q8 and self.batch != null and (N & 255) == 0;
        if (fuse) {
            const rms_q8 = RmsQ8Push{ .N = N, .eps = self.d.rms_eps, .T = T };
            cmd.dispatch(&self.pipes.rms_norm_quant_q8, .{ T, 1, 1 }, .{ 1024, 1, 1 }, &.{ x, w, y, &self.batch.?.act_q8 }, &rms_q8, @sizeOf(RmsQ8Push), 0);
        } else {
            const rms = RmsPush{ .N = N, .eps = self.d.rms_eps };
            cmd.dispatch(&self.pipes.rms_norm, .{ T, 1, 1 }, .{ 256, 1, 1 }, &.{ x, w, y }, &rms, @sizeOf(RmsPush), 0);
        }
        return fuse;
    }

    /// Whether a dense decode projection can consume the packed activation made
    /// by a preceding fused RMS kernel. Keeping this policy in one place lets
    /// same-input projection pairs pack once instead of each dispatcher launching
    /// its own identical quantizer.
    fn decodePreparedQ8Eligible(self: *ForwardCuda, type_: gguf.GGMLType, M: u32, K: u32) bool {
        if (self.d.n_experts != 0 or self.batch == null or (K & 255) != 0) return false;
        return switch (type_) {
            .q4_k => self.use_decode_q8_q4,
            .q5_k => self.use_decode_q8_q5 and M >= 4096,
            .q6_k => self.use_decode_q8_q6_proj and M >= 4096,
            else => false,
        };
    }

    /// Matvec dispatch when a preceding fused RMS kernel has already packed the
    /// activation into `batch.act_q8`. Falls back to the ordinary dispatcher for
    /// unsupported quant/shape combinations.
    fn dmmvPreparedQ8Dispatch(
        self: *ForwardCuda,
        cmd: *command.CudaCommand,
        w: *const LoadedTensor,
        x: *const CudaBuffer,
        y: *const CudaBuffer,
        M: u32,
        K: u32,
        acc_mode: u32,
        a_offset: u32,
        q8_ready: bool,
    ) void {
        if (q8_ready and self.decodePreparedQ8Eligible(w.info.type_, M, K)) {
            const push = DmmvPush{ .M = M, .K = K, .acc_mode = acc_mode, .a_offset = a_offset };
            const pipe = switch (w.info.type_) {
                .q4_k => &self.pipes.dmmv_q4k_q8_fast,
                .q5_k => &self.pipes.dmmv_q5k_q8_fast,
                .q6_k => &self.pipes.dmmv_q6k_q8_fast,
                else => unreachable,
            };
            cmd.dispatch(pipe, .{ M, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ &w.gpu_buffer, &self.batch.?.act_q8, y }, &push, @sizeOf(DmmvPush), 0);
            return;
        }
        self.dmmvDispatch(cmd, w, x, y, M, K, acc_mode, a_offset);
    }

    fn layer(self: *ForwardCuda, L: u32, suffix: []const u8) *const LoadedTensor {
        return self.model.getLayer(L, suffix) orelse {
            log.err("missing tensor blk.{d}.{s}", .{ L, suffix });
            @panic("missing tensor");
        };
    }

    /// Dispatch the right DMMV kernel for `w`'s quant type: y[M] = W[M,K] · x.
    /// Quant types (idx < 4: q4k/q5k/q6k/q8_0) use the fast single-row variant at
    /// block=64; f32 (idx == 4) has no fast variant and keeps the base at block=256.
    fn dmmvDispatch(self: *ForwardCuda, cmd: *command.CudaCommand, w: *const LoadedTensor, x: *const CudaBuffer, y: *const CudaBuffer, M: u32, K: u32, acc_mode: u32, a_offset: u32) void {
        const push = DmmvPush{ .M = M, .K = K, .acc_mode = acc_mode, .a_offset = a_offset };
        const idx = dmmvIdx(w.info.type_);
        if (self.use_decode_q8_q4 and self.d.n_experts == 0 and idx == 0 and self.batch != null and (K & 255) == 0) {
            const qp = QuantActPush{ .K = K, .T = 1 };
            cmd.dispatch(&self.pipes.quantize_act_q8, .{ ceilDiv(K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ x, &self.batch.?.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
            cmd.dispatch(&self.pipes.dmmv_q4k_q8_fast, .{ M, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ &w.gpu_buffer, &self.batch.?.act_q8, y }, &push, @sizeOf(DmmvPush), 0);
            return;
        }
        if (self.use_decode_q8_q6_proj and self.d.n_experts == 0 and idx == 2 and M >= 4096 and self.batch != null and (K & 255) == 0) {
            const qp = QuantActPush{ .K = K, .T = 1 };
            cmd.dispatch(&self.pipes.quantize_act_q8, .{ ceilDiv(K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ x, &self.batch.?.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
            cmd.dispatch(&self.pipes.dmmv_q6k_q8_fast, .{ M, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ &w.gpu_buffer, &self.batch.?.act_q8, y }, &push, @sizeOf(DmmvPush), 0);
            return;
        }
        if (self.use_decode_q8_q5 and self.d.n_experts == 0 and idx == 1 and M >= 4096 and self.batch != null and (K & 255) == 0) {
            const qp = QuantActPush{ .K = K, .T = 1 };
            cmd.dispatch(&self.pipes.quantize_act_q8, .{ ceilDiv(K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ x, &self.batch.?.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
            cmd.dispatch(&self.pipes.dmmv_q5k_q8_fast, .{ M, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ &w.gpu_buffer, &self.batch.?.act_q8, y }, &push, @sizeOf(DmmvPush), 0);
            return;
        }
        if (idx < 4) {
            cmd.dispatch(&self.pipes.dmmv_fast[idx], .{ M, 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(DmmvPush), 0);
        } else {
            cmd.dispatch(&self.pipes.dmmv[idx], .{ M, 1, 1 }, .{ 256, 1, 1 }, &.{ &w.gpu_buffer, x, y }, &push, @sizeOf(DmmvPush), 0);
        }
    }

    fn argmaxDispatch(self: *ForwardCuda, cmd: *command.CudaCommand, out: *const CudaBuffer) void {
        self.argmaxDispatchFrom(cmd, &self.logits_buf, out);
    }

    fn argmaxDispatchFrom(self: *ForwardCuda, cmd: *command.CudaCommand, logits: *const CudaBuffer, out: *const CudaBuffer) void {
        if (self.use_argmax_v2) {
            const push = ArgmaxV2Push{ .N = self.d.vocab, .partials = 128 };
            cmd.dispatch(&self.pipes.argmax_partials, .{ 128, 1, 1 }, .{ 256, 1, 1 }, &.{ logits, &self.argmax_partial_buf }, &push, @sizeOf(ArgmaxV2Push), 0);
            cmd.dispatch(&self.pipes.argmax_finalize, .{ 1, 1, 1 }, .{ 128, 1, 1 }, &.{ &self.argmax_partial_buf, out }, &push, @sizeOf(ArgmaxV2Push), 0);
        } else {
            const push = ArgmaxPush{ .N = self.d.vocab };
            cmd.dispatch(&self.pipes.argmax, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &.{ logits, out }, &push, @sizeOf(ArgmaxPush), 0);
        }
    }

    fn deltaNetDecodeDispatch(
        self: *ForwardCuda,
        cmd: *command.CudaCommand,
        wdt: *const LoadedTensor,
        wa: *const LoadedTensor,
        state: *const CudaBuffer,
        prepared: bool,
    ) void {
        const d = self.d;
        if (self.use_decode_ssm_col_warp and d.d_state == d.head_v_dim and d.d_state <= 128) {
            if (!prepared) {
                const prep = DeltaNetPreparePush{
                    .dt_rank = d.dt_rank,
                    .d_state = d.d_state,
                    .n_group = d.n_group,
                    .ssm_a_is_f16 = boolU32(wa.info.type_ == .f16),
                    .dt_bias_is_f16 = boolU32(wdt.info.type_ == .f16),
                    .has_dt_bias = 1,
                    .has_ssm_a = 1,
                    .n_tok = 1,
                    .conv_stride_tok = d.conv_channels,
                    .ab_stride_tok = d.dt_rank,
                };
                cmd.dispatch(&self.pipes.ssm_delta_net_prepare, .{ d.n_group, 1, 1 }, .{ d.d_state, 1, 1 }, &.{ &self.swiglu_buf, &wdt.gpu_buffer, &self.router_buf, &self.down_buf, &wa.gpu_buffer }, &prep, @sizeOf(DeltaNetPreparePush), 0);
            }
            const scan = DeltaNetColWarpPush{
                .dt_rank = d.dt_rank,
                .head_v_dim = d.head_v_dim,
                .d_state = d.d_state,
                .n_group = d.n_group,
                .n_tok = 1,
                .conv_stride_tok = d.conv_channels,
                .ab_stride_tok = d.dt_rank,
                .y_stride_tok = d.d_inner,
                .fast_reduce = boolU32(self.use_decode_ssm_fast),
            };
            cmd.dispatch(&self.pipes.ssm_delta_net_col_warp, .{ d.dt_rank, ceilDiv(d.head_v_dim, 4), 1 }, .{ 32, 4, 1 }, &.{ &self.swiglu_buf, &self.router_buf, &self.down_buf, state, &self.attn_out_buf }, &scan, @sizeOf(DeltaNetColWarpPush), 0);
            return;
        }
        const scan = DeltaNetPush{
            .d_inner = d.d_inner,
            .dt_rank = d.dt_rank,
            .head_v_dim = d.head_v_dim,
            .d_state = d.d_state,
            .n_group = d.n_group,
            .ssm_a_is_f16 = boolU32(wa.info.type_ == .f16),
            .dt_bias_is_f16 = boolU32(wdt.info.type_ == .f16),
            .has_dt_bias = 1,
            .has_ssm_a = 1,
            .n_tok = 1,
            .conv_stride_tok = d.conv_channels,
            .ab_stride_tok = d.dt_rank,
            .y_stride_tok = d.d_inner,
            .preprocessed = 0,
        };
        cmd.dispatch(&self.pipes.ssm_delta_net, .{ d.dt_rank, d.head_v_dim, 1 }, .{ d.head_v_dim, 1, 1 }, &.{ &self.swiglu_buf, &wdt.gpu_buffer, &self.router_buf, &self.down_buf, &wa.gpu_buffer, state, &self.attn_out_buf }, &scan, @sizeOf(DeltaNetPush), 0);
    }

    fn dmmvQ4PairDispatch(
        self: *ForwardCuda,
        cmd: *command.CudaCommand,
        w0: *const CudaBuffer,
        w1: *const CudaBuffer,
        x: *const CudaBuffer,
        y0: *const CudaBuffer,
        y1: *const CudaBuffer,
        M0: u32,
        M1: u32,
        K: u32,
        q8_ready: bool,
    ) void {
        const push = DmmvPairPush{ .M0 = M0, .M1 = M1, .K = K, .pair_reduce = boolU32(self.use_q4_pair_reduce) };
        if (self.use_decode_q8_q4_pair and M0 >= 4096 and self.batch != null and (K & 255) == 0) {
            if (!q8_ready) {
                const qp = QuantActPush{ .K = K, .T = 1 };
                cmd.dispatch(&self.pipes.quantize_act_q8, .{ ceilDiv(K, 256), 1, 1 }, .{ 256, 1, 1 }, &.{ x, &self.batch.?.act_q8 }, &qp, @sizeOf(QuantActPush), 0);
            }
            cmd.dispatch(&self.pipes.dmmv_q4k_pair_q8, .{ @max(M0, M1), 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ w0, w1, &self.batch.?.act_q8, y0, y1 }, &push, @sizeOf(DmmvPairPush), 0);
        } else {
            cmd.dispatch(&self.pipes.dmmv_q4k_pair, .{ @max(M0, M1), 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ w0, w1, x, y0, y1 }, &push, @sizeOf(DmmvPairPush), 0);
        }
    }

    /// Two-token verifier projection pair using the packed Q8 activation that
    /// rmsNormPrefillDispatch already produced. The paired kernel shares each
    /// activation load between the two Q4_K matrices and both speculative rows.
    fn dmmvQ4PairQ8Btok2(
        self: *ForwardCuda,
        cmd: *command.CudaCommand,
        w0: *const CudaBuffer,
        w1: *const CudaBuffer,
        y0: *const CudaBuffer,
        y1: *const CudaBuffer,
        M0: u32,
        M1: u32,
        K: u32,
    ) void {
        const push = DmmvPairPush{ .M0 = M0, .M1 = M1, .K = K, .pair_reduce = boolU32(self.use_q4_pair_reduce) };
        cmd.dispatch(&self.pipes.dmmv_q4k_pair_q8_btok2, .{ @max(M0, M1), 1, 1 }, .{ dmmv_fast_block, 1, 1 }, &.{ w0, w1, &self.batch.?.act_q8, y0, y1 }, &push, @sizeOf(DmmvPairPush), 0);
    }

    /// Effort 28: GPU-side stacked-MoE expert matvec over all `n_used` experts in
    /// ONE launch (block g handles expert e=g/M, row=g%M; the chosen expert id is
    /// read GPU-side from `ids`). Picks the `dmmv_*_experts` kernel by weight quant
    /// (q4_k/q5_k/q6_k); the per-row math equals `dmmvDispatch` on x[e] bit-for-bit,
    /// so the async path stays token-identical to the host-id fallback. `w.info.type_`
    /// must be `expertsSupported`. x is shared (x_stride=0) for gate/up or per-expert
    /// (x_stride=K) for down; output is slot-major y[e*M + row].
    fn expertsDispatch(self: *ForwardCuda, cmd: *command.CudaCommand, w: *const LoadedTensor, x: *const CudaBuffer, y: *const CudaBuffer, ids: *const CudaBuffer, M: u32, K: u32, slice: u32, x_stride: u32, n_used: u32) void {
        const p = ExpertsPush{ .M = M, .K = K, .slice = slice, .x_stride = x_stride, .n_used = n_used };
        const pipe = switch (dmmvIdx(w.info.type_)) {
            0 => &self.pipes.dmmv_q4k_experts,
            1 => &self.pipes.dmmv_q5k_experts,
            2 => &self.pipes.dmmv_q6k_experts,
            else => unreachable, // gated by expertsSupported(w.info.type_)
        };
        cmd.dispatch(pipe, .{ n_used * M, 1, 1 }, .{ 64, 1, 1 }, &.{ &w.gpu_buffer, x, y, ids }, &p, @sizeOf(ExpertsPush), 0);
    }
};

// Effort 28: the B==1 decodeBatch matvec fast path is default-ON (opt out
// ZINC_BATCH_B1_MATVEC=0/off/false/no — the SAME env knob as the gemma path).
fn b1MatvecOn() bool {
    const v = std.posix.getenv("ZINC_BATCH_B1_MATVEC") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

// Effort 28: the small-B (2..8) token-batch matvec (btok + MoE launch-collapse +
// shared-expert batching) is now DEFAULT-ON (opt out with ZINC_BATCH_MROW=0/off/
// false/no). Flipped 2026-06-15 after the clean-window head-to-head gate:
// ZINCM (mrow ON) clean-beats ZINC0 (mrow OFF) at every batched B — B=2/4/8 medians
// 34.76/48.02/60.33 vs 8.40/16.97/26.36 tok/s (4.14×/2.83×/2.29×), with NO
// regression at B=1 (mrow only engages for 2≤B≤8). The batched mrow-ON path is
// token-identical to N-serial (proven every cycle), so the default flip just makes
// the validated-better path the serving default; the serial decodeStep/prefill path
// never sets decode_mrow → catalog correctness is unaffected by construction.
fn mrowMatvecOn() bool {
    const v = std.posix.getenv("ZINC_BATCH_MROW") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

// Effort 28: the MoE shared-expert batching is default-ON (opt out with
// ZINC_BATCH_MOE_SHARED=0/off/false/no) — it is token-identical, so unlike the
// opt-in matvec levers it ships on. The env knob exists for the in-process A/B.
fn moeSharedBatchedOn() bool {
    const v = std.posix.getenv("ZINC_BATCH_MOE_SHARED") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

fn isFullAttn(L: u32, interval: u32) bool {
    return (L + 1) % interval == 0;
}

/// Alias token `t`'s row of a token-major [T, width] f32 buffer (Effort 26 T0).
/// The returned buffer does not own its handle — free it (wrapper only) after
/// the dispatches that read it; the device memory belongs to `buf`.
fn aliasRow(buf: *const CudaBuffer, t: u32, width: u32) !CudaBuffer {
    return buffer.aliasBuffer(buf, @as(usize, t) * width * @sizeOf(f32), @as(usize, width) * @sizeOf(f32));
}

/// Effort 26 T0: cuBLAS dense-prefill GEMM defaults ON; opt out with
/// ZINC_BATCHED_CUBLAS=0/off/false/no (the A/B kill-switch to the matvec path).
fn cublasDefaultOn() bool {
    const v = std.posix.getenv("ZINC_BATCHED_CUBLAS") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off") or std.mem.eql(u8, v, "false") or std.mem.eql(u8, v, "no"));
}

/// Pre-quantized activation GEMM is the measured ROCm default for Q4/Q5/Q6
/// prefill. CUDA keeps it opt-in. ZINC_PREFILL_Q8 remains an A/B opt-out/opt-in.
fn prefillQ8On() bool {
    const v = std.posix.getenv("ZINC_PREFILL_Q8") orelse return is_rocm;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// Packed-Q8 two-token verifier. Q4_K and Q6_K are the measured/default pair
/// for Qwen3.8 Q4_K_M: they preserve greedy decode's arithmetic partition and
/// beat the float btok verifier. The knobs remain available for clean A/Bs.
fn mtpQ8On() bool {
    const v = std.posix.getenv("ZINC_MTP_Q8") orelse return is_rocm;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

fn mtpQ8TypeOn(idx: usize) bool {
    if (!mtpQ8On() or idx >= 4) return false;
    const enabled = std.posix.getenv("ZINC_MTP_Q8_TYPES") orelse "02";
    return std.mem.indexOfScalar(u8, enabled, @as(u8, '0') + @as(u8, @intCast(idx))) != null;
}

/// Precompute the Q/K normalization and activated gate/beta values once per
/// token/head before the state-compatible block delta-net scan. Measured ROCm
/// default; CUDA remains opt-in. ZINC_SSM_PREPARED is the A/B opt-out/opt-in.
fn ssmPreparedOn() bool {
    const v = std.posix.getenv("ZINC_SSM_PREPARED") orelse return is_rocm;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// Four-column wave32 scan that preserves the block kernel's state layout and
/// exact reduction order. Measured ROCm default; CUDA remains opt-in.
fn ssmColWarpOn() bool {
    const v = std.posix.getenv("ZINC_SSM_COL_WARP") orelse return is_rocm;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// One-reduction variant of the state-compatible column scan. This changes
/// only floating-point association. Measured ROCm default; CUDA remains opt-in.
fn ssmColWarpFastOn() bool {
    const v = std.posix.getenv("ZINC_SSM_COL_WARP_FAST") orelse return is_rocm;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// Reuse one Q8 activation tile across adjacent projections with the same input
/// (Q/K/V, QKV/Z, or gate/up). Exact-work-elimination ROCm default; CUDA opt-in.
fn prefillQ8ReuseOn() bool {
    const v = std.posix.getenv("ZINC_PREFILL_Q8_REUSE") orelse return is_rocm;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// Coalesced wave-per-key prefill attention. Measured ROCm default; CUDA keeps
/// the original reduction-order kernel unless explicitly enabled.
fn prefillAttnV2On() bool {
    const v = std.posix.getenv("ZINC_ATTN_V2") orelse return is_rocm;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// Shape-specialized gfx12 Q8-WMMA token tiles derived from the reference MMQ
/// strategy. Measured ROCm default; CUDA remains opt-in.
fn prefillWmmaTilesOn() bool {
    const v = std.posix.getenv("ZINC_PREFILL_WMMA_TILES") orelse return is_rocm;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// Measured 80-column gfx12 WMMA tile for prompts in (48, 80]. ROCm default;
/// retain a dedicated opt-out so future GPUs can compare the generic tile.
fn prefillWmmaT80On() bool {
    const v = std.posix.getenv("ZINC_PREFILL_WMMA_T80") orelse return is_rocm;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// Reuse exact short-prompt WMMA shapes for the final partial tile of a long
/// prompt. Measured ROCm default; retain an opt-out for other gfx12 revisions.
fn prefillWmmaTailOn() bool {
    const v = std.posix.getenv("ZINC_PREFILL_WMMA_TAIL") orelse return is_rocm;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// Direct Q4_K fragment loads reduce LDS pressure for the 48-column tile.
fn prefillWmmaQ4DirectOn() bool {
    return envFlag("ZINC_PREFILL_WMMA_Q4_DIRECT", is_rocm);
}

/// Q8_0 prefill tensor-core GEMM for shared-expert dense projections. DEFAULT-ON;
/// opt out with ZINC_PREFILL_Q8_TC=0/off/false/no to fall back to gemm_q8_0_tiled_v2.
fn q8TcLowsmemOn() bool {
    const v = std.posix.getenv("ZINC_PREFILL_Q8_TC") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// Experimental fused Q4_K gate+up+SwiGLU prefill kernel. Opt-in because the
/// qwen35 short-core A/B on the 5090 regressed prefill versus the unfused
/// lowsmem TC path.
fn prefillGateUpSwigluOn() bool {
    const v = std.posix.getenv("ZINC_PREFILL_GATE_UP_SWIGLU") orelse return false;
    return std.mem.eql(u8, v, "1") or std.ascii.eqlIgnoreCase(v, "on") or
        std.ascii.eqlIgnoreCase(v, "true") or std.ascii.eqlIgnoreCase(v, "yes");
}

/// Effort 29 T3: the B>27 serving-decode cuBLAS GEMM path. DEFAULT-ON; opt out
/// with ZINC_SERVE_CUBLAS=0/off/false/no (the A/B kill-switch back to the f32
/// tile GEMM). Only engages for B > serve_cublas_min_b, so the production B==1
/// decodeStep and the B≤27 btok serving path are untouched.
fn serveCublasOn() bool {
    const v = std.posix.getenv("ZINC_SERVE_CUBLAS") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off") or std.mem.eql(u8, v, "false") or std.mem.eql(u8, v, "no"));
}

/// Effort 29 T3 option-b: resident fp16 weight cache for the serving-decode
/// cuBLAS GEMMs. OPT-IN (ZINC_SERVE_WCACHE=1) — it removes the per-step dequant
/// round-trip but costs ~3.5× the Q4_K bytes in VRAM (a serving batch-capacity
/// trade), so unlike serve_cublas it does NOT default on.
fn serveWcacheOn() bool {
    const v = std.posix.getenv("ZINC_SERVE_WCACHE") orelse return false;
    return std.mem.eql(u8, v, "1") or std.ascii.eqlIgnoreCase(v, "on") or
        std.ascii.eqlIgnoreCase(v, "true") or std.ascii.eqlIgnoreCase(v, "yes");
}

/// Effort 29 T2: token-batched qwen2-MoE prefill FFN. DEFAULT-ON; ZINC_QWEN_MOE_BATCHED=0
/// opts out to the per-token loop (the A/B baseline arm).
fn qwenMoeBatchedOn() bool {
    const v = std.posix.getenv("ZINC_QWEN_MOE_BATCHED") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.mem.eql(u8, v, "off") or std.mem.eql(u8, v, "false") or std.mem.eql(u8, v, "no"));
}

fn ceilDiv(a: u32, b: u32) u32 {
    return (a + b - 1) / b;
}

fn envFlag(name: []const u8, default: bool) bool {
    const value = std.posix.getenv(name) orelse return default;
    return !(std.mem.eql(u8, value, "0") or std.ascii.eqlIgnoreCase(value, "off") or
        std.ascii.eqlIgnoreCase(value, "false") or std.ascii.eqlIgnoreCase(value, "no"));
}

fn envU32(name: []const u8, default: u32) u32 {
    const value = std.posix.getenv(name) orelse return default;
    return std.fmt.parseUnsigned(u32, std.mem.trim(u8, value, " \t\r\n"), 10) catch default;
}

/// T2 (qwen MoE port): grouped Tensor-core routed gate/up experts. DEFAULT-ON;
/// opt out with ZINC_MOE_TC=0/off/false/no (the A/B kill-switch to the matvec).
fn moeTcDefaultOn() bool {
    const v = std.posix.getenv("ZINC_MOE_TC") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// T2: true only when ZINC_MOE_TC is an EXPLICIT truthy value → run the grouped TC
/// experts at ANY T (bypass moe_tc_min_t) so validate_catalog can exercise the path
/// with the short catalog prompt. Unset = default-on but T-gated; falsy = off.
fn moeTcForced() bool {
    const v = std.posix.getenv("ZINC_MOE_TC") orelse return false;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// T2 (qwen MoE port): put the Q5_K routed DOWN projection on the grouped TC GEMM
/// too (in addition to gate/up). DEFAULT-ON when the gate/up TC path runs; opt out
/// with ZINC_MOE_DOWN_TC=0/off → down falls back to the matvec (the A/B baseline).
fn moeDownTcOn() bool {
    const v = std.posix.getenv("ZINC_MOE_DOWN_TC") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

/// T2 (qwen MoE port): extend down-TC to the 4 Q6_K down layers (gemm_q6k_experts_
/// grouped_tc), in addition to the 36 Q5_K layers. DEFAULT-ON when down-TC runs;
/// opt out / A/B-isolate the Q6_K layers with ZINC_MOE_DOWN_Q6K_TC=0/off → they
/// fall back to the matvec while Q5_K stays on TC.
fn moeDownQ6kTcOn() bool {
    const v = std.posix.getenv("ZINC_MOE_DOWN_Q6K_TC") orelse return true;
    return !(std.mem.eql(u8, v, "0") or std.ascii.eqlIgnoreCase(v, "off") or
        std.ascii.eqlIgnoreCase(v, "false") or std.ascii.eqlIgnoreCase(v, "no"));
}

fn boolU32(b: bool) u32 {
    return if (b) 1 else 0;
}

/// Bytes for one expert's [rows × cols] slice in a stacked MoE weight tensor.
/// rows*cols must be the per-expert element count; `q.blockSize()` divides cols
/// (the contiguous/quantized dim) and `q.bytesPerBlock()` is the block width.
fn expertSliceBytes(q: gguf.GGMLType, rows: u32, cols: u32) u32 {
    return rows * (cols / q.blockSize()) * q.bytesPerBlock();
}

/// Zero a device buffer by uploading a host-side zero array of `n` f32.
fn zeroBuffer(allocator: std.mem.Allocator, ctx: ?*shim.CudaCtx, buf: *CudaBuffer, n: u32) !void {
    const z = try allocator.alloc(f32, n);
    defer allocator.free(z);
    @memset(z, 0);
    buffer.upload(ctx, buf, std.mem.sliceAsBytes(z));
}
