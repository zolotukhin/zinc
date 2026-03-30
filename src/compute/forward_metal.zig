//! Metal inference engine — decode loop for Apple Silicon.
//! This is the Metal equivalent of forward.zig (Vulkan).
//! Uses MSL compute shaders dispatched via the Metal shim.
const std = @import("std");
const config_mod = @import("../model/config.zig");
const ModelConfig = config_mod.ModelConfig;
const gguf = @import("../model/gguf.zig");
const metal_loader = @import("../model/loader_metal.zig");
const metal_device = @import("../metal/device.zig");
const metal_buffer = @import("../metal/buffer.zig");
const MetalBuffer = metal_buffer.MetalBuffer;
const metal_pipeline = @import("../metal/pipeline.zig");
const MetalPipeline = metal_pipeline.MetalPipeline;
const metal_command = @import("../metal/command.zig");
const MetalCommand = metal_command.MetalCommand;
const shim = @import("../metal/c.zig").shim;

const log = std.log.scoped(.forward);

/// Runtime state for the decode loop.
pub const DecodeState = struct {
    position: u32,
    generated_tokens: std.ArrayList(u32),
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) DecodeState {
        return .{
            .position = 0,
            .generated_tokens = .{},
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *DecodeState) void {
        self.generated_tokens.deinit(self.allocator);
        self.* = undefined;
    }
};

/// Push constants for DMMV dispatch (matches GLSL layout).
const DmmvPush = extern struct {
    M: u32, // rows
    K: u32, // cols
    a_offset: u32,
    x_offset: u32,
    y_offset: u32,
};

/// Push constants for SwiGLU dispatch (matches SPIRV-Cross layout: buffer(0)).
const SwiGLUPush = extern struct {
    n: u32, // number of elements
};

/// Push constants for scale_accumulate dispatch (matches SPIRV-Cross layout: buffer(0)).
const ScaleAccPush = extern struct {
    n: u32, // number of elements
    scale_bits: u32, // float reinterpreted as uint32 (SPIRV-Cross convention)
};

/// Push constants for deinterleave dispatch (matches deinterleave.metal: buffer(0)).
const DeinterleavePush = extern struct {
    head_dim: u32,
    n_heads: u32,
};

/// Push constants for fused MoE accumulate dispatch (matches moe_accumulate.metal: buffer(10)).
/// Replaces 8+1 sequential scale_accumulate dispatches, eliminating 8 barriers per layer.
const MoeAccPush = extern struct {
    n: u32,
    w0: f32,
    w1: f32,
    w2: f32,
    w3: f32,
    w4: f32,
    w5: f32,
    w6: f32,
    w7: f32,
    w_sh: f32,
};

/// Push constants for batched Q4_K MoE DMMV.
/// Expert IDs are provided via a small CPU-written Metal buffer.
const MoeDmmvPush = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    expert_stride: u32,
    x_expert_stride: u32,
    x_offset: u32,
    y_offset: u32,
};

/// Push constants for fused batched MoE weighted accumulate.
/// Reads expert outputs from a contiguous [n_experts_used][hidden_dim] buffer.
const MoeAccBatchedPush = extern struct {
    n: u32,
    expert_stride: u32,
    w0: f32,
    w1: f32,
    w2: f32,
    w3: f32,
    w4: f32,
    w5: f32,
    w6: f32,
    w7: f32,
    w_sh: f32,
};

/// Push constants for GPU softmax + top-k routing.
const SoftmaxTopkPush = extern struct {
    n_experts: u32,
    k: u32,
};

/// Push constants for GPU-weighted batched MoE accumulation.
const MoeWeightedAccPush = extern struct {
    n: u32,
    n_used: u32,
    src_stride: u32,
};

/// Push constants for RMS norm dispatch (matches rms_norm_mul.metal: buffer(0)).
const RmsNormPush = extern struct {
    n: u32, // elements per group
    eps: f32, // epsilon
};

/// Push constants for sigmoid multiply dispatch (matches sigmoid_mul.metal: buffer(0)).
const SigmoidMulPush = extern struct {
    n: u32,
};

/// Push constants for RoPE dispatch (matches rope_fused.metal: buffer(0)).
const RopePush = extern struct {
    stride: u32,
    rope_dim: u32,
    n_heads: u32,
    position: u32,
    freq_base_bits: u32,
};

/// Push constants for flash attention dispatch (matches flash_attn.metal: buffer(0)).
const FlashAttnPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_len: u32,
    page_size: u32,
};

/// Push constants for GPU KV-cache writes.
const KvCacheWritePush = extern struct {
    n: u32,
    dst_offset: u32,
};

/// Push constants for SSM conv1d + SiLU dispatch (SPIRV-Cross: buffer(0)).
const SsmConv1dPush = extern struct {
    conv_channels: u32,
    d_conv: u32,
    kernel_is_f16: u32,
};

/// Push constants for SSM delta-net state update dispatch (SPIRV-Cross: buffer(0)).
const SsmDeltaNetPush = extern struct {
    d_inner: u32,
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    n_group: u32,
    ssm_a_is_f16: u32,
    dt_bias_is_f16: u32,
    has_dt_bias: u32,
    has_ssm_a: u32,
};

/// Push constants for SSM gated norm dispatch (SPIRV-Cross: buffer(0)).
const SsmGatedNormPush = extern struct {
    d_inner: u32,
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    norm_per_head: u32,
};

const GGMLType = gguf.GGMLType;

/// Cached per-layer tensor pointers (resolved once at init, not per token).
/// Eliminates 652 O(733) linear scans per token.
const LayerTensors = struct {
    // Attention projections (full attention layers)
    attn_q: ?*const metal_loader.LoadedTensor = null,
    attn_k: ?*const metal_loader.LoadedTensor = null,
    attn_v: ?*const metal_loader.LoadedTensor = null,
    attn_q_norm: ?*const metal_loader.LoadedTensor = null,
    attn_k_norm: ?*const metal_loader.LoadedTensor = null,
    attn_output: ?*const metal_loader.LoadedTensor = null,
    // SSM projections (SSM/delta-net layers)
    attn_qkv: ?*const metal_loader.LoadedTensor = null,
    attn_gate: ?*const metal_loader.LoadedTensor = null,
    ssm_alpha: ?*const metal_loader.LoadedTensor = null,
    ssm_beta: ?*const metal_loader.LoadedTensor = null,
    ssm_conv1d: ?*const metal_loader.LoadedTensor = null,
    ssm_dt_bias: ?*const metal_loader.LoadedTensor = null,
    ssm_a: ?*const metal_loader.LoadedTensor = null,
    ssm_norm: ?*const metal_loader.LoadedTensor = null,
    ssm_out: ?*const metal_loader.LoadedTensor = null,
    // MoE FFN
    ffn_gate_inp: ?*const metal_loader.LoadedTensor = null,
    ffn_gate_exps: ?*const metal_loader.LoadedTensor = null,
    ffn_up_exps: ?*const metal_loader.LoadedTensor = null,
    ffn_down_exps: ?*const metal_loader.LoadedTensor = null,
    // Shared expert
    ffn_gate_shexp: ?*const metal_loader.LoadedTensor = null,
    ffn_up_shexp: ?*const metal_loader.LoadedTensor = null,
    ffn_down_shexp: ?*const metal_loader.LoadedTensor = null,
    ffn_gate_inp_shexp: ?*const metal_loader.LoadedTensor = null,
    // Dense FFN (non-MoE)
    ffn_gate: ?*const metal_loader.LoadedTensor = null,
    ffn_up: ?*const metal_loader.LoadedTensor = null,
    ffn_down: ?*const metal_loader.LoadedTensor = null,
};

/// Metal inference engine — owns GPU buffers, pipelines, and KV cache.
pub const InferenceEngine = struct {
    model: *const metal_loader.Model,
    device: *const metal_device.MetalDevice,
    config: ModelConfig,
    allocator: std.mem.Allocator,

    // Intermediate buffers (Metal shared mode — CPU + GPU accessible)
    hidden_buf: MetalBuffer,
    residual_buf: MetalBuffer,
    norm_buf: MetalBuffer,
    q_buf: MetalBuffer,
    k_buf: MetalBuffer,
    v_buf: MetalBuffer,
    attn_out_buf: MetalBuffer,
    gate_buf: MetalBuffer,
    up_buf: MetalBuffer,
    swiglu_buf: MetalBuffer,
    down_buf: MetalBuffer,
    moe_out_buf: MetalBuffer,
    router_logits_buf: MetalBuffer,
    router_output_buf: MetalBuffer,
    logits_buf: MetalBuffer,
    embed_staging: MetalBuffer,
    expert_ids_buf: MetalBuffer,

    // Batched MoE buffers ([n_experts_used][dim], contiguous)
    expert_gate_batch_buf: MetalBuffer,
    expert_up_batch_buf: MetalBuffer,
    expert_swiglu_batch_buf: MetalBuffer,
    expert_down_batch_buf: MetalBuffer,

    // Per-expert intermediate buffers for parallel MoE dispatch
    expert_gate_bufs: []MetalBuffer,
    expert_up_bufs: []MetalBuffer,
    expert_swiglu_bufs: []MetalBuffer,
    expert_down_bufs: []MetalBuffer,

    // KV cache (per layer)
    kv_k_cache: []MetalBuffer,
    kv_v_cache: []MetalBuffer,
    page_table_buf: MetalBuffer,

    // DMMV compute pipelines (one per quant type)
    dmmv_q4k_pipe: MetalPipeline,
    dmmv_q4k_k2048_pipe: MetalPipeline,
    dmmv_q4k_lmhead_pipe: MetalPipeline,
    dmmv_q4k_lmhead_1024_pipe: MetalPipeline,
    dmmv_q5k_pipe: MetalPipeline,
    dmmv_q6k_pipe: MetalPipeline,
    dmmv_q8_0_pipe: MetalPipeline,
    dmmv_f16_pipe: MetalPipeline,
    dmmv_f32_pipe: MetalPipeline,
    dmmv_q4k_moe_pipe: MetalPipeline,
    dmmv_q4k_moe_k2048_pipe: MetalPipeline,
    dmmv_q4k_moe_k2048_1024_pipe: MetalPipeline,

    // Elementwise compute pipelines (for batched GPU dispatch)
    deinterleave_pipe: MetalPipeline,
    flash_attn_pipe: MetalPipeline,
    kv_cache_write_pipe: MetalPipeline,
    rope_pipe: MetalPipeline,
    sigmoid_mul_pipe: MetalPipeline,
    swiglu_pipe: MetalPipeline,
    swiglu_batched_pipe: MetalPipeline,
    scale_acc_pipe: MetalPipeline,
    rms_norm_pipe: MetalPipeline,
    moe_acc_pipe: MetalPipeline,
    moe_acc_batched_pipe: MetalPipeline,
    softmax_topk_pipe: MetalPipeline,
    sigmoid_scale_acc_pipe: MetalPipeline,
    moe_weighted_acc_pipe: MetalPipeline,

    // Preloaded norm weight buffers (f32, GPU-accessible via UMA)
    attn_norm_bufs: []MetalBuffer,
    attn_q_norm_bufs: []MetalBuffer,
    attn_k_norm_bufs: []MetalBuffer,
    attn_q_norm_present: []bool,
    attn_k_norm_present: []bool,
    ffn_norm_bufs: []MetalBuffer,
    final_norm_gpu: MetalBuffer,

    // SSM GPU pipelines (cross-compiled from GLSL via SPIRV-Cross)
    ssm_conv1d_pipe: MetalPipeline,
    ssm_delta_net_pipe: MetalPipeline,
    ssm_gated_norm_pipe: MetalPipeline,

    // SSM state (Metal buffers — GPU-resident, persistent across tokens)
    ssm_conv_state_bufs: ?[]MetalBuffer,
    ssm_state_bufs: ?[]MetalBuffer,

    // SSM constants as Metal buffers (f32, GPU-accessible via UMA)
    ssm_conv_kernel_bufs: ?[]MetalBuffer,
    ssm_dt_bias_bufs: ?[]MetalBuffer,
    ssm_a_bufs: ?[]MetalBuffer,
    ssm_norm_weight_bufs: ?[]MetalBuffer,
    ssm_norm_per_head: ?[]bool,

    // Preloaded shared expert gate weights (per-layer, f32 — eliminates mid-MoE commitAndWait)
    shexp_gate_weights: ?[][]f32,

    // Cached per-layer tensor pointers (init-time, eliminates per-token O(733) scans)
    layer_tensors: []LayerTensors,
    token_embed: *const metal_loader.LoadedTensor,
    lm_head: *const metal_loader.LoadedTensor,

    // Decode state
    position: u32,

    pub fn init(
        model: *const metal_loader.Model,
        device: *const metal_device.MetalDevice,
        allocator: std.mem.Allocator,
    ) !InferenceEngine {
        const cfg = model.config;
        const ctx = device.ctx;

        // Compute dimension-dependent sizes
        const q_dim: u32 = cfg.n_heads * cfg.head_dim;
        const kv_dim: u32 = cfg.n_kv_heads * cfg.head_dim;
        const inter_dim: u32 = if (cfg.intermediate_dim > 0) cfg.intermediate_dim else cfg.hidden_dim * 4;
        const shexp_inter_dim: u32 = if (cfg.shared_expert_intermediate_dim > 0) cfg.shared_expert_intermediate_dim else inter_dim;
        const d_inner: u32 = cfg.ssm_d_inner;
        const conv_channels: u32 = if (d_inner > 0) d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state else 0;

        // Buffer sizes (max across all uses)
        const hidden_size: usize = @as(usize, cfg.hidden_dim) * @sizeOf(f32);
        const q_gate_size: usize = @as(usize, q_dim) * 2 * @sizeOf(f32);
        const attn_out_size: usize = @max(q_gate_size, @as(usize, conv_channels) * @sizeOf(f32));
        const head_total: usize = @as(usize, q_dim) * @sizeOf(f32);
        const kv_total: usize = @as(usize, kv_dim) * @sizeOf(f32);
        const gate_size: usize = @max(@max(head_total, @as(usize, inter_dim) * @sizeOf(f32)), @as(usize, d_inner) * @sizeOf(f32));
        const up_size: usize = @max(@as(usize, inter_dim) * @sizeOf(f32), @as(usize, shexp_inter_dim) * @sizeOf(f32));
        const swiglu_size: usize = @max(up_size, @as(usize, conv_channels) * @sizeOf(f32));
        const vocab_size: usize = @as(usize, cfg.vocab_size) * @sizeOf(f32);
        const kv_cache_size: usize = @as(usize, 4096) * kv_dim * @sizeOf(f32);
        const page_table_size: usize = @as(usize, 4096) * @sizeOf(u32);
        const router_size: usize = @max(@as(usize, cfg.n_experts), @as(usize, if (cfg.ssm_dt_rank > 0) cfg.ssm_dt_rank else 1)) * @sizeOf(f32);
        const router_output_size: usize = @max(@as(usize, cfg.n_experts_used) * 2 * @sizeOf(u32), 8);
        const expert_ids_size: usize = @max(@as(usize, cfg.n_experts_used) * @sizeOf(u32), 4);
        const expert_inter_batch_size: usize = @max(@as(usize, cfg.n_experts_used) * @as(usize, inter_dim) * @sizeOf(f32), 4);
        const expert_hidden_batch_size: usize = @max(@as(usize, cfg.n_experts_used) * @as(usize, cfg.hidden_dim) * @sizeOf(f32), 4);

        // Allocate intermediate buffers
        var self: InferenceEngine = undefined;
        self.model = model;
        self.device = device;
        self.config = cfg;
        self.allocator = allocator;
        self.position = 0;

        self.hidden_buf = try metal_buffer.createBuffer(ctx, hidden_size);
        self.residual_buf = try metal_buffer.createBuffer(ctx, hidden_size);
        self.norm_buf = try metal_buffer.createBuffer(ctx, hidden_size);
        self.q_buf = try metal_buffer.createBuffer(ctx, head_total);
        self.k_buf = try metal_buffer.createBuffer(ctx, kv_total);
        self.v_buf = try metal_buffer.createBuffer(ctx, kv_total);
        self.attn_out_buf = try metal_buffer.createBuffer(ctx, @max(attn_out_size, 4));
        self.gate_buf = try metal_buffer.createBuffer(ctx, @max(gate_size, 4));
        self.up_buf = try metal_buffer.createBuffer(ctx, @max(up_size, 4));
        self.swiglu_buf = try metal_buffer.createBuffer(ctx, @max(swiglu_size, 4));
        self.down_buf = try metal_buffer.createBuffer(ctx, hidden_size);
        self.moe_out_buf = try metal_buffer.createBuffer(ctx, hidden_size);
        self.router_logits_buf = try metal_buffer.createBuffer(ctx, @max(router_size, 4));
        self.router_output_buf = try metal_buffer.createBuffer(ctx, router_output_size);
        self.logits_buf = try metal_buffer.createBuffer(ctx, vocab_size);
        self.embed_staging = try metal_buffer.createBuffer(ctx, hidden_size);
        self.expert_ids_buf = try metal_buffer.createBuffer(ctx, expert_ids_size);
        self.expert_gate_batch_buf = try metal_buffer.createBuffer(ctx, expert_inter_batch_size);
        self.expert_up_batch_buf = try metal_buffer.createBuffer(ctx, expert_inter_batch_size);
        self.expert_swiglu_batch_buf = try metal_buffer.createBuffer(ctx, expert_inter_batch_size);
        self.expert_down_batch_buf = try metal_buffer.createBuffer(ctx, expert_hidden_batch_size);

        // Per-expert intermediate buffers for parallel MoE dispatch (reduces barriers from ~35 to 2 per layer)
        if (cfg.n_experts_used > 0) {
            self.expert_gate_bufs = try allocator.alloc(MetalBuffer, cfg.n_experts_used);
            self.expert_up_bufs = try allocator.alloc(MetalBuffer, cfg.n_experts_used);
            self.expert_swiglu_bufs = try allocator.alloc(MetalBuffer, cfg.n_experts_used);
            self.expert_down_bufs = try allocator.alloc(MetalBuffer, cfg.n_experts_used);
            for (0..cfg.n_experts_used) |i| {
                self.expert_gate_bufs[i] = try metal_buffer.createBuffer(ctx, @max(@as(usize, inter_dim) * @sizeOf(f32), 4));
                self.expert_up_bufs[i] = try metal_buffer.createBuffer(ctx, @max(@as(usize, inter_dim) * @sizeOf(f32), 4));
                self.expert_swiglu_bufs[i] = try metal_buffer.createBuffer(ctx, @max(@as(usize, inter_dim) * @sizeOf(f32), 4));
                self.expert_down_bufs[i] = try metal_buffer.createBuffer(ctx, @max(hidden_size, 4));
            }
        } else {
            self.expert_gate_bufs = try allocator.alloc(MetalBuffer, 0);
            self.expert_up_bufs = try allocator.alloc(MetalBuffer, 0);
            self.expert_swiglu_bufs = try allocator.alloc(MetalBuffer, 0);
            self.expert_down_bufs = try allocator.alloc(MetalBuffer, 0);
        }

        // Allocate KV cache per layer
        self.kv_k_cache = try allocator.alloc(MetalBuffer, cfg.n_layers);
        self.kv_v_cache = try allocator.alloc(MetalBuffer, cfg.n_layers);
        for (0..cfg.n_layers) |i| {
            self.kv_k_cache[i] = try metal_buffer.createBuffer(ctx, kv_cache_size);
            self.kv_v_cache[i] = try metal_buffer.createBuffer(ctx, kv_cache_size);
        }
        self.page_table_buf = try metal_buffer.createBuffer(ctx, page_table_size);
        {
            const page_table_ptr: [*]u32 = @ptrCast(@alignCast(self.page_table_buf.cpu_ptr.?));
            for (0..4096) |i| {
                page_table_ptr[i] = @intCast(i);
            }
        }

        // Load DMMV compute pipelines for all quant types
        self.dmmv_q4k_pipe = try loadShaderPipeline(ctx, "dmmv_q4k");
        self.dmmv_q4k_k2048_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_k2048");
        self.dmmv_q4k_lmhead_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_lmhead");
        self.dmmv_q4k_lmhead_1024_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_lmhead_1024");
        self.dmmv_q5k_pipe = try loadShaderPipeline(ctx, "dmmv_q5k");
        self.dmmv_q6k_pipe = try loadShaderPipeline(ctx, "dmmv_q6k");
        self.dmmv_q8_0_pipe = try loadShaderPipeline(ctx, "dmmv_q8_0");
        self.dmmv_f16_pipe = try loadShaderPipeline(ctx, "dmmv_f16");
        self.dmmv_f32_pipe = try loadShaderPipeline(ctx, "dmmv_f32");
        self.dmmv_q4k_moe_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe");
        self.dmmv_q4k_moe_k2048_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe_k2048");
        self.dmmv_q4k_moe_k2048_1024_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe_k2048_1024");

        // Elementwise pipelines for batched GPU dispatch
        self.deinterleave_pipe = try loadShaderPipeline(ctx, "deinterleave");
        self.flash_attn_pipe = try loadShaderPipeline(ctx, "flash_attn");
        self.kv_cache_write_pipe = try loadShaderPipeline(ctx, "kv_cache_write");
        self.rope_pipe = try loadShaderPipeline(ctx, "rope_fused");
        self.sigmoid_mul_pipe = try loadShaderPipeline(ctx, "sigmoid_mul");
        self.swiglu_pipe = try loadShaderPipeline(ctx, "swiglu");
        self.swiglu_batched_pipe = try loadShaderPipeline(ctx, "swiglu_batched");
        self.scale_acc_pipe = try loadShaderPipeline(ctx, "scale_accumulate");
        self.rms_norm_pipe = try loadShaderPipeline(ctx, "rms_norm_mul");
        self.moe_acc_pipe = try loadShaderPipeline(ctx, "moe_accumulate");
        self.moe_acc_batched_pipe = try loadShaderPipeline(ctx, "moe_accumulate_batched");
        self.softmax_topk_pipe = try loadShaderPipeline(ctx, "softmax_topk");
        self.sigmoid_scale_acc_pipe = try loadShaderPipeline(ctx, "sigmoid_scale_acc");
        self.moe_weighted_acc_pipe = try loadShaderPipeline(ctx, "moe_weighted_acc");

        // Preload norm weights into f32 Metal buffers (eliminates per-token alloc + mmap dequant)
        self.attn_norm_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
        self.attn_q_norm_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
        self.attn_k_norm_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
        self.attn_q_norm_present = try allocator.alloc(bool, cfg.n_layers);
        self.attn_k_norm_present = try allocator.alloc(bool, cfg.n_layers);
        self.ffn_norm_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
        for (0..cfg.n_layers) |i| {
            const layer: u32 = @intCast(i);
            const an = findLayerTensor(model, layer, "attn_norm.weight") orelse return error.MissingTensor;
            self.attn_norm_bufs[i] = try preloadNormWeights(ctx, model, an, cfg.hidden_dim);
            if (findLayerTensor(model, layer, "attn_q_norm.weight")) |qn| {
                self.attn_q_norm_bufs[i] = try preloadNormWeights(ctx, model, qn, cfg.head_dim);
                self.attn_q_norm_present[i] = true;
            } else {
                self.attn_q_norm_bufs[i] = try metal_buffer.createBuffer(ctx, 4);
                @memset(self.attn_q_norm_bufs[i].cpu_ptr.?[0..4], 0);
                self.attn_q_norm_present[i] = false;
            }
            if (findLayerTensor(model, layer, "attn_k_norm.weight")) |kn| {
                self.attn_k_norm_bufs[i] = try preloadNormWeights(ctx, model, kn, cfg.head_dim);
                self.attn_k_norm_present[i] = true;
            } else {
                self.attn_k_norm_bufs[i] = try metal_buffer.createBuffer(ctx, 4);
                @memset(self.attn_k_norm_bufs[i].cpu_ptr.?[0..4], 0);
                self.attn_k_norm_present[i] = false;
            }
            const fn_t = findLayerTensor(model, layer, "post_attention_norm.weight") orelse
                findLayerTensor(model, layer, "ffn_norm.weight") orelse return error.MissingTensor;
            self.ffn_norm_bufs[i] = try preloadNormWeights(ctx, model, fn_t, cfg.hidden_dim);
        }
        const final_t = findTensorByName(model, "output_norm.weight") orelse return error.MissingTensor;
        self.final_norm_gpu = try preloadNormWeights(ctx, model, final_t, cfg.hidden_dim);

        // SSM GPU pipelines
        self.ssm_conv1d_pipe = try loadShaderPipeline(ctx, "ssm_conv1d");
        self.ssm_delta_net_pipe = try loadShaderPipeline(ctx, "ssm_delta_net");
        self.ssm_gated_norm_pipe = try loadShaderPipeline(ctx, "ssm_gated_norm");

        // SSM state + constants as Metal buffers (GPU-resident via UMA)
        if (d_inner > 0 and cfg.ssm_d_conv > 0) {
            const d_conv_1 = cfg.ssm_d_conv - 1;
            const head_v_dim = d_inner / @max(cfg.ssm_dt_rank, 1);
            const mmap_data = model.mmap_data orelse return error.NoMmapData;
            const tensor_data_off = model.gguf_file.tensor_data_offset;
            const conv_kernel_len: u32 = conv_channels * cfg.ssm_d_conv;

            self.ssm_conv_state_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
            self.ssm_state_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
            self.ssm_conv_kernel_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
            self.ssm_dt_bias_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
            self.ssm_a_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
            self.ssm_norm_weight_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
            self.ssm_norm_per_head = try allocator.alloc(bool, cfg.n_layers);

            for (0..cfg.n_layers) |i| {
                const layer_i: u32 = @intCast(i);
                // Conv state: (d_conv-1) * conv_channels floats, zero-initialized
                const cs_size: usize = @as(usize, d_conv_1) * conv_channels * @sizeOf(f32);
                self.ssm_conv_state_bufs.?[i] = try metal_buffer.createBuffer(ctx, @max(cs_size, 4));
                @memset(self.ssm_conv_state_bufs.?[i].cpu_ptr.?[0..@max(cs_size, 4)], 0);
                // Recurrent state: dt_rank * head_v_dim * head_v_dim floats
                const st_size: usize = @as(usize, cfg.ssm_dt_rank) * head_v_dim * head_v_dim * @sizeOf(f32);
                self.ssm_state_bufs.?[i] = try metal_buffer.createBuffer(ctx, @max(st_size, 4));
                @memset(self.ssm_state_bufs.?[i].cpu_ptr.?[0..@max(st_size, 4)], 0);
                // Conv kernel: dequant to f32
                const ck_size: usize = @as(usize, conv_kernel_len) * @sizeOf(f32);
                self.ssm_conv_kernel_bufs.?[i] = try metal_buffer.createBuffer(ctx, @max(ck_size, 4));
                if (findLayerTensor(model, layer_i, "ssm_conv1d.weight")) |t| {
                    const ck_ptr: [*]f32 = @ptrCast(@alignCast(self.ssm_conv_kernel_bufs.?[i].cpu_ptr.?));
                    const off: usize = @intCast(tensor_data_off + t.info.offset);
                    readMmapFloats(mmap_data, off, t.info.type_, ck_ptr[0..conv_kernel_len]);
                } else @memset(self.ssm_conv_kernel_bufs.?[i].cpu_ptr.?[0..@max(ck_size, 4)], 0);
                // dt_bias: dequant to f32
                const dtb_size: usize = @as(usize, cfg.ssm_dt_rank) * @sizeOf(f32);
                self.ssm_dt_bias_bufs.?[i] = try metal_buffer.createBuffer(ctx, @max(dtb_size, 4));
                if (findLayerTensor(model, layer_i, "ssm_dt.bias")) |t| {
                    const dtb_ptr: [*]f32 = @ptrCast(@alignCast(self.ssm_dt_bias_bufs.?[i].cpu_ptr.?));
                    const off: usize = @intCast(tensor_data_off + t.info.offset);
                    readMmapFloats(mmap_data, off, t.info.type_, dtb_ptr[0..cfg.ssm_dt_rank]);
                } else @memset(self.ssm_dt_bias_bufs.?[i].cpu_ptr.?[0..@max(dtb_size, 4)], 0);
                // ssm_a: dequant to f32
                const a_size: usize = @as(usize, cfg.ssm_dt_rank) * @sizeOf(f32);
                self.ssm_a_bufs.?[i] = try metal_buffer.createBuffer(ctx, @max(a_size, 4));
                if (findLayerTensor(model, layer_i, "ssm_a")) |t| {
                    const a_ptr: [*]f32 = @ptrCast(@alignCast(self.ssm_a_bufs.?[i].cpu_ptr.?));
                    const off: usize = @intCast(tensor_data_off + t.info.offset);
                    readMmapFloats(mmap_data, off, t.info.type_, a_ptr[0..cfg.ssm_dt_rank]);
                } else @memset(self.ssm_a_bufs.?[i].cpu_ptr.?[0..@max(a_size, 4)], 0);
                // norm weights: dequant to f32
                if (findLayerTensor(model, layer_i, "ssm_norm.weight")) |t| {
                    const ne: u32 = @intCast(t.info.numElements());
                    const nw_size: usize = @as(usize, ne) * @sizeOf(f32);
                    self.ssm_norm_weight_bufs.?[i] = try metal_buffer.createBuffer(ctx, @max(nw_size, 4));
                    const nw_ptr: [*]f32 = @ptrCast(@alignCast(self.ssm_norm_weight_bufs.?[i].cpu_ptr.?));
                    const off: usize = @intCast(tensor_data_off + t.info.offset);
                    readMmapFloats(mmap_data, off, t.info.type_, nw_ptr[0..ne]);
                    self.ssm_norm_per_head.?[i] = ne >= d_inner;
                } else {
                    self.ssm_norm_weight_bufs.?[i] = try metal_buffer.createBuffer(ctx, 4);
                    @memset(self.ssm_norm_weight_bufs.?[i].cpu_ptr.?[0..4], 0);
                    self.ssm_norm_per_head.?[i] = false;
                }
            }
        } else {
            self.ssm_conv_state_bufs = null;
            self.ssm_state_bufs = null;
            self.ssm_conv_kernel_bufs = null;
            self.ssm_dt_bias_bufs = null;
            self.ssm_a_bufs = null;
            self.ssm_norm_weight_bufs = null;
            self.ssm_norm_per_head = null;
        }

        // Preload shared expert gate weights (1 × hidden_dim per layer, avoids mid-MoE commitAndWait)
        if (cfg.n_experts > 0) {
            self.shexp_gate_weights = try allocator.alloc([]f32, cfg.n_layers);
            for (0..cfg.n_layers) |i| {
                const layer_i: u32 = @intCast(i);
                if (findLayerTensor(model, layer_i, "ffn_gate_inp_shexp.weight")) |t| {
                    const shexp_mmap = model.mmap_data orelse return error.NoMmapData;
                    self.shexp_gate_weights.?[i] = try allocator.alloc(f32, cfg.hidden_dim);
                    const off: usize = @intCast(model.gguf_file.tensor_data_offset + t.info.offset);
                    readMmapFloats(shexp_mmap, off, t.info.type_, self.shexp_gate_weights.?[i]);
                } else {
                    self.shexp_gate_weights.?[i] = try allocator.alloc(f32, 1);
                    self.shexp_gate_weights.?[i][0] = 0;
                }
            }
        } else {
            self.shexp_gate_weights = null;
        }

        // Cache per-layer tensor pointers (eliminates 652 O(733) linear scans per token)
        self.layer_tensors = try allocator.alloc(LayerTensors, cfg.n_layers);
        for (0..cfg.n_layers) |i| {
            const layer: u32 = @intCast(i);
            self.layer_tensors[i] = .{
                .attn_q = findLayerTensor(model, layer, "attn_q.weight"),
                .attn_k = findLayerTensor(model, layer, "attn_k.weight"),
                .attn_v = findLayerTensor(model, layer, "attn_v.weight"),
                .attn_q_norm = findLayerTensor(model, layer, "attn_q_norm.weight"),
                .attn_k_norm = findLayerTensor(model, layer, "attn_k_norm.weight"),
                .attn_output = findLayerTensor(model, layer, "attn_output.weight"),
                .attn_qkv = findLayerTensor(model, layer, "attn_qkv.weight"),
                .attn_gate = findLayerTensor(model, layer, "attn_gate.weight"),
                .ssm_alpha = findLayerTensor(model, layer, "ssm_alpha.weight"),
                .ssm_beta = findLayerTensor(model, layer, "ssm_beta.weight"),
                .ssm_conv1d = findLayerTensor(model, layer, "ssm_conv1d.weight"),
                .ssm_dt_bias = findLayerTensor(model, layer, "ssm_dt.bias"),
                .ssm_a = findLayerTensor(model, layer, "ssm_a"),
                .ssm_norm = findLayerTensor(model, layer, "ssm_norm.weight"),
                .ssm_out = findLayerTensor(model, layer, "ssm_out.weight"),
                .ffn_gate_inp = findLayerTensor(model, layer, "ffn_gate_inp.weight"),
                .ffn_gate_exps = findLayerTensor(model, layer, "ffn_gate_exps.weight"),
                .ffn_up_exps = findLayerTensor(model, layer, "ffn_up_exps.weight"),
                .ffn_down_exps = findLayerTensor(model, layer, "ffn_down_exps.weight"),
                .ffn_gate_shexp = findLayerTensor(model, layer, "ffn_gate_shexp.weight"),
                .ffn_up_shexp = findLayerTensor(model, layer, "ffn_up_shexp.weight"),
                .ffn_down_shexp = findLayerTensor(model, layer, "ffn_down_shexp.weight"),
                .ffn_gate_inp_shexp = findLayerTensor(model, layer, "ffn_gate_inp_shexp.weight"),
                .ffn_gate = findLayerTensor(model, layer, "ffn_gate.weight"),
                .ffn_up = findLayerTensor(model, layer, "ffn_up.weight"),
                .ffn_down = findLayerTensor(model, layer, "ffn_down.weight"),
            };
        }
        self.lm_head = findTensorByName(model, "output.weight") orelse
            findTensorByName(model, "token_embd.weight") orelse return error.MissingTensor;
        self.token_embed = findTensorByName(model, "token_embd.weight") orelse return error.MissingTensor;

        log.info("Metal inference engine initialized: {d} layers, {d}x{d} heads, dim={d}", .{
            cfg.n_layers, cfg.n_heads, cfg.head_dim, cfg.hidden_dim,
        });
        log.info(
            "Metal pipeline caps: dmmv_q4k tw={d} max={d} stgmem={d} | dmmv_q4k_k2048 tw={d} max={d} stgmem={d} | lmhead512 tw={d} max={d} stgmem={d} | lmhead1024 tw={d} max={d} stgmem={d}",
            .{
                self.dmmv_q4k_pipe.thread_execution_width,
                self.dmmv_q4k_pipe.max_threads_per_threadgroup,
                self.dmmv_q4k_pipe.static_threadgroup_memory_length,
                self.dmmv_q4k_k2048_pipe.thread_execution_width,
                self.dmmv_q4k_k2048_pipe.max_threads_per_threadgroup,
                self.dmmv_q4k_k2048_pipe.static_threadgroup_memory_length,
                self.dmmv_q4k_lmhead_pipe.thread_execution_width,
                self.dmmv_q4k_lmhead_pipe.max_threads_per_threadgroup,
                self.dmmv_q4k_lmhead_pipe.static_threadgroup_memory_length,
                self.dmmv_q4k_lmhead_1024_pipe.thread_execution_width,
                self.dmmv_q4k_lmhead_1024_pipe.max_threads_per_threadgroup,
                self.dmmv_q4k_lmhead_1024_pipe.static_threadgroup_memory_length,
            },
        );
        log.info(
            "Metal pipeline caps: dmmv_q4k_moe tw={d} max={d} stgmem={d} | dmmv_q4k_moe_k2048 tw={d} max={d} stgmem={d} | dmmv_q4k_moe_k2048_1024 tw={d} max={d} stgmem={d}",
            .{
                self.dmmv_q4k_moe_pipe.thread_execution_width,
                self.dmmv_q4k_moe_pipe.max_threads_per_threadgroup,
                self.dmmv_q4k_moe_pipe.static_threadgroup_memory_length,
                self.dmmv_q4k_moe_k2048_pipe.thread_execution_width,
                self.dmmv_q4k_moe_k2048_pipe.max_threads_per_threadgroup,
                self.dmmv_q4k_moe_k2048_pipe.static_threadgroup_memory_length,
                self.dmmv_q4k_moe_k2048_1024_pipe.thread_execution_width,
                self.dmmv_q4k_moe_k2048_1024_pipe.max_threads_per_threadgroup,
                self.dmmv_q4k_moe_k2048_1024_pipe.static_threadgroup_memory_length,
            },
        );
        log.info(
            "Metal pipeline caps: rms_norm tw={d} max={d} | swiglu tw={d} max={d} | swiglu_batched tw={d} max={d} | moe_acc_batched tw={d} max={d}",
            .{
                self.rms_norm_pipe.thread_execution_width,
                self.rms_norm_pipe.max_threads_per_threadgroup,
                self.swiglu_pipe.thread_execution_width,
                self.swiglu_pipe.max_threads_per_threadgroup,
                self.swiglu_batched_pipe.thread_execution_width,
                self.swiglu_batched_pipe.max_threads_per_threadgroup,
                self.moe_acc_batched_pipe.thread_execution_width,
                self.moe_acc_batched_pipe.max_threads_per_threadgroup,
            },
        );

        return self;
    }

    pub fn deinit(self: *InferenceEngine) void {
        metal_buffer.freeBuffer(&self.hidden_buf);
        metal_buffer.freeBuffer(&self.residual_buf);
        metal_buffer.freeBuffer(&self.norm_buf);
        metal_buffer.freeBuffer(&self.q_buf);
        metal_buffer.freeBuffer(&self.k_buf);
        metal_buffer.freeBuffer(&self.v_buf);
        metal_buffer.freeBuffer(&self.attn_out_buf);
        metal_buffer.freeBuffer(&self.gate_buf);
        metal_buffer.freeBuffer(&self.up_buf);
        metal_buffer.freeBuffer(&self.swiglu_buf);
        metal_buffer.freeBuffer(&self.down_buf);
        metal_buffer.freeBuffer(&self.moe_out_buf);
        metal_buffer.freeBuffer(&self.router_logits_buf);
        metal_buffer.freeBuffer(&self.router_output_buf);
        metal_buffer.freeBuffer(&self.logits_buf);
        metal_buffer.freeBuffer(&self.embed_staging);
        metal_buffer.freeBuffer(&self.expert_ids_buf);
        metal_buffer.freeBuffer(&self.expert_gate_batch_buf);
        metal_buffer.freeBuffer(&self.expert_up_batch_buf);
        metal_buffer.freeBuffer(&self.expert_swiglu_batch_buf);
        metal_buffer.freeBuffer(&self.expert_down_batch_buf);

        for (0..self.expert_gate_bufs.len) |i| {
            metal_buffer.freeBuffer(&self.expert_gate_bufs[i]);
            metal_buffer.freeBuffer(&self.expert_up_bufs[i]);
            metal_buffer.freeBuffer(&self.expert_swiglu_bufs[i]);
            metal_buffer.freeBuffer(&self.expert_down_bufs[i]);
        }
        self.allocator.free(self.expert_gate_bufs);
        self.allocator.free(self.expert_up_bufs);
        self.allocator.free(self.expert_swiglu_bufs);
        self.allocator.free(self.expert_down_bufs);

        for (0..self.config.n_layers) |i| {
            metal_buffer.freeBuffer(&self.kv_k_cache[i]);
            metal_buffer.freeBuffer(&self.kv_v_cache[i]);
        }
        self.allocator.free(self.kv_k_cache);
        self.allocator.free(self.kv_v_cache);
        metal_buffer.freeBuffer(&self.page_table_buf);

        metal_pipeline.freePipeline(&self.dmmv_q4k_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q4k_k2048_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q4k_lmhead_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q4k_lmhead_1024_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q5k_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q6k_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q8_0_pipe);
        metal_pipeline.freePipeline(&self.dmmv_f16_pipe);
        metal_pipeline.freePipeline(&self.dmmv_f32_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q4k_moe_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q4k_moe_k2048_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q4k_moe_k2048_1024_pipe);
        metal_pipeline.freePipeline(&self.deinterleave_pipe);
        metal_pipeline.freePipeline(&self.flash_attn_pipe);
        metal_pipeline.freePipeline(&self.kv_cache_write_pipe);
        metal_pipeline.freePipeline(&self.rope_pipe);
        metal_pipeline.freePipeline(&self.sigmoid_mul_pipe);
        metal_pipeline.freePipeline(&self.swiglu_pipe);
        metal_pipeline.freePipeline(&self.swiglu_batched_pipe);
        metal_pipeline.freePipeline(&self.scale_acc_pipe);
        metal_pipeline.freePipeline(&self.rms_norm_pipe);
        metal_pipeline.freePipeline(&self.moe_acc_pipe);
        metal_pipeline.freePipeline(&self.moe_acc_batched_pipe);
        metal_pipeline.freePipeline(&self.softmax_topk_pipe);
        metal_pipeline.freePipeline(&self.sigmoid_scale_acc_pipe);
        metal_pipeline.freePipeline(&self.moe_weighted_acc_pipe);

        for (0..self.config.n_layers) |i| {
            metal_buffer.freeBuffer(&self.attn_norm_bufs[i]);
            metal_buffer.freeBuffer(&self.attn_q_norm_bufs[i]);
            metal_buffer.freeBuffer(&self.attn_k_norm_bufs[i]);
            metal_buffer.freeBuffer(&self.ffn_norm_bufs[i]);
        }
        self.allocator.free(self.attn_norm_bufs);
        self.allocator.free(self.attn_q_norm_bufs);
        self.allocator.free(self.attn_k_norm_bufs);
        self.allocator.free(self.attn_q_norm_present);
        self.allocator.free(self.attn_k_norm_present);
        self.allocator.free(self.ffn_norm_bufs);
        metal_buffer.freeBuffer(&self.final_norm_gpu);
        self.allocator.free(self.layer_tensors);

        metal_pipeline.freePipeline(&self.ssm_conv1d_pipe);
        metal_pipeline.freePipeline(&self.ssm_delta_net_pipe);
        metal_pipeline.freePipeline(&self.ssm_gated_norm_pipe);

        if (self.ssm_conv_state_bufs) |bufs| {
            for (0..bufs.len) |i| metal_buffer.freeBuffer(&self.ssm_conv_state_bufs.?[i]);
            self.allocator.free(bufs);
        }
        if (self.ssm_state_bufs) |bufs| {
            for (0..bufs.len) |i| metal_buffer.freeBuffer(&self.ssm_state_bufs.?[i]);
            self.allocator.free(bufs);
        }
        if (self.ssm_conv_kernel_bufs) |bufs| {
            for (0..bufs.len) |i| metal_buffer.freeBuffer(&self.ssm_conv_kernel_bufs.?[i]);
            self.allocator.free(bufs);
        }
        if (self.ssm_dt_bias_bufs) |bufs| {
            for (0..bufs.len) |i| metal_buffer.freeBuffer(&self.ssm_dt_bias_bufs.?[i]);
            self.allocator.free(bufs);
        }
        if (self.ssm_a_bufs) |bufs| {
            for (0..bufs.len) |i| metal_buffer.freeBuffer(&self.ssm_a_bufs.?[i]);
            self.allocator.free(bufs);
        }
        if (self.ssm_norm_weight_bufs) |bufs| {
            for (0..bufs.len) |i| metal_buffer.freeBuffer(&self.ssm_norm_weight_bufs.?[i]);
            self.allocator.free(bufs);
        }
        if (self.ssm_norm_per_head) |arr| self.allocator.free(arr);
        if (self.shexp_gate_weights) |arr| {
            for (arr) |s| self.allocator.free(s);
            self.allocator.free(arr);
        }
    }

    /// Sample the next token greedily (argmax over logits).
    pub fn sampleGreedy(self: *const InferenceEngine) u32 {
        const logits_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_buf.cpu_ptr.?));
        const logits = logits_ptr[0..self.config.vocab_size];
        var max_val: f32 = -std.math.inf(f32);
        var max_idx: u32 = 0;
        for (logits, 0..) |v, i| {
            if (v > max_val) {
                max_val = v;
                max_idx = @intCast(i);
            }
        }
        return max_idx;
    }

    pub fn resetRequestState(self: *InferenceEngine) void {
        self.position = 0;

        if (self.ssm_conv_state_bufs) |bufs| {
            for (bufs) |buf| {
                @memset(buf.cpu_ptr.?[0..buf.size], 0);
            }
        }
        if (self.ssm_state_bufs) |bufs| {
            for (bufs) |buf| {
                @memset(buf.cpu_ptr.?[0..buf.size], 0);
            }
        }
    }

    pub fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {
        self.resetRequestState();
        state.position = 0;
        state.generated_tokens.clearRetainingCapacity();

        for (prompt_tokens) |token_id| {
            try self.loadTokenEmbedding(token_id);
            try runDecodeStep(self);
        }
        state.position = self.position;
    }

    pub fn decodeStep(self: *InferenceEngine, state: *DecodeState, token_id: u32) !void {
        try self.loadTokenEmbedding(token_id);
        try runDecodeStep(self);
        state.position = self.position;
    }

    fn loadTokenEmbedding(self: *InferenceEngine, token_id: u32) !void {
        const mmap = self.model.mmap_data orelse return error.NoMmapData;
        const embed_data_offset = self.model.gguf_file.tensor_data_offset + self.token_embed.info.offset;
        const embed_raw = mmap[embed_data_offset..];
        const hidden_ptr: [*]f32 = @ptrCast(@alignCast(self.hidden_buf.cpu_ptr.?));
        dequantRow(embed_raw, token_id, self.config.hidden_dim, self.token_embed.info.type_, hidden_ptr[0..self.config.hidden_dim]);
    }

    /// Get the DMMV pipeline, push constant buffer index, rows-per-workgroup, and block size.
    /// Q4_K: native Metal kernel — 32 threads (1 simdgroup) per row, 8 rows per threadgroup (256 threads).
    /// Q4_K wide: specialized large-M kernel — 16 rows per threadgroup (512 threads).
    /// Q4_K LM head 1024: dedicated vocab projection kernel — 32 rows per threadgroup (1024 threads).
    /// Reusing the wider shape outside the LM head improves staged-vector reuse on
    /// the large decode-side Q4_K projections that still dominate token time.
    /// Q5_K/Q6_K/F32: SPIRV-Cross — each thread handles 1 row (64 rows per workgroup, 64 threads).
    /// Q8_0/F16: SPIRV-Cross — each workgroup handles 2 rows (64 threads cooperate via simd_sum).
    fn dmmvPipelineForType(
        self: *InferenceEngine,
        tensor: *const metal_loader.LoadedTensor,
        M: u32,
        K: u32,
    ) ?struct { pipe: *const MetalPipeline, push_idx: u32, rows_per_wg: u32, block_size: u32 } {
        return switch (tensor.info.type_) {
            .q4_k => blk: {
                const k2048_or_less = K <= 2048;
                if (k2048_or_less and
                    tensor == self.lm_head and
                    M >= 65536 and
                    self.dmmv_q4k_lmhead_1024_pipe.max_threads_per_threadgroup >= 1024)
                {
                    break :blk .{ .pipe = &self.dmmv_q4k_lmhead_1024_pipe, .push_idx = 1, .rows_per_wg = 32, .block_size = 1024 };
                }
                if (k2048_or_less and
                    self.dmmv_q4k_lmhead_pipe.max_threads_per_threadgroup >= 512 and
                    ((tensor == self.lm_head and M >= 65536) or M >= 1024))
                {
                    break :blk .{ .pipe = &self.dmmv_q4k_lmhead_pipe, .push_idx = 1, .rows_per_wg = 16, .block_size = 512 };
                }
                if (k2048_or_less) {
                    break :blk .{ .pipe = &self.dmmv_q4k_k2048_pipe, .push_idx = 1, .rows_per_wg = 8, .block_size = 256 };
                }
                break :blk .{ .pipe = &self.dmmv_q4k_pipe, .push_idx = 1, .rows_per_wg = 8, .block_size = 256 };
            },
            .q5_k => .{ .pipe = &self.dmmv_q5k_pipe, .push_idx = 0, .rows_per_wg = 64, .block_size = 64 },
            .q6_k => .{ .pipe = &self.dmmv_q6k_pipe, .push_idx = 0, .rows_per_wg = 64, .block_size = 64 },
            .q8_0 => .{ .pipe = &self.dmmv_q8_0_pipe, .push_idx = 0, .rows_per_wg = 2, .block_size = 64 },
            .f16 => .{ .pipe = &self.dmmv_f16_pipe, .push_idx = 0, .rows_per_wg = 2, .block_size = 64 },
            .f32 => .{ .pipe = &self.dmmv_f32_pipe, .push_idx = 0, .rows_per_wg = 64, .block_size = 64 },
            else => null,
        };
    }
};

// ---------------------------------------------------------------------------
// Shader loading
// ---------------------------------------------------------------------------

fn loadShaderPipeline(ctx: ?*shim.MetalCtx, name: []const u8) !MetalPipeline {
    var path_buf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "src/shaders/metal/{s}.metal", .{name}) catch return error.PathTooLong;
    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        log.err("Failed to open shader '{s}': {s}", .{ name, @errorName(err) });
        return error.ShaderNotFound;
    };
    defer file.close();
    const stat = try file.stat();
    if (stat.size > 1024 * 1024) return error.ShaderTooLarge;
    var source_buf: [1024 * 1024]u8 = undefined;
    const bytes_read = try file.readAll(&source_buf);
    source_buf[bytes_read] = 0;
    var fn_buf: [128]u8 = undefined;
    const fn_name = std.fmt.bufPrintZ(&fn_buf, "main0", .{}) catch return error.NameTooLong;
    return metal_pipeline.createPipeline(ctx, @ptrCast(&source_buf), fn_name);
}

// ---------------------------------------------------------------------------
// Tensor lookup helpers
// ---------------------------------------------------------------------------

fn findTensorByName(model: *const metal_loader.Model, name: []const u8) ?*const metal_loader.LoadedTensor {
    for (model.tensors.items) |*t| {
        if (std.mem.eql(u8, t.info.name, name)) return t;
    }
    return null;
}

fn findLayerTensor(model: *const metal_loader.Model, layer: u32, suffix: []const u8) ?*const metal_loader.LoadedTensor {
    var name_buf: [128]u8 = undefined;
    const name = std.fmt.bufPrint(&name_buf, "blk.{d}.{s}", .{ layer, suffix }) catch return null;
    return findTensorByName(model, name);
}

fn tensorPageOffset(model: *const metal_loader.Model, tensor: *const metal_loader.LoadedTensor) u32 {
    const data_offset: u64 = model.gguf_file.tensor_data_offset + tensor.info.offset;
    const aligned_offset = (data_offset / 4096) * 4096;
    return @intCast(data_offset - aligned_offset);
}

// ---------------------------------------------------------------------------
// DMMV dispatch helpers
// ---------------------------------------------------------------------------

/// Dispatch a DMMV on an existing command buffer (does NOT commit).
fn dispatchDmmvOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    tensor: *const metal_loader.LoadedTensor,
    input_buf: *const MetalBuffer,
    output_buf: *const MetalBuffer,
    M: u32,
    K: u32,
    extra_byte_offset: u32,
) void {
    const pip = engine.dmmvPipelineForType(tensor, M, K) orelse {
        log.err("No DMMV pipeline for quant type {d} (tensor {s})", .{ @intFromEnum(tensor.info.type_), tensor.info.name });
        return;
    };
    const page_off = tensorPageOffset(engine.model, tensor);
    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = page_off + extra_byte_offset,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &tensor.gpu_buffer, input_buf, output_buf };
    const wgs = (M + pip.rows_per_wg - 1) / pip.rows_per_wg;
    cmd.dispatchV2(pip.pipe, .{ wgs, 1, 1 }, .{ pip.block_size, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), pip.push_idx);
}

/// Dispatch a single DMMV and wait for completion.
fn dispatchDmmvAndWait(
    engine: *InferenceEngine,
    tensor: *const metal_loader.LoadedTensor,
    input_buf: *const MetalBuffer,
    output_buf: *const MetalBuffer,
    M: u32,
    K: u32,
    extra_byte_offset: u32,
) !void {
    var cmd = try metal_command.beginCommand(engine.device.ctx);
    dispatchDmmvOnCmd(engine, &cmd, tensor, input_buf, output_buf, M, K, extra_byte_offset);
    cmd.commitAndWait();
}

/// Dispatch a batched Q4_K MoE DMMV on an existing command buffer.
/// grid.y selects the expert slot; expert IDs come from routing_buf.
fn dispatchDmmvMoeQ4kOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    tensor: *const metal_loader.LoadedTensor,
    input_buf: *const MetalBuffer,
    output_buf: *const MetalBuffer,
    routing_buf: *const MetalBuffer,
    M: u32,
    K: u32,
    expert_stride: u32,
    x_expert_stride: u32,
    extra_byte_offset: u32,
) void {
    if (tensor.info.type_ != .q4_k) {
        log.err("Batched MoE DMMV only supports Q4_K (tensor {s})", .{tensor.info.name});
        return;
    }

    const push = MoeDmmvPush{
        .M = M,
        .K = K,
        .a_offset = tensorPageOffset(engine.model, tensor) + extra_byte_offset,
        .expert_stride = expert_stride,
        .x_expert_stride = x_expert_stride,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &tensor.gpu_buffer, input_buf, output_buf, routing_buf };
    const k2048_or_less = K <= 2048;
    const use_1024_k2048 =
        k2048_or_less and
        x_expert_stride != 0 and
        M >= 1024 and
        engine.dmmv_q4k_moe_k2048_1024_pipe.max_threads_per_threadgroup >= 1024;
    const rows_per_wg: u32 = if (use_1024_k2048) 32 else if (k2048_or_less) 16 else 8;
    const block_size: u32 = if (use_1024_k2048) 1024 else if (k2048_or_less) 512 else 256;
    const wgs = (M + rows_per_wg - 1) / rows_per_wg;
    const pipe = if (use_1024_k2048)
        &engine.dmmv_q4k_moe_k2048_1024_pipe
    else if (k2048_or_less)
        &engine.dmmv_q4k_moe_k2048_pipe
    else
        &engine.dmmv_q4k_moe_pipe;
    cmd.dispatchV2(pipe, .{ wgs, engine.config.n_experts_used, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(MoeDmmvPush), 1);
}

/// Preload norm weights from mmap into an f32 Metal buffer (done once at init).
fn preloadNormWeights(
    ctx: ?*shim.MetalCtx,
    model: *const metal_loader.Model,
    tensor: *const metal_loader.LoadedTensor,
    n: u32,
) !MetalBuffer {
    const mmap = model.mmap_data orelse return error.NoMmapData;
    const buf = try metal_buffer.createBuffer(ctx, @as(usize, n) * @sizeOf(f32));
    const dst: [*]f32 = @ptrCast(@alignCast(buf.cpu_ptr.?));
    const off: usize = @intCast(model.gguf_file.tensor_data_offset + tensor.info.offset);
    readMmapFloats(mmap, off, tensor.info.type_, dst[0..n]);
    return buf;
}

/// Dispatch GPU RMS norm on an existing command buffer (does NOT commit).
/// rms_norm_mul.metal: buffer(0)=push, buffer(1)=input, buffer(2)=output, buffer(3)=weights.
/// Block size 64 (hardcoded in shader). Grid = (n_groups, 1, 1).
fn dispatchRmsNormOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    input: *const MetalBuffer,
    output: *const MetalBuffer,
    weights: *const MetalBuffer,
    n: u32,
    n_groups: u32,
) void {
    const push = RmsNormPush{ .n = n, .eps = 1e-6 };
    const bufs = [_]*const MetalBuffer{ input, output, weights };
    cmd.dispatchV2(&engine.rms_norm_pipe, .{ n_groups, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(RmsNormPush), 0);
}

fn dispatchDeinterleaveOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    input: *const MetalBuffer,
    q_output: *const MetalBuffer,
    gate_output: *const MetalBuffer,
    head_dim: u32,
    n_heads: u32,
) void {
    const push = DeinterleavePush{ .head_dim = head_dim, .n_heads = n_heads };
    const bufs = [_]*const MetalBuffer{ q_output, input, gate_output };
    const total = head_dim * n_heads;
    cmd.dispatchV2(&engine.deinterleave_pipe, .{ (total + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DeinterleavePush), 0);
}

fn dispatchFlashAttnOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    layer_idx: usize,
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_len: u32,
) void {
    const push = FlashAttnPush{
        .head_dim = head_dim,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .seq_len = seq_len,
        .page_size = 1,
    };
    const bufs = [_]*const MetalBuffer{
        &engine.page_table_buf,
        &engine.q_buf,
        &engine.kv_k_cache[layer_idx],
        &engine.kv_v_cache[layer_idx],
        &engine.attn_out_buf,
    };
    cmd.dispatchV2(&engine.flash_attn_pipe, .{ n_heads, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(FlashAttnPush), 0);
}

fn dispatchKvCacheWriteOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    layer_idx: usize,
    kv_dim: u32,
    dst_offset: u32,
) void {
    const push = KvCacheWritePush{
        .n = kv_dim,
        .dst_offset = dst_offset,
    };
    const bufs = [_]*const MetalBuffer{
        &engine.k_buf,
        &engine.v_buf,
        &engine.kv_k_cache[layer_idx],
        &engine.kv_v_cache[layer_idx],
    };
    cmd.dispatchV2(&engine.kv_cache_write_pipe, .{ (kv_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(KvCacheWritePush), 0);
}

fn dispatchRopeOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    input: *const MetalBuffer,
    output: *const MetalBuffer,
    stride: u32,
    rope_dim: u32,
    n_heads: u32,
    position: u32,
    freq_base: f32,
) void {
    const push = RopePush{
        .stride = stride,
        .rope_dim = rope_dim,
        .n_heads = n_heads,
        .position = position,
        .freq_base_bits = @bitCast(freq_base),
    };
    const bufs = [_]*const MetalBuffer{ input, output };
    cmd.dispatchV2(&engine.rope_pipe, .{ n_heads, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(RopePush), 0);
}

fn dispatchSigmoidMulOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    gate: *const MetalBuffer,
    data: *const MetalBuffer,
    n: u32,
) void {
    const push = SigmoidMulPush{ .n = n };
    const bufs = [_]*const MetalBuffer{ gate, data, data };
    cmd.dispatchV2(&engine.sigmoid_mul_pipe, .{ (n + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SigmoidMulPush), 0);
}

fn dispatchSoftmaxTopkOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    logits: *const MetalBuffer,
    output: *const MetalBuffer,
    n_experts: u32,
    k: u32,
) void {
    const push = SoftmaxTopkPush{ .n_experts = n_experts, .k = k };
    const bufs = [_]*const MetalBuffer{ logits, output };
    cmd.dispatchV2(&engine.softmax_topk_pipe, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SoftmaxTopkPush), 0);
}

fn dispatchSigmoidScaleAccOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    accum: *const MetalBuffer,
    src: *const MetalBuffer,
    gate: *const MetalBuffer,
    n: u32,
) void {
    const push = ScaleAccPush{ .n = n, .scale_bits = 0 };
    const bufs = [_]*const MetalBuffer{ accum, src, gate };
    cmd.dispatchV2(&engine.sigmoid_scale_acc_pipe, .{ (n + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(ScaleAccPush), 3);
}

fn dispatchMoeWeightedAccOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    accum: *const MetalBuffer,
    src: *const MetalBuffer,
    routing: *const MetalBuffer,
    n: u32,
    n_used: u32,
    src_stride: u32,
) void {
    const push = MoeWeightedAccPush{
        .n = n,
        .n_used = n_used,
        .src_stride = src_stride,
    };
    const bufs = [_]*const MetalBuffer{ accum, src, routing };
    cmd.dispatchV2(&engine.moe_weighted_acc_pipe, .{ (n + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(MoeWeightedAccPush), 3);
}

fn dispatchFullAttnPrepOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    layer_idx: usize,
    lt: LayerTensors,
    q_dim: u32,
    kv_dim: u32,
    hidden_dim: u32,
) !void {
    const cfg = engine.config;
    const q_tensor = lt.attn_q orelse return error.MissingTensor;
    const k_tensor = lt.attn_k orelse return error.MissingTensor;
    const v_tensor = lt.attn_v orelse return error.MissingTensor;
    const q_full_dim = q_dim * 2;
    const rope_dim: u32 = if (cfg.rope_dim > 0) cfg.rope_dim else cfg.head_dim;

    dispatchDmmvOnCmd(engine, cmd, q_tensor, &engine.norm_buf, &engine.attn_out_buf, q_full_dim, hidden_dim, 0);
    dispatchDmmvOnCmd(engine, cmd, k_tensor, &engine.norm_buf, &engine.k_buf, kv_dim, hidden_dim, 0);
    dispatchDmmvOnCmd(engine, cmd, v_tensor, &engine.norm_buf, &engine.v_buf, kv_dim, hidden_dim, 0);
    cmd.barrier();

    dispatchDeinterleaveOnCmd(engine, cmd, &engine.attn_out_buf, &engine.q_buf, &engine.gate_buf, cfg.head_dim, cfg.n_heads);
    cmd.barrier();

    if (engine.attn_q_norm_present[layer_idx]) {
        dispatchRmsNormOnCmd(engine, cmd, &engine.q_buf, &engine.q_buf, &engine.attn_q_norm_bufs[layer_idx], cfg.head_dim, cfg.n_heads);
    }
    if (engine.attn_k_norm_present[layer_idx]) {
        dispatchRmsNormOnCmd(engine, cmd, &engine.k_buf, &engine.k_buf, &engine.attn_k_norm_bufs[layer_idx], cfg.head_dim, cfg.n_kv_heads);
    }
    cmd.barrier();

    dispatchRopeOnCmd(engine, cmd, &engine.q_buf, &engine.q_buf, cfg.head_dim, rope_dim, cfg.n_heads, engine.position, cfg.rope_freq_base);
    dispatchRopeOnCmd(engine, cmd, &engine.k_buf, &engine.k_buf, cfg.head_dim, rope_dim, cfg.n_kv_heads, engine.position, cfg.rope_freq_base);
    cmd.barrier();

    dispatchKvCacheWriteOnCmd(engine, cmd, layer_idx, kv_dim, engine.position * kv_dim);
    cmd.barrier();
}

/// CPU fallback DMMV for K > shared memory limit (4096).
fn cpuDmmvFallback(
    mmap: []const u8,
    tensor: *const metal_loader.LoadedTensor,
    data_offset: u64,
    input: [*]const f32,
    output: [*]f32,
    M: u32,
    K: u32,
    extra_byte_offset: u32,
    allocator: std.mem.Allocator,
) !void {
    const off: usize = @intCast(data_offset + tensor.info.offset + extra_byte_offset);
    const raw = mmap[off..];
    const row_buf = try allocator.alloc(f32, K);
    defer allocator.free(row_buf);
    for (0..M) |row| {
        dequantRow(raw, @intCast(row), K, tensor.info.type_, row_buf);
        var dot: f32 = 0;
        for (0..K) |d| dot += row_buf[d] * input[d];
        output[row] = dot;
    }
}

/// Smart DMMV: GPU for K <= 4096, CPU fallback for larger K.
fn smartDmmv(
    engine: *InferenceEngine,
    tensor: *const metal_loader.LoadedTensor,
    input_buf: *const MetalBuffer,
    output_buf: *const MetalBuffer,
    M: u32,
    K: u32,
    extra_byte_offset: u32,
) !void {
    if (K <= 4096) {
        try dispatchDmmvAndWait(engine, tensor, input_buf, output_buf, M, K, extra_byte_offset);
    } else {
        const input_ptr: [*]const f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
        const output_ptr: [*]f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
        try cpuDmmvFallback(
            engine.model.mmap_data orelse return error.NoMmapData,
            tensor,
            engine.model.gguf_file.tensor_data_offset,
            input_ptr,
            output_ptr,
            M,
            K,
            extra_byte_offset,
            engine.allocator,
        );
    }
}

// ---------------------------------------------------------------------------
// CPU math helpers (all via UMA shared buffers)
// ---------------------------------------------------------------------------

/// RMS norm with weight multiply: output[i] = weight[i] * (input[i] / rms)
fn cpuRmsNormMul(input: [*]const f32, weight: []const f32, output: [*]f32, n: u32, n_groups: u32, eps: f32) void {
    for (0..n_groups) |g| {
        const off = g * n;
        var sq: f32 = 0;
        for (0..n) |i| sq += input[off + i] * input[off + i];
        const rms_inv = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(n)) + eps);
        for (0..n) |i| output[off + i] = weight[i % weight.len] * (input[off + i] * rms_inv);
    }
}

/// Per-head RMS norm with weight multiply (in-place).
fn cpuPerHeadRmsNormMul(data: [*]f32, weight: []const f32, head_dim: u32, n_heads: u32, eps: f32) void {
    for (0..n_heads) |h| {
        const off = h * head_dim;
        var sq: f32 = 0;
        for (0..head_dim) |i| sq += data[off + i] * data[off + i];
        const rms_inv = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(head_dim)) + eps);
        for (0..head_dim) |i| data[off + i] = data[off + i] * rms_inv * weight[i];
    }
}

/// RoPE: apply rotary position embedding.
fn cpuRope(data: [*]f32, stride: u32, rope_dim: u32, n_heads: u32, position: u32, freq_base: f32) void {
    const half_rot = rope_dim / 2;
    for (0..n_heads) |h| {
        const base_idx = @as(u32, @intCast(h)) * stride;
        for (0..half_rot) |i| {
            const exponent = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(rope_dim));
            const freq_i = 1.0 / std.math.pow(f32, freq_base, exponent);
            const theta = @as(f32, @floatFromInt(position)) * freq_i;
            const cos_t = @cos(theta);
            const sin_t = @sin(theta);
            const idx0 = base_idx + @as(u32, @intCast(i));
            const idx1 = idx0 + half_rot;
            const x0 = data[idx0];
            const x1 = data[idx1];
            data[idx0] = x0 * cos_t - x1 * sin_t;
            data[idx1] = x0 * sin_t + x1 * cos_t;
        }
        // Copy remaining dimensions unchanged (IMRoPE: only first rope_dim dims rotated)
    }
}

/// SwiGLU: output[i] = SiLU(gate[i]) * up[i]
fn cpuSwiGLU(gate: [*]const f32, up: [*]const f32, output: [*]f32, n: u32) void {
    for (0..n) |i| {
        const x = gate[i];
        output[i] = (x / (1.0 + @exp(-x))) * up[i];
    }
}

/// CPU single-token attention (no paging, contiguous KV cache).
fn cpuAttention(
    q: [*]const f32,
    k_cache: [*]const f32,
    v_cache: [*]const f32,
    output: [*]f32,
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_len: u32,
) void {
    const gqa_ratio = n_heads / @max(n_kv_heads, 1);
    const kv_dim = n_kv_heads * head_dim;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    for (0..n_heads) |h| {
        const kv_h = @as(u32, @intCast(h)) / gqa_ratio;
        const q_head = q + @as(u32, @intCast(h)) * head_dim;

        var scores: [4096]f32 = undefined;
        var max_score: f32 = -std.math.inf(f32);
        for (0..seq_len) |t| {
            const k_vec = k_cache + @as(usize, t) * kv_dim + @as(usize, kv_h) * head_dim;
            var dot: f32 = 0;
            for (0..head_dim) |d| dot += q_head[d] * k_vec[d];
            scores[t] = dot * scale;
            if (scores[t] > max_score) max_score = scores[t];
        }

        var sum: f32 = 0;
        for (0..seq_len) |t| {
            scores[t] = @exp(scores[t] - max_score);
            sum += scores[t];
        }
        if (sum > 0) for (0..seq_len) |t| {
            scores[t] /= sum;
        };

        const out_head = output + @as(u32, @intCast(h)) * head_dim;
        for (0..head_dim) |d| {
            var val: f32 = 0;
            for (0..seq_len) |t| {
                val += scores[t] * v_cache[@as(usize, t) * kv_dim + @as(usize, kv_h) * head_dim + d];
            }
            out_head[d] = val;
        }
    }
}

// ---------------------------------------------------------------------------
// CPU-side dequantization helpers
// ---------------------------------------------------------------------------

fn getScaleMinK4(j: usize, scales: []const u8) struct { sc: u8, m: u8 } {
    if (j < 4) {
        return .{ .sc = scales[j] & 63, .m = scales[j + 4] & 63 };
    } else {
        return .{
            .sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            .m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        };
    }
}

pub fn dequantRow(raw_data: []const u8, row: u32, cols: u32, quant_type: GGMLType, output: []f32) void {
    switch (quant_type) {
        .f32 => {
            const row_bytes = @as(usize, cols) * 4;
            const offset = @as(usize, row) * row_bytes;
            const src: [*]const f32 = @ptrCast(@alignCast(raw_data[offset..].ptr));
            @memcpy(output, src[0..cols]);
        },
        .f16 => {
            const offset = @as(usize, row) * @as(usize, cols) * 2;
            for (0..cols) |i| {
                const byte_off = offset + i * 2;
                const bits = std.mem.readInt(u16, raw_data[byte_off..][0..2], .little);
                output[i] = @floatCast(@as(f16, @bitCast(bits)));
            }
        },
        .q8_0 => {
            const block_size: usize = 32;
            const bpb: usize = 34;
            const bpr = @as(usize, cols) / block_size;
            const row_off = @as(usize, row) * bpr * bpb;
            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bo = row_off + b * bpb;
                const scale_bits = std.mem.readInt(u16, raw_data[bo..][0..2], .little);
                const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
                for (0..block_size) |j| {
                    const v: i8 = @bitCast(raw_data[bo + 2 + j]);
                    output[out_i] = @as(f32, @floatFromInt(v)) * scale;
                    out_i += 1;
                }
            }
        },
        .q4_k => {
            const bpb: usize = 144;
            const bpr = @as(usize, cols) / 256;
            const row_off = @as(usize, row) * bpr * bpb;
            var out_i: usize = 0;
            for (0..bpr) |bi| {
                const bb = row_off + bi * bpb;
                const d_bits = std.mem.readInt(u16, raw_data[bb..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
                const dm_bits = std.mem.readInt(u16, raw_data[bb + 2 ..][0..2], .little);
                const dmin: f32 = @floatCast(@as(f16, @bitCast(dm_bits)));
                const scales = raw_data[bb + 4 .. bb + 16];
                const qs = raw_data[bb + 16 .. bb + 144];
                var is: usize = 0;
                var qo: usize = 0;
                for (0..4) |_| {
                    const sm0 = getScaleMinK4(is, scales);
                    const d1 = d * @as(f32, @floatFromInt(sm0.sc));
                    const m1 = dmin * @as(f32, @floatFromInt(sm0.m));
                    const sm1 = getScaleMinK4(is + 1, scales);
                    const d2 = d * @as(f32, @floatFromInt(sm1.sc));
                    const m2 = dmin * @as(f32, @floatFromInt(sm1.m));
                    for (0..32) |l| {
                        output[out_i] = d1 * @as(f32, @floatFromInt(qs[qo + l] & 0xF)) - m1;
                        out_i += 1;
                    }
                    for (0..32) |l| {
                        output[out_i] = d2 * @as(f32, @floatFromInt(qs[qo + l] >> 4)) - m2;
                        out_i += 1;
                    }
                    qo += 32;
                    is += 2;
                }
            }
        },
        else => {
            log.warn("Unsupported quant type {d}, using zeros", .{@intFromEnum(quant_type)});
            @memset(output, 0);
        },
    }
}

fn readMmapFloats(mmap: []const u8, base_off: usize, tensor_type: GGMLType, output: []f32) void {
    switch (tensor_type) {
        .f32 => {
            const src: [*]const f32 = @ptrCast(@alignCast(mmap[base_off..].ptr));
            @memcpy(output, src[0..output.len]);
        },
        .f16 => {
            for (0..output.len) |i| {
                const off = base_off + i * 2;
                const bits = std.mem.readInt(u16, mmap[off..][0..2], .little);
                output[i] = @floatCast(@as(f16, @bitCast(bits)));
            }
        },
        else => {
            log.warn("readMmapFloats: unsupported type {s}, zeroing", .{@tagName(tensor_type)});
            @memset(output, 0);
        },
    }
}

pub fn topKSoftmax(logits: []const f32, k: u32, out_ids: []u32, out_weights: []f32) void {
    const n = logits.len;
    var max_val: f32 = -std.math.inf(f32);
    for (logits) |v| if (v > max_val) {
        max_val = v;
    };
    var probs: [256]f32 = undefined;
    var sum: f32 = 0;
    for (0..n) |i| {
        probs[i] = @exp(logits[i] - max_val);
        sum += probs[i];
    }
    if (sum > 0) for (0..n) |i| {
        probs[i] /= sum;
    };
    var used = [_]bool{false} ** 256;
    for (0..k) |ki| {
        var best_idx: u32 = 0;
        var best_val: f32 = -1.0;
        for (0..n) |i| {
            if (!used[i] and probs[i] > best_val) {
                best_val = probs[i];
                best_idx = @intCast(i);
            }
        }
        out_ids[ki] = best_idx;
        out_weights[ki] = best_val;
        used[best_idx] = true;
    }
    var wsum: f32 = 0;
    for (0..k) |i| wsum += out_weights[i];
    if (wsum > 0) for (0..k) |i| {
        out_weights[i] /= wsum;
    };
}

fn expertSliceBytes(quant_type: GGMLType, rows: u32, cols: u32) u32 {
    const bs = quant_type.blockSize();
    const bpb = quant_type.bytesPerBlock();
    if (bs == 0 or bpb == 0) return rows * cols * 4;
    const blocks_per_row = cols / bs;
    return rows * blocks_per_row * bpb;
}

fn canUseBatchedQ4kMoe(engine: *const InferenceEngine, gate_quant: GGMLType, down_quant: GGMLType) bool {
    return engine.config.n_experts_used == 8 and gate_quant == .q4_k and down_quant == .q4_k;
}

fn canUseGpuRoutedBatchedMoe(engine: *const InferenceEngine, lt: LayerTensors) bool {
    const gate_exps = lt.ffn_gate_exps orelse return false;
    const up_exps = lt.ffn_up_exps orelse return false;
    const down_exps = lt.ffn_down_exps orelse return false;

    if (!canUseBatchedQ4kMoe(engine, gate_exps.info.type_, down_exps.info.type_)) return false;
    if (up_exps.info.type_ != .q4_k) return false;

    const has_shexp = lt.ffn_gate_shexp != null and lt.ffn_up_shexp != null and lt.ffn_down_shexp != null;
    if (has_shexp and lt.ffn_gate_inp_shexp != null and engine.sigmoid_scale_acc_pipe.handle == null) return false;

    return engine.softmax_topk_pipe.handle != null and engine.moe_weighted_acc_pipe.handle != null;
}

fn recordGpuRoutedBatchedMoeOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    lt: LayerTensors,
    hidden_dim: u32,
    inter_dim: u32,
    shexp_inter_dim: u32,
) !void {
    const cfg = engine.config;
    const gate_exps = lt.ffn_gate_exps orelse return error.MissingTensor;
    const up_exps = lt.ffn_up_exps orelse return error.MissingTensor;
    const down_exps = lt.ffn_down_exps orelse return error.MissingTensor;
    const gate_shexp = lt.ffn_gate_shexp;
    const up_shexp = lt.ffn_up_shexp;
    const down_shexp = lt.ffn_down_shexp;
    const gate_inp_shexp = lt.ffn_gate_inp_shexp;
    const has_shexp = gate_shexp != null and up_shexp != null and down_shexp != null;
    const expert_gate_bytes = expertSliceBytes(gate_exps.info.type_, inter_dim, hidden_dim);
    const expert_down_bytes = expertSliceBytes(down_exps.info.type_, hidden_dim, inter_dim);

    // The GPU-routed MoE fast path is a straight producer->consumer chain in a
    // single compute encoder. Rely on Metal's in-order dispatch execution here
    // instead of forcing a full buffer-scope barrier between every phase.
    dispatchSoftmaxTopkOnCmd(engine, cmd, &engine.router_logits_buf, &engine.router_output_buf, cfg.n_experts, cfg.n_experts_used);

    dispatchDmmvMoeQ4kOnCmd(engine, cmd, gate_exps, &engine.norm_buf, &engine.expert_gate_batch_buf, &engine.router_output_buf, inter_dim, hidden_dim, expert_gate_bytes, 0, 0);
    dispatchDmmvMoeQ4kOnCmd(engine, cmd, up_exps, &engine.norm_buf, &engine.expert_up_batch_buf, &engine.router_output_buf, inter_dim, hidden_dim, expert_gate_bytes, 0, 0);
    if (has_shexp) {
        dispatchDmmvOnCmd(engine, cmd, gate_shexp.?, &engine.norm_buf, &engine.gate_buf, shexp_inter_dim, hidden_dim, 0);
        dispatchDmmvOnCmd(engine, cmd, up_shexp.?, &engine.norm_buf, &engine.up_buf, shexp_inter_dim, hidden_dim, 0);
        if (gate_inp_shexp) |tensor| {
            dispatchDmmvOnCmd(engine, cmd, tensor, &engine.norm_buf, &engine.router_logits_buf, 1, hidden_dim, 0);
        }
    }

    {
        const swiglu_push = SwiGLUPush{ .n = inter_dim };
        const sw_bufs = [_]*const MetalBuffer{ &engine.expert_gate_batch_buf, &engine.expert_swiglu_batch_buf, &engine.expert_up_batch_buf };
        cmd.dispatchV2(&engine.swiglu_batched_pipe, .{ (inter_dim + 63) / 64, cfg.n_experts_used, 1 }, .{ 64, 1, 1 }, &sw_bufs, &swiglu_push, @sizeOf(SwiGLUPush), 0);
    }
    if (has_shexp) {
        const sw_push = SwiGLUPush{ .n = shexp_inter_dim };
        const sw_bufs = [_]*const MetalBuffer{ &engine.gate_buf, &engine.swiglu_buf, &engine.up_buf };
        cmd.dispatchV2(&engine.swiglu_pipe, .{ (shexp_inter_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &sw_bufs, &sw_push, @sizeOf(SwiGLUPush), 0);
    }

    dispatchDmmvMoeQ4kOnCmd(engine, cmd, down_exps, &engine.expert_swiglu_batch_buf, &engine.expert_down_batch_buf, &engine.router_output_buf, hidden_dim, inter_dim, expert_down_bytes, inter_dim, 0);
    if (has_shexp) {
        dispatchDmmvOnCmd(engine, cmd, down_shexp.?, &engine.swiglu_buf, &engine.down_buf, hidden_dim, shexp_inter_dim, 0);
    }

    // Fold the MoE residual add into the weighted accumulation shader so the
    // fast GPU-routed path does not bounce through an extra hidden-sized buffer.
    dispatchMoeWeightedAccOnCmd(engine, cmd, &engine.hidden_buf, &engine.expert_down_batch_buf, &engine.router_output_buf, hidden_dim, cfg.n_experts_used, hidden_dim);
    if (has_shexp) {
        if (gate_inp_shexp != null) {
            dispatchSigmoidScaleAccOnCmd(engine, cmd, &engine.hidden_buf, &engine.down_buf, &engine.router_logits_buf, hidden_dim);
        } else {
            const shexp_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(@as(f32, 1.0))) };
            const shexp_bufs = [_]*const MetalBuffer{ &engine.hidden_buf, &engine.down_buf };
            cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &shexp_bufs, &shexp_push, @sizeOf(ScaleAccPush), 0);
        }
    }
}

fn acquireLayerCommand(
    engine: *InferenceEngine,
    shared_cmd: ?*MetalCommand,
    local_cmd_storage: *MetalCommand,
    using_local_cmd: *bool,
) !*MetalCommand {
    if (shared_cmd) |cmd| {
        using_local_cmd.* = false;
        return cmd;
    }

    local_cmd_storage.* = try metal_command.beginCommand(engine.device.ctx);
    using_local_cmd.* = true;
    return local_cmd_storage;
}

// ---------------------------------------------------------------------------
// Decode step — runs all layers + final norm + LM head
// ---------------------------------------------------------------------------

fn runDecodeStep(engine: *InferenceEngine) !void {
    const cfg = engine.config;
    const hidden_dim = cfg.hidden_dim;
    const q_dim: u32 = cfg.n_heads * cfg.head_dim;
    const kv_dim: u32 = cfg.n_kv_heads * cfg.head_dim;
    const is_moe = cfg.n_experts > 0;
    const inter_dim: u32 = if (cfg.intermediate_dim > 0) cfg.intermediate_dim else hidden_dim * 4;
    const shexp_inter_dim: u32 = if (cfg.shared_expert_intermediate_dim > 0) cfg.shared_expert_intermediate_dim else inter_dim;
    const full_attn_interval: u32 = if (cfg.full_attn_interval > 0) cfg.full_attn_interval else 1;

    // SSM constants (needed for GPU dispatch sizing)
    const d_inner: u32 = cfg.ssm_d_inner;
    const d_state: u32 = cfg.ssm_d_state;
    const n_group: u32 = cfg.ssm_n_group;
    const dt_rank: u32 = cfg.ssm_dt_rank;
    const conv_channels: u32 = if (d_inner > 0) d_inner + 2 * n_group * d_state else 0;

    const head_v_dim: u32 = if (d_inner > 0) d_inner / @max(dt_rank, 1) else 0;
    const d_conv: u32 = cfg.ssm_d_conv;
    const use_single_gpu_cmd = is_moe and blk: {
        for (engine.layer_tensors) |lt| {
            if (!canUseGpuRoutedBatchedMoe(engine, lt)) break :blk false;
        }
        break :blk true;
    };
    var shared_cmd_storage: MetalCommand = undefined;
    const shared_cmd: ?*MetalCommand = if (use_single_gpu_cmd) blk: {
        shared_cmd_storage = try metal_command.beginCommand(engine.device.ctx);
        break :blk &shared_cmd_storage;
    } else null;

    for (0..cfg.n_layers) |layer_idx| {
        const layer: u32 = @intCast(layer_idx);
        const lt = engine.layer_tensors[layer_idx];
        const is_full_attn = ((layer + 1) % full_attn_interval == 0);
        const use_gpu_routed_moe = is_moe and canUseGpuRoutedBatchedMoe(engine, lt);

        if (is_full_attn) {
            // Keep full attention on-GPU end-to-end so decode does not stall on
            // a CPU-side KV-cache copy between the prep and flash-attention steps.
            var local_cmd_storage: MetalCommand = undefined;
            var using_local_cmd = false;
            const cmd = try acquireLayerCommand(engine, shared_cmd, &local_cmd_storage, &using_local_cmd);
            dispatchRmsNormOnCmd(engine, cmd, &engine.hidden_buf, &engine.norm_buf, &engine.attn_norm_bufs[layer_idx], hidden_dim, 1);
            cmd.barrier();
            try dispatchFullAttnPrepOnCmd(engine, cmd, layer_idx, lt, q_dim, kv_dim, hidden_dim);
            dispatchFlashAttnOnCmd(engine, cmd, layer_idx, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, engine.position + 1);
            cmd.barrier();
            dispatchSigmoidMulOnCmd(engine, cmd, &engine.gate_buf, &engine.attn_out_buf, q_dim);
            cmd.barrier();
            const o_tensor = lt.attn_output orelse return error.MissingTensor;
            dispatchDmmvOnCmd(engine, cmd, o_tensor, &engine.attn_out_buf, &engine.down_buf, hidden_dim, q_dim, 0);
            cmd.barrier();
            {
                const res_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(@as(f32, 1.0))) };
                const res_bufs = [_]*const MetalBuffer{ &engine.hidden_buf, &engine.down_buf };
                cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &res_bufs, &res_push, @sizeOf(ScaleAccPush), 0);
            }
            cmd.barrier();
            dispatchRmsNormOnCmd(engine, cmd, &engine.hidden_buf, &engine.norm_buf, &engine.ffn_norm_bufs[layer_idx], hidden_dim, 1);
            cmd.barrier();
            if (is_moe) {
                const router_t = lt.ffn_gate_inp orelse return error.MissingTensor;
                dispatchDmmvOnCmd(engine, cmd, router_t, &engine.norm_buf, &engine.router_logits_buf, cfg.n_experts, hidden_dim, 0);
                if (use_gpu_routed_moe) {
                    try recordGpuRoutedBatchedMoeOnCmd(engine, cmd, lt, hidden_dim, inter_dim, shexp_inter_dim);
                    if (shared_cmd != null) cmd.barrier();
                }
            }
            if (using_local_cmd) cmd.commitAndWait();
        } else {
            // ===== SSM: fused batch 1 + recurrent body + batch 2 =====
            // There is no CPU dependency between the SSM projections and the
            // recurrent kernels, so keep the whole path in one command buffer.
            var local_cmd_storage: MetalCommand = undefined;
            var using_local_cmd = false;
            const cmd = try acquireLayerCommand(engine, shared_cmd, &local_cmd_storage, &using_local_cmd);
            dispatchRmsNormOnCmd(engine, cmd, &engine.hidden_buf, &engine.norm_buf, &engine.attn_norm_bufs[layer_idx], hidden_dim, 1);
            cmd.barrier();

            const wqkv_t = lt.attn_qkv orelse return error.MissingTensor;
            const z_t = lt.attn_gate orelse return error.MissingTensor;
            const alpha_t = lt.ssm_alpha orelse return error.MissingTensor;
            const beta_t = lt.ssm_beta orelse return error.MissingTensor;
            dispatchDmmvOnCmd(engine, cmd, wqkv_t, &engine.norm_buf, &engine.attn_out_buf, conv_channels, hidden_dim, 0);
            dispatchDmmvOnCmd(engine, cmd, z_t, &engine.norm_buf, &engine.gate_buf, d_inner, hidden_dim, 0);
            dispatchDmmvOnCmd(engine, cmd, alpha_t, &engine.norm_buf, &engine.router_logits_buf, dt_rank, hidden_dim, 0);
            dispatchDmmvOnCmd(engine, cmd, beta_t, &engine.norm_buf, &engine.down_buf, dt_rank, hidden_dim, 0);
            cmd.barrier();

            // Conv1d: attn_out_buf → swiglu_buf
            {
                const push = SsmConv1dPush{ .conv_channels = conv_channels, .d_conv = d_conv, .kernel_is_f16 = 0 };
                const c1_bufs = [_]*const MetalBuffer{ &engine.ssm_conv_kernel_bufs.?[layer_idx], &engine.ssm_conv_state_bufs.?[layer_idx], &engine.attn_out_buf, &engine.swiglu_buf };
                cmd.dispatchV2(&engine.ssm_conv1d_pipe, .{ (conv_channels + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &c1_bufs, &push, @sizeOf(SsmConv1dPush), 0);
            }
            cmd.barrier();

            // Delta-net: swiglu_buf → attn_out_buf
            {
                const push = SsmDeltaNetPush{
                    .d_inner = d_inner,
                    .dt_rank = dt_rank,
                    .head_v_dim = head_v_dim,
                    .d_state = d_state,
                    .n_group = n_group,
                    .ssm_a_is_f16 = 0,
                    .dt_bias_is_f16 = 0,
                    .has_dt_bias = if (lt.ssm_dt_bias != null) @as(u32, 1) else 0,
                    .has_ssm_a = if (lt.ssm_a != null) @as(u32, 1) else 0,
                };
                const dn_bufs = [_]*const MetalBuffer{
                    &engine.swiglu_buf,                    &engine.router_logits_buf,
                    &engine.ssm_dt_bias_bufs.?[layer_idx], &engine.ssm_a_bufs.?[layer_idx],
                    &engine.down_buf,                      &engine.ssm_state_bufs.?[layer_idx],
                    &engine.attn_out_buf,
                };
                cmd.dispatchV2(&engine.ssm_delta_net_pipe, .{ dt_rank, 1, 1 }, .{ 64, 1, 1 }, &dn_bufs, &push, @sizeOf(SsmDeltaNetPush), 0);
            }
            cmd.barrier();

            // Gated norm: attn_out_buf → swiglu_buf
            {
                const push = SsmGatedNormPush{
                    .d_inner = d_inner,
                    .dt_rank = dt_rank,
                    .head_v_dim = head_v_dim,
                    .d_state = d_state,
                    .norm_per_head = if (engine.ssm_norm_per_head.?[layer_idx]) @as(u32, 1) else 0,
                };
                const gn_bufs = [_]*const MetalBuffer{
                    &engine.attn_out_buf, &engine.ssm_norm_weight_bufs.?[layer_idx],
                    &engine.gate_buf,     &engine.swiglu_buf,
                };
                cmd.dispatchV2(&engine.ssm_gated_norm_pipe, .{ dt_rank, 1, 1 }, .{ 64, 1, 1 }, &gn_bufs, &push, @sizeOf(SsmGatedNormPush), 0);
            }
            cmd.barrier();

            // SSM out DMMV: swiglu_buf → down_buf
            const ssm_out_t = lt.ssm_out orelse return error.MissingTensor;
            dispatchDmmvOnCmd(engine, cmd, ssm_out_t, &engine.swiglu_buf, &engine.down_buf, hidden_dim, d_inner, 0);
            cmd.barrier();

            // Residual + FFN norm + router (same as attention batch2)
            {
                const res_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(@as(f32, 1.0))) };
                const res_bufs = [_]*const MetalBuffer{ &engine.hidden_buf, &engine.down_buf };
                cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &res_bufs, &res_push, @sizeOf(ScaleAccPush), 0);
            }
            cmd.barrier();
            dispatchRmsNormOnCmd(engine, cmd, &engine.hidden_buf, &engine.norm_buf, &engine.ffn_norm_bufs[layer_idx], hidden_dim, 1);
            cmd.barrier();
            if (is_moe) {
                const router_t = lt.ffn_gate_inp orelse return error.MissingTensor;
                dispatchDmmvOnCmd(engine, cmd, router_t, &engine.norm_buf, &engine.router_logits_buf, cfg.n_experts, hidden_dim, 0);
                if (use_gpu_routed_moe) {
                    try recordGpuRoutedBatchedMoeOnCmd(engine, cmd, lt, hidden_dim, inter_dim, shexp_inter_dim);
                    if (shared_cmd != null) cmd.barrier();
                }
            }
            if (using_local_cmd) cmd.commitAndWait();
        }

        // ===== MoE / Dense FFN =====
        if (is_moe and !use_gpu_routed_moe) {
            // CPU topK softmax
            const router_ptr: [*]const f32 = @ptrCast(@alignCast(engine.router_logits_buf.cpu_ptr.?));
            var expert_ids: [16]u32 = undefined;
            var expert_weights: [16]f32 = undefined;
            topKSoftmax(router_ptr[0..cfg.n_experts], cfg.n_experts_used, expert_ids[0..cfg.n_experts_used], expert_weights[0..cfg.n_experts_used]);

            // CPU shared expert gate: dot(gate_weights, norm_buf) → sigmoid
            // Eliminates mid-MoE commitAndWait (40 per token) — 2048-dim dot product on CPU (<1µs)
            var shexp_gate_weight: f32 = 1.0;
            if (lt.ffn_gate_inp_shexp != null) {
                if (engine.shexp_gate_weights) |sgw| {
                    const gate_w = sgw[layer_idx];
                    if (gate_w.len >= hidden_dim) {
                        const norm_ptr: [*]const f32 = @ptrCast(@alignCast(engine.norm_buf.cpu_ptr.?));
                        var dot: f32 = 0;
                        for (0..hidden_dim) |i| dot += gate_w[i] * norm_ptr[i];
                        shexp_gate_weight = 1.0 / (1.0 + @exp(-dot));
                    }
                }
            }

            // Expert dispatch — GPU: gate+up → SwiGLU → down → accumulate → [next batch1]
            const gate_exps = lt.ffn_gate_exps orelse return error.MissingTensor;
            const up_exps = lt.ffn_up_exps orelse return error.MissingTensor;
            const down_exps = lt.ffn_down_exps orelse return error.MissingTensor;
            const gate_quant = gate_exps.info.type_;
            const down_quant = down_exps.info.type_;
            const expert_gate_bytes = expertSliceBytes(gate_quant, inter_dim, hidden_dim);
            const expert_down_bytes = expertSliceBytes(down_quant, hidden_dim, inter_dim);
            const use_batched_q4k_moe = canUseBatchedQ4kMoe(engine, gate_quant, down_quant);

            {
                const expert_ids_ptr: [*]u32 = @ptrCast(@alignCast(engine.expert_ids_buf.cpu_ptr.?));
                @memcpy(expert_ids_ptr[0..cfg.n_experts_used], expert_ids[0..cfg.n_experts_used]);
            }

            {
                var cmd = try metal_command.beginCommand(engine.device.ctx);
                const gate_shexp = lt.ffn_gate_shexp;
                const up_shexp = lt.ffn_up_shexp;
                const down_shexp = lt.ffn_down_shexp;
                const has_shexp = gate_shexp != null and up_shexp != null and down_shexp != null;

                if (use_batched_q4k_moe) {
                    // Phase 1: Batched gate+up expert projections (+ shared expert)
                    dispatchDmmvMoeQ4kOnCmd(engine, &cmd, gate_exps, &engine.norm_buf, &engine.expert_gate_batch_buf, &engine.expert_ids_buf, inter_dim, hidden_dim, expert_gate_bytes, 0, 0);
                    dispatchDmmvMoeQ4kOnCmd(engine, &cmd, up_exps, &engine.norm_buf, &engine.expert_up_batch_buf, &engine.expert_ids_buf, inter_dim, hidden_dim, expert_gate_bytes, 0, 0);
                    if (has_shexp) {
                        dispatchDmmvOnCmd(engine, &cmd, gate_shexp.?, &engine.norm_buf, &engine.gate_buf, shexp_inter_dim, hidden_dim, 0);
                        dispatchDmmvOnCmd(engine, &cmd, up_shexp.?, &engine.norm_buf, &engine.up_buf, shexp_inter_dim, hidden_dim, 0);
                    }
                    cmd.barrier();

                    // Phase 2: Batched SwiGLU over all selected experts (+ shared expert)
                    {
                        const swiglu_push = SwiGLUPush{ .n = inter_dim };
                        const sw_bufs = [_]*const MetalBuffer{ &engine.expert_gate_batch_buf, &engine.expert_swiglu_batch_buf, &engine.expert_up_batch_buf };
                        cmd.dispatchV2(&engine.swiglu_batched_pipe, .{ (inter_dim + 63) / 64, cfg.n_experts_used, 1 }, .{ 64, 1, 1 }, &sw_bufs, &swiglu_push, @sizeOf(SwiGLUPush), 0);
                    }
                    if (has_shexp) {
                        const sw_push = SwiGLUPush{ .n = shexp_inter_dim };
                        const sw_bufs2 = [_]*const MetalBuffer{ &engine.gate_buf, &engine.swiglu_buf, &engine.up_buf };
                        cmd.dispatchV2(&engine.swiglu_pipe, .{ (shexp_inter_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &sw_bufs2, &sw_push, @sizeOf(SwiGLUPush), 0);
                    }
                    cmd.barrier();

                    // Phase 3: Batched down expert projection (+ shared expert)
                    dispatchDmmvMoeQ4kOnCmd(engine, &cmd, down_exps, &engine.expert_swiglu_batch_buf, &engine.expert_down_batch_buf, &engine.expert_ids_buf, hidden_dim, inter_dim, expert_down_bytes, inter_dim, 0);
                    if (down_shexp != null) {
                        dispatchDmmvOnCmd(engine, &cmd, down_shexp.?, &engine.swiglu_buf, &engine.down_buf, hidden_dim, shexp_inter_dim, 0);
                    }
                    cmd.barrier();

                    // Phase 4: Fused accumulate from contiguous expert-down buffer (+ shared expert)
                    const moe_push = MoeAccBatchedPush{
                        .n = hidden_dim,
                        .expert_stride = hidden_dim,
                        .w0 = expert_weights[0],
                        .w1 = expert_weights[1],
                        .w2 = expert_weights[2],
                        .w3 = expert_weights[3],
                        .w4 = expert_weights[4],
                        .w5 = expert_weights[5],
                        .w6 = expert_weights[6],
                        .w7 = expert_weights[7],
                        .w_sh = if (has_shexp) shexp_gate_weight else 0.0,
                    };
                    const moe_bufs = [_]*const MetalBuffer{
                        &engine.hidden_buf,
                        &engine.expert_down_batch_buf,
                        &engine.down_buf,
                    };
                    cmd.dispatchV2(&engine.moe_acc_batched_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &moe_bufs, &moe_push, @sizeOf(MoeAccBatchedPush), 3);
                    cmd.barrier();
                } else {
                    // Phase 1: All expert gate+up DMMVs in parallel (+ shared expert)
                    for (0..cfg.n_experts_used) |ei| {
                        const eid = expert_ids[ei];
                        const gate_offset = eid * expert_gate_bytes;
                        const up_offset = eid * expert_gate_bytes;
                        dispatchDmmvOnCmd(engine, &cmd, gate_exps, &engine.norm_buf, &engine.expert_gate_bufs[ei], inter_dim, hidden_dim, gate_offset);
                        dispatchDmmvOnCmd(engine, &cmd, up_exps, &engine.norm_buf, &engine.expert_up_bufs[ei], inter_dim, hidden_dim, up_offset);
                    }
                    if (has_shexp) {
                        dispatchDmmvOnCmd(engine, &cmd, gate_shexp.?, &engine.norm_buf, &engine.gate_buf, shexp_inter_dim, hidden_dim, 0);
                        dispatchDmmvOnCmd(engine, &cmd, up_shexp.?, &engine.norm_buf, &engine.up_buf, shexp_inter_dim, hidden_dim, 0);
                    }
                    cmd.barrier();

                    // Phase 2: All SwiGLU operations in parallel
                    for (0..cfg.n_experts_used) |ei| {
                        const swiglu_push = SwiGLUPush{ .n = inter_dim };
                        const sw_bufs = [_]*const MetalBuffer{ &engine.expert_gate_bufs[ei], &engine.expert_swiglu_bufs[ei], &engine.expert_up_bufs[ei] };
                        cmd.dispatchV2(&engine.swiglu_pipe, .{ (inter_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &sw_bufs, &swiglu_push, @sizeOf(SwiGLUPush), 0);
                    }
                    if (has_shexp) {
                        const sw_push = SwiGLUPush{ .n = shexp_inter_dim };
                        const sw_bufs2 = [_]*const MetalBuffer{ &engine.gate_buf, &engine.swiglu_buf, &engine.up_buf };
                        cmd.dispatchV2(&engine.swiglu_pipe, .{ (shexp_inter_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &sw_bufs2, &sw_push, @sizeOf(SwiGLUPush), 0);
                    }
                    cmd.barrier();

                    // Phase 3: All down DMMVs in parallel
                    for (0..cfg.n_experts_used) |ei| {
                        const eid = expert_ids[ei];
                        const down_offset = eid * expert_down_bytes;
                        dispatchDmmvOnCmd(engine, &cmd, down_exps, &engine.expert_swiglu_bufs[ei], &engine.expert_down_bufs[ei], hidden_dim, inter_dim, down_offset);
                    }
                    if (down_shexp != null) {
                        dispatchDmmvOnCmd(engine, &cmd, down_shexp.?, &engine.swiglu_buf, &engine.down_buf, hidden_dim, shexp_inter_dim, 0);
                    }
                    cmd.barrier();

                    // Phase 4: Fused MoE weighted accumulate — hidden += sum(w[i] * expert_down[i]) + w_sh * shared_down
                    // Single dispatch replaces 8+1 sequential scale_accumulate + barriers (eliminates 8 pipeline flushes per layer)
                    if (cfg.n_experts_used == 8) {
                        const moe_push = MoeAccPush{
                            .n = hidden_dim,
                            .w0 = expert_weights[0],
                            .w1 = expert_weights[1],
                            .w2 = expert_weights[2],
                            .w3 = expert_weights[3],
                            .w4 = expert_weights[4],
                            .w5 = expert_weights[5],
                            .w6 = expert_weights[6],
                            .w7 = expert_weights[7],
                            .w_sh = if (has_shexp) shexp_gate_weight else 0.0,
                        };
                        const moe_bufs = [_]*const MetalBuffer{
                            &engine.hidden_buf,
                            &engine.expert_down_bufs[0],
                            &engine.expert_down_bufs[1],
                            &engine.expert_down_bufs[2],
                            &engine.expert_down_bufs[3],
                            &engine.expert_down_bufs[4],
                            &engine.expert_down_bufs[5],
                            &engine.expert_down_bufs[6],
                            &engine.expert_down_bufs[7],
                            &engine.down_buf,
                        };
                        cmd.dispatchV2(&engine.moe_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &moe_bufs, &moe_push, @sizeOf(MoeAccPush), 10);
                        cmd.barrier();
                    } else {
                        // Fallback for non-8-expert models: sequential accumulate
                        for (0..cfg.n_experts_used) |ei| {
                            const w = expert_weights[ei];
                            const acc_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(w)) };
                            const acc_bufs = [_]*const MetalBuffer{ &engine.hidden_buf, &engine.expert_down_bufs[ei] };
                            cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &acc_bufs, &acc_push, @sizeOf(ScaleAccPush), 0);
                            cmd.barrier();
                        }
                        if (has_shexp) {
                            const shexp_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(shexp_gate_weight)) };
                            const shexp_bufs = [_]*const MetalBuffer{ &engine.hidden_buf, &engine.down_buf };
                            cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &shexp_bufs, &shexp_push, @sizeOf(ScaleAccPush), 0);
                            cmd.barrier();
                        }
                    }
                }

                cmd.commitAndWait();
            }
        } else if (!is_moe) {
            // Dense FFN (non-MoE) — norm_buf already set by GPU batch 2
            const gate_t = lt.ffn_gate orelse return error.MissingTensor;
            const up_t = lt.ffn_up orelse return error.MissingTensor;
            const down_t = lt.ffn_down orelse return error.MissingTensor;

            {
                var cmd = try metal_command.beginCommand(engine.device.ctx);
                dispatchDmmvOnCmd(engine, &cmd, gate_t, &engine.norm_buf, &engine.gate_buf, inter_dim, hidden_dim, 0);
                dispatchDmmvOnCmd(engine, &cmd, up_t, &engine.norm_buf, &engine.up_buf, inter_dim, hidden_dim, 0);
                cmd.barrier();

                const swiglu_push = SwiGLUPush{ .n = inter_dim };
                const sw_bufs = [_]*const MetalBuffer{ &engine.gate_buf, &engine.swiglu_buf, &engine.up_buf };
                cmd.dispatchV2(&engine.swiglu_pipe, .{ (inter_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &sw_bufs, &swiglu_push, @sizeOf(SwiGLUPush), 0);
                cmd.barrier();

                dispatchDmmvOnCmd(engine, &cmd, down_t, &engine.swiglu_buf, &engine.down_buf, hidden_dim, inter_dim, 0);
                cmd.commitAndWait();
            }

            const hidden_ptr: [*]f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));
            const d_ptr: [*]const f32 = @ptrCast(@alignCast(engine.down_buf.cpu_ptr.?));
            for (0..hidden_dim) |i| hidden_ptr[i] += d_ptr[i];
        }
    }

    // ===== Final: GPU norm → LM head (batched) =====
    if (shared_cmd) |cmd| {
        dispatchRmsNormOnCmd(engine, cmd, &engine.hidden_buf, &engine.norm_buf, &engine.final_norm_gpu, hidden_dim, 1);
        cmd.barrier();
        dispatchDmmvOnCmd(engine, cmd, engine.lm_head, &engine.norm_buf, &engine.logits_buf, cfg.vocab_size, hidden_dim, 0);
        cmd.commitAndWait();
    } else {
        var cmd = try metal_command.beginCommand(engine.device.ctx);
        dispatchRmsNormOnCmd(engine, &cmd, &engine.hidden_buf, &engine.norm_buf, &engine.final_norm_gpu, hidden_dim, 1);
        cmd.barrier();
        dispatchDmmvOnCmd(engine, &cmd, engine.lm_head, &engine.norm_buf, &engine.logits_buf, cfg.vocab_size, hidden_dim, 0);
        cmd.commitAndWait();
    }

    engine.position += 1;
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

pub fn generate(
    engine: *InferenceEngine,
    prompt_tokens: []const u32,
    max_tokens: u32,
    eos_id: u32,
    allocator: std.mem.Allocator,
) ![]u32 {
    var output: std.ArrayList(u32) = .{};
    errdefer output.deinit(allocator);

    engine.resetRequestState();

    // Prefill: process each prompt token through all layers
    const prefill_start = std.time.nanoTimestamp();
    for (prompt_tokens) |token_id| {
        try engine.loadTokenEmbedding(token_id);
        try runDecodeStep(engine);
    }
    const prefill_end = std.time.nanoTimestamp();
    const prefill_ns: u64 = @intCast(prefill_end - prefill_start);
    if (prompt_tokens.len > 0) {
        const prefill_tps = @as(f64, @floatFromInt(prompt_tokens.len)) * 1_000_000_000.0 / @as(f64, @floatFromInt(prefill_ns));
        log.info("Prefill: {d} tokens in {d:.1} ms ({d:.1} tok/s)", .{
            prompt_tokens.len, @as(f64, @floatFromInt(prefill_ns)) / 1_000_000.0, prefill_tps,
        });
    }

    // Sample first output token from prefill logits
    if (prompt_tokens.len > 0 and max_tokens > 0) {
        const first_token = engine.sampleGreedy();
        if (first_token == eos_id) {
            log.info("Generated 0 tokens (EOS at first position)", .{});
            return try output.toOwnedSlice(allocator);
        }
        try output.append(allocator, first_token);
    }

    // Decode loop
    const decode_start = std.time.nanoTimestamp();
    var tokens_generated: u32 = 1;
    while (tokens_generated < max_tokens) {
        const input_token = output.items[output.items.len - 1];
        try engine.loadTokenEmbedding(input_token);
        try runDecodeStep(engine);

        const next_token = engine.sampleGreedy();
        if (next_token == eos_id) break;

        try output.append(allocator, next_token);
        tokens_generated += 1;
    }
    const decode_end = std.time.nanoTimestamp();
    const decode_ns: u64 = @intCast(decode_end - decode_start);
    if (tokens_generated > 0) {
        const decode_tps = @as(f64, @floatFromInt(tokens_generated)) * 1_000_000_000.0 / @as(f64, @floatFromInt(decode_ns));
        const ms_per_tok = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(tokens_generated));
        log.info("Generated {d} tokens in {d:.1} ms — {d:.2} tok/s ({d:.1} ms/tok)", .{
            tokens_generated, @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0, decode_tps, ms_per_tok,
        });
    }
    return try output.toOwnedSlice(allocator);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

test "topKSoftmax selects correct top-k with renormalization" {
    var ids: [4]u32 = undefined;
    var weights: [4]f32 = undefined;
    const logits = [_]f32{ 1.0, 3.0, 2.0, 0.5, 4.0, 1.5, 0.1, 2.5 };
    topKSoftmax(&logits, 3, ids[0..3], weights[0..3]);
    try std.testing.expectEqual(@as(u32, 4), ids[0]);
    try std.testing.expectEqual(@as(u32, 1), ids[1]);
    try std.testing.expectEqual(@as(u32, 7), ids[2]);
    const wsum = weights[0] + weights[1] + weights[2];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), wsum, 0.01);
}

test "topKSoftmax with uniform logits returns equal weights" {
    var ids: [3]u32 = undefined;
    var weights: [3]f32 = undefined;
    const logits = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    topKSoftmax(&logits, 3, &ids, &weights);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), weights[0], 0.02);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), weights[1], 0.02);
}

test "expertSliceBytes Q4_K" {
    const bytes = expertSliceBytes(.q4_k, 1024, 2048);
    try std.testing.expectEqual(@as(u32, 1024 * 8 * 144), bytes);
}

test "dequantRow F32 direct copy" {
    var raw: [16]u8 = undefined;
    const src = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    @memcpy(&raw, std.mem.asBytes(&src));
    var out: [4]f32 = undefined;
    dequantRow(&raw, 0, 4, .f32, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), out[3], 0.001);
}

test "batched MoE Metal shaders compile" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var deinterleave_pipe = try loadShaderPipeline(ctx, "deinterleave");
    defer metal_pipeline.freePipeline(&deinterleave_pipe);

    var flash_attn_pipe = try loadShaderPipeline(ctx, "flash_attn");
    defer metal_pipeline.freePipeline(&flash_attn_pipe);

    var kv_cache_write_pipe = try loadShaderPipeline(ctx, "kv_cache_write");
    defer metal_pipeline.freePipeline(&kv_cache_write_pipe);

    var rope_pipe = try loadShaderPipeline(ctx, "rope_fused");
    defer metal_pipeline.freePipeline(&rope_pipe);

    var sigmoid_mul_pipe = try loadShaderPipeline(ctx, "sigmoid_mul");
    defer metal_pipeline.freePipeline(&sigmoid_mul_pipe);

    var dmmv_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe");
    defer metal_pipeline.freePipeline(&dmmv_pipe);

    var dmmv_pipe_k2048 = try loadShaderPipeline(ctx, "dmmv_q4k_k2048");
    defer metal_pipeline.freePipeline(&dmmv_pipe_k2048);

    var dmmv_moe_pipe_k2048 = try loadShaderPipeline(ctx, "dmmv_q4k_moe_k2048");
    defer metal_pipeline.freePipeline(&dmmv_moe_pipe_k2048);
    var dmmv_moe_pipe_k2048_1024 = try loadShaderPipeline(ctx, "dmmv_q4k_moe_k2048_1024");
    defer metal_pipeline.freePipeline(&dmmv_moe_pipe_k2048_1024);

    var swiglu_pipe = try loadShaderPipeline(ctx, "swiglu_batched");
    defer metal_pipeline.freePipeline(&swiglu_pipe);

    var acc_pipe = try loadShaderPipeline(ctx, "moe_accumulate_batched");
    defer metal_pipeline.freePipeline(&acc_pipe);

    var lmhead_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_lmhead");
    defer metal_pipeline.freePipeline(&lmhead_pipe);
    var lmhead_pipe_1024 = try loadShaderPipeline(ctx, "dmmv_q4k_lmhead_1024");
    defer metal_pipeline.freePipeline(&lmhead_pipe_1024);
    var topk_pipe = try loadShaderPipeline(ctx, "softmax_topk");
    defer metal_pipeline.freePipeline(&topk_pipe);
    var sigmoid_scale_acc_pipe = try loadShaderPipeline(ctx, "sigmoid_scale_acc");
    defer metal_pipeline.freePipeline(&sigmoid_scale_acc_pipe);
    var moe_weighted_acc_pipe = try loadShaderPipeline(ctx, "moe_weighted_acc");
    defer metal_pipeline.freePipeline(&moe_weighted_acc_pipe);

    try std.testing.expect(deinterleave_pipe.handle != null);
    try std.testing.expect(flash_attn_pipe.handle != null);
    try std.testing.expect(kv_cache_write_pipe.handle != null);
    try std.testing.expect(rope_pipe.handle != null);
    try std.testing.expect(sigmoid_mul_pipe.handle != null);
    try std.testing.expect(dmmv_pipe.handle != null);
    try std.testing.expect(dmmv_pipe_k2048.handle != null);
    try std.testing.expect(dmmv_moe_pipe_k2048.handle != null);
    try std.testing.expect(dmmv_moe_pipe_k2048_1024.handle != null);
    try std.testing.expect(swiglu_pipe.handle != null);
    try std.testing.expect(acc_pipe.handle != null);
    try std.testing.expect(lmhead_pipe.handle != null);
    try std.testing.expect(lmhead_pipe_1024.handle != null);
    try std.testing.expect(topk_pipe.handle != null);
    try std.testing.expect(sigmoid_scale_acc_pipe.handle != null);
    try std.testing.expect(moe_weighted_acc_pipe.handle != null);
}

test "deinterleave shader splits block-interleaved Q and gate" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "deinterleave");
    defer metal_pipeline.freePipeline(&pipe);

    const head_dim: u32 = 4;
    const n_heads: u32 = 2;
    const total: u32 = head_dim * n_heads;

    var input_buf = try metal_buffer.createBuffer(ctx, total * 2 * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var q_buf = try metal_buffer.createBuffer(ctx, total * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&q_buf);
    var gate_buf = try metal_buffer.createBuffer(ctx, total * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&gate_buf);

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    const q_ptr: [*]const f32 = @ptrCast(@alignCast(q_buf.cpu_ptr.?));
    const gate_ptr: [*]const f32 = @ptrCast(@alignCast(gate_buf.cpu_ptr.?));

    const src = [_]f32{
        10, 11, 12, 13, 20, 21, 22, 23,
        30, 31, 32, 33, 40, 41, 42, 43,
    };
    @memcpy(input_ptr[0..src.len], &src);
    @memset(q_buf.cpu_ptr.?[0..q_buf.size], 0);
    @memset(gate_buf.cpu_ptr.?[0..gate_buf.size], 0);

    const push = DeinterleavePush{ .head_dim = head_dim, .n_heads = n_heads };
    const bufs = [_]*const MetalBuffer{ &q_buf, &input_buf, &gate_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DeinterleavePush), 0);
    cmd.commitAndWait();

    try std.testing.expectEqualSlices(f32, &.{ 10, 11, 12, 13, 30, 31, 32, 33 }, q_ptr[0..total]);
    try std.testing.expectEqualSlices(f32, &.{ 20, 21, 22, 23, 40, 41, 42, 43 }, gate_ptr[0..total]);
}

test "softmax_topk shader selects top experts and normalized weights" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "softmax_topk");
    defer metal_pipeline.freePipeline(&pipe);

    var logits_buf = try metal_buffer.createBuffer(ctx, 8 * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&logits_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, 6 * @sizeOf(u32));
    defer metal_buffer.freeBuffer(&output_buf);

    const logits_ptr: [*]f32 = @ptrCast(@alignCast(logits_buf.cpu_ptr.?));
    @memcpy(logits_ptr[0..8], &[_]f32{ 1.0, 3.0, 2.0, 0.5, 4.0, 1.5, 0.1, 2.5 });
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    const push = SoftmaxTopkPush{ .n_experts = 8, .k = 3 };
    const bufs = [_]*const MetalBuffer{ &logits_buf, &output_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SoftmaxTopkPush), 0);
    cmd.commitAndWait();

    const out_ptr: [*]const u32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    try std.testing.expectEqual(@as(u32, 4), out_ptr[0]);
    try std.testing.expectEqual(@as(u32, 1), out_ptr[1]);
    try std.testing.expectEqual(@as(u32, 7), out_ptr[2]);

    const w0: f32 = @bitCast(out_ptr[3]);
    const w1: f32 = @bitCast(out_ptr[4]);
    const w2: f32 = @bitCast(out_ptr[5]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), w0 + w1 + w2, 0.01);
}

test "kv_cache_write shader writes K and V slices at token offset" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "kv_cache_write");
    defer metal_pipeline.freePipeline(&pipe);

    const kv_dim: u32 = 4;
    const dst_offset: u32 = 8;

    var src_k = try metal_buffer.createBuffer(ctx, kv_dim * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&src_k);
    var src_v = try metal_buffer.createBuffer(ctx, kv_dim * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&src_v);
    var dst_k = try metal_buffer.createBuffer(ctx, 16 * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&dst_k);
    var dst_v = try metal_buffer.createBuffer(ctx, 16 * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&dst_v);

    const src_k_ptr: [*]f32 = @ptrCast(@alignCast(src_k.cpu_ptr.?));
    const src_v_ptr: [*]f32 = @ptrCast(@alignCast(src_v.cpu_ptr.?));
    const dst_k_ptr: [*]f32 = @ptrCast(@alignCast(dst_k.cpu_ptr.?));
    const dst_v_ptr: [*]f32 = @ptrCast(@alignCast(dst_v.cpu_ptr.?));

    @memcpy(src_k_ptr[0..kv_dim], &[_]f32{ 1.0, 2.0, 3.0, 4.0 });
    @memcpy(src_v_ptr[0..kv_dim], &[_]f32{ 5.0, 6.0, 7.0, 8.0 });
    @memset(dst_k.cpu_ptr.?[0..dst_k.size], 0);
    @memset(dst_v.cpu_ptr.?[0..dst_v.size], 0);

    const push = KvCacheWritePush{
        .n = kv_dim,
        .dst_offset = dst_offset,
    };
    const bufs = [_]*const MetalBuffer{ &src_k, &src_v, &dst_k, &dst_v };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(KvCacheWritePush), 0);
    cmd.commitAndWait();

    try std.testing.expectEqualSlices(f32, &.{ 1.0, 2.0, 3.0, 4.0 }, dst_k_ptr[dst_offset .. dst_offset + kv_dim]);
    try std.testing.expectEqualSlices(f32, &.{ 5.0, 6.0, 7.0, 8.0 }, dst_v_ptr[dst_offset .. dst_offset + kv_dim]);
}

test "moe_weighted_acc shader adds weighted experts into destination" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "moe_weighted_acc");
    defer metal_pipeline.freePipeline(&pipe);

    const n: u32 = 4;
    const n_used: u32 = 2;

    var accum_buf = try metal_buffer.createBuffer(ctx, n * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&accum_buf);
    var src_buf = try metal_buffer.createBuffer(ctx, n_used * n * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&src_buf);
    var routing_buf = try metal_buffer.createBuffer(ctx, n_used * 2 * @sizeOf(u32));
    defer metal_buffer.freeBuffer(&routing_buf);

    const accum_ptr: [*]f32 = @ptrCast(@alignCast(accum_buf.cpu_ptr.?));
    @memcpy(accum_ptr[0..n], &[_]f32{ 10.0, 20.0, 30.0, 40.0 });

    const src_ptr: [*]f32 = @ptrCast(@alignCast(src_buf.cpu_ptr.?));
    @memcpy(src_ptr[0 .. n_used * n], &[_]f32{
        1.0, 2.0, 3.0, 4.0,
        5.0, 6.0, 7.0, 8.0,
    });

    const routing_ptr: [*]u32 = @ptrCast(@alignCast(routing_buf.cpu_ptr.?));
    routing_ptr[0] = 11;
    routing_ptr[1] = 29;
    routing_ptr[2] = @bitCast(@as(f32, 0.25));
    routing_ptr[3] = @bitCast(@as(f32, 0.75));

    const push = MoeWeightedAccPush{
        .n = n,
        .n_used = n_used,
        .src_stride = n,
    };
    const bufs = [_]*const MetalBuffer{ &accum_buf, &src_buf, &routing_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(MoeWeightedAccPush), 3);
    cmd.commitAndWait();

    try std.testing.expectApproxEqAbs(@as(f32, 14.0), accum_ptr[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 25.0), accum_ptr[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 36.0), accum_ptr[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 47.0), accum_ptr[3], 0.001);
}
