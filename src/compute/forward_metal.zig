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

pub const GenerateMetrics = struct {
    prefill_tokens: usize,
    prefill_ns: u64,
    prefill_tps: f64,
    generated_tokens: u32,
    decode_ns: u64,
    decode_tps: f64,
    ms_per_token: f64,
    eos_at_first_position: bool,
};

pub const GenerateResult = struct {
    output_tokens: []u32,
    metrics: GenerateMetrics,

    pub fn deinit(self: *GenerateResult, allocator: std.mem.Allocator) void {
        allocator.free(self.output_tokens);
        self.* = undefined;
    }
};

pub const InitOptions = struct {
    profile_enabled: bool = false,
    debug_validation_enabled: bool = false,
};

pub const RuntimeProfile = struct {
    decode_steps: u32 = 0,
    shared_cmd_steps: u32 = 0,
    command_buffers: u32 = 0,
    commit_waits: u32 = 0,
    sample_calls: u32 = 0,
    full_attn_layers: u32 = 0,
    ssm_layers: u32 = 0,
    gpu_routed_moe_layers: u32 = 0,
    fallback_moe_layers: u32 = 0,
    dense_ffn_layers: u32 = 0,
    embedding_ns: u64 = 0,
    layer_record_ns: u64 = 0,
    router_cpu_ns: u64 = 0,
    gpu_routed_moe_record_ns: u64 = 0,
    fallback_moe_record_ns: u64 = 0,
    dense_ffn_record_ns: u64 = 0,
    final_record_ns: u64 = 0,
    submit_wait_ns: u64 = 0,
    sample_ns: u64 = 0,
    total_step_ns: u64 = 0,
    debug_validation_ns: u64 = 0,

    fn reset(self: *RuntimeProfile) void {
        self.* = .{};
    }
};

fn profileStart(enabled: bool) i128 {
    return if (enabled) std.time.nanoTimestamp() else -1;
}

fn profileElapsedNs(start_ns: i128) u64 {
    if (start_ns < 0) return 0;
    const end_ns = std.time.nanoTimestamp();
    if (end_ns <= start_ns) return 0;
    return @intCast(end_ns - start_ns);
}

fn nsToMs(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1_000_000.0;
}

fn avgMs(ns: u64, count: anytype) f64 {
    const denom = @as(u64, count);
    if (denom == 0) return 0.0;
    return nsToMs(ns) / @as(f64, @floatFromInt(denom));
}

fn pctOf(total_ns: u64, part_ns: u64) f64 {
    if (total_ns == 0) return 0.0;
    return @as(f64, @floatFromInt(part_ns)) * 100.0 / @as(f64, @floatFromInt(total_ns));
}

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
    dmmv_q5k_moe_pipe: MetalPipeline,
    dmmv_q6k_moe_pipe: MetalPipeline,
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
    profile_enabled: bool,
    debug_validation_enabled: bool,
    request_profile: RuntimeProfile,

    pub fn init(
        model: *const metal_loader.Model,
        device: *const metal_device.MetalDevice,
        allocator: std.mem.Allocator,
        options: InitOptions,
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
        self.profile_enabled = options.profile_enabled;
        self.debug_validation_enabled = options.debug_validation_enabled;
        self.request_profile = .{};

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
        self.dmmv_q5k_moe_pipe = try loadShaderPipeline(ctx, "dmmv_q5k_moe");
        self.dmmv_q6k_moe_pipe = try loadShaderPipeline(ctx, "dmmv_q6k_moe");
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
            "Metal pipeline caps: dmmv_q4k_moe tw={d} max={d} stgmem={d} | dmmv_q5k_moe tw={d} max={d} stgmem={d} | dmmv_q6k_moe tw={d} max={d} stgmem={d} | dmmv_q4k_moe_k2048 tw={d} max={d} stgmem={d} | dmmv_q4k_moe_k2048_1024 tw={d} max={d} stgmem={d}",
            .{
                self.dmmv_q4k_moe_pipe.thread_execution_width,
                self.dmmv_q4k_moe_pipe.max_threads_per_threadgroup,
                self.dmmv_q4k_moe_pipe.static_threadgroup_memory_length,
                self.dmmv_q5k_moe_pipe.thread_execution_width,
                self.dmmv_q5k_moe_pipe.max_threads_per_threadgroup,
                self.dmmv_q5k_moe_pipe.static_threadgroup_memory_length,
                self.dmmv_q6k_moe_pipe.thread_execution_width,
                self.dmmv_q6k_moe_pipe.max_threads_per_threadgroup,
                self.dmmv_q6k_moe_pipe.static_threadgroup_memory_length,
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
        metal_pipeline.freePipeline(&self.dmmv_q5k_moe_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q6k_moe_pipe);
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
        const sample_start = profileStart(self.profile_enabled);
        defer if (self.profile_enabled) {
            const mutable = @constCast(self);
            mutable.request_profile.sample_calls += 1;
            mutable.request_profile.sample_ns += profileElapsedNs(sample_start);
        };

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
        self.request_profile.reset();

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

    pub fn enableProfiling(self: *InferenceEngine) !void {
        self.profile_enabled = true;
        self.request_profile.reset();
    }

    pub fn logRequestProfileSummary(self: *const InferenceEngine, label: []const u8, prompt_tokens: usize, completion_tokens: u32) void {
        if (!self.profile_enabled) return;

        const profile = self.request_profile;
        if (profile.decode_steps == 0 and profile.sample_calls == 0) return;

        const record_ns = profile.layer_record_ns +
            profile.gpu_routed_moe_record_ns +
            profile.fallback_moe_record_ns +
            profile.dense_ffn_record_ns +
            profile.final_record_ns;
        const traced_request_ns = profile.total_step_ns + profile.sample_ns;

        log.info("Metal profile ({s}): steps={d} prompt={d} completion={d} shared_steps={d} cmds={d} commits={d}", .{
            label,
            profile.decode_steps,
            prompt_tokens,
            completion_tokens,
            profile.shared_cmd_steps,
            profile.command_buffers,
            profile.commit_waits,
        });
        log.info("  cpu: embed {d:.2} ms ({d:.3} ms/step) | record {d:.2} ms ({d:.3} ms/step) | router {d:.2} ms | sample {d:.2} ms ({d:.3} ms/sample)", .{
            nsToMs(profile.embedding_ns),
            avgMs(profile.embedding_ns, profile.decode_steps),
            nsToMs(record_ns),
            avgMs(record_ns, profile.decode_steps),
            nsToMs(profile.router_cpu_ns),
            nsToMs(profile.sample_ns),
            avgMs(profile.sample_ns, profile.sample_calls),
        });
        log.info("  wait: submit {d:.2} ms ({d:.3} ms/step, {d:.1}% of traced time) | record breakdown layer {d:.2} ms gpu-moe {d:.2} ms fallback-moe {d:.2} ms dense {d:.2} ms final {d:.2} ms", .{
            nsToMs(profile.submit_wait_ns),
            avgMs(profile.submit_wait_ns, profile.decode_steps),
            pctOf(traced_request_ns, profile.submit_wait_ns),
            nsToMs(profile.layer_record_ns),
            nsToMs(profile.gpu_routed_moe_record_ns),
            nsToMs(profile.fallback_moe_record_ns),
            nsToMs(profile.dense_ffn_record_ns),
            nsToMs(profile.final_record_ns),
        });
        if (profile.decode_steps > 0) {
            const steps_f = @as(f64, @floatFromInt(profile.decode_steps));
            log.info("  mix/step: attn {d:.1} ssm {d:.1} gpu-moe {d:.1} fallback-moe {d:.1} dense {d:.1}", .{
                @as(f64, @floatFromInt(profile.full_attn_layers)) / steps_f,
                @as(f64, @floatFromInt(profile.ssm_layers)) / steps_f,
                @as(f64, @floatFromInt(profile.gpu_routed_moe_layers)) / steps_f,
                @as(f64, @floatFromInt(profile.fallback_moe_layers)) / steps_f,
                @as(f64, @floatFromInt(profile.dense_ffn_layers)) / steps_f,
            });
        }
        if (profile.gpu_routed_moe_layers == 0 and profile.fallback_moe_layers > 0 and self.layer_tensors.len > 0) {
            const layer0 = self.layer_tensors[0];
            log.info("  fallback-moe path: gate_exps={s} up_exps={s} down_exps={s} (GPU-routed path currently supports q4_k/q4_k/{{q4_k,q5_k}})", .{
                if (layer0.ffn_gate_exps) |t| @tagName(t.info.type_) else "-",
                if (layer0.ffn_up_exps) |t| @tagName(t.info.type_) else "-",
                if (layer0.ffn_down_exps) |t| @tagName(t.info.type_) else "-",
            });
        }
        if (profile.debug_validation_ns > 0) {
            log.info("  debug-validation {d:.2} ms", .{nsToMs(profile.debug_validation_ns)});
        }
    }

    fn loadTokenEmbedding(self: *InferenceEngine, token_id: u32) !void {
        const embed_start = profileStart(self.profile_enabled);
        defer if (self.profile_enabled) {
            self.request_profile.embedding_ns += profileElapsedNs(embed_start);
        };

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
    /// On Apple9/M4, the 1024-thread shape tends to trade away too much
    /// occupancy for reuse, so keep that path reserved for Apple10-class parts.
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
                    self.device.chip.isM5Class() and
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
        engine.device.chip.isM5Class() and
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

fn dispatchDmmvMoeQ5kOnCmd(
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
    if (tensor.info.type_ != .q5_k) {
        log.err("Batched MoE DMMV only supports Q5_K (tensor {s})", .{tensor.info.name});
        return;
    }
    if (engine.dmmv_q5k_moe_pipe.max_threads_per_threadgroup < 256) {
        log.err("Batched Q5_K MoE DMMV requires 256-thread threadgroups", .{});
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
    const rows_per_wg: u32 = 8;
    const block_size: u32 = 256;
    const wgs = (M + rows_per_wg - 1) / rows_per_wg;
    cmd.dispatchV2(&engine.dmmv_q5k_moe_pipe, .{ wgs, engine.config.n_experts_used, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(MoeDmmvPush), 1);
}

fn dispatchDmmvMoeQ6kOnCmd(
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
    if (tensor.info.type_ != .q6_k) {
        log.err("Batched MoE DMMV only supports Q6_K (tensor {s})", .{tensor.info.name});
        return;
    }
    if (engine.dmmv_q6k_moe_pipe.max_threads_per_threadgroup < 256) {
        log.err("Batched Q6_K MoE DMMV requires 256-thread threadgroups", .{});
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
    const rows_per_wg: u32 = 8;
    const block_size: u32 = 256;
    const wgs = (M + rows_per_wg - 1) / rows_per_wg;
    cmd.dispatchV2(&engine.dmmv_q6k_moe_pipe, .{ wgs, engine.config.n_experts_used, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(MoeDmmvPush), 1);
}

fn dispatchDmmvMoeOnCmd(
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
) !void {
    switch (tensor.info.type_) {
        .q4_k => dispatchDmmvMoeQ4kOnCmd(engine, cmd, tensor, input_buf, output_buf, routing_buf, M, K, expert_stride, x_expert_stride, extra_byte_offset),
        .q5_k => dispatchDmmvMoeQ5kOnCmd(engine, cmd, tensor, input_buf, output_buf, routing_buf, M, K, expert_stride, x_expert_stride, extra_byte_offset),
        .q6_k => dispatchDmmvMoeQ6kOnCmd(engine, cmd, tensor, input_buf, output_buf, routing_buf, M, K, expert_stride, x_expert_stride, extra_byte_offset),
        else => return error.UnsupportedQuantType,
    }
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
/// Use one simdgroup per threadgroup so the native Metal kernel stays on the
/// fast simdgroup reduction path without threadgroup-memory barriers.
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
    const simd_width = if (engine.rms_norm_pipe.thread_execution_width > 0 and
        engine.rms_norm_pipe.thread_execution_width <= engine.rms_norm_pipe.max_threads_per_threadgroup)
        engine.rms_norm_pipe.thread_execution_width
    else
        @as(u32, 32);
    cmd.dispatchV2(&engine.rms_norm_pipe, .{ n_groups, 1, 1 }, .{ simd_width, 1, 1 }, &bufs, &push, @sizeOf(RmsNormPush), 0);
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
        // Metal currently keeps the KV cache as a flat contiguous
        // [token][kv_head][head_dim] buffer. Use page_size=0 to select the
        // shader's contiguous-addressing fast path and skip page-table math.
        .page_size = 0,
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

fn dispatchSsmConv1dWithPipe(
    cmd: *MetalCommand,
    pipe: *const MetalPipeline,
    kernel: *const MetalBuffer,
    state: *const MetalBuffer,
    current_input: *const MetalBuffer,
    output: *const MetalBuffer,
    conv_channels: u32,
    d_conv: u32,
    kernel_is_f16: bool,
) void {
    const push = SsmConv1dPush{
        .conv_channels = conv_channels,
        .d_conv = d_conv,
        .kernel_is_f16 = if (kernel_is_f16) 1 else 0,
    };
    const bufs = [_]*const MetalBuffer{ kernel, state, current_input, output };
    cmd.dispatchV2(pipe, .{ (conv_channels + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SsmConv1dPush), 0);
}

fn dispatchSsmGatedNormWithPipe(
    cmd: *MetalCommand,
    pipe: *const MetalPipeline,
    delta_net_output: *const MetalBuffer,
    norm_weight: *const MetalBuffer,
    z_gate: *const MetalBuffer,
    output: *const MetalBuffer,
    d_inner: u32,
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    norm_per_head: bool,
) void {
    const push = SsmGatedNormPush{
        .d_inner = d_inner,
        .dt_rank = dt_rank,
        .head_v_dim = head_v_dim,
        .d_state = d_state,
        .norm_per_head = if (norm_per_head) 1 else 0,
    };
    const bufs = [_]*const MetalBuffer{ delta_net_output, norm_weight, z_gate, output };
    cmd.dispatchV2(pipe, .{ dt_rank, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SsmGatedNormPush), 0);
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

    // Keep the full-attention prep in a single compute encoder and rely on
    // Metal's in-order dispatch execution for the straight write->read chain.
    dispatchDmmvOnCmd(engine, cmd, q_tensor, &engine.norm_buf, &engine.attn_out_buf, q_full_dim, hidden_dim, 0);
    dispatchDmmvOnCmd(engine, cmd, k_tensor, &engine.norm_buf, &engine.k_buf, kv_dim, hidden_dim, 0);
    dispatchDmmvOnCmd(engine, cmd, v_tensor, &engine.norm_buf, &engine.v_buf, kv_dim, hidden_dim, 0);

    dispatchDeinterleaveOnCmd(engine, cmd, &engine.attn_out_buf, &engine.q_buf, &engine.gate_buf, cfg.head_dim, cfg.n_heads);

    if (engine.attn_q_norm_present[layer_idx]) {
        dispatchRmsNormOnCmd(engine, cmd, &engine.q_buf, &engine.q_buf, &engine.attn_q_norm_bufs[layer_idx], cfg.head_dim, cfg.n_heads);
    }
    if (engine.attn_k_norm_present[layer_idx]) {
        dispatchRmsNormOnCmd(engine, cmd, &engine.k_buf, &engine.k_buf, &engine.attn_k_norm_bufs[layer_idx], cfg.head_dim, cfg.n_kv_heads);
    }

    dispatchRopeOnCmd(engine, cmd, &engine.q_buf, &engine.q_buf, cfg.head_dim, rope_dim, cfg.n_heads, engine.position, cfg.rope_freq_base);
    dispatchRopeOnCmd(engine, cmd, &engine.k_buf, &engine.k_buf, cfg.head_dim, rope_dim, cfg.n_kv_heads, engine.position, cfg.rope_freq_base);

    dispatchKvCacheWriteOnCmd(engine, cmd, layer_idx, kv_dim, engine.position * kv_dim);
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
        .q6_k => {
            const bpb: usize = 210;
            const bpr = @as(usize, cols) / 256;
            const row_off = @as(usize, row) * bpr * bpb;
            var out_i: usize = 0;
            for (0..bpr) |bi| {
                const bb = row_off + bi * bpb;
                const d_bits = std.mem.readInt(u16, raw_data[bb + 208 ..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));

                var ql_off: usize = bb;
                var qh_off: usize = bb + 128;
                var sc_off: usize = bb + 192;
                for (0..2) |_| {
                    for (0..32) |l| {
                        const scale_idx = l / 16;
                        const ql_lo = raw_data[ql_off + l];
                        const ql_hi = raw_data[ql_off + l + 32];
                        const qh = raw_data[qh_off + l];

                        const rq0: u8 = (ql_lo & 0xF) | (((qh >> 0) & 3) << 4);
                        const rq1: u8 = (ql_hi & 0xF) | (((qh >> 2) & 3) << 4);
                        const rq2: u8 = (ql_lo >> 4) | (((qh >> 4) & 3) << 4);
                        const rq3: u8 = (ql_hi >> 4) | (((qh >> 6) & 3) << 4);

                        const q0: f32 = @floatFromInt(@as(i16, @intCast(rq0)) - 32);
                        const q1: f32 = @floatFromInt(@as(i16, @intCast(rq1)) - 32);
                        const q2: f32 = @floatFromInt(@as(i16, @intCast(rq2)) - 32);
                        const q3: f32 = @floatFromInt(@as(i16, @intCast(rq3)) - 32);

                        const s0: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_off + scale_idx])));
                        const s1: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_off + scale_idx + 2])));
                        const s2: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_off + scale_idx + 4])));
                        const s3: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_off + scale_idx + 6])));

                        output[out_i + l] = d * s0 * q0;
                        output[out_i + 32 + l] = d * s1 * q1;
                        output[out_i + 64 + l] = d * s2 * q2;
                        output[out_i + 96 + l] = d * s3 * q3;
                    }
                    ql_off += 64;
                    qh_off += 32;
                    sc_off += 8;
                    out_i += 128;
                }
            }
        },
        .q5_k => {
            const bpb: usize = 176;
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
                const qh = raw_data[bb + 16 .. bb + 48];
                const qs = raw_data[bb + 48 .. bb + 176];
                var is: usize = 0;
                for (0..4) |j| {
                    const sm0 = getScaleMinK4(is, scales);
                    const d1 = d * @as(f32, @floatFromInt(sm0.sc));
                    const m1 = dmin * @as(f32, @floatFromInt(sm0.m));
                    const sm1 = getScaleMinK4(is + 1, scales);
                    const d2 = d * @as(f32, @floatFromInt(sm1.sc));
                    const m2 = dmin * @as(f32, @floatFromInt(sm1.m));

                    for (0..32) |l| {
                        const ql_lo: u8 = qs[j * 32 + l] & 0xF;
                        const ql_hi: u8 = qs[j * 32 + l] >> 4;
                        const hb_lo: u8 = (qh[l] >> @intCast(j * 2)) & 1;
                        const hb_hi: u8 = (qh[l] >> @intCast(j * 2 + 1)) & 1;
                        output[out_i + l] = d1 * @as(f32, @floatFromInt(ql_lo | (hb_lo << 4))) - m1;
                        output[out_i + 32 + l] = d2 * @as(f32, @floatFromInt(ql_hi | (hb_hi << 4))) - m2;
                    }
                    out_i += 64;
                    is += 2;
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

fn canUseGpuRoutedMoeDown(engine: *const InferenceEngine, down_quant: GGMLType) bool {
    return switch (down_quant) {
        .q4_k => true,
        .q5_k => engine.dmmv_q5k_moe_pipe.handle != null,
        .q6_k => engine.dmmv_q6k_moe_pipe.handle != null,
        else => false,
    };
}

fn canUseGpuRoutedBatchedMoe(engine: *const InferenceEngine, lt: LayerTensors) bool {
    const gate_exps = lt.ffn_gate_exps orelse return false;
    const up_exps = lt.ffn_up_exps orelse return false;
    const down_exps = lt.ffn_down_exps orelse return false;

    if (engine.config.n_experts_used != 8) return false;
    if (!canUseGpuRoutedMoeDown(engine, gate_exps.info.type_)) return false;
    if (!canUseGpuRoutedMoeDown(engine, up_exps.info.type_)) return false;
    if (!canUseGpuRoutedMoeDown(engine, down_exps.info.type_)) return false;

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

    dispatchSoftmaxTopkOnCmd(engine, cmd, &engine.router_logits_buf, &engine.router_output_buf, cfg.n_experts, cfg.n_experts_used);

    try dispatchDmmvMoeOnCmd(engine, cmd, gate_exps, &engine.norm_buf, &engine.expert_gate_batch_buf, &engine.router_output_buf, inter_dim, hidden_dim, expert_gate_bytes, 0, 0);
    try dispatchDmmvMoeOnCmd(engine, cmd, up_exps, &engine.norm_buf, &engine.expert_up_batch_buf, &engine.router_output_buf, inter_dim, hidden_dim, expert_gate_bytes, 0, 0);
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
    try dispatchDmmvMoeOnCmd(engine, cmd, down_exps, &engine.expert_swiglu_batch_buf, &engine.expert_down_batch_buf, &engine.router_output_buf, hidden_dim, inter_dim, expert_down_bytes, inter_dim, 0);
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
    profile: ?*RuntimeProfile,
) !*MetalCommand {
    if (shared_cmd) |cmd| {
        using_local_cmd.* = false;
        return cmd;
    }

    local_cmd_storage.* = try beginProfiledCommand(engine, profile);
    using_local_cmd.* = true;
    return local_cmd_storage;
}

fn beginProfiledCommand(engine: *InferenceEngine, profile: ?*RuntimeProfile) !MetalCommand {
    const cmd = try metal_command.beginCommand(engine.device.ctx);
    if (profile) |p| p.command_buffers += 1;
    return cmd;
}

fn commitAndWaitProfiled(cmd: *MetalCommand, profile: ?*RuntimeProfile) void {
    const commit_start = profileStart(profile != null);
    cmd.commitAndWait();
    if (profile) |p| {
        p.commit_waits += 1;
        p.submit_wait_ns += profileElapsedNs(commit_start);
    }
}

// ---------------------------------------------------------------------------
// Decode step — runs all layers + final norm + LM head
// ---------------------------------------------------------------------------

fn runDecodeStep(engine: *InferenceEngine) !void {
    const step_start = profileStart(engine.profile_enabled);
    defer if (engine.profile_enabled) {
        engine.request_profile.total_step_ns += profileElapsedNs(step_start);
    };

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
    if (engine.profile_enabled and engine.position == 0 and is_moe and !use_single_gpu_cmd) {
        for (engine.layer_tensors, 0..) |lt, layer_idx| {
            if (canUseGpuRoutedBatchedMoe(engine, lt)) continue;
            log.info("Metal profile: shared token command disabled by layer {d} (gate_exps={s} up_exps={s} down_exps={s})", .{
                layer_idx,
                if (lt.ffn_gate_exps) |t| @tagName(t.info.type_) else "-",
                if (lt.ffn_up_exps) |t| @tagName(t.info.type_) else "-",
                if (lt.ffn_down_exps) |t| @tagName(t.info.type_) else "-",
            });
            break;
        }
    }
    const profile: ?*RuntimeProfile = if (engine.profile_enabled) &engine.request_profile else null;
    if (profile) |p| {
        p.decode_steps += 1;
        if (use_single_gpu_cmd) p.shared_cmd_steps += 1;
    }
    var shared_cmd_storage: MetalCommand = undefined;
    const shared_cmd: ?*MetalCommand = if (use_single_gpu_cmd) blk: {
        shared_cmd_storage = try beginProfiledCommand(engine, profile);
        break :blk &shared_cmd_storage;
    } else null;

    for (0..cfg.n_layers) |layer_idx| {
        const layer: u32 = @intCast(layer_idx);
        const lt = engine.layer_tensors[layer_idx];
        const is_full_attn = ((layer + 1) % full_attn_interval == 0);
        const use_gpu_routed_moe = is_moe and canUseGpuRoutedBatchedMoe(engine, lt);

        if (is_full_attn) {
            if (profile) |p| p.full_attn_layers += 1;
            var local_cmd_storage: MetalCommand = undefined;
            var using_local_cmd = false;
            var cmd = try acquireLayerCommand(engine, shared_cmd, &local_cmd_storage, &using_local_cmd, profile);
            const layer_record_start = profileStart(profile != null);
            dispatchRmsNormOnCmd(engine, cmd, &engine.hidden_buf, &engine.norm_buf, &engine.attn_norm_bufs[layer_idx], hidden_dim, 1);
            try dispatchFullAttnPrepOnCmd(engine, cmd, layer_idx, lt, q_dim, kv_dim, hidden_dim);
            dispatchFlashAttnOnCmd(engine, cmd, layer_idx, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, engine.position + 1);
            dispatchSigmoidMulOnCmd(engine, cmd, &engine.gate_buf, &engine.attn_out_buf, q_dim);
            const o_tensor = lt.attn_output orelse return error.MissingTensor;
            dispatchDmmvOnCmd(engine, cmd, o_tensor, &engine.attn_out_buf, &engine.down_buf, hidden_dim, q_dim, 0);
            cmd.barrier();
            const should_debug_attn_compare = engine.debug_validation_enabled and using_local_cmd and (engine.position == 4 or engine.position == 5) and (layer_idx == 7 or layer_idx == 31);
            if (should_debug_attn_compare) {
                commitAndWaitProfiled(cmd, profile);
                const debug_start = profileStart(profile != null);
                try debugCompareAttentionLayer(engine, layer, layer_idx, lt, hidden_dim, q_dim, kv_dim);
                if (profile) |p| p.debug_validation_ns += profileElapsedNs(debug_start);
                local_cmd_storage = try beginProfiledCommand(engine, profile);
                cmd = &local_cmd_storage;
            }
            {
                const res_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(@as(f32, 1.0))) };
                const res_bufs = [_]*const MetalBuffer{ &engine.hidden_buf, &engine.down_buf };
                cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &res_bufs, &res_push, @sizeOf(ScaleAccPush), 0);
            }
            dispatchRmsNormOnCmd(engine, cmd, &engine.hidden_buf, &engine.norm_buf, &engine.ffn_norm_bufs[layer_idx], hidden_dim, 1);
            if (is_moe) {
                const router_t = lt.ffn_gate_inp orelse return error.MissingTensor;
                dispatchDmmvOnCmd(engine, cmd, router_t, &engine.norm_buf, &engine.router_logits_buf, cfg.n_experts, hidden_dim, 0);
                if (use_gpu_routed_moe) {
                    if (profile) |p| p.layer_record_ns += profileElapsedNs(layer_record_start);
                    if (profile) |p| p.gpu_routed_moe_layers += 1;
                    const moe_record_start = profileStart(profile != null);
                    try recordGpuRoutedBatchedMoeOnCmd(engine, cmd, lt, hidden_dim, inter_dim, shexp_inter_dim);
                    if (profile) |p| p.gpu_routed_moe_record_ns += profileElapsedNs(moe_record_start);
                } else if (profile) |p| {
                    p.layer_record_ns += profileElapsedNs(layer_record_start);
                }
            } else if (profile) |p| {
                p.layer_record_ns += profileElapsedNs(layer_record_start);
            }
            if (using_local_cmd) commitAndWaitProfiled(cmd, profile);
        } else {
            if (profile) |p| p.ssm_layers += 1;
            // ===== SSM: fused batch 1 + recurrent body + batch 2 =====
            var local_cmd_storage: MetalCommand = undefined;
            var using_local_cmd = false;
            var cmd = try acquireLayerCommand(engine, shared_cmd, &local_cmd_storage, &using_local_cmd, profile);
            const layer_record_start = profileStart(profile != null);
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
                dispatchSsmConv1dWithPipe(
                    cmd,
                    &engine.ssm_conv1d_pipe,
                    &engine.ssm_conv_kernel_bufs.?[layer_idx],
                    &engine.ssm_conv_state_bufs.?[layer_idx],
                    &engine.attn_out_buf,
                    &engine.swiglu_buf,
                    conv_channels,
                    d_conv,
                    false,
                );
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
            const should_debug_ssm_compare = engine.debug_validation_enabled and engine.position == 0 and layer_idx == 6 and using_local_cmd;
            if (should_debug_ssm_compare) {
                commitAndWaitProfiled(cmd, profile);
                const debug_start = profileStart(profile != null);
                try debugCompareSsmPreGatedNorm(
                    engine,
                    layer,
                    layer_idx,
                    wqkv_t,
                    z_t,
                    alpha_t,
                    beta_t,
                    hidden_dim,
                    conv_channels,
                    d_inner,
                    d_conv,
                    d_state,
                    n_group,
                    dt_rank,
                    head_v_dim,
                );
                if (profile) |p| p.debug_validation_ns += profileElapsedNs(debug_start);
                local_cmd_storage = try beginProfiledCommand(engine, profile);
                cmd = &local_cmd_storage;
            }

            // Gated norm: attn_out_buf → swiglu_buf
            {
                dispatchSsmGatedNormWithPipe(
                    cmd,
                    &engine.ssm_gated_norm_pipe,
                    &engine.attn_out_buf,
                    &engine.ssm_norm_weight_bufs.?[layer_idx],
                    &engine.gate_buf,
                    &engine.swiglu_buf,
                    d_inner,
                    dt_rank,
                    head_v_dim,
                    d_state,
                    engine.ssm_norm_per_head.?[layer_idx],
                );
            }
            cmd.barrier();

            // SSM out DMMV: swiglu_buf → down_buf
            const ssm_out_t = lt.ssm_out orelse return error.MissingTensor;
            dispatchDmmvOnCmd(engine, cmd, ssm_out_t, &engine.swiglu_buf, &engine.down_buf, hidden_dim, d_inner, 0);
            cmd.barrier();
            if (should_debug_ssm_compare) {
                commitAndWaitProfiled(cmd, profile);
                const debug_start = profileStart(profile != null);
                try debugCompareSsmPostProjection(
                    engine,
                    layer,
                    layer_idx,
                    wqkv_t,
                    z_t,
                    alpha_t,
                    beta_t,
                    ssm_out_t,
                    hidden_dim,
                    conv_channels,
                    d_inner,
                    d_conv,
                    d_state,
                    n_group,
                    dt_rank,
                    head_v_dim,
                );
                if (profile) |p| p.debug_validation_ns += profileElapsedNs(debug_start);
                local_cmd_storage = try beginProfiledCommand(engine, profile);
                cmd = &local_cmd_storage;
            }

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
                    if (profile) |p| p.layer_record_ns += profileElapsedNs(layer_record_start);
                    if (profile) |p| p.gpu_routed_moe_layers += 1;
                    const moe_record_start = profileStart(profile != null);
                    try recordGpuRoutedBatchedMoeOnCmd(engine, cmd, lt, hidden_dim, inter_dim, shexp_inter_dim);
                    if (profile) |p| p.gpu_routed_moe_record_ns += profileElapsedNs(moe_record_start);
                } else if (profile) |p| {
                    p.layer_record_ns += profileElapsedNs(layer_record_start);
                }
            } else if (profile) |p| {
                p.layer_record_ns += profileElapsedNs(layer_record_start);
            }
            if (using_local_cmd) commitAndWaitProfiled(cmd, profile);
        }

        if (engine.debug_validation_enabled and engine.position == 0) {
            logLayerDiagnostics(engine, lt, layer, is_full_attn, "pre_ffn");
        }

        // ===== MoE / Dense FFN =====
        if (is_moe and !use_gpu_routed_moe) {
            if (profile) |p| p.fallback_moe_layers += 1;
            // CPU topK softmax
            const router_start = profileStart(profile != null);
            const router_ptr: [*]const f32 = @ptrCast(@alignCast(engine.router_logits_buf.cpu_ptr.?));
            var expert_ids: [16]u32 = undefined;
            var expert_weights: [16]f32 = undefined;
            topKSoftmax(router_ptr[0..cfg.n_experts], cfg.n_experts_used, expert_ids[0..cfg.n_experts_used], expert_weights[0..cfg.n_experts_used]);
            const should_debug_moe_compare = engine.debug_validation_enabled and engine.position == 0 and layer_idx == 6;
            const hidden_before_snapshot: ?[]f32 = if (should_debug_moe_compare) blk: {
                const snap = try engine.allocator.alloc(f32, hidden_dim);
                const hidden_ptr: [*]const f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));
                @memcpy(snap, hidden_ptr[0..hidden_dim]);
                break :blk snap;
            } else null;
            defer if (hidden_before_snapshot) |snap| engine.allocator.free(snap);

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
            if (profile) |p| p.router_cpu_ns += profileElapsedNs(router_start);

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
                const moe_record_start = profileStart(profile != null);
                var cmd = try beginProfiledCommand(engine, profile);
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

                if (profile) |p| p.fallback_moe_record_ns += profileElapsedNs(moe_record_start);
                commitAndWaitProfiled(&cmd, profile);
                if (hidden_before_snapshot) |snap| {
                    const debug_start = profileStart(profile != null);
                    try debugCompareMoeLayer(
                        engine,
                        layer,
                        lt,
                        expert_ids[0..cfg.n_experts_used],
                        expert_weights[0..cfg.n_experts_used],
                        shexp_gate_weight,
                        snap,
                        hidden_dim,
                        inter_dim,
                        shexp_inter_dim,
                    );
                    if (profile) |p| p.debug_validation_ns += profileElapsedNs(debug_start);
                }
            }
        } else if (!is_moe) {
            if (profile) |p| p.dense_ffn_layers += 1;
            // Dense FFN (non-MoE) — norm_buf already set by GPU batch 2
            const gate_t = lt.ffn_gate orelse return error.MissingTensor;
            const up_t = lt.ffn_up orelse return error.MissingTensor;
            const down_t = lt.ffn_down orelse return error.MissingTensor;

            {
                const dense_record_start = profileStart(profile != null);
                var cmd = try beginProfiledCommand(engine, profile);
                dispatchDmmvOnCmd(engine, &cmd, gate_t, &engine.norm_buf, &engine.gate_buf, inter_dim, hidden_dim, 0);
                dispatchDmmvOnCmd(engine, &cmd, up_t, &engine.norm_buf, &engine.up_buf, inter_dim, hidden_dim, 0);
                cmd.barrier();

                const swiglu_push = SwiGLUPush{ .n = inter_dim };
                const sw_bufs = [_]*const MetalBuffer{ &engine.gate_buf, &engine.swiglu_buf, &engine.up_buf };
                cmd.dispatchV2(&engine.swiglu_pipe, .{ (inter_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &sw_bufs, &swiglu_push, @sizeOf(SwiGLUPush), 0);
                cmd.barrier();

                dispatchDmmvOnCmd(engine, &cmd, down_t, &engine.swiglu_buf, &engine.down_buf, hidden_dim, inter_dim, 0);
                if (profile) |p| p.dense_ffn_record_ns += profileElapsedNs(dense_record_start);
                commitAndWaitProfiled(&cmd, profile);
            }

            const hidden_ptr: [*]f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));
            const d_ptr: [*]const f32 = @ptrCast(@alignCast(engine.down_buf.cpu_ptr.?));
            for (0..hidden_dim) |i| hidden_ptr[i] += d_ptr[i];
        }

        if (engine.debug_validation_enabled and engine.position == 0) {
            logLayerDiagnostics(engine, lt, layer, is_full_attn, "post_ffn");
        }
    }

    // ===== Final: GPU norm → LM head (batched) =====
    const final_record_start = profileStart(profile != null);
    if (shared_cmd) |cmd| {
        dispatchRmsNormOnCmd(engine, cmd, &engine.hidden_buf, &engine.norm_buf, &engine.final_norm_gpu, hidden_dim, 1);
        cmd.barrier();
        dispatchDmmvOnCmd(engine, cmd, engine.lm_head, &engine.norm_buf, &engine.logits_buf, cfg.vocab_size, hidden_dim, 0);
        if (profile) |p| p.final_record_ns += profileElapsedNs(final_record_start);
        commitAndWaitProfiled(cmd, profile);
    } else {
        var cmd = try beginProfiledCommand(engine, profile);
        dispatchRmsNormOnCmd(engine, &cmd, &engine.hidden_buf, &engine.norm_buf, &engine.final_norm_gpu, hidden_dim, 1);
        cmd.barrier();
        dispatchDmmvOnCmd(engine, &cmd, engine.lm_head, &engine.norm_buf, &engine.logits_buf, cfg.vocab_size, hidden_dim, 0);
        if (profile) |p| p.final_record_ns += profileElapsedNs(final_record_start);
        commitAndWaitProfiled(&cmd, profile);
    }
    if (engine.debug_validation_enabled and engine.position == 5) {
        const debug_start = profileStart(profile != null);
        try debugCompareFinalLogits(engine);
        if (profile) |p| p.debug_validation_ns += profileElapsedNs(debug_start);
    }

    engine.position += 1;
}

fn logDebugSliceDiff(layer: u32, label: []const u8, expected: []const f32, actual: []const f32) void {
    if (expected.len == 0 or actual.len == 0) return;
    const n = @min(expected.len, actual.len);
    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    var sum_sq: f64 = 0;
    for (0..n) |i| {
        const diff = actual[i] - expected[i];
        const abs_diff = @abs(diff);
        if (abs_diff > max_abs) {
            max_abs = abs_diff;
            max_idx = i;
        }
        sum_sq += @as(f64, diff) * @as(f64, diff);
    }
    const rms: f64 = @sqrt(sum_sq / @as(f64, @floatFromInt(n)));
    log.info("SSM_REF L{d} {s}: max_diff={d:.6} idx={d} exp={d:.6} got={d:.6} rms_diff={d:.6}", .{
        layer,
        label,
        max_abs,
        max_idx,
        expected[max_idx],
        actual[max_idx],
        rms,
    });
}

fn insertTopLogit(top_ids: *[5]u32, top_vals: *[5]f32, token_id: u32, value: f32) void {
    if (value <= top_vals[4]) return;
    var pos: usize = 4;
    while (pos > 0 and value > top_vals[pos - 1]) : (pos -= 1) {}
    var i: usize = 4;
    while (i > pos) : (i -= 1) {
        top_vals[i] = top_vals[i - 1];
        top_ids[i] = top_ids[i - 1];
    }
    top_vals[pos] = value;
    top_ids[pos] = token_id;
}

fn debugCompareFinalLogits(engine: *InferenceEngine) !void {
    const mmap = engine.model.mmap_data orelse return error.NoMmapData;
    const hidden_dim = engine.config.hidden_dim;
    const vocab_size = engine.config.vocab_size;
    const norm_w: [*]const f32 = @ptrCast(@alignCast(engine.final_norm_gpu.cpu_ptr.?));
    const hidden: [*]const f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));
    const gpu_logits: [*]const f32 = @ptrCast(@alignCast(engine.logits_buf.cpu_ptr.?));

    const allocator = engine.allocator;
    const normed = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(normed);
    const row_buf = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(row_buf);

    var sum_sq: f64 = 0;
    for (0..hidden_dim) |i| {
        const v = hidden[i];
        sum_sq += @as(f64, v) * @as(f64, v);
    }
    const rms_inv: f32 = @floatCast(1.0 / @sqrt(sum_sq / @as(f64, @floatFromInt(hidden_dim)) + 1e-6));
    for (0..hidden_dim) |i| {
        normed[i] = norm_w[i] * hidden[i] * rms_inv;
    }

    const lm_off: usize = @intCast(engine.model.gguf_file.tensor_data_offset + engine.lm_head.info.offset);
    const lm_raw = mmap[lm_off..];

    var cpu_top_ids: [5]u32 = .{ 0, 0, 0, 0, 0 };
    var gpu_top_ids: [5]u32 = .{ 0, 0, 0, 0, 0 };
    var cpu_top_vals = [_]f32{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
    var gpu_top_vals = [_]f32{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };

    for (0..vocab_size) |row| {
        dequantRow(lm_raw, @intCast(row), hidden_dim, engine.lm_head.info.type_, row_buf);
        var dot: f64 = 0;
        for (0..hidden_dim) |i| {
            dot += @as(f64, row_buf[i]) * @as(f64, normed[i]);
        }
        const cpu_val: f32 = @floatCast(dot);
        const gpu_val = gpu_logits[row];
        insertTopLogit(&cpu_top_ids, &cpu_top_vals, @intCast(row), cpu_val);
        insertTopLogit(&gpu_top_ids, &gpu_top_vals, @intCast(row), gpu_val);
    }

    log.info(
        "LOGITS_REF pos={d}: cpu_top=[{d}:{d:.4},{d}:{d:.4},{d}:{d:.4},{d}:{d:.4},{d}:{d:.4}] gpu_top=[{d}:{d:.4},{d}:{d:.4},{d}:{d:.4},{d}:{d:.4},{d}:{d:.4}]",
        .{
            engine.position,
            cpu_top_ids[0],
            cpu_top_vals[0],
            cpu_top_ids[1],
            cpu_top_vals[1],
            cpu_top_ids[2],
            cpu_top_vals[2],
            cpu_top_ids[3],
            cpu_top_vals[3],
            cpu_top_ids[4],
            cpu_top_vals[4],
            gpu_top_ids[0],
            gpu_top_vals[0],
            gpu_top_ids[1],
            gpu_top_vals[1],
            gpu_top_ids[2],
            gpu_top_vals[2],
            gpu_top_ids[3],
            gpu_top_vals[3],
            gpu_top_ids[4],
            gpu_top_vals[4],
        },
    );
}

fn debugCompareSsmPreGatedNorm(
    engine: *InferenceEngine,
    layer: u32,
    layer_idx: usize,
    wqkv_t: *const metal_loader.LoadedTensor,
    z_t: *const metal_loader.LoadedTensor,
    alpha_t: *const metal_loader.LoadedTensor,
    beta_t: *const metal_loader.LoadedTensor,
    hidden_dim: u32,
    conv_channels: u32,
    d_inner: u32,
    d_conv: u32,
    d_state: u32,
    n_group: u32,
    dt_rank: u32,
    head_v_dim: u32,
) !void {
    const allocator = engine.allocator;
    const mmap = engine.model.mmap_data orelse return error.NoMmapData;
    const tensor_data_off = engine.model.gguf_file.tensor_data_offset;

    const norm_in: [*]const f32 = @ptrCast(@alignCast(engine.norm_buf.cpu_ptr.?));
    const gpu_z: [*]const f32 = @ptrCast(@alignCast(engine.gate_buf.cpu_ptr.?));
    const gpu_alpha: [*]const f32 = @ptrCast(@alignCast(engine.router_logits_buf.cpu_ptr.?));
    const gpu_beta: [*]const f32 = @ptrCast(@alignCast(engine.down_buf.cpu_ptr.?));
    const gpu_conv_out: [*]const f32 = @ptrCast(@alignCast(engine.swiglu_buf.cpu_ptr.?));
    const gpu_delta_out: [*]const f32 = @ptrCast(@alignCast(engine.attn_out_buf.cpu_ptr.?));

    const qkv_ref = try allocator.alloc(f32, conv_channels);
    defer allocator.free(qkv_ref);
    const z_ref = try allocator.alloc(f32, d_inner);
    defer allocator.free(z_ref);
    const alpha_ref = try allocator.alloc(f32, dt_rank);
    defer allocator.free(alpha_ref);
    const beta_ref = try allocator.alloc(f32, dt_rank);
    defer allocator.free(beta_ref);
    const conv_state_ref = try allocator.alloc(f32, (d_conv - 1) * conv_channels);
    defer allocator.free(conv_state_ref);
    const conv_out_ref = try allocator.alloc(f32, conv_channels);
    defer allocator.free(conv_out_ref);
    const delta_state_ref = try allocator.alloc(f32, dt_rank * head_v_dim * head_v_dim);
    defer allocator.free(delta_state_ref);
    const delta_out_ref = try allocator.alloc(f32, d_inner);
    defer allocator.free(delta_out_ref);

    @memset(conv_state_ref, 0);
    @memset(delta_state_ref, 0);
    @memset(conv_out_ref, 0);
    @memset(delta_out_ref, 0);

    try cpuDmmvFallback(
        mmap,
        wqkv_t,
        tensor_data_off,
        norm_in,
        qkv_ref.ptr,
        conv_channels,
        hidden_dim,
        0,
        allocator,
    );
    try cpuDmmvFallback(
        mmap,
        z_t,
        tensor_data_off,
        norm_in,
        z_ref.ptr,
        d_inner,
        hidden_dim,
        0,
        allocator,
    );
    try cpuDmmvFallback(
        mmap,
        alpha_t,
        tensor_data_off,
        norm_in,
        alpha_ref.ptr,
        dt_rank,
        hidden_dim,
        0,
        allocator,
    );
    try cpuDmmvFallback(
        mmap,
        beta_t,
        tensor_data_off,
        norm_in,
        beta_ref.ptr,
        dt_rank,
        hidden_dim,
        0,
        allocator,
    );

    const conv_kernel_ptr: [*]const f32 = @ptrCast(@alignCast(engine.ssm_conv_kernel_bufs.?[layer_idx].cpu_ptr.?));
    const dt_bias_ptr: [*]const f32 = @ptrCast(@alignCast(engine.ssm_dt_bias_bufs.?[layer_idx].cpu_ptr.?));
    const ssm_a_ptr: [*]const f32 = @ptrCast(@alignCast(engine.ssm_a_bufs.?[layer_idx].cpu_ptr.?));

    refRunSsmConv1d(
        qkv_ref,
        conv_kernel_ptr[0 .. conv_channels * d_conv],
        conv_state_ref,
        conv_out_ref,
        conv_channels,
        d_conv,
    );
    refRunSsmDeltaNet(
        conv_out_ref,
        alpha_ref,
        dt_bias_ptr[0..dt_rank],
        beta_ref,
        ssm_a_ptr[0..dt_rank],
        delta_state_ref,
        delta_out_ref,
        dt_rank,
        head_v_dim,
        d_state,
        n_group,
    );

    logDebugSliceDiff(layer, "z_gate", z_ref[0..d_inner], gpu_z[0..d_inner]);
    logDebugSliceDiff(layer, "alpha", alpha_ref[0..dt_rank], gpu_alpha[0..dt_rank]);
    logDebugSliceDiff(layer, "beta", beta_ref[0..dt_rank], gpu_beta[0..dt_rank]);
    logDebugSliceDiff(layer, "conv_out", conv_out_ref[0..conv_channels], gpu_conv_out[0..conv_channels]);
    logDebugSliceDiff(layer, "delta_out", delta_out_ref[0..d_inner], gpu_delta_out[0..d_inner]);
}

fn debugCompareSsmPostProjection(
    engine: *InferenceEngine,
    layer: u32,
    layer_idx: usize,
    wqkv_t: *const metal_loader.LoadedTensor,
    z_t: *const metal_loader.LoadedTensor,
    alpha_t: *const metal_loader.LoadedTensor,
    beta_t: *const metal_loader.LoadedTensor,
    ssm_out_t: *const metal_loader.LoadedTensor,
    hidden_dim: u32,
    conv_channels: u32,
    d_inner: u32,
    d_conv: u32,
    d_state: u32,
    n_group: u32,
    dt_rank: u32,
    head_v_dim: u32,
) !void {
    const allocator = engine.allocator;
    const mmap = engine.model.mmap_data orelse return error.NoMmapData;
    const tensor_data_off = engine.model.gguf_file.tensor_data_offset;

    const norm_in: [*]const f32 = @ptrCast(@alignCast(engine.norm_buf.cpu_ptr.?));
    const gpu_gated: [*]const f32 = @ptrCast(@alignCast(engine.swiglu_buf.cpu_ptr.?));
    const gpu_proj: [*]const f32 = @ptrCast(@alignCast(engine.down_buf.cpu_ptr.?));

    const qkv_ref = try allocator.alloc(f32, conv_channels);
    defer allocator.free(qkv_ref);
    const z_ref = try allocator.alloc(f32, d_inner);
    defer allocator.free(z_ref);
    const alpha_ref = try allocator.alloc(f32, dt_rank);
    defer allocator.free(alpha_ref);
    const beta_ref = try allocator.alloc(f32, dt_rank);
    defer allocator.free(beta_ref);
    const conv_state_ref = try allocator.alloc(f32, (d_conv - 1) * conv_channels);
    defer allocator.free(conv_state_ref);
    const conv_out_ref = try allocator.alloc(f32, conv_channels);
    defer allocator.free(conv_out_ref);
    const delta_state_ref = try allocator.alloc(f32, dt_rank * head_v_dim * head_v_dim);
    defer allocator.free(delta_state_ref);
    const delta_out_ref = try allocator.alloc(f32, d_inner);
    defer allocator.free(delta_out_ref);
    const gated_ref = try allocator.alloc(f32, d_inner);
    defer allocator.free(gated_ref);
    const proj_ref = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(proj_ref);

    @memset(conv_state_ref, 0);
    @memset(delta_state_ref, 0);
    @memset(conv_out_ref, 0);
    @memset(delta_out_ref, 0);
    @memset(gated_ref, 0);

    try cpuDmmvFallback(
        mmap,
        wqkv_t,
        tensor_data_off,
        norm_in,
        qkv_ref.ptr,
        conv_channels,
        hidden_dim,
        0,
        allocator,
    );
    try cpuDmmvFallback(
        mmap,
        z_t,
        tensor_data_off,
        norm_in,
        z_ref.ptr,
        d_inner,
        hidden_dim,
        0,
        allocator,
    );
    try cpuDmmvFallback(
        mmap,
        alpha_t,
        tensor_data_off,
        norm_in,
        alpha_ref.ptr,
        dt_rank,
        hidden_dim,
        0,
        allocator,
    );
    try cpuDmmvFallback(
        mmap,
        beta_t,
        tensor_data_off,
        norm_in,
        beta_ref.ptr,
        dt_rank,
        hidden_dim,
        0,
        allocator,
    );

    const conv_kernel_ptr: [*]const f32 = @ptrCast(@alignCast(engine.ssm_conv_kernel_bufs.?[layer_idx].cpu_ptr.?));
    const dt_bias_ptr: [*]const f32 = @ptrCast(@alignCast(engine.ssm_dt_bias_bufs.?[layer_idx].cpu_ptr.?));
    const ssm_a_ptr: [*]const f32 = @ptrCast(@alignCast(engine.ssm_a_bufs.?[layer_idx].cpu_ptr.?));
    const norm_weight_ptr: [*]const f32 = @ptrCast(@alignCast(engine.ssm_norm_weight_bufs.?[layer_idx].cpu_ptr.?));
    const norm_len: u32 = if (engine.ssm_norm_per_head.?[layer_idx]) d_inner else d_state;

    refRunSsmConv1d(
        qkv_ref,
        conv_kernel_ptr[0 .. conv_channels * d_conv],
        conv_state_ref,
        conv_out_ref,
        conv_channels,
        d_conv,
    );
    refRunSsmDeltaNet(
        conv_out_ref,
        alpha_ref,
        dt_bias_ptr[0..dt_rank],
        beta_ref,
        ssm_a_ptr[0..dt_rank],
        delta_state_ref,
        delta_out_ref,
        dt_rank,
        head_v_dim,
        d_state,
        n_group,
    );
    refRunSsmGatedNorm(
        delta_out_ref,
        z_ref,
        norm_weight_ptr[0..norm_len],
        gated_ref,
        dt_rank,
        head_v_dim,
        d_state,
        engine.ssm_norm_per_head.?[layer_idx],
    );
    try cpuDmmvFallback(
        mmap,
        ssm_out_t,
        tensor_data_off,
        gated_ref.ptr,
        proj_ref.ptr,
        hidden_dim,
        d_inner,
        0,
        allocator,
    );

    logDebugSliceDiff(layer, "gated_norm", gated_ref[0..d_inner], gpu_gated[0..d_inner]);
    logDebugSliceDiff(layer, "ssm_out_proj", proj_ref[0..hidden_dim], gpu_proj[0..hidden_dim]);
}

fn debugCompareMoeLayer(
    engine: *InferenceEngine,
    layer: u32,
    lt: LayerTensors,
    expert_ids: []const u32,
    expert_weights: []const f32,
    shexp_gate_weight: f32,
    hidden_before: []const f32,
    hidden_dim: u32,
    inter_dim: u32,
    shexp_inter_dim: u32,
) !void {
    const allocator = engine.allocator;
    const mmap = engine.model.mmap_data orelse return error.NoMmapData;
    const tensor_data_off = engine.model.gguf_file.tensor_data_offset;

    const norm_in: [*]const f32 = @ptrCast(@alignCast(engine.norm_buf.cpu_ptr.?));
    const hidden_after: [*]const f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));

    const gate_exps = lt.ffn_gate_exps orelse return error.MissingTensor;
    const up_exps = lt.ffn_up_exps orelse return error.MissingTensor;
    const down_exps = lt.ffn_down_exps orelse return error.MissingTensor;
    const expert_gate_bytes = expertSliceBytes(gate_exps.info.type_, inter_dim, hidden_dim);
    const expert_down_bytes = expertSliceBytes(down_exps.info.type_, hidden_dim, inter_dim);

    const gate_buf = try allocator.alloc(f32, inter_dim);
    defer allocator.free(gate_buf);
    const up_buf = try allocator.alloc(f32, inter_dim);
    defer allocator.free(up_buf);
    const swiglu_buf = try allocator.alloc(f32, inter_dim);
    defer allocator.free(swiglu_buf);
    const down_buf = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(down_buf);
    const expected_delta = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(expected_delta);
    @memset(expected_delta, 0);

    for (expert_ids, expert_weights) |eid, weight| {
        const gate_offset = eid * expert_gate_bytes;
        const up_offset = eid * expert_gate_bytes;
        const down_offset = eid * expert_down_bytes;

        try cpuDmmvFallback(
            mmap,
            gate_exps,
            tensor_data_off,
            norm_in,
            gate_buf.ptr,
            inter_dim,
            hidden_dim,
            gate_offset,
            allocator,
        );
        try cpuDmmvFallback(
            mmap,
            up_exps,
            tensor_data_off,
            norm_in,
            up_buf.ptr,
            inter_dim,
            hidden_dim,
            up_offset,
            allocator,
        );
        for (0..inter_dim) |i| {
            const g = gate_buf[i];
            swiglu_buf[i] = (g / (1.0 + @exp(-g))) * up_buf[i];
        }
        try cpuDmmvFallback(
            mmap,
            down_exps,
            tensor_data_off,
            swiglu_buf.ptr,
            down_buf.ptr,
            hidden_dim,
            inter_dim,
            down_offset,
            allocator,
        );
        for (0..hidden_dim) |i| {
            expected_delta[i] += weight * down_buf[i];
        }
    }

    if (lt.ffn_gate_shexp != null and lt.ffn_up_shexp != null and lt.ffn_down_shexp != null) {
        const gate_sh = try allocator.alloc(f32, shexp_inter_dim);
        defer allocator.free(gate_sh);
        const up_sh = try allocator.alloc(f32, shexp_inter_dim);
        defer allocator.free(up_sh);
        const sw_sh = try allocator.alloc(f32, shexp_inter_dim);
        defer allocator.free(sw_sh);
        const down_sh = try allocator.alloc(f32, hidden_dim);
        defer allocator.free(down_sh);

        try cpuDmmvFallback(
            mmap,
            lt.ffn_gate_shexp.?,
            tensor_data_off,
            norm_in,
            gate_sh.ptr,
            shexp_inter_dim,
            hidden_dim,
            0,
            allocator,
        );
        try cpuDmmvFallback(
            mmap,
            lt.ffn_up_shexp.?,
            tensor_data_off,
            norm_in,
            up_sh.ptr,
            shexp_inter_dim,
            hidden_dim,
            0,
            allocator,
        );
        for (0..shexp_inter_dim) |i| {
            const g = gate_sh[i];
            sw_sh[i] = (g / (1.0 + @exp(-g))) * up_sh[i];
        }
        try cpuDmmvFallback(
            mmap,
            lt.ffn_down_shexp.?,
            tensor_data_off,
            sw_sh.ptr,
            down_sh.ptr,
            hidden_dim,
            shexp_inter_dim,
            0,
            allocator,
        );
        for (0..hidden_dim) |i| {
            expected_delta[i] += shexp_gate_weight * down_sh[i];
        }
    }

    const actual_delta = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(actual_delta);
    for (0..hidden_dim) |i| {
        actual_delta[i] = hidden_after[i] - hidden_before[i];
    }

    logDebugSliceDiff(layer, "moe_delta", expected_delta[0..hidden_dim], actual_delta[0..hidden_dim]);
}

fn refDeinterleaveQGate(input: []const f32, q: []f32, gate: []f32, head_dim: usize, n_heads: usize) void {
    for (0..n_heads) |h| {
        const src = input[h * head_dim * 2 ..][0 .. head_dim * 2];
        @memcpy(q[h * head_dim ..][0..head_dim], src[0..head_dim]);
        @memcpy(gate[h * head_dim ..][0..head_dim], src[head_dim .. head_dim * 2]);
    }
}

fn refFlashAttnContiguous(
    q: []const f32,
    k_cache: []const f32,
    v_cache: []const f32,
    out: []f32,
    head_dim: usize,
    n_heads: usize,
    n_kv_heads: usize,
    seq_len: usize,
) void {
    const q_per_kv = @max(n_heads / @max(n_kv_heads, 1), 1);
    const token_stride = n_kv_heads * head_dim;
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

    for (0..n_heads) |head| {
        const kv_head = head / q_per_kv;
        const q_head = q[head * head_dim ..][0..head_dim];
        const out_head = out[head * head_dim ..][0..head_dim];
        @memset(out_head, 0);

        var max_score: f32 = -std.math.inf(f32);
        for (0..seq_len) |tok| {
            const kv_base = tok * token_stride + kv_head * head_dim;
            const k_head = k_cache[kv_base..][0..head_dim];
            var score: f32 = 0;
            for (0..head_dim) |i| score += q_head[i] * k_head[i];
            score *= scale;
            if (score > max_score) max_score = score;
        }

        var denom: f32 = 0;
        for (0..seq_len) |tok| {
            const kv_base = tok * token_stride + kv_head * head_dim;
            const k_head = k_cache[kv_base..][0..head_dim];
            const v_head = v_cache[kv_base..][0..head_dim];
            var score: f32 = 0;
            for (0..head_dim) |i| score += q_head[i] * k_head[i];
            const weight = @exp(score * scale - max_score);
            denom += weight;
            for (0..head_dim) |i| out_head[i] += weight * v_head[i];
        }

        if (denom > 0) {
            const inv = 1.0 / denom;
            for (0..head_dim) |i| out_head[i] *= inv;
        }
    }
}

fn debugCompareAttentionLayer(
    engine: *InferenceEngine,
    layer: u32,
    layer_idx: usize,
    lt: LayerTensors,
    hidden_dim: u32,
    q_dim: u32,
    kv_dim: u32,
) !void {
    const allocator = engine.allocator;
    const mmap = engine.model.mmap_data orelse return error.NoMmapData;
    const tensor_data_off = engine.model.gguf_file.tensor_data_offset;
    const cfg = engine.config;
    const head_dim: usize = @intCast(cfg.head_dim);
    const n_heads: usize = @intCast(cfg.n_heads);
    const n_kv_heads: usize = @intCast(cfg.n_kv_heads);
    const rope_dim: u32 = if (cfg.rope_dim > 0) cfg.rope_dim else cfg.head_dim;
    const seq_len: usize = @intCast(engine.position + 1);

    const q_tensor = lt.attn_q orelse return error.MissingTensor;
    const k_tensor = lt.attn_k orelse return error.MissingTensor;
    const v_tensor = lt.attn_v orelse return error.MissingTensor;
    const o_tensor = lt.attn_output orelse return error.MissingTensor;

    const norm_in: [*]const f32 = @ptrCast(@alignCast(engine.norm_buf.cpu_ptr.?));
    const q_actual: [*]const f32 = @ptrCast(@alignCast(engine.q_buf.cpu_ptr.?));
    const k_actual: [*]const f32 = @ptrCast(@alignCast(engine.k_buf.cpu_ptr.?));
    const v_actual: [*]const f32 = @ptrCast(@alignCast(engine.v_buf.cpu_ptr.?));
    const gate_actual: [*]const f32 = @ptrCast(@alignCast(engine.gate_buf.cpu_ptr.?));
    const attn_actual: [*]const f32 = @ptrCast(@alignCast(engine.attn_out_buf.cpu_ptr.?));
    const oproj_actual: [*]const f32 = @ptrCast(@alignCast(engine.down_buf.cpu_ptr.?));
    const k_cache_actual: [*]const f32 = @ptrCast(@alignCast(engine.kv_k_cache[layer_idx].cpu_ptr.?));
    const v_cache_actual: [*]const f32 = @ptrCast(@alignCast(engine.kv_v_cache[layer_idx].cpu_ptr.?));

    const q_full_dim: u32 = q_dim * 2;
    const q_full_ref = try allocator.alloc(f32, q_full_dim);
    defer allocator.free(q_full_ref);
    const q_ref = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_ref);
    const k_ref = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k_ref);
    const v_ref = try allocator.alloc(f32, kv_dim);
    defer allocator.free(v_ref);
    const gate_ref = try allocator.alloc(f32, q_dim);
    defer allocator.free(gate_ref);
    const flash_ref = try allocator.alloc(f32, q_dim);
    defer allocator.free(flash_ref);
    const gated_ref = try allocator.alloc(f32, q_dim);
    defer allocator.free(gated_ref);
    const oproj_ref = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(oproj_ref);

    try cpuDmmvFallback(
        mmap,
        q_tensor,
        tensor_data_off,
        norm_in,
        q_full_ref.ptr,
        q_full_dim,
        hidden_dim,
        0,
        allocator,
    );
    try cpuDmmvFallback(
        mmap,
        k_tensor,
        tensor_data_off,
        norm_in,
        k_ref.ptr,
        kv_dim,
        hidden_dim,
        0,
        allocator,
    );
    try cpuDmmvFallback(
        mmap,
        v_tensor,
        tensor_data_off,
        norm_in,
        v_ref.ptr,
        kv_dim,
        hidden_dim,
        0,
        allocator,
    );

    refDeinterleaveQGate(q_full_ref, q_ref, gate_ref, head_dim, n_heads);
    if (engine.attn_q_norm_present[layer_idx]) {
        const qn_w: [*]const f32 = @ptrCast(@alignCast(engine.attn_q_norm_bufs[layer_idx].cpu_ptr.?));
        cpuRmsNormMul(q_ref.ptr, qn_w[0..head_dim], q_ref.ptr, cfg.head_dim, cfg.n_heads, 1e-6);
    }
    if (engine.attn_k_norm_present[layer_idx]) {
        const kn_w: [*]const f32 = @ptrCast(@alignCast(engine.attn_k_norm_bufs[layer_idx].cpu_ptr.?));
        cpuRmsNormMul(k_ref.ptr, kn_w[0..head_dim], k_ref.ptr, cfg.head_dim, cfg.n_kv_heads, 1e-6);
    }
    cpuRope(q_ref.ptr, cfg.head_dim, rope_dim, cfg.n_heads, engine.position, cfg.rope_freq_base);
    cpuRope(k_ref.ptr, cfg.head_dim, rope_dim, cfg.n_kv_heads, engine.position, cfg.rope_freq_base);

    const kv_offset: usize = @intCast(engine.position * kv_dim);
    logDebugSliceDiff(layer, "attn_q", q_ref[0..q_dim], q_actual[0..q_dim]);
    logDebugSliceDiff(layer, "attn_k", k_ref[0..kv_dim], k_actual[0..kv_dim]);
    logDebugSliceDiff(layer, "attn_v", v_ref[0..kv_dim], v_actual[0..kv_dim]);
    logDebugSliceDiff(layer, "attn_gate", gate_ref[0..q_dim], gate_actual[0..q_dim]);
    logDebugSliceDiff(layer, "kv_write_k", k_ref[0..kv_dim], k_cache_actual[kv_offset .. kv_offset + kv_dim]);
    logDebugSliceDiff(layer, "kv_write_v", v_ref[0..kv_dim], v_cache_actual[kv_offset .. kv_offset + kv_dim]);

    refFlashAttnContiguous(
        q_actual[0..q_dim],
        k_cache_actual[0 .. seq_len * @as(usize, kv_dim)],
        v_cache_actual[0 .. seq_len * @as(usize, kv_dim)],
        flash_ref,
        head_dim,
        n_heads,
        n_kv_heads,
        seq_len,
    );
    for (0..@as(usize, q_dim)) |i| {
        const g = gate_actual[i];
        gated_ref[i] = flash_ref[i] * (1.0 / (1.0 + @exp(-g)));
    }
    try cpuDmmvFallback(
        mmap,
        o_tensor,
        tensor_data_off,
        gated_ref.ptr,
        oproj_ref.ptr,
        hidden_dim,
        q_dim,
        0,
        allocator,
    );

    logDebugSliceDiff(layer, "flash_gated", gated_ref[0..q_dim], attn_actual[0..q_dim]);
    logDebugSliceDiff(layer, "attn_o_proj", oproj_ref[0..hidden_dim], oproj_actual[0..hidden_dim]);
}

fn logLayerDiagnostics(engine: *InferenceEngine, lt: LayerTensors, layer: u32, is_full_attn: bool, stage: []const u8) void {
    const hidden_dim = engine.config.hidden_dim;
    if (hidden_dim == 0) return;

    const hidden_ptr: [*]const f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));
    const hidden = hidden_ptr[0..hidden_dim];

    var sum_sq: f64 = 0;
    var max_abs: f32 = 0;
    for (hidden) |v| {
        sum_sq += @as(f64, v) * @as(f64, v);
        const av = @abs(v);
        if (av > max_abs) max_abs = av;
    }
    const rms: f32 = @floatCast(@sqrt(sum_sq / @as(f64, @floatFromInt(hidden_dim))));

    var logit5: f32 = 0;
    if (hidden_dim <= 8192 and engine.model.mmap_data != null) {
        const norm_w: [*]const f32 = @ptrCast(@alignCast(engine.final_norm_gpu.cpu_ptr.?));
        const rms_inv: f32 = @floatCast(1.0 / @sqrt(sum_sq / @as(f64, @floatFromInt(hidden_dim)) + 1e-6));
        const mmap = engine.model.mmap_data.?;
        const lm_off: usize = @intCast(engine.model.gguf_file.tensor_data_offset + engine.lm_head.info.offset);
        var lm_row: [8192]f32 = undefined;
        dequantRow(mmap[lm_off..], 5, hidden_dim, engine.lm_head.info.type_, lm_row[0..hidden_dim]);
        var dot: f64 = 0;
        for (0..hidden_dim) |i| {
            const normed = @as(f64, norm_w[i]) * @as(f64, hidden[i]) * @as(f64, rms_inv);
            dot += normed * @as(f64, lm_row[i]);
        }
        logit5 = @floatCast(dot);
    }

    if (layer == 0) {
        log.info("L0 quant: qkv={s} attn_gate={s} ssm_out={s} gate_exps={s} down_exps={s}", .{
            if (lt.attn_qkv) |t| @tagName(t.info.type_) else "-",
            if (lt.attn_gate) |t| @tagName(t.info.type_) else "-",
            if (lt.ssm_out) |t| @tagName(t.info.type_) else "-",
            if (lt.ffn_gate_exps) |t| @tagName(t.info.type_) else "-",
            if (lt.ffn_down_exps) |t| @tagName(t.info.type_) else "-",
        });
    }

    if (std.mem.eql(u8, stage, "pre_ffn")) {
        const core_dim: usize = if (is_full_attn) @intCast(engine.config.n_heads * engine.config.head_dim) else @intCast(engine.config.ssm_d_inner);
        const core_ptr: [*]const f32 = @ptrCast(@alignCast(if (is_full_attn) engine.attn_out_buf.cpu_ptr.? else engine.swiglu_buf.cpu_ptr.?));
        const proj_ptr: [*]const f32 = @ptrCast(@alignCast(engine.down_buf.cpu_ptr.?));
        const core = core_ptr[0..core_dim];
        const proj = proj_ptr[0..hidden_dim];

        var core_sq: f64 = 0;
        var core_max: f32 = 0;
        for (core) |v| {
            core_sq += @as(f64, v) * @as(f64, v);
            const av = @abs(v);
            if (av > core_max) core_max = av;
        }
        var proj_sq: f64 = 0;
        var proj_max: f32 = 0;
        for (proj) |v| {
            proj_sq += @as(f64, v) * @as(f64, v);
            const av = @abs(v);
            if (av > proj_max) proj_max = av;
        }

        log.info("L{d} {s} {s}: rms={d:.4} max={d:.4} h0={d:.6} logit5={d:.4} core={d:.4}/{d:.4} proj={d:.4}/{d:.4}", .{
            layer,
            if (is_full_attn) "A" else "S",
            stage,
            rms,
            max_abs,
            hidden[0],
            logit5,
            @as(f32, @floatCast(@sqrt(core_sq / @as(f64, @floatFromInt(core_dim))))),
            core_max,
            @as(f32, @floatCast(@sqrt(proj_sq / @as(f64, @floatFromInt(hidden_dim))))),
            proj_max,
        });
        return;
    }

    log.info("L{d} {s} {s}: rms={d:.4} max={d:.4} h0={d:.6} logit5={d:.4}", .{
        layer,
        if (is_full_attn) "A" else "S",
        stage,
        rms,
        max_abs,
        hidden[0],
        logit5,
    });
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

pub fn generateWithMetrics(
    engine: *InferenceEngine,
    prompt_tokens: []const u32,
    max_tokens: u32,
    eos_id: u32,
    allocator: std.mem.Allocator,
) !GenerateResult {
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
    const prefill_tps = if (prompt_tokens.len > 0 and prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_tokens.len)) * 1_000_000_000.0 / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;

    // Sample first output token from prefill logits
    var eos_at_first_position = false;
    if (prompt_tokens.len > 0 and max_tokens > 0) {
        const first_token = engine.sampleGreedy();
        if (first_token == eos_id) {
            eos_at_first_position = true;
            return .{
                .output_tokens = try output.toOwnedSlice(allocator),
                .metrics = .{
                    .prefill_tokens = prompt_tokens.len,
                    .prefill_ns = prefill_ns,
                    .prefill_tps = prefill_tps,
                    .generated_tokens = 0,
                    .decode_ns = 0,
                    .decode_tps = 0.0,
                    .ms_per_token = 0.0,
                    .eos_at_first_position = true,
                },
            };
        }
        try output.append(allocator, first_token);
    }

    // Decode loop
    const decode_start = std.time.nanoTimestamp();
    var tokens_generated: u32 = @intCast(output.items.len);
    while (tokens_generated < max_tokens and output.items.len > 0) {
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
    const decode_tps = if (tokens_generated > 0 and decode_ns > 0)
        @as(f64, @floatFromInt(tokens_generated)) * 1_000_000_000.0 / @as(f64, @floatFromInt(decode_ns))
    else
        0.0;
    const ms_per_tok = if (tokens_generated > 0)
        @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(tokens_generated))
    else
        0.0;

    return .{
        .output_tokens = try output.toOwnedSlice(allocator),
        .metrics = .{
            .prefill_tokens = prompt_tokens.len,
            .prefill_ns = prefill_ns,
            .prefill_tps = prefill_tps,
            .generated_tokens = tokens_generated,
            .decode_ns = decode_ns,
            .decode_tps = decode_tps,
            .ms_per_token = ms_per_tok,
            .eos_at_first_position = eos_at_first_position,
        },
    };
}

pub fn generate(
    engine: *InferenceEngine,
    prompt_tokens: []const u32,
    max_tokens: u32,
    eos_id: u32,
    allocator: std.mem.Allocator,
) ![]u32 {
    const result = try generateWithMetrics(engine, prompt_tokens, max_tokens, eos_id, allocator);
    if (result.metrics.prefill_tokens > 0) {
        log.info("Prefill: {d} tokens in {d:.1} ms ({d:.1} tok/s)", .{
            result.metrics.prefill_tokens,
            @as(f64, @floatFromInt(result.metrics.prefill_ns)) / 1_000_000.0,
            result.metrics.prefill_tps,
        });
    }
    if (result.metrics.eos_at_first_position) {
        log.info("Generated 0 tokens (EOS at first position)", .{});
    } else if (result.metrics.generated_tokens > 0) {
        log.info("Generated {d} tokens in {d:.1} ms — {d:.2} tok/s ({d:.1} ms/tok)", .{
            result.metrics.generated_tokens,
            @as(f64, @floatFromInt(result.metrics.decode_ns)) / 1_000_000.0,
            result.metrics.decode_tps,
            result.metrics.ms_per_token,
        });
    }
    engine.logRequestProfileSummary("request", prompt_tokens.len, result.metrics.generated_tokens);
    return result.output_tokens;
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

fn refL2Normalize(v: []f32) void {
    var sum_sq: f32 = 0;
    for (v) |x| sum_sq += x * x;
    const denom = @sqrt(sum_sq) + 1e-12;
    for (v) |*x| x.* /= denom;
}

fn refRunSsmDeltaNet(
    conv_out: []const f32,
    alpha: []const f32,
    dt_bias: []const f32,
    beta: []const f32,
    ssm_a: []const f32,
    state: []f32,
    output: []f32,
    dt_rank: usize,
    head_v_dim: usize,
    d_state: usize,
    n_group: usize,
) void {
    const qk_dim = d_state * n_group;
    const k_len = @min(head_v_dim, d_state);

    var q_buf: [128]f32 = [_]f32{0} ** 128;
    var k_buf: [128]f32 = [_]f32{0} ** 128;

    for (0..dt_rank) |h| {
        @memset(q_buf[0..head_v_dim], 0);
        @memset(k_buf[0..head_v_dim], 0);

        const k_hi = if (n_group == dt_rank) h else h % n_group;
        const q_offset = k_hi * d_state;
        const k_offset = qk_dim + k_hi * d_state;
        const v_offset = 2 * qk_dim + h * head_v_dim;

        @memcpy(q_buf[0..k_len], conv_out[q_offset .. q_offset + k_len]);
        @memcpy(k_buf[0..k_len], conv_out[k_offset .. k_offset + k_len]);
        refL2Normalize(q_buf[0..k_len]);
        refL2Normalize(k_buf[0..k_len]);
        const q_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_state)));
        for (0..k_len) |i| q_buf[i] *= q_scale;
        const gate = @exp(@log(1.0 + @exp(alpha[h] + dt_bias[h])) * ssm_a[h]);
        const beta_val = 1.0 / (1.0 + @exp(-beta[h]));
        const s_base = h * head_v_dim * head_v_dim;
        const v_head = conv_out[v_offset .. v_offset + head_v_dim];

        for (0..head_v_dim * head_v_dim) |i| state[s_base + i] *= gate;

        for (0..head_v_dim) |row| {
            const row_base = s_base + row * head_v_dim;
            var sk: f32 = 0;
            for (0..k_len) |col| sk += state[row_base + col] * k_buf[col];
            const d_val = beta_val * (v_head[row] - sk);
            for (0..k_len) |col| state[row_base + col] += k_buf[col] * d_val;

            var o_val: f32 = 0;
            for (0..k_len) |col| o_val += state[row_base + col] * q_buf[col];
            output[h * head_v_dim + row] = o_val;
        }
    }
}

fn refRunSsmConv1d(current_input: []const f32, kernel: []const f32, state: []f32, output: []f32, conv_channels: usize, d_conv: usize) void {
    const d_conv_1 = d_conv - 1;
    for (0..conv_channels) |ch| {
        var sum: f32 = 0;
        for (0..d_conv) |ki| {
            const kw = kernel[ch * d_conv + ki];
            const sv = if (ki < d_conv_1) state[ki * conv_channels + ch] else current_input[ch];
            sum += kw * sv;
        }
        output[ch] = sum / (1.0 + @exp(-sum));
    }
    if (d_conv_1 > 1) {
        const shift = (d_conv_1 - 1) * conv_channels;
        std.mem.copyForwards(f32, state[0..shift], state[conv_channels .. shift + conv_channels]);
    }
    @memcpy(state[(d_conv_1 - 1) * conv_channels ..][0..conv_channels], current_input);
}

fn refRunSsmGatedNorm(
    delta_net_output: []const f32,
    z_gate: []const f32,
    norm_weight: []const f32,
    output: []f32,
    dt_rank: usize,
    head_v_dim: usize,
    d_state: usize,
    norm_per_head: bool,
) void {
    for (0..dt_rank) |h| {
        const base = h * head_v_dim;
        const delta_head = delta_net_output[base .. base + head_v_dim];
        const gate_head = z_gate[base .. base + head_v_dim];
        var sum_sq: f32 = 0;
        for (delta_head) |v| sum_sq += v * v;
        const rms_inv = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(head_v_dim)) + 1e-6);
        for (0..head_v_dim) |i| {
            const norm_idx = if (norm_per_head) base + i else i % d_state;
            const nv = delta_head[i] * rms_inv * norm_weight[norm_idx];
            const z = gate_head[i];
            output[base + i] = nv * (z / (1.0 + @exp(-z)));
        }
    }
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

test "ssm_delta_net shader matches CPU reference" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "ssm_delta_net");
    defer metal_pipeline.freePipeline(&pipe);

    const dt_rank: u32 = 2;
    const head_v_dim: u32 = 4;
    const d_state: u32 = 2;
    const n_group: u32 = 1;
    const d_inner: u32 = dt_rank * head_v_dim;
    const qk_dim: u32 = d_state * n_group;
    const conv_len: u32 = 2 * qk_dim + d_inner;
    const state_len: u32 = dt_rank * head_v_dim * head_v_dim;

    var conv_buf = try metal_buffer.createBuffer(ctx, conv_len * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&conv_buf);
    var alpha_buf = try metal_buffer.createBuffer(ctx, dt_rank * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&alpha_buf);
    var dt_bias_buf = try metal_buffer.createBuffer(ctx, dt_rank * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&dt_bias_buf);
    var ssm_a_buf = try metal_buffer.createBuffer(ctx, dt_rank * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&ssm_a_buf);
    var beta_buf = try metal_buffer.createBuffer(ctx, dt_rank * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&beta_buf);
    var state_buf = try metal_buffer.createBuffer(ctx, state_len * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&state_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, d_inner * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    const conv_ptr: [*]f32 = @ptrCast(@alignCast(conv_buf.cpu_ptr.?));
    const alpha_ptr: [*]f32 = @ptrCast(@alignCast(alpha_buf.cpu_ptr.?));
    const dt_bias_ptr: [*]f32 = @ptrCast(@alignCast(dt_bias_buf.cpu_ptr.?));
    const ssm_a_ptr: [*]f32 = @ptrCast(@alignCast(ssm_a_buf.cpu_ptr.?));
    const beta_ptr: [*]f32 = @ptrCast(@alignCast(beta_buf.cpu_ptr.?));
    const state_ptr: [*]f32 = @ptrCast(@alignCast(state_buf.cpu_ptr.?));
    const output_ptr: [*]f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));

    const conv_init = [_]f32{ 3.0, 4.0, 4.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.5, -1.0, 2.0, -2.0 };
    @memcpy(conv_ptr[0..conv_len], conv_init[0..conv_len]);
    alpha_ptr[0] = 0.1;
    alpha_ptr[1] = -0.2;
    dt_bias_ptr[0] = 0.3;
    dt_bias_ptr[1] = -0.1;
    ssm_a_ptr[0] = 0.5;
    ssm_a_ptr[1] = -0.75;
    beta_ptr[0] = 0.25;
    beta_ptr[1] = -0.5;
    for (0..state_len) |i| {
        state_ptr[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) * 0.1;
    }
    @memset(output_ptr[0..d_inner], 0);

    var ref_state: [32]f32 = undefined;
    var ref_output: [8]f32 = [_]f32{0} ** 8;
    @memcpy(ref_state[0..state_len], state_ptr[0..state_len]);
    refRunSsmDeltaNet(
        conv_ptr[0..conv_len],
        alpha_ptr[0..dt_rank],
        dt_bias_ptr[0..dt_rank],
        beta_ptr[0..dt_rank],
        ssm_a_ptr[0..dt_rank],
        ref_state[0..state_len],
        ref_output[0..d_inner],
        dt_rank,
        head_v_dim,
        d_state,
        n_group,
    );

    const push = SsmDeltaNetPush{
        .d_inner = d_inner,
        .dt_rank = dt_rank,
        .head_v_dim = head_v_dim,
        .d_state = d_state,
        .n_group = n_group,
        .ssm_a_is_f16 = 0,
        .dt_bias_is_f16 = 0,
        .has_dt_bias = 1,
        .has_ssm_a = 1,
    };
    const bufs = [_]*const MetalBuffer{
        &conv_buf,
        &alpha_buf,
        &dt_bias_buf,
        &ssm_a_buf,
        &beta_buf,
        &state_buf,
        &output_buf,
    };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ dt_rank, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SsmDeltaNetPush), 0);
    cmd.commitAndWait();

    for (0..d_inner) |i| {
        try std.testing.expectApproxEqAbs(ref_output[i], output_ptr[i], 0.0005);
    }
    for (0..state_len) |i| {
        try std.testing.expectApproxEqAbs(ref_state[i], state_ptr[i], 0.0005);
    }
}

test "ssm_conv1d shader matches CPU reference" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "ssm_conv1d");
    defer metal_pipeline.freePipeline(&pipe);

    const conv_channels: usize = 5;
    const d_conv: usize = 4;
    const state_len: usize = (d_conv - 1) * conv_channels;
    const kernel_len: usize = conv_channels * d_conv;

    var kernel_buf = try metal_buffer.createBuffer(ctx, kernel_len * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&kernel_buf);
    var state_buf = try metal_buffer.createBuffer(ctx, state_len * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&state_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, conv_channels * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, conv_channels * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    const kernel_ptr: [*]f32 = @ptrCast(@alignCast(kernel_buf.cpu_ptr.?));
    const state_ptr: [*]f32 = @ptrCast(@alignCast(state_buf.cpu_ptr.?));
    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    const output_ptr: [*]f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));

    const kernel_init = [_]f32{
        0.25,  -0.50, 0.75,  1.00,
        -1.25, 0.50,  -0.25, 0.80,
        0.10,  0.20,  0.30,  0.40,
        -0.60, 0.90,  -1.10, 0.70,
        1.20,  -0.70, 0.50,  -0.30,
    };
    const state_init = [_]f32{
        -0.40, 0.25,  1.10, -0.75, 0.60,
        0.50,  -0.20, 0.35, 0.90,  -1.25,
        1.40,  -0.80, 0.15, -0.45, 0.95,
    };
    const input_init = [_]f32{ 0.80, -1.50, 0.45, 1.25, -0.65 };

    @memcpy(kernel_ptr[0..kernel_len], kernel_init[0..kernel_len]);
    @memcpy(state_ptr[0..state_len], state_init[0..state_len]);
    @memcpy(input_ptr[0..conv_channels], input_init[0..conv_channels]);
    @memset(output_ptr[0..conv_channels], 0);

    var ref_state: [state_len]f32 = undefined;
    var ref_output: [conv_channels]f32 = [_]f32{0} ** conv_channels;
    @memcpy(ref_state[0..state_len], state_ptr[0..state_len]);
    refRunSsmConv1d(
        input_ptr[0..conv_channels],
        kernel_ptr[0..kernel_len],
        ref_state[0..state_len],
        ref_output[0..conv_channels],
        conv_channels,
        d_conv,
    );

    var cmd = try metal_command.beginCommand(ctx);
    dispatchSsmConv1dWithPipe(
        &cmd,
        &pipe,
        &kernel_buf,
        &state_buf,
        &input_buf,
        &output_buf,
        @intCast(conv_channels),
        @intCast(d_conv),
        false,
    );
    cmd.commitAndWait();

    for (0..conv_channels) |i| {
        try std.testing.expectApproxEqAbs(ref_output[i], output_ptr[i], 0.0005);
    }
    for (0..state_len) |i| {
        try std.testing.expectApproxEqAbs(ref_state[i], state_ptr[i], 0.0005);
    }
}

test "ssm_gated_norm shader matches CPU reference with shared norm weights" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "ssm_gated_norm");
    defer metal_pipeline.freePipeline(&pipe);

    const dt_rank: usize = 2;
    const head_v_dim: usize = 4;
    const d_state: usize = 2;
    const d_inner: usize = dt_rank * head_v_dim;

    var delta_buf = try metal_buffer.createBuffer(ctx, d_inner * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&delta_buf);
    var gate_buf = try metal_buffer.createBuffer(ctx, d_inner * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&gate_buf);
    var norm_buf = try metal_buffer.createBuffer(ctx, d_inner * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&norm_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, d_inner * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    const delta_ptr: [*]f32 = @ptrCast(@alignCast(delta_buf.cpu_ptr.?));
    const gate_ptr: [*]f32 = @ptrCast(@alignCast(gate_buf.cpu_ptr.?));
    const norm_ptr: [*]f32 = @ptrCast(@alignCast(norm_buf.cpu_ptr.?));
    const output_ptr: [*]f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));

    const delta_init = [_]f32{ 1.0, -2.0, 3.0, -4.0, -0.5, 0.75, -1.25, 1.50 };
    const gate_init = [_]f32{ 0.20, -0.70, 1.10, -1.40, 0.50, -0.30, 0.90, -1.20 };
    const norm_init = [_]f32{ 1.25, 0.80, 1.25, 0.80, 1.25, 0.80, 1.25, 0.80 };

    @memcpy(delta_ptr[0..d_inner], delta_init[0..d_inner]);
    @memcpy(gate_ptr[0..d_inner], gate_init[0..d_inner]);
    @memcpy(norm_ptr[0..d_inner], norm_init[0..d_inner]);
    @memset(output_ptr[0..d_inner], 0);

    var ref_output: [d_inner]f32 = [_]f32{0} ** d_inner;
    refRunSsmGatedNorm(
        delta_ptr[0..d_inner],
        gate_ptr[0..d_inner],
        norm_ptr[0..d_inner],
        ref_output[0..d_inner],
        dt_rank,
        head_v_dim,
        d_state,
        false,
    );

    var cmd = try metal_command.beginCommand(ctx);
    dispatchSsmGatedNormWithPipe(
        &cmd,
        &pipe,
        &delta_buf,
        &norm_buf,
        &gate_buf,
        &output_buf,
        @intCast(d_inner),
        @intCast(dt_rank),
        @intCast(head_v_dim),
        @intCast(d_state),
        false,
    );
    cmd.commitAndWait();

    for (0..d_inner) |i| {
        try std.testing.expectApproxEqAbs(ref_output[i], output_ptr[i], 0.0005);
    }
}

test "ssm_gated_norm shader matches CPU reference with per-head norm weights" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "ssm_gated_norm");
    defer metal_pipeline.freePipeline(&pipe);

    const dt_rank: usize = 2;
    const head_v_dim: usize = 4;
    const d_state: usize = 2;
    const d_inner: usize = dt_rank * head_v_dim;

    var delta_buf = try metal_buffer.createBuffer(ctx, d_inner * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&delta_buf);
    var gate_buf = try metal_buffer.createBuffer(ctx, d_inner * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&gate_buf);
    var norm_buf = try metal_buffer.createBuffer(ctx, d_inner * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&norm_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, d_inner * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    const delta_ptr: [*]f32 = @ptrCast(@alignCast(delta_buf.cpu_ptr.?));
    const gate_ptr: [*]f32 = @ptrCast(@alignCast(gate_buf.cpu_ptr.?));
    const norm_ptr: [*]f32 = @ptrCast(@alignCast(norm_buf.cpu_ptr.?));
    const output_ptr: [*]f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));

    const delta_init = [_]f32{ 2.0, -1.0, 0.5, -0.25, -3.0, 1.5, -0.75, 0.125 };
    const gate_init = [_]f32{ 1.00, -0.25, 0.40, -0.80, 0.60, -1.10, 0.30, -0.45 };
    const norm_init = [_]f32{ 0.90, 1.10, 1.30, 0.70, 1.20, 0.85, 1.05, 0.95 };

    @memcpy(delta_ptr[0..d_inner], delta_init[0..d_inner]);
    @memcpy(gate_ptr[0..d_inner], gate_init[0..d_inner]);
    @memcpy(norm_ptr[0..d_inner], norm_init[0..d_inner]);
    @memset(output_ptr[0..d_inner], 0);

    var ref_output: [d_inner]f32 = [_]f32{0} ** d_inner;
    refRunSsmGatedNorm(
        delta_ptr[0..d_inner],
        gate_ptr[0..d_inner],
        norm_ptr[0..d_inner],
        ref_output[0..d_inner],
        dt_rank,
        head_v_dim,
        d_state,
        true,
    );

    var cmd = try metal_command.beginCommand(ctx);
    dispatchSsmGatedNormWithPipe(
        &cmd,
        &pipe,
        &delta_buf,
        &norm_buf,
        &gate_buf,
        &output_buf,
        @intCast(d_inner),
        @intCast(dt_rank),
        @intCast(head_v_dim),
        @intCast(d_state),
        true,
    );
    cmd.commitAndWait();

    for (0..d_inner) |i| {
        try std.testing.expectApproxEqAbs(ref_output[i], output_ptr[i], 0.0005);
    }
}

test "ssm_delta_net shader matches CPU reference at realistic head width" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "ssm_delta_net");
    defer metal_pipeline.freePipeline(&pipe);

    const dt_rank: usize = 4;
    const head_v_dim: usize = 128;
    const d_state: usize = 16;
    const n_group: usize = 2;
    const d_inner: usize = dt_rank * head_v_dim;
    const qk_dim: usize = d_state * n_group;
    const conv_len: usize = 2 * qk_dim + d_inner;
    const state_len: usize = dt_rank * head_v_dim * head_v_dim;

    var conv_buf = try metal_buffer.createBuffer(ctx, conv_len * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&conv_buf);
    var alpha_buf = try metal_buffer.createBuffer(ctx, dt_rank * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&alpha_buf);
    var dt_bias_buf = try metal_buffer.createBuffer(ctx, dt_rank * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&dt_bias_buf);
    var ssm_a_buf = try metal_buffer.createBuffer(ctx, dt_rank * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&ssm_a_buf);
    var beta_buf = try metal_buffer.createBuffer(ctx, dt_rank * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&beta_buf);
    var state_buf = try metal_buffer.createBuffer(ctx, state_len * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&state_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, d_inner * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    const conv_ptr: [*]f32 = @ptrCast(@alignCast(conv_buf.cpu_ptr.?));
    const alpha_ptr: [*]f32 = @ptrCast(@alignCast(alpha_buf.cpu_ptr.?));
    const dt_bias_ptr: [*]f32 = @ptrCast(@alignCast(dt_bias_buf.cpu_ptr.?));
    const ssm_a_ptr: [*]f32 = @ptrCast(@alignCast(ssm_a_buf.cpu_ptr.?));
    const beta_ptr: [*]f32 = @ptrCast(@alignCast(beta_buf.cpu_ptr.?));
    const state_ptr: [*]f32 = @ptrCast(@alignCast(state_buf.cpu_ptr.?));
    const output_ptr: [*]f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));

    for (0..conv_len) |i| {
        const imod: i32 = @intCast(i % 17);
        const isub: i32 = imod - 8;
        conv_ptr[i] = @as(f32, @floatFromInt(isub)) * 0.07;
    }
    for (0..dt_rank) |i| {
        alpha_ptr[i] = 0.15 * @as(f32, @floatFromInt(@as(i32, @intCast(i)) - 1));
        dt_bias_ptr[i] = -0.1 + 0.08 * @as(f32, @floatFromInt(i));
        ssm_a_ptr[i] = -0.35 + 0.22 * @as(f32, @floatFromInt(i));
        beta_ptr[i] = 0.25 - 0.12 * @as(f32, @floatFromInt(i));
    }
    for (0..state_len) |i| {
        const imod: i32 = @intCast(i % 23);
        const isub: i32 = imod - 11;
        state_ptr[i] = @as(f32, @floatFromInt(isub)) * 0.03;
    }
    @memset(output_ptr[0..d_inner], 0);

    const allocator = std.testing.allocator;
    const ref_state = try allocator.alloc(f32, state_len);
    defer allocator.free(ref_state);
    const ref_output = try allocator.alloc(f32, d_inner);
    defer allocator.free(ref_output);

    @memcpy(ref_state[0..state_len], state_ptr[0..state_len]);
    @memset(ref_output[0..d_inner], 0);
    refRunSsmDeltaNet(
        conv_ptr[0..conv_len],
        alpha_ptr[0..dt_rank],
        dt_bias_ptr[0..dt_rank],
        beta_ptr[0..dt_rank],
        ssm_a_ptr[0..dt_rank],
        ref_state[0..state_len],
        ref_output[0..d_inner],
        dt_rank,
        head_v_dim,
        d_state,
        n_group,
    );

    const push = SsmDeltaNetPush{
        .d_inner = @intCast(d_inner),
        .dt_rank = @intCast(dt_rank),
        .head_v_dim = @intCast(head_v_dim),
        .d_state = @intCast(d_state),
        .n_group = @intCast(n_group),
        .ssm_a_is_f16 = 0,
        .dt_bias_is_f16 = 0,
        .has_dt_bias = 1,
        .has_ssm_a = 1,
    };
    const bufs = [_]*const MetalBuffer{
        &conv_buf,
        &alpha_buf,
        &dt_bias_buf,
        &ssm_a_buf,
        &beta_buf,
        &state_buf,
        &output_buf,
    };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ @intCast(dt_rank), 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SsmDeltaNetPush), 0);
    cmd.commitAndWait();

    for (0..d_inner) |i| {
        try std.testing.expectApproxEqAbs(ref_output[i], output_ptr[i], 0.001);
    }
    for (0..state_len) |i| {
        try std.testing.expectApproxEqAbs(ref_state[i], state_ptr[i], 0.001);
    }
}

test "ssm_delta_net shader matches CPU reference at model-like dimensions" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "ssm_delta_net");
    defer metal_pipeline.freePipeline(&pipe);

    const dt_rank: usize = 32;
    const head_v_dim: usize = 128;
    const d_state: usize = 128;
    const n_group: usize = 16;
    const d_inner: usize = dt_rank * head_v_dim;
    const qk_dim: usize = d_state * n_group;
    const conv_len: usize = 2 * qk_dim + d_inner;
    const state_len: usize = dt_rank * head_v_dim * head_v_dim;

    var conv_buf = try metal_buffer.createBuffer(ctx, conv_len * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&conv_buf);
    var alpha_buf = try metal_buffer.createBuffer(ctx, dt_rank * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&alpha_buf);
    var dt_bias_buf = try metal_buffer.createBuffer(ctx, dt_rank * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&dt_bias_buf);
    var ssm_a_buf = try metal_buffer.createBuffer(ctx, dt_rank * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&ssm_a_buf);
    var beta_buf = try metal_buffer.createBuffer(ctx, dt_rank * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&beta_buf);
    var state_buf = try metal_buffer.createBuffer(ctx, state_len * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&state_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, d_inner * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    const conv_ptr: [*]f32 = @ptrCast(@alignCast(conv_buf.cpu_ptr.?));
    const alpha_ptr: [*]f32 = @ptrCast(@alignCast(alpha_buf.cpu_ptr.?));
    const dt_bias_ptr: [*]f32 = @ptrCast(@alignCast(dt_bias_buf.cpu_ptr.?));
    const ssm_a_ptr: [*]f32 = @ptrCast(@alignCast(ssm_a_buf.cpu_ptr.?));
    const beta_ptr: [*]f32 = @ptrCast(@alignCast(beta_buf.cpu_ptr.?));
    const state_ptr: [*]f32 = @ptrCast(@alignCast(state_buf.cpu_ptr.?));
    const output_ptr: [*]f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));

    for (0..conv_len) |i| {
        const raw: i32 = @intCast((i * 17 + 3) % 41);
        conv_ptr[i] = @as(f32, @floatFromInt(raw - 20)) * 0.03125;
    }
    for (0..dt_rank) |i| {
        alpha_ptr[i] = -0.40 + 0.03 * @as(f32, @floatFromInt(i));
        dt_bias_ptr[i] = 0.20 - 0.015 * @as(f32, @floatFromInt(i));
        ssm_a_ptr[i] = -0.55 + 0.025 * @as(f32, @floatFromInt(i));
        beta_ptr[i] = 0.35 - 0.02 * @as(f32, @floatFromInt(i % 11));
    }
    for (0..state_len) |i| {
        const raw: i32 = @intCast((i * 5 + 11) % 37);
        state_ptr[i] = @as(f32, @floatFromInt(raw - 18)) * 0.015625;
    }
    @memset(output_ptr[0..d_inner], 0);

    const allocator = std.testing.allocator;
    const ref_state = try allocator.alloc(f32, state_len);
    defer allocator.free(ref_state);
    const ref_output = try allocator.alloc(f32, d_inner);
    defer allocator.free(ref_output);

    @memcpy(ref_state[0..state_len], state_ptr[0..state_len]);
    @memset(ref_output[0..d_inner], 0);
    refRunSsmDeltaNet(
        conv_ptr[0..conv_len],
        alpha_ptr[0..dt_rank],
        dt_bias_ptr[0..dt_rank],
        beta_ptr[0..dt_rank],
        ssm_a_ptr[0..dt_rank],
        ref_state[0..state_len],
        ref_output[0..d_inner],
        dt_rank,
        head_v_dim,
        d_state,
        n_group,
    );

    const push = SsmDeltaNetPush{
        .d_inner = @intCast(d_inner),
        .dt_rank = @intCast(dt_rank),
        .head_v_dim = @intCast(head_v_dim),
        .d_state = @intCast(d_state),
        .n_group = @intCast(n_group),
        .ssm_a_is_f16 = 0,
        .dt_bias_is_f16 = 0,
        .has_dt_bias = 1,
        .has_ssm_a = 1,
    };
    const bufs = [_]*const MetalBuffer{
        &conv_buf,
        &alpha_buf,
        &dt_bias_buf,
        &ssm_a_buf,
        &beta_buf,
        &state_buf,
        &output_buf,
    };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ @intCast(dt_rank), 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SsmDeltaNetPush), 0);
    cmd.commitAndWait();

    for (0..d_inner) |i| {
        try std.testing.expectApproxEqAbs(ref_output[i], output_ptr[i], 0.002);
    }
    for (0..state_len) |i| {
        try std.testing.expectApproxEqAbs(ref_state[i], state_ptr[i], 0.002);
    }
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

test "dmmv_q5k shader matches GGML contiguous-half CPU reference ordering" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q5k");
    defer metal_pipeline.freePipeline(&pipe);

    const M: u32 = 1;
    const K: u32 = 256;

    var weight_buf = try metal_buffer.createBuffer(ctx, 176);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0);
    @memset(input_buf.cpu_ptr.?[0..input_buf.size], 0);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    const d_bits = @as(u16, @bitCast(@as(f16, 1.0)));
    weight_buf.cpu_ptr.?[0] = @truncate(d_bits);
    weight_buf.cpu_ptr.?[1] = @truncate(d_bits >> 8);
    weight_buf.cpu_ptr.?[4] = 1;
    weight_buf.cpu_ptr.?[5] = 1;
    weight_buf.cpu_ptr.?[48] = 0x3A;

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    input_ptr[0] = 1.0;
    input_ptr[1] = 100.0;
    input_ptr[32] = 1000.0;

    var ref_row: [256]f32 = undefined;
    dequantRow(weight_buf.cpu_ptr.?[0..weight_buf.size], 0, K, .q5_k, &ref_row);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), ref_row[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), ref_row[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), ref_row[32], 0.001);

    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    var expected: f32 = 0;
    for (0..K) |i| {
        expected += ref_row[i] * input_ptr[i];
    }

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    try std.testing.expectApproxEqAbs(@as(f32, 3010.0), expected, 0.001);
    try std.testing.expectApproxEqAbs(expected, output_ptr[0], 0.001);
}

test "dmmv_q5k shader matches CPU reference across rows and blocks" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q5k");
    defer metal_pipeline.freePipeline(&pipe);

    const M: u32 = 2;
    const K: u32 = 512;
    const blocks_per_row: usize = K / 256;
    const row_bytes: usize = blocks_per_row * 176;

    var weight_buf = try metal_buffer.createBuffer(ctx, M * row_bytes);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0);
    @memset(input_buf.cpu_ptr.?[0..input_buf.size], 0);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    for (0..@as(usize, M)) |row| {
        for (0..blocks_per_row) |blk| {
            const base = row * row_bytes + blk * 176;
            const d_bits = @as(u16, @bitCast(@as(f16, @floatCast(0.25 * @as(f32, @floatFromInt(1 + row + blk))))));
            weight_buf.cpu_ptr.?[base] = @truncate(d_bits);
            weight_buf.cpu_ptr.?[base + 1] = @truncate(d_bits >> 8);
            weight_buf.cpu_ptr.?[base + 2] = 0;
            weight_buf.cpu_ptr.?[base + 3] = 0;
            for (0..12) |i| {
                weight_buf.cpu_ptr.?[base + 4 + i] = @intCast(1 + ((i + row + blk) % 3));
            }
            for (0..32) |i| {
                weight_buf.cpu_ptr.?[base + 16 + i] = @intCast((i + row + blk) & 0xFF);
            }
            for (0..128) |i| {
                const lo: u8 = @intCast((i + blk + row) & 0xF);
                const hi: u8 = @intCast((15 - ((i + blk * 3 + row) & 0xF)) & 0xF);
                weight_buf.cpu_ptr.?[base + 48 + i] = lo | (hi << 4);
            }
        }
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| {
        const raw: i32 = @intCast(i % 11);
        input_ptr[i] = @as(f32, @floatFromInt(raw - 5));
    }

    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    var ref_row: [512]f32 = undefined;
    var expected: [2]f32 = .{ 0, 0 };
    for (0..@as(usize, M)) |row| {
        dequantRow(weight_buf.cpu_ptr.?[0..weight_buf.size], @intCast(row), K, .q5_k, &ref_row);
        for (0..K) |i| expected[row] += ref_row[i] * input_ptr[i];
    }

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    try std.testing.expectApproxEqAbs(expected[0], output_ptr[0], 0.01);
    try std.testing.expectApproxEqAbs(expected[1], output_ptr[1], 0.01);
}

test "dmmv_q5k shader matches CPU reference with nonzero a_offset across many rows" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q5k");
    defer metal_pipeline.freePipeline(&pipe);

    const M: usize = 257;
    const K: usize = 512;
    const blocks_per_row: usize = K / 256;
    const row_bytes: usize = blocks_per_row * 176;
    const matrix_bytes: usize = M * row_bytes;
    const a_offset: usize = matrix_bytes;

    var weight_buf = try metal_buffer.createBuffer(ctx, a_offset + matrix_bytes);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0xCD);
    @memset(input_buf.cpu_ptr.?[0..input_buf.size], 0);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    for (0..M) |row| {
        for (0..blocks_per_row) |blk| {
            const base = a_offset + row * row_bytes + blk * 176;
            const d = @as(f16, @floatCast(0.03125 * @as(f32, @floatFromInt(1 + (row % 5) + blk))));
            const dmin = @as(f16, @floatCast(0.015625 * @as(f32, @floatFromInt(1 + ((row + blk) % 7)))));
            const d_bits = @as(u16, @bitCast(d));
            const dmin_bits = @as(u16, @bitCast(dmin));
            weight_buf.cpu_ptr.?[base] = @truncate(d_bits);
            weight_buf.cpu_ptr.?[base + 1] = @truncate(d_bits >> 8);
            weight_buf.cpu_ptr.?[base + 2] = @truncate(dmin_bits);
            weight_buf.cpu_ptr.?[base + 3] = @truncate(dmin_bits >> 8);
            for (0..12) |i| {
                weight_buf.cpu_ptr.?[base + 4 + i] = @intCast((row * 9 + blk * 5 + i * 3) & 0xFF);
            }
            for (0..32) |i| {
                weight_buf.cpu_ptr.?[base + 16 + i] = @intCast((row * 7 + blk * 11 + i * 5) & 0xFF);
            }
            for (0..128) |i| {
                weight_buf.cpu_ptr.?[base + 48 + i] = @intCast((row * 13 + blk * 17 + i * 7) & 0xFF);
            }
        }
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| {
        const raw: i32 = @intCast((i * 19 + 5) % 21);
        input_ptr[i] = 0.125 * @as(f32, @floatFromInt(raw - 10));
    }

    const push = DmmvPush{
        .M = @intCast(M),
        .K = @intCast(K),
        .a_offset = @intCast(a_offset),
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ @intCast((M + 63) / 64), 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    const allocator = std.testing.allocator;
    const ref_row = try allocator.alloc(f32, K);
    defer allocator.free(ref_row);
    const expected = try allocator.alloc(f32, M);
    defer allocator.free(expected);
    @memset(expected[0..M], 0);

    const matrix_raw = weight_buf.cpu_ptr.?[a_offset .. a_offset + matrix_bytes];
    for (0..M) |row| {
        dequantRow(matrix_raw, @intCast(row), @intCast(K), .q5_k, ref_row);
        for (0..K) |i| {
            expected[row] += ref_row[i] * input_ptr[i];
        }
    }

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..M) |row| {
        try std.testing.expectApproxEqAbs(expected[row], output_ptr[row], 0.05);
    }
}

test "dmmv_q6k shader matches CPU reference with nonzero a_offset across many rows" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q6k");
    defer metal_pipeline.freePipeline(&pipe);

    const M: usize = 257;
    const K: usize = 512;
    const blocks_per_row: usize = K / 256;
    const row_bytes: usize = blocks_per_row * 210;
    const matrix_bytes: usize = M * row_bytes;
    const a_offset: usize = matrix_bytes;

    var weight_buf = try metal_buffer.createBuffer(ctx, a_offset + matrix_bytes);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0xCD);
    @memset(input_buf.cpu_ptr.?[0..input_buf.size], 0);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    for (0..M) |row| {
        for (0..blocks_per_row) |blk| {
            const base = a_offset + row * row_bytes + blk * 210;
            const d = @as(f16, @floatCast(0.03125 * @as(f32, @floatFromInt(1 + (row % 5) + blk))));
            const d_bits = @as(u16, @bitCast(d));
            weight_buf.cpu_ptr.?[base + 208] = @truncate(d_bits);
            weight_buf.cpu_ptr.?[base + 209] = @truncate(d_bits >> 8);
            for (0..192) |i| {
                weight_buf.cpu_ptr.?[base + i] = @intCast((row * 23 + blk * 17 + i * 7) & 0xFF);
            }
            for (0..16) |i| {
                weight_buf.cpu_ptr.?[base + 192 + i] = @intCast((row * 29 + blk * 19 + i * 5) & 0xFF);
            }
        }
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| {
        const raw: i32 = @intCast((i * 17 + 9) % 25);
        input_ptr[i] = 0.125 * @as(f32, @floatFromInt(raw - 12));
    }

    const push = DmmvPush{
        .M = @intCast(M),
        .K = @intCast(K),
        .a_offset = @intCast(a_offset),
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ @intCast((M + 63) / 64), 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    const allocator = std.testing.allocator;
    const ref_row = try allocator.alloc(f32, K);
    defer allocator.free(ref_row);
    const expected = try allocator.alloc(f32, M);
    defer allocator.free(expected);
    @memset(expected[0..M], 0);

    const matrix_raw = weight_buf.cpu_ptr.?[a_offset .. a_offset + matrix_bytes];
    for (0..M) |row| {
        dequantRow(matrix_raw, @intCast(row), @intCast(K), .q6_k, ref_row);
        for (0..K) |i| {
            expected[row] += ref_row[i] * input_ptr[i];
        }
    }

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..M) |row| {
        try std.testing.expectApproxEqAbs(expected[row], output_ptr[row], 0.05);
    }
}

test "dmmv_q4k_moe_k2048 shader matches CPU reference across selected experts" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe_k2048");
    defer metal_pipeline.freePipeline(&pipe);

    const M: usize = 65;
    const K: usize = 512;
    const n_used: usize = 2;
    const n_experts: usize = 3;
    const blocks_per_row: usize = K / 256;
    const row_bytes: usize = blocks_per_row * 144;
    const expert_stride: usize = M * row_bytes;

    var weight_buf = try metal_buffer.createBuffer(ctx, n_experts * expert_stride);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, n_used * K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, n_used * M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);
    var routing_buf = try metal_buffer.createBuffer(ctx, n_used * 2 * @sizeOf(u32));
    defer metal_buffer.freeBuffer(&routing_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0);
    @memset(input_buf.cpu_ptr.?[0..input_buf.size], 0);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);
    @memset(routing_buf.cpu_ptr.?[0..routing_buf.size], 0);

    for (0..n_experts) |expert| {
        for (0..M) |row| {
            for (0..blocks_per_row) |blk| {
                const base = expert * expert_stride + row * row_bytes + blk * 144;
                const d = @as(f16, @floatCast(0.03125 * @as(f32, @floatFromInt(1 + expert + (row % 5) + blk))));
                const dmin = @as(f16, @floatCast(0.015625 * @as(f32, @floatFromInt(1 + ((expert + row + blk) % 7)))));
                const d_bits = @as(u16, @bitCast(d));
                const dmin_bits = @as(u16, @bitCast(dmin));
                weight_buf.cpu_ptr.?[base] = @truncate(d_bits);
                weight_buf.cpu_ptr.?[base + 1] = @truncate(d_bits >> 8);
                weight_buf.cpu_ptr.?[base + 2] = @truncate(dmin_bits);
                weight_buf.cpu_ptr.?[base + 3] = @truncate(dmin_bits >> 8);
                for (0..12) |i| {
                    weight_buf.cpu_ptr.?[base + 4 + i] = @intCast((expert * 31 + row * 9 + blk * 5 + i * 3) & 0xFF);
                }
                for (0..128) |i| {
                    const lo: u8 = @intCast((expert * 19 + row * 13 + blk * 17 + i * 7) & 0x0F);
                    const hi: u8 = @intCast((expert * 11 + row * 5 + blk * 3 + i * 9) & 0x0F);
                    weight_buf.cpu_ptr.?[base + 16 + i] = lo | (hi << 4);
                }
            }
        }
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..n_used) |slot| {
        for (0..K) |i| {
            const raw: i32 = @intCast((slot * 23 + i * 19 + 5) % 21);
            input_ptr[slot * K + i] = 0.125 * @as(f32, @floatFromInt(raw - 10));
        }
    }

    const routing_ptr: [*]u32 = @ptrCast(@alignCast(routing_buf.cpu_ptr.?));
    routing_ptr[0] = 2;
    routing_ptr[1] = 0;

    const push = MoeDmmvPush{
        .M = @intCast(M),
        .K = @intCast(K),
        .a_offset = 0,
        .expert_stride = @intCast(expert_stride),
        .x_expert_stride = @intCast(K),
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf, &routing_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ @intCast((M + 15) / 16), @intCast(n_used), 1 }, .{ 512, 1, 1 }, &bufs, &push, @sizeOf(MoeDmmvPush), 1);
    cmd.commitAndWait();

    const allocator = std.testing.allocator;
    const ref_row = try allocator.alloc(f32, K);
    defer allocator.free(ref_row);

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..n_used) |slot| {
        const expert_id = routing_ptr[slot];
        const matrix_raw = weight_buf.cpu_ptr.?[@as(usize, expert_id) * expert_stride ..][0..expert_stride];
        const input_slice = input_ptr[slot * K .. (slot + 1) * K];
        for (0..M) |row| {
            dequantRow(matrix_raw, @intCast(row), @intCast(K), .q4_k, ref_row);
            var expected: f32 = 0.0;
            for (0..K) |i| {
                expected += ref_row[i] * input_slice[i];
            }
            try std.testing.expectApproxEqAbs(expected, output_ptr[slot * M + row], 0.05);
        }
    }
}

test "dmmv_q5k_moe shader matches CPU reference across selected experts" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q5k_moe");
    defer metal_pipeline.freePipeline(&pipe);

    const M: usize = 65;
    const K: usize = 512;
    const n_used: usize = 2;
    const n_experts: usize = 3;
    const blocks_per_row: usize = K / 256;
    const row_bytes: usize = blocks_per_row * 176;
    const expert_stride: usize = M * row_bytes;

    var weight_buf = try metal_buffer.createBuffer(ctx, n_experts * expert_stride);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, n_used * K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, n_used * M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);
    var routing_buf = try metal_buffer.createBuffer(ctx, n_used * 2 * @sizeOf(u32));
    defer metal_buffer.freeBuffer(&routing_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0);
    @memset(input_buf.cpu_ptr.?[0..input_buf.size], 0);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);
    @memset(routing_buf.cpu_ptr.?[0..routing_buf.size], 0);

    for (0..n_experts) |expert| {
        for (0..M) |row| {
            for (0..blocks_per_row) |blk| {
                const base = expert * expert_stride + row * row_bytes + blk * 176;
                const d = @as(f16, @floatCast(0.03125 * @as(f32, @floatFromInt(1 + expert + (row % 5) + blk))));
                const dmin = @as(f16, @floatCast(0.015625 * @as(f32, @floatFromInt(1 + ((expert + row + blk) % 7)))));
                const d_bits = @as(u16, @bitCast(d));
                const dmin_bits = @as(u16, @bitCast(dmin));
                weight_buf.cpu_ptr.?[base] = @truncate(d_bits);
                weight_buf.cpu_ptr.?[base + 1] = @truncate(d_bits >> 8);
                weight_buf.cpu_ptr.?[base + 2] = @truncate(dmin_bits);
                weight_buf.cpu_ptr.?[base + 3] = @truncate(dmin_bits >> 8);
                for (0..12) |i| {
                    weight_buf.cpu_ptr.?[base + 4 + i] = @intCast((expert * 31 + row * 9 + blk * 5 + i * 3) & 0xFF);
                }
                for (0..32) |i| {
                    weight_buf.cpu_ptr.?[base + 16 + i] = @intCast((expert * 17 + row * 7 + blk * 11 + i * 5) & 0xFF);
                }
                for (0..128) |i| {
                    weight_buf.cpu_ptr.?[base + 48 + i] = @intCast((expert * 19 + row * 13 + blk * 17 + i * 7) & 0xFF);
                }
            }
        }
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..n_used) |slot| {
        for (0..K) |i| {
            const raw: i32 = @intCast((slot * 23 + i * 19 + 5) % 21);
            input_ptr[slot * K + i] = 0.125 * @as(f32, @floatFromInt(raw - 10));
        }
    }

    const routing_ptr: [*]u32 = @ptrCast(@alignCast(routing_buf.cpu_ptr.?));
    routing_ptr[0] = 2;
    routing_ptr[1] = 0;

    const push = MoeDmmvPush{
        .M = @intCast(M),
        .K = @intCast(K),
        .a_offset = 0,
        .expert_stride = @intCast(expert_stride),
        .x_expert_stride = @intCast(K),
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf, &routing_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ @intCast((M + 7) / 8), @intCast(n_used), 1 }, .{ 256, 1, 1 }, &bufs, &push, @sizeOf(MoeDmmvPush), 1);
    cmd.commitAndWait();

    const allocator = std.testing.allocator;
    const ref_row = try allocator.alloc(f32, K);
    defer allocator.free(ref_row);

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..n_used) |slot| {
        const expert_id = routing_ptr[slot];
        const matrix_raw = weight_buf.cpu_ptr.?[@as(usize, expert_id) * expert_stride ..][0..expert_stride];
        const input_slice = input_ptr[slot * K .. (slot + 1) * K];
        for (0..M) |row| {
            dequantRow(matrix_raw, @intCast(row), @intCast(K), .q5_k, ref_row);
            var expected: f32 = 0.0;
            for (0..K) |i| {
                expected += ref_row[i] * input_slice[i];
            }
            try std.testing.expectApproxEqAbs(expected, output_ptr[slot * M + row], 0.05);
        }
    }
}

test "dmmv_q6k_moe shader matches CPU reference across selected experts" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q6k_moe");
    defer metal_pipeline.freePipeline(&pipe);

    const M: usize = 65;
    const K: usize = 512;
    const n_used: usize = 2;
    const n_experts: usize = 3;
    const blocks_per_row: usize = K / 256;
    const row_bytes: usize = blocks_per_row * 210;
    const expert_stride: usize = M * row_bytes;

    var weight_buf = try metal_buffer.createBuffer(ctx, n_experts * expert_stride);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, n_used * K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, n_used * M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);
    var routing_buf = try metal_buffer.createBuffer(ctx, n_used * 2 * @sizeOf(u32));
    defer metal_buffer.freeBuffer(&routing_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0);
    @memset(input_buf.cpu_ptr.?[0..input_buf.size], 0);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);
    @memset(routing_buf.cpu_ptr.?[0..routing_buf.size], 0);

    for (0..n_experts) |expert| {
        for (0..M) |row| {
            for (0..blocks_per_row) |blk| {
                const base = expert * expert_stride + row * row_bytes + blk * 210;
                const d = @as(f16, @floatCast(0.03125 * @as(f32, @floatFromInt(1 + expert + (row % 5) + blk))));
                const d_bits = @as(u16, @bitCast(d));
                weight_buf.cpu_ptr.?[base + 208] = @truncate(d_bits);
                weight_buf.cpu_ptr.?[base + 209] = @truncate(d_bits >> 8);
                for (0..192) |i| {
                    weight_buf.cpu_ptr.?[base + i] = @intCast((expert * 23 + row * 11 + blk * 17 + i * 7) & 0xFF);
                }
                for (0..16) |i| {
                    weight_buf.cpu_ptr.?[base + 192 + i] = @intCast((expert * 29 + row * 13 + blk * 19 + i * 5) & 0xFF);
                }
            }
        }
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..n_used) |slot| {
        for (0..K) |i| {
            const raw: i32 = @intCast((slot * 31 + i * 17 + 9) % 25);
            input_ptr[slot * K + i] = 0.125 * @as(f32, @floatFromInt(raw - 12));
        }
    }

    const routing_ptr: [*]u32 = @ptrCast(@alignCast(routing_buf.cpu_ptr.?));
    routing_ptr[0] = 1;
    routing_ptr[1] = 2;

    const push = MoeDmmvPush{
        .M = @intCast(M),
        .K = @intCast(K),
        .a_offset = 0,
        .expert_stride = @intCast(expert_stride),
        .x_expert_stride = @intCast(K),
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf, &routing_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ @intCast((M + 7) / 8), @intCast(n_used), 1 }, .{ 256, 1, 1 }, &bufs, &push, @sizeOf(MoeDmmvPush), 1);
    cmd.commitAndWait();

    const allocator = std.testing.allocator;
    const ref_row = try allocator.alloc(f32, K);
    defer allocator.free(ref_row);

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..n_used) |slot| {
        const expert_id = routing_ptr[slot];
        const matrix_raw = weight_buf.cpu_ptr.?[@as(usize, expert_id) * expert_stride ..][0..expert_stride];
        const input_slice = input_ptr[slot * K .. (slot + 1) * K];
        for (0..M) |row| {
            dequantRow(matrix_raw, @intCast(row), @intCast(K), .q6_k, ref_row);
            var expected: f32 = 0.0;
            for (0..K) |i| {
                expected += ref_row[i] * input_slice[i];
            }
            try std.testing.expectApproxEqAbs(expected, output_ptr[slot * M + row], 0.05);
        }
    }
}

test "dmmv_q8_0 shader matches CPU reference" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0");
    defer metal_pipeline.freePipeline(&pipe);

    const M: u32 = 1;
    const K: u32 = 32;

    var weight_buf = try metal_buffer.createBuffer(ctx, 34);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0);
    @memset(input_buf.cpu_ptr.?[0..input_buf.size], 0);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    const scale_bits = @as(u16, @bitCast(@as(f16, 0.5)));
    weight_buf.cpu_ptr.?[0] = @truncate(scale_bits);
    weight_buf.cpu_ptr.?[1] = @truncate(scale_bits >> 8);

    const qs: [32]i8 = .{
        -4,  3,  -2,  1,  -8,  7,  -6,  5,
        -12, 11, -10, 9,  -16, 15, -14, 13,
        -1,  2,  -3,  4,  -5,  6,  -7,  8,
        -9,  10, -11, 12, -13, 14, -15, 16,
    };
    for (qs, 0..) |q, i| {
        weight_buf.cpu_ptr.?[2 + i] = @bitCast(q);
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| {
        input_ptr[i] = @as(f32, @floatFromInt((i % 5) + 1));
    }

    var ref_row: [32]f32 = undefined;
    dequantRow(weight_buf.cpu_ptr.?[0..weight_buf.size], 0, K, .q8_0, &ref_row);

    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    var expected: f32 = 0;
    for (0..K) |i| expected += ref_row[i] * input_ptr[i];

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    try std.testing.expectApproxEqAbs(expected, output_ptr[0], 0.001);
}

test "dmmv_q8_0 shader matches CPU reference across rows and blocks" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0");
    defer metal_pipeline.freePipeline(&pipe);

    const M: u32 = 2;
    const K: u32 = 2048;
    const blocks_per_row: usize = K / 32;
    const row_bytes: usize = blocks_per_row * 34;

    var weight_buf = try metal_buffer.createBuffer(ctx, M * row_bytes);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0);
    @memset(input_buf.cpu_ptr.?[0..input_buf.size], 0);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    for (0..@as(usize, M)) |row| {
        for (0..blocks_per_row) |blk| {
            const base = row * row_bytes + blk * 34;
            const scale = @as(f16, @floatCast(0.125 * @as(f32, @floatFromInt(1 + (blk % 3) + row))));
            const scale_bits = @as(u16, @bitCast(scale));
            weight_buf.cpu_ptr.?[base] = @truncate(scale_bits);
            weight_buf.cpu_ptr.?[base + 1] = @truncate(scale_bits >> 8);
            for (0..32) |e| {
                const raw_q: i32 = @intCast((e + blk * 5 + row * 7) % 31);
                const q: i8 = @intCast(raw_q - 15);
                weight_buf.cpu_ptr.?[base + 2 + e] = @bitCast(q);
            }
        }
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| {
        const raw: i32 = @intCast(i % 9);
        input_ptr[i] = @as(f32, @floatFromInt(raw - 4));
    }

    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    var ref_row: [2048]f32 = undefined;
    var expected: [2]f32 = .{ 0, 0 };
    for (0..@as(usize, M)) |row| {
        dequantRow(weight_buf.cpu_ptr.?[0..weight_buf.size], @intCast(row), K, .q8_0, &ref_row);
        for (0..K) |i| expected[row] += ref_row[i] * input_ptr[i];
    }

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    try std.testing.expectApproxEqAbs(expected[0], output_ptr[0], 0.01);
    try std.testing.expectApproxEqAbs(expected[1], output_ptr[1], 0.01);
}

test "dmmv_q8_0 shader matches CPU reference across many workgroups" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0");
    defer metal_pipeline.freePipeline(&pipe);

    const M: usize = 258;
    const K: usize = 2048;
    const blocks_per_row: usize = K / 32;
    const row_bytes: usize = blocks_per_row * 34;

    var weight_buf = try metal_buffer.createBuffer(ctx, M * row_bytes);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0);
    @memset(input_buf.cpu_ptr.?[0..input_buf.size], 0);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    for (0..M) |row| {
        for (0..blocks_per_row) |blk| {
            const base = row * row_bytes + blk * 34;
            const scale_mag = 0.03125 * @as(f32, @floatFromInt(1 + (row % 5) + (blk % 7)));
            const scale = @as(f16, @floatCast(scale_mag));
            const scale_bits = @as(u16, @bitCast(scale));
            weight_buf.cpu_ptr.?[base] = @truncate(scale_bits);
            weight_buf.cpu_ptr.?[base + 1] = @truncate(scale_bits >> 8);
            for (0..32) |e| {
                const raw_q: i32 = @intCast((row * 11 + blk * 7 + e * 5) % 63);
                const q: i8 = @intCast(raw_q - 31);
                weight_buf.cpu_ptr.?[base + 2 + e] = @bitCast(q);
            }
        }
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| {
        const raw: i32 = @intCast((i * 13 + 7) % 29);
        input_ptr[i] = 0.125 * @as(f32, @floatFromInt(raw - 14));
    }

    const push = DmmvPush{
        .M = @intCast(M),
        .K = @intCast(K),
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ @intCast((M + 1) / 2), 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    const allocator = std.testing.allocator;
    const ref_row = try allocator.alloc(f32, K);
    defer allocator.free(ref_row);
    const expected = try allocator.alloc(f32, M);
    defer allocator.free(expected);
    @memset(expected[0..M], 0);

    for (0..M) |row| {
        dequantRow(weight_buf.cpu_ptr.?[0..weight_buf.size], @intCast(row), @intCast(K), .q8_0, ref_row);
        for (0..K) |i| {
            expected[row] += ref_row[i] * input_ptr[i];
        }
    }

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..M) |row| {
        try std.testing.expectApproxEqAbs(expected[row], output_ptr[row], 0.05);
    }
}

test "dmmv_q4k_k2048 shader matches CPU reference" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q4k_k2048");
    defer metal_pipeline.freePipeline(&pipe);

    const M: u32 = 1;
    const K: u32 = 256;

    var weight_buf = try metal_buffer.createBuffer(ctx, 144);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0);
    @memset(input_buf.cpu_ptr.?[0..input_buf.size], 0);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    const d_bits = @as(u16, @bitCast(@as(f16, 1.0)));
    weight_buf.cpu_ptr.?[0] = @truncate(d_bits);
    weight_buf.cpu_ptr.?[1] = @truncate(d_bits >> 8);
    weight_buf.cpu_ptr.?[4] = 1;
    weight_buf.cpu_ptr.?[5] = 2;
    weight_buf.cpu_ptr.?[16] = 0x53;
    weight_buf.cpu_ptr.?[17] = 0x97;

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    input_ptr[0] = 1.0;
    input_ptr[1] = 100.0;
    input_ptr[32] = 1000.0;
    input_ptr[33] = 10000.0;

    var ref_row: [256]f32 = undefined;
    dequantRow(weight_buf.cpu_ptr.?[0..weight_buf.size], 0, K, .q4_k, &ref_row);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), ref_row[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), ref_row[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), ref_row[32], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 18.0), ref_row[33], 0.001);

    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 1);
    cmd.commitAndWait();

    var expected: f32 = 0;
    for (0..K) |i| expected += ref_row[i] * input_ptr[i];

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    try std.testing.expectApproxEqAbs(expected, output_ptr[0], 0.001);
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
    var dmmv_q5k_moe_pipe = try loadShaderPipeline(ctx, "dmmv_q5k_moe");
    defer metal_pipeline.freePipeline(&dmmv_q5k_moe_pipe);
    var dmmv_q6k_moe_pipe = try loadShaderPipeline(ctx, "dmmv_q6k_moe");
    defer metal_pipeline.freePipeline(&dmmv_q6k_moe_pipe);

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
    try std.testing.expect(dmmv_q5k_moe_pipe.handle != null);
    try std.testing.expect(dmmv_q6k_moe_pipe.handle != null);
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

test "rms_norm shader normalizes each token slice with shared weights" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "rms_norm_mul");
    defer metal_pipeline.freePipeline(&pipe);

    const n: u32 = 8;
    const groups: u32 = 2;
    const eps: f32 = 1e-6;
    const total = n * groups;

    var input_buf = try metal_buffer.createBuffer(ctx, total * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, total * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);
    var weight_buf = try metal_buffer.createBuffer(ctx, n * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&weight_buf);

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    const weight_ptr: [*]f32 = @ptrCast(@alignCast(weight_buf.cpu_ptr.?));

    @memcpy(input_ptr[0..total], &[_]f32{
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    });
    @memcpy(weight_ptr[0..n], &[_]f32{ 1.0, 0.5, 1.5, 0.25, 1.0, 0.5, 1.5, 0.25 });
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    const push = RmsNormPush{ .n = n, .eps = eps };
    const bufs = [_]*const MetalBuffer{ &input_buf, &output_buf, &weight_buf };
    const simd_width = if (pipe.thread_execution_width > 0) pipe.thread_execution_width else @as(u32, 32);

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ groups, 1, 1 }, .{ simd_width, 1, 1 }, &bufs, &push, @sizeOf(RmsNormPush), 0);
    cmd.commitAndWait();

    var expected: [16]f32 = undefined;
    for (0..groups) |group| {
        const base = group * n;
        var sum_sq: f32 = 0.0;
        for (0..n) |i| {
            const v = input_ptr[base + i];
            sum_sq += v * v;
        }
        const rms_inv = 1.0 / @sqrt(sum_sq / @as(f32, @floatFromInt(n)) + eps);
        for (0..n) |i| {
            expected[base + i] = weight_ptr[i] * (input_ptr[base + i] * rms_inv);
        }
    }

    for (0..total) |i| {
        try std.testing.expectApproxEqAbs(expected[i], output_ptr[i], 0.001);
    }
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

test "flash_attn shader handles contiguous Metal KV cache fast path" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "flash_attn");
    defer metal_pipeline.freePipeline(&pipe);

    const head_dim: u32 = 4;
    const n_heads: u32 = 1;
    const n_kv_heads: u32 = 1;
    const seq_len: u32 = 2;

    var page_table_buf = try metal_buffer.createBuffer(ctx, seq_len * @sizeOf(u32));
    defer metal_buffer.freeBuffer(&page_table_buf);
    var q_buf = try metal_buffer.createBuffer(ctx, n_heads * head_dim * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&q_buf);
    var k_cache_buf = try metal_buffer.createBuffer(ctx, seq_len * n_kv_heads * head_dim * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&k_cache_buf);
    var v_cache_buf = try metal_buffer.createBuffer(ctx, seq_len * n_kv_heads * head_dim * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&v_cache_buf);
    var out_buf = try metal_buffer.createBuffer(ctx, n_heads * head_dim * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&out_buf);

    const page_table_ptr: [*]u32 = @ptrCast(@alignCast(page_table_buf.cpu_ptr.?));
    page_table_ptr[0] = 0;
    page_table_ptr[1] = 1;

    const q_ptr: [*]f32 = @ptrCast(@alignCast(q_buf.cpu_ptr.?));
    @memcpy(q_ptr[0 .. n_heads * head_dim], &[_]f32{ 1.0, 0.0, 0.0, 0.0 });

    const k_ptr: [*]f32 = @ptrCast(@alignCast(k_cache_buf.cpu_ptr.?));
    @memcpy(k_ptr[0 .. seq_len * n_kv_heads * head_dim], &[_]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
    });

    const v_ptr: [*]f32 = @ptrCast(@alignCast(v_cache_buf.cpu_ptr.?));
    @memcpy(v_ptr[0 .. seq_len * n_kv_heads * head_dim], &[_]f32{
        10.0, 20.0, 30.0, 40.0,
        50.0, 60.0, 70.0, 80.0,
    });

    @memset(out_buf.cpu_ptr.?[0..out_buf.size], 0);

    const push = FlashAttnPush{
        .head_dim = head_dim,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .seq_len = seq_len,
        .page_size = 0,
    };
    const bufs = [_]*const MetalBuffer{ &page_table_buf, &q_buf, &k_cache_buf, &v_cache_buf, &out_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ n_heads, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(FlashAttnPush), 0);
    cmd.commitAndWait();

    var expected: [4]f32 = undefined;
    cpuAttention(
        q_ptr,
        k_ptr,
        v_ptr,
        &expected,
        head_dim,
        n_heads,
        n_kv_heads,
        seq_len,
    );

    const out_ptr: [*]const f32 = @ptrCast(@alignCast(out_buf.cpu_ptr.?));
    for (0..head_dim) |i| {
        try std.testing.expectApproxEqAbs(expected[i], out_ptr[i], 0.001);
    }
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
