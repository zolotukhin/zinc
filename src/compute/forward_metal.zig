//! Metal inference engine — decode loop for Apple Silicon.
//! @section Inference Runtime
//! This is the Metal equivalent of forward.zig (Vulkan).
//! Uses MSL compute shaders dispatched via the Metal shim.
const std = @import("std");
const config_mod = @import("../model/config.zig");
const ModelConfig = config_mod.ModelConfig;
const gguf = @import("../model/gguf.zig");
const memory_plan = @import("../gpu/memory_plan.zig");
const metal_loader = @import("../model/loader_metal.zig");
const metal_device = @import("../metal/device.zig");
const metal_buffer = @import("../metal/buffer.zig");
const MetalBuffer = metal_buffer.MetalBuffer;
const metal_pipeline = @import("../metal/pipeline.zig");
const MetalPipeline = metal_pipeline.MetalPipeline;
const metal_command = @import("../metal/command.zig");
const MetalCommand = metal_command.MetalCommand;
/// Command-encoder mode re-export used by runtime init options.
pub const CommandEncoderMode = metal_command.CommandEncoderMode;
const shim = @import("../metal/c.zig").shim;

const log = std.log.scoped(.forward);
/// Upper bound on the Metal KV-cache allocation: we still honour the model's
/// architectural context length and the unified-memory budget, but we refuse
/// to allocate more tokens than this in one block to keep allocation latency
/// and staging buffers sane. Callers that already right-sized `cfg.context_length`
/// from the device budget (see `memory_plan.autoContextTokensForDeviceBudget`)
/// see this as a soft safety net rather than the primary limit.
pub const runtime_context_cap: u32 = 262144;

/// Runtime state for the decode loop.
pub const DecodeState = struct {
    position: u32,
    generated_tokens: std.ArrayList(u32),
    requested_context_tokens: u32,
    allocator: std.mem.Allocator,

    /// Initialize decode state for a fresh Metal generation request.
    pub fn init(allocator: std.mem.Allocator) DecodeState {
        return .{
            .position = 0,
            .generated_tokens = .{},
            .requested_context_tokens = 0,
            .allocator = allocator,
        };
    }

    /// Release the generated-token buffer owned by this decode state.
    pub fn deinit(self: *DecodeState) void {
        self.generated_tokens.deinit(self.allocator);
        self.* = undefined;
    }
};

/// Metrics from generateWithMetrics: prefill/decode token counts, timing, and throughput.
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

/// Output tokens and performance metrics from a generation run.
pub const GenerateResult = struct {
    output_tokens: []u32,
    metrics: GenerateMetrics,

    /// Free the generated token slice returned by `generateWithMetrics`.
    pub fn deinit(self: *GenerateResult, allocator: std.mem.Allocator) void {
        allocator.free(self.output_tokens);
        self.* = undefined;
    }
};

/// Token sampling parameters: temperature, top-k, top-p, and repetition penalty.
pub const SamplingParams = struct {
    temperature: f32 = 0.0,
    top_p: f32 = 1.0,
    repetition_penalty: f32 = 1.0,
    top_k: u32 = 64,

    /// Return whether sampling settings require CPU-visible logits instead of greedy argmax.
    pub fn requiresLogitsReadback(self: @This()) bool {
        return self.temperature > 0.0001 or self.top_p < 0.9999 or self.repetition_penalty > 1.0001;
    }
};

/// Options for `InferenceEngine.init`: profiling, debug validation, KV cache, and dispatch tuning.
pub const InitOptions = struct {
    profile_enabled: bool = false,
    debug_validation_enabled: bool = false,
    q8_tg_override: ?u32 = null,
    q8_dual_tg_override: ?u32 = null,
    kv_cache_q8_override: ?bool = null,
    private_decode_buffers_override: ?bool = null,
    command_encoder_mode: ?CommandEncoderMode = null,
};

fn tensorBytes(model: *const metal_loader.Model) u64 {
    var total: u64 = 0;
    for (model.gguf_file.tensors.items) |tensor_info| {
        total += tensor_info.sizeBytes();
    }
    return total;
}

fn memoryBudget(device: *const metal_device.MetalDevice) u64 {
    const working_set = device.recommendedMaxWorkingSetSize();
    if (working_set > 0) return working_set;
    return device.totalMemory();
}

fn tokenSeen(history: []const u32, token: u32) bool {
    for (history) |t| {
        if (t == token) return true;
    }
    return false;
}

fn adjustedLogit(logit: f32, token: u32, history: []const u32, repetition_penalty: f32) f32 {
    if (repetition_penalty <= 1.0001 or !tokenSeen(history, token)) return logit;
    if (logit >= 0) return logit / repetition_penalty;
    return logit * repetition_penalty;
}

fn softcapLogit(logit: f32, softcap: f32) f32 {
    if (!(softcap > 0)) return logit;
    return softcap * std.math.tanh(logit / softcap);
}

fn sampleFromLogits(logits: []const f32, history: []const u32, params: SamplingParams, random: std.Random, final_logit_softcapping: f32) u32 {
    if (logits.len == 0) return 0;
    if (params.temperature <= 0.0001) {
        var best_idx: u32 = 0;
        var best_val = adjustedLogit(softcapLogit(logits[0], final_logit_softcapping), 0, history, params.repetition_penalty);
        for (logits[1..], 1..) |raw, i| {
            const val = adjustedLogit(softcapLogit(raw, final_logit_softcapping), @intCast(i), history, params.repetition_penalty);
            if (val > best_val) {
                best_val = val;
                best_idx = @intCast(i);
            }
        }
        return best_idx;
    }

    const max_candidates = 128;
    const top_k: usize = @min(@max(params.top_k, 1), max_candidates);
    const temperature = @max(params.temperature, 0.0001);
    var candidate_ids: [max_candidates]u32 = undefined;
    var candidate_logits: [max_candidates]f32 = undefined;
    var candidate_count: usize = 0;

    for (logits, 0..) |raw_val, i| {
        if (!std.math.isFinite(raw_val)) continue;
        const val = adjustedLogit(softcapLogit(raw_val, final_logit_softcapping), @intCast(i), history, params.repetition_penalty);
        var insert_at = candidate_count;
        while (insert_at > 0 and val > candidate_logits[insert_at - 1]) : (insert_at -= 1) {}
        if (insert_at >= top_k) continue;
        if (candidate_count < top_k) candidate_count += 1;
        var j = candidate_count - 1;
        while (j > insert_at) : (j -= 1) {
            candidate_ids[j] = candidate_ids[j - 1];
            candidate_logits[j] = candidate_logits[j - 1];
        }
        candidate_ids[insert_at] = @intCast(i);
        candidate_logits[insert_at] = val;
    }

    if (candidate_count <= 1) return if (candidate_count == 1) candidate_ids[0] else 0;

    var weights: [max_candidates]f64 = undefined;
    const max_logit = @as(f64, candidate_logits[0]) / @as(f64, temperature);
    var total_weight: f64 = 0.0;
    for (0..candidate_count) |i| {
        const weight = @exp(@as(f64, candidate_logits[i]) / @as(f64, temperature) - max_logit);
        weights[i] = weight;
        total_weight += weight;
    }
    if (!(total_weight > 0.0) or !std.math.isFinite(total_weight)) return candidate_ids[0];

    const safe_top_p = std.math.clamp(params.top_p, 0.0, 1.0);
    var keep_count = candidate_count;
    if (safe_top_p < 0.9999) {
        var cumulative: f64 = 0.0;
        for (0..candidate_count) |i| {
            cumulative += weights[i] / total_weight;
            keep_count = i + 1;
            if (cumulative >= @as(f64, safe_top_p) and i > 0) break;
        }
    }

    var kept_weight: f64 = 0.0;
    for (0..keep_count) |i| kept_weight += weights[i];
    if (!(kept_weight > 0.0)) return candidate_ids[0];

    const target = random.float(f64) * kept_weight;
    var cumulative: f64 = 0.0;
    for (0..keep_count) |i| {
        cumulative += weights[i];
        if (target <= cumulative) return candidate_ids[i];
    }
    return candidate_ids[keep_count - 1];
}

fn readThreadgroupOverride(env_name: [:0]const u8, simd_width: u32, max_threads: u32) ?u32 {
    const raw = std.posix.getenv(env_name) orelse return null;
    if (simd_width == 0 or max_threads == 0) return null;

    const value = std.fmt.parseUnsigned(u32, raw, 10) catch return null;
    if (value == 0 or value > max_threads or value % simd_width != 0) return null;
    return value;
}

fn defaultQ8Threadgroup(chip: metal_device.GpuFamily, simd_width: u32, max_threads: u32) ?u32 {
    if (simd_width == 0 or max_threads < 512) return null;
    if (chip == .apple9 and simd_width == 32) return 512;
    return null;
}

fn defaultQ8DualThreadgroup(chip: metal_device.GpuFamily, simd_width: u32, max_threads: u32) ?u32 {
    if (simd_width == 0 or max_threads < 512) return null;
    if (chip == .apple9 and simd_width == 32) return 512;
    return null;
}

fn preferApple9Q8K2048Path(tensor: *const metal_loader.LoadedTensor, M: u32, K: u32) bool {
    if (K > 2048) return false;

    const name = tensor.info.name;
    if (std.mem.endsWith(u8, name, "ffn_gate_shexp.weight")) {
        return M <= 512 and K == 2048;
    }
    if (std.mem.endsWith(u8, name, "ffn_down_shexp.weight")) {
        return M <= 2048 and K <= 512;
    }
    return false;
}

fn preferApple9Q8WidePath(tensor: *const metal_loader.LoadedTensor, M: u32, K: u32) bool {
    const name = tensor.info.name;
    if (K <= 2048 and M >= 4096 and
        (std.mem.endsWith(u8, name, "attn_qkv.weight") or
            std.mem.endsWith(u8, name, "attn_gate.weight")))
    {
        return true;
    }
    if (K <= 4096 and M >= 2048 and std.mem.endsWith(u8, name, "ssm_out.weight")) {
        return true;
    }
    return false;
}

fn shouldUseGlobalQ8Override(arch: config_mod.Architecture, tensor_name: []const u8) bool {
    if (arch != .gemma) return true;

    return !(std.mem.endsWith(u8, tensor_name, "ffn_gate.weight") or
        std.mem.endsWith(u8, tensor_name, "ffn_up.weight") or
        std.mem.endsWith(u8, tensor_name, "ffn_down.weight") or
        std.mem.endsWith(u8, tensor_name, "ffn_gate_shexp.weight") or
        std.mem.endsWith(u8, tensor_name, "ffn_up_shexp.weight") or
        std.mem.endsWith(u8, tensor_name, "ffn_down_shexp.weight"));
}

fn shouldCpuQ8Fallback(arch: config_mod.Architecture, tensor_name: []const u8) bool {
    _ = arch;
    _ = tensor_name;
    return false;
}

fn shouldDebugAttentionValidation(cfg: ModelConfig, position: u32, layer_idx: usize) bool {
    return (cfg.architecture == .gemma and (position == 0 or position == 4 or position == 5) and (layer_idx == 0 or layer_idx == 5 or layer_idx == 17)) or
        (cfg.architecture != .gemma and position == 5 and (layer_idx == 7 or layer_idx == 31));
}

fn readBoolEnv(env_name: [:0]const u8) ?bool {
    const raw = std.posix.getenv(env_name) orelse return null;
    if (std.mem.eql(u8, raw, "1") or std.ascii.eqlIgnoreCase(raw, "true") or std.ascii.eqlIgnoreCase(raw, "yes")) return true;
    if (std.mem.eql(u8, raw, "0") or std.ascii.eqlIgnoreCase(raw, "false") or std.ascii.eqlIgnoreCase(raw, "no")) return false;
    return null;
}

const DmmvPathClass = enum(u8) {
    other,
    ssm,
    full_attn,
    moe_expert,
    shared_expert,
    router,
    lm_head,
    dense_ffn,
};

const Q8ShapeStat = struct {
    path: DmmvPathClass = .other,
    rows: u32 = 0,
    cols: u32 = 0,
    bytes: u64 = 0,
    calls: u32 = 0,
};

const BarrierClass = enum(u8) {
    embed,
    full_attn,
    ssm,
    router,
    gpu_routed_moe,
    fallback_moe,
    dense_ffn,
    final,
};

/// Per-request profiling counters for dispatch, barrier, and timing breakdown.
pub const RuntimeProfile = struct {
    decode_steps: u32 = 0,
    shared_cmd_steps: u32 = 0,
    command_buffers: u32 = 0,
    commit_waits: u32 = 0,
    dispatch_calls: u32 = 0,
    barrier_calls: u32 = 0,
    embed_barrier_calls: u32 = 0,
    full_attn_barrier_calls: u32 = 0,
    ssm_barrier_calls: u32 = 0,
    router_barrier_calls: u32 = 0,
    gpu_routed_moe_barrier_calls: u32 = 0,
    fallback_moe_barrier_calls: u32 = 0,
    dense_ffn_barrier_calls: u32 = 0,
    final_barrier_calls: u32 = 0,
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
    gpu_completion_wait_ns: u64 = 0,
    sample_ns: u64 = 0,
    total_step_ns: u64 = 0,
    debug_validation_ns: u64 = 0,
    dmmv_total_bytes: u64 = 0,
    dmmv_q4k_bytes: u64 = 0,
    dmmv_q5k_bytes: u64 = 0,
    dmmv_q6k_bytes: u64 = 0,
    dmmv_q8_0_bytes: u64 = 0,
    dmmv_f16_bytes: u64 = 0,
    dmmv_f32_bytes: u64 = 0,
    lm_head_bytes: u64 = 0,
    ssm_bytes: u64 = 0,
    full_attn_bytes: u64 = 0,
    router_bytes: u64 = 0,
    shared_expert_bytes: u64 = 0,
    dense_ffn_bytes: u64 = 0,
    moe_expert_bytes: u64 = 0,
    q8_shape_stats: [16]Q8ShapeStat = [_]Q8ShapeStat{.{}} ** 16,

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

fn profileBarrier(cmd: *MetalCommand, profile: ?*RuntimeProfile, class: BarrierClass) void {
    const encoded = cmd.barrier_enabled;
    cmd.barrier();
    if (!encoded) return;
    if (profile) |p| switch (class) {
        .embed => p.embed_barrier_calls += 1,
        .full_attn => p.full_attn_barrier_calls += 1,
        .ssm => p.ssm_barrier_calls += 1,
        .router => p.router_barrier_calls += 1,
        .gpu_routed_moe => p.gpu_routed_moe_barrier_calls += 1,
        .fallback_moe => p.fallback_moe_barrier_calls += 1,
        .dense_ffn => p.dense_ffn_barrier_calls += 1,
        .final => p.final_barrier_calls += 1,
    };
}

fn fullAttentionInterval(cfg: ModelConfig) u32 {
    return if (cfg.full_attn_interval > 0) cfg.full_attn_interval else 1;
}

fn isFullAttentionLayer(cfg: ModelConfig, layer_idx: usize) bool {
    return ((@as(u32, @intCast(layer_idx)) + 1) % fullAttentionInterval(cfg)) == 0;
}

/// Return the number of full-attention layers in the model.
pub fn attentionLayerCount(cfg: ModelConfig) u32 {
    const interval = fullAttentionInterval(cfg);
    return if (interval == 0) 0 else @divTrunc(cfg.n_layers, interval);
}

fn kvDim(config: ModelConfig) u32 {
    return config.n_kv_heads * config.head_dim;
}

/// Whether Q8 KV cache quantization should be enabled by default for this model.
pub fn defaultKvCacheQ8Enabled(config: ModelConfig, debug_validation_enabled: bool) bool {
    if (debug_validation_enabled) return false;
    // Disable Q8 KV cache for gpt-oss — the OAI SwiGLU activation is sensitive to
    // quantization noise in the KV cache, causing degenerate output.
    if (config.architecture == .gpt_oss) return false;
    // Gemma4 uses the ISWA attention path in llama.cpp, which applies extra
    // K/V rotation handling when the KV cache is quantized. Metal does not
    // implement that path yet, so keep Gemma4 on the unquantized cache for
    // correctness.
    if (config.architecture == .gemma and config.rope_freq_base_swa > 0) return false;
    const kv_dim = kvDim(config);
    return kv_dim > 0 and config.head_dim > 0 and kv_dim % 32 == 0 and config.head_dim % 32 == 0;
}

/// Bytes consumed per token in the KV cache (depends on Q8 quantization setting).
pub fn kvCacheBytesPerToken(config: ModelConfig, q8_enabled: bool) u64 {
    const kv_dim = @as(u64, kvDim(config));
    if (q8_enabled) {
        return @divTrunc(kv_dim, 32) * 34;
    }
    return kv_dim * @sizeOf(f32);
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

fn bytesToGiB(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / 1_073_741_824.0;
}

fn dmmvPathLabel(path: DmmvPathClass) []const u8 {
    return switch (path) {
        .other => "other",
        .ssm => "ssm",
        .full_attn => "attn",
        .moe_expert => "moe",
        .shared_expert => "shared",
        .router => "router",
        .lm_head => "lm-head",
        .dense_ffn => "dense",
    };
}

/// Push constants for DMMV dispatch (matches GLSL layout).
const DmmvPush = extern struct {
    M: u32, // rows
    K: u32, // cols
    a_offset: u32,
    x_offset: u32,
    y_offset: u32,
};

const DualQ8DmmvPush = extern struct {
    M0: u32,
    M1: u32,
    K: u32,
    a0_offset: u32,
    a1_offset: u32,
    x_offset: u32,
    y0_offset: u32,
    y1_offset: u32,
};

const CopyU32Push = extern struct {
    n_words: u32,
    src_offset_words: u32,
    dst_offset_words: u32,
};

const CopyF32Push = extern struct {
    n: u32,
};

const ZeroF32Push = extern struct {
    n: u32,
};

const ArgmaxPush = extern struct {
    n: u32,
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

/// Push constants for the gemm_q4k / gemm_q6k batched matmul kernels.
/// Layout mirrors GemmPush in src/shaders/metal/gemm_q4k.metal.
const GemmPush = extern struct {
    ne00: i32,
    ne02: i32,
    nb01: u64,
    nb02: u64,
    ne12: i32,
    _pad0: u32 = 0,
    nb10: u64,
    nb11: u64,
    nb12: u64,
    ne0: i32,
    ne1: i32,
    src0_off: u32,
};

/// Push constants for rope_batched.
const RopeBatchedPush = extern struct {
    stride: u32,
    rope_dim: u32,
    n_heads: u32,
    position_base: u32,
    freq_base_bits: u32,
    attn_scale_bits: u32,
};

/// Push constants for flash_attn_batched — mirrors BatchedFlashAttnPush in
/// src/shaders/metal/flash_attn_batched.metal.
const BatchedFlashAttnPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    kv_len: u32,
    n_queries: u32,
    kv_pos_offset: u32,
};

/// Push constants for flash_attn_batched_q8 — adds byte strides for the Q8
/// cache layout. Mirrors BatchedFlashAttnQ8Push in the shader header.
const BatchedFlashAttnQ8Push = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    kv_len: u32,
    n_queries: u32,
    kv_pos_offset: u32,
    kv_head_stride_bytes: u32,
    kv_token_stride_bytes: u32,
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

fn createMetalBufferForMode(ctx: ?*shim.MetalCtx, size: usize, use_private: bool) !MetalBuffer {
    return if (use_private)
        metal_buffer.createPrivateBuffer(ctx, size)
    else
        metal_buffer.createBuffer(ctx, size);
}

fn modelSupportsPrivateDecodeBuffers(model: *const metal_loader.Model, cfg: ModelConfig) bool {
    if (cfg.n_experts == 0 or cfg.n_experts_used != 8) return false;

    var saw_moe_experts = false;
    for (model.tensors.items) |tensor| {
        const name = tensor.info.name;
        if (!(std.mem.endsWith(u8, name, "ffn_gate_exps.weight") or
            std.mem.endsWith(u8, name, "ffn_up_exps.weight") or
            std.mem.endsWith(u8, name, "ffn_down_exps.weight")))
        {
            continue;
        }

        saw_moe_experts = true;
        switch (tensor.info.type_) {
            .q4_k, .q5_k, .q6_k => {},
            else => return false,
        }
    }

    return saw_moe_experts;
}

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

const SoftmaxTopkScaledPush = extern struct {
    n_experts: u32,
    k: u32,
    logit_scale_bits: u32,
};

/// Push constants for batched GPU softmax + top-k routing.
/// Outputs one packed routing row per token: [k expert ids][k f32 weights as u32].
const SoftmaxTopkBatchedPush = extern struct {
    n_experts: u32,
    k: u32,
    logits_stride: u32,
    output_stride: u32,
};

/// Push constants for packing batched MoE routes by expert.
/// `ids` stores token_idx * k + topk_slot for each routed token.
const MoeRoutePackPush = extern struct {
    n_tokens: u32,
    n_experts: u32,
    k: u32,
    routing_stride: u32,
    ids_stride: u32,
};

/// Push constants for GPU-weighted batched MoE accumulation.
const MoeWeightedAccPush = extern struct {
    n: u32,
    n_used: u32,
    src_stride: u32,
};

const MoeWeightedAccScaledPush = extern struct {
    n: u32,
    n_used: u32,
    src_stride: u32,
    scale_offset: u32,
};

/// Push constants for RMS norm dispatch (matches rms_norm_mul.metal: buffer(0)).
const RmsNormPush = extern struct {
    n: u32, // elements per group
    eps: f32, // epsilon
};

const RmsNormOffsetPush = extern struct {
    n: u32,
    eps: f32,
    weight_offset: u32,
};

/// Push constants for fused residual-add + RMS norm (matches residual_rms_norm.metal: buffer(0)).
/// Eliminates one barrier per layer vs separate scale_acc + rms_norm.
const ResidualRmsNormPush = extern struct {
    n: u32,
    eps: f32,
    scale: f32,
};

/// Push constants for fused MoE weighted acc + shared expert (matches moe_weighted_acc_shared.metal: buffer(3)).
/// Eliminates one barrier per layer vs separate moe_weighted_acc + sigmoid_scale_acc.
const MoeWeightedAccSharedPush = extern struct {
    n: u32,
    n_used: u32,
    src_stride: u32,
    has_gate: u32,
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
};

/// Push constants for native RoPE dispatch (matches rope_native.metal: buffer(0)).
const RopeNativePush = extern struct {
    stride: u32,
    rope_dim: u32,
    n_heads: u32,
    position: u32,
};

/// Push constants for flash attention dispatch (matches flash_attn.metal: buffer(0)).
const FlashAttnPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_len: u32,
    sliding_window_size: u32,
    page_size: u32,
    attn_scale_bits: u32,
    kv_head_stride_bytes: u32,
    kv_token_stride_bytes: u32,
};

/// Push constants for GPU KV-cache writes.
const KvCacheWritePush = extern struct {
    n: u32,
    dst_offset: u32,
    dst_offset_bytes: u32,
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
    attn_q_bias: ?*const metal_loader.LoadedTensor = null,
    attn_k_bias: ?*const metal_loader.LoadedTensor = null,
    attn_v_bias: ?*const metal_loader.LoadedTensor = null,
    attn_q_norm: ?*const metal_loader.LoadedTensor = null,
    attn_k_norm: ?*const metal_loader.LoadedTensor = null,
    attn_output: ?*const metal_loader.LoadedTensor = null,
    attn_output_bias: ?*const metal_loader.LoadedTensor = null,
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
    ffn_gate_inp_scale: ?*const metal_loader.LoadedTensor = null,
    ffn_gate_exps: ?*const metal_loader.LoadedTensor = null,
    ffn_up_exps: ?*const metal_loader.LoadedTensor = null,
    ffn_gate_up_exps: ?*const metal_loader.LoadedTensor = null,
    ffn_down_exps: ?*const metal_loader.LoadedTensor = null,
    ffn_down_exps_scale: ?*const metal_loader.LoadedTensor = null,
    ffn_gate_exps_bias: ?*const metal_loader.LoadedTensor = null,
    ffn_up_exps_bias: ?*const metal_loader.LoadedTensor = null,
    ffn_down_exps_bias: ?*const metal_loader.LoadedTensor = null,
    ffn_gate_inp_bias: ?*const metal_loader.LoadedTensor = null,
    pre_ffw_norm_2: ?*const metal_loader.LoadedTensor = null,
    post_ffw_norm_1: ?*const metal_loader.LoadedTensor = null,
    post_ffw_norm_2: ?*const metal_loader.LoadedTensor = null,
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

const FullAttnGateMode = struct {
    packed_q_gate: bool,
    separate_attn_gate: bool,
    apply_attn_gate: bool,
};

const LayerAttentionParams = struct {
    q_dim: u32,
    kv_dim: u32,
    head_dim: u32,
    n_kv_heads: u32,
    rope_dim: u32,
    rope_freq_base: f32,
    sliding_window_size: u32,
    use_rope_freq_factors: bool,
    proportional_rope: bool,
    kv_cache_head_stride_bytes: u32,
    kv_cache_bytes_per_token: u32,
    use_k_as_v: bool,
};

fn classifyFullAttnGate(q_rows: u32, q_dim: u32, has_attn_gate: bool) FullAttnGateMode {
    const packed_q_gate = q_rows >= q_dim * 2;
    const separate_attn_gate = !packed_q_gate and has_attn_gate;
    return .{
        .packed_q_gate = packed_q_gate,
        .separate_attn_gate = separate_attn_gate,
        .apply_attn_gate = packed_q_gate or separate_attn_gate,
    };
}

fn resolveLayerAttentionParams(cfg: ModelConfig, lt: LayerTensors, hidden_dim: u32, kv_cache_q8: bool) !LayerAttentionParams {
    const q_tensor = lt.attn_q orelse return error.MissingTensor;
    const k_tensor = lt.attn_k orelse return error.MissingTensor;
    const use_k_as_v = lt.attn_v == null and cfg.architecture == .gemma;
    if (lt.attn_v == null and !use_k_as_v) return error.MissingTensor;

    const kv_tensor = if (use_k_as_v) k_tensor else lt.attn_v.?;
    const kv_dim: u32 = @intCast(kv_tensor.info.numElements() / hidden_dim);
    const q_rows: u32 = @intCast(q_tensor.info.numElements() / hidden_dim);
    if (cfg.n_heads == 0 or q_rows == 0 or q_rows % cfg.n_heads != 0) return error.InvalidTensorShape;

    var head_dim = cfg.head_dim;
    if (lt.attn_q_norm) |qn| {
        head_dim = @intCast(qn.info.numElements());
    } else if (lt.attn_k_norm) |kn| {
        head_dim = @intCast(kn.info.numElements());
    } else if (head_dim == 0) {
        if (cfg.n_kv_heads > 0 and kv_dim % cfg.n_kv_heads == 0) {
            head_dim = kv_dim / cfg.n_kv_heads;
        } else {
            head_dim = q_rows / cfg.n_heads;
        }
    }
    if (head_dim == 0 or kv_dim == 0 or kv_dim % head_dim != 0) return error.InvalidTensorShape;

    const q_head_dim = q_rows / cfg.n_heads;
    const packed_q_gate = q_head_dim == head_dim * 2;
    const q_dim = if (packed_q_gate) q_rows / 2 else q_rows;
    if (q_dim == 0 or q_dim % cfg.n_heads != 0 or q_dim / cfg.n_heads != head_dim) return error.InvalidTensorShape;

    const n_kv_heads = kv_dim / head_dim;

    const proportional_rope = cfg.architecture == .gemma and use_k_as_v;
    const use_swa_rope = cfg.architecture == .gemma and cfg.rope_freq_base_swa > 0 and head_dim < cfg.head_dim;
    const rope_dim = if (proportional_rope)
        head_dim
    else if (cfg.rope_dim > 0)
        @min(cfg.rope_dim, head_dim)
    else
        head_dim;
    const rope_freq_base = if (use_swa_rope)
        cfg.rope_freq_base_swa
    else
        cfg.rope_freq_base;

    const kv_cache_head_stride_bytes: u32 = if (kv_cache_q8)
        @intCast(@divTrunc(head_dim, 32) * 34)
    else
        head_dim * @sizeOf(f32);
    const kv_cache_bytes_per_token: u32 = if (kv_cache_q8)
        @intCast(@divTrunc(kv_dim, 32) * 34)
    else
        kv_dim * @sizeOf(f32);

    return .{
        .q_dim = q_dim,
        .kv_dim = kv_dim,
        .head_dim = head_dim,
        .n_kv_heads = n_kv_heads,
        .rope_dim = rope_dim,
        .rope_freq_base = rope_freq_base,
        .sliding_window_size = if (use_swa_rope) cfg.sliding_window_size else 0,
        .use_rope_freq_factors = !use_swa_rope,
        .proportional_rope = proportional_rope,
        .kv_cache_head_stride_bytes = kv_cache_head_stride_bytes,
        .kv_cache_bytes_per_token = kv_cache_bytes_per_token,
        .use_k_as_v = use_k_as_v,
    };
}

const MoeGateUpLayout = struct {
    gate_tensor: *const metal_loader.LoadedTensor,
    up_tensor: *const metal_loader.LoadedTensor,
    gate_expert_stride: u32,
    up_expert_stride: u32,
    gate_base_offset: u32 = 0,
    up_base_offset: u32 = 0,

    fn gateOffset(self: @This(), expert_id: u32) u32 {
        return self.gate_base_offset + expert_id * self.gate_expert_stride;
    }

    fn upOffset(self: @This(), expert_id: u32) u32 {
        return self.up_base_offset + expert_id * self.up_expert_stride;
    }
};

fn usesGeglu(cfg: ModelConfig) bool {
    return cfg.architecture == .gemma;
}

fn hasExplicitGemmaMoeTensors(cfg: ModelConfig, lt: LayerTensors) bool {
    return cfg.architecture == .gemma and lt.ffn_gate_up_exps != null and
        (lt.ffn_gate_inp_scale != null or
            lt.pre_ffw_norm_2 != null or
            lt.post_ffw_norm_1 != null or
            lt.post_ffw_norm_2 != null or
            lt.ffn_down_exps_scale != null);
}

fn resolveMoeGateUpLayout(lt: LayerTensors, inter_dim: u32, hidden_dim: u32) !MoeGateUpLayout {
    if (lt.ffn_gate_exps) |gate_tensor| {
        const up_tensor = lt.ffn_up_exps orelse return error.MissingTensor;
        return .{
            .gate_tensor = gate_tensor,
            .up_tensor = up_tensor,
            .gate_expert_stride = expertSliceBytes(gate_tensor.info.type_, inter_dim, hidden_dim),
            .up_expert_stride = expertSliceBytes(up_tensor.info.type_, inter_dim, hidden_dim),
        };
    }
    if (lt.ffn_gate_up_exps) |gate_up_tensor| {
        const gate_half_bytes = expertSliceBytes(gate_up_tensor.info.type_, inter_dim, hidden_dim);
        return .{
            .gate_tensor = gate_up_tensor,
            .up_tensor = gate_up_tensor,
            .gate_expert_stride = expertSliceBytes(gate_up_tensor.info.type_, inter_dim * 2, hidden_dim),
            .up_expert_stride = expertSliceBytes(gate_up_tensor.info.type_, inter_dim * 2, hidden_dim),
            .up_base_offset = gate_half_bytes,
        };
    }
    return error.MissingTensor;
}

/// Three-state opt-in for the experimental batched prefill path:
///   - off:      `ZINC_BATCHED_PREFILL` unset / 0 — keep per-token `prefillBatch`.
///   - on:       `ZINC_BATCHED_PREFILL=1` — run only the batched path.
///   - validate: `ZINC_BATCHED_PREFILL=validate` — run the batched path, then
///               re-run the per-token path on a fresh state and compare last-
///               token logits. Logs the max abs diff and warns if it exceeds
///               1e-3. Doubles prefill time; only useful for correctness
///               checks against a real model.
const BatchedPrefillMode = enum { off, on, validate };

fn batchedPrefillMode() BatchedPrefillMode {
    const raw = std.posix.getenv("ZINC_BATCHED_PREFILL") orelse return .off;
    if (std.mem.eql(u8, raw, "1")) return .on;
    if (std.mem.eql(u8, raw, "validate")) return .validate;
    return .off;
}

/// Returns true when the model + engine state match the narrow slice that
/// `prefillBatched` currently knows how to run: LLaMA-style dense attention
/// + dense FFN, Q4_K/Q6_K weights, no Q/K norms, no biases, no attention
/// gate, no post-attn/post-ffn norms, no sliding window, no per-layer output
/// scale, no attention sinks, f32 KV cache, shared-mode decode buffers.
/// Every unsupported case falls back to the per-token path.
fn canUseBatchedPrefill(engine: *const InferenceEngine) bool {
    const cfg = engine.config;
    if (cfg.n_experts > 0) return false;
    if (cfg.ssm_d_inner > 0) return false;
    if (cfg.architecture == .gemma or cfg.architecture == .gpt_oss) return false;
    if (engine.private_decode_buffers) return false;
    // Both f32 and Q8 KV caches are supported — we dispatch the matching
    // flash_attn_batched / kv_cache_write variant below.
    if (fullAttentionInterval(cfg) != 1) return false;
    if (cfg.sliding_window_size != 0) return false;
    if (engine.attn_sink_values != null) return false;

    const supported = [_]GGMLType{ .q4_k, .q6_k };
    const isSupported = struct {
        fn f(t: GGMLType) bool {
            for (supported) |s| if (t == s) return true;
            return false;
        }
    }.f;

    if (!isSupported(engine.lm_head.info.type_)) return false;

    for (0..cfg.n_layers) |i| {
        if (engine.post_attn_norm_present[i]) return false;
        if (engine.post_ffn_norm_present[i]) return false;
        if (engine.layer_output_scales[i] != 1.0) return false;

        const lt = engine.layer_tensors[i];
        if (lt.attn_gate != null) return false;
        if (lt.attn_q_bias != null or lt.attn_k_bias != null or
            lt.attn_v_bias != null or lt.attn_output_bias != null) return false;

        const q = lt.attn_q orelse return false;
        const k = lt.attn_k orelse return false;
        const v = lt.attn_v orelse return false;
        const o = lt.attn_output orelse return false;
        const gate = lt.ffn_gate orelse return false;
        const up = lt.ffn_up orelse return false;
        const down = lt.ffn_down orelse return false;
        for ([_]*const metal_loader.LoadedTensor{ q, k, v, o, gate, up, down }) |t| {
            if (!isSupported(t.info.type_)) return false;
        }

        // Reject packed Q+gate (Qwen3Next): attn_q row count == 2*q_dim.
        const hidden_dim = cfg.hidden_dim;
        const q_rows: u32 = @intCast(q.info.numElements() / hidden_dim);
        if (q_rows >= cfg.n_heads * (cfg.head_dim) * 2) return false;
    }
    return true;
}

/// Scratch GPU buffers needed by `prefillBatched` for a batch of `n_tokens`.
/// All buffers are shared-mode so CPU code can dequantize embeddings directly
/// into `hidden` and read the last-token slice back out at the end.
const BatchedPrefillScratch = struct {
    n_tokens: u32,
    hidden: MetalBuffer,
    norm: MetalBuffer,
    q: MetalBuffer,
    k: MetalBuffer,
    v: MetalBuffer,
    attn_out: MetalBuffer,
    gate: MetalBuffer,
    up: MetalBuffer,
    swiglu: MetalBuffer,
    down: MetalBuffer,

    fn init(engine: *InferenceEngine, n_tokens: u32, q_dim: u32, kv_dim: u32, inter_dim: u32) !BatchedPrefillScratch {
        const ctx = engine.device.ctx;
        const hidden_dim = engine.config.hidden_dim;
        const f32_sz: usize = @sizeOf(f32);
        const n: usize = n_tokens;
        const h = try metal_buffer.createBuffer(ctx, n * hidden_dim * f32_sz);
        errdefer {
            var mut = h;
            metal_buffer.freeBuffer(&mut);
        }
        const nm = try metal_buffer.createBuffer(ctx, n * hidden_dim * f32_sz);
        errdefer {
            var mut = nm;
            metal_buffer.freeBuffer(&mut);
        }
        const qb = try metal_buffer.createBuffer(ctx, n * q_dim * f32_sz);
        errdefer {
            var mut = qb;
            metal_buffer.freeBuffer(&mut);
        }
        const kb = try metal_buffer.createBuffer(ctx, n * kv_dim * f32_sz);
        errdefer {
            var mut = kb;
            metal_buffer.freeBuffer(&mut);
        }
        const vb = try metal_buffer.createBuffer(ctx, n * kv_dim * f32_sz);
        errdefer {
            var mut = vb;
            metal_buffer.freeBuffer(&mut);
        }
        const ao = try metal_buffer.createBuffer(ctx, n * q_dim * f32_sz);
        errdefer {
            var mut = ao;
            metal_buffer.freeBuffer(&mut);
        }
        const gb = try metal_buffer.createBuffer(ctx, n * inter_dim * f32_sz);
        errdefer {
            var mut = gb;
            metal_buffer.freeBuffer(&mut);
        }
        const ub = try metal_buffer.createBuffer(ctx, n * inter_dim * f32_sz);
        errdefer {
            var mut = ub;
            metal_buffer.freeBuffer(&mut);
        }
        const sw = try metal_buffer.createBuffer(ctx, n * inter_dim * f32_sz);
        errdefer {
            var mut = sw;
            metal_buffer.freeBuffer(&mut);
        }
        const db = try metal_buffer.createBuffer(ctx, n * hidden_dim * f32_sz);
        return .{
            .n_tokens = n_tokens,
            .hidden = h,
            .norm = nm,
            .q = qb,
            .k = kb,
            .v = vb,
            .attn_out = ao,
            .gate = gb,
            .up = ub,
            .swiglu = sw,
            .down = db,
        };
    }

    fn deinit(self: *BatchedPrefillScratch) void {
        metal_buffer.freeBuffer(&self.hidden);
        metal_buffer.freeBuffer(&self.norm);
        metal_buffer.freeBuffer(&self.q);
        metal_buffer.freeBuffer(&self.k);
        metal_buffer.freeBuffer(&self.v);
        metal_buffer.freeBuffer(&self.attn_out);
        metal_buffer.freeBuffer(&self.gate);
        metal_buffer.freeBuffer(&self.up);
        metal_buffer.freeBuffer(&self.swiglu);
        metal_buffer.freeBuffer(&self.down);
    }
};

/// Dispatch a weight × matrix batched matmul using the appropriate GEMM kernel
/// for the tensor's quant type. Only Q4_K and Q6_K are supported here — callers
/// must have verified the type via `canUseBatchedPrefill`.
fn dispatchGemmBatchedOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    weight: *const metal_loader.LoadedTensor,
    input: *const MetalBuffer,
    output: *const MetalBuffer,
    M: u32,
    K: u32,
    N: u32,
) void {
    switch (weight.info.type_) {
        .q4_k => dispatchGemmQ4KOnCmd(engine, cmd, weight, input, output, M, K, N),
        .q6_k => dispatchGemmQ6KOnCmd(engine, cmd, weight, input, output, M, K, N),
        else => unreachable,
    }
}

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
    logits_readback_buf: MetalBuffer,
    argmax_buf: MetalBuffer,
    embed_staging: MetalBuffer,
    lm_head_private_buf: MetalBuffer,
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
    attn_sinks_buf: MetalBuffer, // Per-head attention sink values (reused per layer)
    attn_sink_values: ?[][]f32, // Preloaded per-layer sink values

    // DMMV compute pipelines (one per quant type)
    dmmv_q4k_pipe: MetalPipeline,
    dmmv_q4k_k2048_pipe: MetalPipeline,
    dmmv_q4k_lmhead_pipe: MetalPipeline,
    dmmv_q4k_lmhead_1024_pipe: MetalPipeline,
    dmmv_q5k_pipe: MetalPipeline,
    dmmv_q5k_native_pipe: MetalPipeline,
    dmmv_q6k_pipe: MetalPipeline,
    dmmv_q8_0_pipe: MetalPipeline,
    dmmv_q5_0_pipe: MetalPipeline,
    dmmv_q5_1_pipe: MetalPipeline,
    dmmv_mxfp4_pipe: MetalPipeline,
    dmmv_q8_0_k2048_pipe: MetalPipeline,
    dmmv_q8_0_dual_pipe: MetalPipeline,
    dmmv_q8_0_k2048_fused_norm_pipe: MetalPipeline,
    dmmv_q8_0_dual_fused_norm_pipe: MetalPipeline,
    dmmv_q8_0_repacked_pipe: MetalPipeline,
    dmmv_f16_pipe: MetalPipeline,
    dmmv_f32_pipe: MetalPipeline,
    dmmv_q4k_moe_pipe: MetalPipeline,
    dmmv_q5_1_moe_pipe: MetalPipeline,
    dmmv_q5k_moe_pipe: MetalPipeline,
    dmmv_q5k_moe_k2048_pipe: MetalPipeline,
    dmmv_q6k_moe_pipe: MetalPipeline,
    dmmv_q4k_moe_k2048_pipe: MetalPipeline,
    dmmv_q4k_moe_k2048_1024_pipe: MetalPipeline,

    // Elementwise compute pipelines (for batched GPU dispatch)
    deinterleave_pipe: MetalPipeline,
    flash_attn_pipe: MetalPipeline,
    flash_attn_q8_pipe: MetalPipeline,
    kv_cache_write_pipe: MetalPipeline,
    kv_cache_write_q8_pipe: MetalPipeline,
    rope_pipe: MetalPipeline,
    rope_native_pipe: MetalPipeline,
    sigmoid_mul_pipe: MetalPipeline,
    geglu_pipe: MetalPipeline,
    geglu_batched_pipe: MetalPipeline,
    swiglu_pipe: MetalPipeline,
    swiglu_batched_pipe: MetalPipeline,
    scale_acc_pipe: MetalPipeline,
    rms_norm_pipe: MetalPipeline,
    rms_norm_offset_pipe: MetalPipeline,
    moe_acc_pipe: MetalPipeline,
    moe_acc_batched_pipe: MetalPipeline,
    softmax_topk_pipe: MetalPipeline,
    softmax_topk_scaled_pipe: MetalPipeline,
    softmax_topk_batched_pipe: MetalPipeline,
    moe_route_pack_pipe: MetalPipeline,
    sigmoid_scale_acc_pipe: MetalPipeline,
    moe_weighted_acc_pipe: MetalPipeline,
    moe_weighted_acc_scaled_pipe: MetalPipeline,
    residual_rms_norm_pipe: MetalPipeline,
    moe_weighted_acc_shared_pipe: MetalPipeline,
    copy_u32_pipe: MetalPipeline,
    copy_f32_pipe: MetalPipeline,
    zero_f32_pipe: MetalPipeline,
    argmax_pipe: MetalPipeline,

    // Batched GEMM pipelines for prefill (process N tokens per dispatch).
    // Only loaded when the model has Q4_K / Q6_K weights; other quants stay
    // on the DMMV path until GEMM kernels are ported.
    gemm_q4k_pipe: MetalPipeline,
    gemm_q6k_pipe: MetalPipeline,
    // Batched flash attention for prefill — handles N queries with causal masking.
    // `_q8` variant reads K/V from a Q8_0-quantized cache; default reads f32.
    flash_attn_batched_pipe: MetalPipeline,
    flash_attn_batched_q8_pipe: MetalPipeline,
    // Batched rotary position embedding — rotates N tokens at consecutive positions in one dispatch.
    rope_batched_pipe: MetalPipeline,

    // Preloaded norm weight buffers (f32, GPU-accessible via UMA)
    attn_norm_bufs: []MetalBuffer,
    attn_q_norm_bufs: []MetalBuffer,
    attn_k_norm_bufs: []MetalBuffer,
    attn_q_norm_present: []bool,
    attn_k_norm_present: []bool,
    post_attn_norm_bufs: []MetalBuffer,
    post_attn_norm_present: []bool,
    ffn_norm_bufs: []MetalBuffer,
    post_ffn_norm_bufs: []MetalBuffer,
    post_ffn_norm_present: []bool,
    layer_output_scales: []f32,
    final_norm_gpu: MetalBuffer,
    unit_rms_norm_weights: MetalBuffer,
    rope_freq_buf: MetalBuffer,
    rope_variant_freq_buf: MetalBuffer,
    rope_freq_factors: ?[]f32,
    rope_variant_rope_dim: u32,
    rope_variant_freq_base: f32,
    rope_variant_uses_freq_factors: bool,

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
    private_ssm_qkv_bufs: ?[]MetalBuffer,
    private_ssm_gate_bufs: ?[]MetalBuffer,
    private_ssm_out_bufs: ?[]MetalBuffer,
    token_embed: *const metal_loader.LoadedTensor,
    lm_head: *const metal_loader.LoadedTensor,

    // Decode state
    position: u32,
    max_context_tokens: u32,
    profile_enabled: bool,
    debug_validation_enabled: bool,
    gemma_moe_validation_enabled: bool,
    private_decode_buffers: bool,
    command_encoder_mode: CommandEncoderMode,
    kv_cache_q8: bool,
    kv_cache_head_stride_bytes: u32,
    kv_cache_bytes_per_token: u32,
    q8_tg_override: ?u32,
    q8_dual_tg_override: ?u32,
    request_profile: RuntimeProfile,

    /// Initialize the Metal inference engine, allocating GPU buffers and compiling pipelines.
    pub fn init(
        model: *const metal_loader.Model,
        device: *const metal_device.MetalDevice,
        allocator: std.mem.Allocator,
        options: InitOptions,
    ) !InferenceEngine {
        const cfg = model.config;
        const ctx = device.ctx;
        const weights_bytes = tensorBytes(model);
        const runtime_profile = memory_plan.profile(cfg);
        const requested_ctx = memory_plan.requestedContextTokens(cfg, null, runtime_context_cap);
        const max_ctx = runtime_profile.maxContextTokensForUnifiedBudget(
            weights_bytes,
            memoryBudget(device),
            requested_ctx,
        );
        if (max_ctx == 0) {
            log.err("No decode context fits within {d:.2} GiB Metal working-set budget", .{
                @as(f64, @floatFromInt(memoryBudget(device))) / (1024.0 * 1024.0 * 1024.0),
            });
            return error.ContextLengthDoesNotFit;
        }
        if (max_ctx < requested_ctx) {
            log.warn("Metal context trimmed from {d} to {d} tokens to fit current UMA budget", .{
                requested_ctx,
                max_ctx,
            });
        } else {
            log.info("Metal KV cache planned context: requested {d}, reserved {d}", .{
                requested_ctx,
                max_ctx,
            });
        }

        // Compute dimension-dependent sizes
        const q_dim: u32 = cfg.n_heads * cfg.head_dim;
        const kv_dim: u32 = kvDim(cfg);
        const inter_dim: u32 = if (cfg.intermediate_dim > 0) cfg.intermediate_dim else cfg.hidden_dim * 4;
        const shexp_inter_dim: u32 = if (cfg.shared_expert_intermediate_dim > 0) cfg.shared_expert_intermediate_dim else inter_dim;
        const d_inner: u32 = cfg.ssm_d_inner;
        const conv_channels: u32 = if (d_inner > 0) d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state else 0;
        const kv_cache_q8 = options.kv_cache_q8_override orelse
            readBoolEnv("ZINC_METAL_KV_Q8") orelse
            defaultKvCacheQ8Enabled(cfg, options.debug_validation_enabled);
        const kv_cache_bytes_per_token: usize = @intCast(kvCacheBytesPerToken(cfg, kv_cache_q8));
        const kv_cache_head_stride_bytes: u32 = if (kv_cache_q8)
            @intCast(@divTrunc(cfg.head_dim, 32) * 34)
        else
            cfg.head_dim * @sizeOf(f32);

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
        const kv_cache_size: usize = @as(usize, max_ctx) * kv_cache_bytes_per_token;
        const page_table_size: usize = @as(usize, max_ctx) * @sizeOf(u32);
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
        self.max_context_tokens = max_ctx;
        self.profile_enabled = options.profile_enabled;
        self.debug_validation_enabled = options.debug_validation_enabled;
        self.gemma_moe_validation_enabled = readBoolEnv("ZINC_GEMMA_MOE_VALIDATE") orelse false;
        self.private_decode_buffers = if (options.debug_validation_enabled or self.gemma_moe_validation_enabled)
            false
        else
            options.private_decode_buffers_override orelse
                readBoolEnv("ZINC_METAL_PRIVATE_DECODE") orelse
                modelSupportsPrivateDecodeBuffers(model, cfg);
        self.command_encoder_mode = options.command_encoder_mode orelse .concurrent;
        self.kv_cache_q8 = kv_cache_q8;
        self.kv_cache_head_stride_bytes = kv_cache_head_stride_bytes;
        self.kv_cache_bytes_per_token = @intCast(kv_cache_bytes_per_token);
        self.q8_tg_override = null;
        self.q8_dual_tg_override = null;
        self.request_profile = .{};
        self.private_ssm_qkv_bufs = null;
        self.private_ssm_gate_bufs = null;
        self.private_ssm_out_bufs = null;

        self.hidden_buf = try createMetalBufferForMode(ctx, hidden_size, self.private_decode_buffers);
        self.residual_buf = try createMetalBufferForMode(ctx, hidden_size, self.private_decode_buffers);
        self.norm_buf = try createMetalBufferForMode(ctx, hidden_size, self.private_decode_buffers);
        self.q_buf = try createMetalBufferForMode(ctx, head_total, self.private_decode_buffers);
        self.k_buf = try createMetalBufferForMode(ctx, kv_total, self.private_decode_buffers);
        self.v_buf = try createMetalBufferForMode(ctx, kv_total, self.private_decode_buffers);
        self.attn_out_buf = try createMetalBufferForMode(ctx, @max(attn_out_size, 4), self.private_decode_buffers);
        self.gate_buf = try createMetalBufferForMode(ctx, @max(gate_size, 4), self.private_decode_buffers);
        self.up_buf = try createMetalBufferForMode(ctx, @max(up_size, 4), self.private_decode_buffers);
        self.swiglu_buf = try createMetalBufferForMode(ctx, @max(swiglu_size, 4), self.private_decode_buffers);
        self.down_buf = try createMetalBufferForMode(ctx, hidden_size, self.private_decode_buffers);
        self.moe_out_buf = try createMetalBufferForMode(ctx, hidden_size, self.private_decode_buffers);
        self.router_logits_buf = try createMetalBufferForMode(ctx, @max(router_size, 4), self.private_decode_buffers);
        self.router_output_buf = try createMetalBufferForMode(ctx, router_output_size, self.private_decode_buffers);
        self.logits_buf = try createMetalBufferForMode(ctx, vocab_size, self.private_decode_buffers);
        self.logits_readback_buf = if (self.private_decode_buffers)
            try metal_buffer.createBuffer(ctx, vocab_size)
        else
            .{ .handle = null, .size = 0, .cpu_ptr = null, .is_mmap_wrapped = false };
        self.argmax_buf = try metal_buffer.createBuffer(ctx, 2 * @sizeOf(u32));
        self.embed_staging = try metal_buffer.createBuffer(ctx, hidden_size);
        self.lm_head_private_buf = .{ .handle = null, .size = 0, .cpu_ptr = null, .is_mmap_wrapped = false };
        self.expert_ids_buf = try metal_buffer.createBuffer(ctx, expert_ids_size);
        self.expert_gate_batch_buf = try createMetalBufferForMode(ctx, expert_inter_batch_size, self.private_decode_buffers);
        self.expert_up_batch_buf = try createMetalBufferForMode(ctx, expert_inter_batch_size, self.private_decode_buffers);
        self.expert_swiglu_batch_buf = try createMetalBufferForMode(ctx, expert_inter_batch_size, self.private_decode_buffers);
        self.expert_down_batch_buf = try createMetalBufferForMode(ctx, expert_hidden_batch_size, self.private_decode_buffers);

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

        // Allocate KV cache only for full-attention layers.
        self.kv_k_cache = try allocator.alloc(MetalBuffer, cfg.n_layers);
        self.kv_v_cache = try allocator.alloc(MetalBuffer, cfg.n_layers);
        for (0..cfg.n_layers) |i| {
            if (isFullAttentionLayer(cfg, i)) {
                self.kv_k_cache[i] = try createMetalBufferForMode(ctx, kv_cache_size, self.private_decode_buffers);
                self.kv_v_cache[i] = try createMetalBufferForMode(ctx, kv_cache_size, self.private_decode_buffers);
            } else {
                self.kv_k_cache[i] = .{ .handle = null, .size = 0, .cpu_ptr = null, .is_mmap_wrapped = false };
                self.kv_v_cache[i] = .{ .handle = null, .size = 0, .cpu_ptr = null, .is_mmap_wrapped = false };
            }
        }
        self.page_table_buf = try metal_buffer.createBuffer(ctx, page_table_size);
        {
            const page_table_ptr: [*]u32 = @ptrCast(@alignCast(self.page_table_buf.cpu_ptr.?));
            for (0..max_ctx) |i| {
                page_table_ptr[i] = @intCast(i);
            }
        }
        // Attention sinks buffer + preloaded values
        self.attn_sinks_buf = try metal_buffer.createBuffer(ctx, @max(@as(usize, cfg.n_heads) * @sizeOf(f32), 4));
        {
            // Fill with NaN (disabled) by default
            const sink_ptr: [*]f32 = @ptrCast(@alignCast(self.attn_sinks_buf.cpu_ptr.?));
            for (0..cfg.n_heads) |i| sink_ptr[i] = std.math.nan(f32);
        }
        {
            var has_sinks = false;
            for (0..cfg.n_layers) |i| {
                if (findLayerTensor(model, @intCast(i), "attn_sinks.weight") != null) {
                    has_sinks = true;
                    break;
                }
            }
            if (has_sinks) {
                self.attn_sink_values = try allocator.alloc([]f32, cfg.n_layers);
                for (0..cfg.n_layers) |i| {
                    if (findLayerTensor(model, @intCast(i), "attn_sinks.weight")) |t| {
                        const n_heads_for_sinks = @as(usize, t.info.dims[0]);
                        self.attn_sink_values.?[i] = try allocator.alloc(f32, n_heads_for_sinks);
                        const sinks_mmap = model.mmap_data orelse return error.NoMmapData;
                        const off: usize = @intCast(model.gguf_file.tensor_data_offset + t.info.offset);
                        readMmapFloats(sinks_mmap, off, t.info.type_, self.attn_sink_values.?[i]);
                    } else {
                        self.attn_sink_values.?[i] = &.{};
                    }
                }
                log.info("Attention sinks: loaded for {d} layers", .{cfg.n_layers});
            } else {
                self.attn_sink_values = null;
            }
        }

        // Load DMMV compute pipelines for all quant types
        self.dmmv_q4k_pipe = try loadShaderPipeline(ctx, "dmmv_q4k");
        self.dmmv_q4k_k2048_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_k2048");
        self.dmmv_q4k_lmhead_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_lmhead");
        self.dmmv_q4k_lmhead_1024_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_lmhead_1024");
        self.dmmv_q5k_pipe = try loadShaderPipeline(ctx, "dmmv_q5k");
        self.dmmv_q5k_native_pipe = try loadShaderPipeline(ctx, "dmmv_q5k_native");
        self.dmmv_q6k_pipe = try loadShaderPipeline(ctx, "dmmv_q6k");
        self.dmmv_q8_0_pipe = try loadShaderPipeline(ctx, "dmmv_q8_0");
        self.dmmv_q5_0_pipe = try loadShaderPipeline(ctx, "dmmv_q5_0");
        self.dmmv_q5_1_pipe = try loadShaderPipeline(ctx, "dmmv_q5_1");
        self.dmmv_mxfp4_pipe = try loadShaderPipeline(ctx, "dmmv_mxfp4");
        self.dmmv_q8_0_k2048_pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_k2048");
        self.dmmv_q8_0_dual_pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_dual");
        self.dmmv_q8_0_k2048_fused_norm_pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_k2048_fused_norm");
        self.dmmv_q8_0_dual_fused_norm_pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_dual_fused_norm");
        self.dmmv_q8_0_repacked_pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_repacked");
        self.dmmv_f16_pipe = try loadShaderPipeline(ctx, "dmmv_f16");
        self.dmmv_f32_pipe = try loadShaderPipeline(ctx, "dmmv_f32");
        self.dmmv_q4k_moe_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe");
        self.dmmv_q5_1_moe_pipe = try loadShaderPipeline(ctx, "dmmv_q5_1_moe");
        self.dmmv_q5k_moe_pipe = try loadShaderPipeline(ctx, "dmmv_q5k_moe");
        self.dmmv_q5k_moe_k2048_pipe = try loadShaderPipeline(ctx, "dmmv_q5k_moe_k2048");
        self.dmmv_q6k_moe_pipe = try loadShaderPipeline(ctx, "dmmv_q6k_moe");
        self.dmmv_q4k_moe_k2048_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe_k2048");
        self.dmmv_q4k_moe_k2048_1024_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_moe_k2048_1024");

        // Elementwise pipelines for batched GPU dispatch
        self.deinterleave_pipe = try loadShaderPipeline(ctx, "deinterleave");
        self.flash_attn_pipe = try loadShaderPipeline(ctx, "flash_attn");
        self.flash_attn_q8_pipe = try loadShaderPipeline(ctx, "flash_attn_q8");
        self.kv_cache_write_pipe = try loadShaderPipeline(ctx, "kv_cache_write");
        self.kv_cache_write_q8_pipe = try loadShaderPipeline(ctx, "kv_cache_write_q8");
        self.rope_pipe = try loadShaderPipeline(ctx, "rope_fused");
        self.rope_native_pipe = try loadShaderPipeline(ctx, "rope_native");
        self.sigmoid_mul_pipe = try loadShaderPipeline(ctx, "sigmoid_mul");
        self.geglu_pipe = try loadShaderPipeline(ctx, "geglu");
        self.geglu_batched_pipe = try loadShaderPipeline(ctx, "geglu_batched");
        self.swiglu_pipe = try loadShaderPipeline(ctx, "swiglu");
        self.swiglu_batched_pipe = try loadShaderPipeline(ctx, "swiglu_batched");
        self.scale_acc_pipe = try loadShaderPipeline(ctx, "scale_accumulate");
        self.rms_norm_pipe = try loadShaderPipeline(ctx, "rms_norm_mul");
        self.rms_norm_offset_pipe = try loadShaderPipeline(ctx, "rms_norm_mul_offset");
        self.moe_acc_pipe = try loadShaderPipeline(ctx, "moe_accumulate");
        self.moe_acc_batched_pipe = try loadShaderPipeline(ctx, "moe_accumulate_batched");
        self.softmax_topk_pipe = try loadShaderPipeline(ctx, "softmax_topk");
        self.softmax_topk_scaled_pipe = try loadShaderPipeline(ctx, "softmax_topk_scaled");
        self.softmax_topk_batched_pipe = try loadShaderPipeline(ctx, "softmax_topk_batched");
        self.moe_route_pack_pipe = try loadShaderPipeline(ctx, "moe_route_pack");
        self.sigmoid_scale_acc_pipe = try loadShaderPipeline(ctx, "sigmoid_scale_acc");
        self.moe_weighted_acc_pipe = try loadShaderPipeline(ctx, "moe_weighted_acc");
        self.moe_weighted_acc_scaled_pipe = try loadShaderPipeline(ctx, "moe_weighted_acc_scaled");
        self.residual_rms_norm_pipe = try loadShaderPipeline(ctx, "residual_rms_norm");
        self.moe_weighted_acc_shared_pipe = try loadShaderPipeline(ctx, "moe_weighted_acc_shared");
        self.copy_u32_pipe = try loadShaderPipeline(ctx, "copy_u32");
        self.copy_f32_pipe = try loadShaderPipeline(ctx, "copy_f32");
        self.zero_f32_pipe = try loadShaderPipeline(ctx, "zero_f32");
        self.argmax_pipe = try loadShaderPipeline(ctx, "argmax");
        self.gemm_q4k_pipe = try loadShaderPipeline(ctx, "gemm_q4k");
        self.gemm_q6k_pipe = try loadShaderPipeline(ctx, "gemm_q6k");
        self.flash_attn_batched_pipe = try loadShaderPipeline(ctx, "flash_attn_batched");
        self.flash_attn_batched_q8_pipe = try loadShaderPipeline(ctx, "flash_attn_batched_q8");
        self.rope_batched_pipe = try loadShaderPipeline(ctx, "rope_batched");
        const q8_simd_width = if (self.dmmv_q8_0_pipe.thread_execution_width > 0) self.dmmv_q8_0_pipe.thread_execution_width else @as(u32, 32);
        const q8_dual_simd_width = if (self.dmmv_q8_0_dual_pipe.thread_execution_width > 0) self.dmmv_q8_0_dual_pipe.thread_execution_width else @as(u32, 32);
        self.q8_tg_override = options.q8_tg_override orelse
            readThreadgroupOverride("ZINC_METAL_Q8_TG_SIZE", q8_simd_width, self.dmmv_q8_0_pipe.max_threads_per_threadgroup) orelse
            defaultQ8Threadgroup(self.device.chip, q8_simd_width, self.dmmv_q8_0_pipe.max_threads_per_threadgroup);
        self.q8_dual_tg_override = options.q8_dual_tg_override orelse
            readThreadgroupOverride("ZINC_METAL_Q8_DUAL_TG_SIZE", q8_dual_simd_width, self.dmmv_q8_0_dual_pipe.max_threads_per_threadgroup) orelse
            defaultQ8DualThreadgroup(self.device.chip, q8_dual_simd_width, self.dmmv_q8_0_dual_pipe.max_threads_per_threadgroup);

        // Preload norm weights into f32 Metal buffers (eliminates per-token alloc + mmap dequant)
        self.attn_norm_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
        self.attn_q_norm_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
        self.attn_k_norm_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
        self.attn_q_norm_present = try allocator.alloc(bool, cfg.n_layers);
        self.attn_k_norm_present = try allocator.alloc(bool, cfg.n_layers);
        self.post_attn_norm_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
        self.post_attn_norm_present = try allocator.alloc(bool, cfg.n_layers);
        self.ffn_norm_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
        // post_ffw_norm_bufs mirror GGUF post_ffw_norm.weight tensors for
        // Gemma-style post-FFN RMS norms.
        self.post_ffn_norm_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
        self.post_ffn_norm_present = try allocator.alloc(bool, cfg.n_layers);
        self.layer_output_scales = try allocator.alloc(f32, cfg.n_layers);
        for (0..cfg.n_layers) |i| {
            const layer: u32 = @intCast(i);
            const an = findLayerTensor(model, layer, "attn_norm.weight") orelse return error.MissingTensor;
            self.attn_norm_bufs[i] = try preloadNormWeights(ctx, model, an, cfg.hidden_dim);
            if (findLayerTensor(model, layer, "attn_q_norm.weight")) |qn| {
                self.attn_q_norm_bufs[i] = try preloadNormWeights(ctx, model, qn, @intCast(qn.info.numElements()));
                self.attn_q_norm_present[i] = true;
            } else {
                self.attn_q_norm_bufs[i] = try metal_buffer.createBuffer(ctx, 4);
                @memset(self.attn_q_norm_bufs[i].cpu_ptr.?[0..4], 0);
                self.attn_q_norm_present[i] = false;
            }
            if (findLayerTensor(model, layer, "attn_k_norm.weight")) |kn| {
                self.attn_k_norm_bufs[i] = try preloadNormWeights(ctx, model, kn, @intCast(kn.info.numElements()));
                self.attn_k_norm_present[i] = true;
            } else {
                self.attn_k_norm_bufs[i] = try metal_buffer.createBuffer(ctx, 4);
                @memset(self.attn_k_norm_bufs[i].cpu_ptr.?[0..4], 0);
                self.attn_k_norm_present[i] = false;
            }
            if (findLayerTensor(model, layer, "post_attention_norm.weight")) |pan| {
                const has_separate_ffn_norm = findLayerTensor(model, layer, "ffn_norm.weight") != null;
                self.post_attn_norm_bufs[i] = try preloadNormWeights(ctx, model, pan, cfg.hidden_dim);
                self.post_attn_norm_present[i] = has_separate_ffn_norm or (cfg.architecture == .gemma and cfg.rope_freq_base_swa > 0);
            } else {
                self.post_attn_norm_bufs[i] = try metal_buffer.createBuffer(ctx, 4);
                @memset(self.post_attn_norm_bufs[i].cpu_ptr.?[0..4], 0);
                self.post_attn_norm_present[i] = false;
            }
            // FFN norm: prefer ffn_norm.weight via findLayerTensor(model, layer, "ffn_norm.weight"), then fall back to post_attention_norm.weight.
            const fn_t = findLayerTensor(model, layer, "ffn_norm.weight") orelse
                findLayerTensor(model, layer, "post_attention_norm.weight") orelse return error.MissingTensor;
            self.ffn_norm_bufs[i] = try preloadNormWeights(ctx, model, fn_t, cfg.hidden_dim);
            if (findLayerTensor(model, layer, "post_ffw_norm.weight")) |pfn| {
                self.post_ffn_norm_bufs[i] = try preloadNormWeights(ctx, model, pfn, cfg.hidden_dim);
                self.post_ffn_norm_present[i] = true;
            } else {
                self.post_ffn_norm_bufs[i] = try metal_buffer.createBuffer(ctx, 4);
                @memset(self.post_ffn_norm_bufs[i].cpu_ptr.?[0..4], 0);
                self.post_ffn_norm_present[i] = false;
            }
            if (findLayerTensor(model, layer, "layer_output_scale.weight")) |los| {
                const mmap = model.mmap_data orelse return error.NoMmapData;
                const off: usize = @intCast(model.gguf_file.tensor_data_offset + los.info.offset);
                var scalar: [1]f32 = undefined;
                readMmapFloats(mmap, off, los.info.type_, scalar[0..1]);
                self.layer_output_scales[i] = scalar[0];
            } else {
                self.layer_output_scales[i] = 1.0;
            }
        }
        const final_t = findTensorByName(model, "output_norm.weight") orelse return error.MissingTensor;
        self.final_norm_gpu = try preloadNormWeights(ctx, model, final_t, cfg.hidden_dim);
        self.unit_rms_norm_weights = try metal_buffer.createBuffer(ctx, hidden_size);
        {
            const unit_ptr: [*]f32 = @ptrCast(@alignCast(self.unit_rms_norm_weights.cpu_ptr.?));
            for (0..cfg.hidden_dim) |i| unit_ptr[i] = 1.0;
        }

        // Precompute inverse RoPE frequencies into a Metal buffer.
        // If the model provides rope_freqs.weight, divide by those factors.
        {
            const rope_dim: u32 = if (cfg.rope_dim > 0) cfg.rope_dim else cfg.head_dim;
            const half_rot: u32 = rope_dim / 2;
            const freq_buf_size: usize = @as(usize, half_rot) * @sizeOf(f32);
            self.rope_freq_buf = try metal_buffer.createBuffer(ctx, @max(freq_buf_size, 4));
            self.rope_variant_freq_buf = try metal_buffer.createBuffer(ctx, @max(freq_buf_size, 4));
            self.rope_freq_factors = null;
            self.rope_variant_rope_dim = 0;
            self.rope_variant_freq_base = 0;
            self.rope_variant_uses_freq_factors = false;
            const freq_ptr: [*]f32 = @ptrCast(@alignCast(self.rope_freq_buf.cpu_ptr.?));

            // If rope_freqs.weight exists, divide each inv_freq by the corresponding factor.
            if (findTensorByName(model, "rope_freqs.weight")) |rope_freqs_t| {
                const mmap_data = model.mmap_data orelse return error.NoMmapData;
                const tensor_data_off = model.gguf_file.tensor_data_offset;
                const off: usize = @intCast(tensor_data_off + rope_freqs_t.info.offset);
                self.rope_freq_factors = try allocator.alloc(f32, half_rot);
                readMmapFloats(mmap_data, off, rope_freqs_t.info.type_, self.rope_freq_factors.?);
            }
            fillRopeInvFreqs(freq_ptr[0..half_rot], rope_dim, cfg.rope_freq_base, self.rope_freq_factors);

            // YaRN RoPE scaling for extended context (gpt-oss).
            if (cfg.rope_scaling_factor > 1.0 and cfg.rope_original_context > 0) {
                const factor = cfg.rope_scaling_factor;
                const orig_ctx = @as(f32, @floatFromInt(cfg.rope_original_context));
                const beta_fast: f32 = 32.0;
                const beta_slow: f32 = 1.0;
                const low_freq_wavelen = orig_ctx / beta_slow;
                const high_freq_wavelen = orig_ctx / beta_fast;
                for (0..half_rot) |i| {
                    const freq = freq_ptr[i];
                    const wavelen = 2.0 * std.math.pi / freq;
                    if (wavelen < high_freq_wavelen) {
                        // High frequency: no scaling
                    } else if (wavelen > low_freq_wavelen) {
                        freq_ptr[i] = freq / factor;
                    } else {
                        const t = (orig_ctx / wavelen - beta_slow) / (beta_fast - beta_slow);
                        const smooth = @max(@as(f32, 0.0), @min(@as(f32, 1.0), t));
                        freq_ptr[i] = freq * ((1.0 - smooth) / factor + smooth);
                    }
                }
                log.info("RoPE: applied YaRN scaling factor={d:.1} orig_ctx={d}", .{ factor, cfg.rope_original_context });
            }
        }

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
                self.ssm_conv_state_bufs.?[i] = try createMetalBufferForMode(ctx, @max(cs_size, 4), self.private_decode_buffers);
                if (!self.private_decode_buffers) {
                    @memset(self.ssm_conv_state_bufs.?[i].cpu_ptr.?[0..@max(cs_size, 4)], 0);
                }
                // Recurrent state: dt_rank * head_v_dim * head_v_dim floats
                const st_size: usize = @as(usize, cfg.ssm_dt_rank) * head_v_dim * head_v_dim * @sizeOf(f32);
                self.ssm_state_bufs.?[i] = try createMetalBufferForMode(ctx, @max(st_size, 4), self.private_decode_buffers);
                if (!self.private_decode_buffers) {
                    @memset(self.ssm_state_bufs.?[i].cpu_ptr.?[0..@max(st_size, 4)], 0);
                }
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
            const gemma_moe_shared_gate = if (cfg.architecture == .gemma and findLayerTensor(model, layer, "ffn_gate_up_exps.weight") != null)
                (findLayerTensor(model, layer, "ffn_gate_shexp.weight") orelse findLayerTensor(model, layer, "ffn_gate.weight"))
            else
                findLayerTensor(model, layer, "ffn_gate_shexp.weight");
            const gemma_moe_shared_up = if (cfg.architecture == .gemma and findLayerTensor(model, layer, "ffn_gate_up_exps.weight") != null)
                (findLayerTensor(model, layer, "ffn_up_shexp.weight") orelse findLayerTensor(model, layer, "ffn_up.weight"))
            else
                findLayerTensor(model, layer, "ffn_up_shexp.weight");
            const gemma_moe_shared_down = if (cfg.architecture == .gemma and findLayerTensor(model, layer, "ffn_gate_up_exps.weight") != null)
                (findLayerTensor(model, layer, "ffn_down_shexp.weight") orelse findLayerTensor(model, layer, "ffn_down.weight"))
            else
                findLayerTensor(model, layer, "ffn_down_shexp.weight");
            self.layer_tensors[i] = .{
                .attn_q = findLayerTensor(model, layer, "attn_q.weight"),
                .attn_k = findLayerTensor(model, layer, "attn_k.weight"),
                .attn_v = findLayerTensor(model, layer, "attn_v.weight"),
                .attn_q_bias = findLayerTensor(model, layer, "attn_q.bias"),
                .attn_k_bias = findLayerTensor(model, layer, "attn_k.bias"),
                .attn_v_bias = findLayerTensor(model, layer, "attn_v.bias"),
                .attn_q_norm = findLayerTensor(model, layer, "attn_q_norm.weight"),
                .attn_k_norm = findLayerTensor(model, layer, "attn_k_norm.weight"),
                .attn_output = findLayerTensor(model, layer, "attn_output.weight"),
                .attn_output_bias = findLayerTensor(model, layer, "attn_output.bias"),
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
                .ffn_gate_inp_scale = findLayerTensor(model, layer, "ffn_gate_inp.scale"),
                .ffn_gate_exps = findLayerTensor(model, layer, "ffn_gate_exps.weight"),
                .ffn_up_exps = findLayerTensor(model, layer, "ffn_up_exps.weight"),
                .ffn_gate_up_exps = findLayerTensor(model, layer, "ffn_gate_up_exps.weight"),
                .ffn_down_exps = findLayerTensor(model, layer, "ffn_down_exps.weight"),
                .ffn_down_exps_scale = findLayerTensor(model, layer, "ffn_down_exps.scale"),
                .ffn_gate_exps_bias = findLayerTensor(model, layer, "ffn_gate_exps.bias"),
                .ffn_up_exps_bias = findLayerTensor(model, layer, "ffn_up_exps.bias"),
                .ffn_down_exps_bias = findLayerTensor(model, layer, "ffn_down_exps.bias"),
                .ffn_gate_inp_bias = findLayerTensor(model, layer, "ffn_gate_inp.bias"),
                .pre_ffw_norm_2 = findLayerTensor(model, layer, "pre_ffw_norm_2.weight"),
                .post_ffw_norm_1 = findLayerTensor(model, layer, "post_ffw_norm_1.weight"),
                .post_ffw_norm_2 = findLayerTensor(model, layer, "post_ffw_norm_2.weight"),
                .ffn_gate_shexp = gemma_moe_shared_gate,
                .ffn_up_shexp = gemma_moe_shared_up,
                .ffn_down_shexp = gemma_moe_shared_down,
                .ffn_gate_inp_shexp = findLayerTensor(model, layer, "ffn_gate_inp_shexp.weight"),
                .ffn_gate = findLayerTensor(model, layer, "ffn_gate.weight"),
                .ffn_up = findLayerTensor(model, layer, "ffn_up.weight"),
                .ffn_down = findLayerTensor(model, layer, "ffn_down.weight"),
            };
        }
        self.lm_head = findTensorByName(model, "output.weight") orelse
            findTensorByName(model, "token_embd.weight") orelse return error.MissingTensor;
        self.token_embed = findTensorByName(model, "token_embd.weight") orelse return error.MissingTensor;
        log.debug("Metal token/lm quant: token_embd={s} lm_head={s}", .{
            @tagName(self.token_embed.info.type_),
            @tagName(self.lm_head.info.type_),
        });
        if (self.private_decode_buffers and self.lm_head.info.type_ == .q8_0) {
            const lm_head_bytes: usize = @intCast(self.lm_head.info.sizeBytes());
            const lm_K: u32 = @intCast(self.lm_head.info.dims[0]);
            if (self.dmmv_q8_0_repacked_pipe.handle != null and canRepackQ8(lm_K)) {
                self.lm_head_private_buf = try metal_buffer.createBuffer(ctx, lm_head_bytes);
                self.lm_head_private_buf.is_repacked_q8 = true;
                const lm_M: u32 = @intCast(self.lm_head.info.dims[1]);
                const src_ptr: [*]const u8 = self.lm_head.gpu_buffer.cpu_ptr.? + tensorPageOffset(model, self.lm_head);
                repackQ8_0Blocks(src_ptr, self.lm_head_private_buf.cpu_ptr.?, lm_M, lm_K);
                log.info("Metal: repacked lm_head Q8_0 ({d}x{d}) for coalesced access", .{ lm_M, lm_K });
            } else if (lm_head_bytes % @sizeOf(u32) == 0) {
                self.lm_head_private_buf = try metal_buffer.createPrivateBuffer(ctx, lm_head_bytes);
                var cmd = try metal_command.beginCommand(ctx);
                dispatchCopyU32OnCmd(
                    &self,
                    &cmd,
                    &self.lm_head.gpu_buffer,
                    &self.lm_head_private_buf,
                    @intCast(lm_head_bytes / @sizeOf(u32)),
                    @intCast(tensorPageOffset(model, self.lm_head) / @sizeOf(u32)),
                    0,
                );
                cmd.commitAndWait();
            }
        }
        if (self.private_decode_buffers) {
            self.private_ssm_qkv_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
            self.private_ssm_gate_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
            self.private_ssm_out_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
            for (0..cfg.n_layers) |i| {
                self.private_ssm_qkv_bufs.?[i] = .{ .handle = null, .size = 0, .cpu_ptr = null, .is_mmap_wrapped = false };
                self.private_ssm_gate_bufs.?[i] = .{ .handle = null, .size = 0, .cpu_ptr = null, .is_mmap_wrapped = false };
                self.private_ssm_out_bufs.?[i] = .{ .handle = null, .size = 0, .cpu_ptr = null, .is_mmap_wrapped = false };
            }

            const has_repacked = self.dmmv_q8_0_repacked_pipe.handle != null;
            var need_gpu_copy = false;
            var cmd: MetalCommand = undefined;
            for (0..cfg.n_layers) |i| {
                if (self.layer_tensors[i].attn_qkv) |tensor| {
                    const size_bytes: usize = @intCast(tensor.info.sizeBytes());
                    const t_K: u32 = @intCast(tensor.info.dims[0]);
                    if (tensor.info.type_ == .q8_0 and has_repacked and canRepackQ8(t_K)) {
                        var buf = try metal_buffer.createBuffer(ctx, size_bytes);
                        buf.is_repacked_q8 = true;
                        const t_M: u32 = @intCast(tensor.info.dims[1]);
                        repackQ8_0Blocks(tensor.gpu_buffer.cpu_ptr.? + tensorPageOffset(model, tensor), buf.cpu_ptr.?, t_M, t_K);
                        self.private_ssm_qkv_bufs.?[i] = buf;
                    } else if (tensor.info.type_ == .q8_0 and size_bytes % @sizeOf(u32) == 0) {
                        if (!need_gpu_copy) {
                            cmd = try metal_command.beginCommand(ctx);
                            need_gpu_copy = true;
                        }
                        self.private_ssm_qkv_bufs.?[i] = try metal_buffer.createPrivateBuffer(ctx, size_bytes);
                        dispatchCopyU32OnCmd(&self, &cmd, &tensor.gpu_buffer, &self.private_ssm_qkv_bufs.?[i], @intCast(size_bytes / @sizeOf(u32)), @intCast(tensorPageOffset(model, tensor) / @sizeOf(u32)), 0);
                    }
                }
                if (self.layer_tensors[i].attn_gate) |tensor| {
                    const size_bytes: usize = @intCast(tensor.info.sizeBytes());
                    const t_K: u32 = @intCast(tensor.info.dims[0]);
                    if (tensor.info.type_ == .q8_0 and has_repacked and canRepackQ8(t_K)) {
                        var buf = try metal_buffer.createBuffer(ctx, size_bytes);
                        buf.is_repacked_q8 = true;
                        const t_M: u32 = @intCast(tensor.info.dims[1]);
                        repackQ8_0Blocks(tensor.gpu_buffer.cpu_ptr.? + tensorPageOffset(model, tensor), buf.cpu_ptr.?, t_M, t_K);
                        self.private_ssm_gate_bufs.?[i] = buf;
                    } else if (tensor.info.type_ == .q8_0 and size_bytes % @sizeOf(u32) == 0) {
                        if (!need_gpu_copy) {
                            cmd = try metal_command.beginCommand(ctx);
                            need_gpu_copy = true;
                        }
                        self.private_ssm_gate_bufs.?[i] = try metal_buffer.createPrivateBuffer(ctx, size_bytes);
                        dispatchCopyU32OnCmd(&self, &cmd, &tensor.gpu_buffer, &self.private_ssm_gate_bufs.?[i], @intCast(size_bytes / @sizeOf(u32)), @intCast(tensorPageOffset(model, tensor) / @sizeOf(u32)), 0);
                    }
                }
                if (self.layer_tensors[i].ssm_out) |tensor| {
                    const size_bytes: usize = @intCast(tensor.info.sizeBytes());
                    const t_K: u32 = @intCast(tensor.info.dims[0]);
                    if (tensor.info.type_ == .q8_0 and has_repacked and canRepackQ8(t_K)) {
                        var buf = try metal_buffer.createBuffer(ctx, size_bytes);
                        buf.is_repacked_q8 = true;
                        const t_M: u32 = @intCast(tensor.info.dims[1]);
                        repackQ8_0Blocks(tensor.gpu_buffer.cpu_ptr.? + tensorPageOffset(model, tensor), buf.cpu_ptr.?, t_M, t_K);
                        self.private_ssm_out_bufs.?[i] = buf;
                    } else if (tensor.info.type_ == .q8_0 and size_bytes % @sizeOf(u32) == 0) {
                        if (!need_gpu_copy) {
                            cmd = try metal_command.beginCommand(ctx);
                            need_gpu_copy = true;
                        }
                        self.private_ssm_out_bufs.?[i] = try metal_buffer.createPrivateBuffer(ctx, size_bytes);
                        dispatchCopyU32OnCmd(&self, &cmd, &tensor.gpu_buffer, &self.private_ssm_out_bufs.?[i], @intCast(size_bytes / @sizeOf(u32)), @intCast(tensorPageOffset(model, tensor) / @sizeOf(u32)), 0);
                    }
                }
            }
            if (need_gpu_copy) cmd.commitAndWait();
        }

        log.debug("Metal inference engine initialized: {d} layers, {d}x{d} heads, dim={d}", .{
            cfg.n_layers, cfg.n_heads, cfg.head_dim, cfg.hidden_dim,
        });
        log.debug("Metal decode buffers: {s}", .{if (self.private_decode_buffers) "private+staged-readback" else "shared"});
        if (self.gemma_moe_validation_enabled) {
            log.info("Gemma MoE validation enabled for layer 0 position 0", .{});
        }
        if (self.q8_tg_override) |tg| {
            log.debug("Metal q8_0 threadgroup override: {d}", .{tg});
        }
        if (self.q8_dual_tg_override) |tg| {
            log.debug("Metal q8_0 dual threadgroup override: {d}", .{tg});
        }
        if (self.kv_cache_q8) {
            log.debug("Metal KV cache: q8_0 ({d} B/token, {d} B/head)", .{
                self.kv_cache_bytes_per_token,
                self.kv_cache_head_stride_bytes,
            });
        }
        log.debug(
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
        log.debug(
            "Metal pipeline caps: dmmv_q4k_moe tw={d} max={d} stgmem={d} | dmmv_q5k_moe tw={d} max={d} stgmem={d} | dmmv_q5k_moe_k2048 tw={d} max={d} stgmem={d} | dmmv_q6k_moe tw={d} max={d} stgmem={d} | dmmv_q4k_moe_k2048 tw={d} max={d} stgmem={d} | dmmv_q4k_moe_k2048_1024 tw={d} max={d} stgmem={d}",
            .{
                self.dmmv_q4k_moe_pipe.thread_execution_width,
                self.dmmv_q4k_moe_pipe.max_threads_per_threadgroup,
                self.dmmv_q4k_moe_pipe.static_threadgroup_memory_length,
                self.dmmv_q5k_moe_pipe.thread_execution_width,
                self.dmmv_q5k_moe_pipe.max_threads_per_threadgroup,
                self.dmmv_q5k_moe_pipe.static_threadgroup_memory_length,
                self.dmmv_q5k_moe_k2048_pipe.thread_execution_width,
                self.dmmv_q5k_moe_k2048_pipe.max_threads_per_threadgroup,
                self.dmmv_q5k_moe_k2048_pipe.static_threadgroup_memory_length,
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
        log.debug(
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

    /// Release all GPU buffers, pipelines, and associated resources.
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
        metal_buffer.freeBuffer(&self.logits_readback_buf);
        metal_buffer.freeBuffer(&self.argmax_buf);
        metal_buffer.freeBuffer(&self.embed_staging);
        metal_buffer.freeBuffer(&self.lm_head_private_buf);
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
        metal_buffer.freeBuffer(&self.attn_sinks_buf);
        if (self.attn_sink_values) |vals| {
            for (vals) |v| if (v.len > 0) self.allocator.free(v);
            self.allocator.free(vals);
        }

        metal_pipeline.freePipeline(&self.dmmv_q4k_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q4k_k2048_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q4k_lmhead_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q4k_lmhead_1024_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q5k_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q5k_native_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q6k_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q8_0_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q5_0_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q5_1_pipe);
        metal_pipeline.freePipeline(&self.dmmv_mxfp4_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q8_0_k2048_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q8_0_dual_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q8_0_k2048_fused_norm_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q8_0_dual_fused_norm_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q8_0_repacked_pipe);
        metal_pipeline.freePipeline(&self.dmmv_f16_pipe);
        metal_pipeline.freePipeline(&self.dmmv_f32_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q4k_moe_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q5_1_moe_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q5k_moe_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q5k_moe_k2048_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q6k_moe_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q4k_moe_k2048_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q4k_moe_k2048_1024_pipe);
        metal_pipeline.freePipeline(&self.deinterleave_pipe);
        metal_pipeline.freePipeline(&self.flash_attn_pipe);
        metal_pipeline.freePipeline(&self.flash_attn_q8_pipe);
        metal_pipeline.freePipeline(&self.kv_cache_write_pipe);
        metal_pipeline.freePipeline(&self.kv_cache_write_q8_pipe);
        metal_pipeline.freePipeline(&self.rope_pipe);
        metal_pipeline.freePipeline(&self.rope_native_pipe);
        metal_pipeline.freePipeline(&self.sigmoid_mul_pipe);
        metal_pipeline.freePipeline(&self.geglu_pipe);
        metal_pipeline.freePipeline(&self.geglu_batched_pipe);
        metal_pipeline.freePipeline(&self.swiglu_pipe);
        metal_pipeline.freePipeline(&self.swiglu_batched_pipe);
        metal_pipeline.freePipeline(&self.scale_acc_pipe);
        metal_pipeline.freePipeline(&self.rms_norm_pipe);
        metal_pipeline.freePipeline(&self.rms_norm_offset_pipe);
        metal_pipeline.freePipeline(&self.moe_acc_pipe);
        metal_pipeline.freePipeline(&self.moe_acc_batched_pipe);
        metal_pipeline.freePipeline(&self.softmax_topk_pipe);
        metal_pipeline.freePipeline(&self.softmax_topk_scaled_pipe);
        metal_pipeline.freePipeline(&self.softmax_topk_batched_pipe);
        metal_pipeline.freePipeline(&self.moe_route_pack_pipe);
        metal_pipeline.freePipeline(&self.sigmoid_scale_acc_pipe);
        metal_pipeline.freePipeline(&self.moe_weighted_acc_pipe);
        metal_pipeline.freePipeline(&self.moe_weighted_acc_scaled_pipe);
        metal_pipeline.freePipeline(&self.residual_rms_norm_pipe);
        metal_pipeline.freePipeline(&self.moe_weighted_acc_shared_pipe);
        metal_pipeline.freePipeline(&self.copy_u32_pipe);
        metal_pipeline.freePipeline(&self.copy_f32_pipe);
        metal_pipeline.freePipeline(&self.zero_f32_pipe);
        metal_pipeline.freePipeline(&self.argmax_pipe);
        metal_pipeline.freePipeline(&self.gemm_q4k_pipe);
        metal_pipeline.freePipeline(&self.gemm_q6k_pipe);
        metal_pipeline.freePipeline(&self.flash_attn_batched_pipe);
        metal_pipeline.freePipeline(&self.flash_attn_batched_q8_pipe);
        metal_pipeline.freePipeline(&self.rope_batched_pipe);

        for (0..self.config.n_layers) |i| {
            metal_buffer.freeBuffer(&self.attn_norm_bufs[i]);
            metal_buffer.freeBuffer(&self.attn_q_norm_bufs[i]);
            metal_buffer.freeBuffer(&self.attn_k_norm_bufs[i]);
            metal_buffer.freeBuffer(&self.post_attn_norm_bufs[i]);
            metal_buffer.freeBuffer(&self.ffn_norm_bufs[i]);
            metal_buffer.freeBuffer(&self.post_ffn_norm_bufs[i]);
        }
        self.allocator.free(self.attn_norm_bufs);
        self.allocator.free(self.attn_q_norm_bufs);
        self.allocator.free(self.attn_k_norm_bufs);
        self.allocator.free(self.attn_q_norm_present);
        self.allocator.free(self.attn_k_norm_present);
        self.allocator.free(self.post_attn_norm_bufs);
        self.allocator.free(self.post_attn_norm_present);
        self.allocator.free(self.ffn_norm_bufs);
        self.allocator.free(self.post_ffn_norm_bufs);
        self.allocator.free(self.post_ffn_norm_present);
        self.allocator.free(self.layer_output_scales);
        metal_buffer.freeBuffer(&self.final_norm_gpu);
        metal_buffer.freeBuffer(&self.unit_rms_norm_weights);
        metal_buffer.freeBuffer(&self.rope_freq_buf);
        metal_buffer.freeBuffer(&self.rope_variant_freq_buf);
        if (self.rope_freq_factors) |factors| self.allocator.free(factors);
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
        if (self.private_ssm_qkv_bufs) |bufs| {
            for (bufs) |*buf| metal_buffer.freeBuffer(buf);
            self.allocator.free(bufs);
        }
        if (self.private_ssm_gate_bufs) |bufs| {
            for (bufs) |*buf| metal_buffer.freeBuffer(buf);
            self.allocator.free(bufs);
        }
        if (self.private_ssm_out_bufs) |bufs| {
            for (bufs) |*buf| metal_buffer.freeBuffer(buf);
            self.allocator.free(bufs);
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

        if (self.config.final_logit_softcapping <= 0.0) if (self.argmax_buf.cpu_ptr) |ptr| {
            const argmax_words: [*]const u32 = @ptrCast(@alignCast(ptr));
            return argmax_words[0];
        };

        const logits_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_buf.cpu_ptr.?));
        const logits = logits_ptr[0..self.config.vocab_size];
        var max_val: f32 = -std.math.inf(f32);
        var max_idx: u32 = 0;
        for (logits, 0..) |v, i| {
            const softcapped = softcapLogit(v, self.config.final_logit_softcapping);
            if (softcapped > max_val) {
                max_val = softcapped;
                max_idx = @intCast(i);
            }
        }
        return max_idx;
    }

    /// Sample next token using temperature, top-k, top-p, and repetition penalty.
    /// Falls back to greedy if parameters are near-default or buffers are private.
    pub fn sample(self: *const InferenceEngine, history: []const u32, params: SamplingParams, random: std.Random) u32 {
        if (!params.requiresLogitsReadback()) return self.sampleGreedy();
        if (self.private_decode_buffers) return self.sampleGreedy();
        const logits_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_buf.cpu_ptr.?));
        const logits = logits_ptr[0..self.config.vocab_size];
        return sampleFromLogits(logits, history, params, random, self.config.final_logit_softcapping);
    }

    fn normalizeRequestedContext(self: *const InferenceEngine, requested_context_tokens: u32, minimum_tokens: u32) u32 {
        const floor = if (minimum_tokens > 0) minimum_tokens else @as(u32, 1);
        const desired = if (requested_context_tokens > floor) requested_context_tokens else floor;
        return @min(desired, self.max_context_tokens);
    }

    /// Reset position, profiling counters, and SSM state for a new request.
    pub fn resetRequestState(self: *InferenceEngine, requested_context_tokens: u32) !void {
        _ = self.normalizeRequestedContext(requested_context_tokens, 1);
        self.position = 0;
        self.request_profile.reset();

        if (self.ssm_conv_state_bufs) |bufs| {
            if (self.private_decode_buffers) {
                var cmd = try metal_command.beginCommand(self.device.ctx);
                for (bufs) |buf| {
                    dispatchZeroF32OnCmd(self, &cmd, &buf, @intCast(buf.size / @sizeOf(f32)));
                }
                if (self.ssm_state_bufs) |state_bufs| {
                    for (state_bufs) |buf| {
                        dispatchZeroF32OnCmd(self, &cmd, &buf, @intCast(buf.size / @sizeOf(f32)));
                    }
                }
                cmd.commitAndWait();
            } else {
                for (bufs) |buf| {
                    @memset(buf.cpu_ptr.?[0..buf.size], 0);
                }
            }
        } else if (self.ssm_state_bufs) |bufs| {
            for (bufs) |buf| {
                @memset(buf.cpu_ptr.?[0..buf.size], 0);
            }
        }
    }

    /// Run prompt prefill by replaying the decode path for each prompt token.
    pub fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {
        if (prompt_tokens.len == 0) return;

        const prompt_token_count: u32 = @intCast(@min(prompt_tokens.len, std.math.maxInt(u32)));
        const target_context_tokens = if (state.requested_context_tokens > 0)
            @max(state.requested_context_tokens, state.position +| prompt_token_count)
        else
            state.position +| prompt_token_count;
        if (target_context_tokens > self.max_context_tokens) return error.ContextLengthExceeded;

        if (state.position == 0 and state.generated_tokens.items.len == 0) {
            try self.resetRequestState(target_context_tokens);
            state.position = 0;
            state.generated_tokens.clearRetainingCapacity();
        } else if (state.position != self.position) {
            return error.KvStateNotAvailable;
        }

        for (prompt_tokens, 0..) |token_id, i| {
            try self.loadTokenEmbedding(token_id);
            try runDecodeStep(self, i + 1 == prompt_tokens.len);
        }
        state.position = self.position;
    }

    /// Experimental batched prompt prefill gated by `ZINC_BATCHED_PREFILL`.
    ///
    /// Processes all prompt tokens in a single batched forward pass using the
    /// gemm_q4k / gemm_q6k / rope_batched / flash_attn_batched shaders — the
    /// weight matrix for each projection is read once for the whole prompt.
    /// Falls back to the per-token `prefillBatch` when the env flag is off or
    /// when the model architecture is outside the supported slice (see
    /// `canUseBatchedPrefill`). Supports both fresh prefill (state.position==0)
    /// and prefix reuse (state.position>0) — in the latter case, the batched
    /// pass extends the KV cache at offset `state.position` and flash attention
    /// causal masking is computed relative to that offset. With
    /// `ZINC_BATCHED_PREFILL=validate` the batched path runs first, then the
    /// per-token path is replayed on a fresh state and the last-token logits
    /// are diffed; max abs diff is logged and a warning is emitted if it
    /// exceeds 1e-3.
    pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {
        if (prompt_tokens.len == 0) return;
        const mode = batchedPrefillMode();
        if (mode == .off or !canUseBatchedPrefill(self)) {
            return self.prefillBatch(state, prompt_tokens);
        }
        if (state.position != 0 and state.position != self.position) {
            return error.KvStateNotAvailable;
        }
        // validate mode requires a fresh state so we can replay the per-token
        // path against the batched snapshot — delegate extension calls.
        if (mode == .validate and state.position != 0) {
            return self.prefillBatch(state, prompt_tokens);
        }

        const n_tokens: u32 = @intCast(@min(prompt_tokens.len, std.math.maxInt(u32)));
        const position_base: u32 = state.position;
        const target_context_tokens = if (state.requested_context_tokens > 0)
            @max(state.requested_context_tokens, position_base +| n_tokens)
        else
            position_base +| n_tokens;
        if (target_context_tokens > self.max_context_tokens) return error.ContextLengthExceeded;

        const cfg = self.config;
        const hidden_dim = cfg.hidden_dim;
        const inter_dim: u32 = if (cfg.intermediate_dim > 0) cfg.intermediate_dim else hidden_dim * 4;
        const attn = try resolveLayerAttentionParams(cfg, self.layer_tensors[0], hidden_dim, self.kv_cache_q8);

        if (position_base == 0 and state.generated_tokens.items.len == 0) {
            try self.resetRequestState(target_context_tokens);
            state.position = 0;
            state.generated_tokens.clearRetainingCapacity();
        }

        var scratch = try BatchedPrefillScratch.init(self, n_tokens, attn.q_dim, attn.kv_dim, inter_dim);
        defer scratch.deinit();

        // Pre-dequantize embeddings for the whole prompt into scratch.hidden.
        {
            const mmap = self.model.mmap_data orelse return error.NoMmapData;
            const embed_offset = self.model.gguf_file.tensor_data_offset + self.token_embed.info.offset;
            const embed_raw = mmap[embed_offset..];
            const hidden_ptr: [*]f32 = @ptrCast(@alignCast(scratch.hidden.cpu_ptr.?));
            for (prompt_tokens, 0..) |token_id, t| {
                const out_slice = hidden_ptr[t * hidden_dim .. (t + 1) * hidden_dim];
                dequantRow(embed_raw, token_id, hidden_dim, self.token_embed.info.type_, out_slice);
            }
        }

        var cmd = try metal_command.beginCommand(self.device.ctx);

        for (0..cfg.n_layers) |layer_idx| {
            const lt = self.layer_tensors[layer_idx];
            const q_t = lt.attn_q.?;
            const k_t = lt.attn_k.?;
            const v_t = lt.attn_v.?;
            const o_t = lt.attn_output.?;
            const gate_t = lt.ffn_gate.?;
            const up_t = lt.ffn_up.?;
            const down_t = lt.ffn_down.?;

            // === Attention block ===
            dispatchRmsNormOnCmd(self, &cmd, &scratch.hidden, &scratch.norm, &self.attn_norm_bufs[layer_idx], hidden_dim, n_tokens);
            cmd.barrier();

            dispatchGemmBatchedOnCmd(self, &cmd, q_t, &scratch.norm, &scratch.q, attn.q_dim, hidden_dim, n_tokens);
            dispatchGemmBatchedOnCmd(self, &cmd, k_t, &scratch.norm, &scratch.k, attn.kv_dim, hidden_dim, n_tokens);
            dispatchGemmBatchedOnCmd(self, &cmd, v_t, &scratch.norm, &scratch.v, attn.kv_dim, hidden_dim, n_tokens);
            cmd.barrier();

            // Per-head Q/K RMSNorms (Qwen3 and similar). Each head-slice of a token
            // is one workgroup; batching is just n_tokens × n_heads workgroups.
            if (self.attn_q_norm_present[layer_idx]) {
                dispatchRmsNormOnCmd(self, &cmd, &scratch.q, &scratch.q, &self.attn_q_norm_bufs[layer_idx], attn.head_dim, cfg.n_heads * n_tokens);
            }
            if (self.attn_k_norm_present[layer_idx]) {
                dispatchRmsNormOnCmd(self, &cmd, &scratch.k, &scratch.k, &self.attn_k_norm_bufs[layer_idx], attn.head_dim, attn.n_kv_heads * n_tokens);
            }
            if (self.attn_q_norm_present[layer_idx] or self.attn_k_norm_present[layer_idx]) {
                cmd.barrier();
            }

            const rope_freq_buf = selectRopeFreqBuffer(self, attn.rope_dim, attn.rope_freq_base, attn.use_rope_freq_factors);
            dispatchRopeBatchedOnCmd(self, &cmd, &scratch.q, &scratch.q, rope_freq_buf, attn.head_dim, attn.rope_dim, cfg.n_heads, position_base, n_tokens, attn.rope_freq_base, attn.use_rope_freq_factors, 1.0);
            dispatchRopeBatchedOnCmd(self, &cmd, &scratch.k, &scratch.k, rope_freq_buf, attn.head_dim, attn.rope_dim, attn.n_kv_heads, position_base, n_tokens, attn.rope_freq_base, attn.use_rope_freq_factors, 1.0);
            cmd.barrier();

            const kv_len = position_base + n_tokens;
            if (self.kv_cache_q8) {
                const n_blocks = n_tokens * (attn.kv_dim / 32);
                const dst_bytes = position_base * attn.kv_cache_bytes_per_token;
                dispatchKvCacheWriteBatchedQ8OnCmd(self, &cmd, layer_idx, &scratch.k, &scratch.v, dst_bytes, n_blocks);
            } else {
                const dst_elements = position_base * attn.kv_dim;
                dispatchKvCacheWriteBatchedOnCmd(self, &cmd, layer_idx, &scratch.k, &scratch.v, dst_elements, n_tokens * attn.kv_dim);
            }
            cmd.barrier();

            if (self.kv_cache_q8) {
                dispatchFlashAttnBatchedQ8OnCmd(
                    self,
                    &cmd,
                    &scratch.q,
                    &self.kv_k_cache[layer_idx],
                    &self.kv_v_cache[layer_idx],
                    &scratch.attn_out,
                    attn.head_dim,
                    cfg.n_heads,
                    attn.n_kv_heads,
                    kv_len,
                    n_tokens,
                    position_base,
                    attn.kv_cache_head_stride_bytes,
                    attn.kv_cache_bytes_per_token,
                );
            } else {
                dispatchFlashAttnBatchedOnCmd(
                    self,
                    &cmd,
                    &scratch.q,
                    &self.kv_k_cache[layer_idx],
                    &self.kv_v_cache[layer_idx],
                    &scratch.attn_out,
                    attn.head_dim,
                    cfg.n_heads,
                    attn.n_kv_heads,
                    kv_len,
                    n_tokens,
                    position_base,
                );
            }
            cmd.barrier();

            dispatchGemmBatchedOnCmd(self, &cmd, o_t, &scratch.attn_out, &scratch.down, hidden_dim, attn.q_dim, n_tokens);
            cmd.barrier();

            // Fused residual-add + FFN-norm over all N tokens: eliminates
            // separate scale_acc → barrier → rms_norm dispatches per layer.
            {
                const push = ResidualRmsNormPush{ .n = hidden_dim, .eps = cfg.rms_norm_eps, .scale = 1.0 };
                const bufs = [_]*const MetalBuffer{ &scratch.hidden, &scratch.down, &scratch.norm, &self.ffn_norm_bufs[layer_idx] };
                cmd.dispatchV2(&self.residual_rms_norm_pipe, .{ n_tokens, 1, 1 }, .{ 256, 1, 1 }, &bufs, &push, @sizeOf(ResidualRmsNormPush), 0);
            }
            cmd.barrier();

            dispatchGemmBatchedOnCmd(self, &cmd, gate_t, &scratch.norm, &scratch.gate, inter_dim, hidden_dim, n_tokens);
            dispatchGemmBatchedOnCmd(self, &cmd, up_t, &scratch.norm, &scratch.up, inter_dim, hidden_dim, n_tokens);
            cmd.barrier();

            // Batched SwiGLU: grid.x covers inter_dim, grid.y is the token index.
            {
                const push = SwiGLUPush{ .n = inter_dim };
                const bufs = [_]*const MetalBuffer{ &scratch.gate, &scratch.swiglu, &scratch.up };
                cmd.dispatchV2(&self.swiglu_batched_pipe, .{ (inter_dim + 63) / 64, n_tokens, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SwiGLUPush), 0);
            }
            cmd.barrier();

            dispatchGemmBatchedOnCmd(self, &cmd, down_t, &scratch.swiglu, &scratch.down, hidden_dim, inter_dim, n_tokens);
            cmd.barrier();

            {
                const total = n_tokens * hidden_dim;
                const push = ScaleAccPush{ .n = total, .scale_bits = @as(u32, @bitCast(@as(f32, 1.0))) };
                const bufs = [_]*const MetalBuffer{ &scratch.hidden, &scratch.down };
                cmd.dispatchV2(&self.scale_acc_pipe, .{ (total + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(ScaleAccPush), 0);
            }
            cmd.barrier();
        }

        // Final RMSNorm over all N tokens into scratch.norm. The last token is
        // what feeds the LM head.
        dispatchRmsNormOnCmd(self, &cmd, &scratch.hidden, &scratch.norm, &self.final_norm_gpu, hidden_dim, n_tokens);
        cmd.barrier();

        // LM head on the last token via DmmvPush.x_offset (bytes into scratch.norm).
        if (shouldCpuLmHeadFallback(self)) {
            // Rare quant path (Gemma Q8). Fall back to the CPU LM head.
            cmd.commitAndWait();
            const src_base = @as(usize, n_tokens - 1) * hidden_dim;
            const src_ptr: [*]const f32 = @ptrCast(@alignCast(scratch.hidden.cpu_ptr.?));
            const dst_ptr: [*]f32 = @ptrCast(@alignCast(self.hidden_buf.cpu_ptr.?));
            @memcpy(dst_ptr[0..hidden_dim], src_ptr[src_base .. src_base + hidden_dim]);
            var final_cmd = try metal_command.beginCommand(self.device.ctx);
            dispatchRmsNormOnCmd(self, &final_cmd, &self.hidden_buf, &self.norm_buf, &self.final_norm_gpu, hidden_dim, 1);
            final_cmd.commitAndWait();
            const mmap = self.model.mmap_data orelse return error.NoMmapData;
            const tdo = self.model.gguf_file.tensor_data_offset;
            const in_ptr: [*]const f32 = @ptrCast(@alignCast(self.norm_buf.cpu_ptr.?));
            const out_ptr: [*]f32 = @ptrCast(@alignCast(self.logits_buf.cpu_ptr.?));
            try cpuDmmvFallback(mmap, self.lm_head, tdo, in_ptr, out_ptr, cfg.vocab_size, hidden_dim, 0, self.allocator);
            writeCpuArgmax(self, out_ptr, cfg.vocab_size);
        } else {
            const x_offset_bytes: u32 = (n_tokens - 1) * hidden_dim * @sizeOf(f32);
            dispatchLmHeadWithInputOffset(self, &cmd, &scratch.norm, &self.logits_buf, hidden_dim, cfg.vocab_size, x_offset_bytes);
            cmd.barrier();
            dispatchArgmaxOnCmd(self, &cmd, &self.logits_buf, &self.argmax_buf, cfg.vocab_size);
            cmd.commitAndWait();
            // Keep engine.hidden_buf consistent with the advancing position so
            // any subsequent single-token decodeStep sees the right residual.
            const src_base = @as(usize, n_tokens - 1) * hidden_dim;
            const src_ptr: [*]const f32 = @ptrCast(@alignCast(scratch.hidden.cpu_ptr.?));
            const dst_ptr: [*]f32 = @ptrCast(@alignCast(self.hidden_buf.cpu_ptr.?));
            @memcpy(dst_ptr[0..hidden_dim], src_ptr[src_base .. src_base + hidden_dim]);
        }

        self.position = position_base + n_tokens;
        state.position = self.position;

        if (mode == .validate) {
            // Snapshot batched-path logits, then re-run the per-token path on
            // a fresh state and diff. The per-token result becomes authoritative
            // so any subsequent decode steps continue from the trusted state.
            const vocab = cfg.vocab_size;
            const batched_snapshot = try self.allocator.alloc(f32, vocab);
            defer self.allocator.free(batched_snapshot);
            const batched_logits: [*]const f32 = @ptrCast(@alignCast(self.logits_buf.cpu_ptr.?));
            @memcpy(batched_snapshot, batched_logits[0..vocab]);

            self.position = 0;
            state.position = 0;
            state.generated_tokens.clearRetainingCapacity();
            try self.prefillBatch(state, prompt_tokens);

            const ref_logits: [*]const f32 = @ptrCast(@alignCast(self.logits_buf.cpu_ptr.?));
            var max_abs: f32 = 0;
            var max_idx: usize = 0;
            for (0..vocab) |i| {
                const diff = @abs(ref_logits[i] - batched_snapshot[i]);
                if (diff > max_abs) {
                    max_abs = diff;
                    max_idx = i;
                }
            }
            const tol: f32 = 1e-3;
            const level: enum { ok, exceeded } = if (max_abs > tol) .exceeded else .ok;
            log.warn("prefillBatched validate[{s}]: last-token logits max_abs_diff={d:.6} at idx={d} (ref={d:.4} batched={d:.4}) tol={d:.6} n_tokens={d}", .{
                @tagName(level), max_abs, max_idx, ref_logits[max_idx], batched_snapshot[max_idx], tol, n_tokens,
            });
        }
    }

    /// Advance one autoregressive decode step from the given input token.
    pub fn decodeStep(self: *InferenceEngine, state: *DecodeState, token_id: u32) !void {
        if (state.position >= self.max_context_tokens) return error.ContextLengthExceeded;
        if (state.position != self.position) return error.KvStateNotAvailable;
        const next_token_target = if (state.requested_context_tokens > 0)
            @max(state.requested_context_tokens, state.position + 1)
        else
            state.position + 1;
        if (next_token_target > self.max_context_tokens) return error.ContextLengthExceeded;
        try self.loadTokenEmbedding(token_id);
        try runDecodeStep(self, true);
        state.position = self.position;
    }

    /// Enable request-level profiling counters for subsequent decode work.
    pub fn enableProfiling(self: *InferenceEngine) !void {
        self.profile_enabled = true;
        self.request_profile.reset();
    }

    /// Log the collected Metal profiling summary for the current request.
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
        log.info("  wait: commitAndWait {d:.2} ms ({d:.3} ms/step, {d:.1}% of traced time; includes queued GPU work + CPU wait) | record breakdown layer {d:.2} ms gpu-moe {d:.2} ms fallback-moe {d:.2} ms dense {d:.2} ms final {d:.2} ms", .{
            nsToMs(profile.gpu_completion_wait_ns),
            avgMs(profile.gpu_completion_wait_ns, profile.decode_steps),
            pctOf(traced_request_ns, profile.gpu_completion_wait_ns),
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
        if (profile.dmmv_total_bytes > 0) {
            log.info("  dmmv bytes: q8_0 {d:.2} GiB ({d:.1}%) q4_k {d:.2} GiB ({d:.1}%) q5_k {d:.2} GiB ({d:.1}%) q6_k {d:.2} GiB ({d:.1}%)", .{
                bytesToGiB(profile.dmmv_q8_0_bytes),
                pctOf(profile.dmmv_total_bytes, profile.dmmv_q8_0_bytes),
                bytesToGiB(profile.dmmv_q4k_bytes),
                pctOf(profile.dmmv_total_bytes, profile.dmmv_q4k_bytes),
                bytesToGiB(profile.dmmv_q5k_bytes),
                pctOf(profile.dmmv_total_bytes, profile.dmmv_q5k_bytes),
                bytesToGiB(profile.dmmv_q6k_bytes),
                pctOf(profile.dmmv_total_bytes, profile.dmmv_q6k_bytes),
            });
            log.info("  path bytes: ssm {d:.2} GiB attn {d:.2} GiB moe-expert {d:.2} GiB shared {d:.2} GiB lm-head {d:.2} GiB router {d:.2} GiB", .{
                bytesToGiB(profile.ssm_bytes),
                bytesToGiB(profile.full_attn_bytes),
                bytesToGiB(profile.moe_expert_bytes),
                bytesToGiB(profile.shared_expert_bytes),
                bytesToGiB(profile.lm_head_bytes),
                bytesToGiB(profile.router_bytes),
            });
            if (profile.dmmv_q8_0_bytes > 0) {
                var top_idxs: [4]?usize = .{ null, null, null, null };
                for (profile.q8_shape_stats, 0..) |slot, idx| {
                    if (slot.calls == 0) continue;
                    var insert_pos: ?usize = null;
                    for (top_idxs, 0..) |maybe_top_idx, rank| {
                        if (maybe_top_idx) |top_idx| {
                            if (slot.bytes > profile.q8_shape_stats[top_idx].bytes) {
                                insert_pos = rank;
                                break;
                            }
                        } else {
                            insert_pos = rank;
                            break;
                        }
                    }
                    if (insert_pos) |rank| {
                        var shift: usize = top_idxs.len - 1;
                        while (shift > rank) : (shift -= 1) {
                            top_idxs[shift] = top_idxs[shift - 1];
                        }
                        top_idxs[rank] = idx;
                    }
                }
                for (top_idxs, 0..) |maybe_idx, rank| {
                    if (maybe_idx) |idx| {
                        const slot = profile.q8_shape_stats[idx];
                        log.info("  q8 hot #{d}: {s} M={d} K={d} bytes={d:.2} GiB calls={d}", .{
                            rank + 1,
                            dmmvPathLabel(slot.path),
                            slot.rows,
                            slot.cols,
                            bytesToGiB(slot.bytes),
                            slot.calls,
                        });
                    }
                }
            }
        }
        if (profile.gpu_routed_moe_layers == 0 and profile.fallback_moe_layers > 0 and self.layer_tensors.len > 0) {
            const layer0 = self.layer_tensors[0];
            log.info("  fallback-moe path: gate_exps={s} up_exps={s} down_exps={s} (GPU-routed path currently supports q4_k/q4_k/{{q4_k,q5_1,q5_k,q6_k}})", .{
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
        const dst_buf = if (self.private_decode_buffers) &self.embed_staging else &self.hidden_buf;
        const hidden_ptr: [*]f32 = @ptrCast(@alignCast(dst_buf.cpu_ptr.?));
        dequantRow(embed_raw, token_id, self.config.hidden_dim, self.token_embed.info.type_, hidden_ptr[0..self.config.hidden_dim]);
        // Gemma models scale embeddings by sqrt(hidden_dim). Keep parity when
        // config.architecture == .gemma.
        if (self.config.architecture == .gemma) {
            const scale = @as(f32, @floatCast(@sqrt(@as(f64, @floatFromInt(self.config.hidden_dim)))));
            for (hidden_ptr[0..self.config.hidden_dim]) |*value| value.* *= scale;
        }
    }

    /// Get the DMMV pipeline, push constant buffer index, rows-per-workgroup, and block size.
    /// Q4_K: native Metal kernel — 32 threads (1 simdgroup) per row, 4 rows per threadgroup (64 threads).
    /// Q4_K wide: specialized large-M kernel — 16 rows per threadgroup (512 threads).
    /// Q4_K LM head 1024: dedicated vocab projection kernel — 32 rows per threadgroup (1024 threads).
    /// On Apple9/M4, the 1024-thread shape tends to trade away too much
    /// occupancy for reuse, so keep that path reserved for Apple10-class parts.
    /// Reusing the wider shape outside the LM head improves staged-vector reuse on
    /// the large decode-side Q4_K projections that still dominate token time.
    /// Q5_K/Q6_K/F32: SPIRV-Cross — each thread handles 1 row (64 rows per workgroup, 64 threads).
    /// Q8_0/F16: SPIRV-Cross — each workgroup handles 2 rows (64 threads cooperate via simd_sum).
    fn dmmvPipelineForType(
        self: *const InferenceEngine,
        tensor: *const metal_loader.LoadedTensor,
        M: u32,
        K: u32,
    ) ?struct { pipe: *const MetalPipeline, push_idx: u32, rows_per_wg: u32, block_size: u32 } {
        if (tensor.info.type_ == .q8_0 and shouldCpuQ8Fallback(self.config.architecture, tensor.info.name)) {
            return null;
        }
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
                    break :blk .{ .pipe = &self.dmmv_q4k_k2048_pipe, .push_idx = 1, .rows_per_wg = 4, .block_size = 64 };
                }
                break :blk .{ .pipe = &self.dmmv_q4k_pipe, .push_idx = 1, .rows_per_wg = 4, .block_size = 64 };
            },
            .q5_0 => .{ .pipe = &self.dmmv_q5_0_pipe, .push_idx = 0, .rows_per_wg = 2, .block_size = 64 },
            .q5_1 => .{ .pipe = &self.dmmv_q5_1_pipe, .push_idx = 0, .rows_per_wg = 2, .block_size = 64 },
            .mxfp4 => .{ .pipe = &self.dmmv_mxfp4_pipe, .push_idx = 0, .rows_per_wg = 64, .block_size = 64 },
            .q5_k => .{ .pipe = &self.dmmv_q5k_pipe, .push_idx = 0, .rows_per_wg = 64, .block_size = 64 },
            .q6_k => .{ .pipe = &self.dmmv_q6k_pipe, .push_idx = 0, .rows_per_wg = 64, .block_size = 64 },
            .q8_0 => blk: {
                const simd_width = if (self.dmmv_q8_0_pipe.thread_execution_width > 0) self.dmmv_q8_0_pipe.thread_execution_width else @as(u32, 32);
                if (self.device.chip == .apple9 and simd_width == 32) {
                    if (preferApple9Q8K2048Path(tensor, M, K) and
                        self.dmmv_q8_0_k2048_pipe.thread_execution_width == 32 and
                        self.dmmv_q8_0_k2048_pipe.max_threads_per_threadgroup >= 512)
                    {
                        // nr=2: each SG processes 2 rows
                        break :blk .{ .pipe = &self.dmmv_q8_0_k2048_pipe, .push_idx = 0, .rows_per_wg = 32, .block_size = 512 };
                    }
                    if (preferApple9Q8WidePath(tensor, M, K) and self.dmmv_q8_0_pipe.max_threads_per_threadgroup >= 512) {
                        break :blk .{ .pipe = &self.dmmv_q8_0_pipe, .push_idx = 0, .rows_per_wg = 32, .block_size = 512 };
                    }
                    // lm_head (M=248320 for Qwen3.5-35B): use 512-thread wide path.
                    // Adapted from llama.cpp's Q8_0 mul_mat_vec with N_R0=2 (nr=2).
                    if (K <= 2048 and M >= 65536 and tensor == self.lm_head and
                        self.dmmv_q8_0_k2048_pipe.max_threads_per_threadgroup >= 512)
                    {
                        break :blk .{ .pipe = &self.dmmv_q8_0_k2048_pipe, .push_idx = 0, .rows_per_wg = 32, .block_size = 512 };
                    }
                }
                if (self.q8_tg_override) |block_size| {
                    if (!shouldUseGlobalQ8Override(self.config.architecture, tensor.info.name)) {
                        break :blk .{ .pipe = &self.dmmv_q8_0_pipe, .push_idx = 0, .rows_per_wg = 16, .block_size = 256 };
                    }
                    // nr=2: double the rows per workgroup
                    break :blk .{ .pipe = &self.dmmv_q8_0_pipe, .push_idx = 0, .rows_per_wg = block_size / simd_width * 2, .block_size = block_size };
                }
                if (K <= 2048 and
                    M >= 1024 and
                    self.device.chip.isM5Class() and
                    self.dmmv_q8_0_k2048_pipe.thread_execution_width == 32 and
                    self.dmmv_q8_0_k2048_pipe.max_threads_per_threadgroup >= 1024)
                {
                    break :blk .{ .pipe = &self.dmmv_q8_0_k2048_pipe, .push_idx = 0, .rows_per_wg = 64, .block_size = 1024 };
                }
                if (K <= 2048 and
                    self.dmmv_q8_0_k2048_pipe.thread_execution_width == 32 and
                    self.dmmv_q8_0_k2048_pipe.max_threads_per_threadgroup >= 256)
                {
                    break :blk .{ .pipe = &self.dmmv_q8_0_k2048_pipe, .push_idx = 0, .rows_per_wg = 16, .block_size = 256 };
                }
                if (K <= 4096 and
                    M >= 1024 and
                    self.device.chip.isM5Class() and
                    self.dmmv_q8_0_pipe.thread_execution_width == 32 and
                    self.dmmv_q8_0_pipe.max_threads_per_threadgroup >= 1024)
                {
                    break :blk .{ .pipe = &self.dmmv_q8_0_pipe, .push_idx = 0, .rows_per_wg = 64, .block_size = 1024 };
                }
                if (K <= 4096 and
                    self.dmmv_q8_0_pipe.thread_execution_width == 32 and
                    self.dmmv_q8_0_pipe.max_threads_per_threadgroup >= 256)
                {
                    break :blk .{ .pipe = &self.dmmv_q8_0_pipe, .push_idx = 0, .rows_per_wg = 16, .block_size = 256 };
                }
                break :blk .{ .pipe = &self.dmmv_q8_0_pipe, .push_idx = 0, .rows_per_wg = 4, .block_size = 64 };
            },
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
    _ = model;
    return tensor.buffer_offset;
}

/// Repack Q8_0 blocks into a SIMD-coalesced layout for efficient GPU reads.
///
/// Standard Q8_0: each 34-byte block is [2B scale | 32B qs]. When each SIMD
/// lane reads a separate block, the stride-34 access wastes ~88% of bandwidth.
///
/// Repacked: groups of 32 blocks are interleaved so adjacent lanes read
/// adjacent bytes:
///   [0..63]     32 × half scales
///   [64..191]   chunk 0: 32 × char4 qs[0..3]
///   [192..319]  chunk 1: 32 × char4 qs[4..7]  ...etc
///   Total: 1088 bytes (same as 32 × 34)
///
/// Requires: blocks_per_row is a multiple of 32 (K % 1024 == 0).
fn repackQ8_0Blocks(src: [*]const u8, dst: [*]u8, M: u32, K: u32) void {
    const blocks_per_row = K / 32;
    const groups_per_row = blocks_per_row / 32;
    const block_bytes: u32 = 34;
    const group_bytes: u32 = 32 * block_bytes; // 1088

    for (0..M) |row| {
        for (0..groups_per_row) |gi| {
            const src_group: usize = (row * blocks_per_row + gi * 32) * block_bytes;
            const dst_group: usize = (row * groups_per_row + gi) * group_bytes;

            // Pack 32 scales (2 bytes each) into contiguous 64 bytes
            for (0..32) |bi| {
                const so = src_group + bi * block_bytes;
                dst[dst_group + bi * 2] = src[so];
                dst[dst_group + bi * 2 + 1] = src[so + 1];
            }

            // Pack qs: 8 chunks of 32 × 4 bytes = 128 bytes each
            for (0..8) |vi| {
                for (0..32) |bi| {
                    const so = src_group + bi * block_bytes + 2 + vi * 4;
                    const do_ = dst_group + 64 + vi * 128 + bi * 4;
                    dst[do_] = src[so];
                    dst[do_ + 1] = src[so + 1];
                    dst[do_ + 2] = src[so + 2];
                    dst[do_ + 3] = src[so + 3];
                }
            }
        }
    }
}

/// Check whether a Q8_0 tensor can use the repacked kernel (K must be a multiple of 1024).
fn canRepackQ8(K: u32) bool {
    return K >= 1024 and K % 1024 == 0;
}

fn dmmvWeightBytes(quant_type: GGMLType, rows: u32, cols: u32) u64 {
    const bs = quant_type.blockSize();
    const bpb = quant_type.bytesPerBlock();
    if (bs == 0 or bpb == 0) return @as(u64, rows) * @as(u64, cols) * 4;
    const blocks_per_row = cols / bs;
    return @as(u64, rows) * @as(u64, blocks_per_row) * @as(u64, bpb);
}

fn recordDispatchQuantBytes(profile: *RuntimeProfile, quant_type: GGMLType, bytes: u64) void {
    profile.dmmv_total_bytes += bytes;
    switch (quant_type) {
        .q4_k => profile.dmmv_q4k_bytes += bytes,
        .q5_k => profile.dmmv_q5k_bytes += bytes,
        .q6_k => profile.dmmv_q6k_bytes += bytes,
        .q8_0 => profile.dmmv_q8_0_bytes += bytes,
        .f16 => profile.dmmv_f16_bytes += bytes,
        .f32 => profile.dmmv_f32_bytes += bytes,
        else => {},
    }
}

fn classifyDmmvPath(engine: *InferenceEngine, tensor: *const metal_loader.LoadedTensor) DmmvPathClass {
    if (tensor == engine.lm_head) return .lm_head;

    const name = tensor.info.name;
    if (std.mem.endsWith(u8, name, "ffn_gate_inp.weight")) return .router;
    if (std.mem.indexOf(u8, name, "_shexp.")) |_| return .shared_expert;
    if (std.mem.endsWith(u8, name, "attn_qkv.weight") or
        std.mem.endsWith(u8, name, "attn_gate.weight") or
        std.mem.endsWith(u8, name, "ssm_alpha.weight") or
        std.mem.endsWith(u8, name, "ssm_beta.weight") or
        std.mem.endsWith(u8, name, "ssm_out.weight"))
    {
        return .ssm;
    }
    if (std.mem.endsWith(u8, name, "attn_q.weight") or
        std.mem.endsWith(u8, name, "attn_k.weight") or
        std.mem.endsWith(u8, name, "attn_v.weight") or
        std.mem.endsWith(u8, name, "attn_output.weight"))
    {
        return .full_attn;
    }
    if (std.mem.endsWith(u8, name, "ffn_gate.weight") or
        std.mem.endsWith(u8, name, "ffn_up.weight") or
        std.mem.endsWith(u8, name, "ffn_down.weight"))
    {
        return .dense_ffn;
    }
    return .other;
}

fn recordQ8ShapeProfile(
    profile: *RuntimeProfile,
    path: DmmvPathClass,
    rows: u32,
    cols: u32,
    bytes: u64,
) void {
    for (&profile.q8_shape_stats) |*slot| {
        if (slot.calls != 0 and slot.path == path and slot.rows == rows and slot.cols == cols) {
            slot.bytes += bytes;
            slot.calls += 1;
            return;
        }
    }
    for (&profile.q8_shape_stats) |*slot| {
        if (slot.calls == 0) {
            slot.* = .{
                .path = path,
                .rows = rows,
                .cols = cols,
                .bytes = bytes,
                .calls = 1,
            };
            return;
        }
    }
}

fn recordDmmvProfile(
    engine: *InferenceEngine,
    tensor: *const metal_loader.LoadedTensor,
    rows: u32,
    cols: u32,
) void {
    if (!engine.profile_enabled) return;

    const bytes = dmmvWeightBytes(tensor.info.type_, rows, cols);
    var profile = &engine.request_profile;
    recordDispatchQuantBytes(profile, tensor.info.type_, bytes);
    const path = classifyDmmvPath(engine, tensor);
    switch (path) {
        .lm_head => profile.lm_head_bytes += bytes,
        .router => profile.router_bytes += bytes,
        .shared_expert => profile.shared_expert_bytes += bytes,
        .ssm => profile.ssm_bytes += bytes,
        .full_attn => profile.full_attn_bytes += bytes,
        .dense_ffn => profile.dense_ffn_bytes += bytes,
        else => {},
    }
    if (tensor.info.type_ == .q8_0) {
        recordQ8ShapeProfile(profile, path, rows, cols, bytes);
    }
}

fn recordMoeDmmvProfile(
    engine: *InferenceEngine,
    quant_type: GGMLType,
    rows: u32,
    cols: u32,
    expert_count: u32,
) void {
    if (!engine.profile_enabled) return;

    const bytes = @as(u64, expert_count) * dmmvWeightBytes(quant_type, rows, cols);
    var profile = &engine.request_profile;
    recordDispatchQuantBytes(profile, quant_type, bytes);
    profile.moe_expert_bytes += bytes;
}

// ---------------------------------------------------------------------------
// DMMV dispatch helpers
// ---------------------------------------------------------------------------

fn dispatchCopyF32OnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    src_buf: *const MetalBuffer,
    dst_buf: *const MetalBuffer,
    n: u32,
) void {
    if (n == 0) return;
    const push = CopyF32Push{ .n = n };
    const bufs = [_]*const MetalBuffer{ src_buf, dst_buf };
    cmd.dispatchV2(&engine.copy_f32_pipe, .{ (n + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, &bufs, &push, @sizeOf(CopyF32Push), 2);
}

fn dispatchCopyU32OnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    src_buf: *const MetalBuffer,
    dst_buf: *const MetalBuffer,
    n_words: u32,
    src_offset_words: u32,
    dst_offset_words: u32,
) void {
    if (n_words == 0) return;
    const push = CopyU32Push{
        .n_words = n_words,
        .src_offset_words = src_offset_words,
        .dst_offset_words = dst_offset_words,
    };
    const bufs = [_]*const MetalBuffer{ src_buf, dst_buf };
    cmd.dispatchV2(&engine.copy_u32_pipe, .{ (n_words + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, &bufs, &push, @sizeOf(CopyU32Push), 2);
}

fn dispatchZeroF32OnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    dst_buf: *const MetalBuffer,
    n: u32,
) void {
    if (n == 0) return;
    const push = ZeroF32Push{ .n = n };
    const bufs = [_]*const MetalBuffer{dst_buf};
    cmd.dispatchV2(&engine.zero_f32_pipe, .{ (n + 255) / 256, 1, 1 }, .{ 256, 1, 1 }, &bufs, &push, @sizeOf(ZeroF32Push), 1);
}

fn dispatchArgmaxOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    logits_buf: *const MetalBuffer,
    output_buf: *const MetalBuffer,
    n: u32,
) void {
    if (n == 0) return;
    const push = ArgmaxPush{ .n = n };
    const bufs = [_]*const MetalBuffer{ logits_buf, output_buf };
    cmd.dispatchV2(&engine.argmax_pipe, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &bufs, &push, @sizeOf(ArgmaxPush), 2);
}

fn writeCpuArgmax(engine: *const InferenceEngine, logits: [*]const f32, n: u32) void {
    if (n == 0) return;
    const ptr = engine.argmax_buf.cpu_ptr orelse return;

    var max_val: f32 = logits[0];
    var max_idx: u32 = 0;
    for (logits[0..n], 0..) |value, i| {
        if (value > max_val) {
            max_val = value;
            max_idx = @intCast(i);
        }
    }

    const argmax_words: [*]u32 = @ptrCast(@alignCast(ptr));
    argmax_words[0] = max_idx;
    argmax_words[1] = 0;
}

fn canUseFusedNormQ8Dmmv(
    engine: *const InferenceEngine,
    tensor: *const metal_loader.LoadedTensor,
    K: u32,
) bool {
    return tensor.info.type_ == .q8_0 and
        K <= 2048 and
        engine.dmmv_q8_0_k2048_fused_norm_pipe.thread_execution_width == 32 and
        engine.dmmv_q8_0_k2048_fused_norm_pipe.max_threads_per_threadgroup >= 256;
}

fn canUseFusedNormDualQ8Dmmv(
    engine: *const InferenceEngine,
    tensor0: *const metal_loader.LoadedTensor,
    tensor1: *const metal_loader.LoadedTensor,
    M0: u32,
    M1: u32,
    K: u32,
) bool {
    const block_size = engine.q8_dual_tg_override orelse 1024;
    return tensor0.info.type_ == .q8_0 and
        tensor1.info.type_ == .q8_0 and
        K <= 2048 and
        M0 > 0 and
        M1 > 0 and
        engine.dmmv_q8_0_dual_fused_norm_pipe.thread_execution_width == 32 and
        engine.dmmv_q8_0_dual_fused_norm_pipe.max_threads_per_threadgroup >= block_size;
}

fn canUseDualQ8Dmmv(
    engine: *const InferenceEngine,
    tensor0: *const metal_loader.LoadedTensor,
    tensor1: *const metal_loader.LoadedTensor,
    M0: u32,
    M1: u32,
    K: u32,
) bool {
    const block_size = engine.q8_dual_tg_override orelse 1024;
    return tensor0.info.type_ == .q8_0 and
        tensor1.info.type_ == .q8_0 and
        K <= 2048 and
        M0 > 0 and
        M1 > 0 and
        engine.dmmv_q8_0_dual_pipe.thread_execution_width == 32 and
        engine.dmmv_q8_0_dual_pipe.max_threads_per_threadgroup >= block_size;
}

fn dispatchDualQ8DmmvOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    tensor0: *const metal_loader.LoadedTensor,
    tensor1: *const metal_loader.LoadedTensor,
    weight0_buf: *const MetalBuffer,
    weight1_buf: *const MetalBuffer,
    weight0_offset: u32,
    weight1_offset: u32,
    input_buf: *const MetalBuffer,
    output0_buf: *const MetalBuffer,
    output1_buf: *const MetalBuffer,
    M0: u32,
    M1: u32,
    K: u32,
) void {
    recordDmmvProfile(engine, tensor0, M0, K);
    recordDmmvProfile(engine, tensor1, M1, K);

    const push = DualQ8DmmvPush{
        .M0 = M0,
        .M1 = M1,
        .K = K,
        .a0_offset = weight0_offset,
        .a1_offset = weight1_offset,
        .x_offset = 0,
        .y0_offset = 0,
        .y1_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ weight0_buf, weight1_buf, input_buf, output0_buf, output1_buf };
    const total_rows = M0 + M1;
    const block_size = engine.q8_dual_tg_override orelse 1024;
    const simd_width = if (engine.dmmv_q8_0_dual_pipe.thread_execution_width > 0) engine.dmmv_q8_0_dual_pipe.thread_execution_width else @as(u32, 32);
    const rows_per_wg: u32 = block_size / simd_width;
    cmd.dispatchV2(&engine.dmmv_q8_0_dual_pipe, .{ (total_rows + rows_per_wg - 1) / rows_per_wg, 1, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(DualQ8DmmvPush), 0);
}

/// Fused RMSNorm + Dual Q8_0 DMMV: reads raw hidden state, computes norm inline,
/// eliminating the separate RMSNorm dispatch and barrier.
fn dispatchFusedNormDualQ8DmmvOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    tensor0: *const metal_loader.LoadedTensor,
    tensor1: *const metal_loader.LoadedTensor,
    weight0_buf: *const MetalBuffer,
    weight1_buf: *const MetalBuffer,
    weight0_offset: u32,
    weight1_offset: u32,
    hidden_buf: *const MetalBuffer,
    norm_weight_buf: *const MetalBuffer,
    output0_buf: *const MetalBuffer,
    output1_buf: *const MetalBuffer,
    M0: u32,
    M1: u32,
    K: u32,
) void {
    recordDmmvProfile(engine, tensor0, M0, K);
    recordDmmvProfile(engine, tensor1, M1, K);

    const push = DualQ8DmmvPush{
        .M0 = M0,
        .M1 = M1,
        .K = K,
        .a0_offset = weight0_offset,
        .a1_offset = weight1_offset,
        .x_offset = 0,
        .y0_offset = 0,
        .y1_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ weight0_buf, weight1_buf, hidden_buf, output0_buf, output1_buf, norm_weight_buf };
    const total_rows = M0 + M1;
    const block_size = engine.q8_dual_tg_override orelse 1024;
    const simd_width = if (engine.dmmv_q8_0_dual_fused_norm_pipe.thread_execution_width > 0) engine.dmmv_q8_0_dual_fused_norm_pipe.thread_execution_width else @as(u32, 32);
    const rows_per_wg: u32 = block_size / simd_width;
    cmd.dispatchV2(&engine.dmmv_q8_0_dual_fused_norm_pipe, .{ (total_rows + rows_per_wg - 1) / rows_per_wg, 1, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(DualQ8DmmvPush), 0);
}

/// Fused RMSNorm + Q8_0 DMMV (K <= 2048): reads raw hidden state, computes norm inline.
fn dispatchFusedNormQ8DmmvOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    tensor: *const metal_loader.LoadedTensor,
    weight_buf: *const MetalBuffer,
    weight_offset: u32,
    hidden_buf: *const MetalBuffer,
    norm_weight_buf: *const MetalBuffer,
    output_buf: *const MetalBuffer,
    M: u32,
    K: u32,
) void {
    recordDmmvProfile(engine, tensor, M, K);

    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = weight_offset,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ weight_buf, hidden_buf, output_buf, norm_weight_buf };
    const block_size: u32 = 256;
    const rows_per_wg: u32 = block_size / 32;
    const wgs = (M + rows_per_wg - 1) / rows_per_wg;
    cmd.dispatchV2(&engine.dmmv_q8_0_k2048_fused_norm_pipe, .{ wgs, 1, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
}

fn dispatchDmmvOnCmdWithWeightBuf(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    tensor: *const metal_loader.LoadedTensor,
    weight_buf: *const MetalBuffer,
    weight_offset: u32,
    input_buf: *const MetalBuffer,
    output_buf: *const MetalBuffer,
    M: u32,
    K: u32,
    extra_byte_offset: u32,
) void {
    recordDmmvProfile(engine, tensor, M, K);

    // Fast path: SIMD-coalesced repacked Q8_0 layout
    if (weight_buf.is_repacked_q8 and engine.dmmv_q8_0_repacked_pipe.handle != null) {
        const push = DmmvPush{
            .M = M,
            .K = K,
            .a_offset = weight_offset + extra_byte_offset,
            .x_offset = 0,
            .y_offset = 0,
        };
        const bufs = [_]*const MetalBuffer{ weight_buf, input_buf, output_buf };
        const simd_w: u32 = if (engine.dmmv_q8_0_repacked_pipe.thread_execution_width > 0)
            engine.dmmv_q8_0_repacked_pipe.thread_execution_width
        else
            32;
        const block_size: u32 = @min(512, engine.dmmv_q8_0_repacked_pipe.max_threads_per_threadgroup);
        const rows_per_wg: u32 = block_size / simd_w * 2; // nr=2
        const wgs = (M + rows_per_wg - 1) / rows_per_wg;
        cmd.dispatchV2(&engine.dmmv_q8_0_repacked_pipe, .{ wgs, 1, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
        return;
    }

    const pip = engine.dmmvPipelineForType(tensor, M, K) orelse {
        log.err("No DMMV pipeline for quant type {d} (tensor {s})", .{ @intFromEnum(tensor.info.type_), tensor.info.name });
        return;
    };
    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = weight_offset + extra_byte_offset,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ weight_buf, input_buf, output_buf };
    const wgs = (M + pip.rows_per_wg - 1) / pip.rows_per_wg;
    cmd.dispatchV2(pip.pipe, .{ wgs, 1, 1 }, .{ pip.block_size, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), pip.push_idx);
}

fn dispatchLmHeadOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    input_buf: *const MetalBuffer,
    output_buf: *const MetalBuffer,
    hidden_dim: u32,
    vocab_size: u32,
) void {
    if (engine.lm_head_private_buf.handle != null) {
        dispatchDmmvOnCmdWithWeightBuf(engine, cmd, engine.lm_head, &engine.lm_head_private_buf, 0, input_buf, output_buf, vocab_size, hidden_dim, 0);
        return;
    }

    dispatchDmmvOnCmd(engine, cmd, engine.lm_head, input_buf, output_buf, vocab_size, hidden_dim, 0);
}

/// Variant of `dispatchLmHeadOnCmd` that accepts a byte offset into `input_buf`.
/// Used by `prefillBatched` to point at the last token's slice inside the
/// contiguous [N × hidden_dim] normalized hidden buffer without a CPU memcpy.
fn dispatchLmHeadWithInputOffset(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    input_buf: *const MetalBuffer,
    output_buf: *const MetalBuffer,
    hidden_dim: u32,
    vocab_size: u32,
    x_offset_bytes: u32,
) void {
    const tensor = engine.lm_head;
    const weight_buf: *const MetalBuffer = if (engine.lm_head_private_buf.handle != null)
        &engine.lm_head_private_buf
    else
        &tensor.gpu_buffer;
    const weight_offset: u32 = if (engine.lm_head_private_buf.handle != null)
        0
    else
        tensorPageOffset(engine.model, tensor);

    const pip = engine.dmmvPipelineForType(tensor, vocab_size, hidden_dim) orelse {
        log.err("No DMMV pipeline for LM head quant type {d}", .{@intFromEnum(tensor.info.type_)});
        return;
    };
    const push = DmmvPush{
        .M = vocab_size,
        .K = hidden_dim,
        .a_offset = weight_offset,
        .x_offset = x_offset_bytes,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ weight_buf, input_buf, output_buf };
    const wgs = (vocab_size + pip.rows_per_wg - 1) / pip.rows_per_wg;
    cmd.dispatchV2(pip.pipe, .{ wgs, 1, 1 }, .{ pip.block_size, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), pip.push_idx);
}

fn shouldCpuLmHeadFallbackForType(arch: config_mod.Architecture, quant_type: GGMLType) bool {
    return arch == .gemma and quant_type == .q8_0;
}

fn shouldCpuLmHeadFallback(engine: *const InferenceEngine) bool {
    return shouldCpuLmHeadFallbackForType(engine.config.architecture, engine.lm_head.info.type_);
}

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
    dispatchDmmvOnCmdWithInputOffset(engine, cmd, tensor, input_buf, output_buf, M, K, extra_byte_offset, 0);
}

/// DMMV with an additional byte offset into the input vector. Used by the
/// Gemma batched MoE path to read a single expert's activation slice out of a
/// batched [n_experts_used × K] swiglu buffer without copying.
fn dispatchDmmvOnCmdWithInputOffset(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    tensor: *const metal_loader.LoadedTensor,
    input_buf: *const MetalBuffer,
    output_buf: *const MetalBuffer,
    M: u32,
    K: u32,
    extra_byte_offset: u32,
    x_byte_offset: u32,
) void {
    const pip = engine.dmmvPipelineForType(tensor, M, K) orelse {
        // CPU fallback for unsupported quant types. Re-open a command buffer
        // afterwards so later kernels in the same logical sequence still record.
        cmd.commitAndWait();
        const mmap = engine.model.mmap_data orelse return;
        const tdo = engine.model.gguf_file.tensor_data_offset;
        const in_ptr_base: [*]const f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
        const in_ptr: [*]const f32 = @ptrFromInt(@intFromPtr(in_ptr_base) + x_byte_offset);
        const out_ptr: [*]f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
        cpuDmmvFallback(mmap, tensor, tdo, in_ptr, out_ptr, M, K, extra_byte_offset, engine.allocator) catch return;
        cmd.* = metal_command.beginCommand(engine.device.ctx) catch .{
            .handle = null,
            .dispatch_count = 0,
            .barrier_count = 0,
            .barrier_enabled = true,
        };
        return;
    };
    const page_off = tensorPageOffset(engine.model, tensor);
    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = page_off + extra_byte_offset,
        .x_offset = x_byte_offset,
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
    const k2048_or_less = K <= 2048;
    const use_k2048 =
        k2048_or_less and
        engine.dmmv_q5k_moe_k2048_pipe.max_threads_per_threadgroup >= 512;
    const rows_per_wg: u32 = if (use_k2048) 16 else 8;
    const block_size: u32 = if (use_k2048) 512 else 256;
    const wgs = (M + rows_per_wg - 1) / rows_per_wg;
    const pipe = if (use_k2048) &engine.dmmv_q5k_moe_k2048_pipe else &engine.dmmv_q5k_moe_pipe;
    cmd.dispatchV2(pipe, .{ wgs, engine.config.n_experts_used, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(MoeDmmvPush), 1);
}

fn dispatchDmmvMoeQ5_1OnCmd(
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
    if (tensor.info.type_ != .q5_1) {
        log.err("Batched Q5_1 MoE DMMV called with wrong type (tensor {s})", .{tensor.info.name});
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
    // 2 rows per workgroup (matches dmmv_q5_1.metal: 2 simdgroups × 32 threads).
    const rows_per_wg: u32 = 2;
    const block_size: u32 = 64;
    const wgs = (M + rows_per_wg - 1) / rows_per_wg;
    cmd.dispatchV2(&engine.dmmv_q5_1_moe_pipe, .{ wgs, engine.config.n_experts_used, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(MoeDmmvPush), 1);
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
    recordMoeDmmvProfile(engine, tensor.info.type_, M, K, engine.config.n_experts_used);

    switch (tensor.info.type_) {
        .q4_k => dispatchDmmvMoeQ4kOnCmd(engine, cmd, tensor, input_buf, output_buf, routing_buf, M, K, expert_stride, x_expert_stride, extra_byte_offset),
        .q5_1 => dispatchDmmvMoeQ5_1OnCmd(engine, cmd, tensor, input_buf, output_buf, routing_buf, M, K, expert_stride, x_expert_stride, extra_byte_offset),
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
    const push = RmsNormPush{ .n = n, .eps = engine.config.rms_norm_eps };
    const bufs = [_]*const MetalBuffer{ input, output, weights };
    const tg_size: u32 = @min(
        @max(engine.rms_norm_pipe.max_threads_per_threadgroup, 32),
        if (n >= 1024) @as(u32, 1024) else if (n >= 256) @as(u32, 256) else @as(u32, 32),
    );
    cmd.dispatchV2(&engine.rms_norm_pipe, .{ n_groups, 1, 1 }, .{ tg_size, 1, 1 }, &bufs, &push, @sizeOf(RmsNormPush), 0);
}

fn dispatchRmsNormOnCmdWithWeightOffset(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    input: *const MetalBuffer,
    output: *const MetalBuffer,
    weights: *const MetalBuffer,
    weight_offset: u32,
    n: u32,
    n_groups: u32,
) void {
    const push = RmsNormOffsetPush{ .n = n, .eps = engine.config.rms_norm_eps, .weight_offset = weight_offset };
    const bufs = [_]*const MetalBuffer{ input, output, weights };
    const tg_size: u32 = @min(
        @max(engine.rms_norm_offset_pipe.max_threads_per_threadgroup, 32),
        if (n >= 1024) @as(u32, 1024) else if (n >= 256) @as(u32, 256) else @as(u32, 32),
    );
    cmd.dispatchV2(&engine.rms_norm_offset_pipe, .{ n_groups, 1, 1 }, .{ tg_size, 1, 1 }, &bufs, &push, @sizeOf(RmsNormOffsetPush), 0);
}

fn dispatchRmsNormOnCmdWithTensorWeights(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    input: *const MetalBuffer,
    output: *const MetalBuffer,
    weights: *const metal_loader.LoadedTensor,
    n: u32,
    n_groups: u32,
) void {
    const weight_offset: u32 = @intCast(tensorPageOffset(engine.model, weights) / @sizeOf(f32));
    dispatchRmsNormOnCmdWithWeightOffset(engine, cmd, input, output, &weights.gpu_buffer, weight_offset, n, n_groups);
}

/// Fused residual-add + RMS norm: hidden += scale * residual; norm_out = weights * normalize(hidden).
/// Eliminates one barrier per layer vs separate scale_acc + barrier + rms_norm.
/// residual_rms_norm.metal: buffer(0)=push, buffer(1)=hidden, buffer(2)=residual, buffer(3)=norm_out, buffer(4)=weights.
fn dispatchResidualRmsNormOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    hidden: *const MetalBuffer,
    residual: *const MetalBuffer,
    norm_out: *const MetalBuffer,
    weights: *const MetalBuffer,
    n: u32,
    scale: f32,
) void {
    const push = ResidualRmsNormPush{ .n = n, .eps = engine.config.rms_norm_eps, .scale = scale };
    const bufs = [_]*const MetalBuffer{ hidden, residual, norm_out, weights };
    cmd.dispatchV2(&engine.residual_rms_norm_pipe, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &bufs, &push, @sizeOf(ResidualRmsNormPush), 0);
}

/// Dispatch a Q4_K × f32 batched matmul.
///
/// Computes `output[N×M] = weight[M×K] × input[N×K]` where weight rows are
/// stored in Q4_K blocks (144 bytes / 256 elements). Use for prefill when
/// N ≥ ~16 — below that DMMV is faster.
/// gemm_q4k.metal: buffer(0)=push, buffer(1)=weights, buffer(2)=input, buffer(3)=output.
fn dispatchGemmQ4KOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    weight: *const metal_loader.LoadedTensor,
    input: *const MetalBuffer,
    output: *const MetalBuffer,
    M: u32,
    K: u32,
    N: u32,
) void {
    std.debug.assert(K % 256 == 0);
    const push = GemmPush{
        .ne00 = @intCast(K),
        .ne02 = 1,
        .nb01 = @as(u64, K / 256) * 144,
        .nb02 = 0,
        .ne12 = 1,
        .nb10 = 4,
        .nb11 = @as(u64, K) * 4,
        .nb12 = 0,
        .ne0 = @intCast(M),
        .ne1 = @intCast(N),
        .src0_off = tensorPageOffset(engine.model, weight),
    };
    const bufs = [_]*const MetalBuffer{ &weight.gpu_buffer, input, output };
    const grid = [_]u32{ (N + 31) / 32, (M + 63) / 64, 1 };
    cmd.dispatchV2WithTgMem(&engine.gemm_q4k_pipe, grid, .{ 128, 1, 1 }, &bufs, &push, @sizeOf(GemmPush), 0, 8192);
}

/// Dispatch a Q6_K × f32 batched matmul. Same tile layout as gemm_q4k.
/// Q6_K blocks are 210 bytes / 256 elements.
fn dispatchGemmQ6KOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    weight: *const metal_loader.LoadedTensor,
    input: *const MetalBuffer,
    output: *const MetalBuffer,
    M: u32,
    K: u32,
    N: u32,
) void {
    std.debug.assert(K % 256 == 0);
    const push = GemmPush{
        .ne00 = @intCast(K),
        .ne02 = 1,
        .nb01 = @as(u64, K / 256) * 210,
        .nb02 = 0,
        .ne12 = 1,
        .nb10 = 4,
        .nb11 = @as(u64, K) * 4,
        .nb12 = 0,
        .ne0 = @intCast(M),
        .ne1 = @intCast(N),
        .src0_off = tensorPageOffset(engine.model, weight),
    };
    const bufs = [_]*const MetalBuffer{ &weight.gpu_buffer, input, output };
    const grid = [_]u32{ (N + 31) / 32, (M + 63) / 64, 1 };
    cmd.dispatchV2WithTgMem(&engine.gemm_q6k_pipe, grid, .{ 128, 1, 1 }, &bufs, &push, @sizeOf(GemmPush), 0, 8192);
}

/// Dispatch batched RoPE for N tokens at consecutive positions.
/// rope_batched.metal: buffer(0)=push, buffer(1)=in, buffer(2)=out, buffer(3)=inv_freq.
fn dispatchRopeBatchedOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    in_buf: *const MetalBuffer,
    out_buf: *const MetalBuffer,
    freq_buf: *const MetalBuffer,
    head_dim: u32,
    rope_dim: u32,
    n_heads: u32,
    position_base: u32,
    n_tokens: u32,
    freq_base: f32,
    use_freq_buffer: bool,
    attn_scale: f32,
) void {
    const push = RopeBatchedPush{
        .stride = head_dim,
        .rope_dim = rope_dim,
        .n_heads = n_heads,
        .position_base = position_base,
        .freq_base_bits = if (use_freq_buffer) 0 else @bitCast(freq_base),
        .attn_scale_bits = if (attn_scale == 1.0) 0 else @bitCast(attn_scale),
    };
    const bufs = [_]*const MetalBuffer{ in_buf, out_buf, freq_buf };
    const grid = [_]u32{ n_heads, n_tokens, 1 };
    cmd.dispatchV2(&engine.rope_batched_pipe, grid, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(RopeBatchedPush), 0);
}

/// Dispatch batched causal flash attention for N query tokens sharing a KV cache.
/// flash_attn_batched.metal: buffer(0)=push, buffer(1)=q, buffer(2)=k_cache, buffer(3)=v_cache, buffer(4)=out.
fn dispatchFlashAttnBatchedOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    q_buf: *const MetalBuffer,
    k_cache: *const MetalBuffer,
    v_cache: *const MetalBuffer,
    out_buf: *const MetalBuffer,
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    kv_len: u32,
    n_queries: u32,
    kv_pos_offset: u32,
) void {
    const push = BatchedFlashAttnPush{
        .head_dim = head_dim,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .kv_len = kv_len,
        .n_queries = n_queries,
        .kv_pos_offset = kv_pos_offset,
    };
    const bufs = [_]*const MetalBuffer{ q_buf, k_cache, v_cache, out_buf };
    cmd.dispatchV2(&engine.flash_attn_batched_pipe, .{ n_heads, n_queries, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(BatchedFlashAttnPush), 0);
}

/// Q8_0 variant of `dispatchFlashAttnBatchedOnCmd`. KV cache is stored as
/// 34-byte Q8_0 blocks (2-byte half scale + 32 i8 quants).
fn dispatchFlashAttnBatchedQ8OnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    q_buf: *const MetalBuffer,
    k_cache: *const MetalBuffer,
    v_cache: *const MetalBuffer,
    out_buf: *const MetalBuffer,
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    kv_len: u32,
    n_queries: u32,
    kv_pos_offset: u32,
    kv_head_stride_bytes: u32,
    kv_token_stride_bytes: u32,
) void {
    const push = BatchedFlashAttnQ8Push{
        .head_dim = head_dim,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .kv_len = kv_len,
        .n_queries = n_queries,
        .kv_pos_offset = kv_pos_offset,
        .kv_head_stride_bytes = kv_head_stride_bytes,
        .kv_token_stride_bytes = kv_token_stride_bytes,
    };
    const bufs = [_]*const MetalBuffer{ q_buf, k_cache, v_cache, out_buf };
    cmd.dispatchV2(&engine.flash_attn_batched_q8_pipe, .{ n_heads, n_queries, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(BatchedFlashAttnQ8Push), 0);
}

/// Dispatch a batched KV-cache write: copy N_tokens × kv_dim contiguous f32s
/// from `src_k`/`src_v` to the layer's cache starting at `dst_offset_elements`.
fn dispatchKvCacheWriteBatchedOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    layer_idx: usize,
    src_k: *const MetalBuffer,
    src_v: *const MetalBuffer,
    dst_offset_elements: u32,
    n_elements: u32,
) void {
    const push = KvCacheWritePush{
        .n = n_elements,
        .dst_offset = dst_offset_elements,
        .dst_offset_bytes = 0,
    };
    const bufs = [_]*const MetalBuffer{
        src_k,
        src_v,
        &engine.kv_k_cache[layer_idx],
        &engine.kv_v_cache[layer_idx],
    };
    cmd.dispatchV2(&engine.kv_cache_write_pipe, .{ (n_elements + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(KvCacheWritePush), 0);
}

/// Batched KV-cache write into a Q8_0-quantized cache. `n_blocks` is the
/// total number of 32-element blocks to write across all tokens (typically
/// `n_tokens * kv_dim / 32`). `dst_offset_bytes` points at the byte where
/// the first block for the first token starts.
fn dispatchKvCacheWriteBatchedQ8OnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    layer_idx: usize,
    src_k: *const MetalBuffer,
    src_v: *const MetalBuffer,
    dst_offset_bytes: u32,
    n_blocks: u32,
) void {
    const push = KvCacheWritePush{
        .n = n_blocks,
        .dst_offset = 0,
        .dst_offset_bytes = dst_offset_bytes,
    };
    const bufs = [_]*const MetalBuffer{
        src_k,
        src_v,
        &engine.kv_k_cache[layer_idx],
        &engine.kv_v_cache[layer_idx],
    };
    cmd.dispatchV2(&engine.kv_cache_write_q8_pipe, .{ n_blocks, 1, 1 }, .{ 32, 1, 1 }, &bufs, &push, @sizeOf(KvCacheWritePush), 0);
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
    sliding_window_size: u32,
    kv_cache_head_stride_bytes: u32,
    kv_cache_bytes_per_token: u32,
) void {
    const push = FlashAttnPush{
        .head_dim = head_dim,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .seq_len = seq_len,
        .sliding_window_size = sliding_window_size,
        // Metal currently keeps the KV cache as a flat contiguous
        // [token][kv_head][head_dim] buffer. Use page_size=0 to select the
        // shader's contiguous-addressing fast path and skip page-table math.
        .page_size = 0,
        .attn_scale_bits = if (engine.config.attn_scale != 0) @as(u32, @bitCast(engine.config.attn_scale)) else 0,
        .kv_head_stride_bytes = kv_cache_head_stride_bytes,
        .kv_token_stride_bytes = kv_cache_bytes_per_token,
    };
    // Refresh per-head sink values every layer. Layers without sinks must reset to NaN
    // so they do not inherit stale values from a prior layer that did have sinks.
    const sink_ptr: [*]f32 = @ptrCast(@alignCast(engine.attn_sinks_buf.cpu_ptr.?));
    for (0..n_heads) |i| sink_ptr[i] = std.math.nan(f32);
    if (engine.attn_sink_values) |sink_vals| {
        if (sink_vals[layer_idx].len > 0) {
            @memcpy(sink_ptr[0..sink_vals[layer_idx].len], sink_vals[layer_idx]);
        }
    }
    const bufs = [_]*const MetalBuffer{
        &engine.page_table_buf,
        &engine.q_buf,
        &engine.kv_k_cache[layer_idx],
        &engine.kv_v_cache[layer_idx],
        &engine.attn_out_buf,
        &engine.attn_sinks_buf,
    };
    const pipe = if (engine.kv_cache_q8) &engine.flash_attn_q8_pipe else &engine.flash_attn_pipe;
    cmd.dispatchV2(pipe, .{ n_heads, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(FlashAttnPush), 0);
}

fn dispatchKvCacheWriteOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    layer_idx: usize,
    kv_dim: u32,
    dst_offset: u32,
    dst_offset_bytes: u32,
) void {
    const push = KvCacheWritePush{
        .n = if (engine.kv_cache_q8) @divTrunc(kv_dim, 32) else kv_dim,
        .dst_offset = dst_offset,
        .dst_offset_bytes = if (engine.kv_cache_q8) dst_offset_bytes else 0,
    };
    const bufs = [_]*const MetalBuffer{
        &engine.k_buf,
        &engine.v_buf,
        &engine.kv_k_cache[layer_idx],
        &engine.kv_v_cache[layer_idx],
    };
    if (engine.kv_cache_q8) {
        cmd.dispatchV2(&engine.kv_cache_write_q8_pipe, .{ @divTrunc(kv_dim, 32), 1, 1 }, .{ 32, 1, 1 }, &bufs, &push, @sizeOf(KvCacheWritePush), 0);
    } else {
        cmd.dispatchV2(&engine.kv_cache_write_pipe, .{ (kv_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(KvCacheWritePush), 0);
    }
}

fn selectRopeFreqBuffer(
    engine: *InferenceEngine,
    rope_dim: u32,
    freq_base: f32,
    use_freq_factors: bool,
) *const MetalBuffer {
    const default_rope_dim: u32 = if (engine.config.rope_dim > 0) engine.config.rope_dim else engine.config.head_dim;
    if (use_freq_factors and rope_dim == default_rope_dim and freq_base == engine.config.rope_freq_base) {
        return &engine.rope_freq_buf;
    }

    if (engine.rope_variant_rope_dim != rope_dim or engine.rope_variant_freq_base != freq_base or engine.rope_variant_uses_freq_factors != use_freq_factors) {
        const freq_ptr: [*]f32 = @ptrCast(@alignCast(engine.rope_variant_freq_buf.cpu_ptr.?));
        const half_rot: usize = @intCast(rope_dim / 2);
        const freq_factors = if (use_freq_factors) engine.rope_freq_factors else null;
        fillRopeInvFreqs(freq_ptr[0..half_rot], rope_dim, freq_base, freq_factors);
        engine.rope_variant_rope_dim = rope_dim;
        engine.rope_variant_freq_base = freq_base;
        engine.rope_variant_uses_freq_factors = use_freq_factors;
    }
    return &engine.rope_variant_freq_buf;
}

fn dispatchRopeOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    input: *const MetalBuffer,
    _: *const MetalBuffer, // output (unused — native shader modifies input in-place)
    stride: u32,
    rope_dim: u32,
    n_heads: u32,
    position: u32,
    freq_base: f32,
    use_freq_factors: bool,
) void {
    // Always use native RoPE with precomputed frequencies (supports YaRN, rope_freqs.weight, etc.)
    // The SPIRV-Cross rope_fused shader lost its freq_base_bits push constant field,
    // so it can no longer compute frequencies correctly. The native shader reads from
    // the precomputed rope_freq_buf which handles all scaling variants.
    const freq_buf = selectRopeFreqBuffer(engine, rope_dim, freq_base, use_freq_factors);
    const push = RopeNativePush{
        .stride = stride,
        .rope_dim = rope_dim,
        .n_heads = n_heads,
        .position = position,
    };
    const bufs = [_]*const MetalBuffer{ input, freq_buf };
    cmd.dispatchV2(&engine.rope_native_pipe, .{ n_heads, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(RopeNativePush), 0);
}

fn dispatchScaleInPlaceOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    buf: *const MetalBuffer,
    scratch: *const MetalBuffer,
    n: u32,
    scale: f32,
    profile: ?*RuntimeProfile,
    barrier_class: BarrierClass,
) void {
    if (n == 0 or scale == 1.0) return;

    dispatchCopyF32OnCmd(engine, cmd, buf, scratch, n);
    profileBarrier(cmd, profile, barrier_class);
    dispatchZeroF32OnCmd(engine, cmd, buf, n);
    profileBarrier(cmd, profile, barrier_class);

    const push = ScaleAccPush{ .n = n, .scale_bits = @as(u32, @bitCast(scale)) };
    const bufs = [_]*const MetalBuffer{ buf, scratch };
    cmd.dispatchV2(&engine.scale_acc_pipe, .{ (n + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(ScaleAccPush), 0);
    profileBarrier(cmd, profile, barrier_class);
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

fn dispatchFfnActivationOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    gate: *const MetalBuffer,
    output: *const MetalBuffer,
    up: *const MetalBuffer,
    n: u32,
) void {
    const push = SwiGLUPush{ .n = n };
    const bufs = [_]*const MetalBuffer{ gate, output, up };
    const pipe = if (usesGeglu(engine.config)) &engine.geglu_pipe else &engine.swiglu_pipe;
    cmd.dispatchV2(pipe, .{ (n + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SwiGLUPush), 0);
}

fn fillRopeInvFreqs(dst: []f32, rope_dim: u32, freq_base: f32, freq_factors: ?[]const f32) void {
    for (dst, 0..) |*value, i| {
        const exponent = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(rope_dim));
        value.* = 1.0 / std.math.pow(f32, freq_base, exponent);
        if (freq_factors) |factors| {
            if (i < factors.len and factors[i] != 0.0) value.* /= factors[i];
        }
    }
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
    const push = SoftmaxTopkPush{
        .n_experts = n_experts,
        .k = k,
    };
    const bufs = [_]*const MetalBuffer{ logits, output };
    cmd.dispatchV2(&engine.softmax_topk_pipe, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SoftmaxTopkPush), 0);
}

fn dispatchSoftmaxTopkScaledOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    logits: *const MetalBuffer,
    output: *const MetalBuffer,
    n_experts: u32,
    k: u32,
    logit_scale: f32,
) void {
    const push = SoftmaxTopkScaledPush{
        .n_experts = n_experts,
        .k = k,
        .logit_scale_bits = @as(u32, @bitCast(logit_scale)),
    };
    const bufs = [_]*const MetalBuffer{ logits, output };
    cmd.dispatchV2(&engine.softmax_topk_scaled_pipe, .{ 1, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SoftmaxTopkScaledPush), 0);
}

fn dispatchSoftmaxTopkBatchedOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    logits: *const MetalBuffer,
    output: *const MetalBuffer,
    n_tokens: u32,
    n_experts: u32,
    k: u32,
) void {
    const push = SoftmaxTopkBatchedPush{
        .n_experts = n_experts,
        .k = k,
        .logits_stride = n_experts,
        .output_stride = k * 2,
    };
    const bufs = [_]*const MetalBuffer{ logits, output };
    cmd.dispatchV2(&engine.softmax_topk_batched_pipe, .{ n_tokens, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SoftmaxTopkBatchedPush), 0);
}

fn dispatchMoeRoutePackOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    routing: *const MetalBuffer,
    counts: *const MetalBuffer,
    ids: *const MetalBuffer,
    n_tokens: u32,
    n_experts: u32,
    k: u32,
) void {
    const push = MoeRoutePackPush{
        .n_tokens = n_tokens,
        .n_experts = n_experts,
        .k = k,
        .routing_stride = k * 2,
        .ids_stride = n_tokens,
    };
    const bufs = [_]*const MetalBuffer{ routing, counts, ids };
    cmd.dispatchV2(&engine.moe_route_pack_pipe, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &bufs, &push, @sizeOf(MoeRoutePackPush), 0);
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

fn dispatchMoeWeightedAccScaledOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    accum: *const MetalBuffer,
    src: *const MetalBuffer,
    routing: *const MetalBuffer,
    scales: *const metal_loader.LoadedTensor,
    n: u32,
    n_used: u32,
    src_stride: u32,
) void {
    const push = MoeWeightedAccScaledPush{
        .n = n,
        .n_used = n_used,
        .src_stride = src_stride,
        .scale_offset = @intCast(tensorPageOffset(engine.model, scales) / @sizeOf(f32)),
    };
    const bufs = [_]*const MetalBuffer{ accum, src, routing, &scales.gpu_buffer };
    cmd.dispatchV2(&engine.moe_weighted_acc_scaled_pipe, .{ (n + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(MoeWeightedAccScaledPush), 3);
}

/// Fused MoE weighted accumulate + shared expert: accum += sum(w_i * expert_i) + sh_weight * shared.
/// Eliminates one barrier per layer vs separate moe_weighted_acc + sigmoid_scale_acc.
/// moe_weighted_acc_shared.metal: buffer(0)=accum, buffer(1)=src, buffer(2)=routing, buffer(3)=push,
///                                buffer(4)=shared_src, buffer(5)=gate.
fn dispatchMoeWeightedAccSharedOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    accum: *const MetalBuffer,
    src: *const MetalBuffer,
    routing: *const MetalBuffer,
    shared_src: *const MetalBuffer,
    gate: *const MetalBuffer,
    n: u32,
    n_used: u32,
    src_stride: u32,
    has_gate: bool,
) void {
    const push = MoeWeightedAccSharedPush{
        .n = n,
        .n_used = n_used,
        .src_stride = src_stride,
        .has_gate = if (has_gate) 1 else 0,
    };
    const bufs = [_]*const MetalBuffer{ accum, src, routing, shared_src, gate };
    cmd.dispatchV2(&engine.moe_weighted_acc_shared_pipe, .{ (n + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(MoeWeightedAccSharedPush), 3);
}

/// Returns true if the attention gate (sigmoid gating) should be applied after flash attn.
fn dispatchFullAttnPrepOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    profile: ?*RuntimeProfile,
    layer_idx: usize,
    lt: LayerTensors,
    attn: LayerAttentionParams,
    hidden_dim: u32,
) !bool {
    const cfg = engine.config;
    const q_tensor = lt.attn_q orelse return error.MissingTensor;
    const k_tensor = lt.attn_k orelse return error.MissingTensor;
    const v_tensor = if (attn.use_k_as_v) k_tensor else lt.attn_v orelse return error.MissingTensor;

    // Detect packed Q+gate vs separate format (matches Vulkan reference forward.zig:1499-1505).
    // Packed: attn_q has q_dim*2 rows (interleaved [Q,gate] per head) — Qwen3Next style.
    // Separate: attn_q has q_dim rows, gate is in a separate attn_gate tensor — Qwen3.5 MoE style.
    const q_rows: u32 = @intCast(q_tensor.info.numElements() / hidden_dim);
    const gate_mode = classifyFullAttnGate(q_rows, attn.q_dim, lt.attn_gate != null);

    if (gate_mode.packed_q_gate) {
        // Packed: project full Q+gate into attn_out_buf, then deinterleave.
        const q_full_dim = attn.q_dim * 2;
        dispatchDmmvOnCmd(engine, cmd, q_tensor, &engine.norm_buf, &engine.attn_out_buf, q_full_dim, hidden_dim, 0);
        dispatchDmmvOnCmd(engine, cmd, k_tensor, &engine.norm_buf, &engine.k_buf, attn.kv_dim, hidden_dim, 0);
        dispatchDmmvOnCmd(engine, cmd, v_tensor, &engine.norm_buf, &engine.v_buf, attn.kv_dim, hidden_dim, 0);
        profileBarrier(cmd, profile, .full_attn);
        dispatchDeinterleaveOnCmd(engine, cmd, &engine.attn_out_buf, &engine.q_buf, &engine.gate_buf, attn.head_dim, cfg.n_heads);
        profileBarrier(cmd, profile, .full_attn);
    } else {
        // Separate: project Q directly to q_buf, gate (if present) to gate_buf.
        dispatchDmmvOnCmd(engine, cmd, q_tensor, &engine.norm_buf, &engine.q_buf, attn.q_dim, hidden_dim, 0);
        if (gate_mode.separate_attn_gate) {
            dispatchDmmvOnCmd(engine, cmd, lt.attn_gate.?, &engine.norm_buf, &engine.gate_buf, attn.q_dim, hidden_dim, 0);
        }
        dispatchDmmvOnCmd(engine, cmd, k_tensor, &engine.norm_buf, &engine.k_buf, attn.kv_dim, hidden_dim, 0);
        dispatchDmmvOnCmd(engine, cmd, v_tensor, &engine.norm_buf, &engine.v_buf, attn.kv_dim, hidden_dim, 0);
        profileBarrier(cmd, profile, .full_attn);
    }

    if (engine.debug_validation_enabled and shouldDebugAttentionValidation(cfg, engine.position, layer_idx)) {
        commitAndWaitProfiled(cmd, profile);
        const debug_start = profileStart(profile != null);
        try debugCompareAttentionProjectionStage(engine, @intCast(layer_idx), layer_idx, lt, hidden_dim);
        if (profile) |p| p.debug_validation_ns += profileElapsedNs(debug_start);
        cmd.* = try beginProfiledCommand(engine, profile);
    }

    // Apply Q/K/V biases if present (gpt-oss)
    if (lt.attn_q_bias != null or lt.attn_k_bias != null or lt.attn_v_bias != null) {
        cmd.commitAndWait();
        if (lt.attn_q_bias) |b| addBiasFromTensor(engine, @ptrCast(@alignCast(engine.q_buf.cpu_ptr.?)), b, attn.q_dim);
        if (lt.attn_k_bias) |b| addBiasFromTensor(engine, @ptrCast(@alignCast(engine.k_buf.cpu_ptr.?)), b, attn.kv_dim);
        if (!attn.use_k_as_v) {
            if (lt.attn_v_bias) |b| addBiasFromTensor(engine, @ptrCast(@alignCast(engine.v_buf.cpu_ptr.?)), b, attn.kv_dim);
        }
        cmd.* = try metal_command.beginCommand(engine.device.ctx);
    }

    // Q/K norms are independent — concurrent dispatch overlaps them.
    if (engine.attn_q_norm_present[layer_idx]) {
        dispatchRmsNormOnCmd(engine, cmd, &engine.q_buf, &engine.q_buf, &engine.attn_q_norm_bufs[layer_idx], attn.head_dim, cfg.n_heads);
    }
    if (engine.attn_k_norm_present[layer_idx]) {
        dispatchRmsNormOnCmd(engine, cmd, &engine.k_buf, &engine.k_buf, &engine.attn_k_norm_bufs[layer_idx], attn.head_dim, attn.n_kv_heads);
    }
    if (cfg.architecture == .gemma and cfg.rope_freq_base_swa > 0) {
        dispatchRmsNormOnCmd(engine, cmd, &engine.v_buf, &engine.v_buf, &engine.unit_rms_norm_weights, attn.head_dim, attn.n_kv_heads);
    }
    profileBarrier(cmd, profile, .full_attn); // norm outputs visible before rope

    // RoPE Q/K are independent — concurrent dispatch overlaps them.
    dispatchRopeOnCmd(engine, cmd, &engine.q_buf, &engine.q_buf, attn.head_dim, attn.rope_dim, cfg.n_heads, engine.position, attn.rope_freq_base, attn.use_rope_freq_factors);
    dispatchRopeOnCmd(engine, cmd, &engine.k_buf, &engine.k_buf, attn.head_dim, attn.rope_dim, attn.n_kv_heads, engine.position, attn.rope_freq_base, attn.use_rope_freq_factors);
    profileBarrier(cmd, profile, .full_attn); // rope outputs visible before KV cache write

    dispatchKvCacheWriteOnCmd(
        engine,
        cmd,
        layer_idx,
        attn.kv_dim,
        engine.position * attn.kv_dim,
        @intCast(@as(u64, engine.position) * attn.kv_cache_bytes_per_token),
    );
    return gate_mode.apply_attn_gate;
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
    if (tensor.info.type_ == .q8_0) {
        for (0..M) |row| {
            output[row] = dotQ8_0Row(raw, @intCast(row), K, input);
        }
        return;
    }

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

/// Add bias from a tensor to a CPU buffer (f32 only).
fn addBiasFromTensor(engine: *InferenceEngine, output: [*]f32, tensor: *const metal_loader.LoadedTensor, n: u32) void {
    const mmap = engine.model.mmap_data orelse return;
    const off: usize = @intCast(engine.model.gguf_file.tensor_data_offset + tensor.info.offset);
    const bias_ptr: [*]const f32 = @ptrCast(@alignCast(mmap[off..].ptr));
    for (0..n) |i| output[i] += bias_ptr[i];
}

/// Add bias from a slice of a per-expert tensor (2D: [n, n_experts], row-major).
fn addBiasFromTensorSlice(engine: *InferenceEngine, output: [*]f32, tensor: *const metal_loader.LoadedTensor, expert_id: u32, n: u32) void {
    const mmap = engine.model.mmap_data orelse return;
    const off: usize = @intCast(engine.model.gguf_file.tensor_data_offset + tensor.info.offset);
    const base: usize = @as(usize, expert_id) * @as(usize, n) * 4;
    const bias_ptr: [*]const f32 = @ptrCast(@alignCast(mmap[off + base ..].ptr));
    for (0..n) |i| output[i] += bias_ptr[i];
}

fn tensorF32Slice(engine: *InferenceEngine, tensor: *const metal_loader.LoadedTensor, n: u32) ![]const f32 {
    if (tensor.info.type_ != .f32) return error.UnsupportedQuantType;
    const mmap = engine.model.mmap_data orelse return error.NoMmapData;
    const off: usize = @intCast(engine.model.gguf_file.tensor_data_offset + tensor.info.offset);
    const ptr: [*]const f32 = @ptrCast(@alignCast(mmap[off..].ptr));
    return ptr[0..n];
}

fn cpuMulInPlace(values: [*]f32, scale: []const f32, n: u32) void {
    for (0..n) |i| values[i] *= scale[i];
}

fn cpuMulScalarInPlace(values: [*]f32, scale: f32, n: u32) void {
    for (0..n) |i| values[i] *= scale;
}

fn cpuRopeWithFreqs(data: [*]f32, stride: u32, rope_dim: u32, n_heads: u32, position: u32, inv_freq: []const f32) void {
    const half_rot = rope_dim / 2;
    std.debug.assert(inv_freq.len >= half_rot);
    for (0..n_heads) |h| {
        const base_idx = @as(u32, @intCast(h)) * stride;
        for (0..half_rot) |i| {
            const theta = @as(f32, @floatFromInt(position)) * inv_freq[i];
            const cos_t = @cos(theta);
            const sin_t = @sin(theta);
            const idx0 = base_idx + @as(u32, @intCast(i));
            const idx1 = idx0 + half_rot;
            const x0 = data[idx0];
            const x1 = data[idx1];
            data[idx0] = x0 * cos_t - x1 * sin_t;
            data[idx1] = x0 * sin_t + x1 * cos_t;
        }
    }
}

/// SwiGLU: output[i] = SiLU(gate[i]) * up[i]
fn cpuSwiGLU(gate: [*]const f32, up: [*]const f32, output: [*]f32, n: u32) void {
    for (0..n) |i| {
        const x = gate[i];
        output[i] = (x / (1.0 + @exp(-x))) * up[i];
    }
}

/// GeGLU: output[i] = GELU(gate[i]) * up[i]
fn cpuGeGLU(gate: [*]const f32, up: [*]const f32, output: [*]f32, n: u32) void {
    for (0..n) |i| {
        const g = gate[i];
        const g3 = g * g * g;
        var inner = 0.7978845608 * (g + 0.044715 * g3);
        inner = std.math.clamp(inner, -15.0, 15.0);
        const gelu = 0.5 * g * (1.0 + std.math.tanh(inner));
        output[i] = gelu * up[i];
    }
}

/// OAI SwiGLU variant for gpt-oss MoE experts.
/// output = (min(gate, limit) / (1 + exp(alpha * -gate))) * (clamp(up, -limit, limit) + 1)
fn cpuSwiGLU_OAI(gate: [*]const f32, up: [*]const f32, output: [*]f32, n: u32) void {
    const alpha: f32 = 1.702;
    const limit: f32 = 7.0;
    for (0..n) |i| {
        const x = @min(gate[i], limit);
        const y = std.math.clamp(up[i], -limit, limit);
        const glu = x / (1.0 + @exp(alpha * (-x)));
        output[i] = glu * (y + 1.0);
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
    attn_scale: f32,
) void {
    const gqa_ratio = n_heads / @max(n_kv_heads, 1);
    const kv_dim = n_kv_heads * head_dim;
    const scale = if (attn_scale != 0) attn_scale else 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
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

/// Dequantize one row of quantized weights to f32. Supports f32, f16, Q4_K, Q5_K, Q6_K, and Q8_0.
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
        .q5_0 => {
            const block_size: usize = 32;
            const bpb: usize = 22;
            const bpr = @as(usize, cols) / block_size;
            const row_off = @as(usize, row) * bpr * bpb;
            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bo = row_off + b * bpb;
                const scale_bits = std.mem.readInt(u16, raw_data[bo..][0..2], .little);
                const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
                const qh = std.mem.readInt(u32, raw_data[bo + 2 ..][0..4], .little);
                const qs = raw_data[bo + 6 .. bo + bpb];
                for (0..16) |j| {
                    const q_byte = qs[j];
                    const lo = q_byte & 0x0F;
                    const hi = q_byte >> 4;
                    const bit_lo = (qh >> @intCast(j)) & 1;
                    const bit_hi = (qh >> @intCast(j + 16)) & 1;
                    output[out_i + j] = scale * @as(f32, @floatFromInt(@as(i32, @intCast(lo | (bit_lo << 4))) - 16));
                    output[out_i + 16 + j] = scale * @as(f32, @floatFromInt(@as(i32, @intCast(hi | (bit_hi << 4))) - 16));
                }
                out_i += block_size;
            }
        },
        .q5_1 => {
            const block_size: usize = 32;
            const bpb: usize = 24;
            const bpr = @as(usize, cols) / block_size;
            const row_off = @as(usize, row) * bpr * bpb;
            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bo = row_off + b * bpb;
                const scale_bits = std.mem.readInt(u16, raw_data[bo..][0..2], .little);
                const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
                const min_bits = std.mem.readInt(u16, raw_data[bo + 2 ..][0..2], .little);
                const min_val: f32 = @floatCast(@as(f16, @bitCast(min_bits)));
                const qh = std.mem.readInt(u32, raw_data[bo + 4 ..][0..4], .little);
                const qs = raw_data[bo + 8 .. bo + bpb];
                for (0..16) |j| {
                    const q_byte = qs[j];
                    const lo = q_byte & 0x0F;
                    const hi = q_byte >> 4;
                    const bit_lo = (qh >> @intCast(j)) & 1;
                    const bit_hi = (qh >> @intCast(j + 16)) & 1;
                    output[out_i + j] = scale * @as(f32, @floatFromInt(lo | (bit_lo << 4))) + min_val;
                    output[out_i + 16 + j] = scale * @as(f32, @floatFromInt(hi | (bit_hi << 4))) + min_val;
                }
                out_i += block_size;
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
        .mxfp4 => {
            const bpb: usize = 17;
            const bpr = @as(usize, cols) / 32;
            const row_off = @as(usize, row) * bpr * bpb;
            const lut = [16]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };
            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bo = row_off + b * bpb;
                const exp_byte = raw_data[bo];
                const d: f32 = @bitCast(if (exp_byte == 0) @as(u32, 0x00400000) else @as(u32, @intCast(exp_byte)) << 23);
                const qs = raw_data[bo + 1 .. bo + 17];
                for (0..16) |j| {
                    output[out_i + j] = d * lut[qs[j] & 0x0F];
                    output[out_i + j + 16] = d * lut[qs[j] >> 4];
                }
                out_i += 32;
            }
        },
        else => {
            log.warn("Unsupported quant type {d}, using zeros", .{@intFromEnum(quant_type)});
            @memset(output, 0);
        },
    }
}

fn dotQ8_0Row(raw_data: []const u8, row: u32, cols: u32, input: [*]const f32) f32 {
    const block_size: usize = 32;
    const bpb: usize = 34;
    const bpr = @as(usize, cols) / block_size;
    const row_off = @as(usize, row) * bpr * bpb;
    var dot: f32 = 0;
    var in_i: usize = 0;
    for (0..bpr) |b| {
        const bo = row_off + b * bpb;
        const scale_bits = std.mem.readInt(u16, raw_data[bo..][0..2], .little);
        const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
        for (0..block_size) |j| {
            const q: i8 = @bitCast(raw_data[bo + 2 + j]);
            const w = @as(f32, @floatFromInt(q)) * scale;
            dot += w * input[in_i];
            in_i += 1;
        }
    }
    return dot;
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

/// Select the top-k logits, apply softmax, and write indices and weights.
/// SOFTMAX_WEIGHT gating (gpt-oss): select top-k from raw logits, then softmax over selected.
pub fn topKSoftmaxWeight(logits: []const f32, k: u32, out_ids: []u32, out_weights: []f32) void {
    const n = logits.len;
    var used = [_]bool{false} ** 256;
    for (0..k) |ki| {
        var best_idx: u32 = 0;
        var best_val: f32 = -std.math.inf(f32);
        for (0..n) |i| {
            if (!used[i] and logits[i] > best_val) {
                best_val = logits[i];
                best_idx = @intCast(i);
            }
        }
        out_ids[ki] = best_idx;
        out_weights[ki] = logits[best_idx];
        used[best_idx] = true;
    }
    var max_sel: f32 = -std.math.inf(f32);
    for (0..k) |i| if (out_weights[i] > max_sel) {
        max_sel = out_weights[i];
    };
    var sum: f32 = 0;
    for (0..k) |i| {
        out_weights[i] = @exp(out_weights[i] - max_sel);
        sum += out_weights[i];
    }
    if (sum > 0) for (0..k) |i| {
        out_weights[i] /= sum;
    };
}

/// Select top-k experts by logit value, returning softmax-normalized probabilities.
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
        .q5_1 => engine.dmmv_q5_1_moe_pipe.handle != null,
        .q5_k => engine.dmmv_q5k_moe_pipe.handle != null,
        .q6_k => engine.dmmv_q6k_moe_pipe.handle != null,
        else => false,
    };
}

fn isF32Tensor(tensor: ?*const metal_loader.LoadedTensor) bool {
    return if (tensor) |t| t.info.type_ == .f32 else false;
}

fn canUseGpuRoutedBatchedMoe(engine: *const InferenceEngine, lt: LayerTensors) bool {
    if (engine.config.architecture == .gemma) {
        const gate_up_exps = lt.ffn_gate_up_exps orelse return false;
        const down_exps = lt.ffn_down_exps orelse return false;
        const gate_shexp = lt.ffn_gate_shexp orelse return false;
        const up_shexp = lt.ffn_up_shexp orelse return false;
        const down_shexp = lt.ffn_down_shexp orelse return false;

        if (engine.config.n_experts_used != 8) return false;
        if (gate_up_exps.info.type_ != .q4_k) return false;
        if (!canUseGpuRoutedMoeDown(engine, down_exps.info.type_)) return false;
        if (engine.dmmvPipelineForType(gate_shexp, engine.config.shared_expert_intermediate_dim, engine.config.hidden_dim) == null) return false;
        if (engine.dmmvPipelineForType(up_shexp, engine.config.shared_expert_intermediate_dim, engine.config.hidden_dim) == null) return false;
        if (engine.dmmvPipelineForType(down_shexp, engine.config.hidden_dim, engine.config.shared_expert_intermediate_dim) == null) return false;
        if (lt.ffn_gate_inp == null or lt.ffn_gate_inp_bias != null) return false;
        if (lt.ffn_gate_exps_bias != null or lt.ffn_up_exps_bias != null or lt.ffn_down_exps_bias != null) return false;
        if (!isF32Tensor(lt.ffn_gate_inp_scale) or !isF32Tensor(lt.pre_ffw_norm_2) or
            !isF32Tensor(lt.post_ffw_norm_1) or !isF32Tensor(lt.post_ffw_norm_2) or
            !isF32Tensor(lt.ffn_down_exps_scale))
        {
            return false;
        }
        if (lt.ffn_gate_inp_shexp) |gate| {
            if (engine.dmmvPipelineForType(gate, 1, engine.config.hidden_dim) == null) return false;
        }
        return engine.softmax_topk_scaled_pipe.handle != null and
            engine.geglu_batched_pipe.handle != null and
            engine.geglu_pipe.handle != null and
            engine.moe_weighted_acc_scaled_pipe.handle != null and
            engine.sigmoid_scale_acc_pipe.handle != null and
            engine.scale_acc_pipe.handle != null;
    }
    if (lt.ffn_gate_up_exps != null) return false;
    const gate_exps = lt.ffn_gate_exps orelse return false;
    const up_exps = lt.ffn_up_exps orelse return false;
    const down_exps = lt.ffn_down_exps orelse return false;

    if (engine.config.n_experts_used != 8) return false;
    if (!canUseGpuRoutedMoeDown(engine, gate_exps.info.type_)) return false;
    if (!canUseGpuRoutedMoeDown(engine, up_exps.info.type_)) return false;
    if (!canUseGpuRoutedMoeDown(engine, down_exps.info.type_)) return false;

    const has_shexp = lt.ffn_gate_shexp != null and lt.ffn_up_shexp != null and lt.ffn_down_shexp != null;
    if (has_shexp and engine.moe_weighted_acc_shared_pipe.handle == null) return false;

    return engine.softmax_topk_pipe.handle != null and engine.moe_weighted_acc_pipe.handle != null;
}

const SliceDiff = struct {
    max_abs: f32,
    max_idx: usize,
    rms: f64,
};

fn diffF32Slices(expected: []const f32, actual: []const f32) SliceDiff {
    const n = @min(expected.len, actual.len);
    var max_abs: f32 = 0;
    var max_idx: usize = 0;
    var sum_sq: f64 = 0;
    for (0..n) |i| {
        const delta = actual[i] - expected[i];
        const abs_delta = @abs(delta);
        if (abs_delta > max_abs) {
            max_abs = abs_delta;
            max_idx = i;
        }
        sum_sq += @as(f64, delta) * @as(f64, delta);
    }
    const rms = if (n > 0) @sqrt(sum_sq / @as(f64, @floatFromInt(n))) else 0;
    return .{ .max_abs = max_abs, .max_idx = max_idx, .rms = rms };
}

fn shouldValidateGemmaMoe(engine: *const InferenceEngine, layer_idx: usize) bool {
    return engine.gemma_moe_validation_enabled and
        engine.config.architecture == .gemma and
        engine.position == 0 and
        layer_idx == 0;
}

fn validateGemmaMoePostVector(
    engine: *InferenceEngine,
    layer_idx: usize,
    lt: LayerTensors,
    gate_up_layout: MoeGateUpLayout,
    expert_ids: []const u32,
    adjusted_expert_weights: []const f32,
    shexp_gate_weight: f32,
    expert_input_ptr: [*]const f32,
    shared_input_ptr: [*]const f32,
    actual_post_moe: []const f32,
    hidden_dim: u32,
    inter_dim: u32,
    shexp_inter_dim: u32,
) !void {
    const cfg = engine.config;
    const allocator = engine.allocator;
    const mmap = engine.model.mmap_data orelse return error.NoMmapData;
    const tdo = engine.model.gguf_file.tensor_data_offset;
    const gate_exps = gate_up_layout.gate_tensor;
    const up_exps = gate_up_layout.up_tensor;
    const down_exps = lt.ffn_down_exps orelse return error.MissingTensor;
    const gate_shexp = lt.ffn_gate_shexp orelse lt.ffn_gate;
    const up_shexp = lt.ffn_up_shexp orelse lt.ffn_up;
    const down_shexp = lt.ffn_down_shexp orelse lt.ffn_down;
    const has_shexp = gate_shexp != null and up_shexp != null and down_shexp != null;
    const expert_down_bytes = expertSliceBytes(down_exps.info.type_, hidden_dim, inter_dim);

    const expected_expert = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(expected_expert);
    const expected_shared = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(expected_shared);
    const expected_total = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(expected_total);
    @memset(expected_expert, 0);
    @memset(expected_shared, 0);

    const gate_tmp = try allocator.alloc(f32, inter_dim);
    defer allocator.free(gate_tmp);
    const up_tmp = try allocator.alloc(f32, inter_dim);
    defer allocator.free(up_tmp);
    const act_tmp = try allocator.alloc(f32, inter_dim);
    defer allocator.free(act_tmp);
    const down_tmp = try allocator.alloc(f32, hidden_dim);
    defer allocator.free(down_tmp);

    for (expert_ids, adjusted_expert_weights) |eid, weight| {
        try cpuDmmvFallback(mmap, gate_exps, tdo, expert_input_ptr, gate_tmp.ptr, inter_dim, hidden_dim, gate_up_layout.gateOffset(eid), allocator);
        try cpuDmmvFallback(mmap, up_exps, tdo, expert_input_ptr, up_tmp.ptr, inter_dim, hidden_dim, gate_up_layout.upOffset(eid), allocator);
        cpuGeGLU(gate_tmp.ptr, up_tmp.ptr, act_tmp.ptr, inter_dim);
        try cpuDmmvFallback(mmap, down_exps, tdo, act_tmp.ptr, down_tmp.ptr, hidden_dim, inter_dim, eid * expert_down_bytes, allocator);
        if (lt.ffn_down_exps_bias) |bias| {
            addBiasFromTensorSlice(engine, down_tmp.ptr, bias, eid, hidden_dim);
        }
        for (0..hidden_dim) |i| expected_expert[i] += weight * down_tmp[i];
    }
    if (lt.post_ffw_norm_2) |post_norm_2_t| {
        const post_norm_2 = try tensorF32Slice(engine, post_norm_2_t, hidden_dim);
        cpuRmsNormMul(expected_expert.ptr, post_norm_2, expected_expert.ptr, hidden_dim, 1, cfg.rms_norm_eps);
    }

    if (has_shexp) {
        const gate_sh = try allocator.alloc(f32, shexp_inter_dim);
        defer allocator.free(gate_sh);
        const up_sh = try allocator.alloc(f32, shexp_inter_dim);
        defer allocator.free(up_sh);
        const act_sh = try allocator.alloc(f32, shexp_inter_dim);
        defer allocator.free(act_sh);
        try cpuDmmvFallback(mmap, gate_shexp.?, tdo, shared_input_ptr, gate_sh.ptr, shexp_inter_dim, hidden_dim, 0, allocator);
        try cpuDmmvFallback(mmap, up_shexp.?, tdo, shared_input_ptr, up_sh.ptr, shexp_inter_dim, hidden_dim, 0, allocator);
        cpuGeGLU(gate_sh.ptr, up_sh.ptr, act_sh.ptr, shexp_inter_dim);
        try cpuDmmvFallback(mmap, down_shexp.?, tdo, act_sh.ptr, expected_shared.ptr, hidden_dim, shexp_inter_dim, 0, allocator);
        if (lt.post_ffw_norm_1) |post_norm_1_t| {
            const post_norm_1 = try tensorF32Slice(engine, post_norm_1_t, hidden_dim);
            cpuRmsNormMul(expected_shared.ptr, post_norm_1, expected_shared.ptr, hidden_dim, 1, cfg.rms_norm_eps);
        }
        for (0..hidden_dim) |i| expected_shared[i] *= shexp_gate_weight;
    }

    for (0..hidden_dim) |i| expected_total[i] = expected_expert[i] + expected_shared[i];
    if (engine.post_ffn_norm_present[layer_idx]) {
        const post_norm_ptr: [*]const f32 = @ptrCast(@alignCast(engine.post_ffn_norm_bufs[layer_idx].cpu_ptr.?));
        cpuRmsNormMul(expected_total.ptr, post_norm_ptr[0..hidden_dim], expected_total.ptr, hidden_dim, 1, cfg.rms_norm_eps);
    }

    const diff = diffF32Slices(expected_total[0..hidden_dim], actual_post_moe);
    const tol: f32 = 1e-3;
    if (diff.max_abs > tol) {
        log.err("Gemma MoE validate[failed]: pos={d} layer={d} post_moe max_abs_diff={d:.6} idx={d} ref={d:.6} candidate={d:.6} rms_diff={d:.6} tol={d:.6}", .{
            engine.position,
            layer_idx,
            diff.max_abs,
            diff.max_idx,
            expected_total[diff.max_idx],
            actual_post_moe[diff.max_idx],
            diff.rms,
            tol,
        });
        return error.GemmaMoeValidationFailed;
    }
    log.info("Gemma MoE validate[ok]: pos={d} layer={d} post_moe max_abs_diff={d:.6} idx={d} ref={d:.6} candidate={d:.6} rms_diff={d:.6} tol={d:.6}", .{
        engine.position,
        layer_idx,
        diff.max_abs,
        diff.max_idx,
        expected_total[diff.max_idx],
        actual_post_moe[diff.max_idx],
        diff.rms,
        tol,
    });
}

fn runGemmaExplicitMoeFallback(
    engine: *InferenceEngine,
    profile: ?*RuntimeProfile,
    layer_idx: usize,
    lt: LayerTensors,
    hidden_dim: u32,
    inter_dim: u32,
    shexp_inter_dim: u32,
) !void {
    const cfg = engine.config;
    const mmap = engine.model.mmap_data orelse return error.NoMmapData;
    const tdo = engine.model.gguf_file.tensor_data_offset;
    const router_t = lt.ffn_gate_inp orelse return error.MissingTensor;
    const gate_up_layout = try resolveMoeGateUpLayout(lt, inter_dim, hidden_dim);
    const gate_exps = gate_up_layout.gate_tensor;
    const up_exps = gate_up_layout.up_tensor;
    const down_exps = lt.ffn_down_exps orelse return error.MissingTensor;
    const gate_shexp = lt.ffn_gate_shexp orelse lt.ffn_gate;
    const up_shexp = lt.ffn_up_shexp orelse lt.ffn_up;
    const down_shexp = lt.ffn_down_shexp orelse lt.ffn_down;
    const has_shexp = gate_shexp != null and up_shexp != null and down_shexp != null;
    const expert_down_bytes = expertSliceBytes(down_exps.info.type_, hidden_dim, inter_dim);
    const folded_layer_output_scale: f32 = if (engine.debug_validation_enabled) 1.0 else engine.layer_output_scales[layer_idx];

    const hidden_ptr: [*]const f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));
    const norm_ptr: [*]const f32 = @ptrCast(@alignCast(engine.norm_buf.cpu_ptr.?));
    const router_ptr: [*]f32 = @ptrCast(@alignCast(engine.router_logits_buf.cpu_ptr.?));
    const router_input_ptr: [*]f32 = @ptrCast(@alignCast(engine.residual_buf.cpu_ptr.?));
    const expert_accum_ptr: [*]f32 = @ptrCast(@alignCast(engine.moe_out_buf.cpu_ptr.?));
    const shared_out_ptr: [*]f32 = @ptrCast(@alignCast(engine.down_buf.cpu_ptr.?));
    const unit_weights: [*]const f32 = @ptrCast(@alignCast(engine.unit_rms_norm_weights.cpu_ptr.?));
    const should_debug_moe_delta = engine.debug_validation_enabled and (engine.position <= 1 or engine.position == 5) and layer_idx == 0;
    const hidden_before_snapshot: ?[]f32 = if (should_debug_moe_delta) blk: {
        const snap = try engine.allocator.alloc(f32, hidden_dim);
        @memcpy(snap, hidden_ptr[0..hidden_dim]);
        break :blk snap;
    } else null;
    defer if (hidden_before_snapshot) |snap| engine.allocator.free(snap);

    const router_start = profileStart(profile != null);
    cpuRmsNormMul(hidden_ptr, unit_weights[0..hidden_dim], router_input_ptr, hidden_dim, 1, cfg.rms_norm_eps);
    if (lt.ffn_gate_inp_scale) |scale_t| {
        const scale = try tensorF32Slice(engine, scale_t, hidden_dim);
        cpuMulInPlace(router_input_ptr, scale, hidden_dim);
    }
    try cpuDmmvFallback(mmap, router_t, tdo, router_input_ptr, router_ptr, cfg.n_experts, hidden_dim, 0, engine.allocator);
    if (lt.ffn_gate_inp_bias) |b| {
        addBiasFromTensor(engine, router_ptr, b, cfg.n_experts);
    }
    {
        const router_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden_dim)));
        for (router_ptr[0..cfg.n_experts]) |*value| value.* *= router_scale;
    }
    var expert_ids: [16]u32 = undefined;
    var expert_weights: [16]f32 = undefined;
    topKSoftmax(@as([*]const f32, router_ptr)[0..cfg.n_experts], cfg.n_experts_used, expert_ids[0..cfg.n_experts_used], expert_weights[0..cfg.n_experts_used]);
    if (profile) |p| p.router_cpu_ns += profileElapsedNs(router_start);

    var adjusted_expert_weights: [16]f32 = expert_weights;
    // Fold ffn_down_exps_scale into the per-token expert weights so the
    // accumulate kernel absorbs it in one fused multiply. Lets us drop the
    // CPU roundtrip that used to multiply each expert's down output by
    // scales[expert_ids[ei]] before the accumulate commitAndWait.
    if (lt.ffn_down_exps_scale) |scale_t| {
        const scales = try tensorF32Slice(engine, scale_t, cfg.n_experts);
        for (0..cfg.n_experts_used) |ei| {
            adjusted_expert_weights[ei] *= scales[expert_ids[ei]];
        }
    }

    // Gemma 4's router is CPU-computed (see topKSoftmax above); the batched
    // MoE DMMV shaders read expert IDs from router_output_buf. Marshal the
    // CPU IDs into the device-side buffer here so the GPU path can see them.
    // Weights slot goes unused by the DMMV (the weighted accumulate takes
    // `adjusted_expert_weights` as push constants), so leave it alone.
    const use_batched_moe_dispatch =
        gate_exps.info.type_ == .q4_k and
        up_exps.info.type_ == .q4_k;
    if (use_batched_moe_dispatch) {
        const router_out: [*]u32 = @ptrCast(@alignCast(engine.router_output_buf.cpu_ptr.?));
        for (0..cfg.n_experts_used) |ei| router_out[ei] = expert_ids[ei];
    }

    var shexp_gate_weight: f32 = 1.0;
    if (lt.ffn_gate_inp_shexp != null) {
        if (engine.shexp_gate_weights) |sgw| {
            const gate_w = sgw[layer_idx];
            if (gate_w.len >= hidden_dim) {
                var dot: f32 = 0;
                for (0..hidden_dim) |i| dot += gate_w[i] * norm_ptr[i];
                shexp_gate_weight = 1.0 / (1.0 + @exp(-dot));
            }
        }
    }

    const expert_input_buf: *const MetalBuffer = blk: {
        if (lt.pre_ffw_norm_2) |pre_norm_t| {
            const pre_norm = try tensorF32Slice(engine, pre_norm_t, hidden_dim);
            cpuRmsNormMul(hidden_ptr, pre_norm, router_input_ptr, hidden_dim, 1, cfg.rms_norm_eps);
            break :blk &engine.residual_buf;
        }
        break :blk &engine.norm_buf;
    };
    const expert_input_ptr: [*]const f32 = @ptrCast(@alignCast(expert_input_buf.cpu_ptr.?));

    const moe_record_start = profileStart(profile != null);
    var cmd = try beginProfiledCommand(engine, profile);

    if (use_batched_moe_dispatch) {
        // Batched Q4_K gate+up through the fused ffn_gate_up_exps tensor.
        // gate_expert_stride spans each expert's full 2*inter_dim slice; the
        // gate/up base offsets pick the gate half (0) vs the up half
        // (gate_half_bytes). Collapses 16 per-expert dispatches into 2.
        try dispatchDmmvMoeOnCmd(
            engine,
            &cmd,
            gate_exps,
            expert_input_buf,
            &engine.expert_gate_batch_buf,
            &engine.router_output_buf,
            inter_dim,
            hidden_dim,
            gate_up_layout.gate_expert_stride,
            0,
            gate_up_layout.gate_base_offset,
        );
        try dispatchDmmvMoeOnCmd(
            engine,
            &cmd,
            up_exps,
            expert_input_buf,
            &engine.expert_up_batch_buf,
            &engine.router_output_buf,
            inter_dim,
            hidden_dim,
            gate_up_layout.up_expert_stride,
            0,
            gate_up_layout.up_base_offset,
        );
    } else {
        for (0..cfg.n_experts_used) |ei| {
            const eid = expert_ids[ei];
            dispatchDmmvOnCmd(engine, &cmd, gate_exps, expert_input_buf, &engine.expert_gate_bufs[ei], inter_dim, hidden_dim, gate_up_layout.gateOffset(eid));
            dispatchDmmvOnCmd(engine, &cmd, up_exps, expert_input_buf, &engine.expert_up_bufs[ei], inter_dim, hidden_dim, gate_up_layout.upOffset(eid));
        }
    }
    if (has_shexp) {
        dispatchDmmvOnCmd(engine, &cmd, gate_shexp.?, &engine.norm_buf, &engine.gate_buf, shexp_inter_dim, hidden_dim, 0);
        dispatchDmmvOnCmd(engine, &cmd, up_shexp.?, &engine.norm_buf, &engine.up_buf, shexp_inter_dim, hidden_dim, 0);
    }
    profileBarrier(&cmd, profile, .fallback_moe);

    if (use_batched_moe_dispatch) {
        // Batched GeGLU across all n_experts_used experts in one dispatch.
        const push = SwiGLUPush{ .n = inter_dim };
        const bufs = [_]*const MetalBuffer{ &engine.expert_gate_batch_buf, &engine.expert_swiglu_batch_buf, &engine.expert_up_batch_buf };
        cmd.dispatchV2(&engine.geglu_batched_pipe, .{ (inter_dim + 63) / 64, cfg.n_experts_used, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SwiGLUPush), 0);
    } else {
        for (0..cfg.n_experts_used) |ei| {
            dispatchFfnActivationOnCmd(engine, &cmd, &engine.expert_gate_bufs[ei], &engine.expert_swiglu_bufs[ei], &engine.expert_up_bufs[ei], inter_dim);
        }
    }
    if (has_shexp) {
        dispatchFfnActivationOnCmd(engine, &cmd, &engine.gate_buf, &engine.swiglu_buf, &engine.up_buf, shexp_inter_dim);
    }
    profileBarrier(&cmd, profile, .fallback_moe);

    const down_supported = engine.dmmvPipelineForType(down_exps, hidden_dim, inter_dim) != null;
    const use_batched_down =
        use_batched_moe_dispatch and
        down_supported and
        down_exps.info.type_ == .q5_1 and
        cfg.n_experts_used == 8 and
        lt.ffn_down_exps_bias == null;
    if (down_supported) {
        const input_stride_bytes: u32 = inter_dim * @sizeOf(f32);
        if (use_batched_down) {
            try dispatchDmmvMoeOnCmd(
                engine,
                &cmd,
                down_exps,
                &engine.expert_swiglu_batch_buf,
                &engine.expert_down_batch_buf,
                &engine.router_output_buf,
                hidden_dim,
                inter_dim,
                expert_down_bytes,
                inter_dim,
                0,
            );
        } else {
            for (0..cfg.n_experts_used) |ei| {
                const eid = expert_ids[ei];
                if (use_batched_moe_dispatch) {
                    const x_off_bytes: u32 = @as(u32, @intCast(ei)) * input_stride_bytes;
                    dispatchDmmvOnCmdWithInputOffset(
                        engine,
                        &cmd,
                        down_exps,
                        &engine.expert_swiglu_batch_buf,
                        &engine.expert_down_bufs[ei],
                        hidden_dim,
                        inter_dim,
                        eid * expert_down_bytes,
                        x_off_bytes,
                    );
                } else {
                    dispatchDmmvOnCmd(engine, &cmd, down_exps, &engine.expert_swiglu_bufs[ei], &engine.expert_down_bufs[ei], hidden_dim, inter_dim, eid * expert_down_bytes);
                }
            }
        }
    }
    if (has_shexp) {
        dispatchDmmvOnCmd(engine, &cmd, down_shexp.?, &engine.swiglu_buf, &engine.down_buf, hidden_dim, shexp_inter_dim, 0);
    }
    // The interior commit used to sync here was needed for:
    //  (a) the validation-only block below to read expert_{gate,up,swiglu}_bufs
    //  (b) the CPU down fallback when !down_supported
    //  (c) CPU per-expert bias add when ffn_down_exps_bias is present
    // In the steady-state Gemma 4 path none of those fire, so we leave cmd
    // open and let the weighted-accumulate dispatch below join the same CB.
    // This halves the per-layer commit-and-wait count (from 2 to 1), which
    // is the dominant cost on Metal for Gemma's MoE path.
    const needs_mid_commit = engine.debug_validation_enabled or
        !down_supported or
        lt.ffn_down_exps_bias != null;
    if (needs_mid_commit) {
        commitAndWaitProfiled(&cmd, profile);
    }

    if (engine.debug_validation_enabled and engine.position == 0 and layer_idx == 0 and cfg.n_experts_used > 0) {
        const debug_eid = expert_ids[0];
        const gpu_gate_ptr: [*]const f32 = @ptrCast(@alignCast(engine.expert_gate_bufs[0].cpu_ptr.?));
        const gpu_up_ptr: [*]const f32 = @ptrCast(@alignCast(engine.expert_up_bufs[0].cpu_ptr.?));
        const gpu_act_ptr: [*]const f32 = @ptrCast(@alignCast(engine.expert_swiglu_bufs[0].cpu_ptr.?));

        const gate_ref = try engine.allocator.alloc(f32, inter_dim);
        defer engine.allocator.free(gate_ref);
        const up_ref = try engine.allocator.alloc(f32, inter_dim);
        defer engine.allocator.free(up_ref);
        const act_ref = try engine.allocator.alloc(f32, inter_dim);
        defer engine.allocator.free(act_ref);

        try cpuDmmvFallback(mmap, gate_exps, tdo, expert_input_ptr, gate_ref.ptr, inter_dim, hidden_dim, gate_up_layout.gateOffset(debug_eid), engine.allocator);
        try cpuDmmvFallback(mmap, up_exps, tdo, expert_input_ptr, up_ref.ptr, inter_dim, hidden_dim, gate_up_layout.upOffset(debug_eid), engine.allocator);
        cpuGeGLU(gate_ref.ptr, up_ref.ptr, act_ref.ptr, inter_dim);

        logDebugSliceDiff(0, "gemma4_expert_gate", gate_ref[0..inter_dim], gpu_gate_ptr[0..inter_dim]);
        logDebugSliceDiff(0, "gemma4_expert_up", up_ref[0..inter_dim], gpu_up_ptr[0..inter_dim]);
        logDebugSliceDiff(0, "gemma4_expert_geglu", act_ref[0..inter_dim], gpu_act_ptr[0..inter_dim]);

        if (has_shexp) {
            const gpu_shared_gate_ptr: [*]const f32 = @ptrCast(@alignCast(engine.gate_buf.cpu_ptr.?));
            const gpu_shared_up_ptr: [*]const f32 = @ptrCast(@alignCast(engine.up_buf.cpu_ptr.?));
            const gpu_shared_act_ptr: [*]const f32 = @ptrCast(@alignCast(engine.swiglu_buf.cpu_ptr.?));
            const gpu_shared_down_ptr: [*]const f32 = @ptrCast(@alignCast(engine.down_buf.cpu_ptr.?));

            const gate_sh_ref = try engine.allocator.alloc(f32, shexp_inter_dim);
            defer engine.allocator.free(gate_sh_ref);
            const up_sh_ref = try engine.allocator.alloc(f32, shexp_inter_dim);
            defer engine.allocator.free(up_sh_ref);
            const act_sh_ref = try engine.allocator.alloc(f32, shexp_inter_dim);
            defer engine.allocator.free(act_sh_ref);
            const down_sh_ref = try engine.allocator.alloc(f32, hidden_dim);
            defer engine.allocator.free(down_sh_ref);

            try cpuDmmvFallback(mmap, gate_shexp.?, tdo, norm_ptr, gate_sh_ref.ptr, shexp_inter_dim, hidden_dim, 0, engine.allocator);
            try cpuDmmvFallback(mmap, up_shexp.?, tdo, norm_ptr, up_sh_ref.ptr, shexp_inter_dim, hidden_dim, 0, engine.allocator);
            cpuGeGLU(gate_sh_ref.ptr, up_sh_ref.ptr, act_sh_ref.ptr, shexp_inter_dim);
            try cpuDmmvFallback(mmap, down_shexp.?, tdo, act_sh_ref.ptr, down_sh_ref.ptr, hidden_dim, shexp_inter_dim, 0, engine.allocator);

            logDebugSliceDiff(0, "gemma4_shared_gate", gate_sh_ref[0..shexp_inter_dim], gpu_shared_gate_ptr[0..shexp_inter_dim]);
            logDebugSliceDiff(0, "gemma4_shared_up", up_sh_ref[0..shexp_inter_dim], gpu_shared_up_ptr[0..shexp_inter_dim]);
            logDebugSliceDiff(0, "gemma4_shared_geglu", act_sh_ref[0..shexp_inter_dim], gpu_shared_act_ptr[0..shexp_inter_dim]);
            logDebugSliceDiff(0, "gemma4_shared_down", down_sh_ref[0..hidden_dim], gpu_shared_down_ptr[0..hidden_dim]);
        }
    }

    if (!down_supported) {
        for (0..cfg.n_experts_used) |ei| {
            const eid = expert_ids[ei];
            const in_ptr: [*]const f32 = @ptrCast(@alignCast(engine.expert_swiglu_bufs[ei].cpu_ptr.?));
            const out_ptr: [*]f32 = @ptrCast(@alignCast(engine.expert_down_bufs[ei].cpu_ptr.?));
            try cpuDmmvFallback(mmap, down_exps, tdo, in_ptr, out_ptr, hidden_dim, inter_dim, eid * expert_down_bytes, engine.allocator);
        }
    }
    if (lt.ffn_down_exps_bias) |b| {
        for (0..cfg.n_experts_used) |ei| {
            addBiasFromTensorSlice(engine, @ptrCast(@alignCast(engine.expert_down_bufs[ei].cpu_ptr.?)), b, expert_ids[ei], hidden_dim);
        }
    }
    // ffn_down_exps_scale is now folded into adjusted_expert_weights above,
    // so the accumulate kernel absorbs it via its per-expert weight multiply
    // and we can skip the CPU rescale that used to live here.

    // Reopen the command buffer only if we committed it above; otherwise we
    // keep appending dispatches to the same CB for a single commit per layer.
    if (needs_mid_commit) {
        cmd = try beginProfiledCommand(engine, profile);
    } else {
        profileBarrier(&cmd, profile, .fallback_moe);
    }
    dispatchZeroF32OnCmd(engine, &cmd, &engine.moe_out_buf, hidden_dim);
    profileBarrier(&cmd, profile, .fallback_moe);
    if (cfg.n_experts_used == 8) {
        if (use_batched_down) {
            const moe_push = MoeAccBatchedPush{
                .n = hidden_dim,
                .expert_stride = hidden_dim,
                .w0 = adjusted_expert_weights[0],
                .w1 = adjusted_expert_weights[1],
                .w2 = adjusted_expert_weights[2],
                .w3 = adjusted_expert_weights[3],
                .w4 = adjusted_expert_weights[4],
                .w5 = adjusted_expert_weights[5],
                .w6 = adjusted_expert_weights[6],
                .w7 = adjusted_expert_weights[7],
                .w_sh = 0.0,
            };
            const moe_bufs = [_]*const MetalBuffer{
                &engine.moe_out_buf,
                &engine.expert_down_batch_buf,
                &engine.down_buf,
            };
            cmd.dispatchV2(&engine.moe_acc_batched_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &moe_bufs, &moe_push, @sizeOf(MoeAccBatchedPush), 3);
        } else {
            const moe_push = MoeAccPush{
                .n = hidden_dim,
                .w0 = adjusted_expert_weights[0],
                .w1 = adjusted_expert_weights[1],
                .w2 = adjusted_expert_weights[2],
                .w3 = adjusted_expert_weights[3],
                .w4 = adjusted_expert_weights[4],
                .w5 = adjusted_expert_weights[5],
                .w6 = adjusted_expert_weights[6],
                .w7 = adjusted_expert_weights[7],
                .w_sh = 0.0,
            };
            const moe_bufs = [_]*const MetalBuffer{
                &engine.moe_out_buf,
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
        }
    } else {
        for (0..cfg.n_experts_used) |ei| {
            const acc_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(adjusted_expert_weights[ei])) };
            const acc_bufs = [_]*const MetalBuffer{ &engine.moe_out_buf, &engine.expert_down_bufs[ei] };
            cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &acc_bufs, &acc_push, @sizeOf(ScaleAccPush), 0);
            profileBarrier(&cmd, profile, .fallback_moe);
        }
    }
    commitAndWaitProfiled(&cmd, profile);

    if (lt.post_ffw_norm_2) |post_norm_2_t| {
        const post_norm_2 = try tensorF32Slice(engine, post_norm_2_t, hidden_dim);
        cpuRmsNormMul(expert_accum_ptr, post_norm_2, expert_accum_ptr, hidden_dim, 1, cfg.rms_norm_eps);
    }
    if (has_shexp) {
        if (lt.post_ffw_norm_1) |post_norm_1_t| {
            const post_norm_1 = try tensorF32Slice(engine, post_norm_1_t, hidden_dim);
            cpuRmsNormMul(shared_out_ptr, post_norm_1, shared_out_ptr, hidden_dim, 1, cfg.rms_norm_eps);
        }
        for (0..hidden_dim) |i| {
            expert_accum_ptr[i] += shexp_gate_weight * shared_out_ptr[i];
        }
    }
    if (engine.post_ffn_norm_present[layer_idx]) {
        const post_norm_ptr: [*]const f32 = @ptrCast(@alignCast(engine.post_ffn_norm_bufs[layer_idx].cpu_ptr.?));
        cpuRmsNormMul(expert_accum_ptr, post_norm_ptr[0..hidden_dim], expert_accum_ptr, hidden_dim, 1, cfg.rms_norm_eps);
    }
    if (shouldValidateGemmaMoe(engine, layer_idx)) {
        const validation_start = profileStart(profile != null);
        try validateGemmaMoePostVector(
            engine,
            layer_idx,
            lt,
            gate_up_layout,
            expert_ids[0..cfg.n_experts_used],
            adjusted_expert_weights[0..cfg.n_experts_used],
            shexp_gate_weight,
            expert_input_ptr,
            norm_ptr,
            expert_accum_ptr[0..hidden_dim],
            hidden_dim,
            inter_dim,
            shexp_inter_dim,
        );
        if (profile) |p| p.debug_validation_ns += profileElapsedNs(validation_start);
    }
    {
        const hidden_out: [*]f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));
        if (folded_layer_output_scale != 1.0) {
            for (0..hidden_dim) |i| hidden_out[i] = (hidden_out[i] + expert_accum_ptr[i]) * folded_layer_output_scale;
        } else {
            for (0..hidden_dim) |i| hidden_out[i] += expert_accum_ptr[i];
        }
    }
    if (hidden_before_snapshot) |hidden_before| {
        const expected_expert = try engine.allocator.alloc(f32, hidden_dim);
        defer engine.allocator.free(expected_expert);
        const expected_shared = try engine.allocator.alloc(f32, hidden_dim);
        defer engine.allocator.free(expected_shared);
        const expected_total = try engine.allocator.alloc(f32, hidden_dim);
        defer engine.allocator.free(expected_total);
        @memset(expected_expert, 0);
        @memset(expected_shared, 0);

        const gate_tmp = try engine.allocator.alloc(f32, inter_dim);
        defer engine.allocator.free(gate_tmp);
        const up_tmp = try engine.allocator.alloc(f32, inter_dim);
        defer engine.allocator.free(up_tmp);
        const act_tmp = try engine.allocator.alloc(f32, inter_dim);
        defer engine.allocator.free(act_tmp);
        const down_tmp = try engine.allocator.alloc(f32, hidden_dim);
        defer engine.allocator.free(down_tmp);

        for (expert_ids[0..cfg.n_experts_used], adjusted_expert_weights[0..cfg.n_experts_used]) |eid, weight| {
            try cpuDmmvFallback(mmap, gate_exps, tdo, expert_input_ptr, gate_tmp.ptr, inter_dim, hidden_dim, gate_up_layout.gateOffset(eid), engine.allocator);
            try cpuDmmvFallback(mmap, up_exps, tdo, expert_input_ptr, up_tmp.ptr, inter_dim, hidden_dim, gate_up_layout.upOffset(eid), engine.allocator);
            cpuGeGLU(gate_tmp.ptr, up_tmp.ptr, act_tmp.ptr, inter_dim);
            try cpuDmmvFallback(mmap, down_exps, tdo, act_tmp.ptr, down_tmp.ptr, hidden_dim, inter_dim, eid * expert_down_bytes, engine.allocator);
            if (lt.ffn_down_exps_bias) |b| {
                addBiasFromTensorSlice(engine, down_tmp.ptr, b, eid, hidden_dim);
            }
            if (lt.ffn_down_exps_scale) |scale_t| {
                const scales = try tensorF32Slice(engine, scale_t, cfg.n_experts);
                cpuMulScalarInPlace(down_tmp.ptr, scales[eid], hidden_dim);
            }
            for (0..hidden_dim) |i| expected_expert[i] += weight * down_tmp[i];
        }
        if (lt.post_ffw_norm_2) |post_norm_2_t| {
            const post_norm_2 = try tensorF32Slice(engine, post_norm_2_t, hidden_dim);
            cpuRmsNormMul(expected_expert.ptr, post_norm_2, expected_expert.ptr, hidden_dim, 1, cfg.rms_norm_eps);
        }

        if (has_shexp) {
            const gate_sh = try engine.allocator.alloc(f32, shexp_inter_dim);
            defer engine.allocator.free(gate_sh);
            const up_sh = try engine.allocator.alloc(f32, shexp_inter_dim);
            defer engine.allocator.free(up_sh);
            const act_sh = try engine.allocator.alloc(f32, shexp_inter_dim);
            defer engine.allocator.free(act_sh);
            try cpuDmmvFallback(mmap, gate_shexp.?, tdo, norm_ptr, gate_sh.ptr, shexp_inter_dim, hidden_dim, 0, engine.allocator);
            try cpuDmmvFallback(mmap, up_shexp.?, tdo, norm_ptr, up_sh.ptr, shexp_inter_dim, hidden_dim, 0, engine.allocator);
            cpuGeGLU(gate_sh.ptr, up_sh.ptr, act_sh.ptr, shexp_inter_dim);
            try cpuDmmvFallback(mmap, down_shexp.?, tdo, act_sh.ptr, expected_shared.ptr, hidden_dim, shexp_inter_dim, 0, engine.allocator);
            if (lt.post_ffw_norm_1) |post_norm_1_t| {
                const post_norm_1 = try tensorF32Slice(engine, post_norm_1_t, hidden_dim);
                cpuRmsNormMul(expected_shared.ptr, post_norm_1, expected_shared.ptr, hidden_dim, 1, cfg.rms_norm_eps);
            }
            for (0..hidden_dim) |i| expected_shared[i] *= shexp_gate_weight;
        }

        for (0..hidden_dim) |i| expected_total[i] = expected_expert[i] + expected_shared[i];
        if (engine.post_ffn_norm_present[layer_idx]) {
            const post_norm_ptr: [*]const f32 = @ptrCast(@alignCast(engine.post_ffn_norm_bufs[layer_idx].cpu_ptr.?));
            cpuRmsNormMul(expected_total.ptr, post_norm_ptr[0..hidden_dim], expected_total.ptr, hidden_dim, 1, cfg.rms_norm_eps);
        }

        const hidden_after: [*]const f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));
        const actual_delta = try engine.allocator.alloc(f32, hidden_dim);
        defer engine.allocator.free(actual_delta);
        for (0..hidden_dim) |i| actual_delta[i] = hidden_after[i] - hidden_before[i];
        log.info("GEMMA4_MOE_DEBUG pos={d} layer={d}", .{ engine.position, layer_idx });
        logDebugSliceDiff(0, "gemma4_moe_delta", expected_total[0..hidden_dim], actual_delta[0..hidden_dim]);
    }
    if (profile) |p| p.fallback_moe_record_ns += profileElapsedNs(moe_record_start);
}

fn recordGemmaGpuRoutedMoeOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    profile: ?*RuntimeProfile,
    layer_idx: usize,
    lt: LayerTensors,
    hidden_dim: u32,
    inter_dim: u32,
    shexp_inter_dim: u32,
) !void {
    const cfg = engine.config;
    const gate_up_layout = try resolveMoeGateUpLayout(lt, inter_dim, hidden_dim);
    const gate_exps = gate_up_layout.gate_tensor;
    const up_exps = gate_up_layout.up_tensor;
    const down_exps = lt.ffn_down_exps orelse return error.MissingTensor;
    const down_scales = lt.ffn_down_exps_scale orelse return error.MissingTensor;
    const pre_ffw_norm_2 = lt.pre_ffw_norm_2 orelse return error.MissingTensor;
    const post_ffw_norm_1 = lt.post_ffw_norm_1 orelse return error.MissingTensor;
    const post_ffw_norm_2 = lt.post_ffw_norm_2 orelse return error.MissingTensor;
    const gate_shexp = lt.ffn_gate_shexp orelse return error.MissingTensor;
    const up_shexp = lt.ffn_up_shexp orelse return error.MissingTensor;
    const down_shexp = lt.ffn_down_shexp orelse return error.MissingTensor;
    const expert_down_bytes = expertSliceBytes(down_exps.info.type_, hidden_dim, inter_dim);

    const router_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden_dim)));
    dispatchSoftmaxTopkScaledOnCmd(engine, cmd, &engine.router_logits_buf, &engine.router_output_buf, cfg.n_experts, cfg.n_experts_used, router_scale);

    // llama.cpp's build_moe_ffn keeps selected_experts on device and feeds it
    // into mul_mat_id. For Gemma decode N=1, router_output_buf is that compact
    // selected_experts+weights row; the DMMV MoE kernels consume the expert ids
    // directly and the accumulate kernel consumes the weights.
    dispatchRmsNormOnCmdWithTensorWeights(engine, cmd, &engine.hidden_buf, &engine.residual_buf, pre_ffw_norm_2, hidden_dim, 1);
    dispatchZeroF32OnCmd(engine, cmd, &engine.moe_out_buf, hidden_dim);
    profileBarrier(cmd, profile, .gpu_routed_moe); // routes, expert input, and zeroed output visible

    try dispatchDmmvMoeOnCmd(
        engine,
        cmd,
        gate_exps,
        &engine.residual_buf,
        &engine.expert_gate_batch_buf,
        &engine.router_output_buf,
        inter_dim,
        hidden_dim,
        gate_up_layout.gate_expert_stride,
        0,
        gate_up_layout.gate_base_offset,
    );
    try dispatchDmmvMoeOnCmd(
        engine,
        cmd,
        up_exps,
        &engine.residual_buf,
        &engine.expert_up_batch_buf,
        &engine.router_output_buf,
        inter_dim,
        hidden_dim,
        gate_up_layout.up_expert_stride,
        0,
        gate_up_layout.up_base_offset,
    );
    dispatchDmmvOnCmd(engine, cmd, gate_shexp, &engine.norm_buf, &engine.gate_buf, shexp_inter_dim, hidden_dim, 0);
    dispatchDmmvOnCmd(engine, cmd, up_shexp, &engine.norm_buf, &engine.up_buf, shexp_inter_dim, hidden_dim, 0);
    if (lt.ffn_gate_inp_shexp) |gate_tensor| {
        dispatchDmmvOnCmd(engine, cmd, gate_tensor, &engine.norm_buf, &engine.router_logits_buf, 1, hidden_dim, 0);
    }
    profileBarrier(cmd, profile, .gpu_routed_moe); // gate/up/shared projections visible before GeGLU

    {
        const push = SwiGLUPush{ .n = inter_dim };
        const bufs = [_]*const MetalBuffer{ &engine.expert_gate_batch_buf, &engine.expert_swiglu_batch_buf, &engine.expert_up_batch_buf };
        cmd.dispatchV2(&engine.geglu_batched_pipe, .{ (inter_dim + 63) / 64, cfg.n_experts_used, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SwiGLUPush), 0);
    }
    dispatchFfnActivationOnCmd(engine, cmd, &engine.gate_buf, &engine.swiglu_buf, &engine.up_buf, shexp_inter_dim);
    profileBarrier(cmd, profile, .gpu_routed_moe); // activations visible before down projections

    try dispatchDmmvMoeOnCmd(
        engine,
        cmd,
        down_exps,
        &engine.expert_swiglu_batch_buf,
        &engine.expert_down_batch_buf,
        &engine.router_output_buf,
        hidden_dim,
        inter_dim,
        expert_down_bytes,
        inter_dim,
        0,
    );
    dispatchDmmvOnCmd(engine, cmd, down_shexp, &engine.swiglu_buf, &engine.down_buf, hidden_dim, shexp_inter_dim, 0);
    profileBarrier(cmd, profile, .gpu_routed_moe); // down outputs visible before accumulate

    dispatchMoeWeightedAccScaledOnCmd(
        engine,
        cmd,
        &engine.moe_out_buf,
        &engine.expert_down_batch_buf,
        &engine.router_output_buf,
        down_scales,
        hidden_dim,
        cfg.n_experts_used,
        hidden_dim,
    );
    profileBarrier(cmd, profile, .gpu_routed_moe); // expert contribution visible before post expert norm

    dispatchRmsNormOnCmdWithTensorWeights(engine, cmd, &engine.moe_out_buf, &engine.moe_out_buf, post_ffw_norm_2, hidden_dim, 1);
    dispatchRmsNormOnCmdWithTensorWeights(engine, cmd, &engine.down_buf, &engine.down_buf, post_ffw_norm_1, hidden_dim, 1);
    profileBarrier(cmd, profile, .gpu_routed_moe); // post expert/shared norms visible before shared add

    if (lt.ffn_gate_inp_shexp != null) {
        dispatchSigmoidScaleAccOnCmd(engine, cmd, &engine.moe_out_buf, &engine.down_buf, &engine.router_logits_buf, hidden_dim);
    } else {
        const acc_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(@as(f32, 1.0))) };
        const acc_bufs = [_]*const MetalBuffer{ &engine.moe_out_buf, &engine.down_buf };
        cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &acc_bufs, &acc_push, @sizeOf(ScaleAccPush), 0);
    }
    profileBarrier(cmd, profile, .gpu_routed_moe); // combined MoE contribution visible before final post-FFN norm

    if (engine.post_ffn_norm_present[layer_idx]) {
        dispatchRmsNormOnCmd(engine, cmd, &engine.moe_out_buf, &engine.moe_out_buf, &engine.post_ffn_norm_bufs[layer_idx], hidden_dim, 1);
        profileBarrier(cmd, profile, .gpu_routed_moe);
    }

    if (shouldValidateGemmaMoe(engine, layer_idx)) {
        commitAndWaitProfiled(cmd, profile);

        const routing: [*]const u32 = @ptrCast(@alignCast(engine.router_output_buf.cpu_ptr.?));
        const scales = try tensorF32Slice(engine, down_scales, cfg.n_experts);
        var expert_ids: [16]u32 = undefined;
        var adjusted_weights: [16]f32 = undefined;
        const n_used: usize = @intCast(cfg.n_experts_used);
        for (0..n_used) |slot| {
            const expert_id = routing[slot];
            expert_ids[slot] = expert_id;
            adjusted_weights[slot] = @as(f32, @bitCast(routing[n_used + slot])) * scales[@intCast(expert_id)];
        }

        const gate_value: f32 = if (lt.ffn_gate_inp_shexp != null) blk: {
            const gate_ptr: [*]const f32 = @ptrCast(@alignCast(engine.router_logits_buf.cpu_ptr.?));
            break :blk 1.0 / (1.0 + @exp(-gate_ptr[0]));
        } else 1.0;
        const expert_input_ptr: [*]const f32 = @ptrCast(@alignCast(engine.residual_buf.cpu_ptr.?));
        const shared_input_ptr: [*]const f32 = @ptrCast(@alignCast(engine.norm_buf.cpu_ptr.?));
        const actual_post_moe: [*]const f32 = @ptrCast(@alignCast(engine.moe_out_buf.cpu_ptr.?));
        try validateGemmaMoePostVector(
            engine,
            layer_idx,
            lt,
            gate_up_layout,
            expert_ids[0..n_used],
            adjusted_weights[0..n_used],
            gate_value,
            expert_input_ptr,
            shared_input_ptr,
            actual_post_moe[0..hidden_dim],
            hidden_dim,
            inter_dim,
            shexp_inter_dim,
        );
        cmd.* = try beginProfiledCommand(engine, profile);
    }

    const res_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(@as(f32, 1.0))) };
    const res_bufs = [_]*const MetalBuffer{ &engine.hidden_buf, &engine.moe_out_buf };
    cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &res_bufs, &res_push, @sizeOf(ScaleAccPush), 0);
    profileBarrier(cmd, profile, .gpu_routed_moe); // hidden_buf visible to next layer
}

fn recordGpuRoutedBatchedMoeOnCmd(
    engine: *InferenceEngine,
    cmd: *MetalCommand,
    profile: ?*RuntimeProfile,
    layer_idx: usize,
    lt: LayerTensors,
    hidden_dim: u32,
    inter_dim: u32,
    shexp_inter_dim: u32,
) !void {
    const cfg = engine.config;
    if (cfg.architecture == .gemma) {
        return recordGemmaGpuRoutedMoeOnCmd(engine, cmd, profile, layer_idx, lt, hidden_dim, inter_dim, shexp_inter_dim);
    }
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
    profileBarrier(cmd, profile, .gpu_routed_moe); // router_output_buf visible before expert DMMVs

    // Phase B: gate+up expert DMMVs + shared expert DMMVs — all independent, overlap in concurrent mode.
    try dispatchDmmvMoeOnCmd(engine, cmd, gate_exps, &engine.norm_buf, &engine.expert_gate_batch_buf, &engine.router_output_buf, inter_dim, hidden_dim, expert_gate_bytes, 0, 0);
    try dispatchDmmvMoeOnCmd(engine, cmd, up_exps, &engine.norm_buf, &engine.expert_up_batch_buf, &engine.router_output_buf, inter_dim, hidden_dim, expert_gate_bytes, 0, 0);
    if (has_shexp) {
        dispatchDmmvOnCmd(engine, cmd, gate_shexp.?, &engine.norm_buf, &engine.gate_buf, shexp_inter_dim, hidden_dim, 0);
        dispatchDmmvOnCmd(engine, cmd, up_shexp.?, &engine.norm_buf, &engine.up_buf, shexp_inter_dim, hidden_dim, 0);
        if (gate_inp_shexp) |tensor| {
            dispatchDmmvOnCmd(engine, cmd, tensor, &engine.norm_buf, &engine.router_logits_buf, 1, hidden_dim, 0);
        }
    }
    profileBarrier(cmd, profile, .gpu_routed_moe); // gate/up outputs visible before SwiGLU

    // Phase C: SwiGLU — batched experts + shared expert overlap.
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
    profileBarrier(cmd, profile, .gpu_routed_moe); // SwiGLU outputs visible before down DMMVs

    // Phase D: down expert DMMVs + shared down — overlap.
    try dispatchDmmvMoeOnCmd(engine, cmd, down_exps, &engine.expert_swiglu_batch_buf, &engine.expert_down_batch_buf, &engine.router_output_buf, hidden_dim, inter_dim, expert_down_bytes, inter_dim, 0);
    if (has_shexp) {
        dispatchDmmvOnCmd(engine, cmd, down_shexp.?, &engine.swiglu_buf, &engine.down_buf, hidden_dim, shexp_inter_dim, 0);
    }
    profileBarrier(cmd, profile, .gpu_routed_moe); // down outputs visible before accumulate

    // Phase E: weighted accumulate + shared expert into hidden_buf (fused — saves one barrier per layer).
    if (has_shexp) {
        const gate_buf = if (gate_inp_shexp != null) &engine.router_logits_buf else &engine.down_buf;
        dispatchMoeWeightedAccSharedOnCmd(engine, cmd, &engine.hidden_buf, &engine.expert_down_batch_buf, &engine.router_output_buf, &engine.down_buf, gate_buf, hidden_dim, cfg.n_experts_used, hidden_dim, gate_inp_shexp != null);
    } else {
        dispatchMoeWeightedAccOnCmd(engine, cmd, &engine.hidden_buf, &engine.expert_down_batch_buf, &engine.router_output_buf, hidden_dim, cfg.n_experts_used, hidden_dim);
    }
    profileBarrier(cmd, profile, .gpu_routed_moe); // hidden_buf visible to next layer's RMS norm
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
    const cmd = try metal_command.beginCommandWithMode(engine.device.ctx, engine.command_encoder_mode);
    if (profile) |p| p.command_buffers += 1;
    return cmd;
}

fn commitAndWaitProfiled(cmd: *MetalCommand, profile: ?*RuntimeProfile) void {
    if (profile) |p| {
        p.dispatch_calls += cmd.dispatch_count;
        p.barrier_calls += cmd.barrier_count;
    }
    const commit_start = profileStart(profile != null);
    cmd.commitAndWait();
    if (profile) |p| {
        p.commit_waits += 1;
        // This wall time is the full command-buffer completion wait, not just CPU submit overhead.
        p.gpu_completion_wait_ns += profileElapsedNs(commit_start);
    }
}

// ---------------------------------------------------------------------------
// Decode step — runs all layers plus optional final norm + LM head.
// ---------------------------------------------------------------------------

fn runDecodeStep(engine: *InferenceEngine, emit_logits: bool) !void {
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
    const full_attn_interval: u32 = fullAttentionInterval(cfg);

    // SSM constants (needed for GPU dispatch sizing)
    const d_inner: u32 = cfg.ssm_d_inner;
    const d_state: u32 = cfg.ssm_d_state;
    const n_group: u32 = cfg.ssm_n_group;
    const dt_rank: u32 = cfg.ssm_dt_rank;
    const conv_channels: u32 = if (d_inner > 0) d_inner + 2 * n_group * d_state else 0;

    const head_v_dim: u32 = if (d_inner > 0) d_inner / @max(dt_rank, 1) else 0;
    const d_conv: u32 = cfg.ssm_d_conv;
    const use_single_gpu_cmd = !engine.debug_validation_enabled and !engine.gemma_moe_validation_enabled and is_moe and blk: {
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
    if (engine.private_decode_buffers) {
        const cmd = shared_cmd orelse return error.PrivateDecodeFastPathRequiresSharedCommand;
        dispatchCopyF32OnCmd(engine, cmd, &engine.embed_staging, &engine.hidden_buf, hidden_dim);
        profileBarrier(cmd, profile, .embed);
    }

    for (0..cfg.n_layers) |layer_idx| {
        const layer: u32 = @intCast(layer_idx);
        const lt = engine.layer_tensors[layer_idx];
        const is_full_attn = ((layer + 1) % full_attn_interval == 0);
        const use_gpu_routed_moe = is_moe and canUseGpuRoutedBatchedMoe(engine, lt);
        const skip_pre_ffn_router = is_moe and !use_gpu_routed_moe and hasExplicitGemmaMoeTensors(cfg, lt);
        const layer_output_scale = engine.layer_output_scales[layer_idx];

        if (is_full_attn) {
            if (profile) |p| p.full_attn_layers += 1;
            const attn = try resolveLayerAttentionParams(cfg, lt, hidden_dim, engine.kv_cache_q8);
            var local_cmd_storage: MetalCommand = undefined;
            var using_local_cmd = false;
            var cmd = try acquireLayerCommand(engine, shared_cmd, &local_cmd_storage, &using_local_cmd, profile);
            const layer_record_start = profileStart(profile != null);
            dispatchRmsNormOnCmd(engine, cmd, &engine.hidden_buf, &engine.norm_buf, &engine.attn_norm_bufs[layer_idx], hidden_dim, 1);
            profileBarrier(cmd, profile, .full_attn); // norm_buf visible before attn prep reads it
            const apply_attn_gate = try dispatchFullAttnPrepOnCmd(engine, cmd, profile, layer_idx, lt, attn, hidden_dim);
            profileBarrier(cmd, profile, .full_attn); // KV cache + q_buf visible before flash attn
            dispatchFlashAttnOnCmd(
                engine,
                cmd,
                layer_idx,
                attn.head_dim,
                cfg.n_heads,
                attn.n_kv_heads,
                engine.position + 1,
                attn.sliding_window_size,
                attn.kv_cache_head_stride_bytes,
                attn.kv_cache_bytes_per_token,
            );
            if (apply_attn_gate) {
                profileBarrier(cmd, profile, .full_attn); // attn_out_buf visible before sigmoid_mul
                dispatchSigmoidMulOnCmd(engine, cmd, &engine.gate_buf, &engine.attn_out_buf, attn.q_dim);
            }
            profileBarrier(cmd, profile, .full_attn); // attn_out_buf visible before output DMMV
            const o_tensor = lt.attn_output orelse return error.MissingTensor;
            dispatchDmmvOnCmd(engine, cmd, o_tensor, &engine.attn_out_buf, &engine.down_buf, hidden_dim, attn.q_dim, 0);
            profileBarrier(cmd, profile, .full_attn);
            // Apply O projection bias if present (gpt-oss)
            if (lt.attn_output_bias) |b| {
                commitAndWaitProfiled(cmd, profile);
                addBiasFromTensor(engine, @ptrCast(@alignCast(engine.down_buf.cpu_ptr.?)), b, hidden_dim);
                local_cmd_storage = try beginProfiledCommand(engine, profile);
                cmd = &local_cmd_storage;
            }
            if (engine.post_attn_norm_present[layer_idx]) {
                dispatchRmsNormOnCmd(engine, cmd, &engine.down_buf, &engine.down_buf, &engine.post_attn_norm_bufs[layer_idx], hidden_dim, 1);
                profileBarrier(cmd, profile, .full_attn);
            }
            const should_debug_attn_compare = engine.debug_validation_enabled and using_local_cmd and
                shouldDebugAttentionValidation(cfg, engine.position, layer_idx);
            if (should_debug_attn_compare) {
                commitAndWaitProfiled(cmd, profile);
                const debug_start = profileStart(profile != null);
                try debugCompareAttentionLayer(engine, layer, layer_idx, lt, hidden_dim, q_dim, kv_dim);
                if (profile) |p| p.debug_validation_ns += profileElapsedNs(debug_start);
                local_cmd_storage = try beginProfiledCommand(engine, profile);
                cmd = &local_cmd_storage;
            }
            // Fused residual-add + RMS norm: hidden += down; norm_buf = normalize(hidden) * weights.
            // Eliminates one barrier vs separate scale_acc + barrier + rms_norm.
            dispatchResidualRmsNormOnCmd(engine, cmd, &engine.hidden_buf, &engine.down_buf, &engine.norm_buf, &engine.ffn_norm_bufs[layer_idx], hidden_dim, 1.0);
            if (!is_moe) {
                // dense FFN: norm_buf is ready (no barrier needed between fused dispatch and FFN)
            }
            if (is_moe and !skip_pre_ffn_router) {
                profileBarrier(cmd, profile, .router); // norm_buf visible before router DMMV
                const router_t = lt.ffn_gate_inp orelse return error.MissingTensor;
                const router_in_buf: *const MetalBuffer = blk: {
                    if (cfg.architecture == .gemma) {
                        const gate_scale = lt.ffn_gate_inp_scale orelse return error.MissingTensor;
                        dispatchRmsNormOnCmdWithTensorWeights(engine, cmd, &engine.hidden_buf, &engine.residual_buf, gate_scale, hidden_dim, 1);
                        profileBarrier(cmd, profile, .router);
                        break :blk &engine.residual_buf;
                    }
                    break :blk &engine.norm_buf;
                };
                dispatchDmmvOnCmd(engine, cmd, router_t, router_in_buf, &engine.router_logits_buf, cfg.n_experts, hidden_dim, 0);
                profileBarrier(cmd, profile, .router); // router_logits_buf visible before MoE
                if (use_gpu_routed_moe) {
                    if (profile) |p| p.layer_record_ns += profileElapsedNs(layer_record_start);
                    if (profile) |p| p.gpu_routed_moe_layers += 1;
                    const moe_record_start = profileStart(profile != null);
                    try recordGpuRoutedBatchedMoeOnCmd(engine, cmd, profile, layer_idx, lt, hidden_dim, inter_dim, shexp_inter_dim);
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
            const wqkv_t = lt.attn_qkv orelse return error.MissingTensor;
            const z_t = lt.attn_gate orelse return error.MissingTensor;
            const alpha_t = lt.ssm_alpha orelse return error.MissingTensor;
            const beta_t = lt.ssm_beta orelse return error.MissingTensor;
            const wqkv_buf: *const MetalBuffer = if (engine.private_ssm_qkv_bufs) |bufs|
                (if (bufs[layer_idx].handle != null) &bufs[layer_idx] else &wqkv_t.gpu_buffer)
            else
                &wqkv_t.gpu_buffer;
            const z_buf: *const MetalBuffer = if (engine.private_ssm_gate_bufs) |bufs|
                (if (bufs[layer_idx].handle != null) &bufs[layer_idx] else &z_t.gpu_buffer)
            else
                &z_t.gpu_buffer;
            const wqkv_offset: u32 = if (wqkv_buf == &wqkv_t.gpu_buffer) tensorPageOffset(engine.model, wqkv_t) else 0;
            const z_offset: u32 = if (z_buf == &z_t.gpu_buffer) tensorPageOffset(engine.model, z_t) else 0;

            // Fused RMSNorm + DMMV path: all consumers compute norm inline from L1-cached
            // hidden state, eliminating the separate RMSNorm dispatch and barrier.
            const use_fused_norm = false;

            if (use_fused_norm) {
                dispatchFusedNormDualQ8DmmvOnCmd(engine, cmd, wqkv_t, z_t, wqkv_buf, z_buf, wqkv_offset, z_offset, &engine.hidden_buf, &engine.attn_norm_bufs[layer_idx], &engine.attn_out_buf, &engine.gate_buf, conv_channels, d_inner, hidden_dim);
                dispatchFusedNormQ8DmmvOnCmd(engine, cmd, alpha_t, &alpha_t.gpu_buffer, tensorPageOffset(engine.model, alpha_t), &engine.hidden_buf, &engine.attn_norm_bufs[layer_idx], &engine.router_logits_buf, dt_rank, hidden_dim);
                dispatchFusedNormQ8DmmvOnCmd(engine, cmd, beta_t, &beta_t.gpu_buffer, tensorPageOffset(engine.model, beta_t), &engine.hidden_buf, &engine.attn_norm_bufs[layer_idx], &engine.down_buf, dt_rank, hidden_dim);
            } else {
                // Fallback: separate RMSNorm + barrier + DMMVs
                dispatchRmsNormOnCmd(engine, cmd, &engine.hidden_buf, &engine.norm_buf, &engine.attn_norm_bufs[layer_idx], hidden_dim, 1);
                profileBarrier(cmd, profile, .ssm);
                if (false and canUseDualQ8Dmmv(engine, wqkv_t, z_t, conv_channels, d_inner, hidden_dim)) {
                    dispatchDualQ8DmmvOnCmd(engine, cmd, wqkv_t, z_t, wqkv_buf, z_buf, wqkv_offset, z_offset, &engine.norm_buf, &engine.attn_out_buf, &engine.gate_buf, conv_channels, d_inner, hidden_dim);
                } else {
                    dispatchDmmvOnCmdWithWeightBuf(engine, cmd, wqkv_t, wqkv_buf, wqkv_offset, &engine.norm_buf, &engine.attn_out_buf, conv_channels, hidden_dim, 0);
                    dispatchDmmvOnCmdWithWeightBuf(engine, cmd, z_t, z_buf, z_offset, &engine.norm_buf, &engine.gate_buf, d_inner, hidden_dim, 0);
                }
                dispatchDmmvOnCmd(engine, cmd, alpha_t, &engine.norm_buf, &engine.router_logits_buf, dt_rank, hidden_dim, 0);
                dispatchDmmvOnCmd(engine, cmd, beta_t, &engine.norm_buf, &engine.down_buf, dt_rank, hidden_dim, 0);
            }
            profileBarrier(cmd, profile, .ssm);

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
            profileBarrier(cmd, profile, .ssm);

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
                // SPIRV-Cross Metal shader loops over all head_v_dim rows internally
                // (stride-64), unlike the GLSL original which uses gl_WorkGroupID.y
                // for row tiling. grid.y must be 1 to avoid duplicate workgroups
                // racing on the same SSM state memory. (All unit tests already use y=1.)
                cmd.dispatchV2(&engine.ssm_delta_net_pipe, .{ dt_rank, 1, 1 }, .{ 64, 1, 1 }, &dn_bufs, &push, @sizeOf(SsmDeltaNetPush), 0);
            }
            profileBarrier(cmd, profile, .ssm);
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
            profileBarrier(cmd, profile, .ssm);

            // SSM out DMMV: swiglu_buf → down_buf
            const ssm_out_t = lt.ssm_out orelse return error.MissingTensor;
            const ssm_out_buf: *const MetalBuffer = if (engine.private_ssm_out_bufs) |bufs|
                (if (bufs[layer_idx].handle != null) &bufs[layer_idx] else &ssm_out_t.gpu_buffer)
            else
                &ssm_out_t.gpu_buffer;
            const ssm_out_offset: u32 = if (ssm_out_buf == &ssm_out_t.gpu_buffer) tensorPageOffset(engine.model, ssm_out_t) else 0;
            dispatchDmmvOnCmdWithWeightBuf(engine, cmd, ssm_out_t, ssm_out_buf, ssm_out_offset, &engine.swiglu_buf, &engine.down_buf, hidden_dim, d_inner, 0);
            profileBarrier(cmd, profile, .ssm);
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

            // Fused residual-add + RMS norm: hidden += down; norm_buf = normalize(hidden) * weights.
            // Eliminates one barrier vs separate scale_acc + barrier + rms_norm.
            dispatchResidualRmsNormOnCmd(engine, cmd, &engine.hidden_buf, &engine.down_buf, &engine.norm_buf, &engine.ffn_norm_bufs[layer_idx], hidden_dim, 1.0);
            if (is_moe and !skip_pre_ffn_router) {
                profileBarrier(cmd, profile, .router);
                const router_t = lt.ffn_gate_inp orelse return error.MissingTensor;
                const router_in_buf: *const MetalBuffer = blk: {
                    if (cfg.architecture == .gemma) {
                        const gate_scale = lt.ffn_gate_inp_scale orelse return error.MissingTensor;
                        dispatchRmsNormOnCmdWithTensorWeights(engine, cmd, &engine.hidden_buf, &engine.residual_buf, gate_scale, hidden_dim, 1);
                        profileBarrier(cmd, profile, .router);
                        break :blk &engine.residual_buf;
                    }
                    break :blk &engine.norm_buf;
                };
                dispatchDmmvOnCmd(engine, cmd, router_t, router_in_buf, &engine.router_logits_buf, cfg.n_experts, hidden_dim, 0);
                profileBarrier(cmd, profile, .router); // router_logits_buf visible before MoE
                if (use_gpu_routed_moe) {
                    if (profile) |p| p.layer_record_ns += profileElapsedNs(layer_record_start);
                    if (profile) |p| p.gpu_routed_moe_layers += 1;
                    const moe_record_start = profileStart(profile != null);
                    try recordGpuRoutedBatchedMoeOnCmd(engine, cmd, profile, layer_idx, lt, hidden_dim, inter_dim, shexp_inter_dim);
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
            if (hasExplicitGemmaMoeTensors(cfg, lt)) {
                try runGemmaExplicitMoeFallback(engine, profile, layer_idx, lt, hidden_dim, inter_dim, shexp_inter_dim);
            } else {
                // CPU topK softmax
                const router_start = profileStart(profile != null);
                const router_ptr: [*]f32 = @ptrCast(@alignCast(engine.router_logits_buf.cpu_ptr.?));
                // Apply router bias if present (gpt-oss)
                if (lt.ffn_gate_inp_bias) |b| {
                    addBiasFromTensor(engine, router_ptr, b, cfg.n_experts);
                }
                if (cfg.architecture == .gemma) {
                    const router_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden_dim)));
                    for (router_ptr[0..cfg.n_experts]) |*value| value.* *= router_scale;
                }
                var expert_ids: [16]u32 = undefined;
                var expert_weights: [16]f32 = undefined;
                if (cfg.architecture == .gpt_oss) {
                    topKSoftmaxWeight(@as([*]const f32, router_ptr)[0..cfg.n_experts], cfg.n_experts_used, expert_ids[0..cfg.n_experts_used], expert_weights[0..cfg.n_experts_used]);
                } else {
                    topKSoftmax(@as([*]const f32, router_ptr)[0..cfg.n_experts], cfg.n_experts_used, expert_ids[0..cfg.n_experts_used], expert_weights[0..cfg.n_experts_used]);
                }
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
                const gate_up_layout = try resolveMoeGateUpLayout(lt, inter_dim, hidden_dim);
                const gate_exps = gate_up_layout.gate_tensor;
                const up_exps = gate_up_layout.up_tensor;
                const down_exps = lt.ffn_down_exps orelse return error.MissingTensor;
                const gate_quant = gate_exps.info.type_;
                const down_quant = down_exps.info.type_;
                const expert_gate_bytes = gate_up_layout.gate_expert_stride;
                const expert_down_bytes = expertSliceBytes(down_quant, hidden_dim, inter_dim);
                const use_batched_q4k_moe = canUseBatchedQ4kMoe(engine, gate_quant, down_quant) and !usesGeglu(cfg) and lt.ffn_gate_up_exps == null;

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
                    const moe_accum_buf: *const MetalBuffer = if (engine.post_ffn_norm_present[layer_idx]) &engine.moe_out_buf else &engine.hidden_buf;
                    if (engine.post_ffn_norm_present[layer_idx]) {
                        dispatchZeroF32OnCmd(engine, &cmd, moe_accum_buf, hidden_dim);
                        profileBarrier(&cmd, profile, .fallback_moe);
                    }

                    if (use_batched_q4k_moe) {
                        // Phase 1: Batched gate+up expert projections (+ shared expert)
                        dispatchDmmvMoeQ4kOnCmd(engine, &cmd, gate_exps, &engine.norm_buf, &engine.expert_gate_batch_buf, &engine.expert_ids_buf, inter_dim, hidden_dim, expert_gate_bytes, 0, 0);
                        dispatchDmmvMoeQ4kOnCmd(engine, &cmd, up_exps, &engine.norm_buf, &engine.expert_up_batch_buf, &engine.expert_ids_buf, inter_dim, hidden_dim, expert_gate_bytes, 0, 0);
                        if (has_shexp) {
                            dispatchDmmvOnCmd(engine, &cmd, gate_shexp.?, &engine.norm_buf, &engine.gate_buf, shexp_inter_dim, hidden_dim, 0);
                            dispatchDmmvOnCmd(engine, &cmd, up_shexp.?, &engine.norm_buf, &engine.up_buf, shexp_inter_dim, hidden_dim, 0);
                        }
                        profileBarrier(&cmd, profile, .fallback_moe);

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
                        profileBarrier(&cmd, profile, .fallback_moe);

                        // Phase 3: Batched down expert projection (+ shared expert)
                        dispatchDmmvMoeQ4kOnCmd(engine, &cmd, down_exps, &engine.expert_swiglu_batch_buf, &engine.expert_down_batch_buf, &engine.expert_ids_buf, hidden_dim, inter_dim, expert_down_bytes, inter_dim, 0);
                        if (down_shexp != null) {
                            dispatchDmmvOnCmd(engine, &cmd, down_shexp.?, &engine.swiglu_buf, &engine.down_buf, hidden_dim, shexp_inter_dim, 0);
                        }
                        profileBarrier(&cmd, profile, .fallback_moe);

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
                            moe_accum_buf,
                            &engine.expert_down_batch_buf,
                            &engine.down_buf,
                        };
                        cmd.dispatchV2(&engine.moe_acc_batched_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &moe_bufs, &moe_push, @sizeOf(MoeAccBatchedPush), 3);
                        profileBarrier(&cmd, profile, .fallback_moe);
                    } else {
                        // Phase 1: All expert gate+up DMMVs in parallel (+ shared expert)
                        for (0..cfg.n_experts_used) |ei| {
                            const eid = expert_ids[ei];
                            dispatchDmmvOnCmd(engine, &cmd, gate_exps, &engine.norm_buf, &engine.expert_gate_bufs[ei], inter_dim, hidden_dim, gate_up_layout.gateOffset(eid));
                            dispatchDmmvOnCmd(engine, &cmd, up_exps, &engine.norm_buf, &engine.expert_up_bufs[ei], inter_dim, hidden_dim, gate_up_layout.upOffset(eid));
                        }
                        if (has_shexp) {
                            dispatchDmmvOnCmd(engine, &cmd, gate_shexp.?, &engine.norm_buf, &engine.gate_buf, shexp_inter_dim, hidden_dim, 0);
                            dispatchDmmvOnCmd(engine, &cmd, up_shexp.?, &engine.norm_buf, &engine.up_buf, shexp_inter_dim, hidden_dim, 0);
                        }
                        profileBarrier(&cmd, profile, .fallback_moe);

                        // Phase 2: Bias + SwiGLU activation
                        if (cfg.architecture == .gpt_oss) {
                            // gpt-oss: apply expert biases + OAI SwiGLU on CPU
                            cmd.commitAndWait();
                            for (0..cfg.n_experts_used) |ei| {
                                const eid = expert_ids[ei];
                                if (lt.ffn_gate_exps_bias) |b| addBiasFromTensorSlice(engine, @ptrCast(@alignCast(engine.expert_gate_bufs[ei].cpu_ptr.?)), b, eid, inter_dim);
                                if (lt.ffn_up_exps_bias) |b| addBiasFromTensorSlice(engine, @ptrCast(@alignCast(engine.expert_up_bufs[ei].cpu_ptr.?)), b, eid, inter_dim);
                                cpuSwiGLU_OAI(@ptrCast(@alignCast(engine.expert_gate_bufs[ei].cpu_ptr.?)), @ptrCast(@alignCast(engine.expert_up_bufs[ei].cpu_ptr.?)), @ptrCast(@alignCast(engine.expert_swiglu_bufs[ei].cpu_ptr.?)), inter_dim);
                            }
                            cmd = try beginProfiledCommand(engine, profile);
                        } else {
                            // Standard: apply biases on CPU if needed, then GPU SwiGLU
                            if (lt.ffn_gate_exps_bias != null or lt.ffn_up_exps_bias != null) {
                                cmd.commitAndWait();
                                for (0..cfg.n_experts_used) |ei| {
                                    const eid = expert_ids[ei];
                                    if (lt.ffn_gate_exps_bias) |b| addBiasFromTensorSlice(engine, @ptrCast(@alignCast(engine.expert_gate_bufs[ei].cpu_ptr.?)), b, eid, inter_dim);
                                    if (lt.ffn_up_exps_bias) |b| addBiasFromTensorSlice(engine, @ptrCast(@alignCast(engine.expert_up_bufs[ei].cpu_ptr.?)), b, eid, inter_dim);
                                }
                                cmd = try beginProfiledCommand(engine, profile);
                            }
                            for (0..cfg.n_experts_used) |ei| {
                                dispatchFfnActivationOnCmd(engine, &cmd, &engine.expert_gate_bufs[ei], &engine.expert_swiglu_bufs[ei], &engine.expert_up_bufs[ei], inter_dim);
                            }
                        }
                        if (has_shexp) {
                            dispatchFfnActivationOnCmd(engine, &cmd, &engine.gate_buf, &engine.swiglu_buf, &engine.up_buf, shexp_inter_dim);
                        }
                        profileBarrier(&cmd, profile, .fallback_moe);

                        // Phase 3: All down DMMVs in parallel
                        const down_supported = engine.dmmvPipelineForType(down_exps, hidden_dim, inter_dim) != null;
                        if (down_supported) {
                            for (0..cfg.n_experts_used) |ei| {
                                const eid = expert_ids[ei];
                                const down_offset = eid * expert_down_bytes;
                                dispatchDmmvOnCmd(engine, &cmd, down_exps, &engine.expert_swiglu_bufs[ei], &engine.expert_down_bufs[ei], hidden_dim, inter_dim, down_offset);
                            }
                            if (down_shexp != null) {
                                dispatchDmmvOnCmd(engine, &cmd, down_shexp.?, &engine.swiglu_buf, &engine.down_buf, hidden_dim, shexp_inter_dim, 0);
                            }
                            profileBarrier(&cmd, profile, .fallback_moe);
                        } else {
                            commitAndWaitProfiled(&cmd, profile);
                            const mmap = engine.model.mmap_data orelse return error.NoMmapData;
                            const tdo = engine.model.gguf_file.tensor_data_offset;
                            for (0..cfg.n_experts_used) |ei| {
                                const eid = expert_ids[ei];
                                const down_offset = eid * expert_down_bytes;
                                const in_ptr: [*]const f32 = @ptrCast(@alignCast(engine.expert_swiglu_bufs[ei].cpu_ptr.?));
                                const out_ptr: [*]f32 = @ptrCast(@alignCast(engine.expert_down_bufs[ei].cpu_ptr.?));
                                try cpuDmmvFallback(mmap, down_exps, tdo, in_ptr, out_ptr, hidden_dim, inter_dim, down_offset, engine.allocator);
                            }
                            cmd = try beginProfiledCommand(engine, profile);
                            if (down_shexp != null) {
                                dispatchDmmvOnCmd(engine, &cmd, down_shexp.?, &engine.swiglu_buf, &engine.down_buf, hidden_dim, shexp_inter_dim, 0);
                                profileBarrier(&cmd, profile, .fallback_moe);
                            }
                        }

                        // Apply per-expert down biases if present (gpt-oss)
                        if (lt.ffn_down_exps_bias) |b| {
                            cmd.commitAndWait();
                            for (0..cfg.n_experts_used) |ei| {
                                const eid = expert_ids[ei];
                                addBiasFromTensorSlice(engine, @ptrCast(@alignCast(engine.expert_down_bufs[ei].cpu_ptr.?)), b, eid, hidden_dim);
                            }
                            cmd = try beginProfiledCommand(engine, profile);
                        }

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
                                moe_accum_buf,
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
                            profileBarrier(&cmd, profile, .fallback_moe);
                        } else {
                            // Fallback for non-8-expert models: sequential accumulate
                            for (0..cfg.n_experts_used) |ei| {
                                const w = expert_weights[ei];
                                const acc_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(w)) };
                                const acc_bufs = [_]*const MetalBuffer{ moe_accum_buf, &engine.expert_down_bufs[ei] };
                                cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &acc_bufs, &acc_push, @sizeOf(ScaleAccPush), 0);
                                profileBarrier(&cmd, profile, .fallback_moe);
                            }
                            if (has_shexp) {
                                const shexp_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(shexp_gate_weight)) };
                                const shexp_bufs = [_]*const MetalBuffer{ moe_accum_buf, &engine.down_buf };
                                cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &shexp_bufs, &shexp_push, @sizeOf(ScaleAccPush), 0);
                                profileBarrier(&cmd, profile, .fallback_moe);
                            }
                        }
                    }

                    if (profile) |p| p.fallback_moe_record_ns += profileElapsedNs(moe_record_start);
                    commitAndWaitProfiled(&cmd, profile);
                    if (engine.post_ffn_norm_present[layer_idx]) {
                        var residual_cmd = try beginProfiledCommand(engine, profile);
                        dispatchRmsNormOnCmd(engine, &residual_cmd, &engine.moe_out_buf, &engine.moe_out_buf, &engine.post_ffn_norm_bufs[layer_idx], hidden_dim, 1);
                        profileBarrier(&residual_cmd, profile, .fallback_moe);
                        const res_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(@as(f32, 1.0))) };
                        const res_bufs = [_]*const MetalBuffer{ &engine.hidden_buf, &engine.moe_out_buf };
                        residual_cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &res_bufs, &res_push, @sizeOf(ScaleAccPush), 0);
                        commitAndWaitProfiled(&residual_cmd, profile);
                    }
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
                profileBarrier(&cmd, profile, .dense_ffn);

                dispatchFfnActivationOnCmd(engine, &cmd, &engine.gate_buf, &engine.swiglu_buf, &engine.up_buf, inter_dim);
                profileBarrier(&cmd, profile, .dense_ffn);

                dispatchDmmvOnCmd(engine, &cmd, down_t, &engine.swiglu_buf, &engine.down_buf, hidden_dim, inter_dim, 0);
                profileBarrier(&cmd, profile, .dense_ffn);
                if (engine.post_ffn_norm_present[layer_idx]) {
                    dispatchRmsNormOnCmd(engine, &cmd, &engine.down_buf, &engine.down_buf, &engine.post_ffn_norm_bufs[layer_idx], hidden_dim, 1);
                    profileBarrier(&cmd, profile, .dense_ffn);
                }

                if (profile) |p| p.dense_ffn_record_ns += profileElapsedNs(dense_record_start);
                commitAndWaitProfiled(&cmd, profile);

                const hidden_ptr: [*]f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));
                const down_ptr: [*]const f32 = @ptrCast(@alignCast(engine.down_buf.cpu_ptr.?));
                for (0..hidden_dim) |i| hidden_ptr[i] += down_ptr[i];
            }
        }

        const layer_output_scale_folded = skip_pre_ffn_router and !engine.debug_validation_enabled;
        if (layer_output_scale != 1.0 and !layer_output_scale_folded) {
            const scale_barrier_class: BarrierClass = if (is_moe)
                (if (use_gpu_routed_moe) .gpu_routed_moe else .fallback_moe)
            else
                .dense_ffn;
            if (shared_cmd) |cmd| {
                dispatchScaleInPlaceOnCmd(engine, cmd, &engine.hidden_buf, &engine.residual_buf, hidden_dim, layer_output_scale, profile, scale_barrier_class);
            } else {
                var scale_cmd = try beginProfiledCommand(engine, profile);
                dispatchScaleInPlaceOnCmd(engine, &scale_cmd, &engine.hidden_buf, &engine.residual_buf, hidden_dim, layer_output_scale, profile, scale_barrier_class);
                commitAndWaitProfiled(&scale_cmd, profile);
            }
        }

        if (engine.debug_validation_enabled and engine.position == 0) {
            logLayerDiagnostics(engine, lt, layer, is_full_attn, "post_ffn");
        }
    }

    if (!emit_logits) {
        if (shared_cmd) |cmd| {
            commitAndWaitProfiled(cmd, profile);
        }
        engine.position += 1;
        return;
    }

    // ===== Final: GPU norm → LM head (batched) =====
    const final_record_start = profileStart(profile != null);
    if (shared_cmd) |cmd| {
        dispatchRmsNormOnCmd(engine, cmd, &engine.hidden_buf, &engine.norm_buf, &engine.final_norm_gpu, hidden_dim, 1);
        profileBarrier(cmd, profile, .final);
        if (shouldCpuLmHeadFallback(engine)) {
            commitAndWaitProfiled(cmd, profile);
            const mmap = engine.model.mmap_data orelse return error.NoMmapData;
            const tdo = engine.model.gguf_file.tensor_data_offset;
            const in_ptr: [*]const f32 = @ptrCast(@alignCast(engine.norm_buf.cpu_ptr.?));
            const out_ptr: [*]f32 = @ptrCast(@alignCast(engine.logits_buf.cpu_ptr.?));
            try cpuDmmvFallback(mmap, engine.lm_head, tdo, in_ptr, out_ptr, cfg.vocab_size, hidden_dim, 0, engine.allocator);
            writeCpuArgmax(engine, out_ptr, cfg.vocab_size);
            if (profile) |p| p.final_record_ns += profileElapsedNs(final_record_start);
        } else {
            dispatchLmHeadOnCmd(engine, cmd, &engine.norm_buf, &engine.logits_buf, hidden_dim, cfg.vocab_size);
            profileBarrier(cmd, profile, .final);
            dispatchArgmaxOnCmd(engine, cmd, &engine.logits_buf, &engine.argmax_buf, cfg.vocab_size);
            if (profile) |p| p.final_record_ns += profileElapsedNs(final_record_start);
            commitAndWaitProfiled(cmd, profile);
        }
    } else {
        var cmd = try beginProfiledCommand(engine, profile);
        dispatchRmsNormOnCmd(engine, &cmd, &engine.hidden_buf, &engine.norm_buf, &engine.final_norm_gpu, hidden_dim, 1);
        profileBarrier(&cmd, profile, .final);
        if (shouldCpuLmHeadFallback(engine)) {
            commitAndWaitProfiled(&cmd, profile);
            const mmap = engine.model.mmap_data orelse return error.NoMmapData;
            const tdo = engine.model.gguf_file.tensor_data_offset;
            const in_ptr: [*]const f32 = @ptrCast(@alignCast(engine.norm_buf.cpu_ptr.?));
            const out_ptr: [*]f32 = @ptrCast(@alignCast(engine.logits_buf.cpu_ptr.?));
            try cpuDmmvFallback(mmap, engine.lm_head, tdo, in_ptr, out_ptr, cfg.vocab_size, hidden_dim, 0, engine.allocator);
            writeCpuArgmax(engine, out_ptr, cfg.vocab_size);
            if (profile) |p| p.final_record_ns += profileElapsedNs(final_record_start);
        } else {
            dispatchLmHeadOnCmd(engine, &cmd, &engine.norm_buf, &engine.logits_buf, hidden_dim, cfg.vocab_size);
            profileBarrier(&cmd, profile, .final);
            dispatchArgmaxOnCmd(engine, &cmd, &engine.logits_buf, &engine.argmax_buf, cfg.vocab_size);
            if (profile) |p| p.final_record_ns += profileElapsedNs(final_record_start);
            commitAndWaitProfiled(&cmd, profile);
        }
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

    const gate_up_layout = try resolveMoeGateUpLayout(lt, inter_dim, hidden_dim);
    const gate_exps = gate_up_layout.gate_tensor;
    const up_exps = gate_up_layout.up_tensor;
    const down_exps = lt.ffn_down_exps orelse return error.MissingTensor;
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
        const gate_offset = gate_up_layout.gateOffset(eid);
        const up_offset = gate_up_layout.upOffset(eid);
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
        if (usesGeglu(engine.config)) {
            cpuGeGLU(gate_buf.ptr, up_buf.ptr, swiglu_buf.ptr, inter_dim);
        } else {
            cpuSwiGLU(gate_buf.ptr, up_buf.ptr, swiglu_buf.ptr, inter_dim);
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
        if (usesGeglu(engine.config)) {
            cpuGeGLU(gate_sh.ptr, up_sh.ptr, sw_sh.ptr, shexp_inter_dim);
        } else {
            cpuSwiGLU(gate_sh.ptr, up_sh.ptr, sw_sh.ptr, shexp_inter_dim);
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
    attn_scale: f32,
) void {
    const q_per_kv = @max(n_heads / @max(n_kv_heads, 1), 1);
    const token_stride = n_kv_heads * head_dim;
    const scale = if (attn_scale != 0) attn_scale else 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

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

fn debugCompareAttentionProjectionStage(
    engine: *InferenceEngine,
    layer: u32,
    layer_idx: usize,
    lt: LayerTensors,
    hidden_dim: u32,
) !void {
    _ = layer_idx;
    const allocator = engine.allocator;
    const mmap = engine.model.mmap_data orelse return error.NoMmapData;
    const tensor_data_off = engine.model.gguf_file.tensor_data_offset;
    const cfg = engine.config;
    const attn = try resolveLayerAttentionParams(cfg, lt, hidden_dim, engine.kv_cache_q8);
    const q_dim = attn.q_dim;
    const kv_dim = attn.kv_dim;
    const head_dim: usize = @intCast(attn.head_dim);
    const n_heads: usize = @intCast(cfg.n_heads);

    const q_tensor = lt.attn_q orelse return error.MissingTensor;
    const k_tensor = lt.attn_k orelse return error.MissingTensor;
    const v_tensor = if (attn.use_k_as_v) k_tensor else lt.attn_v orelse return error.MissingTensor;
    const q_rows: u32 = @intCast(q_tensor.info.numElements() / hidden_dim);
    const gate_mode = classifyFullAttnGate(q_rows, q_dim, lt.attn_gate != null);

    const norm_in: [*]const f32 = @ptrCast(@alignCast(engine.norm_buf.cpu_ptr.?));
    const q_actual: [*]const f32 = @ptrCast(@alignCast(engine.q_buf.cpu_ptr.?));
    const k_actual: [*]const f32 = @ptrCast(@alignCast(engine.k_buf.cpu_ptr.?));
    const v_actual: [*]const f32 = @ptrCast(@alignCast(engine.v_buf.cpu_ptr.?));
    const gate_actual: [*]const f32 = @ptrCast(@alignCast(engine.gate_buf.cpu_ptr.?));

    const q_ref = try allocator.alloc(f32, q_dim);
    defer allocator.free(q_ref);
    const k_ref = try allocator.alloc(f32, kv_dim);
    defer allocator.free(k_ref);
    const v_ref = try allocator.alloc(f32, kv_dim);
    defer allocator.free(v_ref);
    const gate_ref = try allocator.alloc(f32, q_dim);
    defer allocator.free(gate_ref);

    if (gate_mode.packed_q_gate) {
        const q_full_dim: u32 = q_dim * 2;
        const q_full_ref = try allocator.alloc(f32, q_full_dim);
        defer allocator.free(q_full_ref);

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
        refDeinterleaveQGate(q_full_ref, q_ref, gate_ref, head_dim, n_heads);
    } else {
        try cpuDmmvFallback(
            mmap,
            q_tensor,
            tensor_data_off,
            norm_in,
            q_ref.ptr,
            q_dim,
            hidden_dim,
            0,
            allocator,
        );
        if (lt.attn_gate) |gate_tensor| {
            try cpuDmmvFallback(
                mmap,
                gate_tensor,
                tensor_data_off,
                norm_in,
                gate_ref.ptr,
                q_dim,
                hidden_dim,
                0,
                allocator,
            );
        } else {
            @memset(gate_ref, 0);
        }
    }
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

    logDebugSliceDiff(layer, "attn_q_raw", q_ref[0..q_dim], q_actual[0..q_dim]);
    logDebugSliceDiff(layer, "attn_k_raw", k_ref[0..kv_dim], k_actual[0..kv_dim]);
    logDebugSliceDiff(layer, "attn_v_raw", v_ref[0..kv_dim], v_actual[0..kv_dim]);
    if (gate_mode.apply_attn_gate) {
        logDebugSliceDiff(layer, "attn_gate_raw", gate_ref[0..q_dim], gate_actual[0..q_dim]);
    }

    if (cfg.architecture == .gemma and layer == 17) {
        try dispatchDmmvAndWait(engine, q_tensor, &engine.norm_buf, &engine.attn_out_buf, q_dim, hidden_dim, 0);
        const q_single_actual: [*]const f32 = @ptrCast(@alignCast(engine.attn_out_buf.cpu_ptr.?));
        logDebugSliceDiff(layer, "attn_q_raw_single", q_ref[0..q_dim], q_single_actual[0..q_dim]);
    }
}

fn debugCompareAttentionLayer(
    engine: *InferenceEngine,
    layer: u32,
    layer_idx: usize,
    lt: LayerTensors,
    hidden_dim: u32,
    q_dim_unused: u32,
    kv_dim_unused: u32,
) !void {
    _ = q_dim_unused;
    _ = kv_dim_unused;
    const allocator = engine.allocator;
    const mmap = engine.model.mmap_data orelse return error.NoMmapData;
    const tensor_data_off = engine.model.gguf_file.tensor_data_offset;
    const cfg = engine.config;
    const attn = try resolveLayerAttentionParams(cfg, lt, hidden_dim, engine.kv_cache_q8);
    const q_dim = attn.q_dim;
    const kv_dim = attn.kv_dim;
    const head_dim: usize = @intCast(attn.head_dim);
    const n_heads: usize = @intCast(cfg.n_heads);
    const n_kv_heads: usize = @intCast(attn.n_kv_heads);
    const rope_dim: u32 = attn.rope_dim;
    const seq_len: usize = @intCast(engine.position + 1);
    const unit_weights: [*]const f32 = @ptrCast(@alignCast(engine.unit_rms_norm_weights.cpu_ptr.?));

    const q_tensor = lt.attn_q orelse return error.MissingTensor;
    const k_tensor = lt.attn_k orelse return error.MissingTensor;
    const v_tensor = if (attn.use_k_as_v) k_tensor else lt.attn_v orelse return error.MissingTensor;
    const o_tensor = lt.attn_output orelse return error.MissingTensor;
    const q_rows: u32 = @intCast(q_tensor.info.numElements() / hidden_dim);
    const gate_mode = classifyFullAttnGate(q_rows, q_dim, lt.attn_gate != null);

    const norm_in: [*]const f32 = @ptrCast(@alignCast(engine.norm_buf.cpu_ptr.?));
    const q_actual: [*]const f32 = @ptrCast(@alignCast(engine.q_buf.cpu_ptr.?));
    const k_actual: [*]const f32 = @ptrCast(@alignCast(engine.k_buf.cpu_ptr.?));
    const v_actual: [*]const f32 = @ptrCast(@alignCast(engine.v_buf.cpu_ptr.?));
    const gate_actual: [*]const f32 = @ptrCast(@alignCast(engine.gate_buf.cpu_ptr.?));
    const attn_actual: [*]const f32 = @ptrCast(@alignCast(engine.attn_out_buf.cpu_ptr.?));
    const oproj_actual: [*]const f32 = @ptrCast(@alignCast(engine.down_buf.cpu_ptr.?));
    const k_cache_actual: [*]const f32 = @ptrCast(@alignCast(engine.kv_k_cache[layer_idx].cpu_ptr.?));
    const v_cache_actual: [*]const f32 = @ptrCast(@alignCast(engine.kv_v_cache[layer_idx].cpu_ptr.?));

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

    if (gate_mode.packed_q_gate) {
        const q_full_dim: u32 = q_dim * 2;
        const q_full_ref = try allocator.alloc(f32, q_full_dim);
        defer allocator.free(q_full_ref);

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
        refDeinterleaveQGate(q_full_ref, q_ref, gate_ref, head_dim, n_heads);
    } else {
        try cpuDmmvFallback(
            mmap,
            q_tensor,
            tensor_data_off,
            norm_in,
            q_ref.ptr,
            q_dim,
            hidden_dim,
            0,
            allocator,
        );
        if (lt.attn_gate) |gate_tensor| {
            try cpuDmmvFallback(
                mmap,
                gate_tensor,
                tensor_data_off,
                norm_in,
                gate_ref.ptr,
                q_dim,
                hidden_dim,
                0,
                allocator,
            );
        } else {
            @memset(gate_ref, 0);
        }
    }
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

    // Apply Q/K/V biases to CPU reference (gpt-oss)
    if (lt.attn_q_bias) |b| addBiasFromTensor(engine, q_ref.ptr, b, q_dim);
    if (lt.attn_k_bias) |b| addBiasFromTensor(engine, k_ref.ptr, b, kv_dim);
    if (lt.attn_v_bias) |b| addBiasFromTensor(engine, v_ref.ptr, b, kv_dim);

    if (engine.attn_q_norm_present[layer_idx]) {
        const qn_w: [*]const f32 = @ptrCast(@alignCast(engine.attn_q_norm_bufs[layer_idx].cpu_ptr.?));
        cpuRmsNormMul(q_ref.ptr, qn_w[0..head_dim], q_ref.ptr, attn.head_dim, cfg.n_heads, 1e-6);
    }
    if (engine.attn_k_norm_present[layer_idx]) {
        const kn_w: [*]const f32 = @ptrCast(@alignCast(engine.attn_k_norm_bufs[layer_idx].cpu_ptr.?));
        cpuRmsNormMul(k_ref.ptr, kn_w[0..head_dim], k_ref.ptr, attn.head_dim, attn.n_kv_heads, 1e-6);
    }
    if (cfg.architecture == .gemma and cfg.rope_freq_base_swa > 0) {
        cpuRmsNormMul(v_ref.ptr, unit_weights[0..attn.head_dim], v_ref.ptr, attn.head_dim, attn.n_kv_heads, cfg.rms_norm_eps);
    }
    if (attn.use_rope_freq_factors) {
        const freq_buf = selectRopeFreqBuffer(engine, rope_dim, attn.rope_freq_base, true);
        const inv_freq: [*]const f32 = @ptrCast(@alignCast(freq_buf.cpu_ptr.?));
        cpuRopeWithFreqs(q_ref.ptr, attn.head_dim, rope_dim, cfg.n_heads, engine.position, inv_freq[0 .. rope_dim / 2]);
        cpuRopeWithFreqs(k_ref.ptr, attn.head_dim, rope_dim, attn.n_kv_heads, engine.position, inv_freq[0 .. rope_dim / 2]);
    } else {
        cpuRope(q_ref.ptr, attn.head_dim, rope_dim, cfg.n_heads, engine.position, attn.rope_freq_base);
        cpuRope(k_ref.ptr, attn.head_dim, rope_dim, attn.n_kv_heads, engine.position, attn.rope_freq_base);
    }
    const kv_offset: usize = @intCast(engine.position * kv_dim);
    logDebugSliceDiff(layer, "attn_q", q_ref[0..q_dim], q_actual[0..q_dim]);
    logDebugSliceDiff(layer, "attn_k", k_ref[0..kv_dim], k_actual[0..kv_dim]);
    logDebugSliceDiff(layer, "attn_v", v_ref[0..kv_dim], v_actual[0..kv_dim]);
    if (gate_mode.apply_attn_gate) {
        logDebugSliceDiff(layer, "attn_gate", gate_ref[0..q_dim], gate_actual[0..q_dim]);
    }
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
        cfg.attn_scale,
    );
    for (0..@as(usize, q_dim)) |i| {
        if (gate_mode.apply_attn_gate) {
            const g = gate_ref[i];
            gated_ref[i] = flash_ref[i] * (1.0 / (1.0 + @exp(-g)));
        } else {
            gated_ref[i] = flash_ref[i];
        }
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
    if (lt.attn_output_bias) |b| {
        addBiasFromTensor(engine, oproj_ref.ptr, b, hidden_dim);
    }
    if (engine.post_attn_norm_present[layer_idx]) {
        const post_norm: [*]const f32 = @ptrCast(@alignCast(engine.post_attn_norm_bufs[layer_idx].cpu_ptr.?));
        cpuRmsNormMul(oproj_ref.ptr, post_norm[0..hidden_dim], oproj_ref.ptr, hidden_dim, 1, cfg.rms_norm_eps);
    }

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

/// Run prefill + autoregressive decode, returning generated tokens and timing metrics.
pub fn generateWithMetrics(
    engine: *InferenceEngine,
    prompt_tokens: []const u32,
    max_tokens: u32,
    eos_id: u32,
    allocator: std.mem.Allocator,
) !GenerateResult {
    var output: std.ArrayList(u32) = .{};
    errdefer output.deinit(allocator);

    const prompt_token_count: u32 = @intCast(@min(prompt_tokens.len, std.math.maxInt(u32)));
    if (prompt_token_count > engine.max_context_tokens) return error.ContextLengthExceeded;
    const request_budget = memory_plan.requestBudget(prompt_token_count, max_tokens, engine.max_context_tokens);
    const decode_budget = request_budget.completion_tokens;

    var state = DecodeState.init(allocator);
    defer state.deinit();
    state.requested_context_tokens = request_budget.target_context_tokens;

    // Prefill: process each prompt token through all layers
    const prefill_start = std.time.nanoTimestamp();
    if (prompt_tokens.len > 0) {
        try engine.prefillBatched(&state, prompt_tokens);
    } else {
        try engine.resetRequestState(state.requested_context_tokens);
    }
    const prefill_end = std.time.nanoTimestamp();
    const prefill_ns: u64 = @intCast(prefill_end - prefill_start);
    const prefill_tps = if (prompt_tokens.len > 0 and prefill_ns > 0)
        @as(f64, @floatFromInt(prompt_tokens.len)) * 1_000_000_000.0 / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;

    // Sample first output token from prefill logits
    var eos_at_first_position = false;
    if (prompt_tokens.len > 0 and decode_budget > 0) {
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
        try state.generated_tokens.append(allocator, first_token);
    }

    // Decode loop
    const decode_start = std.time.nanoTimestamp();
    var tokens_generated: u32 = @intCast(output.items.len);
    while (tokens_generated < decode_budget and output.items.len > 0) {
        const input_token = output.items[output.items.len - 1];
        try engine.decodeStep(&state, input_token);

        const next_token = engine.sampleGreedy();
        if (next_token == eos_id) break;

        try output.append(allocator, next_token);
        try state.generated_tokens.append(allocator, next_token);
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

/// Convenience wrapper around `generateWithMetrics` that logs timing and returns tokens.
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

test "classifyFullAttnGate handles packed, separate, and plain layouts" {
    const packed_mode = classifyFullAttnGate(16, 8, false);
    try std.testing.expect(packed_mode.packed_q_gate);
    try std.testing.expect(!packed_mode.separate_attn_gate);
    try std.testing.expect(packed_mode.apply_attn_gate);

    const separate_mode = classifyFullAttnGate(8, 8, true);
    try std.testing.expect(!separate_mode.packed_q_gate);
    try std.testing.expect(separate_mode.separate_attn_gate);
    try std.testing.expect(separate_mode.apply_attn_gate);

    const plain_mode = classifyFullAttnGate(8, 8, false);
    try std.testing.expect(!plain_mode.packed_q_gate);
    try std.testing.expect(!plain_mode.separate_attn_gate);
    try std.testing.expect(!plain_mode.apply_attn_gate);

    const packed_wins = classifyFullAttnGate(16, 8, true);
    try std.testing.expect(packed_wins.packed_q_gate);
    try std.testing.expect(!packed_wins.separate_attn_gate);
    try std.testing.expect(packed_wins.apply_attn_gate);
}

test "resolveLayerAttentionParams infers packed q gate dimensions from kv heads" {
    const null_buf = MetalBuffer{ .handle = null, .size = 0, .cpu_ptr = null, .is_mmap_wrapped = false };
    const q_tensor = metal_loader.LoadedTensor{
        .info = .{
            .name = "blk.3.attn_q.weight",
            .n_dims = 2,
            .dims = .{ 2048, 4096, 1, 1 },
            .type_ = .q4_k,
            .offset = 0,
        },
        .gpu_buffer = null_buf,
    };
    const k_tensor = metal_loader.LoadedTensor{
        .info = .{
            .name = "blk.3.attn_k.weight",
            .n_dims = 2,
            .dims = .{ 2048, 512, 1, 1 },
            .type_ = .q4_k,
            .offset = 0,
        },
        .gpu_buffer = null_buf,
    };
    const v_tensor = metal_loader.LoadedTensor{
        .info = .{
            .name = "blk.3.attn_v.weight",
            .n_dims = 2,
            .dims = .{ 2048, 512, 1, 1 },
            .type_ = .q4_k,
            .offset = 0,
        },
        .gpu_buffer = null_buf,
    };

    const cfg = ModelConfig{
        .architecture = .qwen35,
        .n_layers = 24,
        .n_heads = 8,
        .n_kv_heads = 2,
        .head_dim = 256,
        .hidden_dim = 2048,
        .intermediate_dim = 0,
        .vocab_size = 0,
        .context_length = 0,
        .rope_freq_base = 1_000_000.0,
        .n_experts = 0,
        .n_experts_used = 0,
        .rope_dim = 64,
        .ssm_d_conv = 0,
        .ssm_d_inner = 0,
        .ssm_d_state = 0,
        .ssm_dt_rank = 0,
        .ssm_n_group = 0,
        .full_attn_interval = 0,
        .shared_expert_intermediate_dim = 0,
    };
    const lt = LayerTensors{
        .attn_q = &q_tensor,
        .attn_k = &k_tensor,
        .attn_v = &v_tensor,
    };

    const attn = try resolveLayerAttentionParams(cfg, lt, cfg.hidden_dim, false);
    try std.testing.expectEqual(@as(u32, 2048), attn.q_dim);
    try std.testing.expectEqual(@as(u32, 512), attn.kv_dim);
    try std.testing.expectEqual(@as(u32, 256), attn.head_dim);
    try std.testing.expectEqual(@as(u32, 2), attn.n_kv_heads);
}

test "resolveLayerAttentionParams uses gemma SWA metadata from per-layer tensor shapes" {
    const null_buf = MetalBuffer{ .handle = null, .size = 0, .cpu_ptr = null, .is_mmap_wrapped = false };
    const q_tensor = metal_loader.LoadedTensor{
        .info = .{
            .name = "blk.0.attn_q.weight",
            .n_dims = 2,
            .dims = .{ 2816, 4096, 1, 1 },
            .type_ = .q8_0,
            .offset = 0,
        },
        .gpu_buffer = null_buf,
    };
    const k_tensor = metal_loader.LoadedTensor{
        .info = .{
            .name = "blk.0.attn_k.weight",
            .n_dims = 2,
            .dims = .{ 2816, 2048, 1, 1 },
            .type_ = .q8_0,
            .offset = 0,
        },
        .gpu_buffer = null_buf,
    };
    const v_tensor = metal_loader.LoadedTensor{
        .info = .{
            .name = "blk.0.attn_v.weight",
            .n_dims = 2,
            .dims = .{ 2816, 2048, 1, 1 },
            .type_ = .q8_0,
            .offset = 0,
        },
        .gpu_buffer = null_buf,
    };
    const q_norm = metal_loader.LoadedTensor{
        .info = .{
            .name = "blk.0.attn_q_norm.weight",
            .n_dims = 1,
            .dims = .{ 256, 1, 1, 1 },
            .type_ = .f32,
            .offset = 0,
        },
        .gpu_buffer = null_buf,
    };
    const k_norm = metal_loader.LoadedTensor{
        .info = .{
            .name = "blk.0.attn_k_norm.weight",
            .n_dims = 1,
            .dims = .{ 256, 1, 1, 1 },
            .type_ = .f32,
            .offset = 0,
        },
        .gpu_buffer = null_buf,
    };

    const cfg = ModelConfig{
        .architecture = .gemma,
        .n_layers = 30,
        .n_heads = 16,
        .n_kv_heads = 8,
        .head_dim = 512,
        .hidden_dim = 2816,
        .intermediate_dim = 0,
        .vocab_size = 0,
        .context_length = 0,
        .rope_freq_base = 1_000_000.0,
        .rope_freq_base_swa = 10_000.0,
        .n_experts = 128,
        .n_experts_used = 8,
        .rope_dim = 512,
        .ssm_d_conv = 0,
        .ssm_d_inner = 0,
        .ssm_d_state = 0,
        .ssm_dt_rank = 0,
        .ssm_n_group = 0,
        .full_attn_interval = 0,
        .shared_expert_intermediate_dim = 0,
        .sliding_window_size = 1024,
    };
    const lt = LayerTensors{
        .attn_q = &q_tensor,
        .attn_k = &k_tensor,
        .attn_v = &v_tensor,
        .attn_q_norm = &q_norm,
        .attn_k_norm = &k_norm,
    };

    const attn = try resolveLayerAttentionParams(cfg, lt, cfg.hidden_dim, false);
    try std.testing.expectEqual(@as(u32, 256), attn.head_dim);
    try std.testing.expectEqual(@as(u32, 4096), attn.q_dim);
    try std.testing.expectEqual(@as(u32, 2048), attn.kv_dim);
    try std.testing.expectEqual(@as(u32, 8), attn.n_kv_heads);
    try std.testing.expectEqual(@as(u32, 256), attn.rope_dim);
    try std.testing.expectEqual(@as(f32, 10_000.0), attn.rope_freq_base);
    try std.testing.expectEqual(@as(u32, 1024), attn.sliding_window_size);
    try std.testing.expect(!attn.use_rope_freq_factors);
    try std.testing.expect(!attn.use_k_as_v);
}

test "resolveLayerAttentionParams handles gemma global KV-sharing layers" {
    const null_buf = MetalBuffer{ .handle = null, .size = 0, .cpu_ptr = null, .is_mmap_wrapped = false };
    const q_tensor = metal_loader.LoadedTensor{
        .info = .{
            .name = "blk.5.attn_q.weight",
            .n_dims = 2,
            .dims = .{ 2816, 8192, 1, 1 },
            .type_ = .q8_0,
            .offset = 0,
        },
        .gpu_buffer = null_buf,
    };
    const k_tensor = metal_loader.LoadedTensor{
        .info = .{
            .name = "blk.5.attn_k.weight",
            .n_dims = 2,
            .dims = .{ 2816, 1024, 1, 1 },
            .type_ = .q8_0,
            .offset = 0,
        },
        .gpu_buffer = null_buf,
    };
    const q_norm = metal_loader.LoadedTensor{
        .info = .{
            .name = "blk.5.attn_q_norm.weight",
            .n_dims = 1,
            .dims = .{ 512, 1, 1, 1 },
            .type_ = .f32,
            .offset = 0,
        },
        .gpu_buffer = null_buf,
    };

    const cfg = ModelConfig{
        .architecture = .gemma,
        .n_layers = 30,
        .n_heads = 16,
        .n_kv_heads = 8,
        .head_dim = 512,
        .hidden_dim = 2816,
        .intermediate_dim = 0,
        .vocab_size = 0,
        .context_length = 0,
        .rope_freq_base = 1_000_000.0,
        .rope_freq_base_swa = 10_000.0,
        .n_experts = 128,
        .n_experts_used = 8,
        .rope_dim = 512,
        .ssm_d_conv = 0,
        .ssm_d_inner = 0,
        .ssm_d_state = 0,
        .ssm_dt_rank = 0,
        .ssm_n_group = 0,
        .full_attn_interval = 0,
        .shared_expert_intermediate_dim = 0,
        .sliding_window_size = 1024,
    };
    const lt = LayerTensors{
        .attn_q = &q_tensor,
        .attn_k = &k_tensor,
        .attn_q_norm = &q_norm,
    };

    const attn = try resolveLayerAttentionParams(cfg, lt, cfg.hidden_dim, false);
    try std.testing.expectEqual(@as(u32, 512), attn.head_dim);
    try std.testing.expectEqual(@as(u32, 8192), attn.q_dim);
    try std.testing.expectEqual(@as(u32, 1024), attn.kv_dim);
    try std.testing.expectEqual(@as(u32, 2), attn.n_kv_heads);
    try std.testing.expectEqual(@as(u32, 512), attn.rope_dim);
    try std.testing.expectEqual(@as(u32, 0), attn.sliding_window_size);
    try std.testing.expect(attn.use_rope_freq_factors);
    try std.testing.expect(attn.use_k_as_v);
    try std.testing.expect(attn.proportional_rope);
}

test "rope_native dispatch uses precomputed inverse frequencies" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "rope_native");
    defer metal_pipeline.freePipeline(&pipe);

    const stride: u32 = 6;
    const rope_dim: u32 = 4;
    const n_heads: u32 = 2;
    const position: u32 = 7;

    var data_buf = try metal_buffer.createBuffer(ctx, n_heads * stride * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&data_buf);
    var freq_buf = try metal_buffer.createBuffer(ctx, (rope_dim / 2) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&freq_buf);

    const input = [_]f32{
        1.0,  -2.0, 0.5, 3.0,  9.0, -9.0,
        -1.5, 4.0,  2.5, -0.5, 7.0, 8.0,
    };
    const inv_freq = [_]f32{ 0.125, 0.03125 };

    const data_ptr: [*]f32 = @ptrCast(@alignCast(data_buf.cpu_ptr.?));
    const freq_ptr: [*]f32 = @ptrCast(@alignCast(freq_buf.cpu_ptr.?));
    @memcpy(data_ptr[0 .. n_heads * stride], input[0 .. n_heads * stride]);
    @memcpy(freq_ptr[0 .. rope_dim / 2], inv_freq[0 .. rope_dim / 2]);

    var expected = input;
    cpuRopeWithFreqs(expected[0..].ptr, stride, rope_dim, n_heads, position, inv_freq[0..]);

    var engine: InferenceEngine = undefined;
    engine.rope_native_pipe = pipe;
    engine.rope_freq_buf = freq_buf;
    engine.config = std.mem.zeroes(ModelConfig);
    engine.config.head_dim = rope_dim;
    engine.config.rope_dim = rope_dim;
    engine.config.rope_freq_base = 10_000.0;
    engine.rope_freq_factors = null;
    engine.rope_variant_rope_dim = 0;
    engine.rope_variant_freq_base = 0;
    engine.rope_variant_uses_freq_factors = false;

    var cmd = try metal_command.beginCommand(ctx);
    dispatchRopeOnCmd(&engine, &cmd, &data_buf, &data_buf, stride, rope_dim, n_heads, position, 10_000.0, true);
    cmd.commitAndWait();

    for (0..expected.len) |i| {
        try std.testing.expectApproxEqAbs(expected[i], data_ptr[i], 1e-5);
    }
    try std.testing.expectEqual(input[4], data_ptr[4]);
    try std.testing.expectEqual(input[5], data_ptr[5]);
    try std.testing.expectEqual(input[10], data_ptr[10]);
    try std.testing.expectEqual(input[11], data_ptr[11]);
}

test "residual_rms_norm dispatch normalizes post-residual hidden state" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "residual_rms_norm");
    defer metal_pipeline.freePipeline(&pipe);

    const n: u32 = 8;
    var hidden_buf = try metal_buffer.createBuffer(ctx, n * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&hidden_buf);
    var residual_buf = try metal_buffer.createBuffer(ctx, n * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&residual_buf);
    var norm_buf = try metal_buffer.createBuffer(ctx, n * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&norm_buf);
    var weight_buf = try metal_buffer.createBuffer(ctx, n * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&weight_buf);

    const hidden_init = [_]f32{ 1.0, -2.0, 0.5, 3.0, -1.5, 0.25, 2.0, -0.75 };
    const residual_init = [_]f32{ 0.5, 1.0, -1.5, 0.25, 2.0, -0.5, -1.0, 0.75 };
    const weights = [_]f32{ 1.0, 0.5, 2.0, -1.0, 0.75, 1.25, -0.5, 1.5 };

    const hidden_ptr: [*]f32 = @ptrCast(@alignCast(hidden_buf.cpu_ptr.?));
    const residual_ptr: [*]f32 = @ptrCast(@alignCast(residual_buf.cpu_ptr.?));
    const norm_ptr: [*]f32 = @ptrCast(@alignCast(norm_buf.cpu_ptr.?));
    const weight_ptr: [*]f32 = @ptrCast(@alignCast(weight_buf.cpu_ptr.?));
    @memcpy(hidden_ptr[0..n], hidden_init[0..n]);
    @memcpy(residual_ptr[0..n], residual_init[0..n]);
    @memcpy(weight_ptr[0..n], weights[0..n]);
    @memset(norm_buf.cpu_ptr.?[0..norm_buf.size], 0);

    var expected_hidden = hidden_init;
    for (0..n) |i| expected_hidden[i] += residual_init[i];
    var expected_norm: [n]f32 = undefined;
    cpuRmsNormMul(expected_hidden[0..].ptr, weights[0..], expected_norm[0..].ptr, n, 1, 1e-6);

    var stale_norm: [n]f32 = undefined;
    cpuRmsNormMul(hidden_init[0..].ptr, weights[0..], stale_norm[0..].ptr, n, 1, 1e-6);

    var engine: InferenceEngine = undefined;
    engine.residual_rms_norm_pipe = pipe;

    var cmd = try metal_command.beginCommand(ctx);
    dispatchResidualRmsNormOnCmd(&engine, &cmd, &hidden_buf, &residual_buf, &norm_buf, &weight_buf, n, 1.0);
    cmd.commitAndWait();

    var differs_from_stale = false;
    for (0..n) |i| {
        try std.testing.expectApproxEqAbs(expected_hidden[i], hidden_ptr[i], 1e-5);
        try std.testing.expectApproxEqAbs(expected_norm[i], norm_ptr[i], 1e-5);
        if (@abs(norm_ptr[i] - stale_norm[i]) > 1e-3) differs_from_stale = true;
    }
    try std.testing.expect(differs_from_stale);
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

test "dequantRow q5_0 reads second-half high bits from qh bit 16" {
    var raw = [_]u8{0} ** 22;
    const d_bits = @as(u16, @bitCast(@as(f16, 1.0)));
    raw[0] = @truncate(d_bits);
    raw[1] = @truncate(d_bits >> 8);
    const qh: u32 = 1 << 16;
    raw[2] = @truncate(qh);
    raw[3] = @truncate(qh >> 8);
    raw[4] = @truncate(qh >> 16);
    raw[5] = @truncate(qh >> 24);

    var out = [_]f32{0} ** 32;
    dequantRow(&raw, 0, 32, .q5_0, &out);
    try std.testing.expectApproxEqAbs(@as(f32, -16.0), out[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[16], 0.001);
}

test "dequantRow q5_1 reads second-half high bits from qh bit 16" {
    var raw = [_]u8{0} ** 24;
    const d_bits = @as(u16, @bitCast(@as(f16, 1.0)));
    raw[0] = @truncate(d_bits);
    raw[1] = @truncate(d_bits >> 8);
    const m_bits = @as(u16, @bitCast(@as(f16, 0.0)));
    raw[2] = @truncate(m_bits);
    raw[3] = @truncate(m_bits >> 8);
    const qh: u32 = 1 << 16;
    raw[4] = @truncate(qh);
    raw[5] = @truncate(qh >> 8);
    raw[6] = @truncate(qh >> 16);
    raw[7] = @truncate(qh >> 24);

    var out = [_]f32{0} ** 32;
    dequantRow(&raw, 0, 32, .q5_1, &out);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 16.0), out[16], 0.001);
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

test "dotQ8_0Row matches dequantized row dot" {
    const M: usize = 3;
    const K: usize = 64;
    const block_size: usize = 32;
    const bpb: usize = 34;
    const bpr = K / block_size;

    var raw: [M * bpr * bpb]u8 = undefined;
    for (0..M) |row| {
        for (0..bpr) |block| {
            const bo = (row * bpr + block) * bpb;
            const scale = @as(f32, 0.125) * @as(f32, @floatFromInt(row + block + 1));
            const scale_bits: u16 = @bitCast(@as(f16, @floatCast(scale)));
            std.mem.writeInt(u16, raw[bo..][0..2], scale_bits, .little);
            for (0..block_size) |j| {
                const q: i8 = @intCast(@as(i32, @intCast((row * 17 + block * 11 + j * 3) % 31)) - 15);
                raw[bo + 2 + j] = @bitCast(q);
            }
        }
    }

    var input: [K]f32 = undefined;
    for (&input, 0..) |*v, i| {
        v.* = (@as(f32, @floatFromInt((i * 7) % 19)) - 9.0) * 0.03125;
    }

    var row_buf: [K]f32 = undefined;
    for (0..M) |row| {
        dequantRow(raw[0..], @intCast(row), @intCast(K), .q8_0, row_buf[0..]);
        var expected: f32 = 0;
        for (0..K) |i| expected += row_buf[i] * input[i];
        try std.testing.expectApproxEqAbs(expected, dotQ8_0Row(raw[0..], @intCast(row), @intCast(K), input[0..].ptr), 0.00001);
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

    // nr=2: each SG handles 2 rows, block_size=64 → 2 SGs per WG → 4 rows per WG
    const rows_per_wg: usize = 4;
    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ @intCast((M + rows_per_wg - 1) / rows_per_wg), 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
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

test "global q8 override skips gemma shared expert q8 tensors" {
    try std.testing.expect(!shouldUseGlobalQ8Override(.gemma, "blk.0.ffn_down.weight"));
    try std.testing.expect(!shouldUseGlobalQ8Override(.gemma, "blk.0.ffn_up.weight"));
    try std.testing.expect(!shouldUseGlobalQ8Override(.gemma, "blk.0.ffn_gate.weight"));
    try std.testing.expect(shouldUseGlobalQ8Override(.gemma, "blk.0.attn_q.weight"));
    try std.testing.expect(shouldUseGlobalQ8Override(.qwen35, "blk.0.ffn_down.weight"));
}

test "gemma shared down q8 tensors stay GPU eligible" {
    try std.testing.expect(!shouldCpuQ8Fallback(.gemma, "blk.0.ffn_down.weight"));
    try std.testing.expect(!shouldCpuQ8Fallback(.gemma, "blk.0.ffn_down_shexp.weight"));
    try std.testing.expect(!shouldCpuQ8Fallback(.gemma, "blk.0.ffn_up.weight"));
    try std.testing.expect(!shouldCpuQ8Fallback(.qwen35, "blk.0.ffn_down.weight"));
}

test "gemma q8 lm head uses CPU fallback" {
    try std.testing.expect(shouldCpuLmHeadFallbackForType(.gemma, .q8_0));
    try std.testing.expect(!shouldCpuLmHeadFallbackForType(.gemma, .q4_k));
    try std.testing.expect(!shouldCpuLmHeadFallbackForType(.qwen35, .q8_0));
}

test "defaultKvCacheQ8Enabled disables Gemma ISWA q8 KV cache" {
    const gemma_cfg = ModelConfig{
        .architecture = .gemma,
        .n_layers = 30,
        .n_heads = 16,
        .n_kv_heads = 8,
        .head_dim = 512,
        .hidden_dim = 2816,
        .intermediate_dim = 0,
        .vocab_size = 0,
        .context_length = 0,
        .rope_freq_base = 1_000_000.0,
        .rope_freq_base_swa = 10_000.0,
        .n_experts = 128,
        .n_experts_used = 8,
        .rope_dim = 512,
        .ssm_d_conv = 0,
        .ssm_d_inner = 0,
        .ssm_d_state = 0,
        .ssm_dt_rank = 0,
        .ssm_n_group = 0,
        .full_attn_interval = 0,
        .shared_expert_intermediate_dim = 0,
    };
    try std.testing.expect(!defaultKvCacheQ8Enabled(gemma_cfg, false));

    var qwen_cfg = gemma_cfg;
    qwen_cfg.architecture = .qwen35;
    qwen_cfg.rope_freq_base_swa = 0;
    try std.testing.expect(defaultKvCacheQ8Enabled(qwen_cfg, false));
}

test "dmmv_q8_0_dual shader matches CPU reference across both outputs" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_dual");
    defer metal_pipeline.freePipeline(&pipe);

    const M0: usize = 65;
    const M1: usize = 33;
    const K: usize = 2048;
    const blocks_per_row: usize = K / 32;
    const row_bytes: usize = blocks_per_row * 34;

    var weight0_buf = try metal_buffer.createBuffer(ctx, M0 * row_bytes);
    defer metal_buffer.freeBuffer(&weight0_buf);
    var weight1_buf = try metal_buffer.createBuffer(ctx, M1 * row_bytes);
    defer metal_buffer.freeBuffer(&weight1_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output0_buf = try metal_buffer.createBuffer(ctx, M0 * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output0_buf);
    var output1_buf = try metal_buffer.createBuffer(ctx, M1 * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output1_buf);

    @memset(weight0_buf.cpu_ptr.?[0..weight0_buf.size], 0);
    @memset(weight1_buf.cpu_ptr.?[0..weight1_buf.size], 0);
    @memset(input_buf.cpu_ptr.?[0..input_buf.size], 0);
    @memset(output0_buf.cpu_ptr.?[0..output0_buf.size], 0);
    @memset(output1_buf.cpu_ptr.?[0..output1_buf.size], 0);

    for (0..M0) |row| {
        for (0..blocks_per_row) |blk| {
            const base = row * row_bytes + blk * 34;
            const scale_mag = 0.03125 * @as(f32, @floatFromInt(1 + (row % 5) + (blk % 7)));
            const scale = @as(f16, @floatCast(scale_mag));
            const scale_bits = @as(u16, @bitCast(scale));
            weight0_buf.cpu_ptr.?[base] = @truncate(scale_bits);
            weight0_buf.cpu_ptr.?[base + 1] = @truncate(scale_bits >> 8);
            for (0..32) |e| {
                const raw_q: i32 = @intCast((row * 11 + blk * 7 + e * 5) % 63);
                const q: i8 = @intCast(raw_q - 31);
                weight0_buf.cpu_ptr.?[base + 2 + e] = @bitCast(q);
            }
        }
    }
    for (0..M1) |row| {
        for (0..blocks_per_row) |blk| {
            const base = row * row_bytes + blk * 34;
            const scale_mag = 0.046875 * @as(f32, @floatFromInt(1 + (row % 3) + (blk % 5)));
            const scale = @as(f16, @floatCast(scale_mag));
            const scale_bits = @as(u16, @bitCast(scale));
            weight1_buf.cpu_ptr.?[base] = @truncate(scale_bits);
            weight1_buf.cpu_ptr.?[base + 1] = @truncate(scale_bits >> 8);
            for (0..32) |e| {
                const raw_q: i32 = @intCast((row * 13 + blk * 3 + e * 7) % 61);
                const q: i8 = @intCast(raw_q - 30);
                weight1_buf.cpu_ptr.?[base + 2 + e] = @bitCast(q);
            }
        }
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| {
        const raw: i32 = @intCast((i * 13 + 7) % 29);
        input_ptr[i] = 0.125 * @as(f32, @floatFromInt(raw - 14));
    }

    const push = DualQ8DmmvPush{
        .M0 = @intCast(M0),
        .M1 = @intCast(M1),
        .K = @intCast(K),
        .a0_offset = 0,
        .a1_offset = 0,
        .x_offset = 0,
        .y0_offset = 0,
        .y1_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight0_buf, &weight1_buf, &input_buf, &output0_buf, &output1_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ @intCast((M0 + M1 + 15) / 16), 1, 1 }, .{ 512, 1, 1 }, &bufs, &push, @sizeOf(DualQ8DmmvPush), 0);
    cmd.commitAndWait();

    const allocator = std.testing.allocator;
    const ref_row = try allocator.alloc(f32, K);
    defer allocator.free(ref_row);

    const output0_ptr: [*]const f32 = @ptrCast(@alignCast(output0_buf.cpu_ptr.?));
    const output1_ptr: [*]const f32 = @ptrCast(@alignCast(output1_buf.cpu_ptr.?));

    for (0..M0) |row| {
        var expected: f32 = 0;
        dequantRow(weight0_buf.cpu_ptr.?[0..weight0_buf.size], @intCast(row), @intCast(K), .q8_0, ref_row);
        for (0..K) |i| expected += ref_row[i] * input_ptr[i];
        try std.testing.expectApproxEqAbs(expected, output0_ptr[row], 0.05);
    }
    for (0..M1) |row| {
        var expected: f32 = 0;
        dequantRow(weight1_buf.cpu_ptr.?[0..weight1_buf.size], @intCast(row), @intCast(K), .q8_0, ref_row);
        for (0..K) |i| expected += ref_row[i] * input_ptr[i];
        try std.testing.expectApproxEqAbs(expected, output1_ptr[row], 0.05);
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
    var dmmv_q5_1_moe_pipe = try loadShaderPipeline(ctx, "dmmv_q5_1_moe");
    defer metal_pipeline.freePipeline(&dmmv_q5_1_moe_pipe);
    var dmmv_q5k_moe_pipe = try loadShaderPipeline(ctx, "dmmv_q5k_moe");
    defer metal_pipeline.freePipeline(&dmmv_q5k_moe_pipe);
    var dmmv_q5k_moe_k2048_pipe = try loadShaderPipeline(ctx, "dmmv_q5k_moe_k2048");
    defer metal_pipeline.freePipeline(&dmmv_q5k_moe_k2048_pipe);
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
    var rms_norm_offset_pipe = try loadShaderPipeline(ctx, "rms_norm_mul_offset");
    defer metal_pipeline.freePipeline(&rms_norm_offset_pipe);

    var acc_pipe = try loadShaderPipeline(ctx, "moe_accumulate_batched");
    defer metal_pipeline.freePipeline(&acc_pipe);

    var lmhead_pipe = try loadShaderPipeline(ctx, "dmmv_q4k_lmhead");
    defer metal_pipeline.freePipeline(&lmhead_pipe);
    var lmhead_pipe_1024 = try loadShaderPipeline(ctx, "dmmv_q4k_lmhead_1024");
    defer metal_pipeline.freePipeline(&lmhead_pipe_1024);
    var topk_pipe = try loadShaderPipeline(ctx, "softmax_topk");
    defer metal_pipeline.freePipeline(&topk_pipe);
    var topk_scaled_pipe = try loadShaderPipeline(ctx, "softmax_topk_scaled");
    defer metal_pipeline.freePipeline(&topk_scaled_pipe);
    var topk_batched_pipe = try loadShaderPipeline(ctx, "softmax_topk_batched");
    defer metal_pipeline.freePipeline(&topk_batched_pipe);
    var route_pack_pipe = try loadShaderPipeline(ctx, "moe_route_pack");
    defer metal_pipeline.freePipeline(&route_pack_pipe);
    var sigmoid_scale_acc_pipe = try loadShaderPipeline(ctx, "sigmoid_scale_acc");
    defer metal_pipeline.freePipeline(&sigmoid_scale_acc_pipe);
    var moe_weighted_acc_pipe = try loadShaderPipeline(ctx, "moe_weighted_acc");
    defer metal_pipeline.freePipeline(&moe_weighted_acc_pipe);
    var moe_weighted_acc_scaled_pipe = try loadShaderPipeline(ctx, "moe_weighted_acc_scaled");
    defer metal_pipeline.freePipeline(&moe_weighted_acc_scaled_pipe);

    try std.testing.expect(deinterleave_pipe.handle != null);
    try std.testing.expect(flash_attn_pipe.handle != null);
    try std.testing.expect(kv_cache_write_pipe.handle != null);
    try std.testing.expect(rope_pipe.handle != null);
    try std.testing.expect(sigmoid_mul_pipe.handle != null);
    try std.testing.expect(dmmv_pipe.handle != null);
    try std.testing.expect(dmmv_q5_1_moe_pipe.handle != null);
    try std.testing.expect(dmmv_q5k_moe_pipe.handle != null);
    try std.testing.expect(dmmv_q5k_moe_k2048_pipe.handle != null);
    try std.testing.expect(dmmv_q6k_moe_pipe.handle != null);
    try std.testing.expect(dmmv_pipe_k2048.handle != null);
    try std.testing.expect(dmmv_moe_pipe_k2048.handle != null);
    try std.testing.expect(dmmv_moe_pipe_k2048_1024.handle != null);
    try std.testing.expect(swiglu_pipe.handle != null);
    try std.testing.expect(rms_norm_offset_pipe.handle != null);
    try std.testing.expect(acc_pipe.handle != null);
    try std.testing.expect(lmhead_pipe.handle != null);
    try std.testing.expect(lmhead_pipe_1024.handle != null);
    try std.testing.expect(topk_pipe.handle != null);
    try std.testing.expect(topk_scaled_pipe.handle != null);
    try std.testing.expect(topk_batched_pipe.handle != null);
    try std.testing.expect(route_pack_pipe.handle != null);
    try std.testing.expect(sigmoid_scale_acc_pipe.handle != null);
    try std.testing.expect(moe_weighted_acc_pipe.handle != null);
    try std.testing.expect(moe_weighted_acc_scaled_pipe.handle != null);
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

test "softmax_topk_batched shader routes each token independently" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "softmax_topk_batched");
    defer metal_pipeline.freePipeline(&pipe);

    const n_tokens: u32 = 2;
    const n_experts: u32 = 8;
    const k: u32 = 3;
    const output_stride: u32 = k * 2;

    var logits_buf = try metal_buffer.createBuffer(ctx, n_tokens * n_experts * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&logits_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, n_tokens * output_stride * @sizeOf(u32));
    defer metal_buffer.freeBuffer(&output_buf);

    const logits_ptr: [*]f32 = @ptrCast(@alignCast(logits_buf.cpu_ptr.?));
    @memcpy(logits_ptr[0 .. n_tokens * n_experts], &[_]f32{
        0.0, 1.0, 4.0, 2.0, -1.0, 3.0,  0.5, -2.0,
        7.0, 1.0, 0.0, 6.0, 5.0,  -1.0, 2.0, 3.0,
    });
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    const push = SoftmaxTopkBatchedPush{
        .n_experts = n_experts,
        .k = k,
        .logits_stride = n_experts,
        .output_stride = output_stride,
    };
    const bufs = [_]*const MetalBuffer{ &logits_buf, &output_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ n_tokens, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(SoftmaxTopkBatchedPush), 0);
    cmd.commitAndWait();

    const out_ptr: [*]const u32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    try std.testing.expectEqual(@as(u32, 2), out_ptr[0]);
    try std.testing.expectEqual(@as(u32, 5), out_ptr[1]);
    try std.testing.expectEqual(@as(u32, 3), out_ptr[2]);

    const row1 = output_stride;
    try std.testing.expectEqual(@as(u32, 0), out_ptr[row1 + 0]);
    try std.testing.expectEqual(@as(u32, 3), out_ptr[row1 + 1]);
    try std.testing.expectEqual(@as(u32, 4), out_ptr[row1 + 2]);

    const w00: f32 = @bitCast(out_ptr[k + 0]);
    const w01: f32 = @bitCast(out_ptr[k + 1]);
    const w02: f32 = @bitCast(out_ptr[k + 2]);
    const w10: f32 = @bitCast(out_ptr[row1 + k + 0]);
    const w11: f32 = @bitCast(out_ptr[row1 + k + 1]);
    const w12: f32 = @bitCast(out_ptr[row1 + k + 2]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), w00 + w01 + w02, 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), w10 + w11 + w12, 0.01);
    try std.testing.expect(w00 > w01 and w01 > w02);
    try std.testing.expect(w10 > w11 and w11 > w12);
}

test "moe_route_pack shader groups batched routing by expert" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "moe_route_pack");
    defer metal_pipeline.freePipeline(&pipe);

    const n_tokens: u32 = 2;
    const n_experts: u32 = 6;
    const k: u32 = 3;
    const routing_stride: u32 = k * 2;
    const ids_stride: u32 = n_tokens;

    var routing_buf = try metal_buffer.createBuffer(ctx, n_tokens * routing_stride * @sizeOf(u32));
    defer metal_buffer.freeBuffer(&routing_buf);
    var counts_buf = try metal_buffer.createBuffer(ctx, n_experts * @sizeOf(u32));
    defer metal_buffer.freeBuffer(&counts_buf);
    var ids_buf = try metal_buffer.createBuffer(ctx, n_experts * ids_stride * @sizeOf(u32));
    defer metal_buffer.freeBuffer(&ids_buf);

    const routing_ptr: [*]u32 = @ptrCast(@alignCast(routing_buf.cpu_ptr.?));
    @memcpy(routing_ptr[0 .. n_tokens * routing_stride], &[_]u32{
        2, 5, 3, 0, 0, 0,
        0, 3, 4, 0, 0, 0,
    });
    @memset(counts_buf.cpu_ptr.?[0..counts_buf.size], 0);
    @memset(ids_buf.cpu_ptr.?[0..ids_buf.size], 0xff);

    const push = MoeRoutePackPush{
        .n_tokens = n_tokens,
        .n_experts = n_experts,
        .k = k,
        .routing_stride = routing_stride,
        .ids_stride = ids_stride,
    };
    const bufs = [_]*const MetalBuffer{ &routing_buf, &counts_buf, &ids_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ 1, 1, 1 }, .{ 256, 1, 1 }, &bufs, &push, @sizeOf(MoeRoutePackPush), 0);
    cmd.commitAndWait();

    const counts_ptr: [*]const u32 = @ptrCast(@alignCast(counts_buf.cpu_ptr.?));
    const ids_ptr: [*]const u32 = @ptrCast(@alignCast(ids_buf.cpu_ptr.?));
    try std.testing.expectEqualSlices(u32, &.{ 1, 0, 1, 2, 1, 1 }, counts_ptr[0..n_experts]);
    try std.testing.expectEqual(@as(u32, 3), ids_ptr[0 * ids_stride + 0]);
    try std.testing.expectEqual(@as(u32, 0), ids_ptr[2 * ids_stride + 0]);
    try std.testing.expectEqual(@as(u32, 2), ids_ptr[3 * ids_stride + 0]);
    try std.testing.expectEqual(@as(u32, 4), ids_ptr[3 * ids_stride + 1]);
    try std.testing.expectEqual(@as(u32, 5), ids_ptr[4 * ids_stride + 0]);
    try std.testing.expectEqual(@as(u32, 1), ids_ptr[5 * ids_stride + 0]);
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
        .dst_offset_bytes = 0,
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
    var sinks_buf = try metal_buffer.createBuffer(ctx, n_heads * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&sinks_buf);
    {
        // Fill sinks with NaN (disabled)
        const sp: [*]f32 = @ptrCast(@alignCast(sinks_buf.cpu_ptr.?));
        sp[0] = std.math.nan(f32);
    }

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
        .sliding_window_size = 0,
        .page_size = 0,
        .attn_scale_bits = 0,
        .kv_head_stride_bytes = head_dim * @sizeOf(f32),
        .kv_token_stride_bytes = n_kv_heads * head_dim * @sizeOf(f32),
    };
    const bufs = [_]*const MetalBuffer{ &page_table_buf, &q_buf, &k_cache_buf, &v_cache_buf, &out_buf, &sinks_buf };

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
        0,
    );

    const out_ptr: [*]const f32 = @ptrCast(@alignCast(out_buf.cpu_ptr.?));
    for (0..head_dim) |i| {
        try std.testing.expectApproxEqAbs(expected[i], out_ptr[i], 0.001);
    }
}

test "flash_attn shader respects sliding window mask" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "flash_attn");
    defer metal_pipeline.freePipeline(&pipe);

    const head_dim: u32 = 4;
    const n_heads: u32 = 1;
    const n_kv_heads: u32 = 1;
    const seq_len: u32 = 3;

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
    var sinks_buf = try metal_buffer.createBuffer(ctx, n_heads * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&sinks_buf);

    const page_table_ptr: [*]u32 = @ptrCast(@alignCast(page_table_buf.cpu_ptr.?));
    page_table_ptr[0] = 0;
    page_table_ptr[1] = 1;
    page_table_ptr[2] = 2;

    const sinks_ptr: [*]f32 = @ptrCast(@alignCast(sinks_buf.cpu_ptr.?));
    sinks_ptr[0] = std.math.nan(f32);

    const q_ptr: [*]f32 = @ptrCast(@alignCast(q_buf.cpu_ptr.?));
    @memcpy(q_ptr[0 .. n_heads * head_dim], &[_]f32{ 0.0, 0.0, 1.0, 0.0 });

    const k_ptr: [*]f32 = @ptrCast(@alignCast(k_cache_buf.cpu_ptr.?));
    @memcpy(k_ptr[0 .. seq_len * n_kv_heads * head_dim], &[_]f32{
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
    });

    const v_ptr: [*]f32 = @ptrCast(@alignCast(v_cache_buf.cpu_ptr.?));
    @memcpy(v_ptr[0 .. seq_len * n_kv_heads * head_dim], &[_]f32{
        10.0, 20.0,  30.0,  40.0,
        50.0, 60.0,  70.0,  80.0,
        90.0, 100.0, 110.0, 120.0,
    });

    @memset(out_buf.cpu_ptr.?[0..out_buf.size], 0);

    const push = FlashAttnPush{
        .head_dim = head_dim,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .seq_len = seq_len,
        .sliding_window_size = 1,
        .page_size = 0,
        .attn_scale_bits = 0,
        .kv_head_stride_bytes = head_dim * @sizeOf(f32),
        .kv_token_stride_bytes = n_kv_heads * head_dim * @sizeOf(f32),
    };
    const bufs = [_]*const MetalBuffer{ &page_table_buf, &q_buf, &k_cache_buf, &v_cache_buf, &out_buf, &sinks_buf };

    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ n_heads, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(FlashAttnPush), 0);
    cmd.commitAndWait();

    const out_ptr: [*]const f32 = @ptrCast(@alignCast(out_buf.cpu_ptr.?));
    try std.testing.expectApproxEqAbs(@as(f32, 90.0), out_ptr[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 100.0), out_ptr[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 110.0), out_ptr[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 120.0), out_ptr[3], 0.001);
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

test "dmmv_q5_0 shader matches CPU reference with qh bits and nonzero offset" {
    // Regression test for the Q5_0 unaligned-uint32 qh read bug.
    // The Q5_0 block stores qh at byte offset 2 within a 22-byte block.
    // With nonzero a_offset, the qh address can be non-4-byte-aligned,
    // which caused *((device const uint*)&block[2]) to silently return
    // wrong values on Apple Silicon. Fix: read qh bytes individually.
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q5_0");
    defer metal_pipeline.freePipeline(&pipe);

    const M: u32 = 4; // multiple rows to test alignment at different offsets
    const K: u32 = 32; // 1 block per row
    const bpb: u32 = 22; // Q5_0 bytes per block

    // Allocate with padding to test nonzero a_offset (forces different alignments)
    const padding: u32 = 14; // odd padding to misalign qh reads
    const weight_size: usize = padding + @as(usize, M) * bpb;
    var weight_buf = try metal_buffer.createBuffer(ctx, weight_size);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0);

    // Create Q5_0 blocks at offset `padding` with known data.
    // Each block: [d(f16), qh(u32), qs(16 bytes)]
    const raw = weight_buf.cpu_ptr.?[padding..];
    for (0..M) |row| {
        const bo = row * bpb;
        // Scale: row-dependent fp16 value
        const d_val: f16 = @floatCast(0.1 * @as(f32, @floatFromInt(row + 1)));
        const d_bits: u16 = @bitCast(d_val);
        raw[bo + 0] = @truncate(d_bits);
        raw[bo + 1] = @truncate(d_bits >> 8);
        // qh: set bit j for element j (elements 0-15 get 5th bit = 1)
        // This means elements 0-15 have value (lo | 16), elements 16-31 have (hi | 0)
        const qh: u32 = 0x0000FFFF; // bits 0-15 set, 16-31 clear
        raw[bo + 2] = @truncate(qh);
        raw[bo + 3] = @truncate(qh >> 8);
        raw[bo + 4] = @truncate(qh >> 16);
        raw[bo + 5] = @truncate(qh >> 24);
        // qs: each byte has lo=3, hi=5 → element j: lo=3|(1<<4)=19, element 16+j: hi=5|(0<<4)=5
        for (0..16) |j| {
            raw[bo + 6 + j] = 0x53; // lo=3, hi=5
        }
    }

    // Input: increasing values
    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| input_ptr[i] = @as(f32, @floatFromInt(i + 1));

    // CPU reference
    var expected: [M]f32 = undefined;
    for (0..M) |row| {
        var ref_row: [32]f32 = undefined;
        dequantRow(raw[0 .. weight_size - padding], @intCast(row), K, .q5_0, &ref_row);
        var dot: f32 = 0;
        for (0..K) |i| dot += ref_row[i] * input_ptr[i];
        expected[row] = dot;
    }

    // GPU dispatch with nonzero a_offset
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);
    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = padding,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };
    var cmd = try metal_command.beginCommand(ctx);
    // Q5_0 uses 2 rows per workgroup (64 threads = 2 simdgroups of 32)
    cmd.dispatchV2(&pipe, .{ (M + 1) / 2, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..M) |row| {
        try std.testing.expectApproxEqAbs(expected[row], output_ptr[row], 0.01);
    }
}

test "dmmv_q5_0 shader matches CPU reference across many rows" {
    // Stress test: 64 rows × 3 blocks per row (K=96), with padding
    // that creates various alignment patterns for the qh uint32 read.
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q5_0");
    defer metal_pipeline.freePipeline(&pipe);

    const M: u32 = 64;
    const K: u32 = 96; // 3 Q5_0 blocks per row
    const bpb: u32 = 22;
    const blocks_per_row: u32 = K / 32;
    const row_bytes: u32 = blocks_per_row * bpb;

    const padding: u32 = 6; // misalign so block[2] has various alignments
    const weight_size: usize = padding + @as(usize, M) * row_bytes;
    var weight_buf = try metal_buffer.createBuffer(ctx, weight_size);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    // Fill with pseudo-random Q5_0 data
    const raw = weight_buf.cpu_ptr.?[padding..];
    for (0..M) |row| {
        for (0..blocks_per_row) |b| {
            const bo = row * row_bytes + b * bpb;
            const seed = @as(u32, @intCast(row * 7 + b * 13 + 1));
            // Scale
            const d_val: f16 = @floatCast(0.01 * @as(f32, @floatFromInt(seed % 20 + 1)));
            const d_bits: u16 = @bitCast(d_val);
            raw[bo + 0] = @truncate(d_bits);
            raw[bo + 1] = @truncate(d_bits >> 8);
            // qh: pseudo-random pattern
            const qh: u32 = seed *% 2654435761;
            raw[bo + 2] = @truncate(qh);
            raw[bo + 3] = @truncate(qh >> 8);
            raw[bo + 4] = @truncate(qh >> 16);
            raw[bo + 5] = @truncate(qh >> 24);
            // qs: pseudo-random nibbles
            for (0..16) |j| {
                raw[bo + 6 + j] = @truncate((seed +% @as(u32, @intCast(j)) *% 37) & 0xFF);
            }
        }
    }

    // Input: varying values
    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| input_ptr[i] = @sin(@as(f32, @floatFromInt(i)) * 0.1) * 5.0;

    // CPU reference for all rows
    var expected: [M]f32 = undefined;
    const weight_data = raw[0 .. M * row_bytes];
    for (0..M) |row| {
        var ref_row: [96]f32 = undefined;
        dequantRow(weight_data, @intCast(row), K, .q5_0, ref_row[0..K]);
        var dot: f32 = 0;
        for (0..K) |i| dot += ref_row[i] * input_ptr[i];
        expected[row] = dot;
    }

    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);
    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = padding,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };
    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ (M + 1) / 2, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..M) |row| {
        try std.testing.expectApproxEqAbs(expected[row], output_ptr[row], 0.1);
    }
}

test "dmmv_mxfp4 shader matches CPU reference" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_mxfp4");
    defer metal_pipeline.freePipeline(&pipe);

    const M: u32 = 4;
    const K: u32 = 32; // 1 MXFP4 block per row
    const bpb: u32 = 17; // MXFP4 bytes per block

    var weight_buf = try metal_buffer.createBuffer(ctx, M * bpb);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0);

    // Create MXFP4 blocks: [E8M0(1 byte), qs(16 bytes packed nibbles)]
    const raw = weight_buf.cpu_ptr.?;
    for (0..M) |row| {
        const bo = row * bpb;
        // E8M0 exponent: 127 → 2^0 = 1.0 (full scale)
        raw[bo] = 127;
        // qs: nibbles encoding E2M1 values
        // Nibble 0x2 = kvalues[2] = 1.0, nibble 0x4 = kvalues[4] = 2.0
        for (0..16) |j| {
            raw[bo + 1 + j] = 0x42; // lo=2 (1.0), hi=4 (2.0)
        }
    }

    // Input: all ones
    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| input_ptr[i] = 1.0;

    // CPU reference
    var expected: [M]f32 = undefined;
    for (0..M) |row| {
        var ref_row: [32]f32 = undefined;
        dequantRow(raw[0..weight_buf.size], @intCast(row), K, .mxfp4, &ref_row);
        var dot: f32 = 0;
        for (0..K) |i| dot += ref_row[i] * input_ptr[i];
        expected[row] = dot;
    }

    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);
    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };
    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ M, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..M) |row| {
        try std.testing.expectApproxEqAbs(expected[row], output_ptr[row], 0.01);
    }
}

test "repacked Q8_0 shader matches CPU reference" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_repacked");
    defer metal_pipeline.freePipeline(&pipe);

    const M: u32 = 4;
    const K: u32 = 1024; // 32 blocks per row = 1 group
    const blocks_per_row: u32 = K / 32;
    const row_bytes: usize = blocks_per_row * 34;

    // Original Q8_0 data
    var orig_buf = try metal_buffer.createBuffer(ctx, M * row_bytes);
    defer metal_buffer.freeBuffer(&orig_buf);

    // Fill with deterministic test data
    var rng = std.Random.DefaultPrng.init(42);
    const random = rng.random();
    for (0..M) |row| {
        for (0..blocks_per_row) |bi| {
            const off = row * row_bytes + bi * 34;
            // Scale: small random half value
            const scale_bits: u16 = @bitCast(@as(f16, @floatFromInt(@as(i8, @intCast(@as(i32, random.intRangeAtMost(i8, -10, 10)))))));
            orig_buf.cpu_ptr.?[off] = @truncate(scale_bits);
            orig_buf.cpu_ptr.?[off + 1] = @truncate(scale_bits >> 8);
            // qs: random int8 values
            for (0..32) |j| {
                orig_buf.cpu_ptr.?[off + 2 + j] = @bitCast(random.intRangeAtMost(i8, -127, 127));
            }
        }
    }

    // Repack
    var repacked_buf = try metal_buffer.createBuffer(ctx, M * row_bytes);
    defer metal_buffer.freeBuffer(&repacked_buf);
    repackQ8_0Blocks(orig_buf.cpu_ptr.?, repacked_buf.cpu_ptr.?, M, K);

    // Input vector
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| {
        input_ptr[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)))) - 3.0;
    }

    // CPU reference
    var expected: [4]f32 = .{ 0, 0, 0, 0 };
    for (0..M) |row| {
        var dot: f32 = 0;
        for (0..blocks_per_row) |bi| {
            const off = row * row_bytes + bi * 34;
            const scale = @as(f32, @as(f16, @bitCast(@as(u16, orig_buf.cpu_ptr.?[off]) | (@as(u16, orig_buf.cpu_ptr.?[off + 1]) << 8))));
            for (0..32) |j| {
                const q: f32 = @floatFromInt(@as(i8, @bitCast(orig_buf.cpu_ptr.?[off + 2 + j])));
                dot += scale * q * input_ptr[bi * 32 + j];
            }
        }
        expected[row] = dot;
    }

    // GPU dispatch with repacked kernel
    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &repacked_buf, &input_buf, &output_buf };
    const simd_w: u32 = if (pipe.thread_execution_width > 0) pipe.thread_execution_width else 32;
    const block_size: u32 = @min(512, pipe.max_threads_per_threadgroup);
    const rows_per_wg: u32 = block_size / simd_w * 2;
    const wgs = (M + rows_per_wg - 1) / rows_per_wg;
    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ wgs, 1, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..M) |row| {
        try std.testing.expectApproxEqAbs(expected[row], output_ptr[row], 0.5);
    }
}

test "dmmv_q8_0_k2048 shader matches CPU reference (nr=2)" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_k2048");
    defer metal_pipeline.freePipeline(&pipe);

    const M: u32 = 17; // odd M to test nr=2 boundary (last SG has only 1 valid row)
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

    for (0..M) |row| {
        for (0..blocks_per_row) |blk| {
            const base = row * row_bytes + blk * 34;
            const scale = @as(f16, @floatCast(0.0625 * @as(f32, @floatFromInt(1 + (row % 7) + (blk % 5)))));
            const scale_bits = @as(u16, @bitCast(scale));
            weight_buf.cpu_ptr.?[base] = @truncate(scale_bits);
            weight_buf.cpu_ptr.?[base + 1] = @truncate(scale_bits >> 8);
            for (0..32) |e| {
                const raw_q: i32 = @intCast((row * 3 + blk * 13 + e * 7) % 51);
                const q: i8 = @intCast(raw_q - 25);
                weight_buf.cpu_ptr.?[base + 2 + e] = @bitCast(q);
            }
        }
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| {
        const raw: i32 = @intCast((i * 7 + 3) % 19);
        input_ptr[i] = 0.25 * @as(f32, @floatFromInt(raw - 9));
    }

    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };

    // nr=2 dispatch: 2 rows per SG
    const simd_w: u32 = if (pipe.thread_execution_width > 0) pipe.thread_execution_width else 32;
    const block_size: u32 = @min(256, pipe.max_threads_per_threadgroup);
    const sgs_per_wg = block_size / simd_w;
    const rows_per_wg = sgs_per_wg * 2; // nr=2
    const wgs = (M + rows_per_wg - 1) / rows_per_wg;
    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ wgs, 1, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    var ref_row: [2048]f32 = undefined;
    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..M) |row| {
        dequantRow(weight_buf.cpu_ptr.?[0..weight_buf.size], @intCast(row), K, .q8_0, &ref_row);
        var expected: f32 = 0;
        for (0..K) |i| expected += ref_row[i] * input_ptr[i];
        try std.testing.expectApproxEqAbs(expected, output_ptr[row], 0.05);
    }
}

test "dmmv_q8_0 nr=2 odd M produces correct last row" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0");
    defer metal_pipeline.freePipeline(&pipe);

    // M=3: the last SG has base_row=2, row1 at row 3 is out-of-bounds for the weight
    // buffer. Verify that row 2's result is still correct and no out-of-bounds write
    // corrupts memory.
    const M: u32 = 3;
    const K: u32 = 2048;
    const blocks_per_row: usize = K / 32;
    const row_bytes: usize = blocks_per_row * 34;

    // Allocate extra row of padding to detect OOB writes
    var weight_buf = try metal_buffer.createBuffer(ctx, (M + 1) * row_bytes);
    defer metal_buffer.freeBuffer(&weight_buf);
    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    // Extra sentinel slot at output[M] to detect OOB write
    var output_buf = try metal_buffer.createBuffer(ctx, (M + 1) * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0);
    @memset(input_buf.cpu_ptr.?[0..input_buf.size], 0);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    for (0..M) |row| {
        for (0..blocks_per_row) |blk| {
            const base = row * row_bytes + blk * 34;
            const scale = @as(f16, @floatCast(0.25 * @as(f32, @floatFromInt(1 + row + (blk % 4)))));
            const scale_bits = @as(u16, @bitCast(scale));
            weight_buf.cpu_ptr.?[base] = @truncate(scale_bits);
            weight_buf.cpu_ptr.?[base + 1] = @truncate(scale_bits >> 8);
            for (0..32) |e| {
                const raw_q: i32 = @intCast((row * 5 + blk * 3 + e) % 31);
                const q: i8 = @intCast(raw_q - 15);
                weight_buf.cpu_ptr.?[base + 2 + e] = @bitCast(q);
            }
        }
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| {
        const raw: i32 = @intCast(i % 11);
        input_ptr[i] = @as(f32, @floatFromInt(raw - 5));
    }

    // Write sentinel to detect OOB write at output[M]
    const sentinel: f32 = -999.0;
    const output_ptr: [*]f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    output_ptr[M] = sentinel;

    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };

    // nr=2 dispatch
    const rows_per_wg: u32 = 4; // 2 SGs × 2 rows
    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ (M + rows_per_wg - 1) / rows_per_wg, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    var ref_row: [2048]f32 = undefined;
    for (0..M) |row| {
        dequantRow(weight_buf.cpu_ptr.?[0..weight_buf.size], @intCast(row), K, .q8_0, &ref_row);
        var expected: f32 = 0;
        for (0..K) |i| expected += ref_row[i] * input_ptr[i];
        try std.testing.expectApproxEqAbs(expected, output_ptr[row], 0.01);
    }

    // Sentinel must be untouched — shader must not write past M
    try std.testing.expectApproxEqAbs(sentinel, output_ptr[M], 0.0);
}

test "repacked Q8_0 shader matches CPU reference with multiple groups per row" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_repacked");
    defer metal_pipeline.freePipeline(&pipe);

    const M: u32 = 8;
    const K: u32 = 2048; // 64 blocks per row = 2 groups per row
    const blocks_per_row: u32 = K / 32;
    const row_bytes: usize = blocks_per_row * 34;

    var orig_buf = try metal_buffer.createBuffer(ctx, M * row_bytes);
    defer metal_buffer.freeBuffer(&orig_buf);

    var rng = std.Random.DefaultPrng.init(12345);
    const random = rng.random();
    for (0..M) |row| {
        for (0..blocks_per_row) |bi| {
            const off = row * row_bytes + bi * 34;
            const scale_bits: u16 = @bitCast(@as(f16, @floatFromInt(@as(i8, @intCast(@as(i32, random.intRangeAtMost(i8, -8, 8)))))));
            orig_buf.cpu_ptr.?[off] = @truncate(scale_bits);
            orig_buf.cpu_ptr.?[off + 1] = @truncate(scale_bits >> 8);
            for (0..32) |j| {
                orig_buf.cpu_ptr.?[off + 2 + j] = @bitCast(random.intRangeAtMost(i8, -120, 120));
            }
        }
    }

    var repacked_buf = try metal_buffer.createBuffer(ctx, M * row_bytes);
    defer metal_buffer.freeBuffer(&repacked_buf);
    repackQ8_0Blocks(orig_buf.cpu_ptr.?, repacked_buf.cpu_ptr.?, M, K);

    var input_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&input_buf);
    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| {
        input_ptr[i] = @as(f32, @floatFromInt(@as(i32, @intCast(i % 13)))) - 6.0;
    }

    // CPU reference from original (non-repacked) data
    var expected: [8]f32 = .{ 0, 0, 0, 0, 0, 0, 0, 0 };
    for (0..M) |row| {
        var dot: f32 = 0;
        for (0..blocks_per_row) |bi| {
            const off = row * row_bytes + bi * 34;
            const scale = @as(f32, @as(f16, @bitCast(@as(u16, orig_buf.cpu_ptr.?[off]) | (@as(u16, orig_buf.cpu_ptr.?[off + 1]) << 8))));
            for (0..32) |j| {
                const q: f32 = @floatFromInt(@as(i8, @bitCast(orig_buf.cpu_ptr.?[off + 2 + j])));
                dot += scale * q * input_ptr[bi * 32 + j];
            }
        }
        expected[row] = dot;
    }

    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &repacked_buf, &input_buf, &output_buf };
    const simd_w: u32 = if (pipe.thread_execution_width > 0) pipe.thread_execution_width else 32;
    const block_size: u32 = @min(512, pipe.max_threads_per_threadgroup);
    const rows_per_wg: u32 = block_size / simd_w * 2;
    const wgs = (M + rows_per_wg - 1) / rows_per_wg;
    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ wgs, 1, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..M) |row| {
        try std.testing.expectApproxEqAbs(expected[row], output_ptr[row], 0.5);
    }
}

test "canRepackQ8 boundary conditions" {
    try std.testing.expect(!canRepackQ8(0));
    try std.testing.expect(!canRepackQ8(512));
    try std.testing.expect(canRepackQ8(1024));
    try std.testing.expect(!canRepackQ8(1536));
    try std.testing.expect(canRepackQ8(2048));
    try std.testing.expect(canRepackQ8(4096));
}

test "repackQ8_0Blocks round-trip preserves data" {
    // Verify repacking preserves data by checking specific byte positions
    const M: u32 = 2;
    const K: u32 = 1024; // 32 blocks per row = 1 group
    const blocks_per_row: u32 = K / 32;
    const row_bytes: usize = blocks_per_row * 34;
    const total = M * row_bytes;

    const allocator = std.testing.allocator;
    const src = try allocator.alloc(u8, total);
    defer allocator.free(src);
    const dst = try allocator.alloc(u8, total);
    defer allocator.free(dst);

    // Fill with predictable pattern
    for (0..total) |i| {
        src[i] = @truncate(i *% 37 +% 11);
    }
    repackQ8_0Blocks(src.ptr, dst.ptr, M, K);

    // Verify structure: for each group, scales are contiguous then qs chunks
    for (0..M) |row| {
        for (0..1) |gi| {
            // Check scale of block 0 in this group
            const src_blk0_scale_lo = src[row * row_bytes + gi * 32 * 34];
            const src_blk0_scale_hi = src[row * row_bytes + gi * 32 * 34 + 1];
            const dst_group = (row * 1 + gi) * 1088;
            try std.testing.expectEqual(src_blk0_scale_lo, dst[dst_group]);
            try std.testing.expectEqual(src_blk0_scale_hi, dst[dst_group + 1]);

            // Check first quant byte of block 0, chunk 0
            const src_q0 = src[row * row_bytes + gi * 32 * 34 + 2];
            try std.testing.expectEqual(src_q0, dst[dst_group + 64]);
        }
    }
}

test "dmmv_q8_0_k2048_fused_norm shader matches CPU reference" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_k2048_fused_norm");
    defer metal_pipeline.freePipeline(&pipe);

    const M: u32 = 5;
    const K: u32 = 2048;
    const blocks_per_row: usize = K / 32;
    const row_bytes: usize = blocks_per_row * 34;

    var weight_buf = try metal_buffer.createBuffer(ctx, M * row_bytes);
    defer metal_buffer.freeBuffer(&weight_buf);
    var hidden_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&hidden_buf);
    var norm_weight_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&norm_weight_buf);
    var output_buf = try metal_buffer.createBuffer(ctx, M * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output_buf);

    @memset(weight_buf.cpu_ptr.?[0..weight_buf.size], 0);
    @memset(output_buf.cpu_ptr.?[0..output_buf.size], 0);

    // Fill weights
    for (0..M) |row| {
        for (0..blocks_per_row) |blk| {
            const base = row * row_bytes + blk * 34;
            const scale = @as(f16, @floatCast(0.0625 * @as(f32, @floatFromInt(1 + (row % 3) + (blk % 5)))));
            const scale_bits = @as(u16, @bitCast(scale));
            weight_buf.cpu_ptr.?[base] = @truncate(scale_bits);
            weight_buf.cpu_ptr.?[base + 1] = @truncate(scale_bits >> 8);
            for (0..32) |e| {
                const raw_q: i32 = @intCast((row * 7 + blk * 3 + e * 11) % 41);
                const q: i8 = @intCast(raw_q - 20);
                weight_buf.cpu_ptr.?[base + 2 + e] = @bitCast(q);
            }
        }
    }

    // Fill hidden state and norm weights
    const hidden_ptr: [*]f32 = @ptrCast(@alignCast(hidden_buf.cpu_ptr.?));
    const norm_ptr: [*]f32 = @ptrCast(@alignCast(norm_weight_buf.cpu_ptr.?));
    for (0..K) |i| {
        const raw: i32 = @intCast((i * 17 + 5) % 23);
        hidden_ptr[i] = 0.5 * @as(f32, @floatFromInt(raw - 11));
        norm_ptr[i] = 0.8 + 0.01 * @as(f32, @floatFromInt(@as(i32, @intCast(i % 13))));
    }

    // CPU reference: RMSNorm then DMMV
    var sq_sum: f64 = 0;
    for (0..K) |i| sq_sum += @as(f64, hidden_ptr[i]) * @as(f64, hidden_ptr[i]);
    const rms_inv: f32 = @floatCast(1.0 / @sqrt(sq_sum / @as(f64, K) + 1e-6));

    var normed_input: [2048]f32 = undefined;
    for (0..K) |i| normed_input[i] = norm_ptr[i] * (hidden_ptr[i] * rms_inv);

    const allocator = std.testing.allocator;
    const ref_row = try allocator.alloc(f32, K);
    defer allocator.free(ref_row);

    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    // fused_norm shader: buffer(0)=push, buffer(1)=W, buffer(2)=hidden, buffer(3)=Y, buffer(4)=norm_weight
    const bufs = [_]*const MetalBuffer{ &weight_buf, &hidden_buf, &output_buf, &norm_weight_buf };
    const block_size: u32 = 256;
    const rows_per_wg = block_size / 32;
    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ (M + rows_per_wg - 1) / rows_per_wg, 1, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..M) |row| {
        dequantRow(weight_buf.cpu_ptr.?[0..weight_buf.size], @intCast(row), K, .q8_0, ref_row);
        var expected: f32 = 0;
        for (0..K) |i| expected += ref_row[i] * normed_input[i];
        // fused_norm uses half-precision intermediate; allow wider tolerance
        try std.testing.expectApproxEqAbs(expected, output_ptr[row], 1.5);
    }
}

test "dmmv_q8_0_dual_fused_norm shader matches CPU reference" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_dual_fused_norm");
    defer metal_pipeline.freePipeline(&pipe);

    const M0: u32 = 4;
    const M1: u32 = 3;
    const K: u32 = 2048;
    const blocks_per_row: usize = K / 32;
    const row_bytes: usize = blocks_per_row * 34;

    var weight0_buf = try metal_buffer.createBuffer(ctx, M0 * row_bytes);
    defer metal_buffer.freeBuffer(&weight0_buf);
    var weight1_buf = try metal_buffer.createBuffer(ctx, M1 * row_bytes);
    defer metal_buffer.freeBuffer(&weight1_buf);
    var hidden_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&hidden_buf);
    var norm_weight_buf = try metal_buffer.createBuffer(ctx, K * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&norm_weight_buf);
    var output0_buf = try metal_buffer.createBuffer(ctx, M0 * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output0_buf);
    var output1_buf = try metal_buffer.createBuffer(ctx, M1 * @sizeOf(f32));
    defer metal_buffer.freeBuffer(&output1_buf);

    @memset(weight0_buf.cpu_ptr.?[0..weight0_buf.size], 0);
    @memset(weight1_buf.cpu_ptr.?[0..weight1_buf.size], 0);
    @memset(output0_buf.cpu_ptr.?[0..output0_buf.size], 0);
    @memset(output1_buf.cpu_ptr.?[0..output1_buf.size], 0);

    for (0..M0) |row| {
        for (0..blocks_per_row) |blk| {
            const base = row * row_bytes + blk * 34;
            const scale = @as(f16, @floatCast(0.0625 * @as(f32, @floatFromInt(1 + (row % 4) + (blk % 6)))));
            const scale_bits = @as(u16, @bitCast(scale));
            weight0_buf.cpu_ptr.?[base] = @truncate(scale_bits);
            weight0_buf.cpu_ptr.?[base + 1] = @truncate(scale_bits >> 8);
            for (0..32) |e| {
                const raw_q: i32 = @intCast((row * 5 + blk * 9 + e * 3) % 47);
                const q: i8 = @intCast(raw_q - 23);
                weight0_buf.cpu_ptr.?[base + 2 + e] = @bitCast(q);
            }
        }
    }
    for (0..M1) |row| {
        for (0..blocks_per_row) |blk| {
            const base = row * row_bytes + blk * 34;
            const scale = @as(f16, @floatCast(0.03125 * @as(f32, @floatFromInt(1 + (row % 3) + (blk % 8)))));
            const scale_bits = @as(u16, @bitCast(scale));
            weight1_buf.cpu_ptr.?[base] = @truncate(scale_bits);
            weight1_buf.cpu_ptr.?[base + 1] = @truncate(scale_bits >> 8);
            for (0..32) |e| {
                const raw_q: i32 = @intCast((row * 13 + blk * 7 + e * 11) % 53);
                const q: i8 = @intCast(raw_q - 26);
                weight1_buf.cpu_ptr.?[base + 2 + e] = @bitCast(q);
            }
        }
    }

    const hidden_ptr: [*]f32 = @ptrCast(@alignCast(hidden_buf.cpu_ptr.?));
    const norm_ptr: [*]f32 = @ptrCast(@alignCast(norm_weight_buf.cpu_ptr.?));
    for (0..K) |i| {
        const raw: i32 = @intCast((i * 11 + 3) % 19);
        hidden_ptr[i] = 0.25 * @as(f32, @floatFromInt(raw - 9));
        norm_ptr[i] = 0.9 + 0.02 * @as(f32, @floatFromInt(@as(i32, @intCast(i % 7))));
    }

    // CPU reference: RMSNorm then DMMV
    var sq_sum: f64 = 0;
    for (0..K) |i| sq_sum += @as(f64, hidden_ptr[i]) * @as(f64, hidden_ptr[i]);
    const rms_inv: f32 = @floatCast(1.0 / @sqrt(sq_sum / @as(f64, K) + 1e-6));

    var normed_input: [2048]f32 = undefined;
    for (0..K) |i| normed_input[i] = norm_ptr[i] * (hidden_ptr[i] * rms_inv);

    const allocator = std.testing.allocator;
    const ref_row = try allocator.alloc(f32, K);
    defer allocator.free(ref_row);

    const push = DualQ8DmmvPush{
        .M0 = M0,
        .M1 = M1,
        .K = K,
        .a0_offset = 0,
        .a1_offset = 0,
        .x_offset = 0,
        .y0_offset = 0,
        .y1_offset = 0,
    };
    // dual_fused_norm: buffer(0)=push, buffer(1)=W0, buffer(2)=W1, buffer(3)=hidden, buffer(4)=Y0, buffer(5)=Y1, buffer(6)=norm_weight
    const bufs = [_]*const MetalBuffer{ &weight0_buf, &weight1_buf, &hidden_buf, &output0_buf, &output1_buf, &norm_weight_buf };
    const total_rows = M0 + M1;
    const block_size: u32 = 256;
    const rows_per_wg = block_size / 32;
    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ (total_rows + rows_per_wg - 1) / rows_per_wg, 1, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(DualQ8DmmvPush), 0);
    cmd.commitAndWait();

    const output0_ptr: [*]const f32 = @ptrCast(@alignCast(output0_buf.cpu_ptr.?));
    const output1_ptr: [*]const f32 = @ptrCast(@alignCast(output1_buf.cpu_ptr.?));
    for (0..M0) |row| {
        dequantRow(weight0_buf.cpu_ptr.?[0..weight0_buf.size], @intCast(row), K, .q8_0, ref_row);
        var expected: f32 = 0;
        for (0..K) |i| expected += ref_row[i] * normed_input[i];
        try std.testing.expectApproxEqAbs(expected, output0_ptr[row], 1.5);
    }
    for (0..M1) |row| {
        dequantRow(weight1_buf.cpu_ptr.?[0..weight1_buf.size], @intCast(row), K, .q8_0, ref_row);
        var expected: f32 = 0;
        for (0..K) |i| expected += ref_row[i] * normed_input[i];
        try std.testing.expectApproxEqAbs(expected, output1_ptr[row], 1.5);
    }
}

test "dmmv_q8_0_native scalar shader matches CPU reference" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_native");
    defer metal_pipeline.freePipeline(&pipe);

    const M: u32 = 7;
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

    for (0..M) |row| {
        for (0..blocks_per_row) |blk| {
            const base = row * row_bytes + blk * 34;
            const scale = @as(f16, @floatCast(0.125 * @as(f32, @floatFromInt(1 + (row % 4) + (blk % 3)))));
            const scale_bits = @as(u16, @bitCast(scale));
            weight_buf.cpu_ptr.?[base] = @truncate(scale_bits);
            weight_buf.cpu_ptr.?[base + 1] = @truncate(scale_bits >> 8);
            for (0..32) |e| {
                const raw_q: i32 = @intCast((row * 3 + blk * 7 + e * 11) % 37);
                const q: i8 = @intCast(raw_q - 18);
                weight_buf.cpu_ptr.?[base + 2 + e] = @bitCast(q);
            }
        }
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| {
        const raw: i32 = @intCast((i * 7 + 2) % 13);
        input_ptr[i] = @as(f32, @floatFromInt(raw - 6));
    }

    const push = DmmvPush{
        .M = M,
        .K = K,
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };

    // native shader: one thread per row
    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ M, 1, 1 }, .{ 1, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    var ref_row: [2048]f32 = undefined;
    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..M) |row| {
        dequantRow(weight_buf.cpu_ptr.?[0..weight_buf.size], @intCast(row), K, .q8_0, &ref_row);
        var expected: f32 = 0;
        for (0..K) |i| expected += ref_row[i] * input_ptr[i];
        try std.testing.expectApproxEqAbs(expected, output_ptr[row], 0.01);
    }
}

test "dmmv_q8_0_k2048 large M across many workgroups (nr=2)" {
    const ctx = shim.mtl_init();
    try std.testing.expect(ctx != null);
    defer shim.mtl_destroy(ctx);

    var pipe = try loadShaderPipeline(ctx, "dmmv_q8_0_k2048");
    defer metal_pipeline.freePipeline(&pipe);

    const M: usize = 513; // odd, crosses many WG boundaries with nr=2
    const K: usize = 2048;
    const blocks_per_row: usize = K / 32;
    const row_bytes: usize = blocks_per_row * 34;

    const allocator = std.testing.allocator;

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
            const scale = @as(f16, @floatCast(0.03125 * @as(f32, @floatFromInt(1 + (row % 11) + (blk % 7)))));
            const scale_bits = @as(u16, @bitCast(scale));
            weight_buf.cpu_ptr.?[base] = @truncate(scale_bits);
            weight_buf.cpu_ptr.?[base + 1] = @truncate(scale_bits >> 8);
            for (0..32) |e| {
                const raw_q: i32 = @intCast((row * 7 + blk * 13 + e * 3) % 59);
                const q: i8 = @intCast(raw_q - 29);
                weight_buf.cpu_ptr.?[base + 2 + e] = @bitCast(q);
            }
        }
    }

    const input_ptr: [*]f32 = @ptrCast(@alignCast(input_buf.cpu_ptr.?));
    for (0..K) |i| {
        const raw: i32 = @intCast((i * 11 + 1) % 23);
        input_ptr[i] = 0.125 * @as(f32, @floatFromInt(raw - 11));
    }

    const push = DmmvPush{
        .M = @intCast(M),
        .K = @intCast(K),
        .a_offset = 0,
        .x_offset = 0,
        .y_offset = 0,
    };
    const bufs = [_]*const MetalBuffer{ &weight_buf, &input_buf, &output_buf };

    // nr=2 dispatch with 512-thread WGs (matching real engine config)
    const block_size: u32 = 512;
    const sgs_per_wg = block_size / 32;
    const rows_per_wg = sgs_per_wg * 2; // nr=2
    const wgs: u32 = @intCast((M + rows_per_wg - 1) / rows_per_wg);
    var cmd = try metal_command.beginCommand(ctx);
    cmd.dispatchV2(&pipe, .{ wgs, 1, 1 }, .{ block_size, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), 0);
    cmd.commitAndWait();

    const ref_row = try allocator.alloc(f32, K);
    defer allocator.free(ref_row);
    const expected = try allocator.alloc(f32, M);
    defer allocator.free(expected);
    @memset(expected[0..M], 0);

    for (0..M) |row| {
        dequantRow(weight_buf.cpu_ptr.?[0..weight_buf.size], @intCast(row), @intCast(K), .q8_0, ref_row);
        for (0..K) |i| expected[row] += ref_row[i] * input_ptr[i];
    }

    const output_ptr: [*]const f32 = @ptrCast(@alignCast(output_buf.cpu_ptr.?));
    for (0..M) |row| {
        try std.testing.expectApproxEqAbs(expected[row], output_ptr[row], 0.05);
    }
}
