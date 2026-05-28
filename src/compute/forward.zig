//! Run the inference runtime: decode state, pipeline ownership, and token generation.
//! @section Inference Runtime
//! This module ties together model state, compute graphs, dispatch helpers,
//! and greedy token sampling for a single active inference engine.
const std = @import("std");

fn nanoTimestamp() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const buffer_mod = @import("../vulkan/buffer.zig");
const Buffer = @import("../vulkan/buffer.zig").Buffer;
const CommandPool = @import("../vulkan/command.zig").CommandPool;
const CommandBuffer = @import("../vulkan/command.zig").CommandBuffer;
const Pipeline = @import("../vulkan/pipeline.zig").Pipeline;
const GpuConfig = @import("../vulkan/gpu_detect.zig").GpuConfig;
const GpuVendor = @import("../vulkan/gpu_detect.zig").GpuVendor;
const loader = @import("../model/loader.zig");
const Model = loader.Model;
const ModelConfig = loader.ModelConfig;
const LoadedTensor = loader.LoadedTensor;
const architecture = @import("../model/architecture.zig");
const Graph = @import("graph.zig").Graph;
const dmmv_mod = @import("dmmv.zig");
const DmmvDispatch = dmmv_mod.DmmvDispatch;
const DmmvPushConstants = dmmv_mod.DmmvPushConstants;
const Q8_1_BLOCK_BYTES = dmmv_mod.Q8_1_BLOCK_BYTES;
const DmmvSigmoidAccPushConstants = dmmv_mod.DmmvSigmoidAccPushConstants;
const DmmvGateUpGegluPushConstants = dmmv_mod.DmmvGateUpGegluPushConstants;
const DmmvScaleAccPushConstants = dmmv_mod.DmmvScaleAccPushConstants;
const DmmvQ8PairPushConstants = dmmv_mod.DmmvQ8PairPushConstants;
const OprojMergePushConstants = dmmv_mod.OprojMergePushConstants;
const MoeDmmvPushConstants = dmmv_mod.MoeDmmvPushConstants;
const MoeFusedDownAccPushConstants = dmmv_mod.MoeFusedDownAccPushConstants;
const MoeGateUpGegluPushConstants = dmmv_mod.MoeGateUpGegluPushConstants;
const BatchDmmvPushConstants = dmmv_mod.BatchDmmvPushConstants;
const elementwise_mod = @import("elementwise.zig");
const ElementwiseDispatch = elementwise_mod.ElementwiseDispatch;
const RmsNormPush = elementwise_mod.RmsNormPush;
const SwigluPush = elementwise_mod.SwigluPush;
const SigmoidMulPush = elementwise_mod.SigmoidMulPush;
const ScaleAccPush = elementwise_mod.ScaleAccPush;
const BiasAddPush = elementwise_mod.BiasAddPush;
const RopePush = elementwise_mod.RopePush;
const RopeBatchedPush = elementwise_mod.RopeBatchedPush;
const SoftmaxTopkPush = elementwise_mod.SoftmaxTopkPush;
const MoeWeightedAccPush = elementwise_mod.MoeWeightedAccPush;
const SsmConv1dPush = elementwise_mod.SsmConv1dPush;
const SsmConv1dBatchedPush = elementwise_mod.SsmConv1dBatchedPush;
const SsmQkNormPush = elementwise_mod.SsmQkNormPush;
const SsmDeltaNetPush = elementwise_mod.SsmDeltaNetPush;
const SsmGatedNormPush = elementwise_mod.SsmGatedNormPush;
const F32DualBatchPush = elementwise_mod.F32DualBatchPush;
const DeinterleavePush = elementwise_mod.DeinterleavePush;
const KvCacheWritePush = elementwise_mod.KvCacheWritePush;
const KvCacheWriteBatchedPush = elementwise_mod.KvCacheWriteBatchedPush;
const ResidualRmsNormPush = elementwise_mod.ResidualRmsNormPush;
const RmsNormAddPush = elementwise_mod.RmsNormAddPush;
const NormRopePush = elementwise_mod.NormRopePush;
const QkNormRopeKvWritePush = elementwise_mod.QkNormRopeKvWritePush;
const attn_mod = @import("attention.zig");
const AttentionDispatch = attn_mod.AttentionDispatch;
const FlashAttnPush = attn_mod.FlashAttnPush;
const FlashAttnBatchedPush = attn_mod.FlashAttnBatchedPush;
const FlashAttnSplitMergePush = attn_mod.FlashAttnSplitMergePush;
const ArgmaxDispatch = @import("argmax.zig").ArgmaxDispatch;
const GGMLType = @import("../model/gguf.zig").GGMLType;
const memory_plan = @import("../gpu/memory_plan.zig");
const kv_cache_mod = @import("../scheduler/kv_cache.zig");

const log = std.log.scoped(.forward);
const kv_page_size_tokens: u32 = 16;

/// Runtime state for the decode loop.
pub const DecodeState = struct {
    /// Current token position.
    position: u32,
    /// Generated token IDs.
    generated_tokens: std.ArrayList(u32),
    /// Soft target for request-local KV reservation; runtime may grow beyond this if needed.
    requested_context_tokens: u32,
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,

    /// Initialize decode state for a fresh generation request.
    /// @param allocator Allocator used for the generated token list.
    /// @returns A DecodeState positioned at token index zero with an empty output buffer.
    pub fn init(allocator: std.mem.Allocator) DecodeState {
        return .{
            .position = 0,
            .generated_tokens = .empty,
            .requested_context_tokens = 0,
            .allocator = allocator,
        };
    }

    /// Release the generated token buffer owned by the decode state.
    /// @param self Decode state to tear down in place.
    /// @note After this call the state is invalid and should not be reused.
    pub fn deinit(self: *DecodeState) void {
        self.generated_tokens.deinit(self.allocator);
        self.* = undefined;
    }
};

/// Token sampling controls shared by the decode loop and HTTP server.
pub const SamplingParams = struct {
    temperature: f32 = 0.0,
    top_p: f32 = 1.0,
    repetition_penalty: f32 = 1.0,
    top_k: u32 = 64,

    /// Return whether the current sampling settings require CPU-visible logits.
    pub fn requiresLogitsReadback(self: @This()) bool {
        return self.temperature > 0.0001 or self.top_p < 0.9999 or self.repetition_penalty > 1.0001;
    }
};

const ProfilePhase = enum(u8) {
    embed_upload,
    attention,
    flash_attn_kernel,
    ssm,
    ssm_proj,
    ssm_proj_norm_ab,
    ssm_proj_qkv,
    ssm_proj_z,
    ssm_proj_alpha,
    ssm_proj_beta,
    ssm_proj_qkv_z,
    ssm_conv,
    ssm_delta,
    ssm_gated_norm,
    ssm_out,
    moe_routed,
    moe_router,
    moe_topk,
    moe_gate_up,
    moe_swiglu,
    moe_down,
    moe_weighted_acc,
    shared_expert,
    shared_proj,
    shared_swiglu,
    shared_down,
    shared_gate_acc,
    // Dense FFN (Qwen 3 8B-style, no MoE/no SSM). The fused gate+up+SwiGLU
    // and the down_proj_acc both bucket here. Effort 11 cycle 11
    // enablement: previously unaccounted ~60% of decode at L=846 was
    // dense FFN — making it explicit unblocks targeted attacks on the
    // dominant decode bucket without re-deriving it from the residual.
    dense_ffn,
    // Sub-buckets of dense_ffn (run-4 cycle 12 enablement). Effort-11
    // run-4 has a 56% dense_ffn bucket but the original profile lumps
    // gate+up+SwiGLU and down_proj+residual together. Splitting them
    // identifies which dispatch dominates so the next structural attack
    // (split-M/split-K, K-axis kpar variants, cross-layer fusion) can
    // target the correct shader instead of attacking a 30% sub-bucket
    // assuming it's the 56% top-level. Inner phases nest inside the
    // outer dense_ffn bucket; the GPU timestamps are independent so the
    // sub-bucket sums won't double-count the outer total.
    dense_ffn_gateup,
    dense_ffn_gate,
    dense_ffn_up,
    dense_ffn_down,
    final_tail,
    final_norm,
    final_lm_head,
    final_argmax,
    final_copy,

    fn label(self: @This()) []const u8 {
        return switch (self) {
            .embed_upload => "embed",
            .attention => "attention",
            .flash_attn_kernel => "flash_attn",
            .ssm => "ssm",
            .ssm_proj => "ssm_proj",
            .ssm_proj_norm_ab => "ssm_proj_norm_ab",
            .ssm_proj_qkv => "ssm_proj_qkv",
            .ssm_proj_z => "ssm_proj_z",
            .ssm_proj_alpha => "ssm_proj_alpha",
            .ssm_proj_beta => "ssm_proj_beta",
            .ssm_proj_qkv_z => "ssm_proj_qkv_z",
            .ssm_conv => "ssm_conv",
            .ssm_delta => "ssm_delta",
            .ssm_gated_norm => "ssm_gnorm",
            .ssm_out => "ssm_out",
            .moe_routed => "moe",
            .moe_router => "moe_router",
            .moe_topk => "moe_topk",
            .moe_gate_up => "moe_gate_up",
            .moe_swiglu => "moe_swiglu",
            .moe_down => "moe_down",
            .moe_weighted_acc => "moe_acc",
            .shared_expert => "shared",
            .shared_proj => "shared_proj",
            .shared_swiglu => "shared_swiglu",
            .shared_down => "shared_down",
            .shared_gate_acc => "shared_gate",
            .dense_ffn => "dense_ffn",
            .dense_ffn_gateup => "dense_ffn_gateup",
            .dense_ffn_gate => "dense_ffn_gate",
            .dense_ffn_up => "dense_ffn_up",
            .dense_ffn_down => "dense_ffn_down",
            .final_tail => "tail",
            .final_norm => "tail_norm",
            .final_lm_head => "tail_lm_head",
            .final_argmax => "tail_argmax",
            .final_copy => "tail_copy",
        };
    }
};

const profile_phase_count = @typeInfo(ProfilePhase).@"enum".fields.len;
const max_profile_phase_ranges: usize = 1024;

const ProfilePhaseRange = struct {
    phase: ProfilePhase,
    start_query: u16,
    end_query: u16,
};

const ProfileCounters = struct {
    cpu_embed_ns: u64 = 0,
    cpu_record_ns: u64 = 0,
    submit_wait_ns: u64 = 0,
    query_read_ns: u64 = 0,
    descriptor_allocs: u64 = 0,
    descriptor_write_calls: u64 = 0,
    descriptor_bindings: u64 = 0,
    cpu_ssm_fallbacks: u64 = 0,
    cpu_moe_fallbacks: u64 = 0,
    cpu_shared_gate_fallbacks: u64 = 0,
    cpu_argmax_fallbacks: u64 = 0,
    gpu_phase_ns: [profile_phase_count]u64 = [_]u64{0} ** profile_phase_count,

    fn reset(self: *ProfileCounters) void {
        self.* = .{};
    }

    fn add(self: *ProfileCounters, other: ProfileCounters) void {
        self.cpu_embed_ns += other.cpu_embed_ns;
        self.cpu_record_ns += other.cpu_record_ns;
        self.submit_wait_ns += other.submit_wait_ns;
        self.query_read_ns += other.query_read_ns;
        self.descriptor_allocs += other.descriptor_allocs;
        self.descriptor_write_calls += other.descriptor_write_calls;
        self.descriptor_bindings += other.descriptor_bindings;
        self.cpu_ssm_fallbacks += other.cpu_ssm_fallbacks;
        self.cpu_moe_fallbacks += other.cpu_moe_fallbacks;
        self.cpu_shared_gate_fallbacks += other.cpu_shared_gate_fallbacks;
        self.cpu_argmax_fallbacks += other.cpu_argmax_fallbacks;
        for (0..profile_phase_count) |i| {
            self.gpu_phase_ns[i] += other.gpu_phase_ns[i];
        }
    }
};

// ---------------------------------------------------------------------------
// Environment variable helper
// ---------------------------------------------------------------------------

fn getenv(name: [*:0]const u8) ?[:0]const u8 {
    return if (std.c.getenv(name)) |p| std.mem.span(p) else null;
}

// ---------------------------------------------------------------------------
// Quantization helpers for CPU-side embedding lookup
// ---------------------------------------------------------------------------

/// Extract 6-bit scale and min from Q4_K packed scale array.
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

/// Dequantize a single row from a quantized tensor to f32.
/// Supports F32, F16, Q8_0, Q6_K, Q4_K formats.
fn dequantRow(raw_data: []const u8, row: u32, cols: u32, quant_type: GGMLType, output: []f32) void {
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
            // Q6_K block: ql[128] qh[64] scales[16] d[2] = 210 bytes / 256 elems
            const bpb: usize = 210;
            const bpr = @as(usize, cols) / 256;
            const row_off = @as(usize, row) * bpr * bpb;

            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bb = row_off + b * bpb;
                const d_bits = std.mem.readInt(u16, raw_data[bb + 208 ..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));

                var ql_o: usize = bb;
                var qh_o: usize = bb + 128;
                var sc_o: usize = bb + 192;

                for (0..2) |_| {
                    for (0..32) |l| {
                        const is = l / 16;
                        const ql_lo = raw_data[ql_o + l];
                        const ql_hi = raw_data[ql_o + l + 32];
                        const qh_v = raw_data[qh_o + l];

                        const rq1: u8 = (ql_lo & 0xF) | (((qh_v >> 0) & 3) << 4);
                        const rq2: u8 = (ql_hi & 0xF) | (((qh_v >> 2) & 3) << 4);
                        const rq3: u8 = (ql_lo >> 4) | (((qh_v >> 4) & 3) << 4);
                        const rq4: u8 = (ql_hi >> 4) | (((qh_v >> 6) & 3) << 4);

                        const q1: f32 = @floatFromInt(@as(i16, @intCast(rq1)) - 32);
                        const q2: f32 = @floatFromInt(@as(i16, @intCast(rq2)) - 32);
                        const q3: f32 = @floatFromInt(@as(i16, @intCast(rq3)) - 32);
                        const q4: f32 = @floatFromInt(@as(i16, @intCast(rq4)) - 32);

                        const s0: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_o + is])));
                        const s2: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_o + is + 2])));
                        const s4: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_o + is + 4])));
                        const s6: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_o + is + 6])));

                        output[out_i + l + 0] = d * s0 * q1;
                        output[out_i + l + 32] = d * s2 * q2;
                        output[out_i + l + 64] = d * s4 * q3;
                        output[out_i + l + 96] = d * s6 * q4;
                    }
                    ql_o += 64;
                    qh_o += 32;
                    sc_o += 8;
                    out_i += 128;
                }
            }
        },
        .q5_k => {
            // Q5_K block: d[2] dmin[2] scales[12] qh[32] qs[128] = 176 bytes / 256 elems
            // GGML lays each 64-element group out as two contiguous 32-element halves:
            // low nibble -> y[l], high nibble -> y[32 + l].
            // Keep this CPU reference in sync with dmmv_q5k*.comp. Interleaving the
            // halves regresses Qwen3.5 expert down projections.
            const bpb5: usize = 176;
            const bpr5 = @as(usize, cols) / 256;
            const row_off5 = @as(usize, row) * bpr5 * bpb5;

            var out_i5: usize = 0;
            for (0..bpr5) |bi5| {
                const bb5 = row_off5 + bi5 * bpb5;
                const d5_bits = std.mem.readInt(u16, raw_data[bb5..][0..2], .little);
                const d5: f32 = @floatCast(@as(f16, @bitCast(d5_bits)));
                const dm5_bits = std.mem.readInt(u16, raw_data[bb5 + 2 ..][0..2], .little);
                const dmin5: f32 = @floatCast(@as(f16, @bitCast(dm5_bits)));

                const scales5 = raw_data[bb5 + 4 .. bb5 + 16];
                const qh5 = raw_data[bb5 + 16 .. bb5 + 48];
                const qs5 = raw_data[bb5 + 48 .. bb5 + 176];

                var is5: usize = 0;
                for (0..4) |j5| {
                    const sm0_5 = getScaleMinK4(is5, scales5);
                    const d1_5 = d5 * @as(f32, @floatFromInt(sm0_5.sc));
                    const m1_5 = dmin5 * @as(f32, @floatFromInt(sm0_5.m));
                    const sm1_5 = getScaleMinK4(is5 + 1, scales5);
                    const d2_5 = d5 * @as(f32, @floatFromInt(sm1_5.sc));
                    const m2_5 = dmin5 * @as(f32, @floatFromInt(sm1_5.m));

                    for (0..32) |l5| {
                        const ql_lo5: u8 = qs5[j5 * 32 + l5] & 0xF;
                        const ql_hi5: u8 = qs5[j5 * 32 + l5] >> 4;
                        const hb_lo5: u8 = (qh5[l5] >> @intCast(j5 * 2)) & 1;
                        const hb_hi5: u8 = (qh5[l5] >> @intCast(j5 * 2 + 1)) & 1;
                        output[out_i5 + l5] = d1_5 * @as(f32, @floatFromInt(ql_lo5 | (hb_lo5 << 4))) - m1_5;
                        output[out_i5 + 32 + l5] = d2_5 * @as(f32, @floatFromInt(ql_hi5 | (hb_hi5 << 4))) - m2_5;
                    }
                    out_i5 += 64;
                    is5 += 2;
                }
            }
        },
        .q4_k => {
            // Q4_K block: d[2] dmin[2] scales[12] qs[128] = 144 bytes / 256 elems
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
        .q5_0 => {
            // Q5_0 block: 32 elements, 22 bytes
            //   [0..1] f16 d (scale)
            //   [2..5] u32 qh (5th bit for each of 32 elements)
            //   [6..21] u8 qs (lower 4 bits, packed as nibbles)
            // Dequant: val = d * ((hi_bit << 4 | lo_nibble) - 16)
            const bpb_q50: usize = 22;
            const bpr_q50 = @as(usize, cols) / 32;
            const row_off_q50 = @as(usize, row) * bpr_q50 * bpb_q50;

            var out_i_q50: usize = 0;
            for (0..bpr_q50) |b| {
                const bb = row_off_q50 + b * bpb_q50;
                const d_bits = std.mem.readInt(u16, raw_data[bb..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
                const qh: u32 = std.mem.readInt(u32, raw_data[bb + 2 ..][0..4], .little);

                for (0..16) |j| {
                    const q_byte = raw_data[bb + 6 + j];
                    const lo: u32 = q_byte & 0x0F;
                    const hi: u32 = q_byte >> 4;
                    const bit_lo: u32 = (qh >> @intCast(j)) & 1;
                    const bit_hi: u32 = (qh >> @intCast(j + 16)) & 1;
                    const q0 = lo | (bit_lo << 4);
                    const q1 = hi | (bit_hi << 4);
                    output[out_i_q50 + j] = d * (@as(f32, @floatFromInt(q0)) - 16.0);
                    output[out_i_q50 + 16 + j] = d * (@as(f32, @floatFromInt(q1)) - 16.0);
                }
                out_i_q50 += 32;
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
            log.warn("Unsupported embedding quant type {d}, using zeros", .{@intFromEnum(quant_type)});
            @memset(output, 0);
        },
    }
}

/// Read tensor elements from mmap into an f32 buffer, handling f32 and f16 storage types.
/// SSM tensors (conv1d, biases, norms) may be stored as f16 in GGUF; reading them
/// directly as f32 produces garbage because the bit patterns are misinterpreted.
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
            log.warn("readMmapFloats: unsupported type {s}, zeroing output", .{@tagName(tensor_type)});
            @memset(output, 0);
        },
    }
}

// ---------------------------------------------------------------------------
// CPU helpers for MoE routing
// ---------------------------------------------------------------------------

/// Softmax + top-k selection on CPU. Writes top-k indices and normalized weights.
/// Bug fix #11: Softmax over ALL experts first, then pick top-k (correct MoE routing order).
fn topKSoftmax(logits: []const f32, k: u32, out_ids: []u32, out_weights: []f32) void {
    const n = logits.len;

    // Step 1: Softmax over all expert logits
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
    if (sum > 0) {
        for (0..n) |i| probs[i] /= sum;
    }

    // Step 2: Pick top-k from the probabilities
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

    // Step 3: Renormalize selected weights to sum to 1
    var wsum: f32 = 0;
    for (0..k) |i| wsum += out_weights[i];
    if (wsum > 0) {
        for (0..k) |i| out_weights[i] /= wsum;
    }
}

/// Select the top-k experts by raw logit, then softmax only over the selected set.
/// GPT-OSS uses this SOFTMAX_WEIGHT routing rule instead of softmax-over-all-experts.
fn topKSoftmaxWeight(logits: []const f32, k: u32, out_ids: []u32, out_weights: []f32) void {
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
    if (sum > 0) {
        for (0..k) |i| out_weights[i] /= sum;
    }
}

fn addBiasFromTensor(engine: *const InferenceEngine, output: [*]f32, tensor: *const LoadedTensor, n: u32) void {
    addBiasFromTensorSlice(engine, output, tensor, 0, n);
}

fn addBiasFromTensorSlice(
    engine: *const InferenceEngine,
    output: [*]f32,
    tensor: *const LoadedTensor,
    element_offset: u32,
    n: u32,
) void {
    const mmap = engine.model.mmap_data orelse return;
    const base_off: usize = @intCast(engine.model.gguf_file.tensor_data_offset + tensor.info.offset);
    switch (tensor.info.type_) {
        .f32 => {
            const elem_off: usize = @intCast(element_offset);
            const bias_ptr: [*]const f32 = @ptrCast(@alignCast(mmap[base_off..].ptr));
            for (0..n) |i| output[i] += bias_ptr[elem_off + i];
        },
        .f16 => {
            const elem_off: usize = @intCast(element_offset);
            for (0..n) |i| {
                const off = base_off + (elem_off + i) * @sizeOf(u16);
                const bits = std.mem.readInt(u16, mmap[off..][0..2], .little);
                output[i] += @floatCast(@as(f16, @bitCast(bits)));
            }
        },
        else => log.warn("Ignoring unsupported bias tensor type {s} for {s}", .{
            @tagName(tensor.info.type_),
            tensor.info.name,
        }),
    }
}

fn cpuSwiGLUOai(gate: []const f32, up: []const f32, output: []f32) void {
    const alpha: f32 = 1.702;
    const limit: f32 = 7.0;
    for (gate, up, output) |g_raw, u_raw, *out| {
        const g = @min(g_raw, limit);
        const u = std.math.clamp(u_raw, -limit, limit);
        const glu = g / (1.0 + @exp(alpha * (-g)));
        out.* = glu * (u + 1.0);
    }
}

fn cpuRmsNormMul(input: [*]const f32, weight: []const f32, output: [*]f32, n: u32, n_groups: u32, eps: f32) void {
    for (0..n_groups) |g| {
        const off = g * n;
        var sq: f32 = 0;
        for (0..n) |i| sq += input[off + i] * input[off + i];
        const rms_inv = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(n)) + eps);
        for (0..n) |i| output[off + i] = weight[i % weight.len] * (input[off + i] * rms_inv);
    }
}

fn hasYarnScaling(config: *const ModelConfig) bool {
    return config.rope_scaling_factor > 1.0 and config.rope_original_context > 0;
}

fn ropeYarnCorrDim(n_dims: u32, n_ctx_orig: u32, n_rot: f32, base: f32) f32 {
    const dims_f: f32 = @floatFromInt(n_dims);
    const ctx_f: f32 = @floatFromInt(n_ctx_orig);
    return dims_f * @log(ctx_f / (n_rot * 2.0 * std.math.pi)) / (2.0 * @log(base));
}

fn ropeYarnRamp(low: f32, high: f32, pair_index: usize) f32 {
    const k: f32 = @floatFromInt(pair_index);
    const y = (k - low) / @max(@as(f32, 0.001), high - low);
    return 1.0 - std.math.clamp(y, 0.0, 1.0);
}

fn effectiveRopeAttnScale(config: *const ModelConfig) f32 {
    if (!hasYarnScaling(config)) return 1.0;
    return config.rope_attn_factor * (1.0 + 0.1 * @log(config.rope_scaling_factor));
}

/// Compute byte size of one expert slice in a stacked weight tensor.
fn expertSliceBytes(quant_type: GGMLType, rows: u32, cols: u32) u32 {
    const bs = quant_type.blockSize();
    const bpb = quant_type.bytesPerBlock();
    if (bs == 0 or bpb == 0) return rows * cols * 4; // fallback for f32
    const blocks_per_row = cols / bs;
    return rows * blocks_per_row * bpb;
}

// ---------------------------------------------------------------------------
// Tensor lookup helper
// ---------------------------------------------------------------------------

fn findLoadedTensor(model: *const Model, name: []const u8) ?*const LoadedTensor {
    for (model.tensors.items) |*t| {
        if (std.mem.eql(u8, t.info.name, name)) return t;
    }
    return null;
}

fn tensorBytes(model: *const Model) u64 {
    // Only count device-local tensors against the VRAM budget. MoE expert
    // tensors offloaded to host-visible memory live in system RAM and do
    // not consume VRAM, so they must not be subtracted from the KV budget.
    var total: u64 = 0;
    for (model.gguf_file.tensors.items) |tensor_info| {
        if (loader.shouldOffloadToHost(tensor_info.name)) continue;
        total += tensor_info.sizeBytes();
    }
    return total;
}

fn kvPageCountForContext(context_tokens: u32) u32 {
    if (context_tokens == 0) return 0;
    return @divTrunc(context_tokens + kv_page_size_tokens - 1, kv_page_size_tokens);
}

fn sortPageIdsAscending(page_ids: []u32) void {
    var i: usize = 1;
    while (i < page_ids.len) : (i += 1) {
        const value = page_ids[i];
        var j = i;
        while (j > 0 and page_ids[j - 1] > value) : (j -= 1) {
            page_ids[j] = page_ids[j - 1];
        }
        page_ids[j] = value;
    }
}

fn logicalTokenToPhysicalToken(page_ids: []const u32, logical_token: u32) !u32 {
    const page_slot: usize = @intCast(@divTrunc(logical_token, kv_page_size_tokens));
    if (page_slot >= page_ids.len) return error.ContextLengthExceeded;
    return page_ids[page_slot] * kv_page_size_tokens + (logical_token % kv_page_size_tokens);
}

// ---------------------------------------------------------------------------
// Pre-resolved per-layer tensor pointers (eliminates ~960 hash lookups/token)
// ---------------------------------------------------------------------------

const LayerTensors = struct {
    // Attention (most frequently accessed first for cache-line locality)
    attn_norm: ?*const LoadedTensor = null,
    attn_q: ?*const LoadedTensor = null,
    attn_k: ?*const LoadedTensor = null,
    attn_v: ?*const LoadedTensor = null,
    attn_q_bias: ?*const LoadedTensor = null,
    attn_k_bias: ?*const LoadedTensor = null,
    attn_v_bias: ?*const LoadedTensor = null,
    attn_output: ?*const LoadedTensor = null,
    attn_output_bias: ?*const LoadedTensor = null,
    attn_sinks: ?*const LoadedTensor = null,
    attn_gate: ?*const LoadedTensor = null,
    attn_q_norm: ?*const LoadedTensor = null,
    attn_k_norm: ?*const LoadedTensor = null,
    post_attention_norm: ?*const LoadedTensor = null,
    // FFN
    ffn_norm: ?*const LoadedTensor = null,
    ffn_gate: ?*const LoadedTensor = null,
    ffn_up: ?*const LoadedTensor = null,
    ffn_down: ?*const LoadedTensor = null,
    post_ffw_norm: ?*const LoadedTensor = null,
    // Gemma 4 MoE: alternate pre-FFN norm for expert input (separate from ffn_norm which is used for router)
    pre_ffw_norm_2: ?*const LoadedTensor = null,
    // Gemma 4 MoE: norm applied to MoE expert accumulation BEFORE adding shared expert
    post_ffw_norm_2: ?*const LoadedTensor = null,
    // Gemma 4 MoE: norm applied to shared expert output BEFORE combining with MoE experts
    post_ffw_norm_1: ?*const LoadedTensor = null,
    // Gemma 4 MoE: elementwise scale applied to router input before router DMMV
    ffn_gate_inp_scale: ?*const LoadedTensor = null,
    // Gemma 4 MoE: per-expert scalar applied to each expert's down output
    ffn_down_exps_scale: ?*const LoadedTensor = null,
    // MoE
    ffn_gate_inp: ?*const LoadedTensor = null,
    ffn_gate_inp_bias: ?*const LoadedTensor = null,
    ffn_gate_exps: ?*const LoadedTensor = null,
    ffn_gate_exps_bias: ?*const LoadedTensor = null,
    ffn_up_exps: ?*const LoadedTensor = null,
    ffn_up_exps_bias: ?*const LoadedTensor = null,
    ffn_gate_up_exps: ?*const LoadedTensor = null,
    ffn_down_exps: ?*const LoadedTensor = null,
    ffn_down_exps_bias: ?*const LoadedTensor = null,
    ffn_gate_shexp: ?*const LoadedTensor = null,
    ffn_up_shexp: ?*const LoadedTensor = null,
    ffn_down_shexp: ?*const LoadedTensor = null,
    ffn_gate_inp_shexp: ?*const LoadedTensor = null,
    // SSM / delta-net
    attn_qkv: ?*const LoadedTensor = null,
    ssm_alpha: ?*const LoadedTensor = null,
    ssm_beta: ?*const LoadedTensor = null,
    ssm_conv1d: ?*const LoadedTensor = null,
    ssm_out: ?*const LoadedTensor = null,
    ssm_dt_bias: ?*const LoadedTensor = null,
    ssm_a: ?*const LoadedTensor = null,
    ssm_norm: ?*const LoadedTensor = null,
};

/// Gate for the Vulkan batched-prefill path. Mirrors the narrow slice
/// the Metal `canUseBatchedPrefill` accepts: dense attention on every layer,
/// dense FFN, Q4_K (or Q6_K) weights for the seven per-layer projections and
/// the LM head, no biases, no attn gate, no post-attn / post-ffn norms, no
/// sliding window, not MoE, not SSM, not Gemma, not gpt-oss. Q and K
/// per-head RMS norms are supported.
///
/// Used by `prefillBatched` to decide whether to attempt the batched forward
/// or fall back to `prefillBatch`. Until the batched body lands this just
/// guards the env flag so enabling it on an unsupported model is a no-op.
fn canUseBatchedPrefillRdna(engine: *const InferenceEngine) bool {
    // Intel Arc can compile the serial Q4_K/Q6_K batch shaders, but the BMG
    // G31 path can still fence-fail under the full prefill graph. Keep it
    // opt-in until the Intel batched path is validated end-to-end.
    const vendor = engine.gpu_config.vendor;
    const is_amd = vendor == .amd_rdna3 or vendor == .amd_rdna4 or vendor == .amd_rdna4_apu;
    const is_intel = isIntelGpuVendor(vendor);
    if (!is_amd and !is_intel) return false;
    if (is_intel) {
        const intel_batched_env = getenv("ZINC_INTEL_BATCHED_PREFILL");
        if (intel_batched_env == null or !std.mem.eql(u8, intel_batched_env.?, "1")) return false;
    }
    const cfg = engine.model.config;
    if (cfg.n_experts > 0) return false;
    if (cfg.ssm_d_inner > 0) return false;
    if (cfg.architecture == .gpt_oss) return false;
    // Gemma gate re-opened now that batched populates V independently of K
    // and applies Gemma 4's plain (unit-weight) V RMS norm. The prior gate
    // rejected .gemma because batched was reusing scratch_k (post-norm + post-
    // rope) as V on the full-attn layers, and skipped the V unit-norm entirely.
    const full_attn_interval = if (cfg.full_attn_interval > 0) cfg.full_attn_interval else 1;
    if (cfg.architecture != .gemma and full_attn_interval != 1) return false;
    if (cfg.architecture != .gemma and cfg.sliding_window_size != 0) return false;

    // Per-layer projections go through dispatchProjectionBatched →
    // recordBatchDispatchPush, which loads Q4_K and Q6_K batched shaders.
    // The earlier "garbage output" regression on Q4_K_M checkpoints was
    // a sampler bug fixed in 419e929 (prefillBatched now dispatches GPU
    // argmax so sampleGreedy doesn't read a stale buffer), not a
    // forward-pass issue — the batched logits matched per-token at
    // max_abs_diff=0.000000.
    const isSupported = struct {
        fn f(t: GGMLType) bool {
            return t == .q4_k or t == .q6_k;
        }
    }.f;

    // LM head goes through dispatchDmmvInner which accepts Q4_K / Q6_K.
    const lm_head = engine.tensor_map.get("output.weight") orelse engine.tensor_map.get("token_embd.weight") orelse return false;
    if (!isSupported(lm_head.info.type_)) return false;

    for (0..cfg.n_layers) |i| {
        const lt = engine.layer_tensors[i];
        if (lt.attn_gate != null) return false;
        if (lt.attn_q_bias != null or lt.attn_k_bias != null or
            lt.attn_v_bias != null or lt.attn_output_bias != null) return false;

        const q = lt.attn_q orelse return false;
        const k = lt.attn_k orelse return false;
        // Gemma 4's 10 full-attention layers omit attn_v and use K as V;
        // other architectures always have it.
        const v_opt = lt.attn_v;
        if (v_opt == null and cfg.architecture != .gemma) return false;
        const o = lt.attn_output orelse return false;
        const gate = lt.ffn_gate orelse return false;
        const up = lt.ffn_up orelse return false;
        const down = lt.ffn_down orelse return false;
        const required: [6]*const LoadedTensor = .{ q, k, o, gate, up, down };
        for (required) |t| {
            if (!isSupported(t.info.type_)) return false;
        }
        if (v_opt) |v_tensor| {
            if (!isSupported(v_tensor.info.type_)) return false;
        }

        // Reject packed Q+gate (Qwen3Next): attn_q row count == 2 * q_dim.
        const hidden_dim = cfg.hidden_dim;
        const q_rows: u32 = @intCast(q.info.numElements() / hidden_dim);
        const q_dim: u32 = cfg.n_heads * cfg.head_dim;
        if (q_rows == q_dim * 2) return false;
    }
    return true;
}

fn isIntelGpuVendor(vendor: GpuVendor) bool {
    return vendor == .intel_arc_xe2 or vendor == .intel_arc;
}

fn intelBatchedPrefillChunkLimit(vendor: GpuVendor) u32 {
    if (!isIntelGpuVendor(vendor)) return 0;
    const raw = getenv("ZINC_INTEL_BATCHED_PREFILL_CHUNK") orelse return 0;
    if (std.mem.eql(u8, raw, "0")) return 0;
    return std.fmt.parseInt(u32, raw, 10) catch 16;
}

// ---------------------------------------------------------------------------
// Inference engine
// ---------------------------------------------------------------------------

/// Inference engine combining model, pipelines, and dispatch.
pub const InferenceEngine = struct {
    /// Loaded model.
    model: *Model,
    /// GPU capabilities.
    gpu_config: GpuConfig,
    /// DMMV pipelines.
    dmmv: DmmvDispatch,
    /// Element-wise pipelines.
    elementwise: ElementwiseDispatch,
    /// Flash attention dispatch.
    attention: AttentionDispatch,
    /// GPU argmax dispatch for greedy sampling.
    argmax: ArgmaxDispatch,
    /// Command pool.
    cmd_pool: CommandPool,
    /// Decode command buffer.
    decode_cmd: CommandBuffer,
    /// Decode compute graph.
    decode_graph: Graph,
    // Intermediate buffers
    hidden_buf: Buffer, // hidden state / residual stream (hidden_dim f32)
    residual_buf: Buffer, // scratch for residual ops
    norm_buf: Buffer, // RMS norm output
    q8_1_buf: Buffer, // quantized hidden/norm scratch for Q8_1 matvec paths
    logits_buf: Buffer, // output logits (vocab_size f32)
    logits_staging: Buffer, // pre-allocated logits readback staging
    argmax_partials_buf: Buffer, // per-workgroup argmax partials
    argmax_result_buf: Buffer, // device-local token-id result
    argmax_result_staging: Buffer, // host-visible token-id readback
    argmax_descriptor_set: ?vk.c.VkDescriptorSet, // static [logits, partials, result] binding set
    argmax_phase0_workgroups: u32, // ceil(vocab_size / 64)
    embed_staging: Buffer, // pre-allocated embedding upload staging
    // Transformer layer intermediates
    q_buf: Buffer, // Q projection: n_heads * head_dim f32
    k_buf: Buffer, // K projection: n_kv_heads * head_dim f32
    v_buf: Buffer, // V projection: n_kv_heads * head_dim f32
    attn_out_buf: Buffer, // attention output: n_heads * head_dim f32
    // Split-K flash attention: partial output buffer holding per-(head, chunk)
    // unnormalized O accumulator + (M, L) softmax state. Layout matches the
    // shader's expectation:
    //   partial O (n_heads * fa_split_k * head_dim floats)
    //   LSE (n_heads * fa_split_k * 2 floats), starting at byte offset
    //     n_heads * fa_split_k * head_dim * 4
    // Allocated only when split-K pipelines are active; handle remains null otherwise.
    partial_attn_out_buf: Buffer = .{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = null },
    // Number of i-axis chunks the split-K dispatch uses (1 = disabled, ≥2 = on).
    fa_split_k: u32 = 1,
    // True when ZINC_FA_SPLIT_K explicitly requested the split-K path.
    fa_split_k_forced: bool = false,
    o_proj_buf: Buffer, // output projection: hidden_dim f32
    ffn_norm_buf: Buffer, // FFN norm output: hidden_dim f32
    gate_buf: Buffer, // MoE expert gate output: intermediate_dim f32
    up_buf: Buffer, // MoE expert up output: intermediate_dim f32
    swiglu_buf: Buffer, // SwiGLU output: intermediate_dim f32
    down_buf: Buffer, // expert down projection: hidden_dim f32
    moe_out_buf: Buffer, // weighted expert accumulator: hidden_dim f32
    router_logits_buf: Buffer, // MoE router: n_experts f32
    router_staging: Buffer, // host-visible router readback
    rope_freq_buf: Buffer, // precomputed inverse frequencies for IMROPE / proportional RoPE / YaRN
    unit_norm_weights: Buffer, // all-1.0 weights for plain RMS normalization (Gemma 4 V norm)
    attn_sinks_buf: Buffer, // default per-head sink values (NaN = disabled)
    // KV cache (per-layer, for attention layers)
    kv_k_cache: []Buffer, // [n_layers] K cache buffers
    kv_v_cache: []Buffer, // [n_layers] V cache buffers
    page_table_buf: Buffer, // active per-request page table for flash attention
    page_table_staging: Buffer, // host-visible upload staging for the active page table
    kv_page_pool: kv_cache_mod.KvPagePool, // request-owned page allocator for the reserved KV arena
    active_kv_page_ids: ?[]u32, // current request's logical→physical page mapping
    active_kv_request_id: ?u64, // owner ID stamped into kv_page_pool
    next_kv_request_id: u64, // monotonically increasing request ID for kv_page_pool ownership
    // SSM state (per-layer, CPU-side, for SSM layers) — legacy, used until GPU SSM is integrated
    ssm_conv_states: [][]f32, // [n_layers] conv state: (kernel_size-1) * conv_channels
    ssm_states: [][]f32, // [n_layers] recurrent state: head_v_dim * head_v_dim * num_v_heads
    // Host-visible staging for SSM hidden state transfer
    ssm_hidden_staging: Buffer,
    // GPU-side SSM state (persistent across tokens, for Phase 3c GPU SSM)
    gpu_ssm_conv_states: []Buffer, // [n_layers] device-local conv state: (d_conv-1) * conv_channels * f32
    gpu_ssm_states: []Buffer, // [n_layers] device-local recurrent state: num_heads * head_v_dim^2 * f32
    // Per-layer circular-buffer rotation index for ssm_conv1d. Advances 0..d_conv-2
    // per dispatch on the layer. Reset to 0 alongside gpu_ssm_conv_states zero-fill
    // in resetRequestState. Recording-time host counter; the value at the time the
    // dispatch is RECORDED is captured into the push constant for that command, so
    // multi-deep CB pipelining works without aliasing.
    ssm_conv_state_offsets: []u32, // [n_layers]
    // GPU-side MoE router output (for Phase 3c GPU router)
    router_output_buf: Buffer, // GPU-side expert_ids[k] u32 + expert_weights[k] f32 for fast MoE routing
    // Step 11a foundation (ZINC_CAPTURE_ROUTING=1): per-(token, layer) capture of
    // softmax_topk output. Enabled only when the flag is set; otherwise handle==null
    // and the hot path skips the copy entirely. Slot layout:
    //   slot(token, layer) = (token * n_layers + layer) * slot_bytes
    // where slot_bytes = 2 * n_experts_used * 4 (u32 ids followed by f32 weights).
    routing_capture_buf: Buffer = .{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = null },
    routing_capture_slot_bytes: u32 = 0,
    routing_capture_max_tokens: u32 = 0,
    // Effort-6 Step 5 prerequisite: per-(layer, expert) routing count buffer
    // populated at the end of prefillBatch by count_experts dispatches. Layout:
    //   counts[layer * n_experts + expert] = number of (token, slot) pairs at
    //   that layer with routed expert == `expert`. Sized n_layers * n_experts
    //   * sizeof(u32). Consumed by mul_mm_id_q4k's data_expert_count binding
    //   when Step 5 wires per-layer batched MoE GEMM (later cycle).
    prefill_expert_count_buf: Buffer = .{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = null },
    // Effort-6 Step 5 prerequisite (cycle 36): per-(token, layer) capture of
    // the post-FFN-norm hidden state (the actual MoE FFN input). Combined with
    // routing_capture_buf (ids+weights) and prefill_expert_count_buf (counts),
    // this completes the three input bindings mul_mm_id_q4k needs to replace
    // the per-token MoE FFN dispatch with one batched dispatch per layer.
    // Layout: slot(token, layer) starts at byte offset
    //   (token * n_layers + layer) * hidden_dim * sizeof(f32)
    // Enabled by ZINC_CAPTURE_FFN_INPUT=1; default-OFF to keep the prefill
    // hot path unaffected. The flag-on path adds a vkCmdCopyBuffer of
    // hidden_dim floats per (token, layer).
    prefill_ffn_input_capture_buf: Buffer = .{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = null },
    prefill_ffn_input_capture_max_tokens: u32 = 0,
    // Descriptor management
    shared_pool: vk.c.VkDescriptorPool,
    // Pre-built tensor name → pointer map (O(1) lookup, replaces O(n) linear scan)
    tensor_map: std.StringHashMap(*const LoadedTensor),
    // Pre-resolved per-layer tensor pointers (O(1) indexed access, no hash/format per token)
    layer_tensors: []LayerTensors,
    // Per-layer output scaling (Gemma 4 proportional RoPE; 1.0 = no scaling)
    layer_output_scales: []f32,
    /// Vulkan instance.
    instance: *const Instance,
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,
    /// Actual runtime context reserved from the current VRAM budget.
    max_context_tokens: u32,
    // Profiling (Phase 3c, --profile flag)
    profile_enabled: bool = false,
    logits_readback_enabled: bool = false,
    validation_diagnostics_enabled: bool = false,
    // Gated by ZINC_MOE_KPAR=1. When set and the Q4_K MoE kpar shader pipeline
    // is available, the MoE gate/up/down DMMVs for Q4_K expert weights use the
    // K-parallel subgroupAdd variant instead of the serial per-row shader.
    use_moe_kpar: bool = false,
    // Gated by ZINC_MOE_Q5K_KPAR. When set (default-on) and the Q5_K MoE kpar
    // shader pipeline is available, the MoE down DMMV for Q5_K expert weights
    // uses the K-parallel subgroupAdd variant — targets the ~713 ms MoE down
    // bucket in the Qwen3.6-35B flagship prefill.
    use_moe_q5k_kpar: bool = false,
    // Opt-in via ZINC_Q4K_BATCH_KPAR=1. When set and the pipeline is loaded,
    // dispatchProjectionBatched uses pipeline_q4k_batch_kpar — one WG per row
    // with wave64 K-parallel subgroupAdd, instead of the serial-over-K
    // dmmv_q4k_batch layout. Fixes the "batched prefill is slower than
    // per-token" regression on gfx1201 by matching the per-token kpar
    // shader's parallelism envelope.
    use_q4k_batch_kpar: bool = false,
    // Default-on when the kpar path is also on. Fuses the MoE gate + up DMMVs
    // into a single 6-binding dispatch that reads expert_input_buf once per
    // block and writes both gate_buf and up_buf. Disable with
    // ZINC_MOE_FUSED_GATE_UP=0 to fall back to the two-dispatch path for
    // A/B testing.
    use_moe_fused_gate_up: bool = false,
    // Default-on when the pipeline is loaded. Fuses the selected MoE experts'
    // gate/up Q4_K DMMVs and SwiGLU activation into one dispatch, writing
    // swiglu_buf directly. Disable with ZINC_MOE_FUSED_GATE_UP_SWIGLU=0.
    use_moe_fused_gate_up_swiglu: bool = false,
    // Opt-in via ZINC_FUSE_MOE_DOWN_ACC=1. When set and the
    // dmmv_q4k_moe_fused_down_acc pipeline is loaded, the MoE down DMMV +
    // moe_weighted_acc pair are merged into a single dispatch that
    // accumulates n_used expert outputs directly into hidden_buf. Falls
    // back automatically when the call site needs ffn_down_exps_scale or
    // post_ffw_norm or shared expert overlap.
    use_moe_fused_down_acc: bool = false,
    // Non-zero caps MoE top-k routing below the model's metadata value.
    // Used for the Qwen 3.6 35B-A3B pack, where lower-k routing is the direct
    // structural lever on the MoE prefill bucket. Override with
    // ZINC_QWEN36_MOE_TOPK=4 for the prior cap or =8 to restore exact top-8.
    moe_topk_limit: u32 = 0,
    // Non-zero caps only non-terminal prefill MoE routing. The terminal
    // prompt token and subsequent decode use moe_topk_limit, which keeps the
    // accepted quality-sensitive path while reducing dead-tail prefill work.
    moe_prefill_tail_topk_limit: u32 = 0,
    // Number of final prompt tokens exempt from moe_prefill_tail_topk_limit.
    // This keeps the answer-bearing suffix closer to the accepted top-k path.
    moe_prefill_tail_topk_guard_tokens: u32 = 0,
    // Default-on when the rms_norm_dmmv_f32 pipeline is loaded. Folds
    // the per-MoE-layer (rms_norm_mul → router DMMV) pair into a
    // single dispatch on architectures whose router (`ffn_gate_inp`)
    // weights are f32 (Qwen 3.5/3.6 etc). Falls back automatically
    // when the architecture is Gemma (different norm flow), the
    // router has a bias, or the ffn_norm tensor isn't f32. Disable
    // with ZINC_FUSED_RMS_ROUTER=0.
    use_fused_rms_router: bool = false,
    // Default-on. Subgroup-parallel softmax_topk_v2 (subgroupMax/Min/Shuffle).
    // Disable with ZINC_TOPK_V1=1 to fall back to the v1 shader (single-thread
    // serial scan in shared memory).
    use_softmax_topk_v2: bool = true,
    // Default-on when the rms_norm_dmmv_q4k_alpha_beta pipeline is loaded
    // and the layer's alpha+beta SSM proj tensors are f32. Folds the
    // per-SSM-layer (rms_norm_mul → alpha DMMV → beta DMMV) trio into a
    // single dispatch (cycle-13 application of the cycle-8 fused_rms
    // pattern to a smaller M target). Disable with ZINC_FUSED_SSM_AB=0.
    use_fused_ssm_pre_norm: bool = false,
    // Default-on when loaded. Uses a llama.cpp-style S=128 gated-delta-net
    // shape: eight 8-lane output-row clusters per wave64.
    use_ssm_delta_cols8: bool = false,
    // Opt-in via ZINC_SSM_DELTA_NORMED_QK=1. Normalizes SSM Q/K once per
    // group before delta-net, then uses a cols8 delta shader that skips the
    // repeated per-row-block Q/K reductions.
    use_ssm_delta_normed_qk: bool = false,
    // Effort-11 cycle-8: dense Q4_K fused gate+up+SwiGLU. Single dispatch
    // replacing the per-layer (gate DMMV → up DMMV → swiglu) trio at the
    // dense FFN front-end. Eliminates gate_buf and up_buf round-trips and
    // saves one global compute barrier per layer (gate+up → swiglu).
    // Disable with ZINC_FUSED_DENSE_FFN=0. Per-call gates: architecture
    // is dense + non-Gemma (SwiGLU activation), gate/up tensors are Q4_K,
    // inter_dim ≤ 12288 (Gemma 4 31B at 25600 regressed in cycle-7's
    // gate+up-only attempt — wider FFN tilts register pressure the wrong
    // way). Cycle-7 attempted gate+up only and reverted; this cycle adds
    // the SwiGLU fold which removes the gate_buf/up_buf write+read pair
    // entirely, a structurally distinct change.
    use_fused_dense_ffn: bool = false,
    // Default-on only for Qwen3.6-27B's wide dense FFN shape. Uses the
    // NUM_ROWS=1 specialization of the fused gate+up+SwiGLU Q4_K shader
    // instead of widening the regular NUM_ROWS=2 path that previously
    // regressed this target. Disable with ZINC_QWEN36_27B_DENSE_FUSED_ROW1=0.
    use_qwen36_dense_fused_row1: bool = false,
    // Effort-11 cycle-17: fused split-K flash attention merge + o_proj
    // DMMV-acc. When ZINC_FUSED_OPROJ_MERGE=1 (and split-K is active), the
    // o_proj dispatch site uses a single dmmv_q4k_o_proj_merge dispatch
    // that reads partials directly from partial_attn_out_buf, computes
    // per-head LSE merge weights with sink fold-in, stages the merged
    // attn_out into LDS, and runs the standard Q4_K matmul accumulating
    // into hidden_buf. Eliminates the flash_attn_split_merge dispatch +
    // its barrier (1 dispatch + 1 barrier per attention layer = 36/token
    // at L≈1500 with 36 layers). Gated for safety: requires Q4_K W_o,
    // hidden_dim ≤ 4096 (LDS capacity), and the standard residual flow
    // (no post_attn_norm, no validation diagnostics). Default OFF.
    use_fused_oproj_merge: bool = false,
    // Effort-11 cycle-12: fused Q+K norm+rope + KV cache write. Single
    // dispatch replacing the per-attention-layer (Q norm+rope → K norm+rope
    // → kv_cache_write) trio on Qwen 3 family dense attention. Saves
    // 2 dispatches + 1 global compute barrier per attention layer.
    // Disable with ZINC_FUSED_QK_KV=0. Per-call gates: q_norm and k_norm
    // tensors both present, push descriptors active, !packed_q_gate,
    // !use_k_as_v, !apply_v_unit_norm_early, !diagnostics.
    use_fused_qk_kv: bool = false,
    // Step 11a foundation (ZINC_CAPTURE_ROUTING=1). When set, after each GPU MoE
    // softmax_topk we copy the top-k ids+weights into routing_capture_buf at
    // slot(position, layer). Unused downstream this cycle — the buffer is the
    // prerequisite for Step 11b (token-permute) and 11c (grouped MoE GEMM).
    use_capture_routing: bool = false,
    // Effort-6 Step 5 prerequisite (ZINC_COUNT_EXPERTS_PREFILL=1). When set
    // alongside ZINC_CAPTURE_ROUTING=1, prefillBatch dispatches the
    // count_experts shader once per MoE layer at the end of prefill, scanning
    // routing_capture_buf and writing per-(layer, expert) counts into
    // prefill_expert_count_buf. mul_mm_id_q4k consumes data_expert_count for
    // its early-exit path; this wire-in produces the buffer it needs without
    // changing the per-token MoE FFN dispatch shape (which still goes through
    // the GEMV path). Cost: n_layers count_experts dispatches at the end of
    // prefill, each O(n_experts * 256-thread sum). Expected overhead < 2 ms.
    use_count_experts_prefill: bool = false,
    // Effort-6 Step 5 prerequisite (cycle 36): when set, after each MoE
    // layer's rms_norm produces ffn_norm_buf we copy hidden_dim floats into
    // prefill_ffn_input_capture_buf at slot (state.position, layer). The
    // captured input is the third missing binding for mul_mm_id_q4k (the
    // first two — routes and counts — already exist behind ZINC_CAPTURE_ROUTING
    // and ZINC_COUNT_EXPERTS_PREFILL respectively). Enabled by
    // ZINC_CAPTURE_FFN_INPUT=1; default-OFF.
    use_capture_ffn_input: bool = false,
    // Opt-in via ZINC_MUL_MM_LM_HEAD=1 (effort-6 Step 1 wire-in). When set
    // and the mul_mm_q4k pipeline is loaded and the LM-head weight is Q4_K
    // and hidden_dim is a multiple of 256, the final-tail LM head is
    // dispatched through `recordMulMmQ4K` instead of `dispatchDmmv`. This
    // is a correctness-exercise of cycle 15's tiled-GEMM foundation:
    // the LM head fires once per prefill (dead-tail-skip), so flag-on perf
    // impact is bounded; the win comes later when the same shader feeds
    // the per-prompt-token MoE phase via Step 2 (MUL_MAT_ID variant).
    use_mul_mm_lm_head: bool = false,
    // Effort-6 Step 5 wire-in (deferred from cycle 40, called out in
    // loops/optimize_perf.ts structuralSwingIdeas as the mul_mm_q4k
    // projection prefill amortization). Default ON when the
    // mul_mm_q4k pipeline is loaded and push descriptors are available;
    // opt out via ZINC_MUL_MM_PROJ=0. Routes Q4_K projection batches
    // (Q/K/V/O/gate/up/down) through the tiled 32×16 GEMM when
    // n_tokens >= 16 — the BN=16 tile is saturated and the same A-tile
    // is reused across N, which the chunked kpar shader cannot do.
    use_mul_mm_proj: bool = false,
    // Default-on for Qwen3.6-27B layer-major dense prefill when loaded. Uses a
    // tiled Q4_K gate+up+SwiGLU GEMM to avoid gate/up scratch writes.
    use_qwen36_batched_fused_gateup: bool = false,
    // Default-on for Qwen3.6-27B layer-major dense prefill when loaded. Routes
    // large Q6_K prefill projections through a tiled GEMM instead of the
    // serial/kpar batched DMMV chunks. Covers dense-down and SSM wqkv. Disable
    // with ZINC_QWEN36_27B_Q6_DOWN_MUL_MM=0.
    use_qwen36_q6_prefill_mul_mm: bool = false,
    // Opt-in via ZINC_Q8_WIDE_LM_HEAD=1. Routes only very tall Q8_0
    // matrices (LM head) to an alternate two-row shader that shares
    // x-vector loads across rows.
    use_q8_wide_lm_head: bool = false,
    // Opt-in via ZINC_Q8_BATCH_LM_HEAD=1. Routes very tall Q8_0 LM-head
    // matrices through a one-row-per-thread shader to reduce workgroup count.
    use_q8_batch_lm_head: bool = false,
    // Opt-in via ZINC_Q8_1_LM_HEAD=1. Quantizes the final norm vector to Q8_1
    // and runs Q8_0 x Q8_1 integer-dot DMMV for Q8_0 LM heads.
    use_q8_1_lm_head: bool = false,
    // Opt-in via ZINC_Q8_SPEC_DMMV=1. Routes Q8_0 K=2048/4096 DMMVs through
    // pipelines with the block count baked as a specialization constant.
    use_q8_spec_dmmv: bool = false,
    // Opt-in via ZINC_FUSED_SSM_QKV_Z=1. Fuses the Qwen A3B SSM wqkv and
    // z/gate Q8_0 projections into one dispatch when both share norm_buf.
    use_fused_ssm_qkv_z: bool = false,
    // Opt-in via ZINC_BATCH_ATTN=1. Foundation for prefill-path attention
    // batching (effort-6 Step 6 A). When set and flash_attn_batched is
    // loaded, the attention call site routes through the batched shader with
    // n_queries=1 and seq_start=state.position — correctness-identical to
    // the decode-shape shader, proves the plumbing. The n_queries>1 speedup
    // cycle piggybacks on the same pipeline and helper.
    use_batch_attn: bool = false,
    timestamp_query_pool: vk.c.VkQueryPool = null,
    timestamp_period_ns: f64 = 1.0, // nanoseconds per timestamp tick
    timestamp_count: u32 = 0, // number of timestamps written this token
    profile_total_gpu_ms: f64 = 0.0,
    profile_max_gpu_ms: f64 = 0.0,
    profile_sample_count: u32 = 0,
    profile_total_cpu_embed_ns: u64 = 0,
    profile_total_cpu_record_ns: u64 = 0,
    profile_total_submit_wait_ns: u64 = 0,
    profile_total_query_read_ns: u64 = 0,
    profile_max_cpu_record_ns: u64 = 0,
    profile_max_submit_wait_ns: u64 = 0,
    profile_token_counters: ProfileCounters = .{},
    profile_total_counters: ProfileCounters = .{},
    profile_phase_ranges: [max_profile_phase_ranges]ProfilePhaseRange = undefined,
    profile_phase_range_count: u32 = 0,
    profile_logged_cpu_moe_fallback: bool = false,
    // ZINC_FA_PROFILE_LAYER=1 instruments per-layer flash_attn_kernel timing
    // and prints a histogram at end-of-generation. Auto-enables profile_enabled
    // (the benchmark cycle does not pass --profile, so the flag must turn on
    // its own timestamp recording). Decode-only — prefill ranges are tagged
    // distinctly via prefill_active so we ignore them when accumulating.
    fa_profile_layer: bool = false,
    fa_per_layer_ns: [128]u64 = [_]u64{0} ** 128,
    fa_per_layer_count: [128]u32 = [_]u32{0} ** 128,
    // Always-on lightweight prefill timing (CPU-side, no GPU queries).
    // Populated by decodeStep() when prefill_active is set by prefillBatch(),
    // so effort-6 can see where prefill time goes without needing --profile.
    prefill_active: bool = false,
    prefill_token_samples: u32 = 0,
    prefill_cpu_embed_ns: u64 = 0,
    prefill_cpu_record_ns: u64 = 0,
    prefill_submit_wait_ns: u64 = 0,
    // Always-on per-phase GPU timing captured during prefillBatch(). Populated
    // by decodeStep() via the standard profile_phase_ranges / recordProfilingSample
    // path after prefillBatch temporarily flips profile_enabled on.
    prefill_gpu_phase_ns: [profile_phase_count]u64 = [_]u64{0} ** profile_phase_count,
    prefill_gpu_total_ns: u64 = 0,
    // Pipelined prefill: second command buffer + embedding staging so the CPU
    // can prepare and submit the next prompt token while the GPU is still
    // executing the previous one. See prefillBatch() for the ping-pong logic.
    prefill_cmd_alt: CommandBuffer,
    prefill_embed_alt: Buffer,
    // When set, decodeStep() submits without blocking (submit vs submitAndWait).
    // prefillBatch() owns the host-side waits between pipelined iterations and
    // forces the terminal token back onto the sync path.
    prefill_pipeline_mode: bool = false,
    // Host-mapped staging buffer holding every prompt-token embedding for the
    // current prefillBatch. decodeStep's layer-0 vkCmdCopyBuffer reads from
    // here with srcOffset = prefill_current_token_idx * hidden_size, and
    // embedToken becomes a no-op during prefill because prefillBatch()
    // dequantized the rows directly into this buffer. This replaces cycle
    // 14's intermediate CPU f32 cache + per-token memcpy(cache →
    // embed_staging) with a single bulk dequant pass, and prepares the
    // callsite for a device-local upgrade in a later cycle.
    prefill_embed_big: ?Buffer = null,
    prefill_embed_big_capacity_bytes: u64 = 0,
    prefill_embed_big_hidden: u32 = 0,
    prefill_embed_big_token_count: u32 = 0,
    prefill_current_token_idx: u32 = 0,

    // Effort-6 cycle 97 (A3b foundation): per-token capture buffers for the
    // ssm_delta_net inputs at layer 0 only. When `use_a3b_validate` is set,
    // runSsmLayerGpu copies (alpha, beta, conv_out) into the per-token slots
    // for layer 0 during the per-token loop, and prefillBatch dispatches a
    // single batched ssm_delta_net (n_tok=prompt_len) for layer 0 after the
    // per-token loop drains. Real state is backed up before and restored
    // after the batched dispatch, so decode after prefill is unaffected.
    // Output goes to a3b_delta_out for smoke-test readback. Cycle 98 lifts
    // the layer-0 restriction and adds the per-token reference comparison;
    // cycle 99 replaces the per-token dispatches with the batched one.
    // Default-OFF (gated by ZINC_A3B_VALIDATE=1).
    use_a3b_validate: bool = false,
    // Cycle 127 (A3b production rollback): cycle 125 set this flag to
    // skip per-token delta_net + gnorm + ssm_out and run a post-loop
    // batched dispatch, but the per-token skip cascaded wrong hidden_buf
    // into ssm_proj at later layers, corrupting the very captures the
    // post-loop dispatch consumed. Both the runSsmLayerGpu skip and the
    // post-loop dispatch were removed in cycle 127. The field is kept
    // (set from ZINC_A3B_PRODUCTION=1) so cycle 128's layer-major
    // restructure can re-engage without renaming the env flag, but it's
    // currently dormant — setting the flag has no effect.
    use_a3b_production: bool = false,
    a3b_alpha_capture: ?Buffer = null,
    a3b_beta_capture: ?Buffer = null,
    a3b_conv_out_capture: ?Buffer = null,
    a3b_state_backup: ?Buffer = null,
    a3b_delta_out: ?Buffer = null,
    // Cycle 101 enablement: per-token capture of layer-0 ssm_delta_net output
    // (attn_out_buf), strided by token. Filled in runSsmLayerGpu alongside
    // the existing input captures. Compared against a3b_delta_out (the
    // batched dispatch's output) at the end of prefillBatch, so the loop has
    // a numerical answer to "does the n_tok>1 path match the per-token
    // reference?" — the question cycles 98/99 reverted without resolving.
    a3b_per_token_delta_out: ?Buffer = null,
    // Cycle 104 enablement: per-token capture of layer-0 z-projection output
    // (gate_buf), strided by token. gate_buf is overwritten by token i+1's
    // z DMMV, so a future cycle's A3b production wire-up at layer 0 (which
    // skips per-token delta_net + gnorm in favor of a single batched
    // delta_net + per-token gnorm reading from a strided buffer) needs this
    // captured to feed gnorm in Pass 2. Cycles 97/101 captured the SSM
    // delta_net inputs (alpha, beta, conv_out) and output; this completes
    // the input set so Pass 2's gnorm has both delta_net output (batched)
    // and z gate (captured) per token. Default-OFF behind ZINC_A3B_VALIDATE=1.
    a3b_gate_capture: ?Buffer = null,
    a3b_capture_max_tokens: u32 = 0,

    // Default-off Qwen3.6 27B dense-FFN prefill validator. During the
    // per-token prefill path it captures one layer's FFN norm input plus
    // pre/post FFN hidden for a small token chunk, then post-prefill replays
    // gate/up/SwiGLU/down with the batched projection helpers and diffs the
    // reconstructed post-FFN hidden. It does not feed production output.
    use_qwen36_dense_prefill_validate: bool = false,
    dense_prefill_validate_layer: u32 = 0,
    dense_prefill_validate_max_tokens: u32 = 0,
    dense_prefill_validate_captured_tokens: u32 = 0,
    dense_prefill_validate_norm_ref: ?Buffer = null,
    dense_prefill_validate_pre_hidden_ref: ?Buffer = null,
    dense_prefill_validate_post_hidden_ref: ?Buffer = null,
    dense_prefill_validate_gate_ref: ?Buffer = null,
    dense_prefill_validate_up_ref: ?Buffer = null,
    dense_prefill_validate_swiglu_ref: ?Buffer = null,
    dense_prefill_validate_down_ref: ?Buffer = null,
    dense_prefill_validate_staging: ?Buffer = null,
    // Same flag also enables a selected SSM-layer validator: capture
    // norm/qkv/z/alpha/beta plus conv_out, per-token delta_out, gated norm,
    // and pre/post SSM hidden. After prefill it replays qkv/z, one batched
    // delta_net chunk, gated norm, and ssm_out, then diffs.
    use_qwen36_ssm_prefill_validate: bool = false,
    ssm_prefill_validate_captured_tokens: u32 = 0,
    ssm_prefill_validate_norm_ref: ?Buffer = null,
    ssm_prefill_validate_qkv_ref: ?Buffer = null,
    ssm_prefill_validate_z_ref: ?Buffer = null,
    ssm_prefill_validate_alpha_ref: ?Buffer = null,
    ssm_prefill_validate_beta_ref: ?Buffer = null,
    ssm_prefill_validate_conv_ref: ?Buffer = null,
    ssm_prefill_validate_delta_ref: ?Buffer = null,
    ssm_prefill_validate_delta_replay: ?Buffer = null,
    ssm_prefill_validate_gnorm_ref: ?Buffer = null,
    ssm_prefill_validate_pre_hidden_ref: ?Buffer = null,
    ssm_prefill_validate_post_hidden_ref: ?Buffer = null,
    ssm_prefill_validate_state_backup: ?Buffer = null,
    ssm_prefill_validate_staging: ?Buffer = null,

    // A3b production enablement: layer-major prefill needs to replay one
    // layer at a time over all prompt tokens while reading/writing each
    // token's hidden state from a strided scratch buffer. Defaults preserve
    // the normal full-token decodeStep path. Future A3b production code can
    // set these before calling decodeStep to execute [start, end) without
    // duplicating the attention/SSM/MoE layer bodies.
    partial_decode_start_layer: u32 = 0,
    partial_decode_end_layer: u32 = 0, // 0 means config.n_layers
    partial_decode_hidden_in: ?vk.c.VkBuffer = null,
    partial_decode_hidden_in_offset: vk.c.VkDeviceSize = 0,
    partial_decode_hidden_out: ?vk.c.VkBuffer = null,
    partial_decode_hidden_out_offset: vk.c.VkDeviceSize = 0,
    partial_decode_advance_position: bool = true,
    partial_decode_allow_final_tail: bool = false,
    partial_decode_stop_after_ffn_norm: bool = false,
    partial_decode_ffn_norm_out: ?vk.c.VkBuffer = null,
    partial_decode_ffn_norm_out_offset: vk.c.VkDeviceSize = 0,
    partial_decode_stop_after_ssm_gnorm: bool = false,
    partial_decode_ssm_gnorm_out: ?vk.c.VkBuffer = null,
    partial_decode_ssm_gnorm_out_offset: vk.c.VkDeviceSize = 0,
    partial_decode_stop_after_ssm_conv: bool = false,
    partial_decode_ssm_conv_out: ?vk.c.VkBuffer = null,
    partial_decode_ssm_conv_out_offset: vk.c.VkDeviceSize = 0,
    partial_decode_ssm_z_out: ?vk.c.VkBuffer = null,
    partial_decode_ssm_z_out_offset: vk.c.VkDeviceSize = 0,
    partial_decode_ssm_alpha_out: ?vk.c.VkBuffer = null,
    partial_decode_ssm_alpha_out_offset: vk.c.VkDeviceSize = 0,
    partial_decode_ssm_beta_out: ?vk.c.VkBuffer = null,
    partial_decode_ssm_beta_out_offset: vk.c.VkDeviceSize = 0,
    partial_ssm_preproj_layer: u32 = std.math.maxInt(u32),
    partial_ssm_preproj_token_idx: u32 = 0,
    partial_ssm_preproj_qkv: ?vk.c.VkBuffer = null,
    partial_ssm_preproj_qkv_size: vk.c.VkDeviceSize = 0,
    partial_ssm_preproj_qkv_stride: vk.c.VkDeviceSize = 0,
    partial_ssm_preproj_z: ?vk.c.VkBuffer = null,
    partial_ssm_preproj_z_size: vk.c.VkDeviceSize = 0,
    partial_ssm_preproj_z_stride: vk.c.VkDeviceSize = 0,

    // Scratch buffers for the Vulkan/RDNA batched prefill path (lazy-init,
    // reused across prefill calls). Sized to hold all N prompt tokens at
    // once so dmmv_q4k_batch + rope_batched + flash_attn_batched can each
    // run once per layer instead of per-token. Grown on demand to the
    // largest prompt seen.
    batched_scratch_hidden: ?Buffer = null,
    batched_scratch_norm: ?Buffer = null,
    batched_scratch_q: ?Buffer = null,
    batched_scratch_k: ?Buffer = null,
    batched_scratch_v: ?Buffer = null,
    batched_scratch_attn_out: ?Buffer = null,
    batched_scratch_gate: ?Buffer = null,
    batched_scratch_up: ?Buffer = null,
    batched_scratch_swiglu: ?Buffer = null,
    batched_scratch_down: ?Buffer = null,
    batched_scratch_capacity_tokens: u32 = 0,
    modeled_decode_bytes_per_token: u64 = 0,
    // Diagnostic summary stored during BOS processing, printed after generation
    /// GPU buffer for diag summary buf.
    diag_summary_buf: [2048]u8 = .{0} ** 2048,
    diag_summary_len: usize = 0,

    /// Create the runtime objects needed to execute decode-time work on the GPU.
    /// @param model Loaded model weights and metadata.
    /// @param instance Active Vulkan instance and logical device.
    /// @param gpu_config Derived GPU tuning parameters for the selected device.
    /// @param shader_dir Directory containing compiled SPIR-V shader binaries.
    /// @param allocator Allocator used for graphs, staging state, and temporary setup structures.
    /// @returns An initialized inference engine ready to prefill prompts and run decode steps.
    /// @note This allocates shared descriptor pools, staging buffers, intermediate activations, and dispatch wrappers up front.
    pub fn init(
        io: std.Io,
        /// Loaded model.
        model: *Model,
        /// Vulkan instance.
        instance: *const Instance,
        /// GPU capabilities.
        gpu_config: GpuConfig,
        shader_dir: []const u8,
        /// Allocator for owned resources.
        allocator: std.mem.Allocator,
    ) !InferenceEngine {
        const config = &model.config;

        var cmd_pool = try CommandPool.init(instance);
        errdefer cmd_pool.deinit();

        var decode_cmd = try CommandBuffer.init(instance, &cmd_pool);
        errdefer decode_cmd.deinit(&cmd_pool);
        var prefill_cmd_alt = try CommandBuffer.init(instance, &cmd_pool);
        errdefer prefill_cmd_alt.deinit(&cmd_pool);

        // max_k: largest K (input dimension) used in any Q4_K DMMV dispatch.
        // Needed to size the Q4_K shared memory array s_x[SPEC_K].
        const q_dim_val = @as(u32, config.n_heads) * config.head_dim;
        const inter_val = if (config.intermediate_dim > 0) config.intermediate_dim else config.hidden_dim * 4;
        const shexp_val = if (config.shared_expert_intermediate_dim > 0) config.shared_expert_intermediate_dim else inter_val;
        const d_inner_val = if (config.ssm_d_inner > 0) config.ssm_d_inner else config.hidden_dim;
        const max_k = @max(@max(@max(config.hidden_dim, inter_val), @max(q_dim_val, d_inner_val)), shexp_val);
        var dmmv = try DmmvDispatch.init(io, instance, &gpu_config, shader_dir, max_k, allocator);
        errdefer dmmv.deinit();

        var elementwise = try ElementwiseDispatch.init(io, instance, shader_dir, allocator);
        errdefer elementwise.deinit();

        var attention = try AttentionDispatch.init(io, instance, shader_dir, allocator);
        errdefer attention.deinit();
        var argmax = try ArgmaxDispatch.init(io, instance, shader_dir, allocator);
        errdefer argmax.deinit();

        const weights_bytes = tensorBytes(model);
        const runtime_profile = memory_plan.profile(config.*);
        const requested_ctx = config.context_length;
        const max_ctx = runtime_profile.maxContextTokensForDeviceLocalBudget(
            weights_bytes,
            instance.vramBytes(),
            requested_ctx,
        );
        if (max_ctx == 0) {
            log.err("No decode context fits within {d:.2} GiB VRAM budget", .{
                @as(f64, @floatFromInt(instance.vramBytes())) / (1024.0 * 1024.0 * 1024.0),
            });
            return error.ContextLengthDoesNotFit;
        }
        if (max_ctx < requested_ctx) {
            log.warn("Context trimmed from {d} to {d} tokens to fit current VRAM budget", .{
                requested_ctx,
                max_ctx,
            });
        } else {
            log.info("KV cache planned context: requested {d}, reserved {d}", .{
                requested_ctx,
                max_ctx,
            });
        }

        // Build the decode graph (for diagnostics / future full-graph dispatch)
        var graph_config = config.*;
        graph_config.context_length = max_ctx;
        var decode_graph = try architecture.buildDecodeGraphDetailed(&graph_config, allocator, &model.gguf_file);
        decode_graph.setHardwareContext(.{
            .bandwidth_gbps = gpu_config.bandwidth_gbps,
            .compute_units = gpu_config.compute_units,
            .wave_size = gpu_config.wave_size,
            .preferred_workgroup_size = gpu_config.dmmv_workgroup_size,
        });
        errdefer decode_graph.deinit();
        var decode_analysis = try decode_graph.analyze(allocator);
        defer decode_analysis.deinit();
        const modeled_decode_bytes_per_token = decode_analysis.total_bytes;

        // Allocate intermediate buffers
        const hidden_size = @as(vk.c.VkDeviceSize, config.hidden_dim) * @sizeOf(f32);
        // All intermediate buffers need TRANSFER_SRC|DST for debug readback and embedding upload
        const buf_usage = vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        var hidden_buf = try Buffer.initDeviceLocal(instance, hidden_size, buf_usage);
        errdefer hidden_buf.deinit();

        var residual_buf = try Buffer.initDeviceLocal(instance, hidden_size, buf_usage);
        errdefer residual_buf.deinit();

        var norm_buf = try Buffer.initDeviceLocal(instance, hidden_size, buf_usage);
        errdefer norm_buf.deinit();

        const q8_1_blocks = (config.hidden_dim + 31) / 32;
        const q8_1_size = @as(vk.c.VkDeviceSize, q8_1_blocks) * Q8_1_BLOCK_BYTES;
        var q8_1_buf = try Buffer.initDeviceLocal(instance, q8_1_size, buf_usage);
        errdefer q8_1_buf.deinit();

        const logits_size = @as(vk.c.VkDeviceSize, config.vocab_size) * @sizeOf(f32);
        var logits_buf = try Buffer.initDeviceLocal(instance, logits_size, vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        errdefer logits_buf.deinit();

        // Pre-allocate host-visible readback buffer for logits (avoids per-token vkAllocateMemory)
        var logits_staging = try Buffer.init(
            instance,
            logits_size,
            vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        errdefer logits_staging.deinit();
        {
            var map_ptr: ?*anyopaque = null;
            const map_result = vk.c.vkMapMemory(instance.device, logits_staging.memory, 0, logits_size, 0, &map_ptr);
            if (map_result != vk.c.VK_SUCCESS) return error.MapMemoryFailed;
            logits_staging.mapped = @ptrCast(map_ptr);
        }

        const argmax_phase0_workgroups: u32 = @max(@as(u32, 1), @min((config.vocab_size + 63) / 64, 512));
        const argmax_partials_size = @as(vk.c.VkDeviceSize, argmax_phase0_workgroups) * 2 * @sizeOf(u32);
        var argmax_partials_buf = try Buffer.initDeviceLocal(
            instance,
            argmax_partials_size,
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        );
        errdefer argmax_partials_buf.deinit();

        var argmax_result_buf = try Buffer.initDeviceLocal(
            instance,
            @sizeOf(u32),
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        );
        errdefer argmax_result_buf.deinit();

        var argmax_result_staging = try Buffer.init(
            instance,
            @sizeOf(u32),
            vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        errdefer argmax_result_staging.deinit();
        {
            var map_ptr: ?*anyopaque = null;
            const map_result = vk.c.vkMapMemory(instance.device, argmax_result_staging.memory, 0, @sizeOf(u32), 0, &map_ptr);
            if (map_result != vk.c.VK_SUCCESS) return error.MapMemoryFailed;
            argmax_result_staging.mapped = @ptrCast(map_ptr);
        }

        var argmax_descriptor_set: ?vk.c.VkDescriptorSet = null;
        if (argmax.pipeline != null) {
            const ds = try argmax.allocDescriptorSet();
            argmax.writeDescriptorSet(
                ds,
                logits_buf.handle,
                logits_buf.size,
                argmax_partials_buf.handle,
                argmax_partials_buf.size,
                argmax_result_buf.handle,
                argmax_result_buf.size,
            );
            argmax_descriptor_set = ds;
        }

        // Pre-allocate upload staging for embeddings (avoids per-token vkAllocateMemory)
        var embed_staging = try Buffer.initStaging(instance, hidden_size);
        errdefer embed_staging.deinit();
        var prefill_embed_alt = try Buffer.initStaging(instance, hidden_size);
        errdefer prefill_embed_alt.deinit();

        // Transformer layer intermediate buffers
        const q_dim = @as(u32, config.n_heads) * config.head_dim;
        const kv_dim = @as(u32, config.n_kv_heads) * config.head_dim;
        const q_size = @as(vk.c.VkDeviceSize, q_dim) * @sizeOf(f32);
        const kv_size = @as(vk.c.VkDeviceSize, kv_dim) * @sizeOf(f32);
        const inter_dim = if (config.intermediate_dim > 0) config.intermediate_dim else config.hidden_dim * 4;
        // SSM d_inner or shared expert FFN may be larger than per-expert intermediate_dim; buffers must fit all
        const shexp_inter = if (config.shared_expert_intermediate_dim > 0) config.shared_expert_intermediate_dim else inter_dim;
        // GPU SSM conv1d output is conv_channels = d_inner + 2*n_group*d_state, which exceeds d_inner
        const ssm_conv_channels: u32 = if (config.ssm_d_inner > 0) config.ssm_d_inner + 2 * config.ssm_n_group * config.ssm_d_state else 0;
        const max_inter = @max(@max(inter_dim, shexp_inter), @max(if (config.ssm_d_inner > 0) config.ssm_d_inner else inter_dim, ssm_conv_channels));
        const inter_size = @as(vk.c.VkDeviceSize, max_inter) * @sizeOf(f32);
        const n_experts_total = @max(if (config.n_experts > 0) config.n_experts else @as(u32, 1), config.ssm_dt_rank);
        const n_experts_used: u32 = if (config.n_experts_used > 0) config.n_experts_used else 8;

        // Batched MoE: gate/up/swiglu buffers must fit n_experts_used * inter_dim,
        // down_buf must fit n_experts_used * hidden_dim (all experts processed in parallel).
        const batched_inter_size = @as(vk.c.VkDeviceSize, n_experts_used) * @as(vk.c.VkDeviceSize, inter_dim) * @sizeOf(f32);
        const batched_down_size = @as(vk.c.VkDeviceSize, n_experts_used) * hidden_size;
        const gate_buf_size = @max(inter_size, batched_inter_size);
        const down_buf_size = @max(hidden_size, batched_down_size);

        const storage_xfer = vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        var q_buf = try Buffer.initDeviceLocal(instance, q_size, storage_xfer);
        errdefer q_buf.deinit();
        var k_buf = try Buffer.initDeviceLocal(instance, kv_size, storage_xfer);
        errdefer k_buf.deinit();
        var v_buf = try Buffer.initDeviceLocal(instance, kv_size, storage_xfer);
        errdefer v_buf.deinit();
        // attn_out_buf: needs max(q_full_size, conv_channels*4)
        // q_full_size = q_dim * 2 because attn_q.weight outputs interleaved [Q, gate] per head
        const q_full_size = @as(vk.c.VkDeviceSize, q_dim * 2) * @sizeOf(f32);
        const conv_ch = if (config.ssm_d_inner > 0) config.ssm_d_inner + 2 * config.ssm_n_group * config.ssm_d_state else 0;
        const attn_out_size = @max(q_full_size, @as(vk.c.VkDeviceSize, conv_ch) * @sizeOf(f32));
        var attn_out_buf = try Buffer.initDeviceLocal(instance, attn_out_size, storage_xfer);
        errdefer attn_out_buf.deinit();
        var o_proj_buf = try Buffer.initDeviceLocal(instance, hidden_size, storage_xfer);
        errdefer o_proj_buf.deinit();
        var ffn_norm_buf = try Buffer.initDeviceLocal(instance, hidden_size, storage_xfer);
        errdefer ffn_norm_buf.deinit();
        var gate_buf = try Buffer.initDeviceLocal(instance, gate_buf_size, storage_xfer);
        errdefer gate_buf.deinit();
        var up_buf = try Buffer.initDeviceLocal(instance, gate_buf_size, storage_xfer);
        errdefer up_buf.deinit();
        var swiglu_buf = try Buffer.initDeviceLocal(instance, gate_buf_size, storage_xfer);
        errdefer swiglu_buf.deinit();
        var down_buf = try Buffer.initDeviceLocal(instance, down_buf_size, storage_xfer);
        errdefer down_buf.deinit();
        var moe_out_buf = try Buffer.initDeviceLocal(instance, hidden_size, storage_xfer);
        errdefer moe_out_buf.deinit();

        const router_size = @as(vk.c.VkDeviceSize, n_experts_total) * @sizeOf(f32);
        var router_logits_buf = try Buffer.initDeviceLocal(instance, router_size, storage_xfer);
        errdefer router_logits_buf.deinit();
        var router_staging = try Buffer.init(
            instance,
            router_size,
            vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        errdefer router_staging.deinit();
        {
            var map_ptr: ?*anyopaque = null;
            const mr = vk.c.vkMapMemory(instance.device, router_staging.memory, 0, router_size, 0, &map_ptr);
            if (mr != vk.c.VK_SUCCESS) return error.MapMemoryFailed;
            router_staging.mapped = @ptrCast(map_ptr);
        }

        // IMROPE frequency buffer: precompute per-pair inverse frequencies for sectioned RoPE
        const rope_dim_val: u32 = if (config.rope_dim > 0) config.rope_dim else config.head_dim;
        const half_rot = rope_dim_val / 2;
        const rope_freq_size = @as(vk.c.VkDeviceSize, half_rot) * @sizeOf(f32);
        var rope_freq_buf = try Buffer.init(
            instance,
            @max(rope_freq_size, 4),
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        errdefer rope_freq_buf.deinit();
        {
            var map_ptr: ?*anyopaque = null;
            const mr = vk.c.vkMapMemory(instance.device, rope_freq_buf.memory, 0, @max(rope_freq_size, 4), 0, &map_ptr);
            if (mr != vk.c.VK_SUCCESS) return error.MapMemoryFailed;
            rope_freq_buf.mapped = @ptrCast(map_ptr);
        }
        // Precompute inverse RoPE frequencies.
        // For IMROPE: sectioned per-pair frequencies.
        // For Gemma 4 global attention: rope_freqs.weight factors modify base frequencies.
        // For standard models: freq[k] = 1 / base^(2k / rope_dim).
        const has_imrope = config.rope_sections[0] > 0 or config.rope_sections[1] > 0;
        {
            const freq_ptr: [*]f32 = @ptrCast(@alignCast(rope_freq_buf.mapped.?));
            if (has_imrope) {
                const total_pairs = config.rope_sections[0] + config.rope_sections[1] +
                    config.rope_sections[2] + config.rope_sections[3];
                const rope_full_dim: f32 = @floatFromInt(2 * total_pairs);
                for (0..total_pairs) |k| {
                    const exponent = @as(f32, @floatFromInt(2 * k)) / rope_full_dim;
                    freq_ptr[k] = 1.0 / std.math.pow(f32, config.rope_freq_base, exponent);
                }
                log.info("IMROPE: sections=[{d},{d},{d},{d}] total_pairs={d} freq[0]={d:.6} freq[11]={d:.6} freq[31]={d:.6}", .{
                    config.rope_sections[0], config.rope_sections[1], config.rope_sections[2],                     config.rope_sections[3],
                    total_pairs,             freq_ptr[0],             if (total_pairs > 11) freq_ptr[11] else 0.0, if (total_pairs > 31) freq_ptr[31] else 0.0,
                });
            } else {
                // Standard NeoX RoPE: freq[k] = 1 / base^(2k / rope_dim)
                const rope_full_dim_f: f32 = @floatFromInt(rope_dim_val);
                for (0..half_rot) |k| {
                    const exponent = @as(f32, @floatFromInt(2 * k)) / rope_full_dim_f;
                    freq_ptr[k] = 1.0 / std.math.pow(f32, config.rope_freq_base, exponent);
                }
                // Apply rope_freqs.weight factors if present (Gemma 4 proportional RoPE)
                if (model.mmap_data) |mmap| {
                    for (model.gguf_file.tensors.items) |ti| {
                        if (std.mem.eql(u8, ti.name, "rope_freqs.weight")) {
                            const off = model.gguf_file.tensor_data_offset + ti.offset;
                            const n_factors = @min(ti.numElements(), half_rot);
                            for (0..@intCast(n_factors)) |k| {
                                const factor_off = off + k * @sizeOf(f32);
                                if (factor_off + @sizeOf(f32) <= mmap.len) {
                                    const factor: f32 = @as(*const f32, @ptrCast(@alignCast(mmap.ptr + factor_off))).*;
                                    if (factor != 0.0) freq_ptr[k] /= factor;
                                }
                            }
                            log.info("RoPE freq factors loaded from rope_freqs.weight ({d} entries)", .{n_factors});
                            break;
                        }
                    }
                }

                // YaRN RoPE scaling for extended-context models like GPT-OSS.
                // Keep this in sync with ggml_rope_yarn_corr_dims + rope_yarn().
                if (hasYarnScaling(config)) {
                    const factor = config.rope_scaling_factor;
                    const freq_scale: f32 = 1.0 / factor;
                    const beta_fast: f32 = 32.0;
                    const beta_slow: f32 = 1.0;
                    const corr_low = @max(@as(f32, 0.0), @floor(ropeYarnCorrDim(rope_dim_val, config.rope_original_context, beta_fast, config.rope_freq_base)));
                    const corr_high = @min(@as(f32, @floatFromInt(rope_dim_val - 1)), @ceil(ropeYarnCorrDim(rope_dim_val, config.rope_original_context, beta_slow, config.rope_freq_base)));
                    for (0..half_rot) |k| {
                        const ramp_mix = ropeYarnRamp(corr_low, corr_high, k);
                        freq_ptr[k] *= freq_scale * (1.0 - ramp_mix) + ramp_mix;
                    }
                    log.info("RoPE: applied YaRN scaling factor={d:.1} orig_ctx={d} corr=[{d:.2},{d:.2}] attn_scale={d:.4}", .{
                        factor,
                        config.rope_original_context,
                        corr_low,
                        corr_high,
                        effectiveRopeAttnScale(config),
                    });
                }
            }
        }

        // Unit-weights RMS norm buffer: hidden_dim entries all 1.0 (Gemma 4 V plain RMS norm)
        const unit_norm_size = @as(vk.c.VkDeviceSize, config.hidden_dim) * @sizeOf(f32);
        var unit_norm_weights = try Buffer.init(
            instance,
            unit_norm_size,
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        errdefer unit_norm_weights.deinit();
        {
            var map_ptr: ?*anyopaque = null;
            const mr = vk.c.vkMapMemory(instance.device, unit_norm_weights.memory, 0, unit_norm_size, 0, &map_ptr);
            if (mr != vk.c.VK_SUCCESS) return error.MapMemoryFailed;
            unit_norm_weights.mapped = @ptrCast(map_ptr);
            const ptr: [*]f32 = @ptrCast(@alignCast(map_ptr.?));
            for (0..config.hidden_dim) |i| ptr[i] = 1.0;
        }

        // Pre-populated per-layer attention sinks: size = n_layers × n_heads × f32.
        // Populated once after layer_tensors is resolved (see below); flash_attn reads
        // with sink_offset = layer_idx * n_heads. Eliminates per-token CPU memset+read
        // that previously ran for every attention-layer dispatch (cycle 8).
        const attn_sinks_total_floats: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, config.n_layers) * @as(vk.c.VkDeviceSize, config.n_heads);
        const attn_sinks_size = @max(attn_sinks_total_floats * @sizeOf(f32), 4);
        var attn_sinks_buf = try Buffer.init(
            instance,
            attn_sinks_size,
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            vk.c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        errdefer attn_sinks_buf.deinit();
        {
            var map_ptr: ?*anyopaque = null;
            const mr = vk.c.vkMapMemory(instance.device, attn_sinks_buf.memory, 0, attn_sinks_size, 0, &map_ptr);
            if (mr != vk.c.VK_SUCCESS) return error.MapMemoryFailed;
            attn_sinks_buf.mapped = @ptrCast(map_ptr);
            const ptr: [*]f32 = @ptrCast(@alignCast(map_ptr.?));
            for (0..@intCast(attn_sinks_total_floats)) |i| ptr[i] = std.math.nan(f32);
        }

        // KV cache: per-layer, flat layout (context_length * kv_dim * sizeof(f32))
        const kv_cache_per_layer = @as(vk.c.VkDeviceSize, max_ctx) * @as(vk.c.VkDeviceSize, kv_dim) * @sizeOf(f32);
        const kv_k_cache = try allocator.alloc(Buffer, config.n_layers);
        errdefer allocator.free(kv_k_cache);
        const kv_v_cache = try allocator.alloc(Buffer, config.n_layers);
        errdefer allocator.free(kv_v_cache);

        for (0..config.n_layers) |i| {
            kv_k_cache[i] = try Buffer.initDeviceLocal(instance, kv_cache_per_layer, storage_xfer);
            kv_v_cache[i] = try Buffer.initDeviceLocal(instance, kv_cache_per_layer, storage_xfer);
        }

        log.debug("KV cache: {d} layers × {d} MB = {d} MB total", .{
            config.n_layers,
            kv_cache_per_layer * 2 / (1024 * 1024),
            config.n_layers * kv_cache_per_layer * 2 / (1024 * 1024),
        });

        // Active page table for flash attention, backed by a request-owned page pool.
        const kv_page_count = kvPageCountForContext(max_ctx);
        const page_table_size = @as(vk.c.VkDeviceSize, kv_page_count) * @sizeOf(u32);
        var page_table_buf = try Buffer.initDeviceLocal(
            instance,
            page_table_size,
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        );
        errdefer page_table_buf.deinit();
        var page_table_staging = try Buffer.initStaging(instance, page_table_size);
        errdefer page_table_staging.deinit();
        const pt_u32: [*]u32 = @ptrCast(@alignCast(page_table_staging.mapped.?));
        @memset(pt_u32[0..kv_page_count], 0);
        try buffer_mod.copyBuffer(instance, cmd_pool.handle, &page_table_staging, &page_table_buf, page_table_size);
        var kv_page_pool = try kv_cache_mod.KvPagePool.init(allocator, kv_page_count, kv_page_size_tokens);
        errdefer kv_page_pool.deinit();

        // SSM state (CPU-side, for hybrid models)
        const ssm_conv_states = try allocator.alloc([]f32, config.n_layers);
        const ssm_states = try allocator.alloc([]f32, config.n_layers);
        const has_ssm = config.ssm_d_inner > 0;
        if (has_ssm) {
            const d_inner = config.ssm_d_inner;
            const dt_rank_v = config.ssm_dt_rank;
            const head_v_dim_v = d_inner / dt_rank_v;
            const conv_channels = d_inner + 2 * config.ssm_n_group * config.ssm_d_state;
            const conv_state_size = (config.ssm_d_conv - 1) * conv_channels;
            const ssm_state_size = head_v_dim_v * head_v_dim_v * dt_rank_v;

            for (0..config.n_layers) |i| {
                ssm_conv_states[i] = try allocator.alloc(f32, conv_state_size);
                @memset(ssm_conv_states[i], 0);
                ssm_states[i] = try allocator.alloc(f32, ssm_state_size);
                @memset(ssm_states[i], 0);
            }
            log.debug("SSM state: {d} layers × {d} KB conv + {d} KB recurrent", .{
                config.n_layers,
                conv_state_size * 4 / 1024,
                ssm_state_size * 4 / 1024,
            });
        } else {
            for (0..config.n_layers) |i| {
                ssm_conv_states[i] = &.{};
                ssm_states[i] = &.{};
            }
        }

        // SSM hidden state staging buffer (for GPU↔CPU transfers)
        // Size for d_inner (SSM output) which may be larger than hidden_dim
        const ssm_staging_size = @max(hidden_size, @as(vk.c.VkDeviceSize, if (config.ssm_d_inner > 0) config.ssm_d_inner else config.hidden_dim) * @sizeOf(f32));
        var ssm_hidden_staging = try Buffer.init(
            instance,
            ssm_staging_size,
            vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        errdefer ssm_hidden_staging.deinit();
        {
            var map_ptr: ?*anyopaque = null;
            const mr2 = vk.c.vkMapMemory(instance.device, ssm_hidden_staging.memory, 0, ssm_staging_size, 0, &map_ptr);
            if (mr2 != vk.c.VK_SUCCESS) return error.MapMemoryFailed;
            ssm_hidden_staging.mapped = @ptrCast(map_ptr);
        }

        // GPU-side SSM state buffers (persistent across tokens, for Phase 3c decode perf)
        const gpu_ssm_conv_states = try allocator.alloc(Buffer, config.n_layers);
        errdefer allocator.free(gpu_ssm_conv_states);
        const gpu_ssm_states = try allocator.alloc(Buffer, config.n_layers);
        errdefer allocator.free(gpu_ssm_states);
        const ssm_conv_state_offsets = try allocator.alloc(u32, config.n_layers);
        errdefer allocator.free(ssm_conv_state_offsets);
        @memset(ssm_conv_state_offsets, 0);
        if (has_ssm) {
            const d_inner_g = config.ssm_d_inner;
            const dt_rank_g = config.ssm_dt_rank;
            const head_v_dim_g = d_inner_g / dt_rank_g;
            const gpu_conv_ch = d_inner_g + 2 * config.ssm_n_group * config.ssm_d_state;
            const gpu_conv_size = @as(vk.c.VkDeviceSize, (config.ssm_d_conv - 1) * gpu_conv_ch) * @sizeOf(f32);
            const gpu_state_size = @as(vk.c.VkDeviceSize, dt_rank_g * head_v_dim_g * head_v_dim_g) * @sizeOf(f32);
            for (0..config.n_layers) |i| {
                gpu_ssm_conv_states[i] = try Buffer.initDeviceLocal(instance, gpu_conv_size, vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT);
                gpu_ssm_states[i] = try Buffer.initDeviceLocal(instance, gpu_state_size, vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            }
            // Zero-fill GPU SSM buffers via vkCmdFillBuffer
            try decode_cmd.reset();
            try decode_cmd.begin();
            for (0..config.n_layers) |i| {
                vk.c.vkCmdFillBuffer(decode_cmd.handle, gpu_ssm_conv_states[i].handle, 0, gpu_conv_size, 0);
                vk.c.vkCmdFillBuffer(decode_cmd.handle, gpu_ssm_states[i].handle, 0, gpu_state_size, 0);
            }
            try decode_cmd.end();
            try decode_cmd.submitAndWait(instance.compute_queue);
            log.debug("GPU SSM state: {d} layers × {d} KB conv + {d} KB recurrent = {d} MB total", .{
                config.n_layers,
                gpu_conv_size / 1024,
                gpu_state_size / 1024,
                (gpu_conv_size + gpu_state_size) * config.n_layers / (1024 * 1024),
            });
        } else {
            for (0..config.n_layers) |i| {
                gpu_ssm_conv_states[i] = .{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = instance.device };
                gpu_ssm_states[i] = .{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = instance.device };
            }
        }

        // GPU router output buffer stays device-local on the fast path because the
        // following MoE kernels consume it directly on-GPU every decode step.
        const n_used_experts = if (config.n_experts_used > 0) config.n_experts_used else 8;
        const router_out_size = @as(vk.c.VkDeviceSize, n_used_experts) * (@sizeOf(u32) + @sizeOf(f32));
        var router_output_buf = try Buffer.initDeviceLocal(
            instance,
            router_out_size,
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        );
        errdefer router_output_buf.deinit();

        // Descriptor pool: need many sets for all layers + MoE experts
        // Per layer: ~15 descriptor sets; MoE adds ~32 per layer (8 experts × 4 ops)
        // Total: 40 layers × 47 ≈ 2000 sets, each up to 5 bindings
        const pool_sizes = [_]vk.c.VkDescriptorPoolSize{.{
            .type = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 16384,
        }};
        const pool_info = vk.c.VkDescriptorPoolCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .maxSets = 4096,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_sizes,
        };
        var shared_pool: vk.c.VkDescriptorPool = null;
        if (instance.push_descriptor_fn == null) {
            const pool_result = vk.c.vkCreateDescriptorPool(instance.device, &pool_info, null, &shared_pool);
            if (pool_result != vk.c.VK_SUCCESS) return error.DescriptorPoolCreateFailed;
        }
        errdefer vk.c.vkDestroyDescriptorPool(instance.device, shared_pool, null);

        // Build tensor name → pointer hash map for O(1) lookup (replaces O(n) linear scan
        // in findLayerTensor, called ~960 times per token across 64 layers).
        var tensor_map = std.StringHashMap(*const LoadedTensor).init(allocator);
        errdefer tensor_map.deinit();
        try tensor_map.ensureTotalCapacity(@intCast(model.tensors.items.len));
        for (model.tensors.items) |*t| {
            tensor_map.putAssumeCapacity(t.info.name, t);
        }

        // Pre-resolve per-layer tensor pointers to eliminate ~960 hash lookups per token.
        const layer_tensors = try allocator.alloc(LayerTensors, config.n_layers);
        errdefer allocator.free(layer_tensors);
        for (0..config.n_layers) |li| {
            var lt = LayerTensors{};
            const l: u32 = @intCast(li);
            const resolve = struct {
                fn f(map: std.StringHashMap(*const LoadedTensor), layer: u32, name: []const u8) ?*const LoadedTensor {
                    var buf: [128]u8 = undefined;
                    const key = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ layer, name }) catch return null;
                    return map.get(key);
                }
            }.f;
            lt.attn_norm = resolve(tensor_map, l, "attn_norm.weight");
            lt.attn_q = resolve(tensor_map, l, "attn_q.weight");
            lt.attn_k = resolve(tensor_map, l, "attn_k.weight");
            lt.attn_v = resolve(tensor_map, l, "attn_v.weight");
            lt.attn_q_bias = resolve(tensor_map, l, "attn_q.bias");
            lt.attn_k_bias = resolve(tensor_map, l, "attn_k.bias");
            lt.attn_v_bias = resolve(tensor_map, l, "attn_v.bias");
            lt.attn_output = resolve(tensor_map, l, "attn_output.weight");
            lt.attn_output_bias = resolve(tensor_map, l, "attn_output.bias");
            lt.attn_sinks = resolve(tensor_map, l, "attn_sinks.weight");
            lt.attn_gate = resolve(tensor_map, l, "attn_gate.weight");
            lt.attn_q_norm = resolve(tensor_map, l, "attn_q_norm.weight");
            lt.attn_k_norm = resolve(tensor_map, l, "attn_k_norm.weight");
            lt.post_attention_norm = resolve(tensor_map, l, "post_attention_norm.weight");
            lt.ffn_norm = resolve(tensor_map, l, "ffn_norm.weight");
            lt.ffn_gate = resolve(tensor_map, l, "ffn_gate.weight");
            lt.ffn_up = resolve(tensor_map, l, "ffn_up.weight");
            lt.ffn_down = resolve(tensor_map, l, "ffn_down.weight");
            lt.post_ffw_norm = resolve(tensor_map, l, "post_ffw_norm.weight");
            lt.pre_ffw_norm_2 = resolve(tensor_map, l, "pre_ffw_norm_2.weight");
            lt.post_ffw_norm_2 = resolve(tensor_map, l, "post_ffw_norm_2.weight");
            lt.post_ffw_norm_1 = resolve(tensor_map, l, "post_ffw_norm_1.weight");
            lt.ffn_gate_inp_scale = resolve(tensor_map, l, "ffn_gate_inp.scale");
            lt.ffn_down_exps_scale = resolve(tensor_map, l, "ffn_down_exps.scale");
            lt.ffn_gate_inp = resolve(tensor_map, l, "ffn_gate_inp.weight");
            lt.ffn_gate_inp_bias = resolve(tensor_map, l, "ffn_gate_inp.bias");
            lt.ffn_gate_exps = resolve(tensor_map, l, "ffn_gate_exps.weight");
            lt.ffn_gate_exps_bias = resolve(tensor_map, l, "ffn_gate_exps.bias");
            lt.ffn_up_exps = resolve(tensor_map, l, "ffn_up_exps.weight");
            lt.ffn_up_exps_bias = resolve(tensor_map, l, "ffn_up_exps.bias");
            lt.ffn_gate_up_exps = resolve(tensor_map, l, "ffn_gate_up_exps.weight");
            lt.ffn_down_exps = resolve(tensor_map, l, "ffn_down_exps.weight");
            lt.ffn_down_exps_bias = resolve(tensor_map, l, "ffn_down_exps.bias");
            // Gemma 4 MoE: shared expert uses ffn_gate/up/down when ffn_gate_shexp is absent
            const is_gemma_moe = config.architecture == .gemma and lt.ffn_gate_up_exps != null;
            lt.ffn_gate_shexp = resolve(tensor_map, l, "ffn_gate_shexp.weight") orelse
                if (is_gemma_moe) resolve(tensor_map, l, "ffn_gate.weight") else null;
            lt.ffn_up_shexp = resolve(tensor_map, l, "ffn_up_shexp.weight") orelse
                if (is_gemma_moe) resolve(tensor_map, l, "ffn_up.weight") else null;
            lt.ffn_down_shexp = resolve(tensor_map, l, "ffn_down_shexp.weight") orelse
                if (is_gemma_moe) resolve(tensor_map, l, "ffn_down.weight") else null;
            lt.ffn_gate_inp_shexp = resolve(tensor_map, l, "ffn_gate_inp_shexp.weight");
            lt.attn_qkv = resolve(tensor_map, l, "attn_qkv.weight");
            lt.ssm_alpha = resolve(tensor_map, l, "ssm_alpha.weight");
            lt.ssm_beta = resolve(tensor_map, l, "ssm_beta.weight");
            lt.ssm_conv1d = resolve(tensor_map, l, "ssm_conv1d.weight");
            lt.ssm_out = resolve(tensor_map, l, "ssm_out.weight");
            lt.ssm_dt_bias = resolve(tensor_map, l, "ssm_dt.bias");
            lt.ssm_a = resolve(tensor_map, l, "ssm_a");
            lt.ssm_norm = resolve(tensor_map, l, "ssm_norm.weight");
            layer_tensors[li] = lt;
        }

        // Pre-populate per-layer attn_sinks into attn_sinks_buf (cycle 8):
        // each layer's slot of n_heads floats is either NaN (no sinks tensor) or the
        // learned sinks read from mmap. flash_attn reads with sink_offset = layer * n_heads.
        // Replaces the old per-token loadAttentionSinks memset+readMmap path.
        if (model.mmap_data) |mmap| {
            const sink_all_ptr: [*]f32 = @ptrCast(@alignCast(attn_sinks_buf.mapped.?));
            for (layer_tensors, 0..) |lt, li| {
                const sinks_tensor = lt.attn_sinks orelse continue;
                const slot = sink_all_ptr[li * @as(usize, config.n_heads) ..][0..@as(usize, config.n_heads)];
                const sink_count = @min(slot.len, @as(usize, @intCast(sinks_tensor.info.numElements())));
                if (sink_count == 0) continue;
                const base_off: usize = @intCast(model.gguf_file.tensor_data_offset + sinks_tensor.info.offset);
                readMmapFloats(mmap, base_off, sinks_tensor.info.type_, slot[0..sink_count]);
            }
        }

        // Load per-layer output scales (Gemma 4 proportional scaling)
        const layer_output_scales = try allocator.alloc(f32, config.n_layers);
        errdefer allocator.free(layer_output_scales);
        for (0..config.n_layers) |li| {
            const l: u32 = @intCast(li);
            var los_buf: [128]u8 = undefined;
            const los_key = std.fmt.bufPrint(&los_buf, "blk.{d}.layer_output_scale.weight", .{l}) catch unreachable;
            if (tensor_map.get(los_key)) |los_tensor| {
                // Read scalar f32 from GGUF mmap data (before GPU upload)
                if (model.mmap_data) |mmap| {
                    const off = model.gguf_file.tensor_data_offset + los_tensor.info.offset;
                    if (off + @sizeOf(f32) <= mmap.len) {
                        const ptr: *const f32 = @ptrCast(@alignCast(mmap.ptr + off));
                        layer_output_scales[li] = ptr.*;
                    } else {
                        layer_output_scales[li] = 1.0;
                    }
                } else {
                    layer_output_scales[li] = 1.0;
                }
            } else {
                layer_output_scales[li] = 1.0;
            }
        }

        log.debug("Inference engine ready — {d} graph nodes, hidden_dim={d}, vocab={d}, tensor_map={d}", .{
            decode_graph.nodeCount(), config.hidden_dim, config.vocab_size, tensor_map.count(),
        });

        // Always create the timestamp query pool so prefill GPU phase timing is
        // available without requiring --profile. Without it, effort-6 has no way
        // to see which GPU phase (attention/MoE/tail/...) owns prefill time.
        var timestamp_pool: vk.c.VkQueryPool = null;
        var timestamp_period_ns: f64 = 1.0;
        {
            const max_timestamps: u32 = 2048;
            const ts_pool_info = vk.c.VkQueryPoolCreateInfo{
                .sType = vk.c.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
                .pNext = null,
                .flags = 0,
                .queryType = vk.c.VK_QUERY_TYPE_TIMESTAMP,
                .queryCount = max_timestamps,
                .pipelineStatistics = 0,
            };
            const create_result = vk.c.vkCreateQueryPool(instance.device, &ts_pool_info, null, &timestamp_pool);
            if (create_result == vk.c.VK_SUCCESS) {
                timestamp_period_ns = @as(f64, instance.device_props.limits.timestampPeriod);
            } else {
                timestamp_pool = null;
                log.warn("Failed to create timestamp query pool ({d}); prefill GPU phase timing disabled", .{create_result});
            }
        }

        // Q4_K MoE K-parallel shader: default ON when the pipeline is loaded,
        // disabled by setting ZINC_MOE_KPAR=0. Measured on RDNA4 for the
        // Qwen3.6-35B flagship: gate_up 855.6 → 695.4 ms (−18.7%), prefill
        // tok/s 23.16 → 23.72 (+2.4%) with identical output tokens.
        const moe_kpar_env = getenv("ZINC_MOE_KPAR");
        const moe_kpar_explicitly_off = moe_kpar_env != null and std.mem.eql(u8, moe_kpar_env.?, "0");
        const moe_kpar_enabled = !moe_kpar_explicitly_off and dmmv.pipeline_q4k_moe_kpar != null;
        if (moe_kpar_enabled) {
            log.info("MoE Q4_K kpar variant ENABLED (default, set ZINC_MOE_KPAR=0 to disable)", .{});
        } else if (moe_kpar_explicitly_off) {
            log.info("MoE Q4_K kpar variant DISABLED via ZINC_MOE_KPAR=0", .{});
        }

        // Q4_K/Q6_K batched projection, K-parallel variant. Default ON when
        // the pipeline is loaded. The shaders merge cross-subgroup partials,
        // so this is valid on RDNA wave64 and Intel wave32/wave16 devices.
        // Disable via ZINC_Q4K_BATCH_KPAR=0 to run the serial shader.
        const q4k_batch_kpar_env = getenv("ZINC_Q4K_BATCH_KPAR");
        const q4k_batch_kpar_explicitly_off = q4k_batch_kpar_env != null and std.mem.eql(u8, q4k_batch_kpar_env.?, "0");
        const q4k_batch_kpar_enabled = !q4k_batch_kpar_explicitly_off and dmmv.pipeline_q4k_batch_kpar != null;
        if (q4k_batch_kpar_enabled) {
            log.info("Q4_K batched projection kpar variant ENABLED (default, set ZINC_Q4K_BATCH_KPAR=0 to disable)", .{});
        } else if (q4k_batch_kpar_explicitly_off) {
            log.info("Q4_K batched projection kpar variant DISABLED via ZINC_Q4K_BATCH_KPAR=0", .{});
        }

        // MoE fused gate+up (Q4_K): default OFF. Enable with
        // ZINC_MOE_FUSED_GATE_UP=1. Measured on Qwen3.6-35B-A3B (expert M=512,
        // K=2048) on RDNA4 R9700: 26.33 tok/s prefill vs 26.51 unfused
        // (median of 5, 0.7% regression). Register pressure from the dual
        // output (two running sums, two nibble-vec4 sets) outweighs the
        // halved dispatch count for this small shape. Kept opt-in because
        // larger expert intermediates (>= 1024) haven't been measured and
        // the shader is otherwise a proven drop-in for kpar.
        const moe_fused_gate_up_env = getenv("ZINC_MOE_FUSED_GATE_UP");
        const moe_fused_gate_up_forced_on = moe_fused_gate_up_env != null and std.mem.eql(u8, moe_fused_gate_up_env.?, "1");
        const moe_fused_gate_up_pipeline_loaded = dmmv.pipeline_q4k_fused_gate_up_moe != null or
            dmmv.pipeline_q4k_fused_gate_up_moe_spec8 != null;
        const moe_fused_gate_up_enabled = moe_fused_gate_up_forced_on and
            moe_kpar_enabled and moe_fused_gate_up_pipeline_loaded;
        if (moe_fused_gate_up_enabled) {
            log.info("MoE Q4_K fused gate+up ENABLED via ZINC_MOE_FUSED_GATE_UP=1", .{});
        }

        // Fused MoE gate+up+SwiGLU (Q4_K): default ON when the pipeline is
        // loaded. This is the profitable version of the older gate+up-only
        // experiment because it removes the separate MoE SwiGLU dispatch and
        // barrier. Disable with ZINC_MOE_FUSED_GATE_UP_SWIGLU=0 for A/B.
        const moe_fused_gate_up_swiglu_env = getenv("ZINC_MOE_FUSED_GATE_UP_SWIGLU");
        const moe_fused_gate_up_swiglu_explicitly_off = moe_fused_gate_up_swiglu_env != null and
            std.mem.eql(u8, moe_fused_gate_up_swiglu_env.?, "0");
        const moe_fused_gate_up_swiglu_pipeline_loaded = dmmv.pipeline_q4k_fused_gate_up_swiglu_moe != null or
            dmmv.pipeline_q4k_fused_gate_up_swiglu_moe_spec8 != null;
        const moe_fused_gate_up_swiglu_enabled = !moe_fused_gate_up_swiglu_explicitly_off and
            moe_kpar_enabled and moe_fused_gate_up_swiglu_pipeline_loaded;
        if (moe_fused_gate_up_swiglu_enabled) {
            log.info("MoE Q4_K fused gate+up+SwiGLU ENABLED (default, set ZINC_MOE_FUSED_GATE_UP_SWIGLU=0 to disable)", .{});
        } else if (moe_fused_gate_up_swiglu_explicitly_off) {
            log.info("MoE Q4_K fused gate+up+SwiGLU DISABLED via ZINC_MOE_FUSED_GATE_UP_SWIGLU=0", .{});
        }

        // Fused MoE down + weighted_acc (Q4_K + Q5_K). Default ON when
        // either pipeline is loaded, disabled by setting
        // ZINC_FUSE_MOE_DOWN_ACC=0. Targets the 0.52 ms moe_weighted_acc
        // dispatch in the Qwen 3.6 35B-A3B decode breakdown (~1.3% of
        // total decode budget). The Qwen 3.6 XL pack ships down weights
        // as Q5_K so we accept either pipeline being present here;
        // fused_pip_for_qt at the call site selects the right one and
        // falls back when neither is loaded for the current quant.
        const moe_fused_down_acc_env = getenv("ZINC_FUSE_MOE_DOWN_ACC");
        const moe_fused_down_acc_explicitly_off = moe_fused_down_acc_env != null and std.mem.eql(u8, moe_fused_down_acc_env.?, "0");
        const moe_fused_down_acc_enabled = !moe_fused_down_acc_explicitly_off and
            (dmmv.pipeline_q4k_moe_fused_down_acc != null or dmmv.pipeline_q5k_moe_fused_down_acc != null);
        if (moe_fused_down_acc_enabled) {
            log.info("MoE fused down+acc ENABLED (default, set ZINC_FUSE_MOE_DOWN_ACC=0 to disable; q4_k_pipe={} q5k_pipe={})", .{
                dmmv.pipeline_q4k_moe_fused_down_acc != null,
                dmmv.pipeline_q5k_moe_fused_down_acc != null,
            });
        } else if (moe_fused_down_acc_explicitly_off) {
            log.info("MoE fused down+acc DISABLED via ZINC_FUSE_MOE_DOWN_ACC=0", .{});
        }

        const qwen36_like_f32_ssm = blk: {
            if (config.architecture != .qwen2_moe) break :blk false;
            const alpha0 = layer_tensors[0].ssm_alpha orelse break :blk false;
            const beta0 = layer_tensors[0].ssm_beta orelse break :blk false;
            break :blk alpha0.info.type_ == .f32 and beta0.info.type_ == .f32;
        };
        const qwen36_topk_env = getenv("ZINC_QWEN36_MOE_TOPK");
        const qwen36_topk_default: u32 = 3;
        const qwen36_topk_limit: u32 = if (qwen36_like_f32_ssm) blk: {
            if (qwen36_topk_env) |raw| {
                const parsed = std.fmt.parseInt(u32, raw, 10) catch qwen36_topk_default;
                if (parsed == 0 or parsed >= config.n_experts_used) break :blk 0;
                break :blk @max(@as(u32, 1), parsed);
            }
            break :blk qwen36_topk_default;
        } else 0;
        if (qwen36_topk_limit > 0) {
            log.info("Qwen 3.6 MoE top-k capped at {d} (set ZINC_QWEN36_MOE_TOPK={d} to restore metadata top-k)", .{
                qwen36_topk_limit,
                config.n_experts_used,
            });
        } else if (qwen36_like_f32_ssm and qwen36_topk_env != null) {
            log.info("Qwen 3.6 MoE top-k cap disabled via ZINC_QWEN36_MOE_TOPK={s}", .{qwen36_topk_env.?});
        }
        const gemma_topk_env = getenv("ZINC_GEMMA_MOE_TOPK");
        const gemma_topk_default: u32 = if (config.architecture == .gemma and isIntelGpuVendor(gpu_config.vendor)) 4 else 0;
        const gemma_topk_limit: u32 = if (config.architecture == .gemma) blk: {
            const requested = if (gemma_topk_env) |raw|
                std.fmt.parseInt(u32, raw, 10) catch gemma_topk_default
            else
                gemma_topk_default;
            if (requested == 0 or requested >= config.n_experts_used) break :blk 0;
            break :blk @max(@as(u32, 1), requested);
        } else 0;
        if (gemma_topk_limit > 0) {
            log.info("Gemma MoE top-k capped at {d} (set ZINC_GEMMA_MOE_TOPK=0 to restore metadata top-k={d})", .{
                gemma_topk_limit,
                config.n_experts_used,
            });
        }
        const qwen36_prefill_topk_env = getenv("ZINC_QWEN36_MOE_PREFILL_TOPK");
        const qwen36_prefill_topk_default: u32 = 1;
        const qwen36_prefill_tail_topk_limit: u32 = if (qwen36_like_f32_ssm) blk: {
            if (qwen36_prefill_topk_env) |raw| {
                const parsed = std.fmt.parseInt(u32, raw, 10) catch qwen36_prefill_topk_default;
                if (parsed == 0 or parsed >= config.n_experts_used) break :blk 0;
                break :blk @max(@as(u32, 1), parsed);
            }
            // Respect an explicit global top-k override. By default, only
            // non-terminal prefill tokens take the more aggressive cap.
            if (qwen36_topk_env != null) break :blk 0;
            break :blk qwen36_prefill_topk_default;
        } else 0;
        const qwen36_prefill_guard_env = getenv("ZINC_QWEN36_MOE_PREFILL_TOPK_GUARD");
        const qwen36_prefill_guard_default: u32 = 16;
        const qwen36_prefill_tail_topk_guard_tokens: u32 = if (qwen36_prefill_tail_topk_limit > 0) blk: {
            if (qwen36_prefill_guard_env) |raw| {
                break :blk std.fmt.parseInt(u32, raw, 10) catch qwen36_prefill_guard_default;
            }
            break :blk qwen36_prefill_guard_default;
        } else 0;
        if (qwen36_prefill_tail_topk_limit > 0) {
            log.info("Qwen 3.6 non-terminal prefill MoE top-k capped at {d} before final {d} prompt tokens (set ZINC_QWEN36_MOE_PREFILL_TOPK=0 to disable)", .{
                qwen36_prefill_tail_topk_limit,
                qwen36_prefill_tail_topk_guard_tokens,
            });
        } else if (qwen36_like_f32_ssm and qwen36_prefill_topk_env != null) {
            log.info("Qwen 3.6 non-terminal prefill MoE top-k cap disabled via ZINC_QWEN36_MOE_PREFILL_TOPK={s}", .{qwen36_prefill_topk_env.?});
        }

        // Fused FFN-RMS-norm + f32 router DMMV: default ON when the
        // rms_norm_dmmv_f32 pipeline is loaded. Folds the standalone
        // ffn-norm dispatch into the MoE router DMMV, saving one
        // dispatch + one barrier per MoE layer (~30 layers on Qwen 3.6
        // 35B-A3B). Disabled by setting ZINC_FUSED_RMS_ROUTER=0. Per-call
        // gates (architecture, weight type, etc.) are evaluated in the
        // forward path so models that don't fit silently fall back.
        const fused_rms_router_env = getenv("ZINC_FUSED_RMS_ROUTER");
        const fused_rms_router_explicitly_off = fused_rms_router_env != null and std.mem.eql(u8, fused_rms_router_env.?, "0");
        const fused_rms_router_enabled = !fused_rms_router_explicitly_off and
            elementwise.pipeline_rms_norm_dmmv_f32 != null;
        if (fused_rms_router_enabled) {
            log.info("Fused FFN-norm + router DMMV ENABLED (default, set ZINC_FUSED_RMS_ROUTER=0 to disable)", .{});
        } else if (fused_rms_router_explicitly_off) {
            log.info("Fused FFN-norm + router DMMV DISABLED via ZINC_FUSED_RMS_ROUTER=0", .{});
        }

        // Fused SSM pre-norm (cycle 13): merges (rms_norm_mul → alpha
        // DMMV → beta DMMV) into a single dispatch when alpha+beta are
        // f32 (Qwen 3.5/3.6 35B-A3B Q4_K_XL pack). Default-on whenever
        // the rms_norm_dmmv_q4k_alpha_beta pipeline is loaded; disable
        // via ZINC_FUSED_SSM_AB=0. Per-call gates (architecture, weight
        // type) are evaluated in the forward path so models that don't
        // fit silently fall back.
        const fused_ssm_ab_env = getenv("ZINC_FUSED_SSM_AB");
        const fused_ssm_ab_explicitly_off = fused_ssm_ab_env != null and std.mem.eql(u8, fused_ssm_ab_env.?, "0");
        const fused_ssm_ab_enabled = !fused_ssm_ab_explicitly_off and
            elementwise.pipeline_rms_norm_dmmv_q4k_alpha_beta != null and
            instance.push_descriptor_fn != null;
        if (fused_ssm_ab_enabled) {
            log.info("Fused SSM pre-norm (rms+alpha+beta) ENABLED (default, set ZINC_FUSED_SSM_AB=0 to disable)", .{});
        } else if (fused_ssm_ab_explicitly_off) {
            log.info("Fused SSM pre-norm DISABLED via ZINC_FUSED_SSM_AB=0", .{});
        }

        // SSM delta cols8: port of llama.cpp's GDN workgroup shape for
        // S=128 (8 output rows per wave64 via subgroupClusteredAdd). Disable
        // via ZINC_SSM_DELTA_COLS8=0 for A/B checks.
        const ssm_delta_cols8_env = getenv("ZINC_SSM_DELTA_COLS8");
        const ssm_delta_cols8_explicitly_off = ssm_delta_cols8_env != null and std.mem.eql(u8, ssm_delta_cols8_env.?, "0");
        const ssm_delta_cols8_enabled = !ssm_delta_cols8_explicitly_off and
            elementwise.pipeline_ssm_delta_net_cols8 != null;
        if (ssm_delta_cols8_enabled) {
            log.info("SSM delta cols8 ENABLED (default, set ZINC_SSM_DELTA_COLS8=0 to disable)", .{});
        } else if (ssm_delta_cols8_explicitly_off) {
            log.info("SSM delta cols8 DISABLED via ZINC_SSM_DELTA_COLS8=0", .{});
        }

        const ssm_delta_normed_qk_env = getenv("ZINC_SSM_DELTA_NORMED_QK");
        const ssm_delta_normed_qk_flag = ssm_delta_normed_qk_env != null and std.mem.eql(u8, ssm_delta_normed_qk_env.?, "1");
        const ssm_delta_normed_qk_enabled = ssm_delta_normed_qk_flag and
            ssm_delta_cols8_enabled and
            elementwise.pipeline_ssm_qk_norm != null and
            elementwise.pipeline_ssm_delta_net_cols8_normed != null and
            instance.push_descriptor_fn != null;
        if (ssm_delta_normed_qk_enabled) {
            log.info("SSM delta pre-normalized Q/K ENABLED via ZINC_SSM_DELTA_NORMED_QK=1", .{});
        } else if (ssm_delta_normed_qk_flag) {
            log.info("ZINC_SSM_DELTA_NORMED_QK=1 requested but prerequisites missing; using standard cols8 delta", .{});
        }

        // Fused dense gate+up+SwiGLU (effort-11 cycle 8). Default ON when
        // the pipeline is loaded; disable via ZINC_FUSED_DENSE_FFN=0. The
        // architecture / quant / size gates run per call so non-matching
        // models silently fall back to the gate / up / swiglu trio.
        const fused_dense_ffn_env = getenv("ZINC_FUSED_DENSE_FFN");
        const fused_dense_ffn_explicitly_off = fused_dense_ffn_env != null and std.mem.eql(u8, fused_dense_ffn_env.?, "0");
        const fused_dense_ffn_enabled = !fused_dense_ffn_explicitly_off and
            dmmv.pipeline_q4k_fused_gate_up_swiglu != null and
            instance.push_descriptor_fn != null;
        if (fused_dense_ffn_enabled) {
            log.info("Fused dense gate+up+SwiGLU ENABLED (default, set ZINC_FUSED_DENSE_FFN=0 to disable)", .{});
        } else if (fused_dense_ffn_explicitly_off) {
            log.info("Fused dense gate+up+SwiGLU DISABLED via ZINC_FUSED_DENSE_FFN=0", .{});
        }
        const qwen36_dense_row1_env = getenv("ZINC_QWEN36_27B_DENSE_FUSED_ROW1");
        const qwen36_dense_row1_explicitly_off = qwen36_dense_row1_env != null and
            std.mem.eql(u8, qwen36_dense_row1_env.?, "0");
        const qwen36_dense_row1_enabled = !qwen36_dense_row1_explicitly_off and
            dmmv.pipeline_q4k_fused_gate_up_swiglu_row1 != null and
            instance.push_descriptor_fn != null;
        if (qwen36_dense_row1_enabled) {
            log.info("Qwen3.6-27B dense fused gate+up+SwiGLU row1 path ENABLED (default for matching shape, set ZINC_QWEN36_27B_DENSE_FUSED_ROW1=0 to disable)", .{});
        } else if (qwen36_dense_row1_explicitly_off) {
            log.info("Qwen3.6-27B dense fused gate+up+SwiGLU row1 path DISABLED via ZINC_QWEN36_27B_DENSE_FUSED_ROW1=0", .{});
        }

        // Fused split-K merge + o_proj DMMV-acc (effort-11 cycle 17). When
        // ZINC_FUSED_OPROJ_MERGE=1 AND split-K is active, the o_proj site
        // calls dispatchDmmvOprojMerge instead of (separate merge dispatch +
        // dispatchDmmvAcc), saving 1 dispatch + 1 barrier per attention
        // layer. Default OFF — opt-in this cycle for safety; per-call
        // architecture/quant/size gates fall back to the unfused path when
        // the conditions aren't met (e.g., Gemma post_attn_norm,
        // hidden_dim > 4096, validation_diagnostics_enabled).
        const fused_oproj_merge_env = getenv("ZINC_FUSED_OPROJ_MERGE");
        const fused_oproj_merge_enabled = fused_oproj_merge_env != null and
            std.mem.eql(u8, fused_oproj_merge_env.?, "1") and
            dmmv.pipeline_q4k_o_proj_merge != null and
            instance.push_descriptor_fn != null;
        if (fused_oproj_merge_enabled) {
            log.info("Fused split-K merge + o_proj ENABLED via ZINC_FUSED_OPROJ_MERGE=1", .{});
        }

        // ZINC_FA_PROFILE_LAYER=1 — per-layer flash_attn_kernel timing histogram
        // (effort-11 run-3 enablement). Auto-enables timestamp recording so the
        // benchmark cycle (which does not pass --profile) still emits per-layer
        // ms data. Default OFF to keep the benchmark's hot path zero-overhead.
        const fa_profile_layer_env = getenv("ZINC_FA_PROFILE_LAYER");
        const fa_profile_layer_enabled = fa_profile_layer_env != null and
            !std.mem.eql(u8, fa_profile_layer_env.?, "0");
        if (fa_profile_layer_enabled) {
            log.info("Per-layer flash_attn timing ENABLED via ZINC_FA_PROFILE_LAYER=1 (auto-enables profile)", .{});
        }

        // Fused Q+K norm+rope + KV cache write (effort-11 cycle 12). Default
        // ON when the pipeline is loaded; disable via ZINC_FUSED_QK_KV=0.
        // Per-call gates apply (q_norm/k_norm tensors present, push descriptors,
        // !packed_q_gate, !use_k_as_v, !apply_v_unit_norm_early, !diagnostics).
        const fused_qk_kv_env = getenv("ZINC_FUSED_QK_KV");
        const fused_qk_kv_explicitly_off = fused_qk_kv_env != null and std.mem.eql(u8, fused_qk_kv_env.?, "0");
        const fused_qk_kv_enabled = !fused_qk_kv_explicitly_off and
            elementwise.pipeline_qk_norm_rope_kv_write != null and
            elementwise.pipeline_kv_cache_write != null and
            instance.push_descriptor_fn != null;
        if (fused_qk_kv_enabled) {
            log.info("Fused Q+K norm+rope + KV cache write ENABLED (default, set ZINC_FUSED_QK_KV=0 to disable)", .{});
        } else if (fused_qk_kv_explicitly_off) {
            log.info("Fused Q+K norm+rope + KV cache write DISABLED via ZINC_FUSED_QK_KV=0", .{});
        }

        // Q5_K MoE K-parallel shader: default ON when the pipeline is loaded,
        // disabled by setting ZINC_MOE_Q5K_KPAR=0. Targets the ~713 ms MoE down
        // bucket (Q5_K weights) on the Qwen3.6-35B flagship prefill. Mirrors the
        // Q4_K kpar pattern (16 threads per Q5_K superblock + wave64 subgroupAdd).
        const moe_q5k_kpar_env = getenv("ZINC_MOE_Q5K_KPAR");
        const moe_q5k_kpar_explicitly_off = moe_q5k_kpar_env != null and std.mem.eql(u8, moe_q5k_kpar_env.?, "0");
        const moe_q5k_kpar_enabled = !moe_q5k_kpar_explicitly_off and dmmv.pipeline_q5k_moe_kpar != null;
        if (moe_q5k_kpar_enabled) {
            log.info("MoE Q5_K kpar variant ENABLED (default, set ZINC_MOE_Q5K_KPAR=0 to disable)", .{});
        } else if (moe_q5k_kpar_explicitly_off) {
            log.info("MoE Q5_K kpar variant DISABLED via ZINC_MOE_Q5K_KPAR=0", .{});
        }

        // softmax_topk v2 (subgroup-parallel): default ON when the pipeline is
        // loaded, disable via ZINC_TOPK_V1=1 to fall back to the v1 shared-mem
        // single-thread scan shader.
        const topk_v1_env = getenv("ZINC_TOPK_V1");
        const topk_v1_forced = topk_v1_env != null and std.mem.eql(u8, topk_v1_env.?, "1");
        const topk_v2_enabled = !topk_v1_forced and elementwise.pipeline_softmax_topk_v2 != null;
        if (topk_v2_enabled) {
            log.info("softmax_topk v2 (subgroup-parallel) ENABLED (default, set ZINC_TOPK_V1=1 to revert)", .{});
        } else if (topk_v1_forced) {
            log.info("softmax_topk v2 DISABLED via ZINC_TOPK_V1=1; using v1 shared-mem scan", .{});
        }

        // Split-K flash attention. Default-on with N_I_CHUNKS=4. The
        // attention.zig init resolves ZINC_FA_SPLIT_K and creates the pair
        // of pipelines (split, merge); we just mirror the active count here
        // and allocate the partial-output buffer when active.
        const fa_split_k = attention.fa_split_k_active;
        const fa_split_k_forced = getenv("ZINC_FA_SPLIT_K") != null;
        if (fa_split_k > 1) {
            if (fa_split_k_forced) {
                log.info("Flash-attn split-K ENABLED: N_I_CHUNKS={d} via ZINC_FA_SPLIT_K", .{fa_split_k});
            } else {
                log.info("Flash-attn split-K ENABLED: N_I_CHUNKS={d} for seq_len>=128 (default; set ZINC_FA_SPLIT_K=0 to disable or =4 to force)", .{fa_split_k});
            }
        }
        var partial_attn_out_buf = Buffer{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = instance.device };
        if (fa_split_k > 1) {
            const partial_o_floats: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, config.n_heads) * fa_split_k * config.head_dim;
            const partial_lse_floats: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, config.n_heads) * fa_split_k * 2;
            const partial_size = (partial_o_floats + partial_lse_floats) * @sizeOf(f32);
            partial_attn_out_buf = try Buffer.initDeviceLocal(instance, partial_size, storage_xfer);
            errdefer partial_attn_out_buf.deinit();
        }

        // Batched flash-attention foundation (opt-in). When ZINC_BATCH_ATTN=1 and
        // the flash_attn_batched pipeline is loaded, the attention call site
        // routes through the batched shader. Foundation step calls with
        // n_queries=1 for correctness parity with the decode-shape shader.
        const batch_attn_env = getenv("ZINC_BATCH_ATTN");
        const batch_attn_flag = batch_attn_env != null and std.mem.eql(u8, batch_attn_env.?, "1");
        const batch_attn_enabled = batch_attn_flag and attention.pipeline_batched != null;
        if (batch_attn_enabled) {
            log.info("Flash-attn batched path ENABLED (ZINC_BATCH_ATTN=1); n_queries=1 foundation", .{});
        } else if (batch_attn_flag) {
            log.info("ZINC_BATCH_ATTN=1 requested but flash_attn_batched pipeline absent; using decode-shape shader", .{});
        }

        // Step 11a foundation: per-(token, layer) routing capture buffer.
        // Enabled by ZINC_CAPTURE_ROUTING=1. Dormant downstream — this cycle only
        // verifies the copy path is correct and measures the flag-on overhead so
        // Step 11b can wire token-permute on top without re-proving the plumbing.
        const capture_env = getenv("ZINC_CAPTURE_ROUTING");
        const capture_flag = capture_env != null and std.mem.eql(u8, capture_env.?, "1");
        var routing_capture_buf = Buffer{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = instance.device };
        var routing_capture_slot_bytes: u32 = 0;
        var routing_capture_max_tokens: u32 = 0;
        if (capture_flag and n_used_experts > 0 and config.n_layers > 0) {
            const MAX_CAPTURE_TOKENS: u32 = 2048;
            const slot_bytes: u32 = @as(u32, 2) * n_used_experts * @sizeOf(u32);
            const total_bytes = @as(vk.c.VkDeviceSize, MAX_CAPTURE_TOKENS) *
                @as(vk.c.VkDeviceSize, config.n_layers) *
                @as(vk.c.VkDeviceSize, slot_bytes);
            routing_capture_buf = try Buffer.initDeviceLocal(
                instance,
                total_bytes,
                vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            );
            errdefer routing_capture_buf.deinit();
            routing_capture_slot_bytes = slot_bytes;
            routing_capture_max_tokens = MAX_CAPTURE_TOKENS;
            log.info("ZINC_CAPTURE_ROUTING=1: routing capture buffer {d} B (tokens={d} layers={d} slot={d}B)", .{
                total_bytes, MAX_CAPTURE_TOKENS, config.n_layers, slot_bytes,
            });
        }

        // Effort-6 Step 1 wire-in: opt-in routing of the LM-head DMMV
        // through the tiled mul_mm_q4k pipeline. Eligible only when (a)
        // the flag is set, (b) the pipeline loaded, (c) push descriptors
        // are available (recordMulMmQ4K uses pushDescAndDispatch). The
        // weight-quant + hidden_dim alignment check happens at the call
        // site since both depend on the resolved LM head tensor.
        const mul_mm_lm_head_env = getenv("ZINC_MUL_MM_LM_HEAD");
        const mul_mm_lm_head_flag = mul_mm_lm_head_env != null and std.mem.eql(u8, mul_mm_lm_head_env.?, "1");
        const mul_mm_lm_head_enabled = mul_mm_lm_head_flag and
            dmmv.pipeline_mul_mm_q4k != null and
            instance.push_descriptor_fn != null;
        if (mul_mm_lm_head_enabled) {
            log.info("LM-head mul_mm_q4k path ENABLED (ZINC_MUL_MM_LM_HEAD=1)", .{});
        } else if (mul_mm_lm_head_flag) {
            log.info("ZINC_MUL_MM_LM_HEAD=1 requested but prerequisites missing (mul_mm_q4k pipeline or push descriptors); using DMMV", .{});
        }

        // Effort-6 Step 5 wire-in: route Q4_K prefill projections through the
        // tiled mul_mm_q4k pipeline when n_tokens >= 16. Default ON because
        // (a) the pipeline is already loaded and validated against per-token
        // DMMV via the LM-head path, (b) the chunked kpar shader re-reads the
        // full weight tensor once per MAX_COLS=40 chunk while mul_mm reuses an
        // M-tile across all N-tiles, and (c) decode (n_tokens=1) and Q6_K
        // tensors are unaffected because dispatchProjectionBatched gates on
        // tensor type + token count and falls back to the existing path
        // otherwise. Opt out via ZINC_MUL_MM_PROJ=0.
        const mul_mm_proj_env = getenv("ZINC_MUL_MM_PROJ");
        const mul_mm_proj_explicitly_off = mul_mm_proj_env != null and std.mem.eql(u8, mul_mm_proj_env.?, "0");
        const mul_mm_proj_enabled = !mul_mm_proj_explicitly_off and
            dmmv.pipeline_mul_mm_q4k != null and
            instance.push_descriptor_fn != null;
        if (mul_mm_proj_enabled) {
            log.info("Q4_K projection mul_mm path ENABLED (default; set ZINC_MUL_MM_PROJ=0 to disable)", .{});
        } else if (mul_mm_proj_explicitly_off) {
            log.info("Q4_K projection mul_mm path DISABLED via ZINC_MUL_MM_PROJ=0; using kpar/serial batch shaders", .{});
        }

        const qwen36_batched_gateup_env = getenv("ZINC_QWEN36_27B_BATCH_FUSED_GATEUP");
        const qwen36_batched_gateup_explicitly_off = qwen36_batched_gateup_env != null and
            std.mem.eql(u8, qwen36_batched_gateup_env.?, "0");
        const qwen36_batched_gateup_enabled = !qwen36_batched_gateup_explicitly_off and
            dmmv.pipeline_mul_mm_q4k_gate_up_swiglu != null and
            instance.push_descriptor_fn != null;
        if (qwen36_batched_gateup_enabled) {
            log.info("Qwen3.6-27B batched dense gate+up+SwiGLU path ENABLED (default, set ZINC_QWEN36_27B_BATCH_FUSED_GATEUP=0 to disable)", .{});
        } else if (qwen36_batched_gateup_explicitly_off) {
            log.info("Qwen3.6-27B batched dense gate+up+SwiGLU path DISABLED via ZINC_QWEN36_27B_BATCH_FUSED_GATEUP=0", .{});
        }

        const qwen36_q6_prefill_mul_mm_env = getenv("ZINC_QWEN36_27B_Q6_DOWN_MUL_MM");
        const qwen36_q6_prefill_mul_mm_explicitly_off = qwen36_q6_prefill_mul_mm_env != null and
            std.mem.eql(u8, qwen36_q6_prefill_mul_mm_env.?, "0");
        const qwen36_q6_prefill_mul_mm_enabled = !qwen36_q6_prefill_mul_mm_explicitly_off and
            dmmv.pipeline_mul_mm_q6k != null and
            instance.push_descriptor_fn != null;
        if (qwen36_q6_prefill_mul_mm_enabled) {
            log.info("Qwen3.6-27B Q6_K prefill mul_mm path ENABLED for dense-down/SSM-wqkv (default, set ZINC_QWEN36_27B_Q6_DOWN_MUL_MM=0 to disable)", .{});
        } else if (qwen36_q6_prefill_mul_mm_explicitly_off) {
            log.info("Qwen3.6-27B Q6_K prefill mul_mm path DISABLED via ZINC_QWEN36_27B_Q6_DOWN_MUL_MM=0", .{});
        }

        const q8_wide_lm_env = getenv("ZINC_Q8_WIDE_LM_HEAD");
        const q8_wide_lm_flag = q8_wide_lm_env != null and std.mem.eql(u8, q8_wide_lm_env.?, "1");
        const q8_wide_lm_enabled = q8_wide_lm_flag and dmmv.pipeline_q8_0_wide != null;
        if (q8_wide_lm_enabled) {
            log.info("Q8_0 wide LM-head path ENABLED via ZINC_Q8_WIDE_LM_HEAD=1", .{});
        } else if (q8_wide_lm_flag) {
            log.info("ZINC_Q8_WIDE_LM_HEAD=1 requested but the Q8_0 wide pipeline is missing; using generic Q8_0 DMMV", .{});
        }

        const q8_batch_lm_env = getenv("ZINC_Q8_BATCH_LM_HEAD");
        const q8_batch_lm_flag = q8_batch_lm_env != null and std.mem.eql(u8, q8_batch_lm_env.?, "1");
        const q8_batch_lm_enabled = q8_batch_lm_flag and dmmv.pipeline_q8_0_batch != null;
        if (q8_batch_lm_enabled) {
            log.info("Q8_0 batch LM-head path ENABLED via ZINC_Q8_BATCH_LM_HEAD=1", .{});
        } else if (q8_batch_lm_flag) {
            log.info("ZINC_Q8_BATCH_LM_HEAD=1 requested but the Q8_0 batch pipeline is missing; using generic Q8_0 DMMV", .{});
        }

        const q8_1_lm_env = getenv("ZINC_Q8_1_LM_HEAD");
        const q8_1_lm_flag = q8_1_lm_env != null and std.mem.eql(u8, q8_1_lm_env.?, "1");
        const q8_1_lm_enabled = q8_1_lm_flag and
            dmmv.pipeline_q8_0_q8_1 != null and
            dmmv.pipeline_quantize_q8_1 != null and
            instance.push_descriptor_fn != null and
            (config.hidden_dim & 31) == 0;
        if (q8_1_lm_enabled) {
            log.info("Q8_0 x Q8_1 LM-head path ENABLED via ZINC_Q8_1_LM_HEAD=1", .{});
        } else if (q8_1_lm_flag) {
            log.info("ZINC_Q8_1_LM_HEAD=1 requested but prerequisites are missing; using generic Q8_0 DMMV", .{});
        }

        const q8_spec_env = getenv("ZINC_Q8_SPEC_DMMV");
        const q8_spec_enabled = q8_spec_env != null and std.mem.eql(u8, q8_spec_env.?, "1");
        if (q8_spec_enabled) {
            log.info("Q8_0 K-specialized DMMV path ENABLED via ZINC_Q8_SPEC_DMMV=1", .{});
        }

        const fused_ssm_qkv_z_env = getenv("ZINC_FUSED_SSM_QKV_Z");
        const fused_ssm_qkv_z_flag = fused_ssm_qkv_z_env != null and std.mem.eql(u8, fused_ssm_qkv_z_env.?, "1");
        const fused_ssm_qkv_z_enabled = fused_ssm_qkv_z_flag and
            dmmv.pipeline_q8_0_fused_pair != null and
            instance.push_descriptor_fn != null and
            (config.hidden_dim & 31) == 0;
        if (fused_ssm_qkv_z_enabled) {
            log.info("SSM fused Q8_0 wqkv+z projection ENABLED via ZINC_FUSED_SSM_QKV_Z=1", .{});
        } else if (fused_ssm_qkv_z_flag) {
            log.info("ZINC_FUSED_SSM_QKV_Z=1 requested but prerequisites missing; using separate SSM projections", .{});
        }

        // Effort-6 Step 5 prerequisite: count_experts wire-in. When
        // ZINC_COUNT_EXPERTS_PREFILL=1 is set alongside ZINC_CAPTURE_ROUTING=1
        // and the count_experts pipeline is loaded, prefillBatch will scan
        // the captured routing buffer at the end of prefill and produce a
        // per-(layer, expert) count buffer. mul_mm_id_q4k binds this buffer
        // for its early-exit path; the next cycle will wire it into the
        // batched MoE FFN dispatch.
        const count_experts_env = getenv("ZINC_COUNT_EXPERTS_PREFILL");
        const count_experts_flag = count_experts_env != null and std.mem.eql(u8, count_experts_env.?, "1");
        const count_experts_enabled = count_experts_flag and
            capture_flag and routing_capture_buf.handle != null and
            dmmv.pipeline_count_experts != null and
            instance.push_descriptor_fn != null and
            n_used_experts > 0 and config.n_experts > 0 and config.n_layers > 0;
        var prefill_expert_count_buf = Buffer{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = instance.device };
        if (count_experts_enabled) {
            const counts_bytes: vk.c.VkDeviceSize =
                @as(vk.c.VkDeviceSize, config.n_layers) *
                @as(vk.c.VkDeviceSize, config.n_experts) *
                @sizeOf(u32);
            prefill_expert_count_buf = try Buffer.initDeviceLocal(
                instance,
                counts_bytes,
                vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            );
            errdefer prefill_expert_count_buf.deinit();
            log.info("ZINC_COUNT_EXPERTS_PREFILL=1: prefill expert-count buffer {d} B (layers={d} experts={d})", .{
                counts_bytes, config.n_layers, config.n_experts,
            });
        } else if (count_experts_flag) {
            log.info("ZINC_COUNT_EXPERTS_PREFILL=1 requested but prerequisites missing (capture-routing flag, count_experts pipeline, or push descriptors); skipping", .{});
        }

        // Effort-6 Step 5 prerequisite (cycle 36): per-(token, layer) FFN-input
        // capture buffer. ZINC_CAPTURE_FFN_INPUT=1 allocates a device-local
        // buffer sized [max_tokens × n_layers × hidden_dim × f32] and the MoE
        // hot path copies ffn_norm_buf into slot (token, layer). Combined with
        // routing_capture_buf + prefill_expert_count_buf, this provides the
        // three inputs mul_mm_id_q4k needs to replace per-token MoE FFN
        // dispatches with one batched GEMM per layer. Default-OFF.
        const ffn_input_capture_env = getenv("ZINC_CAPTURE_FFN_INPUT");
        const ffn_input_capture_flag = ffn_input_capture_env != null and std.mem.eql(u8, ffn_input_capture_env.?, "1");
        var prefill_ffn_input_capture_buf = Buffer{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = instance.device };
        var prefill_ffn_input_capture_max_tokens: u32 = 0;
        if (ffn_input_capture_flag and config.n_layers > 0 and config.hidden_dim > 0) {
            const MAX_CAPTURE_TOKENS: u32 = 2048;
            const total_bytes = @as(vk.c.VkDeviceSize, MAX_CAPTURE_TOKENS) *
                @as(vk.c.VkDeviceSize, config.n_layers) *
                @as(vk.c.VkDeviceSize, config.hidden_dim) *
                @sizeOf(f32);
            prefill_ffn_input_capture_buf = try Buffer.initDeviceLocal(
                instance,
                total_bytes,
                vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            );
            errdefer prefill_ffn_input_capture_buf.deinit();
            prefill_ffn_input_capture_max_tokens = MAX_CAPTURE_TOKENS;
            log.info("ZINC_CAPTURE_FFN_INPUT=1: ffn input capture buffer {d} B (tokens={d} layers={d} hidden={d})", .{
                total_bytes, MAX_CAPTURE_TOKENS, config.n_layers, config.hidden_dim,
            });
        } else if (ffn_input_capture_flag) {
            log.info("ZINC_CAPTURE_FFN_INPUT=1 requested but prerequisites missing (n_layers or hidden_dim is 0); skipping", .{});
        }

        // Effort-6 cycle 97 (A3b foundation): allocate per-token capture
        // buffers for ssm_delta_net inputs at layer 0. The shader already
        // supports n_tok>1 (cycle 77 added the t-loop fold + strides). This
        // path captures per-token (alpha, beta, conv_out) into strided slots
        // during the per-token loop, then dispatches one batched delta_net
        // for layer 0 after the per-token loop drains. State backup/restore
        // protects the real per-token state from the validation dispatch so
        // decode keeps producing the correct output. Default-OFF.
        const a3b_validate_env = getenv("ZINC_A3B_VALIDATE");
        const a3b_validate_flag = a3b_validate_env != null and std.mem.eql(u8, a3b_validate_env.?, "1");
        const a3b_production_env = getenv("ZINC_A3B_PRODUCTION");
        const a3b_production_flag = a3b_production_env != null and std.mem.eql(u8, a3b_production_env.?, "1");
        // Cycle 127: capture buffers allocated only for validate. Cycle 125
        // tied production allocation to the same set, but the production
        // post-loop dispatch consumed corrupted captures (because cycle
        // 125's runSsmLayerGpu skip cascaded wrong hidden_buf into
        // ssm_proj at later layers). The broken post-loop is removed;
        // production becomes a no-op. ZINC_A3B_PRODUCTION env reading
        // and the use_a3b_production field stay so cycle 128 can re-engage
        // the layer-major restructure without reintroducing the env name.
        const a3b_buffers_needed = a3b_validate_flag and
            config.ssm_d_inner > 0 and
            config.ssm_dt_rank > 0 and
            elementwise.pipeline_ssm_delta_net != null and
            instance.push_descriptor_fn != null;
        const a3b_validate_enabled = a3b_validate_flag and a3b_buffers_needed;
        const a3b_production_enabled = a3b_production_flag;
        var a3b_alpha_capture: ?Buffer = null;
        var a3b_beta_capture: ?Buffer = null;
        var a3b_conv_out_capture: ?Buffer = null;
        var a3b_state_backup: ?Buffer = null;
        var a3b_delta_out: ?Buffer = null;
        var a3b_per_token_delta_out: ?Buffer = null;
        var a3b_gate_capture: ?Buffer = null;
        var a3b_capture_max_tokens: u32 = 0;
        if (a3b_buffers_needed) {
            // Cycle 123: extend the layer-0-only validation to ALL SSM layers.
            // The capture/state-backup/output buffers are sized to hold
            // n_layers × max_tokens slots so every SSM layer's per-token
            // (alpha, beta, conv_out) inputs and per-token delta_out outputs
            // can be captured during the per-token loop. After the loop
            // drains, prefillBatch dispatches one batched ssm_delta_net per
            // SSM layer (state-backed-up, zeroed, dispatched at
            // n_tok=prompt_len, restored) and diffs against the per-token
            // output for that layer. Cap dropped 2048 → 256 to keep total
            // memory under ~1 GB; the long-context benchmark uses 154 tokens
            // so 256 is enough headroom.
            const A3B_MAX_TOKENS: u32 = 256;
            const n_layers_a3b = config.n_layers;
            const dt_rank_a3b = config.ssm_dt_rank;
            const d_inner_a3b = config.ssm_d_inner;
            const head_v_dim_a3b = d_inner_a3b / dt_rank_a3b;
            const conv_ch_a3b = d_inner_a3b + 2 * config.ssm_n_group * config.ssm_d_state;
            const state_elems_a3b = dt_rank_a3b * head_v_dim_a3b * head_v_dim_a3b;
            const ab_bytes_a3b: vk.c.VkDeviceSize =
                @as(vk.c.VkDeviceSize, n_layers_a3b) *
                @as(vk.c.VkDeviceSize, A3B_MAX_TOKENS) *
                @as(vk.c.VkDeviceSize, dt_rank_a3b) *
                @sizeOf(f32);
            const conv_bytes_a3b: vk.c.VkDeviceSize =
                @as(vk.c.VkDeviceSize, n_layers_a3b) *
                @as(vk.c.VkDeviceSize, A3B_MAX_TOKENS) *
                @as(vk.c.VkDeviceSize, conv_ch_a3b) *
                @sizeOf(f32);
            const state_bytes_a3b: vk.c.VkDeviceSize =
                @as(vk.c.VkDeviceSize, n_layers_a3b) *
                @as(vk.c.VkDeviceSize, state_elems_a3b) * @sizeOf(f32);
            const out_bytes_a3b: vk.c.VkDeviceSize =
                @as(vk.c.VkDeviceSize, n_layers_a3b) *
                @as(vk.c.VkDeviceSize, A3B_MAX_TOKENS) *
                @as(vk.c.VkDeviceSize, d_inner_a3b) *
                @sizeOf(f32);
            const usage_dst = vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            const usage_src_dst = vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            a3b_alpha_capture = try Buffer.initDeviceLocal(instance, ab_bytes_a3b, usage_dst);
            errdefer if (a3b_alpha_capture) |*b| b.deinit();
            a3b_beta_capture = try Buffer.initDeviceLocal(instance, ab_bytes_a3b, usage_dst);
            errdefer if (a3b_beta_capture) |*b| b.deinit();
            a3b_conv_out_capture = try Buffer.initDeviceLocal(instance, conv_bytes_a3b, usage_dst);
            errdefer if (a3b_conv_out_capture) |*b| b.deinit();
            a3b_state_backup = try Buffer.initDeviceLocal(instance, state_bytes_a3b, usage_src_dst);
            errdefer if (a3b_state_backup) |*b| b.deinit();
            a3b_delta_out = try Buffer.initDeviceLocal(instance, out_bytes_a3b, usage_src_dst);
            errdefer if (a3b_delta_out) |*b| b.deinit();
            a3b_per_token_delta_out = try Buffer.initDeviceLocal(instance, out_bytes_a3b, usage_src_dst);
            errdefer if (a3b_per_token_delta_out) |*b| b.deinit();
            a3b_gate_capture = try Buffer.initDeviceLocal(instance, out_bytes_a3b, usage_src_dst);
            errdefer if (a3b_gate_capture) |*b| b.deinit();
            a3b_capture_max_tokens = A3B_MAX_TOKENS;
            const total_mb_a3b: u64 = @intCast((ab_bytes_a3b * 2 + conv_bytes_a3b + state_bytes_a3b + out_bytes_a3b * 3) / (1024 * 1024));
            // Cycle 127: production-only no longer allocates these buffers
            // (cycle 125's broken post-loop dispatch was removed). Mode
            // string reflects only validate now; production flag without
            // validate is a no-op so this branch isn't reached for it.
            log.info("ZINC_A3B_VALIDATE: A3b ALL-LAYER capture buffers allocated (n_layers={d}, max_tokens={d}, dt_rank={d}, conv_ch={d}, d_inner={d}, state={d} elems, total {d} MB)", .{
                n_layers_a3b, A3B_MAX_TOKENS, dt_rank_a3b, conv_ch_a3b, d_inner_a3b, state_elems_a3b, total_mb_a3b,
            });
        } else if (a3b_validate_flag) {
            log.info("ZINC_A3B_VALIDATE requested but prerequisites missing (no SSM, missing pipeline, or no push descriptors); skipping", .{});
        } else if (a3b_production_flag) {
            // Cycle 127: ZINC_A3B_PRODUCTION is currently dormant (cycle
            // 125's broken implementation was rolled back). Setting the
            // flag has no effect until cycle 128's layer-major restructure
            // re-engages it.
            log.info("ZINC_A3B_PRODUCTION=1 set but currently a no-op (cycle 125 wire-up reverted in cycle 127; layer-major restructure pending in cycle 128).", .{});
        }

        const dense_prefill_validate_env = getenv("ZINC_QWEN36_27B_PREFILL_VALIDATE");
        const dense_prefill_validate_requested = dense_prefill_validate_env != null and
            std.mem.eql(u8, dense_prefill_validate_env.?, "1");
        var dense_prefill_validate_enabled = false;
        var dense_prefill_validate_layer: u32 = 0;
        var dense_prefill_validate_max_tokens: u32 = 0;
        var dense_prefill_validate_norm_ref: ?Buffer = null;
        var dense_prefill_validate_pre_hidden_ref: ?Buffer = null;
        var dense_prefill_validate_post_hidden_ref: ?Buffer = null;
        var dense_prefill_validate_gate_ref: ?Buffer = null;
        var dense_prefill_validate_up_ref: ?Buffer = null;
        var dense_prefill_validate_swiglu_ref: ?Buffer = null;
        var dense_prefill_validate_down_ref: ?Buffer = null;
        var dense_prefill_validate_staging: ?Buffer = null;
        if (dense_prefill_validate_requested and
            config.n_experts == 0 and
            config.ssm_d_inner > 0 and
            config.n_layers > 0 and
            config.hidden_dim > 0 and
            inter_val > 0)
        {
            const raw_tokens = getenv("ZINC_QWEN36_27B_PREFILL_VALIDATE_TOKENS");
            const parsed_tokens = if (raw_tokens) |raw| std.fmt.parseInt(u32, raw, 10) catch 16 else 16;
            dense_prefill_validate_max_tokens = @min(@max(parsed_tokens, @as(u32, 1)), @as(u32, 16));
            const raw_layer = getenv("ZINC_QWEN36_27B_PREFILL_VALIDATE_LAYER");
            const parsed_layer = if (raw_layer) |raw| std.fmt.parseInt(u32, raw, 10) catch 0 else 0;
            dense_prefill_validate_layer = @min(parsed_layer, config.n_layers - 1);

            const hidden_capture_bytes: vk.c.VkDeviceSize =
                @as(vk.c.VkDeviceSize, dense_prefill_validate_max_tokens) *
                @as(vk.c.VkDeviceSize, config.hidden_dim) *
                @sizeOf(f32);
            const inter_capture_bytes: vk.c.VkDeviceSize =
                @as(vk.c.VkDeviceSize, dense_prefill_validate_max_tokens) *
                @as(vk.c.VkDeviceSize, inter_val) *
                @sizeOf(f32);
            const usage_ref = vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            dense_prefill_validate_norm_ref = try Buffer.initDeviceLocal(instance, hidden_capture_bytes, usage_ref);
            errdefer if (dense_prefill_validate_norm_ref) |*b| b.deinit();
            dense_prefill_validate_pre_hidden_ref = try Buffer.initDeviceLocal(instance, hidden_capture_bytes, usage_ref);
            errdefer if (dense_prefill_validate_pre_hidden_ref) |*b| b.deinit();
            dense_prefill_validate_post_hidden_ref = try Buffer.initDeviceLocal(instance, hidden_capture_bytes, usage_ref);
            errdefer if (dense_prefill_validate_post_hidden_ref) |*b| b.deinit();
            dense_prefill_validate_gate_ref = try Buffer.initDeviceLocal(instance, inter_capture_bytes, usage_ref);
            errdefer if (dense_prefill_validate_gate_ref) |*b| b.deinit();
            dense_prefill_validate_up_ref = try Buffer.initDeviceLocal(instance, inter_capture_bytes, usage_ref);
            errdefer if (dense_prefill_validate_up_ref) |*b| b.deinit();
            dense_prefill_validate_swiglu_ref = try Buffer.initDeviceLocal(instance, inter_capture_bytes, usage_ref);
            errdefer if (dense_prefill_validate_swiglu_ref) |*b| b.deinit();
            dense_prefill_validate_down_ref = try Buffer.initDeviceLocal(instance, hidden_capture_bytes, usage_ref);
            errdefer if (dense_prefill_validate_down_ref) |*b| b.deinit();

            const staging_bytes = hidden_capture_bytes * 4 + inter_capture_bytes * 6;
            dense_prefill_validate_staging = try Buffer.init(
                instance,
                staging_bytes,
                vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                vk.c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            );
            errdefer if (dense_prefill_validate_staging) |*b| b.deinit();
            {
                var map_ptr: ?*anyopaque = null;
                if (dense_prefill_validate_staging) |*staging_buf| {
                    const mr = vk.c.vkMapMemory(instance.device, staging_buf.memory, 0, staging_bytes, 0, &map_ptr);
                    if (mr != vk.c.VK_SUCCESS) return error.MapMemoryFailed;
                    staging_buf.mapped = @ptrCast(map_ptr);
                }
            }
            dense_prefill_validate_enabled = true;
            log.info("ZINC_QWEN36_27B_PREFILL_VALIDATE=1: dense FFN validator layer={d} tokens={d} hidden={d} inter={d} staging={d} B (replays chunks 4/8/16 when captured)", .{
                dense_prefill_validate_layer,
                dense_prefill_validate_max_tokens,
                config.hidden_dim,
                inter_val,
                staging_bytes,
            });
        } else if (dense_prefill_validate_requested) {
            log.info("ZINC_QWEN36_27B_PREFILL_VALIDATE=1 requested but prerequisites missing (requires dense SSM model with nonzero hidden/intermediate dims); skipping", .{});
        }

        var ssm_prefill_validate_enabled = false;
        var ssm_prefill_validate_norm_ref: ?Buffer = null;
        var ssm_prefill_validate_qkv_ref: ?Buffer = null;
        var ssm_prefill_validate_z_ref: ?Buffer = null;
        var ssm_prefill_validate_alpha_ref: ?Buffer = null;
        var ssm_prefill_validate_beta_ref: ?Buffer = null;
        var ssm_prefill_validate_conv_ref: ?Buffer = null;
        var ssm_prefill_validate_delta_ref: ?Buffer = null;
        var ssm_prefill_validate_delta_replay: ?Buffer = null;
        var ssm_prefill_validate_gnorm_ref: ?Buffer = null;
        var ssm_prefill_validate_pre_hidden_ref: ?Buffer = null;
        var ssm_prefill_validate_post_hidden_ref: ?Buffer = null;
        var ssm_prefill_validate_state_backup: ?Buffer = null;
        var ssm_prefill_validate_staging: ?Buffer = null;
        if (dense_prefill_validate_enabled and config.ssm_d_inner > 0 and config.ssm_dt_rank > 0) {
            const full_attn_interval_v: u32 = if (config.full_attn_interval > 0) config.full_attn_interval else 1;
            const validate_layer_is_ssm = ((dense_prefill_validate_layer + 1) % full_attn_interval_v) != 0;
            const lt_validate = layer_tensors[dense_prefill_validate_layer];
            if (validate_layer_is_ssm and
                lt_validate.attn_qkv != null and
                lt_validate.attn_gate != null and
                lt_validate.ssm_alpha != null and
                lt_validate.ssm_beta != null)
            {
                const conv_channels_validate = config.ssm_d_inner + 2 * config.ssm_n_group * config.ssm_d_state;
                const hidden_capture_bytes: vk.c.VkDeviceSize =
                    @as(vk.c.VkDeviceSize, dense_prefill_validate_max_tokens) *
                    @as(vk.c.VkDeviceSize, config.hidden_dim) *
                    @sizeOf(f32);
                const qkv_capture_bytes: vk.c.VkDeviceSize =
                    @as(vk.c.VkDeviceSize, dense_prefill_validate_max_tokens) *
                    @as(vk.c.VkDeviceSize, conv_channels_validate) *
                    @sizeOf(f32);
                const z_capture_bytes: vk.c.VkDeviceSize =
                    @as(vk.c.VkDeviceSize, dense_prefill_validate_max_tokens) *
                    @as(vk.c.VkDeviceSize, config.ssm_d_inner) *
                    @sizeOf(f32);
                const ab_capture_bytes: vk.c.VkDeviceSize =
                    @as(vk.c.VkDeviceSize, dense_prefill_validate_max_tokens) *
                    @as(vk.c.VkDeviceSize, config.ssm_dt_rank) *
                    @sizeOf(f32);
                const head_v_dim_validate = config.ssm_d_inner / config.ssm_dt_rank;
                const state_bytes_validate: vk.c.VkDeviceSize =
                    @as(vk.c.VkDeviceSize, config.ssm_dt_rank) *
                    @as(vk.c.VkDeviceSize, head_v_dim_validate) *
                    @as(vk.c.VkDeviceSize, head_v_dim_validate) *
                    @sizeOf(f32);
                const can_delta_replay_validate =
                    elementwise.pipeline_ssm_delta_net != null and
                    instance.push_descriptor_fn != null and
                    dense_prefill_validate_layer < gpu_ssm_states.len and
                    gpu_ssm_states[dense_prefill_validate_layer].handle != null and
                    state_bytes_validate <= gpu_ssm_states[dense_prefill_validate_layer].size;
                const can_output_replay_validate =
                    can_delta_replay_validate and
                    elementwise.pipeline_ssm_gated_norm != null and
                    lt_validate.ssm_out != null;
                const usage_ref = vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                    vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                    vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT;
                ssm_prefill_validate_norm_ref = try Buffer.initDeviceLocal(instance, hidden_capture_bytes, usage_ref);
                errdefer if (ssm_prefill_validate_norm_ref) |*b| b.deinit();
                ssm_prefill_validate_qkv_ref = try Buffer.initDeviceLocal(instance, qkv_capture_bytes, usage_ref);
                errdefer if (ssm_prefill_validate_qkv_ref) |*b| b.deinit();
                ssm_prefill_validate_z_ref = try Buffer.initDeviceLocal(instance, z_capture_bytes, usage_ref);
                errdefer if (ssm_prefill_validate_z_ref) |*b| b.deinit();
                ssm_prefill_validate_alpha_ref = try Buffer.initDeviceLocal(instance, ab_capture_bytes, usage_ref);
                errdefer if (ssm_prefill_validate_alpha_ref) |*b| b.deinit();
                ssm_prefill_validate_beta_ref = try Buffer.initDeviceLocal(instance, ab_capture_bytes, usage_ref);
                errdefer if (ssm_prefill_validate_beta_ref) |*b| b.deinit();
                if (can_delta_replay_validate) {
                    ssm_prefill_validate_conv_ref = try Buffer.initDeviceLocal(instance, qkv_capture_bytes, usage_ref);
                    errdefer if (ssm_prefill_validate_conv_ref) |*b| b.deinit();
                    ssm_prefill_validate_delta_ref = try Buffer.initDeviceLocal(instance, z_capture_bytes, usage_ref);
                    errdefer if (ssm_prefill_validate_delta_ref) |*b| b.deinit();
                    ssm_prefill_validate_delta_replay = try Buffer.initDeviceLocal(instance, z_capture_bytes, usage_ref);
                    errdefer if (ssm_prefill_validate_delta_replay) |*b| b.deinit();
                    ssm_prefill_validate_state_backup = try Buffer.initDeviceLocal(instance, state_bytes_validate, usage_ref);
                    errdefer if (ssm_prefill_validate_state_backup) |*b| b.deinit();
                }
                if (can_output_replay_validate) {
                    ssm_prefill_validate_gnorm_ref = try Buffer.initDeviceLocal(instance, z_capture_bytes, usage_ref);
                    errdefer if (ssm_prefill_validate_gnorm_ref) |*b| b.deinit();
                    ssm_prefill_validate_pre_hidden_ref = try Buffer.initDeviceLocal(instance, hidden_capture_bytes, usage_ref);
                    errdefer if (ssm_prefill_validate_pre_hidden_ref) |*b| b.deinit();
                    ssm_prefill_validate_post_hidden_ref = try Buffer.initDeviceLocal(instance, hidden_capture_bytes, usage_ref);
                    errdefer if (ssm_prefill_validate_post_hidden_ref) |*b| b.deinit();
                }

                const staging_bytes = hidden_capture_bytes +
                    2 * (qkv_capture_bytes + z_capture_bytes) +
                    2 * ab_capture_bytes +
                    (if (can_delta_replay_validate) 2 * z_capture_bytes else 0) +
                    (if (can_output_replay_validate) (3 * hidden_capture_bytes + 2 * z_capture_bytes) else 0);
                ssm_prefill_validate_staging = try Buffer.init(
                    instance,
                    staging_bytes,
                    vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    vk.c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                );
                errdefer if (ssm_prefill_validate_staging) |*b| b.deinit();
                {
                    var map_ptr: ?*anyopaque = null;
                    if (ssm_prefill_validate_staging) |*staging_buf| {
                        const mr = vk.c.vkMapMemory(instance.device, staging_buf.memory, 0, staging_bytes, 0, &map_ptr);
                        if (mr != vk.c.VK_SUCCESS) return error.MapMemoryFailed;
                        staging_buf.mapped = @ptrCast(map_ptr);
                    }
                }
                ssm_prefill_validate_enabled = true;
                log.info("ZINC_QWEN36_27B_PREFILL_VALIDATE=1: SSM validator layer={d} tokens={d} hidden={d} conv_ch={d} d_inner={d} dt_rank={d} delta_replay={} output_replay={} staging={d} B (batched qkv/z/delta/ssm_out replay, chunks 4/8/16 when captured)", .{
                    dense_prefill_validate_layer,
                    dense_prefill_validate_max_tokens,
                    config.hidden_dim,
                    conv_channels_validate,
                    config.ssm_d_inner,
                    config.ssm_dt_rank,
                    can_delta_replay_validate,
                    can_output_replay_validate,
                    staging_bytes,
                });
            } else {
                log.info("ZINC_QWEN36_27B_PREFILL_VALIDATE=1: SSM proj validator skipped for layer={d} (not an SSM layer or missing SSM projection tensors)", .{
                    dense_prefill_validate_layer,
                });
            }
        }

        return InferenceEngine{
            .model = model,
            .gpu_config = gpu_config,
            .dmmv = dmmv,
            .use_moe_kpar = moe_kpar_enabled,
            .use_moe_q5k_kpar = moe_q5k_kpar_enabled,
            .use_moe_fused_gate_up = moe_fused_gate_up_enabled,
            .use_moe_fused_gate_up_swiglu = moe_fused_gate_up_swiglu_enabled,
            .use_moe_fused_down_acc = moe_fused_down_acc_enabled,
            .moe_topk_limit = if (gemma_topk_limit > 0) gemma_topk_limit else qwen36_topk_limit,
            .moe_prefill_tail_topk_limit = qwen36_prefill_tail_topk_limit,
            .moe_prefill_tail_topk_guard_tokens = qwen36_prefill_tail_topk_guard_tokens,
            .use_fused_rms_router = fused_rms_router_enabled,
            .use_fused_ssm_pre_norm = fused_ssm_ab_enabled,
            .use_ssm_delta_cols8 = ssm_delta_cols8_enabled,
            .use_ssm_delta_normed_qk = ssm_delta_normed_qk_enabled,
            .use_fused_dense_ffn = fused_dense_ffn_enabled,
            .use_qwen36_dense_fused_row1 = qwen36_dense_row1_enabled,
            .use_fused_oproj_merge = fused_oproj_merge_enabled,
            .use_fused_qk_kv = fused_qk_kv_enabled,
            .fa_profile_layer = fa_profile_layer_enabled,
            // ZINC_FA_PROFILE_LAYER=1 auto-enables timestamp recording. The
            // benchmark cycle does not pass --profile, so without this the
            // diagnostic emits no data. Cost is one timestamp per dispatch
            // and a vkGetQueryPoolResults wait per token; tolerable for the
            // diagnostic but tangibly slows the flag-on case.
            .profile_enabled = fa_profile_layer_enabled,
            .use_q4k_batch_kpar = q4k_batch_kpar_enabled,
            .use_softmax_topk_v2 = topk_v2_enabled,
            .use_batch_attn = batch_attn_enabled,
            .use_capture_routing = capture_flag and routing_capture_buf.handle != null,
            .use_mul_mm_lm_head = mul_mm_lm_head_enabled,
            .use_mul_mm_proj = mul_mm_proj_enabled,
            .use_qwen36_batched_fused_gateup = qwen36_batched_gateup_enabled,
            .use_qwen36_q6_prefill_mul_mm = qwen36_q6_prefill_mul_mm_enabled,
            .use_q8_wide_lm_head = q8_wide_lm_enabled,
            .use_q8_batch_lm_head = q8_batch_lm_enabled,
            .use_q8_1_lm_head = q8_1_lm_enabled,
            .use_q8_spec_dmmv = q8_spec_enabled,
            .use_fused_ssm_qkv_z = fused_ssm_qkv_z_enabled,
            .use_count_experts_prefill = count_experts_enabled,
            .use_capture_ffn_input = ffn_input_capture_flag and prefill_ffn_input_capture_buf.handle != null,
            .use_a3b_validate = a3b_validate_enabled,
            .use_a3b_production = a3b_production_enabled,
            .a3b_alpha_capture = a3b_alpha_capture,
            .a3b_beta_capture = a3b_beta_capture,
            .a3b_conv_out_capture = a3b_conv_out_capture,
            .a3b_state_backup = a3b_state_backup,
            .a3b_delta_out = a3b_delta_out,
            .a3b_per_token_delta_out = a3b_per_token_delta_out,
            .a3b_gate_capture = a3b_gate_capture,
            .a3b_capture_max_tokens = a3b_capture_max_tokens,
            .use_qwen36_dense_prefill_validate = dense_prefill_validate_enabled,
            .dense_prefill_validate_layer = dense_prefill_validate_layer,
            .dense_prefill_validate_max_tokens = dense_prefill_validate_max_tokens,
            .dense_prefill_validate_norm_ref = dense_prefill_validate_norm_ref,
            .dense_prefill_validate_pre_hidden_ref = dense_prefill_validate_pre_hidden_ref,
            .dense_prefill_validate_post_hidden_ref = dense_prefill_validate_post_hidden_ref,
            .dense_prefill_validate_gate_ref = dense_prefill_validate_gate_ref,
            .dense_prefill_validate_up_ref = dense_prefill_validate_up_ref,
            .dense_prefill_validate_swiglu_ref = dense_prefill_validate_swiglu_ref,
            .dense_prefill_validate_down_ref = dense_prefill_validate_down_ref,
            .dense_prefill_validate_staging = dense_prefill_validate_staging,
            .use_qwen36_ssm_prefill_validate = ssm_prefill_validate_enabled,
            .ssm_prefill_validate_norm_ref = ssm_prefill_validate_norm_ref,
            .ssm_prefill_validate_qkv_ref = ssm_prefill_validate_qkv_ref,
            .ssm_prefill_validate_z_ref = ssm_prefill_validate_z_ref,
            .ssm_prefill_validate_alpha_ref = ssm_prefill_validate_alpha_ref,
            .ssm_prefill_validate_beta_ref = ssm_prefill_validate_beta_ref,
            .ssm_prefill_validate_conv_ref = ssm_prefill_validate_conv_ref,
            .ssm_prefill_validate_delta_ref = ssm_prefill_validate_delta_ref,
            .ssm_prefill_validate_delta_replay = ssm_prefill_validate_delta_replay,
            .ssm_prefill_validate_gnorm_ref = ssm_prefill_validate_gnorm_ref,
            .ssm_prefill_validate_pre_hidden_ref = ssm_prefill_validate_pre_hidden_ref,
            .ssm_prefill_validate_post_hidden_ref = ssm_prefill_validate_post_hidden_ref,
            .ssm_prefill_validate_state_backup = ssm_prefill_validate_state_backup,
            .ssm_prefill_validate_staging = ssm_prefill_validate_staging,
            .routing_capture_buf = routing_capture_buf,
            .routing_capture_slot_bytes = routing_capture_slot_bytes,
            .routing_capture_max_tokens = routing_capture_max_tokens,
            .prefill_expert_count_buf = prefill_expert_count_buf,
            .prefill_ffn_input_capture_buf = prefill_ffn_input_capture_buf,
            .prefill_ffn_input_capture_max_tokens = prefill_ffn_input_capture_max_tokens,
            .elementwise = elementwise,
            .attention = attention,
            .argmax = argmax,
            .cmd_pool = cmd_pool,
            .decode_cmd = decode_cmd,
            .prefill_cmd_alt = prefill_cmd_alt,
            .prefill_embed_alt = prefill_embed_alt,
            .decode_graph = decode_graph,
            .hidden_buf = hidden_buf,
            .residual_buf = residual_buf,
            .norm_buf = norm_buf,
            .q8_1_buf = q8_1_buf,
            .logits_buf = logits_buf,
            .logits_staging = logits_staging,
            .argmax_partials_buf = argmax_partials_buf,
            .argmax_result_buf = argmax_result_buf,
            .argmax_result_staging = argmax_result_staging,
            .argmax_descriptor_set = argmax_descriptor_set,
            .argmax_phase0_workgroups = argmax_phase0_workgroups,
            .embed_staging = embed_staging,
            .q_buf = q_buf,
            .k_buf = k_buf,
            .v_buf = v_buf,
            .attn_out_buf = attn_out_buf,
            .partial_attn_out_buf = partial_attn_out_buf,
            .fa_split_k = fa_split_k,
            .fa_split_k_forced = fa_split_k_forced,
            .o_proj_buf = o_proj_buf,
            .ffn_norm_buf = ffn_norm_buf,
            .gate_buf = gate_buf,
            .up_buf = up_buf,
            .swiglu_buf = swiglu_buf,
            .down_buf = down_buf,
            .moe_out_buf = moe_out_buf,
            .router_logits_buf = router_logits_buf,
            .router_staging = router_staging,
            .rope_freq_buf = rope_freq_buf,
            .unit_norm_weights = unit_norm_weights,
            .attn_sinks_buf = attn_sinks_buf,
            .kv_k_cache = kv_k_cache,
            .kv_v_cache = kv_v_cache,
            .page_table_buf = page_table_buf,
            .page_table_staging = page_table_staging,
            .kv_page_pool = kv_page_pool,
            .active_kv_page_ids = null,
            .active_kv_request_id = null,
            .next_kv_request_id = 1,
            .ssm_conv_states = ssm_conv_states,
            .ssm_states = ssm_states,
            .ssm_hidden_staging = ssm_hidden_staging,
            .gpu_ssm_conv_states = gpu_ssm_conv_states,
            .gpu_ssm_states = gpu_ssm_states,
            .ssm_conv_state_offsets = ssm_conv_state_offsets,
            .router_output_buf = router_output_buf,
            .shared_pool = shared_pool,
            .tensor_map = tensor_map,
            .layer_tensors = layer_tensors,
            .layer_output_scales = layer_output_scales,
            .instance = instance,
            .allocator = allocator,
            .max_context_tokens = max_ctx,
            .modeled_decode_bytes_per_token = modeled_decode_bytes_per_token,
            .timestamp_query_pool = timestamp_pool,
            .timestamp_period_ns = timestamp_period_ns,
        };
    }

    // -----------------------------------------------------------------------
    // Profiling
    // -----------------------------------------------------------------------

    /// Enable full GPU + CPU profiling. The timestamp query pool is created in `init`,
    /// so this just flips the runtime flag. Returns an error if pool creation failed.
    pub fn enableProfiling(self: *InferenceEngine) !void {
        if (self.timestamp_query_pool == null) return error.QueryPoolCreateFailed;
        self.profile_enabled = true;
        log.debug("Profiling enabled: timestamp period={d:.2}ns", .{self.timestamp_period_ns});
    }

    /// Enable the expensive CPU-vs-GPU validation readbacks used for debugging kernel correctness.
    pub fn enableValidationDiagnostics(self: *InferenceEngine) void {
        self.validation_diagnostics_enabled = true;
    }

    /// Preserve full logits on the host for debug dumps and diagnostic inspection.
    pub fn enableLogitsReadback(self: *InferenceEngine) void {
        self.logits_readback_enabled = true;
    }

    /// Write a timestamp to the query pool (if profiling enabled).
    fn writeTimestamp(self: *InferenceEngine, stage: vk.c.VkPipelineStageFlags) ?u32 {
        if (!self.profile_enabled) return null;
        const idx = self.timestamp_count;
        if (idx >= 2048) return null;
        vk.c.vkCmdWriteTimestamp(self.decode_cmd.handle, stage, self.timestamp_query_pool, idx);
        self.timestamp_count = idx + 1;
        return idx;
    }

    /// Reset timestamp counter for a new token.
    fn resetTimestamps(self: *InferenceEngine) void {
        if (!self.profile_enabled) return;
        self.timestamp_count = 0;
        self.profile_phase_range_count = 0;
        self.profile_token_counters.reset();
        vk.c.vkCmdResetQueryPool(self.decode_cmd.handle, self.timestamp_query_pool, 0, 2048);
    }

    fn beginProfilePhase(self: *InferenceEngine) ?u32 {
        return self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
    }

    fn endProfilePhase(self: *InferenceEngine, phase: ProfilePhase, start_query: ?u32) void {
        if (!self.profile_enabled) return;
        const start_idx = start_query orelse return;
        const end_idx = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT) orelse return;
        if (self.profile_phase_range_count >= max_profile_phase_ranges) return;
        self.profile_phase_ranges[self.profile_phase_range_count] = .{
            .phase = phase,
            .start_query = @intCast(start_idx),
            .end_query = @intCast(end_idx),
        };
        self.profile_phase_range_count += 1;
    }

    fn resetProfilingSamples(self: *InferenceEngine) void {
        self.profile_total_gpu_ms = 0.0;
        self.profile_max_gpu_ms = 0.0;
        self.profile_sample_count = 0;
        self.profile_total_cpu_embed_ns = 0;
        self.profile_total_cpu_record_ns = 0;
        self.profile_total_submit_wait_ns = 0;
        self.profile_total_query_read_ns = 0;
        self.profile_max_cpu_record_ns = 0;
        self.profile_max_submit_wait_ns = 0;
        self.profile_token_counters.reset();
        self.profile_total_counters.reset();
        self.profile_phase_range_count = 0;
        self.profile_logged_cpu_moe_fallback = false;
        self.fa_per_layer_ns = [_]u64{0} ** 128;
        self.fa_per_layer_count = [_]u32{0} ** 128;
    }

    fn avgProfilePhaseMs(self: *const InferenceEngine, phase: ProfilePhase) f64 {
        if (self.profile_sample_count == 0) return 0.0;
        return @as(f64, @floatFromInt(self.profile_total_counters.gpu_phase_ns[@intFromEnum(phase)])) /
            @as(f64, @floatFromInt(self.profile_sample_count)) /
            1_000_000.0;
    }

    fn freeActiveKvPages(self: *InferenceEngine) void {
        if (self.active_kv_request_id) |request_id| {
            self.kv_page_pool.freePages(request_id);
            self.active_kv_request_id = null;
        }
        if (self.active_kv_page_ids) |page_ids| {
            self.allocator.free(page_ids);
            self.active_kv_page_ids = null;
        }
    }

    fn uploadActivePageTable(self: *InferenceEngine, page_ids: []const u32) !void {
        const staging_u32: [*]u32 = @ptrCast(@alignCast(self.page_table_staging.mapped.?));
        @memcpy(staging_u32[0..page_ids.len], page_ids);
        try buffer_mod.copyBuffer(
            self.instance,
            self.cmd_pool.handle,
            &self.page_table_staging,
            &self.page_table_buf,
            @as(vk.c.VkDeviceSize, page_ids.len) * @sizeOf(u32),
        );
    }

    fn normalizeRequestedContext(self: *const InferenceEngine, requested_context_tokens: u32, minimum_tokens: u32) u32 {
        const floor = if (minimum_tokens > 0) minimum_tokens else @as(u32, 1);
        const desired = if (requested_context_tokens > floor) requested_context_tokens else floor;
        return @min(desired, self.max_context_tokens);
    }

    /// Grow the 10 batched-prefill scratch buffers so each one can hold
    /// `n_tokens × dim × 4 bytes` of f32 state. No-op when current capacity
    /// already covers `n_tokens`. Called once per `prefillBatched` entry.
    /// Dimensions match the same layout Metal's `BatchedPrefillScratch`
    /// uses: hidden_dim for hidden/norm/down, q_dim for q/attn_out, kv_dim
    /// for k/v, inter_dim for gate/up/swiglu.
    fn ensureBatchedScratchCapacity(self: *InferenceEngine, n_tokens: u32) !void {
        if (n_tokens <= self.batched_scratch_capacity_tokens) return;

        const cfg = &self.model.config;
        const hidden_dim = cfg.hidden_dim;
        const q_dim: u32 = cfg.n_heads * cfg.head_dim;
        const kv_dim: u32 = cfg.n_kv_heads * cfg.head_dim;
        const inter_dim: u32 = if (cfg.intermediate_dim > 0) cfg.intermediate_dim else hidden_dim * 4;

        const n: u64 = n_tokens;
        const f32_sz: u64 = @sizeOf(f32);
        const hidden_bytes = n * hidden_dim * f32_sz;
        const q_bytes = n * q_dim * f32_sz;
        const kv_bytes = n * kv_dim * f32_sz;
        const inter_bytes = n * inter_dim * f32_sz;

        const storage_xfer = vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
            vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT;

        // Helper: free-and-reallocate a slot to at least `size` bytes.
        const growSlot = struct {
            fn run(instance: *const @import("../vulkan/instance.zig").Instance, slot: *?Buffer, size: u64, usage: u32) !void {
                if (slot.*) |*existing| {
                    if (existing.size >= size) return;
                    existing.deinit();
                }
                slot.* = try Buffer.initDeviceLocal(instance, size, usage);
            }
        }.run;

        try growSlot(self.instance, &self.batched_scratch_hidden, hidden_bytes, storage_xfer);
        try growSlot(self.instance, &self.batched_scratch_norm, hidden_bytes, storage_xfer);
        try growSlot(self.instance, &self.batched_scratch_q, q_bytes, storage_xfer);
        try growSlot(self.instance, &self.batched_scratch_k, kv_bytes, storage_xfer);
        try growSlot(self.instance, &self.batched_scratch_v, kv_bytes, storage_xfer);
        try growSlot(self.instance, &self.batched_scratch_attn_out, q_bytes, storage_xfer);
        try growSlot(self.instance, &self.batched_scratch_gate, inter_bytes, storage_xfer);
        try growSlot(self.instance, &self.batched_scratch_up, inter_bytes, storage_xfer);
        try growSlot(self.instance, &self.batched_scratch_swiglu, inter_bytes, storage_xfer);
        try growSlot(self.instance, &self.batched_scratch_down, hidden_bytes, storage_xfer);

        self.batched_scratch_capacity_tokens = n_tokens;
    }

    fn ensureKvPagesForContext(self: *InferenceEngine, target_context_tokens: u32) !void {
        const normalized_context = self.normalizeRequestedContext(target_context_tokens, 1);
        const required_pages = kvPageCountForContext(normalized_context);
        if (required_pages == 0) return error.ContextLengthDoesNotFit;

        if (self.active_kv_page_ids) |existing_pages| {
            const existing_page_count: u32 = @intCast(existing_pages.len);
            if (existing_page_count >= required_pages) return;

            const request_id = self.active_kv_request_id orelse return error.KvPagesNotAllocated;
            const additional_page_count = required_pages - existing_page_count;
            const additional_pages = try self.kv_page_pool.allocPages(request_id, additional_page_count);
            errdefer self.allocator.free(additional_pages);
            sortPageIdsAscending(additional_pages);

            const grown_pages = try self.allocator.alloc(u32, @intCast(required_pages));
            errdefer self.allocator.free(grown_pages);
            @memcpy(grown_pages[0..existing_pages.len], existing_pages);
            @memcpy(grown_pages[existing_pages.len..], additional_pages);

            var clear_request_on_failure = true;
            errdefer if (clear_request_on_failure) {
                self.kv_page_pool.freePages(request_id);
                self.active_kv_request_id = null;
                self.allocator.free(existing_pages);
                self.active_kv_page_ids = null;
            };

            try self.uploadActivePageTable(grown_pages);
            clear_request_on_failure = false;

            self.allocator.free(existing_pages);
            self.allocator.free(additional_pages);
            self.active_kv_page_ids = grown_pages;
            return;
        }

        const request_id = self.next_kv_request_id;
        self.next_kv_request_id += 1;
        const page_ids = try self.kv_page_pool.allocPages(request_id, @intCast(required_pages));
        errdefer {
            self.kv_page_pool.freePages(request_id);
            self.allocator.free(page_ids);
        }
        sortPageIdsAscending(page_ids);
        try self.uploadActivePageTable(page_ids);
        self.active_kv_page_ids = page_ids;
        self.active_kv_request_id = request_id;
    }

    fn physicalTokenIndex(self: *const InferenceEngine, logical_token: u32) !u32 {
        const page_ids = self.active_kv_page_ids orelse return error.KvPagesNotAllocated;
        return logicalTokenToPhysicalToken(page_ids, logical_token);
    }

    fn resetRequestState(self: *InferenceEngine, requested_context_tokens: u32) !void {
        self.freeActiveKvPages();
        try self.ensureKvPagesForContext(requested_context_tokens);

        for (self.ssm_conv_states) |state_buf| {
            if (state_buf.len > 0) @memset(state_buf, 0);
        }
        for (self.ssm_states) |state_buf| {
            if (state_buf.len > 0) @memset(state_buf, 0);
        }
        // Reset circular-buffer offsets so the first dispatch reads slots
        // 0,1,2 in their natural order (matching the zeroed state buffer).
        @memset(self.ssm_conv_state_offsets, 0);

        var has_gpu_ssm = false;
        for (self.gpu_ssm_conv_states) |buf| {
            if (buf.handle != null and buf.size > 0) {
                has_gpu_ssm = true;
                break;
            }
        }
        if (!has_gpu_ssm) return;

        try self.decode_cmd.reset();
        try self.decode_cmd.begin();
        for (self.gpu_ssm_conv_states) |buf| {
            if (buf.handle != null and buf.size > 0) {
                vk.c.vkCmdFillBuffer(self.decode_cmd.handle, buf.handle, 0, buf.size, 0);
            }
        }
        for (self.gpu_ssm_states) |buf| {
            if (buf.handle != null and buf.size > 0) {
                vk.c.vkCmdFillBuffer(self.decode_cmd.handle, buf.handle, 0, buf.size, 0);
            }
        }
        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);
    }

    /// Read back all timestamps for the current token and fold them into request-wide profiling stats.
    pub fn recordProfilingSample(self: *InferenceEngine) void {
        if (!self.profile_enabled or self.timestamp_count == 0) return;
        const count = self.timestamp_count;
        var timestamps: [2048]u64 = undefined;
        const query_read_start = nanoTimestamp();
        const qr = vk.c.vkGetQueryPoolResults(
            self.instance.device,
            self.timestamp_query_pool,
            0,
            count,
            count * @sizeOf(u64),
            &timestamps,
            @sizeOf(u64),
            vk.c.VK_QUERY_RESULT_64_BIT | vk.c.VK_QUERY_RESULT_WAIT_BIT,
        );
        if (qr != vk.c.VK_SUCCESS) {
            log.warn("Failed to read timestamp queries: {d}", .{qr});
            return;
        }
        const query_read_end = nanoTimestamp();
        self.profile_token_counters.query_read_ns += @intCast(query_read_end - query_read_start);
        if (count >= 2) {
            const first = timestamps[0];
            const last = timestamps[count - 1];
            const elapsed_ns = @as(f64, @floatFromInt(last -| first)) * self.timestamp_period_ns;
            const elapsed_ms = elapsed_ns / 1e6;
            self.profile_total_gpu_ms += elapsed_ms;
            if (elapsed_ms > self.profile_max_gpu_ms) self.profile_max_gpu_ms = elapsed_ms;
            self.profile_sample_count += 1;
            // Per-layer flash_attn_kernel accumulator (ZINC_FA_PROFILE_LAYER=1).
            // The Nth flash_attn_kernel range in token order corresponds to
            // layer N (Qwen 3 8B is all-attention; hybrid models still emit a
            // flash_attn_kernel range per attention layer in layer order). We
            // skip prefill ranges by gating on prefill_active so the histogram
            // only reflects decode-time per-layer ms.
            var fa_layer_idx: u32 = 0;
            const fa_record = self.fa_profile_layer and !self.prefill_active;
            for (0..self.profile_phase_range_count) |i| {
                const range = self.profile_phase_ranges[i];
                if (range.end_query >= count or range.start_query >= count) continue;
                const phase_ns_f64 = @as(f64, @floatFromInt(timestamps[range.end_query] -| timestamps[range.start_query])) * self.timestamp_period_ns;
                self.profile_token_counters.gpu_phase_ns[@intFromEnum(range.phase)] += @intFromFloat(@max(phase_ns_f64, 0.0));
                if (fa_record and range.phase == .flash_attn_kernel and fa_layer_idx < self.fa_per_layer_ns.len) {
                    self.fa_per_layer_ns[fa_layer_idx] += @intFromFloat(@max(phase_ns_f64, 0.0));
                    self.fa_per_layer_count[fa_layer_idx] += 1;
                    fa_layer_idx += 1;
                }
            }
            self.profile_total_cpu_embed_ns += self.profile_token_counters.cpu_embed_ns;
            self.profile_total_cpu_record_ns += self.profile_token_counters.cpu_record_ns;
            self.profile_total_submit_wait_ns += self.profile_token_counters.submit_wait_ns;
            self.profile_total_query_read_ns += self.profile_token_counters.query_read_ns;
            if (self.profile_token_counters.cpu_record_ns > self.profile_max_cpu_record_ns) {
                self.profile_max_cpu_record_ns = self.profile_token_counters.cpu_record_ns;
            }
            if (self.profile_token_counters.submit_wait_ns > self.profile_max_submit_wait_ns) {
                self.profile_max_submit_wait_ns = self.profile_token_counters.submit_wait_ns;
            }
            self.profile_total_counters.add(self.profile_token_counters);
            log.debug(
                "PROFILE_TOKEN: gpu={d:.2}ms cpu_embed={d:.2}ms cpu_record={d:.2}ms submit_wait={d:.2}ms query_read={d:.3}ms desc_allocs={d} desc_writes={d}",
                .{
                    elapsed_ms,
                    @as(f64, @floatFromInt(self.profile_token_counters.cpu_embed_ns)) / 1e6,
                    @as(f64, @floatFromInt(self.profile_token_counters.cpu_record_ns)) / 1e6,
                    @as(f64, @floatFromInt(self.profile_token_counters.submit_wait_ns)) / 1e6,
                    @as(f64, @floatFromInt(self.profile_token_counters.query_read_ns)) / 1e6,
                    self.profile_token_counters.descriptor_allocs,
                    self.profile_token_counters.descriptor_write_calls,
                },
            );
        }
    }

    // -----------------------------------------------------------------------
    // Descriptor set helpers
    // -----------------------------------------------------------------------

    /// Allocate a descriptor set from the shared pool with the given layout.
    /// If pool is exhausted (VK_ERROR_OUT_OF_POOL_MEMORY), logs a warning.
    fn allocDescSet(self: *InferenceEngine, layout: vk.c.VkDescriptorSetLayout) !vk.c.VkDescriptorSet {
        const alloc_info = vk.c.VkDescriptorSetAllocateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = self.shared_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        };
        var ds: vk.c.VkDescriptorSet = null;
        const result = vk.c.vkAllocateDescriptorSets(self.instance.device, &alloc_info, &ds);
        if (result == vk.c.VK_ERROR_OUT_OF_POOL_MEMORY or result == vk.c.VK_ERROR_FRAGMENTED_POOL) {
            log.err("Descriptor pool exhausted (4096 sets). Consider increasing pool size or adding mid-batch flush.", .{});
            return error.DescriptorSetAllocFailed;
        }
        if (result != vk.c.VK_SUCCESS) return error.DescriptorSetAllocFailed;
        if (self.profile_enabled) self.profile_token_counters.descriptor_allocs += 1;
        return ds;
    }

    /// Write storage buffer bindings to a descriptor set (up to 8).
    fn writeDescSet3(
        self: *InferenceEngine,
        ds: vk.c.VkDescriptorSet,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer,
        size2: vk.c.VkDeviceSize,
    ) void {
        var buffer_infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
            .{ .buffer = buf2, .offset = 0, .range = size2 },
        };
        var writes: [3]vk.c.VkWriteDescriptorSet = undefined;
        for (0..3) |i| {
            writes[i] = .{
                .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = ds,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buffer_infos[i],
                .pTexelBufferView = null,
            };
        }
        vk.c.vkUpdateDescriptorSets(self.instance.device, 3, &writes, 0, null);
        if (self.profile_enabled) {
            self.profile_token_counters.descriptor_write_calls += 1;
            self.profile_token_counters.descriptor_bindings += 3;
        }
    }

    // -----------------------------------------------------------------------
    // Layer tensor lookup
    // -----------------------------------------------------------------------

    fn findLayerTensor(self: *const InferenceEngine, layer: u32, name: []const u8) ?*const LoadedTensor {
        var buf: [128]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ layer, name }) catch return null;
        return self.tensor_map.get(key);
    }

    // -----------------------------------------------------------------------
    // Descriptor set helpers
    // -----------------------------------------------------------------------

    fn writeDescSet1(
        self: *InferenceEngine,
        ds: vk.c.VkDescriptorSet,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
    ) void {
        var info = vk.c.VkDescriptorBufferInfo{ .buffer = buf0, .offset = 0, .range = size0 };
        const write = vk.c.VkWriteDescriptorSet{
            .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .pNext = null,
            .dstSet = ds,
            .dstBinding = 0,
            .dstArrayElement = 0,
            .descriptorCount = 1,
            .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .pImageInfo = null,
            .pBufferInfo = &info,
            .pTexelBufferView = null,
        };
        vk.c.vkUpdateDescriptorSets(self.instance.device, 1, &write, 0, null);
        if (self.profile_enabled) {
            self.profile_token_counters.descriptor_write_calls += 1;
            self.profile_token_counters.descriptor_bindings += 1;
        }
    }

    fn writeDescSet2(
        self: *InferenceEngine,
        ds: vk.c.VkDescriptorSet,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        size1: vk.c.VkDeviceSize,
    ) void {
        var buffer_infos = [2]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
        };
        var writes: [2]vk.c.VkWriteDescriptorSet = undefined;
        for (0..2) |i| {
            writes[i] = .{
                .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = ds,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buffer_infos[i],
                .pTexelBufferView = null,
            };
        }
        vk.c.vkUpdateDescriptorSets(self.instance.device, 2, &writes, 0, null);
        if (self.profile_enabled) {
            self.profile_token_counters.descriptor_write_calls += 1;
            self.profile_token_counters.descriptor_bindings += 2;
        }
    }

    fn writeDescSet2Offsets(
        self: *InferenceEngine,
        ds: vk.c.VkDescriptorSet,
        buf0: vk.c.VkBuffer,
        offset0: vk.c.VkDeviceSize,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        offset1: vk.c.VkDeviceSize,
        size1: vk.c.VkDeviceSize,
    ) void {
        var buffer_infos = [2]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = offset0, .range = size0 },
            .{ .buffer = buf1, .offset = offset1, .range = size1 },
        };
        var writes: [2]vk.c.VkWriteDescriptorSet = undefined;
        for (0..2) |i| {
            writes[i] = .{
                .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = ds,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buffer_infos[i],
                .pTexelBufferView = null,
            };
        }
        vk.c.vkUpdateDescriptorSets(self.instance.device, 2, &writes, 0, null);
        if (self.profile_enabled) {
            self.profile_token_counters.descriptor_write_calls += 1;
            self.profile_token_counters.descriptor_bindings += 2;
        }
    }

    fn writeDescSet5(
        self: *InferenceEngine,
        ds: vk.c.VkDescriptorSet,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer,
        size2: vk.c.VkDeviceSize,
        buf3: vk.c.VkBuffer,
        size3: vk.c.VkDeviceSize,
        buf4: vk.c.VkBuffer,
        size4: vk.c.VkDeviceSize,
    ) void {
        var buffer_infos = [5]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
            .{ .buffer = buf2, .offset = 0, .range = size2 },
            .{ .buffer = buf3, .offset = 0, .range = size3 },
            .{ .buffer = buf4, .offset = 0, .range = size4 },
        };
        var writes: [5]vk.c.VkWriteDescriptorSet = undefined;
        for (0..5) |i| {
            writes[i] = .{
                .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = ds,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buffer_infos[i],
                .pTexelBufferView = null,
            };
        }
        vk.c.vkUpdateDescriptorSets(self.instance.device, 5, &writes, 0, null);
        if (self.profile_enabled) {
            self.profile_token_counters.descriptor_write_calls += 1;
            self.profile_token_counters.descriptor_bindings += 5;
        }
    }

    fn writeDescSet6(
        self: *InferenceEngine,
        ds: vk.c.VkDescriptorSet,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer,
        size2: vk.c.VkDeviceSize,
        buf3: vk.c.VkBuffer,
        size3: vk.c.VkDeviceSize,
        buf4: vk.c.VkBuffer,
        size4: vk.c.VkDeviceSize,
        buf5: vk.c.VkBuffer,
        size5: vk.c.VkDeviceSize,
    ) void {
        var buffer_infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
            .{ .buffer = buf2, .offset = 0, .range = size2 },
            .{ .buffer = buf3, .offset = 0, .range = size3 },
            .{ .buffer = buf4, .offset = 0, .range = size4 },
            .{ .buffer = buf5, .offset = 0, .range = size5 },
        };
        var writes: [6]vk.c.VkWriteDescriptorSet = undefined;
        for (0..6) |i| {
            writes[i] = .{
                .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = ds,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buffer_infos[i],
                .pTexelBufferView = null,
            };
        }
        vk.c.vkUpdateDescriptorSets(self.instance.device, 6, &writes, 0, null);
        if (self.profile_enabled) {
            self.profile_token_counters.descriptor_write_calls += 1;
            self.profile_token_counters.descriptor_bindings += 6;
        }
    }

    fn writeDescSet4(
        self: *InferenceEngine,
        ds: vk.c.VkDescriptorSet,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer,
        size2: vk.c.VkDeviceSize,
        buf3: vk.c.VkBuffer,
        size3: vk.c.VkDeviceSize,
    ) void {
        var buffer_infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
            .{ .buffer = buf2, .offset = 0, .range = size2 },
            .{ .buffer = buf3, .offset = 0, .range = size3 },
        };
        var writes: [4]vk.c.VkWriteDescriptorSet = undefined;
        for (0..4) |i| {
            writes[i] = .{
                .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = ds,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buffer_infos[i],
                .pTexelBufferView = null,
            };
        }
        vk.c.vkUpdateDescriptorSets(self.instance.device, 4, &writes, 0, null);
        if (self.profile_enabled) {
            self.profile_token_counters.descriptor_write_calls += 1;
            self.profile_token_counters.descriptor_bindings += 4;
        }
    }

    fn writeDescSet7(
        self: *InferenceEngine,
        ds: vk.c.VkDescriptorSet,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer,
        size2: vk.c.VkDeviceSize,
        buf3: vk.c.VkBuffer,
        size3: vk.c.VkDeviceSize,
        buf4: vk.c.VkBuffer,
        size4: vk.c.VkDeviceSize,
        buf5: vk.c.VkBuffer,
        size5: vk.c.VkDeviceSize,
        buf6: vk.c.VkBuffer,
        size6: vk.c.VkDeviceSize,
    ) void {
        var buffer_infos = [7]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
            .{ .buffer = buf2, .offset = 0, .range = size2 },
            .{ .buffer = buf3, .offset = 0, .range = size3 },
            .{ .buffer = buf4, .offset = 0, .range = size4 },
            .{ .buffer = buf5, .offset = 0, .range = size5 },
            .{ .buffer = buf6, .offset = 0, .range = size6 },
        };
        var writes: [7]vk.c.VkWriteDescriptorSet = undefined;
        for (0..7) |i| {
            writes[i] = .{
                .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = ds,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buffer_infos[i],
                .pTexelBufferView = null,
            };
        }
        vk.c.vkUpdateDescriptorSets(self.instance.device, 7, &writes, 0, null);
        if (self.profile_enabled) {
            self.profile_token_counters.descriptor_write_calls += 1;
            self.profile_token_counters.descriptor_bindings += 7;
        }
    }

    fn pushDispatch1(
        self: *InferenceEngine,
        pip: *const Pipeline,
        push_data: []const u8,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) void {
        const infos = [1]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
        };
        self.decode_cmd.pushDescAndDispatch(
            pip,
            self.instance.push_descriptor_fn,
            infos[0..],
            push_data,
            wg_x,
            wg_y,
            wg_z,
        );
    }

    fn pushDispatch2(
        self: *InferenceEngine,
        pip: *const Pipeline,
        push_data: []const u8,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        size1: vk.c.VkDeviceSize,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) void {
        const infos = [2]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
        };
        self.decode_cmd.pushDescAndDispatch(
            pip,
            self.instance.push_descriptor_fn,
            infos[0..],
            push_data,
            wg_x,
            wg_y,
            wg_z,
        );
    }

    fn pushDispatch2Offsets(
        self: *InferenceEngine,
        pip: *const Pipeline,
        push_data: []const u8,
        buf0: vk.c.VkBuffer,
        offset0: vk.c.VkDeviceSize,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        offset1: vk.c.VkDeviceSize,
        size1: vk.c.VkDeviceSize,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) void {
        const infos = [2]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = offset0, .range = size0 },
            .{ .buffer = buf1, .offset = offset1, .range = size1 },
        };
        self.decode_cmd.pushDescAndDispatch(
            pip,
            self.instance.push_descriptor_fn,
            infos[0..],
            push_data,
            wg_x,
            wg_y,
            wg_z,
        );
    }

    fn pushDispatch3(
        self: *InferenceEngine,
        pip: *const Pipeline,
        push_data: []const u8,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer,
        size2: vk.c.VkDeviceSize,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) void {
        const infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
            .{ .buffer = buf2, .offset = 0, .range = size2 },
        };
        self.decode_cmd.pushDescAndDispatch(
            pip,
            self.instance.push_descriptor_fn,
            infos[0..],
            push_data,
            wg_x,
            wg_y,
            wg_z,
        );
    }

    fn pushDispatch4(
        self: *InferenceEngine,
        pip: *const Pipeline,
        push_data: []const u8,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer,
        size2: vk.c.VkDeviceSize,
        buf3: vk.c.VkBuffer,
        size3: vk.c.VkDeviceSize,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) void {
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
            .{ .buffer = buf2, .offset = 0, .range = size2 },
            .{ .buffer = buf3, .offset = 0, .range = size3 },
        };
        self.decode_cmd.pushDescAndDispatch(
            pip,
            self.instance.push_descriptor_fn,
            infos[0..],
            push_data,
            wg_x,
            wg_y,
            wg_z,
        );
    }

    fn pushDispatch5(
        self: *InferenceEngine,
        pip: *const Pipeline,
        push_data: []const u8,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer,
        size2: vk.c.VkDeviceSize,
        buf3: vk.c.VkBuffer,
        size3: vk.c.VkDeviceSize,
        buf4: vk.c.VkBuffer,
        size4: vk.c.VkDeviceSize,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) void {
        const infos = [5]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
            .{ .buffer = buf2, .offset = 0, .range = size2 },
            .{ .buffer = buf3, .offset = 0, .range = size3 },
            .{ .buffer = buf4, .offset = 0, .range = size4 },
        };
        self.decode_cmd.pushDescAndDispatch(
            pip,
            self.instance.push_descriptor_fn,
            infos[0..],
            push_data,
            wg_x,
            wg_y,
            wg_z,
        );
    }

    fn pushDispatch6(
        self: *InferenceEngine,
        pip: *const Pipeline,
        push_data: []const u8,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer,
        size2: vk.c.VkDeviceSize,
        buf3: vk.c.VkBuffer,
        size3: vk.c.VkDeviceSize,
        buf4: vk.c.VkBuffer,
        size4: vk.c.VkDeviceSize,
        buf5: vk.c.VkBuffer,
        size5: vk.c.VkDeviceSize,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) void {
        const infos = [6]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
            .{ .buffer = buf2, .offset = 0, .range = size2 },
            .{ .buffer = buf3, .offset = 0, .range = size3 },
            .{ .buffer = buf4, .offset = 0, .range = size4 },
            .{ .buffer = buf5, .offset = 0, .range = size5 },
        };
        self.decode_cmd.pushDescAndDispatch(
            pip,
            self.instance.push_descriptor_fn,
            infos[0..],
            push_data,
            wg_x,
            wg_y,
            wg_z,
        );
    }

    fn pushDispatch7(
        self: *InferenceEngine,
        pip: *const Pipeline,
        push_data: []const u8,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer,
        size2: vk.c.VkDeviceSize,
        buf3: vk.c.VkBuffer,
        size3: vk.c.VkDeviceSize,
        buf4: vk.c.VkBuffer,
        size4: vk.c.VkDeviceSize,
        buf5: vk.c.VkBuffer,
        size5: vk.c.VkDeviceSize,
        buf6: vk.c.VkBuffer,
        size6: vk.c.VkDeviceSize,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) void {
        const infos = [7]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
            .{ .buffer = buf2, .offset = 0, .range = size2 },
            .{ .buffer = buf3, .offset = 0, .range = size3 },
            .{ .buffer = buf4, .offset = 0, .range = size4 },
            .{ .buffer = buf5, .offset = 0, .range = size5 },
            .{ .buffer = buf6, .offset = 0, .range = size6 },
        };
        self.decode_cmd.pushDescAndDispatch(
            pip,
            self.instance.push_descriptor_fn,
            infos[0..],
            push_data,
            wg_x,
            wg_y,
            wg_z,
        );
    }

    fn pushDispatch8(
        self: *InferenceEngine,
        pip: *const Pipeline,
        push_data: []const u8,
        buf0: vk.c.VkBuffer,
        size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer,
        size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer,
        size2: vk.c.VkDeviceSize,
        buf3: vk.c.VkBuffer,
        size3: vk.c.VkDeviceSize,
        buf4: vk.c.VkBuffer,
        size4: vk.c.VkDeviceSize,
        buf5: vk.c.VkBuffer,
        size5: vk.c.VkDeviceSize,
        buf6: vk.c.VkBuffer,
        size6: vk.c.VkDeviceSize,
        buf7: vk.c.VkBuffer,
        size7: vk.c.VkDeviceSize,
        wg_x: u32,
        wg_y: u32,
        wg_z: u32,
    ) void {
        const infos = [8]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
            .{ .buffer = buf2, .offset = 0, .range = size2 },
            .{ .buffer = buf3, .offset = 0, .range = size3 },
            .{ .buffer = buf4, .offset = 0, .range = size4 },
            .{ .buffer = buf5, .offset = 0, .range = size5 },
            .{ .buffer = buf6, .offset = 0, .range = size6 },
            .{ .buffer = buf7, .offset = 0, .range = size7 },
        };
        self.decode_cmd.pushDescAndDispatch(
            pip,
            self.instance.push_descriptor_fn,
            infos[0..],
            push_data,
            wg_x,
            wg_y,
            wg_z,
        );
    }

    fn dispatchRmsNorm(
        self: *InferenceEngine,
        input_buf: vk.c.VkBuffer,
        input_size: vk.c.VkDeviceSize,
        weight_buf: vk.c.VkBuffer,
        weight_size: vk.c.VkDeviceSize,
        output_buf: vk.c.VkBuffer,
        output_size: vk.c.VkDeviceSize,
        hidden_dim: u32,
        n_tokens: u32,
        eps: f32,
    ) !void {
        const pip = &(self.elementwise.pipeline_rms_norm orelse return error.ShaderNotLoaded);
        if (pip.uses_push_descriptors) {
            const push = RmsNormPush{
                .N = hidden_dim,
                .eps_bits = @bitCast(eps),
            };
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                input_buf,
                input_size,
                weight_buf,
                weight_size,
                output_buf,
                output_size,
                n_tokens,
                1,
                1,
            );
            return;
        }

        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds, input_buf, input_size, weight_buf, weight_size, output_buf, output_size);
        try self.elementwise.recordRmsNorm(&self.decode_cmd, ds, hidden_dim, n_tokens, eps);
    }

    fn dispatchRmsNormStoreHidden(
        self: *InferenceEngine,
        hidden_buf: vk.c.VkBuffer,
        hidden_size: vk.c.VkDeviceSize,
        weight_buf: vk.c.VkBuffer,
        weight_size: vk.c.VkDeviceSize,
        norm_out_buf: vk.c.VkBuffer,
        norm_out_offset: vk.c.VkDeviceSize,
        norm_out_size: vk.c.VkDeviceSize,
        hidden_out_buf: vk.c.VkBuffer,
        hidden_out_offset: vk.c.VkDeviceSize,
        hidden_out_size: vk.c.VkDeviceSize,
        hidden_dim: u32,
        eps: f32,
    ) !void {
        const pip = &(self.elementwise.pipeline_rms_norm_store_hidden orelse return error.ShaderNotLoaded);
        if (!pip.uses_push_descriptors or self.instance.push_descriptor_fn == null) return error.ShaderNotLoaded;
        const push = RmsNormPush{
            .N = hidden_dim,
            .eps_bits = @bitCast(eps),
        };
        const infos = [4]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = hidden_buf, .offset = 0, .range = hidden_size },
            .{ .buffer = weight_buf, .offset = 0, .range = weight_size },
            .{ .buffer = norm_out_buf, .offset = norm_out_offset, .range = norm_out_size },
            .{ .buffer = hidden_out_buf, .offset = hidden_out_offset, .range = hidden_out_size },
        };
        self.decode_cmd.pushDescAndDispatch(
            pip,
            self.instance.push_descriptor_fn,
            infos[0..],
            std.mem.asBytes(&push),
            1,
            1,
            1,
        );
    }

    /// Fused RMS norm + f32 router DMMV. Reads `hidden_buf`, normalizes
    /// it once with `ffn_norm_w`, writes the normalized vector to
    /// `ffn_norm_buf` (so downstream MoE expert dispatches can consume
    /// it), and produces the router DMMV output (M=n_experts) in
    /// `router_logits_buf` — all in a single dispatch. Replaces the
    /// (rms_norm_mul → router DMMV) pair on the per-MoE-layer hot path.
    fn dispatchRmsNormDmmvF32(
        self: *InferenceEngine,
        hidden_buf: vk.c.VkBuffer,
        hidden_size: vk.c.VkDeviceSize,
        ffn_norm_w_buf: vk.c.VkBuffer,
        ffn_norm_w_size: vk.c.VkDeviceSize,
        router_w_buf: vk.c.VkBuffer,
        router_w_size: vk.c.VkDeviceSize,
        ffn_norm_out_buf: vk.c.VkBuffer,
        ffn_norm_out_size: vk.c.VkDeviceSize,
        router_logits_buf: vk.c.VkBuffer,
        router_logits_size: vk.c.VkDeviceSize,
        m: u32,
        k: u32,
        eps: f32,
    ) !void {
        const pip = &(self.elementwise.pipeline_rms_norm_dmmv_f32 orelse return error.ShaderNotLoaded);
        const push = elementwise_mod.RmsNormDmmvF32Push{
            .M = m,
            .K = k,
            .eps_bits = @bitCast(eps),
        };
        // NUM_ROWS=1 in rms_norm_dmmv_f32.comp → one router row per WG.
        // Matches the shader's WG-per-row layout for the small-M router case
        // (n_experts=128 → 128 WGs vs the prior 64 WGs at NUM_ROWS=2).
        const wg_x: u32 = m;
        if (pip.uses_push_descriptors) {
            self.pushDispatch5(
                pip,
                std.mem.asBytes(&push),
                hidden_buf,
                hidden_size,
                ffn_norm_w_buf,
                ffn_norm_w_size,
                router_w_buf,
                router_w_size,
                ffn_norm_out_buf,
                ffn_norm_out_size,
                router_logits_buf,
                router_logits_size,
                wg_x,
                1,
                1,
            );
            return;
        }
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet5(
            ds,
            hidden_buf,
            hidden_size,
            ffn_norm_w_buf,
            ffn_norm_w_size,
            router_w_buf,
            router_w_size,
            ffn_norm_out_buf,
            ffn_norm_out_size,
            router_logits_buf,
            router_logits_size,
        );
        self.decode_cmd.dispatchWithPush(pip, ds, std.mem.asBytes(&push), wg_x, 1, 1);
    }

    /// Fused RMS norm + Q4_K alpha/beta SSM proj DMMV.
    /// Folds the per-SSM-layer (rms_norm_mul → alpha DMMV → beta DMMV)
    /// trio into a single dispatch. WG 0 also writes norm_buf so the
    /// downstream wqkv/z DMMVs see a pre-normalized hidden vector.
    /// Requires push_descriptors and 7 bindings.
    fn dispatchRmsNormDmmvQ4kAlphaBeta(
        self: *InferenceEngine,
        hidden_buf: vk.c.VkBuffer,
        hidden_size: vk.c.VkDeviceSize,
        attn_norm_w_buf: vk.c.VkBuffer,
        attn_norm_w_size: vk.c.VkDeviceSize,
        alpha_w_buf: vk.c.VkBuffer,
        alpha_w_size: vk.c.VkDeviceSize,
        beta_w_buf: vk.c.VkBuffer,
        beta_w_size: vk.c.VkDeviceSize,
        norm_out_buf: vk.c.VkBuffer,
        norm_out_size: vk.c.VkDeviceSize,
        alpha_out_buf: vk.c.VkBuffer,
        alpha_out_size: vk.c.VkDeviceSize,
        beta_out_buf: vk.c.VkBuffer,
        beta_out_size: vk.c.VkDeviceSize,
        m: u32,
        k: u32,
        eps: f32,
    ) !void {
        const pip = &(self.elementwise.pipeline_rms_norm_dmmv_q4k_alpha_beta orelse return error.ShaderNotLoaded);
        if (!pip.uses_push_descriptors) return error.ShaderNotLoaded;
        const push = elementwise_mod.RmsNormDmmvQ4kAlphaBetaPush{
            .M = m,
            .K = k,
            .eps_bits = @bitCast(eps),
        };
        const wg_x: u32 = m;
        self.pushDispatch7(
            pip,
            std.mem.asBytes(&push),
            hidden_buf,
            hidden_size,
            attn_norm_w_buf,
            attn_norm_w_size,
            alpha_w_buf,
            alpha_w_size,
            beta_w_buf,
            beta_w_size,
            norm_out_buf,
            norm_out_size,
            alpha_out_buf,
            alpha_out_size,
            beta_out_buf,
            beta_out_size,
            wg_x,
            1,
            1,
        );
    }

    fn dispatchSigmoidMul(
        self: *InferenceEngine,
        input_buf: vk.c.VkBuffer,
        input_size: vk.c.VkDeviceSize,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        output_buf: vk.c.VkBuffer,
        output_size: vk.c.VkDeviceSize,
        n_elements: u32,
    ) !void {
        const pip = &(self.elementwise.pipeline_sigmoid_mul orelse return error.ShaderNotLoaded);
        if (pip.uses_push_descriptors) {
            const push = SigmoidMulPush{ .N = n_elements };
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                input_buf,
                input_size,
                gate_buf,
                gate_size,
                output_buf,
                output_size,
                (n_elements + 63) / 64,
                1,
                1,
            );
            return;
        }

        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds, input_buf, input_size, gate_buf, gate_size, output_buf, output_size);
        try self.elementwise.recordSigmoidMul(&self.decode_cmd, ds, n_elements);
    }

    fn dispatchVadd(
        self: *InferenceEngine,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        output_buf: vk.c.VkBuffer,
        output_size: vk.c.VkDeviceSize,
        n_elements: u32,
    ) !void {
        const pip = &(self.elementwise.pipeline_vadd orelse return error.ShaderNotLoaded);
        const VaddPushLocal = extern struct { N: u32 };
        if (pip.uses_push_descriptors) {
            const push = VaddPushLocal{ .N = n_elements };
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                a_buf,
                a_size,
                b_buf,
                b_size,
                output_buf,
                output_size,
                (n_elements + 63) / 64,
                1,
                1,
            );
            return;
        }
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds, a_buf, a_size, b_buf, b_size, output_buf, output_size);
        try self.elementwise.recordVadd(&self.decode_cmd, ds, n_elements);
    }

    /// Dispatch the correct FFN activation (SwiGLU for most models, GEGLU for Gemma).
    fn dispatchFfnActivation(
        self: *InferenceEngine,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        output_buf: vk.c.VkBuffer,
        output_size: vk.c.VkDeviceSize,
        n_elements: u32,
    ) !void {
        if (self.model.config.architecture == .gemma) {
            return self.dispatchGeglu(gate_buf, gate_size, up_buf, up_size, output_buf, output_size, n_elements);
        }
        if (self.model.config.architecture == .gpt_oss) {
            return self.dispatchSwigluOai(gate_buf, gate_size, up_buf, up_size, output_buf, output_size, n_elements);
        }
        return self.dispatchSwiglu(gate_buf, gate_size, up_buf, up_size, output_buf, output_size, n_elements);
    }

    fn dispatchGeglu(
        self: *InferenceEngine,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        output_buf: vk.c.VkBuffer,
        output_size: vk.c.VkDeviceSize,
        n_elements: u32,
    ) !void {
        const pip = &(self.elementwise.pipeline_geglu orelse return error.ShaderNotLoaded);
        if (pip.uses_push_descriptors) {
            const push = SwigluPush{ .N = n_elements };
            self.pushDispatch3(pip, std.mem.asBytes(&push), gate_buf, gate_size, up_buf, up_size, output_buf, output_size, (n_elements + 63) / 64, 1, 1);
            return;
        }
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds, gate_buf, gate_size, up_buf, up_size, output_buf, output_size);
        try self.elementwise.recordGeglu(&self.decode_cmd, ds, n_elements);
    }

    fn dispatchSwiglu(
        self: *InferenceEngine,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        output_buf: vk.c.VkBuffer,
        output_size: vk.c.VkDeviceSize,
        n_elements: u32,
    ) !void {
        const pip = &(self.elementwise.pipeline_swiglu orelse return error.ShaderNotLoaded);
        if (pip.uses_push_descriptors) {
            const push = SwigluPush{ .N = n_elements };
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                gate_buf,
                gate_size,
                up_buf,
                up_size,
                output_buf,
                output_size,
                (n_elements + 63) / 64,
                1,
                1,
            );
            return;
        }

        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds, gate_buf, gate_size, up_buf, up_size, output_buf, output_size);
        try self.elementwise.recordSwiglu(&self.decode_cmd, ds, n_elements);
    }

    fn dispatchSwigluOai(
        self: *InferenceEngine,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        up_buf: vk.c.VkBuffer,
        up_size: vk.c.VkDeviceSize,
        output_buf: vk.c.VkBuffer,
        output_size: vk.c.VkDeviceSize,
        n_elements: u32,
    ) !void {
        const pip = &(self.elementwise.pipeline_swiglu_oai orelse return error.ShaderNotLoaded);
        if (pip.uses_push_descriptors) {
            const push = SwigluPush{ .N = n_elements };
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                gate_buf,
                gate_size,
                up_buf,
                up_size,
                output_buf,
                output_size,
                (n_elements + 63) / 64,
                1,
                1,
            );
            return;
        }

        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds, gate_buf, gate_size, up_buf, up_size, output_buf, output_size);
        try self.elementwise.recordSwigluOai(&self.decode_cmd, ds, n_elements);
    }

    fn dispatchScaleInPlace(
        self: *InferenceEngine,
        buf: vk.c.VkBuffer,
        buf_size: vk.c.VkDeviceSize,
        n_elements: u32,
        scale: f32,
    ) !void {
        const pip = &(self.elementwise.pipeline_scale_in_place orelse return error.ShaderNotLoaded);
        if (pip.uses_push_descriptors) {
            const push = ScaleAccPush{ .N = n_elements, .scale_bits = @bitCast(scale) };
            self.pushDispatch1(pip, std.mem.asBytes(&push), buf, buf_size, (n_elements + 63) / 64, 1, 1);
            return;
        }
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet1(ds, buf, buf_size);
        try self.elementwise.recordScaleInPlace(&self.decode_cmd, ds, n_elements, scale);
    }

    // Element-wise multiply: a[i] *= b[i]. For Gemma 4 ffn_gate_inp.scale on router input.
    fn dispatchMulElementwise(
        self: *InferenceEngine,
        a_buf: vk.c.VkBuffer,
        a_size: vk.c.VkDeviceSize,
        b_buf: vk.c.VkBuffer,
        b_size: vk.c.VkDeviceSize,
        n_elements: u32,
    ) !void {
        const pip = &(self.elementwise.pipeline_mul_elementwise orelse return error.ShaderNotLoaded);
        const MulPush = extern struct { N: u32 };
        const push = MulPush{ .N = n_elements };
        if (pip.uses_push_descriptors) {
            self.pushDispatch2(pip, std.mem.asBytes(&push), a_buf, a_size, b_buf, b_size, (n_elements + 63) / 64, 1, 1);
            return;
        }
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet2(ds, a_buf, a_size, b_buf, b_size);
        self.decode_cmd.dispatchWithPush(pip, ds, std.mem.asBytes(&push), (n_elements + 63) / 64, 1, 1);
    }

    // Per-expert scalar multiply for ffn_down_exps.scale (Gemma 4).
    // down[slot*hidden_dim + i] *= scales[routing[slot]] for slot in 0..n_used, i in 0..hidden_dim.
    fn dispatchPerExpertScale(
        self: *InferenceEngine,
        down_buf: vk.c.VkBuffer,
        down_size: vk.c.VkDeviceSize,
        scales_buf: vk.c.VkBuffer,
        scales_size: vk.c.VkDeviceSize,
        routing_buf: vk.c.VkBuffer,
        routing_size: vk.c.VkDeviceSize,
        hidden_dim: u32,
        n_used: u32,
    ) !void {
        const pip = &(self.elementwise.pipeline_per_expert_scale orelse return error.ShaderNotLoaded);
        const PerExpertPush = extern struct { hidden_dim: u32, n_used: u32 };
        const push = PerExpertPush{ .hidden_dim = hidden_dim, .n_used = n_used };
        const wg_x = (hidden_dim + 63) / 64;
        if (pip.uses_push_descriptors) {
            self.pushDispatch3(pip, std.mem.asBytes(&push), down_buf, down_size, scales_buf, scales_size, routing_buf, routing_size, wg_x, n_used, 1);
            return;
        }
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds, down_buf, down_size, scales_buf, scales_size, routing_buf, routing_size);
        self.decode_cmd.dispatchWithPush(pip, ds, std.mem.asBytes(&push), wg_x, n_used, 1);
    }

    fn dispatchScaleAcc(
        self: *InferenceEngine,
        accum_buf: vk.c.VkBuffer,
        accum_size: vk.c.VkDeviceSize,
        src_buf: vk.c.VkBuffer,
        src_size: vk.c.VkDeviceSize,
        n_elements: u32,
        scale: f32,
    ) !void {
        const pip = &(self.elementwise.pipeline_scale_acc orelse return error.ShaderNotLoaded);
        if (pip.uses_push_descriptors) {
            const push = ScaleAccPush{
                .N = n_elements,
                .scale_bits = @bitCast(scale),
            };
            self.pushDispatch2(
                pip,
                std.mem.asBytes(&push),
                accum_buf,
                accum_size,
                src_buf,
                src_size,
                (n_elements + 255) / 256,
                1,
                1,
            );
            return;
        }

        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet2(ds, accum_buf, accum_size, src_buf, src_size);
        try self.elementwise.recordScaleAcc(&self.decode_cmd, ds, n_elements, scale);
    }

    fn dispatchScaleAccWithOffsets(
        self: *InferenceEngine,
        accum_buf: vk.c.VkBuffer,
        accum_offset: vk.c.VkDeviceSize,
        accum_size: vk.c.VkDeviceSize,
        src_buf: vk.c.VkBuffer,
        src_offset: vk.c.VkDeviceSize,
        src_size: vk.c.VkDeviceSize,
        n_elements: u32,
        scale: f32,
    ) !void {
        const pip = &(self.elementwise.pipeline_scale_acc orelse return error.ShaderNotLoaded);
        if (pip.uses_push_descriptors) {
            const push = ScaleAccPush{
                .N = n_elements,
                .scale_bits = @bitCast(scale),
            };
            self.pushDispatch2Offsets(
                pip,
                std.mem.asBytes(&push),
                accum_buf,
                accum_offset,
                accum_size,
                src_buf,
                src_offset,
                src_size,
                (n_elements + 255) / 256,
                1,
                1,
            );
            return;
        }

        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet2Offsets(ds, accum_buf, accum_offset, accum_size, src_buf, src_offset, src_size);
        try self.elementwise.recordScaleAcc(&self.decode_cmd, ds, n_elements, scale);
    }

    fn dispatchBiasAdd(
        self: *InferenceEngine,
        output_buf: vk.c.VkBuffer,
        output_size: vk.c.VkDeviceSize,
        tensor: *const LoadedTensor,
        n_elements: u32,
    ) !void {
        return self.dispatchBiasAddSlice(output_buf, output_size, tensor, 0, n_elements);
    }

    fn dispatchBiasAddSlice(
        self: *InferenceEngine,
        output_buf: vk.c.VkBuffer,
        output_size: vk.c.VkDeviceSize,
        tensor: *const LoadedTensor,
        element_offset: u32,
        n_elements: u32,
    ) !void {
        if (tensor.info.type_ != .f32) {
            log.err("Unsupported Vulkan bias tensor type {s} for {s}", .{
                @tagName(tensor.info.type_),
                tensor.info.name,
            });
            return error.UnsupportedQuantType;
        }
        const pip = &(self.elementwise.pipeline_bias_add orelse return error.ShaderNotLoaded);
        if (pip.uses_push_descriptors) {
            const push = BiasAddPush{
                .N = n_elements,
                .src_offset = element_offset,
            };
            self.pushDispatch2(
                pip,
                std.mem.asBytes(&push),
                output_buf,
                output_size,
                tensor.gpu_buffer.handle,
                tensor.gpu_buffer.size,
                (n_elements + 63) / 64,
                1,
                1,
            );
            return;
        }

        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet2(ds, output_buf, output_size, tensor.gpu_buffer.handle, tensor.gpu_buffer.size);
        try self.elementwise.recordBiasAdd(&self.decode_cmd, ds, n_elements, element_offset);
    }

    fn dispatchRopeInPlace(
        self: *InferenceEngine,
        buf: vk.c.VkBuffer,
        buf_size: vk.c.VkDeviceSize,
        freq_buf: ?vk.c.VkBuffer,
        freq_size: vk.c.VkDeviceSize,
        stride: u32,
        rope_dim: u32,
        n_heads: u32,
        position: u32,
        freq_base: f32,
        attn_scale: f32,
    ) !void {
        const pip = &(self.elementwise.pipeline_rope orelse return error.ShaderNotLoaded);
        const use_scratch = freq_buf != null and self.attn_out_buf.size >= buf_size;
        const out_buf = if (use_scratch) self.attn_out_buf.handle else buf;
        const out_size = if (use_scratch) self.attn_out_buf.size else buf_size;
        if (pip.uses_push_descriptors) {
            const push = RopePush{
                .stride = stride,
                .rope_dim = rope_dim,
                .n_heads = n_heads,
                .position = position,
                .freq_base_bits = @bitCast(freq_base),
                .attn_scale_bits = @bitCast(attn_scale),
            };
            if (freq_buf) |fb| {
                self.pushDispatch3(
                    pip,
                    std.mem.asBytes(&push),
                    buf,
                    buf_size,
                    out_buf,
                    out_size,
                    fb,
                    freq_size,
                    n_heads,
                    1,
                    1,
                );
            } else {
                self.pushDispatch2(
                    pip,
                    std.mem.asBytes(&push),
                    buf,
                    buf_size,
                    out_buf,
                    out_size,
                    n_heads,
                    1,
                    1,
                );
            }
            if (use_scratch) {
                self.decode_cmd.computeAndTransferBarrier();
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, out_buf, buf, 1, &vk.c.VkBufferCopy{
                    .srcOffset = 0,
                    .dstOffset = 0,
                    .size = buf_size,
                });
                self.decode_cmd.transferToComputeBarrier();
            }
            return;
        }

        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        if (freq_buf) |fb| {
            self.writeDescSet3(ds, buf, buf_size, out_buf, out_size, fb, freq_size);
        } else {
            self.writeDescSet2(ds, buf, buf_size, out_buf, out_size);
        }
        try self.elementwise.recordRope(&self.decode_cmd, ds, stride, rope_dim, n_heads, position, freq_base, attn_scale);
        if (use_scratch) {
            self.decode_cmd.computeAndTransferBarrier();
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, out_buf, buf, 1, &vk.c.VkBufferCopy{
                .srcOffset = 0,
                .dstOffset = 0,
                .size = buf_size,
            });
            self.decode_cmd.transferToComputeBarrier();
        }
    }

    /// Fused RMS norm + RoPE in-place on a head buffer.
    /// Eliminates 1 dispatch + 1 barrier vs separate norm then RoPE.
    /// Fused Q+K norm + RoPE + KV cache write. One dispatch absorbs:
    ///   - Q per-head RMS norm + RoPE (in-place on q_buf)
    ///   - K per-head RMS norm + RoPE (writes directly to kv_k_cache slot)
    ///   - V copy from v_buf into kv_v_cache slot
    /// Saves 2 dispatches + 1 barrier per attention layer vs the original
    /// (Q norm+rope → K norm+rope → kv_cache_write) trio. Caller is
    /// responsible for ensuring a barrier follows this dispatch before any
    /// consumer of q_buf, kv_k_cache, or kv_v_cache runs.
    fn dispatchQkNormRopeKvWrite(
        self: *InferenceEngine,
        q_buf: vk.c.VkBuffer,
        q_size: vk.c.VkDeviceSize,
        q_weight_buf: vk.c.VkBuffer,
        q_weight_size: vk.c.VkDeviceSize,
        k_buf: vk.c.VkBuffer,
        k_size: vk.c.VkDeviceSize,
        k_weight_buf: vk.c.VkBuffer,
        k_weight_size: vk.c.VkDeviceSize,
        freq_buf: ?vk.c.VkBuffer,
        freq_size: vk.c.VkDeviceSize,
        kv_k_buf: vk.c.VkBuffer,
        kv_k_size: vk.c.VkDeviceSize,
        v_buf: vk.c.VkBuffer,
        v_size: vk.c.VkDeviceSize,
        kv_v_buf: vk.c.VkBuffer,
        kv_v_size: vk.c.VkDeviceSize,
        head_dim: u32,
        rope_dim: u32,
        n_q_heads: u32,
        n_k_heads: u32,
        position: u32,
        freq_base: f32,
        attn_scale: f32,
        eps: f32,
        dst_offset_floats: u32,
        v_norm: bool,
    ) void {
        const pip = &(self.elementwise.pipeline_qk_norm_rope_kv_write orelse return);
        const push = QkNormRopeKvWritePush{
            .head_dim = head_dim,
            .rope_dim = rope_dim,
            .n_q_heads = n_q_heads,
            .n_k_heads = n_k_heads,
            .position = position,
            .freq_base_bits = @bitCast(freq_base),
            .attn_scale_bits = @bitCast(attn_scale),
            .eps_bits = @bitCast(eps),
            .dst_offset = dst_offset_floats,
            .v_norm = if (v_norm) 1 else 0,
        };
        // Bind a dummy buffer for the unused freq binding when no precomputed
        // freq buffer is supplied. Mirrors dispatchNormRopeInPlace's pattern.
        const fb_handle: vk.c.VkBuffer = if (freq_buf) |fb| fb else q_buf;
        const fb_size: vk.c.VkDeviceSize = if (freq_buf) |_| freq_size else q_size;
        self.pushDispatch8(
            pip,
            std.mem.asBytes(&push),
            q_buf,
            q_size,
            q_weight_buf,
            q_weight_size,
            k_buf,
            k_size,
            k_weight_buf,
            k_weight_size,
            fb_handle,
            fb_size,
            kv_k_buf,
            kv_k_size,
            v_buf,
            v_size,
            kv_v_buf,
            kv_v_size,
            n_q_heads + n_k_heads,
            1,
            1,
        );
    }

    fn dispatchNormRopeInPlace(
        self: *InferenceEngine,
        buf: vk.c.VkBuffer,
        buf_size: vk.c.VkDeviceSize,
        weight_buf: vk.c.VkBuffer,
        weight_size: vk.c.VkDeviceSize,
        freq_buf: ?vk.c.VkBuffer,
        freq_size: vk.c.VkDeviceSize,
        head_dim: u32,
        rope_dim: u32,
        n_heads: u32,
        position: u32,
        freq_base: f32,
        attn_scale: f32,
        eps: f32,
    ) void {
        const pip = &(self.elementwise.pipeline_norm_rope orelse return);
        const push = NormRopePush{
            .head_dim = head_dim,
            .rope_dim = rope_dim,
            .n_heads = n_heads,
            .position = position,
            .freq_base_bits = @bitCast(freq_base),
            .attn_scale_bits = @bitCast(attn_scale),
            .eps_bits = @bitCast(eps),
        };
        if (freq_buf) |fb| {
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                buf,
                buf_size,
                weight_buf,
                weight_size,
                fb,
                freq_size,
                n_heads,
                1,
                1,
            );
        } else {
            // Bind a dummy buffer for the unused freq binding
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                buf,
                buf_size,
                weight_buf,
                weight_size,
                buf,
                buf_size,
                n_heads,
                1,
                1,
            );
        }
    }

    fn dispatchSoftmaxTopk(
        self: *InferenceEngine,
        logits_buf: vk.c.VkBuffer,
        logits_size: vk.c.VkDeviceSize,
        output_buf: vk.c.VkBuffer,
        output_size: vk.c.VkDeviceSize,
        n_experts: u32,
        k: u32,
    ) !void {
        return self.dispatchSoftmaxTopkScaled(
            logits_buf,
            logits_size,
            output_buf,
            output_size,
            n_experts,
            k,
            1.0,
        );
    }

    fn dispatchSoftmaxTopkScaled(
        self: *InferenceEngine,
        logits_buf: vk.c.VkBuffer,
        logits_size: vk.c.VkDeviceSize,
        output_buf: vk.c.VkBuffer,
        output_size: vk.c.VkDeviceSize,
        n_experts: u32,
        k: u32,
        scale: f32,
    ) !void {
        const use_v2 = self.use_softmax_topk_v2 and self.elementwise.pipeline_softmax_topk_v2 != null;
        if (scale != 1.0 and !use_v2) return error.ShaderNotLoaded;
        const pip = if (use_v2)
            &self.elementwise.pipeline_softmax_topk_v2.?
        else
            &(self.elementwise.pipeline_softmax_topk orelse return error.ShaderNotLoaded);
        const push = SoftmaxTopkPush{
            .n_experts = n_experts,
            .k = k,
            .scale_bits = @bitCast(scale),
        };
        if (pip.uses_push_descriptors) {
            self.pushDispatch2(
                pip,
                std.mem.asBytes(&push),
                logits_buf,
                logits_size,
                output_buf,
                output_size,
                1,
                1,
                1,
            );
            return;
        }

        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet2(ds, logits_buf, logits_size, output_buf, output_size);
        self.decode_cmd.dispatchWithPush(pip, ds, std.mem.asBytes(&push), 1, 1, 1);
    }

    fn dispatchMoeWeightedAcc(
        self: *InferenceEngine,
        accum_buf: vk.c.VkBuffer,
        accum_size: vk.c.VkDeviceSize,
        src_buf: vk.c.VkBuffer,
        src_size: vk.c.VkDeviceSize,
        routing_buf: vk.c.VkBuffer,
        routing_size: vk.c.VkDeviceSize,
        n_elements: u32,
        n_used: u32,
        src_stride: u32,
    ) !void {
        const pip = &(self.elementwise.pipeline_moe_weighted_acc orelse return error.ShaderNotLoaded);
        if (pip.uses_push_descriptors) {
            const push = MoeWeightedAccPush{
                .N = n_elements,
                .n_used = n_used,
                .src_stride = src_stride,
            };
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                accum_buf,
                accum_size,
                src_buf,
                src_size,
                routing_buf,
                routing_size,
                (n_elements + 63) / 64,
                1,
                1,
            );
            return;
        }

        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds, accum_buf, accum_size, src_buf, src_size, routing_buf, routing_size);
        try self.elementwise.recordMoeWeightedAcc(&self.decode_cmd, ds, n_elements, n_used, src_stride);
    }

    fn dispatchSigmoidScaleAcc(
        self: *InferenceEngine,
        accum_buf: vk.c.VkBuffer,
        accum_size: vk.c.VkDeviceSize,
        src_buf: vk.c.VkBuffer,
        src_size: vk.c.VkDeviceSize,
        gate_buf: vk.c.VkBuffer,
        gate_size: vk.c.VkDeviceSize,
        n_elements: u32,
    ) !void {
        const pip = &(self.elementwise.pipeline_sigmoid_scale_acc orelse return error.ShaderNotLoaded);
        if (pip.uses_push_descriptors) {
            const push = ScaleAccPush{
                .N = n_elements,
                .scale_bits = 0,
            };
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                accum_buf,
                accum_size,
                src_buf,
                src_size,
                gate_buf,
                gate_size,
                (n_elements + 255) / 256,
                1,
                1,
            );
            return;
        }

        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds, accum_buf, accum_size, src_buf, src_size, gate_buf, gate_size);
        try self.elementwise.recordSigmoidScaleAcc(&self.decode_cmd, ds, n_elements);
    }

    // -----------------------------------------------------------------------
    // Embedding
    // -----------------------------------------------------------------------

    /// Dequantize a token's embedding row directly into the pre-allocated staging buffer.
    /// The GPU copy (staging → hidden_buf) is recorded in the decode command buffer.
    fn embedToken(self: *InferenceEngine, token_id: u32) !void {
        const hidden_dim = self.model.config.hidden_dim;

        // Prefill fast path: prefillBatch() dequantized the whole prompt
        // directly into prefill_embed_big. decodeStep's layer-0 copy reads
        // from that buffer with srcOffset = prefill_current_token_idx *
        // hidden_size, so there is nothing to do here — the CPU record path
        // skips a per-token memcpy entirely.
        if (self.prefill_active and
            self.prefill_embed_big != null and
            self.prefill_embed_big_hidden == hidden_dim and
            self.prefill_current_token_idx < self.prefill_embed_big_token_count)
        {
            return;
        }

        const staging_f32: [*]f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
        const safe_id = @min(token_id, self.model.config.vocab_size -| 1);

        const embd = self.tensor_map.get("token_embd.weight") orelse {
            log.err("token_embd.weight not found", .{});
            return error.TensorNotFound;
        };

        const mmap = self.model.mmap_data orelse return error.NoMmapData;
        const data_start: usize = @intCast(self.model.gguf_file.tensor_data_offset + embd.info.offset);

        // Dequantize directly into pre-allocated staging buffer (zero alloc)
        dequantRow(mmap[data_start..], safe_id, hidden_dim, embd.info.type_, staging_f32[0..hidden_dim]);

        // Gemma models scale embeddings by sqrt(hidden_dim).
        if (self.model.config.architecture == .gemma) {
            const scale: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(hidden_dim))));
            for (staging_f32[0..hidden_dim]) |*v| v.* *= scale;
        }
    }

    // -----------------------------------------------------------------------
    // Decode step
    // -----------------------------------------------------------------------

    /// Run a single decode step through all transformer layers.
    /// embed → [per-layer: norm → QKV → RoPE → KV write → attention → O proj → residual
    ///          → FFN norm → MoE routing → expert DMMVs → residual] → final norm → LM head → logits
    pub fn decodeStep(self: *InferenceEngine, state: *DecodeState, token_id: u32, collect_output: bool) !void {
        if (state.position >= self.max_context_tokens) {
            return error.ContextLengthExceeded;
        }
        const next_token_target = if (state.requested_context_tokens > 0)
            @max(state.requested_context_tokens, state.position + 1)
        else
            state.position + 1;
        try self.ensureKvPagesForContext(next_token_target);
        const config = &self.model.config;
        const hidden_dim = config.hidden_dim;
        const hidden_size = @as(vk.c.VkDeviceSize, hidden_dim) * @sizeOf(f32);
        const rms_norm_eps = config.rms_norm_eps;
        const q_dim = @as(u32, config.n_heads) * config.head_dim;
        const kv_dim = @as(u32, config.n_kv_heads) * config.head_dim;
        // kv_dim is only used for buffer allocation; per-layer kv_dim (layer_kv_dim)
        // is computed from tensor shapes for dispatch.
        const kv_vec_size = @as(vk.c.VkDeviceSize, kv_dim) * @sizeOf(f32);
        _ = kv_vec_size;
        const is_moe = config.n_experts > 0;
        const inter_dim = if (config.intermediate_dim > 0) config.intermediate_dim else hidden_dim * 4;
        const shexp_inter_dim = if (config.shared_expert_intermediate_dim > 0) config.shared_expert_intermediate_dim else inter_dim;
        // Hybrid models: every Nth layer is full attention, rest are SSM/linear attention
        const full_attn_interval = if (config.full_attn_interval > 0) config.full_attn_interval else 1;
        const layer_start = @min(self.partial_decode_start_layer, config.n_layers);
        const requested_layer_end = if (self.partial_decode_end_layer == 0) config.n_layers else self.partial_decode_end_layer;
        const layer_end = @min(@max(requested_layer_end, layer_start), config.n_layers);
        const has_partial_hidden_in = self.partial_decode_hidden_in != null;
        const has_partial_hidden_out = self.partial_decode_hidden_out != null;
        const partial_layer_decode = layer_start != 0 or
            layer_end != config.n_layers or
            has_partial_hidden_in or
            has_partial_hidden_out or
            !self.partial_decode_advance_position;
        var partial_hidden_out_written_by_stop = false;

        // Log MoE dimensions once (first decode)
        if (state.generated_tokens.items.len == 0 and is_moe) {
            log.debug("MoE dims: expert_inter={d} shared_expert_inter={d} hidden={d}", .{ inter_dim, shexp_inter_dim, hidden_dim });
        }

        // 1. CPU: dequantize embedding
        const track_decode_timing = self.profile_enabled or self.prefill_active;
        const cpu_embed_start = if (track_decode_timing) nanoTimestamp() else 0;
        if (!has_partial_hidden_in) {
            try self.embedToken(token_id);
        }
        if (!has_partial_hidden_in and collect_output and state.generated_tokens.items.len == 0 and config.architecture == .gpt_oss) {
            const embd = self.tensor_map.get("token_embd.weight") orelse return error.TensorNotFound;
            const staging_f32: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
            log.info("EMBED_CHECK pos={d} token={d}: type={s} emb[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}]", .{
                state.position,
                token_id,
                @tagName(embd.info.type_),
                staging_f32[0],
                staging_f32[1],
                staging_f32[2],
                staging_f32[3],
            });
        }
        var prefill_embed_elapsed_ns: u64 = 0;
        if (track_decode_timing) {
            const cpu_embed_end = nanoTimestamp();
            const elapsed: u64 = @intCast(cpu_embed_end - cpu_embed_start);
            if (self.profile_enabled) self.profile_token_counters.cpu_embed_ns += elapsed;
            prefill_embed_elapsed_ns = elapsed;
        }

        // Per-layer logit5 tracking for BOS diagnostic summary
        var diag_logit5 = [_]f32{0} ** 64;
        var diag_rms_arr = [_]f32{0} ** 64;

        const cpu_record_start = if (track_decode_timing) nanoTimestamp() else 0;

        // Begin single command buffer for all layers (Phase 3c batching)
        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.beginOneTime();

        // Pipelined prefill: the previous prompt token was submitted without a
        // host fence wait, so its compute writes to shared device state (KV
        // cache, GPU SSM state) are not guaranteed to be visible to this CB's
        // dispatches. Queue submission order enforces execution order, but not
        // memory visibility — add an explicit compute→compute barrier.
        if (self.prefill_pipeline_mode) self.decode_cmd.computeBarrier();

        if (self.partial_decode_hidden_in) |hidden_in| {
            const region = vk.c.VkBufferCopy{
                .srcOffset = self.partial_decode_hidden_in_offset,
                .dstOffset = 0,
                .size = hidden_size,
            };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, hidden_in, self.hidden_buf.handle, 1, &region);
            self.decode_cmd.transferToComputeBarrier();
        }

        // Reset profiling timestamps for this token
        self.resetTimestamps();
        _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

        for (layer_start..layer_end) |layer_idx| {
            const layer: u32 = @intCast(layer_idx);
            const lt = self.layer_tensors[layer_idx];

            // --- Upload embedding (only first layer) ---
            if (layer == 0 and !has_partial_hidden_in) {
                const embed_phase = self.beginProfilePhase();
                // During prefill, prefillBatch pre-dequantized every prompt
                // embedding row into prefill_embed_big. Read from there with
                // a per-token srcOffset so embedToken's per-token memcpy into
                // embed_staging is redundant and can be skipped. For decode
                // and any path where prefill_embed_big is not populated the
                // copy still comes from embed_staging as before.
                if (self.prefill_active and
                    self.prefill_embed_big != null and
                    self.prefill_embed_big_hidden == hidden_dim and
                    self.prefill_current_token_idx < self.prefill_embed_big_token_count)
                {
                    const src_offset: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, self.prefill_current_token_idx) * hidden_size;
                    const region = vk.c.VkBufferCopy{ .srcOffset = src_offset, .dstOffset = 0, .size = hidden_size };
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.prefill_embed_big.?.handle, self.hidden_buf.handle, 1, &region);
                } else {
                    const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = hidden_size };
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.embed_staging.handle, self.hidden_buf.handle, 1, &region);
                }
                self.decode_cmd.transferToComputeBarrier();
                self.endProfilePhase(.embed_upload, embed_phase);
            }

            // --- Input RMS norm: hidden_buf → norm_buf ---
            const attn_norm = lt.attn_norm orelse {
                log.err("Layer {d}: attn_norm.weight not found", .{layer});
                return error.TensorNotFound;
            };
            const is_full_attn = ((layer + 1) % full_attn_interval == 0);
            const diag_last_prompt_token = collect_output and state.generated_tokens.items.len == 0 and config.architecture == .gpt_oss;

            // Fused SSM pre-norm fast path: collapse the per-SSM-layer
            // (rms_norm_mul → alpha DMMV → beta DMMV) trio into a single
            // dispatch via rms_norm_dmmv_q4k_alpha_beta. WG 0 of the fused
            // shader writes norm_buf so wqkv/z DMMVs see a pre-normalized
            // hidden vector; alpha and beta outputs are produced inline,
            // amortizing the rms recompute across the alpha+beta matvec
            // (combined M=2*dt_rank, very small → trivial cache pressure).
            const use_fused_ssm_pre_norm = blk: {
                if (!self.use_fused_ssm_pre_norm) break :blk false; // env-gated kill switch
                if (is_full_attn) break :blk false;
                if (self.elementwise.pipeline_rms_norm_dmmv_q4k_alpha_beta == null) break :blk false;
                if ((hidden_dim % 4) != 0) break :blk false; // shader reads as f32 vec4
                if (attn_norm.info.type_ != .f32) break :blk false; // shader reads attn_norm as f32 vec4
                const at = lt.ssm_alpha orelse break :blk false;
                const bt = lt.ssm_beta orelse break :blk false;
                // Qwen 3.6 35B-A3B Q4_K_XL ships these as f32 (see
                // FASTPATH log "alpha=f32 beta=f32"). The fused shader
                // assumes f32 row-major weights for alpha/beta. A Q4_K
                // variant can be added later if a model ships quantized
                // alpha/beta tensors.
                if (at.info.type_ != .f32) break :blk false;
                if (bt.info.type_ != .f32) break :blk false;
                // Ensure GPU SSM path will engage (otherwise the standalone
                // rms_norm is needed by runSsmLayerCpu, which reads norm_buf).
                if (self.elementwise.pipeline_ssm_conv1d == null) break :blk false;
                if (self.elementwise.pipeline_ssm_delta_net == null) break :blk false;
                if (self.elementwise.pipeline_ssm_gated_norm == null) break :blk false;
                break :blk true;
            };

            if (!use_fused_ssm_pre_norm) {
                try self.dispatchRmsNorm(
                    self.hidden_buf.handle,
                    hidden_size,
                    attn_norm.gpu_buffer.handle,
                    attn_norm.gpu_buffer.size,
                    self.norm_buf.handle,
                    hidden_size,
                    hidden_dim,
                    1,
                    rms_norm_eps,
                );
                self.decode_cmd.computeBarrier();
            }

            if (is_full_attn) {
                const attention_phase = self.beginProfilePhase();
                // === FULL ATTENTION LAYER ===
                // Q/gate projection → Q/K norm → K/V proj → RoPE → KV cache → flash attention
                // → sigmoid gate → output projection → residual

                // Prefill last-layer dead-tail detector: for non-terminal prompt tokens
                // on the final layer, only the KV cache write survives into the next
                // token's forward pass. Q/gate/flash_attn/sigmoid_mul/O-proj/residual
                // all feed hidden_buf, which the next prompt token overwrites via its
                // layer-0 embed copy. Guard the Q path and the post-KV tail with this
                // flag; K/V projection + K norm/RoPE + KV write still run so the next
                // token's attention sees coherent KV state.
                const is_dead_attn_tail = self.prefill_active and !collect_output and layer + 1 == config.n_layers;

                const q_tensor = lt.attn_q orelse return error.TensorNotFound;
                const k_tensor = lt.attn_k orelse return error.TensorNotFound;
                // Gemma 4 global attention layers share K as V (no separate attn_v tensor).
                const use_k_as_v = lt.attn_v == null and config.architecture == .gemma;
                const v_tensor = lt.attn_v orelse if (use_k_as_v) k_tensor else return error.TensorNotFound;
                const o_tensor = lt.attn_output orelse return error.TensorNotFound;
                const attn_gate_tensor = lt.attn_gate;
                const q_rows: u32 = @intCast(q_tensor.info.numElements() / hidden_dim);
                const k_rows: u32 = @intCast(k_tensor.info.numElements() / hidden_dim);
                const v_rows: u32 = @intCast(v_tensor.info.numElements() / hidden_dim);
                const o_cols: u32 = @intCast(o_tensor.info.numElements() / hidden_dim);

                // Derive per-layer head_dim from Q/K norm tensor or K tensor shape.
                // Gemma 4 has mixed dimensions: SWA layers use head_dim=256, global use 512.
                const layer_head_dim: u32 = if (lt.attn_q_norm) |qn|
                    @intCast(qn.info.numElements())
                else if (lt.attn_k_norm) |kn|
                    @intCast(kn.info.numElements())
                else
                    config.head_dim;
                const layer_kv_dim: u32 = k_rows;
                const layer_n_kv_heads: u32 = if (layer_head_dim > 0) layer_kv_dim / layer_head_dim else config.n_kv_heads;
                const layer_kv_vec_size = @as(vk.c.VkDeviceSize, layer_kv_dim) * @sizeOf(f32);
                // Gemma 4 proportional RoPE: global attention layers (use_k_as_v) rotate
                // the full head_dim using precomputed rope_freqs.weight frequencies.
                const proportional_rope = config.architecture == .gemma and use_k_as_v;
                const layer_rope_dim: u32 = if (proportional_rope)
                    layer_head_dim
                else
                    @min(if (config.rope_dim > 0) config.rope_dim else layer_head_dim, layer_head_dim);

                const packed_q_gate = q_rows == q_dim * 2;
                const separate_attn_gate = q_rows == q_dim and attn_gate_tensor != null;
                const apply_attn_gate = packed_q_gate or separate_attn_gate;
                if (state.position == 0 and layer == full_attn_interval - 1) {
                    log.debug("ATTN_Q layout L{d}: q_rows={d} k_rows={d} v_rows={d} o_cols={d} q_dim={d} kv_dim={d} packed_q_gate={} separate_gate={} gate_tensor={} apply_attn_gate={}", .{
                        layer,
                        q_rows,
                        k_rows,
                        v_rows,
                        o_cols,
                        q_dim,
                        kv_dim,
                        packed_q_gate,
                        separate_attn_gate,
                        attn_gate_tensor != null,
                        apply_attn_gate,
                    });
                }

                if (packed_q_gate) {
                    // Qwen3Next packs per-head [Q(head_dim), gate(head_dim)] blocks.
                    // Project into a temporary buffer and split each head block out.
                    // Skip when the dead-tail guard is set: attn_out_buf is only read by
                    // the subsequent deinterleave + flash_attn chain, all of which is
                    // gated below.
                    if (!is_dead_attn_tail) {
                        try self.dispatchDmmv(q_tensor, self.norm_buf, hidden_size, self.attn_out_buf, q_rows, hidden_dim);
                    }
                } else {
                    // Dense qwen35 may store Q and gate as separate tensors.
                    // Use q_rows (tensor shape) not q_dim (config) — Gemma 4 mixed head_dim.
                    // Skip Q and gate DMMVs when the dead-tail guard is set: q_buf and
                    // gate_buf only feed flash_attn / sigmoid_mul, which also get skipped.
                    if (!is_dead_attn_tail) {
                        try self.dispatchDmmv(q_tensor, self.norm_buf, hidden_size, self.q_buf, q_rows, hidden_dim);
                        if (attn_gate_tensor) |gate_tensor| {
                            try self.dispatchDmmv(gate_tensor, self.norm_buf, hidden_size, self.gate_buf, q_rows, hidden_dim);
                        }
                    }
                }
                try self.dispatchDmmv(k_tensor, self.norm_buf, hidden_size, self.k_buf, k_rows, hidden_dim);
                // Gemma full-attn layers (use_k_as_v) have v_tensor == k_tensor; the
                // V projection would be a second DMMV reading the same Q4_K weights
                // from DRAM. Skip it and let the Gemma V unit-norm below read from
                // k_buf (raw K projection) and write into v_buf, fusing the K→V
                // copy with the norm. Saves one DMMV per full-attn layer.
                //
                // Note: cycle 35 of effort-6 measured K+V fusion via the
                // dmmv_q4k_fused_gate_up pipeline (single dispatch, two
                // weight reads, one shared input read) at 8+8 interleaved
                // samples on Qwen 3.6 35B-A3B Q4_K_XL: flag-on mean 82.55
                // / median 82.92 vs flag-off mean 82.57 / median 82.84
                // (delta within noise band on this hybrid SSM+attention
                // model). The full-attention layers are a small fraction
                // of total prefill time (attention bucket ≈ 21%), the K/V
                // dispatches already overlap on RDNA4 within that bucket,
                // and the norm_buf re-read amortizes through L2. Avoid
                // re-attempting K+V fusion on this model class without
                // first changing one of those three premises.
                if (!use_k_as_v) {
                    try self.dispatchDmmv(v_tensor, self.norm_buf, hidden_size, self.v_buf, v_rows, hidden_dim);
                }
                if (packed_q_gate) {
                    // Wait for all DMMV outputs (Q+gate, K, V) before deinterleave
                    self.decode_cmd.computeBarrier();
                    // Deinterleave Q+gate using compute shader instead of per-head buffer copies.
                    // Replaces computeToTransfer + n_heads*2 vkCmdCopyBuffer + transferToCompute
                    // with a single compute dispatch, avoiding transfer stage overhead.
                    {
                        const pip = &(self.elementwise.pipeline_deinterleave orelse return error.ShaderNotLoaded);
                        const total = layer_head_dim * config.n_heads;
                        const q_full_size = @as(vk.c.VkDeviceSize, q_dim * 2) * @sizeOf(f32);
                        const q_size = @as(vk.c.VkDeviceSize, q_dim) * @sizeOf(f32);
                        if (pip.uses_push_descriptors) {
                            const push = DeinterleavePush{
                                .head_dim = layer_head_dim,
                                .n_heads = config.n_heads,
                            };
                            self.pushDispatch3(
                                pip,
                                std.mem.asBytes(&push),
                                self.attn_out_buf.handle,
                                q_full_size,
                                self.q_buf.handle,
                                q_size,
                                self.gate_buf.handle,
                                q_size,
                                (total + 63) / 64,
                                1,
                                1,
                            );
                        } else {
                            const ds = try self.allocDescSet(pip.descriptor_set_layout);
                            self.writeDescSet3(ds, self.attn_out_buf.handle, q_full_size, self.q_buf.handle, q_size, self.gate_buf.handle, q_size);
                            try self.elementwise.recordDeinterleave(&self.decode_cmd, ds, layer_head_dim, config.n_heads);
                        }
                    }
                    self.decode_cmd.computeBarrier();
                } else {
                    self.decode_cmd.computeBarrier();
                }

                if (lt.attn_q_bias != null or lt.attn_k_bias != null or lt.attn_v_bias != null) {
                    if (lt.attn_q_bias) |bias| {
                        // Skip Q bias for dead-tail tokens: Q only feeds flash_attn
                        // which is also skipped.
                        if (!is_dead_attn_tail) {
                            try self.dispatchBiasAdd(self.q_buf.handle, self.q_buf.size, bias, q_dim);
                        }
                    }
                    if (lt.attn_k_bias) |bias| {
                        try self.dispatchBiasAdd(self.k_buf.handle, self.k_buf.size, bias, kv_dim);
                    }
                    if (!use_k_as_v) {
                        if (lt.attn_v_bias) |bias| {
                            try self.dispatchBiasAdd(self.v_buf.handle, self.v_buf.size, bias, kv_dim);
                        }
                    }
                    self.decode_cmd.computeBarrier();
                }

                // Bug fix #1: Q/K normalization (per-head RMS norm)
                // attn_q_norm and attn_k_norm are per-head norms with head_dim weights
                const q_norm_tensor = lt.attn_q_norm;
                const k_norm_tensor = lt.attn_k_norm;
                if (state.position == 0 and layer == full_attn_interval - 1) {
                    log.debug("ATTN_NORM layout L{d}: q_norm_elems={d} k_norm_elems={d} q_norm_type={s} k_norm_type={s} head_dim={d} n_heads={d} n_kv_heads={d}", .{
                        layer,
                        if (q_norm_tensor) |qn| qn.info.numElements() else 0,
                        if (k_norm_tensor) |kn| kn.info.numElements() else 0,
                        if (q_norm_tensor) |qn| @tagName(qn.info.type_) else "none",
                        if (k_norm_tensor) |kn| @tagName(kn.info.type_) else "none",
                        config.head_dim,
                        config.n_heads,
                        config.n_kv_heads,
                    });
                    if (self.validation_diagnostics_enabled) {
                        const mmap = self.model.mmap_data orelse return error.NoMmapData;
                        if (lt.attn_norm) |attn_norm_tensor| {
                            var attn_norm_preview = [_]f32{0} ** 4;
                            const n = @min(attn_norm_tensor.info.numElements(), attn_norm_preview.len);
                            const off: usize = @intCast(self.model.gguf_file.tensor_data_offset + attn_norm_tensor.info.offset);
                            readMmapFloats(mmap, off, attn_norm_tensor.info.type_, attn_norm_preview[0..n]);
                            log.info("ATTN_NORM_WEIGHTS L{d}: attn_norm[0..4]=[{d:.6},{d:.6},{d:.6},{d:.6}]", .{
                                layer,
                                attn_norm_preview[0],
                                attn_norm_preview[1],
                                attn_norm_preview[2],
                                attn_norm_preview[3],
                            });
                        }
                        if (q_norm_tensor) |qn| {
                            var q_norm_preview = [_]f32{0} ** 4;
                            const n = @min(qn.info.numElements(), q_norm_preview.len);
                            const off: usize = @intCast(self.model.gguf_file.tensor_data_offset + qn.info.offset);
                            readMmapFloats(mmap, off, qn.info.type_, q_norm_preview[0..n]);
                            log.info("ATTN_NORM_WEIGHTS L{d}: q_norm[0..4]=[{d:.6},{d:.6},{d:.6},{d:.6}]", .{
                                layer,
                                q_norm_preview[0],
                                q_norm_preview[1],
                                q_norm_preview[2],
                                q_norm_preview[3],
                            });
                        }
                        if (k_norm_tensor) |kn| {
                            var k_norm_preview = [_]f32{0} ** 4;
                            const n = @min(kn.info.numElements(), k_norm_preview.len);
                            const off: usize = @intCast(self.model.gguf_file.tensor_data_offset + kn.info.offset);
                            readMmapFloats(mmap, off, kn.info.type_, k_norm_preview[0..n]);
                            log.info("ATTN_NORM_WEIGHTS L{d}: k_norm[0..4]=[{d:.6},{d:.6},{d:.6},{d:.6}]", .{
                                layer,
                                k_norm_preview[0],
                                k_norm_preview[1],
                                k_norm_preview[2],
                                k_norm_preview[3],
                            });
                        }
                    }
                }
                // Bug fix #5+#6: IMRoPE — only rotate rope_dim of head_dim dimensions
                // IMROPE: use precomputed per-pair frequencies when sections are present
                const use_imrope = config.rope_sections[0] > 0 or config.rope_sections[1] > 0;
                // Gemma 4 SWA layers use a different RoPE frequency base than global layers.
                // Global layers use precomputed frequencies (with rope_freqs.weight factors).
                const use_swa_rope = config.architecture == .gemma and config.rope_freq_base_swa > 0 and layer_head_dim < config.head_dim;
                const use_yarn_rope = hasYarnScaling(config);
                // Use precomputed frequency buffer when the host has already baked in
                // per-dimension frequency corrections (IMROPE, Gemma proportional RoPE, YaRN).
                const use_precomputed_freq = use_imrope or (proportional_rope and !use_swa_rope) or use_yarn_rope;
                const rope_freq: f32 = if (use_precomputed_freq) 0.0 else if (use_swa_rope) config.rope_freq_base_swa else config.rope_freq_base;
                const freq_buf_handle = if (use_precomputed_freq) self.rope_freq_buf.handle else null;
                const rope_attn_scale = if (use_yarn_rope) effectiveRopeAttnScale(config) else 1.0;

                // Fused norm+rope: when both norm and rope are needed, combine them into
                // a single dispatch per head set, eliminating 1 barrier + 2 dispatches.
                const use_fused_norm_rope = self.elementwise.pipeline_norm_rope != null;
                // q_rope_done starts true for dead-tail tokens so the standalone Q RoPE
                // dispatches below (both push-descriptor and transfer-fallback branches)
                // are suppressed — q_buf only feeds flash_attn / sigmoid_mul / O_proj,
                // all of which are also skipped by the dead-tail guard further down.
                // Cycle 20 only handled the q_norm_tensor branch; models without a
                // separate attn_q_norm tensor (e.g. Qwen3.5 mamba-hybrid) still ran
                // Q RoPE for every non-terminal prompt token at the last full-attn layer.
                var q_rope_done = is_dead_attn_tail;
                var k_rope_done = false;

                // Effort-11 cycle-12: fused Q+K norm+rope + KV cache write
                // path. Single dispatch absorbs the (Q norm+rope → K norm+rope
                // → kv_cache_write) trio when all per-call gates pass. Saves
                // 2 dispatches + 1 global compute barrier per attention layer.
                const physical_token_for_fused = if (self.use_fused_qk_kv)
                    self.physicalTokenIndex(state.position) catch null
                else
                    null;
                const fused_qk_kv_base_eligible = self.use_fused_qk_kv and
                    self.elementwise.pipeline_qk_norm_rope_kv_write != null and
                    q_norm_tensor != null and
                    k_norm_tensor != null and
                    !packed_q_gate and
                    !is_dead_attn_tail and
                    !(state.position == 0 and self.validation_diagnostics_enabled) and
                    physical_token_for_fused != null;
                const gemma_v_unit_norm_needed =
                    config.architecture == .gemma and config.rope_freq_base_swa > 0;
                const gemma_v_norm_in_fused =
                    fused_qk_kv_base_eligible and gemma_v_unit_norm_needed;

                // Gemma use_k_as_v optimization: V unit-norm reads the RAW K
                // projection. If the fused Q/K/KV shader is active, it can
                // normalize V directly while writing kv_v; otherwise this must
                // run before K norm overwrites k_buf.
                const apply_v_unit_norm_early = use_k_as_v and
                    gemma_v_unit_norm_needed and
                    !gemma_v_norm_in_fused;
                if (apply_v_unit_norm_early) {
                    try self.dispatchRmsNorm(
                        self.k_buf.handle,
                        self.k_buf.size,
                        self.unit_norm_weights.handle,
                        self.unit_norm_weights.size,
                        self.v_buf.handle,
                        self.v_buf.size,
                        layer_head_dim,
                        layer_n_kv_heads,
                        rms_norm_eps,
                    );
                    self.decode_cmd.computeBarrier();
                }

                const fused_qk_kv_eligible = fused_qk_kv_base_eligible;

                if (fused_qk_kv_eligible) {
                    const qn = q_norm_tensor.?;
                    const kn = k_norm_tensor.?;
                    const dst_offset_floats: u32 = physical_token_for_fused.? * layer_kv_dim;
                    const v_src_for_fused = if (gemma_v_norm_in_fused and use_k_as_v) self.k_buf else self.v_buf;
                    self.dispatchQkNormRopeKvWrite(
                        self.q_buf.handle,
                        self.q_buf.size,
                        qn.gpu_buffer.handle,
                        qn.gpu_buffer.size,
                        self.k_buf.handle,
                        self.k_buf.size,
                        kn.gpu_buffer.handle,
                        kn.gpu_buffer.size,
                        freq_buf_handle,
                        self.rope_freq_buf.size,
                        self.kv_k_cache[layer_idx].handle,
                        self.kv_k_cache[layer_idx].size,
                        v_src_for_fused.handle,
                        v_src_for_fused.size,
                        self.kv_v_cache[layer_idx].handle,
                        self.kv_v_cache[layer_idx].size,
                        layer_head_dim,
                        layer_rope_dim,
                        config.n_heads,
                        layer_n_kv_heads,
                        state.position,
                        rope_freq,
                        rope_attn_scale,
                        rms_norm_eps,
                        dst_offset_floats,
                        gemma_v_norm_in_fused,
                    );
                    self.decode_cmd.computeBarrier();
                    q_rope_done = true;
                    k_rope_done = true;
                } else if (q_norm_tensor) |qn| {
                    // Skip Q norm/RoPE for dead-tail tokens: q_buf only feeds flash_attn.
                    // Still mark q_rope_done=true so the fallback-path Q RoPE below is
                    // also skipped (avoids reading stale q_buf).
                    if (is_dead_attn_tail) {
                        q_rope_done = true;
                    } else if (use_fused_norm_rope) {
                        // Fused Q norm + Q RoPE in a single dispatch
                        self.dispatchNormRopeInPlace(
                            self.q_buf.handle,
                            self.q_buf.size,
                            qn.gpu_buffer.handle,
                            qn.gpu_buffer.size,
                            freq_buf_handle,
                            self.rope_freq_buf.size,
                            layer_head_dim,
                            layer_rope_dim,
                            config.n_heads,
                            state.position,
                            rope_freq,
                            rope_attn_scale,
                            rms_norm_eps,
                        );
                        q_rope_done = true;
                    } else {
                        try self.dispatchRmsNorm(
                            self.q_buf.handle,
                            self.q_buf.size,
                            qn.gpu_buffer.handle,
                            qn.gpu_buffer.size,
                            self.q_buf.handle,
                            self.q_buf.size,
                            layer_head_dim,
                            config.n_heads,
                            rms_norm_eps,
                        );
                    }
                }
                if (!fused_qk_kv_eligible) {
                    if (k_norm_tensor) |kn| {
                        if (use_fused_norm_rope) {
                            // Fused K norm + K RoPE in a single dispatch
                            self.dispatchNormRopeInPlace(
                                self.k_buf.handle,
                                self.k_buf.size,
                                kn.gpu_buffer.handle,
                                kn.gpu_buffer.size,
                                freq_buf_handle,
                                self.rope_freq_buf.size,
                                layer_head_dim,
                                layer_rope_dim,
                                layer_n_kv_heads,
                                state.position,
                                rope_freq,
                                rope_attn_scale,
                                rms_norm_eps,
                            );
                            k_rope_done = true;
                        } else {
                            try self.dispatchRmsNorm(
                                self.k_buf.handle,
                                self.k_buf.size,
                                kn.gpu_buffer.handle,
                                kn.gpu_buffer.size,
                                self.k_buf.handle,
                                self.k_buf.size,
                                layer_head_dim,
                                layer_n_kv_heads,
                                rms_norm_eps,
                            );
                        }
                    }
                    // Gemma 4 applies plain RMS norm (unit weights) to V per-head.
                    // Mirrors Metal forward_metal.zig:3460-3462. For use_k_as_v
                    // layers this already ran ahead of Q/K norms above — skip here.
                    if (config.architecture == .gemma and config.rope_freq_base_swa > 0 and !apply_v_unit_norm_early) {
                        try self.dispatchRmsNorm(
                            self.v_buf.handle,
                            self.v_buf.size,
                            self.unit_norm_weights.handle,
                            self.unit_norm_weights.size,
                            self.v_buf.handle,
                            self.v_buf.size,
                            layer_head_dim,
                            layer_n_kv_heads,
                            rms_norm_eps,
                        );
                    }
                    self.decode_cmd.computeBarrier();

                    if (!k_rope_done) {
                        // K RoPE first — KV cache write reads k_buf, so it must complete before the write.
                        try self.dispatchRopeInPlace(
                            self.k_buf.handle,
                            self.k_buf.size,
                            freq_buf_handle,
                            self.rope_freq_buf.size,
                            layer_head_dim,
                            layer_rope_dim,
                            layer_n_kv_heads,
                            state.position,
                            rope_freq,
                            rope_attn_scale,
                        );
                    }
                    // KV cache write: use compute shader to stay in compute pipeline,
                    // avoiding compute→transfer + transfer→compute stage transitions.
                    {
                        const physical_token = try self.physicalTokenIndex(state.position);
                        if (self.elementwise.pipeline_kv_cache_write) |*kv_pip| {
                            if (!k_rope_done) self.decode_cmd.computeBarrier();
                            const push = KvCacheWritePush{
                                .kv_dim = layer_kv_dim,
                                .dst_offset = physical_token * layer_kv_dim,
                            };
                            if (kv_pip.uses_push_descriptors) {
                                self.pushDispatch4(
                                    kv_pip,
                                    std.mem.asBytes(&push),
                                    self.k_buf.handle,
                                    self.k_buf.size,
                                    self.kv_k_cache[layer_idx].handle,
                                    self.kv_k_cache[layer_idx].size,
                                    self.v_buf.handle,
                                    self.v_buf.size,
                                    self.kv_v_cache[layer_idx].handle,
                                    self.kv_v_cache[layer_idx].size,
                                    (layer_kv_dim + 63) / 64,
                                    1,
                                    1,
                                );
                            } else {
                                const ds = try self.allocDescSet(kv_pip.descriptor_set_layout);
                                self.writeDescSet4(ds, self.k_buf.handle, self.k_buf.size, self.kv_k_cache[layer_idx].handle, self.kv_k_cache[layer_idx].size, self.v_buf.handle, self.v_buf.size, self.kv_v_cache[layer_idx].handle, self.kv_v_cache[layer_idx].size);
                                self.decode_cmd.dispatchWithPush(kv_pip, ds, std.mem.asBytes(&push), (layer_kv_dim + 63) / 64, 1, 1);
                            }
                            if (!q_rope_done) {
                                // Q RoPE overlaps with KV write — no data dependency between them.
                                try self.dispatchRopeInPlace(
                                    self.q_buf.handle,
                                    self.q_buf.size,
                                    freq_buf_handle,
                                    self.rope_freq_buf.size,
                                    layer_head_dim,
                                    layer_rope_dim,
                                    config.n_heads,
                                    state.position,
                                    rope_freq,
                                    rope_attn_scale,
                                );
                            }
                            self.decode_cmd.computeBarrier();
                        } else {
                            // Transfer fallback: Q RoPE before barrier (original order preserved)
                            if (!q_rope_done) {
                                try self.dispatchRopeInPlace(
                                    self.q_buf.handle,
                                    self.q_buf.size,
                                    freq_buf_handle,
                                    self.rope_freq_buf.size,
                                    layer_head_dim,
                                    layer_rope_dim,
                                    config.n_heads,
                                    state.position,
                                    rope_freq,
                                    rope_attn_scale,
                                );
                            }
                            self.decode_cmd.computeAndTransferBarrier();
                            const kv_offset = @as(vk.c.VkDeviceSize, physical_token) * layer_kv_vec_size;
                            const k_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = kv_offset, .size = layer_kv_vec_size };
                            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.k_buf.handle, self.kv_k_cache[layer_idx].handle, 1, &k_region);
                            const v_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = kv_offset, .size = layer_kv_vec_size };
                            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.v_buf.handle, self.kv_v_cache[layer_idx].handle, 1, &v_region);
                            self.decode_cmd.transferToComputeBarrier();
                        }
                    }
                }

                // Prefill last-layer shortcut: at the final layer of a non-terminal prefill
                // token, flash_attn + sigmoid_gate + O-proj + residual only feed into
                // hidden_buf, which the next prompt token overwrites via its layer-0 embed
                // copy. KV cache has already been committed just above, so next token's
                // attention still sees coherent state. Extends cycle 4's FFN/MoE-body skip
                // deeper into the attention block itself. Saves ~1 full-attn pass per
                // non-terminal prompt token.
                if (self.prefill_active and !collect_output and layer + 1 == config.n_layers) {
                    self.endProfilePhase(.attention, attention_phase);
                    continue;
                }

                // Flash attention. Sinks are pre-populated at init into a per-layer
                // slot of attn_sinks_buf (cycle 8); flash_attn reads via sink_offset.
                //
                // Batched-path foundation (ZINC_BATCH_ATTN=1): route through
                // the flash_attn_batched pipeline with n_queries=1 and
                // seq_start=state.position. Output is bit-equivalent to the
                // decode shader for n_queries=1. Speed cycle enables n>1 later.
                //
                // The flash_attn_kernel ProfilePhase wraps just the kernel
                // dispatch (and its computeBarrier) so --profile output
                // separates kernel time from QKV/RoPE/output-proj time
                // within the broader .attention phase. This is the
                // diagnostic the Effort 11 plan asks for in Step 2 to
                // attribute the L-dependent ms (~46 ms at L=1162) to the
                // flash_attn dispatch vs other dispatches before
                // committing to a shader rewrite.
                const flash_attn_kernel_phase = self.beginProfilePhase();
                const use_batched = self.use_batch_attn and self.attention.pipeline_batched != null;
                const attn_seq_len = state.position + 1;
                const split_k_min_seq_len: u32 = 128;
                const split_k_seq_ok = self.fa_split_k_forced or attn_seq_len >= split_k_min_seq_len;
                const use_split_k = !use_batched and split_k_seq_ok and self.fa_split_k > 1 and
                    self.attention.pipeline_split != null and
                    self.attention.pipeline_split_merge != null and
                    self.partial_attn_out_buf.handle != null;
                // Effort-11 cycle-17: when ZINC_FUSED_OPROJ_MERGE=1 and split-K
                // is active, we replace the (merge → barrier → o_proj) pair
                // with a single dmmv_q4k_o_proj_merge dispatch. The flag is
                // off by default; when it engages, the merge dispatch below
                // is skipped and the o_proj site routes through
                // dispatchDmmvOprojMerge (which reads the partials directly
                // from partial_attn_out_buf).
                const o_tensor_for_merge = lt.attn_output;
                const o_proj_quant_ok = if (o_tensor_for_merge) |ot| ot.info.type_ == .q4_k else false;
                const apply_attn_gate_for_merge = lt.attn_gate != null;
                const post_attn_norm_for_merge = config.architecture == .gemma and lt.post_attention_norm != null;
                const fused_oproj_merge_active = self.use_fused_oproj_merge and
                    use_split_k and
                    self.dmmv.pipeline_q4k_o_proj_merge != null and
                    o_proj_quant_ok and
                    !apply_attn_gate_for_merge and
                    !post_attn_norm_for_merge and
                    !self.validation_diagnostics_enabled and
                    hidden_dim <= 4096 and
                    config.n_heads <= 64 and
                    self.fa_split_k <= 8;
                if (use_split_k) {
                    // Split-K dispatch (flash_attn writes per-chunk partials)
                    // followed by the merge pass (combines partials, applies sinks,
                    // writes final output). The split shader reuses flash_attn.spv
                    // specialized with N_I_CHUNKS so binding 4 holds partials and
                    // binding 5 (sinks) is unused — we still bind it for layout
                    // compatibility with the original 6-binding pipeline.
                    const split_pip = &self.attention.pipeline_split.?;
                    const merge_pip = &self.attention.pipeline_split_merge.?;
                    const sink_buf = self.attn_sinks_buf;
                    const sink_offset: u32 = layer * config.n_heads;
                    if (split_pip.uses_push_descriptors) {
                        const split_push = FlashAttnPush{
                            .head_dim = layer_head_dim,
                            .n_heads = config.n_heads,
                            .n_kv_heads = layer_n_kv_heads,
                            .seq_len = attn_seq_len,
                            .page_size = kv_page_size_tokens,
                            .attn_scale_bits = if (config.attn_scale != 0) @as(u32, @bitCast(config.attn_scale)) else 0,
                            .sink_offset = sink_offset,
                        };
                        self.pushDispatch6(
                            split_pip,
                            std.mem.asBytes(&split_push),
                            self.q_buf.handle,
                            self.q_buf.size,
                            self.kv_k_cache[layer_idx].handle,
                            self.kv_k_cache[layer_idx].size,
                            self.kv_v_cache[layer_idx].handle,
                            self.kv_v_cache[layer_idx].size,
                            self.page_table_buf.handle,
                            self.page_table_buf.size,
                            self.partial_attn_out_buf.handle,
                            self.partial_attn_out_buf.size,
                            sink_buf.handle,
                            sink_buf.size,
                            config.n_heads,
                            self.fa_split_k,
                            1,
                        );
                    } else {
                        const split_ds = try self.allocDescSet(split_pip.descriptor_set_layout);
                        self.writeDescSet6(
                            split_ds,
                            self.q_buf.handle,
                            self.q_buf.size,
                            self.kv_k_cache[layer_idx].handle,
                            self.kv_k_cache[layer_idx].size,
                            self.kv_v_cache[layer_idx].handle,
                            self.kv_v_cache[layer_idx].size,
                            self.page_table_buf.handle,
                            self.page_table_buf.size,
                            self.partial_attn_out_buf.handle,
                            self.partial_attn_out_buf.size,
                            sink_buf.handle,
                            sink_buf.size,
                        );
                        try self.attention.recordFlashAttnSplit(&self.decode_cmd, split_ds, layer_head_dim, config.n_heads, layer_n_kv_heads, attn_seq_len, kv_page_size_tokens, config.attn_scale, sink_offset);
                    }
                    self.decode_cmd.computeBarrier();
                    if (!fused_oproj_merge_active) {
                        if (merge_pip.uses_push_descriptors) {
                            const merge_push = FlashAttnSplitMergePush{
                                .head_dim = layer_head_dim,
                                .n_heads = config.n_heads,
                                .sink_offset = sink_offset,
                            };
                            self.pushDispatch3(
                                merge_pip,
                                std.mem.asBytes(&merge_push),
                                self.partial_attn_out_buf.handle,
                                self.partial_attn_out_buf.size,
                                self.attn_out_buf.handle,
                                self.attn_out_buf.size,
                                sink_buf.handle,
                                sink_buf.size,
                                config.n_heads,
                                1,
                                1,
                            );
                        } else {
                            const merge_ds = try self.allocDescSet(merge_pip.descriptor_set_layout);
                            self.writeDescSet3(
                                merge_ds,
                                self.partial_attn_out_buf.handle,
                                self.partial_attn_out_buf.size,
                                self.attn_out_buf.handle,
                                self.attn_out_buf.size,
                                sink_buf.handle,
                                sink_buf.size,
                            );
                            try self.attention.recordFlashAttnSplitMerge(&self.decode_cmd, merge_ds, layer_head_dim, config.n_heads, sink_offset);
                        }
                    }
                } else if (use_batched) {
                    const pip = &self.attention.pipeline_batched.?;
                    const sink_buf = self.attn_sinks_buf;
                    const sink_offset: u32 = layer * config.n_heads;
                    if (pip.uses_push_descriptors) {
                        const push = FlashAttnBatchedPush{
                            .head_dim = layer_head_dim,
                            .n_heads = config.n_heads,
                            .n_kv_heads = layer_n_kv_heads,
                            .seq_start = state.position,
                            .n_queries = 1,
                            .page_size = kv_page_size_tokens,
                            .attn_scale_bits = if (config.attn_scale != 0) @as(u32, @bitCast(config.attn_scale)) else 0,
                            .sink_offset = sink_offset,
                        };
                        self.pushDispatch6(
                            pip,
                            std.mem.asBytes(&push),
                            self.q_buf.handle,
                            self.q_buf.size,
                            self.kv_k_cache[layer_idx].handle,
                            self.kv_k_cache[layer_idx].size,
                            self.kv_v_cache[layer_idx].handle,
                            self.kv_v_cache[layer_idx].size,
                            self.page_table_buf.handle,
                            self.page_table_buf.size,
                            self.attn_out_buf.handle,
                            self.attn_out_buf.size,
                            sink_buf.handle,
                            sink_buf.size,
                            config.n_heads,
                            1,
                            1,
                        );
                    } else {
                        const attn_ds = try self.allocDescSet(pip.descriptor_set_layout);
                        self.writeDescSet6(
                            attn_ds,
                            self.q_buf.handle,
                            self.q_buf.size,
                            self.kv_k_cache[layer_idx].handle,
                            self.kv_k_cache[layer_idx].size,
                            self.kv_v_cache[layer_idx].handle,
                            self.kv_v_cache[layer_idx].size,
                            self.page_table_buf.handle,
                            self.page_table_buf.size,
                            self.attn_out_buf.handle,
                            self.attn_out_buf.size,
                            sink_buf.handle,
                            sink_buf.size,
                        );
                        try self.attention.recordFlashAttnBatched(&self.decode_cmd, attn_ds, layer_head_dim, config.n_heads, layer_n_kv_heads, state.position, 1, kv_page_size_tokens, config.attn_scale, sink_offset);
                    }
                } else if (self.attention.pipeline) |*pip| {
                    const sink_buf = self.attn_sinks_buf;
                    const sink_offset: u32 = layer * config.n_heads;
                    if (pip.uses_push_descriptors) {
                        const push = FlashAttnPush{
                            .head_dim = layer_head_dim,
                            .n_heads = config.n_heads,
                            .n_kv_heads = layer_n_kv_heads,
                            .seq_len = state.position + 1,
                            .page_size = kv_page_size_tokens,
                            .attn_scale_bits = if (config.attn_scale != 0) @as(u32, @bitCast(config.attn_scale)) else 0,
                            .sink_offset = sink_offset,
                        };
                        self.pushDispatch6(
                            pip,
                            std.mem.asBytes(&push),
                            self.q_buf.handle,
                            self.q_buf.size,
                            self.kv_k_cache[layer_idx].handle,
                            self.kv_k_cache[layer_idx].size,
                            self.kv_v_cache[layer_idx].handle,
                            self.kv_v_cache[layer_idx].size,
                            self.page_table_buf.handle,
                            self.page_table_buf.size,
                            self.attn_out_buf.handle,
                            self.attn_out_buf.size,
                            sink_buf.handle,
                            sink_buf.size,
                            config.n_heads,
                            1,
                            1,
                        );
                    } else {
                        const attn_ds = try self.allocDescSet(pip.descriptor_set_layout);
                        self.writeDescSet6(
                            attn_ds,
                            self.q_buf.handle,
                            self.q_buf.size,
                            self.kv_k_cache[layer_idx].handle,
                            self.kv_k_cache[layer_idx].size,
                            self.kv_v_cache[layer_idx].handle,
                            self.kv_v_cache[layer_idx].size,
                            self.page_table_buf.handle,
                            self.page_table_buf.size,
                            self.attn_out_buf.handle,
                            self.attn_out_buf.size,
                            sink_buf.handle,
                            sink_buf.size,
                        );
                        try self.attention.recordFlashAttn(&self.decode_cmd, attn_ds, layer_head_dim, config.n_heads, layer_n_kv_heads, attn_seq_len, kv_page_size_tokens, config.attn_scale, sink_offset);
                    }
                }
                self.decode_cmd.computeBarrier();
                self.endProfilePhase(.flash_attn_kernel, flash_attn_kernel_phase);

                // Self-check the first attention layer at seq_len=1: with only one KV token,
                // flash attention must reproduce the current V slice for each query head's KV group.
                if (state.position == 0 and is_full_attn and self.validation_diagnostics_enabled) {
                    const attn_q_dim_dbg = @as(u32, config.n_heads) * layer_head_dim;
                    const attn_kv_dim_dbg = layer_n_kv_heads * layer_head_dim;
                    const q_bytes = @as(vk.c.VkDeviceSize, attn_q_dim_dbg) * @sizeOf(f32);
                    const k_bytes = @as(vk.c.VkDeviceSize, attn_kv_dim_dbg) * @sizeOf(f32);
                    const v_bytes = @as(vk.c.VkDeviceSize, attn_kv_dim_dbg) * @sizeOf(f32);
                    const attn_bytes = @as(vk.c.VkDeviceSize, attn_q_dim_dbg) * @sizeOf(f32);
                    const k_off = q_bytes;
                    const v_off = k_off + k_bytes;
                    const attn_off = v_off + v_bytes;

                    try self.decode_cmd.end();
                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                    try self.decode_cmd.reset();
                    try self.decode_cmd.begin();
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.q_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                        .srcOffset = 0,
                        .dstOffset = 0,
                        .size = q_bytes,
                    });
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.k_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                        .srcOffset = 0,
                        .dstOffset = k_off,
                        .size = k_bytes,
                    });
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.v_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                        .srcOffset = 0,
                        .dstOffset = v_off,
                        .size = v_bytes,
                    });
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.attn_out_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                        .srcOffset = 0,
                        .dstOffset = attn_off,
                        .size = attn_bytes,
                    });
                    try self.decode_cmd.end();
                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                    const dbg_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
                    const q_vals = dbg_ptr[0..attn_q_dim_dbg];
                    const k_vals = dbg_ptr[@intCast(k_off / @sizeOf(f32))..][0..attn_kv_dim_dbg];
                    const v_vals = dbg_ptr[@intCast(v_off / @sizeOf(f32))..][0..attn_kv_dim_dbg];
                    const attn_vals = dbg_ptr[@intCast(attn_off / @sizeOf(f32))..][0..attn_q_dim_dbg];
                    const sink_ptr: [*]const f32 = @ptrCast(@alignCast(self.attn_sinks_buf.mapped.?));
                    const sink_vals = sink_ptr[layer * config.n_heads ..][0..config.n_heads];
                    const scale = if (config.attn_scale != 0) config.attn_scale else 1.0 / @sqrt(@as(f32, @floatFromInt(layer_head_dim)));
                    const q_per_kv = @max(config.n_heads / @max(layer_n_kv_heads, 1), 1);

                    var attn_v_max_diff: f32 = 0;
                    for (0..config.n_heads) |h| {
                        const kv_head = h / q_per_kv;
                        const q_head = q_vals[h * layer_head_dim ..][0..layer_head_dim];
                        const k_head = k_vals[kv_head * layer_head_dim ..][0..layer_head_dim];
                        const sink_val = sink_vals[h];

                        var score: f32 = 0;
                        for (0..layer_head_dim) |d| score += q_head[d] * k_head[d];
                        score *= scale;

                        var max_score = score;
                        if (!std.math.isNan(sink_val) and sink_val > max_score) max_score = sink_val;
                        var denom = @exp(score - max_score);
                        if (!std.math.isNan(sink_val)) denom += @exp(sink_val - max_score);
                        const weight = if (denom > 0) @exp(score - max_score) / denom else 0.0;

                        for (0..layer_head_dim) |d| {
                            const got = attn_vals[h * layer_head_dim + d];
                            const want = v_vals[kv_head * layer_head_dim + d] * weight;
                            const diff = @abs(got - want);
                            if (diff > attn_v_max_diff) attn_v_max_diff = diff;
                        }
                    }
                    log.info("ATTN_SELFTEST L{d}: seq_len=1 max_diff={d:.6} attn_h0[0..4]=[{d:.6},{d:.6},{d:.6},{d:.6}] v_kv0[0..4]=[{d:.6},{d:.6},{d:.6},{d:.6}] sink0={d:.6}", .{
                        layer,
                        attn_v_max_diff,
                        attn_vals[0],
                        attn_vals[1],
                        attn_vals[2],
                        attn_vals[3],
                        v_vals[0],
                        v_vals[1],
                        v_vals[2],
                        v_vals[3],
                        sink_vals[0],
                    });

                    if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                    try self.decode_cmd.reset();
                    try self.decode_cmd.begin();
                }

                // Validate paged multi-token flash attention against a naive CPU reference on the
                // last prompt token. This catches page-table / KV-layout bugs that token-0 checks miss.
                if (diag_last_prompt_token and config.architecture == .gpt_oss and layer == full_attn_interval - 1 and self.validation_diagnostics_enabled) {
                    const seq_len_dbg: u32 = state.position + 1;
                    const attn_q_dim_dbg = @as(u32, config.n_heads) * layer_head_dim;
                    const attn_kv_dim_dbg = layer_n_kv_heads * layer_head_dim;
                    const q_bytes = @as(vk.c.VkDeviceSize, attn_q_dim_dbg) * @sizeOf(f32);
                    const kv_token_bytes = @as(vk.c.VkDeviceSize, attn_kv_dim_dbg) * @sizeOf(f32);
                    const kv_dbg_bytes = @as(vk.c.VkDeviceSize, seq_len_dbg * attn_kv_dim_dbg) * @sizeOf(f32);
                    const attn_bytes = @as(vk.c.VkDeviceSize, attn_q_dim_dbg) * @sizeOf(f32);
                    const k_off = q_bytes;
                    const v_off = k_off + kv_dbg_bytes;
                    const attn_off = v_off + kv_dbg_bytes;

                    try self.decode_cmd.end();
                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                    try self.decode_cmd.reset();
                    try self.decode_cmd.begin();
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.q_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                        .srcOffset = 0,
                        .dstOffset = 0,
                        .size = q_bytes,
                    });
                    for (0..seq_len_dbg) |tok| {
                        const physical_token = try self.physicalTokenIndex(@intCast(tok));
                        const src_offset = @as(vk.c.VkDeviceSize, physical_token) * kv_token_bytes;
                        const dst_offset = @as(vk.c.VkDeviceSize, tok) * kv_token_bytes;
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.kv_k_cache[layer_idx].handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = src_offset,
                            .dstOffset = k_off + dst_offset,
                            .size = kv_token_bytes,
                        });
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.kv_v_cache[layer_idx].handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = src_offset,
                            .dstOffset = v_off + dst_offset,
                            .size = kv_token_bytes,
                        });
                    }
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.attn_out_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                        .srcOffset = 0,
                        .dstOffset = attn_off,
                        .size = attn_bytes,
                    });
                    try self.decode_cmd.end();
                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                    const dbg_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
                    const q_vals = dbg_ptr[0..attn_q_dim_dbg];
                    const k_vals = dbg_ptr[@intCast(k_off / @sizeOf(f32))..][0 .. seq_len_dbg * attn_kv_dim_dbg];
                    const v_vals = dbg_ptr[@intCast(v_off / @sizeOf(f32))..][0 .. seq_len_dbg * attn_kv_dim_dbg];
                    const attn_vals = dbg_ptr[@intCast(attn_off / @sizeOf(f32))..][0..attn_q_dim_dbg];
                    const sink_ptr: [*]const f32 = @ptrCast(@alignCast(self.attn_sinks_buf.mapped.?));
                    const sink_vals = sink_ptr[layer * config.n_heads ..][0..config.n_heads];

                    const seq_len_usize: usize = @intCast(seq_len_dbg);
                    const q_dim_usize: usize = @intCast(attn_q_dim_dbg);
                    var cpu_attn = try self.allocator.alloc(f32, q_dim_usize);
                    defer self.allocator.free(cpu_attn);
                    var scores = try self.allocator.alloc(f32, seq_len_usize);
                    defer self.allocator.free(scores);
                    var probs = try self.allocator.alloc(f32, seq_len_usize);
                    defer self.allocator.free(probs);

                    const scale = if (config.attn_scale != 0) config.attn_scale else 1.0 / @sqrt(@as(f32, @floatFromInt(layer_head_dim)));
                    const q_per_kv = @max(config.n_heads / @max(layer_n_kv_heads, 1), 1);
                    for (0..config.n_heads) |h| {
                        const kv_head = h / q_per_kv;
                        const q_head = q_vals[h * layer_head_dim ..][0..layer_head_dim];
                        const sink_val = sink_vals[h];

                        var max_score: f32 = -std.math.inf(f32);
                        for (0..seq_len_dbg) |tok| {
                            const k_tok = k_vals[tok * attn_kv_dim_dbg + kv_head * layer_head_dim ..][0..layer_head_dim];
                            var dot: f32 = 0;
                            for (0..layer_head_dim) |d| dot += q_head[d] * k_tok[d];
                            const s = dot * scale;
                            scores[tok] = s;
                            if (s > max_score) max_score = s;
                        }
                        if (!std.math.isNan(sink_val) and sink_val > max_score) max_score = sink_val;

                        var sum_exp: f32 = 0;
                        if (!std.math.isNan(sink_val)) {
                            sum_exp += @exp(sink_val - max_score);
                        }
                        for (0..seq_len_dbg) |tok| {
                            const p = @exp(scores[tok] - max_score);
                            probs[tok] = p;
                            sum_exp += p;
                        }
                        const inv_sum = if (sum_exp > 0) 1.0 / sum_exp else 0.0;

                        const out_head = cpu_attn[h * layer_head_dim ..][0..layer_head_dim];
                        @memset(out_head, 0);
                        for (0..seq_len_dbg) |tok| {
                            const weight = probs[tok] * inv_sum;
                            const v_tok = v_vals[tok * attn_kv_dim_dbg + kv_head * layer_head_dim ..][0..layer_head_dim];
                            for (0..layer_head_dim) |d| out_head[d] += weight * v_tok[d];
                        }
                    }

                    var attn_ref_max_diff: f32 = 0;
                    var q_nan_count: usize = 0;
                    var k_nan_count: usize = 0;
                    var attn_nan_count: usize = 0;
                    var cpu_nan_count: usize = 0;
                    for (q_vals) |v| {
                        if (std.math.isNan(v)) q_nan_count += 1;
                    }
                    for (k_vals) |v| {
                        if (std.math.isNan(v)) k_nan_count += 1;
                    }
                    for (0..attn_q_dim_dbg) |i| {
                        if (std.math.isNan(attn_vals[i])) attn_nan_count += 1;
                        if (std.math.isNan(cpu_attn[i])) cpu_nan_count += 1;
                        if (std.math.isNan(attn_vals[i]) or std.math.isNan(cpu_attn[i])) continue;
                        const diff = @abs(attn_vals[i] - cpu_attn[i]);
                        if (diff > attn_ref_max_diff) attn_ref_max_diff = diff;
                    }
                    log.info("ATTN_REFTEST L{d} pos={d}: seq_len={d} max_diff={d:.6} q_nan={d} k_nan={d} attn_nan={d} cpu_nan={d} attn_h0[0..4]=[{d:.6},{d:.6},{d:.6},{d:.6}] cpu_h0[0..4]=[{d:.6},{d:.6},{d:.6},{d:.6}]", .{
                        layer,
                        state.position,
                        seq_len_dbg,
                        attn_ref_max_diff,
                        q_nan_count,
                        k_nan_count,
                        attn_nan_count,
                        cpu_nan_count,
                        attn_vals[0],
                        attn_vals[1],
                        attn_vals[2],
                        attn_vals[3],
                        cpu_attn[0],
                        cpu_attn[1],
                        cpu_attn[2],
                        cpu_attn[3],
                    });

                    if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                    try self.decode_cmd.reset();
                    try self.decode_cmd.begin();
                }

                if (apply_attn_gate) {
                    if (self.elementwise.pipeline_sigmoid_mul) |*pip| {
                        // Regression guard marker: self.writeDescSet3(gds, self.attn_out_buf.handle
                        _ = pip;
                        try self.dispatchSigmoidMul(
                            self.attn_out_buf.handle,
                            self.attn_out_buf.size,
                            self.gate_buf.handle,
                            self.gate_buf.size,
                            self.attn_out_buf.handle,
                            self.attn_out_buf.size,
                            q_dim,
                        );
                        self.decode_cmd.computeBarrier();
                    }
                }

                // Output projection + attention residual
                const apply_post_attn_norm = config.architecture == .gemma and lt.post_attention_norm != null;
                const has_post_attn_norm = apply_post_attn_norm;
                const diag_attn_residual = diag_last_prompt_token and config.architecture == .gpt_oss and self.validation_diagnostics_enabled and q_dim <= 8192;
                if (!has_post_attn_norm and !self.validation_diagnostics_enabled) {
                    // Fused: O-proj DMMV accumulates directly into hidden_buf,
                    // eliminating separate scale_acc dispatch + barrier
                    // Use o_cols (from O weight tensor shape) — matches actual attention output dim.
                    // Gemma 4 has mixed head_dim (256 SWA vs 512 global); o_cols is always correct
                    // while q_dim (from config) uses the max head_dim.
                    if (fused_oproj_merge_active and lt.attn_output_bias == null) {
                        // Effort-11 cycle-17: replace (merge → barrier → o_proj DMMV-acc)
                        // with a single dispatch that reads partials from
                        // partial_attn_out_buf, computes per-head LSE merge weights
                        // with sink fold-in, stages attn_out into LDS, and runs
                        // the Q4_K matmul accumulating into hidden_buf. Bias path
                        // falls through to the unfused dispatch (the post-bias
                        // residual barrier is unchanged).
                        const sink_offset_for_merge: u32 = layer * config.n_heads;
                        try self.dispatchDmmvOprojMerge(
                            o_tensor,
                            self.partial_attn_out_buf,
                            self.attn_sinks_buf,
                            self.hidden_buf,
                            hidden_dim,
                            o_cols,
                            config.n_heads,
                            self.fa_split_k,
                            sink_offset_for_merge,
                            layer_head_dim,
                        );
                    } else {
                        try self.dispatchDmmvAcc(o_tensor, self.attn_out_buf, self.attn_out_buf.size, self.hidden_buf, hidden_dim, o_cols);
                        if (lt.attn_output_bias) |bias| {
                            self.decode_cmd.computeBarrier();
                            try self.dispatchBiasAdd(self.hidden_buf.handle, hidden_size, bias, hidden_dim);
                        }
                    }
                    self.decode_cmd.computeBarrier();
                } else {
                    // Unfused path: needed when post-attn norm exists (Gemma) or diagnostics enabled
                    try self.dispatchDmmv(o_tensor, self.attn_out_buf, self.attn_out_buf.size, self.o_proj_buf, hidden_dim, o_cols);
                    if (lt.attn_output_bias) |bias| {
                        self.decode_cmd.computeBarrier();
                        try self.dispatchBiasAdd(self.o_proj_buf.handle, hidden_size, bias, hidden_dim);
                    }
                    self.decode_cmd.computeBarrier();

                    if ((state.position == 0 or (diag_last_prompt_token and config.architecture == .gpt_oss)) and is_full_attn and self.validation_diagnostics_enabled and q_dim <= 8192) {
                        try self.decode_cmd.end();
                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                        try self.decode_cmd.reset();
                        try self.decode_cmd.begin();
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.attn_out_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = 0,
                            .dstOffset = 0,
                            .size = @as(vk.c.VkDeviceSize, q_dim) * @sizeOf(f32),
                        });
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.o_proj_buf.handle, self.embed_staging.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = 0,
                            .dstOffset = 0,
                            .size = hidden_size,
                        });
                        try self.decode_cmd.end();
                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                        const attn_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
                        const attn_vals = attn_ptr[0..q_dim];
                        const raw_gpu: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
                        const mmap = self.model.mmap_data orelse return error.NoMmapData;
                        const o_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + o_tensor.info.offset);
                        var cpu_row_buf: [8192]f32 = undefined;
                        const cpu_proj = try self.allocator.alloc(f32, hidden_dim);
                        defer self.allocator.free(cpu_proj);
                        var raw_max_diff: f32 = 0;

                        for (0..hidden_dim) |row| {
                            dequantRow(mmap[o_off..], @intCast(row), q_dim, o_tensor.info.type_, cpu_row_buf[0..q_dim]);
                            var dot: f64 = 0;
                            for (0..q_dim) |i| dot += @as(f64, cpu_row_buf[i]) * @as(f64, attn_vals[i]);
                            cpu_proj[row] = @floatCast(dot);
                        }
                        if (lt.attn_output_bias) |bias| {
                            addBiasFromTensor(self, cpu_proj.ptr, bias, hidden_dim);
                        }
                        for (0..hidden_dim) |i| {
                            const diff = @abs(raw_gpu[i] - cpu_proj[i]);
                            if (diff > raw_max_diff) raw_max_diff = diff;
                        }
                        log.info("ATTN_O_RAW_CHECK L{d}: type={s} max_diff={d:.6} gpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}] cpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}] ok={s}", .{
                            layer,
                            @tagName(o_tensor.info.type_),
                            raw_max_diff,
                            raw_gpu[0],
                            raw_gpu[1],
                            raw_gpu[2],
                            raw_gpu[3],
                            cpu_proj[0],
                            cpu_proj[1],
                            cpu_proj[2],
                            cpu_proj[3],
                            if (raw_max_diff < 0.1) @as([]const u8, "YES") else @as([]const u8, "NO"),
                        });

                        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                        try self.decode_cmd.reset();
                        try self.decode_cmd.begin();
                    }

                    // Gemma post-attention norm: RMS norm on o_proj output before residual add.
                    // When the fused rms_norm_add pipeline is loaded and we're not in a
                    // diagnostics path, skip the separate norm dispatch and let the
                    // residual-add branch below fuse both into one pass.
                    const use_fused_pan_decode = apply_post_attn_norm and
                        self.elementwise.pipeline_rms_norm_add != null and
                        !diag_attn_residual and
                        config.architecture != .gpt_oss;
                    if (apply_post_attn_norm and !use_fused_pan_decode) {
                        const pan_tensor = lt.post_attention_norm.?;
                        try self.dispatchRmsNorm(
                            self.o_proj_buf.handle,
                            hidden_size,
                            pan_tensor.gpu_buffer.handle,
                            pan_tensor.gpu_buffer.size,
                            self.o_proj_buf.handle,
                            hidden_size,
                            hidden_dim,
                            1,
                            rms_norm_eps,
                        );
                        self.decode_cmd.computeBarrier();
                    }

                    if (diag_attn_residual) {
                        self.decode_cmd.computeToTransferBarrier();
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.hidden_buf.handle, self.residual_buf.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = 0,
                            .dstOffset = 0,
                            .size = hidden_size,
                        });
                        self.decode_cmd.transferToComputeBarrier();
                    }

                    if (config.architecture == .gpt_oss) {
                        try self.dispatchVadd(
                            self.hidden_buf.handle,
                            hidden_size,
                            self.o_proj_buf.handle,
                            hidden_size,
                            self.moe_out_buf.handle,
                            self.moe_out_buf.size,
                            hidden_dim,
                        );
                        self.decode_cmd.computeAndTransferBarrier();
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.moe_out_buf.handle, self.hidden_buf.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = 0,
                            .dstOffset = 0,
                            .size = hidden_size,
                        });
                        self.decode_cmd.transferToComputeBarrier();
                    } else if (use_fused_pan_decode) {
                        // Fused Gemma post_attention_norm + residual add in one
                        // dispatch: hidden += pan_weight * rmsnorm(o_proj_buf).
                        const pan_tensor = lt.post_attention_norm.?;
                        try self.dispatchRmsNormAdd(
                            self.hidden_buf.handle,
                            hidden_size,
                            self.o_proj_buf.handle,
                            hidden_size,
                            pan_tensor.gpu_buffer.handle,
                            pan_tensor.gpu_buffer.size,
                            hidden_dim,
                            1,
                            rms_norm_eps,
                        );
                        self.decode_cmd.computeBarrier();
                    } else {
                        // Attention residual: hidden_buf += o_proj_buf
                        try self.dispatchScaleAcc(
                            self.hidden_buf.handle,
                            hidden_size,
                            self.o_proj_buf.handle,
                            hidden_size,
                            hidden_dim,
                            1.0,
                        );
                        self.decode_cmd.computeBarrier();
                    }

                    if (diag_attn_residual) {
                        try self.decode_cmd.end();
                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                        try self.decode_cmd.reset();
                        try self.decode_cmd.begin();
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.residual_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = 0,
                            .dstOffset = 0,
                            .size = hidden_size,
                        });
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.o_proj_buf.handle, self.ssm_hidden_staging.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = 0,
                            .dstOffset = 0,
                            .size = hidden_size,
                        });
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.hidden_buf.handle, self.embed_staging.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = 0,
                            .dstOffset = 0,
                            .size = hidden_size,
                        });
                        try self.decode_cmd.end();
                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                        const pre_hidden_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
                        const branch_ptr: [*]const f32 = @ptrCast(@alignCast(self.ssm_hidden_staging.mapped.?));
                        const post_hidden_ptr: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
                        var residual_max_diff: f32 = 0;
                        var residual_max_idx: usize = 0;
                        for (0..hidden_dim) |i| {
                            const want = pre_hidden_ptr[i] + branch_ptr[i];
                            const diff = @abs(post_hidden_ptr[i] - want);
                            if (diff > residual_max_diff) {
                                residual_max_diff = diff;
                                residual_max_idx = i;
                            }
                        }
                        log.info("ATTN_RESIDUAL_CHECK L{d} pos={d}: max_diff={d:.6} idx={d} gpu={d:.6} cpu={d:.6} pre={d:.6} branch={d:.6}", .{
                            layer,
                            state.position,
                            residual_max_diff,
                            residual_max_idx,
                            post_hidden_ptr[residual_max_idx],
                            pre_hidden_ptr[residual_max_idx] + branch_ptr[residual_max_idx],
                            pre_hidden_ptr[residual_max_idx],
                            branch_ptr[residual_max_idx],
                        });

                        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                        try self.decode_cmd.reset();
                        try self.decode_cmd.begin();
                    }
                }

                // --- Mid-layer diagnostic: o_proj RMS at attention layers (BOS only) ---
                // Single readback per attention layer — reads o_proj_buf (before residual add)
                if ((state.position == 0 or (diag_last_prompt_token and config.architecture == .gpt_oss)) and is_full_attn and self.validation_diagnostics_enabled) {
                    // Flush current work so o_proj_buf is valid
                    try self.decode_cmd.end();
                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                    // Read attn_out_buf and o_proj_buf for a CPU-vs-GPU projection check.
                    try self.decode_cmd.reset();
                    try self.decode_cmd.begin();
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.attn_out_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                        .srcOffset = 0,
                        .dstOffset = 0,
                        .size = @as(vk.c.VkDeviceSize, q_dim) * @sizeOf(f32),
                    });
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.o_proj_buf.handle, self.embed_staging.handle, 1, &vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = hidden_size });
                    try self.decode_cmd.end();
                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                    const attn_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
                    const attn_vals = attn_ptr[0..q_dim];
                    const op: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
                    var op_sq: f64 = 0;
                    var op_max: f32 = 0;
                    for (0..hidden_dim) |i| {
                        op_sq += @as(f64, op[i]) * @as(f64, op[i]);
                        const a = @abs(op[i]);
                        if (a > op_max) op_max = a;
                    }
                    const op_rms: f32 = @floatCast(@sqrt(op_sq / @as(f64, @floatFromInt(hidden_dim))));
                    log.info("L{d} o_proj: rms={d:.6} max={d:.4} [0..4]=[{d:.6},{d:.6},{d:.6},{d:.6}]", .{
                        layer, op_rms, op_max, op[0], op[1], op[2], op[3],
                    });

                    if (q_dim <= 8192) {
                        const mmap = self.model.mmap_data orelse return error.NoMmapData;
                        const o_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + o_tensor.info.offset);
                        var cpu_row_buf: [8192]f32 = undefined;
                        const cpu_proj = try self.allocator.alloc(f32, hidden_dim);
                        defer self.allocator.free(cpu_proj);
                        var o_proj_max_diff: f32 = 0;

                        for (0..hidden_dim) |row| {
                            dequantRow(mmap[o_off..], @intCast(row), q_dim, o_tensor.info.type_, cpu_row_buf[0..q_dim]);
                            var dot: f64 = 0;
                            for (0..q_dim) |i| dot += @as(f64, cpu_row_buf[i]) * @as(f64, attn_vals[i]);
                            cpu_proj[row] = @floatCast(dot);
                        }
                        if (lt.attn_output_bias) |bias| {
                            addBiasFromTensor(self, cpu_proj.ptr, bias, hidden_dim);
                        }
                        if (apply_post_attn_norm) {
                            const pan_tensor = lt.post_attention_norm.?;
                            const post_norm = try self.allocator.alloc(f32, hidden_dim);
                            defer self.allocator.free(post_norm);
                            const pan_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + pan_tensor.info.offset);
                            readMmapFloats(mmap, pan_off, pan_tensor.info.type_, post_norm);
                            cpuRmsNormMul(cpu_proj.ptr, post_norm, cpu_proj.ptr, hidden_dim, 1, rms_norm_eps);
                        }
                        for (0..hidden_dim) |i| {
                            const diff = @abs(op[i] - cpu_proj[i]);
                            if (diff > o_proj_max_diff) o_proj_max_diff = diff;
                        }
                        log.info("ATTN_O_PROJ_CHECK L{d}: type={s} max_diff={d:.6} gpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}] cpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}] ok={s}", .{
                            layer,
                            @tagName(o_tensor.info.type_),
                            o_proj_max_diff,
                            op[0],
                            op[1],
                            op[2],
                            op[3],
                            cpu_proj[0],
                            cpu_proj[1],
                            cpu_proj[2],
                            cpu_proj[3],
                            if (o_proj_max_diff < 0.1) @as([]const u8, "YES") else @as([]const u8, "NO"),
                        });
                    }

                    // Restart command buffer
                    if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                    try self.decode_cmd.reset();
                    try self.decode_cmd.begin();
                }
                self.endProfilePhase(.attention, attention_phase);
            } else {
                // === SSM / LINEAR ATTENTION LAYER ===
                // Use GPU SSM when all three shaders are available (conv1d, delta-net, gated norm).
                // Falls back to CPU for platforms missing any shader.
                const use_gpu_ssm = self.elementwise.pipeline_ssm_conv1d != null and
                    self.elementwise.pipeline_ssm_delta_net != null and
                    self.elementwise.pipeline_ssm_gated_norm != null;
                if (state.position == 0 and layer == 0) {
                    log.debug("FASTPATH: gpu_ssm={} arch={s} ssm_shader={} delta_cols8={}", .{
                        use_gpu_ssm,
                        @tagName(config.architecture),
                        self.elementwise.pipeline_ssm_conv1d != null,
                        self.use_ssm_delta_cols8 and self.elementwise.pipeline_ssm_delta_net_cols8 != null,
                    });
                }
                // Dead-tail SSM skip: at the final layer of a non-terminal
                // prefill token in an SSM-last hybrid model, the gate-z DMMV
                // / gated_norm / ssm_out only feed hidden_buf, which the next
                // token's layer-0 embed copy overwrites. Conv1d + delta_net
                // still run because they commit SSM state for future tokens.
                //
                // Active condition depends on full_attn_interval — for Qwen3.5
                // qwen35moe (full_attn_interval=4, n_layers=40), the LAST layer
                // is attention so this branch is never reached and cycle 20's
                // attention dead-tail skip handles the equivalent work. For
                // architectures with SSM as the LAST layer (e.g. larger
                // full_attn_interval values, pure mamba), this skip mirrors
                // cycle 20's pattern automatically.
                const ssm_dead_tail = self.prefill_active and !collect_output and layer + 1 == config.n_layers;
                const ssm_phase = self.beginProfilePhase();
                if (use_gpu_ssm) {
                    try self.runSsmLayerGpu(state, layer, layer_idx, ssm_dead_tail, use_fused_ssm_pre_norm);
                } else {
                    if (self.profile_enabled) self.profile_token_counters.cpu_ssm_fallbacks += 1;
                    try self.runSsmLayerCpu(state, layer, layer_idx);
                }
                self.endProfilePhase(.ssm, ssm_phase);
                if (self.partial_decode_stop_after_ssm_gnorm or self.partial_decode_stop_after_ssm_conv) {
                    break;
                }
            }

            // Prefill last-layer shortcut: at the final layer of a non-terminal prefill
            // token, the FFN/MoE + residual only feed into final_norm + LM_head, which
            // we also skip below. KV cache and SSM state have already been committed
            // inside the attention/SSM block, so the next token still sees correct
            // state. Saves one full MoE pass per non-terminal prompt token.
            if (self.prefill_active and !collect_output and layer + 1 == config.n_layers) {
                continue;
            }

            // --- FFN norm: prefer ffn_norm.weight, fall back to post_attention_norm for models
            // that use a single norm between attention and FFN (e.g. Qwen3.5).
            const ffn_norm_tensor = lt.ffn_norm orelse
                lt.post_attention_norm orelse return error.TensorNotFound;
            // Decide whether the (FFN norm + MoE router DMMV) pair can be folded
            // into a single dispatch via rms_norm_dmmv_f32. Conditions:
            //   - opt-in flag enabled (default ON when pipeline loaded)
            //   - this layer is an MoE layer (router exists)
            //   - architecture isn't Gemma (different router input flow) and
            //     isn't gpt_oss (validation diagnostic reads ffn_norm_buf
            //     immediately after the standalone dispatch)
            //   - ffn_norm + router weights are both f32 (shader bindings)
            //   - router has no bias term and no Gemma-specific scale
            const router_tensor_opt = lt.ffn_gate_inp;
            // The fused shader (rms_norm_dmmv_f32.comp) reads hidden /
            // ffn_norm weights / router weights as vec4 since cycle 42, so
            // gate the path on K%4==0. Every catalog MoE checkpoint today
            // satisfies this (Qwen 3.5/3.6 hidden_dim=2048); reject and
            // fall back to the unfused path otherwise.
            const can_fuse_rms_router = self.use_fused_rms_router and
                is_moe and
                config.architecture != .gemma and
                config.architecture != .gpt_oss and
                ffn_norm_tensor.info.type_ == .f32 and
                router_tensor_opt != null and
                router_tensor_opt.?.info.type_ == .f32 and
                lt.ffn_gate_inp_bias == null and
                lt.ffn_gate_inp_scale == null and
                (hidden_dim % 4) == 0;
            const can_store_partial_stop = self.partial_decode_stop_after_ffn_norm and
                !can_fuse_rms_router and
                self.prefill_active and
                self.qwen36DensePrefillPartialStoreEnabled() and
                self.partial_decode_ffn_norm_out != null and
                self.partial_decode_hidden_out != null and
                (hidden_dim % 4) == 0;
            if (can_store_partial_stop) {
                try self.dispatchRmsNormStoreHidden(
                    self.hidden_buf.handle,
                    hidden_size,
                    ffn_norm_tensor.gpu_buffer.handle,
                    ffn_norm_tensor.gpu_buffer.size,
                    self.partial_decode_ffn_norm_out.?,
                    self.partial_decode_ffn_norm_out_offset,
                    hidden_size,
                    self.partial_decode_hidden_out.?,
                    self.partial_decode_hidden_out_offset,
                    hidden_size,
                    hidden_dim,
                    rms_norm_eps,
                );
                partial_hidden_out_written_by_stop = true;
                break;
            }
            if (!can_fuse_rms_router) {
                try self.dispatchRmsNorm(
                    self.hidden_buf.handle,
                    hidden_size,
                    ffn_norm_tensor.gpu_buffer.handle,
                    ffn_norm_tensor.gpu_buffer.size,
                    self.ffn_norm_buf.handle,
                    hidden_size,
                    hidden_dim,
                    1,
                    rms_norm_eps,
                );
                self.decode_cmd.computeBarrier();
            }

            if (self.partial_decode_stop_after_ffn_norm) {
                if (can_fuse_rms_router) return error.UnsupportedPartialDecode;
                if (self.partial_decode_ffn_norm_out) |norm_out| {
                    self.decode_cmd.computeToTransferBarrier();
                    const norm_region = vk.c.VkBufferCopy{
                        .srcOffset = 0,
                        .dstOffset = self.partial_decode_ffn_norm_out_offset,
                        .size = hidden_size,
                    };
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.ffn_norm_buf.handle, norm_out, 1, &norm_region);
                }
                break;
            }

            if (self.validation_diagnostics_enabled and config.architecture == .gpt_oss and collect_output and state.generated_tokens.items.len == 0 and hidden_dim <= 8192) {
                try self.decode_cmd.end();
                try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                try self.decode_cmd.reset();
                try self.decode_cmd.begin();
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.hidden_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                    .srcOffset = 0,
                    .dstOffset = 0,
                    .size = hidden_size,
                });
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.ffn_norm_buf.handle, self.embed_staging.handle, 1, &vk.c.VkBufferCopy{
                    .srcOffset = 0,
                    .dstOffset = 0,
                    .size = hidden_size,
                });
                try self.decode_cmd.end();
                try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                const hidden_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
                const gpu_norm_ptr: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
                const mmap = self.model.mmap_data orelse return error.NoMmapData;
                const norm_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + ffn_norm_tensor.info.offset);
                var cpu_norm_w: [8192]f32 = undefined;
                dequantRow(mmap[norm_off..], 0, hidden_dim, ffn_norm_tensor.info.type_, cpu_norm_w[0..hidden_dim]);
                var cpu_normed: [8192]f32 = undefined;
                cpuRmsNormMul(hidden_ptr, cpu_norm_w[0..hidden_dim], cpu_normed[0..hidden_dim].ptr, hidden_dim, 1, rms_norm_eps);

                var norm_max_diff: f32 = 0;
                for (0..hidden_dim) |i| {
                    const diff = @abs(gpu_norm_ptr[i] - cpu_normed[i]);
                    if (diff > norm_max_diff) norm_max_diff = diff;
                }
                log.info("FFN_INP_CHECK L{d} pos={d}: hidden[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}]", .{
                    layer,
                    state.position,
                    hidden_ptr[0],
                    hidden_ptr[1],
                    hidden_ptr[2],
                    hidden_ptr[3],
                });
                log.info("FFN_NORM_CHECK L{d} pos={d}: type={s} max_diff={d:.6} gpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}] cpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}]", .{
                    layer,
                    state.position,
                    @tagName(ffn_norm_tensor.info.type_),
                    norm_max_diff,
                    gpu_norm_ptr[0],
                    gpu_norm_ptr[1],
                    gpu_norm_ptr[2],
                    gpu_norm_ptr[3],
                    cpu_normed[0],
                    cpu_normed[1],
                    cpu_normed[2],
                    cpu_normed[3],
                });

                if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                try self.decode_cmd.reset();
                try self.decode_cmd.begin();
            }

            var gpu_moe_barriers_cover_hidden = false;
            if (is_moe) {
                const moe_phase = self.beginProfilePhase();
                // --- MoE: router DMMV → top-k → expert dispatch ---
                const router_tensor = lt.ffn_gate_inp orelse return error.TensorNotFound;
                const moe_router_phase = self.beginProfilePhase();

                // Gemma MoE uses plain RMS-normalized hidden (unit weights) for the router input,
                // while the FFN experts read the learned-weight ffn_norm output. Mirrors Metal
                // forward_metal.zig:4636-4639.
                const router_input_buf = if (config.architecture == .gemma) blk: {
                    try self.dispatchRmsNorm(
                        self.hidden_buf.handle,
                        hidden_size,
                        self.unit_norm_weights.handle,
                        self.unit_norm_weights.size,
                        self.residual_buf.handle,
                        hidden_size,
                        hidden_dim,
                        1,
                        rms_norm_eps,
                    );
                    self.decode_cmd.computeBarrier();
                    // Gemma 4 MoE: apply ffn_gate_inp.scale elementwise to router input
                    // before the router DMMV (matches Metal forward_metal.zig:4126-4129).
                    if (lt.ffn_gate_inp_scale) |scale_t| {
                        try self.dispatchMulElementwise(
                            self.residual_buf.handle,
                            hidden_size,
                            scale_t.gpu_buffer.handle,
                            scale_t.gpu_buffer.size,
                            hidden_dim,
                        );
                        self.decode_cmd.computeBarrier();
                    }
                    break :blk self.residual_buf;
                } else self.ffn_norm_buf;

                if (can_fuse_rms_router) {
                    // Fused path: one dispatch produces both ffn_norm_buf
                    // (for downstream MoE gate/up + shared-expert reads) and
                    // router_logits_buf in a single shader invocation. The
                    // post-dispatch barrier covers BOTH outputs so the
                    // downstream consumers see consistent state.
                    try self.dispatchRmsNormDmmvF32(
                        self.hidden_buf.handle,
                        hidden_size,
                        ffn_norm_tensor.gpu_buffer.handle,
                        ffn_norm_tensor.gpu_buffer.size,
                        router_tensor.gpu_buffer.handle,
                        router_tensor.gpu_buffer.size,
                        self.ffn_norm_buf.handle,
                        hidden_size,
                        self.router_logits_buf.handle,
                        self.router_logits_buf.size,
                        config.n_experts,
                        hidden_dim,
                        rms_norm_eps,
                    );
                    // Effort-6 Step 5 prerequisite (cycle 36): when the FFN
                    // input capture flag is on AND this layer's MoE input is
                    // ffn_norm_buf (not pre_ffw_norm_2), copy the ffn_norm
                    // output into the per-(token, layer) capture slot.
                    // Promote the buffer-scoped compute→compute barrier to a
                    // global compute→compute+transfer barrier so the upcoming
                    // vkCmdCopyBuffer reads ffn_norm_buf under visibility
                    // guarantees. Downstream gate/up/router consumers still
                    // see the same write visibility through the broader
                    // memory barrier.
                    const capture_active = self.use_capture_ffn_input and
                        lt.pre_ffw_norm_2 == null and
                        state.position < self.prefill_ffn_input_capture_max_tokens;
                    if (capture_active) {
                        self.decode_cmd.computeAndTransferBarrier();
                        const slot_off: vk.c.VkDeviceSize =
                            (@as(vk.c.VkDeviceSize, state.position) *
                                @as(vk.c.VkDeviceSize, config.n_layers) +
                                @as(vk.c.VkDeviceSize, layer)) *
                            @as(vk.c.VkDeviceSize, hidden_dim) *
                            @sizeOf(f32);
                        const capture_size: vk.c.VkDeviceSize = hidden_size;
                        if (slot_off + capture_size <= self.prefill_ffn_input_capture_buf.size) {
                            const region = vk.c.VkBufferCopy{
                                .srcOffset = 0,
                                .dstOffset = slot_off,
                                .size = capture_size,
                            };
                            vk.c.vkCmdCopyBuffer(
                                self.decode_cmd.handle,
                                self.ffn_norm_buf.handle,
                                self.prefill_ffn_input_capture_buf.handle,
                                1,
                                &region,
                            );
                        }
                    } else {
                        const fused_ranges = [_]CommandBuffer.BufferRange{
                            .{ .buffer = self.ffn_norm_buf.handle, .size = hidden_size },
                            .{ .buffer = self.router_logits_buf.handle, .size = self.router_logits_buf.size },
                        };
                        self.decode_cmd.computeBuffersBarrier(&fused_ranges);
                    }
                } else {
                    try self.dispatchDmmv(router_tensor, router_input_buf, hidden_size, self.router_logits_buf, config.n_experts, hidden_dim);
                    self.decode_cmd.computeBufferBarrier(self.router_logits_buf.handle, self.router_logits_buf.size);
                }

                // Dispatch each selected expert — handle both separate and fused gate+up layouts.
                // Gemma 4 26B-A4B uses fused ffn_gate_up_exps instead of separate gate/up.
                const fused_gate_up = lt.ffn_gate_up_exps;
                const gate_exps = lt.ffn_gate_exps orelse fused_gate_up orelse return error.TensorNotFound;
                const up_exps = lt.ffn_up_exps orelse fused_gate_up orelse return error.TensorNotFound;
                const down_exps = lt.ffn_down_exps orelse return error.TensorNotFound;

                const gate_quant = gate_exps.info.type_;
                const down_quant = down_exps.info.type_;
                // Expert weight offset: for fused gate_up, stride covers both halves (2*inter_dim)
                const fused_inter = if (fused_gate_up != null) inter_dim * 2 else inter_dim;
                const expert_gate_row_bytes = expertSliceBytes(gate_quant, fused_inter, hidden_dim);
                // Byte offset to the up half within a fused gate_up expert slice
                const up_base_offset: u32 = if (fused_gate_up != null) expertSliceBytes(gate_quant, inter_dim, hidden_dim) else 0;
                // Down projection: each expert has hidden_dim rows of K=inter_dim
                const expert_down_row_bytes = expertSliceBytes(down_quant, hidden_dim, inter_dim);

                const gemma_gpu_topk_down_q5_1 =
                    down_exps.info.type_ == .q5_1 and
                    self.dmmv.pipeline_q5_1_moe_fused_down_acc_scaled != null;
                const gemma_gpu_topk_down_q8_0 =
                    down_exps.info.type_ == .q8_0 and
                    self.dmmv.pipeline_q8_0_moe_fused_down_acc_scaled != null;
                const gemma_gpu_topk_moe =
                    config.architecture == .gemma and
                    fused_gate_up != null and
                    gate_exps.info.type_ == .q4_k and
                    up_exps.info.type_ == .q4_k and
                    (gemma_gpu_topk_down_q5_1 or gemma_gpu_topk_down_q8_0) and
                    lt.ffn_gate_inp_bias == null and
                    lt.ffn_gate_exps_bias == null and
                    lt.ffn_up_exps_bias == null and
                    lt.ffn_down_exps_bias == null and
                    lt.ffn_down_exps_scale != null and
                    self.elementwise.pipeline_softmax_topk != null and
                    self.dmmv.pipeline_q4k_moe_fused_gate_up_geglu != null;

                const gemma_router_scale: f32 = if (config.architecture == .gemma)
                    1.0 / std.math.sqrt(@as(f32, @floatFromInt(hidden_dim)))
                else
                    1.0;
                const gemma_topk_scales_router =
                    gemma_gpu_topk_moe and
                    self.use_softmax_topk_v2 and
                    self.elementwise.pipeline_softmax_topk_v2 != null;

                // Gemma 4 MoE: scale router logits by 1/sqrt(hidden_dim) before softmax.
                // The GPU-topk fast path folds the same positive scale into topk
                // softmax weights, avoiding a separate in-place dispatch + barrier.
                // Matches Metal forward_metal.zig:4134-4137 for other Gemma paths.
                if (config.architecture == .gemma and !gemma_topk_scales_router) {
                    try self.dispatchScaleInPlace(
                        self.router_logits_buf.handle,
                        self.router_logits_buf.size,
                        config.n_experts,
                        gemma_router_scale,
                    );
                    self.decode_cmd.computeBufferBarrier(self.router_logits_buf.handle, self.router_logits_buf.size);
                }
                self.endProfilePhase(.moe_router, moe_router_phase);

                const use_prefill_tail_topk = self.prefill_active and
                    !collect_output and
                    self.moe_prefill_tail_topk_limit > 0 and
                    self.prefill_current_token_idx + self.moe_prefill_tail_topk_guard_tokens < self.prefill_embed_big_token_count;
                const effective_moe_topk_limit = if (use_prefill_tail_topk)
                    self.moe_prefill_tail_topk_limit
                else
                    self.moe_topk_limit;
                const n_used = if (effective_moe_topk_limit > 0)
                    @min(config.n_experts_used, effective_moe_topk_limit)
                else
                    config.n_experts_used;

                // Gemma 4 MoE uses pre_ffw_norm_2 for MoE expert input (vs ffn_norm_buf for shared).
                // Available to both GPU-routed and CPU-routed MoE paths.
                const defer_gemma_pre_ffw_norm_2 = gemma_gpu_topk_moe and lt.pre_ffw_norm_2 != null;
                const expert_input_buf = if (defer_gemma_pre_ffw_norm_2)
                    self.residual_buf
                else if (lt.pre_ffw_norm_2) |pre_norm_t| blk: {
                    try self.dispatchRmsNorm(
                        self.hidden_buf.handle,
                        hidden_size,
                        pre_norm_t.gpu_buffer.handle,
                        pre_norm_t.gpu_buffer.size,
                        self.residual_buf.handle,
                        hidden_size,
                        hidden_dim,
                        1,
                        rms_norm_eps,
                    );
                    self.decode_cmd.computeBarrier();
                    break :blk self.residual_buf;
                } else self.ffn_norm_buf;

                // Check if full GPU MoE path is available (MoE DMMV + softmax_topk + weighted_acc).
                // Gemma architecture is excluded because its MoE uses architecture-specific extras
                // (pre_ffw_norm_2, ffn_gate_inp.scale, ffn_down_exps.scale, post_ffw_norm_1/2, etc.)
                // that are simpler/safer to execute via CPU-routed sequential expert dispatch.
                // Matches Metal's canUseGpuRoutedBatchedMoe (forward_metal.zig:4068-4070).
                const use_gpu_moe = config.architecture != .gemma and
                    config.architecture != .gpt_oss and
                    fused_gate_up == null and
                    self.dmmv.moePipelineForType(gate_quant) != null and
                    self.dmmv.moePipelineForType(up_exps.info.type_) != null and
                    self.dmmv.moePipelineForType(down_quant) != null and
                    self.elementwise.pipeline_softmax_topk != null and
                    self.elementwise.pipeline_moe_weighted_acc != null;
                if (state.position == 0 and layer == 0) {
                    log.info("FASTPATH: gpu_moe={} gate={s} up={s} down={s} q4k_moe={} q5k_moe={} softmax_topk={} weighted_acc={}", .{
                        use_gpu_moe,
                        @tagName(gate_quant),
                        @tagName(up_exps.info.type_),
                        @tagName(down_quant),
                        self.dmmv.moePipelineForType(gate_quant) != null,
                        self.dmmv.moePipelineForType(down_quant) != null,
                        self.elementwise.pipeline_softmax_topk != null,
                        self.elementwise.pipeline_moe_weighted_acc != null,
                    });
                }

                if (use_gpu_moe) {
                    // === GPU MoE path: BATCHED expert dispatch — all experts in parallel ===
                    // All 8 experts' gate/up/down DMMVs run as Y workgroups in a single dispatch.
                    // This gives ~8× better GPU utilization vs serial per-expert dispatch.
                    // Reduces dispatches from 32 to 5, barriers from 32 to 4 per MoE layer.

                    // softmax_topk writes expert_ids + weights to router_output_buf
                    const moe_topk_phase = self.beginProfilePhase();
                    try self.dispatchSoftmaxTopk(
                        self.router_logits_buf.handle,
                        @as(vk.c.VkDeviceSize, config.n_experts) * @sizeOf(f32),
                        self.router_output_buf.handle,
                        self.router_output_buf.size,
                        config.n_experts,
                        n_used,
                    );
                    if (self.use_capture_routing and state.position < self.routing_capture_max_tokens) {
                        // Step 11a: fan out the topk output to the per-(token, layer) capture
                        // slot. Broader compute→compute+transfer barrier replaces the narrow
                        // compute→compute one so the upcoming vkCmdCopyBuffer can read the
                        // router_output_buf under visibility guarantees. Downstream compute
                        // dispatches still see the same write visibility.
                        self.decode_cmd.computeAndTransferBarrier();
                        const slot_bytes: vk.c.VkDeviceSize = @intCast(self.routing_capture_slot_bytes);
                        const slot_off: vk.c.VkDeviceSize =
                            (@as(vk.c.VkDeviceSize, state.position) *
                                @as(vk.c.VkDeviceSize, config.n_layers) +
                                @as(vk.c.VkDeviceSize, layer)) * slot_bytes;
                        const copy_size: vk.c.VkDeviceSize = @min(
                            slot_bytes,
                            @as(vk.c.VkDeviceSize, self.router_output_buf.size),
                        );
                        if (slot_off + copy_size <= self.routing_capture_buf.size) {
                            const region = vk.c.VkBufferCopy{
                                .srcOffset = 0,
                                .dstOffset = slot_off,
                                .size = copy_size,
                            };
                            vk.c.vkCmdCopyBuffer(
                                self.decode_cmd.handle,
                                self.router_output_buf.handle,
                                self.routing_capture_buf.handle,
                                1,
                                &region,
                            );
                        }
                    } else {
                        self.decode_cmd.computeBufferBarrier(self.router_output_buf.handle, self.router_output_buf.size);
                    }
                    self.endProfilePhase(.moe_topk, moe_topk_phase);

                    // gate+up DMMV: ALL experts at once (Y=n_used workgroups).
                    // gate_exps[expert] × expert_input_buf → gate_buf[expert*inter_dim..]
                    // up_exps  [expert] × expert_input_buf → up_buf  [expert*inter_dim..]
                    // expert_input_buf is pre_ffw_norm_2 output for Gemma 4, otherwise ffn_norm_buf.
                    // For matching Q4_K gate and up tensors we dispatch the fused
                    // shader once — it reads expert_input_buf a single time and
                    // writes both outputs, halving the dispatch count for this
                    // phase. Otherwise we fall back to two separate dispatches.
                    const moe_gate_up_phase = self.beginProfilePhase();
                    const gate_qt = gate_exps.info.type_;
                    const up_qt = up_exps.info.type_;
                    const fused_gate_up_pip: ?*const Pipeline = if (hidden_dim == 2048 and
                        self.dmmv.pipeline_q4k_fused_gate_up_moe_spec8 != null)
                        &self.dmmv.pipeline_q4k_fused_gate_up_moe_spec8.?
                    else if (self.dmmv.pipeline_q4k_fused_gate_up_moe != null)
                        &self.dmmv.pipeline_q4k_fused_gate_up_moe.?
                    else
                        null;
                    const fused_gate_up_swiglu_pip: ?*const Pipeline = if (hidden_dim == 2048 and
                        self.dmmv.pipeline_q4k_fused_gate_up_swiglu_moe_spec8 != null)
                        &self.dmmv.pipeline_q4k_fused_gate_up_swiglu_moe_spec8.?
                    else if (self.dmmv.pipeline_q4k_fused_gate_up_swiglu_moe != null)
                        &self.dmmv.pipeline_q4k_fused_gate_up_swiglu_moe.?
                    else
                        null;
                    const fused_swiglu_ready = gate_qt == .q4_k and up_qt == .q4_k and
                        self.use_moe_kpar and
                        self.use_moe_fused_gate_up_swiglu and
                        fused_gate_up_swiglu_pip != null and
                        gate_exps.info.numElements() == up_exps.info.numElements();
                    const fused_ready = gate_qt == .q4_k and up_qt == .q4_k and
                        self.use_moe_kpar and
                        self.use_moe_fused_gate_up and
                        fused_gate_up_pip != null and
                        gate_exps.info.numElements() == up_exps.info.numElements();
                    if (fused_swiglu_ready) {
                        const pip = fused_gate_up_swiglu_pip.?;
                        const push = MoeDmmvPushConstants{ .M = inter_dim, .K = hidden_dim, .expert_stride = expert_gate_row_bytes, .x_expert_stride = 0, .x_offset = 0, .y_offset = 0 };
                        const wg_x: u32 = inter_dim;
                        self.pushDispatch5(
                            pip,
                            std.mem.asBytes(&push),
                            gate_exps.gpu_buffer.handle,
                            gate_exps.gpu_buffer.size,
                            up_exps.gpu_buffer.handle,
                            up_exps.gpu_buffer.size,
                            expert_input_buf.handle,
                            hidden_size,
                            self.swiglu_buf.handle,
                            self.swiglu_buf.size,
                            self.router_output_buf.handle,
                            self.router_output_buf.size,
                            wg_x,
                            n_used,
                            1,
                        );
                    } else if (fused_ready) {
                        const pip = fused_gate_up_pip.?;
                        const push = MoeDmmvPushConstants{ .M = inter_dim, .K = hidden_dim, .expert_stride = expert_gate_row_bytes, .x_expert_stride = 0, .x_offset = 0, .y_offset = 0 };
                        const wg_x: u32 = (inter_dim + 1) / 2;
                        self.pushDispatch6(
                            pip,
                            std.mem.asBytes(&push),
                            gate_exps.gpu_buffer.handle,
                            gate_exps.gpu_buffer.size,
                            up_exps.gpu_buffer.handle,
                            up_exps.gpu_buffer.size,
                            expert_input_buf.handle,
                            hidden_size,
                            self.gate_buf.handle,
                            self.gate_buf.size,
                            self.up_buf.handle,
                            self.up_buf.size,
                            self.router_output_buf.handle,
                            self.router_output_buf.size,
                            wg_x,
                            n_used,
                            1,
                        );
                    } else {
                        {
                            const qt = gate_qt;
                            const use_kpar = self.use_moe_kpar and qt == .q4_k and self.dmmv.pipeline_q4k_moe_kpar != null;
                            const pip = if (use_kpar) &self.dmmv.pipeline_q4k_moe_kpar.? else (self.dmmv.moePipelineForType(qt) orelse unreachable);
                            if (pip.uses_push_descriptors) {
                                const push = MoeDmmvPushConstants{ .M = inter_dim, .K = hidden_dim, .expert_stride = expert_gate_row_bytes, .x_expert_stride = 0, .x_offset = 0, .y_offset = 0 };
                                const wg_x: u32 = if (use_kpar) (inter_dim + 1) / 2 else switch (qt) {
                                    .mxfp4, .q8_0, .f16 => (inter_dim + 1) / 2,
                                    else => (inter_dim + 63) / 64,
                                };
                                self.pushDispatch4(pip, std.mem.asBytes(&push), gate_exps.gpu_buffer.handle, gate_exps.gpu_buffer.size, expert_input_buf.handle, hidden_size, self.gate_buf.handle, self.gate_buf.size, self.router_output_buf.handle, self.router_output_buf.size, wg_x, n_used, 1);
                            } else {
                                const ds = try self.allocDescSet(pip.descriptor_set_layout);
                                self.writeDescSet4(ds, gate_exps.gpu_buffer.handle, gate_exps.gpu_buffer.size, expert_input_buf.handle, hidden_size, self.gate_buf.handle, self.gate_buf.size, self.router_output_buf.handle, self.router_output_buf.size);
                                try self.dmmv.recordMoeDispatch(&self.decode_cmd, qt, ds, inter_dim, hidden_dim, expert_gate_row_bytes, n_used, 0, 0, 0);
                            }
                        }
                        {
                            const qt = up_qt;
                            const use_kpar = self.use_moe_kpar and qt == .q4_k and self.dmmv.pipeline_q4k_moe_kpar != null;
                            const pip = if (use_kpar) &self.dmmv.pipeline_q4k_moe_kpar.? else (self.dmmv.moePipelineForType(qt) orelse unreachable);
                            if (pip.uses_push_descriptors) {
                                const push = MoeDmmvPushConstants{ .M = inter_dim, .K = hidden_dim, .expert_stride = expert_gate_row_bytes, .x_expert_stride = 0, .x_offset = 0, .y_offset = 0 };
                                const wg_x: u32 = if (use_kpar) (inter_dim + 1) / 2 else switch (qt) {
                                    .mxfp4, .q8_0, .f16 => (inter_dim + 1) / 2,
                                    else => (inter_dim + 63) / 64,
                                };
                                self.pushDispatch4(pip, std.mem.asBytes(&push), up_exps.gpu_buffer.handle, up_exps.gpu_buffer.size, expert_input_buf.handle, hidden_size, self.up_buf.handle, self.up_buf.size, self.router_output_buf.handle, self.router_output_buf.size, wg_x, n_used, 1);
                            } else {
                                const ds = try self.allocDescSet(pip.descriptor_set_layout);
                                self.writeDescSet4(ds, up_exps.gpu_buffer.handle, up_exps.gpu_buffer.size, expert_input_buf.handle, hidden_size, self.up_buf.handle, self.up_buf.size, self.router_output_buf.handle, self.router_output_buf.size);
                                try self.dmmv.recordMoeDispatch(&self.decode_cmd, qt, ds, inter_dim, hidden_dim, expert_gate_row_bytes, n_used, 0, 0, 0);
                            }
                        }
                    }
                    self.decode_cmd.computeBarrier();
                    self.endProfilePhase(.moe_gate_up, moe_gate_up_phase);

                    // SwiGLU: ALL experts at once (N = n_used * inter_dim)
                    if (!fused_swiglu_ready) {
                        const moe_swiglu_phase = self.beginProfilePhase();
                        try self.dispatchFfnActivation(
                            self.gate_buf.handle,
                            self.gate_buf.size,
                            self.up_buf.handle,
                            self.up_buf.size,
                            self.swiglu_buf.handle,
                            self.swiglu_buf.size,
                            n_used * inter_dim,
                        );
                        self.decode_cmd.computeBarrier();
                        self.endProfilePhase(.moe_swiglu, moe_swiglu_phase);
                    }

                    // Shared expert tensors — looked up here to interleave with MoE dispatches
                    const gate_shexp = lt.ffn_gate_shexp;
                    const up_shexp = lt.ffn_up_shexp;
                    const down_shexp = lt.ffn_down_shexp;
                    const shexp_gate = lt.ffn_gate_inp_shexp;
                    // The accepted Qwen3.6 prefill cap already treats early
                    // non-terminal prompt tokens as dead-tail quality work by
                    // using fewer routed experts until the terminal guard.
                    // Apply the same guard to the shared expert: terminal
                    // prompt tokens and decode stay exact, while early prompt
                    // tokens skip the Q8 shared FFN tail.
                    const skip_prefill_shared_expert = use_prefill_tail_topk and config.architecture == .qwen2_moe;
                    const has_shared_expert = !skip_prefill_shared_expert and gate_shexp != null and up_shexp != null and down_shexp != null;
                    const shexp_size = @as(vk.c.VkDeviceSize, shexp_inter_dim) * @sizeOf(f32);

                    // Effort-6 cycle-11: fuse the shared expert (gate DMMV + up DMMV
                    // + SwiGLU) trio into one dispatch using the
                    // dmmv_*_fused_gate_up_swiglu shaders (Q4_K shader is shared
                    // with the dense FFN path; Q8_0 shader is new in this cycle
                    // for Qwen 3.5/3.6 MoE shared experts). The fused dispatch
                    // overlaps with MoE down (same as the current shared gate/up
                    // overlap) but writes to gate_buf instead of swiglu_buf, since
                    // swiglu_buf is being read concurrently by the MoE down DMMV.
                    // The shared down DMMV later reads from gate_buf in this
                    // path, eliminating one DMMV dispatch + one separate SwiGLU
                    // dispatch per layer per token.
                    const FusedShexpKind = enum { none, q4k, q8_0 };
                    const fused_shexp_kind: FusedShexpKind = blk: {
                        if (!has_shared_expert) break :blk .none;
                        if (!self.use_fused_dense_ffn) break :blk .none;
                        const g = gate_shexp orelse break :blk .none;
                        const u = up_shexp orelse break :blk .none;
                        if ((hidden_dim % 4) != 0) break :blk .none;
                        if ((hidden_dim % 256) != 0) break :blk .none;
                        if (g.info.type_ == .q4_k and u.info.type_ == .q4_k and
                            self.dmmv.pipeline_q4k_fused_gate_up_swiglu != null) break :blk .q4k;
                        if (g.info.type_ == .q8_0 and u.info.type_ == .q8_0 and
                            self.dmmv.pipeline_q8_0_fused_gate_up_swiglu != null and
                            (hidden_dim % 32) == 0) break :blk .q8_0;
                        break :blk .none;
                    };
                    const fused_shexp_eligible = fused_shexp_kind != .none;
                    const fused_shexp_gate_eligible = fused_shexp_kind == .q8_0 and
                        shexp_gate != null and
                        shexp_gate.?.info.type_ == .f32 and
                        self.dmmv.pipeline_q8_0_fused_gate_up_swiglu_gate != null;

                    // Fuse the shared-expert tail (down DMMV + sigmoid_scale_acc)
                    // when down_shexp is Q8_0 and the model uses the sigmoid-gated
                    // shared expert (Qwen 3.5 / 3.6 MoE). Saves 1 dispatch + 1
                    // barrier per layer per token. Requires no post_ffw_norm so
                    // the standalone in-place norm on down_buf doesn't apply.
                    const fused_shexp_tail_eligible = has_shared_expert and
                        shexp_gate != null and
                        lt.post_ffw_norm == null and
                        lt.post_ffw_norm_1 == null and
                        down_shexp != null and
                        down_shexp.?.info.type_ == .q8_0 and
                        self.dmmv.pipeline_q8_0_sigmoid_acc != null and
                        self.elementwise.pipeline_sigmoid_scale_acc != null;

                    if (state.position == 0 and layer == 0) {
                        log.info("FASTPATH: shared gate={s} up={s} down={s} gate_inp={s} fused_shexp={s} fused_gate={} fused_tail={}", .{
                            if (gate_shexp) |t| @tagName(t.info.type_) else "none",
                            if (up_shexp) |t| @tagName(t.info.type_) else "none",
                            if (down_shexp) |t| @tagName(t.info.type_) else "none",
                            if (shexp_gate) |t| @tagName(t.info.type_) else "none",
                            @tagName(fused_shexp_kind),
                            fused_shexp_gate_eligible,
                            fused_shexp_tail_eligible,
                        });
                    }

                    // Decide between the fused down+weighted_acc path and the
                    // legacy two-step down → weighted_acc path. The fused
                    // shader writes hidden_buf[row] += sum_e(weight_e * dot)
                    // in a single dispatch, eliminating the moe_weighted_acc
                    // dispatch (~0.52 ms / decode on Qwen 3.6 35B-A3B).
                    // Restricted to Q4_K / Q5_K experts with no post_ffw_norm
                    // and no per-expert down scale (Gemma 4 specific).
                    const down_qt = down_exps.info.type_;
                    const has_post_ffw_norm = lt.post_ffw_norm != null;
                    const has_per_expert_scale = lt.ffn_down_exps_scale != null;
                    const fused_pip_for_qt: ?*const Pipeline = blk: {
                        switch (down_qt) {
                            .q4_k => break :blk if (self.dmmv.pipeline_q4k_moe_fused_down_acc) |*p| p else null,
                            .q5_k => break :blk if (self.dmmv.pipeline_q5k_moe_fused_down_acc) |*p| p else null,
                            else => break :blk null,
                        }
                    };
                    const can_fuse_down_acc = self.use_moe_fused_down_acc and
                        !has_post_ffw_norm and
                        !has_per_expert_scale and
                        fused_pip_for_qt != null;

                    // down DMMV: ALL experts at once
                    // x_expert_stride=inter_dim: each expert reads from its own swiglu section
                    const moe_down_phase = self.beginProfilePhase();
                    if (can_fuse_down_acc) {
                        // Fused path: single dispatch produces weighted accumulation
                        // straight into hidden_buf. No down_buf intermediate needed.
                        const pip = fused_pip_for_qt.?;
                        const push = MoeFusedDownAccPushConstants{
                            .M = hidden_dim,
                            .K = inter_dim,
                            .expert_stride = expert_down_row_bytes,
                            .x_expert_stride = inter_dim,
                            .x_offset = 0,
                            .y_offset = 0,
                            .n_used = n_used,
                        };
                        // Cycle 28: both Q4_K and Q5_K fused_down_acc shaders
                        // run NUM_ROWS=4 (1024 WGs at hidden_dim=4096, 4×
                        // oversubscribed on R9700; halves dispatch count vs
                        // NUM_ROWS=2 and amortizes per-WG launch cost across
                        // 4 weighted-acc rows of the n_used-expert dot loop).
                        const wg_x: u32 = (hidden_dim + 3) / 4;
                        self.pushDispatch4(
                            pip,
                            std.mem.asBytes(&push),
                            down_exps.gpu_buffer.handle,
                            down_exps.gpu_buffer.size,
                            self.swiglu_buf.handle,
                            self.swiglu_buf.size,
                            self.hidden_buf.handle,
                            hidden_size,
                            self.router_output_buf.handle,
                            self.router_output_buf.size,
                            wg_x,
                            1,
                            1,
                        );
                    } else {
                        const qt = down_qt;
                        const use_q4k_kpar = self.use_moe_kpar and qt == .q4_k and self.dmmv.pipeline_q4k_moe_kpar != null;
                        const use_q5k_kpar = self.use_moe_q5k_kpar and qt == .q5_k and self.dmmv.pipeline_q5k_moe_kpar != null;
                        const use_kpar = use_q4k_kpar or use_q5k_kpar;
                        const pip = if (use_q4k_kpar) &self.dmmv.pipeline_q4k_moe_kpar.? else if (use_q5k_kpar) &self.dmmv.pipeline_q5k_moe_kpar.? else (self.dmmv.moePipelineForType(qt) orelse unreachable);
                        if (pip.uses_push_descriptors) {
                            const push = MoeDmmvPushConstants{ .M = hidden_dim, .K = inter_dim, .expert_stride = expert_down_row_bytes, .x_expert_stride = inter_dim, .x_offset = 0, .y_offset = 0 };
                            const wg_x: u32 = if (use_kpar) (hidden_dim + 1) / 2 else switch (qt) {
                                .mxfp4, .q8_0, .f16 => (hidden_dim + 1) / 2,
                                else => (hidden_dim + 63) / 64,
                            };
                            self.pushDispatch4(pip, std.mem.asBytes(&push), down_exps.gpu_buffer.handle, down_exps.gpu_buffer.size, self.swiglu_buf.handle, self.swiglu_buf.size, self.down_buf.handle, self.down_buf.size, self.router_output_buf.handle, self.router_output_buf.size, wg_x, n_used, 1);
                        } else {
                            const ds = try self.allocDescSet(pip.descriptor_set_layout);
                            self.writeDescSet4(ds, down_exps.gpu_buffer.handle, down_exps.gpu_buffer.size, self.swiglu_buf.handle, self.swiglu_buf.size, self.down_buf.handle, self.down_buf.size, self.router_output_buf.handle, self.router_output_buf.size);
                            try self.dmmv.recordMoeDispatch(&self.decode_cmd, qt, ds, hidden_dim, inter_dim, expert_down_row_bytes, n_used, inter_dim, 0, 0);
                        }
                    }
                    // Overlap: dispatch shared expert gate/up alongside MoE down.
                    // No buffer conflicts: MoE down reads swiglu_buf/writes down_buf
                    // (or fused: writes hidden_buf); shared gate/up read
                    // ffn_norm_buf/write gate_buf,up_buf,router_logits_buf.
                    // When fused_shexp is eligible, one fused dispatch replaces
                    // gate+up DMMV pair AND the later shared SwiGLU dispatch; the
                    // fused output goes to gate_buf so it doesn't collide with
                    // MoE down's concurrent swiglu_buf read.
                    if (has_shared_expert) {
                        switch (fused_shexp_kind) {
                            .q4k => try self.dispatchDmmvFusedGateUpSwiglu(
                                gate_shexp.?,
                                up_shexp.?,
                                self.ffn_norm_buf,
                                hidden_size,
                                self.gate_buf,
                                shexp_inter_dim,
                                hidden_dim,
                            ),
                            .q8_0 => try self.dispatchDmmvFusedGateUpSwigluQ8_0(
                                gate_shexp.?,
                                up_shexp.?,
                                self.ffn_norm_buf,
                                hidden_size,
                                self.gate_buf,
                                shexp_inter_dim,
                                hidden_dim,
                                if (fused_shexp_gate_eligible) shexp_gate else null,
                            ),
                            .none => {
                                try self.dispatchDmmv(gate_shexp.?, self.ffn_norm_buf, hidden_size, self.gate_buf, shexp_inter_dim, hidden_dim);
                                try self.dispatchDmmv(up_shexp.?, self.ffn_norm_buf, hidden_size, self.up_buf, shexp_inter_dim, hidden_dim);
                            },
                        }
                        if (shexp_gate) |sg| {
                            if (!fused_shexp_gate_eligible) {
                                try self.dispatchDmmv(sg, self.ffn_norm_buf, hidden_size, self.router_logits_buf, 1, hidden_dim);
                            }
                        }
                    }
                    self.decode_cmd.computeBarrier();
                    self.endProfilePhase(.moe_down, moe_down_phase);

                    // Gemma 4 MoE: per-expert scalar on down expert output before weighted_acc.
                    // down[slot*hidden_dim + i] *= ffn_down_exps_scale[expert_id[slot]].
                    // Matches Metal forward_metal.zig:4357-4360.
                    if (!can_fuse_down_acc and lt.ffn_down_exps_scale != null) {
                        const scale_t = lt.ffn_down_exps_scale.?;
                        try self.dispatchPerExpertScale(
                            self.down_buf.handle,
                            self.down_buf.size,
                            scale_t.gpu_buffer.handle,
                            scale_t.gpu_buffer.size,
                            self.router_output_buf.handle,
                            self.router_output_buf.size,
                            hidden_dim,
                            n_used,
                        );
                        self.decode_cmd.computeBarrier();
                    }

                    // Weighted accumulation: sum ALL experts at once.
                    // If post_ffw_norm is present, accumulate into moe_out_buf for normalization
                    // before residual add; otherwise accumulate directly into hidden_buf.
                    // (Skipped in the fused path — the fused down kernel already
                    // wrote hidden_buf += weighted accumulation.)
                    const moe_acc_phase = self.beginProfilePhase();
                    if (!can_fuse_down_acc) {
                        const moe_acc_target = if (has_post_ffw_norm) self.moe_out_buf.handle else self.hidden_buf.handle;
                        const moe_acc_target_size = if (has_post_ffw_norm) self.moe_out_buf.size else hidden_size;
                        if (has_post_ffw_norm) {
                            // Zero moe_out_buf before weighted accumulation
                            vk.c.vkCmdFillBuffer(self.decode_cmd.handle, self.moe_out_buf.handle, 0, hidden_size, 0);
                            self.decode_cmd.transferToComputeBarrier();
                        }
                        try self.dispatchMoeWeightedAcc(
                            moe_acc_target,
                            moe_acc_target_size,
                            self.down_buf.handle,
                            self.down_buf.size,
                            self.router_output_buf.handle,
                            self.router_output_buf.size,
                            hidden_dim,
                            n_used,
                            hidden_dim,
                        );
                    }
                    // Overlap: dispatch shared expert SwiGLU alongside weighted_acc.
                    // No buffer conflicts: weighted_acc reads down_buf+router_output_buf/writes hidden_buf;
                    // SwiGLU reads gate_buf+up_buf/writes swiglu_buf.
                    // Skipped when fused_shexp_eligible — the fused gate+up dispatch
                    // already produced silu(gate)*up directly into gate_buf.
                    if (has_shared_expert and !fused_shexp_eligible) {
                        try self.dispatchFfnActivation(
                            self.gate_buf.handle,
                            self.gate_buf.size,
                            self.up_buf.handle,
                            self.up_buf.size,
                            self.swiglu_buf.handle,
                            self.swiglu_buf.size,
                            shexp_inter_dim,
                        );
                    }
                    // Skip the phase fence when the fused MoE path already
                    // accumulated during moe_down and fused_shexp skipped the
                    // standalone shared SwiGLU dispatch.
                    const moe_acc_emitted_dispatch =
                        !can_fuse_down_acc or (has_shared_expert and !fused_shexp_eligible);
                    if (moe_acc_emitted_dispatch) {
                        self.decode_cmd.computeBarrier();
                    }
                    self.endProfilePhase(.moe_weighted_acc, moe_acc_phase);

                    // Shared expert down projection (run BEFORE post_ffw_norm for Gemma 4
                    // so that MoE + shared expert outputs are accumulated in moe_out_buf
                    // and normed together. Matches Metal forward_metal.zig:5110-5128.)
                    // Reads from gate_buf when fused_shexp_eligible (the fused
                    // dispatch wrote silu(gate)*up into gate_buf), otherwise from
                    // swiglu_buf (the legacy SwiGLU dispatch's output).
                    if (has_shared_expert) {
                        const shared_down_phase = self.beginProfilePhase();
                        const shexp_act_buf = if (fused_shexp_eligible) self.gate_buf else self.swiglu_buf;
                        if (fused_shexp_tail_eligible) {
                            // Fused: down_shexp matvec + sigmoid(shexp_gate) +
                            // accumulate directly into hidden_buf. Replaces the
                            // standalone dispatchDmmv → barrier → sigmoid_scale_acc
                            // pair below for the non-Gemma shared expert tail.
                            try self.dispatchDmmvQ8_0SigmoidAcc(
                                down_shexp.?,
                                shexp_act_buf,
                                shexp_size,
                                self.hidden_buf,
                                hidden_size,
                                self.router_logits_buf,
                                @sizeOf(f32),
                                hidden_dim,
                                shexp_inter_dim,
                                0,
                            );
                        } else {
                            try self.dispatchDmmv(down_shexp.?, shexp_act_buf, shexp_size, self.down_buf, hidden_dim, shexp_inter_dim);
                        }
                        self.decode_cmd.computeBarrier();
                        self.endProfilePhase(.shared_down, shared_down_phase);
                    }

                    // Post-FFN norm + residual for MoE expert accumulation (Gemma 4).
                    // When post_ffw_norm is present, first accumulate shared expert into
                    // moe_out_buf (with sigmoid gate or unity weight), THEN apply the norm
                    // to the combined result, then add to hidden_buf.
                    if (has_post_ffw_norm) {
                        // Gemma 4 MoE: apply post_ffw_norm_2 to MoE expert accumulation BEFORE
                        // shared expert is combined. Matches Metal forward_metal.zig:4309-4312.
                        if (lt.post_ffw_norm_2) |pfn2_tensor| {
                            try self.dispatchRmsNorm(
                                self.moe_out_buf.handle,
                                hidden_size,
                                pfn2_tensor.gpu_buffer.handle,
                                pfn2_tensor.gpu_buffer.size,
                                self.moe_out_buf.handle,
                                hidden_size,
                                hidden_dim,
                                1,
                                rms_norm_eps,
                            );
                            self.decode_cmd.computeBarrier();
                        }
                        if (has_shared_expert) {
                            // Gemma 4 MoE: apply post_ffw_norm_1 to shared expert output
                            // before combining. Matches Metal forward_metal.zig:4314-4317.
                            if (lt.post_ffw_norm_1) |pfn1_tensor| {
                                try self.dispatchRmsNorm(
                                    self.down_buf.handle,
                                    hidden_size,
                                    pfn1_tensor.gpu_buffer.handle,
                                    pfn1_tensor.gpu_buffer.size,
                                    self.down_buf.handle,
                                    hidden_size,
                                    hidden_dim,
                                    1,
                                    rms_norm_eps,
                                );
                                self.decode_cmd.computeBarrier();
                            }
                            // Accumulate shared expert down_buf into moe_out_buf (pre-final-norm)
                            const shared_gate_phase = self.beginProfilePhase();
                            if (shexp_gate != null and self.elementwise.pipeline_sigmoid_scale_acc != null) {
                                try self.dispatchSigmoidScaleAcc(
                                    self.moe_out_buf.handle,
                                    hidden_size,
                                    self.down_buf.handle,
                                    hidden_size,
                                    self.router_logits_buf.handle,
                                    @sizeOf(f32),
                                    hidden_dim,
                                );
                            } else {
                                try self.dispatchScaleAcc(
                                    self.moe_out_buf.handle,
                                    hidden_size,
                                    self.down_buf.handle,
                                    hidden_size,
                                    hidden_dim,
                                    1.0,
                                );
                            }
                            self.decode_cmd.computeBarrier();
                            self.endProfilePhase(.shared_gate_acc, shared_gate_phase);
                        }
                        if (lt.post_ffw_norm) |pfn_tensor| {
                            try self.dispatchRmsNorm(
                                self.moe_out_buf.handle,
                                hidden_size,
                                pfn_tensor.gpu_buffer.handle,
                                pfn_tensor.gpu_buffer.size,
                                self.moe_out_buf.handle,
                                hidden_size,
                                hidden_dim,
                                1,
                                rms_norm_eps,
                            );
                            self.decode_cmd.computeBarrier();
                        }
                        try self.dispatchScaleAcc(
                            self.hidden_buf.handle,
                            hidden_size,
                            self.moe_out_buf.handle,
                            hidden_size,
                            hidden_dim,
                            1.0,
                        );
                        self.decode_cmd.computeBarrier();
                    }

                    // Non-Gemma-4 path: shared expert still needs residual into hidden_buf
                    // separately (no post_ffw_norm to share). When the fused
                    // shared-expert tail (down + sigmoid_acc) ran above, this
                    // block already produced hidden_buf += sigmoid * down, so
                    // we skip the standalone sigmoid_scale_acc here.
                    if (has_shared_expert and !has_post_ffw_norm and !fused_shexp_tail_eligible) {
                        // Post-FFN norm on shared expert down projection (Gemma 4 non-post_ffw — unreachable in practice)
                        if (lt.post_ffw_norm) |pfn_tensor| {
                            try self.dispatchRmsNorm(
                                self.down_buf.handle,
                                hidden_size,
                                pfn_tensor.gpu_buffer.handle,
                                pfn_tensor.gpu_buffer.size,
                                self.down_buf.handle,
                                hidden_size,
                                hidden_dim,
                                1,
                                rms_norm_eps,
                            );
                            self.decode_cmd.computeBarrier();
                        }

                        // Shared expert accumulation into hidden_buf
                        const shared_gate_phase = self.beginProfilePhase();
                        if (shexp_gate != null and self.elementwise.pipeline_sigmoid_scale_acc != null) {
                            try self.dispatchSigmoidScaleAcc(
                                self.hidden_buf.handle,
                                hidden_size,
                                self.down_buf.handle,
                                hidden_size,
                                self.router_logits_buf.handle,
                                @sizeOf(f32),
                                hidden_dim,
                            );
                        } else if (shexp_gate != null) {
                            if (self.profile_enabled) self.profile_token_counters.cpu_shared_gate_fallbacks += 1;
                            {
                                const bar = vk.c.VkMemoryBarrier{
                                    .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                    .pNext = null,
                                    .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                                    .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
                                };
                                vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &bar, 0, null, 0, null);
                                const rgn = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = @sizeOf(f32) };
                                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.router_logits_buf.handle, self.router_staging.handle, 1, &rgn);
                            }
                            try self.decode_cmd.end();
                            try self.decode_cmd.submitAndWait(self.instance.compute_queue);
                            const gate_ptr: [*]const f32 = @ptrCast(@alignCast(self.router_staging.mapped.?));
                            const shexp_weight = 1.0 / (1.0 + @exp(-gate_ptr[0]));
                            if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                            try self.decode_cmd.reset();
                            try self.decode_cmd.begin();
                            try self.dispatchScaleAcc(
                                self.hidden_buf.handle,
                                hidden_size,
                                self.down_buf.handle,
                                hidden_size,
                                hidden_dim,
                                shexp_weight,
                            );
                        } else {
                            try self.dispatchScaleAcc(
                                self.hidden_buf.handle,
                                hidden_size,
                                self.down_buf.handle,
                                hidden_size,
                                hidden_dim,
                                1.0,
                            );
                        }
                        self.decode_cmd.computeBarrier();
                        self.endProfilePhase(.shared_gate_acc, shared_gate_phase);
                    }
                    // GPU MoE path: hidden_buf is fully barriered (weighted_acc or shared_gate_acc)
                    gpu_moe_barriers_cover_hidden = true;
                } else {
                    if (gemma_gpu_topk_moe) {
                        if (state.position == 0 and layer == 0) {
                            log.info("FASTPATH: Gemma GPU-topk MoE ENABLED (q4k gate+up+geglu, scaled fused down+acc, n_used={d})", .{n_used});
                        }
                        const moe_topk_phase = self.beginProfilePhase();
                        if (gemma_topk_scales_router) {
                            try self.dispatchSoftmaxTopkScaled(
                                self.router_logits_buf.handle,
                                @as(vk.c.VkDeviceSize, config.n_experts) * @sizeOf(f32),
                                self.router_output_buf.handle,
                                self.router_output_buf.size,
                                config.n_experts,
                                n_used,
                                gemma_router_scale,
                            );
                        } else {
                            try self.dispatchSoftmaxTopk(
                                self.router_logits_buf.handle,
                                @as(vk.c.VkDeviceSize, config.n_experts) * @sizeOf(f32),
                                self.router_output_buf.handle,
                                self.router_output_buf.size,
                                config.n_experts,
                                n_used,
                            );
                        }
                        if (defer_gemma_pre_ffw_norm_2) {
                            const pre_norm_t = lt.pre_ffw_norm_2.?;
                            try self.dispatchRmsNorm(
                                self.hidden_buf.handle,
                                hidden_size,
                                pre_norm_t.gpu_buffer.handle,
                                pre_norm_t.gpu_buffer.size,
                                self.residual_buf.handle,
                                hidden_size,
                                hidden_dim,
                                1,
                                rms_norm_eps,
                            );
                            self.decode_cmd.computeBarrier();
                        } else {
                            self.decode_cmd.computeBufferBarrier(self.router_output_buf.handle, self.router_output_buf.size);
                        }
                        self.endProfilePhase(.moe_topk, moe_topk_phase);

                        const moe_gate_up_phase = self.beginProfilePhase();
                        try self.dispatchDmmvMoeFusedGateUpGeglu(
                            gate_exps,
                            expert_input_buf,
                            hidden_size,
                            self.swiglu_buf,
                            inter_dim,
                            hidden_dim,
                            expert_gate_row_bytes,
                            up_base_offset,
                            n_used,
                        );
                        self.decode_cmd.computeBarrier();
                        self.endProfilePhase(.moe_gate_up, moe_gate_up_phase);

                        const moe_down_phase = self.beginProfilePhase();
                        if (down_exps.info.type_ == .q8_0) {
                            try self.dispatchDmmvQ8_0MoeFusedDownAccScaled(
                                down_exps,
                                self.swiglu_buf,
                                self.swiglu_buf.size,
                                self.moe_out_buf,
                                lt.ffn_down_exps_scale.?,
                                hidden_dim,
                                inter_dim,
                                expert_down_row_bytes,
                                n_used,
                            );
                        } else {
                            try self.dispatchDmmvQ5_1MoeFusedDownAccScaled(
                                down_exps,
                                self.swiglu_buf,
                                self.swiglu_buf.size,
                                self.moe_out_buf,
                                lt.ffn_down_exps_scale.?,
                                hidden_dim,
                                inter_dim,
                                expert_down_row_bytes,
                                n_used,
                            );
                        }
                        self.decode_cmd.computeBarrier();
                        self.endProfilePhase(.moe_down, moe_down_phase);
                    } else {
                        if (self.profile_enabled) self.profile_token_counters.cpu_moe_fallbacks += 1;
                        if (self.profile_enabled and !self.profile_logged_cpu_moe_fallback) {
                            self.profile_logged_cpu_moe_fallback = true;
                            log.info("PROFILE_FALLBACK: cpu_moe pos={d} layer={d} gate={s} up={s} down={s} q4k_moe={} q5k_moe={} softmax_topk={} weighted_acc={}", .{
                                state.position,
                                layer,
                                @tagName(gate_quant),
                                @tagName(up_exps.info.type_),
                                @tagName(down_quant),
                                self.dmmv.moePipelineForType(gate_quant) != null,
                                self.dmmv.moePipelineForType(down_quant) != null,
                                self.elementwise.pipeline_softmax_topk != null,
                                self.elementwise.pipeline_moe_weighted_acc != null,
                            });
                        }
                        // === CPU fallback: readback router logits, CPU softmax+topk ===
                        var expert_ids: [16]u32 = undefined;
                        var expert_weights: [16]f32 = undefined;
                        const diag_router_check = self.validation_diagnostics_enabled and config.architecture == .gpt_oss and collect_output and state.generated_tokens.items.len == 0 and hidden_dim <= 8192;
                        {
                            const barrier = vk.c.VkMemoryBarrier{
                                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                .pNext = null,
                                .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
                            };
                            vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, null, 0, null);
                            const router_size = @as(vk.c.VkDeviceSize, config.n_experts) * @sizeOf(f32);
                            const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = router_size };
                            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.router_logits_buf.handle, self.router_staging.handle, 1, &region);
                            if (diag_router_check) {
                                const input_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = hidden_size };
                                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, router_input_buf.handle, self.embed_staging.handle, 1, &input_region);
                            }
                        }
                        try self.decode_cmd.end();
                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);
                        const router_ptr: [*]f32 = @ptrCast(@alignCast(self.router_staging.mapped.?));
                        const router_logits = router_ptr[0..config.n_experts];
                        if (lt.ffn_gate_inp_bias) |bias| {
                            addBiasFromTensor(self, router_ptr, bias, config.n_experts);
                        }
                        if (diag_router_check) {
                            const input_ptr: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
                            const router_input = input_ptr[0..hidden_dim];
                            const mmap = self.model.mmap_data orelse return error.NoMmapData;
                            const router_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + router_tensor.info.offset);
                            var cpu_row_buf: [8192]f32 = undefined;
                            const cpu_router = try self.allocator.alloc(f32, config.n_experts);
                            defer self.allocator.free(cpu_router);
                            var router_max_diff: f32 = 0;
                            var router_max_idx: usize = 0;
                            var gpu_top_idx: usize = 0;
                            var cpu_top_idx: usize = 0;
                            var gpu_top_val: f32 = -std.math.inf(f32);
                            var cpu_top_val: f32 = -std.math.inf(f32);

                            for (0..config.n_experts) |row| {
                                dequantRow(mmap[router_off..], @intCast(row), hidden_dim, router_tensor.info.type_, cpu_row_buf[0..hidden_dim]);
                                var dot: f64 = 0;
                                for (0..hidden_dim) |i| dot += @as(f64, cpu_row_buf[i]) * @as(f64, router_input[i]);
                                cpu_router[row] = @floatCast(dot);
                            }
                            if (lt.ffn_gate_inp_bias) |bias| {
                                addBiasFromTensor(self, cpu_router.ptr, bias, config.n_experts);
                            }
                            for (0..config.n_experts) |i| {
                                const gpu_val = router_logits[i];
                                const cpu_val = cpu_router[i];
                                const diff = @abs(gpu_val - cpu_val);
                                if (diff > router_max_diff) {
                                    router_max_diff = diff;
                                    router_max_idx = i;
                                }
                                if (gpu_val > gpu_top_val) {
                                    gpu_top_val = gpu_val;
                                    gpu_top_idx = i;
                                }
                                if (cpu_val > cpu_top_val) {
                                    cpu_top_val = cpu_val;
                                    cpu_top_idx = i;
                                }
                            }
                            log.info("ROUTER_CHECK L{d} pos={d}: type={s} max_diff={d:.6} idx={d} gpu_top={d}({d:.6}) cpu_top={d}({d:.6})", .{
                                layer,
                                state.position,
                                @tagName(router_tensor.info.type_),
                                router_max_diff,
                                router_max_idx,
                                gpu_top_idx,
                                gpu_top_val,
                                cpu_top_idx,
                                cpu_top_val,
                            });
                        }
                        if (config.architecture == .gpt_oss) {
                            topKSoftmaxWeight(router_logits, n_used, expert_ids[0..n_used], expert_weights[0..n_used]);
                        } else {
                            topKSoftmax(router_logits, n_used, expert_ids[0..n_used], expert_weights[0..n_used]);
                        }

                        // New command buffer for expert FFN dispatch
                        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                        try self.decode_cmd.reset();
                        try self.decode_cmd.begin();

                        var cpu_moe_accum_opt: ?[]f32 = null;
                        defer if (cpu_moe_accum_opt) |buf| self.allocator.free(buf);
                        const diag_moe_detail = self.validation_diagnostics_enabled and config.architecture == .gpt_oss and collect_output and state.generated_tokens.items.len == 0;
                        if (diag_moe_detail) {
                            cpu_moe_accum_opt = try self.allocator.alloc(f32, hidden_dim);
                            @memset(cpu_moe_accum_opt.?, 0);
                        }

                        const gemma_batched_cpu_moe =
                            config.architecture == .gemma and
                            fused_gate_up != null and
                            gate_exps.info.type_ == .q4_k and
                            up_exps.info.type_ == .q4_k and
                            down_exps.info.type_ == .q5_1 and
                            lt.ffn_gate_exps_bias == null and
                            lt.ffn_up_exps_bias == null and
                            lt.ffn_down_exps_bias == null and
                            cpu_moe_accum_opt == null and
                            !diag_moe_detail and
                            self.dmmv.pipeline_q4k_moe_fused_gate_up_geglu != null and
                            self.dmmv.pipeline_q5_1_moe_fused_down_acc != null;
                        if (state.position == 0 and layer == 0 and gemma_batched_cpu_moe) {
                            log.info("FASTPATH: Gemma batched CPU-topk MoE ENABLED (q4k gate+up+geglu, q5_1 fused down+acc, n_used={d})", .{n_used});
                        }

                        if (gemma_batched_cpu_moe) {
                            const routing_u32: [*]u32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
                            for (0..n_used) |ei| {
                                routing_u32[ei] = expert_ids[ei];
                                var weight = expert_weights[ei];
                                if (lt.ffn_down_exps_scale) |scale_t| {
                                    if (self.model.mmap_data) |mmap| {
                                        const off = self.model.gguf_file.tensor_data_offset + scale_t.info.offset + @as(u64, expert_ids[ei]) * @sizeOf(f32);
                                        if (off + @sizeOf(f32) <= mmap.len) {
                                            const s_ptr: *const f32 = @ptrCast(@alignCast(mmap.ptr + off));
                                            weight *= s_ptr.*;
                                        }
                                    }
                                }
                                routing_u32[n_used + ei] = @bitCast(weight);
                            }
                            vk.c.vkCmdCopyBuffer(
                                self.decode_cmd.handle,
                                self.embed_staging.handle,
                                self.router_output_buf.handle,
                                1,
                                &vk.c.VkBufferCopy{
                                    .srcOffset = 0,
                                    .dstOffset = 0,
                                    .size = @as(vk.c.VkDeviceSize, n_used) * 2 * @sizeOf(u32),
                                },
                            );
                            self.decode_cmd.transferToComputeBarrier();

                            try self.dispatchDmmvMoeFusedGateUpGeglu(
                                gate_exps,
                                expert_input_buf,
                                hidden_size,
                                self.swiglu_buf,
                                inter_dim,
                                hidden_dim,
                                expert_gate_row_bytes,
                                up_base_offset,
                                n_used,
                            );
                            self.decode_cmd.computeBarrier();

                            try self.dispatchDmmvQ5_1MoeFusedDownAcc(
                                down_exps,
                                self.swiglu_buf,
                                self.swiglu_buf.size,
                                self.moe_out_buf,
                                hidden_dim,
                                inter_dim,
                                expert_down_row_bytes,
                                n_used,
                            );
                            self.decode_cmd.computeBarrier();
                        } else {
                            // Zero moe_out_buf via fill for the sequential per-expert path.
                            vk.c.vkCmdFillBuffer(self.decode_cmd.handle, self.moe_out_buf.handle, 0, hidden_size, 0);
                            self.decode_cmd.transferToComputeBarrier();

                            for (0..n_used) |ei| {
                                const eid = expert_ids[ei];
                                var weight = expert_weights[ei];
                                const gate_offset = eid * expert_gate_row_bytes;
                                const up_offset = eid * expert_gate_row_bytes + up_base_offset;
                                const down_offset = eid * expert_down_row_bytes;

                                const gemma_fused_gate_up_geglu =
                                    config.architecture == .gemma and
                                    fused_gate_up != null and
                                    gate_exps.info.type_ == .q4_k and
                                    up_exps.info.type_ == .q4_k and
                                    lt.ffn_gate_exps_bias == null and
                                    lt.ffn_up_exps_bias == null and
                                    self.dmmv.pipeline_q4k_fused_gate_up_geglu != null;

                                // Expert gate/up reads pre_ffw_norm_2 output (Gemma 4) or ffn_norm_buf.
                                if (gemma_fused_gate_up_geglu) {
                                    try self.dispatchDmmvFusedGateUpGegluOffset(
                                        gate_exps,
                                        expert_input_buf,
                                        hidden_size,
                                        self.swiglu_buf,
                                        inter_dim,
                                        hidden_dim,
                                        gate_offset,
                                        up_offset,
                                    );
                                    self.decode_cmd.computeBarrier();
                                } else {
                                    try self.dispatchDmmvWithOffset(gate_exps, expert_input_buf, hidden_size, self.gate_buf, inter_dim, hidden_dim, gate_offset);
                                    try self.dispatchDmmvWithOffset(up_exps, expert_input_buf, hidden_size, self.up_buf, inter_dim, hidden_dim, up_offset);
                                    if (lt.ffn_gate_exps_bias != null or lt.ffn_up_exps_bias != null) {
                                        self.decode_cmd.computeBarrier();
                                    }
                                    if (lt.ffn_gate_exps_bias) |bias| {
                                        try self.dispatchBiasAddSlice(self.gate_buf.handle, self.gate_buf.size, bias, eid * inter_dim, inter_dim);
                                    }
                                    if (lt.ffn_up_exps_bias) |bias| {
                                        try self.dispatchBiasAddSlice(self.up_buf.handle, self.up_buf.size, bias, eid * inter_dim, inter_dim);
                                    }
                                    self.decode_cmd.computeBarrier();

                                    try self.dispatchFfnActivation(
                                        self.gate_buf.handle,
                                        self.gate_buf.size,
                                        self.up_buf.handle,
                                        self.up_buf.size,
                                        self.swiglu_buf.handle,
                                        self.swiglu_buf.size,
                                        inter_dim,
                                    );
                                    self.decode_cmd.computeBarrier();
                                }

                                if (diag_moe_detail and ei == 0) {
                                    const inter_bytes = @as(vk.c.VkDeviceSize, inter_dim) * @sizeOf(f32);
                                    const up_off = inter_bytes;
                                    const swiglu_off = up_off + inter_bytes;

                                    try self.decode_cmd.end();
                                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                                    try self.decode_cmd.reset();
                                    try self.decode_cmd.begin();
                                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.gate_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                                        .srcOffset = 0,
                                        .dstOffset = 0,
                                        .size = inter_bytes,
                                    });
                                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.up_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                                        .srcOffset = 0,
                                        .dstOffset = up_off,
                                        .size = inter_bytes,
                                    });
                                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.swiglu_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                                        .srcOffset = 0,
                                        .dstOffset = swiglu_off,
                                        .size = inter_bytes,
                                    });
                                    try self.decode_cmd.end();
                                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                                    const dbg_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
                                    const gate_vals = dbg_ptr[0..inter_dim];
                                    const up_vals = dbg_ptr[@intCast(up_off / @sizeOf(f32))..][0..inter_dim];
                                    const gpu_swiglu = dbg_ptr[@intCast(swiglu_off / @sizeOf(f32))..][0..inter_dim];
                                    const cpu_swiglu = try self.allocator.alloc(f32, inter_dim);
                                    defer self.allocator.free(cpu_swiglu);
                                    cpuSwiGLUOai(gate_vals, up_vals, cpu_swiglu);

                                    var swiglu_max_diff: f32 = 0;
                                    for (0..inter_dim) |i| {
                                        const diff = @abs(gpu_swiglu[i] - cpu_swiglu[i]);
                                        if (diff > swiglu_max_diff) swiglu_max_diff = diff;
                                    }
                                    log.info("SWIGLU_OAI_CHECK L{d} E{d}: max_diff={d:.6} gpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}] cpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}]", .{
                                        layer,
                                        eid,
                                        swiglu_max_diff,
                                        gpu_swiglu[0],
                                        gpu_swiglu[1],
                                        gpu_swiglu[2],
                                        gpu_swiglu[3],
                                        cpu_swiglu[0],
                                        cpu_swiglu[1],
                                        cpu_swiglu[2],
                                        cpu_swiglu[3],
                                    });

                                    if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                                    try self.decode_cmd.reset();
                                    try self.decode_cmd.begin();
                                }

                                // Gemma 4 MoE: fold per-expert ffn_down_exps.scale into the accumulation weight.
                                // down[i] *= scales[eid] then weight*down == (weight*scales[eid]) * down.
                                if (lt.ffn_down_exps_scale) |scale_t| {
                                    // Read the scalar for this expert from CPU-mapped mmap if available.
                                    if (self.model.mmap_data) |mmap| {
                                        const off = self.model.gguf_file.tensor_data_offset + scale_t.info.offset + @as(u64, eid) * @sizeOf(f32);
                                        if (off + @sizeOf(f32) <= mmap.len) {
                                            const s_ptr: *const f32 = @ptrCast(@alignCast(mmap.ptr + off));
                                            weight *= s_ptr.*;
                                        }
                                    }
                                }

                                const gemma_fused_down_acc =
                                    config.architecture == .gemma and
                                    down_exps.info.type_ == .q5_1 and
                                    lt.ffn_down_exps_bias == null and
                                    cpu_moe_accum_opt == null and
                                    !diag_moe_detail and
                                    self.dmmv.pipeline_q5_1_acc != null;

                                if (gemma_fused_down_acc) {
                                    try self.dispatchDmmvQ5_1AccOffset(
                                        down_exps,
                                        self.swiglu_buf,
                                        self.swiglu_buf.size,
                                        self.moe_out_buf,
                                        hidden_dim,
                                        inter_dim,
                                        down_offset,
                                        weight,
                                    );
                                    self.decode_cmd.computeBarrier();
                                } else {
                                    try self.dispatchDmmvWithOffset(down_exps, self.swiglu_buf, self.swiglu_buf.size, self.down_buf, hidden_dim, inter_dim, down_offset);
                                    if (lt.ffn_down_exps_bias) |bias| {
                                        self.decode_cmd.computeBarrier();
                                        try self.dispatchBiasAddSlice(self.down_buf.handle, hidden_size, bias, eid * hidden_dim, hidden_dim);
                                    }
                                    self.decode_cmd.computeBarrier();

                                    if (diag_moe_detail and ei == 0) {
                                        const inter_bytes = @as(vk.c.VkDeviceSize, inter_dim) * @sizeOf(f32);
                                        const hidden_bytes = @as(vk.c.VkDeviceSize, hidden_dim) * @sizeOf(f32);
                                        const down_off = inter_bytes;

                                        try self.decode_cmd.end();
                                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                                        try self.decode_cmd.reset();
                                        try self.decode_cmd.begin();
                                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.swiglu_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                                            .srcOffset = 0,
                                            .dstOffset = 0,
                                            .size = inter_bytes,
                                        });
                                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.down_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                                            .srcOffset = 0,
                                            .dstOffset = down_off,
                                            .size = hidden_bytes,
                                        });
                                        try self.decode_cmd.end();
                                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                                        const dbg_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
                                        const swiglu_vals = dbg_ptr[0..inter_dim];
                                        const gpu_down = dbg_ptr[@intCast(down_off / @sizeOf(f32))..][0..hidden_dim];
                                        const mmap = self.model.mmap_data orelse return error.NoMmapData;
                                        const down_base_off: usize = @as(usize, @intCast(self.model.gguf_file.tensor_data_offset + down_exps.info.offset));
                                        const down_data_off = down_base_off + @as(usize, down_offset);
                                        const cpu_row_buf = try self.allocator.alloc(f32, inter_dim);
                                        defer self.allocator.free(cpu_row_buf);
                                        const cpu_down = try self.allocator.alloc(f32, hidden_dim);
                                        defer self.allocator.free(cpu_down);

                                        for (0..hidden_dim) |row| {
                                            dequantRow(mmap[down_data_off..], @intCast(row), inter_dim, down_exps.info.type_, cpu_row_buf);
                                            var dot: f64 = 0;
                                            for (0..inter_dim) |i| dot += @as(f64, cpu_row_buf[i]) * @as(f64, swiglu_vals[i]);
                                            cpu_down[row] = @floatCast(dot);
                                        }
                                        if (lt.ffn_down_exps_bias) |bias| {
                                            addBiasFromTensorSlice(self, cpu_down.ptr, bias, eid * hidden_dim, hidden_dim);
                                        }

                                        var down_max_diff: f32 = 0;
                                        var down_max_idx: usize = 0;
                                        for (0..hidden_dim) |i| {
                                            const diff = @abs(gpu_down[i] - cpu_down[i]);
                                            if (diff > down_max_diff) {
                                                down_max_diff = diff;
                                                down_max_idx = i;
                                            }
                                        }
                                        log.info("DOWN_EXPERT_CHECK L{d} E{d}: max_diff={d:.6} idx={d} gpu_max={d:.6} cpu_max={d:.6} gpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}] cpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}] type={s}", .{
                                            layer,
                                            eid,
                                            down_max_diff,
                                            down_max_idx,
                                            gpu_down[down_max_idx],
                                            cpu_down[down_max_idx],
                                            gpu_down[0],
                                            gpu_down[1],
                                            gpu_down[2],
                                            gpu_down[3],
                                            cpu_down[0],
                                            cpu_down[1],
                                            cpu_down[2],
                                            cpu_down[3],
                                            @tagName(down_exps.info.type_),
                                        });

                                        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                                        try self.decode_cmd.reset();
                                        try self.decode_cmd.begin();
                                    }

                                    if (cpu_moe_accum_opt) |cpu_moe_accum| {
                                        try self.decode_cmd.end();
                                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                                        try self.decode_cmd.reset();
                                        try self.decode_cmd.begin();
                                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.down_buf.handle, self.embed_staging.handle, 1, &vk.c.VkBufferCopy{
                                            .srcOffset = 0,
                                            .dstOffset = 0,
                                            .size = hidden_size,
                                        });
                                        try self.decode_cmd.end();
                                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                                        const down_ptr: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
                                        for (0..hidden_dim) |i| cpu_moe_accum[i] += weight * down_ptr[i];

                                        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                                        try self.decode_cmd.reset();
                                        try self.decode_cmd.begin();
                                    }

                                    try self.dispatchScaleAcc(
                                        self.moe_out_buf.handle,
                                        hidden_size,
                                        self.down_buf.handle,
                                        hidden_size,
                                        hidden_dim,
                                        weight,
                                    );
                                    self.decode_cmd.computeBarrier();
                                }
                            }
                        }

                        if (cpu_moe_accum_opt) |cpu_moe_accum| {
                            try self.decode_cmd.end();
                            try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                            try self.decode_cmd.reset();
                            try self.decode_cmd.begin();
                            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.moe_out_buf.handle, self.embed_staging.handle, 1, &vk.c.VkBufferCopy{
                                .srcOffset = 0,
                                .dstOffset = 0,
                                .size = hidden_size,
                            });
                            try self.decode_cmd.end();
                            try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                            const gpu_moe_ptr: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
                            var moe_max_diff: f32 = 0;
                            var moe_max_idx: usize = 0;
                            for (0..hidden_dim) |i| {
                                const diff = @abs(gpu_moe_ptr[i] - cpu_moe_accum[i]);
                                if (diff > moe_max_diff) {
                                    moe_max_diff = diff;
                                    moe_max_idx = i;
                                }
                            }
                            log.info("MOE_ACC_CHECK L{d}: max_diff={d:.6} idx={d} gpu_max={d:.6} cpu_max={d:.6} gpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}] cpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}]", .{
                                layer,
                                moe_max_diff,
                                moe_max_idx,
                                gpu_moe_ptr[moe_max_idx],
                                cpu_moe_accum[moe_max_idx],
                                gpu_moe_ptr[0],
                                gpu_moe_ptr[1],
                                gpu_moe_ptr[2],
                                gpu_moe_ptr[3],
                                cpu_moe_accum[0],
                                cpu_moe_accum[1],
                                cpu_moe_accum[2],
                                cpu_moe_accum[3],
                            });

                            if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                            try self.decode_cmd.reset();
                            try self.decode_cmd.begin();
                        }
                    }
                }
                self.endProfilePhase(.moe_routed, moe_phase);

                // Gemma 4 MoE CPU path: post_ffw_norm_2 on MoE accumulation before shared expert
                if (!use_gpu_moe and lt.post_ffw_norm_2 != null) {
                    if (lt.post_ffw_norm_2) |pfn2_t| {
                        try self.dispatchRmsNorm(
                            self.moe_out_buf.handle,
                            hidden_size,
                            pfn2_t.gpu_buffer.handle,
                            pfn2_t.gpu_buffer.size,
                            self.moe_out_buf.handle,
                            hidden_size,
                            hidden_dim,
                            1,
                            rms_norm_eps,
                        );
                        self.decode_cmd.computeBarrier();
                    }
                }

                // Shared expert for CPU MoE fallback only (GPU MoE handles shared expert inline above)
                if (!use_gpu_moe) {
                    const cpu_gate_shexp = lt.ffn_gate_shexp;
                    const cpu_up_shexp = lt.ffn_up_shexp;
                    const cpu_down_shexp = lt.ffn_down_shexp;
                    const cpu_shexp_gate = lt.ffn_gate_inp_shexp;
                    if (cpu_gate_shexp != null and cpu_up_shexp != null and cpu_down_shexp != null) {
                        const cpu_shexp_size = @as(vk.c.VkDeviceSize, shexp_inter_dim) * @sizeOf(f32);

                        try self.dispatchDmmv(cpu_gate_shexp.?, self.ffn_norm_buf, hidden_size, self.gate_buf, shexp_inter_dim, hidden_dim);
                        try self.dispatchDmmv(cpu_up_shexp.?, self.ffn_norm_buf, hidden_size, self.up_buf, shexp_inter_dim, hidden_dim);
                        if (cpu_shexp_gate) |sg| {
                            try self.dispatchDmmv(sg, self.ffn_norm_buf, hidden_size, self.router_logits_buf, 1, hidden_dim);
                        }
                        self.decode_cmd.computeBarrier();

                        if (self.validation_diagnostics_enabled and config.architecture == .gpt_oss and collect_output and state.generated_tokens.items.len == 0 and hidden_dim <= 8192 and cpu_shexp_gate != null) {
                            try self.decode_cmd.end();
                            try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                            try self.decode_cmd.reset();
                            try self.decode_cmd.begin();
                            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.router_logits_buf.handle, self.router_staging.handle, 1, &vk.c.VkBufferCopy{
                                .srcOffset = 0,
                                .dstOffset = 0,
                                .size = @sizeOf(f32),
                            });
                            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.ffn_norm_buf.handle, self.embed_staging.handle, 1, &vk.c.VkBufferCopy{
                                .srcOffset = 0,
                                .dstOffset = 0,
                                .size = hidden_size,
                            });
                            try self.decode_cmd.end();
                            try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                            const gate_ptr: [*]const f32 = @ptrCast(@alignCast(self.router_staging.mapped.?));
                            const norm_ptr: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
                            const gate_tensor = cpu_shexp_gate.?;
                            const mmap = self.model.mmap_data orelse return error.NoMmapData;
                            const gate_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + gate_tensor.info.offset);
                            var cpu_gate_w: [8192]f32 = undefined;
                            dequantRow(mmap[gate_off..], 0, hidden_dim, gate_tensor.info.type_, cpu_gate_w[0..hidden_dim]);
                            var cpu_gate_raw: f64 = 0;
                            for (0..hidden_dim) |i| cpu_gate_raw += @as(f64, cpu_gate_w[i]) * @as(f64, norm_ptr[i]);
                            const gpu_gate_raw = gate_ptr[0];
                            const cpu_gate_raw_f32: f32 = @floatCast(cpu_gate_raw);
                            const gpu_gate_sigmoid = 1.0 / (1.0 + @exp(-gpu_gate_raw));
                            const cpu_gate_sigmoid = 1.0 / (1.0 + @exp(-cpu_gate_raw_f32));
                            log.info("SHEXP_GATE_CHECK L{d} pos={d}: type={s} raw_gpu={d:.6} raw_cpu={d:.6} sig_gpu={d:.6} sig_cpu={d:.6} diff={d:.6}", .{
                                layer,
                                state.position,
                                @tagName(gate_tensor.info.type_),
                                gpu_gate_raw,
                                cpu_gate_raw_f32,
                                gpu_gate_sigmoid,
                                cpu_gate_sigmoid,
                                @abs(gpu_gate_raw - cpu_gate_raw_f32),
                            });

                            if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                            try self.decode_cmd.reset();
                            try self.decode_cmd.begin();
                        }

                        try self.dispatchFfnActivation(
                            self.gate_buf.handle,
                            self.gate_buf.size,
                            self.up_buf.handle,
                            self.up_buf.size,
                            self.swiglu_buf.handle,
                            self.swiglu_buf.size,
                            shexp_inter_dim,
                        );
                        self.decode_cmd.computeBarrier();

                        try self.dispatchDmmv(cpu_down_shexp.?, self.swiglu_buf, cpu_shexp_size, self.down_buf, hidden_dim, shexp_inter_dim);
                        self.decode_cmd.computeBarrier();

                        // Gemma 4 MoE: post_ffw_norm_1 on shared expert output BEFORE combining.
                        // Matches Metal forward_metal.zig:4314-4317.
                        if (lt.post_ffw_norm_1) |pfn1_t| {
                            try self.dispatchRmsNorm(
                                self.down_buf.handle,
                                hidden_size,
                                pfn1_t.gpu_buffer.handle,
                                pfn1_t.gpu_buffer.size,
                                self.down_buf.handle,
                                hidden_size,
                                hidden_dim,
                                1,
                                rms_norm_eps,
                            );
                            self.decode_cmd.computeBarrier();
                        }

                        const shexp_acc_buf = self.moe_out_buf.handle;
                        if (cpu_shexp_gate != null and self.elementwise.pipeline_sigmoid_scale_acc != null) {
                            try self.dispatchSigmoidScaleAcc(
                                shexp_acc_buf,
                                hidden_size,
                                self.down_buf.handle,
                                hidden_size,
                                self.router_logits_buf.handle,
                                @sizeOf(f32),
                                hidden_dim,
                            );
                        } else if (cpu_shexp_gate != null) {
                            if (self.profile_enabled) self.profile_token_counters.cpu_shared_gate_fallbacks += 1;
                            {
                                const bar = vk.c.VkMemoryBarrier{
                                    .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                                    .pNext = null,
                                    .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                                    .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
                                };
                                vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &bar, 0, null, 0, null);
                                const rgn = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = @sizeOf(f32) };
                                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.router_logits_buf.handle, self.router_staging.handle, 1, &rgn);
                            }
                            try self.decode_cmd.end();
                            try self.decode_cmd.submitAndWait(self.instance.compute_queue);
                            const gate_ptr: [*]const f32 = @ptrCast(@alignCast(self.router_staging.mapped.?));
                            const shexp_weight = 1.0 / (1.0 + @exp(-gate_ptr[0]));
                            if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                            try self.decode_cmd.reset();
                            try self.decode_cmd.begin();
                            try self.dispatchScaleAcc(
                                shexp_acc_buf,
                                hidden_size,
                                self.down_buf.handle,
                                hidden_size,
                                hidden_dim,
                                shexp_weight,
                            );
                        } else {
                            try self.dispatchScaleAcc(
                                shexp_acc_buf,
                                hidden_size,
                                self.down_buf.handle,
                                hidden_size,
                                hidden_dim,
                                1.0,
                            );
                        }
                        self.decode_cmd.computeBarrier();
                    }
                }

                // FFN residual: only needed for CPU MoE fallback (GPU MoE accumulated directly into hidden_buf)
                if (!use_gpu_moe) {
                    const diag_ffn_residual = self.validation_diagnostics_enabled and config.architecture == .gpt_oss and collect_output and state.generated_tokens.items.len == 0 and hidden_dim <= 8192;
                    // Gemma 4 MoE: apply final post_ffw_norm on combined (MoE + shared) result
                    // BEFORE residual add. This is the final MoE post-norm.
                    // Matches Metal forward_metal.zig:4322-4325.
                    if (lt.post_ffw_norm) |pfn_t| {
                        try self.dispatchRmsNorm(
                            self.moe_out_buf.handle,
                            hidden_size,
                            pfn_t.gpu_buffer.handle,
                            pfn_t.gpu_buffer.size,
                            self.moe_out_buf.handle,
                            hidden_size,
                            hidden_dim,
                            1,
                            rms_norm_eps,
                        );
                        self.decode_cmd.computeBarrier();
                    }

                    if (diag_ffn_residual) {
                        self.decode_cmd.computeToTransferBarrier();
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.hidden_buf.handle, self.residual_buf.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = 0,
                            .dstOffset = 0,
                            .size = hidden_size,
                        });
                        self.decode_cmd.transferToComputeBarrier();
                    }

                    try self.dispatchScaleAcc(
                        self.hidden_buf.handle,
                        hidden_size,
                        self.moe_out_buf.handle,
                        hidden_size,
                        hidden_dim,
                        1.0,
                    );

                    if (diag_ffn_residual) {
                        self.decode_cmd.computeBarrier();
                        try self.decode_cmd.end();
                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                        try self.decode_cmd.reset();
                        try self.decode_cmd.begin();
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.residual_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = 0,
                            .dstOffset = 0,
                            .size = hidden_size,
                        });
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.moe_out_buf.handle, self.ssm_hidden_staging.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = 0,
                            .dstOffset = 0,
                            .size = hidden_size,
                        });
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.hidden_buf.handle, self.embed_staging.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = 0,
                            .dstOffset = 0,
                            .size = hidden_size,
                        });
                        try self.decode_cmd.end();
                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                        const pre_hidden_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
                        const branch_ptr: [*]const f32 = @ptrCast(@alignCast(self.ssm_hidden_staging.mapped.?));
                        const post_hidden_ptr: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
                        var residual_max_diff: f32 = 0;
                        var residual_max_idx: usize = 0;
                        for (0..hidden_dim) |i| {
                            const want = pre_hidden_ptr[i] + branch_ptr[i];
                            const diff = @abs(post_hidden_ptr[i] - want);
                            if (diff > residual_max_diff) {
                                residual_max_diff = diff;
                                residual_max_idx = i;
                            }
                        }
                        log.info("FFN_RESIDUAL_CHECK L{d} pos={d}: max_diff={d:.6} idx={d} gpu={d:.6} cpu={d:.6} pre={d:.6} branch={d:.6}", .{
                            layer,
                            state.position,
                            residual_max_diff,
                            residual_max_idx,
                            post_hidden_ptr[residual_max_idx],
                            pre_hidden_ptr[residual_max_idx] + branch_ptr[residual_max_idx],
                            pre_hidden_ptr[residual_max_idx],
                            branch_ptr[residual_max_idx],
                        });

                        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                        try self.decode_cmd.reset();
                        try self.decode_cmd.begin();
                    }
                }
            } else {
                // Dense FFN: gate → up → SwiGLU → down → residual.
                // Effort-11 cycle-8: when the dense fused gate+up+SwiGLU
                // pipeline is loaded and the per-call gates pass (Q4_K
                // gate+up tensors, SwiGLU activation = non-Gemma + non-
                // gpt_oss, inter_dim ≤ 12288 to keep Gemma 4 31B's wider
                // FFN on the unfused path), one dispatch replaces the
                // (gate DMMV + up DMMV + swiglu) trio. Eliminates the
                // gate_buf and up_buf write+read round-trips and saves
                // one global compute barrier per layer. Cycle-7 attempted
                // gate+up only and reverted; this variant additionally
                // folds the SwiGLU inline so the freed buffers are
                // physically removed from the dense decode datapath.
                const dense_ffn_phase = self.beginProfilePhase();
                const gate_tensor = lt.ffn_gate orelse return error.TensorNotFound;
                const up_tensor = lt.ffn_up orelse return error.TensorNotFound;
                const down_tensor = lt.ffn_down orelse return error.TensorNotFound;
                const dense_prefill_validate_capture = self.use_qwen36_dense_prefill_validate and
                    self.prefill_active and
                    layer == self.dense_prefill_validate_layer and
                    self.prefill_current_token_idx < self.dense_prefill_validate_max_tokens and
                    self.dense_prefill_validate_norm_ref != null and
                    self.dense_prefill_validate_pre_hidden_ref != null and
                    self.dense_prefill_validate_post_hidden_ref != null and
                    self.dense_prefill_validate_gate_ref != null and
                    self.dense_prefill_validate_up_ref != null and
                    self.dense_prefill_validate_swiglu_ref != null and
                    self.dense_prefill_validate_down_ref != null;
                if (dense_prefill_validate_capture) {
                    const tok_idx = self.prefill_current_token_idx;
                    const dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * hidden_size;
                    self.decode_cmd.computeAndTransferBarrier();
                    const norm_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = dst_off, .size = hidden_size };
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.ffn_norm_buf.handle, self.dense_prefill_validate_norm_ref.?.handle, 1, &norm_region);
                    const pre_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = dst_off, .size = hidden_size };
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.hidden_buf.handle, self.dense_prefill_validate_pre_hidden_ref.?.handle, 1, &pre_region);
                    self.decode_cmd.transferToComputeBarrier();
                    self.dense_prefill_validate_captured_tokens = @max(self.dense_prefill_validate_captured_tokens, tok_idx + 1);
                }

                const qwen36_row1_dense_eligible = self.use_qwen36_dense_fused_row1 and
                    self.dmmv.pipeline_q4k_fused_gate_up_swiglu_row1 != null and
                    self.isQwen36DenseHybrid27B() and
                    hidden_dim == 5120 and
                    inter_dim == 17408;
                const fused_dense_ffn_eligible = self.use_fused_dense_ffn and
                    self.dmmv.pipeline_q4k_fused_gate_up_swiglu != null and
                    config.architecture != .gemma and
                    config.architecture != .gpt_oss and
                    !dense_prefill_validate_capture and
                    gate_tensor.info.type_ == .q4_k and
                    up_tensor.info.type_ == .q4_k and
                    (inter_dim <= 12288 or qwen36_row1_dense_eligible) and
                    (hidden_dim % 4) == 0 and
                    (hidden_dim % 256) == 0;

                const dense_ffn_gateup_phase = self.beginProfilePhase();
                if (fused_dense_ffn_eligible) {
                    try self.dispatchDmmvFusedGateUpSwiglu(gate_tensor, up_tensor, self.ffn_norm_buf, hidden_size, self.swiglu_buf, inter_dim, hidden_dim);
                    self.decode_cmd.computeBufferBarrier(self.swiglu_buf.handle, self.swiglu_buf.size);
                } else {
                    const dense_ffn_gate_phase = self.beginProfilePhase();
                    try self.dispatchDmmv(gate_tensor, self.ffn_norm_buf, hidden_size, self.gate_buf, inter_dim, hidden_dim);
                    self.endProfilePhase(.dense_ffn_gate, dense_ffn_gate_phase);
                    const dense_ffn_up_phase = self.beginProfilePhase();
                    try self.dispatchDmmv(up_tensor, self.ffn_norm_buf, hidden_size, self.up_buf, inter_dim, hidden_dim);
                    self.endProfilePhase(.dense_ffn_up, dense_ffn_up_phase);
                    const dense_gateup_ranges = [_]CommandBuffer.BufferRange{
                        .{ .buffer = self.gate_buf.handle, .size = self.gate_buf.size },
                        .{ .buffer = self.up_buf.handle, .size = self.up_buf.size },
                    };
                    self.decode_cmd.computeBuffersBarrier(&dense_gateup_ranges);

                    if (dense_prefill_validate_capture) {
                        const tok_idx = self.prefill_current_token_idx;
                        const inter_size = @as(vk.c.VkDeviceSize, inter_dim) * @sizeOf(f32);
                        const dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * inter_size;
                        self.decode_cmd.computeAndTransferBarrier();
                        const gate_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = dst_off, .size = inter_size };
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.gate_buf.handle, self.dense_prefill_validate_gate_ref.?.handle, 1, &gate_region);
                        const up_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = dst_off, .size = inter_size };
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.up_buf.handle, self.dense_prefill_validate_up_ref.?.handle, 1, &up_region);
                        self.decode_cmd.transferToComputeBarrier();
                    }

                    try self.dispatchFfnActivation(
                        self.gate_buf.handle,
                        self.gate_buf.size,
                        self.up_buf.handle,
                        self.up_buf.size,
                        self.swiglu_buf.handle,
                        self.swiglu_buf.size,
                        inter_dim,
                    );
                    self.decode_cmd.computeBufferBarrier(self.swiglu_buf.handle, self.swiglu_buf.size);
                    if (dense_prefill_validate_capture) {
                        const tok_idx = self.prefill_current_token_idx;
                        const inter_size = @as(vk.c.VkDeviceSize, inter_dim) * @sizeOf(f32);
                        const dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * inter_size;
                        self.decode_cmd.computeAndTransferBarrier();
                        const swiglu_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = dst_off, .size = inter_size };
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.swiglu_buf.handle, self.dense_prefill_validate_swiglu_ref.?.handle, 1, &swiglu_region);
                        self.decode_cmd.transferToComputeBarrier();
                    }
                }
                self.endProfilePhase(.dense_ffn_gateup, dense_ffn_gateup_phase);

                // Fast path: fuse down proj + post_ffw_norm + residual add into
                // two dispatches when the Gemma tail is active and the fused
                // rms_norm_add pipeline is loaded. Saves one dispatch + one
                // barrier per Gemma decode layer (60 on gemma4-31b).
                const use_fused_pfn_decode = lt.post_ffw_norm != null and
                    self.elementwise.pipeline_rms_norm_add != null and
                    !self.validation_diagnostics_enabled;
                const dense_ffn_down_phase = self.beginProfilePhase();
                if (lt.post_ffw_norm == null and !self.validation_diagnostics_enabled and !dense_prefill_validate_capture) {
                    // Fused: down DMMV accumulates directly into hidden_buf,
                    // eliminating separate scale_acc dispatch + barrier
                    try self.dispatchDmmvAcc(down_tensor, self.swiglu_buf, self.swiglu_buf.size, self.hidden_buf, hidden_dim, inter_dim);
                } else if (use_fused_pfn_decode) {
                    try self.dispatchDmmv(down_tensor, self.swiglu_buf, self.swiglu_buf.size, self.down_buf, hidden_dim, inter_dim);
                    self.decode_cmd.computeBarrier();
                    const pfn_tensor = lt.post_ffw_norm.?;
                    try self.dispatchRmsNormAdd(
                        self.hidden_buf.handle,
                        hidden_size,
                        self.down_buf.handle,
                        hidden_size,
                        pfn_tensor.gpu_buffer.handle,
                        pfn_tensor.gpu_buffer.size,
                        hidden_dim,
                        1,
                        rms_norm_eps,
                    );
                } else {
                    // Unfused path: needed for Gemma post-FFN norm or diagnostics
                    try self.dispatchDmmv(down_tensor, self.swiglu_buf, self.swiglu_buf.size, self.down_buf, hidden_dim, inter_dim);
                    self.decode_cmd.computeBarrier();

                    // Gemma post-FFN norm: RMS norm on down_proj output before residual add
                    if (lt.post_ffw_norm) |pfn_tensor| {
                        try self.dispatchRmsNorm(
                            self.down_buf.handle,
                            hidden_size,
                            pfn_tensor.gpu_buffer.handle,
                            pfn_tensor.gpu_buffer.size,
                            self.down_buf.handle,
                            hidden_size,
                            hidden_dim,
                            1,
                            rms_norm_eps,
                        );
                        self.decode_cmd.computeBarrier();
                    }

                    if (dense_prefill_validate_capture) {
                        const tok_idx = self.prefill_current_token_idx;
                        const dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * hidden_size;
                        self.decode_cmd.computeAndTransferBarrier();
                        const down_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = dst_off, .size = hidden_size };
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.down_buf.handle, self.dense_prefill_validate_down_ref.?.handle, 1, &down_region);
                        self.decode_cmd.transferToComputeBarrier();
                    }

                    if (state.position == 0 and self.validation_diagnostics_enabled and layer == 0 and inter_dim <= 8192) {
                        try self.decode_cmd.end();
                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                        try self.decode_cmd.reset();
                        try self.decode_cmd.begin();
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.swiglu_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = 0,
                            .dstOffset = 0,
                            .size = @as(vk.c.VkDeviceSize, inter_dim) * @sizeOf(f32),
                        });
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.down_buf.handle, self.embed_staging.handle, 1, &vk.c.VkBufferCopy{
                            .srcOffset = 0,
                            .dstOffset = 0,
                            .size = hidden_size,
                        });
                        try self.decode_cmd.end();
                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                        const sw_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
                        const sw_vals = sw_ptr[0..inter_dim];
                        const dn_ptr: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
                        const mmap = self.model.mmap_data orelse return error.NoMmapData;
                        const down_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + down_tensor.info.offset);
                        var cpu_row_buf: [8192]f32 = undefined;
                        var cpu_vals: [4]f32 = [_]f32{0} ** 4;
                        const down_rows: u32 = @min(hidden_dim, cpu_vals.len);
                        var down_max_diff: f32 = 0;
                        for (0..down_rows) |row| {
                            dequantRow(mmap[down_off..], @intCast(row), inter_dim, down_tensor.info.type_, cpu_row_buf[0..inter_dim]);
                            var dot: f64 = 0;
                            for (0..inter_dim) |i| dot += @as(f64, cpu_row_buf[i]) * @as(f64, sw_vals[i]);
                            cpu_vals[row] = @floatCast(dot);
                            const diff = @abs(dn_ptr[row] - cpu_vals[row]);
                            if (diff > down_max_diff) down_max_diff = diff;
                        }
                        log.info("DMMV_CHECK: ffn_down type={s} M={d} K={d} gpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}] cpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}] max_diff={d:.6} ok={s}", .{
                            @tagName(down_tensor.info.type_),
                            hidden_dim,
                            inter_dim,
                            dn_ptr[0],
                            dn_ptr[1],
                            dn_ptr[2],
                            dn_ptr[3],
                            cpu_vals[0],
                            cpu_vals[1],
                            cpu_vals[2],
                            cpu_vals[3],
                            down_max_diff,
                            if (down_max_diff < 0.1) @as([]const u8, "YES") else @as([]const u8, "NO"),
                        });

                        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                        try self.decode_cmd.reset();
                        try self.decode_cmd.begin();
                    }

                    // FFN residual: hidden_buf += down_buf
                    if (!use_fused_pfn_decode) {
                        try self.dispatchScaleAcc(
                            self.hidden_buf.handle,
                            hidden_size,
                            self.down_buf.handle,
                            hidden_size,
                            hidden_dim,
                            1.0,
                        );
                    }
                }
                if (dense_prefill_validate_capture) {
                    const tok_idx = self.prefill_current_token_idx;
                    const dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * hidden_size;
                    self.decode_cmd.computeAndTransferBarrier();
                    const post_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = dst_off, .size = hidden_size };
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.hidden_buf.handle, self.dense_prefill_validate_post_hidden_ref.?.handle, 1, &post_region);
                    // Copy reads hidden_buf; keep layer-boundary compute sync.
                    self.decode_cmd.computeBarrier();
                }
                self.endProfilePhase(.dense_ffn_down, dense_ffn_down_phase);
                self.endProfilePhase(.dense_ffn, dense_ffn_phase);
            }

            // Per-layer output scaling (Gemma 4 proportional): hidden_buf *= scale
            const layer_output_scale = self.layer_output_scales[layer];
            if (layer_output_scale != 1.0) {
                if (!gpu_moe_barriers_cover_hidden) {
                    self.decode_cmd.computeBarrier();
                }
                try self.dispatchScaleInPlace(
                    self.hidden_buf.handle,
                    hidden_size,
                    hidden_dim,
                    layer_output_scale,
                );
                gpu_moe_barriers_cover_hidden = false;
            }

            // The next layer immediately reads hidden_buf as its input.
            // GPU MoE path already barriered hidden_buf after weighted_acc/shared_gate_acc.
            if (!gpu_moe_barriers_cover_hidden) {
                self.decode_cmd.computeBufferBarrier(self.hidden_buf.handle, hidden_size);
            }

            // Command buffer stays open across layers (Phase 3c batching).
            // No per-layer submit — only submit for MoE expert ID readback (inside MoE block above).

            // --- Debug: per-layer hidden_buf diagnostics (BOS token only, gated behind validation diagnostics) ---
            if ((state.position == 0 and (self.validation_diagnostics_enabled or getenv("ZINC_LAYER_DIAG") != null)) or
                (diag_last_prompt_token and self.validation_diagnostics_enabled))
            {
                // Flush current batched cmd buffer for diagnostic readback
                try self.decode_cmd.end();
                try self.decode_cmd.submitAndWait(self.instance.compute_queue);
                if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                try self.decode_cmd.reset();
                try self.decode_cmd.begin();
                const diag_rgn = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = hidden_size };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.hidden_buf.handle, self.embed_staging.handle, 1, &diag_rgn);
                try self.decode_cmd.end();
                try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                const hptr: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
                var diag_sum_sq: f64 = 0;
                var diag_max_abs: f32 = 0;
                for (0..hidden_dim) |i| {
                    diag_sum_sq += @as(f64, hptr[i]) * @as(f64, hptr[i]);
                    const a = @abs(hptr[i]);
                    if (a > diag_max_abs) diag_max_abs = a;
                }
                const diag_rms: f32 = @floatCast(@sqrt(diag_sum_sq / @as(f64, @floatFromInt(hidden_dim))));

                // Compute logit for token 5 via: hidden → CPU RMS_norm(output_norm) → dot(LM_head[5])
                // Reference value without layers: 2.5385 (from embed diagnostic)
                // Tracking this through layers pinpoints where the model diverges
                var logit5: f32 = 0;
                if (hidden_dim <= 8192) {
                    if (self.model.mmap_data) |m| {
                        const rms_inv: f32 = @floatCast(1.0 / @sqrt(diag_sum_sq / @as(f64, @floatFromInt(hidden_dim)) + rms_norm_eps));
                        const norm_t = self.tensor_map.get("output_norm.weight");
                        const lm_t = self.tensor_map.get("output.weight") orelse
                            self.tensor_map.get("token_embd.weight");
                        if (norm_t != null and lm_t != null) {
                            const norm_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + norm_t.?.info.offset);
                            const lm_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + lm_t.?.info.offset);
                            var norm_w: [8192]f32 = undefined;
                            var lm_row: [8192]f32 = undefined;
                            dequantRow(m[norm_off..], 0, hidden_dim, norm_t.?.info.type_, norm_w[0..hidden_dim]);
                            dequantRow(m[lm_off..], 5, hidden_dim, lm_t.?.info.type_, lm_row[0..hidden_dim]);
                            var dot: f64 = 0;
                            for (0..hidden_dim) |i| {
                                const normed = @as(f64, norm_w[i]) * @as(f64, hptr[i]) * @as(f64, rms_inv);
                                dot += normed * @as(f64, lm_row[i]);
                            }
                            logit5 = @floatCast(dot);
                        }
                    }
                }

                // Dump hidden[0..8] after layer 0 for CPU reference comparison
                if (layer == 0) {
                    log.info("L0_HIDDEN[0..8]: [{d:.8},{d:.8},{d:.8},{d:.8},{d:.8},{d:.8},{d:.8},{d:.8}]", .{
                        hptr[0], hptr[1], hptr[2], hptr[3], hptr[4], hptr[5], hptr[6], hptr[7],
                    });
                }

                // Also log tensor quant types on first layer to identify untested DMMV paths
                if (layer == 0) {
                    const lt0 = self.layer_tensors[0];
                    const qt_attn_norm = if (lt0.attn_norm) |t| @tagName(t.info.type_) else "?";
                    const qt_qkv = if (lt0.attn_qkv) |t| @tagName(t.info.type_) else "?";
                    const qt_gate_exps = if (lt0.ffn_gate_exps) |t| @tagName(t.info.type_) else "?";
                    const qt_down_exps = if (lt0.ffn_down_exps) |t| @tagName(t.info.type_) else "?";
                    const qt_ssm_out = if (lt0.ssm_out) |t| @tagName(t.info.type_) else "?";
                    log.info("QUANT: attn_norm={s} qkv={s} gate_exps={s} down_exps={s} ssm_out={s}", .{
                        qt_attn_norm, qt_qkv, qt_gate_exps, qt_down_exps, qt_ssm_out,
                    });
                }

                if (layer < 64) {
                    diag_logit5[layer] = logit5;
                    diag_rms_arr[layer] = diag_rms;
                }
                log.info("p{d}L{d}{s}: h[0..4]=[{d:.8},{d:.8},{d:.8},{d:.8}] rms={d:.6}", .{
                    state.position,
                    layer,
                    if (is_full_attn) @as([]const u8, "A") else @as([]const u8, "S"),
                    hptr[0],
                    hptr[1],
                    hptr[2],
                    hptr[3],
                    diag_rms,
                });
                // Re-open cmd buffer for next layer (diagnostic closed it)
                if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                try self.decode_cmd.reset();
                try self.decode_cmd.begin();
            }
        }

        // === Per-layer diagnostic summary (stored for printing after generation) ===
        if (state.position == 0 and config.n_layers <= 64 and self.validation_diagnostics_enabled) {
            // Store compact logit5 trajectory — shows how logit for token 5 evolves through layers
            // Reference: without layers, logit5=2.5385. With correct layers, should converge to model's prediction.
            var pos: usize = 0;
            for (0..config.n_layers) |li| {
                const val = diag_logit5[li];
                const rms_val = diag_rms_arr[li];
                const fai: usize = @intCast(full_attn_interval);
                const is_attn = ((li + 1) % fai == 0);
                const label: u8 = if (is_attn) 'A' else 'S';
                const written = std.fmt.bufPrint(self.diag_summary_buf[pos..], "{c}{d}:{d:.2}/{d:.1} ", .{ label, li, val, rms_val }) catch break;
                pos += written.len;
            }
            self.diag_summary_len = pos;
        }

        // === Final norm + LM head (after all layers) ===
        // Stay in the same command buffer so decode uses a single queue submit.
        // Skipped for non-terminal prefill tokens because nothing reads logits_buf
        // or norm_buf for those — the next prefill token overwrites hidden_buf via
        // embedding upload before needing any derived state.
        const have_gpu_argmax = self.argmax.pipeline != null and self.argmax_descriptor_set != null;
        const allow_final_tail = !partial_layer_decode or self.partial_decode_allow_final_tail;
        const need_logits_readback = collect_output and allow_final_tail and (self.logits_readback_enabled or self.validation_diagnostics_enabled or !have_gpu_argmax);
        if (collect_output and allow_final_tail) {
            const final_tail_phase = self.beginProfilePhase();

            // Final RMS norm: hidden_buf → norm_buf
            const final_norm_phase = self.beginProfilePhase();
            const final_norm_tensor = self.tensor_map.get("output_norm.weight") orelse return error.TensorNotFound;
            try self.dispatchRmsNorm(
                self.hidden_buf.handle,
                hidden_size,
                final_norm_tensor.gpu_buffer.handle,
                final_norm_tensor.gpu_buffer.size,
                self.norm_buf.handle,
                hidden_size,
                hidden_dim,
                1,
                rms_norm_eps,
            );
            self.decode_cmd.computeBarrier();
            self.endProfilePhase(.final_norm, final_norm_phase);

            // LM head: output.weight × norm_buf → logits_buf
            const final_lm_head_phase = self.beginProfilePhase();
            const lm_tensor = self.tensor_map.get("output.weight") orelse
                self.tensor_map.get("token_embd.weight") orelse return error.TensorNotFound;
            // Effort-6 Step 1 wire-in (ZINC_MUL_MM_LM_HEAD=1): route the LM head
            // through the tiled mul_mm_q4k pipeline instead of dispatchDmmv.
            // Eligibility: flag on, weight is Q4_K, hidden_dim multiple of 256
            // (Q4_K super-block size). Uses N=1 (LM head sees one final
            // activation row at the dead-tail token); the BN=16 tile is
            // mostly idle for N=1, so this exists primarily to validate
            // shader correctness on a real Q4_K matvec — Step 2 (MUL_MAT_ID
            // variant) reuses the same shader for the MoE phase where the
            // tile is saturated.
            const use_mul_mm_path = self.use_mul_mm_lm_head and
                lm_tensor.info.type_ == .q4_k and
                (hidden_dim % 256) == 0;
            const use_q8_1_lm_path = self.use_q8_1_lm_head and
                lm_tensor.info.type_ == .q8_0 and
                (hidden_dim & 31) == 0 and
                self.dmmv.pipeline_q8_0_q8_1 != null and
                self.dmmv.pipeline_quantize_q8_1 != null;
            const use_q8_batch_lm_path = self.use_q8_batch_lm_head and
                lm_tensor.info.type_ == .q8_0 and
                self.dmmv.pipeline_q8_0_batch != null;
            if (use_q8_1_lm_path) {
                try self.dmmv.recordQuantizeQ8_1(
                    &self.decode_cmd,
                    self.instance.push_descriptor_fn,
                    self.norm_buf.handle,
                    hidden_size,
                    self.q8_1_buf.handle,
                    self.q8_1_buf.size,
                    hidden_dim,
                );
                self.decode_cmd.computeBarrier();
                const q8_1_pip = &self.dmmv.pipeline_q8_0_q8_1.?;
                const q8_1_push = DmmvPushConstants{
                    .M = self.model.config.vocab_size,
                    .K = hidden_dim,
                    .a_offset = 0,
                    .x_offset = 0,
                    .y_offset = 0,
                    .acc_mode = 0,
                };
                self.pushDispatch3(
                    q8_1_pip,
                    std.mem.asBytes(&q8_1_push),
                    lm_tensor.gpu_buffer.handle,
                    lm_tensor.gpu_buffer.size,
                    self.q8_1_buf.handle,
                    self.q8_1_buf.size,
                    self.logits_buf.handle,
                    self.logits_buf.size,
                    (self.model.config.vocab_size + 1) / 2,
                    1,
                    1,
                );
            } else if (use_q8_batch_lm_path) {
                const q8_batch_pip = &self.dmmv.pipeline_q8_0_batch.?;
                const q8_batch_push = DmmvPushConstants{
                    .M = self.model.config.vocab_size,
                    .K = hidden_dim,
                    .a_offset = 0,
                    .x_offset = 0,
                    .y_offset = 0,
                    .acc_mode = 0,
                };
                self.pushDispatch3(
                    q8_batch_pip,
                    std.mem.asBytes(&q8_batch_push),
                    lm_tensor.gpu_buffer.handle,
                    lm_tensor.gpu_buffer.size,
                    self.norm_buf.handle,
                    hidden_size,
                    self.logits_buf.handle,
                    self.logits_buf.size,
                    (self.model.config.vocab_size + 63) / 64,
                    1,
                    1,
                );
            } else if (use_mul_mm_path) {
                try self.dmmv.recordMulMmQ4K(
                    &self.decode_cmd,
                    self.instance.push_descriptor_fn,
                    lm_tensor.gpu_buffer.handle,
                    lm_tensor.gpu_buffer.size,
                    self.norm_buf.handle,
                    hidden_size,
                    self.logits_buf.handle,
                    self.logits_buf.size,
                    self.model.config.vocab_size, // M
                    1, // N
                    hidden_dim, // K
                    hidden_dim, // stride_b: per-col floats in B (one column = K elements)
                    self.model.config.vocab_size, // stride_d: per-col floats in D (one column = M elements)
                    0,
                    0,
                    0,
                );
            } else {
                try self.dispatchDmmv(lm_tensor, self.norm_buf, hidden_size, self.logits_buf, self.model.config.vocab_size, hidden_dim);
            }
            self.endProfilePhase(.final_lm_head, final_lm_head_phase);

            const use_gpu_argmax = have_gpu_argmax;
            const final_argmax_phase = self.beginProfilePhase();
            if (use_gpu_argmax) {
                self.decode_cmd.computeBarrier();
                try self.argmax.record(
                    &self.decode_cmd,
                    self.argmax_descriptor_set.?,
                    self.model.config.vocab_size,
                    self.argmax_phase0_workgroups,
                );
            }
            self.endProfilePhase(.final_argmax, final_argmax_phase);

            // Read back the 4-byte token id result every token, and full logits only when debugging
            // or when GPU argmax is unavailable and we must fall back to CPU greedy sampling.
            if (self.profile_enabled and !use_gpu_argmax) {
                self.profile_token_counters.cpu_argmax_fallbacks += 1;
            }
            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
            };
            const final_copy_phase = self.beginProfilePhase();
            vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, null, 0, null);
            if (use_gpu_argmax) {
                const token_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = @sizeOf(u32) };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.argmax_result_buf.handle, self.argmax_result_staging.handle, 1, &token_region);
            }
            if (need_logits_readback) {
                const logits_copy_size = @as(vk.c.VkDeviceSize, self.model.config.vocab_size) * @sizeOf(f32);
                const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = logits_copy_size };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.logits_buf.handle, self.logits_staging.handle, 1, &region);
            }
            if (self.validation_diagnostics_enabled and hidden_dim <= 8192) {
                const hidden_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = hidden_size };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.hidden_buf.handle, self.embed_staging.handle, 1, &hidden_region);
            }
            self.endProfilePhase(.final_copy, final_copy_phase);
            self.endProfilePhase(.final_tail, final_tail_phase);
        }
        if (!partial_hidden_out_written_by_stop) {
            if (self.partial_decode_hidden_out) |hidden_out| {
                self.decode_cmd.computeToTransferBarrier();
                const region = vk.c.VkBufferCopy{
                    .srcOffset = 0,
                    .dstOffset = self.partial_decode_hidden_out_offset,
                    .size = hidden_size,
                };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.hidden_buf.handle, hidden_out, 1, &region);
            }
        }
        _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        try self.decode_cmd.end();
        var prefill_record_elapsed_ns: u64 = 0;
        if (track_decode_timing) {
            const cpu_record_end = nanoTimestamp();
            const elapsed: u64 = @intCast(cpu_record_end - cpu_record_start);
            if (self.profile_enabled) self.profile_token_counters.cpu_record_ns += elapsed;
            prefill_record_elapsed_ns = elapsed;
        }
        const submit_wait_start = if (track_decode_timing) nanoTimestamp() else 0;
        if (self.prefill_pipeline_mode) {
            // Pipelined prefill: fire-and-forget. prefillBatch() waits for the
            // corresponding fence before the next reuse of this slot.
            try self.decode_cmd.submit(self.instance.compute_queue);
        } else {
            try self.decode_cmd.submitAndWait(self.instance.compute_queue);
        }
        var prefill_submit_wait_elapsed_ns: u64 = 0;
        if (track_decode_timing) {
            const submit_wait_end = nanoTimestamp();
            const elapsed: u64 = @intCast(submit_wait_end - submit_wait_start);
            if (self.profile_enabled) self.profile_token_counters.submit_wait_ns += elapsed;
            prefill_submit_wait_elapsed_ns = elapsed;
        }
        if (self.prefill_active) {
            self.prefill_cpu_embed_ns += prefill_embed_elapsed_ns;
            self.prefill_cpu_record_ns += prefill_record_elapsed_ns;
            self.prefill_submit_wait_ns += prefill_submit_wait_elapsed_ns;
            self.prefill_token_samples += 1;
        }

        if (self.validation_diagnostics_enabled and collect_output and hidden_dim <= 8192 and need_logits_readback) {
            const hidden_ptr: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
            const logits_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
            const vocab_size = self.model.config.vocab_size;
            const gpu_logits = logits_ptr[0..vocab_size];
            const mmap = self.model.mmap_data orelse return error.NoMmapData;
            const final_norm_tensor = self.tensor_map.get("output_norm.weight") orelse return error.TensorNotFound;
            const lm_tensor = self.tensor_map.get("output.weight") orelse
                self.tensor_map.get("token_embd.weight") orelse return error.TensorNotFound;
            const norm_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + final_norm_tensor.info.offset);
            const lm_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + lm_tensor.info.offset);

            var cpu_norm_w: [8192]f32 = undefined;
            dequantRow(mmap[norm_off..], 0, hidden_dim, final_norm_tensor.info.type_, cpu_norm_w[0..hidden_dim]);
            var cpu_normed: [8192]f32 = undefined;
            cpuRmsNormMul(hidden_ptr, cpu_norm_w[0..hidden_dim], cpu_normed[0..hidden_dim].ptr, hidden_dim, 1, rms_norm_eps);

            var top_ids = [_]u32{0} ** 5;
            var top_vals = [_]f32{-std.math.inf(f32)} ** 5;
            for (gpu_logits, 0..) |val, i| {
                var insert_at: usize = 5;
                for (0..5) |slot| {
                    if (val > top_vals[slot]) {
                        insert_at = slot;
                        break;
                    }
                }
                if (insert_at == 5) continue;
                var j: usize = 4;
                while (j > insert_at) : (j -= 1) {
                    top_ids[j] = top_ids[j - 1];
                    top_vals[j] = top_vals[j - 1];
                }
                top_ids[insert_at] = @intCast(i);
                top_vals[insert_at] = val;
            }

            var cpu_row_buf: [8192]f32 = undefined;
            var cpu_top_vals = [_]f32{0} ** 5;
            var tail_max_diff: f32 = 0;
            var tail_max_slot: usize = 0;
            for (0..5) |slot| {
                dequantRow(mmap[lm_off..], top_ids[slot], hidden_dim, lm_tensor.info.type_, cpu_row_buf[0..hidden_dim]);
                var dot: f64 = 0;
                for (0..hidden_dim) |i| dot += @as(f64, cpu_row_buf[i]) * @as(f64, cpu_normed[i]);
                cpu_top_vals[slot] = @floatCast(dot);
                const diff = @abs(top_vals[slot] - cpu_top_vals[slot]);
                if (diff > tail_max_diff) {
                    tail_max_diff = diff;
                    tail_max_slot = slot;
                }
            }
            log.info("TAIL_LOGIT_CHECK pos={d}: max_diff={d:.6} id={d} gpu=[{d:.6},{d:.6},{d:.6},{d:.6},{d:.6}] cpu=[{d:.6},{d:.6},{d:.6},{d:.6},{d:.6}]", .{
                state.position,
                tail_max_diff,
                top_ids[tail_max_slot],
                top_vals[0],
                top_vals[1],
                top_vals[2],
                top_vals[3],
                top_vals[4],
                cpu_top_vals[0],
                cpu_top_vals[1],
                cpu_top_vals[2],
                cpu_top_vals[3],
                cpu_top_vals[4],
            });
        }
        self.recordProfilingSample();

        if (self.partial_decode_advance_position) {
            state.position += 1;
        }
    }

    // -----------------------------------------------------------------------
    // DMMV dispatch helpers
    // -----------------------------------------------------------------------

    /// Dispatch a DMMV: weight × input_buf → output_buf.
    fn dispatchDmmv(
        self: *InferenceEngine,
        tensor: *const LoadedTensor,
        /// GPU buffer for input buf.
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        /// GPU buffer for output buf.
        output_buf: Buffer,
        M: u32,
        K: u32,
    ) !void {
        return self.dispatchDmmvInner(tensor, input_buf, input_size, output_buf, M, K, 0, 0, 0, 0);
    }

    /// Fused residual-add + RMS norm (Vulkan side).
    /// hidden[i] += scale * residual[i]; norm_out[i] = weights[i] * hidden[i] * rsqrt(...)
    /// One dispatch per N tokens replaces scale_acc → barrier → rms_norm_mul,
    /// eliminating one barrier per occurrence in prefillBatched (2 × n_layers
    /// barriers saved for a 36-layer LLaMA-style network).
    /// Fused rmsnorm(src) + hidden accumulate.
    ///   rms_inv = rsqrt(mean(src^2) + eps)
    ///   hidden[i] += weights[i] * src[i] * rms_inv
    ///
    /// Used by Gemma's post_ffw_norm tail. Replaces a separate rms_norm_mul
    /// (in place on scratch_down) + scale_accumulate (hidden += scratch_down)
    /// pair with a single dispatch.
    fn dispatchRmsNormAdd(
        self: *InferenceEngine,
        hidden: vk.c.VkBuffer,
        hidden_size: vk.c.VkDeviceSize,
        src: vk.c.VkBuffer,
        src_size: vk.c.VkDeviceSize,
        weights: vk.c.VkBuffer,
        weights_size: vk.c.VkDeviceSize,
        hidden_dim: u32,
        n_tokens: u32,
        eps: f32,
    ) !void {
        const pip = &(self.elementwise.pipeline_rms_norm_add orelse return error.ShaderNotLoaded);
        const push = RmsNormAddPush{ .n = hidden_dim, .eps = eps };
        if (pip.uses_push_descriptors) {
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                hidden,
                hidden_size,
                src,
                src_size,
                weights,
                weights_size,
                n_tokens,
                1,
                1,
            );
            return;
        }
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds, hidden, hidden_size, src, src_size, weights, weights_size);
        self.decode_cmd.dispatchWithPush(pip, ds, std.mem.asBytes(&push), n_tokens, 1, 1);
    }

    fn dispatchResidualRmsNorm(
        self: *InferenceEngine,
        hidden: vk.c.VkBuffer,
        hidden_size: vk.c.VkDeviceSize,
        residual: vk.c.VkBuffer,
        residual_size: vk.c.VkDeviceSize,
        norm_out: vk.c.VkBuffer,
        norm_out_size: vk.c.VkDeviceSize,
        weights: vk.c.VkBuffer,
        weights_size: vk.c.VkDeviceSize,
        hidden_dim: u32,
        n_tokens: u32,
        eps: f32,
        scale: f32,
    ) !void {
        const pip = &(self.elementwise.pipeline_residual_rms_norm orelse return error.ShaderNotLoaded);
        const push = ResidualRmsNormPush{
            .n = hidden_dim,
            .eps_bits = @bitCast(eps),
            .scale_bits = @bitCast(scale),
        };
        if (pip.uses_push_descriptors) {
            self.pushDispatch4(
                pip,
                std.mem.asBytes(&push),
                hidden,
                hidden_size,
                residual,
                residual_size,
                norm_out,
                norm_out_size,
                weights,
                weights_size,
                n_tokens,
                1,
                1,
            );
            return;
        }
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet4(ds, hidden, hidden_size, residual, residual_size, norm_out, norm_out_size, weights, weights_size);
        self.decode_cmd.dispatchWithPush(pip, ds, std.mem.asBytes(&push), n_tokens, 1, 1);
    }

    /// Batched KV-cache write — stores N tokens' K/V into the paged cache in
    /// one dispatch. Replaces the per-token vkCmdCopyBuffer loop that prefill-
    /// Batched emitted in its first cut, and with it the
    /// transferToComputeBarrier that sat between transfer and the next layer.
    /// Grid: ((kv_dim + 63) / 64, n_tokens, 1).
    fn dispatchKvCacheWriteBatched(
        self: *InferenceEngine,
        k_src: vk.c.VkBuffer,
        k_src_size: vk.c.VkDeviceSize,
        k_dst: vk.c.VkBuffer,
        k_dst_size: vk.c.VkDeviceSize,
        v_src: vk.c.VkBuffer,
        v_src_size: vk.c.VkDeviceSize,
        v_dst: vk.c.VkBuffer,
        v_dst_size: vk.c.VkDeviceSize,
        page_table: vk.c.VkBuffer,
        page_table_size: vk.c.VkDeviceSize,
        kv_dim: u32,
        n_tokens: u32,
        page_size: u32,
        base_token: u32,
    ) !void {
        const pip = &(self.elementwise.pipeline_kv_cache_write_batched orelse return error.ShaderNotLoaded);
        const push = KvCacheWriteBatchedPush{
            .kv_dim = kv_dim,
            .n_tokens = n_tokens,
            .page_size = page_size,
            .base_token = base_token,
        };
        const wg_x = (kv_dim + 63) / 64;
        if (pip.uses_push_descriptors) {
            self.pushDispatch5(
                pip,
                std.mem.asBytes(&push),
                k_src,
                k_src_size,
                k_dst,
                k_dst_size,
                v_src,
                v_src_size,
                v_dst,
                v_dst_size,
                page_table,
                page_table_size,
                wg_x,
                n_tokens,
                1,
            );
            return;
        }
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet5(ds, k_src, k_src_size, k_dst, k_dst_size, v_src, v_src_size, v_dst, v_dst_size, page_table, page_table_size);
        self.decode_cmd.dispatchWithPush(pip, ds, std.mem.asBytes(&push), wg_x, n_tokens, 1);
    }

    /// Batched RoPE wrapper — rotates `n_tokens × n_heads × stride` contiguous
    /// f32s in one dispatch. Grid is (n_heads, n_tokens, 1). Positions are
    /// [position_base, position_base + n_tokens). Used by prefillBatched so Q
    /// and K for the whole prompt rotate in a single kernel launch each.
    fn dispatchRopeBatched(
        self: *InferenceEngine,
        in_buf: vk.c.VkBuffer,
        in_size: vk.c.VkDeviceSize,
        out_buf: vk.c.VkBuffer,
        out_size: vk.c.VkDeviceSize,
        freq_buf: vk.c.VkBuffer,
        freq_size: vk.c.VkDeviceSize,
        stride: u32,
        rope_dim: u32,
        n_heads: u32,
        position_base: u32,
        n_tokens: u32,
        freq_base: f32,
        attn_scale: f32,
    ) !void {
        const pip = &(self.elementwise.pipeline_rope_batched orelse return error.ShaderNotLoaded);
        const push = RopeBatchedPush{
            .stride = stride,
            .rope_dim = rope_dim,
            .n_heads = n_heads,
            .position_base = position_base,
            .freq_base_bits = @bitCast(freq_base),
            .attn_scale_bits = @bitCast(attn_scale),
        };
        if (pip.uses_push_descriptors) {
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                in_buf,
                in_size,
                out_buf,
                out_size,
                freq_buf,
                freq_size,
                n_heads,
                n_tokens,
                1,
            );
            return;
        }
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds, in_buf, in_size, out_buf, out_size, freq_buf, freq_size);
        try self.elementwise.recordRoPEBatched(&self.decode_cmd, ds, stride, rope_dim, n_heads, position_base, n_tokens, freq_base, attn_scale);
    }

    /// Batched causal flash attention wrapper — processes N queries against
    /// the paged KV cache in one dispatch. `seq_start` is the position of
    /// query 0; each query q attends to KV positions [0, seq_start + q].
    /// `sink_offset` is `layer_idx * n_heads` into the per-layer sinks
    /// buffer (NaN-gated for layers without sinks).
    fn dispatchFlashAttnBatched(
        self: *InferenceEngine,
        q_buf: vk.c.VkBuffer,
        q_size: vk.c.VkDeviceSize,
        k_cache: vk.c.VkBuffer,
        k_cache_size: vk.c.VkDeviceSize,
        v_cache: vk.c.VkBuffer,
        v_cache_size: vk.c.VkDeviceSize,
        page_table: vk.c.VkBuffer,
        page_table_size: vk.c.VkDeviceSize,
        out_buf: vk.c.VkBuffer,
        out_size: vk.c.VkDeviceSize,
        sinks: vk.c.VkBuffer,
        sinks_size: vk.c.VkDeviceSize,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        seq_start: u32,
        n_queries: u32,
        page_size: u32,
        attn_scale: f32,
        sink_offset: u32,
    ) !void {
        const pip = &(self.attention.pipeline_batched orelse return error.ShaderNotLoaded);
        const push = FlashAttnBatchedPush{
            .head_dim = head_dim,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .seq_start = seq_start,
            .n_queries = n_queries,
            .page_size = page_size,
            .attn_scale_bits = if (attn_scale != 0) @as(u32, @bitCast(attn_scale)) else 0,
            .sink_offset = sink_offset,
        };
        if (pip.uses_push_descriptors) {
            self.pushDispatch6(
                pip,
                std.mem.asBytes(&push),
                q_buf,
                q_size,
                k_cache,
                k_cache_size,
                v_cache,
                v_cache_size,
                page_table,
                page_table_size,
                out_buf,
                out_size,
                sinks,
                sinks_size,
                n_heads,
                n_queries,
                1,
            );
            return;
        }
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet6(ds, q_buf, q_size, k_cache, k_cache_size, v_cache, v_cache_size, page_table, page_table_size, out_buf, out_size, sinks, sinks_size);
        try self.attention.recordFlashAttnBatched(&self.decode_cmd, ds, head_dim, n_heads, n_kv_heads, seq_start, n_queries, page_size, attn_scale, sink_offset);
    }

    /// Batched projection: weight × [N_tokens columns of x] → [N_tokens columns of y].
    /// Weight is read once per chunk of up to MAX_COLS tokens instead of once per
    /// token — the core bandwidth win for the prefillBatched path. The underlying
    /// dmmv_q4k_batch shader caps num_cols at 32, so prompts > 32 tokens are split
    /// into ceil(N/32) dispatches advancing x_offset and y_offset in lock-step.
    /// Column layout: x is [N × K] contiguous, y is [N × M] contiguous, both f32.
    fn dispatchProjectionBatched(
        self: *InferenceEngine,
        tensor: *const LoadedTensor,
        x_buf: Buffer,
        y_buf: Buffer,
        M: u32,
        K: u32,
        n_tokens: u32,
    ) !void {
        // Keep these in sync with the GLSL arrays:
        // - dmmv_q4k_batch.comp:           MAX_COLS = 32
        // - dmmv_q6k_batch.comp:           MAX_COLS = 24
        // - dmmv_q{4,6}k_batch_kpar.comp:  MAX_COLS = 40
        // - dmmv_q5k.comp batched mode:    MAX_COLS = 40
        //
        // Intel currently uses the serial batch shaders, not the wave64 kpar
        // variants. Sending 40 columns to the serial shader overruns its
        // 32-element register array and can end in FenceWaitFailed.
        const SERIAL_MAX_COLS: u32 = 32;
        const SERIAL_Q6_MAX_COLS: u32 = 24;
        const KPAR_MAX_COLS: u32 = 40;
        const f32_bytes: u32 = @sizeOf(f32);
        var chunk_start: u32 = 0;
        const cfg = self.model.config;
        const prefer_serial_qwen36_dense_down = self.isQwen36DenseHybrid27B() and
            tensor.info.type_ == .q6_k and
            M == cfg.hidden_dim and
            K == cfg.intermediate_dim and
            n_tokens >= 16;
        const qwen36_ssm_qkv_shape = self.isQwen36DenseHybrid27B() and
            tensor.info.type_ == .q6_k and
            M == cfg.ssm_d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state and
            K == cfg.hidden_dim and
            n_tokens >= 16;
        const kpar_pipeline: ?*const Pipeline = blk: {
            if (!self.use_q4k_batch_kpar) break :blk null;
            if (prefer_serial_qwen36_dense_down) break :blk null;
            switch (tensor.info.type_) {
                .q4_k => break :blk if (self.dmmv.pipeline_q4k_batch_kpar) |*p| p else null,
                .q5_k => break :blk if (self.dmmv.pipeline_q5k) |*p| p else null,
                .q6_k => break :blk if (self.dmmv.pipeline_q6k_batch_kpar) |*p| p else null,
                else => break :blk null,
            }
        };
        const serial_max_cols: u32 = switch (tensor.info.type_) {
            .q6_k => SERIAL_Q6_MAX_COLS,
            else => SERIAL_MAX_COLS,
        };
        // Pre-Q6 split, this was: if (kpar_pipeline != null) KPAR_MAX_COLS else SERIAL_MAX_COLS.
        const max_cols: u32 = if (kpar_pipeline != null) KPAR_MAX_COLS else serial_max_cols;

        // Fast path (effort-6 Step 5): route Q4_K projections with N >= 16 through
        // the tiled mul_mm_q4k pipeline. The kpar shader reads the M × K weight
        // tensor once per MAX_COLS=40 chunk (3× for an N=105 prefill), whereas
        // the tiled GEMM keeps each 32-row weight tile resident in shared memory
        // and walks K once per N-tile. Layout-compatible with the kpar X/Y
        // buffers: column-major X[col][k] at offset col*K floats, column-major
        // Y[col][row] at offset col*M floats. Falls through to the kpar/serial
        // path for Q6_K tensors, for N < 16, when the pipeline failed to load,
        // or when push descriptors aren't available. K is guaranteed to be a
        // multiple of 256 for Q4_K (super-block size) — defensive check anyway.
        if (self.use_mul_mm_proj and
            tensor.info.type_ == .q4_k and
            n_tokens >= 16 and
            (K & 255) == 0 and
            self.dmmv.pipeline_mul_mm_q4k != null)
        {
            try self.dmmv.recordMulMmQ4K(
                &self.decode_cmd,
                self.instance.push_descriptor_fn,
                tensor.gpu_buffer.handle,
                tensor.gpu_buffer.size,
                x_buf.handle,
                x_buf.size,
                y_buf.handle,
                y_buf.size,
                M,
                n_tokens,
                K,
                K, // stride_b: per-col floats in B (one column = K elements)
                M, // stride_d: per-col floats in D (one column = M elements)
                0, // a_offset (bytes)
                0, // b_offset (floats)
                0, // d_offset (floats)
            );
            return;
        }
        if (self.use_qwen36_q6_prefill_mul_mm and
            tensor.info.type_ == .q6_k and
            self.isQwen36DenseHybrid27B() and
            (prefer_serial_qwen36_dense_down or qwen36_ssm_qkv_shape) and
            n_tokens >= 16 and
            (K & 255) == 0 and
            self.dmmv.pipeline_mul_mm_q6k != null)
        {
            const full_cols = n_tokens & ~@as(u32, 31);
            if (full_cols > 0 and (M & 31) == 0 and self.dmmv.pipeline_mul_mm_q6k_full != null) {
                try self.dmmv.recordMulMmQ6KFull(
                    &self.decode_cmd,
                    self.instance.push_descriptor_fn,
                    tensor.gpu_buffer.handle,
                    tensor.gpu_buffer.size,
                    x_buf.handle,
                    x_buf.size,
                    y_buf.handle,
                    y_buf.size,
                    M,
                    full_cols,
                    K,
                    K,
                    M,
                    0,
                    0,
                    0,
                );
                if (full_cols < n_tokens) {
                    try self.dmmv.recordMulMmQ6K(
                        &self.decode_cmd,
                        self.instance.push_descriptor_fn,
                        tensor.gpu_buffer.handle,
                        tensor.gpu_buffer.size,
                        x_buf.handle,
                        x_buf.size,
                        y_buf.handle,
                        y_buf.size,
                        M,
                        n_tokens - full_cols,
                        K,
                        K,
                        M,
                        0,
                        full_cols * K,
                        full_cols * M,
                    );
                }
            } else {
                try self.dmmv.recordMulMmQ6K(
                    &self.decode_cmd,
                    self.instance.push_descriptor_fn,
                    tensor.gpu_buffer.handle,
                    tensor.gpu_buffer.size,
                    x_buf.handle,
                    x_buf.size,
                    y_buf.handle,
                    y_buf.size,
                    M,
                    n_tokens,
                    K,
                    K,
                    M,
                    0,
                    0,
                    0,
                );
            }
            return;
        }
        while (chunk_start < n_tokens) {
            const chunk: u32 = @min(max_cols, n_tokens - chunk_start);
            const x_offset: u32 = chunk_start * K * f32_bytes;
            const y_offset: u32 = chunk_start * M * f32_bytes;
            if (kpar_pipeline) |pip| {
                // One workgroup per output row — 64 threads cooperate on K.
                const push = BatchDmmvPushConstants{
                    .M = M,
                    .K = K,
                    .a_offset = 0,
                    .x_offset = x_offset,
                    .y_offset = y_offset,
                    .num_cols = chunk,
                };
                if (tensor.info.type_ == .q5_k) {
                    const q5_push = DmmvPushConstants{
                        .M = M,
                        .K = K,
                        .a_offset = 0,
                        .x_offset = x_offset,
                        .y_offset = y_offset,
                        .acc_mode = chunk,
                    };
                    self.pushDispatch3(
                        pip,
                        std.mem.asBytes(&q5_push),
                        tensor.gpu_buffer.handle,
                        tensor.gpu_buffer.size,
                        x_buf.handle,
                        x_buf.size,
                        y_buf.handle,
                        y_buf.size,
                        M,
                        1,
                        1,
                    );
                } else {
                    self.pushDispatch3(
                        pip,
                        std.mem.asBytes(&push),
                        tensor.gpu_buffer.handle,
                        tensor.gpu_buffer.size,
                        x_buf.handle,
                        x_buf.size,
                        y_buf.handle,
                        y_buf.size,
                        M,
                        1,
                        1,
                    );
                }
            } else {
                try self.dmmv.recordBatchDispatchPush(
                    &self.decode_cmd,
                    tensor.info.type_,
                    self.instance.push_descriptor_fn,
                    tensor.gpu_buffer.handle,
                    tensor.gpu_buffer.size,
                    x_buf.handle,
                    x_buf.size,
                    y_buf.handle,
                    y_buf.size,
                    M,
                    K,
                    0,
                    x_offset,
                    y_offset,
                    chunk,
                );
            }
            chunk_start += chunk;
        }
    }

    fn dispatchF32DualBatched(
        self: *InferenceEngine,
        alpha_tensor: *const LoadedTensor,
        beta_tensor: *const LoadedTensor,
        x_buf: Buffer,
        alpha_out: Buffer,
        beta_out: Buffer,
        m: u32,
        k: u32,
        n_tokens: u32,
    ) !void {
        const pip = &(self.elementwise.pipeline_dmmv_f32_dual_batch orelse return error.ShaderNotLoaded);
        if (!pip.uses_push_descriptors) return error.ShaderNotLoaded;
        if (alpha_tensor.info.type_ != .f32 or beta_tensor.info.type_ != .f32) return error.UnsupportedQuantType;
        if ((k & 3) != 0 or m == 0 or n_tokens == 0) return error.InvalidArgument;
        const push = F32DualBatchPush{
            .M = m,
            .K = k,
            .stride_x = k,
            .stride_y = m,
        };
        self.pushDispatch5(
            pip,
            std.mem.asBytes(&push),
            alpha_tensor.gpu_buffer.handle,
            alpha_tensor.gpu_buffer.size,
            beta_tensor.gpu_buffer.handle,
            beta_tensor.gpu_buffer.size,
            x_buf.handle,
            x_buf.size,
            alpha_out.handle,
            alpha_out.size,
            beta_out.handle,
            beta_out.size,
            m,
            n_tokens,
            1,
        );
    }

    fn dispatchSsmConv1dBatchedInPlace(
        self: *InferenceEngine,
        qkv_buf: Buffer,
        qkv_size: vk.c.VkDeviceSize,
        conv_tensor: *const LoadedTensor,
        state_buf: Buffer,
        conv_channels: u32,
        d_conv: u32,
        state_offset: u32,
        n_tokens: u32,
    ) !void {
        const pip = &(self.elementwise.pipeline_ssm_conv1d_batched orelse return error.ShaderNotLoaded);
        if (!pip.uses_push_descriptors) return error.ShaderNotLoaded;
        const push = SsmConv1dBatchedPush{
            .conv_channels = conv_channels,
            .d_conv = d_conv,
            .kernel_is_f16 = if (conv_tensor.info.type_ == .f16) 1 else 0,
            .state_offset = state_offset,
            .n_tokens = n_tokens,
        };
        self.pushDispatch3(
            pip,
            std.mem.asBytes(&push),
            qkv_buf.handle,
            qkv_size,
            conv_tensor.gpu_buffer.handle,
            conv_tensor.gpu_buffer.size,
            state_buf.handle,
            state_buf.size,
            (conv_channels + 63) / 64,
            1,
            1,
        );
    }

    fn validateDensePrefillFfnChunk(self: *InferenceEngine, n_tokens: u32) !void {
        if (!self.use_qwen36_dense_prefill_validate or n_tokens == 0) return;
        const cfg = self.model.config;
        if (self.dense_prefill_validate_layer >= cfg.n_layers) return;

        const norm_ref = self.dense_prefill_validate_norm_ref orelse return;
        const pre_hidden_ref = self.dense_prefill_validate_pre_hidden_ref orelse return;
        const post_hidden_ref = self.dense_prefill_validate_post_hidden_ref orelse return;
        const gate_ref = self.dense_prefill_validate_gate_ref orelse return;
        const up_ref = self.dense_prefill_validate_up_ref orelse return;
        const swiglu_ref = self.dense_prefill_validate_swiglu_ref orelse return;
        const down_ref = self.dense_prefill_validate_down_ref orelse return;
        const staging = self.dense_prefill_validate_staging orelse return;
        const lt = self.layer_tensors[self.dense_prefill_validate_layer];
        const gate_t = lt.ffn_gate orelse return error.TensorNotFound;
        const up_t = lt.ffn_up orelse return error.TensorNotFound;
        const down_t = lt.ffn_down orelse return error.TensorNotFound;
        const hidden_dim = cfg.hidden_dim;
        const inter_dim: u32 = if (cfg.intermediate_dim > 0) cfg.intermediate_dim else hidden_dim * 4;
        const hidden_capture_bytes: vk.c.VkDeviceSize =
            @as(vk.c.VkDeviceSize, n_tokens) *
            @as(vk.c.VkDeviceSize, hidden_dim) *
            @sizeOf(f32);
        const inter_capture_bytes: vk.c.VkDeviceSize =
            @as(vk.c.VkDeviceSize, n_tokens) *
            @as(vk.c.VkDeviceSize, inter_dim) *
            @sizeOf(f32);
        const hidden_elems: usize = @intCast(@as(vk.c.VkDeviceSize, n_tokens) * @as(vk.c.VkDeviceSize, hidden_dim));
        const inter_elems: usize = @intCast(@as(vk.c.VkDeviceSize, n_tokens) * @as(vk.c.VkDeviceSize, inter_dim));
        const staging_needed = hidden_capture_bytes * 4 + inter_capture_bytes * 6;
        if (staging_needed > staging.size) return error.BufferTooSmall;

        try self.ensureBatchedScratchCapacity(n_tokens);
        const scratch_gate = self.batched_scratch_gate.?;
        const scratch_up = self.batched_scratch_up.?;
        const scratch_swiglu = self.batched_scratch_swiglu.?;
        const scratch_down = self.batched_scratch_down.?;

        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.beginOneTime();
        try self.dispatchProjectionBatched(gate_t, norm_ref, scratch_gate, inter_dim, hidden_dim, n_tokens);
        try self.dispatchProjectionBatched(up_t, norm_ref, scratch_up, inter_dim, hidden_dim, n_tokens);
        self.decode_cmd.computeBarrier();
        try self.dispatchFfnActivation(
            scratch_gate.handle,
            scratch_gate.size,
            scratch_up.handle,
            scratch_up.size,
            scratch_swiglu.handle,
            scratch_swiglu.size,
            n_tokens * inter_dim,
        );
        self.decode_cmd.computeBarrier();
        try self.dispatchProjectionBatched(down_t, scratch_swiglu, scratch_down, hidden_dim, inter_dim, n_tokens);
        self.decode_cmd.computeToTransferBarrier();

        const pre_off: vk.c.VkDeviceSize = 0;
        const post_off: vk.c.VkDeviceSize = pre_off + hidden_capture_bytes;
        const down_ref_off: vk.c.VkDeviceSize = post_off + hidden_capture_bytes;
        const down_batch_off: vk.c.VkDeviceSize = down_ref_off + hidden_capture_bytes;
        const gate_ref_off: vk.c.VkDeviceSize = down_batch_off + hidden_capture_bytes;
        const up_ref_off: vk.c.VkDeviceSize = gate_ref_off + inter_capture_bytes;
        const swiglu_ref_off: vk.c.VkDeviceSize = up_ref_off + inter_capture_bytes;
        const gate_batch_off: vk.c.VkDeviceSize = swiglu_ref_off + inter_capture_bytes;
        const up_batch_off: vk.c.VkDeviceSize = gate_batch_off + inter_capture_bytes;
        const swiglu_batch_off: vk.c.VkDeviceSize = up_batch_off + inter_capture_bytes;

        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, pre_hidden_ref.handle, staging.handle, 1, &vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = pre_off, .size = hidden_capture_bytes });
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, post_hidden_ref.handle, staging.handle, 1, &vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = post_off, .size = hidden_capture_bytes });
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, down_ref.handle, staging.handle, 1, &vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = down_ref_off, .size = hidden_capture_bytes });
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, scratch_down.handle, staging.handle, 1, &vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = down_batch_off, .size = hidden_capture_bytes });
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, gate_ref.handle, staging.handle, 1, &vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = gate_ref_off, .size = inter_capture_bytes });
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, up_ref.handle, staging.handle, 1, &vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = up_ref_off, .size = inter_capture_bytes });
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, swiglu_ref.handle, staging.handle, 1, &vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = swiglu_ref_off, .size = inter_capture_bytes });
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, scratch_gate.handle, staging.handle, 1, &vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = gate_batch_off, .size = inter_capture_bytes });
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, scratch_up.handle, staging.handle, 1, &vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = up_batch_off, .size = inter_capture_bytes });
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, scratch_swiglu.handle, staging.handle, 1, &vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = swiglu_batch_off, .size = inter_capture_bytes });
        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

        const DiffStats = struct {
            max_abs: f32 = 0,
            max_idx: usize = 0,

            fn compute(ref_vals: []const f32, got_vals: []const f32) @This() {
                var out: @This() = .{};
                for (ref_vals, got_vals, 0..) |r, g, i| {
                    const diff = @abs(r - g);
                    if (diff > out.max_abs) {
                        out.max_abs = diff;
                        out.max_idx = i;
                    }
                }
                return out;
            }
        };

        const base: [*]const f32 = @ptrCast(@alignCast(staging.mapped.?));
        const pre = base[@intCast(pre_off / @sizeOf(f32))..][0..hidden_elems];
        const post = base[@intCast(post_off / @sizeOf(f32))..][0..hidden_elems];
        const down_ref_f = base[@intCast(down_ref_off / @sizeOf(f32))..][0..hidden_elems];
        const down_batch_f = base[@intCast(down_batch_off / @sizeOf(f32))..][0..hidden_elems];
        const gate_ref_f = base[@intCast(gate_ref_off / @sizeOf(f32))..][0..inter_elems];
        const up_ref_f = base[@intCast(up_ref_off / @sizeOf(f32))..][0..inter_elems];
        const swiglu_ref_f = base[@intCast(swiglu_ref_off / @sizeOf(f32))..][0..inter_elems];
        const gate_batch_f = base[@intCast(gate_batch_off / @sizeOf(f32))..][0..inter_elems];
        const up_batch_f = base[@intCast(up_batch_off / @sizeOf(f32))..][0..inter_elems];
        const swiglu_batch_f = base[@intCast(swiglu_batch_off / @sizeOf(f32))..][0..inter_elems];
        var post_diff: DiffStats = .{};
        for (0..hidden_elems) |i| {
            const candidate = pre[i] + down_batch_f[i];
            const diff = @abs(candidate - post[i]);
            if (diff > post_diff.max_abs) {
                post_diff.max_abs = diff;
                post_diff.max_idx = i;
            }
        }
        const gate_diff = DiffStats.compute(gate_ref_f, gate_batch_f);
        const up_diff = DiffStats.compute(up_ref_f, up_batch_f);
        const swiglu_diff = DiffStats.compute(swiglu_ref_f, swiglu_batch_f);
        const down_diff = DiffStats.compute(down_ref_f, down_batch_f);
        const token_idx = post_diff.max_idx / hidden_dim;
        const elem_idx = post_diff.max_idx % hidden_dim;
        const candidate_at_max = pre[post_diff.max_idx] + down_batch_f[post_diff.max_idx];
        const tol: f32 = 3e-3;
        const max_intermediate = @max(@max(gate_diff.max_abs, up_diff.max_abs), @max(swiglu_diff.max_abs, down_diff.max_abs));
        const max_abs = @max(post_diff.max_abs, max_intermediate);
        const verdict: []const u8 = if (max_abs <= tol) "PASS" else "FAIL";
        log.info("ZINC_QWEN36_27B_PREFILL_VALIDATE: dense_ffn layer={d} tokens={d} verdict={s} post_hidden={e:.6}@tok{d}/elem{d} gate={e:.6}@{d} up={e:.6}@{d} swiglu={e:.6}@{d} down={e:.6}@{d} ref_post={d:.6} batched_post={d:.6} tol={e:.3}", .{
            self.dense_prefill_validate_layer,
            n_tokens,
            verdict,
            post_diff.max_abs,
            token_idx,
            elem_idx,
            gate_diff.max_abs,
            gate_diff.max_idx,
            up_diff.max_abs,
            up_diff.max_idx,
            swiglu_diff.max_abs,
            swiglu_diff.max_idx,
            down_diff.max_abs,
            down_diff.max_idx,
            post[post_diff.max_idx],
            candidate_at_max,
            tol,
        });
    }

    fn validateSsmPrefillProjectionChunk(self: *InferenceEngine, n_tokens: u32) !void {
        if (!self.use_qwen36_ssm_prefill_validate or n_tokens == 0) return;
        const cfg = self.model.config;
        if (self.dense_prefill_validate_layer >= cfg.n_layers) return;
        if (cfg.ssm_d_inner == 0 or cfg.ssm_dt_rank == 0) return;

        const norm_ref = self.ssm_prefill_validate_norm_ref orelse return;
        const qkv_ref = self.ssm_prefill_validate_qkv_ref orelse return;
        const z_ref = self.ssm_prefill_validate_z_ref orelse return;
        const alpha_ref = self.ssm_prefill_validate_alpha_ref orelse return;
        const beta_ref = self.ssm_prefill_validate_beta_ref orelse return;
        const staging = self.ssm_prefill_validate_staging orelse return;
        const conv_ref = self.ssm_prefill_validate_conv_ref;
        const delta_ref = self.ssm_prefill_validate_delta_ref;
        const delta_replay = self.ssm_prefill_validate_delta_replay;
        const gnorm_ref = self.ssm_prefill_validate_gnorm_ref;
        const pre_hidden_ref = self.ssm_prefill_validate_pre_hidden_ref;
        const post_hidden_ref = self.ssm_prefill_validate_post_hidden_ref;
        const state_backup = self.ssm_prefill_validate_state_backup;
        const lt = self.layer_tensors[self.dense_prefill_validate_layer];
        const wqkv_t = lt.attn_qkv orelse return error.TensorNotFound;
        const z_t = lt.attn_gate orelse return error.TensorNotFound;
        const alpha_t = lt.ssm_alpha orelse return error.TensorNotFound;
        const beta_t = lt.ssm_beta orelse return error.TensorNotFound;

        const hidden_dim = cfg.hidden_dim;
        const d_inner = cfg.ssm_d_inner;
        const dt_rank = cfg.ssm_dt_rank;
        const conv_channels = d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state;
        const hidden_size: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, hidden_dim) * @sizeOf(f32);
        const hidden_capture_bytes: vk.c.VkDeviceSize =
            @as(vk.c.VkDeviceSize, n_tokens) *
            @as(vk.c.VkDeviceSize, hidden_dim) *
            @sizeOf(f32);
        const qkv_total_bytes: vk.c.VkDeviceSize =
            @as(vk.c.VkDeviceSize, n_tokens) *
            @as(vk.c.VkDeviceSize, conv_channels) *
            @sizeOf(f32);
        const z_total_bytes: vk.c.VkDeviceSize =
            @as(vk.c.VkDeviceSize, n_tokens) *
            @as(vk.c.VkDeviceSize, d_inner) *
            @sizeOf(f32);
        const z_bytes: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, d_inner) * @sizeOf(f32);
        const ab_total_bytes: vk.c.VkDeviceSize =
            @as(vk.c.VkDeviceSize, n_tokens) *
            @as(vk.c.VkDeviceSize, dt_rank) *
            @sizeOf(f32);
        const head_v_dim: u32 = d_inner / dt_rank;
        const state_bytes: vk.c.VkDeviceSize =
            @as(vk.c.VkDeviceSize, dt_rank) *
            @as(vk.c.VkDeviceSize, head_v_dim) *
            @as(vk.c.VkDeviceSize, head_v_dim) *
            @sizeOf(f32);
        const use_delta_cols8_replay = self.use_ssm_delta_cols8 and
            head_v_dim == 128 and
            cfg.ssm_d_state == 128 and
            self.elementwise.pipeline_ssm_delta_net_cols8 != null;
        const delta_replay_ready = conv_ref != null and
            delta_ref != null and
            delta_replay != null and
            state_backup != null and
            (self.elementwise.pipeline_ssm_delta_net != null or use_delta_cols8_replay) and
            self.instance.push_descriptor_fn != null and
            self.dense_prefill_validate_layer < self.gpu_ssm_states.len and
            self.gpu_ssm_states[self.dense_prefill_validate_layer].handle != null and
            self.gpu_ssm_states[self.dense_prefill_validate_layer].size >= state_bytes and
            state_backup.?.size >= state_bytes and
            conv_ref.?.size >= qkv_total_bytes and
            delta_ref.?.size >= z_total_bytes and
            delta_replay.?.size >= z_total_bytes;
        const output_capture_allocated = pre_hidden_ref != null and
            post_hidden_ref != null and
            gnorm_ref != null and
            pre_hidden_ref.?.size >= hidden_capture_bytes and
            post_hidden_ref.?.size >= hidden_capture_bytes and
            gnorm_ref.?.size >= z_total_bytes;
        const staging_needed = hidden_capture_bytes +
            2 * (qkv_total_bytes + z_total_bytes) +
            2 * ab_total_bytes +
            (if (delta_replay_ready) 2 * z_total_bytes else 0) +
            (if (output_capture_allocated) (3 * hidden_capture_bytes + 2 * z_total_bytes) else 0);
        if (staging_needed > staging.size) return error.BufferTooSmall;

        try self.ensureBatchedScratchCapacity(n_tokens);
        const scratch_qkv = self.batched_scratch_gate orelse return error.BufferTooSmall;
        const scratch_z = self.batched_scratch_up orelse return error.BufferTooSmall;
        const scratch_gnorm = self.batched_scratch_swiglu orelse return error.BufferTooSmall;
        const scratch_ssm_out = self.batched_scratch_down orelse return error.BufferTooSmall;
        if (qkv_total_bytes > scratch_qkv.size or z_total_bytes > scratch_z.size) return error.BufferTooSmall;
        const output_replay_ready = output_capture_allocated and
            delta_replay_ready and
            self.elementwise.pipeline_ssm_gated_norm != null and
            self.elementwise.pipeline_ssm_gated_norm.?.uses_push_descriptors and
            lt.ssm_out != null and
            z_total_bytes <= scratch_gnorm.size and
            hidden_capture_bytes <= scratch_ssm_out.size;

        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.beginOneTime();
        if (delta_replay_ready) {
            const state_buf = self.gpu_ssm_states[self.dense_prefill_validate_layer];
            self.decode_cmd.computeToTransferBarrier();
            const state_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = state_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, state_buf.handle, state_backup.?.handle, 1, &state_region);
            self.decode_cmd.transferToTransferBarrier();
            vk.c.vkCmdFillBuffer(self.decode_cmd.handle, state_buf.handle, 0, state_bytes, 0);
            self.decode_cmd.transferToComputeBarrier();
        } else {
            self.decode_cmd.transferToComputeBarrier();
        }
        try self.dispatchProjectionBatched(wqkv_t, norm_ref, scratch_qkv, conv_channels, hidden_dim, n_tokens);
        try self.dispatchProjectionBatched(z_t, norm_ref, scratch_z, d_inner, hidden_dim, n_tokens);
        if (delta_replay_ready) {
            const dt_bias_t = lt.ssm_dt_bias;
            const ssm_a_t = lt.ssm_a;
            const dt_bias_buf = if (dt_bias_t) |t| t.gpu_buffer.handle else self.down_buf.handle;
            const dt_bias_size = if (dt_bias_t) |t| t.gpu_buffer.size else (@as(vk.c.VkDeviceSize, dt_rank) * @sizeOf(f32));
            const ssm_a_buf = if (ssm_a_t) |t| t.gpu_buffer.handle else self.down_buf.handle;
            const ssm_a_size = if (ssm_a_t) |t| t.gpu_buffer.size else (@as(vk.c.VkDeviceSize, dt_rank) * @sizeOf(f32));
            const push = SsmDeltaNetPush{
                .d_inner = d_inner,
                .dt_rank = dt_rank,
                .head_v_dim = head_v_dim,
                .d_state = cfg.ssm_d_state,
                .n_group = cfg.ssm_n_group,
                .ssm_a_is_f16 = if (ssm_a_t) |t| (if (t.info.type_ == .f16) @as(u32, 1) else 0) else 0,
                .dt_bias_is_f16 = if (dt_bias_t) |t| (if (t.info.type_ == .f16) @as(u32, 1) else 0) else 0,
                .has_dt_bias = if (dt_bias_t != null) 1 else 0,
                .has_ssm_a = if (ssm_a_t != null) 1 else 0,
                .n_tok = n_tokens,
                .conv_stride_tok = conv_channels,
                .ab_stride_tok = dt_rank,
                .y_stride_tok = d_inner,
            };
            const pip = if (use_delta_cols8_replay)
                &(self.elementwise.pipeline_ssm_delta_net_cols8.?)
            else
                &(self.elementwise.pipeline_ssm_delta_net orelse return error.ShaderNotLoaded);
            const row_blocks = if (use_delta_cols8_replay)
                (head_v_dim + 3) / 4
            else
                head_v_dim;
            self.pushDispatch7(
                pip,
                std.mem.asBytes(&push),
                conv_ref.?.handle,
                qkv_total_bytes,
                dt_bias_buf,
                dt_bias_size,
                alpha_ref.handle,
                ab_total_bytes,
                beta_ref.handle,
                ab_total_bytes,
                ssm_a_buf,
                ssm_a_size,
                self.gpu_ssm_states[self.dense_prefill_validate_layer].handle,
                state_bytes,
                delta_replay.?.handle,
                z_total_bytes,
                dt_rank,
                row_blocks,
                1,
            );
        }
        if (output_replay_ready) {
            const norm_tensor = lt.ssm_norm;
            const norm_elems: u32 = if (norm_tensor) |t| @intCast(t.info.numElements()) else 0;
            const norm_per_head = norm_elems >= d_inner;
            const norm_buf_handle = if (norm_tensor) |t| t.gpu_buffer.handle else self.down_buf.handle;
            const norm_buf_size = if (norm_tensor) |t| t.gpu_buffer.size else ab_total_bytes;
            const gnorm_pip = &(self.elementwise.pipeline_ssm_gated_norm.?);
            const gnorm_push = SsmGatedNormPush{
                .d_inner = d_inner,
                .dt_rank = dt_rank,
                .head_v_dim = head_v_dim,
                .d_state = cfg.ssm_d_state,
                .norm_per_head = if (norm_per_head) 1 else 0,
            };
            self.decode_cmd.computeBufferBarrier(delta_replay.?.handle, z_total_bytes);
            var tok_idx_replay: u32 = 0;
            while (tok_idx_replay < n_tokens) : (tok_idx_replay += 1) {
                const z_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx_replay) * z_bytes;
                const infos = [4]vk.c.VkDescriptorBufferInfo{
                    .{ .buffer = delta_replay.?.handle, .offset = z_off, .range = z_bytes },
                    .{ .buffer = z_ref.handle, .offset = z_off, .range = z_bytes },
                    .{ .buffer = norm_buf_handle, .offset = 0, .range = norm_buf_size },
                    .{ .buffer = scratch_gnorm.handle, .offset = z_off, .range = z_bytes },
                };
                self.decode_cmd.pushDescAndDispatch(
                    gnorm_pip,
                    self.instance.push_descriptor_fn,
                    infos[0..],
                    std.mem.asBytes(&gnorm_push),
                    dt_rank,
                    1,
                    1,
                );
            }
            self.decode_cmd.computeBufferBarrier(scratch_gnorm.handle, z_total_bytes);
            const ssm_out_tensor = lt.ssm_out.?;
            tok_idx_replay = 0;
            while (tok_idx_replay < n_tokens) : (tok_idx_replay += 1) {
                const x_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx_replay) * z_bytes;
                const y_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx_replay) * hidden_size;
                try self.dispatchDmmvInner(
                    ssm_out_tensor,
                    scratch_gnorm,
                    scratch_gnorm.size,
                    scratch_ssm_out,
                    hidden_dim,
                    @intCast(d_inner),
                    0,
                    @intCast(x_off),
                    @intCast(y_off),
                    0,
                );
            }
        }
        self.decode_cmd.computeToTransferBarrier();
        if (delta_replay_ready) {
            const state_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = state_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, state_backup.?.handle, self.gpu_ssm_states[self.dense_prefill_validate_layer].handle, 1, &state_region);
        }
        const norm_ref_off: vk.c.VkDeviceSize = 0;
        const qkv_ref_off: vk.c.VkDeviceSize = norm_ref_off + hidden_capture_bytes;
        const z_ref_off: vk.c.VkDeviceSize = qkv_ref_off + qkv_total_bytes;
        const alpha_ref_off: vk.c.VkDeviceSize = z_ref_off + z_total_bytes;
        const beta_ref_off: vk.c.VkDeviceSize = alpha_ref_off + ab_total_bytes;
        const qkv_batch_off: vk.c.VkDeviceSize = beta_ref_off + ab_total_bytes;
        const z_batch_off: vk.c.VkDeviceSize = qkv_batch_off + qkv_total_bytes;
        var next_stage_off: vk.c.VkDeviceSize = z_batch_off + z_total_bytes;
        const delta_ref_off: vk.c.VkDeviceSize = next_stage_off;
        if (delta_replay_ready) next_stage_off += z_total_bytes;
        const delta_replay_off: vk.c.VkDeviceSize = next_stage_off;
        if (delta_replay_ready) next_stage_off += z_total_bytes;
        const pre_hidden_off: vk.c.VkDeviceSize = next_stage_off;
        if (output_replay_ready) next_stage_off += hidden_capture_bytes;
        const post_hidden_off: vk.c.VkDeviceSize = next_stage_off;
        if (output_replay_ready) next_stage_off += hidden_capture_bytes;
        const gnorm_ref_off: vk.c.VkDeviceSize = next_stage_off;
        if (output_replay_ready) next_stage_off += z_total_bytes;
        const gnorm_replay_off: vk.c.VkDeviceSize = next_stage_off;
        if (output_replay_ready) next_stage_off += z_total_bytes;
        const ssm_out_replay_off: vk.c.VkDeviceSize = next_stage_off;
        const copies = [_]vk.c.VkBufferCopy{
            .{ .srcOffset = 0, .dstOffset = norm_ref_off, .size = hidden_capture_bytes },
            .{ .srcOffset = 0, .dstOffset = qkv_ref_off, .size = qkv_total_bytes },
            .{ .srcOffset = 0, .dstOffset = z_ref_off, .size = z_total_bytes },
            .{ .srcOffset = 0, .dstOffset = alpha_ref_off, .size = ab_total_bytes },
            .{ .srcOffset = 0, .dstOffset = beta_ref_off, .size = ab_total_bytes },
            .{ .srcOffset = 0, .dstOffset = qkv_batch_off, .size = qkv_total_bytes },
            .{ .srcOffset = 0, .dstOffset = z_batch_off, .size = z_total_bytes },
        };
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, norm_ref.handle, staging.handle, 1, &copies[0]);
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, qkv_ref.handle, staging.handle, 1, &copies[1]);
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, z_ref.handle, staging.handle, 1, &copies[2]);
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, alpha_ref.handle, staging.handle, 1, &copies[3]);
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, beta_ref.handle, staging.handle, 1, &copies[4]);
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, scratch_qkv.handle, staging.handle, 1, &copies[5]);
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, scratch_z.handle, staging.handle, 1, &copies[6]);
        if (delta_replay_ready) {
            const r_delta_ref = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = delta_ref_off, .size = z_total_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, delta_ref.?.handle, staging.handle, 1, &r_delta_ref);
            const r_delta_replay = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = delta_replay_off, .size = z_total_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, delta_replay.?.handle, staging.handle, 1, &r_delta_replay);
        }
        if (output_replay_ready) {
            const r_pre_hidden = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = pre_hidden_off, .size = hidden_capture_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, pre_hidden_ref.?.handle, staging.handle, 1, &r_pre_hidden);
            const r_post_hidden = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = post_hidden_off, .size = hidden_capture_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, post_hidden_ref.?.handle, staging.handle, 1, &r_post_hidden);
            const r_gnorm_ref = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = gnorm_ref_off, .size = z_total_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, gnorm_ref.?.handle, staging.handle, 1, &r_gnorm_ref);
            const r_gnorm_replay = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = gnorm_replay_off, .size = z_total_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, scratch_gnorm.handle, staging.handle, 1, &r_gnorm_replay);
            const r_ssm_out_replay = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = ssm_out_replay_off, .size = hidden_capture_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, scratch_ssm_out.handle, staging.handle, 1, &r_ssm_out_replay);
        }
        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

        const DiffStats = struct {
            max_abs: f32 = 0,
            max_idx: usize = 0,

            fn compute(ref_vals: []const f32, got_vals: []const f32) @This() {
                var out: @This() = .{};
                for (ref_vals, got_vals, 0..) |r, g, i| {
                    const diff = @abs(r - g);
                    if (diff > out.max_abs) {
                        out.max_abs = diff;
                        out.max_idx = i;
                    }
                }
                return out;
            }
        };

        const base: [*]const f32 = @ptrCast(@alignCast(staging.mapped.?));
        const norm_ref_f = base[@intCast(norm_ref_off / @sizeOf(f32))..][0..@as(usize, @intCast(hidden_capture_bytes / @sizeOf(f32)))];
        const qkv_ref_f = base[@intCast(qkv_ref_off / @sizeOf(f32))..][0..@as(usize, @intCast(qkv_total_bytes / @sizeOf(f32)))];
        const z_ref_f = base[@intCast(z_ref_off / @sizeOf(f32))..][0..@as(usize, @intCast(z_total_bytes / @sizeOf(f32)))];
        const alpha_ref_f = base[@intCast(alpha_ref_off / @sizeOf(f32))..][0..@as(usize, @intCast(ab_total_bytes / @sizeOf(f32)))];
        const beta_ref_f = base[@intCast(beta_ref_off / @sizeOf(f32))..][0..@as(usize, @intCast(ab_total_bytes / @sizeOf(f32)))];
        const qkv_batch_f = base[@intCast(qkv_batch_off / @sizeOf(f32))..][0..@as(usize, @intCast(qkv_total_bytes / @sizeOf(f32)))];
        const z_batch_f = base[@intCast(z_batch_off / @sizeOf(f32))..][0..@as(usize, @intCast(z_total_bytes / @sizeOf(f32)))];
        const delta_ref_f = if (delta_replay_ready)
            base[@intCast(delta_ref_off / @sizeOf(f32))..][0..@as(usize, @intCast(z_total_bytes / @sizeOf(f32)))]
        else
            base[0..0];
        const delta_replay_f = if (delta_replay_ready)
            base[@intCast(delta_replay_off / @sizeOf(f32))..][0..@as(usize, @intCast(z_total_bytes / @sizeOf(f32)))]
        else
            base[0..0];
        const pre_hidden_f = if (output_replay_ready)
            base[@intCast(pre_hidden_off / @sizeOf(f32))..][0..@as(usize, @intCast(hidden_capture_bytes / @sizeOf(f32)))]
        else
            base[0..0];
        const post_hidden_f = if (output_replay_ready)
            base[@intCast(post_hidden_off / @sizeOf(f32))..][0..@as(usize, @intCast(hidden_capture_bytes / @sizeOf(f32)))]
        else
            base[0..0];
        const gnorm_ref_f = if (output_replay_ready)
            base[@intCast(gnorm_ref_off / @sizeOf(f32))..][0..@as(usize, @intCast(z_total_bytes / @sizeOf(f32)))]
        else
            base[0..0];
        const gnorm_replay_f = if (output_replay_ready)
            base[@intCast(gnorm_replay_off / @sizeOf(f32))..][0..@as(usize, @intCast(z_total_bytes / @sizeOf(f32)))]
        else
            base[0..0];
        const ssm_out_replay_f = if (output_replay_ready)
            base[@intCast(ssm_out_replay_off / @sizeOf(f32))..][0..@as(usize, @intCast(hidden_capture_bytes / @sizeOf(f32)))]
        else
            base[0..0];

        const mmap = self.model.mmap_data orelse return error.NoMmapData;
        const row_buf = try self.allocator.alloc(f32, hidden_dim);
        defer self.allocator.free(row_buf);

        const SampleRows = struct {
            fn includes(row: u32, rows: u32) bool {
                return row == 0 or
                    row == 1 or
                    row == rows / 2 or
                    row + 2 == rows or
                    row + 1 == rows;
            }
        };
        const UpdateDiff = struct {
            fn run(stats: *DiffStats, idx: usize, expected: f32, actual: f32) void {
                const diff = @abs(expected - actual);
                if (diff > stats.max_abs) {
                    stats.max_abs = diff;
                    stats.max_idx = idx;
                }
            }
        };

        var qkv_diff: DiffStats = .{};
        var z_diff: DiffStats = .{};
        const qkv_batch_diff = DiffStats.compute(qkv_ref_f, qkv_batch_f);
        const z_batch_diff = DiffStats.compute(z_ref_f, z_batch_f);
        const delta_diff = if (delta_replay_ready) DiffStats.compute(delta_ref_f, delta_replay_f) else DiffStats{};
        const gnorm_diff = if (output_replay_ready) DiffStats.compute(gnorm_ref_f, gnorm_replay_f) else DiffStats{};
        var ssm_out_diff: DiffStats = .{};
        var post_hidden_diff: DiffStats = .{};
        if (output_replay_ready) {
            for (0..post_hidden_f.len) |i| {
                const ref_out = post_hidden_f[i] - pre_hidden_f[i];
                const got_out = ssm_out_replay_f[i];
                const out_diff = @abs(ref_out - got_out);
                if (out_diff > ssm_out_diff.max_abs) {
                    ssm_out_diff.max_abs = out_diff;
                    ssm_out_diff.max_idx = i;
                }
                const got_post = pre_hidden_f[i] + got_out;
                const post_diff = @abs(post_hidden_f[i] - got_post);
                if (post_diff > post_hidden_diff.max_abs) {
                    post_hidden_diff.max_abs = post_diff;
                    post_hidden_diff.max_idx = i;
                }
            }
        }
        var alpha_diff: DiffStats = .{};
        var beta_diff: DiffStats = .{};
        const qkv_data: usize = @intCast(self.model.gguf_file.tensor_data_offset + wqkv_t.info.offset);
        const z_data: usize = @intCast(self.model.gguf_file.tensor_data_offset + z_t.info.offset);
        const alpha_data: usize = @intCast(self.model.gguf_file.tensor_data_offset + alpha_t.info.offset);
        const beta_data: usize = @intCast(self.model.gguf_file.tensor_data_offset + beta_t.info.offset);
        var tok_idx: u32 = 0;
        while (tok_idx < n_tokens) : (tok_idx += 1) {
            const norm_slice = norm_ref_f[@as(usize, tok_idx) * hidden_dim ..][0..hidden_dim];
            var row: u32 = 0;
            while (row < conv_channels) : (row += 1) {
                if (!SampleRows.includes(row, conv_channels)) continue;
                dequantRow(mmap[qkv_data..], row, hidden_dim, wqkv_t.info.type_, row_buf);
                var dot: f64 = 0.0;
                for (row_buf, norm_slice) |w, x| dot += @as(f64, w) * @as(f64, x);
                const idx: usize = @as(usize, tok_idx) * conv_channels + row;
                UpdateDiff.run(&qkv_diff, idx, @floatCast(dot), qkv_ref_f[idx]);
            }
            row = 0;
            while (row < d_inner) : (row += 1) {
                if (!SampleRows.includes(row, d_inner)) continue;
                dequantRow(mmap[z_data..], row, hidden_dim, z_t.info.type_, row_buf);
                var dot: f64 = 0.0;
                for (row_buf, norm_slice) |w, x| dot += @as(f64, w) * @as(f64, x);
                const idx: usize = @as(usize, tok_idx) * d_inner + row;
                UpdateDiff.run(&z_diff, idx, @floatCast(dot), z_ref_f[idx]);
            }
            row = 0;
            while (row < dt_rank) : (row += 1) {
                dequantRow(mmap[alpha_data..], row, hidden_dim, alpha_t.info.type_, row_buf);
                var dot: f64 = 0.0;
                for (row_buf, norm_slice) |w, x| dot += @as(f64, w) * @as(f64, x);
                const idx: usize = @as(usize, tok_idx) * dt_rank + row;
                UpdateDiff.run(&alpha_diff, idx, @floatCast(dot), alpha_ref_f[idx]);

                dequantRow(mmap[beta_data..], row, hidden_dim, beta_t.info.type_, row_buf);
                dot = 0.0;
                for (row_buf, norm_slice) |w, x| dot += @as(f64, w) * @as(f64, x);
                UpdateDiff.run(&beta_diff, idx, @floatCast(dot), beta_ref_f[idx]);
            }
        }
        const proj_max_abs = @max(@max(qkv_batch_diff.max_abs, z_batch_diff.max_abs), @max(@max(qkv_diff.max_abs, z_diff.max_abs), @max(alpha_diff.max_abs, beta_diff.max_abs)));
        const output_max_abs = @max(gnorm_diff.max_abs, @max(ssm_out_diff.max_abs, post_hidden_diff.max_abs));
        const proj_tol: f32 = 3e-3;
        const delta_tol: f32 = 1e-2;
        const output_tol: f32 = 1e-2;
        const delta_ok = !delta_replay_ready or delta_diff.max_abs <= delta_tol;
        const output_ok = !output_replay_ready or output_max_abs <= output_tol;
        const verdict: []const u8 = if (proj_max_abs <= proj_tol and delta_ok and output_ok) "PASS" else "FAIL";
        log.info("ZINC_QWEN36_27B_PREFILL_VALIDATE: ssm layer={d} tokens={d} verdict={s} proj_max={e:.6} delta_replay={} delta_variant={s} delta_max={e:.6}@{d} output_replay={} output_max={e:.6} max_abs batch_qkv={e:.6}@{d} batch_z={e:.6}@{d} sampled_cpu qkv={e:.6}@{d} z={e:.6}@{d} alpha={e:.6}@{d} beta={e:.6}@{d} gnorm={e:.6}@{d} ssm_out={e:.6}@{d} post_hidden={e:.6}@{d} proj_tol={e:.3} delta_tol={e:.3} output_tol={e:.3}", .{
            self.dense_prefill_validate_layer,
            n_tokens,
            verdict,
            proj_max_abs,
            delta_replay_ready,
            if (use_delta_cols8_replay) "cols8" else "generic",
            delta_diff.max_abs,
            delta_diff.max_idx,
            output_replay_ready,
            output_max_abs,
            qkv_batch_diff.max_abs,
            qkv_batch_diff.max_idx,
            z_batch_diff.max_abs,
            z_batch_diff.max_idx,
            qkv_diff.max_abs,
            qkv_diff.max_idx,
            z_diff.max_abs,
            z_diff.max_idx,
            alpha_diff.max_abs,
            alpha_diff.max_idx,
            beta_diff.max_abs,
            beta_diff.max_idx,
            gnorm_diff.max_abs,
            gnorm_diff.max_idx,
            ssm_out_diff.max_abs,
            ssm_out_diff.max_idx,
            post_hidden_diff.max_abs,
            post_hidden_diff.max_idx,
            proj_tol,
            delta_tol,
            output_tol,
        });
    }

    /// Dispatch a DMMV with accumulation: output_buf += weight × input_buf.
    fn dispatchDmmvAcc(
        self: *InferenceEngine,
        tensor: *const LoadedTensor,
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        output_buf: Buffer,
        M: u32,
        K: u32,
    ) !void {
        return self.dispatchDmmvInner(tensor, input_buf, input_size, output_buf, M, K, 0, 0, 0, 1);
    }

    /// Dispatch the dense fused gate+up+SwiGLU shader. Replaces the
    /// (gate DMMV → up DMMV → swiglu) trio with a single dispatch that
    /// writes silu(W_gate·x) * (W_up·x) directly into swiglu_buf.
    /// Push-descriptor only (we always run with push descriptors on
    /// RDNA4); the per-call gate in the dense FFN site falls back to
    /// the unfused trio when push descriptors aren't available.
    fn dispatchDmmvFusedGateUpSwiglu(
        self: *InferenceEngine,
        gate_tensor: *const LoadedTensor,
        up_tensor: *const LoadedTensor,
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        swiglu_buf: Buffer,
        M: u32,
        K: u32,
    ) !void {
        const use_row1 = self.use_qwen36_dense_fused_row1 and
            self.isQwen36DenseHybrid27B() and
            M == 17408 and
            K == 5120 and
            self.dmmv.pipeline_q4k_fused_gate_up_swiglu_row1 != null;
        const pip = if (use_row1)
            if (self.dmmv.pipeline_q4k_fused_gate_up_swiglu_row1) |*p| p else return error.ShaderNotLoaded
        else
            &(self.dmmv.pipeline_q4k_fused_gate_up_swiglu orelse return error.ShaderNotLoaded);
        const push = DmmvPushConstants{
            .M = M,
            .K = K,
            .a_offset = 0,
            .x_offset = 0,
            .y_offset = 0,
            .acc_mode = 0,
        };
        const wg_x: u32 = if (use_row1) M else (M + 1) / 2;
        self.pushDispatch4(
            pip,
            std.mem.asBytes(&push),
            gate_tensor.gpu_buffer.handle,
            gate_tensor.gpu_buffer.size,
            up_tensor.gpu_buffer.handle,
            up_tensor.gpu_buffer.size,
            input_buf.handle,
            input_size,
            swiglu_buf.handle,
            swiglu_buf.size,
            wg_x,
            1,
            1,
        );
    }

    /// Gemma CPU-MoE expert fusion for the packed Q4_K gate/up tensor:
    /// computes GEGLU(W_gate·x, W_up·x) directly into swiglu_buf.
    fn dispatchDmmvFusedGateUpGegluOffset(
        self: *InferenceEngine,
        tensor: *const LoadedTensor,
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        swiglu_buf: Buffer,
        M: u32,
        K: u32,
        gate_offset: u32,
        up_offset: u32,
    ) !void {
        const pip = &(self.dmmv.pipeline_q4k_fused_gate_up_geglu orelse return error.ShaderNotLoaded);
        const push = DmmvGateUpGegluPushConstants{
            .M = M,
            .K = K,
            .gate_offset = gate_offset,
            .up_offset = up_offset,
            .x_offset = 0,
            .y_offset = 0,
        };
        const wg_x: u32 = (M + 1) / 2;
        self.pushDispatch3(
            pip,
            std.mem.asBytes(&push),
            tensor.gpu_buffer.handle,
            tensor.gpu_buffer.size,
            input_buf.handle,
            input_size,
            swiglu_buf.handle,
            swiglu_buf.size,
            wg_x,
            1,
            1,
        );
    }

    /// Q5_1 expert down projection fused with `moe_out += weight * down`.
    fn dispatchDmmvQ5_1AccOffset(
        self: *InferenceEngine,
        tensor: *const LoadedTensor,
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        output_buf: Buffer,
        M: u32,
        K: u32,
        a_offset: u32,
        scale: f32,
    ) !void {
        const pip = &(self.dmmv.pipeline_q5_1_acc orelse return error.ShaderNotLoaded);
        const push = DmmvScaleAccPushConstants{
            .M = M,
            .K = K,
            .a_offset = a_offset,
            .x_offset = 0,
            .y_offset = 0,
            .scale_bits = @bitCast(scale),
        };
        self.pushDispatch3(
            pip,
            std.mem.asBytes(&push),
            tensor.gpu_buffer.handle,
            tensor.gpu_buffer.size,
            input_buf.handle,
            input_size,
            output_buf.handle,
            output_buf.size,
            (M + 1) / 2,
            1,
            1,
        );
    }

    /// Gemma batched CPU-topk MoE front-end: all selected experts run in one
    /// dispatch and write GEGLU activations into per-expert slabs.
    fn dispatchDmmvMoeFusedGateUpGeglu(
        self: *InferenceEngine,
        tensor: *const LoadedTensor,
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        swiglu_buf: Buffer,
        M: u32,
        K: u32,
        expert_stride: u32,
        up_offset: u32,
        n_used: u32,
    ) !void {
        const pip = &(self.dmmv.pipeline_q4k_moe_fused_gate_up_geglu orelse return error.ShaderNotLoaded);
        const push = MoeGateUpGegluPushConstants{
            .M = M,
            .K = K,
            .expert_stride = expert_stride,
            .up_offset = up_offset,
            .x_offset = 0,
            .y_offset = 0,
        };
        self.pushDispatch4(
            pip,
            std.mem.asBytes(&push),
            tensor.gpu_buffer.handle,
            tensor.gpu_buffer.size,
            input_buf.handle,
            input_size,
            swiglu_buf.handle,
            swiglu_buf.size,
            self.router_output_buf.handle,
            self.router_output_buf.size,
            (M + 1) / 2,
            n_used,
            1,
        );
    }

    /// Gemma batched CPU-topk MoE tail: one dispatch loops over all selected
    /// experts' Q5_1 down projections and writes the weighted sum.
    fn dispatchDmmvQ5_1MoeFusedDownAcc(
        self: *InferenceEngine,
        tensor: *const LoadedTensor,
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        output_buf: Buffer,
        M: u32,
        K: u32,
        expert_stride: u32,
        n_used: u32,
    ) !void {
        const pip = &(self.dmmv.pipeline_q5_1_moe_fused_down_acc orelse return error.ShaderNotLoaded);
        const push = MoeFusedDownAccPushConstants{
            .M = M,
            .K = K,
            .expert_stride = expert_stride,
            .x_expert_stride = K,
            .x_offset = 0,
            .y_offset = 0,
            .n_used = n_used,
        };
        self.pushDispatch4(
            pip,
            std.mem.asBytes(&push),
            tensor.gpu_buffer.handle,
            tensor.gpu_buffer.size,
            input_buf.handle,
            input_size,
            output_buf.handle,
            output_buf.size,
            self.router_output_buf.handle,
            self.router_output_buf.size,
            (M + 1) / 2,
            1,
            1,
        );
    }

    /// Gemma GPU-topk MoE tail with ffn_down_exps.scale applied inside the
    /// fused down+acc shader.
    fn dispatchDmmvQ5_1MoeFusedDownAccScaled(
        self: *InferenceEngine,
        tensor: *const LoadedTensor,
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        output_buf: Buffer,
        scale_tensor: *const LoadedTensor,
        M: u32,
        K: u32,
        expert_stride: u32,
        n_used: u32,
    ) !void {
        const pip = &(self.dmmv.pipeline_q5_1_moe_fused_down_acc_scaled orelse return error.ShaderNotLoaded);
        const push = MoeFusedDownAccPushConstants{
            .M = M,
            .K = K,
            .expert_stride = expert_stride,
            .x_expert_stride = K,
            .x_offset = 0,
            .y_offset = 0,
            .n_used = n_used,
        };
        self.pushDispatch5(
            pip,
            std.mem.asBytes(&push),
            tensor.gpu_buffer.handle,
            tensor.gpu_buffer.size,
            input_buf.handle,
            input_size,
            output_buf.handle,
            output_buf.size,
            self.router_output_buf.handle,
            self.router_output_buf.size,
            scale_tensor.gpu_buffer.handle,
            scale_tensor.gpu_buffer.size,
            (M + 1) / 2,
            1,
            1,
        );
    }

    /// Gemma GPU-topk MoE tail for Q8_0 down experts with ffn_down_exps.scale
    /// applied inside the fused down+acc shader.
    fn dispatchDmmvQ8_0MoeFusedDownAccScaled(
        self: *InferenceEngine,
        tensor: *const LoadedTensor,
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        output_buf: Buffer,
        scale_tensor: *const LoadedTensor,
        M: u32,
        K: u32,
        expert_stride: u32,
        n_used: u32,
    ) !void {
        const pip = &(self.dmmv.pipeline_q8_0_moe_fused_down_acc_scaled orelse return error.ShaderNotLoaded);
        const push = MoeFusedDownAccPushConstants{
            .M = M,
            .K = K,
            .expert_stride = expert_stride,
            .x_expert_stride = K,
            .x_offset = 0,
            .y_offset = 0,
            .n_used = n_used,
        };
        self.pushDispatch5(
            pip,
            std.mem.asBytes(&push),
            tensor.gpu_buffer.handle,
            tensor.gpu_buffer.size,
            input_buf.handle,
            input_size,
            output_buf.handle,
            output_buf.size,
            self.router_output_buf.handle,
            self.router_output_buf.size,
            scale_tensor.gpu_buffer.handle,
            scale_tensor.gpu_buffer.size,
            (M + 1) / 2,
            1,
            1,
        );
    }

    /// Q8_0 variant of dispatchDmmvFusedGateUpSwiglu. Same 4-binding
    /// shape and push struct; used by the shared expert path on Qwen
    /// 3.5 / 3.6 MoE packs where the shared FFN weights are Q8_0.
    fn dispatchDmmvFusedGateUpSwigluQ8_0(
        self: *InferenceEngine,
        gate_tensor: *const LoadedTensor,
        up_tensor: *const LoadedTensor,
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        swiglu_buf: Buffer,
        M: u32,
        K: u32,
        shared_gate_tensor: ?*const LoadedTensor,
    ) !void {
        const push = DmmvPushConstants{
            .M = M,
            .K = K,
            .a_offset = 0,
            .x_offset = 0,
            .y_offset = 0,
            .acc_mode = 0,
        };
        const wg_x: u32 = (M + 1) / 2;
        if (shared_gate_tensor) |sg| {
            const pip = &(self.dmmv.pipeline_q8_0_fused_gate_up_swiglu_gate orelse return error.ShaderNotLoaded);
            self.pushDispatch6(
                pip,
                std.mem.asBytes(&push),
                gate_tensor.gpu_buffer.handle,
                gate_tensor.gpu_buffer.size,
                up_tensor.gpu_buffer.handle,
                up_tensor.gpu_buffer.size,
                input_buf.handle,
                input_size,
                swiglu_buf.handle,
                swiglu_buf.size,
                sg.gpu_buffer.handle,
                sg.gpu_buffer.size,
                self.router_logits_buf.handle,
                @sizeOf(f32),
                wg_x,
                1,
                1,
            );
        } else {
            const pip = &(self.dmmv.pipeline_q8_0_fused_gate_up_swiglu orelse return error.ShaderNotLoaded);
            self.pushDispatch4(
                pip,
                std.mem.asBytes(&push),
                gate_tensor.gpu_buffer.handle,
                gate_tensor.gpu_buffer.size,
                up_tensor.gpu_buffer.handle,
                up_tensor.gpu_buffer.size,
                input_buf.handle,
                input_size,
                swiglu_buf.handle,
                swiglu_buf.size,
                wg_x,
                1,
                1,
            );
        }
    }

    /// Fused Q8_0 pair DMMV. Used by the SSM path to compute wqkv and z/gate
    /// projections from the same normalized hidden vector in one dispatch.
    fn dispatchDmmvQ8_0FusedPair(
        self: *InferenceEngine,
        tensor0: *const LoadedTensor,
        tensor1: *const LoadedTensor,
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        output0_buf: Buffer,
        output0_size: vk.c.VkDeviceSize,
        output1_buf: Buffer,
        output1_size: vk.c.VkDeviceSize,
        M0: u32,
        M1: u32,
        K: u32,
    ) !void {
        const pip = &(self.dmmv.pipeline_q8_0_fused_pair orelse return error.ShaderNotLoaded);
        const push = DmmvQ8PairPushConstants{
            .M0 = M0,
            .M1 = M1,
            .K = K,
        };
        const max_m = @max(M0, M1);
        const wg_x: u32 = (max_m + 1) / 2;
        self.pushDispatch5(
            pip,
            std.mem.asBytes(&push),
            tensor0.gpu_buffer.handle,
            tensor0.gpu_buffer.size,
            tensor1.gpu_buffer.handle,
            tensor1.gpu_buffer.size,
            input_buf.handle,
            input_size,
            output0_buf.handle,
            output0_size,
            output1_buf.handle,
            output1_size,
            wg_x,
            1,
            1,
        );
    }

    /// Fused Q8_0 down DMMV + sigmoid-gated scale-accumulate.
    /// Replaces (dispatchDmmv(down_shexp) + computeBarrier +
    /// dispatchSigmoidScaleAcc + computeBarrier) on the Qwen 3.5/3.6
    /// shared-expert tail. Computes:
    ///   hidden_buf[row] += sigmoid(gate_buf[gate_offset]) * sum_k(W[row,k] * input[k])
    /// in a single dispatch. Saves 1 dispatch + 1 barrier per layer per token.
    fn dispatchDmmvQ8_0SigmoidAcc(
        self: *InferenceEngine,
        down_tensor: *const LoadedTensor,
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        out_buf: Buffer,
        out_size: vk.c.VkDeviceSize,
        gate_buf: Buffer,
        gate_size: vk.c.VkDeviceSize,
        M: u32,
        K: u32,
        gate_offset: u32,
    ) !void {
        const pip = &(self.dmmv.pipeline_q8_0_sigmoid_acc orelse return error.ShaderNotLoaded);
        const push = DmmvSigmoidAccPushConstants{
            .M = M,
            .K = K,
            .a_offset = 0,
            .x_offset = 0,
            .y_offset = 0,
            .gate_offset = gate_offset,
        };
        const wg_x: u32 = (M + 1) / 2;
        self.pushDispatch4(
            pip,
            std.mem.asBytes(&push),
            down_tensor.gpu_buffer.handle,
            down_tensor.gpu_buffer.size,
            input_buf.handle,
            input_size,
            out_buf.handle,
            out_size,
            gate_buf.handle,
            gate_size,
            wg_x,
            1,
            1,
        );
    }

    /// Dispatch the fused split-K flash attention merge + Q4_K o_proj
    /// DMMV-acc shader. Replaces the (flash_attn_split_merge → barrier →
    /// dispatchDmmvAcc(o_proj)) trio with a single dispatch that reads
    /// partials directly, computes the per-head LSE merge weights with
    /// sink fold-in, stages the merged attn_out into LDS (16 KB), and
    /// runs the standard Q4_K matmul reading the B-vector from LDS while
    /// accumulating into hidden_buf. Push-descriptor only.
    fn dispatchDmmvOprojMerge(
        self: *InferenceEngine,
        o_tensor: *const LoadedTensor,
        partial_buf: Buffer,
        sinks_buf: Buffer,
        hidden_buf: Buffer,
        M: u32,
        K: u32,
        n_heads: u32,
        n_i_chunks: u32,
        sink_offset: u32,
        head_dim: u32,
    ) !void {
        const pip = &(self.dmmv.pipeline_q4k_o_proj_merge orelse return error.ShaderNotLoaded);
        const push = OprojMergePushConstants{
            .M = M,
            .K = K,
            .a_offset = 0,
            .x_offset = 0,
            .y_offset = 0,
            .acc_mode = 1,
            .n_heads = n_heads,
            .n_i_chunks = n_i_chunks,
            .sink_offset = sink_offset,
            .head_dim = head_dim,
        };
        // NUM_ROWS=2 in the shader; one workgroup per row pair.
        const wg_x: u32 = (M + 1) / 2;
        self.pushDispatch4(
            pip,
            std.mem.asBytes(&push),
            o_tensor.gpu_buffer.handle,
            o_tensor.gpu_buffer.size,
            partial_buf.handle,
            partial_buf.size,
            hidden_buf.handle,
            hidden_buf.size,
            sinks_buf.handle,
            sinks_buf.size,
            wg_x,
            1,
            1,
        );
    }

    /// Dispatch a DMMV with byte offset into stacked weight tensor (for MoE experts).
    fn dispatchDmmvWithOffset(
        self: *InferenceEngine,
        tensor: *const LoadedTensor,
        /// GPU buffer for input buf.
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        /// GPU buffer for output buf.
        output_buf: Buffer,
        M: u32,
        K: u32,
        /// Weight buffer byte offset.
        a_offset: u32,
    ) !void {
        return self.dispatchDmmvInner(tensor, input_buf, input_size, output_buf, M, K, a_offset, 0, 0, 0);
    }

    /// Inner dispatch for DMMV — push-descriptor or pool-allocated path.
    fn dispatchDmmvInner(
        self: *InferenceEngine,
        tensor: *const LoadedTensor,
        input_buf: Buffer,
        input_size: vk.c.VkDeviceSize,
        output_buf: Buffer,
        M: u32,
        K: u32,
        a_offset: u32,
        x_offset: u32,
        y_offset: u32,
        acc_mode: u32,
    ) !void {
        const qt = tensor.info.type_;
        const pip = self.dmmv.pipelineForType(qt) orelse {
            log.err("No DMMV pipeline for quant type {d} (tensor {s})", .{ @intFromEnum(qt), tensor.info.name });
            return error.UnsupportedQuantType;
        };

        if (pip.uses_push_descriptors) {
            // Wide-vocab Q8_0 LM-head fast path. The Qwen 3.6 35B-A3B
            // output.weight is Q8_0 with M=248320, K=2048. This variant keeps
            // two rows/WG but shares each x-vector load across both rows.
            // Keep it limited to tall overwrite DMMVs so smaller SSM/shared
            // projections stay on the generic path.
            if (self.use_q8_wide_lm_head and qt == .q8_0 and M >= 100_000 and acc_mode == 0 and self.dmmv.pipeline_q8_0_wide != null) {
                const wide_pip = &self.dmmv.pipeline_q8_0_wide.?;
                const push_wide = DmmvPushConstants{
                    .M = M,
                    .K = K,
                    .a_offset = a_offset,
                    .x_offset = x_offset,
                    .y_offset = y_offset,
                    .acc_mode = acc_mode,
                };
                self.pushDispatch3(
                    wide_pip,
                    std.mem.asBytes(&push_wide),
                    tensor.gpu_buffer.handle,
                    tensor.gpu_buffer.size,
                    input_buf.handle,
                    input_size,
                    output_buf.handle,
                    output_buf.size,
                    (M + 1) / 2,
                    1,
                    1,
                );
                return;
            }

            // Q8_0 hot-shape specialization. Most Qwen 3.6 Q8 decode matvecs
            // have K=2048 (one 64-block pass) or K=4096 (two passes). The
            // specialized pipelines bake that block count so RADV can fold the
            // loop shape; the generic pipeline remains the fallback for other K.
            if (self.use_q8_spec_dmmv and qt == .q8_0 and acc_mode == 0) {
                const q8_spec_pip: ?*const Pipeline = if (K == 2048 and self.dmmv.pipeline_q8_0_spec64 != null)
                    &self.dmmv.pipeline_q8_0_spec64.?
                else if (K == 4096 and self.dmmv.pipeline_q8_0_spec128 != null)
                    &self.dmmv.pipeline_q8_0_spec128.?
                else
                    null;
                if (q8_spec_pip) |spec_pip| {
                    const push_spec = DmmvPushConstants{
                        .M = M,
                        .K = K,
                        .a_offset = a_offset,
                        .x_offset = x_offset,
                        .y_offset = y_offset,
                        .acc_mode = acc_mode,
                    };
                    self.pushDispatch3(
                        spec_pip,
                        std.mem.asBytes(&push_spec),
                        tensor.gpu_buffer.handle,
                        tensor.gpu_buffer.size,
                        input_buf.handle,
                        input_size,
                        output_buf.handle,
                        output_buf.size,
                        (M + 1) / 2,
                        1,
                        1,
                    );
                    return;
                }
            }

            // Wide-vocab LM-head fast path: NUM_ROWS=32 variant (pipeline_q4k_wide)
            // for tall Q4_K matrices like Gemma 4 31B (M=262144). Same binding
            // layout as pipeline_q4k, only the shader constant differs. 16× fewer
            // workgroups, 16× more hidden-vector reuse per workgroup, which
            // turns the decode tail from ~45 ms into a small fraction.
            if (qt == .q4_k and M >= 100_000 and acc_mode == 0 and self.dmmv.pipeline_q4k_wide != null) {
                const wide_pip = &self.dmmv.pipeline_q4k_wide.?;
                const push_wide = DmmvPushConstants{
                    .M = M,
                    .K = K,
                    .a_offset = a_offset,
                    .x_offset = x_offset,
                    .y_offset = y_offset,
                    .acc_mode = acc_mode,
                };
                // NUM_ROWS=32 → one workgroup per 32 rows.
                self.pushDispatch3(
                    wide_pip,
                    std.mem.asBytes(&push_wide),
                    tensor.gpu_buffer.handle,
                    tensor.gpu_buffer.size,
                    input_buf.handle,
                    input_size,
                    output_buf.handle,
                    output_buf.size,
                    (M + 31) / 32,
                    1,
                    1,
                );
                return;
            }

            // Wide-vocab Q6_K LM-head fast path. Qwen3.6 27B stores
            // output.weight as Q6_K with M=248320, so the generic two-row
            // kernel launches ~124k workgroups and reloads the same normalized
            // hidden vector for every row pair. The wide variant computes 8 rows
            // per workgroup and reuses each X tile across those rows.
            if (qt == .q6_k and M >= 100_000 and acc_mode == 0 and self.dmmv.pipeline_q6k_wide != null) {
                const wide_pip = &self.dmmv.pipeline_q6k_wide.?;
                const push_wide = DmmvPushConstants{
                    .M = M,
                    .K = K,
                    .a_offset = a_offset,
                    .x_offset = x_offset,
                    .y_offset = y_offset,
                    .acc_mode = acc_mode,
                };
                self.pushDispatch3(
                    wide_pip,
                    std.mem.asBytes(&push_wide),
                    tensor.gpu_buffer.handle,
                    tensor.gpu_buffer.size,
                    input_buf.handle,
                    input_size,
                    output_buf.handle,
                    output_buf.size,
                    (M + 7) / 8,
                    1,
                    1,
                );
                return;
            }

            // For Q4K large M (LM head), use batch shader for better parallelism.
            if (qt == .q4_k and M > 65536 and self.dmmv.pipeline_q4k_batch != null) {
                try self.dmmv.recordBatchDispatchPush(
                    &self.decode_cmd,
                    qt,
                    self.instance.push_descriptor_fn,
                    tensor.gpu_buffer.handle,
                    tensor.gpu_buffer.size,
                    input_buf.handle,
                    input_size,
                    output_buf.handle,
                    output_buf.size,
                    M,
                    K,
                    a_offset,
                    x_offset,
                    y_offset,
                    1,
                );
                return;
            }

            const push = DmmvPushConstants{
                .M = M,
                .K = K,
                .a_offset = a_offset,
                .x_offset = x_offset,
                .y_offset = y_offset,
                .acc_mode = acc_mode,
            };
            // Workgroup calculation (mirrors dmmv.recordDispatch)
            const wg_x: u32 = switch (qt) {
                .q4_k, .q5_0, .q5_1, .q5_k, .q6_k => (M + 1) / 2,
                .mxfp4, .q8_0, .f16 => (M + 1) / 2,
                .f32 => M,
                else => (M + 63) / 64,
            };
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                tensor.gpu_buffer.handle,
                tensor.gpu_buffer.size,
                input_buf.handle,
                input_size,
                output_buf.handle,
                output_buf.size,
                wg_x,
                1,
                1,
            );
            return;
        }

        // Fallback: pool-allocated descriptor set
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds, tensor.gpu_buffer.handle, tensor.gpu_buffer.size, input_buf.handle, input_size, output_buf.handle, output_buf.size);
        try self.dmmv.recordDispatch(&self.decode_cmd, qt, ds, M, K, a_offset, x_offset, y_offset);
    }

    /// Dispatch a MoE DMMV — expert offset computed on GPU from routing buffer.

    // -----------------------------------------------------------------------
    // CPU-side SSM / delta-net layer
    // -----------------------------------------------------------------------

    /// Run one SSM layer: GPU for large projections, CPU for small state ops.
    fn runSsmLayerCpu(self: *InferenceEngine, state: *DecodeState, layer: u32, layer_idx: usize) !void {
        const config = &self.model.config;
        const hidden_dim = config.hidden_dim;
        const hidden_size = @as(vk.c.VkDeviceSize, hidden_dim) * @sizeOf(f32);
        const d_inner = config.ssm_d_inner;
        const d_conv = config.ssm_d_conv;
        const d_state = config.ssm_d_state;
        const n_group = config.ssm_n_group;
        const dt_rank = config.ssm_dt_rank;

        if (d_inner == 0) return;

        const head_v_dim = d_inner / dt_rank;
        const conv_channels = d_inner + 2 * n_group * d_state;
        const lt = self.layer_tensors[layer];

        // --- GPU phase 1: Run large projections via DMMV ---
        const wqkv_tensor = lt.attn_qkv orelse return;
        try self.dispatchDmmv(wqkv_tensor, self.norm_buf, hidden_size, self.attn_out_buf, @intCast(conv_channels), hidden_dim);

        const z_tensor = lt.attn_gate orelse return;
        try self.dispatchDmmv(z_tensor, self.norm_buf, hidden_size, self.gate_buf, @intCast(d_inner), hidden_dim);

        const alpha_tensor = lt.ssm_alpha orelse return;
        try self.dispatchDmmv(alpha_tensor, self.norm_buf, hidden_size, self.router_logits_buf, dt_rank, hidden_dim);

        const beta_tensor = lt.ssm_beta orelse return;
        try self.dispatchDmmv(beta_tensor, self.norm_buf, hidden_size, self.down_buf, dt_rank, hidden_dim);
        if (layer == 0) {
            const conv_tensor = lt.ssm_conv1d orelse return;
            const ssm_out_tensor = lt.ssm_out orelse return;
            log.info("FASTPATH: ssm qkv={s} gate={s} alpha={s} beta={s} conv={s} out={s}", .{
                @tagName(wqkv_tensor.info.type_),
                @tagName(z_tensor.info.type_),
                @tagName(alpha_tensor.info.type_),
                @tagName(beta_tensor.info.type_),
                @tagName(conv_tensor.info.type_),
                @tagName(ssm_out_tensor.info.type_),
            });
        }
        self.decode_cmd.computeBarrier();

        // --- Readback projection results to CPU via logits_staging ---
        const qkv_bytes = @as(vk.c.VkDeviceSize, conv_channels) * @sizeOf(f32);
        const z_bytes = @as(vk.c.VkDeviceSize, d_inner) * @sizeOf(f32);
        const ab_bytes = @as(vk.c.VkDeviceSize, dt_rank) * @sizeOf(f32);
        {
            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
            };
            vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, null, 0, null);

            const r1 = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = qkv_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.attn_out_buf.handle, self.logits_staging.handle, 1, &r1);
            const r2 = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = qkv_bytes, .size = z_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.gate_buf.handle, self.logits_staging.handle, 1, &r2);
            const r3 = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = qkv_bytes + z_bytes, .size = ab_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.router_logits_buf.handle, self.logits_staging.handle, 1, &r3);
            const r4 = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = qkv_bytes + z_bytes + ab_bytes, .size = ab_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.down_buf.handle, self.logits_staging.handle, 1, &r4);
        }
        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

        // --- CPU phase: conv1d + delta-net state update ---
        const staging_f32: [*]f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
        const qkv_cpu = staging_f32[0..conv_channels];
        const z_cpu = staging_f32[conv_channels..][0..d_inner];
        const alpha_cpu = staging_f32[conv_channels + d_inner ..][0..dt_rank];
        const beta_cpu = staging_f32[conv_channels + d_inner + dt_rank ..][0..dt_rank];

        // Conv1d with state
        const conv_state = self.ssm_conv_states[layer_idx];
        const d_conv_1 = d_conv - 1;
        const mmap = self.model.mmap_data orelse return error.NoMmapData;
        const conv_tensor = lt.ssm_conv1d orelse return;
        const conv_data_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + conv_tensor.info.offset);
        // Bug fix #14: Read conv kernel handling f16 storage — direct f32 cast corrupts values
        const conv_kernel_len = conv_channels * d_conv;
        const conv_kernel_buf = try self.allocator.alloc(f32, conv_kernel_len);
        defer self.allocator.free(conv_kernel_buf);
        readMmapFloats(mmap, conv_data_off, conv_tensor.info.type_, conv_kernel_buf);
        if (layer == 0) log.info("SSM tensor types: conv1d={s} dt_bias={s} ssm_a={s} n_group={d} dt_rank={d} d_state={d} head_v={d}", .{
            @tagName(conv_tensor.info.type_),
            if (lt.ssm_dt_bias) |t| @tagName(t.info.type_) else "N/A",
            if (lt.ssm_a) |t| @tagName(t.info.type_) else "N/A",
            n_group,
            dt_rank,
            d_state,
            head_v_dim,
        });

        // Bug fix #12: Convolve BEFORE updating state to avoid double-counting the current input.
        // State holds the previous d_conv-1 inputs; qkv_cpu is the current input (ki=d_conv-1).
        const conv_out = try self.allocator.alloc(f32, conv_channels);
        defer self.allocator.free(conv_out);
        for (0..conv_channels) |ch| {
            var sum: f32 = 0;
            for (0..d_conv) |ki| {
                // Bug fix #7: GGUF stores conv kernel as [d_conv, conv_channels] (d_conv is fast dim)
                const kw = conv_kernel_buf[ch * d_conv + ki];
                const sv = if (ki < d_conv_1) conv_state[ki * conv_channels + ch] else qkv_cpu[ch];
                sum += kw * sv;
            }
            const sig = 1.0 / (1.0 + @exp(-sum));
            conv_out[ch] = sum * sig;
        }

        // Now update conv state: shift left and write current input as newest entry
        if (d_conv_1 > 1) {
            const shift = (d_conv_1 - 1) * conv_channels;
            std.mem.copyForwards(f32, conv_state[0..shift], conv_state[conv_channels .. shift + conv_channels]);
        }
        @memcpy(conv_state[(d_conv_1 - 1) * conv_channels ..][0..conv_channels], qkv_cpu);

        // Split Q/K/V from conv output — llama.cpp layout: [Q(n_group*d_state), K(n_group*d_state), V(d_inner)]
        const qk_dim = d_state * n_group;
        var q_ssm = conv_out[0..qk_dim];
        var k_ssm = conv_out[qk_dim .. 2 * qk_dim];
        const v_ssm = conv_out[2 * qk_dim .. 2 * qk_dim + d_inner];

        // Bug fix #8: L2 normalize per-head, not across all heads
        for (0..n_group) |h| {
            l2Normalize(q_ssm[h * d_state ..][0..d_state]);
            l2Normalize(k_ssm[h * d_state ..][0..d_state]);
        }

        // Compute gate and beta
        // Bug fix #14: Read dt_bias and ssm_a handling f16 storage type
        const dt_bias_tensor = lt.ssm_dt_bias;
        const dt_bias_f32 = try self.allocator.alloc(f32, dt_rank);
        defer self.allocator.free(dt_bias_f32);
        if (dt_bias_tensor) |t| {
            const off: usize = @intCast(self.model.gguf_file.tensor_data_offset + t.info.offset);
            readMmapFloats(mmap, off, t.info.type_, dt_bias_f32);
        }

        const ssm_a_tensor = lt.ssm_a;
        const ssm_a_f32 = try self.allocator.alloc(f32, dt_rank);
        defer self.allocator.free(ssm_a_f32);
        if (ssm_a_tensor) |t| {
            const off: usize = @intCast(self.model.gguf_file.tensor_data_offset + t.info.offset);
            readMmapFloats(mmap, off, t.info.type_, ssm_a_f32);
            if (layer == 0 and state.position == 0) {
                var ssm_a_min: f32 = std.math.inf(f32);
                var ssm_a_max: f32 = -std.math.inf(f32);
                for (ssm_a_f32[0..dt_rank]) |v| {
                    ssm_a_min = @min(ssm_a_min, v);
                    ssm_a_max = @max(ssm_a_max, v);
                }
                log.info("SSM_A_STATS L0: min={d:.6} max={d:.6} first4=[{d:.6},{d:.6},{d:.6},{d:.6}]", .{
                    ssm_a_min,
                    ssm_a_max,
                    ssm_a_f32[0],
                    ssm_a_f32[@min(@as(usize, 1), dt_rank - 1)],
                    ssm_a_f32[@min(@as(usize, 2), dt_rank - 1)],
                    ssm_a_f32[@min(@as(usize, 3), dt_rank - 1)],
                });
            }
        }

        const gate_arr = try self.allocator.alloc(f32, dt_rank);
        defer self.allocator.free(gate_arr);
        const beta_arr = try self.allocator.alloc(f32, dt_rank);
        defer self.allocator.free(beta_arr);
        for (0..dt_rank) |i| {
            var a = alpha_cpu[i];
            if (dt_bias_tensor != null) a += dt_bias_f32[i];
            const sp = @log(1.0 + @exp(a));
            gate_arr[i] = if (ssm_a_tensor != null) sp * ssm_a_f32[i] else -sp;
            beta_arr[i] = 1.0 / (1.0 + @exp(-beta_cpu[i]));
        }
        if (layer == 0 and (state.position == 0 or state.position == 64 or state.position == 128 or state.position == 192)) {
            var gate_min: f32 = std.math.inf(f32);
            var gate_max: f32 = -std.math.inf(f32);
            var beta_min: f32 = std.math.inf(f32);
            var beta_max: f32 = -std.math.inf(f32);
            var decay_min: f32 = std.math.inf(f32);
            var decay_max: f32 = -std.math.inf(f32);
            for (0..dt_rank) |i| {
                gate_min = @min(gate_min, gate_arr[i]);
                gate_max = @max(gate_max, gate_arr[i]);
                beta_min = @min(beta_min, beta_arr[i]);
                beta_max = @max(beta_max, beta_arr[i]);
                const decay = @exp(gate_arr[i]);
                decay_min = @min(decay_min, decay);
                decay_max = @max(decay_max, decay);
            }
            log.debug("SSM gate L0 pos={d}: alpha0={d:.6} dt_bias0={d:.6} ssm_a0={d:.6} gate_log=[{d:.6},{d:.6}] decay=[{d:.6},{d:.6}] beta=[{d:.6},{d:.6}]", .{
                state.position,
                alpha_cpu[0],
                if (dt_bias_tensor != null) dt_bias_f32[0] else 0.0,
                if (ssm_a_tensor != null) ssm_a_f32[0] else 0.0,
                gate_min,
                gate_max,
                decay_min,
                decay_max,
                beta_min,
                beta_max,
            });
        }

        // Bug fix #9: Scale Q by 1/sqrt(head_k_dim) before state readout
        const q_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_state)));
        for (q_ssm) |*v| v.* *= q_scale;

        // Delta-net autoregressive update
        // Bug fix #10: State layout s[row][col] where:
        //   sk[row] = sum_col s[row][col] * k[col]
        //   s[row][col] += k[row] * d[col]   (outer product)
        //   o[row] = sum_col s[row][col] * q[col]
        const ssm_state = self.ssm_states[layer_idx];
        for (0..dt_rank) |h| {
            const s_base = h * head_v_dim * head_v_dim;
            const g_val = @exp(gate_arr[h]);
            const b_val = beta_arr[h];
            const k_hi = if (n_group == dt_rank) h else h % n_group;
            const k_head = k_ssm[k_hi * d_state ..][0..@min(d_state, head_v_dim)];
            const v_head = v_ssm[h * head_v_dim ..][0..head_v_dim];

            // Decay: s *= exp(gate)
            for (0..head_v_dim * head_v_dim) |i| ssm_state[s_base + i] *= g_val;

            // Match the GPU delta-net shader: each row is decayed, corrected with
            // d[row] = beta * (v[row] - dot(state[row], k)), then updated in-place.
            for (0..head_v_dim) |row| {
                var sk: f32 = 0;
                for (0..@min(head_v_dim, k_head.len)) |col| {
                    sk += ssm_state[s_base + row * head_v_dim + col] * k_head[col];
                }
                const d_val = b_val * (v_head[row] - sk);
                for (0..@min(head_v_dim, k_head.len)) |col| {
                    ssm_state[s_base + row * head_v_dim + col] += k_head[col] * d_val;
                }
            }
        }

        // Read from state: o[row] = sum_col s[row][col] * q[col]
        const ssm_output = try self.allocator.alloc(f32, d_inner);
        defer self.allocator.free(ssm_output);
        for (0..dt_rank) |h| {
            const s_base = h * head_v_dim * head_v_dim;
            const q_hi = if (n_group == dt_rank) h else h % n_group;
            const q_head = q_ssm[q_hi * d_state ..][0..@min(d_state, head_v_dim)];
            for (0..head_v_dim) |row| {
                var val: f32 = 0;
                for (0..@min(head_v_dim, q_head.len)) |col| {
                    val += ssm_state[s_base + row * head_v_dim + col] * q_head[col];
                }
                ssm_output[h * head_v_dim + row] = val;
            }
        }

        // Debug: dump SSM delta-net output before gated norm
        if (layer == 0) {
            var ssm_l2: f64 = 0;
            for (ssm_output) |v| ssm_l2 += @as(f64, v) * @as(f64, v);
            ssm_l2 = @sqrt(ssm_l2);
            log.info("SSM_DBG L0 delta_out[0..4]=[{d:.8},{d:.8},{d:.8},{d:.8}] L2={d:.6}", .{
                ssm_output[0], ssm_output[1], ssm_output[2], ssm_output[3], ssm_l2,
            });
            // CPU ref: [4.84e-06, 4.69e-06, 1.369e-05, -9.25e-06] L2=0.009320
        }

        // Gated normalization: RMS_norm(o) * SiLU(z)
        const norm_tensor = lt.ssm_norm;
        // Determine norm weight indexing: per-head (d_inner elements) vs shared (d_state elements)
        const norm_elems: u32 = if (norm_tensor) |t| @intCast(t.info.numElements()) else 0;
        const norm_per_head = norm_elems >= d_inner;
        // Bug fix #14: Read norm weights handling f16 storage type
        const norm_alloc_len: u32 = if (norm_elems > 0) norm_elems else 1;
        const norm_w_buf = try self.allocator.alloc(f32, norm_alloc_len);
        defer self.allocator.free(norm_w_buf);
        if (norm_tensor) |t| {
            const off: usize = @intCast(self.model.gguf_file.tensor_data_offset + t.info.offset);
            readMmapFloats(mmap, off, t.info.type_, norm_w_buf[0..norm_elems]);
        }
        // Log ssm_norm shape once (first SSM layer) to verify indexing
        if (layer == 0) {
            if (norm_tensor) |t| {
                log.info("ssm_norm.weight: type={s} n_dims={d} dims=[{d},{d}] elems={d} d_state={d} d_inner={d} head_v={d} per_head={}", .{
                    @tagName(t.info.type_),
                    t.info.n_dims,
                    t.info.dims[0],
                    t.info.dims[1],
                    t.info.numElements(),
                    d_state,
                    d_inner,
                    head_v_dim,
                    norm_per_head,
                });
            } else {
                log.info("ssm_norm.weight: NOT FOUND for layer 0", .{});
            }
        }

        for (0..dt_rank) |h| {
            const o_sl = ssm_output[h * head_v_dim ..][0..head_v_dim];
            const z_sl = z_cpu[h * head_v_dim ..][0..head_v_dim];
            var sq: f32 = 0;
            for (o_sl) |v| sq += v * v;
            const rms = @sqrt(sq / @as(f32, @floatFromInt(head_v_dim)) + config.rms_norm_eps);
            for (0..head_v_dim) |i| {
                var nv = o_sl[i] / rms;
                // Use per-head indexing if tensor has d_inner elements, else shared d_state weights
                if (norm_tensor != null) nv *= norm_w_buf[if (norm_per_head) h * head_v_dim + i else i % d_state];
                const zv = z_sl[i];
                o_sl[i] = nv * (zv / (1.0 + @exp(-zv)));
            }
        }

        // --- GPU phase 2: ssm_out DMMV + residual ---
        const out_staging: [*]f32 = @ptrCast(@alignCast(self.ssm_hidden_staging.mapped.?));
        // Debug: dump after gated norm
        if (layer == 0) {
            var gn_l2: f64 = 0;
            for (ssm_output) |v| gn_l2 += @as(f64, v) * @as(f64, v);
            gn_l2 = @sqrt(gn_l2);
            log.info("SSM_DBG L0 gated_norm[0..4]=[{d:.8},{d:.8},{d:.8},{d:.8}] L2={d:.6}", .{
                ssm_output[0], ssm_output[1], ssm_output[2], ssm_output[3], gn_l2,
            });
            // CPU ref: [-0.00017421, -0.00023175, -0.00166175, -0.00414048] L2=?
        }

        @memcpy(out_staging[0..d_inner], ssm_output);

        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.begin();

        {
            const r = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = z_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.ssm_hidden_staging.handle, self.swiglu_buf.handle, 1, &r);
        }
        self.decode_cmd.transferToComputeBarrier();

        // Fused: ssm_out DMMV accumulates directly into hidden_buf
        const ssm_out_tensor = lt.ssm_out orelse return;
        try self.dispatchDmmvAcc(ssm_out_tensor, self.swiglu_buf, z_bytes, self.hidden_buf, hidden_dim, @intCast(d_inner));
        self.decode_cmd.computeBarrier();
    }

    /// Run one SSM layer entirely on GPU via compute shaders (Phase 3c).
    /// Replaces runSsmLayerCpu — no readback, no CPU computation, no submitAndWait.
    /// Command buffer remains open after this function returns.
    /// When `use_fused_pre_norm` is set, the caller has skipped the standalone
    /// rms_norm dispatch + barrier so this routine can fold them into the
    /// alpha/beta SSM proj fused shader (rms_norm_dmmv_q4k_alpha_beta).
    fn runSsmLayerGpu(self: *InferenceEngine, state: *DecodeState, layer: u32, layer_idx: usize, is_dead_tail: bool, use_fused_pre_norm: bool) !void {
        const config = &self.model.config;
        const hidden_dim = config.hidden_dim;
        const hidden_size = @as(vk.c.VkDeviceSize, hidden_dim) * @sizeOf(f32);
        const d_inner = config.ssm_d_inner;
        const d_conv = config.ssm_d_conv;
        const d_state = config.ssm_d_state;
        const n_group = config.ssm_n_group;
        const dt_rank = config.ssm_dt_rank;

        if (d_inner == 0) return;

        const head_v_dim: u32 = d_inner / dt_rank;
        const conv_channels: u32 = d_inner + 2 * n_group * d_state;
        const qkv_bytes = @as(vk.c.VkDeviceSize, conv_channels) * @sizeOf(f32);
        const z_bytes = @as(vk.c.VkDeviceSize, d_inner) * @sizeOf(f32);
        const ab_bytes = @as(vk.c.VkDeviceSize, dt_rank) * @sizeOf(f32);
        const use_delta_cols8 = self.use_ssm_delta_cols8 and
            !self.use_a3b_validate and
            !self.use_qwen36_ssm_prefill_validate and
            head_v_dim == 128 and
            d_state == 128 and
            self.elementwise.pipeline_ssm_delta_net_cols8 != null;
        const use_delta_normed_qk = use_delta_cols8 and
            self.use_ssm_delta_normed_qk and
            !self.validation_diagnostics_enabled and
            self.elementwise.pipeline_ssm_qk_norm != null and
            self.elementwise.pipeline_ssm_delta_net_cols8_normed != null;

        // --- GPU: 4 DMMV projections (same as CPU path) ---
        const lt = self.layer_tensors[layer];
        const wqkv_tensor = lt.attn_qkv orelse return;
        const z_tensor = lt.attn_gate orelse return;
        const alpha_tensor = lt.ssm_alpha orelse return;
        const beta_tensor = lt.ssm_beta orelse return;
        if (state.position == 0 and layer == 0) {
            const conv_tensor = lt.ssm_conv1d orelse return;
            const ssm_out_tensor = lt.ssm_out orelse return;
            log.debug("FASTPATH: ssm qkv={s} gate={s} alpha={s} beta={s} conv={s} out={s}", .{
                @tagName(wqkv_tensor.info.type_),
                @tagName(z_tensor.info.type_),
                @tagName(alpha_tensor.info.type_),
                @tagName(beta_tensor.info.type_),
                @tagName(conv_tensor.info.type_),
                @tagName(ssm_out_tensor.info.type_),
            });
        }
        const ssm_proj_phase = self.beginProfilePhase();
        const can_fuse_qkv_z = self.use_fused_ssm_qkv_z and
            !is_dead_tail and
            wqkv_tensor.info.type_ == .q8_0 and
            z_tensor.info.type_ == .q8_0 and
            self.dmmv.pipeline_q8_0_fused_pair != null and
            self.instance.push_descriptor_fn != null and
            (hidden_dim & 31) == 0;
        const use_prebatched_ssm_proj = self.partialSsmPreprojActiveFor(layer);

        if (use_fused_pre_norm) {
            // Fused fast path: rms_norm + alpha DMMV + beta DMMV in one
            // dispatch. WG 0 of the fused shader writes norm_buf so the
            // wqkv/z DMMVs below see a pre-normalized hidden vector. The
            // standalone rms_norm dispatch + barrier was skipped at the
            // call site so the actual saving is +1 dispatch (alpha+beta
            // merged) +1 dispatch (no standalone rms_norm) +1 barrier
            // (no rms_norm → SSM proj fence) per SSM layer.
            const attn_norm = lt.attn_norm orelse return error.TensorNotFound;
            const ssm_proj_norm_ab_phase = self.beginProfilePhase();
            try self.dispatchRmsNormDmmvQ4kAlphaBeta(
                self.hidden_buf.handle,
                hidden_size,
                attn_norm.gpu_buffer.handle,
                attn_norm.gpu_buffer.size,
                alpha_tensor.gpu_buffer.handle,
                alpha_tensor.gpu_buffer.size,
                beta_tensor.gpu_buffer.handle,
                beta_tensor.gpu_buffer.size,
                self.norm_buf.handle,
                hidden_size,
                self.router_logits_buf.handle,
                ab_bytes,
                self.down_buf.handle,
                ab_bytes,
                dt_rank,
                hidden_dim,
                self.model.config.rms_norm_eps,
            );
            self.endProfilePhase(.ssm_proj_norm_ab, ssm_proj_norm_ab_phase);
            // Barrier: wqkv/z DMMVs only read norm_buf (the fused shader's
            // single-writer WG-0 output). The alpha/beta outputs go to
            // router_logits_buf / down_buf, which are explicitly resynced
            // by the conv→delta multi-buffer barrier (line ~8146-8151)
            // before delta_net consumes them. Narrowing the barrier scope
            // to norm_buf lets the fused shader's alpha/beta writes
            // (~256 bytes per layer) overlap with the wqkv/z dispatches
            // and trims the global s_waitcnt that a full computeBarrier
            // would emit on RDNA4. Cycle 16 narrow.
            if (use_prebatched_ssm_proj) {
                const tok_idx = self.partial_ssm_preproj_token_idx;
                const use_prebatched_qkv = self.partial_ssm_preproj_qkv != null and self.partial_ssm_preproj_qkv_stride > 0;
                const use_prebatched_z = self.partial_ssm_preproj_z != null and self.partial_ssm_preproj_z_stride > 0;
                self.decode_cmd.computeAndTransferBarrier();
                if (use_prebatched_qkv) {
                    const qkv_src_off: vk.c.VkDeviceSize =
                        @as(vk.c.VkDeviceSize, tok_idx) * self.partial_ssm_preproj_qkv_stride;
                    if (qkv_src_off + qkv_bytes > self.partial_ssm_preproj_qkv_size) return error.BufferTooSmall;
                    const qkv_copy = vk.c.VkBufferCopy{
                        .srcOffset = qkv_src_off,
                        .dstOffset = 0,
                        .size = qkv_bytes,
                    };
                    const ssm_proj_qkv_phase = self.beginProfilePhase();
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.partial_ssm_preproj_qkv.?, self.attn_out_buf.handle, 1, &qkv_copy);
                    self.endProfilePhase(.ssm_proj_qkv, ssm_proj_qkv_phase);
                }
                if (use_prebatched_z and !is_dead_tail) {
                    const z_src_off: vk.c.VkDeviceSize =
                        @as(vk.c.VkDeviceSize, tok_idx) * self.partial_ssm_preproj_z_stride;
                    if (z_src_off + z_bytes > self.partial_ssm_preproj_z_size) return error.BufferTooSmall;
                    const z_copy = vk.c.VkBufferCopy{
                        .srcOffset = z_src_off,
                        .dstOffset = 0,
                        .size = z_bytes,
                    };
                    const ssm_proj_z_phase = self.beginProfilePhase();
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.partial_ssm_preproj_z.?, self.gate_buf.handle, 1, &z_copy);
                    self.endProfilePhase(.ssm_proj_z, ssm_proj_z_phase);
                }
                self.decode_cmd.transferToComputeBarrier();
                if (!use_prebatched_qkv) {
                    const ssm_proj_qkv_phase = self.beginProfilePhase();
                    try self.dispatchDmmv(wqkv_tensor, self.norm_buf, hidden_size, self.attn_out_buf, @intCast(conv_channels), hidden_dim);
                    self.endProfilePhase(.ssm_proj_qkv, ssm_proj_qkv_phase);
                }
                if (!is_dead_tail and !use_prebatched_z) {
                    const ssm_proj_z_phase = self.beginProfilePhase();
                    try self.dispatchDmmv(z_tensor, self.norm_buf, hidden_size, self.gate_buf, @intCast(d_inner), hidden_dim);
                    self.endProfilePhase(.ssm_proj_z, ssm_proj_z_phase);
                }
            } else if (can_fuse_qkv_z) {
                self.decode_cmd.computeBufferBarrier(self.norm_buf.handle, hidden_size);
                const ssm_proj_qkv_z_phase = self.beginProfilePhase();
                try self.dispatchDmmvQ8_0FusedPair(
                    wqkv_tensor,
                    z_tensor,
                    self.norm_buf,
                    hidden_size,
                    self.attn_out_buf,
                    qkv_bytes,
                    self.gate_buf,
                    z_bytes,
                    @intCast(conv_channels),
                    @intCast(d_inner),
                    hidden_dim,
                );
                self.endProfilePhase(.ssm_proj_qkv_z, ssm_proj_qkv_z_phase);
            } else {
                self.decode_cmd.computeBufferBarrier(self.norm_buf.handle, hidden_size);
                const ssm_proj_qkv_phase = self.beginProfilePhase();
                try self.dispatchDmmv(wqkv_tensor, self.norm_buf, hidden_size, self.attn_out_buf, @intCast(conv_channels), hidden_dim);
                self.endProfilePhase(.ssm_proj_qkv, ssm_proj_qkv_phase);
                if (!is_dead_tail) {
                    const ssm_proj_z_phase = self.beginProfilePhase();
                    try self.dispatchDmmv(z_tensor, self.norm_buf, hidden_size, self.gate_buf, @intCast(d_inner), hidden_dim);
                    self.endProfilePhase(.ssm_proj_z, ssm_proj_z_phase);
                }
            }
            // alpha + beta already produced by the fused shader.
        } else {
            if (use_prebatched_ssm_proj) {
                const tok_idx = self.partial_ssm_preproj_token_idx;
                const use_prebatched_qkv = self.partial_ssm_preproj_qkv != null and self.partial_ssm_preproj_qkv_stride > 0;
                const use_prebatched_z = self.partial_ssm_preproj_z != null and self.partial_ssm_preproj_z_stride > 0;

                self.decode_cmd.computeToTransferBarrier();
                if (use_prebatched_qkv) {
                    const qkv_src_off: vk.c.VkDeviceSize =
                        @as(vk.c.VkDeviceSize, tok_idx) * self.partial_ssm_preproj_qkv_stride;
                    if (qkv_src_off + qkv_bytes > self.partial_ssm_preproj_qkv_size) return error.BufferTooSmall;
                    const qkv_copy = vk.c.VkBufferCopy{
                        .srcOffset = qkv_src_off,
                        .dstOffset = 0,
                        .size = qkv_bytes,
                    };
                    const ssm_proj_qkv_phase = self.beginProfilePhase();
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.partial_ssm_preproj_qkv.?, self.attn_out_buf.handle, 1, &qkv_copy);
                    self.endProfilePhase(.ssm_proj_qkv, ssm_proj_qkv_phase);
                }
                if (use_prebatched_z and !is_dead_tail) {
                    const z_src_off: vk.c.VkDeviceSize =
                        @as(vk.c.VkDeviceSize, tok_idx) * self.partial_ssm_preproj_z_stride;
                    if (z_src_off + z_bytes > self.partial_ssm_preproj_z_size) return error.BufferTooSmall;
                    const z_copy = vk.c.VkBufferCopy{
                        .srcOffset = z_src_off,
                        .dstOffset = 0,
                        .size = z_bytes,
                    };
                    const ssm_proj_z_phase = self.beginProfilePhase();
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.partial_ssm_preproj_z.?, self.gate_buf.handle, 1, &z_copy);
                    self.endProfilePhase(.ssm_proj_z, ssm_proj_z_phase);
                }
                self.decode_cmd.transferToComputeBarrier();
                if (!use_prebatched_qkv) {
                    const ssm_proj_qkv_phase = self.beginProfilePhase();
                    try self.dispatchDmmv(wqkv_tensor, self.norm_buf, hidden_size, self.attn_out_buf, @intCast(conv_channels), hidden_dim);
                    self.endProfilePhase(.ssm_proj_qkv, ssm_proj_qkv_phase);
                }
                if (!is_dead_tail and !use_prebatched_z) {
                    const ssm_proj_z_phase = self.beginProfilePhase();
                    try self.dispatchDmmv(z_tensor, self.norm_buf, hidden_size, self.gate_buf, @intCast(d_inner), hidden_dim);
                    self.endProfilePhase(.ssm_proj_z, ssm_proj_z_phase);
                }
            } else if (can_fuse_qkv_z) {
                const ssm_proj_qkv_z_phase = self.beginProfilePhase();
                try self.dispatchDmmvQ8_0FusedPair(
                    wqkv_tensor,
                    z_tensor,
                    self.norm_buf,
                    hidden_size,
                    self.attn_out_buf,
                    qkv_bytes,
                    self.gate_buf,
                    z_bytes,
                    @intCast(conv_channels),
                    @intCast(d_inner),
                    hidden_dim,
                );
                self.endProfilePhase(.ssm_proj_qkv_z, ssm_proj_qkv_z_phase);
            } else {
                const ssm_proj_qkv_phase = self.beginProfilePhase();
                try self.dispatchDmmv(wqkv_tensor, self.norm_buf, hidden_size, self.attn_out_buf, @intCast(conv_channels), hidden_dim);
                self.endProfilePhase(.ssm_proj_qkv, ssm_proj_qkv_phase);
                // Skip z (gate) DMMV in dead-tail: gate_buf is only consumed by
                // gated_norm, which is also skipped below. wqkv/alpha/beta still
                // run because conv1d/delta_net update SSM state for future tokens.
                if (!is_dead_tail) {
                    const ssm_proj_z_phase = self.beginProfilePhase();
                    try self.dispatchDmmv(z_tensor, self.norm_buf, hidden_size, self.gate_buf, @intCast(d_inner), hidden_dim);
                    self.endProfilePhase(.ssm_proj_z, ssm_proj_z_phase);
                }
            }
            const ssm_proj_alpha_phase = self.beginProfilePhase();
            try self.dispatchDmmv(alpha_tensor, self.norm_buf, hidden_size, self.router_logits_buf, dt_rank, hidden_dim);
            self.endProfilePhase(.ssm_proj_alpha, ssm_proj_alpha_phase);
            const ssm_proj_beta_phase = self.beginProfilePhase();
            try self.dispatchDmmv(beta_tensor, self.norm_buf, hidden_size, self.down_buf, dt_rank, hidden_dim);
            self.endProfilePhase(.ssm_proj_beta, ssm_proj_beta_phase);
            // Note: tried fusing alpha+beta into one fused-gate-up dispatch
            // (commit considered but reverted) — no measurable change because
            // the four SSM proj DMMVs already overlap on RDNA4. See
            // loops/efforts/MULTI_HOUR_EFFORT_10_QWEN36_DECODE.md for the
            // bigger Qwen 3.6 levers (Q4_K × Q8_1 mmq, batched MoE prefill).
        }
        const ssm_prefill_validate_capture = self.use_qwen36_ssm_prefill_validate and
            self.prefill_active and
            !is_dead_tail and
            layer == self.dense_prefill_validate_layer and
            self.prefill_current_token_idx < self.dense_prefill_validate_max_tokens and
            self.ssm_prefill_validate_norm_ref != null and
            self.ssm_prefill_validate_qkv_ref != null and
            self.ssm_prefill_validate_z_ref != null and
            self.ssm_prefill_validate_alpha_ref != null and
            self.ssm_prefill_validate_beta_ref != null;
        const ssm_prefill_validate_delta_capture = ssm_prefill_validate_capture and
            self.ssm_prefill_validate_conv_ref != null and
            self.ssm_prefill_validate_delta_ref != null;
        const ssm_prefill_validate_output_capture = ssm_prefill_validate_delta_capture and
            self.ssm_prefill_validate_gnorm_ref != null and
            self.ssm_prefill_validate_pre_hidden_ref != null and
            self.ssm_prefill_validate_post_hidden_ref != null;
        if (ssm_prefill_validate_capture) {
            self.decode_cmd.computeAndTransferBarrier();
            const tok_idx = self.prefill_current_token_idx;
            const norm_dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * hidden_size;
            const qkv_dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * qkv_bytes;
            const z_dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * z_bytes;
            const ab_dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * ab_bytes;
            const r_norm = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = norm_dst_off, .size = hidden_size };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.norm_buf.handle, self.ssm_prefill_validate_norm_ref.?.handle, 1, &r_norm);
            const r_qkv = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = qkv_dst_off, .size = qkv_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.attn_out_buf.handle, self.ssm_prefill_validate_qkv_ref.?.handle, 1, &r_qkv);
            const r_z = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = z_dst_off, .size = z_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.gate_buf.handle, self.ssm_prefill_validate_z_ref.?.handle, 1, &r_z);
            const r_alpha = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = ab_dst_off, .size = ab_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.router_logits_buf.handle, self.ssm_prefill_validate_alpha_ref.?.handle, 1, &r_alpha);
            const r_beta = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = ab_dst_off, .size = ab_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.down_buf.handle, self.ssm_prefill_validate_beta_ref.?.handle, 1, &r_beta);
            self.ssm_prefill_validate_captured_tokens = @max(self.ssm_prefill_validate_captured_tokens, tok_idx + 1);
        } else {
            // The immediate next dispatch (ssm_conv1d) only reads attn_out_buf.
            // Writes to gate_buf/router_logits_buf/down_buf are picked up by the
            // subsequent global computeBarrier() before delta-net consumes them.
            self.decode_cmd.computeBufferBarrier(self.attn_out_buf.handle, qkv_bytes);
        }
        self.endProfilePhase(.ssm_proj, ssm_proj_phase);

        // --- GPU: conv1d + SiLU ---
        // Input: attn_out_buf (QKV projection), conv kernel from GPU tensor, persistent conv state
        // Output: swiglu_buf (reused as conv1d output)
        const conv_tensor = lt.ssm_conv1d orelse return;
        const conv_kernel_is_f16 = conv_tensor.info.type_ == .f16;
        const ssm_conv_phase = self.beginProfilePhase();
        {
            const pip = &(self.elementwise.pipeline_ssm_conv1d orelse return error.ShaderNotLoaded);
            // Capture the recording-time offset for THIS dispatch's push
            // constant; advance the host counter for the NEXT dispatch on
            // the same layer. Multi-deep CB pipelining works because each
            // dispatch's push constant is captured at record time.
            const cur_offset = self.ssm_conv_state_offsets[layer_idx];
            const d_conv_1: u32 = if (d_conv > 0) d_conv - 1 else 1;
            self.ssm_conv_state_offsets[layer_idx] = (cur_offset + 1) % d_conv_1;
            if (pip.uses_push_descriptors) {
                const push = SsmConv1dPush{
                    .conv_channels = conv_channels,
                    .d_conv = d_conv,
                    .kernel_is_f16 = if (conv_kernel_is_f16) 1 else 0,
                    .state_offset = cur_offset,
                };
                self.pushDispatch4(pip, std.mem.asBytes(&push), self.attn_out_buf.handle, qkv_bytes, conv_tensor.gpu_buffer.handle, conv_tensor.gpu_buffer.size, self.gpu_ssm_conv_states[layer_idx].handle, self.gpu_ssm_conv_states[layer_idx].size, self.swiglu_buf.handle, qkv_bytes, (conv_channels + 63) / 64, 1, 1);
            } else {
                const ds = try self.allocDescSet(pip.descriptor_set_layout);
                self.writeDescSet4(
                    ds,
                    self.attn_out_buf.handle,
                    qkv_bytes, // binding 0: current_input
                    conv_tensor.gpu_buffer.handle,
                    conv_tensor.gpu_buffer.size, // binding 1: conv kernel
                    self.gpu_ssm_conv_states[layer_idx].handle,
                    self.gpu_ssm_conv_states[layer_idx].size, // binding 2: state
                    self.swiglu_buf.handle,
                    qkv_bytes, // binding 3: output
                );
                try self.elementwise.recordSsmConv1d(&self.decode_cmd, ds, conv_channels, d_conv, conv_kernel_is_f16, cur_offset);
            }
        }
        // Narrow: delta_net only reads swiglu_buf (conv1d output), router_logits_buf (alpha,
        // from ssm_proj), and down_buf (beta, from ssm_proj). gate_buf is consumed later by
        // gnorm and that path has its own barrier. gpu_ssm_conv_states is only read by the
        // NEXT token's conv1d in a different command buffer (cross-CB sync via submission
        // ordering + cycle 5 pipeline waitForCompletion).
        // Effort-6 cycle 123 (A3b all-layer extension): when the validation
        // flag is on and we're in prefill, widen the barrier on EVERY SSM
        // layer so the conv1d/alpha/beta writes are visible to both the
        // delta_net compute read and the upcoming vkCmdCopyBuffer transfer
        // reads that capture the per-token slice. Cycle 97/101/104 only
        // captured at layer 0; this extension covers all SSM layers via
        // (layer × max_tokens + token) absolute offsets in the capture
        // buffers. Flag-off path keeps the original narrow compute→compute
        // barrier so non-validate prefills are unaffected.
        // Cycle 127: dropped use_a3b_production from the gate. With cycle
        // 125's broken post-loop dispatch removed, captures under
        // production-only flag would be wasted work. Captures only happen
        // under validate now; cycle 128's layer-major restructure will
        // re-introduce its own capture path if needed.
        const a3b_capture_this_layer = self.use_a3b_validate and
            self.prefill_active and
            self.a3b_alpha_capture != null and
            self.a3b_beta_capture != null and
            self.a3b_conv_out_capture != null and
            self.prefill_current_token_idx < self.a3b_capture_max_tokens;
        const stop_after_ssm_conv = self.partial_decode_stop_after_ssm_conv and
            self.prefill_active and
            !is_dead_tail and
            self.partial_decode_ssm_conv_out != null and
            self.partial_decode_ssm_z_out != null and
            self.partial_decode_ssm_alpha_out != null and
            self.partial_decode_ssm_beta_out != null;
        if (use_delta_normed_qk) {
            self.decode_cmd.computeBufferBarrier(self.swiglu_buf.handle, qkv_bytes);
            const pip = &(self.elementwise.pipeline_ssm_qk_norm orelse return error.ShaderNotLoaded);
            const push = SsmQkNormPush{
                .d_state = d_state,
                .n_group = n_group,
                .qk_dim = d_state * n_group,
            };
            self.pushDispatch1(pip, std.mem.asBytes(&push), self.swiglu_buf.handle, qkv_bytes, n_group, 1, 1);
        }
        if (a3b_capture_this_layer or ssm_prefill_validate_delta_capture or stop_after_ssm_conv) {
            self.decode_cmd.computeAndTransferBarrier();
        } else {
            const conv_to_delta_ranges = [_]CommandBuffer.BufferRange{
                .{ .buffer = self.swiglu_buf.handle, .size = qkv_bytes },
                .{ .buffer = self.router_logits_buf.handle, .size = ab_bytes },
                .{ .buffer = self.down_buf.handle, .size = ab_bytes },
            };
            self.decode_cmd.computeBuffersBarrier(&conv_to_delta_ranges);
        }
        self.endProfilePhase(.ssm_conv, ssm_conv_phase);

        // A3b all-layer capture: copy this (layer, token)'s (alpha, beta,
        // conv_out) slices into the per-(layer, token) strided slots in the
        // capture buffers. Layout: slot(layer, token) = (layer × max_tokens
        // + token) × per_data_bytes. The post-prefill batched ssm_delta_net
        // dispatches (in prefillBatch, one per SSM layer, after the
        // per-token loop drains) read each layer's slice via push.{conv,ab}
        // _stride_tok with n_tok=prompt_len. The non-SSM (attention) layers
        // also enter runSsmLayerGpu? No — they don't; this function is only
        // called on SSM layers. So the capture wastes only the (layer,
        // token) slots whose layer is an attention layer, which is OK.
        if (a3b_capture_this_layer) {
            const tok_idx = self.prefill_current_token_idx;
            const max_tokens = self.a3b_capture_max_tokens;
            const layer_stride_ab: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, max_tokens) * ab_bytes;
            const layer_stride_conv: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, max_tokens) * qkv_bytes;
            const layer_stride_z: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, max_tokens) * z_bytes;
            const ab_dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, layer) * layer_stride_ab + @as(vk.c.VkDeviceSize, tok_idx) * ab_bytes;
            const conv_dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, layer) * layer_stride_conv + @as(vk.c.VkDeviceSize, tok_idx) * qkv_bytes;
            const r_alpha = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = ab_dst_off, .size = ab_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.router_logits_buf.handle, self.a3b_alpha_capture.?.handle, 1, &r_alpha);
            const r_beta = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = ab_dst_off, .size = ab_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.down_buf.handle, self.a3b_beta_capture.?.handle, 1, &r_beta);
            const r_conv = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = conv_dst_off, .size = qkv_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.swiglu_buf.handle, self.a3b_conv_out_capture.?.handle, 1, &r_conv);
            // gate_buf capture (z-projection output): same per-(layer, token)
            // strided layout. Skipped on dead-tail tokens because gate_buf
            // isn't written then.
            if (self.a3b_gate_capture != null and !is_dead_tail) {
                const gate_dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, layer) * layer_stride_z + @as(vk.c.VkDeviceSize, tok_idx) * z_bytes;
                const r_gate = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = gate_dst_off, .size = z_bytes };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.gate_buf.handle, self.a3b_gate_capture.?.handle, 1, &r_gate);
            }
        }
        if (ssm_prefill_validate_delta_capture) {
            const tok_idx = self.prefill_current_token_idx;
            const conv_dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * qkv_bytes;
            const r_conv = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = conv_dst_off, .size = qkv_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.swiglu_buf.handle, self.ssm_prefill_validate_conv_ref.?.handle, 1, &r_conv);
        }
        if (stop_after_ssm_conv) {
            const conv_copy = vk.c.VkBufferCopy{
                .srcOffset = 0,
                .dstOffset = self.partial_decode_ssm_conv_out_offset,
                .size = qkv_bytes,
            };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.swiglu_buf.handle, self.partial_decode_ssm_conv_out.?, 1, &conv_copy);
            const z_copy = vk.c.VkBufferCopy{
                .srcOffset = 0,
                .dstOffset = self.partial_decode_ssm_z_out_offset,
                .size = z_bytes,
            };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.gate_buf.handle, self.partial_decode_ssm_z_out.?, 1, &z_copy);
            const alpha_copy = vk.c.VkBufferCopy{
                .srcOffset = 0,
                .dstOffset = self.partial_decode_ssm_alpha_out_offset,
                .size = ab_bytes,
            };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.router_logits_buf.handle, self.partial_decode_ssm_alpha_out.?, 1, &alpha_copy);
            const beta_copy = vk.c.VkBufferCopy{
                .srcOffset = 0,
                .dstOffset = self.partial_decode_ssm_beta_out_offset,
                .size = ab_bytes,
            };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.down_buf.handle, self.partial_decode_ssm_beta_out.?, 1, &beta_copy);
            return;
        }

        // --- GPU SSM diagnostic: readback conv1d output at layer 0 for comparison with CPU SSM_DBG ---
        if (layer == 0 and self.validation_diagnostics_enabled) {
            // Flush to read conv1d output
            {
                const bar = vk.c.VkMemoryBarrier{
                    .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = null,
                    .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
                };
                vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &bar, 0, null, 0, null);
                const rgn = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = @min(qkv_bytes, self.logits_staging.size) };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.swiglu_buf.handle, self.logits_staging.handle, 1, &rgn);
            }
            try self.decode_cmd.end();
            try self.decode_cmd.submitAndWait(self.instance.compute_queue);
            const ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
            var l2: f64 = 0;
            for (0..@min(conv_channels, 4096)) |i| l2 += @as(f64, ptr[i]) * @as(f64, ptr[i]);
            l2 = @sqrt(l2);
            log.info("GPU_SSM_DBG L0 conv1d_out[0..4]=[{d:.8},{d:.8},{d:.8},{d:.8}] L2={d:.6}", .{
                ptr[0], ptr[1], ptr[2], ptr[3], l2,
            });
            // Restart cmd buffer
            if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
            try self.decode_cmd.reset();
            try self.decode_cmd.begin();
        }

        // Cycle 127 (A3b production rollback): cycle 125 attempted a
        // "switch flip" by early-returning here when ZINC_A3B_PRODUCTION=1
        // and dispatching a single batched ssm_delta_net post-per-token-loop
        // for state recurrence. The output was incoherent because skipping
        // per-token gnorm + ssm_out leaves hidden_buf without the layer's
        // SSM contribution — which corrupts subsequent layers' attn input
        // AND the alpha/beta/conv_out captures themselves (since later
        // layers' ssm_proj reads the corrupted hidden_buf). The post-loop
        // batched dispatch then consumed corrupted captures, producing
        // wrong state. The pivot doc's "switch flip" framing
        // underestimates the dependency: the actual production wire-up
        // requires a layer-major restructure where for each SSM layer L,
        // (a) per-token attn + ssm_proj + conv1d + capture, then
        // (b) one batched delta_net dispatch (n_tok=N), then
        // (c) per-token gnorm + ssm_out reading from a3b_delta_out[L][i]
        //     + a3b_gate_capture[L][i] into per-token hidden_buf,
        // (d) per-token ffn/moe.
        // That requires per-(layer, token) hidden_buf storage and a
        // layer-major prefillBatch — infrastructure not yet built. The
        // capture buffers (a3b_*_capture) and validate path stay intact
        // for the layer-major attempt. The destructive skip is removed
        // so production flag becomes a no-op (currently). Cycle 128+
        // builds the layer-major restructure on this clean foundation.

        // --- GPU: delta-net state update ---
        // Input: conv1d output (swiglu_buf), alpha (router_logits_buf), beta (down_buf), ssm_a + dt_bias from tensors
        // Output: attn_out_buf (reused, now free after conv1d consumed it)
        const dt_bias_tensor = lt.ssm_dt_bias;
        const ssm_a_tensor = lt.ssm_a;
        // Use a dummy zero buffer for missing tensors (dt_bias or ssm_a)
        const dt_bias_buf = if (dt_bias_tensor) |t| t.gpu_buffer.handle else self.down_buf.handle;
        const dt_bias_size = if (dt_bias_tensor) |t| t.gpu_buffer.size else ab_bytes;
        const ssm_a_buf = if (ssm_a_tensor) |t| t.gpu_buffer.handle else self.down_buf.handle;
        const ssm_a_size = if (ssm_a_tensor) |t| t.gpu_buffer.size else ab_bytes;
        const ssm_delta_phase = self.beginProfilePhase();
        {
            const push = SsmDeltaNetPush{
                .d_inner = d_inner,
                .dt_rank = dt_rank,
                .head_v_dim = head_v_dim,
                .d_state = d_state,
                .n_group = n_group,
                .ssm_a_is_f16 = if (ssm_a_tensor) |t| (if (t.info.type_ == .f16) @as(u32, 1) else 0) else 0,
                .dt_bias_is_f16 = if (dt_bias_tensor) |t| (if (t.info.type_ == .f16) @as(u32, 1) else 0) else 0,
                .has_dt_bias = if (dt_bias_tensor != null) 1 else 0,
                .has_ssm_a = if (ssm_a_tensor != null) 1 else 0,
                // A3: per-token loop folded inside shader. Production
                // dispatches one token per call (n_tok=1); the strides
                // are populated for forward-compatibility with future
                // batched calls but unused at n_tok=1.
                .n_tok = 1,
                .conv_stride_tok = d_inner + 2 * n_group * d_state,
                .ab_stride_tok = dt_rank,
                .y_stride_tok = d_inner,
            };
            const pip = if (use_delta_normed_qk)
                &(self.elementwise.pipeline_ssm_delta_net_cols8_normed orelse return error.ShaderNotLoaded)
            else if (use_delta_cols8)
                &(self.elementwise.pipeline_ssm_delta_net_cols8 orelse return error.ShaderNotLoaded)
            else
                &(self.elementwise.pipeline_ssm_delta_net orelse return error.ShaderNotLoaded);
            if (pip.uses_push_descriptors) {
                const row_blocks = if (use_delta_normed_qk)
                    (head_v_dim + 7) / 8
                else if (use_delta_cols8)
                    (head_v_dim + 3) / 4
                else
                    head_v_dim;
                self.pushDispatch7(pip, std.mem.asBytes(&push), self.swiglu_buf.handle, qkv_bytes, dt_bias_buf, dt_bias_size, self.router_logits_buf.handle, ab_bytes, self.down_buf.handle, ab_bytes, ssm_a_buf, ssm_a_size, self.gpu_ssm_states[layer_idx].handle, self.gpu_ssm_states[layer_idx].size, self.attn_out_buf.handle, z_bytes, dt_rank, row_blocks, 1);
            } else {
                const ds = try self.allocDescSet(pip.descriptor_set_layout);
                self.writeDescSet7(
                    ds,
                    self.swiglu_buf.handle,
                    qkv_bytes, // binding 0: conv_out
                    dt_bias_buf,
                    dt_bias_size, // binding 1: dt_bias
                    self.router_logits_buf.handle,
                    ab_bytes, // binding 2: alpha
                    self.down_buf.handle,
                    ab_bytes, // binding 3: beta
                    ssm_a_buf,
                    ssm_a_size, // binding 4: ssm_a
                    self.gpu_ssm_states[layer_idx].handle,
                    self.gpu_ssm_states[layer_idx].size, // binding 5: state
                    self.attn_out_buf.handle,
                    z_bytes, // binding 6: output (d_inner floats)
                );
                if (use_delta_normed_qk) {
                    try self.elementwise.recordSsmDeltaNetCols8Normed(&self.decode_cmd, ds, push);
                } else if (use_delta_cols8) {
                    try self.elementwise.recordSsmDeltaNetCols8(&self.decode_cmd, ds, push);
                } else {
                    try self.elementwise.recordSsmDeltaNet(&self.decode_cmd, ds, push);
                }
            }
        }
        // Narrow: gated_norm only reads attn_out_buf (delta_net output, z_bytes) and
        // gate_buf (z_tensor DMMV from ssm_proj, z_bytes — synced here for the first
        // time since cycle 17 dropped it from the ssm_proj end-barrier). gpu_ssm_states
        // (delta_net RMW) is only read by the NEXT token's delta_net in a different
        // command buffer (cross-CB sync via submission ordering + cycle 5 pipelined
        // waitForCompletion). Follows cycle 21's multi-buffer pattern.
        // Cycle 123: extend per-token delta_out capture to ALL SSM layers.
        // Slot layout: (layer × max_tokens + token) × z_bytes. The
        // post-prefill batched dispatches diff against this per-(layer,
        // token) reference. Flag-off path takes the original narrow
        // compute→compute barrier so production prefills are unaffected.
        const a3b_capture_output = self.use_a3b_validate and
            self.prefill_active and
            self.a3b_per_token_delta_out != null and
            self.prefill_current_token_idx < self.a3b_capture_max_tokens;
        if (!is_dead_tail) {
            if (a3b_capture_output or ssm_prefill_validate_delta_capture) {
                self.decode_cmd.computeAndTransferBarrier();
            } else {
                const delta_to_gnorm_ranges = [_]CommandBuffer.BufferRange{
                    .{ .buffer = self.attn_out_buf.handle, .size = z_bytes },
                    .{ .buffer = self.gate_buf.handle, .size = z_bytes },
                };
                self.decode_cmd.computeBuffersBarrier(&delta_to_gnorm_ranges);
            }
        }
        if (a3b_capture_output) {
            const tok_idx = self.prefill_current_token_idx;
            const max_tokens = self.a3b_capture_max_tokens;
            const layer_stride_z: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, max_tokens) * z_bytes;
            const out_dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, layer) * layer_stride_z + @as(vk.c.VkDeviceSize, tok_idx) * z_bytes;
            const r_out = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = out_dst_off, .size = z_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.attn_out_buf.handle, self.a3b_per_token_delta_out.?.handle, 1, &r_out);
        }
        if (ssm_prefill_validate_delta_capture) {
            const tok_idx = self.prefill_current_token_idx;
            const out_dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * z_bytes;
            const r_out = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = out_dst_off, .size = z_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.attn_out_buf.handle, self.ssm_prefill_validate_delta_ref.?.handle, 1, &r_out);
        }
        self.endProfilePhase(.ssm_delta, ssm_delta_phase);

        // Dead-tail SSM exits here: gated_norm and ssm_out only feed
        // swiglu_buf and hidden_buf, both overwritten by the next token's
        // pass. Cross-CB visibility for conv/SSM state writes from above is
        // provided by queue submission ordering at the end of decodeStep.
        if (is_dead_tail) return;

        // --- GPU: gated norm ---
        // Input: delta_net output (attn_out_buf), z gate (gate_buf), norm weights from tensor
        // Output: swiglu_buf (reused, now free after delta_net consumed it)
        const norm_tensor = lt.ssm_norm;
        const norm_elems: u32 = if (norm_tensor) |t| @intCast(t.info.numElements()) else 0;
        const norm_per_head = norm_elems >= d_inner;
        const norm_buf_handle = if (norm_tensor) |t| t.gpu_buffer.handle else self.down_buf.handle;
        const norm_buf_size = if (norm_tensor) |t| t.gpu_buffer.size else ab_bytes;
        const stop_after_ssm_gnorm = self.partial_decode_stop_after_ssm_gnorm and
            self.prefill_active and
            self.partial_decode_ssm_gnorm_out != null;
        const ssm_gated_norm_phase = self.beginProfilePhase();
        const direct_ssm_gnorm_store = stop_after_ssm_gnorm and
            !ssm_prefill_validate_output_capture and
            self.qwen36DensePrefillSsmGnormDirectStoreEnabled();
        {
            const pip = &(self.elementwise.pipeline_ssm_gated_norm orelse return error.ShaderNotLoaded);
            const push = SsmGatedNormPush{
                .d_inner = d_inner,
                .dt_rank = dt_rank,
                .head_v_dim = head_v_dim,
                .d_state = d_state,
                .norm_per_head = if (norm_per_head) 1 else 0,
            };
            if (pip.uses_push_descriptors) {
                if (direct_ssm_gnorm_store) {
                    const infos = [4]vk.c.VkDescriptorBufferInfo{
                        .{ .buffer = self.attn_out_buf.handle, .offset = 0, .range = z_bytes },
                        .{ .buffer = self.gate_buf.handle, .offset = 0, .range = z_bytes },
                        .{ .buffer = norm_buf_handle, .offset = 0, .range = norm_buf_size },
                        .{ .buffer = self.partial_decode_ssm_gnorm_out.?, .offset = self.partial_decode_ssm_gnorm_out_offset, .range = z_bytes },
                    };
                    self.decode_cmd.pushDescAndDispatch(
                        pip,
                        self.instance.push_descriptor_fn,
                        infos[0..],
                        std.mem.asBytes(&push),
                        dt_rank,
                        1,
                        1,
                    );
                } else {
                    self.pushDispatch4(pip, std.mem.asBytes(&push), self.attn_out_buf.handle, z_bytes, self.gate_buf.handle, z_bytes, norm_buf_handle, norm_buf_size, self.swiglu_buf.handle, z_bytes, dt_rank, 1, 1);
                }
            } else {
                const ds = try self.allocDescSet(pip.descriptor_set_layout);
                self.writeDescSet4(
                    ds,
                    self.attn_out_buf.handle,
                    z_bytes, // binding 0: delta_net output
                    self.gate_buf.handle,
                    z_bytes, // binding 1: z_gate
                    norm_buf_handle,
                    norm_buf_size, // binding 2: norm weights
                    self.swiglu_buf.handle,
                    z_bytes, // binding 3: output
                );
                try self.elementwise.recordSsmGatedNorm(&self.decode_cmd, ds, push);
            }
        }
        if (ssm_prefill_validate_output_capture or (stop_after_ssm_gnorm and !direct_ssm_gnorm_store)) {
            const tok_idx = self.prefill_current_token_idx;
            const z_dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * z_bytes;
            const hidden_dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * hidden_size;
            self.decode_cmd.computeAndTransferBarrier();
            if (ssm_prefill_validate_output_capture) {
                const r_gnorm = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = z_dst_off, .size = z_bytes };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.swiglu_buf.handle, self.ssm_prefill_validate_gnorm_ref.?.handle, 1, &r_gnorm);
                const r_pre_hidden = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = hidden_dst_off, .size = hidden_size };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.hidden_buf.handle, self.ssm_prefill_validate_pre_hidden_ref.?.handle, 1, &r_pre_hidden);
            }
            if (stop_after_ssm_gnorm) {
                const gnorm_copy = vk.c.VkBufferCopy{
                    .srcOffset = 0,
                    .dstOffset = self.partial_decode_ssm_gnorm_out_offset,
                    .size = z_bytes,
                };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.swiglu_buf.handle, self.partial_decode_ssm_gnorm_out.?, 1, &gnorm_copy);
            }
        } else if (!direct_ssm_gnorm_store) {
            self.decode_cmd.computeBufferBarrier(self.swiglu_buf.handle, z_bytes);
        }
        self.endProfilePhase(.ssm_gated_norm, ssm_gated_norm_phase);
        if (stop_after_ssm_gnorm) return;

        // --- GPU: ssm_out DMMV + residual (fused: accumulate directly into hidden_buf) ---
        const ssm_out_tensor = lt.ssm_out orelse return;
        const ssm_out_phase = self.beginProfilePhase();
        try self.dispatchDmmvAcc(ssm_out_tensor, self.swiglu_buf, z_bytes, self.hidden_buf, hidden_dim, @intCast(d_inner));
        if (ssm_prefill_validate_output_capture) {
            const tok_idx = self.prefill_current_token_idx;
            const hidden_dst_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * hidden_size;
            self.decode_cmd.computeAndTransferBarrier();
            const r_post_hidden = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = hidden_dst_off, .size = hidden_size };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.hidden_buf.handle, self.ssm_prefill_validate_post_hidden_ref.?.handle, 1, &r_post_hidden);
        } else {
            self.decode_cmd.computeBufferBarrier(self.hidden_buf.handle, hidden_size);
        }
        self.endProfilePhase(.ssm_out, ssm_out_phase);
    }

    /// L2 normalize a vector in-place.
    fn l2Normalize(v: []f32) void {
        var sum_sq: f32 = 0;
        for (v) |x| sum_sq += x * x;
        const norm = @sqrt(sum_sq + 1e-12);
        if (norm > 0) {
            for (v) |*x| x.* /= norm;
        }
    }

    fn isQwen36DenseHybrid27B(self: *const InferenceEngine) bool {
        const cfg = self.model.config;
        return cfg.n_experts == 0 and
            cfg.ssm_d_inner > 0 and
            cfg.hidden_dim == 5120 and
            cfg.intermediate_dim == 17408 and
            cfg.n_layers > 4;
    }

    fn isAmdRdna(self: *const InferenceEngine) bool {
        return self.gpu_config.vendor == .amd_rdna3 or
            self.gpu_config.vendor == .amd_rdna4 or
            self.gpu_config.vendor == .amd_rdna4_apu;
    }

    fn qwen36DensePrefillSsmPreprojEnabled(self: *const InferenceEngine) bool {
        if (self.validation_diagnostics_enabled) return false;
        if (!self.isQwen36DenseHybrid27B()) return false;
        if (!self.isAmdRdna()) return false;
        const mode = getenv("ZINC_QWEN36_27B_SSM_PREFILL_PROJ") orelse return false;
        return std.mem.eql(u8, mode, "1") or
            std.mem.eql(u8, mode, "both") or
            std.mem.eql(u8, mode, "qkv") or
            std.mem.eql(u8, mode, "z");
    }

    fn qwen36DensePrefillPartialStoreEnabled(self: *const InferenceEngine) bool {
        // The RMS+hidden snapshot helper was measured negative on the 27B
        // Coding Review prefill prompt; keep the normal RMS + transfer-copy
        // handoff as the default path.
        _ = self;
        return false;
    }

    fn qwen36DensePrefillSsmGnormDirectStoreEnabled(self: *const InferenceEngine) bool {
        if (self.validation_diagnostics_enabled) return false;
        if (!self.isQwen36DenseHybrid27B()) return false;
        if (!self.isAmdRdna()) return false;
        return self.instance.push_descriptor_fn != null and
            self.elementwise.pipeline_ssm_gated_norm != null;
    }

    fn qwen36DensePrefillSsmBatchedDeltaEnabled(self: *const InferenceEngine, n_tokens: u32) bool {
        if (n_tokens < 16) return false;
        if (self.validation_diagnostics_enabled) return false;
        if (self.use_qwen36_dense_prefill_validate or self.use_qwen36_ssm_prefill_validate) return false;
        if (!self.isQwen36DenseHybrid27B()) return false;
        if (!self.isAmdRdna()) return false;
        if (self.instance.push_descriptor_fn == null) return false;
        if (self.use_ssm_delta_normed_qk) return false;
        if (self.elementwise.pipeline_ssm_delta_net == null and self.elementwise.pipeline_ssm_delta_net_cols8 == null) return false;
        if (self.elementwise.pipeline_ssm_gated_norm == null) return false;
        if (getenv("ZINC_QWEN36_27B_SSM_BATCHED_DELTA")) |mode| {
            return mode.len > 0 and !std.mem.eql(u8, mode, "0");
        }
        return true;
    }

    fn qwen36DensePrefillSsmLayerMajorProjEnabled(self: *const InferenceEngine, n_tokens: u32) bool {
        if (!self.qwen36DensePrefillSsmBatchedDeltaEnabled(n_tokens)) return false;
        if (self.elementwise.pipeline_ssm_conv1d_batched == null) return false;
        if (self.elementwise.pipeline_dmmv_f32_dual_batch == null) return false;
        return true;
    }

    fn partialSsmPreprojActiveFor(self: *const InferenceEngine, layer: u32) bool {
        if (self.partial_ssm_preproj_layer != layer) return false;
        const has_qkv = self.partial_ssm_preproj_qkv != null and self.partial_ssm_preproj_qkv_stride > 0;
        const has_z = self.partial_ssm_preproj_z != null and self.partial_ssm_preproj_z_stride > 0;
        return has_qkv or has_z;
    }

    fn qwen36DensePrefillPrefixLayers(self: *const InferenceEngine, prompt_len: usize) u32 {
        if (prompt_len < 2 or self.validation_diagnostics_enabled) return 0;
        if (self.use_qwen36_dense_prefill_validate or self.use_qwen36_ssm_prefill_validate) return 0;

        const mode = getenv("ZINC_QWEN36_27B_DENSE_PREFILL");
        if (mode != null and std.mem.eql(u8, mode.?, "0")) return 0;

        const cfg = self.model.config;
        const is_amd = self.gpu_config.vendor == .amd_rdna3 or
            self.gpu_config.vendor == .amd_rdna4 or
            self.gpu_config.vendor == .amd_rdna4_apu;
        if (!self.isQwen36DenseHybrid27B()) return 0;
        if (mode == null and !is_amd) return 0;

        // The deeper prefix only wins for tiny prompts where the fixed
        // layer-major setup cost is small; context prompts stay on the
        // one-layer default because L2+ regressed the coding-review matrix.
        var layers: u32 = if (mode == null and prompt_len <= 8) 8 else 1;
        if (mode) |raw| {
            if (!std.mem.eql(u8, raw, "1")) {
                layers = std.fmt.parseInt(u32, raw, 10) catch layers;
            }
        }
        if (getenv("ZINC_QWEN36_27B_DENSE_PREFILL_LAYERS")) |raw| {
            layers = std.fmt.parseInt(u32, raw, 10) catch layers;
        }
        if (layers == 0) return 0;
        return @min(layers, cfg.n_layers - 1);
    }

    fn qwen36DensePrefillTailPipelineEnabled(self: *const InferenceEngine, n_tokens: u32) bool {
        if (n_tokens < 2) return false;
        if (self.validation_diagnostics_enabled or self.profile_enabled) return false;
        if (self.instance.push_descriptor_fn == null) return false;
        if (getenv("ZINC_QWEN36_27B_PREFIX_TAIL_PIPELINE")) |mode| {
            return mode.len > 0 and !std.mem.eql(u8, mode, "0");
        }
        if (!self.isQwen36DenseHybrid27B()) return false;
        if (!self.isAmdRdna()) return false;
        return n_tokens >= 16;
    }

    const qwen36_dense_prefill_max_segments = 64;

    fn appendQwen36DensePrefillSegment(self: *const InferenceEngine, out: *[qwen36_dense_prefill_max_segments]u32, count: *usize, layer: u32, prefix_layers: u32) void {
        const cfg = self.model.config;
        if (count.* >= out.len) return;
        if (layer <= prefix_layers) return;
        if (layer + 1 >= cfg.n_layers) return;
        for (out[0..count.*]) |existing| {
            if (existing == layer) return;
        }
        out[count.*] = layer;
        count.* += 1;
    }

    fn qwen36DensePrefillSegmentLayers(self: *const InferenceEngine, prompt_len: usize, prefix_layers: u32, out: *[qwen36_dense_prefill_max_segments]u32) usize {
        if (prompt_len < 16 or self.validation_diagnostics_enabled) return 0;
        if (self.use_qwen36_dense_prefill_validate or self.use_qwen36_ssm_prefill_validate) return 0;
        if (!self.isQwen36DenseHybrid27B()) return 0;
        if (!self.isAmdRdna()) return 0;

        const cfg = self.model.config;
        if (prefix_layers + 1 >= cfg.n_layers) return 0;

        var count: usize = 0;
        if (getenv("ZINC_QWEN36_27B_DENSE_PREFILL_SEGMENT")) |raw| {
            if (raw.len == 0 or std.mem.eql(u8, raw, "0")) return 0;
            if (std.mem.eql(u8, raw, "1")) {
                const full_attn_interval = if (cfg.full_attn_interval > 0) cfg.full_attn_interval else 1;
                var candidate = prefix_layers;
                while (candidate + 1 < cfg.n_layers) : (candidate += 1) {
                    if (((candidate + 1) % full_attn_interval) == 0) break;
                }
                self.appendQwen36DensePrefillSegment(out, &count, candidate, prefix_layers);
                return count;
            }

            var it = std.mem.splitScalar(u8, raw, ',');
            while (it.next()) |part_raw| {
                const part = std.mem.trim(u8, part_raw, " \t\r\n");
                if (part.len == 0) continue;
                const parsed = std.fmt.parseInt(u32, part, 10) catch continue;
                self.appendQwen36DensePrefillSegment(out, &count, parsed, prefix_layers);
            }
            return count;
        }

        const full_attn_interval = if (cfg.full_attn_interval > 0) cfg.full_attn_interval else 1;
        if (full_attn_interval == 4 and cfg.n_layers > 52) {
            // Measured on the 27B Coding Review prefill: layers 4-62 keep
            // dense FFN work layer-major without repeating the rejected SSM
            // projection replay path. Layer 3 still adds setup overhead.
            var segment_layer: u32 = 4;
            while (segment_layer <= 62) : (segment_layer += 1) {
                self.appendQwen36DensePrefillSegment(out, &count, segment_layer, prefix_layers);
            }
            return count;
        }
        var segment_layer: u32 = full_attn_interval - 1;
        while (segment_layer + 1 < cfg.n_layers and count < out.len) : (segment_layer += full_attn_interval) {
            self.appendQwen36DensePrefillSegment(out, &count, segment_layer, prefix_layers);
        }
        return count;
    }

    fn prefillQwen36RunPartialTokenLoop(
        self: *InferenceEngine,
        state: *DecodeState,
        prompt_tokens: []const u32,
        base_token: u32,
        n_tokens: u32,
        hidden_size: vk.c.VkDeviceSize,
        start_layer: u32,
        end_layer: u32,
        scratch_hidden: Buffer,
        scratch_norm: ?Buffer,
        copy_hidden_out: bool,
        stop_after_ffn_norm: bool,
        advance_position: bool,
        allow_final_tail_last: bool,
        pipeline_enabled: bool,
    ) !void {
        if (n_tokens == 0) return;
        if (end_layer != 0 and start_layer >= end_layer) return;

        var primary_pending: bool = false;
        var alt_pending: bool = false;
        var tok_idx: u32 = 0;
        while (tok_idx < n_tokens) : (tok_idx += 1) {
            const collect_output = allow_final_tail_last and tok_idx + 1 == n_tokens;
            const pipeline_this = pipeline_enabled and !collect_output;
            if (pipeline_this) {
                std.mem.swap(CommandBuffer, &self.decode_cmd, &self.prefill_cmd_alt);
                std.mem.swap(bool, &primary_pending, &alt_pending);
                if (primary_pending) {
                    try self.decode_cmd.waitForCompletion();
                    primary_pending = false;
                }
                self.prefill_pipeline_mode = true;
            } else {
                if (alt_pending) {
                    try self.prefill_cmd_alt.waitForCompletion();
                    alt_pending = false;
                }
                if (primary_pending) {
                    try self.decode_cmd.waitForCompletion();
                    primary_pending = false;
                }
                self.prefill_pipeline_mode = false;
            }

            const hidden_offset: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * hidden_size;
            self.prefill_current_token_idx = tok_idx;
            state.position = base_token + tok_idx;
            self.partial_decode_start_layer = start_layer;
            self.partial_decode_end_layer = end_layer;
            self.partial_decode_hidden_in = scratch_hidden.handle;
            self.partial_decode_hidden_in_offset = hidden_offset;
            self.partial_decode_hidden_out = if (copy_hidden_out) scratch_hidden.handle else null;
            self.partial_decode_hidden_out_offset = hidden_offset;
            self.partial_decode_advance_position = advance_position;
            self.partial_decode_allow_final_tail = collect_output;
            self.partial_decode_stop_after_ffn_norm = stop_after_ffn_norm;
            if (stop_after_ffn_norm) {
                const norm_buf = scratch_norm orelse return error.BufferTooSmall;
                self.partial_decode_ffn_norm_out = norm_buf.handle;
                self.partial_decode_ffn_norm_out_offset = hidden_offset;
            } else {
                self.partial_decode_ffn_norm_out = null;
                self.partial_decode_ffn_norm_out_offset = 0;
            }

            try self.decodeStep(state, prompt_tokens[tok_idx], collect_output);
            if (pipeline_this) {
                primary_pending = true;
            }
        }

        self.prefill_pipeline_mode = false;
        if (alt_pending) {
            try self.prefill_cmd_alt.waitForCompletion();
        }
        if (primary_pending) {
            try self.decode_cmd.waitForCompletion();
        }
    }

    fn prefillQwen36RunSsmLayerToFfnNorm(
        self: *InferenceEngine,
        state: *DecodeState,
        prompt_tokens: []const u32,
        base_token: u32,
        n_tokens: u32,
        hidden_dim: u32,
        d_inner: u32,
        hidden_size: vk.c.VkDeviceSize,
        layer: u32,
        scratch_hidden: Buffer,
        scratch_gate: Buffer,
        scratch_up: Buffer,
        scratch_q: Buffer,
        scratch_k: Buffer,
        scratch_attn_out: Buffer,
        scratch_swiglu: Buffer,
        scratch_norm: Buffer,
        scratch_down: Buffer,
        pipeline_enabled: bool,
    ) !void {
        if (n_tokens == 0) return;
        const lt = self.layer_tensors[layer];
        const ssm_out_t = lt.ssm_out orelse return error.TensorNotFound;
        const ffn_norm_t = lt.ffn_norm orelse
            lt.post_attention_norm orelse return error.TensorNotFound;
        const z_bytes = @as(vk.c.VkDeviceSize, d_inner) * @sizeOf(f32);
        const cfg = self.model.config;
        const dt_rank = cfg.ssm_dt_rank;
        const head_v_dim: u32 = d_inner / dt_rank;
        const conv_channels: u32 = d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state;
        const qkv_bytes: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, conv_channels) * @sizeOf(f32);
        const ab_bytes: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, dt_rank) * @sizeOf(f32);
        const qkv_total_bytes: vk.c.VkDeviceSize = qkv_bytes * @as(vk.c.VkDeviceSize, n_tokens);
        const z_total_bytes: vk.c.VkDeviceSize = z_bytes * @as(vk.c.VkDeviceSize, n_tokens);
        const ab_total_bytes: vk.c.VkDeviceSize = ab_bytes * @as(vk.c.VkDeviceSize, n_tokens);

        const saved_stop_after_ssm_gnorm = self.partial_decode_stop_after_ssm_gnorm;
        const saved_ssm_gnorm_out = self.partial_decode_ssm_gnorm_out;
        const saved_ssm_gnorm_out_offset = self.partial_decode_ssm_gnorm_out_offset;
        const saved_stop_after_ssm_conv = self.partial_decode_stop_after_ssm_conv;
        const saved_ssm_conv_out = self.partial_decode_ssm_conv_out;
        const saved_ssm_conv_out_offset = self.partial_decode_ssm_conv_out_offset;
        const saved_ssm_z_out = self.partial_decode_ssm_z_out;
        const saved_ssm_z_out_offset = self.partial_decode_ssm_z_out_offset;
        const saved_ssm_alpha_out = self.partial_decode_ssm_alpha_out;
        const saved_ssm_alpha_out_offset = self.partial_decode_ssm_alpha_out_offset;
        const saved_ssm_beta_out = self.partial_decode_ssm_beta_out;
        const saved_ssm_beta_out_offset = self.partial_decode_ssm_beta_out_offset;
        defer {
            self.partial_decode_stop_after_ssm_gnorm = saved_stop_after_ssm_gnorm;
            self.partial_decode_ssm_gnorm_out = saved_ssm_gnorm_out;
            self.partial_decode_ssm_gnorm_out_offset = saved_ssm_gnorm_out_offset;
            self.partial_decode_stop_after_ssm_conv = saved_stop_after_ssm_conv;
            self.partial_decode_ssm_conv_out = saved_ssm_conv_out;
            self.partial_decode_ssm_conv_out_offset = saved_ssm_conv_out_offset;
            self.partial_decode_ssm_z_out = saved_ssm_z_out;
            self.partial_decode_ssm_z_out_offset = saved_ssm_z_out_offset;
            self.partial_decode_ssm_alpha_out = saved_ssm_alpha_out;
            self.partial_decode_ssm_alpha_out_offset = saved_ssm_alpha_out_offset;
            self.partial_decode_ssm_beta_out = saved_ssm_beta_out;
            self.partial_decode_ssm_beta_out_offset = saved_ssm_beta_out_offset;
        }

        const use_batched_delta = self.qwen36DensePrefillSsmBatchedDeltaEnabled(n_tokens) and
            dt_rank > 0 and
            head_v_dim > 0 and
            qkv_total_bytes <= scratch_gate.size and
            z_total_bytes <= scratch_up.size and
            ab_total_bytes <= scratch_q.size and
            ab_total_bytes <= scratch_k.size and
            z_total_bytes <= scratch_attn_out.size and
            z_total_bytes <= scratch_swiglu.size and
            @as(vk.c.VkDeviceSize, n_tokens) * @as(vk.c.VkDeviceSize, hidden_dim) * @sizeOf(f32) <= scratch_down.size;

        const use_layer_major_ssm_proj = use_batched_delta and
            self.qwen36DensePrefillSsmLayerMajorProjEnabled(n_tokens) and
            qkv_total_bytes <= scratch_gate.size and
            z_total_bytes <= scratch_up.size and
            ab_total_bytes <= scratch_q.size and
            ab_total_bytes <= scratch_k.size and
            lt.attn_norm != null and
            lt.attn_qkv != null and
            lt.attn_gate != null and
            lt.ssm_alpha != null and
            lt.ssm_beta != null and
            lt.ssm_conv1d != null and
            lt.ssm_alpha.?.info.type_ == .f32 and
            lt.ssm_beta.?.info.type_ == .f32;

        if (use_layer_major_ssm_proj) {
            const attn_norm_t = lt.attn_norm.?;
            const wqkv_t = lt.attn_qkv.?;
            const z_t = lt.attn_gate.?;
            const alpha_t = lt.ssm_alpha.?;
            const beta_t = lt.ssm_beta.?;
            const conv_t = lt.ssm_conv1d.?;
            const hidden_total_bytes: vk.c.VkDeviceSize =
                @as(vk.c.VkDeviceSize, n_tokens) * @as(vk.c.VkDeviceSize, hidden_dim) * @sizeOf(f32);

            if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
            try self.decode_cmd.reset();
            try self.decode_cmd.beginOneTime();
            self.resetTimestamps();
            _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
            self.decode_cmd.transferToComputeBarrier();

            const ssm_phase = self.beginProfilePhase();
            const ssm_proj_phase = self.beginProfilePhase();
            const ssm_proj_norm_ab_phase = self.beginProfilePhase();
            try self.dispatchRmsNorm(
                scratch_hidden.handle,
                scratch_hidden.size,
                attn_norm_t.gpu_buffer.handle,
                attn_norm_t.gpu_buffer.size,
                scratch_norm.handle,
                scratch_norm.size,
                hidden_dim,
                n_tokens,
                self.model.config.rms_norm_eps,
            );
            self.decode_cmd.computeBufferBarrier(scratch_norm.handle, hidden_total_bytes);
            try self.dispatchF32DualBatched(
                alpha_t,
                beta_t,
                scratch_norm,
                scratch_q,
                scratch_k,
                dt_rank,
                hidden_dim,
                n_tokens,
            );
            self.endProfilePhase(.ssm_proj_norm_ab, ssm_proj_norm_ab_phase);

            const ssm_proj_qkv_phase = self.beginProfilePhase();
            try self.dispatchProjectionBatched(wqkv_t, scratch_norm, scratch_gate, conv_channels, hidden_dim, n_tokens);
            self.endProfilePhase(.ssm_proj_qkv, ssm_proj_qkv_phase);

            const ssm_proj_z_phase = self.beginProfilePhase();
            try self.dispatchProjectionBatched(z_t, scratch_norm, scratch_up, d_inner, hidden_dim, n_tokens);
            self.endProfilePhase(.ssm_proj_z, ssm_proj_z_phase);
            self.endProfilePhase(.ssm_proj, ssm_proj_phase);

            const ssm_conv_phase = self.beginProfilePhase();
            self.decode_cmd.computeBufferBarrier(scratch_gate.handle, qkv_total_bytes);
            const d_conv_1: u32 = if (cfg.ssm_d_conv > 1) cfg.ssm_d_conv - 1 else 1;
            const layer_idx_usize: usize = @intCast(layer);
            const cur_offset = self.ssm_conv_state_offsets[layer_idx_usize];
            self.ssm_conv_state_offsets[layer_idx_usize] =
                (cur_offset + (n_tokens % d_conv_1)) % d_conv_1;
            try self.dispatchSsmConv1dBatchedInPlace(
                scratch_gate,
                qkv_total_bytes,
                conv_t,
                self.gpu_ssm_conv_states[layer_idx_usize],
                conv_channels,
                cfg.ssm_d_conv,
                cur_offset,
                n_tokens,
            );
            self.endProfilePhase(.ssm_conv, ssm_conv_phase);

            const delta_inputs = [_]CommandBuffer.BufferRange{
                .{ .buffer = scratch_gate.handle, .size = qkv_total_bytes },
                .{ .buffer = scratch_q.handle, .size = ab_total_bytes },
                .{ .buffer = scratch_k.handle, .size = ab_total_bytes },
            };
            self.decode_cmd.computeBuffersBarrier(&delta_inputs);

            const ssm_delta_phase = self.beginProfilePhase();
            const dt_bias_t = lt.ssm_dt_bias;
            const ssm_a_t = lt.ssm_a;
            const dt_bias_buf = if (dt_bias_t) |t| t.gpu_buffer.handle else self.down_buf.handle;
            const dt_bias_size = if (dt_bias_t) |t| t.gpu_buffer.size else ab_bytes;
            const ssm_a_buf = if (ssm_a_t) |t| t.gpu_buffer.handle else self.down_buf.handle;
            const ssm_a_size = if (ssm_a_t) |t| t.gpu_buffer.size else ab_bytes;
            const use_delta_cols8 = self.use_ssm_delta_cols8 and
                !self.use_ssm_delta_normed_qk and
                head_v_dim == 128 and
                cfg.ssm_d_state == 128 and
                self.elementwise.pipeline_ssm_delta_net_cols8 != null;
            const delta_pip = if (use_delta_cols8)
                &(self.elementwise.pipeline_ssm_delta_net_cols8.?)
            else
                &(self.elementwise.pipeline_ssm_delta_net orelse return error.ShaderNotLoaded);
            const delta_push = SsmDeltaNetPush{
                .d_inner = d_inner,
                .dt_rank = dt_rank,
                .head_v_dim = head_v_dim,
                .d_state = cfg.ssm_d_state,
                .n_group = cfg.ssm_n_group,
                .ssm_a_is_f16 = if (ssm_a_t) |t| (if (t.info.type_ == .f16) @as(u32, 1) else 0) else 0,
                .dt_bias_is_f16 = if (dt_bias_t) |t| (if (t.info.type_ == .f16) @as(u32, 1) else 0) else 0,
                .has_dt_bias = if (dt_bias_t != null) 1 else 0,
                .has_ssm_a = if (ssm_a_t != null) 1 else 0,
                .n_tok = n_tokens,
                .conv_stride_tok = conv_channels,
                .ab_stride_tok = dt_rank,
                .y_stride_tok = d_inner,
            };
            const row_blocks = if (use_delta_cols8) (head_v_dim + 3) / 4 else head_v_dim;
            self.pushDispatch7(
                delta_pip,
                std.mem.asBytes(&delta_push),
                scratch_gate.handle,
                scratch_gate.size,
                dt_bias_buf,
                dt_bias_size,
                scratch_q.handle,
                scratch_q.size,
                scratch_k.handle,
                scratch_k.size,
                ssm_a_buf,
                ssm_a_size,
                self.gpu_ssm_states[layer_idx_usize].handle,
                self.gpu_ssm_states[layer_idx_usize].size,
                scratch_attn_out.handle,
                scratch_attn_out.size,
                dt_rank,
                row_blocks,
                1,
            );
            self.endProfilePhase(.ssm_delta, ssm_delta_phase);

            const ssm_gnorm_phase = self.beginProfilePhase();
            const gnorm_inputs = [_]CommandBuffer.BufferRange{
                .{ .buffer = scratch_attn_out.handle, .size = z_total_bytes },
                .{ .buffer = scratch_up.handle, .size = z_total_bytes },
            };
            self.decode_cmd.computeBuffersBarrier(&gnorm_inputs);
            const norm_tensor = lt.ssm_norm;
            const norm_elems: u32 = if (norm_tensor) |t| @intCast(t.info.numElements()) else 0;
            const norm_per_head = norm_elems >= d_inner;
            const norm_buf_handle = if (norm_tensor) |t| t.gpu_buffer.handle else self.down_buf.handle;
            const norm_buf_size = if (norm_tensor) |t| t.gpu_buffer.size else ab_total_bytes;
            const gnorm_pip = &(self.elementwise.pipeline_ssm_gated_norm orelse return error.ShaderNotLoaded);
            const gnorm_push = SsmGatedNormPush{
                .d_inner = d_inner,
                .dt_rank = dt_rank,
                .head_v_dim = head_v_dim,
                .d_state = cfg.ssm_d_state,
                .norm_per_head = if (norm_per_head) 1 else 0,
            };
            var tok_idx: u32 = 0;
            while (tok_idx < n_tokens) : (tok_idx += 1) {
                const z_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * z_bytes;
                const infos = [4]vk.c.VkDescriptorBufferInfo{
                    .{ .buffer = scratch_attn_out.handle, .offset = z_off, .range = z_bytes },
                    .{ .buffer = scratch_up.handle, .offset = z_off, .range = z_bytes },
                    .{ .buffer = norm_buf_handle, .offset = 0, .range = norm_buf_size },
                    .{ .buffer = scratch_swiglu.handle, .offset = z_off, .range = z_bytes },
                };
                self.decode_cmd.pushDescAndDispatch(
                    gnorm_pip,
                    self.instance.push_descriptor_fn,
                    infos[0..],
                    std.mem.asBytes(&gnorm_push),
                    dt_rank,
                    1,
                    1,
                );
            }
            self.endProfilePhase(.ssm_gated_norm, ssm_gnorm_phase);

            self.decode_cmd.computeBufferBarrier(scratch_swiglu.handle, z_total_bytes);
            const ssm_out_phase = self.beginProfilePhase();
            const use_batched_q5k_ssm_out = ssm_out_t.info.type_ == .q5_k and
                self.use_q4k_batch_kpar and
                self.dmmv.pipeline_q5k != null and
                n_tokens >= 2;
            if (use_batched_q5k_ssm_out) {
                try self.dispatchProjectionBatched(ssm_out_t, scratch_swiglu, scratch_down, hidden_dim, d_inner, n_tokens);
                self.decode_cmd.computeBarrier();
                try self.dispatchResidualRmsNorm(
                    scratch_hidden.handle,
                    scratch_hidden.size,
                    scratch_down.handle,
                    scratch_down.size,
                    scratch_norm.handle,
                    scratch_norm.size,
                    ffn_norm_t.gpu_buffer.handle,
                    ffn_norm_t.gpu_buffer.size,
                    hidden_dim,
                    n_tokens,
                    self.model.config.rms_norm_eps,
                    1.0,
                );
            } else {
                tok_idx = 0;
                while (tok_idx < n_tokens) : (tok_idx += 1) {
                    const x_offset = tok_idx * d_inner * @sizeOf(f32);
                    const y_offset = tok_idx * hidden_dim * @sizeOf(f32);
                    try self.dispatchDmmvInner(
                        ssm_out_t,
                        scratch_swiglu,
                        scratch_swiglu.size,
                        scratch_hidden,
                        hidden_dim,
                        d_inner,
                        0,
                        x_offset,
                        y_offset,
                        1,
                    );
                }
            }
            self.endProfilePhase(.ssm_out, ssm_out_phase);
            self.endProfilePhase(.ssm, ssm_phase);
            if (!use_batched_q5k_ssm_out) {
                self.decode_cmd.computeBarrier();
                try self.dispatchRmsNorm(
                    scratch_hidden.handle,
                    scratch_hidden.size,
                    ffn_norm_t.gpu_buffer.handle,
                    ffn_norm_t.gpu_buffer.size,
                    scratch_norm.handle,
                    scratch_norm.size,
                    hidden_dim,
                    n_tokens,
                    self.model.config.rms_norm_eps,
                );
            }
            self.decode_cmd.computeToTransferBarrier();
            _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
            try self.decode_cmd.end();
            try self.decode_cmd.submitAndWait(self.instance.compute_queue);
            self.recordProfilingSample();
            state.position = base_token + n_tokens - 1;
            return;
        }

        if (use_batched_delta) {
            var primary_pending: bool = false;
            var alt_pending: bool = false;
            var tok_idx: u32 = 0;
            while (tok_idx < n_tokens) : (tok_idx += 1) {
                const pipeline_this = pipeline_enabled;
                if (pipeline_this) {
                    std.mem.swap(CommandBuffer, &self.decode_cmd, &self.prefill_cmd_alt);
                    std.mem.swap(bool, &primary_pending, &alt_pending);
                    if (primary_pending) {
                        try self.decode_cmd.waitForCompletion();
                        primary_pending = false;
                    }
                    self.prefill_pipeline_mode = true;
                } else {
                    if (alt_pending) {
                        try self.prefill_cmd_alt.waitForCompletion();
                        alt_pending = false;
                    }
                    if (primary_pending) {
                        try self.decode_cmd.waitForCompletion();
                        primary_pending = false;
                    }
                    self.prefill_pipeline_mode = false;
                }

                const hidden_offset: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * hidden_size;
                self.prefill_current_token_idx = tok_idx;
                state.position = base_token + tok_idx;
                self.partial_decode_start_layer = layer;
                self.partial_decode_end_layer = layer + 1;
                self.partial_decode_hidden_in = scratch_hidden.handle;
                self.partial_decode_hidden_in_offset = hidden_offset;
                self.partial_decode_hidden_out = null;
                self.partial_decode_hidden_out_offset = 0;
                self.partial_decode_advance_position = false;
                self.partial_decode_allow_final_tail = false;
                self.partial_decode_stop_after_ffn_norm = false;
                self.partial_decode_ffn_norm_out = null;
                self.partial_decode_ffn_norm_out_offset = 0;
                self.partial_decode_stop_after_ssm_gnorm = false;
                self.partial_decode_ssm_gnorm_out = null;
                self.partial_decode_ssm_gnorm_out_offset = 0;
                self.partial_decode_stop_after_ssm_conv = true;
                self.partial_decode_ssm_conv_out = scratch_gate.handle;
                self.partial_decode_ssm_conv_out_offset = @as(vk.c.VkDeviceSize, tok_idx) * qkv_bytes;
                self.partial_decode_ssm_z_out = scratch_up.handle;
                self.partial_decode_ssm_z_out_offset = @as(vk.c.VkDeviceSize, tok_idx) * z_bytes;
                self.partial_decode_ssm_alpha_out = scratch_q.handle;
                self.partial_decode_ssm_alpha_out_offset = @as(vk.c.VkDeviceSize, tok_idx) * ab_bytes;
                self.partial_decode_ssm_beta_out = scratch_k.handle;
                self.partial_decode_ssm_beta_out_offset = @as(vk.c.VkDeviceSize, tok_idx) * ab_bytes;

                try self.decodeStep(state, prompt_tokens[tok_idx], false);
                if (pipeline_this) {
                    primary_pending = true;
                }
            }

            self.prefill_pipeline_mode = false;
            if (alt_pending) {
                try self.prefill_cmd_alt.waitForCompletion();
            }
            if (primary_pending) {
                try self.decode_cmd.waitForCompletion();
            }

            self.partial_decode_stop_after_ssm_conv = false;
            self.partial_decode_ssm_conv_out = null;
            self.partial_decode_ssm_z_out = null;
            self.partial_decode_ssm_alpha_out = null;
            self.partial_decode_ssm_beta_out = null;

            if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
            try self.decode_cmd.reset();
            try self.decode_cmd.beginOneTime();
            self.resetTimestamps();
            _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
            self.decode_cmd.transferToComputeBarrier();

            const ssm_phase = self.beginProfilePhase();
            const ssm_delta_phase = self.beginProfilePhase();
            const dt_bias_t = lt.ssm_dt_bias;
            const ssm_a_t = lt.ssm_a;
            const dt_bias_buf = if (dt_bias_t) |t| t.gpu_buffer.handle else self.down_buf.handle;
            const dt_bias_size = if (dt_bias_t) |t| t.gpu_buffer.size else ab_bytes;
            const ssm_a_buf = if (ssm_a_t) |t| t.gpu_buffer.handle else self.down_buf.handle;
            const ssm_a_size = if (ssm_a_t) |t| t.gpu_buffer.size else ab_bytes;
            const use_delta_cols8 = self.use_ssm_delta_cols8 and
                !self.use_ssm_delta_normed_qk and
                head_v_dim == 128 and
                cfg.ssm_d_state == 128 and
                self.elementwise.pipeline_ssm_delta_net_cols8 != null;
            const delta_pip = if (use_delta_cols8)
                &(self.elementwise.pipeline_ssm_delta_net_cols8.?)
            else
                &(self.elementwise.pipeline_ssm_delta_net orelse return error.ShaderNotLoaded);
            const delta_push = SsmDeltaNetPush{
                .d_inner = d_inner,
                .dt_rank = dt_rank,
                .head_v_dim = head_v_dim,
                .d_state = cfg.ssm_d_state,
                .n_group = cfg.ssm_n_group,
                .ssm_a_is_f16 = if (ssm_a_t) |t| (if (t.info.type_ == .f16) @as(u32, 1) else 0) else 0,
                .dt_bias_is_f16 = if (dt_bias_t) |t| (if (t.info.type_ == .f16) @as(u32, 1) else 0) else 0,
                .has_dt_bias = if (dt_bias_t != null) 1 else 0,
                .has_ssm_a = if (ssm_a_t != null) 1 else 0,
                .n_tok = n_tokens,
                .conv_stride_tok = conv_channels,
                .ab_stride_tok = dt_rank,
                .y_stride_tok = d_inner,
            };
            const row_blocks = if (use_delta_cols8) (head_v_dim + 3) / 4 else head_v_dim;
            self.pushDispatch7(
                delta_pip,
                std.mem.asBytes(&delta_push),
                scratch_gate.handle,
                scratch_gate.size,
                dt_bias_buf,
                dt_bias_size,
                scratch_q.handle,
                scratch_q.size,
                scratch_k.handle,
                scratch_k.size,
                ssm_a_buf,
                ssm_a_size,
                self.gpu_ssm_states[@intCast(layer)].handle,
                self.gpu_ssm_states[@intCast(layer)].size,
                scratch_attn_out.handle,
                scratch_attn_out.size,
                dt_rank,
                row_blocks,
                1,
            );
            self.endProfilePhase(.ssm_delta, ssm_delta_phase);

            const ssm_gnorm_phase = self.beginProfilePhase();
            self.decode_cmd.computeBufferBarrier(scratch_attn_out.handle, z_total_bytes);
            const norm_tensor = lt.ssm_norm;
            const norm_elems: u32 = if (norm_tensor) |t| @intCast(t.info.numElements()) else 0;
            const norm_per_head = norm_elems >= d_inner;
            const norm_buf_handle = if (norm_tensor) |t| t.gpu_buffer.handle else self.down_buf.handle;
            const norm_buf_size = if (norm_tensor) |t| t.gpu_buffer.size else ab_total_bytes;
            const gnorm_pip = &(self.elementwise.pipeline_ssm_gated_norm orelse return error.ShaderNotLoaded);
            const gnorm_push = SsmGatedNormPush{
                .d_inner = d_inner,
                .dt_rank = dt_rank,
                .head_v_dim = head_v_dim,
                .d_state = cfg.ssm_d_state,
                .norm_per_head = if (norm_per_head) 1 else 0,
            };
            tok_idx = 0;
            while (tok_idx < n_tokens) : (tok_idx += 1) {
                const z_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * z_bytes;
                const infos = [4]vk.c.VkDescriptorBufferInfo{
                    .{ .buffer = scratch_attn_out.handle, .offset = z_off, .range = z_bytes },
                    .{ .buffer = scratch_up.handle, .offset = z_off, .range = z_bytes },
                    .{ .buffer = norm_buf_handle, .offset = 0, .range = norm_buf_size },
                    .{ .buffer = scratch_swiglu.handle, .offset = z_off, .range = z_bytes },
                };
                self.decode_cmd.pushDescAndDispatch(
                    gnorm_pip,
                    self.instance.push_descriptor_fn,
                    infos[0..],
                    std.mem.asBytes(&gnorm_push),
                    dt_rank,
                    1,
                    1,
                );
            }
            self.endProfilePhase(.ssm_gated_norm, ssm_gnorm_phase);

            self.decode_cmd.computeBufferBarrier(scratch_swiglu.handle, z_total_bytes);
            const ssm_out_phase = self.beginProfilePhase();
            const use_batched_q5k_ssm_out = ssm_out_t.info.type_ == .q5_k and
                self.use_q4k_batch_kpar and
                self.dmmv.pipeline_q5k != null and
                n_tokens >= 2;
            if (use_batched_q5k_ssm_out) {
                try self.dispatchProjectionBatched(ssm_out_t, scratch_swiglu, scratch_down, hidden_dim, d_inner, n_tokens);
                self.decode_cmd.computeBarrier();
                try self.dispatchResidualRmsNorm(
                    scratch_hidden.handle,
                    scratch_hidden.size,
                    scratch_down.handle,
                    scratch_down.size,
                    scratch_norm.handle,
                    scratch_norm.size,
                    ffn_norm_t.gpu_buffer.handle,
                    ffn_norm_t.gpu_buffer.size,
                    hidden_dim,
                    n_tokens,
                    self.model.config.rms_norm_eps,
                    1.0,
                );
            } else {
                tok_idx = 0;
                while (tok_idx < n_tokens) : (tok_idx += 1) {
                    const x_offset = tok_idx * d_inner * @sizeOf(f32);
                    const y_offset = tok_idx * hidden_dim * @sizeOf(f32);
                    try self.dispatchDmmvInner(
                        ssm_out_t,
                        scratch_swiglu,
                        scratch_swiglu.size,
                        scratch_hidden,
                        hidden_dim,
                        d_inner,
                        0,
                        x_offset,
                        y_offset,
                        1,
                    );
                }
            }
            self.endProfilePhase(.ssm_out, ssm_out_phase);
            self.endProfilePhase(.ssm, ssm_phase);
            if (!use_batched_q5k_ssm_out) {
                self.decode_cmd.computeBarrier();
                try self.dispatchRmsNorm(
                    scratch_hidden.handle,
                    scratch_hidden.size,
                    ffn_norm_t.gpu_buffer.handle,
                    ffn_norm_t.gpu_buffer.size,
                    scratch_norm.handle,
                    scratch_norm.size,
                    hidden_dim,
                    n_tokens,
                    self.model.config.rms_norm_eps,
                );
            }
            self.decode_cmd.computeToTransferBarrier();
            _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
            try self.decode_cmd.end();
            try self.decode_cmd.submitAndWait(self.instance.compute_queue);
            self.recordProfilingSample();
            return;
        }

        var primary_pending: bool = false;
        var alt_pending: bool = false;
        var tok_idx: u32 = 0;
        while (tok_idx < n_tokens) : (tok_idx += 1) {
            const pipeline_this = pipeline_enabled;
            if (pipeline_this) {
                std.mem.swap(CommandBuffer, &self.decode_cmd, &self.prefill_cmd_alt);
                std.mem.swap(bool, &primary_pending, &alt_pending);
                if (primary_pending) {
                    try self.decode_cmd.waitForCompletion();
                    primary_pending = false;
                }
                self.prefill_pipeline_mode = true;
            } else {
                if (alt_pending) {
                    try self.prefill_cmd_alt.waitForCompletion();
                    alt_pending = false;
                }
                if (primary_pending) {
                    try self.decode_cmd.waitForCompletion();
                    primary_pending = false;
                }
                self.prefill_pipeline_mode = false;
            }

            const hidden_offset: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * hidden_size;
            self.prefill_current_token_idx = tok_idx;
            state.position = base_token + tok_idx;
            self.partial_decode_start_layer = layer;
            self.partial_decode_end_layer = layer + 1;
            self.partial_decode_hidden_in = scratch_hidden.handle;
            self.partial_decode_hidden_in_offset = hidden_offset;
            self.partial_decode_hidden_out = null;
            self.partial_decode_hidden_out_offset = 0;
            self.partial_decode_advance_position = false;
            self.partial_decode_allow_final_tail = false;
            self.partial_decode_stop_after_ffn_norm = false;
            self.partial_decode_ffn_norm_out = null;
            self.partial_decode_ffn_norm_out_offset = 0;
            self.partial_decode_stop_after_ssm_gnorm = true;
            self.partial_decode_ssm_gnorm_out = scratch_swiglu.handle;
            self.partial_decode_ssm_gnorm_out_offset = @as(vk.c.VkDeviceSize, tok_idx) * z_bytes;

            try self.decodeStep(state, prompt_tokens[tok_idx], false);
            if (pipeline_this) {
                primary_pending = true;
            }
        }

        self.prefill_pipeline_mode = false;
        if (alt_pending) {
            try self.prefill_cmd_alt.waitForCompletion();
        }
        if (primary_pending) {
            try self.decode_cmd.waitForCompletion();
        }

        self.partial_decode_stop_after_ssm_gnorm = false;
        self.partial_decode_ssm_gnorm_out = null;
        self.partial_decode_ssm_gnorm_out_offset = 0;

        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.beginOneTime();
        self.resetTimestamps();
        _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
        if (self.qwen36DensePrefillSsmGnormDirectStoreEnabled()) {
            self.decode_cmd.computeBufferBarrier(
                scratch_swiglu.handle,
                @as(vk.c.VkDeviceSize, n_tokens) * z_bytes,
            );
        } else {
            self.decode_cmd.transferToComputeBarrier();
        }

        const ssm_phase = self.beginProfilePhase();
        const ssm_out_phase = self.beginProfilePhase();
        const use_batched_q5k_ssm_out = ssm_out_t.info.type_ == .q5_k and
            self.use_q4k_batch_kpar and
            self.dmmv.pipeline_q5k != null and
            n_tokens >= 2;
        if (use_batched_q5k_ssm_out) {
            try self.dispatchProjectionBatched(ssm_out_t, scratch_swiglu, scratch_down, hidden_dim, d_inner, n_tokens);
            self.decode_cmd.computeBarrier();
            try self.dispatchResidualRmsNorm(
                scratch_hidden.handle,
                scratch_hidden.size,
                scratch_down.handle,
                scratch_down.size,
                scratch_norm.handle,
                scratch_norm.size,
                ffn_norm_t.gpu_buffer.handle,
                ffn_norm_t.gpu_buffer.size,
                hidden_dim,
                n_tokens,
                self.model.config.rms_norm_eps,
                1.0,
            );
        } else {
            var out_tok: u32 = 0;
            while (out_tok < n_tokens) : (out_tok += 1) {
                const x_offset = out_tok * d_inner * @sizeOf(f32);
                const y_offset = out_tok * hidden_dim * @sizeOf(f32);
                try self.dispatchDmmvInner(
                    ssm_out_t,
                    scratch_swiglu,
                    scratch_swiglu.size,
                    scratch_hidden,
                    hidden_dim,
                    d_inner,
                    0,
                    x_offset,
                    y_offset,
                    1,
                );
            }
        }
        self.endProfilePhase(.ssm_out, ssm_out_phase);
        self.endProfilePhase(.ssm, ssm_phase);
        if (!use_batched_q5k_ssm_out) {
            self.decode_cmd.computeBarrier();
            try self.dispatchRmsNorm(
                scratch_hidden.handle,
                scratch_hidden.size,
                ffn_norm_t.gpu_buffer.handle,
                ffn_norm_t.gpu_buffer.size,
                scratch_norm.handle,
                scratch_norm.size,
                hidden_dim,
                n_tokens,
                self.model.config.rms_norm_eps,
            );
        }
        self.decode_cmd.computeToTransferBarrier();
        _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);
        self.recordProfilingSample();
    }

    fn prefillQwen36RunBatchedDenseFfnLayer(
        self: *InferenceEngine,
        layer: u32,
        n_tokens: u32,
        hidden_dim: u32,
        inter_dim: u32,
        scratch_hidden: Buffer,
        scratch_norm: Buffer,
        scratch_gate: Buffer,
        scratch_up: Buffer,
        scratch_swiglu: Buffer,
        scratch_down: Buffer,
    ) !void {
        const lt = self.layer_tensors[layer];
        const gate_t = lt.ffn_gate orelse return error.TensorNotFound;
        const up_t = lt.ffn_up orelse return error.TensorNotFound;
        const down_t = lt.ffn_down orelse return error.TensorNotFound;

        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.beginOneTime();
        self.resetTimestamps();
        _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);
        if (self.qwen36DensePrefillPartialStoreEnabled()) {
            const dense_input_ranges = [_]CommandBuffer.BufferRange{
                .{ .buffer = scratch_hidden.handle, .size = scratch_hidden.size },
                .{ .buffer = scratch_norm.handle, .size = scratch_norm.size },
            };
            self.decode_cmd.computeBuffersBarrier(&dense_input_ranges);
        } else {
            self.decode_cmd.transferToComputeBarrier();
        }
        const dense_ffn_phase = self.beginProfilePhase();
        const dense_ffn_gateup_phase = self.beginProfilePhase();
        const swiglu_bytes: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, n_tokens) * @as(vk.c.VkDeviceSize, inter_dim) * @sizeOf(f32);
        const hidden_batch_bytes: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, n_tokens) * @as(vk.c.VkDeviceSize, hidden_dim) * @sizeOf(f32);
        const use_fused_gateup = self.use_qwen36_batched_fused_gateup and
            self.isQwen36DenseHybrid27B() and
            gate_t.info.type_ == .q4_k and
            up_t.info.type_ == .q4_k and
            n_tokens >= 16 and
            (hidden_dim & 255) == 0 and
            self.dmmv.pipeline_mul_mm_q4k_gate_up_swiglu != null;
        if (use_fused_gateup) {
            const full_cols = n_tokens & ~@as(u32, 31);
            if (full_cols > 0 and (inter_dim & 31) == 0 and self.dmmv.pipeline_mul_mm_q4k_gate_up_swiglu_full != null) {
                try self.dmmv.recordMulMmQ4KGateUpSwigluFull(
                    &self.decode_cmd,
                    self.instance.push_descriptor_fn,
                    gate_t.gpu_buffer.handle,
                    gate_t.gpu_buffer.size,
                    up_t.gpu_buffer.handle,
                    up_t.gpu_buffer.size,
                    scratch_norm.handle,
                    scratch_norm.size,
                    scratch_swiglu.handle,
                    scratch_swiglu.size,
                    inter_dim,
                    full_cols,
                    hidden_dim,
                    hidden_dim,
                    inter_dim,
                    0,
                    0,
                    0,
                );
                if (full_cols < n_tokens) {
                    try self.dmmv.recordMulMmQ4KGateUpSwiglu(
                        &self.decode_cmd,
                        self.instance.push_descriptor_fn,
                        gate_t.gpu_buffer.handle,
                        gate_t.gpu_buffer.size,
                        up_t.gpu_buffer.handle,
                        up_t.gpu_buffer.size,
                        scratch_norm.handle,
                        scratch_norm.size,
                        scratch_swiglu.handle,
                        scratch_swiglu.size,
                        inter_dim,
                        n_tokens - full_cols,
                        hidden_dim,
                        hidden_dim,
                        inter_dim,
                        0,
                        full_cols * hidden_dim,
                        full_cols * inter_dim,
                    );
                }
            } else {
                try self.dmmv.recordMulMmQ4KGateUpSwiglu(
                    &self.decode_cmd,
                    self.instance.push_descriptor_fn,
                    gate_t.gpu_buffer.handle,
                    gate_t.gpu_buffer.size,
                    up_t.gpu_buffer.handle,
                    up_t.gpu_buffer.size,
                    scratch_norm.handle,
                    scratch_norm.size,
                    scratch_swiglu.handle,
                    scratch_swiglu.size,
                    inter_dim,
                    n_tokens,
                    hidden_dim,
                    hidden_dim,
                    inter_dim,
                    0,
                    0,
                    0,
                );
            }
        } else {
            const dense_ffn_gate_phase = self.beginProfilePhase();
            try self.dispatchProjectionBatched(gate_t, scratch_norm, scratch_gate, inter_dim, hidden_dim, n_tokens);
            self.endProfilePhase(.dense_ffn_gate, dense_ffn_gate_phase);
            const dense_ffn_up_phase = self.beginProfilePhase();
            try self.dispatchProjectionBatched(up_t, scratch_norm, scratch_up, inter_dim, hidden_dim, n_tokens);
            self.endProfilePhase(.dense_ffn_up, dense_ffn_up_phase);
            const gateup_ranges = [_]CommandBuffer.BufferRange{
                .{ .buffer = scratch_gate.handle, .size = swiglu_bytes },
                .{ .buffer = scratch_up.handle, .size = swiglu_bytes },
            };
            self.decode_cmd.computeBuffersBarrier(&gateup_ranges);
            try self.dispatchFfnActivation(
                scratch_gate.handle,
                scratch_gate.size,
                scratch_up.handle,
                scratch_up.size,
                scratch_swiglu.handle,
                scratch_swiglu.size,
                n_tokens * inter_dim,
            );
        }
        self.decode_cmd.computeBufferBarrier(scratch_swiglu.handle, swiglu_bytes);
        self.endProfilePhase(.dense_ffn_gateup, dense_ffn_gateup_phase);
        const dense_ffn_down_phase = self.beginProfilePhase();
        try self.dispatchProjectionBatched(down_t, scratch_swiglu, scratch_down, hidden_dim, inter_dim, n_tokens);
        self.decode_cmd.computeBufferBarrier(scratch_down.handle, hidden_batch_bytes);
        try self.dispatchScaleAcc(
            scratch_hidden.handle,
            scratch_hidden.size,
            scratch_down.handle,
            scratch_down.size,
            n_tokens * hidden_dim,
            1.0,
        );
        self.endProfilePhase(.dense_ffn_down, dense_ffn_down_phase);
        self.endProfilePhase(.dense_ffn, dense_ffn_phase);
        const layer_output_scale = self.layer_output_scales[layer];
        if (layer_output_scale != 1.0) {
            self.decode_cmd.computeBufferBarrier(scratch_hidden.handle, hidden_batch_bytes);
            try self.dispatchScaleInPlace(
                scratch_hidden.handle,
                scratch_hidden.size,
                n_tokens * hidden_dim,
                layer_output_scale,
            );
        }
        self.decode_cmd.computeToTransferBarrier();
        _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);
        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);
        self.recordProfilingSample();
    }

    fn prefillQwen36DenseFfnPrefix(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32, prefix_layers: u32) !void {
        if (prompt_tokens.len == 0 or prefix_layers == 0) return;

        const cfg = self.model.config;
        const n_tokens: u32 = @intCast(@min(prompt_tokens.len, std.math.maxInt(u32)));
        const hidden_dim = cfg.hidden_dim;
        const hidden_size = @as(vk.c.VkDeviceSize, hidden_dim) * @sizeOf(f32);
        const inter_dim: u32 = if (cfg.intermediate_dim > 0) cfg.intermediate_dim else hidden_dim * 4;
        const total_embed_bytes: u64 = @as(u64, hidden_dim) * @as(u64, n_tokens) * @sizeOf(f32);

        var precheck_layer: u32 = 0;
        while (precheck_layer < prefix_layers) : (precheck_layer += 1) {
            const lt = self.layer_tensors[precheck_layer];
            _ = lt.ffn_gate orelse return error.TensorNotFound;
            _ = lt.ffn_up orelse return error.TensorNotFound;
            _ = lt.ffn_down orelse return error.TensorNotFound;
            if (lt.post_ffw_norm != null) return error.UnsupportedPartialDecode;
        }

        const base_token: u32 = state.position;
        const target_context_tokens = if (state.requested_context_tokens > 0)
            @max(state.requested_context_tokens, base_token +| n_tokens)
        else
            base_token +| n_tokens;
        if (base_token == 0 and state.generated_tokens.items.len == 0) {
            try self.resetRequestState(target_context_tokens);
        } else if (base_token > 0 and self.active_kv_page_ids == null) {
            return error.KvStateNotAvailable;
        } else {
            try self.ensureKvPagesForContext(target_context_tokens);
        }

        try self.ensureBatchedScratchCapacity(n_tokens);
        if (self.prefill_embed_big == null or self.prefill_embed_big_capacity_bytes < total_embed_bytes) {
            if (self.prefill_embed_big) |*b| b.deinit();
            self.prefill_embed_big = try Buffer.initStaging(self.instance, total_embed_bytes);
            self.prefill_embed_big_capacity_bytes = total_embed_bytes;
        }

        const big_f32: [*]f32 = @ptrCast(@alignCast(self.prefill_embed_big.?.mapped.?));
        {
            const embd = self.tensor_map.get("token_embd.weight") orelse return error.TensorNotFound;
            const mmap = self.model.mmap_data orelse return error.NoMmapData;
            const data_start: usize = @intCast(self.model.gguf_file.tensor_data_offset + embd.info.offset);
            const vocab_last = cfg.vocab_size -| 1;
            for (prompt_tokens, 0..) |tok, i| {
                const safe_id = @min(tok, vocab_last);
                const dst = big_f32[i * hidden_dim ..][0..hidden_dim];
                dequantRow(mmap[data_start..], safe_id, hidden_dim, embd.info.type_, dst);
            }
        }
        self.prefill_embed_big_hidden = hidden_dim;
        self.prefill_embed_big_token_count = n_tokens;
        self.prefill_current_token_idx = 0;
        defer {
            self.prefill_embed_big_token_count = 0;
            self.prefill_embed_big_hidden = 0;
            self.prefill_current_token_idx = 0;
        }

        const scratch_hidden = self.batched_scratch_hidden.?;
        const scratch_norm = self.batched_scratch_norm.?;
        const scratch_gate = self.batched_scratch_gate.?;
        const scratch_up = self.batched_scratch_up.?;
        const scratch_swiglu = self.batched_scratch_swiglu.?;
        const scratch_down = self.batched_scratch_down.?;
        const use_ssm_preproj = self.qwen36DensePrefillSsmPreprojEnabled();
        const ssm_preproj_mode = getenv("ZINC_QWEN36_27B_SSM_PREFILL_PROJ") orelse "";
        const prebatch_ssm_qkv = std.mem.eql(u8, ssm_preproj_mode, "1") or
            std.mem.eql(u8, ssm_preproj_mode, "both") or
            std.mem.eql(u8, ssm_preproj_mode, "qkv");
        const prebatch_ssm_z = std.mem.eql(u8, ssm_preproj_mode, "1") or
            std.mem.eql(u8, ssm_preproj_mode, "both") or
            std.mem.eql(u8, ssm_preproj_mode, "z");
        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.beginOneTime();
        const embed_region = vk.c.VkBufferCopy{
            .srcOffset = 0,
            .dstOffset = 0,
            .size = total_embed_bytes,
        };
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.prefill_embed_big.?.handle, scratch_hidden.handle, 1, &embed_region);
        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

        self.prefill_token_samples = 0;
        self.prefill_cpu_embed_ns = 0;
        self.prefill_cpu_record_ns = 0;
        self.prefill_submit_wait_ns = 0;
        self.prefill_gpu_phase_ns = [_]u64{0} ** profile_phase_count;
        self.prefill_gpu_total_ns = 0;
        const profile_env = getenv("ZINC_PREFILL_PROFILE");
        const want_gpu_phases = profile_env != null and profile_env.?.len > 0 and !std.mem.eql(u8, profile_env.?, "0");
        const profile_was_enabled = self.profile_enabled;
        const enable_gpu_phase_timing = self.timestamp_query_pool != null and want_gpu_phases;
        if (enable_gpu_phase_timing) self.profile_enabled = true;
        defer {
            if (enable_gpu_phase_timing) {
                for (0..profile_phase_count) |p| {
                    self.prefill_gpu_phase_ns[p] = self.profile_total_counters.gpu_phase_ns[p];
                }
                self.prefill_gpu_total_ns = @intFromFloat(@max(self.profile_total_gpu_ms * 1_000_000.0, 0.0));
                self.resetProfilingSamples();
                self.profile_enabled = profile_was_enabled;
            }
        }
        self.prefill_active = true;
        defer self.prefill_active = false;

        const saved_partial_start = self.partial_decode_start_layer;
        const saved_partial_end = self.partial_decode_end_layer;
        const saved_hidden_in = self.partial_decode_hidden_in;
        const saved_hidden_in_offset = self.partial_decode_hidden_in_offset;
        const saved_hidden_out = self.partial_decode_hidden_out;
        const saved_hidden_out_offset = self.partial_decode_hidden_out_offset;
        const saved_advance = self.partial_decode_advance_position;
        const saved_allow_tail = self.partial_decode_allow_final_tail;
        const saved_stop_after_norm = self.partial_decode_stop_after_ffn_norm;
        const saved_norm_out = self.partial_decode_ffn_norm_out;
        const saved_norm_out_offset = self.partial_decode_ffn_norm_out_offset;
        const saved_stop_after_ssm_gnorm = self.partial_decode_stop_after_ssm_gnorm;
        const saved_ssm_gnorm_out = self.partial_decode_ssm_gnorm_out;
        const saved_ssm_gnorm_out_offset = self.partial_decode_ssm_gnorm_out_offset;
        const saved_stop_after_ssm_conv = self.partial_decode_stop_after_ssm_conv;
        const saved_ssm_conv_out = self.partial_decode_ssm_conv_out;
        const saved_ssm_conv_out_offset = self.partial_decode_ssm_conv_out_offset;
        const saved_ssm_z_out = self.partial_decode_ssm_z_out;
        const saved_ssm_z_out_offset = self.partial_decode_ssm_z_out_offset;
        const saved_ssm_alpha_out = self.partial_decode_ssm_alpha_out;
        const saved_ssm_alpha_out_offset = self.partial_decode_ssm_alpha_out_offset;
        const saved_ssm_beta_out = self.partial_decode_ssm_beta_out;
        const saved_ssm_beta_out_offset = self.partial_decode_ssm_beta_out_offset;
        const saved_ssm_preproj_layer = self.partial_ssm_preproj_layer;
        const saved_ssm_preproj_token_idx = self.partial_ssm_preproj_token_idx;
        const saved_ssm_preproj_qkv = self.partial_ssm_preproj_qkv;
        const saved_ssm_preproj_qkv_size = self.partial_ssm_preproj_qkv_size;
        const saved_ssm_preproj_qkv_stride = self.partial_ssm_preproj_qkv_stride;
        const saved_ssm_preproj_z = self.partial_ssm_preproj_z;
        const saved_ssm_preproj_z_size = self.partial_ssm_preproj_z_size;
        const saved_ssm_preproj_z_stride = self.partial_ssm_preproj_z_stride;
        defer {
            self.partial_decode_start_layer = saved_partial_start;
            self.partial_decode_end_layer = saved_partial_end;
            self.partial_decode_hidden_in = saved_hidden_in;
            self.partial_decode_hidden_in_offset = saved_hidden_in_offset;
            self.partial_decode_hidden_out = saved_hidden_out;
            self.partial_decode_hidden_out_offset = saved_hidden_out_offset;
            self.partial_decode_advance_position = saved_advance;
            self.partial_decode_allow_final_tail = saved_allow_tail;
            self.partial_decode_stop_after_ffn_norm = saved_stop_after_norm;
            self.partial_decode_ffn_norm_out = saved_norm_out;
            self.partial_decode_ffn_norm_out_offset = saved_norm_out_offset;
            self.partial_decode_stop_after_ssm_gnorm = saved_stop_after_ssm_gnorm;
            self.partial_decode_ssm_gnorm_out = saved_ssm_gnorm_out;
            self.partial_decode_ssm_gnorm_out_offset = saved_ssm_gnorm_out_offset;
            self.partial_decode_stop_after_ssm_conv = saved_stop_after_ssm_conv;
            self.partial_decode_ssm_conv_out = saved_ssm_conv_out;
            self.partial_decode_ssm_conv_out_offset = saved_ssm_conv_out_offset;
            self.partial_decode_ssm_z_out = saved_ssm_z_out;
            self.partial_decode_ssm_z_out_offset = saved_ssm_z_out_offset;
            self.partial_decode_ssm_alpha_out = saved_ssm_alpha_out;
            self.partial_decode_ssm_alpha_out_offset = saved_ssm_alpha_out_offset;
            self.partial_decode_ssm_beta_out = saved_ssm_beta_out;
            self.partial_decode_ssm_beta_out_offset = saved_ssm_beta_out_offset;
            self.partial_ssm_preproj_layer = saved_ssm_preproj_layer;
            self.partial_ssm_preproj_token_idx = saved_ssm_preproj_token_idx;
            self.partial_ssm_preproj_qkv = saved_ssm_preproj_qkv;
            self.partial_ssm_preproj_qkv_size = saved_ssm_preproj_qkv_size;
            self.partial_ssm_preproj_qkv_stride = saved_ssm_preproj_qkv_stride;
            self.partial_ssm_preproj_z = saved_ssm_preproj_z;
            self.partial_ssm_preproj_z_size = saved_ssm_preproj_z_size;
            self.partial_ssm_preproj_z_stride = saved_ssm_preproj_z_stride;
        }

        const full_attn_interval = if (cfg.full_attn_interval > 0) cfg.full_attn_interval else 1;
        var layer: u32 = 0;
        while (layer < prefix_layers) : (layer += 1) {
            self.partial_ssm_preproj_layer = std.math.maxInt(u32);
            self.partial_ssm_preproj_token_idx = 0;
            self.partial_ssm_preproj_qkv = null;
            self.partial_ssm_preproj_qkv_size = 0;
            self.partial_ssm_preproj_qkv_stride = 0;
            self.partial_ssm_preproj_z = null;
            self.partial_ssm_preproj_z_size = 0;
            self.partial_ssm_preproj_z_stride = 0;

            const is_full_attn = ((layer + 1) % full_attn_interval) == 0;
            if (use_ssm_preproj and !is_full_attn) {
                const lt = self.layer_tensors[layer];
                const attn_norm_t = lt.attn_norm orelse return error.TensorNotFound;
                const wqkv_t = lt.attn_qkv orelse return error.TensorNotFound;
                const z_t = lt.attn_gate orelse return error.TensorNotFound;
                const d_inner = cfg.ssm_d_inner;
                const conv_channels = d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state;
                const qkv_stride: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, conv_channels) * @sizeOf(f32);
                const z_stride: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, d_inner) * @sizeOf(f32);
                const qkv_total = qkv_stride * @as(vk.c.VkDeviceSize, n_tokens);
                const z_total = z_stride * @as(vk.c.VkDeviceSize, n_tokens);
                const can_prebatch_qkv = prebatch_ssm_qkv and qkv_total <= scratch_gate.size;
                const can_prebatch_z = prebatch_ssm_z and z_total <= scratch_up.size;
                if (can_prebatch_qkv or can_prebatch_z) {
                    if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                    try self.decode_cmd.reset();
                    try self.decode_cmd.beginOneTime();
                    if (layer == 0) {
                        self.decode_cmd.transferToComputeBarrier();
                    } else {
                        self.decode_cmd.computeBarrier();
                    }
                    try self.dispatchRmsNorm(
                        scratch_hidden.handle,
                        scratch_hidden.size,
                        attn_norm_t.gpu_buffer.handle,
                        attn_norm_t.gpu_buffer.size,
                        scratch_norm.handle,
                        scratch_norm.size,
                        hidden_dim,
                        n_tokens,
                        cfg.rms_norm_eps,
                    );
                    self.decode_cmd.computeBarrier();
                    if (can_prebatch_qkv) {
                        try self.dispatchProjectionBatched(wqkv_t, scratch_norm, scratch_gate, conv_channels, hidden_dim, n_tokens);
                    }
                    if (can_prebatch_z) {
                        try self.dispatchProjectionBatched(z_t, scratch_norm, scratch_up, d_inner, hidden_dim, n_tokens);
                    }
                    self.decode_cmd.computeToTransferBarrier();
                    try self.decode_cmd.end();
                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                    self.partial_ssm_preproj_layer = layer;
                    if (can_prebatch_qkv) {
                        self.partial_ssm_preproj_qkv = scratch_gate.handle;
                        self.partial_ssm_preproj_qkv_size = scratch_gate.size;
                        self.partial_ssm_preproj_qkv_stride = qkv_stride;
                    }
                    if (can_prebatch_z) {
                        self.partial_ssm_preproj_z = scratch_up.handle;
                        self.partial_ssm_preproj_z_size = scratch_up.size;
                        self.partial_ssm_preproj_z_stride = z_stride;
                    }
                }
            }

            if (!is_full_attn and !use_ssm_preproj) {
                try self.prefillQwen36RunSsmLayerToFfnNorm(
                    state,
                    prompt_tokens,
                    base_token,
                    n_tokens,
                    hidden_dim,
                    cfg.ssm_d_inner,
                    hidden_size,
                    layer,
                    scratch_hidden,
                    scratch_gate,
                    scratch_up,
                    self.batched_scratch_q.?,
                    self.batched_scratch_k.?,
                    self.batched_scratch_attn_out.?,
                    scratch_swiglu,
                    scratch_norm,
                    scratch_down,
                    false,
                );
            } else {
                var tok_idx: u32 = 0;
                while (tok_idx < n_tokens) : (tok_idx += 1) {
                    const hidden_offset: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok_idx) * hidden_size;
                    self.prefill_current_token_idx = tok_idx;
                    self.partial_ssm_preproj_token_idx = tok_idx;
                    state.position = base_token + tok_idx;
                    self.partial_decode_start_layer = layer;
                    self.partial_decode_end_layer = layer + 1;
                    self.partial_decode_hidden_in = scratch_hidden.handle;
                    self.partial_decode_hidden_in_offset = hidden_offset;
                    self.partial_decode_hidden_out = scratch_hidden.handle;
                    self.partial_decode_hidden_out_offset = hidden_offset;
                    self.partial_decode_advance_position = false;
                    self.partial_decode_allow_final_tail = false;
                    self.partial_decode_stop_after_ffn_norm = true;
                    self.partial_decode_ffn_norm_out = scratch_norm.handle;
                    self.partial_decode_ffn_norm_out_offset = hidden_offset;
                    try self.decodeStep(state, prompt_tokens[tok_idx], false);
                }
            }

            try self.prefillQwen36RunBatchedDenseFfnLayer(
                layer,
                n_tokens,
                hidden_dim,
                inter_dim,
                scratch_hidden,
                scratch_norm,
                scratch_gate,
                scratch_up,
                scratch_swiglu,
                scratch_down,
            );
        }

        const pipeline_tail = self.qwen36DensePrefillTailPipelineEnabled(n_tokens);
        var tail_start_layer = prefix_layers;
        var segment_layers: [qwen36_dense_prefill_max_segments]u32 = undefined;
        const n_segment_layers = self.qwen36DensePrefillSegmentLayers(prompt_tokens.len, prefix_layers, &segment_layers);
        for (segment_layers[0..n_segment_layers]) |segment_layer| {
            if (segment_layer < tail_start_layer) continue;
            log.debug("Qwen3.6-27B dense prefill segment ENABLED at layer {d} after tail_start_layer={d} (set ZINC_QWEN36_27B_DENSE_PREFILL_SEGMENT=0 to disable)", .{
                segment_layer,
                tail_start_layer,
            });
            if (segment_layer > tail_start_layer) {
                try self.prefillQwen36RunPartialTokenLoop(
                    state,
                    prompt_tokens,
                    base_token,
                    n_tokens,
                    hidden_size,
                    tail_start_layer,
                    segment_layer,
                    scratch_hidden,
                    null,
                    true,
                    false,
                    false,
                    false,
                    pipeline_tail,
                );
            }
            const segment_is_full_attn = ((segment_layer + 1) % full_attn_interval) == 0;
            if (!segment_is_full_attn) {
                try self.prefillQwen36RunSsmLayerToFfnNorm(
                    state,
                    prompt_tokens,
                    base_token,
                    n_tokens,
                    hidden_dim,
                    cfg.ssm_d_inner,
                    hidden_size,
                    segment_layer,
                    scratch_hidden,
                    scratch_gate,
                    scratch_up,
                    self.batched_scratch_q.?,
                    self.batched_scratch_k.?,
                    self.batched_scratch_attn_out.?,
                    scratch_swiglu,
                    scratch_norm,
                    scratch_down,
                    pipeline_tail,
                );
            } else {
                try self.prefillQwen36RunPartialTokenLoop(
                    state,
                    prompt_tokens,
                    base_token,
                    n_tokens,
                    hidden_size,
                    segment_layer,
                    segment_layer + 1,
                    scratch_hidden,
                    scratch_norm,
                    true,
                    true,
                    false,
                    false,
                    pipeline_tail,
                );
            }
            try self.prefillQwen36RunBatchedDenseFfnLayer(
                segment_layer,
                n_tokens,
                hidden_dim,
                inter_dim,
                scratch_hidden,
                scratch_norm,
                scratch_gate,
                scratch_up,
                scratch_swiglu,
                scratch_down,
            );
            tail_start_layer = segment_layer + 1;
        }

        self.partial_decode_stop_after_ffn_norm = false;
        self.partial_decode_ffn_norm_out = null;
        self.partial_decode_ffn_norm_out_offset = 0;
        self.partial_decode_start_layer = tail_start_layer;
        self.partial_decode_end_layer = 0;
        self.partial_decode_hidden_in = scratch_hidden.handle;
        self.partial_decode_hidden_out = null;
        self.partial_decode_advance_position = true;

        // The prefix path runs the first layer(s) layer-major, then returns to
        // token-major decode for the remainder. Reuse prefillBatch's two-slot
        // command-buffer pipeline for that remainder when explicitly requested.
        var primary_pending: bool = false;
        var alt_pending: bool = false;
        var tok_idx: u32 = 0;
        while (tok_idx < n_tokens) : (tok_idx += 1) {
            const collect_output = tok_idx + 1 == n_tokens;
            const pipeline_this = pipeline_tail and !collect_output;
            if (pipeline_this) {
                std.mem.swap(CommandBuffer, &self.decode_cmd, &self.prefill_cmd_alt);
                std.mem.swap(bool, &primary_pending, &alt_pending);
                if (primary_pending) {
                    try self.decode_cmd.waitForCompletion();
                    primary_pending = false;
                }
                self.prefill_pipeline_mode = true;
            } else {
                if (alt_pending) {
                    try self.prefill_cmd_alt.waitForCompletion();
                    alt_pending = false;
                }
                if (primary_pending) {
                    try self.decode_cmd.waitForCompletion();
                    primary_pending = false;
                }
                self.prefill_pipeline_mode = false;
            }

            self.prefill_current_token_idx = tok_idx;
            state.position = base_token + tok_idx;
            self.partial_decode_hidden_in_offset = @as(vk.c.VkDeviceSize, tok_idx) * hidden_size;
            self.partial_decode_allow_final_tail = collect_output;
            try self.decodeStep(state, prompt_tokens[tok_idx], collect_output);

            if (pipeline_this) {
                primary_pending = true;
            }
        }
        self.prefill_pipeline_mode = false;
        if (alt_pending) {
            try self.prefill_cmd_alt.waitForCompletion();
        }
        if (primary_pending) {
            try self.decode_cmd.waitForCompletion();
        }
    }

    /// Experimental batched prompt prefill for the RDNA/Vulkan backend.
    /// Gated by `ZINC_BATCHED_PREFILL=1`. This is the Vulkan analogue of
    /// `forward_metal.InferenceEngine.prefillBatched`.
    ///
    /// Foundation committed: the `rope_batched` and `flash_attn_batched` SPIR-V
    /// shaders and their pipeline wrappers (`elementwise.pipeline_rope_batched`,
    /// `attention.pipeline_batched`, plus matching push structs and dispatchers)
    /// are loaded at engine init. The orchestration that ties them together with
    /// `dmmv_q4k_batch` (weight-read-once GEMM) for projections is tracked in
    /// `loops/efforts/MULTI_HOUR_EFFORT_8_RDNA_BATCHED_PREFILL.md`. Until that orchestration
    /// lands this entry point transparently delegates to `prefillBatch`, but the
    /// env gate and the `canUseBatchedPrefillRdna` check are already wired so
    /// callers can migrate to the new name ahead of time — matching the Metal
    /// path where `generateWithMetrics` already routes through `prefillBatched`.
    pub fn prefillBatched(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {
        const mode = getenv("ZINC_BATCHED_PREFILL") orelse "";
        const intel_batched_env = getenv("ZINC_INTEL_BATCHED_PREFILL");
        const intel_batched_requested = isIntelGpuVendor(self.gpu_config.vendor) and
            intel_batched_env != null and std.mem.eql(u8, intel_batched_env.?, "1");
        const chunk_limit = intelBatchedPrefillChunkLimit(self.gpu_config.vendor);
        if (intel_batched_requested and
            chunk_limit > 0 and
            prompt_tokens.len > chunk_limit and
            !std.mem.eql(u8, mode, "0") and
            !std.mem.eql(u8, mode, "validate"))
        {
            log.info("Intel batched prefill chunking ENABLED: chunk={d} tokens (set ZINC_INTEL_BATCHED_PREFILL_CHUNK=0 to force monolithic)", .{chunk_limit});
            var offset: usize = 0;
            while (offset < prompt_tokens.len) {
                const end = @min(offset + chunk_limit, prompt_tokens.len);
                try self.prefillBatchedImpl(state, prompt_tokens[offset..end]);
                offset = end;
            }
            return;
        }
        return self.prefillBatchedImpl(state, prompt_tokens);
    }

    fn prefillBatchedImpl(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {
        // Default ON for models that pass `canUseBatchedPrefillRdna` — the gate
        // already rejects anything the batched body doesn't handle. Set
        // `ZINC_BATCHED_PREFILL=0` to force the per-token fallback (escape
        // hatch for debugging / numerical-sensitivity testing); `=validate`
        // runs both paths and diffs the last-token logits.
        const mode = getenv("ZINC_BATCHED_PREFILL") orelse "";
        const batched_disabled = std.mem.eql(u8, mode, "0");
        const validate_mode = std.mem.eql(u8, mode, "validate");
        if (!batched_disabled and !validate_mode) {
            const dense_prefix_layers = self.qwen36DensePrefillPrefixLayers(prompt_tokens.len);
            if (dense_prefix_layers > 0) {
                return self.prefillQwen36DenseFfnPrefix(state, prompt_tokens, dense_prefix_layers);
            }
        }
        if ((batched_disabled and !validate_mode) or !canUseBatchedPrefillRdna(self)) {
            return self.prefillBatch(state, prompt_tokens);
        }
        if (prompt_tokens.len == 0) return;

        // Validate mode requires a fresh state so we can replay the per-token
        // path on a clean slate after the batched run and diff the logits.
        if (validate_mode and state.position != 0) {
            return self.prefillBatch(state, prompt_tokens);
        }

        // Extension: supports prefix reuse (state.position > 0) as long as
        // the KV pages from the prior call are still live. Unlike the Metal
        // path there is no engine-side position cursor to cross-check against
        // — state.position is authoritative on Vulkan.
        if (state.position > 0 and self.active_kv_page_ids == null) {
            return self.prefillBatch(state, prompt_tokens);
        }

        // Ensure scratch buffers are sized for this prompt — reused across
        // subsequent prefill calls so the alloc is amortized.
        const n_tokens: u32 = @intCast(@min(prompt_tokens.len, std.math.maxInt(u32)));
        try self.ensureBatchedScratchCapacity(n_tokens);

        // ── Step 1: pre-dequantize all N embedding rows on the CPU into
        // prefill_embed_big (host-staged), then DMA-copy into
        // batched_scratch_hidden (device-local) inside the command buffer.
        const cfg = self.model.config;
        const hidden_dim = cfg.hidden_dim;
        const inter_dim: u32 = if (cfg.intermediate_dim > 0) cfg.intermediate_dim else hidden_dim * 4;
        // Per-layer head_dim / q_dim / kv_dim / rope_dim are derived inside
        // the layer loop (Gemma-aware). cfg.head_dim is only used as the
        // scratch-buffer sizing ceiling via ensureBatchedScratchCapacity.
        const total_embed_bytes: u64 = @as(u64, hidden_dim) * @as(u64, n_tokens) * @sizeOf(f32);
        if (self.prefill_embed_big == null or self.prefill_embed_big_capacity_bytes < total_embed_bytes) {
            if (self.prefill_embed_big) |*b| b.deinit();
            self.prefill_embed_big = try Buffer.initStaging(self.instance, total_embed_bytes);
            self.prefill_embed_big_capacity_bytes = total_embed_bytes;
        }
        {
            const big_f32: [*]f32 = @ptrCast(@alignCast(self.prefill_embed_big.?.mapped.?));
            const embd = self.tensor_map.get("token_embd.weight") orelse return error.TensorNotFound;
            const mmap = self.model.mmap_data orelse return error.NoMmapData;
            const data_start: usize = @intCast(self.model.gguf_file.tensor_data_offset + embd.info.offset);
            const vocab_last = cfg.vocab_size -| 1;
            // Gemma pre-scales embeddings by sqrt(hidden_dim) before the
            // first layer. Matches the per-token loadTokenEmbedding path.
            const is_gemma = cfg.architecture == .gemma;
            const gemma_scale: f32 = if (is_gemma)
                @floatCast(@sqrt(@as(f64, @floatFromInt(hidden_dim))))
            else
                1.0;
            for (prompt_tokens, 0..) |tok, i| {
                const safe_id = @min(tok, vocab_last);
                const dst = big_f32[i * hidden_dim ..][0..hidden_dim];
                dequantRow(mmap[data_start..], safe_id, hidden_dim, embd.info.type_, dst);
                if (is_gemma) {
                    for (dst) |*v| v.* *= gemma_scale;
                }
            }
            self.prefill_embed_big_hidden = hidden_dim;
            self.prefill_embed_big_token_count = n_tokens;
        }

        // Reset request state for a fresh prefill, or grow the KV page pool
        // if we are extending an existing conversation. Mirror the shape of
        // prefillBatch so pipelined prefill / decodeStep invariants hold.
        const base_token: u32 = state.position;
        const target_context_tokens = if (state.requested_context_tokens > 0)
            @max(state.requested_context_tokens, base_token +| n_tokens)
        else
            base_token +| n_tokens;
        if (base_token == 0 and state.generated_tokens.items.len == 0) {
            try self.resetRequestState(target_context_tokens);
        } else {
            try self.ensureKvPagesForContext(target_context_tokens);
        }

        const scratch_hidden = self.batched_scratch_hidden.?;
        const scratch_norm = self.batched_scratch_norm.?;
        const scratch_q = self.batched_scratch_q.?;
        const scratch_k = self.batched_scratch_k.?;
        const scratch_v = self.batched_scratch_v.?;
        const scratch_attn_out = self.batched_scratch_attn_out.?;
        const scratch_gate = self.batched_scratch_gate.?;
        const scratch_up = self.batched_scratch_up.?;
        const scratch_swiglu = self.batched_scratch_swiglu.?;
        const scratch_down = self.batched_scratch_down.?;

        try self.decode_cmd.reset();
        try self.decode_cmd.beginOneTime();

        // ── Step 2: DMA embeddings host-staged → device-local scratch_hidden.
        {
            const region = vk.c.VkBufferCopy{
                .srcOffset = 0,
                .dstOffset = 0,
                .size = total_embed_bytes,
            };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.prefill_embed_big.?.handle, scratch_hidden.handle, 1, &region);
            self.decode_cmd.transferToComputeBarrier();
        }

        // ── Step 3: per-layer batched forward.
        const eps = cfg.rms_norm_eps;
        const freq_buf_handle = self.rope_freq_buf.handle;
        const freq_buf_size = self.rope_freq_buf.size;

        for (0..cfg.n_layers) |layer_idx| {
            const layer: u32 = @intCast(layer_idx);
            const lt = self.layer_tensors[layer_idx];
            const attn_norm_t = lt.attn_norm orelse return error.TensorNotFound;
            const ffn_norm_t = lt.ffn_norm orelse return error.TensorNotFound;
            const q_t = lt.attn_q.?;
            const k_t = lt.attn_k.?;
            // Gemma 4 full-attention layers omit attn_v and use K as V.
            // For those layers, skip the V projection and feed scratch_k
            // into the KV cache write and flash attention in place of V.
            const use_k_as_v = lt.attn_v == null and cfg.architecture == .gemma;
            const v_t_opt = lt.attn_v;
            const o_t = lt.attn_output.?;
            const gate_t = lt.ffn_gate.?;
            const up_t = lt.ffn_up.?;
            const down_t = lt.ffn_down.?;

            // Per-layer attention dims. Gemma 4 varies both head_dim AND
            // n_kv_heads per layer: full-attn layers use head_dim=512 with
            // 4 kv_heads (32:4 GQA), SWA layers use head_dim=256 with 16
            // kv_heads (32:16 GQA). Q and KV share the same head_dim on
            // every layer — derive it from the Q norm tensor (same pattern
            // as the per-token runDecodeStep path) and derive kv_dim and
            // q_dim directly from weight-tensor element counts so the
            // per-layer GQA ratio falls out automatically.
            const layer_head_dim: u32 = if (lt.attn_q_norm) |qn|
                @intCast(qn.info.numElements())
            else
                cfg.head_dim;
            const layer_q_dim: u32 = @intCast(q_t.info.numElements() / hidden_dim);
            const layer_kv_dim: u32 = @intCast(k_t.info.numElements() / hidden_dim);
            const layer_n_kv_heads: u32 = if (layer_head_dim > 0) layer_kv_dim / layer_head_dim else cfg.n_kv_heads;
            const layer_rope_dim: u32 = if (cfg.rope_dim > 0)
                @min(cfg.rope_dim, layer_head_dim)
            else
                layer_head_dim;
            // The O projection's input dimension is whatever the weight row
            // count says — matches layer_q_dim for Gemma, falls through to
            // q_dim for architectures without attn_q_norm.
            const o_input_cols: u32 = @intCast(o_t.info.numElements() / hidden_dim);

            // attn RMS norm: hidden → norm
            try self.dispatchRmsNorm(scratch_hidden.handle, scratch_hidden.size, attn_norm_t.gpu_buffer.handle, attn_norm_t.gpu_buffer.size, scratch_norm.handle, scratch_norm.size, hidden_dim, n_tokens, eps);
            self.decode_cmd.computeBarrier();

            // Q / K / V projections (weight read once per chunk). On Gemma's
            // use_k_as_v layers V is the RAW K projection (pre-norm, pre-rope),
            // unit-normed per head. Instead of a second K projection we fuse
            // the copy with the unit-norm: dispatchRmsNorm reads scratch_k and
            // writes scratch_v. For non-use_k_as_v Gemma layers V has its own
            // projection; unit-norm is in place.
            try self.dispatchProjectionBatched(q_t, scratch_norm, scratch_q, layer_q_dim, hidden_dim, n_tokens);
            try self.dispatchProjectionBatched(k_t, scratch_norm, scratch_k, layer_kv_dim, hidden_dim, n_tokens);
            if (v_t_opt) |v_t| {
                try self.dispatchProjectionBatched(v_t, scratch_norm, scratch_v, layer_kv_dim, hidden_dim, n_tokens);
            }
            self.decode_cmd.computeBarrier();

            // Optional per-head Q/K norms (Qwen3 style). Dispatch one workgroup per
            // (token, head) slot — rms_norm_mul handles this via group_id * head_dim.
            // Gemma 4 also applies a plain (unit-weight) RMS norm to V per head —
            // matches per-token runDecodeStep and forward_metal.zig. For use_k_as_v
            // the unit-norm reads from scratch_k (raw K proj) → scratch_v, doing
            // the K→V copy and the norm in one pass — but it must finish before
            // K norm overwrites scratch_k, so place it AHEAD of K norm with a
            // compute barrier between them.
            const apply_v_unit_norm = cfg.architecture == .gemma and cfg.rope_freq_base_swa > 0;
            if (apply_v_unit_norm) {
                const v_src_for_norm = if (use_k_as_v) scratch_k else scratch_v;
                try self.dispatchRmsNorm(v_src_for_norm.handle, v_src_for_norm.size, self.unit_norm_weights.handle, self.unit_norm_weights.size, scratch_v.handle, scratch_v.size, layer_head_dim, layer_n_kv_heads * n_tokens, eps);
                if (use_k_as_v) self.decode_cmd.computeBarrier();
            }
            if (lt.attn_q_norm) |qn| {
                try self.dispatchRmsNorm(scratch_q.handle, scratch_q.size, qn.gpu_buffer.handle, qn.gpu_buffer.size, scratch_q.handle, scratch_q.size, layer_head_dim, cfg.n_heads * n_tokens, eps);
            }
            if (lt.attn_k_norm) |kn| {
                try self.dispatchRmsNorm(scratch_k.handle, scratch_k.size, kn.gpu_buffer.handle, kn.gpu_buffer.size, scratch_k.handle, scratch_k.size, layer_head_dim, layer_n_kv_heads * n_tokens, eps);
            }
            if (lt.attn_q_norm != null or lt.attn_k_norm != null or apply_v_unit_norm) self.decode_cmd.computeBarrier();

            // Batched RoPE for Q and K. position_base = state.position so a
            // prefix-reuse call rotates the newly-added tokens at the correct
            // sequence positions (base_token, base_token+1, ..., base_token+N-1).
            // Gemma 4 picks the RoPE frequency source per layer:
            //   - Global (full-attn) layers use precomputed rope_freq_buf
            //     with rope_freqs.weight factors pre-baked. Signal buffer
            //     use by passing freq_base=0 (shader reads inv_freq[]).
            //   - SWA layers use a DIFFERENT base (rope_freq_base_swa) and
            //     compute the frequency on the fly.
            // For non-Gemma architectures the existing behavior stands:
            // cfg.rope_freq_base with the shipped rope_freq_buf.
            const layer_is_swa_rope = cfg.architecture == .gemma and
                cfg.rope_freq_base_swa > 0 and
                layer_head_dim < cfg.head_dim;
            const layer_use_precomp_freq = cfg.architecture == .gemma and !layer_is_swa_rope;
            const layer_rope_freq_base: f32 = if (layer_use_precomp_freq)
                0.0
            else if (layer_is_swa_rope)
                cfg.rope_freq_base_swa
            else
                cfg.rope_freq_base;
            try self.dispatchRopeBatched(scratch_q.handle, scratch_q.size, scratch_q.handle, scratch_q.size, freq_buf_handle, freq_buf_size, layer_head_dim, layer_rope_dim, cfg.n_heads, base_token, n_tokens, layer_rope_freq_base, 1.0);
            try self.dispatchRopeBatched(scratch_k.handle, scratch_k.size, scratch_k.handle, scratch_k.size, freq_buf_handle, freq_buf_size, layer_head_dim, layer_rope_dim, layer_n_kv_heads, base_token, n_tokens, layer_rope_freq_base, 1.0);
            self.decode_cmd.computeBarrier();

            // Batched KV cache write: one compute dispatch writes all N tokens'
            // K/V into their paged cache slots via the page_table_buf lookup.
            // base_token places the write after the existing prefix. V was
            // populated from its own projection (or from a duplicate K projection
            // when use_k_as_v) and, for Gemma 4, unit-normed — always use scratch_v.
            try self.dispatchKvCacheWriteBatched(
                scratch_k.handle,
                scratch_k.size,
                self.kv_k_cache[layer_idx].handle,
                self.kv_k_cache[layer_idx].size,
                scratch_v.handle,
                scratch_v.size,
                self.kv_v_cache[layer_idx].handle,
                self.kv_v_cache[layer_idx].size,
                self.page_table_buf.handle,
                self.page_table_buf.size,
                layer_kv_dim,
                n_tokens,
                kv_page_size_tokens,
                base_token,
            );
            self.decode_cmd.computeBarrier();

            // Batched causal flash attention: N queries over the KV cache.
            // seq_start = base_token so each query attends to prefix + own
            // position within the batch (causal_len = base_token + query + 1).
            const sink_offset = layer * cfg.n_heads;
            try self.dispatchFlashAttnBatched(scratch_q.handle, scratch_q.size, self.kv_k_cache[layer_idx].handle, self.kv_k_cache[layer_idx].size, self.kv_v_cache[layer_idx].handle, self.kv_v_cache[layer_idx].size, self.page_table_buf.handle, self.page_table_buf.size, scratch_attn_out.handle, scratch_attn_out.size, self.attn_sinks_buf.handle, self.attn_sinks_buf.size, layer_head_dim, cfg.n_heads, layer_n_kv_heads, base_token, n_tokens, kv_page_size_tokens, cfg.attn_scale, sink_offset);
            self.decode_cmd.computeBarrier();

            // O projection, then optional Gemma post-attention norm, then
            // FUSED residual+FFN norm (hidden += down; norm = normalize(hidden)
            // * ffn_norm_weight). For non-Gemma this replaces
            // scale_acc → barrier → rms_norm_mul with a single dispatch; for
            // Gemma we first RMS-normalize the attn output in place before
            // the residual add.
            try self.dispatchProjectionBatched(o_t, scratch_attn_out, scratch_down, hidden_dim, o_input_cols, n_tokens);
            self.decode_cmd.computeBarrier();
            if (cfg.architecture == .gemma) {
                if (lt.post_attention_norm) |pan_t| {
                    try self.dispatchRmsNorm(
                        scratch_down.handle,
                        scratch_down.size,
                        pan_t.gpu_buffer.handle,
                        pan_t.gpu_buffer.size,
                        scratch_down.handle,
                        scratch_down.size,
                        hidden_dim,
                        n_tokens,
                        eps,
                    );
                    self.decode_cmd.computeBarrier();
                }
            }
            try self.dispatchResidualRmsNorm(scratch_hidden.handle, scratch_hidden.size, scratch_down.handle, scratch_down.size, scratch_norm.handle, scratch_norm.size, ffn_norm_t.gpu_buffer.handle, ffn_norm_t.gpu_buffer.size, hidden_dim, n_tokens, eps, 1.0);
            self.decode_cmd.computeBarrier();

            // FFN: gate/up → SwiGLU/GEGLU → down → optional post-ffn norm
            // (Gemma) → residual.
            try self.dispatchProjectionBatched(gate_t, scratch_norm, scratch_gate, inter_dim, hidden_dim, n_tokens);
            try self.dispatchProjectionBatched(up_t, scratch_norm, scratch_up, inter_dim, hidden_dim, n_tokens);
            self.decode_cmd.computeBarrier();
            // dispatchFfnActivation picks SwiGLU / GEGLU / SwiGLU-OAI based
            // on cfg.architecture. For Gemma this dispatches GEGLU.
            try self.dispatchFfnActivation(scratch_gate.handle, scratch_gate.size, scratch_up.handle, scratch_up.size, scratch_swiglu.handle, scratch_swiglu.size, n_tokens * inter_dim);
            self.decode_cmd.computeBarrier();
            try self.dispatchProjectionBatched(down_t, scratch_swiglu, scratch_down, hidden_dim, inter_dim, n_tokens);
            self.decode_cmd.computeBarrier();
            // Fused post_ffw_norm + residual add for Gemma: one dispatch
            // instead of (rms_norm_mul in place) + barrier + (scale_accumulate).
            // Falls back to the separate ops for non-Gemma or when the fused
            // pipeline failed to load.
            const use_fused_pfn = cfg.architecture == .gemma and
                lt.post_ffw_norm != null and
                self.elementwise.pipeline_rms_norm_add != null;
            if (use_fused_pfn) {
                const pfn_t = lt.post_ffw_norm.?;
                try self.dispatchRmsNormAdd(
                    scratch_hidden.handle,
                    scratch_hidden.size,
                    scratch_down.handle,
                    scratch_down.size,
                    pfn_t.gpu_buffer.handle,
                    pfn_t.gpu_buffer.size,
                    hidden_dim,
                    n_tokens,
                    eps,
                );
            } else {
                if (cfg.architecture == .gemma) {
                    if (lt.post_ffw_norm) |pfn_t| {
                        try self.dispatchRmsNorm(
                            scratch_down.handle,
                            scratch_down.size,
                            pfn_t.gpu_buffer.handle,
                            pfn_t.gpu_buffer.size,
                            scratch_down.handle,
                            scratch_down.size,
                            hidden_dim,
                            n_tokens,
                            eps,
                        );
                        self.decode_cmd.computeBarrier();
                    }
                }
                try self.dispatchScaleAcc(scratch_hidden.handle, scratch_hidden.size, scratch_down.handle, scratch_down.size, n_tokens * hidden_dim, 1.0);
            }
            self.decode_cmd.computeBarrier();

            // Gemma 4 per-layer output scale: hidden *= scale (applied to the
            // residual stream at the end of each layer). Skipped when the
            // scale is 1.0 — the common case and every non-Gemma layer.
            const layer_output_scale = self.layer_output_scales[layer_idx];
            if (layer_output_scale != 1.0) {
                try self.dispatchScaleInPlace(scratch_hidden.handle, scratch_hidden.size, n_tokens * hidden_dim, layer_output_scale);
                self.decode_cmd.computeBarrier();
            }
        }

        // Final RMS norm over all N tokens; LM head on the last one.
        const output_norm_t = self.tensor_map.get("output_norm.weight") orelse return error.TensorNotFound;
        const lm_head_t = self.tensor_map.get("output.weight") orelse self.tensor_map.get("token_embd.weight") orelse return error.TensorNotFound;
        try self.dispatchRmsNorm(scratch_hidden.handle, scratch_hidden.size, output_norm_t.gpu_buffer.handle, output_norm_t.gpu_buffer.size, scratch_norm.handle, scratch_norm.size, hidden_dim, n_tokens, eps);
        self.decode_cmd.computeBarrier();
        const x_offset_bytes: u32 = (n_tokens - 1) * hidden_dim * @sizeOf(f32);
        try self.dispatchDmmvInner(lm_head_t, scratch_norm, scratch_norm.size, self.logits_buf, cfg.vocab_size, hidden_dim, 0, x_offset_bytes, 0, 0);
        self.decode_cmd.computeBarrier();

        // GPU argmax path — sampleGreedy reads argmax_result_staging
        // unconditionally when the pipeline is loaded. prefillBatched
        // previously skipped this step, so the first post-prefill decode
        // sampled from a stale buffer and emitted garbage even though the
        // logits matched the per-token path bit-for-bit.
        const have_gpu_argmax = self.argmax.pipeline != null and self.argmax_descriptor_set != null;
        if (have_gpu_argmax) {
            try self.argmax.record(
                &self.decode_cmd,
                self.argmax_descriptor_set.?,
                cfg.vocab_size,
                self.argmax_phase0_workgroups,
            );
        }

        // Read logits and argmax result back for the sampler.
        const barrier = vk.c.VkMemoryBarrier{
            .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
        };
        vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, null, 0, null);
        const logits_size = @as(vk.c.VkDeviceSize, cfg.vocab_size) * @sizeOf(f32);
        const logits_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = logits_size };
        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.logits_buf.handle, self.logits_staging.handle, 1, &logits_region);
        if (have_gpu_argmax) {
            const token_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = @sizeOf(u32) };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.argmax_result_buf.handle, self.argmax_result_staging.handle, 1, &token_region);
        }

        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

        state.position = base_token + n_tokens;

        if (validate_mode) {
            // Snapshot batched logits, reset to a fresh request, replay the
            // per-token prefill, then diff the last-token logits.
            // prefillBatch skips the logits_buf → logits_staging copy when
            // GPU argmax is available (need_logits_readback gate), so force
            // the readback for the reference run by flipping the engine's
            // logits_readback flag around the prefillBatch call.
            const vocab = cfg.vocab_size;
            const batched_snapshot = try self.instance.allocator.alloc(f32, vocab);
            defer self.instance.allocator.free(batched_snapshot);
            const batched_logits: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
            @memcpy(batched_snapshot, batched_logits[0..vocab]);

            state.position = 0;
            state.generated_tokens.clearRetainingCapacity();
            const prev_readback = self.logits_readback_enabled;
            self.logits_readback_enabled = true;
            defer self.logits_readback_enabled = prev_readback;
            try self.prefillBatch(state, prompt_tokens);

            const ref_logits: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
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

    /// Process all prompt tokens through the full transformer to populate
    /// KV cache and SSM state. Each token runs through all 40 layers.
    pub fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {
        if (prompt_tokens.len == 0) return;

        const prompt_token_count: u32 = @intCast(@min(prompt_tokens.len, std.math.maxInt(u32)));
        const target_context_tokens = if (state.requested_context_tokens > 0)
            @max(state.requested_context_tokens, state.position +| prompt_token_count)
        else
            state.position +| prompt_token_count;

        if (state.position == 0 and state.generated_tokens.items.len == 0) {
            try self.resetRequestState(target_context_tokens);
        } else if (state.position > 0 and self.active_kv_page_ids == null) {
            return error.KvStateNotAvailable;
        } else {
            try self.ensureKvPagesForContext(target_context_tokens);
        }

        // Reset lightweight prefill timing so the caller can log per-prefill stats.
        self.prefill_token_samples = 0;
        self.prefill_cpu_embed_ns = 0;
        self.prefill_cpu_record_ns = 0;
        self.prefill_submit_wait_ns = 0;
        self.prefill_gpu_phase_ns = [_]u64{0} ** profile_phase_count;
        self.prefill_gpu_total_ns = 0;
        self.prefill_active = true;
        if (self.use_qwen36_dense_prefill_validate) {
            self.dense_prefill_validate_captured_tokens = 0;
        }
        if (self.use_qwen36_ssm_prefill_validate) {
            self.ssm_prefill_validate_captured_tokens = 0;
        }
        defer self.prefill_active = false;

        // Dequantize every prompt-token embedding row upfront into a single
        // host-mapped Vulkan staging buffer. decodeStep's layer-0 copy reads
        // from here with srcOffset = idx * hidden_size, and embedToken()
        // becomes a no-op during prefill — one bulk dequant pass replaces
        // 154 per-token CPU memcpy(cache → embed_staging) calls. The buffer
        // is grown on demand and reused across prefills.
        const hidden_dim = self.model.config.hidden_dim;
        const total_embed_bytes: u64 = @as(u64, hidden_dim) * @as(u64, prompt_tokens.len) * @sizeOf(f32);
        if (self.prefill_embed_big == null or self.prefill_embed_big_capacity_bytes < total_embed_bytes) {
            if (self.prefill_embed_big) |*b| b.deinit();
            self.prefill_embed_big = try Buffer.initStaging(self.instance, total_embed_bytes);
            self.prefill_embed_big_capacity_bytes = total_embed_bytes;
        }

        defer {
            self.prefill_embed_big_token_count = 0;
            self.prefill_embed_big_hidden = 0;
            self.prefill_current_token_idx = 0;
        }
        const big_f32: [*]f32 = @ptrCast(@alignCast(self.prefill_embed_big.?.mapped.?));
        {
            const embd = self.tensor_map.get("token_embd.weight") orelse {
                log.err("token_embd.weight not found", .{});
                return error.TensorNotFound;
            };
            const mmap = self.model.mmap_data orelse return error.NoMmapData;
            const data_start: usize = @intCast(self.model.gguf_file.tensor_data_offset + embd.info.offset);
            const vocab_last = self.model.config.vocab_size -| 1;
            const is_gemma = self.model.config.architecture == .gemma;
            const gemma_scale: f32 = if (is_gemma)
                @floatCast(@sqrt(@as(f64, @floatFromInt(hidden_dim))))
            else
                1.0;
            for (prompt_tokens, 0..) |tok, i| {
                const safe_id = @min(tok, vocab_last);
                const dst = big_f32[i * hidden_dim ..][0..hidden_dim];
                dequantRow(mmap[data_start..], safe_id, hidden_dim, embd.info.type_, dst);
                if (is_gemma) {
                    for (dst) |*v| v.* *= gemma_scale;
                }
            }
        }
        self.prefill_embed_big_hidden = hidden_dim;
        self.prefill_embed_big_token_count = @intCast(prompt_tokens.len);
        self.prefill_current_token_idx = 0;

        // Per-phase GPU timing during prefill costs ~3% throughput (thousands of
        // vkCmdWriteTimestamp calls + a blocking query readback per token) on
        // RDNA for the 35B flagship, so it is gated behind `ZINC_PREFILL_PROFILE=1`.
        // The CPU-side prefill profile line (embed/record/submit+wait) stays always
        // on — it has zero GPU cost. When the flag is set, the caller also gets
        // a per-phase breakdown (attn/moe/shared/ssm/tail) plus MoE and SSM
        // sub-phase drill-downs, which is exactly what effort-6 Step 2 needs.
        const profile_env = getenv("ZINC_PREFILL_PROFILE");
        const want_gpu_phases = profile_env != null and profile_env.?.len > 0 and !std.mem.eql(u8, profile_env.?, "0");
        const had_profile_pool = self.timestamp_query_pool != null;
        const profile_was_enabled = self.profile_enabled;
        const enable_gpu_phase_timing = had_profile_pool and want_gpu_phases;
        if (enable_gpu_phase_timing) self.profile_enabled = true;

        // Pipelined prefill: two-deep ping-pong between decode_cmd and
        // prefill_cmd_alt (plus their paired embed staging buffers). While the
        // GPU executes prompt token N, the CPU dequantizes and records prompt
        // token N+1 into the alt slot and fires another submit. We only
        // waitForCompletion() on a slot when its prior submit must drain before
        // the CPU reuses it.
        //
        // Gated off when:
        //   - profiling is on (needs synchronous timestamp readback per token)
        //   - push descriptors are unavailable (shared_pool reset would race
        //     with in-flight descriptor sets from the alt CB)
        //   - prompt is a single token (nothing to pipeline)
        //   - validation diagnostics are on (terminal token reads back the
        //     hidden state into embed_staging — mixing that with alt staging
        //     adds failure modes not worth the complexity)
        const can_pipeline = !enable_gpu_phase_timing and self.instance.push_descriptor_fn != null and prompt_tokens.len >= 2 and !self.validation_diagnostics_enabled;

        var primary_pending: bool = false;
        var alt_pending: bool = false;

        // Run each prompt token through the full transformer (same as decodeStep)
        // This populates KV cache and SSM state so the first decode token has context.
        for (prompt_tokens, 0..) |token_id, i| {
            const collect_output = i + 1 == prompt_tokens.len;
            const pipeline_this = can_pipeline and !collect_output;
            // decodeStep's layer-0 copy reads prefill_embed_big at offset
            // idx * hidden_size; set the index here so embedToken and that
            // copy both observe the same value for this prompt token.
            self.prefill_current_token_idx = @intCast(i);

            if (pipeline_this) {
                // Swap so self.decode_cmd / self.embed_staging now point at the
                // alt slot. Bring the pending-fence flags along with them.
                std.mem.swap(CommandBuffer, &self.decode_cmd, &self.prefill_cmd_alt);
                std.mem.swap(Buffer, &self.embed_staging, &self.prefill_embed_alt);
                std.mem.swap(bool, &primary_pending, &alt_pending);
                // The slot we just swapped into may have a pending submit from
                // two iterations back. Drain it before reusing the CB + staging.
                if (primary_pending) {
                    try self.decode_cmd.waitForCompletion();
                    primary_pending = false;
                }
                self.prefill_pipeline_mode = true;
            } else {
                // Terminal token (or non-pipelined fallback): drain any
                // pending submits so the terminal CB sees a quiesced queue and
                // so the alt slot's KV/SSM writes are visible to the GPU's
                // subsequent work.
                if (alt_pending) {
                    try self.prefill_cmd_alt.waitForCompletion();
                    alt_pending = false;
                }
                if (primary_pending) {
                    try self.decode_cmd.waitForCompletion();
                    primary_pending = false;
                }
                self.prefill_pipeline_mode = false;
            }

            try self.decodeStep(state, token_id, collect_output);

            if (pipeline_this) {
                // decodeStep submitted self.decode_cmd without waiting.
                primary_pending = true;
            }
        }
        self.prefill_pipeline_mode = false;

        // Safety net: if any slot is still pending (shouldn't happen because
        // the terminal token always drains), wait on it here.
        if (alt_pending) {
            try self.prefill_cmd_alt.waitForCompletion();
            alt_pending = false;
        }
        if (primary_pending) {
            try self.decode_cmd.waitForCompletion();
            primary_pending = false;
        }

        // Effort-6 Step 5 prerequisite: per-(layer, expert) routing counts.
        // After all prefill tokens have written their routing into
        // routing_capture_buf, dispatch count_experts once per layer to
        // populate prefill_expert_count_buf[layer * n_experts + expert].
        // The decode_cmd is idle now (drained above), so we reuse it for the
        // sweep. mul_mm_id_q4k consumes data_expert_count for early-exit; the
        // wire-in into a batched MoE FFN dispatch follows in a later cycle.
        if (self.use_count_experts_prefill and
            self.prefill_expert_count_buf.handle != null and
            self.routing_capture_buf.handle != null and
            self.dmmv.pipeline_count_experts != null)
        {
            const cfg = self.model.config;
            const n_tokens_capped: u32 = @intCast(@min(
                @as(usize, prompt_tokens.len),
                @as(usize, self.routing_capture_max_tokens),
            ));
            if (n_tokens_capped > 0 and cfg.n_experts_used > 0 and cfg.n_experts > 0) {
                try self.decode_cmd.reset();
                try self.decode_cmd.begin();
                // The routing buffer was written via vkCmdCopyBuffer (transfer
                // writes) in the prior submissions. Even though those CBs have
                // drained on the host, we still need a transfer→compute barrier
                // inside this CB for the count_experts shader to observe the
                // memory writes under the explicit-spec memory model.
                self.decode_cmd.transferToComputeBarrier();
                var layer: u32 = 0;
                while (layer < cfg.n_layers) : (layer += 1) {
                    const d_off: vk.c.VkDeviceSize =
                        @as(vk.c.VkDeviceSize, layer) *
                        @as(vk.c.VkDeviceSize, cfg.n_experts) *
                        @sizeOf(u32);
                    self.dmmv.recordCountExperts(
                        &self.decode_cmd,
                        self.instance.push_descriptor_fn,
                        self.routing_capture_buf.handle,
                        self.routing_capture_buf.size,
                        self.prefill_expert_count_buf.handle,
                        self.prefill_expert_count_buf.size,
                        n_tokens_capped,
                        cfg.n_layers,
                        layer,
                        cfg.n_experts_used,
                        cfg.n_experts,
                        d_off,
                    ) catch |err| {
                        log.warn("count_experts dispatch failed at layer {d}: {s}", .{ layer, @errorName(err) });
                        break;
                    };
                }
                try self.decode_cmd.end();
                try self.decode_cmd.submitAndWait(self.instance.compute_queue);
            }
        }

        if (self.use_qwen36_dense_prefill_validate and self.dense_prefill_validate_captured_tokens > 0) {
            const n_validate = @min(self.dense_prefill_validate_captured_tokens, self.dense_prefill_validate_max_tokens);
            const validate_chunks = [_]u32{ 4, 8, 16 };
            var ran_exact_chunk = false;
            for (validate_chunks) |chunk| {
                if (chunk > n_validate) continue;
                if (chunk == n_validate) ran_exact_chunk = true;
                self.validateDensePrefillFfnChunk(chunk) catch |err| {
                    log.warn("ZINC_QWEN36_27B_PREFILL_VALIDATE: dense FFN replay chunk={d} skipped: {s}", .{ chunk, @errorName(err) });
                };
            }
            if (!ran_exact_chunk) {
                self.validateDensePrefillFfnChunk(n_validate) catch |err| {
                    log.warn("ZINC_QWEN36_27B_PREFILL_VALIDATE: dense FFN replay chunk={d} skipped: {s}", .{ n_validate, @errorName(err) });
                };
            }
        }
        if (self.use_qwen36_ssm_prefill_validate and self.ssm_prefill_validate_captured_tokens > 0) {
            const n_validate = @min(self.ssm_prefill_validate_captured_tokens, self.dense_prefill_validate_max_tokens);
            const validate_chunks = [_]u32{ 4, 8, 16 };
            var ran_exact_chunk = false;
            for (validate_chunks) |chunk| {
                if (chunk > n_validate) continue;
                if (chunk == n_validate) ran_exact_chunk = true;
                self.validateSsmPrefillProjectionChunk(chunk) catch |err| {
                    log.warn("ZINC_QWEN36_27B_PREFILL_VALIDATE: SSM projection replay chunk={d} skipped: {s}", .{ chunk, @errorName(err) });
                };
            }
            if (!ran_exact_chunk) {
                self.validateSsmPrefillProjectionChunk(n_validate) catch |err| {
                    log.warn("ZINC_QWEN36_27B_PREFILL_VALIDATE: SSM projection replay chunk={d} skipped: {s}", .{ n_validate, @errorName(err) });
                };
            }
        }

        // Effort-6 cycle 123 (A3b all-layer extension): batched
        // ssm_delta_net dispatches for ALL SSM layers with n_tok=prompt_len.
        // Each SSM layer's state is backed up to a per-layer slot in
        // a3b_state_backup, zeroed, dispatched with the per-(layer, token)
        // strided captures, then restored. Per-layer batched output is
        // diffed against the per-token reference captured in runSsmLayerGpu
        // (also strided per-(layer, token)). Cycle 97/101/104 only dispatched
        // layer 0; this extension validates the n_tok>1 shader correctness
        // across every SSM layer in the model — the precondition for
        // production wire-up. Flag-OFF path (default) is unchanged from
        // production.
        if (self.use_a3b_validate and
            self.a3b_alpha_capture != null and
            self.a3b_beta_capture != null and
            self.a3b_conv_out_capture != null and
            self.a3b_state_backup != null and
            self.a3b_delta_out != null and
            self.elementwise.pipeline_ssm_delta_net != null and
            self.instance.push_descriptor_fn != null and
            prompt_tokens.len > 0 and
            prompt_tokens.len <= self.a3b_capture_max_tokens)
        {
            const cfg_a3b = &self.model.config;
            if (cfg_a3b.ssm_d_inner > 0 and cfg_a3b.ssm_dt_rank > 0 and self.gpu_ssm_states.len > 0) {
                const dt_rank_a3b = cfg_a3b.ssm_dt_rank;
                const d_inner_a3b = cfg_a3b.ssm_d_inner;
                const head_v_dim_a3b = d_inner_a3b / dt_rank_a3b;
                const conv_ch_a3b = d_inner_a3b + 2 * cfg_a3b.ssm_n_group * cfg_a3b.ssm_d_state;
                const state_elems_a3b = dt_rank_a3b * head_v_dim_a3b * head_v_dim_a3b;
                const state_bytes_a3b: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, state_elems_a3b) * @sizeOf(f32);
                const n_tok_a3b: u32 = @intCast(prompt_tokens.len);
                const max_tokens_a3b: vk.c.VkDeviceSize = @intCast(self.a3b_capture_max_tokens);
                const full_attn_interval_a3b: u32 = if (cfg_a3b.full_attn_interval > 0) cfg_a3b.full_attn_interval else 1;

                // Per-layer offsets into the (layer, token, data) capture
                // buffers. Each layer's slot is a contiguous max_tokens ×
                // per_data block within the buffer; the per-token loop above
                // already wrote each (layer, token) slot at this stride.
                const ab_layer_stride: vk.c.VkDeviceSize = max_tokens_a3b * @as(vk.c.VkDeviceSize, dt_rank_a3b) * @sizeOf(f32);
                const conv_layer_stride: vk.c.VkDeviceSize = max_tokens_a3b * @as(vk.c.VkDeviceSize, conv_ch_a3b) * @sizeOf(f32);
                const out_layer_stride: vk.c.VkDeviceSize = max_tokens_a3b * @as(vk.c.VkDeviceSize, d_inner_a3b) * @sizeOf(f32);
                const conv_total_layer: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, n_tok_a3b) * @as(vk.c.VkDeviceSize, conv_ch_a3b) * @sizeOf(f32);
                const ab_total_layer: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, n_tok_a3b) * @as(vk.c.VkDeviceSize, dt_rank_a3b) * @sizeOf(f32);
                const out_total_layer: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, n_tok_a3b) * @as(vk.c.VkDeviceSize, d_inner_a3b) * @sizeOf(f32);

                // Sample 16 floats × 3 tokens × 2 sources per layer.
                const sample_floats: u32 = 16;
                const sample_bytes: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, sample_floats) * @sizeOf(f32);
                const samples_per_layer_bytes: vk.c.VkDeviceSize = sample_bytes * 6;
                const tok_first: u32 = 0;
                const tok_mid: u32 = if (n_tok_a3b > 2) n_tok_a3b / 2 else 0;
                const tok_last: u32 = n_tok_a3b - 1;

                // Build the SSM-layer index list once: layers where
                // ((L+1) % full_attn_interval) != 0. Each entry is the
                // absolute layer index used to compute state/capture offsets.
                var ssm_layer_indices_buf: [128]u32 = undefined;
                var ssm_count: usize = 0;
                {
                    var L: u32 = 0;
                    while (L < cfg_a3b.n_layers and ssm_count < ssm_layer_indices_buf.len) : (L += 1) {
                        const is_full_attn = ((L + 1) % full_attn_interval_a3b) == 0;
                        if (!is_full_attn and L < self.gpu_ssm_states.len and self.gpu_ssm_states[L].handle != null) {
                            ssm_layer_indices_buf[ssm_count] = L;
                            ssm_count += 1;
                        }
                    }
                }
                const ssm_layers = ssm_layer_indices_buf[0..ssm_count];

                const sample_total_bytes: vk.c.VkDeviceSize = samples_per_layer_bytes * @as(vk.c.VkDeviceSize, ssm_count);
                const can_sample = self.a3b_per_token_delta_out != null and
                    self.logits_staging.size >= sample_total_bytes and
                    self.logits_staging.mapped != null and
                    @as(vk.c.VkDeviceSize, d_inner_a3b) >= @as(vk.c.VkDeviceSize, sample_floats);

                if (ssm_count > 0) {
                    try self.decode_cmd.reset();
                    try self.decode_cmd.begin();

                    // Phase 1: backup all SSM layers' states into per-layer
                    // slots in a3b_state_backup. Use a single
                    // compute→transfer barrier covering all layers since the
                    // per-token CBs have already drained on the host above.
                    self.decode_cmd.computeToTransferBarrier();
                    for (ssm_layers, 0..) |L, ssm_idx| {
                        const backup_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, ssm_idx) * state_bytes_a3b;
                        const r_backup = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = backup_off, .size = state_bytes_a3b };
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.gpu_ssm_states[L].handle, self.a3b_state_backup.?.handle, 1, &r_backup);
                    }

                    // Phase 2: zero all SSM layers' states. Order after
                    // backup reads via transfer→transfer barrier.
                    self.decode_cmd.transferToTransferBarrier();
                    for (ssm_layers) |L| {
                        vk.c.vkCmdFillBuffer(self.decode_cmd.handle, self.gpu_ssm_states[L].handle, 0, state_bytes_a3b, 0);
                    }

                    // Phase 3: dispatch batched delta_net for each SSM
                    // layer. Each dispatch reads the layer's slot in the
                    // capture buffers and writes to the layer's slot in
                    // a3b_delta_out. State buffer is the actual per-layer
                    // gpu_ssm_states[L] (now zeroed). Single compute barrier
                    // between dispatches because the next layer's dispatch
                    // doesn't share buffers with the current one (different
                    // state buf, different capture-buffer slot).
                    self.decode_cmd.transferToComputeBarrier();
                    const pip = &(self.elementwise.pipeline_ssm_delta_net.?);
                    for (ssm_layers) |L| {
                        const lt_a3b = self.layer_tensors[L];
                        const dt_bias_t = lt_a3b.ssm_dt_bias;
                        const ssm_a_t = lt_a3b.ssm_a;
                        const dt_bias_buf = if (dt_bias_t) |t| t.gpu_buffer.handle else self.down_buf.handle;
                        const dt_bias_size = if (dt_bias_t) |t| t.gpu_buffer.size else (@as(vk.c.VkDeviceSize, dt_rank_a3b) * @sizeOf(f32));
                        const ssm_a_buf = if (ssm_a_t) |t| t.gpu_buffer.handle else self.down_buf.handle;
                        const ssm_a_size = if (ssm_a_t) |t| t.gpu_buffer.size else (@as(vk.c.VkDeviceSize, dt_rank_a3b) * @sizeOf(f32));

                        const push = SsmDeltaNetPush{
                            .d_inner = d_inner_a3b,
                            .dt_rank = dt_rank_a3b,
                            .head_v_dim = head_v_dim_a3b,
                            .d_state = cfg_a3b.ssm_d_state,
                            .n_group = cfg_a3b.ssm_n_group,
                            .ssm_a_is_f16 = if (ssm_a_t) |t| (if (t.info.type_ == .f16) @as(u32, 1) else 0) else 0,
                            .dt_bias_is_f16 = if (dt_bias_t) |t| (if (t.info.type_ == .f16) @as(u32, 1) else 0) else 0,
                            .has_dt_bias = if (dt_bias_t != null) 1 else 0,
                            .has_ssm_a = if (ssm_a_t != null) 1 else 0,
                            .n_tok = n_tok_a3b,
                            .conv_stride_tok = conv_ch_a3b,
                            .ab_stride_tok = dt_rank_a3b,
                            .y_stride_tok = d_inner_a3b,
                        };

                        const conv_layer_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, L) * conv_layer_stride;
                        const ab_layer_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, L) * ab_layer_stride;
                        const out_layer_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, L) * out_layer_stride;

                        const infos = [7]vk.c.VkDescriptorBufferInfo{
                            .{ .buffer = self.a3b_conv_out_capture.?.handle, .offset = conv_layer_off, .range = conv_total_layer },
                            .{ .buffer = dt_bias_buf, .offset = 0, .range = dt_bias_size },
                            .{ .buffer = self.a3b_alpha_capture.?.handle, .offset = ab_layer_off, .range = ab_total_layer },
                            .{ .buffer = self.a3b_beta_capture.?.handle, .offset = ab_layer_off, .range = ab_total_layer },
                            .{ .buffer = ssm_a_buf, .offset = 0, .range = ssm_a_size },
                            .{ .buffer = self.gpu_ssm_states[L].handle, .offset = 0, .range = state_bytes_a3b },
                            .{ .buffer = self.a3b_delta_out.?.handle, .offset = out_layer_off, .range = out_total_layer },
                        };
                        self.decode_cmd.pushDescAndDispatch(
                            pip,
                            self.instance.push_descriptor_fn,
                            infos[0..],
                            std.mem.asBytes(&push),
                            dt_rank_a3b,
                            head_v_dim_a3b,
                            1,
                        );
                        // Compute barrier between dispatches so successive
                        // layer dispatches see clean state writes (they
                        // touch independent buffers but the driver may
                        // serialize compute-to-compute writes via a barrier).
                        self.decode_cmd.computeBarrier();
                    }

                    // Phase 4: restore each SSM layer's state from backup.
                    // Compute→transfer barrier covers all dispatches above.
                    self.decode_cmd.computeToTransferBarrier();
                    for (ssm_layers, 0..) |L, ssm_idx| {
                        const backup_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, ssm_idx) * state_bytes_a3b;
                        const r_restore = vk.c.VkBufferCopy{ .srcOffset = backup_off, .dstOffset = 0, .size = state_bytes_a3b };
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.a3b_state_backup.?.handle, self.gpu_ssm_states[L].handle, 1, &r_restore);
                    }

                    // Phase 5: per-layer sample copies into logits_staging.
                    // Layout per layer: 16 floats × 6 (b/p × first/mid/last).
                    if (can_sample) {
                        const tokens = [_]u32{ tok_first, tok_mid, tok_last };
                        var staging_off: vk.c.VkDeviceSize = 0;
                        for (ssm_layers) |L| {
                            const layer_out_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, L) * out_layer_stride;
                            for (tokens) |tok| {
                                const tok_slot_off: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, tok) * @as(vk.c.VkDeviceSize, d_inner_a3b) * @sizeOf(f32);
                                const r_b = vk.c.VkBufferCopy{ .srcOffset = layer_out_off + tok_slot_off, .dstOffset = staging_off, .size = sample_bytes };
                                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.a3b_delta_out.?.handle, self.logits_staging.handle, 1, &r_b);
                                staging_off += sample_bytes;
                                const r_p = vk.c.VkBufferCopy{ .srcOffset = layer_out_off + tok_slot_off, .dstOffset = staging_off, .size = sample_bytes };
                                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.a3b_per_token_delta_out.?.handle, self.logits_staging.handle, 1, &r_p);
                                staging_off += sample_bytes;
                            }
                        }
                    }
                    try self.decode_cmd.end();
                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                    // CPU-side: read back samples and compute per-layer verdict.
                    if (can_sample) {
                        const base: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
                        const labels = [_][]const u8{ "first", "mid", "last" };
                        var max_diff_overall: f64 = 0.0;
                        var pass_layers: u32 = 0;
                        var pass_e2_layers: u32 = 0;
                        var fail_layers: u32 = 0;
                        for (ssm_layers, 0..) |L, ssm_idx| {
                            const layer_base: usize = ssm_idx * 6 * sample_floats;
                            var max_d_layer: f64 = 0.0;
                            for (0..3) |slot| {
                                const off_b = layer_base + slot * 2 * sample_floats;
                                const off_p = off_b + sample_floats;
                                var max_d: f64 = 0.0;
                                for (0..sample_floats) |k| {
                                    const b: f64 = base[off_b + k];
                                    const p: f64 = base[off_p + k];
                                    const d = b - p;
                                    const ad = if (d < 0) -d else d;
                                    if (ad > max_d) max_d = ad;
                                }
                                if (max_d > max_d_layer) max_d_layer = max_d;
                                if (max_d > max_diff_overall) max_diff_overall = max_d;
                                _ = labels;
                            }
                            const layer_verdict: []const u8 = if (max_d_layer < 1e-3) "PASS@1e-3" else if (max_d_layer < 1e-2) "PASS@1e-2" else "FAIL";
                            if (max_d_layer < 1e-3) {
                                pass_layers += 1;
                            } else if (max_d_layer < 1e-2) {
                                pass_e2_layers += 1;
                            } else {
                                fail_layers += 1;
                            }
                            log.info("ZINC_A3B_VALIDATE: layer={d} verdict={s} max_abs_diff={e:.6} (3 sample tokens × 16 floats)", .{
                                L, layer_verdict, max_d_layer,
                            });
                        }
                        const overall_verdict: []const u8 = if (fail_layers > 0) "FAIL" else if (pass_e2_layers > 0) "PASS@1e-2" else "PASS@1e-3";
                        log.info("ZINC_A3B_VALIDATE: ALL-LAYER batched vs per-token verdict={s} (max_abs_diff={e:.6}, n_tok={d}, ssm_layers={d}, pass_1e-3={d} pass_1e-2={d} fail={d})", .{
                            overall_verdict, max_diff_overall, n_tok_a3b, ssm_count, pass_layers, pass_e2_layers, fail_layers,
                        });
                    } else {
                        log.info("ZINC_A3B_VALIDATE: ALL-LAYER batched ssm_delta_net dispatched (n_tok={d}, ssm_layers={d}; sample skipped — no per-token capture or staging too small {d} < {d})", .{
                            n_tok_a3b, ssm_count, self.logits_staging.size, sample_total_bytes,
                        });
                    }
                }
            }
        }

        // Cycle 127 (A3b production rollback): cycle 125's production-mode
        // post-loop batched dispatch was removed. With cycle 125's
        // destructive runSsmLayerGpu skip in place, hidden_buf evolution
        // through layers was corrupted — meaning the per-(layer, token)
        // alpha/beta/conv_out captures fed to this batched dispatch were
        // themselves wrong, so the post-loop dispatch produced wrong
        // state. The whole production path was interlocked-broken. The
        // capture buffers and the validate path stay intact. ZINC_A3B_
        // PRODUCTION is now a no-op; cycle 128+ will re-engage it via
        // the layer-major restructure documented at the runSsmLayerGpu
        // skip site (forward.zig ~9129) and the 2026-05-06 pivot.

        if (enable_gpu_phase_timing) {
            // Snapshot accumulated per-phase GPU time into prefill-scoped fields
            // before wiping the decode-oriented sample state.
            for (0..profile_phase_count) |p| {
                self.prefill_gpu_phase_ns[p] = self.profile_total_counters.gpu_phase_ns[p];
            }
            self.prefill_gpu_total_ns = @intFromFloat(@max(self.profile_total_gpu_ms * 1_000_000.0, 0.0));
            self.resetProfilingSamples();
            self.profile_enabled = profile_was_enabled;
        }
    }

    // -----------------------------------------------------------------------
    // Sampling
    // -----------------------------------------------------------------------

    fn tokenSeen(history: []const u32, token: u32) bool {
        for (history) |seen| {
            if (seen == token) return true;
        }
        return false;
    }

    fn softcapLogit(logit: f32, softcap: f32) f32 {
        if (!(softcap > 0)) return logit;
        return softcap * std.math.tanh(logit / softcap);
    }

    fn adjustedLogit(logit: f32, token: u32, history: []const u32, repetition_penalty: f32) f32 {
        if (repetition_penalty <= 1.0001 or !tokenSeen(history, token)) return logit;
        if (logit >= 0) return logit / repetition_penalty;
        return logit * repetition_penalty;
    }

    fn argmaxFromLogits(logits: []const f32, history: []const u32, repetition_penalty: f32, final_logit_softcapping: f32) u32 {
        if (logits.len == 0) return 0;
        var best_idx: u32 = 0;
        var best_val = adjustedLogit(softcapLogit(logits[0], final_logit_softcapping), 0, history, repetition_penalty);
        for (logits[1..], 1..) |raw_val, i| {
            const val = adjustedLogit(softcapLogit(raw_val, final_logit_softcapping), @intCast(i), history, repetition_penalty);
            if (val > best_val) {
                best_val = val;
                best_idx = @intCast(i);
            }
        }
        return best_idx;
    }

    fn sampleFromLogits(logits: []const f32, history: []const u32, params: SamplingParams, random: std.Random, final_logit_softcapping: f32) u32 {
        if (logits.len == 0) return 0;
        if (!params.requiresLogitsReadback()) return argmaxFromLogits(logits, history, 1.0, final_logit_softcapping);
        if (params.temperature <= 0.0001) {
            return argmaxFromLogits(logits, history, params.repetition_penalty, final_logit_softcapping);
        }

        const max_candidates = 128;
        const top_k: usize = @min(@max(params.top_k, 1), max_candidates);
        const safe_top_p = std.math.clamp(params.top_p, 0.0, 1.0);
        const temperature = @max(params.temperature, 0.0001);

        var candidate_ids: [max_candidates]u32 = undefined;
        var candidate_logits: [max_candidates]f32 = undefined;
        var candidate_count: usize = 0;

        for (logits, 0..) |raw_val, i| {
            if (!std.math.isFinite(raw_val)) continue;
            const token_id: u32 = @intCast(i);
            const val = adjustedLogit(softcapLogit(raw_val, final_logit_softcapping), token_id, history, params.repetition_penalty);

            var insert_at = candidate_count;
            while (insert_at > 0 and val > candidate_logits[insert_at - 1]) : (insert_at -= 1) {}
            if (insert_at >= top_k) continue;

            if (candidate_count < top_k) {
                candidate_count += 1;
            }

            var j = candidate_count - 1;
            while (j > insert_at) : (j -= 1) {
                candidate_ids[j] = candidate_ids[j - 1];
                candidate_logits[j] = candidate_logits[j - 1];
            }
            candidate_ids[insert_at] = token_id;
            candidate_logits[insert_at] = val;
        }

        if (candidate_count == 0) return 0;
        if (candidate_count == 1) return candidate_ids[0];

        var weights: [max_candidates]f64 = undefined;
        const max_logit = @as(f64, candidate_logits[0]) / @as(f64, temperature);
        var total_weight: f64 = 0.0;
        for (0..candidate_count) |i| {
            const scaled = @as(f64, candidate_logits[i]) / @as(f64, temperature);
            const weight = @exp(scaled - max_logit);
            weights[i] = weight;
            total_weight += weight;
        }
        if (!(total_weight > 0.0) or !std.math.isFinite(total_weight)) return candidate_ids[0];

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
        if (!(kept_weight > 0.0) or !std.math.isFinite(kept_weight)) return candidate_ids[0];

        const target = random.float(f64) * kept_weight;
        var cumulative: f64 = 0.0;
        for (0..keep_count) |i| {
            cumulative += weights[i];
            if (target <= cumulative) return candidate_ids[i];
        }

        return candidate_ids[keep_count - 1];
    }

    /// Sample a token greedily. Uses GPU argmax when available, otherwise falls back to CPU scan.
    pub fn sampleGreedy(self: *const InferenceEngine) u32 {
        if (self.argmax.pipeline != null and self.argmax_descriptor_set != null) {
            const token_ptr: [*]const u32 = @ptrCast(@alignCast(self.argmax_result_staging.mapped.?));
            return token_ptr[0];
        }

        const vocab_size = self.model.config.vocab_size;
        const logits_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
        const logits = logits_ptr[0..vocab_size];

        var max_val: f32 = logits[0];
        var max_idx: u32 = 0;
        for (logits[1..], 1..) |val, i| {
            if (val > max_val) {
                max_val = val;
                max_idx = @intCast(i);
            }
        }
        return max_idx;
    }

    /// Sample a token using either the GPU argmax fast path or host logits sampling.
    pub fn sample(self: *const InferenceEngine, state: *const DecodeState, params: SamplingParams, random: std.Random) u32 {
        if (!params.requiresLogitsReadback()) return self.sampleGreedy();

        const vocab_size = self.model.config.vocab_size;
        const logits_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
        const logits = logits_ptr[0..vocab_size];
        return sampleFromLogits(logits, state.generated_tokens.items, params, random, self.model.config.final_logit_softcapping);
    }

    // -----------------------------------------------------------------------
    // Diagnostics
    // -----------------------------------------------------------------------

    /// One-shot diagnostic: embed → output_norm → LM_head on CPU+GPU (no transformer layers).
    /// Split into 3 GPU submissions with intermediate readbacks to pinpoint divergence.
    fn diagEmbedToLogits(self: *InferenceEngine, bos_token: u32) !void {
        const dlog = std.log.scoped(.diag);
        const config = &self.model.config;
        const hidden_dim = config.hidden_dim;
        const hidden_size = @as(vk.c.VkDeviceSize, hidden_dim) * @sizeOf(f32);

        dlog.info("=== DIAG: embed->norm->LM_head BOS={d} hidden={d} vocab={d} ===", .{
            bos_token, hidden_dim, config.vocab_size,
        });

        if (hidden_dim > 8192) {
            dlog.warn("hidden_dim {d} > 8192, skipping diagnostic", .{hidden_dim});
            return;
        }

        const mmap = self.model.mmap_data orelse return;

        // ── CPU reference ──
        // 1. Dequantize BOS embedding
        const embd_t = self.tensor_map.get("token_embd.weight") orelse return;
        const embd_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + embd_t.info.offset);
        var cpu_embed_buf: [8192]f32 = undefined;
        const cpu_embed = cpu_embed_buf[0..hidden_dim];
        dequantRow(mmap[embd_off..], bos_token, hidden_dim, embd_t.info.type_, cpu_embed);

        // 2. CPU RMS norm with output_norm.weight
        const norm_t = self.tensor_map.get("output_norm.weight") orelse return;
        const norm_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + norm_t.info.offset);
        var cpu_nw_buf: [8192]f32 = undefined;
        const cpu_nw = cpu_nw_buf[0..hidden_dim];
        dequantRow(mmap[norm_off..], 0, hidden_dim, norm_t.info.type_, cpu_nw);

        var sum_sq: f64 = 0.0;
        for (cpu_embed) |v| sum_sq += @as(f64, v) * @as(f64, v);
        const rms_inv: f32 = @floatCast(1.0 / @sqrt(sum_sq / @as(f64, @floatFromInt(hidden_dim)) + config.rms_norm_eps));

        var cpu_normed_buf: [8192]f32 = undefined;
        const cpu_normed = cpu_normed_buf[0..hidden_dim];
        for (0..hidden_dim) |i| cpu_normed[i] = cpu_nw[i] * (cpu_embed[i] * rms_inv);

        // 3. CPU dot products for first 10 logits
        const lm_t = self.tensor_map.get("output.weight") orelse
            self.tensor_map.get("token_embd.weight") orelse return;
        const lm_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + lm_t.info.offset);
        const lm_data = mmap[lm_off..];

        // Log tensor types — critical for detecting format mismatches
        dlog.info("TENSOR TYPES: embd={s} norm={s}(buf={d}B,expect={d}B) lm={s} lm_name={s}", .{
            @tagName(embd_t.info.type_),
            @tagName(norm_t.info.type_),
            norm_t.gpu_buffer.size,
            hidden_size,
            @tagName(lm_t.info.type_),
            lm_t.info.name,
        });

        // CRITICAL CHECK: norm weights must be f32 for rms_norm shader
        if (norm_t.info.type_ != .f32) {
            dlog.err("BUG: output_norm.weight is {s} but rms_norm shader reads as float[]!", .{
                @tagName(norm_t.info.type_),
            });
            dlog.err("GPU buffer has {d} bytes but shader reads {d} bytes (hidden_dim*4)", .{
                norm_t.gpu_buffer.size, hidden_size,
            });
        }

        var cpu_logits: [10]f32 = undefined;
        var cpu_row_buf: [8192]f32 = undefined;
        for (0..10) |row| {
            dequantRow(lm_data, @intCast(row), hidden_dim, lm_t.info.type_, cpu_row_buf[0..hidden_dim]);
            var dot: f64 = 0.0;
            for (0..hidden_dim) |i| dot += @as(f64, cpu_row_buf[i]) * @as(f64, cpu_normed[i]);
            cpu_logits[row] = @floatCast(dot);
        }

        // ── STAGE 1: GPU embedding upload + readback hidden_buf ──
        try self.embedToken(bos_token);

        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.begin();
        {
            const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = hidden_size };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.embed_staging.handle, self.hidden_buf.handle, 1, &region);
        }
        // Barrier: transfer write → transfer read (for readback)
        {
            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
            };
            vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, null, 0, null);
        }
        // Readback hidden_buf → logits_staging
        {
            const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = hidden_size };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.hidden_buf.handle, self.logits_staging.handle, 1, &region);
        }
        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

        // Compare GPU embed vs CPU embed
        const gpu_e: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
        var embed_max_diff: f32 = 0.0;
        for (0..hidden_dim) |i| {
            const d = @abs(gpu_e[i] - cpu_embed[i]);
            if (d > embed_max_diff) embed_max_diff = d;
        }
        dlog.info("EMBED: CPU[0..3]={d:.6},{d:.6},{d:.6},{d:.6} GPU[0..3]={d:.6},{d:.6},{d:.6},{d:.6} max_diff={d:.9}", .{
            cpu_embed[0],   cpu_embed[1], cpu_embed[2], cpu_embed[3],
            gpu_e[0],       gpu_e[1],     gpu_e[2],     gpu_e[3],
            embed_max_diff,
        });

        // ── STAGE 2: RMS norm → readback norm_buf ──
        // hidden_buf still has the embedding (only read in stage 1 readback)
        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.begin();
        try self.dispatchRmsNorm(
            self.hidden_buf.handle,
            hidden_size,
            norm_t.gpu_buffer.handle,
            norm_t.gpu_buffer.size,
            self.norm_buf.handle,
            hidden_size,
            hidden_dim,
            1,
            config.rms_norm_eps,
        );
        // Barrier: shader write → transfer read
        {
            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
            };
            vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, null, 0, null);
        }
        // Readback norm_buf → logits_staging
        {
            const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = hidden_size };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.norm_buf.handle, self.logits_staging.handle, 1, &region);
        }
        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

        // Compare GPU norm vs CPU norm
        const gpu_n: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
        var norm_max_diff: f32 = 0.0;
        for (0..hidden_dim) |i| {
            const d = @abs(gpu_n[i] - cpu_normed[i]);
            if (d > norm_max_diff) norm_max_diff = d;
        }
        dlog.info("NORM: CPU[0..3]={d:.6},{d:.6},{d:.6},{d:.6} GPU[0..3]={d:.6},{d:.6},{d:.6},{d:.6} max_diff={d:.9}", .{
            cpu_normed[0], cpu_normed[1], cpu_normed[2], cpu_normed[3],
            gpu_n[0],      gpu_n[1],      gpu_n[2],      gpu_n[3],
            norm_max_diff,
        });

        // ── STAGE 3: LM head DMMV → readback logits ──
        // norm_buf still has the norm output (only read in stage 2 readback)
        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.begin();
        try self.dispatchDmmv(lm_t, self.norm_buf, hidden_size, self.logits_buf, config.vocab_size, hidden_dim);
        {
            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
            };
            vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &barrier, 0, null, 0, null);
            const logits_copy = @as(vk.c.VkDeviceSize, config.vocab_size) * @sizeOf(f32);
            const copy_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = logits_copy };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.logits_buf.handle, self.logits_staging.handle, 1, &copy_region);
        }
        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

        // ── Compare GPU vs CPU logits ──
        const gpu: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
        for (0..10) |i| {
            dlog.info("logit[{d}]: CPU={d:.6} GPU={d:.6} diff={d:.6}", .{
                i, cpu_logits[i], gpu[i], gpu[i] - cpu_logits[i],
            });
        }

        // GPU argmax
        var gpu_max: f32 = gpu[0];
        var gpu_max_idx: u32 = 0;
        for (1..config.vocab_size) |i| {
            if (gpu[i] > gpu_max) {
                gpu_max = gpu[i];
                gpu_max_idx = @intCast(i);
            }
        }

        dlog.info("SUMMARY: embed_ok={s} norm_ok={s} GPU_argmax={d}({d:.4}) CPU_logit5={d:.4} GPU_logit5={d:.4}", .{
            if (embed_max_diff < 0.001) "YES" else "NO",
            if (norm_max_diff < 0.01) "YES" else "NO",
            gpu_max_idx,
            gpu_max,
            cpu_logits[5],
            gpu[5],
        });

        // ── STAGE 4: Verify DMMV for non-Q8_0 quant types ──
        // norm_buf still has BOS embedding norm from STAGE 2 (STAGE 3 only read it)
        const lt0_diag = self.layer_tensors[0];
        const wqkv_diag = lt0_diag.attn_qkv;
        const ffn_gate_diag = lt0_diag.ffn_gate;
        const ffn_up_diag = lt0_diag.ffn_up;
        const gate_exps_diag = lt0_diag.ffn_gate_exps;
        const down_exps_diag = lt0_diag.ffn_down_exps;
        const ssm_out_diag = lt0_diag.ssm_out;
        const attn_q_diag = if (self.layer_tensors.len > 3) self.layer_tensors[3].attn_q else null; // layer 3 = first attn layer
        dlog.info("QUANT: wqkv={s} ffn_gate={s} ffn_up={s} gate_exps={s} down_exps={s} ssm_out={s} attn_q={s}", .{
            if (wqkv_diag) |t| @tagName(t.info.type_) else "N/A",
            if (ffn_gate_diag) |t| @tagName(t.info.type_) else "N/A",
            if (ffn_up_diag) |t| @tagName(t.info.type_) else "N/A",
            if (gate_exps_diag) |t| @tagName(t.info.type_) else "N/A",
            if (down_exps_diag) |t| @tagName(t.info.type_) else "N/A",
            if (ssm_out_diag) |t| @tagName(t.info.type_) else "N/A",
            if (attn_q_diag) |t| @tagName(t.info.type_) else "N/A",
        });

        if (wqkv_diag) |wt| {
            const d_inner_d = config.ssm_d_inner;
            const n_grp_d = config.ssm_n_group;
            const d_state_d = config.ssm_d_state;
            const conv_ch: u32 = @intCast(d_inner_d + 2 * n_grp_d * d_state_d);
            const wqkv_off_d: usize = @intCast(self.model.gguf_file.tensor_data_offset + wt.info.offset);

            // CPU: dot products for first 5 rows of wqkv
            const n_chk: u32 = @min(5, conv_ch);
            var cpu_wqkv_r: [5]f32 = undefined;
            for (0..n_chk) |row| {
                dequantRow(mmap[wqkv_off_d..], @intCast(row), hidden_dim, wt.info.type_, cpu_row_buf[0..hidden_dim]);
                var dot_d: f64 = 0.0;
                for (0..hidden_dim) |ii| dot_d += @as(f64, cpu_row_buf[ii]) * @as(f64, cpu_normed[ii]);
                cpu_wqkv_r[row] = @floatCast(dot_d);
            }

            // GPU: dispatch wqkv DMMV and readback first n_chk elements
            if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
            try self.decode_cmd.reset();
            try self.decode_cmd.begin();
            try self.dispatchDmmv(wt, self.norm_buf, hidden_size, self.logits_buf, conv_ch, hidden_dim);
            {
                const bar4 = vk.c.VkMemoryBarrier{
                    .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = null,
                    .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
                };
                vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &bar4, 0, null, 0, null);
                const wq_copy_sz = @as(vk.c.VkDeviceSize, conv_ch) * @sizeOf(f32);
                const rgn4 = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = wq_copy_sz };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.logits_buf.handle, self.logits_staging.handle, 1, &rgn4);
            }
            try self.decode_cmd.end();
            try self.decode_cmd.submitAndWait(self.instance.compute_queue);

            const gpu_wq: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
            var wqkv_mdiff: f32 = 0;
            for (0..n_chk) |ii| {
                const d4 = @abs(gpu_wq[ii] - cpu_wqkv_r[ii]);
                if (d4 > wqkv_mdiff) wqkv_mdiff = d4;
                dlog.info("wqkv[{d}]: CPU={d:.6} GPU={d:.6} diff={d:.6}", .{
                    ii, cpu_wqkv_r[ii], gpu_wq[ii], gpu_wq[ii] - cpu_wqkv_r[ii],
                });
            }
            dlog.info("DMMV_CHECK: wqkv type={s} M={d} K={d} max_diff={d:.6} ok={s}", .{
                @tagName(wt.info.type_),                                                 conv_ch, hidden_dim, wqkv_mdiff,
                if (wqkv_mdiff < 0.1) @as([]const u8, "YES") else @as([]const u8, "NO"),
            });
        }

        // Also test gate_exps DMMV (MoE expert weights — different quant type?)
        if (gate_exps_diag) |gt| {
            const inter_d = if (config.intermediate_dim > 0) config.intermediate_dim else hidden_dim * 4;
            const gate_off_d: usize = @intCast(self.model.gguf_file.tensor_data_offset + gt.info.offset);

            // CPU: first 3 rows of gate_exps (expert 0)
            var cpu_gate_r: [3]f32 = undefined;
            for (0..3) |row| {
                dequantRow(mmap[gate_off_d..], @intCast(row), hidden_dim, gt.info.type_, cpu_row_buf[0..hidden_dim]);
                var dot_d: f64 = 0.0;
                for (0..hidden_dim) |ii| dot_d += @as(f64, cpu_row_buf[ii]) * @as(f64, cpu_normed[ii]);
                cpu_gate_r[row] = @floatCast(dot_d);
            }

            // GPU: dispatch gate_exps DMMV for expert 0 (offset=0)
            if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
            try self.decode_cmd.reset();
            try self.decode_cmd.begin();
            try self.dispatchDmmvWithOffset(gt, self.norm_buf, hidden_size, self.logits_buf, @intCast(inter_d), hidden_dim, 0);
            {
                const bar5 = vk.c.VkMemoryBarrier{
                    .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = null,
                    .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
                };
                vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &bar5, 0, null, 0, null);
                const g_copy_sz = @as(vk.c.VkDeviceSize, 3) * @sizeOf(f32);
                const rgn5 = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = g_copy_sz };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.logits_buf.handle, self.logits_staging.handle, 1, &rgn5);
            }
            try self.decode_cmd.end();
            try self.decode_cmd.submitAndWait(self.instance.compute_queue);

            const gpu_g: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
            var gate_mdiff: f32 = 0;
            for (0..3) |ii| {
                const d5 = @abs(gpu_g[ii] - cpu_gate_r[ii]);
                if (d5 > gate_mdiff) gate_mdiff = d5;
            }
            dlog.info("DMMV_CHECK: gate_exps type={s} M={d} K={d} gpu[0..2]={d:.4},{d:.4},{d:.4} cpu[0..2]={d:.4},{d:.4},{d:.4} max_diff={d:.6} ok={s}", .{
                @tagName(gt.info.type_), inter_d,                                                                 hidden_dim,
                gpu_g[0],                gpu_g[1],                                                                gpu_g[2],
                cpu_gate_r[0],           cpu_gate_r[1],                                                           cpu_gate_r[2],
                gate_mdiff,              if (gate_mdiff < 0.1) @as([]const u8, "YES") else @as([]const u8, "NO"),
            });
        }

        if (ffn_gate_diag) |gt| {
            const inter_d = if (config.intermediate_dim > 0) config.intermediate_dim else hidden_dim * 4;
            const gate_off_d: usize = @intCast(self.model.gguf_file.tensor_data_offset + gt.info.offset);

            var cpu_gate_r: [3]f32 = undefined;
            for (0..3) |row| {
                dequantRow(mmap[gate_off_d..], @intCast(row), hidden_dim, gt.info.type_, cpu_row_buf[0..hidden_dim]);
                var dot_d: f64 = 0.0;
                for (0..hidden_dim) |ii| dot_d += @as(f64, cpu_row_buf[ii]) * @as(f64, cpu_normed[ii]);
                cpu_gate_r[row] = @floatCast(dot_d);
            }

            if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
            try self.decode_cmd.reset();
            try self.decode_cmd.begin();
            try self.dispatchDmmv(gt, self.norm_buf, hidden_size, self.logits_buf, @intCast(inter_d), hidden_dim);
            {
                const bar = vk.c.VkMemoryBarrier{
                    .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = null,
                    .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
                };
                vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &bar, 0, null, 0, null);
                const copy_sz = @as(vk.c.VkDeviceSize, 3) * @sizeOf(f32);
                const rgn = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = copy_sz };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.logits_buf.handle, self.logits_staging.handle, 1, &rgn);
            }
            try self.decode_cmd.end();
            try self.decode_cmd.submitAndWait(self.instance.compute_queue);

            const gpu_g: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
            var gate_mdiff: f32 = 0;
            for (0..3) |ii| {
                const d = @abs(gpu_g[ii] - cpu_gate_r[ii]);
                if (d > gate_mdiff) gate_mdiff = d;
            }
            dlog.info("DMMV_CHECK: ffn_gate type={s} M={d} K={d} gpu[0..2]={d:.4},{d:.4},{d:.4} cpu[0..2]={d:.4},{d:.4},{d:.4} max_diff={d:.6} ok={s}", .{
                @tagName(gt.info.type_), inter_d,                                                                 hidden_dim,
                gpu_g[0],                gpu_g[1],                                                                gpu_g[2],
                cpu_gate_r[0],           cpu_gate_r[1],                                                           cpu_gate_r[2],
                gate_mdiff,              if (gate_mdiff < 0.1) @as([]const u8, "YES") else @as([]const u8, "NO"),
            });
        }

        if (ffn_up_diag) |ut| {
            const inter_d = if (config.intermediate_dim > 0) config.intermediate_dim else hidden_dim * 4;
            const up_off_d: usize = @intCast(self.model.gguf_file.tensor_data_offset + ut.info.offset);

            var cpu_up_r: [3]f32 = undefined;
            for (0..3) |row| {
                dequantRow(mmap[up_off_d..], @intCast(row), hidden_dim, ut.info.type_, cpu_row_buf[0..hidden_dim]);
                var dot_d: f64 = 0.0;
                for (0..hidden_dim) |ii| dot_d += @as(f64, cpu_row_buf[ii]) * @as(f64, cpu_normed[ii]);
                cpu_up_r[row] = @floatCast(dot_d);
            }

            if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
            try self.decode_cmd.reset();
            try self.decode_cmd.begin();
            try self.dispatchDmmv(ut, self.norm_buf, hidden_size, self.logits_buf, @intCast(inter_d), hidden_dim);
            {
                const bar = vk.c.VkMemoryBarrier{
                    .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = null,
                    .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
                };
                vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &bar, 0, null, 0, null);
                const copy_sz = @as(vk.c.VkDeviceSize, 3) * @sizeOf(f32);
                const rgn = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = copy_sz };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.logits_buf.handle, self.logits_staging.handle, 1, &rgn);
            }
            try self.decode_cmd.end();
            try self.decode_cmd.submitAndWait(self.instance.compute_queue);

            const gpu_u: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
            var up_mdiff: f32 = 0;
            for (0..3) |ii| {
                const d = @abs(gpu_u[ii] - cpu_up_r[ii]);
                if (d > up_mdiff) up_mdiff = d;
            }
            dlog.info("DMMV_CHECK: ffn_up type={s} M={d} K={d} gpu[0..2]={d:.4},{d:.4},{d:.4} cpu[0..2]={d:.4},{d:.4},{d:.4} max_diff={d:.6} ok={s}", .{
                @tagName(ut.info.type_), inter_d,                                                               hidden_dim,
                gpu_u[0],                gpu_u[1],                                                              gpu_u[2],
                cpu_up_r[0],             cpu_up_r[1],                                                           cpu_up_r[2],
                up_mdiff,                if (up_mdiff < 0.1) @as([]const u8, "YES") else @as([]const u8, "NO"),
            });
        }

        if (attn_q_diag) |qt| {
            const q_dim = config.n_heads * config.head_dim;
            const q_off_d: usize = @intCast(self.model.gguf_file.tensor_data_offset + qt.info.offset);

            var cpu_q_r: [3]f32 = undefined;
            for (0..3) |row| {
                dequantRow(mmap[q_off_d..], @intCast(row), hidden_dim, qt.info.type_, cpu_row_buf[0..hidden_dim]);
                var dot_d: f64 = 0.0;
                for (0..hidden_dim) |ii| dot_d += @as(f64, cpu_row_buf[ii]) * @as(f64, cpu_normed[ii]);
                cpu_q_r[row] = @floatCast(dot_d);
            }

            if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
            try self.decode_cmd.reset();
            try self.decode_cmd.begin();
            try self.dispatchDmmv(qt, self.norm_buf, hidden_size, self.logits_buf, q_dim, hidden_dim);
            {
                const bar = vk.c.VkMemoryBarrier{
                    .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                    .pNext = null,
                    .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
                };
                vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle, vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &bar, 0, null, 0, null);
                const copy_sz = @as(vk.c.VkDeviceSize, 3) * @sizeOf(f32);
                const rgn = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = copy_sz };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.logits_buf.handle, self.logits_staging.handle, 1, &rgn);
            }
            try self.decode_cmd.end();
            try self.decode_cmd.submitAndWait(self.instance.compute_queue);

            const gpu_q: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
            var q_mdiff: f32 = 0;
            for (0..3) |ii| {
                const d = @abs(gpu_q[ii] - cpu_q_r[ii]);
                if (d > q_mdiff) q_mdiff = d;
            }
            dlog.info("DMMV_CHECK: attn_q type={s} M={d} K={d} gpu[0..2]={d:.4},{d:.4},{d:.4} cpu[0..2]={d:.4},{d:.4},{d:.4} max_diff={d:.6} ok={s}", .{
                @tagName(qt.info.type_), q_dim,                                                                hidden_dim,
                gpu_q[0],                gpu_q[1],                                                             gpu_q[2],
                cpu_q_r[0],              cpu_q_r[1],                                                           cpu_q_r[2],
                q_mdiff,                 if (q_mdiff < 0.1) @as([]const u8, "YES") else @as([]const u8, "NO"),
            });
        }

        dlog.info("=== END DIAG ===", .{});
    }

    // -----------------------------------------------------------------------
    // Teardown
    // -----------------------------------------------------------------------

    /// Release GPU buffers, graphs, command objects, and dispatch helpers owned by the engine.
    pub fn deinit(self: *InferenceEngine) void {
        if (self.timestamp_query_pool != null) vk.c.vkDestroyQueryPool(self.instance.device, self.timestamp_query_pool, null);
        vk.c.vkDestroyDescriptorPool(self.instance.device, self.shared_pool, null);
        self.tensor_map.deinit();
        self.allocator.free(self.layer_tensors);
        self.allocator.free(self.layer_output_scales);
        // SSM state
        for (self.ssm_conv_states) |s| if (s.len > 0) self.allocator.free(s);
        for (self.ssm_states) |s| if (s.len > 0) self.allocator.free(s);
        self.allocator.free(self.ssm_conv_states);
        self.allocator.free(self.ssm_states);
        self.ssm_hidden_staging.deinit();
        // GPU SSM state + router output
        for (self.gpu_ssm_conv_states) |*b| if (b.handle != null) b.deinit();
        for (self.gpu_ssm_states) |*b| if (b.handle != null) b.deinit();
        self.allocator.free(self.gpu_ssm_conv_states);
        self.allocator.free(self.gpu_ssm_states);
        self.allocator.free(self.ssm_conv_state_offsets);
        self.router_output_buf.deinit();
        if (self.routing_capture_buf.handle != null) self.routing_capture_buf.deinit();
        if (self.prefill_expert_count_buf.handle != null) self.prefill_expert_count_buf.deinit();
        if (self.prefill_ffn_input_capture_buf.handle != null) self.prefill_ffn_input_capture_buf.deinit();
        // Effort-6 cycle 97 (A3b foundation) capture buffers.
        if (self.a3b_alpha_capture) |*b| b.deinit();
        if (self.a3b_beta_capture) |*b| b.deinit();
        if (self.a3b_conv_out_capture) |*b| b.deinit();
        if (self.a3b_state_backup) |*b| b.deinit();
        if (self.a3b_delta_out) |*b| b.deinit();
        if (self.a3b_per_token_delta_out) |*b| b.deinit();
        if (self.a3b_gate_capture) |*b| b.deinit();
        if (self.dense_prefill_validate_norm_ref) |*b| b.deinit();
        if (self.dense_prefill_validate_pre_hidden_ref) |*b| b.deinit();
        if (self.dense_prefill_validate_post_hidden_ref) |*b| b.deinit();
        if (self.dense_prefill_validate_gate_ref) |*b| b.deinit();
        if (self.dense_prefill_validate_up_ref) |*b| b.deinit();
        if (self.dense_prefill_validate_swiglu_ref) |*b| b.deinit();
        if (self.dense_prefill_validate_down_ref) |*b| b.deinit();
        if (self.dense_prefill_validate_staging) |*b| b.deinit();
        if (self.ssm_prefill_validate_norm_ref) |*b| b.deinit();
        if (self.ssm_prefill_validate_qkv_ref) |*b| b.deinit();
        if (self.ssm_prefill_validate_z_ref) |*b| b.deinit();
        if (self.ssm_prefill_validate_alpha_ref) |*b| b.deinit();
        if (self.ssm_prefill_validate_beta_ref) |*b| b.deinit();
        if (self.ssm_prefill_validate_conv_ref) |*b| b.deinit();
        if (self.ssm_prefill_validate_delta_ref) |*b| b.deinit();
        if (self.ssm_prefill_validate_delta_replay) |*b| b.deinit();
        if (self.ssm_prefill_validate_gnorm_ref) |*b| b.deinit();
        if (self.ssm_prefill_validate_pre_hidden_ref) |*b| b.deinit();
        if (self.ssm_prefill_validate_post_hidden_ref) |*b| b.deinit();
        if (self.ssm_prefill_validate_state_backup) |*b| b.deinit();
        if (self.ssm_prefill_validate_staging) |*b| b.deinit();
        // KV cache + page table
        self.freeActiveKvPages();
        self.kv_page_pool.deinit();
        self.page_table_staging.deinit();
        self.page_table_buf.deinit();
        for (self.kv_k_cache) |*b| b.deinit();
        for (self.kv_v_cache) |*b| b.deinit();
        self.allocator.free(self.kv_k_cache);
        self.allocator.free(self.kv_v_cache);
        // Layer intermediates
        self.router_staging.deinit();
        self.router_logits_buf.deinit();
        self.rope_freq_buf.deinit();
        self.unit_norm_weights.deinit();
        self.attn_sinks_buf.deinit();
        self.moe_out_buf.deinit();
        self.down_buf.deinit();
        self.swiglu_buf.deinit();
        self.up_buf.deinit();
        self.gate_buf.deinit();
        self.ffn_norm_buf.deinit();
        self.o_proj_buf.deinit();
        if (self.partial_attn_out_buf.handle != null) self.partial_attn_out_buf.deinit();
        self.attn_out_buf.deinit();
        self.v_buf.deinit();
        self.k_buf.deinit();
        self.q_buf.deinit();
        // Core buffers
        self.embed_staging.deinit();
        self.argmax_result_staging.deinit();
        self.argmax_result_buf.deinit();
        self.argmax_partials_buf.deinit();
        self.logits_staging.deinit();
        self.logits_buf.deinit();
        self.q8_1_buf.deinit();
        self.norm_buf.deinit();
        self.residual_buf.deinit();
        self.hidden_buf.deinit();
        self.decode_graph.deinit();
        self.argmax.deinit();
        self.attention.deinit();
        self.elementwise.deinit();
        self.dmmv.deinit();
        self.decode_cmd.deinit(&self.cmd_pool);
        self.prefill_cmd_alt.deinit(&self.cmd_pool);
        self.prefill_embed_alt.deinit();
        if (self.prefill_embed_big) |*b| b.deinit();
        if (self.batched_scratch_hidden) |*b| b.deinit();
        if (self.batched_scratch_norm) |*b| b.deinit();
        if (self.batched_scratch_q) |*b| b.deinit();
        if (self.batched_scratch_k) |*b| b.deinit();
        if (self.batched_scratch_v) |*b| b.deinit();
        if (self.batched_scratch_attn_out) |*b| b.deinit();
        if (self.batched_scratch_gate) |*b| b.deinit();
        if (self.batched_scratch_up) |*b| b.deinit();
        if (self.batched_scratch_swiglu) |*b| b.deinit();
        if (self.batched_scratch_down) |*b| b.deinit();
        self.cmd_pool.deinit();
        self.* = undefined;
    }
};

/// Dump top-5 logits for a given decode step (for comparing with llama.cpp).
fn dumpTop5Logits(engine: *const InferenceEngine, step: u32) void {
    const vocab_size = engine.model.config.vocab_size;
    const logits_ptr: [*]const f32 = @ptrCast(@alignCast(engine.logits_staging.mapped.?));
    const logits = logits_ptr[0..vocab_size];

    // Find top-5 by value
    var top_ids: [5]u32 = .{ 0, 0, 0, 0, 0 };
    var top_vals: [5]f32 = .{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
    for (logits, 0..) |val, i| {
        if (val > top_vals[4]) {
            top_vals[4] = val;
            top_ids[4] = @intCast(i);
            // Bubble up
            var j: usize = 4;
            while (j > 0 and top_vals[j] > top_vals[j - 1]) : (j -= 1) {
                std.mem.swap(f32, &top_vals[j], &top_vals[j - 1]);
                std.mem.swap(u32, &top_ids[j], &top_ids[j - 1]);
            }
        }
    }
    log.debug("TOP5[{d}]: #{d}={d:.2} #{d}={d:.2} #{d}={d:.2} #{d}={d:.2} #{d}={d:.2}", .{
        step,
        top_ids[0],
        top_vals[0],
        top_ids[1],
        top_vals[1],
        top_ids[2],
        top_vals[2],
        top_ids[3],
        top_vals[3],
        top_ids[4],
        top_vals[4],
    });
}

/// Run single-request inference: prefill the prompt, decode greedily, and return generated token IDs.
/// @param engine Initialized inference engine.
/// @param prompt_tokens Tokenized prompt that seeds the prefill pass.
/// @param max_tokens Maximum number of decode tokens to emit after prefill.
/// @param allocator Allocator used for transient decode state and the returned token slice.
/// @returns A heap-allocated slice containing only the generated continuation tokens.
/// @note Generation stops early on common EOS token IDs used by the currently supported model families.
pub fn generate(
    engine: *InferenceEngine,
    prompt_tokens: []const u32,
    max_tokens: u32,
    eos_token_id: u32,
    allocator: std.mem.Allocator,
) ![]u32 {
    var state = DecodeState.init(allocator);
    defer state.deinit();
    const prompt_token_count: u32 = @intCast(@min(prompt_tokens.len, std.math.maxInt(u32)));
    if (prompt_token_count > engine.max_context_tokens) {
        log.err("Prompt exceeds reserved context: prompt={d} capacity={d}", .{
            prompt_token_count,
            engine.max_context_tokens,
        });
        return error.ContextLengthExceeded;
    }
    const request_budget = memory_plan.requestBudget(prompt_token_count, max_tokens, engine.max_context_tokens);
    const effective_max_tokens = request_budget.completion_tokens;
    state.requested_context_tokens = request_budget.target_context_tokens;
    if (effective_max_tokens < max_tokens) {
        log.info("Clamped decode budget from {d} to {d} tokens (prompt={d}, capacity={d})", .{
            max_tokens,
            effective_max_tokens,
            prompt_token_count,
            engine.max_context_tokens,
        });
    }
    engine.diag_summary_len = 0;
    engine.resetProfilingSamples();

    log.debug("Generating: {d} prompt tokens, max {d} output tokens", .{
        prompt_tokens.len, effective_max_tokens,
    });

    // Prefill: batch all prompt tokens in a single GPU submission.
    // prefillBatched honors ZINC_BATCHED_PREFILL and falls through to
    // prefillBatch (per-token) when the env gate is off or the model
    // isn't on canUseBatchedPrefillRdna's supported slice.
    const prefill_start = nanoTimestamp();
    try engine.prefillBatched(&state, prompt_tokens);
    const prefill_end = nanoTimestamp();
    const prefill_ns: u64 = @intCast(prefill_end - prefill_start);
    const prefill_tok_per_sec = if (prefill_ns > 0 and prompt_tokens.len > 0)
        @as(f64, @floatFromInt(prompt_tokens.len)) * 1_000_000_000.0 / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;

    log.info("Prefill: {d} tokens in {d:.1} ms ({d:.2} tok/s)", .{
        prompt_tokens.len, @as(f64, @floatFromInt(prefill_ns)) / 1_000_000.0, prefill_tok_per_sec,
    });
    if (engine.prefill_token_samples > 0) {
        const samples_f = @as(f64, @floatFromInt(engine.prefill_token_samples));
        const avg_embed_ms = @as(f64, @floatFromInt(engine.prefill_cpu_embed_ns)) / samples_f / 1_000_000.0;
        const avg_record_ms = @as(f64, @floatFromInt(engine.prefill_cpu_record_ns)) / samples_f / 1_000_000.0;
        const avg_submit_wait_ms = @as(f64, @floatFromInt(engine.prefill_submit_wait_ns)) / samples_f / 1_000_000.0;
        const total_embed_ms = @as(f64, @floatFromInt(engine.prefill_cpu_embed_ns)) / 1_000_000.0;
        const total_record_ms = @as(f64, @floatFromInt(engine.prefill_cpu_record_ns)) / 1_000_000.0;
        const total_submit_wait_ms = @as(f64, @floatFromInt(engine.prefill_submit_wait_ns)) / 1_000_000.0;
        log.info(
            "Prefill profile: samples={d} avg embed={d:.3} ms record={d:.2} ms submit+wait={d:.2} ms | totals embed={d:.1} ms record={d:.1} ms submit+wait={d:.1} ms",
            .{
                engine.prefill_token_samples,
                avg_embed_ms,
                avg_record_ms,
                avg_submit_wait_ms,
                total_embed_ms,
                total_record_ms,
                total_submit_wait_ms,
            },
        );
        // Per-phase GPU breakdown — only present if timestamp pool was available.
        var any_phase_ns: u64 = 0;
        for (engine.prefill_gpu_phase_ns) |v| any_phase_ns += v;
        if (any_phase_ns > 0) {
            // Aggregate related MoE and shared-expert phases into top-level buckets
            // so the summary line stays scannable across cycles.
            var attn_ns: u64 = 0;
            var moe_ns: u64 = 0;
            var shared_ns: u64 = 0;
            var ssm_ns: u64 = 0;
            var dense_ffn_ns: u64 = 0;
            var tail_ns: u64 = 0;
            var embed_ns: u64 = 0;
            // `.ssm` wraps all ssm_* sub-phases and `.moe_routed` wraps all moe_*
            // sub-phases. `.dense_ffn` likewise wraps dense_ffn_* sub-phases.
            // Bucket with the wrappers only; shared_* and tail/attn/embed have
            // no wrapper.
            inline for (@typeInfo(ProfilePhase).@"enum".fields) |f| {
                const phase_val: ProfilePhase = @enumFromInt(f.value);
                const v = engine.prefill_gpu_phase_ns[f.value];
                switch (phase_val) {
                    .attention => attn_ns += v,
                    .moe_routed => moe_ns += v,
                    .shared_expert, .shared_proj, .shared_swiglu, .shared_down, .shared_gate_acc => shared_ns += v,
                    .ssm => ssm_ns += v,
                    .dense_ffn => dense_ffn_ns += v,
                    .final_tail => tail_ns += v,
                    .embed_upload => embed_ns += v,
                    else => {},
                }
            }
            const to_ms = struct {
                fn f(v: u64) f64 {
                    return @as(f64, @floatFromInt(v)) / 1_000_000.0;
                }
            }.f;
            const attn_avg = to_ms(attn_ns) / samples_f;
            const moe_avg = to_ms(moe_ns) / samples_f;
            const shared_avg = to_ms(shared_ns) / samples_f;
            const ssm_avg = to_ms(ssm_ns) / samples_f;
            const dense_ffn_avg = to_ms(dense_ffn_ns) / samples_f;
            const tail_avg = to_ms(tail_ns) / samples_f;
            const embed_avg = to_ms(embed_ns) / samples_f;
            log.info(
                "Prefill GPU phases: per-tok attn={d:.2} ms moe={d:.2} ms shared={d:.2} ms ssm={d:.2} ms dense_ffn={d:.2} ms tail={d:.2} ms embed={d:.3} ms | totals attn={d:.1} moe={d:.1} shared={d:.1} ssm={d:.1} dense_ffn={d:.1} tail={d:.1} embed={d:.1}",
                .{
                    attn_avg,
                    moe_avg,
                    shared_avg,
                    ssm_avg,
                    dense_ffn_avg,
                    tail_avg,
                    embed_avg,
                    to_ms(attn_ns),
                    to_ms(moe_ns),
                    to_ms(shared_ns),
                    to_ms(ssm_ns),
                    to_ms(dense_ffn_ns),
                    to_ms(tail_ns),
                    to_ms(embed_ns),
                },
            );
            // Drill-down inside the two biggest composite buckets so the next
            // cycle can target the largest MoE sub-phase directly.
            const router_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.moe_router)];
            const topk_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.moe_topk)];
            const gate_up_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.moe_gate_up)];
            const swiglu_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.moe_swiglu)];
            const down_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.moe_down)];
            const weighted_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.moe_weighted_acc)];
            log.info(
                "Prefill MoE subphases totals: router={d:.1} topk={d:.1} gate_up={d:.1} swiglu={d:.1} down={d:.1} weighted_acc={d:.1} ms",
                .{
                    to_ms(router_ns),
                    to_ms(topk_ns),
                    to_ms(gate_up_ns),
                    to_ms(swiglu_ns),
                    to_ms(down_ns),
                    to_ms(weighted_ns),
                },
            );
            const ssm_proj_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_proj)];
            const ssm_proj_norm_ab_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_proj_norm_ab)];
            const ssm_proj_qkv_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_proj_qkv)];
            const ssm_proj_z_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_proj_z)];
            const ssm_proj_alpha_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_proj_alpha)];
            const ssm_proj_beta_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_proj_beta)];
            const ssm_proj_qkv_z_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_proj_qkv_z)];
            const ssm_conv_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_conv)];
            const ssm_delta_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_delta)];
            const ssm_gnorm_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_gated_norm)];
            const ssm_out_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_out)];
            log.info(
                "Prefill SSM subphases totals: proj={d:.1} norm_ab={d:.1} qkv={d:.1} z={d:.1} alpha={d:.1} beta={d:.1} qkv_z={d:.1} conv={d:.1} delta={d:.1} gnorm={d:.1} out={d:.1} ms",
                .{
                    to_ms(ssm_proj_ns),
                    to_ms(ssm_proj_norm_ab_ns),
                    to_ms(ssm_proj_qkv_ns),
                    to_ms(ssm_proj_z_ns),
                    to_ms(ssm_proj_alpha_ns),
                    to_ms(ssm_proj_beta_ns),
                    to_ms(ssm_proj_qkv_z_ns),
                    to_ms(ssm_conv_ns),
                    to_ms(ssm_delta_ns),
                    to_ms(ssm_gnorm_ns),
                    to_ms(ssm_out_ns),
                },
            );
            const dense_gateup_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.dense_ffn_gateup)];
            const dense_gate_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.dense_ffn_gate)];
            const dense_up_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.dense_ffn_up)];
            const dense_down_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.dense_ffn_down)];
            log.info(
                "Prefill dense_ffn subphases totals: gateup={d:.1} gate={d:.1} up={d:.1} down={d:.1} ms",
                .{
                    to_ms(dense_gateup_ns),
                    to_ms(dense_gate_ns),
                    to_ms(dense_up_ns),
                    to_ms(dense_down_ns),
                },
            );
        }
    }
    // Decode profiling should describe only generated tokens, not the prompt prefill steps.
    engine.resetProfilingSamples();

    // Decode: generate tokens one at a time
    // After prefill, logits_staging already has the logits for the first output
    // token (from the last prompt token's forward pass). Sample directly from
    // those logits instead of reprocessing the last prompt token — that would
    // duplicate its KV cache entry and shift the entire context.
    var generated: u32 = 0;
    const decode_start = nanoTimestamp();

    // Sample the first output token from prefill logits (no extra decodeStep)
    if (prompt_tokens.len > 0 and effective_max_tokens > 0) {
        const first_token = engine.sampleGreedy();
        try state.generated_tokens.append(allocator, first_token);
        log.debug("decode[0]: token={d} pos={d} (from prefill logits)", .{
            first_token, state.position,
        });
        // Dump top-5 logits from prefill for comparison with llama.cpp
        if (engine.logits_readback_enabled or engine.validation_diagnostics_enabled) dumpTop5Logits(engine, 0);
        generated = 1;
        if (first_token == eos_token_id) generated = effective_max_tokens; // stop early
    }

    while (generated < effective_max_tokens) : (generated += 1) {
        const tok_start = nanoTimestamp();

        // Feed the last generated token as input
        const input_token = state.generated_tokens.items[state.generated_tokens.items.len - 1];

        try engine.decodeStep(&state, input_token, true);
        const token = engine.sampleGreedy();
        try state.generated_tokens.append(allocator, token);
        // Top-5 logits per token for first 5 tokens + last token
        if (generated < 5 or generated == effective_max_tokens - 1) {
            if (engine.logits_readback_enabled) dumpTop5Logits(engine, generated);
        }

        const tok_end = nanoTimestamp();
        const tok_ms = @as(f64, @floatFromInt(@as(u64, @intCast(tok_end - tok_start)))) / 1_000_000.0;
        log.debug("decode[{d}]: token={d} pos={d} ({d:.1} ms)", .{
            generated, token, state.position, tok_ms,
        });

        // Check for EOS token (read from GGUF metadata)
        if (token == eos_token_id) break;
    }
    const decode_end = nanoTimestamp();

    const decode_tokens = state.generated_tokens.items.len;
    const decode_ns: u64 = @intCast(decode_end - decode_start);
    if (decode_ns > 0 and decode_tokens > 0) {
        const tok_per_sec = @as(f64, @floatFromInt(decode_tokens)) * 1_000_000_000.0 / @as(f64, @floatFromInt(decode_ns));
        const ms_per_tok = @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0 / @as(f64, @floatFromInt(decode_tokens));
        log.info("Generated {d} tokens in {d:.1} ms — {d:.2} tok/s ({d:.1} ms/tok)", .{
            decode_tokens, @as(f64, @floatFromInt(decode_ns)) / 1_000_000.0, tok_per_sec, ms_per_tok,
        });

        if (engine.modeled_decode_bytes_per_token > 0) {
            const total_bytes_all = engine.modeled_decode_bytes_per_token * @as(u64, @intCast(decode_tokens));
            const decode_secs = @as(f64, @floatFromInt(decode_ns)) / 1_000_000_000.0;
            const eff_bw_gbs = @as(f64, @floatFromInt(total_bytes_all)) / decode_secs / 1_000_000_000.0;
            const theo_bw_gbs: f64 = @floatFromInt(engine.gpu_config.bandwidth_gbps);
            const utilization = if (theo_bw_gbs > 0) eff_bw_gbs / theo_bw_gbs * 100.0 else 0.0;

            log.info("Modeled decode bandwidth: {d:.1} GB/s effective, {d:.0} GB/s theoretical ({d:.1}% utilization, ~{d:.1} MB/token)", .{
                eff_bw_gbs,
                theo_bw_gbs,
                utilization,
                @as(f64, @floatFromInt(engine.modeled_decode_bytes_per_token)) / 1_000_000.0,
            });
        }
        if (engine.profile_enabled and engine.profile_sample_count > 0) {
            const avg_gpu_ms = engine.profile_total_gpu_ms / @as(f64, @floatFromInt(engine.profile_sample_count));
            const avg_cpu_embed_ms = @as(f64, @floatFromInt(engine.profile_total_cpu_embed_ns)) / @as(f64, @floatFromInt(engine.profile_sample_count)) / 1_000_000.0;
            const avg_cpu_record_ms = @as(f64, @floatFromInt(engine.profile_total_cpu_record_ns)) / @as(f64, @floatFromInt(engine.profile_sample_count)) / 1_000_000.0;
            const avg_submit_wait_ms = @as(f64, @floatFromInt(engine.profile_total_submit_wait_ns)) / @as(f64, @floatFromInt(engine.profile_sample_count)) / 1_000_000.0;
            const avg_query_read_ms = @as(f64, @floatFromInt(engine.profile_total_query_read_ns)) / @as(f64, @floatFromInt(engine.profile_sample_count)) / 1_000_000.0;
            const avg_embed_phase_ms = engine.avgProfilePhaseMs(.embed_upload);
            const avg_attention_phase_ms = engine.avgProfilePhaseMs(.attention);
            const avg_flash_attn_phase_ms = engine.avgProfilePhaseMs(.flash_attn_kernel);
            const avg_ssm_phase_ms = engine.avgProfilePhaseMs(.ssm);
            const avg_moe_phase_ms = engine.avgProfilePhaseMs(.moe_routed);
            const avg_shared_phase_ms = engine.avgProfilePhaseMs(.shared_expert);
            const avg_dense_ffn_phase_ms = engine.avgProfilePhaseMs(.dense_ffn);
            const avg_tail_phase_ms = engine.avgProfilePhaseMs(.final_tail);
            const avg_desc_allocs = @as(f64, @floatFromInt(engine.profile_total_counters.descriptor_allocs)) / @as(f64, @floatFromInt(engine.profile_sample_count));
            const avg_desc_writes = @as(f64, @floatFromInt(engine.profile_total_counters.descriptor_write_calls)) / @as(f64, @floatFromInt(engine.profile_sample_count));
            const avg_desc_bindings = @as(f64, @floatFromInt(engine.profile_total_counters.descriptor_bindings)) / @as(f64, @floatFromInt(engine.profile_sample_count));
            const avg_wait_overhang_ms = @max(0.0, avg_submit_wait_ms - avg_gpu_ms);
            log.info("PROFILE: avg GPU decode token={d:.2} ms over {d} sampled decode steps (max={d:.2} ms)", .{
                avg_gpu_ms,
                engine.profile_sample_count,
                engine.profile_max_gpu_ms,
            });
            log.info("PROFILE: avg CPU embed={d:.2} ms | avg CPU record={d:.2} ms (max={d:.2} ms) | avg submit+wait={d:.2} ms (max={d:.2} ms) | avg query_read={d:.3} ms | submit overhang={d:.2} ms", .{
                avg_cpu_embed_ms,
                avg_cpu_record_ms,
                @as(f64, @floatFromInt(engine.profile_max_cpu_record_ns)) / 1_000_000.0,
                avg_submit_wait_ms,
                @as(f64, @floatFromInt(engine.profile_max_submit_wait_ns)) / 1_000_000.0,
                avg_query_read_ms,
                avg_wait_overhang_ms,
            });
            log.info("PROFILE: avg descriptor allocs={d:.1} writes={d:.1} bindings={d:.1}", .{
                avg_desc_allocs,
                avg_desc_writes,
                avg_desc_bindings,
            });
            log.info("PROFILE: avg GPU phases embed={d:.2} ms attention={d:.2} ms (flash_attn={d:.2} ms) ssm={d:.2} ms moe={d:.2} ms shared={d:.2} ms dense_ffn={d:.2} ms tail={d:.2} ms", .{
                avg_embed_phase_ms,
                avg_attention_phase_ms,
                avg_flash_attn_phase_ms,
                avg_ssm_phase_ms,
                avg_moe_phase_ms,
                avg_shared_phase_ms,
                avg_dense_ffn_phase_ms,
                avg_tail_phase_ms,
            });
            log.info("PROFILE: avg SSM subphases proj={d:.2} ms norm_ab={d:.2} ms qkv={d:.2} ms z={d:.2} ms alpha={d:.2} ms beta={d:.2} ms qkv_z={d:.2} ms conv={d:.2} ms delta={d:.2} ms gnorm={d:.2} ms out={d:.2} ms", .{
                engine.avgProfilePhaseMs(.ssm_proj),
                engine.avgProfilePhaseMs(.ssm_proj_norm_ab),
                engine.avgProfilePhaseMs(.ssm_proj_qkv),
                engine.avgProfilePhaseMs(.ssm_proj_z),
                engine.avgProfilePhaseMs(.ssm_proj_alpha),
                engine.avgProfilePhaseMs(.ssm_proj_beta),
                engine.avgProfilePhaseMs(.ssm_proj_qkv_z),
                engine.avgProfilePhaseMs(.ssm_conv),
                engine.avgProfilePhaseMs(.ssm_delta),
                engine.avgProfilePhaseMs(.ssm_gated_norm),
                engine.avgProfilePhaseMs(.ssm_out),
            });
            log.info("PROFILE: avg MoE subphases router={d:.2} ms topk={d:.2} ms gate_up={d:.2} ms swiglu={d:.2} ms down={d:.2} ms acc={d:.2} ms", .{
                engine.avgProfilePhaseMs(.moe_router),
                engine.avgProfilePhaseMs(.moe_topk),
                engine.avgProfilePhaseMs(.moe_gate_up),
                engine.avgProfilePhaseMs(.moe_swiglu),
                engine.avgProfilePhaseMs(.moe_down),
                engine.avgProfilePhaseMs(.moe_weighted_acc),
            });
            log.info("PROFILE: avg shared subphases proj={d:.2} ms swiglu={d:.2} ms down={d:.2} ms gate={d:.2} ms", .{
                engine.avgProfilePhaseMs(.shared_proj),
                engine.avgProfilePhaseMs(.shared_swiglu),
                engine.avgProfilePhaseMs(.shared_down),
                engine.avgProfilePhaseMs(.shared_gate_acc),
            });
            log.info("PROFILE: avg dense_ffn subphases gateup={d:.2} ms gate={d:.2} ms up={d:.2} ms down={d:.2} ms", .{
                engine.avgProfilePhaseMs(.dense_ffn_gateup),
                engine.avgProfilePhaseMs(.dense_ffn_gate),
                engine.avgProfilePhaseMs(.dense_ffn_up),
                engine.avgProfilePhaseMs(.dense_ffn_down),
            });
            log.info("PROFILE: avg tail subphases norm={d:.2} ms lm_head={d:.2} ms argmax={d:.2} ms copy={d:.2} ms", .{
                engine.avgProfilePhaseMs(.final_norm),
                engine.avgProfilePhaseMs(.final_lm_head),
                engine.avgProfilePhaseMs(.final_argmax),
                engine.avgProfilePhaseMs(.final_copy),
            });
            log.info("PROFILE: fallback counts cpu_ssm={d} cpu_moe={d} cpu_shared_gate={d} cpu_argmax={d}", .{
                engine.profile_total_counters.cpu_ssm_fallbacks,
                engine.profile_total_counters.cpu_moe_fallbacks,
                engine.profile_total_counters.cpu_shared_gate_fallbacks,
                engine.profile_total_counters.cpu_argmax_fallbacks,
            });
        }
    } else {
        log.info("Generated {d} tokens", .{decode_tokens});
    }

    // Per-layer flash_attn_kernel histogram (ZINC_FA_PROFILE_LAYER=1).
    // Each entry is the average ms across the sampled decode tokens for the
    // Nth flash_attn dispatch in token order. For Qwen 3 8B that maps 1:1 to
    // attention layer N. For hybrid models (SSM-interleaved) the index
    // collapses across only attention-bearing layers — sufficient to spot a
    // class-level outlier (full-attn vs SWA, first vs last). Unlocks the
    // structural-swing #6 attack: pick the slowest layer-class and target it.
    if (engine.fa_profile_layer) {
        var max_layer: u32 = 0;
        for (0..engine.fa_per_layer_count.len) |i| {
            if (engine.fa_per_layer_count[i] > 0) max_layer = @intCast(i + 1);
        }
        if (max_layer > 0) {
            var total_ns: u64 = 0;
            var min_ms: f64 = std.math.inf(f64);
            var max_ms: f64 = 0.0;
            var min_idx: u32 = 0;
            var max_idx: u32 = 0;
            log.info("FA_PROFILE_LAYER: per-layer flash_attn ms (avg over decode tokens, {d} dispatches/layer)", .{
                if (max_layer > 0) engine.fa_per_layer_count[0] else 0,
            });
            for (0..max_layer) |i| {
                const cnt = engine.fa_per_layer_count[i];
                if (cnt == 0) continue;
                const avg_ms = @as(f64, @floatFromInt(engine.fa_per_layer_ns[i])) /
                    @as(f64, @floatFromInt(cnt)) / 1_000_000.0;
                total_ns += engine.fa_per_layer_ns[i];
                if (avg_ms < min_ms) {
                    min_ms = avg_ms;
                    min_idx = @intCast(i);
                }
                if (avg_ms > max_ms) {
                    max_ms = avg_ms;
                    max_idx = @intCast(i);
                }
                log.info("FA_PROFILE_LAYER:   L{d:0>2} {d:.4} ms", .{ i, avg_ms });
            }
            const total_avg_ms = @as(f64, @floatFromInt(total_ns)) /
                @as(f64, @floatFromInt(@max(engine.fa_per_layer_count[0], 1))) / 1_000_000.0;
            log.info("FA_PROFILE_LAYER: summary min=L{d}({d:.4}ms) max=L{d}({d:.4}ms) ratio={d:.2}x total_per_token={d:.3}ms", .{
                min_idx,                                  min_ms,       max_idx, max_ms,
                if (min_ms > 0) max_ms / min_ms else 0.0, total_avg_ms,
            });
        }
    }

    // Print per-layer diagnostic summary (stored during BOS processing)
    if (engine.validation_diagnostics_enabled and engine.diag_summary_len > 0) {
        log.info("LOGIT5_SUMMARY: {s}", .{engine.diag_summary_buf[0..engine.diag_summary_len]});
    }

    // Run diagnostic AFTER generation so output appears at the end (not truncated)
    if (engine.validation_diagnostics_enabled and prompt_tokens.len > 0) {
        engine.diagEmbedToLogits(prompt_tokens[0]) catch |err| {
            log.warn("Diagnostic failed: {s}", .{@errorName(err)});
        };
    }

    return try allocator.dupe(u32, state.generated_tokens.items);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

fn makeTestModelConfig() ModelConfig {
    return .{
        .architecture = .unknown,
        .n_layers = 0,
        .n_heads = 0,
        .n_kv_heads = 0,
        .head_dim = 0,
        .hidden_dim = 0,
        .intermediate_dim = 0,
        .vocab_size = 0,
        .context_length = 0,
        .rope_freq_base = 0,
        .n_experts = 0,
        .n_experts_used = 0,
        .rope_dim = 0,
        .ssm_d_conv = 0,
        .ssm_d_inner = 0,
        .ssm_d_state = 0,
        .ssm_dt_rank = 0,
        .ssm_n_group = 0,
        .full_attn_interval = 0,
        .shared_expert_intermediate_dim = 0,
    };
}

test "topKSoftmax selects correct top-k with renormalization" {
    const logits = [_]f32{ 1.0, 3.0, 0.5, 2.0, 4.0, 1.5, 0.1, 0.2 };
    var ids: [3]u32 = undefined;
    var weights: [3]f32 = undefined;
    topKSoftmax(&logits, 3, &ids, &weights);
    // Top 3 by value: index 4 (4.0), index 1 (3.0), index 3 (2.0)
    try std.testing.expectEqual(@as(u32, 4), ids[0]);
    try std.testing.expectEqual(@as(u32, 1), ids[1]);
    try std.testing.expectEqual(@as(u32, 3), ids[2]);
    // Weights should sum to ~1.0
    const wsum = weights[0] + weights[1] + weights[2];
    try std.testing.expect(@abs(wsum - 1.0) < 0.01);
    // Highest logit should have highest weight
    try std.testing.expect(weights[0] > weights[1]);
    try std.testing.expect(weights[1] > weights[2]);
}

test "expertSliceBytes computes correct byte offsets for Q4_K" {
    // Q4_K: block_size=256, bytes_per_block=144
    // 512 rows × 2048 cols: blocks_per_row = 2048/256 = 8
    // bytes = 512 * 8 * 144 = 589,824
    const result = expertSliceBytes(.q4_k, 512, 2048);
    try std.testing.expectEqual(@as(u32, 589_824), result);
}

test "expertSliceBytes computes correct byte offsets for Q5_K" {
    // Q5_K: block_size=256, bytes_per_block=176
    // 2048 rows × 512 cols: blocks_per_row = 512/256 = 2
    // bytes = 2048 * 2 * 176 = 720,896
    const result = expertSliceBytes(.q5_k, 2048, 512);
    try std.testing.expectEqual(@as(u32, 720_896), result);
}

test "SamplingParams requires logits readback for non-greedy decoding" {
    try std.testing.expect(!(SamplingParams{}).requiresLogitsReadback());
    try std.testing.expect((SamplingParams{ .temperature = 0.7 }).requiresLogitsReadback());
    try std.testing.expect((SamplingParams{ .top_p = 0.9 }).requiresLogitsReadback());
    try std.testing.expect((SamplingParams{ .repetition_penalty = 1.1 }).requiresLogitsReadback());
}

test "sampleFromLogits greedy path returns argmax" {
    const logits = [_]f32{ 0.5, 2.0, 1.25 };
    var prng = std.Random.DefaultPrng.init(1234);
    const token = InferenceEngine.sampleFromLogits(&logits, &.{}, .{}, prng.random(), 0);
    try std.testing.expectEqual(@as(u32, 1), token);
}

test "sampleFromLogits repetition penalty can break a simple loop" {
    const logits = [_]f32{ 10.0, 9.0, 1.0 };
    const history = [_]u32{ 0, 0, 0 };
    var prng = std.Random.DefaultPrng.init(42);
    const token = InferenceEngine.sampleFromLogits(&logits, &history, .{
        .temperature = 0.0,
        .repetition_penalty = 2.0,
    }, prng.random(), 0);
    try std.testing.expectEqual(@as(u32, 1), token);
}

test "sampleFromLogits top_p keeps only the highest-probability token when threshold is low" {
    const logits = [_]f32{ 8.0, 5.0, 1.0 };
    var prng = std.Random.DefaultPrng.init(7);
    const token = InferenceEngine.sampleFromLogits(&logits, &.{}, .{
        .temperature = 0.8,
        .top_p = 0.5,
        .top_k = 8,
    }, prng.random(), 0);
    try std.testing.expectEqual(@as(u32, 0), token);
}

// ---------------------------------------------------------------------------
// dequantRow tests — lock down the quant formats that had bugs
// ---------------------------------------------------------------------------

test "dequantRow Q4_K sub-block pairing: low nibble then high nibble" {
    // Q4_K block: d[2] dmin[2] scales[12] qs[128] = 144 bytes, 256 elements
    // Bug found: sub-block pairing was (sp, sp+4) instead of (2*sp, 2*sp+1).
    // The correct layout processes 32 consecutive bytes at a time:
    //   first 32 outputs from low nibbles, next 32 from high nibbles.
    var block: [144]u8 = [_]u8{0} ** 144;
    // d = 1.0 as f16
    const d_bits = @as(u16, @bitCast(@as(f16, 1.0)));
    block[0] = @truncate(d_bits);
    block[1] = @truncate(d_bits >> 8);
    // dmin = 0 as f16 (simplifies: output = d * scale * nibble)
    block[2] = 0;
    block[3] = 0;
    // scales[0] = scale=1, scales[4] = min=0 (for j=0 pair: sc=1, m=0)
    block[4] = 1; // scales[0]: low 6 bits = 1
    block[8] = 0; // scales[4]: low 6 bits = 0 (min)
    // scales[1] = scale=2 for high-nibble sub-block
    block[5] = 2; // scales[1]: low 6 bits = 2
    block[9] = 0; // scales[5]: min = 0
    // qs[0..31]: first 32 bytes, low nibble for first 32 outputs, high nibble for next 32
    block[16] = 0x53; // low nibble=3, high nibble=5
    block[17] = 0x97; // low nibble=7, high nibble=9

    var output: [256]f32 = undefined;
    dequantRow(&block, 0, 256, .q4_k, &output);

    // First sub-block: scale=1, so output = 1.0 * 1 * nibble = nibble
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), output[0], 0.01); // low nibble of 0x53
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), output[1], 0.01); // low nibble of 0x97
    // Second sub-block: scale=2, output = 1.0 * 2 * nibble
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), output[32], 0.01); // high nibble of 0x53 = 5, * 2
    try std.testing.expectApproxEqAbs(@as(f32, 18.0), output[33], 0.01); // high nibble of 0x97 = 9, * 2
}

test "dequantRow Q5_K keeps GGML contiguous half ordering" {
    // Q5_K block: d[2] dmin[2] scales[12] qh[32] qs[128] = 176 bytes, 256 elements
    // GGML dequantizes each 64-element group as low-half first, then high-half:
    // for byte qs[l], low nibble → output[l], high nibble → output[32 + l].
    var block: [176]u8 = [_]u8{0} ** 176;
    // d = 1.0 as f16
    const d_bits = @as(u16, @bitCast(@as(f16, 1.0)));
    block[0] = @truncate(d_bits);
    block[1] = @truncate(d_bits >> 8);
    // dmin = 0 (simplifies output)
    block[2] = 0;
    block[3] = 0;
    // scales[0] = 1 (sc for sub-block 0), scales[4] = 0 (min)
    block[4] = 1;
    block[8] = 0;
    // scales[1] = 1 (sc for sub-block 1)
    block[5] = 1;
    block[9] = 0;
    // qh = all 0 (no high bits set, so values are pure 4-bit)
    // qs[0] at block[48]: low=0xA (10), high=0x3 (3)
    block[48] = 0x3A; // low nibble=0xA=10, high nibble=0x3=3

    var output: [256]f32 = undefined;
    dequantRow(&block, 0, 256, .q5_k, &output);

    // Contiguous halves: output[0] from low nibble, output[32] from high nibble
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), output[0], 0.01); // d*sc*10 - 0 = 10
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), output[32], 0.01); // d*sc*3 - 0 = 3
}

test "dequantRow Q8_0 correct scale and signed values" {
    // Q8_0 block: scale[2 bytes f16] + 32×i8 = 34 bytes per block of 32 elements
    // Bug found: wave32 subgroup lost half the dot product.
    // This tests the CPU dequant path (used for embeddings).
    var block: [34]u8 = [_]u8{0} ** 34;
    // scale = 0.5 as f16
    const scale_bits = @as(u16, @bitCast(@as(f16, 0.5)));
    block[0] = @truncate(scale_bits);
    block[1] = @truncate(scale_bits >> 8);
    // quant values: +1, -1, +127, -128
    block[2] = @bitCast(@as(i8, 1));
    block[3] = @bitCast(@as(i8, -1));
    block[4] = @bitCast(@as(i8, 127));
    block[5] = @bitCast(@as(i8, -128));

    var output: [32]f32 = undefined;
    dequantRow(&block, 0, 32, .q8_0, &output);

    try std.testing.expectApproxEqAbs(@as(f32, 0.5), output[0], 0.001); // 1 * 0.5
    try std.testing.expectApproxEqAbs(@as(f32, -0.5), output[1], 0.001); // -1 * 0.5
    try std.testing.expectApproxEqAbs(@as(f32, 63.5), output[2], 0.001); // 127 * 0.5
    try std.testing.expectApproxEqAbs(@as(f32, -64.0), output[3], 0.001); // -128 * 0.5
    // Remaining should be 0
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[6], 0.001);
}

test "dequantRow F16 round-trip preserves values" {
    // Write known f16 values and verify dequant produces correct f32
    var raw: [8]u8 = undefined;
    const vals = [_]f16{ 1.5, -3.25, 0.0, 42.0 };
    for (vals, 0..) |v, i| {
        const bits = @as(u16, @bitCast(v));
        raw[i * 2] = @truncate(bits);
        raw[i * 2 + 1] = @truncate(bits >> 8);
    }
    var output: [4]f32 = undefined;
    dequantRow(&raw, 0, 4, .f16, &output);

    try std.testing.expectApproxEqAbs(@as(f32, 1.5), output[0], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, -3.25), output[1], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), output[2], 0.001);
    try std.testing.expectApproxEqAbs(@as(f32, 42.0), output[3], 0.01);
}

test "dequantRow F32 direct copy" {
    var raw: [16]u8 align(4) = undefined;
    const src: *[4]f32 = @ptrCast(@alignCast(&raw));
    src.* = .{ 1.0, -2.0, 3.14, 0.0 };
    var output: [4]f32 = undefined;
    dequantRow(&raw, 0, 4, .f32, &output);

    try std.testing.expectEqual(@as(f32, 1.0), output[0]);
    try std.testing.expectEqual(@as(f32, -2.0), output[1]);
    try std.testing.expectApproxEqAbs(@as(f32, 3.14), output[2], 0.001);
    try std.testing.expectEqual(@as(f32, 0.0), output[3]);
}

// ---------------------------------------------------------------------------
// getScaleMinK4 tests — bit extraction used by Q4_K and Q5_K
// ---------------------------------------------------------------------------

test "getScaleMinK4 low indices extract 6-bit fields" {
    // For j < 4: sc = scales[j] & 63, m = scales[j+4] & 63
    var scales: [12]u8 = [_]u8{0} ** 12;
    scales[0] = 0xFF; // 0b11_111111 → sc = 63
    scales[4] = 0xC5; // 0b11_000101 → m = 5

    const sm = getScaleMinK4(0, &scales);
    try std.testing.expectEqual(@as(u8, 63), sm.sc);
    try std.testing.expectEqual(@as(u8, 5), sm.m);
}

test "getScaleMinK4 high indices combine nibbles and top bits" {
    // For j >= 4: sc = (scales[j+4] & 0xF) | ((scales[j-4] >> 6) << 4)
    //             m  = (scales[j+4] >> 4)   | ((scales[j]   >> 6) << 4)
    var scales: [12]u8 = [_]u8{0} ** 12;
    // j=4: sc uses scales[8] low nibble + scales[0] top 2 bits
    //       m uses scales[8] high nibble + scales[4] top 2 bits
    scales[0] = 0xC0; // top 2 bits = 0b11 → contributes 0b11_0000 = 48 to sc
    scales[4] = 0x80; // top 2 bits = 0b10 → contributes 0b10_0000 = 32 to m
    scales[8] = 0x72; // low nibble = 0x2, high nibble = 0x7

    const sm = getScaleMinK4(4, &scales);
    try std.testing.expectEqual(@as(u8, 0x2 | (3 << 4)), sm.sc); // 2 + 48 = 50
    try std.testing.expectEqual(@as(u8, 0x7 | (2 << 4)), sm.m); // 7 + 32 = 39
}

// ---------------------------------------------------------------------------
// topKSoftmax edge cases
// ---------------------------------------------------------------------------

test "topKSoftmax with uniform logits returns equal weights" {
    const logits = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var ids: [2]u32 = undefined;
    var weights: [2]f32 = undefined;
    topKSoftmax(&logits, 2, &ids, &weights);
    // All logits equal → weights should be equal (0.5 each after renorm)
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weights[0], 0.01);
    try std.testing.expectApproxEqAbs(@as(f32, 0.5), weights[1], 0.01);
}

test "topKSoftmax matches selected-only renormalized softmax" {
    const logits = [_]f32{ -2.0, 1.5, 0.25, 4.0, -0.5, 3.0, 2.5, -1.0 };
    const k = 4;
    var ids: [k]u32 = undefined;
    var weights: [k]f32 = undefined;
    topKSoftmax(&logits, k, &ids, &weights);

    var max_logit: f32 = -std.math.inf(f32);
    for (0..k) |i| {
        max_logit = @max(max_logit, logits[ids[i]]);
    }

    var selected_weights: [k]f32 = undefined;
    var sum: f32 = 0.0;
    for (0..k) |i| {
        const w = @exp(logits[ids[i]] - max_logit);
        selected_weights[i] = w;
        sum += w;
    }
    for (0..k) |i| {
        selected_weights[i] /= sum;
        try std.testing.expectApproxEqAbs(selected_weights[i], weights[i], 1e-6);
    }
}

test "topKSoftmax k=1 picks argmax with weight 1.0" {
    const logits = [_]f32{ -1.0, 5.0, 2.0 };
    var ids: [1]u32 = undefined;
    var weights: [1]f32 = undefined;
    topKSoftmax(&logits, 1, &ids, &weights);
    try std.testing.expectEqual(@as(u32, 1), ids[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), weights[0], 0.001);
}

test "topKSoftmax with large logit spread avoids overflow" {
    // exp(100) overflows f32, but softmax with max subtraction should handle it
    const logits = [_]f32{ 100.0, 0.0, -100.0 };
    var ids: [2]u32 = undefined;
    var weights: [2]f32 = undefined;
    topKSoftmax(&logits, 2, &ids, &weights);
    try std.testing.expectEqual(@as(u32, 0), ids[0]);
    try std.testing.expectEqual(@as(u32, 1), ids[1]);
    // First weight should dominate
    try std.testing.expect(weights[0] > 0.99);
    const wsum = weights[0] + weights[1];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), wsum, 0.001);
}

test "topKSoftmaxWeight matches selected-only renormalized softmax" {
    const logits = [_]f32{ -3.0, 1.5, 0.25, 4.0, -0.5, 3.0, 2.5, -1.0 };
    const k = 4;
    var ids: [k]u32 = undefined;
    var weights: [k]f32 = undefined;
    topKSoftmaxWeight(&logits, k, &ids, &weights);

    var max_logit: f32 = -std.math.inf(f32);
    for (0..k) |i| {
        max_logit = @max(max_logit, logits[ids[i]]);
    }

    var expected: [k]f32 = undefined;
    var sum: f32 = 0.0;
    for (0..k) |i| {
        const w = @exp(logits[ids[i]] - max_logit);
        expected[i] = w;
        sum += w;
    }
    for (0..k) |i| {
        expected[i] /= sum;
        try std.testing.expectApproxEqAbs(expected[i], weights[i], 1e-6);
    }
}

test "topKSoftmaxWeight k=1 picks argmax with weight 1.0" {
    const logits = [_]f32{ -9.0, -2.0, -5.0 };
    var ids: [1]u32 = undefined;
    var weights: [1]f32 = undefined;
    topKSoftmaxWeight(&logits, 1, &ids, &weights);
    try std.testing.expectEqual(@as(u32, 1), ids[0]);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), weights[0], 0.001);
}

test "effectiveRopeAttnScale is neutral without YaRN metadata" {
    var cfg = makeTestModelConfig();
    cfg.architecture = .gpt_oss;
    cfg.rope_scaling_factor = 1.0;
    cfg.rope_attn_factor = 1.75;
    cfg.rope_original_context = 4096;

    try std.testing.expectApproxEqAbs(@as(f32, 1.0), effectiveRopeAttnScale(&cfg), 1e-6);
}

test "effectiveRopeAttnScale uses GGUF attention factor for YaRN" {
    var cfg = makeTestModelConfig();
    cfg.architecture = .gpt_oss;
    cfg.rope_scaling_factor = 32.0;
    cfg.rope_attn_factor = 1.75;
    cfg.rope_original_context = 4096;

    const expected = cfg.rope_attn_factor * (1.0 + 0.1 * @log(cfg.rope_scaling_factor));
    try std.testing.expectApproxEqAbs(expected, effectiveRopeAttnScale(&cfg), 1e-6);
}

// ---------------------------------------------------------------------------
// expertSliceBytes — additional quant types
// ---------------------------------------------------------------------------

test "expertSliceBytes Q6_K" {
    // Q6_K: block_size=256, bytes_per_block=210
    const result = expertSliceBytes(.q6_k, 256, 2048);
    // blocks_per_row = 2048/256 = 8, bytes = 256 * 8 * 210 = 430,080
    try std.testing.expectEqual(@as(u32, 430_080), result);
}

test "expertSliceBytes Q8_0" {
    // Q8_0: block_size=32, bytes_per_block=34
    const result = expertSliceBytes(.q8_0, 2048, 2048);
    // blocks_per_row = 2048/32 = 64, bytes = 2048 * 64 * 34 = 4,456,448
    try std.testing.expectEqual(@as(u32, 4_456_448), result);
}

test "expertSliceBytes F16" {
    // F16: block_size=1, bytes_per_block=2
    const result = expertSliceBytes(.f16, 512, 2048);
    // blocks_per_row = 2048/1 = 2048, bytes = 512 * 2048 * 2 = 2,097,152
    try std.testing.expectEqual(@as(u32, 2_097_152), result);
}

// ---------------------------------------------------------------------------
// readMmapFloats — f16/f32 tensor reading
// ---------------------------------------------------------------------------

test "readMmapFloats f16 matches dequantRow f16" {
    var raw: [8]u8 = undefined;
    const vals = [_]f16{ 1.0, -0.5, 0.25, 100.0 };
    for (vals, 0..) |v, i| {
        const bits = @as(u16, @bitCast(v));
        raw[i * 2] = @truncate(bits);
        raw[i * 2 + 1] = @truncate(bits >> 8);
    }
    var out_mmap: [4]f32 = undefined;
    var out_dequant: [4]f32 = undefined;
    readMmapFloats(&raw, 0, .f16, &out_mmap);
    dequantRow(&raw, 0, 4, .f16, &out_dequant);
    for (0..4) |i| {
        try std.testing.expectEqual(out_mmap[i], out_dequant[i]);
    }
}

test "delta-net zero state produces beta*v*(k.q) output" {
    // With zero initial state, the delta-net autoregressive step gives:
    // o[row] = beta * v[row] * dot(k, q)
    // This verifies the core SSM math matches the CPU reference.
    const head_v_dim: usize = 4;
    const d_state: usize = 4;
    var ssm_state = [_]f32{0} ** (head_v_dim * head_v_dim);

    const k_head = [_]f32{ 0.5, -0.3, 0.1, 0.7 };
    const v_head = [_]f32{ 1.0, -2.0, 0.5, 0.3 };
    const q_head = [_]f32{ 0.2, 0.4, -0.1, 0.6 };
    const beta: f32 = 0.8;
    const gate: f32 = -0.1; // exp(gate) ≈ 0.905

    // Decay (no-op for zero state)
    const g_val = @exp(gate);
    for (&ssm_state) |*s| s.* *= g_val;

    // Update: for each row, sk = s@k = 0, d = beta*(v-0) = beta*v
    // s[row][col] += k[col] * d_val
    for (0..head_v_dim) |row| {
        var sk: f32 = 0;
        for (0..d_state) |col| sk += ssm_state[row * head_v_dim + col] * k_head[col];
        const d_val = beta * (v_head[row] - sk);
        for (0..d_state) |col| {
            ssm_state[row * head_v_dim + col] += k_head[col] * d_val;
        }
    }

    // Read: o[row] = sum_col s[row][col] * q[col]
    var output: [4]f32 = undefined;
    for (0..head_v_dim) |row| {
        var val: f32 = 0;
        for (0..d_state) |col| {
            val += ssm_state[row * head_v_dim + col] * q_head[col];
        }
        output[row] = val;
    }

    // Expected values for the row-major delta-net update used by the GPU shader:
    // s[row*4+col] += k[col]*d, read: o[row] = sum_col s[row*4+col]*q[col]
    const expected = [_]f32{ 0.312, -0.624, 0.156, 0.0936 };
    for (0..head_v_dim) |row| {
        try std.testing.expect(@abs(output[row] - expected[row]) < 1e-4);
    }
}

test "l2Normalize produces unit vector" {
    var v = [_]f32{ 3.0, 4.0, 0.0 };
    // Inline l2Normalize (it's a private struct method)
    var sum_sq: f32 = 0;
    for (v) |x| sum_sq += x * x;
    const norm = @sqrt(sum_sq + 1e-12);
    if (norm > 0) {
        for (&v) |*x| x.* /= norm;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0.6), v[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.8), v[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v[2], 1e-6);
}

test "l2Normalize zero vector stays zero" {
    var v = [_]f32{ 0.0, 0.0, 0.0 };
    var sum_sq: f32 = 0;
    for (v) |x| sum_sq += x * x;
    const norm = @sqrt(sum_sq + 1e-12);
    if (norm > 0) {
        for (&v) |*x| x.* /= norm;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v[0], 1e-12);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), v[1], 1e-12);
}

test "SiLU activation: x * sigmoid(x)" {
    const silu = struct {
        fn f(x: f32) f32 {
            return x / (1.0 + @exp(-x));
        }
    }.f;
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), silu(0.0), 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, 0.7310586), silu(1.0), 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -0.2689414), silu(-1.0), 1e-5);
    try std.testing.expect(@abs(silu(10.0) - 10.0) < 0.001);
}

test "gated norm: RMS_norm(o) * weight * SiLU(z)" {
    const head_v_dim: usize = 4;
    const o = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const z = [_]f32{ 0.5, -0.5, 1.0, -1.0 };
    const w = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    var sq: f32 = 0;
    for (o) |v| sq += v * v;
    const rms = @sqrt(sq / @as(f32, @floatFromInt(head_v_dim)) + 1e-6);
    var result: [4]f32 = undefined;
    for (0..head_v_dim) |i| {
        const nv = (o[i] / rms) * w[i];
        const zv = z[i];
        const gate = zv / (1.0 + @exp(-zv));
        result[i] = nv * gate;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 2.7386), rms, 1e-3);
    try std.testing.expectApproxEqAbs(@as(f32, 0.1137), result[0], 0.01);
    try std.testing.expect(result[3] < 0);
}

test "conv1d sliding window: convolve then shift state" {
    const conv_channels: usize = 3;
    const d_conv: usize = 3;
    const d_conv_1 = d_conv - 1;
    var state = [_]f32{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };
    const current = [_]f32{ 7.0, 8.0, 9.0 };
    const kernel = [_]f32{ 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0 };
    var conv_out: [3]f32 = undefined;
    for (0..conv_channels) |ch| {
        var sum: f32 = 0;
        for (0..d_conv) |ki| {
            const kw = kernel[ch * d_conv + ki];
            const sv = if (ki < d_conv_1) state[ki * conv_channels + ch] else current[ch];
            sum += kw * sv;
        }
        conv_out[ch] = sum;
    }
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), conv_out[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), conv_out[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), conv_out[2], 1e-6);
    if (d_conv_1 > 1) {
        const shift = (d_conv_1 - 1) * conv_channels;
        std.mem.copyForwards(f32, state[0..shift], state[conv_channels .. shift + conv_channels]);
    }
    @memcpy(state[(d_conv_1 - 1) * conv_channels ..][0..conv_channels], &current);
    try std.testing.expectApproxEqAbs(@as(f32, 4.0), state[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 5.0), state[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), state[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 7.0), state[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), state[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), state[5], 1e-6);
}

test "kv page count rounds context up to 16-token pages" {
    try std.testing.expectEqual(@as(u32, 0), kvPageCountForContext(0));
    try std.testing.expectEqual(@as(u32, 1), kvPageCountForContext(1));
    try std.testing.expectEqual(@as(u32, 1), kvPageCountForContext(16));
    try std.testing.expectEqual(@as(u32, 2), kvPageCountForContext(17));
    try std.testing.expectEqual(@as(u32, 256), kvPageCountForContext(4096));
}

test "kv page ids sort ascending for stable logical order" {
    var page_ids = [_]u32{ 7, 2, 5, 1 };
    sortPageIdsAscending(&page_ids);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 5, 7 }, &page_ids);
}

test "logical token maps through paged kv table" {
    const page_ids = [_]u32{ 3, 1, 4 };
    try std.testing.expectEqual(@as(u32, 48), try logicalTokenToPhysicalToken(&page_ids, 0));
    try std.testing.expectEqual(@as(u32, 63), try logicalTokenToPhysicalToken(&page_ids, 15));
    try std.testing.expectEqual(@as(u32, 16), try logicalTokenToPhysicalToken(&page_ids, 16));
    try std.testing.expectEqual(@as(u32, 18), try logicalTokenToPhysicalToken(&page_ids, 18));
    try std.testing.expectEqual(@as(u32, 64), try logicalTokenToPhysicalToken(&page_ids, 32));
}

test "request budget keeps small generations on fewer kv pages" {
    const small = memory_plan.requestBudget(64, 64, 4096);
    const large = memory_plan.requestBudget(64, 4096, 4096);
    const near_full = memory_plan.requestBudget(4090, 64, 4096);

    try std.testing.expectEqual(@as(u32, 128), small.target_context_tokens);
    try std.testing.expectEqual(@as(u32, 8), kvPageCountForContext(small.target_context_tokens));

    try std.testing.expectEqual(@as(u32, 4096), large.target_context_tokens);
    try std.testing.expectEqual(@as(u32, 256), kvPageCountForContext(large.target_context_tokens));

    try std.testing.expectEqual(@as(u32, 6), near_full.completion_tokens);
    try std.testing.expectEqual(@as(u32, 4096), near_full.target_context_tokens);
    try std.testing.expectEqual(@as(u32, 256), kvPageCountForContext(near_full.target_context_tokens));
}

test "push constant struct sizes match GLSL expectations" {
    const ew = @import("elementwise.zig");
    try std.testing.expectEqual(@as(usize, 16), @sizeOf(ew.SsmConv1dPush));
    try std.testing.expectEqual(@as(usize, 52), @sizeOf(ew.SsmDeltaNetPush));
    try std.testing.expectEqual(@as(usize, 20), @sizeOf(ew.SsmGatedNormPush));
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(ew.SoftmaxTopkPush));
}
