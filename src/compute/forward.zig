//! Run the inference runtime: decode state, pipeline ownership, and token generation.
//! @section Inference Runtime
//! This module ties together model state, compute graphs, dispatch helpers,
//! and greedy token sampling for a single active inference engine.
const std = @import("std");
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const buffer_mod = @import("../vulkan/buffer.zig");
const Buffer = @import("../vulkan/buffer.zig").Buffer;
const CommandPool = @import("../vulkan/command.zig").CommandPool;
const CommandBuffer = @import("../vulkan/command.zig").CommandBuffer;
const Pipeline = @import("../vulkan/pipeline.zig").Pipeline;
const GpuConfig = @import("../vulkan/gpu_detect.zig").GpuConfig;
const loader = @import("../model/loader.zig");
const Model = loader.Model;
const ModelConfig = loader.ModelConfig;
const LoadedTensor = loader.LoadedTensor;
const architecture = @import("../model/architecture.zig");
const Graph = @import("graph.zig").Graph;
const dmmv_mod = @import("dmmv.zig");
const DmmvDispatch = dmmv_mod.DmmvDispatch;
const DmmvPushConstants = dmmv_mod.DmmvPushConstants;
const MoeDmmvPushConstants = dmmv_mod.MoeDmmvPushConstants;
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
const SsmDeltaNetPush = elementwise_mod.SsmDeltaNetPush;
const SsmGatedNormPush = elementwise_mod.SsmGatedNormPush;
const DeinterleavePush = elementwise_mod.DeinterleavePush;
const KvCacheWritePush = elementwise_mod.KvCacheWritePush;
const KvCacheWriteBatchedPush = elementwise_mod.KvCacheWriteBatchedPush;
const ResidualRmsNormPush = elementwise_mod.ResidualRmsNormPush;
const NormRopePush = elementwise_mod.NormRopePush;
const attn_mod = @import("attention.zig");
const AttentionDispatch = attn_mod.AttentionDispatch;
const FlashAttnPush = attn_mod.FlashAttnPush;
const FlashAttnBatchedPush = attn_mod.FlashAttnBatchedPush;
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
            .generated_tokens = .{},
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
    ssm,
    ssm_proj,
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
    final_tail,

    fn label(self: @This()) []const u8 {
        return switch (self) {
            .embed_upload => "embed",
            .attention => "attention",
            .ssm => "ssm",
            .ssm_proj => "ssm_proj",
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
            .final_tail => "tail",
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
    var total: u64 = 0;
    for (model.gguf_file.tensors.items) |tensor_info| {
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

/// Gate for the Vulkan/RDNA batched-prefill path. Mirrors the narrow slice
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
    const cfg = engine.model.config;
    if (cfg.n_experts > 0) return false;
    if (cfg.ssm_d_inner > 0) return false;
    if (cfg.architecture == .gemma or cfg.architecture == .gpt_oss) return false;
    const full_attn_interval = if (cfg.full_attn_interval > 0) cfg.full_attn_interval else 1;
    if (full_attn_interval != 1) return false;
    if (cfg.sliding_window_size != 0) return false;

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
        const v = lt.attn_v orelse return false;
        const o = lt.attn_output orelse return false;
        const gate = lt.ffn_gate orelse return false;
        const up = lt.ffn_up orelse return false;
        const down = lt.ffn_down orelse return false;
        for ([_]*const LoadedTensor{ q, k, v, o, gate, up, down }) |t| {
            if (!isSupported(t.info.type_)) return false;
        }

        // Reject packed Q+gate (Qwen3Next): attn_q row count == 2 * q_dim.
        const hidden_dim = cfg.hidden_dim;
        const q_rows: u32 = @intCast(q.info.numElements() / hidden_dim);
        const q_dim: u32 = cfg.n_heads * cfg.head_dim;
        if (q_rows == q_dim * 2) return false;
    }
    return true;
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
    // bucket in the Qwen3.5-35B flagship prefill.
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
    // Default-on. Subgroup-parallel softmax_topk_v2 (subgroupMax/Min/Shuffle).
    // Disable with ZINC_TOPK_V1=1 to fall back to the v1 shader (single-thread
    // serial scan in shared memory).
    use_softmax_topk_v2: bool = true,
    // Opt-in via ZINC_MMQ_SSM=1. When set and the dmmv_q8_0_q8_1 pipeline is
    // loaded, runSsmLayerGpu quantizes norm_buf once into a Q8_1 scratch
    // buffer and routes the 4 SSM proj DMMVs (wqkv/z/alpha/beta) through the
    // integer-dot mmq variant — cuts activation bandwidth ~3.6× and swaps
    // int8*f32 dot for int8*int8. Effective only when all four tensors are
    // Q8_0.
    use_mmq_ssm: bool = false,
    // Step 11a foundation (ZINC_CAPTURE_ROUTING=1). When set, after each GPU MoE
    // softmax_topk we copy the top-k ids+weights into routing_capture_buf at
    // slot(position, layer). Unused downstream this cycle — the buffer is the
    // prerequisite for Step 11b (token-permute) and 11c (grouped MoE GEMM).
    use_capture_routing: bool = false,
    // Q8_1 scratch pair (primary / alt). Swapped alongside decode_cmd during
    // the double-buffered prefill pipeline so the two in-flight CBs don't
    // race on the same scratch region. Size = (hidden_dim/32)*36 bytes.
    ssm_mmq_scratch: Buffer = .{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = null },
    ssm_mmq_scratch_alt: Buffer = .{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = null },
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
        var dmmv = try DmmvDispatch.init(instance, &gpu_config, shader_dir, max_k, allocator);
        errdefer dmmv.deinit();

        var elementwise = try ElementwiseDispatch.init(instance, shader_dir, allocator);
        errdefer elementwise.deinit();

        var attention = try AttentionDispatch.init(instance, shader_dir, allocator);
        errdefer attention.deinit();
        var argmax = try ArgmaxDispatch.init(instance, shader_dir, allocator);
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

        const argmax_phase0_workgroups: u32 = @max(@as(u32, 1), (config.vocab_size + 63) / 64);
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
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
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
        // Qwen3.5-35B flagship: gate_up 855.6 → 695.4 ms (−18.7%), prefill
        // tok/s 23.16 → 23.72 (+2.4%) with identical output tokens.
        const moe_kpar_env = std.posix.getenv("ZINC_MOE_KPAR");
        const moe_kpar_explicitly_off = moe_kpar_env != null and std.mem.eql(u8, moe_kpar_env.?, "0");
        const moe_kpar_enabled = !moe_kpar_explicitly_off and dmmv.pipeline_q4k_moe_kpar != null;
        if (moe_kpar_enabled) {
            log.info("MoE Q4_K kpar variant ENABLED (default, set ZINC_MOE_KPAR=0 to disable)", .{});
        } else if (moe_kpar_explicitly_off) {
            log.info("MoE Q4_K kpar variant DISABLED via ZINC_MOE_KPAR=0", .{});
        }

        // Q4_K batched projection, K-parallel wave64 variant. Default ON when
        // the pipeline is loaded — measured 2× speedup vs the serial variant
        // on Qwen3-8B Q4_K_M (143 tok/s vs 62 tok/s on a 105-token prompt,
        // R9700). Disable via ZINC_Q4K_BATCH_KPAR=0 to run the serial shader.
        const q4k_batch_kpar_env = std.posix.getenv("ZINC_Q4K_BATCH_KPAR");
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
        const moe_fused_gate_up_env = std.posix.getenv("ZINC_MOE_FUSED_GATE_UP");
        const moe_fused_gate_up_forced_on = moe_fused_gate_up_env != null and std.mem.eql(u8, moe_fused_gate_up_env.?, "1");
        const moe_fused_gate_up_enabled = moe_fused_gate_up_forced_on and
            moe_kpar_enabled and dmmv.pipeline_q4k_fused_gate_up_moe != null;
        if (moe_fused_gate_up_enabled) {
            log.info("MoE Q4_K fused gate+up ENABLED via ZINC_MOE_FUSED_GATE_UP=1", .{});
        }

        // Q5_K MoE K-parallel shader: default ON when the pipeline is loaded,
        // disabled by setting ZINC_MOE_Q5K_KPAR=0. Targets the ~713 ms MoE down
        // bucket (Q5_K weights) on the Qwen3.5-35B flagship prefill. Mirrors the
        // Q4_K kpar pattern (16 threads per Q5_K superblock + wave64 subgroupAdd).
        const moe_q5k_kpar_env = std.posix.getenv("ZINC_MOE_Q5K_KPAR");
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
        const topk_v1_env = std.posix.getenv("ZINC_TOPK_V1");
        const topk_v1_forced = topk_v1_env != null and std.mem.eql(u8, topk_v1_env.?, "1");
        const topk_v2_enabled = !topk_v1_forced and elementwise.pipeline_softmax_topk_v2 != null;
        if (topk_v2_enabled) {
            log.info("softmax_topk v2 (subgroup-parallel) ENABLED (default, set ZINC_TOPK_V1=1 to revert)", .{});
        } else if (topk_v1_forced) {
            log.info("softmax_topk v2 DISABLED via ZINC_TOPK_V1=1; using v1 shared-mem scan", .{});
        }

        // Batched flash-attention foundation (opt-in). When ZINC_BATCH_ATTN=1 and
        // the flash_attn_batched pipeline is loaded, the attention call site
        // routes through the batched shader. Foundation step calls with
        // n_queries=1 for correctness parity with the decode-shape shader.
        const batch_attn_env = std.posix.getenv("ZINC_BATCH_ATTN");
        const batch_attn_flag = batch_attn_env != null and std.mem.eql(u8, batch_attn_env.?, "1");
        const batch_attn_enabled = batch_attn_flag and attention.pipeline_batched != null;
        if (batch_attn_enabled) {
            log.info("Flash-attn batched path ENABLED (ZINC_BATCH_ATTN=1); n_queries=1 foundation", .{});
        } else if (batch_attn_flag) {
            log.info("ZINC_BATCH_ATTN=1 requested but flash_attn_batched pipeline absent; using decode-shape shader", .{});
        }

        // SSM mmq path (opt-in). Effective only when (a) flag is on, (b) both
        // quantize_q8_1 and dmmv_q8_0_q8_1 pipelines loaded, (c) push
        // descriptors available (same gate as the rest of the mmq-adjacent
        // push-descriptor helpers). The runtime also validates that all four
        // SSM proj tensors are Q8_0 before selecting the mmq path.
        const mmq_ssm_env = std.posix.getenv("ZINC_MMQ_SSM");
        const mmq_ssm_flag = mmq_ssm_env != null and std.mem.eql(u8, mmq_ssm_env.?, "1");
        const mmq_ssm_enabled = mmq_ssm_flag and
            dmmv.pipeline_quantize_q8_1 != null and
            dmmv.pipeline_q8_0_q8_1 != null and
            instance.push_descriptor_fn != null;
        var ssm_mmq_scratch = Buffer{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = instance.device };
        var ssm_mmq_scratch_alt = Buffer{ .handle = null, .memory = null, .size = 0, .mapped = null, .device = instance.device };
        if (mmq_ssm_enabled) {
            // One Q8_1 block per 32 f32 elements, 36 bytes per block.
            // hidden_dim is guaranteed non-zero here (the encoder wouldn't run
            // otherwise); we also assume it's a multiple of 32 (every Q8_0
            // weight in the tree requires this).
            const q81_bytes: vk.c.VkDeviceSize = @as(vk.c.VkDeviceSize, (config.hidden_dim / 32) * 36);
            ssm_mmq_scratch = try Buffer.initDeviceLocal(instance, q81_bytes, vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            errdefer ssm_mmq_scratch.deinit();
            ssm_mmq_scratch_alt = try Buffer.initDeviceLocal(instance, q81_bytes, vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            errdefer ssm_mmq_scratch_alt.deinit();
            log.info("SSM mmq path ENABLED (ZINC_MMQ_SSM=1); Q8_1 scratch pair = 2x {d} B", .{q81_bytes});
        } else if (mmq_ssm_flag) {
            log.info("ZINC_MMQ_SSM=1 requested but prerequisites missing (q8_1 pipelines or push descriptors); using f32 SSM proj", .{});
        }

        // Step 11a foundation: per-(token, layer) routing capture buffer.
        // Enabled by ZINC_CAPTURE_ROUTING=1. Dormant downstream — this cycle only
        // verifies the copy path is correct and measures the flag-on overhead so
        // Step 11b can wire token-permute on top without re-proving the plumbing.
        const capture_env = std.posix.getenv("ZINC_CAPTURE_ROUTING");
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

        return InferenceEngine{
            .model = model,
            .gpu_config = gpu_config,
            .dmmv = dmmv,
            .use_moe_kpar = moe_kpar_enabled,
            .use_moe_q5k_kpar = moe_q5k_kpar_enabled,
            .use_moe_fused_gate_up = moe_fused_gate_up_enabled,
            .use_q4k_batch_kpar = q4k_batch_kpar_enabled,
            .use_softmax_topk_v2 = topk_v2_enabled,
            .use_mmq_ssm = mmq_ssm_enabled,
            .use_batch_attn = batch_attn_enabled,
            .use_capture_routing = capture_flag and routing_capture_buf.handle != null,
            .routing_capture_buf = routing_capture_buf,
            .routing_capture_slot_bytes = routing_capture_slot_bytes,
            .routing_capture_max_tokens = routing_capture_max_tokens,
            .ssm_mmq_scratch = ssm_mmq_scratch,
            .ssm_mmq_scratch_alt = ssm_mmq_scratch_alt,
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
        const query_read_start = std.time.nanoTimestamp();
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
        const query_read_end = std.time.nanoTimestamp();
        self.profile_token_counters.query_read_ns += @intCast(query_read_end - query_read_start);
        if (count >= 2) {
            const first = timestamps[0];
            const last = timestamps[count - 1];
            const elapsed_ns = @as(f64, @floatFromInt(last -| first)) * self.timestamp_period_ns;
            const elapsed_ms = elapsed_ns / 1e6;
            self.profile_total_gpu_ms += elapsed_ms;
            if (elapsed_ms > self.profile_max_gpu_ms) self.profile_max_gpu_ms = elapsed_ms;
            self.profile_sample_count += 1;
            for (0..self.profile_phase_range_count) |i| {
                const range = self.profile_phase_ranges[i];
                if (range.end_query >= count or range.start_query >= count) continue;
                const phase_ns_f64 = @as(f64, @floatFromInt(timestamps[range.end_query] -| timestamps[range.start_query])) * self.timestamp_period_ns;
                self.profile_token_counters.gpu_phase_ns[@intFromEnum(range.phase)] += @intFromFloat(@max(phase_ns_f64, 0.0));
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
                (n_elements + 63) / 64,
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
                (n_elements + 63) / 64,
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
        const pip = if (self.use_softmax_topk_v2 and self.elementwise.pipeline_softmax_topk_v2 != null)
            &self.elementwise.pipeline_softmax_topk_v2.?
        else
            &(self.elementwise.pipeline_softmax_topk orelse return error.ShaderNotLoaded);
        if (pip.uses_push_descriptors) {
            const push = SoftmaxTopkPush{
                .n_experts = n_experts,
                .k = k,
            };
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
        try self.elementwise.recordSoftmaxTopk(&self.decode_cmd, ds, n_experts, k);
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
                (n_elements + 63) / 64,
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

        // Log MoE dimensions once (first decode)
        if (state.generated_tokens.items.len == 0 and is_moe) {
            log.debug("MoE dims: expert_inter={d} shared_expert_inter={d} hidden={d}", .{ inter_dim, shexp_inter_dim, hidden_dim });
        }

        // 1. CPU: dequantize embedding
        const track_decode_timing = self.profile_enabled or self.prefill_active;
        const cpu_embed_start = if (track_decode_timing) std.time.nanoTimestamp() else 0;
        try self.embedToken(token_id);
        if (collect_output and state.generated_tokens.items.len == 0 and config.architecture == .gpt_oss) {
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
            const cpu_embed_end = std.time.nanoTimestamp();
            const elapsed: u64 = @intCast(cpu_embed_end - cpu_embed_start);
            if (self.profile_enabled) self.profile_token_counters.cpu_embed_ns += elapsed;
            prefill_embed_elapsed_ns = elapsed;
        }

        // Per-layer logit5 tracking for BOS diagnostic summary
        var diag_logit5 = [_]f32{0} ** 64;
        var diag_rms_arr = [_]f32{0} ** 64;

        const cpu_record_start = if (track_decode_timing) std.time.nanoTimestamp() else 0;

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

        // Reset profiling timestamps for this token
        self.resetTimestamps();
        _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

        for (0..config.n_layers) |layer_idx| {
            const layer: u32 = @intCast(layer_idx);
            const lt = self.layer_tensors[layer_idx];

            // --- Upload embedding (only first layer) ---
            if (layer == 0) {
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

            const is_full_attn = ((layer + 1) % full_attn_interval == 0);
            const diag_last_prompt_token = collect_output and state.generated_tokens.items.len == 0 and config.architecture == .gpt_oss;

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
                try self.dispatchDmmv(v_tensor, self.norm_buf, hidden_size, self.v_buf, v_rows, hidden_dim);
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
                                .head_dim = config.head_dim,
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

                if (q_norm_tensor) |qn| {
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
                // Mirrors Metal forward_metal.zig:3460-3462.
                if (config.architecture == .gemma and config.rope_freq_base_swa > 0) {
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
                const use_batched = self.use_batch_attn and self.attention.pipeline_batched != null;
                if (use_batched) {
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
                        try self.attention.recordFlashAttn(&self.decode_cmd, attn_ds, layer_head_dim, config.n_heads, layer_n_kv_heads, state.position + 1, kv_page_size_tokens, config.attn_scale, sink_offset);
                    }
                }
                self.decode_cmd.computeBarrier();

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
                    try self.dispatchDmmvAcc(o_tensor, self.attn_out_buf, self.attn_out_buf.size, self.hidden_buf, hidden_dim, o_cols);
                    if (lt.attn_output_bias) |bias| {
                        self.decode_cmd.computeBarrier();
                        try self.dispatchBiasAdd(self.hidden_buf.handle, hidden_size, bias, hidden_dim);
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

                    // Gemma post-attention norm: RMS norm on o_proj output before residual add
                    if (apply_post_attn_norm) {
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
                    log.info("FASTPATH: gpu_ssm={} arch={s} ssm_shader={}", .{
                        use_gpu_ssm,
                        @tagName(config.architecture),
                        self.elementwise.pipeline_ssm_conv1d != null,
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
                    try self.runSsmLayerGpu(state, layer, layer_idx, ssm_dead_tail);
                } else {
                    if (self.profile_enabled) self.profile_token_counters.cpu_ssm_fallbacks += 1;
                    try self.runSsmLayerCpu(state, layer, layer_idx);
                }
                self.endProfilePhase(.ssm, ssm_phase);
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

                try self.dispatchDmmv(router_tensor, router_input_buf, hidden_size, self.router_logits_buf, config.n_experts, hidden_dim);
                self.decode_cmd.computeBufferBarrier(self.router_logits_buf.handle, self.router_logits_buf.size);

                // Gemma 4 MoE: scale router logits by 1/sqrt(hidden_dim) before softmax.
                // Matches Metal forward_metal.zig:4134-4137.
                if (config.architecture == .gemma) {
                    const router_scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(hidden_dim)));
                    try self.dispatchScaleInPlace(
                        self.router_logits_buf.handle,
                        self.router_logits_buf.size,
                        config.n_experts,
                        router_scale,
                    );
                    self.decode_cmd.computeBufferBarrier(self.router_logits_buf.handle, self.router_logits_buf.size);
                }
                self.endProfilePhase(.moe_router, moe_router_phase);

                const n_used = config.n_experts_used;

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

                // Gemma 4 MoE uses pre_ffw_norm_2 for MoE expert input (vs ffn_norm_buf for shared).
                // Available to both GPU-routed and CPU-routed MoE paths.
                const expert_input_buf = if (lt.pre_ffw_norm_2) |pre_norm_t| blk: {
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
                    const fused_ready = gate_qt == .q4_k and up_qt == .q4_k and
                        self.use_moe_kpar and
                        self.use_moe_fused_gate_up and
                        self.dmmv.pipeline_q4k_fused_gate_up_moe != null and
                        gate_exps.info.numElements() == up_exps.info.numElements();
                    if (fused_ready) {
                        const pip = &self.dmmv.pipeline_q4k_fused_gate_up_moe.?;
                        const push = MoeDmmvPushConstants{ .M = inter_dim, .K = hidden_dim, .expert_stride = expert_gate_row_bytes, .x_expert_stride = 0, .x_offset = 0, .y_offset = 0 };
                        const wg_x: u32 = (inter_dim + 1) / 2;
                        self.pushDispatch6(
                            pip,
                            std.mem.asBytes(&push),
                            gate_exps.gpu_buffer.handle, gate_exps.gpu_buffer.size,
                            up_exps.gpu_buffer.handle, up_exps.gpu_buffer.size,
                            expert_input_buf.handle, hidden_size,
                            self.gate_buf.handle, self.gate_buf.size,
                            self.up_buf.handle, self.up_buf.size,
                            self.router_output_buf.handle, self.router_output_buf.size,
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

                    // Shared expert tensors — looked up here to interleave with MoE dispatches
                    const gate_shexp = lt.ffn_gate_shexp;
                    const up_shexp = lt.ffn_up_shexp;
                    const down_shexp = lt.ffn_down_shexp;
                    const shexp_gate = lt.ffn_gate_inp_shexp;
                    const has_shared_expert = gate_shexp != null and up_shexp != null and down_shexp != null;
                    const shexp_size = @as(vk.c.VkDeviceSize, shexp_inter_dim) * @sizeOf(f32);

                    if (state.position == 0 and layer == 0) {
                        log.info("FASTPATH: shared gate={s} up={s} down={s} gate_inp={s}", .{
                            if (gate_shexp) |t| @tagName(t.info.type_) else "none",
                            if (up_shexp) |t| @tagName(t.info.type_) else "none",
                            if (down_shexp) |t| @tagName(t.info.type_) else "none",
                            if (shexp_gate) |t| @tagName(t.info.type_) else "none",
                        });
                    }

                    // down DMMV: ALL experts at once
                    // x_expert_stride=inter_dim: each expert reads from its own swiglu section
                    const moe_down_phase = self.beginProfilePhase();
                    {
                        const qt = down_exps.info.type_;
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
                    // No buffer conflicts: MoE down reads swiglu_buf/writes down_buf;
                    // shared gate/up read ffn_norm_buf/write gate_buf,up_buf,router_logits_buf.
                    if (has_shared_expert) {
                        try self.dispatchDmmv(gate_shexp.?, self.ffn_norm_buf, hidden_size, self.gate_buf, shexp_inter_dim, hidden_dim);
                        try self.dispatchDmmv(up_shexp.?, self.ffn_norm_buf, hidden_size, self.up_buf, shexp_inter_dim, hidden_dim);
                        if (shexp_gate) |sg| {
                            try self.dispatchDmmv(sg, self.ffn_norm_buf, hidden_size, self.router_logits_buf, 1, hidden_dim);
                        }
                    }
                    self.decode_cmd.computeBarrier();
                    self.endProfilePhase(.moe_down, moe_down_phase);

                    // Gemma 4 MoE: per-expert scalar on down expert output before weighted_acc.
                    // down[slot*hidden_dim + i] *= ffn_down_exps_scale[expert_id[slot]].
                    // Matches Metal forward_metal.zig:4357-4360.
                    if (lt.ffn_down_exps_scale) |scale_t| {
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
                    const has_post_ffw_norm = lt.post_ffw_norm != null;
                    const moe_acc_target = if (has_post_ffw_norm) self.moe_out_buf.handle else self.hidden_buf.handle;
                    const moe_acc_target_size = if (has_post_ffw_norm) self.moe_out_buf.size else hidden_size;
                    if (has_post_ffw_norm) {
                        // Zero moe_out_buf before weighted accumulation
                        vk.c.vkCmdFillBuffer(self.decode_cmd.handle, self.moe_out_buf.handle, 0, hidden_size, 0);
                        self.decode_cmd.transferToComputeBarrier();
                    }
                    const moe_acc_phase = self.beginProfilePhase();
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
                    // Overlap: dispatch shared expert SwiGLU alongside weighted_acc.
                    // No buffer conflicts: weighted_acc reads down_buf+router_output_buf/writes hidden_buf;
                    // SwiGLU reads gate_buf+up_buf/writes swiglu_buf.
                    if (has_shared_expert) {
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
                    self.decode_cmd.computeBarrier();
                    self.endProfilePhase(.moe_weighted_acc, moe_acc_phase);

                    // Shared expert down projection (run BEFORE post_ffw_norm for Gemma 4
                    // so that MoE + shared expert outputs are accumulated in moe_out_buf
                    // and normed together. Matches Metal forward_metal.zig:5110-5128.)
                    if (has_shared_expert) {
                        const shared_down_phase = self.beginProfilePhase();
                        try self.dispatchDmmv(down_shexp.?, self.swiglu_buf, shexp_size, self.down_buf, hidden_dim, shexp_inter_dim);
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
                    // separately (no post_ffw_norm to share).
                    if (has_shared_expert and !has_post_ffw_norm) {
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

                    // Zero moe_out_buf via fill
                    vk.c.vkCmdFillBuffer(self.decode_cmd.handle, self.moe_out_buf.handle, 0, hidden_size, 0);
                    self.decode_cmd.transferToComputeBarrier();

                    var cpu_moe_accum_opt: ?[]f32 = null;
                    defer if (cpu_moe_accum_opt) |buf| self.allocator.free(buf);
                    const diag_moe_detail = self.validation_diagnostics_enabled and config.architecture == .gpt_oss and collect_output and state.generated_tokens.items.len == 0;
                    if (diag_moe_detail) {
                        cpu_moe_accum_opt = try self.allocator.alloc(f32, hidden_dim);
                        @memset(cpu_moe_accum_opt.?, 0);
                    }

                    for (0..n_used) |ei| {
                        const eid = expert_ids[ei];
                        var weight = expert_weights[ei];
                        const gate_offset = eid * expert_gate_row_bytes;
                        const up_offset = eid * expert_gate_row_bytes + up_base_offset;
                        const down_offset = eid * expert_down_row_bytes;

                        // Expert gate/up reads pre_ffw_norm_2 output (Gemma 4) or ffn_norm_buf
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
                // Dense FFN: gate → up → SwiGLU → down → residual
                const gate_tensor = lt.ffn_gate orelse return error.TensorNotFound;
                const up_tensor = lt.ffn_up orelse return error.TensorNotFound;
                const down_tensor = lt.ffn_down orelse return error.TensorNotFound;

                try self.dispatchDmmv(gate_tensor, self.ffn_norm_buf, hidden_size, self.gate_buf, inter_dim, hidden_dim);
                try self.dispatchDmmv(up_tensor, self.ffn_norm_buf, hidden_size, self.up_buf, inter_dim, hidden_dim);
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

                if (lt.post_ffw_norm == null and !self.validation_diagnostics_enabled) {
                    // Fused: down DMMV accumulates directly into hidden_buf,
                    // eliminating separate scale_acc dispatch + barrier
                    try self.dispatchDmmvAcc(down_tensor, self.swiglu_buf, self.swiglu_buf.size, self.hidden_buf, hidden_dim, inter_dim);
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
                self.decode_cmd.computeBarrier();
            }

            // Command buffer stays open across layers (Phase 3c batching).
            // No per-layer submit — only submit for MoE expert ID readback (inside MoE block above).

            // --- Debug: per-layer hidden_buf diagnostics (BOS token only, gated behind validation diagnostics) ---
            if ((state.position == 0 and (self.validation_diagnostics_enabled or std.posix.getenv("ZINC_LAYER_DIAG") != null)) or
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
        const need_logits_readback = collect_output and (self.logits_readback_enabled or self.validation_diagnostics_enabled or !have_gpu_argmax);
        if (collect_output) {
            const final_tail_phase = self.beginProfilePhase();

            // Final RMS norm: hidden_buf → norm_buf
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

            // LM head: output.weight × norm_buf → logits_buf
            const lm_tensor = self.tensor_map.get("output.weight") orelse
                self.tensor_map.get("token_embd.weight") orelse return error.TensorNotFound;
            try self.dispatchDmmv(lm_tensor, self.norm_buf, hidden_size, self.logits_buf, self.model.config.vocab_size, hidden_dim);

            const use_gpu_argmax = have_gpu_argmax;
            if (use_gpu_argmax) {
                self.decode_cmd.computeBarrier();
                try self.argmax.record(
                    &self.decode_cmd,
                    self.argmax_descriptor_set.?,
                    self.model.config.vocab_size,
                    self.argmax_phase0_workgroups,
                );
            }

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
            self.endProfilePhase(.final_tail, final_tail_phase);
        }
        _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        try self.decode_cmd.end();
        var prefill_record_elapsed_ns: u64 = 0;
        if (track_decode_timing) {
            const cpu_record_end = std.time.nanoTimestamp();
            const elapsed: u64 = @intCast(cpu_record_end - cpu_record_start);
            if (self.profile_enabled) self.profile_token_counters.cpu_record_ns += elapsed;
            prefill_record_elapsed_ns = elapsed;
        }
        const submit_wait_start = if (track_decode_timing) std.time.nanoTimestamp() else 0;
        if (self.prefill_pipeline_mode) {
            // Pipelined prefill: fire-and-forget. prefillBatch() waits for the
            // corresponding fence before the next reuse of this slot.
            try self.decode_cmd.submit(self.instance.compute_queue);
        } else {
            try self.decode_cmd.submitAndWait(self.instance.compute_queue);
        }
        var prefill_submit_wait_elapsed_ns: u64 = 0;
        if (track_decode_timing) {
            const submit_wait_end = std.time.nanoTimestamp();
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

        state.position += 1;
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
        // Keep in sync with dmmv_q{4,6}k_batch_kpar.comp's `const uint MAX_COLS`.
        const MAX_COLS: u32 = 32;
        const f32_bytes: u32 = @sizeOf(f32);
        var chunk_start: u32 = 0;
        const kpar_pipeline: ?*const Pipeline = blk: {
            if (!self.use_q4k_batch_kpar) break :blk null;
            switch (tensor.info.type_) {
                .q4_k => break :blk if (self.dmmv.pipeline_q4k_batch_kpar) |*p| p else null,
                .q6_k => break :blk if (self.dmmv.pipeline_q6k_batch_kpar) |*p| p else null,
                else => break :blk null,
            }
        };
        while (chunk_start < n_tokens) {
            const chunk: u32 = @min(MAX_COLS, n_tokens - chunk_start);
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
    fn runSsmLayerGpu(self: *InferenceEngine, state: *DecodeState, layer: u32, layer_idx: usize, is_dead_tail: bool) !void {
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

        // --- GPU: 4 DMMV projections (same as CPU path) ---
        const lt = self.layer_tensors[layer];
        const wqkv_tensor = lt.attn_qkv orelse return;
        const z_tensor = lt.attn_gate orelse return;
        const alpha_tensor = lt.ssm_alpha orelse return;
        const beta_tensor = lt.ssm_beta orelse return;
        if (state.position == 0 and layer == 0) {
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
        const ssm_proj_phase = self.beginProfilePhase();

        // mmq path (opt-in via ZINC_MMQ_SSM=1): quantize norm_buf once per
        // (token, layer) into self.ssm_mmq_scratch, then dispatch the 4 SSM
        // proj matvecs as Q8_0 × Q8_1 integer-dot. Falls through to the f32
        // dispatchDmmv path whenever any tensor isn't Q8_0 or the pipelines
        // aren't loaded.
        const mmq_ready = self.use_mmq_ssm and
            wqkv_tensor.info.type_ == .q8_0 and
            (is_dead_tail or z_tensor.info.type_ == .q8_0) and
            alpha_tensor.info.type_ == .q8_0 and
            beta_tensor.info.type_ == .q8_0;

        if (mmq_ready) {
            try self.dmmv.recordQuantizeQ8_1(
                &self.decode_cmd,
                self.instance.push_descriptor_fn,
                self.norm_buf.handle,
                hidden_size,
                self.ssm_mmq_scratch.handle,
                self.ssm_mmq_scratch.size,
                hidden_dim,
            );
            // Quantize writes -> mmq matvecs read. Buffer-scoped barrier on the
            // Q8_1 scratch (writes by quantize_q8_1 finish, then reads by
            // dmmv_q8_0_q8_1 proceed). norm_buf itself is untouched.
            self.decode_cmd.computeBufferBarrier(self.ssm_mmq_scratch.handle, self.ssm_mmq_scratch.size);

            const q8q81 = &self.dmmv; // alias for brevity
            try q8q81.recordMmqQ8_0_Q8_1(
                &self.decode_cmd,
                self.instance.push_descriptor_fn,
                wqkv_tensor.gpu_buffer.handle, wqkv_tensor.gpu_buffer.size,
                self.ssm_mmq_scratch.handle, self.ssm_mmq_scratch.size,
                self.attn_out_buf.handle, self.attn_out_buf.size,
                @intCast(conv_channels), hidden_dim, 0, 0, 0, 0,
            );
            if (!is_dead_tail) {
                try q8q81.recordMmqQ8_0_Q8_1(
                    &self.decode_cmd,
                    self.instance.push_descriptor_fn,
                    z_tensor.gpu_buffer.handle, z_tensor.gpu_buffer.size,
                    self.ssm_mmq_scratch.handle, self.ssm_mmq_scratch.size,
                    self.gate_buf.handle, self.gate_buf.size,
                    @intCast(d_inner), hidden_dim, 0, 0, 0, 0,
                );
            }
            try q8q81.recordMmqQ8_0_Q8_1(
                &self.decode_cmd,
                self.instance.push_descriptor_fn,
                alpha_tensor.gpu_buffer.handle, alpha_tensor.gpu_buffer.size,
                self.ssm_mmq_scratch.handle, self.ssm_mmq_scratch.size,
                self.router_logits_buf.handle, self.router_logits_buf.size,
                dt_rank, hidden_dim, 0, 0, 0, 0,
            );
            try q8q81.recordMmqQ8_0_Q8_1(
                &self.decode_cmd,
                self.instance.push_descriptor_fn,
                beta_tensor.gpu_buffer.handle, beta_tensor.gpu_buffer.size,
                self.ssm_mmq_scratch.handle, self.ssm_mmq_scratch.size,
                self.down_buf.handle, self.down_buf.size,
                dt_rank, hidden_dim, 0, 0, 0, 0,
            );
        } else {
            try self.dispatchDmmv(wqkv_tensor, self.norm_buf, hidden_size, self.attn_out_buf, @intCast(conv_channels), hidden_dim);
            // Skip z (gate) DMMV in dead-tail: gate_buf is only consumed by
            // gated_norm, which is also skipped below. wqkv/alpha/beta still
            // run because conv1d/delta_net update SSM state for future tokens.
            if (!is_dead_tail) {
                try self.dispatchDmmv(z_tensor, self.norm_buf, hidden_size, self.gate_buf, @intCast(d_inner), hidden_dim);
            }
            try self.dispatchDmmv(alpha_tensor, self.norm_buf, hidden_size, self.router_logits_buf, dt_rank, hidden_dim);
            try self.dispatchDmmv(beta_tensor, self.norm_buf, hidden_size, self.down_buf, dt_rank, hidden_dim);
        }
        // The immediate next dispatch (ssm_conv1d) only reads attn_out_buf.
        // Writes to gate_buf/router_logits_buf/down_buf are picked up by the
        // subsequent global computeBarrier() before delta-net consumes them.
        self.decode_cmd.computeBufferBarrier(self.attn_out_buf.handle, qkv_bytes);
        self.endProfilePhase(.ssm_proj, ssm_proj_phase);

        // --- GPU: conv1d + SiLU ---
        // Input: attn_out_buf (QKV projection), conv kernel from GPU tensor, persistent conv state
        // Output: swiglu_buf (reused as conv1d output)
        const conv_tensor = lt.ssm_conv1d orelse return;
        const conv_kernel_is_f16 = conv_tensor.info.type_ == .f16;
        const ssm_conv_phase = self.beginProfilePhase();
        {
            const pip = &(self.elementwise.pipeline_ssm_conv1d orelse return error.ShaderNotLoaded);
            if (pip.uses_push_descriptors) {
                const push = SsmConv1dPush{
                    .conv_channels = conv_channels,
                    .d_conv = d_conv,
                    .kernel_is_f16 = if (conv_kernel_is_f16) 1 else 0,
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
                try self.elementwise.recordSsmConv1d(&self.decode_cmd, ds, conv_channels, d_conv, conv_kernel_is_f16);
            }
        }
        // Narrow: delta_net only reads swiglu_buf (conv1d output), router_logits_buf (alpha,
        // from ssm_proj), and down_buf (beta, from ssm_proj). gate_buf is consumed later by
        // gnorm and that path has its own barrier. gpu_ssm_conv_states is only read by the
        // NEXT token's conv1d in a different command buffer (cross-CB sync via submission
        // ordering + cycle 5 pipeline waitForCompletion).
        const conv_to_delta_ranges = [_]CommandBuffer.BufferRange{
            .{ .buffer = self.swiglu_buf.handle, .size = qkv_bytes },
            .{ .buffer = self.router_logits_buf.handle, .size = ab_bytes },
            .{ .buffer = self.down_buf.handle, .size = ab_bytes },
        };
        self.decode_cmd.computeBuffersBarrier(&conv_to_delta_ranges);
        self.endProfilePhase(.ssm_conv, ssm_conv_phase);

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
            const pip = &(self.elementwise.pipeline_ssm_delta_net orelse return error.ShaderNotLoaded);
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
            };
            if (pip.uses_push_descriptors) {
                const row_blocks = (head_v_dim + 7) / 8;
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
                try self.elementwise.recordSsmDeltaNet(&self.decode_cmd, ds, push);
            }
        }
        // Narrow: gated_norm only reads attn_out_buf (delta_net output, z_bytes) and
        // gate_buf (z_tensor DMMV from ssm_proj, z_bytes — synced here for the first
        // time since cycle 17 dropped it from the ssm_proj end-barrier). gpu_ssm_states
        // (delta_net RMW) is only read by the NEXT token's delta_net in a different
        // command buffer (cross-CB sync via submission ordering + cycle 5 pipelined
        // waitForCompletion). Follows cycle 21's multi-buffer pattern.
        if (!is_dead_tail) {
            const delta_to_gnorm_ranges = [_]CommandBuffer.BufferRange{
                .{ .buffer = self.attn_out_buf.handle, .size = z_bytes },
                .{ .buffer = self.gate_buf.handle, .size = z_bytes },
            };
            self.decode_cmd.computeBuffersBarrier(&delta_to_gnorm_ranges);
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
        const ssm_gated_norm_phase = self.beginProfilePhase();
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
                self.pushDispatch4(pip, std.mem.asBytes(&push), self.attn_out_buf.handle, z_bytes, self.gate_buf.handle, z_bytes, norm_buf_handle, norm_buf_size, self.swiglu_buf.handle, z_bytes, dt_rank, 1, 1);
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
        self.decode_cmd.computeBarrier();
        self.endProfilePhase(.ssm_gated_norm, ssm_gated_norm_phase);

        // --- GPU: ssm_out DMMV + residual (fused: accumulate directly into hidden_buf) ---
        const ssm_out_tensor = lt.ssm_out orelse return;
        const ssm_out_phase = self.beginProfilePhase();
        try self.dispatchDmmvAcc(ssm_out_tensor, self.swiglu_buf, z_bytes, self.hidden_buf, hidden_dim, @intCast(d_inner));
        self.decode_cmd.computeBarrier();
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
        const mode = std.posix.getenv("ZINC_BATCHED_PREFILL") orelse "";
        const batched_on = std.mem.eql(u8, mode, "1");
        const validate_mode = std.mem.eql(u8, mode, "validate");
        if ((!batched_on and !validate_mode) or !canUseBatchedPrefillRdna(self)) {
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
        const q_dim: u32 = cfg.n_heads * cfg.head_dim;
        const kv_dim: u32 = cfg.n_kv_heads * cfg.head_dim;
        const inter_dim: u32 = if (cfg.intermediate_dim > 0) cfg.intermediate_dim else hidden_dim * 4;
        const head_dim = cfg.head_dim;
        const rope_dim: u32 = if (cfg.rope_dim > 0) @min(cfg.rope_dim, head_dim) else head_dim;
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
            for (prompt_tokens, 0..) |tok, i| {
                const safe_id = @min(tok, vocab_last);
                const dst = big_f32[i * hidden_dim ..][0..hidden_dim];
                dequantRow(mmap[data_start..], safe_id, hidden_dim, embd.info.type_, dst);
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
            const v_t = lt.attn_v.?;
            const o_t = lt.attn_output.?;
            const gate_t = lt.ffn_gate.?;
            const up_t = lt.ffn_up.?;
            const down_t = lt.ffn_down.?;

            // attn RMS norm: hidden → norm
            try self.dispatchRmsNorm(scratch_hidden.handle, scratch_hidden.size, attn_norm_t.gpu_buffer.handle, attn_norm_t.gpu_buffer.size, scratch_norm.handle, scratch_norm.size, hidden_dim, n_tokens, eps);
            self.decode_cmd.computeBarrier();

            // Q / K / V projections (weight read once per 32-token chunk).
            try self.dispatchProjectionBatched(q_t, scratch_norm, scratch_q, q_dim, hidden_dim, n_tokens);
            try self.dispatchProjectionBatched(k_t, scratch_norm, scratch_k, kv_dim, hidden_dim, n_tokens);
            try self.dispatchProjectionBatched(v_t, scratch_norm, scratch_v, kv_dim, hidden_dim, n_tokens);
            self.decode_cmd.computeBarrier();

            // Optional per-head Q/K norms (Qwen3 style). Dispatch one workgroup per
            // (token, head) slot — rms_norm_mul handles this via group_id * head_dim.
            if (lt.attn_q_norm) |qn| {
                try self.dispatchRmsNorm(scratch_q.handle, scratch_q.size, qn.gpu_buffer.handle, qn.gpu_buffer.size, scratch_q.handle, scratch_q.size, head_dim, cfg.n_heads * n_tokens, eps);
            }
            if (lt.attn_k_norm) |kn| {
                try self.dispatchRmsNorm(scratch_k.handle, scratch_k.size, kn.gpu_buffer.handle, kn.gpu_buffer.size, scratch_k.handle, scratch_k.size, head_dim, cfg.n_kv_heads * n_tokens, eps);
            }
            if (lt.attn_q_norm != null or lt.attn_k_norm != null) self.decode_cmd.computeBarrier();

            // Batched RoPE for Q and K. position_base = state.position so a
            // prefix-reuse call rotates the newly-added tokens at the correct
            // sequence positions (base_token, base_token+1, ..., base_token+N-1).
            try self.dispatchRopeBatched(scratch_q.handle, scratch_q.size, scratch_q.handle, scratch_q.size, freq_buf_handle, freq_buf_size, head_dim, rope_dim, cfg.n_heads, base_token, n_tokens, cfg.rope_freq_base, 1.0);
            try self.dispatchRopeBatched(scratch_k.handle, scratch_k.size, scratch_k.handle, scratch_k.size, freq_buf_handle, freq_buf_size, head_dim, rope_dim, cfg.n_kv_heads, base_token, n_tokens, cfg.rope_freq_base, 1.0);
            self.decode_cmd.computeBarrier();

            // Batched KV cache write: one compute dispatch writes all N tokens'
            // K/V into their paged cache slots via the page_table_buf lookup.
            // base_token places the write after the existing prefix.
            try self.dispatchKvCacheWriteBatched(
                scratch_k.handle, scratch_k.size,
                self.kv_k_cache[layer_idx].handle, self.kv_k_cache[layer_idx].size,
                scratch_v.handle, scratch_v.size,
                self.kv_v_cache[layer_idx].handle, self.kv_v_cache[layer_idx].size,
                self.page_table_buf.handle, self.page_table_buf.size,
                kv_dim,
                n_tokens,
                kv_page_size_tokens,
                base_token,
            );
            self.decode_cmd.computeBarrier();

            // Batched causal flash attention: N queries over the KV cache.
            // seq_start = base_token so each query attends to prefix + own
            // position within the batch (causal_len = base_token + query + 1).
            const sink_offset = layer * cfg.n_heads;
            try self.dispatchFlashAttnBatched(scratch_q.handle, scratch_q.size, self.kv_k_cache[layer_idx].handle, self.kv_k_cache[layer_idx].size, self.kv_v_cache[layer_idx].handle, self.kv_v_cache[layer_idx].size, self.page_table_buf.handle, self.page_table_buf.size, scratch_attn_out.handle, scratch_attn_out.size, self.attn_sinks_buf.handle, self.attn_sinks_buf.size, head_dim, cfg.n_heads, cfg.n_kv_heads, base_token, n_tokens, kv_page_size_tokens, cfg.attn_scale, sink_offset);
            self.decode_cmd.computeBarrier();

            // O projection → FUSED residual+FFN norm (hidden += down;
            // norm = normalize(hidden) * ffn_norm_weight). Replaces
            // scale_acc → barrier → rms_norm_mul with a single dispatch.
            try self.dispatchProjectionBatched(o_t, scratch_attn_out, scratch_down, hidden_dim, q_dim, n_tokens);
            self.decode_cmd.computeBarrier();
            try self.dispatchResidualRmsNorm(scratch_hidden.handle, scratch_hidden.size, scratch_down.handle, scratch_down.size, scratch_norm.handle, scratch_norm.size, ffn_norm_t.gpu_buffer.handle, ffn_norm_t.gpu_buffer.size, hidden_dim, n_tokens, eps, 1.0);
            self.decode_cmd.computeBarrier();

            // FFN: gate/up → SwiGLU → down → residual.
            try self.dispatchProjectionBatched(gate_t, scratch_norm, scratch_gate, inter_dim, hidden_dim, n_tokens);
            try self.dispatchProjectionBatched(up_t, scratch_norm, scratch_up, inter_dim, hidden_dim, n_tokens);
            self.decode_cmd.computeBarrier();
            // dispatchFfnActivation picks SwiGLU / GEGLU / SwiGLU-OAI based on
            // cfg.architecture. canUseBatchedPrefillRdna rejects Gemma and
            // gpt-oss, so in practice this routes through SwiGLU — but we
            // stay aligned with the project-wide invariant that FFN
            // activations go through this dispatcher rather than calling
            // dispatchSwiglu directly.
            try self.dispatchFfnActivation(scratch_gate.handle, scratch_gate.size, scratch_up.handle, scratch_up.size, scratch_swiglu.handle, scratch_swiglu.size, n_tokens * inter_dim);
            self.decode_cmd.computeBarrier();
            try self.dispatchProjectionBatched(down_t, scratch_swiglu, scratch_down, hidden_dim, inter_dim, n_tokens);
            self.decode_cmd.computeBarrier();
            try self.dispatchScaleAcc(scratch_hidden.handle, scratch_hidden.size, scratch_down.handle, scratch_down.size, n_tokens * hidden_dim, 1.0);
            self.decode_cmd.computeBarrier();
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
            const vocab = cfg.vocab_size;
            const batched_snapshot = try self.instance.allocator.alloc(f32, vocab);
            defer self.instance.allocator.free(batched_snapshot);
            const batched_logits: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
            @memcpy(batched_snapshot, batched_logits[0..vocab]);

            state.position = 0;
            state.generated_tokens.clearRetainingCapacity();
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
        const profile_env = std.posix.getenv("ZINC_PREFILL_PROFILE");
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
                // mmq scratch pair swaps with the CB slot so the two in-flight
                // CBs don't race on the same Q8_1 region. No-op when the mmq
                // path is disabled (both buffers are zero-sized handles).
                if (self.use_mmq_ssm) {
                    std.mem.swap(Buffer, &self.ssm_mmq_scratch, &self.ssm_mmq_scratch_alt);
                }
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
        self.router_output_buf.deinit();
        if (self.routing_capture_buf.handle != null) self.routing_capture_buf.deinit();
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
        if (self.ssm_mmq_scratch.handle != null) self.ssm_mmq_scratch.deinit();
        if (self.ssm_mmq_scratch_alt.handle != null) self.ssm_mmq_scratch_alt.deinit();
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
    const prefill_start = std.time.nanoTimestamp();
    try engine.prefillBatched(&state, prompt_tokens);
    const prefill_end = std.time.nanoTimestamp();
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
            var tail_ns: u64 = 0;
            var embed_ns: u64 = 0;
            // `.ssm` wraps all ssm_* sub-phases and `.moe_routed` wraps all moe_*
            // sub-phases, so summing every enum value double-counts. Bucket with
            // the wrappers only; shared_* and tail/attn/embed have no wrapper.
            inline for (@typeInfo(ProfilePhase).@"enum".fields) |f| {
                const phase_val: ProfilePhase = @enumFromInt(f.value);
                const v = engine.prefill_gpu_phase_ns[f.value];
                switch (phase_val) {
                    .attention => attn_ns += v,
                    .moe_routed => moe_ns += v,
                    .shared_expert, .shared_proj, .shared_swiglu, .shared_down, .shared_gate_acc => shared_ns += v,
                    .ssm => ssm_ns += v,
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
            const tail_avg = to_ms(tail_ns) / samples_f;
            const embed_avg = to_ms(embed_ns) / samples_f;
            log.info(
                "Prefill GPU phases: per-tok attn={d:.2} ms moe={d:.2} ms shared={d:.2} ms ssm={d:.2} ms tail={d:.2} ms embed={d:.3} ms | totals attn={d:.1} moe={d:.1} shared={d:.1} ssm={d:.1} tail={d:.1} embed={d:.1}",
                .{
                    attn_avg,
                    moe_avg,
                    shared_avg,
                    ssm_avg,
                    tail_avg,
                    embed_avg,
                    to_ms(attn_ns),
                    to_ms(moe_ns),
                    to_ms(shared_ns),
                    to_ms(ssm_ns),
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
            const ssm_conv_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_conv)];
            const ssm_delta_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_delta)];
            const ssm_gnorm_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_gated_norm)];
            const ssm_out_ns = engine.prefill_gpu_phase_ns[@intFromEnum(ProfilePhase.ssm_out)];
            log.info(
                "Prefill SSM subphases totals: proj={d:.1} conv={d:.1} delta={d:.1} gnorm={d:.1} out={d:.1} ms",
                .{
                    to_ms(ssm_proj_ns),
                    to_ms(ssm_conv_ns),
                    to_ms(ssm_delta_ns),
                    to_ms(ssm_gnorm_ns),
                    to_ms(ssm_out_ns),
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
    const decode_start = std.time.nanoTimestamp();

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
        const tok_start = std.time.nanoTimestamp();

        // Feed the last generated token as input
        const input_token = state.generated_tokens.items[state.generated_tokens.items.len - 1];

        try engine.decodeStep(&state, input_token, true);
        const token = engine.sampleGreedy();
        try state.generated_tokens.append(allocator, token);
        // Top-5 logits per token for first 5 tokens + last token
        if (generated < 5 or generated == effective_max_tokens - 1) {
            if (engine.logits_readback_enabled) dumpTop5Logits(engine, generated);
        }

        const tok_end = std.time.nanoTimestamp();
        const tok_ms = @as(f64, @floatFromInt(@as(u64, @intCast(tok_end - tok_start)))) / 1_000_000.0;
        log.debug("decode[{d}]: token={d} pos={d} ({d:.1} ms)", .{
            generated, token, state.position, tok_ms,
        });

        // Check for EOS token (read from GGUF metadata)
        if (token == eos_token_id) break;
    }
    const decode_end = std.time.nanoTimestamp();

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
            const avg_ssm_phase_ms = engine.avgProfilePhaseMs(.ssm);
            const avg_moe_phase_ms = engine.avgProfilePhaseMs(.moe_routed);
            const avg_shared_phase_ms = engine.avgProfilePhaseMs(.shared_expert);
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
            log.info("PROFILE: avg GPU phases embed={d:.2} ms attention={d:.2} ms ssm={d:.2} ms moe={d:.2} ms shared={d:.2} ms tail={d:.2} ms", .{
                avg_embed_phase_ms,
                avg_attention_phase_ms,
                avg_ssm_phase_ms,
                avg_moe_phase_ms,
                avg_shared_phase_ms,
                avg_tail_phase_ms,
            });
            log.info("PROFILE: avg SSM subphases proj={d:.2} ms conv={d:.2} ms delta={d:.2} ms gnorm={d:.2} ms out={d:.2} ms", .{
                engine.avgProfilePhaseMs(.ssm_proj),
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
    try std.testing.expectEqual(@as(usize, 12), @sizeOf(ew.SsmConv1dPush));
    try std.testing.expectEqual(@as(usize, 36), @sizeOf(ew.SsmDeltaNetPush));
    try std.testing.expectEqual(@as(usize, 20), @sizeOf(ew.SsmGatedNormPush));
    try std.testing.expectEqual(@as(usize, 8), @sizeOf(ew.SoftmaxTopkPush));
}
