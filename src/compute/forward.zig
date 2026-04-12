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
const BatchDmmvPushConstants = dmmv_mod.BatchDmmvPushConstants;
const MoeDmmvPushConstants = dmmv_mod.MoeDmmvPushConstants;
const elementwise_mod = @import("elementwise.zig");
const ElementwiseDispatch = elementwise_mod.ElementwiseDispatch;
const RmsNormPush = elementwise_mod.RmsNormPush;
const SwigluPush = elementwise_mod.SwigluPush;
const SigmoidMulPush = elementwise_mod.SigmoidMulPush;
const ScaleAccPush = elementwise_mod.ScaleAccPush;
const RopePush = elementwise_mod.RopePush;
const SoftmaxTopkPush = elementwise_mod.SoftmaxTopkPush;
const MoeWeightedAccPush = elementwise_mod.MoeWeightedAccPush;
const SsmConv1dPush = elementwise_mod.SsmConv1dPush;
const SsmDeltaNetPush = elementwise_mod.SsmDeltaNetPush;
const SsmGatedNormPush = elementwise_mod.SsmGatedNormPush;
const DeinterleavePush = elementwise_mod.DeinterleavePush;
const KvCacheWritePush = elementwise_mod.KvCacheWritePush;
const NormRopePush = elementwise_mod.NormRopePush;
const attn_mod = @import("attention.zig");
const AttentionDispatch = attn_mod.AttentionDispatch;
const FlashAttnPush = attn_mod.FlashAttnPush;
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

pub const SamplingParams = struct {
    temperature: f32 = 0.0,
    top_p: f32 = 1.0,
    repetition_penalty: f32 = 1.0,
    top_k: u32 = 64,

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
    attn_output: ?*const LoadedTensor = null,
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
    // MoE
    ffn_gate_inp: ?*const LoadedTensor = null,
    ffn_gate_exps: ?*const LoadedTensor = null,
    ffn_up_exps: ?*const LoadedTensor = null,
    ffn_gate_up_exps: ?*const LoadedTensor = null,
    ffn_down_exps: ?*const LoadedTensor = null,
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
    rope_freq_buf: Buffer, // IMROPE precomputed inverse frequencies (rope_dim/2 f32)
    unit_norm_weights: Buffer, // all-1.0 weights for plain RMS normalization (Gemma 4 V norm)
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
        // For IMROPE (Gemma 3 vision): sectioned per-pair frequencies.
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
                    config.rope_sections[0], config.rope_sections[1], config.rope_sections[2], config.rope_sections[3],
                    total_pairs, freq_ptr[0], if (total_pairs > 11) freq_ptr[11] else 0.0, if (total_pairs > 31) freq_ptr[31] else 0.0,
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
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
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
            lt.attn_output = resolve(tensor_map, l, "attn_output.weight");
            lt.attn_gate = resolve(tensor_map, l, "attn_gate.weight");
            lt.attn_q_norm = resolve(tensor_map, l, "attn_q_norm.weight");
            lt.attn_k_norm = resolve(tensor_map, l, "attn_k_norm.weight");
            lt.post_attention_norm = resolve(tensor_map, l, "post_attention_norm.weight");
            lt.ffn_norm = resolve(tensor_map, l, "ffn_norm.weight");
            lt.ffn_gate = resolve(tensor_map, l, "ffn_gate.weight");
            lt.ffn_up = resolve(tensor_map, l, "ffn_up.weight");
            lt.ffn_down = resolve(tensor_map, l, "ffn_down.weight");
            lt.post_ffw_norm = resolve(tensor_map, l, "post_ffw_norm.weight");
            lt.ffn_gate_inp = resolve(tensor_map, l, "ffn_gate_inp.weight");
            lt.ffn_gate_exps = resolve(tensor_map, l, "ffn_gate_exps.weight");
            lt.ffn_up_exps = resolve(tensor_map, l, "ffn_up_exps.weight");
            lt.ffn_gate_up_exps = resolve(tensor_map, l, "ffn_gate_up_exps.weight");
            lt.ffn_down_exps = resolve(tensor_map, l, "ffn_down_exps.weight");
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

        return InferenceEngine{
            .model = model,
            .gpu_config = gpu_config,
            .dmmv = dmmv,
            .elementwise = elementwise,
            .attention = attention,
            .argmax = argmax,
            .cmd_pool = cmd_pool,
            .decode_cmd = decode_cmd,
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
        };
    }

    // -----------------------------------------------------------------------
    // Profiling
    // -----------------------------------------------------------------------

    /// Enable GPU timestamp profiling. Creates a Vulkan query pool for timestamp queries.
    pub fn enableProfiling(self: *InferenceEngine) !void {
        const max_timestamps: u32 = 2048; // enough for ~1000 dispatches per token
        const pool_info = vk.c.VkQueryPoolCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queryType = vk.c.VK_QUERY_TYPE_TIMESTAMP,
            .queryCount = max_timestamps,
            .pipelineStatistics = 0,
        };
        var pool: vk.c.VkQueryPool = null;
        const result = vk.c.vkCreateQueryPool(self.instance.device, &pool_info, null, &pool);
        if (result != vk.c.VK_SUCCESS) return error.QueryPoolCreateFailed;
        self.timestamp_query_pool = pool;
        self.timestamp_period_ns = @as(f64, self.instance.device_props.limits.timestampPeriod);
        self.profile_enabled = true;
        log.debug("Profiling enabled: {d} timestamp queries, period={d:.2}ns", .{ max_timestamps, self.timestamp_period_ns });
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
    ) !void {
        const pip = &(self.elementwise.pipeline_rope orelse return error.ShaderNotLoaded);
        if (pip.uses_push_descriptors) {
            const push = RopePush{
                .stride = stride,
                .rope_dim = rope_dim,
                .n_heads = n_heads,
                .position = position,
                .freq_base_bits = @bitCast(freq_base),
            };
            if (freq_buf) |fb| {
                self.pushDispatch3(
                    pip,
                    std.mem.asBytes(&push),
                    buf,
                    buf_size,
                    buf,
                    buf_size,
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
                    buf,
                    buf_size,
                    n_heads,
                    1,
                    1,
                );
            }
            return;
        }

        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        if (freq_buf) |fb| {
            self.writeDescSet3(ds, buf, buf_size, buf, buf_size, fb, freq_size);
        } else {
            self.writeDescSet2(ds, buf, buf_size, buf, buf_size);
        }
        try self.elementwise.recordRope(&self.decode_cmd, ds, stride, rope_dim, n_heads, position, freq_base);
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
        eps: f32,
    ) void {
        const pip = &(self.elementwise.pipeline_norm_rope orelse return);
        const push = NormRopePush{
            .head_dim = head_dim,
            .rope_dim = rope_dim,
            .n_heads = n_heads,
            .position = position,
            .freq_base_bits = @bitCast(freq_base),
            .eps_bits = @bitCast(eps),
        };
        if (freq_buf) |fb| {
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                buf, buf_size,
                weight_buf, weight_size,
                fb, freq_size,
                n_heads, 1, 1,
            );
        } else {
            // Bind a dummy buffer for the unused freq binding
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                buf, buf_size,
                weight_buf, weight_size,
                buf, buf_size,
                n_heads, 1, 1,
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
        const pip = &(self.elementwise.pipeline_softmax_topk orelse return error.ShaderNotLoaded);
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
        const safe_id = @min(token_id, self.model.config.vocab_size -| 1);

        const embd = self.tensor_map.get("token_embd.weight") orelse {
            log.err("token_embd.weight not found", .{});
            return error.TensorNotFound;
        };

        const mmap = self.model.mmap_data orelse return error.NoMmapData;
        const data_start: usize = @intCast(self.model.gguf_file.tensor_data_offset + embd.info.offset);

        // Dequantize directly into pre-allocated staging buffer (zero alloc)
        const staging_f32: [*]f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
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
        const cpu_embed_start = if (self.profile_enabled) std.time.nanoTimestamp() else 0;
        try self.embedToken(token_id);
        if (self.profile_enabled) {
            const cpu_embed_end = std.time.nanoTimestamp();
            self.profile_token_counters.cpu_embed_ns += @intCast(cpu_embed_end - cpu_embed_start);
        }

        // Per-layer logit5 tracking for BOS diagnostic summary
        var diag_logit5 = [_]f32{0} ** 64;
        var diag_rms_arr = [_]f32{0} ** 64;

        const cpu_record_start = if (self.profile_enabled) std.time.nanoTimestamp() else 0;

        // Begin single command buffer for all layers (Phase 3c batching)
        if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.beginOneTime();

        // Reset profiling timestamps for this token
        self.resetTimestamps();
        _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT);

        for (0..config.n_layers) |layer_idx| {
            const layer: u32 = @intCast(layer_idx);
            const lt = self.layer_tensors[layer_idx];

            // --- Upload embedding (only first layer) ---
            if (layer == 0) {
                const embed_phase = self.beginProfilePhase();
                const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = hidden_size };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.embed_staging.handle, self.hidden_buf.handle, 1, &region);
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

            if (is_full_attn) {
                const attention_phase = self.beginProfilePhase();
                // === FULL ATTENTION LAYER ===
                // Q/gate projection → Q/K norm → K/V proj → RoPE → KV cache → flash attention
                // → sigmoid gate → output projection → residual

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
                    try self.dispatchDmmv(q_tensor, self.norm_buf, hidden_size, self.attn_out_buf, q_rows, hidden_dim);
                } else {
                    // Dense qwen35 may store Q and gate as separate tensors.
                    // Use q_rows (tensor shape) not q_dim (config) — Gemma 4 mixed head_dim.
                    try self.dispatchDmmv(q_tensor, self.norm_buf, hidden_size, self.q_buf, q_rows, hidden_dim);
                    if (attn_gate_tensor) |gate_tensor| {
                        try self.dispatchDmmv(gate_tensor, self.norm_buf, hidden_size, self.gate_buf, q_rows, hidden_dim);
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
                                self.attn_out_buf.handle, q_full_size,
                                self.q_buf.handle, q_size,
                                self.gate_buf.handle, q_size,
                                (total + 63) / 64, 1, 1,
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
                // Use precomputed frequency buffer for global layers (includes rope_freqs.weight)
                // or IMROPE; SWA layers compute inline with swa freq base.
                const use_precomputed_freq = use_imrope or (proportional_rope and !use_swa_rope);
                const rope_freq: f32 = if (use_precomputed_freq) 0.0 else if (use_swa_rope) config.rope_freq_base_swa else config.rope_freq_base;
                const freq_buf_handle = if (use_precomputed_freq) self.rope_freq_buf.handle else null;

                // Fused norm+rope: when both norm and rope are needed, combine them into
                // a single dispatch per head set, eliminating 1 barrier + 2 dispatches.
                const use_fused_norm_rope = self.elementwise.pipeline_norm_rope != null;
                var q_rope_done = false;
                var k_rope_done = false;

                if (q_norm_tensor) |qn| {
                    if (use_fused_norm_rope) {
                        // Fused Q norm + Q RoPE in a single dispatch
                        self.dispatchNormRopeInPlace(
                            self.q_buf.handle, self.q_buf.size,
                            qn.gpu_buffer.handle, qn.gpu_buffer.size,
                            freq_buf_handle, self.rope_freq_buf.size,
                            layer_head_dim, layer_rope_dim, config.n_heads,
                            state.position, rope_freq, rms_norm_eps,
                        );
                        q_rope_done = true;
                    } else {
                        try self.dispatchRmsNorm(
                            self.q_buf.handle, self.q_buf.size,
                            qn.gpu_buffer.handle, qn.gpu_buffer.size,
                            self.q_buf.handle, self.q_buf.size,
                            layer_head_dim, config.n_heads, rms_norm_eps,
                        );
                    }
                }
                if (k_norm_tensor) |kn| {
                    if (use_fused_norm_rope) {
                        // Fused K norm + K RoPE in a single dispatch
                        self.dispatchNormRopeInPlace(
                            self.k_buf.handle, self.k_buf.size,
                            kn.gpu_buffer.handle, kn.gpu_buffer.size,
                            freq_buf_handle, self.rope_freq_buf.size,
                            layer_head_dim, layer_rope_dim, layer_n_kv_heads,
                            state.position, rope_freq, rms_norm_eps,
                        );
                        k_rope_done = true;
                    } else {
                        try self.dispatchRmsNorm(
                            self.k_buf.handle, self.k_buf.size,
                            kn.gpu_buffer.handle, kn.gpu_buffer.size,
                            self.k_buf.handle, self.k_buf.size,
                            layer_head_dim, layer_n_kv_heads, rms_norm_eps,
                        );
                    }
                }
                // Gemma 4 applies plain RMS norm (unit weights) to V per-head.
                // Mirrors Metal forward_metal.zig:3460-3462.
                if (config.architecture == .gemma and config.rope_freq_base_swa > 0) {
                    try self.dispatchRmsNorm(
                        self.v_buf.handle, self.v_buf.size,
                        self.unit_norm_weights.handle, self.unit_norm_weights.size,
                        self.v_buf.handle, self.v_buf.size,
                        layer_head_dim, layer_n_kv_heads, rms_norm_eps,
                    );
                }
                self.decode_cmd.computeBarrier();

                if (!k_rope_done) {
                    // K RoPE first — KV cache write reads k_buf, so it must complete before the write.
                    try self.dispatchRopeInPlace(
                        self.k_buf.handle, self.k_buf.size,
                        freq_buf_handle, self.rope_freq_buf.size,
                        layer_head_dim, layer_rope_dim, layer_n_kv_heads,
                        state.position, rope_freq,
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
                                self.k_buf.handle, self.k_buf.size,
                                self.kv_k_cache[layer_idx].handle, self.kv_k_cache[layer_idx].size,
                                self.v_buf.handle, self.v_buf.size,
                                self.kv_v_cache[layer_idx].handle, self.kv_v_cache[layer_idx].size,
                                (layer_kv_dim + 63) / 64, 1, 1,
                            );
                        } else {
                            const ds = try self.allocDescSet(kv_pip.descriptor_set_layout);
                            self.writeDescSet4(ds, self.k_buf.handle, self.k_buf.size, self.kv_k_cache[layer_idx].handle, self.kv_k_cache[layer_idx].size, self.v_buf.handle, self.v_buf.size, self.kv_v_cache[layer_idx].handle, self.kv_v_cache[layer_idx].size);
                            self.decode_cmd.dispatchWithPush(kv_pip, ds, std.mem.asBytes(&push), (layer_kv_dim + 63) / 64, 1, 1);
                        }
                        if (!q_rope_done) {
                            // Q RoPE overlaps with KV write — no data dependency between them.
                            try self.dispatchRopeInPlace(
                                self.q_buf.handle, self.q_buf.size,
                                freq_buf_handle, self.rope_freq_buf.size,
                                layer_head_dim, layer_rope_dim, config.n_heads,
                                state.position, rope_freq,
                            );
                        }
                        self.decode_cmd.computeBarrier();
                    } else {
                        // Transfer fallback: Q RoPE before barrier (original order preserved)
                        if (!q_rope_done) {
                            try self.dispatchRopeInPlace(
                                self.q_buf.handle, self.q_buf.size,
                                freq_buf_handle, self.rope_freq_buf.size,
                                layer_head_dim, layer_rope_dim, config.n_heads,
                                state.position, rope_freq,
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

                // Flash attention
                if (self.attention.pipeline) |*pip| {
                    if (pip.uses_push_descriptors) {
                        const push = FlashAttnPush{
                            .head_dim = layer_head_dim,
                            .n_heads = config.n_heads,
                            .n_kv_heads = layer_n_kv_heads,
                            .seq_len = state.position + 1,
                            .page_size = kv_page_size_tokens,
                            .attn_scale_bits = if (config.attn_scale != 0) @as(u32, @bitCast(config.attn_scale)) else 0,
                        };
                        self.pushDispatch5(pip, std.mem.asBytes(&push), self.q_buf.handle, self.q_buf.size, self.kv_k_cache[layer_idx].handle, self.kv_k_cache[layer_idx].size, self.kv_v_cache[layer_idx].handle, self.kv_v_cache[layer_idx].size, self.page_table_buf.handle, self.page_table_buf.size, self.attn_out_buf.handle, self.attn_out_buf.size, config.n_heads, 1, 1);
                    } else {
                        const attn_ds = try self.allocDescSet(pip.descriptor_set_layout);
                        self.writeDescSet5(attn_ds, self.q_buf.handle, self.q_buf.size, self.kv_k_cache[layer_idx].handle, self.kv_k_cache[layer_idx].size, self.kv_v_cache[layer_idx].handle, self.kv_v_cache[layer_idx].size, self.page_table_buf.handle, self.page_table_buf.size, self.attn_out_buf.handle, self.attn_out_buf.size);
                        try self.attention.recordFlashAttn(&self.decode_cmd, attn_ds, layer_head_dim, config.n_heads, layer_n_kv_heads, state.position + 1, kv_page_size_tokens, config.attn_scale);
                    }
                }
                self.decode_cmd.computeBarrier();

                // Self-check the first attention layer at seq_len=1: with only one KV token,
                // flash attention must reproduce the current V slice for each query head's KV group.
                if (state.position == 0 and is_full_attn and self.validation_diagnostics_enabled) {
                    try self.decode_cmd.end();
                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                    try self.decode_cmd.reset();
                    try self.decode_cmd.begin();
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.attn_out_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                        .srcOffset = 0,
                        .dstOffset = 0,
                        .size = self.attn_out_buf.size,
                    });
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.v_buf.handle, self.embed_staging.handle, 1, &vk.c.VkBufferCopy{
                        .srcOffset = 0,
                        .dstOffset = 0,
                        .size = self.v_buf.size,
                    });
                    try self.decode_cmd.end();
                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                    const attn_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
                    const attn_vals = attn_ptr[0..q_dim];
                    const v_ptr: [*]const f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
                    const v_vals = v_ptr[0..kv_dim];

                    var attn_v_max_diff: f32 = 0;
                    for (0..config.n_heads) |h| {
                        const kv_head = h / (config.n_heads / config.n_kv_heads);
                        for (0..config.head_dim) |d| {
                            const got = attn_vals[h * config.head_dim + d];
                            const want = v_vals[kv_head * config.head_dim + d];
                            const diff = @abs(got - want);
                            if (diff > attn_v_max_diff) attn_v_max_diff = diff;
                        }
                    }
                    log.info("ATTN_SELFTEST L{d}: seq_len=1 attn_vs_v max_diff={d:.6} attn_h0[0..4]=[{d:.6},{d:.6},{d:.6},{d:.6}] v_kv0[0..4]=[{d:.6},{d:.6},{d:.6},{d:.6}]", .{
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
                    });

                    if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                    try self.decode_cmd.reset();
                    try self.decode_cmd.begin();
                }

                // Validate multi-token flash attention against a naive CPU reference on the
                // first full-attention layer once the 5-token prompt is fully prefilling.
                if (state.position == 4 and layer == full_attn_interval - 1 and self.validation_diagnostics_enabled) {
                    const seq_len_dbg: u32 = state.position + 1;
                    const q_bytes = @as(vk.c.VkDeviceSize, q_dim) * @sizeOf(f32);
                    const kv_dbg_bytes = @as(vk.c.VkDeviceSize, seq_len_dbg * kv_dim) * @sizeOf(f32);
                    const attn_bytes = @as(vk.c.VkDeviceSize, q_dim) * @sizeOf(f32);
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
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.kv_k_cache[layer_idx].handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                        .srcOffset = 0,
                        .dstOffset = k_off,
                        .size = kv_dbg_bytes,
                    });
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.kv_v_cache[layer_idx].handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                        .srcOffset = 0,
                        .dstOffset = v_off,
                        .size = kv_dbg_bytes,
                    });
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.attn_out_buf.handle, self.logits_staging.handle, 1, &vk.c.VkBufferCopy{
                        .srcOffset = 0,
                        .dstOffset = attn_off,
                        .size = attn_bytes,
                    });
                    try self.decode_cmd.end();
                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                    const dbg_ptr: [*]const f32 = @ptrCast(@alignCast(self.logits_staging.mapped.?));
                    const q_vals = dbg_ptr[0..q_dim];
                    const k_vals = dbg_ptr[@intCast(k_off / @sizeOf(f32))..][0 .. seq_len_dbg * kv_dim];
                    const v_vals = dbg_ptr[@intCast(v_off / @sizeOf(f32))..][0 .. seq_len_dbg * kv_dim];
                    const attn_vals = dbg_ptr[@intCast(attn_off / @sizeOf(f32))..][0..q_dim];

                    const seq_len_usize: usize = @intCast(seq_len_dbg);
                    const q_dim_usize: usize = @intCast(q_dim);
                    var cpu_attn = try self.allocator.alloc(f32, q_dim_usize);
                    defer self.allocator.free(cpu_attn);
                    var scores = try self.allocator.alloc(f32, seq_len_usize);
                    defer self.allocator.free(scores);
                    var probs = try self.allocator.alloc(f32, seq_len_usize);
                    defer self.allocator.free(probs);

                    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(config.head_dim)));
                    for (0..config.n_heads) |h| {
                        const kv_head = h / (config.n_heads / config.n_kv_heads);
                        const q_head = q_vals[h * config.head_dim ..][0..config.head_dim];

                        var max_score: f32 = -std.math.inf(f32);
                        for (0..seq_len_dbg) |tok| {
                            const k_tok = k_vals[tok * kv_dim + kv_head * config.head_dim ..][0..config.head_dim];
                            var dot: f32 = 0;
                            for (0..config.head_dim) |d| dot += q_head[d] * k_tok[d];
                            const s = dot * scale;
                            scores[tok] = s;
                            if (s > max_score) max_score = s;
                        }

                        var sum_exp: f32 = 0;
                        for (0..seq_len_dbg) |tok| {
                            const p = @exp(scores[tok] - max_score);
                            probs[tok] = p;
                            sum_exp += p;
                        }
                        const inv_sum = if (sum_exp > 0) 1.0 / sum_exp else 0.0;

                        const out_head = cpu_attn[h * config.head_dim ..][0..config.head_dim];
                        @memset(out_head, 0);
                        for (0..seq_len_dbg) |tok| {
                            const weight = probs[tok] * inv_sum;
                            const v_tok = v_vals[tok * kv_dim + kv_head * config.head_dim ..][0..config.head_dim];
                            for (0..config.head_dim) |d| out_head[d] += weight * v_tok[d];
                        }
                    }

                    var attn_ref_max_diff: f32 = 0;
                    for (0..q_dim) |i| {
                        const diff = @abs(attn_vals[i] - cpu_attn[i]);
                        if (diff > attn_ref_max_diff) attn_ref_max_diff = diff;
                    }
                    log.info("ATTN_REFTEST L{d}: seq_len={d} max_diff={d:.6} attn_h0[0..4]=[{d:.6},{d:.6},{d:.6},{d:.6}] cpu_h0[0..4]=[{d:.6},{d:.6},{d:.6},{d:.6}]", .{
                        layer,
                        seq_len_dbg,
                        attn_ref_max_diff,
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
                const has_post_attn_norm = lt.post_attention_norm != null and lt.ffn_norm != null;
                if (!has_post_attn_norm and !self.validation_diagnostics_enabled) {
                    // Fused: O-proj DMMV accumulates directly into hidden_buf,
                    // eliminating separate scale_acc dispatch + barrier
                    // Use o_cols (from O weight tensor shape) — matches actual attention output dim.
                    // Gemma 4 has mixed head_dim (256 SWA vs 512 global); o_cols is always correct
                    // while q_dim (from config) uses the max head_dim.
                    try self.dispatchDmmvAcc(o_tensor, self.attn_out_buf, self.attn_out_buf.size, self.hidden_buf, hidden_dim, o_cols);
                    self.decode_cmd.computeBarrier();
                } else {
                    // Unfused path: needed when post-attn norm exists (Gemma) or diagnostics enabled
                    try self.dispatchDmmv(o_tensor, self.attn_out_buf, self.attn_out_buf.size, self.o_proj_buf, hidden_dim, o_cols);
                    self.decode_cmd.computeBarrier();

                    // Gemma post-attention norm: RMS norm on o_proj output before residual add
                    if (lt.post_attention_norm) |pan_tensor| {
                        if (lt.ffn_norm != null) {
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
                    }

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

                // --- Mid-layer diagnostic: o_proj RMS at attention layers (BOS only) ---
                // Single readback per attention layer — reads o_proj_buf (before residual add)
                if (state.position == 0 and is_full_attn and self.validation_diagnostics_enabled) {
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
                        var cpu_vals: [4]f32 = [_]f32{0} ** 4;
                        const o_rows: u32 = @min(hidden_dim, cpu_vals.len);
                        var o_proj_max_diff: f32 = 0;
                        for (0..o_rows) |row| {
                            dequantRow(mmap[o_off..], @intCast(row), q_dim, o_tensor.info.type_, cpu_row_buf[0..q_dim]);
                            var dot: f64 = 0;
                            for (0..q_dim) |i| dot += @as(f64, cpu_row_buf[i]) * @as(f64, attn_vals[i]);
                            cpu_vals[row] = @floatCast(dot);
                            const diff = @abs(op[row] - cpu_vals[row]);
                            if (diff > o_proj_max_diff) o_proj_max_diff = diff;
                        }
                        log.info("DMMV_CHECK: attn_output type={s} M={d} K={d} gpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}] cpu[0..3]=[{d:.6},{d:.6},{d:.6},{d:.6}] max_diff={d:.6} ok={s}", .{
                            @tagName(o_tensor.info.type_),
                            hidden_dim,
                            q_dim,
                            op[0],
                            op[1],
                            op[2],
                            op[3],
                            cpu_vals[0],
                            cpu_vals[1],
                            cpu_vals[2],
                            cpu_vals[3],
                            o_proj_max_diff,
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
                const ssm_phase = self.beginProfilePhase();
                if (use_gpu_ssm) {
                    try self.runSsmLayerGpu(state, layer, layer_idx);
                } else {
                    if (self.profile_enabled) self.profile_token_counters.cpu_ssm_fallbacks += 1;
                    try self.runSsmLayerCpu(state, layer, layer_idx);
                }
                self.endProfilePhase(.ssm, ssm_phase);
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

            var gpu_moe_barriers_cover_hidden = false;
            if (is_moe) {
                const moe_phase = self.beginProfilePhase();
                // --- MoE: router DMMV → top-k → expert dispatch ---
                const router_tensor = lt.ffn_gate_inp orelse return error.TensorNotFound;
                const moe_router_phase = self.beginProfilePhase();
                try self.dispatchDmmv(router_tensor, self.ffn_norm_buf, hidden_size, self.router_logits_buf, config.n_experts, hidden_dim);
                self.decode_cmd.computeBufferBarrier(self.router_logits_buf.handle, self.router_logits_buf.size);
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

                // Check if full GPU MoE path is available (MoE DMMV + softmax_topk + weighted_acc)
                const use_gpu_moe = self.dmmv.moePipelineForType(gate_quant) != null and
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
                    self.decode_cmd.computeBufferBarrier(self.router_output_buf.handle, self.router_output_buf.size);
                    self.endProfilePhase(.moe_topk, moe_topk_phase);

                    // gate DMMV: ALL experts at once (Y=n_used workgroups)
                    // gate_exps[expert] × ffn_norm_buf → gate_buf[expert*inter_dim..]
                    // x_expert_stride=0: all experts read same input (ffn_norm_buf)
                    const moe_gate_up_phase = self.beginProfilePhase();
                    {
                        const qt = gate_exps.info.type_;
                        const pip = self.dmmv.moePipelineForType(qt) orelse unreachable;
                        if (pip.uses_push_descriptors) {
                            const push = MoeDmmvPushConstants{ .M = inter_dim, .K = hidden_dim, .expert_stride = expert_gate_row_bytes, .x_expert_stride = 0, .x_offset = 0, .y_offset = 0 };
                            const wg_x: u32 = switch (qt) { .q8_0, .f16 => (inter_dim + 1) / 2, else => (inter_dim + 63) / 64 };
                            self.pushDispatch4(pip, std.mem.asBytes(&push), gate_exps.gpu_buffer.handle, gate_exps.gpu_buffer.size, self.ffn_norm_buf.handle, hidden_size, self.gate_buf.handle, self.gate_buf.size, self.router_output_buf.handle, self.router_output_buf.size, wg_x, n_used, 1);
                        } else {
                            const ds = try self.allocDescSet(pip.descriptor_set_layout);
                            self.writeDescSet4(ds, gate_exps.gpu_buffer.handle, gate_exps.gpu_buffer.size, self.ffn_norm_buf.handle, hidden_size, self.gate_buf.handle, self.gate_buf.size, self.router_output_buf.handle, self.router_output_buf.size);
                            try self.dmmv.recordMoeDispatch(&self.decode_cmd, qt, ds, inter_dim, hidden_dim, expert_gate_row_bytes, n_used, 0, 0, 0);
                        }
                    }
                    // up DMMV: ALL experts at once
                    {
                        const qt = up_exps.info.type_;
                        const pip = self.dmmv.moePipelineForType(qt) orelse unreachable;
                        if (pip.uses_push_descriptors) {
                            const push = MoeDmmvPushConstants{ .M = inter_dim, .K = hidden_dim, .expert_stride = expert_gate_row_bytes, .x_expert_stride = 0, .x_offset = 0, .y_offset = 0 };
                            const wg_x: u32 = switch (qt) { .q8_0, .f16 => (inter_dim + 1) / 2, else => (inter_dim + 63) / 64 };
                            self.pushDispatch4(pip, std.mem.asBytes(&push), up_exps.gpu_buffer.handle, up_exps.gpu_buffer.size, self.ffn_norm_buf.handle, hidden_size, self.up_buf.handle, self.up_buf.size, self.router_output_buf.handle, self.router_output_buf.size, wg_x, n_used, 1);
                        } else {
                            const ds = try self.allocDescSet(pip.descriptor_set_layout);
                            self.writeDescSet4(ds, up_exps.gpu_buffer.handle, up_exps.gpu_buffer.size, self.ffn_norm_buf.handle, hidden_size, self.up_buf.handle, self.up_buf.size, self.router_output_buf.handle, self.router_output_buf.size);
                            try self.dmmv.recordMoeDispatch(&self.decode_cmd, qt, ds, inter_dim, hidden_dim, expert_gate_row_bytes, n_used, 0, 0, 0);
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
                        const pip = self.dmmv.moePipelineForType(qt) orelse unreachable;
                        if (pip.uses_push_descriptors) {
                            const push = MoeDmmvPushConstants{ .M = hidden_dim, .K = inter_dim, .expert_stride = expert_down_row_bytes, .x_expert_stride = inter_dim, .x_offset = 0, .y_offset = 0 };
                            const wg_x: u32 = switch (qt) { .q8_0, .f16 => (hidden_dim + 1) / 2, else => (hidden_dim + 63) / 64 };
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

                    // Post-FFN norm + residual for MoE expert accumulation (Gemma 4)
                    if (has_post_ffw_norm) {
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

                    // Remaining shared expert steps (sequential — buffer reuse prevents further overlap)
                    if (has_shared_expert) {
                        // Shared down DMMV: swiglu_buf → down_buf
                        const shared_down_phase = self.beginProfilePhase();
                        try self.dispatchDmmv(down_shexp.?, self.swiglu_buf, shexp_size, self.down_buf, hidden_dim, shexp_inter_dim);
                        self.decode_cmd.computeBarrier();
                        self.endProfilePhase(.shared_down, shared_down_phase);

                        // Post-FFN norm on shared expert down projection (Gemma 4)
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
                    }
                    try self.decode_cmd.end();
                    try self.decode_cmd.submitAndWait(self.instance.compute_queue);
                    const router_ptr: [*]const f32 = @ptrCast(@alignCast(self.router_staging.mapped.?));
                    const router_logits = router_ptr[0..config.n_experts];
                    topKSoftmax(router_logits, n_used, expert_ids[0..n_used], expert_weights[0..n_used]);

                    // New command buffer for expert FFN dispatch
                    if (self.instance.push_descriptor_fn == null) _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                    try self.decode_cmd.reset();
                    try self.decode_cmd.begin();

                    // Zero moe_out_buf via fill
                    vk.c.vkCmdFillBuffer(self.decode_cmd.handle, self.moe_out_buf.handle, 0, hidden_size, 0);
                    self.decode_cmd.transferToComputeBarrier();

                    for (0..n_used) |ei| {
                        const eid = expert_ids[ei];
                        const weight = expert_weights[ei];
                        const gate_offset = eid * expert_gate_row_bytes;
                        const up_offset = eid * expert_gate_row_bytes + up_base_offset;
                        const down_offset = eid * expert_down_row_bytes;

                        try self.dispatchDmmvWithOffset(gate_exps, self.ffn_norm_buf, hidden_size, self.gate_buf, inter_dim, hidden_dim, gate_offset);
                        try self.dispatchDmmvWithOffset(up_exps, self.ffn_norm_buf, hidden_size, self.up_buf, inter_dim, hidden_dim, up_offset);
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

                        try self.dispatchDmmvWithOffset(down_exps, self.swiglu_buf, self.swiglu_buf.size, self.down_buf, hidden_dim, inter_dim, down_offset);
                        self.decode_cmd.computeBarrier();

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
                self.endProfilePhase(.moe_routed, moe_phase);

                // CPU MoE fallback: apply post_ffw_norm to expert accumulation before shared expert
                if (!use_gpu_moe and lt.post_ffw_norm != null) {
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

                        // Post-FFN norm on shared expert down projection (Gemma 4)
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
                    try self.dispatchScaleAcc(
                        self.hidden_buf.handle,
                        hidden_size,
                        self.moe_out_buf.handle,
                        hidden_size,
                        hidden_dim,
                        1.0,
                    );
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
            if (state.position == 0 and self.validation_diagnostics_enabled) {
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

        const use_gpu_argmax = collect_output and self.argmax.pipeline != null and self.argmax_descriptor_set != null;
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
        const need_logits_readback = collect_output and (self.logits_readback_enabled or self.validation_diagnostics_enabled or !use_gpu_argmax);
        if (self.profile_enabled and collect_output and !use_gpu_argmax) {
            self.profile_token_counters.cpu_argmax_fallbacks += 1;
        }
        if (collect_output) {
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
        }
        self.endProfilePhase(.final_tail, final_tail_phase);
        _ = self.writeTimestamp(vk.c.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT);

        try self.decode_cmd.end();
        if (self.profile_enabled) {
            const cpu_record_end = std.time.nanoTimestamp();
            self.profile_token_counters.cpu_record_ns += @intCast(cpu_record_end - cpu_record_start);
        }
        const submit_wait_start = if (self.profile_enabled) std.time.nanoTimestamp() else 0;
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);
        if (self.profile_enabled) {
            const submit_wait_end = std.time.nanoTimestamp();
            self.profile_token_counters.submit_wait_ns += @intCast(submit_wait_end - submit_wait_start);
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
            // For Q4K large M (LM head), use batch shader for better parallelism
            if (qt == .q4_k and M > 65536) {
                if (self.dmmv.pipeline_q4k_batch) |*batch_pip| {
                    const batch_push = BatchDmmvPushConstants{
                        .M = M, .K = K,
                        .a_offset = a_offset, .x_offset = x_offset, .y_offset = y_offset,
                        .num_cols = 1,
                    };
                    self.pushDispatch3(
                        batch_pip,
                        std.mem.asBytes(&batch_push),
                        tensor.gpu_buffer.handle, tensor.gpu_buffer.size,
                        input_buf.handle, input_size,
                        output_buf.handle, output_buf.size,
                        (M + 63) / 64, 1, 1,
                    );
                    return;
                }
            }

            const push = DmmvPushConstants{
                .M = M, .K = K,
                .a_offset = a_offset, .x_offset = x_offset, .y_offset = y_offset,
                .acc_mode = acc_mode,
            };
            // Workgroup calculation (mirrors dmmv.recordDispatch)
            const wg_x: u32 = switch (qt) {
                .q4_k, .q5_k, .q6_k => (M + 1) / 2,
                .q8_0, .f16 => (M + 1) / 2,
                .f32 => M,
                else => (M + 63) / 64,
            };
            self.pushDispatch3(
                pip,
                std.mem.asBytes(&push),
                tensor.gpu_buffer.handle, tensor.gpu_buffer.size,
                input_buf.handle, input_size,
                output_buf.handle, output_buf.size,
                wg_x, 1, 1,
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
    fn runSsmLayerGpu(self: *InferenceEngine, state: *DecodeState, layer: u32, layer_idx: usize) !void {
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
        try self.dispatchDmmv(wqkv_tensor, self.norm_buf, hidden_size, self.attn_out_buf, @intCast(conv_channels), hidden_dim);
        try self.dispatchDmmv(z_tensor, self.norm_buf, hidden_size, self.gate_buf, @intCast(d_inner), hidden_dim);
        try self.dispatchDmmv(alpha_tensor, self.norm_buf, hidden_size, self.router_logits_buf, dt_rank, hidden_dim);
        try self.dispatchDmmv(beta_tensor, self.norm_buf, hidden_size, self.down_buf, dt_rank, hidden_dim);
        self.decode_cmd.computeBarrier();
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
        self.decode_cmd.computeBarrier();
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
        self.decode_cmd.computeBarrier();
        self.endProfilePhase(.ssm_delta, ssm_delta_phase);

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

        // Run each prompt token through the full transformer (same as decodeStep)
        // This populates KV cache and SSM state so the first decode token has context.
        for (prompt_tokens, 0..) |token_id, i| {
            const collect_output = i + 1 == prompt_tokens.len;
            try self.decodeStep(state, token_id, collect_output);
        }

        // Upload last token's embedding
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

    // Prefill: batch all prompt tokens in a single GPU submission
    const prefill_start = std.time.nanoTimestamp();
    try engine.prefillBatch(&state, prompt_tokens);
    const prefill_end = std.time.nanoTimestamp();
    const prefill_ns: u64 = @intCast(prefill_end - prefill_start);
    const prefill_tok_per_sec = if (prefill_ns > 0 and prompt_tokens.len > 0)
        @as(f64, @floatFromInt(prompt_tokens.len)) * 1_000_000_000.0 / @as(f64, @floatFromInt(prefill_ns))
    else
        0.0;

    log.debug("Prefill complete: {d} tokens in {d:.1} ms ({d:.2} tok/s)", .{
        prompt_tokens.len, @as(f64, @floatFromInt(prefill_ns)) / 1_000_000.0, prefill_tok_per_sec,
    });
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
