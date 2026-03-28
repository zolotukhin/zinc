//! Run the inference runtime: decode state, pipeline ownership, and token generation.
//! @section Inference Runtime
//! This module ties together model state, compute graphs, dispatch helpers,
//! and greedy token sampling for a single active inference engine.
const std = @import("std");
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const Buffer = @import("../vulkan/buffer.zig").Buffer;
const CommandPool = @import("../vulkan/command.zig").CommandPool;
const CommandBuffer = @import("../vulkan/command.zig").CommandBuffer;
const GpuConfig = @import("../vulkan/gpu_detect.zig").GpuConfig;
const loader = @import("../model/loader.zig");
const Model = loader.Model;
const ModelConfig = loader.ModelConfig;
const LoadedTensor = loader.LoadedTensor;
const architecture = @import("../model/architecture.zig");
const Graph = @import("graph.zig").Graph;
const DmmvDispatch = @import("dmmv.zig").DmmvDispatch;
const ElementwiseDispatch = @import("elementwise.zig").ElementwiseDispatch;
const AttentionDispatch = @import("attention.zig").AttentionDispatch;
const GGMLType = @import("../model/gguf.zig").GGMLType;

const log = std.log.scoped(.forward);

/// Runtime state for the decode loop.
pub const DecodeState = struct {
    /// Current token position.
    position: u32,
    /// Generated token IDs.
    generated_tokens: std.ArrayList(u32),
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,

    /// Initialize decode state for a fresh generation request.
    /// @param allocator Allocator used for the generated token list.
    /// @returns A DecodeState positioned at token index zero with an empty output buffer.
    pub fn init(allocator: std.mem.Allocator) DecodeState {
        return .{
            .position = 0,
            .generated_tokens = .{},
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
            // Element ordering is INTERLEAVED: y[2l] from low nibble, y[2l+1] from high nibble
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
                        output[out_i5] = d1_5 * @as(f32, @floatFromInt(ql_lo5 | (hb_lo5 << 4))) - m1_5;
                        out_i5 += 1;
                        output[out_i5] = d2_5 * @as(f32, @floatFromInt(ql_hi5 | (hb_hi5 << 4))) - m2_5;
                        out_i5 += 1;
                    }
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
    for (logits) |v| if (v > max_val) { max_val = v; };

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
    // KV cache (per-layer, for attention layers)
    kv_k_cache: []Buffer, // [n_layers] K cache buffers
    kv_v_cache: []Buffer, // [n_layers] V cache buffers
    page_table_buf: Buffer, // identity page table for flash attention (page_ids[i] = i)
    // SSM state (per-layer, CPU-side, for SSM layers)
    ssm_conv_states: [][]f32, // [n_layers] conv state: (kernel_size-1) * conv_channels
    ssm_states: [][]f32, // [n_layers] recurrent state: head_v_dim * head_v_dim * num_v_heads
    // Host-visible staging for SSM hidden state transfer
    ssm_hidden_staging: Buffer,
    // Descriptor management
    shared_pool: vk.c.VkDescriptorPool,
    /// Vulkan instance.
    instance: *const Instance,
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,
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

        // Build the decode graph (for diagnostics / future full-graph dispatch)
        var decode_graph = try architecture.buildDecodeGraph(config, allocator);
        errdefer decode_graph.deinit();

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
        const max_inter = @max(@max(inter_dim, shexp_inter), if (config.ssm_d_inner > 0) config.ssm_d_inner else inter_dim);
        const inter_size = @as(vk.c.VkDeviceSize, max_inter) * @sizeOf(f32);
        const n_experts_total = if (config.n_experts > 0) config.n_experts else @as(u32, 1);

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
        var gate_buf = try Buffer.initDeviceLocal(instance, inter_size, storage_xfer);
        errdefer gate_buf.deinit();
        var up_buf = try Buffer.initDeviceLocal(instance, inter_size, storage_xfer);
        errdefer up_buf.deinit();
        var swiglu_buf = try Buffer.initDeviceLocal(instance, inter_size, storage_xfer);
        errdefer swiglu_buf.deinit();
        var down_buf = try Buffer.initDeviceLocal(instance, hidden_size, storage_xfer);
        errdefer down_buf.deinit();
        var moe_out_buf = try Buffer.initDeviceLocal(instance, hidden_size, storage_xfer);
        errdefer moe_out_buf.deinit();

        const router_size = @as(vk.c.VkDeviceSize, n_experts_total) * @sizeOf(f32);
        var router_logits_buf = try Buffer.initDeviceLocal(instance, router_size, storage_xfer);
        errdefer router_logits_buf.deinit();
        var router_staging = try Buffer.init(
            instance, router_size,
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

        // KV cache: per-layer, flat layout (context_length * kv_dim * sizeof(f32))
        const max_ctx: u32 = @min(config.context_length, 4096); // cap for now
        const kv_cache_per_layer = @as(vk.c.VkDeviceSize, max_ctx) * @as(vk.c.VkDeviceSize, kv_dim) * @sizeOf(f32);
        const kv_k_cache = try allocator.alloc(Buffer, config.n_layers);
        errdefer allocator.free(kv_k_cache);
        const kv_v_cache = try allocator.alloc(Buffer, config.n_layers);
        errdefer allocator.free(kv_v_cache);

        for (0..config.n_layers) |i| {
            kv_k_cache[i] = try Buffer.initDeviceLocal(instance, kv_cache_per_layer, storage_xfer);
            kv_v_cache[i] = try Buffer.initDeviceLocal(instance, kv_cache_per_layer, storage_xfer);
        }

        log.info("KV cache: {d} layers × {d} MB = {d} MB total", .{
            config.n_layers,
            kv_cache_per_layer * 2 / (1024 * 1024),
            config.n_layers * kv_cache_per_layer * 2 / (1024 * 1024),
        });

        // Identity page table for flash attention: page_ids[i] = i (flat KV layout)
        const page_table_size = @as(vk.c.VkDeviceSize, max_ctx) * @sizeOf(u32);
        var page_table_buf = try Buffer.init(
            instance, page_table_size,
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );
        errdefer page_table_buf.deinit();
        {
            var map_ptr: ?*anyopaque = null;
            const mr_pt = vk.c.vkMapMemory(instance.device, page_table_buf.memory, 0, page_table_size, 0, &map_ptr);
            if (mr_pt != vk.c.VK_SUCCESS) return error.MapMemoryFailed;
            const pt_u32: [*]u32 = @ptrCast(@alignCast(map_ptr));
            for (0..max_ctx) |i| pt_u32[i] = @intCast(i);
            vk.c.vkUnmapMemory(instance.device, page_table_buf.memory);
        }

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
            log.info("SSM state: {d} layers × {d} KB conv + {d} KB recurrent", .{
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
            instance, ssm_staging_size,
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
        const pool_result = vk.c.vkCreateDescriptorPool(instance.device, &pool_info, null, &shared_pool);
        if (pool_result != vk.c.VK_SUCCESS) return error.DescriptorPoolCreateFailed;
        errdefer vk.c.vkDestroyDescriptorPool(instance.device, shared_pool, null);

        log.info("Inference engine ready — {d} graph nodes, hidden_dim={d}, vocab={d}", .{
            decode_graph.nodeCount(), config.hidden_dim, config.vocab_size,
        });

        return InferenceEngine{
            .model = model,
            .gpu_config = gpu_config,
            .dmmv = dmmv,
            .elementwise = elementwise,
            .attention = attention,
            .cmd_pool = cmd_pool,
            .decode_cmd = decode_cmd,
            .decode_graph = decode_graph,
            .hidden_buf = hidden_buf,
            .residual_buf = residual_buf,
            .norm_buf = norm_buf,
            .logits_buf = logits_buf,
            .logits_staging = logits_staging,
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
            .kv_k_cache = kv_k_cache,
            .kv_v_cache = kv_v_cache,
            .page_table_buf = page_table_buf,
            .ssm_conv_states = ssm_conv_states,
            .ssm_states = ssm_states,
            .ssm_hidden_staging = ssm_hidden_staging,
            .shared_pool = shared_pool,
            .instance = instance,
            .allocator = allocator,
        };
    }

    // -----------------------------------------------------------------------
    // Descriptor set helpers
    // -----------------------------------------------------------------------

    /// Allocate a descriptor set from the shared pool with the given layout.
    fn allocDescSet(self: *const InferenceEngine, layout: vk.c.VkDescriptorSetLayout) !vk.c.VkDescriptorSet {
        const alloc_info = vk.c.VkDescriptorSetAllocateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = self.shared_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &layout,
        };
        var ds: vk.c.VkDescriptorSet = null;
        const result = vk.c.vkAllocateDescriptorSets(self.instance.device, &alloc_info, &ds);
        if (result != vk.c.VK_SUCCESS) return error.DescriptorSetAllocFailed;
        return ds;
    }

    /// Write storage buffer bindings to a descriptor set (up to 8).
    fn writeDescSet3(
        self: *const InferenceEngine,
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
    }

    // -----------------------------------------------------------------------
    // Layer tensor lookup
    // -----------------------------------------------------------------------

    fn findLayerTensor(self: *const InferenceEngine, layer: u32, name: []const u8) ?*const LoadedTensor {
        var buf: [128]u8 = undefined;
        const key = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ layer, name }) catch return null;
        return findLoadedTensor(self.model, key);
    }

    // -----------------------------------------------------------------------
    // Descriptor set helpers
    // -----------------------------------------------------------------------

    fn writeDescSet2(
        self: *const InferenceEngine,
        ds: vk.c.VkDescriptorSet,
        buf0: vk.c.VkBuffer, size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer, size1: vk.c.VkDeviceSize,
    ) void {
        var buffer_infos = [2]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = buf0, .offset = 0, .range = size0 },
            .{ .buffer = buf1, .offset = 0, .range = size1 },
        };
        var writes: [2]vk.c.VkWriteDescriptorSet = undefined;
        for (0..2) |i| {
            writes[i] = .{
                .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null, .dstSet = ds, .dstBinding = @intCast(i),
                .dstArrayElement = 0, .descriptorCount = 1,
                .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null, .pBufferInfo = &buffer_infos[i], .pTexelBufferView = null,
            };
        }
        vk.c.vkUpdateDescriptorSets(self.instance.device, 2, &writes, 0, null);
    }

    fn writeDescSet5(
        self: *const InferenceEngine,
        ds: vk.c.VkDescriptorSet,
        buf0: vk.c.VkBuffer, size0: vk.c.VkDeviceSize,
        buf1: vk.c.VkBuffer, size1: vk.c.VkDeviceSize,
        buf2: vk.c.VkBuffer, size2: vk.c.VkDeviceSize,
        buf3: vk.c.VkBuffer, size3: vk.c.VkDeviceSize,
        buf4: vk.c.VkBuffer, size4: vk.c.VkDeviceSize,
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
                .pNext = null, .dstSet = ds, .dstBinding = @intCast(i),
                .dstArrayElement = 0, .descriptorCount = 1,
                .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null, .pBufferInfo = &buffer_infos[i], .pTexelBufferView = null,
            };
        }
        vk.c.vkUpdateDescriptorSets(self.instance.device, 5, &writes, 0, null);
    }

    // -----------------------------------------------------------------------
    // Embedding
    // -----------------------------------------------------------------------

    /// Dequantize a token's embedding row directly into the pre-allocated staging buffer.
    /// The GPU copy (staging → hidden_buf) is recorded in the decode command buffer.
    fn embedToken(self: *InferenceEngine, token_id: u32) !void {
        const hidden_dim = self.model.config.hidden_dim;
        const safe_id = @min(token_id, self.model.config.vocab_size -| 1);

        const embd = findLoadedTensor(self.model, "token_embd.weight") orelse {
            log.err("token_embd.weight not found", .{});
            return error.TensorNotFound;
        };

        const mmap = self.model.mmap_data orelse return error.NoMmapData;
        const data_start: usize = @intCast(self.model.gguf_file.tensor_data_offset + embd.info.offset);

        // Dequantize directly into pre-allocated staging buffer (zero alloc)
        const staging_f32: [*]f32 = @ptrCast(@alignCast(self.embed_staging.mapped.?));
        dequantRow(mmap[data_start..], safe_id, hidden_dim, embd.info.type_, staging_f32[0..hidden_dim]);
    }

    // -----------------------------------------------------------------------
    // Decode step
    // -----------------------------------------------------------------------

    /// Run a single decode step through all transformer layers.
    /// embed → [per-layer: norm → QKV → RoPE → KV write → attention → O proj → residual
    ///          → FFN norm → MoE routing → expert DMMVs → residual] → final norm → LM head → logits
    pub fn decodeStep(self: *InferenceEngine, state: *DecodeState, token_id: u32) !void {
        const config = &self.model.config;
        const hidden_dim = config.hidden_dim;
        const hidden_size = @as(vk.c.VkDeviceSize, hidden_dim) * @sizeOf(f32);
        const q_dim = @as(u32, config.n_heads) * config.head_dim;
        const kv_dim = @as(u32, config.n_kv_heads) * config.head_dim;
        const kv_vec_size = @as(vk.c.VkDeviceSize, kv_dim) * @sizeOf(f32);
        const is_moe = config.n_experts > 0;
        const inter_dim = if (config.intermediate_dim > 0) config.intermediate_dim else hidden_dim * 4;
        const shexp_inter_dim = if (config.shared_expert_intermediate_dim > 0) config.shared_expert_intermediate_dim else inter_dim;
        // Hybrid models: every Nth layer is full attention, rest are SSM/linear attention
        const full_attn_interval = if (config.full_attn_interval > 0) config.full_attn_interval else 1;

        // Log MoE dimensions once (first decode)
        if (state.generated_tokens.items.len == 0 and is_moe) {
            log.info("MoE dims: expert_inter={d} shared_expert_inter={d} hidden={d}", .{ inter_dim, shexp_inter_dim, hidden_dim });
        }

        // 1. CPU: dequantize embedding
        try self.embedToken(token_id);

        // Per-layer logit5 tracking for BOS diagnostic summary
        var diag_logit5 = [_]f32{0} ** 64;
        var diag_rms_arr = [_]f32{0} ** 64;

        for (0..config.n_layers) |layer_idx| {
            const layer: u32 = @intCast(layer_idx);

            // Reset descriptor pool for this layer's work
            _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
            try self.decode_cmd.reset();
            try self.decode_cmd.begin();

            // --- Upload embedding (only first layer) ---
            if (layer == 0) {
                const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = hidden_size };
                vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.embed_staging.handle, self.hidden_buf.handle, 1, &region);
                self.decode_cmd.transferToComputeBarrier();

            }


            // --- Input RMS norm: hidden_buf → norm_buf ---
            const attn_norm = self.findLayerTensor(layer, "attn_norm.weight") orelse {
                log.err("Layer {d}: attn_norm.weight not found", .{layer});
                return error.TensorNotFound;
            };
            {
                const pip = &(self.elementwise.pipeline_rms_norm orelse return error.ShaderNotLoaded);
                const ds = try self.allocDescSet(pip.descriptor_set_layout);
                self.writeDescSet3(ds, self.hidden_buf.handle, hidden_size,
                    attn_norm.gpu_buffer.handle, attn_norm.gpu_buffer.size,
                    self.norm_buf.handle, hidden_size);
                try self.elementwise.recordRmsNorm(&self.decode_cmd, ds, hidden_dim, 1, 1e-6);
            }
            self.decode_cmd.computeBarrier();

            const is_full_attn = ((layer + 1) % full_attn_interval == 0);

            if (is_full_attn) {
                // === FULL ATTENTION LAYER ===
                // Q+gate projection → strided Q extraction → Q/K norm → K/V proj → RoPE →
                // KV cache → flash attention → sigmoid gate → output projection → residual

                const q_tensor = self.findLayerTensor(layer, "attn_q.weight") orelse return error.TensorNotFound;
                const k_tensor = self.findLayerTensor(layer, "attn_k.weight") orelse return error.TensorNotFound;
                const v_tensor = self.findLayerTensor(layer, "attn_v.weight") orelse return error.TensorNotFound;

                // Bug fix #4: Q+gate are interleaved per head as [Q_head, gate_head, Q_head, gate_head, ...]
                // attn_q outputs (head_dim * 2) * n_heads. Q is at stride-2 offsets.
                // For now, project Q+gate together, then extract Q with strided copy.
                const q_full_dim = q_dim * 2;
                try self.dispatchDmmv(q_tensor, self.norm_buf, hidden_size, self.attn_out_buf, q_full_dim, hidden_dim);
                try self.dispatchDmmv(k_tensor, self.norm_buf, hidden_size, self.k_buf, kv_dim, hidden_dim);
                try self.dispatchDmmv(v_tensor, self.norm_buf, hidden_size, self.v_buf, kv_dim, hidden_dim);
                self.decode_cmd.computeBarrier();

                // Q+gate are block-interleaved per head:
                // [head0_Q(head_dim), head0_gate(head_dim), head1_Q(head_dim), head1_gate(head_dim), ...]
                // Extract Q and gate into separate buffers using per-head copies
                {
                    const hd = config.head_dim;
                    const hd_bytes = @as(vk.c.VkDeviceSize, hd) * @sizeOf(f32);
                    const stride_bytes = hd_bytes * 2;
                    for (0..config.n_heads) |h| {
                        const src_off = @as(vk.c.VkDeviceSize, @intCast(h)) * stride_bytes;
                        const dst_off = @as(vk.c.VkDeviceSize, @intCast(h)) * hd_bytes;
                        const qr = vk.c.VkBufferCopy{ .srcOffset = src_off, .dstOffset = dst_off, .size = hd_bytes };
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.attn_out_buf.handle, self.q_buf.handle, 1, &qr);
                        const gr = vk.c.VkBufferCopy{ .srcOffset = src_off + hd_bytes, .dstOffset = dst_off, .size = hd_bytes };
                        vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.attn_out_buf.handle, self.gate_buf.handle, 1, &gr);
                    }
                }
                self.decode_cmd.transferToComputeBarrier();

                // Bug fix #1: Q/K normalization (per-head RMS norm)
                // attn_q_norm and attn_k_norm are per-head norms with head_dim weights
                const q_norm_tensor = self.findLayerTensor(layer, "attn_q_norm.weight");
                const k_norm_tensor = self.findLayerTensor(layer, "attn_k_norm.weight");
                if (q_norm_tensor) |qn| {
                    const pip = &(self.elementwise.pipeline_rms_norm orelse return error.ShaderNotLoaded);
                    // Apply RMS norm to each Q head (n_heads workgroups, head_dim elements each)
                    const ds = try self.allocDescSet(pip.descriptor_set_layout);
                    self.writeDescSet3(ds, self.q_buf.handle, self.q_buf.size,
                        qn.gpu_buffer.handle, qn.gpu_buffer.size,
                        self.q_buf.handle, self.q_buf.size); // in-place via same output
                    try self.elementwise.recordRmsNorm(&self.decode_cmd, ds, config.head_dim, config.n_heads, 1e-6);
                }
                if (k_norm_tensor) |kn| {
                    const pip = &(self.elementwise.pipeline_rms_norm orelse return error.ShaderNotLoaded);
                    const ds = try self.allocDescSet(pip.descriptor_set_layout);
                    self.writeDescSet3(ds, self.k_buf.handle, self.k_buf.size,
                        kn.gpu_buffer.handle, kn.gpu_buffer.size,
                        self.k_buf.handle, self.k_buf.size);
                    try self.elementwise.recordRmsNorm(&self.decode_cmd, ds, config.head_dim, config.n_kv_heads, 1e-6);
                }
                self.decode_cmd.computeBarrier();

                // Bug fix #5+#6: IMRoPE — only rotate rope_dim of head_dim dimensions
                const rope_freq = config.rope_freq_base;
                const rope_dim: u32 = if (config.rope_dim > 0) config.rope_dim else config.head_dim;
                {
                    const pip = &(self.elementwise.pipeline_rope orelse return error.ShaderNotLoaded);
                    const q_ds = try self.allocDescSet(pip.descriptor_set_layout);
                    self.writeDescSet2(q_ds, self.q_buf.handle, self.q_buf.size, self.q_buf.handle, self.q_buf.size);
                    try self.elementwise.recordRope(&self.decode_cmd, q_ds, config.head_dim, rope_dim, config.n_heads, state.position, rope_freq);

                    const k_ds = try self.allocDescSet(pip.descriptor_set_layout);
                    self.writeDescSet2(k_ds, self.k_buf.handle, self.k_buf.size, self.k_buf.handle, self.k_buf.size);
                    try self.elementwise.recordRope(&self.decode_cmd, k_ds, config.head_dim, rope_dim, config.n_kv_heads, state.position, rope_freq);
                }
                self.decode_cmd.computeBarrier();

                // KV cache write
                {
                    const kv_offset = @as(vk.c.VkDeviceSize, state.position) * kv_vec_size;
                    const k_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = kv_offset, .size = kv_vec_size };
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.k_buf.handle, self.kv_k_cache[layer_idx].handle, 1, &k_region);
                    const v_region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = kv_offset, .size = kv_vec_size };
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.v_buf.handle, self.kv_v_cache[layer_idx].handle, 1, &v_region);
                }
                self.decode_cmd.transferToComputeBarrier();

                // Flash attention
                if (self.attention.pipeline) |*pip| {
                    const attn_ds = try self.allocDescSet(pip.descriptor_set_layout);
                    self.writeDescSet5(attn_ds,
                        self.q_buf.handle, self.q_buf.size,
                        self.kv_k_cache[layer_idx].handle, self.kv_k_cache[layer_idx].size,
                        self.kv_v_cache[layer_idx].handle, self.kv_v_cache[layer_idx].size,
                        self.page_table_buf.handle, self.page_table_buf.size,
                        self.attn_out_buf.handle, self.attn_out_buf.size);
                    try self.attention.recordFlashAttn(&self.decode_cmd, attn_ds,
                        config.head_dim, config.n_heads, config.n_kv_heads,
                        state.position + 1, 1);
                }
                self.decode_cmd.computeBarrier();

                // Bug fix #2: Attention gating — attn_output = attn_output * sigmoid(gate)
                // gate_buf has the per-head gate values, attn_out_buf has the attention output
                if (self.elementwise.pipeline_sigmoid_mul) |*pip| {
                    const gds = try self.allocDescSet(pip.descriptor_set_layout);
                    self.writeDescSet3(gds, self.attn_out_buf.handle, self.attn_out_buf.size,
                        self.gate_buf.handle, self.gate_buf.size,
                        self.attn_out_buf.handle, self.attn_out_buf.size); // in-place
                    try self.elementwise.recordSigmoidMul(&self.decode_cmd, gds, q_dim);
                    self.decode_cmd.computeBarrier();
                }

                // Output projection: attn_output.weight
                const o_tensor = self.findLayerTensor(layer, "attn_output.weight") orelse return error.TensorNotFound;
                try self.dispatchDmmv(o_tensor, self.attn_out_buf, self.attn_out_buf.size, self.o_proj_buf, hidden_dim, q_dim);
                self.decode_cmd.computeBarrier();

                // Attention residual: hidden_buf += o_proj_buf
                {
                    const pip = &(self.elementwise.pipeline_scale_acc orelse return error.ShaderNotLoaded);
                    const ds = try self.allocDescSet(pip.descriptor_set_layout);
                    self.writeDescSet2(ds, self.hidden_buf.handle, hidden_size, self.o_proj_buf.handle, hidden_size);
                    try self.elementwise.recordScaleAcc(&self.decode_cmd, ds, hidden_dim, 1.0);
                }
                self.decode_cmd.computeBarrier();
            } else {
                // === SSM / LINEAR ATTENTION LAYER (CPU-side delta-net) ===
                try self.runSsmLayerCpu(state, layer, layer_idx);
            }

            // --- Post-attention norm (Qwen3.5 uses post_attention_norm, not ffn_norm) ---
            const ffn_norm_tensor = self.findLayerTensor(layer, "post_attention_norm.weight") orelse
                self.findLayerTensor(layer, "ffn_norm.weight") orelse return error.TensorNotFound;
            {
                const pip = &(self.elementwise.pipeline_rms_norm orelse return error.ShaderNotLoaded);
                const ds = try self.allocDescSet(pip.descriptor_set_layout);
                self.writeDescSet3(ds, self.hidden_buf.handle, hidden_size,
                    ffn_norm_tensor.gpu_buffer.handle, ffn_norm_tensor.gpu_buffer.size,
                    self.ffn_norm_buf.handle, hidden_size);
                try self.elementwise.recordRmsNorm(&self.decode_cmd, ds, hidden_dim, 1, 1e-6);
            }
            self.decode_cmd.computeBarrier();

            if (is_moe) {
                // --- MoE: router DMMV → readback → CPU top-k → expert dispatch ---
                const router_tensor = self.findLayerTensor(layer, "ffn_gate_inp.weight") orelse return error.TensorNotFound;
                try self.dispatchDmmv(router_tensor, self.ffn_norm_buf, hidden_size, self.router_logits_buf, config.n_experts, hidden_dim);

                // Readback router logits for CPU top-k
                {
                    const barrier = vk.c.VkMemoryBarrier{
                        .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = null,
                        .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                        .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
                    };
                    vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle,
                        vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
                        0, 1, &barrier, 0, null, 0, null);
                    const router_size = @as(vk.c.VkDeviceSize, config.n_experts) * @sizeOf(f32);
                    const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = router_size };
                    vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.router_logits_buf.handle, self.router_staging.handle, 1, &region);
                }

                // Submit and wait — need router logits on CPU for top-k
                try self.decode_cmd.end();
                try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                // CPU: softmax + top-k expert selection
                const n_used = config.n_experts_used;
                const router_ptr: [*]const f32 = @ptrCast(@alignCast(self.router_staging.mapped.?));
                const router_logits = router_ptr[0..config.n_experts];

                var expert_ids: [16]u32 = undefined; // max 16 experts
                var expert_weights: [16]f32 = undefined;
                topKSoftmax(router_logits, n_used, expert_ids[0..n_used], expert_weights[0..n_used]);

                // New command buffer for expert FFN dispatch
                _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                try self.decode_cmd.reset();
                try self.decode_cmd.begin();

                // Zero moe_out_buf via fill
                vk.c.vkCmdFillBuffer(self.decode_cmd.handle, self.moe_out_buf.handle, 0, hidden_size, 0);
                self.decode_cmd.transferToComputeBarrier();

                // Dispatch each selected expert
                const gate_exps = self.findLayerTensor(layer, "ffn_gate_exps.weight") orelse return error.TensorNotFound;
                const up_exps = self.findLayerTensor(layer, "ffn_up_exps.weight") orelse return error.TensorNotFound;
                const down_exps = self.findLayerTensor(layer, "ffn_down_exps.weight") orelse return error.TensorNotFound;

                const gate_quant = gate_exps.info.type_;
                const down_quant = down_exps.info.type_;
                // Expert weight offset: each expert has inter_dim rows of K=hidden_dim
                const expert_gate_row_bytes = expertSliceBytes(gate_quant, inter_dim, hidden_dim);
                // Down projection: each expert has hidden_dim rows of K=inter_dim
                const expert_down_row_bytes = expertSliceBytes(down_quant, hidden_dim, inter_dim);

                for (0..n_used) |ei| {
                    const eid = expert_ids[ei];
                    const weight = expert_weights[ei];
                    const gate_offset = eid * expert_gate_row_bytes;
                    const up_offset = eid * expert_gate_row_bytes; // same shape as gate
                    const down_offset = eid * expert_down_row_bytes;

                    // gate DMMV: ffn_gate_exps[expert] × ffn_norm_buf → gate_buf
                    try self.dispatchDmmvWithOffset(gate_exps, self.ffn_norm_buf, hidden_size, self.gate_buf, inter_dim, hidden_dim, gate_offset);
                    // up DMMV: ffn_up_exps[expert] × ffn_norm_buf → up_buf
                    try self.dispatchDmmvWithOffset(up_exps, self.ffn_norm_buf, hidden_size, self.up_buf, inter_dim, hidden_dim, up_offset);
                    self.decode_cmd.computeBarrier();

                    // SwiGLU: gate_buf, up_buf → swiglu_buf
                    {
                        const pip = &(self.elementwise.pipeline_swiglu orelse return error.ShaderNotLoaded);
                        const ds = try self.allocDescSet(pip.descriptor_set_layout);
                        self.writeDescSet3(ds, self.gate_buf.handle, self.gate_buf.size,
                            self.up_buf.handle, self.up_buf.size,
                            self.swiglu_buf.handle, self.swiglu_buf.size);
                        try self.elementwise.recordSwiglu(&self.decode_cmd, ds, inter_dim);
                    }
                    self.decode_cmd.computeBarrier();

                    // down DMMV: ffn_down_exps[expert] × swiglu_buf → down_buf
                    try self.dispatchDmmvWithOffset(down_exps, self.swiglu_buf, self.swiglu_buf.size, self.down_buf, hidden_dim, inter_dim, down_offset);
                    self.decode_cmd.computeBarrier();

                    // Weighted accumulate: moe_out_buf += weight * down_buf
                    {
                        const pip = &(self.elementwise.pipeline_scale_acc orelse return error.ShaderNotLoaded);
                        const ds = try self.allocDescSet(pip.descriptor_set_layout);
                        self.writeDescSet2(ds, self.moe_out_buf.handle, hidden_size, self.down_buf.handle, hidden_size);
                        try self.elementwise.recordScaleAcc(&self.decode_cmd, ds, hidden_dim, weight);
                    }
                    self.decode_cmd.computeBarrier();
                }

                // Bug fix #3: Shared expert — runs every token alongside the routed experts
                const gate_shexp = self.findLayerTensor(layer, "ffn_gate_shexp.weight");
                const up_shexp = self.findLayerTensor(layer, "ffn_up_shexp.weight");
                const down_shexp = self.findLayerTensor(layer, "ffn_down_shexp.weight");
                const shexp_gate = self.findLayerTensor(layer, "ffn_gate_inp_shexp.weight");

                if (gate_shexp != null and up_shexp != null and down_shexp != null) {
                    // Shared expert has its own intermediate dim (feed_forward_length), different from per-expert dim
                    const shexp_size = @as(vk.c.VkDeviceSize, shexp_inter_dim) * @sizeOf(f32);

                    // Shared expert FFN: gate + up → SwiGLU → down
                    try self.dispatchDmmv(gate_shexp.?, self.ffn_norm_buf, hidden_size, self.gate_buf, shexp_inter_dim, hidden_dim);
                    try self.dispatchDmmv(up_shexp.?, self.ffn_norm_buf, hidden_size, self.up_buf, shexp_inter_dim, hidden_dim);
                    // Dispatch shared expert gate projection in parallel (1 scalar output)
                    if (shexp_gate) |sg| {
                        try self.dispatchDmmv(sg, self.ffn_norm_buf, hidden_size, self.router_logits_buf, 1, hidden_dim);
                    }
                    self.decode_cmd.computeBarrier();

                    {
                        const pip = &(self.elementwise.pipeline_swiglu orelse return error.ShaderNotLoaded);
                        const ds2 = try self.allocDescSet(pip.descriptor_set_layout);
                        self.writeDescSet3(ds2, self.gate_buf.handle, self.gate_buf.size,
                            self.up_buf.handle, self.up_buf.size,
                            self.swiglu_buf.handle, self.swiglu_buf.size);
                        try self.elementwise.recordSwiglu(&self.decode_cmd, ds2, shexp_inter_dim);
                    }
                    self.decode_cmd.computeBarrier();

                    try self.dispatchDmmv(down_shexp.?, self.swiglu_buf, shexp_size, self.down_buf, hidden_dim, shexp_inter_dim);
                    self.decode_cmd.computeBarrier();

                    // Apply shared expert gate: sigmoid(ffn_gate_inp_shexp @ ffn_norm_buf)
                    var shexp_weight: f32 = 1.0;
                    if (shexp_gate != null) {
                        // Readback gate scalar from router_logits_buf
                        {
                            const bar = vk.c.VkMemoryBarrier{
                                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = null,
                                .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
                            };
                            vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle,
                                vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
                                0, 1, &bar, 0, null, 0, null);
                            const rgn = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = @sizeOf(f32) };
                            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.router_logits_buf.handle, self.router_staging.handle, 1, &rgn);
                        }
                        try self.decode_cmd.end();
                        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

                        const gate_ptr: [*]const f32 = @ptrCast(@alignCast(self.router_staging.mapped.?));
                        shexp_weight = 1.0 / (1.0 + @exp(-gate_ptr[0]));

                        _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
                        try self.decode_cmd.reset();
                        try self.decode_cmd.begin();
                    }
                    // Shared expert output → add to moe_out_buf with gated weight
                    {
                        const pip = &(self.elementwise.pipeline_scale_acc orelse return error.ShaderNotLoaded);
                        const ds2 = try self.allocDescSet(pip.descriptor_set_layout);
                        self.writeDescSet2(ds2, self.moe_out_buf.handle, hidden_size, self.down_buf.handle, hidden_size);
                        try self.elementwise.recordScaleAcc(&self.decode_cmd, ds2, hidden_dim, shexp_weight);
                    }
                    self.decode_cmd.computeBarrier();
                }

                // FFN residual: hidden_buf += moe_out_buf
                {
                    const pip = &(self.elementwise.pipeline_scale_acc orelse return error.ShaderNotLoaded);
                    const ds = try self.allocDescSet(pip.descriptor_set_layout);
                    self.writeDescSet2(ds, self.hidden_buf.handle, hidden_size, self.moe_out_buf.handle, hidden_size);
                    try self.elementwise.recordScaleAcc(&self.decode_cmd, ds, hidden_dim, 1.0);
                }
            } else {
                // Dense FFN: gate → up → SwiGLU → down → residual
                const gate_tensor = self.findLayerTensor(layer, "ffn_gate.weight") orelse return error.TensorNotFound;
                const up_tensor = self.findLayerTensor(layer, "ffn_up.weight") orelse return error.TensorNotFound;
                const down_tensor = self.findLayerTensor(layer, "ffn_down.weight") orelse return error.TensorNotFound;

                try self.dispatchDmmv(gate_tensor, self.ffn_norm_buf, hidden_size, self.gate_buf, inter_dim, hidden_dim);
                try self.dispatchDmmv(up_tensor, self.ffn_norm_buf, hidden_size, self.up_buf, inter_dim, hidden_dim);
                self.decode_cmd.computeBarrier();

                {
                    const pip = &(self.elementwise.pipeline_swiglu orelse return error.ShaderNotLoaded);
                    const ds = try self.allocDescSet(pip.descriptor_set_layout);
                    self.writeDescSet3(ds, self.gate_buf.handle, self.gate_buf.size,
                        self.up_buf.handle, self.up_buf.size,
                        self.swiglu_buf.handle, self.swiglu_buf.size);
                    try self.elementwise.recordSwiglu(&self.decode_cmd, ds, inter_dim);
                }
                self.decode_cmd.computeBarrier();

                try self.dispatchDmmv(down_tensor, self.swiglu_buf, self.swiglu_buf.size, self.down_buf, hidden_dim, inter_dim);
                self.decode_cmd.computeBarrier();

                // FFN residual: hidden_buf += down_buf
                {
                    const pip = &(self.elementwise.pipeline_scale_acc orelse return error.ShaderNotLoaded);
                    const ds = try self.allocDescSet(pip.descriptor_set_layout);
                    self.writeDescSet2(ds, self.hidden_buf.handle, hidden_size, self.down_buf.handle, hidden_size);
                    try self.elementwise.recordScaleAcc(&self.decode_cmd, ds, hidden_dim, 1.0);
                }
            }

            // If not MoE (command buffer still open), or after MoE expert dispatch
            if (!is_moe) {
                try self.decode_cmd.end();
                try self.decode_cmd.submitAndWait(self.instance.compute_queue);
            } else {
                // MoE path: command buffer is still open from expert dispatch
                try self.decode_cmd.end();
                try self.decode_cmd.submitAndWait(self.instance.compute_queue);
            }

            // --- Debug: per-layer hidden_buf diagnostics (BOS token only for compact output) ---
            if (state.position == 0) {
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
                        const rms_inv: f32 = @floatCast(1.0 / @sqrt(diag_sum_sq / @as(f64, @floatFromInt(hidden_dim)) + 1e-6));
                        const norm_t = findLoadedTensor(self.model, "output_norm.weight");
                        const lm_t = findLoadedTensor(self.model, "output.weight") orelse
                            findLoadedTensor(self.model, "token_embd.weight");
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
                    const qt_attn_norm = if (self.findLayerTensor(0, "attn_norm.weight")) |t| @tagName(t.info.type_) else "?";
                    const qt_qkv = if (self.findLayerTensor(0, "attn_qkv.weight")) |t| @tagName(t.info.type_) else "?";
                    const qt_gate_exps = if (self.findLayerTensor(0, "ffn_gate_exps.weight")) |t| @tagName(t.info.type_) else "?";
                    const qt_down_exps = if (self.findLayerTensor(0, "ffn_down_exps.weight")) |t| @tagName(t.info.type_) else "?";
                    const qt_ssm_out = if (self.findLayerTensor(0, "ssm_out.weight")) |t| @tagName(t.info.type_) else "?";
                    log.info("QUANT: attn_norm={s} qkv={s} gate_exps={s} down_exps={s} ssm_out={s}", .{
                        qt_attn_norm, qt_qkv, qt_gate_exps, qt_down_exps, qt_ssm_out,
                    });
                }

                if (layer < 64) {
                    diag_logit5[layer] = logit5;
                    diag_rms_arr[layer] = diag_rms;
                }
                log.info("L{d} {s}: rms={d:.4} max={d:.4} h0={d:.6} logit5={d:.4}", .{
                    layer,
                    if (is_full_attn) @as([]const u8, "A") else @as([]const u8, "S"),
                    diag_rms, diag_max_abs, hptr[0], logit5,
                });
            }
        }

        // === Per-layer diagnostic summary (stored for printing after generation) ===
        if (state.position == 0 and config.n_layers <= 64) {
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
        _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.begin();

        // Final RMS norm: hidden_buf → norm_buf
        const final_norm_tensor = findLoadedTensor(self.model, "output_norm.weight") orelse return error.TensorNotFound;
        {
            const pip = &(self.elementwise.pipeline_rms_norm orelse return error.ShaderNotLoaded);
            const ds = try self.allocDescSet(pip.descriptor_set_layout);
            self.writeDescSet3(ds, self.hidden_buf.handle, hidden_size,
                final_norm_tensor.gpu_buffer.handle, final_norm_tensor.gpu_buffer.size,
                self.norm_buf.handle, hidden_size);
            try self.elementwise.recordRmsNorm(&self.decode_cmd, ds, hidden_dim, 1, 1e-6);
        }
        self.decode_cmd.computeBarrier();

        // LM head: output.weight × norm_buf → logits_buf
        const lm_tensor = findLoadedTensor(self.model, "output.weight") orelse
            findLoadedTensor(self.model, "token_embd.weight") orelse return error.TensorNotFound;
        try self.dispatchDmmv(lm_tensor, self.norm_buf, hidden_size, self.logits_buf, self.model.config.vocab_size, hidden_dim);

        // Readback logits
        {
            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
            };
            vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle,
                vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &barrier, 0, null, 0, null);
            const logits_copy_size = @as(vk.c.VkDeviceSize, self.model.config.vocab_size) * @sizeOf(f32);
            const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = logits_copy_size };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.logits_buf.handle, self.logits_staging.handle, 1, &region);
        }

        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

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
        input_buf: Buffer, input_size: vk.c.VkDeviceSize,
        /// GPU buffer for output buf.
        output_buf: Buffer,
        M: u32, K: u32,
    ) !void {
        const qt = tensor.info.type_;
        const pip = self.dmmv.pipelineForType(qt) orelse {
            log.err("No DMMV pipeline for quant type {d} (tensor {s})", .{ @intFromEnum(qt), tensor.info.name });
            return error.UnsupportedQuantType;
        };
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds,
            tensor.gpu_buffer.handle, tensor.gpu_buffer.size,
            input_buf.handle, input_size,
            output_buf.handle, output_buf.size);
        try self.dmmv.recordDispatch(&self.decode_cmd, qt, ds, M, K, 0, 0, 0);
    }

    /// Dispatch a DMMV with byte offset into stacked weight tensor (for MoE experts).
    fn dispatchDmmvWithOffset(
        self: *InferenceEngine,
        tensor: *const LoadedTensor,
        /// GPU buffer for input buf.
        input_buf: Buffer, input_size: vk.c.VkDeviceSize,
        /// GPU buffer for output buf.
        output_buf: Buffer,
        M: u32, K: u32,
        /// Weight buffer byte offset.
        a_offset: u32,
    ) !void {
        const qt = tensor.info.type_;
        const pip = self.dmmv.pipelineForType(qt) orelse {
            log.err("No DMMV pipeline for quant type {d} (tensor {s})", .{ @intFromEnum(qt), tensor.info.name });
            return error.UnsupportedQuantType;
        };
        const ds = try self.allocDescSet(pip.descriptor_set_layout);
        self.writeDescSet3(ds,
            tensor.gpu_buffer.handle, tensor.gpu_buffer.size,
            input_buf.handle, input_size,
            output_buf.handle, output_buf.size);
        try self.dmmv.recordDispatch(&self.decode_cmd, qt, ds, M, K, a_offset, 0, 0);
    }

    // -----------------------------------------------------------------------
    // CPU-side SSM / delta-net layer
    // -----------------------------------------------------------------------

    /// Run one SSM layer: GPU for large projections, CPU for small state ops.
    fn runSsmLayerCpu(self: *InferenceEngine, _: *DecodeState, layer: u32, layer_idx: usize) !void {
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

        // --- GPU phase 1: Run large projections via DMMV ---
        const wqkv_tensor = self.findLayerTensor(layer, "attn_qkv.weight") orelse return;
        try self.dispatchDmmv(wqkv_tensor, self.norm_buf, hidden_size, self.attn_out_buf, @intCast(conv_channels), hidden_dim);

        const z_tensor = self.findLayerTensor(layer, "attn_gate.weight") orelse return;
        try self.dispatchDmmv(z_tensor, self.norm_buf, hidden_size, self.gate_buf, @intCast(d_inner), hidden_dim);

        const alpha_tensor = self.findLayerTensor(layer, "ssm_alpha.weight") orelse return;
        try self.dispatchDmmv(alpha_tensor, self.norm_buf, hidden_size, self.router_logits_buf, dt_rank, hidden_dim);

        const beta_tensor = self.findLayerTensor(layer, "ssm_beta.weight") orelse return;
        try self.dispatchDmmv(beta_tensor, self.norm_buf, hidden_size, self.down_buf, dt_rank, hidden_dim);
        self.decode_cmd.computeBarrier();

        // --- Readback projection results to CPU via logits_staging ---
        const qkv_bytes = @as(vk.c.VkDeviceSize, conv_channels) * @sizeOf(f32);
        const z_bytes = @as(vk.c.VkDeviceSize, d_inner) * @sizeOf(f32);
        const ab_bytes = @as(vk.c.VkDeviceSize, dt_rank) * @sizeOf(f32);
        {
            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
            };
            vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle,
                vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &barrier, 0, null, 0, null);

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
        const conv_tensor = self.findLayerTensor(layer, "ssm_conv1d.weight") orelse return;
        const conv_data_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + conv_tensor.info.offset);
        // Bug fix #14: Read conv kernel handling f16 storage — direct f32 cast corrupts values
        const conv_kernel_len = conv_channels * d_conv;
        const conv_kernel_buf = try self.allocator.alloc(f32, conv_kernel_len);
        defer self.allocator.free(conv_kernel_buf);
        readMmapFloats(mmap, conv_data_off, conv_tensor.info.type_, conv_kernel_buf);
        if (layer == 0) log.info("SSM tensor types: conv1d={s} dt_bias={s} ssm_a={s} n_group={d} dt_rank={d} d_state={d} head_v={d}", .{
            @tagName(conv_tensor.info.type_),
            if (self.findLayerTensor(layer, "ssm_dt.bias")) |t| @tagName(t.info.type_) else "N/A",
            if (self.findLayerTensor(layer, "ssm_a")) |t| @tagName(t.info.type_) else "N/A",
            n_group, dt_rank, d_state, head_v_dim,
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
            std.mem.copyForwards(f32, conv_state[0..shift], conv_state[conv_channels..shift + conv_channels]);
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
        const dt_bias_tensor = self.findLayerTensor(layer, "ssm_dt.bias");
        const dt_bias_f32 = try self.allocator.alloc(f32, dt_rank);
        defer self.allocator.free(dt_bias_f32);
        if (dt_bias_tensor) |t| {
            const off: usize = @intCast(self.model.gguf_file.tensor_data_offset + t.info.offset);
            readMmapFloats(mmap, off, t.info.type_, dt_bias_f32);
        }

        const ssm_a_tensor = self.findLayerTensor(layer, "ssm_a");
        const ssm_a_f32 = try self.allocator.alloc(f32, dt_rank);
        defer self.allocator.free(ssm_a_f32);
        if (ssm_a_tensor) |t| {
            const off: usize = @intCast(self.model.gguf_file.tensor_data_offset + t.info.offset);
            readMmapFloats(mmap, off, t.info.type_, ssm_a_f32);
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

            // sk = s @ k (per-row dot with k vector)
            // d = beta * (v - sk), then s += outer(k, d) = k[row] * d[col]
            for (0..head_v_dim) |row| {
                var sk: f32 = 0;
                for (0..@min(head_v_dim, k_head.len)) |col| {
                    sk += ssm_state[s_base + row * head_v_dim + col] * k_head[col];
                }
                const d_val = b_val * (v_head[row] - sk);
                // Bug fix #13: Outer product s[row][col] += k[col] * d[row]
                // d_val is d[row] (scalar for this row), update row's slice of state
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
        const norm_tensor = self.findLayerTensor(layer, "ssm_norm.weight");
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
                    t.info.n_dims, t.info.dims[0], t.info.dims[1],
                    t.info.numElements(), d_state, d_inner, head_v_dim, norm_per_head,
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
            const rms = @sqrt(sq / @as(f32, @floatFromInt(head_v_dim)) + 1e-6);
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

        _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.begin();

        {
            const r = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = z_bytes };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.ssm_hidden_staging.handle, self.swiglu_buf.handle, 1, &r);
        }
        self.decode_cmd.transferToComputeBarrier();

        const ssm_out_tensor = self.findLayerTensor(layer, "ssm_out.weight") orelse return;
        try self.dispatchDmmv(ssm_out_tensor, self.swiglu_buf, z_bytes, self.o_proj_buf, hidden_dim, @intCast(d_inner));
        self.decode_cmd.computeBarrier();

        // Residual: hidden_buf += o_proj_buf
        {
            const pip = &(self.elementwise.pipeline_scale_acc orelse return error.ShaderNotLoaded);
            const ds = try self.allocDescSet(pip.descriptor_set_layout);
            self.writeDescSet2(ds, self.hidden_buf.handle, hidden_size, self.o_proj_buf.handle, hidden_size);
            try self.elementwise.recordScaleAcc(&self.decode_cmd, ds, hidden_dim, 1.0);
        }
        self.decode_cmd.computeBarrier();
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
    fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {
        if (prompt_tokens.len == 0) return;

        // Run each prompt token through the full transformer (same as decodeStep)
        // This populates KV cache and SSM state so the first decode token has context.
        for (prompt_tokens) |token_id| {
            try self.decodeStep(state, token_id);
        }

        // Upload last token's embedding
    }

    // -----------------------------------------------------------------------
    // Sampling
    // -----------------------------------------------------------------------

    /// Sample a token from the pre-copied logits staging buffer (greedy argmax).
    /// The logits were already copied to staging during decodeStep — zero alloc here.
    pub fn sampleGreedy(self: *const InferenceEngine) u32 {
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
        const embd_t = findLoadedTensor(self.model, "token_embd.weight") orelse return;
        const embd_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + embd_t.info.offset);
        var cpu_embed_buf: [8192]f32 = undefined;
        const cpu_embed = cpu_embed_buf[0..hidden_dim];
        dequantRow(mmap[embd_off..], bos_token, hidden_dim, embd_t.info.type_, cpu_embed);

        // 2. CPU RMS norm with output_norm.weight
        const norm_t = findLoadedTensor(self.model, "output_norm.weight") orelse return;
        const norm_off: usize = @intCast(self.model.gguf_file.tensor_data_offset + norm_t.info.offset);
        var cpu_nw_buf: [8192]f32 = undefined;
        const cpu_nw = cpu_nw_buf[0..hidden_dim];
        dequantRow(mmap[norm_off..], 0, hidden_dim, norm_t.info.type_, cpu_nw);

        var sum_sq: f64 = 0.0;
        for (cpu_embed) |v| sum_sq += @as(f64, v) * @as(f64, v);
        const rms_inv: f32 = @floatCast(1.0 / @sqrt(sum_sq / @as(f64, @floatFromInt(hidden_dim)) + 1e-6));

        var cpu_normed_buf: [8192]f32 = undefined;
        const cpu_normed = cpu_normed_buf[0..hidden_dim];
        for (0..hidden_dim) |i| cpu_normed[i] = cpu_nw[i] * (cpu_embed[i] * rms_inv);

        // 3. CPU dot products for first 10 logits
        const lm_t = findLoadedTensor(self.model, "output.weight") orelse
            findLoadedTensor(self.model, "token_embd.weight") orelse return;
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

        _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.begin();
        {
            const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = hidden_size };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.embed_staging.handle, self.hidden_buf.handle, 1, &region);
        }
        // Barrier: transfer write → transfer read (for readback)
        {
            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
            };
            vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle,
                vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &barrier, 0, null, 0, null);
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
            cpu_embed[0],  cpu_embed[1],  cpu_embed[2],  cpu_embed[3],
            gpu_e[0],      gpu_e[1],      gpu_e[2],      gpu_e[3],
            embed_max_diff,
        });

        // ── STAGE 2: RMS norm → readback norm_buf ──
        // hidden_buf still has the embedding (only read in stage 1 readback)
        _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.begin();
        {
            const pip = &(self.elementwise.pipeline_rms_norm orelse return);
            const ds = try self.allocDescSet(pip.descriptor_set_layout);
            self.writeDescSet3(ds, self.hidden_buf.handle, hidden_size,
                norm_t.gpu_buffer.handle, norm_t.gpu_buffer.size,
                self.norm_buf.handle, hidden_size);
            try self.elementwise.recordRmsNorm(&self.decode_cmd, ds, hidden_dim, 1, 1e-6);
        }
        // Barrier: shader write → transfer read
        {
            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
            };
            vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle,
                vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &barrier, 0, null, 0, null);
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
        _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
        try self.decode_cmd.reset();
        try self.decode_cmd.begin();
        try self.dispatchDmmv(lm_t, self.norm_buf, hidden_size, self.logits_buf, config.vocab_size, hidden_dim);
        {
            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
            };
            vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle,
                vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &barrier, 0, null, 0, null);
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
            if (gpu[i] > gpu_max) { gpu_max = gpu[i]; gpu_max_idx = @intCast(i); }
        }

        dlog.info("SUMMARY: embed_ok={s} norm_ok={s} GPU_argmax={d}({d:.4}) CPU_logit5={d:.4} GPU_logit5={d:.4}", .{
            if (embed_max_diff < 0.001) "YES" else "NO",
            if (norm_max_diff < 0.01) "YES" else "NO",
            gpu_max_idx, gpu_max,
            cpu_logits[5], gpu[5],
        });

        // ── STAGE 4: Verify DMMV for non-Q8_0 quant types ──
        // norm_buf still has BOS embedding norm from STAGE 2 (STAGE 3 only read it)
        const wqkv_diag = self.findLayerTensor(0, "attn_qkv.weight");
        const gate_exps_diag = self.findLayerTensor(0, "ffn_gate_exps.weight");
        const down_exps_diag = self.findLayerTensor(0, "ffn_down_exps.weight");
        const ssm_out_diag = self.findLayerTensor(0, "ssm_out.weight");
        const attn_q_diag = self.findLayerTensor(3, "attn_q.weight"); // layer 3 = first attn layer
        dlog.info("QUANT: wqkv={s} gate_exps={s} down_exps={s} ssm_out={s} attn_q={s}", .{
            if (wqkv_diag) |t| @tagName(t.info.type_) else "N/A",
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
            _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
            try self.decode_cmd.reset();
            try self.decode_cmd.begin();
            try self.dispatchDmmv(wt, self.norm_buf, hidden_size, self.logits_buf, conv_ch, hidden_dim);
            {
                const bar4 = vk.c.VkMemoryBarrier{
                    .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = null,
                    .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
                };
                vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle,
                    vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 1, &bar4, 0, null, 0, null);
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
                @tagName(wt.info.type_), conv_ch, hidden_dim, wqkv_mdiff,
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
            _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);
            try self.decode_cmd.reset();
            try self.decode_cmd.begin();
            try self.dispatchDmmvWithOffset(gt, self.norm_buf, hidden_size, self.logits_buf, @intCast(inter_d), hidden_dim, 0);
            {
                const bar5 = vk.c.VkMemoryBarrier{
                    .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER, .pNext = null,
                    .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                    .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
                };
                vk.c.vkCmdPipelineBarrier(self.decode_cmd.handle,
                    vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0, 1, &bar5, 0, null, 0, null);
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
                @tagName(gt.info.type_), inter_d, hidden_dim,
                gpu_g[0], gpu_g[1], gpu_g[2],
                cpu_gate_r[0], cpu_gate_r[1], cpu_gate_r[2],
                gate_mdiff,
                if (gate_mdiff < 0.1) @as([]const u8, "YES") else @as([]const u8, "NO"),
            });
        }

        dlog.info("=== END DIAG ===", .{});
    }

    // -----------------------------------------------------------------------
    // Teardown
    // -----------------------------------------------------------------------

    /// Release GPU buffers, graphs, command objects, and dispatch helpers owned by the engine.
    pub fn deinit(self: *InferenceEngine) void {
        vk.c.vkDestroyDescriptorPool(self.instance.device, self.shared_pool, null);
        // SSM state
        for (self.ssm_conv_states) |s| if (s.len > 0) self.allocator.free(s);
        for (self.ssm_states) |s| if (s.len > 0) self.allocator.free(s);
        self.allocator.free(self.ssm_conv_states);
        self.allocator.free(self.ssm_states);
        self.ssm_hidden_staging.deinit();
        // KV cache + page table
        self.page_table_buf.deinit();
        for (self.kv_k_cache) |*b| b.deinit();
        for (self.kv_v_cache) |*b| b.deinit();
        self.allocator.free(self.kv_k_cache);
        self.allocator.free(self.kv_v_cache);
        // Layer intermediates
        self.router_staging.deinit();
        self.router_logits_buf.deinit();
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
        self.logits_staging.deinit();
        self.logits_buf.deinit();
        self.norm_buf.deinit();
        self.residual_buf.deinit();
        self.hidden_buf.deinit();
        self.decode_graph.deinit();
        self.attention.deinit();
        self.elementwise.deinit();
        self.dmmv.deinit();
        self.decode_cmd.deinit(&self.cmd_pool);
        self.cmd_pool.deinit();
        self.* = undefined;
    }
};

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

    log.info("Generating: {d} prompt tokens, max {d} output tokens", .{
        prompt_tokens.len, max_tokens,
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

    log.info("Prefill complete: {d} tokens in {d:.1} ms ({d:.2} tok/s)", .{
        prompt_tokens.len, @as(f64, @floatFromInt(prefill_ns)) / 1_000_000.0, prefill_tok_per_sec,
    });

    // Decode: generate tokens one at a time
    var generated: u32 = 0;
    const decode_start = std.time.nanoTimestamp();
    while (generated < max_tokens) : (generated += 1) {
        const tok_start = std.time.nanoTimestamp();

        // Feed the last token (from prompt or generation) as input
        const input_token = if (state.generated_tokens.items.len > 0)
            state.generated_tokens.items[state.generated_tokens.items.len - 1]
        else if (prompt_tokens.len > 0)
            prompt_tokens[prompt_tokens.len - 1]
        else
            0;

        try engine.decodeStep(&state, input_token);
        const token = engine.sampleGreedy();
        try state.generated_tokens.append(allocator, token);

        const tok_end = std.time.nanoTimestamp();
        const tok_ms = @as(f64, @floatFromInt(@as(u64, @intCast(tok_end - tok_start)))) / 1_000_000.0;
        log.info("decode[{d}]: token={d} pos={d} ({d:.1} ms)", .{
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

        // Bandwidth metrics: compute bytes read per decode token
        const config = &engine.model.config;
        const lm_tensor = findLoadedTensor(engine.model, "output.weight") orelse
            findLoadedTensor(engine.model, "token_embd.weight");
        if (lm_tensor) |t| {
            const lm_quant = t.info.type_;
            const bpb = lm_quant.bytesPerBlock();
            const bs = lm_quant.blockSize();
            // LM head: vocab_size rows x hidden_dim cols
            const lm_blocks_per_row = @as(u64, config.hidden_dim) / @as(u64, bs);
            const lm_bytes: u64 = @as(u64, config.vocab_size) * lm_blocks_per_row * @as(u64, bpb);
            // Norm weights: hidden_dim * 4 bytes (f32)
            const norm_bytes: u64 = @as(u64, config.hidden_dim) * 4;
            // Input/output vectors: hidden_dim * 4 + vocab_size * 4
            const vec_bytes: u64 = (@as(u64, config.hidden_dim) + @as(u64, config.vocab_size)) * 4;
            // Logits readback: vocab_size * 4
            const readback_bytes: u64 = @as(u64, config.vocab_size) * 4;
            const total_bytes_per_tok = lm_bytes + norm_bytes + vec_bytes + readback_bytes;
            const total_bytes_all = total_bytes_per_tok * @as(u64, @intCast(decode_tokens));

            const decode_secs = @as(f64, @floatFromInt(decode_ns)) / 1_000_000_000.0;
            const eff_bw_gbs = @as(f64, @floatFromInt(total_bytes_all)) / decode_secs / 1_000_000_000.0;
            const theo_bw_gbs: f64 = @floatFromInt(engine.gpu_config.bandwidth_gbps);
            const utilization = if (theo_bw_gbs > 0) eff_bw_gbs / theo_bw_gbs * 100.0 else 0.0;

            log.info("Bandwidth: {d:.1} GB/s effective, {d:.0} GB/s theoretical ({d:.1}% utilization)", .{
                eff_bw_gbs, theo_bw_gbs, utilization,
            });
            log.info("Per-token: {d:.2} MB read ({s} LM head {d}x{d})", .{
                @as(f64, @floatFromInt(total_bytes_per_tok)) / 1_000_000.0,
                @tagName(lm_quant),
                config.vocab_size,
                config.hidden_dim,
            });
        }
    } else {
        log.info("Generated {d} tokens", .{decode_tokens});
    }

    // Print per-layer diagnostic summary (stored during BOS processing)
    if (engine.diag_summary_len > 0) {
        log.info("LOGIT5_SUMMARY: {s}", .{engine.diag_summary_buf[0..engine.diag_summary_len]});
    }

    // Run diagnostic AFTER generation so output appears at the end (not truncated)
    if (prompt_tokens.len > 0) {
        engine.diagEmbedToLogits(prompt_tokens[0]) catch |err| {
            log.warn("Diagnostic failed: {s}", .{@errorName(err)});
        };
    }

    return try allocator.dupe(u32, state.generated_tokens.items);
}
