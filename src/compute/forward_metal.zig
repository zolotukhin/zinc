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

/// Push constants for DMMV dispatch (matches GLSL layout).
const DmmvPush = extern struct {
    M: u32, // rows
    K: u32, // cols
    a_offset: u32,
    x_offset: u32,
    y_offset: u32,
};

/// Push constants for RMS norm.
const RmsNormPush = extern struct {
    hidden_dim: u32,
    n_tokens: u32,
    eps: f32,
};

/// Push constants for elementwise ops.
const ElementwisePush = extern struct {
    n_elements: u32,
};

/// Push constants for RoPE.
const RopePush = extern struct {
    stride: u32,
    rope_dim: u32,
    n_heads: u32,
    position: u32,
    freq_base: f32,
};

/// Push constants for scale_accumulate.
const ScaleAccPush = extern struct {
    n_elements: u32,
    scale: f32,
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
    logits_buf: MetalBuffer,
    embed_staging: MetalBuffer,

    // KV cache (per layer)
    kv_k_cache: []MetalBuffer,
    kv_v_cache: []MetalBuffer,

    // Compute pipelines
    dmmv_q4k_pipe: MetalPipeline,
    rms_norm_pipe: MetalPipeline,
    swiglu_pipe: MetalPipeline,
    rope_pipe: MetalPipeline,
    flash_attn_pipe: MetalPipeline,
    vadd_pipe: MetalPipeline,
    scale_acc_pipe: MetalPipeline,
    sigmoid_mul_pipe: MetalPipeline,
    deinterleave_pipe: MetalPipeline,

    // Decode state
    position: u32,

    pub fn init(
        model: *const metal_loader.Model,
        device: *const metal_device.MetalDevice,
        allocator: std.mem.Allocator,
    ) !InferenceEngine {
        const cfg = model.config;
        const ctx = device.ctx;

        // Buffer sizes
        const hidden_size = cfg.hidden_dim * @sizeOf(f32);
        const head_total = cfg.n_heads * cfg.head_dim * @sizeOf(f32);
        const kv_total = cfg.n_kv_heads * cfg.head_dim * @sizeOf(f32);
        const intermediate_size = cfg.intermediate_dim * @sizeOf(f32);
        const vocab_size = cfg.vocab_size * @sizeOf(f32);
        const kv_cache_size: usize = @as(usize, 4096) * cfg.n_kv_heads * cfg.head_dim * @sizeOf(f32);

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
        self.attn_out_buf = try metal_buffer.createBuffer(ctx, head_total);
        self.gate_buf = try metal_buffer.createBuffer(ctx, intermediate_size);
        self.up_buf = try metal_buffer.createBuffer(ctx, intermediate_size);
        self.swiglu_buf = try metal_buffer.createBuffer(ctx, intermediate_size);
        self.down_buf = try metal_buffer.createBuffer(ctx, hidden_size);
        self.moe_out_buf = try metal_buffer.createBuffer(ctx, hidden_size);
        self.router_logits_buf = try metal_buffer.createBuffer(ctx, @max(cfg.n_experts, 1) * @sizeOf(f32));
        self.logits_buf = try metal_buffer.createBuffer(ctx, vocab_size);
        self.embed_staging = try metal_buffer.createBuffer(ctx, hidden_size);

        // Allocate KV cache per layer
        self.kv_k_cache = try allocator.alloc(MetalBuffer, cfg.n_layers);
        self.kv_v_cache = try allocator.alloc(MetalBuffer, cfg.n_layers);
        for (0..cfg.n_layers) |i| {
            self.kv_k_cache[i] = try metal_buffer.createBuffer(ctx, kv_cache_size);
            self.kv_v_cache[i] = try metal_buffer.createBuffer(ctx, kv_cache_size);
        }

        // Load compute pipelines from cross-compiled MSL
        self.dmmv_q4k_pipe = try loadShaderPipeline(ctx, "dmmv_q4k");
        self.rms_norm_pipe = try loadShaderPipeline(ctx, "rms_norm_mul");
        self.swiglu_pipe = try loadShaderPipeline(ctx, "swiglu");
        self.rope_pipe = try loadShaderPipeline(ctx, "rope_fused");
        self.flash_attn_pipe = try loadShaderPipeline(ctx, "flash_attn");
        self.vadd_pipe = try loadShaderPipeline(ctx, "vadd");
        self.scale_acc_pipe = try loadShaderPipeline(ctx, "scale_accumulate");
        self.sigmoid_mul_pipe = try loadShaderPipeline(ctx, "sigmoid_mul");
        self.deinterleave_pipe = try loadShaderPipeline(ctx, "deinterleave");

        log.info("Metal inference engine initialized: {d} layers, {d}x{d} heads, dim={d}", .{
            cfg.n_layers, cfg.n_heads, cfg.head_dim, cfg.hidden_dim,
        });

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
        metal_buffer.freeBuffer(&self.logits_buf);
        metal_buffer.freeBuffer(&self.embed_staging);

        for (0..self.config.n_layers) |i| {
            metal_buffer.freeBuffer(&self.kv_k_cache[i]);
            metal_buffer.freeBuffer(&self.kv_v_cache[i]);
        }
        self.allocator.free(self.kv_k_cache);
        self.allocator.free(self.kv_v_cache);
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
};

/// Load an MSL shader from src/shaders/metal/ and compile it at runtime.
fn loadShaderPipeline(ctx: ?*shim.MetalCtx, name: []const u8) !MetalPipeline {
    // Read the .metal file from the shader directory
    var path_buf: [256]u8 = undefined;
    const path = std.fmt.bufPrint(&path_buf, "src/shaders/metal/{s}.metal", .{name}) catch return error.PathTooLong;

    const file = std.fs.cwd().openFile(path, .{}) catch |err| {
        log.err("Failed to open shader '{s}': {s}", .{ name, @errorName(err) });
        return error.ShaderNotFound;
    };
    defer file.close();

    const stat = try file.stat();
    if (stat.size > 1024 * 1024) return error.ShaderTooLarge; // 1MB limit

    var source_buf: [1024 * 1024]u8 = undefined;
    const bytes_read = try file.readAll(&source_buf);
    source_buf[bytes_read] = 0; // null-terminate for C string

    var fn_buf: [128]u8 = undefined;
    // SPIRV-Cross names the entry point "main0"
    const fn_name = std.fmt.bufPrintZ(&fn_buf, "main0", .{}) catch return error.NameTooLong;

    return metal_pipeline.createPipeline(ctx, @ptrCast(&source_buf), fn_name);
}

// ---------------------------------------------------------------------------
// CPU-side dequantization helpers (shared logic with forward.zig)
// ---------------------------------------------------------------------------

const GGMLType = gguf.GGMLType;

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
            log.warn("Unsupported embedding quant type {d}, using zeros", .{@intFromEnum(quant_type)});
            @memset(output, 0);
        },
    }
}

/// Read tensor elements from mmap into an f32 buffer, handling f32 and f16 storage.
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

/// Softmax + top-k selection on CPU for MoE routing.
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
    if (sum > 0) {
        for (0..n) |i| probs[i] /= sum;
    }

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
    if (wsum > 0) {
        for (0..k) |i| out_weights[i] /= wsum;
    }
}

/// Compute byte size of one expert slice in a stacked weight tensor.
fn expertSliceBytes(quant_type: GGMLType, rows: u32, cols: u32) u32 {
    const bs = quant_type.blockSize();
    const bpb = quant_type.bytesPerBlock();
    if (bs == 0 or bpb == 0) return rows * cols * 4;
    const blocks_per_row = cols / bs;
    return rows * blocks_per_row * bpb;
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

/// Generate tokens using the Metal inference engine.
pub fn generate(
    engine: *InferenceEngine,
    prompt_tokens: []const u32,
    max_tokens: u32,
    eos_id: u32,
    allocator: std.mem.Allocator,
) ![]u32 {
    const cfg = engine.config;
    const mmap = engine.model.mmap_data orelse return error.NoMmapData;

    var output: std.ArrayList(u32) = .{};
    errdefer output.deinit(allocator);

    // Embedding dequant buffer (CPU-side for now)
    const embed_buf = try allocator.alloc(f32, cfg.hidden_dim);
    defer allocator.free(embed_buf);

    // Find embedding tensor
    const embed_tensor = engine.model.gguf_file.findTensor("token_embd.weight") orelse return error.MissingTensor;
    const embed_data_offset = engine.model.gguf_file.tensor_data_offset + embed_tensor.offset;
    const embed_raw = mmap[embed_data_offset..];

    // Process prompt tokens (simplified single-token prefill for now)
    for (prompt_tokens) |token_id| {
        // Dequant embedding for this token
        dequantRow(embed_raw, token_id, cfg.hidden_dim, embed_tensor.type_, embed_buf);

        // Upload to GPU hidden buffer
        const dst: [*]f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));
        @memcpy(dst[0..cfg.hidden_dim], embed_buf);

        engine.position += 1;
    }

    // Find output weight tensor (LM head)
    const output_tensor = engine.model.gguf_file.findTensor("output.weight") orelse return error.MissingTensor;
    const output_data_offset = engine.model.gguf_file.tensor_data_offset + output_tensor.offset;

    // Find the output weight mmap'd Metal buffer
    var output_weight_buf: ?*const MetalBuffer = null;
    for (engine.model.tensors.items) |*t| {
        if (std.mem.eql(u8, t.info.name, "output.weight")) {
            output_weight_buf = &t.gpu_buffer;
            break;
        }
    }
    _ = output_data_offset;

    // Decode loop
    var tokens_generated: u32 = 0;
    while (tokens_generated < max_tokens) {
        // TODO: Full per-layer dispatch (attention, SSM, MoE FFN)
        // Current stub: skip 40-layer transform, directly project hidden→logits
        // This produces wrong tokens but proves the DMMV pipeline works

        // LM head: output.weight × hidden_buf → logits_buf
        // For now, do this on CPU since the DMMV shader binding isn't wired yet
        {
            const output_raw = mmap[engine.model.gguf_file.tensor_data_offset + output_tensor.offset ..];
            const hidden_ptr: [*]const f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));
            const logits_ptr: [*]f32 = @ptrCast(@alignCast(engine.logits_buf.cpu_ptr.?));

            // CPU matmul: logits[v] = sum_d(output_weight[v][d] * hidden[d])
            // This is slow but proves correctness. GPU DMMV will replace this.
            var row_buf: [4096]f32 = undefined;
            for (0..cfg.vocab_size) |v| {
                dequantRow(output_raw, @intCast(v), cfg.hidden_dim, output_tensor.type_, row_buf[0..cfg.hidden_dim]);
                var dot: f32 = 0;
                for (0..cfg.hidden_dim) |d| {
                    dot += row_buf[d] * hidden_ptr[d];
                }
                logits_ptr[v] = dot;
            }
        }

        // Sample
        const next_token = engine.sampleGreedy();
        if (next_token == eos_id) break;

        try output.append(allocator, next_token);
        tokens_generated += 1;

        // Dequant embedding for next token
        dequantRow(embed_raw, next_token, cfg.hidden_dim, embed_tensor.type_, embed_buf);
        const dst: [*]f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));
        @memcpy(dst[0..cfg.hidden_dim], embed_buf);

        engine.position += 1;
    }

    log.info("Generated {d} tokens", .{tokens_generated});

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

    // Top-3 by logit value: index 4 (4.0), index 1 (3.0), index 7 (2.5)
    try std.testing.expectEqual(@as(u32, 4), ids[0]);
    try std.testing.expectEqual(@as(u32, 1), ids[1]);
    try std.testing.expectEqual(@as(u32, 7), ids[2]);

    // Weights should sum to ~1.0
    const wsum = weights[0] + weights[1] + weights[2];
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), wsum, 0.01);
}

test "topKSoftmax with uniform logits returns equal weights" {
    var ids: [3]u32 = undefined;
    var weights: [3]f32 = undefined;
    const logits = [_]f32{ 1.0, 1.0, 1.0, 1.0 };
    topKSoftmax(&logits, 3, &ids, &weights);

    // Uniform logits → each selected expert gets ~1/3
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), weights[0], 0.02);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0 / 3.0), weights[1], 0.02);
}

test "expertSliceBytes Q4_K" {
    const bytes = expertSliceBytes(.q4_k, 1024, 2048);
    // Q4_K: 256 elems/block, 144 bytes/block → 2048/256 = 8 blocks/row → 1024 * 8 * 144
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
