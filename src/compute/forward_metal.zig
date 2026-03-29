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

/// Push constants for SwiGLU dispatch (matches SPIRV-Cross layout: buffer(0)).
const SwiGLUPush = extern struct {
    n: u32, // number of elements
};

/// Push constants for scale_accumulate dispatch (matches SPIRV-Cross layout: buffer(0)).
const ScaleAccPush = extern struct {
    n: u32, // number of elements
    scale_bits: u32, // float reinterpreted as uint32 (SPIRV-Cross convention)
};

/// Push constants for RMS norm dispatch (matches rms_norm_mul.metal: buffer(0)).
const RmsNormPush = extern struct {
    n: u32, // elements per group
    eps: f32, // epsilon
};

const GGMLType = gguf.GGMLType;

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

    // DMMV compute pipelines (one per quant type)
    dmmv_q4k_pipe: MetalPipeline,
    dmmv_q5k_pipe: MetalPipeline,
    dmmv_q6k_pipe: MetalPipeline,
    dmmv_q8_0_pipe: MetalPipeline,
    dmmv_f16_pipe: MetalPipeline,
    dmmv_f32_pipe: MetalPipeline,

    // Elementwise compute pipelines (for batched GPU dispatch)
    swiglu_pipe: MetalPipeline,
    scale_acc_pipe: MetalPipeline,
    rms_norm_pipe: MetalPipeline,

    // Preloaded norm weight buffers (f32, GPU-accessible via UMA)
    attn_norm_bufs: []MetalBuffer,
    ffn_norm_bufs: []MetalBuffer,
    final_norm_gpu: MetalBuffer,

    // SSM state (CPU-side, allocated only if model has SSM layers)
    ssm_conv_states: ?[][]f32,
    ssm_states: ?[][]f32,

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
        const router_size: usize = @max(@as(usize, cfg.n_experts), @as(usize, if (cfg.ssm_dt_rank > 0) cfg.ssm_dt_rank else 1)) * @sizeOf(f32);

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
        self.logits_buf = try metal_buffer.createBuffer(ctx, vocab_size);
        self.embed_staging = try metal_buffer.createBuffer(ctx, hidden_size);

        // Allocate KV cache per layer
        self.kv_k_cache = try allocator.alloc(MetalBuffer, cfg.n_layers);
        self.kv_v_cache = try allocator.alloc(MetalBuffer, cfg.n_layers);
        for (0..cfg.n_layers) |i| {
            self.kv_k_cache[i] = try metal_buffer.createBuffer(ctx, kv_cache_size);
            self.kv_v_cache[i] = try metal_buffer.createBuffer(ctx, kv_cache_size);
        }

        // Load DMMV compute pipelines for all quant types
        self.dmmv_q4k_pipe = try loadShaderPipeline(ctx, "dmmv_q4k");
        self.dmmv_q5k_pipe = try loadShaderPipeline(ctx, "dmmv_q5k");
        self.dmmv_q6k_pipe = try loadShaderPipeline(ctx, "dmmv_q6k");
        self.dmmv_q8_0_pipe = try loadShaderPipeline(ctx, "dmmv_q8_0");
        self.dmmv_f16_pipe = try loadShaderPipeline(ctx, "dmmv_f16");
        self.dmmv_f32_pipe = try loadShaderPipeline(ctx, "dmmv_f32");

        // Elementwise pipelines for batched GPU dispatch
        self.swiglu_pipe = try loadShaderPipeline(ctx, "swiglu");
        self.scale_acc_pipe = try loadShaderPipeline(ctx, "scale_accumulate");
        self.rms_norm_pipe = try loadShaderPipeline(ctx, "rms_norm_mul");

        // Preload norm weights into f32 Metal buffers (eliminates per-token alloc + mmap dequant)
        self.attn_norm_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
        self.ffn_norm_bufs = try allocator.alloc(MetalBuffer, cfg.n_layers);
        for (0..cfg.n_layers) |i| {
            const layer: u32 = @intCast(i);
            const an = findLayerTensor(model, layer, "attn_norm.weight") orelse return error.MissingTensor;
            self.attn_norm_bufs[i] = try preloadNormWeights(ctx, model, an, cfg.hidden_dim);
            const fn_t = findLayerTensor(model, layer, "post_attention_norm.weight") orelse
                findLayerTensor(model, layer, "ffn_norm.weight") orelse return error.MissingTensor;
            self.ffn_norm_bufs[i] = try preloadNormWeights(ctx, model, fn_t, cfg.hidden_dim);
        }
        const final_t = findTensorByName(model, "output_norm.weight") orelse return error.MissingTensor;
        self.final_norm_gpu = try preloadNormWeights(ctx, model, final_t, cfg.hidden_dim);

        // SSM state allocation
        if (d_inner > 0 and cfg.ssm_d_conv > 0) {
            const d_conv_1 = cfg.ssm_d_conv - 1;
            const head_v_dim = d_inner / @max(cfg.ssm_dt_rank, 1);
            self.ssm_conv_states = try allocator.alloc([]f32, cfg.n_layers);
            self.ssm_states = try allocator.alloc([]f32, cfg.n_layers);
            for (0..cfg.n_layers) |i| {
                self.ssm_conv_states.?[i] = try allocator.alloc(f32, @as(usize, d_conv_1) * conv_channels);
                @memset(self.ssm_conv_states.?[i], 0);
                self.ssm_states.?[i] = try allocator.alloc(f32, @as(usize, cfg.ssm_dt_rank) * head_v_dim * head_v_dim);
                @memset(self.ssm_states.?[i], 0);
            }
        } else {
            self.ssm_conv_states = null;
            self.ssm_states = null;
        }

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

        metal_pipeline.freePipeline(&self.dmmv_q4k_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q5k_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q6k_pipe);
        metal_pipeline.freePipeline(&self.dmmv_q8_0_pipe);
        metal_pipeline.freePipeline(&self.dmmv_f16_pipe);
        metal_pipeline.freePipeline(&self.dmmv_f32_pipe);
        metal_pipeline.freePipeline(&self.swiglu_pipe);
        metal_pipeline.freePipeline(&self.scale_acc_pipe);
        metal_pipeline.freePipeline(&self.rms_norm_pipe);

        for (0..self.config.n_layers) |i| {
            metal_buffer.freeBuffer(&self.attn_norm_bufs[i]);
            metal_buffer.freeBuffer(&self.ffn_norm_bufs[i]);
        }
        self.allocator.free(self.attn_norm_bufs);
        self.allocator.free(self.ffn_norm_bufs);
        metal_buffer.freeBuffer(&self.final_norm_gpu);

        if (self.ssm_conv_states) |cs| {
            for (cs) |s| self.allocator.free(s);
            self.allocator.free(cs);
        }
        if (self.ssm_states) |ss| {
            for (ss) |s| self.allocator.free(s);
            self.allocator.free(ss);
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

    /// Get the DMMV pipeline, push constant buffer index, and rows-per-workgroup for a quant type.
    /// Q4_K/Q5_K/Q6_K/F32: each thread handles 1 row (64 rows per workgroup).
    /// Q8_0/F16: each workgroup handles 2 rows (64 threads cooperate via simd_sum).
    fn dmmvPipelineForType(self: *InferenceEngine, qt: GGMLType) ?struct { pipe: *const MetalPipeline, push_idx: u32, rows_per_wg: u32 } {
        return switch (qt) {
            .q4_k => .{ .pipe = &self.dmmv_q4k_pipe, .push_idx = 1, .rows_per_wg = 64 },
            .q5_k => .{ .pipe = &self.dmmv_q5k_pipe, .push_idx = 0, .rows_per_wg = 64 },
            .q6_k => .{ .pipe = &self.dmmv_q6k_pipe, .push_idx = 0, .rows_per_wg = 64 },
            .q8_0 => .{ .pipe = &self.dmmv_q8_0_pipe, .push_idx = 0, .rows_per_wg = 2 },
            .f16 => .{ .pipe = &self.dmmv_f16_pipe, .push_idx = 0, .rows_per_wg = 2 },
            .f32 => .{ .pipe = &self.dmmv_f32_pipe, .push_idx = 0, .rows_per_wg = 64 },
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
    const pip = engine.dmmvPipelineForType(tensor.info.type_) orelse {
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
    cmd.dispatchV2(pip.pipe, .{ wgs, 1, 1 }, .{ 64, 1, 1 }, &bufs, &push, @sizeOf(DmmvPush), pip.push_idx);
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

/// L2 normalize a vector in-place.
fn cpuL2Normalize(data: []f32) void {
    var sq: f32 = 0;
    for (data) |v| sq += v * v;
    const inv = if (sq > 0) 1.0 / @sqrt(sq) else 1.0;
    for (data) |*v| v.* *= inv;
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

// ---------------------------------------------------------------------------
// Decode step — runs all layers + final norm + LM head
// ---------------------------------------------------------------------------

fn decodeStep(engine: *InferenceEngine) !void {
    const cfg = engine.config;
    const hidden_dim = cfg.hidden_dim;
    const q_dim: u32 = cfg.n_heads * cfg.head_dim;
    const kv_dim: u32 = cfg.n_kv_heads * cfg.head_dim;
    const is_moe = cfg.n_experts > 0;
    const inter_dim: u32 = if (cfg.intermediate_dim > 0) cfg.intermediate_dim else hidden_dim * 4;
    const shexp_inter_dim: u32 = if (cfg.shared_expert_intermediate_dim > 0) cfg.shared_expert_intermediate_dim else inter_dim;
    const full_attn_interval: u32 = if (cfg.full_attn_interval > 0) cfg.full_attn_interval else 1;
    const mmap = engine.model.mmap_data orelse return error.NoMmapData;
    const data_base = engine.model.gguf_file.tensor_data_offset;

    // SSM constants (needed for GPU batch 1 dispatch sizing)
    const d_inner: u32 = cfg.ssm_d_inner;
    const d_state: u32 = cfg.ssm_d_state;
    const n_group: u32 = cfg.ssm_n_group;
    const dt_rank: u32 = cfg.ssm_dt_rank;
    const conv_channels: u32 = if (d_inner > 0) d_inner + 2 * n_group * d_state else 0;

    for (0..cfg.n_layers) |layer_idx| {
        const layer: u32 = @intCast(layer_idx);
        const is_full_attn = ((layer + 1) % full_attn_interval == 0);

        // ===== GPU BATCH 1: attn_norm → projections =====
        // Batches RMS norm with following DMMVs — eliminates CPU norm + alloc.
        {
            var cmd = try metal_command.beginCommand(engine.device.ctx);
            dispatchRmsNormOnCmd(engine, &cmd, &engine.hidden_buf, &engine.norm_buf, &engine.attn_norm_bufs[layer_idx], hidden_dim, 1);
            cmd.barrier();

            if (is_full_attn) {
                const q_tensor = findLayerTensor(engine.model, layer, "attn_q.weight") orelse return error.MissingTensor;
                const k_tensor = findLayerTensor(engine.model, layer, "attn_k.weight") orelse return error.MissingTensor;
                const v_tensor = findLayerTensor(engine.model, layer, "attn_v.weight") orelse return error.MissingTensor;
                const q_full_dim = q_dim * 2;
                dispatchDmmvOnCmd(engine, &cmd, q_tensor, &engine.norm_buf, &engine.attn_out_buf, q_full_dim, hidden_dim, 0);
                dispatchDmmvOnCmd(engine, &cmd, k_tensor, &engine.norm_buf, &engine.k_buf, kv_dim, hidden_dim, 0);
                dispatchDmmvOnCmd(engine, &cmd, v_tensor, &engine.norm_buf, &engine.v_buf, kv_dim, hidden_dim, 0);
            } else {
                const wqkv_t = findLayerTensor(engine.model, layer, "attn_qkv.weight") orelse return error.MissingTensor;
                const z_t = findLayerTensor(engine.model, layer, "attn_gate.weight") orelse return error.MissingTensor;
                const alpha_t = findLayerTensor(engine.model, layer, "ssm_alpha.weight") orelse return error.MissingTensor;
                const beta_t = findLayerTensor(engine.model, layer, "ssm_beta.weight") orelse return error.MissingTensor;
                dispatchDmmvOnCmd(engine, &cmd, wqkv_t, &engine.norm_buf, &engine.attn_out_buf, conv_channels, hidden_dim, 0);
                dispatchDmmvOnCmd(engine, &cmd, z_t, &engine.norm_buf, &engine.gate_buf, d_inner, hidden_dim, 0);
                dispatchDmmvOnCmd(engine, &cmd, alpha_t, &engine.norm_buf, &engine.router_logits_buf, dt_rank, hidden_dim, 0);
                dispatchDmmvOnCmd(engine, &cmd, beta_t, &engine.norm_buf, &engine.down_buf, dt_rank, hidden_dim, 0);
            }
            cmd.commitAndWait();
        }

        // ===== CPU WORK: attention or SSM =====
        if (is_full_attn) {
            // Deinterleave Q+gate → q_buf, gate_buf (CPU)
            const attn_out_ptr: [*]const f32 = @ptrCast(@alignCast(engine.attn_out_buf.cpu_ptr.?));
            const q_ptr: [*]f32 = @ptrCast(@alignCast(engine.q_buf.cpu_ptr.?));
            const gate_ptr: [*]f32 = @ptrCast(@alignCast(engine.gate_buf.cpu_ptr.?));
            for (0..cfg.n_heads) |h| {
                const src_off = h * cfg.head_dim * 2;
                const dst_off = h * cfg.head_dim;
                for (0..cfg.head_dim) |d| {
                    q_ptr[dst_off + d] = attn_out_ptr[src_off + d];
                    gate_ptr[dst_off + d] = attn_out_ptr[src_off + cfg.head_dim + d];
                }
            }

            // Per-head Q/K norm (CPU — small data, not worth GPU dispatch overhead)
            const q_norm_t = findLayerTensor(engine.model, layer, "attn_q_norm.weight");
            if (q_norm_t) |qn| {
                const qn_off: usize = @intCast(data_base + qn.info.offset);
                const qn_w = try engine.allocator.alloc(f32, cfg.head_dim);
                defer engine.allocator.free(qn_w);
                readMmapFloats(mmap, qn_off, qn.info.type_, qn_w);
                cpuPerHeadRmsNormMul(q_ptr, qn_w, cfg.head_dim, cfg.n_heads, 1e-6);
            }
            const k_norm_t = findLayerTensor(engine.model, layer, "attn_k_norm.weight");
            const k_ptr: [*]f32 = @ptrCast(@alignCast(engine.k_buf.cpu_ptr.?));
            if (k_norm_t) |kn| {
                const kn_off: usize = @intCast(data_base + kn.info.offset);
                const kn_w = try engine.allocator.alloc(f32, cfg.head_dim);
                defer engine.allocator.free(kn_w);
                readMmapFloats(mmap, kn_off, kn.info.type_, kn_w);
                cpuPerHeadRmsNormMul(k_ptr, kn_w, cfg.head_dim, cfg.n_kv_heads, 1e-6);
            }

            // RoPE (CPU)
            const rope_freq = cfg.rope_freq_base;
            const rope_dim: u32 = if (cfg.rope_dim > 0) cfg.rope_dim else cfg.head_dim;
            cpuRope(q_ptr, cfg.head_dim, rope_dim, cfg.n_heads, engine.position, rope_freq);
            cpuRope(k_ptr, cfg.head_dim, rope_dim, cfg.n_kv_heads, engine.position, rope_freq);

            // KV cache write (CPU, via UMA)
            const kc_ptr: [*]f32 = @ptrCast(@alignCast(engine.kv_k_cache[layer_idx].cpu_ptr.?));
            const vc_ptr: [*]f32 = @ptrCast(@alignCast(engine.kv_v_cache[layer_idx].cpu_ptr.?));
            const v_ptr: [*]const f32 = @ptrCast(@alignCast(engine.v_buf.cpu_ptr.?));
            const kv_offset: usize = @as(usize, engine.position) * kv_dim;
            for (0..kv_dim) |d| {
                kc_ptr[kv_offset + d] = k_ptr[d];
                vc_ptr[kv_offset + d] = v_ptr[d];
            }

            // Attention (CPU)
            const attn_out_w: [*]f32 = @ptrCast(@alignCast(engine.attn_out_buf.cpu_ptr.?));
            cpuAttention(q_ptr, kc_ptr, vc_ptr, attn_out_w, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads, engine.position + 1);

            // Sigmoid gate: attn_out *= sigmoid(gate) (CPU)
            for (0..q_dim) |i| {
                attn_out_w[i] *= 1.0 / (1.0 + @exp(-gate_ptr[i]));
            }
        } else {
            // SSM CPU core: conv1d, state update, gated norm → swiglu_buf
            try runSsmCpuCore(engine, layer, layer_idx, mmap, data_base);
        }

        // ===== GPU BATCH 2: output proj → residual → ffn_norm → router =====
        // Batches 4 operations that were previously 2 separate commits + CPU ops.
        {
            var cmd = try metal_command.beginCommand(engine.device.ctx);

            // Output / SSM-out projection DMMV
            if (is_full_attn) {
                const o_tensor = findLayerTensor(engine.model, layer, "attn_output.weight") orelse return error.MissingTensor;
                dispatchDmmvOnCmd(engine, &cmd, o_tensor, &engine.attn_out_buf, &engine.down_buf, hidden_dim, q_dim, 0);
            } else {
                const ssm_out_t = findLayerTensor(engine.model, layer, "ssm_out.weight") orelse return error.MissingTensor;
                dispatchDmmvOnCmd(engine, &cmd, ssm_out_t, &engine.swiglu_buf, &engine.down_buf, hidden_dim, d_inner, 0);
            }
            cmd.barrier();

            // GPU residual: hidden += down
            {
                const res_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(@as(f32, 1.0))) };
                const res_bufs = [_]*const MetalBuffer{ &engine.hidden_buf, &engine.down_buf };
                cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &res_bufs, &res_push, @sizeOf(ScaleAccPush), 0);
            }
            cmd.barrier();

            // GPU FFN norm: hidden → norm
            dispatchRmsNormOnCmd(engine, &cmd, &engine.hidden_buf, &engine.norm_buf, &engine.ffn_norm_bufs[layer_idx], hidden_dim, 1);
            cmd.barrier();

            // Router DMMV (MoE only — norm_buf → router_logits_buf)
            if (is_moe) {
                const router_t = findLayerTensor(engine.model, layer, "ffn_gate_inp.weight") orelse return error.MissingTensor;
                dispatchDmmvOnCmd(engine, &cmd, router_t, &engine.norm_buf, &engine.router_logits_buf, cfg.n_experts, hidden_dim, 0);
            }
            cmd.commitAndWait();
        }

        // ===== MoE / Dense FFN =====
        if (is_moe) {
            // CPU topK softmax
            const router_ptr: [*]const f32 = @ptrCast(@alignCast(engine.router_logits_buf.cpu_ptr.?));
            var expert_ids: [16]u32 = undefined;
            var expert_weights: [16]f32 = undefined;
            topKSoftmax(router_ptr[0..cfg.n_experts], cfg.n_experts_used, expert_ids[0..cfg.n_experts_used], expert_weights[0..cfg.n_experts_used]);

            // Zero moe_out_buf
            const moe_ptr: [*]f32 = @ptrCast(@alignCast(engine.moe_out_buf.cpu_ptr.?));
            @memset(moe_ptr[0..hidden_dim], 0);

            // Expert dispatch — all experts batched into ONE GPU command buffer
            const gate_exps = findLayerTensor(engine.model, layer, "ffn_gate_exps.weight") orelse return error.MissingTensor;
            const up_exps = findLayerTensor(engine.model, layer, "ffn_up_exps.weight") orelse return error.MissingTensor;
            const down_exps = findLayerTensor(engine.model, layer, "ffn_down_exps.weight") orelse return error.MissingTensor;
            const gate_quant = gate_exps.info.type_;
            const down_quant = down_exps.info.type_;
            const expert_gate_bytes = expertSliceBytes(gate_quant, inter_dim, hidden_dim);
            const expert_down_bytes = expertSliceBytes(down_quant, hidden_dim, inter_dim);

            {
                var cmd = try metal_command.beginCommand(engine.device.ctx);

                for (0..cfg.n_experts_used) |ei| {
                    const eid = expert_ids[ei];
                    const weight = expert_weights[ei];
                    const gate_offset = eid * expert_gate_bytes;
                    const up_offset = eid * expert_gate_bytes;
                    const down_offset = eid * expert_down_bytes;

                    // Gate + Up DMMVs (parallel — different output buffers)
                    dispatchDmmvOnCmd(engine, &cmd, gate_exps, &engine.norm_buf, &engine.gate_buf, inter_dim, hidden_dim, gate_offset);
                    dispatchDmmvOnCmd(engine, &cmd, up_exps, &engine.norm_buf, &engine.up_buf, inter_dim, hidden_dim, up_offset);
                    cmd.barrier();

                    // SwiGLU on GPU: output = SiLU(gate) * up
                    const swiglu_push = SwiGLUPush{ .n = inter_dim };
                    const sw_bufs = [_]*const MetalBuffer{ &engine.gate_buf, &engine.swiglu_buf, &engine.up_buf };
                    cmd.dispatchV2(&engine.swiglu_pipe, .{ (inter_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &sw_bufs, &swiglu_push, @sizeOf(SwiGLUPush), 0);
                    cmd.barrier();

                    // Down DMMV: swiglu_buf → down_buf
                    dispatchDmmvOnCmd(engine, &cmd, down_exps, &engine.swiglu_buf, &engine.down_buf, hidden_dim, inter_dim, down_offset);
                    cmd.barrier();

                    // Weighted accumulate on GPU: moe_out_buf += weight * down_buf
                    const scale_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(weight)) };
                    const sa_bufs = [_]*const MetalBuffer{ &engine.moe_out_buf, &engine.down_buf };
                    cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &sa_bufs, &scale_push, @sizeOf(ScaleAccPush), 0);
                    cmd.barrier();
                }

                // Shared expert (if present) — also batched in same command buffer
                const gate_shexp = findLayerTensor(engine.model, layer, "ffn_gate_shexp.weight");
                const up_shexp = findLayerTensor(engine.model, layer, "ffn_up_shexp.weight");
                const down_shexp = findLayerTensor(engine.model, layer, "ffn_down_shexp.weight");
                const shexp_gate_t = findLayerTensor(engine.model, layer, "ffn_gate_inp_shexp.weight");

                if (gate_shexp != null and up_shexp != null and down_shexp != null) {
                    dispatchDmmvOnCmd(engine, &cmd, gate_shexp.?, &engine.norm_buf, &engine.gate_buf, shexp_inter_dim, hidden_dim, 0);
                    dispatchDmmvOnCmd(engine, &cmd, up_shexp.?, &engine.norm_buf, &engine.up_buf, shexp_inter_dim, hidden_dim, 0);
                    if (shexp_gate_t) |sg| {
                        dispatchDmmvOnCmd(engine, &cmd, sg, &engine.norm_buf, &engine.router_logits_buf, 1, hidden_dim, 0);
                    }
                    cmd.barrier();

                    // SwiGLU on GPU
                    const sw_push = SwiGLUPush{ .n = shexp_inter_dim };
                    const sw_bufs2 = [_]*const MetalBuffer{ &engine.gate_buf, &engine.swiglu_buf, &engine.up_buf };
                    cmd.dispatchV2(&engine.swiglu_pipe, .{ (shexp_inter_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &sw_bufs2, &sw_push, @sizeOf(SwiGLUPush), 0);
                    cmd.barrier();

                    // Down DMMV
                    dispatchDmmvOnCmd(engine, &cmd, down_shexp.?, &engine.swiglu_buf, &engine.down_buf, hidden_dim, shexp_inter_dim, 0);
                    cmd.barrier();

                    if (shexp_gate_t != null) {
                        // Need gate scalar readback — commit batch, CPU sigmoid, CPU accumulate
                        cmd.commitAndWait();
                        const sg_ptr: [*]const f32 = @ptrCast(@alignCast(engine.router_logits_buf.cpu_ptr.?));
                        const shexp_weight = 1.0 / (1.0 + @exp(-sg_ptr[0]));
                        const d_ptr: [*]const f32 = @ptrCast(@alignCast(engine.down_buf.cpu_ptr.?));
                        for (0..hidden_dim) |i| moe_ptr[i] += shexp_weight * d_ptr[i];
                    } else {
                        // No gate — accumulate with scale=1.0 on GPU
                        const one_push = ScaleAccPush{ .n = hidden_dim, .scale_bits = @as(u32, @bitCast(@as(f32, 1.0))) };
                        const sa_bufs2 = [_]*const MetalBuffer{ &engine.moe_out_buf, &engine.down_buf };
                        cmd.dispatchV2(&engine.scale_acc_pipe, .{ (hidden_dim + 63) / 64, 1, 1 }, .{ 64, 1, 1 }, &sa_bufs2, &one_push, @sizeOf(ScaleAccPush), 0);
                    }
                }

                cmd.commitAndWait();
            }

            // FFN residual: hidden += moe_out (CPU, via UMA)
            const hidden_ptr: [*]f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));
            for (0..hidden_dim) |i| hidden_ptr[i] += moe_ptr[i];
        } else {
            // Dense FFN (non-MoE) — norm_buf already set by GPU batch 2
            const gate_t = findLayerTensor(engine.model, layer, "ffn_gate.weight") orelse return error.MissingTensor;
            const up_t = findLayerTensor(engine.model, layer, "ffn_up.weight") orelse return error.MissingTensor;
            const down_t = findLayerTensor(engine.model, layer, "ffn_down.weight") orelse return error.MissingTensor;

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
    {
        var cmd = try metal_command.beginCommand(engine.device.ctx);
        dispatchRmsNormOnCmd(engine, &cmd, &engine.hidden_buf, &engine.norm_buf, &engine.final_norm_gpu, hidden_dim, 1);
        cmd.barrier();
        const lm_tensor = findTensorByName(engine.model, "output.weight") orelse
            findTensorByName(engine.model, "token_embd.weight") orelse return error.MissingTensor;
        dispatchDmmvOnCmd(engine, &cmd, lm_tensor, &engine.norm_buf, &engine.logits_buf, cfg.vocab_size, hidden_dim, 0);
        cmd.commitAndWait();
    }

    engine.position += 1;
}

// ---------------------------------------------------------------------------
// SSM CPU core — conv1d, state update, gated norm.
// Assumes projection results already in UMA buffers (from GPU batch 1).
// Writes SSM output to swiglu_buf for subsequent GPU SSM out DMMV.
// ---------------------------------------------------------------------------

fn runSsmCpuCore(
    engine: *InferenceEngine,
    layer: u32,
    layer_idx: usize,
    mmap: []const u8,
    data_base: u64,
) !void {
    const cfg = engine.config;
    const d_inner = cfg.ssm_d_inner;
    const d_conv = cfg.ssm_d_conv;
    const d_state = cfg.ssm_d_state;
    const n_group = cfg.ssm_n_group;
    const dt_rank = cfg.ssm_dt_rank;

    if (d_inner == 0) return;

    const head_v_dim = d_inner / @max(dt_rank, 1);
    const conv_channels: u32 = d_inner + 2 * n_group * d_state;

    // Read projection results from UMA buffers (written by GPU batch 1)
    const qkv_cpu: [*]const f32 = @ptrCast(@alignCast(engine.attn_out_buf.cpu_ptr.?));
    const z_cpu: [*]const f32 = @ptrCast(@alignCast(engine.gate_buf.cpu_ptr.?));
    const alpha_cpu: [*]const f32 = @ptrCast(@alignCast(engine.router_logits_buf.cpu_ptr.?));
    const beta_cpu: [*]const f32 = @ptrCast(@alignCast(engine.down_buf.cpu_ptr.?));

    // Conv1d with state
    const conv_state = engine.ssm_conv_states.?[layer_idx];
    const d_conv_1 = d_conv - 1;

    const conv_t = findLayerTensor(engine.model, layer, "ssm_conv1d.weight") orelse return;
    const conv_off: usize = @intCast(data_base + conv_t.info.offset);
    const conv_kernel_len: usize = @as(usize, conv_channels) * d_conv;
    const conv_kernel = try engine.allocator.alloc(f32, conv_kernel_len);
    defer engine.allocator.free(conv_kernel);
    readMmapFloats(mmap, conv_off, conv_t.info.type_, conv_kernel);

    const conv_out = try engine.allocator.alloc(f32, conv_channels);
    defer engine.allocator.free(conv_out);
    for (0..conv_channels) |ch| {
        var sum: f32 = 0;
        for (0..d_conv) |ki| {
            const kw = conv_kernel[ch * d_conv + ki];
            const sv = if (ki < d_conv_1) conv_state[ki * conv_channels + ch] else qkv_cpu[ch];
            sum += kw * sv;
        }
        const sig = 1.0 / (1.0 + @exp(-sum));
        conv_out[ch] = sum * sig;
    }

    // Update conv state: shift left, write current input
    if (d_conv_1 > 1) {
        const shift = (d_conv_1 - 1) * conv_channels;
        std.mem.copyForwards(f32, conv_state[0..shift], conv_state[conv_channels .. shift + conv_channels]);
    }
    for (0..conv_channels) |ch| conv_state[(d_conv_1 - 1) * conv_channels + ch] = qkv_cpu[ch];

    // Split Q/K/V from conv output
    const qk_dim = d_state * n_group;
    var q_ssm = conv_out[0..qk_dim];
    var k_ssm = conv_out[qk_dim .. 2 * qk_dim];
    const v_ssm = conv_out[2 * qk_dim .. 2 * qk_dim + d_inner];

    // L2 normalize per group
    for (0..n_group) |h| {
        cpuL2Normalize(q_ssm[h * d_state ..][0..d_state]);
        cpuL2Normalize(k_ssm[h * d_state ..][0..d_state]);
    }

    // Compute gate and beta
    const dt_bias_t = findLayerTensor(engine.model, layer, "ssm_dt.bias");
    const dt_bias = try engine.allocator.alloc(f32, dt_rank);
    defer engine.allocator.free(dt_bias);
    if (dt_bias_t) |t| {
        const off: usize = @intCast(data_base + t.info.offset);
        readMmapFloats(mmap, off, t.info.type_, dt_bias);
    } else @memset(dt_bias, 0);

    const ssm_a_t = findLayerTensor(engine.model, layer, "ssm_a");
    const ssm_a = try engine.allocator.alloc(f32, dt_rank);
    defer engine.allocator.free(ssm_a);
    if (ssm_a_t) |t| {
        const off: usize = @intCast(data_base + t.info.offset);
        readMmapFloats(mmap, off, t.info.type_, ssm_a);
    } else @memset(ssm_a, 0);

    const gate_arr = try engine.allocator.alloc(f32, dt_rank);
    defer engine.allocator.free(gate_arr);
    const beta_arr = try engine.allocator.alloc(f32, dt_rank);
    defer engine.allocator.free(beta_arr);
    for (0..dt_rank) |i| {
        var a = alpha_cpu[i];
        if (dt_bias_t != null) a += dt_bias[i];
        const sp = @log(1.0 + @exp(a));
        gate_arr[i] = if (ssm_a_t != null) sp * ssm_a[i] else -sp;
        beta_arr[i] = 1.0 / (1.0 + @exp(-beta_cpu[i]));
    }

    // Scale Q
    const q_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_state)));
    for (q_ssm) |*v| v.* *= q_scale;

    // Delta-net state update
    const ssm_state = engine.ssm_states.?[layer_idx];
    for (0..dt_rank) |h| {
        const s_base = h * head_v_dim * head_v_dim;
        const g_val = @exp(gate_arr[h]);
        const b_val = beta_arr[h];
        const k_hi = if (n_group == dt_rank) h else h % n_group;
        const k_head = k_ssm[k_hi * d_state ..][0..@min(d_state, head_v_dim)];
        const v_head = v_ssm[h * head_v_dim ..][0..head_v_dim];

        // Decay
        for (0..head_v_dim * head_v_dim) |i| ssm_state[s_base + i] *= g_val;

        // Update
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

    // Read from state
    const ssm_output = try engine.allocator.alloc(f32, d_inner);
    defer engine.allocator.free(ssm_output);
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

    // Gated normalization: RMS_norm(o) * SiLU(z)
    const norm_t = findLayerTensor(engine.model, layer, "ssm_norm.weight");
    const norm_elems_ssm: u32 = if (norm_t) |t| @intCast(t.info.numElements()) else 0;
    const norm_per_head = norm_elems_ssm >= d_inner;
    const norm_alloc_len: u32 = if (norm_elems_ssm > 0) norm_elems_ssm else 1;
    const norm_w_ssm = try engine.allocator.alloc(f32, norm_alloc_len);
    defer engine.allocator.free(norm_w_ssm);
    if (norm_t) |t| {
        const off: usize = @intCast(data_base + t.info.offset);
        readMmapFloats(mmap, off, t.info.type_, norm_w_ssm[0..norm_elems_ssm]);
    }

    for (0..dt_rank) |h| {
        const o_sl = ssm_output[h * head_v_dim ..][0..head_v_dim];
        const z_sl_off = h * head_v_dim;
        var sq: f32 = 0;
        for (o_sl) |v| sq += v * v;
        const rms = @sqrt(sq / @as(f32, @floatFromInt(head_v_dim)) + 1e-6);
        for (0..head_v_dim) |i| {
            var nv = o_sl[i] / rms;
            if (norm_t != null) nv *= norm_w_ssm[if (norm_per_head) h * head_v_dim + i else i % d_state];
            const zv = z_cpu[z_sl_off + i];
            o_sl[i] = nv * (zv / (1.0 + @exp(-zv)));
        }
    }

    // Write SSM output to swiglu_buf for subsequent GPU SSM out DMMV
    const sw_ptr: [*]f32 = @ptrCast(@alignCast(engine.swiglu_buf.cpu_ptr.?));
    for (0..d_inner) |i| sw_ptr[i] = ssm_output[i];
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
    const cfg = engine.config;
    const mmap = engine.model.mmap_data orelse return error.NoMmapData;

    var output: std.ArrayList(u32) = .{};
    errdefer output.deinit(allocator);

    // Embedding dequant buffer
    const embed_buf = try allocator.alloc(f32, cfg.hidden_dim);
    defer allocator.free(embed_buf);

    const embed_tensor = engine.model.gguf_file.findTensor("token_embd.weight") orelse return error.MissingTensor;
    const embed_data_offset = engine.model.gguf_file.tensor_data_offset + embed_tensor.offset;
    const embed_raw = mmap[embed_data_offset..];

    const hidden_ptr: [*]f32 = @ptrCast(@alignCast(engine.hidden_buf.cpu_ptr.?));

    // Prefill: process each prompt token through all layers
    const prefill_start = std.time.nanoTimestamp();
    for (prompt_tokens) |token_id| {
        dequantRow(embed_raw, token_id, cfg.hidden_dim, embed_tensor.type_, embed_buf);
        @memcpy(hidden_ptr[0..cfg.hidden_dim], embed_buf);
        try decodeStep(engine);
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
        dequantRow(embed_raw, input_token, cfg.hidden_dim, embed_tensor.type_, embed_buf);
        @memcpy(hidden_ptr[0..cfg.hidden_dim], embed_buf);

        try decodeStep(engine);

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
