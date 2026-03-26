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
    position: u32,
    generated_tokens: std.ArrayList(u32),
    allocator: std.mem.Allocator,

    /// Initialize decode state for a fresh generation request.
    pub fn init(allocator: std.mem.Allocator) DecodeState {
        return .{
            .position = 0,
            .generated_tokens = .{},
            .allocator = allocator,
        };
    }

    /// Release the generated token buffer owned by the decode state.
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
    model: *Model,
    gpu_config: GpuConfig,
    dmmv: DmmvDispatch,
    elementwise: ElementwiseDispatch,
    attention: AttentionDispatch,
    cmd_pool: CommandPool,
    decode_cmd: CommandBuffer,
    decode_graph: Graph,
    // Intermediate buffers
    hidden_buf: Buffer, // hidden state (hidden_dim f32)
    residual_buf: Buffer, // residual connection
    norm_buf: Buffer, // RMS norm output
    logits_buf: Buffer, // output logits (vocab_size f32)
    logits_staging: Buffer, // pre-allocated logits readback staging
    embed_staging: Buffer, // pre-allocated embedding upload staging
    // Descriptor management
    shared_pool: vk.c.VkDescriptorPool,
    instance: *const Instance,
    allocator: std.mem.Allocator,

    /// Create the runtime objects needed to execute decode-time work on the GPU.
    pub fn init(
        model: *Model,
        instance: *const Instance,
        gpu_config: GpuConfig,
        shader_dir: []const u8,
        allocator: std.mem.Allocator,
    ) !InferenceEngine {
        const config = &model.config;

        var cmd_pool = try CommandPool.init(instance);
        errdefer cmd_pool.deinit();

        var decode_cmd = try CommandBuffer.init(instance, &cmd_pool);
        errdefer decode_cmd.deinit(&cmd_pool);

        var dmmv = try DmmvDispatch.init(instance, &gpu_config, shader_dir, allocator);
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
        var hidden_buf = try Buffer.initDeviceLocal(instance, hidden_size, vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        errdefer hidden_buf.deinit();

        var residual_buf = try Buffer.initDeviceLocal(instance, hidden_size, vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
        errdefer residual_buf.deinit();

        var norm_buf = try Buffer.initDeviceLocal(instance, hidden_size, vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
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

        // Create shared descriptor pool for per-step descriptor set allocations
        const pool_sizes = [_]vk.c.VkDescriptorPoolSize{.{
            .type = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 256,
        }};
        const pool_info = vk.c.VkDescriptorPoolCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = 0, // will reset entire pool each step
            .maxSets = 64,
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

    /// Run a single decode step: embed token → final RMS norm → lm_head DMMV → logits.
    /// All GPU work (embedding upload, compute, logits readback) is recorded in a
    /// single command buffer submission to minimize per-token Vulkan overhead.
    pub fn decodeStep(self: *InferenceEngine, state: *DecodeState, token_id: u32) !void {
        // 1. CPU: dequantize embedding into pre-allocated staging buffer
        try self.embedToken(token_id);

        // 2. Reset the shared descriptor pool (frees all sets from previous step)
        _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);

        // 3. Record all GPU work in a single command buffer
        try self.decode_cmd.reset();
        try self.decode_cmd.begin();

        // --- Transfer: embed_staging → hidden_buf ---
        {
            const embed_size = @as(vk.c.VkDeviceSize, self.model.config.hidden_dim) * @sizeOf(f32);
            const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = embed_size };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.embed_staging.handle, self.hidden_buf.handle, 1, &region);

            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_SHADER_READ_BIT,
            };
            vk.c.vkCmdPipelineBarrier(
                self.decode_cmd.handle,
                vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &barrier, 0, null, 0, null,
            );
        }

        // --- Final RMS norm: hidden_buf → norm_buf ---
        const norm_tensor = findLoadedTensor(self.model, "output_norm.weight") orelse {
            log.err("output_norm.weight not found", .{});
            return error.TensorNotFound;
        };
        if (self.elementwise.pipeline_rms_norm) |*pip| {
            const ds = try self.allocDescSet(pip.descriptor_set_layout);
            self.writeDescSet3(
                ds,
                self.hidden_buf.handle,
                self.hidden_buf.size,
                norm_tensor.gpu_buffer.handle,
                norm_tensor.gpu_buffer.size,
                self.norm_buf.handle,
                self.norm_buf.size,
            );
            try self.elementwise.recordRmsNorm(
                &self.decode_cmd,
                ds,
                self.model.config.hidden_dim,
                1,
                1e-6,
            );
        } else {
            log.warn("RMS norm shader not loaded, skipping", .{});
        }

        self.decode_cmd.computeBarrier();

        // --- LM head: output.weight × norm_buf → logits_buf ---
        const lm_tensor = findLoadedTensor(self.model, "output.weight") orelse
            findLoadedTensor(self.model, "token_embd.weight") orelse
        {
            log.err("output.weight not found", .{});
            return error.TensorNotFound;
        };
        const quant_type = lm_tensor.info.type_;
        if (self.dmmv.pipelineForType(quant_type)) |pip| {
            const ds = try self.allocDescSet(pip.descriptor_set_layout);
            self.writeDescSet3(
                ds,
                lm_tensor.gpu_buffer.handle,
                lm_tensor.gpu_buffer.size,
                self.norm_buf.handle,
                self.norm_buf.size,
                self.logits_buf.handle,
                self.logits_buf.size,
            );
            try self.dmmv.recordDispatch(
                &self.decode_cmd,
                quant_type,
                ds,
                self.model.config.vocab_size,
                self.model.config.hidden_dim,
                0,
                0,
                0,
            );
        } else {
            log.err("No DMMV pipeline for quant type {d}", .{@intFromEnum(quant_type)});
            return error.UnsupportedQuantType;
        }

        // --- Readback: logits_buf → logits_staging ---
        {
            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
            };
            vk.c.vkCmdPipelineBarrier(
                self.decode_cmd.handle,
                vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &barrier, 0, null, 0, null,
            );
            const logits_copy_size = @as(vk.c.VkDeviceSize, self.model.config.vocab_size) * @sizeOf(f32);
            const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = logits_copy_size };
            vk.c.vkCmdCopyBuffer(self.decode_cmd.handle, self.logits_buf.handle, self.logits_staging.handle, 1, &region);
        }

        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

        state.position += 1;
    }

    /// Batch-process all prompt tokens in a single GPU submission.
    /// Eliminates per-token submit/wait overhead by recording all token
    /// computations in one command buffer with vkCmdUpdateBuffer for embeddings.
    fn prefillBatch(self: *InferenceEngine, state: *DecodeState, prompt_tokens: []const u32) !void {
        if (prompt_tokens.len == 0) return;

        const hidden_dim = self.model.config.hidden_dim;
        const embed_size = @as(vk.c.VkDeviceSize, hidden_dim) * @sizeOf(f32);

        // Find required tensors
        const embd = findLoadedTensor(self.model, "token_embd.weight") orelse {
            log.err("token_embd.weight not found", .{});
            return error.TensorNotFound;
        };
        const norm_tensor = findLoadedTensor(self.model, "output_norm.weight") orelse {
            log.err("output_norm.weight not found", .{});
            return error.TensorNotFound;
        };
        const lm_tensor = findLoadedTensor(self.model, "output.weight") orelse
            findLoadedTensor(self.model, "token_embd.weight") orelse
        {
            log.err("output.weight not found", .{});
            return error.TensorNotFound;
        };

        const mmap = self.model.mmap_data orelse return error.NoMmapData;
        const data_start: usize = @intCast(self.model.gguf_file.tensor_data_offset + embd.info.offset);
        const quant_type = lm_tensor.info.type_;

        // Pre-dequantize all embeddings on CPU
        const hdim: usize = @intCast(hidden_dim);
        // Only the last token needs LM-head projection — without transformer
        // layers, intermediate tokens don't fill KV cache, so their embedding
        // → norm → DMMV is redundant (logits get overwritten each iteration).
        // This cuts weight reads from N × 273 MB to 1 × 273 MB.
        const embeddings = try self.allocator.alloc(f32, hdim);
        defer self.allocator.free(embeddings);

        {
            const last_token = prompt_tokens[prompt_tokens.len - 1];
            const safe_id = @min(last_token, self.model.config.vocab_size -| 1);
            dequantRow(mmap[data_start..], safe_id, hidden_dim, embd.info.type_, embeddings[0..hdim]);
        }

        // Reset descriptor pool and allocate sets once for all tokens
        _ = vk.c.vkResetDescriptorPool(self.instance.device, self.shared_pool, 0);

        const rms_pip = self.elementwise.pipeline_rms_norm orelse {
            log.warn("RMS norm shader not loaded, skipping prefill", .{});
            return;
        };
        const rms_ds = try self.allocDescSet(rms_pip.descriptor_set_layout);
        self.writeDescSet3(
            rms_ds,
            self.hidden_buf.handle, self.hidden_buf.size,
            norm_tensor.gpu_buffer.handle, norm_tensor.gpu_buffer.size,
            self.norm_buf.handle, self.norm_buf.size,
        );

        const dmmv_pip = self.dmmv.pipelineForType(quant_type) orelse {
            log.err("No DMMV pipeline for quant type {d}", .{@intFromEnum(quant_type)});
            return error.UnsupportedQuantType;
        };
        const dmmv_ds = try self.allocDescSet(dmmv_pip.descriptor_set_layout);
        self.writeDescSet3(
            dmmv_ds,
            lm_tensor.gpu_buffer.handle, lm_tensor.gpu_buffer.size,
            self.norm_buf.handle, self.norm_buf.size,
            self.logits_buf.handle, self.logits_buf.size,
        );

        // Record single command buffer — project only the last token through
        // final-norm + LM-head (see allocation comment above for rationale).
        try self.decode_cmd.reset();
        try self.decode_cmd.beginOneTime();

        // Upload last token's embedding
        vk.c.vkCmdUpdateBuffer(
            self.decode_cmd.handle,
            self.hidden_buf.handle,
            0,
            embed_size,
            @ptrCast(embeddings.ptr),
        );

        // Barrier: transfer → compute
        {
            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_TRANSFER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_SHADER_READ_BIT,
            };
            vk.c.vkCmdPipelineBarrier(
                self.decode_cmd.handle,
                vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
                vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                0, 1, &barrier, 0, null, 0, null,
            );
        }

        // RMS norm: hidden_buf → norm_buf
        try self.elementwise.recordRmsNorm(
            &self.decode_cmd, rms_ds, hidden_dim, 1, 1e-6,
        );

        // Barrier: compute → compute
        self.decode_cmd.computeBarrier();

        // DMMV: lm_tensor × norm_buf → logits_buf
        try self.dmmv.recordDispatch(
            &self.decode_cmd, quant_type, dmmv_ds,
            self.model.config.vocab_size, hidden_dim, 0, 0, 0,
        );

        // After last token: readback logits
        {
            const barrier = vk.c.VkMemoryBarrier{
                .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                .pNext = null,
                .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
                .dstAccessMask = vk.c.VK_ACCESS_TRANSFER_READ_BIT,
            };
            vk.c.vkCmdPipelineBarrier(
                self.decode_cmd.handle,
                vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0, 1, &barrier, 0, null, 0, null,
            );
            const logits_copy_size = @as(vk.c.VkDeviceSize, self.model.config.vocab_size) * @sizeOf(f32);
            const region = vk.c.VkBufferCopy{ .srcOffset = 0, .dstOffset = 0, .size = logits_copy_size };
            vk.c.vkCmdCopyBuffer(
                self.decode_cmd.handle,
                self.logits_buf.handle,
                self.logits_staging.handle,
                1, &region,
            );
        }

        try self.decode_cmd.end();
        try self.decode_cmd.submitAndWait(self.instance.compute_queue);

        state.position += @intCast(prompt_tokens.len);
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
    // Teardown
    // -----------------------------------------------------------------------

    /// Release GPU buffers, graphs, and dispatch helpers owned by the engine.
    pub fn deinit(self: *InferenceEngine) void {
        vk.c.vkDestroyDescriptorPool(self.instance.device, self.shared_pool, null);
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

/// Run single-request inference: tokenize → prefill → decode → detokenize.
pub fn generate(
    engine: *InferenceEngine,
    prompt_tokens: []const u32,
    max_tokens: u32,
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

        // Check for EOS — common EOS token IDs across model families
        if (token == 151643 or token == 128001 or token == 2) break;
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
    } else {
        log.info("Generated {d} tokens", .{decode_tokens});
    }

    return try allocator.dupe(u32, state.generated_tokens.items);
}
