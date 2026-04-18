//! Wrap the decode-time matrix-vector shader family used for projection ops.
//! @section Shader Dispatch
//! This helper selects quantization-specific DMMV pipelines and records the
//! push constants and workgroup sizes needed for single-token decode.
const std = @import("std");
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const Pipeline = @import("../vulkan/pipeline.zig").Pipeline;
const pipeline_mod = @import("../vulkan/pipeline.zig");
const CommandBuffer = @import("../vulkan/command.zig").CommandBuffer;
const Buffer = @import("../vulkan/buffer.zig").Buffer;
const GpuConfig = @import("../vulkan/gpu_detect.zig").GpuConfig;
const GGMLType = @import("../model/gguf.zig").GGMLType;

const log = std.log.scoped(.dmmv);
const descriptor_pool_max_sets: u32 = 256;
const descriptors_per_set: u32 = 3;

/// Push constants for DMMV shaders (must match GLSL layout).
pub const DmmvPushConstants = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    x_offset: u32,
    y_offset: u32,
    acc_mode: u32 = 0, // 0 = overwrite (y = result), 1 = accumulate (y += result)
};

/// Push constants for batch DMMV shaders (prefill: multiple columns).
pub const BatchDmmvPushConstants = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    x_offset: u32,
    y_offset: u32,
    num_cols: u32,
};

/// Push constants for MoE DMMV shaders (must match GLSL layout).
/// Batched expert dispatch: workgroup Y dimension selects expert slot.
pub const MoeDmmvPushConstants = extern struct {
    M: u32,
    K: u32,
    expert_stride: u32,
    x_expert_stride: u32,
    x_offset: u32,
    y_offset: u32,
};

/// Manages DMMV pipelines for different quantization types.
pub const DmmvDispatch = struct {
    /// Q4K pipeline, or null.
    pipeline_q4k: ?Pipeline,
    /// Q5K pipeline, or null.
    pipeline_q5k: ?Pipeline,
    /// Q6K pipeline, or null.
    pipeline_q6k: ?Pipeline,
    /// MXFP4 pipeline, or null.
    pipeline_mxfp4: ?Pipeline,
    /// Q5_0 pipeline, or null.
    pipeline_q5_0: ?Pipeline,
    /// Q5_1 pipeline, or null.
    pipeline_q5_1: ?Pipeline,
    /// Q8 0 pipeline, or null.
    pipeline_q8_0: ?Pipeline,
    /// F16 pipeline, or null.
    pipeline_f16: ?Pipeline,
    /// F32 pipeline, or null.
    pipeline_f32: ?Pipeline,
    /// Batch Q4K pipeline for prefill (3 bindings: A, X_batch, Y_batch).
    pipeline_q4k_batch: ?Pipeline,
    /// Batch Q8_0 pipeline for prefill (3 bindings: A, X_batch, Y_batch).
    pipeline_q8_0_batch: ?Pipeline,
    /// Q4K integer dot product pipeline (3 bindings: A_q4k, B_q8_1, Y).
    pipeline_q4k_idp: ?Pipeline,
    /// Q8_1 quantization pipeline (2 bindings: X_f32, Q8_1_out).
    pipeline_quantize_q8_1: ?Pipeline,
    /// MoE Q4K pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_q4k_moe: ?Pipeline,
    /// MoE Q5K pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_q5k_moe: ?Pipeline,
    /// MoE MXFP4 pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_mxfp4_moe: ?Pipeline,
    /// MoE Q5_1 pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_q5_1_moe: ?Pipeline,
    /// MoE Q6K pipeline (4 bindings: A, x, y, routing), or null.
    pipeline_q6k_moe: ?Pipeline,
    /// Descriptor pool for this dispatch.
    descriptor_pool: vk.c.VkDescriptorPool,
    /// Logical device.
    device: vk.c.VkDevice,

    /// Create the DMMV dispatch wrapper and load the supported quantized pipelines.
    /// @param instance Active Vulkan instance and logical device.
    /// @param gpu_config Derived GPU tuning parameters.
    /// @param shader_dir Directory containing compiled SPIR-V shader binaries.
    /// @param allocator Allocator used for temporary pipeline creation state.
    /// @returns A DmmvDispatch ready to record projection work.
    pub fn init(
        /// Vulkan instance.
        instance: *const Instance,
        /// GPU capabilities.
        gpu_config: *const GpuConfig,
        shader_dir: []const u8,
        /// Hidden state width.
        hidden_dim: u32,
        /// Allocator for owned resources.
        allocator: std.mem.Allocator,
    ) !DmmvDispatch {
        _ = gpu_config;

        // Create descriptor pool
        const pool_size = vk.c.VkDescriptorPoolSize{
            .type = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            // Size for runtime reuse plus rotating hot-bench working sets.
            .descriptorCount = descriptor_pool_max_sets * descriptors_per_set,
        };
        const pool_info = vk.c.VkDescriptorPoolCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.c.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = descriptor_pool_max_sets,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        var descriptor_pool: vk.c.VkDescriptorPool = null;
        if (instance.push_descriptor_fn == null) {
            const result = vk.c.vkCreateDescriptorPool(instance.device, &pool_info, null, &descriptor_pool);
            if (result != vk.c.VK_SUCCESS) return error.DescriptorPoolCreateFailed;
        }

        // Load pipelines (3 bindings: A matrix, x vector, y output)
        const push_size = @sizeOf(DmmvPushConstants);

        // Specialization constant: SPEC_K (id=1) = max_k to size the shared memory
        // array in the Q4_K shader (s_x[SPEC_K]). Must be >= the largest K value
        // used in any Q4_K dispatch (hidden_dim, inter_dim, q_dim, d_inner).
        const spec_k = [_]pipeline_mod.SpecConst{.{ .id = 1, .value = hidden_dim }};
        const has_push_desc = instance.push_descriptor_fn != null;
        const push_desc_options = pipeline_mod.PipelineOptions{
            .push_descriptors = has_push_desc,
        };
        const push_desc_wave64_options = pipeline_mod.PipelineOptions{
            .required_subgroup_size = 64,
            .require_full_subgroups = true,
            .push_descriptors = has_push_desc,
        };

        var path_buf: [512]u8 = undefined;

        const q4k_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k = pipeline_mod.createFromSpirvWithOptions(instance, q4k_path, 3, push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q4_K shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q8_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0 = pipeline_mod.createFromSpirvWithOptions(instance, q8_path, 3, push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q8_0 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const mxfp4_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_mxfp4.spv", .{shader_dir}) catch unreachable;
        const pipeline_mxfp4 = pipeline_mod.createFromSpirvWithOptions(instance, mxfp4_path, 3, push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("MXFP4 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q5_0_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5_0.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5_0 = pipeline_mod.createFromSpirvWithOptions(instance, q5_0_path, 3, push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_0 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q5_1_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5_1 = pipeline_mod.createFromSpirvWithOptions(instance, q5_1_path, 3, push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("Q5_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q5k_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5k.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5k = pipeline_mod.createFromSpirvWithOptions(instance, q5k_path, 3, push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q5_K shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q6k_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q6k.spv", .{shader_dir}) catch unreachable;
        const pipeline_q6k = pipeline_mod.createFromSpirvWithOptions(instance, q6k_path, 3, push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q6_K shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const f16_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_f16.spv", .{shader_dir}) catch unreachable;
        const pipeline_f16 = pipeline_mod.createFromSpirvWithOptions(instance, f16_path, 3, push_size, &.{}, push_desc_wave64_options, allocator) catch |err| blk: {
            log.warn("F16 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const f32_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_f32.spv", .{shader_dir}) catch unreachable;
        const pipeline_f32 = pipeline_mod.createFromSpirvWithOptions(instance, f32_path, 3, push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("F32 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Batch DMMV for prefill: 3 bindings (A, X_batch, Y_batch), batch push constants
        const batch_push_size = @sizeOf(BatchDmmvPushConstants);
        const q4k_batch_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_batch = pipeline_mod.createFromSpirvWithOptions(instance, q4k_batch_path, 3, batch_push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q4_K batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q8_0_batch_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0_batch.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0_batch = pipeline_mod.createFromSpirvWithOptions(instance, q8_0_batch_path, 3, batch_push_size, &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q8_0 batch shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Integer dot product (IDP) DMMV: Q4_K × Q8_1 with dotPacked4x8EXT
        const IdpPush = extern struct { M: u32, K: u32, a_offset: u32, b_offset: u32, y_offset: u32 };
        const q4k_idp_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_idp.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_idp = pipeline_mod.createFromSpirvWithOptions(instance, q4k_idp_path, 3, @sizeOf(IdpPush), &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q4_K IDP shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Q8_1 quantization shader: 2 bindings (f32 input, Q8_1 output)
        const Q8Push = extern struct { K: u32, x_offset: u32 };
        const q8_path2 = std.fmt.bufPrint(&path_buf, "{s}/quantize_q8_1.spv", .{shader_dir}) catch unreachable;
        const pipeline_quantize_q8_1 = pipeline_mod.createFromSpirvWithOptions(instance, q8_path2, 2, @sizeOf(Q8Push), &.{}, push_desc_options, allocator) catch |err| blk: {
            log.warn("quantize_q8_1 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        if (pipeline_q4k_idp != null and pipeline_quantize_q8_1 != null) {
            log.info("IDP DMMV pipeline loaded — integer dot product acceleration enabled", .{});
        }

        // MoE DMMV pipelines: 4 bindings (A, x, y, routing), different push constants
        const moe_push_size = @sizeOf(MoeDmmvPushConstants);

        const q4k_moe_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k_moe = pipeline_mod.createFromSpirvWithOptions(instance, q4k_moe_path, 4, moe_push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q4_K MoE shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q5k_moe_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5k_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5k_moe = pipeline_mod.createFromSpirvWithOptions(instance, q5k_moe_path, 4, moe_push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q5_K MoE shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const mxfp4_moe_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_mxfp4_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_mxfp4_moe = pipeline_mod.createFromSpirvWithOptions(instance, mxfp4_moe_path, 4, moe_push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("MXFP4 MoE shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q5_1_moe_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q5_1_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_q5_1_moe = pipeline_mod.createFromSpirvWithOptions(instance, q5_1_moe_path, 4, moe_push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q5_1 MoE shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q6k_moe_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q6k_moe.spv", .{shader_dir}) catch unreachable;
        const pipeline_q6k_moe = pipeline_mod.createFromSpirvWithOptions(instance, q6k_moe_path, 4, moe_push_size, &spec_k, push_desc_options, allocator) catch |err| blk: {
            log.warn("Q6_K MoE shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        if (pipeline_q4k_moe != null and pipeline_q5k_moe != null and pipeline_q6k_moe != null) {
            log.info("MoE DMMV pipelines loaded — GPU expert dispatch enabled (no readback)", .{});
        }

        return DmmvDispatch{
            .pipeline_q4k = pipeline_q4k,
            .pipeline_mxfp4 = pipeline_mxfp4,
            .pipeline_q5_0 = pipeline_q5_0,
            .pipeline_q5_1 = pipeline_q5_1,
            .pipeline_q5k = pipeline_q5k,
            .pipeline_q6k = pipeline_q6k,
            .pipeline_q8_0 = pipeline_q8_0,
            .pipeline_f16 = pipeline_f16,
            .pipeline_f32 = pipeline_f32,
            .pipeline_q4k_batch = pipeline_q4k_batch,
            .pipeline_q8_0_batch = pipeline_q8_0_batch,
            .pipeline_q4k_idp = pipeline_q4k_idp,
            .pipeline_quantize_q8_1 = pipeline_quantize_q8_1,
            .pipeline_q4k_moe = pipeline_q4k_moe,
            .pipeline_mxfp4_moe = pipeline_mxfp4_moe,
            .pipeline_q5_1_moe = pipeline_q5_1_moe,
            .pipeline_q5k_moe = pipeline_q5k_moe,
            .pipeline_q6k_moe = pipeline_q6k_moe,
            .descriptor_pool = descriptor_pool,
            .device = instance.device,
        };
    }

    /// Select the quantization-specific pipeline used for a weight matrix format.
    /// @param self Dispatch wrapper containing the loaded DMMV pipelines.
    /// @param quant_type GGML quantization format for the weight matrix.
    /// @returns A pipeline pointer when that quantization format has a loaded shader implementation.
    /// @note Unsupported or unloaded formats return `null` so callers can surface `error.UnsupportedQuantType`.
    pub fn pipelineForType(self: *const DmmvDispatch, quant_type: GGMLType) ?*const Pipeline {
        return switch (quant_type) {
            .q4_k => if (self.pipeline_q4k) |*p| p else null,
            .mxfp4 => if (self.pipeline_mxfp4) |*p| p else null,
            .q5_0 => if (self.pipeline_q5_0) |*p| p else null,
            .q5_1 => if (self.pipeline_q5_1) |*p| p else null,
            .q5_k => if (self.pipeline_q5k) |*p| p else null,
            .q6_k => if (self.pipeline_q6k) |*p| p else null,
            .q8_0 => if (self.pipeline_q8_0) |*p| p else null,
            .f16 => if (self.pipeline_f16) |*p| p else null,
            .f32 => if (self.pipeline_f32) |*p| p else null,
            else => null,
        };
    }

    /// Select the MoE quantization-specific pipeline (4 bindings: A, x, y, routing).
    pub fn moePipelineForType(self: *const DmmvDispatch, quant_type: GGMLType) ?*const Pipeline {
        return switch (quant_type) {
            .q4_k => if (self.pipeline_q4k_moe) |*p| p else null,
            .mxfp4 => if (self.pipeline_mxfp4_moe) |*p| p else null,
            .q5_1 => if (self.pipeline_q5_1_moe) |*p| p else null,
            .q5_k => if (self.pipeline_q5k_moe) |*p| p else null,
            .q6_k => if (self.pipeline_q6k_moe) |*p| p else null,
            else => null,
        };
    }

    /// Record a batched MoE DMMV dispatch — all experts run in parallel via Y workgroups.
    /// expert_stride: bytes per expert in stacked weight tensor.
    /// n_experts_y: number of experts to process (dispatched as Y workgroups).
    /// x_expert_stride: elements between experts' inputs (0=shared input, K=per-expert).
    pub fn recordMoeDispatch(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        quant_type: GGMLType,
        descriptor_set: vk.c.VkDescriptorSet,
        M: u32,
        K: u32,
        expert_stride: u32,
        n_experts_y: u32,
        x_expert_stride: u32,
        x_offset: u32,
        y_offset: u32,
    ) !void {
        const pip = self.moePipelineForType(quant_type) orelse return error.UnsupportedQuantType;

        const push = MoeDmmvPushConstants{
            .M = M,
            .K = K,
            .expert_stride = expert_stride,
            .x_expert_stride = x_expert_stride,
            .x_offset = x_offset,
            .y_offset = y_offset,
        };

        const workgroups_x = switch (quant_type) {
            .mxfp4, .q8_0, .f16 => (M + 1) / 2,
            else => (M + 63) / 64,
        };

        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups_x, n_experts_y, 1);
    }

    /// Record a decode-time matrix-vector multiply dispatch.
    /// @param self Dispatch wrapper containing the quantization-specific pipelines.
    /// @param cmd Command buffer currently being recorded.
    /// @param quant_type GGML quantization format for the weight matrix.
    /// @param descriptor_set Descriptor set containing matrix, input vector, and output buffers.
    /// @param M Output row count.
    /// @param K Input feature width.
    /// @param a_offset Byte offset for the weight matrix.
    /// @param x_offset Byte offset for the input vector.
    /// @param y_offset Byte offset for the output vector.
    /// @returns `error.UnsupportedQuantType` when no pipeline is available for `quant_type`.
    /// @note The helper uses one workgroup per 64 output rows.
    pub fn recordDispatch(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        /// Quantization type.
        quant_type: GGMLType,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        M: u32,
        K: u32,
        /// Weight buffer byte offset.
        a_offset: u32,
        /// Input buffer byte offset.
        x_offset: u32,
        /// Output buffer byte offset.
        y_offset: u32,
    ) !void {
        const pip = self.pipelineForType(quant_type) orelse return error.UnsupportedQuantType;

        const push = DmmvPushConstants{
            .M = M,
            .K = K,
            .a_offset = a_offset,
            .x_offset = x_offset,
            .y_offset = y_offset,
        };

        // K-parallel (NUM_ROWS=2) for most DMMVs.
        // For very large M (LM head with M>64K), K-parallel creates too many workgroups.
        // Use the batch shader (1 thread per row, 64 rows/WG) which has fewer WGs and
        // better memory access patterns for large fan-out.
        const use_kparallel = switch (quant_type) {
            .q4_k => M <= 65536,
            .q6_k => true,
            else => false,
        };
        const workgroups_x = if (use_kparallel) switch (quant_type) {
            .q4_k, .q6_k => (M + 1) / 2,
            else => unreachable,
        } else switch (quant_type) {
            .q4_k => blk: {
                // Use batch shader in single-column mode (1 thread per row)
                if (self.pipeline_q4k_batch) |*batch_pip| {
                    const batch_push = BatchDmmvPushConstants{
                        .M = M,
                        .K = K,
                        .a_offset = a_offset,
                        .x_offset = x_offset,
                        .y_offset = y_offset,
                        .num_cols = 1,
                    };
                    cmd.dispatchWithPush(batch_pip, descriptor_set, std.mem.asBytes(&batch_push), (M + 63) / 64, 1, 1);
                    return;
                }
                break :blk (M + 1) / 2; // fallback to K-parallel
            },
            .q5_0, .q5_1, .mxfp4, .q8_0, .f16 => (M + 1) / 2,
            .f32 => M, // K-parallel: 64 threads per row via subgroupAdd
            else => (M + 63) / 64,
        };

        cmd.dispatchWithPush(
            pip,
            descriptor_set,
            std.mem.asBytes(&push),
            workgroups_x,
            1,
            1,
        );
    }

    /// Record a batch DMMV dispatch for prefill (multiple input columns).
    /// Weight matrix read once, multiplied against num_cols input vectors.
    /// Input/output are column-major: X[K, num_cols], Y[M, num_cols].
    pub fn recordBatchDispatch(
        self: *const DmmvDispatch,
        cmd: *CommandBuffer,
        quant_type: GGMLType,
        descriptor_set: vk.c.VkDescriptorSet,
        M: u32,
        K: u32,
        a_offset: u32,
        x_offset: u32,
        y_offset: u32,
        num_cols: u32,
    ) !void {
        // Q4_K and Q8_0 have dedicated batch shaders; fall back to sequential for others.
        const pip = switch (quant_type) {
            .q4_k => if (self.pipeline_q4k_batch) |*p| p else null,
            .q8_0 => if (self.pipeline_q8_0_batch) |*p| p else null,
            else => null,
        };

        if (pip) |p| {
            const push = BatchDmmvPushConstants{
                .M = M,
                .K = K,
                .a_offset = a_offset,
                .x_offset = x_offset,
                .y_offset = y_offset,
                .num_cols = num_cols,
            };
            const workgroups_x = (M + 63) / 64;
            cmd.dispatchWithPush(p, descriptor_set, std.mem.asBytes(&push), workgroups_x, 1, 1);
        } else {
            // Fallback: dispatch N single-column DMMVs
            const single_pip = self.pipelineForType(quant_type) orelse return error.UnsupportedQuantType;
            for (0..num_cols) |col| {
                const col_u32: u32 = @intCast(col);
                const push = DmmvPushConstants{
                    .M = M,
                    .K = K,
                    .a_offset = a_offset,
                    .x_offset = x_offset + col_u32 * K * @sizeOf(f32),
                    .y_offset = y_offset + col_u32 * M * @sizeOf(f32),
                };
                const workgroups_x_single = switch (quant_type) {
                    .q4_k, .q5_0, .q5_1, .q6_k => (M + 1) / 2,
                    .mxfp4, .q8_0, .f16 => (M + 1) / 2,
                    else => (M + 63) / 64,
                };
                cmd.dispatchWithPush(single_pip, descriptor_set, std.mem.asBytes(&push), workgroups_x_single, 1, 1);
            }
        }
    }

    /// Destroy the loaded pipelines and descriptor pool.
    /// @param self Dispatch wrapper to tear down in place.
    pub fn deinit(self: *DmmvDispatch) void {
        if (self.pipeline_q4k) |*p| p.deinit();
        if (self.pipeline_q5_1) |*p| p.deinit();
        if (self.pipeline_q5k) |*p| p.deinit();
        if (self.pipeline_q6k) |*p| p.deinit();
        if (self.pipeline_q8_0) |*p| p.deinit();
        if (self.pipeline_f16) |*p| p.deinit();
        if (self.pipeline_f32) |*p| p.deinit();
        if (self.pipeline_q4k_batch) |*p| p.deinit();
        if (self.pipeline_q8_0_batch) |*p| p.deinit();
        if (self.pipeline_q4k_idp) |*p| p.deinit();
        if (self.pipeline_quantize_q8_1) |*p| p.deinit();
        if (self.pipeline_q4k_moe) |*p| p.deinit();
        if (self.pipeline_q5k_moe) |*p| p.deinit();
        if (self.pipeline_q6k_moe) |*p| p.deinit();
        vk.c.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.* = undefined;
    }
};
