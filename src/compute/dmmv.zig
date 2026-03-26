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

/// Push constants for DMMV shaders (must match GLSL layout).
const DmmvPushConstants = extern struct {
    M: u32,
    K: u32,
    a_offset: u32,
    x_offset: u32,
    y_offset: u32,
};

/// Manages DMMV pipelines for different quantization types.
pub const DmmvDispatch = struct {
    pipeline_q4k: ?Pipeline,
    pipeline_q8_0: ?Pipeline,
    pipeline_f16: ?Pipeline,
    descriptor_pool: vk.c.VkDescriptorPool,
    device: vk.c.VkDevice,

    /// Create the DMMV dispatch wrapper and load the supported quantized pipelines.
    /// @param instance Active Vulkan instance and logical device.
    /// @param gpu_config Derived GPU tuning parameters.
    /// @param shader_dir Directory containing compiled SPIR-V shader binaries.
    /// @param allocator Allocator used for temporary pipeline creation state.
    /// @returns A DmmvDispatch ready to record projection work.
    pub fn init(
        instance: *const Instance,
        gpu_config: *const GpuConfig,
        shader_dir: []const u8,
        allocator: std.mem.Allocator,
    ) !DmmvDispatch {
        _ = gpu_config;

        // Create descriptor pool
        const pool_size = vk.c.VkDescriptorPoolSize{
            .type = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3 * 3, // 3 buffers per pipeline * 3 pipelines
        };
        const pool_info = vk.c.VkDescriptorPoolCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.c.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 16,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        var descriptor_pool: vk.c.VkDescriptorPool = null;
        const result = vk.c.vkCreateDescriptorPool(instance.device, &pool_info, null, &descriptor_pool);
        if (result != vk.c.VK_SUCCESS) return error.DescriptorPoolCreateFailed;

        // Load pipelines (3 bindings: A matrix, x vector, y output)
        const push_size = @sizeOf(DmmvPushConstants);

        var path_buf: [512]u8 = undefined;

        const q4k_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q4k.spv", .{shader_dir}) catch unreachable;
        const pipeline_q4k = pipeline_mod.createFromSpirv(instance, q4k_path, 3, push_size, &.{}, allocator) catch |err| blk: {
            log.warn("Q4_K shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const q8_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_q8_0.spv", .{shader_dir}) catch unreachable;
        const pipeline_q8_0 = pipeline_mod.createFromSpirv(instance, q8_path, 3, push_size, &.{}, allocator) catch |err| blk: {
            log.warn("Q8_0 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        const f16_path = std.fmt.bufPrint(&path_buf, "{s}/dmmv_f16.spv", .{shader_dir}) catch unreachable;
        const pipeline_f16 = pipeline_mod.createFromSpirv(instance, f16_path, 3, push_size, &.{}, allocator) catch |err| blk: {
            log.warn("F16 shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        return DmmvDispatch{
            .pipeline_q4k = pipeline_q4k,
            .pipeline_q8_0 = pipeline_q8_0,
            .pipeline_f16 = pipeline_f16,
            .descriptor_pool = descriptor_pool,
            .device = instance.device,
        };
    }

    /// Select the appropriate pipeline for a quantization type.
    pub fn pipelineForType(self: *const DmmvDispatch, quant_type: GGMLType) ?*const Pipeline {
        return switch (quant_type) {
            .q4_k => if (self.pipeline_q4k) |*p| p else null,
            .q8_0 => if (self.pipeline_q8_0) |*p| p else null,
            .f16 => if (self.pipeline_f16) |*p| p else null,
            else => null,
        };
    }

    /// Record a DMMV dispatch into a command buffer.
    pub fn recordDispatch(
        self: *const DmmvDispatch,
        cmd: *const CommandBuffer,
        quant_type: GGMLType,
        descriptor_set: vk.c.VkDescriptorSet,
        M: u32,
        K: u32,
        a_offset: u32,
        x_offset: u32,
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

        // 2 rows per workgroup → ceil(M/2) workgroups
        const workgroups_x = (M + 1) / 2;

        cmd.dispatchWithPush(
            pip,
            descriptor_set,
            std.mem.asBytes(&push),
            workgroups_x,
            1,
            1,
        );
    }

    /// Destroy the loaded pipelines and descriptor pool.
    /// @param self Dispatch wrapper to tear down in place.
    pub fn deinit(self: *DmmvDispatch) void {
        if (self.pipeline_q4k) |*p| p.deinit();
        if (self.pipeline_q8_0) |*p| p.deinit();
        if (self.pipeline_f16) |*p| p.deinit();
        vk.c.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.* = undefined;
    }
};
