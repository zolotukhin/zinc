//! Wrap the GPU argmax reduction used for greedy token sampling.
//! @section Sampling
//! This helper owns the compute pipeline for the two-phase argmax shader and
//! records the reduction dispatches that pick the next token entirely on GPU.
const std = @import("std");
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const Pipeline = @import("../vulkan/pipeline.zig").Pipeline;
const pipeline_mod = @import("../vulkan/pipeline.zig");
const CommandBuffer = @import("../vulkan/command.zig").CommandBuffer;

const log = std.log.scoped(.argmax);

const ArgmaxPush = extern struct {
    N: u32,
    phase: u32,
};

/// GPU-accelerated two-phase argmax reduction for greedy token sampling.
pub const ArgmaxDispatch = struct {
    pipeline: ?Pipeline,
    descriptor_pool: vk.c.VkDescriptorPool,
    device: vk.c.VkDevice,

    /// Create the argmax compute pipeline and descriptor pool on the given Vulkan instance.
    pub fn init(
        instance: *const Instance,
        shader_dir: []const u8,
        allocator: std.mem.Allocator,
    ) !ArgmaxDispatch {
        const pool_size = vk.c.VkDescriptorPoolSize{
            .type = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3 * 4,
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
        const pool_result = vk.c.vkCreateDescriptorPool(instance.device, &pool_info, null, &descriptor_pool);
        if (pool_result != vk.c.VK_SUCCESS) return error.DescriptorPoolCreateFailed;

        var path_buf: [512]u8 = undefined;
        const argmax_path = std.fmt.bufPrint(&path_buf, "{s}/argmax.spv", .{shader_dir}) catch unreachable;
        const pipeline = pipeline_mod.createFromSpirv(instance, argmax_path, 3, @sizeOf(ArgmaxPush), &.{}, allocator) catch |err| blk: {
            log.warn("argmax shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        return .{
            .pipeline = pipeline,
            .descriptor_pool = descriptor_pool,
            .device = instance.device,
        };
    }

    /// Record the two-phase argmax reduction into a command buffer.
    pub fn record(
        self: *const ArgmaxDispatch,
        cmd: *const CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        n_logits: u32,
        phase0_workgroups: u32,
    ) !void {
        const pip = if (self.pipeline) |*p| p else return error.ShaderNotLoaded;

        const phase0 = ArgmaxPush{
            .N = n_logits,
            .phase = 0,
        };
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&phase0), phase0_workgroups, 1, 1);
        cmd.computeBarrier();

        const phase1 = ArgmaxPush{
            .N = phase0_workgroups,
            .phase = 1,
        };
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&phase1), 1, 1, 1);
    }

    /// Allocate a descriptor set from the argmax descriptor pool.
    pub fn allocDescriptorSet(self: *const ArgmaxDispatch) !vk.c.VkDescriptorSet {
        const pip = if (self.pipeline) |*p| p else return error.ShaderNotLoaded;
        const alloc_info = vk.c.VkDescriptorSetAllocateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
            .pNext = null,
            .descriptorPool = self.descriptor_pool,
            .descriptorSetCount = 1,
            .pSetLayouts = &pip.descriptor_set_layout,
        };
        var ds: vk.c.VkDescriptorSet = null;
        const result = vk.c.vkAllocateDescriptorSets(self.device, &alloc_info, &ds);
        if (result != vk.c.VK_SUCCESS) return error.DescriptorSetAllocFailed;
        return ds;
    }

    /// Bind the logits, partials, and result buffers to a descriptor set.
    pub fn writeDescriptorSet(
        self: *const ArgmaxDispatch,
        descriptor_set: vk.c.VkDescriptorSet,
        logits_buf: vk.c.VkBuffer,
        logits_size: vk.c.VkDeviceSize,
        partials_buf: vk.c.VkBuffer,
        partials_size: vk.c.VkDeviceSize,
        result_buf: vk.c.VkBuffer,
        result_size: vk.c.VkDeviceSize,
    ) void {
        var buffer_infos = [3]vk.c.VkDescriptorBufferInfo{
            .{ .buffer = logits_buf, .offset = 0, .range = logits_size },
            .{ .buffer = partials_buf, .offset = 0, .range = partials_size },
            .{ .buffer = result_buf, .offset = 0, .range = result_size },
        };
        var writes: [3]vk.c.VkWriteDescriptorSet = undefined;
        for (0..3) |i| {
            writes[i] = .{
                .sType = vk.c.VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
                .pNext = null,
                .dstSet = descriptor_set,
                .dstBinding = @intCast(i),
                .dstArrayElement = 0,
                .descriptorCount = 1,
                .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pImageInfo = null,
                .pBufferInfo = &buffer_infos[i],
                .pTexelBufferView = null,
            };
        }
        vk.c.vkUpdateDescriptorSets(self.device, writes.len, &writes, 0, null);
    }

    /// Destroy the pipeline and descriptor pool.
    pub fn deinit(self: *ArgmaxDispatch) void {
        if (self.pipeline) |*p| p.deinit();
        vk.c.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.* = undefined;
    }
};
