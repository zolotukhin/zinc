const std = @import("std");
const vk = @import("vk.zig");
const Instance = @import("instance.zig").Instance;
const Pipeline = @import("pipeline.zig").Pipeline;

const log = std.log.scoped(.command);

/// Command pool for allocating command buffers.
pub const CommandPool = struct {
    handle: vk.c.VkCommandPool,
    device: vk.c.VkDevice,

    pub fn init(instance: *const Instance) !CommandPool {
        const pool_info = vk.c.VkCommandPoolCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
            .queueFamilyIndex = instance.compute_queue_family,
        };

        var handle: vk.c.VkCommandPool = null;
        const result = vk.c.vkCreateCommandPool(instance.device, &pool_info, null, &handle);
        if (result != vk.c.VK_SUCCESS) {
            log.err("vkCreateCommandPool failed: {d}", .{result});
            return error.CommandPoolCreateFailed;
        }

        return CommandPool{
            .handle = handle,
            .device = instance.device,
        };
    }

    pub fn deinit(self: *CommandPool) void {
        vk.c.vkDestroyCommandPool(self.device, self.handle, null);
        self.* = undefined;
    }
};

/// A recorded command buffer that can be submitted and replayed.
pub const CommandBuffer = struct {
    handle: vk.c.VkCommandBuffer,
    fence: vk.c.VkFence,
    device: vk.c.VkDevice,

    pub fn init(instance: *const Instance, pool: *const CommandPool) !CommandBuffer {
        const alloc_info = vk.c.VkCommandBufferAllocateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
            .pNext = null,
            .commandPool = pool.handle,
            .level = vk.c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
            .commandBufferCount = 1,
        };

        var handle: vk.c.VkCommandBuffer = null;
        var result = vk.c.vkAllocateCommandBuffers(instance.device, &alloc_info, &handle);
        if (result != vk.c.VK_SUCCESS) {
            log.err("vkAllocateCommandBuffers failed: {d}", .{result});
            return error.CommandBufferAllocFailed;
        }

        const fence_info = vk.c.VkFenceCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_FENCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
        };

        var fence: vk.c.VkFence = null;
        result = vk.c.vkCreateFence(instance.device, &fence_info, null, &fence);
        if (result != vk.c.VK_SUCCESS) {
            log.err("vkCreateFence failed: {d}", .{result});
            return error.FenceCreateFailed;
        }

        return CommandBuffer{
            .handle = handle,
            .fence = fence,
            .device = instance.device,
        };
    }

    /// Begin recording commands.
    pub fn begin(self: *const CommandBuffer) !void {
        const begin_info = vk.c.VkCommandBufferBeginInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = 0,
            .pInheritanceInfo = null,
        };
        const result = vk.c.vkBeginCommandBuffer(self.handle, &begin_info);
        if (result != vk.c.VK_SUCCESS) return error.BeginCommandBufferFailed;
    }

    /// Begin recording for one-time submission.
    pub fn beginOneTime(self: *const CommandBuffer) !void {
        const begin_info = vk.c.VkCommandBufferBeginInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
            .pNext = null,
            .flags = vk.c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
            .pInheritanceInfo = null,
        };
        const result = vk.c.vkBeginCommandBuffer(self.handle, &begin_info);
        if (result != vk.c.VK_SUCCESS) return error.BeginCommandBufferFailed;
    }

    /// Record a compute dispatch.
    pub fn dispatch(
        self: *const CommandBuffer,
        pipeline: *const Pipeline,
        descriptor_set: vk.c.VkDescriptorSet,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) void {
        vk.c.vkCmdBindPipeline(self.handle, vk.c.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        vk.c.vkCmdBindDescriptorSets(
            self.handle,
            vk.c.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline_layout,
            0,
            1,
            &descriptor_set,
            0,
            null,
        );
        vk.c.vkCmdDispatch(self.handle, group_count_x, group_count_y, group_count_z);
    }

    /// Record a compute dispatch with push constants.
    pub fn dispatchWithPush(
        self: *const CommandBuffer,
        pipeline: *const Pipeline,
        descriptor_set: vk.c.VkDescriptorSet,
        push_data: []const u8,
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    ) void {
        vk.c.vkCmdBindPipeline(self.handle, vk.c.VK_PIPELINE_BIND_POINT_COMPUTE, pipeline.pipeline);
        vk.c.vkCmdPushConstants(
            self.handle,
            pipeline.pipeline_layout,
            vk.c.VK_SHADER_STAGE_COMPUTE_BIT,
            0,
            @intCast(push_data.len),
            push_data.ptr,
        );
        vk.c.vkCmdBindDescriptorSets(
            self.handle,
            vk.c.VK_PIPELINE_BIND_POINT_COMPUTE,
            pipeline.pipeline_layout,
            0,
            1,
            &descriptor_set,
            0,
            null,
        );
        vk.c.vkCmdDispatch(self.handle, group_count_x, group_count_y, group_count_z);
    }

    /// Record a pipeline barrier (compute → compute).
    pub fn computeBarrier(self: *const CommandBuffer) void {
        const barrier = vk.c.VkMemoryBarrier{
            .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
            .dstAccessMask = vk.c.VK_ACCESS_SHADER_READ_BIT,
        };
        vk.c.vkCmdPipelineBarrier(
            self.handle,
            vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            0,
            1,
            &barrier,
            0,
            null,
            0,
            null,
        );
    }

    /// End recording.
    pub fn end(self: *const CommandBuffer) !void {
        const result = vk.c.vkEndCommandBuffer(self.handle);
        if (result != vk.c.VK_SUCCESS) return error.EndCommandBufferFailed;
    }

    /// Submit the command buffer and wait for completion.
    pub fn submitAndWait(self: *const CommandBuffer, queue: vk.c.VkQueue) !void {
        const submit_info = vk.c.VkSubmitInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.handle,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        };

        var result = vk.c.vkQueueSubmit(queue, 1, &submit_info, self.fence);
        if (result != vk.c.VK_SUCCESS) {
            log.err("vkQueueSubmit failed: {d}", .{result});
            return error.QueueSubmitFailed;
        }

        result = vk.c.vkWaitForFences(self.device, 1, &self.fence, vk.c.VK_TRUE, std.math.maxInt(u64));
        if (result != vk.c.VK_SUCCESS) return error.FenceWaitFailed;

        _ = vk.c.vkResetFences(self.device, 1, &self.fence);
    }

    /// Submit without waiting (for pre-recorded replay). Caller must manage synchronization.
    pub fn submit(self: *const CommandBuffer, queue: vk.c.VkQueue) !void {
        const submit_info = vk.c.VkSubmitInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
            .pNext = null,
            .waitSemaphoreCount = 0,
            .pWaitSemaphores = null,
            .pWaitDstStageMask = null,
            .commandBufferCount = 1,
            .pCommandBuffers = &self.handle,
            .signalSemaphoreCount = 0,
            .pSignalSemaphores = null,
        };

        const result = vk.c.vkQueueSubmit(queue, 1, &submit_info, self.fence);
        if (result != vk.c.VK_SUCCESS) {
            log.err("vkQueueSubmit failed: {d}", .{result});
            return error.QueueSubmitFailed;
        }
    }

    /// Wait for a previously submitted command buffer to complete.
    pub fn waitForCompletion(self: *const CommandBuffer) !void {
        const result = vk.c.vkWaitForFences(self.device, 1, &self.fence, vk.c.VK_TRUE, std.math.maxInt(u64));
        if (result != vk.c.VK_SUCCESS) return error.FenceWaitFailed;
        _ = vk.c.vkResetFences(self.device, 1, &self.fence);
    }

    /// Reset the command buffer for re-recording.
    pub fn reset(self: *const CommandBuffer) !void {
        const result = vk.c.vkResetCommandBuffer(self.handle, 0);
        if (result != vk.c.VK_SUCCESS) return error.ResetCommandBufferFailed;
    }

    pub fn deinit(self: *CommandBuffer, pool: *const CommandPool) void {
        vk.c.vkDestroyFence(self.device, self.fence, null);
        vk.c.vkFreeCommandBuffers(self.device, pool.handle, 1, &self.handle);
        self.* = undefined;
    }
};
