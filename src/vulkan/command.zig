//! Create reusable compute command pools and command buffers.
//! @section Vulkan Runtime
//! The decode runtime uses these wrappers to record dispatches, insert barriers,
//! and synchronize submitted compute work.
const std = @import("std");
const vk = @import("vk.zig");
const Instance = @import("instance.zig").Instance;
const Pipeline = @import("pipeline.zig").Pipeline;

const log = std.log.scoped(.command);

const MemoryBarrierSpec = struct {
    src_stage: vk.c.VkPipelineStageFlags,
    dst_stage: vk.c.VkPipelineStageFlags,
    src_access: vk.c.VkAccessFlags,
    dst_access: vk.c.VkAccessFlags,
};

fn computeBarrierSpec() MemoryBarrierSpec {
    return .{
        .src_stage = vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .dst_stage = vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .src_access = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
        .dst_access = vk.c.VK_ACCESS_SHADER_READ_BIT,
    };
}

fn computeToTransferBarrierSpec() MemoryBarrierSpec {
    return .{
        .src_stage = vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .dst_stage = vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
        .src_access = vk.c.VK_ACCESS_SHADER_WRITE_BIT,
        .dst_access = vk.c.VK_ACCESS_TRANSFER_READ_BIT | vk.c.VK_ACCESS_TRANSFER_WRITE_BIT,
    };
}

fn transferToComputeBarrierSpec() MemoryBarrierSpec {
    return .{
        .src_stage = vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT,
        .dst_stage = vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        .src_access = vk.c.VK_ACCESS_TRANSFER_WRITE_BIT,
        .dst_access = vk.c.VK_ACCESS_SHADER_READ_BIT,
    };
}

/// Command pool for allocating command buffers.
pub const CommandPool = struct {
    /// Vulkan handle.
    handle: vk.c.VkCommandPool,
    /// Logical device.
    device: vk.c.VkDevice,

    /// Create a command pool bound to the selected compute queue family.
    /// @param instance Active Vulkan instance and logical device.
    /// @returns A CommandPool ready to allocate compute command buffers.
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

    /// Destroy the underlying Vulkan command pool.
    /// @param self Command pool to tear down in place.
    pub fn deinit(self: *CommandPool) void {
        vk.c.vkDestroyCommandPool(self.device, self.handle, null);
        self.* = undefined;
    }
};

/// A recorded command buffer that can be submitted and replayed.
pub const CommandBuffer = struct {
    /// Vulkan handle.
    handle: vk.c.VkCommandBuffer,
    /// Fence for command buffer completion sync.
    fence: vk.c.VkFence,
    /// Logical device.
    device: vk.c.VkDevice,

    /// Allocate a primary command buffer and fence from a compute command pool.
    /// @param instance Active Vulkan instance and logical device.
    /// @param pool Command pool used for command buffer allocation.
    /// @returns A CommandBuffer paired with a completion fence.
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

    /// Begin recording a reusable command buffer.
    /// @param self Command buffer to begin recording into.
    /// @returns `error.BeginCommandBufferFailed` when Vulkan rejects the begin request.
    /// @note Use `reset()` or wait for prior submissions before recording into the same buffer again.
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

    /// Begin recording for a single submit-and-discard style workload.
    /// @param self Command buffer to begin recording into.
    /// @returns `error.BeginCommandBufferFailed` when Vulkan rejects the begin request.
    /// @note This sets `VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT` so the driver can optimize transient work.
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

    /// Record a compute dispatch with an already-created descriptor set.
    /// @param self Command buffer currently being recorded.
    /// @param pipeline Compute pipeline to bind before dispatch.
    /// @param descriptor_set Descriptor set bound at set `0`.
    /// @param group_count_x Workgroup count in the X dimension.
    /// @param group_count_y Workgroup count in the Y dimension.
    /// @param group_count_z Workgroup count in the Z dimension.
    /// @note This helper binds pipeline and descriptors only; required barriers must be recorded separately.
    pub fn dispatch(
        self: *const CommandBuffer,
        /// Vulkan compute pipeline, or null if unavailable.
        pipeline: *const Pipeline,
        /// Allocated descriptor set.
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

    /// Record a compute dispatch that also uploads a serialized push-constant block.
    /// @param self Command buffer currently being recorded.
    /// @param pipeline Compute pipeline to bind before dispatch.
    /// @param descriptor_set Descriptor set bound at set `0`.
    /// @param push_data Raw bytes copied into the pipeline's push-constant range at offset `0`.
    /// @param group_count_x Workgroup count in the X dimension.
    /// @param group_count_y Workgroup count in the Y dimension.
    /// @param group_count_z Workgroup count in the Z dimension.
    /// @note The caller is responsible for matching `push_data` to the shader layout declared by `pipeline`.
    pub fn dispatchWithPush(
        self: *const CommandBuffer,
        /// Vulkan compute pipeline, or null if unavailable.
        pipeline: *const Pipeline,
        /// Allocated descriptor set.
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

    fn recordMemoryBarrier(self: *const CommandBuffer, spec: MemoryBarrierSpec) void {
        const barrier = vk.c.VkMemoryBarrier{
            .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_BARRIER,
            .pNext = null,
            .srcAccessMask = spec.src_access,
            .dstAccessMask = spec.dst_access,
        };
        vk.c.vkCmdPipelineBarrier(
            self.handle,
            spec.src_stage,
            spec.dst_stage,
            0,
            1,
            &barrier,
            0,
            null,
            0,
            null,
        );
    }

    /// Insert a compute-to-compute pipeline barrier (shader write → shader read).
    /// @param self Command buffer currently being recorded.
    /// @note Uses a coarse global memory barrier; prefer buffer barriers for fine-grained sync.
    pub fn computeBarrier(self: *const CommandBuffer) void {
        self.recordMemoryBarrier(computeBarrierSpec());
    }

    /// Insert a compute-to-transfer pipeline barrier (shader write → transfer read/write).
    /// @param self Command buffer currently being recorded.
    /// @note Use this before copying from or into buffers produced by compute shaders.
    pub fn computeToTransferBarrier(self: *const CommandBuffer) void {
        self.recordMemoryBarrier(computeToTransferBarrierSpec());
    }

    /// Insert a transfer-to-compute pipeline barrier (copy/upload → shader read).
    /// @param self Command buffer currently being recorded.
    /// @note Ensures prior transfer writes are visible before subsequent compute dispatches.
    pub fn transferToComputeBarrier(self: *const CommandBuffer) void {
        self.recordMemoryBarrier(transferToComputeBarrierSpec());
    }

    /// Finalize command recording so the buffer can be submitted.
    /// @param self Command buffer to finalize.
    /// @returns `error.EndCommandBufferFailed` when Vulkan rejects the recorded command stream.
    pub fn end(self: *const CommandBuffer) !void {
        const result = vk.c.vkEndCommandBuffer(self.handle);
        if (result != vk.c.VK_SUCCESS) return error.EndCommandBufferFailed;
    }

    /// Submit the command buffer and block until the GPU signals completion.
    /// @param self Recorded command buffer to submit.
    /// @param queue Queue to submit the work on.
    /// @returns `error.QueueSubmitFailed` or `error.FenceWaitFailed` when submission or synchronization fails.
    /// @note The fence is reset before returning so the command buffer can be reused by a later step.
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

    /// Submit recorded work and return immediately.
    /// @param self Recorded command buffer to submit.
    /// @param queue Queue to submit the work on.
    /// @returns `error.QueueSubmitFailed` when Vulkan rejects the submission.
    /// @note Pair this with `waitForCompletion()` before resetting or re-recording the command buffer.
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

    /// Wait for the command buffer's fence to signal and then reset it.
    /// @param self Command buffer whose most recent submission should complete before returning.
    /// @returns `error.FenceWaitFailed` when the wait operation fails.
    /// @note After this returns, the command buffer can be reset and recorded again.
    pub fn waitForCompletion(self: *const CommandBuffer) !void {
        const result = vk.c.vkWaitForFences(self.device, 1, &self.fence, vk.c.VK_TRUE, std.math.maxInt(u64));
        if (result != vk.c.VK_SUCCESS) return error.FenceWaitFailed;
        _ = vk.c.vkResetFences(self.device, 1, &self.fence);
    }

    /// Reset the command buffer so new commands can be recorded into it.
    /// @param self Command buffer to reset.
    /// @returns `error.ResetCommandBufferFailed` when Vulkan rejects the reset request.
    /// @note The caller must ensure the previous submission has completed before calling this.
    pub fn reset(self: *const CommandBuffer) !void {
        const result = vk.c.vkResetCommandBuffer(self.handle, 0);
        if (result != vk.c.VK_SUCCESS) return error.ResetCommandBufferFailed;
    }

    /// Destroy the command buffer fence and free the command buffer back to its pool.
    /// @param self Command buffer to tear down in place.
    /// @param pool Command pool that owns the Vulkan command buffer allocation.
    /// @note Callers should ensure the GPU is no longer using the buffer before teardown.
    pub fn deinit(self: *CommandBuffer, pool: *const CommandPool) void {
        vk.c.vkDestroyFence(self.device, self.fence, null);
        vk.c.vkFreeCommandBuffers(self.device, pool.handle, 1, &self.handle);
        self.* = undefined;
    }
};

test "compute barrier spec stays shader-write to shader-read" {
    const spec = computeBarrierSpec();
    try std.testing.expectEqual(@as(@TypeOf(spec.src_stage), vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT), spec.src_stage);
    try std.testing.expectEqual(@as(@TypeOf(spec.dst_stage), vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT), spec.dst_stage);
    try std.testing.expectEqual(@as(@TypeOf(spec.src_access), vk.c.VK_ACCESS_SHADER_WRITE_BIT), spec.src_access);
    try std.testing.expectEqual(@as(@TypeOf(spec.dst_access), vk.c.VK_ACCESS_SHADER_READ_BIT), spec.dst_access);
}

test "compute-to-transfer barrier spec exposes shader writes to copies" {
    const spec = computeToTransferBarrierSpec();
    try std.testing.expectEqual(@as(@TypeOf(spec.src_stage), vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT), spec.src_stage);
    try std.testing.expectEqual(@as(@TypeOf(spec.dst_stage), vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT), spec.dst_stage);
    try std.testing.expectEqual(@as(@TypeOf(spec.src_access), vk.c.VK_ACCESS_SHADER_WRITE_BIT), spec.src_access);
    try std.testing.expectEqual(@as(@TypeOf(spec.dst_access), vk.c.VK_ACCESS_TRANSFER_READ_BIT | vk.c.VK_ACCESS_TRANSFER_WRITE_BIT), spec.dst_access);
}

test "transfer-to-compute barrier spec exposes copy writes to shaders" {
    const spec = transferToComputeBarrierSpec();
    try std.testing.expectEqual(@as(@TypeOf(spec.src_stage), vk.c.VK_PIPELINE_STAGE_TRANSFER_BIT), spec.src_stage);
    try std.testing.expectEqual(@as(@TypeOf(spec.dst_stage), vk.c.VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT), spec.dst_stage);
    try std.testing.expectEqual(@as(@TypeOf(spec.src_access), vk.c.VK_ACCESS_TRANSFER_WRITE_BIT), spec.src_access);
    try std.testing.expectEqual(@as(@TypeOf(spec.dst_access), vk.c.VK_ACCESS_SHADER_READ_BIT), spec.dst_access);
}
