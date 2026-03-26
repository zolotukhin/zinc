//! Allocate Vulkan buffers used by weights, intermediates, and staging copies.
//! @section Vulkan Runtime
//! These helpers centralize buffer creation, memory mapping, and one-shot copy
//! utilities so the rest of the runtime can work with higher-level abstractions.
const std = @import("std");
const vk = @import("vk.zig");
const Instance = @import("instance.zig").Instance;

const log = std.log.scoped(.buffer);

/// Vulkan buffer allocation paired with its device memory and optional mapped pointer.
pub const Buffer = struct {
    handle: vk.c.VkBuffer,
    memory: vk.c.VkDeviceMemory,
    size: vk.c.VkDeviceSize,
    mapped: ?[*]u8,
    device: vk.c.VkDevice,

    /// Create a GPU buffer with the given usage and memory properties.
    pub fn init(
        instance: *const Instance,
        size: vk.c.VkDeviceSize,
        usage: vk.c.VkBufferUsageFlags,
        mem_properties: vk.c.VkMemoryPropertyFlags,
    ) !Buffer {
        const buf_info = vk.c.VkBufferCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .size = size,
            .usage = usage,
            .sharingMode = vk.c.VK_SHARING_MODE_EXCLUSIVE,
            .queueFamilyIndexCount = 0,
            .pQueueFamilyIndices = null,
        };

        var handle: vk.c.VkBuffer = null;
        var result = vk.c.vkCreateBuffer(instance.device, &buf_info, null, &handle);
        if (result != vk.c.VK_SUCCESS) {
            log.err("vkCreateBuffer failed: {d}", .{result});
            return error.BufferCreateFailed;
        }
        errdefer vk.c.vkDestroyBuffer(instance.device, handle, null);

        var mem_reqs: vk.c.VkMemoryRequirements = undefined;
        vk.c.vkGetBufferMemoryRequirements(instance.device, handle, &mem_reqs);

        const mem_type = instance.findMemoryType(mem_reqs.memoryTypeBits, mem_properties) orelse {
            log.err("No suitable memory type found", .{});
            return error.NoSuitableMemoryType;
        };

        const alloc_info = vk.c.VkMemoryAllocateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
            .pNext = null,
            .allocationSize = mem_reqs.size,
            .memoryTypeIndex = mem_type,
        };

        var memory: vk.c.VkDeviceMemory = null;
        result = vk.c.vkAllocateMemory(instance.device, &alloc_info, null, &memory);
        if (result != vk.c.VK_SUCCESS) {
            log.err("vkAllocateMemory failed: {d}", .{result});
            return error.MemoryAllocFailed;
        }
        errdefer vk.c.vkFreeMemory(instance.device, memory, null);

        result = vk.c.vkBindBufferMemory(instance.device, handle, memory, 0);
        if (result != vk.c.VK_SUCCESS) {
            log.err("vkBindBufferMemory failed: {d}", .{result});
            return error.BufferBindFailed;
        }

        return Buffer{
            .handle = handle,
            .memory = memory,
            .size = size,
            .mapped = null,
            .device = instance.device,
        };
    }

    /// Create a device-local buffer (GPU VRAM, not host-visible).
    pub fn initDeviceLocal(instance: *const Instance, size: vk.c.VkDeviceSize, usage: vk.c.VkBufferUsageFlags) !Buffer {
        return init(
            instance,
            size,
            usage | vk.c.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            vk.c.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
        );
    }

    /// Create a host-visible staging buffer for CPU→GPU transfers.
    pub fn initStaging(instance: *const Instance, size: vk.c.VkDeviceSize) !Buffer {
        var buf = try init(
            instance,
            size,
            vk.c.VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            vk.c.VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | vk.c.VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        );

        // Map staging buffers immediately
        var ptr: ?*anyopaque = null;
        const result = vk.c.vkMapMemory(instance.device, buf.memory, 0, size, 0, &ptr);
        if (result != vk.c.VK_SUCCESS) {
            log.err("vkMapMemory failed: {d}", .{result});
            buf.deinit();
            return error.MapMemoryFailed;
        }
        buf.mapped = @ptrCast(ptr);

        return buf;
    }

    /// Write data to a mapped staging buffer.
    pub fn upload(self: *const Buffer, data: []const u8) void {
        std.debug.assert(self.mapped != null);
        std.debug.assert(data.len <= self.size);
        @memcpy(self.mapped.?[0..data.len], data);
    }

    /// Destroy the Vulkan buffer, free memory, and unmap any mapped staging pointer.
    /// @param self Buffer to tear down in place.
    pub fn deinit(self: *Buffer) void {
        if (self.mapped != null) {
            vk.c.vkUnmapMemory(self.device, self.memory);
            self.mapped = null;
        }
        vk.c.vkDestroyBuffer(self.device, self.handle, null);
        vk.c.vkFreeMemory(self.device, self.memory, null);
        self.* = undefined;
    }
};

/// Record and execute a buffer-to-buffer copy via a one-shot command buffer.
pub fn copyBuffer(
    instance: *const Instance,
    cmd_pool: vk.c.VkCommandPool,
    src: *const Buffer,
    dst: *const Buffer,
    size: vk.c.VkDeviceSize,
) !void {
    // Allocate one-shot command buffer
    const alloc_info = vk.c.VkCommandBufferAllocateInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
        .pNext = null,
        .commandPool = cmd_pool,
        .level = vk.c.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
        .commandBufferCount = 1,
    };

    var cmd_buf: vk.c.VkCommandBuffer = null;
    var result = vk.c.vkAllocateCommandBuffers(instance.device, &alloc_info, &cmd_buf);
    if (result != vk.c.VK_SUCCESS) return error.AllocCmdBufFailed;

    const begin_info = vk.c.VkCommandBufferBeginInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
        .pNext = null,
        .flags = vk.c.VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT,
        .pInheritanceInfo = null,
    };
    _ = vk.c.vkBeginCommandBuffer(cmd_buf, &begin_info);

    const region = vk.c.VkBufferCopy{
        .srcOffset = 0,
        .dstOffset = 0,
        .size = size,
    };
    vk.c.vkCmdCopyBuffer(cmd_buf, src.handle, dst.handle, 1, &region);

    _ = vk.c.vkEndCommandBuffer(cmd_buf);

    const submit_info = vk.c.VkSubmitInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_SUBMIT_INFO,
        .pNext = null,
        .waitSemaphoreCount = 0,
        .pWaitSemaphores = null,
        .pWaitDstStageMask = null,
        .commandBufferCount = 1,
        .pCommandBuffers = &cmd_buf,
        .signalSemaphoreCount = 0,
        .pSignalSemaphores = null,
    };
    result = vk.c.vkQueueSubmit(instance.compute_queue, 1, &submit_info, @as(vk.c.VkFence, null));
    if (result != vk.c.VK_SUCCESS) return error.QueueSubmitFailed;

    _ = vk.c.vkQueueWaitIdle(instance.compute_queue);
    vk.c.vkFreeCommandBuffers(instance.device, cmd_pool, 1, &cmd_buf);
}
