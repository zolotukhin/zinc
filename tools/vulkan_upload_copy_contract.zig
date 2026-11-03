//! CPU-only fault injection for the real upload primitive. These executable
//! exports replace every Vulkan entrypoint it uses; no instance/device is opened.
const std = @import("std");
const host_io = @import("host_io");
const vulkan = @import("vulkan");
const vk = vulkan.vk.c;
const buffer = vulkan.buffer;
const Instance = vulkan.Instance;

var mode: []const u8 = "success";
var pending = false;
var region: vk.VkBufferCopy = undefined;
var source: [32]u8 = undefined;
var destination: [64]u8 = @splat(0xA5);

fn event(name: []const u8) void {
    std.debug.print("event={s}\n", .{name});
}

fn cleanup(name: []const u8) void {
    event(name);
    if (pending) {
        event("unsafe-pending-cleanup");
        std.process.exit(97);
    }
}

pub fn main(process: std.process.Init) !void {
    host_io.init(process);
    const args = try host_io.argsAlloc(std.heap.page_allocator);
    defer host_io.argsFree(std.heap.page_allocator, args);
    if (args.len != 2) return error.ExpectedFailureMode;
    mode = args[1];
    for (&source, 0..) |*byte, index| byte.* = @truncate(index * 17);
    var instance: Instance = undefined;
    instance.device = @ptrFromInt(1);
    instance.compute_queue = @ptrFromInt(2);
    var src: buffer.Buffer = .{
        .handle = @ptrFromInt(3),
        .memory = @ptrFromInt(4),
        .size = source.len,
        .mapped = &source,
        .device = instance.device,
    };
    defer src.deinit();
    var dst: buffer.Buffer = .{
        .handle = @ptrFromInt(5),
        .memory = @ptrFromInt(6),
        .size = destination.len,
        .mapped = null,
        .device = instance.device,
    };
    defer dst.deinit();
    buffer.copyBufferRegion(&instance, @ptrFromInt(7), &src, &dst, 3, 7, 17) catch |err| {
        std.debug.print("result={s}\n", .{@errorName(err)});
        return;
    };
    // Guards verify the real primitive's offsets and exact size reach Vulkan.
    if (!std.mem.eql(u8, destination[7..24], source[3..20])) return error.CopyBytesDiffer;
    for (destination[0..7]) |byte| {
        if (byte != 0xA5) return error.CopyGuardChanged;
    }
    for (destination[24..]) |byte| {
        if (byte != 0xA5) return error.CopyGuardChanged;
    }
    event("verified-exact-copy-and-guards");
    std.debug.print("result=success\n", .{});
}

export fn vkAllocateCommandBuffers(_: vk.VkDevice, _: [*c]const vk.VkCommandBufferAllocateInfo, commands: [*c]vk.VkCommandBuffer) vk.VkResult {
    event("allocate");
    if (std.mem.eql(u8, mode, "allocate")) return vk.VK_ERROR_OUT_OF_HOST_MEMORY;
    commands[0] = @ptrFromInt(8);
    return vk.VK_SUCCESS;
}

export fn vkFreeCommandBuffers(_: vk.VkDevice, _: vk.VkCommandPool, _: u32, _: [*c]const vk.VkCommandBuffer) void {
    cleanup("free-command");
}

export fn vkBeginCommandBuffer(_: vk.VkCommandBuffer, _: [*c]const vk.VkCommandBufferBeginInfo) vk.VkResult {
    event("begin");
    return if (std.mem.eql(u8, mode, "begin")) vk.VK_ERROR_OUT_OF_HOST_MEMORY else vk.VK_SUCCESS;
}

export fn vkCmdCopyBuffer(_: vk.VkCommandBuffer, _: vk.VkBuffer, _: vk.VkBuffer, count: u32, regions: [*c]const vk.VkBufferCopy) void {
    event("record-copy");
    std.debug.assert(count == 1);
    region = regions[0];
}

export fn vkEndCommandBuffer(_: vk.VkCommandBuffer) vk.VkResult {
    event("end");
    return if (std.mem.eql(u8, mode, "end")) vk.VK_ERROR_OUT_OF_HOST_MEMORY else vk.VK_SUCCESS;
}

export fn vkQueueSubmit(_: vk.VkQueue, _: u32, _: [*c]const vk.VkSubmitInfo, _: vk.VkFence) vk.VkResult {
    event("submit");
    if (std.mem.eql(u8, mode, "submit")) return vk.VK_ERROR_OUT_OF_HOST_MEMORY;
    pending = true;
    return vk.VK_SUCCESS;
}

export fn vkQueueWaitIdle(_: vk.VkQueue) vk.VkResult {
    event("wait");
    const failures = .{
        .{ "wait-host-oom", vk.VK_ERROR_OUT_OF_HOST_MEMORY },
        .{ "wait-device-oom", vk.VK_ERROR_OUT_OF_DEVICE_MEMORY },
        .{ "wait-device-lost", vk.VK_ERROR_DEVICE_LOST },
        .{ "wait-unknown", vk.VK_ERROR_UNKNOWN },
    };
    inline for (failures) |failure| {
        if (std.mem.eql(u8, mode, failure[0])) return failure[1];
    }
    @memcpy(destination[region.dstOffset..][0..region.size], source[region.srcOffset..][0..region.size]);
    pending = false;
    return vk.VK_SUCCESS;
}

export fn vkUnmapMemory(_: vk.VkDevice, _: vk.VkDeviceMemory) void {
    cleanup("unmap");
}

export fn vkDestroyBuffer(_: vk.VkDevice, _: vk.VkBuffer, _: [*c]const vk.VkAllocationCallbacks) void {
    cleanup("destroy-buffer");
}

export fn vkFreeMemory(_: vk.VkDevice, _: vk.VkDeviceMemory, _: [*c]const vk.VkAllocationCallbacks) void {
    cleanup("free-memory");
}
