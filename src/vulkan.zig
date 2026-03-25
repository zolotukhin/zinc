const std = @import("std");
const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});

pub const Device = struct {
    name: [256]u8,
    name_len: usize,
    vendor_id: u32,
    device_type: c.VkPhysicalDeviceType,
    vram_bytes: u64,
    compute_queue_family: u32,
    coopmat_support: bool,
    subgroup_size: u32,

    pub fn nameSlice(self: *const Device) []const u8 {
        return self.name[0..self.name_len];
    }
};

var instance: c.VkInstance = null;
var devices: std.ArrayList(Device) = .{};
var devices_allocator: std.mem.Allocator = undefined;

pub fn init(allocator: std.mem.Allocator) !void {
    devices = .{};
    devices_allocator = allocator;

    // Create Vulkan instance
    const app_info = c.VkApplicationInfo{
        .sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        .pNext = null,
        .pApplicationName = "zinc",
        .applicationVersion = c.VK_MAKE_VERSION(0, 1, 0),
        .pEngineName = "zinc",
        .engineVersion = c.VK_MAKE_VERSION(0, 1, 0),
        .apiVersion = c.VK_API_VERSION_1_2,
    };

    const create_info = c.VkInstanceCreateInfo{
        .sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .pApplicationInfo = &app_info,
        .enabledLayerCount = 0,
        .ppEnabledLayerNames = null,
        .enabledExtensionCount = 0,
        .ppEnabledExtensionNames = null,
    };

    const result = c.vkCreateInstance(&create_info, null, &instance);
    if (result != c.VK_SUCCESS) {
        std.log.err("Failed to create Vulkan instance: {}", .{result});
        return error.VulkanInitFailed;
    }

    // Enumerate physical devices
    var dev_count: u32 = 0;
    _ = c.vkEnumeratePhysicalDevices(instance, &dev_count, null);

    if (dev_count == 0) {
        std.log.err("No Vulkan devices found", .{});
        return error.NoDevices;
    }

    const phys_devices = try allocator.alloc(c.VkPhysicalDevice, dev_count);
    defer allocator.free(phys_devices);
    _ = c.vkEnumeratePhysicalDevices(instance, &dev_count, phys_devices.ptr);

    for (phys_devices[0..dev_count], 0..) |pdev, i| {
        var props: c.VkPhysicalDeviceProperties = undefined;
        c.vkGetPhysicalDeviceProperties(pdev, &props);

        var dev = Device{
            .name = undefined,
            .name_len = 0,
            .vendor_id = props.vendorID,
            .device_type = props.deviceType,
            .vram_bytes = 0,
            .compute_queue_family = 0,
            .coopmat_support = false,
            .subgroup_size = 0,
        };

        // Copy device name
        const name_slice = std.mem.sliceTo(&props.deviceName, 0);
        dev.name_len = @min(name_slice.len, dev.name.len);
        @memcpy(dev.name[0..dev.name_len], name_slice[0..dev.name_len]);

        // Find compute queue family
        var qf_count: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, null);
        const qf_props = try allocator.alloc(c.VkQueueFamilyProperties, qf_count);
        defer allocator.free(qf_props);
        c.vkGetPhysicalDeviceQueueFamilyProperties(pdev, &qf_count, qf_props.ptr);

        for (qf_props[0..qf_count], 0..) |qf, qi| {
            if (qf.queueFlags & c.VK_QUEUE_COMPUTE_BIT != 0) {
                dev.compute_queue_family = @intCast(qi);
                break;
            }
        }

        // Get VRAM size
        var mem_props: c.VkPhysicalDeviceMemoryProperties = undefined;
        c.vkGetPhysicalDeviceMemoryProperties(pdev, &mem_props);
        for (mem_props.memoryHeaps[0..mem_props.memoryHeapCount]) |heap| {
            if (heap.flags & c.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0) {
                dev.vram_bytes = heap.size;
            }
        }

        std.log.info("GPU {d}: {s} | VRAM: {d} MB | vendor: 0x{x:0>4}", .{
            i,
            dev.nameSlice(),
            dev.vram_bytes / (1024 * 1024),
            dev.vendor_id,
        });

        try devices.append(allocator, dev);
    }
}

pub fn deinit() void {
    devices.deinit(devices_allocator);
    if (instance != null) {
        c.vkDestroyInstance(instance, null);
    }
}
