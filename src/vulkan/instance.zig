//! Initialize Vulkan, select a compute-capable device, and expose memory utilities.
//! @section Vulkan Runtime
//! This is the entry point for GPU setup: instance creation, device selection,
//! queue discovery, and VRAM inspection.
const std = @import("std");
const vk = @import("vk.zig");

const log = std.log.scoped(.vulkan);

/// Active Vulkan instance, selected physical device, logical device, and memory metadata.
pub const Instance = struct {
    /// Vulkan handle.
    handle: vk.c.VkInstance,
    /// Selected physical device (GPU).
    physical_device: vk.c.VkPhysicalDevice,
    /// Logical device.
    device: vk.c.VkDevice,
    /// Compute queue for dispatch.
    compute_queue: vk.c.VkQueue,
    /// Compute queue family index.
    compute_queue_family: u32,
    /// Physical device properties.
    device_props: vk.c.VkPhysicalDeviceProperties,
    /// Device memory properties.
    mem_props: vk.c.VkPhysicalDeviceMemoryProperties,
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,

    /// Create a Vulkan instance and select a compute-capable device.
    /// @param allocator Allocator used for temporary device and queue enumeration state.
    /// @param preferred_device Preferred physical device index when multiple GPUs are present.
    /// @returns An initialized Instance bound to a logical device and compute queue.
    pub fn init(allocator: std.mem.Allocator, preferred_device: u32) !Instance {
        // Create Vulkan instance
        const app_info = vk.c.VkApplicationInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_APPLICATION_INFO,
            .pNext = null,
            .pApplicationName = "zinc",
            .applicationVersion = vk.c.VK_MAKE_VERSION(0, 1, 0),
            .pEngineName = "zinc",
            .engineVersion = vk.c.VK_MAKE_VERSION(0, 1, 0),
            .apiVersion = vk.c.VK_API_VERSION_1_3,
        };

        const create_info = vk.c.VkInstanceCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .pApplicationInfo = &app_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = null,
        };

        var instance: vk.c.VkInstance = null;
        var result = vk.c.vkCreateInstance(&create_info, null, &instance);
        if (result != vk.c.VK_SUCCESS) {
            log.err("vkCreateInstance failed: {d}", .{result});
            return error.VulkanInitFailed;
        }
        errdefer vk.c.vkDestroyInstance(instance, null);

        // Enumerate physical devices
        var dev_count: u32 = 0;
        _ = vk.c.vkEnumeratePhysicalDevices(instance, &dev_count, null);
        if (dev_count == 0) {
            log.err("No Vulkan physical devices found", .{});
            return error.NoDevices;
        }

        const phys_devices = try allocator.alloc(vk.c.VkPhysicalDevice, dev_count);
        defer allocator.free(phys_devices);
        _ = vk.c.vkEnumeratePhysicalDevices(instance, &dev_count, phys_devices.ptr);

        // Log all devices
        for (phys_devices[0..dev_count], 0..) |pdev, i| {
            var props: vk.c.VkPhysicalDeviceProperties = undefined;
            vk.c.vkGetPhysicalDeviceProperties(pdev, &props);
            const name = std.mem.sliceTo(&props.deviceName, 0);
            log.info("GPU {d}: {s} (vendor 0x{x:0>4})", .{ i, name, props.vendorID });
        }

        // Select device
        const dev_idx: u32 = if (preferred_device < dev_count) preferred_device else 0;
        const physical_device = phys_devices[dev_idx];

        var device_props: vk.c.VkPhysicalDeviceProperties = undefined;
        vk.c.vkGetPhysicalDeviceProperties(physical_device, &device_props);

        var mem_props: vk.c.VkPhysicalDeviceMemoryProperties = undefined;
        vk.c.vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

        const dev_name = std.mem.sliceTo(&device_props.deviceName, 0);
        log.info("Selected GPU {d}: {s}", .{ dev_idx, dev_name });

        // Find compute queue family
        var qf_count: u32 = 0;
        vk.c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count, null);
        const qf_props = try allocator.alloc(vk.c.VkQueueFamilyProperties, qf_count);
        defer allocator.free(qf_props);
        vk.c.vkGetPhysicalDeviceQueueFamilyProperties(physical_device, &qf_count, qf_props.ptr);

        var compute_family: ?u32 = null;
        // Prefer a compute-only queue (dedicated async compute)
        for (qf_props[0..qf_count], 0..) |qf, qi| {
            if (qf.queueFlags & vk.c.VK_QUEUE_COMPUTE_BIT != 0 and
                qf.queueFlags & vk.c.VK_QUEUE_GRAPHICS_BIT == 0)
            {
                compute_family = @intCast(qi);
                break;
            }
        }
        // Fall back to any queue with compute
        if (compute_family == null) {
            for (qf_props[0..qf_count], 0..) |qf, qi| {
                if (qf.queueFlags & vk.c.VK_QUEUE_COMPUTE_BIT != 0) {
                    compute_family = @intCast(qi);
                    break;
                }
            }
        }
        const compute_queue_family = compute_family orelse {
            log.err("No compute queue family found", .{});
            return error.NoComputeQueue;
        };

        // Create logical device with compute queue
        const queue_priority: f32 = 1.0;
        const queue_create_info = vk.c.VkDeviceQueueCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueFamilyIndex = compute_queue_family,
            .queueCount = 1,
            .pQueuePriorities = &queue_priority,
        };

        const device_create_info = vk.c.VkDeviceCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = null,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_create_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = 0,
            .ppEnabledExtensionNames = null,
            .pEnabledFeatures = null,
        };

        var device: vk.c.VkDevice = null;
        result = vk.c.vkCreateDevice(physical_device, &device_create_info, null, &device);
        if (result != vk.c.VK_SUCCESS) {
            log.err("vkCreateDevice failed: {d}", .{result});
            return error.DeviceCreateFailed;
        }
        errdefer vk.c.vkDestroyDevice(device, null);

        var compute_queue: vk.c.VkQueue = null;
        vk.c.vkGetDeviceQueue(device, compute_queue_family, 0, &compute_queue);

        log.info("Vulkan device ready — compute queue family {d}", .{compute_queue_family});

        return Instance{
            .handle = instance,
            .physical_device = physical_device,
            .device = device,
            .compute_queue = compute_queue,
            .compute_queue_family = compute_queue_family,
            .device_props = device_props,
            .mem_props = mem_props,
            .allocator = allocator,
        };
    }

    /// Wait for outstanding work, destroy the logical device, and destroy the Vulkan instance.
    /// @param self Vulkan instance wrapper to tear down in place.
    pub fn deinit(self: *Instance) void {
        _ = vk.c.vkDeviceWaitIdle(self.device);
        vk.c.vkDestroyDevice(self.device, null);
        vk.c.vkDestroyInstance(self.handle, null);
        self.* = undefined;
    }

    /// Find a Vulkan memory type that satisfies both compatibility and property requirements.
    /// @param self Active Vulkan instance and memory properties.
    /// @param type_filter Bitmask of compatible memory types reported by Vulkan.
    /// @param properties Required Vulkan memory property flags.
    /// @returns The matching memory type index, or `null` when no heap satisfies the request.
    /// @note All requested property bits must be present on the returned memory type.
    pub fn findMemoryType(self: *const Instance, type_filter: u32, properties: vk.c.VkMemoryPropertyFlags) ?u32 {
        for (0..self.mem_props.memoryTypeCount) |i| {
            const idx: u5 = @intCast(i);
            if (type_filter & (@as(u32, 1) << idx) != 0 and
                self.mem_props.memoryTypes[i].propertyFlags & properties == properties)
            {
                return @intCast(i);
            }
        }
        return null;
    }

    /// Sum the size of all device-local memory heaps exposed by the selected GPU.
    /// @param self Active Vulkan instance and memory properties.
    /// @returns The total number of bytes in device-local heaps.
    /// @note Drivers may expose multiple heaps, so this is an aggregate capacity rather than a single contiguous pool.
    pub fn vramBytes(self: *const Instance) u64 {
        var total: u64 = 0;
        for (self.mem_props.memoryHeaps[0..self.mem_props.memoryHeapCount]) |heap| {
            if (heap.flags & vk.c.VK_MEMORY_HEAP_DEVICE_LOCAL_BIT != 0) {
                total += heap.size;
            }
        }
        return total;
    }
};

test "Instance struct size is reasonable" {
    try std.testing.expect(@sizeOf(Instance) > 0);
}
