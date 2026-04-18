//! Initialize Vulkan, select a compute-capable device, and expose memory utilities.
//! @section Vulkan Runtime
//! This is the entry point for GPU setup: instance creation, device selection,
//! queue discovery, and VRAM inspection.
const std = @import("std");
const vk = @import("vk.zig");

const log = std.log.scoped(.vulkan);

/// Function pointer type for `vkCmdPushDescriptorSetKHR` when the extension is enabled.
pub const PushDescriptorFn = *const fn (
    vk.c.VkCommandBuffer,
    vk.c.VkPipelineBindPoint,
    vk.c.VkPipelineLayout,
    u32,
    u32,
    [*]const vk.c.VkWriteDescriptorSet,
) callconv(.c) void;

/// Queried Vulkan device capabilities that affect pipeline creation choices.
pub const DeviceCapabilities = struct {
    /// Shader `requiredSubgroupSize` can be requested at pipeline creation.
    subgroup_size_control: bool = false,
    /// `REQUIRE_FULL_SUBGROUPS` can be requested for compute stages.
    compute_full_subgroups: bool = false,
    /// Shader stages that accept `requiredSubgroupSize`.
    required_subgroup_size_stages: vk.c.VkShaderStageFlags = 0,
    /// Minimum supported subgroup width.
    min_subgroup_size: u32 = 0,
    /// Maximum supported subgroup width.
    max_subgroup_size: u32 = 0,
    /// 16-bit storage-buffer access is supported and enabled.
    storage_buffer16: bool = false,
    /// Float16 arithmetic is supported and enabled.
    shader_float16: bool = false,
    /// Shader subgroup extended types are supported and enabled.
    subgroup_extended_types: bool = false,
    /// Cooperative matrix support is available and enabled.
    cooperative_matrix: bool = false,
    /// Push descriptors are supported by the device extension list.
    push_descriptor: bool = false,

    /// Return whether a compute shader can request the given subgroup size.
    pub fn supportsRequiredSubgroupSize(self: DeviceCapabilities, size: u32) bool {
        if (!self.subgroup_size_control) return false;
        if (size < self.min_subgroup_size or size > self.max_subgroup_size) return false;
        return (self.required_subgroup_size_stages & vk.c.VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    }
};

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
    /// Actual selected physical-device index after bounds clamping.
    selected_device_index: u32,
    /// Queried device capability bits used by pipeline creation.
    caps: DeviceCapabilities = .{},
    /// Loaded `vkCmdPushDescriptorSetKHR` entrypoint when push descriptors are enabled.
    push_descriptor_fn: ?PushDescriptorFn = null,
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

        // Device enumeration is useful when debugging selection issues, but
        // normal user-facing commands like `model list` should stay quiet.
        for (phys_devices[0..dev_count], 0..) |pdev, i| {
            var props: vk.c.VkPhysicalDeviceProperties = undefined;
            vk.c.vkGetPhysicalDeviceProperties(pdev, &props);
            const name = std.mem.sliceTo(&props.deviceName, 0);
            log.debug("GPU {d}: {s} (vendor 0x{x:0>4})", .{ i, name, props.vendorID });
        }

        // Select device
        const dev_idx: u32 = if (preferred_device < dev_count) preferred_device else 0;
        const physical_device = phys_devices[dev_idx];

        var device_props: vk.c.VkPhysicalDeviceProperties = undefined;
        vk.c.vkGetPhysicalDeviceProperties(physical_device, &device_props);

        var mem_props: vk.c.VkPhysicalDeviceMemoryProperties = undefined;
        vk.c.vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_props);

        const device_caps = try queryDeviceCapabilities(allocator, physical_device);
        const dev_name = std.mem.sliceTo(&device_props.deviceName, 0);
        log.debug("Selected GPU {d}: {s}", .{ dev_idx, dev_name });
        if (device_caps.subgroup_size_control) {
            log.debug("Subgroup size control enabled: {d}-{d}", .{
                device_caps.min_subgroup_size,
                device_caps.max_subgroup_size,
            });
        }
        if (device_caps.cooperative_matrix) {
            log.debug("Cooperative matrix enabled", .{});
        }

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

        var enabled_extensions: [2][*:0]const u8 = undefined;
        var enabled_extension_count: u32 = 0;
        if (device_caps.cooperative_matrix) {
            enabled_extensions[enabled_extension_count] = vk.c.VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME;
            enabled_extension_count += 1;
        }
        if (device_caps.push_descriptor) {
            enabled_extensions[enabled_extension_count] = vk.c.VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
            enabled_extension_count += 1;
        }

        var subgroup_size_control_features = vk.c.VkPhysicalDeviceSubgroupSizeControlFeatures{
            .sType = vk.c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
            .pNext = null,
            .subgroupSizeControl = if (device_caps.subgroup_size_control) vk.c.VK_TRUE else vk.c.VK_FALSE,
            .computeFullSubgroups = if (device_caps.compute_full_subgroups) vk.c.VK_TRUE else vk.c.VK_FALSE,
        };
        var storage_16bit_features = vk.c.VkPhysicalDevice16BitStorageFeatures{
            .sType = vk.c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
            .pNext = &subgroup_size_control_features,
            .storageBuffer16BitAccess = if (device_caps.storage_buffer16) vk.c.VK_TRUE else vk.c.VK_FALSE,
            .uniformAndStorageBuffer16BitAccess = vk.c.VK_FALSE,
            .storagePushConstant16 = vk.c.VK_FALSE,
            .storageInputOutput16 = vk.c.VK_FALSE,
        };
        var shader_float16_int8_features = vk.c.VkPhysicalDeviceShaderFloat16Int8Features{
            .sType = vk.c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
            .pNext = &storage_16bit_features,
            .shaderFloat16 = if (device_caps.shader_float16) vk.c.VK_TRUE else vk.c.VK_FALSE,
            .shaderInt8 = vk.c.VK_FALSE,
        };
        var subgroup_extended_types_features = vk.c.VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures{
            .sType = vk.c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES,
            .pNext = &shader_float16_int8_features,
            .shaderSubgroupExtendedTypes = if (device_caps.subgroup_extended_types) vk.c.VK_TRUE else vk.c.VK_FALSE,
        };
        var cooperative_matrix_features = vk.c.VkPhysicalDeviceCooperativeMatrixFeaturesKHR{
            .sType = vk.c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
            .pNext = &subgroup_extended_types_features,
            .cooperativeMatrix = if (device_caps.cooperative_matrix) vk.c.VK_TRUE else vk.c.VK_FALSE,
            .cooperativeMatrixRobustBufferAccess = vk.c.VK_FALSE,
        };

        const device_create_info = vk.c.VkDeviceCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
            .pNext = if (device_caps.cooperative_matrix) &cooperative_matrix_features else &subgroup_extended_types_features,
            .flags = 0,
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &queue_create_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = null,
            .enabledExtensionCount = enabled_extension_count,
            .ppEnabledExtensionNames = if (enabled_extension_count > 0)
                enabled_extensions[0..@intCast(enabled_extension_count)].ptr
            else
                null,
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

        const push_descriptor_fn: ?PushDescriptorFn = if (device_caps.push_descriptor)
            if (vk.c.vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR")) |fn_ptr|
                @ptrCast(fn_ptr)
            else
                null
        else
            null;
        if (push_descriptor_fn != null) {
            log.info("VK_KHR_push_descriptor available", .{});
        }

        log.debug("Vulkan device ready — compute queue family {d}", .{compute_queue_family});

        return Instance{
            .handle = instance,
            .physical_device = physical_device,
            .device = device,
            .compute_queue = compute_queue,
            .compute_queue_family = compute_queue_family,
            .device_props = device_props,
            .mem_props = mem_props,
            .selected_device_index = dev_idx,
            .caps = device_caps,
            .push_descriptor_fn = push_descriptor_fn,
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

fn queryDeviceCapabilities(allocator: std.mem.Allocator, physical_device: vk.c.VkPhysicalDevice) !DeviceCapabilities {
    var ext_count: u32 = 0;
    var result = vk.c.vkEnumerateDeviceExtensionProperties(physical_device, null, &ext_count, null);
    if (result != vk.c.VK_SUCCESS) return error.DeviceExtensionEnumerationFailed;

    const ext_props = try allocator.alloc(vk.c.VkExtensionProperties, ext_count);
    defer allocator.free(ext_props);
    if (ext_count > 0) {
        result = vk.c.vkEnumerateDeviceExtensionProperties(physical_device, null, &ext_count, ext_props.ptr);
        if (result != vk.c.VK_SUCCESS) return error.DeviceExtensionEnumerationFailed;
    }
    const extensions = ext_props[0..ext_count];
    const coop_extension_supported = hasDeviceExtension(extensions, "VK_KHR_cooperative_matrix");
    const push_descriptor_supported = hasDeviceExtension(extensions, "VK_KHR_push_descriptor");

    var subgroup_props = vk.c.VkPhysicalDeviceSubgroupSizeControlProperties{
        .sType = vk.c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
        .pNext = null,
        .minSubgroupSize = 0,
        .maxSubgroupSize = 0,
        .maxComputeWorkgroupSubgroups = 0,
        .requiredSubgroupSizeStages = 0,
    };
    var props2 = vk.c.VkPhysicalDeviceProperties2{
        .sType = vk.c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2,
        .pNext = &subgroup_props,
        .properties = undefined,
    };
    vk.c.vkGetPhysicalDeviceProperties2(physical_device, &props2);

    var subgroup_size_control_features = vk.c.VkPhysicalDeviceSubgroupSizeControlFeatures{
        .sType = vk.c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES,
        .pNext = null,
        .subgroupSizeControl = vk.c.VK_FALSE,
        .computeFullSubgroups = vk.c.VK_FALSE,
    };
    var storage_16bit_features = vk.c.VkPhysicalDevice16BitStorageFeatures{
        .sType = vk.c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES,
        .pNext = &subgroup_size_control_features,
        .storageBuffer16BitAccess = vk.c.VK_FALSE,
        .uniformAndStorageBuffer16BitAccess = vk.c.VK_FALSE,
        .storagePushConstant16 = vk.c.VK_FALSE,
        .storageInputOutput16 = vk.c.VK_FALSE,
    };
    var shader_float16_int8_features = vk.c.VkPhysicalDeviceShaderFloat16Int8Features{
        .sType = vk.c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES,
        .pNext = &storage_16bit_features,
        .shaderFloat16 = vk.c.VK_FALSE,
        .shaderInt8 = vk.c.VK_FALSE,
    };
    var subgroup_extended_types_features = vk.c.VkPhysicalDeviceShaderSubgroupExtendedTypesFeatures{
        .sType = vk.c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES,
        .pNext = &shader_float16_int8_features,
        .shaderSubgroupExtendedTypes = vk.c.VK_FALSE,
    };
    var cooperative_matrix_features = vk.c.VkPhysicalDeviceCooperativeMatrixFeaturesKHR{
        .sType = vk.c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR,
        .pNext = &subgroup_extended_types_features,
        .cooperativeMatrix = vk.c.VK_FALSE,
        .cooperativeMatrixRobustBufferAccess = vk.c.VK_FALSE,
    };

    var features2 = vk.c.VkPhysicalDeviceFeatures2{
        .sType = vk.c.VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2,
        .pNext = if (coop_extension_supported) &cooperative_matrix_features else &subgroup_extended_types_features,
        .features = undefined,
    };
    vk.c.vkGetPhysicalDeviceFeatures2(physical_device, &features2);

    return .{
        .subgroup_size_control = subgroup_size_control_features.subgroupSizeControl == vk.c.VK_TRUE,
        .compute_full_subgroups = subgroup_size_control_features.computeFullSubgroups == vk.c.VK_TRUE,
        .required_subgroup_size_stages = subgroup_props.requiredSubgroupSizeStages,
        .min_subgroup_size = subgroup_props.minSubgroupSize,
        .max_subgroup_size = subgroup_props.maxSubgroupSize,
        .storage_buffer16 = storage_16bit_features.storageBuffer16BitAccess == vk.c.VK_TRUE,
        .shader_float16 = shader_float16_int8_features.shaderFloat16 == vk.c.VK_TRUE,
        .subgroup_extended_types = subgroup_extended_types_features.shaderSubgroupExtendedTypes == vk.c.VK_TRUE,
        .cooperative_matrix = coop_extension_supported and cooperative_matrix_features.cooperativeMatrix == vk.c.VK_TRUE,
        .push_descriptor = push_descriptor_supported,
    };
}

fn hasDeviceExtension(ext_props: []const vk.c.VkExtensionProperties, name: []const u8) bool {
    for (ext_props) |ext| {
        if (std.mem.eql(u8, std.mem.sliceTo(&ext.extensionName, 0), name)) return true;
    }
    return false;
}

test "Instance struct size is reasonable" {
    try std.testing.expect(@sizeOf(Instance) > 0);
}
