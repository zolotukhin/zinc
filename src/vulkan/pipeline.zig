//! Load SPIR-V compute shaders into Vulkan pipelines.
//! @section Vulkan Runtime
//! Dispatch helpers use this module to build descriptor layouts, pipeline
//! layouts, and compute pipelines from the compiled shader binaries.
const std = @import("std");
const vk = @import("vk.zig");
const Instance = @import("instance.zig").Instance;

const log = std.log.scoped(.pipeline);

/// A compute pipeline wrapping a SPIR-V shader module.
pub const Pipeline = struct {
    /// Compiled SPIR-V shader module.
    shader_module: vk.c.VkShaderModule,
    /// Descriptor set layout for buffer bindings.
    descriptor_set_layout: vk.c.VkDescriptorSetLayout,
    /// Pipeline layout with push-constant ranges.
    pipeline_layout: vk.c.VkPipelineLayout,
    /// Vulkan compute pipeline, or null if unavailable.
    pipeline: vk.c.VkPipeline,
    /// Logical device.
    device: vk.c.VkDevice,

    /// Destroy the shader module, descriptor layout, pipeline layout, and pipeline.
    /// @param self Pipeline object to tear down in place.
    pub fn deinit(self: *Pipeline) void {
        vk.c.vkDestroyPipeline(self.device, self.pipeline, null);
        vk.c.vkDestroyPipelineLayout(self.device, self.pipeline_layout, null);
        vk.c.vkDestroyDescriptorSetLayout(self.device, self.descriptor_set_layout, null);
        vk.c.vkDestroyShaderModule(self.device, self.shader_module, null);
        self.* = undefined;
    }
};

/// Specialization constant entry for compute pipelines.
pub const SpecConst = struct {
    /// Unique identifier.
    id: u32,
    /// Integer value.
    value: u32,
};

/// Optional compute-pipeline knobs for subgroup-sensitive shaders.
pub const PipelineOptions = struct {
    /// Request a specific subgroup width when supported by the device.
    required_subgroup_size: ?u32 = null,
    /// Require the workgroup to be composed of full subgroups when supported.
    require_full_subgroups: bool = false,
};

/// Create a compute pipeline from a SPIR-V file.
/// @param instance Active Vulkan instance and logical device.
/// @param spirv_path Filesystem path to the compiled SPIR-V module.
/// @param binding_count Number of storage-buffer bindings expected by the shader.
/// @param push_constant_size Size of the push-constant block in bytes.
/// @param spec_constants Specialization constants applied at pipeline creation time.
/// @param allocator Allocator used for shader bytes and temporary Vulkan structs.
/// @returns A fully created compute pipeline and its associated layouts.
pub fn createFromSpirv(
    instance: *const Instance,
    spirv_path: []const u8,
    binding_count: u32,
    push_constant_size: u32,
    spec_constants: []const SpecConst,
    allocator: std.mem.Allocator,
) !Pipeline {
    return createFromSpirvWithOptions(
        instance,
        spirv_path,
        binding_count,
        push_constant_size,
        spec_constants,
        .{},
        allocator,
    );
}

/// Create a compute pipeline from a SPIR-V file with optional subgroup controls.
pub fn createFromSpirvWithOptions(
    instance: *const Instance,
    spirv_path: []const u8,
    binding_count: u32,
    push_constant_size: u32,
    spec_constants: []const SpecConst,
    options: PipelineOptions,
    allocator: std.mem.Allocator,
) !Pipeline {
    // Read SPIR-V binary
    const file = std.fs.cwd().openFile(spirv_path, .{}) catch |err| {
        log.err("Failed to open SPIR-V file '{s}': {s}", .{ spirv_path, @errorName(err) });
        return error.ShaderFileNotFound;
    };
    defer file.close();

    const stat = try file.stat();
    const spirv_code = try allocator.alloc(u8, stat.size);
    defer allocator.free(spirv_code);
    const bytes_read = try file.readAll(spirv_code);
    if (bytes_read != stat.size) return error.ShaderReadIncomplete;

    // Create shader module
    const module_info = vk.c.VkShaderModuleCreateInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .codeSize = stat.size,
        .pCode = @ptrCast(@alignCast(spirv_code.ptr)),
    };

    var shader_module: vk.c.VkShaderModule = null;
    var result = vk.c.vkCreateShaderModule(instance.device, &module_info, null, &shader_module);
    if (result != vk.c.VK_SUCCESS) {
        log.err("vkCreateShaderModule failed: {d}", .{result});
        return error.ShaderModuleCreateFailed;
    }
    errdefer vk.c.vkDestroyShaderModule(instance.device, shader_module, null);

    // Descriptor set layout: N storage buffers
    const bindings = try allocator.alloc(vk.c.VkDescriptorSetLayoutBinding, binding_count);
    defer allocator.free(bindings);

    for (0..binding_count) |i| {
        bindings[i] = vk.c.VkDescriptorSetLayoutBinding{
            .binding = @intCast(i),
            .descriptorType = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = vk.c.VK_SHADER_STAGE_COMPUTE_BIT,
            .pImmutableSamplers = null,
        };
    }

    const ds_layout_info = vk.c.VkDescriptorSetLayoutCreateInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .bindingCount = binding_count,
        .pBindings = bindings.ptr,
    };

    var descriptor_set_layout: vk.c.VkDescriptorSetLayout = null;
    result = vk.c.vkCreateDescriptorSetLayout(instance.device, &ds_layout_info, null, &descriptor_set_layout);
    if (result != vk.c.VK_SUCCESS) {
        log.err("vkCreateDescriptorSetLayout failed: {d}", .{result});
        return error.DescriptorSetLayoutFailed;
    }
    errdefer vk.c.vkDestroyDescriptorSetLayout(instance.device, descriptor_set_layout, null);

    // Pipeline layout with optional push constants
    const push_range = vk.c.VkPushConstantRange{
        .stageFlags = vk.c.VK_SHADER_STAGE_COMPUTE_BIT,
        .offset = 0,
        .size = push_constant_size,
    };

    const layout_info = vk.c.VkPipelineLayoutCreateInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .setLayoutCount = 1,
        .pSetLayouts = &descriptor_set_layout,
        .pushConstantRangeCount = if (push_constant_size > 0) 1 else 0,
        .pPushConstantRanges = if (push_constant_size > 0) &push_range else null,
    };

    var pipeline_layout: vk.c.VkPipelineLayout = null;
    result = vk.c.vkCreatePipelineLayout(instance.device, &layout_info, null, &pipeline_layout);
    if (result != vk.c.VK_SUCCESS) {
        log.err("vkCreatePipelineLayout failed: {d}", .{result});
        return error.PipelineLayoutFailed;
    }
    errdefer vk.c.vkDestroyPipelineLayout(instance.device, pipeline_layout, null);

    // Specialization constants
    const spec_entries = try allocator.alloc(vk.c.VkSpecializationMapEntry, spec_constants.len);
    defer allocator.free(spec_entries);
    const spec_data = try allocator.alloc(u32, spec_constants.len);
    defer allocator.free(spec_data);

    for (spec_constants, 0..) |sc, i| {
        spec_entries[i] = vk.c.VkSpecializationMapEntry{
            .constantID = sc.id,
            .offset = @intCast(i * @sizeOf(u32)),
            .size = @sizeOf(u32),
        };
        spec_data[i] = sc.value;
    }

    const spec_info = vk.c.VkSpecializationInfo{
        .mapEntryCount = @intCast(spec_constants.len),
        .pMapEntries = if (spec_constants.len > 0) spec_entries.ptr else null,
        .dataSize = spec_constants.len * @sizeOf(u32),
        .pData = if (spec_constants.len > 0) @ptrCast(spec_data.ptr) else null,
    };

    const required_subgroup_size = options.required_subgroup_size orelse 0;
    const use_required_subgroup_size = required_subgroup_size > 0 and
        instance.caps.supportsRequiredSubgroupSize(required_subgroup_size);
    const require_full_subgroups = options.require_full_subgroups and instance.caps.compute_full_subgroups;
    if (options.required_subgroup_size != null and !use_required_subgroup_size) {
        log.debug("Ignoring required subgroup size {d} for {s}", .{ required_subgroup_size, spirv_path });
    }
    if (options.require_full_subgroups and !require_full_subgroups) {
        log.debug("Ignoring full-subgroup request for {s}", .{spirv_path});
    }

    var subgroup_info = vk.c.VkPipelineShaderStageRequiredSubgroupSizeCreateInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .pNext = null,
        .requiredSubgroupSize = required_subgroup_size,
    };

    // Compute pipeline
    const stage_info = vk.c.VkPipelineShaderStageCreateInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
        .pNext = if (use_required_subgroup_size) &subgroup_info else null,
        .flags = if (require_full_subgroups) vk.c.VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT else 0,
        .stage = vk.c.VK_SHADER_STAGE_COMPUTE_BIT,
        .module = shader_module,
        .pName = "main",
        .pSpecializationInfo = if (spec_constants.len > 0) &spec_info else null,
    };

    const pipeline_info = vk.c.VkComputePipelineCreateInfo{
        .sType = vk.c.VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .pNext = null,
        .flags = 0,
        .stage = stage_info,
        .layout = pipeline_layout,
        .basePipelineHandle = @as(vk.c.VkPipeline, null),
        .basePipelineIndex = -1,
    };

    var pipeline: vk.c.VkPipeline = null;
    result = vk.c.vkCreateComputePipelines(
        instance.device,
        @as(vk.c.VkPipelineCache, null),
        1,
        &pipeline_info,
        null,
        &pipeline,
    );
    if (result != vk.c.VK_SUCCESS) {
        log.err("vkCreateComputePipelines failed: {d}", .{result});
        return error.ComputePipelineCreateFailed;
    }

    return Pipeline{
        .shader_module = shader_module,
        .descriptor_set_layout = descriptor_set_layout,
        .pipeline_layout = pipeline_layout,
        .pipeline = pipeline,
        .device = instance.device,
    };
}
