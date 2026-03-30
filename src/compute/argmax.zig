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

pub const ArgmaxDispatch = struct {
    pipeline: ?Pipeline,
    descriptor_pool: vk.c.VkDescriptorPool,
    device: vk.c.VkDevice,

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

    pub fn deinit(self: *ArgmaxDispatch) void {
        if (self.pipeline) |*p| p.deinit();
        vk.c.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.* = undefined;
    }
};
