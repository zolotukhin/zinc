//! Wrap the flash-attention compute shader and its dispatch parameters.
//! @section Shader Dispatch
//! This helper owns the pipeline resources needed to bind paged attention
//! inputs and record a flash-attention compute pass.
const std = @import("std");
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const Pipeline = @import("../vulkan/pipeline.zig").Pipeline;
const pipeline_mod = @import("../vulkan/pipeline.zig");
const CommandBuffer = @import("../vulkan/command.zig").CommandBuffer;

const log = std.log.scoped(.attention);

/// Push constants for flash attention shader.
const FlashAttnPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_len: u32,
    page_size: u32,
};

/// Manages flash attention pipeline and dispatch.
pub const AttentionDispatch = struct {
    pipeline: ?Pipeline,
    descriptor_pool: vk.c.VkDescriptorPool,
    device: vk.c.VkDevice,

    /// Create the flash-attention dispatch wrapper and load its shader pipeline.
    /// @param instance Active Vulkan instance and logical device.
    /// @param shader_dir Directory containing compiled SPIR-V shader binaries.
    /// @param allocator Allocator used for temporary pipeline creation state.
    /// @returns An AttentionDispatch ready to record flash-attention passes.
    pub fn init(
        instance: *const Instance,
        shader_dir: []const u8,
        allocator: std.mem.Allocator,
    ) !AttentionDispatch {
        const pool_size = vk.c.VkDescriptorPoolSize{
            .type = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 5 * 2, // 5 bindings, 2 sets
        };
        const pool_info = vk.c.VkDescriptorPoolCreateInfo{
            .sType = vk.c.VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
            .pNext = null,
            .flags = vk.c.VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT,
            .maxSets = 8,
            .poolSizeCount = 1,
            .pPoolSizes = &pool_size,
        };
        var descriptor_pool: vk.c.VkDescriptorPool = null;
        const result = vk.c.vkCreateDescriptorPool(instance.device, &pool_info, null, &descriptor_pool);
        if (result != vk.c.VK_SUCCESS) return error.DescriptorPoolCreateFailed;

        var path_buf: [512]u8 = undefined;

        // Flash attention: 5 bindings (Q, K cache, V cache, page table, output)
        const attn_path = std.fmt.bufPrint(&path_buf, "{s}/flash_attn.spv", .{shader_dir}) catch unreachable;
        const pipeline = pipeline_mod.createFromSpirv(instance, attn_path, 5, @sizeOf(FlashAttnPush), &.{}, allocator) catch |err| blk: {
            log.warn("flash_attn shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        return AttentionDispatch{
            .pipeline = pipeline,
            .descriptor_pool = descriptor_pool,
            .device = instance.device,
        };
    }

    /// Record a flash-attention dispatch for the current decode position.
    /// @param self Dispatch wrapper containing the flash-attention pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set containing query, KV-cache, page-table, and output buffers.
    /// @param head_dim Hidden width per attention head.
    /// @param n_heads Number of query heads to process.
    /// @param n_kv_heads Number of KV heads present in the cache.
    /// @param seq_len Current decoded sequence length.
    /// @param page_size Tokens stored in each KV-cache page.
    /// @returns `error.ShaderNotLoaded` when the flash-attention shader pipeline is unavailable.
    /// @note The helper dispatches one workgroup per query head.
    pub fn recordFlashAttn(
        self: *const AttentionDispatch,
        cmd: *const CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        seq_len: u32,
        page_size: u32,
    ) !void {
        const pip = if (self.pipeline) |*p| p else return error.ShaderNotLoaded;

        const push = FlashAttnPush{
            .head_dim = head_dim,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .seq_len = seq_len,
            .page_size = page_size,
        };

        // One workgroup per query head
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), n_heads, 1, 1);
    }

    /// Destroy the loaded pipeline and descriptor pool.
    /// @param self Dispatch wrapper to tear down in place.
    pub fn deinit(self: *AttentionDispatch) void {
        if (self.pipeline) |*p| p.deinit();
        vk.c.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.* = undefined;
    }
};
