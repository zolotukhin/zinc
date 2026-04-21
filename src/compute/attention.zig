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
pub const FlashAttnPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_len: u32,
    page_size: u32,
    attn_scale_bits: u32, // float scale bits (0 = use 1/sqrt(head_dim))
    sink_offset: u32, // layer_idx * n_heads — starting index into sink_data for this layer
};

/// Push constants for flash_attn_batched — matches the shader header in
/// src/shaders/flash_attn_batched.comp. Used by the Vulkan batched prefill
/// path to process N queries (with per-query causal masking) in one dispatch.
pub const BatchedFlashAttnPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    kv_len: u32,
    page_size: u32,
    attn_scale_bits: u32,
    n_queries: u32,
    kv_pos_offset: u32,
};

/// Manages flash attention pipeline and dispatch.
pub const AttentionDispatch = struct {
    /// Vulkan compute pipeline, or null if unavailable.
    pipeline: ?Pipeline,
    /// Batched variant — processes N queries per dispatch with causal mask.
    pipeline_batched: ?Pipeline,
    /// Descriptor pool for this dispatch.
    descriptor_pool: vk.c.VkDescriptorPool,
    /// Logical device.
    device: vk.c.VkDevice,

    /// Create the flash-attention dispatch wrapper and load its shader pipeline.
    /// @param instance Active Vulkan instance and logical device.
    /// @param shader_dir Directory containing compiled SPIR-V shader binaries.
    /// @param allocator Allocator used for temporary pipeline creation state.
    /// @returns An AttentionDispatch ready to record flash-attention passes.
    pub fn init(
        /// Vulkan instance.
        instance: *const Instance,
        shader_dir: []const u8,
        /// Allocator for owned resources.
        allocator: std.mem.Allocator,
    ) !AttentionDispatch {
        const pool_size = vk.c.VkDescriptorPoolSize{
            .type = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 6 * 2, // 6 bindings, 2 sets
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
        if (instance.push_descriptor_fn == null) {
            const result = vk.c.vkCreateDescriptorPool(instance.device, &pool_info, null, &descriptor_pool);
            if (result != vk.c.VK_SUCCESS) return error.DescriptorPoolCreateFailed;
        }

        var path_buf: [512]u8 = undefined;
        const wave64_push_options = pipeline_mod.PipelineOptions{
            .required_subgroup_size = 64,
            .require_full_subgroups = true,
            .push_descriptors = instance.push_descriptor_fn != null,
        };

        // Flash attention: 6 bindings (Q, K cache, V cache, page table, output, per-head sinks)
        const attn_path = std.fmt.bufPrint(&path_buf, "{s}/flash_attn.spv", .{shader_dir}) catch unreachable;
        const pipeline = pipeline_mod.createFromSpirvWithOptions(instance, attn_path, 6, @sizeOf(FlashAttnPush), &.{}, wave64_push_options, allocator) catch |err| blk: {
            log.warn("flash_attn shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Batched flash attention: 5 bindings (Q, K cache, V cache, page table, output).
        // Keeps the paged layout from flash_attn.comp but accepts n_queries and
        // kv_pos_offset push constants so a single dispatch handles all prompt
        // tokens with per-query causal masking. Foundation for prefillBatched.
        const attn_batched_path = std.fmt.bufPrint(&path_buf, "{s}/flash_attn_batched.spv", .{shader_dir}) catch unreachable;
        const pipeline_batched = pipeline_mod.createFromSpirvWithOptions(instance, attn_batched_path, 5, @sizeOf(BatchedFlashAttnPush), &.{}, wave64_push_options, allocator) catch |err| blk: {
            log.warn("flash_attn_batched shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        return AttentionDispatch{
            .pipeline = pipeline,
            .pipeline_batched = pipeline_batched,
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
        cmd: *CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        /// Per-head dimension.
        head_dim: u32,
        /// Number of query heads.
        n_heads: u32,
        /// Number of KV heads (GQA).
        n_kv_heads: u32,
        /// Sequence length.
        seq_len: u32,
        /// KV cache page size.
        page_size: u32,
        /// Attention scale (0 = use 1/sqrt(head_dim)).
        attn_scale: f32,
        /// Offset into the per-model sink buffer for this layer (layer_idx * n_heads).
        sink_offset: u32,
    ) !void {
        const pip = if (self.pipeline) |*p| p else return error.ShaderNotLoaded;

        const push = FlashAttnPush{
            .head_dim = head_dim,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .seq_len = seq_len,
            .page_size = page_size,
            .attn_scale_bits = if (attn_scale != 0) @as(u32, @bitCast(attn_scale)) else 0,
            .sink_offset = sink_offset,
        };

        // One workgroup per query head
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), n_heads, 1, 1);
    }

    /// Record a batched flash-attention dispatch for prefill.
    /// Grid is (n_heads, n_queries, 1); each (head, query) workgroup streams
    /// over the paged KV cache with causal_len = kv_pos_offset + query + 1.
    /// @returns `error.ShaderNotLoaded` when the batched pipeline is unavailable.
    pub fn recordFlashAttnBatched(
        self: *const AttentionDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        kv_len: u32,
        page_size: u32,
        attn_scale: f32,
        n_queries: u32,
        kv_pos_offset: u32,
    ) !void {
        const pip = if (self.pipeline_batched) |*p| p else return error.ShaderNotLoaded;
        const push = BatchedFlashAttnPush{
            .head_dim = head_dim,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .kv_len = kv_len,
            .page_size = page_size,
            .attn_scale_bits = if (attn_scale != 0) @as(u32, @bitCast(attn_scale)) else 0,
            .n_queries = n_queries,
            .kv_pos_offset = kv_pos_offset,
        };
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), n_heads, n_queries, 1);
    }

    /// Destroy the loaded pipeline and descriptor pool.
    /// @param self Dispatch wrapper to tear down in place.
    pub fn deinit(self: *AttentionDispatch) void {
        if (self.pipeline) |*p| p.deinit();
        if (self.pipeline_batched) |*p| p.deinit();
        vk.c.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.* = undefined;
    }
};
