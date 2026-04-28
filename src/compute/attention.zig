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

/// Push constants for flash_attn_batched. Shared by two callers:
///  - prefill batched path: processes N queries sharing a KV cache,
///    seq_start is the position of the first query (0 on fresh prefill).
///  - decode-shape foundation (ZINC_BATCH_ATTN=1): n_queries=1 with
///    seq_start=state.position, bit-equivalent to the non-batched shader.
/// sink_offset is the per-layer offset into the per-head sinks buffer
/// (layer_idx * n_heads) — honoured by gpt-oss, NaN-gated otherwise.
pub const FlashAttnBatchedPush = extern struct {
    head_dim: u32,
    n_heads: u32,
    n_kv_heads: u32,
    seq_start: u32,
    n_queries: u32,
    page_size: u32,
    attn_scale_bits: u32,
    sink_offset: u32,
};

/// Push constants for flash_attn_split_merge. Reads N_I_CHUNKS partial outputs
/// per head from binding 0, applies the per-head sink, normalizes, writes the
/// final output to binding 1.
pub const FlashAttnSplitMergePush = extern struct {
    head_dim: u32,
    n_heads: u32,
    sink_offset: u32,
};

/// Manages flash attention pipeline and dispatch.
pub const AttentionDispatch = struct {
    /// Vulkan compute pipeline, or null if unavailable.
    pipeline: ?Pipeline,
    /// Batched variant — processes N queries per dispatch with causal mask.
    pipeline_batched: ?Pipeline,
    /// Split-K variant — same flash_attn.spv specialized with N_I_CHUNKS=fa_split_k_active
    /// so it writes per-chunk partials into partial_attn_out_buf instead of the
    /// final normalized output. Created lazily when ZINC_FA_SPLIT_K is set.
    pipeline_split: ?Pipeline,
    /// Split-K merge pass — combines per-chunk partials and applies sinks.
    pipeline_split_merge: ?Pipeline,
    /// N_I_CHUNKS the split pipelines were specialized with (1 if disabled).
    fa_split_k_active: u32,
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

        // Batched flash attention: 6 bindings (Q, K cache, V cache, page table,
        // output, sinks). Keeps the paged layout from flash_attn.comp and
        // accepts n_queries + seq_start push constants so a single dispatch
        // handles all prompt tokens with per-query causal masking. Used by
        // both the prefill batched path (n_queries=N) and the decode-shape
        // foundation gated by ZINC_BATCH_ATTN=1 (n_queries=1).
        const attn_batched_path = std.fmt.bufPrint(&path_buf, "{s}/flash_attn_batched.spv", .{shader_dir}) catch unreachable;
        const pipeline_batched = pipeline_mod.createFromSpirvWithOptions(instance, attn_batched_path, 6, @sizeOf(FlashAttnBatchedPush), &.{}, wave64_push_options, allocator) catch |err| blk: {
            log.warn("flash_attn_batched shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Split-K variant. The pipeline reuses flash_attn.spv with the
        // N_I_CHUNKS spec const set; its "output" binding (4) is wired to
        // partial_attn_out_buf at dispatch time. The merge pass (3 bindings:
        // partial, output, sinks) takes the per-head sink merge from the
        // original Phase 5. Default N_I_CHUNKS=4 (delivers +3.7% at L≈846 and
        // +11% at L=5 vs the no-split path on R9700; the 32→128 WG count
        // amortizes the SIMD pool starvation that 32 WGs across 64 CUs hit).
        // Override with ZINC_FA_SPLIT_K=N (N ∈ {0,1,2,4}; 0 or 1 disable).
        var pipeline_split: ?Pipeline = null;
        var pipeline_split_merge: ?Pipeline = null;
        var fa_split_k_active: u32 = 1;
        const fa_split_k_env = std.posix.getenv("ZINC_FA_SPLIT_K");
        const fa_split_k_request: u32 = blk: {
            if (fa_split_k_env) |raw| {
                const parsed = std.fmt.parseInt(u32, raw, 10) catch 0;
                if (parsed == 2 or parsed == 4) break :blk parsed;
                break :blk 1; // any non-{2,4} value disables
            }
            break :blk 4; // default-on with N=4
        };
        if (fa_split_k_request > 1) {
            // path_buf was reused by the batched-shader path above; rebuild
            // the flash_attn.spv path before specializing the split-K variant.
            const split_attn_path = std.fmt.bufPrint(&path_buf, "{s}/flash_attn.spv", .{shader_dir}) catch unreachable;
            const split_specs = [_]pipeline_mod.SpecConst{.{ .id = 0, .value = fa_split_k_request }};
            pipeline_split = pipeline_mod.createFromSpirvWithOptions(instance, split_attn_path, 6, @sizeOf(FlashAttnPush), &split_specs, wave64_push_options, allocator) catch |err| blk: {
                log.warn("flash_attn split-K specialization not loaded: {s}", .{@errorName(err)});
                break :blk null;
            };

            const merge_path = std.fmt.bufPrint(&path_buf, "{s}/flash_attn_split_merge.spv", .{shader_dir}) catch unreachable;
            const merge_specs = [_]pipeline_mod.SpecConst{.{ .id = 0, .value = fa_split_k_request }};
            // The merge shader uses local_size_x=64 but does not require wave64;
            // still pass the same options for consistency with the other
            // wave64 attention pipelines.
            pipeline_split_merge = pipeline_mod.createFromSpirvWithOptions(instance, merge_path, 3, @sizeOf(FlashAttnSplitMergePush), &merge_specs, wave64_push_options, allocator) catch |err| blk: {
                log.warn("flash_attn_split_merge shader not loaded: {s}", .{@errorName(err)});
                break :blk null;
            };

            if (pipeline_split != null and pipeline_split_merge != null) {
                fa_split_k_active = fa_split_k_request;
            } else {
                if (pipeline_split) |*p| p.deinit();
                if (pipeline_split_merge) |*p| p.deinit();
                pipeline_split = null;
                pipeline_split_merge = null;
            }
        }

        return AttentionDispatch{
            .pipeline = pipeline,
            .pipeline_batched = pipeline_batched,
            .pipeline_split = pipeline_split,
            .pipeline_split_merge = pipeline_split_merge,
            .fa_split_k_active = fa_split_k_active,
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

    /// Record a batched flash-attention dispatch.
    /// Grid is (n_heads, n_queries, 1); each (head, query) workgroup streams
    /// over the paged KV cache with causal_len = seq_start + query + 1.
    /// `sink_offset` is layer_idx * n_heads into the per-layer sinks buffer.
    /// @returns `error.ShaderNotLoaded` when the batched pipeline is unavailable.
    pub fn recordFlashAttnBatched(
        self: *const AttentionDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        seq_start: u32,
        n_queries: u32,
        page_size: u32,
        attn_scale: f32,
        sink_offset: u32,
    ) !void {
        const pip = if (self.pipeline_batched) |*p| p else return error.ShaderNotLoaded;
        const push = FlashAttnBatchedPush{
            .head_dim = head_dim,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .seq_start = seq_start,
            .n_queries = n_queries,
            .page_size = page_size,
            .attn_scale_bits = if (attn_scale != 0) @as(u32, @bitCast(attn_scale)) else 0,
            .sink_offset = sink_offset,
        };
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), n_heads, n_queries, 1);
    }

    /// Record the split-K flash attention dispatch (per-chunk partial pass).
    /// Grid: (n_heads, n_chunks, 1). Each WG runs the same flash_attn body but
    /// scoped to its (head, chunk_id) i-range and writes (O_partial, M, L) to
    /// the partial output buffer bound at slot 4.
    pub fn recordFlashAttnSplit(
        self: *const AttentionDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        head_dim: u32,
        n_heads: u32,
        n_kv_heads: u32,
        seq_len: u32,
        page_size: u32,
        attn_scale: f32,
        sink_offset: u32,
    ) !void {
        const pip = if (self.pipeline_split) |*p| p else return error.ShaderNotLoaded;
        const push = FlashAttnPush{
            .head_dim = head_dim,
            .n_heads = n_heads,
            .n_kv_heads = n_kv_heads,
            .seq_len = seq_len,
            .page_size = page_size,
            .attn_scale_bits = if (attn_scale != 0) @as(u32, @bitCast(attn_scale)) else 0,
            .sink_offset = sink_offset,
        };
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), n_heads, self.fa_split_k_active, 1);
    }

    /// Record the split-K merge pass dispatch — combines per-chunk partials
    /// for each head, applies the per-head sink term, and writes the final
    /// normalized attention output. One workgroup per head.
    pub fn recordFlashAttnSplitMerge(
        self: *const AttentionDispatch,
        cmd: *CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        head_dim: u32,
        n_heads: u32,
        sink_offset: u32,
    ) !void {
        const pip = if (self.pipeline_split_merge) |*p| p else return error.ShaderNotLoaded;
        const push = FlashAttnSplitMergePush{
            .head_dim = head_dim,
            .n_heads = n_heads,
            .sink_offset = sink_offset,
        };
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), n_heads, 1, 1);
    }

    /// Destroy the loaded pipeline and descriptor pool.
    /// @param self Dispatch wrapper to tear down in place.
    pub fn deinit(self: *AttentionDispatch) void {
        if (self.pipeline) |*p| p.deinit();
        if (self.pipeline_batched) |*p| p.deinit();
        if (self.pipeline_split) |*p| p.deinit();
        if (self.pipeline_split_merge) |*p| p.deinit();
        vk.c.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.* = undefined;
    }
};
