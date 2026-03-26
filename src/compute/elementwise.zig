const std = @import("std");
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const Pipeline = @import("../vulkan/pipeline.zig").Pipeline;
const pipeline_mod = @import("../vulkan/pipeline.zig");
const CommandBuffer = @import("../vulkan/command.zig").CommandBuffer;

const log = std.log.scoped(.elementwise);

/// Push constants for RMS norm shader.
const RmsNormPush = extern struct {
    N: u32,
    eps_bits: u32, // float bits reinterpreted as u32
};

/// Push constants for SwiGLU shader.
const SwigluPush = extern struct {
    N: u32,
};

/// Push constants for RoPE shader.
const RopePush = extern struct {
    head_dim: u32,
    n_heads: u32,
    position: u32,
    freq_base_bits: u32, // float bits reinterpreted as u32
};

/// Manages element-wise fused kernel pipelines.
pub const ElementwiseDispatch = struct {
    pipeline_rms_norm: ?Pipeline,
    pipeline_swiglu: ?Pipeline,
    pipeline_rope: ?Pipeline,
    descriptor_pool: vk.c.VkDescriptorPool,
    device: vk.c.VkDevice,

    pub fn init(
        instance: *const Instance,
        shader_dir: []const u8,
        allocator: std.mem.Allocator,
    ) !ElementwiseDispatch {
        const pool_size = vk.c.VkDescriptorPoolSize{
            .type = vk.c.VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 3 * 3,
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
        const result = vk.c.vkCreateDescriptorPool(instance.device, &pool_info, null, &descriptor_pool);
        if (result != vk.c.VK_SUCCESS) return error.DescriptorPoolCreateFailed;

        var path_buf: [512]u8 = undefined;

        // RMS norm: 2 inputs (x, weight) + 1 output = 3 bindings
        const rms_path = std.fmt.bufPrint(&path_buf, "{s}/rms_norm_mul.spv", .{shader_dir}) catch unreachable;
        const pipeline_rms_norm = pipeline_mod.createFromSpirv(instance, rms_path, 3, @sizeOf(RmsNormPush), &.{}, allocator) catch |err| blk: {
            log.warn("rms_norm_mul shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // SwiGLU: 2 inputs (gate, up) + 1 output = 3 bindings
        const swiglu_path = std.fmt.bufPrint(&path_buf, "{s}/swiglu.spv", .{shader_dir}) catch unreachable;
        const pipeline_swiglu = pipeline_mod.createFromSpirv(instance, swiglu_path, 3, @sizeOf(SwigluPush), &.{}, allocator) catch |err| blk: {
            log.warn("swiglu shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // RoPE: 1 input + 1 output = 2 bindings
        const rope_path = std.fmt.bufPrint(&path_buf, "{s}/rope_fused.spv", .{shader_dir}) catch unreachable;
        const pipeline_rope = pipeline_mod.createFromSpirv(instance, rope_path, 2, @sizeOf(RopePush), &.{}, allocator) catch |err| blk: {
            log.warn("rope_fused shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        return ElementwiseDispatch{
            .pipeline_rms_norm = pipeline_rms_norm,
            .pipeline_swiglu = pipeline_swiglu,
            .pipeline_rope = pipeline_rope,
            .descriptor_pool = descriptor_pool,
            .device = instance.device,
        };
    }

    /// Record an RMS norm + scale dispatch.
    pub fn recordRmsNorm(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        hidden_dim: u32,
        n_tokens: u32,
        eps: f32,
    ) !void {
        const pip = if (self.pipeline_rms_norm) |*p| p else return error.ShaderNotLoaded;
        const push = RmsNormPush{
            .N = hidden_dim,
            .eps_bits = @bitCast(eps),
        };
        // One workgroup per token
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), n_tokens, 1, 1);
    }

    /// Record a SwiGLU dispatch.
    pub fn recordSwiglu(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
    ) !void {
        const pip = if (self.pipeline_swiglu) |*p| p else return error.ShaderNotLoaded;
        const push = SwigluPush{ .N = n_elements };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record a RoPE dispatch.
    pub fn recordRope(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        head_dim: u32,
        n_heads: u32,
        position: u32,
        freq_base: f32,
    ) !void {
        const pip = if (self.pipeline_rope) |*p| p else return error.ShaderNotLoaded;
        const push = RopePush{
            .head_dim = head_dim,
            .n_heads = n_heads,
            .position = position,
            .freq_base_bits = @bitCast(freq_base),
        };
        // One workgroup per head
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), n_heads, 1, 1);
    }

    pub fn deinit(self: *ElementwiseDispatch) void {
        if (self.pipeline_rms_norm) |*p| p.deinit();
        if (self.pipeline_swiglu) |*p| p.deinit();
        if (self.pipeline_rope) |*p| p.deinit();
        vk.c.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.* = undefined;
    }
};
