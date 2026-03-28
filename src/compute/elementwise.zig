//! Wrap the fused element-wise shader family used by the decode loop.
//! @section Shader Dispatch
//! This helper loads the RMS norm, SwiGLU, and RoPE pipelines and records the
//! push constants needed for their dispatches.
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

/// Push constants for vector add shader.
const VaddPush = extern struct {
    N: u32,
};

/// Push constants for deinterleave shader.
const DeinterleavePush = extern struct {
    head_dim: u32,
    n_heads: u32,
};

/// Push constants for sigmoid multiply shader.
const SigmoidMulPush = extern struct {
    N: u32,
};

/// Push constants for scale-accumulate shader.
const ScaleAccPush = extern struct {
    N: u32,
    scale_bits: u32, // float reinterpreted as u32
};

/// Push constants for RoPE shader (with partial rotation / IMRoPE support).
const RopePush = extern struct {
    stride: u32, // full head dimension (distance between heads in memory)
    rope_dim: u32, // number of dimensions to rotate (<= stride)
    n_heads: u32,
    position: u32,
    freq_base_bits: u32, // float bits reinterpreted as u32
};

/// Push constants for SSM conv1d + SiLU shader.
pub const SsmConv1dPush = extern struct {
    conv_channels: u32,
    d_conv: u32,
    kernel_is_f16: u32,
};

/// Push constants for SSM delta-net state update shader.
pub const SsmDeltaNetPush = extern struct {
    d_inner: u32,
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    n_group: u32,
    ssm_a_is_f16: u32,
    dt_bias_is_f16: u32,
    has_dt_bias: u32,
    has_ssm_a: u32,
};

/// Push constants for SSM gated norm shader.
pub const SsmGatedNormPush = extern struct {
    d_inner: u32,
    dt_rank: u32,
    head_v_dim: u32,
    d_state: u32,
    norm_per_head: u32,
};

/// Push constants for softmax + top-k MoE router shader.
pub const SoftmaxTopkPush = extern struct {
    n_experts: u32,
    k: u32,
};

/// Manages element-wise fused kernel pipelines.
pub const ElementwiseDispatch = struct {
    /// RMS NORM pipeline, or null.
    pipeline_rms_norm: ?Pipeline,
    /// SWIGLU pipeline, or null.
    pipeline_swiglu: ?Pipeline,
    /// ROPE pipeline, or null.
    pipeline_rope: ?Pipeline,
    /// DEINTERLEAVE pipeline, or null.
    pipeline_deinterleave: ?Pipeline,
    /// SIGMOID MUL pipeline, or null.
    pipeline_sigmoid_mul: ?Pipeline,
    /// VADD pipeline, or null.
    pipeline_vadd: ?Pipeline,
    /// SCALE ACC pipeline, or null.
    pipeline_scale_acc: ?Pipeline,
    /// SSM CONV1D pipeline, or null.
    pipeline_ssm_conv1d: ?Pipeline,
    /// SSM DELTA NET pipeline, or null.
    pipeline_ssm_delta_net: ?Pipeline,
    /// SSM GATED NORM pipeline, or null.
    pipeline_ssm_gated_norm: ?Pipeline,
    /// SOFTMAX TOPK pipeline, or null.
    pipeline_softmax_topk: ?Pipeline,
    /// SIGMOID SCALE ACC pipeline: a[i] += sigmoid(c[0]) * b[i], 3 bindings.
    pipeline_sigmoid_scale_acc: ?Pipeline,
    /// Descriptor pool for this dispatch.
    descriptor_pool: vk.c.VkDescriptorPool,
    /// Logical device.
    device: vk.c.VkDevice,

    /// Create the fused element-wise dispatch wrapper and load its shaders.
    /// @param instance Active Vulkan instance and logical device.
    /// @param shader_dir Directory containing compiled SPIR-V shader binaries.
    /// @param allocator Allocator used for temporary pipeline creation state.
    /// @returns An ElementwiseDispatch ready to record element-wise passes.
    pub fn init(
        /// Vulkan instance.
        instance: *const Instance,
        shader_dir: []const u8,
        /// Allocator for owned resources.
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

        // deinterleave: 1 input + 2 outputs = 3 bindings
        const deinterleave_path = std.fmt.bufPrint(&path_buf, "{s}/deinterleave.spv", .{shader_dir}) catch unreachable;
        const pipeline_deinterleave = pipeline_mod.createFromSpirv(instance, deinterleave_path, 3, @sizeOf(DeinterleavePush), &.{}, allocator) catch |err| blk: {
            log.warn("deinterleave shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // sigmoid_mul: 2 inputs + 1 output = 3 bindings
        const sigmoid_path = std.fmt.bufPrint(&path_buf, "{s}/sigmoid_mul.spv", .{shader_dir}) catch unreachable;
        const pipeline_sigmoid_mul = pipeline_mod.createFromSpirv(instance, sigmoid_path, 3, @sizeOf(SigmoidMulPush), &.{}, allocator) catch |err| blk: {
            log.warn("sigmoid_mul shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // vadd: 2 inputs + 1 output = 3 bindings
        const vadd_path = std.fmt.bufPrint(&path_buf, "{s}/vadd.spv", .{shader_dir}) catch unreachable;
        const pipeline_vadd = pipeline_mod.createFromSpirv(instance, vadd_path, 3, @sizeOf(VaddPush), &.{}, allocator) catch |err| blk: {
            log.warn("vadd shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // scale_accumulate: 1 read-write + 1 read = 2 bindings
        const sacc_path = std.fmt.bufPrint(&path_buf, "{s}/scale_accumulate.spv", .{shader_dir}) catch unreachable;
        const pipeline_scale_acc = pipeline_mod.createFromSpirv(instance, sacc_path, 2, @sizeOf(ScaleAccPush), &.{}, allocator) catch |err| blk: {
            log.warn("scale_accumulate shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // SSM conv1d + SiLU: 4 bindings (input, kernel, state, output)
        const conv1d_path = std.fmt.bufPrint(&path_buf, "{s}/ssm_conv1d.spv", .{shader_dir}) catch unreachable;
        const pipeline_ssm_conv1d = pipeline_mod.createFromSpirv(instance, conv1d_path, 4, @sizeOf(SsmConv1dPush), &.{}, allocator) catch |err| blk: {
            log.warn("ssm_conv1d shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // SSM delta-net: 7 bindings (conv_out, dt_bias, alpha, beta, ssm_a, state, output)
        const delta_path = std.fmt.bufPrint(&path_buf, "{s}/ssm_delta_net.spv", .{shader_dir}) catch unreachable;
        const pipeline_ssm_delta_net = pipeline_mod.createFromSpirv(instance, delta_path, 7, @sizeOf(SsmDeltaNetPush), &.{}, allocator) catch |err| blk: {
            log.warn("ssm_delta_net shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // SSM gated norm: 4 bindings (delta_output, z_gate, norm_weights, output)
        const gnorm_path = std.fmt.bufPrint(&path_buf, "{s}/ssm_gated_norm.spv", .{shader_dir}) catch unreachable;
        const pipeline_ssm_gated_norm = pipeline_mod.createFromSpirv(instance, gnorm_path, 4, @sizeOf(SsmGatedNormPush), &.{}, allocator) catch |err| blk: {
            log.warn("ssm_gated_norm shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // Softmax + top-k: 2 bindings (logits, output)
        const topk_path = std.fmt.bufPrint(&path_buf, "{s}/softmax_topk.spv", .{shader_dir}) catch unreachable;
        const pipeline_softmax_topk = pipeline_mod.createFromSpirv(instance, topk_path, 2, @sizeOf(SoftmaxTopkPush), &.{}, allocator) catch |err| blk: {
            log.warn("softmax_topk shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        // sigmoid_scale_acc: a[i] += sigmoid(c[0]) * b[i], 3 bindings (accum, src, gate)
        const ssa_path = std.fmt.bufPrint(&path_buf, "{s}/sigmoid_scale_acc.spv", .{shader_dir}) catch unreachable;
        const pipeline_sigmoid_scale_acc = pipeline_mod.createFromSpirv(instance, ssa_path, 3, @sizeOf(ScaleAccPush), &.{}, allocator) catch |err| blk: {
            log.warn("sigmoid_scale_acc shader not loaded: {s}", .{@errorName(err)});
            break :blk null;
        };

        return ElementwiseDispatch{
            .pipeline_rms_norm = pipeline_rms_norm,
            .pipeline_swiglu = pipeline_swiglu,
            .pipeline_rope = pipeline_rope,
            .pipeline_deinterleave = pipeline_deinterleave,
            .pipeline_sigmoid_mul = pipeline_sigmoid_mul,
            .pipeline_vadd = pipeline_vadd,
            .pipeline_scale_acc = pipeline_scale_acc,
            .pipeline_ssm_conv1d = pipeline_ssm_conv1d,
            .pipeline_ssm_delta_net = pipeline_ssm_delta_net,
            .pipeline_ssm_gated_norm = pipeline_ssm_gated_norm,
            .pipeline_softmax_topk = pipeline_softmax_topk,
            .pipeline_sigmoid_scale_acc = pipeline_sigmoid_scale_acc,
            .descriptor_pool = descriptor_pool,
            .device = instance.device,
        };
    }

    /// Record an RMS-norm-plus-scale dispatch for a batch of tokens.
    ///
    /// This binds the fused normalization shader used before attention and MLP
    /// projections so each token is normalized against its hidden dimension.
    /// @param self Dispatch wrapper containing the RMS norm pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set containing input, weight, and output buffers.
    /// @param hidden_dim Hidden width processed per token.
    /// @param n_tokens Number of tokens covered by the dispatch.
    /// @param eps Numerical stability epsilon passed to the shader.
    /// @returns `error.ShaderNotLoaded` when the RMS norm pipeline is unavailable.
    /// @note The helper dispatches one workgroup per token.
    pub fn recordRmsNorm(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        /// Hidden state width.
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

    /// Record a SwiGLU activation dispatch.
    /// @param self Dispatch wrapper containing the SwiGLU pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set containing gate, up, and output buffers.
    /// @param n_elements Total number of output elements to compute.
    /// @returns `error.ShaderNotLoaded` when the SwiGLU pipeline is unavailable.
    /// @note Workgroups are sized as `ceil(n_elements / 64)`.
    pub fn recordSwiglu(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
    ) !void {
        const pip = if (self.pipeline_swiglu) |*p| p else return error.ShaderNotLoaded;
        const push = SwigluPush{ .N = n_elements };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record a rotary-position-embedding dispatch for the active decode position.
    ///
    /// This applies RoPE in-place semantics through the dedicated shader so
    /// attention inputs are rotated consistently with the current token index.
    /// @param self Dispatch wrapper containing the RoPE pipeline.
    /// @param cmd Command buffer currently being recorded.
    /// @param descriptor_set Descriptor set containing input and output buffers.
    /// @param head_dim Hidden width per attention head.
    /// @param n_heads Number of heads to rotate.
    /// @param position Decode position being encoded.
    /// @param freq_base Base rotary frequency parameter.
    /// @returns `error.ShaderNotLoaded` when the RoPE pipeline is unavailable.
    /// @note The helper dispatches one workgroup per head.
    /// Record a RoPE dispatch with partial rotation support (IMRoPE).
    /// @param stride Full head dimension (distance between heads in buffer).
    /// @param rope_dim Number of dimensions to rotate (rest copied unchanged).
    pub fn recordRope(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        stride: u32,
        /// RoPE dimensions (0 = all).
        rope_dim: u32,
        /// Number of query heads.
        n_heads: u32,
        /// Current token position.
        position: u32,
        freq_base: f32,
    ) !void {
        const pip = if (self.pipeline_rope) |*p| p else return error.ShaderNotLoaded;
        const push = RopePush{
            .stride = stride,
            .rope_dim = rope_dim,
            .n_heads = n_heads,
            .position = position,
            .freq_base_bits = @bitCast(freq_base),
        };
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), n_heads, 1, 1);
    }

    /// Record a deinterleave dispatch: split element-interleaved Q+gate into separate buffers.
    pub fn recordDeinterleave(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        /// Per-head dimension.
        head_dim: u32,
        /// Number of query heads.
        n_heads: u32,
    ) !void {
        const pip = if (self.pipeline_deinterleave) |*p| p else return error.ShaderNotLoaded;
        const push = DeinterleavePush{ .head_dim = head_dim, .n_heads = n_heads };
        const total = head_dim * n_heads;
        const workgroups = (total + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record a sigmoid multiply dispatch: out = input * sigmoid(gate).
    pub fn recordSigmoidMul(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
    ) !void {
        const pip = if (self.pipeline_sigmoid_mul) |*p| p else return error.ShaderNotLoaded;
        const push = SigmoidMulPush{ .N = n_elements };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record a vector add dispatch: c = a + b.
    pub fn recordVadd(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
    ) !void {
        const pip = if (self.pipeline_vadd) |*p| p else return error.ShaderNotLoaded;
        const push = VaddPush{ .N = n_elements };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record a scale-accumulate dispatch: a[i] += scale * b[i].
    pub fn recordScaleAcc(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        /// Allocated descriptor set.
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
        scale: f32,
    ) !void {
        const pip = if (self.pipeline_scale_acc) |*p| p else return error.ShaderNotLoaded;
        const push = ScaleAccPush{ .N = n_elements, .scale_bits = @bitCast(scale) };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record SSM conv1d + SiLU dispatch.
    pub fn recordSsmConv1d(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        conv_channels: u32,
        d_conv: u32,
        kernel_is_f16: bool,
    ) !void {
        const pip = if (self.pipeline_ssm_conv1d) |*p| p else return error.ShaderNotLoaded;
        const push = SsmConv1dPush{
            .conv_channels = conv_channels,
            .d_conv = d_conv,
            .kernel_is_f16 = if (kernel_is_f16) 1 else 0,
        };
        const workgroups = (conv_channels + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Record SSM delta-net state update dispatch.
    pub fn recordSsmDeltaNet(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        push: SsmDeltaNetPush,
    ) !void {
        const pip = if (self.pipeline_ssm_delta_net) |*p| p else return error.ShaderNotLoaded;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), push.dt_rank, 1, 1);
    }

    /// Record SSM gated norm dispatch.
    pub fn recordSsmGatedNorm(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        push: SsmGatedNormPush,
    ) !void {
        const pip = if (self.pipeline_ssm_gated_norm) |*p| p else return error.ShaderNotLoaded;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), push.dt_rank, 1, 1);
    }

    /// Record softmax + top-k MoE router dispatch.
    pub fn recordSoftmaxTopk(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        n_experts: u32,
        k: u32,
    ) !void {
        const pip = if (self.pipeline_softmax_topk) |*p| p else return error.ShaderNotLoaded;
        const push = SoftmaxTopkPush{ .n_experts = n_experts, .k = k };
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), 1, 1, 1);
    }

    /// Record sigmoid-gated scale-accumulate: a[i] += sigmoid(c[0]) * b[i].
    pub fn recordSigmoidScaleAcc(
        self: *const ElementwiseDispatch,
        cmd: *const CommandBuffer,
        descriptor_set: vk.c.VkDescriptorSet,
        n_elements: u32,
    ) !void {
        const pip = if (self.pipeline_sigmoid_scale_acc) |*p| p else return error.ShaderNotLoaded;
        // Push constant only needs N (uses same layout as ScaleAccPush but only N is read)
        const push = ScaleAccPush{ .N = n_elements, .scale_bits = 0 };
        const workgroups = (n_elements + 63) / 64;
        cmd.dispatchWithPush(pip, descriptor_set, std.mem.asBytes(&push), workgroups, 1, 1);
    }

    /// Destroy the loaded pipelines and descriptor pool.
    /// @param self Dispatch wrapper to tear down in place.
    pub fn deinit(self: *ElementwiseDispatch) void {
        if (self.pipeline_rms_norm) |*p| p.deinit();
        if (self.pipeline_swiglu) |*p| p.deinit();
        if (self.pipeline_rope) |*p| p.deinit();
        if (self.pipeline_deinterleave) |*p| p.deinit();
        if (self.pipeline_sigmoid_mul) |*p| p.deinit();
        if (self.pipeline_vadd) |*p| p.deinit();
        if (self.pipeline_scale_acc) |*p| p.deinit();
        if (self.pipeline_ssm_conv1d) |*p| p.deinit();
        if (self.pipeline_ssm_delta_net) |*p| p.deinit();
        if (self.pipeline_ssm_gated_norm) |*p| p.deinit();
        if (self.pipeline_softmax_topk) |*p| p.deinit();
        if (self.pipeline_sigmoid_scale_acc) |*p| p.deinit();
        vk.c.vkDestroyDescriptorPool(self.device, self.descriptor_pool, null);
        self.* = undefined;
    }
};
