//! Shared runtime memory accounting helpers for Vulkan and Metal backends.
//!
//! The helpers in this module turn model dimensions plus backend-specific
//! runtime characteristics into a comparable memory budget so diagnostics,
//! server load policy, and inference engines size context and KV consistently.
const std = @import("std");
const config_mod = @import("../model/config.zig");

const ModelConfig = config_mod.ModelConfig;

/// Backend-agnostic breakdown of runtime memory as:
/// - fixed bytes that do not scale with context length
/// - bytes that scale linearly per token
pub const RuntimeMemoryProfile = struct {
    /// Device-local or GPU-private/shared runtime bytes excluding KV cache.
    fixed_device_local_bytes: u64,
    /// Host-visible / CPU-accessible runtime bytes excluding context-scaled tables.
    fixed_host_visible_bytes: u64,
    /// Device-local bytes added per context token (KV cache).
    device_local_bytes_per_token: u64,
    /// Host-visible bytes added per context token (page table).
    host_visible_bytes_per_token: u64,
    /// GPU SSM state bytes included in `fixed_device_local_bytes`.
    gpu_ssm_bytes: u64,

    pub fn deviceLocalContextBytes(self: @This(), context_tokens: u32) u64 {
        return @as(u64, context_tokens) * self.device_local_bytes_per_token;
    }

    pub fn hostVisibleContextBytes(self: @This(), context_tokens: u32) u64 {
        return @as(u64, context_tokens) * self.host_visible_bytes_per_token;
    }

    pub fn runtimeDeviceLocalBytes(self: @This(), context_tokens: u32) u64 {
        return self.fixed_device_local_bytes + self.deviceLocalContextBytes(context_tokens);
    }

    pub fn runtimeHostVisibleBytes(self: @This(), context_tokens: u32) u64 {
        return self.fixed_host_visible_bytes + self.hostVisibleContextBytes(context_tokens);
    }

    pub fn runtimeUnifiedBytes(self: @This(), context_tokens: u32) u64 {
        return self.runtimeDeviceLocalBytes(context_tokens) + self.runtimeHostVisibleBytes(context_tokens);
    }

    pub fn totalDeviceLocalBytes(self: @This(), weights_bytes: u64, context_tokens: u32) u64 {
        return weights_bytes + self.runtimeDeviceLocalBytes(context_tokens);
    }

    pub fn totalUnifiedBytes(self: @This(), weights_bytes: u64, context_tokens: u32) u64 {
        return weights_bytes + self.runtimeUnifiedBytes(context_tokens);
    }

    pub fn maxContextTokensForDeviceLocalBudget(
        self: @This(),
        weights_bytes: u64,
        budget_bytes: u64,
        ceiling: u32,
    ) u32 {
        return maxContextTokensForBudget(
            weights_bytes,
            budget_bytes,
            self.fixed_device_local_bytes,
            self.device_local_bytes_per_token,
            ceiling,
        );
    }

    pub fn maxContextTokensForUnifiedBudget(
        self: @This(),
        weights_bytes: u64,
        budget_bytes: u64,
        ceiling: u32,
    ) u32 {
        return maxContextTokensForBudget(
            weights_bytes,
            budget_bytes,
            self.fixed_device_local_bytes + self.fixed_host_visible_bytes,
            self.device_local_bytes_per_token + self.host_visible_bytes_per_token,
            ceiling,
        );
    }
};

pub fn effectiveContextCeiling(config: ModelConfig, requested_context_length: ?u32) u32 {
    return @min(config.context_length, requested_context_length orelse config.context_length);
}

pub fn applyRequestedContextLimit(config: *ModelConfig, requested_context_length: ?u32) void {
    config.context_length = effectiveContextCeiling(config.*, requested_context_length);
}

pub fn requestedContextTokens(config: ModelConfig, requested_context_length: ?u32, backend_cap: u32) u32 {
    return @min(effectiveContextCeiling(config, requested_context_length), backend_cap);
}

pub fn remainingContextTokens(used_context_tokens: u32, context_capacity_tokens: u32) u32 {
    return context_capacity_tokens -| used_context_tokens;
}

pub fn clampedCompletionTokens(
    used_context_tokens: u32,
    requested_completion_tokens: u32,
    context_capacity_tokens: u32,
) u32 {
    return @min(requested_completion_tokens, remainingContextTokens(used_context_tokens, context_capacity_tokens));
}

pub fn requestContextTarget(
    used_context_tokens: u32,
    requested_completion_tokens: u32,
    context_capacity_tokens: u32,
) u32 {
    return @min(
        context_capacity_tokens,
        used_context_tokens +| clampedCompletionTokens(used_context_tokens, requested_completion_tokens, context_capacity_tokens),
    );
}

pub const RequestBudget = struct {
    completion_tokens: u32,
    target_context_tokens: u32,
};

pub fn requestBudget(
    used_context_tokens: u32,
    requested_completion_tokens: u32,
    context_capacity_tokens: u32,
) RequestBudget {
    const completion_tokens = clampedCompletionTokens(
        used_context_tokens,
        requested_completion_tokens,
        context_capacity_tokens,
    );
    return .{
        .completion_tokens = completion_tokens,
        .target_context_tokens = requestContextTarget(
            used_context_tokens,
            requested_completion_tokens,
            context_capacity_tokens,
        ),
    };
}

pub fn profile(config: ModelConfig) RuntimeMemoryProfile {
    const hidden_size = @as(u64, config.hidden_dim) * @sizeOf(f32);
    const logits_size = @as(u64, config.vocab_size) * @sizeOf(f32);
    const q_dim = @as(u64, config.n_heads) * config.head_dim;
    const kv_dim = @as(u64, config.n_kv_heads) * config.head_dim;
    const q_size = q_dim * @sizeOf(f32);
    const kv_size = kv_dim * @sizeOf(f32);
    const inter_dim = if (config.intermediate_dim > 0) config.intermediate_dim else config.hidden_dim * 4;
    const shexp_inter = if (config.shared_expert_intermediate_dim > 0) config.shared_expert_intermediate_dim else inter_dim;
    const ssm_conv_channels: u32 = if (config.ssm_d_inner > 0) config.ssm_d_inner + 2 * config.ssm_n_group * config.ssm_d_state else 0;
    const max_inter = @max(
        @max(inter_dim, shexp_inter),
        @max(if (config.ssm_d_inner > 0) config.ssm_d_inner else inter_dim, ssm_conv_channels),
    );
    const inter_size = @as(u64, max_inter) * @sizeOf(f32);
    const n_experts_total: u32 = if (config.n_experts > 0) config.n_experts else 1;
    const n_experts_used: u32 = if (config.n_experts_used > 0) config.n_experts_used else 8;
    const batched_inter_size = @as(u64, n_experts_used) * inter_dim * @sizeOf(f32);
    const batched_down_size = @as(u64, n_experts_used) * hidden_size;
    const gate_buf_size = @max(inter_size, batched_inter_size);
    const down_buf_size = @max(hidden_size, batched_down_size);
    const q_full_size = @as(u64, q_dim * 2) * @sizeOf(f32);
    const conv_ch: u32 = if (config.ssm_d_inner > 0) config.ssm_d_inner + 2 * config.ssm_n_group * config.ssm_d_state else 0;
    const attn_out_size = @max(q_full_size, @as(u64, conv_ch) * @sizeOf(f32));
    const router_size = @as(u64, n_experts_total) * @sizeOf(f32);

    const gpu_ssm_bytes = blk: {
        if (config.ssm_d_inner == 0) break :blk @as(u64, 0);
        const d_inner = config.ssm_d_inner;
        const dt_rank = config.ssm_dt_rank;
        if (dt_rank == 0) break :blk @as(u64, 0);
        const head_v_dim = d_inner / dt_rank;
        const gpu_conv_ch = d_inner + 2 * config.ssm_n_group * config.ssm_d_state;
        const gpu_conv_size = @as(u64, (config.ssm_d_conv - 1) * gpu_conv_ch) * @sizeOf(f32);
        const gpu_state_size = @as(u64, dt_rank) * head_v_dim * head_v_dim * @sizeOf(f32);
        break :blk @as(u64, config.n_layers) * (gpu_conv_size + gpu_state_size);
    };

    const ssm_staging_size = @max(hidden_size, @as(u64, if (config.ssm_d_inner > 0) config.ssm_d_inner else config.hidden_dim) * @sizeOf(f32));
    const router_out_size = @as(u64, n_experts_used) * (@sizeOf(u32) + @sizeOf(f32));

    return .{
        .fixed_device_local_bytes = hidden_size + hidden_size + hidden_size + logits_size +
            q_size + kv_size + kv_size + attn_out_size + hidden_size + hidden_size +
            gate_buf_size + gate_buf_size + gate_buf_size + down_buf_size + hidden_size +
            router_size + gpu_ssm_bytes,
        .fixed_host_visible_bytes = logits_size + hidden_size + router_size + ssm_staging_size + router_out_size,
        .device_local_bytes_per_token = @as(u64, config.n_layers) * kv_dim * @sizeOf(f32) * 2,
        .host_visible_bytes_per_token = @sizeOf(u32),
        .gpu_ssm_bytes = gpu_ssm_bytes,
    };
}

fn maxContextTokensForBudget(
    weights_bytes: u64,
    budget_bytes: u64,
    fixed_runtime_bytes: u64,
    bytes_per_token: u64,
    ceiling: u32,
) u32 {
    if (ceiling == 0) return 0;
    const required_without_context = weights_bytes + fixed_runtime_bytes;
    if (required_without_context >= budget_bytes) return 0;
    if (bytes_per_token == 0) return ceiling;

    const available = budget_bytes - required_without_context;
    const max_tokens_u64 = @divTrunc(available, bytes_per_token);
    const capped: u64 = @min(max_tokens_u64, @as(u64, ceiling));
    return @intCast(capped);
}

test "profile reports context-scaled and fixed bytes" {
    const cfg = ModelConfig{
        .architecture = .qwen2_moe,
        .n_layers = 40,
        .n_heads = 16,
        .n_kv_heads = 2,
        .head_dim = 256,
        .hidden_dim = 2048,
        .intermediate_dim = 512,
        .vocab_size = 248320,
        .context_length = 32768,
        .rope_freq_base = 10000000.0,
        .rms_norm_eps = 1e-6,
        .n_experts = 256,
        .n_experts_used = 8,
        .rope_dim = 64,
        .ssm_d_conv = 4,
        .ssm_d_inner = 4096,
        .ssm_d_state = 128,
        .ssm_dt_rank = 32,
        .ssm_n_group = 16,
        .full_attn_interval = 4,
        .shared_expert_intermediate_dim = 512,
    };

    const p = profile(cfg);
    const ctx: u32 = 4096;
    try std.testing.expect(p.deviceLocalContextBytes(ctx) > 0);
    try std.testing.expect(p.runtimeDeviceLocalBytes(ctx) > p.deviceLocalContextBytes(ctx));
    try std.testing.expect(p.runtimeHostVisibleBytes(ctx) > p.hostVisibleContextBytes(ctx));
    try std.testing.expect(p.gpu_ssm_bytes > 0);
}

test "maxContextTokensForBudget clamps to available budget and ceiling" {
    const profile_value = RuntimeMemoryProfile{
        .fixed_device_local_bytes = 6 * 1024,
        .fixed_host_visible_bytes = 2 * 1024,
        .device_local_bytes_per_token = 256,
        .host_visible_bytes_per_token = 4,
        .gpu_ssm_bytes = 0,
    };

    try std.testing.expectEqual(@as(u32, 8), profile_value.maxContextTokensForDeviceLocalBudget(0, 8 * 1024, 32));
    try std.testing.expectEqual(@as(u32, 16), profile_value.maxContextTokensForDeviceLocalBudget(0, 16 * 1024, 16));
    try std.testing.expectEqual(@as(u32, 0), profile_value.maxContextTokensForUnifiedBudget(10 * 1024, 8 * 1024, 32));
}

test "requested context helpers clamp to model ceiling and backend cap" {
    var config = ModelConfig{
        .architecture = .qwen2_moe,
        .n_layers = 40,
        .n_heads = 16,
        .n_kv_heads = 2,
        .head_dim = 256,
        .hidden_dim = 2048,
        .intermediate_dim = 512,
        .vocab_size = 248320,
        .context_length = 32768,
        .rope_freq_base = 10000000.0,
        .rms_norm_eps = 1e-6,
        .n_experts = 256,
        .n_experts_used = 8,
        .rope_dim = 64,
        .ssm_d_conv = 4,
        .ssm_d_inner = 4096,
        .ssm_d_state = 128,
        .ssm_dt_rank = 32,
        .ssm_n_group = 16,
        .full_attn_interval = 4,
        .shared_expert_intermediate_dim = 512,
    };

    try std.testing.expectEqual(@as(u32, 8192), effectiveContextCeiling(config, 8192));
    try std.testing.expectEqual(@as(u32, 4096), requestedContextTokens(config, 8192, 4096));

    applyRequestedContextLimit(&config, 65536);
    try std.testing.expectEqual(@as(u32, 32768), config.context_length);
}

test "request context helpers clamp completion budget to remaining capacity" {
    try std.testing.expectEqual(@as(u32, 3996), remainingContextTokens(100, 4096));
    try std.testing.expectEqual(@as(u32, 64), clampedCompletionTokens(32, 64, 4096));
    try std.testing.expectEqual(@as(u32, 6), clampedCompletionTokens(4090, 64, 4096));
    try std.testing.expectEqual(@as(u32, 0), clampedCompletionTokens(5000, 64, 4096));
    try std.testing.expectEqual(@as(u32, 96), requestContextTarget(32, 64, 4096));
    try std.testing.expectEqual(@as(u32, 4096), requestContextTarget(4090, 64, 4096));

    const unclamped = requestBudget(32, 64, 4096);
    try std.testing.expectEqual(@as(u32, 64), unclamped.completion_tokens);
    try std.testing.expectEqual(@as(u32, 96), unclamped.target_context_tokens);

    const clamped = requestBudget(4090, 64, 4096);
    try std.testing.expectEqual(@as(u32, 6), clamped.completion_tokens);
    try std.testing.expectEqual(@as(u32, 4096), clamped.target_context_tokens);
}
