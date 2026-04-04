//! Metal-specific model loading — zero-copy via mmap + newBufferWithBytesNoCopy.
//! This replaces the Vulkan loader's staging-buffer DMA with direct mmap wrapping.
const std = @import("std");
const gguf = @import("gguf.zig");
const config_mod = @import("config.zig");
const ModelConfig = config_mod.ModelConfig;
const shim = @import("../metal/c.zig").shim;
const metal_buffer = @import("../metal/buffer.zig");
const MetalBuffer = metal_buffer.MetalBuffer;

const log = std.log.scoped(.loader);

/// Summary returned by `inspectModel`: config plus file and tensor size statistics.
pub const ModelInspection = struct {
    config: ModelConfig,
    file_size: u64,
    tensor_bytes: u64,
    tensor_count: u64,
    metadata_count: usize,
};

/// A tensor descriptor paired with a Metal buffer wrapping its mmap'd data.
pub const LoadedTensor = struct {
    info: gguf.TensorInfo,
    gpu_buffer: MetalBuffer,
};

/// Runtime model state backed by a memory-mapped GGUF file and zero-copy Metal buffers.
pub const Model = struct {
    config: ModelConfig,
    gguf_file: gguf.GGUFFile,
    tensors: std.ArrayList(LoadedTensor),
    mmap_data: ?[]align(std.heap.page_size_min) const u8,
    mmap_file: ?std.fs.File,
    allocator: std.mem.Allocator,

    /// Release Metal buffers, GGUF metadata, and the backing file mapping.
    pub fn deinit(self: *Model) void {
        for (self.tensors.items) |*t| {
            metal_buffer.freeBuffer(&t.gpu_buffer);
        }
        self.tensors.deinit(self.allocator);

        if (self.mmap_data) |data| {
            std.posix.munmap(data);
        }
        if (self.mmap_file) |f| {
            var file = f;
            file.close();
        }

        self.gguf_file.deinit();
        self.* = undefined;
    }
};

/// Extract model configuration from GGUF metadata (platform-independent).
fn extractConfigWithLogging(gf: *const gguf.GGUFFile, log_metadata: bool) ModelConfig {
    const arch_str = gf.getString("general.architecture") orelse "unknown";
    const arch = config_mod.parseArchitecture(arch_str);
    const prefix = arch_str;

    var key_buf: [128]u8 = undefined;

    const n_layers = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.block_count", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };
    const n_heads = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.attention.head_count", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };
    const n_kv_heads = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.attention.head_count_kv", .{prefix}) catch break :blk n_heads;
        break :blk gf.getU32(key) orelse n_heads;
    };
    const hidden_dim = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.embedding_length", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };
    const head_dim = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.attention.key_length", .{prefix}) catch break :blk if (n_heads > 0) hidden_dim / n_heads else @as(u32, 0);
        break :blk gf.getU32(key) orelse (if (n_heads > 0) hidden_dim / n_heads else 0);
    };
    const intermediate_dim = blk: {
        const exp_key = std.fmt.bufPrint(&key_buf, "{s}.expert_feed_forward_length", .{prefix}) catch break :blk @as(u32, 0);
        const exp_val = gf.getU32(exp_key);
        if (exp_val) |v| if (v > 0) break :blk v;
        const key = std.fmt.bufPrint(&key_buf, "{s}.feed_forward_length", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };
    const shared_expert_intermediate_dim = blk: {
        const shared_key = std.fmt.bufPrint(&key_buf, "{s}.expert_shared_feed_forward_length", .{prefix}) catch break :blk @as(u32, 0);
        if (gf.getU32(shared_key)) |v| {
            if (v > 0) break :blk v;
        }
        const key = std.fmt.bufPrint(&key_buf, "{s}.feed_forward_length", .{prefix}) catch break :blk @as(u32, 0);
        if (gf.getU32(key)) |v| {
            if (v > 0) break :blk v;
        }

        var name_buf: [96]u8 = undefined;
        for (0..n_layers) |layer| {
            const gate_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_gate_shexp.weight", .{layer}) catch break;
            if (gf.findTensor(gate_name)) |t| break :blk @as(u32, @intCast(t.dims[1]));

            const up_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_up_shexp.weight", .{layer}) catch break;
            if (gf.findTensor(up_name)) |t| break :blk @as(u32, @intCast(t.dims[1]));

            const down_name = std.fmt.bufPrint(&name_buf, "blk.{d}.ffn_down_shexp.weight", .{layer}) catch break;
            if (gf.findTensor(down_name)) |t| break :blk @as(u32, @intCast(t.dims[0]));
        }

        break :blk @as(u32, 0);
    };
    const vocab_size = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.vocab_size", .{prefix}) catch break :blk @as(u32, 0);
        const from_meta = gf.getU32(key);
        if (from_meta) |v| if (v > 0) break :blk v;
        if (gf.findTensor("output.weight")) |t| break :blk @as(u32, @intCast(t.dims[1]));
        if (gf.findTensor("token_embd.weight")) |t| break :blk @as(u32, @intCast(t.dims[1]));
        break :blk @as(u32, 0);
    };
    const context_length = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.context_length", .{prefix}) catch break :blk @as(u32, 4096);
        break :blk gf.getU32(key) orelse 4096;
    };
    const n_experts = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.expert_count", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };
    const n_experts_used = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.expert_used_count", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };
    const rope_dim = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.rope.dimension_count", .{prefix}) catch "") orelse 0;
    const rms_norm_eps = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.attention.layer_norm_rms_epsilon", .{prefix}) catch break :blk @as(f32, 1e-6);
        break :blk gf.getF32(key) orelse 1e-6;
    };
    const ssm_d_conv = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.ssm.conv_kernel", .{prefix}) catch "") orelse 0;
    const ssm_d_inner = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.ssm.inner_size", .{prefix}) catch "") orelse 0;
    const ssm_d_state = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.ssm.state_size", .{prefix}) catch "") orelse 0;
    const ssm_dt_rank = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.ssm.time_step_rank", .{prefix}) catch "") orelse 0;
    const ssm_n_group = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.ssm.group_count", .{prefix}) catch "") orelse 0;
    const full_attn_interval = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.full_attention_interval", .{prefix}) catch "") orelse
        if (ssm_d_inner > 0) @as(u32, 4) else @as(u32, 1);

    const rope_freq_base: f32 = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.rope.freq_base", .{prefix}) catch break :blk @as(f32, 10000.0);
        const val = gf.metadata.get(key);
        if (val) |v| {
            switch (v) {
                .float32 => |f| break :blk f,
                .uint32 => |u| break :blk @floatFromInt(u),
                else => {},
            }
        }
        break :blk @as(f32, 10000.0);
    };

    if (log_metadata) {
        log.info("Architecture: {s} | {d} layers | {d} heads ({d} KV) | dim {d} | vocab {d}", .{
            arch_str, n_layers, n_heads, n_kv_heads, hidden_dim, vocab_size,
        });
        if (n_experts > 0) {
            log.info("MoE: {d} experts, {d} active | intermediate {d} | shared expert {d}", .{
                n_experts, n_experts_used, intermediate_dim, shared_expert_intermediate_dim,
            });
        }
        if (ssm_d_inner > 0) {
            log.info("SSM: d_conv={d} d_inner={d} d_state={d} dt_rank={d} n_group={d}", .{
                ssm_d_conv, ssm_d_inner, ssm_d_state, ssm_dt_rank, ssm_n_group,
            });
        }
    }

    return ModelConfig{
        .architecture = arch,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .hidden_dim = hidden_dim,
        .intermediate_dim = intermediate_dim,
        .vocab_size = vocab_size,
        .context_length = context_length,
        .rope_freq_base = rope_freq_base,
        .rms_norm_eps = rms_norm_eps,
        .n_experts = n_experts,
        .n_experts_used = n_experts_used,
        .rope_dim = rope_dim,
        .ssm_d_conv = ssm_d_conv,
        .ssm_d_inner = ssm_d_inner,
        .ssm_d_state = ssm_d_state,
        .ssm_dt_rank = ssm_dt_rank,
        .ssm_n_group = ssm_n_group,
        .full_attn_interval = full_attn_interval,
        .shared_expert_intermediate_dim = shared_expert_intermediate_dim,
    };
}

fn extractConfig(gf: *const gguf.GGUFFile) ModelConfig {
    return extractConfigWithLogging(gf, true);
}

/// Inspect a GGUF file and extract only the model configuration (no GPU operations).
pub fn inspectConfig(path: []const u8, allocator: std.mem.Allocator) !ModelConfig {
    const file = try std.fs.cwd().openFile(path, .{});
    defer {
        var close_file = file;
        close_file.close();
    }

    const stat = try file.stat();
    const mmap_data = try std.posix.mmap(
        null,
        stat.size,
        std.posix.PROT.READ,
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    defer std.posix.munmap(mmap_data);

    var gf = try gguf.parseWithOptions(mmap_data, allocator, .{ .log_summary = false });
    defer gf.deinit();

    return extractConfigWithLogging(&gf, false);
}

/// Inspect a GGUF file and return exact tensor upload bytes plus normalized config.
pub fn inspectModel(path: []const u8, allocator: std.mem.Allocator) !ModelInspection {
    const file = try std.fs.cwd().openFile(path, .{});
    defer {
        var close_file = file;
        close_file.close();
    }

    const stat = try file.stat();
    const mmap_data = try std.posix.mmap(
        null,
        stat.size,
        std.posix.PROT.READ,
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    defer std.posix.munmap(mmap_data);

    var gf = try gguf.parseWithOptions(mmap_data, allocator, .{ .log_summary = false });
    defer gf.deinit();

    var tensor_bytes: u64 = 0;
    for (gf.tensors.items) |tensor_info| {
        tensor_bytes += tensor_info.sizeBytes();
    }

    return .{
        .config = extractConfigWithLogging(&gf, false),
        .file_size = stat.size,
        .tensor_bytes = tensor_bytes,
        .tensor_count = gf.tensor_count,
        .metadata_count = gf.metadata.count(),
    };
}

/// Load a GGUF model with zero-copy Metal buffers wrapping mmap'd tensor data.
pub fn load(
    path: []const u8,
    metal_ctx: ?*shim.MetalCtx,
    allocator: std.mem.Allocator,
) !Model {
    log.info("Loading model: {s}", .{path});

    const file = try std.fs.cwd().openFile(path, .{});
    errdefer file.close();

    const stat = try file.stat();
    const file_size = stat.size;
    log.info("File size: {d} MB", .{file_size / (1024 * 1024)});

    const mmap_data = try std.posix.mmap(
        null,
        file_size,
        std.posix.PROT.READ,
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    errdefer std.posix.munmap(mmap_data);

    var gf = try gguf.parse(mmap_data, allocator);
    errdefer gf.deinit();

    const config = extractConfig(&gf);

    if (config.architecture == .unknown) {
        log.err("Unsupported model architecture. Supported: qwen2, qwen2_moe, qwen35, mistral, mamba, jamba", .{});
        return error.UnsupportedArchitecture;
    }

    // Wrap tensor data as Metal shared buffers (zero-copy from mmap)
    var loaded_tensors: std.ArrayList(LoadedTensor) = .{};
    errdefer {
        for (loaded_tensors.items) |*t| {
            metal_buffer.freeBuffer(&t.gpu_buffer);
        }
        loaded_tensors.deinit(allocator);
    }

    var total_size: u64 = 0;
    for (gf.tensors.items) |tensor_info| {
        const tensor_size = tensor_info.sizeBytes();
        const data_offset = gf.tensor_data_offset + tensor_info.offset;

        // Page-align the offset for Metal buffer wrapping
        const page_size: u64 = std.heap.page_size_min;
        const aligned_offset = (data_offset / page_size) * page_size;
        const offset_within_page = data_offset - aligned_offset;
        const aligned_size = ((tensor_size + offset_within_page + page_size - 1) / page_size) * page_size;

        // Wrap the mmap'd region as a Metal buffer
        const ptr: [*]u8 = @ptrCast(@constCast(mmap_data.ptr + aligned_offset));
        const gpu_buf = metal_buffer.wrapMmap(metal_ctx, ptr, @intCast(aligned_size)) catch |err| {
            log.err("Failed to wrap tensor at offset {d} ({d} bytes): {s}", .{
                data_offset, tensor_size, @errorName(err),
            });
            return err;
        };

        try loaded_tensors.append(allocator, .{
            .info = tensor_info,
            .gpu_buffer = gpu_buf,
        });

        total_size += tensor_size;
    }

    log.info("Loaded {d} tensors | {d} MB (zero-copy mmap)", .{
        loaded_tensors.items.len,
        total_size / (1024 * 1024),
    });

    return Model{
        .config = config,
        .gguf_file = gf,
        .tensors = loaded_tensors,
        .mmap_data = mmap_data,
        .mmap_file = file,
        .allocator = allocator,
    };
}
