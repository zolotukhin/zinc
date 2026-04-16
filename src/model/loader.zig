//! Build runtime model state from GGUF metadata and GPU-resident tensor buffers.
//! @section Model Format & Loading
//! This module translates an on-disk GGUF file into the normalized model
//! configuration and uploaded tensors consumed by the inference runtime.
const std = @import("std");
const gguf = @import("gguf.zig");
const config_mod = @import("config.zig");
const vk = @import("../vulkan/vk.zig");
const Instance = @import("../vulkan/instance.zig").Instance;
const Buffer = @import("../vulkan/buffer.zig").Buffer;
const buffer_mod = @import("../vulkan/buffer.zig");
const CommandPool = @import("../vulkan/command.zig").CommandPool;

const log = std.log.scoped(.loader);

/// Supported model architectures (re-exported from config.zig).
pub const Architecture = config_mod.Architecture;

/// Normalized model dimensions and hyperparameters (re-exported from config.zig).
pub const ModelConfig = config_mod.ModelConfig;

/// Summary returned by `inspectModel`: config plus file and tensor size statistics.
pub const ModelInspection = struct {
    config: ModelConfig,
    file_size: u64,
    tensor_bytes: u64,
    tensor_count: u64,
    metadata_count: usize,
};

/// A tensor descriptor paired with the GPU buffer that stores its contents.
pub const LoadedTensor = struct {
    /// GGUF tensor descriptor.
    info: gguf.TensorInfo,
    /// Device-local GPU buffer.
    gpu_buffer: Buffer,
};

/// Runtime model state backed by a memory-mapped GGUF file and uploaded tensor buffers.
pub const Model = struct {
    /// Model dimensions and metadata.
    config: ModelConfig,
    /// Parsed GGUF header.
    gguf_file: gguf.GGUFFile,
    /// Tensor descriptors.
    tensors: std.ArrayList(LoadedTensor),
    /// Memory-mapped GGUF file view.
    mmap_data: ?[]align(std.heap.page_size_min) const u8,
    /// File handle for mmap.
    mmap_file: ?std.fs.File,
    /// Allocator for owned resources.
    allocator: std.mem.Allocator,

    /// Release tensor buffers, GGUF metadata, and the backing file mapping owned by the model.
    /// @param self Model instance to tear down in place.
    /// @param instance Active Vulkan instance that created the device resources.
    pub fn deinit(self: *Model, instance: *const Instance) void {
        _ = instance;
        for (self.tensors.items) |*t| {
            var buf = t.gpu_buffer;
            buf.deinit();
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

const parseArchitecture = config_mod.parseArchitecture;

/// Extract model configuration from GGUF metadata.
fn extractConfigWithLogging(gf: *const gguf.GGUFFile, log_metadata: bool) ModelConfig {
    const arch_str = gf.getString("general.architecture") orelse "unknown";
    const arch = parseArchitecture(arch_str);
    const prefix = arch_str;

    // Helper to look up arch-prefixed metadata keys
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
        if (gf.getU32(key)) |v| break :blk v;
        // Gemma 4: head_count_kv is a per-layer array. Use the maximum value for buffer sizing.
        if (gf.metadata.get(key)) |val| {
            switch (val) {
                .array => |arr| {
                    var max_kv: u32 = 0;
                    for (arr) |item| {
                        const v = item.asU32() orelse continue;
                        if (v > max_kv) max_kv = v;
                    }
                    if (max_kv > 0) break :blk max_kv;
                },
                else => {},
            }
        }
        break :blk n_heads;
    };

    const hidden_dim = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.embedding_length", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };

    // head_dim: prefer attention.key_length from GGUF (Qwen3.5 uses 256, not hidden_dim/n_heads=128).
    // Gemma 4 has separate key_length (global=512) and key_length_swa (sliding=256).
    // Use the max for buffer allocation; the forward pass derives per-layer dims from tensors.
    const head_dim = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.attention.key_length", .{prefix}) catch break :blk if (n_heads > 0) hidden_dim / n_heads else @as(u32, 0);
        break :blk gf.getU32(key) orelse (if (n_heads > 0) hidden_dim / n_heads else 0);
    };

    const intermediate_dim = blk: {
        // For MoE models: use expert_feed_forward_length (per-expert intermediate dim)
        // Falls back to feed_forward_length, then 0
        const exp_key = std.fmt.bufPrint(&key_buf, "{s}.expert_feed_forward_length", .{prefix}) catch break :blk @as(u32, 0);
        const exp_val = gf.getU32(exp_key);
        if (exp_val) |v| if (v > 0) break :blk v;
        const key = std.fmt.bufPrint(&key_buf, "{s}.feed_forward_length", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };

    // Shared expert intermediate dim: prefer metadata, but Qwen3.5 GGUFs may omit it.
    // Fall back to the actual shared-expert tensor shape when the metadata is zero.
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
        // Try metadata first
        const key = std.fmt.bufPrint(&key_buf, "{s}.vocab_size", .{prefix}) catch break :blk @as(u32, 0);
        const from_meta = gf.getU32(key);
        if (from_meta) |v| if (v > 0) break :blk v;
        // Infer from output.weight or token_embd.weight tensor
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

    // RoPE dimension count (partial rotation / IMRoPE)
    const rope_dim = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.rope.dimension_count", .{prefix}) catch "") orelse 0;
    const rms_norm_eps = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.attention.layer_norm_rms_epsilon", .{prefix}) catch break :blk @as(f32, 1e-6);
        break :blk gf.getF32(key) orelse 1e-6;
    };

    // SSM parameters (hybrid models like Qwen3.5)
    const ssm_d_conv = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.ssm.conv_kernel", .{prefix}) catch "") orelse 0;
    const ssm_d_inner = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.ssm.inner_size", .{prefix}) catch "") orelse 0;
    const ssm_d_state = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.ssm.state_size", .{prefix}) catch "") orelse 0;
    const ssm_dt_rank = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.ssm.time_step_rank", .{prefix}) catch "") orelse 0;
    const ssm_n_group = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.ssm.group_count", .{prefix}) catch "") orelse 0;
    const full_attn_interval = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.full_attention_interval", .{prefix}) catch "") orelse
        if (ssm_d_inner > 0) @as(u32, 4) else @as(u32, 1);

    if (log_metadata) {
        log.info("Architecture: {s} | {d} layers | {d} heads ({d} KV) | dim {d} | vocab {d}", .{
            arch_str, n_layers, n_heads, n_kv_heads, hidden_dim, vocab_size,
        });
        if (rope_dim > 0) {
            log.info("RoPE: dim={d}/{d} freq_base={d:.0}", .{ rope_dim, head_dim, @as(f64, @floatCast(blk: {
                const key3 = std.fmt.bufPrint(&key_buf, "{s}.rope.freq_base", .{prefix}) catch break :blk @as(f32, 10000.0);
                const val2 = gf.metadata.get(key3);
                if (val2) |v| {
                    switch (v) {
                        .float32 => |fv| break :blk fv,
                        else => {},
                    }
                }
                break :blk @as(f32, 10000.0);
            })) });
        }
        if (gf.metadata.get(std.fmt.bufPrint(&key_buf, "{s}.rope.dimension_sections", .{prefix}) catch "")) |sections_val| {
            switch (sections_val) {
                .array => |arr| {
                    var vals: [8]u32 = [_]u32{0} ** 8;
                    const n = @min(arr.len, vals.len);
                    for (arr[0..n], 0..) |item, i| vals[i] = item.asU32() orelse 0;
                    log.info("RoPE sections ({d}): [{d},{d},{d},{d},{d},{d},{d},{d}]", .{
                        arr.len,
                        vals[0],
                        vals[1],
                        vals[2],
                        vals[3],
                        vals[4],
                        vals[5],
                        vals[6],
                        vals[7],
                    });
                },
                else => {},
            }
        }
        log.info("RMSNorm epsilon: {d:.8}", .{rms_norm_eps});
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
        .rope_freq_base = blk: {
            const key2 = std.fmt.bufPrint(&key_buf, "{s}.rope.freq_base", .{prefix}) catch break :blk @as(f32, 10000.0);
            const val = gf.metadata.get(key2);
            if (val) |v| {
                switch (v) {
                    .float32 => |f| break :blk f,
                    .uint32 => |u| break :blk @floatFromInt(u),
                    else => {},
                }
            }
            break :blk @as(f32, 10000.0);
        },
        .rope_freq_base_swa = blk: {
            const swa_key = std.fmt.bufPrint(&key_buf, "{s}.rope.freq_base_swa", .{prefix}) catch break :blk @as(f32, 0);
            const swa_val = gf.metadata.get(swa_key);
            if (swa_val) |v| {
                switch (v) {
                    .float32 => |fv| break :blk fv,
                    .uint32 => |u| break :blk @as(f32, @floatFromInt(u)),
                    else => {},
                }
            }
            break :blk @as(f32, 0);
        },
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
        .final_logit_softcapping = blk: {
            const key4 = std.fmt.bufPrint(&key_buf, "{s}.final_logit_softcapping", .{prefix}) catch break :blk @as(f32, 0.0);
            break :blk gf.getF32(key4) orelse 0.0;
        },
        .attn_scale = blk: {
            const key5 = std.fmt.bufPrint(&key_buf, "{s}.attention.scale", .{prefix}) catch break :blk @as(f32, 0.0);
            if (gf.getF32(key5)) |v| break :blk v;
            // Gemma 4 uses a fixed attention scaling factor of 1.0 even when
            // the GGUF omits an explicit attention.scale key.
            if (std.mem.eql(u8, arch_str, "gemma4")) break :blk @as(f32, 1.0);
            break :blk @as(f32, 0.0);
        },
        .sliding_window_size = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.attention.sliding_window", .{prefix}) catch "") orelse 0,
        .rope_scaling_factor = blk: {
            const rsk = std.fmt.bufPrint(&key_buf, "{s}.rope.scaling.factor", .{prefix}) catch break :blk @as(f32, 0.0);
            break :blk gf.getF32(rsk) orelse 0.0;
        },
        .rope_attn_factor = blk: {
            const atk = std.fmt.bufPrint(&key_buf, "{s}.rope.scaling.attn_factor", .{prefix}) catch break :blk @as(f32, 1.0);
            break :blk gf.getF32(atk) orelse 1.0;
        },
        .rope_original_context = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.rope.scaling.original_context_length", .{prefix}) catch "") orelse 0,
        .rope_sections = blk: {
            var sections = [_]u32{ 0, 0, 0, 0 };
            if (gf.metadata.get(std.fmt.bufPrint(&key_buf, "{s}.rope.dimension_sections", .{prefix}) catch "")) |val| {
                switch (val) {
                    .array => |arr| {
                        const n = @min(arr.len, sections.len);
                        for (arr[0..n], 0..) |item, i| sections[i] = item.asU32() orelse 0;
                    },
                    else => {},
                }
            }
            break :blk sections;
        },
    };
}

fn extractConfig(gf: *const gguf.GGUFFile) ModelConfig {
    return extractConfigWithLogging(gf, true);
}

/// Inspect a GGUF file and extract only the normalized model configuration.
/// @param path Path to the GGUF file on disk.
/// @param allocator Allocator used for the parsed metadata structures.
/// @returns A ModelConfig derived from GGUF metadata without uploading tensors to the GPU.
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

/// Load a GGUF model: memory-map the file, parse headers, and DMA tensors to GPU VRAM.
/// @param path Path to the GGUF file on disk.
/// @param instance Active Vulkan instance used for buffer allocation.
/// @param cmd_pool Command pool used for staging copy operations.
/// @param allocator Allocator used for metadata, tensor lists, and temporary state.
/// @returns A fully populated Model with parsed metadata and uploaded tensors.
pub fn load(
    path: []const u8,
    instance: *const Instance,
    cmd_pool: *const CommandPool,
    allocator: std.mem.Allocator,
) !Model {
    log.info("Loading model: {s}", .{path});

    // Open and memory-map the file
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

    // Parse GGUF headers
    var gf = try gguf.parse(mmap_data, allocator);
    errdefer gf.deinit();

    const config = extractConfig(&gf);

    if (config.architecture == .unknown) {
        log.err("Unsupported model architecture. Supported: qwen2, qwen2_moe, qwen35, mistral, mamba, jamba", .{});
        return error.UnsupportedArchitecture;
    }

    // Load tensors to GPU
    var loaded_tensors: std.ArrayList(LoadedTensor) = .{};
    errdefer {
        for (loaded_tensors.items) |*t| {
            var buf = t.gpu_buffer;
            buf.deinit();
        }
        loaded_tensors.deinit(allocator);
    }

    var total_vram: u64 = 0;
    for (gf.tensors.items) |tensor_info| {
        const tensor_size = tensor_info.sizeBytes();
        const data_offset = gf.tensor_data_offset + tensor_info.offset;

        // Create device-local buffer
        var gpu_buf = try Buffer.initDeviceLocal(
            instance,
            tensor_size,
            vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        );
        errdefer gpu_buf.deinit();

        // Stage and copy data to GPU
        const src_data = mmap_data[data_offset..][0..@intCast(tensor_size)];
        var staging = try Buffer.initStaging(instance, tensor_size);
        defer staging.deinit();

        staging.upload(src_data);
        try buffer_mod.copyBuffer(instance, cmd_pool.handle, &staging, &gpu_buf, tensor_size);

        try loaded_tensors.append(allocator, .{
            .info = tensor_info,
            .gpu_buffer = gpu_buf,
        });

        total_vram += tensor_size;
    }

    log.info("Loaded {d} tensors | {d} MB VRAM", .{
        loaded_tensors.items.len,
        total_vram / (1024 * 1024),
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

test "parseArchitecture" {
    try std.testing.expectEqual(Architecture.qwen2, parseArchitecture("qwen2"));
    try std.testing.expectEqual(Architecture.qwen2_moe, parseArchitecture("qwen2moe"));
    try std.testing.expectEqual(Architecture.qwen35, parseArchitecture("qwen35"));
    try std.testing.expectEqual(Architecture.mamba, parseArchitecture("mamba"));
    try std.testing.expectEqual(Architecture.unknown, parseArchitecture("gpt2"));
}

test "extractConfig defaults gemma4 attention scale to 1.0" {
    const allocator = std.testing.allocator;

    var gf = gguf.GGUFFile{
        .version = .v3,
        .tensor_count = 0,
        .metadata = .{},
        .tensors = .{},
        .tensor_data_offset = 0,
        .allocator = allocator,
    };
    defer gf.deinit();

    try gf.metadata.put(allocator, try allocator.dupe(u8, "general.architecture"), .{ .string = try allocator.dupe(u8, "gemma4") });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gemma4.block_count"), .{ .uint32 = 30 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gemma4.attention.head_count"), .{ .uint32 = 16 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gemma4.embedding_length"), .{ .uint32 = 2816 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gemma4.attention.key_length"), .{ .uint32 = 512 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gemma4.vocab_size"), .{ .uint32 = 262144 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gemma4.context_length"), .{ .uint32 = 8192 });

    const cfg = extractConfigWithLogging(&gf, false);
    try std.testing.expectEqual(@as(f32, 1.0), cfg.attn_scale);
}

test "extractConfig uses max gemma4 head_count_kv array entry" {
    const allocator = std.testing.allocator;

    var gf = gguf.GGUFFile{
        .version = .v3,
        .tensor_count = 0,
        .metadata = .{},
        .tensors = .{},
        .tensor_data_offset = 0,
        .allocator = allocator,
    };
    defer gf.deinit();

    const kv_heads = try allocator.alloc(gguf.MetadataValue, 4);
    kv_heads[0] = .{ .int32 = 2 };
    kv_heads[1] = .{ .int32 = 8 };
    kv_heads[2] = .{ .int32 = 4 };
    kv_heads[3] = .{ .int32 = 1 };

    try gf.metadata.put(allocator, try allocator.dupe(u8, "general.architecture"), .{ .string = try allocator.dupe(u8, "gemma4") });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gemma4.block_count"), .{ .uint32 = 30 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gemma4.attention.head_count"), .{ .uint32 = 16 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gemma4.attention.head_count_kv"), .{ .array = kv_heads });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gemma4.embedding_length"), .{ .uint32 = 2816 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gemma4.attention.key_length"), .{ .uint32 = 512 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gemma4.vocab_size"), .{ .uint32 = 262144 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gemma4.context_length"), .{ .uint32 = 8192 });

    const cfg = extractConfigWithLogging(&gf, false);
    try std.testing.expectEqual(@as(u32, 8), cfg.n_kv_heads);
}

test "extractConfig reads rope attention factor for gpt-oss YaRN models" {
    const allocator = std.testing.allocator;

    var gf = gguf.GGUFFile{
        .version = .v3,
        .tensor_count = 0,
        .metadata = .{},
        .tensors = .{},
        .tensor_data_offset = 0,
        .allocator = allocator,
    };
    defer gf.deinit();

    try gf.metadata.put(allocator, try allocator.dupe(u8, "general.architecture"), .{ .string = try allocator.dupe(u8, "gpt-oss") });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gpt-oss.block_count"), .{ .uint32 = 24 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gpt-oss.attention.head_count"), .{ .uint32 = 64 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gpt-oss.attention.head_count_kv"), .{ .uint32 = 8 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gpt-oss.embedding_length"), .{ .uint32 = 2880 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gpt-oss.attention.key_length"), .{ .uint32 = 512 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gpt-oss.vocab_size"), .{ .uint32 = 201088 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gpt-oss.context_length"), .{ .uint32 = 131072 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gpt-oss.rope.scaling.factor"), .{ .float32 = 32.0 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gpt-oss.rope.scaling.attn_factor"), .{ .float32 = 1.75 });
    try gf.metadata.put(allocator, try allocator.dupe(u8, "gpt-oss.rope.scaling.original_context_length"), .{ .uint32 = 4096 });

    const cfg = extractConfigWithLogging(&gf, false);
    try std.testing.expectEqual(@as(f32, 32.0), cfg.rope_scaling_factor);
    try std.testing.expectEqual(@as(f32, 1.75), cfg.rope_attn_factor);
    try std.testing.expectEqual(@as(u32, 4096), cfg.rope_original_context);
}
