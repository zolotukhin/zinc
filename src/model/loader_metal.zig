//! Metal-specific model loading — zero-copy via mmap + newBufferWithBytesNoCopy.
//! This replaces the Vulkan loader's staging-buffer DMA with direct mmap wrapping.
//! @section Model Format & Loading
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
    buffer_offset: u32 = 0,
};

/// Runtime model state backed by a memory-mapped GGUF file and zero-copy Metal buffers.
pub const Model = struct {
    config: ModelConfig,
    gguf_file: gguf.GGUFFile,
    tensors: std.ArrayList(LoadedTensor),
    tensor_arenas: std.ArrayList(MetalBuffer),
    mmap_data: ?[]align(std.heap.page_size_min) const u8,
    mmap_file: ?std.fs.File,
    allocator: std.mem.Allocator,
    /// Residency set wiring all model weight buffers down on macOS 15+.
    /// Null on older systems or if creation failed (the loader logs and degrades).
    weight_rset: ?*shim.MetalRSet = null,

    /// Release Metal buffers, GGUF metadata, and the backing file mapping.
    pub fn deinit(self: *Model) void {
        // The residency set holds (retained) MTLBuffer references. End its
        // residency before freeing the underlying buffers so Metal stops
        // tracking them first.
        if (self.weight_rset) |rs| {
            shim.mtl_rset_free(rs);
            self.weight_rset = null;
        }
        for (self.tensors.items) |*t| {
            metal_buffer.freeBuffer(&t.gpu_buffer);
        }
        self.tensors.deinit(self.allocator);
        for (self.tensor_arenas.items) |*arena| {
            metal_buffer.freeBuffer(arena);
        }
        self.tensor_arenas.deinit(self.allocator);

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
        .rope_original_context = gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.rope.scaling.original_context_length", .{prefix}) catch "") orelse 0,
    };
}

fn extractConfig(gf: *const gguf.GGUFFile) ModelConfig {
    return extractConfigWithLogging(gf, true);
}

fn shouldCopyOutOfMmap(config: ModelConfig, tensor_info: gguf.TensorInfo, data_offset: u64) bool {
    if (config.architecture == .gemma and config.n_experts == 0 and config.hidden_dim >= 5000) {
        switch (tensor_info.type_) {
            .q4_k, .q6_k => return true,
            else => {},
        }
    }

    if (config.architecture != .gemma) return false;
    if (!(config.n_experts > 0 and config.rope_freq_base_swa > 0)) return false;
    if (data_offset <= std.math.maxInt(u32)) return false;
    if (tensor_info.type_ != .q8_0) return false;

    const name = tensor_info.name;
    return std.mem.endsWith(u8, name, "attn_q.weight") or
        std.mem.endsWith(u8, name, "attn_k.weight") or
        std.mem.endsWith(u8, name, "attn_v.weight") or
        std.mem.endsWith(u8, name, "attn_output.weight");
}

const copied_tensor_arena_alignment: usize = 4096;
const copied_tensor_arena_limit: usize = 256 * 1024 * 1024;

fn alignForwardPow2(value: usize, alignment: usize) usize {
    return (value + alignment - 1) & ~(alignment - 1);
}

fn shouldPackCopiedTensorArenas(config: ModelConfig) bool {
    return config.architecture == .gemma and config.n_experts == 0 and config.hidden_dim >= 5000;
}

fn planCopiedTensorArenas(
    gf: *const gguf.GGUFFile,
    config: ModelConfig,
    allocator: std.mem.Allocator,
) !std.ArrayList(usize) {
    var arena_sizes: std.ArrayList(usize) = .{};
    errdefer arena_sizes.deinit(allocator);

    if (!shouldPackCopiedTensorArenas(config)) return arena_sizes;

    var current_size: usize = 0;
    for (gf.tensors.items) |tensor_info| {
        const tensor_size: usize = @intCast(tensor_info.sizeBytes());
        const data_offset = gf.tensor_data_offset + tensor_info.offset;
        if (!shouldCopyOutOfMmap(config, tensor_info, data_offset)) continue;

        const aligned_size = alignForwardPow2(tensor_size, copied_tensor_arena_alignment);
        if (aligned_size > copied_tensor_arena_limit) {
            if (current_size > 0) {
                try arena_sizes.append(allocator, current_size);
                current_size = 0;
            }
            try arena_sizes.append(allocator, aligned_size);
            continue;
        }

        if (current_size == 0) {
            current_size = aligned_size;
        } else if (current_size + aligned_size > copied_tensor_arena_limit) {
            try arena_sizes.append(allocator, current_size);
            current_size = aligned_size;
        } else {
            current_size += aligned_size;
        }
    }

    if (current_size > 0) try arena_sizes.append(allocator, current_size);
    return arena_sizes;
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

    var tensor_arenas: std.ArrayList(MetalBuffer) = .{};
    errdefer {
        for (tensor_arenas.items) |*arena| {
            metal_buffer.freeBuffer(arena);
        }
        tensor_arenas.deinit(allocator);
    }

    var arena_sizes = try planCopiedTensorArenas(&gf, config, allocator);
    defer arena_sizes.deinit(allocator);
    for (arena_sizes.items) |arena_size| {
        try tensor_arenas.append(allocator, try metal_buffer.createBuffer(metal_ctx, arena_size));
    }

    var total_size: u64 = 0;
    var copied_tensor_count: usize = 0;
    var copied_tensor_bytes: u64 = 0;
    var arena_index: usize = 0;
    var arena_offset: usize = 0;
    for (gf.tensors.items) |tensor_info| {
        const tensor_size = tensor_info.sizeBytes();
        const data_offset = gf.tensor_data_offset + tensor_info.offset;

        const copy_out = shouldCopyOutOfMmap(config, tensor_info, data_offset);
        const gpu_buf, const buffer_offset = if (copy_out and tensor_arenas.items.len > 0) blk: {
            const tensor_bytes: usize = @intCast(tensor_size);
            const aligned_size = alignForwardPow2(tensor_bytes, copied_tensor_arena_alignment);
            while (arena_index < tensor_arenas.items.len and arena_offset + aligned_size > tensor_arenas.items[arena_index].size) {
                arena_index += 1;
                arena_offset = 0;
            }
            if (arena_index >= tensor_arenas.items.len) return error.MetalBufferAllocFailed;

            const arena = &tensor_arenas.items[arena_index];
            const src_off: usize = @intCast(data_offset);
            @memcpy(arena.cpu_ptr.?[arena_offset .. arena_offset + tensor_bytes], mmap_data[src_off .. src_off + tensor_bytes]);
            copied_tensor_count += 1;
            copied_tensor_bytes += tensor_size;
            const buf = metal_buffer.aliasBuffer(arena, arena_offset, tensor_bytes);
            const offset: u32 = @intCast(arena_offset);
            arena_offset += aligned_size;
            break :blk .{ buf, offset };
        } else if (copy_out) blk: {
            var buf = try metal_buffer.createBuffer(metal_ctx, @intCast(tensor_size));
            const src_off: usize = @intCast(data_offset);
            const tensor_bytes: usize = @intCast(tensor_size);
            @memcpy(buf.cpu_ptr.?[0..tensor_bytes], mmap_data[src_off .. src_off + tensor_bytes]);
            copied_tensor_count += 1;
            copied_tensor_bytes += tensor_size;
            break :blk .{ buf, @as(u32, 0) };
        } else blk: {
            // Page-align the offset for Metal buffer wrapping
            const page_size: u64 = std.heap.page_size_min;
            const aligned_offset = (data_offset / page_size) * page_size;
            const offset_within_page = data_offset - aligned_offset;
            const aligned_size = ((tensor_size + offset_within_page + page_size - 1) / page_size) * page_size;

            // Wrap the mmap'd region as a Metal buffer
            const ptr: [*]u8 = @ptrCast(@constCast(mmap_data.ptr + aligned_offset));
            const wrapped = metal_buffer.wrapMmap(metal_ctx, ptr, @intCast(aligned_size)) catch |err| {
                log.err("Failed to wrap tensor at offset {d} ({d} bytes): {s}", .{
                    data_offset, tensor_size, @errorName(err),
                });
                return err;
            };
            break :blk .{ wrapped, @as(u32, @intCast(offset_within_page)) };
        };

        try loaded_tensors.append(allocator, .{
            .info = tensor_info,
            .gpu_buffer = gpu_buf,
            .buffer_offset = buffer_offset,
        });

        total_size += tensor_size;
    }

    log.info("Loaded {d} tensors | {d} MB (zero-copy mmap)", .{
        loaded_tensors.items.len,
        total_size / (1024 * 1024),
    });
    if (copied_tensor_count > 0) {
        log.info("Metal loader copied {d} tensors ({d} MB) out of mmap for stable access", .{
            copied_tensor_count,
            copied_tensor_bytes / (1024 * 1024),
        });
        if (tensor_arenas.items.len > 0) {
            log.info("Metal loader packed copied tensors into {d} shared arenas", .{tensor_arenas.items.len});
        }
    }

    // Wire all weight buffers into a single MTLResidencySet so the OS can't
    // page them out between layer dispatches. This mirrors llama.cpp's
    // `ggml_metal_buffer_rset_init` and is the missing piece behind the
    // 130x bench-vs-real bandwidth gap on dense Gemma 31B: kernel microbench
    // sees 491 GB/s, real inference sees ~4 GB/s. Without residency hints,
    // `commandBufferWithUnretainedReferences` cannot guarantee that the 17 GB
    // working set stays wired across the 60 layers per token.
    var weight_rset: ?*shim.MetalRSet = null;
    if (shim.mtl_rset_supported() != 0) {
        const initial_capacity: u32 = @intCast(tensor_arenas.items.len + loaded_tensors.items.len);
        if (shim.mtl_rset_create(metal_ctx, initial_capacity)) |rs| {
            for (tensor_arenas.items) |*arena| {
                if (arena.handle) |h| shim.mtl_rset_add_buffer(rs, h);
            }
            // Tensors that aren't aliased into an arena (mmap-wrapped or
            // standalone copies) own their own MTLBuffer and need to be
            // added directly.
            for (loaded_tensors.items) |*t| {
                if (!t.gpu_buffer.owns_handle) continue;
                if (t.gpu_buffer.handle) |h| shim.mtl_rset_add_buffer(rs, h);
            }
            shim.mtl_rset_commit_and_request(rs);
            weight_rset = rs;
            log.info("Metal loader wired weight buffers into MTLResidencySet (commandBufferWithUnretainedReferences-safe)", .{});
        } else {
            log.warn("Metal loader: MTLResidencySet creation failed; weights remain pageable", .{});
        }
    }

    return Model{
        .config = config,
        .gguf_file = gf,
        .tensors = loaded_tensors,
        .tensor_arenas = tensor_arenas,
        .mmap_data = mmap_data,
        .mmap_file = file,
        .allocator = allocator,
        .weight_rset = weight_rset,
    };
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

test "shouldCopyOutOfMmap only selects gemma ISWA q8 attention tensors above u32 offset" {
    const cfg = ModelConfig{
        .architecture = .gemma,
        .n_layers = 30,
        .n_heads = 16,
        .n_kv_heads = 8,
        .head_dim = 512,
        .hidden_dim = 2816,
        .intermediate_dim = 0,
        .vocab_size = 262144,
        .context_length = 8192,
        .rope_freq_base = 1_000_000.0,
        .rope_freq_base_swa = 10_000.0,
        .n_experts = 128,
        .n_experts_used = 8,
        .rope_dim = 512,
        .ssm_d_conv = 0,
        .ssm_d_inner = 0,
        .ssm_d_state = 0,
        .ssm_dt_rank = 0,
        .ssm_n_group = 0,
        .full_attn_interval = 0,
        .shared_expert_intermediate_dim = 0,
    };

    const attn_q = gguf.TensorInfo{
        .name = "blk.0.attn_q.weight",
        .n_dims = 2,
        .dims = .{ 2816, 4096, 1, 1 },
        .type_ = .q8_0,
        .offset = 0,
    };
    const ffn_down = gguf.TensorInfo{
        .name = "blk.0.ffn_down.weight",
        .n_dims = 2,
        .dims = .{ 2112, 2816, 1, 1 },
        .type_ = .q8_0,
        .offset = 0,
    };

    try std.testing.expect(shouldCopyOutOfMmap(cfg, attn_q, @as(u64, std.math.maxInt(u32)) + 1));
    try std.testing.expect(!shouldCopyOutOfMmap(cfg, attn_q, 4096));
    try std.testing.expect(!shouldCopyOutOfMmap(cfg, ffn_down, @as(u64, std.math.maxInt(u32)) + 1));

    var non_gemma = cfg;
    non_gemma.architecture = .qwen35;
    non_gemma.rope_freq_base_swa = 0;
    non_gemma.n_experts = 0;
    try std.testing.expect(!shouldCopyOutOfMmap(non_gemma, attn_q, @as(u64, std.math.maxInt(u32)) + 1));
}

test "shouldCopyOutOfMmap selects dense gemma 31b q4/q6 hot weights" {
    const cfg = ModelConfig{
        .architecture = .gemma,
        .n_layers = 60,
        .n_heads = 32,
        .n_kv_heads = 16,
        .head_dim = 256,
        .hidden_dim = 5376,
        .intermediate_dim = 21504,
        .vocab_size = 262144,
        .context_length = 262144,
        .rope_freq_base = 1_000_000.0,
        .rope_freq_base_swa = 10_000.0,
        .n_experts = 0,
        .n_experts_used = 0,
        .rope_dim = 64,
        .ssm_d_conv = 0,
        .ssm_d_inner = 0,
        .ssm_d_state = 0,
        .ssm_dt_rank = 0,
        .ssm_n_group = 0,
        .full_attn_interval = 1,
        .shared_expert_intermediate_dim = 0,
    };

    const q4_weight = gguf.TensorInfo{
        .name = "blk.0.ffn_gate.weight",
        .n_dims = 2,
        .dims = .{ 5376, 21504, 1, 1 },
        .type_ = .q4_k,
        .offset = 0,
    };
    var q6_weight = q4_weight;
    q6_weight.type_ = .q6_k;
    var f32_norm = q4_weight;
    f32_norm.name = "blk.0.attn_norm.weight";
    f32_norm.type_ = .f32;

    try std.testing.expect(shouldCopyOutOfMmap(cfg, q4_weight, 4096));
    try std.testing.expect(shouldCopyOutOfMmap(cfg, q6_weight, 4096));
    try std.testing.expect(!shouldCopyOutOfMmap(cfg, f32_norm, 4096));

    var moe_gemma = cfg;
    moe_gemma.n_experts = 128;
    moe_gemma.n_experts_used = 8;
    try std.testing.expect(!shouldCopyOutOfMmap(moe_gemma, q4_weight, 4096));
}
