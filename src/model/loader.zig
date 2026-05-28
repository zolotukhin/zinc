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
    /// Sum of MoE expert tensor bytes (the share that would be allocated in
    /// host-visible memory under `ZINC_OFFLOAD_MOE_EXPERTS=1`). Zero for dense
    /// models. Use this for exact, model-specific offload-fit accounting.
    offloadable_tensor_bytes: u64 = 0,
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

/// Return true if this GGUF tensor name designates a fused MoE expert weight tensor.
/// Matches the four suffixes emitted by GGUF for sparse-MoE architectures:
/// `ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight`, and
/// `ffn_down_exps_scale.weight` (Q4_K_M variants only).
/// Dense tensors and non-expert MoE tensors (router gate, attention, embeddings, etc.)
/// are not matched.
pub fn isMoEExpertTensor(name: []const u8) bool {
    return std.mem.endsWith(u8, name, "ffn_gate_exps.weight") or
        std.mem.endsWith(u8, name, "ffn_up_exps.weight") or
        std.mem.endsWith(u8, name, "ffn_down_exps.weight") or
        std.mem.endsWith(u8, name, "ffn_down_exps_scale.weight");
}

/// Decided offload state for the most recently loaded model:
///   -1 = no decision yet (no model has been loaded)
///    0 = decided off (model fits in VRAM, or env forces off)
///    1 = decided on (model needs offload to fit, or env forces on)
/// Set by `decideOffloadForLoad` once per `loader.load` call. Read by
/// `shouldOffloadToHost` from per-tensor allocation paths and from
/// `tensorBytes` budget accounting in forward.zig and model_manager.zig.
var offload_state: std.atomic.Value(i8) = .init(-1);

const OffloadOverride = enum { auto, force_on, force_off };

/// Read `ZINC_OFFLOAD_MOE_EXPERTS` if set. Recognized values:
///   "1"/"true"  → force on
///   "0"/"false" → force off
///   anything else (or unset) → auto-decide based on fit
fn readOffloadOverride() OffloadOverride {
    const val = getenv("ZINC_OFFLOAD_MOE_EXPERTS") orelse return .auto;
    if (std.mem.eql(u8, val, "1") or std.mem.eql(u8, val, "true")) return .force_on;
    if (std.mem.eql(u8, val, "0") or std.mem.eql(u8, val, "false")) return .force_off;
    return .auto;
}

fn getenv(key: [*:0]const u8) ?[]const u8 {
    const ptr = std.c.getenv(key) orelse return null;
    return std.mem.span(ptr);
}

/// Pure decision function — returns the offload decision without mutating
/// state, for tests and for use by the side-effecting wrapper below.
pub fn computeOffloadDecision(
    override: OffloadOverride,
    total_tensor_bytes: u64,
    offloadable_tensor_bytes: u64,
    vram_budget_bytes: u64,
) bool {
    return switch (override) {
        .force_on => true,
        .force_off => false,
        .auto => blk: {
            // Reserve ~20% of the VRAM budget for KV cache and runtime scratch.
            // Conservative heuristic — the alternative is OOM at runtime.
            const usable = vram_budget_bytes - vram_budget_bytes / 5;
            if (total_tensor_bytes <= usable) break :blk false;
            if (offloadable_tensor_bytes == 0) break :blk false;
            if (total_tensor_bytes - offloadable_tensor_bytes <= usable) break :blk true;
            break :blk false;
        },
    };
}

/// Decide whether to offload MoE expert tensors for the next model load and
/// cache the decision. Honors `ZINC_OFFLOAD_MOE_EXPERTS` if set, otherwise
/// auto-decides:
///   - If the full model fits in VRAM (with headroom for KV/runtime): no offload.
///   - If the model only fits with MoE experts in host RAM: enable offload.
///   - If neither fits: don't enable (let allocation fail with a clear OOM
///     instead of pretending to fit).
pub fn decideOffloadForLoad(total_tensor_bytes: u64, offloadable_tensor_bytes: u64, vram_budget_bytes: u64) bool {
    const override = readOffloadOverride();
    const decision = computeOffloadDecision(override, total_tensor_bytes, offloadable_tensor_bytes, vram_budget_bytes);
    offload_state.store(if (decision) 1 else 0, .release);
    if (decision) {
        const reason = if (override == .force_on) "ZINC_OFFLOAD_MOE_EXPERTS=1" else "auto: weights exceed VRAM budget";
        log.info("MoE expert offload: ON ({s}) | weights {d} MB, offloadable {d} MB, VRAM budget {d} MB", .{
            reason,
            total_tensor_bytes / (1024 * 1024),
            offloadable_tensor_bytes / (1024 * 1024),
            vram_budget_bytes / (1024 * 1024),
        });
    }
    return decision;
}

/// Return whether MoE expert tensors should be in host-visible memory for the
/// currently-loaded model. Reads the cached decision from `decideOffloadForLoad`.
/// Returns false until a load has happened.
pub fn offloadEnabled() bool {
    return offload_state.load(.acquire) == 1;
}

/// Return true if this tensor should be routed to host-visible memory.
/// Combines the MoE-expert classifier with the cached offload decision.
pub fn shouldOffloadToHost(name: []const u8) bool {
    return offloadEnabled() and isMoEExpertTensor(name);
}

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
    mmap_file: ?std.Io.File,
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
            // Use raw syscall since deinit doesn't have io.
            _ = std.os.linux.close(f.handle);
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
/// @param io IO interface used for file operations.
/// @param path Path to the GGUF file on disk.
/// @param allocator Allocator used for the parsed metadata structures.
/// @returns A ModelConfig derived from GGUF metadata without uploading tensors to the GPU.
pub fn inspectConfig(io: std.Io, path: []const u8, allocator: std.mem.Allocator) !ModelConfig {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer {
        var close_file = file;
        close_file.close(io);
    }

    const stat = try file.stat(io);
    const mmap_data = try std.posix.mmap(
        null,
        stat.size,
        std.posix.PROT{ .READ = true },
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
pub fn inspectModel(io: std.Io, path: []const u8, allocator: std.mem.Allocator) !ModelInspection {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer {
        var close_file = file;
        close_file.close(io);
    }

    const stat = try file.stat(io);
    const mmap_data = try std.posix.mmap(
        null,
        stat.size,
        std.posix.PROT{ .READ = true },
        .{ .TYPE = .PRIVATE },
        file.handle,
        0,
    );
    defer std.posix.munmap(mmap_data);

    var gf = try gguf.parseWithOptions(mmap_data, allocator, .{ .log_summary = false });
    defer gf.deinit();

    var tensor_bytes: u64 = 0;
    var offloadable_tensor_bytes: u64 = 0;
    for (gf.tensors.items) |tensor_info| {
        const sz = tensor_info.sizeBytes();
        tensor_bytes += sz;
        if (isMoEExpertTensor(tensor_info.name)) offloadable_tensor_bytes += sz;
    }

    return .{
        .config = extractConfigWithLogging(&gf, false),
        .file_size = stat.size,
        .tensor_bytes = tensor_bytes,
        .offloadable_tensor_bytes = offloadable_tensor_bytes,
        .tensor_count = gf.tensor_count,
        .metadata_count = gf.metadata.count(),
    };
}

/// Load a GGUF model: memory-map the file, parse headers, and DMA tensors to GPU VRAM.
/// @param io IO interface used for file operations.
/// @param path Path to the GGUF file on disk.
/// @param instance Active Vulkan instance used for buffer allocation.
/// @param cmd_pool Command pool used for staging copy operations.
/// @param allocator Allocator used for metadata, tensor lists, and temporary state.
/// @returns A fully populated Model with parsed metadata and uploaded tensors.
pub fn load(
    io: std.Io,
    path: []const u8,
    instance: *const Instance,
    cmd_pool: *const CommandPool,
    allocator: std.mem.Allocator,
) !Model {
    log.info("Loading model: {s}", .{path});

    // Open and memory-map the file
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    errdefer file.close(io);

    const stat = try file.stat(io);
    const file_size = stat.size;
    log.info("File size: {d} MB", .{file_size / (1024 * 1024)});

    const mmap_data = try std.posix.mmap(
        null,
        file_size,
        std.posix.PROT{ .READ = true },
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
    var loaded_tensors: std.ArrayList(LoadedTensor) = .empty;
    errdefer {
        for (loaded_tensors.items) |*t| {
            var buf = t.gpu_buffer;
            buf.deinit();
        }
        loaded_tensors.deinit(allocator);
    }

    // Pass 1: sum tensor totals so the offload decision sees the model size
    // before any allocation happens. This is cheap (just integer adds) and
    // lets us avoid offloading when the model fits in VRAM.
    var total_tensor_bytes: u64 = 0;
    var offloadable_tensor_bytes: u64 = 0;
    for (gf.tensors.items) |tensor_info| {
        const sz = tensor_info.sizeBytes();
        total_tensor_bytes += sz;
        if (isMoEExpertTensor(tensor_info.name)) offloadable_tensor_bytes += sz;
    }
    _ = decideOffloadForLoad(total_tensor_bytes, offloadable_tensor_bytes, instance.vramBytes());

    var total_vram: u64 = 0;
    var total_host_visible: u64 = 0;
    for (gf.tensors.items) |tensor_info| {
        const tensor_size = tensor_info.sizeBytes();
        const data_offset = gf.tensor_data_offset + tensor_info.offset;
        const src_data = mmap_data[data_offset..][0..@intCast(tensor_size)];
        const offload = shouldOffloadToHost(tensor_info.name);

        var gpu_buf = blk: {
            if (offload) {
                break :blk try Buffer.initHostVisibleStorage(instance, tensor_size);
            } else {
                break :blk try Buffer.initDeviceLocal(
                    instance,
                    tensor_size,
                    vk.c.VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                );
            }
        };
        errdefer gpu_buf.deinit();

        if (offload) {
            // Host-visible: GPU reads directly over PCIe; memcpy from mmap, no staging.
            gpu_buf.upload(src_data);
            total_host_visible += tensor_size;
        } else {
            // Device-local: stage in host memory, then GPU-side copy into VRAM.
            var staging = try Buffer.initStaging(instance, tensor_size);
            defer staging.deinit();
            staging.upload(src_data);
            try buffer_mod.copyBuffer(instance, cmd_pool.handle, &staging, &gpu_buf, tensor_size);
            total_vram += tensor_size;
        }

        try loaded_tensors.append(allocator, .{
            .info = tensor_info,
            .gpu_buffer = gpu_buf,
        });
    }

    if (total_host_visible > 0) {
        log.info("Loaded {d} tensors | {d} MB device-local VRAM | {d} MB host-visible (system RAM)", .{
            loaded_tensors.items.len,
            total_vram / (1024 * 1024),
            total_host_visible / (1024 * 1024),
        });
    } else {
        log.info("Loaded {d} tensors | {d} MB VRAM", .{
            loaded_tensors.items.len,
            total_vram / (1024 * 1024),
        });
    }

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
    try std.testing.expectEqual(Architecture.qwen35, parseArchitecture("qwen3_5"));
    try std.testing.expectEqual(Architecture.qwen35, parseArchitecture("qwen3_6"));
    try std.testing.expectEqual(Architecture.mamba, parseArchitecture("mamba"));
    try std.testing.expectEqual(Architecture.unknown, parseArchitecture("gpt2"));
}

test "isMoEExpertTensor matches MoE expert tensor suffixes" {
    // Fused per-layer MoE expert tensors.
    try std.testing.expect(isMoEExpertTensor("blk.0.ffn_gate_exps.weight"));
    try std.testing.expect(isMoEExpertTensor("blk.47.ffn_up_exps.weight"));
    try std.testing.expect(isMoEExpertTensor("blk.7.ffn_down_exps.weight"));
    // Q4_K_M variants emit a separate per-row scale tensor.
    try std.testing.expect(isMoEExpertTensor("blk.3.ffn_down_exps_scale.weight"));

    // Non-expert tensors — must stay device-local.
    try std.testing.expect(!isMoEExpertTensor("blk.0.attn_q.weight"));
    try std.testing.expect(!isMoEExpertTensor("blk.0.attn_k.weight"));
    try std.testing.expect(!isMoEExpertTensor("blk.0.attn_v.weight"));
    try std.testing.expect(!isMoEExpertTensor("blk.0.attn_output.weight"));
    try std.testing.expect(!isMoEExpertTensor("blk.0.ffn_gate.weight"));
    try std.testing.expect(!isMoEExpertTensor("blk.0.ffn_up.weight"));
    try std.testing.expect(!isMoEExpertTensor("blk.0.ffn_down.weight"));
    try std.testing.expect(!isMoEExpertTensor("blk.0.ffn_gate_inp.weight"));
    try std.testing.expect(!isMoEExpertTensor("output.weight"));
    try std.testing.expect(!isMoEExpertTensor("token_embd.weight"));
}

test "shouldOffloadToHost is false until a load decides" {
    // The cache starts at -1 (no decision yet). offloadEnabled returns false in
    // that state, so even MoE expert tensors stay device-local. This is the
    // safe default before any model has been loaded.
    const prev = offload_state.load(.acquire);
    defer offload_state.store(prev, .release);
    offload_state.store(-1, .release);
    try std.testing.expect(!shouldOffloadToHost("blk.0.ffn_gate_exps.weight"));
}

test "computeOffloadDecision: auto fits without offload when weights are small" {
    // 10 GB weights, 30 GB budget → fits comfortably.
    try std.testing.expect(!computeOffloadDecision(.auto, 10 * 1024 * 1024 * 1024, 5 * 1024 * 1024 * 1024, 30 * 1024 * 1024 * 1024));
}

test "computeOffloadDecision: auto enables offload when weights exceed budget but offload-shape fits" {
    // 22 GB weights, 18 GB offloadable, 16 GB budget.
    // Without offload: 22 > 12.8 (16 * 0.8) → doesn't fit.
    // With offload: 22 - 18 = 4 GB ≤ 12.8 → fits.
    try std.testing.expect(computeOffloadDecision(.auto, 22 * 1024 * 1024 * 1024, 18 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024));
}

test "computeOffloadDecision: auto declines for dense model that doesn't fit" {
    // 30 GB dense weights (no offloadable share), 16 GB budget. Offload can't help.
    try std.testing.expect(!computeOffloadDecision(.auto, 30 * 1024 * 1024 * 1024, 0, 16 * 1024 * 1024 * 1024));
}

test "computeOffloadDecision: auto declines when neither full nor offloaded fits" {
    // 100 GB weights, 80 GB offloadable, 16 GB budget. Even offloaded (20 GB) > 12.8 GB usable.
    try std.testing.expect(!computeOffloadDecision(.auto, 100 * 1024 * 1024 * 1024, 80 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024));
}

test "computeOffloadDecision: force_on always offloads (even when it would fit normally)" {
    try std.testing.expect(computeOffloadDecision(.force_on, 1 * 1024 * 1024 * 1024, 1 * 1024 * 1024 * 1024, 100 * 1024 * 1024 * 1024));
}

test "computeOffloadDecision: force_off always disables (even when it wouldn't fit otherwise)" {
    try std.testing.expect(!computeOffloadDecision(.force_off, 100 * 1024 * 1024 * 1024, 80 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024));
}

test "decideOffloadForLoad caches the decision into offload_state" {
    const prev = offload_state.load(.acquire);
    defer offload_state.store(prev, .release);

    // Force_off via the env-aware wrapper is hard to set without setenv; skip.
    // Instead verify the auto path: 10 GB weights, 30 GB budget → fits.
    if (readOffloadOverride() != .auto) return;

    offload_state.store(-1, .release);
    _ = decideOffloadForLoad(10 * 1024 * 1024 * 1024, 5 * 1024 * 1024 * 1024, 30 * 1024 * 1024 * 1024);
    try std.testing.expectEqual(@as(i8, 0), offload_state.load(.acquire));
    try std.testing.expect(!offloadEnabled());

    // 22 GB weights with 18 GB offloadable on a 16 GB budget → offload kicks in.
    _ = decideOffloadForLoad(22 * 1024 * 1024 * 1024, 18 * 1024 * 1024 * 1024, 16 * 1024 * 1024 * 1024);
    try std.testing.expectEqual(@as(i8, 1), offload_state.load(.acquire));
    try std.testing.expect(offloadEnabled());
}

test "shouldOffloadToHost returns true for MoE tensors when gate forced on" {
    // Override the cache directly (same module = field access). The state is
    // restored at end-of-test so the gate-off test above keeps working under
    // any test order.
    const prev = offload_state.load(.acquire);
    defer offload_state.store(prev, .release);
    offload_state.store(1, .release);

    try std.testing.expect(shouldOffloadToHost("blk.0.ffn_gate_exps.weight"));
    try std.testing.expect(shouldOffloadToHost("blk.5.ffn_down_exps.weight"));
    // Dense tensors stay device-local even with gate on — the classifier
    // is independent of the gate.
    try std.testing.expect(!shouldOffloadToHost("blk.0.attn_q.weight"));
    try std.testing.expect(!shouldOffloadToHost("output.weight"));
}

test "shouldOffloadToHost returns false for MoE tensors when gate forced off" {
    const prev = offload_state.load(.acquire);
    defer offload_state.store(prev, .release);
    offload_state.store(0, .release);

    try std.testing.expect(!shouldOffloadToHost("blk.0.ffn_gate_exps.weight"));
    try std.testing.expect(!shouldOffloadToHost("blk.0.attn_q.weight"));
}

test "offloadEnabled reflects cached state and never mutates it" {
    const prev = offload_state.load(.acquire);
    defer offload_state.store(prev, .release);

    // Forced-on cache → returns true, cache unchanged.
    offload_state.store(1, .release);
    try std.testing.expect(offloadEnabled());
    try std.testing.expectEqual(@as(i8, 1), offload_state.load(.acquire));

    // Forced-off cache → returns false, cache unchanged.
    offload_state.store(0, .release);
    try std.testing.expect(!offloadEnabled());
    try std.testing.expectEqual(@as(i8, 0), offload_state.load(.acquire));

    // Uncached state (-1) is the safe default before any load: returns false
    // and stays -1 until decideOffloadForLoad sets it.
    offload_state.store(-1, .release);
    try std.testing.expect(!offloadEnabled());
    try std.testing.expectEqual(@as(i8, -1), offload_state.load(.acquire));
}

test "extractConfig defaults gemma4 attention scale to 1.0" {
    const allocator = std.testing.allocator;

    var gf = gguf.GGUFFile{
        .version = .v3,
        .tensor_count = 0,
        .metadata = .{},
        .tensors = .empty,
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
        .tensors = .empty,
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
        .tensors = .empty,
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
