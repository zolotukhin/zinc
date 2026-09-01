//! CUDA-specific model loading — mmap the GGUF, upload every tensor to the
//! NVIDIA device, and expose them to the CUDA forward pass.
//!
//! Unlike the Metal loader (Apple unified memory, zero-copy
//! newBufferWithBytesNoCopy), CUDA device memory is NOT CPU-visible, so each
//! tensor is copied host->device once at load time via `cuda_buffer.uploadMmap`.
//! The file mapping is kept alive after upload because the embedding lookup
//! (`dequantEmbeddingRow`) dequantizes a single `token_embd.weight` row on the
//! CPU directly out of the mapping.
//!
//! GGUF parsing and config extraction are backend-agnostic; the config
//! extractor below is copied verbatim from loader_metal.zig (pure metadata
//! reads, no backend dependency). `dequantRow`/`getScaleMinK4` are copied from
//! compute/forward_metal.zig to avoid importing the Metal forward pass (which
//! would drag in Metal-only dependencies).
//! @section Model Format & Loading
const std = @import("std");
const gguf = @import("gguf.zig");
const config_mod = @import("config.zig");
const ModelConfig = config_mod.ModelConfig;
const GGMLType = gguf.GGMLType;
const shim = @import("../cuda/c.zig").shim;
const cuda_buffer = @import("../cuda/buffer.zig");

const log = std.log.scoped(.loader_cuda);

/// A GGUF tensor descriptor paired with the device buffer holding its weights.
/// `info` carries the on-disk shape/quant/offset metadata; `gpu_buffer` is the
/// device-local copy uploaded at load time.
pub const LoadedTensor = struct {
    info: gguf.TensorInfo,
    gpu_buffer: cuda_buffer.CudaBuffer,
};

/// Runtime model state: parsed config, the live CUDA context, every weight
/// tensor resident on the GPU, and the still-mapped GGUF file (kept alive for
/// CPU-side embedding dequantization).
pub const Model = struct {
    config: ModelConfig,
    /// CUDA context that owns all device buffers (borrowed; not destroyed here).
    ctx: ?*shim.CudaCtx,
    allocator: std.mem.Allocator,

    // --- internal state ---
    /// Parsed GGUF header/metadata/tensor table (owns tensor-name strings).
    gguf_file: gguf.GGUFFile,
    /// Backing storage for every uploaded tensor; pointers into this array are
    /// stable for the model's lifetime (we never append after `load`).
    tensors: std.ArrayList(LoadedTensor),
    /// Name -> *LoadedTensor index for O(1) lookup by exact tensor name.
    tensor_map: std.StringHashMap(*LoadedTensor),
    /// The memory-mapped GGUF file; kept resident so `dequantEmbeddingRow` can
    /// read the raw `token_embd.weight` bytes on the CPU.
    mmap_data: ?[]align(std.heap.page_size_min) const u8,
    /// Open file handle backing the mapping; closed in `deinit`.
    mmap_file: ?std.fs.File,

    /// Open + mmap the GGUF at `path`, parse it, extract the config, and upload
    /// every tensor to `ctx`. The mapping is retained for embedding lookups.
    /// @param allocator Owns the GGUF metadata, the tensor list, and the name map.
    /// @param ctx Live CUDA context that will own every uploaded weight buffer.
    /// @param path Filesystem path to the GGUF model.
    /// @returns A fully resident `Model`; call `deinit` to release device + host memory.
    pub fn load(allocator: std.mem.Allocator, ctx: ?*shim.CudaCtx, path: []const u8) !Model {
        log.info("Loading model (CUDA): {s}", .{path});

        // Open and memory-map the file (read-only, private). The mapping stays
        // alive for the model's lifetime — uploadMmap reads through it and the
        // embedding dequant reads the raw token_embd bytes later.
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

        // Parse GGUF headers (backend-agnostic).
        var gf = try gguf.parse(mmap_data, allocator);
        errdefer gf.deinit();

        const config = extractConfig(&gf);
        if (config.architecture == .unknown) {
            log.err("Unsupported model architecture in {s}", .{path});
            return error.UnsupportedArchitecture;
        }

        // Pre-size the tensor list so its backing buffer never reallocates while
        // we hand out *LoadedTensor pointers into it below.
        var tensors = std.ArrayList(LoadedTensor){};
        errdefer {
            for (tensors.items) |*t| cuda_buffer.freeBuffer(&t.gpu_buffer);
            tensors.deinit(allocator);
        }
        try tensors.ensureTotalCapacityPrecise(allocator, gf.tensors.items.len);

        var tensor_map = std.StringHashMap(*LoadedTensor).init(allocator);
        errdefer tensor_map.deinit();
        try tensor_map.ensureTotalCapacity(@intCast(gf.tensors.items.len));

        // Upload every tensor host->device. uploadMmap is a single H2D copy from
        // the mapped file region into a fresh device-local allocation.
        var uploaded_bytes: u64 = 0;
        for (gf.tensors.items) |info| {
            const src_off: usize = @intCast(gf.tensor_data_offset + info.offset);
            const sz: usize = @intCast(info.sizeBytes());
            const buf = try cuda_buffer.uploadMmap(ctx, mmap_data.ptr + src_off, sz);

            tensors.appendAssumeCapacity(.{ .info = info, .gpu_buffer = buf });
            const slot = &tensors.items[tensors.items.len - 1];
            // info.name is owned by the GGUF file (alive for the model lifetime),
            // so it is safe to use directly as the map key.
            tensor_map.putAssumeCapacity(slot.info.name, slot);
            uploaded_bytes += sz;
        }

        log.info("Uploaded {d} tensors ({d} MB) to GPU", .{
            tensors.items.len, uploaded_bytes / (1024 * 1024),
        });

        return Model{
            .config = config,
            .ctx = ctx,
            .allocator = allocator,
            .gguf_file = gf,
            .tensors = tensors,
            .tensor_map = tensor_map,
            .mmap_data = mmap_data,
            .mmap_file = file,
        };
    }

    /// Look up a tensor by its exact GGUF name (e.g. `"blk.3.attn_q.weight"`).
    /// @returns A pointer to the loaded tensor, or null if absent.
    pub fn get(self: *const Model, name: []const u8) ?*const LoadedTensor {
        return self.tensor_map.get(name);
    }

    /// Look up a per-layer tensor by formatting `"blk.{layer}.{suffix}"`.
    /// @param layer Zero-based transformer block index.
    /// @param suffix Tensor suffix within the block, e.g. `"attn_q.weight"`.
    /// @returns A pointer to the loaded tensor, or null if absent.
    pub fn getLayer(self: *const Model, layer: u32, suffix: []const u8) ?*const LoadedTensor {
        var buf: [128]u8 = undefined;
        const name = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ layer, suffix }) catch return null;
        return self.tensor_map.get(name);
    }

    /// Dequantize one row of `token_embd.weight` to f32 on the CPU.
    /// Reads the raw tensor bytes directly out of the live mapping (no device
    /// round-trip). `token_id` is clamped to the last valid row.
    /// @param token_id Token whose embedding row to fetch.
    /// @param out Destination slice of at least `hidden_dim` f32 values; filled in place.
    pub fn dequantEmbeddingRow(self: *const Model, token_id: u32, out: []f32) void {
        const t = self.get("token_embd.weight") orelse {
            log.warn("token_embd.weight missing; zeroing embedding row", .{});
            @memset(out, 0);
            return;
        };
        const data = self.mmap_data orelse {
            @memset(out, 0);
            return;
        };

        // GGUF stores token_embd as [hidden_dim, vocab_size] (innermost-first):
        // dims[0] = columns per row (hidden), dims[1] = number of rows (vocab).
        const cols: u32 = @intCast(t.info.dims[0]);
        const vocab: u32 = @intCast(t.info.dims[1]);
        const row: u32 = if (token_id >= vocab) (if (vocab > 0) vocab - 1 else 0) else token_id;

        const base: usize = @intCast(self.gguf_file.tensor_data_offset + t.info.offset);
        const sz: usize = @intCast(t.info.sizeBytes());
        const raw = data[base .. base + sz];

        const n = @min(out.len, @as(usize, cols));
        dequantRow(raw, row, cols, t.info.type_, out[0..n]);
    }

    /// Release device buffers, the GGUF metadata, the name map, and the mapping.
    pub fn deinit(self: *Model) void {
        for (self.tensors.items) |*t| {
            cuda_buffer.freeBuffer(&t.gpu_buffer);
        }
        self.tensors.deinit(self.allocator);
        self.tensor_map.deinit();

        if (self.mmap_data) |data| {
            std.posix.munmap(data);
            self.mmap_data = null;
        }
        if (self.mmap_file) |f| {
            var file = f;
            file.close();
            self.mmap_file = null;
        }

        self.gguf_file.deinit();
        self.* = undefined;
    }
};

/// Inspect a GGUF model's config without uploading any weights to the GPU.
/// mmaps the file, parses the header, extracts the `ModelConfig`, then unmaps —
/// used by `zinc --check` to report architecture/dims on the CUDA backend.
/// @param path Filesystem path to the GGUF model.
/// @param allocator Owns the transient GGUF metadata (freed before returning).
/// @returns The parsed `ModelConfig`.
pub fn inspectConfig(path: []const u8, allocator: std.mem.Allocator) !ModelConfig {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

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

    var gf = try gguf.parse(mmap_data, allocator);
    defer gf.deinit();

    return extractConfigWithLogging(&gf, false);
}

// ===========================================================================
// Config extraction — copied verbatim from loader_metal.zig:102-294.
// Pure GGUF metadata reads (no backend dependency): reads "{arch}.block_count",
// "{arch}.attention.head_count[_kv]", "{arch}.ssm.*",
// "{arch}.full_attention_interval", rope sections, etc.
// ===========================================================================

/// Extract model configuration from GGUF metadata (platform-independent).
fn extractConfigWithLogging(gf: *const gguf.GGUFFile, log_metadata: bool) ModelConfig {
    const arch_str = gf.getString("general.architecture") orelse "unknown";
    const arch = config_mod.parseArchitecture(arch_str);
    const prefix = arch_str;

    var key_buf: [128]u8 = undefined;

    const block_count = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.block_count", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };
    const declared_nextn_layers = blk: {
        const key = std.fmt.bufPrint(&key_buf, "{s}.nextn_predict_layers", .{prefix}) catch break :blk @as(u32, 0);
        break :blk gf.getU32(key) orelse 0;
    };
    const layer_counts = config_mod.normalizeLayerCounts(block_count, declared_nextn_layers);
    const n_layers = layer_counts.decoder;
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
    const full_attn_interval = blk: {
        if (std.posix.getenv("ZINC_FULL_ATTN_INTERVAL")) |v| {
            if (std.fmt.parseInt(u32, v, 10)) |parsed| break :blk parsed else |_| {}
        }
        break :blk gf.getU32(std.fmt.bufPrint(&key_buf, "{s}.full_attention_interval", .{prefix}) catch "") orelse
            if (ssm_d_inner > 0) @as(u32, 4) else @as(u32, 1);
    };
    log.info("full_attn_interval = {d} (override: ZINC_FULL_ATTN_INTERVAL)", .{full_attn_interval});
    log.info("rope_dim = {d}, head_dim = {d}, n_head = {d}, n_kv_head = {d}", .{ rope_dim, head_dim, n_heads, n_kv_heads });

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
        if (layer_counts.nextn > 0) {
            log.info("NextN/MTP: {d} appended block(s) retained but excluded from ordinary inference", .{layer_counts.nextn});
        } else if (declared_nextn_layers > 0) {
            log.warn("Ignoring invalid NextN/MTP layer count {d} for block_count {d}", .{ declared_nextn_layers, block_count });
        }
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
        .n_nextn_layers = layer_counts.nextn,
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
        .logit_scale = blk: {
            const lsk = std.fmt.bufPrint(&key_buf, "{s}.logit_scale", .{prefix}) catch break :blk @as(f32, 0.0);
            break :blk gf.getF32(lsk) orelse 0.0;
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

// ===========================================================================
// Row dequantization — copied from compute/forward_metal.zig (getScaleMinK4 +
// dequantRow). Depends only on std + gguf.GGMLType; used for the CPU embedding
// lookup so we don't pull in the Metal forward pass.
// ===========================================================================

fn getScaleMinK4(j: usize, scales: []const u8) struct { sc: u8, m: u8 } {
    if (j < 4) {
        return .{ .sc = scales[j] & 63, .m = scales[j + 4] & 63 };
    } else {
        return .{
            .sc = (scales[j + 4] & 0xF) | ((scales[j - 4] >> 6) << 4),
            .m = (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        };
    }
}

/// Dequantize one row of quantized weight data to f32 values.
/// Supports f32, f16, Q5_0, Q5_1, Q8_0, Q4_K, Q5_K, Q6_K, and MXFP4.
/// Unsupported types log a warning and zero the output slice.
/// @param raw_data Raw GGUF tensor bytes for the full matrix.
/// @param row Zero-based row index to dequantize.
/// @param cols Number of columns (elements) per row.
/// @param quant_type GGML quantization type describing the on-disk layout.
/// @param output Caller-allocated slice of at least `cols` f32 values; filled in place.
pub fn dequantRow(raw_data: []const u8, row: u32, cols: u32, quant_type: GGMLType, output: []f32) void {
    switch (quant_type) {
        .f32 => {
            const row_bytes = @as(usize, cols) * 4;
            const offset = @as(usize, row) * row_bytes;
            const src: [*]const f32 = @ptrCast(@alignCast(raw_data[offset..].ptr));
            @memcpy(output, src[0..cols]);
        },
        .f16 => {
            const offset = @as(usize, row) * @as(usize, cols) * 2;
            for (0..cols) |i| {
                const byte_off = offset + i * 2;
                const bits = std.mem.readInt(u16, raw_data[byte_off..][0..2], .little);
                output[i] = @floatCast(@as(f16, @bitCast(bits)));
            }
        },
        .q5_0 => {
            const block_size: usize = 32;
            const bpb: usize = 22;
            const bpr = @as(usize, cols) / block_size;
            const row_off = @as(usize, row) * bpr * bpb;
            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bo = row_off + b * bpb;
                const scale_bits = std.mem.readInt(u16, raw_data[bo..][0..2], .little);
                const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
                const qh = std.mem.readInt(u32, raw_data[bo + 2 ..][0..4], .little);
                const qs = raw_data[bo + 6 .. bo + bpb];
                for (0..16) |j| {
                    const q_byte = qs[j];
                    const lo = q_byte & 0x0F;
                    const hi = q_byte >> 4;
                    const bit_lo = (qh >> @intCast(j)) & 1;
                    const bit_hi = (qh >> @intCast(j + 16)) & 1;
                    output[out_i + j] = scale * @as(f32, @floatFromInt(@as(i32, @intCast(lo | (bit_lo << 4))) - 16));
                    output[out_i + 16 + j] = scale * @as(f32, @floatFromInt(@as(i32, @intCast(hi | (bit_hi << 4))) - 16));
                }
                out_i += block_size;
            }
        },
        .q5_1 => {
            const block_size: usize = 32;
            const bpb: usize = 24;
            const bpr = @as(usize, cols) / block_size;
            const row_off = @as(usize, row) * bpr * bpb;
            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bo = row_off + b * bpb;
                const scale_bits = std.mem.readInt(u16, raw_data[bo..][0..2], .little);
                const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
                const min_bits = std.mem.readInt(u16, raw_data[bo + 2 ..][0..2], .little);
                const min_val: f32 = @floatCast(@as(f16, @bitCast(min_bits)));
                const qh = std.mem.readInt(u32, raw_data[bo + 4 ..][0..4], .little);
                const qs = raw_data[bo + 8 .. bo + bpb];
                for (0..16) |j| {
                    const q_byte = qs[j];
                    const lo = q_byte & 0x0F;
                    const hi = q_byte >> 4;
                    const bit_lo = (qh >> @intCast(j)) & 1;
                    const bit_hi = (qh >> @intCast(j + 16)) & 1;
                    output[out_i + j] = scale * @as(f32, @floatFromInt(lo | (bit_lo << 4))) + min_val;
                    output[out_i + 16 + j] = scale * @as(f32, @floatFromInt(hi | (bit_hi << 4))) + min_val;
                }
                out_i += block_size;
            }
        },
        .q8_0 => {
            const block_size: usize = 32;
            const bpb: usize = 34;
            const bpr = @as(usize, cols) / block_size;
            const row_off = @as(usize, row) * bpr * bpb;
            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bo = row_off + b * bpb;
                const scale_bits = std.mem.readInt(u16, raw_data[bo..][0..2], .little);
                const scale: f32 = @floatCast(@as(f16, @bitCast(scale_bits)));
                for (0..block_size) |j| {
                    const v: i8 = @bitCast(raw_data[bo + 2 + j]);
                    output[out_i] = @as(f32, @floatFromInt(v)) * scale;
                    out_i += 1;
                }
            }
        },
        .q6_k => {
            const bpb: usize = 210;
            const bpr = @as(usize, cols) / 256;
            const row_off = @as(usize, row) * bpr * bpb;
            var out_i: usize = 0;
            for (0..bpr) |bi| {
                const bb = row_off + bi * bpb;
                const d_bits = std.mem.readInt(u16, raw_data[bb + 208 ..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));

                var ql_off: usize = bb;
                var qh_off: usize = bb + 128;
                var sc_off: usize = bb + 192;
                for (0..2) |_| {
                    for (0..32) |l| {
                        const scale_idx = l / 16;
                        const ql_lo = raw_data[ql_off + l];
                        const ql_hi = raw_data[ql_off + l + 32];
                        const qh = raw_data[qh_off + l];

                        const rq0: u8 = (ql_lo & 0xF) | (((qh >> 0) & 3) << 4);
                        const rq1: u8 = (ql_hi & 0xF) | (((qh >> 2) & 3) << 4);
                        const rq2: u8 = (ql_lo >> 4) | (((qh >> 4) & 3) << 4);
                        const rq3: u8 = (ql_hi >> 4) | (((qh >> 6) & 3) << 4);

                        const q0: f32 = @floatFromInt(@as(i16, @intCast(rq0)) - 32);
                        const q1: f32 = @floatFromInt(@as(i16, @intCast(rq1)) - 32);
                        const q2: f32 = @floatFromInt(@as(i16, @intCast(rq2)) - 32);
                        const q3: f32 = @floatFromInt(@as(i16, @intCast(rq3)) - 32);

                        const s0: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_off + scale_idx])));
                        const s1: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_off + scale_idx + 2])));
                        const s2: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_off + scale_idx + 4])));
                        const s3: f32 = @floatFromInt(@as(i8, @bitCast(raw_data[sc_off + scale_idx + 6])));

                        output[out_i + l] = d * s0 * q0;
                        output[out_i + 32 + l] = d * s1 * q1;
                        output[out_i + 64 + l] = d * s2 * q2;
                        output[out_i + 96 + l] = d * s3 * q3;
                    }
                    ql_off += 64;
                    qh_off += 32;
                    sc_off += 8;
                    out_i += 128;
                }
            }
        },
        .q5_k => {
            const bpb: usize = 176;
            const bpr = @as(usize, cols) / 256;
            const row_off = @as(usize, row) * bpr * bpb;
            var out_i: usize = 0;
            for (0..bpr) |bi| {
                const bb = row_off + bi * bpb;
                const d_bits = std.mem.readInt(u16, raw_data[bb..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
                const dm_bits = std.mem.readInt(u16, raw_data[bb + 2 ..][0..2], .little);
                const dmin: f32 = @floatCast(@as(f16, @bitCast(dm_bits)));
                const scales = raw_data[bb + 4 .. bb + 16];
                const qh = raw_data[bb + 16 .. bb + 48];
                const qs = raw_data[bb + 48 .. bb + 176];
                var is: usize = 0;
                for (0..4) |j| {
                    const sm0 = getScaleMinK4(is, scales);
                    const d1 = d * @as(f32, @floatFromInt(sm0.sc));
                    const m1 = dmin * @as(f32, @floatFromInt(sm0.m));
                    const sm1 = getScaleMinK4(is + 1, scales);
                    const d2 = d * @as(f32, @floatFromInt(sm1.sc));
                    const m2 = dmin * @as(f32, @floatFromInt(sm1.m));

                    for (0..32) |l| {
                        const ql_lo: u8 = qs[j * 32 + l] & 0xF;
                        const ql_hi: u8 = qs[j * 32 + l] >> 4;
                        const hb_lo: u8 = (qh[l] >> @intCast(j * 2)) & 1;
                        const hb_hi: u8 = (qh[l] >> @intCast(j * 2 + 1)) & 1;
                        output[out_i + l] = d1 * @as(f32, @floatFromInt(ql_lo | (hb_lo << 4))) - m1;
                        output[out_i + 32 + l] = d2 * @as(f32, @floatFromInt(ql_hi | (hb_hi << 4))) - m2;
                    }
                    out_i += 64;
                    is += 2;
                }
            }
        },
        .q4_k => {
            const bpb: usize = 144;
            const bpr = @as(usize, cols) / 256;
            const row_off = @as(usize, row) * bpr * bpb;
            var out_i: usize = 0;
            for (0..bpr) |bi| {
                const bb = row_off + bi * bpb;
                const d_bits = std.mem.readInt(u16, raw_data[bb..][0..2], .little);
                const d: f32 = @floatCast(@as(f16, @bitCast(d_bits)));
                const dm_bits = std.mem.readInt(u16, raw_data[bb + 2 ..][0..2], .little);
                const dmin: f32 = @floatCast(@as(f16, @bitCast(dm_bits)));
                const scales = raw_data[bb + 4 .. bb + 16];
                const qs = raw_data[bb + 16 .. bb + 144];
                var is: usize = 0;
                var qo: usize = 0;
                for (0..4) |_| {
                    const sm0 = getScaleMinK4(is, scales);
                    const d1 = d * @as(f32, @floatFromInt(sm0.sc));
                    const m1 = dmin * @as(f32, @floatFromInt(sm0.m));
                    const sm1 = getScaleMinK4(is + 1, scales);
                    const d2 = d * @as(f32, @floatFromInt(sm1.sc));
                    const m2 = dmin * @as(f32, @floatFromInt(sm1.m));
                    for (0..32) |l| {
                        output[out_i] = d1 * @as(f32, @floatFromInt(qs[qo + l] & 0xF)) - m1;
                        out_i += 1;
                    }
                    for (0..32) |l| {
                        output[out_i] = d2 * @as(f32, @floatFromInt(qs[qo + l] >> 4)) - m2;
                        out_i += 1;
                    }
                    qo += 32;
                    is += 2;
                }
            }
        },
        .mxfp4 => {
            const bpb: usize = 17;
            const bpr = @as(usize, cols) / 32;
            const row_off = @as(usize, row) * bpr * bpb;
            const lut = [16]f32{ 0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6 };
            var out_i: usize = 0;
            for (0..bpr) |b| {
                const bo = row_off + b * bpb;
                const exp_byte = raw_data[bo];
                const d: f32 = @bitCast(if (exp_byte == 0) @as(u32, 0x00400000) else @as(u32, @intCast(exp_byte)) << 23);
                const qs = raw_data[bo + 1 .. bo + 17];
                for (0..16) |j| {
                    output[out_i + j] = d * lut[qs[j] & 0x0F];
                    output[out_i + j + 16] = d * lut[qs[j] >> 4];
                }
                out_i += 32;
            }
        },
        else => {
            log.warn("Unsupported quant type {d}, using zeros", .{@intFromEnum(quant_type)});
            @memset(output, 0);
        },
    }
}
