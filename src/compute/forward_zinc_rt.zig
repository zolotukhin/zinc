//! ZINC_RT forward-pass bring-up.
//! M1 wires a scalar forward path for the hybrid Qwen MoE+SSM model used by
//! the RDNA migration harness, plus the first AMDGPU-CS-produced token
//! boundary. The old no-layer tail remains as a smoke fallback for unsupported
//! shapes.
//! @section Inference Runtime
const std = @import("std");
const gguf = @import("gguf");
const zinc_rt = @import("zinc_rt");
const ring = zinc_rt.ring;
const cpu_ring = zinc_rt.cpu_ring;
const dequant = zinc_rt.kernels.dequant;

const thread_pool = @import("thread_pool.zig");
const log = std.log.scoped(.zinc_rt_forward);

fn getenv(name: [*:0]const u8) ?[:0]const u8 {
    return if (std.c.getenv(name)) |p| std.mem.span(p) else null;
}

/// Decode-token budget used by the M0 smoke tail and by benchmarks that want a
/// short, bounded run on the scalar reference path. The full M1 forward respects
/// the caller-supplied `max_tokens` and uses this only as a hard ceiling when no
/// other budget is in scope. Override via ZINC_RT_MAX_DECODE_TOKENS.
pub const m0_max_decode_tokens_default: u32 = 8;

fn m0MaxDecodeTokens() u32 {
    if (getenv("ZINC_RT_MAX_DECODE_TOKENS")) |raw| {
        if (std.fmt.parseInt(u32, raw, 10) catch null) |parsed| {
            if (parsed > 0) return parsed;
        }
    }
    return m0_max_decode_tokens_default;
}

const WeightView = struct {
    raw: []const u8,
    type_: gguf.GGMLType,
};

/// Loaded GGUF model plus all the load-time scratch derived from it.
/// Owns the mmap'd file, the resolved per-layer tensor table, the cached F32
/// dequants of small norm/SSM-side tensors, and the re-quantized weight blobs
/// the decode path streams in place of the larger source-format weights. A
/// `Model` is built once with `load` and consumed by `generate` /
/// `generateWithOptions`; the GPU rings get the same handle so they can read
/// the underlying bytes too.
pub const Model = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    file: std.Io.File,
    mmap_data: []align(std.heap.page_size_min) const u8,
    gguf_file: gguf.GGUFFile,
    config: CpuModelConfig,
    layer_tensors: []LayerTensors,
    rope_inv_freq: []f32,
    /// Parallel inv_freq table built from rope_freq_base_swa for Gemma SWA
    /// layers. Null when the model doesn't carry a SWA base.
    rope_inv_freq_swa: ?[]f32,
    /// Per-layer scalar applied to the layer output before the next layer
    /// reads it. Gemma 4 carries `blk.N.layer_output_scale.weight` (size 1).
    /// Null when the model doesn't ship them; treat as 1.0 in that case.
    layer_output_scales: ?[]f32,
    attn_sinks: []f32,
    ssm_conv1d_kernels: []?[]f32,
    embed_info: gguf.TensorInfo,
    final_norm_info: gguf.TensorInfo,
    lm_head_info: gguf.TensorInfo,
    /// `Q4_0`-re-quantized copy of the `Q8_0` LM-head weights, built at load time
    /// so the per-token logit matvec streams ~half the DRAM bytes. Null when the
    /// source is not Q8_0 or re-quantization could not be allocated; decode then
    /// falls back to `lm_head_info`.
    lm_head_q4_0: ?[]u8 = null,
    /// Load-time re-encoded copies of weight matrices whose per-token DRAM / L3
    /// stream is worth shrinking and that tolerate a coarser format, keyed by the
    /// source tensor's mmap pointer. Three kinds: `Q8_0` per-layer projection
    /// matrices re-quantized to `Q4_0` (attention q/k/v/output, the SSM
    /// in/gate/output projections — `attn_qkv` is the biggest weight matrix
    /// streamed every token at ~17 MiB/layer, far past L3, and its q/k rows are
    /// L2-normalised per group so per-weight noise averages out — and the
    /// shared-MoE expert FFN), routed-expert `Q5_K`/`Q6_K` down projections
    /// re-quantized to `Q4_0`, and the `F32` MoE router (`ffn_gate_inp`,
    /// ~2 MiB/layer, matvec'd serially every layer every token) re-quantized to
    /// `Q8_0`. `requantOrRaw` maps a source tensor to its blob (`Q8_0`→`Q4_0`,
    /// `F32`→`Q8_0`); empty / a miss falls back to the original.
    requant_blobs: std.AutoHashMapUnmanaged(usize, []u8) = .{},
    /// More aggressive expert re-encodings used only after prefill has produced
    /// the first token, so the prompt boundary stays on higher-fidelity weights.
    decode_requant_blobs: std.AutoHashMapUnmanaged(usize, []u8) = .{},
    /// Load-time F32 dequant of the small per-layer norm/SSM-side tensors that
    /// the scalar decode path re-reads every token (`attn_norm`, `ffn_norm`,
    /// `post_attention_norm`, `attn_q_norm`, `attn_k_norm`, `ssm_norm`,
    /// `ssm_dt_bias`, `ssm_a`, and the final output norm). Keyed by GGUF
    /// tensor offset. Across 40 layers + final norm this removes ~190
    /// per-token `readTensorFlat → dequant.row` dispatches and ~1 MiB of
    /// redundant memcpy into `state.row_scratch`, which would otherwise evict
    /// the big matvec's weight rows from L1/L2 every layer.
    small_tensor_f32_cache: std.AutoHashMapUnmanaged(u64, []f32) = .{},
    final_norm_weight: []f32,
    hidden_dim: u32,
    vocab_size: u32,
    n_layers: u32,
    n_experts: u32,
    moe_topk_limit: u32,
    moe_topk_override: bool,
    lm_head_decode_rows: u32,
    full_attn_interval: u32,
    has_ssm: bool,
    rms_norm_eps: f32,

    /// Memory-map a GGUF model file, parse its metadata, and pre-build the
    /// load-time caches the scalar decode path relies on (small F32 norm/SSM
    /// tensors, Q8_0→Q4_0 and F32→Q8_0 re-quantizations of the heavier
    /// per-token weight streams, RoPE inverse-frequency table, attention sinks,
    /// SSM conv1d kernels). The returned handle must eventually be released
    /// with `deinit`.
    /// @param path Filesystem path to the GGUF model.
    /// @param allocator Owns every secondary allocation reachable from `Model`.
    /// @returns A fully-initialised `Model`, or an error if the file is
    /// unreadable, the tensors don't match the expected shapes, or the GGUF
    /// metadata is malformed.
    pub fn load(io: std.Io, path: []const u8, allocator: std.mem.Allocator) !Model {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);

        const stat = try file.stat(io);
        const mmap_data = try std.posix.mmap(
            null,
            stat.size,
            std.posix.PROT{ .READ = true },
            .{ .TYPE = .PRIVATE },
            file.handle,
            0,
        );
        errdefer std.posix.munmap(mmap_data);

        var gf = try gguf.parseWithOptions(mmap_data, allocator, .{ .log_summary = false });
        errdefer gf.deinit();

        const embed_info = (gf.findTensor("token_embd.weight") orelse return error.TensorNotFound).*;
        const final_norm_info = (gf.findTensor("output_norm.weight") orelse return error.TensorNotFound).*;
        const lm_head_info = (gf.findTensor("output.weight") orelse gf.findTensor("token_embd.weight") orelse return error.TensorNotFound).*;

        const dims = inferMatrixDims(embed_info);
        const hidden_dim = dims.cols;
        const vocab_size = dims.rows;
        if (hidden_dim == 0 or vocab_size == 0) return error.UnsupportedShape;

        const lm_dims = inferMatrixDims(lm_head_info);
        if (lm_dims.cols != hidden_dim or lm_dims.rows == 0) return error.ShapeMismatch;
        const effective_vocab = @min(vocab_size, lm_dims.rows);

        const norm_elems = tensorElements(final_norm_info);
        if (norm_elems < hidden_dim) return error.ShapeMismatch;

        const final_norm_weight = try allocator.alloc(f32, hidden_dim);
        errdefer allocator.free(final_norm_weight);
        try dequant.row(
            tensorDataRaw(mmap_data, &gf, final_norm_info),
            0,
            hidden_dim,
            final_norm_info.type_,
            final_norm_weight,
        );

        const arch = gf.getString("general.architecture") orelse "unknown";
        const config = extractCpuModelConfig(&gf, arch, hidden_dim, effective_vocab);
        const layer_tensors = try resolveLayerTensors(&gf, allocator, config.n_layers);
        errdefer allocator.free(layer_tensors);
        const raw_moe_topk_override = getenv("ZINC_QWEN36_MOE_TOPK");
        const moe_topk_limit = resolveQwen36MoeTopkLimitForEnv(config, layer_tensors, raw_moe_topk_override);
        if (moe_topk_limit > 0) {
            log.info("M1 host-assisted Qwen 3.6 MoE top-k capped at {d} (set ZINC_QWEN36_MOE_TOPK={d} to restore metadata top-k)", .{
                moe_topk_limit,
                config.n_experts_used,
            });
        } else if (isQwen36LikeF32Ssm(config, layer_tensors) and raw_moe_topk_override != null) {
            log.info("M1 host-assisted Qwen 3.6 MoE top-k cap disabled via ZINC_QWEN36_MOE_TOPK", .{});
        }
        const raw_lm_rows_override = getenv("ZINC_RT_LM_HEAD_ROWS");
        const lm_head_decode_rows = resolveQwen36LmHeadDecodeRowsForEnv(config, layer_tensors, raw_lm_rows_override);
        if (lm_head_decode_rows < effective_vocab) {
            log.info("M1 host-assisted decode LM-head row scan capped at {d}/{d} rows (set ZINC_RT_LM_HEAD_ROWS=0 to restore full vocab)", .{
                lm_head_decode_rows,
                effective_vocab,
            });
        }
        const rope_inv_freq = try buildRopeInvFreqs(&gf, allocator, arch, config, config.rope_freq_base, 0);
        errdefer allocator.free(rope_inv_freq);
        // Gemma 4 global layers consume the proportional rope_freqs.weight
        // divisor. Applied only to the global table; SWA layers keep raw inv_freq.
        if (config.is_gemma) applyRopeFreqFactors(mmap_data, &gf, rope_inv_freq);

        // Per-layer output scales for Gemma 4 (`blk.N.layer_output_scale.weight`).
        // Default to 1.0 when absent; only allocate when at least one is non-trivial.
        const layer_output_scales: ?[]f32 = if (config.is_gemma) blk: {
            const scales = try allocator.alloc(f32, config.n_layers);
            errdefer allocator.free(scales);
            var any_nontrivial = false;
            for (0..config.n_layers) |li| {
                scales[li] = 1.0;
                var name_buf: [128]u8 = undefined;
                const name = std.fmt.bufPrint(&name_buf, "blk.{d}.layer_output_scale.weight", .{li}) catch continue;
                for (gf.tensors.items) |ti| {
                    if (!std.mem.eql(u8, ti.name, name)) continue;
                    if (ti.type_ != .f32) break;
                    const off = gf.tensor_data_offset + ti.offset;
                    if (off + @sizeOf(f32) > mmap_data.len) break;
                    const ptr: *const f32 = @ptrCast(@alignCast(mmap_data.ptr + off));
                    scales[li] = ptr.*;
                    if (scales[li] != 1.0) any_nontrivial = true;
                    break;
                }
            }
            if (!any_nontrivial) {
                allocator.free(scales);
                break :blk null;
            }
            break :blk scales;
        } else null;
        errdefer if (layer_output_scales) |s| allocator.free(s);
        // Gemma 4 SWA layers carry a smaller per-layer head_dim than the model
        // metadata's `attention.key_length`; the SWA RoPE table must use that
        // smaller head_dim for the exponent normalization.
        const rope_inv_freq_swa: ?[]f32 = if (config.is_gemma and config.rope_freq_base_swa > 0) blk: {
            var swa_rope_dim: u32 = 0;
            for (layer_tensors) |lt| {
                // SWA layers carry attn_v; global layers drop it.
                if (lt.attn_v == null) continue;
                if (lt.attn_q_norm) |qn| {
                    swa_rope_dim = @intCast(qn.numElements());
                    break;
                }
                if (lt.attn_k_norm) |kn| {
                    swa_rope_dim = @intCast(kn.numElements());
                    break;
                }
            }
            break :blk try buildRopeInvFreqs(&gf, allocator, arch, config, config.rope_freq_base_swa, swa_rope_dim);
        } else null;
        errdefer if (rope_inv_freq_swa) |s| allocator.free(s);
        const attn_sinks = try buildAttentionSinks(mmap_data, &gf, allocator, layer_tensors, config);
        errdefer allocator.free(attn_sinks);
        const ssm_conv1d_kernels = try buildSsmConv1dKernelCache(mmap_data, &gf, allocator, layer_tensors, config);
        errdefer freeLayerF32Cache(allocator, ssm_conv1d_kernels);
        var small_tensor_f32_cache = try buildSmallTensorF32Cache(mmap_data, &gf, allocator, layer_tensors, final_norm_info);
        errdefer {
            var stf_it = small_tensor_f32_cache.valueIterator();
            while (stf_it.next()) |v| allocator.free(v.*);
            small_tensor_f32_cache.deinit(allocator);
        }

        // Re-quantize a Q8_0 LM head down to Q4_0 (decode is DRAM-bandwidth bound
        // on these large Q8_0 matmuls; ~halving the logit-matvec byte stream is a
        // real per-token win and argmax tolerates the extra rounding noise).
        // Nothing fallible runs after this, so a failed alloc just declines it.
        const lm_head_q4_0 = buildRequantizedQ4_0(
            allocator,
            tensorDataRaw(mmap_data, &gf, lm_head_info),
            lm_head_info.type_,
            effective_vocab,
            hidden_dim,
        );
        if (lm_head_q4_0) |buf| {
            const q8_mib = @as(usize, effective_vocab) * (@as(usize, hidden_dim) / 32) * 34 / (1024 * 1024);
            log.info("M1 host-assisted re-quantized Q8_0 lm_head -> Q4_0 for decode ({d} MiB -> {d} MiB)", .{ q8_mib, buf.len / (1024 * 1024) });
        }

        // Same trick for the rest of the model's Q8_0 weight matrices (attention
        // q/k/v/output projections, the SSM in/gate/output projections, the
        // shared-MoE expert FFN), the routed-expert Q5_K/Q6_K down projections,
        // plus re-quantize the F32 MoE router
        // (`ffn_gate_inp`) down to Q8_0. The Q8->Q4 part halves those
        // projections' L3/DRAM footprint; the router is a ~2 MiB/layer F32 matvec
        // run *serially* every layer every token (sub-threshold for the worker
        // pool), so cutting it to ~0.5 MiB at Q8_0 — ample precision for a
        // softmax-then-top-k gate — is a real serial-path win. `attn_qkv` (the
        // SSM in-projection, ~17 MiB/layer at Q8_0 — the single largest weight
        // matrix streamed every decode token, far past L3 so genuinely
        // DRAM-bound) is now re-quantized too: its q/k rows are L2-normalised
        // per group before the recurrence (so per-weight noise is averaged away)
        // and 4-bit-ish dense projections are standard for this already-Q4_K
        // model. Anything that can't be re-encoded (alloc failure / wrong shape
        // / wrong type) keeps its original form via `requantOrRaw`; nothing
        // fallible runs after this.
        var requant_blobs: std.AutoHashMapUnmanaged(usize, []u8) = .{};
        var decode_requant_blobs: std.AutoHashMapUnmanaged(usize, []u8) = .{};
        {
            var saved_bytes: usize = 0;
            var q4_count: usize = 0;
            var decode_gate_up_count: usize = 0;
            var down_k_count: usize = 0;
            var router_count: usize = 0;
            for (layer_tensors) |lt| {
                const q4_candidates = [_]?gguf.TensorInfo{
                    lt.attn_q,         lt.attn_k,         lt.attn_v,
                    lt.attn_output,    lt.attn_qkv,       lt.attn_gate,
                    lt.ssm_out,        lt.ffn_gate_shexp, lt.ffn_up_shexp,
                    lt.ffn_down_shexp,
                };
                for (q4_candidates) |maybe| {
                    const info = maybe orelse continue;
                    if (info.type_ != .q8_0) continue;
                    const md = inferMatrixDims(info);
                    if (md.rows == 0 or md.cols == 0 or md.cols % 32 != 0) continue;
                    const src_raw = tensorDataRaw(mmap_data, &gf, info);
                    const key = @intFromPtr(src_raw.ptr);
                    if (requant_blobs.contains(key)) continue;
                    const q40 = buildRequantizedQ4_0(allocator, src_raw, info.type_, md.rows, md.cols) orelse continue;
                    requant_blobs.put(allocator, key, q40) catch {
                        allocator.free(q40);
                        continue;
                    };
                    saved_bytes += src_raw.len -| q40.len;
                    q4_count += 1;
                }
                const gate_up_k_candidates = [_]?gguf.TensorInfo{ lt.ffn_gate_exps, lt.ffn_up_exps };
                for (gate_up_k_candidates) |maybe| {
                    const info = maybe orelse continue;
                    if (info.type_ != .q5_k and info.type_ != .q6_k) continue;
                    const md = inferFlatMatrixDims(info);
                    if (md.rows == 0 or md.cols == 0 or md.cols % 32 != 0) continue;
                    const src_raw = tensorDataRaw(mmap_data, &gf, info);
                    const key = @intFromPtr(src_raw.ptr);
                    if (decode_requant_blobs.contains(key)) continue;
                    const q40 = buildRequantizedQ4_0(allocator, src_raw, info.type_, md.rows, md.cols) orelse continue;
                    decode_requant_blobs.put(allocator, key, q40) catch {
                        allocator.free(q40);
                        continue;
                    };
                    saved_bytes += src_raw.len -| q40.len;
                    decode_gate_up_count += 1;
                }
                add_down_k: {
                    const info = lt.ffn_down_exps orelse break :add_down_k;
                    if (info.type_ != .q5_k and info.type_ != .q6_k) break :add_down_k;
                    const md = inferFlatMatrixDims(info);
                    if (md.rows == 0 or md.cols == 0 or md.cols % 32 != 0) break :add_down_k;
                    const src_raw = tensorDataRaw(mmap_data, &gf, info);
                    const key = @intFromPtr(src_raw.ptr);
                    if (requant_blobs.contains(key)) break :add_down_k;
                    const q40 = buildRequantizedQ4_0(allocator, src_raw, info.type_, md.rows, md.cols) orelse break :add_down_k;
                    requant_blobs.put(allocator, key, q40) catch {
                        allocator.free(q40);
                        break :add_down_k;
                    };
                    saved_bytes += src_raw.len -| q40.len;
                    down_k_count += 1;
                }
                add_router: {
                    const router_info = lt.ffn_gate_inp orelse break :add_router;
                    if (router_info.type_ != .f32) break :add_router;
                    const md = inferMatrixDims(router_info);
                    if (md.rows == 0 or md.cols == 0 or md.cols % 32 != 0) break :add_router;
                    const src_raw = tensorDataRaw(mmap_data, &gf, router_info);
                    const key = @intFromPtr(src_raw.ptr);
                    if (requant_blobs.contains(key)) break :add_router;
                    const q8_buf = buildRequantizedQ8_0FromF32(allocator, src_raw, md.rows, md.cols) orelse break :add_router;
                    requant_blobs.put(allocator, key, q8_buf) catch {
                        allocator.free(q8_buf);
                        break :add_router;
                    };
                    saved_bytes += src_raw.len -| q8_buf.len;
                    router_count += 1;
                }
            }
            if (q4_count > 0 or decode_gate_up_count > 0 or down_k_count > 0 or router_count > 0) {
                log.info("M1 host-assisted re-encoded weights for decode: {d} Q8_0 matrices -> Q4_0, {d} decode-only Q5_K/Q6_K expert gate/up matrices -> Q4_0, {d} Q5_K/Q6_K expert-down matrices -> Q4_0, {d} F32 routers -> Q8_0 (~{d} MiB less source-format weight traffic when touched)", .{
                    q4_count, decode_gate_up_count, down_k_count, router_count, saved_bytes / (1024 * 1024),
                });
            }
        }

        return .{
            .allocator = allocator,
            .io = io,
            .file = file,
            .mmap_data = mmap_data,
            .gguf_file = gf,
            .config = config,
            .layer_tensors = layer_tensors,
            .rope_inv_freq = rope_inv_freq,
            .rope_inv_freq_swa = rope_inv_freq_swa,
            .layer_output_scales = layer_output_scales,
            .attn_sinks = attn_sinks,
            .ssm_conv1d_kernels = ssm_conv1d_kernels,
            .embed_info = embed_info,
            .final_norm_info = final_norm_info,
            .lm_head_info = lm_head_info,
            .lm_head_q4_0 = lm_head_q4_0,
            .requant_blobs = requant_blobs,
            .decode_requant_blobs = decode_requant_blobs,
            .small_tensor_f32_cache = small_tensor_f32_cache,
            .final_norm_weight = final_norm_weight,
            .hidden_dim = hidden_dim,
            .vocab_size = effective_vocab,
            .n_layers = config.n_layers,
            .n_experts = config.n_experts,
            .moe_topk_limit = moe_topk_limit,
            .moe_topk_override = raw_moe_topk_override != null,
            .lm_head_decode_rows = lm_head_decode_rows,
            .full_attn_interval = config.full_attn_interval,
            .has_ssm = config.ssm_d_inner > 0,
            .rms_norm_eps = rmsNormEps(&gf),
        };
    }

    /// Release every allocation reachable from `Model`: the re-quantized
    /// weight blobs, the small-tensor F32 cache, the SSM/RoPE/attention-sink
    /// scratch, the GGUF parse state, the mmap, and the file handle. Poisons
    /// `self`; the handle must not be reused afterwards.
    pub fn deinit(self: *Model) void {
        if (self.lm_head_q4_0) |buf| self.allocator.free(buf);
        {
            var it = self.requant_blobs.valueIterator();
            while (it.next()) |v| self.allocator.free(v.*);
            self.requant_blobs.deinit(self.allocator);
        }
        {
            var it = self.decode_requant_blobs.valueIterator();
            while (it.next()) |v| self.allocator.free(v.*);
            self.decode_requant_blobs.deinit(self.allocator);
        }
        {
            var it = self.small_tensor_f32_cache.valueIterator();
            while (it.next()) |v| self.allocator.free(v.*);
            self.small_tensor_f32_cache.deinit(self.allocator);
        }
        self.allocator.free(self.attn_sinks);
        freeLayerF32Cache(self.allocator, self.ssm_conv1d_kernels);
        self.allocator.free(self.rope_inv_freq);
        if (self.rope_inv_freq_swa) |swa| self.allocator.free(swa);
        if (self.layer_output_scales) |s| self.allocator.free(s);
        self.allocator.free(self.layer_tensors);
        self.allocator.free(self.final_norm_weight);
        self.gguf_file.deinit();
        std.posix.munmap(self.mmap_data);
        self.file.close(self.io);
        self.* = undefined;
    }

    fn tensorData(self: *const Model, info: gguf.TensorInfo) []const u8 {
        return tensorDataRaw(self.mmap_data, &self.gguf_file, info);
    }

    /// Look up the load-time F32 dequant of a small norm-like tensor; null if
    /// the tensor was not pre-cached (caller falls back to dequant-on-read).
    fn cachedF32(self: *const Model, info: gguf.TensorInfo) ?[]const f32 {
        return self.small_tensor_f32_cache.get(info.offset);
    }

    /// Decode-time backing store for `info`: the load-time re-encoded blob if one
    /// was built (`Q8_0`→`Q4_0`, or `F32`→`Q8_0` for the MoE router — a lighter
    /// weight stream), otherwise the original mmap'd tensor.
    fn requantOrRaw(self: *const Model, info: gguf.TensorInfo) WeightView {
        const raw = tensorDataRaw(self.mmap_data, &self.gguf_file, info);
        switch (info.type_) {
            .q8_0, .q5_k, .q6_k, .f32 => {
                if (self.requant_blobs.get(@intFromPtr(raw.ptr))) |blob| {
                    return .{ .raw = blob, .type_ = if (info.type_ == .f32) .q8_0 else .q4_0 };
                }
            },
            else => {},
        }
        return .{ .raw = raw, .type_ = info.type_ };
    }

    fn decodeRequantOrRaw(self: *const Model, info: gguf.TensorInfo) WeightView {
        const raw = tensorDataRaw(self.mmap_data, &self.gguf_file, info);
        switch (info.type_) {
            .q5_k, .q6_k => {
                if (self.decode_requant_blobs.get(@intFromPtr(raw.ptr))) |blob| {
                    return .{ .raw = blob, .type_ = .q4_0 };
                }
            },
            else => {},
        }
        return self.requantOrRaw(info);
    }

    /// Emit the per-token decode IR for this model's shape and run the
    /// validator on it, returning a node/layer count summary the caller can
    /// log or assert against. Used as an admission gate before a real decode
    /// run so a malformed graph fails loud and early instead of corrupting the
    /// scalar kernels mid-token.
    /// @param allocator Used for the throwaway IR graph; released before
    /// return.
    /// @returns Summary of the emitted-and-validated decode graph.
    pub fn validateDecodeGraph(self: *const Model, allocator: std.mem.Allocator) !DecodeGraphSummary {
        return emitDecodeGraphForShape(
            allocator,
            self.n_layers,
            self.full_attn_interval,
            self.n_experts,
            self.has_ssm,
        );
    }

    fn canRunScalarHybrid(self: *const Model) bool {
        const cfg = self.config;
        if (cfg.n_layers == 0 or cfg.hidden_dim == 0) return false;
        if (self.layer_tensors.len != cfg.n_layers) return false;

        const has_any_moe = cfg.n_experts > 0 and cfg.n_experts_used > 0;
        const has_any_ssm = cfg.ssm_d_inner > 0 and cfg.ssm_dt_rank > 0 and cfg.ssm_d_state > 0;
        // Pure-dense models are routed via canRunScalarDense.
        if (!has_any_moe and !has_any_ssm) return false;

        for (self.layer_tensors, 0..) |lt, li| {
            if (lt.attn_norm == null or (lt.ffn_norm == null and lt.post_attention_norm == null)) return false;

            // FFN: accept either MoE or dense tensors (route per layer at eval time).
            // Gemma 4 MoE uses a fused `ffn_gate_up_exps` tensor instead of
            // separate gate/up; treat that as a valid MoE shape too.
            const has_moe_layer = lt.ffn_gate_inp != null and
                ((lt.ffn_gate_exps != null and lt.ffn_up_exps != null and lt.ffn_down_exps != null) or
                    (lt.ffn_gate_up_exps != null and lt.ffn_down_exps != null));
            const has_dense_layer = lt.ffn_gate != null and lt.ffn_up != null and lt.ffn_down != null;
            if (!has_moe_layer and !has_dense_layer) return false;

            const layer: u32 = @intCast(li);
            const ssm_eligible_layer = has_any_ssm and !isFullAttentionLayer(cfg, layer);
            if (ssm_eligible_layer) {
                if (lt.attn_qkv == null or lt.attn_gate == null or lt.ssm_alpha == null or
                    lt.ssm_beta == null or lt.ssm_conv1d == null or lt.ssm_out == null) return false;
            } else {
                if (lt.attn_q == null or lt.attn_k == null or lt.attn_output == null) return false;
                if (lt.attn_v == null and !cfg.is_gemma) return false;
            }
        }
        return true;
    }

    fn canRunScalarDense(self: *const Model) bool {
        const cfg = self.config;
        if (cfg.n_layers == 0 or cfg.hidden_dim == 0) return false;
        if (cfg.n_experts != 0 or cfg.ssm_d_inner != 0) return false;
        if (self.layer_tensors.len != cfg.n_layers) return false;
        for (self.layer_tensors) |lt| {
            if (lt.attn_norm == null or (lt.ffn_norm == null and lt.post_attention_norm == null)) return false;
            if (lt.attn_q == null or lt.attn_k == null or lt.attn_output == null) return false;
            // Gemma global-attention layers omit attn_v (K reused as V); allow it
            // only on Gemma where runAttentionLayer falls back to a K-as-V copy.
            if (lt.attn_v == null and !cfg.is_gemma) return false;
            if (lt.ffn_gate == null or lt.ffn_up == null or lt.ffn_down == null) return false;
        }
        return true;
    }

    fn effectiveMoeTopK(self: *const Model) u32 {
        if (self.moe_topk_limit > 0) return @min(self.config.n_experts_used, self.moe_topk_limit);
        return self.config.n_experts_used;
    }

    fn effectiveDecodeMoeTopK(self: *const Model) u32 {
        const prefill_topk = self.effectiveMoeTopK();
        if (!self.moe_topk_override and
            self.moe_topk_limit == 3 and
            isQwen36LikeF32Ssm(self.config, self.layer_tensors))
        {
            // MIGRATE path: keep prefill at top-3 for the first token, then use
            // Qwen3.6's always-active shared expert during decode. This removes
            // the host-routed expert branch until the router/top-1 slice is
            // lowered into the direct runtime.
            return 0;
        }
        return prefill_topk;
    }

    fn effectiveLmHeadRows(self: *const Model, decode_phase: bool) u32 {
        if (!decode_phase) return self.config.vocab_size;
        return @min(self.config.vocab_size, @max(@as(u32, 1), self.lm_head_decode_rows));
    }
};

/// Build a `Q4_0` re-quantized copy of a blocked weight matrix (row-major,
/// `cols` per row, `rows` rows). Returns null if the source is unsupported / not
/// 32-aligned / empty, or if the destination or scratch allocation fails — the
/// caller then keeps using the original tensor.
fn buildRequantizedQ4_0(
    allocator: std.mem.Allocator,
    src_raw: []const u8,
    src_type: gguf.GGMLType,
    rows: u32,
    cols: u32,
) ?[]u8 {
    switch (src_type) {
        .q8_0, .q5_k, .q6_k => {},
        else => return null,
    }
    if (rows == 0 or cols == 0 or cols % 32 != 0) return null;
    const row_bytes: usize = (@as(usize, cols) / 32) * 18;
    const total: usize = @as(usize, rows) * row_bytes;
    const dst = allocator.alloc(u8, total) catch return null;
    const scratch = allocator.alloc(f32, cols) catch {
        allocator.free(dst);
        return null;
    };
    defer allocator.free(scratch);
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        dequant.row(src_raw, r, cols, src_type, scratch) catch {
            allocator.free(dst);
            return null;
        };
        dequant.quantizeRowToQ4_0(scratch, dst[@as(usize, r) * row_bytes ..][0..row_bytes]);
    }
    return dst;
}

/// Build a `Q8_0` re-quantized copy of an `F32` weight matrix (row-major, `cols`
/// little-endian f32s per row, `rows` rows). Returns null if `cols` isn't a
/// multiple of 32, the shape is empty, the source is short, or an allocation
/// fails — the caller then keeps the original tensor.
fn buildRequantizedQ8_0FromF32(allocator: std.mem.Allocator, src_raw: []const u8, rows: u32, cols: u32) ?[]u8 {
    if (rows == 0 or cols == 0 or cols % 32 != 0) return null;
    if (src_raw.len < @as(usize, rows) * @as(usize, cols) * 4) return null;
    const row_bytes: usize = (@as(usize, cols) / 32) * 34;
    const dst = allocator.alloc(u8, @as(usize, rows) * row_bytes) catch return null;
    const scratch = allocator.alloc(f32, cols) catch {
        allocator.free(dst);
        return null;
    };
    defer allocator.free(scratch);
    var r: u32 = 0;
    while (r < rows) : (r += 1) {
        const src_off = @as(usize, r) * @as(usize, cols) * 4;
        for (0..cols) |c| {
            scratch[c] = @bitCast(std.mem.readInt(u32, src_raw[src_off + c * 4 ..][0..4], .little));
        }
        dequant.quantizeRowToQ8_0(scratch, dst[@as(usize, r) * row_bytes ..][0..row_bytes]);
    }
    return dst;
}

const CpuModelConfig = struct {
    hidden_dim: u32,
    vocab_size: u32,
    n_layers: u32,
    n_heads: u32,
    n_kv_heads: u32,
    head_dim: u32,
    q_dim: u32,
    kv_dim: u32,
    intermediate_dim: u32,
    n_experts: u32,
    n_experts_used: u32,
    rope_dim: u32,
    rope_freq_base: f32,
    full_attn_interval: u32,
    shared_expert_intermediate_dim: u32,
    ssm_d_conv: u32,
    ssm_d_inner: u32,
    ssm_d_state: u32,
    ssm_dt_rank: u32,
    ssm_n_group: u32,
    rms_norm_eps: f32,
    is_gemma: bool,
    rope_freq_base_swa: f32,
    /// Overrides the standard 1/sqrt(head_dim) attention softmax scale. Zero
    /// means use the default; Gemma 4 uses 1.0 even when the GGUF omits an
    /// explicit `attention.scale` key.
    attn_scale: f32,
};

const LayerTensors = struct {
    attn_norm: ?gguf.TensorInfo = null,
    attn_q: ?gguf.TensorInfo = null,
    attn_k: ?gguf.TensorInfo = null,
    attn_v: ?gguf.TensorInfo = null,
    attn_gate: ?gguf.TensorInfo = null,
    attn_output: ?gguf.TensorInfo = null,
    attn_q_norm: ?gguf.TensorInfo = null,
    attn_k_norm: ?gguf.TensorInfo = null,
    ffn_norm: ?gguf.TensorInfo = null,
    post_attention_norm: ?gguf.TensorInfo = null,
    ffn_gate: ?gguf.TensorInfo = null,
    ffn_up: ?gguf.TensorInfo = null,
    ffn_down: ?gguf.TensorInfo = null,
    post_ffw_norm: ?gguf.TensorInfo = null,
    pre_ffw_norm_2: ?gguf.TensorInfo = null,
    ffn_gate_inp: ?gguf.TensorInfo = null,
    ffn_gate_exps: ?gguf.TensorInfo = null,
    ffn_up_exps: ?gguf.TensorInfo = null,
    ffn_down_exps: ?gguf.TensorInfo = null,
    ffn_gate_shexp: ?gguf.TensorInfo = null,
    ffn_up_shexp: ?gguf.TensorInfo = null,
    ffn_down_shexp: ?gguf.TensorInfo = null,
    ffn_gate_inp_shexp: ?gguf.TensorInfo = null,
    // Gemma 4 MoE: parallel shared MLP + routed experts.
    ffn_gate_up_exps: ?gguf.TensorInfo = null,
    ffn_post_norm_1: ?gguf.TensorInfo = null,
    ffn_post_norm_2: ?gguf.TensorInfo = null,
    ffn_gate_inp_scale: ?gguf.TensorInfo = null,
    attn_qkv: ?gguf.TensorInfo = null,
    ssm_alpha: ?gguf.TensorInfo = null,
    ssm_beta: ?gguf.TensorInfo = null,
    ssm_conv1d: ?gguf.TensorInfo = null,
    ssm_out: ?gguf.TensorInfo = null,
    ssm_dt_bias: ?gguf.TensorInfo = null,
    ssm_a: ?gguf.TensorInfo = null,
    ssm_norm: ?gguf.TensorInfo = null,
    attn_sinks: ?gguf.TensorInfo = null,
};

fn extractCpuModelConfig(gf: *const gguf.GGUFFile, arch: []const u8, hidden_dim: u32, vocab_size: u32) CpuModelConfig {
    const n_layers = archU32(gf, arch, "block_count", 0);
    const n_heads = archU32(gf, arch, "attention.head_count", 0);
    const n_kv_heads = archU32(gf, arch, "attention.head_count_kv", n_heads);
    const head_dim = archU32(gf, arch, "attention.key_length", if (n_heads > 0) hidden_dim / n_heads else 0);
    const q_dim = n_heads * head_dim;
    const kv_dim = n_kv_heads * head_dim;
    const ssm_d_inner = archU32(gf, arch, "ssm.inner_size", 0);
    const full_attn_interval = archU32(gf, arch, "full_attention_interval", if (ssm_d_inner > 0) 4 else 1);
    const intermediate_dim = blk: {
        const expert = archU32(gf, arch, "expert_feed_forward_length", 0);
        if (expert > 0) break :blk expert;
        break :blk archU32(gf, arch, "feed_forward_length", hidden_dim * 4);
    };
    const shared_intermediate_dim = blk: {
        const explicit = archU32(gf, arch, "expert_shared_feed_forward_length", 0);
        if (explicit > 0) break :blk explicit;
        const dense = archU32(gf, arch, "feed_forward_length", 0);
        if (dense > 0) break :blk dense;
        break :blk intermediate_dim;
    };
    const rope_dim = archU32(gf, arch, "rope.dimension_count", if (head_dim > 0) head_dim else hidden_dim);

    return .{
        .hidden_dim = hidden_dim,
        .vocab_size = vocab_size,
        .n_layers = n_layers,
        .n_heads = n_heads,
        .n_kv_heads = n_kv_heads,
        .head_dim = head_dim,
        .q_dim = q_dim,
        .kv_dim = kv_dim,
        .intermediate_dim = intermediate_dim,
        .n_experts = archU32(gf, arch, "expert_count", 0),
        .n_experts_used = archU32(gf, arch, "expert_used_count", 0),
        .rope_dim = rope_dim,
        .rope_freq_base = archF32(gf, arch, "rope.freq_base", 10000.0),
        .full_attn_interval = full_attn_interval,
        .shared_expert_intermediate_dim = shared_intermediate_dim,
        .ssm_d_conv = archU32(gf, arch, "ssm.conv_kernel", 0),
        .ssm_d_inner = ssm_d_inner,
        .ssm_d_state = archU32(gf, arch, "ssm.state_size", 0),
        .ssm_dt_rank = archU32(gf, arch, "ssm.time_step_rank", 0),
        .ssm_n_group = archU32(gf, arch, "ssm.group_count", 0),
        .rms_norm_eps = rmsNormEps(gf),
        .is_gemma = std.mem.startsWith(u8, arch, "gemma"),
        .rope_freq_base_swa = archF32(gf, arch, "rope.freq_base_swa", 0),
        .attn_scale = blk: {
            const explicit = archF32(gf, arch, "attention.scale", 0);
            if (explicit > 0) break :blk explicit;
            // Gemma 4 fixes the softmax scale at 1.0 (no 1/sqrt(head_dim));
            // see model loader_metal for the reference. Override with
            // ZINC_GEMMA4_ATTN_SCALE_DEFAULT to A/B test (0 = use default 1/sqrt).
            if (std.mem.eql(u8, arch, "gemma4")) {
                if (getenv("ZINC_GEMMA4_ATTN_SCALE_DEFAULT")) |_| {
                    break :blk 0.0;
                }
                break :blk @as(f32, 1.0);
            }
            break :blk 0.0;
        },
    };
}

fn isQwen36LikeF32Ssm(cfg: CpuModelConfig, layer_tensors: []const LayerTensors) bool {
    if (cfg.n_experts == 0 or cfg.n_experts_used == 0 or cfg.ssm_d_inner == 0) return false;
    if (layer_tensors.len == 0) return false;
    const alpha0 = layer_tensors[0].ssm_alpha orelse return false;
    const beta0 = layer_tensors[0].ssm_beta orelse return false;
    return alpha0.type_ == .f32 and beta0.type_ == .f32;
}

fn resolveQwen36MoeTopkLimit(cfg: CpuModelConfig, layer_tensors: []const LayerTensors) u32 {
    return resolveQwen36MoeTopkLimitForEnv(cfg, layer_tensors, getenv("ZINC_QWEN36_MOE_TOPK"));
}

fn resolveQwen36MoeTopkLimitForEnv(cfg: CpuModelConfig, layer_tensors: []const LayerTensors, raw_override: ?[]const u8) u32 {
    if (!isQwen36LikeF32Ssm(cfg, layer_tensors)) return 0;

    const default_topk: u32 = 3;
    if (raw_override) |raw| {
        const parsed = std.fmt.parseInt(u32, raw, 10) catch default_topk;
        if (parsed == 0 or parsed >= cfg.n_experts_used) return 0;
        return @max(@as(u32, 1), parsed);
    }
    // No env override: use metadata top-k. The historical default of 3 was a
    // benchmark shortcut that traded MoE quality for ~2x decode tok/s; it
    // produced "the answer of the answer" loops on real prompts. Set
    // ZINC_QWEN36_MOE_TOPK=3 to opt back into the perf path.
    return 0;
}

fn resolveQwen36LmHeadDecodeRowsForEnv(cfg: CpuModelConfig, layer_tensors: []const LayerTensors, raw_override: ?[]const u8) u32 {
    if (!isQwen36LikeF32Ssm(cfg, layer_tensors)) return cfg.vocab_size;

    if (raw_override) |raw| {
        const parsed = std.fmt.parseInt(u32, raw, 10) catch cfg.vocab_size;
        if (parsed == 0 or parsed >= cfg.vocab_size) return cfg.vocab_size;
        return @max(@as(u32, 1), parsed);
    }
    // No env override: scan the full vocab. The 4K-row cap was a benchmark
    // shortcut (~3.5–4.3 ms/token saved on RDNA4 R9700 Qwen3.6) that silently
    // dropped low-frequency tokens from decode. Set ZINC_RT_LM_HEAD_ROWS=4096
    // to opt back into the perf path; see docs/ZINC_RT_DESIGN.md §1834.
    return cfg.vocab_size;
}

fn resolveLayerTensors(gf: *const gguf.GGUFFile, allocator: std.mem.Allocator, n_layers: u32) ![]LayerTensors {
    const layers = try allocator.alloc(LayerTensors, n_layers);
    errdefer allocator.free(layers);
    for (layers, 0..) |*lt, li| {
        const layer: u32 = @intCast(li);
        lt.* = .{
            .attn_norm = findLayerTensor(gf, layer, "attn_norm.weight"),
            .attn_q = findLayerTensor(gf, layer, "attn_q.weight"),
            .attn_k = findLayerTensor(gf, layer, "attn_k.weight"),
            .attn_v = findLayerTensor(gf, layer, "attn_v.weight"),
            .attn_gate = findLayerTensor(gf, layer, "attn_gate.weight"),
            .attn_output = findLayerTensor(gf, layer, "attn_output.weight"),
            .attn_q_norm = findLayerTensor(gf, layer, "attn_q_norm.weight"),
            .attn_k_norm = findLayerTensor(gf, layer, "attn_k_norm.weight"),
            .ffn_norm = findLayerTensor(gf, layer, "ffn_norm.weight"),
            .post_attention_norm = findLayerTensor(gf, layer, "post_attention_norm.weight"),
            .ffn_gate = findLayerTensor(gf, layer, "ffn_gate.weight"),
            .ffn_up = findLayerTensor(gf, layer, "ffn_up.weight"),
            .ffn_down = findLayerTensor(gf, layer, "ffn_down.weight"),
            .post_ffw_norm = findLayerTensor(gf, layer, "post_ffw_norm.weight"),
            .pre_ffw_norm_2 = findLayerTensor(gf, layer, "pre_ffw_norm_2.weight"),
            .ffn_gate_inp = findLayerTensor(gf, layer, "ffn_gate_inp.weight"),
            .ffn_gate_exps = findLayerTensor(gf, layer, "ffn_gate_exps.weight"),
            .ffn_up_exps = findLayerTensor(gf, layer, "ffn_up_exps.weight"),
            .ffn_down_exps = findLayerTensor(gf, layer, "ffn_down_exps.weight"),
            .ffn_gate_shexp = findLayerTensor(gf, layer, "ffn_gate_shexp.weight"),
            .ffn_up_shexp = findLayerTensor(gf, layer, "ffn_up_shexp.weight"),
            .ffn_down_shexp = findLayerTensor(gf, layer, "ffn_down_shexp.weight"),
            .ffn_gate_inp_shexp = findLayerTensor(gf, layer, "ffn_gate_inp_shexp.weight"),
            .ffn_gate_up_exps = findLayerTensor(gf, layer, "ffn_gate_up_exps.weight"),
            .ffn_post_norm_1 = findLayerTensor(gf, layer, "post_ffw_norm_1.weight"),
            .ffn_post_norm_2 = findLayerTensor(gf, layer, "post_ffw_norm_2.weight"),
            .ffn_gate_inp_scale = findLayerTensor(gf, layer, "ffn_gate_inp.scale"),
            .attn_qkv = findLayerTensor(gf, layer, "attn_qkv.weight"),
            .ssm_alpha = findLayerTensor(gf, layer, "ssm_alpha.weight"),
            .ssm_beta = findLayerTensor(gf, layer, "ssm_beta.weight"),
            .ssm_conv1d = findLayerTensor(gf, layer, "ssm_conv1d.weight"),
            .ssm_out = findLayerTensor(gf, layer, "ssm_out.weight"),
            .ssm_dt_bias = findLayerTensor(gf, layer, "ssm_dt.bias"),
            .ssm_a = findLayerTensor(gf, layer, "ssm_a"),
            .ssm_norm = findLayerTensor(gf, layer, "ssm_norm.weight"),
            .attn_sinks = findLayerTensor(gf, layer, "attn_sinks.weight"),
        };
    }
    return layers;
}

fn findLayerTensor(gf: *const gguf.GGUFFile, layer: u32, name: []const u8) ?gguf.TensorInfo {
    var buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ layer, name }) catch return null;
    return if (gf.findTensor(key)) |t| t.* else null;
}

fn buildRopeInvFreqs(
    gf: *const gguf.GGUFFile,
    allocator: std.mem.Allocator,
    arch: []const u8,
    cfg: CpuModelConfig,
    freq_base: f32,
    rope_dim_override: u32,
) ![]f32 {
    const rope_dim = if (rope_dim_override > 0) rope_dim_override else cfg.rope_dim;
    const half_rot: usize = @intCast(@max(rope_dim, 2) / 2);
    const freqs = try allocator.alloc(f32, half_rot);
    errdefer allocator.free(freqs);

    var key_buf: [128]u8 = undefined;
    const sections_key = std.fmt.bufPrint(&key_buf, "{s}.rope.dimension_sections", .{arch}) catch "";
    var section_pairs: u32 = 0;
    if (gf.metadata.get(sections_key)) |val| {
        switch (val) {
            .array => |items| {
                for (items) |item| section_pairs += item.asU32() orelse 0;
            },
            else => {},
        }
    }
    const rope_full_dim: f32 = if (section_pairs > 0)
        @floatFromInt(section_pairs * 2)
    else
        @floatFromInt(rope_dim);

    for (freqs, 0..) |*freq, i| {
        const exponent = @as(f32, @floatFromInt(2 * i)) / rope_full_dim;
        freq.* = 1.0 / std.math.pow(f32, freq_base, exponent);
    }
    return freqs;
}

/// Gemma 4 global-attention layers carry a `rope_freqs.weight` tensor that
/// proportionally rescales each inv_freq entry (long-context positional
/// stretching). Vulkan applies the same divisor at forward.zig:1707. The
/// SWA inv_freq table is left untouched.
fn applyRopeFreqFactors(
    mmap_data: []align(std.heap.page_size_min) const u8,
    gf: *const gguf.GGUFFile,
    freqs: []f32,
) void {
    for (gf.tensors.items) |ti| {
        if (!std.mem.eql(u8, ti.name, "rope_freqs.weight")) continue;
        if (ti.type_ != .f32) return;
        const off = gf.tensor_data_offset + ti.offset;
        const n_factors = @min(@as(usize, @intCast(ti.numElements())), freqs.len);
        for (0..n_factors) |k| {
            const factor_off = off + k * @sizeOf(f32);
            if (factor_off + @sizeOf(f32) > mmap_data.len) break;
            const factor: f32 = @as(*const f32, @ptrCast(@alignCast(mmap_data.ptr + factor_off))).*;
            if (factor != 0.0) freqs[k] /= factor;
        }
        log.info("Gemma 4 RoPE freq factors applied to global table ({d} entries)", .{n_factors});
        return;
    }
}

fn buildAttentionSinks(
    mmap_data: []align(std.heap.page_size_min) const u8,
    gf: *const gguf.GGUFFile,
    allocator: std.mem.Allocator,
    layers: []const LayerTensors,
    cfg: CpuModelConfig,
) ![]f32 {
    const count: usize = @as(usize, cfg.n_layers) * @as(usize, cfg.n_heads);
    const sinks = try allocator.alloc(f32, count);
    errdefer allocator.free(sinks);
    for (sinks) |*v| v.* = std.math.nan(f32);

    for (layers, 0..) |lt, layer| {
        const info = lt.attn_sinks orelse continue;
        const n = @min(@as(usize, @intCast(info.numElements())), @as(usize, cfg.n_heads));
        if (n == 0) continue;
        const base = layer * @as(usize, cfg.n_heads);
        try readTensorFlat(tensorDataRaw(mmap_data, gf, info), info.type_, sinks[base..][0..n]);
    }
    return sinks;
}

fn buildSsmConv1dKernelCache(
    mmap_data: []align(std.heap.page_size_min) const u8,
    gf: *const gguf.GGUFFile,
    allocator: std.mem.Allocator,
    layers: []const LayerTensors,
    cfg: CpuModelConfig,
) ![]?[]f32 {
    const cache = try allocator.alloc(?[]f32, layers.len);
    for (cache) |*slot| slot.* = null;
    errdefer freeLayerF32Cache(allocator, cache);

    var cached_layers: usize = 0;
    var cached_elems: usize = 0;
    for (layers, 0..) |lt, i| {
        const info = lt.ssm_conv1d orelse continue;
        const elems: usize = @intCast(info.numElements());
        if (elems == 0) continue;
        const row_major = try allocator.alloc(f32, elems);
        readTensorFlat(tensorDataRaw(mmap_data, gf, info), info.type_, row_major) catch |err| {
            allocator.free(row_major);
            return err;
        };
        if (cfg.ssm_d_conv == 4) {
            const conv_ch = @as(usize, cfg.ssm_d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state);
            if (elems != conv_ch * 4) {
                allocator.free(row_major);
                return error.ShapeMismatch;
            }
            const transposed = try allocator.alloc(f32, elems);
            transposeDConv4Kernel(row_major, transposed, conv_ch);
            allocator.free(row_major);
            cache[i] = transposed;
        } else {
            cache[i] = row_major;
        }
        cached_layers += 1;
        cached_elems += elems;
    }

    if (cached_layers > 0) {
        log.info("M1 host-assisted cached SSM conv1d kernels: {d} layers ({d} MiB f32 resident)", .{
            cached_layers,
            (cached_elems * @sizeOf(f32)) / (1024 * 1024),
        });
    }
    return cache;
}

fn transposeDConv4Kernel(row_major: []const f32, transposed: []f32, conv_ch: usize) void {
    std.debug.assert(row_major.len >= conv_ch * 4);
    std.debug.assert(transposed.len >= conv_ch * 4);
    var ch: usize = 0;
    while (ch < conv_ch) : (ch += 1) {
        const src = ch * 4;
        transposed[ch] = row_major[src];
        transposed[conv_ch + ch] = row_major[src + 1];
        transposed[2 * conv_ch + ch] = row_major[src + 2];
        transposed[3 * conv_ch + ch] = row_major[src + 3];
    }
}

fn freeLayerF32Cache(allocator: std.mem.Allocator, cache: []?[]f32) void {
    for (cache) |maybe_buf| {
        if (maybe_buf) |buf| allocator.free(buf);
    }
    allocator.free(cache);
}

fn buildSmallTensorF32Cache(
    mmap_data: []align(std.heap.page_size_min) const u8,
    gf: *const gguf.GGUFFile,
    allocator: std.mem.Allocator,
    layers: []const LayerTensors,
    final_norm_info: gguf.TensorInfo,
) !std.AutoHashMapUnmanaged(u64, []f32) {
    var cache: std.AutoHashMapUnmanaged(u64, []f32) = .{};
    errdefer {
        var it = cache.valueIterator();
        while (it.next()) |v| allocator.free(v.*);
        cache.deinit(allocator);
    }

    var cached_count: usize = 0;
    var cached_bytes: usize = 0;

    for (layers) |lt| {
        const fields = [_]?gguf.TensorInfo{
            lt.attn_norm,
            lt.attn_q_norm,
            lt.attn_k_norm,
            lt.ffn_norm,
            lt.post_attention_norm,
            lt.post_ffw_norm,
            lt.pre_ffw_norm_2,
            lt.ffn_post_norm_1,
            lt.ffn_post_norm_2,
            lt.ffn_gate_inp_scale,
            lt.ssm_norm,
            lt.ssm_dt_bias,
            lt.ssm_a,
        };
        for (fields) |maybe_info| {
            const info = maybe_info orelse continue;
            if (cache.contains(info.offset)) continue;
            const elems: usize = @intCast(info.numElements());
            if (elems == 0) continue;
            const buf = try allocator.alloc(f32, elems);
            errdefer allocator.free(buf);
            try readTensorFlat(tensorDataRaw(mmap_data, gf, info), info.type_, buf);
            try cache.put(allocator, info.offset, buf);
            cached_count += 1;
            cached_bytes += elems * @sizeOf(f32);
        }
    }
    final_blk: {
        if (cache.contains(final_norm_info.offset)) break :final_blk;
        const elems: usize = @intCast(final_norm_info.numElements());
        if (elems == 0) break :final_blk;
        const buf = allocator.alloc(f32, elems) catch break :final_blk;
        readTensorFlat(tensorDataRaw(mmap_data, gf, final_norm_info), final_norm_info.type_, buf) catch {
            allocator.free(buf);
            break :final_blk;
        };
        cache.put(allocator, final_norm_info.offset, buf) catch {
            allocator.free(buf);
            break :final_blk;
        };
        cached_count += 1;
        cached_bytes += elems * @sizeOf(f32);
    }

    if (cached_count > 0) {
        log.info("M1 host-assisted cached small norm/SSM tensors: {d} tensors ({d} KiB f32 resident)", .{
            cached_count,
            cached_bytes / 1024,
        });
    }
    return cache;
}

/// Counts produced by `Model.validateDecodeGraph` — how many IR nodes were
/// emitted, and how the per-layer mix (full attention, SSM, MoE) breaks down.
/// Useful for asserting the lowered graph matches the model's expected shape.
pub const DecodeGraphSummary = struct {
    nodes: u32,
    layers: u32,
    attention_layers: u32,
    ssm_layers: u32,
    moe_layers: u32,
};

/// Tag identifying which direct-compute shortcut the active tier executed for
/// the current decode step. `none` means the scalar host path retired the
/// token; the other variants name the kernel the GPU ring actually ran (a
/// first-element RMSNorm, an argmax, an argmax composed with that RMSNorm, or
/// a row-range dequantized matvec).
pub const DirectComputeKind = enum {
    none,
    rms_norm_elem0,
    argmax,
    argmax_rms_norm_elem0,
    dmmv_row_range,
};

/// Flags marking which benchmark-only fast-paths influenced the run. The
/// performance suite consults these to decide whether a number is comparable
/// to the reference scalar path or whether a measurement-only shortcut was in
/// effect (top-k forced to zero, LM-head row count capped, decode budget
/// clamped).
pub const BenchmarkShortcutFlags = struct {
    decode_moe_topk_zero: bool = false,
    lm_head_rows_capped: bool = false,
    decode_budget_clamped: bool = false,

    /// True if any benchmark shortcut was applied during the run.
    pub fn any(self: BenchmarkShortcutFlags) bool {
        return self.decode_moe_topk_zero or self.lm_head_rows_capped or self.decode_budget_clamped;
    }
};

/// Output of a `generate` / `generateWithOptions` call: the produced token
/// stream, prefill / decode wall-clock splits, the originally-requested and
/// effective decode budgets, and a set of direct-tier instrumentation
/// counters that report how much of the per-token work was actually retired
/// by the GPU ring versus by the scalar fallback.
/// The token slice is allocator-owned and must be released with `deinit`.
pub const GenerateResult = struct {
    tokens: []u32,
    prefill_ns: u64,
    decode_ns: u64,
    requested_max_tokens: u32,
    effective_max_tokens: u32,
    direct_token_boundary_copies: u32 = 0,
    direct_token_boundary_ib_bytes: u32 = 0,
    direct_token_boundary_last_fence: u64 = 0,
    direct_model_ops: u32 = 0,
    direct_compute_ops: u32 = 0,
    direct_compute_kind: DirectComputeKind = .none,
    consumed_gpu_compute_value: bool = false,
    real_model_slice: bool = false,
    direct_compute_token: u32 = 0,
    consumed_gpu_model_value: bool = false,
    direct_model_value_bits: u32 = 0,
    benchmark_shortcuts: BenchmarkShortcutFlags = .{},

    /// Free the produced token slice and poison the handle.
    /// @param allocator Same allocator that was passed to `generate`.
    pub fn deinit(self: *GenerateResult, allocator: std.mem.Allocator) void {
        allocator.free(self.tokens);
        self.* = undefined;
    }
};

/// Knobs threaded through to `generateWithOptions`. Lets callers opt out of
/// the per-token direct-tier admission validation when they only want the
/// scalar reference numbers.
pub const GenerateOptions = struct {
    /// Validate that the selected direct tier can retire the current
    /// token-boundary COPY_DATA packet. This is a one-shot admission gate; until
    /// a lowered decode op consumes the copied token, per-token validation
    /// submits are a measurement tax rather than model work.
    enable_direct_token_boundary: bool = true,
};

/// Run the full ZINC_RT forward pass against `model` with default
/// `GenerateOptions`: prefill the prompt, validate the per-token decode IR,
/// and decode at most `max_tokens` tokens (stopping early on `eos_token_id`).
/// Falls back to a no-layer smoke tail for models the scalar hybrid path
/// cannot run. Equivalent to `generateWithOptions(..., .{})`.
/// @param model Loaded GGUF model.
/// @param prompt_tokens Tokenised prompt; must be non-empty.
/// @param max_tokens Upper bound on tokens to produce after prefill.
/// @param eos_token_id Stop-token id; honoured by the scalar hybrid path.
/// @param allocator Owns the returned `GenerateResult.tokens`.
/// @returns A `GenerateResult` the caller must release via its `deinit`.
pub fn generate(
    io: std.Io,
    model: *const Model,
    prompt_tokens: []const u32,
    max_tokens: u32,
    eos_token_id: u32,
    allocator: std.mem.Allocator,
) !GenerateResult {
    return generateWithOptions(io, model, prompt_tokens, max_tokens, eos_token_id, allocator, .{});
}

/// Full ZINC_RT forward pass with caller-supplied `GenerateOptions`. Logs the
/// validated decode-graph summary, then picks between the scalar hybrid
/// MoE+SSM path (for Qwen 3.6-shaped models whose tensors fully resolve) and
/// the no-layer smoke tail (everything else). The scalar hybrid path is the
/// one that consumes the GPU ring's direct-compute results; the smoke tail is
/// CPU-only and only retires the embedding / final-norm boundary.
/// @param model Loaded GGUF model.
/// @param prompt_tokens Tokenised prompt; must be non-empty.
/// @param max_tokens Upper bound on tokens to produce after prefill.
/// @param eos_token_id Stop-token id.
/// @param allocator Owns the returned `GenerateResult.tokens` and the
/// throwaway decode IR.
/// @param options Per-call configuration; see `GenerateOptions`.
/// @returns A `GenerateResult` the caller must release via its `deinit`.
pub fn generateWithOptions(
    io: std.Io,
    model: *const Model,
    prompt_tokens: []const u32,
    max_tokens: u32,
    eos_token_id: u32,
    allocator: std.mem.Allocator,
    options: GenerateOptions,
) !GenerateResult {
    if (prompt_tokens.len == 0) return error.EmptyPrompt;

    const graph_summary = try model.validateDecodeGraph(allocator);
    log.info("M1 decode IR verified: nodes={d} layers={d} attn={d} ssm={d} moe={d}", .{
        graph_summary.nodes,
        graph_summary.layers,
        graph_summary.attention_layers,
        graph_summary.ssm_layers,
        graph_summary.moe_layers,
    });

    if (model.canRunScalarHybrid()) {
        log.info("M1 host-assisted full-forward path enabled for hybrid MoE+SSM model", .{});
        return generateScalarHybrid(io, model, prompt_tokens, max_tokens, eos_token_id, allocator, options);
    }

    if (model.canRunScalarDense()) {
        log.info("M1 host-assisted full-forward path enabled for dense attention model", .{});
        return generateScalarDense(io, model, prompt_tokens, max_tokens, eos_token_id, allocator, options);
    }

    log.info("M1 host-assisted full-forward path unavailable; falling back to no-layer smoke tail", .{});
    return generateNoLayer(model, prompt_tokens, max_tokens, eos_token_id, allocator);
}

fn generateNoLayer(
    model: *const Model,
    prompt_tokens: []const u32,
    max_tokens: u32,
    eos_token_id: u32,
    allocator: std.mem.Allocator,
) !GenerateResult {
    const effective_max_tokens = @min(max_tokens, m0MaxDecodeTokens());
    var generated: std.ArrayList(u32) = .empty;
    errdefer generated.deinit(allocator);

    const hidden = try allocator.alloc(f32, model.hidden_dim);
    defer allocator.free(hidden);
    const norm = try allocator.alloc(f32, model.hidden_dim);
    defer allocator.free(norm);
    const row_scratch = try allocator.alloc(f32, model.hidden_dim);
    defer allocator.free(row_scratch);
    const logits = try allocator.alloc(f32, model.vocab_size);
    defer allocator.free(logits);
    var next_token: u32 = 0;

    var rt = cpu_ring.CpuRing.init();
    defer rt.deinit();

    const prefill_start = nanoTimestamp();
    try evalToken(model, &rt, prompt_tokens[prompt_tokens.len - 1], hidden, norm, row_scratch, logits, &next_token);
    const prefill_end = nanoTimestamp();

    const decode_start = nanoTimestamp();
    if (effective_max_tokens > 0) {
        try generated.append(allocator, next_token);
    }

    while (generated.items.len < effective_max_tokens and next_token != eos_token_id) {
        try evalToken(model, &rt, next_token, hidden, norm, row_scratch, logits, &next_token);
        try generated.append(allocator, next_token);
    }
    const decode_end = nanoTimestamp();

    if (effective_max_tokens < max_tokens) {
        log.info("M1 host-assisted no-layer path clamped decode budget from {d} to {d} tokens", .{
            max_tokens,
            effective_max_tokens,
        });
    }

    return .{
        .tokens = try generated.toOwnedSlice(allocator),
        .prefill_ns = elapsedNs(prefill_start, prefill_end),
        .decode_ns = elapsedNs(decode_start, decode_end),
        .requested_max_tokens = max_tokens,
        .effective_max_tokens = effective_max_tokens,
        .direct_token_boundary_copies = 0,
        .benchmark_shortcuts = .{
            .decode_budget_clamped = effective_max_tokens < max_tokens,
        },
    };
}

const ScalarDecodeState = struct {
    allocator: std.mem.Allocator,
    max_seq: u32,
    hidden: []f32,
    norm: []f32,
    ffn_norm: []f32,
    q_full: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attn_out: []f32,
    branch: []f32,
    qkv: []f32,
    z: []f32,
    alpha: []f32,
    beta: []f32,
    conv_out: []f32,
    ssm_out: []f32,
    dt_bias: []f32,
    ssm_a: []f32,
    ssm_norm_w: []f32,
    router_logits: []f32,
    gate: []f32,
    up: []f32,
    swiglu: []f32,
    down: []f32,
    /// Gemma 4 MoE scratch: holds the shared-MLP output while the routed
    /// experts overwrite gate/up/swiglu/down. Sized hidden_dim.
    mlp_save: []f32,
    moe_worker_scratch: []f32,
    moe_worker_gate: []f32,
    moe_worker_up: []f32,
    moe_worker_swiglu: []f32,
    moe_worker_down: []f32,
    logits: []f32,
    row_scratch: []f32,
    scores: []f32,
    probs: []f32,
    expert_ids: []u32,
    expert_weights: []f32,
    kv_k: []f32,
    kv_v: []f32,
    ssm_conv_states: []f32,
    ssm_states: []f32,
    moe_expert_workers: usize,
    moe_topk_active: u32,
    decode_phase: bool = false,
    direct_router_row_range_done: bool = false,
    pool: ?*thread_pool.Pool = null,
    fast_pool: ?*zinc_rt.fast_pool.FastPool = null,

    fn init(allocator: std.mem.Allocator, model: *const Model, max_seq: u32) !ScalarDecodeState {
        const cfg = model.config;
        const conv_ch = cfg.ssm_d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state;
        const max_inter = @max(cfg.intermediate_dim, cfg.shared_expert_intermediate_dim);
        const max_work = @max(max_inter, cfg.q_dim);
        const max_vec = @max(@max(@max(cfg.hidden_dim, cfg.q_dim * 2), @max(cfg.kv_dim, conv_ch)), @max(max_inter, cfg.ssm_d_inner));
        const scratch_len = @max(max_vec, conv_ch * @max(cfg.ssm_d_conv, 1));
        const moe_worker_slots: usize = @intCast(@max(cfg.n_experts_used, 1));
        const moe_worker_inter_stride: usize = @intCast(max_inter);
        const moe_worker_scratch_stride: usize = @intCast(@max(cfg.hidden_dim, max_inter));
        const moe_worker_down_stride: usize = @intCast(cfg.hidden_dim);
        const conv_state_elems = @as(usize, cfg.n_layers) * @as(usize, @max(cfg.ssm_d_conv, 1) - 1) * @as(usize, conv_ch);
        const head_v_dim = if (cfg.ssm_dt_rank > 0) cfg.ssm_d_inner / cfg.ssm_dt_rank else 0;
        const ssm_state_elems = @as(usize, cfg.n_layers) * @as(usize, cfg.ssm_dt_rank) * @as(usize, head_v_dim) * @as(usize, head_v_dim);
        const kv_elems = @as(usize, cfg.n_layers) * @as(usize, max_seq) * @as(usize, cfg.kv_dim);

        const state = ScalarDecodeState{
            .allocator = allocator,
            .max_seq = max_seq,
            .hidden = try allocator.alloc(f32, cfg.hidden_dim),
            .norm = try allocator.alloc(f32, cfg.hidden_dim),
            .ffn_norm = try allocator.alloc(f32, cfg.hidden_dim),
            .q_full = try allocator.alloc(f32, @max(cfg.q_dim * 2, cfg.q_dim)),
            .q = try allocator.alloc(f32, cfg.q_dim),
            .k = try allocator.alloc(f32, cfg.kv_dim),
            .v = try allocator.alloc(f32, cfg.kv_dim),
            .attn_out = try allocator.alloc(f32, cfg.q_dim),
            .branch = try allocator.alloc(f32, cfg.hidden_dim),
            .qkv = try allocator.alloc(f32, conv_ch),
            .z = try allocator.alloc(f32, cfg.ssm_d_inner),
            .alpha = try allocator.alloc(f32, cfg.ssm_dt_rank),
            .beta = try allocator.alloc(f32, cfg.ssm_dt_rank),
            .conv_out = try allocator.alloc(f32, conv_ch),
            .ssm_out = try allocator.alloc(f32, cfg.ssm_d_inner),
            .dt_bias = try allocator.alloc(f32, cfg.ssm_dt_rank),
            .ssm_a = try allocator.alloc(f32, cfg.ssm_dt_rank),
            .ssm_norm_w = try allocator.alloc(f32, @max(cfg.ssm_d_inner, cfg.ssm_d_state)),
            .router_logits = try allocator.alloc(f32, cfg.n_experts),
            .gate = try allocator.alloc(f32, max_work),
            .up = try allocator.alloc(f32, max_work),
            .swiglu = try allocator.alloc(f32, max_work),
            .down = try allocator.alloc(f32, cfg.hidden_dim),
            .mlp_save = try allocator.alloc(f32, cfg.hidden_dim),
            .moe_worker_scratch = try allocator.alloc(f32, moe_worker_slots * moe_worker_scratch_stride),
            .moe_worker_gate = try allocator.alloc(f32, moe_worker_slots * moe_worker_inter_stride),
            .moe_worker_up = try allocator.alloc(f32, moe_worker_slots * moe_worker_inter_stride),
            .moe_worker_swiglu = try allocator.alloc(f32, moe_worker_slots * moe_worker_inter_stride),
            .moe_worker_down = try allocator.alloc(f32, moe_worker_slots * moe_worker_down_stride),
            .logits = try allocator.alloc(f32, cfg.vocab_size),
            .row_scratch = try allocator.alloc(f32, scratch_len),
            .scores = try allocator.alloc(f32, max_seq),
            .probs = try allocator.alloc(f32, max_seq),
            .expert_ids = try allocator.alloc(u32, @max(cfg.n_experts_used, 1)),
            .expert_weights = try allocator.alloc(f32, @max(cfg.n_experts_used, 1)),
            .kv_k = try allocator.alloc(f32, kv_elems),
            .kv_v = try allocator.alloc(f32, kv_elems),
            .ssm_conv_states = try allocator.alloc(f32, conv_state_elems),
            .ssm_states = try allocator.alloc(f32, ssm_state_elems),
            .moe_expert_workers = moeExpertWorkerCount(model.effectiveMoeTopK(), std.Thread.getCpuCount() catch 1),
            .moe_topk_active = model.effectiveMoeTopK(),
        };
        @memset(state.kv_k, 0);
        @memset(state.kv_v, 0);
        @memset(state.ssm_conv_states, 0);
        @memset(state.ssm_states, 0);
        return state;
    }

    fn deinit(self: *ScalarDecodeState) void {
        const a = self.allocator;
        a.free(self.ssm_states);
        a.free(self.ssm_conv_states);
        a.free(self.kv_v);
        a.free(self.kv_k);
        a.free(self.expert_weights);
        a.free(self.expert_ids);
        a.free(self.probs);
        a.free(self.scores);
        a.free(self.row_scratch);
        a.free(self.logits);
        a.free(self.moe_worker_down);
        a.free(self.moe_worker_swiglu);
        a.free(self.moe_worker_up);
        a.free(self.moe_worker_gate);
        a.free(self.moe_worker_scratch);
        a.free(self.mlp_save);
        a.free(self.down);
        a.free(self.swiglu);
        a.free(self.up);
        a.free(self.gate);
        a.free(self.router_logits);
        a.free(self.ssm_norm_w);
        a.free(self.ssm_a);
        a.free(self.dt_bias);
        a.free(self.ssm_out);
        a.free(self.conv_out);
        a.free(self.beta);
        a.free(self.alpha);
        a.free(self.z);
        a.free(self.qkv);
        a.free(self.branch);
        a.free(self.attn_out);
        a.free(self.v);
        a.free(self.k);
        a.free(self.q);
        a.free(self.q_full);
        a.free(self.ffn_norm);
        a.free(self.norm);
        a.free(self.hidden);
        self.* = undefined;
    }

    fn kvOffset(self: *const ScalarDecodeState, cfg: CpuModelConfig, layer: u32, pos: u32) usize {
        return (@as(usize, layer) * @as(usize, self.max_seq) + @as(usize, pos)) * @as(usize, cfg.kv_dim);
    }

    fn convStateForLayer(self: *ScalarDecodeState, cfg: CpuModelConfig, layer: u32) []f32 {
        const conv_ch = cfg.ssm_d_inner + 2 * cfg.ssm_n_group * cfg.ssm_d_state;
        const len = (@max(cfg.ssm_d_conv, 1) - 1) * conv_ch;
        const off = @as(usize, layer) * @as(usize, len);
        return self.ssm_conv_states[off..][0..len];
    }

    fn ssmStateForLayer(self: *ScalarDecodeState, cfg: CpuModelConfig, layer: u32) []f32 {
        const head_v_dim = cfg.ssm_d_inner / cfg.ssm_dt_rank;
        const len = cfg.ssm_dt_rank * head_v_dim * head_v_dim;
        const off = @as(usize, layer) * @as(usize, len);
        return self.ssm_states[off..][0..len];
    }

    fn setMoeTopK(self: *ScalarDecodeState, n_used: u32, cpu_count: usize) void {
        self.moe_topk_active = n_used;
        self.moe_expert_workers = moeExpertWorkerCount(n_used, cpu_count);
    }
};

const DirectComputeTracking = struct {
    boundary: *zinc_rt.cs.TokenBoundary,
    ops: *u32,
    kind: *DirectComputeKind,
    consumed: *bool,
    real_model_slice: *bool,
};

fn generateScalarHybrid(
    io: std.Io,
    model: *const Model,
    prompt_tokens: []const u32,
    max_tokens: u32,
    eos_token_id: u32,
    allocator: std.mem.Allocator,
    options: GenerateOptions,
) !GenerateResult {
    const effective_max_tokens = @min(max_tokens, m0MaxDecodeTokens());
    const max_seq: u32 = @intCast(prompt_tokens.len + effective_max_tokens + 1);
    var state = try ScalarDecodeState.init(allocator, model, max_seq);
    defer state.deinit();

    // Persistent decode worker pool: every executed matvec/MoE fan-out re-uses
    // these threads instead of spawning a fresh thread per op (~1k spawns/tok
    // otherwise — the dominant non-compute cost in the T-CPU path).
    var decode_pool: thread_pool.Pool = undefined;
    var decode_pool_ready = false;
    const decode_pool_workers = decodeWorkerThreadCount(io);
    const cpu_count = std.Thread.getCpuCount() catch 1;
    if (decode_pool_workers > 1) {
        if (decode_pool.init(.{ .allocator = std.heap.smp_allocator, .n_jobs = decode_pool_workers })) {
            decode_pool_ready = true;
        } else |err| {
            log.warn("M1 host-assisted decode worker pool unavailable ({s}); falling back to per-op threads", .{@errorName(err)});
        }
    }
    defer if (decode_pool_ready) decode_pool.deinit();
    state.pool = if (decode_pool_ready) &decode_pool else null;
    if (decode_pool_ready) {
        log.info("M1 host-assisted persistent decode worker pool enabled: {d} workers (matvec + MoE fan-out)", .{decode_pool_workers});
    }

    // Atomic-counter-based fan-out pool used for the matvec direct dispatch
    // path. Each per-token decode issues ~200 short fan-outs to 3-4 workers;
    // thread_pool.Pool's mutex/condvar/heap-alloc closure machinery shows up
    // as several ms of overhead on the 9800X3D. The FastPool replaces just
    // that hot path with persistent spin-workers + per-slot atomic seq
    // synchronisation. Init failure (or opt-out via ZINC_RT_FAST_POOL=0) is
    // non-fatal: matvec dispatch falls back to the std pool above.
    var fast_pool: zinc_rt.fast_pool.FastPool = undefined;
    var fast_pool_ready = false;
    const fast_pool_enabled = blk: {
        const raw = getenv("ZINC_RT_FAST_POOL") orelse break :blk true;
        if (raw.len == 0) break :blk true;
        if (std.mem.eql(u8, raw, "0")) break :blk false;
        if (std.mem.eql(u8, raw, "false")) break :blk false;
        break :blk true;
    };
    if (decode_pool_workers > 1 and fast_pool_enabled) {
        if (fast_pool.init(std.heap.smp_allocator, decode_pool_workers)) {
            fast_pool_ready = true;
        } else |err| {
            log.warn("M1 host-assisted FastPool unavailable ({s}); matvec uses std pool", .{@errorName(err)});
        }
    }
    defer if (fast_pool_ready) fast_pool.deinit();
    state.fast_pool = if (fast_pool_ready) &fast_pool else null;
    if (fast_pool_ready) {
        log.info("M1 host-assisted FastPool matvec dispatch enabled: {d} workers (atomic-counter fan-out)", .{decode_pool_workers});
    }
    matvec_fast_pool = state.fast_pool;
    defer matvec_fast_pool = null;

    if (state.moe_expert_workers > 1) {
        log.info("M1 host-assisted parallel MoE expert branches enabled: topk={d} workers={d}", .{
            state.moe_topk_active,
            state.moe_expert_workers,
        });
    }

    var token_boundary_storage: zinc_rt.cs.TokenBoundary = undefined;
    var token_boundary: ?*zinc_rt.cs.TokenBoundary = null;
    if (options.enable_direct_token_boundary) {
        if (zinc_rt.cs.TokenBoundary.initDefault(io)) |boundary| {
            token_boundary_storage = boundary;
            token_boundary = &token_boundary_storage;
            log.info("M1 AMDGPU CS direct token boundary validating once: PM4 COPY_DATA token_id -> embedding input", .{});
        } else |err| {
            log.warn("M1 AMDGPU CS direct token boundary unavailable ({s}); scalar token ids remain host-provided", .{@errorName(err)});
        }
    } else {
        log.info("M1 AMDGPU CS direct token boundary disabled for selected tier", .{});
    }
    defer if (token_boundary != null) token_boundary_storage.deinit();
    var direct_token_boundary_copies: u32 = 0;
    var direct_prompt0_token: ?u32 = null;
    if (prompt_tokens.len > 0) {
        direct_prompt0_token = try directBoundaryToken(token_boundary, prompt_tokens[0], &direct_token_boundary_copies);
    }
    var direct_model_ops: u32 = 0;
    const direct_final_norm_weight0 = try directModelFinalNormWeight0(token_boundary, model, &direct_model_ops);
    const direct_model_value_bits: u32 = if (direct_final_norm_weight0) |w| @bitCast(w) else 0;
    if (direct_model_ops > 0) {
        log.info("M1 AMDGPU CS direct model value enabled: output_norm.weight[0] COPY_DATA bits=0x{x}", .{direct_model_value_bits});
    }
    var consumed_gpu_model_value = false;
    var direct_compute_ops: u32 = 0;
    var direct_compute_kind: DirectComputeKind = .none;
    var consumed_gpu_compute_value = false;
    var real_model_slice = false;
    var direct_compute_token: u32 = 0;

    var generated: std.ArrayList(u32) = .empty;
    errdefer generated.deinit(allocator);

    var next_token: u32 = 0;
    const prefill_start = nanoTimestamp();
    for (prompt_tokens, 0..) |token, pos| {
        const eval_token = if (pos == 0) direct_prompt0_token orelse token else token;
        var selection: ArgmaxTop2Result = .{};
        const selection_out: ?*ArgmaxTop2Result = if (pos + 1 == prompt_tokens.len) &selection else null;
        const direct_compute_tracking: ?DirectComputeTracking = if (pos + 1 == prompt_tokens.len and token_boundary != null)
            .{
                .boundary = token_boundary.?,
                .ops = &direct_compute_ops,
                .kind = &direct_compute_kind,
                .consumed = &consumed_gpu_compute_value,
                .real_model_slice = &real_model_slice,
            }
        else
            null;
        try scalarEvalToken(
            model,
            &state,
            eval_token,
            @intCast(pos),
            &next_token,
            direct_final_norm_weight0,
            &consumed_gpu_model_value,
            selection_out,
            direct_compute_tracking,
        );
        if (selection_out != null) {
            const gpu_token = directComputeArgmaxTop2(token_boundary, selection) catch |err| blk: {
                log.warn("M1 AMDGPU CS direct argmax compute unavailable ({s}); first token remains host-selected", .{@errorName(err)});
                break :blk null;
            };
            if (gpu_token) |token_from_gpu| {
                next_token = token_from_gpu;
                direct_compute_ops += 1;
                mergeDirectComputeKind(&direct_compute_kind, .argmax);
                consumed_gpu_compute_value = true;
                direct_compute_token = token_from_gpu;
                log.info("M1 AMDGPU CS direct compute consumed: direct_compute_ops={d} direct_compute_kind=argmax variant=top2 token={d} best=({d},{d:.4}) second=({d},{d:.4})", .{
                    direct_compute_ops,
                    token_from_gpu,
                    selection.best.index,
                    selection.best.value,
                    selection.second.index,
                    selection.second.value,
                });
            }
        }
    }
    const prefill_end = nanoTimestamp();

    const decode_start = nanoTimestamp();
    if (effective_max_tokens > 0) try generated.append(allocator, next_token);
    state.decode_phase = true;
    const decode_topk = model.effectiveDecodeMoeTopK();
    const decode_moe_topk_zero = decode_topk == 0 and state.moe_topk_active > 0;
    if (decode_topk != state.moe_topk_active) {
        const prefill_topk = state.moe_topk_active;
        state.setMoeTopK(decode_topk, cpu_count);
        log.info("M1 host-assisted decode MoE top-k lowered to {d} after prefill (prefill top-k={d})", .{
            decode_topk,
            prefill_topk,
        });
    }
    var position: u32 = @intCast(prompt_tokens.len);
    while (generated.items.len < effective_max_tokens and next_token != eos_token_id) : (position += 1) {
        try scalarEvalToken(model, &state, next_token, position, &next_token, direct_final_norm_weight0, &consumed_gpu_model_value, null, null);
        try generated.append(allocator, next_token);
    }
    const decode_end = nanoTimestamp();

    if (effective_max_tokens < max_tokens) {
        log.info("M1 host-assisted path clamped decode budget from {d} to {d} tokens", .{
            max_tokens,
            effective_max_tokens,
        });
    }

    return .{
        .tokens = try generated.toOwnedSlice(allocator),
        .prefill_ns = elapsedNs(prefill_start, prefill_end),
        .decode_ns = elapsedNs(decode_start, decode_end),
        .requested_max_tokens = max_tokens,
        .effective_max_tokens = effective_max_tokens,
        .direct_token_boundary_copies = direct_token_boundary_copies,
        .direct_token_boundary_ib_bytes = if (token_boundary != null) token_boundary_storage.last_ib_bytes else 0,
        .direct_token_boundary_last_fence = if (token_boundary != null) token_boundary_storage.last_fence_handle else 0,
        .direct_model_ops = direct_model_ops,
        .direct_compute_ops = direct_compute_ops,
        .direct_compute_kind = direct_compute_kind,
        .consumed_gpu_compute_value = consumed_gpu_compute_value,
        .real_model_slice = real_model_slice,
        .direct_compute_token = direct_compute_token,
        .consumed_gpu_model_value = consumed_gpu_model_value,
        .direct_model_value_bits = direct_model_value_bits,
        .benchmark_shortcuts = .{
            .decode_moe_topk_zero = decode_moe_topk_zero,
            .lm_head_rows_capped = model.effectiveLmHeadRows(true) < model.config.vocab_size,
            .decode_budget_clamped = effective_max_tokens < max_tokens,
        },
    };
}

fn generateScalarDense(
    io: std.Io,
    model: *const Model,
    prompt_tokens: []const u32,
    max_tokens: u32,
    eos_token_id: u32,
    allocator: std.mem.Allocator,
    options: GenerateOptions,
) !GenerateResult {
    _ = options;
    const effective_max_tokens = @min(max_tokens, m0MaxDecodeTokens());
    const max_seq: u32 = @intCast(prompt_tokens.len + effective_max_tokens + 1);
    var state = try ScalarDecodeState.init(allocator, model, max_seq);
    defer state.deinit();

    var decode_pool: thread_pool.Pool = undefined;
    var decode_pool_ready = false;
    const decode_pool_workers = decodeWorkerThreadCount(io);
    if (decode_pool_workers > 1) {
        if (decode_pool.init(.{ .allocator = std.heap.smp_allocator, .n_jobs = decode_pool_workers })) {
            decode_pool_ready = true;
        } else |err| {
            log.warn("M1 host-assisted dense decode worker pool unavailable ({s}); falling back to per-op threads", .{@errorName(err)});
        }
    }
    defer if (decode_pool_ready) decode_pool.deinit();
    state.pool = if (decode_pool_ready) &decode_pool else null;
    if (decode_pool_ready) {
        log.info("M1 host-assisted dense decode worker pool enabled: {d} workers", .{decode_pool_workers});
    }

    var fast_pool: zinc_rt.fast_pool.FastPool = undefined;
    var fast_pool_ready = false;
    if (decode_pool_workers > 1) {
        if (fast_pool.init(std.heap.smp_allocator, decode_pool_workers)) {
            fast_pool_ready = true;
        } else |err| {
            log.warn("M1 host-assisted dense FastPool unavailable ({s})", .{@errorName(err)});
        }
    }
    defer if (fast_pool_ready) fast_pool.deinit();
    state.fast_pool = if (fast_pool_ready) &fast_pool else null;
    matvec_fast_pool = state.fast_pool;
    defer matvec_fast_pool = null;

    var generated: std.ArrayList(u32) = .empty;
    errdefer generated.deinit(allocator);

    var next_token: u32 = 0;
    const prefill_start = nanoTimestamp();
    for (prompt_tokens, 0..) |token, pos| {
        const need_logits = pos + 1 == prompt_tokens.len;
        try scalarEvalTokenDense(model, &state, token, @intCast(pos), &next_token, need_logits);
    }
    const prefill_end = nanoTimestamp();

    const decode_start = nanoTimestamp();
    if (effective_max_tokens > 0) try generated.append(allocator, next_token);
    state.decode_phase = true;
    var position: u32 = @intCast(prompt_tokens.len);
    while (generated.items.len < effective_max_tokens and next_token != eos_token_id) : (position += 1) {
        try scalarEvalTokenDense(model, &state, next_token, position, &next_token, true);
        try generated.append(allocator, next_token);
    }
    const decode_end = nanoTimestamp();

    if (effective_max_tokens < max_tokens) {
        log.info("M1 host-assisted dense path clamped decode budget from {d} to {d} tokens", .{
            max_tokens,
            effective_max_tokens,
        });
    }

    return .{
        .tokens = try generated.toOwnedSlice(allocator),
        .prefill_ns = elapsedNs(prefill_start, prefill_end),
        .decode_ns = elapsedNs(decode_start, decode_end),
        .requested_max_tokens = max_tokens,
        .effective_max_tokens = effective_max_tokens,
        .direct_token_boundary_copies = 0,
        .direct_token_boundary_ib_bytes = 0,
        .direct_token_boundary_last_fence = 0,
        .direct_model_ops = 0,
        .direct_compute_ops = 0,
        .direct_compute_kind = .none,
        .consumed_gpu_compute_value = false,
        .real_model_slice = false,
        .direct_compute_token = 0,
        .consumed_gpu_model_value = false,
        .direct_model_value_bits = 0,
        .benchmark_shortcuts = .{
            .decode_moe_topk_zero = false,
            .lm_head_rows_capped = model.effectiveLmHeadRows(true) < model.config.vocab_size,
            .decode_budget_clamped = effective_max_tokens < max_tokens,
        },
    };
}

fn scalarEvalTokenDense(
    model: *const Model,
    state: *ScalarDecodeState,
    token_id: u32,
    position: u32,
    next_token: *u32,
    need_logits: bool,
) !void {
    const cfg = model.config;
    const safe_id = @min(token_id, cfg.vocab_size -| 1);
    try dequant.row(model.tensorData(model.embed_info), safe_id, cfg.hidden_dim, model.embed_info.type_, state.hidden);
    if (cfg.is_gemma) {
        const scale: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(cfg.hidden_dim))));
        for (state.hidden) |*v| v.* *= scale;
    }

    for (model.layer_tensors, 0..) |lt, li| {
        const layer: u32 = @intCast(li);
        try rmsNormTensor(model, lt.attn_norm.?, state.hidden, state.norm, state.row_scratch);
        try runAttentionLayer(model, state, lt, layer, position);
        try runDenseFfnLayer(model, state, lt);
        if (model.layer_output_scales) |scales| {
            const scale = scales[li];
            if (scale != 1.0) {
                for (state.hidden) |*v| v.* *= scale;
            }
        }
    }

    if (!need_logits) {
        next_token.* = 0;
        return;
    }

    try rmsNormTensor(model, model.final_norm_info, state.hidden, state.norm, state.row_scratch);
    const lm_head_rows = model.effectiveLmHeadRows(state.decode_phase);
    if (model.lm_head_q4_0) |q40| {
        next_token.* = try argmaxMatvecRaw(state.pool, q40, .q4_0, state.norm, lm_head_rows, state.row_scratch);
    } else {
        const lm_head = model.requantOrRaw(model.lm_head_info);
        if (canDotDirect(lm_head.type_, @intCast(state.norm.len))) {
            next_token.* = try argmaxMatvecRaw(state.pool, lm_head.raw, lm_head.type_, state.norm, lm_head_rows, state.row_scratch);
        } else {
            try matvecRaw(state.pool, lm_head.raw, lm_head.type_, state.norm, lm_head_rows, state.row_scratch, state.logits);
            next_token.* = argmaxSlice(state.logits[0..lm_head_rows]);
        }
    }
}

fn runDenseFfnLayer(
    model: *const Model,
    state: *ScalarDecodeState,
    lt: LayerTensors,
) !void {
    const cfg = model.config;
    const ffn_norm = lt.ffn_norm orelse lt.post_attention_norm orelse return error.TensorNotFound;
    try rmsNormTensor(model, ffn_norm, state.hidden, state.ffn_norm, state.row_scratch);

    const gate_t = lt.ffn_gate.?;
    const up_t = lt.ffn_up.?;
    const down_t = lt.ffn_down.?;
    const inter: u32 = @intCast(gate_t.numElements() / cfg.hidden_dim);

    // Fuse gate + up matvecs into one pool dispatch.
    try matvecFusedTensors(state.pool, model, &[_]FusedPart{
        .{ .info = gate_t, .rows = inter, .out = state.gate[0..inter] },
        .{ .info = up_t, .rows = inter, .out = state.up[0..inter] },
    }, state.ffn_norm, state.row_scratch);

    if (cfg.is_gemma) {
        geluGate(state.gate[0..inter], state.up[0..inter], state.swiglu[0..inter]);
    } else {
        swiglu(state.gate[0..inter], state.up[0..inter], state.swiglu[0..inter]);
    }

    // down: project intermediate back to hidden and add to the layer residual.
    try matvecTensor(state.pool, model, down_t, state.swiglu[0..inter], cfg.hidden_dim, state.row_scratch, state.down);
    // Gemma applies post_ffw_norm to the FFN output before merging into the
    // residual stream.
    if (cfg.is_gemma) {
        if (lt.post_ffw_norm) |pfn| {
            try rmsNormTensor(model, pfn, state.down, state.down, state.row_scratch);
        }
    }
    for (state.hidden, state.down) |*h, v| h.* += v;
}

fn directBoundaryToken(
    boundary: ?*zinc_rt.cs.TokenBoundary,
    token_id: u32,
    copies: *u32,
) !u32 {
    const active = boundary orelse return token_id;
    const copied = try active.produceToken(token_id);
    if (copied != token_id) return error.DirectTokenBoundaryMismatch;
    copies.* += 1;
    return copied;
}

fn directModelFinalNormWeight0(
    boundary: ?*zinc_rt.cs.TokenBoundary,
    model: *const Model,
    ops: *u32,
) !?f32 {
    const active = boundary orelse return null;
    if (model.final_norm_weight.len == 0) return null;
    const cpu_value = model.final_norm_weight[0];
    const cpu_bits: u32 = @bitCast(cpu_value);
    const copied_bits = try active.copyU32(cpu_bits);
    if (copied_bits != cpu_bits) return error.DirectModelValueMismatch;
    ops.* += 1;
    return @bitCast(copied_bits);
}

fn directComputeArgmaxTop2(
    boundary: ?*zinc_rt.cs.TokenBoundary,
    selection: ArgmaxTop2Result,
) !?u32 {
    const active = boundary orelse return null;
    const selected = try active.argmaxTop2(
        selection.best.index,
        selection.best.value,
        selection.second.index,
        selection.second.value,
    );
    if (selected != selection.best.index) return error.DirectArgmaxTop2Mismatch;
    return selected;
}

fn mergeDirectComputeKind(current: *DirectComputeKind, added: DirectComputeKind) void {
    if (added == .dmmv_row_range) {
        current.* = .dmmv_row_range;
        return;
    }
    current.* = switch (current.*) {
        .none => added,
        .rms_norm_elem0 => switch (added) {
            .argmax => .argmax_rms_norm_elem0,
            .none, .rms_norm_elem0 => .rms_norm_elem0,
            .argmax_rms_norm_elem0 => .argmax_rms_norm_elem0,
            .dmmv_row_range => .dmmv_row_range,
        },
        .argmax => switch (added) {
            .rms_norm_elem0 => .argmax_rms_norm_elem0,
            .none, .argmax => .argmax,
            .argmax_rms_norm_elem0 => .argmax_rms_norm_elem0,
            .dmmv_row_range => .dmmv_row_range,
        },
        .argmax_rms_norm_elem0 => switch (added) {
            .dmmv_row_range => .dmmv_row_range,
            else => .argmax_rms_norm_elem0,
        },
        .dmmv_row_range => .dmmv_row_range,
    };
}

fn scalarEvalToken(
    model: *const Model,
    state: *ScalarDecodeState,
    token_id: u32,
    position: u32,
    next_token: *u32,
    direct_final_norm_weight0: ?f32,
    consumed_gpu_model_value: *bool,
    selection_out: ?*ArgmaxTop2Result,
    direct_compute_tracking: ?DirectComputeTracking,
) !void {
    const cfg = model.config;
    const safe_id = @min(token_id, cfg.vocab_size -| 1);
    try dequant.row(model.tensorData(model.embed_info), safe_id, cfg.hidden_dim, model.embed_info.type_, state.hidden);
    // Gemma scales embeddings by sqrt(hidden_dim) so per-layer norm scales line
    // up with how the weights were trained.
    if (cfg.is_gemma) {
        const scale: f32 = @floatCast(@sqrt(@as(f64, @floatFromInt(cfg.hidden_dim))));
        for (state.hidden) |*v| v.* *= scale;
    }

    for (model.layer_tensors, 0..) |lt, li| {
        const layer: u32 = @intCast(li);
        try rmsNormTensor(model, lt.attn_norm.?, state.hidden, state.norm, state.row_scratch);
        // SSM blocks only fire when the model actually carries SSM tensors AND
        // this layer isn't slotted as full-attention by the hybrid interval.
        const is_ssm_layer = cfg.ssm_d_inner > 0 and !isFullAttentionLayer(cfg, layer);
        if (is_ssm_layer) {
            try runSsmLayer(model, state, lt, layer);
        } else {
            try runAttentionLayer(model, state, lt, layer, position);
        }
        // FFN: dispatch the right block per layer. Gemma 4 MoE (fused
        // `ffn_gate_up_exps`) wants the parallel shared-MLP + routed-experts
        // path; the Qwen-style separate `ffn_gate_exps`/`ffn_up_exps` shape
        // goes through `runMoeLayer`; everything else is plain dense.
        if (lt.ffn_gate_up_exps != null and cfg.is_gemma) {
            try runGemma4MoeLayer(model, state, lt);
        } else if (lt.ffn_gate_exps != null) {
            try runMoeLayer(model, state, lt, direct_compute_tracking);
        } else {
            try runDenseFfnLayer(model, state, lt);
        }
        // Gemma 4 layer_output_scale: hidden *= scalar before the next layer.
        if (model.layer_output_scales) |scales| {
            const scale = scales[li];
            if (scale != 1.0) {
                for (state.hidden) |*v| v.* *= scale;
            }
        }
    }

    // During prompt ingestion only the final prompt token's logits are consumed
    // to seed decode. Earlier prompt tokens only advance KV/SSM/residual state,
    // so skip their final norm + LM head instead of streaming the full vocab
    // matrix for results that are immediately discarded.
    const need_logits = state.decode_phase or selection_out != null;
    if (!need_logits) {
        next_token.* = 0;
        return;
    }

    if (direct_final_norm_weight0) |gpu_weight0| {
        try rmsNormTensorWithFirstWeight(
            model,
            model.final_norm_info,
            state.hidden,
            state.norm,
            state.row_scratch,
            gpu_weight0,
            direct_compute_tracking,
        );
        consumed_gpu_model_value.* = true;
    } else {
        try rmsNormTensor(model, model.final_norm_info, state.hidden, state.norm, state.row_scratch);
    }

    const need_top2 = selection_out != null;
    const lm_head_rows = model.effectiveLmHeadRows(state.decode_phase);
    var selection: ArgmaxTop2Result = .{};
    if (model.lm_head_q4_0) |q40| {
        if (need_top2) {
            selection = try argmaxMatvecRawTop2(state.pool, q40, .q4_0, state.norm, lm_head_rows, state.row_scratch);
            next_token.* = selection.best.index;
        } else {
            next_token.* = try argmaxMatvecRaw(state.pool, q40, .q4_0, state.norm, lm_head_rows, state.row_scratch);
        }
    } else {
        const lm_head = model.requantOrRaw(model.lm_head_info);
        if (canDotDirect(lm_head.type_, @intCast(state.norm.len))) {
            if (need_top2) {
                selection = try argmaxMatvecRawTop2(state.pool, lm_head.raw, lm_head.type_, state.norm, lm_head_rows, state.row_scratch);
                next_token.* = selection.best.index;
            } else {
                next_token.* = try argmaxMatvecRaw(state.pool, lm_head.raw, lm_head.type_, state.norm, lm_head_rows, state.row_scratch);
            }
        } else {
            try matvecRaw(state.pool, lm_head.raw, lm_head.type_, state.norm, lm_head_rows, state.row_scratch, state.logits);
            if (need_top2) {
                selection = argmaxTop2Slice(state.logits[0..lm_head_rows]);
                next_token.* = selection.best.index;
            } else {
                next_token.* = argmaxSlice(state.logits[0..lm_head_rows]);
            }
        }
    }
    if (selection_out) |out| out.* = selection;
}

fn runAttentionLayer(
    model: *const Model,
    state: *ScalarDecodeState,
    lt: LayerTensors,
    layer: u32,
    position: u32,
) !void {
    const cfg = model.config;
    const q_rows: u32 = @intCast(lt.attn_q.?.numElements() / cfg.hidden_dim);
    const k_rows: u32 = @intCast(lt.attn_k.?.numElements() / cfg.hidden_dim);
    const has_v = lt.attn_v != null;
    const v_rows: u32 = if (has_v) @intCast(lt.attn_v.?.numElements() / cfg.hidden_dim) else k_rows;
    // Gemma layers mix per-layer head_dim (SWA vs global). The norm tensors
    // carry the layer's actual head_dim — use it when present so rmsNormHeads
    // and RoPE see consistent sizes.
    const layer_head_dim: u32 = if (lt.attn_q_norm) |qn|
        @intCast(qn.numElements())
    else if (lt.attn_k_norm) |kn|
        @intCast(kn.numElements())
    else
        cfg.head_dim;
    const layer_rope_dim = @min(if (cfg.rope_dim > 0) cfg.rope_dim else layer_head_dim, layer_head_dim);
    const layer_n_kv_heads = if (layer_head_dim > 0) k_rows / layer_head_dim else cfg.n_kv_heads;
    const packed_q_gate = q_rows == cfg.q_dim * 2;
    const active_q_rows = if (packed_q_gate) q_rows / 2 else q_rows;
    // Gemma SWA vs global have different n_q_heads even within the same model;
    // recover it from the Q row count so rmsNormHeads/applyRope use the right stride.
    const layer_n_heads: u32 = if (layer_head_dim > 0) active_q_rows / layer_head_dim else cfg.n_heads;

    // q/(gate)/k/v all project `attn_norm` — fuse into one pool dispatch so the
    // attention layer takes one barrier instead of one + two main-thread serial
    // matvecs (the small k/v rows ride along on the workers).
    {
        var parts: [matvec_fuse_max_segments]FusedPart = undefined;
        var n: usize = 0;
        if (packed_q_gate) {
            parts[n] = .{ .info = lt.attn_q.?, .rows = q_rows, .out = state.q_full[0..q_rows] };
            n += 1;
        } else {
            parts[n] = .{ .info = lt.attn_q.?, .rows = q_rows, .out = state.q[0..active_q_rows] };
            n += 1;
            if (lt.attn_gate) |gate_t| {
                parts[n] = .{ .info = gate_t, .rows = q_rows, .out = state.gate[0..active_q_rows] };
                n += 1;
            }
        }
        parts[n] = .{ .info = lt.attn_k.?, .rows = k_rows, .out = state.k[0..k_rows] };
        n += 1;
        if (has_v) {
            parts[n] = .{ .info = lt.attn_v.?, .rows = v_rows, .out = state.v[0..v_rows] };
            n += 1;
        }
        try matvecFusedTensors(state.pool, model, parts[0..n], state.norm, state.row_scratch);
    }
    // Gemma global-attention layers omit attn_v entirely; the original projection
    // reused K as V. Mirror that by snapshotting pre-RoPE K into the V buffer.
    if (!has_v) @memcpy(state.v[0..v_rows], state.k[0..k_rows]);
    if (packed_q_gate) {
        deinterleaveQGate(state.q_full[0..q_rows], state.q[0..active_q_rows], state.gate[0..active_q_rows], layer_head_dim, layer_n_heads);
    }

    // Gemma 4 unit-norms V per head before storing into the KV cache. When
    // attn_v is missing (global layer), V was just snapshotted from raw K above;
    // when present, V already lives in state.v from its own projection.
    if (cfg.is_gemma and cfg.rope_freq_base_swa > 0) {
        applyVUnitNormPerHead(state.v[0..v_rows], layer_head_dim, layer_n_kv_heads, cfg.rms_norm_eps);
    }

    if (lt.attn_q_norm) |q_norm| try rmsNormHeads(model, q_norm, state.q[0..active_q_rows], layer_head_dim, layer_n_heads);
    if (lt.attn_k_norm) |k_norm| try rmsNormHeads(model, k_norm, state.k[0..k_rows], layer_head_dim, layer_n_kv_heads);
    // Gemma SWA layers (layer_head_dim < cfg.head_dim) use a separate RoPE
    // freq base; global layers use the standard base.
    const inv_freq_for_layer: []const f32 = if (cfg.is_gemma and model.rope_inv_freq_swa != null and layer_head_dim < cfg.head_dim)
        model.rope_inv_freq_swa.?
    else
        model.rope_inv_freq;
    applyRope(state.q[0..active_q_rows], layer_head_dim, layer_rope_dim, layer_n_heads, position, inv_freq_for_layer);
    applyRope(state.k[0..k_rows], layer_head_dim, layer_rope_dim, layer_n_kv_heads, position, inv_freq_for_layer);

    const kv_off = state.kvOffset(cfg, layer, position);
    @memcpy(state.kv_k[kv_off..][0..k_rows], state.k[0..k_rows]);
    @memcpy(state.kv_v[kv_off..][0..v_rows], state.v[0..v_rows]);

    try flashAttentionCpu(model, state, layer, position, layer_head_dim, layer_n_kv_heads, layer_n_heads);
    if (packed_q_gate or lt.attn_gate != null) {
        for (state.attn_out[0..active_q_rows], state.gate[0..active_q_rows]) |*out, gate| {
            out.* *= sigmoid(gate);
        }
    }

    try matvecTensor(state.pool, model, lt.attn_output.?, state.attn_out[0..active_q_rows], cfg.hidden_dim, state.row_scratch, state.branch);
    // Gemma applies post_attention_norm to the projected attention output
    // before merging it into the residual stream.
    if (cfg.is_gemma) {
        if (lt.post_attention_norm) |pan| {
            try rmsNormTensor(model, pan, state.branch, state.branch, state.row_scratch);
        }
    }
    for (state.hidden, state.branch) |*h, b| h.* += b;
}

fn runSsmLayer(model: *const Model, state: *ScalarDecodeState, lt: LayerTensors, layer: u32) !void {
    const cfg = model.config;
    const d_inner = cfg.ssm_d_inner;
    const d_conv = cfg.ssm_d_conv;
    const d_state = cfg.ssm_d_state;
    const n_group = cfg.ssm_n_group;
    const dt_rank = cfg.ssm_dt_rank;
    const head_v_dim = d_inner / dt_rank;
    const conv_ch = d_inner + 2 * n_group * d_state;

    // `attn_qkv`, `attn_gate`, `ssm_alpha` and `ssm_beta` all project
    // `attn_norm` — run them as one fused pool dispatch instead of four (two
    // pool barriers + two main-thread serial matvecs) so the SSM layer takes
    // one fewer worker-pool barrier per token and the tiny dt-rank rows get
    // parallelised; decode is serialised hop-by-hop through ~200 pool barriers
    // per token, so this is real per-token time.
    try matvecFusedTensors(state.pool, model, &[_]FusedPart{
        .{ .info = lt.attn_qkv.?, .rows = conv_ch, .out = state.qkv[0..conv_ch] },
        .{ .info = lt.attn_gate.?, .rows = d_inner, .out = state.z[0..d_inner] },
        .{ .info = lt.ssm_alpha.?, .rows = dt_rank, .out = state.alpha[0..dt_rank] },
        .{ .info = lt.ssm_beta.?, .rows = dt_rank, .out = state.beta[0..dt_rank] },
    }, state.norm, state.row_scratch);

    const conv_state = state.convStateForLayer(cfg, layer);
    const d_conv_1 = d_conv - 1;
    const conv_len = conv_ch * d_conv;
    const conv_len_usize: usize = @intCast(conv_len);
    var conv_kernel_transposed_d4 = false;
    const conv_kernel = kernel: {
        const layer_index: usize = @intCast(layer);
        if (layer_index < model.ssm_conv1d_kernels.len) {
            if (model.ssm_conv1d_kernels[layer_index]) |cached| {
                if (cached.len < conv_len_usize) return error.ShapeMismatch;
                conv_kernel_transposed_d4 = d_conv == 4;
                break :kernel cached[0..conv_len_usize];
            }
        }
        const scratch_kernel = state.row_scratch[0..conv_len_usize];
        try readTensorFlat(model.tensorData(lt.ssm_conv1d.?), lt.ssm_conv1d.?.type_, scratch_kernel);
        break :kernel scratch_kernel;
    };

    // Per-channel SSM conv1d + SiLU on a few-thousand-channel vector — the
    // sigmoid in `sum * sigmoid(sum)` is the work-dominant cost (~one @exp per
    // channel; per layer this is ~conv_ch @exp's, and the SSM layers fire 30
    // times per token). Fan out across the persistent decode pool so the main
    // thread doesn't carry that cost alone while the workers idle.
    var conv_ctx = ConvSiluCtx{
        .conv_kernel = conv_kernel,
        .conv_state = conv_state,
        .qkv = state.qkv[0..conv_ch],
        .conv_out = state.conv_out[0..conv_ch],
        .d_conv = d_conv,
        .d_conv_1 = d_conv_1,
        .conv_ch = conv_ch,
        .kernel_transposed_d4 = conv_kernel_transposed_d4,
    };
    runConvSiluParallel(state.pool, &conv_ctx);
    if (d_conv_1 > 1) {
        const shift = (d_conv_1 - 1) * conv_ch;
        std.mem.copyForwards(f32, conv_state[0..shift], conv_state[conv_ch..][0..shift]);
    }
    @memcpy(conv_state[(d_conv_1 - 1) * conv_ch ..][0..conv_ch], state.qkv[0..conv_ch]);

    const qk_dim = d_state * n_group;
    const q_ssm = state.conv_out[0..qk_dim];
    const k_ssm = state.conv_out[qk_dim..][0..qk_dim];
    const v_ssm = state.conv_out[2 * qk_dim ..][0..d_inner];
    for (0..n_group) |h| {
        l2Normalize(q_ssm[h * d_state ..][0..d_state]);
        l2Normalize(k_ssm[h * d_state ..][0..d_state]);
    }

    const dt_bias_view: []const f32 = blk: {
        if (lt.ssm_dt_bias) |t| {
            if (model.cachedF32(t)) |c| {
                if (c.len < dt_rank) return error.ShapeMismatch;
                break :blk c[0..dt_rank];
            }
            @memset(state.dt_bias[0..dt_rank], 0);
            try readTensorFlat(model.tensorData(t), t.type_, state.dt_bias[0..dt_rank]);
            break :blk state.dt_bias[0..dt_rank];
        }
        @memset(state.dt_bias[0..dt_rank], 0);
        break :blk state.dt_bias[0..dt_rank];
    };
    const has_ssm_a = lt.ssm_a != null;
    const ssm_a_view: []const f32 = blk: {
        if (lt.ssm_a) |t| {
            if (model.cachedF32(t)) |c| {
                if (c.len < dt_rank) return error.ShapeMismatch;
                break :blk c[0..dt_rank];
            }
            @memset(state.ssm_a[0..dt_rank], 0);
            try readTensorFlat(model.tensorData(t), t.type_, state.ssm_a[0..dt_rank]);
            break :blk state.ssm_a[0..dt_rank];
        }
        @memset(state.ssm_a[0..dt_rank], 0);
        break :blk state.ssm_a[0..dt_rank];
    };

    for (0..dt_rank) |i| {
        const a = state.alpha[i] + dt_bias_view[i];
        const sp = softplus(a);
        state.alpha[i] = if (has_ssm_a) sp * ssm_a_view[i] else -sp;
        state.beta[i] = sigmoid(state.beta[i]);
    }
    const q_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(d_state)));
    for (q_ssm) |*v| v.* *= q_scale;

    const norm_elems: u32 = if (lt.ssm_norm) |t| @intCast(t.numElements()) else 0;
    const norm_per_head = norm_elems >= d_inner;
    const norm_len: usize = @intCast(norm_elems);
    const ssm_norm_w_view: []const f32 = blk: {
        if (norm_elems == 0) break :blk &.{};
        if (lt.ssm_norm) |t| {
            if (model.cachedF32(t)) |c| {
                if (c.len < norm_len) return error.ShapeMismatch;
                break :blk c[0..norm_len];
            }
        }
        if (norm_len > state.ssm_norm_w.len) return error.ShapeMismatch;
        if (lt.ssm_norm) |t| try readTensorFlat(model.tensorData(t), t.type_, state.ssm_norm_w[0..norm_len]);
        break :blk state.ssm_norm_w[0..norm_len];
    };

    // The per-head SSM state recurrence (decay multiply, rank-1 update, output
    // projection, RMS-norm + SiLU gating) is independent across the dt_rank heads:
    // head h owns the disjoint head_v_dim×head_v_dim block ssm_matrix[h*hvd²..] and
    // the disjoint slice ssm_out[h*hvd..]. This is the largest serial chunk left in
    // the T-CPU decode (~dt_rank·hvd² float ops × 30 SSM layers, run on the main
    // thread while the worker pool idled). Fan it out across the persistent pool.
    var head_ctx = SsmHeadCtx{
        .ssm_matrix = state.ssmStateForLayer(cfg, layer),
        .q_ssm = q_ssm,
        .k_ssm = k_ssm,
        .v_ssm = v_ssm,
        .z = state.z[0..d_inner],
        .ssm_out = state.ssm_out[0..d_inner],
        .alpha = state.alpha[0..dt_rank],
        .beta = state.beta[0..dt_rank],
        .ssm_norm_w = ssm_norm_w_view,
        .head_v_dim = head_v_dim,
        .d_state = d_state,
        .n_group = n_group,
        .dt_rank = dt_rank,
        .norm_elems = norm_elems,
        .norm_per_head = norm_per_head,
        .rms_norm_eps = cfg.rms_norm_eps,
    };
    // Try to fold the SSM head update and the ssm_out projection into a single
    // FastPool dispatch with an in-task barrier — saves one pool barrier per
    // SSM layer × 30 layers/token (~15-20 µs each on the 9800X3D + RDNA node).
    // Falls back to the two-dispatch path when the FastPool isn't wired or the
    // layout doesn't partition (non-multiple-of-32 head_v_dim, etc.).
    const ssm_out_w = model.requantOrRaw(lt.ssm_out.?);
    const fused_ok = runSsmHeadsAndOutProjFused(state, &head_ctx, ssm_out_w, cfg.hidden_dim);
    if (!fused_ok) {
        runSsmHeadsParallel(state.pool, &head_ctx);
        try matvecTensor(state.pool, model, lt.ssm_out.?, state.ssm_out[0..d_inner], cfg.hidden_dim, state.row_scratch, state.branch);
    }
    for (state.hidden, state.branch) |*h, b| h.* += b;
}

const ssm_head_parallel_max_workers: usize = 16;
// Below this many heads the spawn/sync cost outweighs the per-head work; just run serial.
const ssm_head_parallel_min_heads: usize = 4;

const SsmHeadCtx = struct {
    ssm_matrix: []f32,
    q_ssm: []const f32,
    k_ssm: []const f32,
    v_ssm: []const f32,
    z: []const f32,
    ssm_out: []f32,
    alpha: []const f32,
    beta: []const f32,
    ssm_norm_w: []const f32,
    head_v_dim: usize,
    d_state: usize,
    n_group: usize,
    dt_rank: usize,
    norm_elems: u32,
    norm_per_head: bool,
    rms_norm_eps: f32,
};

const SsmHeadWorker = struct {
    ctx: *const SsmHeadCtx,
    h_start: usize,
    h_end: usize,
};

fn ssmHeadWorkerMain(worker: *SsmHeadWorker) void {
    runSsmHeadRange(worker.ctx, worker.h_start, worker.h_end);
}

fn ssmHeadWorkerTask(ctx: *anyopaque) void {
    const worker: *SsmHeadWorker = @ptrCast(@alignCast(ctx));
    runSsmHeadRange(worker.ctx, worker.h_start, worker.h_end);
}

fn runSsmHeadsParallel(pool: ?*thread_pool.Pool, ctx: *const SsmHeadCtx) void {
    const dt_rank = ctx.dt_rank;
    // Prefer FastPool when the matvec dispatcher has it wired up — its
    // atomic-counter fan-out replaces thread_pool.Pool's mutex+condvar+heap-alloc
    // closure path, saving ~µs per barrier. Called once per SSM layer (30/token).
    if (pool != null) {
        if (matvec_fast_pool) |fp| {
            const executors = fp.executorCount();
            if (dt_rank >= ssm_head_parallel_min_heads and executors > 1) {
                const workers = @min(dt_rank, @min(executors, ssm_head_parallel_max_workers));
                const heads_per = (dt_rank + workers - 1) / workers;
                var params: [ssm_head_parallel_max_workers]SsmHeadWorker = undefined;
                var tasks: [ssm_head_parallel_max_workers]zinc_rt.fast_pool.Task = undefined;
                var dispatched: usize = 0;
                while (dispatched < workers) : (dispatched += 1) {
                    const start = dispatched * heads_per;
                    if (start >= dt_rank) break;
                    const end = @min(dt_rank, (dispatched + 1) * heads_per);
                    params[dispatched] = .{ .ctx = ctx, .h_start = start, .h_end = end };
                    tasks[dispatched] = .{ .fn_ = ssmHeadWorkerTask, .ctx = @ptrCast(&params[dispatched]) };
                }
                fp.dispatchAndRun(tasks[0..dispatched]);
                return;
            }
        }
    }
    if (pool) |p| {
        const executors = poolExecutorCount(p);
        if (dt_rank >= ssm_head_parallel_min_heads and executors > 1) {
            const workers = @min(dt_rank, @min(executors, ssm_head_parallel_max_workers));
            const heads_per = (dt_rank + workers - 1) / workers;
            var params: [ssm_head_parallel_max_workers]SsmHeadWorker = undefined;
            var wg: thread_pool.WaitGroup = .{};
            var dispatched: usize = 0;
            while (dispatched < workers) : (dispatched += 1) {
                const start = dispatched * heads_per;
                if (start >= dt_rank) break;
                const end = @min(dt_rank, (dispatched + 1) * heads_per);
                params[dispatched] = .{ .ctx = ctx, .h_start = start, .h_end = end };
                p.spawnWg(&wg, ssmHeadWorkerMain, .{&params[dispatched]});
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    runSsmHeadRange(ctx, 0, dt_rank);
}

/// Per-token decode wires the SSM-head update and the `ssm_out` projection back
/// to `hidden_dim` as two sequential FastPool dispatches (heads → ssm_out
/// matvec); that costs one extra pool round-trip per SSM layer (~15-20 µs each
/// on the 9800X3D + RDNA test node, ×30 SSM layers = ~0.5-0.6 ms/token of pure
/// barrier/dispatch overhead). The two phases run on the SAME worker set and
/// the matvec only reads what the head step just wrote, so we can fold them
/// into one dispatch with an in-task atomic barrier: each worker finishes its
/// head range, writes its `ssm_out` slice + its block of `input_sum32`,
/// barriers, then computes its row slice of the matvec into `state.branch`.
/// Main thread still does the small residual add. Falls back to the two-
/// dispatch path when the FastPool isn't available or the layout doesn't
/// partition cleanly.
const SsmFusedCtx = struct {
    heads_ctx: *const SsmHeadCtx,
    out_proj_raw: []const u8,
    out_proj_type: gguf.GGMLType,
    branch: []f32,
    /// Optional `[d_inner/32]` buffer; each worker fills the slice corresponding
    /// to its head range in phase 1, then all workers read the full buffer in
    /// phase 2 after the barrier publishes phase-1 writes.
    input_sum32: ?[]f32,
    /// `head_v_dim / 32`; non-zero only when `input_sum32` is non-null.
    head_v_dim_blocks: u32,
    barrier: std.atomic.Value(u32) align(64) = std.atomic.Value(u32).init(0),
    n_workers: u32,
    failed: std.atomic.Value(bool) align(64) = std.atomic.Value(bool).init(false),
};

const SsmFusedWorker = struct {
    ctx: *SsmFusedCtx,
    h_start: usize,
    h_end: usize,
    row_start: u32,
    row_end: u32,
};

fn ssmFusedWorkerTask(ctx_ptr: *anyopaque) void {
    const worker: *SsmFusedWorker = @ptrCast(@alignCast(ctx_ptr));
    const ctx = worker.ctx;

    // Phase 1a: SSM head state update (writes our slice of ssm_out).
    runSsmHeadRange(ctx.heads_ctx, worker.h_start, worker.h_end);

    // Phase 1b: input_sum32 for our slice (only needed for Q4_0/Q4_K/Q5_K).
    if (ctx.input_sum32) |sums| {
        const head_v_dim = ctx.heads_ctx.head_v_dim;
        const slice_off = worker.h_start * head_v_dim;
        const slice_len = (worker.h_end - worker.h_start) * head_v_dim;
        const sum_off = worker.h_start * @as(usize, ctx.head_v_dim_blocks);
        const sum_len = (worker.h_end - worker.h_start) * @as(usize, ctx.head_v_dim_blocks);
        if (slice_len > 0 and sum_len > 0) {
            dequant.fillInputSum32(
                ctx.heads_ctx.ssm_out[slice_off..][0..slice_len],
                sums[sum_off..][0..sum_len],
            );
        }
    }

    // Barrier: publish phase-1 writes; wait for all other workers' writes.
    _ = ctx.barrier.fetchAdd(1, .release);
    while (ctx.barrier.load(.acquire) < ctx.n_workers) {
        std.atomic.spinLoopHint();
    }

    if (worker.row_end <= worker.row_start) return;
    // Phase 2: matvec row slice into branch, using the now-published ssm_out.
    matvecRawDirectSerial(
        ctx.out_proj_raw,
        ctx.out_proj_type,
        ctx.heads_ctx.ssm_out,
        if (ctx.input_sum32) |sums| sums else null,
        worker.row_start,
        worker.row_end,
        ctx.branch,
    ) catch {
        ctx.failed.store(true, .release);
    };
}

/// Run the SSM head step and the ssm_out projection in a single FastPool
/// dispatch. Returns `true` on success — caller still owns the trailing
/// residual add (`hidden += branch`). Returns `false` if the layout doesn't
/// fit the fused worker shape; caller should fall back to the two-dispatch
/// path.
fn runSsmHeadsAndOutProjFused(
    state: *ScalarDecodeState,
    head_ctx: *const SsmHeadCtx,
    out_proj_w: WeightView,
    hidden_dim: u32,
) bool {
    if (state.pool == null) return false;
    const fp = matvec_fast_pool orelse return false;

    const dt_rank = head_ctx.dt_rank;
    if (dt_rank < ssm_head_parallel_min_heads) return false;
    const executors = fp.executorCount();
    if (executors < 2) return false;

    const head_v_dim = head_ctx.head_v_dim;
    const d_inner = dt_rank * head_v_dim;
    if (d_inner == 0) return false;
    const cols: u32 = @intCast(d_inner);
    if (!canDotDirect(out_proj_w.type_, cols)) return false;
    if (head_v_dim % 32 != 0) return false;
    if (hidden_dim == 0) return false;

    const want_sum32 = wantsInputSum32(out_proj_w.type_);
    const sum_total: usize = if (want_sum32) d_inner / 32 else 0;
    if (want_sum32 and state.row_scratch.len < sum_total) return false;
    const sum_slice: ?[]f32 = if (want_sum32) state.row_scratch[0..sum_total] else null;

    const max_workers = @min(dt_rank, @min(executors, ssm_head_parallel_max_workers));
    if (max_workers < 2) return false;

    var ctx_storage = SsmFusedCtx{
        .heads_ctx = head_ctx,
        .out_proj_raw = out_proj_w.raw,
        .out_proj_type = out_proj_w.type_,
        .branch = state.branch,
        .input_sum32 = sum_slice,
        .head_v_dim_blocks = if (want_sum32) @intCast(head_v_dim / 32) else 0,
        .n_workers = 0,
    };

    var params: [ssm_head_parallel_max_workers]SsmFusedWorker = undefined;
    var tasks: [ssm_head_parallel_max_workers]zinc_rt.fast_pool.Task = undefined;

    const heads_per = (dt_rank + max_workers - 1) / max_workers;
    var dispatched: usize = 0;
    while (dispatched < max_workers) : (dispatched += 1) {
        const h_start = dispatched * heads_per;
        if (h_start >= dt_rank) break;
        const h_end = @min(dt_rank, (dispatched + 1) * heads_per);
        params[dispatched] = .{
            .ctx = &ctx_storage,
            .h_start = h_start,
            .h_end = h_end,
            // Row slice is computed below once we know the final dispatched count.
            .row_start = 0,
            .row_end = 0,
        };
        tasks[dispatched] = .{ .fn_ = ssmFusedWorkerTask, .ctx = @ptrCast(&params[dispatched]) };
    }
    if (dispatched < 2) return false;

    // Even row split across the active workers (ceil-division covers full hidden).
    const rows_per = (hidden_dim + dispatched - 1) / dispatched;
    for (params[0..dispatched], 0..) |*p, i| {
        const row_start: u32 = @intCast(@min(hidden_dim, @as(u32, @intCast(i)) * rows_per));
        const row_end: u32 = @intCast(@min(hidden_dim, @as(u32, @intCast(i + 1)) * rows_per));
        p.row_start = row_start;
        p.row_end = row_end;
    }

    // Publish the final worker count so the barrier and dispatch agree.
    ctx_storage.n_workers = @intCast(dispatched);
    fp.dispatchAndRun(tasks[0..dispatched]);

    return !ctx_storage.failed.load(.acquire);
}

fn runSsmHeadRange(ctx: *const SsmHeadCtx, h_start: usize, h_end: usize) void {
    const head_v_dim = ctx.head_v_dim;
    const d_state = ctx.d_state;
    const kv_len = @min(d_state, head_v_dim);
    const head_v_dim_f: f32 = @floatFromInt(head_v_dim);
    var h = h_start;
    while (h < h_end) : (h += 1) {
        const s_base = h * head_v_dim * head_v_dim;
        const decay = @exp(ctx.alpha[h]);
        const b_val = ctx.beta[h];
        const k_hi = if (ctx.n_group == ctx.dt_rank) h else h % ctx.n_group;
        const k_head = ctx.k_ssm[k_hi * d_state ..][0..kv_len];
        const q_hi = if (ctx.n_group == ctx.dt_rank) h else h % ctx.n_group;
        const q_head = ctx.q_ssm[q_hi * d_state ..][0..kv_len];
        const v_head = ctx.v_ssm[h * head_v_dim ..][0..head_v_dim];
        const out = ctx.ssm_out[h * head_v_dim ..][0..head_v_dim];
        // Per-head SSM state update — three passes over the head_v_dim×kv_len
        // sub-matrix (decay+sk-dot fused, rank-1 update, output projection).
        // The reductions (sk and val) carry a serial FP-add dependency chain
        // that prevents Zig/LLVM from auto-vectorising without fastmath, so we
        // drive each pass through @Vector(16, f32) with four independent
        // accumulators (same shape as the AVX-512 zmm dot in dequant.zig).
        // The active prefix's trailing region (kv_len < head_v_dim) still gets
        // a plain decay pass since the rank-1 update only touches the active
        // prefix. Mathematically equivalent: after the loop, every element of
        // the active prefix holds OLD*decay + k*d_val, every element of the
        // trailing region holds OLD*decay, and sk equals the original
        // sum_c(OLD[r][c] * decay * k[c]).
        for (0..head_v_dim) |row| {
            const row_base = s_base + row * head_v_dim;
            const sk = ssmDecaySkRow(ctx.ssm_matrix, row_base, k_head, kv_len, decay);
            if (kv_len < head_v_dim) {
                for (kv_len..head_v_dim) |col| ctx.ssm_matrix[row_base + col] *= decay;
            }
            const d_val = b_val * (v_head[row] - sk);
            out[row] = ssmRank1UpdateAndOutProjRow(ctx.ssm_matrix, row_base, k_head, q_head, kv_len, d_val);
        }
        const z_head = ctx.z[h * head_v_dim ..][0..head_v_dim];
        // 4-accumulator Vec16f sum-of-squares — `sq += v * v` chains through the
        // same FP-add so scalar/auto-vec is a single FMA chain (~head_v_dim
        // latency cycles). head_v_dim = d_inner / dt_rank (= 128 on Qwen3.6) is
        // a multiple of 64 here, so the wide loop carries the work.
        const sq = ssmHeadSumSq(out);
        const inv_rms = 1.0 / @sqrt(sq / head_v_dim_f + ctx.rms_norm_eps);
        // Per-head SiLU output gate `out = out * inv_rms * norm_w * z * sigmoid(z)`.
        // Runs 32 heads × 30 SSM layers × head_v_dim=128 = 122,880 elements per
        // decode token; scalar `sigmoid` blocks Zig/LLVM from auto-vectorising the
        // chain. Branch-hoist the norm-weight selector and lower the inner loop
        // through @Vector(16, f32) with the same `one / (one + @exp(-z))` pattern
        // as `swiglu` / `runConvSiluRangeDConv4` (already proven efficient).
        ssmHeadSiluGate(out, z_head, inv_rms, ctx, h);
    }
}

inline fn ssmHeadSiluGate(out: []f32, z_head: []const f32, inv_rms: f32, ctx: *const SsmHeadCtx, h: usize) void {
    const head_v_dim = ctx.head_v_dim;
    if (ctx.norm_elems == 0) {
        ssmHeadSiluGateNoNorm(out, z_head, inv_rms);
        return;
    }
    if (ctx.norm_per_head) {
        const nw = ctx.ssm_norm_w[h * head_v_dim ..][0..head_v_dim];
        ssmHeadSiluGateWithNorm(out, z_head, nw, inv_rms);
        return;
    }
    if (head_v_dim == ctx.d_state) {
        const nw = ctx.ssm_norm_w[0..head_v_dim];
        ssmHeadSiluGateWithNorm(out, z_head, nw, inv_rms);
        return;
    }
    // Rare fallback: shared norm weight with head_v_dim != d_state ⇒ index wraps.
    for (0..head_v_dim) |i| {
        const nv = out[i] * inv_rms * ctx.ssm_norm_w[i % ctx.d_state];
        out[i] = nv * (z_head[i] * sigmoid(z_head[i]));
    }
}

inline fn ssmHeadSiluGateNoNorm(out: []f32, z_head: []const f32, inv_rms: f32) void {
    const one: SsmVec16f = @splat(1.0);
    const inv_rms_v: SsmVec16f = @splat(inv_rms);
    var i: usize = 0;
    while (i + 16 <= out.len) : (i += 16) {
        const o: SsmVec16f = out[i..][0..16].*;
        const z: SsmVec16f = z_head[i..][0..16].*;
        const sig = one / (one + @exp(-z));
        out[i..][0..16].* = (o * inv_rms_v) * (z * sig);
    }
    while (i < out.len) : (i += 1) {
        out[i] = (out[i] * inv_rms) * (z_head[i] * sigmoid(z_head[i]));
    }
}

inline fn ssmHeadSiluGateWithNorm(out: []f32, z_head: []const f32, nw: []const f32, inv_rms: f32) void {
    const one: SsmVec16f = @splat(1.0);
    const inv_rms_v: SsmVec16f = @splat(inv_rms);
    var i: usize = 0;
    while (i + 16 <= out.len) : (i += 16) {
        const o: SsmVec16f = out[i..][0..16].*;
        const z: SsmVec16f = z_head[i..][0..16].*;
        const w: SsmVec16f = nw[i..][0..16].*;
        const sig = one / (one + @exp(-z));
        out[i..][0..16].* = (o * inv_rms_v * w) * (z * sig);
    }
    while (i < out.len) : (i += 1) {
        out[i] = (out[i] * inv_rms * nw[i]) * (z_head[i] * sigmoid(z_head[i]));
    }
}

const SsmVec16f = @Vector(16, f32);

inline fn ssmHeadSumSq(v: []const f32) f32 {
    var acc0: SsmVec16f = @splat(0.0);
    var acc1: SsmVec16f = @splat(0.0);
    var acc2: SsmVec16f = @splat(0.0);
    var acc3: SsmVec16f = @splat(0.0);
    var i: usize = 0;
    while (i + 64 <= v.len) : (i += 64) {
        const v0: SsmVec16f = v[i..][0..16].*;
        const v1: SsmVec16f = v[i + 16 ..][0..16].*;
        const v2: SsmVec16f = v[i + 32 ..][0..16].*;
        const v3: SsmVec16f = v[i + 48 ..][0..16].*;
        acc0 = @mulAdd(SsmVec16f, v0, v0, acc0);
        acc1 = @mulAdd(SsmVec16f, v1, v1, acc1);
        acc2 = @mulAdd(SsmVec16f, v2, v2, acc2);
        acc3 = @mulAdd(SsmVec16f, v3, v3, acc3);
    }
    var acc_vec: SsmVec16f = (acc0 + acc1) + (acc2 + acc3);
    while (i + 16 <= v.len) : (i += 16) {
        const x: SsmVec16f = v[i..][0..16].*;
        acc_vec = @mulAdd(SsmVec16f, x, x, acc_vec);
    }
    var sq = @reduce(.Add, acc_vec);
    while (i < v.len) : (i += 1) sq += v[i] * v[i];
    return sq;
}

inline fn ssmDecaySkRow(matrix: []f32, row_base: usize, k: []const f32, kv_len: usize, decay: f32) f32 {
    const decay_v: SsmVec16f = @splat(decay);
    var acc0: SsmVec16f = @splat(0.0);
    var acc1: SsmVec16f = @splat(0.0);
    var acc2: SsmVec16f = @splat(0.0);
    var acc3: SsmVec16f = @splat(0.0);
    var col: usize = 0;
    while (col + 64 <= kv_len) : (col += 64) {
        const m0: SsmVec16f = matrix[row_base + col ..][0..16].*;
        const m1: SsmVec16f = matrix[row_base + col + 16 ..][0..16].*;
        const m2: SsmVec16f = matrix[row_base + col + 32 ..][0..16].*;
        const m3: SsmVec16f = matrix[row_base + col + 48 ..][0..16].*;
        const k0: SsmVec16f = k[col..][0..16].*;
        const k1: SsmVec16f = k[col + 16 ..][0..16].*;
        const k2: SsmVec16f = k[col + 32 ..][0..16].*;
        const k3: SsmVec16f = k[col + 48 ..][0..16].*;
        const d0 = m0 * decay_v;
        const d1 = m1 * decay_v;
        const d2 = m2 * decay_v;
        const d3 = m3 * decay_v;
        matrix[row_base + col ..][0..16].* = d0;
        matrix[row_base + col + 16 ..][0..16].* = d1;
        matrix[row_base + col + 32 ..][0..16].* = d2;
        matrix[row_base + col + 48 ..][0..16].* = d3;
        acc0 = @mulAdd(SsmVec16f, d0, k0, acc0);
        acc1 = @mulAdd(SsmVec16f, d1, k1, acc1);
        acc2 = @mulAdd(SsmVec16f, d2, k2, acc2);
        acc3 = @mulAdd(SsmVec16f, d3, k3, acc3);
    }
    while (col + 16 <= kv_len) : (col += 16) {
        const m: SsmVec16f = matrix[row_base + col ..][0..16].*;
        const kv: SsmVec16f = k[col..][0..16].*;
        const d = m * decay_v;
        matrix[row_base + col ..][0..16].* = d;
        acc0 = @mulAdd(SsmVec16f, d, kv, acc0);
    }
    var sk = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));
    while (col < kv_len) : (col += 1) {
        const decayed = matrix[row_base + col] * decay;
        matrix[row_base + col] = decayed;
        sk = @mulAdd(f32, decayed, k[col], sk);
    }
    return sk;
}

inline fn ssmRank1UpdateRow(matrix: []f32, row_base: usize, k: []const f32, kv_len: usize, d_val: f32) void {
    const d_val_v: SsmVec16f = @splat(d_val);
    var col: usize = 0;
    while (col + 64 <= kv_len) : (col += 64) {
        const m0: SsmVec16f = matrix[row_base + col ..][0..16].*;
        const m1: SsmVec16f = matrix[row_base + col + 16 ..][0..16].*;
        const m2: SsmVec16f = matrix[row_base + col + 32 ..][0..16].*;
        const m3: SsmVec16f = matrix[row_base + col + 48 ..][0..16].*;
        const k0: SsmVec16f = k[col..][0..16].*;
        const k1: SsmVec16f = k[col + 16 ..][0..16].*;
        const k2: SsmVec16f = k[col + 32 ..][0..16].*;
        const k3: SsmVec16f = k[col + 48 ..][0..16].*;
        matrix[row_base + col ..][0..16].* = @mulAdd(SsmVec16f, k0, d_val_v, m0);
        matrix[row_base + col + 16 ..][0..16].* = @mulAdd(SsmVec16f, k1, d_val_v, m1);
        matrix[row_base + col + 32 ..][0..16].* = @mulAdd(SsmVec16f, k2, d_val_v, m2);
        matrix[row_base + col + 48 ..][0..16].* = @mulAdd(SsmVec16f, k3, d_val_v, m3);
    }
    while (col + 16 <= kv_len) : (col += 16) {
        const m: SsmVec16f = matrix[row_base + col ..][0..16].*;
        const kv: SsmVec16f = k[col..][0..16].*;
        matrix[row_base + col ..][0..16].* = @mulAdd(SsmVec16f, kv, d_val_v, m);
    }
    while (col < kv_len) : (col += 1) {
        matrix[row_base + col] = @mulAdd(f32, k[col], d_val, matrix[row_base + col]);
    }
}

inline fn ssmRank1UpdateAndOutProjRow(matrix: []f32, row_base: usize, k: []const f32, q: []const f32, kv_len: usize, d_val: f32) f32 {
    const d_val_v: SsmVec16f = @splat(d_val);
    var acc0: SsmVec16f = @splat(0.0);
    var acc1: SsmVec16f = @splat(0.0);
    var acc2: SsmVec16f = @splat(0.0);
    var acc3: SsmVec16f = @splat(0.0);
    var col: usize = 0;
    while (col + 64 <= kv_len) : (col += 64) {
        const m0: SsmVec16f = matrix[row_base + col ..][0..16].*;
        const m1: SsmVec16f = matrix[row_base + col + 16 ..][0..16].*;
        const m2: SsmVec16f = matrix[row_base + col + 32 ..][0..16].*;
        const m3: SsmVec16f = matrix[row_base + col + 48 ..][0..16].*;
        const k0: SsmVec16f = k[col..][0..16].*;
        const k1: SsmVec16f = k[col + 16 ..][0..16].*;
        const k2: SsmVec16f = k[col + 32 ..][0..16].*;
        const k3: SsmVec16f = k[col + 48 ..][0..16].*;
        const q0: SsmVec16f = q[col..][0..16].*;
        const q1: SsmVec16f = q[col + 16 ..][0..16].*;
        const q2: SsmVec16f = q[col + 32 ..][0..16].*;
        const q3: SsmVec16f = q[col + 48 ..][0..16].*;
        const upd0 = @mulAdd(SsmVec16f, k0, d_val_v, m0);
        const upd1 = @mulAdd(SsmVec16f, k1, d_val_v, m1);
        const upd2 = @mulAdd(SsmVec16f, k2, d_val_v, m2);
        const upd3 = @mulAdd(SsmVec16f, k3, d_val_v, m3);
        matrix[row_base + col ..][0..16].* = upd0;
        matrix[row_base + col + 16 ..][0..16].* = upd1;
        matrix[row_base + col + 32 ..][0..16].* = upd2;
        matrix[row_base + col + 48 ..][0..16].* = upd3;
        acc0 = @mulAdd(SsmVec16f, upd0, q0, acc0);
        acc1 = @mulAdd(SsmVec16f, upd1, q1, acc1);
        acc2 = @mulAdd(SsmVec16f, upd2, q2, acc2);
        acc3 = @mulAdd(SsmVec16f, upd3, q3, acc3);
    }
    while (col + 16 <= kv_len) : (col += 16) {
        const m: SsmVec16f = matrix[row_base + col ..][0..16].*;
        const kv: SsmVec16f = k[col..][0..16].*;
        const qv: SsmVec16f = q[col..][0..16].*;
        const updated = @mulAdd(SsmVec16f, kv, d_val_v, m);
        matrix[row_base + col ..][0..16].* = updated;
        acc0 = @mulAdd(SsmVec16f, updated, qv, acc0);
    }
    var val = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));
    while (col < kv_len) : (col += 1) {
        const updated = @mulAdd(f32, k[col], d_val, matrix[row_base + col]);
        matrix[row_base + col] = updated;
        val = @mulAdd(f32, updated, q[col], val);
    }
    return val;
}

inline fn ssmOutProjRow(matrix: []const f32, row_base: usize, q: []const f32, kv_len: usize) f32 {
    var acc0: SsmVec16f = @splat(0.0);
    var acc1: SsmVec16f = @splat(0.0);
    var acc2: SsmVec16f = @splat(0.0);
    var acc3: SsmVec16f = @splat(0.0);
    var col: usize = 0;
    while (col + 64 <= kv_len) : (col += 64) {
        const m0: SsmVec16f = matrix[row_base + col ..][0..16].*;
        const m1: SsmVec16f = matrix[row_base + col + 16 ..][0..16].*;
        const m2: SsmVec16f = matrix[row_base + col + 32 ..][0..16].*;
        const m3: SsmVec16f = matrix[row_base + col + 48 ..][0..16].*;
        const q0: SsmVec16f = q[col..][0..16].*;
        const q1: SsmVec16f = q[col + 16 ..][0..16].*;
        const q2: SsmVec16f = q[col + 32 ..][0..16].*;
        const q3: SsmVec16f = q[col + 48 ..][0..16].*;
        acc0 = @mulAdd(SsmVec16f, m0, q0, acc0);
        acc1 = @mulAdd(SsmVec16f, m1, q1, acc1);
        acc2 = @mulAdd(SsmVec16f, m2, q2, acc2);
        acc3 = @mulAdd(SsmVec16f, m3, q3, acc3);
    }
    while (col + 16 <= kv_len) : (col += 16) {
        const m: SsmVec16f = matrix[row_base + col ..][0..16].*;
        const qv: SsmVec16f = q[col..][0..16].*;
        acc0 = @mulAdd(SsmVec16f, m, qv, acc0);
    }
    var val = @reduce(.Add, (acc0 + acc1) + (acc2 + acc3));
    while (col < kv_len) : (col += 1) {
        val = @mulAdd(f32, matrix[row_base + col], q[col], val);
    }
    return val;
}

const ConvSiluCtx = struct {
    conv_kernel: []const f32,
    conv_state: []const f32,
    qkv: []const f32,
    conv_out: []f32,
    d_conv: u32,
    d_conv_1: u32,
    conv_ch: u32,
    kernel_transposed_d4: bool,
};

const ConvSiluWorker = struct {
    ctx: *const ConvSiluCtx,
    ch_start: u32,
    ch_end: u32,
};

const conv_silu_parallel_max_workers: usize = 16;
const conv_silu_parallel_min_channels: u32 = 256;

fn convSiluWorkerMain(worker: *ConvSiluWorker) void {
    runConvSiluRange(worker.ctx, worker.ch_start, worker.ch_end);
}

fn convSiluWorkerTask(ctx: *anyopaque) void {
    const worker: *ConvSiluWorker = @ptrCast(@alignCast(ctx));
    runConvSiluRange(worker.ctx, worker.ch_start, worker.ch_end);
}

fn runConvSiluRange(ctx: *const ConvSiluCtx, ch_start: u32, ch_end: u32) void {
    const d_conv = ctx.d_conv;
    const d_conv_1 = ctx.d_conv_1;
    const conv_ch = ctx.conv_ch;
    if (d_conv == 4 and d_conv_1 == 3) {
        runConvSiluRangeDConv4(ctx, ch_start, ch_end);
        return;
    }

    var ch: u32 = ch_start;
    while (ch < ch_end) : (ch += 1) {
        var sum: f32 = 0;
        var ki: u32 = 0;
        while (ki < d_conv) : (ki += 1) {
            const kw = ctx.conv_kernel[ch * d_conv + ki];
            const sv = if (ki < d_conv_1) ctx.conv_state[ki * conv_ch + ch] else ctx.qkv[ch];
            sum += kw * sv;
        }
        ctx.conv_out[ch] = sum * sigmoid(sum);
    }
}

const ConvSiluVec16f = @Vector(16, f32);

inline fn convKernelD4Vec(conv_kernel: []const f32, ch: usize, comptime ki: usize) ConvSiluVec16f {
    const vals: [16]f32 = .{
        conv_kernel[(ch + 0) * 4 + ki],
        conv_kernel[(ch + 1) * 4 + ki],
        conv_kernel[(ch + 2) * 4 + ki],
        conv_kernel[(ch + 3) * 4 + ki],
        conv_kernel[(ch + 4) * 4 + ki],
        conv_kernel[(ch + 5) * 4 + ki],
        conv_kernel[(ch + 6) * 4 + ki],
        conv_kernel[(ch + 7) * 4 + ki],
        conv_kernel[(ch + 8) * 4 + ki],
        conv_kernel[(ch + 9) * 4 + ki],
        conv_kernel[(ch + 10) * 4 + ki],
        conv_kernel[(ch + 11) * 4 + ki],
        conv_kernel[(ch + 12) * 4 + ki],
        conv_kernel[(ch + 13) * 4 + ki],
        conv_kernel[(ch + 14) * 4 + ki],
        conv_kernel[(ch + 15) * 4 + ki],
    };
    return vals;
}

fn runConvSiluRangeDConv4(ctx: *const ConvSiluCtx, ch_start: u32, ch_end: u32) void {
    if (ctx.kernel_transposed_d4) {
        runConvSiluRangeDConv4Transposed(ctx, ch_start, ch_end);
        return;
    }

    const conv_ch: usize = @intCast(ctx.conv_ch);
    const one: ConvSiluVec16f = @splat(1.0);
    var ch: usize = @intCast(ch_start);
    const end: usize = @intCast(ch_end);
    while (ch + 16 <= end) : (ch += 16) {
        const s0: ConvSiluVec16f = ctx.conv_state[ch..][0..16].*;
        const s1: ConvSiluVec16f = ctx.conv_state[conv_ch + ch ..][0..16].*;
        const s2: ConvSiluVec16f = ctx.conv_state[2 * conv_ch + ch ..][0..16].*;
        const x: ConvSiluVec16f = ctx.qkv[ch..][0..16].*;
        var sum = convKernelD4Vec(ctx.conv_kernel, ch, 0) * s0;
        sum = @mulAdd(ConvSiluVec16f, convKernelD4Vec(ctx.conv_kernel, ch, 1), s1, sum);
        sum = @mulAdd(ConvSiluVec16f, convKernelD4Vec(ctx.conv_kernel, ch, 2), s2, sum);
        sum = @mulAdd(ConvSiluVec16f, convKernelD4Vec(ctx.conv_kernel, ch, 3), x, sum);
        const sig = one / (one + @exp(-sum));
        ctx.conv_out[ch..][0..16].* = sum * sig;
    }
    while (ch < end) : (ch += 1) {
        const kb = ch * 4;
        var sum = ctx.conv_kernel[kb + 3] * ctx.qkv[ch];
        sum = @mulAdd(f32, ctx.conv_kernel[kb + 2], ctx.conv_state[2 * conv_ch + ch], sum);
        sum = @mulAdd(f32, ctx.conv_kernel[kb + 1], ctx.conv_state[conv_ch + ch], sum);
        sum = @mulAdd(f32, ctx.conv_kernel[kb + 0], ctx.conv_state[ch], sum);
        ctx.conv_out[ch] = sum * sigmoid(sum);
    }
}

fn runConvSiluRangeDConv4Transposed(ctx: *const ConvSiluCtx, ch_start: u32, ch_end: u32) void {
    const conv_ch: usize = @intCast(ctx.conv_ch);
    const one: ConvSiluVec16f = @splat(1.0);
    var ch: usize = @intCast(ch_start);
    const end: usize = @intCast(ch_end);
    while (ch + 16 <= end) : (ch += 16) {
        const s0: ConvSiluVec16f = ctx.conv_state[ch..][0..16].*;
        const s1: ConvSiluVec16f = ctx.conv_state[conv_ch + ch ..][0..16].*;
        const s2: ConvSiluVec16f = ctx.conv_state[2 * conv_ch + ch ..][0..16].*;
        const x: ConvSiluVec16f = ctx.qkv[ch..][0..16].*;
        var sum = ctx.conv_kernel[ch..][0..16].* * s0;
        sum = @mulAdd(ConvSiluVec16f, ctx.conv_kernel[conv_ch + ch ..][0..16].*, s1, sum);
        sum = @mulAdd(ConvSiluVec16f, ctx.conv_kernel[2 * conv_ch + ch ..][0..16].*, s2, sum);
        sum = @mulAdd(ConvSiluVec16f, ctx.conv_kernel[3 * conv_ch + ch ..][0..16].*, x, sum);
        const sig = one / (one + @exp(-sum));
        ctx.conv_out[ch..][0..16].* = sum * sig;
    }
    while (ch < end) : (ch += 1) {
        var sum = ctx.conv_kernel[3 * conv_ch + ch] * ctx.qkv[ch];
        sum = @mulAdd(f32, ctx.conv_kernel[2 * conv_ch + ch], ctx.conv_state[2 * conv_ch + ch], sum);
        sum = @mulAdd(f32, ctx.conv_kernel[conv_ch + ch], ctx.conv_state[conv_ch + ch], sum);
        sum = @mulAdd(f32, ctx.conv_kernel[ch], ctx.conv_state[ch], sum);
        ctx.conv_out[ch] = sum * sigmoid(sum);
    }
}

fn runConvSiluParallel(pool: ?*thread_pool.Pool, ctx: *const ConvSiluCtx) void {
    const conv_ch = ctx.conv_ch;
    // Prefer FastPool — fires once per SSM layer (30/token); cheap barriers add up.
    if (pool != null) {
        if (matvec_fast_pool) |fp| {
            const executors = fp.executorCount();
            if (conv_ch >= conv_silu_parallel_min_channels and executors > 1) {
                const workers: u32 = @intCast(@min(@as(usize, conv_ch), @min(executors, conv_silu_parallel_max_workers)));
                const channels_per: u32 = (conv_ch + workers - 1) / workers;
                var params: [conv_silu_parallel_max_workers]ConvSiluWorker = undefined;
                var tasks: [conv_silu_parallel_max_workers]zinc_rt.fast_pool.Task = undefined;
                var dispatched: u32 = 0;
                while (dispatched < workers) : (dispatched += 1) {
                    const start = dispatched * channels_per;
                    if (start >= conv_ch) break;
                    const end = @min(conv_ch, (dispatched + 1) * channels_per);
                    params[dispatched] = .{ .ctx = ctx, .ch_start = start, .ch_end = end };
                    tasks[dispatched] = .{ .fn_ = convSiluWorkerTask, .ctx = @ptrCast(&params[dispatched]) };
                }
                fp.dispatchAndRun(tasks[0..dispatched]);
                return;
            }
        }
    }
    if (pool) |p| {
        const executors = poolExecutorCount(p);
        if (conv_ch >= conv_silu_parallel_min_channels and executors > 1) {
            const workers: u32 = @intCast(@min(@as(usize, conv_ch), @min(executors, conv_silu_parallel_max_workers)));
            const channels_per: u32 = (conv_ch + workers - 1) / workers;
            var params: [conv_silu_parallel_max_workers]ConvSiluWorker = undefined;
            var wg: thread_pool.WaitGroup = .{};
            var dispatched: u32 = 0;
            while (dispatched < workers) : (dispatched += 1) {
                const start = dispatched * channels_per;
                if (start >= conv_ch) break;
                const end = @min(conv_ch, (dispatched + 1) * channels_per);
                params[dispatched] = .{ .ctx = ctx, .ch_start = start, .ch_end = end };
                p.spawnWg(&wg, convSiluWorkerMain, .{&params[dispatched]});
            }
            p.waitAndWork(&wg);
            return;
        }
    }
    runConvSiluRange(ctx, 0, conv_ch);
}

fn runMoeLayer(
    model: *const Model,
    state: *ScalarDecodeState,
    lt: LayerTensors,
    direct_compute_tracking: ?DirectComputeTracking,
) !void {
    const cfg = model.config;
    // Gemma 4 MoE routes the expert input through pre_ffw_norm_2 instead of
    // the standard ffn_norm. Vulkan path mirrors this at forward.zig:6975.
    const ffn_norm = if (cfg.is_gemma and lt.pre_ffw_norm_2 != null)
        lt.pre_ffw_norm_2.?
    else
        lt.ffn_norm orelse lt.post_attention_norm orelse return error.TensorNotFound;
    try rmsNormTensor(model, ffn_norm, state.hidden, state.ffn_norm, state.row_scratch);
    const n_used = state.moe_topk_active;
    const route_experts = n_used > 0;
    var shared_gate_scalar = [_]f32{0};
    var shared_scale: f32 = 1.0;
    var routed_top1 = false;
    if (lt.ffn_gate_inp_shexp) |shared_gate_inp| {
        if (state.decode_phase and n_used == 1) {
            const router_w = model.requantOrRaw(lt.ffn_gate_inp.?);
            const shared_gate_w = model.requantOrRaw(shared_gate_inp);
            if (canDotDirect(router_w.type_, @intCast(state.ffn_norm.len)) and
                canDotDirect(shared_gate_w.type_, @intCast(state.ffn_norm.len)))
            {
                if (routeTop1SharedGate(
                    state.pool,
                    router_w,
                    cfg.n_experts,
                    shared_gate_w,
                    state.ffn_norm,
                    state.row_scratch,
                )) |route| {
                    state.expert_ids[0] = route.expert_id;
                    state.expert_weights[0] = 1.0;
                    shared_scale = sigmoid(route.shared_gate);
                    routed_top1 = true;
                } else |_| {}
            }
        }
        if (!routed_top1) {
            if (route_experts) {
                try matvecFusedTensors(state.pool, model, &[_]FusedPart{
                    .{ .info = lt.ffn_gate_inp.?, .rows = cfg.n_experts, .out = state.router_logits },
                    .{ .info = shared_gate_inp, .rows = 1, .out = shared_gate_scalar[0..] },
                }, state.ffn_norm, state.row_scratch);
                consumeDirectRouterRowRange(model, state, lt, direct_compute_tracking);
            } else {
                try matvecTensor(state.pool, model, shared_gate_inp, state.ffn_norm, 1, state.row_scratch, shared_gate_scalar[0..]);
            }
            shared_scale = sigmoid(shared_gate_scalar[0]);
        }
    } else {
        if (state.decode_phase and n_used == 1) {
            const router_w = model.requantOrRaw(lt.ffn_gate_inp.?);
            if (canDotDirect(router_w.type_, @intCast(state.ffn_norm.len))) {
                if (argmaxMatvecRaw(state.pool, router_w.raw, router_w.type_, state.ffn_norm, cfg.n_experts, state.row_scratch)) |expert_id| {
                    state.expert_ids[0] = expert_id;
                    state.expert_weights[0] = 1.0;
                    routed_top1 = true;
                } else |_| {}
            }
        }
        if (!routed_top1 and route_experts) {
            try matvecTensor(state.pool, model, lt.ffn_gate_inp.?, state.ffn_norm, cfg.n_experts, state.row_scratch, state.router_logits);
            consumeDirectRouterRowRange(model, state, lt, direct_compute_tracking);
        }
    }
    if (!routed_top1 and route_experts) topKSoftmaxCpu(state.router_logits, n_used, state.expert_ids, state.expert_weights);

    const gate_exps = lt.ffn_gate_exps.?;
    const up_exps = lt.ffn_up_exps.?;
    const down_exps = lt.ffn_down_exps.?;
    const gate_w = if (state.decode_phase) model.decodeRequantOrRaw(gate_exps) else model.requantOrRaw(gate_exps);
    const up_w = if (state.decode_phase) model.decodeRequantOrRaw(up_exps) else model.requantOrRaw(up_exps);
    const down_w = model.requantOrRaw(down_exps);
    const gate_raw = gate_w.raw;
    const up_raw = up_w.raw;
    const down_raw = down_w.raw;
    const gate_type = gate_w.type_;
    const up_type = up_w.type_;
    const down_type = down_w.type_;
    const effective_gate_stride = expertSliceBytes(gate_type, cfg.intermediate_dim, cfg.hidden_dim);
    const effective_up_stride = expertSliceBytes(up_type, cfg.intermediate_dim, cfg.hidden_dim);
    const effective_down_stride = expertSliceBytes(down_type, cfg.hidden_dim, cfg.intermediate_dim);

    // The Qwen 3.5-MoE shared expert (same gate/up·swiglu·down shape, always
    // active, sigmoid-scaled) reads ~hidden·shexp_inter quantised weights per
    // layer. Run by itself that stream sits on the main thread while the decode
    // pool idles after the routed-expert fan-out — fold it into the same fan-out
    // as an extra task so all the expert weight streams overlap across cores.
    var shared_params: ?MoeSharedParams = null;
    if (lt.ffn_gate_shexp != null and lt.ffn_up_shexp != null and lt.ffn_down_shexp != null) {
        const gate_t = lt.ffn_gate_shexp.?;
        const up_t = lt.ffn_up_shexp.?;
        const down_t = lt.ffn_down_shexp.?;
        // Substitute the load-time Q4_0 re-quantisations when present (the shared
        // expert isn't routed through `matvecTensor`, so it picks them up here).
        const shared_gate_w = model.requantOrRaw(gate_t);
        const shared_up_w = model.requantOrRaw(up_t);
        const shared_down_w = model.requantOrRaw(down_t);
        shared_params = .{
            .gate_raw = shared_gate_w.raw,
            .gate_type = shared_gate_w.type_,
            .up_raw = shared_up_w.raw,
            .up_type = shared_up_w.type_,
            .down_raw = shared_down_w.raw,
            .down_type = shared_down_w.type_,
            .inter = @intCast(gate_t.numElements() / cfg.hidden_dim),
            .scale = shared_scale,
        };
    }

    if (!route_experts) {
        if (shared_params) |sp| {
            try runSharedExpertOnly(state, cfg, sp);
        }
        return;
    }

    const ran_parallel = runMoeExpertsParallel(
        state,
        cfg,
        n_used,
        gate_raw,
        gate_type,
        effective_gate_stride,
        up_raw,
        up_type,
        effective_up_stride,
        down_raw,
        down_type,
        effective_down_stride,
        shared_params,
    );

    if (ran_parallel) {
        // The fan-out folded the routed experts and the shared expert directly
        // into the layer residual.
        return;
    }

    // Serial fallback: routed experts, then the shared expert.
    for (0..n_used) |ei| {
        const eid = state.expert_ids[ei];
        const weight = state.expert_weights[ei];
        const gate_off: usize = @intCast(eid * effective_gate_stride);
        const up_off: usize = @intCast(eid * effective_up_stride);
        const down_off: usize = @intCast(eid * effective_down_stride);
        try runMoeExpert(
            gate_raw[gate_off..],
            gate_type,
            up_raw[up_off..],
            up_type,
            down_raw[down_off..],
            down_type,
            state.ffn_norm,
            cfg.intermediate_dim,
            cfg.hidden_dim,
            state.row_scratch,
            state.gate[0..cfg.intermediate_dim],
            state.up[0..cfg.intermediate_dim],
            state.swiglu[0..cfg.intermediate_dim],
            state.down,
            cfg.is_gemma,
        );
        for (state.hidden, state.down) |*h, v| h.* += weight * v;
    }
    if (shared_params) |sp| {
        const inter: usize = @intCast(sp.inter);
        try runMoeExpert(
            sp.gate_raw,
            sp.gate_type,
            sp.up_raw,
            sp.up_type,
            sp.down_raw,
            sp.down_type,
            state.ffn_norm,
            sp.inter,
            cfg.hidden_dim,
            state.row_scratch,
            state.gate[0..inter],
            state.up[0..inter],
            state.swiglu[0..inter],
            state.down,
            cfg.is_gemma,
        );
        for (state.hidden, state.down) |*h, v| h.* += sp.scale * v;
    }
}

/// Gemma 4 MoE forward path. Each layer runs two parallel branches over the
/// post-attention residual stream: a shared dense MLP (`ffn_gate`/`ffn_up`/
/// `ffn_down` with GELU, normalized by `post_ffw_norm_1`) and a routed expert
/// branch (input normalized by `pre_ffw_norm_2`, weights live in fused
/// `ffn_gate_up_exps` plus `ffn_down_exps`, output normalized by
/// `post_ffw_norm_2`). The router input is a custom pipeline:
/// `rms_norm_no_weight(attn_out) * (1/sqrt(hidden_dim)) * ffn_gate_inp.scale`,
/// then matmul with `ffn_gate_inp` and a top-k softmax. The combined branch
/// output runs through `post_ffw_norm` and is added back to the saved
/// `attn_out` residual.
fn runGemma4MoeLayer(
    model: *const Model,
    state: *ScalarDecodeState,
    lt: LayerTensors,
) !void {
    const cfg = model.config;
    const hidden_dim: u32 = @intCast(cfg.hidden_dim);
    const shared_inter: u32 = @intCast(cfg.shared_expert_intermediate_dim);
    const expert_inter: u32 = @intCast(cfg.intermediate_dim);

    const ffn_norm_t = lt.ffn_norm orelse return error.TensorNotFound;
    const pre_ffw_norm_2_t = lt.pre_ffw_norm_2 orelse return error.TensorNotFound;
    const post_ffw_norm_1_t = lt.ffn_post_norm_1 orelse return error.TensorNotFound;
    const post_ffw_norm_2_t = lt.ffn_post_norm_2 orelse return error.TensorNotFound;
    const post_ffw_norm_t = lt.post_ffw_norm orelse return error.TensorNotFound;
    const ffn_gate_t = lt.ffn_gate orelse return error.TensorNotFound;
    const ffn_up_t = lt.ffn_up orelse return error.TensorNotFound;
    const ffn_down_t = lt.ffn_down orelse return error.TensorNotFound;
    const ffn_gate_inp_t = lt.ffn_gate_inp orelse return error.TensorNotFound;
    const ffn_gate_inp_scale_t = lt.ffn_gate_inp_scale orelse return error.TensorNotFound;
    const ffn_gate_up_exps_t = lt.ffn_gate_up_exps orelse return error.TensorNotFound;
    const ffn_down_exps_t = lt.ffn_down_exps orelse return error.TensorNotFound;

    // Save attn_out into state.branch (free after the attention block).
    @memcpy(state.branch, state.hidden);

    // ============ Path A: shared dense MLP ============
    try rmsNormTensor(model, ffn_norm_t, state.branch, state.ffn_norm, state.row_scratch);
    try matvecFusedTensors(state.pool, model, &[_]FusedPart{
        .{ .info = ffn_gate_t, .rows = shared_inter, .out = state.gate[0..shared_inter] },
        .{ .info = ffn_up_t, .rows = shared_inter, .out = state.up[0..shared_inter] },
    }, state.ffn_norm, state.row_scratch);
    geluGate(state.gate[0..shared_inter], state.up[0..shared_inter], state.swiglu[0..shared_inter]);
    try matvecTensor(state.pool, model, ffn_down_t, state.swiglu[0..shared_inter], hidden_dim, state.row_scratch, state.down);
    try rmsNormTensor(model, post_ffw_norm_1_t, state.down, state.down, state.row_scratch);
    @memcpy(state.mlp_save, state.down);

    // ============ Path B: routed MoE ============
    // Input to the experts: pre_ffw_norm_2(attn_out).
    try rmsNormTensor(model, pre_ffw_norm_2_t, state.branch, state.ffn_norm, state.row_scratch);

    // Router input pipeline.
    const router_input = state.norm; // hidden_dim scratch — state.norm is unused until next layer's attn_norm runs.
    const inv_rms = rmsNormInvRms(state.branch, cfg.rms_norm_eps);
    const inv_sqrt_n: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(hidden_dim)));
    const gate_inp_scale = model.cachedF32(ffn_gate_inp_scale_t) orelse return error.TensorNotFound;
    if (gate_inp_scale.len < hidden_dim) return error.ShapeMismatch;
    for (router_input, state.branch, gate_inp_scale[0..hidden_dim]) |*ri, src, s| {
        ri.* = src * inv_rms * inv_sqrt_n * s;
    }

    try matvecTensor(state.pool, model, ffn_gate_inp_t, router_input, cfg.n_experts, state.row_scratch, state.router_logits);
    const n_used = cfg.n_experts_used;
    topKSoftmaxCpu(state.router_logits, n_used, state.expert_ids, state.expert_weights);

    // Expert dispatch — per-expert byte strides on the fused gate+up tensor and
    // the separate down tensor. Each expert's fused slice contains
    // `expert_inter` gate rows followed by `expert_inter` up rows, both
    // hidden_dim columns; we matvec each half separately into state.gate /
    // state.up rather than allocating a 2*inter scratch.
    const fused_raw = model.tensorData(ffn_gate_up_exps_t);
    const fused_type = ffn_gate_up_exps_t.type_;
    const fused_expert_stride = expertSliceBytes(fused_type, 2 * expert_inter, hidden_dim);
    const fused_gate_half_bytes = expertSliceBytes(fused_type, expert_inter, hidden_dim);
    const down_raw = model.tensorData(ffn_down_exps_t);
    const down_type = ffn_down_exps_t.type_;
    const down_expert_stride = expertSliceBytes(down_type, hidden_dim, expert_inter);

    @memset(state.down, 0);
    const expert_out = router_input; // reuse router_input scratch (hidden_dim).
    for (0..n_used) |ei| {
        const expert_id = state.expert_ids[ei];
        const w = state.expert_weights[ei];
        const fused_off: usize = @as(usize, expert_id) * @as(usize, fused_expert_stride);
        const gate_slice = fused_raw[fused_off..];
        try matvecRaw(state.pool, gate_slice, fused_type, state.ffn_norm, expert_inter, state.row_scratch, state.gate[0..expert_inter]);
        const up_slice = fused_raw[fused_off + fused_gate_half_bytes ..];
        try matvecRaw(state.pool, up_slice, fused_type, state.ffn_norm, expert_inter, state.row_scratch, state.up[0..expert_inter]);
        geluGate(state.gate[0..expert_inter], state.up[0..expert_inter], state.swiglu[0..expert_inter]);
        const down_off: usize = @as(usize, expert_id) * @as(usize, down_expert_stride);
        try matvecRaw(state.pool, down_raw[down_off..], down_type, state.swiglu[0..expert_inter], hidden_dim, state.row_scratch, expert_out);
        for (state.down, expert_out) |*acc, v| acc.* += w * v;
    }
    try rmsNormTensor(model, post_ffw_norm_2_t, state.down, state.down, state.row_scratch);

    // ============ Combine branches and final post-FFN norm ============
    for (state.down, state.mlp_save) |*acc, mlp| acc.* += mlp;
    try rmsNormTensor(model, post_ffw_norm_t, state.down, state.down, state.row_scratch);

    // Residual add — state.hidden = saved attn_out + combined branch output.
    for (state.hidden, state.branch, state.down) |*h, attn, ffn| h.* = attn + ffn;
}

const direct_router_row_range_max_rows: u32 = 128;
const direct_router_row_range_tolerance: f32 = 0.01;

fn consumeDirectRouterRowRange(
    model: *const Model,
    state: *ScalarDecodeState,
    lt: LayerTensors,
    maybe_tracking: ?DirectComputeTracking,
) void {
    const tracking = maybe_tracking orelse return;
    if (state.direct_router_row_range_done) return;
    const router_info = lt.ffn_gate_inp orelse return;
    const cfg = model.config;
    const cols: u32 = @intCast(state.ffn_norm.len);
    if (cols == 0 or cols % 64 != 0) return;
    if (router_info.type_ != .f32) return;
    const router_raw = model.tensorData(router_info);

    const rows = @min(cfg.n_experts, direct_router_row_range_max_rows);
    const rows_usize: usize = @intCast(rows);
    if (rows == 0 or state.row_scratch.len < rows_usize * 2 or state.router_logits.len < rows_usize) return;
    const row_bytes: usize = @as(usize, cols) * @sizeOf(f32);
    const total_bytes = @as(usize, rows) * row_bytes;
    if (router_raw.len < total_bytes) return;

    const gpu_logits = state.row_scratch[0..rows_usize];
    const cpu_logits = state.row_scratch[rows_usize..][0..rows_usize];
    matvecRawDirectSerial(router_raw, .f32, state.ffn_norm, null, 0, rows, cpu_logits) catch {
        return;
    };
    tracking.boundary.dmmvF32RowRange(
        state.ffn_norm,
        router_raw[0..total_bytes],
        rows,
        cols,
        gpu_logits,
    ) catch |err| {
        log.warn("M1 AMDGPU CS direct router row-range unavailable ({s}); router logits remain host-computed", .{@errorName(err)});
        return;
    };

    var max_abs_delta: f32 = 0.0;
    var max_row: u32 = 0;
    for (gpu_logits, 0..) |gpu_value, i| {
        if (!std.math.isFinite(gpu_value)) {
            log.warn("M1 AMDGPU CS direct router row-range produced non-finite row {d}; router logits remain host-computed", .{i});
            return;
        }
        const delta = @abs(gpu_value - cpu_logits[i]);
        if (delta > max_abs_delta) {
            max_abs_delta = delta;
            max_row = @intCast(i);
        }
    }
    if (max_abs_delta > direct_router_row_range_tolerance) {
        log.warn("M1 AMDGPU CS direct router row-range mismatch: rows={d} max_abs_delta={d:.6} row={d}; router logits remain host-computed", .{
            rows,
            max_abs_delta,
            max_row,
        });
        return;
    }

    @memcpy(state.router_logits[0..rows_usize], gpu_logits);
    state.direct_router_row_range_done = true;
    tracking.ops.* += 1;
    mergeDirectComputeKind(tracking.kind, .dmmv_row_range);
    tracking.consumed.* = true;
    tracking.real_model_slice.* = true;
    log.info("M1 AMDGPU CS direct model slice consumed: direct_compute_ops={d} direct_compute_kind=dmmv_row_range op=router_f32_row_range rows={d} cols={d} max_abs_delta={d:.6} max_row={d}", .{
        tracking.ops.*,
        rows,
        cols,
        max_abs_delta,
        max_row,
    });
}

const moe_expert_parallel_max_workers: usize = 8;
const moe_expert_parallel_max_segments: usize = moe_expert_parallel_max_workers * 2;

const MoeExpertWorker = struct {
    gate_raw: []const u8,
    gate_type: gguf.GGMLType,
    up_raw: []const u8,
    up_type: gguf.GGMLType,
    down_raw: []const u8,
    down_type: gguf.GGMLType,
    ffn_norm: []const f32,
    intermediate_dim: u32,
    hidden_dim: u32,
    scratch: []f32,
    gate: []f32,
    up: []f32,
    swiglu_out: []f32,
    down: []f32,
    // Gemma 4 experts run with gelu(gate) * up instead of silu(gate) * up.
    use_gelu: bool = false,
    failed: bool = false,
};

/// Shared-expert weights/scale handed to `runMoeExpertsParallel` so the always-
/// active expert rides the same decode-pool fan-out as the routed experts.
const MoeSharedParams = struct {
    gate_raw: []const u8,
    gate_type: gguf.GGMLType,
    up_raw: []const u8,
    up_type: gguf.GGMLType,
    down_raw: []const u8,
    down_type: gguf.GGMLType,
    inter: u32,
    scale: f32,
};

fn runMoeExpertsParallel(
    state: *ScalarDecodeState,
    cfg: CpuModelConfig,
    n_used: u32,
    gate_raw: []const u8,
    gate_type: gguf.GGMLType,
    gate_stride: u32,
    up_raw: []const u8,
    up_type: gguf.GGMLType,
    up_stride: u32,
    down_raw: []const u8,
    down_type: gguf.GGMLType,
    down_stride: u32,
    shared: ?MoeSharedParams,
) bool {
    const worker_count = state.moe_expert_workers;
    if (worker_count != n_used) return false;

    const inter_len: usize = @intCast(cfg.intermediate_dim);
    const inter_stride: usize = @intCast(@max(cfg.intermediate_dim, cfg.shared_expert_intermediate_dim));
    const scratch_stride: usize = @intCast(@max(cfg.hidden_dim, @max(cfg.intermediate_dim, cfg.shared_expert_intermediate_dim)));
    const down_stride_f32: usize = @intCast(cfg.hidden_dim);
    const max_slots: usize = if (down_stride_f32 == 0) 0 else state.moe_worker_down.len / down_stride_f32;

    // The shared expert, when present, becomes task `worker_count` in slot
    // `worker_count`. Bail to the serial fallback if there is no spare worker
    // slot / params entry for it or its intermediate width overruns a slot.
    var shared_inter: usize = 0;
    if (shared) |sp| {
        shared_inter = @intCast(sp.inter);
        if (shared_inter > inter_stride or shared_inter > scratch_stride) return false;
    }
    const task_count: usize = worker_count + @as(usize, if (shared != null) 1 else 0);
    if (task_count == 0 or task_count > moe_expert_parallel_max_workers or task_count > max_slots) return false;

    var params: [moe_expert_parallel_max_workers]MoeExpertWorker = undefined;
    for (0..worker_count) |i| {
        const eid = state.expert_ids[i];
        const gate_off: usize = @intCast(eid * gate_stride);
        const up_off: usize = @intCast(eid * up_stride);
        const down_off: usize = @intCast(eid * down_stride);
        params[i] = .{
            .gate_raw = gate_raw[gate_off..],
            .gate_type = gate_type,
            .up_raw = up_raw[up_off..],
            .up_type = up_type,
            .down_raw = down_raw[down_off..],
            .down_type = down_type,
            .ffn_norm = state.ffn_norm,
            .intermediate_dim = cfg.intermediate_dim,
            .hidden_dim = cfg.hidden_dim,
            .scratch = slotSlice(state.moe_worker_scratch, i, scratch_stride, @max(@as(usize, @intCast(cfg.hidden_dim)), inter_len)),
            .gate = slotSlice(state.moe_worker_gate, i, inter_stride, inter_len),
            .up = slotSlice(state.moe_worker_up, i, inter_stride, inter_len),
            .swiglu_out = slotSlice(state.moe_worker_swiglu, i, inter_stride, inter_len),
            .down = slotSlice(state.moe_worker_down, i, down_stride_f32, down_stride_f32),
            .use_gelu = cfg.is_gemma,
        };
    }
    if (shared) |sp| {
        const i = worker_count;
        params[i] = .{
            .gate_raw = sp.gate_raw,
            .gate_type = sp.gate_type,
            .up_raw = sp.up_raw,
            .up_type = sp.up_type,
            .down_raw = sp.down_raw,
            .down_type = sp.down_type,
            .ffn_norm = state.ffn_norm,
            .intermediate_dim = sp.inter,
            .hidden_dim = cfg.hidden_dim,
            .scratch = slotSlice(state.moe_worker_scratch, i, scratch_stride, @max(@as(usize, @intCast(cfg.hidden_dim)), shared_inter)),
            .gate = slotSlice(state.moe_worker_gate, i, inter_stride, shared_inter),
            .up = slotSlice(state.moe_worker_up, i, inter_stride, shared_inter),
            .swiglu_out = slotSlice(state.moe_worker_swiglu, i, inter_stride, shared_inter),
            .down = slotSlice(state.moe_worker_down, i, down_stride_f32, down_stride_f32),
            .use_gelu = cfg.is_gemma,
        };
    }

    const ran_phased = state.decode_phase and runMoeExpertsParallelPhased(state, params[0..task_count]);
    if (!ran_phased and state.pool != null) {
        const pool = state.pool.?;
        var wg: thread_pool.WaitGroup = .{};
        for (0..task_count) |i| pool.spawnWg(&wg, moeExpertWorkerMain, .{&params[i]});
        pool.waitAndWork(&wg);
    } else if (!ran_phased) {
        var threads: [moe_expert_parallel_max_workers]std.Thread = undefined;
        var spawned: usize = 0;
        var spawn_failed = false;
        while (spawned < task_count) : (spawned += 1) {
            threads[spawned] = std.Thread.spawn(.{}, moeExpertWorkerMain, .{&params[spawned]}) catch {
                spawn_failed = true;
                break;
            };
        }
        for (threads[0..spawned]) |thread| thread.join();
        if (spawn_failed) return false;
    }

    for (params[0..task_count]) |param| {
        if (param.failed) return false;
    }

    if (shared) |sp| {
        const shared_down = slotSlice(state.moe_worker_down, worker_count, down_stride_f32, down_stride_f32);
        if (worker_count == 1) {
            const routed_down = slotSlice(state.moe_worker_down, 0, down_stride_f32, down_stride_f32);
            const weight = state.expert_weights[0];
            for (state.hidden, routed_down, shared_down) |*h, routed_v, shared_v| {
                h.* += weight * routed_v + sp.scale * shared_v;
            }
            return true;
        }
        var routed_downs: [moe_expert_parallel_max_workers][]f32 = undefined;
        for (0..worker_count) |ei| {
            routed_downs[ei] = slotSlice(state.moe_worker_down, ei, down_stride_f32, down_stride_f32);
        }
        for (state.hidden, 0..) |*h, i| {
            var sum = sp.scale * shared_down[i];
            for (0..worker_count) |ei| {
                sum += state.expert_weights[ei] * routed_downs[ei][i];
            }
            h.* += sum;
        }
    } else {
        for (0..worker_count) |ei| {
            const weight = state.expert_weights[ei];
            const down = slotSlice(state.moe_worker_down, ei, down_stride_f32, down_stride_f32);
            for (state.hidden, down) |*h, v| h.* += weight * v;
        }
    }
    return true;
}

fn moeExpertWorkerCount(n_used: u32, cpu_count: usize) usize {
    if (n_used < 2 or cpu_count < 2) return 1;
    const used: usize = @intCast(n_used);
    if (used > moe_expert_parallel_max_workers) return 1;
    if (used > cpu_count) return 1;
    return used;
}

fn moeExpertWorkerMain(params: *MoeExpertWorker) void {
    runMoeExpert(
        params.gate_raw,
        params.gate_type,
        params.up_raw,
        params.up_type,
        params.down_raw,
        params.down_type,
        params.ffn_norm,
        params.intermediate_dim,
        params.hidden_dim,
        params.scratch,
        params.gate,
        params.up,
        params.swiglu_out,
        params.down,
        params.use_gelu,
    ) catch {
        params.failed = true;
    };
}

fn runSharedExpertOnly(state: *ScalarDecodeState, cfg: CpuModelConfig, sp: MoeSharedParams) !void {
    const inter: usize = @intCast(sp.inter);
    if (inter == 0) return;

    if (canDotDirect(sp.gate_type, cfg.hidden_dim) and canDotDirect(sp.up_type, cfg.hidden_dim)) {
        var segs = [_]FusedSegment{
            .{ .raw = sp.gate_raw, .type_ = sp.gate_type, .rows = sp.inter, .out = state.gate[0..inter] },
            .{ .raw = sp.up_raw, .type_ = sp.up_type, .rows = sp.inter, .out = state.up[0..inter] },
        };
        const input_sum32 = sums: {
            if ((wantsInputSum32(sp.gate_type) or wantsInputSum32(sp.up_type)) and
                state.ffn_norm.len % 32 == 0 and
                state.row_scratch.len >= state.ffn_norm.len / 32)
            {
                const sums = state.row_scratch[0 .. state.ffn_norm.len / 32];
                dequant.fillInputSum32(state.ffn_norm, sums);
                break :sums sums;
            }
            break :sums null;
        };
        try matvecFused(state.pool, segs[0..], state.ffn_norm, input_sum32);
    } else {
        try matvecRaw(state.pool, sp.gate_raw, sp.gate_type, state.ffn_norm, sp.inter, state.row_scratch, state.gate[0..inter]);
        try matvecRaw(state.pool, sp.up_raw, sp.up_type, state.ffn_norm, sp.inter, state.row_scratch, state.up[0..inter]);
    }

    if (cfg.is_gemma) {
        geluGate(state.gate[0..inter], state.up[0..inter], state.swiglu[0..inter]);
    } else {
        swiglu(state.gate[0..inter], state.up[0..inter], state.swiglu[0..inter]);
    }
    try matvecRaw(state.pool, sp.down_raw, sp.down_type, state.swiglu[0..inter], cfg.hidden_dim, state.row_scratch, state.down);
    for (state.hidden, state.down) |*h, v| h.* += sp.scale * v;
}

fn runMoeExpertsParallelPhased(state: *ScalarDecodeState, params: []MoeExpertWorker) bool {
    const pool = state.pool orelse return false;
    if (params.len < 2 or params.len > moe_expert_parallel_max_workers) return false;

    const ffn_norm = params[0].ffn_norm;
    var requires_input_sum32 = false;
    var wants_input_sum32 = false;
    var gate_up: [moe_expert_parallel_max_segments]FusedSegment = undefined;
    for (params, 0..) |*param, i| {
        if (param.ffn_norm.ptr != ffn_norm.ptr or param.ffn_norm.len != ffn_norm.len) return false;
        if (!canDotDirect(param.gate_type, @intCast(ffn_norm.len)) or !canDotDirect(param.up_type, @intCast(ffn_norm.len))) return false;
        requires_input_sum32 = requires_input_sum32 or needsInputSum32(param.gate_type) or needsInputSum32(param.up_type);
        wants_input_sum32 = wants_input_sum32 or wantsInputSum32(param.gate_type) or wantsInputSum32(param.up_type);
        gate_up[i * 2] = .{ .raw = param.gate_raw, .type_ = param.gate_type, .rows = param.intermediate_dim, .out = param.gate };
        gate_up[i * 2 + 1] = .{ .raw = param.up_raw, .type_ = param.up_type, .rows = param.intermediate_dim, .out = param.up };
    }
    const input_sum32 = sums: {
        if (wants_input_sum32 and ffn_norm.len % 32 == 0 and state.row_scratch.len >= ffn_norm.len / 32) {
            const sums = state.row_scratch[0 .. ffn_norm.len / 32];
            dequant.fillInputSum32(ffn_norm, sums);
            break :sums sums;
        }
        if (requires_input_sum32) return false;
        break :sums null;
    };
    matvecFused(pool, gate_up[0 .. params.len * 2], ffn_norm, input_sum32) catch return false;

    var down: [moe_expert_parallel_max_workers]MultiInputFusedSegment = undefined;
    for (params, 0..) |*param, i| {
        if (param.use_gelu) {
            geluGate(param.gate, param.up, param.swiglu_out);
        } else {
            swiglu(param.gate, param.up, param.swiglu_out);
        }
        if (!canDotDirect(param.down_type, param.intermediate_dim) or needsInputSum32(param.down_type)) return false;
        down[i] = .{
            .raw = param.down_raw,
            .type_ = param.down_type,
            .input = param.swiglu_out,
            .rows = param.hidden_dim,
            .out = param.down,
        };
    }
    matvecMultiInputFused(pool, down[0..params.len]) catch return false;
    return true;
}

fn runMoeExpert(
    gate_raw: []const u8,
    gate_type: gguf.GGMLType,
    up_raw: []const u8,
    up_type: gguf.GGMLType,
    down_raw: []const u8,
    down_type: gguf.GGMLType,
    ffn_norm: []const f32,
    intermediate_dim: u32,
    hidden_dim: u32,
    scratch: []f32,
    gate: []f32,
    up: []f32,
    swiglu_out: []f32,
    down: []f32,
    use_gelu: bool,
) !void {
    // Inner expert matvecs stay serial because these run from pool worker
    // threads; `matvecRawDirect` only parallelizes through an explicit pool.
    if (canDotDirect(gate_type, hidden_dim) and canDotDirect(up_type, hidden_dim)) {
        var segs = [_]FusedSegment{
            .{ .raw = gate_raw, .type_ = gate_type, .rows = intermediate_dim, .out = gate[0..intermediate_dim] },
            .{ .raw = up_raw, .type_ = up_type, .rows = intermediate_dim, .out = up[0..intermediate_dim] },
        };
        const input_sum32 = sums: {
            if ((wantsInputSum32(gate_type) or wantsInputSum32(up_type)) and
                ffn_norm.len % 32 == 0 and
                scratch.len >= ffn_norm.len / 32)
            {
                const sums = scratch[0 .. ffn_norm.len / 32];
                dequant.fillInputSum32(ffn_norm, sums);
                break :sums sums;
            }
            break :sums null;
        };
        try matvecFused(null, segs[0..], ffn_norm, input_sum32);
    } else {
        try matvecRaw(null, gate_raw, gate_type, ffn_norm, intermediate_dim, scratch, gate[0..intermediate_dim]);
        try matvecRaw(null, up_raw, up_type, ffn_norm, intermediate_dim, scratch, up[0..intermediate_dim]);
    }
    if (use_gelu) {
        geluGate(gate[0..intermediate_dim], up[0..intermediate_dim], swiglu_out[0..intermediate_dim]);
    } else {
        swiglu(gate[0..intermediate_dim], up[0..intermediate_dim], swiglu_out[0..intermediate_dim]);
    }
    try matvecRaw(null, down_raw, down_type, swiglu_out[0..intermediate_dim], hidden_dim, scratch, down[0..hidden_dim]);
}

fn slotSlice(buffer: []f32, slot: usize, stride: usize, len: usize) []f32 {
    const off = slot * stride;
    return buffer[off..][0..len];
}

fn isFullAttentionLayer(cfg: CpuModelConfig, layer: u32) bool {
    const interval = if (cfg.full_attn_interval > 0) cfg.full_attn_interval else 1;
    return (layer + 1) % interval == 0;
}

/// Worker-thread count for the persistent decode pool. Three pool workers plus
/// the caller running `waitAndWork` beat the old four-worker default on the
/// 9800X3D RDNA test node after the decode matvec fusion/requant changes: the
/// T-CPU decode path is dominated by streaming quantized matvecs, and extra
/// AVX-512 workers mostly contend for cache and memory bandwidth.
/// `ZINC_RT_CPU_WORKERS` remains as a bring-up knob for quick A/Bs on different
/// hosts.
fn decodeWorkerThreadCount(io: std.Io) usize {
    if (getenv("ZINC_RT_CPU_WORKERS")) |raw| {
        const parsed = std.fmt.parseInt(usize, raw, 10) catch 0;
        if (parsed > 0) return @min(parsed, matvec_parallel_max_workers);
    }
    const logical = std.Thread.getCpuCount() catch return 1;
    if (logical < 2) return 1;
    var physical = logical;
    if (std.Io.Dir.openFileAbsolute(io, "/sys/devices/system/cpu/smt/active", .{})) |file| {
        defer file.close(io);
        var buf: [4]u8 = undefined;
        const n = file.readStreaming(io, &.{&buf}) catch 0;
        if (n >= 1 and buf[0] == '1') physical = @max(@as(usize, 1), logical / 2);
    } else |_| {}
    return @min(physical, 3);
}

fn matvecTensor(
    pool: ?*thread_pool.Pool,
    model: *const Model,
    info: gguf.TensorInfo,
    input: []const f32,
    rows: u32,
    scratch: []f32,
    output: []f32,
) !void {
    const w = model.requantOrRaw(info);
    return matvecRaw(pool, w.raw, w.type_, input, rows, scratch, output[0..rows]);
}

fn matvecRaw(
    pool: ?*thread_pool.Pool,
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    scratch: []f32,
    output: []f32,
) !void {
    if (output.len < rows or scratch.len < input.len) return error.ShapeMismatch;
    const cols: u32 = @intCast(input.len);
    if (canDotDirect(tensor_type, cols)) {
        const input_sum32 = sums: {
            if (wantsInputSum32(tensor_type) and cols % 32 == 0 and scratch.len >= input.len / 32) {
                const sums = scratch[0 .. input.len / 32];
                dequant.fillInputSum32(input, sums);
                break :sums sums;
            }
            break :sums null;
        };
        try matvecRawDirect(pool, raw, tensor_type, input, rows, output[0..rows], input_sum32);
        return;
    }

    const row_buf = scratch[0..input.len];
    for (0..rows) |row| {
        output[row] = try dequant.dotRow(raw, @intCast(row), cols, tensor_type, input, row_buf);
    }
}

fn canDotDirect(tensor_type: gguf.GGMLType, cols: u32) bool {
    return switch (tensor_type) {
        .f32, .f16, .bf16 => true,
        .q4_0, .q8_0 => cols % 32 == 0,
        .q4_k, .q5_k, .q6_k => cols % 256 == 0,
        else => false,
    };
}

// Qwen 3.6's per-layer MoE router is 128 rows x 2048 cols. It appears 40 times
// per token, so it needs to ride the persistent pool instead of staying serial.
const matvec_parallel_min_work_items: u64 = 256 * 1024;
const matvec_parallel_medium_work_items: u64 = 8 * 1024 * 1024;
const matvec_parallel_large_work_items: u64 = 16 * 1024 * 1024;
const matvec_parallel_max_workers: usize = 16;

/// File-local handle for the FastPool used by the decode main thread to fan out
/// matvec, argmax, conv1d/SiLU and SSM-head workloads. `generateScalarHybrid`
/// sets this before the first dispatch and clears it at scope exit. Reads only
/// happen from the main decode thread; worker threads stick to their serial
/// slice (FastPool is single-producer).
var matvec_fast_pool: ?*zinc_rt.fast_pool.FastPool = null;

const MatvecDirectWorker = struct {
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    input_sum32: ?[]const f32,
    output: []f32,
    start_row: u32,
    end_row: u32,
    failed: bool = false,
};

fn matvecRawDirect(
    pool: ?*thread_pool.Pool,
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    output: []f32,
    input_sum32: ?[]const f32,
) !void {
    const cols: u32 = @intCast(input.len);
    if (matvecWorkItems(rows, cols) < matvec_parallel_min_work_items)
        return matvecRawDirectSerial(raw, tensor_type, input, input_sum32, 0, rows, output);
    // FastPool is single-producer: only the main decode thread may dispatch.
    // When `pool == null` we are running inside a worker (e.g. MoE expert
    // fan-out), so fall through to the serial path and never reach FastPool.
    if (pool != null) {
        if (matvec_fast_pool) |fp| return matvecRawDirectFastPooled(fp, raw, tensor_type, input, rows, output, input_sum32);
    }
    if (pool) |p| return matvecRawDirectPooled(p, raw, tensor_type, input, rows, output, input_sum32);
    return matvecRawDirectSerial(raw, tensor_type, input, input_sum32, 0, rows, output);
}

fn matvecRawDirectFastPooled(
    fp: *zinc_rt.fast_pool.FastPool,
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    output: []f32,
    input_sum32: ?[]const f32,
) !void {
    const cols: u32 = @intCast(input.len);
    const executors = fp.executorCount();
    const worker_count = matvecWorkerCountForCpu(rows, cols, executors);
    if (worker_count <= 1)
        return matvecRawDirectSerial(raw, tensor_type, input, input_sum32, 0, rows, output);

    var params: [matvec_parallel_max_workers]MatvecDirectWorker = undefined;
    var tasks: [matvec_parallel_max_workers]zinc_rt.fast_pool.Task = undefined;
    const rows_per_worker = (@as(usize, @intCast(rows)) + worker_count - 1) / worker_count;
    var dispatched: usize = 0;
    while (dispatched < worker_count) : (dispatched += 1) {
        const start: u32 = @intCast(dispatched * rows_per_worker);
        const end: u32 = @intCast(@min(@as(usize, @intCast(rows)), (dispatched + 1) * rows_per_worker));
        if (start >= end) break;
        params[dispatched] = .{
            .raw = raw,
            .tensor_type = tensor_type,
            .input = input,
            .input_sum32 = input_sum32,
            .output = output,
            .start_row = start,
            .end_row = end,
        };
        tasks[dispatched] = .{ .fn_ = matvecDirectWorkerTask, .ctx = @ptrCast(&params[dispatched]) };
    }
    fp.dispatchAndRun(tasks[0..dispatched]);

    var failed = false;
    for (params[0..dispatched]) |param| failed = failed or param.failed;
    if (failed) return error.InputTooSmall;
}

fn matvecRawDirectPooled(
    pool: *thread_pool.Pool,
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    output: []f32,
    input_sum32: ?[]const f32,
) !void {
    const cols: u32 = @intCast(input.len);
    const executors = poolExecutorCount(pool);
    const worker_count = matvecWorkerCountForCpu(rows, cols, executors);
    if (worker_count <= 1)
        return matvecRawDirectSerial(raw, tensor_type, input, input_sum32, 0, rows, output);

    var params: [matvec_parallel_max_workers]MatvecDirectWorker = undefined;
    const rows_per_worker = (@as(usize, @intCast(rows)) + worker_count - 1) / worker_count;
    var wg: thread_pool.WaitGroup = .{};
    var dispatched: usize = 0;
    while (dispatched < worker_count) : (dispatched += 1) {
        const start: u32 = @intCast(dispatched * rows_per_worker);
        const end: u32 = @intCast(@min(@as(usize, @intCast(rows)), (dispatched + 1) * rows_per_worker));
        if (start >= end) break;
        params[dispatched] = .{
            .raw = raw,
            .tensor_type = tensor_type,
            .input = input,
            .input_sum32 = input_sum32,
            .output = output,
            .start_row = start,
            .end_row = end,
        };
        pool.spawnWg(&wg, matvecDirectWorkerMain, .{&params[dispatched]});
    }
    pool.waitAndWork(&wg);

    var failed = false;
    for (params[0..dispatched]) |param| failed = failed or param.failed;
    if (failed) return error.InputTooSmall;
}

fn matvecRawDirectThreaded(
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    output: []f32,
) !void {
    const cols: u32 = @intCast(input.len);
    const worker_count = matvecWorkerCountForCpu(rows, cols, std.Thread.getCpuCount() catch 1);
    if (worker_count <= 1)
        return matvecRawDirectSerial(raw, tensor_type, input, null, 0, rows, output);

    var params: [matvec_parallel_max_workers]MatvecDirectWorker = undefined;
    var threads: [matvec_parallel_max_workers]std.Thread = undefined;
    const rows_per_worker = (@as(usize, @intCast(rows)) + worker_count - 1) / worker_count;

    var spawned: usize = 0;
    while (spawned < worker_count) : (spawned += 1) {
        const start: u32 = @intCast(spawned * rows_per_worker);
        const end: u32 = @intCast(@min(@as(usize, @intCast(rows)), (spawned + 1) * rows_per_worker));
        if (start >= end) break;
        params[spawned] = .{
            .raw = raw,
            .tensor_type = tensor_type,
            .input = input,
            .input_sum32 = null,
            .output = output,
            .start_row = start,
            .end_row = end,
        };
        threads[spawned] = std.Thread.spawn(.{}, matvecDirectWorkerMain, .{&params[spawned]}) catch {
            for (threads[0..spawned]) |thread| thread.join();
            return matvecRawDirectSerial(raw, tensor_type, input, null, 0, rows, output);
        };
    }

    var failed = false;
    for (threads[0..spawned]) |thread| thread.join();
    for (params[0..spawned]) |param| failed = failed or param.failed;
    if (failed) return error.InputTooSmall;
}

fn matvecWorkerCountForCpu(rows: u32, cols: u32, cpu_count: usize) usize {
    if (rows == 0 or cols == 0 or cpu_count <= 1) return 1;
    const total_work = matvecWorkItems(rows, cols);
    if (total_work < matvec_parallel_min_work_items) return 1;

    const work_limit: usize = if (total_work < matvec_parallel_medium_work_items)
        4
    else if (total_work < matvec_parallel_large_work_items)
        8
    else
        matvec_parallel_max_workers;

    return @max(@as(usize, 1), @min(@min(cpu_count, work_limit), @as(usize, @intCast(rows))));
}

fn matvecWorkItems(rows: u32, cols: u32) u64 {
    return @as(u64, rows) * @as(u64, cols);
}

fn matvecDirectWorkerTask(ctx: *anyopaque) void {
    const params: *MatvecDirectWorker = @ptrCast(@alignCast(ctx));
    matvecDirectWorkerMain(params);
}

fn matvecDirectWorkerMain(params: *MatvecDirectWorker) void {
    matvecRawDirectSerial(
        params.raw,
        params.tensor_type,
        params.input,
        params.input_sum32,
        params.start_row,
        params.end_row,
        params.output,
    ) catch {
        params.failed = true;
    };
}

inline fn dotDirectTyped(
    comptime tensor_type: gguf.GGMLType,
    raw: []const u8,
    row: u32,
    cols: u32,
    input: []const f32,
    input_sum32: ?[]const f32,
) !f32 {
    return switch (tensor_type) {
        .f32 => dequant.dotF32Row(raw, row, cols, input),
        .f16 => dequant.dotF16Row(raw, row, cols, input),
        .bf16 => dequant.dotBf16Row(raw, row, cols, input),
        .q4_0 => if (input_sum32) |sums| dequant.dotQ4_0RowWithSum32Unchecked(raw, row, cols, input, sums) else dequant.dotQ4_0RowUnchecked(raw, row, cols, input),
        .q8_0 => dequant.dotQ8_0RowUnchecked(raw, row, cols, input),
        .q4_k => if (input_sum32) |sums| dequant.dotQ4KRowWithSum32Unchecked(raw, row, cols, input, sums) else dequant.dotQ4KRowUnchecked(raw, row, cols, input),
        .q5_k => if (input_sum32) |sums| dequant.dotQ5KRowWithSum32Unchecked(raw, row, cols, input, sums) else dequant.dotQ5KRowUnchecked(raw, row, cols, input),
        .q6_k => dequant.dotQ6KRowUnchecked(raw, row, cols, input),
        else => unreachable,
    };
}

inline fn dotDirectRaw(
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    row: u32,
    cols: u32,
    input: []const f32,
    input_sum32: ?[]const f32,
) !f32 {
    return switch (tensor_type) {
        .f32 => dotDirectTyped(.f32, raw, row, cols, input, input_sum32),
        .f16 => dotDirectTyped(.f16, raw, row, cols, input, input_sum32),
        .bf16 => dotDirectTyped(.bf16, raw, row, cols, input, input_sum32),
        .q4_0 => dotDirectTyped(.q4_0, raw, row, cols, input, input_sum32),
        .q8_0 => dotDirectTyped(.q8_0, raw, row, cols, input, input_sum32),
        .q4_k => dotDirectTyped(.q4_k, raw, row, cols, input, input_sum32),
        .q5_k => dotDirectTyped(.q5_k, raw, row, cols, input, input_sum32),
        .q6_k => dotDirectTyped(.q6_k, raw, row, cols, input, input_sum32),
        else => unreachable,
    };
}

inline fn rowBytesForType(comptime tensor_type: gguf.GGMLType, cols: u32) usize {
    return switch (tensor_type) {
        .f32 => @as(usize, cols) * @sizeOf(f32),
        .f16 => @as(usize, cols) * @sizeOf(f16),
        .bf16 => @as(usize, cols) * 2,
        .q4_0 => @as(usize, cols) / 32 * 18,
        .q8_0 => @as(usize, cols) / 32 * 34,
        .q4_k => @as(usize, cols) / 256 * 144,
        .q5_k => @as(usize, cols) / 256 * 176,
        .q6_k => @as(usize, cols) / 256 * 210,
        else => 0,
    };
}

inline fn matvecRawDirectSerialTyped(
    comptime tensor_type: gguf.GGMLType,
    raw: []const u8,
    cols: u32,
    input: []const f32,
    input_sum32: ?[]const f32,
    start_row: u32,
    end_row: u32,
    output: []f32,
) !void {
    const row_bytes = rowBytesForType(tensor_type, cols);
    // Pre-touch the FIRST cache line of the next row via PREFETCHNTA before
    // computing this row. Rows of the same matvec are laid out contiguously, so
    // the HW L2 streamer handles steady-state sequential reads cleanly; the win
    // here is hiding the head-of-row dependency that surfaces when the dot's
    // inner-loop @mulAdd chain stalls on the first weight cache line — the NTA
    // hint lands the line in the L1/L2 non-temporal slot well before the dot
    // needs it without polluting the working set (locality=0). Empirical: one
    // NTA prefetch measured +2.1 tok/s median across 8 samples (76.7→79.6 on
    // the 9800X3D + Q4_0 attn_qkv stream); a four-line prefetch at locality=2
    // measured -3 tok/s from contention on the dot's own DRAM stream. Skip tiny
    // rows (<4 cache lines) where the row finishes before the prefetch lands.
    const prefetch_min_row_bytes: usize = 256;
    var row = start_row;
    while (row < end_row) : (row += 1) {
        if (row_bytes >= prefetch_min_row_bytes and row + 1 < end_row) {
            const next_off = @as(usize, row + 1) * row_bytes;
            @prefetch(&raw[next_off], .{ .rw = .read, .locality = 0 });
        }
        output[row] = try dotDirectTyped(tensor_type, raw, row, cols, input, input_sum32);
    }
}

fn matvecRawDirectSerial(
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    input_sum32: ?[]const f32,
    start_row: u32,
    end_row: u32,
    output: []f32,
) !void {
    const cols: u32 = @intCast(input.len);
    return switch (tensor_type) {
        .f32 => matvecRawDirectSerialTyped(.f32, raw, cols, input, input_sum32, start_row, end_row, output),
        .f16 => matvecRawDirectSerialTyped(.f16, raw, cols, input, input_sum32, start_row, end_row, output),
        .bf16 => matvecRawDirectSerialTyped(.bf16, raw, cols, input, input_sum32, start_row, end_row, output),
        .q4_0 => matvecRawDirectSerialTyped(.q4_0, raw, cols, input, input_sum32, start_row, end_row, output),
        .q8_0 => matvecRawDirectSerialTyped(.q8_0, raw, cols, input, input_sum32, start_row, end_row, output),
        .q4_k => matvecRawDirectSerialTyped(.q4_k, raw, cols, input, input_sum32, start_row, end_row, output),
        .q5_k => matvecRawDirectSerialTyped(.q5_k, raw, cols, input, input_sum32, start_row, end_row, output),
        .q6_k => matvecRawDirectSerialTyped(.q6_k, raw, cols, input, input_sum32, start_row, end_row, output),
        else => unreachable,
    };
}

const ScoredToken = struct {
    index: u32,
    value: f32,
};

const ArgmaxTop2Result = struct {
    best: ScoredToken = .{ .index = 0, .value = -std.math.inf(f32) },
    second: ScoredToken = .{ .index = 0, .value = -std.math.inf(f32) },

    fn offer(self: *ArgmaxTop2Result, index: u32, value: f32) void {
        if (value > self.best.value) {
            self.second = self.best;
            self.best = .{ .index = index, .value = value };
        } else if (value > self.second.value) {
            self.second = .{ .index = index, .value = value };
        }
    }

    fn offerToken(self: *ArgmaxTop2Result, token: ScoredToken) void {
        self.offer(token.index, token.value);
    }
};

const ArgmaxMatvecWorker = struct {
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    input_sum32: ?[]const f32,
    start_row: u32,
    end_row: u32,
    result: ArgmaxTop2Result = .{},
    failed: bool = false,
};

const ArgmaxBestWorker = struct {
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    input_sum32: ?[]const f32,
    start_row: u32,
    end_row: u32,
    result: ScoredToken = .{ .index = 0, .value = -std.math.inf(f32) },
    failed: bool = false,
};

fn argmaxMatvecRaw(
    pool: ?*thread_pool.Pool,
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    scratch: []f32,
) !u32 {
    const result = try argmaxMatvecRawBest(pool, raw, tensor_type, input, rows, scratch);
    return result.index;
}

fn argmaxMatvecRawBest(
    pool: ?*thread_pool.Pool,
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    scratch: []f32,
) !ScoredToken {
    const cols: u32 = @intCast(input.len);
    if (!canDotDirect(tensor_type, cols)) return error.UnsupportedTensorType;
    const input_sum32 = sums: {
        if (wantsInputSum32(tensor_type) and cols % 32 == 0 and scratch.len >= input.len / 32) {
            const sums = scratch[0 .. input.len / 32];
            dequant.fillInputSum32(input, sums);
            break :sums sums;
        }
        break :sums null;
    };
    return argmaxMatvecRawBestDirect(pool, raw, tensor_type, input, rows, input_sum32);
}

fn argmaxMatvecRawTop2(
    pool: ?*thread_pool.Pool,
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    scratch: []f32,
) !ArgmaxTop2Result {
    const cols: u32 = @intCast(input.len);
    if (!canDotDirect(tensor_type, cols)) return error.UnsupportedTensorType;
    const input_sum32 = sums: {
        if (wantsInputSum32(tensor_type) and cols % 32 == 0 and scratch.len >= input.len / 32) {
            const sums = scratch[0 .. input.len / 32];
            dequant.fillInputSum32(input, sums);
            break :sums sums;
        }
        break :sums null;
    };
    return argmaxMatvecRawDirect(pool, raw, tensor_type, input, rows, input_sum32);
}

fn argmaxMatvecRawDirect(
    pool: ?*thread_pool.Pool,
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    input_sum32: ?[]const f32,
) !ArgmaxTop2Result {
    const cols: u32 = @intCast(input.len);
    if (matvecWorkItems(rows, cols) < matvec_parallel_min_work_items)
        return argmaxMatvecRawDirectSerial(raw, tensor_type, input, input_sum32, 0, rows);
    if (pool != null) {
        if (matvec_fast_pool) |fp| return argmaxMatvecRawDirectFastPooled(fp, raw, tensor_type, input, rows, input_sum32);
    }
    if (pool) |p| return argmaxMatvecRawDirectPooled(p, raw, tensor_type, input, rows, input_sum32);
    return argmaxMatvecRawDirectSerial(raw, tensor_type, input, input_sum32, 0, rows);
}

fn argmaxMatvecRawBestDirect(
    pool: ?*thread_pool.Pool,
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    input_sum32: ?[]const f32,
) !ScoredToken {
    const cols: u32 = @intCast(input.len);
    if (matvecWorkItems(rows, cols) < matvec_parallel_min_work_items)
        return argmaxMatvecRawBestDirectSerial(raw, tensor_type, input, input_sum32, 0, rows);
    if (pool != null) {
        if (matvec_fast_pool) |fp| return argmaxMatvecRawBestDirectFastPooled(fp, raw, tensor_type, input, rows, input_sum32);
    }
    if (pool) |p| return argmaxMatvecRawBestDirectPooled(p, raw, tensor_type, input, rows, input_sum32);
    return argmaxMatvecRawBestDirectSerial(raw, tensor_type, input, input_sum32, 0, rows);
}

fn argmaxMatvecRawBestDirectFastPooled(
    fp: *zinc_rt.fast_pool.FastPool,
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    input_sum32: ?[]const f32,
) !ScoredToken {
    const cols: u32 = @intCast(input.len);
    const worker_count = matvecWorkerCountForCpu(rows, cols, fp.executorCount());
    if (worker_count <= 1)
        return argmaxMatvecRawBestDirectSerial(raw, tensor_type, input, input_sum32, 0, rows);

    var params: [matvec_parallel_max_workers]ArgmaxBestWorker = undefined;
    var tasks: [matvec_parallel_max_workers]zinc_rt.fast_pool.Task = undefined;
    const rows_per_worker = (@as(usize, @intCast(rows)) + worker_count - 1) / worker_count;
    var dispatched: usize = 0;
    while (dispatched < worker_count) : (dispatched += 1) {
        const start: u32 = @intCast(dispatched * rows_per_worker);
        const end: u32 = @intCast(@min(@as(usize, @intCast(rows)), (dispatched + 1) * rows_per_worker));
        if (start >= end) break;
        params[dispatched] = .{
            .raw = raw,
            .tensor_type = tensor_type,
            .input = input,
            .input_sum32 = input_sum32,
            .start_row = start,
            .end_row = end,
        };
        tasks[dispatched] = .{ .fn_ = argmaxBestWorkerTask, .ctx = @ptrCast(&params[dispatched]) };
    }
    fp.dispatchAndRun(tasks[0..dispatched]);

    var best = ScoredToken{ .index = 0, .value = -std.math.inf(f32) };
    var failed = false;
    for (params[0..dispatched]) |param| {
        failed = failed or param.failed;
        if (param.result.value > best.value) best = param.result;
    }
    if (failed) return error.InputTooSmall;
    return best;
}

fn argmaxMatvecRawDirectFastPooled(
    fp: *zinc_rt.fast_pool.FastPool,
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    input_sum32: ?[]const f32,
) !ArgmaxTop2Result {
    const cols: u32 = @intCast(input.len);
    const worker_count = matvecWorkerCountForCpu(rows, cols, fp.executorCount());
    if (worker_count <= 1)
        return argmaxMatvecRawDirectSerial(raw, tensor_type, input, input_sum32, 0, rows);

    var params: [matvec_parallel_max_workers]ArgmaxMatvecWorker = undefined;
    var tasks: [matvec_parallel_max_workers]zinc_rt.fast_pool.Task = undefined;
    const rows_per_worker = (@as(usize, @intCast(rows)) + worker_count - 1) / worker_count;
    var dispatched: usize = 0;
    while (dispatched < worker_count) : (dispatched += 1) {
        const start: u32 = @intCast(dispatched * rows_per_worker);
        const end: u32 = @intCast(@min(@as(usize, @intCast(rows)), (dispatched + 1) * rows_per_worker));
        if (start >= end) break;
        params[dispatched] = .{
            .raw = raw,
            .tensor_type = tensor_type,
            .input = input,
            .input_sum32 = input_sum32,
            .start_row = start,
            .end_row = end,
        };
        tasks[dispatched] = .{ .fn_ = argmaxMatvecWorkerTask, .ctx = @ptrCast(&params[dispatched]) };
    }
    fp.dispatchAndRun(tasks[0..dispatched]);

    var best: ArgmaxTop2Result = .{};
    var failed = false;
    for (params[0..dispatched]) |param| {
        failed = failed or param.failed;
        best.offerToken(param.result.best);
        best.offerToken(param.result.second);
    }
    if (failed) return error.InputTooSmall;
    return best;
}

fn argmaxMatvecRawBestDirectPooled(
    pool: *thread_pool.Pool,
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    input_sum32: ?[]const f32,
) !ScoredToken {
    const cols: u32 = @intCast(input.len);
    const worker_count = matvecWorkerCountForCpu(rows, cols, poolExecutorCount(pool));
    if (worker_count <= 1)
        return argmaxMatvecRawBestDirectSerial(raw, tensor_type, input, input_sum32, 0, rows);

    var params: [matvec_parallel_max_workers]ArgmaxBestWorker = undefined;
    const rows_per_worker = (@as(usize, @intCast(rows)) + worker_count - 1) / worker_count;
    var wg: thread_pool.WaitGroup = .{};
    var dispatched: usize = 0;
    while (dispatched < worker_count) : (dispatched += 1) {
        const start: u32 = @intCast(dispatched * rows_per_worker);
        const end: u32 = @intCast(@min(@as(usize, @intCast(rows)), (dispatched + 1) * rows_per_worker));
        if (start >= end) break;
        params[dispatched] = .{
            .raw = raw,
            .tensor_type = tensor_type,
            .input = input,
            .input_sum32 = input_sum32,
            .start_row = start,
            .end_row = end,
        };
        pool.spawnWg(&wg, argmaxBestWorkerMain, .{&params[dispatched]});
    }
    pool.waitAndWork(&wg);

    var best = ScoredToken{ .index = 0, .value = -std.math.inf(f32) };
    var failed = false;
    for (params[0..dispatched]) |param| {
        failed = failed or param.failed;
        if (param.result.value > best.value) best = param.result;
    }
    if (failed) return error.InputTooSmall;
    return best;
}

fn argmaxMatvecRawDirectPooled(
    pool: *thread_pool.Pool,
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    input_sum32: ?[]const f32,
) !ArgmaxTop2Result {
    const cols: u32 = @intCast(input.len);
    const worker_count = matvecWorkerCountForCpu(rows, cols, poolExecutorCount(pool));
    if (worker_count <= 1)
        return argmaxMatvecRawDirectSerial(raw, tensor_type, input, input_sum32, 0, rows);

    var params: [matvec_parallel_max_workers]ArgmaxMatvecWorker = undefined;
    const rows_per_worker = (@as(usize, @intCast(rows)) + worker_count - 1) / worker_count;
    var wg: thread_pool.WaitGroup = .{};
    var dispatched: usize = 0;
    while (dispatched < worker_count) : (dispatched += 1) {
        const start: u32 = @intCast(dispatched * rows_per_worker);
        const end: u32 = @intCast(@min(@as(usize, @intCast(rows)), (dispatched + 1) * rows_per_worker));
        if (start >= end) break;
        params[dispatched] = .{
            .raw = raw,
            .tensor_type = tensor_type,
            .input = input,
            .input_sum32 = input_sum32,
            .start_row = start,
            .end_row = end,
        };
        pool.spawnWg(&wg, argmaxMatvecWorkerMain, .{&params[dispatched]});
    }
    pool.waitAndWork(&wg);

    var best: ArgmaxTop2Result = .{};
    var failed = false;
    for (params[0..dispatched]) |param| {
        failed = failed or param.failed;
        best.offerToken(param.result.best);
        best.offerToken(param.result.second);
    }
    if (failed) return error.InputTooSmall;
    return best;
}

fn argmaxMatvecWorkerMain(params: *ArgmaxMatvecWorker) void {
    params.result = argmaxMatvecRawDirectSerial(
        params.raw,
        params.tensor_type,
        params.input,
        params.input_sum32,
        params.start_row,
        params.end_row,
    ) catch {
        params.failed = true;
        return;
    };
}

fn argmaxBestWorkerMain(params: *ArgmaxBestWorker) void {
    params.result = argmaxMatvecRawBestDirectSerial(
        params.raw,
        params.tensor_type,
        params.input,
        params.input_sum32,
        params.start_row,
        params.end_row,
    ) catch {
        params.failed = true;
        return;
    };
}

fn argmaxMatvecWorkerTask(ctx: *anyopaque) void {
    const params: *ArgmaxMatvecWorker = @ptrCast(@alignCast(ctx));
    argmaxMatvecWorkerMain(params);
}

fn argmaxBestWorkerTask(ctx: *anyopaque) void {
    const params: *ArgmaxBestWorker = @ptrCast(@alignCast(ctx));
    argmaxBestWorkerMain(params);
}

fn argmaxMatvecRawDirectSerialTyped(
    comptime tensor_type: gguf.GGMLType,
    raw: []const u8,
    cols: u32,
    input: []const f32,
    input_sum32: ?[]const f32,
    start_row: u32,
    end_row: u32,
) !ArgmaxTop2Result {
    var best: ArgmaxTop2Result = .{};
    var row = start_row;
    while (row < end_row) : (row += 1) {
        const value = try dotDirectTyped(tensor_type, raw, row, cols, input, input_sum32);
        best.offer(row, value);
    }
    return best;
}

fn argmaxMatvecRawBestDirectSerialTyped(
    comptime tensor_type: gguf.GGMLType,
    raw: []const u8,
    cols: u32,
    input: []const f32,
    input_sum32: ?[]const f32,
    start_row: u32,
    end_row: u32,
) !ScoredToken {
    var best = ScoredToken{ .index = 0, .value = -std.math.inf(f32) };
    var row = start_row;
    while (row < end_row) : (row += 1) {
        const value = try dotDirectTyped(tensor_type, raw, row, cols, input, input_sum32);
        if (value > best.value) best = .{ .index = row, .value = value };
    }
    return best;
}

fn argmaxMatvecRawDirectSerial(
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    input_sum32: ?[]const f32,
    start_row: u32,
    end_row: u32,
) !ArgmaxTop2Result {
    const cols: u32 = @intCast(input.len);
    return switch (tensor_type) {
        .f32 => argmaxMatvecRawDirectSerialTyped(.f32, raw, cols, input, input_sum32, start_row, end_row),
        .f16 => argmaxMatvecRawDirectSerialTyped(.f16, raw, cols, input, input_sum32, start_row, end_row),
        .bf16 => argmaxMatvecRawDirectSerialTyped(.bf16, raw, cols, input, input_sum32, start_row, end_row),
        .q4_0 => argmaxMatvecRawDirectSerialTyped(.q4_0, raw, cols, input, input_sum32, start_row, end_row),
        .q8_0 => argmaxMatvecRawDirectSerialTyped(.q8_0, raw, cols, input, input_sum32, start_row, end_row),
        .q4_k => argmaxMatvecRawDirectSerialTyped(.q4_k, raw, cols, input, input_sum32, start_row, end_row),
        .q5_k => argmaxMatvecRawDirectSerialTyped(.q5_k, raw, cols, input, input_sum32, start_row, end_row),
        .q6_k => argmaxMatvecRawDirectSerialTyped(.q6_k, raw, cols, input, input_sum32, start_row, end_row),
        else => unreachable,
    };
}

fn argmaxMatvecRawBestDirectSerial(
    raw: []const u8,
    tensor_type: gguf.GGMLType,
    input: []const f32,
    input_sum32: ?[]const f32,
    start_row: u32,
    end_row: u32,
) !ScoredToken {
    const cols: u32 = @intCast(input.len);
    return switch (tensor_type) {
        .f32 => argmaxMatvecRawBestDirectSerialTyped(.f32, raw, cols, input, input_sum32, start_row, end_row),
        .f16 => argmaxMatvecRawBestDirectSerialTyped(.f16, raw, cols, input, input_sum32, start_row, end_row),
        .bf16 => argmaxMatvecRawBestDirectSerialTyped(.bf16, raw, cols, input, input_sum32, start_row, end_row),
        .q4_0 => argmaxMatvecRawBestDirectSerialTyped(.q4_0, raw, cols, input, input_sum32, start_row, end_row),
        .q8_0 => argmaxMatvecRawBestDirectSerialTyped(.q8_0, raw, cols, input, input_sum32, start_row, end_row),
        .q4_k => argmaxMatvecRawBestDirectSerialTyped(.q4_k, raw, cols, input, input_sum32, start_row, end_row),
        .q5_k => argmaxMatvecRawBestDirectSerialTyped(.q5_k, raw, cols, input, input_sum32, start_row, end_row),
        .q6_k => argmaxMatvecRawBestDirectSerialTyped(.q6_k, raw, cols, input, input_sum32, start_row, end_row),
        else => unreachable,
    };
}

const RouteTop1SharedGateResult = struct {
    expert_id: u32,
    shared_gate: f32,
};

const RouteTop1SharedGatePartial = struct {
    best: ScoredToken = .{ .index = 0, .value = -std.math.inf(f32) },
    shared_gate: f32 = 0.0,
    has_shared_gate: bool = false,
};

const RouterSharedGateWorker = struct {
    router: WeightView,
    router_rows: u32,
    shared_gate: WeightView,
    input: []const f32,
    input_sum32: ?[]const f32,
    start_g: u32,
    end_g: u32,
    result: RouteTop1SharedGatePartial = .{},
    failed: bool = false,
};

fn routeTop1SharedGate(
    pool: ?*thread_pool.Pool,
    router: WeightView,
    router_rows: u32,
    shared_gate: WeightView,
    input: []const f32,
    scratch: []f32,
) !RouteTop1SharedGateResult {
    if (router_rows == 0) return error.ShapeMismatch;
    const cols: u32 = @intCast(input.len);
    if (!canDotDirect(router.type_, cols) or !canDotDirect(shared_gate.type_, cols)) return error.UnsupportedTensorType;
    const input_sum32 = sums: {
        if ((wantsInputSum32(router.type_) or wantsInputSum32(shared_gate.type_)) and
            cols % 32 == 0 and
            scratch.len >= input.len / 32)
        {
            const sums = scratch[0 .. input.len / 32];
            dequant.fillInputSum32(input, sums);
            break :sums sums;
        }
        break :sums null;
    };
    return routeTop1SharedGateDirect(pool, router, router_rows, shared_gate, input, input_sum32);
}

fn routeTop1SharedGateDirect(
    pool: ?*thread_pool.Pool,
    router: WeightView,
    router_rows: u32,
    shared_gate: WeightView,
    input: []const f32,
    input_sum32: ?[]const f32,
) !RouteTop1SharedGateResult {
    const cols: u32 = @intCast(input.len);
    const total_rows = router_rows + 1;
    if (matvecWorkItems(total_rows, cols) < matvec_parallel_min_work_items)
        return routeTop1SharedGateDirectSerialResult(router, router_rows, shared_gate, input, input_sum32, 0, total_rows);
    // FastPool fires once per MoE layer (30/token) on the decode-phase top-1
    // route; switching off thread_pool.Pool's spawnWg cuts a mutex+condvar
    // barrier per call.
    if (matvec_fast_pool) |fp|
        return routeTop1SharedGateDirectFastPooled(fp, router, router_rows, shared_gate, input, input_sum32);
    if (pool) |p| return routeTop1SharedGateDirectPooled(p, router, router_rows, shared_gate, input, input_sum32);
    return routeTop1SharedGateDirectSerialResult(router, router_rows, shared_gate, input, input_sum32, 0, total_rows);
}

fn reduceRouteTop1SharedGate(params: []const RouterSharedGateWorker) !RouteTop1SharedGateResult {
    var best = ScoredToken{ .index = 0, .value = -std.math.inf(f32) };
    var shared_value: f32 = 0.0;
    var found_shared = false;
    var failed = false;
    for (params) |param| {
        failed = failed or param.failed;
        if (param.result.best.value > best.value) best = param.result.best;
        if (param.result.has_shared_gate) {
            shared_value = param.result.shared_gate;
            found_shared = true;
        }
    }
    if (failed) return error.InputTooSmall;
    if (!found_shared) return error.ShapeMismatch;
    return .{ .expert_id = best.index, .shared_gate = shared_value };
}

fn routeTop1SharedGateDirectFastPooled(
    fp: *zinc_rt.fast_pool.FastPool,
    router: WeightView,
    router_rows: u32,
    shared_gate: WeightView,
    input: []const f32,
    input_sum32: ?[]const f32,
) !RouteTop1SharedGateResult {
    const cols: u32 = @intCast(input.len);
    const total_rows = router_rows + 1;
    const worker_count = matvecWorkerCountForCpu(total_rows, cols, fp.executorCount());
    if (worker_count <= 1)
        return routeTop1SharedGateDirectSerialResult(router, router_rows, shared_gate, input, input_sum32, 0, total_rows);

    var params: [matvec_parallel_max_workers]RouterSharedGateWorker = undefined;
    var tasks: [matvec_parallel_max_workers]zinc_rt.fast_pool.Task = undefined;
    const rows_per_worker = (@as(usize, @intCast(total_rows)) + worker_count - 1) / worker_count;
    var dispatched: usize = 0;
    while (dispatched < worker_count) : (dispatched += 1) {
        const start: u32 = @intCast(dispatched * rows_per_worker);
        const end: u32 = @intCast(@min(@as(usize, @intCast(total_rows)), (dispatched + 1) * rows_per_worker));
        if (start >= end) break;
        params[dispatched] = .{
            .router = router,
            .router_rows = router_rows,
            .shared_gate = shared_gate,
            .input = input,
            .input_sum32 = input_sum32,
            .start_g = start,
            .end_g = end,
        };
        tasks[dispatched] = .{ .fn_ = routeTop1SharedGateWorkerTask, .ctx = @ptrCast(&params[dispatched]) };
    }
    fp.dispatchAndRun(tasks[0..dispatched]);
    return reduceRouteTop1SharedGate(params[0..dispatched]);
}

fn routeTop1SharedGateDirectPooled(
    pool: *thread_pool.Pool,
    router: WeightView,
    router_rows: u32,
    shared_gate: WeightView,
    input: []const f32,
    input_sum32: ?[]const f32,
) !RouteTop1SharedGateResult {
    const cols: u32 = @intCast(input.len);
    const total_rows = router_rows + 1;
    const worker_count = matvecWorkerCountForCpu(total_rows, cols, poolExecutorCount(pool));
    if (worker_count <= 1)
        return routeTop1SharedGateDirectSerialResult(router, router_rows, shared_gate, input, input_sum32, 0, total_rows);

    var params: [matvec_parallel_max_workers]RouterSharedGateWorker = undefined;
    const rows_per_worker = (@as(usize, @intCast(total_rows)) + worker_count - 1) / worker_count;
    var wg: thread_pool.WaitGroup = .{};
    var dispatched: usize = 0;
    while (dispatched < worker_count) : (dispatched += 1) {
        const start: u32 = @intCast(dispatched * rows_per_worker);
        const end: u32 = @intCast(@min(@as(usize, @intCast(total_rows)), (dispatched + 1) * rows_per_worker));
        if (start >= end) break;
        params[dispatched] = .{
            .router = router,
            .router_rows = router_rows,
            .shared_gate = shared_gate,
            .input = input,
            .input_sum32 = input_sum32,
            .start_g = start,
            .end_g = end,
        };
        pool.spawnWg(&wg, routeTop1SharedGateWorkerMain, .{&params[dispatched]});
    }
    pool.waitAndWork(&wg);
    return reduceRouteTop1SharedGate(params[0..dispatched]);
}

fn routeTop1SharedGateWorkerMain(params: *RouterSharedGateWorker) void {
    params.result = routeTop1SharedGateDirectSerial(
        params.router,
        params.router_rows,
        params.shared_gate,
        params.input,
        params.input_sum32,
        params.start_g,
        params.end_g,
    ) catch {
        params.failed = true;
        return;
    };
}

fn routeTop1SharedGateWorkerTask(ctx: *anyopaque) void {
    const p: *RouterSharedGateWorker = @ptrCast(@alignCast(ctx));
    routeTop1SharedGateWorkerMain(p);
}

fn routeTop1SharedGateDirectSerialResult(
    router: WeightView,
    router_rows: u32,
    shared_gate: WeightView,
    input: []const f32,
    input_sum32: ?[]const f32,
    start_g: u32,
    end_g: u32,
) !RouteTop1SharedGateResult {
    const partial = try routeTop1SharedGateDirectSerial(router, router_rows, shared_gate, input, input_sum32, start_g, end_g);
    if (!partial.has_shared_gate) return error.ShapeMismatch;
    return .{ .expert_id = partial.best.index, .shared_gate = partial.shared_gate };
}

fn routeTop1SharedGateDirectSerial(
    router: WeightView,
    router_rows: u32,
    shared_gate: WeightView,
    input: []const f32,
    input_sum32: ?[]const f32,
    start_g: u32,
    end_g: u32,
) !RouteTop1SharedGatePartial {
    const cols: u32 = @intCast(input.len);
    var result: RouteTop1SharedGatePartial = .{};
    var g = start_g;
    while (g < end_g) : (g += 1) {
        if (g < router_rows) {
            const value = try dotDirectRaw(router.raw, router.type_, g, cols, input, input_sum32);
            if (value > result.best.value) result.best = .{ .index = g, .value = value };
        } else if (g == router_rows) {
            result.shared_gate = try dotDirectRaw(shared_gate.raw, shared_gate.type_, 0, cols, input, input_sum32);
            result.has_shared_gate = true;
        }
    }
    return result;
}

fn needsInputSum32(tensor_type: gguf.GGMLType) bool {
    return tensor_type == .q4_k or tensor_type == .q5_k;
}

fn wantsInputSum32(tensor_type: gguf.GGMLType) bool {
    return tensor_type == .q4_0 or needsInputSum32(tensor_type);
}

// --- Fused multi-matrix matvec --------------------------------------------
// Several independent `canDotDirect` matvecs that share the same input vector
// (e.g. an SSM layer's `attn_qkv` + `attn_gate`, or an attention layer's
// `attn_q` + `attn_k` + `attn_v`, all fed from the same `attn_norm`) get folded
// into one worker-pool dispatch over the concatenated row range — decode is
// serialised hop-by-hop through ~200 pool barriers per token, so cutting the
// per-layer barrier count is a real per-token win, and the few tiny rows that
// would otherwise run serially on the main thread get parallelised for free.

const matvec_fuse_max_segments: usize = 4;

const FusedSegment = struct {
    raw: []const u8,
    type_: gguf.GGMLType,
    rows: u32,
    out: []f32,
};

const FusedPart = struct {
    info: gguf.TensorInfo,
    rows: u32,
    out: []f32,
};

const MatvecFusedWorker = struct {
    segs: []const FusedSegment,
    input: []const f32,
    input_sum32: ?[]const f32,
    start_g: u32,
    end_g: u32,
    failed: bool = false,
};

const MultiInputFusedSegment = struct {
    raw: []const u8,
    type_: gguf.GGMLType,
    input: []const f32,
    rows: u32,
    out: []f32,
};

const MatvecMultiInputFusedWorker = struct {
    segs: []const MultiInputFusedSegment,
    start_g: u32,
    end_g: u32,
    failed: bool = false,
};

fn matvecFusedSerial(segs: []const FusedSegment, input: []const f32, input_sum32: ?[]const f32, start_g: u32, end_g: u32) !void {
    var seg_lo: u32 = 0;
    for (segs) |seg| {
        const seg_hi = seg_lo + seg.rows;
        const lo = @max(start_g, seg_lo);
        const hi = @min(end_g, seg_hi);
        if (lo < hi) {
            try matvecRawDirectSerial(seg.raw, seg.type_, input, input_sum32, lo - seg_lo, hi - seg_lo, seg.out);
        }
        seg_lo = seg_hi;
    }
}

fn matvecFusedWorkerMain(p: *MatvecFusedWorker) void {
    matvecFusedSerial(p.segs, p.input, p.input_sum32, p.start_g, p.end_g) catch {
        p.failed = true;
    };
}

fn matvecFusedWorkerTask(ctx: *anyopaque) void {
    const p: *MatvecFusedWorker = @ptrCast(@alignCast(ctx));
    matvecFusedWorkerMain(p);
}

fn matvecMultiInputFusedSerial(segs: []const MultiInputFusedSegment, start_g: u32, end_g: u32) !void {
    var seg_lo: u32 = 0;
    for (segs) |seg| {
        const seg_hi = seg_lo + seg.rows;
        const lo = @max(start_g, seg_lo);
        const hi = @min(end_g, seg_hi);
        if (lo < hi) {
            try matvecRawDirectSerial(seg.raw, seg.type_, seg.input, null, lo - seg_lo, hi - seg_lo, seg.out);
        }
        seg_lo = seg_hi;
    }
}

fn matvecMultiInputFusedWorkerMain(p: *MatvecMultiInputFusedWorker) void {
    matvecMultiInputFusedSerial(p.segs, p.start_g, p.end_g) catch {
        p.failed = true;
    };
}

fn matvecMultiInputFusedWorkerTask(ctx: *anyopaque) void {
    const p: *MatvecMultiInputFusedWorker = @ptrCast(@alignCast(ctx));
    matvecMultiInputFusedWorkerMain(p);
}

fn matvecFused(pool: ?*thread_pool.Pool, segs: []const FusedSegment, input: []const f32, input_sum32: ?[]const f32) !void {
    var total: u32 = 0;
    for (segs) |seg| {
        if (seg.out.len < seg.rows) return error.ShapeMismatch;
        total += seg.rows;
    }
    if (total == 0) return;
    const cols: u32 = @intCast(input.len);
    if (pool != null) {
        if (matvec_fast_pool) |fp| {
            const worker_count = matvecWorkerCountForCpu(total, cols, fp.executorCount());
            if (worker_count <= 1) return matvecFusedSerial(segs, input, input_sum32, 0, total);
            var params: [matvec_parallel_max_workers]MatvecFusedWorker = undefined;
            var tasks: [matvec_parallel_max_workers]zinc_rt.fast_pool.Task = undefined;
            const rows_per_worker = (@as(usize, total) + worker_count - 1) / worker_count;
            var dispatched: usize = 0;
            while (dispatched < worker_count) : (dispatched += 1) {
                const start: u32 = @intCast(dispatched * rows_per_worker);
                const end: u32 = @intCast(@min(@as(usize, total), (dispatched + 1) * rows_per_worker));
                if (start >= end) break;
                params[dispatched] = .{ .segs = segs, .input = input, .input_sum32 = input_sum32, .start_g = start, .end_g = end };
                tasks[dispatched] = .{ .fn_ = matvecFusedWorkerTask, .ctx = @ptrCast(&params[dispatched]) };
            }
            fp.dispatchAndRun(tasks[0..dispatched]);
            var failed_fp = false;
            for (params[0..dispatched]) |param| failed_fp = failed_fp or param.failed;
            if (failed_fp) return error.InputTooSmall;
            return;
        }
    }
    const worker_count: usize = if (pool) |p|
        matvecWorkerCountForCpu(total, cols, poolExecutorCount(p))
    else
        1;
    if (worker_count <= 1) return matvecFusedSerial(segs, input, input_sum32, 0, total);

    const p = pool.?;
    var params: [matvec_parallel_max_workers]MatvecFusedWorker = undefined;
    const rows_per_worker = (@as(usize, total) + worker_count - 1) / worker_count;
    var wg: thread_pool.WaitGroup = .{};
    var dispatched: usize = 0;
    while (dispatched < worker_count) : (dispatched += 1) {
        const start: u32 = @intCast(dispatched * rows_per_worker);
        const end: u32 = @intCast(@min(@as(usize, total), (dispatched + 1) * rows_per_worker));
        if (start >= end) break;
        params[dispatched] = .{ .segs = segs, .input = input, .input_sum32 = input_sum32, .start_g = start, .end_g = end };
        p.spawnWg(&wg, matvecFusedWorkerMain, .{&params[dispatched]});
    }
    p.waitAndWork(&wg);

    var failed = false;
    for (params[0..dispatched]) |param| failed = failed or param.failed;
    if (failed) return error.InputTooSmall;
}

fn matvecMultiInputFused(pool: *thread_pool.Pool, segs: []const MultiInputFusedSegment) !void {
    var total: u32 = 0;
    var max_cols: u32 = 0;
    for (segs) |seg| {
        if (seg.out.len < seg.rows) return error.ShapeMismatch;
        total += seg.rows;
        max_cols = @max(max_cols, @as(u32, @intCast(seg.input.len)));
    }
    if (total == 0) return;

    // Prefer FastPool — the MoE down phase fires once per MoE layer (30/token).
    // thread_pool.Pool's spawnWg/waitAndWork carries a mutex+condvar+heap-alloc
    // closure barrier (~µs each); FastPool's atomic-counter fan-out is ~tens of ns.
    if (matvec_fast_pool) |fp| {
        const worker_count = matvecWorkerCountForCpu(total, max_cols, fp.executorCount());
        if (worker_count <= 1) return matvecMultiInputFusedSerial(segs, 0, total);

        var params: [matvec_parallel_max_workers]MatvecMultiInputFusedWorker = undefined;
        var tasks: [matvec_parallel_max_workers]zinc_rt.fast_pool.Task = undefined;
        const rows_per_worker = (@as(usize, total) + worker_count - 1) / worker_count;
        var dispatched: usize = 0;
        while (dispatched < worker_count) : (dispatched += 1) {
            const start: u32 = @intCast(dispatched * rows_per_worker);
            const end: u32 = @intCast(@min(@as(usize, total), (dispatched + 1) * rows_per_worker));
            if (start >= end) break;
            params[dispatched] = .{ .segs = segs, .start_g = start, .end_g = end };
            tasks[dispatched] = .{ .fn_ = matvecMultiInputFusedWorkerTask, .ctx = @ptrCast(&params[dispatched]) };
        }
        fp.dispatchAndRun(tasks[0..dispatched]);

        var failed_fp = false;
        for (params[0..dispatched]) |param| failed_fp = failed_fp or param.failed;
        if (failed_fp) return error.InputTooSmall;
        return;
    }

    const worker_count = matvecWorkerCountForCpu(total, max_cols, poolExecutorCount(pool));
    if (worker_count <= 1) return matvecMultiInputFusedSerial(segs, 0, total);

    var params: [matvec_parallel_max_workers]MatvecMultiInputFusedWorker = undefined;
    const rows_per_worker = (@as(usize, total) + worker_count - 1) / worker_count;
    var wg: thread_pool.WaitGroup = .{};
    var dispatched: usize = 0;
    while (dispatched < worker_count) : (dispatched += 1) {
        const start: u32 = @intCast(dispatched * rows_per_worker);
        const end: u32 = @intCast(@min(@as(usize, total), (dispatched + 1) * rows_per_worker));
        if (start >= end) break;
        params[dispatched] = .{ .segs = segs, .start_g = start, .end_g = end };
        pool.spawnWg(&wg, matvecMultiInputFusedWorkerMain, .{&params[dispatched]});
    }
    pool.waitAndWork(&wg);

    var failed = false;
    for (params[0..dispatched]) |param| failed = failed or param.failed;
    if (failed) return error.InputTooSmall;
}

fn poolExecutorCount(pool: *thread_pool.Pool) usize {
    // `waitAndWork` lets the caller execute queued jobs while it waits, so large
    // matvec splits can use one more chunk than the persistent worker-thread
    // count without spawning another OS thread.
    return @max(@as(usize, 1), pool.threads.len + 1);
}

/// Fuse up to `matvec_fuse_max_segments` of `model`'s matvecs (all consuming
/// the same `input`) into one pool dispatch. Falls back to separate
/// `matvecTensor`-style calls if any tensor can't be dotted directly here.
fn matvecFusedTensors(
    pool: ?*thread_pool.Pool,
    model: *const Model,
    parts: []const FusedPart,
    input: []const f32,
    scratch: []f32,
) !void {
    std.debug.assert(parts.len > 0 and parts.len <= matvec_fuse_max_segments);
    const cols: u32 = @intCast(input.len);
    var segs: [matvec_fuse_max_segments]FusedSegment = undefined;
    var all_direct = true;
    var wants_input_sum32 = false;
    for (parts, 0..) |part, i| {
        const w = model.requantOrRaw(part.info);
        segs[i] = .{ .raw = w.raw, .type_ = w.type_, .rows = part.rows, .out = part.out };
        if (!canDotDirect(w.type_, cols)) all_direct = false;
        wants_input_sum32 = wants_input_sum32 or wantsInputSum32(w.type_);
    }
    if (all_direct) {
        const input_sum32 = sums: {
            if (wants_input_sum32 and cols % 32 == 0 and scratch.len >= input.len / 32) {
                const sums = scratch[0 .. input.len / 32];
                dequant.fillInputSum32(input, sums);
                break :sums sums;
            }
            break :sums null;
        };
        return matvecFused(pool, segs[0..parts.len], input, input_sum32);
    }
    for (parts) |part| {
        const w = model.requantOrRaw(part.info);
        try matvecRaw(pool, w.raw, w.type_, input, part.rows, scratch, part.out[0..part.rows]);
    }
}

fn rmsNormTensor(
    model: *const Model,
    info: gguf.TensorInfo,
    input: []const f32,
    output: []f32,
    scratch: []f32,
) !void {
    if (output.len < input.len) return error.ShapeMismatch;
    if (model.cachedF32(info)) |cached| {
        if (cached.len < input.len) return error.ShapeMismatch;
        rmsNormWithWeight(input, cached[0..input.len], output[0..input.len], model.config.rms_norm_eps);
        return;
    }
    if (scratch.len < input.len) return error.ShapeMismatch;
    const weights = scratch[0..input.len];
    try readTensorFlat(model.tensorData(info), info.type_, weights);
    rmsNormWithWeight(input, weights, output[0..input.len], model.config.rms_norm_eps);
}

fn rmsNormTensorWithFirstWeight(
    model: *const Model,
    info: gguf.TensorInfo,
    input: []const f32,
    output: []f32,
    scratch: []f32,
    first_weight: f32,
    direct_compute_tracking: ?DirectComputeTracking,
) !void {
    if (output.len < input.len) return error.ShapeMismatch;
    const weights = blk: {
        if (model.cachedF32(info)) |cached| {
            if (cached.len < input.len) return error.ShapeMismatch;
            break :blk cached[0..input.len];
        }
        if (scratch.len < input.len) return error.ShapeMismatch;
        const tmp = scratch[0..input.len];
        try readTensorFlat(model.tensorData(info), info.type_, tmp);
        break :blk tmp;
    };
    const inv_rms = rmsNormWithWeightInv(input, weights, output[0..input.len], model.config.rms_norm_eps);
    if (input.len == 0) return;

    const cpu_elem0 = input[0] * inv_rms * first_weight;
    if (direct_compute_tracking) |tracking| {
        const gpu_elem0 = tracking.boundary.rmsNormElement0(input[0], inv_rms, first_weight) catch |err| blk: {
            log.warn("M1 AMDGPU CS direct rms_norm elem0 unavailable ({s}); final norm element remains host-computed", .{@errorName(err)});
            break :blk null;
        };
        if (gpu_elem0) |value| {
            output[0] = value;
            tracking.ops.* += 1;
            mergeDirectComputeKind(tracking.kind, .rms_norm_elem0);
            tracking.consumed.* = true;
            log.info("M1 AMDGPU CS direct compute consumed: direct_compute_ops={d} direct_compute_kind=rms_norm_elem0 cpu={d:.6} gpu={d:.6}", .{
                tracking.ops.*,
                cpu_elem0,
                value,
            });
            return;
        }
    }

    output[0] = cpu_elem0;
}

fn rmsNormWithWeight(input: []const f32, weight: []const f32, output: []f32, eps: f32) void {
    _ = rmsNormWithWeightInv(input, weight, output, eps);
}

fn rmsNormWithWeightInv(input: []const f32, weight: []const f32, output: []f32, eps: f32) f32 {
    const inv_rms = rmsNormInvRms(input, eps);
    const Vec16f = @Vector(16, f32);
    const inv_v: Vec16f = @splat(inv_rms);
    var i: usize = 0;
    while (i + 64 <= input.len) : (i += 64) {
        const x0: Vec16f = input[i..][0..16].*;
        const x1: Vec16f = input[i + 16 ..][0..16].*;
        const x2: Vec16f = input[i + 32 ..][0..16].*;
        const x3: Vec16f = input[i + 48 ..][0..16].*;
        const w0: Vec16f = weight[i..][0..16].*;
        const w1: Vec16f = weight[i + 16 ..][0..16].*;
        const w2: Vec16f = weight[i + 32 ..][0..16].*;
        const w3: Vec16f = weight[i + 48 ..][0..16].*;
        output[i..][0..16].* = (x0 * inv_v) * w0;
        output[i + 16 ..][0..16].* = (x1 * inv_v) * w1;
        output[i + 32 ..][0..16].* = (x2 * inv_v) * w2;
        output[i + 48 ..][0..16].* = (x3 * inv_v) * w3;
    }
    while (i + 16 <= input.len) : (i += 16) {
        const x: Vec16f = input[i..][0..16].*;
        const w: Vec16f = weight[i..][0..16].*;
        output[i..][0..16].* = (x * inv_v) * w;
    }
    while (i < input.len) : (i += 1) output[i] = input[i] * inv_rms * weight[i];
    return inv_rms;
}

fn rmsNormInvRms(input: []const f32, eps: f32) f32 {
    const Vec16f = @Vector(16, f32);
    var acc0: Vec16f = @splat(0.0);
    var acc1: Vec16f = @splat(0.0);
    var acc2: Vec16f = @splat(0.0);
    var acc3: Vec16f = @splat(0.0);
    var i: usize = 0;
    while (i + 64 <= input.len) : (i += 64) {
        const v0: Vec16f = input[i..][0..16].*;
        const v1: Vec16f = input[i + 16 ..][0..16].*;
        const v2: Vec16f = input[i + 32 ..][0..16].*;
        const v3: Vec16f = input[i + 48 ..][0..16].*;
        acc0 = @mulAdd(Vec16f, v0, v0, acc0);
        acc1 = @mulAdd(Vec16f, v1, v1, acc1);
        acc2 = @mulAdd(Vec16f, v2, v2, acc2);
        acc3 = @mulAdd(Vec16f, v3, v3, acc3);
    }
    var acc_vec: Vec16f = (acc0 + acc1) + (acc2 + acc3);
    while (i + 16 <= input.len) : (i += 16) {
        const v: Vec16f = input[i..][0..16].*;
        acc_vec = @mulAdd(Vec16f, v, v, acc_vec);
    }
    var sq = @reduce(.Add, acc_vec);
    while (i < input.len) : (i += 1) sq += input[i] * input[i];

    return 1.0 / @sqrt(sq / @as(f32, @floatFromInt(input.len)) + eps);
}

fn rmsNormHeads(model: *const Model, info: gguf.TensorInfo, data: []f32, head_dim: u32, n_heads: u32) !void {
    if (head_dim == 0) return error.ShapeMismatch;
    if (model.cachedF32(info)) |cached| {
        if (cached.len < head_dim) return error.ShapeMismatch;
        const weights = cached[0..head_dim];
        for (0..n_heads) |h| {
            const head = data[h * head_dim ..][0..head_dim];
            rmsNormWithWeight(head, weights, head, model.config.rms_norm_eps);
        }
        return;
    }
    var weights_buf: [1024]f32 = undefined;
    if (head_dim > weights_buf.len) return error.UnsupportedShape;
    const weights = weights_buf[0..head_dim];
    try readTensorFlat(model.tensorData(info), info.type_, weights);
    for (0..n_heads) |h| {
        const head = data[h * head_dim ..][0..head_dim];
        rmsNormWithWeight(head, weights, head, model.config.rms_norm_eps);
    }
}

fn readTensorFlat(raw: []const u8, tensor_type: gguf.GGMLType, output: []f32) !void {
    if (output.len == 0) return;
    try dequant.row(raw, 0, @intCast(output.len), tensor_type, output);
}

/// Gemma 4 normalizes V vectors per head with unit weights (no learned scale)
/// before they enter the KV cache. Equivalent to `x / rms(x)` per head.
fn applyVUnitNormPerHead(data: []f32, head_dim: u32, n_heads: u32, eps: f32) void {
    const head_dim_usize: usize = @intCast(head_dim);
    for (0..n_heads) |h| {
        const head = data[h * head_dim_usize ..][0..head_dim_usize];
        const inv = rmsNormInvRms(head, eps);
        const Vec16f = @Vector(16, f32);
        const inv_v: Vec16f = @splat(inv);
        var i: usize = 0;
        while (i + 64 <= head.len) : (i += 64) {
            const x0: Vec16f = head[i..][0..16].*;
            const x1: Vec16f = head[i + 16 ..][0..16].*;
            const x2: Vec16f = head[i + 32 ..][0..16].*;
            const x3: Vec16f = head[i + 48 ..][0..16].*;
            head[i..][0..16].* = x0 * inv_v;
            head[i + 16 ..][0..16].* = x1 * inv_v;
            head[i + 32 ..][0..16].* = x2 * inv_v;
            head[i + 48 ..][0..16].* = x3 * inv_v;
        }
        while (i + 16 <= head.len) : (i += 16) {
            const x: Vec16f = head[i..][0..16].*;
            head[i..][0..16].* = x * inv_v;
        }
        while (i < head.len) : (i += 1) head[i] = head[i] * inv;
    }
}

fn deinterleaveQGate(src: []const f32, q: []f32, gate: []f32, head_dim: u32, n_heads: u32) void {
    for (0..n_heads) |h| {
        const src_head = src[h * head_dim * 2 ..][0 .. head_dim * 2];
        @memcpy(q[h * head_dim ..][0..head_dim], src_head[0..head_dim]);
        @memcpy(gate[h * head_dim ..][0..head_dim], src_head[head_dim..][0..head_dim]);
    }
}

fn applyRope(data: []f32, stride: u32, rope_dim: u32, n_heads: u32, position: u32, inv_freq: []const f32) void {
    const half = rope_dim / 2;
    // Precompute the per-frequency sin/cos pairs once per call instead of once
    // per (head, freq). `position` is constant across heads at a given attn
    // layer, so the trig values are loop-invariant in `h`; the hot decode path
    // pays for these trig calls 10 attn layers × (n_heads + n_kv_heads) ×
    // half-rope times per token otherwise. 1024 entries cover head_dim ≤ 1024
    // (Gemma 4 31B global layers use head_dim=512 → trig_needed=512).
    var trig_buf: [1024]f32 = undefined;
    const trig_needed: usize = @as(usize, @intCast(half)) * 2;
    if (trig_needed > trig_buf.len) {
        // Fallback for unexpectedly large rope: inline trig per element.
        for (0..n_heads) |h| {
            const base = h * stride;
            for (0..half) |i| {
                const freq = if (i < inv_freq.len) inv_freq[i] else 0.0;
                const theta = @as(f32, @floatFromInt(position)) * freq;
                const c = @cos(theta);
                const s = @sin(theta);
                const a = data[base + i];
                const b = data[base + i + half];
                data[base + i] = a * c - b * s;
                data[base + i + half] = a * s + b * c;
            }
        }
        return;
    }
    const pos_f: f32 = @floatFromInt(position);
    for (0..half) |i| {
        const freq = if (i < inv_freq.len) inv_freq[i] else 0.0;
        const theta = pos_f * freq;
        trig_buf[i * 2] = @cos(theta);
        trig_buf[i * 2 + 1] = @sin(theta);
    }
    for (0..n_heads) |h| {
        const base = h * stride;
        for (0..half) |i| {
            const c = trig_buf[i * 2];
            const s = trig_buf[i * 2 + 1];
            const a = data[base + i];
            const b = data[base + i + half];
            data[base + i] = a * c - b * s;
            data[base + i + half] = a * s + b * c;
        }
    }
}

inline fn flashAttnDot(a: []const f32, b: []const f32) f32 {
    std.debug.assert(a.len == b.len);
    const Vec16f = @Vector(16, f32);
    var acc0: Vec16f = @splat(0.0);
    var acc1: Vec16f = @splat(0.0);
    var acc2: Vec16f = @splat(0.0);
    var acc3: Vec16f = @splat(0.0);
    var i: usize = 0;
    while (i + 64 <= a.len) : (i += 64) {
        const a0: Vec16f = a[i..][0..16].*;
        const a1: Vec16f = a[i + 16 ..][0..16].*;
        const a2: Vec16f = a[i + 32 ..][0..16].*;
        const a3: Vec16f = a[i + 48 ..][0..16].*;
        const b0: Vec16f = b[i..][0..16].*;
        const b1: Vec16f = b[i + 16 ..][0..16].*;
        const b2: Vec16f = b[i + 32 ..][0..16].*;
        const b3: Vec16f = b[i + 48 ..][0..16].*;
        acc0 = @mulAdd(Vec16f, a0, b0, acc0);
        acc1 = @mulAdd(Vec16f, a1, b1, acc1);
        acc2 = @mulAdd(Vec16f, a2, b2, acc2);
        acc3 = @mulAdd(Vec16f, a3, b3, acc3);
    }
    var acc_vec: Vec16f = (acc0 + acc1) + (acc2 + acc3);
    while (i + 16 <= a.len) : (i += 16) {
        const av: Vec16f = a[i..][0..16].*;
        const bv: Vec16f = b[i..][0..16].*;
        acc_vec = @mulAdd(Vec16f, av, bv, acc_vec);
    }
    var dot = @reduce(.Add, acc_vec);
    while (i < a.len) : (i += 1) dot += a[i] * b[i];
    return dot;
}

fn flashAttentionCpu(
    model: *const Model,
    state: *ScalarDecodeState,
    layer: u32,
    position: u32,
    head_dim: u32,
    n_kv_heads: u32,
    n_q_heads: u32,
) !void {
    const cfg = model.config;
    const q_per_kv = @max(n_q_heads / @max(n_kv_heads, 1), 1);
    const scale: f32 = if (cfg.attn_scale > 0)
        cfg.attn_scale
    else
        1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
    for (0..n_q_heads) |h| {
        const kv_head = h / q_per_kv;
        const q_head = state.q[h * head_dim ..][0..head_dim];
        var max_score: f32 = -std.math.inf(f32);
        for (0..position + 1) |pos| {
            const kv_off = state.kvOffset(cfg, layer, @intCast(pos));
            const k_head = state.kv_k[kv_off + kv_head * head_dim ..][0..head_dim];
            // Manual 4-accumulator Vec16f reduce. The scalar `dot += qv * kv` chains
            // every iteration through the same FP-add, so Zig/LLVM cannot
            // auto-vectorise without fastmath; head_dim is typically a multiple of
            // 64 (128 in Qwen3.6) so the wide loop dominates.
            const dot = flashAttnDot(q_head, k_head);
            const score = dot * scale;
            state.scores[pos] = score;
            if (score > max_score) max_score = score;
        }
        const sink = model.attn_sinks[@as(usize, layer) * cfg.n_heads + h];
        if (!std.math.isNan(sink) and sink > max_score) max_score = sink;
        var sum_exp: f32 = 0;
        if (!std.math.isNan(sink)) sum_exp += @exp(sink - max_score);
        for (0..position + 1) |pos| {
            const p = @exp(state.scores[pos] - max_score);
            state.probs[pos] = p;
            sum_exp += p;
        }
        const inv_sum = if (sum_exp > 0) 1.0 / sum_exp else 0;
        const out_head = state.attn_out[h * head_dim ..][0..head_dim];
        @memset(out_head, 0);
        for (0..position + 1) |pos| {
            const weight = state.probs[pos] * inv_sum;
            const kv_off = state.kvOffset(cfg, layer, @intCast(pos));
            const v_head = state.kv_v[kv_off + kv_head * head_dim ..][0..head_dim];
            for (out_head, v_head) |*out, vv| out.* += weight * vv;
        }
    }
}

fn l2Normalize(v: []f32) void {
    const Vec16f = @Vector(16, f32);
    var acc0: Vec16f = @splat(0.0);
    var acc1: Vec16f = @splat(0.0);
    var acc2: Vec16f = @splat(0.0);
    var acc3: Vec16f = @splat(0.0);
    var i: usize = 0;
    while (i + 64 <= v.len) : (i += 64) {
        const v0: Vec16f = v[i..][0..16].*;
        const v1: Vec16f = v[i + 16 ..][0..16].*;
        const v2: Vec16f = v[i + 32 ..][0..16].*;
        const v3: Vec16f = v[i + 48 ..][0..16].*;
        acc0 = @mulAdd(Vec16f, v0, v0, acc0);
        acc1 = @mulAdd(Vec16f, v1, v1, acc1);
        acc2 = @mulAdd(Vec16f, v2, v2, acc2);
        acc3 = @mulAdd(Vec16f, v3, v3, acc3);
    }
    var acc_vec: Vec16f = (acc0 + acc1) + (acc2 + acc3);
    while (i + 16 <= v.len) : (i += 16) {
        const x: Vec16f = v[i..][0..16].*;
        acc_vec = @mulAdd(Vec16f, x, x, acc_vec);
    }
    var sq = @reduce(.Add, acc_vec);
    while (i < v.len) : (i += 1) sq += v[i] * v[i];

    const inv = if (sq > 0) 1.0 / @sqrt(sq + 1e-12) else 0.0;
    const inv_v: Vec16f = @splat(inv);
    i = 0;
    while (i + 64 <= v.len) : (i += 64) {
        const v0: Vec16f = v[i..][0..16].*;
        const v1: Vec16f = v[i + 16 ..][0..16].*;
        const v2: Vec16f = v[i + 32 ..][0..16].*;
        const v3: Vec16f = v[i + 48 ..][0..16].*;
        v[i..][0..16].* = v0 * inv_v;
        v[i + 16 ..][0..16].* = v1 * inv_v;
        v[i + 32 ..][0..16].* = v2 * inv_v;
        v[i + 48 ..][0..16].* = v3 * inv_v;
    }
    while (i + 16 <= v.len) : (i += 16) {
        const x: Vec16f = v[i..][0..16].*;
        v[i..][0..16].* = x * inv_v;
    }
    while (i < v.len) : (i += 1) v[i] *= inv;
}

fn swiglu(gate: []const f32, up: []const f32, output: []f32) void {
    const Vec16f = @Vector(16, f32);
    const one: Vec16f = @splat(1.0);
    var i: usize = 0;
    while (i + 16 <= output.len) : (i += 16) {
        const g: Vec16f = gate[i..][0..16].*;
        const u: Vec16f = up[i..][0..16].*;
        const sig = one / (one + @exp(-g));
        output[i..][0..16].* = (g * sig) * u;
    }
    while (i < output.len) : (i += 1) {
        const g = gate[i];
        output[i] = (g * sigmoid(g)) * up[i];
    }
}

/// Gemma 4 uses gelu(gate) * up (tanh-approx GELU matches GGML's gelu_quick).
/// Matches llama.cpp build_ffn with LLM_FFN_GELU + LLM_FFN_PAR.
fn geluGate(gate: []const f32, up: []const f32, output: []f32) void {
    // tanh-approx GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    const k_sqrt_2_over_pi: f32 = 0.7978845608028654;
    const k_cubic: f32 = 0.044715;
    var i: usize = 0;
    while (i < output.len) : (i += 1) {
        const g = gate[i];
        const inner = k_sqrt_2_over_pi * (g + k_cubic * g * g * g);
        const t = std.math.tanh(inner);
        const gel = 0.5 * g * (1.0 + t);
        output[i] = gel * up[i];
    }
}

fn sigmoid(x: f32) f32 {
    return 1.0 / (1.0 + @exp(-x));
}

fn softplus(x: f32) f32 {
    if (x > 20.0) return x;
    if (x < -20.0) return @exp(x);
    return @log(1.0 + @exp(x));
}

fn topKSoftmaxCpu(logits: []const f32, k: u32, out_ids: []u32, out_weights: []f32) void {
    if (k == 0) return;
    if (k == 1) {
        var best_idx: u32 = 0;
        var best_val = logits[0];
        for (logits[1..], 1..) |v, i| {
            if (v > best_val) {
                best_val = v;
                best_idx = @intCast(i);
            }
        }
        out_ids[0] = best_idx;
        out_weights[0] = 1.0;
        return;
    }

    var used = [_]bool{false} ** 256;
    for (0..k) |slot| {
        var best_idx: u32 = 0;
        var best_val: f32 = -std.math.inf(f32);
        for (0..logits.len) |i| {
            if (!used[i] and logits[i] > best_val) {
                best_val = logits[i];
                best_idx = @intCast(i);
            }
        }
        out_ids[slot] = best_idx;
        out_weights[slot] = best_val;
        used[best_idx] = true;
    }

    var selected_max: f32 = -std.math.inf(f32);
    for (out_weights[0..k]) |w| selected_max = @max(selected_max, w);

    var selected_sum: f32 = 0;
    for (out_weights[0..k]) |*w| {
        w.* = @exp(w.* - selected_max);
        selected_sum += w.*;
    }
    if (selected_sum > 0) {
        for (out_weights[0..k]) |*w| w.* /= selected_sum;
    }
}

fn argmaxSlice(values: []const f32) u32 {
    var best_idx: u32 = 0;
    var best = values[0];
    for (values[1..], 1..) |v, i| {
        if (v > best) {
            best = v;
            best_idx = @intCast(i);
        }
    }
    return best_idx;
}

fn argmaxTop2Slice(values: []const f32) ArgmaxTop2Result {
    var best: ArgmaxTop2Result = .{};
    for (values, 0..) |v, i| best.offer(@intCast(i), v);
    return best;
}

fn expertSliceBytes(tensor_type: gguf.GGMLType, rows: u32, cols: u32) u32 {
    const bs = tensor_type.blockSize();
    const bpb = tensor_type.bytesPerBlock();
    if (bs == 0 or bpb == 0) return rows * cols * @sizeOf(f32);
    return rows * (cols / bs) * bpb;
}

/// Build a `Tokenizer` from `model`'s GGUF vocab metadata. Convenience
/// wrapper around `Tokenizer.init` so callers don't have to reach into the
/// `Model`'s parse state.
/// @param model Loaded GGUF model whose vocab table is consulted.
/// @param allocator Owns the resulting tokenizer's vocab and id table.
/// @returns A ready-to-use `Tokenizer`.
pub fn initTokenizer(model: *const Model, allocator: std.mem.Allocator) !Tokenizer {
    return Tokenizer.init(&model.gguf_file, allocator);
}

/// Minimal longest-match BPE-ish tokenizer that reads the vocab and EOS id
/// straight out of a GGUF file. Encodes prompts via the GPT-2 byte-to-unicode
/// mapping and falls back to byte 0 on misses so the scalar M1 forward path
/// always sees a well-formed token stream.
pub const Tokenizer = struct {
    allocator: std.mem.Allocator,
    vocab: []const []const u8,
    token_to_id: std.StringHashMapUnmanaged(u32),
    eos_id: u32,
    bos_id: ?u32,
    add_bos: bool,
    /// Gemma 4 (and other SentencePiece-flavoured vocabs) encode spaces as ▁
    /// (U+2581) and store raw UTF-8 in the vocab table, rather than the GPT-2
    /// byte-to-unicode mapping used by Qwen-family BPE.
    is_sentencepiece: bool,

    /// Build a tokenizer by reading `tokenizer.ggml.tokens` and
    /// `tokenizer.ggml.eos_token_id` out of `gf`. Defaults the EOS id to `2`
    /// when the metadata key is missing.
    /// @param gf GGUF file whose tokenizer metadata is consulted.
    /// @param allocator Owns the vocab slice and the id hash table.
    /// @returns A `Tokenizer` ready for `encodePrompt` / `decodeToken`.
    pub fn init(gf: *const gguf.GGUFFile, allocator: std.mem.Allocator) !Tokenizer {
        const tokens_val = gf.metadata.get("tokenizer.ggml.tokens") orelse return error.NoTokenizerVocab;
        const tokens_array = switch (tokens_val) {
            .array => |items| items,
            else => return error.NoTokenizerVocab,
        };

        var vocab = try allocator.alloc([]const u8, tokens_array.len);
        errdefer allocator.free(vocab);

        var token_to_id: std.StringHashMapUnmanaged(u32) = .{};
        errdefer token_to_id.deinit(allocator);
        try token_to_id.ensureTotalCapacity(allocator, @intCast(tokens_array.len));

        for (tokens_array, 0..) |item, index| {
            const token = item.asString() orelse return error.NoTokenizerVocab;
            vocab[index] = token;
            try token_to_id.put(allocator, token, @intCast(index));
        }

        const bos_id = gf.getU32("tokenizer.ggml.bos_token_id");
        // Default for Gemma 4: prepend BOS unless metadata says otherwise.
        // For other architectures, default to false (most don't prepend).
        const arch_str = gf.getString("general.architecture") orelse "";
        const default_add_bos = std.mem.startsWith(u8, arch_str, "gemma");
        const add_bos = gf.getBool("tokenizer.ggml.add_bos_token") orelse default_add_bos;
        const tokenizer_model = gf.getString("tokenizer.ggml.model") orelse "";
        const pre_name = gf.getString("tokenizer.ggml.pre") orelse "";
        const is_sentencepiece = std.mem.eql(u8, tokenizer_model, "gemma4") or
            std.mem.eql(u8, tokenizer_model, "llama") or
            std.mem.eql(u8, pre_name, "gemma4");
        return .{
            .allocator = allocator,
            .vocab = vocab,
            .token_to_id = token_to_id,
            .eos_id = gf.getU32("tokenizer.ggml.eos_token_id") orelse 2,
            .bos_id = bos_id,
            .add_bos = add_bos and bos_id != null,
            .is_sentencepiece = is_sentencepiece,
        };
    }

    /// Release the vocab slice and the id hash table, then poison the handle.
    pub fn deinit(self: *Tokenizer) void {
        self.token_to_id.deinit(self.allocator);
        self.allocator.free(self.vocab);
        self.* = undefined;
    }

    /// Return the stop token id the caller should pass to `generate`.
    pub fn eosId(self: *const Tokenizer) u32 {
        return self.eos_id;
    }

    /// Wrap the user prompt with Gemma's chat-turn special tokens so the
    /// instruction-tuned model has the expected scaffolding. Returns null when
    /// the vocab doesn't carry the Gemma `<start_of_turn>` / `<end_of_turn>`
    /// special-token strings (i.e. the model isn't Gemma-templated).
    pub fn encodeGemmaChat(self: *const Tokenizer, user_text: []const u8, allocator: std.mem.Allocator) !?[]u32 {
        // Gemma 2/3 use <start_of_turn>/<end_of_turn>. Gemma 4 uses <|turn>/<turn|>.
        const start_id = self.token_to_id.get("<|turn>") orelse
            self.token_to_id.get("<start_of_turn>") orelse return null;
        const end_id = self.token_to_id.get("<turn|>") orelse
            self.token_to_id.get("<end_of_turn>") orelse return null;
        const newline_id = self.token_to_id.get("\n") orelse self.token_to_id.get("Ċ"); // GPT-2 mapping of '\n'

        var tokens: std.ArrayList(u32) = .empty;
        errdefer tokens.deinit(allocator);
        // The Gemma chat template starts with `{{ bos_token }}` regardless of
        // `tokenizer.ggml.add_bos_token` (some Gemma 4 GGUFs ship that flag
        // as false even though the template still wants BOS).
        if (self.bos_id) |bos| try tokens.append(allocator, bos);
        try tokens.append(allocator, start_id);
        // "user\n" — encode through the longest-match scanner, then strip BOS
        // if it added one (we're only emitting the role label here).
        const user_label_tokens = try self.encodePromptNoBos("user", allocator);
        defer allocator.free(user_label_tokens);
        for (user_label_tokens) |t| try tokens.append(allocator, t);
        if (newline_id) |nid| try tokens.append(allocator, nid);
        const body_tokens = try self.encodePromptNoBos(user_text, allocator);
        defer allocator.free(body_tokens);
        for (body_tokens) |t| try tokens.append(allocator, t);
        try tokens.append(allocator, end_id);
        if (newline_id) |nid| try tokens.append(allocator, nid);
        try tokens.append(allocator, start_id);
        const model_label_tokens = try self.encodePromptNoBos("model", allocator);
        defer allocator.free(model_label_tokens);
        for (model_label_tokens) |t| try tokens.append(allocator, t);
        if (newline_id) |nid| try tokens.append(allocator, nid);
        return try tokens.toOwnedSlice(allocator);
    }

    /// Prepare `text` into a byte stream the longest-match scanner consumes.
    /// SentencePiece vocabs (Gemma 4, llama) substitute spaces with the
    /// SPIECE underline (▁, U+2581) and pass raw UTF-8 through unchanged;
    /// GPT-2-flavour vocabs (Qwen) remap every byte through the byte-to-
    /// unicode table so the scanner sees vocab-shaped pieces.
    fn prepBytesForEncode(self: *const Tokenizer, text: []const u8, out: *std.ArrayList(u8), allocator: std.mem.Allocator) !void {
        if (self.is_sentencepiece) {
            for (text) |byte| {
                if (byte == ' ') {
                    try out.appendSlice(allocator, "\xe2\x96\x81");
                } else {
                    try out.append(allocator, byte);
                }
            }
        } else {
            for (text) |byte| {
                const mapped = gpt2ByteToUnicode(byte);
                const n = std.mem.indexOfScalar(u8, &mapped, 0) orelse mapped.len;
                try out.appendSlice(allocator, mapped[0..n]);
            }
        }
    }

    fn encodePromptNoBos(self: *const Tokenizer, text: []const u8, allocator: std.mem.Allocator) ![]u32 {
        var encoded: std.ArrayList(u8) = .empty;
        defer encoded.deinit(allocator);
        try self.prepBytesForEncode(text, &encoded, allocator);
        var tokens: std.ArrayList(u32) = .empty;
        errdefer tokens.deinit(allocator);
        var pos: usize = 0;
        while (pos < encoded.items.len) {
            var best_id: ?u32 = null;
            var best_len: usize = 0;
            var end = encoded.items.len;
            while (end > pos) : (end -= 1) {
                const piece = encoded.items[pos..end];
                if (self.token_to_id.get(piece)) |id| {
                    best_id = id;
                    best_len = piece.len;
                    break;
                }
            }
            if (best_id) |id| {
                try tokens.append(allocator, id);
                pos += best_len;
            } else {
                try tokens.append(allocator, 0);
                pos += 1;
            }
        }
        return tokens.toOwnedSlice(allocator);
    }

    /// Encode `text` into a token id stream using a longest-match scan over
    /// the GPT-2 byte-to-unicode mapping. Unmatched single bytes fall back to
    /// token id 0 so the output is always well-formed.
    /// @param text Raw UTF-8 prompt bytes.
    /// @param allocator Owns the returned token slice.
    /// @returns Token ids ready to feed into `generate`.
    pub fn encodePrompt(self: *const Tokenizer, text: []const u8, allocator: std.mem.Allocator) ![]u32 {
        var encoded: std.ArrayList(u8) = .empty;
        defer encoded.deinit(allocator);
        try self.prepBytesForEncode(text, &encoded, allocator);

        var tokens: std.ArrayList(u32) = .empty;
        errdefer tokens.deinit(allocator);
        if (self.add_bos) {
            if (self.bos_id) |bos| try tokens.append(allocator, bos);
        }
        var pos: usize = 0;
        while (pos < encoded.items.len) {
            var best_id: ?u32 = null;
            var best_len: usize = 0;
            var end = encoded.items.len;
            while (end > pos) : (end -= 1) {
                const piece = encoded.items[pos..end];
                if (self.token_to_id.get(piece)) |id| {
                    best_id = id;
                    best_len = piece.len;
                    break;
                }
            }
            if (best_id) |id| {
                try tokens.append(allocator, id);
                pos += best_len;
            } else {
                try tokens.append(allocator, 0);
                pos += 1;
            }
        }
        return tokens.toOwnedSlice(allocator);
    }

    /// Render one token id back to its UTF-8 byte form into `buf`, reversing
    /// the GPT-2 byte-to-unicode mapping. Truncates instead of erroring when
    /// `buf` is too small; returns an empty slice for out-of-range ids.
    /// @param token_id Token id produced by `generate` or `encodePrompt`.
    /// @param buf Scratch buffer the decoded bytes are written into.
    /// @returns The prefix of `buf` containing the decoded bytes.
    pub fn decodeToken(self: *const Tokenizer, token_id: u32, buf: []u8) []const u8 {
        if (token_id >= self.vocab.len) return "";
        const token = self.vocab[token_id];
        var out: usize = 0;
        var i: usize = 0;
        while (i < token.len and out < buf.len) {
            const cp_len = utf8SequenceLength(token[i]);
            if (i + cp_len > token.len) break;
            const cp = decodeUtf8Codepoint(token[i .. i + cp_len]) catch {
                buf[out] = token[i];
                out += 1;
                i += 1;
                continue;
            };
            if (self.is_sentencepiece) {
                if (cp == 0x2581) {
                    buf[out] = ' ';
                    out += 1;
                } else {
                    const n = @min(cp_len, buf.len - out);
                    @memcpy(buf[out..][0..n], token[i..][0..n]);
                    out += n;
                }
                i += cp_len;
                continue;
            }
            const byte = gpt2UnicodeToByte(cp) orelse {
                const n = @min(cp_len, buf.len - out);
                @memcpy(buf[out..][0..n], token[i..][0..n]);
                out += n;
                i += cp_len;
                continue;
            };
            buf[out] = byte;
            out += 1;
            i += cp_len;
        }
        return buf[0..out];
    }
};

fn gpt2ByteToUnicode(byte: u8) [4]u8 {
    const cp: u21 = switch (byte) {
        '!'...'~', 0xA1...0xAC, 0xAE...0xFF => byte,
        else => @as(u21, 256) + @as(u21, switch (byte) {
            0...0x20 => byte,
            0x7F...0xA0 => byte - 0x7F + 33,
            0xAD => 33 + 34,
            else => byte,
        }),
    };

    var buf: [4]u8 = .{ 0, 0, 0, 0 };
    if (cp < 0x80) {
        buf[0] = @intCast(cp);
    } else if (cp < 0x800) {
        buf[0] = @intCast(0xC0 | (cp >> 6));
        buf[1] = @intCast(0x80 | (cp & 0x3F));
    } else {
        buf[0] = @intCast(0xE0 | (cp >> 12));
        buf[1] = @intCast(0x80 | ((cp >> 6) & 0x3F));
        buf[2] = @intCast(0x80 | (cp & 0x3F));
    }
    return buf;
}

fn gpt2UnicodeToByte(cp: u21) ?u8 {
    return switch (cp) {
        '!'...'~', 0xA1...0xAC, 0xAE...0xFF => @intCast(cp),
        0x0100...0x0120 => @intCast(cp - 0x0100),
        0x0121...0x0142 => @intCast(cp - 162),
        0x0143 => 0xAD,
        else => null,
    };
}

fn utf8SequenceLength(byte0: u8) usize {
    if (byte0 < 0x80) return 1;
    if ((byte0 & 0xE0) == 0xC0) return 2;
    if ((byte0 & 0xF0) == 0xE0) return 3;
    if ((byte0 & 0xF8) == 0xF0) return 4;
    return 1;
}

fn decodeUtf8Codepoint(bytes: []const u8) !u21 {
    return switch (bytes.len) {
        1 => bytes[0],
        2 => (@as(u21, bytes[0] & 0x1F) << 6) | @as(u21, bytes[1] & 0x3F),
        3 => (@as(u21, bytes[0] & 0x0F) << 12) |
            (@as(u21, bytes[1] & 0x3F) << 6) |
            @as(u21, bytes[2] & 0x3F),
        4 => (@as(u21, bytes[0] & 0x07) << 18) |
            (@as(u21, bytes[1] & 0x3F) << 12) |
            (@as(u21, bytes[2] & 0x3F) << 6) |
            @as(u21, bytes[3] & 0x3F),
        else => error.InvalidUtf8,
    };
}

fn evalToken(
    model: *const Model,
    rt: *cpu_ring.CpuRing,
    token_id: u32,
    hidden: []f32,
    norm: []f32,
    row_scratch: []f32,
    logits: []f32,
    next_token: *u32,
) !void {
    const packets = [_]ring.Packet{
        .{ .embed = .{
            .raw_data = model.tensorData(model.embed_info),
            .tensor_type = model.embed_info.type_,
            .token_id = token_id,
            .hidden_dim = model.hidden_dim,
            .vocab_size = model.vocab_size,
            .output = hidden,
        } },
        .{ .rms_norm = .{
            .input = hidden,
            .weight = model.final_norm_weight,
            .output = norm,
            .eps = model.rms_norm_eps,
        } },
        .{ .lm_head = .{
            .raw_data = model.tensorData(model.lm_head_info),
            .tensor_type = model.lm_head_info.type_,
            .hidden = norm,
            .row_scratch = row_scratch,
            .logits = logits,
        } },
        .{ .argmax = .{
            .logits = logits,
            .output = next_token,
        } },
    };
    try rt.submitAndWait(.{ .packets = &packets });
}

fn inferMatrixDims(info: gguf.TensorInfo) struct { cols: u32, rows: u32 } {
    if (info.n_dims == 1) {
        return .{ .cols = @intCast(info.dims[0]), .rows = 1 };
    }
    if (info.n_dims >= 2) {
        return .{ .cols = @intCast(info.dims[0]), .rows = @intCast(info.dims[1]) };
    }
    return .{ .cols = 0, .rows = 0 };
}

fn inferFlatMatrixDims(info: gguf.TensorInfo) struct { cols: u32, rows: u32 } {
    if (info.n_dims == 0) return .{ .cols = 0, .rows = 0 };
    const cols: u32 = @intCast(info.dims[0]);
    var rows: u64 = 1;
    for (info.dims[1..info.n_dims]) |dim| rows *= dim;
    if (info.n_dims == 1) rows = 1;
    return .{ .cols = cols, .rows = @intCast(rows) };
}

fn tensorElements(info: gguf.TensorInfo) u64 {
    var n: u64 = 1;
    for (info.dims[0..info.n_dims]) |dim| n *= dim;
    return n;
}

fn tensorDataRaw(
    mmap_data: []align(std.heap.page_size_min) const u8,
    gf: *const gguf.GGUFFile,
    info: gguf.TensorInfo,
) []const u8 {
    const start: usize = @intCast(gf.tensor_data_offset + info.offset);
    const size: usize = @intCast(info.sizeBytes());
    return mmap_data[start..][0..size];
}

fn rmsNormEps(gf: *const gguf.GGUFFile) f32 {
    const arch = gf.getString("general.architecture") orelse return 1e-6;
    var key_buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(
        &key_buf,
        "{s}.attention.layer_norm_rms_epsilon",
        .{arch},
    ) catch return 1e-6;
    return gf.getF32(key) orelse 1e-6;
}

fn archU32(gf: *const gguf.GGUFFile, arch: []const u8, suffix: []const u8, default: u32) u32 {
    var key_buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ arch, suffix }) catch return default;
    return gf.getU32(key) orelse default;
}

fn archF32(gf: *const gguf.GGUFFile, arch: []const u8, suffix: []const u8, default: f32) f32 {
    var key_buf: [128]u8 = undefined;
    const key = std.fmt.bufPrint(&key_buf, "{s}.{s}", .{ arch, suffix }) catch return default;
    return gf.getF32(key) orelse default;
}

fn emitDecodeGraphForShape(
    allocator: std.mem.Allocator,
    n_layers: u32,
    full_attn_interval_raw: u32,
    n_experts: u32,
    has_ssm: bool,
) !DecodeGraphSummary {
    var graph = zinc_rt.ir_graph.Graph.init(allocator);
    defer graph.deinit();

    const token = graph.addBuffer();
    const hidden = graph.addBuffer();
    const qkv = graph.addBuffer();
    const attn_out = graph.addBuffer();
    const ssm_out = graph.addBuffer();
    const router = graph.addBuffer();
    const moe_gate_up = graph.addBuffer();
    const moe_act = graph.addBuffer();
    const norm = graph.addBuffer();
    const logits = graph.addBuffer();
    const next_token = graph.addBuffer();

    _ = try graph.addNode(.embed, &.{token}, &.{hidden});

    const full_attn_interval = if (full_attn_interval_raw > 0) full_attn_interval_raw else 1;
    var attention_layers: u32 = 0;
    var ssm_layers: u32 = 0;
    var moe_layers: u32 = 0;

    for (0..n_layers) |layer_index| {
        const layer: u32 = @intCast(layer_index);
        const is_full_attn = !has_ssm or ((layer + 1) % full_attn_interval == 0);

        _ = try graph.addNode(.rms_norm_fused_qkv, &.{hidden}, &.{qkv});
        if (is_full_attn) {
            attention_layers += 1;
            _ = try graph.addNode(.rope, &.{qkv}, &.{qkv});
            _ = try graph.addNode(.flash_attn, &.{qkv}, &.{attn_out});
            _ = try graph.addNode(.residual_rms_norm, &.{ hidden, attn_out }, &.{hidden});
        } else {
            ssm_layers += 1;
            _ = try graph.addNode(.ssm_conv1d, &.{qkv}, &.{ssm_out});
            _ = try graph.addNode(.ssm_delta_net, &.{ssm_out}, &.{ssm_out});
            _ = try graph.addNode(.ssm_gated_norm, &.{ssm_out}, &.{ssm_out});
            _ = try graph.addNode(.residual_rms_norm, &.{ hidden, ssm_out }, &.{hidden});
        }

        if (n_experts > 0) {
            moe_layers += 1;
            _ = try graph.addNode(.moe_gate_topk, &.{hidden}, &.{router});
            _ = try graph.addNode(.moe_gate_up, &.{ hidden, router }, &.{moe_gate_up});
            _ = try graph.addNode(.moe_swiglu, &.{moe_gate_up}, &.{moe_act});
            _ = try graph.addNode(.moe_down_acc, &.{ moe_act, router, hidden }, &.{hidden});
            _ = try graph.addNode(.shared_expert, &.{hidden}, &.{hidden});
        }
    }

    _ = try graph.addNode(.rms_norm, &.{hidden}, &.{norm});
    _ = try graph.addNode(.lm_head, &.{norm}, &.{logits});
    _ = try graph.addNode(.argmax, &.{logits}, &.{next_token});

    try graph.verify();
    return .{
        .nodes = @intCast(graph.nodes.items.len),
        .layers = n_layers,
        .attention_layers = attention_layers,
        .ssm_layers = ssm_layers,
        .moe_layers = moe_layers,
    };
}

fn nanoTimestamp() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(std.c.CLOCK.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

fn elapsedNs(start: i128, end: i128) u64 {
    if (end <= start) return 0;
    return @intCast(end - start);
}

test "inferMatrixDims treats GGUF matrix dims as cols then rows" {
    const info = gguf.TensorInfo{
        .name = "output.weight",
        .n_dims = 2,
        .dims = .{ 2048, 151936, 1, 1 },
        .type_ = .q8_0,
        .offset = 0,
    };
    const dims = inferMatrixDims(info);
    try std.testing.expectEqual(@as(u32, 2048), dims.cols);
    try std.testing.expectEqual(@as(u32, 151936), dims.rows);
}

test "inferFlatMatrixDims folds expert tensors into a row-major matrix" {
    const info = gguf.TensorInfo{
        .name = "blk.0.ffn_down_exps.weight",
        .n_dims = 3,
        .dims = .{ 512, 2048, 256, 1 },
        .type_ = .q5_k,
        .offset = 0,
    };
    const dims = inferFlatMatrixDims(info);
    try std.testing.expectEqual(@as(u32, 512), dims.cols);
    try std.testing.expectEqual(@as(u32, 2048 * 256), dims.rows);
}

test "elapsedNs clamps non-positive durations" {
    try std.testing.expectEqual(@as(u64, 0), elapsedNs(10, 9));
    try std.testing.expectEqual(@as(u64, 5), elapsedNs(10, 15));
}

test "mergeDirectComputeKind preserves both consumed direct slices" {
    var kind: DirectComputeKind = .none;
    mergeDirectComputeKind(&kind, .rms_norm_elem0);
    try std.testing.expectEqual(DirectComputeKind.rms_norm_elem0, kind);
    mergeDirectComputeKind(&kind, .argmax);
    try std.testing.expectEqual(DirectComputeKind.argmax_rms_norm_elem0, kind);

    kind = .argmax;
    mergeDirectComputeKind(&kind, .rms_norm_elem0);
    try std.testing.expectEqual(DirectComputeKind.argmax_rms_norm_elem0, kind);
}

test "BenchmarkShortcutFlags reports any active shortcut" {
    try std.testing.expect(!(BenchmarkShortcutFlags{}).any());
    try std.testing.expect((BenchmarkShortcutFlags{ .decode_moe_topk_zero = true }).any());
    try std.testing.expect((BenchmarkShortcutFlags{ .lm_head_rows_capped = true }).any());
    try std.testing.expect((BenchmarkShortcutFlags{ .decode_budget_clamped = true }).any());
}

test "resolveQwen36MoeTopkLimit mirrors Vulkan default and env escape hatch" {
    const cfg = CpuModelConfig{
        .hidden_dim = 2048,
        .vocab_size = 248320,
        .n_layers = 40,
        .n_heads = 16,
        .n_kv_heads = 2,
        .head_dim = 256,
        .q_dim = 4096,
        .kv_dim = 512,
        .intermediate_dim = 768,
        .n_experts = 128,
        .n_experts_used = 8,
        .rope_dim = 64,
        .rope_freq_base = 10000000.0,
        .full_attn_interval = 4,
        .shared_expert_intermediate_dim = 768,
        .ssm_d_conv = 4,
        .ssm_d_inner = 4096,
        .ssm_d_state = 128,
        .ssm_dt_rank = 32,
        .ssm_n_group = 16,
        .rms_norm_eps = 0.000001,
    };
    const layers = [_]LayerTensors{.{
        .ssm_alpha = testTensorInfo(.f32),
        .ssm_beta = testTensorInfo(.f32),
    }};

    try std.testing.expect(isQwen36LikeF32Ssm(cfg, &layers));
    // No env override → no cap (use metadata top-k). Setting "3" opts back
    // into the historical benchmark shortcut.
    try std.testing.expectEqual(@as(u32, 0), resolveQwen36MoeTopkLimitForEnv(cfg, &layers, null));
    try std.testing.expectEqual(@as(u32, 3), resolveQwen36MoeTopkLimitForEnv(cfg, &layers, "3"));
    try std.testing.expectEqual(@as(u32, 2), resolveQwen36MoeTopkLimitForEnv(cfg, &layers, "2"));
    try std.testing.expectEqual(@as(u32, 0), resolveQwen36MoeTopkLimitForEnv(cfg, &layers, "8"));
    try std.testing.expectEqual(@as(u32, 0), resolveQwen36MoeTopkLimitForEnv(cfg, &layers, "0"));
}

test "resolveQwen36LmHeadDecodeRows caps only the Qwen36 M1 decode path" {
    const cfg = CpuModelConfig{
        .hidden_dim = 2048,
        .vocab_size = 248320,
        .n_layers = 40,
        .n_heads = 16,
        .n_kv_heads = 2,
        .head_dim = 256,
        .q_dim = 4096,
        .kv_dim = 512,
        .intermediate_dim = 768,
        .n_experts = 128,
        .n_experts_used = 8,
        .rope_dim = 64,
        .rope_freq_base = 10000000.0,
        .full_attn_interval = 4,
        .shared_expert_intermediate_dim = 768,
        .ssm_d_conv = 4,
        .ssm_d_inner = 4096,
        .ssm_d_state = 128,
        .ssm_dt_rank = 32,
        .ssm_n_group = 16,
        .rms_norm_eps = 0.000001,
    };
    const qwen_layers = [_]LayerTensors{.{
        .ssm_alpha = testTensorInfo(.f32),
        .ssm_beta = testTensorInfo(.f32),
    }};
    const dense_layers = [_]LayerTensors{.{}};

    // No env override → full vocab. Setting an explicit row count opts back
    // into the historical benchmark shortcut.
    try std.testing.expectEqual(cfg.vocab_size, resolveQwen36LmHeadDecodeRowsForEnv(cfg, &qwen_layers, null));
    try std.testing.expectEqual(@as(u32, 4 * 1024), resolveQwen36LmHeadDecodeRowsForEnv(cfg, &qwen_layers, "4096"));
    try std.testing.expectEqual(@as(u32, 32768), resolveQwen36LmHeadDecodeRowsForEnv(cfg, &qwen_layers, "32768"));
    try std.testing.expectEqual(cfg.vocab_size, resolveQwen36LmHeadDecodeRowsForEnv(cfg, &qwen_layers, "0"));
    try std.testing.expectEqual(cfg.vocab_size, resolveQwen36LmHeadDecodeRowsForEnv(cfg, &dense_layers, null));
}

test "topKSoftmaxCpu returns selected-only renormalized softmax weights" {
    const logits = [_]f32{ -2.0, 1.5, 0.25, 4.0, -0.5, 3.0, 2.5, -1.0 };
    var top1_id = [_]u32{0};
    var top1_weight = [_]f32{0.0};
    topKSoftmaxCpu(&logits, 1, &top1_id, &top1_weight);
    try std.testing.expectEqual(@as(u32, 3), top1_id[0]);
    try std.testing.expectEqual(@as(f32, 1.0), top1_weight[0]);

    const k = 4;
    var ids: [k]u32 = undefined;
    var weights: [k]f32 = undefined;
    topKSoftmaxCpu(&logits, k, &ids, &weights);

    try std.testing.expectEqual(@as(u32, 3), ids[0]);
    try std.testing.expectEqual(@as(u32, 5), ids[1]);
    try std.testing.expectEqual(@as(u32, 6), ids[2]);
    try std.testing.expectEqual(@as(u32, 1), ids[3]);

    var max_logit: f32 = -std.math.inf(f32);
    for (ids) |id| max_logit = @max(max_logit, logits[id]);
    var expected: [k]f32 = undefined;
    var sum: f32 = 0.0;
    for (ids, 0..) |id, i| {
        expected[i] = @exp(logits[id] - max_logit);
        sum += expected[i];
    }
    for (&expected) |*w| w.* /= sum;
    for (expected, weights) |want, got| {
        try std.testing.expectApproxEqAbs(want, got, 1e-6);
    }
}

test "routeTop1SharedGate returns router argmax and shared gate dot" {
    const router_vals = [_]f32{
        1.0,  0.0,
        -1.0, 2.0,
        0.5,  0.5,
    };
    const shared_vals = [_]f32{ 3.0, -1.0 };
    const input = [_]f32{ 1.0, 2.0 };
    var scratch: [2]f32 = undefined;

    const result = try routeTop1SharedGate(
        null,
        .{ .raw = std.mem.sliceAsBytes(&router_vals), .type_ = .f32 },
        3,
        .{ .raw = std.mem.sliceAsBytes(&shared_vals), .type_ = .f32 },
        &input,
        &scratch,
    );

    try std.testing.expectEqual(@as(u32, 1), result.expert_id);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), result.shared_gate, 0.00001);
}

test "Qwen router-sized matvec uses the persistent pool" {
    try std.testing.expectEqual(@as(usize, 4), matvecWorkerCountForCpu(128, 2048, 4));
    try std.testing.expectEqual(@as(usize, 1), matvecWorkerCountForCpu(127, 2048, 4));
}

test "matvecWorkerCountForCpu parallelizes large non-LM projections" {
    try std.testing.expectEqual(@as(usize, 4), matvecWorkerCountForCpu(256, 2048, 16));
    try std.testing.expectEqual(@as(usize, 4), matvecWorkerCountForCpu(768, 2048, 16));
    try std.testing.expectEqual(@as(usize, 4), matvecWorkerCountForCpu(2048, 2048, 16));
    try std.testing.expectEqual(@as(usize, 8), matvecWorkerCountForCpu(4096, 2048, 16));
    try std.testing.expectEqual(@as(usize, 16), matvecWorkerCountForCpu(12288, 2048, 32));
    try std.testing.expectEqual(@as(usize, 1), matvecWorkerCountForCpu(12288, 2048, 1));
}

test "moeExpertWorkerCount parallelizes one branch per selected expert" {
    try std.testing.expectEqual(@as(usize, 1), moeExpertWorkerCount(1, 16));
    try std.testing.expectEqual(@as(usize, 3), moeExpertWorkerCount(3, 16));
    try std.testing.expectEqual(@as(usize, 1), moeExpertWorkerCount(3, 2));
    try std.testing.expectEqual(@as(usize, 8), moeExpertWorkerCount(8, 32));
    try std.testing.expectEqual(@as(usize, 1), moeExpertWorkerCount(9, 32));
}

fn fillUnitQ4KBlock(block: *[144]u8) void {
    @memset(block, 0);
    const one_bits: u16 = @bitCast(@as(f16, 1.0));
    std.mem.writeInt(u16, block[0..2], one_bits, .little);
    std.mem.writeInt(u16, block[2..4], one_bits, .little);
    for (block[4..8]) |*b| b.* = 1;
    for (block[8..16]) |*b| b.* = 0x11;
    for (block[16..144]) |*b| b.* = 0x22;
}

test "fused Q4_K matvec can use precomputed input sums" {
    var raw_a: [144]u8 = undefined;
    var raw_b: [144]u8 = undefined;
    fillUnitQ4KBlock(&raw_a);
    fillUnitQ4KBlock(&raw_b);

    var input: [256]f32 = undefined;
    var expected: f32 = 0.0;
    for (&input, 0..) |*v, i| {
        v.* = (@as(f32, @floatFromInt((i * 7) % 31)) - 15.0) * 0.125;
        expected += v.*;
    }

    var sums: [8]f32 = undefined;
    dequant.fillInputSum32(&input, &sums);

    var out_a = [_]f32{0.0};
    var out_b = [_]f32{0.0};
    const segs = [_]FusedSegment{
        .{ .raw = &raw_a, .type_ = .q4_k, .rows = 1, .out = &out_a },
        .{ .raw = &raw_b, .type_ = .q4_k, .rows = 1, .out = &out_b },
    };
    try matvecFused(null, &segs, &input, &sums);

    try std.testing.expectApproxEqAbs(expected, out_a[0], 0.0001);
    try std.testing.expectApproxEqAbs(expected, out_b[0], 0.0001);
}

test "multi-input fused matvec handles disjoint segment inputs" {
    const raw_a_vals = [_]f32{
        1.0,  2.0,
        -1.0, 0.5,
    };
    const raw_b_vals = [_]f32{ 2.0, 0.5, -1.0 };
    const input_a = [_]f32{ 3.0, 4.0 };
    const input_b = [_]f32{ 1.0, 2.0, 3.0 };
    var out_a = [_]f32{ 0.0, 0.0 };
    var out_b = [_]f32{0.0};
    const segs = [_]MultiInputFusedSegment{
        .{
            .raw = std.mem.sliceAsBytes(&raw_a_vals),
            .type_ = .f32,
            .input = &input_a,
            .rows = 2,
            .out = &out_a,
        },
        .{
            .raw = std.mem.sliceAsBytes(&raw_b_vals),
            .type_ = .f32,
            .input = &input_b,
            .rows = 1,
            .out = &out_b,
        },
    };

    try matvecMultiInputFusedSerial(&segs, 0, 3);

    try std.testing.expectApproxEqAbs(@as(f32, 11.0), out_a[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), out_a[1], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), out_b[0], 0.0001);
}

fn testTensorInfo(tensor_type: gguf.GGMLType) gguf.TensorInfo {
    return .{
        .name = "test",
        .n_dims = 1,
        .dims = .{ 1, 1, 1, 1 },
        .type_ = tensor_type,
        .offset = 0,
    };
}

test "vectorized normalization helpers match scalar references" {
    const len: usize = 90;
    var input: [len]f32 = undefined;
    var weight: [len]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 17) % 61)) - 30.0) * 0.03125;
    for (&weight, 0..) |*v, i| v.* = 0.5 + @as(f32, @floatFromInt((i * 11) % 29)) * 0.01;

    var got: [len]f32 = undefined;
    rmsNormWithWeight(&input, &weight, &got, 0.000001);

    var sq: f32 = 0;
    for (input) |x| sq += x * x;
    const inv_rms = 1.0 / @sqrt(sq / @as(f32, @floatFromInt(len)) + 0.000001);
    for (0..len) |i| {
        const expected = input[i] * inv_rms * weight[i];
        try std.testing.expectApproxEqAbs(expected, got[i], 0.00001);
    }

    var l2_got = input;
    l2Normalize(&l2_got);
    var l2_sq: f32 = 0;
    for (input) |x| l2_sq += x * x;
    const l2_inv = if (l2_sq > 0) 1.0 / @sqrt(l2_sq + 1e-12) else 0.0;
    for (0..len) |i| {
        try std.testing.expectApproxEqAbs(input[i] * l2_inv, l2_got[i], 0.00001);
    }
}

test "vectorized swiglu matches scalar reference" {
    const len: usize = 37;
    var gate: [len]f32 = undefined;
    var up: [len]f32 = undefined;
    for (&gate, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 13) % 41)) - 20.0) * 0.125;
    for (&up, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 7) % 29)) - 14.0) * 0.0625;

    var got: [len]f32 = undefined;
    swiglu(&gate, &up, &got);

    for (0..len) |i| {
        const expected = (gate[i] * sigmoid(gate[i])) * up[i];
        try std.testing.expectApproxEqAbs(expected, got[i], 0.00001);
    }
}

test "fused SSM rank update and output projection matches separate passes" {
    const rows: usize = 3;
    const cols: usize = 96;
    const kv_len: usize = 80;
    var matrix_separate: [rows * cols]f32 = undefined;
    var matrix_fused: [rows * cols]f32 = undefined;
    var k: [kv_len]f32 = undefined;
    var q: [kv_len]f32 = undefined;

    for (&matrix_separate, 0..) |*v, i| {
        v.* = (@as(f32, @floatFromInt((i * 17) % 53)) - 26.0) * 0.015625;
    }
    matrix_fused = matrix_separate;
    for (&k, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 7) % 31)) - 15.0) * 0.03125;
    for (&q, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 11) % 37)) - 18.0) * 0.02734375;

    for (0..rows) |row| {
        const row_base = row * cols;
        const d_val = (@as(f32, @floatFromInt(row)) - 1.0) * 0.125;
        ssmRank1UpdateRow(&matrix_separate, row_base, &k, kv_len, d_val);
        const expected = ssmOutProjRow(&matrix_separate, row_base, &q, kv_len);
        const got = ssmRank1UpdateAndOutProjRow(&matrix_fused, row_base, &k, &q, kv_len, d_val);

        try std.testing.expectApproxEqAbs(expected, got, 0.00001);
        for (0..cols) |col| {
            try std.testing.expectApproxEqAbs(matrix_separate[row_base + col], matrix_fused[row_base + col], 0.00001);
        }
    }
}

test "vectorized dconv4 conv silu matches scalar reference" {
    const conv_ch: usize = 37;
    const d_conv: usize = 4;
    var conv_kernel: [conv_ch * d_conv]f32 = undefined;
    var conv_state: [conv_ch * (d_conv - 1)]f32 = undefined;
    var qkv: [conv_ch]f32 = undefined;
    var got: [conv_ch]f32 = undefined;

    for (&conv_kernel, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 9) % 31)) - 15.0) * 0.03125;
    for (&conv_state, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 7) % 23)) - 11.0) * 0.0625;
    for (&qkv, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 5) % 19)) - 9.0) * 0.05;

    const ctx = ConvSiluCtx{
        .conv_kernel = &conv_kernel,
        .conv_state = &conv_state,
        .qkv = &qkv,
        .conv_out = &got,
        .d_conv = d_conv,
        .d_conv_1 = d_conv - 1,
        .conv_ch = conv_ch,
        .kernel_transposed_d4 = false,
    };
    runConvSiluRange(&ctx, 0, conv_ch);

    for (0..conv_ch) |ch| {
        var sum: f32 = 0;
        for (0..d_conv) |ki| {
            const sv = if (ki < d_conv - 1) conv_state[ki * conv_ch + ch] else qkv[ch];
            sum += conv_kernel[ch * d_conv + ki] * sv;
        }
        try std.testing.expectApproxEqAbs(sum * sigmoid(sum), got[ch], 0.00001);
    }
}

test "transposed dconv4 kernel path matches row-major conv silu" {
    const conv_ch: usize = 37;
    const d_conv: usize = 4;
    var row_major: [conv_ch * d_conv]f32 = undefined;
    var transposed: [conv_ch * d_conv]f32 = undefined;
    var conv_state: [conv_ch * (d_conv - 1)]f32 = undefined;
    var qkv: [conv_ch]f32 = undefined;
    var got_row: [conv_ch]f32 = undefined;
    var got_transposed: [conv_ch]f32 = undefined;

    for (&row_major, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 9) % 31)) - 15.0) * 0.03125;
    for (&conv_state, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 7) % 23)) - 11.0) * 0.0625;
    for (&qkv, 0..) |*v, i| v.* = (@as(f32, @floatFromInt((i * 5) % 19)) - 9.0) * 0.05;
    transposeDConv4Kernel(&row_major, &transposed, conv_ch);

    const row_ctx = ConvSiluCtx{
        .conv_kernel = &row_major,
        .conv_state = &conv_state,
        .qkv = &qkv,
        .conv_out = &got_row,
        .d_conv = d_conv,
        .d_conv_1 = d_conv - 1,
        .conv_ch = conv_ch,
        .kernel_transposed_d4 = false,
    };
    runConvSiluRange(&row_ctx, 0, conv_ch);

    const transposed_ctx = ConvSiluCtx{
        .conv_kernel = &transposed,
        .conv_state = &conv_state,
        .qkv = &qkv,
        .conv_out = &got_transposed,
        .d_conv = d_conv,
        .d_conv_1 = d_conv - 1,
        .conv_ch = conv_ch,
        .kernel_transposed_d4 = true,
    };
    runConvSiluRange(&transposed_ctx, 0, conv_ch);

    for (got_row, got_transposed) |expected, actual| {
        try std.testing.expectApproxEqAbs(expected, actual, 0.00001);
    }
}

test "emitDecodeGraphForShape covers all hybrid MoE layers" {
    const summary = try emitDecodeGraphForShape(std.testing.allocator, 40, 4, 60, true);
    try std.testing.expectEqual(@as(u32, 40), summary.layers);
    try std.testing.expectEqual(@as(u32, 10), summary.attention_layers);
    try std.testing.expectEqual(@as(u32, 30), summary.ssm_layers);
    try std.testing.expectEqual(@as(u32, 40), summary.moe_layers);
    try std.testing.expect(summary.nodes > 300);
}

test "emitDecodeGraphForShape treats dense models as all attention" {
    const summary = try emitDecodeGraphForShape(std.testing.allocator, 4, 1, 0, false);
    try std.testing.expectEqual(@as(u32, 4), summary.attention_layers);
    try std.testing.expectEqual(@as(u32, 0), summary.ssm_layers);
    try std.testing.expectEqual(@as(u32, 0), summary.moe_layers);
}
