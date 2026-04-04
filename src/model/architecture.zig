//! Build static decode graphs for the supported model families.
//! @section Decode Planning
//! These graphs describe the logical order of decode-time operations so runtime
//! code can bind buffers and record compute work against a stable structure.
const std = @import("std");
const graph_mod = @import("../compute/graph.zig");
const Graph = graph_mod.Graph;
const OpType = graph_mod.OpType;
const ExecDomain = graph_mod.ExecDomain;
const loader = @import("loader.zig");
const ModelConfig = loader.ModelConfig;
const gguf = @import("gguf.zig");

const log = std.log.scoped(.architecture);

const MetricSpec = struct {
    layer_index: ?u32 = null,
    domain: ExecDomain = .gpu_compute,
    read_bytes: u64 = 0,
    write_bytes: u64 = 0,
    weight_bytes: u64 = 0,
    flops: u64 = 0,
    workgroups: [3]u32 = .{ 1, 1, 1 },
    threads_per_workgroup: u32 = 64,
    requires_host_sync: bool = false,
    note: ?[]const u8 = null,
};

fn divCeilU32(n: u32, d: u32) u32 {
    return @intCast((@as(u64, n) + d - 1) / d);
}

fn vecBytes(n: anytype) u64 {
    return @as(u64, @intCast(n)) * @sizeOf(f32);
}

fn approxF16TensorBytes(rows: anytype, cols: anytype) u64 {
    return @as(u64, @intCast(rows)) * @as(u64, @intCast(cols)) * @sizeOf(u16);
}

fn layerTensorBytes(gf: ?*const gguf.GGUFFile, layer_index: u32, suffix: []const u8, fallback: u64) u64 {
    if (gf) |file| {
        var buf: [128]u8 = undefined;
        const full_name = std.fmt.bufPrint(&buf, "blk.{d}.{s}", .{ layer_index, suffix }) catch return fallback;
        if (file.findTensor(full_name)) |tensor| return tensor.sizeBytes();
    }
    return fallback;
}

fn globalTensorBytes(gf: ?*const gguf.GGUFFile, name: []const u8, fallback: u64) u64 {
    if (gf) |file| {
        if (file.findTensor(name)) |tensor| return tensor.sizeBytes();
    }
    return fallback;
}

fn addAnnotatedNode(g: *Graph, layer_index: ?u32, op: OpType, short_name: []const u8, metrics: MetricSpec) !u32 {
    var label_buf: [96]u8 = undefined;
    const label = if (layer_index) |layer|
        try std.fmt.bufPrint(&label_buf, "blk.{d}.{s}", .{ layer, short_name })
    else
        short_name;

    const id = try g.addNode(op, label);
    g.setLayerIndex(id, if (metrics.layer_index != null) metrics.layer_index else layer_index);
    g.setExecDomain(id, metrics.domain);
    g.setWorkgroups(id, metrics.workgroups[0], metrics.workgroups[1], metrics.workgroups[2]);
    g.setThreadsPerWorkgroup(id, metrics.threads_per_workgroup);
    g.setCostEstimate(id, metrics.read_bytes, metrics.write_bytes, metrics.weight_bytes, metrics.flops);
    g.setHostSync(id, metrics.requires_host_sync);
    g.setNote(id, metrics.note);
    return id;
}

fn embedMetrics(hidden_dim: u32, row_bytes: u64) MetricSpec {
    return .{
        .domain = .cpu_host,
        .read_bytes = row_bytes,
        .write_bytes = vecBytes(hidden_dim),
        .requires_host_sync = true,
        .note = "Embedding dequant/upload is host-driven in the current runtime.",
    };
}

fn rmsNormMetrics(width: u32, tokens: u32, weight_bytes: u64) MetricSpec {
    const total_elems = @as(u64, width) * @as(u64, tokens);
    return .{
        .read_bytes = vecBytes(total_elems),
        .write_bytes = vecBytes(total_elems),
        .weight_bytes = weight_bytes,
        .flops = total_elems * 5,
        .workgroups = .{ tokens, 1, 1 },
    };
}

fn dmmvMetrics(total_rows: u32, k_dim: u32, weight_bytes: u64, rows_per_x: u32, workgroups_y: u32) MetricSpec {
    return .{
        .read_bytes = vecBytes(k_dim),
        .write_bytes = vecBytes(total_rows),
        .weight_bytes = weight_bytes,
        .flops = 2 * @as(u64, total_rows) * @as(u64, k_dim),
        .workgroups = .{ divCeilU32(rows_per_x, 64), workgroups_y, 1 },
    };
}

fn ropeMetrics(head_dim: u32, n_heads: u32, rope_dim: u32) MetricSpec {
    const elems = @as(u64, head_dim) * @as(u64, n_heads);
    return .{
        .read_bytes = vecBytes(elems),
        .write_bytes = vecBytes(elems),
        .flops = @as(u64, rope_dim) * @as(u64, n_heads) * 6,
        .workgroups = .{ n_heads, 1, 1 },
    };
}

fn kvWriteMetrics(kv_dim: u32) MetricSpec {
    const bytes = vecBytes(@as(u64, kv_dim) * 2);
    return .{
        .domain = .gpu_transfer,
        .read_bytes = bytes,
        .write_bytes = bytes,
        .note = "KV cache writes are copy traffic, not compute.",
    };
}

fn flashAttnMetrics(head_dim: u32, n_heads: u32, n_kv_heads: u32, seq_len: u32) MetricSpec {
    const q_elems = @as(u64, head_dim) * @as(u64, n_heads);
    const kv_elems = @as(u64, seq_len) * @as(u64, head_dim) * @as(u64, n_kv_heads) * 2;
    const out_elems = q_elems;
    return .{
        .read_bytes = vecBytes(q_elems + kv_elems),
        .write_bytes = vecBytes(out_elems),
        .flops = 4 * @as(u64, head_dim) * @as(u64, n_heads) * @as(u64, seq_len),
        .workgroups = .{ n_heads, 1, 1 },
    };
}

fn addMetrics(n_elems: u32) MetricSpec {
    return .{
        .read_bytes = vecBytes(@as(u64, n_elems) * 2),
        .write_bytes = vecBytes(n_elems),
        .flops = @as(u64, n_elems),
        .workgroups = .{ divCeilU32(n_elems, 64), 1, 1 },
    };
}

fn swigluMetrics(n_elems: u32) MetricSpec {
    return .{
        .read_bytes = vecBytes(@as(u64, n_elems) * 2),
        .write_bytes = vecBytes(n_elems),
        .flops = @as(u64, n_elems) * 8,
        .workgroups = .{ divCeilU32(n_elems, 64), 1, 1 },
    };
}

fn moeGatherMetrics(hidden_dim: u32, n_used: u32) MetricSpec {
    return .{
        .read_bytes = vecBytes(@as(u64, hidden_dim) * @as(u64, n_used) + @as(u64, n_used)),
        .write_bytes = vecBytes(hidden_dim),
        .flops = 2 * @as(u64, hidden_dim) * @as(u64, n_used),
        .workgroups = .{ divCeilU32(hidden_dim, 64), 1, 1 },
    };
}

/// Build a compute graph for a single transformer decode step.
/// This creates the graph structure; actual buffer bindings are set at runtime.
/// @param config Normalized model dimensions and architecture metadata.
/// @param allocator Allocator used for graph storage.
/// @returns A Graph describing the decode-time op order for the selected architecture.
pub fn buildDecodeGraph(config: *const ModelConfig, allocator: std.mem.Allocator) !Graph {
    return buildDecodeGraphDetailed(config, allocator, null);
}

/// Build a compute graph with per-op weight-size annotations derived from a GGUF file.
pub fn buildDecodeGraphDetailed(config: *const ModelConfig, allocator: std.mem.Allocator, gf: ?*const gguf.GGUFFile) !Graph {
    return switch (config.architecture) {
        .mistral, .qwen2 => try buildLlamaDecodeGraph(config, allocator, gf),
        .qwen2_moe => try buildMoeDecodeGraph(config, allocator, gf),
        .qwen35, .mamba, .jamba => try buildMambaDecodeGraph(config, allocator, gf),
        .unknown => error.UnsupportedArchitecture,
    };
}

/// Standard transformer decode graph (LLaMA/Mistral/Qwen2).
/// Per-layer structure:
///   input_norm → QKV projection → RoPE → flash attention → residual add →
///   ffn_norm → gate+up projection → SwiGLU → down projection → residual add
fn buildLlamaDecodeGraph(config: *const ModelConfig, allocator: std.mem.Allocator, gf: ?*const gguf.GGUFFile) !Graph {
    var g = Graph.init(allocator, "llama_decode");
    errdefer g.deinit();

    const q_dim = config.n_heads * config.head_dim;
    const kv_dim = config.n_kv_heads * config.head_dim;
    const inter_dim = if (config.intermediate_dim > 0) config.intermediate_dim else config.hidden_dim * 4;
    const rope_dim = if (config.rope_dim > 0) config.rope_dim else config.head_dim;
    const seq_len = @min(config.context_length, 4096);
    g.setAssumedDecodeSeqLen(seq_len);

    // Token embedding lookup
    const embed = try addAnnotatedNode(&g, null, .embed, "token_embed", .{
        .domain = .cpu_host,
        .read_bytes = globalTensorBytes(gf, "token_embd.weight", approxF16TensorBytes(config.vocab_size, config.hidden_dim)) / @max(@as(u64, 1), @as(u64, config.vocab_size)),
        .write_bytes = vecBytes(config.hidden_dim),
        .requires_host_sync = true,
        .note = "Embedding dequant/upload is host-driven in the current runtime.",
    });

    var prev_residual = embed;

    for (0..config.n_layers) |layer_idx| {
        const layer: u32 = @intCast(layer_idx);

        // Input normalization
        const input_norm = try addAnnotatedNode(&g, layer, .rms_norm_mul, "input_norm", rmsNormMetrics(
            config.hidden_dim,
            1,
            layerTensorBytes(gf, layer, "attn_norm.weight", config.hidden_dim * @sizeOf(f32)),
        ));
        g.addDependency(input_norm, prev_residual);

        // QKV projection (decode = matmul-vec)
        const q_proj = try addAnnotatedNode(&g, layer, .dmmv, "q_proj", dmmvMetrics(
            q_dim,
            config.hidden_dim,
            layerTensorBytes(gf, layer, "attn_q.weight", approxF16TensorBytes(q_dim, config.hidden_dim)),
            q_dim,
            1,
        ));
        g.addDependency(q_proj, input_norm);

        const k_proj = try addAnnotatedNode(&g, layer, .dmmv, "k_proj", dmmvMetrics(
            kv_dim,
            config.hidden_dim,
            layerTensorBytes(gf, layer, "attn_k.weight", approxF16TensorBytes(kv_dim, config.hidden_dim)),
            kv_dim,
            1,
        ));
        g.addDependency(k_proj, input_norm);

        const v_proj = try addAnnotatedNode(&g, layer, .dmmv, "v_proj", dmmvMetrics(
            kv_dim,
            config.hidden_dim,
            layerTensorBytes(gf, layer, "attn_v.weight", approxF16TensorBytes(kv_dim, config.hidden_dim)),
            kv_dim,
            1,
        ));
        g.addDependency(v_proj, input_norm);

        // RoPE on Q and K
        const rope_q = try addAnnotatedNode(&g, layer, .rope, "rope_q", ropeMetrics(config.head_dim, config.n_heads, rope_dim));
        g.addDependency(rope_q, q_proj);

        const rope_k = try addAnnotatedNode(&g, layer, .rope, "rope_k", ropeMetrics(config.head_dim, config.n_kv_heads, rope_dim));
        g.addDependency(rope_k, k_proj);

        // Write K/V to cache
        const kv_write = try addAnnotatedNode(&g, layer, .kv_cache_write, "kv_write", kvWriteMetrics(kv_dim));
        g.addDependency(kv_write, rope_k);
        g.addDependency(kv_write, v_proj);

        // Flash attention
        const attn = try addAnnotatedNode(&g, layer, .flash_attn, "flash_attn", flashAttnMetrics(
            config.head_dim,
            config.n_heads,
            config.n_kv_heads,
            seq_len,
        ));
        g.addDependency(attn, rope_q);
        g.addDependency(attn, kv_write);

        // Attention output projection
        const o_proj = try addAnnotatedNode(&g, layer, .dmmv, "o_proj", dmmvMetrics(
            config.hidden_dim,
            q_dim,
            layerTensorBytes(gf, layer, "attn_output.weight", approxF16TensorBytes(config.hidden_dim, q_dim)),
            config.hidden_dim,
            1,
        ));
        g.addDependency(o_proj, attn);

        // Residual add
        const attn_residual = try addAnnotatedNode(&g, layer, .add, "attn_residual", addMetrics(config.hidden_dim));
        g.addDependency(attn_residual, prev_residual);
        g.addDependency(attn_residual, o_proj);

        // FFN normalization
        const ffn_norm = try addAnnotatedNode(&g, layer, .rms_norm_mul, "ffn_norm", rmsNormMetrics(
            config.hidden_dim,
            1,
            layerTensorBytes(gf, layer, "ffn_norm.weight", config.hidden_dim * @sizeOf(f32)),
        ));
        g.addDependency(ffn_norm, attn_residual);

        // Gate + Up projection (SwiGLU FFN)
        const gate_proj = try addAnnotatedNode(&g, layer, .dmmv, "gate_proj", dmmvMetrics(
            inter_dim,
            config.hidden_dim,
            layerTensorBytes(gf, layer, "ffn_gate.weight", approxF16TensorBytes(inter_dim, config.hidden_dim)),
            inter_dim,
            1,
        ));
        g.addDependency(gate_proj, ffn_norm);

        const up_proj = try addAnnotatedNode(&g, layer, .dmmv, "up_proj", dmmvMetrics(
            inter_dim,
            config.hidden_dim,
            layerTensorBytes(gf, layer, "ffn_up.weight", approxF16TensorBytes(inter_dim, config.hidden_dim)),
            inter_dim,
            1,
        ));
        g.addDependency(up_proj, ffn_norm);

        // SwiGLU activation
        const swiglu = try addAnnotatedNode(&g, layer, .swiglu, "swiglu", swigluMetrics(inter_dim));
        g.addDependency(swiglu, gate_proj);
        g.addDependency(swiglu, up_proj);

        // Down projection
        const down_proj = try addAnnotatedNode(&g, layer, .dmmv, "down_proj", dmmvMetrics(
            config.hidden_dim,
            inter_dim,
            layerTensorBytes(gf, layer, "ffn_down.weight", approxF16TensorBytes(config.hidden_dim, inter_dim)),
            config.hidden_dim,
            1,
        ));
        g.addDependency(down_proj, swiglu);

        // FFN residual add
        const ffn_residual = try addAnnotatedNode(&g, layer, .add, "ffn_residual", addMetrics(config.hidden_dim));
        g.addDependency(ffn_residual, attn_residual);
        g.addDependency(ffn_residual, down_proj);

        prev_residual = ffn_residual;
    }

    // Final normalization
    const final_norm = try addAnnotatedNode(&g, null, .rms_norm_mul, "final_norm", rmsNormMetrics(
        config.hidden_dim,
        1,
        globalTensorBytes(gf, "output_norm.weight", config.hidden_dim * @sizeOf(f32)),
    ));
    g.addDependency(final_norm, prev_residual);

    // Output projection (lm_head)
    const lm_head = try addAnnotatedNode(&g, null, .dmmv, "lm_head", dmmvMetrics(
        config.vocab_size,
        config.hidden_dim,
        blk: {
            const out_bytes = globalTensorBytes(gf, "output.weight", 0);
            if (out_bytes > 0) break :blk out_bytes;
            break :blk globalTensorBytes(gf, "token_embd.weight", approxF16TensorBytes(config.vocab_size, config.hidden_dim));
        },
        config.vocab_size,
        1,
    ));
    g.addDependency(lm_head, final_norm);

    log.info("Built LLaMA decode graph: {d} nodes, {d} layers", .{
        g.nodeCount(), config.n_layers,
    });

    return g;
}

/// MoE transformer decode graph (Qwen2-MoE).
/// Same as LLaMA but FFN is replaced with expert routing + sparse expert matmuls.
fn buildMoeDecodeGraph(config: *const ModelConfig, allocator: std.mem.Allocator, gf: ?*const gguf.GGUFFile) !Graph {
    var g = Graph.init(allocator, "moe_decode");
    errdefer g.deinit();

    const q_dim = config.n_heads * config.head_dim;
    const kv_dim = config.n_kv_heads * config.head_dim;
    const inter_dim = if (config.intermediate_dim > 0) config.intermediate_dim else config.hidden_dim * 4;
    const rope_dim = if (config.rope_dim > 0) config.rope_dim else config.head_dim;
    const seq_len = @min(config.context_length, 4096);
    const n_used = if (config.n_experts_used > 0) config.n_experts_used else 8;
    g.setAssumedDecodeSeqLen(seq_len);

    const embed = try addAnnotatedNode(&g, null, .embed, "token_embed", .{
        .domain = .cpu_host,
        .read_bytes = globalTensorBytes(gf, "token_embd.weight", approxF16TensorBytes(config.vocab_size, config.hidden_dim)) / @max(@as(u64, 1), @as(u64, config.vocab_size)),
        .write_bytes = vecBytes(config.hidden_dim),
        .requires_host_sync = true,
        .note = "Embedding dequant/upload is host-driven in the current runtime.",
    });
    var prev_residual = embed;

    for (0..config.n_layers) |layer_idx| {
        const layer: u32 = @intCast(layer_idx);
        // Attention block (same as LLaMA)
        const input_norm = try addAnnotatedNode(&g, layer, .rms_norm_mul, "input_norm", rmsNormMetrics(
            config.hidden_dim,
            1,
            layerTensorBytes(gf, layer, "attn_norm.weight", config.hidden_dim * @sizeOf(f32)),
        ));
        g.addDependency(input_norm, prev_residual);

        const q_proj = try addAnnotatedNode(&g, layer, .dmmv, "q_proj", dmmvMetrics(
            q_dim,
            config.hidden_dim,
            layerTensorBytes(gf, layer, "attn_q.weight", approxF16TensorBytes(q_dim, config.hidden_dim)),
            q_dim,
            1,
        ));
        g.addDependency(q_proj, input_norm);
        const k_proj = try addAnnotatedNode(&g, layer, .dmmv, "k_proj", dmmvMetrics(
            kv_dim,
            config.hidden_dim,
            layerTensorBytes(gf, layer, "attn_k.weight", approxF16TensorBytes(kv_dim, config.hidden_dim)),
            kv_dim,
            1,
        ));
        g.addDependency(k_proj, input_norm);
        const v_proj = try addAnnotatedNode(&g, layer, .dmmv, "v_proj", dmmvMetrics(
            kv_dim,
            config.hidden_dim,
            layerTensorBytes(gf, layer, "attn_v.weight", approxF16TensorBytes(kv_dim, config.hidden_dim)),
            kv_dim,
            1,
        ));
        g.addDependency(v_proj, input_norm);

        const rope_q = try addAnnotatedNode(&g, layer, .rope, "rope_q", ropeMetrics(config.head_dim, config.n_heads, rope_dim));
        g.addDependency(rope_q, q_proj);
        const rope_k = try addAnnotatedNode(&g, layer, .rope, "rope_k", ropeMetrics(config.head_dim, config.n_kv_heads, rope_dim));
        g.addDependency(rope_k, k_proj);

        const kv_write = try addAnnotatedNode(&g, layer, .kv_cache_write, "kv_write", kvWriteMetrics(kv_dim));
        g.addDependency(kv_write, rope_k);
        g.addDependency(kv_write, v_proj);

        const attn = try addAnnotatedNode(&g, layer, .flash_attn, "flash_attn", flashAttnMetrics(
            config.head_dim,
            config.n_heads,
            config.n_kv_heads,
            seq_len,
        ));
        g.addDependency(attn, rope_q);
        g.addDependency(attn, kv_write);

        const o_proj = try addAnnotatedNode(&g, layer, .dmmv, "o_proj", dmmvMetrics(
            config.hidden_dim,
            q_dim,
            layerTensorBytes(gf, layer, "attn_output.weight", approxF16TensorBytes(config.hidden_dim, q_dim)),
            config.hidden_dim,
            1,
        ));
        g.addDependency(o_proj, attn);

        const attn_residual = try addAnnotatedNode(&g, layer, .add, "attn_residual", addMetrics(config.hidden_dim));
        g.addDependency(attn_residual, prev_residual);
        g.addDependency(attn_residual, o_proj);

        // MoE FFN block
        const ffn_norm = try addAnnotatedNode(&g, layer, .rms_norm_mul, "ffn_norm", rmsNormMetrics(
            config.hidden_dim,
            1,
            layerTensorBytes(gf, layer, "post_attention_norm.weight", config.hidden_dim * @sizeOf(f32)),
        ));
        g.addDependency(ffn_norm, attn_residual);

        // Expert routing: softmax + top-k
        const moe_gate = try addAnnotatedNode(&g, layer, .moe_gate, "moe_gate", .{
            .read_bytes = vecBytes(config.hidden_dim),
            .write_bytes = vecBytes(config.n_experts),
            .weight_bytes = layerTensorBytes(gf, layer, "ffn_gate_inp.weight", approxF16TensorBytes(config.n_experts, config.hidden_dim)),
            .flops = 2 * @as(u64, config.n_experts) * @as(u64, config.hidden_dim) + @as(u64, config.n_experts) * 8,
            .workgroups = .{ @max(@as(u32, 1), divCeilU32(config.n_experts, 64)), 1, 1 },
        });
        g.addDependency(moe_gate, ffn_norm);

        // Sparse expert execution (gate+up → SwiGLU → down, only for top-k experts)
        const gate_proj = try addAnnotatedNode(&g, layer, .dmmv, "expert_gate", dmmvMetrics(
            inter_dim * n_used,
            config.hidden_dim,
            layerTensorBytes(gf, layer, "ffn_gate_exps.weight", approxF16TensorBytes(inter_dim * config.n_experts, config.hidden_dim)) / @max(@as(u64, 1), @as(u64, config.n_experts)) * n_used,
            inter_dim,
            n_used,
        ));
        g.addDependency(gate_proj, ffn_norm);
        g.addDependency(gate_proj, moe_gate);

        const up_proj = try addAnnotatedNode(&g, layer, .dmmv, "expert_up", dmmvMetrics(
            inter_dim * n_used,
            config.hidden_dim,
            layerTensorBytes(gf, layer, "ffn_up_exps.weight", approxF16TensorBytes(inter_dim * config.n_experts, config.hidden_dim)) / @max(@as(u64, 1), @as(u64, config.n_experts)) * n_used,
            inter_dim,
            n_used,
        ));
        g.addDependency(up_proj, ffn_norm);
        g.addDependency(up_proj, moe_gate);

        const swiglu = try addAnnotatedNode(&g, layer, .swiglu, "expert_swiglu", swigluMetrics(inter_dim * n_used));
        g.addDependency(swiglu, gate_proj);
        g.addDependency(swiglu, up_proj);

        const down_proj = try addAnnotatedNode(&g, layer, .dmmv, "expert_down", dmmvMetrics(
            config.hidden_dim * n_used,
            inter_dim,
            layerTensorBytes(gf, layer, "ffn_down_exps.weight", approxF16TensorBytes(config.hidden_dim * config.n_experts, inter_dim)) / @max(@as(u64, 1), @as(u64, config.n_experts)) * n_used,
            config.hidden_dim,
            n_used,
        ));
        g.addDependency(down_proj, swiglu);

        // Gather expert outputs
        const moe_gather = try addAnnotatedNode(&g, layer, .moe_gather, "moe_gather", moeGatherMetrics(config.hidden_dim, n_used));
        g.addDependency(moe_gather, down_proj);
        g.addDependency(moe_gather, moe_gate);

        const ffn_residual = try addAnnotatedNode(&g, layer, .add, "ffn_residual", addMetrics(config.hidden_dim));
        g.addDependency(ffn_residual, attn_residual);
        g.addDependency(ffn_residual, moe_gather);

        prev_residual = ffn_residual;
    }

    const final_norm = try addAnnotatedNode(&g, null, .rms_norm_mul, "final_norm", rmsNormMetrics(
        config.hidden_dim,
        1,
        globalTensorBytes(gf, "output_norm.weight", config.hidden_dim * @sizeOf(f32)),
    ));
    g.addDependency(final_norm, prev_residual);
    const lm_head = try addAnnotatedNode(&g, null, .dmmv, "lm_head", dmmvMetrics(
        config.vocab_size,
        config.hidden_dim,
        blk: {
            const out_bytes = globalTensorBytes(gf, "output.weight", 0);
            if (out_bytes > 0) break :blk out_bytes;
            break :blk globalTensorBytes(gf, "token_embd.weight", approxF16TensorBytes(config.vocab_size, config.hidden_dim));
        },
        config.vocab_size,
        1,
    ));
    g.addDependency(lm_head, final_norm);

    log.info("Built MoE decode graph: {d} nodes, {d} layers, {d} experts (top-{d})", .{
        g.nodeCount(), config.n_layers, config.n_experts, config.n_experts_used,
    });

    return g;
}

/// Mamba/Jamba hybrid decode graph.
/// Interleaves SSM layers (conv + gated delta net + sigmoid_mul) with attention layers.
fn buildMambaDecodeGraph(config: *const ModelConfig, allocator: std.mem.Allocator, gf: ?*const gguf.GGUFFile) !Graph {
    var g = Graph.init(allocator, "mamba_decode");
    errdefer g.deinit();

    const q_dim = config.n_heads * config.head_dim;
    const kv_dim = config.n_kv_heads * config.head_dim;
    const inter_dim = if (config.intermediate_dim > 0) config.intermediate_dim else config.hidden_dim * 4;
    const rope_dim = if (config.rope_dim > 0) config.rope_dim else config.head_dim;
    const seq_len = @min(config.context_length, 4096);
    const full_attn_interval = if (config.full_attn_interval > 0) config.full_attn_interval else 6;
    const conv_channels = if (config.ssm_d_inner > 0) config.ssm_d_inner + 2 * config.ssm_n_group * config.ssm_d_state else config.hidden_dim;
    g.setAssumedDecodeSeqLen(seq_len);

    const embed = try addAnnotatedNode(&g, null, .embed, "token_embed", .{
        .domain = .cpu_host,
        .read_bytes = globalTensorBytes(gf, "token_embd.weight", approxF16TensorBytes(config.vocab_size, config.hidden_dim)) / @max(@as(u64, 1), @as(u64, config.vocab_size)),
        .write_bytes = vecBytes(config.hidden_dim),
        .requires_host_sync = true,
        .note = "Embedding dequant/upload is host-driven in the current runtime.",
    });
    var prev_residual = embed;

    for (0..config.n_layers) |layer_idx| {
        const layer: u32 = @intCast(layer_idx);
        const input_norm = try addAnnotatedNode(&g, layer, .rms_norm_mul, "input_norm", rmsNormMetrics(
            config.hidden_dim,
            1,
            layerTensorBytes(gf, layer, "attn_norm.weight", config.hidden_dim * @sizeOf(f32)),
        ));
        g.addDependency(input_norm, prev_residual);

        // Decide if this is an attention or SSM layer
        const is_attention_layer = ((layer_idx + 1) % full_attn_interval == 0);

        if (is_attention_layer) {
            // Standard attention block
            const q_proj = try addAnnotatedNode(&g, layer, .dmmv, "q_proj", dmmvMetrics(
                q_dim,
                config.hidden_dim,
                layerTensorBytes(gf, layer, "attn_q.weight", approxF16TensorBytes(q_dim, config.hidden_dim)),
                q_dim,
                1,
            ));
            g.addDependency(q_proj, input_norm);
            const k_proj = try addAnnotatedNode(&g, layer, .dmmv, "k_proj", dmmvMetrics(
                kv_dim,
                config.hidden_dim,
                layerTensorBytes(gf, layer, "attn_k.weight", approxF16TensorBytes(kv_dim, config.hidden_dim)),
                kv_dim,
                1,
            ));
            g.addDependency(k_proj, input_norm);
            const v_proj = try addAnnotatedNode(&g, layer, .dmmv, "v_proj", dmmvMetrics(
                kv_dim,
                config.hidden_dim,
                layerTensorBytes(gf, layer, "attn_v.weight", approxF16TensorBytes(kv_dim, config.hidden_dim)),
                kv_dim,
                1,
            ));
            g.addDependency(v_proj, input_norm);

            const rope_q = try addAnnotatedNode(&g, layer, .rope, "rope_q", ropeMetrics(config.head_dim, config.n_heads, rope_dim));
            g.addDependency(rope_q, q_proj);
            const rope_k = try addAnnotatedNode(&g, layer, .rope, "rope_k", ropeMetrics(config.head_dim, config.n_kv_heads, rope_dim));
            g.addDependency(rope_k, k_proj);

            const kv_write = try addAnnotatedNode(&g, layer, .kv_cache_write, "kv_write", kvWriteMetrics(kv_dim));
            g.addDependency(kv_write, rope_k);
            g.addDependency(kv_write, v_proj);

            const attn = try addAnnotatedNode(&g, layer, .flash_attn, "flash_attn", flashAttnMetrics(
                config.head_dim,
                config.n_heads,
                config.n_kv_heads,
                seq_len,
            ));
            g.addDependency(attn, rope_q);
            g.addDependency(attn, kv_write);

            const o_proj = try addAnnotatedNode(&g, layer, .dmmv, "o_proj", dmmvMetrics(
                config.hidden_dim,
                q_dim,
                layerTensorBytes(gf, layer, "attn_output.weight", approxF16TensorBytes(config.hidden_dim, q_dim)),
                config.hidden_dim,
                1,
            ));
            g.addDependency(o_proj, attn);

            const residual = try addAnnotatedNode(&g, layer, .add, "attn_residual", addMetrics(config.hidden_dim));
            g.addDependency(residual, prev_residual);
            g.addDependency(residual, o_proj);
            prev_residual = residual;
        } else {
            // SSM block: in_proj → conv → sigmoid_mul gating → SSM → out_proj
            const in_proj = try addAnnotatedNode(&g, layer, .dmmv, "ssm_in_proj", dmmvMetrics(
                conv_channels,
                config.hidden_dim,
                layerTensorBytes(gf, layer, "attn_qkv.weight", approxF16TensorBytes(conv_channels, config.hidden_dim)),
                conv_channels,
                1,
            ));
            g.addDependency(in_proj, input_norm);

            const ssm_gate = try addAnnotatedNode(&g, layer, .sigmoid_mul, "ssm_gate", .{
                .read_bytes = vecBytes(conv_channels + config.ssm_d_inner + config.ssm_dt_rank * 2 + config.ssm_dt_rank * config.ssm_d_inner),
                .write_bytes = vecBytes(config.ssm_d_inner),
                .flops = @as(u64, config.ssm_d_inner) * (@as(u64, config.ssm_d_conv) + 12),
                .workgroups = .{ @max(@as(u32, 1), config.ssm_dt_rank), 1, 1 },
                .note = "Rolls conv/state update/gating into one logical hotspot for graph analysis.",
            });
            g.addDependency(ssm_gate, in_proj);

            const out_proj = try addAnnotatedNode(&g, layer, .dmmv, "ssm_out_proj", dmmvMetrics(
                config.hidden_dim,
                config.ssm_d_inner,
                layerTensorBytes(gf, layer, "ssm_out.weight", approxF16TensorBytes(config.hidden_dim, config.ssm_d_inner)),
                config.hidden_dim,
                1,
            ));
            g.addDependency(out_proj, ssm_gate);

            const residual = try addAnnotatedNode(&g, layer, .add, "ssm_residual", addMetrics(config.hidden_dim));
            g.addDependency(residual, prev_residual);
            g.addDependency(residual, out_proj);
            prev_residual = residual;
        }

        // FFN block (shared across layer types)
        const ffn_norm = try addAnnotatedNode(&g, layer, .rms_norm_mul, "ffn_norm", rmsNormMetrics(
            config.hidden_dim,
            1,
            layerTensorBytes(gf, layer, "post_attention_norm.weight", config.hidden_dim * @sizeOf(f32)),
        ));
        g.addDependency(ffn_norm, prev_residual);

        const gate_proj = try addAnnotatedNode(&g, layer, .dmmv, "gate_proj", dmmvMetrics(
            inter_dim,
            config.hidden_dim,
            layerTensorBytes(gf, layer, "ffn_gate.weight", approxF16TensorBytes(inter_dim, config.hidden_dim)),
            inter_dim,
            1,
        ));
        g.addDependency(gate_proj, ffn_norm);
        const up_proj = try addAnnotatedNode(&g, layer, .dmmv, "up_proj", dmmvMetrics(
            inter_dim,
            config.hidden_dim,
            layerTensorBytes(gf, layer, "ffn_up.weight", approxF16TensorBytes(inter_dim, config.hidden_dim)),
            inter_dim,
            1,
        ));
        g.addDependency(up_proj, ffn_norm);

        const swiglu = try addAnnotatedNode(&g, layer, .swiglu, "swiglu", swigluMetrics(inter_dim));
        g.addDependency(swiglu, gate_proj);
        g.addDependency(swiglu, up_proj);

        const down_proj = try addAnnotatedNode(&g, layer, .dmmv, "down_proj", dmmvMetrics(
            config.hidden_dim,
            inter_dim,
            layerTensorBytes(gf, layer, "ffn_down.weight", approxF16TensorBytes(config.hidden_dim, inter_dim)),
            config.hidden_dim,
            1,
        ));
        g.addDependency(down_proj, swiglu);

        const ffn_residual = try addAnnotatedNode(&g, layer, .add, "ffn_residual", addMetrics(config.hidden_dim));
        g.addDependency(ffn_residual, prev_residual);
        g.addDependency(ffn_residual, down_proj);

        prev_residual = ffn_residual;
    }

    const final_norm = try addAnnotatedNode(&g, null, .rms_norm_mul, "final_norm", rmsNormMetrics(
        config.hidden_dim,
        1,
        globalTensorBytes(gf, "output_norm.weight", config.hidden_dim * @sizeOf(f32)),
    ));
    g.addDependency(final_norm, prev_residual);
    const lm_head = try addAnnotatedNode(&g, null, .dmmv, "lm_head", dmmvMetrics(
        config.vocab_size,
        config.hidden_dim,
        blk: {
            const out_bytes = globalTensorBytes(gf, "output.weight", 0);
            if (out_bytes > 0) break :blk out_bytes;
            break :blk globalTensorBytes(gf, "token_embd.weight", approxF16TensorBytes(config.vocab_size, config.hidden_dim));
        },
        config.vocab_size,
        1,
    ));
    g.addDependency(lm_head, final_norm);

    log.info("Built Mamba/Jamba decode graph: {d} nodes, {d} layers", .{
        g.nodeCount(), config.n_layers,
    });

    return g;
}

test "buildDecodeGraph: standard transformer 2 layers" {
    const allocator = std.testing.allocator;
    const config = ModelConfig{
        .architecture = .qwen2,
        .n_layers = 2,
        .n_heads = 32,
        .n_kv_heads = 8,
        .head_dim = 128,
        .hidden_dim = 4096,
        .intermediate_dim = 14336,
        .vocab_size = 128256,
        .context_length = 8192,
        .rope_freq_base = 500000.0,
        .n_experts = 0,
        .n_experts_used = 0,
        .rope_dim = 0,
        .ssm_d_conv = 0,
        .ssm_d_inner = 0,
        .ssm_d_state = 0,
        .ssm_dt_rank = 0,
        .ssm_n_group = 0,
        .full_attn_interval = 0,
        .shared_expert_intermediate_dim = 0,
    };

    var g = try buildDecodeGraph(&config, allocator);
    defer g.deinit();

    // embed(1) + 2 layers * 16 nodes/layer + final_norm(1) + lm_head(1) = 35
    try std.testing.expectEqual(@as(usize, 35), g.nodeCount());

    // Verify topological sort works (no cycles)
    const order = try g.topologicalOrder(allocator);
    defer allocator.free(order);
    try std.testing.expectEqual(@as(usize, 35), order.len);
}

test "buildDecodeGraph: moe 1 layer" {
    const allocator = std.testing.allocator;
    const config = ModelConfig{
        .architecture = .qwen2_moe,
        .n_layers = 1,
        .n_heads = 16,
        .n_kv_heads = 4,
        .head_dim = 128,
        .hidden_dim = 2048,
        .intermediate_dim = 5632,
        .vocab_size = 151936,
        .context_length = 32768,
        .rope_freq_base = 10000.0,
        .n_experts = 60,
        .n_experts_used = 4,
        .rope_dim = 0,
        .ssm_d_conv = 0,
        .ssm_d_inner = 0,
        .ssm_d_state = 0,
        .ssm_dt_rank = 0,
        .ssm_n_group = 0,
        .full_attn_interval = 0,
        .shared_expert_intermediate_dim = 0,
    };

    var g = try buildDecodeGraph(&config, allocator);
    defer g.deinit();

    try std.testing.expect(g.nodeCount() > 0);

    const order = try g.topologicalOrder(allocator);
    defer allocator.free(order);
    try std.testing.expectEqual(g.nodeCount(), order.len);
}

test "buildDecodeGraph: qwen35 dense hybrid 1 layer" {
    const allocator = std.testing.allocator;
    const config = ModelConfig{
        .architecture = .qwen35,
        .n_layers = 1,
        .n_heads = 16,
        .n_kv_heads = 4,
        .head_dim = 256,
        .hidden_dim = 4096,
        .intermediate_dim = 12288,
        .vocab_size = 248320,
        .context_length = 262144,
        .rope_freq_base = 10000000.0,
        .n_experts = 0,
        .n_experts_used = 0,
        .rope_dim = 64,
        .ssm_d_conv = 4,
        .ssm_d_inner = 4096,
        .ssm_d_state = 128,
        .ssm_dt_rank = 32,
        .ssm_n_group = 16,
        .full_attn_interval = 4,
        .shared_expert_intermediate_dim = 0,
    };

    var g = try buildDecodeGraph(&config, allocator);
    defer g.deinit();

    try std.testing.expect(g.nodeCount() > 0);

    const order = try g.topologicalOrder(allocator);
    defer allocator.free(order);
    try std.testing.expectEqual(g.nodeCount(), order.len);
}
