const std = @import("std");
const graph_mod = @import("../compute/graph.zig");
const Graph = graph_mod.Graph;
const OpType = graph_mod.OpType;
const loader = @import("loader.zig");
const ModelConfig = loader.ModelConfig;

const log = std.log.scoped(.architecture);

/// Build a compute graph for a single transformer decode step.
/// This creates the graph structure — actual buffer bindings are set at runtime.
pub fn buildDecodeGraph(config: *const ModelConfig, allocator: std.mem.Allocator) !Graph {
    return switch (config.architecture) {
        .llama, .mistral, .qwen2 => try buildLlamaDecodeGraph(config, allocator),
        .qwen2_moe => try buildMoeDecodeGraph(config, allocator),
        .mamba, .jamba => try buildMambaDecodeGraph(config, allocator),
        .unknown => error.UnsupportedArchitecture,
    };
}

/// Standard transformer decode graph (LLaMA/Mistral/Qwen2).
/// Per-layer structure:
///   input_norm → QKV projection → RoPE → flash attention → residual add →
///   ffn_norm → gate+up projection → SwiGLU → down projection → residual add
fn buildLlamaDecodeGraph(config: *const ModelConfig, allocator: std.mem.Allocator) !Graph {
    var g = Graph.init(allocator, "llama_decode");
    errdefer g.deinit();

    // Token embedding lookup
    const embed = try g.addNode(.embed, "token_embed");

    var prev_residual = embed;

    for (0..config.n_layers) |layer_idx| {
        _ = layer_idx;

        // Input normalization
        const input_norm = try g.addNode(.rms_norm_mul, "input_norm");
        g.addDependency(input_norm, prev_residual);

        // QKV projection (decode = matmul-vec)
        const q_proj = try g.addNode(.dmmv, "q_proj");
        g.addDependency(q_proj, input_norm);

        const k_proj = try g.addNode(.dmmv, "k_proj");
        g.addDependency(k_proj, input_norm);

        const v_proj = try g.addNode(.dmmv, "v_proj");
        g.addDependency(v_proj, input_norm);

        // RoPE on Q and K
        const rope_q = try g.addNode(.rope, "rope_q");
        g.addDependency(rope_q, q_proj);

        const rope_k = try g.addNode(.rope, "rope_k");
        g.addDependency(rope_k, k_proj);

        // Write K/V to cache
        const kv_write = try g.addNode(.kv_cache_write, "kv_write");
        g.addDependency(kv_write, rope_k);
        g.addDependency(kv_write, v_proj);

        // Flash attention
        const attn = try g.addNode(.flash_attn, "flash_attn");
        g.addDependency(attn, rope_q);
        g.addDependency(attn, kv_write);

        // Attention output projection
        const o_proj = try g.addNode(.dmmv, "o_proj");
        g.addDependency(o_proj, attn);

        // Residual add
        const attn_residual = try g.addNode(.add, "attn_residual");
        g.addDependency(attn_residual, prev_residual);
        g.addDependency(attn_residual, o_proj);

        // FFN normalization
        const ffn_norm = try g.addNode(.rms_norm_mul, "ffn_norm");
        g.addDependency(ffn_norm, attn_residual);

        // Gate + Up projection (SwiGLU FFN)
        const gate_proj = try g.addNode(.dmmv, "gate_proj");
        g.addDependency(gate_proj, ffn_norm);

        const up_proj = try g.addNode(.dmmv, "up_proj");
        g.addDependency(up_proj, ffn_norm);

        // SwiGLU activation
        const swiglu = try g.addNode(.swiglu, "swiglu");
        g.addDependency(swiglu, gate_proj);
        g.addDependency(swiglu, up_proj);

        // Down projection
        const down_proj = try g.addNode(.dmmv, "down_proj");
        g.addDependency(down_proj, swiglu);

        // FFN residual add
        const ffn_residual = try g.addNode(.add, "ffn_residual");
        g.addDependency(ffn_residual, attn_residual);
        g.addDependency(ffn_residual, down_proj);

        prev_residual = ffn_residual;
    }

    // Final normalization
    const final_norm = try g.addNode(.rms_norm_mul, "final_norm");
    g.addDependency(final_norm, prev_residual);

    // Output projection (lm_head)
    const lm_head = try g.addNode(.dmmv, "lm_head");
    g.addDependency(lm_head, final_norm);

    log.info("Built LLaMA decode graph: {d} nodes, {d} layers", .{
        g.nodeCount(), config.n_layers,
    });

    return g;
}

/// MoE transformer decode graph (Qwen2-MoE).
/// Same as LLaMA but FFN is replaced with expert routing + sparse expert matmuls.
fn buildMoeDecodeGraph(config: *const ModelConfig, allocator: std.mem.Allocator) !Graph {
    var g = Graph.init(allocator, "moe_decode");
    errdefer g.deinit();

    const embed = try g.addNode(.embed, "token_embed");
    var prev_residual = embed;

    for (0..config.n_layers) |_| {
        // Attention block (same as LLaMA)
        const input_norm = try g.addNode(.rms_norm_mul, "input_norm");
        g.addDependency(input_norm, prev_residual);

        const q_proj = try g.addNode(.dmmv, "q_proj");
        g.addDependency(q_proj, input_norm);
        const k_proj = try g.addNode(.dmmv, "k_proj");
        g.addDependency(k_proj, input_norm);
        const v_proj = try g.addNode(.dmmv, "v_proj");
        g.addDependency(v_proj, input_norm);

        const rope_q = try g.addNode(.rope, "rope_q");
        g.addDependency(rope_q, q_proj);
        const rope_k = try g.addNode(.rope, "rope_k");
        g.addDependency(rope_k, k_proj);

        const kv_write = try g.addNode(.kv_cache_write, "kv_write");
        g.addDependency(kv_write, rope_k);
        g.addDependency(kv_write, v_proj);

        const attn = try g.addNode(.flash_attn, "flash_attn");
        g.addDependency(attn, rope_q);
        g.addDependency(attn, kv_write);

        const o_proj = try g.addNode(.dmmv, "o_proj");
        g.addDependency(o_proj, attn);

        const attn_residual = try g.addNode(.add, "attn_residual");
        g.addDependency(attn_residual, prev_residual);
        g.addDependency(attn_residual, o_proj);

        // MoE FFN block
        const ffn_norm = try g.addNode(.rms_norm_mul, "ffn_norm");
        g.addDependency(ffn_norm, attn_residual);

        // Expert routing: softmax + top-k
        const moe_gate = try g.addNode(.moe_gate, "moe_gate");
        g.addDependency(moe_gate, ffn_norm);

        // Sparse expert execution (gate+up → SwiGLU → down, only for top-k experts)
        const gate_proj = try g.addNode(.dmmv, "expert_gate");
        g.addDependency(gate_proj, ffn_norm);
        g.addDependency(gate_proj, moe_gate);

        const up_proj = try g.addNode(.dmmv, "expert_up");
        g.addDependency(up_proj, ffn_norm);
        g.addDependency(up_proj, moe_gate);

        const swiglu = try g.addNode(.swiglu, "expert_swiglu");
        g.addDependency(swiglu, gate_proj);
        g.addDependency(swiglu, up_proj);

        const down_proj = try g.addNode(.dmmv, "expert_down");
        g.addDependency(down_proj, swiglu);

        // Gather expert outputs
        const moe_gather = try g.addNode(.moe_gather, "moe_gather");
        g.addDependency(moe_gather, down_proj);
        g.addDependency(moe_gather, moe_gate);

        const ffn_residual = try g.addNode(.add, "ffn_residual");
        g.addDependency(ffn_residual, attn_residual);
        g.addDependency(ffn_residual, moe_gather);

        prev_residual = ffn_residual;
    }

    const final_norm = try g.addNode(.rms_norm_mul, "final_norm");
    g.addDependency(final_norm, prev_residual);
    const lm_head = try g.addNode(.dmmv, "lm_head");
    g.addDependency(lm_head, final_norm);

    log.info("Built MoE decode graph: {d} nodes, {d} layers, {d} experts (top-{d})", .{
        g.nodeCount(), config.n_layers, config.n_experts, config.n_experts_used,
    });

    return g;
}

/// Mamba/Jamba hybrid decode graph.
/// Interleaves SSM layers (conv + gated delta net + sigmoid_mul) with attention layers.
fn buildMambaDecodeGraph(config: *const ModelConfig, allocator: std.mem.Allocator) !Graph {
    var g = Graph.init(allocator, "mamba_decode");
    errdefer g.deinit();

    const embed = try g.addNode(.embed, "token_embed");
    var prev_residual = embed;

    for (0..config.n_layers) |layer_idx| {
        const input_norm = try g.addNode(.rms_norm_mul, "input_norm");
        g.addDependency(input_norm, prev_residual);

        // Decide if this is an attention or SSM layer
        // Jamba pattern: every 6th layer is attention, rest are SSM
        const is_attention_layer = (layer_idx % 6 == 0);

        if (is_attention_layer) {
            // Standard attention block
            const q_proj = try g.addNode(.dmmv, "q_proj");
            g.addDependency(q_proj, input_norm);
            const k_proj = try g.addNode(.dmmv, "k_proj");
            g.addDependency(k_proj, input_norm);
            const v_proj = try g.addNode(.dmmv, "v_proj");
            g.addDependency(v_proj, input_norm);

            const rope_q = try g.addNode(.rope, "rope_q");
            g.addDependency(rope_q, q_proj);
            const rope_k = try g.addNode(.rope, "rope_k");
            g.addDependency(rope_k, k_proj);

            const kv_write = try g.addNode(.kv_cache_write, "kv_write");
            g.addDependency(kv_write, rope_k);
            g.addDependency(kv_write, v_proj);

            const attn = try g.addNode(.flash_attn, "flash_attn");
            g.addDependency(attn, rope_q);
            g.addDependency(attn, kv_write);

            const o_proj = try g.addNode(.dmmv, "o_proj");
            g.addDependency(o_proj, attn);

            const residual = try g.addNode(.add, "attn_residual");
            g.addDependency(residual, prev_residual);
            g.addDependency(residual, o_proj);
            prev_residual = residual;
        } else {
            // SSM block: in_proj → conv → sigmoid_mul gating → SSM → out_proj
            const in_proj = try g.addNode(.dmmv, "ssm_in_proj");
            g.addDependency(in_proj, input_norm);

            const ssm_gate = try g.addNode(.sigmoid_mul, "ssm_gate");
            g.addDependency(ssm_gate, in_proj);

            const out_proj = try g.addNode(.dmmv, "ssm_out_proj");
            g.addDependency(out_proj, ssm_gate);

            const residual = try g.addNode(.add, "ssm_residual");
            g.addDependency(residual, prev_residual);
            g.addDependency(residual, out_proj);
            prev_residual = residual;
        }

        // FFN block (shared across layer types)
        const ffn_norm = try g.addNode(.rms_norm_mul, "ffn_norm");
        g.addDependency(ffn_norm, prev_residual);

        const gate_proj = try g.addNode(.dmmv, "gate_proj");
        g.addDependency(gate_proj, ffn_norm);
        const up_proj = try g.addNode(.dmmv, "up_proj");
        g.addDependency(up_proj, ffn_norm);

        const swiglu = try g.addNode(.swiglu, "swiglu");
        g.addDependency(swiglu, gate_proj);
        g.addDependency(swiglu, up_proj);

        const down_proj = try g.addNode(.dmmv, "down_proj");
        g.addDependency(down_proj, swiglu);

        const ffn_residual = try g.addNode(.add, "ffn_residual");
        g.addDependency(ffn_residual, prev_residual);
        g.addDependency(ffn_residual, down_proj);

        prev_residual = ffn_residual;
    }

    const final_norm = try g.addNode(.rms_norm_mul, "final_norm");
    g.addDependency(final_norm, prev_residual);
    const lm_head = try g.addNode(.dmmv, "lm_head");
    g.addDependency(lm_head, final_norm);

    log.info("Built Mamba/Jamba decode graph: {d} nodes, {d} layers", .{
        g.nodeCount(), config.n_layers,
    });

    return g;
}

test "buildDecodeGraph: llama 2 layers" {
    const allocator = std.testing.allocator;
    const config = ModelConfig{
        .architecture = .llama,
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
    };

    var g = try buildDecodeGraph(&config, allocator);
    defer g.deinit();

    try std.testing.expect(g.nodeCount() > 0);

    const order = try g.topologicalOrder(allocator);
    defer allocator.free(order);
    try std.testing.expectEqual(g.nodeCount(), order.len);
}
