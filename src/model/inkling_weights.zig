//! Tensor names and shapes shared by the Vulkan and ROCm Inkling decoders.
//! @section Model Format & Loading
//!
//! Both backends validate the same per-layer weights before running the loop.
const std = @import("std");
const Layout = @import("inkling.zig").Layout;
const log = std.log.scoped(.inkling_weights);

pub fn Tensors(comptime Model: type, comptime LoadedTensor: type) type {
    return struct {
        pub const Weight = *const LoadedTensor;

        pub const Controller = struct {
            down: Weight,
            gain: Weight,
            forget: Weight,
        };
        pub const Loop = struct { query: Weight, controls: [4]Controller };

        pub const Layer = struct {
            q: Weight,
            k: Weight,
            v: Weight,
            gate: Weight,
            output: Weight,
            relative: Weight,
            sink: Weight,
            q_clip: Weight,
            rel_clip: Weight,
            rel_bias: Weight,
            k_conv: Weight,
            v_conv: Weight,
            attn_post: Weight,
            ffn_post: Weight,
            k_conv_proj: ?Weight,
            v_conv_proj: ?Weight,
            input_norms: [3]Weight,
            attn_query: ?Weight,
            ffn_query: Weight,
            ffn_norm: Weight,
            router: Weight,
            router_bias: ?Weight,
            gate_up: Weight,
            down: Weight,
            loop: ?Loop,

            pub fn read(model: *const Model, d: Layout, layer: u32) !Layer {
                const h = d.hidden;
                const q = d.heads * d.head_dim;
                const kv = d.kv_heads * d.head_dim;
                const local = d.sliding[layer];
                const relative_extent = if (local) d.window else d.relative_extent;
                const norms = if (local) blk: {
                    const norm = try layerWeight(model, layer, "attn_norm.weight", &.{h}, .float);
                    break :blk [3]Weight{ norm, norm, norm };
                } else [3]Weight{
                    try layerWeight(model, layer, "attn_q_inp_norm.weight", &.{h}, .float),
                    try layerWeight(model, layer, "attn_k_inp_norm.weight", &.{h}, .float),
                    try layerWeight(model, layer, "attn_v_inp_norm.weight", &.{h}, .float),
                };
                var loop: ?Loop = null;
                if (layer >= d.core_start and layer < d.core_end) {
                    var controls: [4]Controller = undefined;
                    inline for (.{ "q", "k", "v", "ffn" }, 0..) |name, i| {
                        controls[i] = .{
                            .down = try layerWeight(model, layer, "loop_" ++ name ++ "_ctrl_down.weight", &.{ 2 * h, d.gate_rank }, .matrix),
                            .gain = try layerWeight(model, layer, "loop_" ++ name ++ "_ctrl_gain.weight", &.{ d.gate_rank, h }, .matrix),
                            .forget = try layerWeight(model, layer, "loop_" ++ name ++ "_ctrl_forget.weight", &.{ d.gate_rank, h }, .matrix),
                        };
                    }
                    loop = .{
                        .query = try layerWeight(model, layer, "loop_msg_query.weight", &.{ h, 4 }, .float),
                        .controls = controls,
                    };
                }
                return .{
                    .q = try layerWeight(model, layer, "attn_q.weight", &.{ h, q }, .matrix),
                    .k = try layerWeight(model, layer, "attn_k.weight", &.{ h, kv }, .matrix),
                    .v = try layerWeight(model, layer, "attn_v.weight", &.{ h, kv }, .matrix),
                    .gate = try layerWeight(model, layer, "attn_gate.weight", &.{ h, q }, .matrix),
                    .output = try layerWeight(model, layer, "attn_output.weight", &.{ q, h }, .matrix),
                    .relative = try layerWeight(model, layer, "attn_rel.weight", &.{ h, d.heads * d.relative_dim }, .matrix),
                    .sink = try layerWeight(model, layer, "attn_sink_proj.weight", &.{ h, d.heads }, .matrix),
                    .q_clip = try layerWeight(model, layer, "attn_q_clip.weight", &.{ d.head_dim, d.heads }, .float),
                    .rel_clip = try layerWeight(model, layer, "attn_rel_clip.weight", &.{ d.relative_dim, d.heads }, .float),
                    .rel_bias = try layerWeight(model, layer, "attn_rel_bias.weight", &.{ d.relative_dim, relative_extent }, .float),
                    .k_conv = try layerWeight(model, layer, "attn_k_conv.weight", &.{ d.taps, kv }, .float),
                    .v_conv = try layerWeight(model, layer, "attn_v_conv.weight", &.{ d.taps, kv }, .float),
                    .attn_post = try layerWeight(model, layer, "attn_post_conv.weight", &.{ d.taps, h }, .float),
                    .ffn_post = try layerWeight(model, layer, "ffn_post_conv.weight", &.{ d.taps, h }, .float),
                    .k_conv_proj = if (local) null else try layerWeight(model, layer, "attn_k_conv_proj.weight", &.{ h, d.taps * d.kv_heads }, .matrix),
                    .v_conv_proj = if (local) null else try layerWeight(model, layer, "attn_v_conv_proj.weight", &.{ h, d.taps * d.kv_heads }, .matrix),
                    .input_norms = norms,
                    .attn_query = if (layer == 0) null else try layerWeight(model, layer, "attn_res_query.weight", &.{ h, if (local) 1 else 3 }, .float),
                    .ffn_query = try layerWeight(model, layer, "ffn_res_query.weight", &.{h}, .float),
                    .ffn_norm = try layerWeight(model, layer, "ffn_norm.weight", &.{h}, .float),
                    .router = try layerWeight(model, layer, "ffn_gate_inp.weight", &.{ h, d.experts }, .matrix),
                    .router_bias = if (model.getLayer(layer, "exp_probs_b.bias")) |b| try checkWeight(b, &.{d.experts}, .float) else null,
                    .gate_up = try layerWeight(model, layer, "ffn_gate_up_exps.weight", &.{ h, 2 * d.expert_width, d.experts }, .matrix),
                    .down = try layerWeight(model, layer, "ffn_down_exps.weight", &.{ d.expert_width, h, d.experts }, .matrix),
                    .loop = loop,
                };
            }
        };

        const Kind = enum { matrix, float };

        fn checkWeight(weight: Weight, shape: []const u32, kind: Kind) !Weight {
            for (weight.info.dims, 0..) |dim, i| {
                if (dim != (if (i < shape.len) shape[i] else 1)) {
                    log.err("Invalid Inkling tensor shape: {s}", .{weight.info.name});
                    return error.InvalidInklingTensor;
                }
            }
            const supported = switch (kind) {
                .float => weight.info.type_ == .f32,
                .matrix => switch (weight.info.type_) {
                    .f32, .f16, .bf16, .q4_0, .q8_0, .q4_k, .q5_k, .q6_k => shape[0] % weight.info.type_.blockSize() == 0,
                    else => false,
                },
            };
            if (!supported) {
                log.err("Unsupported Inkling tensor type: {s} ({s})", .{ weight.info.name, @tagName(weight.info.type_) });
                return error.UnsupportedInklingQuantization;
            }
            return weight;
        }

        fn layerWeight(model: *const Model, layer: u32, suffix: []const u8, shape: []const u32, kind: Kind) !Weight {
            const weight = model.getLayer(layer, suffix) orelse {
                log.err("Missing Inkling tensor: blk.{d}.{s}", .{ layer, suffix });
                return error.MissingInklingTensor;
            };
            return checkWeight(weight, shape, kind);
        }

        pub fn modelWeight(model: *const Model, name: []const u8, shape: []const u32, kind: Kind) !Weight {
            const weight = model.get(name) orelse {
                log.err("Missing Inkling tensor: {s}", .{name});
                return error.MissingInklingTensor;
            };
            return checkWeight(weight, shape, kind);
        }
    };
}
