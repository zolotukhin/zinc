const std = @import("std");
const gguf = @import("src/model/gguf.zig");

fn printValue(key: []const u8, v: anytype) void {
    std.debug.print("{s}=", .{key});
    switch (v) {
        .string => |s| std.debug.print("string:{s}\n", .{s}),
        .bool_ => |b| std.debug.print("bool:{}\n", .{b}),
        .uint8 => |u| std.debug.print("u8:{}\n", .{u}),
        .uint16 => |u| std.debug.print("u16:{}\n", .{u}),
        .uint32 => |u| std.debug.print("u32:{}\n", .{u}),
        .uint64 => |u| std.debug.print("u64:{}\n", .{u}),
        .int8 => |i| std.debug.print("i8:{}\n", .{i}),
        .int16 => |i| std.debug.print("i16:{}\n", .{i}),
        .int32 => |i| std.debug.print("i32:{}\n", .{i}),
        .int64 => |i| std.debug.print("i64:{}\n", .{i}),
        .float32 => |f| std.debug.print("f32:{d}\n", .{f}),
        .float64 => |f| std.debug.print("f64:{d}\n", .{f}),
        .array => |arr| {
            std.debug.print("array[len={}]", .{arr.len});
            for (arr, 0..) |item, i| {
                if (i == 0) std.debug.print(" [", .{}) else std.debug.print(", ", .{});
                if (item.asU32()) |u| {
                    std.debug.print("{}", .{u});
                } else if (item.asF32()) |f| {
                    std.debug.print("{d}", .{f});
                } else if (item.asBool()) |b| {
                    std.debug.print("{}", .{b});
                } else {
                    std.debug.print("?", .{});
                }
            }
            if (arr.len > 0) std.debug.print("]", .{});
            std.debug.print("\n", .{});
        },
    }
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var args = try std.process.argsWithAllocator(allocator);
    defer args.deinit();
    _ = args.skip();
    const path = args.next() orelse return error.MissingArg;

    const file = try std.fs.cwd().openFile(path, .{});
    defer {
        var f = file;
        f.close();
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

    const keys = [_][]const u8{
        "general.architecture",
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.add_bos_token",
        "tokenizer.ggml.add_eos_token",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "qwen35.block_count",
        "qwen35.full_attention_interval",
        "qwen35.attention.head_count",
        "qwen35.attention.head_count_kv",
        "qwen35.attention.key_length",
        "qwen35.attention.sliding_window",
        "qwen35.attention.scale",
        "qwen35.attention.layer_norm_rms_epsilon",
        "qwen35.rope.freq_base",
        "qwen35.rope.freq_base_swa",
        "qwen35.rope.dimension_count",
        "qwen35.rope.scaling.factor",
        "qwen35.rope.scaling.original_context_length",
        "qwen35.final_logit_softcapping",
        "qwen35.ssm.conv_kernel",
        "qwen35.ssm.inner_size",
        "qwen35.ssm.state_size",
        "qwen35.ssm.time_step_rank",
        "qwen35.ssm.group_count",
        "qwen3.block_count",
        "qwen3.attention.head_count",
        "qwen3.attention.head_count_kv",
        "qwen3.attention.key_length",
        "qwen3.attention.sliding_window",
        "qwen3.attention.scale",
        "qwen3.attention.layer_norm_rms_epsilon",
        "qwen3.rope.freq_base",
        "qwen3.rope.dimension_count",
        "qwen3.rope.scaling.factor",
        "qwen3.rope.scaling.original_context_length",
        "qwen3.final_logit_softcapping",
    };
    for (keys) |key| {
        if (gf.metadata.get(key)) |v| printValue(key, v);
    }

    const tensors = [_][]const u8{
        "token_embd.weight",
        "output.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_gate.weight",
        "blk.0.attn_q_norm.weight",
        "blk.0.attn_k_norm.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.post_attention_norm.weight",
        "blk.0.ffn_norm.weight",
        "blk.0.attn_qkv.weight",
        "blk.0.ssm_conv1d.weight",
    };
    for (tensors) |name| {
        if (gf.findTensor(name)) |t| {
            std.debug.print(
                "tensor {s}: type={s} dims=({},{},{},{}) elems={} bytes={}\n",
                .{
                    name,
                    @tagName(t.type_),
                    t.dims[0],
                    t.dims[1],
                    t.dims[2],
                    t.dims[3],
                    t.numElements(),
                    t.sizeBytes(),
                },
            );
        }
    }

    if (gf.metadata.get("general.name")) |v| printValue("general.name", v);
    if (gf.metadata.get("general.basename")) |v| printValue("general.basename", v);
    if (gf.metadata.get("tokenizer.ggml.padding_token_id")) |v| printValue("tokenizer.ggml.padding_token_id", v);

    if (gf.metadata.get("tokenizer.ggml.tokens")) |tokens_val| {
        const tokens_array = switch (tokens_val) {
            .array => |a| a,
            else => return,
        };

        var seen = std.StringHashMap(u32).init(allocator);
        defer seen.deinit();
        var dup_count: usize = 0;
        for (tokens_array, 0..) |tok_val, i| {
            const tok_str = switch (tok_val) {
                .string => |s| s,
                else => "",
            };
            if (seen.get(tok_str)) |_| {
                dup_count += 1;
            } else {
                try seen.put(tok_str, @intCast(i));
            }
        }
        std.debug.print("tokenizer.duplicate_count={}\n", .{dup_count});

        const interesting_ids = [_]u32{ 785, 6722, 315, 9856, 374, 74593, 87297, 115982, 151643, 151644, 151645, 151667, 151668 };
        for (interesting_ids) |id| {
            if (id < tokens_array.len) {
                const tok_str = switch (tokens_array[id]) {
                    .string => |s| s,
                    else => "",
                };
                std.debug.print("token[{d}]={s}\n", .{ id, tok_str });
            }
        }
    }
}
