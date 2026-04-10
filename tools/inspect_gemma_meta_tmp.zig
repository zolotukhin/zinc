const std = @import("std");
const gguf = @import("../src/model/gguf.zig");

fn printTensor(gf: *const gguf.GGUFFile, name: []const u8) void {
    if (gf.findTensor(name)) |t| {
        std.debug.print("tensor {s}: type={s} dims=({d},{d},{d},{d}) elems={d} bytes={d}\n", .{
            name,
            @tagName(t.type_),
            t.dims[0],
            t.dims[1],
            t.dims[2],
            t.dims[3],
            t.numElements(),
            t.sizeBytes(),
        });
    } else {
        std.debug.print("tensor {s}: MISSING\n", .{name});
    }
}

fn printKey(gf: *const gguf.GGUFFile, key: []const u8) void {
    if (gf.metadata.get(key)) |v| {
        std.debug.print("meta {s}=", .{key});
        switch (v) {
            .uint32 => |x| std.debug.print("u32:{d}\n", .{x}),
            .uint64 => |x| std.debug.print("u64:{d}\n", .{x}),
            .float32 => |x| std.debug.print("f32:{d}\n", .{x}),
            .bool_ => |x| std.debug.print("bool:{}\n", .{x}),
            .string => |x| std.debug.print("string:{s}\n", .{x}),
            .array => |arr| {
                std.debug.print("array[len={d}]", .{arr.len});
                for (arr, 0..) |item, i| {
                    if (i == 0) std.debug.print(" [", .{}) else std.debug.print(", ", .{});
                    switch (item) {
                        .string => |s| std.debug.print("{s}", .{s}),
                        .uint32 => |x| std.debug.print("{d}", .{x}),
                        .float32 => |x| std.debug.print("{d}", .{x}),
                        .bool_ => |x| std.debug.print("{}", .{x}),
                        else => std.debug.print("?", .{}),
                    }
                }
                if (arr.len > 0) std.debug.print("]", .{});
                std.debug.print("\n", .{});
            },
            else => std.debug.print("other\n", .{}),
        }
    } else {
        std.debug.print("meta {s}=MISSING\n", .{key});
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
        "gemma4.block_count",
        "gemma4.attention.head_count",
        "gemma4.attention.head_count_kv",
        "gemma4.attention.key_length",
        "gemma4.attention.sliding_window",
        "gemma4.attention.scale",
        "gemma4.rope.dimension_count",
        "gemma4.full_attention_interval",
        "gemma4.global_head_dim",
        "gemma4.num_kv_shared_layers",
        "gemma4.attention_k_eq_v",
        "gemma4.hidden_size_per_layer_input",
        "gemma4.vocab_size_per_layer_input",
        "gemma4.use_double_wide_mlp",
    };
    for (keys) |key| printKey(&gf, key);

    const tensors = [_][]const u8{
        "token_embd.weight",
        "token_embd_per_layer.weight",
        "blk.0.attn_q.weight",
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.5.attn_q.weight",
        "blk.5.attn_k.weight",
        "blk.5.attn_v.weight",
        "blk.5.attn_output.weight",
        "blk.0.ffn_gate.weight",
        "blk.0.ffn_up.weight",
        "blk.0.ffn_down.weight",
        "blk.0.ffn_gate_inp.weight",
        "blk.0.ffn_gate_up_exps.weight",
        "blk.0.ffn_down_exps.weight",
        "blk.0.post_ffw_norm_1.weight",
        "blk.0.post_ffw_norm_2.weight",
        "blk.0.pre_ffw_norm_2.weight",
        "blk.0.layer_output_scale.weight",
        "blk.0.per_layer_input_gate.weight",
        "blk.0.per_layer_projection.weight",
        "blk.0.post_per_layer_input_norm.weight",
        "blk.5.layer_output_scale.weight",
    };
    for (tensors) |name| printTensor(&gf, name);
}
