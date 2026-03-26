//! CLI entrypoints for configuring ZINC and starting local inference.
//! @section CLI & Entrypoints
//! This module wires together Vulkan initialization, model loading, tokenization,
//! and the single-process decode loop used for prompt-mode execution.
const std = @import("std");
const instance_mod = @import("vulkan/instance.zig");
const gpu_detect = @import("vulkan/gpu_detect.zig");
const loader_mod = @import("model/loader.zig");
const tokenizer_mod = @import("model/tokenizer.zig");
const forward_mod = @import("compute/forward.zig");
const CommandPool = @import("vulkan/command.zig").CommandPool;

const log = std.log.scoped(.zinc);

// Force compilation and testing of all modules
comptime {
    _ = @import("vulkan/vk.zig");
    _ = @import("vulkan/buffer.zig");
    _ = @import("vulkan/pipeline.zig");
    _ = @import("vulkan/command.zig");
    _ = @import("model/gguf.zig");
    _ = @import("model/loader.zig");
    _ = @import("model/tokenizer.zig");
    _ = @import("model/architecture.zig");
    _ = @import("compute/graph.zig");
    _ = @import("compute/dmmv.zig");
    _ = @import("compute/elementwise.zig");
    _ = @import("compute/attention.zig");
    _ = @import("compute/forward.zig");
}

/// Runtime configuration built from CLI flags and default values.
pub const Config = struct {
    model_path: ?[]const u8 = null,
    port: u16 = 8080,
    device_index: u32 = 0,
    context_length: u32 = 4096,
    max_parallel: u32 = 4,
    prompt: ?[]const u8 = null,
    kv_quant: u8 = 0, // 0=disabled, 2/3/4=TurboQuant bits
    show_help: bool = false,
};

const banner =
    \\ZINC — Zig INferenCe Engine for AMD GPUs
    \\
    \\Usage: zinc [options]
    \\  -m, --model <path>       Path to GGUF model file
    \\  -p, --port <port>        Server port (default: 8080)
    \\  -d, --device <id>        Vulkan device index (default: 0)
    \\  -c, --context <size>     Context length (default: 4096)
    \\  --parallel <n>           Max concurrent requests (default: 4)
    \\  --prompt <text>          Single prompt (CLI mode, no server)
    \\  --kv-quant <bits>        TurboQuant KV cache bits: 0/2/3/4 (default: 0=off)
    \\  -h, --help               Show this help
    \\
;

/// Parse the process argument vector into a validated runtime configuration.
/// @param args Raw argv slice, including argv[0].
/// @returns A populated Config value or a validation error describing the first invalid flag.
pub fn parseArgs(args: []const [:0]const u8) !Config {
    var config = Config{};
    var i: usize = 1; // skip argv[0]

    while (i < args.len) : (i += 1) {
        const arg = args[i];

        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            config.show_help = true;
            return config;
        } else if (std.mem.eql(u8, arg, "-m") or std.mem.eql(u8, arg, "--model")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.model_path = args[i];
        } else if (std.mem.eql(u8, arg, "-p") or std.mem.eql(u8, arg, "--port")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.port = std.fmt.parseInt(u16, args[i], 10) catch return error.InvalidPort;
        } else if (std.mem.eql(u8, arg, "-d") or std.mem.eql(u8, arg, "--device")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.device_index = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidDevice;
        } else if (std.mem.eql(u8, arg, "-c") or std.mem.eql(u8, arg, "--context")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.context_length = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidContext;
        } else if (std.mem.eql(u8, arg, "--parallel")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.max_parallel = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidParallel;
        } else if (std.mem.eql(u8, arg, "--prompt")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.prompt = args[i];
        } else if (std.mem.eql(u8, arg, "--kv-quant")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.kv_quant = std.fmt.parseInt(u8, args[i], 10) catch return error.InvalidKvQuant;
            if (config.kv_quant != 0 and config.kv_quant != 2 and config.kv_quant != 3 and config.kv_quant != 4) {
                return error.InvalidKvQuant;
            }
        } else {
            return error.UnknownArgument;
        }
    }

    return config;
}

/// Start the ZINC process in prompt mode or server mode.
/// @note Fatal startup errors are logged and terminate the process rather than bubbling to the caller.
pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const config = parseArgs(args) catch |err| {
        log.err("Argument error: {s}", .{@errorName(err)});
        std.fs.File.stderr().writeAll(banner) catch {};
        std.process.exit(1);
    };

    if (config.show_help) {
        std.fs.File.stdout().writeAll(banner) catch {};
        return;
    }

    // Initialize Vulkan
    var vk_instance = instance_mod.Instance.init(allocator, config.device_index) catch |err| {
        log.err("Vulkan init failed: {s}", .{@errorName(err)});
        std.process.exit(1);
    };
    defer vk_instance.deinit();

    // Detect GPU capabilities
    const gpu_config = gpu_detect.detect(&vk_instance);
    gpu_config.log_info();

    if (config.model_path == null) {
        log.warn("No model specified (-m). Use --help for usage.", .{});
        return;
    }

    const model_path = config.model_path.?;
    log.info("Model: {s}", .{model_path});

    // Load model
    var cmd_pool = try CommandPool.init(&vk_instance);
    defer cmd_pool.deinit();

    var model = loader_mod.load(model_path, &vk_instance, &cmd_pool, allocator) catch |err| {
        log.err("Failed to load model: {s}", .{@errorName(err)});
        std.process.exit(1);
    };
    defer model.deinit(&vk_instance);

    // Determine shader directory
    const shader_dir = "zig-out/share/zinc/shaders";

    // Initialize inference engine
    var engine = forward_mod.InferenceEngine.init(&model, &vk_instance, gpu_config, shader_dir, allocator) catch |err| {
        log.err("Failed to init inference engine: {s}", .{@errorName(err)});
        std.process.exit(1);
    };
    defer engine.deinit();

    if (config.prompt) |prompt| {
        log.info("Prompt: {s}", .{prompt});

        // Tokenize prompt — try external tokenizer, fall back to dummy tokens
        var prompt_tokens: []u32 = &.{};
        var owns_tokens = false;

        blk: {
            var tokenizer = tokenizer_mod.Tokenizer.init(allocator, model_path, .sentencepiece);
            defer tokenizer.deinit();

            prompt_tokens = tokenizer.encode(prompt) catch |err| {
                log.warn("External tokenizer failed ({s}), using dummy tokens", .{@errorName(err)});
                break :blk;
            };
            if (prompt_tokens.len > 0) {
                owns_tokens = true;
                break :blk;
            }
            log.warn("External tokenizer returned 0 tokens, using dummy tokens", .{});
        }

        // Fallback: generate simple dummy token IDs from prompt bytes
        if (prompt_tokens.len == 0) {
            const words = std.mem.count(u8, prompt, " ") + 1;
            const n_tokens: usize = @min(words * 2, 64); // ~2 tokens per word, cap at 64
            const dummy = try allocator.alloc(u32, n_tokens);
            for (dummy, 0..) |*t, i| {
                // Use prompt bytes to seed token IDs in valid vocab range
                const byte_idx = (i * prompt.len) / n_tokens;
                t.* = @as(u32, prompt[byte_idx]) + @as(u32, @intCast(i)) * 137 + 1000;
            }
            prompt_tokens = dummy;
            owns_tokens = true;
        }
        defer if (owns_tokens) allocator.free(prompt_tokens);

        log.info("Prompt tokens: {d}", .{prompt_tokens.len});

        // Generate
        const max_tokens: u32 = 256;
        const output_tokens = try forward_mod.generate(&engine, prompt_tokens, max_tokens, allocator);
        defer allocator.free(output_tokens);

        // Detokenize
        const output_text = tokenizer.decode(output_tokens) catch |err| {
            log.err("Detokenization failed: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer allocator.free(output_text);

        std.fs.File.stdout().writeAll(output_text) catch {};
        std.fs.File.stdout().writeAll("\n") catch {};
    } else {
        log.info("Server mode — port {d}, max {d} concurrent requests", .{ config.port, config.max_parallel });
        // TODO: start HTTP server (Phase 4)
    }
}

test "parseArgs: defaults" {
    const args = [_][:0]const u8{"zinc"};
    const config = try parseArgs(&args);
    try std.testing.expectEqual(@as(u16, 8080), config.port);
    try std.testing.expectEqual(@as(u32, 4096), config.context_length);
    try std.testing.expectEqual(@as(u8, 0), config.kv_quant);
    try std.testing.expect(config.model_path == null);
    try std.testing.expect(config.prompt == null);
}

test "parseArgs: full args" {
    const args = [_][:0]const u8{
        "zinc",          "-m",       "model.gguf", "-p",     "9090",
        "-d",            "1",        "-c",         "8192",   "--parallel",
        "8",             "--prompt",  "hello",      "--kv-quant", "3",
    };
    const config = try parseArgs(&args);
    try std.testing.expectEqualStrings("model.gguf", config.model_path.?);
    try std.testing.expectEqual(@as(u16, 9090), config.port);
    try std.testing.expectEqual(@as(u32, 1), config.device_index);
    try std.testing.expectEqual(@as(u32, 8192), config.context_length);
    try std.testing.expectEqual(@as(u32, 8), config.max_parallel);
    try std.testing.expectEqualStrings("hello", config.prompt.?);
    try std.testing.expectEqual(@as(u8, 3), config.kv_quant);
}

test "parseArgs: help flag" {
    const args = [_][:0]const u8{ "zinc", "--help" };
    const config = try parseArgs(&args);
    try std.testing.expect(config.show_help);
}

test "parseArgs: invalid kv-quant" {
    const args = [_][:0]const u8{ "zinc", "--kv-quant", "5" };
    try std.testing.expectError(error.InvalidKvQuant, parseArgs(&args));
}

test "parseArgs: unknown argument" {
    const args = [_][:0]const u8{ "zinc", "--foo" };
    try std.testing.expectError(error.UnknownArgument, parseArgs(&args));
}
