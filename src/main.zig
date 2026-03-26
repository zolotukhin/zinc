const std = @import("std");
const instance_mod = @import("vulkan/instance.zig");
const gpu_detect = @import("vulkan/gpu_detect.zig");

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
}

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

    if (config.model_path) |path| {
        log.info("Model: {s}", .{path});
    } else {
        log.warn("No model specified (-m). Use --help for usage.", .{});
    }

    if (config.prompt) |prompt| {
        log.info("Prompt: {s}", .{prompt});
        // TODO: single-request inference (Phase 3)
    } else if (config.model_path != null) {
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
