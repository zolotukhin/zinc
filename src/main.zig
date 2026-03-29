//! CLI entrypoints for configuring ZINC and starting local inference.
//! @section CLI & Entrypoints
//! This module wires together GPU initialization, model loading, tokenization,
//! and the single-process decode loop used for prompt-mode execution.
const std = @import("std");
const builtin = @import("builtin");
const gpu = @import("gpu/interface.zig");
const tokenizer_mod = @import("model/tokenizer.zig");
const config_mod = @import("model/config.zig");
// These modules import vulkan/ transitively — only available on Linux until T010-T014 refactor.
// On macOS they are stubbed out; the GPU abstraction refactor will make them platform-independent.
const loader_mod = if (gpu.is_vulkan) @import("model/loader.zig") else struct {};
const architecture_mod = if (gpu.is_vulkan) @import("model/architecture.zig") else struct {};
const forward_mod = if (gpu.is_vulkan) @import("compute/forward.zig") else struct {};

// Backend-specific imports (only one branch compiles per platform)
const instance_mod = if (gpu.is_vulkan) @import("vulkan/instance.zig") else gpu.backend;
const gpu_detect = if (gpu.is_vulkan) @import("vulkan/gpu_detect.zig") else struct {};
const CommandPool = if (gpu.is_vulkan) @import("vulkan/command.zig").CommandPool else struct {
    pub fn init(_: anytype) !@This() { return .{}; }
    pub fn deinit(_: *@This()) void {}
};

const log = std.log.scoped(.zinc);

// Force compilation and testing of all modules (platform-conditional).
// Modules that directly import vulkan/ are only compiled on Linux.
// On macOS, Metal-specific modules are compiled instead.
// T010-T014 will refactor compute/ and model/ to use gpu/interface.zig,
// after which all modules compile on both platforms.
comptime {
    if (gpu.is_vulkan) {
        _ = @import("vulkan/vk.zig");
        _ = @import("vulkan/buffer.zig");
        _ = @import("vulkan/pipeline.zig");
        _ = @import("vulkan/command.zig");
        // These modules import vulkan/ directly — Vulkan-only until T010-T014 refactor
        _ = @import("compute/dmmv.zig");
        _ = @import("compute/elementwise.zig");
        _ = @import("compute/attention.zig");
        _ = @import("compute/forward.zig");
        _ = @import("model/loader.zig");
        _ = @import("model/architecture.zig");
    }
    if (gpu.is_metal) {
        _ = @import("metal/device.zig");
        _ = @import("metal/buffer.zig");
        _ = @import("metal/pipeline.zig");
        _ = @import("metal/command.zig");
        _ = @import("model/loader_metal.zig");
        _ = @import("compute/forward_metal.zig");
    }
    // Platform-independent modules
    _ = @import("model/config.zig");
    _ = @import("model/gguf.zig");
    _ = @import("model/tokenizer.zig");
    _ = @import("compute/graph.zig");
    _ = @import("server/http.zig");
    _ = @import("server/routes.zig");
    _ = @import("server/session.zig");
    _ = @import("scheduler/request.zig");
    _ = @import("scheduler/scheduler.zig");
    _ = @import("scheduler/kv_cache.zig");
}

/// Runtime configuration built from CLI flags and default values.
pub const Config = struct {
    /// Path to GGUF model file.
    model_path: ?[]const u8 = null,
    /// HTTP server port.
    port: u16 = 8080,
    /// Vulkan device index.
    device_index: u32 = 0,
    /// Max sequence length.
    context_length: u32 = 4096,
    /// Max concurrent requests.
    max_parallel: u32 = 4,
    /// CLI prompt text.
    prompt: ?[]const u8 = null,
    /// Max tokens to generate.
    max_tokens: u32 = 256,
    kv_quant: u8 = 0, // 0=disabled, 2/3/4=TurboQuant bits
    /// Graph JSON report path.
    graph_report_path: ?[]const u8 = null,
    /// Graph DOT file path.
    graph_dot_path: ?[]const u8 = null,
    /// Enable per-dispatch GPU profiling.
    profile: bool = false,
    /// Print usage and exit.
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
    \\  --graph-report <path>    Write decode-graph JSON report from GGUF metadata
    \\  --graph-dot <path>       Write decode-graph Graphviz DOT from GGUF metadata
    \\  --profile                Enable per-dispatch GPU timing profiling
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
            if (config.context_length > 32768) return error.InvalidContext;
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
        } else if (std.mem.eql(u8, arg, "--graph-report")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.graph_report_path = args[i];
        } else if (std.mem.eql(u8, arg, "--graph-dot")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.graph_dot_path = args[i];
        } else if (std.mem.eql(u8, arg, "--max-tokens") or std.mem.eql(u8, arg, "-n")) {
            i += 1;
            if (i >= args.len) return error.MissingArgValue;
            config.max_tokens = std.fmt.parseInt(u32, args[i], 10) catch return error.InvalidMaxTokens;
        } else if (std.mem.eql(u8, arg, "--profile")) {
            config.profile = true;
        } else {
            return error.UnknownArgument;
        }
    }

    return config;
}

/// Build the static decode graph from GGUF metadata and write debugging artifacts.
/// Only available on Vulkan backend (loader.zig depends on Vulkan until T010-T014 refactor).
const exportDecodeGraphArtifacts = if (gpu.is_vulkan) exportDecodeGraphArtifactsImpl else (struct {
    fn f(_: []const u8, _: ?[]const u8, _: ?[]const u8, _: std.mem.Allocator) !void {
        log.warn("Graph export not yet available on Metal backend", .{});
    }
}).f;

fn exportDecodeGraphArtifactsImpl(
    model_path: []const u8,
    report_path: ?[]const u8,
    dot_path: ?[]const u8,
    allocator: std.mem.Allocator,
) !void {
    const model_config = try loader_mod.inspectConfig(model_path, allocator);
    var decode_graph = try architecture_mod.buildDecodeGraph(&model_config, allocator);
    defer decode_graph.deinit();

    if (report_path) |path| {
        const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer {
            var close_file = file;
            close_file.close();
        }

        var file_buffer: [4096]u8 = undefined;
        var file_writer = file.writer(&file_buffer);
        try decode_graph.writeJsonReport(&file_writer.interface, allocator);
        try file_writer.interface.flush();
        log.info("Wrote decode graph JSON report to {s}", .{path});
    }

    if (dot_path) |path| {
        const file = try std.fs.cwd().createFile(path, .{ .truncate = true });
        defer {
            var close_file = file;
            close_file.close();
        }

        var file_buffer: [4096]u8 = undefined;
        var file_writer = file.writer(&file_buffer);
        try decode_graph.writeDot(&file_writer.interface, allocator);
        try file_writer.interface.flush();
        log.info("Wrote decode graph DOT export to {s}", .{path});
    }

    var analysis = try decode_graph.analyze(allocator);
    defer analysis.deinit();

    log.info(
        "Decode graph {s}: {d} nodes | {d} edges | critical path {d} nodes ({d} edges) | max parallel width {d}",
        .{
            analysis.name,
            analysis.node_count,
            analysis.edge_count,
            analysis.critical_path_node_count,
            analysis.critical_path_edge_count,
            analysis.max_parallel_width,
        },
    );

    const top_n = @min(analysis.op_counts.len, 5);
    for (analysis.op_counts[0..top_n]) |entry| {
        log.info("  op {s}: {d}", .{ @tagName(entry.op), entry.count });
    }
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

    if (config.model_path == null) {
        log.warn("No model specified (-m). Use --help for usage.", .{});
        return;
    }

    const model_path = config.model_path.?;
    log.info("Model: {s}", .{model_path});

    if (config.graph_report_path != null or config.graph_dot_path != null) {
        exportDecodeGraphArtifacts(model_path, config.graph_report_path, config.graph_dot_path, allocator) catch |err| {
            log.err("Failed to export decode graph artifacts: {s}", .{@errorName(err)});
            std.process.exit(1);
        };

        if (config.prompt == null) return;
    }

    // Initialize GPU backend
    if (comptime gpu.is_metal) {
        const metal_device = @import("metal/device.zig");
        const metal_loader = @import("model/loader_metal.zig");
        const forward_metal = @import("compute/forward_metal.zig");

        var device = metal_device.MetalDevice.init(allocator, config.device_index) catch |err| {
            log.err("Metal init failed: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer device.deinit();

        log.info("ZINC Metal backend — Apple Silicon ({s})", .{@tagName(device.chip)});
        log.info("Memory: {d} GB | Max buffer: {d} GB", .{
            device.totalMemory() / (1024 * 1024 * 1024),
            device.maxBufferSize() / (1024 * 1024 * 1024),
        });

        // Load model (zero-copy mmap)
        var model = metal_loader.load(model_path, device.ctx, allocator) catch |err| {
            log.err("Failed to load model: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer model.deinit();

        if (config.prompt) |prompt| {
            log.info("Prompt: {s}", .{prompt});

            var tokenizer = tokenizer_mod.Tokenizer.initFromGGUF(&model.gguf_file, allocator) catch |err| {
                log.err("Failed to init tokenizer from GGUF: {s}", .{@errorName(err)});
                std.process.exit(1);
            };
            defer tokenizer.deinit();

            const raw_tokens = try tokenizer.encode(prompt);
            defer allocator.free(raw_tokens);

            // Prepend BOS token
            const prompt_tokens = try allocator.alloc(u32, raw_tokens.len + 1);
            prompt_tokens[0] = tokenizer.bosId();
            @memcpy(prompt_tokens[1..], raw_tokens);
            defer allocator.free(prompt_tokens);

            log.info("Prompt tokens ({d}): {any}", .{ prompt_tokens.len, prompt_tokens[0..@min(prompt_tokens.len, 15)] });

            // Initialize inference engine
            var engine = forward_metal.InferenceEngine.init(&model, &device, allocator) catch |err| {
                log.err("Failed to init Metal inference engine: {s}", .{@errorName(err)});
                std.process.exit(1);
            };
            defer engine.deinit();

            // Generate
            const max_tokens = config.max_tokens;
            const output_tokens = forward_metal.generate(&engine, prompt_tokens, max_tokens, tokenizer.eosId(), allocator) catch |err| {
                log.err("Failed to generate: {s}", .{@errorName(err)});
                std.process.exit(1);
            };
            defer allocator.free(output_tokens);

            if (output_tokens.len == 0) {
                log.warn("Metal decode loop not yet implemented. Engine initialized successfully with {d} pipelines.", .{9});
            } else {
                // Decode tokens to text
                var text_buf: std.ArrayList(u8) = .{};
                defer text_buf.deinit(allocator);
                for (output_tokens) |tid| {
                    if (tid < tokenizer.vocab.len) {
                        try text_buf.appendSlice(allocator, tokenizer.vocab[tid]);
                    } else {
                        try text_buf.appendSlice(allocator, "<?>");
                    }
                }
                log.info("Output ({d} tokens): {s}", .{ output_tokens.len, text_buf.items });
            }
        } else {
            log.info("Server mode — port {d} (Metal server not yet implemented)", .{config.port});
        }
        return;
    }

    // Vulkan backend (Linux)
    var vk_instance = instance_mod.Instance.init(allocator, config.device_index) catch |err| {
        log.err("Vulkan init failed: {s}", .{@errorName(err)});
        std.process.exit(1);
    };
    defer vk_instance.deinit();

    // Detect GPU capabilities
    const gpu_config = gpu_detect.detect(&vk_instance);
    gpu_config.log_info();

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

    // Enable profiling if requested
    if (config.profile) {
        engine.enableProfiling() catch |err| {
            log.warn("Failed to enable profiling: {s}", .{@errorName(err)});
        };
    }

    if (config.prompt) |prompt| {
        log.info("Prompt: {s}", .{prompt});

        // Initialize native BPE tokenizer from GGUF metadata
        var tokenizer = tokenizer_mod.Tokenizer.initFromGGUF(&model.gguf_file, allocator) catch |err| {
            log.err("Failed to init tokenizer from GGUF: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer tokenizer.deinit();

        // Tokenize prompt
        const raw_tokens = try tokenizer.encode(prompt);
        defer allocator.free(raw_tokens);

        // Prepend BOS token
        const prompt_tokens = try allocator.alloc(u32, raw_tokens.len + 1);
        prompt_tokens[0] = tokenizer.bosId();
        @memcpy(prompt_tokens[1..], raw_tokens);
        defer allocator.free(prompt_tokens);

        log.info("Prompt tokens ({d}): {any}", .{prompt_tokens.len, prompt_tokens[0..@min(prompt_tokens.len, 15)]});
        // Decode prompt tokens for verification
        {
            var pt_buf: std.ArrayList(u8) = .{};
            defer pt_buf.deinit(allocator);
            for (prompt_tokens) |tid| {
                if (tid < tokenizer.vocab.len) {
                    try pt_buf.appendSlice(allocator, tokenizer.vocab[tid]);
                } else {
                    try pt_buf.appendSlice(allocator, "<?>");
                }
            }
            log.info("Prompt decoded: \"{s}\"", .{pt_buf.items});
        }

        // Generate
        const max_tokens = config.max_tokens;
        const output_tokens = try forward_mod.generate(&engine, prompt_tokens, max_tokens, tokenizer.eosId(), allocator);
        defer allocator.free(output_tokens);

        // Output token IDs
        log.info("Output tokens ({d}): {any}", .{
            output_tokens.len,
            output_tokens[0..@min(output_tokens.len, 20)],
        });

        // Debug: dump first 5 generated tokens with their vocabulary text
        {
            const show_n = @min(output_tokens.len, 5);
            for (0..show_n) |ti| {
                const tok_str = if (output_tokens[ti] < tokenizer.vocab.len) tokenizer.vocab[output_tokens[ti]] else "?";
                log.info("  gen[{d}]: id={d} \"{s}\"", .{ ti, output_tokens[ti], tok_str });
            }
        }
        // Check specific token logits (Paris=11751, not=524)
        {
            const logits_ptr2: [*]const f32 = @ptrCast(@alignCast(engine.logits_staging.mapped.?));
            log.info("  logit[11751 'Paris']={d:.4} logit[524 'not']={d:.4} logit[264 'a']={d:.4}", .{
                logits_ptr2[11751], logits_ptr2[524], logits_ptr2[264],
            });
        }
        // Debug: dump top-5 logits from the last decode step
        {
            const vocab_size = model.config.vocab_size;
            const logits_ptr: [*]const f32 = @ptrCast(@alignCast(engine.logits_staging.mapped.?));
            const logits = logits_ptr[0..vocab_size];
            // Find top 5
            var top_ids: [5]u32 = .{ 0, 0, 0, 0, 0 };
            var top_vals: [5]f32 = .{ -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32), -std.math.inf(f32) };
            for (logits, 0..) |v, i| {
                if (v > top_vals[4]) {
                    top_vals[4] = v;
                    top_ids[4] = @intCast(i);
                    // Bubble sort to maintain top 5
                    var j: usize = 4;
                    while (j > 0 and top_vals[j] > top_vals[j - 1]) : (j -= 1) {
                        const tv = top_vals[j]; top_vals[j] = top_vals[j-1]; top_vals[j-1] = tv;
                        const ti = top_ids[j]; top_ids[j] = top_ids[j-1]; top_ids[j-1] = ti;
                    }
                }
            }
            for (0..5) |k| {
                const tok_str = if (top_ids[k] < tokenizer.vocab.len) tokenizer.vocab[top_ids[k]] else "?";
                log.info("  logit #{d}: id={d} val={d:.4} \"{s}\"", .{ k, top_ids[k], top_vals[k], tok_str });
            }
            // Also check: are logits mostly zero? NaN? Inf?
            var n_zero: u32 = 0;
            var n_nan: u32 = 0;
            var n_inf: u32 = 0;
            var sum_abs: f64 = 0;
            for (logits) |v| {
                if (v == 0) n_zero += 1;
                if (std.math.isNan(v)) n_nan += 1;
                if (std.math.isInf(v)) n_inf += 1;
                sum_abs += @abs(@as(f64, v));
            }
            log.info("  logit stats: zeros={d} NaN={d} Inf={d} mean_abs={d:.4}", .{
                n_zero, n_nan, n_inf, sum_abs / @as(f64, @floatFromInt(vocab_size)),
            });
        }

        // Decode tokens to text using the vocabulary
        {
            var text_buf: std.ArrayList(u8) = .{};
            defer text_buf.deinit(allocator);
            for (output_tokens) |tid| {
                if (tid < tokenizer.vocab.len) {
                    try text_buf.appendSlice(allocator, tokenizer.vocab[tid]);
                } else {
                    try text_buf.appendSlice(allocator, "<?>");
                }
            }
            log.info("Output text: {s}", .{text_buf.items});
        }
    } else {
        log.info("Server mode — port {d}, max {d} concurrent requests", .{ config.port, config.max_parallel });

        // Initialize tokenizer for chat template + prompt encoding
        var tokenizer = tokenizer_mod.Tokenizer.initFromGGUF(&model.gguf_file, allocator) catch |err| {
            log.err("Failed to init tokenizer from GGUF: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer tokenizer.deinit();

        // Start HTTP server
        const http_mod = @import("server/http.zig");
        var server = http_mod.Server.init(allocator, config.port) catch |err| {
            log.err("Failed to start HTTP server: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer server.deinit();
        log.info("Server listening on 0.0.0.0:{d}", .{config.port});
        log.info("Press Ctrl+C to stop", .{});

        // Graceful shutdown: set flag on SIGINT/SIGTERM
        const posix = std.posix;
        const Handler = struct {
            var shutdown_requested: bool = false;
            fn handler(_: c_int) callconv(.c) void {
                shutdown_requested = true;
            }
        };
        const sa = posix.Sigaction{
            .handler = .{ .handler = Handler.handler },
            .mask = std.mem.zeroes(posix.sigset_t),
            .flags = 0,
        };
        posix.sigaction(posix.SIG.INT, &sa, null);
        posix.sigaction(posix.SIG.TERM, &sa, null);

        // Server loop — accepts connections and handles requests sequentially.
        // Each request runs to completion before the next is accepted.
        // Concurrent streaming (US2) requires a poll-based event loop — deferred
        // to a future iteration since the inference engine is single-threaded
        // and GPU-bound (true concurrency needs batched multi-sequence decode).
        while (!Handler.shutdown_requested) {
            var conn = server.accept() catch |err| {
                if (Handler.shutdown_requested) break;
                log.warn("Accept failed: {s}", .{@errorName(err)});
                continue;
            };

            const routes = @import("server/routes.zig");
            routes.handleConnection(&conn, &engine, &tokenizer, &model, allocator) catch |err| {
                log.warn("Request failed: {s}", .{@errorName(err)});
            };
            conn.close();
        }
        log.info("Shutting down...", .{});
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
        "--graph-report", "graph.json", "--graph-dot", "graph.dot",
    };
    const config = try parseArgs(&args);
    try std.testing.expectEqualStrings("model.gguf", config.model_path.?);
    try std.testing.expectEqual(@as(u16, 9090), config.port);
    try std.testing.expectEqual(@as(u32, 1), config.device_index);
    try std.testing.expectEqual(@as(u32, 8192), config.context_length);
    try std.testing.expectEqual(@as(u32, 8), config.max_parallel);
    try std.testing.expectEqualStrings("hello", config.prompt.?);
    try std.testing.expectEqual(@as(u8, 3), config.kv_quant);
    try std.testing.expectEqualStrings("graph.json", config.graph_report_path.?);
    try std.testing.expectEqualStrings("graph.dot", config.graph_dot_path.?);
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

test "parseArgs: profile flag" {
    const args = [_][:0]const u8{ "zinc", "--profile", "--prompt", "test" };
    const config = try parseArgs(&args);
    try std.testing.expect(config.profile);
    try std.testing.expectEqualStrings("test", config.prompt.?);
}

test "parseArgs: profile defaults to false" {
    const args = [_][:0]const u8{ "zinc", "--prompt", "hi" };
    const config = try parseArgs(&args);
    try std.testing.expect(!config.profile);
}
