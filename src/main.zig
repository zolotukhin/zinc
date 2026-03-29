//! CLI entrypoints for configuring ZINC and starting local inference.
//! @section CLI & Entrypoints
//! This module wires together Vulkan initialization, model loading, tokenization,
//! and the single-process decode loop used for prompt-mode execution.
const std = @import("std");
const instance_mod = @import("vulkan/instance.zig");
const gpu_detect = @import("vulkan/gpu_detect.zig");
const loader_mod = @import("model/loader.zig");
const gguf_mod = @import("model/gguf.zig");
const architecture_mod = @import("model/architecture.zig");
const tokenizer_mod = @import("model/tokenizer.zig");
const forward_mod = @import("compute/forward.zig");
const graph_mod = @import("compute/graph.zig");
const diagnostics_mod = @import("diagnostics.zig");
const http_mod = @import("server/http.zig");
const routes_mod = @import("server/routes.zig");
const CommandPool = @import("vulkan/command.zig").CommandPool;
const Graph = graph_mod.Graph;

const log = std.log.scoped(.zinc);

pub var is_debug_mode: bool = false;

pub const std_options = std.Options{
    .log_level = .debug,
    .logFn = myLogFn,
};

pub fn myLogFn(
    comptime level: std.log.Level,
    comptime scope: @TypeOf(.enum_literal),
    comptime format: []const u8,
    args: anytype,
) void {
    if (level == .debug and !is_debug_mode) return;
    const scope_prefix = if (scope == .default) ": " else "(" ++ @tagName(scope) ++ "): ";
    const prefix = @tagName(level) ++ scope_prefix;
    std.debug.print(prefix ++ format ++ "\n", args);
}

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
    _ = @import("regression_tests.zig");
    _ = @import("server/http.zig");
    _ = @import("server/routes.zig");
    _ = @import("server/session.zig");
    _ = @import("scheduler/request.zig");
    _ = @import("scheduler/scheduler.zig");
    _ = @import("scheduler/kv_cache.zig");
    _ = @import("diagnostics.zig");
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
    /// Wrap CLI prompt in the model's chat template before tokenization.
    chat: bool = false,
    kv_quant: u8 = 0, // 0=disabled, 2/3/4=TurboQuant bits
    /// Graph JSON report path.
    graph_report_path: ?[]const u8 = null,
    /// Graph DOT file path.
    graph_dot_path: ?[]const u8 = null,
    /// Enable per-dispatch GPU profiling.
    profile: bool = false,
    /// Enable verbose debug logging.
    debug: bool = false,
    /// Print usage and exit.
    show_help: bool = false,
    /// Run diagnostics and exit.
    check: bool = false,
};

const ConnectionWorker = struct {
    conn: http_mod.Connection,
    engine: *forward_mod.InferenceEngine,
    tokenizer: *tokenizer_mod.Tokenizer,
    model: *loader_mod.Model,
    server_state: *routes_mod.ServerState,

    fn run(self: *ConnectionWorker) void {
        defer std.heap.page_allocator.destroy(self);
        defer self.conn.close();

        routes_mod.handleConnection(
            &self.conn,
            self.engine,
            self.tokenizer,
            self.model,
            self.server_state,
            std.heap.page_allocator,
        ) catch |err| {
            log.warn("Request failed: {s}", .{@errorName(err)});
        };
    }
};

const PreparedPrompt = struct {
    text: []const u8,
    owned_buf: ?[]u8 = null,

    fn deinit(self: *PreparedPrompt, allocator: std.mem.Allocator) void {
        if (self.owned_buf) |buf| allocator.free(buf);
        self.* = undefined;
    }
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
    \\  --chat                   Apply the model chat template to --prompt
    \\  --kv-quant <bits>        TurboQuant KV cache bits: 0/2/3/4 (default: 0=off)
    \\  --graph-report <path>    Write decode-graph analysis JSON report
    \\  --graph-dot <path>       Write decode-graph Graphviz DOT from GGUF metadata
    \\  --profile                Enable per-dispatch GPU timing profiling
    \\  --debug                  Enable verbose debug logging
    \\  --check                  Run system diagnostics and verify dependencies
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
        } else if (std.mem.eql(u8, arg, "--chat")) {
            config.chat = true;
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
        } else if (std.mem.eql(u8, arg, "--profile")) {
            config.profile = true;
        } else if (std.mem.eql(u8, arg, "--debug")) {
            config.debug = true;
        } else if (std.mem.eql(u8, arg, "--check")) {
            config.check = true;
        } else {
            return error.UnknownArgument;
        }
    }

    return config;
}

fn prepareCliPrompt(tokenizer: *const tokenizer_mod.Tokenizer, prompt: []const u8, chat: bool, allocator: std.mem.Allocator) !PreparedPrompt {
    if (!chat) {
        return .{ .text = prompt };
    }

    const roles = [_][]const u8{"user"};
    const contents = [_][]const u8{prompt};
    const chat_capacity = prompt.len + 256;
    const chat_buf = try allocator.alloc(u8, chat_capacity);
    errdefer allocator.free(chat_buf);

    const formatted = try tokenizer.applyChatTemplate(&roles, &contents, chat_buf);
    return .{
        .text = formatted,
        .owned_buf = chat_buf,
    };
}

fn trimCliOutputText(text: []const u8, chat: bool) []const u8 {
    if (!chat) return text;
    if (std.mem.indexOf(u8, text, "<|im_end|>")) |stop_pos| {
        return text[0..stop_pos];
    }
    return text;
}

/// Build the static decode graph from GGUF metadata and write debugging artifacts.
/// @param model_path Path to the GGUF file to inspect.
/// @param report_path Optional JSON destination for the structural graph report.
/// @param dot_path Optional DOT destination for Graphviz rendering.
/// @param allocator Allocator used for GGUF parsing and graph analysis.
fn exportDecodeGraphArtifacts(
    model_path: []const u8,
    report_path: ?[]const u8,
    dot_path: ?[]const u8,
    allocator: std.mem.Allocator,
) !void {
    const model_config = try loader_mod.inspectConfig(model_path, allocator);
    const file = try std.fs.cwd().openFile(model_path, .{});
    defer {
        var close_file = file;
        close_file.close();
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

    var gguf_file = try gguf_mod.parse(mmap_data, allocator);
    defer gguf_file.deinit();

    var decode_graph = try architecture_mod.buildDecodeGraphDetailed(&model_config, allocator, &gguf_file);
    defer decode_graph.deinit();
    try writeDecodeGraphArtifacts(&decode_graph, report_path, dot_path, allocator);
}

fn writeDecodeGraphArtifacts(
    decode_graph: *const Graph,
    report_path: ?[]const u8,
    dot_path: ?[]const u8,
    allocator: std.mem.Allocator,
) !void {
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

    const hotspot_n = @min(analysis.hotspots.len, 5);
    for (analysis.hotspots[0..hotspot_n]) |entry| {
        const bw_time = if (entry.estimated_bandwidth_time_us) |us|
            us
        else
            0.0;
        log.info("  hot {s}: {d:.1}% | {d:.2} MB | {d:.2} us bw-floor | {s}", .{
            entry.name,
            entry.estimated_share_pct,
            @as(f64, @floatFromInt(entry.total_bytes)) / 1_000_000.0,
            bw_time,
            @tagName(entry.bottleneck),
        });
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

    if (config.check) {
        diagnostics_mod.run(allocator) catch |err| {
            log.err("Diagnostics completed with error: {s}", .{@errorName(err)});
        };
        return;
    }

    is_debug_mode = config.debug or std.posix.getenv("ZINC_DEBUG") != null;

    if (config.model_path == null) {
        log.warn("No model specified (-m). Use --help for usage.", .{});
        return;
    }

    const model_path = config.model_path.?;
    log.info("Model: {s}", .{model_path});

    const wants_graph_artifacts = config.graph_report_path != null or config.graph_dot_path != null;
    if (wants_graph_artifacts and config.prompt == null) {
        exportDecodeGraphArtifacts(model_path, config.graph_report_path, config.graph_dot_path, allocator) catch |err| {
            log.err("Failed to export decode graph artifacts: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
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

    if (wants_graph_artifacts) {
        writeDecodeGraphArtifacts(&engine.decode_graph, config.graph_report_path, config.graph_dot_path, allocator) catch |err| {
            log.err("Failed to export decode graph artifacts: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
    }

    // Enable profiling if requested
    if (config.profile) {
        engine.enableProfiling() catch |err| {
            log.warn("Failed to enable profiling: {s}", .{@errorName(err)});
        };
    }

    if (config.prompt) |prompt| {
        log.debug("Prompt: {s}", .{prompt});

        // Initialize native BPE tokenizer from GGUF metadata
        var tokenizer = tokenizer_mod.Tokenizer.initFromGGUF(&model.gguf_file, allocator) catch |err| {
            log.err("Failed to init tokenizer from GGUF: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer tokenizer.deinit();

        var prepared_prompt = try prepareCliPrompt(&tokenizer, prompt, config.chat, allocator);
        defer prepared_prompt.deinit(allocator);
        if (config.chat) {
            log.debug("Prompt mode: chat template ({d} chars)", .{prepared_prompt.text.len});
        }

        // Tokenize prompt
        const raw_tokens = try tokenizer.encode(prepared_prompt.text);
        defer allocator.free(raw_tokens);

        // Prepend BOS token
        const prepend_bos = tokenizer.shouldPrependBos();
        const bos_extra: usize = if (prepend_bos) 1 else 0;
        const prompt_tokens = try allocator.alloc(u32, raw_tokens.len + bos_extra);
        if (prepend_bos) {
            prompt_tokens[0] = tokenizer.bosId();
            @memcpy(prompt_tokens[1..], raw_tokens);
        } else {
            @memcpy(prompt_tokens, raw_tokens);
        }
        defer allocator.free(prompt_tokens);

        log.debug("Prompt tokens ({d}): {any}", .{ prompt_tokens.len, prompt_tokens[0..@min(prompt_tokens.len, 15)] });
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
            log.debug("Prompt decoded: \"{s}\"", .{pt_buf.items});
        }

        // Generate
        const max_tokens: u32 = 256;
        const output_tokens = try forward_mod.generate(&engine, prompt_tokens, max_tokens, tokenizer.eosId(), allocator);
        defer allocator.free(output_tokens);

        // Output token IDs
        log.debug("Output tokens ({d}): {any}", .{
            output_tokens.len,
            output_tokens[0..@min(output_tokens.len, 20)],
        });

        // Debug: dump first 5 generated tokens with their vocabulary text
        {
            const show_n = @min(output_tokens.len, 5);
            for (0..show_n) |ti| {
                const tok_str = if (output_tokens[ti] < tokenizer.vocab.len) tokenizer.vocab[output_tokens[ti]] else "?";
                log.debug("  gen[{d}]: id={d} \"{s}\"", .{ ti, output_tokens[ti], tok_str });
            }
        }
        // Check specific token logits (Paris=11751, not=524)
        {
            const logits_ptr2: [*]const f32 = @ptrCast(@alignCast(engine.logits_staging.mapped.?));
            log.debug("  logit[11751 'Paris']={d:.4} logit[524 'not']={d:.4} logit[264 'a']={d:.4}", .{
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
                        const tv = top_vals[j];
                        top_vals[j] = top_vals[j - 1];
                        top_vals[j - 1] = tv;
                        const ti = top_ids[j];
                        top_ids[j] = top_ids[j - 1];
                        top_ids[j - 1] = ti;
                    }
                }
            }
            for (0..5) |k| {
                const tok_str = if (top_ids[k] < tokenizer.vocab.len) tokenizer.vocab[top_ids[k]] else "?";
                log.debug("  logit #{d}: id={d} val={d:.4} \"{s}\"", .{ k, top_ids[k], top_vals[k], tok_str });
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
            log.debug("  logit stats: zeros={d} NaN={d} Inf={d} mean_abs={d:.4}", .{
                n_zero, n_nan, n_inf, sum_abs / @as(f64, @floatFromInt(vocab_size)),
            });
        }

        // Decode tokens to text using the vocabulary
        {
            var text_buf: std.ArrayList(u8) = .{};
            defer text_buf.deinit(allocator);
            for (output_tokens) |tid| {
                var dec_buf: [256]u8 = undefined;
                const decoded = tokenizer.decodeToken(tid, &dec_buf);
                if (decoded.len > 0) {
                    try text_buf.appendSlice(allocator, decoded);
                } else {
                    try text_buf.appendSlice(allocator, "<?>");
                }
            }
            const output_text = trimCliOutputText(text_buf.items, config.chat);
            log.info("Output text: {s}", .{output_text});
        }
    } else {
        log.info("Server mode — port {d}, max {d} concurrent requests", .{ config.port, config.max_parallel });

        // Initialize tokenizer for chat template + prompt encoding
        var tokenizer = tokenizer_mod.Tokenizer.initFromGGUF(&model.gguf_file, allocator) catch |err| {
            log.err("Failed to init tokenizer from GGUF: {s}", .{@errorName(err)});
            std.process.exit(1);
        };
        defer tokenizer.deinit();

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

        var server_state = routes_mod.ServerState.init(std.time.timestamp());

        // Server loop — accepts connections concurrently so operational endpoints
        // like /health remain responsive, while generation itself is serialized
        // behind a shared lock inside the route handlers.
        while (!Handler.shutdown_requested) {
            var conn = server.accept() catch |err| {
                if (Handler.shutdown_requested) break;
                log.warn("Accept failed: {s}", .{@errorName(err)});
                continue;
            };

            const worker = std.heap.page_allocator.create(ConnectionWorker) catch |err| {
                log.warn("Failed to allocate connection worker: {s}", .{@errorName(err)});
                conn.close();
                continue;
            };
            worker.* = .{
                .conn = conn,
                .engine = &engine,
                .tokenizer = &tokenizer,
                .model = &model,
                .server_state = &server_state,
            };

            const thread = std.Thread.spawn(.{}, ConnectionWorker.run, .{worker}) catch |err| {
                log.warn("Failed to spawn connection worker: {s}", .{@errorName(err)});
                std.heap.page_allocator.destroy(worker);
                conn.close();
                continue;
            };
            thread.detach();
        }

        while (server_state.active_requests.load(.monotonic) != 0 or
            server_state.queued_requests.load(.monotonic) != 0)
        {
            std.Thread.sleep(50 * std.time.ns_per_ms);
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
    try std.testing.expect(!config.chat);
}

test "parseArgs: full args" {
    const args = [_][:0]const u8{
        "zinc", "-m",             "model.gguf", "-p",          "9090",
        "-d",   "1",              "-c",         "8192",        "--parallel",
        "8",    "--prompt",       "hello",      "--chat",      "--kv-quant",
        "3",    "--graph-report", "graph.json", "--graph-dot", "graph.dot",
    };
    const config = try parseArgs(&args);
    try std.testing.expectEqualStrings("model.gguf", config.model_path.?);
    try std.testing.expectEqual(@as(u16, 9090), config.port);
    try std.testing.expectEqual(@as(u32, 1), config.device_index);
    try std.testing.expectEqual(@as(u32, 8192), config.context_length);
    try std.testing.expectEqual(@as(u32, 8), config.max_parallel);
    try std.testing.expectEqualStrings("hello", config.prompt.?);
    try std.testing.expect(config.chat);
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

test "parseArgs: chat flag" {
    const args = [_][:0]const u8{ "zinc", "--prompt", "hi", "--chat" };
    const config = try parseArgs(&args);
    try std.testing.expect(config.chat);
    try std.testing.expectEqualStrings("hi", config.prompt.?);
}

fn makeTestTokenizer(chat_template: ?[]const u8) tokenizer_mod.Tokenizer {
    return .{
        .vocab = &.{},
        .token_to_id = std.StringHashMap(u32).init(std.testing.allocator),
        .merges = &.{},
        .scores = null,
        .bos_id = 1,
        .eos_id = 2,
        .prepend_bos = true,
        .chat_template = chat_template,
        .allocator = std.testing.allocator,
    };
}

test "prepareCliPrompt leaves non-chat prompts unowned" {
    var tok = makeTestTokenizer(null);
    defer tok.token_to_id.deinit();

    var prepared = try prepareCliPrompt(&tok, "hello", false, std.testing.allocator);
    defer prepared.deinit(std.testing.allocator);

    try std.testing.expectEqualStrings("hello", prepared.text);
    try std.testing.expect(prepared.owned_buf == null);
}

test "prepareCliPrompt returns full owned chat buffer" {
    var tok = makeTestTokenizer(null);
    defer tok.token_to_id.deinit();

    var prepared = try prepareCliPrompt(&tok, "Hello", true, std.testing.allocator);
    defer prepared.deinit(std.testing.allocator);

    try std.testing.expect(prepared.owned_buf != null);
    try std.testing.expect(prepared.text.ptr == prepared.owned_buf.?.ptr);
    try std.testing.expect(prepared.owned_buf.?.len >= prepared.text.len);
    try std.testing.expectEqualStrings("<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n", prepared.text);
}

test "trimCliOutputText strips chat terminator only in chat mode" {
    try std.testing.expectEqualStrings("Paris", trimCliOutputText("Paris<|im_end|>", true));
    try std.testing.expectEqualStrings("Paris<|im_end|>", trimCliOutputText("Paris<|im_end|>", false));
    try std.testing.expectEqualStrings("Paris", trimCliOutputText("Paris", true));
}
